"""1 단계 (열화상용 상대 깊이 모델) 만 재기 — 모양 실험 (run.sh shape, README '모양 실험'). 학습 없음.

사진마다 스케일 · 기준점을 맞춘 뒤 (역깊이에서 a · x + b 를 최소제곱으로, rel_check 와 같은 식) 잰다. 2 단계 (거리) 는 거치지 않는다.
  AbsRel · δ1   맞춘 깊이와 기준의 차 (모양의 오차)
  경계 F1      common.edge_f1 — 이웃 화소의 깊이 비로 정한 경계가 기준의 경계와 겹치는가 (뭉개진 예측은 낮다)
세트:
  정답이 있는 세트   MS2 val · test 셋 · 처음 보는 네 곳. 화소를 고르는 규칙은 evaluate.py 와 같다 (0.001–80 m, ViViD 는 테두리 규칙).
  빼 둔 쌍 세트 (--hold)   그 세트의 열화상을 넣고, 짝 RGB 에 선생이 낸 깊이 (라벨) 와 견준다 — 학습에 쓰지 않은 카메라에서 RGB 선생과 모양이 얼마나 맞는가.
                           방식은 이 값으로 고른다 (처음 보는 대상의 값은 고르는 데 쓰지 않는다).
--origin (학습을 시작한 가중치) 을 주면 표현이 어떻게 달라졌는지도 잰다 — 빼 둔 쌍 세트에서, 인코더 마지막 층의 패치 토큰으로 (사진 평균):
  drift = 같은 열화상을 원래 모델에 넣은 토큰과의 코사인 (1 = 그대로) · cross = 짝 RGB 를 원래 모델에 넣은 토큰과의 코사인 · gram = 그 둘의 패치 사이 유사도 행렬의 차 (feat_gram).
  cross · gram 은 RGB 가 밝은 패치만, 그런 패치가 둘 이상인 사진만 (밤 사진은 빠진다). 원래 모델 자신의 값 (cross_origin · gram_origin) 도 함께 적는다.

사용: python src/rel_eval.py --ckpt <rel.pt> --out <폴더> [--hold freiburg (하나만)] [--origin <.pth>] [--encoder vitl] [--bs 16] [--workers 4] [--amp 1] [--limit 0]
      python src/rel_eval.py --summary <runs 폴더> [--out <폴더>]      # 변형마다 한 줄 ([표8] 모양 · [표9] 경계 · [표10] 표현과 기울기)
출력: <out>/rel_metrics.json, 화면에 세트마다 한 줄.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import model as M  # noqa: E402
from common import MIN_VALID, PL_DARK, depth_metrics, edge_f1, eval_mask, feat_gram, write_json  # noqa: E402
from rel_check import measure  # noqa: E402
from teacher_check import aligned  # noqa: E402

GT_SETS = ("val", "test_day", "test_night", "test_rainy", "vivid", "fire", "vivid_in", "sthereo")
TARGETS = ("vivid", "fire", "vivid_in", "sthereo")
ORDER = ("nopair", "pair", "edge", "freeze", "thr_cos", "thr_gram", "rgb_cos", "rgb_gram")       # run.sh shape 의 변형 (표에 찍는 순서)


def variant_of(run):
    """'shape_rgb_gram_s0' → ('rgb_gram', '0')."""
    k = run.rfind("_s")
    return run[len("shape_"):k], run[k + 2:]


def score(disp, ref, mask):
    """역깊이 꼴 예측 → (AbsRel, δ1, 경계 F1) 또는 None."""
    al = aligned(disp, ref, mask)
    r = depth_metrics(al[mask], ref[mask].double())
    return (r["abs_rel"], r["d1"], edge_f1(al, ref, mask)) if np.isfinite(r["abs_rel"]) else None


def pool(rows):
    """[(AbsRel, δ1, F1 또는 None)] → 세트의 평균 (F1 은 경계가 있는 사진만)."""
    f1 = [r[2] for r in rows if r[2] is not None]
    return {"abs_rel": float(np.mean([r[0] for r in rows])), "d1": float(np.mean([r[1] for r in rows])), "n": len(rows),
            "edge_f1": float(np.mean(f1)) if f1 else None, "edge_n": len(f1)}


@torch.no_grad()
def gt_set(model, name, dev, bs, workers, limit, amp):
    """정답이 있는 세트 → 사진별 [(AbsRel, δ1, F1)] — 1 단계의 val 과 같은 길 (rel_check.measure, ViViD 는 테두리 규칙)."""
    return [r[1:] for r in measure(model, D.Thermal(name, limit), dev, bs, workers, amp=amp, edges=True)]


@torch.no_grad()
def pair_set(model, name, dev, bs, workers, limit, amp, origin=None):
    """빼 둔 쌍 세트 → (사진별 [(AbsRel, δ1, F1)] — 선생 라벨과 견준 것, 표현의 변화 {이름: 평균} 또는 None)."""
    ds, rows, rep = D.PairLabels([name], limit, rgb=origin is not None), [], {}
    mean, std = D.MEAN.flatten().tolist(), D.STD.flatten().tolist()
    for bt in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers):
        x, lab = bt[0].to(dev), bt[1].to(dev)
        if origin is None:
            disp = M.teacher_disp(model, x, amp=amp)
        else:
            disp, tok = M.rel_forward_feats(model, x, lab.shape[-2:], amp)
            rgb = bt[-1].to(dev)
            t0, t1, pm = M.origin_tokens(origin, x, amp), M.origin_tokens(origin, rgb, amp), M.patch_bright(rgb, mean, std, PL_DARK)
            c0, c1, c01 = (F.cosine_similarity(u.float(), v.float(), dim=-1) for u, v in ((tok, t0), (tok, t1), (t0, t1)))
            rep.setdefault("drift", []).extend(c0.mean(1).tolist())                    # 같은 열화상끼리 — 모든 패치
            for j in range(len(x)):
                if int(pm[j].sum()) >= 2:                                                # 짝 RGB 와 견주는 것은 RGB 가 밝은 패치가 있는 사진만 (밤 사진이 0 으로 섞이지 않게)
                    rep.setdefault("cross", []).append(float(c1[j][pm[j]].mean()))
                    rep.setdefault("cross_origin", []).append(float(c01[j][pm[j]].mean()))
                    rep.setdefault("gram", []).append(float(feat_gram(tok[j:j + 1], t1[j:j + 1], pm[j:j + 1])))
                    rep.setdefault("gram_origin", []).append(float(feat_gram(t0[j:j + 1], t1[j:j + 1], pm[j:j + 1])))
        disp = disp.float()
        for j in range(len(x)):
            m = eval_mask(lab[j])
            r = score(disp[j], lab[j], m) if int(m.sum()) >= MIN_VALID else None
            if r:
                rows.append(r)
    return rows, ({k: float(np.mean(v)) for k, v in rep.items()} if rep else None)


def summary(runs, out):
    """runs/shape_<변형>_s<시드>/eval/rel_metrics.json (+ rel/done.json) 을 모아 변형마다 한 줄씩 세 표."""
    items = []
    for d in sorted(os.listdir(runs)):
        p = os.path.join(runs, d, "eval", "rel_metrics.json")
        if d.startswith("shape_") and os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))
            done = os.path.join(runs, d, "rel", "done.json")
            r["train"] = json.load(open(done, encoding="utf-8")) if os.path.exists(done) else {}
            items.append((d, r))
    items.sort(key=lambda it: (ORDER.index(variant_of(it[0])[0]) if variant_of(it[0])[0] in ORDER else len(ORDER), it[0]))
    f = lambda v, k=4: "—" if v is None else f"{v:.{k}f}"  # noqa: E731
    table = {}
    for d, r in items:
        s, h, tr = r["sets"], r.get("hold") or {}, r["train"]
        ms2 = [s[k] for k in ("test_day", "test_night", "test_rainy") if k in s]
        hk = next(iter(h), None)
        le = tr.get("last_epoch") or {}
        table[d] = {"hold": hk, "hold_abs_rel": h[hk]["abs_rel"] if hk else None, "hold_edge_f1": h[hk]["edge_f1"] if hk else None,
                    "val": s.get("val", {}).get("abs_rel"), "ms2_test": float(np.mean([v["abs_rel"] for v in ms2])) if len(ms2) == 3 else None,
                    "ms2_test_edge_f1": float(np.mean([v["edge_f1"] for v in ms2 if v["edge_f1"] is not None])) if len(ms2) == 3 else None,
                    **{k: s.get(k, {}).get("abs_rel") for k in TARGETS}, **{k + "_edge_f1": s.get(k, {}).get("edge_f1") for k in TARGETS},
                    "repr": h[hk].get("repr") if hk else None, "minutes": tr.get("minutes"), "val_train": tr.get("val_aligned_abs_rel"),
                    "fa_sim": le.get("fa_sim"), "fa_on": le.get("fa_on"), "gcos_ms2": le.get("gcos_ms2"), "gcos_pair": le.get("gcos_pair"), "gratio": le.get("gratio")}
    print("[표8] 모양 (1 단계만) — 사진마다 스케일 · 기준점을 맞춘 뒤 AbsRel ↓ | 빼 둔 쌍 (선생과 — 방식을 고르는 값) | MS2 val | MS2 test 평균 | ViViD 밤 | FIReStereo | ViViD 실내 | STheReO | 학습 (분)")
    for d, t in table.items():
        print(f"[표8] {d} | {t['hold'] or ''} {f(t['hold_abs_rel'])} | {f(t['val'])} | {f(t['ms2_test'])} | " + " | ".join(f(t[k]) for k in TARGETS) + f" | {t['minutes']}")
    print("[표9] 경계 F1 ↑ (이웃 화소의 깊이 비로 정한 경계가 기준과 겹치는 정도, common.edge_f1) | 빼 둔 쌍 (선생과) | MS2 test 평균 | ViViD 밤 | FIReStereo | ViViD 실내 | STheReO")
    for d, t in table.items():
        print(f"[표9] {d} | {f(t['hold_edge_f1'], 3)} | {f(t['ms2_test_edge_f1'], 3)} | " + " | ".join(f(t[k + '_edge_f1'], 3) for k in TARGETS))
    print("[표10] 표현과 기울기 — 빼 둔 쌍에서 마지막 층 토큰: 원래 모델과 (같은 열화상) · 짝 RGB 의 원래 모델 토큰과 (원래 모델끼리의 값) · 관계 차 (원래 모델끼리의 값)"
          " | 학습 마지막 epoch: 목표와의 유사도 · 손실에 든 토큰 · 기울기 코사인 (정렬 · MS2 / 정렬 · 쌍) · 크기 비")
    for d, t in table.items():
        rp = t["repr"]
        rtxt = f"{rp['drift']:.3f} · {rp['cross']:.3f} ({rp['cross_origin']:.3f}) · {rp.get('gram', float('nan')):.4f} ({rp.get('gram_origin', float('nan')):.4f})" if rp else "—"
        gtxt = (f"{f(t['fa_sim'], 3)} · {f(t['fa_on'] * 100 if t['fa_on'] is not None else None, 0)} % · {f(t['gcos_ms2'], 3)} / {f(t['gcos_pair'], 3)} · {f(t['gratio'], 2)}"
                if t["fa_sim"] is not None else "—")
        print(f"[표10] {d} | {rtxt} | {gtxt}")
    if out:
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "shape.json"), table)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--hold", default="")
    ap.add_argument("--origin", default="", help="학습을 시작한 가중치 (공식 상대 깊이 체크포인트). 주면 표현의 변화도 잰다")
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1, help="bf16 혼합 정밀도 (GPU 에서만) — 모든 변형에 같게")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary, a.out)
    if not a.ckpt or not a.out:
        raise SystemExit("!!! [eval-rel] --ckpt 와 --out 이 필요")
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(os.path.dirname(os.path.normpath(a.out)))
    dev = torch.device(a.device)
    amp = bool(a.amp) and dev.type == "cuda"
    model = M.build_rel(a.encoder, "")
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True), strict=True)
    model = model.to(dev).eval().requires_grad_(False)
    origin = M.origin_encoder(M.build_teacher(a.encoder, a.origin)) if a.origin else None
    if origin is not None:
        origin = (origin[0].to(dev), origin[1])
    holds = [v for v in a.hold.split(",") if v]
    if len(holds) > 1:
        raise SystemExit("!!! [eval-rel] --hold 는 하나만 (방식을 고르는 세트)")
    res = {"sets": {}, "hold": {}, "ckpt": os.path.basename(a.ckpt), "origin": bool(a.origin), "amp": amp}
    for s in GT_SETS:
        rows = gt_set(model, s, dev, a.bs, a.workers, a.limit, amp)
        if rows:
            res["sets"][s] = pool(rows)
            v = res["sets"][s]
            print(f"[eval-rel {name}] {s} {len(rows):,} 장 · 맞춘 뒤 AbsRel {v['abs_rel']:.4f} δ1 {v['d1']:.4f} · 경계 F1 "
                  + (f"{v['edge_f1']:.3f} ({v['edge_n']:,} 장)" if v["edge_f1"] is not None else "—"), flush=True)
    for h in holds:
        rows, rep = pair_set(model, h, dev, a.bs, a.workers, a.limit, amp, origin)
        if rows:
            res["hold"][h] = pool(rows) | {"repr": rep}
            v = res["hold"][h]
            print(f"[eval-rel {name}] 빼 둔 쌍 {h} {len(rows):,} 장 · 선생과 맞춘 뒤 AbsRel {v['abs_rel']:.4f} δ1 {v['d1']:.4f} · 경계 F1 "
                  + (f"{v['edge_f1']:.3f}" if v["edge_f1"] is not None else "—")
                  + (f" · 원래 모델과의 유사도 {rep['drift']:.3f} · 짝 RGB 와 {rep['cross']:.3f} (원래 {rep['cross_origin']:.3f})" if rep else ""), flush=True)
    write_json(os.path.join(a.out, "rel_metrics.json"), res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
