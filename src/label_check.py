"""선생 라벨 점검 (run.sh labelcheck) — 학습 없음. 읽는 법은 README '라벨 점검' (결과를 보기 전에 고정).

쌍의 선생 라벨은 UniDepthV2 에 카메라 정보를 주지 않고 뽑았다 (src/labels.py). 선생이 쌍의 카메라를 잘못 짐작해 절대 깊이가 한쪽으로 쏠렸는지 본다.
쌍 세트마다 고르게 뽑은 RGB 에 선생을 두 번 돌린다: ① 카메라 없이 (지금 라벨을 뽑는 길) ② 실제 카메라 값 (CAMS) 을 주고.
  사진마다: 선생이 짐작한 초점거리 (①의 출력) · 쓸 화소에서 깊이 중앙값 둘 · 그 비 (① / ②) · 스케일을 맞춘 뒤 남는 차 (모양의 차).
  초점을 모르는 세트 (LLVIP) 는 ①만 돌려 짐작한 초점과 깊이만 적는다.
MS2 의 RGB (LiDAR 정답과 카메라 값이 있다) 에서는 ① · ② 를 정답과 견준다 — 카메라 값을 주는 길이 절대 스케일을 바로잡는지의 확인.
쓸 화소는 라벨과 같은 규칙 (밝기 ≥ PL_DARK · 깊이 PL_MIN–PL_MAX m), MS2 는 정답 0.001–80 m.

사용: python src/label_check.py --data <푼 폴더> --out <폴더> [--encoder vitl|vits] [--limit 1000] [--bs 8]
출력: <out>/per_image.csv · summary.json, 화면에 표와 판정.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
from common import CAMS, MAX_DEPTH, MIN_DEPTH, PL_DARK, PL_MAX, PL_MIN, write_json  # noqa: E402
from teacher_check import build_unidepth, frames  # noqa: E402

INFLATE = 1.15            # 사진별 깊이 비 (카메라 없이 / 카메라 값을 주고) 의 중앙값이 이 이상이면 '멀게 부풂', 1 / 이 값 이하면 '가깝게 쏠림'
MS2_TOL = 0.01            # MS2: 카메라 값을 줬을 때 스케일 오차 중앙값이 이만큼 넘게 줄면 '바로잡음', 늘면 '나빠짐'

def kmat(c):
    return torch.tensor([[c[0], 0.0, c[2]], [0.0, c[1], c[3]], [0.0, 0.0, 1.0]], dtype=torch.float32)


def pick(items, limit):
    """고르게 limit 개 (0 이면 전부)."""
    if limit and limit < len(items):
        return [items[round(i * (len(items) - 1) / max(limit - 1, 1))] for i in range(limit)]
    return items


class PairRgb(Dataset):
    def __init__(self, name, root, limit):
        self.items = pick([(k, rp) for k, _, rp in D.pair_frames(name, root)], limit)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(Image.open(self.items[i][1]).convert("RGB")).transpose(2, 0, 1).copy())
        return x, torch.zeros(x.shape[1:]), i                     # 정답 없음 (0)


class Ms2Rgb(Dataset):
    def __init__(self, root, limit):
        self.root = root
        self.items = pick([(f"{s}/{n}", s, n) for _, s, n in frames(root, 0)], limit)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _, s, n = self.items[i]
        x = torch.from_numpy(np.asarray(Image.open(os.path.join(self.root, "sync_data", s, "rgb", "img_left", n)).convert("RGB")).transpose(2, 0, 1).copy())
        gt = torch.from_numpy(np.asarray(Image.open(os.path.join(self.root, "proj_depth", s, "rgb", "depth_filtered", n))).astype(np.float32) / 256.0)
        return x, gt, i


def ratio_stats(d0, d1, ok):
    """깊이 둘 (H, W) · 쓸 화소 → (중앙값 0, 중앙값 1, 스케일을 맞춘 뒤의 평균 상대 차). 쓸 화소가 100 개 미만이면 None."""
    if int(ok.sum()) < 100:
        return None
    a, b = d0[ok].double(), d1[ok].double()
    ma, mb = float(a.median()), float(b.median())
    return ma, mb, float(((a / ma - b / mb).abs() / (b / mb)).mean())


def scale_err(pred, gt):
    """정답 대비 (log (정답 중앙값 / 예측 중앙값), raw AbsRel). 정답 화소가 100 개 미만이면 None."""
    m = (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH) & torch.isfinite(pred) & (pred > 0)
    if int(m.sum()) < 100:
        return None
    p, g = pred[m].double(), gt[m].double()
    return float(torch.log(g.median() / p.median())), float(((p - g).abs() / g).mean())


@torch.no_grad()
def run_set(ud, name, ds, dev, bs, workers):
    cam = CAMS[name]
    K = kmat(cam).to(dev) if cam else None
    rows, size = [], None
    for x8, gt, idx in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers):
        x8, gt = x8.to(dev), gt.to(dev)
        size = tuple(x8.shape[-2:])
        o0 = ud.infer(x8)
        d0, f0 = o0["depth"][:, 0].float(), o0["intrinsics"][:, 0, 0].float()
        # 카메라 값은 한 장씩 준다 — 공식 코드의 길 (README 예시) 이 그렇고, 배치로 주면 광선을 만드는 데서 크기가 어긋난다
        d1 = torch.cat([ud.infer(x8[j], K.clone())["depth"][:, 0].float() for j in range(len(x8))]) if K is not None else None
        lum = F.avg_pool2d((x8.float() / 255.0).mean(1, keepdim=True), 7, 1, 3, count_include_pad=False)[:, 0]
        use = lambda d: torch.isfinite(d) & (d >= PL_MIN) & (d <= PL_MAX) & (lum >= PL_DARK)  # noqa: E731  (B, H, W)
        u0, u1 = use(d0), (use(d1) if d1 is not None else None)
        for j in range(len(idx)):
            r = {"set": name, "key": ds.items[int(idx[j])][0], "fx_pred": float(f0[j])}
            ok0 = u0[j]
            r["med_free"] = float(d0[j][ok0].median()) if int(ok0.sum()) >= 100 else None
            if d1 is not None:
                st = ratio_stats(d0[j], d1[j], ok0 & u1[j])
                if st:
                    r.update(med_free=st[0], med_cam=st[1], ratio=st[0] / st[1], shape_diff=st[2])
            if name == "ms2_rgb":
                for tag, d in (("free", d0[j]), ("cam", d1[j])):
                    e = scale_err(d, gt[j])
                    if e:
                        r[f"ls_{tag}"], r[f"absrel_{tag}"] = e
            rows.append(r)
    return rows, size


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def col(rows, k):
    return [r[k] for r in rows if r.get(k) is not None]


def summarize(name, rows, size):
    cam = CAMS[name]
    fx = col(rows, "fx_pred")
    s = {"n": len(rows), "size": list(size[::-1]), "fx_true": cam[0] if cam else None, "fx_pred_med": q(fx, 50), "fx_pred_q": [q(fx, 25), q(fx, 75)],
         "med_free": q(col(rows, "med_free"), 50)}
    line = f"[labelcheck] {name:8s} {size[1]}×{size[0]} · {len(rows):,} 장 · 실제 초점 " + (f"{cam[0]:.0f} px" if cam else "모름")
    line += f" · 선생이 짐작한 초점 중앙 {s['fx_pred_med']:.0f} px (25–75 %: {s['fx_pred_q'][0]:.0f}–{s['fx_pred_q'][1]:.0f}" + (f", 실제의 {s['fx_pred_med'] / cam[0]:.2f} 배)" if cam else ")")
    rt = col(rows, "ratio")
    if rt:
        s.update(med_cam=q(col(rows, "med_cam"), 50), ratio_med=q(rt, 50), ratio_q=[q(rt, 25), q(rt, 75)], shape_diff=q(col(rows, "shape_diff"), 50))
        line += (f" · 깊이 중앙 {s['med_free']:.1f} → {s['med_cam']:.1f} m (카메라 없이 → 주고) · 사진별 비 중앙 {s['ratio_med']:.3f} (25–75 %: {s['ratio_q'][0]:.3f}–{s['ratio_q'][1]:.3f})"
                 f" · 스케일을 맞춘 뒤의 차 {s['shape_diff']:.4f}")
    else:
        line += f" · 깊이 중앙 {s['med_free']:.1f} m (카메라 없이)"
    if name == "ms2_rgb":
        for tag in ("free", "cam"):
            ls = np.array(col(rows, f"ls_{tag}"))
            s[f"scale_{tag}"] = {"abs_med": float(np.median(np.abs(ls))), "bias": float(np.median(ls)), "abs_rel": float(np.mean(col(rows, f"absrel_{tag}")))} if len(ls) else None
        a, b = s["scale_free"], s["scale_cam"]
        if a and b:
            line += (f"\n[labelcheck] {name:8s} 정답 대비 (카메라 없이 → 주고): 스케일 오차 중앙 / 치우침 {a['abs_med']:.4f} / {a['bias']:+.4f} → {b['abs_med']:.4f} / {b['bias']:+.4f}"
                     f" · raw AbsRel {a['abs_rel']:.4f} → {b['abs_rel']:.4f}")   # 화면은 한 줄을 400 자에서 자르므로 줄을 나눈다
    return s, line


def verdicts(S):
    """요약 {세트: …} → (판정 dict, 화면 줄들). 기준은 README '라벨 점검'."""
    V, L, parts = {}, [], []
    for name in ("kaist", "freiburg", "tartan"):
        r = (S.get(name) or {}).get("ratio_med")
        if r is None or not np.isfinite(r):
            continue
        V[name] = "멀게 부풂" if r >= INFLATE else "가깝게 쏠림" if r <= 1.0 / INFLATE else "차이 작음"
        parts.append(f"{name} {r:.3f} {V[name]}")
    if parts:
        L.append(f"[판정] 라벨의 쏠림 (카메라 없이 / 카메라 값을 주고, 사진별 깊이 비의 중앙값 — 기준 {INFLATE} · {1 / INFLATE:.2f}): " + " · ".join(parts))
    m = S.get("ms2_rgb") or {}
    if m.get("scale_free") and m.get("scale_cam"):
        d = m["scale_cam"]["abs_med"] - m["scale_free"]["abs_med"]
        V["ms2_rgb"] = "바로잡음" if d < -MS2_TOL else "나빠짐" if d > MS2_TOL else "차이 없음"
        L.append(f"[판정] 카메라 값의 효과 (MS2 RGB, 정답 대비 스케일 오차 중앙값 {m['scale_free']['abs_med']:.4f} → {m['scale_cam']['abs_med']:.4f}, 기준 ±{MS2_TOL}): {V['ms2_rgb']}")
    return V, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--limit", type=int, default=1000, help="세트마다 고르게 뽑을 장 수 (0 = 전부)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    sets = [(n, PairRgb(n, a.data, a.limit)) for n in D.LABEL_SETS] + [("ms2_rgb", Ms2Rgb(a.data, a.limit))]
    empty = [n for n, ds in sets if not len(ds)]
    if empty:
        raise SystemExit(f"!!! [labelcheck] 사진이 없는 세트: {', '.join(empty)}")
    t0 = time.time()
    ud = build_unidepth(a.encoder, dev)
    print(f"[labelcheck] 선생 UniDepthV2 {a.encoder} · {dev.type} · 세트마다 최대 {a.limit or '전부'} 장 · 배치 {a.bs}", flush=True)
    S, rows_all = {}, []
    for name, ds in sets:
        rows, size = run_set(ud, name, ds, dev, a.bs, a.workers)
        S[name], line = summarize(name, rows, size)
        rows_all += rows
        print(line, flush=True)
    V, L = verdicts(S)
    print("\n".join(L), flush=True)
    keys = ["set", "key", "fx_pred", "med_free", "med_cam", "ratio", "shape_diff", "ls_free", "ls_cam", "absrel_free", "absrel_cam"]
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_all:
            w.writerow({k: ("" if r.get(k) is None else f"{r[k]:.5f}" if isinstance(r[k], float) else r[k]) for k in keys})
    write_json(os.path.join(a.out, "summary.json"), {"sets": S, "verdict": V, "cams": {k: list(v) if v else None for k, v in CAMS.items()},
                                                    "encoder": a.encoder, "limit": a.limit, "minutes": round((time.time() - t0) / 60, 1)})
    gpu = f" · GPU 최대 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if dev.type == "cuda" else ""
    print(f"[labelcheck] 끝 — {(time.time() - t0) / 60:.1f} 분{gpu}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
