"""RGB 기준 — RGB 용 metric 깊이 모델 (UniDepthV2, 우리 선생과 같은 가중치) 을 우리 평가 세트에서 잰다. README 'RGB 기준'.

  ① 열화상을 넣은 것: 열화상 평가 세트 전부 (MS2 test 셋 · FIReStereo · ViViD 밤 · 낮 · 실내 · STheReO) — 논문 표 I 의 'RGB 모델을 열화상에 그대로' 줄.
  ② RGB 를 넣은 것: ViViD 의 같은 프레임을 RGB 카메라로 (밤 야외 · 낮 야외 · 실내) — '어둠에서 RGB 카메라의 깊이가 열화상보다 나쁜가'.
     세트 이름 rgb_vivid · rgb_vivid_day · rgb_vivid_in (data.RgbView). 정답 = RGB 시점으로 옮긴 같은 LiDAR (원본 Warped_Depth/data_RGB).
모델에는 세트의 카메라 값 (common.CAMS) 을 준다 — 한 장씩 (공식 infer 는 카메라를 줄 때 배치를 받지 않는다. GPU 에서는 공식 코드가 fp16 혼합 정밀도로 돈다).
열화상 입력 = 우리 모델과 같은 톤 처리 (사진마다 1–99 백분위 → 0–255) 를 세 채널로 겹친 것.
채점은 우리 평가와 같다 (evaluate.eval_set: 0.001–80 m · 사진 평균 · ViViD 는 정답 테두리를 뺌 · raw / median / focal). 실내는 어두운 시퀀스 셋 · 밝은 둘로도 나눠 적는다.
결과 (--out): metrics.json · per_image.csv · samples/ (src/evaluate.py 와 같은 꼴 — 표본은 우리 팔과 같은 프레임, RGB 쪽은 rgb_<세트>_<번호>.png).
열화상 모델 (MonoTher-Depth · Baseline · Ours) 과의 비교는 다른 작업의 결과로 로컬에서 한다 — 읽는 법은 README 'RGB 기준' 에 결과 전에 고정.

  python src/rgb_eval.py --out <폴더> [--encoder vitl|vits] [--limit N] [--sets …] [--rgb_sets …]
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import data as D  # noqa: E402
import evaluate as E  # noqa: E402
from common import CAMS, write_json  # noqa: E402
from teacher_check import build_unidepth  # noqa: E402

DARK = ("indoor_robust_dark", "indoor_aggresive_dark", "indoor_unstable_dark")    # ViViD 실내 test 시퀀스 다섯 가운데 이름이 '어둠' 인 셋 (나머지 둘 = 불빛이 바뀜 · 국소 조명)
THERMAL_SETS = E.TEST_SETS
RGB_SETS = tuple(D.RGB_SETS)


def thermal_u8(raw):
    """16 비트 열화상 → (3, H, W) uint8 — 우리 톤 처리 (data.tone) 를 0–255 로, 세 채널."""
    x = np.rint(D.tone(raw) * 255.0).astype(np.uint8)
    return np.repeat(x[None], 3, 0)


def predictor(ud, fx_key, dev):
    fx, fy, cx, cy = CAMS[fx_key]
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=dev)

    @torch.no_grad()
    def predict(x, hw):
        out = []
        for j in range(len(x)):
            d = ud.infer(x[j].to(torch.uint8), K.clone())["depth"][:, 0].float()    # (1, H, W) m
            if d.shape[-2:] != hw:
                d = F.interpolate(d[:, None], hw, mode="bilinear", align_corners=True)[:, 0]
            out.append(torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0))
        return torch.cat(out)
    return predict


def split_indoor(rows):
    """실내 세트의 사진별 행 → {'dark': 요약, 'lit': 요약} (키의 시퀀스 이름으로)."""
    out = {}
    for g, keep in (("dark", lambda k: k.split("/")[0] in DARK), ("lit", lambda k: k.split("/")[0] not in DARK)):
        sub = [r for r in rows if keep(r["key"])]
        if sub:
            out[g] = E.mean_of(sub) | {"n": len(sub)}
    return out


def tables(res):
    f = lambda s, v="raw": f"{res[s][v]['abs_rel']:.4f} / {res[s][v]['rmse']:.3f} / {res[s][v]['d1']:.4f}" if s in res else "—"  # noqa: E731
    L = ["[표R1] RGB 모델 (UniDepthV2, 카메라 값을 줌) 에 열화상을 그대로 — raw AbsRel / RMSE / δ1 | MS2 낮 | 밤 | 비 | ViViD 밤 | FIReStereo | ViViD 실내 | STheReO | ViViD 낮",
         "[표R1] 열화상 입력 | " + " | ".join(f(s) for s in ("test_day", "test_night", "test_rainy", "vivid", "fire", "vivid_in", "sthereo", "vivid_day"))]
    ms2 = [s for s in ("test_day", "test_night", "test_rainy") if s in res]
    unseen = [s for s in ("vivid", "fire", "vivid_in", "sthereo") if s in res]
    if len(ms2) == 3 and len(unseen) == 4:
        L.append(f"[표R1] MS2 평균 AbsRel {np.mean([res[s]['raw']['abs_rel'] for s in ms2]):.4f} · 처음 보는 네 곳 평균 {np.mean([res[s]['raw']['abs_rel'] for s in unseen]):.4f}")
    L.append("[표R2] 같은 ViViD 프레임 — RGB 카메라 (UniDepthV2 에 RGB) 대 열화상 카메라 (UniDepthV2 에 열화상). raw AbsRel / RMSE / δ1 (median AbsRel)")
    for name, rs, ts in (("밤 야외", "rgb_vivid", "vivid"), ("낮 야외", "rgb_vivid_day", "vivid_day"), ("실내 전체", "rgb_vivid_in", "vivid_in")):
        rr = f"{f(rs)} ({res[rs]['median']['abs_rel']:.4f})" if rs in res else "—"
        tt = f"{f(ts)} ({res[ts]['median']['abs_rel']:.4f})" if ts in res else "—"
        L.append(f"[표R2] {name} | RGB {rr} | 열화상 {tt}")
    for g, gn in (("dark", "실내 어둠 (셋)"), ("lit", "실내 밝음 (둘)")):
        rr = res.get("rgb_vivid_in", {}).get(g)
        tt = res.get("vivid_in", {}).get(g)
        h = lambda x: f"{x['raw']['abs_rel']:.4f} / {x['raw']['rmse']:.3f} / {x['raw']['d1']:.4f} ({x['median']['abs_rel']:.4f}, {x['n']} 장)" if x else "—"  # noqa: E731
        L.append(f"[표R2] {gn} | RGB {h(rr)} | 열화상 {h(tt)}")
    L.append("[표R2] 열화상 모델 (MonoTher-Depth · Baseline · Ours) 과의 비교는 앞선 작업의 결과로 — README 'RGB 기준' 의 읽는 법 (결과 전에 고정)")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="vitl", choices=["vitl", "vits"])
    ap.add_argument("--sets", default=",".join(THERMAL_SETS))
    ap.add_argument("--rgb_sets", default=",".join(RGB_SETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    ud = build_unidepth(a.encoder, dev)
    print(f"[rgb] UniDepthV2 {a.encoder} · 파라미터 {sum(p.numel() for p in ud.parameters()) / 1e6:.1f} M · {dev.type}", flush=True)
    todo = [("thermal", s) for s in a.sets.split(",") if s] + [("rgb", s) for s in a.rgb_sets.split(",") if s]
    res = {}
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "key", "raw_abs_rel", "raw_rmse", "raw_d1", "median_abs_rel", "median_rmse", "median_d1", "median_scale"])
        for kind, s in todo:
            ds = D.Thermal(s, a.limit, prep=thermal_u8) if kind == "thermal" else D.RgbView(s, a.limit)
            if not len(ds):
                print(f"[rgb] {s}: 사진 없음 — 건너뜀 (묶음에 없는 세트)", flush=True)
                continue
            res[s], rows = E.eval_set(ud, s, dev, a.limit, a.bs, a.workers, False, a.samples, a.out, ds=ds, predict=predictor(ud, ds.fx_key, dev))
            if s in ("vivid_in", "rgb_vivid_in"):
                res[s] |= split_indoor(rows)
            for r in rows:
                w.writerow([s, r["key"]] + [f"{r[v][k]:.5f}" for v in ("raw", "median") for k in ("abs_rel", "rmse", "d1")] + [f"{r['median']['scale']:.4f}"])
            print(f"[eval rgbmodel {'열화상' if kind == 'thermal' else 'RGB'}] " + E.line(s, res[s]), flush=True)
    print("\n".join(tables(res)), flush=True)
    write_json(os.path.join(a.out, "metrics.json"), res | {"_model": f"UniDepthV2 {a.encoder}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
