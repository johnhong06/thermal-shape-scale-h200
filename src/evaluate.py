"""평가 — 세트마다 사진별 지표를 구해 평균한다 (규칙은 common.py 맨 위 · README '판정 기준').

  python src/evaluate.py --ckpt <final.pt> --out <폴더> [--sets test_day,test_night,test_rainy,fire,vivid,vivid_day,vivid_in,sthereo] [--limit N] [--samples 16]
결과 (--out): metrics.json (세트 · 예측 종류 (raw · focal · median) 별 평균, fire 는 낮 (hawkins) · 밤비 (frick) 로도 나눔.
                             scale = 사진별 스케일 오차 log (정답 중앙값 / 예측 중앙값) 의 |·| 중앙값 (abs_med) 과 치우침 (bias, 부호 있는 중앙값) — 오차를 스케일 몫과 모양 몫으로 나눠 보려고)
              per_image.csv (사진마다 raw · median 의 AbsRel · RMSE · δ1 과 중앙값 배율)
              samples/<세트>_<번호>.png (예측 깊이 × 256, 16 비트 — 그림용, 같은 간격으로 --samples 장)
화면: 세트마다 한 줄 (이름과 숫자만).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from common import CAMS, FX, FX_TRAIN, METRICS, VARIANTS, eval_mask, variants, write_json  # noqa: E402

TEST_SETS = ("test_day", "test_night", "test_rainy", "fire", "vivid", "vivid_day", "vivid_in", "sthereo")


def mean_of(rows):
    return {v: {k: float(np.mean([r[v][k] for r in rows])) for k in METRICS} for v in VARIANTS} | {"n": len(rows)}


@torch.no_grad()
def eval_set(model, name, dev, limit=0, bs=16, workers=4, amp=False, samples=0, out=None):
    """→ (요약 dict, 사진별 행). 범위 안 정답이 거의 없는 사진은 건너뛴다 (n 과 n_all 로 구분)."""
    ds = D.Thermal(name, limit)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=dev.type == "cuda")
    keep = {round(i * (len(ds) - 1) / max(samples - 1, 1)) for i in range(samples)} if samples else set()
    rows = []
    model.eval()
    cam = torch.tensor(CAMS[ds.fx_key], dtype=torch.float32, device=dev)   # 카메라 정보를 쓰는 모델 (TR 의 ray · canon) 만 읽는다
    fx = FX_TRAIN if bool(getattr(model, "canon", False)) else FX[ds.fx_key]   # 기준 초점으로 배운 모델은 predict 가 이미 초점을 되돌렸다 → focal 은 raw 와 같게 둔다
    for x, gt, idx in loader:
        gt = gt.to(dev)
        pred = M.predict(model, x.to(dev, non_blocking=True), gt.shape[-2:], amp, cam[None].expand(len(x), 4))
        for j in range(len(idx)):
            i = int(idx[j])
            v = variants(pred[j], gt[j], eval_mask(gt[j], ds.border), fx)
            if v is not None:
                rows.append({"i": i, "key": ds.items[i][0], **v})
            if i in keep and out:
                os.makedirs(os.path.join(out, "samples"), exist_ok=True)
                arr = (pred[j].clamp(0, 255.99) * 256).round().cpu().numpy().astype(np.uint16)
                Image.fromarray(arr).save(os.path.join(out, "samples", f"{name}_{i:05d}.png"))
    if not rows:
        raise SystemExit(f"!!! [eval] {name}: 평가할 사진이 없음")
    summ = mean_of(rows) | {"n_all": len(ds)}
    ls = np.log([max(r["median"]["scale"], 1e-6) for r in rows])     # median 의 배율 = 정답 중앙값 / 예측 중앙값 → 그 log 가 사진의 스케일 오차
    summ["scale"] = {"abs_med": float(np.median(np.abs(ls))), "bias": float(np.median(ls))}
    if name == "fire":                                             # 저자 설명: Hawkins = 낮, Frick = 비 오는 밤
        for g, pre in (("day", "hawkins"), ("night", "frick")):
            sub = [r for r in rows if r["key"].startswith(pre)]
            if sub:
                summ[g] = mean_of(sub)
    return summ, rows


def line(name, s):
    f = lambda v: f"AbsRel {s[v]['abs_rel']:.4f} RMSE {s[v]['rmse']:.3f} δ1 {s[v]['d1']:.4f}"  # noqa: E731
    return f"{name} {s['n']:,} 장 · raw {f('raw')} · median {f('median')} · focal AbsRel {s['focal']['abs_rel']:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=",".join(TEST_SETS))
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(os.path.dirname(os.path.normpath(a.out)))
    dev = torch.device(a.device)
    model = M.load(a.encoder, a.ckpt).to(dev)
    res = {}
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "key", "raw_abs_rel", "raw_rmse", "raw_d1", "median_abs_rel", "median_rmse", "median_d1", "median_scale"])
        for s in [x for x in a.sets.split(",") if x]:
            res[s], rows = eval_set(model, s, dev, a.limit, a.bs, a.workers, bool(a.amp) and dev.type == "cuda", a.samples, a.out)
            for r in rows:
                w.writerow([s, r["key"]] + [f"{r[v][k]:.5f}" for v in ("raw", "median") for k in ("abs_rel", "rmse", "d1")] + [f"{r['median']['scale']:.4f}"])
            print(f"[eval {name}] " + line(s, res[s]), flush=True)
    write_json(os.path.join(a.out, "metrics.json"), res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
