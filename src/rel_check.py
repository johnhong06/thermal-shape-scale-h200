"""상대 깊이 모델 점검 (run.sh relcheck) — 열화상 판 TR2M 의 단계 1. 학습 없음. 판정 기준은 README '열화상 판 TR2M'.

열화상 판 TR2M 은 고정한 상대 깊이 모델 R 의 출력에 작은 모듈이 화소별 scale · shift 맵을 걸어 m 단위 깊이를 낸다. R 이 열화상에서도 모양을 잘 내는지 먼저 잰다.
Depth Anything V2 상대 깊이 (ViT-L · ViT-S) 를 고정한 채 MS2 val 의 RGB 와 열화상에 돌려, 사진마다 역깊이에서 a · x + b 를 정답에 최소제곱으로 맞춘 뒤의
AbsRel · δ1 을 잰다 (선생 점검의 '모양 AbsRel' 과 같은 식). 사진 평균.
  열화상 = 학생과 같은 입력 (data.Thermal: 1–99 백분위 → 3 채널 → ImageNet 정규화), 정답 = 열화상 시점의 LiDAR
  RGB    = ImageNet 정규화, 정답 = RGB 시점의 LiDAR (선생 점검과 같은 묶음)
  둘 다 망 입력은 가로 · 세로를 14 의 배수로 반올림 (model.teacher_disp), fp32. MS2 의 test 분할은 쓰지 않는다.
  넣는 크기 넷 (VARIANTS): RGB 원래 크기 (384×1224 → 망 378×1218) · RGB 를 열화상과 같은 높이로 줄인 것 (256×816 → 망 252×812) ·
  열화상 원래 크기 (256×640 → 망 252×644, 학생과 같음) · 열화상을 두 배로 키운 것 (512×1280 → 망 518×1274). 예측은 정답 크기로 올려 잰다.
판정 (결과 전에 고정): 열화상 (원래 크기) AbsRel ≤ RGB (열화상과 같은 높이) AbsRel + GAP 인 모델이 있으면 그 가운데 열화상 오차가 작은 것을 R 로 고정한다.
  없으면 열화상용 상대 모델을 먼저 만든다. 나머지 두 크기는 진단으로만 찍는다 (RGB 원래 크기의 ViT-L 값은 선생 점검의 모양 AbsRel 0.1467 과 같아야 한다).

사용: python src/rel_check.py --data <푼 폴더> --out <폴더> [--models vitl,vits] [--weights_l <.pth>] [--weights_s <.pth>] [--limit N]
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
import model as M  # noqa: E402
from common import MIN_VALID, depth_metrics, edge_f1, eval_mask, seed_all, write_json  # noqa: E402
from teacher_check import aligned, frames  # noqa: E402

GAP = 0.02                # 열화상의 모양 오차가 RGB 보다 이만큼 넘게 나쁘면 고정해 쓰지 않는다
# (이름, 영상 종류, 크기 배율, 판정에 쓰는가, 화면 이름). RGB 를 256 / 384 로 줄이면 높이가 열화상 (256) 과 같아진다
VARIANTS = (("rgb", "rgb", 1.0, False, "RGB 원래 크기"), ("rgb_low", "rgb", 256 / 384, True, "RGB 열화상 높이"),
            ("thr", "thr", 1.0, True, "열화상"), ("thr_x2", "thr", 2.0, False, "열화상 ×2"))


class Rgb(Dataset):
    """MS2 val 의 RGB 왼쪽 + RGB 시점 정답 → (정규화된 RGB (3, H, W), 정답 (H, W) m, 번호)."""

    def __init__(self, root, limit=0):
        self.root = root
        self.items = [(s, n) for split, s, n in frames(root, 0) if split == "val"]
        if limit and limit < len(self.items):
            self.items = [self.items[round(i * (len(self.items) - 1) / max(limit - 1, 1))] for i in range(limit)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, n = self.items[i]
        rgb = np.asarray(Image.open(os.path.join(self.root, "sync_data", s, "rgb", "img_left", n)).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0
        gt = np.asarray(Image.open(os.path.join(self.root, "proj_depth", s, "rgb", "depth_filtered", n))).astype(np.float32) / 256.0
        if gt.shape != rgb.shape[1:]:
            raise ValueError(f"영상과 정답의 크기가 다름: {s}/{n}")
        return torch.from_numpy((rgb - D.MEAN) / D.STD), torch.from_numpy(gt), i


@torch.no_grad()
def measure(model, ds, dev, bs, workers, scale=1.0, amp=False, edges=False):
    """→ 사진별 [(번호, AbsRel, δ1)] (edges=True 면 끝에 경계 F1 (common.edge_f1, 경계가 없으면 None) 도) — 범위 안 정답이 MIN_VALID 미만인 사진은 건너뛴다.
    scale ≠ 1 이면 입력의 크기를 바꿔 넣고 예측을 정답 크기로 올린다. 정답 화소는 evaluate.py 와 같은 규칙 (ds.border 가 참이면 ViViD 의 테두리를 뺀다).
    1 단계의 val (train_rel) · 상대 깊이 점검 · 모양 실험의 평가 (rel_eval) 가 이 한 길을 쓴다."""
    rows = []
    for x, gt, idx in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers):
        gt, x = gt.to(dev), x.to(dev)
        if scale != 1.0:
            x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False, antialias=scale < 1.0)
        disp = M.teacher_disp(model, x, amp=amp)
        if disp.shape[-2:] != gt.shape[-2:]:
            disp = F.interpolate(disp[:, None], gt.shape[-2:], mode="bilinear", align_corners=True)[:, 0]
        for j in range(len(idx)):
            m = eval_mask(gt[j], getattr(ds, "border", False))
            if int(m.sum()) < MIN_VALID:
                continue
            al = aligned(disp[j], gt[j], m)
            r = depth_metrics(al[m], gt[j][m].double())
            if np.isfinite(r["abs_rel"]):
                rows.append((int(idx[j]), r["abs_rel"], r["d1"]) + ((edge_f1(al, gt[j], m),) if edges else ()))
    return rows


def verdict(stats, gap=GAP):
    """stats = {모델: {"rgb_low": AbsRel, "thr": AbsRel, …}} → (고를 모델 또는 None, 모델마다 통과 여부). 견주는 것은 열화상과 같은 높이로 줄인 RGB."""
    ok = {k: v["thr"] <= v["rgb_low"] + gap for k, v in stats.items()}
    passed = [k for k in stats if ok[k]]
    return (min(passed, key=lambda k: stats[k]["thr"]) if passed else None), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="vitl,vits")
    ap.add_argument("--weights_l", default="", help="Depth Anything V2 상대 깊이 ViT-L 체크포인트. 비우면 무작위 초기화 (흐름 시험)")
    ap.add_argument("--weights_s", default="", help="같은 ViT-S")
    ap.add_argument("--limit", type=int, default=0, help="양쪽에서 고르게 뽑을 사진 수 (0 = 전부)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seed_all(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thr, rgb = D.Thermal("val", a.limit, root=a.data), Rgb(a.data, a.limit)
    if not len(thr) or not len(rgb):
        print(f"!!! [relcheck] 사진이 없음 (열화상 {len(thr)} · RGB {len(rgb)}) — 데이터 폴더 확인")
        return 1
    models = [x for x in a.models.split(",") if x]
    weights = {"vitl": a.weights_l, "vits": a.weights_s}
    print(f"[relcheck] MS2 val · 열화상 {len(thr):,} 장 · RGB {len(rgb):,} 장 · 모델 {', '.join(models)} · {dev.type} · fp32", flush=True)
    t0, stats, out = time.time(), {}, []
    sets = {"rgb": rgb, "thr": thr}
    for enc in models:
        model = M.build_teacher(enc, weights[enc]).to(dev)
        st = {}
        for name, kind, scale, _, label in VARIANTS:
            ds = sets[kind]
            rows = measure(model, ds, dev, a.bs if scale <= 1.0 else max(1, a.bs // 4), a.workers, scale)
            if not rows:
                print(f"!!! [relcheck] {enc} {label}: 잴 수 있는 사진이 없음")
                return 1
            st[name], st[name + "_d1"], st[name + "_n"] = float(np.mean([r[1] for r in rows])), float(np.mean([r[2] for r in rows])), len(rows)
            keys = [it[0] for it in ds.items] if kind == "thr" else [f"{q}/{n}" for q, n in ds.items]   # 열화상 = (키, 영상, 정답) · RGB = (시퀀스, 이름)
            out += [{"model": enc, "input": name, "key": keys[i], "abs_rel": f"{ar:.5f}", "d1": f"{d1:.5f}"} for i, ar, d1 in rows]
            print(f"[relcheck] {enc} {label}: {len(rows):,} 장 · AbsRel {st[name]:.4f} · {(time.time() - t0) / 60:.1f} 분", flush=True)
        stats[enc] = st
        write_json(os.path.join(a.out, "summary.json"), {"stats": stats, "gap": GAP, "done": list(stats), "minutes": round((time.time() - t0) / 60, 1)})   # 모델마다 남긴다
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    pick, ok = verdict(stats)
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "input", "key", "abs_rel", "d1"])
        w.writeheader()
        w.writerows(out)
    write_json(os.path.join(a.out, "summary.json"), {"stats": stats, "gap": GAP, "pass": ok, "pick": pick, "random_init": {k: not weights[k] for k in models},
                                                       "minutes": round((time.time() - t0) / 60, 1)})
    L = ["[표] 상대 깊이 모델 (고정) 의 모양 오차 — MS2 val, 사진마다 역깊이에서 a · x + b 를 정답에 맞춘 뒤의 AbsRel · δ1. 판정은 '열화상' 과 'RGB 열화상 높이' 로"]
    for enc in models:
        s = stats[enc]
        L.append(f"[표] DAv2 {enc} | " + " | ".join(f"{label} {s[n]:.4f} · {s[n + '_d1']:.4f}" for n, _, _, _, label in VARIANTS)
                 + f" | 차 (열화상 − RGB 열화상 높이) {s['thr'] - s['rgb_low']:+.4f}" + (" (무작위 초기화 — 수치는 뜻이 없다)" if not weights[enc] else ""))
    L.append(f"[판정] 열화상 판 TR2M 의 상대 깊이 모델 (열화상 AbsRel ≤ 열화상과 같은 높이의 RGB AbsRel + {GAP}): "
             + " · ".join(f"{k} {'통과' if v else '못 넘음'}" for k, v in ok.items())
             + f" → {'DAv2 ' + pick + ' 를 고정해 쓴다' if pick else '열화상용 상대 모델을 먼저 만든다'}")
    print("\n".join(L), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
