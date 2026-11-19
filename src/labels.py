"""쌍의 선생 라벨 (팔 ours) — UniDepthV2 를 쌍의 RGB 에 한 번 돌려 m 단위 깊이를 16 비트 PNG (× 256) 로 둔다. 학습 중에는 선생을 돌리지 않는다.

선생을 돌리는 길은 선생 점검 (src/teacher_check.py) 과 같다: 공식 예시대로 카메라 정보를 주지 않고 infer() 를 부른다 (공식 코드가 GPU 에서는 fp16 혼합 정밀도로 돈다).
못 쓰는 화소는 0 으로 적는다: RGB 밝기 (세 채널 평균, 7×7 평균) 가 PL_DARK 미만인 곳 (선생이 볼 수 없는 어두운 곳) · 선생 값이 PL_MIN–PL_MAX m 밖이거나 유한하지 않은 곳.
라벨은 열화상과 같은 크기다 (쌍은 화소가 맞춰져 있다). 읽는 쪽 = data.PairLabels.

  python src/labels.py --sets kaist,llvip,freiburg,tartan [--encoder vitl|vits] [--bs 8] [--workers 4]
출력: $TD_DATA/labels/<세트>/<쌍의 키>.png · labels/<세트>.json (장 수 · 쓸 화소 비율 · 깊이 중앙값). 끝난 세트는 다시 부르면 건너뛴다.
화면: 세트마다 한 줄.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
from common import PL_DARK, PL_MAX, PL_MIN, write_json  # noqa: E402
from teacher_check import build_unidepth  # noqa: E402


class Rgb(Dataset):
    def __init__(self, name):
        self.items = [(k, rp) for k, _, rp in D.pair_frames(name)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(Image.open(self.items[i][1]).convert("RGB")).transpose(2, 0, 1).copy()), i


def save_png(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path + ".tmp", "PNG")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def label_batch(ud, x8):
    """RGB (B, 3, H, W) uint8 → 라벨 (B, H, W) float32 m, 못 쓰는 화소는 0."""
    d = ud.infer(x8)["depth"][:, 0].float()
    lum = F.avg_pool2d((x8.float() / 255.0).mean(1, keepdim=True), 7, 1, 3, count_include_pad=False)[:, 0]
    ok = torch.isfinite(d) & (d >= PL_MIN) & (d <= PL_MAX) & (lum >= PL_DARK)
    return torch.where(ok, d, torch.zeros_like(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(D.LABEL_SETS))
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    root = os.path.join(os.environ["TD_DATA"], "labels")
    os.makedirs(root, exist_ok=True)
    ud = None
    for name in [x for x in a.sets.split(",") if x]:
        assert name in D.LABEL_SETS, name
        ds = Rgb(name)
        done = os.path.join(root, name + ".json")
        if os.path.exists(done):
            print(f"[labels] {name}: 이미 있음 — 건너뜀", flush=True)
            continue
        if not len(ds):
            raise SystemExit(f"!!! [labels] {name}: 쌍이 없음")
        if ud is None:
            ud = build_unidepth(a.encoder, dev)
        t0, last, n, use, meds = time.time(), time.time(), 0, 0.0, []
        with ThreadPoolExecutor(max(a.workers, 1)) as pool:
            jobs = []
            for x8, idx in DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers):
                lab = label_batch(ud, x8.to(dev))
                if lab.shape[-2:] != x8.shape[-2:]:
                    raise SystemExit(f"!!! [labels] {name}: 선생 출력의 크기가 입력과 다름 {tuple(lab.shape[-2:])} / {tuple(x8.shape[-2:])}")
                arr = (lab * 256.0).round().clamp(0, 65535).cpu().numpy().astype(np.uint16)
                for j in range(len(idx)):
                    v = lab[j][lab[j] > 0]
                    use += float((lab[j] > 0).float().mean())
                    if v.numel():
                        meds.append(float(v.median()))
                    jobs.append(pool.submit(save_png, D.label_path(name, ds.items[int(idx[j])][0]), arr[j]))
                n += len(idx)
                if time.time() - last > 120:
                    last = time.time()
                    print(f"[labels] {name}: {n:,} / {len(ds):,} 장 · {(time.time() - t0) / 60:.1f} 분", flush=True)
                if len(jobs) > 512:                                # 쓰기가 밀리면 여기서 기다린다 (메모리에 쌓이지 않게)
                    for f in jobs:
                        f.result()
                    jobs = []
            for f in jobs:
                f.result()
        info = {"n": n, "usable": use / max(n, 1), "depth_median_m": float(np.median(meds)) if meds else None, "encoder": a.encoder, "minutes": round((time.time() - t0) / 60, 1)}
        write_json(done, info)
        print(f"[labels] {name}: {n:,} 장 · 쓸 화소 {info['usable'] * 100:.1f} % · 깊이 중앙 {info['depth_median_m'] or float('nan'):.1f} m · {info['minutes']} 분", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
