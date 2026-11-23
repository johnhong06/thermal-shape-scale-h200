"""데이터 — 풀린 묶음 (환경 변수 TD_DATA) 에서 16 비트 열화상과 깊이 정답 (÷256 = m) 을 읽는다.

세트 이름
  train · val · test_day · test_night · test_rainy   MS2 (분할 목록 = splits/ms2/<이름>_list.txt, 시퀀스 순서 → 프레임 이름순 = 공식 로더 순서)
  fire                                                FIReStereo 저자 test 폴더 전부 (학습에 쓰지 않는 zero-shot 대상)
  vivid · vivid_day                                   ViViD 야외 밤 (night1 · night2 = 선행 논문의 zero-shot 평가) · 낮 (day1 · day2, 참고). 학습에 쓰지 않는다
  vivid_in                                            ViViD 실내 — 저자 (ThermalSfMLearner-MS) 의 test 시퀀스 다섯. 학습에 쓰지 않는다
  sthereo                                             STheReO Valley Evening (차량 주행) — 정답은 LiDAR 점을 직접 옮겨 찍어 만든 것 (README '데이터'). 학습에 쓰지 않는다
묶음에는 이미 솎은 프레임만 들어 있다 (README '데이터') — 여기서는 있는 것을 전부 쓴다.

열화상 → 입력 (tone): 사진마다 1–99 백분위로 자르고 [0, 1] 로 (MS2 공식 벤치마크의 TensorIWMM 과 같은 처리) → 3 채널로 복사 → ImageNet 평균 · 표준편차.
학습 증강 = 좌우 뒤집기 (p 0.5) 만 — 공식 Depth Anything V2 metric 학습과 같다.
증강의 난수는 (시드, epoch, 사진 번호) 로 정한다 → 이어 하기 · 작업자 수와 무관하게 같다.

정답 없는 RGB-열화상 쌍 (Pairs — 의사 라벨 팔 pl · plv 의 학습에만 쓴다. 증강 없음)
  freiburg · tartan 은 팔 ours 가 쓴다 (PairLabels — 열화상 + 미리 뽑아 둔 선생 라벨, src/labels.py).
  kaist      KAIST 다중분광 보행자, 영상마다 10 장마다 1 장 (640×512, 화소 정렬, 8 비트 JPEG)
  llvip      LLVIP, 3 장마다 1 장 (640×512 로 줄인 것, 화소 정렬, 8 비트 JPEG)
  vivid_day  ViViD 야외 낮 — 열화상은 평가 묶음의 것 (640×512, 16 비트), RGB 는 다른 카메라 (640×480) 라 common.vivid_grid 로 옮겨 쓴다

  python src/data.py check     # 세트마다 장 수를 setup/data.json 과 대조 + 첫 장 모양 확인
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CAMS  # noqa: E402

MS2_SETS = ("train", "val", "test_day", "test_night", "test_rainy")
VIVID = {"vivid": ("outdoor_robust_night1", "outdoor_robust_night2"), "vivid_day": ("outdoor_robust_day1", "outdoor_robust_day2"),
         "vivid_in": ("indoor_robust_dark", "indoor_robust_varying", "indoor_aggresive_dark", "indoor_unstable_dark", "indoor_aggresive_local")}
SETS = MS2_SETS + ("fire",) + tuple(VIVID) + ("sthereo",)
PAIR_SETS = ("kaist", "llvip", "vivid_day", "freiburg", "tartan")
LABEL_SETS = ("kaist", "llvip", "freiburg", "tartan")       # 팔 ours · rtp · mtd 가 쓰는 쌍 — 평가 대상 (ViViD) 의 것은 넣지 않는다
LABEL_GROUPS = (("kaist", "llvip", "tartan"), ("freiburg",))     # 같은 배치에 묶는 쌍 — 크기가 같은 것끼리 (640×512 · 640×298)
CAM_LABEL_SETS = tuple(n for n in LABEL_SETS if CAMS[n])          # 카메라 값을 아는 쌍 (kaist · freiburg · tartan) — 카메라 정보를 쓰는 팔 (rtn · rtf · rtk) 의 쌍
CAM_LABEL_GROUPS = (("kaist", "tartan"), ("freiburg",))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def frames(name, root=None):
    """[(키, 열화상 경로, 깊이 경로)] — 공식 순서. 묶음에 없는 시퀀스는 건너뛴다 (smoke 묶음)."""
    root = root or os.environ["TD_DATA"]
    out = []
    if name == "fire":
        base = os.path.join(root, "firestereo")
        for d in sorted(x for x in (os.listdir(base) if os.path.isdir(base) else []) if x.endswith("_test")):
            for n in sorted(x for x in os.listdir(os.path.join(base, d, "img_left")) if x.endswith(".png")):
                out.append((f"{d}/{n}", os.path.join(base, d, "img_left", n), os.path.join(base, d, "depth_filtered", n)))
        return out
    if name in VIVID:
        for s in VIVID[name]:
            img = os.path.join(root, "vivid", s, "Thermal")
            for n in sorted(x for x in (os.listdir(img) if os.path.isdir(img) else []) if x.endswith(".png")):
                out.append((f"{s}/{n}", os.path.join(img, n), os.path.join(root, "vivid", s, "Depth_T", n)))
        return out
    if name == "sthereo":
        base = os.path.join(root, "sthereo")
        for q in sorted(os.listdir(base) if os.path.isdir(base) else []):
            img = os.path.join(base, q, "Thermal")
            for n in sorted(x for x in (os.listdir(img) if os.path.isdir(img) else []) if x.endswith(".png")):
                out.append((f"{q}/{n}", os.path.join(img, n), os.path.join(base, q, "Depth_T", n)))
        return out
    with open(os.path.join(HERE, "..", "splits", "ms2", f"{name}_list.txt")) as f:
        seqs = [s.strip() for s in f if s.strip()]
    for s in seqs:
        img = os.path.join(root, "sync_data", s, "thr", "img_left")
        if not os.path.isdir(img):
            continue
        for n in sorted(x for x in os.listdir(img) if x.endswith(".png")):
            out.append((f"{s}/{n}", os.path.join(img, n), os.path.join(root, "proj_depth", s, "thr", "depth_filtered", n)))
    return out


def pair_frames(name, root=None):
    """[(키, 열화상 경로, RGB 경로)] — 이름순."""
    root = root or os.environ["TD_DATA"]
    ls = lambda d: sorted(os.listdir(d)) if os.path.isdir(d) else []  # noqa: E731
    out = []
    if name == "kaist":
        base = os.path.join(root, "pairs", "kaist")
        for st in ls(base):
            for v in ls(os.path.join(base, st)):
                for n in ls(os.path.join(base, st, v, "lwir")):
                    out.append((f"{st}/{v}/{n}", os.path.join(base, st, v, "lwir", n), os.path.join(base, st, v, "visible", n)))
    elif name == "llvip":
        base = os.path.join(root, "pairs", "llvip")
        out = [(n, os.path.join(base, "infrared", n), os.path.join(base, "visible", n)) for n in ls(os.path.join(base, "infrared"))]
    elif name == "freiburg":                       # Freiburg Thermal: 640×298, 열화상 16 비트 PNG · RGB JPEG
        base = os.path.join(root, "pairs", "freiburg")
        for seq in ls(base):
            for sub in ls(os.path.join(base, seq)) if os.path.isdir(os.path.join(base, seq)) else []:
                for n in ls(os.path.join(base, seq, sub, "ir")):
                    out.append((f"{seq}/{sub}/{n}", os.path.join(base, seq, sub, "ir", n), os.path.join(base, seq, sub, "rgb", n[:-4] + ".jpg")))
    elif name == "tartan":                         # TartanRGBT: 640×512, 열화상 8 비트 PNG · RGB (열화상 시점으로 옮긴 것, 빈 곳은 0) JPEG
        base = os.path.join(root, "pairs", "tartan")
        for day in ls(base):
            for tid in ls(os.path.join(base, day)) if os.path.isdir(os.path.join(base, day)) else []:
                for n in ls(os.path.join(base, day, tid, "thermal")):
                    out.append((f"{day}/{tid}/{n}", os.path.join(base, day, tid, "thermal", n), os.path.join(base, day, tid, "rgb", n[:-4] + ".jpg")))
    elif name == "vivid_day":
        for s in VIVID["vivid_day"]:
            base = os.path.join(root, "vivid", s)
            out += [(f"{s}/{n}", os.path.join(base, "Thermal", n), os.path.join(base, "RGB", n)) for n in ls(os.path.join(base, "Thermal"))]
    else:
        raise ValueError(name)
    return out


def tone(raw):
    """16 비트 열화상 (H, W) → [0, 1] float32: 사진마다 1–99 백분위로 자른다."""
    x = raw.astype(np.float32)
    lo, hi = (float(v) for v in np.percentile(x, [1.0, 99.0]))      # 파이썬 float 로 — numpy 스칼라와 섞이면 float64 로 올라간다
    return np.clip((x - lo) / max(hi - lo, 1.0), 0.0, 1.0).astype(np.float32, copy=False)


def load_pair(img_path, depth_path):
    raw = np.array(Image.open(img_path))
    dep = np.array(Image.open(depth_path))
    if raw.ndim != 2 or raw.dtype != np.uint16 or dep.shape != raw.shape:
        raise ValueError(f"예상과 다른 사진: {os.path.basename(img_path)} {raw.shape} {raw.dtype} / 깊이 {dep.shape} {dep.dtype}")
    return raw, dep.astype(np.float32) / 256.0


def flipped(seed, epoch, i):
    """학습 증강 (좌우 뒤집기) 이 이 사진에 걸렸는가 — Thermal 이 쓰는 것과 같은 난수. 카메라 값을 같이 뒤집을 때 학습 쪽이 다시 부른다."""
    return bool(np.random.default_rng([seed, epoch, i]).random() < 0.5)


class Thermal(Dataset):
    def __init__(self, name, limit=0, aug=None, seed=0, root=None):
        assert name in SETS and aug in (None, "base")
        self.name, self.aug, self.seed, self.epoch = name, aug, seed, 0
        self.items = frames(name, root)
        if limit and limit < len(self.items):                       # smoke: 같은 간격으로 limit 장
            self.items = [self.items[round(i * (len(self.items) - 1) / max(limit - 1, 1))] for i in range(limit)]
        self.fx_key = name if name in ("fire", "sthereo") else "vivid" if name in VIVID else "ms2"
        self.border = name in VIVID                                 # 평가 때 정답 테두리를 뺄지 (common.border_box)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, ip, dp = self.items[i]
        raw, dep = load_pair(ip, dp)
        x = tone(raw)
        if self.aug and flipped(self.seed, self.epoch, i):
            x, dep = x[:, ::-1], dep[:, ::-1]
        x = (np.repeat(np.ascontiguousarray(x)[None], 3, 0) - MEAN) / STD
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(dep)), i


class Pairs(Dataset):
    """정답 없는 쌍 → (열화상 입력 (3, H, W), RGB 입력 (3, Hr, Wr)). 둘 다 ImageNet 정규화. 열화상은 평가 때와 같은 고정 톤 처리."""

    def __init__(self, names, limit=0, root=None):
        self.items = [x for n in names for x in pair_frames(n, root)]
        if limit and limit < len(self.items):
            self.items = [self.items[round(i * (len(self.items) - 1) / max(limit - 1, 1))] for i in range(limit)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _, tp, rp = self.items[i]
        t = Image.open(tp)
        t = np.array(t if t.mode in ("L", "I;16", "I") else t.convert("L"))      # KAIST 의 lwir 는 세 채널이 같은 JPEG
        x = (np.repeat(tone(t)[None], 3, 0) - MEAN) / STD
        rgb = (np.asarray(Image.open(rp).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0 - MEAN) / STD
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(rgb))


def label_path(name, key, root=None, sub="labels"):
    """쌍 (세트, 키) 의 선생 라벨 파일 — src/labels.py 가 쓰고 PairLabels 가 읽는다. sub = labels (카메라 없이 뽑은 것) · labels_cam (카메라 값을 주고 뽑은 것)."""
    return os.path.join(root or os.environ["TD_DATA"], sub, name, os.path.splitext(key)[0] + ".png")


class PairLabels(Dataset):
    """쌍 + 미리 뽑아 둔 선생 라벨 → (열화상 입력 (3, H, W), 선생 깊이 (H, W) m — 못 쓰는 화소는 0). 한 세트 안에서는 크기가 같다."""

    def __init__(self, names, limit=0, root=None, sub="labels", cams=False):
        self.items = [(n, k, tp) for n in names for k, tp, _ in pair_frames(n, root)]
        self.root, self.sub, self.cams = root, sub, cams              # cams = 세트의 카메라 값 (common.CAMS) 도 함께 낸다
        if limit and limit < len(self.items):
            self.items = [self.items[round(i * (len(self.items) - 1) / max(limit - 1, 1))] for i in range(limit)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, key, tp = self.items[i]
        t = Image.open(tp)
        t = np.array(t if t.mode in ("L", "I;16", "I") else t.convert("L"))
        lab = np.array(Image.open(label_path(name, key, self.root, self.sub)))
        if lab.dtype != np.uint16 or lab.shape != t.shape:
            raise ValueError(f"예상과 다른 라벨: {name}/{key} {lab.shape} {lab.dtype} / 열화상 {t.shape}")
        x = (np.repeat(tone(t)[None], 3, 0) - MEAN) / STD
        out = (torch.from_numpy(x), torch.from_numpy(lab.astype(np.float32) / 256.0))
        return out + (torch.tensor(CAMS[name], dtype=torch.float32),) if self.cams else out


def check():
    spec = json.load(open(os.path.join(HERE, "..", "setup", "data.json"), encoding="utf-8"))
    kind = open(os.path.join(os.environ["TD_DATA"], ".done")).read().split()[0]      # all | smoke (setup/get_data.py 가 남김)
    want, bad, parts = spec["counts"][kind], 0, []
    for name in SETS:
        ds = Thermal(name)
        ok = len(ds) == want[name]
        bad += not ok
        x, d, _ = ds[0]
        v = (d > 0).float().mean().item()
        parts.append(f"{name} {len(ds):,}" + ("" if ok else f" (!!! 기대 {want[name]:,})") + f" [{x.shape[1]}×{x.shape[2]} · 정답 {v * 100:.0f} %]")
    for name in PAIR_SETS:
        ds = Pairs([name])
        ok = len(ds) == want["pair_" + name]
        bad += not ok
        x, rgb = ds[0]
        parts.append(f"쌍 {name} {len(ds):,}" + ("" if ok else f" (!!! 기대 {want['pair_' + name]:,})") + f" [{x.shape[1]}×{x.shape[2]} / RGB {rgb.shape[1]}×{rgb.shape[2]}]")
    print(f"[data] 세트 확인 ({kind}): " + " · ".join(parts))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(check() if sys.argv[1:] == ["check"] else 2)
