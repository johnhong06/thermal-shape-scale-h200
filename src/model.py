"""모델 — 공식 Depth Anything V2 의 metric 판 (metric_depth/depth_anything_v2/dpt.py) 을 그대로 불러 쓴다. 코드는 setup/fetch_code.sh 가 받는다 (TD_CODE).

공식 metric 학습 (metric_depth/train.py) 과 같은 것: 상대 깊이 체크포인트에서 인코더 ('pretrained' 가 들어간 키) 만 싣고 DPT 헤드는 새로 초기화,
출력 = Sigmoid × max_depth. 다른 것: max_depth = 80 m (MS2 · 선행 평가 상한), 입력 크기.
입력 크기: 사진의 가로 · 세로를 14 의 배수로 반올림 (256×640 → 252×644, 512×640 → 518×644) — 학습 · 평가 · 다른 데이터 모두 같은 규칙이라 화소 축척이 유지된다.
"""
import os
import sys

import torch
import torch.nn.functional as F

from common import MAX_DEPTH

CFG = {"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
       "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
       "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}}


def build(encoder="vitl", weights=""):
    """weights = 상대 깊이 체크포인트 (.pth) 경로. 비우면 무작위 초기화 (시험용)."""
    code = os.path.join(os.environ["TD_CODE"], "metric_depth")
    if code not in sys.path:
        sys.path.insert(0, code)
    from depth_anything_v2.dpt import DepthAnythingV2
    m = DepthAnythingV2(**CFG[encoder], max_depth=MAX_DEPTH)
    if weights:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        r = m.load_state_dict({k: v for k, v in sd.items() if "pretrained" in k}, strict=False)
        if r.unexpected_keys or any(not k.startswith("depth_head.") for k in r.missing_keys):
            raise SystemExit(f"!!! [model] 인코더 가중치가 모델과 안 맞음 (남는 키 {len(r.unexpected_keys)}, 빠진 인코더 키 있음)")
    return m


def net_size(h, w):
    return max(14, round(h / 14) * 14), max(14, round(w / 14) * 14)


def predict(model, x, out_hw, amp=False):
    """정규화된 입력 (B, 3, H, W) → 깊이 (B, out_h, out_w) m, float32. 공식 평가처럼 bilinear · align_corners=True 로 정답 크기에 맞춘다."""
    x = F.interpolate(x, net_size(*x.shape[-2:]), mode="bilinear", align_corners=False)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
        d = model(x)
    return F.interpolate(d.float()[:, None], out_hw, mode="bilinear", align_corners=True)[:, 0]


def count_params(model):
    enc = sum(p.numel() for n, p in model.named_parameters() if "pretrained" in n)
    head = sum(p.numel() for n, p in model.named_parameters() if "pretrained" not in n)
    return {"encoder_m": round(enc / 1e6, 2), "head_m": round(head / 1e6, 2)}


def build_teacher(encoder="vitl", weights=""):
    """선생 = 공식 Depth Anything V2 상대 깊이 모델 (고정). 상대 깊이판 dpt.py 와 metric 판의 차이는 마지막 활성 (ReLU 대 Sigmoid) 과 max_depth 곱뿐이라
    (2026-11-16 두 파일을 대조), metric 판 클래스에 상대 깊이 체크포인트를 전부 싣고 마지막 Sigmoid 를 ReLU 로 바꾸면 같은 모델이 된다. 출력 = 역깊이 꼴의 상대 깊이.
    weights 를 비우면 무작위 초기화 (시험용)."""
    code = os.path.join(os.environ["TD_CODE"], "metric_depth")
    if code not in sys.path:
        sys.path.insert(0, code)
    from depth_anything_v2.dpt import DepthAnythingV2
    m = DepthAnythingV2(**CFG[encoder], max_depth=1.0)
    if weights:
        m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    last = m.depth_head.scratch.output_conv2
    if not isinstance(last[-1], torch.nn.Sigmoid):
        raise SystemExit("!!! [model] 공식 코드의 헤드 구조가 예상과 다름 (마지막 층이 Sigmoid 가 아님)")
    last[-1] = torch.nn.ReLU()
    return m.eval().requires_grad_(False)


@torch.no_grad()
def teacher_disp(teacher, rgb, amp=False):
    """정규화된 RGB (B, 3, H, W) → 상대 깊이 (B, H, W), float32. 입력 크기 규칙은 학생과 같다 (14 의 배수로 반올림)."""
    x = F.interpolate(rgb, net_size(*rgb.shape[-2:]), mode="bilinear", align_corners=False)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
        d = teacher(x)
    return F.interpolate(d.float()[:, None], rgb.shape[-2:], mode="bilinear", align_corners=True)[:, 0]

