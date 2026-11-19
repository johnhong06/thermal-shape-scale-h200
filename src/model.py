"""모델 — 공식 Depth Anything V2 의 metric 판 (metric_depth/depth_anything_v2/dpt.py) 을 그대로 불러 쓴다. 코드는 setup/fetch_code.sh 가 받는다 (TD_CODE).

공식 metric 학습 (metric_depth/train.py) 과 같은 것: 상대 깊이 체크포인트에서 인코더 ('pretrained' 가 들어간 키) 만 싣고 DPT 헤드는 새로 초기화,
출력 = Sigmoid × max_depth. 다른 것: max_depth = 80 m (MS2 · 선행 평가 상한), 입력 크기.
입력 크기: 사진의 가로 · 세로를 14 의 배수로 반올림 (256×640 → 252×644, 512×640 → 518×644) — 학습 · 평가 · 다른 데이터 모두 같은 규칙이라 화소 축척이 유지된다.

모양 + 스케일 학생 (Sep — 팔 sep · ours): 같은 인코더 · 같은 DPT 헤드에서 마지막 Sigmoid 만 떼어 그 앞의 값 y 를 log 깊이 꼴로 읽고,
  모양 zn = y − (사진 안 y 의 중앙값), 스케일 s = 인코더 마지막 층의 대표 토큰 → 작은 MLP, 깊이 = exp (s + zn).
"""
import os
import sys

import torch
import torch.nn.functional as F

from common import MAX_DEPTH, MIN_DEPTH, S0

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


class Fp32(torch.nn.Module):
    """감싼 층을 fp32 로 계산한다 (자동 혼합 정밀도 밖에서)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return self.m(x.float())


class Sep(torch.nn.Module):
    """모양 + 스케일 학생 — 공식 metric 판 모델을 감싼다. forward → (y (B, h, w): log 깊이 꼴의 값, s (B,): 장면 중앙 깊이의 log)."""

    def __init__(self, net):
        super().__init__()
        last = net.depth_head.scratch.output_conv2
        if not isinstance(last[-1], torch.nn.Sigmoid):
            raise SystemExit("!!! [model] 공식 코드의 헤드 구조가 예상과 다름 (마지막 층이 Sigmoid 가 아님)")
        last[-1] = torch.nn.Identity()
        # y 는 log 깊이라 값의 크기가 1–4 쯤이고, 그 상수 위치를 묶는 손실이 없다. bf16 으로 내면 눈금이 0.008–0.03 (깊이의 1–3 %) 이 되므로 마지막 1×1 합성곱만 fp32 로 계산한다
        last[-2] = Fp32(last[-2])
        self.net = net
        self.scale_head = torch.nn.Sequential(torch.nn.Linear(net.pretrained.embed_dim, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))
        torch.nn.init.zeros_(self.scale_head[-1].weight)             # 처음에는 모든 사진에 같은 값 S0 (MS2 학습 정답의 중앙 깊이) 를 낸다
        torch.nn.init.constant_(self.scale_head[-1].bias, S0)

    def forward(self, x):
        net = self.net
        feats = net.pretrained.get_intermediate_layers(x, net.intermediate_layer_idx[net.encoder], return_class_token=True)
        y = net.depth_head(feats, x.shape[-2] // 14, x.shape[-1] // 14)[:, 0]
        with torch.autocast(x.device.type, enabled=False):          # 스케일은 값 하나가 사진 전체를 옮기므로 fp32 로 낸다 (bf16 은 3 근처에서 눈금이 0.016)
            s = self.scale_head(feats[-1][1].float())[:, 0]
        return y, s


def net_size(h, w):
    return max(14, round(h / 14) * 14), max(14, round(w / 14) * 14)


def predict_sep(model, x, out_hw, amp=False):
    """Sep 학생: 정규화된 입력 (B, 3, H, W) → (모양 zn (B, out_h, out_w), 스케일 s (B,)), float32. 중앙값은 망의 해상도에서 빼고, 그 뒤에 정답 크기로 올린다."""
    x = F.interpolate(x, net_size(*x.shape[-2:]), mode="bilinear", align_corners=False)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
        y, s = model(x)
    y = y.float()
    zn = y - y.flatten(1).median(1).values[:, None, None]
    return F.interpolate(zn[:, None], out_hw, mode="bilinear", align_corners=True)[:, 0], s.float()


def sep_depth(zn, s):
    return torch.exp((s[:, None, None] + zn).clamp(-10.0, 10.0)).clamp(MIN_DEPTH, MAX_DEPTH)


def predict(model, x, out_hw, amp=False):
    """정규화된 입력 (B, 3, H, W) → 깊이 (B, out_h, out_w) m, float32. 공식 평가처럼 bilinear · align_corners=True 로 정답 크기에 맞춘다."""
    if isinstance(model, Sep):
        return sep_depth(*predict_sep(model, x, out_hw, amp))
    x = F.interpolate(x, net_size(*x.shape[-2:]), mode="bilinear", align_corners=False)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
        d = model(x)
    return F.interpolate(d.float()[:, None], out_hw, mode="bilinear", align_corners=True)[:, 0]


def count_params(model):
    enc = sum(p.numel() for n, p in model.named_parameters() if "pretrained" in n)
    head = sum(p.numel() for n, p in model.named_parameters() if "pretrained" not in n)
    out = {"encoder_m": round(enc / 1e6, 2), "head_m": round(head / 1e6, 2)}
    if isinstance(model, Sep):
        out["scale_head_m"] = round(sum(p.numel() for p in model.scale_head.parameters()) / 1e6, 3)      # head_m 에 들어 있는 몫
    return out


def load(encoder, ckpt):
    """학습한 모델 (final.pt) 을 싣는다 — 스케일 헤드가 들어 있으면 Sep 구조로."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m = build(encoder)
    if any(k.startswith("scale_head.") for k in sd):
        m = Sep(m)
    m.load_state_dict(sd)
    return m


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

