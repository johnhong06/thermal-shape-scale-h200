"""모델 — 공식 Depth Anything V2 의 metric 판 (metric_depth/depth_anything_v2/dpt.py) 을 그대로 불러 쓴다. 코드는 setup/fetch_code.sh 가 받는다 (TD_CODE).

공식 metric 학습 (metric_depth/train.py) 과 같은 것: 상대 깊이 체크포인트에서 인코더 ('pretrained' 가 들어간 키) 만 싣고 DPT 헤드는 새로 초기화,
출력 = Sigmoid × max_depth. 다른 것: max_depth = 80 m (MS2 · 선행 평가 상한), 입력 크기.
입력 크기: 사진의 가로 · 세로를 14 의 배수로 반올림 (256×640 → 252×644, 512×640 → 518×644) — 학습 · 평가 · 다른 데이터 모두 같은 규칙이라 화소 축척이 유지된다.

열화상 판 TR2M (TR — 팔 tr): 고정한 상대 깊이 모델 (Depth Anything V2 상대 깊이) 의 출력 D_r 에, 같은 모델의 마지막 층 패치 토큰을 받는 작은 모듈 (ScaleMap) 이
  화소별 scale 맵 A 와 shift 맵 B 를 내서 깊이 = 1 / (A ⊙ D_r + B). 학습하는 것은 모듈뿐이다. README '열화상 판 TR2M'.
  카메라 정보를 쓰는 판 (README '카메라 정보'): canon = 깊이를 기준 초점 (common.F_CANON) 에서 본 것으로 배우고 평가 때 카메라 값으로 되돌린다 ·
  ray = 토큰마다 시선 방향의 임베딩 (ray_tokens) 을 모듈 입력에 더한다. 둘 다 평가 때 카메라 값 (fx · fy · cx · cy) 이 있어야 한다.
모양 + 스케일 학생 (Sep — 팔 sep · ours): 같은 인코더 · 같은 DPT 헤드에서 마지막 Sigmoid 만 떼어 그 앞의 값 y 를 log 깊이 꼴로 읽고,
  모양 zn = y − (사진 안 y 의 중앙값), 스케일 s = 인코더 마지막 층의 대표 토큰 → 작은 MLP, 깊이 = exp (s + zn).
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

from common import MAX_DEPTH, MIN_DEPTH, S0, cam_resize, canon_scale

RAY_BANDS = 4                                                      # 시선 방향 임베딩의 주파수 띠 수 (2^k π, k = 0…3). 토큰 사이의 방향 차는 14 / 초점 ≈ 0.036 (크기를 0.71 배로 줄이면 0.05)
                                                                   # 이라 가장 높은 띠의 위상 차가 0.9–1.3 rad — 토큰 격자에서 겹치지 않는다 (k = 5 는 3.6 rad 로 겹친다, 2026-11-23 독립 검토)
RAY_DIM = 3 + 6 * RAY_BANDS                                        # 방향 3 + (sin · cos) × 3 × 띠

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


class RCU(torch.nn.Module):
    """잔차 합성곱 (DPT · TR2M 의 ResidualConvUnit, 정규화 없음): 활성 → 3×3 → 활성 → 3×3 + 건너뜀."""

    def __init__(self, c):
        super().__init__()
        self.c1, self.c2 = torch.nn.Conv2d(c, c, 3, 1, 1), torch.nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        return self.c2(F.relu(self.c1(F.relu(x)))) + x


class ScaleMap(torch.nn.Module):
    """TR2M 의 ScaleMapModel (공개 코드 scalemap_depth.py) 에서 텍스트 교차 어텐션만 뺀 것. 크기도 공개 코드의 기본값 그대로 (out 192 · features 128 · 머리 8).
    패치 토큰 (B, ph · pw, dim) → 자기 어텐션 + MLP → 격자 → 1×1 · 전치 합성곱 ×2 · 3×3 → (잔차 합성곱 · 1×1 · ×2 업샘플) 두 번 → 헤드 둘 → 망 입력 크기의
    scale 맵 · shift 맵 (B, 14 ph, 14 pw), 둘 다 coef · exp(·) 라 양수. 헤드의 마지막 두 층과 exp 는 fp32 로.
    공개 코드와 다른 점 하나: 두 헤드의 마지막 1×1 합성곱을 0 으로 시작해 처음 맵이 모든 화소에서 coef (0.01) 가 되게 한다.
    공개 코드의 초기화 그대로면 exp 앞의 값이 수백까지 퍼져 (ViT-L 토큰 크기에서 어림) 깊이가 0.001 m 나 80 m 쪽으로 몰리고 기울기가 사라진다 (2026-11-19 독립 검토)."""

    def __init__(self, dim, heads=8, out_ch=192, feat=128, coef=0.01, ray_dim=0):
        super().__init__()
        self.coef = coef
        self.ln1, self.ln2 = torch.nn.LayerNorm(dim), torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), torch.nn.GELU(), torch.nn.Linear(dim * 4, dim))
        self.proj = torch.nn.Conv2d(dim, out_ch, 1)
        self.up = torch.nn.ConvTranspose2d(out_ch, out_ch, 2, 2)
        self.rn = torch.nn.Conv2d(out_ch, feat, 3, 1, 1, bias=False)
        self.rcu1, self.oc1 = RCU(feat), torch.nn.Conv2d(feat, feat, 1)
        self.rcu2, self.oc2 = RCU(feat), torch.nn.Conv2d(feat, feat, 1)
        self.s1, self.t1 = torch.nn.Conv2d(feat, feat // 2, 3, 1, 1), torch.nn.Conv2d(feat, feat // 2, 3, 1, 1)
        tail = lambda: Fp32(torch.nn.Sequential(torch.nn.Conv2d(feat // 2, 32, 3, 1, 1), torch.nn.ReLU(True), torch.nn.Conv2d(32, 1, 1)))  # noqa: E731
        self.s2, self.t2 = tail(), tail()
        for m in self.modules():                                       # 공개 코드의 init_weights 와 같게 (합성곱 · 선형만, 전치 합성곱은 기본값)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        for t in (self.s2, self.t2):                                   # 처음 맵 = coef (위 설명)
            torch.nn.init.zeros_(t.m[-1].weight)
            torch.nn.init.zeros_(t.m[-1].bias)
        # 시선 방향 임베딩 (ray_tokens) 을 토큰에 더하는 투영 — 카메라 정보 (팔 rtk). 다른 층을 다 초기화한 뒤에 만든다: 같은 시드면 나머지 가중치가 임베딩 없는 모듈과 같게.
        # 0 으로 시작 — 처음에는 임베딩이 없는 모듈과 같은 출력.
        self.ray = torch.nn.Linear(ray_dim, dim) if ray_dim else None
        if self.ray is not None:
            torch.nn.init.zeros_(self.ray.weight)
            torch.nn.init.zeros_(self.ray.bias)

    def forward(self, tok, ph, pw, ray=None):
        if self.ray is not None:
            tok = tok + self.ray(ray.to(tok.dtype))
        h = self.ln1(tok)
        x = tok + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], ph, pw)
        x = self.rn(self.up(self.proj(x)))
        x = F.interpolate(self.oc1(self.rcu1(x)), scale_factor=2, mode="bilinear", align_corners=True)
        x = F.interpolate(self.oc2(self.rcu2(x)), scale_factor=2, mode="bilinear", align_corners=True)
        hw = (ph * 14, pw * 14)
        a = self.s2(F.interpolate(self.s1(x), hw, mode="bilinear", align_corners=True))[:, 0]
        b = self.t2(F.interpolate(self.t1(x), hw, mode="bilinear", align_corners=True))[:, 0]
        return self.coef * torch.exp(a.float().clamp(-30.0, 30.0)), self.coef * torch.exp(b.float().clamp(-30.0, 30.0))


class TR(torch.nn.Module):
    """열화상 판 TR2M. rel = 고정한 상대 깊이 모델 (build_teacher 의 것). forward → (D_r, A, B), 모두 (B, 14 ph, 14 pw) float32.
    고정한 쪽은 기울기 없이 돌리고 늘 eval 로 둔다. 영상 특징 = rel 의 마지막 층 패치 토큰 (정규화한 것) — 큰 모델을 한 번만 돌린다."""

    def __init__(self, rel, ray=False, canon=False):
        super().__init__()
        self.rel = rel.eval().requires_grad_(False)
        self.scalemap = ScaleMap(rel.pretrained.embed_dim, ray_dim=RAY_DIM if ray else 0)
        self.register_buffer("canon", torch.tensor(bool(canon)))      # 출력이 기준 초점의 깊이인가 (common.canon_scale) — 체크포인트에 남겨 평가가 되돌린다

    def train(self, mode=True):
        super().train(mode)
        self.rel.eval()
        return self

    def forward(self, x, ray=None):
        ph, pw = x.shape[-2] // 14, x.shape[-1] // 14
        with torch.no_grad():
            feats = self.rel.pretrained.get_intermediate_layers(x, self.rel.intermediate_layer_idx[self.rel.encoder], return_class_token=True)
            dr = self.rel.depth_head(feats, ph, pw)[:, 0].float()      # 상대 깊이 (역깊이 꼴, ReLU)
        a, b = self.scalemap(feats[-1][0], ph, pw, ray)
        return dr, a, b


def tr_depth(dr, a, b):
    """1 / (A ⊙ D_r + B). 자르지 않는다 — 자르면 끝에 붙은 화소의 기울기가 0 이 된다. a · b > 0 이고 D_r ≥ 0 이라 분모는 늘 양수다.
    평가는 common.variants 가 0.001–80 m 로 자른다 (다른 팔과 같은 규칙)."""
    return 1.0 / (a * dr + b).clamp(min=1e-8)


def build_tr(encoder="vitl", weights="", ray=False, canon=False):
    """weights = 상대 깊이 체크포인트 (전부 싣는다). 비우면 무작위 초기화 (시험용). ray · canon = 카메라 정보를 쓰는 팔 (TR 의 설명)."""
    return TR(build_teacher(encoder, weights), ray, canon)


def ray_tokens(cam, in_hw, net_hw, bands=RAY_BANDS):
    """카메라 값 cam (B, 4) = 입력 영상 (in_hw) 기준 fx · fy · cx · cy → 토큰 (14×14 패치의 가운데) 마다 시선 방향의 푸리에 임베딩 (B, ph · pw, RAY_DIM).
    방향 = normalize(K⁻¹ [u, v, 1]) — ZeroDepth (Guizilini 외, ICCV 2023) 의 기하 임베딩과 같은 꼴. 망에 넣으며 영상 크기가 바뀐 만큼 카메라 값도 바꾼다."""
    (h, w), (hn, wn) = in_hw, net_hw
    c = cam_resize(cam.float(), wn / w, hn / h)
    ph, pw = hn // 14, wn // 14
    u = (torch.arange(pw, device=cam.device, dtype=torch.float32) + 0.5) * 14 - 0.5
    v = (torch.arange(ph, device=cam.device, dtype=torch.float32) + 0.5) * 14 - 0.5
    x = ((u[None, None, :] - c[:, 2, None, None]) / c[:, 0, None, None]).expand(-1, ph, -1)
    y = ((v[None, :, None] - c[:, 3, None, None]) / c[:, 1, None, None]).expand(-1, -1, pw)
    r = torch.stack([x, y, torch.ones_like(x)], -1)
    r = (r / r.norm(dim=-1, keepdim=True)).reshape(len(cam), ph * pw, 3)
    ang = r[..., None] * ((2.0 ** torch.arange(bands, device=cam.device, dtype=torch.float32)) * math.pi)
    return torch.cat([r, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)


def canon_of(cam, in_hw):
    """입력 영상 (in_hw) 의 카메라 값 (B, 4) → 기준 초점으로의 환산 배율 (B,): 기준 깊이 = 실제 깊이 × 이 값.
    망이 보는 크기 (14 의 배수로 맞춘 뒤) 의 초점으로 센다 — 시선 방향 임베딩 (ray_tokens) 과 같은 기준이고, 학습 · 평가가 이 함수 하나를 쓴다."""
    (h, w), (hn, wn) = in_hw, net_size(*in_hw)
    return canon_scale(cam_resize(cam.float(), wn / w, hn / h))


def predict_tr(model, x, out_hw, amp=False, cam=None):
    """TR: 정규화된 입력 → (깊이 (B, out_h, out_w), A, B, D_r (망 입력 크기), 망 입력 크기의 입력). 학습의 손실이 맵과 입력을 함께 쓴다.
    cam (B, 4) = 입력 영상 기준의 카메라 값 — 시선 방향 임베딩을 쓰는 모듈에는 꼭 준다. 깊이는 모듈이 낸 그대로다 (model.canon 이면 기준 초점의 깊이 — 되돌리기는 predict 가)."""
    in_hw = x.shape[-2:]
    x = F.interpolate(x, net_size(*in_hw), mode="bilinear", align_corners=False)
    ray = None
    if model.scalemap.ray is not None:
        if cam is None:
            raise ValueError("시선 방향 임베딩을 쓰는 모델에는 카메라 값을 줘야 한다")
        ray = ray_tokens(cam, in_hw, x.shape[-2:])
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
        dr, a, b = model(x, ray)
    d = F.interpolate(tr_depth(dr, a, b)[:, None], out_hw, mode="bilinear", align_corners=True)[:, 0]
    return d, a, b, dr, x


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


def predict(model, x, out_hw, amp=False, cam=None):
    """정규화된 입력 (B, 3, H, W) → 깊이 (B, out_h, out_w) m, float32. 공식 평가처럼 bilinear · align_corners=True 로 정답 크기에 맞춘다.
    cam (B, 4) = 입력 영상의 카메라 값 — 카메라 정보를 쓰는 TR (시선 방향 임베딩 · 기준 초점) 에만 쓰인다. 기준 초점의 깊이는 여기서 실제 깊이로 되돌린다."""
    if isinstance(model, Sep):
        return sep_depth(*predict_sep(model, x, out_hw, amp))
    if isinstance(model, TR):
        d = predict_tr(model, x, out_hw, amp, cam)[0]
        if bool(model.canon):
            if cam is None:
                raise ValueError("기준 초점으로 배운 모델에는 카메라 값을 줘야 한다")
            d = d / canon_of(cam, x.shape[-2:])[:, None, None]
        return d
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
    out["trainable_m"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
    return out


def load(encoder, ckpt):
    """학습한 모델 (final.pt) 을 싣는다 — 스케일 헤드가 들어 있으면 Sep, 축척 맵 모듈이 들어 있으면 TR (고정한 상대 깊이 모델까지 체크포인트에 들어 있다)."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if any(k.startswith("scalemap.") for k in sd):
        sd.setdefault("canon", torch.tensor(False))                    # 앞선 판의 체크포인트에는 없다
        m = build_tr(encoder, ray="scalemap.ray.weight" in sd, canon=bool(sd["canon"]))
        m.load_state_dict(sd)
        return m
    m = build(encoder)
    if any(k.startswith("scale_head.") for k in sd):
        m = Sep(m)
    m.load_state_dict(sd)
    return m


def build_teacher(encoder="vitl", weights=""):
    """선생 = 공식 Depth Anything V2 상대 깊이 모델을 고정한 것 (build_rel 의 설명)."""
    return build_rel(encoder, weights).eval().requires_grad_(False)


def build_rel(encoder="vitl", weights=""):
    """공식 Depth Anything V2 상대 깊이 모델 (학습할 수 있게). 상대 깊이판 dpt.py 와 metric 판의 차이는 마지막 활성 (ReLU 대 Sigmoid) 과 max_depth 곱뿐이라
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
    return m


def rel_forward(model, x, out_hw, amp=False):
    """상대 깊이 모델을 학습할 때: 정규화된 입력 (B, 3, H, W) → 역깊이 꼴 (B, out_h, out_w), float32. teacher_disp 와 같은 크기 규칙인데 기울기가 흐른다."""
    x = F.interpolate(x, net_size(*x.shape[-2:]), mode="bilinear", align_corners=False)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
        d = model(x)
    return F.interpolate(d.float()[:, None], out_hw, mode="bilinear", align_corners=True)[:, 0]


@torch.no_grad()
def teacher_disp(teacher, rgb, amp=False):
    """정규화된 RGB (B, 3, H, W) → 상대 깊이 (B, H, W), float32. 입력 크기 규칙은 학생과 같다 (14 의 배수로 반올림)."""
    x = F.interpolate(rgb, net_size(*rgb.shape[-2:]), mode="bilinear", align_corners=False)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
        d = teacher(x)
    return F.interpolate(d.float()[:, None], rgb.shape[-2:], mode="bilinear", align_corners=True)[:, 0]

