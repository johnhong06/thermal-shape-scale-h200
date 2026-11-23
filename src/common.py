"""공용: 고정 값 (평가 규칙) · 시드 · 깊이 지표.

평가 규칙 (README '판정 기준' — 결과를 보기 전에 고정). 선행 코드 둘과 같은 것: 정답 MIN_DEPTH < gt < MAX_DEPTH 인 픽셀만 · 자르기 없음 ·
  예측을 정답 해상도로 올린 뒤 (bilinear) [MIN_DEPTH, MAX_DEPTH] 로 자름 · 사진마다 지표를 구하고 사진 평균.
  예측 세 가지를 함께 잰다:
    raw    모델 출력 그대로 — MonoTher-Depth 공개 코드 (zoedepth/utils/misc.py compute_metrics) 의 방식. 주 지표.
    median 사진마다 중앙값 맞춤 — MS2 공식 벤치마크 코드 (SupDepth4Thermal eval_metric.py, align=True) 의 방식. 벤치마크 표의 수치와 견줄 때.
    focal  학습 카메라와의 초점거리 비 (fx / FX_TRAIN) 를 곱함 — 카메라가 다른 데이터에서 raw 의 치우침이 초점거리 때문인지 보려고.
  ViViD 만: 정답의 테두리를 뺀다 (border_box) — 선행 논문이 ViViD 를 그렇게 쟀고, 그 표와 견주려는 세트라서.

모양 + 스케일 학생 (팔 sep · ours, README '본 실험 3') 의 손실 = shape_scale. 학생은 모양 zn (log 깊이에서 사진의 중앙값을 뺀 것) 과 스케일 s 를 따로 낸다.
"""
import json
import math
import os
import random

import numpy as np
import torch

MIN_DEPTH = 1e-3          # m — 공식 Depth Anything V2 metric 학습 (train.py --min-depth) 과 같은 값
MAX_DEPTH = 80.0          # m — 평가 상한이자 모델 출력 상한 (Sigmoid × MAX_DEPTH)
MIN_VALID = 10            # 범위 안 정답 픽셀이 이보다 적은 사진은 건너뜀 (공식 train.py 의 평가와 같음)
# 열화상 카메라 초점거리 (px, 가로 640 기준). ms2 = calib.npy 의 K_thrL (20 시퀀스 같음) · fire = 저자 firestereo.yaml cam0 · vivid = calib_ther_to_rgb.yaml 의 K_Thermal
FX = {"ms2": 387.787, "fire": 406.332, "vivid": 437.389, "sthereo": 429.433}      # sthereo = thermal_14bit_left.yaml 의 camera_matrix (영상은 왜곡을 편 것)
FX_TRAIN = FX["ms2"]
# 카메라 값 (fx, fy, cx, cy) px — 묶음의 영상 크기 기준. 열화상 평가 세트 넷 + 쌍 (열화상 시점) + MS2 의 RGB. None = 모름. README '카메라 정보'.
_FR = (1017.8272, 1016.6098, 969.2854, 300.0874)                  # Freiburg Thermal 의 RGB 앞 왼쪽 (1920×650) — 저자 코드 저장소의 front_stereo_calibration.yaml
_FR_BOX, _FR_SIZE = (307, 0, 1702, 650), (640, 298)               # 우리 묶음이 자른 상자 (왼 · 위 · 오른 + 1 · 아래 + 1) 와 줄인 크기
_sx, _sy = _FR_SIZE[0] / (_FR_BOX[2] - _FR_BOX[0]), _FR_SIZE[1] / (_FR_BOX[3] - _FR_BOX[1])
CAMS = {
    "ms2": (FX["ms2"], 387.787, 300.82, 151.626),                              # calib.npy 의 K_thrL (640×256)
    "fire": (FX["fire"], 406.954, 311.512, 241.759),                           # firestereo.yaml cam0 (640×512)
    "vivid": (FX["vivid"], 437.295, 323.528, 256.363),                         # calib_ther_to_rgb.yaml 의 K_Thermal (640×512)
    "sthereo": (FX["sthereo"], 429.531, 311.119, 266.128),                     # thermal_14bit_left.yaml (640×512, 왜곡을 편 영상)
    "kaist": (256.0 / math.tan(math.radians(19.5)),) * 2 + (320.0, 256.0),    # 논문: 맞춘 영상 640×512 의 세로 화각 39° (FLIR A35, 9 mm). 주점은 가운데로 둔다
    "freiburg": (_FR[0] * _sx, _FR[1] * _sy, (_FR[2] - _FR_BOX[0]) * _sx, (_FR[3] - _FR_BOX[1]) * _sy),   # 맞춘 열화상은 RGB 시점. 공개 영상이 왜곡을 편 것인지는 모른다
    "tartan": (421.23237, 420.80872, 317.55166, 255.54589),                    # 공개 저장소 calibration/thermal_left.yaml (640×512, RGB 는 이 시점으로 옮긴 것)
    "llvip": None,                                                              # 논문 · 묶음에 값이 없다
    "ms2_rgb": (764.514, 764.514, 579.547, 153.005),                           # calib.npy 의 K_rgbL (1224×384, 20 시퀀스 같음)
}
F_CANON = FX_TRAIN        # 기준 초점거리 (px) — 깊이를 이 카메라에서 본 것으로 환산해 가르친다 (MS2 는 그대로)
JITTER = tuple(2.0 ** (k / 8.0) for k in range(-4, 5))            # 크기 바꾸기 증강의 배율 아홉 (0.71–1.41, 1.0 포함) — 카메라 값도 같이 바꾼다
# ViViD 후처리본의 보정값 (calib_ther_to_rgb.yaml) — 영상은 왜곡을 편 것이라 K 만 쓴다. X_rgb = R · X_thermal + T (m).
# 두 시점의 LiDAR 깊이 지도로 확인: 이 식으로 열화상 시점의 점을 RGB 로 옮기면 99.8 % 가 RGB 시점 깊이와 맞는다 (2026-11-16).
VIVID_K_T = ((437.38861083256637, 0.0, 323.5284494924228), (0.0, 437.29475745770907, 256.36315482047905), (0.0, 0.0, 1.0))
VIVID_K_RGB = ((531.0787710720505, 0.0, 322.0434384516226), (0.0, 530.9280349280389, 239.58555406488455), (0.0, 0.0, 1.0))
VIVID_R = ((0.9999117397341355, 0.002936303473050537, 0.012957270691089122), (-0.002938371484737371, 0.999995673097488, 0.00014056782332314203),
           (-0.012956801876454241, -0.00017862869148840368, 0.9999160411639161))
VIVID_T = (0.02284737993266349, -0.04897547901167786, 0.033484737180487244)
PL_TRIM = 0.2             # 의사 라벨 손실에서 사진마다 버리는 큰 오차의 비율 (MiDaS 의 trimmed 손실과 같은 값)
PL_DARK = 0.08            # RGB 밝기 (0–1, 7×7 평균) 가 이보다 낮은 픽셀은 의사 라벨에서 뺀다 — 선생 (RGB 깊이 모델) 이 볼 수 없는 곳
PL_MIN, PL_MAX = 0.1, 200.0   # 쌍의 라벨로 쓰는 선생 (UniDepthV2) 깊이의 범위 (m) — 밖의 화소는 뺀다
S0 = 2.72                 # 스케일 헤드의 시작값 = MS2 학습 정답의 사진별 중앙 깊이 (log) 의 중앙값 — 학습 묶음에서 같은 간격으로 뽑은 608 장에서 잰 값 (15.2 m)
PL_Z0 = 15.0              # ViViD: RGB 시점의 선생 깊이를 열화상 시점으로 옮길 때 가정하는 거리 (m). 두 카메라가 6 cm 떨어져 있어 5 m 에서 약 4 픽셀, 10 m 밖은 1 픽셀 안의 어긋남
METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3")
VARIANTS = ("raw", "focal", "median")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def depth_metrics(pred, gt):
    """1 차원 텐서 (범위 안 픽셀) → 지표 dict (float). 흔히 쓰는 정의 (Eigen 등): 사진 한 장의 픽셀 평균."""
    pred = pred.clamp(min=1e-6)
    thr = torch.maximum(pred / gt, gt / pred)
    d = pred - gt
    dl = torch.log(pred) - torch.log(gt)
    return {"abs_rel": (d.abs() / gt).mean().item(), "sq_rel": (d * d / gt).mean().item(),
            "rmse": (d * d).mean().sqrt().item(), "rmse_log": (dl * dl).mean().sqrt().item(),
            "d1": (thr < 1.25).double().mean().item(), "d2": (thr < 1.25 ** 2).double().mean().item(), "d3": (thr < 1.25 ** 3).double().mean().item()}


def variants(pred, gt, mask, fx):
    """예측 (H, W) · 정답 (H, W) · 평가 마스크 → {raw · focal · median: 지표 dict}. 마스크 픽셀이 MIN_VALID 미만이면 None."""
    if int(mask.sum()) < MIN_VALID:
        return None
    p, g = pred[mask].double(), gt[mask].double()
    s = (g.median() / p.median().clamp(min=1e-6)).item()
    out = {k: depth_metrics((p * f).clamp(MIN_DEPTH, MAX_DEPTH), g) for k, f in (("raw", 1.0), ("focal", fx / FX_TRAIN), ("median", s))}
    out["median"]["scale"] = s
    return out


def eval_mask(gt, border=False):
    m = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    if border:
        top, bottom, left, right = border_box(gt)
        keep = torch.zeros_like(m)
        keep[top:bottom, left:right] = True
        m &= keep
    return m


def border_box(gt, tol=0.4, cut_off=100, eps=0.01, min_border=40):
    """ViViD 평가에서 빼는 테두리 — MonoTher-Depth 공개 코드가 ViViD 정답에 거는 규칙을 다시 쓴 것 (그 표와 같은 조건으로 재려고).
    (zoedepth/data/data_thermal_rgbtd.py: get_black_border(depth_gt, tolerance=0.4, cut_off=100, level_diff_threshold=0.01, min_border=40))
    변마다 min_border 번째 줄에서 시작해, 그 줄에서 값 없는 픽셀 (|gt| < eps) 의 비율이 tol 을 넘는 동안 안쪽으로 한 줄씩 — cut_off 를 넘으면 멈춤.
    → (top, bottom, left, right): gt[top:bottom, left:right] 만 평가에 쓴다."""
    h, w = gt.shape
    empty = gt.abs() < eps
    rows, cols = empty.float().mean(1).tolist(), empty.float().mean(0).tolist()
    top = min_border
    while rows[top] > tol and top < h - 1:
        top += 1
        if top > cut_off:
            break
    bottom = h - min_border
    while rows[bottom] > tol and bottom > 0:
        bottom -= 1
        if h - bottom > cut_off:
            break
    left = min_border
    while cols[left] > tol and left < w - 1:
        left += 1
        if left > cut_off:
            break
    right = w - min_border
    while cols[right] > tol and right > 0:
        right -= 1
        if w - right > cut_off:
            break
    return top, bottom, left, right


def silog(pred, gt, mask, lambd=0.5):
    """공식 Depth Anything V2 metric_depth/util/loss.py 의 SiLogLoss 와 같은 식."""
    dl = torch.log(gt[mask]) - torch.log(pred[mask])
    return torch.sqrt(dl.pow(2).mean() - lambd * dl.mean().pow(2))


def ssi_trim(pred, disp, mask, trim=PL_TRIM):
    """의사 라벨 손실 — 축척 · 이동에 무관한 (scale-and-shift-invariant) 절대 오차, 큰 오차 일부를 버림 (MiDaS, Ranftl et al. 2020 의 방식을 따름).
    pred = 학생의 깊이 (B, H, W) m · disp = 선생의 상대 깊이 (역깊이 꼴, 축척 · 이동 미정) · mask = 쓸 픽셀.
    사진마다: 학생의 역깊이 1/pred 와 선생 값을 각각 (값 − 중앙값) / 평균 절대 편차 로 맞춘 뒤 |차| 를 구하고, 큰 쪽 trim 비율을 버린 평균. 쓸 픽셀이 100 개 미만인 사진은 건너뛴다."""
    def norm(x):
        x = x - x.median()
        return x / x.abs().mean().clamp(min=1e-6)
    total, n = pred.sum() * 0.0, 0
    for b in range(pred.shape[0]):
        m = mask[b]
        if int(m.sum()) < 100:
            continue
        r = (norm(1.0 / pred[b][m].clamp(min=0.1)) - norm(disp[b][m].float())).abs()
        total = total + r.topk(max(1, int(r.numel() * (1 - trim))), largest=False).values.mean()
        n += 1
    return total / max(n, 1)


def shape_scale(zn, s, logt, mask, trim=0.0, min_px=MIN_VALID):
    """모양 + 스케일 학생의 손실 — 사진마다 따로 구해 사진 평균. → (모양 손실, 스케일 손실, 쓴 사진 수)
    zn = 학생의 모양 (B, H, W): log 깊이에서 스케일을 뺀 것 · s = 학생의 스케일 (B,): 그 장면의 중앙 깊이의 log · logt = 목표의 log 깊이 (B, H, W) · mask = 쓸 화소.
    목표가 정답 (LiDAR) 일 때 trim = 0: 모양 = (logt − zn) 의 표준편차 — SiLog 의 λ = 1 꼴이라 스케일에 무관하다.
    목표가 선생의 라벨일 때 trim > 0: 모양 = (logt − zn) 에서 중앙값을 뺀 |차| 를 큰 쪽 trim 비율만큼 버리고 평균 (선생이 크게 틀린 화소를 버린다).
    스케일 = |s − 중앙값 (logt − zn)| — 목표 쪽 (중앙값) 으로는 기울기를 흘리지 않는다. 쓸 화소가 min_px 미만인 사진은 건너뛴다."""
    shape, scale, n = zn.sum() * 0.0, zn.sum() * 0.0, 0
    for b in range(zn.shape[0]):
        m = mask[b]
        if int(m.sum()) < min_px:
            continue
        d = logt[b][m].float() - zn[b][m].float()
        med = d.median()
        if trim > 0:
            r = (d - med).abs()
            shape = shape + r.topk(max(1, int(r.numel() * (1 - trim))), largest=False).values.mean()
        else:
            shape = shape + torch.sqrt((d - d.mean()).pow(2).mean() + 1e-12)
        scale = scale + (s[b].float() - med.detach()).abs()
        n += 1
    return shape / max(n, 1), scale / max(n, 1), n


TP_THR = 0.99            # 열화상 판 TR2M 의 의사 정답: 상대 깊이를 정답에 맞춘 깊이의 δ1 이 이보다 큰 사진만 쓴다 (TR2M 공개 코드 SelfThresholdScaleLoss 의 값)


def silog_px(pred, gt, mask, min_px=100, lambd=0.5):
    """silog 와 같은 식인데, 쓸 화소가 min_px 미만이면 0 (기울기는 이어진다). 쌍의 선생 라벨용 — 어두운 쌍에서는 쓸 화소가 거의 없을 수 있다."""
    if int(mask.sum()) < min_px:
        return pred.sum() * 0.0
    return silog(pred, gt, mask, lambd)


def ssi_disp(pred, gt, mask, trim=0.0, min_px=MIN_VALID):
    """상대 깊이 모델 학습의 손실 — 역깊이 꼴 예측 pred (B, H, W) 와 정답 gt (B, H, W, m) 의 역깊이 1 / gt 를 사진마다 (값 − 중앙값) / 평균 절대 편차 로
    맞춘 뒤의 절대 오차 평균 (MiDaS 의 스케일 · 기준점 무관 손실, ssi_trim 과 같은 맞춤). trim > 0 이면 큰 오차 그 비율을 버린다. 쓸 화소가 min_px 미만인 사진은 건너뛴다."""
    def norm(x):
        x = x - x.median()
        return x / x.abs().mean().clamp(min=1e-6)
    total, n = pred.sum() * 0.0, 0
    for b in range(pred.shape[0]):
        m = mask[b]
        if int(m.sum()) < min_px:
            continue
        r = (norm(pred[b][m].float()) - norm(1.0 / gt[b][m].float().clamp(min=MIN_DEPTH))).abs()
        if trim > 0:
            r = r.topk(max(1, int(r.numel() * (1 - trim))), largest=False).values
        total = total + r.mean()
        n += 1
    return total / max(n, 1)


def fit_inv(dr, gt, mask):
    """사진마다 s · dr + t ≈ 1 / gt 를 마스크 화소에서 최소제곱으로 (MiDaS 의 compute_scale_and_shift 와 같은 닫힌 꼴). dr · gt · mask = (B, H, W) → (s, t) 각 (B,), double.
    쓸 화소가 둘 미만이거나 dr 이 한 값뿐이면 그 사진은 (0, 0)."""
    m, p = mask.double(), dr.double()
    y = torch.where(mask, 1.0 / gt.double().clamp(min=MIN_DEPTH), torch.zeros_like(p))
    a00, a01, a11 = (m * p * p).sum((1, 2)), (m * p).sum((1, 2)), m.sum((1, 2))
    b0, b1 = (m * p * y).sum((1, 2)), (m * y).sum((1, 2))
    det = a00 * a11 - a01 * a01
    ok = det.abs() > 1e-12
    s = torch.where(ok, (a11 * b0 - a01 * b1) / torch.where(ok, det, torch.ones_like(det)), torch.zeros_like(det))
    t = torch.where(ok, (-a01 * b0 + a00 * b1) / torch.where(ok, det, torch.ones_like(det)), torch.zeros_like(det))
    return s, t


def tr_pseudo(a, b, dr, gt, mask, thr=TP_THR, dmax=None):
    """TR2M 의 의사 정답 손실 (공개 코드 SelfThresholdScaleLoss 를 옮김). 사진마다 상대 깊이 dr 을 역깊이 정답에 맞추고 (s, t), 맞춘 깊이 1 / (s · dr + t) 가
    정답 화소의 thr 넘게 δ1 (1.25 안) 에 들면 그 사진의 scale 맵 a · shift 맵 b 를 상수 s · t 쪽으로 (L1, 화소 평균). 정답이 드문 화소까지 맵을 가르치려는 것.
    a · b = (B, h, w) 맵 · dr · gt · mask = (B, H, W) 정답 해상도. → (손실, 쓴 사진 수). 쓴 사진이 없으면 0 (기울기는 이어진다).
    dmax (B,) = 맞춘 깊이의 상한 — 정답을 기준 초점의 깊이로 바꿔 줄 때 80 m 도 같이 바꾼 값 (안 주면 MAX_DEPTH)."""
    with torch.no_grad():
        s, t = fit_inv(dr, gt, mask)
        den = s[:, None, None] * dr.double() + t[:, None, None]
        d = 1.0 / (den.clamp(min=1.0 / MAX_DEPTH) if dmax is None else torch.maximum(den, (1.0 / dmax.double())[:, None, None]))
        r = d / gt.double().clamp(min=MIN_DEPTH)
        good = ((r < 1.25) & (1.0 / r < 1.25) & mask).double().sum((1, 2)) / mask.double().sum((1, 2)).clamp(min=1.0)
        use = (good > thr) & (s > 0)
    if not bool(use.any()):
        return (a.sum() + b.sum()) * 0.0, 0
    loss = (a[use] - s[use, None, None].to(a.dtype)).abs().mean() + (b[use] - t[use, None, None].to(b.dtype)).abs().mean()
    return loss, int(use.sum())


def edge_smooth(m, img):
    """경계를 아는 매끄러움 손실 — TR2M 공개 코드의 EdgeAwareSmoothLoss 와 같은 식 (맵을 평균으로 나누지 않는다). 맵 m (B, h, w) 의 이웃 차를
    영상 img (B, C, h, w, 망에 넣는 정규화된 입력) 의 이웃 차가 큰 곳에서 덜 벌한다. 가로 · 세로 평균의 합."""
    m = m[:, None]
    gx, gy = (m[..., :, :-1] - m[..., :, 1:]).abs(), (m[..., :-1, :] - m[..., 1:, :]).abs()
    ix = (img[..., :, :-1] - img[..., :, 1:]).abs().mean(1, keepdim=True)
    iy = (img[..., :-1, :] - img[..., 1:, :]).abs().mean(1, keepdim=True)
    return (gx * torch.exp(-ix)).mean() + (gy * torch.exp(-iy)).mean()


def vivid_grid(h=512, w=640, hr=480, wr=640, z0=PL_Z0):
    """ViViD: 열화상 픽셀 (h × w) 마다 RGB 영상 (hr × wr) 에서 볼 자리 → (grid_sample 용 격자 (1, h, w, 2), RGB 화면 안인지 (h, w)).
    열화상 픽셀의 시선에 거리 z0 의 점을 놓고 X_rgb = R · X + T 로 옮겨 RGB 에 투영한다."""
    kt, kr = torch.tensor(VIVID_K_T, dtype=torch.float64), torch.tensor(VIVID_K_RGB, dtype=torch.float64)
    rot, tr = torch.tensor(VIVID_R, dtype=torch.float64), torch.tensor(VIVID_T, dtype=torch.float64)
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    x = torch.stack([(u - kt[0, 2]) / kt[0, 0] * z0, (v - kt[1, 2]) / kt[1, 1] * z0, torch.full_like(u, z0)], -1) @ rot.T + tr
    ur, vr = kr[0, 0] * x[..., 0] / x[..., 2] + kr[0, 2], kr[1, 1] * x[..., 1] / x[..., 2] + kr[1, 2]
    inside = (ur >= 0) & (ur <= wr - 1) & (vr >= 0) & (vr <= hr - 1)
    grid = torch.stack([ur / (wr - 1) * 2 - 1, vr / (hr - 1) * 2 - 1], -1)[None].float()
    return grid, inside


def cam_resize(cam, rx, ry):
    """영상을 가로 rx · 세로 ry 배로 바꿨을 때의 카메라 값. cam (..., 4) = fx · fy · cx · cy (화소 가운데가 정수 좌표)."""
    return torch.stack([cam[..., 0] * rx, cam[..., 1] * ry, (cam[..., 2] + 0.5) * rx - 0.5, (cam[..., 3] + 0.5) * ry - 0.5], -1)


def cam_flip(cam, w):
    """영상을 좌우로 뒤집었을 때의 카메라 값 (가로 w 화소)."""
    return torch.stack([cam[..., 0], cam[..., 1], (w - 1) - cam[..., 2], cam[..., 3]], -1)


def canon_scale(cam):
    """기준 초점으로의 환산 배율 (...,): 기준 깊이 = 실제 깊이 × 이 값. 같은 모습이면 초점이 긴 카메라일수록 실제로는 멀다 — 그 몫을 뺀다."""
    return F_CANON / (cam[..., 0] * cam[..., 1]).sqrt()
