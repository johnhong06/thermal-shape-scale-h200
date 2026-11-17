"""선생 점검 (run.sh teacher) — RGB 를 보는 선생 둘이 LiDAR 정답과 얼마나 맞는지를 MS2 의 RGB 에서 잰다. 학습 없음. 판정 기준은 README '선생 점검'.

선생: Depth Anything V2 상대 깊이 (지금 pl · plv 팔이 쓰는 것 — 모양만 준다) · UniDepthV2 (m 단위 — 스케일을 줄 후보이자 모양 후보).
  표1 스케일 · 모양   UniDepthV2 를 그대로 쓴 오차, 사진별 스케일 오차 |log (정답 중앙값 / 예측 중앙값)|, 그리고 두 선생의 모양 오차
                     (모양 = 사진마다 역깊이에서 a · x + b 를 정답에 최소제곱으로 맞춘 뒤의 오차 — 두 선생을 같은 식으로)
  표2 픽셀 신뢰도     정답 없이 구하는 단서 (밝기 · 좌우 뒤집기 일관성 · 두 선생의 어긋남 · UniDepthV2 의 신뢰도 출력 · 거리 · 경계) 가 모양 오차를 가려내는가
  표3 영상 신뢰도     같은 단서의 사진별 요약이 스케일 오차를 가려내는가
판정은 val 을 밝은 사진 · 어두운 사진으로 나눠 따로 내고 나쁜 쪽을 따른다 (val 1,425 장 가운데 밤은 한 시퀀스 254 장뿐이고, 쌍 데이터는 밤이 많다).
단서의 방향 · 합치는 식 (릿지 회귀) 은 train 시퀀스에서만 맞추고, 수치는 val 시퀀스 (다른 주행) 에서 잰다. MS2 의 test 분할은 쓰지 않는다 (학생의 평가 세트).
출력: <out>/per_image.csv · pixels.npz (뽑은 픽셀의 단서 · 오차) · summary.json, 화면에 표 셋과 판정.

사용: python src/teacher_check.py --data <푼 폴더> --out <폴더> [--encoder vitl|vits] [--weights <DAv2 상대 깊이 .pth>] [--limit N] [--amp 0]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M  # noqa: E402
from common import MAX_DEPTH, MIN_DEPTH, depth_metrics, seed_all, write_json  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
UNIDEPTH = {"vitl": ("lpiccinelli/unidepth-v2-vitl14", "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9"),
            "vits": ("lpiccinelli/unidepth-v2-vits14", "038c238f06c87b6c2f5b3749fd51fbf442b1f218")}     # vits = 로컬 흐름 시험용
PIX = 512                 # 사진마다 뽑는 정답 픽셀 수 (신뢰도 분석용)
DARK = 0.25               # 사진 평균 밝기 (0–1) 가 이보다 낮으면 '어두움' (밤) 으로 나눠 적는다
CUES = ("lum", "flip_da", "flip_ud", "disagree", "ud_conf", "logz", "edge")
IMG_CUES = ("lum", "flip_ud", "disagree", "ud_conf")     # UniDepthV2 가 예측한 초점거리 (fx_rel) 는 넣지 않는다 — MS2 는 카메라가 하나라 그 값을 외울 뿐이다 (per_image.csv 에는 남긴다)
BASE_CUES = ("lum", "logz")                              # 픽셀 신뢰도가 넘어야 할 단순한 단서: 밝기 · 거리
MIN_GROUP = 30                                           # 밝음 · 어두움 묶음을 따로 판정하는 데 필요한 사진 수
# 판정 기준 (README '선생 점검' — 결과를 보기 전에 고정)
SCALE_OK, SCALE_WEAK = 0.10, 0.20          # 스케일 오차 중앙값: 이하 = 쓸 만함 / 이하 = 신뢰도와 함께만 / 넘으면 못 씀
SHAPE_GAP = 0.005                          # 두 선생의 모양 AbsRel 차이가 이보다 작으면 '차이 없음' (지금 쓰는 Depth Anything V2 유지)
RHO_OK, KEEP_OK, RHO_GAIN = 0.30, 0.80, 0.05   # 신뢰도: 순위 상관 · 믿는 절반의 오차 비 · (픽셀) 밝기 · 거리 중 나은 단서보다 나아진 정도


def fit_ss(x, y):
    """a · x + b ≈ y 의 최소제곱 (1 차원 double 텐서) → (a, b)."""
    xc = x - x.mean()
    a = (xc * (y - y.mean())).sum() / (xc * xc).sum().clamp(min=1e-12)
    return a, y.mean() - a * x.mean()


def aligned(inv, gt, m):
    """역깊이 꼴의 예측 (H, W) 을 정답에 맞춘 깊이 (H, W): 범위 안 픽셀에서 a · inv + b ≈ 1 / gt."""
    a, b = fit_ss(inv[m].double(), 1.0 / gt[m].double())
    return (1.0 / (a * inv.double() + b).clamp(min=1.0 / MAX_DEPTH)).clamp(MIN_DEPTH, MAX_DEPTH)


def rank(x):
    r = np.empty(len(x), dtype=np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def keep_half(score, err, agg=np.mean):
    """score 가 낮은 (= 믿는) 절반의 오차 / 전체 오차. 1 보다 작을수록 잘 가려낸다."""
    k = max(1, len(err) // 2)
    return float(agg(err[np.argsort(score, kind="stable")[:k]]) / max(agg(err), 1e-12))


def ridge_fit(X, y, lam=1.0):
    """표준화한 단서와 그 제곱 → log 오차 의 릿지 회귀. 돌려주는 것 = 예측 함수 (값이 클수록 오차가 클 것으로 봄)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9

    def feats(Z):
        Z = np.clip((Z - mu) / sd, -5, 5)
        return np.concatenate([Z, Z * Z, np.ones((len(Z), 1))], 1)
    A = feats(X)
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return lambda Z: feats(Z) @ w


def sign_fit(x, err):
    """단서 하나의 방향: train 에서 오차와 같은 쪽으로 움직이면 +1, 반대면 −1."""
    r = spearman(x, err)
    return 1.0 if not np.isfinite(r) or r >= 0 else -1.0


def groups(idx_of):
    """사진 번호 배열 → 사진별 인덱스 묶음."""
    order = np.argsort(idx_of, kind="stable")
    cut = np.flatnonzero(np.diff(idx_of[order])) + 1
    return np.split(order, cut)


def per_image(score, err, img):
    """사진마다 (순위 상관, 믿는 절반의 오차 비) 를 구해 사진 평균."""
    rho, keep = [], []
    for g in groups(img):
        if len(g) >= 50:
            rho.append(spearman(score[g], err[g]))
            keep.append(keep_half(score[g], err[g]))
    rho = [r for r in rho if np.isfinite(r)]
    return (float(np.mean(rho)) if rho else float("nan")), (float(np.mean(keep)) if keep else float("nan"))


def frames(root, limit):
    """푼 폴더에서 (분할, 시퀀스, 이름) 목록. 분할 = 저장소의 공식 목록 (splits/ms2) 으로 정한다."""
    out = []
    for split in ("train", "val"):
        seqs = [s.strip() for s in open(os.path.join(REPO, "splits", "ms2", f"{split}_list.txt")) if s.strip()]
        rows = []
        for s in seqs:
            d = os.path.join(root, "sync_data", s, "rgb", "img_left")
            if os.path.isdir(d):
                rows += [(split, s, n) for n in sorted(os.listdir(d)) if n.endswith(".png")]
        if limit and len(rows) > limit:
            rows = [rows[round(j * (len(rows) - 1) / max(limit - 1, 1))] for j in range(limit)]
        out += rows
    return out


def build_unidepth(size, dev):
    code = os.environ["TD_UNIDEPTH"]
    if code not in sys.path:
        sys.path.insert(0, code)
    import warnings
    warnings.filterwarnings("ignore", message=".*resolution_level.*")
    from unidepth.models import UniDepthV2
    repo, rev = UNIDEPTH[size]
    m = UniDepthV2.from_pretrained(repo, revision=rev)
    m.interpolation_mode = "bilinear"            # 공식 예시 (README) 와 같은 길: 카메라 정보를 주지 않는다
    return m.to(dev).eval()


@torch.no_grad()
def run_teachers(da, ud, rgb8, dev, amp):
    """RGB (H, W, 3) uint8 → 두 선생의 출력과 좌우 뒤집어 다시 낸 출력 (모두 (H, W), float32)."""
    x8 = torch.from_numpy(rgb8).permute(2, 0, 1).to(dev)
    x = ((x8.float() / 255.0) - x8.new_tensor(MEAN, dtype=torch.float32)[:, None, None]) / x8.new_tensor(STD, dtype=torch.float32)[:, None, None]
    xb = torch.stack([x, x.flip(-1)])
    disp = M.teacher_disp(da, xb, amp)
    o, of = ud.infer(x8), ud.infer(x8.flip(-1))
    lum = F.avg_pool2d((x8.float() / 255.0).mean(0)[None, None], 7, 1, 3, count_include_pad=False)[0, 0]
    return {"disp": disp[0], "disp_f": disp[1].flip(-1), "ud": o["depth"][0, 0].float(), "ud_f": of["depth"][0, 0].float().flip(-1),
            "conf": o["confidence"][0, 0].float(), "fx": float(o["intrinsics"][0, 0, 0]), "lum": lum}


def one(t, gt, gen):
    """사진 한 장: 지표 (dict) 와 뽑은 픽셀의 단서 · 오차 ((PIX, len(CUES) + 2) 배열). 범위 안 정답이 적으면 None."""
    m = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
    if int(m.sum()) < 500:
        return None
    g = gt[m].double()
    ud = t["ud"].clamp(MIN_DEPTH, MAX_DEPTH)
    raw = depth_metrics(ud[m].double(), g)
    log_r = float(torch.log(g.median() / ud[m].double().median()))
    med = depth_metrics((ud[m].double() * g.median() / ud[m].double().median()).clamp(MIN_DEPTH, MAX_DEPTH), g)
    da_al, ud_al = aligned(t["disp"], gt, m), aligned(1.0 / ud, gt, m)
    sh_da, sh_ud = depth_metrics(da_al[m], g), depth_metrics(ud_al[m], g)
    # 정답 없이 구하는 단서
    eps = 1e-6
    flip_da = (t["disp"] - t["disp_f"]).abs() / (t["disp"] + t["disp_f"] + eps) * 2.0
    flip_ud = (torch.log(ud) - torch.log(t["ud_f"].clamp(MIN_DEPTH, MAX_DEPTH))).abs()
    a, b = fit_ss(t["disp"].double().flatten(), (1.0 / ud).double().flatten())            # Depth Anything 의 모양을 UniDepthV2 의 값에 맞춤 (정답 없음)
    da_as_ud = (1.0 / (a * t["disp"].double() + b).clamp(min=1.0 / MAX_DEPTH)).clamp(MIN_DEPTH, MAX_DEPTH)
    disagree = (torch.log(da_as_ud) - torch.log(ud.double())).abs().float()
    lz = torch.log(ud)
    edge = torch.zeros_like(lz)
    edge[:, 1:-1] += (lz[:, 2:] - lz[:, :-2]).abs()
    edge[1:-1, :] += (lz[2:, :] - lz[:-2, :]).abs()
    conf = torch.log(t["conf"].clamp(min=1e-8))                                           # UniDepthV2 의 신뢰도 출력은 지수 꼴이라 log 로 본다
    cue = {"lum": t["lum"], "flip_da": flip_da, "flip_ud": flip_ud, "disagree": disagree, "ud_conf": conf, "logz": lz, "edge": edge}
    err_da = (torch.log(da_al) - torch.log(gt.double().clamp(min=MIN_DEPTH))).abs().float()
    err_ud = (torch.log(ud_al) - torch.log(gt.double().clamp(min=MIN_DEPTH))).abs().float()
    idx = m.flatten().nonzero()[:, 0]
    pick = idx[torch.randperm(len(idx), generator=gen)[:PIX].to(idx.device)]
    px = torch.stack([cue[c].flatten()[pick] for c in CUES] + [err_da.flatten()[pick], err_ud.flatten()[pick]], 1).cpu().numpy()
    w = gt.shape[1]
    row = {"valid": int(m.sum()), "lum": float(t["lum"].mean()), "ud_absrel": raw["abs_rel"], "ud_d1": raw["d1"], "log_r": log_r, "ud_med_absrel": med["abs_rel"],
           "da_shape_absrel": sh_da["abs_rel"], "ud_shape_absrel": sh_ud["abs_rel"], "da_shape_d1": sh_da["d1"], "ud_shape_d1": sh_ud["d1"],
           "flip_ud": float(flip_ud.median()), "disagree": float(disagree.median()), "ud_conf": float(conf.mean()), "fx_rel": t["fx"] / w}
    return row, px


def center(X, img):
    """사진마다 평균을 뺀다 — 사진 안의 순위를 맞추는 데는 사진 사이의 차이가 방해가 된다."""
    X = np.array(X, dtype=np.float64)
    for g in groups(img):
        X[g] -= X[g].mean(0)
    return X


def worst(labels, order):
    return max(labels, key=order.index) if labels else None


def tables(rows, px, img_of):
    """표 셋과 판정 → (화면 줄 목록, summary dict). rows = 사진별 dict 목록, px = (N, len(CUES) + 2), img_of = 픽셀의 사진 번호.
    판정은 val 을 밝은 사진 · 어두운 사진으로 나눠 따로 내고 나쁜 쪽을 따른다 (val 의 대부분이 낮이라 한 덩어리로 보면 밤이 묻힌다)."""
    L, S = [], {}
    A = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0] if k not in ("split", "seq", "name")}
    split, seq = np.array([r["split"] for r in rows]), np.array([r["seq"] for r in rows])
    val, tr_m = split == "val", split == "train"
    G = {"전체": val, "밝음": val & (A["lum"] >= DARK), "어두움": val & (A["lum"] < DARK)}
    judged = [g for g in ("밝음", "어두움") if G[g].sum() >= MIN_GROUP] or ["전체"]      # 사진이 너무 적은 쪽은 판정에서 뺀다 (둘 다 적으면 전체로)
    # ── 표1
    L.append("[표1] 스케일 · 모양 (사진 평균)   n | UniDepthV2 그대로 AbsRel · δ1 | 스케일 오차 중앙값 · 90 % · 치우침 · 흩어짐 | 모양 AbsRel: DAv2 · UniDepthV2 | UniDepthV2 중앙값 맞춤")
    for name, sel in [(f"val {g}", m) for g, m in G.items()] + [("train 전체", tr_m)]:
        if not sel.any():
            continue
        lr = A["log_r"][sel]
        d = {"n": int(sel.sum()), "ud_absrel": A["ud_absrel"][sel].mean(), "ud_d1": A["ud_d1"][sel].mean(), "scale_med": np.median(np.abs(lr)), "scale_p90": np.percentile(np.abs(lr), 90),
             "scale_bias": np.median(lr), "scale_spread": np.median(np.abs(lr - np.median(lr))), "da_shape": A["da_shape_absrel"][sel].mean(), "ud_shape": A["ud_shape_absrel"][sel].mean(),
             "ud_med": A["ud_med_absrel"][sel].mean()}
        S[name] = {k: float(v) for k, v in d.items()}
        L.append(f"[표1] {name:9s} {d['n']:5d} | {d['ud_absrel']:.4f} · {d['ud_d1']:.4f} | {d['scale_med']:.4f} · {d['scale_p90']:.4f} · {d['scale_bias']:+.4f} · {d['scale_spread']:.4f} | "
                 f"{d['da_shape']:.4f} · {d['ud_shape']:.4f} | {d['ud_med']:.4f}")
    per_seq = [(q, np.median(np.abs(A["log_r"][val & (seq == q)])), int((val & (seq == q)).sum())) for q in sorted(set(seq[val]))]
    S["val_per_seq_scale"] = {q: float(v) for q, v, _ in per_seq}
    L.append("[표1] val 시퀀스별 스케일 오차 중앙값 (장 수): " + " · ".join(f"{q[1:]} {v:.4f} ({n})" for q, v, n in per_seq))
    tr_img = np.flatnonzero(tr_m)
    if len(tr_img) < 8 or val.sum() < 8:
        L.append("[판정] 사진이 너무 적어 신뢰도 표는 건너뜀")
        return L, S
    # ── 표2: 픽셀 신뢰도 — 단서의 방향 · 릿지는 train 의 '사진 안' 관계로 맞춘다 (재는 것도 사진 안의 순위라서)
    tr = np.isin(img_of, tr_img)
    X = px[:, :len(CUES)].astype(np.float64)
    Xc = center(X, img_of)
    pix_of = {g: np.isin(img_of, np.flatnonzero(m)) for g, m in G.items()}
    L.append("[표2] 픽셀 신뢰도 (val, 사진 평균)   단서 | 모양 오차와의 순위 상관 · 믿는 절반의 오차 비 (평균) — DAv2 | UniDepthV2")
    S["pixel"] = {}
    for j, c in enumerate(list(CUES) + ["모두 (릿지)"]):
        score = {}
        for k, tgt in enumerate(("da", "ud")):
            e = px[:, len(CUES) + k].astype(np.float64)
            if j < len(CUES):
                r_tr = per_image(X[tr, j], e[tr], img_of[tr])[0]
                score[tgt] = (1.0 if not np.isfinite(r_tr) or r_tr >= 0 else -1.0) * X[:, j]
            else:
                score[tgt] = ridge_fit(Xc[tr], center(np.log(e + 1e-3)[:, None], img_of)[tr, 0])(Xc)
            for g, sel in pix_of.items():
                if sel.any():
                    rho, keep = per_image(score[tgt][sel], e[sel], img_of[sel])
                    S["pixel"][f"{c}/{tgt}/{g}"] = {"rho": rho, "keep": keep}
        for g in (["전체"] if j < len(CUES) else list(G)):
            cells = [f"{S['pixel'][f'{c}/{t}/{g}']['rho']:+.3f} · {S['pixel'][f'{c}/{t}/{g}']['keep']:.3f}" if f"{c}/{t}/{g}" in S["pixel"] else "—" for t in ("da", "ud")]
            L.append(f"[표2] {c + ('' if j < len(CUES) else ' ' + g):14s} | {cells[0]} | {cells[1]}")
    # ── 표3: 영상 신뢰도 (스케일) — 밝은 사진끼리 · 어두운 사진끼리의 순위도 본다 (낮과 밤을 가르는 것만으로 통과하지 않게)
    s_err = np.abs(A["log_r"])
    Xi = np.stack([A[c] for c in IMG_CUES], 1)
    L.append("[표3] 영상 신뢰도 (val)   단서 | 스케일 오차와의 순위 상관 · 믿는 절반의 스케일 오차 비 (중앙값) — 전체 | 밝은 사진끼리 | 어두운 사진끼리")
    S["image"] = {}
    for j, c in enumerate(list(IMG_CUES) + ["모두 (릿지)"]):
        if j < len(IMG_CUES):
            score = sign_fit(Xi[tr_img, j], s_err[tr_img]) * Xi[:, j]
        else:
            score = ridge_fit(Xi[tr_img], np.log(s_err[tr_img] + 1e-3))(Xi)
        cells = []
        for g, m in G.items():
            ii = np.flatnonzero(m)
            if len(ii) < 8:
                cells.append("—")
                continue
            rho, keep = spearman(score[ii], s_err[ii]), keep_half(score[ii], s_err[ii], np.median)
            S["image"][f"{c}/{g}"] = {"rho": rho, "keep": keep}
            cells.append(f"{rho:+.3f} · {keep:.3f}")
        L.append(f"[표3] {c:12s} | " + " | ".join(cells))
    # ── 판정 (기준은 README '선생 점검')
    V = S["verdict"] = {"judged_groups": judged}
    names = ("쓸 만함", "신뢰도와 함께만", "못 씀")
    sc = {g: names[0] if S[f"val {g}"]["scale_med"] <= SCALE_OK else names[1] if S[f"val {g}"]["scale_med"] <= SCALE_WEAK else names[2] for g in judged}
    V["scale"] = worst(list(sc.values()), names)
    gaps = {g: S[f"val {g}"]["ud_shape"] - S[f"val {g}"]["da_shape"] for g in judged}
    V["shape"] = ("UniDepthV2" if all(v < -SHAPE_GAP for v in gaps.values()) else "Depth Anything V2" if all(v > SHAPE_GAP for v in gaps.values())
                  else "차이 없음 (Depth Anything V2 유지)" if all(abs(v) <= SHAPE_GAP for v in gaps.values()) else "조건에 따라 다름 (Depth Anything V2 유지)")
    tgt = "ud" if V["shape"] == "UniDepthV2" else "da"
    pix = {}
    for g in judged:
        p = S["pixel"][f"모두 (릿지)/{tgt}/{g}"]
        base = max([0.0] + [r for r in (S["pixel"][f"{b}/{tgt}/{g}"]["rho"] for b in BASE_CUES) if np.isfinite(r)])   # 순위를 못 매기는 단서 (사진 안에서 값이 같음) 는 0 으로 본다
        pix[g] = (p["rho"], p["keep"], p["rho"] - base, p["rho"] >= RHO_OK and p["keep"] <= KEEP_OK and p["rho"] - base >= RHO_GAIN)
    V["pixel_conf"] = "쓸 만함" if all(v[3] for v in pix.values()) else "단순 규칙 유지"
    img = {g: (S["image"][f"모두 (릿지)/{g}"]["rho"], S["image"][f"모두 (릿지)/{g}"]["keep"]) for g in judged}
    V["image_conf"] = "쓸 만함" if all(r >= RHO_OK and k <= KEEP_OK for r, k in img.values()) else "못 씀"
    L.append(f"[판정] 스케일 선생 (UniDepthV2 스케일 오차 중앙값 — " + " · ".join(f"{g} {S[f'val {g}']['scale_med']:.4f} {sc[g]}" for g in judged) + f", 기준 {SCALE_OK} · {SCALE_WEAK}): {V['scale']}")
    L.append("[판정] 모양 선생 (모양 AbsRel: UniDepthV2 − DAv2 — " + " · ".join(f"{g} {v:+.4f}" for g, v in gaps.items()) + f", 기준 ±{SHAPE_GAP}): {V['shape']}")
    L.append(f"[판정] 픽셀 신뢰도 (순위 상관 ≥ {RHO_OK} · 오차 비 ≤ {KEEP_OK} · 밝기 · 거리 중 나은 것보다 ≥ +{RHO_GAIN} — "
             + " · ".join(f"{g} {r:+.3f} / {k:.3f} / {d:+.3f}" for g, (r, k, d, _) in pix.items()) + f"): {V['pixel_conf']}")
    L.append(f"[판정] 영상 신뢰도 (같은 밝기끼리 순위 상관 ≥ {RHO_OK} · 오차 비 ≤ {KEEP_OK} — " + " · ".join(f"{g} {r:+.3f} / {k:.3f}" for g, (r, k) in img.items()) + f"): {V['image_conf']}")
    return L, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="vitl", choices=["vitl", "vits"])
    ap.add_argument("--weights", default="", help="Depth Anything V2 상대 깊이 체크포인트. 비우면 무작위 초기화 (흐름 시험)")
    ap.add_argument("--limit", type=int, default=0, help="분할마다 고르게 뽑을 사진 수 (0 = 전부)")
    ap.add_argument("--amp", type=int, default=0, help="Depth Anything V2 쪽의 bf16 — 두 선생을 같은 정밀도로 견주려고 기본은 끔")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seed_all(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(a.amp) and dev.type == "cuda"
    fr = frames(a.data, a.limit)
    if not fr:
        print("!!! [teacher] 사진이 없음 — 데이터 폴더 확인")
        return 1
    da = M.build_teacher(a.encoder, a.weights).to(dev)
    ud = build_unidepth(a.encoder, dev)
    print(f"[teacher] 사진 {len(fr)} 장 (train {sum(f[0] == 'train' for f in fr)} · val {sum(f[0] == 'val' for f in fr)}) · 선생 Depth Anything V2 {a.encoder}"
          f"{'' if a.weights else ' (무작위 초기화)'} · UniDepthV2 {a.encoder} · {dev.type}", flush=True)
    gen = torch.Generator().manual_seed(0)
    rows, pxs, img_of, t0, mark, bad = [], [], [], time.time(), time.time(), 0
    for i, (split, seq, name) in enumerate(fr):
        rgb8 = np.asarray(Image.open(os.path.join(a.data, "sync_data", seq, "rgb", "img_left", name)).convert("RGB"))
        gt = torch.from_numpy(np.asarray(Image.open(os.path.join(a.data, "proj_depth", seq, "rgb", "depth_filtered", name))).astype(np.float32) / 256.0).to(dev)
        if gt.shape != rgb8.shape[:2]:
            print(f"!!! [teacher] 영상과 정답의 크기가 다름: {seq}/{name}")
            return 1
        r = one(run_teachers(da, ud, rgb8, dev, amp), gt, gen)
        if r is not None:
            row, px = r
            if not np.isfinite(px).all() or not all(np.isfinite(v) for v in row.values()):
                bad += 1                                           # 한 장 때문에 작업 전체를 버리지 않는다 — 건너뛰고 센다
                continue
            rows.append(dict(split=split, seq=seq, name=name, **row))
            pxs.append(px)
            img_of.append(np.full(len(px), len(rows) - 1))
        if time.time() - mark > 300:
            print(f"[teacher] {i + 1} / {len(fr)} 장 · {(time.time() - t0) / 60:.1f} 분", flush=True)
            mark = time.time()
    if not rows:
        print("!!! [teacher] 쓸 수 있는 사진이 없음")
        return 1
    px, img_of = np.concatenate(pxs), np.concatenate(img_of)
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(a.out, "pixels.npz"), px=px.astype(np.float32), img=img_of.astype(np.int32), cols=np.array(list(CUES) + ["err_da", "err_ud"]))
    lines, summ = tables(rows, px, img_of)
    summ["_meta"] = {"images": len(rows), "pixels": int(len(px)), "encoder": a.encoder, "da_random_init": not a.weights, "minutes": round((time.time() - t0) / 60, 1)}
    write_json(os.path.join(a.out, "summary.json"), summ)
    print(f"[teacher] 끝 — 사진 {len(rows)} 장 · 픽셀 {len(px):,} 개 · {(time.time() - t0) / 60:.1f} 분" + (f" · !!! 유한하지 않은 값이 나와 건너뛴 사진 {bad} 장" if bad else "")
          + (" · 선생이 무작위 초기화 (ENC=vits) — 아래 수치는 뜻이 없다" if not a.weights else ""), flush=True)
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
