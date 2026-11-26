"""학습 — 공식 Depth Anything V2 metric 학습 루프 (metric_depth/train.py) 와 같은 순서 · 같은 값, 데이터와 팔 (arm) 만 다르다.

공식 그대로인 것: 손실 SiLog (λ 0.5) · 정답 MIN_DEPTH ≤ gt ≤ MAX_DEPTH 인 픽셀만 · AdamW (betas 0.9 · 0.999, weight decay 0.01) ·
  학습률 인코더 5e-6 · 나머지 (헤드) 10 배 · 스텝마다 lr × (1 − 스텝 / 전체)^0.9, warmup 없음 · 마지막 못 채운 배치는 버림.
다른 것: 배치 16 (GPU 1 장) · 예측을 정답 해상도로 올려 손실 (LiDAR 정답이 드물어 정답을 줄이지 않는다) · 시드 고정 ·
  bf16 자동 혼합 정밀도 (--amp 1, GPU 만) · epoch 끝마다 val 을 재서 한 줄 (지켜보기만 — 고르는 데 쓰지 않고 마지막 epoch 을 쓴다) · last.pt (이어 하기).
팔
  base  뒤집기만
  pl    base + 스텝마다 정답 없는 쌍 배치 하나를 더 본다 (KAIST · LLVIP). 쌍의 RGB 에 선생 (고정한 Depth Anything V2 상대 깊이 모델) 을 돌린 출력을
        의사 라벨로, 같은 화면의 열화상에 대한 학생 예측과 축척 · 이동 무관 손실 (common.ssi_trim). RGB 가 너무 어두운 픽셀은 뺀다.
        MS2 배치의 손실 · 스텝 수 · 학습률은 base 와 같다 — 달라지는 것은 더해지는 쌍 손실뿐.
  plv   pl 과 같은 방식인데 쌍이 ViViD 야외 낮 (평가할 데이터의 다른 분할 = 선행 논문의 '대상 데이터에 적응' 설정을 우리 파이프라인으로 옮긴 비교 팔).
        ViViD 는 RGB 와 열화상 카메라가 달라 선생 출력을 열화상 시점으로 옮겨 쓴다 (common.vivid_grid).
  plv 는 시작할 때 선생을 확인한다: ViViD 낮의 LiDAR 정답에 선생 출력 (축척 · 이동을 맞춘 것) 을 대 본 AbsRel · δ1 을 한 줄 찍는다 (pl 은 ViViD 를 읽지 않는다).
  sep   모양 + 스케일 학생 (model.Sep) 을 MS2 만으로. 손실 = 모양 (정답 화소에서 log 깊이 차의 표준편차) + 스케일 (|s − 정답에서 구한 스케일|) — common.shape_scale.
        구조와 손실만 바꾼 줄: 스텝 수 · 학습률 · 증강은 base 와 같다 (스케일 헤드의 학습률은 DPT 헤드와 같다).
  ours  sep + 스텝마다 쌍 배치 하나 (KAIST · LLVIP · Freiburg · TartanRGBT). 쌍의 목표는 미리 뽑아 둔 선생 (UniDepthV2) 의 깊이 (src/labels.py):
        모양 = 중앙값을 맞춘 log 깊이 차의 절대값을 큰 쪽 20 % 버리고 평균, 스케일 = |s − 선생의 스케일|. 학습 중에는 선생을 돌리지 않는다.
        쌍의 크기가 둘이라 (640×512 · 640×298) 한 배치는 한 크기끼리만 묶고, 어느 쪽을 뽑을지는 쌍 수에 비례한 확률로 정한다 (시드 · 스텝으로 정해짐).
  tr    열화상 판 TR2M (model.TR) 을 MS2 만으로 — 고정한 상대 깊이 모델 위의 작은 모듈만 학습한다 (학습률 = base 의 헤드와 같은 lr × 10).
        손실 = SiLog (base 와 같음) + 0.5 × 의사 정답 (common.tr_pseudo) + 0.01 × 맵 매끄러움 (common.edge_smooth, A · B 각각). README '열화상 판 TR2M'.
  rt    tr 과 같은데, 고정하는 상대 깊이 모델이 공식 체크포인트가 아니라 먼저 MS2 열화상으로 미세조정한 것이다 (src/train_rel.py 가 만든 rel.pt 를 --weights 로).
  rtp   rt + 쌍 (README '본 실험 4'): 1 단계 (train_rel.py --pairs) 에 선생의 모양, 2 단계 (여기) 에 선생의 절대 깊이 — 스텝마다 쌍 배치 하나에 SiLog.
  rts   rt + 쌍을 1 단계 (train_rel.py --pairs, 선생의 모양) 에만. 2 단계 (여기) 는 rt 와 똑같이 MS2 만 — 쌍의 절대 깊이를 뺐을 때를 보는 진단 팔 (README '본 실험 4' 의 '진단').
  rtn · rtf · rtk   카메라 정보 실험 (README '카메라 정보'). 셋 다 1 단계는 rt 와 같고 (MS2 만), 2 단계에 쌍의 선생 절대 깊이 (SiLog) 를 더한다.
        쌍은 카메라 값을 아는 셋 (KAIST · Freiburg · TartanRGBT), 라벨은 선생에게 그 카메라 값을 주고 뽑은 것 (labels.py --cam).
        rtn = 카메라 정보 없이 · rtf = 깊이를 기준 초점의 것으로 환산해 배움 (common.canon_scale) + 크기 바꾸기 증강 (common.JITTER, 카메라 값도 같이) ·
        rtk = rtf + 모듈 입력에 시선 방향 임베딩 (model.ray_tokens). 증강의 배율은 스텝 · 시드로 정해지고 MS2 배치와 쌍 배치가 따로 뽑는다.
        rtj = rtk 와 같다 (여기 2 단계는 똑같다). 다른 것은 1 단계: 상대 깊이 모델도 같은 크기 바꾸기 증강으로 학습한 것을 쓴다 (train_rel.py --jitter, README '카메라 정보' 의 조정).
  rtfn · rtc   보완 실험 (README '보완 실험') — 크기 바꾸기 증강 없이 카메라 정보를 쓴다. 1 단계 · 쌍 · 라벨 · 손실은 rtf 와 같다.
        rtfn = rtf 에서 크기 바꾸기 증강만 뺀 것 (깊이 라벨을 기준 초점으로 환산하는 것은 그대로) — MS2 의 손해가 그 증강에서 오는지 가르는 진단 팔.
        rtc  = 라벨을 환산하는 대신 입력 영상을 기준 초점의 크기로 맞춰 넣는다 (model.to_canon — 학습의 쌍과 평가의 모든 영상. MS2 는 기준 초점 그대로라 바뀌지 않는다).
               배율이 쌍 세트마다 달라 쌍 배치를 세트별로 묶는다 (어느 세트를 뽑을지는 쌍 수에 비례 — 세트의 몫은 rtf 와 같다). 시선 방향 임베딩은 쓰지 않는다.
  rtkp · rtfnp   모양도 쌍으로 (README '모양도 쌍으로') — 여기 2 단계는 rtk · rtfn 과 똑같다. 다른 것은 1 단계: 상대 깊이 모델을 MS2 의 LiDAR 에 더해
        쌍 넷 (KAIST · LLVIP · TartanRGBT · Freiburg) 의 선생 깊이로도 학습한 것을 쓴다 (train_rel.py --pairs — 배율 · 기준점과 무관한 손실이라 모양만 배운다).
  mtd   base (한꺼번에 내는 구조, 전체 미세조정) + 쌍의 선생 절대 깊이에 SiLog. MonoTher-Depth 의 방식을 평가와 무관한 쌍으로 옮긴 비교 팔 (신뢰도 가중은 없음).
        rtp · mtd 의 쌍은 ours 와 같다 (KAIST · LLVIP · Freiburg · TartanRGBT, 선생 = UniDepthV2, 크기별 배치 · 쌍 수에 비례한 뽑기). 쓸 화소 = 라벨 0 초과 80 m 이하.

결과 (--out): train_log.csv (50 스텝마다) · val_log.csv (epoch 마다) · done.json. 체크포인트 (--ckpt): last.pt (이어 하기용, 끝나면 지움) · final.pt (가중치만)
"""
import argparse
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from common import (CAMS, JITTER, MAX_DEPTH, MIN_DEPTH, PL_DARK, PL_MIN, PL_TRIM, cam_flip, cam_resize, depth_metrics, edge_smooth, eval_mask,  # noqa: E402
                    seed_all, shape_scale, silog, silog_px, ssi_trim, tr_pseudo, vivid_grid, write_json)
from evaluate import eval_set  # noqa: E402

ARMS = ("base", "pl", "plv", "sep", "ours", "tr", "rt", "rtp", "rts", "rtn", "rtf", "rtk", "rtj", "rtfn", "rtc", "rtkp", "rtfnp", "mtd")
TR_PSEUDO, TR_SMOOTH = 0.5, 0.01                                  # tr 의 손실 비중 (TR2M 의 λ2 · λ4)
PAIRS = {"pl": ("kaist", "llvip"), "plv": ("vivid_day",)}
SEP = ("sep", "ours")                                             # 모양 + 스케일 학생을 쓰는 팔
LABEL_GROUPS = D.LABEL_GROUPS                                      # 선생 라벨을 쓰는 팔 (ours · rtp · mtd) 의 쌍 — 크기가 같은 것끼리
LABEL_ARMS = ("ours", "rtp", "mtd")
# 카메라 정보를 쓰는 팔 → (깊이를 기준 초점의 것으로 배움, 시선 방향 임베딩, 입력 크기를 다루는 법: "jit" = 크기 바꾸기 증강 · "image" = 기준 초점의 크기로 맞춤 · "" = 그대로)
CAM_ARMS = {"rtn": (False, False, ""), "rtf": (True, False, "jit"), "rtk": (True, True, "jit"), "rtj": (True, True, "jit"), "rtfn": (True, False, ""), "rtc": (True, False, "image"),
            "rtkp": (True, True, "jit"), "rtfnp": (True, False, "")}   # rtkp · rtfnp 의 2 단계 = rtk · rtfn (1 단계만 다르다)


def jitter(x, cam, r):
    """크기 바꾸기 증강 (model.rescale): 입력 (B, 3, H, W) 을 r 배로 바꾸고 카메라 값 (B, 4) 도 같이 바꾼다."""
    x, rx, ry = M.rescale(x, r)
    return x, (cam if r == 1.0 else cam_resize(cam, rx, ry))


def bright(rgb):
    """정규화된 RGB (B, 3, H, W) → 선생이 볼 수 있는 픽셀 (B, H, W): 밝기 (세 채널 평균, 7×7 평균) ≥ PL_DARK."""
    mean, std = rgb.new_tensor(D.MEAN), rgb.new_tensor(D.STD)
    return F.avg_pool2d((rgb * std + mean).mean(1, keepdim=True), 7, 1, 3, count_include_pad=False)[:, 0] >= PL_DARK   # 가장자리는 있는 픽셀만으로 평균


def pseudo(teacher, rgb, amp, grid=None, inside=None):
    """쌍의 RGB → (열화상 시점의 선생 상대 깊이 (B, H, W), 쓸 픽셀). grid 를 주면 (ViViD) RGB 시점의 것을 열화상 시점으로 옮긴다."""
    disp, ok = M.teacher_disp(teacher, rgb, amp), bright(rgb)
    if grid is None:
        return disp, ok
    g = grid.expand(len(rgb), -1, -1, -1)
    disp = F.grid_sample(disp[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
    ok = (F.grid_sample(ok[:, None].float(), g, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0] > 0.99) & inside
    return disp, ok


@torch.no_grad()
def teacher_check(teacher, dev, amp, limit=64):
    """선생 점검 — ViViD 낮의 LiDAR 정답 (열화상 시점) 에 선생 출력을 대 본다. 사진마다 a · 선생 + b ≈ 1 / 정답 을 최소제곱으로 맞춘 뒤의 AbsRel · δ1 (사진 평균).
    선생 싣기 · RGB → 열화상 옮기기가 틀렸으면 여기서 드러난다. 학습 · 판정에는 쓰지 않는다."""
    gts, prs = D.Thermal("vivid_day", limit), D.Pairs(["vivid_day"], limit)
    if not len(gts) or [k for k, _, _ in gts.items] != [k for k, _, _ in prs.items]:
        return None
    grid, inside = (t.to(dev) for t in vivid_grid())
    rows = []
    for i in range(len(gts)):
        gt, rgb = gts[i][1].to(dev), prs[i][1][None].to(dev)
        disp, ok = pseudo(teacher, rgb, amp, grid, inside)
        m = eval_mask(gt, border=True) & ok[0]
        if int(m.sum()) < 100:
            continue
        t, y = disp[0][m].double(), 1.0 / gt[m].double()
        tc = t - t.mean()
        a = (tc * (y - y.mean())).sum() / (tc * tc).sum().clamp(min=1e-12)
        pred = 1.0 / (a * t + (y.mean() - a * t.mean())).clamp(min=1.0 / MAX_DEPTH)
        rows.append(depth_metrics(pred, gt[m].double()))
    if not rows:
        return None
    return {"n": len(rows), "abs_rel": sum(r["abs_rel"] for r in rows) / len(rows), "d1": sum(r["d1"] for r in rows) / len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--pair_bs", type=int, default=8, help="쌍 배치 (512×640 이라 MS2 배치 16 장과 화소 수가 같다)")
    ap.add_argument("--pl_weight", type=float, default=1.0, help="쌍 손실의 비중 (ours 에서는 쌍의 모양 손실)")
    ap.add_argument("--pl_scale_weight", type=float, default=1.0, help="ours: 쌍의 스케일 손실의 비중")
    ap.add_argument("--jitter", default="", help="rtf · rtk · rtj: 크기 바꾸기 증강의 배율들 (쉼표). 비우면 common.JITTER (0.71–1.41 아홉) — 로컬 흐름 시험은 메모리 때문에 1 이하만 준다")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--weights", default="", help="상대 깊이 체크포인트 (학생은 인코더만 싣고, 선생 · tr · rt 의 고정 모델은 전부 싣는다). 비우면 무작위 초기화 (시험)")
    ap.add_argument("--limit", type=int, default=0, help="학습 장 수 제한 (smoke)")
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.ckpt, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.out))
    dev = torch.device(a.device)
    amp = bool(a.amp) and dev.type == "cuda"

    seed_all(a.seed)
    cam_arm = a.arm in CAM_ARMS
    canon, ray, how = CAM_ARMS.get(a.arm, (False, False, ""))
    jit = tuple(float(v) for v in a.jitter.split(",") if v) or JITTER
    torch.backends.cudnn.benchmark = how != "jit"                  # 입력 크기가 몇 가지로 고정일 때만 (크기 바꾸기 증강을 쓰면 끈다)
    ds = D.Thermal("train", a.limit, aug="base", seed=a.seed)
    sep, tr = a.arm in SEP, a.arm in ("tr", "rt", "rtp", "rts") or cam_arm
    if tr:                                                         # 고정한 상대 깊이 모델 + 작은 모듈 — 모듈만 학습한다
        model = M.build_tr(a.encoder, a.weights, ray, canon, how == "image").to(dev)
        groups = [{"params": list(model.scalemap.parameters()), "lr": a.lr * 10.0}]
    else:
        model = M.build(a.encoder, a.weights)
        model = (M.Sep(model) if sep else model).to(dev)
        enc = [p for n, p in model.named_parameters() if "pretrained" in n]
        head = [p for n, p in model.named_parameters() if "pretrained" not in n]
        groups = [{"params": enc, "lr": a.lr}, {"params": head, "lr": a.lr * 10.0}]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
    steps_ep = len(ds) // a.bs
    if steps_ep < 1:
        raise SystemExit(f"!!! [train {name}] 학습 사진 {len(ds)} 장이 배치 {a.bs} 보다 적음")
    total = steps_ep * a.epochs
    step, ep0 = 0, 0
    last = os.path.join(a.ckpt, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, ep0 = ck["step"], ck["epoch"]
        torch.set_rng_state(ck["rng"])
        print(f"[train {name}] 이어 하기: epoch {ep0} · step {step}", flush=True)
    p = M.count_params(model)
    print(f"[train {name}] {a.arm} · {a.encoder} · 시드 {a.seed} · train {len(ds):,} 장 · {a.epochs} epoch × {steps_ep} 스텝 (배치 {a.bs}) · "
          f"파라미터 {p['encoder_m']} + {p['head_m']} M (학습 {p['trainable_m']} M) · amp {int(amp)} · 시작 가중치 {os.path.basename(a.weights) if a.weights else '무작위'}", flush=True)

    teacher = pds = grid = inside = check = None
    if a.arm in PAIRS:
        pds = D.Pairs(PAIRS[a.arm], a.limit)
        if len(pds) < a.pair_bs:
            raise SystemExit(f"!!! [train {name}] 쌍 {len(pds)} 개가 쌍 배치 {a.pair_bs} 보다 적음")
        teacher = M.build_teacher(a.encoder, a.weights).to(dev)
        if a.arm == "plv":
            grid, inside = (t.to(dev) for t in vivid_grid())
        # 선생 확인은 ViViD 낮을 학습에 쓰는 plv 에서만 한다. pl 은 평가 대상 (ViViD) 의 다른 분할을 읽지 않는다 — 선생의 품질은 run.sh teacher (MS2 의 RGB) 가 잰다.
        check = teacher_check(teacher, dev, amp) if a.arm == "plv" else None
        print(f"[train {name}] 쌍 {' + '.join(PAIRS[a.arm])} {len(pds):,} 개 (배치 {a.pair_bs}, 손실 비중 {a.pl_weight}) · 선생 확인 (ViViD 낮 LiDAR 대비, 축척 · 이동 맞춤): "
              + (f"AbsRel {check['abs_rel']:.4f} δ1 {check['d1']:.4f} ({check['n']} 장)" if check else "안 함" if a.arm != "plv" else "건너뜀 (ViViD 낮 자료 없음)"), flush=True)

    def pair_batches(d=None, salt=0):                              # 쌍을 끝없이 돈다 (한 바퀴마다 다시 섞음). 이어 할 때는 새 순서로 시작
        k = 0
        while True:
            g = torch.Generator()
            g.manual_seed(a.seed * 100003 + step * 31 + k + salt * 7)
            yield from DataLoader(pds if d is None else d, batch_size=a.pair_bs, shuffle=True, generator=g, num_workers=a.workers // 2, drop_last=True, pin_memory=dev.type == "cuda")
            k += 1
    pit = pair_batches() if pds is not None else None
    lds = lits = None
    lgroups = D.CAM_LABEL_GROUPS if cam_arm else LABEL_GROUPS
    if how == "image":                                             # 영상을 기준 초점에 맞추는 배율이 세트마다 다르다 → 한 배치는 한 세트끼리
        lgroups = tuple((n,) for g in lgroups for n in g)
    if a.arm in LABEL_ARMS or cam_arm:                             # 쌍 + 미리 뽑아 둔 선생 라벨, 크기별로 따로
        lds = [d for d in (D.PairLabels(g, a.limit, sub="labels_cam" if cam_arm else "labels", cams=cam_arm) for g in lgroups) if len(d) >= a.pair_bs]
        if not lds:
            raise SystemExit(f"!!! [train {name}] 쌍이 쌍 배치 {a.pair_bs} 보다 적음")
        lits = [pair_batches(d, i + 1) for i, d in enumerate(lds)]
        print(f"[train {name}] 쌍 (선생 라벨) " + " · ".join(f"{len(d):,}" for d in lds) + f" 개 ({'세트별' if how == 'image' else '크기별'}, 배치 {a.pair_bs}, "
              + (f"비중 모양 {a.pl_weight} · 스케일 {a.pl_scale_weight})" if a.arm == "ours" else f"선생의 절대 깊이에 SiLog, 비중 {a.pl_weight})")
              + (f" · 라벨 = 카메라 값을 준 것 · 기준 초점 환산 {'켬' if canon else '끔'} · 크기 바꾸기 {f'{min(jit):.2f}–{max(jit):.2f}' if how == 'jit' else '기준 초점의 크기로 맞춤' if how == 'image' else '끔'}"
                 f" · 시선 방향 임베딩 {'켬' if ray else '끔'}"
                 if cam_arm else ""), flush=True)
    cam_ms2 = torch.tensor(CAMS["ms2"], dtype=torch.float32, device=dev)

    new = not (a.resume and step > 0 and os.path.exists(os.path.join(a.out, "train_log.csv")))
    fh = open(os.path.join(a.out, "train_log.csv"), "w" if new else "a", newline="")
    vh = open(os.path.join(a.out, "val_log.csv"), "w" if new else "a", newline="")
    log, vlog = csv.writer(fh), csv.writer(vh)
    if new:
        log.writerow(["epoch", "step", "loss", "loss_pl", "lr", "sec", "ms2_scale", "pl_scale"])
        vlog.writerow(["epoch", "loss", "loss_pl", "val_abs_rel", "val_rmse", "val_d1", "val_n"])
    t_start = time.time()
    for ep in range(ep0, a.epochs):
        model.train()
        ds.epoch = ep
        g = torch.Generator()
        g.manual_seed(a.seed * 1000 + ep)                          # 이어 해도 같은 순서
        loader = DataLoader(ds, batch_size=a.bs, shuffle=True, generator=g, num_workers=a.workers, drop_last=True, pin_memory=dev.type == "cuda")
        t0, run, run_pl, run_sc, run_psc, n, n_tp = time.time(), 0.0, 0.0, 0.0, 0.0, 0, 0
        for x, gt, idx in loader:
            x, gt = x.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
            lr = a.lr * (1 - step / total) ** 0.9
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * (1 - step / total) ** 0.9
            ok = (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH)
            sc = psc = None
            if tr:
                cam, gtc, dmax = None, gt, None
                if canon:                                          # 카메라 값: 뒤집은 사진은 주점도 뒤집고, 크기를 바꾸면 같이 바꾼다. 정답은 기준 초점의 깊이로
                    fl = torch.tensor([D.flipped(a.seed, ep, int(i)) for i in idx], device=dev)
                    cam = cam_ms2.expand(len(x), 4)
                    cam = torch.where(fl[:, None], cam_flip(cam, x.shape[-1]), cam)
                    if how == "jit":
                        x, cam = jitter(x, cam, random.Random((a.seed * 1000003 + step) * 4 + 1).choice(jit))
                    elif how == "image":
                        x, cam = M.to_canon(x, cam)                # MS2 는 기준 초점 그대로라 바뀌지 않는다
                    cs = M.canon_of(cam, x.shape[-2:])
                    gtc, dmax = gt * cs[:, None, None], MAX_DEPTH * cs     # 의사 정답의 상한 (80 m) 도 기준 초점의 값으로
                pred, ma, mb, dr, xn = M.predict_tr(model, x, gt.shape[-2:], amp, cam if ray else None)
                ltp, k = tr_pseudo(ma, mb, F.interpolate(dr[:, None], gt.shape[-2:], mode="bilinear", align_corners=True)[:, 0], gtc, ok, dmax=dmax)
                n_tp += k
                loss = silog(pred, gtc, ok) + TR_PSEUDO * ltp + TR_SMOOTH * (edge_smooth(ma, xn.float()) + edge_smooth(mb, xn.float()))
            elif sep:
                zn, s_hat = M.predict_sep(model, x, gt.shape[-2:], amp)
                sh, sc, _ = shape_scale(zn, s_hat, torch.log(gt.clamp(min=MIN_DEPTH)), ok)
                loss = sh + sc
            else:
                loss = silog(M.predict(model, x, gt.shape[-2:], amp), gt, ok)
            lp = None
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                if pit is not None:                                # 쌍 배치는 따로 계산해 기울기만 더한다 (메모리는 한 배치만큼만)
                    xt, rgb = (t.to(dev, non_blocking=True) for t in next(pit))
                    disp, ok = pseudo(teacher, rgb, amp, grid, inside)
                    lp = ssi_trim(M.predict(model, xt, xt.shape[-2:], amp), disp, ok)
                    if torch.isfinite(lp):
                        (a.pl_weight * lp).backward()
                if lits is not None:
                    gi = random.Random(a.seed * 1000003 + step).choices(range(len(lds)), weights=[len(d) for d in lds])[0]   # 전체 스텝 < 1000003 이라 시드끼리 안 겹친다
                    xt, lab, *pc = (t.to(dev, non_blocking=True) for t in next(lits[gi]))
                    if cam_arm:                                    # rtn · rtf · rtk: 선생의 절대 깊이에 SiLog — 카메라 값을 아는 쌍, 2 단계의 모듈만 배운다
                        okp, pc = (lab > 0) & (lab <= MAX_DEPTH), pc[0]
                        if canon:
                            if how == "jit":
                                xt, pc = jitter(xt, pc, random.Random((a.seed * 1000003 + step) * 4 + 2).choice(jit))
                            elif how == "image":
                                xt, pc = M.to_canon(xt, pc)        # 쌍의 영상을 기준 초점의 크기로 — 라벨에는 망 크기 맞추기로 남은 1–2 % 만 곱해진다
                            lab = lab * M.canon_of(pc, xt.shape[-2:])[:, None, None]
                        lp = a.pl_weight * silog_px(M.predict_tr(model, xt, lab.shape[-2:], amp, pc if ray else None)[0], lab, okp)
                    elif a.arm == "ours":
                        zn, s_hat = M.predict_sep(model, xt, xt.shape[-2:], amp)
                        psh, psc, _ = shape_scale(zn, s_hat, torch.log(lab.clamp(min=PL_MIN)), lab > 0, PL_TRIM, 100)
                        lp = a.pl_weight * psh + a.pl_scale_weight * psc
                    else:                                          # rtp · mtd: 선생의 절대 깊이에 SiLog (MS2 와 같은 범위)
                        lp = a.pl_weight * silog_px(M.predict(model, xt, xt.shape[-2:], amp), lab, (lab > 0) & (lab <= MAX_DEPTH))
                    if torch.isfinite(lp):
                        lp.backward()
            if not torch.isfinite(loss) or (lp is not None and not torch.isfinite(lp)):
                write_json(os.path.join(a.out, "fail.json"), {"epoch": ep, "step": step, "why": "loss 가 유한하지 않음"})
                print(f"!!! [train {name}] epoch {ep + 1} step {step}: loss 가 유한하지 않음 — 멈춤", flush=True)
                return 3
            opt.step()
            step += 1
            run += loss.item()
            run_pl += lp.item() if lp is not None else 0.0
            run_sc += sc.item() if sc is not None else 0.0
            run_psc += psc.item() if psc is not None else 0.0
            n += 1
            if step % 50 == 0 or step == 1:
                log.writerow([ep + 1, step, round(loss.item(), 5), round(lp.item(), 5) if lp is not None else "", f"{opt.param_groups[0]['lr']:.3g}", round(time.time() - t_start, 1),
                              round(sc.item(), 5) if sc is not None else "", round(psc.item(), 5) if psc is not None else ""])
                fh.flush()
        sec = time.time() - t0
        v, _ = eval_set(model, "val", dev, a.val_limit, a.bs * 2, min(a.workers, 4), amp)
        has_pl = pit is not None or lits is not None
        vlog.writerow([ep + 1, round(run / max(n, 1), 5), round(run_pl / max(n, 1), 5) if has_pl else "",
                       round(v["raw"]["abs_rel"], 5), round(v["raw"]["rmse"], 4), round(v["raw"]["d1"], 5), v["n"]])
        vh.flush()
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        print(f"[train {name}] epoch {ep + 1}/{a.epochs} · loss {run / max(n, 1):.4f}" + (f" (스케일 {run_sc / max(n, 1):.4f})" if sep else "")
              + (f" · 쌍 손실 {run_pl / max(n, 1):.4f}" if has_pl else "") + (f" (스케일 {run_psc / max(n, 1):.4f})" if a.arm == "ours" else "")
              + (f" · 의사 정답을 쓴 사진 {n_tp:,} / {n * a.bs:,}" if tr else "")
              + f" · val AbsRel {v['raw']['abs_rel']:.4f} RMSE {v['raw']['rmse']:.3f} δ1 {v['raw']['d1']:.4f} · "
              f"{sec / 60:.1f} 분 ({sec / max(n, 1):.2f} 초/스텝, 전체 {total} 스텝이면 약 {sec / max(n, 1) * total / 3600:.1f} 시간) · GPU 최대 {mem:.1f} GiB", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep + 1, "rng": torch.get_rng_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)
    fh.close()
    vh.close()
    torch.save(model.state_dict(), os.path.join(a.ckpt, "final.pt"))
    os.remove(last)
    write_json(os.path.join(a.out, "done.json"), {
        "arm": a.arm, "structure": a.arm if tr else "sep" if sep else "joint", "encoder": a.encoder, "seed": a.seed, "epochs": a.epochs, "bs": a.bs, "lr": a.lr,
        "lr_groups": [g["base_lr"] for g in groups], "amp": int(amp), "images": len(ds), "steps": step,
        "pairs": {"sets": PAIRS[a.arm], "n": len(pds), "bs": a.pair_bs, "weight": a.pl_weight, "teacher_check": check} if pds is not None else
                 {"sets": lgroups, "n": [len(d) for d in lds], "bs": a.pair_bs, "weight": a.pl_weight,
                  **({"loss": "shape_scale", "scale_weight": a.pl_scale_weight} if a.arm == "ours" else {"loss": "silog"})} if lds is not None else None,
        "camera": {"labels": "labels_cam", "canon": canon, "jitter": list(jit) if how == "jit" else None, "resize": how == "image", "ray": ray} if cam_arm else None,
        "minutes": round((time.time() - t_start) / 60, 1), "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None,
        "params": p})
    print(f"[train {name}] 끝 — {step} 스텝", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
