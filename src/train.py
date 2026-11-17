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

결과 (--out): train_log.csv (50 스텝마다) · val_log.csv (epoch 마다) · done.json. 체크포인트 (--ckpt): last.pt (이어 하기용, 끝나면 지움) · final.pt (가중치만)
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from common import MAX_DEPTH, MIN_DEPTH, PL_DARK, depth_metrics, eval_mask, seed_all, silog, ssi_trim, vivid_grid, write_json  # noqa: E402
from evaluate import eval_set  # noqa: E402

ARMS = ("base", "pl", "plv")
PAIRS = {"pl": ("kaist", "llvip"), "plv": ("vivid_day",)}


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
    ap.add_argument("--pl_weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--weights", default="", help="상대 깊이 체크포인트 (학생은 인코더만 싣고, 선생은 전부 싣는다). 비우면 무작위 초기화 (시험)")
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
    torch.backends.cudnn.benchmark = True                          # 입력 크기가 고정
    ds = D.Thermal("train", a.limit, aug="base", seed=a.seed)
    model = M.build(a.encoder, a.weights).to(dev)
    enc = [p for n, p in model.named_parameters() if "pretrained" in n]
    head = [p for n, p in model.named_parameters() if "pretrained" not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": head, "lr": a.lr * 10.0}], lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
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
          f"파라미터 {p['encoder_m']} + {p['head_m']} M · amp {int(amp)}", flush=True)

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

    def pair_batches():                                            # 쌍을 끝없이 돈다 (한 바퀴마다 다시 섞음). 이어 할 때는 새 순서로 시작
        k = 0
        while True:
            g = torch.Generator()
            g.manual_seed(a.seed * 100003 + step * 31 + k)
            yield from DataLoader(pds, batch_size=a.pair_bs, shuffle=True, generator=g, num_workers=a.workers // 2, drop_last=True, pin_memory=dev.type == "cuda")
            k += 1
    pit = pair_batches() if pds is not None else None

    new = not (a.resume and step > 0 and os.path.exists(os.path.join(a.out, "train_log.csv")))
    fh = open(os.path.join(a.out, "train_log.csv"), "w" if new else "a", newline="")
    vh = open(os.path.join(a.out, "val_log.csv"), "w" if new else "a", newline="")
    log, vlog = csv.writer(fh), csv.writer(vh)
    if new:
        log.writerow(["epoch", "step", "loss", "loss_pl", "lr", "sec"])
        vlog.writerow(["epoch", "loss", "loss_pl", "val_abs_rel", "val_rmse", "val_d1", "val_n"])
    t_start = time.time()
    for ep in range(ep0, a.epochs):
        model.train()
        ds.epoch = ep
        g = torch.Generator()
        g.manual_seed(a.seed * 1000 + ep)                          # 이어 해도 같은 순서
        loader = DataLoader(ds, batch_size=a.bs, shuffle=True, generator=g, num_workers=a.workers, drop_last=True, pin_memory=dev.type == "cuda")
        t0, run, run_pl, n = time.time(), 0.0, 0.0, 0
        for x, gt, _ in loader:
            x, gt = x.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
            lr = a.lr * (1 - step / total) ** 0.9
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lr, lr * 10.0
            pred = M.predict(model, x, gt.shape[-2:], amp)
            loss = silog(pred, gt, (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH))
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
            if not torch.isfinite(loss) or (lp is not None and not torch.isfinite(lp)):
                write_json(os.path.join(a.out, "fail.json"), {"epoch": ep, "step": step, "why": "loss 가 유한하지 않음"})
                print(f"!!! [train {name}] epoch {ep + 1} step {step}: loss 가 유한하지 않음 — 멈춤", flush=True)
                return 3
            opt.step()
            step += 1
            run += loss.item()
            run_pl += lp.item() if lp is not None else 0.0
            n += 1
            if step % 50 == 0 or step == 1:
                log.writerow([ep + 1, step, round(loss.item(), 5), round(lp.item(), 5) if lp is not None else "", f"{lr:.3g}", round(time.time() - t_start, 1)])
                fh.flush()
        sec = time.time() - t0
        v, _ = eval_set(model, "val", dev, a.val_limit, a.bs * 2, min(a.workers, 4), amp)
        vlog.writerow([ep + 1, round(run / max(n, 1), 5), round(run_pl / max(n, 1), 5) if pit is not None else "",
                       round(v["raw"]["abs_rel"], 5), round(v["raw"]["rmse"], 4), round(v["raw"]["d1"], 5), v["n"]])
        vh.flush()
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        print(f"[train {name}] epoch {ep + 1}/{a.epochs} · loss {run / max(n, 1):.4f}" + (f" · 쌍 손실 {run_pl / max(n, 1):.4f}" if pit is not None else "")
              + f" · val AbsRel {v['raw']['abs_rel']:.4f} RMSE {v['raw']['rmse']:.3f} δ1 {v['raw']['d1']:.4f} · "
              f"{sec / 60:.1f} 분 ({sec / max(n, 1):.2f} 초/스텝, 전체 {total} 스텝이면 약 {sec / max(n, 1) * total / 3600:.1f} 시간) · GPU 최대 {mem:.1f} GiB", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep + 1, "rng": torch.get_rng_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)
    fh.close()
    vh.close()
    torch.save(model.state_dict(), os.path.join(a.ckpt, "final.pt"))
    os.remove(last)
    write_json(os.path.join(a.out, "done.json"), {
        "arm": a.arm, "encoder": a.encoder, "seed": a.seed, "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "amp": int(amp), "images": len(ds), "steps": step,
        "pairs": {"sets": PAIRS[a.arm], "n": len(pds), "bs": a.pair_bs, "weight": a.pl_weight, "teacher_check": check} if pds is not None else None,
        "minutes": round((time.time() - t_start) / 60, 1), "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None,
        "params": p})
    print(f"[train {name}] 끝 — {step} 스텝", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
