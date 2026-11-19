"""열화상용 상대 깊이 모델 학습 — 팔 rt 의 1 단계 (README '열화상 판 TR2M' 의 B). 끝나면 고정해 그 위에서 TR2M 꼴 모듈을 학습한다 (train.py --arm rt).

Depth Anything V2 상대 깊이 모델 (체크포인트 전부, model.build_rel) 을 MS2 열화상과 LiDAR 정답으로 미세조정한다. 출력은 그대로 역깊이 꼴이고 스케일 · 기준점은 배우지 않는다.
손실 = common.ssi_disp (사진마다 중앙값 · 평균 절대 편차로 맞춘 역깊이의 절대 오차, 정답 화소만). 나머지는 base 와 같다:
  AdamW (weight decay 0.01) · 학습률 인코더 lr · 헤드 lr × 10 · 스텝마다 (1 − 진행)^0.9 · 배치 · epoch · 좌우 뒤집기 증강 · bf16 · 시드 · 이어 하기.
val 은 epoch 마다 재서 찍기만 한다: 사진마다 역깊이에서 a · x + b 를 정답에 맞춘 뒤의 AbsRel (상대 깊이 모델 점검과 같은 식 — 고정한 공식 모델은 0.1267 이었다).

결과 (--out): train_log.csv · val_log.csv · done.json. 체크포인트 (--ckpt): last.pt (이어 하기, 끝나면 지움) · rel.pt (가중치만 — train.py --arm rt --weights 로 넘긴다)
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from common import MAX_DEPTH, MIN_DEPTH, seed_all, ssi_disp, write_json  # noqa: E402
from rel_check import measure  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--weights", default="", help="Depth Anything V2 상대 깊이 체크포인트 (전부 싣는다). 비우면 무작위 초기화 (시험)")
    ap.add_argument("--limit", type=int, default=0)
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
    name = os.path.basename(os.path.dirname(os.path.normpath(a.out)))
    dev = torch.device(a.device)
    amp = bool(a.amp) and dev.type == "cuda"

    seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    ds = D.Thermal("train", a.limit, aug="base", seed=a.seed)
    vds = D.Thermal("val", a.val_limit)
    model = M.build_rel(a.encoder, a.weights).to(dev)
    enc = [p for n, p in model.named_parameters() if "pretrained" in n]
    head = [p for n, p in model.named_parameters() if "pretrained" not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": head, "lr": a.lr * 10.0}], lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
    steps_ep = len(ds) // a.bs
    if steps_ep < 1:
        raise SystemExit(f"!!! [train-rel {name}] 학습 사진 {len(ds)} 장이 배치 {a.bs} 보다 적음")
    total = steps_ep * a.epochs
    step, ep0 = 0, 0
    last = os.path.join(a.ckpt, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, ep0 = ck["step"], ck["epoch"]
        torch.set_rng_state(ck["rng"])
        print(f"[train-rel {name}] 이어 하기: epoch {ep0} · step {step}", flush=True)
    print(f"[train-rel {name}] 열화상용 상대 깊이 모델 · {a.encoder} · 시드 {a.seed} · train {len(ds):,} 장 · {a.epochs} epoch × {steps_ep} 스텝 (배치 {a.bs}) · "
          f"{'공식 상대 깊이 체크포인트에서' if a.weights else '무작위 초기화 (흐름 시험)'} · amp {int(amp)}", flush=True)
    new = not (a.resume and step > 0 and os.path.exists(os.path.join(a.out, "train_log.csv")))
    fh = open(os.path.join(a.out, "train_log.csv"), "w" if new else "a", newline="")
    vh = open(os.path.join(a.out, "val_log.csv"), "w" if new else "a", newline="")
    log, vlog = csv.writer(fh), csv.writer(vh)
    if new:
        log.writerow(["epoch", "step", "loss", "lr", "sec"])
        vlog.writerow(["epoch", "loss", "val_aligned_abs_rel", "val_aligned_d1", "val_n"])
    t_start, v = time.time(), None
    for ep in range(ep0, a.epochs):
        model.train()
        ds.epoch = ep
        g = torch.Generator()
        g.manual_seed(a.seed * 1000 + ep)                          # base 와 같은 순서
        loader = DataLoader(ds, batch_size=a.bs, shuffle=True, generator=g, num_workers=a.workers, drop_last=True, pin_memory=dev.type == "cuda")
        t0, run, n, zero, tot = time.time(), 0.0, 0, 0, 0
        for x, gt, _ in loader:
            x, gt = x.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
            f = (1 - step / total) ** 0.9
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = a.lr * f, a.lr * 10.0 * f
            ok = (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH)
            pred = M.rel_forward(model, x, gt.shape[-2:], amp)
            loss = ssi_disp(pred, gt, ok)
            zero += int(((pred <= 0) & ok).sum())                       # ReLU 에 막혀 0 이 된 정답 화소 (기울기가 없다)
            tot += int(ok.sum())
            if not torch.isfinite(loss):
                write_json(os.path.join(a.out, "fail.json"), {"epoch": ep, "step": step, "why": "loss 가 유한하지 않음"})
                print(f"!!! [train-rel {name}] epoch {ep + 1} step {step}: loss 가 유한하지 않음 — 멈춤", flush=True)
                return 3
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            run += loss.item()
            n += 1
            if step % 50 == 0 or step == 1:
                log.writerow([ep + 1, step, round(loss.item(), 5), f"{opt.param_groups[0]['lr']:.3g}", round(time.time() - t_start, 1)])
                fh.flush()
        sec = time.time() - t0
        model.eval()
        rows = measure(model, vds, dev, a.bs * 2, min(a.workers, 4))
        v = (float(np.mean([r[1] for r in rows])), float(np.mean([r[2] for r in rows])), len(rows)) if rows else (float("nan"), float("nan"), 0)
        vlog.writerow([ep + 1, round(run / max(n, 1), 5), round(v[0], 5), round(v[1], 5), v[2]])
        vh.flush()
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        print(f"[train-rel {name}] epoch {ep + 1}/{a.epochs} · loss {run / max(n, 1):.4f} · 예측이 0 인 정답 화소 {zero / max(tot, 1) * 100:.2f} % · val (사진마다 맞춘 뒤) AbsRel {v[0]:.4f} δ1 {v[1]:.4f} · "
              f"{sec / 60:.1f} 분 ({sec / max(n, 1):.2f} 초/스텝, 전체 {total} 스텝이면 약 {sec / max(n, 1) * total / 3600:.1f} 시간) · GPU 최대 {mem:.1f} GiB", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep + 1, "rng": torch.get_rng_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)
    fh.close()
    vh.close()
    torch.save(model.state_dict(), os.path.join(a.ckpt, "rel.pt.tmp"))
    os.replace(os.path.join(a.ckpt, "rel.pt.tmp"), os.path.join(a.ckpt, "rel.pt"))
    os.remove(last)
    write_json(os.path.join(a.out, "done.json"), {"stage": "rel", "encoder": a.encoder, "seed": a.seed, "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "amp": int(amp),
                                                    "images": len(ds), "steps": step, "val_aligned_abs_rel": v[0] if v else None, "val_aligned_d1": v[1] if v else None,
                                                    "minutes": round((time.time() - t_start) / 60, 1)})
    print(f"[train-rel {name}] 끝 — {step} 스텝", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
