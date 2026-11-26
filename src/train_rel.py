"""열화상용 상대 깊이 모델 학습 — 팔 rt 의 1 단계 (README '열화상 판 TR2M' 의 B). 끝나면 고정해 그 위에서 TR2M 꼴 모듈을 학습한다 (train.py --arm rt).

Depth Anything V2 상대 깊이 모델 (체크포인트 전부, model.build_rel) 을 MS2 열화상과 LiDAR 정답으로 미세조정한다. 출력은 그대로 역깊이 꼴이고 스케일 · 기준점은 배우지 않는다.
손실 = common.ssi_disp (사진마다 중앙값 · 평균 절대 편차로 맞춘 역깊이의 절대 오차, 정답 화소만). 나머지는 base 와 같다:
  AdamW (weight decay 0.01) · 학습률 인코더 lr · 헤드 lr × 10 · 스텝마다 (1 − 진행)^0.9 · 배치 · epoch · 좌우 뒤집기 증강 · bf16 · 시드 · 이어 하기.
val 은 epoch 마다 재서 찍기만 한다: 사진마다 역깊이에서 a · x + b 를 정답에 맞춘 뒤의 AbsRel (상대 깊이 모델 점검과 같은 식 — 고정한 공식 모델은 0.1267 이었다).
--pairs (팔 rtp · rts): 스텝마다 쌍 배치 하나를 더 본다. 목표 = 미리 뽑아 둔 선생 (UniDepthV2) 의 깊이, 같은 ssi_disp 에 큰 오차 20 % 버림 (PL_TRIM), 쓸 화소 = 라벨 0 초과 80 m 이하.
  쌍은 train.py 의 ours · rtp · mtd 와 같다 (크기별 배치, 쌍 수에 비례한 뽑기). MS2 쪽 손실 · 스텝 수 · 학습률은 그대로이고 쌍 손실의 기울기를 더한다.

--jitter (팔 rtj): MS2 배치의 입력 크기를 스텝마다 바꾼다 (model.rescale, 배율은 common.JITTER). 손실이 스케일 · 기준점과 무관해 카메라 값은 필요 없다.
  2 단계의 크기 바꾸기가 이 모델에 처음 보는 크기를 넣던 것을 없애려는 것 (README '카메라 정보' 의 조정). 큰 배율 (1.2 초과) 은 배치를 둘로 나눠 계산한다 — 메모리 때문이고 기울기는 같다.

결과 (--out): train_log.csv · val_log.csv · done.json. 체크포인트 (--ckpt): last.pt (이어 하기, 끝나면 지움) · rel.pt (가중치만 — train.py --arm rt --weights 로 넘긴다)
"""
import argparse
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import data as D  # noqa: E402
import model as M  # noqa: E402
from common import JITTER, MAX_DEPTH, MIN_DEPTH, PL_DARK, PL_TRIM, feat_cos, feat_gram, grad_match, seed_all, ssi_disp, write_json  # noqa: E402
from rel_check import measure  # noqa: E402


FA_CHOICES = ("thr_cos", "thr_gram", "rgb_cos", "rgb_gram")          # 목표 (thr = 같은 열화상 · rgb = 짝 RGB, 둘 다 고정한 원래 모델에 넣은 것) × 형태 (cos = 토큰별 · gram = 패치 사이 관계)
FA_WEIGHT = {"cos": 1.0, "gram": 5.0}                                # cos = Depth Anything 의 feature alignment loss 의 비중.
#   gram = 처음 한 걸음에 토큰을 당기는 힘 (기울기 크기) 이 cos 와 같도록 실제 쌍에서 맞춘 값 — 열화상 학습 전 모델, 짝 RGB 목표, 세트마다 10 쌍: cos / gram = 4.0–5.6 (README '모양 실험')


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
    ap.add_argument("--pairs", action="store_true", help="쌍 + 선생 라벨을 더한다 (팔 rtp · rts)")
    ap.add_argument("--pair_bs", type=int, default=8)
    ap.add_argument("--pl_weight", type=float, default=1.0)
    ap.add_argument("--hold", default="", help="쌍에서 빼 둘 세트 (쉼표) — 방식을 고를 때 처음 보는 카메라 노릇을 시킨다 (run.sh shape). --pairs 와 함께")
    ap.add_argument("--grad_match", type=float, default=0.0, help="쌍의 촘촘한 라벨에 거는 경계 손실 (common.grad_match) 의 비중 (Depth Anything V2 는 2). 0 = 끔")
    ap.add_argument("--fa", default="", choices=("",) + FA_CHOICES, help="쌍 배치에 거는 특징 정렬 — 인코더 마지막 층의 패치 토큰을 고정한 원래 모델 (학습을 시작한 가중치) 의 토큰에. "
                    "thr_ = 같은 열화상을 넣은 것 (원래 모델에서 멀어지지 않게) · rgb_ = 짝 RGB 를 넣은 것 (RGB 쪽으로 끌어옴, 어둡거나 빈 패치는 뺌). "
                    "_cos = 토큰별 1 − 코사인, 이미 --fa_margin 보다 비슷한 토큰은 뺌 (Depth Anything 의 식) · _gram = 패치 사이 유사도 행렬을 맞춤 (DINOv3 의 Gram anchoring 꼴)")
    ap.add_argument("--fa_weight", type=float, default=-1.0, help="특징 정렬의 비중. 음수면 방식의 기본값 (FA_WEIGHT)")
    ap.add_argument("--fa_margin", type=float, default=0.85, help="_cos 의 여유 — 유사도가 이보다 큰 토큰은 손실에서 뺀다")
    ap.add_argument("--diag_every", type=int, default=100, help="특징 정렬이 있을 때 이 스텝마다 인코더 마지막 블록에서 정렬 손실의 기울기와 깊이 손실 (MS2 · 쌍) 의 기울기의 코사인을 잰다. 0 = 안 잼")
    ap.add_argument("--freeze_enc", action="store_true", help="인코더를 고정하고 헤드만 학습한다")
    ap.add_argument("--jitter", default="", help="크기 바꾸기 증강의 배율들 (쉼표, default = common.JITTER). 비우면 끔 — 팔 rtj 의 1 단계")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.ckpt, exist_ok=True)
    name = os.path.basename(os.path.dirname(os.path.normpath(a.out)))
    dev = torch.device(a.device)
    amp = bool(a.amp) and dev.type == "cuda"
    jit = JITTER if a.jitter == "default" else tuple(float(v) for v in a.jitter.split(",") if v)
    hold = tuple(v for v in a.hold.split(",") if v)
    if (a.fa or a.grad_match > 0 or hold) and not a.pairs:
        raise SystemExit("!!! [train-rel] --fa · --grad_match · --hold 는 --pairs 와 함께만")
    if any(h not in D.LABEL_SETS for h in hold):
        raise SystemExit(f"!!! [train-rel] --hold 는 {D.LABEL_SETS} 가운데서")
    fa_tgt, fa_form = a.fa.split("_") if a.fa else ("", "")
    fa_w = (FA_WEIGHT[fa_form] if a.fa_weight < 0 else a.fa_weight) if a.fa else 0.0
    margin = a.fa_margin if fa_form == "cos" else None

    seed_all(a.seed)
    torch.backends.cudnn.benchmark = not jit                       # 입력 크기가 고정일 때만
    ds = D.Thermal("train", a.limit, aug="base", seed=a.seed)
    vds = D.Thermal("val", a.val_limit)
    model = M.build_rel(a.encoder, a.weights).to(dev)
    origin = M.origin_encoder(model) if a.fa else None             # 특징 정렬의 목표를 내는 고정한 원래 인코더 = 학습을 시작한 가중치 그대로 (이어 하기 전에 떠 둔다)
    enc = [p for n, p in model.named_parameters() if "pretrained" in n]
    if a.freeze_enc:
        for q in enc:
            q.requires_grad_(False)
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
    lds = lits = None
    if a.pairs:                                                    # 쌍 + 선생 라벨, 크기별로 따로 (train.py 와 같은 뽑기)
        groups = [g for g in (tuple(n for n in g if n not in hold) for g in D.LABEL_GROUPS) if g]      # 빼 둔 세트는 학습에 쓰지 않는다
        lds = [d for d in (D.PairLabels(g, a.limit, rgb=fa_tgt == "rgb") for g in groups) if len(d) >= a.pair_bs]
        if not lds:
            raise SystemExit(f"!!! [train-rel {name}] 쌍이 쌍 배치 {a.pair_bs} 보다 적음")

        def pair_batches(d, salt):
            k = 0
            while True:
                g = torch.Generator()
                g.manual_seed(a.seed * 100003 + step * 31 + k + salt * 7)
                yield from DataLoader(d, batch_size=a.pair_bs, shuffle=True, generator=g, num_workers=a.workers // 2, drop_last=True, pin_memory=dev.type == "cuda")
                k += 1
        lits = [pair_batches(d, i + 1) for i, d in enumerate(lds)]
    print(f"[train-rel {name}] 열화상용 상대 깊이 모델 · {a.encoder} · 시드 {a.seed} · train {len(ds):,} 장 · {a.epochs} epoch × {steps_ep} 스텝 (배치 {a.bs}) · "
          f"{'공식 상대 깊이 체크포인트에서' if a.weights else '무작위 초기화 (흐름 시험)'} · amp {int(amp)}" + (f" · 크기 바꾸기 {min(jit):.2f}–{max(jit):.2f}" if jit else "")
          + (f" · 쌍 (선생 모양) " + " · ".join(f"{len(d):,}" for d in lds) + f" 개 (크기별, 배치 {a.pair_bs}, 비중 {a.pl_weight})" if lds else "")
          + (f" · 빼 둔 쌍 {','.join(hold)}" if hold else "") + (f" · 경계 손실 비중 {a.grad_match}" if a.grad_match > 0 else "")
          + (f" · 특징 정렬 {a.fa} (목표 = {'같은 열화상' if fa_tgt == 'thr' else '짝 RGB'} · {'토큰별' if fa_form == 'cos' else '관계'} · 비중 {fa_w}" + (f" · 여유 {margin}" if margin else "") + ")" if a.fa else "")
          + (" · 인코더 고정" if a.freeze_enc else ""), flush=True)
    diag = bool(a.fa) and not a.freeze_enc and a.diag_every > 0           # 기울기 충돌을 재는 곳 = 인코더 마지막 블록 (정렬하는 토큰을 바로 내는 곳)
    last_blk = [q for q in model.pretrained.blocks[-1].parameters() if q.requires_grad] if diag else []
    new = not (a.resume and step > 0 and os.path.exists(os.path.join(a.out, "train_log.csv")))
    fh = open(os.path.join(a.out, "train_log.csv"), "w" if new else "a", newline="")
    vh = open(os.path.join(a.out, "val_log.csv"), "w" if new else "a", newline="")
    log, vlog = csv.writer(fh), csv.writer(vh)
    if new:
        log.writerow(["epoch", "step", "loss", "loss_pl", "lr", "sec"])
        vlog.writerow(["epoch", "loss", "loss_pl", "val_aligned_abs_rel", "val_aligned_d1", "val_n"])
    t_start, v = time.time(), None
    for ep in range(ep0, a.epochs):
        model.train()
        if a.freeze_enc:
            model.pretrained.eval()
        ds.epoch = ep
        g = torch.Generator()
        g.manual_seed(a.seed * 1000 + ep)                          # base 와 같은 순서
        loader = DataLoader(ds, batch_size=a.bs, shuffle=True, generator=g, num_workers=a.workers, drop_last=True, pin_memory=dev.type == "cuda")
        t0, run, run_pl, n, zero, tot = time.time(), 0.0, 0.0, 0, 0, 0
        run_gm, run_fa = 0.0, 0.0
        fa_s, fa_k, fa_in, fa_all = 0.0, 0, 0, 0                     # 목표와의 코사인 합 · 그 토큰 수 (쓸 토큰만) · 손실에 든 토큰 수 · 전체 토큰 수
        dg = {"ms2": [], "pair": [], "ratio": []}                     # 기울기 코사인 (정렬 · MS2 깊이), (정렬 · 쌍 깊이), 크기 비 (비중을 곱한 정렬 / 쌍 깊이)
        for x, gt, _ in loader:
            x, gt = x.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
            f = (1 - step / total) ** 0.9
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = a.lr * f, a.lr * 10.0 * f
            ok = (gt >= MIN_DEPTH) & (gt <= MAX_DEPTH)
            r = random.Random((a.seed * 1000003 + step) * 4 + 3).choice(jit) if jit else 1.0
            x = M.rescale(x, r)[0]
            parts = 2 if r > 1.2 and len(x) > 1 else 1                  # 큰 배율은 배치를 둘로 나눠 계산한다 (ViT 전체에 기울기가 흘러 메모리가 토큰 수보다 빨리 는다)
            opt.zero_grad()
            loss = None
            for xs, gs, oks in zip(x.chunk(parts), gt.chunk(parts), ok.chunk(parts)):
                pred = M.rel_forward(model, xs, gs.shape[-2:], amp)
                part = ssi_disp(pred, gs, oks) / parts
                zero += int(((pred <= 0) & oks).sum())                  # ReLU 에 막혀 0 이 된 정답 화소 (기울기가 없다)
                tot += int(oks.sum())
                if not torch.isfinite(part):
                    write_json(os.path.join(a.out, "fail.json"), {"epoch": ep, "step": step, "why": "loss 가 유한하지 않음"})
                    print(f"!!! [train-rel {name}] epoch {ep + 1} step {step}: loss 가 유한하지 않음 — 멈춤", flush=True)
                    return 3
                part.backward()
                loss = part.detach() if loss is None else loss + part.detach()
            lp = None
            dstep = diag and step % a.diag_every == 0
            g_ms2 = [q.grad.detach().float().clone() if q.grad is not None else torch.zeros_like(q, dtype=torch.float32) for q in last_blk] if dstep else None
            if lits is not None:                                   # 쌍 배치는 따로 계산해 기울기만 더한다 (메모리는 한 배치만큼만)
                gi = random.Random(a.seed * 1000003 + step).choices(range(len(lds)), weights=[len(d) for d in lds])[0]
                bt = [t.to(dev, non_blocking=True) for t in next(lits[gi])]
                xt, lab = bt[0], bt[1]
                okp = (lab > 0) & (lab <= MAX_DEPTH)
                predp, tok = M.rel_forward_feats(model, xt, lab.shape[-2:], amp) if a.fa else (M.rel_forward(model, xt, lab.shape[-2:], amp), None)
                lp = a.pl_weight * ssi_disp(predp, lab, okp, PL_TRIM, 100)          # 쌍의 깊이 손실 (기록은 이것만 — 다른 변형과 견줄 수 있게)
                task = lp
                if a.grad_match > 0:                               # 경계 손실 — 촘촘한 선생 라벨에만
                    lg = grad_match(predp, lab, okp)
                    task = task + a.grad_match * lg
                    run_gm += lg.item()
                total_p = task
                if a.fa:                                           # 특징 정렬 — 목표는 고정한 원래 모델의 토큰 (기울기 없음)
                    if fa_tgt == "thr":                            # 같은 열화상 → 원래 모델에서 멀어지지 않게 붙잡기
                        tgt, pm = M.origin_tokens(origin, xt, amp), None
                    else:                                          # 짝 RGB → 열화상을 RGB 쪽으로. RGB 가 어둡거나 비어 있는 패치는 뺀다
                        tgt, pm = M.origin_tokens(origin, bt[-1], amp), M.patch_bright(bt[-1], D.MEAN.flatten().tolist(), D.STD.flatten().tolist(), PL_DARK)
                    lf = feat_cos(tok, tgt, pm, margin) if fa_form == "cos" else feat_gram(tok, tgt, pm)
                    total_p = task + fa_w * lf
                    run_fa += lf.item()
                    with torch.no_grad():                          # 기록용: 쓸 토큰에서 학생과 목표의 평균 유사도 · 손실에 든 토큰의 비율 — 토큰 수로 모아 epoch 끝에 나눈다
                        c = torch.nn.functional.cosine_similarity(tok.float(), tgt, dim=-1)
                        keep = torch.ones_like(c, dtype=torch.bool) if pm is None else pm
                        fa_s += float((c * keep).sum())
                        fa_k += int(keep.sum())
                        fa_in += int(((c < margin) & keep).sum()) if margin else int(keep.sum())
                        fa_all += c.numel()
                    if dstep:                                      # 기울기 충돌 — 마지막 블록에서만 (손실은 그대로 아래에서 한 번 더 흘린다)
                        gp = torch.autograd.grad(task, last_blk, retain_graph=True, allow_unused=True)
                        gf = torch.autograd.grad(lf, last_blk, retain_graph=True, allow_unused=True)
                        flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(q)).float().flatten() for g, q in zip(gs, last_blk)])  # noqa: E731
                        vf, vp, vm = flat(gf), flat(gp), torch.cat([g.flatten() for g in g_ms2])
                        if float(vf.norm()) > 1e-12:               # 여유 안에서 정렬 손실이 0 인 스텝은 세지 않는다
                            if float(vm.norm()) > 1e-12:
                                dg["ms2"].append(float(torch.nn.functional.cosine_similarity(vf, vm, dim=0)))
                            if float(vp.norm()) > 1e-12:
                                dg["pair"].append(float(torch.nn.functional.cosine_similarity(vf, vp, dim=0)))
                                dg["ratio"].append(float(fa_w * vf.norm() / vp.norm()))
                if not torch.isfinite(total_p):
                    write_json(os.path.join(a.out, "fail.json"), {"epoch": ep, "step": step, "why": "쌍 손실이 유한하지 않음"})
                    print(f"!!! [train-rel {name}] epoch {ep + 1} step {step}: 쌍 손실이 유한하지 않음 — 멈춤", flush=True)
                    return 3
                total_p.backward()
            opt.step()
            step += 1
            run += loss.item()
            run_pl += lp.item() if lp is not None else 0.0
            n += 1
            if step % 50 == 0 or step == 1:
                log.writerow([ep + 1, step, round(loss.item(), 5), round(lp.item(), 5) if lp is not None else "", f"{opt.param_groups[0]['lr']:.3g}", round(time.time() - t_start, 1)])
                fh.flush()
        sec = time.time() - t0
        model.eval()
        rows = measure(model, vds, dev, a.bs * 2, min(a.workers, 4))
        v = (float(np.mean([r[1] for r in rows])), float(np.mean([r[2] for r in rows])), len(rows)) if rows else (float("nan"), float("nan"), 0)
        vlog.writerow([ep + 1, round(run / max(n, 1), 5), round(run_pl / max(n, 1), 5) if lits is not None else "", round(v[0], 5), round(v[1], 5), v[2]])
        vh.flush()
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        mean = lambda v: float(np.mean(v)) if v else None  # noqa: E731
        extra = {"grad_match": run_gm / max(n, 1) if a.grad_match > 0 else None, "fa": run_fa / max(n, 1) if a.fa else None, "fa_sim": fa_s / fa_k if a.fa and fa_k else None,
                 "fa_on": fa_in / fa_all if a.fa and fa_all else None, "gcos_ms2": mean(dg["ms2"]), "gcos_pair": mean(dg["pair"]), "gratio": mean(dg["ratio"]), "gn": len(dg["pair"])}
        dtxt = (f" · 기울기 코사인 (정렬 · MS2 {extra['gcos_ms2']:+.3f} · 정렬 · 쌍 {extra['gcos_pair']:+.3f}, 크기 비 {extra['gratio']:.2f}, {extra['gn']} 번)"
                if extra["gcos_ms2"] is not None and extra["gcos_pair"] is not None else (" · 기울기: 정렬 손실이 0 (여유 안)" if diag else ""))
        print(f"[train-rel {name}] epoch {ep + 1}/{a.epochs} · loss {run / max(n, 1):.4f}" + (f" · 쌍 손실 {run_pl / max(n, 1):.4f}" if lits is not None else "")
              + (f" (경계 {extra['grad_match']:.4f})" if a.grad_match > 0 else "") + (f" (정렬 {extra['fa']:.4f} · 목표와의 유사도 " + (f"{extra['fa_sim']:.3f}" if extra["fa_sim"] is not None else "—") + f" · 손실에 든 토큰 {(extra['fa_on'] or 0) * 100:.0f} %{dtxt})" if a.fa else "")
              + f" · 예측이 0 인 정답 화소 {zero / max(tot, 1) * 100:.2f} % · val (사진마다 맞춘 뒤) AbsRel {v[0]:.4f} δ1 {v[1]:.4f} · "
              f"{sec / 60:.1f} 분 ({sec / max(n, 1):.2f} 초/스텝, 전체 {total} 스텝이면 약 {sec / max(n, 1) * total / 3600:.1f} 시간) · GPU 최대 {mem:.1f} GiB", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep + 1, "rng": torch.get_rng_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)
    fh.close()
    vh.close()
    torch.save(model.state_dict(), os.path.join(a.ckpt, "rel.pt.tmp"))
    os.replace(os.path.join(a.ckpt, "rel.pt.tmp"), os.path.join(a.ckpt, "rel.pt"))
    os.remove(last)
    write_json(os.path.join(a.out, "done.json"), {"stage": "rel", "pairs": [len(d) for d in lds] if lds else None, "pair_bs": a.pair_bs if lds else None, "pl_weight": a.pl_weight if lds else None, "pl_trim": PL_TRIM if lds else None, "encoder": a.encoder, "seed": a.seed, "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "amp": int(amp),
                                                    "images": len(ds), "steps": step, "val_aligned_abs_rel": v[0] if v else None, "val_aligned_d1": v[1] if v else None,
                                                    "jitter": list(jit) if jit else None, "minutes": round((time.time() - t_start) / 60, 1),
                                                    "hold": list(hold) or None, "grad_match": a.grad_match or None, "fa": a.fa or None, "fa_weight": fa_w or None, "fa_margin": margin,
                                                    "freeze_enc": bool(a.freeze_enc), "last_epoch": extra if a.epochs > ep0 else None})
    print(f"[train-rel {name}] 끝 — {step} 스텝", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
