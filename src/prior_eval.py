"""선행 방법 (MonoTher-Depth) 을 우리 평가 세트에서 잰다 — 저자가 공개한 코드와 가중치 (MS2 로 학습한 것) 를 그대로 쓴다. README '선행 방법'.

  python src/prior_eval.py --code <저자 코드 폴더> --ckpt <저자 가중치> --base <Depth Anything V1 ViT-L 가중치> --out <폴더> [--sets …] [--limit N]
  (--encoder vits = 흐름 시험: 작은 인코더를 무작위 초기화로, 가중치 없이 — 수치는 뜻이 없다)
저자 코드 (setup/fetch_code.sh monother, 고정 커밋) 는 고치지 않는다. 우리가 하는 일은 셋뿐이다:
  ① 입력 — 저자 데이터 코드와 같은 열화상 전처리 (their_tone): 사진마다 2–98 백분위로 자르고 0–1 로, 세 채널로 겹친다. MS2 · ViViD 에 저자가 쓴 것이 이것 하나다.
  ② 모델 — 저자 평가 스크립트 (evaluate_rgbt.py) 와 같은 길로 만든다: get_config("zoedepth_rgbt", "eval", …) → build_model. 설정은 MS2 세트에 ms2_teach_nofs,
     나머지 (640×512 영상) 에는 저자가 zero-shot 을 잰 vivid_outdoor. 모델은 열화상 · RGB 두 가지를 함께 받지만 둘 사이에 융합이 없어 (fusion_block_name = "")
     열화상 가지의 출력은 RGB 입력과 무관하다 → RGB 자리에 같은 열화상을 넣고 열화상 가지의 출력만 쓴다 (우리 평가 세트 셋에는 RGB 가 없다).
  ③ 채점 — 우리 평가와 같은 규칙 (src/evaluate.py · common.variants). 저자 코드의 규칙과 같다: 예측을 정답 크기로 (bilinear · align_corners) 올리고 0.001–80 m 로 자름,
     정답 0.001–80 m 인 화소, 사진별 평균, ViViD 는 정답 테두리를 뺌. 저자처럼 한 장씩 (bs 1), 혼합 정밀도 없이.
결과 (--out): metrics.json · per_image.csv · samples/ (src/evaluate.py 와 같은 꼴 — 예측 표본은 우리 팔과 같은 사진이다).
화면: 세트마다 한 줄 + [선행] 표 + 재현 확인 (논문이 보고한 MS2 · ViViD 밤 값과 견줌 — 허용 폭은 README 에 미리 고정).
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import data as D  # noqa: E402
import evaluate as E  # noqa: E402
from common import write_json  # noqa: E402

MS2 = ("test_day", "test_night", "test_rainy")
CONFIG = {True: ("ms2thermal", "ms2_teach_nofs"), False: ("vivid_outdoor", "vivid_outdoor")}   # MS2 세트인가 → (저자 코드의 데이터셋 이름, 설정 판)
PAPER = {"ms2": (0.072, 0.003), "vivid": (0.153, 0.008)}          # 논문 표 I · III 의 AbsRel 과 재현으로 보는 허용 폭 (README '선행 방법' 에 결과 전에 고정)
BASE_NAME = "depth_anything_vitl14.pth"                          # 저자 코드가 모델을 만들 때 ./checkpoints/ 에서 읽는 이름


def their_tone(raw):
    """16 비트 열화상 (H, W) → (3, H, W) float32, 0–1. 저자 코드의 rawthermal_autoScale (zoedepth/data/ms2thermal.py · data_thermal_rgbtd.py) 과 같은 계산:
    사진의 2 · 98 백분위로 자르고 그 사이를 0–1 로. 평균 · 표준편차 정규화는 모델 안에서 한다 (PrepForMidas_Dual)."""
    x = raw.astype(np.float32) / 65535.0
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return np.repeat(x[None], 3, 0)


def small_encoder(code):
    """흐름 시험용: 저자 코드가 고정해 둔 ViT-L 대신 ViT-S 로 만들게 한다 (그 파일은 건드리지 않고 불러온 이름만 바꾼다). 가중치는 싣지 않는다."""
    import functools
    if code not in sys.path:
        sys.path.insert(0, code)
    import zoedepth.models.base_models.depth_anything as da
    if not isinstance(da.DPT_DINOv2_dual, functools.partial):
        da.DPT_DINOv2_dual = functools.partial(da.DPT_DINOv2_dual, encoder="vits")


def build(code, ckpt, base, ms2, dev):
    """저자 코드로 모델을 만든다. ckpt = None 이면 가중치를 싣지 않는다 (흐름 시험). 저자 코드가 ./torchhub · ./checkpoints 를 상대 경로로 읽으므로 그 폴더에서 만든다."""
    os.makedirs(os.path.join(code, "checkpoints"), exist_ok=True)
    link = os.path.join(code, "checkpoints", BASE_NAME)
    if os.path.realpath(link) != os.path.realpath(base):
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(base), link)
    if code not in sys.path:
        sys.path.insert(0, code)
    here = os.getcwd()
    os.chdir(code)
    try:
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config
        dataset, version = CONFIG[ms2]
        kw = {"pretrained_resource": "local::" + os.path.abspath(ckpt)} if ckpt else {"pretrained_resource": ""}
        cfg = get_config("zoedepth_rgbt", "eval", dataset, config_version=version, dataloader_type=2, modality="rgbtd", **kw)
        model = build_model(cfg)
    finally:
        os.chdir(here)
    return model.to(dev).eval(), dataset


def predictor(model, dataset):
    def predict(x, hw):
        out = model(x, x.clone(), dataset=dataset)[0]               # (열화상 가지, RGB 가지) 의 앞쪽 — 저자 평가의 infer 와 같은 호출
        d = out["metric_depth"]
        if d.shape[-2:] != hw:
            d = F.interpolate(d, hw, mode="bilinear", align_corners=True)
        return d[:, 0].float()
    return predict


def table(res):
    """화면의 [선행] 줄들. 재현 확인은 논문이 보고한 두 값 (MS2 세 조건 평균 · ViViD 밤) 과 견준다."""
    f = lambda s: f"{res[s]['raw']['abs_rel']:.4f} / {res[s]['raw']['rmse']:.3f} / {res[s]['raw']['d1']:.4f}" if s in res else "—"  # noqa: E731
    L = ["[선행] MonoTher-Depth (저자 공개 가중치 · MS2 로 학습) raw — AbsRel / RMSE / δ1 | MS2 낮 | 밤 | 비 | ViViD 밤 | FIReStereo | ViViD 실내 | STheReO | ViViD 낮",
         "[선행] mono | " + " | ".join(f(s) for s in MS2 + ("vivid", "fire", "vivid_in", "sthereo", "vivid_day"))]
    m = lambda s: f"{res[s]['median']['abs_rel']:.4f}" if s in res else "—"  # noqa: E731
    L.append("[선행] mono median AbsRel (사진마다 스케일을 맞춤) | " + " | ".join(m(s) for s in MS2 + ("vivid", "fire", "vivid_in", "sthereo", "vivid_day")))
    got = {"ms2": float(np.mean([res[s]["raw"]["abs_rel"] for s in MS2])) if all(s in res for s in MS2) else None,
           "vivid": res["vivid"]["raw"]["abs_rel"] if "vivid" in res else None}
    hits, parts = [], []
    for k, name in (("ms2", "MS2 평균"), ("vivid", "ViViD 밤")):
        want, tol = PAPER[k]
        hits.append(None if got[k] is None else abs(got[k] - want) <= tol)
        parts.append(f"{name} 안 잼" if got[k] is None else f"{name} {got[k]:.4f} (논문 {want:.3f}, 허용 ±{tol:.3f}) {'맞음' if hits[-1] else '다름'}")
    ok = None if None in hits else all(hits)
    verdict = "판정 안 함 (두 세트를 다 재야 한다)" if ok is None else "재현됨 — 빈칸을 이 값으로 채운다" if ok else "재현 안 됨 — 빈칸을 채우지 않는다 (전처리 · 설정을 먼저 본다)"
    L.append("[선행] 재현 확인: " + " · ".join(parts) + " → " + verdict)
    return L, {"got": got, "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--base")
    ap.add_argument("--encoder", default="vitl", choices=["vitl", "vits"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=",".join(E.TEST_SETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev, code = torch.device(a.device), os.path.abspath(a.code)
    if a.encoder == "vits":                                         # 흐름 시험: 빈 가중치 파일을 Depth Anything V1 자리에 둔다 (저자 코드가 strict=False 로 읽는다)
        small_encoder(code)
        a.ckpt, a.base = None, os.path.join(a.out, "empty_base.pth")
        torch.save({}, a.base)
    elif not (a.ckpt and a.base):
        ap.error("--ckpt · --base 가 필요하다")
    sets = [s for s in a.sets.split(",") if s]
    res = {}
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "key", "raw_abs_rel", "raw_rmse", "raw_d1", "median_abs_rel", "median_rmse", "median_d1", "median_scale"])
        for ms2 in (True, False):                                   # 설정 판마다 모델을 한 번 만든다 (입력 크기 맞추기가 설정에 들어 있다)
            todo = [s for s in sets if (s in MS2) == ms2]
            if not todo:
                continue
            model, dataset = build(code, a.ckpt, a.base, ms2, dev)
            print(f"[선행] 모델 — 설정 {CONFIG[ms2][1]} · 파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
            for s in todo:
                res[s], rows = E.eval_set(model, s, dev, a.limit, a.bs, a.workers, False, a.samples, a.out,
                                          ds=D.Thermal(s, a.limit, prep=their_tone), predict=predictor(model, dataset))
                for r in rows:
                    w.writerow([s, r["key"]] + [f"{r[v][k]:.5f}" for v in ("raw", "median") for k in ("abs_rel", "rmse", "d1")] + [f"{r['median']['scale']:.4f}"])
                print("[eval mono] " + E.line(s, res[s]), flush=True)
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    L, check = table(res)
    print("\n".join(L), flush=True)
    write_json(os.path.join(a.out, "metrics.json"), res | {"_check": check})
    return 0


if __name__ == "__main__":
    sys.exit(main())
