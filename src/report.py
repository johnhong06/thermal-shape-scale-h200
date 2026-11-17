"""결과 표와 판정 — runs/<팔>_s<시드>/eval/metrics.json 을 모아 화면 (이슈 댓글) 에서 바로 읽을 수 있는 표를 찍고 summary/ 에 남긴다.

  python src/report.py summary --runs <runs 폴더> --out <summary 폴더>
화면에는 이름과 숫자만 찍는다 (공개). 줄 수를 46 줄 안으로 (팔 셋 × 시드 셋일 때 37 줄) — run.sh 가 화면의 끝 50 줄을 남기므로 마지막 표는 늘 보인다.
팔이 넷을 넘으면 (넷 47 줄) 앞쪽 표가 끝 50 줄에서 밀려난다 — 그때는 작업을 나눠 돌리거나 표를 줄인다 (판정 줄은 맨 끝이라 남는다).
수치 = 시드 평균 ± 표준편차 (시드 2 개 이상, n − 1 로 나눔). 판정 기준은 README '판정 기준' 과 같다 (결과를 보기 전에 고정한 것).
"""
import argparse
import json
import os
import re
import sys

MS2 = ("test_day", "test_night", "test_rainy")
BASE_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "setup", "base_ref.json")   # 앞선 작업의 base 결과 (있을 때만)
GATE_ABSREL = 0.085       # 기준선 재현: MS2 세 조건 평균 raw AbsRel 이 이 값 이하
GAP_ABSREL = 0.05         # zero-shot 격차: 대상 (FIReStereo · ViViD 밤) 의 raw AbsRel − MS2 평균 raw AbsRel 이 이 값 이상
ZS = (("FIReStereo", ["fire"]), ("ViViD 밤", ["vivid"]))     # zero-shot 대상 (이름, 세트)


def load(runs):
    out = {}
    for d in sorted(os.listdir(runs) if os.path.isdir(runs) else []):
        m = re.fullmatch(r"([a-z]+)_s(\d+)", d)
        p = os.path.join(runs, d, "eval", "metrics.json")
        if m and os.path.exists(p):
            r = json.load(open(p, encoding="utf-8"))
            done = os.path.join(runs, d, "done.json")
            r["_min"] = json.load(open(done)).get("minutes") if os.path.exists(done) else None
            out.setdefault(m.group(1), {})[int(m.group(2))] = r
    return out


def stat(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else None
    return mu, sd, (max(xs) - min(xs))


def get(r, sets, var, key):
    """세트 여러 개면 세트 평균 (MS2 세 조건 평균 — 공식 벤치마크 README 의 방식). 'fire/day' 꼴은 FIReStereo 의 낮 · 밤비."""
    vals = []
    for s in sets:
        node = r.get(s.split("/")[0])
        if node is not None and "/" in s:
            node = node.get(s.split("/")[1])
        val = (node or {}).get(var, {}).get(key)                    # 앞선 작업의 기준 (base_ref.json) 에는 일부 값만 있다
        if val is None:
            return None
        vals.append(val)
    return sum(vals) / len(vals)


def cell(seeds, sets, var, keys=("abs_rel", "rmse", "d1")):
    parts = []
    for k in keys:
        xs = [v for v in (get(r, sets, var, k) for r in seeds.values()) if v is not None]
        if not xs:
            parts.append("—")
            continue
        mu, sd, _ = stat(xs)
        prec = 3 if k == "rmse" else 4
        parts.append(f"{mu:.{prec}f}" + (f"±{sd:.{prec}f}" if sd is not None else ""))
    return "—" if all(p == "—" for p in parts) else " / ".join(parts)


def summary(runs, out):
    R = load(runs)
    if not R:
        print("[표] 아직 끝난 평가가 없음")
        return 0
    ref = {}
    if "base" not in R and os.path.exists(BASE_REF):               # 이 작업에서 base 를 안 돌렸으면 앞선 작업의 base 를 견줄 기준으로 쓴다
        spec = json.load(open(BASE_REF, encoding="utf-8"))
        R["base"] = {int(s): r for s, r in spec["seeds"].items()}
        ref["base"] = f" ({spec['source']})"
    os.makedirs(out, exist_ok=True)
    L = []
    order = ("base", "pl", "plv")
    arms = [a for a in order if a in R] + sorted(a for a in R if a not in order)
    L.append("[표1] MS2 test (학습과 같은 데이터) · raw — AbsRel / RMSE / δ1, 시드 평균±표준편차")
    for a in arms:
        L.append(f"[표1] {a:5s} 시드 {len(R[a])}{ref.get(a, str())} | 낮 {cell(R[a], ['test_day'], 'raw')} | 밤 {cell(R[a], ['test_night'], 'raw')} | "
                 f"비 {cell(R[a], ['test_rainy'], 'raw')} | 평균 {cell(R[a], MS2, 'raw')}")
    L.append("[표2] MS2 test 평균 · median (중앙값 맞춤 — 공식 벤치마크 방식) — AbsRel / RMSE / δ1")
    for a in arms:
        L.append(f"[표2] {a:5s} | {cell(R[a], MS2, 'median')}")
    L.append("[표3] FIReStereo test (zero-shot, 학습에 안 씀) — AbsRel / RMSE / δ1")
    for a in arms:
        L.append(f"[표3] {a:5s} | raw {cell(R[a], ['fire'], 'raw')} | focal {cell(R[a], ['fire'], 'focal')} | median {cell(R[a], ['fire'], 'median')} | "
                 f"raw 낮 {cell(R[a], ['fire/day'], 'raw')} | raw 밤비 {cell(R[a], ['fire/night'], 'raw')}")
    L.append("[표5] ViViD 야외 (zero-shot, 정답 테두리 뺌) — AbsRel / RMSE / δ1. 선행 표 (MonoTher-Depth, 밤 raw): DepthAnything 0.166 / 4.929 / 0.719 · 그 방법 0.153 / 4.440 / 0.768")
    for a in arms:
        L.append(f"[표5] {a:5s} | 밤 raw {cell(R[a], ['vivid'], 'raw')} | 밤 median {cell(R[a], ['vivid'], 'median')} | 낮 raw {cell(R[a], ['vivid_day'], 'raw')}")
    L.append("[표6] 스케일 오차 — 사진별 |log (정답 중앙값 / 예측 중앙값)| 의 중앙값 / 치우침 (+ 면 실제보다 가깝게 예측) · MS2 평균 | FIReStereo | ViViD 밤")
    for a in arms:
        sc = lambda sets: cell(R[a], sets, "scale", ("abs_med", "bias"))  # noqa: E731
        L.append(f"[표6] {a:5s} | {sc(MS2)} | {sc(['fire'])} | {sc(['vivid'])}")
    L.append("[표4] 시드별 raw AbsRel — MS2 낮 · 밤 · 비 · 평균 | FIReStereo raw · median | ViViD 밤 raw · median · 낮 raw | 학습 시간 (분)")
    for a in arms:
        for s, r in sorted(R[a].items()):
            f = lambda sets, var="raw": "—" if get(r, sets, var, "abs_rel") is None else f"{get(r, sets, var, 'abs_rel'):.4f}"  # noqa: E731
            L.append(f"[표4] {a}_s{s} | {f(['test_day'])} · {f(['test_night'])} · {f(['test_rainy'])} · {f(MS2)} | {f(['fire'])} · {f(['fire'], 'median')} | "
                     f"{f(['vivid'])} · {f(['vivid'], 'median')} · {f(['vivid_day'])} | {r.get('_min')}")

    J = {}
    if "base" in R:
        ms2 = [v for v in (get(r, MS2, "raw", "abs_rel") for r in R["base"].values()) if v is not None]
        zs = {n: [v for v in (get(r, sets, "raw", "abs_rel") for r in R["base"].values()) if v is not None] for n, sets in ZS}
        if ms2:
            mu = stat(ms2)[0]
            J["gate"] = {"ms2_abs_rel": mu, "ok": mu <= GATE_ABSREL}
            L.append(f"[판정] 기준선 재현: base MS2 평균 raw AbsRel {mu:.4f} (기준 ≤ {GATE_ABSREL}) → " + ("재현됨" if mu <= GATE_ABSREL else "재현 안 됨 — 학습 설정을 먼저 본다"))
        J["gap"] = {}
        for n, _ in ZS:
            if ms2 and zs[n]:
                gap = stat(zs[n])[0] - stat(ms2)[0]
                J["gap"][n] = {"gap": gap, "ok": gap >= GAP_ABSREL}
                L.append(f"[판정] zero-shot 격차 ({n}): {stat(zs[n])[0]:.4f} − MS2 {stat(ms2)[0]:.4f} = {gap:+.4f} (기준 ≥ {GAP_ABSREL}) → "
                         + ("격차 있음" if gap >= GAP_ABSREL else "격차 작음"))
        for a in [x for x in arms if x != "base"]:
            m2 = [v for v in (get(r, MS2, "raw", "abs_rel") for r in R[a].values()) if v is not None]
            J[a] = {}
            for n, sets in ZS:
                z2 = [v for v in (get(r, sets, "raw", "abs_rel") for r in R[a].values()) if v is not None]
                if not (zs[n] and z2 and ms2 and m2):
                    continue
                d_zs, d_ms2 = stat(z2)[0] - stat(zs[n])[0], stat(m2)[0] - stat(ms2)[0]
                if min(len(zs[n]), len(z2)) < 2:
                    verdict = "시드가 하나라 판정하지 않음"
                else:
                    nz, nm = stat(zs[n])[2], stat(ms2)[2]            # 잡음 폭 = base 의 시드 범위 (최대 − 최소)
                    verdict = ("개선" if d_zs < -nz and d_ms2 <= nm else "MS2 에서 나빠짐" if d_zs < -nz else "나빠짐" if d_zs > nz else "차이 없음 (잡음 안)")
                    verdict += f" · 잡음 폭 {nz:.4f} · MS2 {nm:.4f}"
                J[a][n] = {"d_zs": d_zs, "d_ms2": d_ms2, "verdict": verdict}
                L.append(f"[판정] {a} − base ({n}): raw AbsRel {d_zs:+.4f} · MS2 평균 {d_ms2:+.4f} → {verdict}")
    print("\n".join(L), flush=True)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": R, "judge": J}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["summary"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit(summary(a.runs, a.out))
