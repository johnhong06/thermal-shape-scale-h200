"""결과 표와 판정 — runs/<팔>_s<시드>/eval/metrics.json 을 모아 화면 (이슈 댓글) 에서 바로 읽을 수 있는 표를 찍고 summary/ 에 남긴다.

  python src/report.py summary --runs <runs 폴더> --out <summary 폴더>
화면에는 이름과 숫자만 찍는다 (공개). 줄 수를 46 줄 안으로 (시드 셋일 때 팔 셋 32 줄 · 넷 41 줄) — run.sh 가 화면의 끝 50 줄을 남기므로 마지막 표는 늘 보인다.
팔이 다섯이면 50 줄이라 앞쪽 표가 끝 50 줄에서 밀려난다 — 그때는 작업을 나눠 돌린다 (판정 줄은 맨 끝이라 남는다).
수치 = 시드 평균 ± 표준편차 (시드 2 개 이상, n − 1 로 나눔). 판정 기준은 README '판정 기준' · '본 실험 3' 과 같다 (결과를 보기 전에 고정한 것).
"""
import argparse
import json
import os
import re
import sys

MS2 = ("test_day", "test_night", "test_rainy")
BASE_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "setup", "base_ref.json")   # 앞선 작업의 base 결과 (있을 때만)
GATE_ABSREL = 0.085       # 기준선 재현: MS2 세 조건 평균 raw AbsRel 이 이 값 이하
GAP_ABSREL = 0.05         # zero-shot 격차: 대상의 raw AbsRel − MS2 평균 raw AbsRel 이 이 값 이상
ZS = (("ViViD 밤", "vivid"), ("FIReStereo", "fire"), ("ViViD 실내", "vivid_in"), ("STheReO", "sthereo"))     # zero-shot 대상 (이름, 세트)
# 잡음 폭의 바닥 — 앞선 두 실험에서 base 를 여섯 번 돌려 본 raw AbsRel 의 범위 (README '기준 수치'). 같은 작업의 시드 범위가 이보다 좁으면 이 값을 쓴다.
# 새 대상 둘은 앞선 값이 없어 같은 작업의 시드 범위를 쓰되, 학습과 같은 데이터인 MS2 의 바닥보다 좁게 잡지는 않는다 (세 시드의 범위가 우연히 아주 좁을 때를 막는다).
NOISE_FLOOR = {"ms2": 0.0013, "vivid": 0.0066, "fire": 0.0256}
CLAIM_MIN = 2             # '여러 곳' = 대상 넷 가운데 이만큼 이상에서 개선이고 나빠진 곳이 없을 때


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
    if "base" not in R and os.path.exists(BASE_REF):               # 이 작업에서 base 를 안 돌렸으면 앞선 작업의 base 를 견줄 기준으로 쓴다 (새 대상의 값은 없다)
        spec = json.load(open(BASE_REF, encoding="utf-8"))
        R["base"] = {int(s): r for s, r in spec["seeds"].items()}
        ref["base"] = f" ({spec['source']})"
    os.makedirs(out, exist_ok=True)
    L = []
    order = ("base", "sep", "ours", "pl", "plv")
    arms = [a for a in order if a in R] + sorted(a for a in R if a not in order)
    zs_names = " | ".join(n for n, _ in ZS)
    L.append("[표1] MS2 test (학습과 같은 데이터) · raw — AbsRel / RMSE / δ1, 시드 평균±표준편차")
    for a in arms:
        L.append(f"[표1] {a:5s} 시드 {len(R[a])}{ref.get(a, str())} | 낮 {cell(R[a], ['test_day'], 'raw')} | 밤 {cell(R[a], ['test_night'], 'raw')} | "
                 f"비 {cell(R[a], ['test_rainy'], 'raw')} | 평균 {cell(R[a], MS2, 'raw')}")
    L.append(f"[표2] 처음 보는 대상 · raw (그대로) — AbsRel / RMSE / δ1 | {zs_names}")
    for a in arms:
        L.append(f"[표2] {a:5s} | " + " | ".join(cell(R[a], [k], "raw") for _, k in ZS))
    L.append(f"[표3] median (사진마다 스케일을 맞춤 — 모양의 오차) — AbsRel / RMSE / δ1 | MS2 평균 | {zs_names}")
    for a in arms:
        L.append(f"[표3] {a:5s} | {cell(R[a], MS2, 'median')} | " + " | ".join(cell(R[a], [k], "median") for _, k in ZS))
    L.append(f"[표4] 스케일 오차 — 사진별 |log (정답 중앙값 / 예측 중앙값)| 의 중앙값 / 치우침 (+ 면 실제보다 가깝게 예측) | MS2 평균 | {zs_names}")
    for a in arms:
        L.append(f"[표4] {a:5s} | " + " | ".join(cell(R[a], sets, "scale", ("abs_med", "bias")) for sets in [MS2] + [[k] for _, k in ZS]))
    L.append(f"[표5] 시드별 AbsRel — MS2 평균 raw | 대상마다 raw · median ({zs_names}) | ViViD 낮 raw · FIReStereo raw 낮 · 밤비 | 학습 시간 (분)")
    for a in arms:
        for s, r in sorted(R[a].items()):
            f = lambda sets, var="raw": "—" if get(r, sets, var, "abs_rel") is None else f"{get(r, sets, var, 'abs_rel'):.4f}"  # noqa: E731
            L.append(f"[표5] {a}_s{s} | {f(MS2)} | " + " | ".join(f"{f([k])} · {f([k], 'median')}" for _, k in ZS)
                     + f" | {f(['vivid_day'])} · {f(['fire/day'])} · {f(['fire/night'])} | {r.get('_min')}")

    J = {}
    if "base" in R:
        vals = lambda arm, sets: [v for v in (get(r, sets, "raw", "abs_rel") for r in R[arm].values()) if v is not None]  # noqa: E731
        ms2 = vals("base", MS2)
        zs = {k: vals("base", [k]) for _, k in ZS}
        if ms2:
            mu = stat(ms2)[0]
            J["gate"] = {"ms2_abs_rel": mu, "ok": mu <= GATE_ABSREL}
            L.append(f"[판정] 기준선 재현: base MS2 평균 raw AbsRel {mu:.4f} (기준 ≤ {GATE_ABSREL}) → " + ("재현됨" if mu <= GATE_ABSREL else "재현 안 됨 — 학습 설정을 먼저 본다"))
            gaps = [(n, k, stat(zs[k])[0] - mu) for n, k in ZS if zs[k]]
            J["gap"] = {k: {"gap": g, "ok": g >= GAP_ABSREL} for _, k, g in gaps}
            if gaps:
                L.append(f"[판정] zero-shot 격차 (대상 − MS2 {mu:.4f}, 기준 ≥ {GAP_ABSREL}): "
                         + " · ".join(f"{n} {g:+.4f} {'있음' if g >= GAP_ABSREL else '작음'}" for n, _, g in gaps))
        for a in [x for x in arms if x != "base"]:
            m2 = vals(a, MS2)
            if not (ms2 and m2):
                continue
            d_ms2 = stat(m2)[0] - stat(ms2)[0]
            judged = min(len(ms2), len(m2)) >= 2
            nm = max(stat(ms2)[2], NOISE_FLOOR["ms2"])               # 잡음 폭 = base 의 시드 범위 (최대 − 최소), 바닥은 앞선 실험에서 본 범위
            keep = d_ms2 <= nm
            J[a] = {"d_ms2": d_ms2, "noise_ms2": nm, "ms2_kept": keep if judged else None, "targets": {}}
            parts, good, bad = [], 0, 0
            for n, k in ZS:
                z2 = vals(a, [k])
                if not (zs[k] and z2):
                    continue
                d = stat(z2)[0] - stat(zs[k])[0]
                nz = max(stat(zs[k])[2], NOISE_FLOOR.get(k, NOISE_FLOOR["ms2"]))
                v = "판정 안 함" if not judged else "개선" if d < -nz else "나빠짐" if d > nz else "차이 없음"
                good, bad = good + (v == "개선"), bad + (v == "나빠짐")
                J[a]["targets"][k] = {"d": d, "noise": nz, "verdict": v}
                parts.append(f"{n} {d:+.4f} {v} (잡음 {nz:.4f})")
            L.append(f"[판정] {a} − base raw AbsRel: " + " · ".join(parts))
            if not judged:
                L.append(f"[판정] {a}: MS2 평균 {d_ms2:+.4f} — 시드가 하나라 판정하지 않음")
                continue
            head = f"[판정] {a}: MS2 평균 {d_ms2:+.4f} (잡음 {nm:.4f}) {'유지' if keep else '나빠짐'}"
            if a == "sep":                                        # sep 의 기준은 MS2 유지뿐이다 (구조만 바꾼 줄)
                L.append(head + " → " + ("구조를 바꿔도 MS2 점수 유지" if keep else "구조를 바꾸자 MS2 점수가 나빠짐"))
                continue
            if len(J[a]["targets"]) < len(ZS):                    # base 의 값이 없는 대상이 있으면 (앞선 작업의 base 를 기준으로 쓸 때) '여러 곳' 은 말하지 않는다
                L.append(head + f" · 개선 {good} 곳 · 나빠짐 {bad} 곳 → 대상 {len(ZS)} 곳 가운데 {len(J[a]['targets'])} 곳만 견줄 수 있어 '여러 곳' 은 판정하지 않음")
                continue
            claim = keep and bad == 0 and good >= CLAIM_MIN
            J[a]["claim"] = claim
            L.append(head + f" · 개선 {good} 곳 · 나빠짐 {bad} 곳 → "
                     + (f"여러 곳에서 개선 (기준: MS2 유지 · 나빠진 곳 없음 · 개선 {CLAIM_MIN} 곳 이상)" if claim else "주장의 기준에 못 미침"))
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
