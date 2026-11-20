#!/usr/bin/env bash
# H200 진입점 (gpu-request 이슈의 실행 명령 한 줄) — 열화상 한 장 깊이 추정이 학습에 안 쓴 열화상 데이터 (ViViD · FIReStereo · STheReO) 에서도 되게:
#   기준선 · 모양 + 스케일 학생 · 비교 팔 · 선생 점검.
#   bash run.sh check                     # 1–2 분: 파이썬 · torch · GPU · CPU · RAM · 경로 여유 · 쓰기 가능 · 올려 둔 묶음 · 외부 접속. 설치 안 함
#   bash run.sh smoke                     # 끝까지 아주 작게: 작은 묶음 (학습 48 장 · 평가 세트마다 8–16 장) · 1 epoch · 시드 0, 모든 단계
#   bash run.sh all [ARMS=…] [SEEDS=…]    # 준비 → 데이터 → (시드마다 팔마다) 학습 · 평가 → 표 → results.zip (실행 하나가 끝날 때마다 표와 묶음을 다시 만듦)
#   bash run.sh eval FROM=<앞선 작업 이름>  # 학습 없이 평가만: 앞선 작업이 CKPT=keep 으로 남긴 모델 (/app/output/<작업>/ckpt/*.pt) 을 지금의 평가 세트로 다시 잰다
#                                         #   — 평가 데이터를 나중에 더했을 때 쓴다. SETS=<세트,…> 로 일부만
#   bash run.sh teacher [LIMIT=N]         # 선생 점검 (학습 없음, README '선생 점검'): MS2 의 RGB 에서 두 선생 (Depth Anything V2 · UniDepthV2) 을 LiDAR 정답에 대 본다.
#                                         #   KIND=smoke 면 16 장짜리 작은 묶음으로. 표와 판정이 화면 끝에 나온다
#   bash run.sh relcheck                  # 상대 깊이 모델 점검 (학습 없음, README '열화상 판 TR2M' 단계 1): Depth Anything V2 상대 깊이 ViT-L · ViT-S 를 고정한 채
#                                         #   MS2 val 의 RGB · 열화상에서 모양 오차를 잰다. KIND=smoke 면 작은 묶음으로
#   bash run.sh setup [DRY_RUN=1] | data  # 한 단계만 (작업끼리 파일이 안 이어지므로 data 는 setup 도 먼저 한다)
# KEY=값 (이슈 한 줄 명령에는 환경 변수를 못 넣으므로 인자로):
#   ARMS=base,sep,ours   돌릴 팔 (쉼표, 안 주면 이 셋). base = 기준선 · sep = 모양 + 스케일 학생을 MS2 만으로 ·
#                      ours = sep + 다른 데이터의 쌍 넷에 선생 (UniDepthV2) 의 모양 · 스케일 라벨. README '팔' · '본 실험 3'.
#                      앞선 비교 팔도 그대로 돈다: pl = 쌍 (KAIST · LLVIP) 에 모양만 · plv = 같은 방식을 ViViD 낮 쌍으로.
#                      tr = 열화상 판 TR2M 을 MS2 만으로 (고정한 공식 상대 깊이 모델 + 작은 모듈, README '열화상 판 TR2M')
#                      rt = 같은 꼴인데 상대 깊이 모델을 먼저 MS2 열화상으로 학습해 고정한다 (src/train_rel.py → src/train.py, 두 단계)
#                      rtp = rt + 쌍 (1 단계에 선생의 모양, 2 단계에 선생의 절대 깊이) — 우리 방법 · mtd = base + 쌍의 선생 절대 깊이 (기존 방식의 비교). README '본 실험 4'
#                      ours · rtp · mtd 가 들어 있으면 학습 전에 선생 라벨을 한 번 뽑는다 (src/labels.py, 37,233 장) — 환경은 선생 점검의 것 (envs/teacher.txt) 을 쓴다
#   SETS=<세트,…>      평가할 세트 (all · smoke · eval 모드). 안 주면 전부. 단계 2 는 MS2 test 만 (처음 보는 대상은 설계에 쓰지 않으므로 보지 않는다)
#   SEEDS=0,1,2   EPOCHS=10   BS=16   PAIR_BS=8 (pl · plv · ours · rtp · mtd 의 쌍 배치, 선생 라벨 단계도 한 번에 이만큼)   AMP=1 (bf16 혼합 정밀도, 0 = 끔)   LIMIT=0 (학습 장 수 제한)   WORKERS=8 (사진을 읽는 작업자 수)
#   CKPT=none|keep|seed0|all   학습한 모델 (final.pt, 하나에 1.3 GB). none = 버림 (기본) · keep = 결과 폴더의 ckpt/ 에 남김 (results.zip 에는 안 넣음 — 나중에 eval 모드로 씀) ·
#                              seed0 · all = results.zip 에 넣음
#   STALL_MIN=40 (로그가 이만큼 그대로면 멈춘 것으로 보고 끔)   PROGRESS_SEC=1800   RETRY=2 (학습이 죽으면 last.pt 에서 다시)
#   LINKS=<Drive 파일 ID> 데이터 묶음을 받을 곳의 목록 파일. 묶음과 받는 주소는 저장소에 없다 (다시 배포하면 안 되는 데이터) — 이 값은 실행 명령에만 적는다.
#         /app/data 에 묶음이 올라가 있으면 없어도 된다.
#   JOB=<결과 폴더 이름>   DRY_RUN=1 (setup: 큰 것을 받지 않음)
#   (로컬 시험) WORK=<작업 폴더>  PY=<쓸 파이썬>  APP_ROOT=<가짜 /app>  DATA_DIR=<묶음이 있는 폴더>  HF_DIR=<HF 캐시>
#               ENC=vits (작은 인코더를 무작위 초기화로 — 메모리 1 GB 남짓으로 모든 단계의 흐름만 본다. 수치는 뜻이 없다)
# 경로 (H200): /app/data 읽기 전용 (올려 둔 묶음이 있으면 그것을 쓴다) · /app/scratch/thermal = 작업 폴더 (환경 · 캐시 · 코드 · 데이터 · 체크포인트, 작업마다 비워짐)
#   /app/output/<작업>/ = 결과만: logs/ · runs/ · summary/ · results.zip
# 화면 출력 = 짧은 요약 (앞 150 줄 + 끝 50 줄, 65,000 자 안) — 결과 표가 늘 끝에 오게 한다. 이름과 숫자만 찍는다 (이슈 댓글은 공개).
#   LINKS 값 · Drive 주소 · 파일 ID 는 화면에 찍지 않는다. 토큰은 쓰지 않는다.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
REPO=$PWD
MODE=${1:-}
[ $# -gt 0 ] && shift
MODES="check setup data smoke all eval teacher relcheck"
[[ " $MODES " == *" $MODE "* ]] && [ -n "$MODE" ] || { echo "!!! 모드: ${MODES// / | } (받은 값: '$MODE'). 사용법은 run.sh 맨 위"; exit 2; }
KEYS="ARMS SEEDS EPOCHS BS PAIR_BS AMP LIMIT WORKERS ENC LINKS FROM SETS KIND CKPT STALL_MIN PROGRESS_SEC RETRY JOB DRY_RUN WORK PY APP_ROOT DATA_DIR HF_DIR"
for k in $KEYS; do unset "$k"; done            # 컨테이너 환경 변수와 섞이지 않게 — 값은 KEY=값 인자로만
for a in "$@"; do
  k=${a%%=*}
  if [ "$k" = "$a" ] || [[ " $KEYS " != *" $k "* ]]; then echo "!!! 모르는 인자: ${a%%=*} (KEY=값 꼴, KEY = $KEYS)"; exit 2; fi
  printf -v "$k" '%s' "${a#*=}"
done
INT='^[0-9]+$'; NAME='^[A-Za-z0-9_.+-]+$'; LIST='^[a-z0-9,]*$'
chk() { [ -z "${!1:-}" ] || [[ ${!1} =~ $2 ]] || { echo "!!! $1=${!1} — 형식이 맞지 않음"; exit 2; }; }
chk ARMS "$LIST"; chk SEEDS "$LIST"; chk EPOCHS "$INT"; chk BS "$INT"; chk PAIR_BS "$INT"; chk LIMIT "$INT"; chk WORKERS "$INT"; chk STALL_MIN "$INT"; chk PROGRESS_SEC "$INT"; chk RETRY "$INT"; chk JOB "$NAME"
[ -z "${LINKS:-}" ] || [[ $LINKS =~ ^[A-Za-z0-9_-]{20,80}$ ]] || { echo "!!! LINKS — Drive 파일 ID 꼴이 아님"; exit 2; }
case ${AMP:-1} in 0|1) ;; *) echo "!!! AMP = 0 | 1"; exit 2;; esac
case ${ENC:-vitl} in vitl|vits) ;; *) echo "!!! ENC = vitl | vits"; exit 2;; esac
case ${CKPT:-none} in none|keep|seed0|all) ;; *) echo "!!! CKPT = none | keep | seed0 | all"; exit 2;; esac
chk FROM "$NAME"; [ -z "${SETS:-}" ] || [[ $SETS =~ ^[a-z_,]+$ ]] || { echo "!!! SETS — 세트 이름을 쉼표로"; exit 2; }
case ${KIND:-} in ""|smoke|all) ;; *) echo "!!! KIND = smoke | all"; exit 2;; esac
[ "$MODE" != eval ] || [ -n "${FROM:-}" ] || { echo "!!! eval 모드에는 FROM=<앞선 작업 이름> 이 필요"; exit 2; }
case ${DRY_RUN:-0} in 0) ;; 1) [ "$MODE" = setup ] || { echo "!!! DRY_RUN=1 은 setup 모드에만"; exit 2; };; *) echo "!!! DRY_RUN = 0 | 1"; exit 2;; esac
_a=${ARMS:-}; for x in ${_a//,/ }; do case $x in base|pl|plv|sep|ours|tr|rt|rtp|mtd) ;; *) echo "!!! ARMS 의 '$x' — base | sep | ours | tr | rt | rtp | mtd | pl | plv"; exit 2;; esac; done
if [ "$MODE" = smoke ]; then KIND=${KIND:-smoke}; SEEDS=${SEEDS:-0}; EPOCHS=${EPOCHS:-1}; fi
KIND=${KIND:-all}; FROM=${FROM:-}; SETS=${SETS:-}
ARMS=${ARMS:-base,sep,ours}; SEEDS=${SEEDS:-0,1,2}; EPOCHS=${EPOCHS:-10}; BS=${BS:-16}; PAIR_BS=${PAIR_BS:-8}; AMP=${AMP:-1}; LIMIT=${LIMIT:-0}; WORKERS=${WORKERS:-8}; ENC=${ENC:-vitl}
CKPT=${CKPT:-none}; STALL_MIN=${STALL_MIN:-40}; PROGRESS_SEC=${PROGRESS_SEC:-1800}; RETRY=${RETRY:-2}; DRY_RUN=${DRY_RUN:-0}
PY=${PY:-}; DATA_DIR=${DATA_DIR:-}; LINKS=${LINKS:-}
[ -n "$SEEDS" ] || { echo "!!! SEEDS 가 비어 있음"; exit 2; }
SEED0=${SEEDS%%,*}
STALL=$((STALL_MIN * 60))

# ───────────────────────── 경로 ─────────────────────────
APP=${APP_ROOT:-/app}
if [ -d "$APP" ]; then H200=1; else H200=0; fi
if [ -n "${WORK:-}" ]; then OUTROOT=$WORK/output
elif [ $H200 = 1 ]; then WORK=$APP/scratch/thermal; OUTROOT=$APP/output
else WORK=$REPO/.work; OUTROOT=$REPO/results; fi
[ $H200 = 1 ] && OUTROOT=$APP/output
mkdir -p "$WORK" || { echo "!!! 작업 폴더를 못 만듦"; exit 1; }
WORK=$(cd "$WORK" && pwd)
JOB=${JOB:-${MODE}_$(date +%m%d_%H%M%S)}
OUTJ=$OUTROOT/$JOB; LOGS=$OUTJ/logs; RUNS=$OUTJ/runs; SUMM=$OUTJ/summary
mkdir -p "$LOGS" || { echo "!!! 결과 폴더를 못 만듦: $OUTJ"; exit 1; }
CONSOLE=$LOGS/console.log
# 화면 = 앞 150 줄 (40,000 자 안) + 끝 50 줄 (줄마다 400 자) — 이슈 댓글 65,000 자 한도. 전체는 console.log (tee).
exec > >(tee -a "$CONSOLE" | awk -v HL=150 -v HC=40000 -v TL=50 -v W=400 '
  { if (length($0) > W) $0 = substr($0, 1, W) " …"
    if (!cut && n < HL && c + length($0) + 1 <= HC) { print; fflush(); n++; c += length($0) + 1; next }
    cut = 1; k++; buf[k % TL] = $0 }
  END { if (k > TL) print "[… " (k - TL) " 줄 생략 — 전체는 logs/console.log]"
        for (i = (k > TL ? k - TL + 1 : 1); i <= k; i++) print buf[i % TL] }') 2>&1
TEE=$!
CUR=""; STOP=0; T_START=$(date +%s)
finish() {
  local rc=$?
  echo "[run] $(date '+%F %T') 끝 — 모드 $MODE, 종료 코드 $rc, $(( ($(date +%s) - T_START) / 60 )) 분. 결과 $OUTJ"
  exec >&- 2>&-
  wait "$TEE" 2>/dev/null   # H200 의 bash 5.1 은 인자 없는 wait 가 tee 까지 기다려 멈춘다 → PID 를 준다
  exit "$rc"
}
trap finish EXIT
trap 'STOP=1; echo "[run] 멈춤 신호 — 지금 단계를 정리하고 결과를 묶는다"; [ -n "$CUR" ] && kill -TERM "$CUR" 2>/dev/null' TERM INT
SYSPY=$(command -v python3 || echo python3)
SETSID=$(command -v setsid || true)

export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_DIR:-$WORK/hf} TORCH_HOME=$WORK/torch PIP_CACHE_DIR=$WORK/pip UV_CACHE_DIR=$WORK/uv/cache UV_PYTHON_INSTALL_DIR=$WORK/uv/python XDG_CACHE_HOME=$WORK/cache
export TD_CODE=$WORK/code/Depth-Anything-V2 TD_UNIDEPTH=$WORK/code/UniDepth TD_DATA=$WORK/data
CK=$WORK/ck; STATE=$WORK/state
[ "$MODE" = check ] || mkdir -p "$CK" "$STATE" "$RUNS" "$SUMM"
under() { local p q; p=$(realpath -m "$1"); q=$(realpath -m "$2"); [ "$p" = "$q" ] || [[ $p == "$q"/* ]]; }
for v in WORK HF_HOME UV_CACHE_DIR TD_CODE TD_DATA CK; do
  under "${!v}" "$OUTROOT" && { echo "!!! $v 가 결과 전용 경로 ($OUTROOT) 아래 — 캐시 · 데이터 · 체크포인트는 작업 폴더에. 중단"; exit 2; }
done
echo "[run] $(date '+%F %T') 모드 $MODE · $([ $H200 = 1 ] && echo "H200 ($APP)" || echo 로컬) · 커밋 $(git rev-parse --short HEAD 2>/dev/null || echo '?')" \
     "· 작업 폴더 여유 $(df -h "$WORK" 2>/dev/null | awk 'NR==2 {print $4}') · 결과 $OUTJ · 결과 경로에 보이는 다른 작업 폴더 $(( $(find "$OUTROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l) - 1 )) 개"
[ "$MODE" = check ] || echo "[run] ARMS=$ARMS · SEEDS=$SEEDS · EPOCHS=$EPOCHS · BS=$BS · PAIR_BS=$PAIR_BS · AMP=$AMP · LIMIT=$LIMIT · WORKERS=$WORKERS · CKPT=$CKPT$([ "$ENC" = vitl ] || echo " · ENC=$ENC (무작위 초기화 — 흐름 시험)")"

# ───────────────────────── 단계 실행기: 멈춤 감지 ─────────────────────────
brief_fail() {  # 실패한 로그에서 마지막 예외 한 줄과 위치 (파일:줄 (함수)) 만 — 코드 내용은 찍지 않는다 (공개 댓글)
  "$SYSPY" -I -B - "$1" <<'PY'
import re, sys
try:
    L = open(sys.argv[1], errors="replace").read().splitlines()
except OSError:
    sys.exit()
tb = [i for i, x in enumerate(L) if x.startswith("Traceback")]
if not tb:
    for x in [x for x in L if x.strip()][-3:]:
        print("  로그 끝: " + x.strip()[:300])
    sys.exit()
body = L[tb[-1] + 1:]
frames = [re.sub(r'^\s*File "(?:.*/)?([^/"]+)", line (\d+), in (.*)$', r"\1:\2 (\3)", x) for x in body if x.lstrip().startswith('File "')]
exc = [x for x in body if x and not x.startswith((" ", "\t"))]
print("  예외: " + (exc[-1][:300] if exc else "?"))
for f in frames[-4:]:
    print("  위치: " + f[:200])
PY
}
run_step() {  # 이름 로그 보이기(show|quiet) 명령… → 종료 코드 (124 = 멈춤 감지로 끔, 130 = 멈춤 신호). WATCH=<파일> 이면 그 크기도 '진행' 으로 본다
  local name=$1 log=$2 show=$3; shift 3
  local tp t=0 idle=0 last=-1 sz rc="" seen n lastshow=0 new
  [ "$STOP" = 1 ] && { echo "[$name] 건너뜀 (멈춤 신호)"; return 130; }
  touch "$log"; seen=$(wc -l < "$log")
  $SETSID "$@" >> "$log" 2>&1 < /dev/null &
  tp=$!; CUR=$tp
  while kill -0 "$tp" 2>/dev/null; do
    sleep 5; t=$((t + 5))
    # shellcheck disable=SC2086
    sz=$(stat -c %s "$log" ${WATCH:-} 2>/dev/null | awk '{ s += $1 } END { print s + 0 }')
    if [ "$sz" = "$last" ]; then idle=$((idle + 5)); else idle=0; last=$sz; fi
    n=$(wc -l < "$log")
    if [ "$n" -gt "$seen" ]; then
      new=$(tail -n "+$((seen + 1))" "$log" | head -n $((n - seen))); seen=$n
      [ "$show" = show ] && printf '%s\n' "$new" | grep -aE '^(\[(train|eval|data|labels|code|weights|env|teacher|relcheck|판정|표[0-9]?)|!!!)' | cut -c1-400 && lastshow=$t
    fi
    if [ $((t - lastshow)) -ge "$PROGRESS_SEC" ]; then echo "[$name] $(date +%H:%M) 진행 중 $((t / 60)) 분 · 로그 변화 없음 $((idle / 60)) 분"; lastshow=$t; fi
    if [ "$STOP" = 1 ]; then rc=130; break; fi
    if [ "$idle" -ge "$STALL" ]; then
      echo "[$name] !!! 로그가 ${STALL_MIN} 분 동안 그대로 — 멈춘 것으로 보고 끈다"
      kill -TERM -- "-$tp" 2>/dev/null; kill -TERM "$tp" 2>/dev/null; sleep 30
      kill -KILL -- "-$tp" 2>/dev/null; kill -KILL "$tp" 2>/dev/null; rc=124; break
    fi
  done
  if [ "${rc:-}" = 130 ]; then
    for ((n = 0; n < 120; n += 5)); do kill -0 "$tp" 2>/dev/null || break; sleep 5; done
    kill -KILL -- "-$tp" 2>/dev/null; kill -KILL "$tp" 2>/dev/null
  fi
  wait "$tp" 2>/dev/null; n=$?
  CUR=""; rc=${rc:-$n}
  [ "$rc" = 0 ] || { echo "[$name] $(date +%H:%M:%S) 실패 — 종료 $rc ($((t / 60)) 분, 기록 ${log#"$OUTJ"/})"; brief_fail "$log"; }
  return "$rc"
}

# ───────────────────────── 준비: 환경 · 공식 코드 · 가중치 ─────────────────────────
UV_VER=0.12.13; PY_VER=3.12.14; PIP_VER=26.2.1   # trimodal-uav-h200 과 같은 고정 판 (2026-11 H200 에서 확인)
uv_bin() {  # 고정 판 uv — PATH 의 uv 는 판이 같을 때만, 아니면 작업 폴더에 pip --target 으로 깐다 (이미지 파이썬에는 손대지 않음)
  local u
  for u in "$(command -v uv 2>/dev/null)" "$WORK/uv/pkg/bin/uv"; do
    [ -n "$u" ] && [ -x "$u" ] && [ "$("$u" --version 2>/dev/null | awk '{print $2}')" = "$UV_VER" ] && { echo "$u"; return 0; }
  done
  "$SYSPY" -m pip install -q --upgrade --target "$WORK/uv/pkg" "uv==$UV_VER" >&2 || { echo "!!! [env] uv==$UV_VER 설치 실패" >&2; return 1; }
  [ -x "$WORK/uv/pkg/bin/uv" ] && echo "$WORK/uv/pkg/bin/uv"
}
mkenv_body() {
  local req=$1 E=$2 h=$3 u
  echo "[env] 학습 환경 만들기 ($req)"
  u=$(uv_bin) || return 1
  rm -rf "$E"
  "$u" venv -q -p "$PY_VER" "$E" || { echo "!!! [env] uv venv (Python $PY_VER) 실패"; return 1; }
  if [ "$DRY_RUN" = 1 ]; then
    env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$u" pip install --dry-run -p "$E/bin/python" -r "$req" --index-strategy unsafe-best-match 2>&1 | tail -n 3
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "!!! [env] 판 해석 실패"; return 1; }
    echo "[env] DRY_RUN: 판 해석 통과 — 설치는 건너뜀"; return 0
  fi
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$u" pip install -q -p "$E/bin/python" "pip==$PIP_VER" || { echo "!!! [env] pip==$PIP_VER 실패"; return 1; }
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$E/bin/python" -m pip install -q --no-input -r "$req" || { echo "!!! [env] 패키지 설치 실패"; return 1; }
  "$E/bin/python" -m pip freeze > "$E/freeze.txt" 2>/dev/null
  grep -E '^(torch|torchvision)==' "$E/freeze.txt" > "$E/constraints.txt"
  echo "[env] $("$E/bin/python" -V 2>&1) · $(tr '\n' ' ' < "$E/constraints.txt")"
  echo "$h" > "$E/.ok"
}
TPY=""
# UniDepthV2 가 필요한가: 선생 점검, 그리고 팔 ours · rtp · mtd 를 학습하는 작업 (선생 라벨을 뽑는다). 그때는 학습 환경 + UniDepthV2 가 쓰는 꾸러미 (envs/teacher.txt, torch 판은 같다)
NEED_UD=0; [ "$MODE" = teacher ] && NEED_UD=1
case "$MODE:,$ARMS," in smoke:*,ours,*|all:*,ours,*|smoke:*,rtp,*|all:*,rtp,*|smoke:*,mtd,*|all:*,mtd,*) NEED_UD=1;; esac   # 선생 라벨을 쓰는 팔
ENVNAME=train; [ $NEED_UD = 1 ] && ENVNAME=teacher
setup_env() {
  local E=$WORK/envs/$ENVNAME h rc a n0 L=$LOGS/setup_env.log
  if [ -n "$PY" ]; then TPY=$PY; echo "[setup] 학습 파이썬 (주어진 것)"; return 0; fi
  h=$({ cat "envs/$ENVNAME.txt"; echo "uv $UV_VER python $PY_VER pip $PIP_VER"; } | sha256sum | cut -c1-16)
  if [ ! -f "$E/.ok" ] || [ "$(cat "$E/.ok")" != "$h" ]; then
    for a in 1 2 3; do                                             # uv venv 가 원인 모르게 한 번 실패한 적이 있음 (trimodal 작업 B) → 다시 하기
      touch "$L"; n0=$(wc -l < "$L")
      mkenv_body "envs/$ENVNAME.txt" "$E" "$h" >> "$L" 2>&1; rc=$?
      tail -n "+$((n0 + 1))" "$L" | grep -aE '^\[env\]|^!!!' | tail -n 2
      [ $rc = 0 ] && break
      tail -n "+$((n0 + 1))" "$L" | grep -avE '^\[env\]|^!!!|^[[:space:]]*$' | tail -n 4 | cut -c1-300 | sed 's/^/  로그: /'   # 이번 시도의 uv · pip 오류 줄
      [ $a -lt 3 ] && { echo "[env] 60 초 뒤 다시 ($a/3)"; sleep 60; }
    done
    [ $rc = 0 ] || return 1
    [ "$DRY_RUN" = 1 ] && return 0
  fi
  TPY=$E/bin/python
  [ -s "$E/constraints.txt" ] && export PIP_CONSTRAINT=$E/constraints.txt UV_CONSTRAINT=$E/constraints.txt
  mkdir -p "$OUTJ/meta"; cp "$E/freeze.txt" "$OUTJ/meta/pip_freeze.txt" 2>/dev/null
}
export_weights() {
  local m
  m=$("$SYSPY" -I -B setup/get_weights.py --dir dav2_vitl) || { echo "!!! [setup] 가중치가 캐시에 없음"; return 1; }
  export TD_DAV2=$m/depth_anything_v2_vitl.pth HF_HUB_OFFLINE=1
  if [ "$MODE" = relcheck ] && [ "$ENC" = vitl ]; then
    m=$("$SYSPY" -I -B setup/get_weights.py --dir dav2_vits) || { echo "!!! [setup] ViT-S 가중치가 캐시에 없음"; return 1; }
    export TD_DAV2S=$m/depth_anything_v2_vits.pth
  fi
}
setup_all() {
  [ -f "$STATE/.ok_setup_${ENVNAME}_$DRY_RUN" ] && { setup_env >/dev/null; export_weights || return 1; return 0; }
  setup_env || { echo "!!! [setup] 환경 실패 — logs/setup_env.log"; return 1; }
  run_step code "$LOGS/setup_code.log" show bash setup/fetch_code.sh "$TD_CODE" || return 1
  if [ $NEED_UD = 1 ]; then run_step code "$LOGS/setup_code.log" show bash setup/fetch_code.sh "$TD_UNIDEPTH" unidepth || return 1; fi
  if [ "$DRY_RUN" = 1 ]; then
    run_step weights "$LOGS/setup_weights.log" show "$SYSPY" -I -B setup/get_weights.py --dry_run || return 1
    echo "[setup] DRY_RUN: 패키지 설치 · 가중치 받기는 건너뜀"; touch "$STATE/.ok_setup_${ENVNAME}_1"; return 0
  fi
  local only=dav2_vitl; [ $NEED_UD = 1 ] && [ "$ENC" = vitl ] && only=dav2_vitl,unidepth_v2_vitl
  [ "$MODE" = relcheck ] && [ "$ENC" = vitl ] && only=dav2_vitl,dav2_vits   # 로컬 흐름 시험 (ENC=vits) 은 무작위 초기화라 받지 않는다
  run_step weights "$LOGS/setup_weights.log" show "$TPY" -I -B setup/get_weights.py --only "$only" || \
    run_step weights "$LOGS/setup_weights.log" show env HF_HUB_DISABLE_XET=1 "$TPY" -I -B setup/get_weights.py --only "$only" || return 1
  export_weights || return 1
  local g=""; [ $H200 = 1 ] && g=--gpu
  run_step env_check "$LOGS/setup_check.log" show env HF_HUB_OFFLINE=1 "$TPY" -I -B setup/check.py env $g || return 1
  touch "$STATE/.ok_setup_${ENVNAME}_0"
}
data_step() {
  local s=(--search)                                              # 올려 둔 묶음을 찾을 곳 (없는 것은 LINKS 의 목록에서 받는다)
  [ -d "$APP/data" ] && s+=("$APP/data")
  [ -n "$DATA_DIR" ] && s+=("$DATA_DIR")
  [ -f "$STATE/.ok_data_$KIND" ] && return 0
  WATCH="$WORK/dl/*.part" run_step data "$LOGS/data.log" show "$SYSPY" -I -B -u setup/get_data.py --kind "$KIND" --dest "$TD_DATA" --dl "$WORK/dl" \
    ${LINKS:+--links "$LINKS"} "${s[@]}" || return 1
  run_step data_check "$LOGS/data_check.log" show "$TPY" -I -B -u src/data.py check || return 1
  touch "$STATE/.ok_data_$KIND"
}

# ───────────────────────── 학습 · 평가 ─────────────────────────
train_eval() {  # 팔 시드 — 끝난 것은 건너뜀 (같은 작업 안)
  local arm=$1 seed=$2 name=$1_s$2 a
  local rd=$RUNS/$name ck=$CK/$name
  mkdir -p "$rd" "$ck"
  local w=""; [ "$ENC" = vitl ] && w=$TD_DAV2
  if [ ! -f "$rd/done.json" ]; then
    if [ "$arm" = rt ] || [ "$arm" = rtp ]; then                  # 1 단계: 열화상용 상대 깊이 모델 → $ck/rel/rel.pt (2 단계가 끝날 때까지 둔다)
      local rp=(); [ "$arm" = rtp ] && rp=(--pairs --pair_bs "$PAIR_BS")   # rtp 는 1 단계에도 쌍 (선생의 모양)
      for ((a = 0; a <= RETRY; a++)); do
        [ -f "$ck/rel/rel.pt" ] && [ -f "$rd/rel/done.json" ] && break   # 이번 작업의 1 단계가 끝낸 것만 (작업 폴더에 남은 옛 rel.pt 는 쓰지 않는다)
        WATCH="$rd/rel/train_log.csv $rd/rel/val_log.csv" run_step "상대 깊이 학습 $name" "$LOGS/train_rel_$name.log" show "$TPY" -I -B -u src/train_rel.py --seed "$seed" \
          --epochs "$EPOCHS" --bs "$BS" --amp "$AMP" --limit "$LIMIT" --workers "$WORKERS" --encoder "$ENC" --weights "$w" --out "$rd/rel" --ckpt "$ck/rel" "${rp[@]}" --resume
        [ "$STOP" = 1 ] && return 130
        [ -f "$rd/rel/fail.json" ] && return 1
      done
      [ -f "$ck/rel/rel.pt" ] && [ -f "$rd/rel/done.json" ] || { echo "!!! [상대 깊이 학습 $name] 끝내지 못함"; return 1; }
      w=$ck/rel/rel.pt
    fi
    for ((a = 0; a <= RETRY; a++)); do
      WATCH="$rd/train_log.csv $rd/val_log.csv" run_step "학습 $name" "$LOGS/train_$name.log" show "$TPY" -I -B -u src/train.py --arm "$arm" --seed "$seed" \
        --epochs "$EPOCHS" --bs "$BS" --pair_bs "$PAIR_BS" --amp "$AMP" --limit "$LIMIT" --workers "$WORKERS" --encoder "$ENC" --weights "$w" --out "$rd" --ckpt "$ck" --resume
      [ -f "$rd/done.json" ] && break
      [ "$STOP" = 1 ] && return 130
      [ -f "$rd/fail.json" ] && return 1                       # loss 발산은 다시 해도 같음
      [ $a -lt "$RETRY" ] && echo "[학습 $name] 다시 시작 ($((a + 1))/$RETRY) — last.pt 에서"
    done
    [ -f "$rd/done.json" ] || { echo "!!! [학습 $name] 끝내지 못함"; return 1; }
    rm -f "$ck/rel/rel.pt"                                        # rt: 고정한 상대 깊이 모델은 final.pt 안에도 들어 있다
  fi
  if [ ! -f "$rd/eval/metrics.json" ]; then
    run_step "평가 $name" "$LOGS/eval_$name.log" show "$TPY" -I -B -u src/evaluate.py --ckpt "$ck/final.pt" --out "$rd/eval" --amp "$AMP" --bs "$BS" --workers "$WORKERS" --encoder "$ENC" \
      ${SETS:+--sets "$SETS"} || return 1
  fi
  if [ "$CKPT" = keep ]; then                                     # 결과 폴더에 남긴다 (results.zip 에는 안 넣음) — 나중에 eval 모드가 읽는다
    mkdir -p "$OUTJ/ckpt" && mv "$ck/final.pt" "$OUTJ/ckpt/$name.pt" && echo "[ckpt] $name.pt 남김 ($(du -h "$OUTJ/ckpt/$name.pt" | cut -f1))"
    return 0
  fi
  [ "$CKPT" = all ] || { [ "$CKPT" = seed0 ] && [ "$seed" = "$SEED0" ]; } || rm -f "$ck/final.pt"   # 안 묶을 체크포인트는 평가 뒤에 지운다 (작업 폴더 공간)
}
zip_results() {  # /app/output/<작업>/results.zip — logs · runs (기록 · 지표 · 사진별 표 · 예측 표본) · summary · meta + CKPT 에 따른 final.pt
  local cks=() d
  if [ "$CKPT" != none ] && [ "$CKPT" != keep ] && [ -d "$CK" ]; then
    for d in "$CK"/*/; do [ -f "$d/final.pt" ] && cks+=("$d/final.pt"); done
  fi
  "$SYSPY" -I -B - "$OUTJ/results.zip" "$OUTJ" "${cks[@]}" <<'PY'
import os, sys, zipfile
dst, root, cks = sys.argv[1], sys.argv[2], sys.argv[3:]
items = []
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        if not p.startswith(dst) and not os.path.relpath(p, root).startswith("ckpt" + os.sep):   # 남겨 둔 모델 (CKPT=keep) 은 묶지 않는다
            items.append((p, os.path.relpath(p, root)))
items += [(c, os.path.join("ckpt", os.path.basename(os.path.dirname(c)) + ".pt")) for c in cks]
with zipfile.ZipFile(dst + ".tmp", "w", allowZip64=True) as z:
    for src, arc in items:
        z.write(src, arc, compress_type=zipfile.ZIP_STORED if arc.endswith((".pt", ".png")) else zipfile.ZIP_DEFLATED)
os.replace(dst + ".tmp", dst)
print(f"[zip] results.zip — 파일 {len(items)} 개 (체크포인트 {len(cks)}), {os.path.getsize(dst) / 2**20:.1f} MiB")
PY
}
report() { [ -n "$TPY" ] && [ "$MODE" != teacher ] && [ "$MODE" != relcheck ] || return 0; run_step report "$LOGS/report.log" show "$TPY" -I -B -u src/report.py summary --runs "$RUNS" --out "$SUMM"; }
fail() { echo "!!! [$MODE] $1"; report; zip_results; exit 1; }

case "$MODE" in
  check) "$SYSPY" -I -B setup/check.py box --app "$APP" ${LINKS:+--links "$LINKS"};;
  setup) setup_all || fail "준비 실패";;
  data) setup_all || fail "준비 실패"; data_step || fail "데이터 실패";;
  smoke|all)
    setup_all || fail "준비 실패"
    data_step || fail "데이터 실패"
    if [ $NEED_UD = 1 ]; then                                     # 팔 ours · rtp · mtd 의 쌍 라벨: 선생을 쌍의 RGB 에 한 번 돌려 둔다 (학습 중에는 안 돌린다)
      off=(env); [ "$ENC" = vitl ] || off=(env -u HF_HUB_OFFLINE)   # 작은 인코더 (로컬 흐름 시험) 의 UniDepthV2 는 목록에 없어 그때 받는다
      run_step "선생 라벨" "$LOGS/labels.log" show "${off[@]}" "$TPY" -I -B -u src/labels.py --encoder "$ENC" --bs "$PAIR_BS" --workers "$WORKERS" || fail "선생 라벨 실패"   # 한 번에 PAIR_BS 장 (로컬 시험은 1 — 메모리)
    fi
    for s in ${SEEDS//,/ }; do                                    # 시드 먼저 — 중간에 끊겨도 팔들을 같은 시드로 견줄 수 있게
      for arm in ${ARMS//,/ }; do
        train_eval "$arm" "$s" || fail "$arm 시드 $s 실패"
        report; zip_results
      done
    done;;
  teacher)
    setup_all || fail "준비 실패"
    tk=teacher; [ "$KIND" = smoke ] && tk=teacher_smoke
    s=(--search); [ -d "$APP/data" ] && s+=("$APP/data"); [ -n "$DATA_DIR" ] && s+=("$DATA_DIR")
    WATCH="$WORK/dl/*.part" run_step data "$LOGS/data.log" show "$SYSPY" -I -B -u setup/get_data.py --kind "$tk" --dest "$WORK/data_teacher" --dl "$WORK/dl" \
      ${LINKS:+--links "$LINKS"} "${s[@]}" || fail "데이터 실패"
    off=(env); [ "$ENC" = vitl ] || off=(env -u HF_HUB_OFFLINE)   # 작은 인코더 (로컬 흐름 시험) 의 UniDepthV2 는 목록에 없어 그때 받는다
    run_step "선생 점검" "$LOGS/teacher.log" show "${off[@]}" "$TPY" -I -B -u src/teacher_check.py --data "$WORK/data_teacher" --out "$OUTJ/teacher" \
      --encoder "$ENC" --weights "$([ "$ENC" = vitl ] && echo "$TD_DAV2")" --limit "$LIMIT" --amp 0 || fail "선생 점검 실패"   # Depth Anything V2 는 fp32 로 (UniDepthV2 는 공식 infer 가 GPU 에서 fp16 혼합 정밀도로 돈다)
    zip_results;;
  relcheck)
    setup_all || fail "준비 실패"
    rk=val.tar,ms2_rgb_val.tar; [ "$KIND" = smoke ] && rk=smoke.tar,smoke_teacher.tar   # MS2 val 의 열화상 · RGB 묶음 (같은 폴더 구조라 한곳에 푼다)
    s=(--search); [ -d "$APP/data" ] && s+=("$APP/data"); [ -n "$DATA_DIR" ] && s+=("$DATA_DIR")
    WATCH="$WORK/dl/*.part" run_step data "$LOGS/data.log" show "$SYSPY" -I -B -u setup/get_data.py --kind all --only "$rk" --dest "$WORK/data_rel" --dl "$WORK/dl" \
      ${LINKS:+--links "$LINKS"} "${s[@]}" || fail "데이터 실패"
    if [ "$ENC" = vitl ]; then rm=(--models vitl,vits --weights_l "$TD_DAV2" --weights_s "$TD_DAV2S"); else rm=(--models vits); fi   # 로컬 흐름 시험은 작은 모델을 무작위 초기화로
    run_step "상대 깊이 점검" "$LOGS/relcheck.log" show "$TPY" -I -B -u src/rel_check.py --data "$WORK/data_rel" --out "$OUTJ/relcheck" "${rm[@]}" \
      --limit "$LIMIT" --bs "$BS" --workers "$WORKERS" || fail "상대 깊이 점검 실패"
    zip_results;;
  eval)
    src=$OUTROOT/$FROM/ckpt
    ls "$src"/*.pt >/dev/null 2>&1 || fail "앞선 작업의 모델이 안 보임: 결과 경로의 $FROM/ckpt (그 작업을 CKPT=keep 으로 돌렸는지, 결과 경로가 작업 사이에 남는지 확인)"
    setup_all || fail "준비 실패"
    data_step || fail "데이터 실패"
    for f in "$src"/*.pt; do
      name=$(basename "$f" .pt)
      [[ $name =~ ^[a-z]+_s[0-9]+$ ]] || continue
      mkdir -p "$RUNS/$name"
      run_step "평가 $name" "$LOGS/eval_$name.log" show "$TPY" -I -B -u src/evaluate.py --ckpt "$f" --out "$RUNS/$name/eval" --amp "$AMP" --bs "$BS" --workers "$WORKERS" \
        --encoder "$ENC" ${SETS:+--sets "$SETS"} || fail "$name 평가 실패"
    done
    report; zip_results;;
esac
