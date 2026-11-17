#!/usr/bin/env bash
# 선생 점검 (run.sh teacher) 의 로컬 시험 (CPU 만, 메모리 한도 3 GB): 작은 묶음 (smoke_teacher.tar, 16 장) 으로 끝까지 — 작은 인코더로 흐름만.
#   Depth Anything V2 = 무작위 초기화한 vits (수치는 뜻이 없다) · UniDepthV2 = 공개된 작은 판 (vits14, 실제 가중치 약 0.14 GB 를 HF 캐시에 받는다 —
#   출력이 m 단위로 나오는지, 정답과 단위 · 크기가 맞는지를 여기서 본다). ViT-L 은 로컬에서 돌리지 않는다 (H200 의 KIND=smoke 에서 본다).
# 사용: SMOKE_DIR=<smoke_teacher.tar 가 있는 폴더> bash tests/run_local_teacher.sh [작업 폴더=~/data/tdz_work]
#   PY (기본 ~/venv/unidepth/bin/python — envs/teacher.txt 의 꾸러미가 깔린 파이썬) · HF_DIR 은 환경 변수로 바꾼다.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-$HOME/data/tdz_work}
PY=${PY:-$HOME/venv/unidepth/bin/python}
SMOKE_DIR=${SMOKE_DIR:?"smoke_teacher.tar 가 있는 폴더를 SMOKE_DIR 로 준다"}
HF_DIR=${HF_DIR:-$HOME/.cache/huggingface}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
CAP=(systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 nice -n 19)
mkdir -p "$W"
bash setup/fetch_code.sh "$W/code/Depth-Anything-V2"
TD_CODE=$W/code/Depth-Anything-V2 "${CAP[@]}" "$PY" -I -B -W ignore::ResourceWarning tests/test_parts.py TeacherCheck
rm -rf "$W/output/teacher_local"
"${CAP[@]}" bash run.sh teacher KIND=smoke WORK="$W" PY="$PY" DATA_DIR="$SMOKE_DIR" HF_DIR="$HF_DIR" ENC=vits JOB=teacher_local
