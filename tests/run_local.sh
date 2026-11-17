#!/usr/bin/env bash
# 로컬 시험 (CPU 만, 메모리 한도 3 GB): ① 부품 시험 ② 작은 묶음 (smoke.tar) 으로 run.sh smoke 를 끝까지 — 작은 인코더 (ENC=vits, 무작위 초기화) 로 흐름만.
# 사진 읽기 작업자를 따로 띄우지 않는다 (WORKERS=0) — 작업자마다 torch 를 올려 수백 MB 씩 먹는다 (2026-11-16 작업자 1 로 3 GB 한도에 걸림).
# 실제 모델 (ViT-L) 학습은 로컬에서 돌리지 않는다: CPU 로도 12 GB 를 넘겨 이 PC 의 다른 실험을 밀어낸다 (2026-11-16 두 번 메모리 한도로 죽음). ViT-L 은 H200 의 smoke 에서 본다.
# 로컬 GPU 는 다른 실험이 쓰므로 CUDA_VISIBLE_DEVICES='' 로 가린다. 작업 폴더는 디스크에 둔다 (/tmp 는 메모리).
# 사용: bash tests/run_local.sh [작업 폴더=~/data/tdz_work] — 끝나면 작업 폴더를 지워도 된다.
#   SMOKE_DIR=<smoke.tar 가 있는 폴더> 는 꼭 준다. PY (기본 ~/venv/depthvlm/bin/python) · HF_DIR (가중치가 있는 HF 캐시) 는 환경 변수로 바꾼다.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-$HOME/data/tdz_work}
PY=${PY:-$HOME/venv/depthvlm/bin/python}
SMOKE_DIR=${SMOKE_DIR:?"smoke.tar 가 있는 폴더를 SMOKE_DIR 로 준다"}
HF_DIR=${HF_DIR:-$HOME/.cache/huggingface}
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
CAP=(systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 nice -n 19)
mkdir -p "$W"
bash setup/fetch_code.sh "$W/code/Depth-Anything-V2"
TD_CODE=$W/code/Depth-Anything-V2 "${CAP[@]}" "$PY" -I -B -W ignore::ResourceWarning tests/test_parts.py
rm -rf "$W/output/smoke_local"
"${CAP[@]}" bash run.sh smoke WORK="$W" PY="$PY" DATA_DIR="$SMOKE_DIR" HF_DIR="$HF_DIR" BS=2 PAIR_BS=1 WORKERS=0 ENC=vits CKPT=keep JOB=smoke_local
# 남긴 모델로 평가만 다시 (eval 모드) — 일부 세트만
rm -rf "$W/output/eval_local"
"${CAP[@]}" bash run.sh eval FROM=smoke_local SETS=vivid,fire KIND=smoke WORK="$W" PY="$PY" DATA_DIR="$SMOKE_DIR" HF_DIR="$HF_DIR" BS=2 WORKERS=0 ENC=vits JOB=eval_local
