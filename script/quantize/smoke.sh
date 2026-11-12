#!/bin/bash
# 2 层冒烟：只跑前 2 个 decoder layer，约 1 分钟，用来验证适配器/配方没被改坏。
# 用法: ./smoke.sh [c8|v1]
set -euo pipefail

MODE=${1:-c8}
REPO=${MSMODELSLIM_REPO:-/mnt/sfs_turbo/work/liuhaiyang/modelslim_full}

export IQUEST_MAX_LAYERS=2
export KEEP_LAYER_LIMIT=1
OMP_NUM_THREADS=${OMP_NUM_THREADS:-64} MKL_NUM_THREADS=${MKL_NUM_THREADS:-64} \
  exec bash "$(dirname "$0")/quantize.sh" "$MODE" "$REPO/out_smoke_$MODE"
