#!/bin/bash
# IQuest-Q1 全 C8 HiFloat8（Q/K/V per-tensor）量化。
# 标定集 qwen3_cot_w4a4.json；CPU 量化，容器内跑（venv 的 python 只在容器里存在）。
set -uo pipefail
REPO=/mnt/sfs_turbo/work/liuhaiyang/modelslim_full
cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

OUT=${OUT:-$REPO/out_full_c8_hif8}
CONFIG=${CONFIG:-$REPO/lab_practice/iquest_q1/iquest_q1_w8a8c8_hif8.yaml}
MODEL=${MODEL:-/mnt/sfs_turbo/models/IQuest-Q1}

rm -rf "$OUT"
echo "START $(date -Is)  CONFIG=$CONFIG OUT=$OUT MAX_LAYERS=${IQUEST_MAX_LAYERS:-all}"
/mnt/sfs_turbo/work/liuhaiyang/venv/bin/msmodelslim quant \
  --model_path "$MODEL" \
  --save_path "$OUT" \
  --device cpu \
  --config_path "$CONFIG" \
  --model_type IQuest-Q1 \
  --trust_remote_code true
RC=$?
echo "EXIT=$RC $(date -Is) OUT=$OUT"
exit $RC
