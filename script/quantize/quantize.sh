#!/bin/bash
# IQuest-Q1 一条命令出成品：量化 → C8 postfix → K 中心化 μ
#
# 用法: bash script/quantize/quantize.sh {v1|c8} [输出目录]
#   DEVICE=npu|cpu        量化设备（默认 cpu；npu 快约 3~5 倍）
#   SKIP_POSTFIX=1        跳过 C8 包整理（调试用）
#   SKIP_KCENTER=1        跳过 K 中心化 μ（调试用）
#   KCENTER_SAMPLES/N     统计 μ 用的标定样本数（默认 16）
#   KCENTER_LAYERS=all    给全部 88 层都算 μ（默认只算 25 个 full 层）
#
# 产物（c8）：
#   <OUT>/quant_model_weights-*.safetensors   主权重
#   <OUT>/c8_kv_v_scales.safetensors          25 层 fa_v.scale（V 的静态尺度）
#   <OUT>/kv_center/{layer}-{rank}.pt         K 中心化的逐通道均值（运行时按需读取）
set -uo pipefail

MODE=${1:?用法: $0 {v1|c8} [输出目录]}
case "$MODE" in
  v1) CFG=iquest_q1_w8a8_linear_mxfp8.yaml ;;
  c8) CFG=iquest_q1_w8a8c8_mxfp8.yaml ;;
  *) echo "用法: $0 {v1|c8} [输出目录]" >&2; exit 1 ;;
esac

REPO=${MSMODELSLIM_REPO:-/mnt/sfs_turbo/work/liuhaiyang/modelslim_full}
BIN=${MSMODELSLIM_BIN:-/mnt/sfs_turbo/work/liuhaiyang/venv/bin/msmodelslim}
SRC=${SRC_MODEL:-/mnt/sfs_turbo/models/IQuest-Q1}
OUT=${2:-$REPO/out_full_$MODE}
CFG_PATH=$REPO/lab_practice/iquest_q1/$CFG
DEVICE=${DEVICE:-cpu}
PY=$REPO/../venv/bin/python

say() { echo "[$(date -Is)] $*"; }

[ -x "$BIN" ] || { echo "找不到 msmodelslim: $BIN" >&2; exit 2; }
PYBIN=$(head -1 "$BIN" | sed 's/^#!//')
[ -x "$PYBIN" ] || { echo "msmodelslim 的 python 不存在: $PYBIN（请在容器内执行）" >&2; exit 2; }
[ -f "$CFG_PATH" ] || { echo "找不到配方: $CFG_PATH" >&2; exit 2; }
[ -d "$SRC" ] || { echo "找不到原始权重: $SRC" >&2; exit 2; }

export PYTHONPATH=$REPO
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-64}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-64}
if [ "${KEEP_LAYER_LIMIT:-0}" != "1" ]; then unset IQUEST_MAX_LAYERS; fi

say "MODE=$MODE DEVICE=$DEVICE"
say "SRC=$SRC"
say "OUT=$OUT"
say "CFG=$CFG_PATH"

rm -rf "$OUT"
say "步骤 1/3 量化"
"$BIN" quant \
  --model_path "$SRC" \
  --save_path "$OUT" \
  --device "$DEVICE" \
  --config_path "$CFG_PATH" \
  --model_type IQuest-Q1 \
  --trust_remote_code true
rc=$?
[ $rc -eq 0 ] || { say "量化失败 rc=$rc"; exit $rc; }

if [ "$MODE" = "c8" ]; then
  if [ "${SKIP_POSTFIX:-0}" != "1" ]; then
    say "步骤 2/3 C8 包整理（fa_v.scale 外置 + 描述补 kv_cache_type + MTP）"
    "$PY" "$REPO/script/quantize/postfix_c8.py" --src-pkg "$OUT" --in-place
    rc=$?
    [ $rc -eq 0 ] || { say "postfix 失败 rc=$rc"; exit $rc; }
  else
    say "步骤 2/3 已跳过（SKIP_POSTFIX=1）"
  fi

  if [ "${SKIP_KCENTER:-0}" != "1" ]; then
    say "步骤 3/3 K 中心化 μ（层=${KCENTER_LAYERS:-full}，样本=${KCENTER_SAMPLES:-16}）"
    "$PY" "$REPO/script/quantize/kv_center_calib.py" \
      --pkg "$OUT" \
      --device "$DEVICE" \
      --samples "${KCENTER_SAMPLES:-16}" \
      --layers "${KCENTER_LAYERS:-full}"
    rc=$?
    [ $rc -eq 0 ] || { say "kv_center 失败 rc=$rc"; exit $rc; }
  else
    say "步骤 3/3 已跳过（SKIP_KCENTER=1）"
  fi
else
  say "v1 不做 C8 postfix / K 中心化，跳过步骤 2-3"
fi

say "完成 $(du -sh "$OUT" 2>/dev/null | cut -f1)  $OUT"
