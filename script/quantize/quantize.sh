#!/bin/bash
# ============================================================================
# IQuest-Q1 一条命令量化
#
#   ./quantize.sh v1            # 线性 W8A8-MXFP8，KV 保持 BF16
#   ./quantize.sh c8            # 追加 25 层全注意力 KV C8（fa_v per-channel）
#
# 环境变量（一般不用改）:
#   MSMODELSLIM_REPO  适配器仓库根目录，默认 /mnt/sfs_turbo/work/liuhaiyang/modelslim_full
#   MSMODELSLIM_BIN   msmodelslim 可执行文件，默认 $MSMODELSLIM_REPO 上级 venv/bin/msmodelslim
#   SRC_MODEL         BF16 原始权重，默认 /mnt/sfs_turbo/models/IQuest-Q1
#   OMP_NUM_THREADS   CPU 线程数，默认 64（MoE 校准吃满核）
# ============================================================================
set -euo pipefail

MODE=${1:-}
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

[ -x "$BIN" ] || { echo "找不到 msmodelslim: $BIN" >&2; exit 2; }
# venv 的 python 只存在于容器镜像里，宿主机上执行会 "required file not found"
PYBIN=$(head -1 "$BIN" | sed 's/^#!//')
[ -x "$PYBIN" ] || {
  echo "msmodelslim 的 python 不存在: $PYBIN" >&2
  echo "这个 venv 依赖容器内的解释器，请在容器里跑：" >&2
  echo "  ssh root@10.222.44.142 'docker exec -it iq-own bash'" >&2
  exit 2
}
[ -f "$CFG_PATH" ] || { echo "找不到配方: $CFG_PATH" >&2; exit 2; }
[ -d "$SRC" ] || { echo "找不到原始权重: $SRC" >&2; exit 2; }

# 防呆：确认跑的是改过的适配器（不是上游原版）
for probe in "Top-k dispatch" "Calibration shell must be bf16"; do
  grep -rq "$probe" "$REPO/msmodelslim/model/iquest_q1/" \
    || echo "WARN: 适配器里找不到「$probe」，可能用的是未修补的上游代码" >&2
done

export PYTHONPATH=$REPO
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-64}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-64}
# 正式量化必须跑满 88 层；只有显式 KEEP_LAYER_LIMIT=1（冒烟脚本）才允许限层
if [ "${KEEP_LAYER_LIMIT:-0}" != "1" ]; then unset IQUEST_MAX_LAYERS; fi

echo "MODE=$MODE"
echo "SRC=$SRC"
echo "OUT=$OUT"
echo "CFG=$CFG_PATH"
echo "START $(date -Is)"

rm -rf "$OUT"
"$BIN" quant \
  --model_path "$SRC" \
  --save_path "$OUT" \
  --device cpu \
  --config_path "$CFG_PATH" \
  --model_type IQuest-Q1 \
  --trust_remote_code true

echo "EXIT=0 $(date -Is)"
