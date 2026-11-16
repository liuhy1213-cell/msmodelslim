#!/bin/bash
# IQuest-Q1 W8A8C8 (MXFP8) 一条命令：量化 + 成品整理。
#
#   MODEL_PATH=/path/to/IQuest-Q1 OUT=/path/to/out ./iquest_q1_c8_oneshot.sh
#
# 可选环境变量：
#   DEVICE   设备规格，默认 npu；也接受 'npu:0,1,2,3'、'cpu'。
#            npu 约 14 s/层，cpu 约 80 s/层。两者 weight_scale 逐字节相同，
#            fp8 数值差 ~1 LSB（npu 的 cast 向零截断，cpu 是 round-to-nearest）。
#   IQUEST_CALIB_SEQ_LEN  标定序列截断长度，默认 4096
#                         （qwen3_cot_w4a4.json 全长 1243~2661，实际不截断）
#   PKGCLEAN=0            只保留 msmodelslim 原始输出，跳过成品整理
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)

MODEL_PATH=${MODEL_PATH:?用 MODEL_PATH 指定 BF16 源模型目录}
OUT=${OUT:-$PWD/out_iquest_q1_c8}
DEVICE=${DEVICE:-npu}
CFG=${CFG:-$repo/lab_practice/iquest_q1/iquest_q1_w8a8c8_mxfp8.yaml}

export PYTHONPATH=$repo${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-64}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-64}

rm -rf "$OUT"
echo "START $(date -Is) device=$DEVICE out=$OUT"
msmodelslim quant \
  --model_path "$MODEL_PATH" \
  --save_path "$OUT" \
  --device "$DEVICE" \
  --config_path "$CFG" \
  --model_type IQuest-Q1 \
  --trust_remote_code true
rc=$?
echo "EXIT=$rc $(date -Is)"
[ $rc -ne 0 ] && exit $rc

if [ "${PKGCLEAN:-1}" = "1" ]; then
  # fa_v.scale 压平外置、摘 fa_v.offset、补 kv_cache_type / chat_template / mtp
  python "$here/postfix_c8.py" --src-pkg "$OUT" --in-place --mtp-src "$MODEL_PATH"
  echo "PKG=$? $(date -Is)"
fi
