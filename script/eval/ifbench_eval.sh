#!/bin/bash
# ============================================================================
# IFBench 固定口径评测（与 zuoqian/qping 参考跑同口径）
#
# 在「持有推理容器的节点」上执行；容器里的服务必须已经 ready。
# 从跳板机调 远 端 用 run_remote.sh。
#
# 口径（不要随意改，改了就不可比）：
#   temperature 1.0, top_p 0.95, top_k 20, max_tokens 16384,
#   seed 42, limit 300, batch 64, dataset allenai/IFBench_test (local snapshot)
#
# 用法:
#   ifbench_eval.sh -c <容器名> -m <model_id> -o <输出目录> [选项]
#
# 选项:
#   -c  容器名（必填，evalscope 在容器里跑）
#   -m  服务端 /v1/models 返回的 id（必填，如 IQuest-Q1 / q1-bf16）
#   -o  输出目录（必填）
#   -u  api url（默认 http://127.0.0.1:8000/v1/chat/completions）
#   -n  limit（默认 300；调试用 20）
#   -b  batch/并发（默认 64，不得超过服务的 --max-num-seqs）
#   -t  max_tokens（默认 16384，必须 <= 服务 max_model_len - 输入长度）
#   -l  标签（写进 results.csv，如 v1 / bf16 / c8）
#   -h  帮助
# ============================================================================
set -uo pipefail

W=/mnt/sfs_turbo/work/liuhaiyang
PACK=$W/script/eval
SNAP=$W/hf_cache/hub/datasets--allenai--IFBench_test/snapshots/ed7c12452594b7028b15807b86d3938150b663ae
PY=$W/evalvenv/bin/evalscope

CT=""; MODEL=""; OUT=""; URL="http://127.0.0.1:8000/v1/chat/completions"
LIMIT=300; BATCH=64; MAX_TOKENS=16384; LABEL=""

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts ":c:m:o:u:n:b:t:l:h" opt; do
  case $opt in
    c) CT=$OPTARG ;;
    m) MODEL=$OPTARG ;;
    o) OUT=$OPTARG ;;
    u) URL=$OPTARG ;;
    n) LIMIT=$OPTARG ;;
    b) BATCH=$OPTARG ;;
    t) MAX_TOKENS=$OPTARG ;;
    l) LABEL=$OPTARG ;;
    h) usage 0 ;;
    *) echo "未知参数 -$OPTARG" >&2; usage 1 ;;
  esac
done
[ -n "$CT" ] && [ -n "$MODEL" ] && [ -n "$OUT" ] || usage 1

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ---- 0. 前置检查 -----------------------------------------------------------
[ -x "$PY" ] || { echo "找不到 evalscope: $PY" >&2; exit 2; }
[ -d "$SNAP" ] || { echo "找不到 IFBench 本地快照: $SNAP" >&2; exit 2; }
[ -d "$W/nltk_data" ] || echo "WARN: 缺 $W/nltk_data，IFBench 的 nltk 依赖会失败" >&2

MODELS_JSON=$(curl -s -m 10 "${URL%/chat/completions}/models" 2>/dev/null)
if ! echo "$MODELS_JSON" | grep -q "\"$MODEL\""; then
  echo "服务未就绪或 model id 不匹配: url=$URL model=$MODEL" >&2
  echo "服务返回: ${MODELS_JSON:0:300}" >&2
  exit 3
fi
MAX_LEN=$(echo "$MODELS_JSON" | python3 -c "import sys,json;d=json.load(sys.stdin)['data'];print(next((m.get('max_model_len') or 0) for m in d if m['id']=='$MODEL'))" 2>/dev/null)
if [ -n "${MAX_LEN:-}" ] && [ "$MAX_LEN" -gt 0 ] 2>/dev/null && [ "$MAX_TOKENS" -gt "$MAX_LEN" ]; then
  echo "max_tokens=$MAX_TOKENS 超过服务 max_model_len=$MAX_LEN，请求会 400" >&2
  exit 4
fi

mkdir -p "$OUT"
RUN_TS=$(date -u +%Y%m%d_%H%M%S)
META=$OUT/meta_$RUN_TS.json
{
  echo "{"
  echo "  \"label\": \"$LABEL\","
  echo "  \"model\": \"$MODEL\","
  echo "  \"url\": \"$URL\","
  echo "  \"container\": \"$CT\","
  echo "  \"limit\": $LIMIT,"
  echo "  \"batch\": $BATCH,"
  echo "  \"max_tokens\": $MAX_TOKENS,"
  echo "  \"temperature\": 1.0,"
  echo "  \"top_p\": 0.95,"
  echo "  \"top_k\": 20,"
  echo "  \"seed\": 42,"
  echo "  \"evalscope\": \"$($PY --version 2>/dev/null | tr -d '\n')\","
  echo "  \"started_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"served_models\": $MODELS_JSON"
  echo "}"
} > "$META"

log "开始评测 label=$LABEL model=$MODEL url=$URL limit=$LIMIT batch=$BATCH max_tokens=$MAX_TOKENS"
log "输出目录: $OUT"

# ---- 1. evalscope ----------------------------------------------------------
docker exec \
  -e NLTK_DATA=$W/nltk_data \
  -e HF_HOME=$W/hf_cache \
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  "$CT" "$PY" eval \
    --model "$MODEL" --api-url "$URL" --datasets ifbench \
    --dataset-hub huggingface \
    --dataset-args "{\"ifbench\":{\"local_path\":\"$SNAP\"}}" \
    --generation-config "{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"max_tokens\":$MAX_TOKENS,\"timeout\":3600}" \
    --limit "$LIMIT" --seed 42 --eval-batch-size "$BATCH" \
    --work-dir "$OUT"
RC=$?
if [ $RC -ne 0 ]; then
  log "evalscope 失败 rc=$RC"
  exit $RC
fi

# ---- 2. 汇总 ---------------------------------------------------------------
log "汇总四项指标"
python3 "$PACK/summarize.py" "$OUT" --max-tokens "$MAX_TOKENS" --label "$LABEL" \
  --csv-out "$PACK/results.csv"
log "完成"
