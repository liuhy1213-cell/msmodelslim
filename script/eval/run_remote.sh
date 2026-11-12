#!/bin/bash
# 从跳板机（iquest）把评测投到目标节点上执行。
#
# 用法:
#   run_remote.sh <节点IP> <容器名> <model_id> <端口> <输出目录> [limit] [-l 标签] [-b 并发] ...
#
# 例:
#   run_remote.sh 10.222.44.20 iq1120p IQuest-Q1 8007 \
#       /mnt/sfs_turbo/work/liuhaiyang/c8own/eval_v1 300 -l v1 -b 64
set -uo pipefail

IP=${1:?用法: run_remote.sh <ip> <container> <model_id> <port> <out_dir> [limit] [args...]}
CT=${2:?}
MODEL=${3:?}
PORT=${4:?}
OUT=${5:?}
shift 5

# 第 6 个参数如果是纯数字，当作 limit，转成 -n
ARGS=""
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  ARGS="-n $1"
  shift
fi
# 其余参数原样透传（-n/-b/-t/-l 等）
for a in "$@"; do ARGS="$ARGS $(printf '%q' "$a")"; done

ssh -o BatchMode=yes -o ConnectTimeout=15 "root@$IP" bash -s -- "$CT" "$MODEL" "$PORT" "$OUT" "$ARGS" <<'REMOTE'
set -uo pipefail
CT=$1; MODEL=$2; PORT=$3; OUT=$4; shift 4
eval "set -- $*"
exec bash /mnt/sfs_turbo/work/liuhaiyang/script/eval/ifbench_eval.sh \
  -c "$CT" -m "$MODEL" -u "http://127.0.0.1:$PORT/v1/chat/completions" -o "$OUT" "$@"
REMOTE
