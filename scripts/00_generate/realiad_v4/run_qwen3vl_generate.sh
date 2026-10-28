#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LIST="${GPU_LIST:-0}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
TOTAL_SHARDS="${#GPUS[@]}"

if [[ "$TOTAL_SHARDS" -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU id" >&2
  exit 2
fi

PIDS=()
for SHARD_ID in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[$SHARD_ID]}"
  echo "Starting shard ${SHARD_ID}/${TOTAL_SHARDS} on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 \
    python "$SCRIPT_DIR/generate_realiad_traces_qwen3vl.py" \
      --shard_id "$SHARD_ID" \
      --total_shards "$TOTAL_SHARDS" \
      "$@" &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    STATUS=1
  fi
done
exit "$STATUS"
