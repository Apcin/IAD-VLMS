#!/bin/bash
# Run eval for ONE checkpoint × ONE dataset on a given GPU.
# Usage:  run_abc_eval_one.sh <ckpt_dir> <dsmvtec|visa> <gpu_id>

set -eo pipefail
CKPT=$1
DATASET=$2
GPU=$3

if [ -z "$CKPT" ] || [ -z "$DATASET" ] || [ -z "$GPU" ]; then
  echo "usage: $0 <ckpt_dir> <dsmvtec|visa> <gpu>"; exit 1
fi
[ -d "$CKPT" ] || { echo "ckpt not found: $CKPT"; exit 1; }

# Paths. WORK_DIR is the workspace holding LlamaFactory/, outputs/ and hf_cache/; it
# defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
EVAL_SCRIPT=$HERE/evaluate_qwen25vl_7b_trainprompt.py
LOG=$CKPT/eval_${DATASET}_full_trainprompt.log

case "$DATASET" in
  dsmvtec) FLAG=--ds-mvtec-only ;;
  visa)    FLAG=--visa-only ;;
  *) echo "unknown dataset: $DATASET"; exit 1 ;;
esac

OUT=$CKPT/eval_${DATASET}_full_trainprompt.json
if [ -f "$OUT" ]; then
  echo "[$(date)] SKIP $CKPT $DATASET (eval JSON already exists)"
  exit 0
fi

echo "[$(date)] eval $(basename $(dirname $CKPT))/$(basename $CKPT) -> $DATASET on CUDA $GPU"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
export HF_HOME=$WORK_DIR/hf_cache
export CUDA_VISIBLE_DEVICES=$GPU
python "$EVAL_SCRIPT" \
  --checkpoint "$CKPT" \
  $FLAG --batch-size 16 \
  --output "$OUT" \
  > "$LOG" 2>&1

echo "[$(date)] eval done -> $OUT"
