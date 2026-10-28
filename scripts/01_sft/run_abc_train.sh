#!/bin/bash
# Generic train-only script for arms A / B / C.
# Usage:  run_abc_train.sh A "0,3"   →   train arm A on CUDA 0,3
#         run_abc_train.sh B "1,2"   →   train arm B on CUDA 1,2

set -eo pipefail

ARM=$1
GPUS=$2

if [ -z "$ARM" ] || [ -z "$GPUS" ]; then
  echo "usage: $0 <A|B|C> <CUDA_VISIBLE_DEVICES, e.g. 0,3>"
  exit 1
fi

WORK_DIR="/bulk/aacudad/reasoning_traces"
YAML="$WORK_DIR/Training/sft_abc_${ARM}.yaml"
if [ ! -f "$YAML" ]; then
  echo "missing YAML: $YAML"
  exit 1
fi

# Output dir is decided inside the YAML.  We just need a log path.
LOG_DIR="$WORK_DIR/logs/abc_${ARM}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=$GPUS
export FORCE_TORCHRUN=1
export HF_HOME=$WORK_DIR/hf_cache
export TRANSFORMERS_CACHE=$WORK_DIR/hf_cache
export TRITON_CACHE_DIR=/tmp/triton_cache_aacudad
mkdir -p /tmp/triton_cache_aacudad

echo "════════════════════════════════════════════════════════════════════════"
echo " ARM $ARM  —  SFT 7B (frozen ViT, 4 epochs, from base)"
echo " Start:   $(date)"
echo " GPUs:    CUDA $GPUS"
echo " YAML:    $YAML"
echo " Log dir: $LOG_DIR"
echo "════════════════════════════════════════════════════════════════════════"

cd "$WORK_DIR/LlamaFactory"
conda run -n llama_sft --no-capture-output \
    llamafactory-cli train "$YAML" \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""
echo "[$(date)] ARM $ARM SFT DONE."

# Drop a done flag the orchestrator can poll
case "$ARM" in
  A) touch "$WORK_DIR/outputs/sft_qwen25vl_7b_abc_A_kept/train.done" ;;
  B) touch "$WORK_DIR/outputs/sft_qwen25vl_7b_abc_B_kept_corrected/train.done" ;;
  C) touch "$WORK_DIR/outputs/sft_qwen25vl_7b_abc_C_full_patched/train.done" ;;
esac
