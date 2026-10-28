#!/bin/bash
# Full Qwen3-VL-8B-Instruct Arm-C 6K SFT (4 epochs, frozen ViT, ZeRO-3 CPU offload).
# Usage: run_qwen3vl_8b_armC.sh "1,2"   (default 1,2 = the NV4 NVLink pair freed by GRPO; 0,3 also works)
# All HF caches/downloads point under $WORK_DIR.
set -eo pipefail
# Paths. WORK_DIR is the workspace holding LlamaFactory/, outputs/ and hf_cache/; it
# defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"

# The YAML carries ${WORK_DIR} / ${REPO_ROOT} placeholders; expand them into a temp copy.
YAML=$(mktemp -t sft_qwen3vl_8b_armC.yaml.XXXX)
sed -e "s|\${WORK_DIR}|$WORK_DIR|g" -e "s|\${REPO_ROOT}|$REPO_ROOT|g" \
    "$REPO_ROOT/configs/sft/sft_qwen3vl_8b_armC.yaml" > "$YAML"
GPUS="${1:-1,2}"
LOG_DIR="$WORK_DIR/logs/qwen3vl_8b_armC_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=$GPUS
export FORCE_TORCHRUN=1
export HF_HOME=$WORK_DIR/hf_cache
export HF_HUB_CACHE=$WORK_DIR/hf_cache/hub
export TRANSFORMERS_CACHE=$WORK_DIR/hf_cache
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_$USER}"
mkdir -p "$TRITON_CACHE_DIR" "$WORK_DIR/hf_cache"

echo "════════════════════════════════════════════════════════════════════════"
echo " Qwen3-VL-8B-Instruct — Arm-C 6K SFT (frozen ViT, 4 epochs, from base)"
echo " Start:   $(date)"
echo " GPUs:    CUDA $GPUS   |  HF_HOME: $HF_HOME"
echo " YAML:    $YAML"
echo " Log dir: $LOG_DIR"
echo "════════════════════════════════════════════════════════════════════════"

cd "$WORK_DIR/LlamaFactory"
conda run -n llama_sft --no-capture-output \
    llamafactory-cli train "$YAML" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "[$(date)] Qwen3-VL-8B Arm-C SFT DONE."
touch "$WORK_DIR/outputs/sft_qwen3vl_8b_armC/train.done"
