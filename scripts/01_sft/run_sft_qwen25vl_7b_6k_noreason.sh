#!/bin/bash
# NO-REASONING (labels-only) SFT ablation for Qwen2.5-VL-7B on the 6K dataset, 4 epochs.
#
# Control experiment: same 6000 images and hyperparameters as the reported SFT-6K run
# (80.16 / 64.78), but the model is trained ONLY on the verdict -- the prompt asks
# "Answer with yes or no." and the target is <answer>Yes|No</answer> (no <think>, no
# <type>, no <location>). Comparing the two isolates what the reasoning traces buy us.
#
# GPUs 0+3 (shares GPU 0 with another job by design). Eval is driven separately by
# scripts/04_eval/watch_and_eval_noreason.sh on GPU 2.
#
# Registry: needs the dataset key iad_sft_6k_noreason_train. See
# configs/dataset_info_additions.json and merge it into LlamaFactory/data/dataset_info.json.
#
# Run inside tmux:  tmux new -s sft_noreason 'bash scripts/01_sft/run_sft_qwen25vl_7b_6k_noreason.sh'
set -e

export CUDA_VISIBLE_DEVICES=0,3
export FORCE_TORCHRUN=1

# Paths. WORK_DIR is the workspace holding LlamaFactory/, outputs/ and hf_cache/; it
# defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"

export HF_HOME=$WORK_DIR/hf_cache
export TRANSFORMERS_CACHE=$WORK_DIR/hf_cache
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_$USER}"
mkdir -p "$TRITON_CACHE_DIR"

# The YAML carries ${WORK_DIR} / ${REPO_ROOT} placeholders. Expand them into a temp copy
# so llamafactory-cli sees real paths (see the header of the YAML).
YAML=$(mktemp -t sft_noreason_XXXX.yaml)
sed -e "s|\${WORK_DIR}|$WORK_DIR|g" -e "s|\${REPO_ROOT}|$REPO_ROOT|g" \
    "$REPO_ROOT/configs/sft/sft_qwen25vl_7b_6k_noreason.yaml" > "$YAML"

LOG_DIR="$WORK_DIR/logs/sft_6k_noreason_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

cd $WORK_DIR/LlamaFactory
conda run -n llama_sft --no-capture-output \
    llamafactory-cli train "$YAML" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "Done."
