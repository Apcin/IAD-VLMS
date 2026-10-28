#!/bin/bash
# SFT for Qwen2.5-VL-3B — 15k regenerated dataset, vision tower FROZEN, LR=1e-5, 4 epochs
set -e

export CUDA_VISIBLE_DEVICES=0,3
export FORCE_TORCHRUN=1

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export TRITON_CACHE_DIR=/tmp/triton_cache_aacudad

mkdir -p /tmp/triton_cache_aacudad

WORK_DIR="/bulk/aacudad/reasoning_traces"

cd $WORK_DIR/LlamaFactory
conda run -n llama_sft --no-capture-output \
    llamafactory-cli train /bulk/aacudad/reasoning_traces/Training/sft_qwen25vl_3b_15k_frozen.yaml \
    2>&1 | tee /bulk/aacudad/reasoning_traces/Training/sft_qwen25vl_3b_15k_frozen.log

echo "Done."
