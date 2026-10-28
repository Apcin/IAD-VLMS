#!/bin/bash
# SFT for Qwen2.5-VL-7B — 15k regenerated dataset, full finetune (unfrozen), LR=1e-6, 4 epochs
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
    llamafactory-cli train /bulk/aacudad/reasoning_traces/Training/sft_qwen25vl_7b_15k_unfrozen.yaml \
    2>&1 | tee /bulk/aacudad/reasoning_traces/Training/sft_qwen25vl_7b_15k_unfrozen.log

echo "Done."
