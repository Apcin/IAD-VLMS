#!/bin/bash
# Zero-Shot SFT for Qwen2.5-VL-7B — 6k combined dataset (sft_train + grpo_train)
# Full finetune (vision tower + projector + LLM), 2 epochs
# Output: outputs/sft_qwen25vl_7b_zeroshot_6k  (does NOT touch sft_qwen25vl_7b_zeroshot)
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
    llamafactory-cli train /bulk/aacudad/reasoning_traces/Training/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml \
    2>&1 | tee /bulk/aacudad/reasoning_traces/Training/sft_qwen25vl_7b_zeroshot_6k_frozen.log

echo "Done."
