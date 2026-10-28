#!/bin/bash
# 6K SFT, vision encoder UNFROZEN, at the frozen recipe's learning rate (1e-5). Same yaml as the
# 7B 6K frozen run except freeze_vision_tower: false. Closes the learning-rate confound of Table 4.2.
set -e
export CUDA_VISIBLE_DEVICES=1,2
export FORCE_TORCHRUN=1
export HF_HOME=${WORK_DIR}/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_aacudad
export TRITON_CACHE_DIR=/tmp/triton_cache_aacudad
mkdir -p /tmp/triton_cache_aacudad
OUT=${WORK_DIR}/outputs/sft_qwen25vl_7b_6k_unfrozen_lr1e-5
mkdir -p $OUT
cd ${WORK_DIR}/LlamaFactory
conda run -n llama_sft --no-capture-output llamafactory-cli train ${WORK_DIR}/Training/sft_qwen25vl_7b_6k_unfrozen_lr1e-5.yaml 2>&1 | tee $OUT/train.log
touch $OUT/SFT_DONE.flag
echo "SFT DONE $(date)"
