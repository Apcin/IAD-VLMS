#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES=2 bash run_sft_mmad_train1600.sh   (set the GPUs at launch)
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export FORCE_TORCHRUN=1
export HF_HOME=${WORK_DIR}/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_aacudad
export TRITON_CACHE_DIR=/tmp/triton_cache_aacudad
mkdir -p /tmp/triton_cache_aacudad /tmp/hf_datasets_aacudad
OUT=${WORK_DIR}/outputs/sft_qwen25vl_7b_mmad_train1600
mkdir -p $OUT
cd ${WORK_DIR}/LlamaFactory
conda run -n llama_sft --no-capture-output llamafactory-cli train ${WORK_DIR}/mmad_leak_experiment/sft_mmad_train1600.yaml 2>&1 | tee $OUT/train.log
touch $OUT/SFT_DONE.flag
echo "SFT DONE $(date)"
