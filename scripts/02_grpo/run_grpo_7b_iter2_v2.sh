#!/bin/bash
# Plan A v2 — GRPO-Iter2 (CORRECTED)
# Init: SFT-Iter2 v2 final checkpoint (3 epochs trained).
# 2 epochs of GRPO (matches baseline's run1+run2 = 2 epochs total).
# --reward_funcs accuracy format (CRITICAL — avoids default which adds reasoning judge).
# CUDA 0,3 (2 GPUs; GPU 1 has poor NCCL connectivity with 0+3 → hangs).
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export PYTHONPATH=/bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl

# SFT-Iter2 v2 — set externally (master launcher passes the 3-epoch checkpoint)
: "${SFT_ITER2_CKPT:=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_iter2_v2_frozen}"
export MODEL_NAME_OR_PATH=${SFT_ITER2_CKPT}

export DATASET_NAME=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json
export OUTPUT_DIR=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_iter2_v2_full

export WANDB_MODE=disabled
export DEBUG_MODE=True
export LOG_PATH=${OUTPUT_DIR}/reward.log

export CUDA_VISIBLE_DEVICES=0,3
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=2 --nnodes=1 --master_port=29530 \
  /bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl/grpo_ad.py \
  --deepspeed /bulk/aacudad/reasoning_traces/Training/zero3_offload.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --dataset_name ${DATASET_NAME} \
  --reward_funcs accuracy format \
  --image_path / \
  --use_vllm_for_gen false \
  --use_system_prompt false \
  --max_prompt_length 4096 \
  --max_completion_length 512 \
  --num_generations 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --bf16 true \
  --report_to none \
  --gradient_checkpointing true \
  --attn_implementation sdpa \
  --max_pixels 480000 \
  --save_steps 265 \
  --save_only_model true \
  --num_train_epochs 2 \
  --single_img 1 \
  2>&1 | tee ${OUTPUT_DIR}/train.log
