#!/bin/bash
# GRPO training — resume from grpo_7b_6k_frozen_ep3_full_run1/checkpoint-530
# 2 more epochs. CUDA 1,2.

set -e

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export PYTHONPATH=/bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl

export MODEL_NAME_OR_PATH=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run1/checkpoint-530
export DATASET_NAME=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json
export OUTPUT_DIR=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2

export WANDB_MODE=disabled
export DEBUG_MODE=True
export LOG_PATH=${OUTPUT_DIR}/reward.log

export CUDA_VISIBLE_DEVICES=1,2
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=2 --nnodes=1 --master_port=29508 \
  /bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl/grpo_ad.py \
  --deepspeed /bulk/aacudad/reasoning_traces/Training/zero3_offload.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --dataset_name ${DATASET_NAME} \
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
  --save_steps 530 \
  --num_train_epochs 2 \
  --single_img 1 \
  2>&1 | tee ${OUTPUT_DIR}/train.log
