#!/bin/bash
# GRPO training using IAD-R1's SCGRPOTrainer + our custom type_reward
# Model: sft_qwen25vl_3b_zeroshot_6k_frozen  Dataset: datasets_small_15k_c1_only/grpo_train.json
# Full fine-tuning (no LoRA) with DeepSpeed ZeRO-3, single GPU

set -e

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache

# Use our custom stage_rl as PYTHONPATH (has our type_reward.py)
export PYTHONPATH=/bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl

# Paths
export MODEL_NAME_OR_PATH=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_3b_zeroshot_6k_frozen
export DATASET_NAME=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json
export OUTPUT_DIR=/bulk/aacudad/reasoning_traces/outputs/grpo_3b_6k_frozen_full_run1

# Wandb
export WANDB_MODE=disabled

# Debug log
export DEBUG_MODE=True
export LOG_PATH=${OUTPUT_DIR}/reward.log

export CUDA_VISIBLE_DEVICES=0,3
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=2 --nnodes=1 --master_port=29505 \
  /bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl/grpo_ad.py \
  --deepspeed /bulk/aacudad/reasoning_traces/iad_r1_unchanged/IAD-R1/scripts/train/zero3.json \
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
  --save_steps 315 \
  --num_train_epochs 1 \
  --single_img 1 \
  2>&1 | tee ${OUTPUT_DIR}/train.log
