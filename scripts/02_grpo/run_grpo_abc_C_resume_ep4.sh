#!/bin/bash
# RESUME GRPO-on-C from ckpt-1060 (end of epoch 2) -> end of epoch 4.
#
# Full DeepSpeed state lives in checkpoint-1060/global_step1060 (2-rank optim
# shards), so we MUST resume on exactly 2 ranks. We keep the same NVLink pair
# (CUDA 0,3) and the same recipe as the original run.
#
# KEY DIFFERENCE vs the original launch: --save_only_model true. New checkpoints
# (1325/1590/1855/2120) save the 16 GB model ONLY, not the ~93 GB DeepSpeed
# optimizer dir -> no "no space left on device". --save_total_limit 6 caps the
# number of retained model checkpoints.

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export PYTHONPATH=/bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl

export MODEL_NAME_OR_PATH=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376
export DATASET_NAME=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json
export OUTPUT_DIR=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_abc_C_grpo
export RESUME_CKPT=${OUTPUT_DIR}/checkpoint-1060

export WANDB_MODE=disabled
export DEBUG_MODE=True
export LOG_PATH=${OUTPUT_DIR}/reward_resume.log

# GPU0 is occupied by an unrelated long-running job (emc_ensemble_ips), so use
# the OTHER NVLink NV4 pair: CUDA 1,2 (both idle). Leaves GPU3 free for eval.
export CUDA_VISIBLE_DEVICES=1,2
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TRITON_CACHE_DIR

unset NCCL_P2P_DISABLE
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

mkdir -p ${OUTPUT_DIR}

echo "════════════════════════════════════════════════════════════════════════"
echo " RESUME GRPO-on-C  ckpt-1060 (ep2) -> ep4   2 GPUs (CUDA 1,2 NVLink NV4)"
echo " Resume:  ${RESUME_CKPT}"
echo " Output:  ${OUTPUT_DIR}"
echo " SAVE:    model-only (save_only_model=true), save_total_limit=6"
echo " Start:   $(date)"
echo "════════════════════════════════════════════════════════════════════════"

torchrun --nproc_per_node=2 --nnodes=1 --master_port=29511 \
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
  --save_steps 265 \
  --num_train_epochs 4 \
  --save_only_model true \
  --save_total_limit 6 \
  --resume_from_checkpoint ${RESUME_CKPT} \
  --single_img 1 \
  2>&1 | tee ${OUTPUT_DIR}/train_resume.log

echo "[$(date)] GRPO-on-C RESUME DONE."
touch ${OUTPUT_DIR}/train_resume.done
