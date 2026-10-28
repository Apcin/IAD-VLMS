#!/bin/bash
# GRPO on top of the Arm-C SFT (kept + corrected + rewritten, 6K, from base).
# Init from the best Arm-C checkpoint:
#   outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376
#   (82.80% DS-MVTec / 72.07% VisA)
#
# Uses the production GRPO recipe (4-component reward via nomic-similarity type reward,
# group size G=4, KL β=0.04, LR 5e-6).
#
# 2 GPUs (CUDA 0,3 — NVLink NV4-paired) — ~22 h estimated, matching run-2's
# 2-GPU throughput. We tried 4 GPUs first; cross-pair NCCL was the bottleneck.

set -eo pipefail

# Activate the conda env that has torch + transformers + deepspeed + trl (with GRPOConfig) + peft
# (same env the headline run-2 used)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export PYTHONPATH=/bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl

# === Init checkpoint = Arm-C ep2 (best on both benchmarks) =====================
export MODEL_NAME_OR_PATH=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376

# Same GRPO training set as the published run-2
export DATASET_NAME=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json

# Output dir
export OUTPUT_DIR=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_abc_C_grpo

# === Runtime knobs ============================================================
export WANDB_MODE=disabled
export DEBUG_MODE=True
export LOG_PATH=${OUTPUT_DIR}/reward.log

# Fall back to 2 GPUs on the NVLink NV4-paired pair (GPU0 ↔ GPU3 = NV4 direct).
# Run-2 used CUDA 1,2 (also an NV4 pair) at ~75-80 s/step; the same pattern on
# 0,3 should give the same throughput. 4 GPUs across the mixed NVLink/PCIe
# topology turned out slower (148 s/step under NCCL_P2P_DISABLE) than 2 GPUs
# over a single NVLink, so we keep the proven 2-GPU recipe.
export CUDA_VISIBLE_DEVICES=0,3
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TRITON_CACHE_DIR

# NVLink-paired GPUs: keep P2P enabled (the previous 4-GPU hang was the
# cross-pair PCIe path during the first BROADCAST; NV4 direct does not have
# that issue).
unset NCCL_P2P_DISABLE
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

mkdir -p ${OUTPUT_DIR}

# === Launch ===================================================================
echo "════════════════════════════════════════════════════════════════════════"
echo " GRPO on Arm-C ckpt-376 — 2 GPUs (CUDA 0,3 NVLink NV4)"
echo " Init:    ${MODEL_NAME_OR_PATH}"
echo " Dataset: ${DATASET_NAME}"
echo " Output:  ${OUTPUT_DIR}"
echo " GPUs:    CUDA ${CUDA_VISIBLE_DEVICES}"
echo " Start:   $(date)"
echo "════════════════════════════════════════════════════════════════════════"

torchrun --nproc_per_node=2 --nnodes=1 --master_port=29510 \
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
  --single_img 1 \
  2>&1 | tee ${OUTPUT_DIR}/train.log

echo "[$(date)] GRPO-on-C DONE."
touch ${OUTPUT_DIR}/train.done
