#!/bin/bash
# Short dense-checkpoint GRPO burst from Arm-C ckpt-376 to test the advantage-estimator
# hypothesis. Identical recipe to the production GRPO-on-C run EXCEPT: max_steps 120,
# save_steps 20 (dense), and the advantage estimator selected by ARM.
#
# Usage: run_grpo_probe_arm.sh <ctrl|drgrpo|g2rpo> <CUDA_DEVICES> <MASTER_PORT>
#   e.g. run_grpo_probe_arm.sh ctrl   0,3 29531
#        run_grpo_probe_arm.sh drgrpo 0,3 29532
#        run_grpo_probe_arm.sh g2rpo  0,3 29533
set -eo pipefail
ARM="${1:?arm: ctrl|drgrpo|g2rpo}"
GPUS="${2:?cuda devices, e.g. 0,3}"
PORT="${3:?master port}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export PYTHONPATH=/bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl
export MODEL_NAME_OR_PATH=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376
export DATASET_NAME=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json
export OUTPUT_DIR=/bulk/aacudad/reasoning_traces/outputs/grpo_probe_${ARM}
export WANDB_MODE=disabled
export LOG_PATH=${OUTPUT_DIR}/reward.log
export GEMINI_JUDGE_URL=http://127.0.0.1:5200   # nomic type-reward server
export CUDA_VISIBLE_DEVICES=${GPUS}
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton_${ARM}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NCCL_P2P_DISABLE; export NCCL_IB_DISABLE=1; export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1; export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
mkdir -p "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

# advantage-estimator flag per arm
ARMFLAG=""
case "$ARM" in
  ctrl)   ARMFLAG="" ;;                       # vanilla group z-score (/std)
  drgrpo) ARMFLAG="--use_drgrpo true" ;;      # mean-center only, no /std
  g2rpo)  ARMFLAG="--use_g2rpo true" ;;       # rank->Gaussian-quantile (tie-robust)
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
echo "════════════════════════════════════════════════════════════════"
echo " GRPO PROBE ARM = ${ARM}   GPUs=${GPUS} (${NGPU})   port=${PORT}"
echo " init: Arm-C ckpt-376   max_steps=120  save_steps=20  G=4  flag: ${ARMFLAG:-<vanilla>}"
echo " out:  ${OUTPUT_DIR}    start: $(date)"
echo "════════════════════════════════════════════════════════════════"

torchrun --nproc_per_node=${NGPU} --nnodes=1 --master_port=${PORT} \
  /bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl/grpo_ad.py \
  --deepspeed /bulk/aacudad/reasoning_traces/Training/zero3_offload.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --dataset_name ${DATASET_NAME} \
  --image_path / --use_vllm_for_gen false --use_system_prompt false \
  --max_prompt_length 4096 --max_completion_length 512 \
  --num_generations 4 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --logging_steps 1 --bf16 true --report_to none --gradient_checkpointing true \
  --attn_implementation sdpa --max_pixels 480000 \
  --save_steps 20 --max_steps 120 ${ARMFLAG} \
  --single_img 1 \
  2>&1 | tee ${OUTPUT_DIR}/train.log

echo "[$(date)] PROBE ARM ${ARM} DONE."
touch ${OUTPUT_DIR}/train.done
