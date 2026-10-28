#!/bin/bash
# KL-penalty drift test: 0.5 epoch GRPO on top of the Arm-C SFT (82.80/72.07),
# IDENTICAL recipe to run_grpo_abc_C_4gpu.sh EXCEPT:
#   - KL penalty turned ON   (--beta $BETA ; prior runs used beta=0.0 = no penalty)
#   - 0.5 epoch only          (--max_steps 265 ; 4236 prompts / EBS 8 = 530 steps/epoch)
#   - intermediate checkpoints (--save_steps 53 -> ckpts 53/106/159/212/265 for a drift curve)
#   - runs on CUDA 1,2        (0=embed server, 3=held-out rollout)
# Same dataset (grpo_train.json 4k), same LR (1e-6 default), same G=4, same prompt.
# REWARD: accuracy + format only (--reward_funcs accuracy format). We DROP the
# "reasoning" reward: its :5100 judge is down so it returned a flat ~0.5, which
# (a) is cancelled by GRPO's per-group advantage normalisation (so it never changed
# gradients vs abc_C) and (b) only cluttered the logged reward. Dropping it makes
# the reward = the core accuracy+format signal and the logs clean.
# Embed server :5200 must be UP for the type-similarity part of accuracy_reward.
set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export PYTHONPATH=/bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl
export GEMINI_JUDGE_URL=http://127.0.0.1:5200   # embed/type-similarity server

# ===== the one knob under test =====
BETA=0.1
# ===================================

export MODEL_NAME_OR_PATH=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376
export DATASET_NAME=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json
export OUTPUT_DIR=/bulk/aacudad/reasoning_traces/outputs/grpo_abc_C_kl${BETA}_halfep
export WANDB_MODE=disabled
export DEBUG_MODE=True
export LOG_PATH=${OUTPUT_DIR}/reward.log

export CUDA_VISIBLE_DEVICES=1,2
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TRITON_CACHE_DIR $OUTPUT_DIR
unset NCCL_P2P_DISABLE
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

EVAL=/bulk/aacudad/reasoning_traces/Training/evaluate_qwen25vl_7b_trainprompt.py
BASE=Qwen/Qwen2.5-VL-7B-Instruct

echo "════════════════════════════════════════════════════════════════════════"
echo " GRPO KL-DRIFT TEST — 0.5 epoch, beta=${BETA}, CUDA 1,2"
echo " Init:    ${MODEL_NAME_OR_PATH}  (Arm-C 82.80/72.07)"
echo " Dataset: ${DATASET_NAME}  (4,236 prompts)"
echo " Output:  ${OUTPUT_DIR}"
echo " Steps:   265 (0.5 epoch), save every 53 -> 5 ckpts"
echo " Start:   $(date)"
echo "════════════════════════════════════════════════════════════════════════"

# ===== Stage 1: train =====
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
  --reward_funcs accuracy format \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --beta ${BETA} \
  --logging_steps 1 \
  --bf16 true \
  --report_to none \
  --gradient_checkpointing true \
  --attn_implementation sdpa \
  --max_pixels 480000 \
  --save_steps 53 \
  --save_only_model true \
  --max_steps 265 \
  --single_img 1 \
  2>&1 | tee ${OUTPUT_DIR}/train.log

echo "[$(date)] GRPO-KL training DONE."
touch ${OUTPUT_DIR}/.traindone
# Evaluation is handled by the decoupled watchdog watch_and_eval_grpo_kl.sh
# (tmux 'grpo_eval'), which fires on .traindone: DS->cuda1, VisA->cuda2.
