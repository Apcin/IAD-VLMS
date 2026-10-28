#!/bin/bash
# GRPO epoch 2 restart: continue from the epoch-1 GRPO checkpoint (ckpt-530,
# DS 87.66 / VisA 72.58) after the Aug 30 maintenance killed the original run at
# step 846. Identical recipe to run_grpo_llava_ep1.sh (rewards accuracy+format,
# G=4, beta=0 (TRL default), zero3_offload, GRPO_MAX_IMAGE_PIXELS=480000,
# save_only_model). Fresh output dir; 530 steps = one epoch; weights saved at end.
# Note: dataloader order restarts from the seed (epoch-1 order), a minor deviation
# from an uninterrupted 2-epoch run's reshuffle.
# Usage: run_grpo_llava_ep2.sh <gpus> <port> [max_steps] [save_steps]
set -eo pipefail
GPUS="${1:?gpus e.g. 1,2}"; PORT="${2:?port}"; MAXSTEPS="${3:-530}"; SAVESTEPS="${4:-530}"
# Paths. WORK_DIR is the workspace that holds outputs/, hf_cache/, Training/ and the
# GRPO trainer checkout. It defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
# The LLaVA path needs the SigLIP ZeRO-3 init guard and GRPO_MAX_IMAGE_PIXELS.
# Both are in the in-repo snapshot since the stage_rl re-sync, so that snapshot is
# the default. Override GRPO_STAGE_RL to run against another checkout.
GRPO_STAGE_RL="${GRPO_STAGE_RL:-$REPO_ROOT/scripts/02_grpo/stage_rl}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
export HF_HOME=$WORK_DIR/hf_cache
export PYTHONPATH=$GRPO_STAGE_RL
export MODEL_NAME_OR_PATH=$WORK_DIR/outputs/grpo_llava_ov_from_ep1/checkpoint-530
export DATASET_NAME=$WORK_DIR/Training/datasets_small_15k_c1_only/grpo_train.json
export OUTPUT_DIR=$WORK_DIR/outputs/grpo_llava_ov_from_ep1_ep2
export WANDB_MODE=disabled
export GEMINI_JUDGE_URL=http://127.0.0.1:5300
export GRPO_MAX_IMAGE_PIXELS=480000
export CUDA_VISIBLE_DEVICES=${GPUS}
export TRITON_CACHE_DIR=$WORK_DIR/tmp_cache/triton_grpo_llava
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NCCL_P2P_DISABLE; export NCCL_IB_DISABLE=1; export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$TRITON_CACHE_DIR" "$OUTPUT_DIR"
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
echo "════ GRPO-LLaVA EPOCH 2 from ckpt-530 | GPUs=$GPUS steps=$MAXSTEPS save=$SAVESTEPS ════"
torchrun --nproc_per_node=${NGPU} --nnodes=1 --master_port=${PORT} \
  $GRPO_STAGE_RL/grpo_ad.py \
  --deepspeed $REPO_ROOT/configs/deepspeed/zero3_offload.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --dataset_name ${DATASET_NAME} \
  --reward_funcs accuracy format \
  --image_path / --use_vllm_for_gen false --use_system_prompt false \
  --max_prompt_length 8192 --max_completion_length 512 \
  --num_generations 4 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --logging_steps 1 --bf16 true --report_to none --gradient_checkpointing true \
  --attn_implementation sdpa \
  --save_steps ${SAVESTEPS} --max_steps ${MAXSTEPS} \
  --save_only_model true \
  --ddp_timeout 3600 \
  --single_img 1 \
  2>&1 | tee -a ${OUTPUT_DIR}/train.log
echo "[$(date)] GRPO-LLAVA EPOCH2 RUN DONE."
touch ${OUTPUT_DIR}/train.done
