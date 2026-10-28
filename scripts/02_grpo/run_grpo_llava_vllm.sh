#!/bin/bash
# GRPO on LLaVA-OV ep1 ckpt — vLLM-accelerated generation variant (_vllm stack, py3.12 env).
# Training ranks on <gpus>; vLLM V1 engine in-process on rank0, DEDICATED physical GPU <vllm_gpu>.
# Usage: run_grpo_llava_vllm.sh <train_gpus e.g. 1,2> <vllm_gpu e.g. 3> <port> [max_steps] [save_steps]
set -eo pipefail
GPUS="${1:?train gpus}"; VGPU="${2:?vllm gpu}"; PORT="${3:?port}"; MAXSTEPS="${4:-2120}"; SAVESTEPS="${5:-530}"
# Paths. WORK_DIR is the workspace that holds outputs/, hf_cache/, Training/ and the
# GRPO trainer checkout. It defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
# NOTE: this launcher needs the vLLM-generation variant of the trainer
# (iad_r1_grpo_custom_vllm/stage_rl), which is a separate checkout and is NOT the
# snapshot in scripts/02_grpo/stage_rl/. The in-repo snapshot mirrors the
# HF-generation stack used by every other GRPO launcher here. Only the vLLM
# variant has the in-process vLLM engine and the weight-sync path that
# --use_vllm_for_gen true drives, so GRPO_STAGE_RL must point at that checkout.
GRPO_STAGE_RL="${GRPO_STAGE_RL:-$WORK_DIR/Training/iad_r1_grpo_custom_vllm/stage_rl}"

# Python 3.12 env with vLLM + TRL. Override with GRPO_FAST_ENV.
ENV="${GRPO_FAST_ENV:-$WORK_DIR/envs/grpo_fast312}"
export PATH="${ENV}/bin:$PATH"
export HF_HOME=$WORK_DIR/hf_cache
export PYTHONPATH=$GRPO_STAGE_RL
export MODEL_NAME_OR_PATH=$WORK_DIR/outputs/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188
export DATASET_NAME=$WORK_DIR/Training/datasets_small_15k_c1_only/grpo_train.json
export OUTPUT_DIR=$WORK_DIR/outputs/grpo_llava_ov_from_ep1_vllm
export WANDB_MODE=disabled
export GEMINI_JUDGE_URL=http://127.0.0.1:5300
export GRPO_MAX_IMAGE_PIXELS=480000
export GRPO_VLLM_DEVICE=${VGPU}
export GRPO_VLLM_MAX_MODEL_LEN=10240
export GRPO_VLLM_SYNC_EVERY=4
export CUDA_VISIBLE_DEVICES=${GPUS}
export TRITON_CACHE_DIR=$WORK_DIR/tmp_cache/triton_grpo_llava_vllm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NCCL_P2P_DISABLE; export NCCL_IB_DISABLE=1; export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
if [ "$MAXSTEPS" -le 10 ]; then
  export GRPO_COMPLETION_LOG=${OUTPUT_DIR}/smoke_completions.log
  export GRPO_VLLM_SYNC_DEBUG=${OUTPUT_DIR}/smoke_weightsync.log
fi
mkdir -p "$TRITON_CACHE_DIR" "$OUTPUT_DIR"
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
export GRPO_VLLM_PORT=${GRPO_VLLM_PORT:-8009}
export GRPO_VLLM_GROUP_PORT=${GRPO_VLLM_GROUP_PORT:-51316}
# ── start the vLLM server on the dedicated GPU (separate process, own tmux) ──
tmux kill-session -t vllm_server 2>/dev/null || true
tmux new-session -d -s vllm_server "HF_HOME=$HF_HOME NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES=${VGPU} ${ENV}/bin/trl vllm-serve   --model ${MODEL_NAME_OR_PATH} --port ${GRPO_VLLM_PORT} --gpu-memory-utilization ${GRPO_VLLM_GPU_UTIL:-0.5}   --max-model-len ${GRPO_VLLM_MAX_MODEL_LEN} --dtype bfloat16 2>&1 | tee ${OUTPUT_DIR}/vllm_server.log"
echo "waiting for vllm server health..."
for i in $(seq 1 120); do
  curl -s -m 3 "http://127.0.0.1:${GRPO_VLLM_PORT}/health/" >/dev/null 2>&1 && { echo "vllm server UP"; break; }
  sleep 5
done
echo "════ GRPO-LLaVA + vLLM-server | train=$GPUS vllm=$VGPU steps=$MAXSTEPS ════"
${ENV}/bin/torchrun --nproc_per_node=${NGPU} --nnodes=1 --master_port=${PORT} \
  $GRPO_STAGE_RL/grpo_ad.py \
  --deepspeed $REPO_ROOT/configs/deepspeed/zero3_offload.json \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --dataset_name ${DATASET_NAME} \
  --reward_funcs accuracy format \
  --image_path / --use_vllm_for_gen true --use_system_prompt false \
  --max_prompt_length 8192 --max_completion_length 512 \
  --num_generations 4 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --logging_steps 1 --bf16 true --report_to none --gradient_checkpointing true \
  --attn_implementation sdpa \
  --save_steps ${SAVESTEPS} --max_steps ${MAXSTEPS} \
  --save_only_model true \
  --ddp_timeout 3600 \
  --single_img 1 \
  2>&1 | tee ${OUTPUT_DIR}/train.log
echo "[$(date)] GRPO-LLAVA-VLLM DONE."
touch ${OUTPUT_DIR}/train.done
