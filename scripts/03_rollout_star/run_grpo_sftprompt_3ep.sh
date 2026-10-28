#!/bin/bash
# 3-epoch GRPO from Arm-C 82.80/72.07, β=0.1, EVAL-ALIGNED prompt
# (--prompt_style sft_sys = system "Please answer by yes or no" + "Analyze the
# {product}...", byte-identical to the eval). TRAIN-ONLY on CUDA 1,2 — evaluation
# is handled by the decoupled cuda-3 watcher (watch_and_eval_cuda3.sh), so there
# is NO inline eval here. Saves a checkpoint every 106 steps (15 over 1,590).
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# --prompt_style lives in the in-repo stage_rl snapshot, so that is the default.
# Override GRPO_STAGE_RL to run against another checkout.
GRPO_STAGE_RL="${GRPO_STAGE_RL:-$REPO_ROOT/scripts/02_grpo/stage_rl}"
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export PYTHONPATH=$GRPO_STAGE_RL
export GEMINI_JUDGE_URL=http://127.0.0.1:5200
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1; export NCCL_DEBUG=WARN; unset NCCL_P2P_DISABLE
export WANDB_MODE=disabled; export DEBUG_MODE=True

BETA=0.1
MODEL=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376
DATA=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json
OUT=/bulk/aacudad/reasoning_traces/outputs/grpo_sftprompt_kl0.1_sys_3ep
export LOG_PATH=$OUT/reward.log
mkdir -p "$OUT"

echo "════ GRPO sft_sys (eval-aligned) | β=$BETA | CUDA 1,2 | 3 epochs | $(date) ════"

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --nnodes=1 --master_port=29551 \
  $GRPO_STAGE_RL/grpo_ad.py \
  --deepspeed $REPO_ROOT/configs/deepspeed/zero3_offload.json \
  --output_dir "$OUT" --model_name_or_path "$MODEL" --dataset_name "$DATA" \
  --image_path / --use_vllm_for_gen false --use_system_prompt false \
  --prompt_style sft_sys \
  --max_prompt_length 4096 --max_completion_length 512 \
  --num_generations 4 --reward_funcs accuracy format \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --beta $BETA --learning_rate 1e-6 \
  --logging_steps 1 --bf16 true --report_to none --gradient_checkpointing true \
  --attn_implementation sdpa --max_pixels 480000 \
  --save_steps 106 --save_only_model true --num_train_epochs 3 --single_img 1 \
  2>&1 | tee "$OUT/train.log"

echo "[$(date)] 3-epoch training DONE."
touch "$OUT/.traindone"
# Eval is done by watch_and_eval_cuda3.sh (CUDA 3, sequential). No inline eval.
