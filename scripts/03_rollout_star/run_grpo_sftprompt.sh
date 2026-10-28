#!/bin/bash
# 1-epoch GRPO from Arm-C 82.80/72.07 with the SFT/eval-aligned prompt
# (--prompt_style sft), so train and eval use the SAME prompt. Parameterized to
# run two betas in parallel on the two NVLink pairs.
# Env: BETA, GPUS (e.g. "1,2"), PORT, OUTNAME, EVAL_DS_GPU, EVAL_VISA_GPU
# Full CPU offload (zero3_offload.json) so GPU memory stays low (two jobs + embed
# server coexist). Reward = accuracy + format (reasoning dropped, as in the KL run).
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

BETA="${BETA:?}"; GPUS="${GPUS:?}"; PORT="${PORT:?}"; OUTNAME="${OUTNAME:?}"
EVAL_DS_GPU="${EVAL_DS_GPU:?}"; EVAL_VISA_GPU="${EVAL_VISA_GPU:?}"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
MODEL=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376
DATA=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json
OUT=/bulk/aacudad/reasoning_traces/outputs/$OUTNAME
EVAL=/bulk/aacudad/reasoning_traces/Training/evaluate_qwen25vl_7b_trainprompt.py
BASE=Qwen/Qwen2.5-VL-7B-Instruct
export LOG_PATH=$OUT/reward.log
mkdir -p "$OUT"

echo "════ GRPO SFT-prompt | beta=$BETA | GPUS=$GPUS | out=$OUTNAME | $(date) ════"

# ── train: 1 epoch, save every 106 (5 ckpts), model-only ──
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --nnodes=1 --master_port=$PORT \
  $GRPO_STAGE_RL/grpo_ad.py \
  --deepspeed $REPO_ROOT/configs/deepspeed/zero3_offload.json \
  --output_dir "$OUT" --model_name_or_path "$MODEL" --dataset_name "$DATA" \
  --image_path / --use_vllm_for_gen false --use_system_prompt false \
  --prompt_style sft \
  --max_prompt_length 4096 --max_completion_length 512 \
  --num_generations 4 --reward_funcs accuracy format \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --beta $BETA --learning_rate 1e-6 \
  --logging_steps 1 --bf16 true --report_to none --gradient_checkpointing true \
  --attn_implementation sdpa --max_pixels 480000 \
  --save_steps 106 --save_only_model true --num_train_epochs 1 --single_img 1 \
  2>&1 | tee "$OUT/train.log"
echo "[$(date)] train done."; touch "$OUT/.traindone"

# ── eval each checkpoint on this job's own pair (DS->EVAL_DS_GPU, VisA->EVAL_VISA_GPU) ──
for ck in $(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  [ -f "$ck/config.json" ] || continue
  echo "[$(date +%T)] EVAL $(basename $ck): DS->cuda$EVAL_DS_GPU VisA->cuda$EVAL_VISA_GPU"
  CUDA_VISIBLE_DEVICES=$EVAL_DS_GPU python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --ds-mvtec-only \
      --output "$ck/eval_dsmvtec_full_trainprompt.json" > "$ck/eval_ds.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=$EVAL_VISA_GPU python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --visa-only \
      --output "$ck/eval_visa_full_trainprompt.json" > "$ck/eval_visa.log" 2>&1 &
  wait
done
touch "$OUT/.evaldone"

# ── BA table ──
echo "════ BA TABLE ($OUTNAME, beta=$BETA) ════" | tee "$OUT/ba_table.txt"
for ck in $(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  for b in dsmvtec visa; do f="$ck/eval_${b}_full_trainprompt.json"
    [ -f "$f" ] && python3 /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/results/compute_ba.py "$f" 2>/dev/null \
        | sed "s|^|$(basename $ck) $b: |" | tee -a "$OUT/ba_table.txt"; done
done
echo "[$(date)] ALL DONE -> $OUT/ba_table.txt"; touch "$OUT/done.flag"
