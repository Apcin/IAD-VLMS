#!/bin/bash
# Watches for new checkpoints from 7B GRPO training (full fine-tuning) and runs
# DS-MVTec + VisA eval after each one.
#
# Usage (in a separate tmux session):
#   bash /bulk/aacudad/reasoning_traces/Training/watch_and_eval_grpo_7b.sh

OUTPUT_DIR="/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2"
BASE_MODEL="/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564"
EVAL_SCRIPT="/bulk/aacudad/reasoning_traces/Training/evaluate_qwen25vl_7b_trainprompt.py"
WORK_DIR="/bulk/aacudad/reasoning_traces"
EVAL_GPU=0
BATCH_SIZE=4
POLL_INTERVAL=60  # seconds between directory scans
SAVE_WAIT=120     # seconds to wait after detecting new checkpoint

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache

cd "$WORK_DIR"

echo "[grpo-7b-watcher] Started — polling $OUTPUT_DIR every ${POLL_INTERVAL}s"
echo "[grpo-7b-watcher] Eval GPU: $EVAL_GPU | Base model: $BASE_MODEL"

declare -A evaluated

while true; do
    for ckpt_dir in "$OUTPUT_DIR"/checkpoint-*/; do
        [[ -d "$ckpt_dir" ]] || continue
        ckpt_name=$(basename "$ckpt_dir")

        # Only evaluate full model checkpoints (have config.json)
        [[ -f "$ckpt_dir/config.json" ]] || continue

        if [[ -z "${evaluated[$ckpt_name]}" ]]; then
            echo "[grpo-7b-watcher] New checkpoint: $ckpt_name — waiting ${SAVE_WAIT}s for save to finish..."
            sleep $SAVE_WAIT

            # --- DS-MVTec eval ---
            DSMVTEC_OUT="$ckpt_dir/eval_dsmvtec_full_trainprompt.json"
            if [[ ! -f "$DSMVTEC_OUT" ]]; then
                echo "[grpo-7b-watcher] Running DS-MVTec eval for $ckpt_name on CUDA $EVAL_GPU..."
                CUDA_VISIBLE_DEVICES=$EVAL_GPU python "$EVAL_SCRIPT" \
                    --checkpoint "$ckpt_dir" \
                    --base-model "$BASE_MODEL" \
                    --ds-mvtec-only \
                    --batch-size $BATCH_SIZE \
                    --grpo-eval \
                    --output "$DSMVTEC_OUT"
                echo "[grpo-7b-watcher] DS-MVTec eval done for $ckpt_name"
            else
                echo "[grpo-7b-watcher] DS-MVTec eval already exists for $ckpt_name, skipping"
            fi

            # --- VisA eval ---
            VISA_OUT="$ckpt_dir/eval_visa_full_trainprompt.json"
            if [[ ! -f "$VISA_OUT" ]]; then
                echo "[grpo-7b-watcher] Running VisA eval for $ckpt_name on CUDA $EVAL_GPU..."
                CUDA_VISIBLE_DEVICES=$EVAL_GPU python "$EVAL_SCRIPT" \
                    --checkpoint "$ckpt_dir" \
                    --base-model "$BASE_MODEL" \
                    --visa-only \
                    --batch-size $BATCH_SIZE \
                    --grpo-eval \
                    --output "$VISA_OUT"
                echo "[grpo-7b-watcher] VisA eval done for $ckpt_name"
            else
                echo "[grpo-7b-watcher] VisA eval already exists for $ckpt_name, skipping"
            fi

            evaluated[$ckpt_name]=1
            echo "[grpo-7b-watcher] All evals complete for $ckpt_name"
        fi
    done

    sleep "$POLL_INTERVAL"
done
