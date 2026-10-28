#!/bin/bash
# Watches the NO-REASONING (labels-only) SFT run for new checkpoints and evaluates each one
# sequentially: DS-MVTec first, then VisA.
#
# IMPORTANT -- prompt mode: this baseline's NATIVE mode is `noreasonprompt`
#   (no system message; "<image>\nAnalyze the provided image of the {product}. Determine if there
#    are any anomalies present. Answer with yes or no."), which matches its training data exactly.
# Per docs/prompt_modes.md every model is scored under its native mode, so we pass
# --noreason-prompt and write the matching _noreasonprompt suffix. Do NOT score this model
# under _trainprompt: it was never trained to emit reasoning and the suffix must not lie.
#
# Run inside tmux:  tmux new -s eval_noreason 'bash scripts/04_eval/watch_and_eval_noreason.sh'

# Paths. WORK_DIR is the workspace holding outputs/ and hf_cache/; it defaults to the
# parent of this repository. On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/outputs/sft_qwen25vl_7b_6k_noreason}"
EVAL_SCRIPT="$HERE/evaluate_qwen25vl_7b_trainprompt.py"
EVAL_GPU=2          # GPU 2 shares with train_vae.py (13GB used, ~36GB free) -- a 7B eval fits
BATCH_SIZE=4
POLL_INTERVAL=60    # seconds between directory scans
SAVE_WAIT=120       # seconds to let a new checkpoint finish writing

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=$WORK_DIR/hf_cache
export TRANSFORMERS_CACHE=$WORK_DIR/hf_cache

cd "$WORK_DIR"

echo "[noreason-watcher] started - polling $OUTPUT_DIR every ${POLL_INTERVAL}s"
echo "[noreason-watcher] eval GPU: $EVAL_GPU | mode: noreasonprompt (--noreason-prompt)"

declare -A evaluated

while true; do
    for ckpt_dir in "$OUTPUT_DIR"/checkpoint-*/; do
        [[ -d "$ckpt_dir" ]] || continue
        ckpt_name=$(basename "$ckpt_dir")

        # only full model checkpoints
        [[ -f "$ckpt_dir/config.json" ]] || continue

        if [[ -z "${evaluated[$ckpt_name]}" ]]; then
            echo "[noreason-watcher] new checkpoint: $ckpt_name - waiting ${SAVE_WAIT}s for save to finish..."
            sleep $SAVE_WAIT

            # --- DS-MVTec ---
            DSMVTEC_OUT="$ckpt_dir/eval_dsmvtec_full_noreasonprompt.json"
            if [[ ! -f "$DSMVTEC_OUT" ]]; then
                echo "[noreason-watcher] DS-MVTec eval for $ckpt_name on CUDA $EVAL_GPU..."
                CUDA_VISIBLE_DEVICES=$EVAL_GPU python "$EVAL_SCRIPT" \
                    --checkpoint "$ckpt_dir" \
                    --ds-mvtec-only \
                    --batch-size $BATCH_SIZE \
                    --noreason-prompt \
                    --output "$DSMVTEC_OUT"
                echo "[noreason-watcher] DS-MVTec done for $ckpt_name"
            else
                echo "[noreason-watcher] DS-MVTec already exists for $ckpt_name, skipping"
            fi

            # --- VisA ---
            VISA_OUT="$ckpt_dir/eval_visa_full_noreasonprompt.json"
            if [[ ! -f "$VISA_OUT" ]]; then
                echo "[noreason-watcher] VisA eval for $ckpt_name on CUDA $EVAL_GPU..."
                CUDA_VISIBLE_DEVICES=$EVAL_GPU python "$EVAL_SCRIPT" \
                    --checkpoint "$ckpt_dir" \
                    --visa-only \
                    --batch-size $BATCH_SIZE \
                    --noreason-prompt \
                    --output "$VISA_OUT"
                echo "[noreason-watcher] VisA done for $ckpt_name"
            else
                echo "[noreason-watcher] VisA already exists for $ckpt_name, skipping"
            fi

            evaluated[$ckpt_name]=1
            echo "[noreason-watcher] all evals complete for $ckpt_name"
            python "$REPO_ROOT/results/compute_ba.py" "$DSMVTEC_OUT" "$VISA_OUT" 2>/dev/null || true
        fi
    done
    sleep "$POLL_INTERVAL"
done
