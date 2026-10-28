#!/bin/bash
# Eval watchdog for GRPO-on-Arm-C run (grpo_qwen25vl_7b_abc_C_grpo).
# DS-MVTec on CUDA 1, VisA on CUDA 2 in parallel; training uses CUDA 0,3.

OUTPUT_DIR="/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_abc_C_grpo"
BASE_MODEL="/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376"
EVAL_SCRIPT="/bulk/aacudad/reasoning_traces/Training/evaluate_qwen25vl_7b_trainprompt.py"
WORK_DIR="/bulk/aacudad/reasoning_traces"
DSMVTEC_GPU=1
VISA_GPU=2
BATCH_SIZE=4
POLL_INTERVAL=60
SAVE_WAIT=120

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache

cd "$WORK_DIR"

echo "[grpo-c-watcher] Started at $(date)"
echo "[grpo-c-watcher] Polling $OUTPUT_DIR every ${POLL_INTERVAL}s"
echo "[grpo-c-watcher] DS-MVTec on CUDA $DSMVTEC_GPU | VisA on CUDA $VISA_GPU (parallel)"
echo "[grpo-c-watcher] Base model: $BASE_MODEL"

declare -A evaluated

while true; do
    for ckpt_dir in "$OUTPUT_DIR"/checkpoint-*/; do
        [[ -d "$ckpt_dir" ]] || continue
        ckpt_name=$(basename "$ckpt_dir")

        # only evaluate full model checkpoints (have config.json)
        [[ -f "$ckpt_dir/config.json" ]] || continue

        if [[ -z "${evaluated[$ckpt_name]}" ]]; then
            echo "[grpo-c-watcher] New checkpoint: $ckpt_name — waiting ${SAVE_WAIT}s for save to finish..."
            sleep $SAVE_WAIT

            DSMVTEC_OUT="$ckpt_dir/eval_dsmvtec_full_trainprompt.json"
            VISA_OUT="$ckpt_dir/eval_visa_full_trainprompt.json"

            # --- run DS-MVTec and VisA in parallel ---
            pids=()

            if [[ ! -f "$DSMVTEC_OUT" ]]; then
                echo "[grpo-c-watcher] [$ckpt_name] DS-MVTec eval on CUDA $DSMVTEC_GPU..."
                ( CUDA_VISIBLE_DEVICES=$DSMVTEC_GPU python "$EVAL_SCRIPT" \
                    --checkpoint "$ckpt_dir" \
                    --base-model "$BASE_MODEL" \
                    --ds-mvtec-only \
                    --batch-size $BATCH_SIZE \
                    --grpo-eval \
                    --output "$DSMVTEC_OUT" \
                    > "$ckpt_dir/eval_dsmvtec.log" 2>&1 ) &
                pids+=($!)
            else
                echo "[grpo-c-watcher] [$ckpt_name] DS-MVTec eval already exists, skipping"
            fi

            if [[ ! -f "$VISA_OUT" ]]; then
                echo "[grpo-c-watcher] [$ckpt_name] VisA eval on CUDA $VISA_GPU..."
                ( CUDA_VISIBLE_DEVICES=$VISA_GPU python "$EVAL_SCRIPT" \
                    --checkpoint "$ckpt_dir" \
                    --base-model "$BASE_MODEL" \
                    --visa-only \
                    --batch-size $BATCH_SIZE \
                    --grpo-eval \
                    --output "$VISA_OUT" \
                    > "$ckpt_dir/eval_visa.log" 2>&1 ) &
                pids+=($!)
            else
                echo "[grpo-c-watcher] [$ckpt_name] VisA eval already exists, skipping"
            fi

            # wait for both parallel evals to finish before moving on
            for pid in "${pids[@]}"; do
                wait "$pid"
            done

            # print headline result if eval json has balanced_accuracy
            if [[ -f "$DSMVTEC_OUT" ]]; then
                ds_bal=$(python -c "import json; d=json.load(open('$DSMVTEC_OUT')); print(d.get('balanced_accuracy', d.get('metrics',{}).get('balanced_accuracy','?')))" 2>/dev/null)
                echo "[grpo-c-watcher] [$ckpt_name] DS-MVTec balanced_accuracy = $ds_bal"
            fi
            if [[ -f "$VISA_OUT" ]]; then
                visa_bal=$(python -c "import json; d=json.load(open('$VISA_OUT')); print(d.get('balanced_accuracy', d.get('metrics',{}).get('balanced_accuracy','?')))" 2>/dev/null)
                echo "[grpo-c-watcher] [$ckpt_name] VisA balanced_accuracy = $visa_bal"
            fi

            evaluated[$ckpt_name]=1
            echo "[grpo-c-watcher] [$ckpt_name] DONE at $(date)"
        fi
    done

    sleep "$POLL_INTERVAL"
done
