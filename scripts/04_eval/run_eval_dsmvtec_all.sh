#!/bin/bash
# Evaluate all 4 checkpoints of all 4 trained models on DS-MVTec (full 1670 samples).
# Strategy: 1 GPU per model, 2 checkpoints in parallel per GPU per wave, 2 waves.
#   GPU 0 -> 3B frozen
#   GPU 1 -> 3B unfrozen
#   GPU 2 -> 7B frozen   (no root-dir merged model; checkpoint-1812 dir used for ep4)
#   GPU 3 -> 7B unfrozen

BASE_DIR="/bulk/aacudad/reasoning_traces"
EVAL_SCRIPT="Training/evaluate_qwen25vl_7b_trainprompt.py"
LOGS_DIR="$BASE_DIR/logs/eval_dsmvtec_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS_DIR"

echo "=== DS-MVTec full eval — $(date) ==="
echo "Logs: $LOGS_DIR"
echo ""

# ---------------------------------------------------------------------------
# GPU 0 — 3B frozen
# ---------------------------------------------------------------------------
(
    cd "$BASE_DIR"
    MODEL="sft_qwen25vl_3b_15k_frozen"
    BASE="outputs/$MODEL"
    GPU=0
    BS=8

    echo "[GPU $GPU] $MODEL — wave 1 start"
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-453" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-453/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-906" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-906/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep2.log" 2>&1 &
    wait
    echo "[GPU $GPU] $MODEL — wave 1 done, starting wave 2"

    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-1359" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-1359/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep3.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-1812" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-1812/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep4.log" 2>&1 &
    wait
    echo "[GPU $GPU] $MODEL — ALL DONE"
) &
GPU0_PID=$!

# ---------------------------------------------------------------------------
# GPU 1 — 3B unfrozen
# ---------------------------------------------------------------------------
(
    cd "$BASE_DIR"
    MODEL="sft_qwen25vl_3b_15k_unfrozen"
    BASE="outputs/$MODEL"
    GPU=1
    BS=8

    echo "[GPU $GPU] $MODEL — wave 1 start"
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-453" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-453/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-906" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-906/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep2.log" 2>&1 &
    wait
    echo "[GPU $GPU] $MODEL — wave 1 done, starting wave 2"

    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-1359" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-1359/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep3.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-1812" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-1812/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep4.log" 2>&1 &
    wait
    echo "[GPU $GPU] $MODEL — ALL DONE"
) &
GPU1_PID=$!

# ---------------------------------------------------------------------------
# GPU 2 — 7B frozen
# ---------------------------------------------------------------------------
(
    cd "$BASE_DIR"
    MODEL="sft_qwen25vl_7b_15k_frozen"
    BASE="outputs/$MODEL"
    GPU=2
    BS=4

    echo "[GPU $GPU] $MODEL — wave 1 start"
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-453" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-453/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-906" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-906/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep2.log" 2>&1 &
    wait
    echo "[GPU $GPU] $MODEL — wave 1 done, starting wave 2"

    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-1359" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-1359/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep3.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-1812" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-1812/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep4.log" 2>&1 &
    wait
    echo "[GPU $GPU] $MODEL — ALL DONE"
) &
GPU2_PID=$!

# ---------------------------------------------------------------------------
# GPU 3 — 7B unfrozen
# ---------------------------------------------------------------------------
(
    cd "$BASE_DIR"
    MODEL="sft_qwen25vl_7b_15k_unfrozen"
    BASE="outputs/$MODEL"
    GPU=3
    BS=4

    echo "[GPU $GPU] $MODEL — wave 1 start"
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-453" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-453/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-906" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-906/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep2.log" 2>&1 &
    wait
    echo "[GPU $GPU] $MODEL — wave 1 done, starting wave 2"

    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-1359" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-1359/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep3.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU python $EVAL_SCRIPT \
        --checkpoint "$BASE/checkpoint-1812" \
        --ds-mvtec-only --batch-size $BS \
        --output "$BASE/checkpoint-1812/eval_dsmvtec_full_trainprompt.json" \
        > "$LOGS_DIR/${MODEL}_ep4.log" 2>&1 &
    wait
    echo "[GPU $GPU] $MODEL — ALL DONE"
) &
GPU3_PID=$!

# ---------------------------------------------------------------------------
# Wait for all 4 GPU streams
# ---------------------------------------------------------------------------
echo "All 4 GPU streams launched. Waiting..."
wait $GPU0_PID && echo "[GPU 0] 3B frozen OK" || echo "[GPU 0] 3B frozen FAILED"
wait $GPU1_PID && echo "[GPU 1] 3B unfrozen OK" || echo "[GPU 1] 3B unfrozen FAILED"
wait $GPU2_PID && echo "[GPU 2] 7B frozen OK" || echo "[GPU 2] 7B frozen FAILED"
wait $GPU3_PID && echo "[GPU 3] 7B unfrozen OK" || echo "[GPU 3] 7B unfrozen FAILED"

echo ""
echo "=== All evals complete — $(date) ==="
echo "Logs: $LOGS_DIR"
echo ""
echo "Result files:"
for MODEL in sft_qwen25vl_3b_15k_frozen sft_qwen25vl_3b_15k_unfrozen \
             sft_qwen25vl_7b_15k_frozen sft_qwen25vl_7b_15k_unfrozen; do
    for EP in 453 906 1359 1812; do
        F="$BASE_DIR/outputs/$MODEL/checkpoint-${EP}/eval_dsmvtec_full_trainprompt.json"
        if [ -f "$F" ]; then
            echo "  OK      $F"
        else
            echo "  MISSING $F"
        fi
    done
done
