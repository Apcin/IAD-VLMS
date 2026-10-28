#!/bin/bash
# Master orchestrator for the three-arm A/B/C SFT experiment.
#
# PHASE 1:  B on CUDA 0,3   +   C on CUDA 1,2   (parallel, until both done)
# PHASE 2:  A on CUDA 0,3   +   start evaluating B+C ckpts on CUDA 1,2
# PHASE 3:  When A SFT done, use ALL 4 GPUs (0,1,2,3) for remaining eval
#
# Run inside a tmux session.  Polls flag files; emits checkpoints to /tmp.

set -eo pipefail
WORK=/bulk/aacudad/reasoning_traces
TRAIN=$WORK/Training/run_abc_train.sh
EVAL=$WORK/Training/run_abc_eval_one.sh
chmod +x "$TRAIN" "$EVAL"

A_DIR=$WORK/outputs/sft_qwen25vl_7b_abc_A_kept
B_DIR=$WORK/outputs/sft_qwen25vl_7b_abc_B_kept_corrected
C_DIR=$WORK/outputs/sft_qwen25vl_7b_abc_C_full_patched
mkdir -p "$A_DIR" "$B_DIR" "$C_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG=$WORK/logs/abc_master_${STAMP}.log
mkdir -p $WORK/logs
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "════════════════════════════════════════════════════════════════════════"
echo " A/B/C three-arm SFT comparison — master orchestrator"
echo " Start: $(date)"
echo " B (4,369 items) on CUDA 0,3   |   C (6,000 items) on CUDA 1,2"
echo " then A (2,978 items) on CUDA 0,3 + eval on CUDA 1,2"
echo " then remaining eval on all 4 GPUs"
echo "════════════════════════════════════════════════════════════════════════"

# ---------------------------------------------------------------------
# PHASE 1: B + C in parallel
# ---------------------------------------------------------------------
echo ""
echo "[$(date)] PHASE 1 — kick off B (CUDA 0,3) and C (CUDA 1,2)"
"$TRAIN" B "0,3" > "$WORK/logs/abc_B_${STAMP}.log" 2>&1 &
B_PID=$!
"$TRAIN" C "1,2" > "$WORK/logs/abc_C_${STAMP}.log" 2>&1 &
C_PID=$!
echo "  B pid=$B_PID  C pid=$C_PID"

# Wait for BOTH to finish before starting A
wait $B_PID && echo "[$(date)] B SFT done."
wait $C_PID && echo "[$(date)] C SFT done."

# ---------------------------------------------------------------------
# PHASE 2: A on CUDA 0,3   +   start eval of B+C on CUDA 1,2
# ---------------------------------------------------------------------
echo ""
echo "[$(date)] PHASE 2 — A on CUDA 0,3, start eval of B+C on CUDA 1,2"
"$TRAIN" A "0,3" > "$WORK/logs/abc_A_${STAMP}.log" 2>&1 &
A_PID=$!

# Build the eval queue (B + C ckpts × 2 datasets)
EVAL_QUEUE=()
for arm_dir in "$B_DIR" "$C_DIR"; do
  for ckpt in "$arm_dir"/checkpoint-*; do
    [ -d "$ckpt" ] || continue
    EVAL_QUEUE+=("$ckpt|dsmvtec")
    EVAL_QUEUE+=("$ckpt|visa")
  done
done
echo "  initial eval queue length (B+C): ${#EVAL_QUEUE[@]} jobs"

# Worker function: pulls from queue, runs eval on a given GPU
declare -A GPU_BUSY=([1]=0 [2]=0)

run_worker() {
  local gpu=$1
  while [ ${#EVAL_QUEUE[@]} -gt 0 ]; do
    job=${EVAL_QUEUE[0]}
    EVAL_QUEUE=("${EVAL_QUEUE[@]:1}")
    IFS='|' read -r CKPT DS <<<"$job"
    echo "[$(date)] GPU $gpu picking $DS on $(basename $(dirname $CKPT))/$(basename $CKPT)"
    "$EVAL" "$CKPT" "$DS" "$gpu" || echo "  WARN eval failed: $job"
  done
}

# Run two workers (CUDA 1 and CUDA 2) in parallel during Phase 2
run_worker 1 &
W1_PID=$!
run_worker 2 &
W2_PID=$!

# Wait for A SFT to finish; meanwhile workers chew through B+C eval queue
wait $A_PID && echo "[$(date)] A SFT done."

# Add A ckpts to the queue (they're now available)
for ckpt in "$A_DIR"/checkpoint-*; do
  [ -d "$ckpt" ] || continue
  EVAL_QUEUE+=("$ckpt|dsmvtec")
  EVAL_QUEUE+=("$ckpt|visa")
done
echo "[$(date)] PHASE 3 — added A ckpts. queue length now: ${#EVAL_QUEUE[@]}"

# ---------------------------------------------------------------------
# PHASE 3: add GPU 0 and 3 to the worker pool
# ---------------------------------------------------------------------
run_worker 0 &
W0_PID=$!
run_worker 3 &
W3_PID=$!

# Wait for ALL workers to drain the queue
wait $W0_PID $W1_PID $W2_PID $W3_PID
echo ""
echo "[$(date)] ALL EVALS DONE."

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " Summary  —  $(date)"
echo "════════════════════════════════════════════════════════════════════════"
for arm_dir in "$A_DIR" "$B_DIR" "$C_DIR"; do
  echo ""
  echo "  $(basename $arm_dir):"
  for ckpt in "$arm_dir"/checkpoint-*; do
    name=$(basename $ckpt)
    ds=$ckpt/eval_dsmvtec_full_trainprompt.json
    vs=$ckpt/eval_visa_full_trainprompt.json
    DS=$(python3 -c "import json,sys;d=json.load(open('$ds'));m=d['metrics'];tp,tn,fp,fn=m['tp'],m['tn'],m['fp'],m['fn'];print(f'{(tp/(tp+fn)+tn/(tn+fp))/2*100:.2f}')" 2>/dev/null || echo "?")
    VS=$(python3 -c "import json,sys;d=json.load(open('$vs'));m=d['metrics'];tp,tn,fp,fn=m['tp'],m['tn'],m['fp'],m['fn'];print(f'{(tp/(tp+fn)+tn/(tn+fp))/2*100:.2f}')" 2>/dev/null || echo "?")
    echo "    $name   dsmvtec=$DS%   visa=$VS%"
  done
done
echo "════════════════════════════════════════════════════════════════════════"
