#!/bin/bash
# Rolling epoch-eval watcher for the Qwen3-VL-8B SFT on the LLaVA-derived KCR Arm-C corpus.
#
# It is watch_and_eval_qwen3vl_cuda3.sh with three changes:
#   1. the default output dir is the llavaC run
#   2. it derives the expected final step from trainer_state.json instead of
#      multiplying the first save interval by four (the armc_autostart.sh bug)
#   3. it writes a log file and says out loud what it decided
# The eval invocation itself is byte-identical to the armC watcher, so the numbers
# land in the same column as the Qwen3-VL-8B armC row (82.62/77.67 at epoch 1).
#
# As each epoch checkpoint appears it evals it on cuda 3, DS-MVTec THEN VisA,
# SEQUENTIALLY, never two at once. Prompt = the default trainprompt protocol
# (system "Please answer by yes or no" + Analyze), the same one behind every
# other number in this project. Waits for cuda 3 to fall below 10 GB before each
# eval, so it coexists with any other eval already on that card.
# Skip-if-done, so it is safe to re-run. Exits when train.done exists AND every
# checkpoint has both JSONs.
#
# Usage: watch_and_eval_qwen3vl_llavaC_cuda3.sh [OUT_DIR] [GPU]
# Env:   QLC_EVAL_LOG   log file (default outputs/qwen3_llavaC_evals.log)
#        QLC_BATCH      pass --batch-size N. LEAVE UNSET. Unset reproduces the
#                       armC watcher exactly. Setting it changes the invocation.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
# Paths. WORK_DIR is the workspace holding LlamaFactory/, outputs/ and hf_cache/; it
# defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
export HF_HOME=$WORK_DIR/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME

OUT="${1:-$WORK_DIR/outputs/sft_qwen3vl_8b_llavaC}"
GPU="${2:-3}"
EVAL=$HERE/evaluate_qwen25vl_7b_trainprompt.py
BASE=Qwen/Qwen3-VL-8B-Instruct
BA=$REPO_ROOT/results/compute_ba.py
LOG="${QLC_EVAL_LOG:-$WORK_DIR/outputs/qwen3_llavaC_evals.log}"
BATCH_ARG=""
[ -n "${QLC_BATCH:-}" ] && BATCH_ARG="--batch-size ${QLC_BATCH}"

mkdir -p "$OUT" "$(dirname "$LOG")"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [q3-llavaC-evals] $*" | tee -a "$LOG"; }

if [ "$GPU" = "0" ]; then
  log "refusing to use GPU 0, it belongs to another user"; exit 1
fi
for f in "$EVAL" "$BA"; do
  [ -f "$f" ] || { log "FATAL: missing $f"; exit 1; }
done

log "════════════════════════════════════════════════════════════════════"
log "armed for $OUT"
log "DS-MVTec then VisA, sequential, cuda $GPU, base/processor = $BASE"
log "eval flags: ${BATCH_ARG:-none (script default, same as the armC watcher)}"
log "════════════════════════════════════════════════════════════════════"

gpu_mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
wait_gpu(){
  local m
  while :; do
    m=$(gpu_mem "$GPU")
    [ -n "$m" ] || { log "cuda$GPU not readable, waiting"; sleep 60; continue; }
    [ "$m" -le 10000 ] && return 0
    log "cuda$GPU busy (${m} MiB), waiting"
    sleep 60
  done
}

# ---------------------------------------------------------------------------
# Cadence. Read max_steps out of the first checkpoint's trainer_state.json.
# Never multiply the first save interval by four. armc_autostart.sh did that,
# the first save was at 188, it armed on 752, the run really ended at 748 and
# the watcher waited for ever. The number was in trainer_state.json all along.
# This value is used for logging and for the final tally only. The main loop
# evaluates whatever checkpoint directory actually appears, so a wrong guess
# here cannot make it miss or stall on anything.
# ---------------------------------------------------------------------------
FINAL=""
report_cadence(){
  [ -n "$FINAL" ] && return 0
  local first
  first=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | head -1)
  [ -n "$first" ] || return 1
  [ -s "$OUT/checkpoint-$first/trainer_state.json" ] || return 1
  FINAL=$(python3 - "$OUT/checkpoint-$first/trainer_state.json" <<'EOF'
import json,sys
print(int(json.load(open(sys.argv[1]))["max_steps"]))
EOF
)
  [ -n "$FINAL" ] || return 1
  log "cadence: first checkpoint is checkpoint-$first, so the save interval is $first"
  log "cadence: max_steps = $FINAL, read from checkpoint-$first/trainer_state.json"
  if [ "$FINAL" -ne $((4 * first)) ]; then
    log "cadence: four times the interval would be $((4 * first)). That is NOT the final step."
    log "cadence: this is exactly the case armc_autostart.sh got wrong. Using $FINAL."
  else
    log "cadence: four times the interval also gives $FINAL here, so both agree. Still using max_steps."
  fi
  return 0
}

eval_one(){
  local ck="$1"
  local dsj="$ck/eval_dsmvtec_full_trainprompt.json" vsj="$ck/eval_visa_full_trainprompt.json"
  if [ ! -f "$dsj" ]; then
    wait_gpu; log "EVAL $(basename "$ck") DS-MVTec on cuda$GPU"
    CUDA_VISIBLE_DEVICES=$GPU python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --ds-mvtec-only \
        $BATCH_ARG --output "$dsj" > "$ck/eval_ds.log" 2>&1
    [ -f "$dsj" ] || log "WARNING: DS-MVTec produced no JSON for $(basename "$ck"), see $ck/eval_ds.log"
  fi
  if [ ! -f "$vsj" ]; then
    wait_gpu; log "EVAL $(basename "$ck") VisA on cuda$GPU"
    CUDA_VISIBLE_DEVICES=$GPU python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --visa-only \
        $BATCH_ARG --output "$vsj" > "$ck/eval_visa.log" 2>&1
    [ -f "$vsj" ] || log "WARNING: VisA produced no JSON for $(basename "$ck"), see $ck/eval_visa.log"
  fi
  local d v
  d=$(python3 "$BA" "$dsj" 2>/dev/null | grep -oE "BA=[0-9.]+")
  v=$(python3 "$BA" "$vsj" 2>/dev/null | grep -oE "BA=[0-9.]+")
  log "$(basename "$ck"): DS $d  VisA $v"
}

TICK=0
while :; do
  report_cadence >/dev/null 2>&1
  for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    [ -f "$ck/config.json" ] || continue
    { [ -f "$ck/eval_dsmvtec_full_trainprompt.json" ] && [ -f "$ck/eval_visa_full_trainprompt.json" ]; } && continue
    eval_one "$ck"
  done
  if [ -f "$OUT/train.done" ]; then
    pending=0
    for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null); do
      [ -f "$ck/config.json" ] || continue
      { [ -f "$ck/eval_dsmvtec_full_trainprompt.json" ] && [ -f "$ck/eval_visa_full_trainprompt.json" ]; } || pending=$((pending+1))
    done
    if [ "$pending" -eq 0 ]; then
      log "train.done is present and nothing is pending. Watcher finished."
      [ -n "$FINAL" ] && { [ -d "$OUT/checkpoint-$FINAL" ] \
        && log "final checkpoint-$FINAL is present, all four epochs are covered" \
        || log "NOTE: checkpoint-$FINAL is not on disk. The run stopped short of max_steps."; }
      break
    fi
  fi
  if [ $((TICK % 15)) -eq 0 ]; then
    n=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | wc -l)
    log "waiting. $n checkpoint dir(s) so far, train.done $([ -f "$OUT/train.done" ] && echo present || echo absent), cuda$GPU $(gpu_mem "$GPU") MiB"
  fi
  TICK=$((TICK+1))
  sleep 120
done

log "=== Qwen3-VL-8B on the LLaVA Arm-C corpus, BA trajectory ==="
for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  d=$(python3 "$BA" "$ck/eval_dsmvtec_full_trainprompt.json" 2>/dev/null | grep -oE "BA=[0-9.]+")
  v=$(python3 "$BA" "$ck/eval_visa_full_trainprompt.json" 2>/dev/null | grep -oE "BA=[0-9.]+")
  log "  $(basename "$ck"): DS $d  VisA $v"
done
log "compare against the Qwen-corpus run: $R/outputs/sft_qwen3vl_8b_armC"
log "  checkpoint-188 DS 82.62 VisA 77.67 | 376 85.82/76.52 | 564 85.18/76.62 | 752 82.59/73.26"
