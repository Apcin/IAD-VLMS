#!/bin/bash
# Eval watcher for one LLaVA-OV frozen SFT run: waits for training to complete (train.done),
# then evaluates every per-epoch checkpoint on FULL DS-MVTec + VisA with the canonical
# trainprompt protocol (same prompt/parser/samples as the Qwen thesis numbers).
# DS-MVTec runs on GPU_A, VisA on GPU_B (in parallel per checkpoint, sequential across ckpts).
# EVAL_MAX_IMAGE_PIXELS=262144 enforces the 512x512 protocol cap for the anyres processor.
#
# Usage: watch_llava_evals.sh <run_dir_name> <GPU_A> <GPU_B> [--with-baseline]
#   e.g. watch_llava_evals.sh sft_llava_ov_7b_frozen_iad_sft_6k_train 1 2 --with-baseline
set -uo pipefail
RUN="${1:?run dir name under outputs/}"
GA="${2:?gpu for DS-MVTec}"; GB="${3:?gpu for VisA}"
BASELINE="${4:-}"

# Paths. WORK_DIR is the workspace holding outputs/, hf_cache/ and the python envs;
# it defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
PY="${PYTHON_BIN:-python3}"
EVAL=$HERE/evaluate_qwen25vl_7b_trainprompt.py
OUT=$WORK_DIR/outputs/$RUN
BASE=llava-hf/llava-onevision-qwen2-7b-si-hf
export HF_HOME=$WORK_DIR/hf_cache
export EVAL_MAX_IMAGE_PIXELS=262144
LOG=$OUT/eval_watcher.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

bal(){ $PY - "$1" <<'EOF'
import json,sys
try:
    m=json.load(open(sys.argv[1]))['metrics']; tp,tn,fp,fn=m['tp'],m['tn'],m['fp'],m['fn']
    print(f"{50*(tp/((tp+fn) or 1)+tn/((tn+fp) or 1)):.2f}")
except Exception: print("nan")
EOF
}

mkdir -p "$OUT"
log "=== watcher for $RUN (DS on cuda$GA, VisA on cuda$GB) — waiting for train.done ==="
until [ -f "$OUT/train.done" ]; do
  if ! pgrep -f "frozen_${RUN##*frozen_}" >/dev/null 2>&1 && [ ! -f "$OUT/train.done" ]; then
    sleep 120  # grace: process gone could be transient scan timing; re-check twice
    [ -f "$OUT/train.done" ] && break
    if ! pgrep -f "frozen_${RUN##*frozen_}" >/dev/null 2>&1; then
      log "!! training process gone without train.done — watcher keeps waiting (restart training or kill me)"
      sleep 480
    fi
  fi
  sleep 120
done
log "training done. starting evals."

eval_ckpt(){
  local CKD="$1" TAG="$2"
  log "eval $TAG: DS-MVTec + VisA (full sets)..."
  CUDA_VISIBLE_DEVICES=$GA $PY "$EVAL" --checkpoint "$CKD" --base-model "$BASE" \
    --ds-mvtec-only --batch-size 4 --output "$CKD/eval_dsmvtec_full_trainprompt.json" \
    >"$CKD/eval_ds.log" 2>&1 &
  local p1=$!
  CUDA_VISIBLE_DEVICES=$GB $PY "$EVAL" --checkpoint "$CKD" --base-model "$BASE" \
    --visa-only --batch-size 4 --output "$CKD/eval_visa_full_trainprompt.json" \
    >"$CKD/eval_va.log" 2>&1 &
  local p2=$!
  wait $p1 $p2
  local ds va
  ds=$(bal "$CKD/eval_dsmvtec_full_trainprompt.json"); va=$(bal "$CKD/eval_visa_full_trainprompt.json")
  log "RESULT $TAG: DS-MVTec $ds | VisA $va (balanced acc)"
}

for ck in 188 376 564 748 752; do
  CKD="$OUT/checkpoint-$ck"
  [ -d "$CKD" ] || { log "(no checkpoint-$ck, skip)"; continue; }
  eval_ckpt "$CKD" "$RUN/ckpt-$ck (epoch $((ck/188)))"
done

if [ "$BASELINE" = "--with-baseline" ]; then
  BOUT=$WORK_DIR/outputs/llava_ov_7b_zeroshot_eval; mkdir -p "$BOUT"
  log "baseline zero-shot eval (base model, full sets)..."
  CUDA_VISIBLE_DEVICES=$GA $PY "$EVAL" --base-model "$BASE" \
    --ds-mvtec-only --batch-size 4 --output "$BOUT/eval_dsmvtec_full_trainprompt.json" \
    >"$BOUT/eval_ds.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=$GB $PY "$EVAL" --base-model "$BASE" \
    --visa-only --batch-size 4 --output "$BOUT/eval_visa_full_trainprompt.json" \
    >"$BOUT/eval_va.log" 2>&1 &
  p2=$!
  wait $p1 $p2
  log "RESULT base zero-shot: DS-MVTec $(bal "$BOUT/eval_dsmvtec_full_trainprompt.json") | VisA $(bal "$BOUT/eval_visa_full_trainprompt.json")"
fi

log "=== ALL EVALS DONE for $RUN ==="
grep 'RESULT' "$LOG"
touch "$OUT/evals.done"
