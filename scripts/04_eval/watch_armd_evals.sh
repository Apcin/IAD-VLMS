#!/bin/bash
# Rolling epoch-eval watcher for the Arm D SFT.
#
# Same harness that produced every Arm A/B/C row: scripts/04_eval/run_abc_eval_one.sh,
# which calls evaluate_qwen25vl_7b_trainprompt.py with --batch-size 16 in the
# llama_sft conda env. Nothing new is invented here, so the numbers drop straight
# into the same table.
#
# CADENCE. armc_autostart.sh took the first save interval and multiplied it by 4.
# The LLaVA run saved at 188 and its real final step was 748, not 752, so the
# watcher waited forever for a checkpoint that never existed. This script reads
# max_steps out of the first checkpoint's trainer_state.json instead, which has
# carried the right answer all along, and logs every step of the derivation.
#
# Usage: watch_armd_evals.sh [gpu]        (default gpu 3)
#   env: ARMD_OUTDIR   the SFT output dir to watch
#        ARMD_PID      pid of run_star_arm_d.sh. Used only as a kill -0 liveness
#                      probe, so the watcher can tell "not saved yet" from
#                      "the run died". Nothing is ever killed. Unset means the
#                      watcher keeps waiting instead of giving up.
set -uo pipefail

GPU="${1:-3}"
# Paths. WORK_DIR is the workspace holding outputs/; it defaults to the parent of this
# repository. On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
R="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
OUTDIR="${ARMD_OUTDIR:-$R/outputs/sft_qwen25vl_7b_abc_D_star_swap}"
RUN_PID="${ARMD_PID:-}"
EVAL_ONE=$HERE/run_abc_eval_one.sh
LOG="${ARMD_EVAL_LOG:-$R/outputs/armd_evals.log}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [evals] $*" | tee -a "$LOG"; }

if [ "$GPU" = "0" ]; then
  log "refusing to use GPU 0, it belongs to another user"; exit 1
fi

bal(){ python3 - "$1" <<'EOF'
import json,sys
try:
    m=json.load(open(sys.argv[1]))["metrics"]
    tp,tn,fp,fn=m["tp"],m["tn"],m["fp"],m["fn"]
    print(f"{50*(tp/((tp+fn) or 1)+tn/((tn+fp) or 1)):.2f}")
except Exception:
    print("nan")
EOF
}

# Weights complete: a single model.safetensors whose header parses, or every
# shard named in the index present and summing to the recorded total size.
weights_complete(){ python3 - "$1" <<'EOF'
import json,os,struct,sys
d=sys.argv[1]
idx=os.path.join(d,"model.safetensors.index.json")
single=os.path.join(d,"model.safetensors")
if os.path.exists(idx):
    j=json.load(open(idx)); shards=set(j["weight_map"].values())
    have=[os.path.join(d,s) for s in shards if os.path.exists(os.path.join(d,s))]
    if len(have)!=len(shards): sys.exit(1)
    tot=j.get("metadata",{}).get("total_size")
    sys.exit(0 if (tot is None or sum(os.path.getsize(p) for p in have)>=tot*0.99) else 1)
if os.path.exists(single) and os.path.getsize(single) > 1_000_000_000:
    try:
        with open(single,"rb") as f:
            n=struct.unpack("<Q",f.read(8))[0]; json.loads(f.read(n))
        sys.exit(0)
    except Exception:
        sys.exit(1)
sys.exit(1)
EOF
}

sft_alive(){ [ -z "$RUN_PID" ] && return 0; kill -0 "$RUN_PID" 2>/dev/null; }

# ---------------------------------------------------------------------------
# 1. Derive the epoch cadence from the real trainer state.
# ---------------------------------------------------------------------------
log "════ Arm D epoch evals on cuda$GPU ════"
log "watching $OUTDIR"
log "waiting for the first checkpoint so the cadence can be read from trainer_state.json ..."

FIRST=""
MISSES=0
until [ -n "$FIRST" ]; do
  FIRST=$(ls -d "$OUTDIR"/checkpoint-* 2>/dev/null \
          | sed 's/.*checkpoint-//' | sort -n | head -1)
  if [ -n "$FIRST" ]; then break; fi
  if ! sft_alive; then
    MISSES=$((MISSES+1))
    log "run_star_arm_d.sh (pid $RUN_PID) is gone and no checkpoint exists yet (strike $MISSES/3)"
    [ "$MISSES" -ge 3 ] && { log "SFT died before any checkpoint, aborting the watcher"; exit 1; }
  else
    MISSES=0
  fi
  sleep 120
done

STATE="$OUTDIR/checkpoint-$FIRST/trainer_state.json"
until [ -s "$STATE" ]; do log "waiting for $STATE ..."; sleep 60; done

read -r SAVE_EVERY MAX_STEPS N_EPOCHS <<<"$(python3 - "$STATE" <<'EOF'
import json,sys
s=json.load(open(sys.argv[1]))
print(int(s["global_step"]), int(s["max_steps"]), int(round(float(s["num_train_epochs"]))))
EOF
)"

log "trainer_state: global_step=$SAVE_EVERY  max_steps=$MAX_STEPS  num_train_epochs=$N_EPOCHS"

STEPS=""
i=1
while [ $((i * SAVE_EVERY)) -lt "$MAX_STEPS" ]; do
  STEPS="$STEPS $((i * SAVE_EVERY))"
  i=$((i + 1))
done
STEPS="$STEPS $MAX_STEPS"
STEPS=$(echo $STEPS)
N_STEPS=$(echo "$STEPS" | wc -w)

log "derived cadence: $STEPS"
log "  save interval $SAVE_EVERY from the first checkpoint, final step $MAX_STEPS from max_steps"
log "  NOT ${SAVE_EVERY} x ${N_EPOCHS} = $((SAVE_EVERY * N_EPOCHS)), which is the mistake armc_autostart.sh made"
if [ "$N_STEPS" != "$N_EPOCHS" ]; then
  log "  WARNING: $N_STEPS checkpoints derived but num_train_epochs is $N_EPOCHS. Using the derived list."
fi

# ---------------------------------------------------------------------------
# 2. Evaluate each checkpoint as it lands.
# ---------------------------------------------------------------------------
for STEP in $STEPS; do
  D="$OUTDIR/checkpoint-$STEP"
  DS="$D/eval_dsmvtec_full_trainprompt.json"
  VS="$D/eval_visa_full_trainprompt.json"
  if [ -s "$DS" ] && [ -s "$VS" ]; then
    log "checkpoint-$STEP already evaluated, skipping"
    continue
  fi

  log "waiting for checkpoint-$STEP ..."
  MISSES=0
  until [ -d "$D" ]; do
    if ! sft_alive; then
      MISSES=$((MISSES+1))
      log "run_star_arm_d.sh (pid $RUN_PID) is gone and checkpoint-$STEP never appeared (strike $MISSES/3)"
      [ "$MISSES" -ge 3 ] && { log "giving up on checkpoint-$STEP and everything after it"; exit 1; }
    else
      MISSES=0
    fi
    sleep 120
  done

  log "checkpoint-$STEP appeared, waiting for the weights to be written ..."
  until weights_complete "$D"; do sleep 60; done
  S1=$(du -sb "$D" | cut -f1); sleep 60; S2=$(du -sb "$D" | cut -f1)
  while [ "$S1" != "$S2" ]; do S1=$S2; sleep 60; S2=$(du -sb "$D" | cut -f1); done
  log "checkpoint-$STEP complete ($((S2 / 1000000000)) GB)"

  log "eval checkpoint-$STEP DS-MVTec on cuda$GPU ..."
  bash "$EVAL_ONE" "$D" dsmvtec "$GPU" >>"$LOG" 2>&1 || log "WARN DS-MVTec eval failed for checkpoint-$STEP"
  log "eval checkpoint-$STEP VisA on cuda$GPU ..."
  bash "$EVAL_ONE" "$D" visa "$GPU" >>"$LOG" 2>&1 || log "WARN VisA eval failed for checkpoint-$STEP"

  log "RESULT Arm D checkpoint-$STEP: DS-MVTec $(bal "$DS") | VisA $(bal "$VS") (balanced accuracy)"
done

log "════ ARM D EVALS DONE ════"
grep 'RESULT Arm D' "$LOG"
