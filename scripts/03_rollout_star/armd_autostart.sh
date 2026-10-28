#!/bin/bash
# ===========================================================================
# Auto-launcher for the Arm D (STaR) run.
#
# Waits until the machine is genuinely free, then hands over to
# Training/run_star_arm_d.sh. It waits for two things:
#
#   1. the Arm-C LLaVA SFT on GPUs 1 and 2 has exited
#   2. the ckpt-748 vLLM eval on GPU 3 has finished
#
# It uses no GPU itself, so it is safe to arm at any time.
#
# WHAT WENT WRONG LAST TIME. armc_autostart.sh derived the checkpoint cadence
# as four times the first save interval. The first save was at step 188, so it
# armed the eval watcher on "188 376 564 752". The real final step was 748 and
# the watcher waited for a checkpoint that was never going to exist. The value
# was sitting in checkpoint-188/trainer_state.json as max_steps the whole time.
# This script reads max_steps, never multiplies, and logs every decision it
# makes so the next person can check it without re-deriving anything.
#
# Nothing here kills anything. Every wait is a poll.
#
# Arm it with:
#   tmux new-session -d -s armd_autostart \
#     "bash scripts/03_rollout_star/armd_autostart.sh"
#
# Environment knobs:
#   ARMD_DRY_RUN=1        do all the waiting and logging, do not launch
#   ARMD_WAIT_FOR_EVAL=0  do not wait for the ckpt-748 eval. Only set this if
#                         you have decided to let the rationalisation pass share
#                         the box with that eval. It runs on GPU 1, the eval on
#                         GPU 3, so it is safe, it just was not what was asked.
#   plus everything run_star_arm_d.sh reads (ARMD_VARIANT, ARMD_RULE, ...)
# ===========================================================================
set -uo pipefail

# Paths. WORK_DIR is the workspace holding Training/ and outputs/; it defaults to the
# parent of this repository. On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
R="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
T="${T:-$R/Training}"
LLAVA_OUT=$R/outputs/sft_llava_ov_7b_frozen_llava_iter1_C
LOG="${ARMD_LOG:-$R/outputs/armd_autostart.log}"
RUNNER=$HERE/run_star_arm_d.sh

SFT_GPUS="${ARMD_SFT_GPUS:-1,2}"
EVAL_GPU="${ARMD_EVAL_GPU:-3}"
WAIT_FOR_EVAL="${ARMD_WAIT_FOR_EVAL:-1}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [autostart] $*" | tee -a "$LOG"; }
gpu_mib(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null; }
free_gpu(){ local m; m=$(gpu_mib "$1"); [ -n "$m" ] && [ "$m" -lt 3000 ]; }

log "════════════════════════════════════════════════════════════════════"
log "Arm D autostart armed. It uses no GPU while it waits."
log "gate 1: the Arm-C LLaVA SFT on GPUs $SFT_GPUS has exited"
log "gate 2: the final-checkpoint vLLM eval on GPU $EVAL_GPU has finished"
log "then:   bash $RUNNER"
log "GPU 0 is another user's and is never touched."
log "════════════════════════════════════════════════════════════════════"

[ -x "$RUNNER" ] || [ -f "$RUNNER" ] || { log "FATAL: $RUNNER not found"; exit 1; }

# ---------------------------------------------------------------------------
# Step 0. Derive the LLaVA run's real final step. No multiplication.
# ---------------------------------------------------------------------------
log "step 0: deriving the Arm-C LLaVA final step from the trainer state"

FINAL=""
SRC=""
FIRST_CKPT=$(ls -d "$LLAVA_OUT"/checkpoint-* 2>/dev/null \
             | sed 's/.*checkpoint-//' | sort -n | head -1)
if [ -n "$FIRST_CKPT" ] && [ -s "$LLAVA_OUT/checkpoint-$FIRST_CKPT/trainer_state.json" ]; then
  FINAL=$(python3 - "$LLAVA_OUT/checkpoint-$FIRST_CKPT/trainer_state.json" <<'EOF'
import json,sys
print(int(json.load(open(sys.argv[1]))["max_steps"]))
EOF
)
  SRC="checkpoint-$FIRST_CKPT/trainer_state.json:max_steps"
  log "  first checkpoint is checkpoint-$FIRST_CKPT, so the save interval is $FIRST_CKPT"
  log "  four times that would be $((4 * FIRST_CKPT)). That is the number armc_autostart.sh used and it was wrong."
elif [ -s "$LLAVA_OUT/trainer_log.jsonl" ]; then
  FINAL=$(python3 - "$LLAVA_OUT/trainer_log.jsonl" <<'EOF'
import json,sys
last=[json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]
print(int(last["total_steps"]))
EOF
)
  SRC="trainer_log.jsonl:total_steps"
fi

if [ -z "$FINAL" ]; then
  log "  WARNING: no trainer state found under $LLAVA_OUT. Falling back to the GPU-idle gates only."
else
  log "  final step = $FINAL, read from $SRC"
fi

# ---------------------------------------------------------------------------
# Step 1. Wait for the Arm-C LLaVA SFT to exit.
# ---------------------------------------------------------------------------
log "step 1: waiting for the Arm-C LLaVA SFT to exit"

TICK=0
while true; do
  M1=$(gpu_mib 1); M2=$(gpu_mib 2)
  if free_gpu 1 && free_gpu 2; then
    sleep 60
    if free_gpu 1 && free_gpu 2; then
      log "  GPUs 1 and 2 are free (${M1} MiB / ${M2} MiB), twice in a row"
      break
    fi
  fi
  if [ $((TICK % 10)) -eq 0 ]; then
    PROG=$(python3 - "$LLAVA_OUT/trainer_log.jsonl" <<'EOF' 2>/dev/null
import json, sys
try:
    lines = [l for l in open(sys.argv[1]) if l.strip()]
    d = json.loads(lines[-1])
    print(f"step {d.get('current_steps')}/{d.get('total_steps')}, "
          f"{d.get('percentage')}%, remaining {d.get('remaining_time')}")
except Exception:
    print("no trainer_log line")
EOF
)
    log "  still busy: GPU1 ${M1} MiB, GPU2 ${M2} MiB. LLaVA $PROG"
  fi
  TICK=$((TICK+1))
  sleep 120
done

if tmux has-session -t sft_armC 2>/dev/null; then
  log "  note: the tmux session sft_armC still exists but the GPUs are idle. Continuing."
else
  log "  tmux session sft_armC is gone"
fi

if [ -n "$FINAL" ]; then
  if [ -d "$LLAVA_OUT/checkpoint-$FINAL" ]; then
    log "  final checkpoint-$FINAL exists, the Arm-C SFT completed"
  else
    log "  WARNING: checkpoint-$FINAL does not exist. The Arm-C SFT did not reach its last save."
    log "  Arm D does not depend on it, so this is a note, not a blocker."
  fi
fi

# ---------------------------------------------------------------------------
# Step 2. Wait for the final-checkpoint vLLM eval on the eval GPU.
# ---------------------------------------------------------------------------
if [ "$WAIT_FOR_EVAL" != "1" ]; then
  log "step 2: skipped, ARMD_WAIT_FOR_EVAL=0"
elif [ -z "$FINAL" ]; then
  log "step 2: no final step known, waiting only for GPU $EVAL_GPU to go idle"
  until free_gpu "$EVAL_GPU"; do
    log "  GPU $EVAL_GPU holds $(gpu_mib "$EVAL_GPU") MiB, waiting"
    sleep 120
  done
  log "  GPU $EVAL_GPU is idle"
else
  DSJ=$LLAVA_OUT/checkpoint-$FINAL/eval_dsmvtec_full_trainprompt_vllm.json
  VSJ=$LLAVA_OUT/checkpoint-$FINAL/eval_visa_full_trainprompt_vllm.json
  log "step 2: waiting for the checkpoint-$FINAL vLLM eval on GPU $EVAL_GPU"
  log "  looking for $DSJ"
  log "  and         $VSJ"

  IDLE_STREAK=0
  TICK=0
  while true; do
    if [ -s "$DSJ" ] && [ -s "$VSJ" ] && free_gpu "$EVAL_GPU"; then
      log "  both eval JSONs are written and GPU $EVAL_GPU is idle ($(gpu_mib "$EVAL_GPU") MiB)"
      break
    fi
    # Escape hatch: the eval never started and never will. Do not stall for ever.
    if free_gpu "$EVAL_GPU" && ! tmux has-session -t armC_evals 2>/dev/null; then
      IDLE_STREAK=$((IDLE_STREAK+1))
      log "  GPU $EVAL_GPU idle and the armC_evals session is gone (strike $IDLE_STREAK/15)"
      if [ "$IDLE_STREAK" -ge 15 ]; then
        log "  the checkpoint-$FINAL eval is not going to run. Continuing without it."
        break
      fi
    else
      IDLE_STREAK=0
      if [ $((TICK % 10)) -eq 0 ]; then
        log "  GPU $EVAL_GPU holds $(gpu_mib "$EVAL_GPU") MiB, eval still running"
      fi
    fi
    TICK=$((TICK+1))
    sleep 120
  done
fi

# ---------------------------------------------------------------------------
# Step 3. Preflight that does not belong in the runner's critical path.
# ---------------------------------------------------------------------------
log "step 3: preflight"
JURL="${GEMINI_JUDGE_URL:-http://127.0.0.1:5300}"
if curl -s -m 5 -o /dev/null "$JURL"; then
  log "  embedding judge server answering at $JURL"
else
  log "  WARNING: no answer from $JURL. run_star_arm_d.sh will refuse to start."
  log "  Start it first (tmux rank_judge), otherwise the type reward silently"
  log "  falls back to lexical similarity and the keep rule is not the production one."
fi
for g in ${SFT_GPUS//,/ } "$EVAL_GPU"; do
  log "  GPU $g: $(gpu_mib "$g") MiB used"
done
log "  GPU 0: $(gpu_mib 0) MiB used, not ours, not touched"

# ---------------------------------------------------------------------------
# Step 4. Hand over.
# ---------------------------------------------------------------------------
if [ "${ARMD_DRY_RUN:-0}" = "1" ]; then
  log "step 4: ARMD_DRY_RUN=1, stopping here. Would have run: bash $RUNNER"
  exit 0
fi

if tmux has-session -t armD 2>/dev/null; then
  log "step 4: a tmux session named armD already exists. Not launching a second run."
  exit 1
fi

log "step 4: launching the Arm D run in tmux session armD"
tmux new-session -d -s armD "bash $RUNNER 2>&1 | tee -a $R/outputs/armd_launch.log"
sleep 3
if tmux has-session -t armD 2>/dev/null; then
  log "  armD is up. Follow it with: tail -f $R/outputs/armd_run.log"
  log "  Evals land in: tail -f $R/outputs/armd_evals.log"
else
  log "  FATAL: the armD session did not start"
  exit 1
fi
log "armed and handed over. done."
