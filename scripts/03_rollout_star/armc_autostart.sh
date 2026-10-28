#!/bin/bash
# Auto-launcher for the LLaVA Arm-C SFT + its epoch-eval watcher.
# Waits for the post-rollout pipeline to finish (pipeline.done), sanity-checks the
# Arm-C dataset, launches the frozen-tower SFT (identical recipe to all LLaVA arms)
# in tmux, derives the per-epoch checkpoint cadence from the first saved checkpoint,
# then arms the vLLM eval watcher on cuda 3 for all four epochs.
#
# KNOWN BUG, kept as run: the cadence below is four times the first save interval.
# The LLaVA run saved at 188 but really ended at step 748, not 752, so the watcher
# waited for a checkpoint that never existed. Read max_steps out of the first
# checkpoint's trainer_state.json instead, the way scripts/04_eval/watch_armd_evals.sh
# does. Left unfixed here so the script still matches the run that produced the
# committed Arm-C LLaVA results.
set -uo pipefail
# Paths. WORK_DIR is the workspace holding Training/ and outputs/; it defaults to the
# parent of this repository. On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
P0="${LLAVA_PHASE0_DIR:-$WORK_DIR/Training/phase0_llava_10k_20260901}"
C=$WORK_DIR/Training/datasets_sft_llava_iter1/sft_llava_C_train.json
OUTDIR=$WORK_DIR/outputs/sft_llava_ov_7b_frozen_llava_iter1_C
LOG=$WORK_DIR/outputs/armc_autostart.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "waiting for pipeline.done ..."
until [ -f "$P0/pipeline.done" ]; do sleep 120; done

log "pipeline done — sanity-checking Arm C dataset..."
python3 - "$C" <<'EOF' || { echo "SANITY FAIL — not launching"; exit 1; }
import json, os, random, sys
d = json.load(open(sys.argv[1]))
assert len(d) >= 4000, f"too small: {len(d)}"
r = d[0]
assert set(r.keys()) == {"messages", "images"} and [m["role"] for m in r["messages"]] == ["user", "assistant"]
assert r["messages"][0]["content"].startswith("<image>\n")
random.seed(0)
assert all(os.path.exists(x["images"][0]) for x in random.sample(d, 10))
ng = sum(1 for x in d if "/NG/" in x["images"][0] or "_NG_" in x["images"][0])
print(f"OK: {len(d)} items, NG={ng} OK={len(d)-ng}")
EOF

# GPUs 1,2 must be free (rollout shards ended; nothing else should hold them)
for G in 1 2; do
  M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)
  [ "$M" -lt 3000 ] || { log "GPU $G busy (${M}MiB) — waiting"; }
done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 3000 ] && \
      [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)" -lt 3000 ]; do sleep 60; done

log "launching Arm-C SFT (llava_iter1_C, GPUs 1,2)..."
tmux new-session -d -s sft_armC "bash $REPO_ROOT/scripts/01_sft/run_sft_llava_ov_7b_frozen.sh llava_iter1_C 1,2 12456 2>&1 | tee -a $WORK_DIR/outputs/sft_armC_launch.log"

log "waiting for first checkpoint to derive epoch cadence..."
FIRST=""
until [ -n "$FIRST" ]; do
  sleep 300
  FIRST=$(ls -d $OUTDIR/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | head -1)
  # bail out if the SFT session died before any checkpoint
  tmux has-session -t sft_armC 2>/dev/null || { [ -n "$FIRST" ] || { log "SFT session died pre-checkpoint — aborting watcher arm"; exit 1; }; }
done
S1=$FIRST; STEPS="$S1 $((2*S1)) $((3*S1)) $((4*S1))"
log "epoch cadence: $STEPS — arming eval watcher on cuda 3"
tmux new-session -d -s armC_evals "GRPO_WATCH_DIR=$OUTDIR GRPO_WATCH_STEPS='$STEPS' bash $REPO_ROOT/scripts/04_eval/watch_grpo_llava_vllm_evals.sh 3"
log "armed. done."
