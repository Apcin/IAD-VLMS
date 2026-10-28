#!/bin/bash
# Paths below assume the TU Delft workspace /bulk/aacudad/reasoning_traces (the parent of this repo). Replace with your WORK_DIR.
# LLaVA Arm-C SFT on the CORRECTED corpus (sft_llava_C_original_train.json), plus its
# per-epoch vLLM eval watcher. Same recipe as the original Arm-C run in every respect
# except the dataset: this corpus is a strict subset of the 6,000-image SFT split and
# is balanced on the verdict the trace teaches, not on the folder name.
set -uo pipefail
R=/bulk/aacudad/reasoning_traces
DS=iad_sft_llava_iter1_C_original
C=$R/Training/datasets_sft_llava_iter1/sft_llava_C_original_train.json
SPLIT=$R/Training/datasets_small_new_v4/combined_6k_train.json
OUTDIR=$R/outputs/sft_llava_ov_7b_frozen_${DS}
LOG=$R/outputs/armc_original_autostart.log
STEPS="188 376 564 748"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "sanity-checking the corrected Arm-C corpus..."
python3 - "$C" "$SPLIT" <<'PYEOF' || { log "SANITY FAIL - not launching"; exit 1; }
import json, os, random, sys, re
d = json.load(open(sys.argv[1])); split = json.load(open(sys.argv[2]))
assert len(d) == 6000, f"expected 6000, got {len(d)}"
r = d[0]
assert set(r.keys()) == {"messages", "images"} and [m["role"] for m in r["messages"]] == ["user", "assistant"]
assert r["messages"][0]["content"].startswith("<image>\n")
random.seed(0)
assert all(os.path.exists(x["images"][0]) for x in random.sample(d, 20)), "missing image files"
key = lambda p: p[p.find("Real-IAD/"):] if "Real-IAD/" in p else p
k6 = {key(x["images"][0]) for x in split}
outside = sum(1 for x in d if key(x["images"][0]) not in k6)
assert outside == 0, f"{outside} images outside the 6K SFT split"
# balance on the verdict the trace teaches, NOT on the folder name (that was the old bug)
A = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
v = [(A.search(x["messages"][1]["content"]) or [None, ""])[1].strip().lower() for x in d]
yes, no = v.count("yes"), v.count("no")
assert yes + no == 6000, f"unparseable verdicts: {6000 - yes - no}"
assert abs(yes - no) <= 2, f"not balanced: {yes} yes / {no} no"
print(f"OK: 6000 items, {yes} yes / {no} no by <answer>, 0 outside the 6K split")
PYEOF

until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 3000 ] && \
      [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)" -lt 3000 ]; do
  log "GPU 1/2 busy, waiting"; sleep 60; done

log "launching SFT on GPUs 1,2 -> $OUTDIR"
tmux new-session -d -s sft_armC_orig \
  "bash $R/Training/run_sft_llava_ov_7b_frozen.sh $DS 1,2 12457 2>&1 | tee -a $R/outputs/sft_armC_original_launch.log"

log "waiting for the first checkpoint before arming the watcher..."
until [ -d "$OUTDIR/checkpoint-188" ]; do
  sleep 300
  tmux has-session -t sft_armC_orig 2>/dev/null || { [ -d "$OUTDIR/checkpoint-188" ] || { log "SFT died pre-checkpoint - aborting"; exit 1; }; }
done
log "checkpoint-188 present, arming eval watcher on cuda 3 for steps: $STEPS"
tmux new-session -d -s armC_orig_evals \
  "GRPO_WATCH_DIR=$OUTDIR GRPO_WATCH_STEPS='$STEPS' bash $R/Training/watch_grpo_llava_vllm_evals.sh 3"
log "armed."
