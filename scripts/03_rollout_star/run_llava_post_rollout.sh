#!/bin/bash
# LLaVA native-KCR post-rollout chain: waits for the 3 rollout shards, then
# merge -> bucket -> phase1a Gemini judge -> combine -> phase1b correct ->
# phase1b rewrite -> build Arm A/B/C. Mirrors run_pipeline_10k.sh stages B-F.
set -e
# Paths. WORK_DIR is the workspace holding Training/, outputs/ and hf_cache/; it defaults
# to the parent of this repository. HERE is this script's own directory, which is where
# phase0_bucket.py / phase1a_gemini_judge.py / phase1b_gemini_correct.py live.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
OUT="${LLAVA_PHASE0_DIR:-$WORK_DIR/Training/phase0_llava_10k_20260901}"
PY="${PYTHON_BIN:-python3}"
export HF_HOME=$WORK_DIR/hf_cache
# Vertex service-account JSON. Never stored in this repository; set it in your environment
# or in the git-ignored .env (see .env.example).
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:?set GOOGLE_APPLICATION_CREDENTIALS to your Vertex service-account JSON}"
export GEMINI_JUDGE_URL="${GEMINI_JUDGE_URL:-http://127.0.0.1:5300}"
cd "$HERE"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for 3 rollout shards to finish..."
while true; do
  DONE=0
  for S in 0 1 2; do grep -q "DONE in" $OUT/rollout_s$S.log 2>/dev/null && DONE=$((DONE+1)); done
  [ "$DONE" -eq 3 ] && break
  sleep 300
done
log "all shards done. merging..."
cat $OUT/rollouts_raw_shard0.jsonl $OUT/rollouts_raw_shard1.jsonl $OUT/rollouts_raw_shard2.jsonl > $OUT/rollouts_raw.jsonl
log "merged: $(wc -l < $OUT/rollouts_raw.jsonl) records"

log "Stage B: bucketing..."
$PY $HERE/phase0_bucket.py \
    --input_jsonl "$OUT/rollouts_raw.jsonl" --out_dir "$OUT" \
    --threshold_yes 1.5 --threshold_no 1.0 --weak_threshold_yes 1.8 \
    2>&1 | tee "$OUT/bucket.log"

log "Stage C: phase1a Gemini judge (concurrency 12)..."
$PY $HERE/phase1a_gemini_judge.py \
    --input "$OUT/judge_input.json" \
    --output_faithful "$OUT/good_traces_faithful.json" \
    --output_rewrite "$OUT/needs_rewrite_from_judge.json" \
    --report_jsonl "$OUT/judge_report.jsonl" \
    --keep_threshold 2 --max_concurrent 12 \
    2>&1 | tee "$OUT/phase1a.log"

log "Stage D: combining rewrite lists..."
$PY - <<PYEOF
import json
local = json.load(open('$OUT/needs_rewrite.json'))
judge = json.load(open('$OUT/needs_rewrite_from_judge.json'))
seen, combined = set(), []
for src in [local, judge]:
    for it in src:
        if it['image_path'] not in seen:
            seen.add(it['image_path']); combined.append(it)
json.dump(combined, open('$OUT/needs_rewrite_combined.json','w'), indent=2)
print(f'local {len(local)} + judge {len(judge)} -> combined {len(combined)}')
PYEOF

log "Stage E: phase1b CORRECT..."
$PY $HERE/phase1b_gemini_correct.py \
    --input "$OUT/needs_correction.json" --output "$OUT/gemini_corrected.jsonl" \
    --mode correct --max_concurrent 12 2>&1 | tee "$OUT/phase1b_correct.log"

log "Stage F: phase1b REWRITE..."
$PY $HERE/phase1b_gemini_correct.py \
    --input "$OUT/needs_rewrite_combined.json" --output "$OUT/gemini_rewritten.jsonl" \
    --mode rewrite --max_concurrent 12 2>&1 | tee "$OUT/phase1b_rewrite.log"

log "Stage G: building Arm A/B/C..."
$PY $HERE/build_llava_arms.py "$OUT" "$WORK_DIR/Training/datasets_sft_llava_iter1" 2>&1 | tee "$OUT/build_arms.log"

log "ALL DONE — arms in Training/datasets_sft_llava_iter1/"
touch "$OUT/pipeline.done"
