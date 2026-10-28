#!/bin/bash
# Resume the crashed 10k pipeline run. Same OUT_DIR as before, so the rollout
# script picks up only the 4347 remaining items (skips the 5889 already done).
# CUDA 0 (most free memory).

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache

# REUSE the existing output dir so resume kicks in
OUT_DIR=/bulk/aacudad/reasoning_traces/Training/phase0_full_10k_20260529_015821

MODEL=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530

echo "════════════════════════════════════════════════════════════════════════"
echo " RESUME 10k pipeline — Phase 0 (4347 items remain) → 1a → 1b"
echo " Output dir: $OUT_DIR"
echo " Resume:     5889 already done"
echo " GPU:        CUDA 0"
echo " Start:      $(date)"
echo "════════════════════════════════════════════════════════════════════════"

# Stage A — Phase 0 rollout, RESUMES (skips items already in rollouts_raw.jsonl)
echo ""
echo "[A — $(date +%T)] Phase 0 rollout (resume) on CUDA 0 …"

CUDA_VISIBLE_DEVICES=0 \
python /bulk/aacudad/reasoning_traces/Training/phase0_rollout_kscoring.py \
    --model_path "$MODEL" \
    --output_jsonl "$OUT_DIR/rollouts_raw.jsonl" \
    --k 8 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --max_pixels 480000 \
    --seed 42 \
    2>&1 | tee -a "$OUT_DIR/rollout.log"

echo "[A — $(date +%T)] Done. $(wc -l < $OUT_DIR/rollouts_raw.jsonl) items total."

# Stage B — bucket
echo ""
echo "[B — $(date +%T)] Bucketing …"
python /bulk/aacudad/reasoning_traces/Training/phase0_bucket.py \
    --input_jsonl "$OUT_DIR/rollouts_raw.jsonl" \
    --out_dir "$OUT_DIR" \
    --threshold_yes 1.5 --threshold_no 1.0 --weak_threshold_yes 1.8 \
    2>&1 | tee "$OUT_DIR/bucket.log"

# Stage C — Phase 1a Gemini judge
echo ""
echo "[C — $(date +%T)] Phase 1a Gemini judge (max_concurrent=12) …"
python /bulk/aacudad/reasoning_traces/Training/phase1a_gemini_judge.py \
    --input           "$OUT_DIR/judge_input.json" \
    --output_faithful "$OUT_DIR/good_traces_faithful.json" \
    --output_rewrite  "$OUT_DIR/needs_rewrite_from_judge.json" \
    --report_jsonl    "$OUT_DIR/judge_report.jsonl" \
    --keep_threshold 2 --max_concurrent 12 \
    2>&1 | tee "$OUT_DIR/phase1a.log"

# Stage D — combine
echo ""
echo "[D — $(date +%T)] Building needs_rewrite_combined.json …"
python <<PYEOF
import json
with open('$OUT_DIR/needs_rewrite.json') as f: local = json.load(f)
with open('$OUT_DIR/needs_rewrite_from_judge.json') as f: judge = json.load(f)
seen = set(); combined = []
for src in [local, judge]:
    for it in src:
        if it['image_path'] not in seen:
            seen.add(it['image_path']); combined.append(it)
with open('$OUT_DIR/needs_rewrite_combined.json', 'w') as f:
    json.dump(combined, f, indent=2)
print(f'  local: {len(local)}, judge: {len(judge)}, combined: {len(combined)}')
PYEOF

# Stage E — 1b correct
echo ""
echo "[E — $(date +%T)] Phase 1b CORRECT …"
python /bulk/aacudad/reasoning_traces/Training/phase1b_gemini_correct.py \
    --input  "$OUT_DIR/needs_correction.json" \
    --output "$OUT_DIR/gemini_corrected.jsonl" \
    --mode correct --max_concurrent 12 \
    2>&1 | tee "$OUT_DIR/phase1b_correct.log"

# Stage F — 1b rewrite
echo ""
echo "[F — $(date +%T)] Phase 1b REWRITE …"
python /bulk/aacudad/reasoning_traces/Training/phase1b_gemini_correct.py \
    --input  "$OUT_DIR/needs_rewrite_combined.json" \
    --output "$OUT_DIR/gemini_rewritten.jsonl" \
    --mode rewrite --max_concurrent 12 \
    2>&1 | tee "$OUT_DIR/phase1b_rewrite.log"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " ALL STAGES COMPLETE — $(date)"
echo "════════════════════════════════════════════════════════════════════════"
touch "$OUT_DIR/done.flag"
