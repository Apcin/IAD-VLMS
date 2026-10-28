#!/bin/bash
# Full 10k run: Phase 0 → 1a → 1b on all 10,236 items in the cleaned SFT+GRPO pools.
# CUDA 3. Rollout supports resume on crash (appends to rollouts_raw.jsonl).
# Phase 1a uses unchanged judge prompt; Phase 1b uses new minimal-change prompts.

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache

OUT_DIR=/bulk/aacudad/reasoning_traces/Training/phase0_full_10k_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"

MODEL=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530

echo "════════════════════════════════════════════════════════════════════════"
echo " FULL 10k RUN — Phase 0 → 1a → 1b on cleaned data"
echo " Output dir: $OUT_DIR"
echo " Start:      $(date)"
echo " GPU:        CUDA 3"
echo " Model:      run2/ckpt-530"
echo " Items:      ALL (10,236 = 6000 SFT + 4236 GRPO)"
echo " k:          8"
echo "════════════════════════════════════════════════════════════════════════"

# ═══ Stage A — Phase 0 rollout (resumable on crash) ════════════════════════
echo ""
echo "[A — $(date +%T)] Phase 0 rollout (resumable) on CUDA 3 …"
echo "  Estimated time: ~15-18 hours single-GPU"

CUDA_VISIBLE_DEVICES=3 \
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

echo "[A — $(date +%T)] Done. $(wc -l < $OUT_DIR/rollouts_raw.jsonl) items written."

# ═══ Stage B — Phase 0 bucket ═══════════════════════════════════════════════
echo ""
echo "[B — $(date +%T)] Bucketing …"

python /bulk/aacudad/reasoning_traces/Training/phase0_bucket.py \
    --input_jsonl "$OUT_DIR/rollouts_raw.jsonl" \
    --out_dir "$OUT_DIR" \
    --threshold_yes 1.5 \
    --threshold_no 1.0 \
    --weak_threshold_yes 1.8 \
    2>&1 | tee "$OUT_DIR/bucket.log"

# ═══ Stage C — Phase 1a Gemini judge ═══════════════════════════════════════
echo ""
echo "[C — $(date +%T)] Phase 1a Gemini judge (max_concurrent=12) …"

python /bulk/aacudad/reasoning_traces/Training/phase1a_gemini_judge.py \
    --input           "$OUT_DIR/judge_input.json" \
    --output_faithful "$OUT_DIR/good_traces_faithful.json" \
    --output_rewrite  "$OUT_DIR/needs_rewrite_from_judge.json" \
    --report_jsonl    "$OUT_DIR/judge_report.jsonl" \
    --keep_threshold 2 \
    --max_concurrent 12 \
    2>&1 | tee "$OUT_DIR/phase1a.log"

echo "[C — $(date +%T)] Done."

# ═══ Stage D — Build needs_rewrite_combined ═════════════════════════════════
echo ""
echo "[D — $(date +%T)] Building needs_rewrite_combined.json …"

python <<PYEOF
import json
with open('$OUT_DIR/needs_rewrite.json') as f: local = json.load(f)
with open('$OUT_DIR/needs_rewrite_from_judge.json') as f: judge = json.load(f)
seen = set()
combined = []
for src in [local, judge]:
    for it in src:
        if it['image_path'] not in seen:
            seen.add(it['image_path'])
            combined.append(it)
with open('$OUT_DIR/needs_rewrite_combined.json', 'w') as f:
    json.dump(combined, f, indent=2)
print(f'  local: {len(local)}, judge: {len(judge)}, combined (union): {len(combined)}')
PYEOF

# ═══ Stage E — Phase 1b CORRECT (minimal-change, resumable) ════════════════
echo ""
echo "[E — $(date +%T)] Phase 1b CORRECT (max_concurrent=12) …"

python /bulk/aacudad/reasoning_traces/Training/phase1b_gemini_correct.py \
    --input  "$OUT_DIR/needs_correction.json" \
    --output "$OUT_DIR/gemini_corrected.jsonl" \
    --mode correct \
    --max_concurrent 12 \
    2>&1 | tee "$OUT_DIR/phase1b_correct.log"

# ═══ Stage F — Phase 1b REWRITE (minimal-change, resumable) ════════════════
echo ""
echo "[F — $(date +%T)] Phase 1b REWRITE (max_concurrent=12) …"

python /bulk/aacudad/reasoning_traces/Training/phase1b_gemini_correct.py \
    --input  "$OUT_DIR/needs_rewrite_combined.json" \
    --output "$OUT_DIR/gemini_rewritten.jsonl" \
    --mode rewrite \
    --max_concurrent 12 \
    2>&1 | tee "$OUT_DIR/phase1b_rewrite.log"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " ALL STAGES COMPLETE — $(date)"
echo " Output dir:  $OUT_DIR"
echo "════════════════════════════════════════════════════════════════════════"
touch "$OUT_DIR/done.flag"
