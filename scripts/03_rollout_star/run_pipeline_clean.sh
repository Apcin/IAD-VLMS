#!/bin/bash
# Re-run Phase 0 → 1a → 1b on the CLEANED training data (type-code fix + location normalization).
# Same 100 items as previous pilot (seed=42) so outputs are directly comparable.
# CUDA 3 (per user). Judge prompt unchanged. Phase 1b uses NEW minimal-change prompts.
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache

OUT_DIR=/bulk/aacudad/reasoning_traces/Training/phase0_pilot_$(date +%Y%m%d_%H%M%S)_clean
mkdir -p "$OUT_DIR"

MODEL=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530

echo "════════════════════════════════════════════════════════════════════════"
echo " RE-RUN on CLEAN data — Phase 0 → 1a → 1b"
echo " Output dir:    $OUT_DIR"
echo " Start:         $(date)"
echo " GPU:           CUDA 3"
echo " Model:         run2/ckpt-530"
echo " Seed:          42 (same 100 items as previous pilot)"
echo " Judge prompt:  UNCHANGED (per user)"
echo " Phase 1b:      NEW minimal-change prompts"
echo "════════════════════════════════════════════════════════════════════════"

# ═══ Stage A — Phase 0 rollout (k=8 on 100 items) ═══════════════════════════
echo ""
echo "[A — $(date +%T)] Phase 0 rollout (k=8 × 100 items) on CUDA 3 …"

CUDA_VISIBLE_DEVICES=3 \
python /bulk/aacudad/reasoning_traces/Training/phase0_rollout_kscoring.py \
    --model_path "$MODEL" \
    --output_jsonl "$OUT_DIR/rollouts_raw.jsonl" \
    --k 8 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 \
    --max_pixels 480000 \
    --limit 100 \
    --seed 42 \
    2>&1 | tee "$OUT_DIR/rollout.log"

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
echo "[C — $(date +%T)] Phase 1a Gemini judge (unchanged prompt) …"

python /bulk/aacudad/reasoning_traces/Training/phase1a_gemini_judge.py \
    --input           "$OUT_DIR/judge_input.json" \
    --output_faithful "$OUT_DIR/good_traces_faithful.json" \
    --output_rewrite  "$OUT_DIR/needs_rewrite_from_judge.json" \
    --report_jsonl    "$OUT_DIR/judge_report.jsonl" \
    --keep_threshold 2 \
    --max_concurrent 4 \
    2>&1 | tee "$OUT_DIR/phase1a.log"

echo "[C — $(date +%T)] Done."

# ═══ Stage D — Union needs_rewrite (local) + needs_rewrite_from_judge ══════
echo ""
echo "[D — $(date +%T)] Building needs_rewrite_combined.json …"

python <<PYEOF
import json
with open('$OUT_DIR/needs_rewrite.json') as f: local = json.load(f)
with open('$OUT_DIR/needs_rewrite_from_judge.json') as f: judge = json.load(f)
seen = set()
combined = []
for src, name in [(local, 'local'), (judge, 'judge')]:
    for it in src:
        if it['image_path'] not in seen:
            seen.add(it['image_path'])
            combined.append(it)
with open('$OUT_DIR/needs_rewrite_combined.json', 'w') as f:
    json.dump(combined, f, indent=2)
print(f'  local: {len(local)}, judge: {len(judge)}, combined (union): {len(combined)}')
PYEOF

# ═══ Stage E — Phase 1b CORRECT (minimal-change) ═══════════════════════════
echo ""
echo "[E — $(date +%T)] Phase 1b CORRECT with new minimal-change prompt …"

python /bulk/aacudad/reasoning_traces/Training/phase1b_gemini_correct.py \
    --input  "$OUT_DIR/needs_correction.json" \
    --output "$OUT_DIR/gemini_corrected.jsonl" \
    --mode correct \
    --max_concurrent 4 \
    2>&1 | tee "$OUT_DIR/phase1b_correct.log"

# ═══ Stage F — Phase 1b REWRITE (minimal-change) ═══════════════════════════
echo ""
echo "[F — $(date +%T)] Phase 1b REWRITE with new minimal-change prompt …"

python /bulk/aacudad/reasoning_traces/Training/phase1b_gemini_correct.py \
    --input  "$OUT_DIR/needs_rewrite_combined.json" \
    --output "$OUT_DIR/gemini_rewritten.jsonl" \
    --mode rewrite \
    --max_concurrent 4 \
    2>&1 | tee "$OUT_DIR/phase1b_rewrite.log"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " ALL STAGES COMPLETE — $(date)"
echo " Output dir:  $OUT_DIR"
echo "════════════════════════════════════════════════════════════════════════"
touch "$OUT_DIR/done.flag"
