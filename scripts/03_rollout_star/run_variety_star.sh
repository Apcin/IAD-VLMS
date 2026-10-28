#!/bin/bash
# Variety STaR pipeline — EXACT Arm-C recipe applied to Real-IAD Variety C1.
#   roll out the 82.80 Arm-C model (ckpt-376) -> Gemini faithfulness judge ->
#   Gemini correct (0/k fails) + rewrite (marginal) -> (kept+corrected+rewritten).
# Generator/init: Arm-C checkpoint-376 (82.80/72.07). Pool: Variety 6K SFT split.
# 2 GPUs: CUDA 0 + 3 (NVLink NV4 pair). Gated on the Gemini generation completing.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache

RG=/bulk/aacudad/reasoning_traces/reasoning_traces_gen
TR=/bulk/aacudad/reasoning_traces/Training
MODEL=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376  # 82.80 Arm-C
OUT_DIR=$TR/phase0_variety_star_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"
SFT_POOL=$RG/variety_c1_compiled/sft_pool_with_traces.json
EMPTY_POOL=$RG/variety_c1_compiled/_empty_pool.json
echo "[]" > "$EMPTY_POOL"

echo "════════════════════════════════════════════════════════════════════════"
echo " VARIETY STaR — rollout(ckpt-376) -> judge -> correct -> rewrite"
echo " Out: $OUT_DIR   GPUs: CUDA 0+3   k=8"
echo "════════════════════════════════════════════════════════════════════════"

# 0) gate on Gemini generation; then assemble the SFT pool (image_path+product+gt_trace)
echo "[gate $(date +%T)] waiting for GEN_DONE.flag ..."
while [ ! -f "$RG/GEN_DONE.flag" ]; do sleep 120; done
echo "[assemble $(date +%T)] merging Gemini traces into the rollout pool ..."
( cd "$RG" && GEMINI_API_KEYS=dummy python3 assemble_variety_sft_pool.py 2>&1 | tee "$OUT_DIR/assemble.log" )

# A) rollout, 2 shards across CUDA 0 and 3 (each loads full pool, filters by shard)
echo "[A $(date +%T)] Phase 0 rollout on CUDA 0+3 (k=8) ..."
CUDA_VISIBLE_DEVICES=0 python "$TR/phase0_rollout_kscoring.py" \
    --model_path "$MODEL" --sft_pool "$SFT_POOL" --grpo_pool "$EMPTY_POOL" \
    --output_jsonl "$OUT_DIR/rollouts_shard0.jsonl" \
    --k 8 --temperature 0.7 --top_p 0.9 --max_new_tokens 512 --max_pixels 480000 \
    --num_shards 2 --shard_id 0 --seed 42 \
    > "$OUT_DIR/rollout_shard0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=3 python "$TR/phase0_rollout_kscoring.py" \
    --model_path "$MODEL" --sft_pool "$SFT_POOL" --grpo_pool "$EMPTY_POOL" \
    --output_jsonl "$OUT_DIR/rollouts_shard1.jsonl" \
    --k 8 --temperature 0.7 --top_p 0.9 --max_new_tokens 512 --max_pixels 480000 \
    --num_shards 2 --shard_id 1 --seed 42 \
    > "$OUT_DIR/rollout_shard1.log" 2>&1 &
P1=$!
wait $P0; wait $P1
cat "$OUT_DIR/rollouts_shard0.jsonl" "$OUT_DIR/rollouts_shard1.jsonl" > "$OUT_DIR/rollouts_raw.jsonl"
echo "[A $(date +%T)] rollout done — $(wc -l < $OUT_DIR/rollouts_raw.jsonl) items"

# B) bucket into good / needs_rewrite / needs_correction (+ judge_input)
echo "[B $(date +%T)] bucketing ..."
python "$TR/phase0_bucket.py" --input_jsonl "$OUT_DIR/rollouts_raw.jsonl" --out_dir "$OUT_DIR" \
    --threshold_yes 1.5 --threshold_no 1.0 --weak_threshold_yes 1.8 2>&1 | tee "$OUT_DIR/bucket.log"

# C) Gemini faithfulness judge
echo "[C $(date +%T)] Phase 1a Gemini judge ..."
python "$TR/phase1a_gemini_judge.py" \
    --input "$OUT_DIR/judge_input.json" \
    --output_faithful "$OUT_DIR/good_traces_faithful.json" \
    --output_rewrite "$OUT_DIR/needs_rewrite_from_judge.json" \
    --report_jsonl "$OUT_DIR/judge_report.jsonl" \
    --keep_threshold 2 --max_concurrent 12 2>&1 | tee "$OUT_DIR/phase1a.log"

# D) union needs_rewrite (local + judge)
python - "$OUT_DIR" <<'PYEOF'
import json,sys
o=sys.argv[1]
local=json.load(open(f"{o}/needs_rewrite.json"))
judge=json.load(open(f"{o}/needs_rewrite_from_judge.json"))
seen=set(); comb=[]
for src in (local,judge):
    for it in src:
        if it["image_path"] not in seen:
            seen.add(it["image_path"]); comb.append(it)
json.dump(comb,open(f"{o}/needs_rewrite_combined.json","w"),indent=2)
print(f"  rewrite union: local={len(local)} judge={len(judge)} -> {len(comb)}")
PYEOF

# E) Gemini CORRECT (0/k fails)
echo "[E $(date +%T)] Phase 1b CORRECT ..."
python "$TR/phase1b_gemini_correct.py" --input "$OUT_DIR/needs_correction.json" \
    --output "$OUT_DIR/gemini_corrected.jsonl" --mode correct --max_concurrent 12 \
    2>&1 | tee "$OUT_DIR/phase1b_correct.log"

# F) Gemini REWRITE (marginal)
echo "[F $(date +%T)] Phase 1b REWRITE ..."
python "$TR/phase1b_gemini_correct.py" --input "$OUT_DIR/needs_rewrite_combined.json" \
    --output "$OUT_DIR/gemini_rewritten.jsonl" --mode rewrite --max_concurrent 12 \
    2>&1 | tee "$OUT_DIR/phase1b_rewrite.log"

# G) assemble FINAL SFT corpus: EXACTLY 6,000, stratified product x normal/anomaly
echo "[G $(date +%T)] assembling exactly-6K stratified SFT corpus ..."
python "$TR/assemble_variety_star_6k.py" \
    --phase0_dir "$OUT_DIR" \
    --out "$OUT_DIR/variety_star_sft_6k.json" \
    --target 6000 2>&1 | tee "$OUT_DIR/assemble_6k.log"

echo "════════════════════════════════════════════════════════════════════════"
echo " VARIETY STaR STAGES COMPLETE — $(date)   Out: $OUT_DIR"
echo " Final 6K SFT corpus: $OUT_DIR/variety_star_sft_6k.json"
echo " Next (manual): SFT a fresh model on this 6K (frozen vision, Arm-C config)"
echo "════════════════════════════════════════════════════════════════════════"
touch "$OUT_DIR/done.flag"
echo "$OUT_DIR" > "$TR/.last_variety_star_dir"
