#!/bin/bash
# Run one rollout shard of the held-out 4k pool.
# Usage: bash run_heldout_shard.sh <shard_id> <cuda_id>
set -eo pipefail   # pipefail so failures inside python|tee aren't silently swallowed
SHARD_ID=$1
CUDA_ID=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache

OUT_DIR=/bulk/aacudad/reasoning_traces/Training/phase0_heldout_20260601
HELDOUT=/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only_fixed/new_sft_c1_train.json
EMPTY=/bulk/aacudad/reasoning_traces/Training/empty_pool.json
MODEL=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530
mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════"
echo " HELD-OUT shard $SHARD_ID on CUDA $CUDA_ID"
echo " Start: $(date)"
echo "════════════════════════════════════════"

CUDA_VISIBLE_DEVICES=$CUDA_ID \
python /bulk/aacudad/reasoning_traces/Training/phase0_rollout_kscoring.py \
    --model_path "$MODEL" \
    --sft_pool "$HELDOUT" \
    --grpo_pool "$EMPTY" \
    --output_jsonl "$OUT_DIR/rollouts_shard${SHARD_ID}.jsonl" \
    --shard_id $SHARD_ID --num_shards 2 \
    --k 8 \
    --temperature 0.7 --top_p 0.9 \
    --max_new_tokens 512 --max_pixels 480000 \
    --seed 42 \
    2>&1 | tee -a "$OUT_DIR/rollout_shard${SHARD_ID}.log"

echo "[done] shard $SHARD_ID @ $(date)"
touch "$OUT_DIR/done_shard${SHARD_ID}.flag"
