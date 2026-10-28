#!/bin/bash
# Top-level orchestrator: rollout → bucket → judge → 1b → assemble → SFT → eval
# Runs everything sequentially with explicit polling for each stage's done.flag.
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

HELDOUT_OUT=/bulk/aacudad/reasoning_traces/Training/phase0_heldout_20260601
SFT_OUT=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_iter2_heldout

# ─── 1. Wait for both rollout shards (launched separately in their own tmux) ───
echo "[chain] Waiting for held-out pipeline.done flag (rollout+bucket+judge+1b) …"
until [ -f "$HELDOUT_OUT/pipeline.done" ]; do
  sleep 120
done
echo "[chain] Held-out pipeline done @ $(date)"

# ─── 2. Build the SFT-Iter2 heldout dataset ───
echo "[chain] Building SFT dataset …"
python /bulk/aacudad/reasoning_traces/Training/build_sft_iter2_heldout_dataset.py
DATASET=/bulk/aacudad/reasoning_traces/Training/datasets_sft_iter2/sft_iter2_heldout_train.json
ls -lh "$DATASET" || { echo "[chain] Dataset build failed"; exit 1; }

# ─── 3. Register in LF dataset_info.json ───
echo "[chain] Registering iad_sft_iter2_heldout in LLaMA-Factory …"
python <<PYEOF
import json
INFO = '/bulk/aacudad/reasoning_traces/LlamaFactory/data/dataset_info.json'
with open(INFO) as f: info = json.load(f)
info['iad_sft_iter2_heldout'] = {
    "file_name": "$DATASET",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {"role_tag": "role", "content_tag": "content",
             "user_tag": "user", "assistant_tag": "assistant"},
}
with open(INFO, 'w') as f: json.dump(info, f, indent=2)
print("✓ Registered iad_sft_iter2_heldout")
PYEOF

# ─── 4. Launch SFT training ───
echo "[chain] Launching SFT training …"
mkdir -p "$SFT_OUT"
export CUDA_VISIBLE_DEVICES=0,3
export FORCE_TORCHRUN=1
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export TRITON_CACHE_DIR=/tmp/triton_cache_aacudad
mkdir -p /tmp/triton_cache_aacudad

cd /bulk/aacudad/reasoning_traces/LlamaFactory
conda run -n llama_sft --no-capture-output \
    llamafactory-cli train /bulk/aacudad/reasoning_traces/Training/sft_iter2_heldout_qwen25vl_7b.yaml \
    2>&1 | tee "$SFT_OUT/train.log"
touch "$SFT_OUT/done.flag"
echo "[chain] SFT done @ $(date)"

# ─── 5. Eval lanes ───
echo "[chain] Launching eval lanes …"
EVAL_SCRIPT=/bulk/aacudad/reasoning_traces/Training/evaluate_qwen25vl_7b_trainprompt.py
BASE=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564

CKPTS=()
for c in $(ls -d $SFT_OUT/checkpoint-* 2>/dev/null | sort -V); do CKPTS+=("$c"); done

run_lane() {
  local gpu=$1; local dataset_flag=$2; local ds_label=$3
  local logfile=$SFT_OUT/eval_lane_g${gpu}_${ds_label}.log
  {
    for ckpt in "${CKPTS[@]}"; do
      out="$ckpt/eval_${ds_label}_full_trainprompt.json"
      [ -f "$out" ] && continue
      echo "─── [g$gpu $(date +%H:%M:%S)] $(basename $ckpt) @ $ds_label ───"
      CUDA_VISIBLE_DEVICES=$gpu \
      python "$EVAL_SCRIPT" \
        --checkpoint "$ckpt" \
        --base-model "$BASE" \
        $dataset_flag \
        --batch-size 4 \
        --output "$out"
    done
    touch "$SFT_OUT/eval_g${gpu}.done"
  } > "$logfile" 2>&1
}

run_lane 0 "--ds-mvtec-only" "dsmvtec" &
PID0=$!
run_lane 3 "--visa-only" "visa" &
PID3=$!
wait $PID0 $PID3
echo "[chain] All evals done @ $(date)"
touch "$SFT_OUT/all_evals.done"
