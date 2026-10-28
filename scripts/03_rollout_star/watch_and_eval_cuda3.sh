#!/bin/bash
# Decoupled eval watchdog on CUDA 3 for the 3-epoch GRPO run (trains on CUDA 1,2).
# As each new checkpoint appears, evals it on CUDA 3 — DS-MVTec THEN VisA,
# SEQUENTIALLY (never two evals at once, so it coexists with whatever else is on
# CUDA 3). Eval prompt = default (system "Please answer by yes or no" + Analyze),
# matching the 82.80 baseline + previous runs for a fair comparison.
# Skip-if-already-evaluated -> safely re-runnable. Exits when training done.flag
# is set AND every checkpoint has both eval JSONs.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache

OUT="${1:-/bulk/aacudad/reasoning_traces/outputs/grpo_sftprompt_kl0.1_3ep}"
EVAL=/bulk/aacudad/reasoning_traces/Training/evaluate_qwen25vl_7b_trainprompt.py
BASE=Qwen/Qwen2.5-VL-7B-Instruct
GPU=3
mkdir -p "$OUT"
echo "[$(date +%T)] cuda3 eval watchdog armed for $OUT (DS then VisA, sequential, on GPU $GPU)"

eval_one(){
  local ck="$1"
  local dsj="$ck/eval_dsmvtec_full_trainprompt.json" vsj="$ck/eval_visa_full_trainprompt.json"
  if [ ! -f "$dsj" ]; then
    echo "[$(date +%T)] EVAL $(basename $ck) DS-MVTec on cuda$GPU"
    CUDA_VISIBLE_DEVICES=$GPU python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --ds-mvtec-only \
        --output "$dsj" > "$ck/eval_ds.log" 2>&1
  fi
  if [ ! -f "$vsj" ]; then
    echo "[$(date +%T)] EVAL $(basename $ck) VisA on cuda$GPU"
    CUDA_VISIBLE_DEVICES=$GPU python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --visa-only \
        --output "$vsj" > "$ck/eval_visa.log" 2>&1
  fi
  local d v
  d=$(python3 /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/results/compute_ba.py "$dsj" 2>/dev/null | grep -oE "BA=[0-9.]+")
  v=$(python3 /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/results/compute_ba.py "$vsj" 2>/dev/null | grep -oE "BA=[0-9.]+")
  echo "[$(date +%T)] $(basename $ck): DS $d  VisA $v"
}

while :; do
  # eval any checkpoint (with a full model = config.json) lacking eval JSONs, in step order
  for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    [ -f "$ck/config.json" ] || continue
    { [ -f "$ck/eval_dsmvtec_full_trainprompt.json" ] && [ -f "$ck/eval_visa_full_trainprompt.json" ]; } && continue
    eval_one "$ck"
  done
  # done condition: training finished AND all checkpoints evaluated
  if [ -f "$OUT/.traindone" ]; then
    pending=0
    for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null); do
      [ -f "$ck/config.json" ] || continue
      { [ -f "$ck/eval_dsmvtec_full_trainprompt.json" ] && [ -f "$ck/eval_visa_full_trainprompt.json" ]; } || pending=$((pending+1))
    done
    [ "$pending" -eq 0 ] && { echo "[$(date +%T)] all checkpoints evaluated — watchdog done."; break; }
  fi
  sleep 120
done

# BA trajectory table
echo "════ BA TRAJECTORY ($(basename $OUT)) ════" | tee "$OUT/ba_table.txt"
for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  for b in dsmvtec visa; do f="$ck/eval_${b}_full_trainprompt.json"
    [ -f "$f" ] && python3 /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/results/compute_ba.py "$f" 2>/dev/null \
        | sed "s|^|$(basename $ck) $b: |" | tee -a "$OUT/ba_table.txt"; done
done
touch "$OUT/.evalwatch_done"
