#!/bin/bash
# Decoupled eval watchdog for the 0.5-epoch KL GRPO drift test.
# Fires when the GRPO training writes .traindone (cuda 1,2 free again), then
# evaluates EVERY checkpoint: DS-MVTec -> CUDA 1, VisA -> CUDA 2 (in parallel),
# and prints a balanced-accuracy drift table. Robust to the exact beta value
# (globs grpo_abc_C_kl*_halfep) and re-runnable (skips checkpoints already evaled).
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache

EVAL=/bulk/aacudad/reasoning_traces/Training/evaluate_qwen25vl_7b_trainprompt.py
BASE=Qwen/Qwen2.5-VL-7B-Instruct
GLOB=/bulk/aacudad/reasoning_traces/outputs/grpo_abc_C_kl*_halfep

echo "[$(date +%T)] grpo eval watchdog armed; waiting for <grpo_abc_C_kl*_halfep>/.traindone"

# 1) wait for GRPO training to finish
OUT=""
while [ -z "$OUT" ]; do
  for d in $GLOB; do [ -f "$d/.traindone" ] && OUT="$d"; done
  [ -z "$OUT" ] && sleep 60
done
echo "[$(date +%T)] training done -> $OUT"

# 2) wait until CUDA 1,2 are actually free (training procs exited)
while :; do
  u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null)
  u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 2>/dev/null)
  if [ "${u1:-9999}" -lt 3000 ] && [ "${u2:-9999}" -lt 3000 ]; then break; fi
  echo "[$(date +%T)] waiting for cuda1/2 to free (cuda1=${u1} cuda2=${u2} MiB)"
  sleep 30
done
echo "[$(date +%T)] CUDA 1,2 free -> evaluating checkpoints"

# 3) eval each checkpoint: DS->cuda1, VisA->cuda2 in parallel
for ck in $(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  [ -f "$ck/config.json" ] || continue
  dsj="$ck/eval_dsmvtec_full_trainprompt.json"; vsj="$ck/eval_visa_full_trainprompt.json"
  [ -f "$dsj" ] && [ -f "$vsj" ] && { echo "[$(date +%T)] $(basename $ck) already evaled, skip"; continue; }
  echo "[$(date +%T)] EVAL $(basename $ck): DS->cuda1, VisA->cuda2"
  CUDA_VISIBLE_DEVICES=1 python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --ds-mvtec-only \
      --output "$dsj" > "$ck/eval_ds.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=2 python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --visa-only \
      --output "$vsj" > "$ck/eval_visa.log" 2>&1 &
  wait
  echo "[$(date +%T)] $(basename $ck) done"
done
touch "$OUT/.evaldone"

# 4) BA drift table
echo "════════ KL-DRIFT BA TABLE ($(basename $OUT)) ════════" | tee "$OUT/ba_table.txt"
for ck in $(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  for b in dsmvtec visa; do
    f="$ck/eval_${b}_full_trainprompt.json"
    [ -f "$f" ] && python3 /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/results/compute_ba.py "$f" 2>/dev/null \
        | sed "s|^|$(basename $ck) $b: |" | tee -a "$OUT/ba_table.txt"
  done
done
echo "[$(date +%T)] ALL DONE -> $OUT/ba_table.txt"
touch "$OUT/done.flag"
