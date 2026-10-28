#!/bin/bash
# Watchdog: waits for each varmix continuation-SFT run to finish (.traindone), then evaluates
# every saved epoch checkpoint on DS-MVTec + VisA (trainprompt, full) on the run's GPUs.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
cd /bulk/aacudad/reasoning_traces
EVAL=Training/evaluate_qwen25vl_7b_trainprompt.py
BASE=Qwen/Qwen2.5-VL-7B-Instruct

names=(armC grpo)
declare -A OUT=( [armC]=outputs/sft_varmix_from_armC [grpo]=outputs/sft_varmix_from_grpo )
declare -A GDS=( [armC]=0 [grpo]=1 )
declare -A GVS=( [armC]=3 [grpo]=2 )

eval_run () {
  local name=$1 dir=${OUT[$1]} gds=${GDS[$1]} gvs=${GVS[$1]}
  echo "[$(date)] $name: training done -> evaluating checkpoints in $dir"
  for ckpt in $(ls -d "$dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    echo "[$(date)]   eval $ckpt  (DS->cuda$gds, VisA->cuda$gvs)"
    CUDA_VISIBLE_DEVICES=$gds python "$EVAL" --checkpoint "$ckpt" --base-model "$BASE" --ds-mvtec-only \
        --output "$ckpt/eval_dsmvtec_full_trainprompt.json" > "$ckpt/eval_ds.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$gvs python "$EVAL" --checkpoint "$ckpt" --base-model "$BASE" --visa-only \
        --output "$ckpt/eval_visa_full_trainprompt.json" > "$ckpt/eval_visa.log" 2>&1 &
    wait
    echo "[$(date)]   done $ckpt"
  done
  touch "$dir/.evaldone"
  echo "[$(date)] $name: all checkpoint evals complete"
}

echo "[$(date)] varmix watchdog up; waiting for .traindone in: ${OUT[armC]} , ${OUT[grpo]}"
while true; do
  for name in "${names[@]}"; do
    dir=${OUT[$name]}
    if [ -f "$dir/.traindone" ] && [ ! -f "$dir/.evaldone" ]; then
      eval_run "$name"
    fi
  done
  if [ -f "${OUT[armC]}/.evaldone" ] && [ -f "${OUT[grpo]}/.evaldone" ]; then
    echo "[$(date)] ALL DONE — both varmix runs trained + evaluated."
    echo "=== final balanced accuracy (init: Arm-C 82.80/72.07, GRPO 82.73/70.39) ==="
    for name in "${names[@]}"; do
      for ckpt in $(ls -d "${OUT[$name]}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
        for b in dsmvtec visa; do
          f="$ckpt/eval_${b}_full_trainprompt.json"
          [ -f "$f" ] && python repository_tu_delft_vlms/results/compute_ba.py "$f" 2>/dev/null | sed "s|^|$name $(basename $ckpt) $b: |"
        done
      done
    done
    break
  fi
  sleep 60
done
