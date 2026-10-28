#!/bin/bash
# Every 10 min: log step/loss of the unfrozen run. When SFT_DONE.flag appears, evaluate checkpoints
# (epoch 3 and 4 first, in parallel on cuda 1 and 2, then 2 and 1) with the train prompt, strict BA.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh; conda activate llama_sft
export HF_HOME=${WORK_DIR}/hf_cache
cd ${WORK_DIR}
OUT=${WORK_DIR}/outputs/sft_qwen25vl_7b_6k_unfrozen_lr1e-5
EVAL=${WORK_DIR}/Training/evaluate_qwen25vl_7b_trainprompt.py
BASE=Qwen/Qwen2.5-VL-7B-Instruct
BA=${WORK_DIR}/Training/strict_ba.py
ST=$OUT/watcher_status.log
while [ ! -f "$OUT/SFT_DONE.flag" ]; do
  last=$(grep -o "{'loss': [^}]*}" $OUT/train.log 2>/dev/null | tail -1)
  prog=$(tr '\r' '\n' < $OUT/train.log 2>/dev/null | grep -o "[0-9]*/[0-9]* \[[0-9:]*<[0-9:]*" | tail -1)
  err=$(grep -c "Traceback\|CUDA out of memory\|Error" $OUT/train.log 2>/dev/null)
  echo "[$(date +%F' '%T)] progress=$prog last=$last errors=$err ckpts=$(ls -d $OUT/checkpoint-* 2>/dev/null | wc -l)" | tee -a $ST
  if ! pgrep -f "sft_qwen25vl_7b_6k_unfrozen_lr1e-5.yaml" >/dev/null && [ ! -f "$OUT/SFT_DONE.flag" ]; then echo "[$(date +%T)] TRAINING PROCESS GONE WITHOUT DONE FLAG" | tee -a $ST; break; fi
  sleep 600
done
[ -f "$OUT/SFT_DONE.flag" ] || exit 1
echo "[$(date +%T)] SFT complete, evaluating" | tee -a $ST
evalck(){ ck=$1; gpu=$2
  dsj="$ck/eval_dsmvtec_full_trainprompt.json"; vsj="$ck/eval_visa_full_trainprompt.json"
  [ -f "$dsj" ] || CUDA_VISIBLE_DEVICES=$gpu python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --ds-mvtec-only --output "$dsj" > "$ck/eval_ds.log" 2>&1
  [ -f "$vsj" ] || CUDA_VISIBLE_DEVICES=$gpu python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --visa-only --output "$vsj" > "$ck/eval_visa.log" 2>&1
  echo "[$(date +%T)] $(basename $ck): DS $(python3 $BA $dsj 2>/dev/null | grep -oE 'BA=[0-9.]+')  VisA $(python3 $BA $vsj 2>/dev/null | grep -oE 'BA=[0-9.]+')" | tee -a $ST; }
cks=($(ls -d $OUT/checkpoint-* | sort -t- -k2 -n)); n=${#cks[@]}
# order: last two first (epochs 3,4 when n=4), then the rest
order=(); for ((i=n-2;i<n;i++)); do [ $i -ge 0 ] && order+=("${cks[$i]}"); done; for ((i=0;i<n-2;i++)); do order+=("${cks[$i]}"); done
for ((i=0;i<${#order[@]};i+=2)); do
  evalck "${order[$i]}" 1 & p1=$!
  [ $((i+1)) -lt ${#order[@]} ] && { evalck "${order[$((i+1))]}" 2 & p2=$!; } || p2=""
  wait $p1; [ -n "$p2" ] && wait $p2
done
echo "=== unfrozen lr1e-5 6K, all epochs ===" | tee -a $ST
for ck in "${cks[@]}"; do echo "  $(basename $ck): DS $(python3 $BA $ck/eval_dsmvtec_full_trainprompt.json 2>/dev/null | grep -oE 'BA=[0-9.]+')  VisA $(python3 $BA $ck/eval_visa_full_trainprompt.json 2>/dev/null | grep -oE 'BA=[0-9.]+')" | tee -a $ST; done
touch $OUT/ALL_EVALS_DONE.flag; echo "[$(date +%T)] watcher done" | tee -a $ST
