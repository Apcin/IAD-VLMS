#!/bin/bash
# Decoupled eval watcher for the Qwen3-VL-8B Arm-C SFT (which trains on cuda 1,2).
# As each epoch checkpoint appears, evals it on cuda 3 -- DS-MVTec THEN VisA, SEQUENTIALLY
# (never two evals at once). Default eval prompt (system "Please answer by yes or no" + Analyze) =
# the exact 82.80-baseline prompt, so the 8B numbers are directly comparable to the 7B Arm-C.
# Waits for cuda 3 to be free (<10GB) before each eval, so it coexists with the GRPO watcher's
# tail evals. Skip-if-done -> re-runnable. Exits when train.done exists AND every ckpt is eval'd.
# Usage: watch_and_eval_qwen3vl_cuda3.sh [OUT_DIR] [GPU]
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
# Paths. WORK_DIR is the workspace holding LlamaFactory/, outputs/ and hf_cache/; it
# defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
export HF_HOME=$WORK_DIR/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME

OUT="${1:-$WORK_DIR/outputs/sft_qwen3vl_8b_armC}"
GPU="${2:-3}"
EVAL=$HERE/evaluate_qwen25vl_7b_trainprompt.py
BASE=Qwen/Qwen3-VL-8B-Instruct
BA=$REPO_ROOT/results/compute_ba.py
mkdir -p "$OUT"
echo "[$(date +%T)] qwen3vl eval watcher armed for $OUT (DS then VisA, sequential, cuda $GPU, base=$BASE)"

gpu_mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
wait_gpu(){ while [ "$(gpu_mem $GPU)" -gt 10000 ]; do echo "[$(date +%T)] cuda$GPU busy ($(gpu_mem $GPU)MiB) -- waiting"; sleep 60; done; }

eval_one(){
  local ck="$1"
  local dsj="$ck/eval_dsmvtec_full_trainprompt.json" vsj="$ck/eval_visa_full_trainprompt.json"
  if [ ! -f "$dsj" ]; then
    wait_gpu; echo "[$(date +%T)] EVAL $(basename $ck) DS-MVTec on cuda$GPU"
    CUDA_VISIBLE_DEVICES=$GPU python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --ds-mvtec-only \
        --output "$dsj" > "$ck/eval_ds.log" 2>&1
  fi
  if [ ! -f "$vsj" ]; then
    wait_gpu; echo "[$(date +%T)] EVAL $(basename $ck) VisA on cuda$GPU"
    CUDA_VISIBLE_DEVICES=$GPU python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --visa-only \
        --output "$vsj" > "$ck/eval_visa.log" 2>&1
  fi
  local d v
  d=$(python3 "$BA" "$dsj" 2>/dev/null | grep -oE "BA=[0-9.]+")
  v=$(python3 "$BA" "$vsj" 2>/dev/null | grep -oE "BA=[0-9.]+")
  echo "[$(date +%T)] $(basename $ck): DS $d  VisA $v"
}

while :; do
  for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    [ -f "$ck/config.json" ] || continue
    { [ -f "$ck/eval_dsmvtec_full_trainprompt.json" ] && [ -f "$ck/eval_visa_full_trainprompt.json" ]; } && continue
    eval_one "$ck"
  done
  if [ -f "$OUT/train.done" ]; then
    pending=0
    for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null); do
      [ -f "$ck/config.json" ] || continue
      { [ -f "$ck/eval_dsmvtec_full_trainprompt.json" ] && [ -f "$ck/eval_visa_full_trainprompt.json" ]; } || pending=$((pending+1))
    done
    [ "$pending" -eq 0 ] && { echo "[$(date +%T)] all checkpoints evaluated -- watcher done."; break; }
  fi
  sleep 120
done

echo "[$(date +%T)] === Qwen3-VL-8B Arm-C BA trajectory ==="
for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  d=$(python3 "$BA" "$ck/eval_dsmvtec_full_trainprompt.json" 2>/dev/null | grep -oE "BA=[0-9.]+")
  v=$(python3 "$BA" "$ck/eval_visa_full_trainprompt.json" 2>/dev/null | grep -oE "BA=[0-9.]+")
  echo "  $(basename $ck): DS $d  VisA $v"
done
