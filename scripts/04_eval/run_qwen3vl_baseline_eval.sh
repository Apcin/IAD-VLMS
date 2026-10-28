#!/bin/bash
# Baseline eval of Qwen3-VL-8B-Instruct (NO fine-tuning) on cuda 0: DS-MVTec THEN VisA, sequential.
# Default eval prompt (system "Please answer by yes or no" + Analyze) = the 82.80-baseline prompt,
# so this is directly comparable to the Qwen2.5-VL-7B base row (69.01/53.80) and the Arm-C students.
# Also pre-downloads the 16GB weights into $WORK_DIR/hf_cache and validates the eval path on real Qwen3-VL inference.
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
export HF_HUB_CACHE=$WORK_DIR/hf_cache/hub
export TRANSFORMERS_CACHE=$WORK_DIR/hf_cache
EVAL=$HERE/evaluate_qwen25vl_7b_trainprompt.py
BA=$REPO_ROOT/results/compute_ba.py
BASE=Qwen/Qwen3-VL-8B-Instruct
OUT=$WORK_DIR/outputs/qwen3vl_8b_baseline_eval
GPU="${1:-0}"
mkdir -p "$OUT"

echo "[$(date '+%F %T')] Qwen3-VL-8B-Instruct BASELINE eval on cuda $GPU — DS-MVTec then VisA"
DSJ="$OUT/eval_dsmvtec_full_trainprompt.json"
VSJ="$OUT/eval_visa_full_trainprompt.json"

if [ ! -f "$DSJ" ]; then
  echo "[$(date +%T)] === DS-MVTec (1670) ==="
  CUDA_VISIBLE_DEVICES=$GPU python "$EVAL" --base-model "$BASE" --ds-mvtec-only --num-samples 0 \
      --batch-size 6 --output "$DSJ"
fi
echo "[$(date +%T)] DS-MVTec BA: $(python3 "$BA" "$DSJ" 2>/dev/null | grep -oE 'BA=[0-9.]+')"

if [ ! -f "$VSJ" ]; then
  echo "[$(date +%T)] === VisA (2141) ==="
  CUDA_VISIBLE_DEVICES=$GPU python "$EVAL" --base-model "$BASE" --visa-only --num-samples 0 \
      --batch-size 6 --output "$VSJ"
fi
echo "[$(date +%T)] VisA BA: $(python3 "$BA" "$VSJ" 2>/dev/null | grep -oE 'BA=[0-9.]+')"

echo "[$(date '+%F %T')] DONE. Qwen3-VL-8B base:  DS $(python3 "$BA" "$DSJ" 2>/dev/null | grep -oE 'BA=[0-9.]+')  VisA $(python3 "$BA" "$VSJ" 2>/dev/null | grep -oE 'BA=[0-9.]+')  (vs Qwen2.5-VL-7B base 69.01/53.80)"
touch "$OUT/eval.done"
