#!/bin/bash
# Auto-eval the resumed GRPO-on-C checkpoints once GRPO finishes.
# DS-MVTec on CUDA 1, VisA on CUDA 2 (parallel), full subsets, GRPO eval prompt
# (matches how ckpt-265/530/1060 were evaluated). Model-only checkpoints, so the
# Arm-C ckpt-376 supplies the processor/config via --base-model.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
WORK=/bulk/aacudad/reasoning_traces
TR=$WORK/Training
GRPO=$WORK/outputs/grpo_qwen25vl_7b_abc_C_grpo
BASE=$WORK/outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376
export HF_HOME=$WORK/hf_cache TRANSFORMERS_CACHE=$WORK/hf_cache
export PROBE_SAMPLES=0   # full subsets (1670 DS / 2141 VisA), no truncation
LOGD=$WORK/logs/grpo_eval_$(date +%Y%m%d_%H%M%S); mkdir -p "$LOGD"

# 1) wait for GRPO to finish and release CUDA 1,2
echo "[grpo-eval-wd $(date +%T)] waiting for GRPO done (train_resume.done / ckpt-2120) ..."
while [ ! -f "$GRPO/train_resume.done" ] && [ ! -d "$GRPO/checkpoint-2120" ]; do sleep 120; done
echo "[grpo-eval-wd $(date +%T)] GRPO end detected; waiting for grpo_ad.py to exit (free CUDA 1,2) ..."
while pgrep -f grpo_ad.py >/dev/null 2>&1; do sleep 60; done
echo "[grpo-eval-wd $(date +%T)] GRPO gone — evaluating new checkpoints"

EVGPU=3   # only free GPU (GPU0=other user, 1+2=Variety SFT); run ds then visa sequentially
eval_ckpt () {  # $1 = checkpoint dir name
  local name="$1"; local ck="$GRPO/$name"
  [ -d "$ck" ] || { echo "  $name: missing, skip"; return; }
  local did=0
  if [ ! -f "$ck/eval_dsmvtec_full_trainprompt.json" ]; then
    CUDA_VISIBLE_DEVICES=$EVGPU python "$TR/evaluate_qwen25vl_7b_trainprompt.py" \
      --checkpoint "$ck/" --base-model "$BASE" --ds-mvtec-only --grpo-eval \
      --output "$ck/eval_dsmvtec_full_trainprompt.json" > "$LOGD/${name}_dsmvtec.log" 2>&1
    did=1
  fi
  if [ ! -f "$ck/eval_visa_full_trainprompt.json" ]; then
    CUDA_VISIBLE_DEVICES=$EVGPU python "$TR/evaluate_qwen25vl_7b_trainprompt.py" \
      --checkpoint "$ck/" --base-model "$BASE" --visa-only --grpo-eval \
      --output "$ck/eval_visa_full_trainprompt.json" > "$LOGD/${name}_visa.log" 2>&1
    did=1
  fi
  [ "$did" -eq 0 ] && { echo "  $name: already evaluated, skip"; return; }
  echo "[grpo-eval-wd $(date +%T)] $name evaluated"
}

for c in checkpoint-1325 checkpoint-1590 checkpoint-1855 checkpoint-2120; do
  eval_ckpt "$c"
done
echo "[grpo-eval-wd $(date)] ALL GRPO EVALS DONE"
touch "$GRPO/grpo_eval.done"
