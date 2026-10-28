#!/bin/bash
# Evaluate published external baselines on OUR protocol, sequentially on one GPU:
#   JUDO (ICLR 2026, Qwen2.5-VL)  ->  DS-MVTec, then VisA
#
# IMPORTANT -- what these numbers mean (and do not mean):
#   JUDO's NATIVE setting is TWO images (query + a NORMAL template) with an MMAD multiple-choice
#   question, emitting <seg>/<think>/<answer>A|B|C|D</answer>. Here we run it under OUR protocol:
#   ZERO-SHOT, SINGLE image, binary yes/no, balanced accuracy. Its <answer>A/B</answer> parses via
#   normalize_answer (a->yes, b->no), so the harness reads it fine -- but this is NOT JUDO's
#   published setting and will UNDERSTATE it (it is denied the normal reference image it was
#   trained to compare against). Report it as "JUDO, single-image zero-shot (non-native)",
#   never as JUDO's paper number (81.20% MMAD 7-task MCQ average -- a different task and metric).
#   grpoprompt is used because JUDO is GRPO-trained, matching the IAD-R1 re-canon convention.
#
# EMIT is NOT here: it is CustomizedInternVLChatModel (InternVL3-8B, custom remote code) and cannot
# run through this Qwen-based harness without an adapter.
#
# Run inside tmux:  tmux new -s eval_baselines 'bash scripts/04_eval/run_eval_baselines.sh'
set -u

# Paths. WORK_DIR is the workspace holding outputs/ and hf_cache/; it defaults to the
# parent of this repository. On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
EVAL_SCRIPT="$HERE/evaluate_qwen25vl_7b_trainprompt.py"
EVAL_GPU=1
BATCH_SIZE=4

# JUDO weights are not redistributed here; download them and point JUDO_CKPT at your copy.
JUDO_CKPT="${JUDO_CKPT:-$WORK_DIR/JUDO-weights}"
JUDO_OUT="${JUDO_OUT:-$WORK_DIR/outputs/judo_eval}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
export HF_HOME=$WORK_DIR/hf_cache
export TRANSFORMERS_CACHE=$WORK_DIR/hf_cache
cd "$WORK_DIR"

mkdir -p "$JUDO_OUT"

run_eval () {  # $1=name  $2=ckpt  $3=outdir  $4=bench_flag  $5=outfile
  local out="$3/$5"
  if [[ -f "$out" ]]; then
    echo "[baselines] $1 $4 already done -> skip"
    return
  fi
  echo "[baselines] === $1 : $4 on CUDA $EVAL_GPU ==="
  CUDA_VISIBLE_DEVICES=$EVAL_GPU python "$EVAL_SCRIPT" \
      --checkpoint "$2" \
      "$4" \
      --batch-size $BATCH_SIZE \
      --grpo-eval \
      --output "$out"
  echo "[baselines] $1 $4 DONE -> $out"
}

echo "[baselines] start $(date)"
run_eval "JUDO" "$JUDO_CKPT" "$JUDO_OUT" "--ds-mvtec-only" "eval_dsmvtec_full_grpoprompt.json"
run_eval "JUDO" "$JUDO_CKPT" "$JUDO_OUT" "--visa-only"     "eval_visa_full_grpoprompt.json"
echo "[baselines] ALL DONE $(date)"

python - "$JUDO_OUT" <<'PY'
import json, os, sys
ba = lambda m: 0.5*(m['tp']/(m['tp']+m['fn']) + m['tn']/(m['tn']+m['fp']))
d=sys.argv[1]
for b,f in [("DS-MVTec","eval_dsmvtec_full_grpoprompt.json"),("VisA","eval_visa_full_grpoprompt.json")]:
    p=os.path.join(d,f)
    if os.path.exists(p):
        m=json.load(open(p))["metrics"]
        print(f"JUDO (single-image zero-shot, non-native) {b}: BA={ba(m)*100:.2f} "
              f"recall={m['tp']/(m['tp']+m['fn'])*100:.1f}% spec={m['tn']/(m['tn']+m['fp'])*100:.1f}% n={m['total']}")
PY
