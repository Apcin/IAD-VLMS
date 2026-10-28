#!/bin/bash
# Re-evaluate ALL compared LLaVA-OV checkpoints with the vLLM backend on ONE GPU,
# exact watcher protocol (trainprompt default mode, EVAL_MAX_IMAGE_PIXELS=262144,
# greedy, 1024 max tokens). HF result files are left untouched; vLLM results get
# the *_vllm.json suffix. Order: iter2 ckpt-748 first (restarted eval), then the
# rest, then the base model in both compared prompt modes.
# Usage: run_vllm_reevals_cuda3.sh [gpu]   (default 3)
set -uo pipefail
GPU="${1:-3}"
# Paths. WORK_DIR is the workspace holding outputs/, hf_cache/ and the python envs;
# it defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
# Python 3.12 env with vLLM. Override with VLLM_PYTHON.
PY="${VLLM_PYTHON:-$WORK_DIR/envs/grpo_fast312/bin/python}"
EV=$HERE/evaluate_vllm_llava.py
BASE=llava-hf/llava-onevision-qwen2-7b-si-hf
export HF_HOME=$WORK_DIR/hf_cache
LOG=$WORK_DIR/outputs/vllm_reevals.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

bal(){ $PY - "$1" <<'EOF'
import json,sys
try:
    m=json.load(open(sys.argv[1]))['metrics']; tp,tn,fp,fn=m['tp'],m['tn'],m['fp'],m['fn']
    print(f"{50*(tp/((tp+fn) or 1)+tn/((tn+fp) or 1)):.2f}")
except Exception: print("nan")
EOF
}

eval_ckpt(){
  local CKD="$1" TAG="$2"
  log "vllm-eval $TAG (DS + VisA, one engine)..."
  CUDA_VISIBLE_DEVICES=$GPU $PY "$EV" --checkpoint "$CKD" --base-model "$BASE" \
    --out-ds   "$CKD/eval_dsmvtec_full_trainprompt_vllm.json" \
    --out-visa "$CKD/eval_visa_full_trainprompt_vllm.json" \
    > "$CKD/eval_vllm.log" 2>&1
  local rc=$?
  local ds va
  ds=$(bal "$CKD/eval_dsmvtec_full_trainprompt_vllm.json")
  va=$(bal "$CKD/eval_visa_full_trainprompt_vllm.json")
  log "RESULT(vllm) $TAG: DS-MVTec $ds | VisA $va (balanced acc) [rc=$rc]"
}

I2=$WORK_DIR/outputs/sft_llava_ov_7b_frozen_iad_sft_iter2
K6=$WORK_DIR/outputs/sft_llava_ov_7b_frozen_iad_sft_6k_train

log "════ vLLM re-evals on cuda$GPU start ════"
eval_ckpt "$I2/checkpoint-748" "iter2/ckpt-748 (epoch 4)"
eval_ckpt "$I2/checkpoint-188" "iter2/ckpt-188 (epoch 1)"
eval_ckpt "$I2/checkpoint-376" "iter2/ckpt-376 (epoch 2)"
eval_ckpt "$I2/checkpoint-564" "iter2/ckpt-564 (epoch 3)"
eval_ckpt "$K6/checkpoint-188" "6k/ckpt-188 (epoch 1)"
eval_ckpt "$K6/checkpoint-376" "6k/ckpt-376 (epoch 2)"
eval_ckpt "$K6/checkpoint-564" "6k/ckpt-564 (epoch 3)"
eval_ckpt "$K6/checkpoint-752" "6k/ckpt-752 (epoch 4)"

BOUT=$WORK_DIR/outputs/llava_ov_7b_zeroshot_eval
log "vllm-eval base zero-shot, trainprompt mode..."
CUDA_VISIBLE_DEVICES=$GPU $PY "$EV" --base-model "$BASE" \
  --out-ds   "$BOUT/eval_dsmvtec_full_trainprompt_vllm.json" \
  --out-visa "$BOUT/eval_visa_full_trainprompt_vllm.json" \
  > "$BOUT/eval_vllm_trainprompt.log" 2>&1
log "RESULT(vllm) base trainprompt: DS-MVTec $(bal "$BOUT/eval_dsmvtec_full_trainprompt_vllm.json") | VisA $(bal "$BOUT/eval_visa_full_trainprompt_vllm.json")"

log "vllm-eval base zero-shot, yesno-user mode..."
CUDA_VISIBLE_DEVICES=$GPU $PY "$EV" --base-model "$BASE" --yesno-user \
  --out-ds   "$BOUT/eval_dsmvtec_full_yesnouser_vllm.json" \
  --out-visa "$BOUT/eval_visa_full_yesnouser_vllm.json" \
  > "$BOUT/eval_vllm_yesnouser.log" 2>&1
log "RESULT(vllm) base yesno-user: DS-MVTec $(bal "$BOUT/eval_dsmvtec_full_yesnouser_vllm.json") | VisA $(bal "$BOUT/eval_visa_full_yesnouser_vllm.json")"

log "════ ALL vLLM RE-EVALS DONE ════"
grep 'RESULT(vllm)' "$LOG"
