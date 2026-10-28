#!/bin/bash
# Overnight scheduler: wait until the GRPO 3-epoch trainer AND its cuda-3 eval watcher have
# finished AND cuda 0 + cuda 3 are free, then run a 2-step smoke pre-flight, and ONLY if the
# smoke passes, launch the full Qwen3-VL-8B Arm-C 4-epoch SFT on cuda 0,3.
# Launch it in tmux:  tmux new -s q3_sched 'bash scripts/01_sft/schedule_qwen3vl_after_grpo.sh'
set -uo pipefail
# Paths. WORK_DIR is the workspace holding LlamaFactory/, outputs/ and hf_cache/; it
# defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
WORK="$WORK_DIR"
LOG="$WORK/logs/qwen3vl_scheduler.log"
# The YAML carries ${WORK_DIR} / ${REPO_ROOT} placeholders; expand them once here.
YAML=$(mktemp -t sft_qwen3vl_8b_armC.yaml.XXXX)
sed -e "s|\${WORK_DIR}|$WORK_DIR|g" -e "s|\${REPO_ROOT}|$REPO_ROOT|g" \
    "$REPO_ROOT/configs/sft/sft_qwen3vl_8b_armC.yaml" > "$YAML"
mkdir -p "$WORK/logs"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
gpu_mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

log "scheduler armed (PID $$); waiting for the GRPO trainer to finish + cuda 1,2 to free (the cuda-3 watcher finishes its tail evals independently)"
stable=0
while :; do
  grpo=$(pgrep -fc "grpo_ad.py" || true)
  m1=$(gpu_mem 1); m2=$(gpu_mem 2)
  if [ "${grpo:-0}" -eq 0 ] && [ "${m1:-99999}" -lt 4000 ] && [ "${m2:-99999}" -lt 4000 ]; then
    stable=$((stable+1)); log "clear check $stable/2 (grpo=$grpo cuda1=${m1}MiB cuda2=${m2}MiB)"
  else
    [ "$stable" -ne 0 ] && log "reset (grpo=$grpo cuda1=${m1}MiB cuda2=${m2}MiB)"; stable=0
  fi
  [ "$stable" -ge 2 ] && break
  sleep 300
done
log "GO: GRPO trainer done, cuda1=$(gpu_mem 1)MiB cuda2=$(gpu_mem 2)MiB."

export CUDA_VISIBLE_DEVICES=1,2
export FORCE_TORCHRUN=1
export HF_HOME=$WORK/hf_cache
export HF_HUB_CACHE=$WORK/hf_cache/hub
export TRANSFORMERS_CACHE=$WORK/hf_cache
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_$USER}"
mkdir -p "$TRITON_CACHE_DIR" "$WORK/hf_cache"
cd "$WORK/LlamaFactory"

SMOKE_OUT="$WORK/outputs/_smoke_qwen3vl"; rm -rf "$SMOKE_OUT"
log "SMOKE pre-flight: 2 steps (real batch), no save -> logs/qwen3vl_smoke.log"
conda run -n llama_sft --no-capture-output llamafactory-cli train "$YAML" \
    max_steps=2 save_strategy=no output_dir="$SMOKE_OUT" overwrite_output_dir=true \
    > "$WORK/logs/qwen3vl_smoke.log" 2>&1
rc=$?
if [ $rc -ne 0 ] || ! grep -qiE "'loss'|loss=|train_runtime" "$WORK/logs/qwen3vl_smoke.log"; then
  log "SMOKE FAILED (rc=$rc) — NOT launching the full run. Inspect logs/qwen3vl_smoke.log"
  exit 1
fi
log "SMOKE PASSED. Arming the cuda-3 eval watcher, then launching the full 4-epoch Qwen3-VL-8B Arm-C SFT on cuda 1,2."
nohup bash "$REPO_ROOT/scripts/04_eval/watch_and_eval_qwen3vl_cuda3.sh" "$WORK/outputs/sft_qwen3vl_8b_armC" 3 >> "$WORK/logs/qwen3vl_eval_watcher.log" 2>&1 &
log "eval watcher armed (PID $!) on cuda 3 -> logs/qwen3vl_eval_watcher.log (DS then VisA per checkpoint, sequential)"
bash "$REPO_ROOT/scripts/01_sft/run_qwen3vl_8b_armC.sh" 1,2 >> "$LOG" 2>&1
log "Qwen3-VL-8B Arm-C SFT training finished; eval watcher will finish the last checkpoint then exit."
