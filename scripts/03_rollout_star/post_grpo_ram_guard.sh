#!/bin/bash
# Post-GRPO RAM guard: protect the long SFT (CUDA 0,3) by making the lighter,
# resumable GRPO-eval the OOM victim. Pins eval procs oom_score_adj=1000; if RAM
# gets critical, kills the eval (it resumes later, skipping done checkpoints).
KILL_GB=12
SFT_DONE=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_variety_star_6k/train.done
EVAL_DONE=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_abc_C_grpo/grpo_eval.done
while true; do
  evalpids=""
  for p in $(pgrep -f evaluate_qwen25vl_7b_trainprompt 2>/dev/null); do
    echo 1000 > /proc/$p/oom_score_adj 2>/dev/null; evalpids="$evalpids $p"
  done
  avail=$(free -g | awk '/Mem/{print $7}')
  echo "$(date +%T) avail=${avail}G eval_procs=$(echo $evalpids|wc -w)"
  [ -f "$SFT_DONE" ] && [ -f "$EVAL_DONE" ] && { echo "SFT+eval done — guard exit"; break; }
  if [ "${avail:-99}" -lt "$KILL_GB" ] && [ -n "$evalpids" ]; then
    echo "!!! avail<${KILL_GB}G -> killing GRPO-eval to protect SFT:$evalpids"
    tmux kill-session -t grpo_eval 2>/dev/null
    for p in $evalpids; do kill -9 $p 2>/dev/null; done
    echo "EVAL KILLED (resumable: re-run skips done checkpoints)"
  fi
  sleep 20
done
