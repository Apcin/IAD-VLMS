#!/bin/bash
# Persistent RAM guard (runs in tmux, server-side). Keeps the SFT (CUDA 0,3 procs)
# pinned as the OOM victim so the GRPO resume on CUDA 1,2 is never the one killed,
# and proactively kills the SFT if available RAM gets critically low (<12G) to
# avoid a hard kernel OOM. Exits when SFT finishes or dies.
KILL_GB=12
DONE=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_variety_star_6k/train.done
while true; do
  sftpids=""
  for p in $(pgrep -f "llamafactory|torchrun|python" 2>/dev/null); do
    cvd=$(tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep -m1 '^CUDA_VISIBLE_DEVICES=')
    [ "$cvd" = "CUDA_VISIBLE_DEVICES=0,3" ] && { echo 1000 >/proc/$p/oom_score_adj 2>/dev/null; sftpids="$sftpids $p"; }
  done
  avail=$(free -g | awk '/Mem/{print $7}')
  echo "$(date +%T) avail=${avail}G sft_procs=$(echo $sftpids|wc -w)"
  [ -f "$DONE" ] && { echo "SFT done flag — guard exit"; break; }
  [ -z "$sftpids" ] && { echo "no SFT procs — guard exit"; break; }
  if [ "${avail:-99}" -lt "$KILL_GB" ]; then
    echo "!!! avail<${KILL_GB}G -> killing SFT to protect GRPO:$sftpids"
    tmux kill-session -t variety_sft_live 2>/dev/null
    for p in $sftpids; do kill -9 $p 2>/dev/null; done
    echo "SFT KILLED"; break
  fi
  sleep 15
done
