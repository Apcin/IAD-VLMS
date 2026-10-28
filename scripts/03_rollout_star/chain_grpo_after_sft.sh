#!/bin/bash
# Waits for the filtered-6K SFT (train + ALL checkpoint evals) to finish, then
# fires the 0.5-epoch KL-penalty GRPO drift test on the now-free CUDA 1,2.
# The SFT launcher touches .evaldone after the last checkpoint eval completes.
set -uo pipefail

SFT_DONE=/bulk/aacudad/reasoning_traces/outputs/sft_filtered6kcc_from_base/.evaldone
GRPO=/bulk/aacudad/reasoning_traces/Training/run_grpo_kl_halfep.sh
LOG=/bulk/aacudad/reasoning_traces/Training/chain_grpo.log

echo "[$(date +%T)] chain watcher armed; waiting for $SFT_DONE" | tee -a "$LOG"

# 1) wait for SFT train+eval to complete
while [ ! -f "$SFT_DONE" ]; do sleep 60; done
echo "[$(date +%T)] SFT .evaldone seen." | tee -a "$LOG"

# 2) wait until CUDA 1 and 2 are actually free (<3 GB used), to be safe
while :; do
  u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null)
  u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 2>/dev/null)
  echo "[$(date +%T)] cuda1=${u1}MiB cuda2=${u2}MiB" | tee -a "$LOG"
  if [ "${u1:-9999}" -lt 3000 ] && [ "${u2:-9999}" -lt 3000 ]; then break; fi
  sleep 30
done
echo "[$(date +%T)] CUDA 1,2 free -> launching GRPO KL drift test." | tee -a "$LOG"

# 3) verify embed server :5200 is up before launching (reward dependency)
if ! python3 -c "import urllib.request,json,sys; r=urllib.request.Request('http://127.0.0.1:5200/embed_similarity',data=json.dumps({'text1':'scratch','text2':'scratch'}).encode(),headers={'Content-Type':'application/json'}); urllib.request.urlopen(r,timeout=10)" 2>/dev/null; then
  echo "[$(date +%T)] WARNING: embed server :5200 not responding — type_reward will fall back to lexical." | tee -a "$LOG"
fi

# 4) fire
bash "$GRPO" 2>&1 | tee -a "$LOG"
echo "[$(date +%T)] GRPO chain complete." | tee -a "$LOG"
