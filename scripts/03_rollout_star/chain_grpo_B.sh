#!/bin/bash
# Sequential chain: wait for Job A (beta=0.1) to finish TRAINING (.traindone),
# then launch Job B (beta=0.04) on the now-free NV pair {0,3}. Two full-offload
# jobs can't coexist (4x113GB > 503GB host RAM), so they run back-to-back.
# B trains on {0,3} while A's eval runs on {1,2} — no conflict.
set -uo pipefail
A_DONE=/bulk/aacudad/reasoning_traces/outputs/grpo_sftprompt_kl0.1/.traindone
LOG=/bulk/aacudad/reasoning_traces/Training/chain_grpo_B.log
echo "[$(date +%T)] chain armed; waiting for Job A .traindone" | tee -a "$LOG"
while [ ! -f "$A_DONE" ]; do sleep 300; done
echo "[$(date +%T)] Job A training done -> launching Job B (beta=0.04) on {0,3}" | tee -a "$LOG"
# confirm 0,3 free
while :; do
  u0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
  u3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 2>/dev/null)
  [ "${u0:-9999}" -lt 3000 ] && [ "${u3:-9999}" -lt 3000 ] && break
  sleep 30
done
BETA=0.04 GPUS=0,3 PORT=29542 OUTNAME=grpo_sftprompt_kl0.04 EVAL_DS_GPU=0 EVAL_VISA_GPU=3 \
  bash /bulk/aacudad/reasoning_traces/Training/run_grpo_sftprompt.sh 2>&1 | tee -a "$LOG"
echo "[$(date +%T)] Job B chain complete." | tee -a "$LOG"
