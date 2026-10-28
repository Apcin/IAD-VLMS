#!/bin/bash
# Sharded Real-IAD Variety trace generation (Vertex / Gemini-3-Flash, thinking MINIMAL).
# Waits for the download to finish, runs 8 generation shards in parallel, then
# (ONLY at the end) validates format+location and retries the failures up to
# MAX_RETRY_PASSES times by purging invalid traces and regenerating them.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
cd /bulk/aacudad/reasoning_traces/reasoning_traces_gen
export GEMINI_API_KEYS=dummy   # config_mmad import gate; Vertex creds come from the environment
# Needs GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT set, see .env.example

LOGDIR=variety_logs; mkdir -p "$LOGDIR"
SHARDS=8
MAX_RETRY_PASSES=3

# 1) wait for the download to complete
echo "[$(date)] waiting for DL_DONE.flag ..."
while [ ! -f DL_DONE.flag ]; do sleep 60; done
echo "[$(date)] download done — starting generation ($SHARDS shards)"

run_all_shards () {
  local tag="$1"
  local pids=()
  for s in $(seq 0 $((SHARDS-1))); do
    python3 generate_variety_traces.py --shard_id "$s" --total_shards "$SHARDS" \
      > "$LOGDIR/gen_${tag}_shard${s}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

# 2) main generation pass
run_all_shards "main"
echo "[$(date)] main generation pass complete"

# 3) end-of-run validation + retry (purge invalid -> regenerate)
for pass in $(seq 1 $MAX_RETRY_PASSES); do
  if python3 validate_variety_traces.py --purge > "$LOGDIR/validate_pass${pass}.log" 2>&1; then
    echo "[$(date)] validation pass $pass: all traces valid — done"
    break
  fi
  echo "[$(date)] validation pass $pass: invalid traces purged, regenerating ..."
  tail -3 "$LOGDIR/validate_pass${pass}.log"
  run_all_shards "retry${pass}"
done

# 4) final report
python3 validate_variety_traces.py --report > "$LOGDIR/validate_final.log" 2>&1 || true
echo "[$(date)] FINAL:"; tail -3 "$LOGDIR/validate_final.log"
echo "[$(date)] GENERATION PIPELINE DONE"
touch GEN_DONE.flag
