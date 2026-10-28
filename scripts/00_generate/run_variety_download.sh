#!/bin/bash
# Sharded download of ALL 160 Real-IAD Variety categories, C1 view only.
# 4 parallel shards, each with its OWN _zips temp dir (no race). Resumable:
# categories already extracted are skipped. Writes DL_DONE.flag when complete.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
cd /bulk/aacudad/reasoning_traces/reasoning_traces_gen
export GEMINI_API_KEYS=dummy   # satisfies config_mmad import gate (Vertex used for gen, not this)

LOGDIR=variety_logs; mkdir -p "$LOGDIR"
rm -f DL_DONE.flag

# Full category list from the metadata jsons
mapfile -t CATS < <(ls data/realiad-variety/Real-IAD_Variety_jsons/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' | sort)
N=${#CATS[@]}
SHARDS=4
echo "[$(date)] Downloading $N categories (C1 only) across $SHARDS shards"

pids=()
for s in $(seq 0 $((SHARDS-1))); do
  sub=()
  for ((i=s; i<N; i+=SHARDS)); do sub+=("${CATS[$i]}"); done
  echo "[dl-shard $s] ${#sub[@]} categories -> $LOGDIR/dl_shard${s}.log"
  VARIETY_ZIP_TMP="data/realiad-variety/_zips_shard${s}" \
    python3 download_variety.py --categories "${sub[@]}" --views C1 \
    > "$LOGDIR/dl_shard${s}.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

# tidy per-shard temp dirs
rm -rf data/realiad-variety/_zips_shard* 2>/dev/null

kept=$(find data/realiad-variety -name "*_C1_*.png" ! -name "*_mask*" 2>/dev/null | wc -l)
echo "[$(date)] DOWNLOAD COMPLETE — C1 images on disk: $kept  (shard failures: $fail)"
touch DL_DONE.flag
