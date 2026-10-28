#!/bin/bash
# Auto-SFT watchdog for the Variety STaR pipeline.
# Waits for run_variety_star.sh to finish (done.flag + the exactly-6K corpus),
# stages the corpus at the registered LlamaFactory path, waits for the CUDA 0+3
# NVLink pair to be free (the rollout releases it well before the pipeline ends),
# then launches the SFT (frozen ViT, 4 epochs, from base) on CUDA 0+3.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

WORK=/bulk/aacudad/reasoning_traces
TR=$WORK/Training
GPUS=1,2   # GPU0 is held by another user (augustasjaruse); use the free 1,2 NVLink pair
REG=$TR/datasets_variety/variety_star_sft_6k.json   # registered in dataset_info.json
mkdir -p "$TR/datasets_variety"

# 1) wait for the STaR pipeline to finish and the 6K corpus to exist
echo "[sft-wd $(date +%T)] waiting for variety_star done.flag + 6K corpus ..."
while true; do
  D=$(cat "$TR/.last_variety_star_dir" 2>/dev/null || true)
  if [ -n "$D" ] && [ -f "$D/done.flag" ] && [ -f "$D/variety_star_sft_6k.json" ]; then break; fi
  sleep 120
done
echo "[sft-wd $(date +%T)] STaR pipeline complete: $D"

# 2) stage the corpus at the registered path
cp "$D/variety_star_sft_6k.json" "$REG"
N=$(python3 -c "import json;print(len(json.load(open('$REG'))))")
echo "[sft-wd $(date +%T)] staged 6K corpus ($N items) -> $REG"

# 3) wait for CUDA 0+3 free AND enough host RAM. The SFT uses ZeRO-3 CPU offload;
#    running it alongside the GRPO resume (also ZeRO-3 offload, ~228GB) risks the
#    host-RAM OOM that killed earlier concurrent offload jobs. Requiring >=260GB
#    free effectively waits for GRPO on CUDA 1,2 to finish before launching.
MIN_FREE_GB=260
echo "[sft-wd $(date +%T)] waiting for CUDA $GPUS free AND >=${MIN_FREE_GB}GB host RAM (avoids offload OOM) ..."
while true; do
  mx=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$1==1||$1==2{print $2}' | sort -rn | head -1)
  freeg=$(free -g | awk '/Mem/{print $7}')
  if [ "${mx:-999999}" -lt 5000 ] && [ "${freeg:-0}" -ge "$MIN_FREE_GB" ]; then break; fi
  sleep 120
done
echo "[sft-wd $(date +%T)] resources free (gpu max ${mx} MiB, host free ${freeg}GB) — launching SFT"

# 4) launch SFT (mirrors run_abc_train.sh C)
LOG_DIR="$WORK/logs/variety_sft_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=$GPUS
export FORCE_TORCHRUN=1
export HF_HOME=$WORK/hf_cache
export TRANSFORMERS_CACHE=$WORK/hf_cache
export TRITON_CACHE_DIR=/tmp/triton_cache_aacudad
mkdir -p /tmp/triton_cache_aacudad

echo "════════════════════════════════════════════════════════════════════════"
echo " VARIETY STaR SFT — 7B, frozen ViT, 4 epochs, from base   CUDA $GPUS"
echo " Corpus:  $REG ($N items)"
echo " Start:   $(date)"
echo "════════════════════════════════════════════════════════════════════════"
cd "$WORK/LlamaFactory"
llamafactory-cli train "$TR/sft_variety_star.yaml" 2>&1 | tee "$LOG_DIR/train.log"

mkdir -p "$WORK/outputs/sft_qwen25vl_7b_variety_star_6k"
touch "$WORK/outputs/sft_qwen25vl_7b_variety_star_6k/train.done"
echo "[sft-wd $(date)] VARIETY STaR SFT DONE."
