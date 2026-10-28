#!/bin/bash
# Targeted-Variety+Arm-C-rehearsal continuation-SFT from the 82.80% Arm-C checkpoint, CUDA 0+3.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
cd /bulk/aacudad/reasoning_traces
export CUDA_VISIBLE_DEVICES=0,3
export FORCE_TORCHRUN=1
OUT=outputs/sft_varmix_from_armC
mkdir -p "$OUT"
echo "[$(date)] launching varmix continuation-SFT from Arm-C (82.8%) on GPUs 0,3 -> $OUT"
( llamafactory-cli train Training/sft_varmix_from_armC.yaml && touch "$OUT/.traindone" ) 2>&1 | tee "$OUT/train.log"
echo "[$(date)] armC-varmix training exited (rc=${PIPESTATUS[0]})"
