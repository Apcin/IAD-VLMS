#!/bin/bash
# Targeted-Variety+Arm-C-rehearsal continuation-SFT from the 82.73% GRPO checkpoint, CUDA 1+2.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
cd /bulk/aacudad/reasoning_traces
export CUDA_VISIBLE_DEVICES=1,2
export FORCE_TORCHRUN=1
OUT=outputs/sft_varmix_from_grpo
mkdir -p "$OUT"
echo "[$(date)] launching varmix continuation-SFT from GRPO (82.7%) on GPUs 1,2 -> $OUT"
( llamafactory-cli train Training/sft_varmix_from_grpo.yaml && touch "$OUT/.traindone" ) 2>&1 | tee "$OUT/train.log"
echo "[$(date)] grpo-varmix training exited (rc=${PIPESTATUS[0]})"
