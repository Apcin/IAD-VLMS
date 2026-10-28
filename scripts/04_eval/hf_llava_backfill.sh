#!/bin/bash
# Paths below assume the TU Delft workspace /bulk/aacudad/reasoning_traces (the parent of this repo). Replace with your WORK_DIR.
# HF-path (HuggingFace generate, greedy, 1024 tokens) LLaVA-OneVision evals, same recipe as night_worker.sh llava_hf.
# usage: hf_llava_backfill.sh <GPU> <ckpt-rel-path> [<ckpt-rel-path> ...]
GPU=$1; shift; R=/bulk/aacudad/reasoning_traces; EVAL=$R/Training/evaluate_qwen25vl_7b_trainprompt.py
PY=/users/aacudad/miniconda3/envs/iad_r1_sft/bin/python; BASE=llava-hf/llava-onevision-qwen2-7b-si-hf; LOG=$R/logs/hf_llava_backfill_gpu$GPU.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
for rel in "$@"; do ck=$R/outputs/$rel; dsj=$ck/eval_dsmvtec_full_trainprompt.json; vsj=$ck/eval_visa_full_trainprompt.json; t0=$(date +%s)
  [ -s "$dsj" ] || { log "START $rel DS-MVTec"; env EVAL_MAX_IMAGE_PIXELS=262144 CUDA_VISIBLE_DEVICES=$GPU $PY $EVAL --checkpoint "$ck" --base-model $BASE --ds-mvtec-only --output "$dsj" > $ck/eval_ds_hf_backfill.log 2>&1 || log "FAILED $rel DS-MVTec"; }
  [ -s "$vsj" ] || { log "START $rel VisA";     env EVAL_MAX_IMAGE_PIXELS=262144 CUDA_VISIBLE_DEVICES=$GPU $PY $EVAL --checkpoint "$ck" --base-model $BASE --visa-only     --output "$vsj" > $ck/eval_visa_hf_backfill.log 2>&1 || log "FAILED $rel VisA"; }
  log "DONE $rel [$(( ($(date +%s)-t0)/60 )) min]"
done; log "worker gpu$GPU finished"
