#!/bin/bash
# Usage: bash eval_ckpts_on_gpu.sh <gpu> <mem_util> <run_dir> <split.json> <evals_dir> <ckpt_num> [<ckpt_num> ...]
GPU=$1; MEM=$2; RUN=$3; SPLIT=$4; EVALS=$5; shift 5
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=${WORK_DIR}/hf_cache MMAD_SPLIT=$SPLIT
PY=${WORK_DIR}/envs/grpo_fast312/bin/python
HERE=${WORK_DIR}/mmad_leak_experiment
SRC=${WORK_DIR}/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530_vllmcompat
for n in "$@"; do
  ck=$RUN/checkpoint-$n; cd_=$HERE/compat/$(basename $RUN)/checkpoint-$n
  [ -s $ck/model.safetensors ] || { echo "no weights for $ck"; continue; }
  if [ ! -s $cd_/config.json ]; then mkdir -p $cd_; for f in chat_template.json config.json generation_config.json merges.txt preprocessor_config.json tokenizer_config.json tokenizer.json vocab.json; do cp $SRC/$f $cd_/; done; ln -sfn $ck/model.safetensors $cd_/model.safetensors; fi
  mkdir -p $EVALS/checkpoint-$n
  for s in DS-MVTec VisA GoodsAD MVTec-LOCO; do
    f=$EVALS/checkpoint-$n/eval_${s}_heldout.json
    [ -s $f ] && { echo "skip $f"; continue; }
    echo "=== $(date +%H:%M) gpu$GPU eval ckpt-$n on $s ==="
    $PY $HERE/eval_heldout_vllm.py --checkpoint $cd_ --subset $s --keys test --output $f --gpu-mem-util $MEM 2>&1 | grep -E "vllm-eval|Traceback|entries" | tail -3
  done
done
echo "GPU$GPU HELPER DONE $(date)"
