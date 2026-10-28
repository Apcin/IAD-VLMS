#!/bin/bash
# memorisation check: score a checkpoint on the images it was trained on (run 1, epoch 3)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} HF_HOME=${WORK_DIR}/hf_cache
PY=${WORK_DIR}/envs/grpo_fast312/bin/python
HERE=${WORK_DIR}/mmad_leak_experiment
CK=${WORK_DIR}/outputs/sft_qwen25vl_7b_mmad_train1600/checkpoint-150_vllmcompat
mkdir -p $HERE/evals/checkpoint-150
for s in DS-MVTec VisA GoodsAD MVTec-LOCO; do
  f=$HERE/evals/checkpoint-150/eval_${s}_trainkeys.json
  [ -s $f ] && continue
  echo "=== $(date +%H:%M) train-keys eval ckpt-150 on $s ==="
  $PY $HERE/eval_heldout_vllm.py --checkpoint $CK --subset $s --keys train --output $f 2>&1 | grep -E "vllm-eval|Traceback|entries" | tail -3
done
echo "TRAINKEYS DONE $(date)"
