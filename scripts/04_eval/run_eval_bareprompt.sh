#!/bin/bash
# Eval the SFT-bareprompt and GRPO-bareprompt checkpoints.
# Both DS-MVTec and VisA. Two lanes (GPU 0 + GPU 3, parallel).
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft

SFT_DIR=/bulk/aacudad/reasoning_traces/outputs/sft_qwen25vl_7b_bareprompt_frozen
GRPO_DIR=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_bareprompt_full

SFT_CKPTS=($(ls -1d $SFT_DIR/checkpoint-*/ 2>/dev/null | sed 's:/$::'))
GRPO_CKPTS=($(ls -1d $GRPO_DIR/checkpoint-*/ 2>/dev/null | sed 's:/$::'))

# Spec format: ckpt|flag|suffix|dataset_flag
# SFT-bare evaluated with --bare-question (matches training prompt exactly)
# GRPO evaluated with --grpo-eval (matches GRPO training prompt)
: > /tmp/bareprompt_specs_g0.txt
for ckpt in "${SFT_CKPTS[@]}"; do
    echo "${ckpt}|--bare-question|bareprompt|--ds-mvtec-only" >> /tmp/bareprompt_specs_g0.txt
    echo "${ckpt}|--bare-question|bareprompt|--visa-only" >> /tmp/bareprompt_specs_g0.txt
done

: > /tmp/bareprompt_specs_g3.txt
for ckpt in "${GRPO_CKPTS[@]}"; do
    echo "${ckpt}|--grpo-eval|grpoprompt|--ds-mvtec-only" >> /tmp/bareprompt_specs_g3.txt
    echo "${ckpt}|--grpo-eval|grpoprompt|--visa-only" >> /tmp/bareprompt_specs_g3.txt
done

echo "GPU 0 lane: $(wc -l < /tmp/bareprompt_specs_g0.txt) evals (SFT-bare ckpts)"
echo "GPU 3 lane: $(wc -l < /tmp/bareprompt_specs_g3.txt) evals (GRPO-bare ckpts)"

for g in 0 3; do
    tmux kill-session -t eval_bare_g$g 2>/dev/null
    tmux new-session -d -s eval_bare_g$g \
        "bash /bulk/aacudad/reasoning_traces/Training/run_eval_v2.sh $g /tmp/bareprompt_specs_g$g.txt 2>&1 | tee /tmp/eval_bare_g${g}.log"
    echo "Launched eval_bare_g$g tmux"
done
