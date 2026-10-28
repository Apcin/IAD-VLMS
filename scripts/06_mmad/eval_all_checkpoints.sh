#!/bin/bash
# Evaluate the base model and every epoch checkpoint of the MMAD-1600 SFT on the four
# held-out MMAD subsets, one GPU, vLLM. Skips evaluations whose output file already exists.
# Usage: CUDA_VISIBLE_DEVICES=2 bash eval_all_checkpoints.sh [run_dir]
set -u
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export HF_HOME=${WORK_DIR}/hf_cache
PY=${WORK_DIR}/envs/grpo_fast312/bin/python
HERE=${WORK_DIR}/mmad_leak_experiment
RUN=${1:-${WORK_DIR}/outputs/sft_qwen25vl_7b_mmad_train1600}
EVALS=$HERE/evals
mkdir -p $EVALS/base
SUBSETS="DS-MVTec VisA GoodsAD MVTec-LOCO"

run_one() {   # $1 = checkpoint dir or "" for base, $2 = output dir
  local ck="$1" out="$2"
  mkdir -p "$out"
  for s in $SUBSETS; do
    local f="$out/eval_${s}_heldout.json"
    if [ -s "$f" ]; then echo "skip $f"; continue; fi
    echo "=== $(date +%H:%M) eval ${ck:-base} on $s (held-out) ==="
    if [ -n "$ck" ]; then
      $PY $HERE/eval_heldout_vllm.py --checkpoint "$ck" --subset $s --keys test --output "$f" 2>&1 | grep -E "vllm-eval|Error|Traceback|entries" | tail -5
    else
      $PY $HERE/eval_heldout_vllm.py --subset $s --keys test --output "$f" 2>&1 | grep -E "vllm-eval|Error|Traceback|entries" | tail -5
    fi
  done
}

# transformers-5 checkpoints cannot be parsed by the 4.57 vLLM env (no "architectures", rope under
# rope_parameters). Build a *_vllmcompat twin: stock 4.x config + tokenizer/processor files copied
# from the thesis compat dir, and a symlink to the checkpoint's model.safetensors.
COMPAT_SRC=${WORK_DIR}/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530_vllmcompat
make_compat() {   # $1 = checkpoint dir -> prints compat dir (kept OUTSIDE the run dir: the trainer's
                  # save_total_limit rotation counts any checkpoint-* dir and deleted checkpoint-50 of run 1)
  local ck="$1" cd="$HERE/compat/$(basename $RUN)/$(basename $ck)"
  if [ ! -s "$cd/config.json" ]; then
    mkdir -p "$cd"
    for f in chat_template.json config.json generation_config.json merges.txt preprocessor_config.json tokenizer_config.json tokenizer.json vocab.json; do
      cp "$COMPAT_SRC/$f" "$cd/"
    done
    ln -sfn "$ck/model.safetensors" "$cd/model.safetensors"
  fi
  echo "$cd"
}

run_one "" "$EVALS/base"
for ck in $(ls -d $RUN/checkpoint-* 2>/dev/null | grep -v vllmcompat | sort -t- -k2 -n); do
  [ -s "$ck/model.safetensors" ] || continue
  run_one "$(make_compat $ck)" "$EVALS/$(basename $ck)"
done
echo "=== strict held-out BA ==="
$PY $HERE/score_heldout.py $HERE/split.json $EVALS/*/eval_*_heldout.json 2>/dev/null | grep -E "json|held-out"
echo "EVAL ALL DONE $(date)"
