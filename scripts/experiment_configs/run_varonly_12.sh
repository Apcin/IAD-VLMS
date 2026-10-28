#!/bin/bash
# Targeted-Variety ONLY (1k, no rehearsal) continuation, probe (ii) of tab:variety-cont.
# Runs ENTIRELY on CUDA 1+2, in PARALLEL with the varmix evals (which own 0/3). No waiting.
# Trains both inits (armC, grpo) on cuda 1,2; then evals each checkpoint with DS->cuda1, VisA->cuda2.
set -u
cd /bulk/aacudad/reasoning_traces
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
LOG=outputs/varonly_orchestrator.log
EVAL=Training/evaluate_qwen25vl_7b_trainprompt.py
BASE=Qwen/Qwen2.5-VL-7B-Instruct
say(){ echo "[$(date)] $*" | tee -a "$LOG"; }

say "varonly orchestrator (cuda 1+2) START -- training immediately, parallel to varmix evals on 0/3."

# --- Phase 1: train both inits on cuda 1,2 (2-GPU, sequential) ---
for tag in armC grpo; do
  OUT=outputs/sft_varonly_from_$tag
  mkdir -p "$OUT"
  say "TRAIN varonly_$tag (cuda 1,2)"
  CUDA_VISIBLE_DEVICES=1,2 FORCE_TORCHRUN=1 llamafactory-cli train Training/sft_varonly_from_$tag.yaml > "$OUT/train.log" 2>&1 \
    && touch "$OUT/.traindone" && say "  trained $tag" || say "  TRAIN FAILED $tag (see $OUT/train.log)"
done

# --- Phase 2: eval each checkpoint (DS->cuda1, VisA->cuda2 in parallel) ---
for tag in armC grpo; do
  for ck in outputs/sft_varonly_from_$tag/checkpoint-*; do
    [ -d "$ck" ] || continue
    say "EVAL $ck (DS->cuda1, VisA->cuda2)"
    CUDA_VISIBLE_DEVICES=1 python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --ds-mvtec-only \
        --output "$ck/eval_dsmvtec_full_trainprompt.json" > "$ck/eval_ds.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --visa-only \
        --output "$ck/eval_visa_full_trainprompt.json" > "$ck/eval_visa.log" 2>&1 &
    wait
  done
  touch "outputs/sft_varonly_from_$tag/.evaldone"
done

# --- Phase 3: print BA table ---
say "=== VARONLY (1k-only, no rehearsal) BA ==="
for tag in armC grpo; do
  for ck in outputs/sft_varonly_from_$tag/checkpoint-*; do
    for b in dsmvtec visa; do
      f="$ck/eval_${b}_full_trainprompt.json"
      [ -f "$f" ] && python3 repository_tu_delft_vlms/results/compute_ba.py "$f" 2>/dev/null | sed "s|^|$tag $(basename "$ck") $b: |" | tee -a "$LOG"
    done
  done
done
say "varonly ALL DONE"
