#!/bin/bash
# Re-run the interrupted varonly (1k-only) eval. Two parallel chains: armC on GPU0(DS)/1(VisA),
# grpo on GPU2(DS)/3(VisA). All 6 checkpoints (3 per init) x {DS-MVTec, VisA}.
set -u
cd /bulk/aacudad/reasoning_traces
source ~/miniconda3/etc/profile.d/conda.sh; conda activate llama_sft
EVAL=Training/evaluate_qwen25vl_7b_trainprompt.py; BASE=Qwen/Qwen2.5-VL-7B-Instruct
LOG=outputs/varonly_eval.log
say(){ echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }

eval_chain(){  # $1=tag $2=ds_gpu $3=visa_gpu
  for ck in outputs/sft_varonly_from_$1/checkpoint-*; do
    [ -d "$ck" ] || continue
    say "EVAL $ck (DS->cuda$2, VisA->cuda$3)"
    CUDA_VISIBLE_DEVICES=$2 python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --ds-mvtec-only \
        --output "$ck/eval_dsmvtec_full_trainprompt.json" > "$ck/eval_ds.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$3 python "$EVAL" --checkpoint "$ck" --base-model "$BASE" --visa-only \
        --output "$ck/eval_visa_full_trainprompt.json" > "$ck/eval_visa.log" 2>&1 &
    wait
  done
  touch "outputs/sft_varonly_from_$1/.evaldone"
  say "chain $1 done"
}
say "varonly eval start (armC on 0/1, grpo on 2/3)"
eval_chain armC 0 1 &
eval_chain grpo 2 3 &
wait
say "VARONLY EVAL ALL DONE"
# print table
for tag in armC grpo; do for ck in outputs/sft_varonly_from_$tag/checkpoint-*; do
  for b in dsmvtec visa; do f="$ck/eval_${b}_full_trainprompt.json";
    [ -f "$f" ] && python3 repository_tu_delft_vlms/results/compute_ba.py "$f" 2>/dev/null | sed "s|^|$tag $(basename $ck) $b: |" | tee -a "$LOG"; done
done; done
