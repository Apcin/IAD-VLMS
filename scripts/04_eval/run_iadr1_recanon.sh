#!/bin/bash
# Re-evaluate the IAD-R1 Qwen checkpoint on the CANONICAL MMAD subsets (1670 DS-MVTec / 2141 VisA)
# through OUR harness with --grpo-eval -> identical prompt/parser/data as our own models (true common footing).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh; conda activate llama_sft
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
EVAL=/bulk/aacudad/reasoning_traces/Training/evaluate_qwen25vl_7b_trainprompt.py
CKPT=/bulk/aacudad/reasoning_traces/iad_r1_qwen_model
BASE=Qwen/Qwen2.5-VL-7B-Instruct
OUT=/bulk/aacudad/reasoning_traces/outputs/iad_r1_qwen_recanon
mkdir -p "$OUT"; cd /bulk/aacudad/reasoning_traces
echo "[iadr1-recanon] start $(date)"
CUDA_VISIBLE_DEVICES=1 python "$EVAL" --checkpoint "$CKPT" --base-model "$BASE" \
   --ds-mvtec-only --grpo-eval --batch-size 4 --output "$OUT/eval_dsmvtec_full_trainprompt.json" > "$OUT/ds.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python "$EVAL" --checkpoint "$CKPT" --base-model "$BASE" \
   --visa-only --grpo-eval --batch-size 4 --output "$OUT/eval_visa_full_trainprompt.json" > "$OUT/visa.log" 2>&1 &
P2=$!
wait $P1 $P2
python3 - <<'PY'
import json,os
def bal(p):
    m=json.load(open(p))["metrics"]; tp,tn,fp,fn=m["tp"],m["tn"],m["fp"],m["fn"]
    return round(100*((tp/(tp+fn))+(tn/(tn+fp)))/2,2), m["total"]
o="/bulk/aacudad/reasoning_traces/outputs/iad_r1_qwen_recanon"
for n,f in [("DS-MVTec","eval_dsmvtec_full_trainprompt.json"),("VisA","eval_visa_full_trainprompt.json")]:
    try: b,t=bal(f"{o}/{f}"); print(f"[iadr1-recanon] {n}: balanced_acc={b}  (n={t})")
    except Exception as e: print(f"[iadr1-recanon] {n}: FAILED {e}")
PY
echo "[iadr1-recanon] DONE $(date)"
touch /bulk/aacudad/reasoning_traces/Training/iadr1_recanon.done
