#!/bin/bash
# Zero-shot MMAD eval of the Gemini 3.x Flash family, same harness/prompt as outputs/gemini3flash_eval (thinking level low: "minimal" returns empty text on 3.5 and is rejected by 3.7 and 3.8).
# One process per (model, benchmark), all eight in parallel, no sharding. Vertex project/key are the harness defaults.
source $CONDA_SH; conda activate llama_sft
cd ${WORK_DIR:-..}
export CUDA_VISIBLE_DEVICES="" HF_HOME=${WORK_DIR:-..}/hf_cache
EVAL=Training/evaluate_qwen25vl_7b_trainprompt.py
for M in gemini-3.5-flash gemini-3.6-flash gemini-3.7-flash gemini-3.8-flash; do
  D=outputs/gemini_flash_family_eval/$M; mkdir -p $D/logs
  for B in ds visa; do
    FLAG=$([ $B = ds ] && echo --ds-mvtec-only || echo --visa-only)
    nohup python $EVAL --gemini --gemini-model $M --gemini-thinking-level low $FLAG --batch-size 4 \
      --output $D/eval_${M}_low_${B}.json > $D/logs/${B}.log 2>&1 &
    echo "[$(date +%T)] started $M $B pid $!" | tee -a outputs/gemini_flash_family_eval/launch.log
  done
done
wait
echo "[$(date +%T)] all done" | tee -a outputs/gemini_flash_family_eval/launch.log
