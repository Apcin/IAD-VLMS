#!/bin/bash
# Sequential vLLM eval pipeline on ONE GPU (default cuda 3):
#   1. Finish the interrupted iter2 ckpt-748 DS-MVTec eval (VisA-748 already done via HF).
#   2. Watch the running GRPO run (outputs/grpo_llava_ov_from_ep1) and evaluate each
#      checkpoint (530, 1060) on full DS-MVTec + VisA as soon as its shards are complete.
# Protocol identical to the HF watcher evals (trainprompt mode, 262144-px cap, greedy,
# 1024 max tokens) via scripts/04_eval/evaluate_vllm_llava.py. HF result files untouched.
# Usage: watch_grpo_llava_vllm_evals.sh [gpu]
set -uo pipefail
GPU="${1:-3}"
# Paths. WORK_DIR is the workspace holding outputs/, hf_cache/ and the python envs;
# it defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
# Python 3.12 env with vLLM. Override with VLLM_PYTHON.
PY="${VLLM_PYTHON:-$WORK_DIR/envs/grpo_fast312/bin/python}"
EV=$HERE/evaluate_vllm_llava.py
BASE=llava-hf/llava-onevision-qwen2-7b-si-hf
GRPO="${GRPO_WATCH_DIR:-$WORK_DIR/outputs/grpo_llava_ov_from_ep1}"
REFCKPT="${GRPO_REF_CKPT:-$WORK_DIR/outputs/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188}"
export HF_HOME=$WORK_DIR/hf_cache
LOG=$WORK_DIR/outputs/vllm_evals_cuda3.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

bal(){ $PY - "$1" <<'EOF'
import json,sys
try:
    m=json.load(open(sys.argv[1]))['metrics']; tp,tn,fp,fn=m['tp'],m['tn'],m['fp'],m['fn']
    print(f"{50*(tp/((tp+fn) or 1)+tn/((tn+fp) or 1)):.2f}")
except Exception: print("nan")
EOF
}

# Weights complete: either all shards named in an index exist with matching total size,
# or a single-file model.safetensors exists and passes a header parse (fully written).
shards_complete(){ $PY - "$1" <<'EOF'
import json,os,struct,sys
d=sys.argv[1]; idx=os.path.join(d,"model.safetensors.index.json")
single=os.path.join(d,"model.safetensors")
if os.path.exists(idx):
    j=json.load(open(idx)); shards=set(j["weight_map"].values())
    sizes=[os.path.getsize(os.path.join(d,s)) for s in shards if os.path.exists(os.path.join(d,s))]
    if len(sizes)!=len(shards): sys.exit(1)
    tot=j.get("metadata",{}).get("total_size")
    sys.exit(0 if (tot is None or sum(sizes)>=tot*0.99) else 1)
if os.path.exists(single) and os.path.getsize(single) > 1_000_000_000:
    try:  # safetensors header: u64 header-length, then JSON header; parses only when intact
        with open(single,"rb") as f:
            n=struct.unpack("<Q",f.read(8))[0]
            json.loads(f.read(n))
        sys.exit(0)
    except Exception:
        sys.exit(1)
sys.exit(1)
EOF
}

wait_ckpt_complete(){  # arg: ckpt dir — wait for shards complete AND sizes stable 60s
  local D="$1"
  until [ -d "$D" ]; do sleep 300; done
  log "checkpoint dir $D appeared, waiting for complete shards..."
  until shards_complete "$D"; do sleep 60; done
  local s1 s2
  s1=$(du -sb "$D" | cut -f1); sleep 60; s2=$(du -sb "$D" | cut -f1)
  while [ "$s1" != "$s2" ]; do s1=$s2; sleep 60; s2=$(du -sb "$D" | cut -f1); done
  # The training env (transformers 5.0) writes tokenizer/processor JSONs in a format
  # transformers 4.57 (vllm env) cannot parse. GRPO never changes the tokenizer, so
  # FORCE-copy all tokenizer/processor files from the init ckpt (keep the trainer's
  # config.json / model weights untouched).
  for f in preprocessor_config.json processor_config.json tokenizer_config.json \
           tokenizer.json added_tokens.json special_tokens_map.json vocab.json \
           merges.txt chat_template.jinja chat_template.json; do
    [ -e "$REFCKPT/$f" ] && cp "$REFCKPT/$f" "$D/$f"
  done
  log "tokenizer/processor files replaced from ckpt-188 (transformers-5 -> 4.57 compat)"
  # transformers 5.0 writes rope under text_config.rope_parameters, which 4.57 ignores ->
  # silent rope_theta=10000 (100x wrong) at eval load. Inject the explicit 4.x key.
  $PY - "$D/config.json" <<'EOF'
import json,sys
p=sys.argv[1]; c=json.load(open(p))
tc=c.get("text_config",{})
rp=tc.get("rope_parameters",{})
if "rope_theta" not in tc:
    tc["rope_theta"]=float(rp.get("rope_theta",1000000.0))
    json.dump(c,open(p,"w"),indent=2)
    print("injected text_config.rope_theta =",tc["rope_theta"])
EOF
  log "config.json rope_theta verified/injected (transformers-5 rope_parameters trap)"
  log "checkpoint $D complete."
}

log "════ vLLM eval pipeline on cuda$GPU ════"

# ── 1. finish iter2 ckpt-748: DS-MVTec via vLLM ──
CK748=$WORK_DIR/outputs/sft_llava_ov_7b_frozen_iad_sft_iter2/checkpoint-748
if [ ! -s "$CK748/eval_dsmvtec_full_trainprompt_vllm.json" ]; then
  log "eval iter2/ckpt-748 DS-MVTec (vllm)..."
  CUDA_VISIBLE_DEVICES=$GPU $PY "$EV" --checkpoint "$CK748" --base-model "$BASE" \
    --out-ds "$CK748/eval_dsmvtec_full_trainprompt_vllm.json" \
    > "$CK748/eval_ds_vllm.log" 2>&1
  log "RESULT iter2/ckpt-748 (epoch 4): DS-MVTec $(bal "$CK748/eval_dsmvtec_full_trainprompt_vllm.json") [vllm] | VisA $(bal "$CK748/eval_visa_full_trainprompt.json") [hf] (balanced acc)"
else
  log "(ckpt-748 DS vllm eval already present, skip)"
fi

# ── 2. watch GRPO checkpoints ──
for STEP in ${GRPO_WATCH_STEPS:-530 1060}; do
  D=$GRPO/checkpoint-$STEP
  if [ -s "$D/eval_dsmvtec_full_trainprompt_vllm.json" ] && [ -s "$D/eval_visa_full_trainprompt_vllm.json" ]; then
    log "(grpo ckpt-$STEP already evaluated, skip)"; continue
  fi
  log "waiting for GRPO checkpoint-$STEP ..."
  wait_ckpt_complete "$D"
  log "eval grpo/ckpt-$STEP: DS-MVTec + VisA (vllm, one engine)..."
  CUDA_VISIBLE_DEVICES=$GPU $PY "$EV" --checkpoint "$D" --base-model "$BASE" \
    --out-ds   "$D/eval_dsmvtec_full_trainprompt_vllm.json" \
    --out-visa "$D/eval_visa_full_trainprompt_vllm.json" \
    > "$D/eval_vllm.log" 2>&1
  log "RESULT grpo/ckpt-$STEP: DS-MVTec $(bal "$D/eval_dsmvtec_full_trainprompt_vllm.json") | VisA $(bal "$D/eval_visa_full_trainprompt_vllm.json") (balanced acc, vllm)"
done

log "════ PIPELINE DONE ════"
grep 'RESULT' "$LOG"
