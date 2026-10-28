#!/bin/bash
# ===========================================================================
# Arm D, the STaR self-rationalisation arm. Full run, end to end.
#
#   stage 1  preflight
#   stage 2  rationalise every item the policy failed, k=4, one GPU, vLLM
#   stage 3  build the 6,000-item 50/50 Arm D corpus and register it
#   stage 4  write sft_abc_D.yaml as Arm C's YAML with two lines changed
#   stage 5  arm the rolling epoch-eval watcher on the eval GPU
#   stage 6  SFT from Qwen2.5-VL-7B base, byte-identical recipe to Arm C
#
# The ONLY difference between the Arm C headline run and this one is the
# corpus. The base model, the frozen vision tower, the trainable projector,
# the batch geometry, the DeepSpeed ZeRO-3 CPU-offload config, the learning
# rate, the schedule, the epochs, the pixel cap and the cutoff length are all
# inherited from Training/sft_abc_C.yaml by substitution, and stage 4 refuses
# to continue if more than the dataset and the output directory changed.
#
# Registry: needs the dataset key iad_sft_armd_swap (or _pure). build_armd_dataset.py
# --register writes it; see also configs/dataset_info_additions.json.
#
# THIS RESULT IS NOT FOR THE THESIS. It is a post-upload / defence artefact.
#
# INPUTS THAT ARE NOT IN THIS REPOSITORY. The two phase-0 pools
# (needs_correction.json, good_traces.json, gemini_corrected.jsonl), the Arm C corpus
# and the GRPO ckpt-530 weights all live outside the repo: the first three carry
# absolute Real-IAD image paths, which we do not redistribute (README section 3), and
# the weights are on HuggingFace. Regenerate the pools with the phase-0 scripts in
# this directory, or point WORK_DIR at a workspace that already has them.
#
# Run it inside tmux. Never nohup.
#   tmux new-session -d -s armD "bash scripts/03_rollout_star/run_star_arm_d.sh"
#
# Environment knobs (all optional):
#   ARMD_VARIANT   swap (default) or pure. See build_armd_dataset.py.
#   ARMD_RULE      production (default) or answer_format. Keep rule for a
#                  rationalisation. production is the bar every kept trace in
#                  Arms A/B/C had to clear.
#   ARMD_K         hinted samples per failed item, default 4.
#   ARMD_RAT_GPU   GPU for the rationalisation pass, default 1.
#   ARMD_SFT_GPUS  GPUs for the SFT, default 1,2.
#   ARMD_EVAL_GPU  GPU for the rolling evals, default 3.
#   ARMD_SKIP_EVALS=1  do not arm the eval watcher.
# ===========================================================================
set -uo pipefail

# Paths. WORK_DIR is the workspace holding Training/, outputs/ and the python envs; it
# defaults to the parent of this repository.
# On another machine:  export WORK_DIR=/path/to/workspace
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
R="${WORK_DIR:-$(dirname "$REPO_ROOT")}"
T="${T:-$R/Training}"

VARIANT="${ARMD_VARIANT:-swap}"
RULE="${ARMD_RULE:-production}"
K="${ARMD_K:-4}"
RAT_GPU="${ARMD_RAT_GPU:-1}"
SFT_GPUS="${ARMD_SFT_GPUS:-1,2}"
EVAL_GPU="${ARMD_EVAL_GPU:-3}"

RUN_DIR="${ARMD_RUN_DIR:-$R/outputs/armd_star}"
STAR_JSONL=$RUN_DIR/star_full_k${K}.jsonl
STAR_LOG=$RUN_DIR/star_full_k${K}.log
LOG="${ARMD_LOG:-$R/outputs/armd_run.log}"

CKPT_530="${GRPO_CKPT_530:-$R/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530}"
# Python 3.12 env with vLLM. Override with VLLM_PYTHON.
VLLM_PY="${VLLM_PYTHON:-$R/envs/grpo_fast312/bin/python}"
# Arm C's YAML, and where the generated Arm D YAML goes. Both live in $T because
# scripts/01_sft/run_abc_train.sh resolves sft_abc_<ARM>.yaml there. The repository copy
# of the C recipe is configs/sft/sft_abc_C.yaml; it is the same file.
YAML_C="${YAML_C:-$T/sft_abc_C.yaml}"
[ -f "$YAML_C" ] || YAML_C=$REPO_ROOT/configs/sft/sft_abc_C.yaml
YAML_D="${YAML_D:-$T/sft_abc_D.yaml}"

case "$VARIANT" in
  swap) DSKEY=iad_sft_armd_swap; OUTDIR=$R/outputs/sft_qwen25vl_7b_abc_D_star_swap ;;
  pure) DSKEY=iad_sft_armd_pure; OUTDIR=$R/outputs/sft_qwen25vl_7b_abc_D_star_pure ;;
  *)    echo "ARMD_VARIANT must be swap or pure, got '$VARIANT'"; exit 1 ;;
esac
DSFILE=$T/datasets_sft_armd/sft_D_star_${VARIANT}.json

mkdir -p "$RUN_DIR" "$R/outputs"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [armD] $*" | tee -a "$LOG"; }
die(){ log "FATAL: $*"; exit 1; }

gpu_mib(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null; }

wait_gpu_free(){   # $1 gpu, $2 label, waits up to 20 minutes
  local g=$1 lbl=$2 i=0 m
  while [ $i -lt 40 ]; do
    m=$(gpu_mib "$g")
    if [ -n "$m" ] && [ "$m" -lt 3000 ]; then
      log "GPU $g is free (${m} MiB) $lbl"; return 0
    fi
    log "GPU $g still holds ${m} MiB $lbl, waiting 30s"
    sleep 30; i=$((i+1))
  done
  return 1
}

log "════════════════════════════════════════════════════════════════════"
log "Arm D full run. variant=$VARIANT keep-rule=$RULE k=$K"
log "rationalise on cuda $RAT_GPU | SFT on cuda $SFT_GPUS | evals on cuda $EVAL_GPU"
log "corpus  -> $DSFILE"
log "outputs -> $OUTDIR"
log "NOT FOR THE THESIS. Post-upload artefact."
log "════════════════════════════════════════════════════════════════════"

# ---------------------------------------------------------------------------
# Stage 1. Preflight.
# ---------------------------------------------------------------------------
log "stage 1: preflight"

case ",$SFT_GPUS,$RAT_GPU,$EVAL_GPU," in
  *,0,*) die "GPU 0 belongs to another user. Never use it." ;;
esac

for f in "$HERE/star_rationalise.py" "$HERE/build_armd_dataset.py" \
         "$REPO_ROOT/scripts/01_sft/run_abc_train.sh" \
         "$REPO_ROOT/scripts/04_eval/run_abc_eval_one.sh" \
         "$REPO_ROOT/scripts/04_eval/watch_armd_evals.sh" "$YAML_C" \
         "$REPO_ROOT/configs/deepspeed/ds_z3_cpu_offload.json" "$VLLM_PY" \
         "$T/datasets_sft_iter2/sft_iter2_train.json"; do
  [ -e "$f" ] || die "missing: $f"
done
for d in "$CKPT_530" "$T/phase0_full_10k_20260529_015821" "$T/phase0_heldout_20260601"; do
  [ -d "$d" ] || die "missing directory: $d"
done
log "  every input path is present"

FREE_GB=$(df -BG --output=avail "$R" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 120 ] || die "only ${FREE_GB} GB free on $R, four 16 GB checkpoints need about 70 GB"
log "  $R has ${FREE_GB} GB free"

# The type reward asks the Nomic embedding server for a similarity score. If the
# server is down, score_one_detailed silently falls back to lexical similarity
# and the keep rule quietly changes. The May rollout logs show that happening.
JURL="${GEMINI_JUDGE_URL:-http://127.0.0.1:5300}"
if curl -s -m 5 -o /dev/null "$JURL"; then
  log "  embedding judge server answering at $JURL"
else
  die "no answer from the embedding judge server at $JURL. Start it (tmux rank_judge) first, otherwise the type score falls back to lexical similarity and the keep rule is not the production one."
fi

if [ -f "$OUTDIR/train.done" ]; then
  die "$OUTDIR/train.done already exists. This run has already been done. Move it aside first."
fi

# ---------------------------------------------------------------------------
# Stage 2. Rationalise every failure.
# ---------------------------------------------------------------------------
log "stage 2: self-rationalisation, k=$K, on cuda $RAT_GPU"
log "  policy = $(basename "$CKPT_530") of grpo_qwen25vl_7b_6k_frozen_ep3_full_run2"
log "  pool   = the two needs_correction.json files, 1,942 items, 1,940 rationalisable"
log "  hint   = the gold yes/no only. No type, no location, no mask, no teacher."

wait_gpu_free "$RAT_GPU" "before the rationalisation pass" || die "cuda $RAT_GPU never freed up"

if [ -s "$STAR_JSONL" ]; then
  log "  $STAR_JSONL exists with $(wc -l <"$STAR_JSONL") lines, star_rationalise.py will resume"
fi

CUDA_VISIBLE_DEVICES="$RAT_GPU" "$VLLM_PY" "$HERE/star_rationalise.py" \
  --model_path "$CKPT_530" \
  --output_jsonl "$STAR_JSONL" \
  --k "$K" \
  2>&1 | tee -a "$STAR_LOG"
RAT_RC=${PIPESTATUS[0]}
[ "$RAT_RC" -eq 0 ] || die "star_rationalise.py exited $RAT_RC, see $STAR_LOG"

N_RAT=$(wc -l <"$STAR_JSONL")
log "  rationalisation pass finished, $N_RAT records in $STAR_JSONL"
[ "$N_RAT" -ge 1900 ] || die "only $N_RAT records, expected about 1,940. Not building a corpus from a partial pass."

wait_gpu_free "$RAT_GPU" "after the rationalisation pass" || die "cuda $RAT_GPU did not release its memory"

# ---------------------------------------------------------------------------
# Stage 3. Build and register the corpus.
# ---------------------------------------------------------------------------
log "stage 3: building the Arm D corpus (same size, class balance and 50/50 rule as Arm C)"

python3 "$HERE/build_armd_dataset.py" \
  --star_jsonl "$STAR_JSONL" \
  --variant both \
  --rule "$RULE" \
  --register 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "build_armd_dataset.py failed"
[ -s "$DSFILE" ] || die "corpus not written: $DSFILE"

log "stage 3: sanity check on $DSFILE"
python3 - "$DSFILE" "$T/datasets_sft_iter2/sft_iter2_train.json" <<'EOF' 2>&1 | tee -a "$LOG"
import json, os, random, sys
d = json.load(open(sys.argv[1]))
c = json.load(open(sys.argv[2]))
assert len(d) == len(c), f"size {len(d)} != Arm C {len(c)}"
ng = sum(1 for x in d if "/NG/" in x["images"][0] or "_NG_" in x["images"][0])
assert ng == len(d) - ng, f"not 50/50: {ng} NG / {len(d)-ng} OK"
for r in d:
    assert set(r.keys()) == {"messages", "images"}
    assert [m["role"] for m in r["messages"]] == ["user", "assistant"]
    assert r["messages"][0]["content"].startswith("<image>\n")
    assert r["messages"][1]["content"].strip()
paths = [x["images"][0] for x in d]
assert len(set(paths)) == len(paths), "duplicate images in the corpus"
random.seed(0)
missing = [p for p in random.sample(paths, 40) if not os.path.exists(p)]
assert not missing, f"images not on disk: {missing[:3]}"
print(f"SANITY OK: {len(d)} items, {ng} NG / {len(d)-ng} OK, unique images, schema matches Arm C")
EOF
[ "${PIPESTATUS[0]}" -eq 0 ] || die "corpus sanity check failed, not launching the SFT"

# ---------------------------------------------------------------------------
# Stage 4. Derive the D YAML from the C YAML.
# ---------------------------------------------------------------------------
log "stage 4: writing $YAML_D from $YAML_C"
{
  echo "### Arm D, STaR self-rationalisation. GENERATED by run_star_arm_d.sh, do not hand-edit."
  echo "### This file is sft_abc_C.yaml with two lines changed, dataset and output_dir."
  echo "### The Arm C header below is left in place on purpose so the diff stays two lines."
  echo "### variant=$VARIANT keep-rule=$RULE k=$K   generated $(date '+%Y-%m-%d %H:%M:%S')"
  echo "### NOT FOR THE THESIS."
  sed -e "s|^dataset: .*|dataset: $DSKEY|" \
      -e "s|^output_dir: .*|output_dir: $OUTDIR|" "$YAML_C"
} > "$YAML_D" || die "could not write $YAML_D"

STRIP='^[[:space:]]*(#|$)'
NDIFF=$(diff <(grep -vE "$STRIP" "$YAML_C") <(grep -vE "$STRIP" "$YAML_D") | grep -c '^[<>]')
CHANGED=$(diff <(grep -vE "$STRIP" "$YAML_C") <(grep -vE "$STRIP" "$YAML_D") \
          | grep '^[<>]' | sed 's/^[<>] //' | cut -d: -f1 | sort -u | tr '\n' ' ')
log "  settings that differ from Arm C: $CHANGED"
diff <(grep -vE "$STRIP" "$YAML_C") <(grep -vE "$STRIP" "$YAML_D") | tee -a "$LOG"
[ "$NDIFF" -eq 4 ] || die "$NDIFF changed lines between the C and D YAMLs, expected 4 (two keys, removed and added). Refusing to run a recipe that is not Arm C's."
[ "$CHANGED" = "dataset output_dir " ] || die "changed keys are '$CHANGED', expected 'dataset output_dir '"
grep -q '^freeze_vision_tower: true' "$YAML_D" || die "frozen vision tower missing from $YAML_D"
grep -q '^model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct' "$YAML_D" || die "not training from base"
grep -q '^deepspeed: .*ds_z3_cpu_offload.json' "$YAML_D" || die "DeepSpeed config missing from $YAML_D"
log "  verified: from base, frozen vision tower, trainable projector, ZeRO-3 CPU offload,"
log "  per-device batch 8 x grad-accum 2 x 2 GPUs, LR 1e-5 cosine, 20 warmup, 4 epochs, save per epoch"

# ---------------------------------------------------------------------------
# Stage 5. Arm the rolling eval watcher.
# ---------------------------------------------------------------------------
if [ "${ARMD_SKIP_EVALS:-0}" = "1" ]; then
  log "stage 5: ARMD_SKIP_EVALS=1, not arming the eval watcher"
else
  log "stage 5: arming the epoch-eval watcher on cuda $EVAL_GPU (tmux armD_evals)"
  tmux kill-session -t armD_evals 2>/dev/null
  # ARMD_PID lets the watcher tell "the SFT has not saved yet" from "the SFT
  # died". It is a kill -0 liveness probe on this script, nothing is ever killed.
  tmux new-session -d -s armD_evals \
    "ARMD_OUTDIR=$OUTDIR ARMD_PID=$$ bash $REPO_ROOT/scripts/04_eval/watch_armd_evals.sh $EVAL_GPU"
  sleep 2
  tmux has-session -t armD_evals 2>/dev/null \
    && log "  armD_evals is up, it derives the cadence from trainer_state.json" \
    || log "  WARNING: armD_evals did not start, evaluate the checkpoints by hand"
fi

# ---------------------------------------------------------------------------
# Stage 6. Train.
# ---------------------------------------------------------------------------
log "stage 6: SFT on cuda $SFT_GPUS, about 5.1 hours for 6,000 items over 4 epochs"
for g in ${SFT_GPUS//,/ }; do
  wait_gpu_free "$g" "before the SFT" || die "cuda $g is not free"
done

# conda.sh is third-party code, do not let set -u in this script abort on it
set +u
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
set -u

bash "$REPO_ROOT/scripts/01_sft/run_abc_train.sh" D "$SFT_GPUS" 2>&1 | tee -a "$LOG"
SFT_RC=${PIPESTATUS[0]}
if [ "$SFT_RC" -ne 0 ]; then
  log "SFT exited $SFT_RC. The eval watcher gives up on its own once this process is gone"
  log "and a checkpoint it is waiting for has still not appeared after three polls."
  exit "$SFT_RC"
fi

touch "$OUTDIR/train.done"
log "════════════════════════════════════════════════════════════════════"
log "Arm D SFT done. Checkpoints in $OUTDIR"
log "The eval watcher on cuda $EVAL_GPU finishes the last pair about 35 minutes from now."
log "Watch it with: tail -f $R/outputs/armd_evals.log"
log "Reminder: this number does not go into the thesis."
log "════════════════════════════════════════════════════════════════════"
