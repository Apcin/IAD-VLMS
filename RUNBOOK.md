# RUNBOOK — Reasoning-Enhanced VLMs for Explainable Industrial Anomaly Detection

End-to-end setup and reproduction guide for a new student on the **same TU Delft cluster**.
Project: MSc thesis, Adnane Acudad, 2026 — Qwen2.5-VL fine-tuned for explainable industrial
anomaly detection (AnomalyThink corpus → SFT → GRPO → STaR).

This document is the *operational* companion to the README. Every command below is
copy-pasteable. After each key step the **expected headline number** is stated so you
can confirm you reproduced the result, not just ran the script.

> **Authoritative numbers.** All balanced-accuracy (BA) values in this runbook were
> recomputed directly from the eval JSONs in [`results/`](results/) using
> [`results/compute_ba.py`](results/compute_ba.py). Where the thesis PDF disagrees,
> **trust this runbook and the JSONs.** See [`docs/UNVERIFIED.md`](docs/UNVERIFIED.md)
> (if present) and the README for the catalogue of thesis transcription errors.

---

## 0. Conventions and a note on paths

The scripts in this repo were authored against the original working tree, where everything
lived under `/bulk/aacudad/reasoning_traces/`:

| Original working-tree location              | What it is                          | Shipped in this repo as            |
| ------------------------------------------- | ----------------------------------- | ---------------------------------- |
| `Training/*.yaml`                           | SFT configs                         | [`configs/sft/`](configs/sft/)     |
| `Training/datasets_*/`                      | LlamaFactory dataset JSONs          | [`traces/`](traces/)               |
| `Training/iad_r1_grpo_custom/stage_rl/`     | GRPO trainer + rewards              | [`scripts/02_grpo/stage_rl/`](scripts/02_grpo/stage_rl/) |
| `Training/evaluate_qwen25vl_7b_trainprompt.py` | Eval harness                     | [`scripts/04_eval/`](scripts/04_eval/) |
| `outputs/<run>/`                            | Trained checkpoints + eval JSONs    | weights NOT shipped; eval JSONs in [`results/`](results/) |

**Because of this, the shell/yaml scripts contain absolute `/bulk/.../Training/...` paths.**
Two ways to run them:

1. **Re-create the original layout** (recommended for byte-identical reproduction): clone this
   repo into `/bulk/<you>/reasoning_traces/`, then symlink or copy `configs/`, `scripts/`,
   `traces/` into the `Training/`-shaped paths the scripts expect (see §1.5).
2. **Edit the absolute paths** in the one script you are about to run to point at this repo's
   `configs/`, `scripts/`, `traces/`. The variables to change are always near the top
   (`MODEL_NAME_OR_PATH`, `DATASET_NAME`, `OUTPUT_DIR`, `EVAL_SCRIPT`, `PYTHONPATH`).

Throughout, replace `/bulk/aacudad/` with your own `/bulk/<you>/` if you are not the author.

---

## 1. Prerequisites & setup

### 1.1 Conda environment

All training, GRPO, and eval run inside the `llama_sft` conda env (torch + transformers +
deepspeed + trl with `GRPOConfig` + peft + LlamaFactory).

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sft
```

If the env does not yet exist, create it from Python 3.11 and install LlamaFactory plus the
GRPO stack:

```bash
conda create -n llama_sft python=3.11 -y
conda activate llama_sft
# LlamaFactory (training CLI) — clone + editable install
git clone https://github.com/hiyouga/LLaMA-Factory.git /bulk/aacudad/reasoning_traces/LlamaFactory
pip install -e "/bulk/aacudad/reasoning_traces/LlamaFactory[torch,metrics,deepspeed]"
# GRPO extras used by stage_rl
pip install trl peft accelerate qwen-vl-utils fastapi uvicorn python-dotenv \
            google-genai sentence-transformers
```

> `trl` must expose `GRPOConfig` (the custom `stage_rl` trainer subclasses it). If GRPO import
> fails, pin a `trl` version that ships `GRPOConfig`.

### 1.2 Caches and scratch dirs

Set these in **every** shell (the scripts also export them, but set them yourself so ad-hoc
commands behave):

```bash
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache
export TRITON_CACHE_DIR=/bulk/aacudad/reasoning_traces/tmp_cache/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"
```

### 1.3 Hugging Face auth (model + dataset downloads)

```bash
huggingface-cli login        # paste your HF token, or:
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The base model `Qwen/Qwen2.5-VL-7B-Instruct` (and `-3B-Instruct`) downloads into `$HF_HOME`
on first use. Real-IAD download requires a token with access to the gated dataset.

> **SECURITY — read before pushing.** Several download/judge scripts still contain
> **hardcoded secrets** (e.g. an HF token literal in
> [`scripts/00_generate/download_realiad.py`](scripts/00_generate/download_realiad.py) line 12,
> and a Vertex service-account key path in the judge servers). **Rotate every token/key, move
> them to `.env` / env vars, add a `.gitignore`, and run `gitleaks detect` before the first
> public commit.** Do not assume the shipped literals are valid.

### 1.4 Gemini / Vertex auth (only for trace *generation* and the GRPO/STaR judge)

Trace generation and the Gemini judge use Google's `genai` SDK. Two paths:

- **Vertex AI (service account)** — used by the Gemini judge servers and the variety generator:
  ```bash
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/vertexai-key.json
  ```
  Note: [`config_mmad.py`](scripts/00_generate/config_mmad.py) has an *import-time gate* that
  raises `ValueError` if `GEMINI_API_KEYS` is empty — even when you intend to use Vertex. The
  established workaround (used by the variety scripts) is to set a dummy value:
  ```bash
  export GEMINI_API_KEYS=dummy   # satisfies the config_mmad import gate; Vertex creds come from the key file
  ```

- **Gemini API (keys)** — used by the Real-IAD trace generator. Put comma-separated keys in a
  `.env` at the directory where you run the generator (loaded by `python-dotenv`):
  ```bash
  # .env
  GEMINI_API_KEYS=key1,key2,key3
  ```
  `config_mmad.GEMINI_API_KEYS` becomes the parsed list; the generator round-robins across them.

### 1.5 Register the datasets with LlamaFactory

LlamaFactory resolves a config's `dataset:` key against a `dataset_info.json`. This repo ships
[`configs/dataset_info.json`](configs/dataset_info.json). It maps logical names to file paths,
e.g. the headline SFT dataset:

```json
"iad_sft_6k_train": { "file_name": "/bulk/aacudad/reasoning_traces/Training/datasets_small_new_v4/combined_6k_train.json",
                       "formatting": "sharegpt",
                       "columns": { "messages": "messages", "images": "images" }, ... }
```

The shipped trace file for that same dataset is
[`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json).
**You must reconcile the path.** Either:

```bash
# Option A: point dataset_info.json at the shipped traces (edit file_name fields), then
cp /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/configs/dataset_info.json \
   /bulk/aacudad/reasoning_traces/LlamaFactory/data/dataset_info.json
```
or
```bash
# Option B: re-create the original layout so the existing absolute paths resolve
mkdir -p /bulk/aacudad/reasoning_traces/Training/datasets_small_new_v4
ln -s /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/traces/anomalythink_6k/combined_6k_train.json \
      /bulk/aacudad/reasoning_traces/Training/datasets_small_new_v4/combined_6k_train.json
```

Key logical dataset names (see [`configs/dataset_info.json`](configs/dataset_info.json) for all):

| Logical name              | Shipped trace file                                              | Used by                          |
| ------------------------- | --------------------------------------------------------------- | -------------------------------- |
| `iad_sft_6k_train`        | `traces/anomalythink_6k/combined_6k_train.json`                 | **headline 7B SFT (80.16)**      |
| `iad_grpo_15k_c1_train`   | `traces/anomalythink_15k/grpo_train.json`                       | GRPO (4,236 prompts)             |
| `iad_sft_15k_c1_train`    | `traces/anomalythink_15k/combined_sft_train.json`               | 15K SFT ablations                |
| `iad_sft_6k_bareprompt_train` | `traces/anomalythink_6k/iad_sft_6k_bareprompt_train.json`   | bare-prompt SFT ablation         |
| `iad_sft_A_kept_balanced` | `traces/teacher_ablation_abc/sft_A_kept_balanced.json`          | teacher-ablation Arm A           |
| `iad_sft_B_kept_corrected`| `traces/teacher_ablation_abc/sft_B_kept_corrected.json`         | teacher-ablation Arm B           |
| `iad_sft_iter2`           | `traces/iter2/sft_iter2_train.json`                             | teacher-ablation Arm C (best SFT)|
| `variety_star_sft_6k`     | `traces/variety_star_6k/variety_star_sft_6k.json`               | Real-IAD Variety STaR (WIP)      |

> **Image paths inside the trace JSONs are RELATIVE** (LlamaFactory `{messages, images}`
> sharegpt format). They point at the downloadable source images (Real-IAD / MVTec / MMAD).
> See §2 to materialise the images and make those relative paths resolve.

---

## 2. Where the data and images come from

The repo ships **text traces only** — no source/overlay images (re-downloadable) and no model
weights. Download scripts live in [`scripts/00_generate/`](scripts/00_generate/):

| Script                                                                 | Downloads                                  |
| ---------------------------------------------------------------------- | ------------------------------------------ |
| [`download_realiad.py`](scripts/00_generate/download_realiad.py)       | Real-IAD images + JSON (gated HF dataset)  |
| [`download_mmad.py`](scripts/00_generate/download_mmad.py)             | MMAD benchmark (DS-MVTec + VisA subsets)   |
| [`download_variety.py`](scripts/00_generate/download_variety.py)       | Real-IAD Variety pool (160-product STaR)   |

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama_sft
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export HF_TOKEN=hf_xxx     # token with Real-IAD access; ROTATE the literal in the script first

cd /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/scripts/00_generate
python download_realiad.py     # -> data/Real-IAD/{images,json}
python download_mmad.py        # -> data/MMAD/DS-MVTec , data/MMAD/VisA
```

**Pointing trace image paths at the images.** The relative paths in the trace JSONs are rooted
at the data dir the original tree used (`reasoning_traces_gen/data/...`). After download, set
your data root so the relative paths resolve. The simplest approach is to run training/eval
from a working directory where `data/Real-IAD/...` and `data/MMAD/...` exist exactly as the
trace JSON paths expect, or to symlink the download targets into that layout. The eval harness
builds its own image map from the MMAD layout (DS-MVTec / VisA) — see the `image_map`
construction in [`evaluate_qwen25vl_7b_trainprompt.py`](scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py).

---

## 3. The six stages

Each stage states the **real script name**, a copy-pasteable command, and the expected number.

### Stage 0 — Generate reasoning traces (Gemini-2.5-Flash, C1-only)

> You normally **skip this** — the 14,472 traces are already shipped under
> [`traces/`](traces/). Run it only to regenerate or extend the corpus.

Generator config: Gemini-2.5-Flash, temperature 1.0, batch 5–10, target 120–200 words inside
`<think>`, 3 images per anomaly (original + overlay + reference), 1 image per normal. QC is
**in-prompt auto-reject rules + JSON/Pydantic schema validation** (no second-reviewer model,
no GPT-5-mini, no 8-rubric filter).

```bash
cd /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/scripts/00_generate
# Gemini API path (.env with GEMINI_API_KEYS), sharded for throughput:
python generate_realiad_traces.py --shard_id 0 --total_shards 8 --keys "$GEMINI_API_KEYS"
# ... shards 1..7 in parallel, then merge:
python ../03_rollout_star/merge_rollout_shards.py
```

The full corpus is then partitioned into **3 disjoint stratified (product × normal/anomaly)
splits**: **6,000 SFT / 4,236 GRPO / 4,236 held-out RealIAD** ("AnomalyThink-15K" = the union,
≈ 14,472). The 6K SFT split is a 100% subset of the union; both 6K and 15K have the same mean
trace length (~141 words).

### Stage 1 — SFT (LlamaFactory) — **HEADLINE: 7B-Frozen-6K → DS-MVTec BA 80.16**

The headline config is
[`configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml`](configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml):
full fine-tune with **vision tower frozen** (`freeze_vision_tower: true`), 4 epochs, LR 1e-5,
cosine, per-device BS 4 × GA 4, bf16, DeepSpeed ZeRO-3 CPU-offload, 2 GPUs.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama_sft
export CUDA_VISIBLE_DEVICES=0,3
export FORCE_TORCHRUN=1
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
export TRANSFORMERS_CACHE=/bulk/aacudad/reasoning_traces/hf_cache

cd /bulk/aacudad/reasoning_traces/LlamaFactory
llamafactory-cli train \
  /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml
```

The launcher wrapper is
[`scripts/01_sft/run_sft_qwen25vl_7b_zeroshot_6k_frozen.sh`](scripts/01_sft/run_sft_qwen25vl_7b_zeroshot_6k_frozen.sh).
Before running, in the yaml fix `deepspeed:` (use a shipped config under
[`configs/deepspeed/`](configs/deepspeed/), e.g. `ds_z3_cpu_offload.json`), the `output_dir:`,
and confirm `dataset: iad_sft_6k_train` resolves to the shipped trace (§1.5).

**Checkpoint to keep: `checkpoint-564` (epoch 3).**

Expected (recomputed from JSON):

| Checkpoint   | DS-MVTec BA | VisA BA | Acc   | F1    |
| ------------ | ----------- | ------- | ----- | ----- |
| **ckpt-564** | **80.16**   | **64.78** | 80.36 | 85.76 |

JSONs: [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/) (+ `_visa_`).

Other SFT configs (same CLI, swap the yaml) live in [`configs/sft/`](configs/sft/) and have
matching launchers in [`scripts/01_sft/`](scripts/01_sft/). The **teacher-ablation Arm C**
config [`sft_abc_C.yaml`](configs/sft/sft_abc_C.yaml) → `checkpoint-376` is the best SFT
(**DS 82.80 / VisA 72.07**) and is the init for GRPO-on-C and the Variety STaR generator;
launcher [`scripts/01_sft/run_abc_master.sh`](scripts/01_sft/run_abc_master.sh).

### Stage 2 — GRPO — **HEADLINE RL: Run-2 ckpt-530 → DS-MVTec BA 82.73**

GRPO uses the IAD-R1-derived `SCGRPOTrainer` in
[`scripts/02_grpo/stage_rl/`](scripts/02_grpo/stage_rl/) with a custom 4-component reward
(format `w_f=0.3`, answer `w_a=0.3`, type `w_t=0.2`, location `w_l=0.2`).

**Authoritative GRPO config** (from the saved `training_args.bin` of run-2):

- base / reference = **7B-Frozen-6K SFT ckpt-564**
- `G` (num_generations) = **4**; `beta` (KL coef) = **0** — KL is *monitored* via Schulman k3,
  **not added to the loss**
- `eta` (LR) = **1e-6**; `epsilon` (clip) = **0.2**
- samples = **4,236** (2,118 anomaly + 2,118 normal); max steps **1,060**; save/eval every 250
- reward weights `(w_f, w_a, w_t, w_l) = (0.3, 0.3, 0.2, 0.2)`
- **reasoning/judge reward (server :5100) NOT used → flat 0.5 ("no reasoning reward")**
- type-embedding model `nomic-embed-text-v2-moe` served on `:5200`
- per-device BS 2, GA 4, EBS 8; max prompt 2048 / completion 512; bf16; seed 42
- 2× RTX A6000, DeepSpeed ZeRO-3 CPU offload

> Any thesis values of `G=2`, `beta=0.04`, `eta=5e-6`, or "13K samples" are **OLD/WRONG**. Some
> shell-script comments (e.g. in `run_grpo_abc_C_4gpu.sh`) also quote the old `β=0.04 / 5e-6` —
> ignore the comments, trust the values above.

**Optional switches in the trainer, all off unless you set them.** The snapshot in
[`scripts/02_grpo/stage_rl/`](scripts/02_grpo/stage_rl/) carries a few study-only features.
Every one is read from the environment and every one is a no-op when the variable is unset, so
none of them changes the runs described above. Leave them alone to reproduce.

| Variable | Effect when set |
| --- | --- |
| `REWARD_GATE_TAGS=1` | Type and location credit is paid only when the predicted verdict matches the gold label. Closes the hedging path. |
| `REWARD_BAND_FORM=1` | Full band-form answer reward, `1 + b*(q - 1/2)` on a correct verdict with `b=0.1`. Implies the gate. |
| `REWARD_COMPONENTS` | Comma list, default `verdict,format,type,loc`. Omit a name to zero that sub-signal. |
| `GRPO_MAX_IMAGE_PIXELS` | Area cap applied to images before the processor. Needed for LLaVA-OneVision, which ignores `--max_pixels`. |
| `GRPO_MASK_ZERO_STD_GROUPS=1` | Groups whose rollout rewards are exactly tied contribute no loss. |
| `GRPO_DRGRPO_TRUE=1` | Replaces the per-response length normaliser with the constant generation budget. Only does anything together with `--use_drgrpo`. |
| `GRPO_GROUP_STATS=<path>` | Appends one JSON line per step with the per-group rewards, standard deviations and the entropy proxy. |
| `GRPO_COMPLETION_LOG=<path>` | Dumps the first completions of each reward call. Smoke-test aid. |

[`scripts/02_grpo/stage_rl/test_reward_gating.py`](scripts/02_grpo/stage_rl/test_reward_gating.py)
is a CPU-only test file for the reward switches. Its first three tests assert that with all of
them unset the reward values are exactly the shipped ones. Run it with
`cd scripts/02_grpo/stage_rl && python test_reward_gating.py`.

**Start the type-embedding server** (required; the type reward calls it at
`http://127.0.0.1:5200/embed_similarity`):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama_sft
export JUDGE_PORT=5200
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/vertexai-key.json   # ROTATE first
export CUDA_VISIBLE_DEVICES=3
python /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/scripts/02_grpo/gemini_judge_server_v2.py
#   -> FastAPI/uvicorn on :5200; loads nomic-embed-text-v2-moe + serves /embed_similarity
```

The reasoning-judge server (Qwen2.5-VL on `:5100`,
[`judge_server.py`](scripts/02_grpo/judge_server.py)) is **not needed** for the headline run
(reasoning reward disabled → flat 0.5). Start it only if you re-enable that reward.

**Run-2 is the headline run.** It resumes from run-1's `checkpoint-530`. So reproduce in two
phases using [`scripts/02_grpo/run_grpo_7b_6k_frozen_ep3.sh`](scripts/02_grpo/run_grpo_7b_6k_frozen_ep3.sh)
(run-1, 1 epoch) then
[`scripts/02_grpo/run_grpo_7b_resume_run2.sh`](scripts/02_grpo/run_grpo_7b_resume_run2.sh)
(run-2, +2 epochs):

```bash
# Phase 1 (run-1): init = SFT ckpt-564
bash /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/scripts/02_grpo/run_grpo_7b_6k_frozen_ep3.sh
# Phase 2 (run-2): init = run-1/checkpoint-530, +2 epochs
bash /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/scripts/02_grpo/run_grpo_7b_resume_run2.sh
```

Before running, fix in each script: `PYTHONPATH` → this repo's
[`scripts/02_grpo/stage_rl`](scripts/02_grpo/stage_rl/), `MODEL_NAME_OR_PATH`, `DATASET_NAME`
(→ `traces/anomalythink_15k/grpo_train.json`), `OUTPUT_DIR`, the `grpo_ad.py` path, and the
`--deepspeed` config (→ [`configs/deepspeed/zero3_offload.json`](configs/deepspeed/)).

**Checkpoint to keep: run-2 `checkpoint-530`.** Expected (recomputed):

| Checkpoint              | DS-MVTec BA | VisA BA | Held-out RealIAD BA |
| ----------------------- | ----------- | ------- | ------------------- |
| **run-2 ckpt-530**      | **82.73**   | **70.39** | **80.87** (n=4236)  |

JSONs: [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/)
(`eval_dsmvtec_full_trainprompt.json`, `eval_visa_full_trainprompt.json`,
`eval_realiad4k_full_trainprompt.json`).

**GRPO-on-Arm-C** variant: [`scripts/02_grpo/run_grpo_abc_C_4gpu.sh`](scripts/02_grpo/run_grpo_abc_C_4gpu.sh)
(init = Arm-C ckpt-376). Its `checkpoint-265` reaches VisA 72.68; results in
[`results/grpo_qwen25vl_7b_abc_C_grpo/`](results/grpo_qwen25vl_7b_abc_C_grpo/).

### Stage 3 — Rollout / STaR (Real-IAD Variety — in progress, thesis placeholder)

The Variety STaR loop: **roll out Arm-C ckpt-376 → Gemini faithfulness judge → correct
(0/k fails) + rewrite (marginal) → SFT on a 6K stratified variety corpus** (3,000 anomaly /
3,000 normal, 160 products). Orchestrated by
[`scripts/03_rollout_star/run_variety_star.sh`](scripts/03_rollout_star/run_variety_star.sh),
which chains:

```text
assemble_variety_sft_pool.py     # merge Gemini traces into the rollout pool
phase0_rollout_kscoring.py       # k=8 rollouts of Arm-C ckpt-376 (2 shards, CUDA 0+3)
phase0_bucket.py                 # good / needs_rewrite / needs_correction
phase1a_gemini_judge.py          # Gemini faithfulness judge
phase1b_gemini_correct.py        # correct + rewrite
assemble_variety_star_6k.py      # build the 6K variety SFT corpus -> variety_star_sft_6k.json
```

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama_sft
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/vertexai-key.json   # ROTATE first
export GEMINI_API_KEYS=dummy   # config_mmad import gate (Vertex used for the judge)
bash /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/scripts/03_rollout_star/run_variety_star.sh
```

Then SFT on the resulting corpus with
[`configs/sft/sft_variety_star.yaml`](configs/sft/sft_variety_star.yaml)
(`dataset: variety_star_sft_6k`). The **A/B/C teacher-ablation** datasets are built by
[`scripts/03_rollout_star/build_abc_datasets.py`](scripts/03_rollout_star/build_abc_datasets.py)
(A = pure-kept 3,000 balanced; B = kept + corrected; C = kept + corrected + rewritten 6K).

### Stage 4 — Evaluation

Harness:
[`scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py`](scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py).
Benchmark flags: `--ds-mvtec-only` (n=1670), `--visa-only` (n=2141), `--realiad-4k` (held-out
n=4236). Batch via `--batch-size`. Base-model baseline: pass `--base-model Qwen/Qwen2.5-VL-7B-Instruct`
with no `--checkpoint` (or `--checkpoint` of a trained ckpt).

**Prompt-mode contract = the OUTPUT FILENAME SUFFIX.** The flag chooses the prompt; *you* encode
it in `--output` so downstream tooling can tell modes apart:

| Prompt mode    | Eval flag             | Filename suffix you must use |
| -------------- | --------------------- | ---------------------------- |
| train prompt   | (default / `--no-system-prompt`) | `..._trainprompt.json`  |
| GRPO prompt    | `--grpo-eval`         | `..._grpoprompt.json`        |
| bare prompt    | `--bare-question`     | `..._bareprompt.json`        |

Canonical output path: `outputs/<run>/checkpoint-X/eval_<bench>_full_<mode>.json`.

Headline eval of the GRPO run-2 checkpoint:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama_sft
export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
CKPT=/bulk/aacudad/reasoning_traces/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530
EVAL=/bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py

CUDA_VISIBLE_DEVICES=1 python "$EVAL" --checkpoint "$CKPT" --ds-mvtec-only --batch-size 4 \
    --output "$CKPT/eval_dsmvtec_full_trainprompt.json"     # -> DS-MVTec BA 82.73
CUDA_VISIBLE_DEVICES=2 python "$EVAL" --checkpoint "$CKPT" --visa-only    --batch-size 4 \
    --output "$CKPT/eval_visa_full_trainprompt.json"        # -> VisA BA 70.39
CUDA_VISIBLE_DEVICES=2 python "$EVAL" --checkpoint "$CKPT" --realiad-4k   --batch-size 4 \
    --output "$CKPT/eval_realiad4k_full_trainprompt.json"   # -> Held-out RealIAD BA 80.87
```

**Expected headline numbers per stage:**

| Model / stage                                   | DS-MVTec BA | VisA BA | Notes                         |
| ----------------------------------------------- | ----------- | ------- | ----------------------------- |
| Qwen2.5-VL-7B base                              | ~69.01      | 53.79   | no checkpoint / `--base-model`|
| 7B-Frozen-6K SFT (ckpt-564) — headline SFT      | **80.16**   | 64.78   | acc 80.36, F1 85.76           |
| Arm-C SFT (ckpt-376) — best SFT                 | 82.80       | 72.07   | GRPO/STaR init                |
| **GRPO Run-2 (ckpt-530)** — headline RL         | **82.73**   | 70.39   | held-out RealIAD 80.87        |
| IAD-R1 recanon baseline                         | 81.92       | 71.34   | common-footing reference      |

Batch wrappers: [`run_eval_dsmvtec_all.sh`](scripts/04_eval/run_eval_dsmvtec_all.sh),
[`run_eval_visa_all.sh`](scripts/04_eval/run_eval_visa_all.sh),
[`run_eval_bareprompt.sh`](scripts/04_eval/run_eval_bareprompt.sh). IAD-R1 common-footing
re-eval: [`run_iadr1_recanon.sh`](scripts/04_eval/run_iadr1_recanon.sh) (uses `--grpo-eval`).
Summary table: [`summarize_evals.py`](scripts/04_eval/summarize_evals.py).

### Stage 5 — Figures

Figure scripts in [`scripts/05_figures/`](scripts/05_figures/) read the eval JSONs / source
images and write into the thesis `figures/` dir.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama_sft
python /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/scripts/05_figures/make_qual_figures.py
python /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/scripts/05_figures/make_fig_grpo_decoupling.py
```

- [`make_qual_figures.py`](scripts/05_figures/make_qual_figures.py) — qualitative anomaly/normal
  panels (original | red-GT-mask overlay) from `data/MMAD/DS-MVTec`.
- [`make_fig_grpo_decoupling.py`](scripts/05_figures/make_fig_grpo_decoupling.py) — the
  reward/accuracy decoupling figure for the GRPO-on-Arm-C estimator ablation.
- [`make_iter2_overlays.py`](scripts/05_figures/make_iter2_overlays.py) and the
  `build_*_html.py` reviewers under the same dir.

Both write to the thesis `figures/` path hardcoded near the top of each script — edit `OUT=` to
your thesis tree if different.

---

## 4. Recompute any reported BA

Every BA in the thesis/README/this runbook is reproducible from the eval JSON's `metrics`
block `{tp, tn, fp, fn}` with:

```
BA = 0.5 * ( TP/(TP+FN) + TN/(TN+FP) )
```

Use [`results/compute_ba.py`](results/compute_ba.py):

```bash
python /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/results/compute_ba.py \
  /bulk/aacudad/reasoning_traces/repository_tu_delft_vlms/results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json
# -> ... BA=82.73  acc=...  F1=...
```

It accepts multiple files. The full audited inventory of every JSON's BA is in
[`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt).

---

## 5. Known thesis-vs-data mismatches (do not be alarmed)

A handful of values in the thesis PDF do **not** match the JSONs and are documented as errors
(transcription typos, cross-contaminated table rows, and a fabricated Appendix-C per-product /
confusion-matrix table). The README and `docs/UNVERIFIED.md` carry the full catalogue. When in
doubt: **the JSON wins, and `compute_ba.py` is the arbiter.**
