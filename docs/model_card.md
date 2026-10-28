# Model Card — Reasoning-Enhanced VLMs for Explainable Industrial Anomaly Detection

> TU Delft MSc thesis, Adnane Acudad, 2026 —
> *"Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection."*
> Base family: **Qwen2.5-VL** (3B / 7B Instruct).

This card documents the trained checkpoints produced in the thesis: a supervised fine-tuning (SFT) line that
distills Gemini-2.5-Flash chain-of-thought reasoning into Qwen2.5-VL, and a reinforcement-learning (GRPO) line that
further sharpens the SFT model. All reported numbers are **balanced accuracy (BA)** recomputed directly from the
shipped eval JSONs (see [Evaluation & numbers](#evaluation--numbers)), so they are reproducible from this repository
alone.

> **Weights are NOT shipped in this repository.** Re-train via [`docs/`](.) / the runbook using the configs in
> [`configs/`](../configs) and the text datasets in [`traces/`](../traces), or fetch published weights from Hugging
> Face if/when uploaded. This repo ships scripts, configs, prompts, the **text** training corpora (LlamaFactory
> `{messages, images}` format, where image fields are *relative paths* to re-downloadable source images), and the
> eval JSONs that back every number below.

---

## TL;DR — headline checkpoints

| Model | Role | DS-MVTec BA | VisA BA | Held-out RealIAD BA | Backing eval JSON |
|---|---|---:|---:|---:|---|
| Qwen2.5-VL-7B-Instruct (base) | Reference / lower bound | **69.01** | **53.79** | — | [`results/qwen25vl_baseline_eval/`](../results/qwen25vl_baseline_eval) |
| **7B-Frozen-6K SFT, ckpt-564** | **Headline SFT** | **80.16** | **64.78** | — | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/`](../results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564) |
| Arm-C SFT, ckpt-376 | Teacher-ablation best; GRPO-on-C / iter-2 base | 82.80 | 72.07 | — | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/`](../results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376) |
| **GRPO Run-2, ckpt-530** | **Headline RL** | **82.73** | **70.39** | **80.87** | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/`](../results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530) |
| IAD-R1 (Qwen) re-canon baseline | External RL baseline | 81.92 | 71.34 | — | [`results/iad_r1_qwen_recanon/`](../results/iad_r1_qwen_recanon) |

BA is defined as `BA = 0.5 * (TP/(TP+FN) + TN/(TN+FP))`, computed from the `metrics` block `{tp, tn, fp, fn}` of each
eval JSON. Recompute any row with [`results/compute_ba.py`](../results/compute_ba.py):

```bash
python results/compute_ba.py results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json
```

The full ground-truth BA inventory across all runs/checkpoints/prompt-modes is in
[`results/eval_ba_inventory.txt`](../results/eval_ba_inventory.txt).

---

## Intended use

- **Primary task:** *explainable* binary industrial anomaly detection on single product images — the model emits a
  free-text reasoning trace inside `<think>...</think>`, then a verdict (anomaly / normal). When anomalous, it also
  describes the defect type and location. The reasoning is the product, not just the label.
- **Intended users:** researchers and practitioners studying reasoning distillation, RL for VLMs, and explainable
  visual inspection. Useful as a starting point for domain-specific inspection assistants where a human reviews the
  model's explanation.
- **Domain:** the models are trained and evaluated on consumer/industrial part imagery (Real-IAD source for training;
  MVTec-AD / VisA / RealIAD families for evaluation). They are **not** general-purpose VLMs and are not tuned for
  medical, satellite, document, or other out-of-domain imagery.

### Out of scope / not intended for

- Safety-critical, autonomous, or unsupervised pass/fail gating without a human in the loop.
- Pixel-precise segmentation or bounding-box localization (the models give *textual* defect descriptions, not masks).
- Domains far from the training distribution; expect substantial degradation (see [Limitations](#limitations)).

---

## Model details

- **Architecture:** Qwen2.5-VL-Instruct (vision encoder + multimodal projector + LLM decoder). Two sizes were trained:
  **7B** (all headline results) and **3B** (smaller variants / ablations).
- **Base checkpoint:** `Qwen/Qwen2.5-VL-7B-Instruct` (and `-3B-Instruct`).
- **Languages:** English reasoning traces.
- **Training framework:** LlamaFactory for SFT (`llamafactory-cli train <yaml>`); a custom GRPO trainer derived from
  IAD-R1 / AnomalyR1 for RL.
- **Compute:** 2× RTX A6000, DeepSpeed ZeRO-3 with CPU offload, bf16.

---

## Training data — AnomalyThink

All training reasoning traces are **Gemini-2.5-Flash, C1-only** chain-of-thought distillations (~14,472 traces total,
referred to as **"AnomalyThink-15K"**, ~15K). They are partitioned into **3 disjoint, stratified
(product × normal/anomaly) splits**:

| Split | Size | Purpose |
|---|---:|---|
| SFT (6K) | 6,000 | Headline SFT corpus (~30 distinct products) |
| GRPO | 4,236 (2,118 anomaly + 2,118 normal) | RL training (23 products) |
| Held-out RealIAD | 4,236 | OOD evaluation only (23 products) |

Key facts (verified against the result files):

- The **6K SFT split is a 100% subset** of the ~14,472-trace union. Both the 6K and the full 15K corpora have the
  **same mean trace length (~141 words)**.
- Consequently, **"6K beats 15K" is a data-count / composition effect at fixed compute** — *not* a length/verbosity
  effect and *not* the result of a quality filter.
- **Generation recipe:** Gemini-2.5-Flash, temperature 1.0, batches of 5–10, target 120–200 words inside `<think>`,
  3 images per anomaly (original + overlay + reference) and 1 image per normal.
- **Quality control:** in-prompt auto-reject rules **plus** JSON / Pydantic schema validation. There is **no**
  second-reviewer model, **no** GPT-5-mini stage, and **no** 8-rubric filter.
- **Source imagery:** Real-IAD (not redistributed here; re-downloadable). Appendix-A nominally lists a 25-train / 5
  held-out per-product split; actual products-per-split differ from that table (see split counts above).

The shipped text corpora live under [`traces/`](../traces): `anomalythink_6k/`, `anomalythink_15k/`,
`teacher_ablation_abc/` (Arm A/B/C), `variety_star_6k/`, and `iter2/`. Dataset registration for LlamaFactory is in
[`configs/dataset_info.json`](../configs/dataset_info.json).

> **Variety STaR (in progress; thesis placeholder).** A newer self-taught-reasoner loop — rollout Arm-C ckpt-376 →
> Gemini judge → correct/rewrite → SFT on a 6K stratified variety corpus (3,000 anomaly / 3,000 normal, 160 products).
> Datasets are shipped under [`traces/variety_star_6k/`](../traces/variety_star_6k), but no headline numbers are
> claimed for it.

---

## Training recipe

### SFT (supervised fine-tuning)

Config: [`configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml`](../configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml)
(headline 7B-Frozen-6K). Other SFT YAMLs are in [`configs/sft/`](../configs/sft).

| Setting | Value |
|---|---|
| Finetuning type | Full fine-tuning (`finetuning_type: full`) |
| Vision tower | **Frozen** (`freeze_vision_tower: true`) |
| Multimodal projector / LLM | Trained |
| Learning rate | **1e-5**, cosine schedule, 20 warmup steps |
| Epochs | **4** |
| Per-device batch × grad-accum | 4 × 4 (effective batch 16, single GPU process) |
| Precision | bf16, gradient checkpointing |
| DeepSpeed | ZeRO-3 CPU offload |
| Template | `qwen2_vl`, cutoff 12,144 tokens |

> **Note on the "LR 2e-5 / frozen-ViT full-FT" recipe summary.** The shipped headline 7B-Frozen-6K config uses
> `learning_rate: 1.0e-5`. If you are reconciling with a recipe-summary string that cites *2e-5*, treat the value in
> the YAML as authoritative for that run, and check the specific YAML for any other variant. Frozen ViT + full FT +
> 4 epochs is correct across the SFT line.

The headline SFT (ckpt-564) is the **base/reference for GRPO**.

### GRPO (RL)

Run script: [`scripts/02_grpo/run_grpo_7b_resume_run2.sh`](../scripts/02_grpo/run_grpo_7b_resume_run2.sh). The
authoritative hyperparameters below are read from the saved **`training_args.bin`** of Run-2 (these supersede any
older thesis values such as G=2, beta=0.04, eta=5e-6, or "13K samples", which are wrong):

| Setting | Value |
|---|---|
| Base / reference policy | 7B-Frozen-6K SFT **ckpt-564** |
| `num_generations` (G) | **4** |
| KL coefficient (beta) | **0** — KL is **monitored** via Schulman k3, **not** added to the loss |
| Learning rate (eta) | **1e-6** |
| Clip epsilon | **0.2** |
| Training samples | **4,236** (2,118 anomaly + 2,118 normal) |
| Max steps / save / eval | 1,060 / every 250 / every 250 |
| Per-device BS × GA → EBS | 2 × 4 → 8 |
| Max prompt / completion | 2,048 / 512 tokens |
| Precision / seed | bf16 / 42 |
| Hardware | 2× RTX A6000, ZeRO-3 CPU offload |

**Reward (4-component), weights `(w_f, w_a, w_t, w_l) = (0.3, 0.3, 0.2, 0.2)`:**

- `w_f` — format reward (well-formed `<think>` + answer structure).
- `w_a` — accuracy reward (correct anomaly/normal verdict).
- `w_t` — defect-type reward, scored against a type embedding from `nomic-embed-text-v2-moe` (server on :5200).
- `w_l` — defect-location reward.
- **Reasoning reward** (LLM-judge server on :5100) was **not used** in Run-2 → it contributes a flat 0.5
  ("no reasoning reward").

> Run-2 resumes from `grpo_..._run1/checkpoint-530` for 2 more epochs; **ckpt-530 of Run-2** is the headline RL
> checkpoint.

---

## Evaluation & numbers

**Harness sizes:** DS-MVTec full `n=1670`; VisA full `n=2141`; held-out RealIAD `n=4236`.

**Prompt-mode contract = filename suffix.** Each eval JSON encodes which prompt was used:
`_trainprompt` / `_grpoprompt` / `_bareprompt`. Path convention:
`results/<run>/checkpoint-X/eval_<bench>_full_<mode>.json`. **Always compare like-for-like prompt modes** — mixing
modes (or a `_default` variant) changes BA by several points and is the single biggest source of apples-to-oranges
comparisons.

### Full per-model table (BA, recomputed)

| Model | Checkpoint | DS-MVTec BA | VisA BA | Held-out BA | Eval JSON dir |
|---|---|---:|---:|---:|---|
| Qwen2.5-VL-7B base | — | 69.01 | 53.79 | — | [`results/qwen25vl_baseline_eval/`](../results/qwen25vl_baseline_eval) |
| **7B-Frozen-6K SFT (headline)** | ckpt-564 | **80.16** | **64.78** | — | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/`](../results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564) |
| Arm-C SFT | ckpt-376 | 82.80 | 72.07 | — | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/`](../results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376) |
| **GRPO Run-2 (headline RL)** | ckpt-530 | **82.73** | **70.39** | **80.87** | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/`](../results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530) |
| IAD-R1 (Qwen) re-canon | — | 81.92 | 71.34 | — | [`results/iad_r1_qwen_recanon/`](../results/iad_r1_qwen_recanon) |
| 7B-Frozen-15K SFT (ep3) | ckpt-1359 | 71.66 | 64.28 | — | [`results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/`](../results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359) |
| 7B-Unfrozen-15K SFT (ep4) | ckpt-1812 | 70.27 | 58.16 | — | [`results/sft_qwen25vl_7b_15k_unfrozen/checkpoint-1812/`](../results/sft_qwen25vl_7b_15k_unfrozen/checkpoint-1812) |
| 3B-Frozen-6K SFT (ep4) | ckpt-752 | 69.08 | 57.22 | — | [`results/sft_qwen25vl_3b_zeroshot_6k_frozen/checkpoint-752/`](../results/sft_qwen25vl_3b_zeroshot_6k_frozen/checkpoint-752) |
| 3B-Frozen-15K SFT (ep4) | ckpt-1812 | 68.65 | 64.85 | — | [`results/sft_qwen25vl_3b_15k_frozen/checkpoint-1812/`](../results/sft_qwen25vl_3b_15k_frozen/checkpoint-1812) |
| 3B-Unfrozen-15K SFT (ep4) | ckpt-1812 | 68.58 | 59.60 | — | [`results/sft_qwen25vl_3b_15k_unfrozen/checkpoint-1812/`](../results/sft_qwen25vl_3b_15k_unfrozen/checkpoint-1812) |

All DS-MVTec / VisA rows above use the `_trainprompt` eval mode.

### Headline SFT confusion-matrix detail (DS-MVTec, ckpt-564, `_trainprompt`)

From the backing JSON `metrics` block: `TP 988 / TN 354 / FP 90 / FN 238` → **BA 80.16**, accuracy **80.36**,
F1 **85.76**, precision **91.65**, recall **80.59**.

---

## Limitations

- **Held-out OOD drop.** On the unseen RealIAD held-out split (`n=4236`), GRPO Run-2 reaches BA **80.87** with
  recall **0.683** (TP 1455 / FN 674) — markedly lower recall than on DS-MVTec, i.e. the model misses more
  anomalies on never-seen products. Expect degradation as the product distribution drifts from the training set.
- **Per-product variance is large.** On DS-MVTec, GRPO Run-2 per-product BA ranges from ~68 (`zipper` 68.84,
  `cable` 68.16, `capsule` 69.78) up to ~98 (`leather` 98.44, `wood` 97.50, `carpet` 95.51, `grid` 94.74). Aggregate
  BA hides this spread; evaluate on your own products before trusting the headline number.
- **Prompt sensitivity.** BA depends materially on prompt mode (`_trainprompt` vs `_grpoprompt` vs `_bareprompt`).
  Use the same prompt template at inference that the checkpoint was trained/evaluated with.
- **GRPO vs SFT is roughly a wash on the in-distribution benchmarks.** GRPO Run-2 (DS 82.73 / VisA 70.39) does not
  clearly beat the Arm-C SFT teacher (DS 82.80 / VisA 72.07) on DS-MVTec/VisA; the main GRPO payoff observed here is
  held-out generalization.
- **Reasoning is not independently verified.** Defect-type/location text is scored only by embedding similarity and
  format/accuracy rewards during training; the reasoning-judge reward was disabled in the headline RL run. Treat the
  explanations as plausibility-of-rationale, not certified ground truth.
- **Bias toward the training imagery.** Training imagery is sourced from Real-IAD; lighting, framing, and defect
  styles outside that distribution are under-represented.

---

## Provenance, caveats & known thesis mismatches

This card reflects **verified-from-disk** ground truth, which in several places **supersedes the thesis text**.
Documented discrepancies (see the repository `UNVERIFIED.md` for the full list):

- **`tab:baseline`** — Qwen2.5-VL-3B base DS-MVTec BA printed as 55.80; actual **56.14** (isolated transcription typo).
- **`tab:sft-summary`** — several cross-contaminated rows (epoch/size/frozen-vs-unfrozen mix-ups) corrected against the
  real eval JSONs; the **headline 7B-Frozen-6K ep3 (ckpt-564) row is fully correct** (80.16 / 64.78 / 80.36 / 85.76).
- **Appendix C** — the SFT+GRPO per-product column of `tab:perprod-full` and the entire `tab:perprod-cm` are
  **fabricated** (per-product counts physically impossible; rows do not sum to the genuine total). Only the column
  **Average (82.73)** and the CM **Total** are genuine. Base and SFT per-product columns are genuine.
- **Unverifiable rows** (no eval JSON on disk): `tab:sft-summary` 3B/7B-Unfrozen-**6K** (these configs were never run;
  rows duplicate 15K-unfrozen data); `tab:grpo-on-c` ckpt-795; `tab:grpo-estimator-ablation` step-0 "init" cells.
- **Superseded GRPO hyperparameters:** any reference to G=2, beta=0.04, eta=5e-6, or "13K samples" is OLD/WRONG — use
  the [GRPO recipe](#grpo-rl) values above (from `training_args.bin`).

---

## Reproduction & how to load

- **Environment:** conda env `llama_sft`:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate llama_sft
  export HF_HOME=/bulk/aacudad/reasoning_traces/hf_cache
  ```
- **Re-train SFT:** `llamafactory-cli train configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml`
- **Re-train GRPO:** `bash scripts/02_grpo/run_grpo_7b_resume_run2.sh` (custom IAD-R1-derived GRPO trainer).
- **Re-evaluate:** scripts in [`scripts/04_eval/`](../scripts/04_eval); recompute BA with
  [`results/compute_ba.py`](../results/compute_ba.py).
- **Source images** (MVTec / VisA / RealIAD / MMAD) and **model weights** are not shipped — re-download the datasets
  and re-train, or fetch weights from Hugging Face if published.

> **SECURITY — before any public push.** HF tokens and the Vertex service-account key are still present in some
> `download_*.py` / judge scripts. **Rotate all secrets, add a `.gitignore`, and run `gitleaks`** before the first
> public commit. Gemini access is via Vertex (`GOOGLE_APPLICATION_CREDENTIALS`) or API (`GEMINI_API_KEYS` in `.env`).

---

## Citation

```bibtex
@mastersthesis{acudad2026reasoning,
  title  = {Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection},
  author = {Acudad, Adnane},
  school = {Delft University of Technology},
  year   = {2026}
}
```
