# Evaluation Prompt-Mode Contract

This document specifies the **prompt-mode contract** of the evaluation harness for the
TU Delft MSc thesis *"Reasoning-Enhanced Vision-Language Models for Explainable Industrial
Anomaly Detection"* (Adnane Acudad, 2026; Qwen2.5-VL).

A "prompt mode" is the exact `(system message, user text)` pair fed to the model at
inference time. Because every model in this project is fine-tuned to a *specific* prompt,
the mode used at eval time must match the mode the model was trained under. **Mixing modes
silently corrupts cross-model comparisons** (see [§4](#4-why-mixing-modes-corrupts-comparisons)).

The single source of truth is the eval harness:
[`scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py`](../scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py).
All strings, flags and the filename convention below are copied verbatim from that file.

---

## 1. The three modes at a glance

| Mode | CLI flag | System message | User text (after `<image>`) | Filename suffix |
|---|---|---|---|---|
| **trainprompt** | `--no-system-prompt` | *(none)* | `\n` + the product preamble | `_trainprompt` |
| **grpoprompt** | `--grpo-eval` | *(none)* | `GRPO_EVAL_PROMPT` (IAD-R1 wording) | `_grpoprompt` |
| **bareprompt** | `--bare-question` | *(none)* | `Are there any defects in the query image?` | `_bareprompt` |

There is also a fourth, **rarely-used default** mode (no flag passed at all) that injects the
system message `"Please answer by yes or no"` plus the product preamble. It is *not* part of
the headline contract and is recorded only as the `_trainprompt_default` suffix on a couple of
GRPO probe JSONs. **No thesis table uses it** — treat default mode as a debug artefact, not a
reportable mode.

The three flags are mutually exclusive in practice. In code (`run_inference_batch`) they are
evaluated in the order `bare_question` → `grpo_eval` → `no_system_prompt` → default, so the
first flag set wins.

---

## 2. The exact prompt strings (verbatim from the harness)

### 2.1 `trainprompt` — `--no-system-prompt`

The model receives **no system message**. The user turn is the single test image followed by
a per-product preamble built by `make_train_prompt(product_name)`:

```
<image>
Analyze the provided image of the {product_name}. Determine if there are any anomalies present. If an anomaly is detected, specify its type and location, and provide a detailed reasoning for your conclusion.
```

`{product_name}` is parsed from the `image_id` (e.g. `DS-MVTec/zipper/test/...` → `zipper`).
A leading `\n` is prepended to the preamble (the literal `"\n{prompt}"` in the code). This is
the mode that elicits the full `<think>…</think><type>…</type><location>…</location><answer>…</answer>`
trace, and it is the **headline mode for the SFT models**.

### 2.2 `grpoprompt` — `--grpo-eval`

No system message. The user turn is the test image plus the fixed IAD-R1 question
(`GRPO_EVAL_PROMPT`), reproduced exactly:

```
<image>
You are an expert in detecting defects in image. Your task is to detect if there are any defects in the test image.Are there any defects in the query image?
```

(The missing space after `image.` and `test image.` is intentional — it reproduces the exact
string the GRPO trainer used at rollout time, character for character. Do not "fix" it.)
This is the mode the **GRPO models were trained under** and the mode used to re-score the
released IAD-R1 checkpoint on a common footing.

### 2.3 `bareprompt` — `--bare-question`

No system message. The user turn is the test image plus only the bare question:

```
<image>
Are there any defects in the query image?
```

This matches the **SFT-Iter2 (rollout-and-filter)** training prompt exactly, where the
rollout writer emitted the user message as `"<image>\n{question}"`.

### 2.4 default mode (no flag) — *not reportable*

A system message is injected and the train preamble is reused:

```
[system] Please answer by yes or no
[user]   <image>
         Analyze the provided image of the {product_name}. Determine if there are any anomalies present. ...
```

Used only for ad-hoc probes; see the `_trainprompt_default` note in §1.

---

## 3. Mode selection and the filename-suffix convention

### 3.1 How the flag is selected

The flags are plain `argparse` switches on the harness
([`evaluate_qwen25vl_7b_trainprompt.py`](../scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py)):

```
--no-system-prompt   # trainprompt: exact SFT training format, no system message
--grpo-eval          # grpoprompt:  exact GRPO training prompt (IAD-R1 style, no system message)
--bare-question      # bareprompt:  bare "Are there any defects in the query image?"
(no flag)            # default:     system "Please answer by yes or no" + preamble  (debug only)
```

Note the script *filename* says `trainprompt`, but the file implements **all four** modes —
the name is historical, not a restriction. The mode is chosen at the command line, never by
which script you run.

### 3.2 The suffix is the contract

The harness does **not** record the mode inside the JSON body. The mode is carried entirely by
the **output filename suffix**, which the caller sets via `--output`. The convention is:

```
outputs/<run>/checkpoint-<N>/eval_<bench>_full_<mode>.json
```

where `<bench> ∈ {dsmvtec, visa, realiad4k}` and `<mode> ∈ {trainprompt, grpoprompt, bareprompt}`.
Examples on disk:

```
outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json
outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json
outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json
outputs/grpo_qwen25vl_7b_g2rpo_v2_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json
outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_bareprompt.json
```

**The suffix is a naming discipline, not an enforced invariant.** Nothing in the harness
checks that the suffix matches the flag actually passed. Two consequences you must respect:

1. **Always pass the suffix that matches the flag.** A file ending `_grpoprompt.json` must have
   been produced with `--grpo-eval`, etc.
2. **Watch out for two known mislabels in the repo:**
   - The IAD-R1 re-canonicalisation run
     ([`run_iadr1_recanon.sh`](../scripts/04_eval/run_iadr1_recanon.sh)) writes
     `eval_dsmvtec_full_trainprompt.json` / `eval_visa_full_trainprompt.json` but actually runs
     with `--grpo-eval`. The *contents* are grpoprompt; the *suffix* says trainprompt. The IAD-R1
     baseline numbers (DS 81.92 / VisA 71.34) are therefore **grpoprompt** numbers despite the
     filename — this is correct, because IAD-R1 is a GRPO model and grpoprompt is its native mode.
   - The GRPO run-2 directory additionally contains `eval_dsmvtec_full_trainprompt_default.json`
     (default mode, BA 80.74) alongside the real `_trainprompt.json` (BA 82.73). Only the latter
     is reportable.

Always recompute balanced accuracy from the `metrics` block rather than trusting the suffix
alone — use [`results/compute_ba.py`](../results/compute_ba.py) and cross-check against the
[ground-truth inventory](../results/eval_ba_inventory.txt).

### 3.3 Reference launchers

| Launcher | Mode it drives | Flag |
|---|---|---|
| [`run_eval_dsmvtec_all.sh`](../scripts/04_eval/run_eval_dsmvtec_all.sh) | trainprompt | `--no-system-prompt` (implicit via train format) |
| [`run_eval_visa_all.sh`](../scripts/04_eval/run_eval_visa_all.sh) | trainprompt | as above |
| [`run_eval_bareprompt.sh`](../scripts/04_eval/run_eval_bareprompt.sh) | bareprompt + grpoprompt | SFT-bare ckpts → `--bare-question`; GRPO ckpts → `--grpo-eval` |
| [`run_iadr1_recanon.sh`](../scripts/04_eval/run_iadr1_recanon.sh) | grpoprompt (mislabelled suffix) | `--grpo-eval` |

---

## 4. Why mixing modes corrupts comparisons

Every model in this project is fine-tuned against **one** prompt, and Qwen2.5-VL is highly
prompt-sensitive. Evaluating a model under a mode it was *not* trained on measures
instruction-following robustness, not detection capability. Concrete failure modes:

- **Output-format drift breaks the parser.** The trainprompt model emits the full
  `<think>…<answer>…</answer>` block; the harness extracts the verdict from `<answer>`
  (`extract_tags` → `normalize_answer`, with a `fuzzy_yes_no` fallback). Run a bare-question
  model under trainprompt and it may not emit the expected tags, so the parser falls back to
  fuzzy matching or scores a missing answer — depressing the number for reasons that have
  nothing to do with detection.
- **Decision threshold shifts.** The IAD-R1 wording (`grpoprompt`) primes a terser yes/no
  posture than the verbose "provide a detailed reasoning" trainprompt preamble; the same
  weights produce different TP/FP balances under the two prompts. The repo shows this directly:
  GRPO run-2 ckpt-530 scores DS 82.73 under `_trainprompt` but 82.53 under `_bareprompt` and
  80.74 under the `_default` variant — same checkpoint, different prompt, different BA.
- **Cross-model A/B becomes apples-to-oranges.** Comparing an SFT model (trainprompt) against a
  GRPO model (grpoprompt) is only fair if *both* are scored under each model's native mode, on
  the same data, with the same parser. That is exactly what the suffix discipline guarantees and
  what the IAD-R1 re-canon (grpoprompt for a GRPO model) and the SFT trainprompt evals enforce.

**Rule:** report each model under its *native* mode, and never compare numbers carrying
different suffixes as if they were the same metric. When you need a same-prompt comparison
(e.g. SFT vs. GRPO under one fixed prompt), pick one mode, evaluate *all* contenders under it,
and label every file with that single suffix.

---

## 5. Which mode each headline result / thesis table uses

All balanced-accuracy (BA) figures below are recomputed from the `metrics` block
(`BA = 0.5·(TP/(TP+FN) + TN/(TN+FP))`) and cross-checked against
[`results/eval_ba_inventory.txt`](../results/eval_ba_inventory.txt).

| Result / thesis table | Model | Native mode | DS / VisA BA | Eval JSON |
|---|---|---|---|---|
| Baseline (`tab:baseline`) | Qwen2.5-VL-7B base | trainprompt | 69.01 / 53.79 | `outputs/qwen25vl_baseline_eval/eval_dsmvtec_full_trainprompt.json` |
| SFT grid (`tab:sft-summary`) | all SFT checkpoints | trainprompt | — | `…/checkpoint-*/eval_<bench>_full_trainprompt.json` |
| **Headline SFT** | 7B-frozen-6K ckpt-564 | trainprompt | 80.16 / 64.78 | `outputs/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json` |
| Teacher-ablation best | Arm-C ckpt-376 | trainprompt | 82.80 / 72.07 | `outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json` |
| **Headline GRPO** (`tab:grpo-results`) | run-2 ckpt-530 | trainprompt | 82.73 / 70.39 | `outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json` |
| Compound experiments (`tab:…` caption) | — | trainprompt | — | `…/eval_<bench>_full_trainprompt.json` |
| Estimator-ablation / g2rpo runs | g2rpo, iter2 GRPO | grpoprompt | — | `…/eval_<bench>_full_grpoprompt.json` |
| IAD-R1 re-canon (SOTA compare) | released IAD-R1 Qwen | grpoprompt* | 81.92 / 71.34 | `outputs/iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json` |

\* The IAD-R1 re-canon **filename** says `trainprompt` but the run used `--grpo-eval`; see
§3.2. These are grpoprompt numbers, which is the correct native mode for a GRPO model.

**Summary of the convention used by the thesis:**

- **trainprompt** is the headline mode for the **baseline, SFT grid, headline SFT, Arm-C, the
  headline GRPO run-2, and the compound experiments**. All `tab:baseline`, `tab:sft-summary`,
  `tab:grpo-results` and the compound-experiment table numbers come from `_trainprompt.json`
  files.
- **grpoprompt** is the native mode for the **g2rpo / iter-2 advantage-estimator GRPO runs and
  for the IAD-R1 re-canonicalisation** (the latter mislabelled at the filename level only).
- **bareprompt** is the native mode for the **SFT-Iter2 (rollout-and-filter) checkpoints** and is
  used as a same-prompt cross-check on the headline GRPO run; it backs no headline thesis table
  on its own.

---

## 6. Relationship to the thesis text (divergence — RESOLVED 2026-06-12)

Earlier thesis drafts (Appendix B / `04_sft.tex` §sec:sft-data) described a **two-image**
system-prompted format ("*You are an expert industrial anomaly detector. You are given two
images…*") — a leftover from an abandoned one-shot design that was never used for any reported
number. **This has been corrected in the thesis**: Appendix B and Chapters 1/3/4/5/7/8 now
describe the actual zero-shot, single-image setup documented here (SFT: no system turn +
product-conditioned question; train-prompt eval: the same question + the one-line
"`Please answer by yes or no`" system message; GRPO: `single_img=1`, no system message,
fixed "query image" question). The numbers in
[`results/eval_ba_inventory.txt`](../results/eval_ba_inventory.txt) were always produced by the
single-image harness, so no reported number changed — only the description of how those numbers
were obtained. This document remains authoritative for the prompt-mode contract of the eval JSONs.

The thesis-side generation/inspector prompt (the structured trace generator, with its in-prompt
auto-reject rules) is preserved separately under
[`prompts/inspector_prompt.txt`](../prompts/inspector_prompt.txt) and
[`prompts/inspector_prompt_test_v2.txt`](../prompts/inspector_prompt_test_v2.txt); it is the
**generation** prompt, not an evaluation prompt mode, and is unrelated to the contract above.

---

## 7. Quick checklist for adding a new eval

1. Pick the model's **native** mode (trainprompt for SFT/headline-GRPO, grpoprompt for
   estimator/IAD-R1-style GRPO, bareprompt for rollout-filter SFT).
2. Pass the matching flag (`--no-system-prompt` / `--grpo-eval` / `--bare-question`).
3. Name the output `eval_<bench>_full_<mode>.json` with the **matching** suffix.
4. Recompute BA with [`results/compute_ba.py`](../results/compute_ba.py) and append to the
   inventory.
5. Never put two different suffixes in the same comparison table without re-running all
   contenders under one fixed mode.
