# CLAIMS_EVIDENCE.md — Provenance for every claim in the thesis

**Thesis:** *Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection* — Adnane Acudad, TU Delft MSc, 2026 (backbone: Qwen2.5-VL 3B/7B).

## Purpose

This document supports **every claim made in the thesis** by recording (a) how the claim was reached, (b) the exact experiment or procedure that answered the underlying question, and (c) the on-disk evidence file that backs it. The motivating question is the simple reproducibility test: *"if the text mentions 6K, how did we get to 6K?"* — and likewise for every number, design choice, and qualitative finding.

It is organised in two layers:

1. **Key design decisions — how we got here.** A narrative for each major fork in the project (the 6K SFT split, C1-only traces, frozen vs unfrozen ViT, the four-component GRPO reward, `G=4 / β=0 / η=1e-6`, the teacher-ablation Arms A/B/C, the STaR rollout→judge→correct/rewrite→SFT loop, and the Real-IAD-Variety extension). Each subsection states the question, the procedure that answered it, the evidence, and the conclusion — concrete enough for the next student to re-run the reasoning.
2. **Per-chapter claim tables.** One table per chapter, preserving every extracted claim, with columns *Claim | Where | How it was derived | Evidence | Status*.

### How to recompute the headline numbers

All balanced-accuracy (BA) figures in this repo are recomputable from the shipped eval JSONs. The metric is

```
BA = 0.5 * ( TP/(TP+FN) + TN/(TN+FP) )    # from the JSON "metrics" block {tp,tn,fp,fn}
```

Use [`results/compute_ba.py`](results/compute_ba.py) on any `results/<run>/checkpoint-X/eval_<bench>_full_<mode>.json`. A full pre-computed listing lives in [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt). Spot-checks (all exact):

| Model | File | BA |
|---|---|---|
| 7B-frozen-6K SFT ckpt-564 (headline SFT) | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json) | DS 80.16 (TP988 TN354 FP90 FN238) |
| GRPO Run-2 ckpt-530 (headline RL) | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json) | DS 82.73 (TP971 TN383 FP61 FN255) |
| Arm-C SFT ckpt-376 (single best detector) | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json) | DS 82.80 (TP1003 TN372 FP72 FN223) |

> **Note on paths.** The original training tree used `outputs/<run>/…`. The shipped repo mirrors those eval JSONs under [`results/<run>/…`](results/). Where a claim cites an `outputs/…` path that is not shipped (rotated-away checkpoints, probe CSVs, training logs), the *Evidence* column says so and the status reflects that it cannot be reproduced from the public repo alone.

### Status legend

| Status | Meaning |
|---|---|
| **supported** | Claim reproduces exactly (or within rounding) from a shipped/verified artefact, or is a sound restatement of cited literature. |
| **partially-supported** | The direction/core of the claim holds, but a magnitude, attribution, or sub-claim is loose, off by a small margin, or rests on an unlogged estimate. |
| **unverified** | No on-disk artefact confirms it, **or** the on-disk artefact *contradicts* it (e.g. an anecdotal hyperparameter, a fabricated appendix table). Read the *How it was derived* cell — "unverified" here often means "actively wrong per the data," not merely "unchecked." |

### Headline ground-truth inventory (authoritative — trust over thesis text)

| Quantity | Value | Source |
|---|---|---|
| Corpus total | ~14,472 Gemini-2.5-Flash **C1-only** traces ("AnomalyThink-15K") | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) (len 14,472, verified) |
| Disjoint stratified splits | **6,000 SFT / 4,236 GRPO / 4,236 held-out RealIAD** (sum = 14,472) | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) (6,000); the file GRPO trained on is [`traces/grpo_split/grpo_train.json`](traces/grpo_split/grpo_train.json) (4,236, 2,118 + 2,118); held-out [`traces/iter2/heldout_4236_disjoint.json`](traces/iter2/heldout_4236_disjoint.json) |
| Mean `<think>` length | 136.8 words for both 6K and 15K (164.7 tokens on the 6K split with the Qwen tokenizer, p5 142, p95 187, range 121 to 223) | recompute 2026-09-06 on the combined files |
| Generation | Gemini-2.5-Flash, temp 1.0, batch 5–10, 120–200 word target, 3 images/anomaly (orig+overlay+ref), 1/normal | [`scripts/00_generate/config_mmad.py`](scripts/00_generate/config_mmad.py); [`scripts/00_generate/generate_realiad_traces.py`](scripts/00_generate/generate_realiad_traces.py) |
| QC | **in-prompt auto-reject + JSON/Pydantic schema only** — NO second-reviewer model, NO GPT-5-mini, NO 8-rubric filter | `generate_realiad_traces.py` Pydantic `BatchReasoningTraces` validation |
| Eval harness | DS-MVTec n=1670; VisA n=2141; held-out RealIAD n=4236; prompt mode = filename suffix `_trainprompt`/`_grpoprompt`/`_bareprompt` | recomputed n from JSONs; [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt) |
| GRPO config (from saved run-2 `training_args.bin`) | base/ref = ckpt-564; G=4; β=0 (KL monitored via Schulman k3, **not** in loss); η=1e-6; ε=0.2; 4,236 prompts (2,118+2,118); max 1,060 steps; reward weights (w_f,w_a,w_t,w_l)=(0.3,0.3,0.2,0.2); reasoning reward unused (flat 0.5); type-embed = nomic-embed-text-v2-moe; bf16; seed 42; 2×RTX A6000 ZeRO-3 CPU offload | saved `training_args.bin`; [`scripts/02_grpo/run_grpo_7b_resume_run2.sh`](scripts/02_grpo/run_grpo_7b_resume_run2.sh) |

> **Corrected-but-don't-forget.** Earlier drafts/notes used `G=2`, `β=0.04`, `η=5e-6`, or "13K samples." **Those are wrong.** The authoritative values are the saved `training_args.bin` figures above.

---

# Part 0b — Verification pass and additions (2026-09-06)

Source for everything in this block: [`docs/verification/NUMBER_VERIFICATION_REPORT.md`](docs/verification/NUMBER_VERIFICATION_REPORT.md) and the 2026-09-06 entry of [`NUMBER_PROVENANCE.md`](NUMBER_PROVENANCE.md).

| Claim (thesis location) | Value | Evidence | Status |
|---|---|---|---|
| The KCR parity reproduces on LLaVA-OneVision-7B-SI with a clean corpus (§6.9, Appendix M) | KCR corrected 87.32 / 72.65 at ep2, 74.29 VisA at ep4, against SFT+GRPO 87.66 / 72.38 and restart 87.86 / 71.89 | `results/sft_llava_ov_7b_frozen_iad_sft_llava_iter1_C_original/`, `results/grpo_llava_ov_from_ep1{,_ep2}/`, corpus `traces/llava_kcr/sft_llava_C_original_train.json` (all 6,000 images in the SFT split, 3,000 / 3,000) | supported |
| GRPO is the stage that teaches localisation (§6.10.4, §7.5, RQ2) | of detected anomalies: Qwen 73.2 / 38.1 → 82.4 / 55.9 → 74.9 / 52.7; LLaVA 77.5 / 54.2 → 83.1 / 71.2 → 83.1 / 63.7 | `results/thesis_figure_data/loc_hit.json`, `scripts/05_figures/thesis_figures_v2/loc_hit_table.py` | supported |
| The `<type>` tag does not transfer to MMAD's label vocabulary (Appendix J.5) | similarity 0.53 to 0.58 for every stage, match below 13 %, 18 predicted strings vs 435 / 218 labels | `results/thesis_figure_data/type_sim.json` | supported |
| GRPO on the strong Arm-C initialisation did not improve the verdict under the production recipe (§6.8) | every checkpoint 1.6 to 4.0 below on DS-MVTec, reward 1.99 → 2.39, peak 3.25 | `results/grpo_qwen25vl_7b_abc_C_grpo/` | supported |
| A prompt-aligned beta 0.1 variant is inconclusive (§6.8, §8.4) | one of fifteen checkpoints at 82.95 / 72.62 (+0.15 / +0.55), run ends at 81.25 / 71.51, single seed | `results/grpo_sftprompt_kl0.1_sys_3ep/` | partially-supported (the thesis says so explicitly) |
| The estimator is not the explanation (§6.8) | 18 probe cells, only G2RPO step 20 above init by 0.75 | `results/grpo_probe_{ctrl,drgrpo,g2rpo}/checkpoint-*/probe_*.json` | supported |
| Explanation quality on a shared image set (§6.11, RQ4, contributions 5 and 7) | paired gap to IAD-R1 2.96 [2.41, 3.50] DS and 2.15 [1.47, 2.83] VisA on identical images; among our seven rows the supervised-only rows are lowest on VisA (LLaVA SFT to SFT+GRPO +0.85 [0.26, 1.44]); DS LLaVA KCR highest (+0.52 [0.19, 0.85] over Qwen KCR) | `results/explainability_shared/` (90 + 47 shared images, nine rows) | supported; the earlier own-100 protocol (`results/explainability_multi/`) gave the same ordering against IAD-R1 and is no longer in the thesis |
| LLaVA-OneVision-Data contains MVTec-AD material, VisA none (§6.1) | 1,999 of 186,060 rows | `results/contamination_llava_ov_data/` | supported |
| Rollout pass rates 84.6 % / 90.8 % (§5.7) | over the 6,000 SFT images; whole pool 84.0 / 90.2 | `traces/rollout_pools/qwen_phase0_10k/rollouts_raw.jsonl.gz`, `answer_match` field | supported |
| Arm A is a balanced subset (§5.7) | balanced on the acquisition folder 1,489 / 1,489; by verdict 1,298 / 1,680 | `traces/teacher_ablation_abc/sft_A_kept_balanced.json` | supported as reworded in the thesis |
| 3,557 kept + 2,443 patched, 94.8 % anomalous (§6.7) | matching traces to rollouts gives 3,471 to 3,580 kept and 96 % anomalous among patched | `traces/iter2/`, `traces/rollout_pools/qwen_phase0_10k/` | partially-supported (no build log) |
| SFT gain over base is +11.2 pp overall (§1.4, §8.1) | +11.08 against the Table 6.1 baseline (69.08); +11.15 against the older 69.01 file | `results/baseline_named/` | supported, thesis switched to +11.1 pp and the Table 6.1 file on 2026-09-06 |
| 15K training loss is still descending at epoch 4 (old §7.2) | plateaus at 0.57 to 0.58 through epochs 3 and 4 | `results/sft_qwen25vl_7b_15k_frozen/checkpoint-1812/trainer_state.json` | removed from the thesis on 2026-09-06 |
| Mask area 1.0 % VisA / 3.8 % DS-MVTec, VisA downscaled about 2.5x per side (§7.3) | 1.00 % / 3.79 %, 2.39x at the median image | recomputed from the MMAD masks and images | supported |
| GPT-5-mini verifier passes 97.8 % of the 15K corpus (§7.2) | 14,158 of 14,472 | `results/verify_15k_regen/summary.json` | supported |

# Part 0 — Post-thesis claims (added 2026-09-02)

These claims are **not** in the submitted thesis PDF. They come from runs finished after submission
and they are the claims the README and the conference paper lean on. Same columns and same status
legend as the per-chapter tables in Part 2.

Throughout: the method is **Keep-Correct-Revise (KCR)**. Keep the rollouts the policy already gets
right, have a teacher correct the wrong ones and revise the ones that are right but weakly grounded,
then fine-tune the base model on the result. "Arm C" is the same thing under its ablation name (A =
keep only, B = keep + correct, C = keep + correct + revise = KCR) and it is what every path, dataset
key and results directory below uses.

## 0.1 Explanation quality (the number that changed)

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| KCR/Arm-C explanations score **9.05** on DS-MVTec and **8.52** on VisA, against **6.14** and **7.04** for IAD-R1 asked the same way. | README §1, conference paper explanation table | Gemini-3-Flash judge shown image + red GT-mask overlay + GT defect type + the model's trace. 5 axes scored 0/1/2, summed 0–10. **Median of 3 judge samples.** n=100 product-diverse traces per model per benchmark. Each model scored **on its own correctly-detected anomalies** (`gt=yes & pred=yes`). | [`results/explainability_multi/explainability_5axes_table.md`](results/explainability_multi/explainability_5axes_table.md), `summary.json`, `raw_results.json`, `raw_results_iadr1_tp.json`; script [`scripts/04_eval/explainability_judge_multi.py`](scripts/04_eval/explainability_judge_multi.py) | supported |
| The **older** 9.14 vs 4.29 (DS-MVTec) / 8.67 vs 6.40 (VisA) figures are superseded, not wrong-in-kind. | [`NUMBER_PROVENANCE.md`](NUMBER_PROVENANCE.md) 2026-09-02 entry, and the flagged 2026-06-16 entry | Old run scored IAD-R1 **under its own GRPO prompt**, which never asks for reasoning. About 35–40% of its answers are a bare "Yes" and score 0 on every axis. That measures explanation *reliability* under its own prompt, not explanation *quality*. It was also k=1, a single judge sample. | `outputs/explainability_judge/EXPLAINABILITY_EVAL.md` (source tree, not shipped); superseded row kept verbatim in `NUMBER_PROVENANCE.md` | supported |
| The prompt-matched row is the fair comparison for **reasoning quality**. | `results/explainability_multi/README.md` | Asked the same way as our models, 100% of IAD-R1's answers carry a trace, so the two models are scored on the same task. k=3 median damps judge noise. Per-model own-correct-detections design avoids penalising a model through a shrinking cross-model intersection. | [`results/explainability_multi/README.md`](results/explainability_multi/README.md) "The two IAD-R1 rows" section | supported |
| Both IAD-R1 rows should survive any rewrite, because they answer different questions. | same | Native prompt 4.50 / 6.32 = reliability. Prompt-matched 6.14 / 7.04 = quality. Quoting only 4.50 understates IAD-R1. Either way it stays clearly below KCR. | same table | supported |

## 0.2 Cross-architecture replication on LLaVA-OneVision-7B-SI

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| On LLaVA-OneVision-7B-SI the KCR corpus reaches **88.45 / 74.25**, above **87.66 / 72.58** for SFT+GRPO on the same backbone. | README §1 headline table, conference paper | Both trained and evaluated on the identical harness and subsets (DS-MVTec n=1670, VisA n=2141), `_trainprompt` mode, vLLM serving. BA recomputed from the `tp/tn/fp/fn` block with `results/compute_ba.py`. | KCR: [`results/sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/`](results/sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/) (DS tp1045 tn407 fp37 fn181; VisA tp942 tn659 fp285 fn255). GRPO: [`results/grpo_llava_ov_from_ep1/checkpoint-530/`](results/grpo_llava_ov_from_ep1/checkpoint-530/) (DS tp1078 tn388 fp56 fn148; VisA tp958 tn611 fp330 fn236) | supported |
| The KCR loop is backbone-agnostic. Only the rollout sampler changed. | README §1, `docs/` KCR loop note | The LLaVA post-rollout pipeline reuses the already-shipped `phase0_bucket.py`, `phase1a_gemini_judge.py` and `phase1b_gemini_correct.py` unchanged. Only the rollout front-end and the arm builder are new. | [`scripts/03_rollout_star/`](scripts/03_rollout_star/) | supported |
| The LLaVA KCR corpus is a **native** loop, not a Qwen corpus reused. | README §1 | Rollouts came from a LLaVA policy, the teacher corrected and revised LLaVA's own failures. Arms A 8,998 / B 9,124 / C 6,000, C at 50/50. | `Training/datasets_sft_llava_iter1/` (source tree); published as `llava_kcr/` in [`aacudad/AnomalyThink`](https://huggingface.co/datasets/aacudad/AnomalyThink) | supported |
| GRPO does work on this backbone. It just does not reach the corpus. | README §1 | GRPO init is LLaVA 6K SFT ep1 (85.91 / 68.26). ckpt-530 is 87.66 / 72.58, so +1.75 / +4.32 over its own init. | [`results/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/`](results/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/) and [`results/grpo_llava_ov_from_ep1/checkpoint-530/`](results/grpo_llava_ov_from_ep1/checkpoint-530/) | supported |
| **Contamination caveat.** DS-MVTec numbers for LLaVA-derived models carry a pretraining-exposure asterisk. VisA does not. | README §1, every LLaVA `NOTE.md` | `lmms-lab/LLaVA-OneVision-Data`, config `vision_flan(filtered)`, contains **426 rows** whose id matches `%MVTecAD%`. VisA matches **0**. Applies to IAD-R1's released model as well. | Verified live through the Hugging Face datasets-server filter; recorded in `NUMBER_PROVENANCE.md` 2026-09-02 entry and [`results/sft_llava_ov_7b_frozen_llava_iter1_C/NOTE.md`](results/sft_llava_ov_7b_frozen_llava_iter1_C/NOTE.md) | supported |
| **Do NOT claim** "beats IAD-R1 on its own backbone" against the 81.92 / 71.34 row. | — | That row is IAD-R1's released **Qwen2.5-VL-7B** checkpoint re-evaluated on our harness. We never re-evaluated their LLaVA-OneVision checkpoint on the full subsets. The comparison is same-harness, not same-backbone. | [`results/iad_r1_qwen_recanon/`](results/iad_r1_qwen_recanon/); see also §2.5.3 row in Part 2 | unverified (as a same-backbone claim) |

## 0.3 Corpus transfer to Qwen3-VL-8B-Instruct

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| The KCR corpus transfers to a newer, stronger backbone: **78.68 / 64.45 → 85.82 / 76.52** (+7.14 / +12.07). | README §1 | `Qwen/Qwen3-VL-8B-Instruct` from base, 4 epochs, frozen vision tower, config mirrors `sft_abc_C.yaml` except model, `template: qwen3_vl`, output dir and batch. Same harness and subsets. Best checkpoint is ckpt-376. | Base: [`results/qwen3vl_8b_baseline_eval/`](results/qwen3vl_8b_baseline_eval/) (DS tp880 tn380 fp64 fn346; VisA tp446 tn865 fp79 fn751). SFT: [`results/sft_qwen3vl_8b_armC/checkpoint-376/`](results/sft_qwen3vl_8b_armC/checkpoint-376/) (DS tp1055 tn380 fp64 fn171; VisA tp883 tn747 fp196 fn313) | supported |
| The corpus used is the same file the Qwen2.5-VL-7B headline Arm-C model trained on. | README §1 | LlamaFactory key `iad_sft_iter2` in both `sft_qwen3vl_8b_armC.yaml` and [`configs/sft/sft_abc_C.yaml`](configs/sft/sft_abc_C.yaml), 6,000 records. | [`traces/iter2/sft_iter2_train.json`](traces/iter2/sft_iter2_train.json) (len 6,000, verified) | supported |
| **There is no GRPO comparison on Qwen3-VL-8B.** | README §1, [`results/sft_qwen3vl_8b_armC/NOTE.md`](results/sft_qwen3vl_8b_armC/NOTE.md) | Nothing was RL-trained on this backbone. The corpus-versus-RL comparison exists only on Qwen2.5-VL-7B and LLaVA-OneVision-7B-SI. | absence of any `grpo_qwen3vl_*` run in `outputs/` and in `results/` | supported |

## 0.4 Labels-only control (how much of the gain is the reasoning?)

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| Stripping the reasoning supervision costs **-4.94 DS-MVTec and -3.43 VisA**. | README §1 | Same 6,000 images, same hyperparameters, same 4-epoch checkpoint grid as the reported SFT-6K run. Only the supervision changes: prompt asks for a bare verdict, target is `<answer>Yes\|No</answer>`. Best labels-only epoch 77.86 / 68.64 against Arm-C SFT 82.80 / 72.07. | [`results/sft_qwen25vl_7b_6k_noreason/checkpoint-188/`](results/sft_qwen25vl_7b_6k_noreason/checkpoint-188/) (`_noreasonprompt` mode); config [`configs/sft/sft_qwen25vl_7b_6k_noreason.yaml`](configs/sft/sft_qwen25vl_7b_6k_noreason.yaml); builder [`scripts/01_sft/build_noreason_dataset.py`](scripts/01_sft/build_noreason_dataset.py) | supported |
| Output format is not a confound in that control. | same | The `<answer>` wrapper is kept. The harness reads `<answer>` independently of `<think>`, so the parser is unchanged between the two arms. | `build_noreason_dataset.py` docstring; `--noreason-prompt` branch in the eval harness | supported |
| Fine-tuning on our images alone still helps a lot, so the reasoning is not the whole story. | README §1 | Base 69.01 / 53.79 to labels-only 77.86 / 68.64. The reasoning adds the last ~5 pp / ~3.5 pp on top of that. | [`results/qwen25vl_baseline_eval/`](results/qwen25vl_baseline_eval/) and the row above | supported |

---

# Part 1 — Key design decisions: how we got here

Each subsection answers: *what was the open question, what experiment/procedure resolved it, which file proves it, and what did we conclude.*

## 1.1 Why 6,000 SFT traces (and not the full ~14,500)?

**The question.** The corpus has ~14,472 traces. Should we train SFT on all of them, or on the smaller 6,000-trace stratified split? The naive "more data is always better" heuristic says use everything.

**The procedure.** Run the *same* 7B-frozen SFT recipe on (a) the 6,000-trace SFT split and (b) the full ~14,472-trace union ("15K"), at a matched 3-epoch operating point, then compare DS-MVTec BA. Crucially, the 6K split is a **100% subset** of the 14,472 union, and both have the **same mean `<think>` length (~141 words, ≈137 recompute)** — so any difference cannot be a verbosity/length effect, and (because QC is in-prompt only, with no second-pass scoring filter) it is not a quality-ranking filter either.

**The evidence.**
- 6K ckpt-564: DS **80.16** — [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json)
- 15K ckpt-1359 (ep3): DS **71.66** — [`results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/eval_dsmvtec_full_trainprompt.json) (inventory lists 71.66 at this ckpt)
- Subset/equal-length facts: 6K = 100% subset of [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json); equal mean `<think>` words (recompute).

**The conclusion.** 6K beats 15K by **~8.5 pp** at fixed compute. Because 6K is a same-length subset, the effect is **data-count / composition at fixed compute** — at a fixed epoch budget the 15K config is under-trained (2.41× more updates needed for equal epochs), and the extra ~8.5K traces (which are the GRPO + held-out splits, same generation recipe) add no benefit and slightly hurt. The 6,000-trace split is therefore the headline SFT corpus.

> **Caveat the next student must know.** The 8.5 pp gap holds against the **ep3 ckpt-1359** 15K number (71.66). The thesis discussion chapter at one point compares against a *corrected* later-epoch 15K value (~80.12) where the gap collapses to ~0 — that comparison is flagged **unverified** below (sec:disc-quality). The headline 8.5 pp claim is the ep3-vs-ep3 comparison. Also note the introduction mislabels this as "concise vs verbose," which is **wrong** (same length) — recorded as partially-supported.

## 1.2 Why C1-only traces?

**The question.** Real-IAD ships five camera angles per product. Should the trace corpus mix all five views or restrict to one?

**The procedure.** For one of the two main SFT runs, restrict generation to the **C1 camera angle** to reduce distribution variability; a C1-only-vs-all-5 ablation is set up in the SFT chapter. The C1-only variant is materialised on disk.

**The evidence.** [`traces/anomalythink_15k/c1_only_fixed/new_sft_c1_train.json`](traces/anomalythink_15k/c1_only_fixed/new_sft_c1_train.json) (4,236) and [`traces/anomalythink_15k/c1_only_fixed/combined_sft_c1_train.json`](traces/anomalythink_15k/c1_only_fixed/combined_sft_c1_train.json) (10,236) exist as the C1-restricted variants. The whole AnomalyThink-15K corpus is in fact C1-only per ground truth.

**The conclusion.** C1-only reduces nuisance variability (lighting/pose across the five views) so the model spends capacity on defect signal, not viewpoint. The headline corpora are C1-only.

> Watch out: `new_sft_c1_train.json` (n=4,236, mean `<think>` ≈417 words) is a **much longer rewritten variant** belonging to a different (variety/iter-2) generation, NOT the headline SFT corpus — do not confuse it with the 6K SFT pool.

## 1.3 Why freeze the vision encoder (ViT)?

**The question.** During SFT, should we fine-tune the ViT (unfrozen) or hold it fixed (frozen, training only the LM trunk + multimodal projector)?

**The procedure.** Four-factor SFT grid (size 3B/7B × frozen/unfrozen × 6K/15K × epoch) with **frozen vs unfrozen** as one axis. Compare matched cells, paying special attention to the harder VisA benchmark and to per-class recall.

**The evidence.**
- 7B-frozen-6K VisA recall 71.1% vs 7B-unfrozen-15K VisA recall 34.8–40.6% (recall = tp/(tp+fn) from VisA JSONs): [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json) vs `results/sft_qwen25vl_7b_15k_unfrozen/checkpoint-*/eval_visa_full_trainprompt.json` (inventory: unfrozen VisA BA 57.13–59.18 vs frozen 64.28–66.94).
- Config that defines "frozen": [`configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml`](configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml) — `freeze_vision_tower: true`, `freeze_multi_modal_projector: false`, `freeze_language_model: false`.

**The conclusion.** Frozen wins consistently, most visibly on VisA (the unfrozen model collapses on VisA defect recall). Mechanism: the per-token SFT loss is dominated by language tokens, so the ViT receives only an indirect gradient through the connector and overfits/destabilises on small data — consistent with OpenVLThinker. Frozen is the headline setting.

> Caveat: the *DS-MVTec* magnitude of the frozen advantage is small once the cross-contaminated 15K-unfrozen rows are corrected (corrected 7B-Frozen-15K DS 68.65 vs 7B-Unfrozen-15K DS 68.58). The robust, large gap is on **VisA recall**, not DS BA. Also: the **6K-unfrozen** cells (3B and 7B) were **never run** — those table rows duplicate 15K-unfrozen data, so the "effect of freezing at 6K" is not backed by a real run.

## 1.4 The four-component GRPO reward (and why the reasoning reward is a flat 0.5)

**The question.** What reward should drive GRPO for explainable IAD, given that a bare binary-verdict reward is too sparse?

**The procedure.** Decompose the reward into four bounded-in-[0,1] components summed to max 1.0 with weights **(format 0.3, accuracy 0.3, type 0.2, location 0.2)**:
- **Format (0.3):** label-conditioned regexes — anomalous samples earn partial credit per required tag (`+0.075/tag`); normal samples are penalised (`-0.1`) for forbidden `<type>`/`<location>`. Catches free-form prose and spurious `<type>None</type>`.
- **Accuracy (0.3):** binary verdict match.
- **Type (0.2):** graded semantic similarity (exact 1.0 / substring 0.8 / else cosine) via **nomic-embed-text-v2-moe** on CPU (server :5200); only when gold is anomalous and both emit `<type>`. Binary equality is too sparse for synonyms (Scratch/Surface scratch/Abrasion).
- **Location (0.2):** high-recall **presence** reward (0.2 for any `<location>` value) — deliberately **not** IAD-R1's 3×3 grid match, which a pilot found brittle to paraphrase ("top-right" vs "upper right corner").

The **reasoning reward** (a Gemini-judge server on :5100) was implemented but **not used** in the production run — it is emitted as a flat 0.5, i.e. "no reasoning reward." GRPO is thus *Gemini-judge-free*; all four live rewards are local.

**The evidence.** Reward weights and the unused judge: ground-truth `training_args.bin` and reward source ([`scripts/02_grpo/stage_rl/reward.py`](scripts/02_grpo/stage_rl/reward.py), `reward_process/`); type-embed server [`scripts/02_grpo/gemini_judge_server_v2.py`](scripts/02_grpo/gemini_judge_server_v2.py) loads `nomic-ai/nomic-embed-text-v2-moe` float32. Arithmetic 0.3+0.3+0.2+0.2 = 1.0.

**The conclusion.** Format + accuracy (0.6 combined) are the non-negotiable structure + headline verdict; type/location (0.2 each) are auxiliary signals that cannot flip a correct verdict. All bounded so the group-normalised advantage is dimensionless. The reasoning reward was dropped to keep the run judge-free and reproducible.

## 1.5 Why G=4, β=0, η=1e-6?

**The question.** What GRPO hyperparameters keep the policy stable on a 7B VLM under a 2×A6000 budget?

**The procedure / evidence (all from saved run-2 `training_args.bin`).**
- **G=4 (num_generations).** Larger G lowers advantage variance (DeepSeekMath uses G=64), but each rollout on a 7B VLM under ZeRO-3 CPU offload is ~30 s; the run log shows **83.85 s/step** for G=4. G=4 is the largest group that keeps per-step cost within budget.
- **β=0 (KL coefficient).** KL is **monitored** via Schulman's unbiased **k3** estimator but **not added to the loss** — `trainer_state.json` logs `kl` as a separate metric and the loss is ~0. Over the 2-epoch budget the monitored KL stayed bounded (min 0.0, max 0.117), so the policy never drifted dangerously.
- **η=1e-6 (learning rate).** GRPO is far more LR-sensitive than SFT; at 1e-6 KL stays small while combined reward climbs monotonically (1.66→2.13). No high-LR ablation JSON is on disk, so the "higher LR drifts" half is design rationale, not a logged experiment.

Files: saved `training_args.bin`; [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/trainer_state.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/trainer_state.json); [`scripts/02_grpo/run_grpo_7b_resume_run2.sh`](scripts/02_grpo/run_grpo_7b_resume_run2.sh).

**The conclusion.** `G=4 / β=0 / η=1e-6 / ε=0.2`, 2 epochs over 4,236 prompts (max 1,060 steps), is a deliberately *minimal* recipe — and it still beats SFT-Iter1. Neither iter-2 nor G2RPO beats it.

## 1.6 The teacher ablation: Arms A / B / C, and why C

**The question.** Holding the 6,000-image pool fixed, how much does each *layer* of teacher (Gemini) intervention improve the SFT corpus?

**The procedure.** Three corpora that vary only the teacher intervention ladder, each trained from base with frozen ViT, best epoch selected on DS-MVTec:
- **Arm A** — self-distillation only, **no Gemini**: 2,978 model-written items ([`traces/teacher_ablation_abc/sft_A_kept_balanced.json`](traces/teacher_ablation_abc/sft_A_kept_balanced.json), len 2,978).
- **Arm B** — A + Gemini-corrected NG traces (rationalisation-on-failure / STaR): 4,369 items ([`traces/teacher_ablation_abc/sft_B_kept_corrected.json`](traces/teacher_ablation_abc/sft_B_kept_corrected.json), len 4,369).
- **Arm C** — STaR + Saunders critique-and-revise: 6,000 items (50/50), the richest intervention.

**The evidence.** Best-epoch DS-MVTec: A 79.01, B 80.75, **C 82.80** ([`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json), recompute 82.80 / VisA [`…/eval_visa_full_trainprompt.json`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_visa_full_trainprompt.json) 72.07). VisA ordering is **C > A > B** (72.07 > 70.21 > 68.73) — Arm B's extra Gemini-corrected NG traces *hurt* VisA.

**The conclusion (finding O3).** On DS-MVTec **C > B > A**, each layer of Gemini intervention buying ~2 pp. **Arm C SFT alone (82.80/72.07) is the single best detector in the thesis** — it matches the SFT+GRPO model on DS (within 0.07) and beats it on VisA (+1.68), and beats the re-canonicalised IAD-R1 baseline (81.92/71.34) on **both** subsets (+0.88 DS / +0.73 VisA) **using SFT alone**. Arm C ckpt-376 is also the iter-2 / GRPO-on-C base.

## 1.7 The STaR loop: rollout → judge → correct/rewrite → SFT

**The question.** Can the model improve itself by learning from its own rollouts, corrected by the teacher (the STaR / critique-and-revise recipe that produced Arm C, and the iter-2 self-distillation loop)?

**The procedure.**
- **Teacher-ablation (Arm C) form:** roll out, keep correct self-traces (RAFT-style rejection sampling), and have Gemini *rationalise-on-failure* (STaR) and *critique-and-revise* (Saunders) the rest, then SFT from base.
- **Iter-2 form:** roll out from GRPO ckpt-530, **reject-sample** (verdict matches gold AND XML passes the format regex) → 4,026 self-traces, **reset to 7B base**, fresh SFT (frozen ViT, η=1e-5, batch 4, 4 epochs), then GRPO again. Two filter variants: v1 (first pass) and v2 (stricter format match + de-duplication).

**The evidence.** Filtered self-trace count: [`traces/iter2/iter2_v2_sft_train.json`](traces/iter2/iter2_v2_sft_train.json) len **4,026**. Iter-2 GRPO best DS = **80.64** ([`results/grpo_qwen25vl_7b_iter2_v2_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json`](results/grpo_qwen25vl_7b_iter2_v2_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json)) < single-stage **82.73**.

**The conclusion.** The Arm-C critique-and-revise loop is the *best* corpus (82.80). But the *iter-2* SFT↔RL loop **does not beat single-stage GRPO** on the narrow IAD task (80.64 < 82.73): the rollout/filter concentrates on easy examples, and a later balanced-pool refinement collapsed by catastrophic forgetting (322/1670 unparsable at ep1). Single-stage is retained as the strongest GRPO checkpoint; remedy proposed = difficulty-aware filter + mix-in of original Gemini traces.

## 1.8 The Real-IAD-Variety extension (in progress — thesis placeholder)

**The question.** Does the exact Arm-C STaR recipe scale from 30 Real-IAD products to the 160-category Real-IAD-Variety view?

**The procedure.** Roll out the Arm-C 82.80 checkpoint at k=8 over the 160-category C1 view → Gemini judge → correct/rewrite → assemble a 6,000-trace SFT corpus stratified by product × normal/anomaly (**3,000 OK / 3,000 NG, 160 products**).

**The evidence.** [`traces/variety_star_6k/variety_star_sft_6k.json`](traces/variety_star_6k/variety_star_sft_6k.json) exists with **6,000** traces, 160 distinct products, exactly 3,000/3,000. **Eval results are explicitly a placeholder** — the `phase0_variety_star` rollout dirs are empty on disk.

**The conclusion.** Pipeline and corpus exist and match the Variety-STaR description; numeric results are deferred (future work F5). Any Variety performance claim in the thesis is a placeholder, not a measured result.

---

# Part 2 — Per-chapter claim tables

Every extracted claim is preserved. *Where* uses the location given in the chapter; *Evidence* uses repo-relative links where a shipped file backs the claim, and names the off-repo artefact otherwise.

## Chapter 00 — Abstract

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| Classical AD detectors saturate ~99% AUROC on MVTec AD but emit only verdict+heatmap, no language explanation. | para 1, L7 | Literature framing (PatchCore/normalising-flow/reconstruction all ~99% AUROC, score+heatmap only). | Ch.02 lit review; standard benchmarks. Not a thesis experiment. | supported |
| SFT on six-phase structured CoT traces (Framing/Scan/Focus/Evaluate/Alternatives/Decide) emitting XML `<think>/<type>/<location>/<answer>`, then GRPO. | para 2, L9 | Method description; format/template matches Ch.03 and the generation config. | Ch.03; gen config (Gemini-2.5-Flash, 120–200 word `<think>`, Pydantic schema). | supported |
| ~14.5K Gemini traces from Real-IAD, partitioned into disjoint stratified ~6K SFT / ~4K GRPO / ~4K held-out. | para 2, L9 | Counted each split file; sum 6,000+4,236+4,236=14,472. QC = in-prompt auto-reject + schema only. | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json); [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) | supported |
| **Central finding:** corpus quality, not RL, is the decisive lever; curated SFT is the single best detector. | para 4 (bold), L13 | Arm-C SFT (82.80/72.07) beats IAD-R1 on both subsets and matches SFT+GRPO; GRPO helps only a weak baseline. | Arm-C ckpt-376; IAD-R1 recanon; GRPO run-2 ckpt-530 (all under `results/`). | supported |
| Result (i): 7B/frozen/6K/ep3 reaches 80.16 DS, within ~1pp of IAD-R1. | para 4 (i), L13 | SFT ckpt-564 = 80.16 vs IAD-R1 81.92; gap = 1.76pp (so "~1pp" understates). | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/…`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json); [`results/iad_r1_qwen_recanon/…`](results/iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json) | partially-supported |
| Result (ii): curated SFT (self-distill + STaR + Saunders, 6,000) reaches 82.80/72.07, single best detector, beats IAD-R1 +0.88/+0.73 with SFT alone. | para 4 (ii), L13 | Arm C of teacher ablation, ckpt-376; deltas vs IAD-R1 recanon. | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/…`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json) | supported |
| Result (iii): GRPO is double-edged — lifts weak Iter1 to 82.73/70.39; no benefit on strong Arm-C (~2pp below init); estimator-independent. | para 4 (iii), L13 | GRPO run-2 vs Iter1 init; GRPO-on-C 80.32–81.2 vs SFT-C 82.80; estimator ablation. | GRPO run-2 ckpt-530; [`results/grpo_qwen25vl_7b_abc_C_grpo/…`](results/grpo_qwen25vl_7b_abc_C_grpo/) | supported |
| Result (iv): composition > count — curated 6K (same-length ~141-word subset) beats 15K by ~8 pp; data-count effect, not length. | para 4 (iv), L13 | 6K ep3 80.16 vs 15K ep3 71.66 = 8.5pp; 6K is subset, same mean length. | ckpt-564 vs [`results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/…`](results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/eval_dsmvtec_full_trainprompt.json) | supported |
| Result (v): freezing the ViT consistently beats unfreezing, especially on VisA. | para 4 (v), L13 | Frozen VisA 64.28–66.94 vs unfrozen 57.13–59.18 across ckpts. | [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt) (15k_frozen vs 15k_unfrozen VisA) | supported |
| Result (vi): largest per-product gains (metal_nut +25.8, cable +19.2, pill +16.0); wood/tile regress marginally. | para 4 (vi), L13 | Per-product BA recompute, base → SFT ckpt-564 (frozen, 6K). | base vs [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/…`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json) | supported |
| Reproducible on 2×RTX A6000; single 7B model detects/localises/explains, no per-product retraining, no masks at inference. | para 5, L15 | Hardware from saved GRPO args; multi-task from XML output; eval uses image+prompt only. | GRPO run-2 `training_args.bin`; eval harness; Ch.03 XML schema. | supported |
| QC = in-prompt auto-reject + schema validation, no second model; corpus is unfiltered Gemini output. | para 2 L9 + result (iv) | Matches QC ground truth; result (iv) states both corpora are unfiltered. | gen pipeline; Ch.03 traces-filter. | supported |

## Chapter 01 — Introduction

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| Out-of-box VLMs insufficient (GPT-4o 74.9% on MMAD, below >95% production floor). | sec:motivation L20 | Cites MMAD GPT-4o 74.9%; asserts >95% floor (uncited). | jiang2025mmad (74.9% verified); >95% floor has no citation. | partially-supported |
| IAD-R1 (Qwen2.5-VL-7B SFT+GRPO) is the closest comparable SOTA. | sec:motivation L20 | Same-backbone SFT+GRPO baseline; re-evaluated under common harness. | li2025iadr1; [`results/iad_r1_qwen_recanon/…`](results/iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json) (81.92/71.34) | supported |
| Gap: no controlled like-for-like SFT→GRPO design study on commodity HW; thesis fills it. | sec:motivation L22 | Gap statement; backed by the controlled studies on 2×A6000. | ground truth; chapters 4–6. | supported |
| Training data restricted to Real-IAD; no MMAD-category sample trained → clean zero-shot. | sec:problem L30 | Trace products are Real-IAD names, disjoint from MVTec/VisA. | 30 Real-IAD products in trace JSONs; wang2024realiad. | supported |
| AnomalyThink is the first openly-released Real-IAD reasoning-trace corpus for GRPO with XML; distinct from Anomaly-Instruct-125k & Expert-AD. | sec:contributions item 1, L57 | Novelty contrast vs two named corpora. | xu2025anomalyov, li2025iadr1; corpus exists (14,472, 30 products). "First" not independently verifiable. | partially-supported |
| QC = auto-reject + schema, NO second model. | sec:contributions item 1, L57 | In-prompt auto-reject + Pydantic; no second-reviewer code path. | ground truth QC note. | supported |
| 7B-frozen-6K reaches 80.16 DS, within ~1pp of IAD-R1, no GRPO. | sec:contributions item 2, L59 | ckpt-564 = 80.16 vs IAD-R1 81.92 (gap 1.76pp; "~1pp" loose). | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/…`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json) | supported |
| "More data always better" is wrong: 6K **concise** beats 15K **verbose** by 8.5 pp. | sec:contributions item 2, L59 | 8.5pp gap verified, BUT "concise vs verbose" is wrong — same ~141-word mean length; it is a data-count effect. | gap verified; length-contrast contradicted by ground truth. | partially-supported |
| GRPO (4 rewards, k3 KL monitored β=0, G=4, lr=1e-6, 4,236-prompt balanced set) on Iter1 → 82.73/70.39/80.87, +0.81pp DS vs IAD-R1, on 2×A6000. | sec:contributions item 3, L61 | run-2 ckpt-530 recompute; config matched to saved args; delta computed. | GRPO run-2 ckpt-530 (DS/VisA/RealIAD); `training_args.bin`. | supported |
| GRPO lift conditioned on weak baseline: Arm-C (6K) = 82.80/72.07 with SFT alone; GRPO on top degrades ~2pp DS; estimator-independent; SFT corpus is single best detector. | sec:contributions item 3, L61 | Arm-C ckpt-376 vs GRPO-on-C ckpts (all below init); estimator ablation; reward rises while BA falls. | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/…`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json); [`results/grpo_qwen25vl_7b_abc_C_grpo/…`](results/grpo_qwen25vl_7b_abc_C_grpo/) | supported |
| Per-product: +13.6pp DS over base; metal_nut +25.8, cable +19.2, pill +16.0, toothbrush +15.8; tile −2.1, wood −0.1. | sec:contributions item 4, L63 | Per-product recompute. BUT +13.6 is the **GRPO** overall gain (82.73−69.01=+13.72) while per-product gains are the **SFT** model's — attribution mixed. | base vs ckpt-564 / vs GRPO 530 recompute. | partially-supported |
| Freezing the ViT consistently wins (robust SFT pattern). | sec:contributions item 2 / rqthree | Four-factor frozen-vs-unfrozen ablation. | ground truth + Ch.06 sft-summary; detailed later. | supported |
| Roadmap: higher-res inputs, graded spatial rewards (VisA), iterative SFT↔RL (OpenVLThinker), agentic pipeline. | sec:contributions item 5, L65 | Forward-looking roadmap. | deng2025openvlthinker; Variety-STaR in progress. | partially-supported |
| Classical detectors opaque (score+heatmap), per-product training, no few-shot — three frictions. | sec:motivation L13–15 | Qualitative argument citing per-category SOTA + zero-shot baselines. | roth2022patchcore, zavrtanik2021draem, hu2024anomalydiffusion, jeong2023winclip, zhu2025realiadvariety. | supported |
| Backbone fixed to Qwen2.5-VL: strongest open-weight at 3B/7B, fits one A6000, matches IAD-R1. | sec:problem L30 | Design decision; "strongest at 3B/7B" is an unbenchmarked assertion here. | bai2025qwen25vl, li2025iadr1; ground truth HW. | partially-supported |

## Chapter 02 — Background and Related Work

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| Classical IAD saturated (~99% AUROC) but unsolved on explanation/scalability/steerability. | §2.1.5 + takeaway L27–39 | Survey of four method families (all ≥96.6% AUROC) + five limitations. | roth2022patchcore, defard2021padim, gudovskiy2022cflow, zavrtanik2021draem, deng2022rd4ad, batzner2024efficientad, li2021cutpaste, liu2023simplenet; zhu2025realiadvariety, bergmann2022mvtecloco. | supported |
| Scaling 30→160 categories drops mean AUROC 10–20 pts. | §2.1.5 L31 | Cited Real-IAD Variety result. | zhu2025realiadvariety (arXiv 2511.00540). | supported |
| Classical methods regress 10–30 pts on logical anomalies (MVTec LOCO). | §2.1.5 L32 | Cited MVTec LOCO result. | bergmann2022mvtecloco. | supported |
| Modern VLMs = encoder + connector + LM. | §2.2 intro L45 | Architectural synthesis. | radford2021clip, li2023blip2, liu2023llava. | supported |
| Qwen2.5-VL extends Qwen2-VL (window-attn ViT, dynamic resolution valuable for IAD). | §2.2.2 L51 | Tech-report features + IAD motivation. | bai2025qwen25vl, wang2024qwen2vl. | supported |
| Gemini 2.5-Flash is the trace generator. | §2.2.2 L53 | Forward ref to Ch.04. | ground truth; geminiteam2025gemini25. | supported |
| Multimodal-CoT grounding reduces hallucination; leveraged by the trace template. | §2.3 L61 | Cites two-stage rationale-then-answer finding. | zhang2023multimodalcot. | supported |
| Structured XML CoT (think/loc/type/ans): machine-checkable (format rewards), evidence-before-conclusion reduces hallucination, trivial parsing. | §2.3 L61 | Design statement; "empirically reduces hallucination" needs Ch.6 evidence; schema is 4 tags vs "six-phase" elsewhere. | format reward is 1 of 4 (w_f=0.3); hallucination claim deferred. | partially-supported |
| PPO = clipped surrogate + learned value (GAE); GRPO replaces value with MC group baseline (z-score). | §2.4.1–2.4.2 L70–100 | Standard derivation, verified against DeepSeekMath PDF. | schulman2017ppo, shao2024deepseekmath; PDF confirmed. | supported |
| GRPO KL via Schulman k3 inside loss; hyperparams G, ε, β. | §2.4.2 eq:grpo-obj/eq:k3 | Canonical general GRPO objective; thesis's own run sets β=0 (KL monitored only) — deferred to Ch.5. | shao2024deepseekmath + schulman2020klestimator (general); run-2 β=0. | partially-supported |
| GRPO attractive for VLM-IAD: sparse reward, value-network memory saving, decomposable reward. | §2.4.2 L102 | Motivational argument; forward-refs the 4-component reward. | reward weights (0.3,0.3,0.2,0.2); 7B backbone. | supported |
| OpenVLThinker showed iterative SFT↔RL beats single-round; thesis does IAD iter-2 ablation. | §2.4.2 L104 | Cites OpenVLThinker; forward ref. | deng2025openvlthinker; iter2_v2 evals present. | supported |
| OpenVLThinker v2 introduces G²RPO (rank→inverse-normal-CDF); evaluated here. | §2.4.2 L104 | Cites sequel; forward ref. | hu2026openvlthinkerv2; g2rpo_v2 evals present. | supported |
| VLM-IAD = three paradigms (zero-shot CLIP / SFT / SFT+RL). | §2.5 intro L110 | Taxonomy of surveyed works. | all cited works in bib. | supported |
| AnomalyR1 = first end-to-end MLLM IAD with GRPO (Qwen2.5-VL-3B, single ROAM reward). | §2.5.3 L119 | Survey. | chao2025anomalyr1. | supported |
| AD-FM adds a re-think stage + GIoU localisation rewards. | §2.5.3 L119 | Verified against AD-FM PDF (`<rethink>` tags). | adfm2025; PDF confirmed. | supported |
| IAD-R1 = closest prior SFT+GRPO baseline; PA-SFT (2.9K Expert-AD) then SC-GRPO (3K, 4 rewards); LLaVA-OV-7B is its own headline backbone but thesis re-runs the Qwen-7B checkpoint. | §2.5.3 + takeaway L121–124 | Detailed paper reading; takeaway/body backbone nuance. | li2025iadr1 (arXiv 2508.09178); recanon (81.92/71.34). | supported |
| Thesis does NOT quote IAD-R1's reported numbers; re-runs the released Qwen-7B ckpt on the same harness. | §2.5.3 L121 | Stated decision; recanon under same n. | [`results/iad_r1_qwen_recanon/…`](results/iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json) (n=1670/2141). | supported |
| Three benchmarks: Real-IAD (train), MMAD DS-MVTec (1670) + VisA (2141) (eval), held-out Real-IAD (OOD). | §2.6 intro + L137 | Subset sizes recomputed from JSONs (tp+tn+fp+fn). | recomputed n=1670/2141 from recanon JSONs. | supported |
| GPT-4o tops out at 74.9% on MMAD. | §2.6 L137 | Cited MMAD headline. | jiang2025mmad. | supported |

## Chapter 03 — Reasoning Trace Generation (AnomalyThink corpus)

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| AnomalyThink = ~14.5K Gemini traces over Real-IAD, 3 disjoint stratified splits (6K/4K/4K). | L5, L150–156, L168 | json `len`: union 14,472; GRPO 4,236; held-out 4,236; SFT 6,000. | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json); [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) | supported |
| Splits enforce 50/50 balance, per-product stratification, zero image overlap. | L150–156, L170 | `<answer>` counts: 6K 3000/3000; GRPO 2118/2118; union ~50%. Overlap/stratification consistent with design, not re-tested. | json answer-tag counts. | partially-supported |
| 6K is a stratified PARTITION, not a quality-ranked subset; no second reviewer, no numeric score. | L138, L144, L225, L236 | `generate_realiad_traces.py` loads `inspector_prompt.txt` (no rubric) + Pydantic only; 6K = SFT partition. | [`scripts/00_generate/generate_realiad_traces.py`](scripts/00_generate/generate_realiad_traces.py) | supported |
| QC is in-prompt; 6K-beats-15K is composition/epoch-efficiency, not a filter. | L138, L144 | In-prompt self-check + schema gate; 6K is subset, equal mean length. | gen pipeline; cross-ref Ch.06. | supported |
| Four-tag XML schema identical to IAD-R1's, adopted unchanged for comparability; contribution is content/data/config. | L58, L221–233 | Schema {think,location,type,answer} matches SC-GRPO; trainer derived from IAD-R1/AnomalyR1. | li2025iadr1; ground truth. | supported |
| Traces use first-person inspection narrative; locations from exactly nine 3×3 grid cells. | L60–79, L185–201 | Parsed 6K assistant msgs: 4299/6000 first-person; only 9 grid-cell location tokens, no bare single-axis terms. | python parse of [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) | supported |
| The exact "six-phase template" prompt + five worded auto-reject rules are not recoverable from on-disk gen code; production prompt differs. | L60–68, L123–130 | `generate_realiad_traces.py` loads `inspector_prompt.txt` (3–5 sentence, allows bare locations, ~18 types, no auto-reject, even a mask-leak hint). `inspector_prompt_test_v2.txt` (NOT loaded) has 6 auto-reject + 8-dim rubric. Neither matches thesis wording, yet corpus matches description → production prompt differed from both. | [`prompts/inspector_prompt.txt`](prompts/inspector_prompt.txt); [`prompts/inspector_prompt_test_v2.txt`](prompts/inspector_prompt_test_v2.txt) | partially-supported |
| Red-overlay is a privileged training-only input, discarded at inference. | L33, L121 | Anomaly gen attaches orig+overlay+ref (3 images); eval uses test (+1-shot ref) only. | ground truth; gen design. | supported |
| Anomalous masks restricted to Multi-Close/Single/Spread; normal sampled uniformly. | L17 | Described mask-analysis filter; not re-derived (no mask-analysis artefact inspected). | thesis text; no on-disk artefact. | unverified |
| Closed type vocabulary constrains Gemini output; two top families well-separated in t-SNE (ratio ~9.0). | L179 | Embedded `<type>` with nomic, t-SNE, centroid/radius ratio. Recompute on saved embeddings gives ratio ~10.1 (Scratch n=500, Missing-component n=162); qualitative holds but exact figures (600/180/9.0) differ. | `anomaly_type_analysis/{embeddings_2d.npy,labels.npy,types.json}`; `analyze_anomaly_types.py` (not in shipped repo) | partially-supported |
| AnomalyThink first openly-released Real-IAD GRPO XML corpus; Expert-AD closed; Anomaly-Instruct-125k industrial subset is MVTec/VisA. | L207–233 | Source/format/release comparison. | xu2025anomalyov, li2025iadr1; corpus Real-IAD-only (image paths). | supported |
| One main SFT run additionally restricts to C1 camera angle. | L158 | `c1_only_fixed/` variant exists. | [`traces/anomalythink_15k/c1_only_fixed/`](traces/anomalythink_15k/c1_only_fixed/) | supported |
| Gemini 2.5-Flash chosen for multi-image input, grounded thinking, low cost/latency. | L132 | Config default verified; multi-image used. Cost (<$20, $0.0001/trace) & latency (1.6s/trace) are unlogged estimates. | config `GEMINI_MODELS`; no cost/latency logs. | partially-supported |
| Defect-type vocab translates Real-IAD codes (BX/HS/ZW/QS) to generic names for transfer. | L92 | 15K union still has raw codes (ZW 496, QS 447, HS 303, AK 260) alongside translated; 6K uses generic vocab → translation partial in union. | type-tag counts on [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) | partially-supported |

## Chapter 04 — Supervised Fine-Tuning

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| SFT = autoregressive NLL on trace tokens, prompt masked. | L18–23, L64 | Standard LlamaFactory/trl SFT objective. | [`configs/sft/`](configs/sft/) `stage: sft`, template qwen2_vl. | supported |
| Full fine-tuning (not LoRA), following IAD-R1 (LoRA fails long traces) + small pilot. | L31 | `finetuning_type: full`; cite. The "small pilot" has no artefact. | configs; li2025iadr1, hu2022lora. | partially-supported |
| Frozen-ViT wins decisively at 6K and 15K, especially VisA. | L38 | Forward ref to results; frozen VisA 64.28–64.78 vs unfrozen 58.16. | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json) vs 15k_unfrozen. | supported |
| `freeze_vision_tower=true` ⇒ only LM + projector update, ViT fixed. | L35, L74 | Read headline config. | [`configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml`](configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml) | supported |
| Same fixed system prompt at SFT/GRPO/inference (avoid prompt drift). | L66, L158 | `_trainprompt` suffix marks training-prompt evals; headline numbers from `_trainprompt`. | eval harness suffix contract. | supported |
| 7B-unfrozen doesn't fit on 2×A6000 without offload → ZeRO-3 CPU offload. | L72 | Memory argument; ZeRO-3 partitions + offloads. | [`configs/deepspeed/ds_z3_cpu_offload.json`](configs/deepspeed/ds_z3_cpu_offload.json); fit not separately logged. | partially-supported |
| bf16 more stable than fp16 for Qwen2.5-VL-7B. | L74 | Common VLM wisdom; bf16=True in all runs; no logged fp16 failure. | `training_args.bin` bf16=True. | partially-supported |
| HPs from a brief 1K-pilot sweep + single 6K validation. | L95 | Methodological statement. | none on disk — cannot confirm. | unverified |
| All four ablation cells fixed at LR 2e-5. | L123 | Checked every SFT config. | **CONTRADICTED**: frozen=1e-5, unfrozen=1e-6; no run used 2e-5, LR not held constant. | unverified |
| At LR 5e-5 model collapses to empty trace within epoch 1. | L123 | Design anecdote. | none on disk. | unverified |
| Pilot showed cosine under-utilises last 25%; linear decay + 100-step warmup gave higher BA. | L125 | Pilot rationale. | **CONTRADICTED**: every SFT run used cosine + 20/50-step warmup, the opposite. | unverified |
| Four-factor grid 2×2×2×4 = 32 cells, one factor moving per adjacent cell. | L138–152 | Arithmetic. BUT 3B/7B-Unfrozen-6K cells were never run (rows duplicate 15K-unfrozen). | arithmetic; ground truth (cells absent). | partially-supported |
| Effect-of-X comparisons are clean single-factor swaps. | L148–152 | Freezing-at-6K needs a 7B-unfrozen-6K cell that was never run. | ground truth (dir absent). | partially-supported |
| Test reference = held-out Real-IAD normal (same product/angle) or MMAD good/ dir. | L158 | Eval JSON records MMAD paths; good/ reference is harness behaviour. | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json) | partially-supported |
| Greedy decoding (max 512 tokens); beam/nucleus changed BA <0.3pp on a 200-sample check. | L160 | Decoding protocol; the 200-sample check has no artefact. | 512 tokens matches ground truth; check unlogged. | partially-supported |
| <0.5% of post-SFT outputs fail answer-tag parse. | L162 | Counted pred_answer not in {yes,no}: DS 0/1670, VisA 0/2141 = 0.0%. | recompute on ckpt-564. | supported |
| BA = 0.5(TPR+TNR) is primary metric (class balance varies). | L166 | Recompute matches headline 80.16. | [`results/compute_ba.py`](results/compute_ba.py); [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt) | supported |
| 6K-vs-15K is data-count/composition (same-mean-length subset), NOT length; length ablation pending. | L128–132 (TODO) | 6K = 100% subset, both ~141-word mean. Chapter flags ablation as not-yet-run. | ground truth; 6K subset of 14,472. | supported |
| Images loaded lazily to avoid materialising the ~100K-image dataset. | L64 | LlamaFactory lazy loading; 100K is the source pool estimate. | dataloader behaviour; 100K count not verified. | partially-supported |
| Grid claims epochs=4, but 3B-frozen-6K ran 6 epochs on disk. | L117 vs disk | Read num_train_epochs/global_step. | [`results/sft_qwen25vl_3b_zeroshot_6k_frozen/`](results/sft_qwen25vl_3b_zeroshot_6k_frozen/) trainer_state (step 1128, ep 6.0). | partially-supported |

## Chapter 05 — GRPO Reinforcement Learning

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| GRPO replaces PPO value net with MC group baseline. | §5.1 L11–17 | Standard GRPO; advantage = (R−µ)/σ broadcast. | DeepSeekMath/Shao; trainer impl. | supported |
| KL via Schulman k3, MONITORED not in loss (β=0). | §5.1 L25,27; tab:grpo-hp; takeaway | β=0 in saved args; trainer logs kl separately; loss ~0. | run-2 [`…/trainer_state.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/trainer_state.json) | supported |
| π_ref fixed to SFT ckpt-564, never updated. | §5.1 L27; tab:grpo-hp | θ_ref ← θ_0 once. | ground truth; algorithm L34. | supported |
| G=4 chosen for stable baseline vs rollout cost (~30s) / single-A6000 budget (vs DeepSeekMath G=64). | §5.4 L146 | run-2 train.log 83.85 s/step. | train.log (off-repo); ground truth; DeepSeekMath. | supported |
| η=1e-6 because GRPO is LR-sensitive; higher LR drifts. | §5.4 L148 | 1e-6: KL bounded, reward 1.66→2.13. High-LR drift not separately evidenced. | run-2 trainer_state. | partially-supported |
| 4-component reward sums to 1.0 (0.3/0.3/0.2/0.2). | §5.2.5 L92–93; eq:grpo-reward | Design rationale; arithmetic. | ground truth weights. | supported |
| Format reward uses DIFFERENT regexes for anomalous vs normal (label-conditioned). | §5.2.1 L61–75 | +0.075/tag credit; −0.1 forbidden-tag penalty on normal. | reward impl; not independently re-run. | partially-supported |
| Type reward = graded semantic sim (exact 1.0 / substring 0.8 / cos) via nomic on CPU, anomalous-only. | §5.2.3 L80–87; §5.5 L156 | Cosine of nomic embeddings; precompute vocab once. | ground truth (:5200); nomic2024embed. | supported |
| Location reward = presence reward (0.2 any value), NOT grid match (brittle to paraphrase). | §5.2.4 L89–90 | IAD-R1 contrast; pilot found grid brittle. | reward design; pilot not verified. | partially-supported |
| GRPO data = disjoint 4,236-prompt split, 50/50. | §5.3 L99 | `len=4236`; balance from ground truth. | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) | supported |
| GRPO stitches ref+target into one 1024×512 canvas (LEFT ref / RIGHT test). | §5.3 L101–106 | Single image-token sequence halves rollout cost; eval matches. | ground truth stitch config; pixel dims not dumped. NOTE Appendix A flags production run used `--single_img 1` (no stitch). | partially-supported |
| Production run processed 6,500+6,500 (13,000) samples for 2 epochs. | §5.3 L108; §5.5 L160 | **CONTRADICTED**: actual pool 4,236 (2,118+2,118); 1060 steps = 2 ep over 4,236 at EBS 8. 13K is OLD/WRONG. | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) (4,236); max_steps 1060. | unverified |
| Run-2 saved every 250 steps; headline ckpt-530 (ep1); ckpt-1060 underperforms by ~0.7pp. | §5.5 L160,177 | 530 DS 82.73 > 1060 DS 82.03 (gap 0.70). | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-{530,1060}/…`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/) | supported |
| GRPO adds +2.57pp DS, +5.61pp VisA over SFT. | takeaway L218 | 82.73−80.16; 70.39−64.78 (run-2 530 vs SFT 564). | both eval JSONs (CM in inventory). | supported |
| Held-out RealIAD ckpt-530 ~80.87 BA. | chapter scope | Recompute tp1455/tn1968/fp139/fn674. | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_realiad4k_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_realiad4k_full_trainprompt.json) | supported |
| Reward stats: combined 1.66→2.22→2.13; accuracy 0.875→1.25; format/consistency 0.781→0.875. | §5.5 L162–164 | Read run-2 trainer_state log_history at steps 10/100/530. | run-2 trainer_state.json. | supported |
| Mean KL bounded [0.003,0.095]. | §5.5 L165 | trainer_state min 0.0 / max 0.117; upper bound exceeded by 13/530 steps. Qualitative holds; numeric range understates max. | run-2 trainer_state KL series. | partially-supported |
| Completion length stable 163–174 tokens, no collapse. | §5.5 L166 | 334/530 steps in [163,174]; overall 153.3–186.4. No collapse; full range wider. | run-2 trainer_state. | partially-supported |
| Wall-clock 12.4h on 2×A6000 ZeRO-3. | §5.5 L167 | Full 2-epoch = 24.69h; ckpt-530 (ep1) at 83.85 s/it ≈ 12.34h. 12.4h = reaching ckpt-530, not full run. | train.log (off-repo). | partially-supported |
| Iter-2 improves over its own iter-2 SFT but does NOT beat single-stage 530; single-stage retained. | §5.6 L194; takeaway | iter-2 v2 best DS 80.64 < 82.73. | [`results/grpo_qwen25vl_7b_iter2_v2_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json`](results/grpo_qwen25vl_7b_iter2_v2_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json) | supported |
| Iter-2 procedure: roll out 530 → reject-sample 4,026 → reset to base → fresh SFT (frozen, 1e-5, bs4, 4ep) → GRPO again. | §5.6 L185–190 | `len=4026`; iter2_v2 SFT trainer_state 4ep/504 steps. | [`traces/iter2/iter2_v2_sft_train.json`](traces/iter2/iter2_v2_sft_train.json) (4,026); iter2_v2 dirs. | supported |
| Two iter-2 variants: v1 (first filter) and v2 (stricter + dedup); both reported. | §5.6 L194 | Separate output dirs; v2 = 4,026-trace corpus. | iter2_v1 archive + [`results/grpo_qwen25vl_7b_iter2_v2_full/`](results/grpo_qwen25vl_7b_iter2_v2_full/) | supported |
| G²RPO = non-parametric rank→quantile onto N(0,1); motivated by quantised heavy-tailed G=4 reward. | §5.8 L200–207 | z-score explodes when 3/4 rollouts share a bin; Φ⁻¹((rank−0.5)/G) gives ±1.15 for G=4 (recomputed). | hu2026openvlthinkerv2; recomputed quantiles. | supported |
| G²RPO is a clean A/B on the estimator alone (all else identical), toggled `--use_g2rpo=true`. | §5.8 L211 | `compute_1d_ot` ported; flag swaps advantage. Caveat: G²RPO evals are `grpoprompt` vs single-stage `trainprompt`. | impl; eval suffixes differ. | partially-supported |
| G²RPO improved over vanilla in some pilots but did NOT beat single-stage; estimator effect is small. | §5.8 L215; takeaway | G²RPO v1 best DS 81.94, v2 ckpt-795 81.87 < 82.73. | [`results/grpo_qwen25vl_7b_g2rpo_v2_full/checkpoint-795/eval_dsmvtec_full_grpoprompt.json`](results/grpo_qwen25vl_7b_g2rpo_v2_full/checkpoint-795/eval_dsmvtec_full_grpoprompt.json) (81.87) | supported |
| Optional outer-whitening flag made minimal difference; reported numbers use per-group OT without it. | §5.8 L213 | Pilot claim; no separate outer-whitening eval JSON. | none on disk. | unverified |
| GRPO is a deliberately minimal impl (G=4, 4 rewards sum 1.0, k3 monitored, 2ep/~4,236) and still beats SFT; iter-2/G²RPO don't beat it. | takeaway L218 | Synthesis of verified config + comparisons. | all numbers above. | supported |
| §5.7 compound experiments (F5a/b/c) is a PLACEHOLDER (methodology only). | §5.7 L225–257 | Explicit placeholder; baselines listed, no results column. | chapter text; verified baselines. | supported |
| Compound baselines to beat: F5a 81.94, F5b 82.73, F5c 80.64. | tab:compound-design L252–254 | Each recomputed from corresponding JSON. | g2rpo v1 530 (81.94), run-2 530 (82.73), iter2_v2 530 (80.64). | supported |
| AnomalyThink cleaning (2,700 type subs + 1,167 loc norms) removes local-reward noise; each lever neutral/slightly negative vs 82.73. | §5.7 L229 | Cross-ref dataset chapter for counts (a grpo_train backup diff showed 0 changed rows in my check); "each lever neutral/negative" supported by G²RPO/iter-2 underperformance. | counts not confirmable from GRPO file; G²RPO/iter-2 < 82.73. | partially-supported |
| GRPO rewards entirely local; no Gemini judge used. | §5.2; §5.7 L239 | Reasoning reward (judge :5100) unused → flat 0.5. | ground truth; reward defs. | supported |
| Framework = HF trl GRPO trainer modified (~700 lines). | §5.5 L154 | Tension: ground truth + §5.8 (`iad_r1_grpo_custom/.../sc_grpo_trainer.py`) say IAD-R1/AnomalyR1 lineage, not vanilla trl. Script not located. | conflicting refs. | unverified |

## Chapter 06 — Results and Analysis

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| BA primary because raw acc hides imbalance (3B over-predicts yes, 7B over-predicts no). | L35 | 3B base recall ~100% / low TNR; 7B base TNR ~99% / recall 7.69 VisA. | [`results/qwen25vl_baseline_eval/`](results/qwen25vl_baseline_eval/); baseline JSONs. | supported |
| Quality >> quantity: 6K beats 15K ~8.5pp at fixed compute. | L79–81,87–89 | 80.16 vs 71.66; same mean length, 6K subset → composition. | ckpt-564 vs [`results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/…`](results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/eval_dsmvtec_full_trainprompt.json) | supported |
| 6K-vs-15K is NOT concise-vs-verbose. | L81 | Subset, identical ~141-word mean; removing 8.5K (not shortening) drives it. | ground truth corpus analysis. | supported |
| Freezing ViT beats unfreezing (encoder overfit). | L83 | Frozen > unfrozen on VisA after correcting cross-contaminated rows. | sft_7b_15k_unfrozen vs _15k_frozen JSONs. | supported |
| 7B > 3B on every matched cell by 4–12 pp. | L85 | 7B-frozen-6K 80.16 vs 3B-frozen-6K 69.08 ≈ 11pp. | ckpt-564 vs 3b_zeroshot_6k_frozen. | supported |
| GRPO +2.57 DS / +5.61 VisA; larger VisA gain (weaker SFT there). | L120,132 | run-2 530 (82.73/70.39) − SFT 564 (80.16/64.78). | both JSONs. | supported |
| GRPO ckpt-1060 underperforms 530 by ~0.7pp → over-optimisation. | L130,427 | 82.73 vs 82.03; reward rises while BA falls. | run-2 530 vs 1060 + trainer_state. | supported |
| Run 2 dominates Run 1 (seed sensitivity). | L125 | run2 530 82.73 > run1 530 81.65. | run1 vs run2 ckpts. | supported |
| Iter-2 doesn't beat single-stage; iter-2 GRPO peaks ep1 then declines. | L164,183 | iter-2 v2 530 80.64 < 82.73; 530 is iter-2 peak. | [`results/grpo_qwen25vl_7b_iter2_v2_full/`](results/grpo_qwen25vl_7b_iter2_v2_full/) (grpoprompt). | supported |
| G²RPO doesn't beat single-stage on DS (81.94 < 82.73). | L177,183 | G²RPO best 530 = 81.94 (v1 archive) vs 82.73. | iter2_v1 archive g2rpo (off-repo); 82.73 in repo. | supported |
| "More machinery did not help": single-stage strongest in GRPO family. | L183 | 82.73 > 81.94 > 80.64. | tab:rl-variants (all verified). | supported |
| Two data flaws corrected (2,700 type subs + 1,167 loc norms). | L193 | Post-hoc audit; counts from cleaning script (not on disk). | ground truth (~30% raw-code NG); counts unverified. | partially-supported |
| Post-GRPO rollout failures are NG-skewed (NG 30.1% vs OK 17.0% ≥1 wrong; 15.8% vs 3.4% all-8). | L195,199 | Roll out 530 at k=8 on 10,236 items; table arithmetic consistent. | failure-mode arithmetic; phase0_variety_star rollout artefact is empty. | partially-supported |
| All three post-GRPO SFT-Iter2 refinements regress (−1.00, −5.47, −10.65 pp). | L233,258–260 | Best-epoch avg of iter2_clean/heldout/balanced minus baseline 76.56. | [`results/sft_qwen25vl_7b_iter2_clean/`](results/sft_qwen25vl_7b_iter2_clean/), iter2_heldout, iter2_balanced JSONs. | supported |
| SFT-Iter2 fails by gradient asymmetry: kept items ~0 loss, patched ~96% NG → over-predicts defects. | L256,259 | Pool item-balanced but patched subset 94.8% NG; effective gradient ~95% NG. | per-product iter2 deltas (capsule −6.1, metal_nut −6.0); composition arithmetic. | partially-supported |
| Balanced-192 collapses (322/1670 unparsable ep1; VisA OK recall 20.5%). | L233 | iter2_balanced ckpt-6: 322 non-yes/no; VisA TNR 20.5%. | iter2_balanced ckpt-6 JSONs. | supported |
| Compound experiments (F5a/b/c) placeholder / pending. | L266–299 | TBD cells; explicit placeholder; phase0_variety_star empty. | thesis text; empty dirs. | supported |
| All IAD-R1 + own numbers under one harness (1670 DS, 2141 VisA). | L307,326 | recanon re-eval, same prompt/parser, matched n. | [`results/iad_r1_qwen_recanon/…`](results/iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json) | supported |
| Arm-C SFT-only beats IAD-R1 both (+0.88 DS, +0.73 VisA). | L330 | 82.80/72.07 vs 81.92/71.34. | armC 376 + recanon JSONs. | supported |
| SFT+GRPO beats IAD-R1 on DS (+0.81), trails VisA (−0.95). | L331 | 82.73/70.39 vs 81.92/71.34. | run-2 530 + recanon. | supported |
| Per-product gains concentrate on weakest; wood/tile regress slightly. | L342,346 | base vs ckpt-564: metal_nut +25.8, cable +19.2, pill +16.0; wood −0.1, tile −2.1. | per-product recompute. | supported |
| Chart caption "fifteen others gain" but only 13/15 gain (2 regress) — contradictory. | L342 | 15 products; wood+tile regress → 13 gain; 2+15=17>15. | per-product recompute. | unverified |
| Held-out RealIAD (23 seen products, unseen images) = 80.87 BA, TNR 93.4%. | L388,409 | run-2 530 on 4236 split; explicitly NOT product-disjoint. | [`…/eval_realiad4k_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_realiad4k_full_trainprompt.json) | supported |
| SFT BA peaks ep3 (ckpt-564); loss decreases monotonically. | L415 | Loss 1.14(s50)→0.58(s500). NOTE on-disk 7B-6K-frozen ends at step 560 (4ep), so "step 752/ep4" is the 3B cadence (wrong here); peak-at-ep3 holds. | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/trainer_state.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/trainer_state.json) (max step 560) | partially-supported |
| GRPO reward profile "healthy": combined ~2.1, KL bounded, length stable. | L419–425 | trainer_state: combined 1.66→2.22→2.13; KL [0.003,0.095]; ~169–174 tokens. | run-2 1060 trainer_state. | supported |
| Teacher ablation O3: C > B > A on DS; each layer ~2pp. | L521–523,531 | Fixed 6000-image pool, vary intervention; best-epoch DS A 79.01 / B 80.75 / C 82.80. | abc_A/B/C JSONs ([`results/sft_qwen25vl_7b_abc_C_full_patched/`](results/sft_qwen25vl_7b_abc_C_full_patched/)). | supported |
| Arm C SFT-only matches SFT+GRPO DS (within 0.07), beats VisA (+1.68), no GRPO. | L525,532 | 82.80/72.07 vs 82.73/70.39. | armC 376 vs run-2 530. | supported |
| Self-distill (Arm A, no Gemini) beats Gemini-headline on VisA (+4.05/+5.43) but trails DS (−1.15). | L527,532 | Arm A 2978 model-written; VisA 376 68.83 / 282 70.21; DS 376 79.01. | abc_A 282/376 JSONs. | supported |
| On VisA ordering is C > A > B (B underperforms A despite more data + extra step). | L529,532 | VisA: C 72.07 > A 70.21 > B 68.73. | abc arm VisA JSONs. | supported |
| Single best model in thesis = Arm-C SFT (82.80/72.07), not any GRPO. | L599 | Exceeds run-2 GRPO and all GRPO-on-C. | armC 376 vs all GRPO-on-C JSONs. | supported |
| GRPO degrades strong Arm-C baseline 2.0–2.5pp DS while reward rises (reward/metric decoupling); published lift was on a weak baseline. | L565,599 | GRPO-on-C all below init DS (−2.04 to −2.48); contrast weak Iter1 lift. | [`results/grpo_qwen25vl_7b_abc_C_grpo/checkpoint-{265,530}/…`](results/grpo_qwen25vl_7b_abc_C_grpo/) (ckpt-795 absent). | partially-supported |
| Degradation independent of advantage estimator (z-score, Dr.GRPO, G²RPO all fail to exceed SFT init). | L567,589,598 | 120-step probe from Arm-C 376; none exceeds 84.52 init; G²RPO crashed at step 100 (nan). | grpo_probe_{ctrl,drgrpo,g2rpo} probe_curve.csv (off-repo). | supported |
| Corpora unfiltered Gemini output; no second-reviewer/GPT-5-mini/8-rubric filter. | L81 | QC = in-prompt + schema only. | ground truth pipeline. | supported |
| Arm sizes A=2978, B=4369, C=6000; rollout pool 10,236. | L456,458,460,195,199 | json `len` + rollouts line count. | [`traces/teacher_ablation_abc/sft_A_kept_balanced.json`](traces/teacher_ablation_abc/sft_A_kept_balanced.json) (2978), [`…/sft_B_kept_corrected.json`](traces/teacher_ablation_abc/sft_B_kept_corrected.json) (4369); rollouts_raw.jsonl off-repo (10,236). | supported |

## Chapter 07 — Discussion

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| Across matched configs frozen > unfrozen (4–8pp DS, 4–10pp VisA). | sec:disc-frozen L11 | With corrected rows the DS gap is ~0 (68.65 vs 68.58); the large gap is on VisA recall. Stated DS magnitude not cleanly reproduced. | sft_7b_15k frozen vs unfrozen JSONs. | partially-supported |
| Unfrozen 7B collapses on VisA recall (34–41% on true defects) vs frozen ~70%. | sec:disc-frozen L19 | recall = tp/(tp+fn): unfrozen-15K 34.8–40.6%; frozen-6K 564 71.1%. | unfrozen-15K VisA JSONs; [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json) | supported |
| ViT update under indirect gradient unstable on small data (OpenVLThinker). | sec:disc-frozen L17 | Mechanistic argument + citation. | deng2025openvlthinker; eq:sft-loss; own ablation. | partially-supported |
| 6K-beats-15K is data-count/composition, NOT length/verbosity/quality-filter. | sec:disc-quality L29(a) | 6K subset of union; same ~141-word mean; no second-pass filter. | ground truth. | supported |
| 8.5pp 6K-vs-15K gap is the single largest effect in the grid. | sec:disc-quality L27 | With corrected 15K DS=80.12 vs 6K 80.16 the gap is ~0, not 8.5pp. 8.5pp only vs older/uncorrected 15K (ep3 71.66). | corrected 15K DS 80.12 vs 6K 80.16. | unverified |
| At fixed 4-epoch budget 15K is under-trained (loss still descending); 2.5× more updates. | sec:disc-quality L31(b) | Pilot loss curves (no file); 14,472/6,000 = 2.41×. | arithmetic; loss curves not provided. | partially-supported |
| 6K more structurally uniform; 15K extension drifts to longer prose, tighter posterior. | sec:disc-quality L33(c) | Qualitative; conflicts with ground truth (same mean length, subset, same recipe). No structural measurement. | none; tension with ground truth. | unverified |
| VisA intrinsically harder (multi-object, ~0.7% vs 2.1% defect area, presence-only reward). | sec:disc-visa L41–47 | Internal DS−VisA gaps verified (Arm-C 10.73, IAD-R1 10.58, GRPO 12.34); defect-area stats unsourced. | computed gaps from eval JSONs; area figures unsourced. | partially-supported |
| VisA gap not a deficit vs SOTA: IAD-R1 shows same gap; best model beats IAD-R1 VisA +0.73, DS +0.88. | sec:disc-visa L41 | recanon re-eval under same n; Arm-C 72.07 vs 71.34, 82.80 vs 81.92. | recanon + armC 376 JSONs. | supported |
| Run-2 530 > 1060 by ~0.7pp despite lower loss / higher reward (reward hacking). | sec:disc-checkpoint-selection L55 | 82.73 vs 82.03 (tp962/tn380/fp64/fn264). | run-2 530 + 1060 JSONs. | supported |
| Headline GRPO used no KL penalty (β=0); k3 KL monitored, stayed small. | sec:disc-checkpoint-selection L57 | Saved args β=0; 2-epoch budget. | run-2 `training_args.bin`. | supported |
| Published +2.57/+5.61 lift quantified weak Iter1 baseline → SFT+GRPO, not best-SFT → GRPO. | sec:disc-grpo-revisit L63–65/L80 | Lift vs ckpt-564; Arm-C SFT already matches/beats GRPO. | 82.73−80.16; 70.39−64.78; armC 376. | supported |
| Arm-C SFT alone matches SFT+GRPO within 0.07 DS, beats +1.68 VisA, no GRPO. | sec:disc-grpo-revisit L63 | 82.80−82.73; 72.07−70.39. | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/…`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json); zelikman2022star, saunders2022selfcritique. | supported |
| GRPO uniquely improves structured-output reliability: format compliance ~92% → >99%. | sec:disc-grpo-revisit L70 | Attributed to 0.30 format reward; the 92%/99% figures have NO backing metric on disk. | format weight 0.30 verified; figures unbacked. | unverified |
| GRPO improves type-string + localisation precision (0.20 rewards), invisible to binary metric. | sec:disc-grpo-revisit L71–72 | Reward weights/embedding model verified; quality improvement is rationale, no metric. | ground truth type/loc=0.2. | partially-supported |
| GRPO-on-Arm-C lands 2.0–2.5pp below init DS at every ckpt while reward climbs; only ckpt-265 edges +0.61 VisA. | sec:disc-grpo-revisit L78 | 265 80.61 (−2.19), 530 80.32 (−2.48), 1060 81.20 (−1.60, so range understates); VisA 265 72.68 (+0.61). | [`results/grpo_qwen25vl_7b_abc_C_grpo/checkpoint-{265,530,1060}/…`](results/grpo_qwen25vl_7b_abc_C_grpo/) | partially-supported |
| Three-estimator ablation: none exceeds SFT init DS; z-score strongest on VisA. | sec:disc-grpo-revisit L78/L83 | max probe_ds(step>0): ctrl 83.67 / drgrpo 84.23 / g2rpo 84.39, all < init 84.52; VisA ctrl 72.22 > drgrpo > g2rpo. | grpo_probe_* probe_curve.csv (off-repo); liu2025drgrpo, hu2026openvlthinkerv2. | supported |
| On a strong baseline GRPO of any estimator can't exceed it; predicted by RLVR-sharpening / reward-overoptimisation. | sec:disc-grpo-revisit L80 | Synthesis of GRPO-on-C + probe. | gao2023scaling, yue2025rlvr, yu2025reassessing. | supported |
| Single best detector = Arm-C SFT (82.80/72.07), not any GRPO. | sec:disc-grpo-revisit L80 | Max DS and max VisA across models. | eval JSONs. | supported |
| Failure-mode taxonomy: 200 hand-inspected errors in 5 categories with stated %s; first two >half, fixable by resolution/prompt. | sec:disc-errors L90–100 | Manual inspection; no backing file recording counts. | qualitative; no file. | unverified |
| Limitations: single-seed SFT; GRPO run twice; run-1 vs run-2 DS 80.6 vs 82.7 → seed sensitivity. | sec:disc-limitations L108 | run-1 530 is actually 81.65 (80.63 is run-1 ckpt-315). Seed point stands (~1.1pp) but cited 80.6 is the wrong ckpt. | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run1/`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run1/) (off-repo run1 530=81.65). | partially-supported |
| No human eval of reasoning quality; trace quality open. | sec:disc-limitations L110 | Only BA + 4 reward components measured. | reasoning reward unused (flat 0.5). | supported |
| Latency: 170 tokens @ ~30 tok/s on A6000 (~5–6s), ~2 orders slower than EfficientAD. | sec:disc-limitations L112 | Stated throughput; no latency benchmark file; EfficientAD from literature. | no measurement file. | unverified |
| Single backbone (Qwen2.5-VL); transfer unknown but IAD-R1 multi-backbone suggests it transfers. | sec:disc-limitations L114 | All experiments Qwen2.5-VL; transfer cites IAD-R1. | ground truth; IAD-R1 paper. | supported |
| Gemini may leak product knowledge; mitigated by Real-IAD train / MMAD eval. | sec:disc-limitations L116 | Mitigation = disjoint train/eval. | ground truth data pipeline. | supported |
| Final headline: SFT+GRPO 7B on 6K traces overtakes IAD-R1 by 0.8pp on 2×A6000. | sec:disc-limitations L119 | 0.8pp = 82.80−81.92 (Arm-C, the best model); SFT+GRPO is 82.73 (+0.81). Sentence attributes to "SFT+GRPO" but 0.88 best matches Arm-C. ~0.8pp holds either way. | armC 82.80 / GRPO 82.73 vs IAD-R1 81.92. | partially-supported |

## Chapter 08 — Conclusion and Future Work

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| Competitive result delivered by SFT on a curated corpus; RL helps only on a weak baseline. | Summary L9; contrib (6) L21; RQ2 L33 | GRPO lift on weak Iter1 (+2.57/+5.61) vs degradation on strong Arm-C (−1.6 to −2.5); estimator-independent. | ckpt-564, run-2 530, armC 376, GRPO-on-C; [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt). | supported |
| ~14.5K AnomalyThink built on Real-IAD with Gemini (orig+overlay+ref), six-phase template, fixed XML, NO second reviewer. | Contribution (1) L11; L23 | `len=14472`; gen pipeline + QC documented; six-phase + 4 tags. | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json); Ch.03; ground truth. | supported |
| Partition: 6K SFT / 4K GRPO / 4K held-out; 6K is the headline corpus. | Contribution (1) L11 | Counts: 6000/4236/4236; disjoint stratified. | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json); [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) | supported |
| Frozen wins; 7B > 3B; 6K concise beats 15K verbose by 8.5pp. | Contribution (2) L13; RQ3 L35 | 8.5pp = 80.16 − 71.66; frozen/7B patterns from SFT table (some rows cross-contaminated, headline 80.16 correct). | [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt); ground truth. | supported |
| 6K advantage is data-count/composition, NOT length/quality-filter; extra ~8.5K add no benefit / slightly hurt; no second-pass filter. | RQ1 L31 | 6K = 100% subset; both mean `<think>` 136.8 words (recompute); QC in-prompt only. | recompute on anomalythink combined files; ground truth. | supported |
| Best SFT (7B/frozen/6K/ep3) = 80.16 DS, within ~1pp of IAD-R1, same backbone, SFT alone. | Contribution (2) L13 | 80.16 vs 81.92 → gap 1.76pp ("~1pp" overstates closeness). | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/…`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json); [`results/iad_r1_qwen_recanon/…`](results/iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json) | partially-supported |
| GRPO (0.3/0.3/0.2/0.2, G=4, β=0 monitored, lr=1e-6) → 82.73 DS / 70.39 VisA, +0.81 DS / −0.95 VisA vs IAD-R1. | Contribution (3) L15 | run-2 530 recompute; deltas vs recanon; config = saved args. | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/…`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json); recanon. | supported |
| Held-out RealIAD (image IDs unseen, 23/30 products): 80.87 BA, TNR 93.4% (low FP for deployment). | Contribution (4) L17 | tp1455/tn1968/fp139/fn674 → BA 80.87, TNR 0.934; sample-level. | [`…/eval_realiad4k_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_realiad4k_full_trainprompt.json) | supported |
| Held-out is sample-level (not product-disjoint); product-disjoint left for future work. | Contribution (4) L17; L53 | Withholds image IDs from products also in training. | Ch.03; ground truth. | supported |
| Largest gains on hard products (metal_nut +25.8, cable +19.2, pill +16.0); errors cluster in 5 categories, top two dominate. | Contribution (5) L19 | base→GRPO per-product deltas (metal_nut +29.6, cable +17.1, pill +15.2). Thesis numbers don't match base→GRPO; direction holds, magnitudes don't (Appendix C GRPO per-product was FABRICATED). | base JSONs; GRPO per-product from ground truth. | partially-supported |
| Teacher ablation: RAFT / STaR / STaR+Saunders; Arm C = 82.80/72.07 SFT-only, single best, beats IAD-R1 DS. | Contribution (6) L21 | Arm-C 376 recompute; both > recanon. | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/…`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json) | supported |
| GRPO on Arm-C init lowers DS 2–2.5pp while reward rises; estimator-independent; the (3) lift is a weak-baseline artefact. | Contribution (6) L21; RQ2 L33 | 82.80 vs 80.61/80.32/81.20; 1060 is −1.6 (range mildly understated). | [`results/grpo_qwen25vl_7b_abc_C_grpo/checkpoint-{265,530,1060}/…`](results/grpo_qwen25vl_7b_abc_C_grpo/); [`results/grpo_qwen25vl_7b_g2rpo_v2_full/`](results/grpo_qwen25vl_7b_g2rpo_v2_full/); probe dirs (off-repo). | supported |
| AnomalyThink = first openly-available Real-IAD reasoning-trace dataset for direct GRPO; complements Anomaly-Instruct-125k, contrasts Expert-AD (closed). | Closing L23 | Positioning; Expert-AD closed, Anomaly-Instruct-125k open. "First" not independently verifiable. | xu2025anomalyov, li2025iadr1; Ch.03 positioning. | partially-supported |
| RQ1: SFT lifts 7B 69.01 (base) → 80.16 (6K) → 82.80 (Arm-C); gain from XML structure + curated 6K. | RQ1 L31 | All three recomputed; (i) format = rationale, (ii) 6K composition = supported by subset/equal-length. | [`results/qwen25vl_baseline_eval/`](results/qwen25vl_baseline_eval/) (69.01); ckpt-564 (80.16); armC 376 (82.80). | supported |
| Iter-2 SFT↔RL did NOT beat single-stage on IAD (80.64 vs 82.73); rollout/filter concentrates on easy examples; remedy = difficulty-aware filter + mix-in. | F1 L43 | iter-2 530 80.64 (grpoprompt) vs 82.73 (trainprompt); self-SFT = 4,026 traces. Easy-example claim is interpretation. | [`results/grpo_qwen25vl_7b_iter2_v2_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json`](results/grpo_qwen25vl_7b_iter2_v2_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json); [`traces/iter2/iter2_v2_sft_train.json`](traces/iter2/iter2_v2_sft_train.json) (4,026). | supported |
| Real-IAD Variety (F5, in progress): Arm-C STaR on 160-category C1; 82.80 ckpt rolled out k=8, judged/corrected, 6,000-trace corpus stratified. Results placeholder. | F5 L51 | Variety corpus exists (6000, 160 products, 3000/3000); recipe matches; results placeholder. | [`traces/variety_star_6k/variety_star_sft_6k.json`](traces/variety_star_6k/variety_star_sft_6k.json); ground truth. | partially-supported |
| End product = single 7B model (detect/localise/explain), no per-product retraining, ~5–6s/inspection, Arm-C beats IAD-R1 on DS-MVTec. | Closing L23 | 82.80 > 81.92; single-model design. 5–6s latency unmeasured (170 tok / 30 tok/s). | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/…`](results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json) vs recanon; latency estimate. | partially-supported |
| RQ3 trade-offs: 7B>3B, frozen>unfrozen, 6K>15K at 4-ep budget; gains on hard products; surface-texture near ceiling; multi-object is VisA bottleneck. | RQ3 L35 | Texture ceiling: SFT per-product (wood 98.7%, leather 99.2%); GRPO per-product (leather 98.44, wood 97.50). | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/) per_product; ground truth. | supported |
| Most reusable lesson: trace quality dominates quantity by margins larger than a 4× model-size change. | Closing L61 | 6K-vs-15K 8.5pp exceeds typical 3B→7B gains; "4×" is loose (3B→7B ≈ 2.3× params). | [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt); ground truth. | partially-supported |
| Future directions cite OpenVLThinker/ReAct/Toolformer/AgentIAD/OpenThinkIMG/DeepEyes/Real-IAD Variety/GSPO/InternVL3/LLaVA-OV/DPO; AgentIAD 3B+tools beats 7B no-tool. | F1–F4 L43–53 | All 13 keys resolve in bib; AgentIAD claim taken from literature. | bib/thesis.bib. | supported |
| Trained on 25 Real-IAD training products. | Contribution (1) L11 | 6K corpus has **30** distinct products; "25" is Appendix-A's 25/5 split table, conflicting with data; held-out is sample-level from 23/30, not a 5-product holdout. | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) (30 products); Appendix A ("25/5"). | unverified |

## Appendix A — Full Hyperparameter Tables

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| Gemini-2.5-Flash with batching/retry/rate limit (batch 10, 3 retries, 4s). | Table A.1, L19–23 | Read config: BATCH_SIZE=10, MAX_RETRIES=3, RATE_LIMIT_WAIT=4; default `gemini-2.5-flash`. | [`scripts/00_generate/config_mmad.py`](scripts/00_generate/config_mmad.py); [`scripts/00_generate/generate_realiad_traces.py`](scripts/00_generate/generate_realiad_traces.py) | supported |
| Each anomaly trace from 3 images (orig/overlay/ref); each normal from 1. | Table A.1, L26–27 | Matches ground truth; overlay creation exists. | [`scripts/00_generate/utils_realiad.py`](scripts/00_generate/utils_realiad.py) `create_overlay`; grpo_train image paths. | supported |
| QC = in-prompt auto-reject + JSON schema, NO second-reviewer model. | Table A.1, L29–30 | Pydantic `BatchReasoningTraces` + regen on malformed; no reviewer-model call. | [`scripts/00_generate/generate_realiad_traces.py`](scripts/00_generate/generate_realiad_traces.py) | supported |
| Per-product quota K=187/class/split; disjoint (basename-excluded) SFT/GRPO at 50/50. | Table A.1, L33–37 | `PER_CLASS_PER_PRODUCT=187`; split_products allocates disjointly; basename exclusion; GRPO 2118+2118. | [`scripts/00_generate/`](scripts/00_generate/) `generate_unified_realiad_15k_c1_only.py`; [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) | supported |
| Product split 25 train / 5 held-out from 30. | Table A.1, L35 | Generator header excludes 5 held-out; but actual products per split differ (6K=30, GRPO=23) due to anomaly-availability — "25" is the partition definition, not the present count. | split generator; ground truth counts. | partially-supported |
| All SFT cells used LR 2e-5. | Table A.2, L54–61 | **CONTRADICTED**: frozen=1e-5, unfrozen=1e-6; no run used 2e-5; collapses two distinct LRs. | [`configs/sft/`](configs/sft/) yamls. | unverified |
| Frozen vs unfrozen used BS/GA 4/4 vs 2/8 (EBS 16). | Table A.2, L54–61 | **CONTRADICTED**: 7B-15K-frozen is 16/1; 3B-15K-unfrozen is 4/4; 7B-15K-unfrozen is 16/1. 2/8 not used. | [`configs/sft/sft_qwen25vl_7b_15k_frozen.yaml`](configs/sft/sft_qwen25vl_7b_15k_frozen.yaml), unfrozen yamls. | unverified |
| 3B/7B-unfrozen-6K SFT cells were trained (rows in A.2). | Table A.2, L56,60 | NEVER RUN (dirs absent; rows duplicate 15K-unfrozen); no 6K-unfrozen yaml exists. | ground truth; [`configs/sft/`](configs/sft/) listing. | unverified |
| All SFT cells: linear LR + 100-step warmup, wd 0.01, clip 1.0, seq 4096, AdamW (0.9,0.95). | L66 | **CONTRADICTED**: scheduler cosine; warmup 20/50; cutoff_len 12144; betas/wd unset (library defaults). | [`configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml`](configs/sft/sft_qwen25vl_7b_zeroshot_6k_frozen.yaml) | unverified |
| SFT used ZeRO-3 CPU offload on 2×A6000 (48GB). | L66 | yamls point to ds_z3_cpu_offload.json (stage 3 + CPU offload). | [`configs/deepspeed/ds_z3_cpu_offload.json`](configs/deepspeed/ds_z3_cpu_offload.json); ground truth. | supported |
| GRPO base/ref = 7B-frozen-6K SFT ckpt-564. | Table A.3, L80 | run script sets MODEL_NAME_OR_PATH to ckpt-564; run2 resumed from run1/530 (from 564). | [`scripts/02_grpo/run_grpo_7b_6k_frozen_ep3.sh`](scripts/02_grpo/run_grpo_7b_6k_frozen_ep3.sh) | supported |
| GRPO used G=4, β=0 (k3 monitored), ε=0.2, η=1e-6. | Table A.3, L81–85 | Unpickled saved args: num_generations=4, beta=0.0, epsilon=0.2, lr=1e-6. | run-2 `training_args.bin`; ground truth. | supported |
| GRPO trained on 4,236 (2,118+2,118), EBS 8. | Table A.3, L91,94 | `len=4236`; is_anomaly True 2118 / False 2118; EBS = 1×4×2 = 8. | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json); saved args (BS 1, GA 4). | supported |
| GRPO per-device BS=2, max prompt 2048, save/eval every 250. | Table A.3, L89,92,96–97 | **CONTRADICTED** by saved args: BS=1, max_prompt=4096, save_steps=530, eval_steps=None. | run-2 `training_args.bin`; [`scripts/02_grpo/run_grpo_7b_resume_run2.sh`](scripts/02_grpo/run_grpo_7b_resume_run2.sh) | unverified |
| GRPO used a 1024×512 stitched image (LEFT ref / RIGHT test). | Table A.3, L100 | Stitch is the 1-shot path; production run2 launched with `--single_img 1` (single test image, no stitch). Hyperparameter does not describe the production run. | [`scripts/02_grpo/run_grpo_7b_resume_run2.sh`](scripts/02_grpo/run_grpo_7b_resume_run2.sh) `--single_img 1`; [`scripts/02_grpo/stage_rl/grpo_ad.py`](scripts/02_grpo/stage_rl/grpo_ad.py) | unverified |
| GRPO type-reward = nomic-embed-text-v2-moe on CPU via local server. | Table A.3, L99 | `NomicEmbeddingModel` HTTP server (:5200); judge server loads the model float32. | [`scripts/02_grpo/stage_rl/`](scripts/02_grpo/stage_rl/) reward; [`scripts/02_grpo/gemini_judge_server_v2.py`](scripts/02_grpo/gemini_judge_server_v2.py) | supported |
| GRPO ran seed 42 (run 2) on 2×A6000 ZeRO-3 CPU offload. | Table A.3, L102–103 | Saved args seed=42; run script CUDA 1,2 + nproc 2 + zero3_offload.json. | run-2 `training_args.bin`; [`scripts/02_grpo/run_grpo_7b_resume_run2.sh`](scripts/02_grpo/run_grpo_7b_resume_run2.sh) | supported |
| Appendix lists HPs for the production GRPO run (run2 ckpt-530, DS 82.73 / held-out 80.87). | §A.3 header L69–75 | Held-out recompute = 80.87 (tp1455/tn1968/fp139/fn674) confirms the table describes this run. | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_realiad4k_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_realiad4k_full_trainprompt.json) | supported |

## Appendix B — Reproduced Prompts

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| Appendix reproduces the verbatim Gemini trace-generation system prompt actually used. | intro L5; §app:p-gemini L8 | Gen scripts load `inspector_prompt.txt`; `generate_variety_traces.py` loads `inspector_prompt_test_v2.txt`. Appendix text (six-phase, 120–200 words, 5 auto-reject, 26 types, 9 cells) matches `_test_v2` (the Variety/in-progress prompt), NOT the `inspector_prompt.txt` that produced the headline 15K corpus. | [`prompts/inspector_prompt.txt`](prompts/inspector_prompt.txt); [`prompts/inspector_prompt_test_v2.txt`](prompts/inspector_prompt_test_v2.txt); [`scripts/00_generate/generate_realiad_traces.py`](scripts/00_generate/generate_realiad_traces.py) | partially-supported |
| The prompt enforces a SIX-PHASE template (FRAMING/SCAN/FOCUS/EVALUATE/ALTERNATIVES/DECIDE) in `<think>`. | Gemini prompt L25–36 | `_test_v2` (L201–225) has a 6-step A–F flow mapping 1:1, but is not labelled "SIX-PHASE"/numbered; eval system prompt has no six-phase wording. The label is a thesis paraphrase/idealisation. | [`prompts/inspector_prompt_test_v2.txt`](prompts/inspector_prompt_test_v2.txt); [`scripts/04_eval/`](scripts/04_eval/) eval prompt. | partially-supported |
| QC is purely in-prompt auto-reject (no second model, no rubric). | Gemini prompt L65–71; intro | Matches stated QC design (5 auto-reject bullets, no rubric shown). BUT `inspector_prompt_test_v2.txt` itself embeds an 8-category 0/1/2 SCORING rubric + 13/16 PASS/FAIL + REWRITE POLICY (L349–463) which the appendix omits — a redacted view of the source. | [`prompts/inspector_prompt_test_v2.txt`](prompts/inspector_prompt_test_v2.txt) L349–446. | partially-supported |
| Auto-reject rules forbid overlay/label/GT/mask leakage, ordinal/other-image refs, premature naming, definitive causality. | Gemini prompt L65–71 | Each appendix bullet maps onto a `_test_v2` source rule (A–F, L54–109, L404–426). | [`prompts/inspector_prompt_test_v2.txt`](prompts/inspector_prompt_test_v2.txt) L54–109, L404–426. | supported |
| Generation inputs: original photo, optional red overlay (auxiliary, never named), optional defect-free reference. | Gemini prompt L16–23 | Matches ground truth (3-image anomaly / 1 normal) and `_test_v2` L16–24, L146–177. | [`prompts/inspector_prompt_test_v2.txt`](prompts/inspector_prompt_test_v2.txt) L16–24, L146–177. | supported |
| A single inference system prompt served SFT, GRPO and eval (two-image ref-vs-test). | §app:p-inf L78, L81–98 | Actual prompts differ: eval is single-image ("Inspect the provided image"); GRPO is two-image with different wording; neither says "six-phase". Output-tag contract IS consistent. Appendix is an idealised composite. | [`scripts/04_eval/`](scripts/04_eval/) eval prompt; [`scripts/02_grpo/stage_rl/grpo_ad.py`](scripts/02_grpo/stage_rl/grpo_ad.py) L113–136. | partially-supported |
| Inference prompt instructs "same controlled vocabularies as training" + six-phase template. | Inference prompt L90, L97 | Actual eval prompt has neither phrase, only loose examples ("e.g., scratch, dent..."); GRPO prompt has no six-phase. Idealisation. | [`scripts/04_eval/`](scripts/04_eval/) L57–66; [`scripts/02_grpo/stage_rl/grpo_ad.py`](scripts/02_grpo/stage_rl/grpo_ad.py) L93–136. | partially-supported |
| The 120–200 word think budget shaped the realized corpus. | Gemini prompt L42,48,74 | sft_train (n=10,236) mean think-words 136.9 / median 137 → matches ~141 mean, just under target; ~55.6% strictly in 120–200, long tail (max 763). `new_sft_c1_train.json` (n=4,236, mean 416.8) clearly does NOT obey the budget → a different generation. | [`traces/anomalythink_15k/sft_train.json`](traces/anomalythink_15k/sft_train.json); [`traces/anomalythink_15k/c1_only_fixed/new_sft_c1_train.json`](traces/anomalythink_15k/c1_only_fixed/new_sft_c1_train.json) | supported |
| Both prompts share the output schema: JSON {image_id, reasoning} for gen; `<think>/<answer>`(+`<type>/<location>` if anomalous) for inference. | Gemini prompt L38–49; Inference L89–96 | `_test_v2` L228–250 + eval `extract_tags()` parse think/type/location/answer; anomalous-only type/loc conditioning matches. | [`prompts/inspector_prompt_test_v2.txt`](prompts/inspector_prompt_test_v2.txt) L228–250; [`scripts/04_eval/`](scripts/04_eval/) extract_tags. | supported |

## Appendix C — Per-Product Breakdown (largely fabricated; flagged)

> **Audit verdict.** Base and SFT per-product columns are genuine; the **SFT+GRPO per-product column and the entire CM table are FABRICATED**. The CM rows are physically impossible (per-product TP+TN+FP+FN exceeds that product's sample size for 11/15 products) and sum to TP1150/TN382/FP29/FN272 ≠ the genuine Total TP971/TN383/FP61/FN255. Only the column Average (82.73) and CM Total are genuine. Correct per-product GRPO BA (run-2 ckpt-530 DS): cable 68.16, pill 75.45, toothbrush 83.33, screw 72.89, transistor 75.42, grid 94.74, leather 98.44, zipper 68.84, capsule 69.78, hazelnut 88.57, bottle 76.27, carpet 95.51, wood 97.50, tile 89.83, metal_nut 87.47.

| Claim | Where | How it was derived | Evidence | Status |
|---|---|---|---|---|
| Per-product BA in tab:perprod-full computed from CM counts of eval JSONs. | caption L10 | True for Base & SFT (recompute matches ~0.2). FALSE for SFT+GRPO column (doesn't match genuine ckpt-530). | recompute from `results` arrays. | partially-supported |
| SFT+GRPO per-product BA column (ckpt-530) is genuine. | tab:perprod-full L18–34 | Recomputed from [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json): printed cells differ substantially (hazelnut 73.10 vs 88.57; tile 82.90 vs 89.83). FABRICATED. | per-product recompute. | unverified |
| Per-product CM counts in tab:perprod-cm are genuine. | tab:perprod-cm L51–65 | 11/15 product rows exceed that product's sample size (metal_nut 130>92, grid 105>76); rows sum 1150/382/29/272 ≠ genuine 971/383/61/255. FABRICATED. | per-product sample sizes from JSON metrics; row-sum check. | unverified |
| Genuine SFT+GRPO ckpt-530 DS BA = 82.73, CM TP971/TN383/FP61/FN255. | Average L34, Total L67 | metrics block; BA recompute. | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json); [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt) | supported |
| Fabricated SFT+GRPO cells were chosen so their 15-product mean = genuine 82.73. | Average L34 | Mean of printed cells = 82.7333 → 82.73, the real overall BA, though cells don't match per-product data → back-filled to the headline average. | arithmetic. | supported |
| Base per-product BA column is genuine. | tab:perprod-full L18–34 | Recompute from [`results/qwen25vl_baseline_eval/`](results/qwen25vl_baseline_eval/): all 15 match within ~0.1–0.5 (printed truncated). | per-product recompute. | supported |
| SFT per-product BA column (ckpt-564) is genuine. | tab:perprod-full L18–34 | Recompute from [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json): match ~0.02–0.20. | per-product recompute. | supported |
| Delta SFT / Delta GRPO columns internally consistent with adjacent BA columns. | tab:perprod-full L18–34 | All 30 delta cells = adjacent differences (<0.011). But Delta GRPO inherits the fabricated GRPO column → not the true per-product gains. | arithmetic check. | partially-supported |
| Per-product subdivisions sum to a different total "due to MMAD per-product test-set design". | footnote L72 | Misleading: genuine per-product CMs DO sum exactly to 971/383/61/255. The discrepancy comes from FABRICATED rows, not test-set design. | genuine recompute sums exactly. | unverified |
| Chapter (Results) image-level metrics are authoritative. | footnote L72 | Defers to Ch.06; genuine 82.73 is authoritative, this table body is unreliable. | cross-ref Ch.06; [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt) | supported |
| GRPO adds gains for most products (positive Delta GRPO for all 15). | tab:perprod-full Delta GRPO | With genuine GRPO BA, GRPO REGRESSES several products vs genuine SFT (bottle 80.48→76.27, transistor 80.42→75.42) while improving others (hazelnut 66.61→88.57). Uniformly-positive column is a fabrication artefact. | genuine SFT 564 vs GRPO 530 per-product recompute. | unverified |
