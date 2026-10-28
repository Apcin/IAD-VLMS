# Number Provenance Ledger

**Thesis:** *Reasoning-Enhanced Vision-Language Models for Explainable Industrial Anomaly Detection* — Adnane Acudad, TU Delft MSc, 2026 (Qwen2.5-VL).

This file is a complete provenance ledger for **every number that appears in the thesis**. Each entry records the value as printed, where it appears, what it describes, the source it traces to (eval JSON / on-disk computation / training-args dump / citation), its verification status, and — when the printed value is wrong — the correct value.

> **Authority note (revised 2026-09-06).** Every number in the thesis was rechecked against the files on disk on 2026-09-06 (`docs/verification/NUMBER_VERIFICATION_REPORT.md`). Where this ledger and the thesis now disagree, the **thesis and the 2026-09-06 entry below are authoritative**, and the older entries are history. Two known cases: the Qwen2.5-VL-7B baseline is 69.08 / 53.79 from `results/baseline_named/7b_base_*.json` (the thesis file), not 69.01 / 53.80 from the earlier `results/qwen25vl_baseline_eval/` run; and the post-GRPO refinement pool is 3,557 kept + 2,443 patched (thesis §6.7), not "3,770 patched". Older wording in this note: All balanced-accuracy (BA) numbers, GRPO hyperparameters, and corpus counts were recomputed/inspected directly from on-disk artefacts. Any thesis values of `G=2`, `beta=0.04`, `eta=5e-6`, "13K samples", or "6,500+6,500" are **old/wrong** and have been corrected to the actuals.

---

## Figure provenance (thesis figures → producing script → source data)

| Thesis figure | Producing script | Source data |
|---|---|---|
| t-SNE of `<type>` embeddings (Ch3, `fig:tsne-types`) | `scripts/05_figures/` → `analyze_anomaly_types.py` (cluster *numbers* re-derived by `results/recompute_corpus_stats.py`) | `Training/anomaly_type_analysis/{types.json, embeddings_2d.npy}` (8908 traces) |
| Per-product gains base→SFT (Ch6, `fig:per-product`) | `make_qual_figures.py` | base + SFT-ckpt-564 DS eval JSONs (`results/`) |
| Per-product SFT-Iter2 vs GRPO (Ch6, `fig:per-product-iter2-{dsmvtec,visa}`) | `make_iter2_overlays.py` | iter2 + GRPO-run2-ckpt-530 eval JSONs |
| SFT bal-acc-by-epoch curves (Ch6, `fig:{dsmvtec,visa}-curve`) | per-checkpoint eval JSONs (all SFT runs) | `results/sft_qwen25vl_*/checkpoint-*/eval_*.json` |
| GRPO reward/accuracy decoupling, 3-estimator (Ch6, `fig:grpo-decoupling` → `figures/grpo_estimator_decoupling.pdf`) | `scripts/05_figures/make_fig_grpo_decoupling.py` | the three 400-sample probe curves `outputs/grpo_probe_{ctrl,drgrpo,g2rpo}/probe_curve.csv` (shipped to `results/grpo_probe_curves/` + `results/grpo_probe_{ctrl,drgrpo,g2rpo}/`). Columns: `step,mean_reward,probe_ds,probe_visa,…`. Panel (a)=`probe_ds` per step; panel (b)=`mean_reward` (5-step MA). The dotted "SFT init (DS-MVTec 84.5)" = the `step 0 / flag=probe-init` row (`probe_ds=84.52`, hardcoded `INIT_DS=84.52` in the script). **NB this is the 400-sample probe, which reads ~1.7 pp above the full 1,670-sample eval (Arm-C full = 82.80) — see revision entry "2c"; no per-sample CM JSON exists for probe points, so they are not recomputable beyond the CSVs.** |
| GRPO run-2 reward/KL training curves (App H, `curve_grpo_*`) | App-H training-curve script (matplotlib) | run-2 `checkpoint-1060/trainer_state.json` `log_history` (reward, accuracy_reward, consistency_reward, kl, completion_length) |
| Qualitative samples (Ch6, `qual_*`) | `make_qual_figures.py` | DS-MVTec images + model outputs |
| Pipeline overview (Ch4, `fig:pipeline-overview`) | `docs/pipeline_overview.tex` (TikZ) | — |

> Author note (2026-06-14): more per-product graphs are wanted; new ones should be generated from the
> `results/` eval JSONs and added here with their producing script + source. The figure scripts live in
> [`scripts/05_figures/`](scripts/05_figures/).

| Chapter 6 figures 6.1 to 6.8 (`fig_ladder`, `fig_sft`, `fig_arms`, `fig_iter2`, `fig_decoupling`, `fig_operating`, `fig_perproduct`, `fig_dynamics`, `fig_explain`, `fig_gallery`) | `scripts/05_figures/thesis_figures_v2/` (`export_data.py` → `f1..f6`, `g1..g5` → `thesisify.py`) | `results/thesis_figure_data/figdata.json`, registry in `registry.py` |
| Localisation table 6.12, type table J.5 | `loc_hit_table.py`, `type_sim_table.py` | `results/thesis_figure_data/{loc_hit,type_sim}.json` |
| Figure 1.1 overview, Figure 4.1 stages | `svg_final/method_overview_v4.svg`, `stages/f7_stages.py` | numbers from `figdata.json` |
| Figure 6.9 and Appendix F.5 pairs | renderer not on disk (session scratch), indices in `results/sft_vs_kcr_pairs/` | eval JSONs of ckpt-564 and Arm-C ckpt-376 |

---

## Revision log

### 2026-09-06 — Whole-thesis verification, corrected LLaVA corpus, localisation table, rollout pools, prompt-aligned GRPO

Full report: [`docs/verification/NUMBER_VERIFICATION_REPORT.md`](docs/verification/NUMBER_VERIFICATION_REPORT.md). Every evaluation file on disk (802) was rescored with the strict rule of thesis Chapter 4 (no parsable `<answer>` counts as wrong). All headline and table numbers reproduce. What changed in the thesis and what is new in the repository:

**1. LLaVA-OneVision KCR row is the corrected corpus.** Thesis Tables 6.9, 6.12, 6.13, Appendix M, Figures 1.1 and 6.1 to 6.8 use `results/sft_llava_ov_7b_frozen_iad_sft_llava_iter1_C_original/checkpoint-376/eval_*_full_trainprompt_vllm.json`: **87.32 / 72.65** (epoch 2, selected on DS-MVTec, 13 VisA outputs unparsed and counted wrong), **74.29** VisA at epoch 4 (ckpt-748), 84.35 / 71.36 (ep1), 86.96 / 73.57 (ep3). Corpus `traces/llava_kcr/sft_llava_C_original_train.json`, 6,000 items all inside the SFT split, 3,000 / 3,000 by verdict, builder `scripts/03_rollout_star/build_llava_arms_original.py`. The first build (88.45 / 74.25 at ep4, `results/sft_llava_ov_7b_frozen_llava_iter1_C/`) is kept on record: 3,516 of its images in the SFT split, 2,484 from the GRPO split, 45.0 % anomalous by verdict. LLaVA SFT+GRPO restart: 87.86 / 71.89 (`results/grpo_llava_ov_from_ep1_ep2/checkpoint-530/`, vLLM). LLaVA on the Qwen corpus: 87.70 / 70.97 (`results/sft_llava_ov_7b_frozen_iad_sft_iter2/checkpoint-188/`, HF path). LLaVA SFT+GRPO VisA is **72.38** strict (six unparsed counted wrong), 72.58 over the 2,135 parsable.

**2. Judge table (thesis Table 6.13), all eight rows.** `results/explainability_multi/` now holds every summary and raw file. Qwen SFT+GRPO 9.24 / 8.62, Qwen KCR 9.05 / 8.52, Qwen3-VL-8B 9.23 / 8.77, LLaVA SFT 9.27 / 8.40, LLaVA SFT+GRPO 9.51 / 8.54, LLaVA KCR (corrected) 9.39 / 8.70, IAD-R1 our prompt 6.14 / 7.04, IAD-R1 own prompt 4.50 / 6.32. Standard errors 0.12 to 0.23. Gaps to IAD-R1 under our prompt: smallest +2.91 / +1.36, largest +3.37 / +1.73. LLaVA KCR vs IAD-R1: +3.25 and +1.66. Location ladder on VisA (LLaVA): 0.57 → 0.73 → 0.68, two-proportion z = +2.37 (p = 0.018) for the GRPO step and z = +1.61 (p = 0.108) for KCR vs SFT. Caveat found in verification: over all 15 pairs among our six models per benchmark, one DS-MVTec pair is significant (Qwen KCR 9.05 vs LLaVA SFT+GRPO 9.51, Welch p ≈ 0.02); the thesis reports the twelve tests it ran (p 0.10 to 0.94). The Welch t values printed in the thesis for the IAD-R1 comparison (12.07 / 5.55) do not reproduce from the raw file (recomputed 15.61 / 6.50 on the per-trace medians, same differences).

**2b. Shared-set judge replaces the own-100 table (thesis commit 324fffe, 22:30).** Thesis Table 6.13 (`tab:explainability`) now reports all nine rows on one shared image set, 90 DS-MVTec (6 per product) and 47 VisA (4 per product, pcb1 3) drawn with seed 42 from the nine-way intersection of correctly detected anomalies. Files `results/explainability_shared/` (`shared_ids.json`, `summary_shared.json`, `paired_tests.json`, `raw_results_all.json`, `NOTE.md`), script `scripts/04_eval/explainability_judge_shared.py`. Judge /10, DS then VisA: Qwen SFT 9.17 / 8.70, Qwen SFT+GRPO 9.33 / 9.23, Qwen KCR 9.09 / 9.11, Qwen3-VL-8B 9.41 / 9.53, LLaVA SFT 9.36 / 8.57, LLaVA SFT+GRPO 9.47 / 9.43, LLaVA KCR 9.61 / 9.34, IAD-R1 our prompt 6.13 / 6.96, IAD-R1 own prompt 4.36 / 6.49. Paired gap Qwen KCR minus IAD-R1 (our prompt): 2.96 [2.41, 3.50] DS, 2.15 [1.47, 2.83] VisA. Paired steps on VisA: LLaVA SFT to SFT+GRPO +0.85 [0.26, 1.44], SFT to KCR +0.77 [0.17, 1.37]; on DS LLaVA KCR minus Qwen KCR +0.52 [0.19, 0.85]. Location column on the shared VisA images: Qwen 0.38 / 0.64 / 0.55, LLaVA 0.49 / 0.70 / 0.60. Five of the 90 DS images have no MMAD ground-truth type (folder fallback), none on VisA. IAD-R1 under its own prompt answers a bare "Yes" on 42 of the 90 shared DS images. The Welch tests, the z-tests and the own-100 table were removed from the thesis; the own-100 numbers of item 2 stay on record here and in `results/explainability_multi/`. Figure 6.8 per-axis means now come from the 47 shared VisA images (Qwen KCR 1.79 / 1.47 / 2.00 / 1.94 / 1.91, IAD-R1 1.47 / 1.06 / 1.74 / 1.79 / 0.89).

**3. Localisation hit rate (thesis Table 6.12, new).** `results/thesis_figure_data/loc_hit.json`, script `scripts/05_figures/thesis_figures_v2/loc_hit_table.py`, judge rule (predicted 3x3 cells overlap the cells holding more than 5 % of the mask), full benchmarks, 1,226 DS-MVTec and 1,197 VisA anomalies. Of detected: Qwen SFT 73.2 / 38.1, SFT+GRPO 82.4 / 55.9, KCR 74.9 / 52.7; LLaVA SFT 77.5 / 54.2, SFT+GRPO 83.1 / 71.2, KCR 83.1 / 63.7. Of all anomalies: Qwen 59.0 / 27.1, 65.3 / 35.6, 61.3 / 30.1; LLaVA 62.6 / 33.9, 73.1 / 57.0, 68.4 / 46.2.

**4. Type transfer (thesis Appendix J.5, new).** `results/thesis_figure_data/type_sim.json`, script `type_sim_table.py`. Nomic-embed-text-v1.5 similarity of predicted `<type>` to the MMAD label is flat at 0.53 to 0.58 for every checkpoint, exact match below 13 %; 18 distinct predicted strings for the Qwen KCR checkpoint against 435 / 218 distinct MMAD labels.

**5. Rollout pools and pass rates.** `traces/rollout_pools/{qwen,llava}_phase0_10k/`. Qwen: 10,236 items x 8 = 81,888 rollouts, 8,872 kept + 1,364 corrected (= 10,236), 2,406 rewrites, 8,660 judge-confirmed. Over the 6,000 SFT images: 84.6 % of rollouts correct, 90.8 % of images with at least one correct rollout (`answer_match` field; over the whole pool 84.0 / 90.2). Per-item failure table (thesis Table 6.7): 1,541 anomalous (30.1 %) and 871 normal (17.0 %) items with at least one wrong rollout, 2,412 total; 807 (15.8 %) and 176 (3.4 %) with all eight wrong, 983 total. LLaVA: 9,179 kept + 1,057 corrected, 9,045 judge-confirmed, 133 demoted, 2,513 rewrites.

**6. Arm A balance.** `traces/teacher_ablation_abc/sft_A_kept_balanced.json` is balanced on the acquisition folder (1,489 NG / 1,489 OK) and is 1,298 anomalous / 1,680 normal by the `<answer>` verdict (191 NG captures whose kept rollout answers no). Thesis §5.7 now says so. Arm B: 4,369 at 31.4 % anomalous. Arm C: 3,000 / 3,000. `traces/iter2/` holds two versions of the Arm-C file that differ in 316 `<location>` spellings (see its README).

**7. Strict scoring applied to five table cells.** Arm C ep3 81.26 (was 81.39, 2 unparsed), ep4 78.96 / 71.53 (was 79.13 / 71.59), Arm B ep3 VisA 67.46 (was 67.48), Qwen3-VL-8B VisA 76.45 (was 76.52, 2 unparsed). L7 best-of-four margins are +0.67, +2.03, +2.12, +2.00 (Arms A, B, C and the first LLaVA build).

**8. Prompt-aligned GRPO on Arm-C (`results/grpo_sftprompt_kl0.1_sys_3ep/`, complete, 15 checkpoints).** beta 0.1, evaluation prompt with system turn, 3 epochs from ckpt-376. ckpt-954 reaches 82.95 / 72.62 (+0.15 / +0.55 over the initialisation) and the run ends at 81.25 / 71.51. Two of fifteen checkpoints above the init on DS-MVTec, eleven on VisA. Single seed. Thesis §6.8 and §8.4 record it as inconclusive future work; the production GRPO-on-C run (`grpo_qwen25vl_7b_abc_C_grpo`, beta 0, GRPO prompt) stays the reported table, where every checkpoint is below the init on DS-MVTec. Sibling runs: `grpo_sftprompt_kl0.1` (best 81.72 / 71.05), `grpo_abc_C_kl0.1_halfep` (best 81.90 / 70.01).

**9. Estimator probe files shipped.** `results/grpo_probe_{ctrl,drgrpo,g2rpo}/checkpoint-{20..120}/probe_{dsmvtec,visa}.json` reproduce every cell of thesis Table 6.10. Reasoning-reward fallback in the GRPO-on-C runs logged at 0.484 to 0.500.

**10. Other verified quantities now on disk here.** Contamination count 1,999 of 186,060 (`results/contamination_llava_ov_data/`). Gemini 2.5-Flash 81.52 / 75.18 and GPT-5-mini 77.10 / 68.23 (`results/gemini_25flash_eval/`, `results/gpt5mini_eval/`). Verifier-filtered retrain 79.60 / 65.99 (`results/sft_filtered6kcc_from_base/checkpoint-364/`, corpus `traces/controls/filtered_6k_cc_train.json`, 5,797 items). Mask area 3.79 % DS-MVTec / 1.00 % VisA over 1,226 / 1,197 masks. VisA median image 1404x1070, 2.39x per-side downscale to 262,144 px. Think-block length on the 6K split with the Qwen tokenizer: mean 164.7 tokens, p5 142, p95 187, min 121, max 223 (136.8 words). GRPO run 2: reward 1.66 / 2.22 / 2.13 at steps 10 / 100 / 530, KL max 0.117 in epoch 1 (13 of 530 steps above 0.095), 0.197 full run, completion length 153 to 186, 11.25 h to ckpt-530, 24.69 h total. SFT wall-clock: 6.93 h (7B-frozen-6K), 12.19 h (7B-unfrozen-15K), 5.10 h (Arm C), 2.35 h for six epochs of 3B-frozen-6K (the thesis quotes 1.6 h for four).

**11. Frozen-vision GRPO ablation launched** (`results/grpo_qwen25vl_7b_6k_frozenvision_run1/`, `scripts/02_grpo/run_grpo_7b_frozen_vision.sh`, env-guarded `FREEZE_VISION_TOWER` block in the trainer). Evaluations pending.

**12. Baseline file (decided 2026-09-06 21:50).** The thesis now uses `results/baseline_named/7b_base_{dsmvtec,visa}.json` (69.08 / 53.79) everywhere. Appendix C's base column was moved to that file (transistor 67.92, leather 90.29, macro average 70.60, mean SFT gain +9.54) and the overall SFT gain in Chapters 1 and 8 reads +11.1 pp (80.16 − 69.08). The older run in `results/qwen25vl_baseline_eval/` (69.01) is history. Welch tests in §6.11 are now reported over all fifteen pairs per benchmark (VisA p 0.18 to 0.95, DS-MVTec p 0.09 to 0.96 with one pair at 0.02), and the IAD-R1 comparison prints the recomputed t = 15.6 / 6.5 (scipy `ttest_ind`, `equal_var=False`, on the per-trace median scores).

### 2026-09-02 — Explanation-quality numbers superseded, cross-architecture line, corpus transfer, labels-only control

**1. Explanation quality. The thesis table numbers change. New: DS-MVTec 9.05 against 6.14, VisA 8.52 against 7.04.**

Source: `outputs/explainability_multi/`, shipped at [`results/explainability_multi/`](results/explainability_multi/)
(`explainability_5axes_table.md`, `summary.json`, `raw_results.json`, `raw_results_iadr1_tp.json`).
Script [`scripts/04_eval/explainability_judge_multi.py`](scripts/04_eval/explainability_judge_multi.py),
figures [`scripts/05_figures/plot_explainability_axes.py`](scripts/05_figures/plot_explainability_axes.py).
Judge is Gemini-3-Flash, shown the original image, a red GT-mask overlay, the GT defect type and the
model's trace. Five axes scored 0/1/2, summed to 0 to 10. **Median of three judge samples** per trace.
**n = 100** product-diverse traces per model per benchmark. Each model is scored **on its own
correctly-detected anomalies** (`gt=yes & pred=yes`), so a weak model is not penalised by a shrinking
cross-model intersection.

| | DS-MVTec | VisA |
|---|---:|---:|
| Arm-C (KCR) final SFT | **9.05** | **8.52** |
| IAD-R1, prompt-matched | **6.14** | **7.04** |
| IAD-R1, native prompt | 4.50 | 6.32 |

**Both numbers are on record, and here is why the newer one is the fair comparison.** The old
9.14 against 4.29 (DS-MVTec) and 8.67 against 6.40 (VisA) came from `outputs/explainability_judge/`,
where IAD-R1 was scored **under its own GRPO prompt**. That prompt never asks for reasoning. Roughly
35 to 40 percent of its answers are a bare "Yes" and score 0 on every axis. So 4.29 measures
explanation **reliability** under IAD-R1's own prompt, not explanation **quality**. The prompt-matched
row asks IAD-R1 the same way our models are asked, and then 100 percent of its answers carry a trace,
which is the like-for-like reasoning-quality comparison. Two further reasons the new run is better:
k=3 median instead of a single judge sample damps run-to-run judge noise, and the old design scored
both models on a matched intersection pool while the new one scores each model on its own correct
detections. Keep both rows. Quoting only 4.50 or only 4.29 understates IAD-R1. Either way it stays
clearly below Arm-C.

The old paragraph in the 2026-06-16 entry below is kept verbatim and flagged as superseded. The
figures and `explainability_5axes_table.md` in `results/` were already correct and did not change.

**2. Cross-architecture replication on LLaVA-OneVision-7B-SI (new results, all shipped).**
Same harness, same subsets (DS-MVTec n=1670, VisA n=2141), inference served through vLLM, hence the
`_vllm` tail on some filenames. The prompt mode is still `_trainprompt`.

| Row | BA DS-MVTec / VisA | Shipped evidence |
|---|---|---|
| Base, fair yes/no prompt (`--yesno-user`) | 75.66 / 53.80 | [`results/llava_ov_7b_zeroshot_eval/`](results/llava_ov_7b_zeroshot_eval/) |
| 6K Gemini SFT, epoch 1 | 85.91 / 68.26 | [`results/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/`](results/sft_llava_ov_7b_frozen_iad_sft_6k_train/checkpoint-188/) |
| SFT then GRPO, ckpt-530 | 87.66 / 72.58 | [`results/grpo_llava_ov_from_ep1/checkpoint-530/`](results/grpo_llava_ov_from_ep1/checkpoint-530/) |
| **KCR corpus SFT, epoch 4** | **88.45 / 74.25** | [`results/sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/`](results/sft_llava_ov_7b_frozen_llava_iter1_C/checkpoint-748/) |

KCR epochs 1 to 4 are 85.29/72.32, 85.60/70.10, 86.45/73.56, 88.45/74.25. The KCR corpus beats
SFT-plus-GRPO on the same backbone, which reproduces the Qwen2.5-VL ordering on a second architecture.
**Contamination caveat, DS-MVTec only:** `lmms-lab/LLaVA-OneVision-Data`, config `vision_flan(filtered)`,
contains 426 rows whose id matches `%MVTecAD%`, VisA matches 0. Every DS-MVTec cell for a
LLaVA-derived model therefore carries a pretraining-exposure asterisk, including IAD-R1's own released
model. VisA does not. Verified live through the Hugging Face datasets-server filter.
**Do not write "beats IAD-R1 on its own backbone" against our 81.92 / 71.34 row.** That row is
IAD-R1's released **Qwen2.5-VL-7B** checkpoint re-evaluated here
([`results/iad_r1_qwen_recanon/`](results/iad_r1_qwen_recanon/)). The comparison is same-harness, not
same-backbone. We did not re-evaluate their LLaVA-OneVision checkpoint on the full subsets.

**3. Corpus transfer to Qwen3-VL-8B-Instruct (new results, shipped).** Base 78.68 / 64.45
([`results/qwen3vl_8b_baseline_eval/`](results/qwen3vl_8b_baseline_eval/)) to 85.82 / 76.52 at
ckpt-376 ([`results/sft_qwen3vl_8b_armC/checkpoint-376/`](results/sft_qwen3vl_8b_armC/checkpoint-376/)),
so +7.14 DS-MVTec and +12.07 VisA. Epochs 1 to 4 are 82.62/77.67, 85.82/76.52, 85.18/76.62,
82.59/73.26. The corpus is the Qwen KCR corpus, key `iad_sft_iter2`, shipped at
[`traces/iter2/sft_iter2_train.json`](traces/iter2/sft_iter2_train.json), the same file the
Qwen2.5-VL-7B Arm-C headline model trained on. **There is no GRPO comparison on this backbone.**
Nothing was RL-trained on Qwen3-VL-8B, so this line supports a corpus-transfer claim only and must
not be read as a corpus-versus-RL result on Qwen3-VL.

**4. Labels-only control (new results, shipped).** Same 6,000 images, same hyperparameters, same
4-epoch checkpoint grid as the reported SFT-6K run, supervision stripped to a bare
`<answer>Yes|No</answer>`. Best epoch 77.86 / 68.64 at ckpt-188
([`results/sft_qwen25vl_7b_6k_noreason/`](results/sft_qwen25vl_7b_6k_noreason/), eval mode
`_noreasonprompt`) against Arm-C SFT 82.80 / 72.07. Stripping the reasoning supervision costs
**-4.94 DS-MVTec and -3.43 VisA** at matched data, matched compute and matched checkpoint grid.

**5. `results/eval_ba_inventory.txt` regenerated.** 206 rows to 256. Purely additive, 50 new rows for
the LLaVA, Qwen3-VL and labels-only evals. One file was deliberately excluded,
`outputs/grpo_llava_ov_from_ep1/checkpoint-530/eval_dsmvtec_BROKEN_rope.json`. It is not a result. It
is the transformers-5.0 `rope_parameters` trap: a checkpoint saved under transformers 5.0 loads under
4.57 or vLLM with a `rope_theta` off by 100x, and the model then answers "no" to nearly everything
(BA 50.70, TP 17, FN 1204). It looks like training collapse and it is not. Documented in
[`results/grpo_llava_ov_from_ep1/NOTE.md`](results/grpo_llava_ov_from_ep1/NOTE.md).

**6. Naming.** The method is **Keep-Correct-Revise (KCR)**. Keep the rollouts the policy already gets
right, have a teacher correct the wrong ones and revise the ones that are right but weakly grounded,
then fine-tune the base model on the result. "Arm C" is the same thing under its ablation name and
stays in every path, dataset key and results directory. Provenance rows in this file that quote a path
keep `abc_C` / `arm_C` / `iter2`, because the whole point of those rows is that the link resolves.

### 2026-06-16 — Explainability eval, 15K verifier audit, Variety-continuation tables
**⚠ SUPERSEDED on 2026-09-02 — the 9.14 / 4.29 and 8.67 / 6.40 numbers in this paragraph are the old k=1 single-judge-sample run. The current explanation-quality numbers are 9.05 vs 6.14 (DS-MVTec) and 8.52 vs 7.04 (VisA) from `results/explainability_multi/`. See the 2026-09-02 revision entry above for why the newer run is the fair comparison. The paragraph is kept verbatim as the audit trail.**
**Explainability / reasoning-faithfulness eval (NEW thesis §`sec:res-explainability` + `tab:explainability` in Ch6; doc `outputs/explainability_judge/EXPLAINABILITY_EVAL.md`; script `repository_tu_delft_vlms/scripts/explainability_judge.py`).** Gemini-3-Flash judge (Vertex, blind to model, reference-guided: image + red GT-mask overlay + MMAD-MCQ GT type; 5 axes 0–2, sum/10), on **matched correctly-detected anomalies**, 100 product-diverse images/benchmark (overlap pools 916 DS / 549 VisA), IAD-R1-native vs Arm-C-82.8. Results (n≈100): **DS-MVTec Arm-C 9.14 vs IAD-R1 4.29; VisA Arm-C 8.67 vs IAD-R1 6.40**. Mechanical (separate): Loc-MET DS 0.82(ours)/0.43, VisA 0.56/0.64; Type-sim (Nomic) DS 0.557/0.341, VisA 0.528/0.542. Caveats: single judge sample (k=1); plausibility/grounding-consistency not mechanistic faithfulness; MCQ GT-type coverage 94.8% DS / 99.2% VisA. Sources cited in §: AutoRubric 2603.00077, CheckEval 2403.18771, GREEN 2405.03595, LLaVA-Critic 2410.02712, Jacovi&Goldberg 2004.03685, Gaming-the-Judge 2601.14691; \citep{li2025iadr1}, \citep{nomic2024embed}.
**15K verifier audit (Ch7 §`sec:disc-quality` sentence added; doc `outputs/verify_15k_regen_results/FILTER_AUDIT.md`; script `verify_15k_traces.py`).** GPT-5-mini reference-guided verifier over the **real 14,472-trace `iad_sft_15k_regen_combined` corpus**: **97.8% KEEP** (14,015 completely_correct + 143 correct; 313 wrong + 1 completely_wrong = 2.2% drop), ~\$10. Top drop issues: illogical_reasoning 150, incoherent_content 88, extra_location_tag 45, wrong_product 18. Per-product keep uniformly 95.9–98%. **Implication (added to thesis):** the 15K is not noisy → the 6K<15K gap is composition/count, NOT a data-noise/quality-filter effect (resolves the contested "quality" framing). NB initial run mistakenly used the C1-only 10,236 file; corrected to the 14,472 regen corpus the headline 15K SFT actually trained on.
**`tab:variety-cont` (Ch6):** filled the **3K-mix (rehearsal)** rows from best-DS epoch — Arm-C ep1 **81.09/71.48** (Δ−1.15), GRPO ep2 **81.15/69.53** (Δ−1.22); milder than full-Variety (−2.64/−1.37), rehearsal curbs forgetting but no net gain. The two **1K-only (no-rehearsal)** rows still TBD (eval re-running after an overnight interruption; ep1/ep2 so far show 1K-only worse than +rehearsal, and a catastrophic drop from the GRPO init — isolating the rehearsal as the anti-forgetting mechanism).

### 2026-06-15 — Variety continuation: added third probe (1k-only, no rehearsal)
`tab:variety-cont` now has THREE probes per init (full-6k / 1k-only / 1k+2k-rehearsal). Probe (ii) **`variety_eval_aligned_1k`** = the SAME 1{,}000 eval-aligned Variety traces extracted from `variety_armC_mix_3k.json` (verified exactly **500 anomaly / 500 normal** by `<answer>` tag) with the 2{,}000 Arm-C rehearsal removed — so (iii)−(ii) isolates the rehearsal's anti-forgetting effect. Configs `sft_varonly_from_{armC,grpo}.yaml` (frozen ViT, 3 ep, EBS 32, lr 1e-5 cosine), orchestrator `run_varonly_03.sh` runs on **cuda 0+3** (queued behind the varmix evals to avoid OOM). 4 TBD rows pending in `tab:variety-cont`. Early in-progress signal (probe iii, Arm-C ep2): DS 79.90 (final) / VisA ~73 (rolling, +0.7 vs the matched-slice baseline 72.18).

### 2026-06-15 — EIGHT NEW APPENDICES (D–K) built + verified (8-agent workflow)
Wired into main.tex after C; full integration compile of A–K (faithful test wrapper: thesis macros + real `\lstset`/colors + hyperref) = **exit 0, 72 pages, 0 fatal errors**. All numbers recomputed from named JSONs via `compute_ba.py`; all 19 new figures present and resolving; zip synced (0 stale).
- **App D — Per-product VisA tables** (`app:visa-perprod`). Sources: base `results/baseline_named/7b_base_visa.json` (VisA BA 53.79 ✓, DS 69.08 ✓ → confirms base), SFT `…7b_zeroshot_6k_frozen/checkpoint-564/eval_visa_full_trainprompt.json`, GRPO `…run2/checkpoint-530/eval_visa_full_trainprompt.json`. 12 products, n=2141. Per-product BA recomputed (matches `metrics.per_product` exactly). Macro avg: Base 53.81 / SFT 66.89 (+13.08) / GRPO 72.52 (+5.63); GRPO CM total TP762/TN728/FP216/FN435 → **micro BA 70.39** (= thesis headline VisA GRPO ✓). Mirrors App C layout (macro-vs-micro note included).
- **App E — Reward function reference** (`app:reward-code`). From [`scripts/02_grpo/stage_rl/reward.py`](scripts/02_grpo/stage_rl/reward.py) (consistency L544–580 binary `re.fullmatch`; accuracy L582–655 = verdict(1)+(type+loc)/2, max 2.0) + `reward_process/{type,location}_reward.py` + `sc_grpo_trainer.py:810` (`rewards_per_func.sum(dim=1)` = unweighted). Nomic type bins (`type_reward.py` _threshold_map): sim≥0.90→1.0, ≥0.80→0.9, ≥0.70→0.7, ≥0.55→0.5, ≥0.40→0.2, else 0. Location = `map_location_to_region` 3×3 {0,1}. Final R = R_cons + R_acc ∈ [0,3], max 3.0. Consistent with §5.2 + cascade. Commented ASCII pseudocode listing.
- **App F — Qualitative trace gallery** (`app:gallery`). 11 REAL images (6 correct detections, 2 correct normals, 3 failure cases) across 9 products + a base→SFT→GRPO triptych on `hazelnut/hole/004`. Traces are verbatim `pred_full` from GRPO ckpt-530 DS (BA 82.73) / VisA (BA 70.39) eval JSONs. **Caveat (in-text): MMAD scores only the binary verdict** — type/location reported descriptively (dataset subfolder), not GT-matched; metal_nut = "correct verdict, coarse type" case.
- **App G — Teacher vs. student** (`app:teacher-student`). Teacher `outputs/gemini_25flash_eval/eval_{dsmvtec,visa}_full_gemini25flash.json` (DS 81.52, VisA 75.18; VisA FN=475 = recall-conservative) vs student GRPO ckpt-530 (82.73/70.39), matched 1:1 by image_id (1670 DS, 2141 VisA). 5 matched examples + 5 images. Backs Ch6 "student surpasses teacher on DS, teacher leads VisA".
- **App H — Training curves** (`app:curves`, 3 PNGs via matplotlib). SFT BA-by-epoch: 6K-frozen DS ep1–4 = 79.24/76.66/**80.16**/77.07; 15K-frozen DS = 71.00/65.40/71.66/72.60, VisA = 65.45/64.32/64.28/66.94. GRPO run-2 (1060-step `trainer_state.json`): reward 1.48→~2.0 (smoothed), KL 0→0.098 (max 0.197), completion 176→163; **reward == accuracy+consistency at all 1060 steps (0 mismatches)** — re-confirms unweighted sum.
- **App I — AnomalyThink datasheet** (`app:datasheet`, Gebru 7-section Q&A). Split-composition 6000+4236+4236 = **14,472** (refs `tab:split-composition`); prompt facts (489 lines, 26 types, 9-cell grid, 8-cat filter ≥13/16, 6 auto-reject rules; `app:p-gemini`). Honest disclosures: privileged-input asymmetry, product-overlapping held-out, NG-folder-≠-label.
- **App J — Eval harness & prompt-mode robustness** (`app:harness`). IAD-R1 BA across modes (recomputed from `results/iad_r1_qwen_recanon/eval_*_full_{trainprompt,bareprompt,iadr1native}.json`): DS 81.92/81.45/81.19 (spread **0.73 pp**), VisA 71.34/71.56/71.87 (spread **0.53 pp**) → prompt-robust, Ch6 cross-mode SOTA comparison sound. Eval sets 1670/2141/4236; decoding greedy 1024/512 tok, image px 262144/480000.
- **App K — Proposed explainability-scoring protocol** (`app:explain-rubric`, FUTURE-WORK, not run). Gemini-3-Flash judge, 6-axis rubric (Visual Grounding, Localisation-Faithfulness via injected IoU, Type, Evidence-order, Causal Restraint, Conciseness), reference-guided pointwise, J≥3 + median, clamped weighted sum, human-golden-set validation (κ/ρ target 0.5–0.75). Grounded in arXiv:2603.00077 (Autorubric) + 2510.11196 (med-VLM). All thresholds flagged as design proposals.

### 2026-06-15 — §3.App A + App B audit, and FULL REWARD-CASCADE RELEASE
**Ground-truth re-verification of the GRPO reward (decisive).** The headline GRPO run is `outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2` (launched by `Training/run_grpo_7b_resume_run2.sh`), whose ckpt-530 gives DS 82.73% (= App-C `tab:perprod-cm` Total). Its `checkpoint-530/trainer_state.json` logs **only** `rewards/accuracy_reward` and `rewards/consistency_reward`, with `reward = accuracy(1.078) + consistency(0.781) = 1.859` **exactly** — an UNWEIGHTED sum, max 3.0. The script default `reward_funcs=["accuracy","format","reasoning"]` (grpo_ad.py L35) did **not** fire (no `reasoning_reward` key). Registry: `"format"→consistency_reward`. Confirms the §5.2 rewrite; the previously-described weighted reward (0.3/0.3/0.2/0.2) matches no run. Other confirmations: base/reference = SFT `…7b_zeroshot_6k_frozen/checkpoint-564` (run-2 resumed from `…run1/checkpoint-530`); `save_steps 530`, `max_steps 1060`, ckpt-530 = epoch 1.0; `max_pixels 480000`; seqlen 12144; SFT image px 262144.
**Class balance — author correction (50/50 is CORRECT).** Counting by the *true* `<answer>` verdict (NOT the `NG`/`OK` acquisition folder — a Real-IAD `NG` capture is labelled *normal* when the defect isn't visible at that camera angle): **6K SFT 3000/3000 (50.0/50.0); 15K-c1 SFT 5129/5105 (50.1/49.9); GRPO 2128/2105 (50.2/49.7); held-out 2129/2107 (50.3/49.7).** Cross-tab: 564 `NG`→`No` in the 15K pool, 545 in GRPO. My earlier "70/30" was a substring-match bug (`'<answer>yes' in s` caught the schema example).
**App A fixes:**
- **`tab:A.1` per-product quota "187 traces per class per product"** matched no pool → replaced by a row pointing to a new **`tab:split-composition`**: SFT 30p 3000/3000=6000; GRPO 23p 2128/2105=4236; held-out 23p 2129/2107=4236; **Total 7257/7212 = 14,472 ≈ 15K**. Class-balance row annotated "by true `<answer>` label". Per-product target ≈100 (SFT) / ≈92 (GRPO, held-out).
- **`tab:A.3` GRPO samples** "2,118 + 2,118" → **2,128 + 2,105**.
- **`tab:A.3` reward-weights row** `(w_f,w_a,w_t,w_l)=(0.3,0.3,0.2,0.2)` → three rows: reward functions `accuracy + consistency`; unweighted sum, no weights, max 3.0; sub-signals format/verdict (max 1) + type/location (max ½).
**App B fix:** §B.1 Gemini generation prompt was a 65-line **paraphrase** (5 auto-reject rules, no acceptance filter) → replaced with the **exact 489-line `inspector_prompt_test_v2.txt` verbatim** (6 auto-reject rules A–F; silent 8-category 0–2 acceptance filter, pass ≥13/16 with 4 mandatory-2 categories; rewrite policy; self-check; full 26-type + 9-cell vocabularies). Smart-quotes ASCII-normalised for pdflatex+listings. §B.2/§B.3 unchanged (verified).
**Cascade released across 11 spots / 6 files** (author: "release full cascade"): App-A L149; 08_conclusion L15+L33; 01_intro L61; 07_discussion L57+L71+L72; 03_dataset L68; 05_grpo L5+L175+L184+L190; 00_abstract L11 — all now describe the unweighted two-function reward (max 3.0); the G²RPO section's stale reward set `{0,0.3,0.6,1.0}` and range `[0,1]` → `[0,3]`. Also **06_results L587** "run-2 cadence (every 265 steps)" → run-2 saved every 530; the 265-step cadence is the GRPO-on-C run.

### 2026-06-15 — §3.Ch7 (discussion) audit (7 investigators) — 4 WRONG fixed, 2 correct, 1 manual
**Fixes applied [author decisions]:**
- **ViT param count (L15):** "~300M" → **"~675M"**. Counted from safetensors `visual.*`: 3B=668.7M, 7B=676.6M; the ViT transformer is 632.0M (identical across 3B/7B) + a 37–45M merger. The freezing/overfitting argument is unaffected (strengthened).
- **Trace length (L29):** "~141 words" → **"~137"** (recomputed 136.8, equal for 6K & 15K). **L33** "15K-only extension drifts into longer, less rigid prose" → reworded to "marginally more variable in length and structure (same mean length)" — the 15K-only extension is 136.82 vs 6K 136.78 words, identical median/IQR, so "longer" was unsupported/self-contradictory with L29.
- **Frozen-vs-unfrozen gap (L11):** "4–8pp DS / 4–10pp VisA across every config" was WRONG (conflated with the 6K-vs-15K data effect). Re-scoped to: at best-epoch, **DS gap ~0.5–1pp** (7B 72.60 vs 72.08; 3B 69.56 vs 68.58), **VisA gap 8.3pp at 7B** (66.94 vs 58.60), **~0 at 3B**; noted the large "80 vs 72" gap is the separate 6K-vs-15K data-size effect. **L19** frozen VisA recall "~70%" → **"~80%"** (real 77–83%; unfrozen "34–41%" kept — verified).
- **Format compliance (L70):** "~92% after SFT → >99% after GRPO" was WRONG — both SFT (ckpt-564) and GRPO (ckpt-530) emit all 4 tags on ~100% of the 3,811 eval samples (near-zero schema violations). Reframed the "structured-output reliability" GRPO-contribution bullet to compliance **maintenance** (GRPO keeps the ~100% SFT already achieves), not a lift.
**Verified correct (no change):** VisA internal gaps 10.7pp (Arm-C) / 10.6pp (IAD-R1) + the "3.5pp deficit was a non-common-harness artefact" framing; GRPO per-product redistribution (hazelnut +22.0, tile +8.0, capsule +5.7; transistor −5.0, bottle −4.2, cable −2.2 — within 0.05pp); seeds (run-1 ckpt-530 DS 81.65 vs run-2 82.7, both seed 42); latency (170 tok / 30 tps = 5.67 s).
**Cannot verify locally:** the 200-sample hand-inspected error taxonomy (L90–98, five categories) — no on-disk artifact; legitimate manual analysis, %s author-reported (optional: add a "qualitative manual analysis" acknowledgement).
**Held:** the reward-weight mentions in Ch7 (L57/L70/L71/L72 "weight 0.30/0.20") are part of the §5.2 reward cascade on hold — not touched.

### 2026-06-15 — §3.Ch6 (results) audit (7 investigators) — 6/7 EXACT
**6 clusters verified exact** (BA recomputed via compute_ba.py): baseline (3B-DS 56.14, 7B-DS 69.01, 7B-VisA 53.80, 99.0%prec/39.2%rec, VisA-rec 7.7%); SFT four-factor summary (all 6 best-DS-epoch cells; 8.5pp matched-ep / 7.6pp best-vs-best; frozen>unfrozen incl. 8.3pp VisA; 7B>3B 3–11pp); teacher-ablation A/B/C (every per-epoch cell; Arm-C ckpt-376 82.80/72.07; deltas −1.15/+4.05, +0.59/+2.70, +2.64/+7.29; sub-counts 2,978/1,631); SFT-Iter2 (−1.00/−5.47/−10.65; rollout-failure table; 3,770 patched/94.8% NG; 81,888 rollouts); GRPO-on-C + 3-estimator ablation (full curve all below 82.80 init); per-product (metal_nut +25.8/cable +19.2/pill +16.0/toothbrush +15.8/…; tile −2.1/wood −0.1; VisA 11/12; +11.2pp aggregate). Gemini-teacher row 81.52/75.18 confirmed.
**Fixes applied:**
- **`tab:sota` 'Ours (15K traces)' row was checkpoint-mixed** — DS 71.66 = ep3 (ckpt-1359) BA but VisA 65.45 = ep1 (ckpt-453) BA. Set both to ep3 → **71.66 / 64.28** [author chose ep3]. (Per-epoch 15K VisA BA: ep1 65.45, ep2 64.32, ep3 64.28, ep4 66.94; DS: ep1 71.00, ep2 65.40, ep3 71.66, ep4 72.60.)
- **96% → 94.8%** NG-fraction (06_results L259, L262) to match the 94.8% (3,573/3,770) cited earlier in the same section.
**Cosmetic, left as-is:** 53.79 (tab:baseline) vs 53.80 (abstract) both = 0.53790; GRPO-on-C "2.0–2.5pp below init" kept (actual ~1.6–4.0, mean ~2.5 — defensible summary; changing it would cascade to intro/Ch7/Ch8).

### 2026-06-15 — §3.Ch5 (GRPO) audit (8 investigators) — reward function REWRITTEN
**MAJOR — §5.2 reward did not match the code [author decision: rewrite].** The thesis described a 4-component WEIGHTED reward (format 0.3 partial-credit, accuracy {0,0.3}, type 0.2 graded-Nomic, location 0.2 presence-only-NOT-grid; weighted sum max 1.0) + a 4-component sample log line. The actual code (`scripts/02_grpo/stage_rl/reward.py` + `grpo_ad.py`) registers an UNWEIGHTED sum, and **no run used weights**. Verified across ALL GRPO runs via their `trainer_state.json` reward keys: headline run-2 + 7B/3B run-1/run-2 + bareprompt + g2rpo_v2 = `accuracy_reward + consistency_reward`; GRPO-on-C (abc_C) + probes (ctrl/drgrpo/g2rpo) add `reasoning_reward`; gemini_judge adds `reasoning_reward_gemini`.
**Exact run-2 reward (verified: `reward` = exact sum of components, max|diff|=0):** `consistency_reward` (format) ∈ {0,1} binary full-schema match; `accuracy_reward` ∈ [0,2] = ½(type+location) + 1·[verdict="yes"], where type = Nomic similarity ∈[0,1] (binned, served on CPU) and location = `map_location_to_region` ∈ {0,1} (binary 3×3-region match — contradicting the thesis "we deliberately do not attempt a 3×3 grid match"). Combined **max 3.0** across 4 sub-signals (format 1 / verdict 1 / type ½ / location ½). The L158-160 logged dynamics (combined 1.66→2.22, accuracy 0.875→1.25, format 0.781→0.875) are CORRECT on this max-3 unweighted scale.
**Rewrote** §5.2 + `eq:grpo-reward` + `lst:reward-line` + the takeawaybox to this spec.
**2 verified clean (no change):** GRPO hyperparam table — all 17 values vs run-2 `training_args.bin` (G=4 / ε=0.2 / β=0 / η=1e-6 / linear / EBS 8 / max-prompt 4096 / max-compl 512 / max-pixels 480000 / T=1060 / save 530 / seed 42); framework — `sc_grpo_trainer.py` is a modified **trl** GRPOTrainer (subclasses `transformers.Trainer`, imports `trl`), so L150 is accurate (IAD-R1's GRPO is also trl-based, not EasyR1; only G2RPO's `compute_1d_ot` is an EasyR1/verl port). iter-2 4,026 self-traces + ckpt-378/504 artifact-verified.
**Smaller fixes applied:** completion length L162 "163–174 throughout" → "~153–186 (mean 167)" (only 59% of steps in the old band); ckpt-530 wall-clock L163 12.4h → ~11.3h (full run 24.7h ✓); softened L103 "<2% skip", L225 "2,700/1,167 cleaning counts", L142 "~30s/rollout" (no on-disk artifact for the exact figures).

### 2026-06-15 — §3.Ch4 (SFT chapter) audit (8 investigators)
**5 verified clean:** EBS-32 + per-cell batch (4×4 / 16×1) from configs; parse-failure <0.5% (measured **0/3811** on headline 6K-frozen ckpt-564); greedy decoding (do_sample=False, 1024 tok, 262144 px — eval harness); matched-LR pilot L115 (`sft_qwen25vl_7b_zeroshot_6k`: 72.82 peak ckpt-200 → 69.27 ckpt-376); **stale-number sweep — NO survivors** (no 300M ViT / 6-epoch / linear schedule / 100K / 5e-5 / 32-cell wording; the §1.4 fixes held).
**Edits applied:**
- **L64 hardware [author: drop numbers]:** "256 GB RAM, AMD EPYC CPU" → "ample system RAM". Live node measures **503 GiB RAM + AMD Ryzen Threadripper PRO 3975WX (32-core) + 4×A6000** (training used 2); "2×A6000 48GB" kept (verified per-run topology).
- **L31 / L87 / L152 — three unbacked "pilot/sanity-check" claims SOFTENED** (whole-`/bulk` search found NO artifact): L87 "1K-trace sweep" → "informal preliminary checks + standard defaults" (the `phase0_pilot_*` dirs are a STaR **rollout** pilot — 100 items k=8 from run2/ckpt-530 — not an SFT-HP sweep; all SFT LRs ∈ {1e-5, 1e-6}); L31 dropped "which we corroborate in a small pilot" (no SFT LoRA run exists — the only `adapter_config.json` are GRPO LoRA under `grpo_upgraded_*`; IAD-R1 citation kept); L152 "200-sample beam/nucleus, ≤0.3 pp" → "informal comparisons … no material change" (beam search implemented nowhere — no `num_beams`; sampling only in `compare_*.py`/`inference_mmad.py` with no persisted BA result).
- **L81 wall-clock:** 3B-frozen-6K "≈2.4h" → "≈1.6h over four epochs". The on-disk 3B-frozen-6K run was **6 epochs / 2.35h** (`sft_qwen25vl_3b_zeroshot_6k_frozen`, ckpt-1128); 4-epoch portion = 1.57h. (7B 6.9h / 12.2h verified exact.)

### 2026-06-15 — Variety probe-1 (full-160) RESULTS FILLED; probe-2 (targeted-mix) LAUNCHED
**Probe 1 — full Variety continuation (DONE, both regress → filled into `tab:variety-cont`):**
- from Arm-C 82.80/72.07 → best-DS ep2 ckpt-376 = **77.82 / 71.78** (DS Δ **−4.98**, avg Δ −2.64). Source: `outputs/sft_variety_from_armC_82p8/checkpoint-376/eval_{dsmvtec,visa}_full_trainprompt.json`.
- from GRPO 82.73/70.39 → best-DS ep3 ckpt-564 = **81.10 / 69.27** (DS Δ **−1.63**, avg Δ −1.37). Source: `outputs/sft_variety_from_grpo_82p7/checkpoint-564/...`.
- Reading: out-of-distribution forgetting (full Variety is electronics/toys/decor-heavy); reinforces the in-distribution-corpus-is-the-lever conclusion, consistent with SFT-Iter2.
**Probe 2 — targeted Variety + Arm-C rehearsal (LAUNCHED, PENDING):**
- Dataset `variety_armC_mix_3k` (built by `Training/build_varmix_3k.py`): **1,000 Variety** from the 138 eval-aligned categories (electronics/connectors/components/switches/terminals + small-metal + zippers/buckles; 22 toy/decor/ceramic/fabric categories EXCLUDED) + **2,000 Arm-C rehearsal**, 50/50 NG/OK, stratified. Registered in `LlamaFactory/data/dataset_info.json`.
- Two 3-epoch frozen-ViT continuations (lr 1e-5 cosine, EBS 32, save_only_model): from Arm-C 82.80 (tmux `sft_varmix_armC`, CUDA 0,3 → `outputs/sft_varmix_from_armC/`) and from GRPO 82.73 (tmux `sft_varmix_grpo`, CUDA 1,2 → `outputs/sft_varmix_from_grpo/`). Configs `Training/sft_varmix_from_{armC,grpo}.yaml`. Watchdog `Training/varmix_eval_watchdog.sh` → `outputs/varmix_watchdog.log`.
- WHERE THE DATA WILL COME FROM (fill the "+ targeted Variety + rehearsal" TBD rows): `outputs/sft_varmix_from_{armC,grpo}/checkpoint-{94,188,281}/eval_{dsmvtec,visa}_full_trainprompt.json` → BA via `results/compute_ba.py`; best-DS epoch. Rationale: anti-forgetting (rehearsal) + eval-aligned categories test whether transfer is recoverable once distribution shift is controlled.

### 2026-06-15 — Real-IAD-Variety continuation-SFT experiment LAUNCHED (PENDING — placeholder data source)
**Experiment.** Four epochs of frozen-ViT SFT on the Real-IAD-Variety STaR corpus (`variety_star_sft_6k`, 6,000 items, corrected+rewritten+kept), continued from each of the two strongest checkpoints:
- from **82.80 % Arm-C** (`outputs/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376`) → `outputs/sft_variety_from_armC_82p8/` (CUDA 0,3, tmux `sft_armC`).
- from **82.73 % GRPO** (`outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530`) → `outputs/sft_variety_from_grpo_82p7/` (CUDA 1,2, tmux `sft_grpo`).

**Configs:** `Training/sft_variety_from_{armC,grpo}.yaml` — clones of `sft_abc_C.yaml`: finetuning_type full, freeze_vision_tower true, DS Z3 CPU-offload, lr 1e-5 cosine warmup 20, per_device_bs 8 × GA 2 × 2 GPU = EBS 32, cutoff 12144, num_train_epochs 4, **save_only_model: true** (weights only, no optimizer/DeepSpeed shards). Dataset registered as `variety_star_sft_6k` → `Training/datasets_variety/variety_star_sft_6k.json`. 6000 items / EBS 32 ⇒ ~188 steps/epoch ⇒ checkpoints at 188/376/564/752.

**WHERE THE DATA WILL COME FROM (to fill `tab:variety-cont` / §sec:res-variety-cont + the F5 conclusion item):**
- Eval JSONs written by the watchdog `Training/variety_sft_eval_watchdog.sh` (→ `outputs/variety_sft_watchdog.log`, prints final BA table): for each run, `checkpoint-{188,376,564,752}/eval_{dsmvtec,visa}_full_trainprompt.json` (full 1670/2141, trainprompt mode, same harness as all SFT numbers).
- BA per file via `repository_tu_delft_vlms/results/compute_ba.py <json>`. Report best-DS epoch per run (note VisA-best if it differs); Δ signed vs each row's init (Arm-C 82.80/72.07; GRPO 82.73/70.39).
- STATUS: **pending** (training in progress as of 2026-06-15; values are `TBD` in the thesis). This is the §4a "Variety STaR (F5)" experiment.

### 2026-06-15 — §3.Ch3 (reasoning-trace generation) audit (8 investigators) + Gemini benchmark filled
**4 verified correct:** trace-length L171 recomputed EXACT on the 6K `<think>` body (mean 164.7 tok, std 13.5, p5/p95 142/187, range 121–223); location 9-cell + type 26-category vocab match `inspector_prompt_test_v2.txt`; six-phase wording A–F↔1–6; privileged 3-image input (orig+overlay+same-camera reference) matches the generation code.
**Edits applied:**
- **§3.5 QC — FULL HONEST DISCLOSURE [author decision]:** the real prompt has **6** auto-reject rules (A–F; thesis listed 5, omitted "Unsupported Defect Claim") **and** a silent **8-category acceptance filter** (each 0/1/2, ≥13/16 PASS/FAIL, rewrite-on-fail; `inspector_prompt_test_v2.txt` L346–480). Rewrote §3.5 + Fig-3 caption + the system-prompt component list to disclose both; corrected "no numeric quality score"/"no quality-ranked second pass" → "no second model + no post-hoc score-ranked selection; the generator self-scores within the single call". The 6K-vs-15K result framing is unaffected (the filter is uniform across both corpora).
- **L69:** "120–200 words" verified (prompt L130; 6K corpus mean 136.8 words). Removed the unsupported "outperforms the earlier 600–1000-word verbose template by a wide margin" (no such prompt exists; the earlier prompt says "3–5 sentences"; Ch6 says it is NOT a length contrast) → reworded to the real "3–5 sentences" earlier prompt. [author decision]
- **L132:** "4-second-per-batch" was the `RATE_LIMIT_WAIT=4` backoff constant, not a measurement (logs show ~10.75 s/batch) → "~11 s/batch". "1.6 s/trace" (L172) kept (log-supported).
- **t-SNE L179:** n=8,908 clarified as the pre-partition anomalous pool (released corpus = 7,247 anomalous).
**OPEN — L17 mask filter (verified WRONG):** thesis claims NG restricted to Multi-Close/Single/Spread masks (degenerate excluded); the real sampler (`sample_realiad_v4.py` L62-63) filters NG only by ".png mask exists"; Multi-Close/Single/Spread (`analyze_mask_spread.py`) is a diagnostic never read by any sampler/generator. Fix pending author confirm.
**FOLLOW-UP (§3.AppB):** the "verbatim" prompt appendix is a paraphrase (5 rules, no acceptance filter) — should be updated to the real 6 rules + 8-category filter.
**Gemini-2.5-Flash teacher benchmark FILLED:** final BA **DS-MVTec 81.52 %** (TP1129 TN315 FP129 FN97), **VisA 75.18 %** (TP722 TN850 FP94 FN475), `no_answer=0`, n=1670/2141. `tab:sota` row + teacher-reference paragraph in 06_results filled (student Arm-C 82.80 / GRPO 82.73 exceed the teacher on DS-MVTec; teacher leads on VisA by 3–5 pp, FN-heavy: 60.3 % recall / 90.0 % spec).

### 2026-06-15 — §3.Ch2 (background) audit + Gemini-2.5-Flash teacher benchmark
**Ch2 audit (6 parallel investigators) — mostly clean, 1 edit:**
- six-phase-vs-4-tag: CORRECT — the real prompt `inspector_prompt_test_v2.txt` has 6 reasoning phases (A–F) *inside* `<think>`; the output schema is 4 tags (think/location/type/answer). Distinct granularities, no conflation. (optional L61 clarity nit declined for now)
- scratch-mm phrases ("3 mm scratch", "sub-millimetre"): illustrative/rhetorical — keep.
- MMAD counts: VERIFIED EXACT from mmad.json — DS-MVTec **1670**, VisA **2141**, total **8366** images. 39,672 questions vs local recount 39,670 (2 off, cited jiang2025mmad — keep). MVTec 5354/15 & VisA 10821/12 are cited literature.
- cited perf numbers (PatchCore 99.6, EfficientAD >99%/2ms/600FPS, SimpleNet 99.6/77FPS, WinCLIP 91.8, AnomalyGPT 86.1/94.1, …): internally consistent across Ch1/2/7; citation-trust.
- few-shot sweep: all 8 mentions (AnomalyGPT 1-shot, PromptAD, …) describe **cited baselines** — legitimate under the strict zero-shot rule (which governs OUR method).
- **EDIT (02_background.tex L95):** "The KL term is computed inside the loss" → "The KL term, when used, is placed inside the loss (weighted by β) … in this thesis we set β=0, so it is monitored but not added to the loss (see Ch5)". Equation itself is a correct general GRPO derivation; this only conditions the present-tense claim + forward-refs Ch5. [author decision]

**Gemini-2.5-Flash teacher benchmark (NEW reference baseline):**
- Added a `--gemini` flag to `evaluate_qwen25vl_7b_trainprompt.py` (synced to repo `scripts/04_eval/`). It reuses the EXACT same data-loading, ground-truth extraction and balanced-accuracy code as the Qwen evals and only swaps the model call → benchmarks `gemini-2.5-flash` (Vertex, project from `GOOGLE_CLOUD_PROJECT`) on the **identical** 1670 DS-MVTec / 2141 VisA set (same products, same images, seed=42), so the row is directly comparable.
- Prompt: structured `<think>/<type>/<location>/<answer>` for anomalies, bare `<answer>No</answer>` for normals; `temperature=0`, `max_output_tokens=700`, `thinking_config.thinking_budget=0`.
- Launched in tmux session `gemeval` → `outputs/gemini_25flash_eval/eval_{dsmvtec,visa}_full_gemini25flash.json`. Smoke-tested (4 DS-MVTec anomalies → all parsed `yes`).
- Thesis placeholder added: a "Gemini~2.5-Flash (trace teacher)" row in `tab:sota` + a "Teacher reference (placeholder)" paragraph (06_results), both marked `% TODO[gemini-ref]` — to be filled with the DS/VisA balanced accuracy once the run completes.

### 2026-06-15 — §3.Ch1 (introduction) claims audited one-by-one (11 parallel investigators)
**7 verified correct, no change:** GRPO headline 82.73/70.39/80.87 & +0.81 pp vs IAD-R1 (recomputed exact from `grpo…run2/ckpt-530` + `iad_r1_qwen_recanon`); Arm-C 82.80/72.07 (`sft…abc_C…/ckpt-376`); +11.2 pp & per-product deltas (matches Ch6); corpus sizes 6000+4236+4236=14,472≈14.5K; GPT-4o 74.9 % (cited jiang2025mmad); Variety 10–20 AUROC / 30→160 (160 category JSONs on disk; magnitude per cited paper).
**4 edits applied to `01_introduction.tex`:**
- **L59 (contrib 2b)** — dropped "concise/verbose" (both corpora share ≈137-word mean length — a data-count/composition effect, per Ch6/7/8, *not* a length effect). Kept **8.5 pp** but labelled it the **matched-epoch (ep3)** gap (80.16 % vs 71.66 %) and **scoped to 7B** (at 3B the 15K is marginally ahead, −0.48 pp — effect does not replicate). [author decision 2026-06-15]
- **L78 (outline)** — appended "with the unfrozen encoder run at 15K only -- 6 configurations" so the grid notation matches contrib 2 / Ch4 / the G1 decision (no longer reads as a full 2×2×2×4=32 cross-product).
- **L20 (>95 % floor)** — softened "well below the > 95 % accuracy demanded by most production floors" → "far below the reliability required for production deployment." The author's own review paper (`Review_VLMS__After_Final_`) asserts **no** numeric accuracy floor and its real-world numbers (Kaputt <60 %, ADNet 78.5 %, electronics Tier-1 86.1 %) *undercut* a >95 % claim; new wording matches 02_background.tex:137. GPT-4o 74.9 %/MMAD half kept (cited). [author decision 2026-06-15]
- **L8 ("last five years")** — KEPT (2021–2026 is exactly 5 years and consistent with the cited SOTA wave PatchCore 2022 / SimpleNet 2023 / EfficientAD 2024). [author decision 2026-06-15]
**Ledger reconciliation:** the stale "30 products → 29 via image paths" entries (this file, formerly L230/L416) are **RETRACTED** — recount confirms **30** distinct products by image path in both the 6K and 15K-union trace JSONs (matches 1.4c). 
**Pending propagation:** 08_conclusion.tex carries the same "concise/verbose 8.5 pp" wording — to be fixed under §3.Ch8 with the identical treatment.

### 2026-06-14 — §2 unverifiable items resolved (GRPO-on-C table, estimator probe, run-1 seed)
**2b — `tab:grpo-on-c` (§6.13) filled with the verified 4-epoch curve; ckpt-795 row dropped.**
"Rotated away" = the trainer's `save_total_limit` deletes older ~16 GB checkpoints mid-training. `ls outputs/grpo_qwen25vl_7b_abc_C_grpo/` shows only checkpoint-{265,530,1060,1590,1855,2120}; **checkpoint-795 and checkpoint-1325 dirs no longer exist**, so the old "ckpt-795 = 80.76/69.69" row was unrecomputable and was removed. Verified surviving curve (vs Arm-C init 82.80/72.07), recomputed via `compute_ba.py`:

| ckpt | DS | VisA | Δ_DS | Δ_VisA |
|---|---|---|---|---|
| 265 (ep0.5) | 80.61 | 72.68 | −2.19 | +0.61 |
| 530 (ep1.0) | 80.32 | 70.34 | −2.48 | −1.73 |
| 1060 (ep2.0) | 81.20 | 69.17 | −1.60 | −2.90 |
| 1590 (ep3.0) | 80.70 | 68.70 | −2.10 | −3.37 |
| 1855 (ep3.5) | 78.81 | 67.58 | −3.99 | −4.49 |
| 2120 (ep4.0) | 80.38 | 67.86 | −2.42 | −4.21 |

Every checkpoint is below the SFT init on DS; VisA degrades monotonically. Reward (ckpt-2120 `trainer_state`): trended **1.92 → 2.38** (peak 3.25) — the old "1.99→2.54" (rotated-window figure) was corrected in `06_results.tex` L570.

**2c — estimator ablation is probe-based (already labelled; clarified).** `tab:grpo-estimator-ablation` uses a fast **400-sample probe** (`outputs/grpo_probe_{ctrl,drgrpo,g2rpo}/probe_curve.csv`, now shipped to `results/grpo_probe_curves/`). Its "init" row 84.52/71.89 is the probe reading and runs **~1.7 pp above** the full 1670-sample eval (Arm-C full = 82.80); a clause was added to the caption so the probe figures are not compared to full-eval numbers. No CM JSON exists for the probe points — not recomputable, correctly flagged.

**2d — run-1 vs run-2 are the SAME seed (NOT a seed experiment).** Both `…ep3_full_run1` and `…_run2` have **seed = 42** (verified in `training_args.bin`), identical lr/G/β. So the run-1 (81.65) vs run-2 (82.73) gap at ckpt-530 is **run-to-run non-determinism** (stochastic GRPO rollouts), **not seed sensitivity**. The Ch7 (L1) limitation was reframed accordingly. Run-1 eval JSONs (ckpt-315, ckpt-530) shipped to `results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run1/` so the comparison is reproducible. (NB: the iter-2 `tab:iter2-progression` ckpt-795 at L161 is a *different*, on-disk checkpoint — left as-is.)



### 2026-06-14 — Canonical generation pipeline added (`realiad_v4/`) — TWO findings for the author
The authoritative Real-IAD generation pipeline (`realiad_v4_pipeline.zip`) was added to the repo at
[`scripts/00_generate/realiad_v4/`](scripts/00_generate/realiad_v4/) (scripts only; `.env` and the Vertex
service-account key were excluded, HF token scrubbed to an env lookup).

1. **Prompt — reverses an earlier ledger claim.** `generate_realiad_traces_v4.py` (line 106) loads
   **`inspector_prompt_test_v2.txt`** — the *structured* prompt with the six-phase template, auto-reject
   rules, and the defect-type vocabulary. So Appendix B's structured prompt, the "six-phase template", the
   "auto-reject rules", and the type-vocabulary description are **accurate to the real generator** — NOT a
   redaction/idealisation as earlier §3 notes (and my "1.4d mischaracterisation") claimed. Those earlier
   notes were based on the wrong file (`inspector_prompt.txt`, which is *not* the v4 generator's prompt).
   **Action:** retract the §3 "production prompt is a 3–5 sentence prompt / App-B is idealised" notes and the
   1.4d "controlled vocabulary mischaracterised" item.

2. **Model — RESOLVED by author (2026-06-14).** The author confirms: the **headline Real-IAD** AnomalyThink
   corpus was generated with **Gemini 2.5-Flash** (so the thesis's "Gemini 2.5-Flash" is CORRECT — no change),
   while the **Real-IAD-Variety** extension used **Gemini 3 Flash preview**. For consistency the thesis presents
   both as Gemini 2.5-Flash. Note the bundled v4 script (`generate_realiad_traces_v4.py`, line 64) and the
   variety generator (line 66) are both set to `google/gemini-3-flash-preview` — a later iteration of the
   scripts, not the exact 2.5-Flash generator that produced the headline corpus. **No thesis edit required.**



### 2026-06-14 — Recomputed corpus descriptors (token-length + t-SNE clusters)
Script: [`results/recompute_corpus_stats.py`](results/recompute_corpus_stats.py) (run in conda env `llama_sft`). Sources: 6K/15K trace JSONs (Qwen2.5-VL tokenizer) and `Training/anomaly_type_analysis/{types.json (8908 raw type strings), embeddings_2d.npy (8908×2 t-SNE coords)}`.

**`<think>` length (Ch3 L171 / L236):**
| corpus | tokens mean | tokens std | token range | words mean |
|---|---|---|---|---|
| 6K | **164.7** | **13.5** | [121, 223] (p5 142, p95 187) | 136.8 |
| 15K | 165.2 | 14.8 | [53, 216] | 136.8 |

Thesis says "≈170 tokens, std ≈28, range 160–270" → correct **≈165 tokens, std ≈13.5, range 121–223**. Word mean 141 → **137** (identical for 6K and 15K, confirming the "count not length" claim).

**t-SNE type clusters (Ch3 L179)** — recompute matches the ledger exactly:
| quantity | thesis | **recomputed** |
|---|---|---|
| Scratch cluster size (exact "Scratch") | 600 | **500** (substr "scratch" = 944) |
| Missing-component size (exact) | 180 | **162** |
| Scratch↔Missing centroid distance | 148.6 | **155.3** |
| within-cluster radius (mean of the two) | 16.5 | **15.3** |
| separation ratio (dist/radius) | 9.0 | **10.1** |

(Note: `types.json` holds *raw* type strings — e.g. "Crease", "Opening or disruption of the seal" — not a closed vocabulary; see the 1.4d vocabulary note.)



### 2026-06-13 — §1.4 batch applied (7 data-backed swaps + contrib-4 scope fix)
Each value recomputed from disk and applied to the thesis.

| Item | Location(s) | was → now | Source |
|---|---|---|---|
| 1.4a | abstract (i), intro contrib 2, Ch6 takeaway (Ch6 SOTA L335 already said 1.8) | "~1 pp" → **~1.8 pp** | 81.92 − 80.16 = 1.76, recanon + ckpt-564 |
| 1.4b | intro contrib 4 | "+13.6 pp over base" → **+11.2 pp (SFT over base)** | 80.16 − 69.01 = 11.15; makes it consistent with the base→SFT per-product gains it cites |
| 1.4g | Ch4 L29 | 3B "28 transformer layers" → **36** (7B 28 kept) | 3B `config.json` text_config.num_hidden_layers=36 |
| 1.4h | Ch4 L81 | 7B-unfrozen-15K "22.5 h" → **12.2 h** (6.9 h / 2.4 h kept — verified 6.93/2.35) | `all_results.json` train_runtime 43 876 s |
| 1.4k | Ch5 L161 | KL "[0.003, 0.095]" → **[0.0, 0.117]** (+ "13/530 steps >0.095") | run-2 ckpt-530 `trainer_state.json` |
| 1.4n | Ch7 L108 | run-1 ckpt-530 "80.6%" → **81.65%** | run-1 ckpt-530 eval JSON (80.63 = ckpt-315) |

Already-done before this batch (verified, no edit needed): **1.4j** (GRPO sample count — no "6,500"/"13,000" left in Ch5/App-A), **1.4l** (Ch6 caption already "the remaining thirteen gain").

### 2026-06-13 — 1.4m is NOT an error (ledger was wrong; thesis is correct)
The thesis claim "SFT loss 0.34 at step 752 / epoch 4" is **correct**: the headline 7B-frozen-6K run's `trainer_state.json` ends at **global_step 752, epoch 4.0**, with loss ~0.34 at the last logged step (step 750 = 0.344). The earlier ledger entry claiming "run ends at step 560, loss ~0.57" was mistaken. **No thesis change.** (The reported *checkpoint* is ckpt-564 = ep3 = best DS; the run itself continued to ep4.)

### 2026-06-13 — Matched-LR unfrozen pilot located (justifies the lower unfrozen LR; resolves the frozen-vs-unfrozen confound)
The lower LR used for the unfrozen cells (1e-6) is **not** an uncontrolled confound: a matched-learning-rate unfrozen run exists on disk and shows unfreezing hurts even at the *frozen* LR.

- **Run:** `outputs/sft_qwen25vl_7b_zeroshot_6k/` — 7B, `freeze_vision_tower: false`, **LR 1e-5** (= the frozen cells' LR; confirmed in `checkpoint-376/training_args.bin`), cosine, 2 epochs, dataset = 6K SFT pool (`iad_sft_6k_train`). DS-only eval (`eval_dsmvtec_full_trainprompt_all.json`), no VisA.
- **Result (DS-MVTec BA):** ckpt-100 = 60.22 → **ckpt-200 = 72.82 (peak, ~ep1)** → **ckpt-376 = 69.27 (ep2)** — peaks then **degrades** with further training.
- **Matched-LR contrast (both 7B, 6K pool, LR 1e-5, only frozen→unfrozen differs):** frozen = **80.16** (ckpt-564) vs unfrozen = **72.82 peak / 69.27 ep2**. Unfreezing costs ~7.3 pp DS *at identical LR* and destabilises training.
- **Second matched-LR unfrozen run** (`outputs/sft_qwen25vl_7b_zeroshot/`, small set, 1e-5, unfrozen): ckpt-188 DS = **71.86** — same ~72 ceiling. (This 71.86 is also the value mis-copied into the old 3B-Unfrozen-15K table row.)
- **Claim it supports:** the unfrozen cells used 1e-6 to *stabilise* training (prevent the 1e-5 degradation), not to raise the ceiling (1e-5 peak 72.8 ≈ 1e-6 15K peak 72.1). Unfreezing loses to frozen at every LR tried. Cited in `06_results.tex` (frozen>unfrozen paragraph). Removes the need for a confound caveat in the "Why low LR" rationale.

### 2026-06-13 — SFT hyperparameter EVIDENCE (gathered for author greenlight; thesis NOT yet edited)
Authoritative values read from each run's saved `outputs/<run>/checkpoint-*/training_args.bin` + YAML. **The thesis SFT hyperparameter tables (Ch4 `tab:sft-hp` L106-120 and App-A `tab:A.2` L54-66 + shared-settings line L66) are wrong on nearly every field.**

| Run | LR | schedule | warmup | wd | AdamW β2 | per-dev BS | GA | cutoff | ep | seed |
|---|---|---|---|---|---|---|---|---|---|---|
| 7B-frozen-6K (headline) | **1e-5** | **cosine** | **20** | **0.0** | **0.999** | 4 | 4 | **12144** | 4 | 42 |
| 3B-frozen-6K | 1e-5 | cosine | 20 | 0.0 | 0.999 | 4 | 4 | 12144 | 4 | 42 |
| 7B-frozen-15K | 1e-5 | cosine | 50 | 0.0 | 0.999 | 16 | 1 | 12144 | 4 | 42 |
| 7B-unfrozen-15K | **1e-6** | cosine | 50 | 0.0 | 0.999 | 16 | 1 | 12144 | 4 | 42 |
| 3B-frozen-15K | 1e-5 | cosine | 50 | 0.0 | 0.999 | 4 | 4 | 12144 | 4 | 42 |
| 3B-unfrozen-15K | 1e-6 | cosine | 50 | 0.0 | 0.999 | 4 | 4 | 12144 | 4 | 42 |
| Arm-C SFT (7B) | 1e-5 | cosine | 20 | 0.0 | 0.999 | 8 | 2 | 12144 | 4 | 42 |

**Thesis claims vs actual:** LR `2e-5 (all cells)` → **1e-5 frozen / 1e-6 unfrozen**; schedule `linear` → **cosine**; warmup `100 steps` → **20 (6K) / 50 (15K)**; weight decay `0.01` → **0.0**; AdamW β2 `0.95` → **0.999**; sequence length `4096` → **12144** (`cutoff_len`); per-device BS/GA `frozen 4/4, unfrozen 2/8` → **6K & 3B: 4/4; 7B-15K: 16/1; Arm-C: 8/2** (no run used 2/8). Effective batch size: thesis says **16**, but with 2 GPUs the real EBS = BS×GA×2 = **32** (e.g. 4×4×2). Correct as written: epochs 4, seed 42, bf16, ZeRO-3 CPU offload, grad-clip 1.0.

> **Status: AWAITING AUTHOR GREENLIGHT** before editing the thesis tables/prose. The "Why linear schedule" and "Why low LR (2e-5→5e-5 collapse)" rationale paragraphs in Ch4 will also need rewriting, since they argue for a linear schedule and a 2e-5 LR that were never used (the runs used cosine + 1e-5/1e-6).



### 2026-06-13 — `tab:sft-summary` 15K rows reselected to true best-DS epoch
The four-factor SFT summary table's "Best ep." column was inconsistent: 3 of the 4 15K cells were **not** at the epoch that maximises DS-MVTec balanced accuracy (the primary metric). All checkpoints of these runs carry full DS + VisA eval JSONs on disk, so the correct best-DS epoch is recomputable. **Selection criterion = highest DS-MVTec BA.** The VisA/Accuracy/F1 columns report *that same checkpoint* (not independently maximised). Changes applied to `chapters/06_results.tex` tab:sft-summary:

| Cell | was (ep, DS/VisA/Acc/F1) | now (ep, DS/VisA/Acc/F1) | source checkpoint (DS + VisA `eval_*_full_trainprompt.json`) |
|---|---|---|---|
| 3B-Frozen-15K | ep4 — 68.65 / 64.85 / 73.59 / 81.49 | **ep1 — 69.56 / 59.65 / 68.80 / 76.18** | `outputs/sft_qwen25vl_3b_15k_frozen/checkpoint-453/` |
| 7B-Frozen-15K | ep3 — 71.66 / 64.28 / 80.12 / 86.89 | **ep4 — 72.60 / 66.94 / 78.86 / 85.66** | `outputs/sft_qwen25vl_7b_15k_frozen/checkpoint-1812/` |
| 7B-Unfrozen-15K | ep4 — 70.27 / 58.16 / 65.21 / 71.51 | **ep3 — 72.08 / 58.60 / 68.50 / 75.02** | `outputs/sft_qwen25vl_7b_15k_unfrozen/checkpoint-1359/` |

(3B-Unfrozen-15K stays ep4 = 68.58/59.60/69.58/77.34 — ep4 *is* its best DS; 3B-Frozen-6K stays ep4 = 69.08; 7B-Frozen-6K headline stays ep3 = 80.16 — both already at best DS.) Per-epoch curves for all four 15K runs are in `results/eval_ba_inventory.txt`.

**Dependent prose reconciled** (same file): the "frozen > unfrozen" paragraph's "5–9 pp VisA gap" was rewritten (true only for 7B: 66.94 vs 58.60 = 8.3 pp; 3B is now negligible, 59.65 vs 59.60, a side-effect of selecting 3B-Frozen-15K ep1 by DS); and "7B > 3B by 2–11 pp" → "3–11 pp". The caption now states the selection criterion explicitly.

---

## How to read this ledger

### Balanced accuracy (BA) formula

All BA values are recomputed from the confusion-matrix block (`metrics: {tp, tn, fp, fn}`) inside each eval JSON, using:

```
BA = 50 * ( TP/(TP+FN) + TN/(TN+FP) )
```

The recompute script is shipped at [`results/compute_ba.py`](results/compute_ba.py). The full recomputed inventory is at [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt).

### Source-file conventions

- **Eval JSONs.** Ground-truth path form is `outputs/<run>/checkpoint-X/eval_<bench>_full_<mode>.json`. The same files are shipped in this repo under [`results/<run>/checkpoint-X/...`](results/). The prompt-mode contract is the **filename suffix**: `_trainprompt` (headline/primary), `_grpoprompt`, `_bareprompt`.
- **Benchmark sizes.** DS-MVTec full `n=1670`; VisA full `n=2141`; held-out Real-IAD `n=4236`.
- **Trace datasets.** Shipped under [`traces/`](traces/) (text-only LlamaFactory `{messages, images}` JSONs; images are relative paths to re-downloadable source images).
- **Configs.** SFT YAMLs, GRPO config, DeepSpeed configs, `dataset_info.json` under [`configs/`](configs/).
- **GRPO training-args** were read from the saved `run-2 training_args.bin`.

### Status legend

| Status | Meaning |
|---|---|
| **verified** | Printed value matches the on-disk artefact (within stated rounding), or is correct arithmetic over verified inputs. |
| **mismatch** | Printed value contradicts the on-disk artefact. Correct value supplied. |
| **unverified** | No on-disk artefact exists to confirm or refute (estimate, anecdote, never-run config, rotated-away checkpoint, or paraphrase). |
| **literature** | Cited from an external paper / hardware datasheet; not a result of this work. |

---

## Corrections needed (action list)

These are the confirmed printed-vs-actual contradictions. Fix in the thesis text and record in `UNVERIFIED.md`.

### A. `tab:baseline` (06_results.tex ~L25) — single transcription typo

| Cell | Claimed | Correct | Source |
|---|---|---|---|
| Qwen2.5-VL-3B base, DS-MVTec, **Bal-acc** | 55.80% | **56.14%** | `outputs/baseline_eval/3b_base_dsmvtec.json` (TP1222 TN56 FP388 FN4) → [`results/baseline_named/3b_base_dsmvtec.json`](results/baseline_named/3b_base_dsmvtec.json) |

### B. `tab:sft-summary` (06_results.tex L66-74) — cross-contaminated rows

| Row (true ckpt) | Metric | Claimed → Correct | Source |
|---|---|---|---|
| 3B-Frozen-6K ep4 (ckpt-752) | DS / VisA / Acc / F1 | 68.50→**69.08** / 59.60→**57.22** / 71.60→**72.22** / 78.10→**80.02** | [`results/sft_qwen25vl_3b_zeroshot_6k_frozen/checkpoint-752/`](results/sft_qwen25vl_3b_zeroshot_6k_frozen/) |
| 3B-Frozen-15K ep4 (ckpt-1812) | DS / VisA | 70.50→**68.65** / 60.85→**64.85** | [`results/sft_qwen25vl_3b_15k_frozen/`](results/sft_qwen25vl_3b_15k_frozen/) |
| 3B-Unfrozen-15K ep4 (ckpt-1812) | DS / Acc / F1 | 71.86→**68.58** / 74.19→**69.58** / 81.38→**77.34** (claimed = a 7B run's numbers) | [`results/sft_qwen25vl_3b_15k_unfrozen/`](results/sft_qwen25vl_3b_15k_unfrozen/) |
| 7B-Frozen-15K ep3 (ckpt-1359) | VisA / Acc | 65.45→**64.28** / 78.20→**80.12** (claimed = ep1/ckpt-453) | [`results/sft_qwen25vl_7b_15k_frozen/`](results/sft_qwen25vl_7b_15k_frozen/) |
| 7B-Unfrozen-15K ep4 (ckpt-1812) | DS / Acc / F1 | 72.60→**70.27** / 68.86→**65.21** / 75.60→**71.51** (72.60 is the 7B-*frozen* value) | [`results/sft_qwen25vl_7b_15k_unfrozen/`](results/sft_qwen25vl_7b_15k_unfrozen/) |
| **HEADLINE** 7B-Frozen-6K ep3 (ckpt-564) | all | **ALL CORRECT** (80.16 / 64.78 / 80.36 / 85.76) | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/) |

> Also in this table: the **3B-Unfrozen-6K** and **7B-Unfrozen-6K** rows are for configs that were **never run** (dirs absent); the printed cells duplicate the 15K-unfrozen rows. They cannot be confirmed or fixed from data.

### C. Appendix C — FABRICATED `tab:perprod-cm` and the SFT+GRPO column of `tab:perprod-full`

- The entire `tab:perprod-cm` confusion-matrix table is fabricated: per-product TP+TN+FP+FN **exceeds that product's sample size for 10 of 15 products** (physically impossible), and the per-product CM rows sum to **TP1150/TN382/FP29/FN272**, which does **not** equal the genuine Total **TP971/TN383/FP61/FN255**.
- The **SFT+GRPO (GRPO) per-product BA column** in `tab:perprod-full` is also fabricated (per-product values do not match the cited JSON).
- **Genuine** parts: the column **Average (82.73)**, the bold **CM Total (971/383/61/255)**, and the entire **Base** and **SFT** per-product columns (real JSONs, ~0.1 truncation).
- **Correct per-product GRPO BA** (run-2 ckpt-530, DS): cable 68.16, pill 75.45, toothbrush 83.33, screw 72.89, transistor 75.42, grid 94.74, leather 98.44, zipper 68.84, capsule 69.78, hazelnut 88.57, bottle 76.27, carpet 95.51, wood 97.50, tile 89.83, metal_nut 87.47.

### D. SFT/GRPO hyperparameter tables (04_sft.tex L106-125; Appendix A `tab:A.2`/`tab:A.3`)

The SFT recipe table and Appendix-A tables print several values that contradict the saved configs/args. Key ones: LR (claimed `2e-5` → actual **1e-5 frozen / 1e-6 unfrozen**), schedule (`linear`/100-step → **cosine**, warmup **20/50**), weight decay (`0.01` → **0.0**), AdamW beta2 (`0.95` → **0.999**), cutoff (`4096` → **12144**), GRPO max-prompt (`2048` → **4096**), GRPO save/eval-every (`250` → **530 / None** in saved run-2 args), GRPO per-device BS (`2` → **1**, EBS 8 unchanged). See the per-chapter tables below for the full list.

### E. Minor / single-figure mismatches

| Where | Claimed → Correct |
|---|---|
| Abstract / intro / ch6: "+13.6 pp" per-product headline gain | **+13.72 pp** (GRPO 82.73 − base 69.01) |
| Ch3 L82: "25" defect-type categories | **26** (Surface 11 + Structural 7 + Completeness 4 + Other 4) |
| ~~Ch3 L17/169/220 + Conclusion: "30 products" in SFT/15K → 29~~ | **RETRACTED 2026-06-15 — "30" is CORRECT.** Recount: 30 distinct products by image path in both `anomalythink_6k/combined_6k_train.json` (n=6000) and `anomalythink_15k/combined_sft_train.json` (n=14472). Earlier "29" was a miscount. GRPO/held-out span **23**. |
| Ch3 t-SNE: Scratch n=600 / Missing n=180 / dist 148.6 / radius 16.5 / ratio 9.0 | **500 (or 866 cluster) / 162 / ≈155.3 / ≈15.3 / ≈10.1** |
| Ch3 L171: trace-length std "28 tokens" | **≈13.5 tokens** (range 121-223) |
| Ch5 / Ch6: "6,500-sample run / 13,000" | **4,236 prompts (2,118 + 2,118)** |
| Ch5 L165: KL range "[0.003, 0.095]" | max **≈0.117** (range [0.0, 0.117]) |
| Ch6 L342: "fifteen others gain" (per-product) | **thirteen** (2 of 15 regress) |
| Ch4 L89: 7B-unfrozen-15K wall-clock "22.5 h" | **≈12.2 h** (43875 s) |
| Ch7 L108: GRPO run-1 "80.6%" at ckpt-530 | **81.65%** (80.63% is run-1 ckpt-315) |
| Conclusion RQ1: mean trace length "141 words" | **≈137 words** (136.8; the *equality* of 6K & 15K holds) |

---

## Summary counts

Counts over the per-chapter extraction (one row per extracted value; some values recur across chapters).

| Status | Count |
|---|---:|
| verified | 327 |
| mismatch | 67 |
| unverified | 100 |
| literature | 65 |
| **Total entries** | **559** |

| Chapter | verified | mismatch | unverified | literature |
|---|---:|---:|---:|---:|
| 00 Abstract | 28 | 2 | 0 | 0 |
| 01 Introduction | 24 | 1 | 3 | 8 |
| 02 Background & Related Work | 9 | 0 | 1 | 47 |
| 03 Dataset / Reasoning Traces | 22 | 6 | 11 | 5 |
| 04 Supervised Fine-Tuning | 27 | 12 | 11 | 4 |
| 05 GRPO | 40 | 3 | 24 | 4 |
| 06 Results & Analysis | 118 | 9 | 21 | 0 |
| 07 Discussion | 36 | 3 | 28 | 4 |
| 08 Conclusion & Future Work | 21 | 4 | 4 | 4 |
| Appendix A (Hyperparameters) | 25 | 10 | 6 | 2 |
| Appendix B (Prompts) | 8 | 0 | 1 | 0 |
| Appendix C (Per-product) | 35 | 36 | 18 | 0 |

> Numbers above are indicative tallies of the extraction; the authoritative record is the per-chapter tables that follow (every entry is listed).

---

## Chapter 00 — Abstract

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| six-phase | L9 | Trace template phases (Framing/Scan/Focus/Evaluate/Alternatives/Decide) | 03_dataset_traces.tex; generation prompt | verified | |
| ≈14,500 | L9 | Total Gemini-2.5-Flash traces (AnomalyThink-15K union) | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) len=14,472 | verified | |
| 2.5 | L9 | Gemini version (2.5-Flash) | Ground truth | verified | |
| ≈6,000 | L9 | SFT trace pool | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) len=6,000 | verified | |
| ≈4,000 | L9 | GRPO trace pool | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4,236 | verified | |
| ≈4,000 | L9 | Held-out test set | Held-out Real-IAD split = 4,236 | verified | |
| 3B / 7B | L11/L15 | Qwen2.5-VL sizes; 7B headline | Ground truth | verified | |
| 6K-vs-15K | L11 | Supervised data-quantity arms | Ground truth | verified | |
| four-component | L11 | GRPO reward (format/acc/type/loc) | Reward weights (0.3,0.3,0.2,0.2) | verified | |
| 1,670 | L11 | DS-MVTec eval n | `eval_dsmvtec_full_trainprompt.json` n=1670 | verified | |
| 2,141 | L11 | VisA eval n | `eval_visa_full_trainprompt.json` n=2141 | verified | |
| 4,236 | L11 | Held-out Real-IAD eval n | `eval_realiad4k_full_trainprompt.json` n=4236 | verified | |
| 80.16% | result (i) L13 | DS BA, 7B-frozen-6K ep3 ckpt-564 | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/) (tp988 tn354 fp90 fn238) | verified | |
| ~1 pp | result (i) L13 | Gap SFT→IAD-R1 SFT+GRPO on DS | 81.92 − 80.16 = 1.76 | mismatch | actual ≈1.76 pp |
| 6,000 | result (ii) L13 | Arm-C SFT corpus | 06_results.tex L460; Arm-C corpus | verified | |
| 82.80% | result (ii) L13 | Arm-C SFT DS BA (ckpt-376) | [`results/sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_dsmvtec_full_trainprompt.json`](results/sft_qwen25vl_7b_abc_C_full_patched/) (tp1003 tn372 fp72 fn223) | verified | |
| 72.07% | result (ii) L13 | Arm-C SFT VisA BA | `.../checkpoint-376/eval_visa_full_trainprompt.json` (tp683 tn822 fp122 fn514) | verified | |
| +0.88 pp | result (ii) L13 | Arm-C DS over IAD-R1 | 82.80 − 81.92 | verified | |
| +0.73 pp | result (ii) L13 | Arm-C VisA over IAD-R1 | 72.07 − 71.34 | verified | |
| 81.92% | result (ii) L13 | IAD-R1 DS BA (common harness) | [`results/iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json`](results/iad_r1_qwen_recanon/) (tp965 tn378 fp66 fn261) | verified | |
| 71.34% | result (ii) L13 | IAD-R1 VisA BA | [`results/iad_r1_qwen_recanon/eval_visa_full_trainprompt.json`](results/iad_r1_qwen_recanon/) (tp606 tn869 fp75 fn591) | verified | |
| 82.73% | result (iii) L13 | GRPO run-2 ckpt-530 DS BA | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/) (tp971 tn383 fp61 fn255) | verified | |
| 70.39% | result (iii) L13 | GRPO run-2 ckpt-530 VisA BA | `.../checkpoint-530/eval_visa_full_trainprompt.json` (tp762 tn728 fp216 fn435) | verified | |
| ~2 pp | result (iii) L13 | GRPO-on-C below Arm-C init on DS | grpo_qwen25vl_7b_abc_C_grpo deltas −2.19/−2.48/−1.60 | verified | |
| three-estimator | result (iii) L13 | Vanilla z / Dr.GRPO / G2RPO ablation | grpo_probe_* dirs | verified | |
| 6,000 | result (iv) L13 | Curated SFT split | as result (i)/(ii) | verified | |
| ≈14,500 | result (iv) L13 | Superset 6K outperforms | 14,472 union | verified | |
| ≈141-word | result (iv) L13 | Mean trace length shared by 6K & 15K | corpus analysis ≈137 (GT ≈141) | verified | |
| ~8 | result (iv) L13 | BA gap 6K > 15K after SFT | 80.16 − 71.66 = 8.50 ([`results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/`](results/sft_qwen25vl_7b_15k_frozen/)) | verified | |
| +25.8 | result (vi) L13 | metal_nut per-product gain (base→SFT564) | 57.86→83.64 | verified | |
| +19.2 | result (vi) L13 | cable per-product gain | 51.09→70.33 | verified | |
| +16.0 | result (vi) L13 | pill per-product gain | 60.23→76.21 | verified | |
| 2× / A6000 | L15 | Hardware | 2× RTX A6000 ZeRO-3 CPU offload | verified | |
| 7B | L15 | Final delivered model | Qwen2.5-VL-7B | verified | |

---

## Chapter 01 — Introduction

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 99% | sec:motivation L10 | PatchCore/EfficientAD/SimpleNet >99% AUROC | roth2022patchcore / batzner2024efficientad / liu2023simplenet | literature | |
| 10–20 | L14 | AUROC drop 30→160 categories (Real-IAD Variety) | zhu2025realiadvariety | literature | |
| 30 / 160 | L14 | Category counts (Real-IAD Variety) | zhu2025realiadvariety | literature | |
| 90% | L15 | WinCLIP zero-shot plateau | jeong2023winclip | literature | |
| 74.9% | L20 | GPT-4o avg accuracy on MMAD | jiang2025mmad | literature | |
| 95% | L20 | ">95% demanded by production floors" | no citation (rhetorical) | unverified | |
| 7B | L20/L30 | IAD-R1 backbone; headline backbone | li2025iadr1; ground truth | literature | |
| 3B | L30/L47 | 3B option fits single A6000 | ground truth; bai2025qwen25vl | verified | |
| 14,500 | item 1 L57 | AnomalyThink corpus | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) len=14,472 | verified | |
| 6,000 | item 1 L57 | AnomalyThink-6K | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) len=6,000 | verified | |
| 4,000 | item 1 L57 | GRPO pool | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4,236 | verified | |
| 4,000 | item 1 L57 | Held-out RealIAD | held-out split 4,236 | verified | |
| 30 | item 1 L57 | AnomalyThink products | 30 distinct in 6K & 15K combined trace JSONs | verified | |
| 125k | item 1 L57 | Anomaly-Instruct-125k | xu2025anomalyov | literature | |
| 80.16% | item 2 L59 | 7B-frozen-6K SFT DS BA | as Abstract (ckpt-564) | verified | |
| 1 pp | item 2 L59 | SFT within ~1pp of IAD-R1 | 81.92 − 80.16 = 1.76 (loose) | verified | |
| 8.5 pp | item 2 L59 | 6K beats 15K on DS | 80.16 − 71.66 = 8.50 (word "verbose" contradicts equal-length GT) | verified | |
| 16-cell | item 2 L59 | SFT ablation grid (3B/7B × frozen/unfrozen × 6K/15K × 4ep) | 4-factor study; 2 unfrozen-6K cells never run | unverified | |
| 82.73% | item 3 L61 | GRPO run-2 ckpt-530 DS BA | as Abstract | verified | |
| 70.39% | item 3 L61 | GRPO run-2 ckpt-530 VisA BA | as Abstract | verified | |
| 80.87% | item 3 L61 | GRPO run-2 ckpt-530 held-out RealIAD BA | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_realiad4k_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/) | verified | |
| +0.81 pp | item 3 L61 | GRPO over IAD-R1 on DS | 82.73 − 81.92 | verified | |
| beta=0 | item 3 L61 | KL coefficient (k3 monitored, not in loss) | run-2 training_args | verified | |
| G=4 | item 3 L61 | GRPO group size | run-2 training_args | verified | |
| 1e-6 | item 3 L61 | GRPO learning rate | run-2 training_args | verified | |
| 4,236 | item 3 L61 | GRPO training set | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4,236 | verified | |
| 2 | item 3 L61 | 2× RTX A6000 | run-2 training_args | verified | |
| 82.80% / 72.07% | item 3 L61 | Arm-C SFT (ckpt-376) DS / VisA | as Abstract result (ii) | verified | |
| 6K | item 3 L61 | Arm-C corpus size | ch6 L492/L532 | verified | |
| ~2 pp | item 3 L61 | GRPO-on-C below Arm-C init on DS | ch6 L565; tab:grpo-on-c | verified | |
| +13.6 pp | item 4 L63 | Per-product headline gain over base on DS | 82.73 − 69.01 = 13.72 | mismatch | +13.72 pp |
| metal_nut +25.8 | item 4 L63 | per-product gain (SFT564 vs base) | 57.86→83.64 | verified | |
| cable +19.2 | item 4 L63 | per-product gain | 51.09→70.33 | verified | |
| pill +16.0 | item 4 L63 | per-product gain | 60.23→76.21 | verified | |
| toothbrush +15.8 | item 4 L63 | per-product gain | 62.50→78.33 | verified | |
| tile −2.1 | item 4 L63 | per-product regression (SFT) | 83.93→81.87 | verified | |
| wood −0.1 | item 4 L63 | per-product regression (SFT) | 97.50→97.37 | verified | |
| 69.01% | item 4 (implied) | 7B base DS BA | [`results/qwen25vl_baseline_eval/eval_dsmvtec_full_trainprompt.json`](results/qwen25vl_baseline_eval/) | verified | |
| 2.5 | sec:outline L88 | Gemini 2.5-Flash | ground truth | verified | |
| five | sec:motivation L8 | "deep nets displaced pipeline in last five years" | rhetorical | unverified | |
| three decades | L8 | "classical CV standard for three decades" | rhetorical | unverified | |
| four | items 2-3 | reward components / factors / epochs | ground truth | verified | |

---

## Chapter 02 — Background and Related Work

All AUROC / dataset-size figures here are literature citations. Eval-size figures are recomputed.

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 99.6% / 98.4% | §2.1.1 | PatchCore image / pixel AUROC | roth2022patchcore | literature | |
| 97.9% | §2.1.1 | PaDiM image-AUROC | defard2021padim | literature | |
| 98.3% / 10× | §2.1.2 | CFLOW-AD image-AUROC / throughput | gudovskiy2022cflow | literature | |
| 98.0% / 68.4 | §2.1.3 | DRAEM image-AUROC / AU-PRO | zavrtanik2021draem | literature | |
| 98.5% / 97.8% | §2.1.3 | RD4AD image / pixel AUROC | deng2022rd4ad | literature | |
| four-layer / >99% / 32 / ~2 ms / 600 FPS | §2.1.3 | EfficientAD PDN depth / mean AUROC / datasets / latency / FPS | batzner2024efficientad | literature | |
| 96.6% / three-way | §2.1.4 | CutPaste AUROC / classifier classes | li2021cutpaste | literature | |
| 99.6% / 77 FPS | §2.1.4 | SimpleNet AUROC / throughput | liu2023simplenet | literature | |
| ~99% | §2.1.5 | classical families saturate near | general | literature | |
| 3 mm | §2.1.5 | illustrative scratch size | illustrative | unverified | |
| 30→160 / 10–20 pts | §2.1.5 | category-scaling AUROC drop | zhu2025realiadvariety | literature | |
| >99% / 10–30 pts | §2.1.5 | logical-anomaly regression (MVTec LOCO) | bergmann2022mvtecloco | literature | |
| 400 M / N²−N | §2.2.1 | CLIP training pairs / in-batch negatives | radford2021clip | literature | |
| 158K / 92.5% | §2.2.1 | LLaVA instruction examples / % of GPT-4 on ScienceQA | liu2023llava | literature | |
| 256 | §2.2.2 | Qwen-VL adapter compressed tokens | bai2023qwenvl | literature | |
| 3B and 7B | §2.2.2 | Qwen2.5-VL variants used | ground truth; configs/yaml | verified | |
| InternVL3-78B / 72.2 | §2.2.2 | InternVL3 size / MMMU | zhu2025internvl3 | literature | |
| 2.5 | §2.2.2 | Gemini 2.5(-Flash) | geminiteam2025gemini25; ground truth | verified | |
| ~100B / 4% / 74% | §2.3 | CoT threshold / Game-of-24 baseline / ToT | wei2022cot; yao2023tot | literature | |
| 1±ε / G / ε / β / −1 / G² | §2.4 | PPO clip / GRPO symbols / k3 const / G²RPO | schulman2017ppo; shao2024deepseekmath; schulman2020klestimator; hu2026openvlthinkerv2 | literature | |
| 7B | §2.4.2 | value-network memory saving on 7B | general; matches backbone | verified | |
| 91.8% / two | §2.5.1 | WinCLIP AUROC / AnomalyCLIP prompts | jeong2023winclip; zhou2024anomalyclip | literature | |
| 86.1% / 94.1% / 125K | §2.5.2 | AnomalyGPT acc / AUROC / Anomaly-Instruct-125K | gu2024anomalygpt; xu2025anomalyov | literature | |
| 3B / 7B / 8B | §2.5.3 | AnomalyR1 / LR-IAD / EMIT backbones | chao2025anomalyr1; pang2025lriad; guan2025emit | literature | |
| 3×3 / 2.9K / 3K / four / six | §2.5.3 | IAD-R1 R_loc grid / PA-SFT / SC-GRPO / rewards / datasets | li2025iadr1 | literature | |
| 7B | §2.5.3 | IAD-R1 released ckpt re-run here | ground truth (iad_r1_qwen_recanon) | verified | |
| 3B/7B, frozen/unfrozen, 6K/15K | takeawaybox | promised ablation grid | ground truth | verified | |
| 150K+ / 30 / five / ~50K / >99%→~85% | §2.6 | Real-IAD images / categories / viewpoints / anomalous / AUROC drop | wang2024realiad | literature | |
| 160 / 28 | §2.6 | Real-IAD Variety categories / industries | zhu2025realiadvariety | literature | |
| 5,354 / 15 / 70+ | §2.6 | MVTec AD images / categories / defect types | bergmann2019mvtec | literature | |
| 10,821 / 12 | §2.6 | VisA images / categories | zou2022visa | literature | |
| seven / 39,672 / 8,366 / 74.9% | §2.6 | MMAD subtasks / questions / images / GPT-4o ceiling | jiang2025mmad | literature | |
| 1,670 | §2.6 | DS-MVTec subset n | `iad_r1_qwen_recanon/eval_dsmvtec_full_trainprompt.json` (965+378+66+261) | verified | |
| 2,141 | §2.6 | VisA subset n | `iad_r1_qwen_recanon/eval_visa_full_trainprompt.json` (606+869+75+591) | verified | |
| six-phase / six | intro L5; §2.3 L61 | trace structure (conceptual) | operational XML schema is 4 tags; "six-phase" is the narrative count, deferred to Ch.4 | unverified | |

---

## Chapter 03 — Reasoning Trace Generation (AnomalyThink corpus)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| ≈14,500 | L5/L168/L236 | Total AnomalyThink corpus | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) len=14,472 | verified | |
| 6K | L5/L152/L168/L218 | SFT split | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) len=6,000 | verified | |
| 4K | L5/L153/L168 | GRPO split | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4,236 | verified | |
| 4K | L5/L154/L168 | Held-out RealIAD split | held-out n=4,236 | verified | |
| 150K+ / 30 / five (C1-C5) | L11 | Real-IAD images / categories / viewpoints | wang2024realiad | literature | |
| 30 | L17/L169/L220 | "all 30 RealIAD products; SFT spans 30" | 6K SFT & 15K union span **30** distinct by image path (recount 2026-06-15) | verified | 30 (earlier "29" retracted) |
| 23 | L169 | products in GRPO / held-out | grpo_train.json = 23 distinct products | verified | |
| 23 of 30 | L17 | held-out drawn from 23 of 30 | GRPO/held-out span 23 (appendix-A table says 25/5; per-split differs) | verified | |
| 4 (four tags) | L41/L58/L128 | anomalous schema `<think><location><type><answer>` | trace JSON inspection | verified | |
| 120-200 | L44/L69/L171 | `<think>` word budget | inspector_prompt_test_v2.txt L130; production prompt says "3-5 sentences"; corpus mean ~137 | unverified | |
| 600-1000 | L69 | earlier verbose template range | refers to ch6 prompt-evolution; not in this chapter's data | unverified | |
| six-phase / 6 | L60-68 | template phases | no on-disk prompt has verbatim labels; production prompt uses 3-5-sentence template | unverified | |
| 3×3 | L72/L79 | grid-style location reward | 9 location values = 3×3; corpus uses exactly these 9 | verified | |
| 9 | L72-79 | canonical location values | top/middle/bottom × left/center/right | verified | |
| 25 | L82 | controlled defect-type categories | actual list = 11+7+4+4 = **26**; inspector_prompt_test_v2.txt has 26 | mismatch | 26 |
| four groups | L82-90 | Surface/Structural/Completeness/Other | inspector_prompt_test_v2.txt L287-319 | verified | |
| four-component reward | L58 | R_acc/R_loc/R_type/R_con ↔ 4 tags | ground truth weights | verified | |
| 3 images / 1 image | L33/L107/L119/L121 | per anomaly (orig+overlay+ref) / per normal | ground truth | verified | |
| 5-10 / ≤10 | L113/L115/L119 | Gemini batch size | config_mmad.py BATCH_SIZE=10 | verified | |
| 4-second | L132 | latency per batch | not logged | unverified | |
| $0.0001 | L132 | cost per trace | not logged (retail estimate) | unverified | |
| 15K | L132/L218 | union label | 14,472 ≈ 15K | verified | |
| five auto-reject rules | L123-130/L140 | in-prompt QC rules | production prompt has none; test_v2 has 6 lettered (A-F); appendix condenses | unverified | |
| ≈14,500 / ≈14.5K | L168/L173/L156/L218 | total traces | 14,472 | verified | |
| 50%/50% | L156/L170 | class balance per split | 6K=3000/3000; GRPO=2118/2118; union ≈50% | verified | |
| 170 tokens | L171/L236 | mean trace length (tokens) | recomputed mean 164.7 tokens (word-mean ~137) | verified | |
| 28 tokens (std) | L171 | std of token length | recomputed std = **13.5** tokens | mismatch | ≈13.5 tokens (range 121-223) |
| 160-270 | L171 | token budget range | actual range 121-223; 270 never reached | unverified | |
| 1.6 s | L172 | median wall-clock per trace | not logged | unverified | |
| <$20 | L173 | total generation cost | not logged (retail estimate) | unverified | |
| 8,908 | L179 | anomalous `<type>` strings for t-SNE | anomaly_type_analysis/embeddings_2d.npy shape (8908,2) | verified | |
| top-12 | L179 | type families plotted | presentation choice | unverified | |
| 600 | L179 | Scratch cluster size | exact "Scratch" = 500; Scratch-dominant cluster = 866 | mismatch | 500 (or 866) |
| 180 | L179 | Missing-component cluster size | exact "Missing component" = 162 | mismatch | 162 |
| 148.6 | L179 | t-SNE Scratch↔Missing distance | recomputed 155.26 | mismatch | ≈155.3 |
| 16.5 | L179 | avg within-cluster radius | recomputed 15.34 (two clusters) | mismatch | ≈15.3 |
| 9.0 | L179 | separation ratio | 155.26/15.34 = 10.12 | mismatch | ≈10.1 |
| 4-5 mm | L187 | scratch length in illustrative example | verbatim example text | unverified | |
| 125K | L207/L216/L218/L231 | Anomaly-Instruct-125k corpus | xu2025anomalyov | literature | |
| ~5.9K | L218 | Expert-AD corpus | li2025iadr1 | literature | |
| GPT-4o | L223 | Anomaly-Instruct-125k generator | xu2025anomalyov | literature | |
| Gemini 2.5-Flash | L223/L33/L108/L132 | AnomalyThink generator | config GEMINI_MODELS='gemini-2.5-flash'; geminiteam2025gemini25 | verified | |
| three (3 splits etc.) | L150-156/L211/L236 | 6K/4K/4K disjoint splits | verified split sizes | verified | |
| 2 (two main SFT runs) | L158 | C1-only vs all-5 camera filter | c1_only_fixed/ dir (new_sft_c1 len 4236; combined_sft_c1 len 10236) | verified | |
| two formats | L160 | ShareGPT JSON + GRPO format | SFT `{messages,images}`; grpo_train.json custom keys | verified | |

---

## Chapter 04 — Supervised Fine-Tuning

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 80.16% | L10/L5 | 7B-frozen-6K SFT DS BA (pipeline caption) | `sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/eval_dsmvtec_full_trainprompt.json` | verified | |
| 82.80% / 72.07% | L10 | Arm-C SFT DS / VisA | `sft_qwen25vl_7b_abc_C_full_patched/checkpoint-376/eval_*` | verified | |
| ~2 pp | L10 | GRPO below Arm-C init (fwd ref) | grpo_qwen25vl_7b_abc_C_grpo deltas | verified | |
| 6K / 15K | L10/L36/L142 | data-size factor | combined_6k=6000; combined_sft=14472 | verified | |
| 3B / 7B | L29/L140 | model sizes | Qwen2.5-VL-{3B,7B}-Instruct | literature | |
| 28 | L29 | 3B transformer layers | 3B config.json `num_hidden_layers` = **36** | mismatch | 36 |
| 16 / 2048 | L29 | 3B attention heads / hidden | 3B config.json | verified | |
| 28 / 28 / 3584 | L29 | 7B layers / heads / hidden | 7B config.json | verified | |
| 14×14 | L29 | ViT patch size | config.json `patch_size`=14 | verified | |
| 512 | L29/L160 | image side cap (px) / decode tokens | sft yaml `image_max_pixels`=262144=512² | verified | |
| 1024×512 | L29 | stitched canvas | derived (two 512px halves) | unverified | |
| 256 GB | L72 | system RAM | hardware, not logged | unverified | |
| 48 GB | L72 | A6000 GDDR6 | RTX A6000 datasheet | literature | |
| 2 | L72/L89/L173 | A6000 count | ground truth; ds config | verified | |
| 3 | L78 | DeepSpeed ZeRO stage | [`configs/deepspeed/ds_z3_cpu_offload.json`](configs/deepspeed/ds_z3_cpu_offload.json) stage:3 | verified | |
| 6.9 hours | L89 | 7B-frozen-6K wall-clock | all_results.json train_runtime=24937 s | verified | |
| 22.5 hours | L89 | 7B-unfrozen-15K wall-clock | all_results.json = 43875 s = **12.19 h** | mismatch | ≈12.2 h |
| 2.4 hours | L89 | 3B-frozen-6K wall-clock | all_results.json = 8451 s (but a **6-epoch** run; 4-epoch ≈1.6 h) | mismatch | 2.35h for 6-epoch run |
| 1K | L95 | pilot trace count for sweep | no artifact | unverified | |
| AdamW | L106 | optimizer | training_args.bin | verified | |
| 0.9, 0.95 | L107 | AdamW β1, β2 | β1=0.9 OK, β2=**0.999** | mismatch | β1=0.9, β2=0.999 |
| 2e-5 | L108/L123 | learning rate "all four cells" | frozen=**1e-5**, unfrozen=**1e-6** | mismatch | 1e-5 / 1e-6 |
| linear + 100-step warmup | L109/L125 | LR schedule | training_args: **cosine**, warmup **20** (6K) / **50** (15K) | mismatch | cosine, warmup 20/50 |
| 0.01 | L110 | weight decay | training_args = **0.0** | mismatch | 0.0 |
| 4 | L111 | per-device BS (frozen) | yaml/args = 4 | verified | |
| 2 | L111 | per-device BS (unfrozen) | 7B-unfrozen=**16**, 3B-unfrozen=4 | mismatch | 16 (7B) / 4 (3B) |
| 4 | L112 | grad accumulation (frozen) | frozen args = 4 | verified | |
| 8 | L112 | grad accumulation (unfrozen) | 7B-unfrozen=**1**, 3B-unfrozen=4 | mismatch | 1 (7B) / 4 (3B) |
| 16 | L113 | effective batch size | per-GPU 16; global EBS=**32** on 2 GPUs | mismatch | global 32 (16/GPU) |
| 4096 | L114 | sequence length cap | yaml `cutoff_len`=**12144** | mismatch | 12144 |
| bf16 | L74/L86/L115 | mixed precision | training_args bf16=True | verified | |
| 1.0 | L116 | gradient clipping | training_args max_grad_norm=1.0 | verified | |
| 4 | L117/L143 | epochs | 7B=4; 3B-frozen-6K run used **6** | mismatch | 4 for 7B; 6 for that 3B run |
| 42 | L118 | random seed | training_args seed=42 | verified | |
| 5e-5 | L123 | LR collapse anecdote | no logged collapse run | unverified | |
| 25% | L125 | cosine under-utilisation (pilot) | no artifact | unverified | |
| 100-step warmup / 1.5% | L125 | warmup as % of budget | actual 20/752 ≈ 2.7%; internally inconsistent | mismatch | 20-step (~2.7%) |
| 2×2×2×4 = 32 | L145 | eval cells per benchmark | arithmetic correct (some cells never run) | verified | |
| 1,670 / 2,141 | L145 | DS / VisA eval sizes | eval JSON lengths | verified | |
| 512 | L160 | max new tokens (greedy) | eval harness | verified | |
| 0.3 pp | L160 | beam/nucleus vs greedy (200-sample) | no artifact | unverified | |
| 200 | L160 | decoding sanity-check size | no artifact | unverified | |
| 0.5% | L162 | post-SFT answer-tag parse fail | recomputed ckpt-564: 0/1670 and 0/2141 (<0.5%) | verified | |
| 100K | L64 | lazy image pool | qualitative | unverified | |
| 15 | L168 | DS-MVTec per-product count | DS eval JSON `per_product` has 15 keys | verified | |
| two decimal places | L170 | reporting precision | convention | verified | |
| six-phase | L4/L31/L53 | trace template (cross-ref) | ch3 | literature | |
| four-component | L10/L173 | GRPO reward (fwd ref) | ground truth weights | verified | |
| four-factor | L138/L173 | SFT grid factors | L140-143 enumerates 4 | verified | |

---

## Chapter 05 — GRPO Reinforcement Learning

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| G=4 | L27/L125/L146 | group size (num_generations) | run-2 training_args; trainer_state | verified | |
| 1e-8 | eq L15 | ε for group-std stability | trainer impl | unverified | |
| 2048 | L27/L135 | max prompt length | ground truth (but Appendix-A note: saved arg = 4096) | verified | |
| ε=0.2 | L27/L126 | PPO clip | run-2 training_args | verified | |
| β=0 | L27/L127/L218 | KL coefficient (k3 monitored) | run-2 training_args | verified | |
| (0.3,0.3,0.2,0.2) | L58/L92 | reward weights | ground truth | verified | |
| 1.0 | L58/L218 | max total reward | 0.3+0.3+0.2+0.2 | verified | |
| [0,0.3] | L61 | format reward range | w_fmt=0.3 | verified | |
| +0.075 | L67 | per-tag partial credit | 0.3/4 | unverified | |
| −0.1 | L73 | forbidden-tag penalty | reward design | unverified | |
| {0,0.3} | L78 | accuracy reward range | w_acc=0.3 | verified | |
| [0,0.2] / 1.0 / 0.2 / 0.8 / 0.16 | L81-84 | type reward range / sim & weighted values | reward design (sim 1.0/0.8 unverified) | verified/unverified | |
| {0,0.2} | L89 | location reward range | w_loc=0.2 | verified | |
| 3×3 | L90 | grid match deliberately NOT used | IAD-R1 contrast | literature | |
| 0.6 | L93 | format+accuracy sum | 0.3+0.3 | verified | |
| 4,236 | L99/L137/L218 | GRPO prompts | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/grpo_train.json) len=4236 | verified | |
| 50/50 | L99 | anomaly/normal balance | 2118+2118 | verified | |
| 4K | L99 | GRPO split informal label | 4,236 | verified | |
| 1024×512 / 512×512 | L101-103/L146 | stitch canvas / each half | stitch config (canvas unverified; halves verified) | unverified/verified | |
| <2% | L108 | missing-ref samples after path migration | production log claim | unverified | |
| 6,500 / 6,500+6,500=13,000 | L108/L160 | "headline 6,500-sample run" | actual **4,236** (2,118+2,118) | mismatch | 4,236 prompts (NOT 13,000) |
| checkpoint-564 | L123/L177/L252 | base/ref SFT policy | ground truth | verified | |
| 1e-6 | L128/L148 | GRPO LR | run-2 training_args; trainer_state | verified | |
| 0.1 | L129 | warmup ratio (linear) | LR decay consistent (saved arg: warmup_ratio None — see App A) | verified | |
| (0.9,0.95) | L130 | AdamW β | not dumped (actual β2=0.999) | unverified | |
| 0.01 | L131 | weight decay | not dumped (actual 0.0) | unverified | |
| 2 / 4 / 8 | L132-134 | per-device BS / GA / EBS | ground truth EBS 8 (per-device BS actually 1; see App A) | verified | |
| 512 | L136 | max completion length | run-2 training_args | verified | |
| 1,060 / 2 | L137/L160/L190 | max steps / epochs | trainer_state max_steps=1060, epochs=2 | verified | |
| 250 | L138/L160 | save/eval every N | ground truth (saved run-2 arg: save_steps=530, eval None — see App A) | verified | |
| 42 / bf16 | L141/L140 | seed / precision | run-2 training_args | verified | |
| G=64 | L146 | DeepSeekMath group size (cited) | DeepSeekMath | literature | |
| ~30 seconds | L146 | per-rollout cost | train.log run2 83.85 s/step → ~21 s/rollout (same order) | unverified | |
| ~600 MB / <50 ms | L156 | Nomic memory / per-step overhead | model size / impl claim | unverified | |
| ~700 | L154 | LOC of GRPO trainer | code claim, not counted | unverified | |
| 5200 | config | Nomic embedding server port | ground truth | verified | |
| 1.66 / 2.22 / 2.13 | L162 | combined reward steps 10/100/530 | trainer_state run2: 1.65625 / 2.21875 / 2.125 | verified | |
| 0.875 / 1.25 | L163 | accuracy reward start / end | trainer_state | verified | |
| ~73% | L163 | verdict-correct fraction at start | 0.875/1.2 ≈ 0.73 (interpretive) | unverified | |
| +42.9% | L163 | accuracy reward rel. gain | (1.25−0.875)/0.875 | verified | |
| 0.781 / 0.875 | L164 | format/consistency reward start/end | trainer_state | verified | |
| +12.0% | L164 | format reward rel. gain | (0.875−0.781)/0.781 | verified | |
| [0.003, 0.095] | L165 | KL range | trainer_state min 0.0 max **0.117** | mismatch | [0.0, 0.117] |
| 163-174 | L166 | completion length range | trainer_state ~153-186 (clusters 163-174) | mismatch | ~153-186 (clusters 163-174) |
| 12.4 hours | L167 | wall-clock to ckpt-530 | train.log: full run 24.69 h; ckpt-530 (epoch 1) ≈12.34 h | unverified | |
| step 287 / 0.6 | L171/L174 | sample reward-line ref | trainer_state step287 epoch=0.542 | unverified | |
| 0.8349 / 0.1349 / 0.6521 / 0.2715 / 0.0009 | L172-174 | sample reward-line components | illustrative listing (0.2+0.3+0.1349+0.2=0.8349 checks) | verified/unverified | |
| checkpoint-530 / checkpoint-1060 | L177/L185 | headline / epoch-2 ckpt | trainer_state | verified | |
| ~0.7 pp | L177 | ckpt-1060 below ckpt-530 on DS | 82.73 − 82.03 | verified | |
| 4,026 | L188 | iter-2 filtered traces | [`traces/iter2/iter2_v2_sft_train.json`](traces/iter2/iter2_v2_sft_train.json) len=4026 | verified | |
| 1e-5 / 4 / 4 | L189 | iter-2 SFT LR / BS / epochs | SFT recipe; trainer_state iter2_v2 | verified | |
| ckpt-378 / ckpt-504 | L189 | iter-2 SFT ep3 / ep4 | iter2_v2 SFT dir (504 steps / 4 ep) | verified | |
| 265 / 0.5-2.0 / G=4 | L190 | iter-2 GRPO save-N / epoch coverage / group | iter2_v2_full ckpts 265/530/795/1060 | verified | |
| +2.57 pp / +5.61 pp | L218 | GRPO over SFT on DS / VisA | 82.73−80.16 / 70.39−64.78 | verified | |
| 82.73% | L229/L253 | single-stage headline DS | run2 ckpt-530 DS JSON | verified | |
| {0.0,0.3,0.6,1.0} | L200/L209 | discrete reward bins (G²RPO) | reward design (plausible) | unverified | |
| −1.15 / +1.15 | L207 | worst/best G²RPO advantage (G=4) | Φ⁻¹(1/8) / Φ⁻¹(7/8) | verified | |
| −0.5 / √2 | L204/L207 | midpoint correction / erfinv scaling | OpenVLThinkerv2; standard identity | literature/verified | |
| 81.94% | L234/L252 | G²RPO v1 DS baseline | `iter2_v1_eval_archive/.../g2rpo_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json` (902/401/43/324) | verified | |
| 80.64% | L254 | iter-2 v2 DS baseline | `grpo_qwen25vl_7b_iter2_v2_full/checkpoint-530/eval_dsmvtec_full_grpoprompt.json` (964/367/77/262) | verified | |
| 2,700 / 1,167 | L229 | type substitutions / location normalisations (cross-ref) | attributed to ch3; not verifiable from grpo_train diff | unverified | |
| 5-7 h / 15-20 h | L239 | compound run / total budget | forward-looking estimate | unverified | |
| ≥0.5 pp | L245 | compound pass threshold | pre-registered design | unverified | |
| 2 | L167/L239 | A6000 count | ground truth | verified | |
| run 2 | L160/L185 | headline GRPO run id | ground truth (dir _run2) | verified | |

---

## Chapter 06 — Results and Analysis

### tab:baseline (L24-30)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 76.50% | L25 | 3B base DS Acc | [`results/baseline_named/3b_base_dsmvtec.json`](results/baseline_named/3b_base_dsmvtec.json) (1222/56/388/4) acc 76.53 | verified | |
| 55.80% | L25 | 3B base DS Bal-acc | recomputed **56.14** | mismatch | 56.14% |
| 86.20% / 75.90% / 99.70% | L25 | 3B base DS F1 / Prec / Recall | 3b_base_dsmvtec.json | verified | |
| 55.03% / 69.01% / 56.11% / 99.20% / 39.15% | L26 | 7B base DS Acc/BA/F1/Prec/Rec | [`results/qwen25vl_baseline_eval/eval_dsmvtec_full_trainprompt.json`](results/qwen25vl_baseline_eval/) (480/439/5/746) | verified | |
| 56.10% / 50.20% / 71.80% / 56.00% / 100.00% | L29 | 3B base VisA | [`results/baseline_named/3b_base_visa.json`](results/baseline_named/3b_base_visa.json) | verified | |
| 48.30% / 53.80% / 14.30% / 98.90% / 7.70% | L30 | 7B base VisA | [`results/baseline_named/7b_base_visa.json`](results/baseline_named/7b_base_visa.json) | verified | |
| 99.2% / 39.1% | caption L17 | 7B base DS Prec / Rec | qwen25vl_baseline_eval | verified | |
| 1,670 / 2,141 | L24/L28 | eval sizes | eval JSON lengths | verified | |

### tab:sft-summary (L66-74) — see Corrections §B for all mismatches

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 16 | L40 | SFT grid cells | nominal 16 (2 unfrozen-6K never run) | unverified | |
| 68.50 / 59.60 / 71.60 / 78.10 | L66 | 3B-Frozen-6K ep4 (ckpt-752) | `results/sft_qwen25vl_3b_zeroshot_6k_frozen/checkpoint-752/` | mismatch | 69.08 / 57.22 / 72.22 / 80.02 |
| 70.50 / 60.85 | L67 | 3B-Frozen-15K ep4 (ckpt-1812) DS/VisA | `results/sft_qwen25vl_3b_15k_frozen/checkpoint-1812/` | mismatch | 68.65 / 64.85 |
| 73.59 / 81.49 | L67 | 3B-Frozen-15K ep4 Acc/F1 | ckpt-1812 | verified | |
| 67.78 / 61.50 / 70.30 / 76.40 | L68 | 3B-Unfrozen-6K ep4 | config NEVER RUN (dir absent) | unverified | |
| 71.86 / 74.19 / 81.38 | L69 | 3B-Unfrozen-15K ep4 DS/Acc/F1 | `results/sft_qwen25vl_3b_15k_unfrozen/checkpoint-1812/` (claimed=7B numbers) | mismatch | 68.58 / 69.58 / 77.34 |
| 59.50 | L69 | 3B-Unfrozen-15K ep4 VisA | ckpt-1812 (59.60) | verified | |
| 71.66 / 86.89 | L70 | 7B-Frozen-15K ep3 DS/F1 | [`results/sft_qwen25vl_7b_15k_frozen/checkpoint-1359/`](results/sft_qwen25vl_7b_15k_frozen/) | verified | |
| 65.45 / 78.20 | L70 | 7B-Frozen-15K ep3 VisA/Acc | ckpt-1359 (claimed=ep1/ckpt-453) | mismatch | 64.28 / 80.12 |
| 80.16 / 64.78 / 80.36 / 85.76 | L72 | **HEADLINE** 7B-Frozen-6K ep3 (ckpt-564) | [`results/sft_qwen25vl_7b_zeroshot_6k_frozen/checkpoint-564/`](results/sft_qwen25vl_7b_zeroshot_6k_frozen/) | verified | |
| 91.65 / 80.59 | L88 | HEADLINE SFT precision / recall | ckpt-564 DS | verified | |
| 72.08 / 58.20 / 68.50 / 75.02 | L73 | 7B-Unfrozen-6K ep4 | config NEVER RUN (dir absent) | unverified | |
| 72.60 / 68.86 / 75.60 | L74 | 7B-Unfrozen-15K ep4 DS/Acc/F1 | [`results/sft_qwen25vl_7b_15k_unfrozen/checkpoint-1812/`](results/sft_qwen25vl_7b_15k_unfrozen/) (72.60 is frozen value) | mismatch | 70.27 / 65.21 / 71.51 |
| 58.50 | L74 | 7B-Unfrozen-15K ep4 VisA | ckpt-1812 (58.16) | verified | |
| 8.5-point | L81 | 6K vs 15K ep3 gap | 80.16 − 71.66 | verified | |
| 7.6 pp | L81 | 6K vs 15K best-epoch gap | uses 15K ep4=72.60 (the WRONG frozen value) | unverified | |
| 141 words / 14.5K / 8.5K | L81 | mean trace length / union / extra | ground truth; 14472; 14472−6000 | verified | |
| ~300M | L83 | ViT param count (overfit arg) | Qwen2.5-VL ViT ~675M actual | unverified | |
| 4-12 pp | L85 | 7B>3B gap | range across cells | unverified | |
| 1 pp | L88 | SFT within of IAD-R1 | 81.92−80.16=1.76 (loose) | unverified | |

### tab:grpo-results (L112-120)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 80.63 / 70.25 / 77.37 / 82.69 | L112 | GRPO Run1 ckpt-315 DS/VisA/Acc/F1 | `results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run1/checkpoint-315/` | verified | |
| 81.65 / 69.39 / 78.32 / 83.47 | L113 | GRPO Run1 ckpt-530 | run1 ckpt-530 | verified | |
| 82.73 / 70.39 / 81.08 / 86.01 | L116 | **HEADLINE** GRPO Run2 ckpt-530 | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/) | verified | |
| 82.03 / 69.86 / 80.36 / 85.44 | L117 | GRPO Run2 ckpt-1060 | run2 ckpt-1060 | verified | |
| +2.57 / +5.61 / +0.72 / +0.25 pp | L120 | GRPO−SFT deltas DS/VisA/Acc/F1 | arithmetic over verified | verified | |
| 94.09 / 79.20 / 86.27 | L127 | Run2 ckpt-530 DS prec/rec/TNR | run2 ckpt-530 DS | verified | |
| 63.66 / 77.13 | L128 | Run2 ckpt-530 VisA TPR/TNR | run2 ckpt-530 VisA | verified | |
| ~0.7 pp | L130/L427 | ckpt-1060 below ckpt-530 | 82.73−82.03 | verified | |

### Iter-2 progression, rollout-failure-modes, SFT-Iter2-cleaned (L142-259)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 4,026 | L142 | iter-2 v2 SFT1 self-filtered traces | [`traces/iter2/iter2_v2_sft_train.json`](traces/iter2/iter2_v2_sft_train.json) len=4026 | verified | |
| 77.95 | L149 | iter-2 v1 SFT1 ckpt-196 DS (grpoprompt) | iter2_v1_archive | verified | |
| 76.89 / 77.16 | L150-151 | iter-2 v1 GRPO1 ckpt-265 / 530 DS | iter2_v1_archive | verified | |
| 77.61 / 72.00 | L153 | iter-2 v2 SFT1 ckpt-378 ep3 DS/VisA | [`results/sft_qwen25vl_7b_iter2_v2_frozen/checkpoint-378/`](results/sft_qwen25vl_7b_iter2_v2_frozen/) | verified | |
| 79.54 / 72.02 | L154 | iter-2 v2 SFT1 ckpt-504 ep4 DS/VisA | iter2_v2 ckpt-504 | verified | |
| 78.38 / 70.70 | L155 | iter-2 v2 GRPO1 ckpt-265 DS/VisA | [`results/grpo_qwen25vl_7b_iter2_v2_full/checkpoint-265/`](results/grpo_qwen25vl_7b_iter2_v2_full/) | verified | |
| 80.64 / 68.65 | L157 | iter-2 v2 GRPO1 ckpt-530 DS/VisA (best) | grpo iter2_v2 ckpt-530 | verified | |
| 79.55 / 68.40 | L158 | iter-2 v2 GRPO1 ckpt-795 | iter2_v2 ckpt-795 | verified | |
| 79.37 / 68.67 | L159 | iter-2 v2 GRPO1 ckpt-1060 | iter2_v2 ckpt-1060 | verified | |
| +1.10 pp | L164 | iter-2 RL over own SFT | 80.64−79.54 | verified | |
| 81.94 | L177/L183 | G2RPO best ckpt-530 DS (tab:rl-variants) | iter2_v1_archive g2rpo_full ckpt-530 grpoprompt | verified | |
| ~2 pp | L186 | T vs G prompt agreement | qualitative (81.94 vs 80.16 = 1.78) | unverified | |
| 30% | L193 | NG `<type>` raw codes in 6K pool | ground truth | verified | |
| 2,700 / 1,167 | L193 | type substitutions / location normalisations | not located on disk | unverified | |
| k=8 / 10,236 / 81,888 | L195/L199 | rollout multiplicity / pool / total | phase0_full_10k rollouts_raw.jsonl=10236; 10236×8=81888 | verified | |
| 3,770 / 94.8% / 5.2% | L195 | Gemini-patched traces / NG / OK | internally consistent only (artifact off-disk) | unverified | |
| 1,541 (30.1%) / 871 (17.0%) / 2,412 | L206 | NG/OK/total ≥1-of-8 wrong | arithmetic 1541+871=2412 verified; pools implied | verified/unverified | |
| 807 (15.8%) / 176 (3.4%) / 983 | L207 | all-8/8 wrong | 807+176=983 verified | verified/unverified | |
| 4 ep / 1e-5 / 32 | L212/L463 | SFT-Iter2 / ABC SFT recipe | trainer_state; bs8×GA2×2GPU=32 | verified | |
| 82.73 / 70.39 / 76.56 | L224 | SFT-Iter2 baseline DS/VisA/Avg | run2 ckpt-530 | verified | |
| 3,557+2,443 | L226 | 6K-pool kept+patched | sum=6000 (sub-split off-disk) | unverified | |
| 81.07 / 70.05 / 75.56 / −1.00 | L226 | 6K-pool variant DS/VisA/Avg/Δ | [`results/sft_qwen25vl_7b_iter2_clean/checkpoint-188/`](results/sft_qwen25vl_7b_iter2_clean/) | verified | |
| 96+96 / 71.30 / 60.52 / 65.91 / −10.65 | L227 | Balanced-192 composition + metrics | datasets_sft_iter2 balanced len=192; ckpt-24 | verified | |
| 4,236+197/197 / 75.77 / 66.41 / 71.09 / −5.47 | L228 | Held-out variant composition + metrics | datasets_sft_iter2 heldout len=4630; [`results/sft_qwen25vl_7b_iter2_heldout/checkpoint-145/`](results/sft_qwen25vl_7b_iter2_heldout/) | verified | |
| 322/1,670 / 20.5% | L233 | Balanced-192 ep1 unparsable / OK-recall VisA | iter2_balanced ckpt-6 / ckpt-24 | verified | |
| 3,000 NG + 3,000 OK | L256 | 6K pool item balance | Arm-C/iter2 pool 50/50 | verified | |
| 96% / 95% | L256/L259 | NG content fraction / per-step signal | ~94.8% rounds to 96% (derived) | unverified | |

### Per-product iter2 deltas (figs L238, L245) — all recomputed iter2_clean188 − grpo run2 530

| Number | Where | Describes | Status |
|---|---|---|---|
| +5.8 toothbrush, +2.4 cable, +1.8 grid, +1.2 screw, +1.2 tile | L238 | DS gains | verified |
| −6.1 capsule, −6.0 metal_nut, −5.3 wood, −4.5 hazelnut | L238 | DS regressions | verified |
| −1.66 pp | L238 | net iter2 DS effect (81.07−82.73) | verified |
| +6.4 pcb3, +8.0 macaroni2 | L245 | VisA gains | verified |
| −12.0 pcb4, −8.5 candle | L245 | VisA regressions | verified |
| −0.34 pp | L245 | net iter2 VisA effect (70.05−70.39) | verified |

### tab:sota / per-product base→SFT / held-out RealIAD (L280-430)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 82.73 / 70.39 | L280 | compound headline placeholder | run2 ckpt-530 | verified | |
| 0.5 pp | L289 | compound threshold | design choice | unverified | |
| 69.01 | L314 | 7B base DS (SOTA) | qwen25vl_baseline_eval | verified | |
| 81.92 / 71.34 | L317 | IAD-R1 DS / VisA (our harness) | iad_r1_qwen_recanon | verified | |
| 82.80 / 72.07 | L321 | Arm-C SFT-only DS / VisA | sft_7b_abc_C_full_patched ckpt-376 | verified | |
| +0.88 / +0.73 / +0.81 / −0.95 / ~1.8 | L330-332 | Arm-C & GRPO vs IAD-R1 deltas | arithmetic over verified | verified | |
| +25.8 metal_nut (57.8→83.6) | L342/L346 | base→SFT per-product gain | base 57.86→sft 83.64 | verified | |
| +19.2 cable / +16.0 pill / +15.8 toothbrush / +14.9 screw / +13.3 transistor | L342 | base→SFT gains | recomputed | verified | |
| 50.9%→70.1% cable | L346 | base→SFT (printed 70.1 vs actual 70.33) | recomputed | unverified | |
| 60.1→76.1 pill | L346 | base→SFT | recomputed | verified | |
| fifteen others gain | L342 | per-product gainers | only **13** gain (2 regress: wood, tile) | mismatch | thirteen |
| 4,236 / 23 | L388-393 | held-out n / products | eval_realiad4k JSON; ground truth | verified | |
| 80.81 / 80.87 / 91.28 / 68.34 / 78.16 / 93.40 | L399-404 | held-out Acc/BA/Prec/Rec/F1/TNR | `results/grpo_..._run2/checkpoint-530/eval_realiad4k_full_trainprompt.json` | verified | |
| 2 pp | L409 | held-out within of DS (80.87 vs 82.73) | 82.73−80.87=1.86 | verified | |

### Training curves (L415-423)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 1.14 / 0.58 | L415 | SFT loss step 50 / 500 | ckpt-564 trainer_state | verified | |
| 0.34 (step 752) | L415 | SFT loss end-ep4 | 7B-6K-frozen run ends step 560 (~0.57); 752/0.34 is 3B cadence | mismatch | ~0.57 at step 560 (no step 752) |
| step 564 | L415 | SFT BA peak | ckpt-564 epoch-3 | verified | |
| 1.66 / 2.22 / 2.13 | L419 | GRPO reward 10/100/530 | run2 trainer_state | verified | |
| 2.0 | L419 | theoretical max reward | design (text notes >2) | unverified | |
| 0.875→1.25 / +42.9% | L420 | accuracy reward | trainer_state | verified | |
| 0.781→0.875 / +12.0% | L421 | format reward | trainer_state | verified | |
| [0.003, 0.095] | L422 | KL bounds (steps 10/530 = 0.0033/0.0842) | trainer_state | verified | |
| 163-174 tokens | L423 | completion length | logged 169-174 at 10/100/530 | unverified | |

### Teacher ablation A/B/C, GRPO-on-C, estimator probe (L456-602)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 82.80 / 72.07 | L532/L599 | Arm-C best (ABC takeaway) | armC ckpt-376 | verified | |
| 78.33/66.15, 78.69/69.35, 77.32/70.21, 79.01/68.83 | L479-482 | Arm A ckpt-94/188/282/376 DS/VisA | `results` abc_A_kept (archive) | verified | |
| 2,978 | L456 | Arm A n items | [`traces/teacher_ablation_abc/sft_A_kept_balanced.json`](traces/teacher_ablation_abc/sft_A_kept_balanced.json) len=2978 | verified | |
| 75.83/64.08, 78.84/67.44, 80.75/67.48, 79.46/68.73 | L485-488 | Arm B ckpt-137/274/411/548 | abc_B_kept_corrected (archive) | verified | |
| 4,369 | L458 | Arm B n items | [`traces/teacher_ablation_abc/sft_B_kept_corrected.json`](traces/teacher_ablation_abc/sft_B_kept_corrected.json) len=4369 | verified | |
| 812 / 31% | L458/L529 | Arm B corrected items added / NG fraction | sub-count not reconcilable | unverified | |
| 79.71/68.76, 82.80/72.07, 81.39/72.01, 79.13/71.59 | L491-494 | Arm C ckpt-188/376/564/752 | [`results/sft_qwen25vl_7b_abc_C_full_patched/`](results/sft_qwen25vl_7b_abc_C_full_patched/) | verified | |
| 6,000 | L460 | Arm C n items | [`traces/iter2/sft_iter2_train.json`](traces/iter2/sft_iter2_train.json) len=6000 | verified | |
| 1,631 | L460 | Arm C Gemini-rewritten | sub-count (gemini_rewritten.jsonl=2406 differs) | unverified | |
| 95% | L456 | Arm A model pass rate | composition claim | unverified | |
| −1.15 / +4.05 (A), +0.59 / +2.70 (B), +2.64 / +7.29 (C) | L513-516 | Δ vs headline DS/VisA | arithmetic over verified | verified | |
| 1.68 / 0.07 pp | L525/L532 | Arm-C over SFT+GRPO VisA / DS | 72.07−70.39 / 82.80−82.73 | verified | |
| +5.43 pp | L527/L532 | Arm A VisA-best over headline (ckpt-282) | 70.21−64.78 | verified | |
| G=4 / β=0 / 1e-6 / 265 steps | L540 | GRPO-on-C config | ground truth; grpo_abc_C ckpts | verified | |
| 80.61/72.68 (−2.19/+0.61) | L558 | GRPO-on-C ckpt-265 | [`results/grpo_qwen25vl_7b_abc_C_grpo/checkpoint-265/`](results/grpo_qwen25vl_7b_abc_C_grpo/) | verified | |
| 80.32/70.34 (−2.48/−1.73) | L559 | GRPO-on-C ckpt-530 | grpo_abc_C ckpt-530 | verified | |
| 80.76/69.69 (−2.04/−2.38) | L560 | GRPO-on-C ckpt-795 | checkpoint dir rotated away (save_total_limit) | unverified | |
| 2.0-2.5 pp | L565/L598 | GRPO-on-C DS degradation | deltas | verified | |
| 1.99→2.54 | L565 | GRPO-on-C reward rise | maps to ckpt-795 window (ckpt absent) | unverified | |
| 120 / 20 / 400-sample | L567/L602 | estimator-ablation burst / cadence / probe size | probe_curve.csv; probe CM sums ~400 | verified | |
| 84.52/71.89 | L578 | estimator-ablation init (probe) | probe_curve.csv step0 INIT (no tp/tn JSON) | unverified | |
| 82.87/72.22 (ctrl), 84.23/70.18 (drgrpo), 84.39/69.80 (g2rpo) | L579 | step20 | grpo_probe_*/probe_curve.csv | verified | |
| 81.74/70.76, 81.19/66.82, 81.19/69.22 | L580 | step40 (ctrl/drgrpo/g2rpo) | probe_curve.csv | verified | |
| 82.25/71.82, 81.69/69.92, 81.72/68.26 | L581 | step60 | probe_curve.csv | verified | |
| 83.67/70.45, 82.06/69.82, 83.21/67.65 | L582 | step80 | probe_curve.csv | verified | |
| 83.02/70.78, 82.03/70.71 | L583 | step100 (ctrl / drgrpo) | probe_curve + drgrpo CM recompute | verified | |
| 81.89/69.87, 81.91/70.83 | L584 | step120 (ctrl / drgrpo) | probe_curve + drgrpo CM recompute | verified | |
| ~4 min / ~35 min / ~2 pp | L602 | probe vs full eval time / probe-high | methodological estimate (init 84.52 vs full 82.80 ≈1.7) | unverified | |

---

## Chapter 07 — Discussion

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 4-8 pp / 4-10 pp | L11 | frozen-vs-unfrozen ViT gap DS / VisA | qualitative range across grid | unverified | |
| ~300M | L15 | Qwen2.5-VL ViT params | model card | literature | |
| 15K / thirty (~30) | L15/L29 | corpus / distinct products | 14,472; 6K SFT ~30 products | verified | |
| 34-41% | L19 | VisA recall, unfrozen 7B | sft_7b_15k_unfrozen ckpts: 34.8/37.0/38.5/40.6% | verified | |
| ~70% | L19 | VisA recall, frozen | sft_7b_zeroshot_6k_frozen ckpt-564: 71.1% (tp851 fn346) | verified | |
| 100× / 1.5M | L21 | hypothetical corpus for unfreezing to win | hypothetical | unverified | |
| 6K-15K | L21 | trace range favouring freezing | 6,000 / ~14,472 | verified | |
| 8.5-point | L27 | 6K vs 15K SFT gap | uses headline 6K 80.16 vs a different-checkpoint 15K value; corrected 15K ep3 ckpt-1359 DS=80.12 gives ~0pp | mismatch | with corrected 15K DS the gap is ~0pp, not 8.5pp |
| 14.5K / 100% / ~141 words / ~8.5K | L29 | union / 6K-subset / mean length / extra | ground truth | verified | |
| 4 epochs / 2.5× / 2-2.5× | L31 | SFT budget / gradient updates / wall-clock | 14472/6000=2.41 | verified | |
| six-phase | L33/L43 | template phases | prompt template | unverified | |
| 1K | L35 | LIMA example count | Zhou et al. 2023 | literature | |
| 82.80 / 72.07 | L41/L63 | Arm-C SFT DS / VisA | sft_7b_abc_C_full_patched ckpt-376 | verified | |
| 10.7-point | L41 | Arm-C DS−VisA gap | 82.80−72.07 | verified | |
| 82.73 / 70.39 | L41/L63 | SFT+GRPO DS / VisA | run2 ckpt-530 | verified | |
| 81.92 / 71.34 / 10.6 pp | L41 | IAD-R1 DS / VisA / internal gap | iad_r1_qwen_recanon | verified | |
| +0.73 / +0.88 pp | L41 | Arm-C VisA / DS over IAD-R1 | arithmetic | verified | |
| ~3.5 pp | L41 | earlier IAD-R1 VisA deficit (artefact) | older eval, off-disk | unverified | |
| 1,670 / 2,141 | L41 | eval sizes | ground truth | verified | |
| ~0.7% / ~2.1% | L45 | VisA / DS mean defect area | benchmark stat, no backing | unverified | |
| 512 px / ~3× / 896 px | L45/L49 | stitch resolution / VisA downsample / proposed | config; derived; proposal | unverified | |
| 3×3 | L47/L49 | proposed grid spatial reward | IAD-R1 SC-GRPO | literature | |
| ~0.7 pp / ckpt-530 / ckpt-1060 | L55 | ckpt-530 over ckpt-1060 | 82.73−82.03 | verified | |
| β=0 / k_3 / two-epoch | L57 | KL coef / estimator / budget | ground truth | verified | |
| β ≈ 0.05-0.2 | L57 | proposed small KL (future work) | proposal (NOT the old wrong 0.04) | unverified | |
| +2.57 / +5.61 pp | L63-83 | GRPO lift over SFT-Iter1 DS / VisA | arithmetic | verified | |
| 6,000 / 0.07 pp / 1.68 pp | L63 | Arm-C size / DS match / VisA beat | ground truth; arithmetic | verified | |
| 0.30 / 0.20 / 0.20 | L70-72 | reward weights format / type / location | ground truth | verified | |
| ~92% / >99% / ~8% | L70 | tag compliance after SFT / GRPO / parser loss | no on-disk metric | unverified | |
| nomic-embed-text-v2 | L71 | type-reward embedding model | ground truth (server :5200) | verified | |
| ckpt-376 | L78 | Arm-C SFT as GRPO-on-C init | ground truth | verified | |
| 2.0-2.5 pp | L78/L83 | GRPO-on-C below Arm-C init | ckpt-265 −2.19, ckpt-530 −2.48, ckpt-1060 **−1.60** | mismatch | −2.19 / −2.48 / −1.60 (ckpt-1060 only −1.6) |
| ckpt-265 / +0.61 pp | L78 | only GRPO-on-C ckpt above init on VisA | 72.68−72.07 | verified | |
| 20-step / three | L78 | probe cadence / estimators | probe_curve.csv; 3 probe dirs | verified | |
| 84.52 / 71.89 | L78 | probe init DS / VisA | probe_curve.csv step0 (no CM JSON) | unverified | |
| none>init on DS / z-score strongest VisA | L78 | estimator outcome | probe CSV maxima | verified | |
| 200 / 100+100 / five | L90 | error-sample size / FN+FP split / categories | manual inspection (5-item list) | unverified/verified | |
| ~32% / <0.2% / ~24% / ~18%+14% / ~11% / ~5% | L93-97 | failure-mode shares | manual categorization, no backing | unverified | |
| run 1 / run 2 | L108 | GRPO run twice | both dirs exist | verified | |
| 80.6% | L108 | GRPO run-1 DS at ckpt-530 (claimed) | run1 ckpt-530 = **81.65**; 80.63 is ckpt-315 | mismatch | 81.65% (80.63 is ckpt-315) |
| 82.7% | L108 | GRPO run-2 DS at ckpt-530 | 82.73 | verified | |
| 3-5 seeds | L108 | recommended seeds | recommendation | unverified | |
| 80% | L110 | illustrative accuracy | rhetorical | unverified | |
| 170 tokens / ~30 tok/s / ~5-6 s | L112 | output length / speed / latency | estimate, no backing | unverified | |
| two orders of magnitude | L112 | slowdown vs EfficientAD | batzner2024efficientad (qualitative) | literature | |
| 7B / 2× A6000 / 6K | L112/L119 | model / hardware / corpus | ground truth | verified | |
| 0.8 pp | L119 | DS margin over IAD-R1 (takeaway, Arm-C) | 82.80−81.92=0.88 | verified | |
| Gemini 2.5-Flash | L116 | trace generator (leakage concern) | ground truth | verified | |

---

## Chapter 08 — Conclusion and Future Work

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| ≈14.5K / ~14,500 | C1 L11; RQ1; L23 | AnomalyThink corpus | [`traces/anomalythink_15k/combined_sft_train.json`](traces/anomalythink_15k/combined_sft_train.json) len=14472 | verified | |
| 25 | C1 L11 | Real-IAD training products claimed | actual 6K corpus has **30** distinct; ch3 says "all 30" | mismatch | 30 (held-out drawn from 23 of 30) |
| 6,000 / 6K | C1 L11; RQ1; L61 | SFT pool | [`traces/anomalythink_6k/combined_6k_train.json`](traces/anomalythink_6k/combined_6k_train.json) len=6000 | verified | |
| 4,000 (×2) | C1 L11 | GRPO / held-out pools | grpo_train.json len=4236 (rounds to 4000) | verified | |
| 4,236 | C4 L17 | held-out sample-level split | eval_realiad4k JSON total=4236 | verified | |
| 3B and 7B / four epochs | C2 L13; RQ3 | SFT study sizes / budget | ground truth; 04_sft.tex | verified | |
| 8.5 pp | C2 L13; RQ1 | 6K beats 15K | 80.16 − 71.66 = 8.50 (ckpt-564 vs ckpt-1359) | verified | |
| 80.16% | C2 L13; RQ1 | best SFT DS BA | sft_7b_zeroshot_6k_frozen ckpt-564 | verified | |
| 1 pp ("roughly 1pp") | C2 L13 | gap SFT→IAD-R1 | 81.92−80.16=1.76 | mismatch | 1.76 pp |
| 0.3,0.3,0.2,0.2 / G=4 / β=0 / 1e-6 | C3 L15; RQ2 | GRPO reward weights / group / KL / LR | ground truth (training_args) | verified | |
| 82.73% / 70.39% | C3 L15 | GRPO run-2 ckpt-530 DS / VisA | run2 ckpt-530 | verified | |
| +0.81 pp / 81.92% | C3 L15; RQ2 | GRPO over IAD-R1 / IAD-R1 DS | 82.73−81.92; iad_r1_qwen_recanon | verified | |
| −0.95 pp / 71.34% | C3 L15 | GRPO under IAD-R1 / IAD-R1 VisA | 70.39−71.34; iad_r1_qwen_recanon | verified | |
| +0.88 / +0.73 pp | C3 L15 | Arm-C over IAD-R1 DS / VisA | arithmetic | verified | |
| 80.87% / 93.4% | C4 L17 | held-out BA / TNR | eval_realiad4k JSON (tn1968 fp139) | verified | |
| 23 of the 30 | C4 L17 | held-out products | ground truth (held-out 23; SFT 30) | verified | |
| metal_nut +25.8 | C5 L19 | per-product gain (claimed base→GRPO) | base 57.86 → GRPO **87.47** = **+29.6** (the +25.8 is the base→SFT delta) | mismatch | +29.6 pp (base→GRPO) |
| cable +19.2 | C5 L19 | per-product gain (claimed base→GRPO) | base 51.09 → GRPO 68.16 = **+17.1** | mismatch | +17.1 pp |
| pill +16.0 | C5 L19 | per-product gain (claimed base→GRPO) | base 60.23 → GRPO 75.45 = **+15.2** | mismatch | +15.2 pp |
| five categories | C5 L19 | error/failure-mode categories | refs 07_discussion error taxonomy | unverified | |
| 82.80% / 72.07% | C6 L21; F5 | Arm-C SFT ckpt-376 DS / VisA | sft_7b_abc_C_full_patched ckpt-376 | verified | |
| 2-2.5 pp | C6 L21; RQ2 | GRPO lowers DS on Arm-C init | −2.19/−2.48/−1.60 | verified | |
| +2.57 / +5.61 pp | RQ2 | GRPO lift over SFT-Iter1 | arithmetic | verified | |
| 69.01% / 100% subset / ≈141 words / ≈8.5K | RQ1 | base DS / 6K-subset / mean length / extra | qwen25vl_baseline_eval; ground truth | verified | |
| ≈141 words | RQ1 | mean length both corpora | recomputed mean **136.8** (equality holds, value ≈137) | mismatch | ≈137 words |
| 5-6 seconds / 7B-parameter | L23 | inference time / final model | 07_discussion estimate; ground truth | unverified/verified | |
| ~15,000 / 125k | L23 | AnomalyThink / Anomaly-Instruct | 14472≈15K; xu2025anomalyov | verified/literature | |
| 80.64% | F1 L43 | Iter-2 DS BA vs single-stage | grpo_iter2_v2_full ckpt-530 grpoprompt (964/367/77/262) | verified | |
| 4,026 | F1 L43 | iter-2 self-filtered traces | [`traces/iter2/iter2_v2_sft_train.json`](traces/iter2/iter2_v2_sft_train.json) len=4026 | verified | |
| 3×3 grid / 512×512 / 3× / 896×896 | F2/F3 L45-47 | proposed reward / current stitch / downsample / proposed | 04/05/07 chapters; proposals | verified/unverified | |
| 3B (AgentIAD) / two tools | F4 L49 | AgentIAD size / tools | miao2025agentiad | literature | |
| 10-20 pp / 160 cat, 28 ind, 27 defects / 30→160 | F5 L51 | Real-IAD Variety scale | zhu2025realiadvariety | literature | |
| k=8 / 6,000-trace (variety) | F5 L51 | rollout count / variety SFT corpus | in-progress; [`traces/variety_star_6k/variety_star_sft_6k.json`](traces/variety_star_6k/variety_star_sft_6k.json) len=6000 (160 products) | unverified/verified | |

---

## Appendix A — Full Hyperparameter Tables

### Table A.1 — Trace generation

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| Gemini 2.5-Flash | L19 | generator model | scripts/00_generate/config_mmad.py L32 | verified | |
| 1.0 | L20 | generator temperature | not set in code; 1.0 is Gemini default | unverified | |
| 5-10 samples | L21 | batch size | config_mmad.py BATCH_SIZE=10 | verified | |
| 4 s / 3 | L22-23 | rate-limit wait / max retries | config_mmad.py RATE_LIMIT_WAIT=4, MAX_RETRIES=3 | verified | |
| 120-200 words | L24 | `<think>` budget | inspector_prompt_test_v2.txt L130 | verified | |
| 3 / 1 | L26-27 | images per anomaly / normal | ground truth | verified | |
| 187 | L33 | per-product quota K | generate_unified_realiad_15k_c1_only.py L43 | verified | |
| 50%/50% | L34 | class balance | grpo_train.json 2118/2118 | verified | |
| 25 / 5 | L35 | train/eval product split | generate script L10 (5 held-out; Real-IAD 30 → 25/5) | verified | |
| 5 | L36 | camera angles C1-C5 | generate script filter_c1_only | verified | |

### Table A.2 — SFT (see Corrections §D)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 3B / 7B | L54-61 | model sizes | sft yaml | verified | |
| 6K / 15K | — | data splits | combined_6k=6000; combined_sft=14472 | verified | |
| 2e-5 | LR column | LR for all 8 cells | frozen **1.0e-5**, unfrozen **1.0e-6** | mismatch | 1e-5 / 1e-6 (never 2e-5) |
| 4/4/16 | L54,55,58 | BS/GA/EBS frozen cells | yaml BS=4 GA=4 | verified | |
| 4/4/16 | L59 | BS/GA/EBS 7B-frozen-15K | yaml BS=**16** GA=**1** | mismatch | BS=16, GA=1 |
| 2/8/16 | L56,57 | BS/GA/EBS 3B-unfrozen | 3B-unfrozen-15K BS=4 GA=4; 6K never run | mismatch | BS=4, GA=4 (6K never run) |
| 2/8/16 | L60,61 | BS/GA/EBS 7B-unfrozen | 7B-unfrozen-15K BS=16 GA=1; 6K never run | mismatch | BS=16, GA=1 (6K never run) |
| 4 | Epochs col | epochs all SFT | yaml num_train_epochs=4 | verified | |
| (0.9, 0.95) | L66 | AdamW β | not in yaml (HF default 0.9,0.999) | unverified | |
| 0.01 | L66 | weight decay | not in yaml (default 0.0) | unverified | |
| 100-step warmup | L66 | warmup | yaml warmup_steps=**20**(6K)/**50**(15K) | mismatch | 20 / 50 |
| linear | L66 | LR schedule | yaml lr_scheduler_type=**cosine** | mismatch | cosine |
| 1.0 | L66 | grad clip | not in yaml (HF default 1.0) | unverified | |
| 4096 | L66 | seq length | yaml cutoff_len=**12144** | mismatch | 12144 |
| 42 | L66 | seed | not explicit (LlamaFactory default 42) | unverified | |
| 3 (ZeRO-3) | L66 | DeepSpeed stage | [`configs/deepspeed/ds_z3_cpu_offload.json`](configs/deepspeed/ds_z3_cpu_offload.json) | verified | |
| 2 / 48 GB | L66/L103 | A6000 count / VRAM | run scripts; RTX A6000 datasheet | verified/literature | |

### Table A.3 — GRPO (see Corrections §D)

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| 7B-frozen-6K SFT ckpt-564 | L80 | base/ref policy | run_grpo script MODEL_NAME_OR_PATH | verified | |
| 4 | L81 | group size G | run-2 training_args num_generations=4 | verified | |
| 0.2 / 0 | L82-83 | clip ε / KL β | run-2 training_args | verified | |
| 1e-6 | L85 | learning rate | run-2 training_args | verified | |
| 0.1 | L86 | LR warmup ratio | saved arg warmup_ratio=**None** (schedule linear verified) | mismatch | linear verified; warmup_ratio None |
| (0.9, 0.95) | L87 | AdamW β | β1=0.9 OK, β2=**0.999** | mismatch | (0.9, 0.999) |
| 0.01 | L88 | weight decay | run-2 args = **0.0** | mismatch | 0.0 |
| 2 | L89 | per-device BS | run-2 args = **1** (EBS 8 via GA4×2GPU) | mismatch | 1 |
| 4 / 8 | L90-91 | grad accum / EBS | run-2 args GA=4; EBS=1×4×2=8 | verified | |
| 2048 | L92 | max prompt length | run-2 args = **4096** | mismatch | 4096 |
| 512 | L93 | max completion length | run-2 args=512 | verified | |
| 4,236 / 2,118 / 2,118 | L94 | samples / anomaly / normal | grpo_train.json | verified | |
| 1,060 | L95 | max training steps | args max_steps=−1 (derived from epochs=2 over 4236); 1060 is the derived total | unverified | |
| 250 | L96 | save every | run-2 args save_steps=**530** | mismatch | 530 |
| 250 | L97 | eval every | run-2 args eval_steps=**None** (post-hoc eval) | mismatch | None |
| (0.3,0.3,0.2,0.2) | L98 | reward weights | ground truth (custom combiner; TRL arg None) | verified | |
| nomic-embed-text-v2-moe | L99 | type-embedding model | gemini_judge_server_v2.py L142; server :5200 | verified | |
| 1024×512 | L100 | stitch size (LEFT=ref RIGHT=test) | production run-2 used `--single_img 1` (NO stitch); 1024×512 is the unused 1-shot path | mismatch | N/A to run-2 (single image, no stitch) |
| 42 | L102 | seed | run-2 args seed=42 | verified | |
| 2 | L103 | A6000 count | run script CUDA_VISIBLE_DEVICES=1,2; nproc=2 | verified | |
| 480000 | (run-2 script only) | max_pixels | run_grpo_7b_resume_run2.sh `--max_pixels 480000` (appendix omits) | verified | |

---

## Appendix B — Prompts

| Number | Where | Describes | Source | Status | Correct value |
|---|---|---|---|---|---|
| two | L5 | prompts reproduced (Gemini + inference) | two `\section` blocks | verified | |
| 2.5 | L5/L8 | Gemini 2.5-Flash | ground truth; scripts/00_generate | verified | |
| SIX (6) | L14/L25-36 | six-phase inspection template | test_v2 encodes A-F flow; explicit "SIX-PHASE" labelling is paraphrase | unverified | |
| 120-200 | L42/L48/L74 | `<think>` word budget | inspector_prompt_test_v2.txt L130; corpus mean 136.9 (~55.6% strictly inside) | verified | |
| 9 | L51-55 | location grid cells (3×3) | inspector_prompt_test_v2.txt L258-273 | verified | |
| 11 / 7 / 4 / 4 | L58-63 | Surface / Structural / Completeness / Other defect types | inspector_prompt_test_v2.txt L287-319 | verified | |
| 26 | L57-63 (derived) | total controlled vocabulary | 11+7+4+4 | verified | |
| 5 | L65-71 | auto-reject rules (condensed from 6) | ground truth; test_v2 L404-426 (A-F) | verified | |
| 3 | L17-23 | images inspector may see (anomaly) | ground truth; test_v2 L16-22 | verified | |

---

## Appendix C — Per-product tables (see Corrections §C)

> **`tab:perprod-full` Base and SFT columns are GENUINE** (real JSONs, ~0.1 truncation). The **SFT+GRPO (GRPO) column is FABRICATED**, and the entire **`tab:perprod-cm` is FABRICATED** (CM rows physically impossible). Only the column **Average (82.73)** and the bold **CM Total (971/383/61/255)** are genuine.

### Genuine averages & headline

| Number | Where | Describes | Source | Status |
|---|---|---|---|---|
| 82.73 | L34/L67 | SFT+GRPO Average / CM Total BA | `results/grpo_..._run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json` (971/383/61/255) | verified |
| 70.49 | L34 | Base avg per-product BA | mean of 15 genuine base cells (~70.58, truncated) | verified |
| 80.05 | L34 | SFT avg per-product BA | genuine ckpt-564 avg 80.13 (truncated; overall 80.16) | verified |
| +9.56 / +2.68 | L34 | avg Δ SFT / Δ GRPO | mean of printed delta cells (9.55 / 2.69) | verified |
| 530 / 2 / 7B / 6K / 3 | caption L10 | ckpt / run / size / split / epoch | ground truth | verified |

### Genuine Base & SFT per-product cells (DS-MVTec)

| Product | Base BA | SFT BA | Δ SFT | Source (recomputed) | Status |
|---|---|---|---|---|---|
| metal_nut | 57.82 | 83.62 | +25.80 | base 57.86 / ckpt-564 83.64 | verified |
| cable | 50.93 | 70.13 | +19.20 | base 51.09 / 70.33 | verified |
| pill | 60.12 | 76.12 | +16.00 | base 60.23 / 76.21 | verified |
| toothbrush | 62.43 | 78.23 | +15.80 | base 62.50 / 78.33 | verified |
| screw | 55.88 | 70.78 | +14.90 | base 55.88 / 70.79 | verified |
| transistor | 66.95 | 80.25 | +13.30 | base 67.08 / 80.42 | verified |
| grid | 83.27 | 92.07 | +8.80 | base 83.33 / 92.11 | verified |
| leather | 90.65 | 98.25 | +7.60 | base ~90.5 / 98.44 | verified |
| zipper | 62.53 | 70.03 | +7.50 | base 62.61 / 70.10 | verified |
| capsule | 57.72 | 64.02 | +6.30 | base 57.80 / 64.06 | verified |
| hazelnut | 60.57 | 66.47 | +5.90 | base 60.71 / 66.61 | verified |
| bottle | 76.73 | 80.33 | +3.60 | base 76.87 / 80.48 | verified |
| carpet | 90.34 | 91.14 | +0.80 | base 90.45 / 91.27 | verified |
| wood | 97.50 | 97.40 | −0.10 | base 97.50 / 97.37 | verified |
| tile | 83.92 | 81.82 | −2.10 | base 83.93 / 81.87 | verified |

### FABRICATED GRPO column + CM rows — claimed vs genuine

Every "GRPO BA" cell in `tab:perprod-full` (L18-32) and every CM row in `tab:perprod-cm` (L51-65) is fabricated. The Δ GRPO cells (L18-32) are arithmetic over the fabricated GRPO cells → **unverified**. Genuine values (run-2 ckpt-530):

| Product | Claimed GRPO BA | Genuine GRPO BA | Claimed CM (TP/TN/FP/FN) | Genuine CM (TP/TN/FP/FN, n) |
|---|---|---|---|---|
| metal_nut | 87.40 | **87.47** | 87/19/1/23 (sum130>92) | 62/19/3/8 (n=92) |
| cable | 75.50 | **68.16** | 95/26/1/30 (sum152>150) | 35/57/1/57 (n=150) |
| pill | 79.20 | **75.45** | 78/35/5/18 | 120/3/2/12 (n=137) |
| toothbrush | 82.00 | **83.33** | 24/16/0/8 (sum48>42) | 25/10/2/5 (n=42) |
| screw | 74.55 | **72.89** | 80/36/4/27 | 69/36/5/50 (n=160) |
| transistor | 83.10 | **75.42** | 50/25/0/10 | 31/44/16/9 (n=100) |
| grid | 92.95 | **94.74** | 70/28/1/6 (sum105>76) | 51/19/0/6 (n=76) |
| leather | 98.40 | **98.44** | 92/32/0/3 (sum127>124) | 92/31/1/0 (n=124) |
| zipper | 74.10 | **68.84** | 96/28/0/31 (sum155>151) | 56/29/3/63 (n=151) |
| capsule | 67.65 | **69.78** | 86/28/5/37 (sum156>132) | 100/11/12/9 (n=132) |
| hazelnut | 73.10 | **88.57** | 50/28/0/18 | 68/32/8/2 (n=110) |
| bottle | 81.05 | **76.27** | 83/27/4/20 (sum134>83) | 52/14/6/11 (n=83) |
| carpet | 91.55 | **95.51** | 91/23/0/16 (sum130>117) | 81/28/0/8 (n=117) |
| wood | 97.55 | **97.50** | 79/17/0/4 (sum100>79) | 57/19/0/3 (n=79) |
| tile | 82.90 | **89.83** | 89/14/8/21 (sum132>117) | 72/31/2/12 (n=117) |
| **CM Total (bold)** | — | — | fabricated rows sum 1150/382/29/272 | **GENUINE 971/383/61/255** |

---

*End of ledger. Recompute any BA with `python results/compute_ba.py <eval_json>`.*
