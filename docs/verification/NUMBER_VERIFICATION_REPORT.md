# Number and claim verification of the local thesis, with a repo-vs-not-in-repo map

Date: 6 September 2026, evening. Scope: the local thesis at `/bulk/aacudad/ol_thesis` (commit b834f4f, 9 commits ahead of `origin/main`). Nothing was changed, committed or pushed. Everything below is recomputed from files on disk under `/bulk/aacudad/reasoning_traces`, without subagents.

Method in one paragraph. Every evaluation JSON on disk and in the repo (802 files) was rescored with the strict rule of Chapter 4: an output with no parsable `<answer>` counts as wrong. Every chapter and appendix was read in full and each number was matched to a source file (eval JSON, `trainer_state.json`, `training_args.bin`, yaml, dataset file, rollout pool, judge output, paper PDF). Then every cited file and script was looked up in `repository_tu_delft_vlms` by md5.

Headline result: the eight ladder numbers (Qwen base 69.08/53.79, SFT 80.16/64.78, SFT+GRPO 82.73/70.39, KCR 82.80/72.07, LLaVA base 75.66/53.80, SFT 85.91/68.26, SFT+GRPO 87.66/72.38, KCR 87.32/72.65), the IAD-R1 rows, the Gemini and GPT-5-mini rows, the per-epoch tables, the GRPO-on-C table, the estimator probe table, the localisation table, the judge table, the appendix tables and the dataset statistics all reproduce exactly. The problems are in a small number of derived claims, listed in Part 1.

---

## Part 1. Findings that need a decision (ranked by weight)

### 1. An on-disk run contradicts the universal negative on GRPO-on-C

The abstract, §6.8, §7.5 and §8.1(5) say that GRPO on top of the Arm-C initialisation "did not improve binary accuracy in our experiments" and that "every checkpoint lands below the initialisation on DS-MVTec". That is true for the run the tables report (`grpo_qwen25vl_7b_abc_C_grpo`, β=0, GRPO prompt).

There is a second, complete run on disk from the same initialisation that the thesis does not mention: `outputs/grpo_sftprompt_kl0.1_sys_3ep` (β=0.1, eval-aligned prompt with the system turn, 3 epochs, 15 checkpoints, 1,590 steps). Its full-benchmark evaluations, strict scoring:

| ckpt | DS-MVTec | VisA |
|---|---|---|
| Arm-C init | 82.80 | 72.07 |
| 530 | 82.88 | 71.34 |
| 636 | 81.41 | 73.70 |
| 954 | **82.95** | **72.62** |
| 1590 (end) | 81.25 | 71.51 |

Checkpoint 954 is above the initialisation on both benchmarks. Two of fifteen checkpoints beat it on DS-MVTec and eleven of fifteen on VisA. The repo's `THESIS_HEADLINE_DECISION.md` records the decision to treat this as future work and not claim it, but the thesis text as written is a universal statement ("in our experiments"), which this run falsifies by a small margin. Options: narrow the claim to the production recipe (β=0, GRPO prompt), or add one sentence in §6.8 or §8.4 that a prompt-aligned β=0.1 run gives at most +0.15/+0.55 at one checkpoint and is left as preliminary. The repo already holds the ckpt-954 evals and a NOTE, but not the other 14 checkpoints.

### 2. "Every pairwise test among our six models is non-significant" is not true for all pairs

§6.11 reports twelve Welch tests with p between 0.10 and 0.94, and §8.2 (RQ4) generalises this to "every pairwise test among them is non-significant". Recomputed on the per-trace medians in `outputs/explainability_multi/raw_results*.json`, all 15 pairs per benchmark:

| benchmark | smallest p | pair | t |
|---|---|---|---|
| DS-MVTec | 0.019 | Qwen KCR (9.05) vs LLaVA SFT+GRPO (9.51) | −2.35 |
| VisA | 0.18 | Qwen3-VL vs LLaVA SFT | 1.33 |

So the six-pair selection in the chapter is fine as stated, but the RQ4 sentence should say "the twelve tests we ran" or the DS-MVTec pair should be acknowledged. Also the two Welch t-statistics quoted against IAD-R1 (t=12.07 and t=5.55) recompute as t=15.61 and t=6.50 with the same differences (+3.25, +1.66). The conclusion does not change, but the printed t values do not reproduce from the raw file.

### 3. Some table cells use lenient scoring (unparsed outputs dropped) against the rule stated in Chapter 4

Chapter 4 says a missing `<answer>` counts as wrong. Files with unparsed rows were scored the other way in a few cells:

| where | thesis | strict | unparsed rows |
|---|---|---|---|
| Table 6.6, Arm C ep3 DS | 81.39 | 81.26 | 2 |
| Table 6.6, Arm C ep4 DS | 79.13 | 78.96 | 5 |
| Table 6.6, Arm C ep4 VisA | 71.59 | 71.53 | 2 |
| Table 6.6, Arm B ep3 VisA | 67.48 | 67.46 | 1 |
| Table 6.9, Qwen3-VL-8B ep2 VisA | 76.52 | 76.45 | 2 |

None of these is a headline cell, and the headline files have zero unparsed rows. L7's "+2.04" for Arm C also rests on the lenient 81.39/79.13. Note the LLaVA KCR VisA cell (72.65, 13 unparsed) is already scored strictly and footnoted, so the chapter is inconsistent with itself here.

### 4. Appendix C uses the older baseline file, and the "+11.2 pp" in Chapters 1 and 8 comes from it

Table 6.1 and every chapter use `outputs/baseline_eval/7b_base_dsmvtec.json` (69.08). Appendix C's base column comes from `outputs/qwen25vl_baseline_eval/eval_dsmvtec_full_trainprompt.json` (69.01). Two cells differ: transistor 67.08 (App C) vs 67.92 (Table 6.1 file), leather 90.83 vs 90.29, and the macro average 70.58 vs 70.60. The contribution list in §1.4 and §8.1(4) says "+11.2 pp overall (sample-weighted)". With the Table 6.1 baseline the overall gain is 80.16 − 69.08 = +11.08. The 11.2 figure is 80.16 − 69.01 rounded. The repo ledger `NUMBER_PROVENANCE.md` also carries 69.01/53.80 as the baseline and states it is authoritative over the thesis, which is a second inconsistency to settle before the repo is pushed.

### 5. Smaller number mismatches

| where | thesis says | recomputed | note |
|---|---|---|---|
| §6.10.2 per product | LLaVA `pcb3` loses 6.5 on VisA | −7.5 | from `figdata.json` (corrected corpus) |
| §7.2 (b) | 15K "training loss still descending at end of epoch 4" | flat at 0.57–0.58 through epochs 3 and 4 | `trainer_state.json` of ckpt-1812 |
| §6.3 | 6K and 15K loss "converging to the same floor" | 0.57 vs 0.58 at matched epoch 3, but 0.34 vs 0.58 at epoch 4 | true only at the matched epoch |
| §1.4 vs §7.2 | 15K gets "2.4×" (Ch1) / "2.5×" (Ch7) more optimiser steps | 14,472 / 6,000 = 2.41× | pick one |
| §6.14, §7.4 | ckpt-1060 worse "despite slightly lower training loss" | GRPO loss is 0.000 at both checkpoints | the loss comparison is meaningless for GRPO |
| §6.4 | specificity 86.27, VisA specificity 77.13 | 86.26, 77.12 | rounding |
| §5.7 Arm A | "balanced subset of 2,978 (50/50 NG/OK ...)" | 1,298 NG / 1,680 OK = 43.6% NG | the "up to availability" clause exists, but 50/50 is wrong as written |
| §5.7 LLaVA pool | 9,179 kept, judge confirms 9,045, "demotes 133" | 9,179 − 9,045 = 134, `needs_rewrite_from_judge.json` has 133 | off by one |
| §6.9 | HF vs vLLM agreement on Qwen/Qwen3 rows "79.6 to 90.4%" | 87.6 to 94.3% on the files now on disk | range not reproduced, probably an earlier checkpoint set |
| §6.7 | 3,557 kept + 2,443 patched, 94.8% anomalous | by matching traces to rollouts: 3,471–3,580 kept, 96% anomalous | close but not exact, no build log found, the repo ledger also marks it unverified |
| Appendix A / Ch4 | 3B-frozen-6K "4 epochs" | the run trained 6 epochs (`all_results.json` epoch 6.0, checkpoints to 1128) | table reports epochs 1–4 only, and the 1.6 h wall-clock is 2.35 h scaled by 4/6 |
| Appendix E | `reward.py` lines 256–275 / 277–341, trainer line 801 | 544–581 / 582–660, line 825 (repo copy: 810) | stale line references |
| §5.2 | t-SNE over "every anomalous trace, n=8,908" | the 15K union has 7,247 anomalous traces | the cached embeddings have n=8,908 and all three statistics (155.3, 15.3, 10.1, 500 Scratch) reproduce from them, but no dataset file with 8,908 type tags was found |
| Appendix A | generator "Temperature 1.0", "batch 5 to 10" | no temperature is set in the code (API default), `BATCH_SIZE = 10` | wording |
| Table 6.5 | G²RPO 81.94 next to single-stage 82.73 | 81.94 is under the GRPO prompt, 82.73 under the train prompt | the caveat paragraph says so, the caption does not |
| Table 6.9 | rows mix HF (SFT 6K, Arm-C carried) and vLLM (GRPO, KCR) paths | agreement 93.9–98.4% as stated | covered by the "two generation paths" paragraph, noted only |

### 6. Claims that could not be verified from disk

- Appendix H, Figure H.1: the 6K run has a VisA evaluation only at ckpt-564, so the VisA-versus-epoch curve for the 6K run has no data source on disk. The script that drew `curve_sft_ba.png` was not found.
- Figure 6.9 and Appendix F.5 pair images: the generator script is not on disk anywhere (it lived in a session scratchpad). The 21 PNGs and `index.json` exist.
- §6.11: MMAD ground-truth type coverage "94.8% DS / 99.3% VisA" (the repo ledger says 99.2%). Not recomputed.
- §6.11: "40 of 100 sampled IAD-R1 outputs are a bare Yes". On the full set 336 of 964 true positives (35%) are bare "Yes", so 40 of a 100-sample is plausible but the sample itself is not stored.
- §5.4: "matching the group size in the IAD-R1 configuration". G=4 is not in the IAD-R1 paper text, it would be in their released config.
- Probe initialisation 84.52/71.89 (Table 6.11 step 0): no probe file for ckpt-376 exists, the value equals 82.80 + 1.72, consistent with the stated offset.
- Literature numbers checked against local PDFs and confirmed: MMAD 39,672 / 8,366 / GPT-4o 74.9, IAD-R1 5.9K / 2.9K / 3K, Real-IAD Variety 160 categories / 28 industries / 10–20% drop, AnomalyGPT 86.1 / 94.1, SimpleNet 99.6, LIMA 1,000 / 65B, RAFT-vs-GRPO 56.1 / 56.3. Not checkable locally (no PDF): PatchCore 99.6 / 98.4 (the paper's pixel AUROC is 98.1–98.2, worth a look), CFLOW 98.3, DRAEM 98.0, RD4AD 98.5, CutPaste 96.6, SimpleNet 77 FPS, EfficientAD 2 ms / 600 FPS, WinCLIP 91.8, InternVL3 72.2, LLaVA 158K / 92.5, ToT 4→74, OmniAD 79.1, ViT 675M, MVTec 5,354, VisA 10,821, Real-IAD 150K / ~50K.
- The four Hugging Face URLs in §8.5 were not checked (offline).

---

## Part 2. Verification ledger by chapter

Status codes: OK = reproduces exactly from the named source. ~ = reproduces within rounding or with a wording caveat. X = does not reproduce (see Part 1). ? = not verifiable from disk.

### Abstract and Chapter 1

| claim | status | source |
|---|---|---|
| 14,472 traces = 6,000 + 4,236 + 4,236 | OK | `datasets_small_new_v4/combined_6k_train.json`, `15k_dataset_regenerated/combined_sft_train.json`, `datasets_small_15k_c1_only/grpo_train.json`, `datasets_sft_iter2/heldout_4236_disjoint.json` |
| DS-MVTec 1,670 / VisA 2,141 | OK | every eval file |
| SFT 80.16/64.78, GRPO 82.73/70.39, KCR 82.80/72.07, held-out 80.87 | OK | registry files |
| IAD-R1 81.92/71.34 | OK | `iad_r1_qwen_recanon/eval_*_full_trainprompt.json` |
| LLaVA KCR 87.32/72.65 (ep2), 74.29 (ep4), SFT+GRPO 87.66/72.38 | OK | `..._C_original/checkpoint-376,748/*_vllm.json`, `grpo_llava_ov_from_ep1/checkpoint-530/*_vllm.json` |
| k=8 over 10,236 = 6,000 + 4,236 | OK | `phase0_full_10k_20260529_015821/rollouts_raw.jsonl` (81,888 rollouts) |
| 6K beats 15K by 8.5 at matched epoch (80.16 vs 71.66) | OK | `sft_qwen25vl_7b_15k_frozen/checkpoint-1359` |
| GRPO-on-C 1.6 to 4.0 below, three estimators, 120 steps = 0.23 epoch | OK | `grpo_qwen25vl_7b_abc_C_grpo`, `grpo_probe_{ctrl,drgrpo,g2rpo}` |
| both datasets ~137 words mean | OK | 136.8 and 136.8 |
| 5–6 s per image, 170 tokens, 30 tok/s | ? | estimate, stated as such in L3 |
| within ~1.8 pp of IAD-R1 | OK | 81.92 − 80.16 = 1.76 |
| 2.4× optimiser steps | ~ | 2.41× (Ch7 says 2.5×) |
| +0.81 / −0.95 vs IAD-R1 | OK | |
| +11.2 pp overall SFT over base | X | +11.08 with the Table 6.1 baseline, see Part 1 §4 |
| metal_nut +25.8, cable +19.2, pill +16.0, toothbrush +15.8, tile −2.1, wood −0.1 | OK | recomputed per product |
| Figure 1.1 numbers | OK | same files |

### Chapter 2

Literature only, see Part 1 §6 for what was and was not checkable. Internal consistency: β=0, G=4 stated correctly.

### Chapter 3

| claim | status | source |
|---|---|---|
| 6K: 3,000/3,000, 30 products, C1 only | OK | dataset file |
| 15K: 7,247/7,225, 30 products | OK | |
| GRPO: 2,118/2,118, 23 products, per-product 8–302 | OK | `gt_label` and `product` fields |
| held-out: 2,129/2,107, 23 products | OK | |
| think block mean 165 tokens / 137 words, p5–p95 142–187, range 121–223 | OK | Qwen tokenizer on the `<think>` text of the 6K split (164.7 / 142 / 187 / 121 / 223). For the 15K union the minimum is 53 tokens |
| 26-type vocabulary, 9 locations, 8 codes, rubric 13/16 | OK | prompt file (489 lines) |
| batch ≤10, 3 retries, 4 s wait | OK | `config_mmad.py` |
| Expert-AD 5.9K, Anomaly-Instruct-125k | OK | IAD-R1 PDF |
| 30% raw codes in the original 6K type tags | OK | `combined_6k_train.json.bak_pre_typefix`: 900 of 3,000 |

### Chapter 4

| claim | status | source |
|---|---|---|
| 3B 36 layers / 16 heads / 2048, 7B 28 / 28 / 3584 | OK | checkpoint `config.json` |
| image_max_pixels 262,144 train and eval | OK | yamls, `evaluate_qwen25vl_7b_trainprompt.py` |
| lr 1e-5 / 1e-6, cosine, warmup 20 / 50, wd 0, EBS 32, cutoff 12,144, bf16, clip 1.0, seed 42 | OK | yamls (`sft_qwen25vl_3b_15k_frozen_ep5.yaml` uses warmup 10, but that run is not in the grid) |
| batch 4/4, 16/1, 8/2 | OK | yamls |
| wall-clock 6.9 h, 12.2 h, 1.6 h, 5.1 h | ~ | `all_results.json`: 6.93, 12.19, 2.35 h for 6 epochs (1.57 scaled), 5.10 |
| 7B-unfrozen-6K pilot peak 72.8 then 69.27 | OK | `sft_qwen25vl_7b_zeroshot_6k/checkpoint-200,376` |
| greedy, 1024 tokens, system line | OK | eval scripts |
| <0.5% parse failures | OK | 0 to 13 unparsed on trained checkpoints |
| DeepSpeed listing | OK | `ds_z3_cpu_offload.json` |

### Chapter 5

| claim | status | source |
|---|---|---|
| G=4, ε=0.2, β=0, lr 1e-6, linear, batch 1×4×2, 4096/512, 480,000 px, 1,060 steps, save 530, seed 42, bf16 | OK | `training_args.bin` of run 2 |
| per-sequence loss normalisation (Eq. 5.2) | OK | custom trainer line 908 (the trl `loss_type=dapo` flag in the args is not used by this trainer) |
| type bins 0.90/0.80/0.70/0.55/0.40 | OK | `reward_process/type_reward.py` |
| Nomic loaded in-process on CPU for run 2 | OK | run-2 `train.log` shows the model load, 0 lexical fallbacks (the current code calls the judge server over HTTP and falls back to lexical similarity when it is down) |
| t-SNE 8,908 / 500 Scratch / 155.3 / 15.3 / 10.1 | ~ | reproduces from `anomaly_type_analysis/`, source pool not identified |
| run 2: reward 1.66 → 2.22 → 2.13, acc 0.875 → 1.25 (+42.9%), format 0.781 → 0.875, KL [0, 0.117] with 13 of 530 > 0.095, length 153–186 mean 167, 11.3 h / 24.7 h, ckpt-1060 −0.70 / −0.53 | OK | `checkpoint-1060/trainer_state.json`, `train.log` (11:14:54 at step 530, 24:41:17 total) |
| GRPO-on-C probes: reasoning reward fallback 0.484–0.500 | OK | `grpo_qwen25vl_7b_abc_C_grpo` trainer_state |
| iter-2: 4,026 traces, ckpt-378/504, saves every 265 | OK | checkpoint names |
| Arm A 2,978 (84.6% / 90.8% pass on the 6,000 SFT images) | ~ | pass rates OK on the `sft_6k` subset (whole pool: 84.0 / 90.2), balance not 50/50 |
| Arm B 3,557 + 812 = 4,369, 31% NG, discards 1,631 | OK | file has 4,369 at 31.4% NG |
| Arm C 6,000 50/50, threshold 1.8 | OK | |
| 8,872 kept + 1,364 corrected = 10,236, 2,406 rewrites | OK | `phase0_full_10k_20260529_015821/` |
| LLaVA pool 9,179 / 1,057 / 9,045 / 2,513 | OK | `phase0_llava_10k_20260901/` (133 vs 134, see Part 1 §5) |
| Arm C 82.80 vs iter2-clean 81.07 (+1.73) | OK | |

### Chapter 6

| claim | status |
|---|---|
| Table 6.1 (all 30 cells) | OK |
| LLaVA-OneVision-Data 1,999 of 186,060 rows, zero for VisA | OK, `defense_prep_20260901/scratch_contam/totals.json` and `answers/contamination_evidence.md` |
| Table 6.2 (six cells, acc and F1) | OK |
| 7.56 / 2.16, 3B union ahead on both, 15K trails at every epoch, best is last | OK |
| freezing 69.56 vs 68.58, 72.60 vs 72.08, VisA 66.94 vs 58.60 | OK |
| 7B beats 3B by 3 to 11 | OK (3.04, 3.50, 11.08) |
| verdict-only control: 1.2 to 3.7 behind, FPR 20.3 vs 34.0, VisA 69.39 vs 64.78 | OK, `sft_qwen25vl_7b_6k_noreason` |
| Table 6.3 GRPO (acc, F1, precision 94.09) | OK (specificities 86.26 / 77.12 vs printed 86.27 / 77.13) |
| Table 6.4 iter-2 progression (six rows, GRPO prompt) | OK, `grpo_qwen25vl_7b_iter2_v2_full/*_grpoprompt.json` |
| Table 6.5 G²RPO 81.94 | OK, `iter2_v1_eval_archive_20260517_2343/grpo_qwen25vl_7b_g2rpo_full/checkpoint-530/*_grpoprompt.json` |
| Table 6.6 per-epoch arms (12 rows) | OK except the four lenient cells of Part 1 §3 |
| Table 6.7 deltas | OK |
| Arm A VisA-best +5.43 | OK |
| 30% raw codes, 81,888 rollouts, failure table 1,541 / 871 / 2,412 / 807 / 176 / 983 with percentages | OK |
| 3,557 kept + 2,443 patched, 94.8% anomalous | ~ (Part 1 §5) |
| Table 6.8 (three variants, averages, deltas, 322 unparsable) | OK |
| Figure 6.5 net −1.66, five up / eight down / two unchanged | not recomputed |
| Table 6.9 GRPO-on-C (six checkpoints) | OK |
| reward 1.9 → 2.4 over 2,120 steps, peak 3.25 | OK |
| Table 6.10 estimator probe (18 cells), averages 70.98 / 69.71 / 69.26, +0.75 at step 20, probe +1.7 | OK, `grpo_probe_*/checkpoint-*/probe_*.json` |
| Table 6.11 LLaVA (all rows incl. Qwen3-VL-8B 78.68/64.45 and 85.82) | OK except 76.52 (strict 76.45) |
| 72.58 over 2,135 parsable | OK |
| restart 87.86 / 71.89, 1.91 at ep4, −0.34 / +0.27 | OK |
| first build 3,516 / 2,484, 45.0% NG, 88.45 / 74.25 | OK |
| HF vs vLLM 93.9 to 98.4% on LLaVA rows, DS middle rows swap | OK, Qwen range not reproduced |
| Qwen3-VL lift > 7 and > 12 | OK (7.14, 12.07) |
| operating point 57.1 / 87.1 vs 72.5 / 72.8, 0.58, −6.6 / +10.0 sensitivity/specificity, base 7.6 | OK |
| per product: VisA +7.3 overall, pcb4 +23.2, macaroni2 +10.5, fryum +0.5, all twelve up, DS +2.6, six of fifteen regress, pill −17.7, transistor −7.9, leather −7.8 | OK |
| LLaVA pcb3 −6.5 | X (−7.5) |
| cases 42 / 42 / 121 / 107, composition (102 of 121 normal, 104 of 107 defects) | OK |
| 65-image pool, 21 shown | OK (`sft_vs_kcr_samples_all65/index.json`) |
| Table 6.12 localisation (12 cells), 1,226 / 1,197 anomalies, deltas 9.2 / 17.8 / 5.6 / 17.0 | OK, `thesis_figures_v2/loc_hit.json` |
| Table 6.13 judge (8 rows × 6), spreads 0.46 / 0.37, SE 0.12–0.23, gaps +2.91 / +1.36 / +3.37 / +1.73, +3.25 / +1.66, evidence axis 1.95–2.00, loc z=2.37 p=0.018 and z=1.61 p=0.11 | OK except the Welch t values and the "every pair" sentence (Part 1 §2) |
| Table 6.14 SOTA (all rows) | OK |
| Gemini within 1.4 of 6K, leads by 3.11 / 4.79, 60.3 / 90.0; GPT-5-mini 96.1 / 58.1, Arm C +5.70 / +3.84 / +1.28 | OK |
| SFT loss 1.14 / 0.58 / 0.34 at steps 50 / 500 / 752 | OK (logged at 750) |
| KL 0.10–0.20, max 0.197; length 153–186 | OK |

### Chapter 7

| claim | status |
|---|---|
| unfrozen 7B VisA recall 34–41%, frozen ~80% | OK |
| GPT-5-mini verifier passes 97.8% | OK, `outputs/verify_15k_regen_results/summary.json` (14,158 of 14,472) |
| filtered-6K retrain 79.60 / 65.99 | OK, `sft_filtered6kcc_from_base/checkpoint-364` (the filter file has 5,797 items, 96.6%) |
| 15K loss still descending | X |
| LIMA 1,000 / 52,000 / 16× | OK |
| gaps 10.7 and 10.6 | OK |
| mask area 1.0% VisA / 3.8% DS over 1,197 / 1,226 masks | OK (1.00%, 3.79%) |
| VisA downsampled ~2.5× per side | OK (2.39× at the median 1404×1070) |
| completion length 176 at step 1, 163 at 1060 | OK |
| per-product GRPO redistribution (+22.0, +8.0, +5.7, −5.0, −4.2, −2.2) | OK |
| loc 82.4 / 55.9 / 73.2 / 38.1 / 74.9 / 52.7 | OK |
| L1: run 1 81.65 vs run 2 82.7 | OK |
| L7: +0.67, +2.03, +2.04, +2.00, average 1.7 | OK, the four runs are Arms A, B, C (lenient) and the first LLaVA build |
| L7: VisA optimum at a different epoch on two of three arms | OK |

### Chapter 8

Restates verified numbers. Two carried-over issues: "+11.2 pp" (Part 1 §4) and "every pairwise test among them is non-significant" (Part 1 §2). "8.3 pp on VisA" = 8.34 OK. Seven contributions counted OK. Variety STaR 6K dataset exists (`variety_star_sft_6k`).

### Appendices

| appendix | status |
|---|---|
| A: product coverage (7 SFT-only products), split composition, per-cell SFT table, GRPO table, DeepSpeed diff | OK (3B-frozen-6K trained 6 epochs, table shows 4) |
| B: 489-line prompt | OK |
| C: per-product DS table and confusion counts | OK except the base column (Part 1 §4) |
| D: per-product VisA table and confusion counts | OK |
| E: reward semantics | OK, line numbers stale |
| G: 475 VisA false negatives for the teacher | OK |
| H: reward range 0.56–2.83, MA 1.48 → 2.0, KL, length | OK; 6K VisA curve data source unknown |
| I: datasheet numbers | OK |
| J: prompt-mode table, confusion counts (six rows), type-similarity table, 18 / 435 / 218 | OK |
| L: held-out 80.81 / 80.87 / 91.28 / 68.34 / 78.16 / 93.40 | OK |
| M: per-epoch corrected build, first build, 1.85 | OK |

---

## Part 3. What is in `repository_tu_delft_vlms` and what is not

State: last commit 2773d6e, working tree clean. The repo does not yet know about anything produced after 3 September.

### 3a. Evaluation files the thesis cites (by md5 against `results/`)

Identical copies exist for 64 of the 83 cited files, including the four Qwen and LLaVA baselines (under `results/baseline_named/`), every Qwen SFT grid cell, run 1 and run 2, iter-2, the arms, GRPO-on-C, the IAD-R1 re-runs, the held-out split, the noreason control, the first LLaVA build and the Qwen3-VL Arm-C run.

Missing from the repo:

| what | local path | thesis use |
|---|---|---|
| corrected LLaVA KCR run, all four epochs (HF and vLLM files) | `outputs/sft_llava_ov_7b_frozen_iad_sft_llava_iter1_C_original/` | Tables 6.9, 6.12, 6.13, Appendix M, Figures 1.1, 6.1, 6.6, 6.7, 6.8 |
| LLaVA trained on the Qwen Arm-C corpus | `outputs/sft_llava_ov_7b_frozen_iad_sft_iter2/` | Table 6.9 row 3 |
| LLaVA GRPO restart | `outputs/grpo_llava_ov_from_ep1_ep2/` | §6.9 "87.86 / 71.89" |
| LLaVA GRPO HF-path evals | `outputs/grpo_llava_ov_from_ep1/checkpoint-530/eval_*_full_trainprompt.json` | §6.9 two-paths paragraph |
| LLaVA SFT vLLM evals and the HF backfill for the other LLaVA epochs | same dirs, `*_vllm.json` | §6.9 two-paths paragraph |
| Gemini 2.5-Flash and GPT-5-mini zero-shot | `outputs/gemini_25flash_eval/`, `outputs/gpt5mini_eval/` | Table 6.14, §6.12, Appendix G |
| filtered-6K retrain | `outputs/sft_filtered6kcc_from_base/` (repo has only a `ba_summary.txt` under another name) | §7.2 |
| 400-sample probe evals of the three estimator arms | `outputs/grpo_probe_{ctrl,drgrpo,g2rpo}/checkpoint-*/probe_*.json` (repo has the checkpoint dirs without the probe JSONs) | Table 6.10, Figure 6.6 |
| Qwen3-VL-8B on the LLaVA corpus | `outputs/sft_qwen3vl_8b_llavaC/` | not in the thesis |
| prompt-aligned β=0.1 run, 14 of 15 checkpoints | `outputs/grpo_sftprompt_kl0.1_sys_3ep/` (repo has ckpt-954 only) and `outputs/grpo_sftprompt_kl0.1/`, `outputs/grpo_abc_C_kl0.1_halfep/` | Part 1 §1 |
| frozen-vision GRPO run (in progress) | `outputs/grpo_qwen25vl_7b_6k_frozenvision_run1/` | planned appendix |
| judge outputs for the LLaVA rows | `outputs/explainability_multi/{summary,raw_results}_{corrected,llava_sft,llava_grpo,llava_armC}.json`, `summary_iadr1_tp.json` | Table 6.13 (repo has only `summary.json` and `raw_results.json`) |
| contamination count evidence | `defense_prep_20260901/scratch_contam/` (`totals.json`, `mvtec_ids.json`, `count_shards.py`, `vflan4v.py`) and `answers/contamination_evidence.md` | §6.1 caveat |
| localisation and type tables | `thesis_figures_v2/loc_hit.json`, `type_sim.json`, `tab_loc_hit.tex`, `tab_type_sim.tex`, `figdata.json` | Tables 6.12, J.5, Figures 6.1–6.8 |

### 3b. Training corpora and rollout pools (`traces/`)

| corpus | in repo | note |
|---|---|---|
| 6K SFT, 15K union, held-out 4,236, Arm A, Arm B, iter-2 held-out source | yes, identical | |
| Arm C | yes, but the older version | repo file equals `sft_iter2_train.json.old`. The current local file differs in 316 traces, all `<location>` spelling fixes (center-right → middle-right). Which version trained `abc_C_full_patched/checkpoint-376` is not recorded in any log found. §6.7 says the spelling was corrected before the arms were run, which points to the current file |
| GRPO 4,236 split | no | `traces/anomalythink_15k/grpo_train.json` has the same 4,236 images but different traces and a different question string (0 identical traces). The file GRPO actually trained on is `Training/datasets_small_15k_c1_only/grpo_train.json` |
| LLaVA KCR corpus, corrected and first build | no | `Training/datasets_sft_llava_iter1/sft_llava_C_original_train.json`, `sft_llava_C_train.json` |
| filtered-6K, noreason 6K, iter-2 balanced 192 | no | |
| Qwen rollout pool (10,236 raw rollouts, judge report, kept, corrected, rewritten, difficulty) | no | `Training/phase0_full_10k_20260529_015821/` (~100 MB raw) |
| LLaVA rollout pool | no | `Training/phase0_llava_10k_20260901/` |
| GPT-5-mini verifier outputs | partly (`results/verify_15k_regen`, `verify_c1_union`) | |

### 3c. Scripts

Missing from the repo: `build_llava_arms_original.py` (the corrected-corpus builder), `evaluate_vllm_qwen3vl.py`, `hf_llava_backfill.sh`, `run_grpo_7b_frozen_vision.sh`, `watch_and_eval_grpo_frozenvision.sh`, `analyze_anomaly_types.py`, `run_on_folder.py`, the whole `thesis_figures_v2/` figure pipeline (`registry.py`, `export_data.py`, `f1_ladder.py` … `f6_gallery.py`, `g1_sft.py` … `g5_flow.py`, `thesisify.py`, `loc_hit_table.py`, `type_sim_table.py`, `stages/f7_stages.py`, `svgkit.py`), the contamination scripts, the `FREEZE_VISION_TOWER` block in `sc_grpo_trainer.py`, and the pair-image generator (not on disk at all).

Present in the repo but differing from the working copy: `explainability_judge_multi.py` (18 lines), `evaluate_vllm_llava.py` (7), `kcr_rollout_vllm.py` (9), `phase0_rollout_llava_vllm.py` (11), `evaluate_qwen25vl_7b_trainprompt.py` (84), `gemini_judge_server_v2.py` (230), `run_grpo_sftprompt_3ep.sh` (10), `run_grpo_llava_vllm.sh` (32), `trainer/sc_grpo_trainer.py` (15, the freeze block). Identical: `build_llava_arms.py`, `phase0_rollout_kscoring.py`, `build_abc_datasets.py`, `reward.py`.

### 3d. Ledgers and docs

`NUMBER_PROVENANCE.md`, `CLAIMS_EVIDENCE.md` and `UNVERIFIED.md` contain none of: the corrected LLaVA numbers (84.35, 87.32, 72.65, 86.96, 74.29, 87.86), the localisation table, the type-similarity appendix, the contamination count (1,999 of 186,060), the frozen-vision run, the 65-image pair pool. They carry the older baseline (69.01 / 53.80) and a "3,770 patched" count that the thesis no longer uses, and `NUMBER_PROVENANCE.md` declares itself authoritative over the thesis. `THESIS_HEADLINE_DECISION.md` still describes the prompt-aligned run as in progress at epoch 1.6, it has since finished.

### 3e. The thesis itself

Local `ol_thesis` is 9 commits ahead of `origin/main` (56 files, +4,183 / −423 lines, incl. the 21 pair PNGs, the new Chapter 6 figures, `overview_big.pdf`, `06_pairs_fig.tex`, `F_pairs_generated.tex`, `M_llava_corrected.tex`, the research-question rewrite).

---

## Part 4. Run status at 20:31

Frozen-vision GRPO: 76 of 1,060 steps, 0 errors, 0 fallbacks, reward 1.44 → 1.68, KL 0.0064, now 82 s/step because GPU 3 is shared, ckpt-265 expected around 00:50. All three tmux sessions up.
