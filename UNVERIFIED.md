# UNVERIFIED — Honest Ledger of Unbacked Thesis Claims

This document is the to-fix / to-verify list for the MSc thesis **"Reasoning-Enhanced
Vision-Language Models for Explainable Industrial Anomaly Detection"** (Adnane Acudad,
TU Delft, 2026). It records everything in the thesis text that is **not (yet) backed by
on-disk evidence** in this repository, separated into four classes:

1. **Confirmed inconsistencies** — the data exists, and the thesis number is demonstrably wrong.
   **STATUS UPDATE (2026-06-12): §1.1, §1.2 and §1.3 have been FIXED in the thesis source**
   (`tab:baseline` cell → 56.14 + all baseline cells set to exact 2-dp values; `tab:sft-summary`
   cross-contaminated cells corrected from the true checkpoints and the two phantom Unfrozen-6K
   rows commented out with a `% TODO`; Appendix C `tab:perprod-full` SFT+GRPO column, `Δ GRPO`
   column and the entire `tab:perprod-cm` regenerated exactly from
   `grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json`,
   the misleading footnote replaced with an honest macro/micro-average note, and the dependent
   Ch.7 "GRPO concentrates gains on cable/metal_nut/pill" claim rewritten — GRPO in fact
   *regresses* cable/pill/transistor/zipper/bottle and its true gains are hazelnut/tile/capsule).
   The tables below are retained as the audit record of what was wrong.
2. **Unverifiable** — the artifact needed to check it is not on disk (checkpoint rotated away, run never executed, probe-only metric).
3. **Unverified / partially-supported claims** — rhetorical, design-rationale, estimate, or cross-reference values with no backing file.
4. **Pending / in-progress** — placeholder sections awaiting results.

**Authoritative numbers** are recomputed from the eval JSONs in [`results/`](results/) using
[`results/compute_ba.py`](results/compute_ba.py); the ground-truth balanced-accuracy (BA)
inventory is [`results/eval_ba_inventory.txt`](results/eval_ba_inventory.txt).
BA = `0.5 * (TP/(TP+FN) + TN/(TN+FP))` from each JSON's `metrics {tp,tn,fp,fn}`.
Where the thesis and this ledger disagree, **trust this ledger.**

Eval-set sizes: DS-MVTec n=1670, VisA n=2141, held-out Real-IAD n=4236.
Headline checkpoints: SFT = 7B-frozen-6K **ckpt-564** (DS 80.16 / VisA 64.78);
GRPO = run-2 **ckpt-530** (DS 82.73 / VisA 70.39 / held-out 80.87);
Arm-C SFT = **ckpt-376** (DS 82.80 / VisA 72.07, single best detector);
IAD-R1 recanon (DS 81.92 / VisA 71.34).

---

## 1. Confirmed inconsistencies (data exists; thesis number is WRONG — must be fixed)

These are not judgement calls. The backing JSON is on disk and the thesis figure does not
match it. Fix each before any public release.

### 1.1 Baseline table — isolated transcription typo

| Location | Claimed | Correct | Backing JSON | Fix |
|---|---|---|---|---|
| `06_results.tex` tab:baseline L25 — Qwen2.5-VL-3B base DS-MVTec BA | **55.80%** | **56.14%** | `outputs/baseline_eval/3b_base_dsmvtec.json` (TP1222 TN56 FP388 FN4) | Replace 55.80 → 56.14 |

### 1.2 tab:sft-summary — cross-contaminated rows (`06_results.tex` L66–74)

Several rows carry numbers belonging to *other* runs/epochs/checkpoints. The HEADLINE
7B-Frozen-6K ep3 (ckpt-564) row is the only fully-correct row (80.16 / 64.78 / 80.36 / 85.76).

| Row (true ckpt) | Field | Claimed | Correct | Backing |
|---|---|---|---|---|
| 3B-Frozen-6K ep4 (ckpt-752) | DS | 68.50 | **69.08** | [`results/sft_qwen25vl_3b_zeroshot_6k_frozen/`](results/sft_qwen25vl_3b_zeroshot_6k_frozen/) |
| 3B-Frozen-6K ep4 (ckpt-752) | VisA | 59.60 | **57.22** | same |
| 3B-Frozen-6K ep4 (ckpt-752) | Acc | 71.60 | **72.22** | same |
| 3B-Frozen-6K ep4 (ckpt-752) | F1 | 78.10 | **80.02** | same |
| 3B-Frozen-15K ep4 (ckpt-1812) | DS | 70.50 | **68.65** | [`results/sft_qwen25vl_3b_15k_frozen/`](results/sft_qwen25vl_3b_15k_frozen/) |
| 3B-Frozen-15K ep4 (ckpt-1812) | VisA | 60.85 | **64.85** | same |
| 3B-Unfrozen-15K ep4 (ckpt-1812) | DS | 71.86 | **68.58** (71.86 is a 7B number) | [`results/sft_qwen25vl_3b_15k_unfrozen/`](results/sft_qwen25vl_3b_15k_unfrozen/) |
| 3B-Unfrozen-15K ep4 (ckpt-1812) | Acc | 74.19 | **69.58** | same |
| 3B-Unfrozen-15K ep4 (ckpt-1812) | F1 | 81.38 | **77.34** | same |
| 7B-Frozen-15K ep3 (ckpt-1359) | VisA | 65.45 (= ep1/ckpt-453) | **64.28** | [`results/sft_qwen25vl_7b_15k_frozen/`](results/sft_qwen25vl_7b_15k_frozen/) |
| 7B-Frozen-15K ep3 (ckpt-1359) | Acc | 78.20 (= ep1/ckpt-453) | **80.12** | same |
| 7B-Unfrozen-15K ep4 (ckpt-1812) | DS | 72.60 (= 7B-FROZEN value) | **70.27** | [`results/sft_qwen25vl_7b_15k_unfrozen/`](results/sft_qwen25vl_7b_15k_unfrozen/) |
| 7B-Unfrozen-15K ep4 (ckpt-1812) | Acc | 68.86 | **65.21** | same |
| 7B-Unfrozen-15K ep4 (ckpt-1812) | F1 | 75.60 | **71.51** | same |

**Downstream consequence — RETRACTED (2026-06-12 recheck):** an earlier revision of this note
claimed the corrected 7B-Frozen-15K ep3 "DS = 80.12" would collapse the 6K-vs-15K gap to ~0 pp.
That confused **plain accuracy with balanced accuracy**: 80.12 is the DS-MVTec *accuracy* at
ckpt-1359; the DS **BA** there is **71.66** (TP/TN/FP/FN-recomputed), exactly what the thesis
table prints. The "8.5 pp" matched-epoch gap (80.16 − 71.66 = 8.50) and the "~7.6 pp" best-epoch
gap (80.16 − 72.60 at ep4, = 7.56) are therefore **both genuine** and the thesis prose needs no
change. Only the row's VisA (65.45 → 64.28) and Acc (78.20 → 80.12) cells were wrong.

### 1.3 Appendix C — FABRICATED SFT+GRPO per-product column and the entire CM table

This is the most serious integrity issue. In [`appendix/C_*`](.) (`tab:perprod-full` SFT+GRPO column
L18–34 and the whole `tab:perprod-cm` L51–65):

- The printed per-product **SFT+GRPO BA cells do not match** the genuine ckpt-530 DS JSON
  ([`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/eval_dsmvtec_full_trainprompt.json`](results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530/)).
- The per-product **confusion-matrix counts are physically impossible**: for 10–11 of 15 products
  the row TP+TN+FP+FN *exceeds that product's test-set sample size* (e.g. metal_nut row sums 130 > n=92;
  grid 105 > 76; bottle 134 > 83; capsule 156 > 132).
- The CM rows sum to TP1150/TN382/FP29/FN272, which **does not equal** the genuine bold Total
  TP971/TN383/FP61/FN255.
- The fabricated column was evidently **back-filled so its 15-product mean = 82.7333 → 82.73**, the
  genuine headline overall BA — i.e. the average is real but the per-product breakdown is invented.
- The footnote (L72) blames a "MMAD per-product test-set design" for the row/total mismatch. This is
  **misleading**: genuine per-product CMs sum *exactly* to the chapter Total; the discrepancy is purely
  an artifact of the fabricated rows.

**Only genuine in Appendix C:** the Base column, the SFT (ckpt-564) column, the column Average (82.73),
and the CM Total (971/383/61/255).

**Correct per-product GRPO values** (recompute from ckpt-530 DS JSON; format = BA% (CM TP/TN/FP/FN, n)):

| Product | Printed (fabricated) | Correct BA | Correct CM (TP/TN/FP/FN, n) |
|---|---|---|---|
| cable | 75.50 | **68.16** | 35/57/1/57, n=150 |
| pill | 79.20 | **75.45** | 120/3/2/12, n=137 |
| toothbrush | 82.00 | **83.33** | 25/10/2/5, n=42 |
| screw | 74.55 | **72.89** | 69/36/5/50, n=160 |
| transistor | 83.10 | **75.42** | 31/44/16/9, n=100 |
| grid | 92.95 | **94.74** | 51/19/0/6, n=76 |
| leather | 98.40 | **98.44** | 92/31/1/0, n=124 |
| zipper | 74.10 | **68.84** | 56/29/3/63, n=151 |
| capsule | 67.65 | **69.78** | 100/11/12/9, n=132 |
| hazelnut | 73.10 | **88.57** | 68/32/8/2, n=110 |
| bottle | 81.05 | **76.27** | 52/14/6/11, n=83 |
| carpet | 91.55 | **95.51** | 81/28/0/8, n=117 |
| wood | 97.55 | **97.50** | 57/19/0/3, n=79 |
| tile | 82.90 | **89.83** | 72/31/2/12, n=117 |
| metal_nut | 87.40 | **87.47** | 62/19/3/8, n=92 |

**Fix:** regenerate the SFT+GRPO column and the CM table directly from the JSON, delete the misleading
footnote, and *re-derive every Delta-GRPO cell* (they are internally consistent but inherit fabricated
SFT+GRPO numbers, so the reported per-product gains are wrong). Note that with genuine numbers GRPO does
**not** improve every product — it regresses bottle, transistor, pill, etc. — so the "positive Delta GRPO
for all 15 products" finding is itself an artifact of the fabrication and must be retracted.

### 1.4 Other confirmed-wrong numbers scattered through the text

These have data on disk that contradicts them; each should be corrected at the cited location.

| Location | Claimed | Correct | Backing |
|---|---|---|---|
| `00_abstract` / `01_intro` / `08_concl` — gap SFT→IAD-R1 | "~1 pp" / "roughly 1pp" | **1.76 pp** (81.92 − 80.16) | [`results/iad_r1_qwen_recanon/`](results/iad_r1_qwen_recanon/) |
| `01_intro` contrib 4 — DS gain over base | +13.6 pp | **+13.72 pp** (GRPO 82.73 − base 69.01) | [`results/qwen25vl_baseline_eval/`](results/qwen25vl_baseline_eval/) |
| `03_dataset` L17/L169/L220 — products in corpus | 30 | **29** (6K SFT & 15K union span 29 of 30; GRPO/held-out span 23) | trace JSONs in [`traces/`](traces/) |
| `03_dataset` L82 — controlled defect-type categories | 25 | **26** (Surface 11 + Structural 7 + Completeness 4 + Other 4 — the thesis's own list) | [`prompts/`](prompts/) |
| `03_dataset` L171 — token-length std | "≈28 tokens" | **≈13.5 tokens** (Qwen tokenizer; range 121–223) | corpus recompute |
| `03_dataset` L179 — Scratch cluster size | n=600 | **500** (exact 'Scratch' strings) or 866 (Scratch-dominant cluster) | `anomaly_type_analysis/types.json` |
| `03_dataset` L179 — Missing-component cluster | n=180 | **162** (exact 'Missing component' strings) | same |
| `03_dataset` L179 — t-SNE inter-centroid distance | 148.6 | **≈155.3 units** | `embeddings_2d.npy` recompute |
| `03_dataset` L179 — within-cluster radius | 16.5 | **≈15.3** | same |
| `03_dataset` L179 — separation ratio | 9.0 | **≈10.1** (155.3/15.3) | same |
| `04_sft` L29 — 3B transformer layers | 28 | **36** | `config.json` text_config.num_hidden_layers |
| `04_sft` L89 — 7B-unfrozen-15K wall-clock | 22.5 h | **~12.2 h** (43875 s) | `all_results.json` |
| `04_sft` L107 — AdamW beta2 | 0.95 | **0.999** | `training_args.bin` |
| `04_sft` L108/L123 + App-A — SFT learning rate | 2e-5 ("all cells") | **1e-5 frozen / 1e-6 unfrozen** (no run used 2e-5; LR was *not* constant) | sft yamls / `training_args.bin` |
| `04_sft` L109/L125 + App-A — LR schedule | linear, 100-step warmup | **cosine, warmup 20 (6K) / 50 (15K)** | `training_args.bin` |
| `04_sft` L110 + App-A — weight decay | 0.01 | **0.0** | `training_args.bin` |
| `04_sft` L111 — per-device BS (unfrozen) | 2 | **16 (7B) / 4 (3B)** | sft yamls |
| `04_sft` L112 — grad accumulation (unfrozen) | 8 | **1 (7B) / 4 (3B)** | `training_args.bin` |
| `04_sft` L114 + App-A — sequence length | 4096 | **12144** (`cutoff_len`) | sft yamls |
| `05_grpo` L108/L160 + App-A — GRPO sample count | 6,500 + 6,500 = 13,000 | **4,236 (2,118 + 2,118)** | [`traces/anomalythink_15k/grpo_train.json`](traces/anomalythink_15k/) len=4236 |
| `05_grpo` L165 — mean-KL range | [0.003, 0.095] | **[0.0, 0.117]** (max ~0.117; 13/530 steps exceed 0.095) | run-2 trainer_state |
| `06_results` L342 — products that gain | "fifteen others gain" | **thirteen** (wood & tile regress) | per-product recompute |
| `06_results` L415 — SFT loss at "step 752 / 0.34 / ep4" | step 752, loss 0.34 | run ends at **step 560** (no step 752); final loss **~0.57**. "752/0.34/ep4" is 3B-run cadence | ckpt-564 trainer_state |
| `07_discussion` L108 — GRPO run-1 DS at ckpt-530 | 80.6% | **81.65%** (80.63 is run-1 *ckpt-315*) — wrong checkpoint, overstates the seed gap | [`results/grpo_qwen25vl_7b_6k_frozen_ep3_full_run1/`](results/) (run-1 not shipped; see §2) |
| `08_concl` contrib 5 — metal_nut/cable/pill gains | +25.8 / +19.2 / +16.0 | **CORRECT as written** — these are **base→SFT** gains (verified: 57.9→83.6, 51.1→70.3, 60.2→76.2), matching the Ch6 per-product chart the conclusion cites. The earlier "base→GRPO correction" (+29.6/+17.1/+15.2) was **mis-scoped**; no change needed. | base + SFT ckpt-564 |
| `08_concl` RQ1 L31 — mean trace length | ≈141 words | **≈137 words** (136.8 for *both* 6K and 15K; the equality claim holds, the absolute does not) | corpus recompute |
| App-A Table A.3 — GRPO warmup_ratio | 0.1 | **None** (schedule linear is correct; warmup_ratio not set in run-2 args) | run-2 `training_args.bin` |
| App-A Table A.3 — GRPO AdamW beta2 | 0.95 | **0.999** | run-2 `training_args.bin` |
| App-A Table A.3 — GRPO weight decay | 0.01 | **0.0** | run-2 `training_args.bin` |
| App-A Table A.3 — GRPO per-device BS | 2 | **1** (EBS still 8 via GA=4 × 2 GPUs) | run-2 `training_args.bin` |
| App-A Table A.3 — GRPO max prompt length | 2048 | **4096** | run-2 `training_args.bin` |
| App-A Table A.3 — GRPO save every | 250 | **530** (run-2) | run-2 `training_args.bin` |
| App-A Table A.3 — GRPO eval every | 250 | **None** (no in-training eval; checkpoints evaluated post-hoc) | run-2 `training_args.bin` |
| App-A Table A.3 — GRPO image stitch | 1024×512 | **N/A to production run** (run-2 launched `--single_img 1`, no stitch; 1024×512 is the unused 1-shot path) | run script + `2IMAGE_IMPLEMENTATION_SUMMARY.md` |

> **Note on App-A vs an earlier ground-truth note for GRPO BS/save/eval/max-prompt:** the *saved
> `training_args.bin` is authoritative* — per-device BS=1, save=530, eval=None, max_prompt=4096. Earlier
> internal notes that said BS=2 / save=250 / max_prompt=2048 are themselves wrong and must not be
> "corrected back" into the thesis.

---

## 2. Unverifiable (no eval JSON / checkpoint on disk — cannot confirm OR fix from data)

These cannot be checked because the artifact that would settle them is absent. They are not
necessarily *wrong*, but they are **unsupported** and should be flagged in the thesis as such.

| Item | Location | Why it can't be checked | What would be needed |
|---|---|---|---|
| **3B-Unfrozen-6K** and **7B-Unfrozen-6K** sft-summary rows | `06_results.tex` tab:sft-summary | 3B-unfrozen-6K was **never run** (no dir, no yaml). 7B-unfrozen-6K has only a **partial** run (`outputs/sft_qwen25vl_7b_zeroshot_6k`, 2 epochs, DS-only, best DS 72.82) — never extended to 4 epochs/VisA. The previously-printed rows had duplicated 15K-unfrozen data. | **RESOLVED (2026-06-13):** decided **not** to run them — unfreezing is already shown to give no DS gain and to hurt VisA at 15K, and IAD-R1 recommends a frozen ViT. Both rows removed; grid reframed to **6 configurations** (unfrozen evaluated at 15K only) in intro / Ch4 §design / 06_results L41 / Ch6, with the omission stated as a deliberate, literature-backed design choice. |
| **GRPO-on-C ckpt-795** (DS 80.76 / VisA 69.69, deltas −2.04/−2.38) | `06_results.tex` tab:grpo-on-c L560; `07_discussion` L78 | Checkpoint dir was **rotated away** by `save_total_limit`. The shipped [`results/grpo_qwen25vl_7b_abc_C_grpo/`](results/grpo_qwen25vl_7b_abc_C_grpo/) has ckpt-265/530/1060 (and 1325/1590/1855) but **no ckpt-795**. | Re-run GRPO-on-C and re-save ckpt-795, or report only the surviving checkpoints (265: −2.19, 530: −2.48, 1060: −1.60) and drop the 795 row. The "1.99 → 2.54 reward rise" attributed to the ckpt-795 window is likewise unverifiable (surviving ckpt-1060 trainer_state shows 1.92 → 2.09). |
| **Estimator-ablation step-0 "init" cells** (DS 84.52 / VisA 71.89) | `06_results.tex` tab:grpo-estimator-ablation L578; `07_discussion` L78 | These live only in `probe_curve.csv` (probe-only fast eval); there is **no tp/tn/fp/fn JSON** for step 0, and the probe dirs are **not shipped in this repo's `results/`**. The 84.52 differs from the full-eval Arm-C 82.80 (probe runs ~1.7 pp high). | Ship the probe CSVs/dirs, or recompute step-0 on the full subset to get a CM-backed number. State clearly that probe figures are not full-eval comparable. |
| **GRPO run-1** ckpt-315/530 numbers | `07_discussion` L108 (run-1 vs run-2 seed comparison) | The run-1 output dir is **not shipped** in `results/` (only run-2 is). The §1.4 correction (run-1 ckpt-530 = 81.65, not 80.6) was made from the original outputs tree, not this repo. | Ship run-1 eval JSONs if the seed-sensitivity claim is to be defensible from this repo. |

---

## 3. Unverified / partially-supported claims (no backing file; rhetorical, estimate, or design-rationale)

Grouped by chapter. None of these is necessarily false, but **none can be substantiated from disk** and
several are flatly *contradicted* by the configs (those are called out). They should be softened,
sourced, or removed.

### Ch.1 Introduction
- **">95% accuracy demanded by most production floors"** (sec:motivation L20) — rhetorical industrial
  threshold, no citation. (Likewise GPT-4o "below the >95% production requirement" — the 74.9% MMAD
  figure is sourced; the 95% floor is not.)
- **"last five years"**, **"three decades"** (sec:motivation L8) — rhetorical historical timeline, no source.
- **"16-cell SFT ablation grid"** (sec:contributions item 2; also `06_results` L40) — 2 of the 16 cells
  (3B/7B-Unfrozen-6K) were **never run** (see §2). The grid is not fully populated.
- **"6K concise traces outperform 15K verbose ones"** (sec:contributions item 2) — **internally
  inconsistent**: both corpora have the same ~137-word mean length and the 6K is a 100% subset. The
  effect is data-count/composition, NOT length/verbosity. The "concise vs verbose" framing must be removed.
- **"25 / 30 Real-IAD training products"** — Appendix A's 25/5 split table conflicts with §3's "all 30"
  and the actual 29 products in the corpus image paths. Reconcile (see §1.4 product-count fix).

### Ch.2 Background
- **"3 mm" scratch**, **"4–5 mm" scratch** — illustrative example sizes, not measurements.
- **"six-phase template"** (chapter intro L5, §2.3 L61) — the operational XML schema is **4 tags**
  (think/loc/type/ans); "six-phase" is a conceptual narrative-phase count deferred to Ch.4. The on-disk
  production prompt ([`prompts/inspector_prompt.txt`](prompts/inspector_prompt.txt)) uses a "3–5 sentence"
  template, not a labelled six-phase one (see Appendix-B notes below).
- **GRPO objective with `−β·KL` inside the loss** (§2.4.2) — correct for *general* GRPO but **not for
  this thesis's run** (β=0; KL is monitored via k3, not penalised). The numerical divergence is deferred to Ch.5.

### Ch.3 Dataset / Traces
- **120–200 word `<think>` budget** (L44/L69/L171) — only [`prompts/inspector_prompt_test_v2.txt`](prompts/)
  (a **test** prompt, *not* the one the production script loads) says "Target 120 to 200 words". The
  production [`inspector_prompt.txt`](prompts/inspector_prompt.txt) says "3–5 sentences". Corpus mean ~137 words.
- **"600–1000" earlier verbose template range** (L69) — refers to a Ch.6 prompt-evolution experiment; no data here.
- **"six-phase template (Framing/Scan/Focus/Evaluate/Alternatives/Decide)"** (L60–68) — **no on-disk prompt
  contains this verbatim.** Corpus *content* does follow a structured first-person narrative (4299/6000
  open with "I am beginning/Scanning"), but the exact six-phase wording is not recoverable from the shipped code.
- **"five auto-reject rules"** (L123–130, L140) — the production prompt has **none**; the test prompt has
  6 lettered rules (A–F). QC is in-prompt auto-reject + Pydantic schema only.
- **"160–270 token budget"** (L171) — actual 6K think-token range is 121–223 (5th/95th pctile 142–187);
  the 270 upper bound is never reached.
- **"4-second latency / batch"**, **"$0.0001/trace"**, **"1.6 s median/trace"**, **"<$20 total"** (L132/L172/L173)
  — **none logged on disk**; all are retail-pricing/timing estimates.
- **"top-12 type families"** (L179) — presentation choice; not an independently verified count.
- **Anomalous-mask filter (Multi-Close/Single/Spread)** (L17) — described pipeline filter; **no
  mask-classification artifact inspected on disk.**

### Ch.4 SFT
- **"1K-trace pilot sweep"** (L95) — no 1K-pilot artifact on disk.
- ~~**"At 5e-5 the model collapses to an empty trace"**~~ — **RESOLVED 2026-06-13: DELETED.** Verified no 5e-5 run exists in any YAML or training_args.bin on disk; the anecdote was removed from Ch4.
- ~~**"cosine under-utilises... linear+100-step warmup gave higher BA"**~~ — **RESOLVED 2026-06-13: paragraph rewritten** (runs used cosine + 20/50 warmup; the false linear claim removed). Original note for the record:
  **CONTRADICTED by configs**: every SFT run actually used **cosine** with 20/50-step warmup. The deployed
  runs are the *opposite* of what this rationale recommends. The "100-step / 1.5% of budget" figure is also
  internally inconsistent (actual 20/752 ≈ 2.7%).
- **"beam/nucleus changed BA by <0.3 pp on a 200-sample check"** (L160) — no 200-sample sanity-check on disk.
- **"100K-image dataset loaded lazily"** (L64) — qualitative source-pool figure, not directly counted.
- **"256 GB system RAM"** (L72) — hardware spec, not logged.
- **"~300M ViT parameters"** (overfitting argument, `06_results` L83) — Qwen2.5-VL ViT is ~675M; not verified.
- **"epochs=4 across the grid"** — the on-disk 3B-frozen-6K run actually ran **6 epochs** (global_step 1128).
- **"full-FT-over-LoRA pilot"** (L31) — `finetuning_type:full` is confirmed; the "small pilot" has no artifact.
- **"bf16 more stable than fp16"** (L74) — bf16 is confirmed in args; no logged fp16-failure artifact.
- **AdamW (0.9, 0.95), weight decay 0.01, grad clip 1.0, seed 42** (App-A L66) — **not set in any sft yaml**;
  these fall to library defaults (0.9, 0.999) / 0.0 / 1.0 / 42. The (0.9, 0.95) and wd 0.01 are unsupported.

### Ch.5 GRPO
- Reward-design constants — **+0.075/tag partial credit**, **−0.1 forbidden-tag penalty**, **type sim
  1.0 exact / 0.8 substring**, **ε=1e-8 std stabiliser** — design values consistent with the trainer impl,
  not independently dumped.
- **"<2% GRPO samples with unreadable reference after path migration"** (L108) — production-log claim,
  not verified.
- **AdamW (0.9, 0.95), weight decay 0.01** (tab:grpo-hp) — see §1.4 (actual 0.999 / 0.0).
- **"~30 s/rollout"** (L146) — train.log shows 83.85 s/step for G=4 → ~21 s/rollout gen portion; same order, not exact.
- **"~600 MB Nomic footprint", "<50 ms/step embedding overhead", "~700 lines of Python"** (L154/L156) —
  impl/code claims, not measured or counted.
- **"~73% verdict-correct at start"** (L163) — interpretive (0.875/1.2 ≈ 0.73), consistent.
- **"completion length stable 163–174 tokens"** (L166) — central tendency holds, but full range is
  153–186 (extremes exceed the stated band).
- **"12.4 h wall-clock"** (L167) — matches reaching headline ckpt-530 (epoch 1); the **full 2-epoch run is
  ~24.7 h**. State which the figure refers to.
- **Sample reward-line listing** (lst:reward-line L171–174, "step 287 / 0.6521 / 0.2715 / 0.0009 / Epoch 0.6")
  — illustrative per-batch snapshot; does not match the trainer_state step-287 reward (1.703). Label as illustrative.
- **G2RPO discrete reward bins {0.0, 0.3, 0.6, 1.0}** (L200/L209) — plausible per design, not exhaustively enumerated.
- **"2,700 type substitutions" / "1,167 location normalisations"** (L229, also `06_results` L193) —
  cross-referenced to the dataset chapter; a `grpo_train.json` type/loc-fix diff showed **0 changed rows**,
  so these counts **cannot be confirmed from the GRPO file** and the underlying cleaning artifact is not on disk.
- **"5–7 h / run", "15–20 h total"** (L239) — forward-looking compound-experiment estimates (placeholder section).
- **"≥0.5 pp pass threshold"** (compound design) — pre-registered design choice.
- **Framework = "HF trl GRPO trainer, train_grpo_oneshot_upgraded.py ~700 lines"** (L154) — **tension** with
  the rest of the chapter and ground truth, which place the trainer in IAD-R1/AnomalyR1 lineage
  (`iad_r1_grpo_custom/.../sc_grpo_trainer.py`). The exact script was not located/counted; reconcile the
  trl-vs-IAD-R1 lineage statement.
- **Outer-whitening flag "minimal difference"** (L213) — pilot claim; no separate outer-whitening eval JSON found.

### Ch.6 Results
- **"16 SFT grid cells"** (L40) — see §2 (2 cells never run).
- **3B/7B-Unfrozen-6K rows** (L68, L73) — config never run; numbers duplicate 15K-unfrozen (see §2).
- **"7.6 pp" 6K-vs-15K gap** (L81) and **"~2 pp T-vs-G prompt agreement"** (L186) — depend on the
  cross-contaminated 15K row / prompt-mode caveat; directionally OK, magnitude tied to a flagged number.
- **"~300M ViT"**, **"4–12 pp 7B>3B"**, **"~1 pp SFT vs IAD-R1"** (L83/L85/L88) — qualitative ranges /
  loose rounding (the true SFT→IAD-R1 gap is 1.76 pp).
- **Data-cleaning counts 2,700 / 1,167** (L193) — artifact not on disk (see Ch.5 above).
- **Gemini-patch counts: 3,770 patched, 94.8% NG, 5.2% OK** (L195) — internally consistent
  (3770×0.948=3574); underlying artifact not on disk.
- **Rollout-failure-mode table** (L206–207: 1,541 / 871 / 807 / 176) — arithmetic internally consistent;
  the backing rollout artifact (`phase0_variety_star`) is an **empty placeholder on disk**.
- **6K-pool composition 3,557 kept + 2,443 patched** (L226) — sums to 6000 (matches the SFT file), but the
  kept/patched sub-split is not on disk.
- **"96% / 95% NG gradient signal"** (L256/L259) — derived estimate, not directly measured.
- **Arm sub-counts: "812 Arm-B corrected (31% NG)", "1,631 Arm-C rewritten", "95% Arm-A pass rate"**
  (L456–460) — arm **totals** verified (A=2978, B=4369, C=6000) but these **sub-counts cannot be
  reconciled** (B−A=1391≠812; `gemini_rewritten.jsonl`=2406≠1631). NG splits not on disk.
- **GRPO-on-C ckpt-795** (80.76/69.69) and **estimator-ablation init** (84.52/71.89) — see §2.

### Ch.7 Discussion
- **"4–8 pp DS / 4–10 pp VisA frozen-vs-unfrozen gap"** (sec:disc-frozen L11) — qualitative range; with the
  *corrected* 15K rows the DS gap is ~0 (frozen 68.65 vs unfrozen 68.58); the gap is real and large **on
  VisA recall** but the stated DS magnitude is not cleanly reproducible.
- **"100× / 1.5M-trace corpus at which unfreezing could win"** (L21) — hypothetical, no backing.
- **"8.5-point 6K-vs-15K DS gap is the single largest effect"** (sec:disc-quality L27) — see §1.2; with
  corrected 15K ep3 DS=80.12 the matched-epoch gap is ~0 pp. State the comparison consistently.
- **"6K more structurally uniform; 15K drifts into longer prose"** (sec:disc-quality L33) — **conflicts** with
  the fact that 6K is a literal subset with equal mean length; no structural-uniformity measurement provided.
- **"15K under-trained at 4 epochs / 2.5× more updates"** (L31) — pilot loss-curve observation; arithmetic
  (2.41×) consistent, but no curve file shipped.
- **"~0.7% VisA / ~2.1% DS mean defect area", "~3× downsampling", "512/896 px stitch resolution"** (sec:disc-visa)
  — benchmark statistics / engineering estimates with no backing file.
- **"~3.5 pp earlier VisA deficit (now an artefact)"** (L41) — older eval, not on disk as JSON.
- **"β ≈ 0.05–0.2 future small-KL safeguard"** (L57) — future-work proposal (NOT the old/wrong β=0.04).
- **"~92% SFT / >99% GRPO format compliance" and "~8% decisions lost to parser fall-throughs"** (L70) —
  **no format-compliance metric exists in any eval JSON or log on disk.** This is the headline GRPO
  "structured-output reliability" claim and it is entirely unbacked — measure it or remove it.
- **"2.0–2.5 pp below Arm-C init at every checkpoint"** (L78/L83) — holds for ckpt-265 (−2.19) and
  ckpt-530 (−2.48); ckpt-1060 is only **−1.60**. Tighten the wording.
- **Error taxonomy: 200 hand-inspected (100 FN / 100 FP), ~32% sub-pixel FN, ~24% specular FP, ~18%/14%
  layout, ~11% annotation, ~5% format** (sec:disc-errors L90–97) — **manual analysis with no file on disk
  recording the inspection or counts.** The entire five-category failure-mode breakdown is unbacked.
- **"3–5 seeds recommended", "80% illustrative accuracy"** (L108/L110) — recommendation / rhetorical.
- **Latency: "170 tokens at ~30 tok/s → 5–6 s/inspection, 2 orders slower than EfficientAD"** (L112) —
  no latency-benchmark file on disk; carried into the Ch.8 closing summary as well.

### Ch.8 Conclusion
- **"25 Real-IAD training products"** (contrib 1) — conflicts with §3's 30 and the actual 29; see §1.4.
- **"roughly 1pp" SFT→IAD-R1**, **metal_nut/cable/pill per-product gains** — see §1.4 (corrected magnitudes).
- **"≈141 words mean trace length"** (RQ1) — ≈137 (the *equality* of 6K and 15K holds; absolute is wrong).
- **"five error categories", "5–6 s/inspection"** — cross-refs to the unbacked Ch.7 error taxonomy & latency estimate.
- **Future-work proposals** (3×3 grid reward, 3× VisA downsample, 896×896 stitch, k=8 rollout) — methodological proposals.

### Appendix A
- **Generator temperature 1.0** (Table A.1) — the production script's `GenerateContentConfig` does **not set
  temperature**; 1.0 is the Gemini API default, not explicit in shipped code.
- **SFT LR 2e-5, schedule linear, warmup 100, wd 0.01, seqlen 4096, BS/GA 4/4 vs 2/8, AdamW (0.9,0.95)** —
  see §1.4 (all contradicted by yamls / `training_args.bin`).
- **3B/7B-Unfrozen-6K rows present in Table A.2** — those cells were never run; no 6K-unfrozen yaml exists.
- **GRPO warmup_ratio 0.1, AdamW 0.95, wd 0.01, BS 2, max-prompt 2048, save/eval 250, 1024×512 stitch** —
  see §1.4 (contradicted by saved run-2 args / `--single_img 1`).
- **"max_steps 1,060"** (Table A.3) — saved args have `max_steps=-1` (driven by 2 epochs over 4,236 at EBS 8
  → 1059≈1060). 1060 is the *derived* total, not a set `max_steps`. Reword.

### Appendix B (Gemini & inference prompts)
- **"SIX-PHASE INSPECTION TEMPLATE (FRAMING…DECIDE)"** — the appendix reproduces an **idealised/paraphrased**
  prompt. The production script loads [`inspector_prompt.txt`](prompts/inspector_prompt.txt) (an older
  "3–5 sentence" prompt, no six-phase labels, no auto-reject rules), while the six-phase / 120–200-word /
  26-type-vocab / auto-reject structure matches [`inspector_prompt_test_v2.txt`](prompts/) — a **test prompt
  loaded only by the in-progress Variety-STaR generator**, not by the headline-corpus generator. The
  appendix is therefore a redacted/composite view, not the verbatim prompt that built AnomalyThink-15K.
- **"5 auto-reject rules only, no scoring rubric"** — faithful to the *stated* QC design, but the source
  `inspector_prompt_test_v2.txt` it is condensed from actually **embeds an 8-category 0–2 scoring rubric +
  13/16 PASS/FAIL threshold + rewrite policy** that the appendix omits. Disclose that the reproduced prompt
  is redacted.
- **"A single inference system prompt served SFT, GRPO and evaluation"** — **not literally true**: the eval
  script uses a single-image prompt with no six-phase / no controlled-vocab enforcement; the GRPO script uses
  a two-image prompt with different wording. Only the output-tag contract (think/answer/type/location) is
  consistent. The appendix prompt is an idealised composite.

---

## 4. Pending / in-progress (placeholders awaiting numbers)

These sections are explicitly forward-looking. They are **honest placeholders** today but must either be
filled with real eval numbers or clearly marked as future work in the final submission.

- **Real-IAD Variety STaR (F5)** — `05_grpo` §5.7, `06_results` compound section, `08_concl` F5. The
  6,000-trace variety corpus exists on disk
  ([`traces/variety_star_6k/`](traces/variety_star_6k/), 3000 OK / 3000 NG, 160 products), but the SFT/GRPO
  **eval results are not produced**; the `phase0_variety_star` rollout dirs are empty placeholders. The
  compound-design table (`tab:compound-design`) lists only baselines-to-beat (F5a 81.94 / F5b 82.73 / F5c 80.64),
  no results column.
- **GRPO-on-C continuation** — surviving checkpoints (ckpt-265/530/1060/1325/1590/1855) are in
  [`results/grpo_qwen25vl_7b_abc_C_grpo/`](results/grpo_qwen25vl_7b_abc_C_grpo/), but ckpt-795 (cited in the
  thesis) was rotated away (see §2). The continuation past ckpt-1060 has no thesis numbers attached.
- **Compound experiments F5a/F5b/F5c** (`05_grpo` §5.7, `06_results` `tab:compound-results`) — methodology
  written, results cells are TBD. Explicitly marked placeholder; keep it that way until run.

---

*Bottom line:* the headline results (SFT 80.16, GRPO 82.73/70.39/80.87, Arm-C 82.80/72.07, IAD-R1 81.92/71.34,
and the "corpus quality > RL" central finding) **are genuine and recompute exactly** from the shipped JSONs.
The integrity problems are concentrated in (a) the Appendix-C fabricated per-product GRPO column/CM table,
(b) the cross-contaminated tab:sft-summary rows, and (c) a large body of hyperparameter values that
contradict the saved configs. Fix §1, disclose §2–§4, before any public commit.


## 5. Verification pass of 2026-09-06 (what still cannot be backed from disk)

Full report: [`docs/verification/NUMBER_VERIFICATION_REPORT.md`](docs/verification/NUMBER_VERIFICATION_REPORT.md).

| Item | Thesis location | Why it is here |
|---|---|---|
| Appendix H, 6K VisA-versus-epoch curve | Figure H.1 | only ckpt-564 of the 6K run has a VisA evaluation on disk; drawing script not found |
| Figure 6.9 / Appendix F.5 renderer | §6.10.3, F.5 | the PNGs and index files exist (`results/sft_vs_kcr_pairs/`), the arrow renderer was written in a session scratch area and is not on disk |
| 3,557 kept + 2,443 patched, 94.8 % anomalous | §6.7 | reproduced only approximately (3,471 to 3,580 kept, 96 %), no build log |
| MMAD ground-truth type coverage 94.8 % / 99.3 % | §6.11 | not recomputed (an older note says 99.2 % for VisA) |
| "40 of 100 sampled IAD-R1 outputs are a bare Yes" | §6.11 | the 100-sample is not stored; over the full set 336 of 964 true positives (35 %) are bare "Yes" |
| t-SNE pool n = 8,908 | §5.2 | the cached embeddings in `results/anomaly_type_analysis/` have n = 8,908 and reproduce 155.3 / 15.3 / 10.1 / 500 Scratch, but no dataset file with 8,908 type tags was identified (the 15K union has 7,247 anomalous traces) |
| Probe initialisation 84.52 / 71.89 | Table 6.10 step 0 | no probe file for Arm-C ckpt-376 on disk; equals 82.80 + 1.72, consistent with the stated offset |
| IAD-R1 group size G = 4 "matching their configuration" | §5.4 | not in the IAD-R1 paper text, would be in their released config |
| Literature numbers without a local PDF | Chapter 2 | PatchCore 98.4 pixel AUROC (the paper reports 98.1 to 98.2), CFLOW 98.3, DRAEM 98.0, RD4AD 98.5, CutPaste 96.6, SimpleNet 77 FPS, EfficientAD 2 ms / 600 FPS, WinCLIP 91.8, InternVL3 72.2, LLaVA 158K / 92.5, ToT 4 to 74, OmniAD 79.1, ViT 675M, MVTec 5,354, VisA 10,821, Real-IAD 150K / ~50K |
| Which Arm-C file version trained ckpt-376 | §5.7, `traces/iter2/` | two versions on disk differing in 316 `<location>` spellings, no log records which was present at launch |
| 3B-frozen-6K epochs | Appendix A | the run trained six epochs (checkpoints to 1128); the thesis reports epochs 1 to 4 by decision |
