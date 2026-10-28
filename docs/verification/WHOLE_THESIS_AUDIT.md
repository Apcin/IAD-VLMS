# Whole-thesis audit, BEFORE vs AFTER (4 Sep 2026, 22:20)

Scope: both builds (BEFORE = origin/main, 141 pp. AFTER = working tree, 150 pp), every page rendered as a separate image (871 images, 90 dpi), every figure sent to Gemini 3.8 Flash individually at full resolution, every chapter and appendix audited from the live .tex, rubric scored, and six peer theses read section by section. No subagents. Nothing in the thesis was changed. Nothing was pushed.

Audit outputs: `thesis_figures_v2/audit/{A_figs,B_chapters,D_claims,E_rubric,F_beforeafter,G_peers,H_appendix}.txt`. Page images: `/tmp/pages/{before,after,peer_*}/p-NNN.png`.

## 1. Verdict in five lines

1. AFTER is the better thesis. Gemini prefers it blind, and it is stronger than the peer set in 5 of 6 sections (abstract on par).
2. Rubric estimate 8.5/10. Weakest criterion is 1a (single seed, no error bars). Everything else is scored 8 or 9.
3. The results chapter should keep SFT vs SFT+GRPO as the GRPO ablation. Base vs GRPO cannot be answered because no GRPO-from-base run exists. Section 2 below.
4. 19 concrete inconsistencies remain, 6 of them number-level. All small, all listed with the exact fix in Section 5. None changes a conclusion.
5. Figures: 12 of 19 in-chapter figures are house style. 5 are default matplotlib, 2 are neither. Chapter 6 itself is fully consistent (10/10 house style, 9 print-ready after fixes, 1 with a residual).

## 2. SFT vs GRPO, or base vs GRPO, or SFT vs base?

The question is which pairwise comparison the results chapter should be built around. The data answers it.

| Backbone | Base | SFT | SFT+GRPO | KCR (SFT on GRPO-derived data) |
|---|---|---|---|---|
| Qwen2.5-VL-7B DS-MVTec | 69.08 | 80.16 | 82.73 | 82.80 |
| Qwen2.5-VL-7B VisA | (Fig. 6.1) | 64.78 | 70.39 | 72.07 |
| LLaVA-OneVision-7B VisA (strict) | (Fig. 6.1) | 68.26 | 72.38 | 74.25 leaky / 72.65 corrected ep2 |

Pairwise, Qwen: base to SFT +11.08 DS. SFT to SFT+GRPO +2.57 DS / +5.61 VisA. SFT to KCR +2.64 DS / +7.29 VisA. SFT+GRPO to KCR +0.07 DS / +1.68 VisA.

Three facts decide the question.

- Every GRPO run in this thesis initialises from an SFT checkpoint. There is no GRPO-from-base run. So "base vs GRPO" is not an ablation the thesis can report. It would only be the cumulative ladder, which Figure 6.1 already shows.
- GRPO from base is not a sensible experiment here. The base model emits no `<think>/<answer>` tags, so the format reward is zero for every rollout in every group and the group-relative advantage is degenerate. That is exactly why the field (DeepSeek-R1, IAD-R1, OpenVLThinker) uses a cold-start SFT first. One sentence saying this belongs in §6.4.
- SFT vs SFT+GRPO is the only comparison that isolates what GRPO adds. It is the right ablation and the chapter already uses it.

Recommendation, no new experiments:
1. Keep §6.4 as SFT vs SFT+GRPO. Add one sentence: "We do not run GRPO from the base model. The base emits no structured tags, so the format reward would be zero in every group and GRPO would have no relative signal. A cold-start SFT stage is therefore required, as in DeepSeek-R1 and IAD-R1."
2. Keep base to SFT as its own explicit step. It is in Figure 6.1 and Table 6.1. The intro and conclusion also cite per-product base to SFT numbers "in Chapter 6", but those live only in Appendix C now. Fix the pointer (item 5.3 below).
3. Do not present KCR as "SFT vs KCR" without SFT+GRPO in between. The chapter's own honest reading is that KCR is a third SFT stage whose data came from GRPO. The ladder Base, SFT, SFT+GRPO, KCR is the correct story and it is what the figure shows.

If a committee member asks for GRPO-from-base at the defence, the answer is the format-reward argument above. Running it is feasible (about one day on two GPUs) but it would not change any conclusion and I do not recommend spending the last ten days on it.

## 3. BEFORE vs AFTER

Gemini, blind, prefers AFTER. What changed: Chapter 6 went from 12 to 14 sections with a protocol section first and a summary last. Appendix M added. Table 6.1 now carries the LLaVA base row. What AFTER lost: the SFT curves and frozen/unfrozen bars were on page 2 of the chapter in BEFORE and are now later. Gemini calls this a loss of "immediate visual presentation". My view: the protocol-first opening is what the peers do and what the rubric rewards. Keep it.

Section names: BEFORE names were closer to the peer convention (technical noun phrases). The hybrid already applied keeps AFTER structure with BEFORE-style names. Confirmed in place.

Page counts: Intro p12, Results p46, Discussion p70, Conclusion p78 (AFTER). Results p46, Discussion p66, Conclusion p72 (BEFORE). Chapter 6 grew by 4 pages.

## 4. Figures

### 4.1 Style consistency across the whole thesis (19 in-chapter figures)

House style (12): fig_ladder, fig_sft, fig_arms, fig_decoupling, fig_operating, fig_perproduct, fig_gallery, fig_explain, fig_iter2, fig_dynamics, overview_big (Fig 1.1), plus the trace-pipeline SVGs in Ch 3.
Default matplotlib (5): tsne_anomaly_types (Ch 3), per_product_iter2_visa (Ch 4/6), curve_sft_ba, curve_grpo_reward, curve_grpo_kl (App H).
Neither (2): rl_correction_pipeline (Fig 5.x, user-supplied), fig_limitations (Ch 7).

Chapter 6 alone: 10/10 house style. That was the user's requirement and it holds.

### 4.2 Chapter 6 figures, final Gemini verdicts at full resolution

| Figure | Verdict | Residual |
|---|---|---|
| fig_ladder | print-ready | none (delta rounding fixed) |
| fig_sft | print-ready | "Left/Right" wording in caption vs top/bottom layout. Caption fix only |
| fig_arms | print-ready | verified by eye at 22:09. Knockouts hold, legend present. A's "collision" was a stale render |
| fig_decoupling | print-ready | |
| fig_operating | print-ready | |
| fig_perproduct | print-ready | |
| fig_gallery | print-ready | header counts are low contrast, cosmetic |
| fig_explain | print-ready | |
| fig_iter2 | print-ready | "six products" corrected to five |
| fig_dynamics | needs a decision | duplicates all three Appendix H curves. Either drop the H curves or drop panel (a) to (c) here |

### 4.3 Figure 1.1 (overview_big.pdf, user-supplied)

Carries stale numbers. LLaVA SFT 85.65/68.13 is the vLLM-path result (`checkpoint-188/eval_*_vllm.json`). The thesis tables use the HF-path files, 85.91/68.26. LLaVA SFT+GRPO VisA 72.58 should be 72.38 (strict convention). Qwen base DS 69.01 should be 69.08. Three numbers, one figure, the first figure in the thesis. This is the highest-priority figure fix because a reader will cross-check Figure 1.1 against Table 6.1.

### 4.4 Truncated axes

Five Chapter 6 figures start the y-axis at 50 or higher. Both Gemini audits flagged it. BA = 50 is chance, so the floor is defensible, but none of the five captions says so. Add "the axis starts at chance (50)" to each caption. One clause each.

## 5. Section-by-section inconsistencies (verified against live .tex)

Severity: H = a number or claim a reader can catch. M = wording that reads as a contradiction. L = style.

| # | Where | Finding | Fix |
|---|---|---|---|
| 5.1 H | Abstract | conflates the 1.6 to 4.0 pp decoupling range with the 120-step probe | say "in a 120-step probe on the Arm-C init, every checkpoint lands 1.6 to 4.0 points below it" |
| 5.2 H | Abstract | LLaVA claim carries no corpus caveat | add "on VisA" or point to App M |
| 5.3 H | Intro §1.x, Ch 8 | cite base to SFT per-product numbers "in Chapter 6". They are now only in Appendix C | change the ref to App C |
| 5.4 H | Ch 8 | "Six concrete contributions" but items (1) to (7) | "Seven" |
| 5.5 H | Ch 8 | "beats 15K by 8.5 pp at matched epochs" then "80.16 against 72.60". 72.60 is the 15K best epoch (ep4). Matched epoch 3 is 71.66, which gives 8.50 | use 71.66 in that sentence, or say "7.56 pp best-vs-best" |
| 5.6 H | Ch 6 §6.14, Ch 8 | judge gaps. §6.11: smallest +2.91 DS / +1.36 VisA. §6.14: "2.91 and 1.48" (Qwen KCR's own). Ch 8: "weakest model at +2.91 and +1.36" mixes Qwen KCR (DS) with LLaVA SFT (VisA, 8.40 minus 7.04) | §6.14 and Ch 8: "the smallest gap on either benchmark is +2.91 and +1.36". Drop "weakest model" |
| 5.7 M | Ch 6 §6.13 vs fig_dynamics caption | prose "the four components evolve", caption "its two components". Ch 5 defines two reward functions, the accuracy one has three sub-signals | prose: "the reward and its two components" |
| 5.8 M | Ch 6 §6.13 | attributes the drop to "over-optimisation of the format reward at the expense of type and location rewards", but no type/location curve is shown | either add the curves or soften to "we conjecture" |
| 5.9 M | Ch 5 §5.8 arms | Arm B = 3,557 kept + 812 corrected = 4,369. The 1,631 weakly-grounded correct items are never said to be dropped from Arm B. A reader adds and asks where they went | one sentence: "Arm B discards the 1,631 correct-but-weakly-grounded items. Arm C keeps them after revision" |
| 5.10 M | Ch 5 §5.8 Arm A | "pass rate is approx 95%" next to "812 items failed all rollouts". 8,872/10,236 = 86.7% at item level | define the 95% (per-rollout, k=8) or replace with 86.7% item-level |
| 5.11 M | Intro | "single judge sample per trace". Ch 6 uses the median of three | "the median of three judge samples" |
| 5.12 M | Intro | IAD-R1 sentence: "most recent strong SFT+GRPO open-weight baseline ... a strong recent reference point". Doubled | drop the trailing clause |
| 5.13 M | Intro | backbone of IAD-R1 stated two ways | pick Qwen2.5-VL-7B, once |
| 5.14 M | Ch 3 | "we use the five camera viewpoints" vs "restricted to C1" | keep C1 |
| 5.15 M | Ch 4 | per-device batch 8 x accum 2 vs another batch statement. Stale "no epoch-3 value for 7B-frozen-15K" (it is 71.66, in Table 6.2) | align to Ch 6, delete the stale sentence |
| 5.16 M | Ch 7 §(b) | "At a fixed wall-clock budget of 4 epochs". Same wording the supervisor flagged in the intro. Epochs are not wall-clock | "Under the same four-epoch schedule" |
| 5.17 M | Ch 7 title | "Why Quality Dominates Quantity" but the section argues it is a count-and-composition effect, not a quality effect | retitle "Why 6K Beats 15K" or "Why Composition Dominates Count" |
| 5.18 M | App F | intro: "Arm-C ckpt-376". §C.3: "the 6K SFT checkpoint (ckpt-564)". Those are two different 6K corpora (zeroshot-6K ep3 vs Arm C ep2) | §C.3: "the earlier zeroshot-6K SFT checkpoint (ckpt-564)" |
| 5.19 M | App J | "greedy, zero run-to-run variance" vs "well within run-to-run reporting noise" | "well within the prompt-to-prompt spread" |
| 5.20 L | App K | scope "anomalous only" then rules for normal samples | "applies to all test samples. Normal samples drop the localisation axis" |
| 5.21 L | Ch 5, Ch 6 | "This reward rewards", "three arms ... three recipes ... same ... same", "two columns ... not two", "rate ... rate". Roadmap sentence in Ch 5 uses semicolons | reword |
| 5.22 L | Ch 6, 7, 8 | 8 sentences over 45 words, listed in B_chapters.txt | split |

Discarded Gemini findings (its knowledge cutoff): "Gemini 2.5 Flash / Gemini 3 Flash / GPT-5-mini do not exist". They do. "42.9% violates two-decimal rule": the rule is for balanced accuracy, and 42.86 rounds to 42.9. "Location reward presence vs loc-tag correctness": consistent on reading.

Number scan: every balanced-accuracy number in Ch 6 traces to an eval file. D_claims found 5 untraceable claims, all five verified by hand. 0 uncited bib entries (all 92 cited). 0 undefined references in the AFTER build.

## 6. Rubric mapping (E_rubric, Gemini on the AFTER build)

| Criterion | Band | Evidence used | Single action to move up |
|---|---|---|---|
| 1a Research method | 8 | four-factor grid, single harness | multi-seed error bars (L1) |
| 1b Theory | 8 | STaR/self-critique grounding, GRPO formulation | formal decoupling condition |
| 1c Interpretation | 9 | decoupling diagnosis, se/sp trade-off reading | mechanistic probe of ViT freezing |
| 1d Significance | 9 | beats IAD-R1 under one harness, open release | venue paper or industrial pilot |
| 3a Content | 9 | error taxonomy, contamination disclosure | human inspector study (L2) |
| 3b Form | 9 | ladder figure, dev/test split in tables | in-text qualitative panels |
| 3c Writing | 9 | abstract, motivation | remove `\newline\newline` artefacts |

Overall 8.5. Weakest 1a. The cheapest defensible move on 1a before the defence is not new seeds. It is the existing two-seed GRPO pair plus the prompt-robustness spread in App J, stated in one paragraph of §6.1 as the noise floor. That already exists in the thesis, it is just not framed as an error estimate.

## 7. Peer lessons (G_peers, six theses)

Verdict per section: abstract ON PAR, intro/method/results/discussion/conclusion STRONGER.

The five "borrow" items, in order of value for the defence:
1. Results: yangli reports mean plus or minus sd. We cannot add seeds, but §6.1 can state the noise floor (see 6 above).
2. Abstract: linuesa closes with an interpretive scope sentence. Ours closes on probe numbers. Swap the 1.6 to 4.0 / 120-step clause for one sentence on what the result means for practitioners. This also fixes 5.1.
3. Intro: vaessen groups the RQs. Ours splits RQ1 from RQ2 to 4 by Figure 1.1. Move the figure below the four RQs.
4. Method: linuesa has a chapter-contained pipeline schematic. Ours back-references "Panel (B) of Figure 4.1". fig:rollouts-big already covers this. Fix the back-reference text only.
5. Discussion: linuesa has a "barriers to implementation" subsection. Ours has the material scattered under "Concrete remediation". A two-paragraph "Deployment barriers" subsection would collect it.

## 8. Pending and untouched

- Corrected LLaVA run at 477/748 (22:13), ckpt-564 about 00:05, ckpt-748 about 03:20, evals about an hour after each. §6.9 and App M get their final rows then.
- `07_discussion.tex:31` not touched, per instruction.
- No Overleaf push. No HF upload. No thesis edits in this audit.
- Optional: the 5 matplotlib figures outside Ch 6, the fig_dynamics/App H duplication decision, Figure 1.1 regeneration by the user.
