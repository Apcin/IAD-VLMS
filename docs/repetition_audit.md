# Thesis Repetition Audit & Dedup Log
*Created 2026-06-21. Source: multi-agent workflow `w3cs982hp` (9 chapter scanners + 1 appendix sweep + 1 synthesis; 11 agents). Verified by hand against the Overleaf source before applying — the workflow over-counted in places (corrections noted below).*

## Method
Each chapter/appendix was scanned for (a) every place that defines/describes **Real-IAD** or claims it is "the largest publicly available" dataset, and (b) any other claim/definition/framing likely repeated elsewhere. A synthesis agent built a cross-thesis repetition map. **Dedup rule:** the full definition/derivation/numbers live in **one canonical section**; every other mention becomes a *named cross-reference* with no restated facts. Exception — the thesis statement and the Arm-C/GRPO headline legitimately recur at abstract→intro→conclusion; there we only make wording consistent, never delete.

## Real-IAD (the original question)
- **Described ~22 times** across the thesis (most are fine — bare `\realiad` uses).
- **"largest publicly available" superlative: 2 instances** (intro §1.3 l.30, §3.1 l.11) — **NOT 4** (the workflow mis-counted; §2.6 uses the *quantified* "order of magnitude larger than MVTec, AUROC 99%→85%" framing, and Appendix I l.49 carries the stat-block as a legitimate datasheet entry — neither says "largest").
- **⚠️ The superlative is also questionable:** Real-IAD Variety (cited in this thesis, §2.1/§2.6) is *larger* (160 categories, 198,950 images), so "Real-IAD is the largest publicly available" is arguably false now. Dropping it is doubly justified.
- **Canonical home:** §2.6 (`sec:bg-mmad`) keeps the full stat-block + quantified size. §3.1 keeps only the two properties it actually uses (C1–C5 viewpoints, defect masks). Appendix I datasheet keeps the stat-block as the formal dataset card.

## Applied — Round 1 (commit `c17399f`, `58c3859`, and earlier)
| Edit | File | Change |
|---|---|---|
| Real-IAD superlative → §2.6 ref | `01_introduction.tex` l.30 | dropped "the largest publicly available industrial dataset with per-defect masks" → "(§2.6)"; also de-dashed |
| Real-IAD superlative + stat-block → §2.6 ref | `03_dataset_traces.tex` l.11 | dropped "largest…" + "150K+ images / 30 categories"; kept C1–C5 + masks (used locally); back-ref §2.6 |
| 137-word "same-length" clause collapsed | `00_abstract.tex` ¶4(iv) | "…of which it is a same-length (≈137-word) subset… a data-count/composition effect rather than a trace-length one" → "(a data-count effect, not trace length; Ch.6)" |
| (earlier) §2.6 "two datasets" reframe | `02_background.tex` | three benchmarks → two datasets, held-out kept |
| (earlier) §3.1/intro Defect-Spectrum cite, Real-IAD-Variety verified | Ch.2/Ch.3 | — |

## Repetition map — remaining (apply as we walk each chapter)
Worst offenders first. "Keep full in X" = canonical home; everywhere else → one-clause cross-reference.

| Claim | × | Keep full in | Collapse elsewhere |
|---|---|---|---|
| "~137-word same-length subset → count not length" | 7 | §6 (proof) + §7 (interpretation) | abstract ✅done; intro Contrib-2, ch3, ch4, ch8 → "(same-length subset; §6)" |
| Six-phase template **enumerated** (Framing…Decide) | 8 | §3.2 + Appendix B (verbatim) | abstract, ch1, ch2 l.61, ch4, ch7, ch8 → "the six-phase template (§3.2)" |
| GRPO reward decomposition (4 sub-signals, max 3.0) | 8 | §5 (`grpo-rewards`) + Appendix E | **Appendix alone restates it 4× (E/K/H/A) — consolidate**; narrative chapters → "(four sub-signals, max 3.0; §5)" |
| Quality>quantity (80.16 vs 71.66, 8.5pp) | 8 | §6 l.84 | abstract ✅; ch1, ch3, ch4, ch8 → headline + "(§6)" |
| Frozen-encoder-wins | 7 | §6 l.86 + §7 (asymmetry analysis) | abstract, ch1, ch4, ch8 → one clause. **ch4 repeats internally — verify lines** |
| GRPO double-edged (+lift / −2pp) | 7 | §6 + §7 | abstract, ch1, ch4, ch8 → one clause (headline recurrence OK) |
| β=0 KL config (k3, monitored-not-in-loss) | 7 | §5 l.27 + Appendix A table | ch5 has prose(l27)+table(l101) — keep both; others → "β=0 (KL monitored; §5)" |
| DS-MVTec(1,670)/VisA(2,141) definition | 6 | §2.6 l.138-141 | later chapters use the names as settled terms, no re-explain |
| Held-out Real-IAD "sample-level not product-disjoint" caveat | 6 | §3.1 / §6 (result) | repeated across Appendix I/J/A — consolidate |
| IAD-R1 "closest baseline" framing | 6 | §2.5 l.122 | "the IAD-R1 baseline (§2.5)" elsewhere |
| Common-harness "re-run not quoted" framing | 4 | §6 (`res-sota`) | ch2/ch1/ch7 → "(§6)" |
| AnomalyThink corpus identity (14,472 / 6K+4K+4K) | 6 | §3 intro ✅ | abstract, ch1, ch4, ch8 → "(§3)" |
| AnomalyThink novelty ("first openly-released…") | 4 | §3.7 | ch1, ch8, appendix → cite §3.7 |
| Commodity 2×A6000 / ZeRO-3 / ~300GB | 6 | §4 l.64 + Appendix A | others → "commodity 2×A6000" only |
| MVTec ~99% saturation | 3 | §2.x l.27 | abstract, ch1 → one clause |
| GPT-4o 74.9% on MMAD | 2 | §2.6 l.138 | ch1 → cited clause |
| OpenVLThinker iter-2 (80.64 vs 82.73) | 6 | §6 l.167 | ch1/ch8 → "(§6)" |
| G²RPO result (81.94 vs 82.73) | 4 | §6 l.141/186 | ch2/ch5/ch7 cross-ref |
| Per-product gains (metal_nut +25.8 …) | 4 | §6 l.391 | abstract/ch1/ch8 → 2 exemplars + "(§6)" |

## Corrections to the workflow's claims (verify before trusting line numbers)
- "largest publicly available" = **2×, not 4×** (see above).
- The flagged **ch4 intra-chapter dupes** (frozen l.38+l.137, LoRA l.31+l.165, pixel-cap l.29+l.150) **did not hold up**: l.137 is the 2×2×2×4 factorial-count explanation, l.150 is the inference-inputs paragraph, l.165 is a legitimate 3-feature chapter summary. The agents' line numbers drift — **re-check each instance against the source before editing**.

## Status
Round 1 applied (Real-IAD superlative + abstract 137-word clause). The rest is best applied **as we walk each chapter** in the paragraph review (dedup Ch.4 when we reach it, etc.), since that's when the surrounding text is already open and being edited.
