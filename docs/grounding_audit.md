# What the grounding audit actually found

Source: 28 Gemini 3.8 Flash readers over `audit_flagged_ALL.txt` (2,364 flagged traces),
cross-checked against the raw records in `Training/trace_audit_gemini/audit_grounding.jsonl`.
Every number below is recomputed from the jsonl, not taken from the readers.

Scope: this audits `combined_6k_train.json` (the SFT corpus, Arm A) and
`grpo_train.json`. It does NOT cover the Arm B and Arm C corpora, which are
model rollouts plus teacher correction and were built separately.

## The one-line answer

The labels are clean. The reasoning is not.

| split | traces | flagged | rate |
|---|---|---|---|
| all | 10,236 | 2,364 | 23.1% |
| sft_6k | 6,000 | 958 | 16.0% |
| grpo_4k | 4,236 | 1,406 | 33.2% |

Of the 958 flagged SFT traces, **zero** have a verdict mismatch. All 958 reach the
correct yes/no answer through at least one ungrounded statement. Across the whole
10,236, only 29 traces disagree with the gold label. So 99.7% of the corpus carries
the right supervision signal on the label and 23.1% carries a wrong statement inside
the reasoning that produced it.

## Traces with at least one issue of each type

| issue type | traces | share |
|---|---|---|
| location_error | 841 | 8.22% |
| defect_type_error | 703 | 6.87% |
| feature_absent | 384 | 3.75% |
| count_error | 297 | 2.90% |
| viewpoint_error | 294 | 2.87% |
| invented_defect | 264 | 2.58% |
| colour_finish_error | 186 | 1.82% |
| overlay_leak | 185 | 1.81% |
| marking_text_error | 121 | 1.18% |
| missed_defect | 44 | 0.43% |
| verdict_mismatch | 29 | 0.28% |
| not_visible_in_view | 8 | 0.08% |

## The five biggest problems, ranked

### 1. Overlay leak. The only finding that threatens the method, not just the data.

185 traces describe the red ground-truth mask as a physical feature of the part.
That is 3.6% of the 5,118 anomalous traces, and 180 of the 185 sit in the SFT 6K split.

Typical case, `button_battery_NG_AK_S0027`:

> generator: "there is a distinct red speck of contamination. This small, bright red
> particle is sitting on the surface"
> judge: "The red dot is the defect annotation mask from the overlay image, not a
> physical red particle on the battery."

This matters more than its 1.81% share suggests. The teacher was shown the overlay,
so the trace is grounded in the annotation rather than the part. A student trained on
those traces learns that anomalies are red. At inference there is no overlay. All 28
readers put this in their top three findings independently. It is also the one a
reviewer can find from the generation script alone, because the script visibly sends
the overlay image.

### 2. Spatial grounding is the largest single class.

841 traces, 8.2%. Axis flips, mirrored left and right, and defaulting to "center".

> `pcb_NG_HS_S0019`: "In the bottom section, I see connection points B+, B-, and a
> circular pad." The judge: those are at the top.

Caveat: on rotationally symmetric parts (rolled_strip_base, mounts, plastic_nut) the
judge gives contradictory orientation rules across identical parts. Part of this
count is genuine ambiguity, not generator error. Treat 841 as an upper bound.

### 3. Defect polarity inversion.

703 traces have a defect type error, and the dominant pattern is a sign flip. Missing
material is called excess flash, and chips are called protrusions.

> `regulator_NG_QS_S0022`: "the plastic material has overflowed the intended boundary
> of the rim. This extra material, known as flash..."
> judge: "The defect is missing plastic / a missing part, not excess plastic/flash."

Named by almost every reader. Systematic, not random noise.

### 4. Viewpoint hallucination.

294 traces. The teacher invents side walls, helical threads, circumferential grooves
and stepped 3D geometry on flat top-down images. Heavily concentrated in fire_hood,
plastic_plug, end_cap and mint. Chunk 10 estimates over 40% of its GRPO subset.

### 5. Product-specific memorised priors.

The teacher applies a canonical template for a part class regardless of the image.
Chunk 14 is the clearest case: all 24 audited `sim_card_set` traces hallucinate an
ejector pinhole that is not visible. Others: transistor pin counts, `CR2032` read as
`CB5035` or `MUHTIU` on button batteries, and regulator ribs counted as three when
there are four.

## Where the judge itself is unreliable

The issue counts are teacher-judge flags, not verified ground truth. Four recurring
judge failures, each seen by three or more independent readers:

- **Vocabulary policing.** The judge marks a trace wrong when the physical description
  is right but does not match the dataset's gold category string. It flags `Crack` as
  wrong for a gold label of `Missing Parts` while conceding in its own rationale that
  material is broken away. Chunks 5, 9, 13, 17, 20, 24. `defect_type_error` = 703 is an
  upper bound.
- **Sub-quadrant pedantry.** Flags bottom-left against bottom-center on masks that
  straddle the boundary. Chunks 5, 11, 19, 26. `location_error` = 841 is an upper bound.
- **Self-contradiction on counts.** For `transistor1` the judge asserts 16 leads on one
  trace, 20 on another and corrects 14 to 16 on a third, on the same component. Chunks
  23 and 24. `count_error` = 510 is partly the judge's own error.
- **Overlay colour injected into its own fix.** It flags an overlay leak and then writes
  "red contamination" into the correction. Chunks 2, 13, 25, 26.

## Are the 2,353 rewrites safe to use

Mostly yes, with one hard bug.

- Every non-empty rewrite retains a parseable `<answer>` tag. Zero of 2,353 lost it.
- Verdict preservation is consistent. Rewrites correct the description without flipping
  the label, except where the flip is the correction.
- **Hard bug: 11 traces are marked wrong or completely wrong with a completely empty
  rewrite.** 8 in sft_6k, 3 in grpo_4k. These must not enter any corpus:
  `end_cap_NG_ZW_S0075`, `end_cap_NG_ZW_S0113`, `eraser_NG_QS_S0084`,
  `bottle_cap_NG_ZW_S0103`, `toy_NG_QS_S0099`, `toy_NG_QS_S0113`, `pcb_NG_QS_S0083`,
  `mint_NG_ZW_S0078`, `usb_NG_ZW_S0033`, `usb_adaptor_NG_AK_S0051`,
  `sim_card_set_NG_ZW_S0064`.
- 241 individual issues carry a blank "what it should be". 122 are `feature_absent` and
  62 are `viewpoint_error`, where deletion is the correct fix, so those are benign.
- Two soft defects. The rewrite sometimes injects a new specific visual claim that is
  not verifiable from the prompt (chunks 4, 8, 10, 15, 16, 17, 18). And it sometimes
  patches one sentence while leaving a contradicting sentence further down the trace
  (chunks 11, 12, 17, 21).

## What is risky to publish as-is

1. The 11 empty rewrites. A parsing bug for anyone who downloads the corpus.
2. The 185 overlay-leak traces. Disclose the count and the cause.
3. The 8 `not_visible_in_view` traces. Real-IAD labels a sample from multiple camera
   views, and the label does not always apply to the single view that was sent. There
   the correct answer is unreachable from the image, so the trace has to hallucinate.
4. The 317 `completely_wrong` traces. 176 in GRPO, 141 in SFT.

## Recommended framing for the thesis

The audit is a strength. No comparable industrial-anomaly reasoning corpus reports a
per-trace grounding audit at all. Report it as: 99.7% of traces carry the correct
verdict, 23.1% contain at least one ungrounded statement in the reasoning, and the
leading failure is spatial grounding. State the overlay leak explicitly as a
limitation with its count. State that the issue-type counts are judge flags with a
known over-flagging tendency on the type and location categories, and do not present
them as verified error rates.
