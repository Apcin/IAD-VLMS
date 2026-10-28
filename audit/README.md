# Grounding audit of the AnomalyThink reasoning traces

Every one of the 10,236 training traces was re-checked against its own images by an
independent judge, and the flagged ones were corrected in place. This directory holds
the judge, its prompt, the full per-trace record, and the builder that turns those
records into a corrected corpus.

## Headline

| split | traces | flagged | corrections available | unfixable |
|---|---|---|---|---|
| SFT 6K (`combined_6k_train.json`) | 6,000 | 958 (16.0%) | 950 | 8 |
| GRPO 4K (`grpo_train.json`) | 4,236 | 1,406 (33.2%) | 1,403 | 3 |
| **total** | **10,236** | **2,364 (23.1%)** | **2,353** | **11** |

Only 29 traces in the whole corpus disagree with the gold label. Of the 958 flagged
SFT traces, zero do. So the binary supervision is 99.7% consistent and the damage is
inside the reasoning, not the label.

Traces carrying at least one issue of each type, over all 10,236:

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

Read `../docs/grounding_audit.md` for the full analysis, including the failure modes
and the places where the judge itself is unreliable.

## How the audit works

Judge: `gemini-3.7-flash`. For each anomalous item it receives three images, the clean
query image, an overlay with the ground-truth mask in red, and a reference normal of
the same product and camera view. Normal items receive the query image only. It is
shown the original `gemini-2.5-flash` trace and asked to flag every claim the images
do not support, then to correct the trace **in place** rather than rewrite it.

The prompt enforces minimal editing: same voice, same tag structure, within ten percent
of the original word count, and it must never mention the overlay. Measured over the
950 SFT corrections, the median rewrite/original word ratio is 0.977 and 78.6% land
within ten percent. All 950 carry an XML tag skeleton identical to the original.

## Caveat, and it matters

These are judge flags, not verified ground truth. The judge over-flags in two known
ways. It marks a physically correct description wrong when the wording does not match
the dataset's gold category string, which inflates `defect_type_error`. And it splits
`bottom-left` against `bottom-center` on masks that straddle the boundary, which
inflates `location_error`. Treat both counts as upper bounds. It also contradicts
itself on pin counts for `transistor1` across different traces of the same component,
so part of `count_error` is the judge's own error.

Do not cite these numbers as measured error rates. Cite them as flag rates from an
automated judge with a documented over-flagging tendency.

## The overlay leak is the finding that matters

185 traces describe the red ground-truth mask as a physical feature of the part, and
180 of them are in the SFT split. The cause is visible in the generation script: the
teacher was shown the overlay. A student trained on those traces learns that anomalies
are red, and at inference there is no overlay. This is disclosed as a limitation in
the thesis.

## Files

| file | what |
|---|---|
| `audit_traces_gemini.py` | the judge. Credentials come from `GEMINI_API_KEY` or `VERTEX_SA_JSON`, never from disk in this repo |
| `prompt_grounding_v2.txt` | the judge prompt, including the minimal-edit rules |
| `build_corpus.py` | flattens the two training JSONs into the audit's input format |
| `audit_grounding.jsonl` | **all 10,236 records**: category, per-issue claim / why-wrong / what-it-should-be, and the corrected trace |
| `build_anomalythink_v2.py` | applies the SFT-split corrections to produce AnomalyThink-v2 |
| `reader_findings/` | 28 independent qualitative summaries of the flagged traces, one per shard |

## Reproducing

```bash
python build_corpus.py                       # writes corpus.jsonl
GEMINI_API_KEY=... python audit_traces_gemini.py --out audit_grounding.jsonl
python build_anomalythink_v2.py              # writes ../traces/anomalythink_v2/
```

Image paths inside these records are absolute TU Delft cluster paths. Use
`../traces/rewrite_image_paths.py` to remap them.

## What is assembled, and what is not

`audit_grounding.jsonl` carries the corrections for **all** 10,236 traces, both splits.
Only the 6,000-item SFT split is pre-assembled into a corrected corpus, as
`../traces/anomalythink_v2/`. The 1,403 GRPO-split corrections are present in the
records but are not assembled, because that split is a rollout image pool rather than
a supervised training set. Assemble it with the same builder if you need it.
