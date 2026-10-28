# Chunk 20 — chunk_20.txt

## 1 Counts
- **Total Traces**: 39
- **Split Breakdown**: `grpo_4k`: 19, `sft_6k`: 20
- **Category Distribution**: `wrong`: 33, `completely_wrong`: 6
- **Issue Type Distribution**:
  - `defect_type_error`: 42
  - `count_error`: 39
  - `location_error`: 23
  - `viewpoint_error`: 10
  - `overlay_leak`: 9
  - `invented_defect`: 5
  - `colour_finish_error`: 3
  - `feature_absent`: 3
  - `marking_text_error`: 2
  - `missed_defect`: 1
  - `other`: 2

## 2 Generator failure modes
1. **Systematic Terminal Block Miscounting**: Under-counting terminal positions/screws (hallucinating 5 or 7 instead of 6) across 11 traces.
   - Claim: `"I can see a row of five screw terminals on the front side of the block."` (`terminalblock_OK_S0279_...`)
   - Judge: `"There are six screw terminals clearly visible in a row, not five."`
2. **Defect Type & Morphological Hallucination**: Misinterpreting fine surface defects (e.g., calling pits/missing parts "stains", "flash", or "fraying") across 17 traces.
   - Claim: `"This is a classic case of flash—excess plastic that leaked between the mold halves..."` (`terminalblock_NG_QS_S0163_...`)
   - Judge: `"The defect in the top-right corner is broken/missing plastic (missing parts/chip damage), not excess plastic or flash."`
3. **Red Mask Overlay Leaks**: Hallucinating red or pink coloration onto foreign particles caused by leakage from ground-truth inspection overlays across 7 traces.
   - Claim: `"there are several bright, reddish-pink clumps of material caught between the bristles."` (`toothbrush_NG_ZW_S0088_...`)
   - Judge: `"The foreign material in the query image is white/translucent, not reddish-pink; the trace is describing the red overlay mask."`
4. **Spatial Orientation and Layout Inversion**: Inverting coordinates, labeling bottom text as "top/upper-left", or misreading rows as vertical columns across 13 traces.
   - Claim: `"molded text/symbols in the upper-left area."` (`toy_NG_AK_S0014_...`)
   - Judge: `"The molded CCC logo and text are located at the bottom-center/bottom area in this query image."`

## 3 Judge mistakes
1. **`toothbrush_NG_QS_S0114_...`**: The judge flags `<type>missing parts</type>` as a defect type error and replaces it with `<type>bent bristles</type>`. The gold dataset label is strictly `gold_type=Missing Parts`; the judge erroneously forces an out-of-ontology label.
2. **`toothbrush_NG_ZW_S0004_...`**: Over-strict location correction. The judge claims the lower mark is strictly `middle-right`, penalizing the generator's `"bottom-center"` description despite the defect being adjacent to the lower neck boundary.

## 4 Rewrite quality
Rewrites consistently correct specific factual claims without altering final `<answer>` verdicts. However, they occasionally patch descriptions awkwardly or alter formatting syntax:
- *Clean patch*: In `terminalblock_OK_S0344_...`, the rewrite smoothly swaps incorrect terminal counts from 7 to 6 throughout the trace while preserving structure and the `No` verdict.
- *Problematic patch*: In `toothbrush_NG_QS_S0114_...`, the rewrite introduces the unsupported non-standard label `<type>bent bristles</type>`, clashing with the ground truth taxonomy.

## 5 Risks for a thesis that publishes this corpus
- **Overlay Leaks**: Traces directly reference red annotation artifacts instead of actual image pixels (`toothbrush_NG_ZW_S0012_...`, `toothbrush_NG_ZW_S0018_...`, `toothbrush_NG_ZW_S0055_...`, `toothbrush_NG_ZW_S0088_...`, `toothbrush_NG_ZW_S0096_...`, `toothbrush_NG_ZW_S0125_...`, `toothbrush_NG_ZW_S0134_...`, `toothbrush_NG_QS_S0136_...`).
- **Invented Defects & False Grounding**: Trace hallucinates an entirely fictional mechanism (`toothbrush_NG_ZW_S0031_...` invents an `"exposed staple or anchoring wire"`; `terminalblock_NG_QS_S0053_...` invents a `"vertical white mark or scuff"`).
- **Taxonomy Mismatches**: Severe disagreement between generated `<type>` and gold annotations (`toothbrush_NG_CH_S0017_...` calls bristle abrasion a `Chip`; `terminalblock_NG_QS_S0018_...` calls missing chunks a `Crack`).
