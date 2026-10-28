# Chunk 19 — chunk_19.txt

## 1 Counts
- **Total Traces**: 25 (all product: `terminalblock`).
- **Split**: `grpo_4k`: 25.
- **Category**: `wrong`: 22, `completely_wrong`: 3 (`...S0040...`, `...S0143...`, `...S0145...`).
- **Issue Types**:
  - `count_error`: 38
  - `location_error`: 19
  - `invented_defect`: 9
  - `feature_absent`: 7
  - `viewpoint_error`: 4
  - `defect_type_error`: 2
  - `verdict_mismatch`: 2

## 2 Generator failure modes
1. **Pervasive terminal miscounting (5 vs. 6)**: The generator persistently hallucinated 5 terminals/screws on a standard 6-pole terminal block across ~18 traces.
   - Claim: `"I can see five screw terminals here."` (`...S0113...`)
   - Judge why-wrong: `"There are six screw terminals visible along the front face."`
2. **Substrate transposition & invented defect sites**: Traces fabricated contamination on metal terminals/bores when contamination actually sat on the orange plastic housing (or vice versa) in ~7 traces.
   - Claim: `"In both of these connection points, I can observe a dark, foreign substance adhered to the metallic terminals."` (`...S0097...`)
   - Judge why-wrong: `"The dark contamination is located on the orange plastic body along the central step/ridge above the third screw, not adhered to the metallic terminals inside the openings."`
3. **Severe defect hallucination on pristine samples**: Generator hallucinated gross damage, contamination, or missing pieces on nominal items in 2 traces.
   - Claim: `"The plastic housing in this area appears to be broken, and the metal terminal component is partially dislodged or missing."` (`...S0145...`)
   - Judge why-wrong: `"The third terminal and its surrounding plastic housing are fully intact with no breakage or missing parts."`
4. **Orientation and feature transposition**: Confusing horizontal terminal rows with vertical stacks or fabricating dual rows in ~4 traces.
   - Claim: `"There are six such openings arranged vertically."` (`...S0014...`)
   - Judge why-wrong: `"The six openings are arranged horizontally in a row, not vertically."`

## 3 Judge mistakes
The judge was mostly accurate on terminalblock geometry, but exhibited occasional pedantry or slight inconsistencies:
1. `...S0168...`: Trace states the defect is `"just above the leftmost screw head and slightly to its left"`, which judge flags because the defect is above the second screw from the left. However, the judge's rewrite leaves `<location>center</location>` intact despite the text placing it in the 2nd position (middle-left).
2. `...S0181...`: Claim `"The mounting clips on the left side seem robust"` was flagged because tabs exist across the upper section. On perspective/side-angle views of modular terminal blocks, interlocking dovetails often sit specifically on the left face; penalizing this as `location_error` without disambiguating side tabs vs top latch tabs is overly strict.

## 4 Rewrite quality
Rewrites effectively repair factual counts and substrate confusions while preserving gold labels/verdicts. However, rewrites occasionally leave slight internal tensions:
- **Clean repair (`...S0040...`)**: Fixes count (5 to 6), moves contamination claim from terminal 4/5 metal screws to the top-right housing edge, cleanly aligning reasoning with `<location>top-right</location>`.
- **Incomplete trace cleanup (`...S0045...`)**: Changed the analyzed screw from the 4th to 5th screw, corrected location from `center` to `middle-right`, and successfully aligned the metadata tags without inventing new unsupported content.

## 5 Risks for a thesis that publishes this corpus
1. **Verdict mismatches on nominal parts**: `...S0143...` and `...S0145...` were gold-label `no` (normal), but generator outputs concluded `Yes` with hallucinated defects ("Missing Parts", "Contamination"). If used for GRPO/RL fine-tuning, positive reward on misaligned verdicts causes catastrophic policy degeneration.
2. **Overlay/Prior Grounding Leaks**: In `...S0040...`, the generator declared `<location>top-right</location>` in the output tag despite its internal thought trace concluding the defect was on `"the fourth and fifth terminals from the left"`. This indicates strong target-label memorization/prior leakage where the final tag matches gold metadata rather than the model's actual reasoning.
3. **Systematic Count Hallucinations**: 18 of 25 traces hallucinate a 5-pin block instead of a 6-pin block. Training on ungrounded traces solidifies blind counting priors.
