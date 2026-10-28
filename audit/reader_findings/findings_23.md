# Chunk 23 — chunk_23.txt

## 1 Counts
- Total traces: 45
- Split: `grpo_4k`: 16, `sft_6k`: 29
- Product: `toy_brick`: 23, `transistor1`: 22
- Category: `wrong`: 40, `completely_wrong`: 5
- Issue types: `viewpoint_error`: 31, `feature_absent`: 16, `colour_finish_error`: 10, `defect_type_error`: 22, `count_error`: 27, `location_error`: 13, `overlay_leak`: 4, `invented_defect`: 2, `missed_defect`: 2, `verdict_mismatch`: 1

## 2 Generator failure modes
1. Hallucinating 3D isometric perspectives on planar 2D top-down images (~15 traces).
   - Claim: "The view captures two adjacent faces of the block: one side face and the top face." (`toy_brick_OK_S0127`)
   - Judge why-wrong: "The image captures a direct top/head-on view of a single square face of the wooden brick."
2. Miscounting pin counts on standard IC packages (~17 traces).
   - Claim: "I'm inspecting an integrated circuit (IC) package with 14 metal leads, seven on each side." (`transistor1_NG_BX_S0132`)
   - Judge why-wrong: "The package has 16 leads in total, 8 on each side."
3. Defect class confusion, especially mistaking broken/missing pins or voids as bending, flash, or chips (~12 traces).
   - Claim: "There is a small piece of extra metal or a sliver of solder attached to the side of the lead... This type of burr or excess material is a clear assembly defect." (`transistor1_NG_QS_S0071`)
   - Judge why-wrong: "The defect is a missing/broken lead (only a short broken stub remains), not an extra metallic burr or protrusion."
4. Ground-truth overlay leakage, hallucinating red/pink coloration from segmentation masks (~4 traces).
   - Claim: "In the middle-left area, there is a small, distinct red mark that stands out against the natural tan color of the wood." (`toy_brick_NG_HS_S0037`)
   - Judge why-wrong: "The red mark is the annotation mask in the overlay image, not a feature of the physical query sample."

## 3 Judge mistakes
1. `transistor1_OK_S0022`: The judge flagged "fourteen metallic leads" and rewrote it to "sixteen metallic leads" ("The IC package has 16 leads in total... not fourteen"), yet all other identical `transistor1` components in the batch are 18-lead SOIC packages (9 leads per side).
2. `transistor1_NG_QS_S0132`: The judge claims "There are only 7 pins on the top edge because the eighth pin on the top right is missing" and updates the total to "seven pins on the top and eight on the bottom", which contradicts the standard 9-pins-per-side (18 total) baseline seen across all other samples of this product class.
3. `toy_brick_NG_BX_S0063`: The judge flagged pin count claiming 18 total leads (9 on each side), yet the second sub-issue re-quotes the generator's count claim without identifying any error, stating: "On the bottom row of 9 leads, the bent leads are the 7th and 8th leads from the left... SHOULD BE: The seventh and eighth leads...".

## 4 Rewrite quality
Rewrites successfully preserve gold verdicts while scrubbing invalid claims, but occasionally introduce unsupported assumptions or retain flawed rationale:
- Successful scrub: In `toy_brick_NG_HS_S0037`, the rewrite successfully removes the red overlay artifact ("small, distinct red mark... ink or paint") and replaces it cleanly with mechanical damage ("distinct indentation scratch that disrupts the natural surface texture").
- Unsupported introduction: In `toy_brick_OK_S0080`, the rewrite removes the fictitious top face but introduces unprompted assertions about "natural wood checks near the top edge" to explain texture variations without confirming if checks are actually present.

## 5 Risks for a thesis that publishes this corpus
- Overlay leaks: Multiple traces directly describe the evaluation mask colors (`toy_brick_NG_HS_S0037`, `toy_brick_NG_ZW_S0034`, `toy_brick_NG_ZW_S0064`, `transistor1_NG_ZW_S0004`), proving contamination from ground-truth mask overlays.
- Invented defects on normal items: In `toy_brick_NG_ZW_S0026`, the generator hallucinates a scratch defect on a defect-free part, requiring complete narrative inversion.
- Label and verdict mismatch: `toy_brick_NG_ZW_S0026` produced a `<answer>Yes</answer>` verdict against a gold label of `no`.
- Ground-truth misclassification: Traces like `transistor1_NG_QS_S0002` and `transistor1_NG_QS_S0016` show the generator defaulting to geometric "bending" instead of recognizing severed or missing components, indicating weak semantic alignment with standard anomaly taxonomies.
