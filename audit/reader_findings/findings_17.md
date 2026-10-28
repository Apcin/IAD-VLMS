# Chunk 17 — chunk_17.txt

## 1 Counts
* **Total traces:** 38
* **By split:** `grpo_4k`: 21 | `sft_6k`: 17
* **By category:** `wrong`: 29 | `completely_wrong`: 9
* **Issue type distribution (total = 67):**
  * `defect_type_error`: 24
  * `location_error`: 22
  * `count_error`: 10
  * `viewpoint_error`: 10
  * `invented_defect`: 6
  * `overlay_leak`: 6
  * `feature_absent`: 5
  * `colour_finish_error`: 2
  * `verdict_mismatch`: 1

## 2 Generator failure modes
* **Defect Type & Morphological Inversion** (~14 traces): Misinterprets negative-space defects (missing plastic/notches) as positive excess (flash/burr/stains).
  * *Claim:* `"There's a triangular piece of extra black plastic sticking out from the side... This looks like molding flash"` (`switch_NG_QS_S0011`)
  * *Judge:* `"The feature is not excess flash/material; rather a retention tab/wing or missing plastic/breakage issue"`
* **Hardware Count Hallucination** (~8 traces): Systematically hallucinates five terminal screws on 6-position blocks (or four terminals on switches).
  * *Claim:* `"There are five visible screw terminals on the front face."` (`terminalblock_NG_AK_S0032`)
  * *Judge:* `"There are 6 screw terminals visible on the front face."`
* **Defect Substrate & Location Confusion** (~12 traces): Relocates housing pits onto metal contact surfaces or screw heads.
  * *Claim:* `"Yes, there are definitely small pits on the surface of both of these metal terminals."` (`terminalblock_NG_AK_S0009`)
  * *Judge:* `"The pit is located on the orange plastic housing, not on the metal terminals."`
* **Ground-Truth Overlay Mask Leakage** (~6 traces): Describes red annotation masks on dark/clear objects as real red stains or foreign plastic.
  * *Claim:* `"This is a red-colored scratch or abrasion that appears to have transferred material onto the black plastic."` (`switch_NG_HS_S0012`)
  * *Judge:* `"The scratch on the query image is not red; the red color is from the ground-truth defect mask overlay."`

## 3 Judge mistakes
* **Over-strict naming on non-standard vocabulary (`switch_NG_ZW_S0082`):** Flagged `defect_type_error` on `<type>Poor Finish</type>` when the gold was `Contamination`, despite the generator plausibly rationalizing the anomalous patch.
* **Semantic pedantry on defect description (`switch_NG_QS_S0001`):** Flagged a peeling rim edge as missing part breakage rather than flash, even though irregular plastic edges from injection fractures blur between burr/breakage.
* **Over-correcting viewpoint descriptors (`tape_NG_PS_S0068`):** Flagged `"three-quarter view"` as an error requiring `"axial/top-down view"` where perspective tilting is ambiguous in 2D projection.

## 4 Rewrite quality
* **Fixes claim without altering verdict:** Generally preserves logic while correcting specific hallucinations (e.g., in `switch_NG_HS_S0019`, safely strips `"diagonal red line"` to `"diagonal scratch mark"`).
* **Fixes issue but leaves internal contradictions:** In `switch_OK_S0495`, the rewrite corrects the opening to `"presents a side profile view"`, but leaves the downstream original line untouched: `"Since the view is from the front, the terminals... are largely obscured."`
* **Introduces newly invented descriptions:** In `terminalblock_NG_AK_S0078`, the rewrite inserts an ungrounded inspection narrative: `"Position 1: normal... Position 2: fine... Position 6: fine"` that was entirely fabricated to pad the 6-terminal correction.

## 5 Risks for a thesis that publishes this corpus
* **Overlay Leaks (Training Contamination):** Vision-language models trained on traces explicitly describing annotation masks will fail out-of-distribution without colored prompts (`switch_NG_HS_S0019`, `tape_NG_PS_S0128`).
* **Hallucinated Defect Rationalization on Normal Items:** Generator fabricates mechanical damage on pristine reference parts (`tape_NG_QS_S0098`).
* **Verdict / Ground-Truth Substrate Mismatch:** Model predicts `Damage` or `Missing Parts` by anchoring to wrong coordinate components, teaching reward models false visual grounding (`terminalblock_NG_AK_S0088`, `switch_NG_QS_S0017`).
