# Chunk 06 — chunk_06.txt

## 1 Counts
* **Total traces:** 37 (mint: 18, mounts: 16, pcb: 3).
* **By split:** `grpo_4k`: 16, `sft_6k`: 21.
* **By category:** `wrong`: 29, `completely_wrong`: 8.
* **Issue type distribution (87 total flags):**
  * `location_error`: 31
  * `defect_type_error`: 27
  * `overlay_leak`: 13
  * `feature_absent`: 8
  * `colour_finish_error`: 6
  * `viewpoint_error`: 10
  * `invented_defect`: 6
  * `marking_text_error`: 4
  * `missed_defect`: 1
  * `not_visible_in_view`: 1

## 2 Generator failure modes
1. **Overlay Mask Leakage (Hallucinating Annotations as Real Features):** Generator treats red/pink ground-truth bounding masks or highlights as real physical features on the product (~11 traces).
   * Claim: `"However, in the center-right area, there is a very fine, hair-like red fiber stuck to the surface of the tablet."` (`mint_NG_QS_S0005...`)
   * Judge why wrong: `"The red mark is the defect overlay mask marking missing/chipped areas, not a red fiber on the product."`
2. **Defect Inversion & Type Misclassification:** Misinterpreting missing chips, voids, or pits as positive extrusions (flash) or stains/contamination (~14 traces).
   * Claim: `"Right in the bottom-center area, there is a prominent protrusion of jagged, translucent plastic... likely flash or a molding overflow."` (`mounts_NG_QS_S0012...`)
   * Judge why wrong: `"The bottom edge exhibits a triangular chip or indentation where material is missing (torn/broken off), not excess protrusion/flash."`
3. **Perspective & Feature Fabrication (Template Hallucination):** Generator assumes an angled/profile perspective or fabricates canonical features that do not exist in planar top-down views (~8 traces).
   * Claim: `"This image is taken from an angled side view, heavily highlighting the profile and the side wall texture of the white, ring-shaped tablet."` (`mint_NG_ZW_S0190...`)
   * Judge why wrong: `"The image shows a top-down view of the ring-shaped tablet face rather than an angled side view."`
4. **Spatial Incoherence / Swapped Cardinal Coordinates:** Correctly identifying anomaly morphology but hallucinating mirrored or incorrect locations (~16 traces).
   * Claim: `"On the right side, there's a sharp break in the plastic, and a whole section is missing."` (`mounts_NG_QS_S0019...`)
   * Judge why wrong: `"The break in the plastic is on the middle-left side, not on the right side of a top cutout."`

## 3 Judge mistakes
1. **`mint_NG_ZW_S0078_mint_0078_NG_ZW_C1_20230910095658`:** Judge flagged `[not_visible_in_view]` claiming the defect is completely invisible in the query image and failed to provide a valid rewrite (`SHOULD BE:` and `GEMINI REWRITE` are completely empty), creating a broken corpus entry while the gold label remains `yes`.
2. **`mounts_NG_ZW_S0160_mint_0160_NG_YW_C1_20230910133333`:** The judge claims in its rewrite that `"foreign fibers are present, including a red/pink thread-like fiber along the outer edge"`, re-introducing the exact red overlay leak it flagged as erroneous in other samples (e.g., `mint_NG_YW_S0012`).

## 4 Rewrite quality
Rewrites generally fix spatial flags and text hallucinations without changing the binary verdict, but occasionally drop critical reasoning structure or patch over ungrounded artifacts:
* **Clean fix:** In `pcb_NG_HS_S0003_pcb_0003_NG_HS_C1_20231028114201`, the judge accurately re-anchors the scratch from the blue solder mask onto the metallic shield of the USB connector without altering the final defect diagnosis or tags.
* **Incomplete / Empty rewrite:** In `mint_NG_ZW_S0078_mint_0078_NG_ZW_C1_20230910095658`, the judge flagged the trace as completely ungrounded because the anomaly was invisible, but left the `GEMINI REWRITE` completely blank, producing a corrupt empty trace.

## 5 Risks for a thesis that publishes this corpus
* **Direct Overlay Contamination Leaks:** Training on traces describing synthetic red markings (e.g., `mint_NG_QS_S0001`, `mounts_NG_QS_S0079`, `mounts_NG_ZW_S0064`) teaches models to attend to annotation artifacts rather than physical surface defects.
* **Post-hoc Rationalization of Label Noise:** If an item is labeled defective due to a microscopic sub-pixel artifact (`mint_NG_ZW_S0078`), models hallucinate macro defects (stains, chips) to justify the prompt's ground-truth label.
* **Corrupt/Empty Generation Outputs:** Empty rewrites (`mint_NG_ZW_S0078`) or rewritten traces retaining overlay artifact descriptions (`mint_NG_YW_S0160`) degrade dataset integrity and fine-tuning pipelines.
