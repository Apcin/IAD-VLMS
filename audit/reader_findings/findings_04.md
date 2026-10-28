# Chunk 04 — chunk_04.txt

## 1 Counts
- **Total Traces**: 37
- **Split**: `grpo_4k`: 21, `sft_6k`: 16
- **Category**: `wrong`: 33, `completely_wrong`: 4
- **Issue Types**: `viewpoint_error`: 29, `defect_type_error`: 24, `location_error`: 18, `feature_absent`: 15, `overlay_leak`: 11, `marking_text_error`: 7, `count_error`: 7, `colour_finish_error`: 4, `invented_defect`: 3, `missed_defect`: 3, `other`: 2, `verdict_mismatch`: 2, `not_visible_in_view`: 2

## 2 Generator failure modes
1. **Viewpoint Hallucination & Invisible Feature Fabrication** (~15 traces): The model hallucinates 3D side walls, grooves, and stepped geometry on purely 2D top-down orthogonal views.
   - Claim: *"Next, I'll inspect the cylindrical side wall. I can see the two distinct circumferential grooves."* (`fire_hood_NG_HS_S0038...`)
   - Why wrong: *"The image is an axial, top-down planar view of the circular face and does not show the cylindrical side wall or circumferential grooves."*
2. **Defect-Type Misclassification & Morphological Inversion** (~14 traces): Confusing material deficits (chips, missing parts, scratches) with additive flash/burrs, or stains with structural damage.
   - Claim: *"There is a jagged, protruding piece of extra material that extends beyond the straight edge of the cap. This material appears to be flash..."* (`end_cap_NG_QS_S0054...`)
   - Why wrong: *"The top-left corner is actually chipped/broken off (missing material/part), not extra protruding flash."*
3. **Ground-Truth Defect Overlay Leaks** (~8 traces): Misinterpreting artificial red annotation masks as red physical contaminants, stains, or wires.
   - Claim: *"Specifically, on the inner ring of the central circular feature, there is a small, misplaced piece of red material."* (`end_cap_NG_PS_S0110...`)
   - Why wrong: *"The trace treats the red mask annotation from the overlay as a red material/foreign object defect on the physical part."*
4. **Marking OCR and Spatial Inversion** (~11 traces): Inverting vertical/horizontal axes or misreading low-relief molded alphanumeric stamps.
   - Claim: *"Next, I will examine the embossed text 'OB-3030' on the lower left of the base."* (`end_cap_OK_S0151...`)
   - Why wrong: *"The embossed text 'OB-3030' is located on the upper-left / top-left area, not the lower left."*

## 3 Judge mistakes
1. **`end_cap_NG_ZW_S0075_end_cap_0075_NG_ZW_C1_20231010190443`**: The judge flags an `overlay_leak` claiming the red spot is an artifact, but immediately self-contradicts in the reason (*"...wait, looking at Image 1: in Image 1 the red spot is visible inside the top circular indentation!"*), leaving the rewrite completely blank.
2. **`fire_hood_NG_HS_S0057_fire_hood_0057_NG_HS_C1_20230928160157`**: The judge flags `[other]` because the generator called a diagonal mark "horizontal", yet ignores major viewpoint hallucinations in the same trace (*"lower cylindrical body"*, *"junction between the lower shaft and the top flange"* on a top-down view).

## 4 Rewrite quality
Judges generally excise unsupported claims without modifying valid final verdicts; however, some rewrites introduce newly fabricated visual assertions or leave empty outputs:
- **Good Rewrite (`end_cap_NG_PS_S0110...`)**: Successfully replaces the red overlay leak (*"misplaced piece of red material"*) with the genuine physical flaw (*"damaged, chipped area on the upper-right inner lip"*), fixing `<type>Foreign Object</type>` to `<type>Damage</type>` while preserving `<answer>Yes</answer>`.
- **Bad / Failed Rewrites**:
  - `end_cap_NG_ZW_S0113...` & `eraser_NG_QS_S0084...`: Completely empty rewrite blocks because the defect is invisible from the given camera angle.
  - `fire_hood_NG_HS_S0055...`: The rewrite introduces brand new unsupported visual claims (*"It's a dark line mark across the grain"* and *"The center core itself looks good"*), replacing one ungrounded trace with another.

## 5 Risks for a thesis that publishes this corpus
- **Overlay Mask Data Leakage**: Traces hallucinate red paint, strings, or foreign bodies directly caused by training on annotated ground-truth images (`end_cap_NG_ZW_S0050...`, `eraser_NG_HS_S0082...`, `eraser_NG_ZW_S0107...`).
- **Verdict Mismatches & False Positives**: Traces hallucinatory defects and outputs `<answer>Yes</answer>` on normal items, or judge flips the verdict entirely (`eraser_NG_ZW_S1243...`).
- **Defects Invisible in Evaluated View**: Multi-view Real-IAD labels applied to single 2D views force traces to hallucinate defects where none exist (`eraser_NG_QS_S0084...`, `end_cap_NG_ZW_S0113...`).
- **Label Incoherence**: Models classify missing material as flash while maintaining the target label `<type>missing parts</type>`, polluting reasoning alignment (`end_cap_NG_QS_S0029...`, `end_cap_NG_QS_S0051...`).
