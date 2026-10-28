# Chunk 05 — chunk_05.txt

## 1 Counts
- **Total traces:** 34
- **By split:** `grpo_4k`: 26, `sft_6k`: 8
- **By category:** `wrong`: 30, `completely_wrong`: 4
- **Issue types:**
  - `viewpoint_error`: 27
  - `location_error`: 19
  - `invented_defect`: 11
  - `defect_type_error`: 10
  - `marking_text_error`: 7
  - `feature_absent`: 7
  - `colour_finish_error`: 4
  - `verdict_mismatch`: 3
  - `missed_defect`: 1

## 2 Generator failure modes
1. **Hallucination of 3D Side Profiles from Top-Down Views:** The generator regularly invents stepped body walls, side grooves, and cylindrical profiles on planar en-face images (~18 traces).
   - Claim: *"Next, I will inspect the cylindrical side. This section features two horizontal grooves."* (`fire_hood_OK_S0079`)
   - Judge: *"The camera view is directly top-down, showing only the flat circular face and not the cylindrical side or its grooves."*
2. **Fabrication of Massive Structural Defects on Minor or Non-Existent Flaws:** The generator hallucinates missing letters, broken chunks, or large chips instead of recognizing intact features or minor localized spots (~7 traces).
   - Claim: *"There is a large area where the expected embossed text should be, but instead, I see only the smooth... surface"* (`mint_NG_QS_S0074`)
   - Judge: *"All characters 'Y-a-S-h-u-o' are present and clearly formed; the actual defect is a tiny chip/missing part on the outer perimeter edge..."*
3. **Severe Location Drift / Defaulting to Center:** The model routinely references defects at the "center" or flips coordinates despite features being along borders (~12 traces).
   - Claim: *"Completing the scan at the top-left with 'Y', 'A', I confirm that the primary anomaly I observed is that dark speck located on the right side of the mint..."* (`mint_NG_YW_S1427`)
   - Judge: *"The foreign fiber/object defect is located on the outer edge/perimeter on the middle-left side near the 'Y'/'a' text, not on the right side."*

## 3 Judge mistakes
1. **Over-strict semantic policing on physical description (`fire_hood_NG_HS_S0073`):** Judge insists claiming a "crack" is wrong because the gold category is "Scratch", arguing: *"The defect is a scratch rather than a fracture."* Visually, distinguishing an incised scratch from an unseparated hairline crack in wood/composite is ambiguous; penalizing this as a reasoning hallucination is overly strict.
2. **Dubious sub-location splitting (`fire_hood_NG_QS_S0092`):** Judge flags `top-right` vs `top-center`: *"The chipped missing chunk is located at the top-center edge, slightly right of top-center, but primarily top-center."* The generator's observation was grounded on the correct chip, making the flag pedantic.

## 4 Rewrite quality
- **Verdict preservation:** Rewrites consistently preserve the corrected verdict while stripping hallucinations. For example, in `mint_NG_YW_S1001`, the rewrite changes the verdict from `Yes` to `No` to match the gold standard, removing the hallucinated chip along the lower-left edge.
- **Introduction of unsupported content:** Rewrites occasionally hallucinate specific fine details not strictly verifiable from the raw prompt context. In `mint_NG_QS_S0110`, the rewrite replaces a missing letter hallucination with: *"Around the 'a' of this section, a significant patch of embossed material is chipped and missing..."*, asserting specific letter identities that the generator may not have properly resolved without prior knowledge.

## 5 Risks for a thesis that publishes this corpus
1. **Invented Defects on Normal Items:** Models hallucinatory rejections of clean parts; e.g., `mint_NG_YW_S1019` invents a foreign dark particle on a pristine mint to output `Yes` instead of `No`.
2. **False Verdict Mismatches / Flips:** Rejection vs. acceptance inversions contaminate training; e.g., `mint_NG_YW_S0013` (gold `yes`, predicted `No`) and `fire_hood_NG_ZW_S0103` (gold `no`, predicted `Yes`).
3. **Visual Perspective / Viewpoint Hallucinations:** Systematic generation of imaginary side surfaces on orthogonal face shots across entire classes (`fire_hood_OK_S0018`, `mint_NG_ZW_S0077`).
4. **Metadata / Overlay Leaking:** While bounding box coordinates are not explicitly named as red overlays in these samples, filename metadata leakage is evident where the trace reads class or view features directly (e.g., `mint_NG_ZW_S0115` hallucinating characters *"that seem to spell out 'ZW'"* directly mirroring the defect code `ZW` in the sample ID).
