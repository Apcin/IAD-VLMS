# Chunk 28 — chunk_28.txt

## 1 Counts
- **Total traces:** 17 (all product `zipper`)
- **Split:** `sft_6k` (17)
- **Category:** `wrong` (14), `completely_wrong` (3)
- **Issue types:** `defect_type_error` (12), `location_error` (9), `colour_finish_error` (6), `invented_defect` (6), `count_error` (2)

## 2 Generator failure modes
1. **Misclassifying contamination/stains as physical/mechanical damage:** Generator hallucinates bent, crushed, or missing teeth when only topical foreign material exists (~6 traces).
   - `zipper_NG_ZW_S0011_...`: *"Several consecutive teeth in the bottom row are out of alignment, appearing bent or shifted downward..."* | Judge: *"The teeth are not bent or shifted; there is dark contamination/staining across the center teeth/tape area."*
   - `zipper_NG_ZW_S0078_...`: *"One of the interlocking teeth appears to be deformed or partially broken, creating a gap and a jagged edge..."* | Judge: *"The teeth are intact; the defect is dark contamination/foreign material caught between the teeth in the center."*
2. **Material composition hallucination (plastic vs. metallic):** Generates descriptions of clear or white molded plastic teeth on standard silver metallic coil zippers (~4 traces).
   - `zipper_OK_S0122_...`: *"I am inspecting a section of a transparent plastic zipper attached to a black fabric tape."* | Judge: *"The zipper teeth/coils are metallic silver/chrome in appearance, not transparent plastic."*
   - `zipper_NG_QS_S0106_...`: *"I am observing the alignment and continuity of the translucent plastic teeth..."* | Judge: *"The zipper teeth are shiny metallic/silver-coiled elements, not translucent plastic."*
3. **Mislocating center defects to "middle-right" / "center-right":** Over-predicts right-of-center placement for strictly centered anomalies (~6 traces).
   - `zipper_NG_QS_S0061_...`: *"However, as I reach the center-right area, there is a distinct interruption in the pattern."* | Judge: *"The defect is located near the center of the image, not the center-right area."*
   - `zipper_NG_ZW_S0119_...`: *"Scanning along the top row of teeth, I notice a section in the center-right area..."* | Judge: *"The discolored blue marks are located at the center along the top edge of the zipper, not in the center-right area."*

## 3 Judge mistakes
1. `zipper_NG_ZW_S0111_zipper_0111_NG_ZW_C1_20231008155039`: The judge flags `<type>Deformation</type>` as wrong because the defect is contamination, but leaves the body unchanged in the rewrite (*"significant contamination or a localized melting... blockage a definitive structural and functional anomaly"*), creating an internal contradiction between the think block and the final tag.
2. `zipper_NG_QS_S0114_zipper_0114_NG_QS_C1_20231008150304`: Strictness on `count_error` (*"three consecutive teeth missing... not just a single tooth"*). Flagging *"a missing tooth"* as a factual error when multiple missing teeth form a continuous gap is pedantic; it is a single missing gap defect.

## 4 Rewrite quality
Rewrites successfully preserve the verdict (`Yes`/`No`) and fix target claims, but occasionally insert unsupported assertions:
- **Clean fix:** In `zipper_NG_QS_S0082_zipper_0082_NG_QS_C1_20231008144145`, seamlessly replaces false flash defect with missing tooth: *"One of the teeth is missing from the zipper track, leaving an empty gap... This is a missing part defect..."* while keeping `<answer>Yes</answer>`.
- **Unsupported addition:** In `zipper_NG_ZW_S0011_zipper_0011_NG_ZW_C1_20231008152420`, the rewrite inserts *"There is dark contamination staining the fabric tape and upper elements..."*, yet the original trace focused on the bottom row, and the judge had previously noted contamination across center teeth/tape without specifying upper elements.

## 5 Risks for a thesis that publishes this corpus
- **Overlay leaks / Hallucinations:** In `zipper_NG_QS_S0090_zipper_0090_NG_QS_C1_20231008144505`, the generator describes a *"visible reddish-brown discoloration or speck on the underlying white carrier tape"*, which the judge notes is likely an artifact or overlay leak rather than a real feature.
- **Type/Verdict Mismatches:** In `zipper_NG_ZW_S0028_zipper_0028_NG_ZW_C1_20231008152908` and `zipper_NG_QS_S0082_zipper_0082_NG_QS_C1_20231008144145`, models output completely wrong defect classes (`Missing Parts` instead of `Contamination`, `Flash` instead of `Missing Parts`). If trained on original traces, downstream models learn incorrect visual grounding associations for defect taxonomy.
- **Missing Parts vs. Missing Thread Ambiguity:** In `zipper_NG_QS_S0068_zipper_0068_NG_QS_C1_20231008143619` and `zipper_NG_QS_S0125_zipper_0125_NG_QS_C1_20231008150644`, gold label `Missing Parts` represents missing stitch thread, but the generator hallucinated sheared/missing teeth. Publishing original traces would reinforce false positive tooth-detection logic.
