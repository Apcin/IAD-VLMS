# Chunk 09 — chunk_09.txt

## 1 Counts
- **Total Traces**: 44
- **By Split**: `sft_6k`: 31, `grpo_4k`: 13
- **By Category**: `pcb`: 23, `phone_battery`: 10, `plastic_nut`: 11
- **Issue Types**: `defect_type_error`: 42, `location_error`: 31, `overlay_leak`: 16, `colour_finish_error`: 6, `invented_defect`: 5, `count_error`: 4, `feature_absent`: 3, `viewpoint_error`: 1 (Total flags: 108)

## 2 Generator failure modes
1. **Overlay Mask Hallucination**: Generator misinterprets synthetic red ground-truth overlay bounding masks as real physical defects (16 traces).
   - Claim: *"there is a bright red foreign object. It appears to be a small piece of plastic or wire insulation..."* (`pcb_NG_YW_S0100`)
   - Judge why-wrong: *"The actual foreign object is a metallic/silver wire or staple, not bright red; the red color comes solely from the overlay mask."*
2. **Polarity/Sign Inversion on Defects**: Inverts missing features into additive flash, protrusions, or extra components (8 traces).
   - Claim: *"There is a protrusion of excess material extending outward from the intended sharp corner. This looks like flash..."* (`plastic_nut_NG_QS_S0023`)
   - Judge why-wrong: *"The defect at the bottom-center corner is a broken/missing chunk of plastic material (incomplete corner/missing part), not excess flash protrusion."*
3. **Severe Spatial Disorientation**: Flips board axes, mirrors horizontal/vertical dimensions, or shifts clock positions (15 traces).
   - Claim: *"Checking the contacts on the left side of the component: I see three separate metallic pads."* (`phone_battery_OK_S0164`)
   - Judge why-wrong: *"The metallic contacts are located on the right side of the component in the image, not the left."*
4. **Phantom Component & Defect Fabrication**: Hallucinates standard design features as defects or invents completely fabricated structures (6 traces).
   - Claim: *"There is an empty set of rectangular solder pads located between the button and the 'C1' marking."* (`pcb_NG_QS_S0021`)
   - Judge why-wrong: *"There is no separate empty set of rectangular solder pads between the button and C1."*

## 3 Judge mistakes
1. **Unsalvaged Empty Output (`pcb_NG_QS_S0083`)**: The judge correctly caught severe hallucination (*"empty metallic frame with internal contact point exposed"*), but left `SHOULD BE:` and `GEMINI REWRITE` completely blank, producing a broken corpus item.
2. **Over-Strict Fine Semantic Flagging (`pcb_NG_ZW_S0092`)**: Flagged trace tag `<type>Damage</type>` as a `defect_type_error` because the ground truth is `Contamination`, even though the trace explicitly identified the defect as a *"dark red, irregular smear... external contaminant or residue"* (a taxonomy classification quibble rather than visual hallucination).
3. **Doubtful Claim on Foreign Color (`phone_battery_NG_ZW_S0030`)**: In flagging a tear vs. contamination, the judge claims the object is *"foreign red contamination adhering to the outer edge"*, which likely reflects the overlay mask rather than the true query artifact.

## 4 Rewrite quality
Rewrites generally fix factual errors and preserve gold verdict classifications (`Yes`/`No`), but occasionally leave stylistic residue or fail entirely:
- **Clean Fix (`phone_battery_NG_AK_S0057`)**: Successfully replaced hallucinated overlay leak (*"small, bright red mark or inclusion"*) with grounded morphology (*"small pit or indentation located at the bottom edge of the metal terminal"*), fixing the tag from `Contamination` to `Pit`.
- **Incomplete / Omission Failure (`pcb_NG_QS_S0083`)**: The judge stripped the trace due to fabrication but completely failed to generate a rewrite, leaving a null string.

## 5 Risks for a thesis that publishes this corpus
- **Overlay Leaks Contaminating Training**: Traces explicitly training models to treat red overlay artifacts as physical anomalies (`pcb_NG_YW_S0113`, `pcb_NG_YW_S0063`, `phone_battery_NG_PS_S0032`), teaching models spurious correlations.
- **Normal Features Flagged as Defects**: Genuine test points and normal structures are flagged as defects (`pcb_NG_HS_S0023` treating an unmasked copper test pad as a chemical leak/stain).
- **False Negative Ground Truth / Mislabeled Gold**: Normal OK samples containing fabricated structural claims (`pcb_OK_S0468` claiming four corner mounting holes when only one exists), allowing hallucinated reasoning to yield correct final answers.
- **Null Rewrites**: Publishing truncated or missing rewrites (`pcb_NG_QS_S0083`) introduces fatal formatting and parsing errors into fine-tuning pipelines.
