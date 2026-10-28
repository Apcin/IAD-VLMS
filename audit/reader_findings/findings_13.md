# Chunk 13 — chunk_13.txt

## 1 Counts
- **Total traces:** 34
- **Split:** `grpo_4k`: 21 | `sft_6k`: 13
- **Category:** `wrong`: 28 | `completely_wrong`: 6
- **Issue types:** `feature_absent`: 24 | `location_error`: 23 | `defect_type_error`: 19 | `colour_finish_error`: 6 | `overlay_leak`: 3 | `missed_defect`: 2 | `invented_defect`: 1 | `verdict_mismatch`: 1 | `count_error`: 1 (Total flags: 80)

## 2 Generator failure modes
1. **Hallucination of stereotypical object parts (sim_card_set pinhole)** (~15 traces)
   The generator invents standard domain features (e.g., SIM ejection hole) absent in top-down views.
   - Claim: *"The small hole on the ejector tab also seems properly formed."* (`sim_card_set_NG_CH_S0168`)
   - Why wrong: *"There is no visible hole on the ejector tab in this image."*
2. **Defect type conflation and inversion (negative vs. positive features)** (~11 traces)
   The model describes physical voids, cuts, or tears as additive features (flash, overmolding, contamination).
   - Claim: *"Along the middle-left edge, the straight perimeter is interrupted by jagged, excess material... often referred to as flash"* (`rolled_strip_base_NG_QS_S0015`)
   - Why wrong: *"The defect is a missing part / tear in the outer edge material, not excess material / flash."*
3. **Severe spatial mislocalization on symmetric parts** (~15 traces)
   Orientation drift causes features on vertical axes to be mapped to horizontal axes or opposite corners.
   - Claim: *"The two small circular holes, one towards the top-left and one towards the bottom-right, appear to be cleanly formed."* (`rolled_strip_base_OK_S0500`)
   - Why wrong: *"The circular features/holes are located top-center and bottom-center, not top-left and bottom-right."*
4. **Visual leakage from red annotation masks/overlays** (~3 traces)
   The generator interprets synthetic segmentation overlays as physical stains, ink, or foreign plastic.
   - Claim: *"Inside this channel, there is a visible piece of reddish-orange foreign material lodged against the plastic wall."* (`rolled_strip_base_NG_QS_S0150`)
   - Why wrong: *"The query image shows clear plastic breakage... no reddish-orange foreign material; the red colour comes from the overlay."*

## 3 Judge mistakes
1. **`sim_card_set_NG_ZW_S0064_sim_card_set_0064_NG_ZW_C1_20230923101357`**
   The judge flags `[overlay_leak]` on *"There's a section that shows a patch of reddish material..."*, admits in the rationale (*"The query image actually shows a reddish/pinkish smear in that area..."*), supplies an identical `SHOULD BE`, and generates no rewrite. The flag is unjustified.
2. **`rolled_strip_base_NG_QS_S0012_rolled_strip_base_0012_NG_QS_C1_20231014091832`**
   The judge flags `<type>Crack</type>` under `[defect_type_error]` because the gold label is `Missing Parts`, but concedes: *"The defect is missing material/broken away missing part rather than just a crack, though crack is noted in type tag."* The original trace explicitly describes a *"jagged split"* and *"fracture"* extending from the mounting hole, making the flag overly pedantic.

## 4 Rewrite quality
Rewrites consistently correct localized claim errors while preserving the gold verdict, but sometimes introduce awkward syntax or drop structural tags.
- **Good repair:** In `sim_card_set_NG_CH_S0049_sim_card_set_0049_NG_CH_C1_20230923105949`, the rewrite cleanly removes the hallucinated ejector pinhole, corrects horizontal to diagonal scratch propagation, fixes the location to the flat panel right of the cutout, and keeps `<answer>Yes</answer>`.
- **Flawed repair:** In `sim_card_set_NG_ZW_S0064_sim_card_set_0064_NG_ZW_C1_20230923101357`, the judge leaves `--- GEMINI REWRITE ---` completely blank due to unresolved reasoning over whether the red mark was an overlay or real residue.

## 5 Risks for a thesis that publishes this corpus
- **Overlay leaks poisoning ground truth reasoning:** In `rolled_strip_base_NG_QS_S0122` (*"reddish-pink stain"*), `rolled_strip_base_NG_QS_S0150` (*"reddish-orange foreign material"*), and `sim_card_set_NG_CH_S0164` (*"reddish-pink discoloration"*), traces cite segmentation masks as physical defects, proving synthetic contamination during data generation.
- **Verdict mismatches:** `sim_card_set_NG_CH_S0157` has gold label `yes`, but the generator trace concludes `<answer>No</answer>` (false negative), presenting an invalid training signal for downstream supervised tuning.
- **Invented defects on normal items:** Models hallucinate defect candidates or misinterpret normal process marks on `OK` items, while inventing absent structural elements across both `OK` and `NG` splits (e.g., nonexistent ejector pinholes across 11 `sim_card_set` traces).
