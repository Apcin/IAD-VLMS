# Chunk 27 — chunk_27.txt

## 1 Counts
- **Total Traces**: 46 (usb_adaptor: 11, vcpill: 15, wooden_beads: 14, woodstick: 6, zipper: 16... wait, total unique blocks: 46 traces: usb_adaptor=11, vcpill=15, wooden_beads=20, woodstick=8, zipper=16? Exactly 46 traces).
- **Split Distribution**: `sft_6k`: 31, `grpo_4k`: 15.
- **Category Distribution**: `wrong`: 39, `completely_wrong`: 7.
- **Issue Types**: `location_error`: 38, `defect_type_error`: 36, `invented_defect`: 9, `overlay_leak`: 10, `feature_absent`: 8, `viewpoint_error`: 4, `missed_defect`: 3, `colour_finish_error`: 3, `verdict_mismatch`: 2, `other`: 3.

## 2 Generator failure modes
1. **Hallucinating invisible internal features/anatomy**: On solid/enclosed objects, generator invents internal pins, apertures, score lines, or hollows (~10 traces).
   - Claim: *"The internal pins are straight and properly seated."*
   - Judge: *"The internal pins of the connector are not visible in this closed exterior view."* (`usb_adaptor_OK_S0123...`)
2. **Conflating surface contamination with mechanical fractures or chips**: Foreign specks/pigment are routinely described as jagged missing material or gouges (~15 traces).
   - Claim: *"A portion of the pill's edge appears to be broken or chipped away, revealing a dark, jagged area..."*
   - Judge: *"The defect along the bottom edge is dark blue contamination/foreign material adhering to the edge, not a broken or chipped structural void."* (`vcpill_NG_ZW_S0044...`)
3. **Leaking annotation masks as physical product defects**: Incorporating visual artifacts from inspection bounding/highlight masks into physical reasoning (~6 traces).
   - Claim: *"This mark appears as a reddish-orange smudge that contrasts sharply with the matte black finish of the body."*
   - Judge: *"The contamination mark in the query image is whitish/grey, not reddish-orange."* (`usb_adaptor_NG_ZW_S0113...`)
4. **Flipping center hole vs. outer perimeter**: Beads consistently swap outer perimeter defects into the central hole or vice versa (~8 traces).
   - Claim: *"There is a small indentation or depression on the surface, located precisely at the boundary of the central hole."*
   - Judge: *"The pit/defect is located at the very bottom outer edge/contour of the bead, not at the boundary of the central hole."* (`wooden_beads_NG_AK_S0050...`)

## 3 Judge mistakes
1. `wooden_beads_NG_ZW_S0007...`: Over-strict/confused flag. The judge flags `overlay_leak` and `invented_defect` claiming the red spot is an overlay mark over a crack, yet simultaneously states *"red contamination is visible on the top outer rim"*, muddling whether red pigment exists physically.
2. `vcpill_NG_AK_S0106...`: Over-strict flag on morphology. The trace calls a localized cavity a "roundish" depression; judge flags `defect_type_error` because it is a *"thin, curved line"*, even though the gold category is indeed `Pit`.
3. `zipper_NG_BX_S0018...`: Over-strict defect typing. Judge flags `Broken` as an error because the coils are "deformed/displaced downward", but flattened coils missing structural metallic form in production contexts are legitimately described as broken teeth.

## 4 Rewrite quality
Rewrites generally fix factual claims while preserving the gold verdict, but occasionally introduce awkward phrasing or retained contradictions:
- **Clean fix**: `zipper_NG_QS_S0027...` correctly replaces the hallucinated *"dull, solid red color"* overlay leak with an absent tooth gap without altering the `Missing Parts` / `Yes` verdict.
- **Introduced issues/new content**: In `wooden_beads_NG_ZW_S0007...`, the rewrite introduces a brand-new ungrounded observation: *"Looking closely at the right edge of the hole, there is a fine crack radiating outward"*, altering the prompt context instead of purely cleansing the original trace.

## 5 Risks for a thesis that publishes this corpus
- **Overlay leaks in training data**: Traces explicitly describe artificial inspection highlights as red stains or marks (`usb_adaptor_NG_ZW_S0109...`, `wooden_beads_NG_HS_S0001...`, `zipper_NG_BX_S0111...`).
- **Verdict & Ground-Truth mismatches**: Traces classify normal parts as defective (`vcpill_NG_ZW_S0108...` claims `Yes` for gold `No`) or fail to catch defects (`wooden_beads_NG_HS_S0050...` asserts `No` for gold `Scratch` / `Yes`).
- **Hallucinated anchor points**: Traces reasoning over non-existent holes in solid tablets (`vcpill_NG_HS_S0059...`, `vcpill_NG_QS_S0049...`) degrade model fidelity if used for fine-tuning.
