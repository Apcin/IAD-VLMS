# Chunk 24 — chunk_24.txt

## 1 Counts
- **Total traces:** 41
- **Split distribution:** `sft_6k`: 24; `grpo_4k`: 17
- **Category distribution:** `wrong`: 33; `completely_wrong`: 8
- **Issue types:** `count_error`: 27; `defect_type_error`: 27; `location_error`: 21; `invented_defect`: 11; `colour_finish_error`: 10; `feature_absent`: 7; `viewpoint_error`: 4; `overlay_leak`: 4; `verdict_mismatch`: 2; `missed_defect`: 2

## 2 Generator failure modes
1. **Severe counting blindness on repetitive structures (~17 traces):** Generator reliably hallucinates 14, 16, or 20 pins on standard 18-lead packages.
   - Claim: `"There are eight leads on each side, for a total of sixteen."`
   - Judge: `"There are nine leads on each side for a total of eighteen leads."` (`transistor1_OK_S0163`)
2. **Substrate & component confusion (~11 traces):** Generator projects defects situated on blue plastic housings onto metallic shells, pins, or cavity walls.
   - Claim: `"There's a distinct line that appears to be on the metal surface."`
   - Judge: `"The scratch is on the blue plastic, contrasting with the blue background."` (`usb_NG_HS_S0072`)
3. **Inner vs. outer perimeter inversions on annular parts (~8 traces):** Systematic spatial disorientation on circular boundaries.
   - Claim: `"A small wedge of material appears to be missing from the outer edge..."`
   - Judge: `"The missing material is located on the inner edge (inner bore), not the outer edge."` (`u_block_NG_QS_S0076`)
4. **False-positive hallucination on defect-free samples (~3 traces):** Fabricating severe mechanical damage on normal controls.
   - Claim: `"There appears to be a significant scratch or scoring mark. It's running lengthwise along the housing..."`
   - Judge: `"The metal housing is normal and free of deep scratches or defects."` (`usb_NG_HS_S0001`)

## 3 Judge mistakes
1. **Wrong ground-truth pin count on `transistor1_OK_S0382`:** Trace stated 14 pins (7/side). The judge claimed: `"The component has 16 pins in total, with 8 on the top and 8 on the bottom"`, whereas the standard `transistor1` package across all other 16 audit items is an 18-pin device (SOIC-18 / ULN2803AG).
2. **Contradictory geometry fix on `transistor1_OK_S0445`:** Judge asserted: `"There are ten pins on the top and ten pins on the bottom (a 20-pin SOP/SOIC package)"`, directly contradicting every identical `transistor1` sample evaluated as 18-pin SOP-18.
3. **Over-strict semantic policing on `u_block_NG_QS_S0120`:** Flagged `[defect_type_error]` on `<type>Damage</type>` because gold label was `Missing Parts`, despite the trace correctly diagnosing `"a localized indentation or chip where the material appears to have been pressed in or removed"`.

## 4 Rewrite quality
Rewrites successfully correct targeted claims and maintain verdicts, but occasionally introduce awkward, repetitive phrasing:
- **Clean rewrite (`transistor1_OK_S0116`):** Swapped hallucinated `"16 metallic leads (8 on each side)"` cleanly to `"18 metallic leads (9 on each side)"` without altering inspection logic.
- **Syntactic artifact (`usb_NG_HS_S0073`):** Judge replaced a description with an abrupt sentence fragment: `"On the lower-right section of the outer rim, there is a visible deformation. Instead, there is a distinct scratch mark..."`, introducing disjointed reasoning.

## 5 Risks for a thesis that publishes this corpus
- **Overlay leaks:** Generator directly hallucinates annotations from defect masks (`"reddish tint"`, `"bright red speck"`, `"red linear mark"`): `u_block_NG_CH_S0063`, `u_block_NG_HS_S0054`, `u_block_NG_HS_S0099`, `u_block_NG_HS_S0116`.
- **Verdict & label mismatches:** Model outputs `<answer>Yes</answer>` or missing tags on normal parts (`usb_NG_HS_S0001`, `usb_NG_HS_S0114`, `usb_NG_BX_S0017`).
- **Hallucinated defects on golden normals:** Traces conclude `No` while describing catastrophic bent pins or warped shields (`usb_NG_HS_S0036`, `usb_NG_HS_S0061`).
- **Pervasive spatial/viewpoint ungrounding:** Hallucinating front ports on planar top-views (`usb_NG_HS_S0024`, `usb_NG_HS_S0105`, `usb_NG_HS_S0116`).
