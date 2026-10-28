# Chunk 25 — chunk_25.txt

## 1 Counts
- **Total traces:** 36 (19 `grpo_4k`, 17 `sft_6k`; 23 `usb`, 13 `usb_adaptor`)
- **Category:** `wrong`: 23, `completely_wrong`: 13
- **Issue types:** `defect_type_error`: 28, `location_error`: 19, `invented_defect`: 14, `overlay_leak`: 9, `viewpoint_error`: 9, `feature_absent`: 7, `count_error`: 7, `colour_finish_error`: 6, `not_visible_in_view`: 2, `other`: 1 (Total flags: 96)

## 2 Generator failure modes
1. **Overlay hallucination**: Reads artificial red/orange training bounding masks as physical product defects (~9 traces).
   - Claim: `"In the center area, between the two rectangular raised features, I notice a thin, horizontal red sliver."` (`usb_NG_BX_S0014`)
   - Judge: `"The red sliver only exists in the overlay image and represents the defect mask, not the actual defect on the component."`
2. **Defect misattribution (pins vs. housing)**: Blindly attributes missing/damaged plastic or chassis tabs to missing terminal pins (~12 traces).
   - Claim: `"It appears that one or more of the smaller pins are completely missing from their designated slots."` (`usb_NG_QS_S0009`)
   - Judge: `"The defect is missing blue plastic material (short shot/broken molding) along the bottom edge, not missing metal pins."`
3. **Perspective & viewpoint fabrication**: Hallucinates internal cavities/ports on orthogonal planar exterior views (~9 traces).
   - Claim: `"While the internal contacts aren't clearly visible, the external housing and the blue plastic interior look normal."` (`usb_adaptor_NG_AK_S0019`)
   - Judge: `"The USB-A female port interior is completely facing away / not visible from this perspective."`
4. **Defect type conflation**: Conflates physical structural deformation/cracks with foreign object contamination (~8 traces).
   - Claim: `"A thick, dark, zig-zagging wire or thread-like piece of material is protruding from the bottom-left area."` (`usb_NG_BX_S0116`)
   - Judge: `"The protruding feature is a deformed or bent metal tab/lead of the connector rather than foreign debris."`

## 3 Judge mistakes
1. `usb_NG_ZW_S0121_usb_0121_NG_ZW_C1_20231020164701`: Judge flags overlay leak incorrectly as grounded: *"bright red speck"* is retained as valid in the rewrite despite earlier recognizing red specks as defect overlay masks (e.g., in `usb_NG_QS_S0067`).
2. `usb_NG_BX_S0038_usb_0038_NG_BX_C1_20231020113937`: Judge labels copper/reddish color as `colour_finish_error` claiming the protruding pin is strictly silver; real-world copper/phosphor-bronze sub-layers are commonly exposed on sheared USB contacts.

## 4 Rewrite quality
Rewrites successfully correct location and defect types while preserving original XML verdicts, but occasionally introduce unsupported reference claims:
1. **Clean fix**: `usb_NG_QS_S0009_usb_0009_NG_QS_C1_20231020134244` shifts claim from "missing pins" to "missing plastic housing section" while keeping gold tag `<type>Missing Parts</type>` and `<answer>Yes</answer>`.
2. **Introduces unsupported reference leak**: In `usb_NG_QS_S0040_usb_0040_NG_QS_C1_20231020134833`, rewrite adds `"On the reference unit, this area is properly covered by the metal casing..."`, inventing access to an external reference pair not established in single-image inspection.
3. **Empty rewrites**: For pure overlay leaks on invisible defects (`usb_NG_ZW_S0033`, `usb_adaptor_NG_AK_S0051`), the judge yields empty rewrites, failing to resolve the trace.

## 5 Risks for a thesis that publishes this corpus
- **Overlay leaks**: Training traces learn to hallucinate red annotations (`usb_NG_ZW_S0033`, `usb_NG_BX_S0014`, `usb_NG_BX_S0061`, `usb_NG_QS_S0047`).
- **Gold-label contamination / invisible defects**: Normal views are forced to justify false-positive defect labels (`usb_adaptor_NG_AK_S0051`, `gold_label=yes` but judge notes *"overlay shows no visible defect/mask in this view"*).
- **Invented defects on normal items**: Trace describes unseen internal geometry to justify normal labels (`usb_adaptor_NG_AK_S0056`, `usb_adaptor_NG_CH_S0030`, `usb_adaptor_NG_CH_S0054`).
- **Tag vs Reason disconnect**: Traces arrive at correct gold answers via completely bogus intermediate causal explanations (`usb_NG_QS_S0014`, `usb_NG_QS_S0082`).
