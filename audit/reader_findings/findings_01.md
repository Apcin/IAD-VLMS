# Chunk 01 — chunk_01.txt

## 1 Counts
- **Total Traces**: 34 (`grpo_4k`: 20, `sft_6k`: 14)
- **Product Breakdown**: `audiojack`: 24, `bottle_cap`: 10
- **Category Breakdown**: `wrong`: 23, `completely_wrong`: 11
- **Issue Type Distribution**: `defect_type_error`: 27, `invented_defect`: 10, `viewpoint_error`: 9, `location_error`: 9, `count_error`: 7, `colour_finish_error`: 6, `feature_absent`: 6, `overlay_leak`: 4, `missed_defect`: 3, `verdict_mismatch`: 2 (Total: 83 flags)

## 2 Generator failure modes
1. **Misattributing Defect Substrate (Plastic vs. Metal Contacts)** (~12 traces): Generator hallucinated warping or indentation of the plastic housing instead of identifying bent terminal leads, or claimed missing contacts when plastic was chipped.
   - *Claim*: "In the central area of the component, viewed from the side, the plastic housing seems to be pushed inwards or bent." (`audiojack_NG_BX_S0057`)
   - *Judge*: "The deformation is on the central internal metal contact spring/terminal, which is severely bent/twisted, not in the plastic housing."
2. **Viewpoint and Perspective Hallucination** (~8 traces): Describing direct planar top-down product images as side-view or angled perspectives, hallucinating unobservable side profiles or threads.
   - *Claim*: "This is a white plastic screw-on cap, presented in a side view." (`bottle_cap_NG_AK_S0045`)
   - *Judge*: "The image shows a top-down view (plan view) of the cap, not a side view."
3. **Overlay Mask Leakage** (~4 traces): Generating hallucinated descriptions of red marks, stains, or reflections caused by the visual defect-annotation overlay mask.
   - *Claim*: "There is a distinct, dark red mark or streak on the surface of the plastic." (`audiojack_NG_QS_S0014`)
   - *Judge*: "The red mark comes from the ground-truth overlay and is not on the physical sample."
4. **Incorrect Component Pin/Pad Counts** (~6 traces): Generating inaccurate pin counts (4 or 5) for standard 6-pad/pin arrays.
   - *Claim*: "On the right side, the green PCB section shows four solder points." (`audiojack_OK_S0433`)
   - *Judge*: "There are 6 solder points visible on the green PCB section arranged in a 2x3 grid, not 4."

## 3 Judge mistakes
1. `audiojack_NG_BX_S0128`: The trace predicted `<location>bottom-right</location>`. The judge flagged this as a `location_error` stating it is `middle-right`. However, the trace text explicitly states the pin is "in the bottom-right section of this internal cavity", which accurately describes the fine-grained location in that view; flagging this as strictly erroneous is overly rigid.
2. `bottle_cap_NG_HS_S0079`: The judge flagged a `defect_type_error` ("The defect is a scratch, not a pit"), but the generator had already concluded the trace with `<type>Scratch</type>`. The trace merely explored hypothesis testing ("If this mark is deep enough...") before correctly selecting Scratch.
3. `audiojack_NG_QS_S0027`: The judge claimed "The defect is a missing internal metallic part/terminal or structural portion at the bottom-left", yet in related audiojack traces (e.g. `audiojack_NG_QS_S0049`), the identical defect is recognized as chipped/missing plastic casing.

## 4 Rewrite quality
Rewrites generally excise hallucinations and fix attributes while preserving the label, but occasionally generate unsupported or slightly incongruent details:
- **Clean Fix**: In `audiojack_NG_BX_S0019`, rewrote "five metallic contact pins" and "leaning significantly towards the left" to "metallic contact pins on the PCB section" and "protruding significantly towards the right side", repairing factual inaccuracies cleanly.
- **Introduced Unsupported Detail**: In `bottle_cap_NG_ZW_S0039`, the judge replaced a general protrusion claim with "I can see a tiny speck of purple/foreign material located along the bottom edge". The color attribution ("purple") is an unverified detail injected directly by the judge's rewrite without image re-verification.

## 5 Risks for a thesis that publishes this corpus
- **Overlay Leaks Invalidate Visual Reasoning**: Traces describing ground-truth annotation masks (`audiojack_NG_BX_S0076`, `audiojack_NG_BX_S0086`, `audiojack_NG_QS_S0014`, `audiojack_NG_ZW_S0042`) prove models were fed labeled overlays or memorized annotation colors, poisoning the dataset for vision-language training.
- **Verdict & Tag Mismatches**: False negatives predicting `<answer>No</answer>` on gold positive samples (`audiojack_NG_BX_S0012`, `audiojack_NG_QS_S0108`) corrupt supervision.
- **Invented Defects with 'Lucky' Labels**: Multiple traces (`audiojack_NG_BX_S0087`, `audiojack_NG_QS_S0017`, `audiojack_NG_QS_S0100`) predict the correct gold class and location via hallucinated defect mechanisms (e.g., claiming missing internal pins when an external plastic rim was chipped). Fine-tuning on these traces teaches ungrounded rationalization.
