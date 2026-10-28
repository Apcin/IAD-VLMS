# Chunk 03 — chunk_03.txt

## 1 Counts
- **Total Traces**: 32
- **Split Breakdown**: `sft_6k`: 16, `grpo_4k`: 16
- **Category Breakdown**: `wrong`: 26, `completely_wrong`: 6
- **Issue Type Distribution**: `overlay_leak`: 20, `defect_type_error`: 19, `location_error`: 16, `marking_text_error`: 9, `invented_defect`: 8, `feature_absent`: 6, `colour_finish_error`: 6, `verdict_mismatch`: 3, `other`: 2, `count_error`: 2, `viewpoint_error`: 2, `missed_defect`: 1.

## 2 Generator Failure Modes
1. **Overlay Mask Leakage (12 traces)**: Generator hallucinates real red/orange markings caused by the defect mask overlay.
   - Claim: *"There is a tiny, bright red spot located right on the edge of the circular symbol"* (`button_battery_NG_AK_S0029`)
   - Judge: *"The defect is a surface pit/blemish, not a bright red spot, which comes from the defect mask overlay."*
   - Claim: *"Inside the square icon... there is a small, bright red-orange speck."* (`button_battery_NG_AK_S0056`)
   - Judge: *"The speck in the query image is a tiny light/white pit, not bright red-orange; red-orange comes from the mask overlay."*
2. **Defect Morphological Misclassification (11 traces)**: Protruding edge flash/burrs are inverted into gouges, missing material, or dents.
   - Claim: *"There is a deep, jagged notch where the material has been gouged or chipped away."* (`button_battery_NG_CH_S0071`)
   - Judge: *"The defect consists of prominent surface scratches/scuffs/abrasions across the bottom region, not a deep notch of missing material."*
   - Claim: *"It looks like there's a significant chunk missing or broken off."* (`end_cap_NG_PS_S0001`)
   - Judge: *"The defect at the bottom edge is extra material/a hanging burr or plastic sliver... not missing material"*
3. **Hallucinated Features & Disoriented Spatial Grounding (12 traces)**: Traces invent text, counts, or swap cardinal axes entirely.
   - Claim: *"a number 'C1' is visible on the bottom edge."* (`end_cap_NG_ZW_S0041`)
   - Judge: *"The marking on the bottom right is '14', not 'C1'."*
   - Claim: *"I see the code 'OB-3030' molded near the top right."* (`end_cap_OK_S0097`)
   - Judge: *"The marking 'OB-3030' is located along the left side (or top-left), not the top right."*

## 3 Judge Mistakes
1. `button_battery_NG_ZW_S0047`: Flagged as `colour_finish_error` claiming the stain is blue, not red. While the overlay mask is red, the judge calls it an overlay leak despite admitting real staining existed, without clearly distinguishing true stain pigment from mask hue.
2. `end_cap_NG_HS_S0059`: Flagged as `other` for *"It runs roughly horizontally in that area."* In injection-molded parts with arbitrary orientation, calling a slightly diagonal/curved scratch "strictly vertical" vs "roughly horizontal" is pedantic over-strictness.
3. `end_cap_NG_ZW_S0012`: Flagged as `feature_absent` for text `'OB-3030'`, which is legitimate, but the judge also pruned harmless standard inspection sentences (e.g. scanning perimeter text) rather than just the absent string.

## 4 Rewrite Quality
Rewrites generally excise hallucinations and fix coordinate tags without flipping valid final answers. However, rewrites occasionally leave residual contradictions or invent grounding details not in the original prompt:
- **Good Rewrite**: In `button_battery_NG_AK_S0066`, safely strips the red overlay claim and corrects tag `<type>Foreign Object</type>` to `<type>Pit</type>` and location to `<location>center</location>` matching ground truth.
- **Problematic/Inventive Rewrite**: In `end_cap_NG_ZW_S0035`, the original trace reasoned about a central dusty defect. The rewrite transplants the entire paragraph to describe *"a small spot of foreign discolouration"* in the *"lower indent area"*, completely fabricating a new visual path to preserve the downstream answer.

## 5 Risks for a Thesis That Publishes This Corpus
- **Overlay Leaks**: Ground-truth masks bled into visual input tokens during corpus generation, causing models to rely on synthetic red cues (`button_battery_NG_AK_S0029`, `button_battery_NG_ZW_S0097`, `button_battery_NG_ZW_S0121`).
- **Verdict Mismatches & Invented Defects**: Hallucinating defects on pristine normal parts (`end_cap_NG_HS_S0057`, `end_cap_NG_HS_S0085`, `end_cap_NG_ZW_S0104`), or completely missing an anomaly on defective parts (`end_cap_NG_QS_S0121`).
- **Corrupted / Blank Data**: Blank/solid-black input images evaluated as normal products with fabricated detailed descriptions (`button_battery_OK_S0183`).
