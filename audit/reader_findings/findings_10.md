# Chunk 10 — chunk_10.txt

## 1 Counts
- **Total Traces**: 36
- **Traces by Split**: `grpo_4k`: 17, `sft_6k`: 19
- **Traces by Category**: `wrong`: 30, `completely_wrong`: 6
- **Issue Type Distribution** (total flags = 82):
  - `defect_type_error`: 31
  - `feature_absent`: 21
  - `location_error`: 16
  - `viewpoint_error`: 7
  - `invented_defect`: 3
  - `overlay_leak`: 2
  - `verdict_mismatch`: 1
  - `count_error`: 1

## 2 Generator failure modes
1. **Hallucination of 3D / Side-View CAD Features (GRPO)**: Model invents external helical threads, protruding shafts, or peripheral feet/ribs on flat, top-down planar views (~15 traces).
   - Claim: *"It has a circular base with several small, evenly spaced protrusions around its perimeter. Rising from the center of the base is a cylindrical section with prominent external threads."*
   - Judge: *"The plastic plug does not have protrusions around the perimeter, nor are there any external threads visible on the cylindrical center from this top-down view."*
2. **Defect Polarity Reversal (Flash vs. Missing Parts)**: Model systematically misinterprets broken edges/chips as excess flash/burrs (~9 traces).
   - Claim: *"there is a small piece of excess material sticking out from the side wall. This looks like plastic flash that wasn't properly trimmed"*
   - Judge: *"The defect is a missing chunk / break / chip out of the outer edge, not flash or excess material."*
3. **Overlay / Annotation Leakage**: Model detects ground-truth defect masks (rendered red) instead of the underlying visual artifact (~3 traces).
   - Claim: *"I identify a small, bright red speck on the middle-left side of the component's top surface."*
   - Judge: *"The speck is light-colored/whitish in the query image; it only appears red in the defect overlay."*
4. **Coarse Defect Classification Shifts**: Conflating surface scuffs/abrasions with brittle porcelain chips (~6 traces).
   - Claim: *"The shape of the defect is elongated and jagged, which is characteristic of a surface chip in a hard material like porcelain."*
   - Judge: *"The defect is a surface abrasion or scuff mark, not a structural chip."*

## 3 Judge mistakes
1. **`plastic_nut_NG_ZW_S0020_plastic_nut_0020_NG_ZW_C1_20230914093814`**: The judge issues a `count_error` because the generator mentions *"including the five outer points"*. However, the generator was inspecting the *remaining* intact points (excluding the one being discussed), a common natural-language idiom rather than a geometric hallucination.
2. **`porcelain_doll_NG_CH_S0002_porcelain_doll_0002_NG_CH_C1_20230915180623`**: The judge flags `Chip` as a `defect_type_error` insisting strictly on `Abrasion`, even though the judge concedes the feature exposes *"the white ceramic base underneath"*, which is standard industrial terminology for chipping.

## 4 Rewrite quality
Rewrites effectively correct hallucinations, inaccurate defect types, and viewpoint errors while strictly maintaining the original binary verdict (`Yes`/`No`).
- **Clean Fix Without New Content**: `plastic_nut_NG_ZW_S0019_plastic_nut_0019_NG_ZW_C1_20231004093804` seamlessly replaces *"bright red speck"* with *"light-colored speck"*, removing the overlay leak without distorting reasoning flow.
- **Introduction of Hallucinated Numbers/Features**: In `plastic_plug_NG_HS_S0020_plastic_plug_0020_NG_HS_C1_20230913151031`, the rewrite injects specific unprompted markings (*"The molded marking '01' inside the recess is visible and clear"*), substituting the generator's fabricated threads with an ungrounded hallucination in the rewrite itself.

## 5 Risks for a thesis that publishes this corpus
- **Overlay Mask Leakage**: Visual traces refer directly to prompt/mask colors (`plastic_nut_NG_ZW_S0019`, `plastic_nut_ZW_S0038`, `plastic_nut_ZW_S0046`). Models trained on this corpus will overfit to annotation artifacts.
- **Pervasive Viewpoint Hallucination**: Over 40% of the GRPO subset hallucinates 3D helical threads and side views from perpendicular 2D images (`plastic_plug_NG_AK_S0023`, `plastic_plug_OK_S0054`), polluting physical reasoning.
- **Invented Defects on Defect-Free Objects & Verdict Inconsistencies**: Traces hallucinate tool wear or severe flash on gold standard normal items (`plastic_plug_NG_AK_S0149`, `plastic_plug_NG_ZW_S1072`, `plastic_plug_NG_ZW_S1115`), breaking grounding fidelity and requiring full re-labeling.
