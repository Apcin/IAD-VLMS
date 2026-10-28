# Chunk 15 — chunk_15.txt

## 1 Counts
- **Total Traces**: 30
- **Split Breakdown**: `grpo_4k`: 17, `sft_6k`: 13
- **Category Breakdown**: `wrong`: 23, `completely_wrong`: 7
- **Issue Type Distribution**:
  - `location_error`: 31
  - `defect_type_error`: 18
  - `feature_absent`: 15
  - `invented_defect`: 15
  - `colour_finish_error`: 5
  - `viewpoint_error`: 4
  - `count_error`: 4
  - `overlay_leak`: 2
  - `missed_defect`: 1
  - `other`: 1

## 2 Generator failure modes
1. **Hallucination of canonical features (e.g., SIM ejection pinholes)** (~8 traces)
   Generator assumes standard design elements exist regardless of orientation or actual part design.
   - Claim: *"featuring a large central cutout and a small retention hole at the handle end."*
   - Judge why-wrong: *"There is no retention/ejection pinhole visible on the tray handle in this image."*
2. **Defect misattribution across sub-assemblies (Rocker vs. Housing)** (~9 traces)
   Generator spots an anomaly (or hallucinates one) on the primary moving part rather than the body.
   - Claim: *"The defect is confirmed as a scratch, centrally located on the red actuator."*
   - Judge why-wrong: *"The scratch is on the black plastic latch/side retention clip in the center of the housing, not on the red button face."*
3. **Mischaracterizing terminal damage as missing slots/housing voids** (~6 traces)
   Generator interprets a broken spade lug as an unpopulated secondary slot or missing plastic shell.
   - Claim: *"there's a distinct empty slot or aperture in the plastic housing where another metallic spade terminal would normally be attached."*
   - Judge why-wrong: *"The defect is that the tip/end of the single visible metallic terminal is missing/broken off, not an entirely missing second terminal in a slot."*
4. **Classification mismatch between material removal and surface marking** (~7 traces)
   Generator hallucinates structural fracture ("Chip") when the defect is strictly surface scraping.
   - Claim: *"a section of the metal has been chipped or broken away, leaving a silver, jagged edge..."*
   - Judge why-wrong: *"The defect is a cluster of surface abrasions/scratches exposing underlying silver metal, not a chipped or broken-away section altering the outer profile."*

## 3 Judge mistakes
1. **`switch_NG_HS_S0096_switch_0096_NG_HS_C1_20231023095847`**: The trace correctly finds no defect (`<answer>No</answer>`), matching `gold_label=no`. The judge flags viewpoint errors and claims mounting ribs are visible, but overrides reasonable visual uncertainty from an oblique angle.
2. **`switch_NG_QS_S0025_switch_0025_NG_QS_C1_20231023105419`**: Judge penalizes count: *"Only one metallic terminal blade (with a small supporting prong beneath/behind it) is visible"*. Marking a terminal cluster with secondary prongs as strictly `count_error` when inspecting low-res profile views is overly pedantic.
3. **`sim_card_set_OK_S0466_sim_card_set_0466_OK_C1_20230922172453`**: Judge flags *"The four corners appear to be true right angles"* as `feature_absent` because one corner has a SIM chamfer. Calling this an invented defect rather than an imprecise general geometric remark is overly strict.

## 4 Rewrite quality
Rewrites successfully preserve final decisions without changing verdicts while stripping hallucinated claims, but occasionally inject speculative fine details not verified against reference images.
- **Good fix without drift (`sim_card_set_OK_S0442_sim_card_set_0442_OK_C1_20230922164537`)**: Replaces *"The small circular hole designed for the ejector pin is visible..."* with *"The ejector tab on the right side appears clean and smooth"*, keeping `<answer>No</answer>`.
- **Introduced unsupported content (`switch_NG_QS_S0035_switch_0035_NG_QS_C1_20231023105722`)**: Rewrite adds *"On the side latch mechanism near the center, there is a distinct cut or missing piece in the plastic rib"*, compounding multiple defect locations into one trace when the primary anomaly was purely terminal-focused.

## 5 Risks for a thesis that publishes this corpus
1. **Overlay Mask Leakage**: Traces hallucinating synthetic annotations as physical defects (`sim_card_set_NG_ZW_S0066_sim_card_set_0066_NG_ZW_C1_20230923101204`: trace interprets red ground-truth bounding/segmentation overlay as *"stray pigment on the side"* or *"foreign substance, likely ink or paint"*).
2. **Right Answer, Wrong Reason (Unfaithful CoT)**: Traces outputting correct `<location>` and `<type>` through complete visual confabulation (`switch_NG_HS_S0004_switch_0004_NG_HS_C1_20231023091355`, `switch_NG_QS_S0016_switch_0016_NG_QS_C1_20231023104923`: describing voids in the red actuator that do not exist, yet matching the gold label `<answer>Yes</answer>`).
3. **Severe Defect Hallucination on Missing Parts**: Traces claiming missing parts exist when totally absent (`switch_NG_QS_S0059_switch_0059_NG_QS_C1_20231023110942`, `switch_NG_QS_S0075_switch_0075_NG_QS_C1_20231023111426`: trace hallucinates inspecting markings on the red rocker when the entire rocker is missing from the switch).
