# Chunk 11 — chunk_11.txt

## 1 Counts
- **Total Traces Flagged:** 38
- **By Split:** `sft_6k`: 23, `grpo_4k`: 15
- **By Category:** `wrong`: 31, `completely_wrong`: 7
- **Issue Type Distribution:**
  - `defect_type_error`: 42
  - `location_error`: 38
  - `count_error`: 12
  - `overlay_leak`: 7
  - `invented_defect`: 6
  - `feature_absent`: 6
  - `colour_finish_error`: 3
  - `marking_text_error`: 2
  - `other`: 1
  - `viewpoint_error`: 1
  - `missed_defect`: 1

## 2 Generator failure modes
1. **Defect Type Misclassification / Inversion:** Generator repeatedly hallucinates structural chips, cracks, or excess flash when the defect is an abrasion, stain, or missing piece. (~19 traces)
   - *Claim:* `In the bottom-right region, there is a visible deformation where the plastic material has overflowed the intended boundary of the rim. This extra material, known as flash...`
   - *Why Wrong:* `The defect is missing plastic / a missing part... not excess plastic/flash.` (`regulator_NG_QS_S0022`)
2. **Component Symmetry & Rotation Hallucination:** Systematically confuses part orientation, miscounting features or placing mounting holes/ribs 90° out of position. (~18 traces)
   - *Claim:* `The three radial ribs that connect the center to the outer wall are equally spaced...`
   - *Why Wrong:* `There are four radial ribs extending towards the outer wall (top, bottom, left, and right), not three.` (`regulator_OK_S0023`)
3. **Overlay & Heatmap Mask Leakage:** Directly describes red/pink ground-truth segmentation masks as literal stains, discoloration, or foreign matter on the physical object. (4 traces)
   - *Claim:* `There is a distinct red mark or stain on the edge of the plastic housing.`
   - *Why Wrong:* `The trace describes the red defect mask overlay as an actual red mark or stain on the part.` (`regulator_NG_QS_S0037`)
4. **Invented Spatial Structures:** Hallucinates missing material on non-existent features (e.g., imagining corner holes) instead of noticing the actual broken rib or edge. (~8 traces)
   - *Claim:* `Let me look closer at the top-left corner where a mounting hole is usually located on these types of parts...`
   - *Why Wrong:* `Mounting holes are located at the middle-left and middle-right (or top/bottom in rotated views), not the corners.` (`rolled_strip_base_NG_QS_S0075`)

## 3 Judge mistakes
1. **Over-strict parsing of sub-feature vs quadrant in** `rolled_strip_base_NG_QS_S0003`: Judge flags location as an error because the text says `top edge`, but the trace correctly deduced `<location>center</location>` matching the gold annotation.
2. **Questionable tagging in** `porcelain_doll_NG_ZW_S0022`: Judge claims `bottom-left` is wrong because `mask position is bottom-center`, but admits in its own rationale that the contamination spans `bottom-center/bottom-left inner rim`.
3. **Over-critical marking split in** `regulator_OK_S0227`: Flagged claim `The embossed markings "OT" and "10" are visible` as wrong because it's rotated "10", treating an understandable character rotation read as a visual grounding failure.

## 4 Rewrite quality
Rewrites consistently correct the invalid descriptive statements while retaining final `<answer>` labels, though they occasionally perform surgical text replacement that leaves surrounding reasoning slightly clunky.
- *Good Fix:* In `regulator_NG_QS_S0037`, cleanly removes red mask overlay hallucination (`red stain`) and replaces it with the actual physical defect (`missing section or fracture`), keeping the `<answer>Yes</answer>` intact.
- *Introduced Inconsistency / Weak Patch:* In `rolled_strip_base_NG_QS_S0135`, rewrite changes the target from the fastener hole to `the right edge of the central raised square platform`, but retains the later sentence `This could potentially interfere with cable tie insertion` without adjusting the earlier text stating `two circular holes on the top and bottom... likely for screws`, leaving a slightly disjointed chain-of-thought.

## 5 Risks for a thesis that publishes this corpus
- **Overlay Leaks:** Synthetic reasoning models will learn to condition on visual artifact anomalies from annotations rather than raw physical geometry (`regulator_NG_QS_S0011`, `regulator_NG_QS_S0037`, `regulator_NG_QS_S0039`, `regulator_NG_QS_S0048`, `regulator_NG_QS_S0103`).
- **Pervasive False Grounding / Invented Features on Defective Items:** Traces arrive at the correct `<answer>Yes</answer>` through purely hallucinated defect geometry (e.g. inventing a missing fastener hole in `rolled_strip_base_NG_QS_S0020` and `rolled_strip_base_NG_QS_S0075`).
- **Systematic Feature Miscounts on Normal Controls:** Normal (`OK`) components consistently hallucinate 3 instead of 4 ribs (`regulator_OK_S0023`, `regulator_OK_S0039`, `regulator_OK_S0129`), poisoning the chain-of-thought baseline for non-anomalous parts.
