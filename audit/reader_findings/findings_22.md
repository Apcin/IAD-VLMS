# Chunk 22 — chunk_22.txt

## 1 Counts
- **Total traces:** 34 (15 `toy`, 19 `toy_brick`).
- **By split:** `grpo_4k`: 20; `sft_6k`: 14.
- **By category:** `wrong`: 30; `completely_wrong`: 4.
- **Issue types:** `viewpoint_error`: 18; `location_error`: 17; `defect_type_error`: 17; `feature_absent`: 10; `marking_text_error`: 6; `colour_finish_error`: 5; `overlay_leak`: 4; `invented_defect`: 3; `not_visible_in_view`: 2; `verdict_mismatch`: 1.

## 2 Generator failure modes
1. **Hallucinating 3D isometric perspectives on 2D planar top-down images:** (16 traces). Models systematically invent right/front/top sides and orthogonal grain directions for single flat faces.
   - *Claim:* "I can see three faces: the top, the front, and the right side." (`toy_brick_NG_AK_S0121_toy_brick_0121_NG_AK_C1_20230917144723`)
   - *Why wrong:* "The image is a direct planar top-down view showing only a single square face."
2. **Overlay leakage (interpreting red GT annotation masks as physical product features):** (4 traces). Models mistake bounding/segmentation masks for actual red parts, flash, or paint stains.
   - *Claim:* "There is a small, circular red mark... appears to be a drop of misplaced paint or a stain." (`toy_NG_AK_S0028_toy_0028_NG_AK_C1_20230928100813`)
   - *Why wrong:* "The red mark described is from the overlay annotation mask, not a real red stain on the product, which actually shows a pit/void indentation defect at that spot."
3. **Mislabelling structural defects vs. surface contamination/pits:** (11 traces). Models regularly confuse surface dirt deposits for structural cracks, flash, or voids.
   - *Claim:* "There is a small, dark notch where the green outer layer has been gouged or chipped away..." (`toy_NG_ZW_S0083_toy_0083_NG_ZW_C1_20230927185846`)
   - *Why wrong:* "The defect is a small black foreign contamination/mark on the green rim, not a chip or gouged-out structural void."
4. **Hallucinating absent mechanical components (e.g., wheels) or drilled features:** (6 traces). Prior knowledge of rolling toys induces phantom parts.
   - *Claim:* "On the lower right side, a white wheel is partially visible beneath the green layer..." (`toy_OK_S0407_toy_0407_OK_C1_20230927133848`)
   - *Why wrong:* "There is no wheel visible anywhere beneath the green layer in this image."

## 3 Judge mistakes
1. **`toy_brick_NG_AK_S0010_toy_brick_0010_NG_AK_C1_20230917142453`**: The judge states: *"The brick is solid and does not naturally have a design hole; the hole/void shown is the pit defect itself."* Flagging the generator's description of a prominent cavity as a "hole" vs "pit" as a `feature_absent` defect is overly pedantic; the generator correctly identified and localized the pit defect.
2. **`toy_NG_ZW_S0009_toy_0009_NG_ZW_C1_20230927184155`**: The judge issues 4 separate flags asserting that a dark mark is strictly *"surface foreign contamination... not a puncture or void."* On low-resolution industrial imagery, distinguishing an uncleaned dark pit cavity from foreign particulate contamination is subjective; treating semantic divergence as outright hallucination is over-strict.

## 4 Rewrite quality
Rewrites generally excise hallucinations cleanly while preserving the verdict, but occasionally drop required outputs or fail to fully correct internal logic:
- **Clean fix:** In `toy_OK_S0407_toy_0407_OK_C1_20230927133848`, the hallucinated wheel claim was neatly replaced with: *"Around the perimeter, the outer green lettuce layer forms a smooth edge beneath the yellow border, indicating consistent molding of that feature,"* keeping the normal verdict intact.
- **Degraded/Blank rewrite:** In `toy_NG_QS_S0099_toy_0099_NG_QS_C1_20230927173602` and `toy_NG_QS_S0113_toy_0113_NG_QS_C1_20230927174001`, the judge completely erased the rewrite block (leaving it blank) because the defect was not discernible in that camera view, breaking the expected reasoning format.

## 5 Risks for a thesis that publishes this corpus
- **Ground-Truth Data Leakage:** Severe vulnerability where synthetic traces explicitly mention the red annotation mask (`toy_NG_AK_S0028`, `toy_NG_QS_S0029`, `toy_NG_QS_S0061`, `toy_NG_QS_S0068`). Training vision models on these will teach shortcuts based on evaluation artifacts.
- **Multi-View Ambiguity / Invisible Defects:** Traces like `toy_NG_QS_S0099` and `toy_NG_QS_S0113` show defects labeled "Yes" in Real-IAD that are physically occluded in Camera 1 (`C1`). Forcing models to reason about absent defects encourages hallucinations.
- **Verdict Mismatch:** In `toy_brick_NG_HS_S0111_toy_brick_0111_NG_HS_C1_20230917161123`, the generator hallucinates a scratch on an OK item and answers `<answer>Yes</answer>` when gold is `no`. The rewrite forcefully flips the verdict to `No`, altering the chain-of-thought post-hoc.
- **Systematic Text Misreading:** Consistent failure across multiple OK burger traces (`toy_OK_S0384`, `toy_OK_S0388`, `toy_OK_S0403`, `toy_OK_S0415`, `toy_OK_S0434`, `toy_OK_S0484`) hallucinating code `'O 001103'` instead of `'O 007703'`.
