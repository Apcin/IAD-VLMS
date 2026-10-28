# Chunk 12 — chunk_12.txt

## 1 Counts
- **Total Traces**: 25 (all product=`rolled_strip_base`).
- **By Split**: `grpo_4k`: 25.
- **By Category**: `wrong`: 25.
- **Issue Type Distribution**:
  - `location_error`: 29
  - `colour_finish_error`: 7
  - `count_error`: 4
  - `viewpoint_error`: 4
  - `feature_absent`: 2
  - `invented_defect`: 1

## 2 Generator failure modes
1. **Hallucinating mounting hole / feature coordinates** (~18 traces): Model consistently defaults to "top corners", "top-left/bottom-right", or "top/bottom" without resolving part orientation.
   - Trace `rolled_strip_base_OK_S0025_...`: *"I observe the two visible circular depressions—one upper-left and one lower-right..."*
   - Judge: *"The two circular depressions are located top-center and bottom-center, not upper-left and lower-right."*
2. **Color blindness / default dark speck assumption** (~5 traces): Assumes surface contamination is a dark/black speck rather than recognizing vivid artificial dyes/inks.
   - Trace `rolled_strip_base_NG_ZW_S0029_...`: *"It appears as a small, dark spot or speck on the surface."*
   - Judge: *"The defect visible at top-center is a bright red ink/stain mark, not a dark spot or dark speck."*
3. **Inventing camera viewpoints / profile geometry** (~2 traces): Fabricates perspective angles and side profiles on planar top-down images.
   - Trace `rolled_strip_base_OK_S0056_...`: *"The image provided is a perspective shot, showing the profile view."*
   - Judge: *"The image is a direct top-down plan view, not a perspective or profile view."*

## 3 Judge mistakes
1. **Contradictory ground-truth orientation rules between identical parts**: In `rolled_strip_base_OK_S0182_...`, the generator said holes were on left/right, and the judge ruled: *"The mounting holes are located on the top and bottom of the central square, not left and right."* Yet in `rolled_strip_base_OK_S0467_...`, the generator said left/right and the judge also insisted *"located at the top and bottom"*, while across multiple other identical normal samples (e.g., `rolled_strip_base_OK_S0027_...`, `rolled_strip_base_OK_S0038_...`, `rolled_strip_base_OK_S0481_...`), the judge flipped orientation and asserted the exact opposite: *"The mounting holes/recesses are located at the middle-left and middle-right sides, not near the top corners"*.
2. **Pedantic over-splitting on speculative reasoning**: In `rolled_strip_base_NG_ZW_S0151_...`, the generator stated: *"It could be carbon residue, a speck of dirt, or burned material..."*. The judge flagged this under `invented_defect` (*"The defect is a blue ink/dye stain/mark, not carbon residue..."*), penalizing the model's exploratory hypothesis generation despite the trace correctly concluding generic `Contamination`.

## 4 Rewrite quality
Rewrites reliably preserve tags, locations, and verdicts (`Yes`/`No`), fixing factual tokens directly. However, rewrites occasionally leave residual logic mismatches.
- **Clean repair**: In `rolled_strip_base_NG_ZW_S0029_...`, replacing *"small dark speck"* with *"red spot"* directly repaired the description while keeping `<location>top-center</location>`, `<type>Contamination</type>`, `<answer>Yes</answer>`.
- **Incomplete / incongruous repair**: In `rolled_strip_base_NG_ZW_S0056_...`, the rewrite replaced the hole location with *"middle-right area between the diagonal ribs"*, but left untouched the earlier generator claim *"I observe the top and bottom circular depressions"*, resulting in an awkward trace narrative where features shift without visual continuity.

## 5 Risks for a thesis that publishes this corpus
- **Orientation ambiguities & rotational invariance**: Traces across symmetric square mounts arbitrarily define 90° rotations as "top/bottom" vs "left/right", causing models trained on this corpus to learn contradictory spatial reasoning for canonical CAD features (e.g. `rolled_strip_base_OK_S0182_...` vs `rolled_strip_base_OK_S0481_...`).
- **Confirmation bias / 'Right for the wrong reasons'**: On defective samples, generators routinely infer the correct gold label while hallucinating the core physical phenomenon—e.g. guessing generic molding carbon inclusion when the ground-truth defect is external dye/ink (`rolled_strip_base_NG_ZW_S0001_...`, `rolled_strip_base_NG_ZW_S0119_...`, `rolled_strip_base_NG_ZW_S0151_...`). Training on uncorrected traces rewards hallucinated visual evidence.
