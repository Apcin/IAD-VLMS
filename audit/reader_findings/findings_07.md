# Chunk 07 — chunk_07.txt

## 1 Counts
- **Total traces:** 25
- **By split:** `grpo_4k` (25)
- **By category:** `wrong` (21), `completely_wrong` (4)
- **Issue type distribution (66 total flags):**
  - `location_error`: 31
  - `colour_finish_error`: 13
  - `defect_type_error`: 10
  - `invented_defect`: 7
  - `feature_absent`: 5
  - `contradiction`: 1
  - `verdict_mismatch`: 1
  - `count_error`: 1
  - `viewpoint_error`: 2
  - `marking_text_error`: 2
  - `missed_defect`: 1
  - `other`: 1

## 2 Generator failure modes
1. **Misattributing component defects to the PCB solder mask/substrate** (~11 traces)
   - Claim: *"There is a distinct line running diagonally across the blue solder mask in this central region."* (`pcb_NG_HS_S0084...`)
   - Judge: *"The scratch is located on the black plastic housing of push button switch K1, not on the blue solder mask."*
2. **Hallucinating empty SMD pads for non-pad structural/component defects** (~7 traces)
   - Claim: *"Looking at the location marked 'C3', I can see the pads for this component, but there is no component present on these pads."* (`pcb_NG_QS_S0042...`)
   - Judge: *"Component C3 is present on the board (to the right of U1), while the actual defect is damage/missing material or a defect on the center push button actuator."*
3. **Severe spatial orientation inversion (top vs. bottom, left vs. right)** (~14 traces)
   - Claim: *"In the bottom section, I see connection points B+, B-, and a circular pad."* (`pcb_NG_HS_S0019...`)
   - Judge: *"Connection points B+, B-, and the circular pad/hole are located at the top section, not the bottom."*
4. **Color/material hallucination on foreign particles (defaulting to 'dark speck')** (~7 traces)
   - Claim: *"It's a small, dark speck."* (`pcb_NG_YW_S0047...`)
   - Judge: *"The foreign object in the bottom-center region is a thin, shiny metallic/golden sliver, not a small dark speck."*

## 3 Judge mistakes
1. **`pcb_NG_QS_S0073...`**: Judge flagged `defect_type_error` and rewrote the defect as bent metal on the USB connector under gold label `Missing Parts`. Calling physical connector deformation `Missing Parts` is logically broken.
2. **`pcb_NG_HS_S0095...`**: Judge flagged `colour_finish_error` for claiming the scratch runs "horizontally" vs "diagonally". Trajectory angle is a geometric orientation disagreement, not a color/finish issue.

## 4 Rewrite quality
- Rewrites successfully correct grounding errors without altering target gold labels or final predictions.
- **Example 1 (`pcb_NG_QS_S0087...`):** Fixes a false positive verdict mismatch by flipping `<answer>Yes</answer>` to `<answer>No</answer>` and replacing the hallucinated particle description with a grounded confirmation of clean substrate.
- **Example 2 (`pcb_NG_HS_S0057...`):** Accurately relocates the scratch from the blue solder mask near C1 to the metallic USB shield without introducing unsupported details.

## 5 Risks for a thesis that publishes this corpus
- **Gold label / Class mismatch:** Broken/missing physical substrate corners (`pcb_NG_QS_S0001...`, `pcb_NG_QS_S0063...`, `pcb_NG_QS_S0068...`) or damaged button actuators (`pcb_NG_QS_S0042...`) are labeled `Missing Parts`, training models to confuse missing electrical SMDs with mechanical chipping.
- **Invented defects on normal items:** Trace `pcb_NG_QS_S0087...` invents a black contaminant speck on a clean negative sample, yielding an incorrect `Yes` verdict.
- **Overlay/Prompt leaks:** Trace `pcb_NG_QS_S0068...` explicitly states: *"However, the hint points to the bottom-left. Let me re-examine the bottom-left area..."*, proving external supervision leaked directly into the generator's reasoning trace. Trace `pcb_NG_YW_S0013...` similarly leaks: *"The hint suggests I should pay close attention to foreign objects, especially in the top-center area."*
