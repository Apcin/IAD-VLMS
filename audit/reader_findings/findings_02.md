# Chunk 02 — chunk_02.txt

## 1 Counts
- **Total Traces**: 36
- **Traces by Split**: `grpo_4k`: 18, `sft_6k`: 18
- **Traces by Category**: `wrong`: 34, `completely_wrong`: 2
- **Issue Type Distribution**:
  - `defect_type_error`: 27
  - `marking_text_error`: 24
  - `location_error`: 18
  - `overlay_leak`: 9
  - `colour_finish_error`: 7
  - `viewpoint_error`: 6
  - `feature_absent`: 2
  - `missed_defect`: 2
  - `verdict_mismatch`: 1
  - `not_visible_in_view`: 1
  - `other`: 1

## 2 Generator failure modes
1. **Misreading Text & OCR Hallucination (Coin Cells)** (~14 traces): Generator hallucinates garbled or mirrored text (`CK5035`, `CB5035`, `MUHTIU`) instead of `CR2032` and `LITHIUM`.
   - *Claim*: `"The markings on the surface, including reversed text like 'CB5035' and logos..."`
   - *Judge*: `"The visible text marking on the battery is 'CR2032', 'LITHIUM +', and 'BATTERY 3V', not 'CB5035'."` (`button_battery_OK_S0014...`)
2. **Overlay Leak / Artifact Hallucination** (~7 traces): Generator mistake synthetic red inspection overlay annotations for real-world defects (red ink, paint, or red plastic).
   - *Claim*: `"...there is a distinct red speck of contamination. This small, bright red particle is sitting on the surface..."`
   - *Judge*: `"The red dot is the defect annotation mask from the overlay image, not a physical red particle on the battery."` (`button_battery_NG_AK_S0027...`)
3. **Defect Type Substitution (Pits vs. Contamination)** (~10 traces): Generator consistently interprets geometric void pits/depressions as particulate contamination.
   - *Claim*: `"...I identify a distinct, dark red or brown foreign inclusion embedded in the plastic material itself."`
   - *Judge*: `"The defect is a hole / pit (puncture/void through the plastic showing the dark background), not a foreign inclusion or contamination."` (`bottle_cap_NG_AK_S0019...`)
4. **Viewpoint Inversion** (~5 traces): Generator misidentifies direct planar top-down images as side-profile or angled perspectives.
   - *Claim*: `"The image displays a side view of the component, showing its metallic edge band and a part of the black top face..."`
   - *Judge*: `"The image is an angled top/face view looking directly down onto the black circular top face of the coin cell battery."` (`button_battery_NG_AK_S0072...`)

## 3 Judge mistakes
1. `bottle_cap_NG_ZW_S0073...`: Judge flags text describing a dented perimeter as a `defect_type_error` claiming it is a *"red contamination mark"*, injecting the artificial red overlay into its own ground truth critique: `"The defect on the middle-left edge is a red contamination mark, not a deformed or flattened edge."`
2. `bottle_cap_NG_HS_S0108...`: Over-strict flag on location tag. The scratch extends across the inner radius from center toward middle-left. The judge concedes `<location>bottom-left</location> is appropriate` but rejects `<location>middle-left</location>` pedantically based strictly on the metadata string.

## 4 Rewrite quality
- Rewrites generally fix identified grounding claims without flipping the verdict:
  - *Example 1* (`button_battery_NG_AK_S0003...`): Replaces hallucinated contamination claims with proper pit descriptions while retaining the `<answer>Yes</answer>` verdict and adjusting metadata tags cleanly.
- However, rewrites occasionally delete or leave empty content:
  - *Example 2* (`bottle_cap_NG_ZW_S0103...`): Categorized as `completely_wrong`, the judge produces a completely blank rewrite (`--- GEMINI REWRITE ---` followed by nothing), rendering the item unusable.

## 5 Risks for a thesis that publishes this corpus
- **Overlay Leaks**: Traces frequently hallucinate physical red pigment from annotation masks (`bottle_cap_NG_ZW_S0014...`, `button_battery_NG_AK_S0013...`, `button_battery_NG_AK_S0027...`), proving ungrounded synthetic training contamination.
- **Verdict Mismatches & False Negatives**: Models classify defective items as defect-free (`button_battery_NG_AK_S0072...` predicts `<answer>No</answer>` against gold `Yes`).
- **Pervasive OCR Hallucination**: Training on button batteries propagates ungrounded string hallucinations like `CB5035` and `MUHTIU` (`button_battery_OK_S0192...`, `button_battery_NG_HS_S0081...`) to downstream models.
