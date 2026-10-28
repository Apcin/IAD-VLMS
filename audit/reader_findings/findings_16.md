# Chunk 16 — chunk_16.txt

## 1 Counts
- **Total traces:** 24
- **By split:** `grpo_4k`: 24
- **By category:** `wrong`: 18, `completely_wrong`: 6
- **Issue types:**
  - `colour_finish_error`: 19
  - `location_error`: 19
  - `invented_defect`: 15
  - `viewpoint_error`: 10
  - `count_error`: 9
  - `feature_absent`: 4
  - `defect_type_error`: 4
  - `marking_text_error`: 3
  - `missed_defect`: 2

## 2 Generator failure modes
1. **Inventing a missing terminal when the actual defect is housing/tab damage** (~7 traces).
   - Claim: *"I notice an issue with the arrangement of the terminals. The terminal on the right... appears to be missing."* (`switch_NG_QS_S0080`)
   - Judge why-wrong: *"The terminals on the left are present... whereas the missing part is the actuator on the right."*
2. **Hallucinating dark specks/smudges on black plastic instead of bright/white/colored contamination** (~10 traces).
   - Claim: *"These particles are visible as tiny dark specks, clustered together."* (`switch_NG_ZW_S0013`)
   - Judge why-wrong: *"The contamination on the black housing consists of light/pinkish-white paint or foreign material, not dark specks."*
3. **Misidentifying side/perspective profile view as direct front or rear view** (~7 traces).
   - Claim: *"This view is a direct front-on shot, showing the entire red rocker actuator..."* (`switch_NG_ZW_S0028`)
   - Judge why-wrong: *"The image is a side/perspective profile view showing the side body and terminal clearly, not a direct front-on shot."*
4. **Over-counting terminal tabs on single-view switch images** (~6 traces).
   - Claim: *"There are three terminals visible in this orientation."* (`switch_NG_QS_S0136`)
   - Judge why-wrong: *"Only two metallic terminals/tabs are visible extending from the left side of the switch body."*

## 3 Judge mistakes
1. **Overly pedantic marking text flag in `switch_OK_S0425`**: Flagged `0 and 1` as wrong because standard symbols are `O` and `-`. Industrial rocker markings `O`/`I` are universally read as 0/1; calling this an error is needlessly strict.
2. **Nitpicking `switch_NG_ZW_S0073`**: Flagged *"dark or brownish speck"* on contamination, forcing *"white smear or residue"*, when small contamination flakes under direct lighting often exhibit dark/brownish specular shadows.
3. **`switch_NG_ZW_S0014`**: The judge flags the generator for examining the side wall, claiming contamination is *"directly on the red actuator button"*, despite the gold annotation covering a broad `middle-right` region where both bezel and rocker meet.

## 4 Rewrite quality
Rewrites successfully preserve the final `<answer>` and `<type>` verdicts without drifting into contradictions, but sometimes add overly detailed grounding specifics not visible in standard industrial images.
- **Good fix (`switch_NG_QS_S0095`)**: Replaces the fabricated missing terminal claim with the exact broken tip geometry (*"only the base and a partial semicircle of the hole remain"*), keeping `<type>Missing Parts</type>` intact.
- **Introduced unsupported/unverified content (`switch_NG_ZW_S0166`)**: The judge rewrite changes the trace to state *"there are noticeable scratches and surface defects across the red button face"* for a sample annotated as `Missing Parts`. It introduces an unverified defect type not aligned with the dataset annotation.

## 5 Risks for a thesis that publishes this corpus
- **Invented defects on normal items:** Models hallucinate non-existent pins or misread profiles, creating fragile reasoning chains (e.g., `switch_OK_S0242` where the trace relies on verifying 3 spade terminals when only 1 is visible).
- **Label & grounding mismatches:** Trace in `switch_NG_ZW_S0166` discusses contamination/scratches despite gold label being `Missing Parts`, risking training models to justify correct binary labels with completely false causal reasoning.
- **Viewpoint/Orientation hallucinations:** Traces repeatedly assert frontal perspectives on profile views (e.g., `switch_OK_S0067`, `switch_OK_S0315`), which will degrade spatial-reasoning benchmarks.
- **Overlay leaks / Prompt biases:** Traces frequently guess generic defect classes matching dataset labels (like defaulting to "terminal missing" for any `Missing Parts` label, seen in `switch_NG_QS_S0109` and `switch_NG_QS_S0170`).
