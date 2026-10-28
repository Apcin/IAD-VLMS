# Chunk 18 — chunk_18.txt

## 1 Counts
- **Total Traces**: 25 (all `product=terminalblock`)
- **Split**: `grpo_4k`: 25
- **Category Distribution**: `wrong`: 14, `completely_wrong`: 11
- **Issue Type Distribution**: 
  - `count_error`: 27
  - `invented_defect`: 19
  - `location_error`: 19
  - `defect_type_error`: 9
  - `missed_defect`: 6
  - `viewpoint_error`: 6
  - `feature_absent`: 5
  - `colour_finish_error`: 2
  - `other`: 1

## 2 Generator failure modes
1. **Systematic 5 vs 6 Element Miscount**: Hallucinates 5 terminal screws/holes instead of 6 (~11 traces).
   - *Claim*: `"showing the five screw heads that are used to tighten down the wire clamps."` (`...S0118...`)
   - *Why Wrong*: `"There are 6 screw heads visible in the image, not 5."`
2. **Hallucinating Missing Screws Instead of Plastic Chipping**: Reports a missing metal screw when the gold "Missing Parts" defect is actually chipped orange housing plastic (~10 traces).
   - *Claim*: `"Where the metal terminal and screw should be located, I can only see the orange plastic housing."` (`...S0090...`)
   - *Why Wrong*: `"All six metal screw terminals are fully present in their holes."`
3. **Misplacing Surface Defects onto Metal/Internal Features**: Projects housing pits/stains onto internal wire holes or screw recesses (~7 traces).
   - *Claim*: `"This pit is located on the inner surface of the wire entry hole."` (`...S0110...`)
   - *Why Wrong*: `"The pit is located on the flat orange plastic housing surface just above the fourth screw hole, not on the inner surface of a wire entry hole."`
4. **Viewpoint and Coordinate Disorientation**: Misidentifies orientation as vertical stacks rather than horizontal rows (~4 traces).
   - *Claim*: `"This view shows the side with the wire insertion openings, stacked vertically."` (`...S0045...`)
   - *Why Wrong*: `"The image shows the front face with 6 screw terminals arranged horizontally, not wire insertion openings stacked vertically."`

## 3 Judge mistakes
1. **`...S0056...`**: Judge flagged `<type>Missing Parts</type>` as `defect_type_error`, stating: *"Although the dataset gold label is Missing Parts / broken plastic under position 3, the trace hallucinated a missing 6th screw."* However, the predicted tag `<type>Missing Parts</type>` exactly matched `gold_type=Missing Parts`, making a tag-level issue flag inappropriate.
2. **`...S0151...`**: Judge flagged a benign analytical remark about insulation as `[other]`: *"The presence of such a pit could potentially affect the electrical contact..."* The judge treated standard chain-of-thought impact speculation as a factual visual falsehood.

## 4 Rewrite quality
- **Verdict preservation**: The rewrites successfully fix factual hallucinations without altering final tags/verdicts.
- **Unsupported content**: Rewrites sometimes invent specific physical damage mechanisms not verified in the input prompt:
  1. `...S0154...`: Introduces detailed visual semantics: *"ragged, stress-whitened fractured edges where plastic material is absent."*
  2. `...S0001...`: Injects an unprompted geometric descriptor: *"A V-shaped section of the plastic wall is broken off and missing..."*

## 5 Risks for a thesis that publishes this corpus
- **Pervasive False Reasoning for Correct Answers (Right for Wrong Reason)**: The generator repeatedly hits the correct gold label (`Missing Parts`, `Yes`) by inventing absent screws instead of detecting cracked plastic frames (`...S0001...`, `...S0041...`, `...S0042...`, `...S0049...`, `...S0051...`, `...S0070...`, `...S0072...`, `...S0089...`, `...S0091...`, `...S0107...`, `...S0126...`, `...S0128...`, `...S0130...`).
- **Hallucinated Normal Inspections**: Normal parts (`gold_label=no`) are validated through fabricated element counts (`...S0152...`, `...S0035...`, `...S0065...`, `...S0156...`), generating untrustworthy negative reasoning.
- **Tag Discrepancies**: Trace `...S0157...` initially had `tags_ok=False` due to outputting `<location>middle-right</location>` when gold was `center`, indicating generator misalignment with spatial ground truth.
