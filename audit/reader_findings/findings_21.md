# Chunk 21 — chunk_21.txt

## 1 Counts
* **Total traces:** 24 (all product=`toy`).
* **By split:** `grpo_4k`: 24.
* **By category:** `wrong`: 19, `completely_wrong`: 5.
* **By gold label:** `yes` (defective): 10, `no` (normal): 14.
* **Issue type distribution (54 total flags across 24 traces):**
  * `feature_absent`: 22
  * `location_error`: 16
  * `marking_text_error`: 9
  * `defect_type_error`: 8
  * `invented_defect`: 5
  * `colour_finish_error`: 3
  * `other`: 1
  * `viewpoint_error`: 1

## 2 Generator failure modes
1. **Hallucination of non-existent features (specifically vehicle wheels)** (~10 traces)
   * *Claim:* "I can see a portion of the bottom of the toy, which seems to have a white wheel attached." (`toy_OK_S0017_...`)
   * *Judge:* "No bottom wheel is visible in this top-down image of the toy burger."
2. **Defect misattribution to sesame seeds instead of outer perimeter rim** (~6 traces)
   * *Claim:* "Instead of a white dot, the orange surface of the bun is visible. This indicates that a sesame seed detail is missing from this location." (`toy_NG_QS_S0110_...`)
   * *Judge:* "The sesame seeds are normally distributed and none is missing; the actual defect is missing material along the top outer edge/lettuce layer."
3. **Misreading embossed text / serial markings** (~8 traces)
   * *Claim:* "The CCC logo and the numbers 'O 001103' are visible." (`toy_NG_QS_S0041_...`)
   * *Judge:* "The printed numbers are 'O 007703' (upside down), not 'O 001103'."
4. **Spatial orientation and directional inversion errors** (~13 traces)
   * *Claim:* "In the upper-left area, the CCC logo and the part number are impressed into the plastic." (`toy_OK_S0189_...`)
   * *Judge:* "The CCC logo and part number are located on the right side (middle-right), not in the upper-left area."

## 3 Judge mistakes
1. **`toy_OK_S0245_toy_0245_OK_C1_20230927112929`**: The judge tagged the claim `"A small portion of a white wheel is visible beneath the lower structure."` as `[invented_defect]`. This is an absent feature hallucination on a normal sample (`gold_label=no`), not an invented defect claim; the appropriate tag was `feature_absent`.
2. **`toy_NG_ZW_S0024_toy_0024_NG_ZW_C1_20230927184453`**: The judge flagged `<location>center</location>` as a `location_error` issue and changed the final XML tag to `<location>middle-right</location>`. Rewriting tag targets changes metadata rather than strictly evaluating grounded visual reasoning trace text.

## 4 Rewrite quality
Rewrites generally excise hallucinations and fix orientation/text while preserving the verdict, but occasionally leave awkward synthetic artifacts:
* **Clean fix:** In `toy_OK_S0036_toy_0036_OK_C1_20230927103558`, the judge smoothly excised the hallucinated white wheel sentence and replaced it with `"Finally, I note the outer structure around the bottom edge, which appears clean and properly molded without significant flashing."` while leaving the `<answer>No</answer>` unchanged.
* **Unbalanced patch:** In `toy_NG_ZW_S0024_toy_0024_NG_ZW_C1_20230927184453`, the judge rewrote internal reasoning to cite both `middle-right` and `top-left` defects, but modified the `<location>` tag to only `<location>middle-right</location>`, discarding the multi-location target (`gold_location=top-left, middle-right`).

## 5 Risks for a thesis that publishes this corpus
1. **Right verdict for entirely wrong reasons:** Traces arrive at correct labels purely through hallucinated flaws (e.g., claiming missing sesame seeds when the defect is a chunk missing from the lettuce rim in `toy_NG_QS_S0072_...`, `toy_NG_QS_S0076_...`, and `toy_NG_QS_S0112_...`). Training models on these traces rewards ungrounded reasoning.
2. **Object prior hallucinations:** Severe prior leakage across 10 normal samples (`toy_OK_S0017`, `toy_OK_S0033`, `toy_OK_S0063`, `toy_OK_S0075`, `toy_OK_S0088`, `toy_OK_S0105`, `toy_OK_S0147`, `toy_OK_S0160`, `toy_OK_S0169`, `toy_OK_S0193`, `toy_OK_S0204`, `toy_OK_S0205`, `toy_OK_S0210`, `toy_OK_S0211`, `toy_OK_S0293`, `toy_OK_S0341`, `toy_OK_S0356`, `toy_OK_S0369`) hallucinating a rolling "white wheel" mechanism not present in top-down views.
3. **Reference image / leakage hints:** In `toy_NG_QS_S0041_...`, the judge notes the generator failed because the defect is missing compared to the golden reference, highlighting that the generator is blind to comparative baseline geometry without seeing a reference pair.
