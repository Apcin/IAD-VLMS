# Chunk 08 — chunk_08.txt

## 1 Counts
* **Total traces:** 22
* **By split:** `grpo_4k`: 22 (100%)
* **By category:** `wrong`: 21, `completely_wrong`: 1 (`pcb_NG_ZW_S0079`)
* **Issue types (53 total flags):**
  * `location_error`: 34
  * `feature_absent`: 7
  * `colour_finish_error`: 6
  * `invented_defect`: 3
  * `count_error`: 2
  * `defect_type_error`: 1

---

## 2 Generator failure modes

### Mode 1: Component & spatial hallucination (orientation confusion)
The generator repeatedly misplaces fixed board components (e.g., claiming USB is top/right, IC U1 is below switch or 8-pin, B+/B- and M+/M- transposed).
* **Count:** ~16 traces
* **Trace `pcb_OK_S0062`**:
  * *Claim:* "I can identify a micro USB connector at the top"
  * *Why wrong:* "The micro USB connector is located on the left side of the PCB, not at the top."
* **Trace `pcb_OK_S0114`**:
  * *Claim:* "The associated integrated circuit (IC) labeled U1 is visible just below the switch."
  * *Why wrong:* "IC U1 is located above the tactile switch, not below it."

### Mode 2: Fabrication of SMD features (through-hole resistor color bands)
The model regularly hallucinates colored resistance bands on standard rectangular SMD 0603/0805 chip resistors.
* **Count:** ~4 traces
* **Trace `pcb_OK_S0079`**:
  * *Claim:* "The color bands on R1 are clear and distinct, indicating the resistance value is identifiable."
  * *Why wrong:* "R1 is a surface-mount chip resistor (SMD), which does not have color bands."
* **Trace `pcb_OK_S0105`**:
  * *Claim:* "The colored bands on the resistor, which indicate its resistance value, are clearly visible, distinct, and legible."
  * *Why wrong:* "The resistor R1 is an SMD chip resistor with a black rectangular package, not an axial resistor with colored bands."

### Mode 3: Defect fabrication / hallucinated mechanism to match gold label
When detecting an anomaly, the model fabricates elaborate narratives (e.g., massive liquid smears, missing parts) instead of grounding the actual visual mark.
* **Count:** ~5 traces
* **Trace `pcb_NG_ZW_S0079`**:
  * *Claim:* "There is a distinct gap where a component or a connection should be... This is a clear case of missing parts."
  * *Why wrong:* "The board has no missing parts; M+ and M- are solder pads and all components match reference normal."
* **Trace `pcb_NG_ZW_S0018`**:
  * *Claim:* "I see a distinct, pale, hazy residue spread across the blue solder mask... whitish, semi-transparent film..."
  * *Why wrong:* "The contamination in the top-center is a distinct black/dark fiber/particle protruding over the top edge/pad area, not a pale whitish hazy film."

---

## 3 Judge mistakes
1. **`pcb_NG_ZW_S0024`**: Judge over-penalizes initial description as `location_error`. Claim: *"The metallic housing of the connector appears clean and free from any dents, scratches, or corrosion."* Why wrong: *"The metallic housing of the micro-USB connector actually contains the dark contamination mark..."* In standard chain-of-thought, an inspector evaluates mechanical integrity (dents/scratches) before inspecting for surface contamination; calling this an outright error conflates structural inspection with cleanliness check.
2. **`pcb_OK_S0175`**: Judge flags marking text `DACJ5` as `marking_text_error`, asserting *"The IC labeled U1 has no visible 'DACJ5' text printed on its top surface"*, yet later in `pcb_OK_S0314` the judge allows `"U1/DACJ3"`. On low-contrast laser etchings on SOT-23 packages, reading laser top-markings is plausible image resolution interpretation rather than pure hallucination.

---

## 4 Rewrite quality
* **Verdict preservation:** Rewrites consistently fix factual/spatial hallucination while strictly preserving gold verdicts (`Yes` or `No`) and defect types.
* **Example 1 (Clean correction):** In `pcb_OK_S0138`, the rewrite adjusts pin counts from 8 to 6 seamlessly: *"U1. It's a 6-pin package... I'm looking closely at each of the six pins."*
* **Example 2 (Unsupported content injected):** In `pcb_NG_ZW_S0006`, the judge changes the original text describing clean edges to: *"The board edges show dark marks, and the light blue perimeter outline has contamination."* This injects defect details into the introductory general sweep paragraph where a real human/model would not yet have isolated the anomaly, front-loading the conclusion.

---

## 5 Risks for a thesis that publishes this corpus
1. **Invented defects on normal items:** Models hallucinating unpopulated components (`pcb_OK_S0137`, `pcb_OK_S0148`: claiming U1 is an unpopulated footprint) could train downstream systems to excuse missing parts as "intended variants."
2. **Gold tag / verdict mismatches:** `pcb_NG_ZW_S0079` outputs `<type>Missing Parts</type>` despite gold label being `Contamination`. Relying on unedited traces produces catastrophic reasoning misalignment during fine-tuning.
3. **Multi-defect location truncations:** In multi-label items (`pcb_NG_ZW_S0006`, `pcb_NG_ZW_S0038`), traces select only one location tag (`center` or `middle-right`) instead of multi-label ground truth (`bottom-right, top-center`), teaching models poor localization fidelity.
