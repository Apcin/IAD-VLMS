# Chunk 26 — chunk_26.txt

## 1 Counts
- **Traces:** 35 total (all `usb_adaptor`).
- **Splits:** `grpo_4k`: 17, `sft_6k`: 18.
- **Categories:** `wrong`: 31, `completely_wrong`: 4.
- **Issue types:** `location_error`: 27, `viewpoint_error`: 22, `feature_absent`: 17, `defect_type_error`: 16, `overlay_leak`: 13, `invented_defect`: 10, `colour_finish_error`: 8 (Total = 113).

## 2 Generator failure modes
1. **Hallucinating invisible internal interfaces from side-profile views** (~16 traces): Generator hallucinates inside female USB-A ports (blue plastic tongue, contact pins).
   - Claim: `"Looking inside the port, the internal blue plastic insulator is visible."` (`usb_adaptor_NG_ZW_S0008`)
   - Judge why-wrong: `"The interior of the female port and any blue plastic insulator are not visible in this side/top-down view."`
2. **Defect misattribution to main housing instead of metallic connector** (~11 traces): Defects on gold Type-C plug are hallucinated as spots on black housing.
   - Claim: `"As I scan towards the middle-left area of the black housing, something catches my eye. There's a distinct spot..."` (`usb_adaptor_NG_ZW_S0096`)
   - Judge why-wrong: `"The spot/contamination is on the gold metal USB connector, not on the black housing."`
3. **Severe ground-truth overlay leakage in reasoning text** (~12 traces): Generator sees annotation masks and cites red/copper colors instead of actual physical defects.
   - Claim: `"In the center region of the black housing, I see a small, pinpoint red mark."` (`usb_adaptor_NG_AK_S0030`)
   - Judge why-wrong: `"The red mark comes entirely from the overlay mask; in the query image it is a small dark surface pit/depression."`
4. **Defect type confabulation** (~10 traces): Surface pits/indents labeled as contamination/stains due to color-tint hallucinations.
   - Claim: `"<type>Contamination</type>"` (`usb_adaptor_NG_AK_S0015`)
   - Judge why-wrong: `"The defect is a pit (small indentation/surface void), not foreign material or paint."`

## 3 Judge mistakes
1. `usb_adaptor_NG_HS_S0017`: The judge flags `bottom-left` as wrong and demands `bottom-center` for both claim and `<location>` tag (`"The defect region is bottom-center, not bottom-left"`), ignoring that the defect sits along the bottom transitional edge near the left plug where `bottom-left` is an acceptable grid approximation.
2. `usb_adaptor_NG_ZW_S0042`: The judge flags the description as a `defect_type_error` rather than an `overlay_leak`, yet inserts `"red contamination"` into `SHOULD BE`, despite red being the standard overlay artifact on Real-IAD masks.
3. `usb_adaptor_OK_S0276`: The judge flags `"There are two parallel grooves machined into the side"` as `feature_absent`, but other traces on the same component describe subtle horizontal extrusion/machining lines as expected design texturing.

## 4 Rewrite quality
Rewrites successfully preserve gold verdicts (`Yes`/`No`) and fix hallucinated internal geometries, but occasionally retain uncorrected defects or introduce subtle discrepancies:
- **Clean fix:** In `usb_adaptor_OK_S0115`, the rewrite replaces detailed hallucinations of the internal blue plastic insert and pins with an exterior assessment of the male Type-C connector while retaining the `No` verdict.
- **Imperfect fix / leftover artifact:** In `usb_adaptor_NG_HS_S0007`, the judge removes the red overlay leak, but replaces it with `"there is a small white/light scratch or nick"`, inferring an unverified white color to substitute for the removed mask artifact.

## 5 Risks for a thesis that publishes this corpus
- **Overlay leaks invalidate visual grounding:** Traces directly describe synthetic annotation masks (`"bright red speck"` in `usb_adaptor_NG_AK_S0015`, `"pinpoint red mark"` in `usb_adaptor_NG_AK_S0030`, `"bright red, thin linear mark"` in `usb_adaptor_NG_HS_S0052`). Using these models for evaluation measures mask detection rather than industrial inspection.
- **Pervasive confabulation on normal items:** Models hallucinate intricate sub-features (e.g., `usb_adaptor_OK_S0079` describing internal pins inside an obscured port cavity), creating false training signal.
- **Label & tag mismatches:** Model outputs arbitrary defect tags that contradict actual labels (e.g., `Burr` instead of `Pit` in `usb_adaptor_NG_AK_S0061`; `Scratch` instead of `Contamination` in `usb_adaptor_NG_ZW_S0043`).
