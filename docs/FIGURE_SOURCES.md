# Figure sources: which file draws which thesis figure

Written 9 Sep 2026 after two figure-versus-text mismatches were found in Chapter 3
(Figure 3.2 listed eleven "hard rules" with invented names, Figure 3.1 still said "skipped or
relabelled normal"). This ledger says, for every figure in the thesis, where the live source is,
how the PDF was produced, and what the figure must agree with. Update it whenever a figure or the
text it depends on changes.

Rules that would have prevented both mismatches:

1. A figure that quotes a list from the prompt or the code (rule names, reward bins, split counts)
   copies it from the source file, not from memory. Sources: `prompts/inspector_prompt_test_v2.txt`
   (hard rules, filter, auto-reject rules), `results/thesis_figure_data/figdata.json` (numbers),
   `scripts/03_grpo/` (reward bins).
2. When a sentence in the text changes a fact that a figure also states (for example the relabel
   rule of section 3.3), grep the SVG sources for the old wording before the commit:
   `grep -l "old wording" scripts/05_figures/thesis_figures_v2/svg_final/**/*.svg`.
3. Before installing a figure, send the render to Gemini 3.8 Flash with the claims it must satisfy
   and ask for CONFIRMED / NOT CONFIRMED with quotes. Gemini catches wrong text; it does not know
   the thesis, so give it the claims.
4. Never trust file timestamps on `/bulk` to decide which copy is newest. A keep-alive watchdog
   touches every file. Compare content instead (the text of the installed PDF against the SVG).

## Hand-drawn SVGs (exported to PDF by Adnane, Verdana embedded)

These are drawn as SVG, then exported to PDF on Adnane's machine because the render box has no
Verdana. cairosvg previews substitute Noto Sans, which is narrower, so a text that fits in a
preview can overflow in the real export. Keep about eight percent slack.

| Thesis figure | Installed file | Live source (this repo) | Local working copy | Must agree with |
|---|---|---|---|---|
| 1.1 method overview | `figures/overview_big.pdf` | `scripts/05_figures/thesis_figures_v2/svg_final/method_overview_v4.svg` | `all_svg/method_overview_v4.svg` (v6 export installed 8 Sep) | Tables 6.2 and 6.14, split counts of section 3.5 |
| 3.1 source and payload | `figures/data_source_payload.pdf` | `svg_final/chapter3/data_source_payload.svg` | `all_svg/anomalythink_source_payload_v3.svg` | section 3.3 relabel rule, datasheet App. I |
| 3.2 generation | `figures/data_generation.pdf` | `svg_final/chapter3/data_generation.svg` | `all_svg/anomalythink_generation_v6.svg` | the eleven HARD RULES, the six auto-reject rules and the 13/16 filter in `prompts/inspector_prompt_test_v2.txt` (App. B) |
| 3.3 instance and splits | `figures/data_instance_splits.pdf` | `svg_final/chapter3/data_instance_splits.svg` | `all_svg/anomalythink_instance_splits_filled.svg` | split counts of section 3.5 (6,000 = 3,000 + 3,000; 4,236 = 2,118 + 2,118; 4,236 = 2,129 + 2,107) |
| 4.1 SFT and GRPO stages | `figures/sft_grpo_stages.pdf` | `svg_final/sft_grpo_stages.svg` (generator `stages/f7_stages.py`) | `thesis_figures_v2/stages/fig_stages.svg` | Table 4.1 hyper-parameters, section 5.4 reward |
| 5.x KCR pipeline | `figures/rl_correction_pipeline.pdf` | `svg_final/rl_correction_pipeline.svg` | `all_svg/rl_correction_pipeline.svg` | keep / correct / rewrite counts of section 5.7 |

Older variants with " (1)", " (2)" suffixes in `all_svg/` are superseded drafts. The live copy is
the one whose text matches the installed PDF (`pdftotext figures/<file>.pdf -`). For Chapter 3
the live copies on 9 Sep 2026 were `anomalythink_generation (5).svg`,
`anomalythink_source_payload_filled (2).svg` and `anomalythink_instance_splits_filled.svg`; the
edited versions are `_v6` and `_v3`.

Edits made 9 Sep 2026, awaiting Adnane's PDF export:

- 3.1: yellow box "NG, defect not visible at C1 -> skipped or relabelled normal" became
  "-> relabelled normal" (matches section 3.3 and the datasheet).
- 3.2: the Phase 2 box now lists the eleven HARD RULES of the prompt by their real names
  (Single-image independence, No metadata leakage, Evidence before conclusion, Do not invent
  unseen history, Natural, controlled reasoning, Reasoning length 120 to 200 words, Use the
  product type immediately, Reference image usage, Region cue usage, Normal cases must still be
  specific, Ambiguity handling). The old box had eight invented names. Title now says
  "the eleven hard rules (App. B)".
- 3.2: the top strip now says "phases 1 to 4 happen inside this one call, phase 5 runs afterwards
  in Python", and the right box is titled "Phase 5: after the call, dataset processing", because
  the box sits inside the frame titled "one Gemini 2.5-Flash call per batch".

Known, accepted: 3.3 colours the SFT split blue while Figures 1.1 and 4.1 colour SFT green.
Blue reads as "data" in 1.1, and the split is data, so it stays unless Adnane wants it changed.
All three Chapter 3 figures carry a burned-in footer sentence that duplicates the LaTeX caption.

## Generated figures (python, house style, `svgkit.py`)

All in `scripts/05_figures/thesis_figures_v2/`. Each script writes an SVG next to itself and,
via `thesisify.py` or its own cairosvg call, the PDF into the thesis `figures/` folder. Widths are
measured with Noto Sans at the class pixel sizes because that is what cairosvg renders here.

| Thesis figure | Installed file | Generator | Data |
|---|---|---|---|
| 3.4 anomalous trace example | `figures/trace_example_woodstick.pdf` | `f10_trace_example.py woodstick_0028_NG_ZW` | `Training/datasets_small_new_v4/combined_6k_train.json` (6K SFT split), Real-IAD masks |
| 3.5 normal trace example | `figures/trace_example_transistor1.pdf` | `f10_trace_example.py transistor1_0234_OK` (normal mode) | same 6K split |
| 5.2 t-SNE of type strings | `figures/tsne_anomaly_types.png` | `f11_tsne_reward_path.py` (reward path, 85 strings; the old search_document figure is kept as `tsne_anomaly_types_v1_search_document.png`) | `results/anomaly_type_analysis/final_corpus/types_final_anom.json` |
| 6.1 ladder | `figures/fig_ladder.pdf` | `f1_ladder.py` | `figdata.json` |
| 6.2 arms | `figures/fig_arms.pdf` | `f2_arms.py` | `figdata.json` |
| 6.3 SFT curves | `figures/fig_sft.pdf` | `g1_sft.py` | trainer states |
| 6.4 decoupling | `figures/fig_decoupling.pdf` | `g2_decoupling.py` | eval files |
| 6.5 iter-2 | `figures/fig_iter2.pdf` | `g3_iter2.py` | eval files |
| 6.6 operating points | `figures/fig_operating.pdf` | `f3_operating.py` | eval files |
| 6.7 per-product heat map | `figures/fig_perproduct.pdf` | `alt_figs.py` (`alt_perproduct`) | per-product tables |
| 6.8 per-product bars, base against KCR | `figures/perproduct_bars_base_vs_kcr.pdf` | `f12_perproduct_bars.py Base` (also `SFT`, `SFT+GRPO`) | the eval files of Tables 6.2 and 6.8, strict per-product BA |
| 7.1 bidirectional failures | `figures/fig_limitations.pdf` | `f13_limitations.py` (six KCR cases, ids in the script) | KCR eval file, Appendix F.2 card |
| 6.9 explainability | `figures/fig_explain.pdf` | `f5_explain.py` | `results/explainability_shared/` |
| 6.9 two SFT vs KCR pairs | `figures/pairs_cashew.pdf`, `pairs_screw.pdf` | `f8_pairs.py` | `results/sft_vs_kcr_pairs/` |
| 6.10 gallery | `figures/fig_gallery.pdf` | `f6_gallery.py` | `prep_gallery.py` output |
| 6.11 dynamics | `figures/fig_dynamics.pdf` | `g4_dynamics.py` | trainer states |
| App. E heat map and Table 5.1 | `figures/type_cosine_heatmap.png` | `results/anomaly_type_analysis/final_corpus/reward_path_tables_and_tsne.py` | Nomic embeddings, reward server path (`search_query:` prefix, masked mean pooling) |
| App. F.1 to F.4, G.1 to G.5 | `figures/app_f*.pdf`, `app_g*.pdf` | `f9_appendix_examples.py` | `results/sft_vs_kcr_pairs/appendix_examples_selection.json` |
| App. F.5 21 pairs | `figures/pairs_svg/pair_*.pdf` | `f8_pairs.py` with `ALL_PAIRS=1` | `results/sft_vs_kcr_pairs/index_selected21.json` |
| App. H curves | `figures/curve_*.png` | `appendix/h0_sft_curves.py`, `h1_grpo_curves.py` | trainer states |

## Mask marker rule (10/11 Sep 2026)

Every mask-bearing figure (3.4, 6.9, 7.1, F.1 to F.5, G.1 to G.5) draws the marker in `f8_pairs.image_panel`: connected components of the mask, merged into clusters when centroids lie within 20 % of the image diagonal (the Single / Multi-Close / Spread rule of `analyze_mask_spread.py`), one red circle per cluster enclosing its pixels (radius floor 14 px, cap 45 % of the panel), arrow to the largest cluster. Before 11 Sep the circle was a fixed 14 px marker at the centroid.

## Prompt versions, so the appendix and the corpus are not confused again

`prompts/inspector_prompt_test_v2.txt` in this repo is the file Appendix B reproduces. It is
byte-identical (after whitespace) to the copy in `reasoning_traces_gen/prompts/` on the server.
The eleven HARD RULES, the eight-category acceptance filter, the six auto-reject rules and the
rewrite policy are the same in every copy of the file.

The generation runs on Adnane's laptop loaded a copy of the same file name whose location block
differed in about twenty lines: it did not print the 3x3 grid and it allowed the words
center-left, center-right, left, right, top and bottom. The 6K SFT split before the location fix
contains center-right 234 times, center-left 134 times and bare right 3 times, so that older
block is what the teacher saw. Those tags were mapped onto the nine grid cells afterwards
(vocabulary normalisation, datasheet App. I, "Location quantisation" in Figure 3.2). Every other
line of the prompt is identical between the two copies. If Appendix B is ever described as
"exact", say that the location block shown is the final wording. Other files in the laptop
folder (`inspector_prompt_test.txt`, `_v3`, `_v5`, `_v6`, `inspector_prompt.txt`) were
experiments and the old 82-line template prompt; none generated corpus traces.
