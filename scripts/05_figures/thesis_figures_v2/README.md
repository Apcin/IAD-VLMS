# Chapter 6 figure pipeline

Every figure in thesis Chapter 6 (and the ladder numbers in Figure 1.1) is drawn from one data file by
small SVG generators, then cropped, stripped of in-canvas prose and converted to PDF for the thesis.

```
WORK_DIR=/path/to/workspace   # parent of this repo, holds outputs/ (or point registry.py at results/)
python export_data.py         # registry.py -> figdata.json  (ladder, per-product, arms)
python loc_hit_table.py       # -> loc_hit.json, tab_loc_hit.tex   (thesis Table 6.12)
python type_sim_table.py      # -> type_sim.json, tab_type_sim.tex (thesis Appendix J.5)
python f1_ladder.py f2_arms.py f3_operating.py f4_perproduct.py f5_explain.py f6_gallery.py
python g1_sft.py g2_decoupling.py g3_iter2.py g4_dynamics.py g5_flow.py
THESIS_FIGURES=/path/to/thesis/figures python thesisify.py   # crops, strips captions, writes thesis/*.pdf and installs
```

- `registry.py` maps every (backbone, stage, benchmark) cell to its eval JSON. Change a checkpoint here and
  nothing else. The LLaVA SFT+GRPO and KCR rows point at the vLLM files by decision.
- `svgkit.py` holds the house style (Verdana, colours, markers). `thesisify.py` needs `cairosvg`
  (`CAIROSVG` env if not on PATH).
- `stages/f7_stages.py` draws Figure 4.1 (SFT stage next to GRPO stage). `svg_final/` holds the final SVGs
  as installed, including the Figure 1.1 overview (`method_overview_v4.svg`, exported to PDF by hand).
- `prep_gallery.py` and `prep_images.py` prepare the case images for `f6_gallery.py`.
- The generated data files are also shipped under `results/thesis_figure_data/`.

- `f8_pairs.py` draws Figure 6.9 and the 21 Appendix F.5 pairs (image beside the SFT and KCR traces, decisive
  sentence shaded). `ALL_PAIRS=1` renders all 21; the pair list is `results/sft_vs_kcr_pairs/index_selected21.json`.
- `f9_appendix_examples.py` draws the Appendix F.1 to F.4 and G.1 to G.5 example figures in the same style.
  Selection is by judge score, not by hand: F.1 from KCR images the explainability judge scored 10/10, F.2 from
  wrong verdicts plus judge scores of 5 or lower, F.3 from images where the judge scored SFT+GRPO 10/10 and SFT
  answered no. The chosen ids are in `results/sft_vs_kcr_pairs/appendix_examples_selection.json`.
  Both scripts measure text width with Noto Sans (what cairosvg substitutes for Verdana on the render box).
- `f13_limitations.py` draws Figure 7.1 (three false alarms and three misses of KCR) with the Appendix F.2 card.
- The mask marker of every gallery figure is one circle per separated mask region (20 % of diagonal rule), see docs/FIGURE_SOURCES.md.
- `f12_perproduct_bars.py <Base|SFT|SFT+GRPO>` draws the per-product standing bars of that stage against KCR on both
  benchmarks (Figure 6.8 uses `Base`), strict BA from the same eval files as the tables.
- `f10_trace_example.py` draws Figures 3.4 and 3.5, one anomalous and one normal training trace from the 6K SFT split beside
  their image in the same layout (`python f10_trace_example.py woodstick_0028_NG_ZW`, `... transistor1_0234_OK`). A record
  without a `<type>` tag is drawn in normal mode: no mask, no arrow, header with the answer tag only. Real-IAD masks sit next to the image as `.png`.

- 7 Sep 2026: Figure 6.7 is now the heat map from `alt_figs.py` (`alt_perproduct`), chosen from eight alternative
  renderings (`alt_*.svg`, one per Chapter 6 figure); the other seven figures keep their original generators.
  `thesisify.py` accepts figure names as arguments to install a subset. `f4_perproduct.py` is the retired dot-ladder version.
