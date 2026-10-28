#!/usr/bin/env python
"""Visualise the 5 explainability axes (§6.8 judge) across models, per benchmark.

Two forms:
  1. HEATMAP  (models x axes, one panel per benchmark) -- the sanctioned form for a grid of
     magnitudes; every cell annotated so it doubles as the table view.
  2. GROUPED BARS (axes on x, models as series, one panel per benchmark) -- precise comparison.

Palette: the documented reference categorical palette, slots 1-6, unmodified order (validated for
the adjacent pairlist = bars). Value labels are always drawn (relief rule: magenta/yellow/aqua sit
below 3:1 on the light surface). Sequential blue ramp for the heatmap (magnitude = one hue).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SUM = "/bulk/aacudad/reasoning_traces/outputs/explainability_multi/summary.json"
OUT = "/bulk/aacudad/reasoning_traces/outputs/explainability_multi"

# documented reference categorical palette, slots 1-6 (light mode), fixed order
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK, INK_MUTED, GRID = "#1a1a19", "#5c5b55", "#e6e5df"

AXES = [("j_visual_grounding", "Visual\ngrounding"), ("j_defect_faithfulness", "Defect\nfaithfulness"),
        ("j_evidence_before_conclusion", "Evidence before\nconclusion"), ("j_coherence", "Coherence"),
        ("j_conciseness", "Conciseness")]
# display order: pipeline story (bases -> finetuned), IAD-R1 as external baseline
MODELS = [("qwen3_base", "Qwen3-VL base"), ("base_qwen25", "Qwen2.5-VL base"), ("iadr1", "IAD-R1 (native prompt)"),
          ("iadr1_trainprompt", "IAD-R1 (prompt-matched)"),
          ("armC_finalsft", "Arm-C (final SFT)"), ("sft_grpo", "SFT+GRPO"), ("qwen3_sft", "Qwen3-VL SFT")]
BENCHES = ["DS-MVTec", "VisA"]

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)

def main():
    S = json.load(open(SUM))
    get = lambda m, b, k: S.get(f"{m}|{b}", {}).get(k)

    # ---------- 1. HEATMAP ----------
    blue = LinearSegmentedColormap.from_list("seq_blue", ["#f2f7fd", "#2a78d6", "#123a68"])
    blue.set_bad("#e9e8e2")   # missing data must NOT read as a low score
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor=SURFACE)
    # identical row set in both panels so rows line up; missing cells shown as n/a
    for ax, bench in zip(axs, BENCHES):
        M = np.array([[get(m, bench, k) if f"{m}|{bench}" in S else np.nan for k, _ in AXES]
                      for m, _ in MODELS], dtype=float)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=blue, vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(AXES))); ax.set_xticklabels([n for _, n in AXES], fontsize=9)
        ax.set_yticks(range(len(MODELS)))
        if ax is axs[0]:
            ax.set_yticklabels([lbl for _, lbl in MODELS], fontsize=9.5)
        else:
            ax.set_yticklabels([])          # shared rows -> label once, avoids collision
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=8.5, color=INK_MUTED, style="italic")
                else:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                            color="#ffffff" if v > 1.15 else INK)
        ax.set_title(bench, fontsize=11, color=INK, pad=8)
        ax.tick_params(colors=INK_MUTED, length=0)
        for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    cb.set_label("axis score (0–2)", color=INK_MUTED, fontsize=9)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8); cb.outline.set_visible(False)
    fig.suptitle("Explainability judge — five reasoning axes per model (mean over 100 correct anomalies, median of 3)",
                 fontsize=12.5, color=INK, y=0.98)
    fig.savefig(f"{OUT}/explainability_5axes_heatmap.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # ---------- 2. GROUPED BARS ----------
    fig, axs = plt.subplots(2, 1, figsize=(13.5, 9.6), facecolor=SURFACE)
    # colour + slot are bound to the MODEL, not to its position among the survivors:
    # a benchmark missing one model must not repaint/shift the others.
    COLOR = {m: CAT[i] for i, (m, _) in enumerate(MODELS)}
    for ax, bench in zip(axs, BENCHES):
        n = len(MODELS); x = np.arange(len(AXES)); w = 0.78 / n
        for i, (m, lbl) in enumerate(MODELS):
            if f"{m}|{bench}" not in S:
                continue                      # leave its slot empty (gap = no data)
            vals = [get(m, bench, k) for k, _ in AXES]
            pos = x - 0.39 + w * (i + 0.5)
            ax.bar(pos, vals, width=w * 0.9, color=COLOR[m], label=lbl, zorder=3)
            for xp, v in zip(pos, vals):   # relief rule: always label
                ax.text(xp, v + 0.045, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2, color=INK_MUTED, zorder=4)
        ax.set_xticks(x); ax.set_xticklabels([nm for _, nm in AXES], fontsize=9.5)
        ax.set_ylim(0, 2.25); ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.set_ylabel("score (0–2)", fontsize=9.5, color=INK_MUTED)
        ax.grid(axis="y", color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)
        ax.set_title(bench, fontsize=11, color=INK, loc="left", pad=6)
        style(ax)
    axs[0].legend(ncol=6, frameon=False, fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, 1.30), labelcolor=INK)
    fig.suptitle("Explainability judge — five reasoning axes per model", fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/explainability_5axes_bars.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # ---------- 3. markdown table (table view / accessibility) ----------
    with open(f"{OUT}/explainability_5axes_table.md", "w") as f:
        f.write("| Model | Benchmark | " + " | ".join(n.replace("\n", " ") for _, n in AXES) + " | overall /10 | n |\n")
        f.write("|---|---|" + "--:|" * (len(AXES) + 2) + "\n")
        for m, lbl in MODELS:
            for b in BENCHES:
                if f"{m}|{b}" not in S: continue
                v = [f"{get(m,b,k):.2f}" for k, _ in AXES]
                f.write(f"| {lbl} | {b} | " + " | ".join(v) + f" | {get(m,b,'j_overall'):.2f} | {get(m,b,'n')} |\n")
    print("wrote:")
    for fn in ["explainability_5axes_heatmap.png", "explainability_5axes_bars.png", "explainability_5axes_table.md"]:
        kb = os.path.getsize(os.path.join(OUT, fn)) / 1024
        print(f"  {OUT}/{fn}  ({kb:.0f} KB)")

if __name__ == "__main__":
    main()
