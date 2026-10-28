#!/usr/bin/env python
"""GRPO on top of Arm-C: checkpoint trajectory + beta sweep.

Point of the figure: the 82.95 that looks like "GRPO beat Arm-C" is the MAXIMUM of 15 noisy
checkpoints. The mean sits below the Arm-C baseline, and only 2/15 checkpoints clear it.

All numbers recomputed from the raw eval JSONs (tp/tn/fp/fn), never transcribed.
Line chart = the right form for an ordered trajectory; one shared BA axis (never a dual axis);
baseline drawn as a reference rule, not a series.
"""
import json, os, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/bulk/aacudad/reasoning_traces"
OUT = f"{R}/outputs/grpo_on_armC_trajectory"; os.makedirs(OUT, exist_ok=True)

# documented reference categorical palette (light), fixed order
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#1a1a19", "#5c5b55", "#e6e5df"
BASE_DS, BASE_VISA = 82.80, 72.07          # Arm-C ckpt-376 baseline

ba = lambda m: 0.5 * (m["tp"] / (m["tp"] + m["fn"]) + m["tn"] / (m["tn"] + m["fp"])) * 100

def traj(run_dir):
    """checkpoint -> (DS, VisA) recomputed from the eval JSONs."""
    out = {}
    d = f"{R}/outputs/{run_dir}"
    if not os.path.isdir(d): return out
    for ck in os.listdir(d):
        if not ck.startswith("checkpoint-"): continue
        vals = {}
        for f, key in [("eval_dsmvtec_full_trainprompt.json", "ds"), ("eval_visa_full_trainprompt.json", "va")]:
            p = os.path.join(d, ck, f)
            if os.path.exists(p): vals[key] = ba(json.load(open(p))["metrics"])
        if "ds" in vals: out[int(ck.split("-")[1])] = (vals["ds"], vals.get("va"))
    return dict(sorted(out.items()))

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)

run3 = traj("grpo_sftprompt_kl0.1_sys_3ep")
b10, b04 = traj("grpo_sftprompt_kl0.1"), traj("grpo_sftprompt_kl0.04")

fig, (axA, axV, axB) = plt.subplots(3, 1, figsize=(11.5, 12.4), facecolor=SURFACE)

xs = list(run3); ds = [run3[c][0] for c in xs]; va = [run3[c][1] for c in xs]
mean_ds, mean_va = statistics.mean(ds), statistics.mean(va)
above = sum(1 for v in ds if v > BASE_DS)
above_v = sum(1 for v in va if v > BASE_VISA)

# ---------- A: DS-MVTec trajectory (the main story) ----------
axA.axhline(BASE_DS, color=INK_MUTED, ls="--", lw=1.5, zorder=2)
axA.text(xs[0], BASE_DS + .10, f"Arm-C baseline {BASE_DS:.2f}", fontsize=8.5, color=INK_MUTED, va="bottom")
axA.axhline(mean_ds, color=BLUE, ls=":", lw=1.4, alpha=.8, zorder=2)
axA.text(xs[0], mean_ds + .12, f"mean of 15 ckpts {mean_ds:.2f}", fontsize=8.5, color=BLUE, ha="left", va="bottom")
axA.plot(xs, ds, "-o", color=BLUE, lw=2, ms=5.5, zorder=3)
best = max(range(len(ds)), key=lambda i: ds[i])
axA.annotate(f'ckpt-{xs[best]} = {ds[best]:.2f}  ← the "82.95"\n(best of 15, +0.15 over baseline)',
             xy=(xs[best], ds[best]), xytext=(xs[best] - 55, ds[best] - 2.45), fontsize=9, color=INK,
             ha="right", arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.2))
axA.set_ylim(min(ds) - 1.9, max(ds) + 0.55)
axA.set_title(f"GRPO on top of Arm-C (β=0.1, eval-aligned, 3 epochs) — DS-MVTec: only {above}/15 checkpoints beat the baseline",
              fontsize=11.5, color=INK, loc="left", pad=8)
axA.set_ylabel("DS-MVTec BA", fontsize=9.5, color=INK_MUTED); style(axA)

# ---------- V: VisA trajectory (opposite pattern) ----------
axV.axhline(BASE_VISA, color=INK_MUTED, ls="--", lw=1.5, zorder=2)
axV.text(xs[0], BASE_VISA + .10, f"Arm-C baseline {BASE_VISA:.2f}", fontsize=8.5, color=INK_MUTED, va="bottom")
axV.axhline(mean_va, color=ORANGE, ls=":", lw=1.4, alpha=.8, zorder=2)
axV.text(xs[-1], mean_va - .42, f"mean {mean_va:.2f}  (= baseline, no net gain)", fontsize=8.5, color=ORANGE, ha="right", va="top")
axV.plot(xs, va, "-o", color=ORANGE, lw=2, ms=5.5, zorder=3)
axV.set_ylim(min(va) - 0.9, max(va) + 0.6)
axV.set_title(f"same run, VisA: {above_v}/15 checkpoints sit above the baseline, but the mean ({mean_va:.2f}) equals it ({BASE_VISA:.2f}) — no net gain either",
              fontsize=11.5, color=INK, loc="left", pad=8)
axV.set_xlabel("checkpoint (step)", fontsize=9.5, color=INK_MUTED)
axV.set_ylabel("VisA BA", fontsize=9.5, color=INK_MUTED); style(axV)

# ---------- B: beta sweep ----------
allv = []
for run, lbl, col in [(b10, "β = 0.1  (strict KL)", BLUE), (b04, "β = 0.04 (loose KL)", ORANGE)]:
    if not run: continue
    x = list(run); y = [run[c][0] for c in x]; allv += y
    axB.plot(x, y, "-o", color=col, lw=2, ms=6, label=lbl, zorder=3)
    for xi, yi in zip(x, y):
        axB.text(xi, yi + .07, f"{yi:.2f}", ha="center", fontsize=7.6, color=INK_MUTED, zorder=4)
axB.axhline(BASE_DS, color=INK_MUTED, ls="--", lw=1.5, zorder=2)
axB.text(list(b10)[0] if b10 else 106, BASE_DS + .06, f"Arm-C baseline {BASE_DS:.2f}",
         fontsize=8.5, color=INK_MUTED, va="bottom")
axB.set_ylim(min(allv) - 0.75, BASE_DS + 0.45)
axB.legend(frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22), labelcolor=INK)
axB.set_title("β sweep (1 epoch): looser KL earns more reward but drifts down — neither β reaches the baseline",
              fontsize=11.5, color=INK, loc="left", pad=8)
axB.set_xlabel("checkpoint (step)", fontsize=9.5, color=INK_MUTED)
axB.set_ylabel("DS-MVTec BA", fontsize=9.5, color=INK_MUTED); style(axB)

fig.tight_layout()
fig.savefig(f"{OUT}/grpo_on_armC_trajectory.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
print(f"wrote {OUT}/grpo_on_armC_trajectory.png")

# ---------- the numbers, as a table ----------
with open(f"{OUT}/trajectory_table.md", "w") as f:
    f.write("| checkpoint | DS-MVTec | VisA | avg |\n|---|--:|--:|--:|\n")
    f.write(f"| **init (Arm-C ckpt-376)** | **{BASE_DS:.2f}** | **{BASE_VISA:.2f}** | **{(BASE_DS+BASE_VISA)/2:.2f}** |\n")
    for c in xs:
        d_, v_ = run3[c]
        f.write(f"| {c} | {d_:.2f} | {v_:.2f} | {(d_+v_)/2:.2f} |\n")
    avg = [(run3[c][0] + run3[c][1]) / 2 for c in xs]
    f.write(f"\nDS mean **{mean_ds:.2f}** (sd {statistics.pstdev(ds):.2f}, min {min(ds):.2f}, max {max(ds):.2f}) "
            f"vs baseline {BASE_DS:.2f}. Avg mean **{statistics.mean(avg):.2f}** vs baseline "
            f"{(BASE_DS+BASE_VISA)/2:.2f}. Checkpoints above baseline DS: **{above}/15**.\n")
print(f"wrote {OUT}/trajectory_table.md")
