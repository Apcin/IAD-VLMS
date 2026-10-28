"""
Generate the fig:mask-spread figure: one Single, one Multi-Close, one Spread Real-IAD anomalous
example (original image with the GT mask overlaid in red) + its reasoning trace beneath each.
Classification matches analyze_mask_spread.py (connected components + 20%-diagonal spread threshold).
"""
import json, re, textwrap
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label, center_of_mass
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/bulk/aacudad/reasoning_traces")
CORPUS = ROOT / "Training/datasets_small_15k_c1_only/combined_sft_c1_train.json"
FIGOUT = Path("/tmp/thesis_review/thesis_tud/figures/mask_spread_examples.png")

def classify(mask_path):
    arr = np.array(Image.open(mask_path).convert("L")); b = arr > 128
    if not b.any(): return None
    lab, n = label(b)
    if n <= 1: return ("Single", 1, 0.0)
    cs = np.array(center_of_mass(b, lab, range(1, n+1)))
    h, w = arr.shape; diag = np.hypot(h, w); md = 0.0
    for i in range(len(cs)):
        for j in range(i+1, len(cs)):
            md = max(md, np.linalg.norm(cs[i]-cs[j]))
    nd = md/diag
    return ("Spread" if nd > 0.2 else "Multi-Close", n, nd)

def overlay(img_path, mask_path):
    base = Image.open(img_path).convert("RGB"); m = Image.open(mask_path).convert("L")
    if m.size != base.size: m = m.resize(base.size, Image.NEAREST)
    a = np.array(base); mm = np.array(m) > 128
    a[mm] = (0.45*a[mm] + np.array([255,0,0])*0.55).astype(np.uint8)
    im = Image.fromarray(a)
    im.thumbnail((520, 520))  # keep file size small
    return im

def trace_excerpt(trace, n_words=42):
    th = re.search(r"<think>(.*?)</think>", trace, re.S)
    body = re.sub(r"\s+", " ", th.group(1)).strip() if th else ""
    w = body.split()
    if len(w) > n_words: body = " ".join(w[:n_words]) + " ..."
    loc = (re.search(r"<location>(.*?)</location>", trace, re.S) or [None, "-"])[1].strip() if "<location>" in trace else "-"
    typ = (re.search(r"<type>(.*?)</type>", trace, re.S) or [None, "-"])[1].strip() if "<type>" in trace else "-"
    return body, typ, loc

def main():
    corpus = json.load(open(CORPUS))
    picks = {}  # mask_type -> (product, img, mask, trace)
    want = {"Single", "Multi-Close", "Spread"}
    seen_prod = set()
    for x in corpus:
        if len(picks) == 3: break
        img = Path(x["images"][0]); mask = img.with_suffix(".png")
        asst = [m for m in x["messages"] if m["role"] == "assistant"][0]["content"]
        if "<answer>Yes" not in asst and "<answer> Yes" not in asst: continue
        if not mask.exists(): continue
        cl = classify(mask)
        if cl is None: continue
        mtype, ncomp, nd = cl
        prod = img.parent.parent.parent.name if "OK" in str(img) or "NG" in str(img) else "?"
        prod = re.split(r"_[0-9]", img.stem)[0]
        # pick a clear representative: Spread needs nd in a readable range; Multi-Close 2-4 comps
        if mtype in want and mtype not in picks:
            if mtype == "Spread" and nd < 0.28: continue
            if mtype == "Multi-Close" and ncomp > 5: continue
            picks[mtype] = (prod, str(img), str(mask), asst, ncomp, nd)
    if len(picks) < 3:
        print("WARN only found:", list(picks));
    order = ["Single", "Multi-Close", "Spread"]
    order = [o for o in order if o in picks]
    fig, axes = plt.subplots(1, len(order), figsize=(4.2*len(order), 5.0))
    if len(order) == 1: axes = [axes]
    for ax, mt in zip(axes, order):
        prod, imgp, maskp, trace, ncomp, nd = picks[mt]
        ax.imshow(overlay(imgp, maskp)); ax.axis("off")
        body, typ, loc = trace_excerpt(trace)
        title = f"{mt}  ({ncomp} region{'s' if ncomp>1 else ''}" + (f", {nd:.2f} diag" if ncomp>1 else "") + f")\n{prod}"
        ax.set_title(title, fontsize=10, fontweight="bold")
        cap = textwrap.fill(body, 46) + f"\n<type>: {typ}    <location>: {loc}    <answer>: Yes"
        ax.text(0.5, -0.04, cap, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, family="monospace", wrap=True)
    plt.subplots_adjust(bottom=0.34, top=0.9, wspace=0.06)
    FIGOUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGOUT, dpi=130, bbox_inches="tight"); plt.close()
    sz = FIGOUT.stat().st_size/1024
    print(f"saved {FIGOUT} ({sz:.0f} KB)")
    for mt in order:
        p = picks[mt]; print(f"  {mt}: {p[0]}  comps={p[4]} dist={p[5]:.2f}  {Path(p[1]).name}")

if __name__ == "__main__": main()
