"""Merge the 4 shards/subset for an API zero-shot MMAD eval and report balanced accuracy
(BA = 0.5*(TPR+TNR)), comparable to the Gemini-2.5 row.  Usage: merge_api_eval.py <outdir> <label>"""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/gpt5mini_eval")
LABEL = sys.argv[2] if len(sys.argv) > 2 else OUT.name
N_SHARDS = 4


def merge_subset(subset: str):
    results, missing = [], []
    for k in range(N_SHARDS):
        p = OUT / f"{subset}_shard{k}of{N_SHARDS}.json"
        if not p.exists():
            missing.append(k); continue
        try:
            results.extend(json.load(open(p)).get("results", []))
        except Exception:
            missing.append(k)
    tp = sum(1 for r in results if r["gt_answer"] == "yes" and r["pred_answer"] == "yes")
    tn = sum(1 for r in results if r["gt_answer"] == "no" and r["pred_answer"] == "no")
    fp = sum(1 for r in results if r["gt_answer"] == "no" and r["pred_answer"] == "yes")
    fn = sum(1 for r in results if r["gt_answer"] == "yes" and r["pred_answer"] == "no")
    no_ans = sum(1 for r in results if not r["pred_answer"])
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    ba = 0.5 * (tpr + tnr)
    acc = (tp + tn) / len(results) if results else 0.0
    json.dump({"subset": subset, "n": len(results), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
               "no_answer": no_ans, "BA": ba, "TPR_recall": tpr, "TNR_specificity": tnr,
               "accuracy": acc, "missing_shards": missing, "results": results},
              open(OUT / f"eval_{LABEL}_{subset}_merged.json", "w"), indent=1)
    return ba, tpr, tnr, acc, len(results), no_ans, missing


print("=" * 78)
print(f"{LABEL} — MMAD balanced accuracy")
print("=" * 78)
bas = {}
for sub, lab in [("ds", "DS-MVTec"), ("visa", "VisA")]:
    ba, tpr, tnr, acc, n, na, missing = merge_subset(sub)
    bas[lab] = ba
    warn = f"  ⚠ MISSING shards {missing}" if missing else ""
    print(f"{lab:9s} n={n:4d}  BA={ba*100:5.2f}%   recall/TPR={tpr*100:5.1f}  spec/TNR={tnr*100:5.1f}  "
          f"acc={acc*100:4.1f}  no_answer={na}{warn}")
if len(bas) == 2:
    print("-" * 78)
    print(f"References:  Gemini-2.5-Flash 81.52/75.18  |  GPT-5-mini 77.10/68.23  |  "
          f"Arm-C SFT 82.80/72.07  ||  {LABEL} = {bas['DS-MVTec']*100:.2f}/{bas['VisA']*100:.2f}")
