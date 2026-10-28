"""
Strict balanced accuracy of an eval_*.json restricted to the MMAD held-out test keys.

Usage: python score_heldout.py split.json eval_dsmvtec.json [eval_visa.json ...]
Prints, per file: BA on the full subset (as the thesis harness does), BA on the held-out
test images only, BA on the training images only (memorisation check), and unparsed counts.
Strict scoring: an unparsed prediction counts as wrong, as in Training/strict_ba.py.
"""
import json
import os
import sys
from collections import defaultdict

MMAD_ROOT = os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/reasoning_traces_gen/data/MMAD/"


def ba(rows):
    P = sum(x["gt_answer"] == "yes" for x in rows)
    N = len(rows) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = sum(x["gt_answer"] == "yes" and x["pred_answer"] == "yes" for x in rows)
    tn = sum(x["gt_answer"] == "no" and x["pred_answer"] == "no" for x in rows)
    return 50.0 * (tp / P + tn / N)


def key_of(row):
    p = row["absolute_path"].replace("\\", "/")
    i = p.find(MMAD_ROOT)
    return p[i + len(MMAD_ROOT):] if i >= 0 else p


def main():
    split = json.load(open(sys.argv[1]))
    train = set(split["train_keys"])
    test = set(split["test_keys"])
    for path in sys.argv[2:]:
        rows = json.load(open(path))["results"]
        r_test = [x for x in rows if key_of(x) in test]
        r_train = [x for x in rows if key_of(x) in train]
        r_other = [x for x in rows if key_of(x) not in test and key_of(x) not in train]
        un = sum(x["pred_answer"] not in ("yes", "no") for x in rows)
        print(f"{path}")
        print(f"  full     n={len(rows):5d} unparsed={un:3d} BA={ba(rows):6.2f}")
        print(f"  held-out n={len(r_test):5d}                BA={ba(r_test):6.2f}")
        print(f"  train    n={len(r_train):5d}                BA={ba(r_train):6.2f}")
        if r_other:
            print(f"  unmatched n={len(r_other)} (not in split, e.g. dropped images)")
        per = defaultdict(list)
        for x in r_test:
            per[x["product"]].append(x)
        print("  held-out per product: " + ", ".join(f"{p} {ba(v):.1f}" for p, v in sorted(per.items())))


if __name__ == "__main__":
    main()
