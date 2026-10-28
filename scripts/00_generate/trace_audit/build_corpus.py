#!/usr/bin/env python3
"""Step 1 of the trace grounding audit: resolve the full 10,236-trace corpus.

Pool (the 10,236-image rollout pool, re-checked here):
  SFT split  : Training/datasets_small_new_v4/combined_6k_train.json          6,000 items
  GRPO split : Training/datasets_small_15k_c1_only_fixed/grpo_train.json      4,236 items
The two splits are disjoint on image path and their union is the 10,236-image rollout pool.

For every item this writes one JSON line with
  id, split, product, camera, defect_code, gold_label, gold_type, gold_location,
  image_path (absolute), mask_path (absolute or null), ref_normal_path (absolute),
  question, trace.

Mask recipe: Real-IAD stores the mask next to the capture, same stem, .png.
Some NG samples have no C1 mask because the defect is not visible in the top-down view.
Those keep mask_path = null and are reported at the end.

Reference normal: same product, same camera (the whole pool is C1), an OK capture from a
different physical sample folder. Picked deterministically with md5(id) so re-runs are stable.

Usage: python3 build_corpus.py
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter
from glob import glob

import numpy as np
from PIL import Image

# NOT RUNNABLE WITHOUT THE RAW DATA. This step needs the two trace splits AND the raw
# Real-IAD images and masks, which this repository does not redistribute (README section 3).
# WORK_DIR is the workspace holding Training/ and reasoning_traces_gen/; it defaults to the
# parent of this repository. On another machine:  export WORK_DIR=/path/to/workspace
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("WORK_DIR") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
SFT_JSON = os.environ.get("SFT_JSON", f"{R}/Training/datasets_small_new_v4/combined_6k_train.json")
GRPO_JSON = os.environ.get("GRPO_JSON", f"{R}/Training/datasets_small_15k_c1_only_fixed/grpo_train.json")
IMG_ROOT = os.environ.get("REALIAD_IMAGES", f"{R}/reasoning_traces_gen/data/Real-IAD/images")
OUT = os.path.join(HERE, "corpus.jsonl")

# Official Real-IAD code map, copied from reasoning_traces_gen/utils_realiad.py
ANOMALY_MAP = {
    "OK": "Normal", "AK": "Pit", "BX": "Deformation", "CH": "Abrasion",
    "HS": "Scratch", "PS": "Damage", "QS": "Missing Parts",
    "YW": "Foreign Objects", "ZW": "Contamination",
}


# ----------------------------------------------------------------- ids and paths
def item_id(path):
    """product_OK_S0485_<stem>  or  product_NG_ZW_S0131_<stem>.

    Identical to the image_id already stored in grpo_train.json.
    """
    parts = path.split("/images/")[1].split("/")
    return "_".join(parts[:-1]) + "_" + os.path.splitext(parts[-1])[0]


def parse_path(path):
    parts = path.split("/images/")[1].split("/")
    product = parts[0]
    is_ng = parts[1] == "NG"
    defect = parts[2] if is_ng else "OK"
    m = re.search(r"_(C\d)_", os.path.basename(path))
    return product, is_ng, defect, (m.group(1) if m else "?")


def mask_for(path):
    mp = path[:-4] + ".png"
    return mp if os.path.exists(mp) else None


_ok_pool = {}


def ok_pool(product, camera):
    key = (product, camera)
    if key not in _ok_pool:
        _ok_pool[key] = sorted(glob(f"{IMG_ROOT}/{product}/OK/*/*_{camera}_*.jpg"))
    return _ok_pool[key]


def ref_normal(path, product, camera, iid):
    pool = ok_pool(product, camera)
    own = os.path.dirname(path)
    cands = [p for p in pool if os.path.dirname(p) != own]
    if not cands:
        cands = [p for p in pool if p != path] or pool
    if not cands:
        return None
    h = int(hashlib.md5(iid.encode()).hexdigest(), 16)
    return cands[h % len(cands)]


# ----------------------------------------------------------------- gold location
def _label_runs(binary):
    """Connected components (8-neighbour) via run-length union-find. No scipy needed."""
    h, w = binary.shape
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs = []          # (row, start, end_exclusive, label)
    prev_runs = []
    nxt = 0
    for y in range(h):
        row = binary[y]
        idx = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        cur = []
        for s, e in zip(idx[0::2], idx[1::2]):
            parent[nxt] = nxt
            lab = nxt
            nxt += 1
            for ps, pe, pl in prev_runs:
                if ps <= e and s <= pe:       # 8-connected overlap
                    union(lab, pl)
            cur.append((s, e, lab))
            runs.append((y, s, e, lab))
        prev_runs = cur
    if not runs:
        return None, None
    sums = {}
    for y, s, e, lab in runs:
        root = find(lab)
        a = sums.setdefault(root, [0.0, 0.0, 0])
        n = e - s
        a[0] += (s + e - 1) / 2.0 * n     # sum of x
        a[1] += y * n                     # sum of y
        a[2] += n                         # area
    return sums, (h, w)


def _cell(sx, sy, area, h, w):
    nx, ny = (sx / area) / w, (sy / area) / h
    hp = "left" if nx < 0.33 else ("right" if nx > 0.66 else "center")
    vp = "top" if ny < 0.33 else ("bottom" if ny > 0.66 else "middle")
    if vp == "middle":
        return "center" if hp == "center" else f"middle-{hp}"
    if hp == "center":
        return f"{vp}-center"
    return f"{vp}-{hp}"


def gold_location(mask_path):
    """Returns (multi, primary).

    multi   = same rule as utils_realiad.get_mask_location, up to 3 largest blobs.
    primary = the single largest blob only. The stored <location> tags follow the
              single-blob convention more closely (95 % vs 90 % exact agreement on
              the 4,236 GRPO items), so both are kept.
    """
    if not mask_path:
        return None, None
    try:
        arr = np.array(Image.open(mask_path).convert("L"))
        sums, shape = _label_runs(arr > 128)
        if not sums:
            return None, None
        h, w = shape
        blobs = sorted(sums.values(), key=lambda a: -a[2])
        locs = []
        for sx, sy, area in blobs[:3]:
            loc = _cell(sx, sy, area, h, w)
            if loc not in locs:
                locs.append(loc)
        return ", ".join(locs), _cell(*blobs[0], h, w)
    except Exception:
        return None, None


# ----------------------------------------------------------------- corpus build
def load_sft():
    out = []
    for it in json.load(open(SFT_JSON)):
        p = it["images"][0]
        out.append({
            "split": "sft_6k",
            "image_path": p,
            "question": it["messages"][0]["content"].replace("<image>\n", ""),
            "trace": it["messages"][1]["content"],
        })
    return out


def load_grpo():
    out = []
    for it in json.load(open(GRPO_JSON)):
        out.append({
            "split": "grpo_4k",
            "image_path": it["image_path"],
            "question": it["question"],
            "trace": it["answer"],
        })
    return out


def main():
    items = load_sft() + load_grpo()
    paths = [it["image_path"] for it in items]
    assert len(items) == 10236, len(items)
    assert len(set(paths)) == 10236, "image paths are not unique"

    rows, no_mask, missing_ref, cams = [], 0, 0, Counter()
    for it in items:
        p = it["image_path"]
        product, is_ng, defect, cam = parse_path(p)
        cams[cam] += 1
        iid = item_id(p)
        mp = mask_for(p) if is_ng else None
        if is_ng and mp is None:
            no_mask += 1
        ref = ref_normal(p, product, cam, iid)
        if ref is None:
            missing_ref += 1
        # Mask-aware gold label, NOT the folder name: an NG capture with no C1 mask shows
        # nothing in the top-down view and is labelled "no". This rule reproduces every one
        # of the 10,236 gold_label values in the corpus the audit actually ran on.
        gt_label = "yes" if (is_ng and mp) else "no"
        loc_multi, loc_primary = gold_location(mp)
        rows.append({
            "id": iid,
            "split": it["split"],
            "product": product,
            "camera": cam,
            "defect_code": defect,
            "gold_label": gt_label,  # mask-aware, see above
            "gold_type": ANOMALY_MAP.get(defect, defect) if is_ng else None,
            "gold_location": loc_multi,
            "gold_location_primary": loc_primary,
            "image_path": p,
            "mask_path": mp,
            "ref_normal_path": ref,
            "question": it["question"],
            "trace": it["trace"],
        })

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ng = sum(r["gold_label"] == "yes" for r in rows)
    print(f"wrote {OUT}")
    print(f"items                : {len(rows)}")
    print(f"  sft_6k             : {sum(r['split']=='sft_6k' for r in rows)}")
    print(f"  grpo_4k            : {sum(r['split']=='grpo_4k' for r in rows)}")
    print(f"gold yes / no        : {ng} / {len(rows)-ng}")
    print(f"cameras              : {dict(cams)}")
    print(f"anomalous w/o C1 mask: {no_mask}")
    print(f"items w/o reference  : {missing_ref}")
    print(f"products             : {len(set(r['product'] for r in rows))}")
    byprod = Counter(r["product"] for r in rows if r["mask_path"] is None and r["gold_label"] == "yes")
    print("no-mask by product   :", byprod.most_common(8))


if __name__ == "__main__":
    sys.exit(main())
