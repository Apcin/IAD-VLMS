#!/usr/bin/env python3
"""Build the 200-item calibration sample and the 40-item repeat set.

Design
  seed = 20260902 (the same seed as the transistor lead-count sample)
  transistor1        : 20 items (10 normal, 10 anomalous). The thesis flagship normal
                       trace is forced in, so 19 other transistor1 traces come along.
                       transistor1 is the accuracy check on the judge, because the part
                       has a verified 9 leads per side / 18 total and 91 % of the traces
                       that state a count state it wrong.
  other 29 products  : 6 each (3 normal, 3 anomalous) = 174, plus 6 spread over the six
                       largest remaining products (3 normal, 3 anomalous) = 180.
  total              : 200 = 100 normal + 100 anomalous, all 30 products covered.

The repeat set is 40 of the 200, drawn with the same seed, stratified 20 normal /
20 anomalous, with the flagship forced in.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260902
FLAGSHIP = "transistor1_OK_S0234_transistor1_0234_OK_C1_20230923170453"

rows = [json.loads(l) for l in open(HERE / "corpus.jsonl")]
by = defaultdict(list)
for r in rows:
    by[(r["product"], r["gold_label"])].append(r)
for k in by:
    by[k].sort(key=lambda r: r["id"])          # deterministic order before sampling

rng = random.Random(SEED)
picked, ids = [], set()


def take(product, label, n, force=None):
    pool = [r for r in by[(product, label)] if r["id"] not in ids]
    out = []
    if force:
        f = [r for r in pool if r["id"] == force]
        assert f, f"forced id not in pool: {force}"
        out.append(f[0])
        pool = [r for r in pool if r["id"] != force]
    out += rng.sample(pool, n - len(out))
    for r in out:
        ids.add(r["id"])
    picked.extend(out)
    return out


# ---- transistor1, the judge accuracy check -------------------------------------
take("transistor1", "no", 10, force=FLAGSHIP)
take("transistor1", "yes", 10)

# ---- the other 29 products ------------------------------------------------------
others = sorted(p for p in set(r["product"] for r in rows) if p != "transistor1")
for p in others:
    take(p, "no", 3)
    take(p, "yes", 3)

# ---- 6 extras, on the biggest remaining products, 3 normal + 3 anomalous ---------
size = Counter(r["product"] for r in rows)
big = sorted(others, key=lambda p: -size[p])
for p in big[:3]:
    take(p, "no", 1)
for p in big[3:6]:
    take(p, "yes", 1)

assert len(picked) == 200, len(picked)
assert len(set(r["id"] for r in picked)) == 200
assert sum(r["gold_label"] == "no" for r in picked) == 100
assert len(set(r["product"] for r in picked)) == 30
assert FLAGSHIP in ids

picked.sort(key=lambda r: r["id"])
with open(HERE / "calib_200.jsonl", "w") as f:
    for r in picked:
        f.write(json.dumps(r) + "\n")

# ---- the 40-item repeat set -----------------------------------------------------
rng2 = random.Random(SEED)
norm = [r for r in picked if r["gold_label"] == "no"]
anom = [r for r in picked if r["gold_label"] == "yes"]
flag = [r for r in norm if r["id"] == FLAGSHIP]
rep = flag + rng2.sample([r for r in norm if r["id"] != FLAGSHIP], 19) + rng2.sample(anom, 20)
rep.sort(key=lambda r: r["id"])
assert len(rep) == 40 and len(set(r["id"] for r in rep)) == 40
with open(HERE / "calib_repeat_40.jsonl", "w") as f:
    for r in rep:
        f.write(json.dumps(r) + "\n")

print(f"seed {SEED}")
print(f"calib_200.jsonl        200 items | normal 100 | anomalous 100 | products 30")
print(f"  transistor1          {sum(r['product']=='transistor1' for r in picked)} "
      f"(flagship forced in, {sum(r['product']=='transistor1' for r in picked)-1} others)")
print(f"  anomalous w/o mask   {sum(1 for r in picked if r['gold_label']=='yes' and not r['mask_path'])}")
print(f"  split                {dict(Counter(r['split'] for r in picked))}")
print(f"calib_repeat_40.jsonl  40 items | normal {sum(r['gold_label']=='no' for r in rep)} "
      f"| anomalous {sum(r['gold_label']=='yes' for r in rep)} "
      f"| products {len(set(r['product'] for r in rep))}")
print("per-product counts:", dict(sorted(Counter(r["product"] for r in picked).items())))
