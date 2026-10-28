#!/usr/bin/env python3
"""
Build the A / B datasets for the three-arm SFT comparison.
C already exists as Training/datasets_sft_iter2/sft_iter2_train.json (6,000 items, full patched).

A = pure-kept (RAFT-style rejection sampling), 3,000 balanced
B = kept + corrected (STaR-style, no rewrites), the SAME 3,557 kept + 812 corrected from C, minus the 1,631 rewritten items
"""
import json, os, random
from collections import Counter
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PHASE0  = 'Training/phase0_full_10k_20260529_015821'
HELDOUT = 'Training/phase0_heldout_20260601'
C_PATH  = 'Training/datasets_sft_iter2/sft_iter2_train.json'

OUT_A   = 'Training/datasets_sft_iter2/sft_A_kept_balanced.json'
OUT_B   = 'Training/datasets_sft_iter2/sft_B_kept_corrected.json'

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def is_ng(img_path):
    return ('/NG/' in img_path) or ('_NG_' in img_path)

def load_jsonl_corrected_paths(path):
    """Return set of image_path that Gemini *corrected* (model failed all rollouts)."""
    if not os.path.exists(path): return set()
    out = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('corrected_trace'):
                out.add(r['image_path'])
    return out

def load_jsonl_rewritten_paths(path):
    if not os.path.exists(path): return set()
    out = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('corrected_trace'):  # same field name in both files
                out.add(r['image_path'])
    return out

def load_corrected_trace_map(path):
    """image_path -> corrected_trace text"""
    if not os.path.exists(path): return {}
    m = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('corrected_trace'):
                m[r['image_path']] = r['corrected_trace']
    return m

# ---------------------------------------------------------------------
# Identify which items in C are kept / corrected / rewritten
# ---------------------------------------------------------------------
corrected_paths = set()
rewritten_paths = set()
for d in (PHASE0, HELDOUT):
    corrected_paths |= load_jsonl_corrected_paths(f'{d}/gemini_corrected.jsonl')
    rewritten_paths |= load_jsonl_rewritten_paths(f'{d}/gemini_rewritten.jsonl')

c_data = json.load(open(C_PATH))
kept_in_C       = [d for d in c_data if d['images'][0] not in corrected_paths and d['images'][0] not in rewritten_paths]
corrected_in_C  = [d for d in c_data if d['images'][0] in corrected_paths]
rewritten_in_C  = [d for d in c_data if d['images'][0] in rewritten_paths]

print(f'C breakdown: kept={len(kept_in_C)}  corrected={len(corrected_in_C)}  rewritten={len(rewritten_in_C)}  total={len(c_data)}')

# ---------------------------------------------------------------------
# B = kept + corrected (no rewrites) — strict subset of C minus rewrites
# ---------------------------------------------------------------------
B = kept_in_C + corrected_in_C
random.shuffle(B)
b_ng = sum(1 for d in B if is_ng(d['images'][0]))
b_ok = len(B) - b_ng
print(f'B: {len(B):,} items  ({b_ng} NG  {b_ok} OK  =  {100*b_ng/len(B):.1f}% NG)')

json.dump(B, open(OUT_B, 'w'), indent=2)
print(f'  saved {OUT_B}')

# ---------------------------------------------------------------------
# A = pure kept balanced ~3K from the wider phase0 pool (more images than C)
# Targets: 1,500 NG + 1,500 OK, stratified per product when possible
# ---------------------------------------------------------------------
all_kept = []
for d in (PHASE0, HELDOUT):
    p = f'{d}/good_traces.json'
    if os.path.exists(p):
        all_kept.extend(json.load(open(p)))

kept_NG = [d for d in all_kept if is_ng(d['images'][0])]
kept_OK = [d for d in all_kept if not is_ng(d['images'][0])]
print(f'\nKept pool (both phase0 dirs): NG={len(kept_NG)}  OK={len(kept_OK)}')

# Per-product stratification: aim for 50 NG + 50 OK per product (×30 products = 3,000)
def product_of(img_path):
    if '/images/' in img_path:
        return img_path.split('/images/')[1].split('/')[0]
    return 'unknown'

from collections import defaultdict
ng_by_prod = defaultdict(list); ok_by_prod = defaultdict(list)
for d in kept_NG: ng_by_prod[product_of(d['images'][0])].append(d)
for d in kept_OK: ok_by_prod[product_of(d['images'][0])].append(d)

# Determine the actual minimum-per-product we can hit
products = sorted(set(ng_by_prod) | set(ok_by_prod))
print(f'  products with kept items: {len(products)}')

TARGET_PER_PROD_PER_CLASS = 50  # 50 NG + 50 OK per product × 30 = 3,000

A = []
for prod in products:
    n_avail = min(len(ng_by_prod.get(prod, [])), len(ok_by_prod.get(prod, [])), TARGET_PER_PROD_PER_CLASS)
    if n_avail == 0:
        print(f'  WARN: {prod} has no balanced kept items')
        continue
    A.extend(random.sample(ng_by_prod[prod], n_avail))
    A.extend(random.sample(ok_by_prod[prod], n_avail))

random.shuffle(A)
a_ng = sum(1 for d in A if is_ng(d['images'][0]))
a_ok = len(A) - a_ng
print(f'A: {len(A):,} items  ({a_ng} NG  {a_ok} OK  =  {100*a_ng/len(A):.1f}% NG)')

json.dump(A, open(OUT_A, 'w'), indent=2)
print(f'  saved {OUT_A}')

# ---------------------------------------------------------------------
# Per-product distribution sanity check
# ---------------------------------------------------------------------
for name, dset in [('A', A), ('B', B)]:
    counts = Counter(product_of(d['images'][0]) for d in dset)
    print(f'\n{name}: per-product item counts')
    for p in sorted(counts):
        print(f'   {p:25s}  {counts[p]:4d}')
