#!/usr/bin/env python
"""Assemble the final SFT-Iter2-heldout training dataset.

Composition:
  - All 4236 held-out items, each represented by:
      * If failed all rollouts → Gemini-corrected trace
      * If weak/judge-flagged → Gemini-rewritten trace
      * Else → model's best passing rollout (RFT)
  - PLUS 197 OK + 197 NG balanced patches from the existing 10k rollout
    (stratified to match OK's per-product distribution)

Total: ~4,630 items.
"""
import json, random, os, re
from collections import defaultdict

random.seed(42)

# ── Held-out pipeline output ──
HELDOUT_DIR = '/bulk/aacudad/reasoning_traces/Training/phase0_heldout_20260601'
# Existing 10k pipeline output (source of the 197+197 patches)
EXISTING_DIR = '/bulk/aacudad/reasoning_traces/Training/phase0_full_10k_20260529_015821'
# Held-out source pool (for user prompts)
HELDOUT_POOL = '/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only_fixed/new_sft_c1_train.json'

OUT = '/bulk/aacudad/reasoning_traces/Training/datasets_sft_iter2/sft_iter2_heldout_train.json'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── Load user prompts for the held-out pool (sharegpt format) ──
with open(HELDOUT_POOL) as f:
    heldout_pool = json.load(f)
heldout_user_prompt = {r['images'][0]: r['messages'][0]['content'] for r in heldout_pool}
heldout_paths = set(heldout_user_prompt.keys())
print(f"Held-out pool: {len(heldout_paths)} items")

# ── Load held-out pipeline outputs ──
heldout_corrected = {}
with open(f'{HELDOUT_DIR}/gemini_corrected.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('success') and r['image_path'] in heldout_paths:
            heldout_corrected[r['image_path']] = r['corrected_trace']
print(f"  held-out corrected: {len(heldout_corrected)}")

heldout_rewritten = {}
with open(f'{HELDOUT_DIR}/gemini_rewritten.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('success') and r['image_path'] in heldout_paths:
            heldout_rewritten[r['image_path']] = r['corrected_trace']
print(f"  held-out rewritten: {len(heldout_rewritten)}")

with open(f'{HELDOUT_DIR}/needs_rewrite_combined.json') as f:
    heldout_rewrite_paths = {it['image_path'] for it in json.load(f) if it['image_path'] in heldout_paths}
with open(f'{HELDOUT_DIR}/good_traces.json') as f:
    heldout_good = json.load(f)
heldout_kept = {}  # image_path → assistant_text
for it in heldout_good:
    p = it['images'][0]
    if p in heldout_paths and p not in heldout_rewrite_paths:
        heldout_kept[p] = it['messages'][1]['content']
print(f"  held-out kept (good and not in rewrite): {len(heldout_kept)}")
print(f"  held-out total covered: {len(heldout_corrected) + len(heldout_rewritten) + len(heldout_kept)}")

# ── Load 197 OK + 197 NG balanced patches from existing 10k ──
# Per-item gt and product from rollouts_raw
ex_gt, ex_prod = {}, {}
with open(f'{EXISTING_DIR}/rollouts_raw.jsonl') as f:
    for line in f:
        r = json.loads(line)
        ex_gt[r['image_path']] = r.get('gt_answer','?')
        ex_prod[r['image_path']] = r.get('product','?')

# All patched items in the 10k (corrected + rewritten)
existing_patches = {}
for fp in [f'{EXISTING_DIR}/gemini_corrected.jsonl', f'{EXISTING_DIR}/gemini_rewritten.jsonl']:
    with open(fp) as f:
        for line in f:
            r = json.loads(line)
            if r.get('success'):
                existing_patches[r['image_path']] = r['corrected_trace']

# Split by gt
ok_patches = [p for p in existing_patches if ex_gt.get(p) == 'no']
ng_patches = [p for p in existing_patches if ex_gt.get(p) == 'yes']
print(f"\nExisting 10k patches: {len(ok_patches)} OK + {len(ng_patches)} NG")

# Stratify NG to match OK's per-product distribution
ok_by_prod = defaultdict(list)
for p in ok_patches:
    ok_by_prod[ex_prod[p]].append(p)
ng_by_prod = defaultdict(list)
for p in ng_patches:
    ng_by_prod[ex_prod[p]].append(p)

ng_selected = []
for prod, ok_list in ok_by_prod.items():
    avail = ng_by_prod.get(prod, [])
    sampled = random.sample(avail, min(len(ok_list), len(avail)))
    ng_selected.extend(sampled)
print(f"  Sampled: {len(ok_patches)} OK + {len(ng_selected)} NG balanced")

# ── Need user_prompts for the existing patches too (look up from SFT/GRPO pools) ──
existing_user_prompt = {}
with open('/bulk/aacudad/reasoning_traces/Training/datasets_small_new_v4/combined_6k_train.json') as f:
    for r in json.load(f):
        existing_user_prompt[r['images'][0]] = r['messages'][0]['content']
# GRPO pool: build prompt from question template (matches SFT format)
with open('/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json') as f:
    grpo_pool = json.load(f)
for r in grpo_pool:
    if r['image_path'] not in existing_user_prompt:
        prod = r.get('product','?')
        # Use the same make_train_prompt format as SFT
        prompt = (f"<image>\nAnalyze the provided image of the {prod}. "
                  "Determine if there are any anomalies present. "
                  "If an anomaly is detected, specify its type and location, "
                  "and provide a detailed reasoning for your conclusion.")
        existing_user_prompt[r['image_path']] = prompt

# ── Build the final dataset ──
records = []

# Add held-out items
for img_path in heldout_paths:
    if img_path in heldout_corrected:
        trace = heldout_corrected[img_path]
    elif img_path in heldout_rewritten:
        trace = heldout_rewritten[img_path]
    elif img_path in heldout_kept:
        trace = heldout_kept[img_path]
    else:
        continue
    records.append({
        "messages": [
            {"role": "user", "content": heldout_user_prompt[img_path]},
            {"role": "assistant", "content": trace},
        ],
        "images": [img_path],
    })
n_heldout = len(records)
print(f"\nAdded {n_heldout} held-out items")

# Add 197 OK + 197 NG balanced patches
for p in ok_patches + ng_selected:
    records.append({
        "messages": [
            {"role": "user", "content": existing_user_prompt.get(p, '')},
            {"role": "assistant", "content": existing_patches[p]},
        ],
        "images": [p],
    })
n_patches = len(records) - n_heldout
print(f"Added {n_patches} balanced patches")

random.shuffle(records)
with open(OUT, 'w') as f:
    json.dump(records, f, indent=2)
print(f"\n✓ Wrote {OUT}")
print(f"  Total records: {len(records)}")
print(f"  File size: {os.path.getsize(OUT)/1024/1024:.1f} MB")

# Final composition sanity
from collections import Counter
ng = sum(1 for r in records if '/NG/' in r['images'][0])
ok = sum(1 for r in records if '/OK/' in r['images'][0])
print(f"  NG-path items: {ng}")
print(f"  OK-path items: {ok}")
