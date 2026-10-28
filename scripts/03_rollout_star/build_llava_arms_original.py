import os
#!/usr/bin/env python3
"""Build the LLaVA Arm-C SFT dataset the ORIGINAL way, i.e. the way the Qwen
iter-2 Arm C was built, so that LLaVA-C and Qwen-C differ in one factor only
(who produced the traces) and not in which images they cover.

Differences from build_llava_arms.py, and there are exactly two:
  1. The candidate pool is restricted to the 6,000 images of the 6K SFT split
     (Training/datasets_small_new_v4/combined_6k_train.json). The old script
     drew from the whole 10,236-image rollout pool, so it leaked 2,484 images
     of the GRPO split into Arm C.
  2. balance() balances on the verdict parsed from the trace's own <answer>
     tag, not on whether "_NG_" / "/NG/" appears in the file path. The folder
     name is not what the trace teaches: NG-folder captures without a C1 mask
     carry a correct "No" trace, and the old script counted those as positives.

Everything else is kept: the keep > correct > rewrite precedence, the record
schema, the user prompt template, the caps and the random seed.

Only Arm C is written, as sft_llava_C_original_train.json. Arm A and Arm B are
still computed (so the RNG stream and the printed stats stay comparable) but
they are not dumped, because writing them would overwrite the old Arm A/B files
and the old corpus has to stay reproducible.

Usage: build_llava_arms_original.py <phase0_dir> <out_dir> [<sft_split_json>]
"""
import json
import random
import re
import sys
from pathlib import Path

random.seed(42)
BASE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SFT_SPLIT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
    (os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+"/Training/datasets_small_new_v4/combined_6k_train.json"))
OUT.mkdir(parents=True, exist_ok=True)

USER_PROMPT_TPL = (
    "<image>\nAnalyze the provided image of the {product}. Determine if there are any "
    "anomalies present. If an anomaly is detected, specify its type and location, and "
    "provide a detailed reasoning for your conclusion."
)

# Same answer regex the reward / rollout scorer uses
# (Training/phase0_rollout_kscoring.py, extract of gt and model answer).
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def product_from_path(p):
    m = re.search(r"/Real-IAD/images/([^/]+)/", p)
    return m.group(1) if m else "object"


def is_ng(p):
    return ("/NG/" in p) or ("_NG_" in p)


def verdict_of(trace):
    """Verdict the trace actually teaches: yes / no / None if unparseable."""
    m = ANSWER_RE.search(trace or "")
    return m.group(1).strip().lower() if m else None


def split_key(p):
    """Stored paths are absolute, so compare from 'Real-IAD/' onwards."""
    i = p.find("Real-IAD/")
    return p[i:] if i >= 0 else p


def rec(image_path, trace):
    return {"messages": [
        {"role": "user", "content": USER_PROMPT_TPL.format(product=product_from_path(image_path))},
        {"role": "assistant", "content": trace}],
        "images": [image_path]}


def pair_of_good(item):
    # good_traces.json items are already sharegpt records: {messages:[user,assistant], images:[path]}
    return item["images"][0], item["messages"][1]["content"]


def pair_of_fixed(item):
    # gemini_corrected/rewritten jsonl: {image_path, corrected_trace, success, ...}
    return item["image_path"], item["corrected_trace"]


good = json.load(open(BASE / "good_traces.json"))
corrected = [json.loads(l) for l in open(BASE / "gemini_corrected.jsonl")] if (BASE / "gemini_corrected.jsonl").exists() else []
rewritten = [json.loads(l) for l in open(BASE / "gemini_rewritten.jsonl")] if (BASE / "gemini_rewritten.jsonl").exists() else []
print(f"sources: kept={len(good)} corrected={len(corrected)} rewritten={len(rewritten)}")

corrected = [it for it in corrected if it.get("success", True) and it.get("corrected_trace")]
rewritten = [it for it in rewritten if it.get("success", True) and it.get("corrected_trace")]
print(f"after success-filter: corrected={len(corrected)} rewritten={len(rewritten)}")

# CHANGE 1: reference image set = the 6K SFT split, matched from "Real-IAD/" onwards.
sft_split = json.load(open(SFT_SPLIT))
sft_keys = {split_key(r["images"][0]) for r in sft_split}
print(f"sft split: {len(sft_split)} records, {len(sft_keys)} unique images ({SFT_SPLIT})")

pool = {}  # image_path -> (trace, tier); precedence: corrected > kept; rewrites overlay in C
for it in good:
    p, t = pair_of_good(it)
    pool[p] = (t, "kept")
for it in corrected:
    p, t = pair_of_fixed(it)
    pool[p] = (t, "corrected")
rewrite_map = dict(pair_of_fixed(it) for it in rewritten)

n_pool_all = len(pool)
pool = {p: v for p, v in pool.items() if split_key(p) in sft_keys}
print(f"pool restricted to sft split: {n_pool_all} -> {len(pool)} images "
      f"({len(sft_keys) - len(pool)} sft-split images missing from the rollout pool)")


def balance(items, cap=None):
    # CHANGE 2: balance on the parsed <answer> verdict, not on the folder name.
    yes = [x for x in items if verdict_of(x[1]) == "yes"]
    no = [x for x in items if verdict_of(x[1]) == "no"]
    dropped = len(items) - len(yes) - len(no)
    if dropped:
        print(f"  warning: {dropped} item(s) with an unparseable <answer>, excluded")
    n = min(len(yes), len(no))
    if cap:
        n = min(n, cap // 2)
    random.shuffle(yes); random.shuffle(no)
    out = yes[:n] + no[:n]
    random.shuffle(out)
    return out, len(yes), len(no)


def dump(name, items):
    data = [rec(p, t) for p, t in items]
    with open(OUT / name, "w") as f:
        json.dump(data, f)
    return len(data)


# Arm A: kept only (computed, not written)
a_items = [(p, tr) for p, (tr, tier) in pool.items() if tier == "kept"]
a_bal, ayes, ano = balance(a_items)
na = len(a_bal)

# Arm B: kept + corrected (computed, not written)
b_items = [(p, tr) for p, (tr, tier) in pool.items()]
b_bal, byes, bno = balance(b_items)
nb = len(b_bal)

# Arm C: kept + corrected, with rewrites replacing their images' traces; cap 6000
c_items = [(p, rewrite_map.get(p, tr)) for p, (tr, tier) in pool.items()]
c_bal, cyes, cno = balance(c_items, cap=6000)
nc = dump("sft_llava_C_original_train.json", c_bal)

print(f"Arm A: {na} (from yes {ayes}/no {ano}) [not written]")
print(f"Arm B: {nb} (from yes {byes}/no {bno}) [not written]")
print(f"Arm C: {nc} (from yes {cyes}/no {cno}, rewrites applied to "
      f"{sum(1 for p, _ in c_items if p in rewrite_map)} images) "
      f"-> {OUT / 'sft_llava_C_original_train.json'}")
