#!/usr/bin/env python3
"""Build Arm A/B/C SFT datasets from the LLaVA phase-0/1a/1b pipeline outputs,
mirroring the Qwen iter-2 arm recipe:
  A = kept only, 50/50 balanced
  B = kept + corrected, 50/50 balanced
  C = kept + corrected + rewritten (rewrite takes precedence per image), 50/50, capped 6,000
Schema matches sft_iter2_train.json: {messages:[user,assistant], images:[path]}.
Usage: build_llava_arms.py <phase0_dir> <out_dir>
"""
import json
import random
import re
import sys
from pathlib import Path

random.seed(42)
BASE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

USER_PROMPT_TPL = (
    "<image>\nAnalyze the provided image of the {product}. Determine if there are any "
    "anomalies present. If an anomaly is detected, specify its type and location, and "
    "provide a detailed reasoning for your conclusion."
)


def product_from_path(p):
    m = re.search(r"/Real-IAD/images/([^/]+)/", p)
    return m.group(1) if m else "object"


def is_ng(p):
    return ("/NG/" in p) or ("_NG_" in p)


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
pool = {}  # image_path -> (trace, tier); precedence: corrected > kept; rewrites overlay in C
for it in good:
    p, t = pair_of_good(it)
    pool[p] = (t, "kept")
for it in corrected:
    p, t = pair_of_fixed(it)
    pool[p] = (t, "corrected")
rewrite_map = dict(pair_of_fixed(it) for it in rewritten)


def balance(items, cap=None):
    ng = [x for x in items if is_ng(x[0])]
    ok = [x for x in items if not is_ng(x[0])]
    n = min(len(ng), len(ok))
    if cap:
        n = min(n, cap // 2)
    random.shuffle(ng); random.shuffle(ok)
    out = ng[:n] + ok[:n]
    random.shuffle(out)
    return out, len(ng), len(ok)


def dump(name, items):
    data = [rec(p, t) for p, t in items]
    with open(OUT / name, "w") as f:
        json.dump(data, f)
    return len(data)


# Arm A: kept only
a_items = [(p, tr) for p, (tr, tier) in pool.items() if tier == "kept"]
a_bal, ang, aok = balance(a_items)
na = dump("sft_llava_A_kept_balanced.json", a_bal)

# Arm B: kept + corrected
b_items = [(p, tr) for p, (tr, tier) in pool.items()]
b_bal, bng, bok = balance(b_items)
nb = dump("sft_llava_B_kept_corrected.json", b_bal)

# Arm C: kept + corrected, with rewrites replacing their images' traces; cap 6000
c_items = [(p, rewrite_map.get(p, tr)) for p, (tr, tier) in pool.items()]
c_bal, cng, cok = balance(c_items, cap=6000)
nc = dump("sft_llava_C_train.json", c_bal)

print(f"Arm A: {na} (from NG {ang}/OK {aok})")
print(f"Arm B: {nb} (from NG {bng}/OK {bok})")
print(f"Arm C: {nc} (from NG {cng}/OK {cok}, rewrites applied to {sum(1 for p,_ in c_items if p in rewrite_map)} images)")
