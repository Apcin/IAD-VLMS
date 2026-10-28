#!/usr/bin/env python3
"""Build the 10K SFT dataset: max-polish (kept + corrected + rewritten, no overlap, 10,236 total).

Output: datasets_sft_iter2/sft_10k_max_polish_train.json
Schema: matches sft_iter2_train.json — list of {messages: [user, assistant], images: [path]}
"""
import json, os, re
from pathlib import Path

BASE = Path("/bulk/aacudad/reasoning_traces/Training/phase0_full_10k_20260529_015821")
OUT  = Path("/bulk/aacudad/reasoning_traces/Training/datasets_sft_iter2/sft_10k_max_polish_train.json")

USER_PROMPT_TPL = (
    "<image>\nAnalyze the provided image of the {product}. Determine if there are any "
    "anomalies present. If an anomaly is detected, specify its type and location, and "
    "provide a detailed reasoning for your conclusion."
)

def product_from_path(p: str) -> str:
    # .../Real-IAD/images/PRODUCT/(NG|OK)/...
    m = re.search(r"/Real-IAD/images/([^/]+)/", p)
    return m.group(1) if m else "object"

def make_record(image_path: str, assistant_trace: str) -> dict:
    product = product_from_path(image_path)
    return {
        "messages": [
            {"role": "user",      "content": USER_PROMPT_TPL.format(product=product)},
            {"role": "assistant", "content": assistant_trace},
        ],
        "images": [image_path],
    }

# Load sources
good      = json.load(open(BASE / "good_traces.json"))
corrected = [json.loads(l) for l in open(BASE / "gemini_corrected.jsonl")]
rewritten = [json.loads(l) for l in open(BASE / "gemini_rewritten.jsonl")]

# Build image-path → source-of-record map
# Priority: rewritten > corrected > good (rewritten overrides good for the 2,406 weak ones)
rewritten_paths = {r["image_path"] for r in rewritten}
corrected_paths = {r["image_path"] for r in corrected}

records = []

# 1. Rewritten items (2,406) — these replace the kept versions for the 'weak' subset
for r in rewritten:
    records.append(make_record(r["image_path"], r["corrected_trace"]))

# 2. Corrected items (1,364) — Gemini fix of originally-failed rollouts
for r in corrected:
    records.append(make_record(r["image_path"], r["corrected_trace"]))

# 3. Plain-kept (good \ rewritten) — original model rollout traces
seen_paths = rewritten_paths | corrected_paths
plain_kept_n = 0
for g in good:
    img = g["images"][0]
    if img in seen_paths:
        continue
    # good_traces.json already has the right messages/images schema — keep as is
    records.append(g)
    plain_kept_n += 1

print(f"  rewritten   : {len(rewritten):5}  (Gemini-rewritten weak-but-kept)")
print(f"  corrected   : {len(corrected):5}  (Gemini-fixed failures)")
print(f"  plain-kept  : {plain_kept_n:5}  (good \\ rewritten)")
print(f"  TOTAL       : {len(records):5}")

# Quick yes/no class check
yes_no = {"yes": 0, "no": 0, "?": 0}
for r in records:
    a = r["messages"][1]["content"].lower()
    m = re.search(r"<answer>\s*(yes|no)\s*</answer>", a)
    yes_no[m.group(1) if m else "?"] += 1
print(f"  NG/OK by <answer> tag: {yes_no}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(records, f, indent=None)  # one big array, no per-item indent
print(f"\nWrote {len(records)} records to {OUT} ({OUT.stat().st_size/1024/1024:.1f} MB)")
