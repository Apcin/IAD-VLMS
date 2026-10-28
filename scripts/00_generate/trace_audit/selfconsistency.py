#!/usr/bin/env python3
"""Self-consistency of the judge: pass 1 vs pass 2 on the same 40 items."""
# Calibration inputs (calib_200.jsonl, calib_200_pass1.jsonl, calib_repeat_40_pass2.jsonl,
# calib_rerun50_v2_c16.jsonl) are NOT shipped in this repository, see README.md in this
# directory. Regenerate them with make_calib_sample.py + audit_traces_gemini.py first.
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
p1 = {json.loads(l)["id"]: json.loads(l)
      for l in open(sys.argv[1] if len(sys.argv) > 1 else HERE / "calib_200_pass1.jsonl")}
p2 = {json.loads(l)["id"]: json.loads(l)
      for l in open(sys.argv[2] if len(sys.argv) > 2 else HERE / "calib_repeat_40_pass2.jsonl")}

ids = [i for i in p2 if i in p1 and p1[i]["status"] == "ok" and p2[i]["status"] == "ok"]
ids.sort()
print(f"paired items: {len(ids)}")

FLAG = lambda c: c in ("wrong", "completely_wrong")

same_cat = sum(p1[i]["category"] == p2[i]["category"] for i in ids)
same_flag = sum(FLAG(p1[i]["category"]) == FLAG(p2[i]["category"]) for i in ids)
same_tags = sum(p1[i]["tags_ok"] == p2[i]["tags_ok"] for i in ids)
print(f"\nexact category identical      : {same_cat}/{len(ids)} = {100*same_cat/len(ids):.1f} %")
print(f"flag/no-flag identical        : {same_flag}/{len(ids)} = {100*same_flag/len(ids):.1f} %")
print(f"tags_ok identical             : {same_tags}/{len(ids)} = {100*same_tags/len(ids):.1f} %")

print("\nagreement by pass-1 category:")
for c in ("completely_correct", "correct", "wrong", "completely_wrong"):
    sub = [i for i in ids if p1[i]["category"] == c]
    if sub:
        s = sum(p1[i]["category"] == p2[i]["category"] for i in sub)
        print(f"  {c:20s} n={len(sub):3d}  identical {s:3d}  {100*s/len(sub):5.1f} %")

print("\nagreement by gold label:")
for g in ("no", "yes"):
    sub = [i for i in ids if p1[i]["gold_label"] == g]
    s = sum(p1[i]["category"] == p2[i]["category"] for i in sub)
    print(f"  {g:4s} n={len(sub):3d}  identical {s:3d}  {100*s/len(sub):5.1f} %")

print("\nconfusion (pass1 -> pass2), disagreements only:")
for i in ids:
    if p1[i]["category"] != p2[i]["category"]:
        print(f"  {i[-42:]:44s} {p1[i]['category']:19s} -> {p2[i]['category']}")
print("\ntransition counts:", dict(Counter((p1[i]["category"], p2[i]["category"])
                                           for i in ids if p1[i]["category"] != p2[i]["category"])))

n1 = sum(len(p1[i]["issues"]) for i in ids)
n2 = sum(len(p2[i]["issues"]) for i in ids)
print(f"\nissue entries: pass1 {n1}  pass2 {n2}")
both = [i for i in ids if FLAG(p1[i]["category"]) and FLAG(p2[i]["category"])]
print(f"items flagged in both passes: {len(both)}")
