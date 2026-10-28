#!/usr/bin/env python3
"""Prompt v1 vs prompt v2 on the same 50 items."""
# Calibration inputs (calib_200.jsonl, calib_200_pass1.jsonl, calib_repeat_40_pass2.jsonl,
# calib_rerun50_v2_c16.jsonl) are NOT shipped in this repository, see README.md in this
# directory. Regenerate them with make_calib_sample.py + audit_traces_gemini.py first.
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
corp = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "calib_200.jsonl")}
v1 = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "calib_200_pass1.jsonl")}
v2 = {json.loads(l)["id"]: json.loads(l)
      for l in open(sys.argv[1] if len(sys.argv) > 1 else HERE / "calib_rerun50_v2_c16.jsonl")}
ids = sorted(i for i in v2 if v2[i]["status"] == "ok" and v1.get(i, {}).get("status") == "ok")
print(f"paired items: {len(ids)}")

CATS = ["completely_correct", "correct", "wrong", "completely_wrong"]
print("\n--- category distribution ---")
print(f"  {'':22s} {'v1':>6s} {'v2':>6s}")
for c in CATS:
    print(f"  {c:22s} {sum(v1[i]['category']==c for i in ids):6d} "
          f"{sum(v2[i]['category']==c for i in ids):6d}")

wc = lambda s: len(s.split())
tags = lambda s: re.findall(r"</?(\w+)>", s)

BAD = re.compile(r"reference normal|the reference\b|gold verdict|overlay|red (mask|region|overlay)|"
                 r"compar\w+ (with|to) (the )?(specification|reference|standard|normal)|"
                 r"specification sheet|the dataset|quality records|according to (the )?(label|record)",
                 re.I)


def audit_rewrites(store, name):
    rw = [i for i in ids if store[i]["corrected_trace"].strip()]
    nvv = [i for i in ids if store[i].get("not_visible_in_view")]
    nomask = [i for i in ids if corp[i]["gold_label"] == "yes" and not corp[i]["mask_path"]]
    leak = [i for i in rw if BAD.search(store[i]["corrected_trace"])]
    tagchg = [i for i in rw if tags(corp[i]["trace"]) != tags(store[i]["corrected_trace"])]
    big = [i for i in rw
           if abs(wc(store[i]["corrected_trace"]) - wc(corp[i]["trace"])) / wc(corp[i]["trace"]) > 0.10]
    rw_nomask = [i for i in rw if i in nomask]
    print(f"\n--- {name} rewrite hygiene ---")
    print(f"  rewrites produced                         : {len(rw)}")
    print(f"  not_visible_in_view flag set              : {len(nvv)}")
    print(f"  REWRITES ON no-mask anomalous items       : {len(rw_nomask)} of {len(nomask)}"
          f"   <-- fabrication risk")
    print(f"  rewrite leaks audit vocabulary            : {len(leak)}")
    print(f"  rewrite changes the tag structure         : {len(tagchg)}")
    print(f"  rewrite outside the +-10 % length rule    : {len(big)} of {len(rw)}")
    return set(rw_nomask), set(leak), set(tagchg)


a = audit_rewrites(v1, "v1")
b = audit_rewrites(v2, "v2")

# ---------------------------------------------------------------- transistor1
print("\n--- transistor1 count recall (deterministic truth: 9 per side / 18 total) ---")
det = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "calib_countable.jsonl")}
COUNTW = re.compile(r"\bcount|\bpins?\b|\bleads?\b|\bterminals?\b|\bcontacts?\b|SOIC|SOP|DIP", re.I)


def count_flagged(rec):
    for it in rec.get("issues") or []:
        if it.get("issue_type") == "count_error":
            return True
        blob = " ".join([it.get("claim", ""), it.get("why_wrong", ""),
                         it.get("what_it_should_be", "")])
        if it.get("why_wrong", "").strip().upper().startswith("UNCERTAIN_COUNT"):
            continue
        if COUNTW.search(blob) and re.search(r"\d|\bnine\b|\beight\b|\bten\b|\bsixteen\b|"
                                             r"\beighteen\b|\bseven\b|\bfourteen\b", blob, re.I):
            return True
    return False


tr = [i for i in ids if corp[i]["product"] == "transistor1"]
wrongset = [i for i in tr if det[i]["det_verdict"] == "count_wrong"]
rightset = [i for i in tr if det[i]["det_verdict"] == "count_correct"]
print(f"  transistor1 items in the 50            : {len(tr)}")
print(f"  deterministic: count wrong / right / not stated : {len(wrongset)} / {len(rightset)} / "
      f"{len(tr)-len(wrongset)-len(rightset)}")
for name, store in (("v1", v1), ("v2", v2)):
    rec = sum(count_flagged(store[i]) for i in wrongset)
    fp = sum(count_flagged(store[i]) for i in rightset)
    print(f"  {name}: caught {rec}/{len(wrongset)} wrong counts = {100*rec/len(wrongset):.0f} % recall"
          f" | false flags on right counts {fp}/{len(rightset)}")
# v2 also writes its own counts down
nc = sum(len(v2[i].get("counts") or []) for i in tr)
mism = [(i, c) for i in tr for c in (v2[i].get("counts") or [])
        if c.get("i_counted") is not None and c.get("trace_says") is not None
        and c["i_counted"] != c["trace_says"]]
nine = [c for i in tr for c in (v2[i].get("counts") or []) if c.get("i_counted") in (9, 18)]
allc = [c for i in tr for c in (v2[i].get("counts") or []) if c.get("i_counted") is not None]
print(f"  v2 wrote {nc} count entries on transistor1; {len(mism)} disagree with the trace")
print(f"  v2 own lead counts equal to the verified 9 or 18: {len(nine)} of {len(allc)}")

print("\n--- flagship trace ---")
F = "transistor1_OK_S0234_transistor1_0234_OK_C1_20230923170453"
for name, store in (("v1", v1), ("v2", v2)):
    if F in store:
        r = store[F]
        print(f"  {name}: {r['category']:19s} conf {r['confidence']} "
              f"counts={r.get('counts')}")

print("\n--- issue_type histogram (v2 only) ---")
for k, n in Counter(it.get("issue_type") for i in ids for it in v2[i]["issues"]).most_common():
    print(f"  {str(k):22s} {n}")

print("\n--- category moves v1 -> v2 ---")
for k, n in Counter((v1[i]["category"], v2[i]["category"]) for i in ids
                    if v1[i]["category"] != v2[i]["category"]).most_common():
    print(f"  {k[0]:19s} -> {k[1]:19s} {n}")
same = sum(v1[i]["category"] == v2[i]["category"] for i in ids)
print(f"  identical: {same}/{len(ids)} = {100*same/len(ids):.1f} %")
