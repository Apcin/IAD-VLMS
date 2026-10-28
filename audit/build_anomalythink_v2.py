#!/usr/bin/env python3
"""Build AnomalyThink-v2: the 6K SFT corpus with grounding-audit corrections applied.

Same 6,000 images and the same prompts as v1. Only the assistant trace changes,
and only for traces the Gemini-3.7-Flash grounding audit flagged as ungrounded.
Corpus size is held at exactly 6,000 so a v1-vs-v2 SFT run is a controlled A/B on
trace quality alone.
"""
import json, os, re, collections

R = "/bulk/aacudad/reasoning_traces/Training"
V1 = f"{R}/datasets_small_new_v4/combined_6k_train.json"
AUDIT = f"{R}/trace_audit_gemini/audit_grounding.jsonl"
CORPUS = f"{R}/trace_audit_gemini/corpus.jsonl"
OUTDIR = f"{R}/datasets_anomalythink_v2"

aud = {}
for line in open(AUDIT):
    d = json.loads(line)
    aud[d["id"]] = d

by_image = {}
for line in open(CORPUS):
    d = json.loads(line)
    by_image[os.path.realpath(d["image_path"])] = d

v1 = json.load(open(V1))
tags = lambda t: tuple(re.findall(r"</?([a-z_]+)>", t or ""))

v2, manifest = [], []
stat = collections.Counter()
for it in v1:
    img = os.path.realpath(it["images"][0])
    rec = by_image[img]
    a = aud[rec["id"]]
    orig = it["messages"][1]["content"]
    ct = (a.get("corrected_trace") or "").strip()
    flagged = a["category"] != "completely_correct"

    if not flagged:
        action = "kept_clean"
        new = orig
    elif ct and tags(ct) == tags(orig):
        action = "corrected"
        new = ct
    else:
        # flagged but unfixable: the audit deliberately withholds a rewrite when the
        # defect is not visible in the evaluated view. Keep v1 so the A/B stays controlled.
        action = "unfixable_kept_original"
        new = orig

    stat[action] += 1
    out = json.loads(json.dumps(it))
    out["messages"][1]["content"] = new
    v2.append(out)
    manifest.append({
        "id": rec["id"], "product": rec["product"], "gold_label": rec["gold_label"],
        "action": action, "category": a["category"],
        "not_visible_in_view": a.get("not_visible_in_view", False),
        "issue_types": sorted({i["issue_type"] for i in a["issues"]}),
        "n_issues": len(a["issues"]),
        "words_v1": len(orig.split()), "words_v2": len(new.split()),
    })

assert len(v2) == 6000
json.dump(v2, open(f"{OUTDIR}/anomalythink_v2_train.json", "w"), ensure_ascii=False, indent=1)
with open(f"{OUTDIR}/anomalythink_v2_manifest.jsonl", "w") as f:
    for m in manifest:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

changed = [m for m in manifest if m["action"] == "corrected"]
print("AnomalyThink-v2 built:", len(v2), "items")
for k, v in stat.most_common():
    print(f"  {k:26s} {v:5d}  {100*v/len(v2):5.2f}%")
print("\nissue types among corrected traces:")
c = collections.Counter(t for m in changed for t in m["issue_types"])
for k, v in c.most_common():
    print(f"  {k:22s} {v:4d}")
print("\nunfixable (v1 trace retained):")
for m in manifest:
    if m["action"] == "unfixable_kept_original":
        print("  ", m["category"], m["id"][:58])
w1 = sum(m["words_v1"] for m in changed); w2 = sum(m["words_v2"] for m in changed)
print(f"\nwords in corrected traces: v1 {w1}  v2 {w2}  ({100*(w2-w1)/w1:+.2f}%)")
