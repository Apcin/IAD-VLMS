#!/usr/bin/env python3
"""Analysis of the 200-item calibration pass.

Clusters the free-text why_wrong strings into failure families with keyword rules,
scores the transistor1 subset against the verified 9-per-side / 18-total truth,
and extrapolates wall clock and token cost to the full 10,236-item pool.
"""
# Calibration inputs (calib_200.jsonl, calib_200_pass1.jsonl, calib_repeat_40_pass2.jsonl,
# calib_rerun50_v2_c16.jsonl) are NOT shipped in this repository, see README.md in this
# directory. Regenerate them with make_calib_sample.py + audit_traces_gemini.py first.
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASS1 = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "calib_200_pass1.jsonl")
SAMPLE = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "calib_200.jsonl")

corp = {json.loads(l)["id"]: json.loads(l) for l in open(SAMPLE)}
rows = [json.loads(l) for l in open(PASS1)]
ok = [r for r in rows if r["status"] == "ok"]

print("=" * 78)
print(f"CALIBRATION  n={len(rows)}  ok={len(ok)}  failed={len(rows)-len(ok)}")
print("=" * 78)

# ------------------------------------------------------------------ categories
cats = ["completely_correct", "correct", "wrong", "completely_wrong"]
c_all = Counter(r["category"] for r in ok)
print("\n--- category distribution (all 200) ---")
for c in cats:
    print(f"  {c:20s} {c_all[c]:4d}   {100*c_all[c]/len(ok):5.1f} %")
flagged = sum(c_all[c] for c in ("wrong", "completely_wrong"))
print(f"  {'FLAGGED (wrong+cw)':20s} {flagged:4d}   {100*flagged/len(ok):5.1f} %")

print("\n--- category by gold label ---")
for g, name in (("no", "normal"), ("yes", "anomalous")):
    sub = [r for r in ok if r["gold_label"] == g]
    cc = Counter(r["category"] for r in sub)
    f = cc["wrong"] + cc["completely_wrong"]
    print(f"  {name:10s} n={len(sub):3d} | " +
          " ".join(f"{c.replace('completely_','c_')}={cc[c]}" for c in cats) +
          f" | flagged {f} ({100*f/len(sub):.1f} %)")

print("\n--- category by split ---")
for s in ("sft_6k", "grpo_4k"):
    sub = [r for r in ok if corp[r["id"]]["split"] == s]
    cc = Counter(r["category"] for r in sub)
    f = cc["wrong"] + cc["completely_wrong"]
    print(f"  {s:10s} n={len(sub):3d} | " +
          " ".join(f"{c.replace('completely_','c_')}={cc[c]}" for c in cats) +
          f" | flagged {f} ({100*f/len(sub):.1f} %)")

print("\n--- anomalous, mask vs no mask ---")
for has, name in ((True, "with mask"), (False, "no mask")):
    sub = [r for r in ok if r["gold_label"] == "yes" and bool(r["mask_path"]) == has]
    cc = Counter(r["category"] for r in sub)
    f = cc["wrong"] + cc["completely_wrong"]
    print(f"  {name:10s} n={len(sub):3d} | " +
          " ".join(f"{c.replace('completely_','c_')}={cc[c]}" for c in cats) +
          f" | flagged {f} ({100*f/len(sub) if sub else 0:.1f} %)")

print("\n--- tags_ok ---")
t = Counter((r["gold_label"], r["tags_ok"]) for r in ok)
print(f"  overall false: {sum(v for k, v in t.items() if k[1] is False)} / {len(ok)}")
for g in ("no", "yes"):
    n = sum(v for k, v in t.items() if k[0] == g)
    print(f"  {g:4s} tags_ok=False {t[(g, False)]:3d} / {n}")

# ------------------------------------------------------- failure families
FAM = [
    ("count_error",        r"\bcount|\bcounts\b|number of|there are (only )?\d|miscount|"
                           r"\d+ (pins|leads|terminals|teeth|holes|beads|slots|screws|segments|"
                           r"contacts|bricks|cells|studs|prongs)"),
    ("verdict_mismatch",   r"gold verdict|verdict (is|contradicts)|answer tag (contradicts|is)|"
                           r"concludes (that )?(there are )?no defect"),
    ("wrong_defect_type",  r"not (a |an )?(scratch|dent|flash|burr|crack|chip|pit|contamination|"
                           r"deformation|abrasion)|defect is (a |an )?\w+, not|"
                           r"is missing (material|parts)|type tag|misidentif|mischaracter"),
    ("wrong_location",     r"\b(left|right|top|bottom|centre|center|upper|lower)\b.{0,40}\bnot\b|"
                           r"location tag|region is|not on the|is on the .{0,25}not"),
    ("overlay_confusion",  r"overlay|red (mask|region|fill)|mask (colour|color)"),
    ("not_visible_view",   r"not visible in this (camera )?view|only (a |the )?(flat |top.?down )?"
                           r"view|no .{0,30} visible in this view|camera view (only )?shows"),
    ("colour_finish",      r"colou?r|finish|matte|glossy|shiny|texture|material is|plastic|metal"),
    ("marking_text",       r"marking|etched|printed|text (says|reads)|label|logo|part number|"
                           r"legible|inscription"),
    ("feature_absent",     r"there is no |no such |does not (have|exist)|not present on|"
                           r"no .{0,25} (hole|slot|groove|ridge|notch|indentation) "),
    ("uncertain_count",    r"^UNCERTAIN_COUNT"),
]
fam_hits = Counter()
fam_items = defaultdict(set)
unmatched = []
for r in ok:
    for i in r["issues"]:
        w = i["why_wrong"]
        hit = False
        for name, pat in FAM:
            if re.search(pat, w, re.I):
                fam_hits[name] += 1
                fam_items[name].add(r["id"])
                hit = True
        if not hit:
            unmatched.append((r["id"], w))
print(f"\n--- failure families over {sum(len(r['issues']) for r in ok)} issue entries "
      f"(one entry can match several) ---")
for name, n in fam_hits.most_common():
    print(f"  {name:20s} {n:4d} entries   {len(fam_items[name]):3d} traces")
print(f"  {'(no family matched)':20s} {len(unmatched):4d} entries")

# ---------------------------------------------------------- transistor1 subset
print("\n" + "=" * 78)
print("TRANSISTOR1 SUBSET  (verified truth: 9 leads per side, 18 total)")
print("=" * 78)
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
       "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
       "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
       "twenty": 20}
UNIT = r"(?:pins?|leads?|terminals?|contacts?|legs?)"


def stated_counts(trace):
    """Every lead/pin count the trace states, with per-side / total scope."""
    out = []
    t = trace
    for m in re.finditer(rf"\b(\d{{1,2}}|{'|'.join(NUM)})[- ]?(?:metallic |silver-?colou?red |"
                         rf"gold-?colou?red |shiny |visible )*{UNIT}\b", t, re.I):
        v = m.group(1)
        v = int(v) if v.isdigit() else NUM[v.lower()]
        ctx = t[max(0, m.start() - 90):m.end() + 90].lower()
        if v == 1:
            continue
        scope = "total"
        if re.search(r"each side|per side|on (the )?(top|bottom|either) (row|side|edge)|"
                     r"on each (row|edge)|both sides", ctx):
            scope = "per_side"
        out.append((v, scope, m.group(0)))
    for m in re.finditer(r"\bSOI?[CP]-?(\d{1,2})\b|\b(\d{1,2})-pin\b", t, re.I):
        v = int(m.group(1) or m.group(2))
        out.append((v, "total", m.group(0)))
    return out


def det_wrong(trace):
    """True if any stated count contradicts 9-per-side / 18-total."""
    cs = stated_counts(trace)
    if not cs:
        return None, cs
    for v, scope, _ in cs:
        if scope == "per_side" and v != 9:
            return True, cs
        if scope == "total" and v not in (18,):
            return True, cs
    return False, cs


tr = [r for r in ok if r["product"] == "transistor1"]
det_states, det_bad, agree, disagree_missed, disagree_false = 0, 0, 0, 0, 0
print(f"\n  n = {len(tr)}  (10 normal, 10 anomalous)")
cc = Counter(r["category"] for r in tr)
print("  categories:", {c: cc[c] for c in cats if cc[c]})
rows_out = []
for r in tr:
    tracetext = corp[r["id"]]["trace"]
    dw, cs = det_wrong(tracetext)
    judge_count_flag = any(re.search(FAM[0][1], i["why_wrong"], re.I) for i in r["issues"])
    if dw is not None:
        det_states += 1
        if dw:
            det_bad += 1
            if judge_count_flag:
                agree += 1
            else:
                disagree_missed += 1
        else:
            if judge_count_flag:
                disagree_false += 1
    rows_out.append((r["id"], r["gold_label"], r["category"], dw, judge_count_flag,
                     [c[2] for c in cs]))
print(f"\n  deterministic regex on the same 20 traces:")
print(f"    states a lead/pin count : {det_states}")
print(f"    of those, count wrong   : {det_bad}  ({100*det_bad/det_states:.1f} %)"
      f"   [corpus-wide reference: 91 %]")
print(f"\n  judge vs the deterministic rule, on the {det_bad} traces with a wrong count:")
print(f"    judge also flagged a count problem : {agree}  ({100*agree/det_bad:.1f} % recall)")
print(f"    judge missed it                    : {disagree_missed}")
print(f"    judge flagged a count on a trace whose stated count is right : {disagree_false}")
print(f"\n  flagged (wrong+completely_wrong) on transistor1: "
      f"{cc['wrong']+cc['completely_wrong']} / {len(tr)}")
FLAG = "transistor1_OK_S0234_transistor1_0234_OK_C1_20230923170453"
fl = [r for r in tr if r["id"] == FLAG]
if fl:
    r = fl[0]
    print(f"\n  THESIS FLAGSHIP TRACE ({FLAG[:40]}...)")
    print(f"    category   : {r['category']}   confidence {r['confidence']}")
    print(f"    tags_ok    : {r['tags_ok']}")
    for i in r["issues"]:
        print(f"    issue      : claim={i['claim'][:70]!r}")
        print(f"                 why={i['why_wrong'][:100]}")
        print(f"                 fix={i['what_it_should_be'][:100]}")
print("\n  per-trace detail (id | gold | category | det_count_wrong | judge_count_flag | counts)")
for iid, g, c, dw, jf, cs in sorted(rows_out, key=lambda x: (x[1], x[0])):
    print(f"    {iid[-28:]:30s} {g:4s} {c:18s} det={str(dw):5s} judge={str(jf):5s} {cs}")

# -------------------------------------------------------------- cost and time
print("\n" + "=" * 78)
print("THROUGHPUT AND COST")
print("=" * 78)
lat = [r["latency_s"] for r in ok]
ptok = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
otok = [r["output_tokens"] for r in ok if r.get("output_tokens") is not None]
thk = [r.get("thought_tokens") or 0 for r in ok if r.get("prompt_tokens")]
print(f"  per-item latency  mean {st.mean(lat):6.1f} s   median {st.median(lat):6.1f} s   "
      f"p90 {sorted(lat)[int(.9*len(lat))]:6.1f} s   max {max(lat):6.1f} s")
print(f"  input tokens      mean {st.mean(ptok):8.0f}  median {st.median(ptok):8.0f}")
print(f"  output tokens     mean {st.mean(otok):8.0f}  median {st.median(otok):8.0f}  "
      f"(answer text only)")
print(f"  thinking tokens   mean {st.mean(thk):8.0f}")
print(f"  billed output     mean {st.mean(otok)+st.mean(thk):8.0f}  (answer + thinking)")
for g, name in (("no", "normal   "), ("yes", "anomalous")):
    sub = [r for r in ok if r["gold_label"] == g and r.get("prompt_tokens")]
    print(f"    {name}  in {st.mean([r['prompt_tokens'] for r in sub]):7.0f}  "
          f"out {st.mean([r['output_tokens'] for r in sub]):5.0f}  "
          f"lat {st.mean([r['latency_s'] for r in sub]):5.1f} s  n={len(sub)}")
N = 10236
print(f"\n  measured wall clock for this 200-item pass at concurrency 8:")
print(f"    see the log. sec/item below is derived from that wall clock, not from latency.")
