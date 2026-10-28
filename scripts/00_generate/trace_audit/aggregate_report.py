#!/usr/bin/env python3
"""Aggregate report for the full grounding sweep over the 10,236-trace pool.

Reads the sweep output and the corpus, writes a markdown report to stdout and to
audit_report.md.

  python3 aggregate_report.py [audit_grounding.jsonl] [corpus.jsonl]

Runs fine while the sweep is still going. It reports coverage first, so a partial
file is reported as partial and never as a finished result.

Carry these three limits into any text written from this report. They come from
the 200-item calibration (make_calib_sample.py + analyse_calib.py).

  1. Count errors are a LOWER BOUND. The judge catches 71 % of them and its own
     lead count on transistor1 is right only 55.7 % of the time. Any published
     count-error rate must come from the deterministic rule below, not from the
     judge.
  2. Per-item verdicts flip about 7.5 % of the time between identical runs.
     Aggregate rates over 10,236 items are stable. A single quoted verdict is not.
  3. The `correct` category never fires. The scale is three-valued in practice.
"""
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "audit_grounding.jsonl"
CORPUS = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "corpus.jsonl"
OUTMD = HERE / "audit_report.md"

CATS = ["completely_correct", "correct", "wrong", "completely_wrong"]
FLAGGED = ("wrong", "completely_wrong")

L = []


def w(s=""):
    L.append(s)


def table(header, rows):
    w("| " + " | ".join(header) + " |")
    w("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        w("| " + " | ".join(str(x) for x in r) + " |")
    w()


# ------------------------------------------------------------------ load
corp = {}
for line in open(CORPUS):
    o = json.loads(line)
    corp[o["id"]] = o

seen = {}
dupes = 0
for line in open(RES):
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
    except json.JSONDecodeError:
        continue
    if o["id"] in seen:
        dupes += 1
    seen[o["id"]] = o                      # last record for an id wins
rows = list(seen.values())
ok = [r for r in rows if r["status"] == "ok"]
bad = [r for r in rows if r["status"] != "ok"]
missing = [i for i in corp if i not in seen]

w("# Grounding audit of the 10,236-trace pool")
w()
w(f"Source: `{RES}`  ")
w(f"Corpus: `{CORPUS}`  ")
if ok:
    w(f"Model: {ok[0].get('model')}  Prompt sha1: {ok[0].get('prompt_sha1')}")
w()

w("## 1. Coverage")
w()
table(["", "n", "share of pool"],
      [["corpus items", len(corp), "100.0 %"],
       ["records on disk (unique ids)", len(seen), f"{100*len(seen)/len(corp):.1f} %"],
       ["status ok", len(ok), f"{100*len(ok)/len(corp):.1f} %"],
       ["status not ok", len(bad), f"{100*len(bad)/len(corp):.1f} %"],
       ["never attempted", len(missing), f"{100*len(missing)/len(corp):.1f} %"],
       ["duplicate lines collapsed", dupes, ""]])
if bad:
    w("Failure statuses: " + ", ".join(f"{k} {v}" for k, v in
                                       Counter(r["status"] for r in bad).most_common()))
    w()
if missing or bad:
    w("**This is not a finished sweep.** Rerun `run_full_sweep.sh` before quoting "
      "anything below as a pool rate.")
    w()

if not ok:
    print("\n".join(L))
    OUTMD.write_text("\n".join(L))
    sys.exit(0)

N = len(ok)


def catrow(name, sub):
    if not sub:
        return [name, 0] + ["" for _ in CATS] + [""]
    c = Counter(r["category"] for r in sub)
    f = sum(c[x] for x in FLAGGED)
    return [name, len(sub)] + [c[x] for x in CATS] + [f"{100*f/len(sub):.1f} %"]


HDR = ["cut", "n"] + [c.replace("completely_", "c_") for c in CATS] + ["flagged"]

# ------------------------------------------------------------------ categories
w("## 2. Verdict distribution")
w()
c_all = Counter(r["category"] for r in ok)
flag_n = sum(c_all[c] for c in FLAGGED)
table(["category", "n", "share"],
      [[c, c_all[c], f"{100*c_all[c]/N:.1f} %"] for c in CATS] +
      [["**flagged (wrong + completely_wrong)**", flag_n, f"**{100*flag_n/N:.1f} %**"]])

w("### Cuts")
w()
cuts = [catrow("all", ok),
        catrow("normal", [r for r in ok if r["gold_label"] == "no"]),
        catrow("anomalous", [r for r in ok if r["gold_label"] == "yes"]),
        catrow("split sft_6k", [r for r in ok if corp[r["id"]]["split"] == "sft_6k"]),
        catrow("split grpo_4k", [r for r in ok if corp[r["id"]]["split"] == "grpo_4k"]),
        catrow("anomalous, mask exists",
               [r for r in ok if r["gold_label"] == "yes" and r["mask_path"]]),
        catrow("anomalous, no mask",
               [r for r in ok if r["gold_label"] == "yes" and not r["mask_path"]])]
table(HDR, cuts)

# ------------------------------------------------------------------ flags
w("## 3. Tag integrity and unusable items")
w()
tags_bad = [r for r in ok if r.get("tags_ok") is False]
nviv = [r for r in ok if r.get("not_visible_in_view")]
table(["", "n", "share of ok"],
      [["`tags_ok` false", len(tags_bad), f"{100*len(tags_bad)/N:.1f} %"],
       ["  of those, normal", sum(1 for r in tags_bad if r["gold_label"] == "no"), ""],
       ["  of those, anomalous", sum(1 for r in tags_bad if r["gold_label"] == "yes"), ""],
       ["`not_visible_in_view` true", len(nviv), f"{100*len(nviv)/N:.1f} %"],
       ["  of those, no C1 mask", sum(1 for r in nviv if not r["mask_path"]), ""]])
w("`not_visible_in_view` items cannot be repaired by a text rewrite. Route them out "
  "of the corpus, do not send them to a rewrite pass.")
w()

# ------------------------------------------------------------------ issue types
w("## 4. Issue types")
w()
it = Counter()
it_items = defaultdict(set)
untyped = 0
for r in ok:
    for i in r.get("issues") or []:
        t = (i.get("issue_type") or "").strip()
        if not t:
            untyped += 1
            t = "(untyped)"
        it[t] += 1
        it_items[t].add(r["id"])
tot_issues = sum(it.values())
w(f"{tot_issues} issue entries over {N} audited traces. One trace can carry several.")
w()
table(["issue_type", "entries", "traces", "share of pool"],
      [[t, n, len(it_items[t]), f"{100*len(it_items[t])/N:.1f} %"]
       for t, n in it.most_common()])
if untyped:
    w(f"{untyped} entries carried no `issue_type`. Prompt v2 should give 0.")
    w()

# ------------------------------------------------------------------ per product
w("## 5. Flag rate per product")
w()
per = defaultdict(list)
for r in ok:
    per[r["product"]].append(r)
prod_rows = []
for p, sub in per.items():
    c = Counter(r["category"] for r in sub)
    f = sum(c[x] for x in FLAGGED)
    nv = sum(1 for r in sub if r.get("not_visible_in_view"))
    prod_rows.append([p, len(sub), f, f"{100*f/len(sub):.1f} %", nv])
prod_rows.sort(key=lambda x: -float(x[3].rstrip(" %")))
table(["product", "n", "flagged", "flag rate", "not_visible"], prod_rows)

# ------------------------------------------------------------------ counting
w("## 6. The counting channel")
w()
w("The judge writes down every count it made before it decides. This is the noisiest "
  "channel in the whole audit, read it as a lower bound.")
w()
with_counts = [r for r in ok if r.get("counts")]
n_entries = sum(len(r["counts"]) for r in with_counts)
sure_dis = 0
unsure = 0
for r in with_counts:
    for c in r["counts"]:
        if c.get("trace_says") is None or c.get("i_counted") is None:
            continue
        if not c.get("sure", True):
            unsure += 1
        elif c["trace_says"] != c["i_counted"]:
            sure_dis += 1
table(["", "n"],
      [["traces where the judge recorded a count", len(with_counts)],
       ["count entries", n_entries],
       ["entries where the judge was sure and disagreed with the trace", sure_dis],
       ["entries the judge marked not sure", unsure]])

# deterministic transistor1 rule, the only publishable count number
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
       "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
       "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
       "twenty": 20}
UNIT = r"(?:pins?|leads?|terminals?|contacts?|legs?)"


def stated_counts(trace):
    out = []
    for m in re.finditer(rf"\b(\d{{1,2}}|{'|'.join(NUM)})[- ]?(?:metallic |silver-?colou?red |"
                         rf"gold-?colou?red |shiny |visible )*{UNIT}\b", trace, re.I):
        v = m.group(1)
        v = int(v) if v.isdigit() else NUM[v.lower()]
        if v == 1:
            continue
        ctx = trace[max(0, m.start() - 90):m.end() + 90].lower()
        scope = "total"
        if re.search(r"each side|per side|on (the )?(top|bottom|either) (row|side|edge)|"
                     r"on each (row|edge)|both sides", ctx):
            scope = "per_side"
        out.append((v, scope, m.group(0)))
    for m in re.finditer(r"\bSOI?[CP]-?(\d{1,2})\b|\b(\d{1,2})-pin\b", trace, re.I):
        out.append((int(m.group(1) or m.group(2)), "total", m.group(0)))
    return out


def det_wrong(trace):
    cs = stated_counts(trace)
    if not cs:
        return None
    for v, scope, _ in cs:
        if scope == "per_side" and v != 9:
            return True
        if scope == "total" and v != 18:
            return True
    return False


tr = [r for r in ok if r["product"] == "transistor1"]
states = bad_c = 0
judge_agreed = 0
for r in tr:
    dw = det_wrong(corp[r["id"]]["trace"])
    if dw is None:
        continue
    states += 1
    if dw:
        bad_c += 1
        if any((i.get("issue_type") or "") == "count_error" for i in r.get("issues") or []):
            judge_agreed += 1
w("### Deterministic rule on `transistor1` (truth: 9 leads per side, 18 total)")
w()
w("This rule uses no model. It is the number to publish.")
w()
table(["", "n"],
      [["transistor1 traces audited", len(tr)],
       ["state a lead or pin count", states],
       ["of those, count wrong", f"{bad_c}" +
        (f"  ({100*bad_c/states:.1f} %)" if states else "")],
       ["judge also filed a count_error on those", judge_agreed]])
w("Calibration reference: 91 to 93 % wrong, judge recall 71 %, judge precision 100 %.")
w()

# ------------------------------------------------------------------ rewrites
w("## 7. Rewrites")
w()
flag = [r for r in ok if r["category"] in FLAGGED]
has_rw = [r for r in flag if (r.get("corrected_trace") or "").strip()]
no_rw = [r for r in flag if not (r.get("corrected_trace") or "").strip()]
ratios = []
for r in has_rw:
    a = len(corp[r["id"]]["trace"].split())
    b = len(r["corrected_trace"].split())
    if a:
        ratios.append(b / a)
table(["", "n", "share of flagged"],
      [["flagged traces", len(flag), "100.0 %"],
       ["carry a rewrite", len(has_rw),
        f"{100*len(has_rw)/len(flag):.1f} %" if flag else ""],
       ["no rewrite (routed out)", len(no_rw),
        f"{100*len(no_rw)/len(flag):.1f} %" if flag else ""],
       ["  of those, not_visible_in_view",
        sum(1 for r in no_rw if r.get("not_visible_in_view")), ""]])
if ratios:
    outside = sum(1 for x in ratios if x < 0.9 or x > 1.1)
    table(["rewrite length vs original", "value"],
          [["median word ratio", f"{st.median(ratios):.3f}"],
           ["mean word ratio", f"{st.mean(ratios):.3f}"],
           ["outside the plus or minus 10 % rule",
            f"{outside} of {len(ratios)}  ({100*outside/len(ratios):.1f} %)"]])
leak_pat = re.compile(r"overlay|red mask|reference normal|compared to the reference|"
                      r"comparing with specifications|gold verdict|ground.truth|dataset label",
                      re.I)
leaks = [r["id"] for r in has_rw if leak_pat.search(r["corrected_trace"])]
w(f"Rewrites that leak audit vocabulary: **{len(leaks)}**. Prompt v2 should give 0. "
  f"Any hit has to be dropped by hand before the rewrite goes into a corpus.")
w()
if leaks[:10]:
    w("First offenders: " + ", ".join(f"`{x}`" for x in leaks[:10]))
    w()

# ------------------------------------------------------------------ cost
w("## 8. Throughput and tokens")
w()
lat = [r["latency_s"] for r in ok if r.get("latency_s")]
pt = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
ot = [r["output_tokens"] or 0 for r in ok if r.get("prompt_tokens")]
th = [r.get("thought_tokens") or 0 for r in ok if r.get("prompt_tokens")]
att = [r.get("attempts") or 1 for r in ok]
table(["", "mean", "median", "total"],
      [["latency per item (s)", f"{st.mean(lat):.1f}", f"{st.median(lat):.1f}", ""],
       ["input tokens", f"{st.mean(pt):.0f}", f"{st.median(pt):.0f}", f"{sum(pt):,}"],
       ["output tokens", f"{st.mean(ot):.0f}", f"{st.median(ot):.0f}", f"{sum(ot):,}"],
       ["thinking tokens", f"{st.mean(th):.0f}", f"{st.median(th):.0f}", f"{sum(th):,}"],
       ["billed output", f"{st.mean(ot)+st.mean(th):.0f}", "",
        f"{sum(ot)+sum(th):,}"],
       ["attempts per item", f"{st.mean(att):.2f}", "", f"{sum(att):,}"]])
w(f"Items that needed more than one attempt: {sum(1 for a in att if a > 1)} of {N}.")
w()

# ------------------------------------------------------------------ caveats
w("## 9. Limits to carry into any text written from this")
w()
w("1. **Count errors are a lower bound.** Judge recall 71 %, precision 100 %, and the "
  "judge's own lead count on `transistor1` is right only 55.7 % of the time. Publish "
  "the deterministic number from section 6, never a judge count rate.")
w("2. **A single item verdict flips about 7.5 % of the time** between identical runs at "
  "temperature 0. Aggregate rates over 10,236 items are stable. One quoted trace is not.")
w("3. **The `correct` category is dead.** It fired 0 times in 200 calibration items. "
  "Treat the scale as three-valued when buckets are built.")
w("4. Anomalous items with no C1 mask cannot be repaired by a rewrite. They are a "
  "routing decision, not a text problem.")
w()

txt = "\n".join(L)
OUTMD.write_text(txt)
print(txt)
print(f"\n[written] {OUTMD}", file=sys.stderr)
