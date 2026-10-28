#!/usr/bin/env python3
"""Step 5 of the trace grounding audit: deterministic countable cross-check.

For products whose countable ground truth is established, this pulls every stated
count out of the trace with a regex, scores it against the truth, and then compares
that model-free verdict with what Gemini said in the grounding audit.

Only transistor1 has a verified truth today. Adding a product means adding one entry
to PRODUCT_TRUTH below. Nothing else changes.

  transistor1 : 9 leads per side, 18 in total. Established by three independent routes,
                one of them a model-free pixel counter whose modal result over all 1,000
                top-down transistor1 captures is (top 9, bottom 9).

Reads  corpus.jsonl and, when present, the audit output.
Writes countable_check.jsonl and prints the agreement table.

Usage
  python3 countable_check.py                                  # deterministic only
  python3 countable_check.py --audit audit_grounding.jsonl    # plus the agreement table
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------- ground truth
PRODUCT_TRUTH = {
    "transistor1": {
        "per_side": 9,
        "total": 18,
        "nouns": r"(?:pins?|leads?|terminals?|contacts?|legs?)",
        # package family names that carry an implied pin count
        "packages": r"(?:SOIC|SOP|SSOP|TSSOP|MSOP|DIP|PDIP|QFP|QFN)",
        "source": "model-free pixel count, modal (9, 9) over 1,000 C1 captures",
    },
    # template for the next product:
    # "usb": {"per_side": None, "total": 4, "nouns": r"(?:contacts?|pins?)",
    #         "packages": None, "source": "<how it was verified>"},
}

WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "twenty-four": 24, "thirty": 30}
NUM = r"(?:" + "|".join(sorted(WORD, key=len, reverse=True)) + r"|\d{1,3})"

PER_SIDE_CUE = re.compile(
    r"\b(?:on|per|along|at)\s+(?:each|either|every|one)\s+(?:side|row|edge)"
    r"|\bper\s+(?:side|row|edge)\b"
    r"|\b(?:each|either|every)\s+(?:side|row|edge)\b"
    r"|\bon\s+the\s+(?:top|bottom|left|right)\s*(?:side|row|edge)?\b", re.I)
TOTAL_CUE = re.compile(
    r"\b(?:in\s+total|total(?:ing|ling)?|altogether|combined|overall|"
    r"across\s+both\s+sides|on\s+both\s+sides|all\s+told)\b", re.I)

# A count that is the subject of a defect statement is not a claim about how many
# items the part has. "one pin is missing" must not be scored against the truth.
DEFECT_WORD = (r"missing|absent|bent|deformed|damaged|broken|cracked|chipped|corroded|"
               r"oxidi[sz]ed|discolo(?:u)?red|contaminated|scratched|lifted|shorted|"
               r"bridged|misaligned|skewed|obstructed|protruding|detached")
DEFECT_PRED = re.compile(
    rf"\b(?:is|are|appears?|seems?|looks?|has|have|being|was|were)\s+"
    rf"(?:\w+\s+){{0,3}}(?:{DEFECT_WORD})\b", re.I)
DEFECT_PRE = re.compile(rf"\b(?:{DEFECT_WORD}|without|except|apart\s+from)\s*$", re.I)


def to_int(tok):
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return WORD.get(tok)


def extract_counts(text, spec):
    """Every stated count, with the scope the sentence gives it."""
    out = []
    noun = spec["nouns"]

    # "ten pins", "18 metallic leads", "14-pin"
    pat = re.compile(rf"\b({NUM})[-\s]+(?:\w+[-\s]+){{0,2}}?({noun})\b", re.I)
    for m in pat.finditer(text):
        v = to_int(m.group(1))
        if v is None:
            continue
        out.append({"value": v, "surface": m.group(0), "pos": m.start(),
                    "kind": "noun_count"})

    # "totaling eighteen", "a total of 18", "18 in total"
    pat2 = re.compile(rf"\b(?:total(?:ing|ling)?|a\s+total\s+of)\s+({NUM})\b", re.I)
    for m in pat2.finditer(text):
        v = to_int(m.group(1))
        if v is not None:
            out.append({"value": v, "surface": m.group(0), "pos": m.start(),
                        "kind": "total_phrase", "scope": "total"})

    # "SOIC-18", "SOP 16", "DIP-14"
    if spec.get("packages"):
        pat3 = re.compile(rf"\b({spec['packages']})[-\s]?(\d{{1,3}})\b", re.I)
        for m in pat3.finditer(text):
            out.append({"value": int(m.group(2)), "surface": m.group(0), "pos": m.start(),
                        "kind": "package_name", "scope": "total"})

    # "(9 on each side)", "9 per side" with the noun left out
    pat4 = re.compile(rf"\b({NUM})\s+(?:on|per)\s+(?:each|either|every)\s+(?:side|row|edge)\b", re.I)
    for m in pat4.finditer(text):
        v = to_int(m.group(1))
        if v is not None:
            out.append({"value": v, "surface": m.group(0), "pos": m.start(),
                        "kind": "bare_per_side", "scope": "per_side"})

    # scope from the words around the match. The forward window stops at a bracket so
    # that "all 18 metallic leads (9 on each side)" does not scope the 18 as per-side.
    for c in out:
        if "scope" in c:
            continue
        end = c["pos"] + len(c["surface"])
        fwd = text[end: end + 60]
        cut = min([i for i in (fwd.find("("), fwd.find(")")) if i >= 0] or [len(fwd)])
        window = text[max(0, c["pos"] - 40): c["pos"]] + c["surface"] + fwd[:cut]
        if PER_SIDE_CUE.search(window):
            c["scope"] = "per_side"
        elif TOTAL_CUE.search(window):
            c["scope"] = "total"
        else:
            c["scope"] = "ambiguous"

    out.sort(key=lambda c: c["pos"])
    # de-duplicate identical surface at the same offset
    seen, dedup = set(), []
    for c in out:
        k = (c["pos"], c["surface"].lower())
        if k in seen:
            continue
        seen.add(k)
        dedup.append(c)
    return dedup


def score_count(c, spec, text):
    """correct / wrong / not_applicable. not_applicable means it is not a claim about
    how many items the part has."""
    end = c["pos"] + len(c["surface"])
    if c["kind"] != "package_name":
        if c["value"] < 2:
            c["skip_reason"] = "singular"
            return "not_applicable"
        if DEFECT_PRED.search(text[end: end + 45]) or DEFECT_PRE.search(text[max(0, c["pos"] - 25): c["pos"]]):
            c["skip_reason"] = "defect_subset"
            return "not_applicable"
    ps, tot = spec.get("per_side"), spec.get("total")
    if c["scope"] == "per_side":
        return "correct" if (ps is not None and c["value"] == ps) else "wrong"
    if c["scope"] == "total":
        return "correct" if (tot is not None and c["value"] == tot) else "wrong"
    allowed = {v for v in (ps, tot) if v is not None}
    return "correct" if c["value"] in allowed else "wrong"


def check_trace(row):
    spec = PRODUCT_TRUTH.get(row["product"])
    if not spec:
        return {"id": row["id"], "product": row["product"],
                "status": "no_ground_truth", "counts": [], "n_counts": 0,
                "n_wrong": 0, "det_verdict": None}
    counts = extract_counts(row["trace"], spec)
    for c in counts:
        c["score"] = score_count(c, spec, row["trace"])
    n_wrong = sum(c["score"] == "wrong" for c in counts)
    scored = [c for c in counts if c["score"] != "not_applicable"]
    if not scored:
        det = "no_count_stated"
    elif n_wrong:
        det = "count_wrong"
    else:
        det = "count_correct"
    return {"id": row["id"], "product": row["product"], "split": row["split"],
            "gold_label": row["gold_label"], "status": "checked",
            "truth": {"per_side": spec.get("per_side"), "total": spec.get("total"),
                      "source": spec["source"]},
            "counts": counts, "n_counts": len(scored), "n_wrong": n_wrong,
            "det_verdict": det}


# --------------------------------------------------------------- Gemini side
COUNT_WORDS = re.compile(
    r"\bcount(?:ed|ing|s)?\b|\bnumber\s+of\b|\bpins?\b|\bleads?\b|\bterminals?\b|"
    r"\bcontacts?\b|\bhow\s+many\b|\bSOIC|\bSOP\b|\bDIP\b|\bmiscount", re.I)


def gemini_count_signal(rec):
    """Did the judge raise a count problem, and did it declare itself unsure."""
    if not rec or rec.get("status") != "ok":
        return {"available": False}
    flagged, uncertain = False, False
    for it in rec.get("issues") or []:
        why = (it.get("why_wrong") or "")
        blob = " ".join([it.get("claim", ""), why, it.get("what_it_should_be", "")])
        if why.strip().upper().startswith("UNCERTAIN_COUNT"):
            uncertain = True
            continue
        if COUNT_WORDS.search(blob) and re.search(rf"\b{NUM}\b", blob, re.I):
            flagged = True
    return {"available": True, "category": rec.get("category"),
            "count_flagged": flagged, "uncertain_count": uncertain,
            "tags_ok": rec.get("tags_ok"), "confidence": rec.get("confidence")}


def agreement(det, gem):
    if not gem.get("available"):
        return "no_audit_record"
    if det == "no_count_stated":
        return "no_count_stated"
    if det == "count_wrong":
        return "agree_wrong" if gem["count_flagged"] else "gemini_missed"
    return "gemini_false_flag" if gem["count_flagged"] else "agree_correct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "corpus.jsonl"))
    ap.add_argument("--audit", default=None, help="audit_grounding.jsonl, optional")
    ap.add_argument("--out", default=str(HERE / "countable_check.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus)]
    audit = {}
    if args.audit and Path(args.audit).exists():
        for line in open(args.audit):
            try:
                r = json.loads(line)
            except Exception:
                continue
            audit[r["id"]] = r

    res = []
    for row in rows:
        if row["product"] not in PRODUCT_TRUTH:
            continue
        rec = check_trace(row)
        gem = gemini_count_signal(audit.get(row["id"]))
        rec["gemini"] = gem
        rec["agreement"] = agreement(rec["det_verdict"], gem)
        res.append(rec)

    with open(args.out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {args.out}")
    print(f"products with verified counts : {sorted(PRODUCT_TRUTH)}")
    print(f"traces covered                : {len(res)} of {len(rows)}")
    for prod in sorted(PRODUCT_TRUTH):
        sub = [r for r in res if r["product"] == prod]
        if not sub:
            continue
        dv = Counter(r["det_verdict"] for r in sub)
        stated = dv["count_wrong"] + dv["count_correct"]
        print()
        print(f"[{prod}]  truth per_side={PRODUCT_TRUTH[prod].get('per_side')} "
              f"total={PRODUCT_TRUTH[prod].get('total')}")
        print(f"  traces                      : {len(sub)}")
        print(f"  state a count               : {stated}")
        if stated:
            print(f"  count wrong                 : {dv['count_wrong']} "
                  f"({100.0*dv['count_wrong']/stated:.1f} % of the traces that state one)")
            print(f"  count correct               : {dv['count_correct']}")
        print(f"  state no count              : {dv['no_count_stated']}")
        for lab in ("no", "yes"):
            s2 = [r for r in sub if r["gold_label"] == lab]
            d2 = Counter(r["det_verdict"] for r in s2)
            st2 = d2["count_wrong"] + d2["count_correct"]
            print(f"    gold={lab:3s} n={len(s2):4d}  states {st2:4d}  wrong {d2['count_wrong']:4d}")
        wrong_vals = Counter(c["surface"].lower() for r in sub for c in r["counts"]
                             if c["score"] == "wrong")
        print("  most common wrong surfaces  :", wrong_vals.most_common(8))

    graded = [r for r in res if r["gemini"].get("available")]
    print()
    if not graded:
        print("no audit records joined, pass --audit to get the agreement table")
        return
    print(f"AGREEMENT with the Gemini grounding audit ({len(graded)} joined records)")
    for k, v in Counter(r["agreement"] for r in graded).most_common():
        print(f"  {k:20s} {v}")
    print("  Gemini category by deterministic verdict:")
    for dv in ("count_wrong", "count_correct", "no_count_stated"):
        sub = [r for r in graded if r["det_verdict"] == dv]
        if sub:
            print(f"    {dv:16s} -> {dict(Counter(r['gemini']['category'] for r in sub))}")
    unc = sum(r["gemini"].get("uncertain_count") for r in graded)
    print(f"  judge declared UNCERTAIN_COUNT : {unc}")


if __name__ == "__main__":
    main()
