#!/usr/bin/env python3
"""Step 2 of the trace grounding audit: deterministic format check. No model involved.

Every regex here is copied from the shipped GRPO reward function
  scripts/02_grpo/stage_rl/reward.py
    line 257  pattern_no   (the two negative look-aheads that reject <location>/<type>
                            on a normal trace)
    line 258  pattern_yes
    line 267/271  re.fullmatch(..., re.DOTALL)          -> consistency_reward
    line 287-296  <answer> parse and the gt == "no" branch -> accuracy_reward

Per trace we record: which tags are present, whether the tag order is correct,
whether a normal trace violates the negative look-aheads, the parsed answer, and
whether that answer matches the gold label from the dataset.

Reads  corpus.jsonl
Writes format_check.jsonl  (one object per line)

Usage: python3 format_check.py
"""
import json
import re
from collections import Counter

import os

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.environ.get("AUDIT_CORPUS", os.path.join(HERE, "corpus.jsonl"))
OUT = os.path.join(HERE, "format_check.jsonl")

# ---- verbatim from reward.py:257-258 -----------------------------------------
PATTERN_NO = r"^(?!.*<location>)(?!.*<type>).*<think>.*?</think>\s*<answer>.*?</answer>.*$"
PATTERN_YES = r".*<think>.*?</think>\s*<location>.*?</location>\s*<type>.*?</type>\s*<answer>.*?</answer>.*"
# ---- verbatim from reward.py:287 / 293 ---------------------------------------
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>")

TAGS = ("think", "location", "type", "answer")
ORDER_YES = ["think", "location", "type", "answer"]
ORDER_NO = ["think", "answer"]


def tag_spans(text):
    """First open-tag offset per tag, and whether the tag is properly closed."""
    present, opens = {}, {}
    for t in TAGS:
        o = re.search(rf"<{t}>", text)
        c = re.search(rf"</{t}>", text)
        present[t] = bool(o) and bool(c) and c.start() > o.start()
        opens[t] = o.start() if o else None
    return present, opens


def check(row):
    text = row["trace"]
    gold = row["gold_label"]
    present, opens = tag_spans(text)

    # counts, to catch duplicated tags
    counts = {t: len(re.findall(rf"<{t}>", text)) for t in TAGS}

    expected = ORDER_YES if gold == "yes" else ORDER_NO
    have_expected = all(present[t] for t in expected)
    seq = [t for t in expected if opens[t] is not None]
    order_ok = have_expected and seq == expected and \
        all(opens[seq[i]] < opens[seq[i + 1]] for i in range(len(seq) - 1))

    extra = [t for t in TAGS if present[t] and t not in expected]

    # the negative look-ahead violation, only meaningful for gold == "no"
    lookahead_violation = None
    if gold == "no":
        lookahead_violation = bool(re.search(r"<location>", text) or re.search(r"<type>", text))

    # the shipped consistency_reward, verbatim behaviour
    if gold == "yes":
        consistency = 1.0 if re.fullmatch(PATTERN_YES, text, re.DOTALL) else 0.0
    elif gold == "no":
        consistency = 1.0 if re.fullmatch(PATTERN_NO, text, re.DOTALL) else 0.0
    else:
        consistency = 0.0

    m = RE_ANSWER.search(text)
    raw_answer = m.group(1).strip() if m else None
    parsed = raw_answer.lower() if raw_answer is not None else None
    if parsed in ("yes", "no"):
        parse_status = "ok"
    elif parsed is None:
        parse_status = "no_answer_tag"
    else:
        parse_status = "unparsable_value"
    answer_matches_gold = (parse_status == "ok" and parsed == gold)

    think = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    think_len = len(think.group(1).strip()) if think else 0

    return {
        "id": row["id"],
        "split": row["split"],
        "product": row["product"],
        "gold_label": gold,
        "gold_type": row["gold_type"],
        "tags_present": present,
        "tag_counts": counts,
        "extra_tags": extra,
        "tag_order_ok": order_ok,
        "normal_lookahead_violation": lookahead_violation,
        "consistency_reward": consistency,
        "answer_raw": raw_answer,
        "answer_parsed": parsed,
        "answer_parse_status": parse_status,
        "answer_matches_gold": answer_matches_gold,
        "think_chars": think_len,
        "think_too_short": think_len < 20,   # reward.py:251 drops these completions
        "trace_words": len(text.split()),
    }


def main():
    rows = [json.loads(l) for l in open(CORPUS)]
    res = [check(r) for r in rows]
    with open(OUT, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")

    n = len(res)
    def pc(k):
        return f"{k} ({100.0*k/n:5.2f} %)"

    print(f"wrote {OUT}")
    print(f"traces checked                    : {n}")
    print(f"  sft_6k / grpo_4k                : "
          f"{sum(r['split']=='sft_6k' for r in res)} / {sum(r['split']=='grpo_4k' for r in res)}")
    print()
    print("TAGS")
    for t in TAGS:
        print(f"  <{t}> present and closed         : {sum(r['tags_present'][t] for r in res)}")
    print(f"  duplicated tag somewhere        : "
          f"{sum(any(v>1 for v in r['tag_counts'].values()) for r in res)}")
    print(f"  extra tag for its class         : {sum(bool(r['extra_tags']) for r in res)}")
    print(f"  tag order correct               : {pc(sum(r['tag_order_ok'] for r in res))}")
    print()
    print("NORMAL TRACES (gold = no)")
    no = [r for r in res if r["gold_label"] == "no"]
    print(f"  count                           : {len(no)}")
    print(f"  negative look-ahead violation   : {sum(r['normal_lookahead_violation'] for r in no)}")
    print()
    print("ANSWER PARSE")
    print("  status                          :", dict(Counter(r["answer_parse_status"] for r in res)))
    print(f"  answer matches gold label       : {pc(sum(r['answer_matches_gold'] for r in res))}")
    bad = [r for r in res if not r["answer_matches_gold"]]
    print(f"  mismatches                      : {len(bad)}")
    for r in bad[:20]:
        print(f"    {r['id']}  gold={r['gold_label']} parsed={r['answer_parsed']!r} "
              f"status={r['answer_parse_status']}")
    print()
    print("SHIPPED consistency_reward (reward.py:256-275)")
    print(f"  scores 1.0                      : {pc(sum(r['consistency_reward']==1.0 for r in res))}")
    for sp in ("sft_6k", "grpo_4k"):
        s = [r for r in res if r["split"] == sp]
        print(f"    {sp:8s}                    : {sum(r['consistency_reward']==1.0 for r in s)} / {len(s)}")
    print(f"  <think> shorter than 20 chars   : {sum(r['think_too_short'] for r in res)}")
    print()
    print("FULLY CLEAN (order ok, no extra tags, answer parses and matches gold)")
    clean = [r for r in res if r["tag_order_ok"] and not r["extra_tags"] and r["answer_matches_gold"]]
    print(f"  {pc(len(clean))}")


if __name__ == "__main__":
    main()
