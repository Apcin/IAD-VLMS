"""
Post-generation FILTER for Real-IAD Variety traces.

Validates every produced trace_<id>.json against the required structure:

  ANOMALY (ground-truth NG):
    <think>...</think><location>L</location><type>T</type><answer>Yes</answer>
    - exactly one of each tag, in this order, nothing after </answer>
    - every comma-separated L is one of the 9 grid cells
    - answer == Yes

  NORMAL (ground-truth OK):
    <think>...</think><answer>No</answer>
    - no <location>/<type> tags
    - answer == No

Usage:
  python validate_variety_traces.py --report          # just count valid/invalid
  python validate_variety_traces.py --purge           # delete invalid trace files
                                                       # (so the generator regenerates them)

Ground truth (image_id -> is_anomaly) is taken from the generator's own loaders
so the id derivation matches exactly. Run from reasoning_traces_gen/.
"""
import argparse
import re
from pathlib import Path

import generate_variety_traces as G

GRID = {
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
}

ANOM_RE = re.compile(
    r"^<think>(?P<th>.*?)</think>\s*"
    r"<location>(?P<loc>.*?)</location>\s*"
    r"<type>(?P<typ>.*?)</type>\s*"
    r"<answer>\s*(?P<ans>yes)\s*</answer>\s*$",
    re.S | re.I,
)
NORM_RE = re.compile(
    r"^<think>(?P<th>.*?)</think>\s*"
    r"<answer>\s*(?P<ans>no)\s*</answer>\s*$",
    re.S | re.I,
)


def validate(reasoning: str, is_anom: bool):
    """Return (ok: bool, reason: str)."""
    r = (reasoning or "").strip()
    if is_anom:
        m = ANOM_RE.match(r)
        if not m:
            # diagnose
            if "<location>" not in r or "<type>" not in r:
                return False, "anom-missing-tag"
            return False, "anom-bad-structure"
        if not m.group("th").strip():
            return False, "empty-think"
        for part in m.group("loc").split(","):
            p = part.strip()
            if p not in GRID:
                return False, f"bad-location:{p}"
        return True, "ok"
    else:
        m = NORM_RE.match(r)
        if not m:
            if "<location>" in r or "<type>" in r:
                return False, "normal-has-extra-tag"
            return False, "normal-bad-structure"
        if not m.group("th").strip():
            return False, "empty-think"
        return True, "ok"


def build_gt():
    """image_id -> is_anomaly, for both grpo and sft pools, split by output dir."""
    grpo_ng, grpo_ok = G.load_grpo_questions()
    sft_ng, sft_ok = G.load_sft_questions()
    gt = {
        "grpo": {q["image_id"]: True for q in grpo_ng},
        "sft":  {q["image_id"]: True for q in sft_ng},
    }
    for q in grpo_ok:
        gt["grpo"][q["image_id"]] = False
    for q in sft_ok:
        gt["sft"][q["image_id"]] = False
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", action="store_true", help="delete invalid trace files so they get regenerated")
    ap.add_argument("--report", action="store_true", help="report only (default if --purge not given)")
    args = ap.parse_args()

    import json
    import collections
    import logging
    logging.basicConfig(level=logging.WARNING)

    gt = build_gt()
    dirs = {"grpo": G.GRPO_DIR, "sft": G.SFT_DIR}

    total = ok = invalid = unknown = 0
    reasons = collections.Counter()
    purged = 0
    for pool, d in dirs.items():
        for f in Path(d).glob("trace_*.json"):
            total += 1
            iid = f.stem[len("trace_"):]
            if iid not in gt[pool]:
                unknown += 1
                continue
            try:
                reasoning = json.loads(f.read_text()).get("reasoning", "")
            except Exception:
                reasoning = ""
            good, reason = validate(reasoning, gt[pool][iid])
            if good:
                ok += 1
            else:
                invalid += 1
                reasons[reason.split(":")[0]] += 1
                if args.purge:
                    f.unlink()
                    purged += 1

    print(f"[validate] total={total} ok={ok} invalid={invalid} unknown_id={unknown}")
    if reasons:
        print("[validate] invalid reasons:", dict(reasons))
    if args.purge:
        print(f"[validate] purged {purged} invalid trace files (will be regenerated)")

    # exit code 0 if nothing invalid, else 2 (so a driver loop can detect)
    raise SystemExit(0 if invalid == 0 else 2)


if __name__ == "__main__":
    main()
