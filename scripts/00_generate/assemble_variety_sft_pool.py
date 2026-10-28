"""
Assemble the Variety SFT rollout pool from the Gemini-generated traces.

The STaR rollout (phase0_rollout_kscoring.py) scores each model rollout against
a ground-truth trace. For Variety, that GT trace is the Gemini output produced by
generate_variety_traces.py (one trace_<image_id>.json per image). This script
merges those traces with the SFT split into a PER-ITEM pool in the schema
phase0 expects (so normalize_per_item gets product + gt_trace right; the sharegpt
path would lose the product because Variety paths have no 'images/' segment).

Output: variety_c1_compiled/sft_pool_with_traces.json
  [{image_path, product, answer=<gemini trace>, is_anomaly, gt_label}, ...]

Run from reasoning_traces_gen/ (needs images on disk + generated traces).
"""
import json
from pathlib import Path

import generate_variety_traces as G

OUT = Path("variety_c1_compiled/sft_pool_with_traces.json")


def main():
    ng, ok = G.load_sft_questions()          # each: image_id, image_path(Path), category, ...
    items = [(q, True) for q in ng] + [(q, False) for q in ok]

    out, missing, empty = [], 0, 0
    for q, is_anom in items:
        tf = G.SFT_DIR / f"trace_{q['image_id']}.json"
        if not tf.exists():
            missing += 1
            continue
        try:
            reasoning = json.loads(tf.read_text()).get("reasoning", "").strip()
        except Exception:
            reasoning = ""
        if not reasoning:
            empty += 1
            continue
        out.append({
            "image_path": str(q["image_path"]),
            "product": q["category"],
            "answer": reasoning,                 # <- gt_trace for rollout scoring
            "is_anomaly": is_anom,
            "gt_label": "yes" if is_anom else "no",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    n_anom = sum(1 for x in out if x["is_anomaly"])
    print(f"[assemble] wrote {OUT}  ({len(out)} items: {n_anom} NG / {len(out)-n_anom} OK; "
          f"missing_trace={missing}, empty_trace={empty})")


if __name__ == "__main__":
    main()
