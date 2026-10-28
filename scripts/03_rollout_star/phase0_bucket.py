#!/usr/bin/env python
"""Phase 0/1 bucketer — read raw k-rollouts JSONL, produce 4 outputs:

    good_traces.json       — LLaMA-Factory sharegpt; ready for SFT
    needs_rewrite.json     — items with ≥1 correct rollout but weak quality
                             (e.g., low type score for yes-items, or format issues)
    needs_correction.json  — items where 0/k rollouts pass; full GT for Gemini
    difficulty.json        — image_path → difficulty score in [0,1]

Bucket logic per item with k rollouts:
    num_correct = count of rollouts with format==1.0 AND acc >= threshold
                  (threshold = 1.5 for NG, 1.0 for OK; matches v2)
    difficulty  = (k - num_correct) / k

    if num_correct == k:    → GOOD (best=shortest correct)
    elif num_correct >= 1:  → GOOD (best=shortest correct)
                              ALSO maybe → REWRITE if quality is low even on best
    else (num_correct==0):  → CORRECTION (best_attempt=shortest of the failures)

Quality heuristic for "weak but correct" rewrite eligibility:
    for NG items: best correct trace has acc < 1.8 (i.e., type or location was weak)
    for OK items: rarely flagged for rewrite (binary answer)
"""

import argparse
import json
import os
import re
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", required=True,
                   help="Output of phase0_rollout_kscoring.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--threshold_yes", type=float, default=1.5,
                   help="Min acc reward for an NG rollout to count as 'correct'")
    p.add_argument("--threshold_no", type=float, default=1.0,
                   help="Min acc reward for an OK rollout to count as 'correct'")
    p.add_argument("--weak_threshold_yes", type=float, default=1.8,
                   help="If best NG correct rollout has acc below this, flag for rewrite")
    return p.parse_args()


def is_passing(rollout, gt_answer, thresh_yes, thresh_no):
    if rollout["format"] != 1.0:
        return False
    th = thresh_yes if gt_answer == "yes" else thresh_no
    return rollout["acc"] >= th


def to_sharegpt(image_path, user_prompt_template, response):
    """Output in LLaMA-Factory sharegpt format.

    NOTE: we save the user message in make_train_prompt format (matching SFT-Iter1).
    The user_prompt field on the input is what the rollout model SAW;
    in the SFT-data output we re-format it consistently.
    """
    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{user_prompt_template}"},
            {"role": "assistant", "content": response},
        ],
        "images": [image_path],
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    good, needs_rewrite, needs_correction, judge_input = [], [], [], []
    difficulty = {}
    stats = defaultdict(int)

    with open(args.input_jsonl) as f:
        for line in f:
            r = json.loads(line)
            gt = r["gt_answer"]
            k = len(r["rollouts"])
            passing = [ro for ro in r["rollouts"] if is_passing(ro, gt, args.threshold_yes, args.threshold_no)]
            num_correct = len(passing)
            difficulty[r["image_path"]] = (k - num_correct) / k
            stats[f"k{k}_correct{num_correct}_{gt}"] += 1

            user_template = r["user_prompt"]  # already make_train_prompt unless --use_grpo_prompt was set

            # Extract GT type/location from the gt_trace (for downstream Gemini work)
            import re
            gt_type_m = re.search(r"<type>(.*?)</type>", r["gt_trace"], re.DOTALL)
            gt_loc_m = re.search(r"<location>(.*?)</location>", r["gt_trace"], re.DOTALL)
            gt_type = gt_type_m.group(1).strip() if gt_type_m else None
            gt_location = gt_loc_m.group(1).strip() if gt_loc_m else None

            if num_correct == 0:
                # Genuine failure → Gemini correction needed
                # Pick the shortest failed attempt (gives Gemini the cleanest starting point)
                failed_sorted = sorted(r["rollouts"], key=lambda x: x["len"])
                best_failed = failed_sorted[0]
                needs_correction.append({
                    "image_path": r["image_path"],
                    "product": r["product"],
                    "gt_trace": r["gt_trace"],
                    "gt_answer": gt,
                    "gt_type": gt_type,
                    "gt_location": gt_location,
                    "is_anomaly": r["is_anomaly"],
                    "source_pool": r["source_pool"],
                    "user_prompt_template": user_template,
                    "best_failed_attempt": best_failed["text"],
                    "best_failed_acc": best_failed["acc"],
                    "best_failed_format": best_failed["format"],
                    "all_attempts_acc": [ro["acc"] for ro in r["rollouts"]],
                    "difficulty": difficulty[r["image_path"]],
                })
                stats["bucket_correction"] += 1

                # ALSO send to judge so we get faithfulness rationale on failed traces too
                judge_input.append({
                    "image_path": r["image_path"],
                    "product": r["product"],
                    "gt_trace": r["gt_trace"],
                    "gt_answer": gt,
                    "gt_type": gt_type,
                    "gt_location": gt_location,
                    "is_anomaly": r["is_anomaly"],
                    "source_pool": r["source_pool"],
                    "user_prompt_template": user_template,
                    "chosen_trace": best_failed["text"],
                    "chosen_acc": best_failed["acc"],
                    "chosen_format": best_failed["format"],
                    "chosen_type_score": best_failed.get("type_score"),
                    "chosen_loc_score": best_failed.get("loc_score"),
                    "chosen_len": best_failed["len"],
                    "difficulty": difficulty[r["image_path"]],
                    "local_bucket": "correction",
                })
            else:
                # At least one correct → keep the shortest correct
                passing.sort(key=lambda x: x["len"])
                best = passing[0]
                good.append(to_sharegpt(r["image_path"], user_template, best["text"]))
                stats["bucket_good"] += 1

                # Build judge_input entry — Phase 1a uses this to ask Gemini for faithfulness
                judge_input.append({
                    "image_path": r["image_path"],
                    "product": r["product"],
                    "gt_trace": r["gt_trace"],
                    "gt_answer": gt,
                    "gt_type": gt_type,
                    "gt_location": gt_location,
                    "is_anomaly": r["is_anomaly"],
                    "source_pool": r["source_pool"],
                    "user_prompt_template": user_template,
                    "chosen_trace": best["text"],
                    "chosen_acc": best["acc"],
                    "chosen_format": best["format"],
                    "chosen_type_score": best.get("type_score"),
                    "chosen_loc_score": best.get("loc_score"),
                    "chosen_len": best["len"],
                    "difficulty": difficulty[r["image_path"]],
                    "local_bucket": "good",
                })

                # Also flag weak-but-correct items for local-reward-based rewrite
                # (separate from the Gemini-judge-based rewrite that Phase 1a will produce)
                if gt == "yes" and best["acc"] < args.weak_threshold_yes:
                    needs_rewrite.append({
                        "image_path": r["image_path"],
                        "product": r["product"],
                        "gt_trace": r["gt_trace"],
                        "gt_answer": gt,
                        "gt_type": gt_type,
                        "gt_location": gt_location,
                        "is_anomaly": r["is_anomaly"],
                        "source_pool": r["source_pool"],
                        "user_prompt_template": user_template,
                        "original_trace": best["text"],
                        "original_acc": best["acc"],
                        "flagged_by": "local_reward",  # vs "gemini_judge" (from Phase 1a output)
                        "difficulty": difficulty[r["image_path"]],
                    })
                    stats["bucket_rewrite_local"] += 1

    # Write outputs
    for name, data in [
        ("good_traces",       good),
        ("judge_input",       judge_input),
        ("needs_rewrite",     needs_rewrite),
        ("needs_correction",  needs_correction),
    ]:
        path = os.path.join(args.out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {name}: {len(data)} items → {path}")

    diff_path = os.path.join(args.out_dir, "difficulty.json")
    with open(diff_path, "w") as f:
        json.dump(difficulty, f, indent=2)
    print(f"  difficulty: {len(difficulty)} items → {diff_path}")

    print()
    print("Per-bucket stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Quick histogram of difficulty for the report
    bins = defaultdict(int)
    for d in difficulty.values():
        bins[round(d, 2)] += 1
    print()
    print("Difficulty histogram (rounded to 0.01):")
    for d in sorted(bins.keys()):
        bar = "█" * min(50, bins[d] // 2 + 1)
        print(f"  {d:.2f}: {bins[d]:>4}  {bar}")


if __name__ == "__main__":
    main()
