#!/usr/bin/env python
"""Read every DS-MVTec eval JSON under outputs/ and print a BalAcc table."""
import glob
import json
import os
import re

ROOT = "/bulk/aacudad/reasoning_traces/outputs"

# (path glob → label group)
GROUPS = [
    ("sft_qwen25vl_7b_zeroshot_6k_frozen", "SFT-Iter1 (baseline)"),
    ("grpo_qwen25vl_7b_6k_frozen_ep3_full_run2", "GRPO-Iter1 = run2 (baseline)"),
    ("sft_qwen25vl_7b_iter2_frozen", "SFT-Iter2 (Plan A)"),
    ("grpo_qwen25vl_7b_iter2_full", "GRPO-Iter2 (Plan A)"),
    ("grpo_qwen25vl_7b_g2rpo_full", "G²RPO (Plan B)"),
]

PROMPT_LABEL = {
    "trainprompt": "default (make_train_prompt)",
    "trainprompt_default": "default (make_train_prompt)",
    "grpoprompt": "--grpo-eval",
    "bareprompt": "--bare-question",
}


def bal_acc(m):
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (tpr + tnr) / 2, tpr, tnr


def main():
    rows = []
    for group_dir, group_label in GROUPS:
        base = os.path.join(ROOT, group_dir)
        if not os.path.isdir(base):
            continue
        # collect from both top-level (e.g. for final model) and checkpoint-* dirs
        for json_path in sorted(glob.glob(f"{base}/**/eval_dsmvtec_full_*.json", recursive=True)):
            fname = os.path.basename(json_path)
            m = re.match(r"eval_dsmvtec_full_(.+)\.json", fname)
            if not m:
                continue
            suffix = m.group(1)
            prompt = PROMPT_LABEL.get(suffix, suffix)
            ckpt_dir = os.path.basename(os.path.dirname(json_path))
            try:
                with open(json_path) as f:
                    d = json.load(f)
                metrics = d.get("metrics", {})
                if not all(k in metrics for k in ("tp", "tn", "fp", "fn")):
                    continue
                ba, tpr, tnr = bal_acc(metrics)
                rows.append({
                    "group": group_label,
                    "ckpt": ckpt_dir,
                    "prompt": prompt,
                    "n": metrics["total"],
                    "raw_acc": metrics["accuracy"] * 100,
                    "bal_acc": ba * 100,
                    "ng_recall": tpr * 100,
                    "ok_recall": tnr * 100,
                })
            except Exception as e:
                print(f"err {json_path}: {e}")

    if not rows:
        print("No eval JSONs found.")
        return

    # Print table
    print(f"\n{'='*132}")
    print(f"{'Group':<32} {'Checkpoint':<22} {'Prompt':<28} {'N':>5} {'RawAcc':>8} {'BalAcc':>8} {'NG_Rec':>8} {'OK_Rec':>8}")
    print(f"{'='*132}")
    current_group = None
    for r in rows:
        if r["group"] != current_group:
            print(f"{'-'*132}")
            current_group = r["group"]
        print(f"{r['group']:<32} {r['ckpt']:<22} {r['prompt']:<28} {r['n']:>5} "
              f"{r['raw_acc']:>7.2f}% {r['bal_acc']:>7.2f}% {r['ng_recall']:>7.2f}% {r['ok_recall']:>7.2f}%")
    print(f"{'='*132}\n")

    # Best per group
    print("BEST PER GROUP (by BalAcc):")
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for g, rs in by_group.items():
        best = max(rs, key=lambda x: x["bal_acc"])
        print(f"  {g}: {best['bal_acc']:.2f} BalAcc on {best['ckpt']} @ {best['prompt']}")
    print()

    # Headline comparison
    baseline_b = 76.56  # run2 ckpt-530 grpo-eval avg (we only have MVTec here so use MVTec)
    print("Headline DS-MVTec comparisons (against run2 ckpt-530 @ --grpo-eval = 82.73 BalAcc):")
    for r in rows:
        if "Iter2" in r["group"] and "grpo-eval" in r["prompt"]:
            delta = r["bal_acc"] - 82.73
            print(f"  {r['ckpt']} @ grpo-eval: BalAcc {r['bal_acc']:.2f}  (delta {delta:+.2f})")
        if "G²RPO" in r["group"] and "grpo-eval" in r["prompt"]:
            delta = r["bal_acc"] - 82.73
            print(f"  {r['ckpt']} @ grpo-eval: BalAcc {r['bal_acc']:.2f}  (delta {delta:+.2f})")


if __name__ == "__main__":
    main()
