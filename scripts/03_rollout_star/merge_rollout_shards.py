#!/usr/bin/env python
"""Merge per-shard rollout outputs into one balanced SFT-Iter2 dataset."""
import argparse
import glob
import json
import os
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", required=True,
                   help="Dir containing iter2_sft_shard*.json files")
    p.add_argument("--output_path", required=True,
                   help="Final merged + balanced SFT-Iter2 JSON")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    yes, no, all_stats = [], [], {}
    shard_files = sorted(glob.glob(f"{args.shard_dir}/iter2_sft_shard*.json"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files in {args.shard_dir}")

    for sf in shard_files:
        with open(sf) as f:
            d = json.load(f)
        yes.extend(d["yes"])
        no.extend(d["no"])
        for k, v in d.get("stats", {}).items():
            all_stats[k] = all_stats.get(k, 0) + v
        print(f"  {os.path.basename(sf)}: +{len(d['yes'])} yes, +{len(d['no'])} no")

    print()
    print(f"TOTAL collected: {len(yes)} yes + {len(no)} no")
    n_min = min(len(yes), len(no))
    print(f"Balancing to {n_min} each")

    random.Random(args.seed).shuffle(yes)
    random.Random(args.seed).shuffle(no)
    yes = yes[:n_min]
    no = no[:n_min]

    final = yes + no
    random.Random(args.seed).shuffle(final)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(final, f, indent=2)

    print(f"Wrote {len(final)} balanced items -> {args.output_path}")
    print(f"Total source stats: {all_stats}")


if __name__ == "__main__":
    main()
