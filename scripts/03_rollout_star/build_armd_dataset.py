#!/usr/bin/env python3
"""Build the Arm D SFT corpus out of the star_rationalise.py rationalisation pass.

Arm D is the STaR arm. The trace for a failed item is written by the policy
itself, with only the gold yes/no pasted into the prompt as a hint, and the hint
stripped again before the trace is stored (STaR, arXiv 2203.14465 section 3.2).

Two variants are written, both 6,000 items and 3,000 NG / 3,000 OK, i.e. the same
size and the same class balance as Arm C.

  swap  (default, the file the SFT trains on)
        Arm C's own 6,000 images. Every item whose trace Arm C took from the
        Gemini `correct` pass is replaced by the policy's own rationalisation.
        An item with no passing rationalisation is dropped and replaced by a kept
        trace of the same class from the pool Arm C itself was drawn from, so the
        corpus stays 6,000 and 50/50. Kept and rewritten items are byte identical
        to Arm C. This is the one-variable contrast against the headline run:
        who writes the trace for the items the model got wrong.

  pure  (built for reference, not trained by default)
        No teacher text anywhere. Pool = every kept trace plus every passing
        rationalisation, run through build_llava_arms.balance with cap 6000,
        which is the function that produced the Arm C 50/50 cap.

Nothing here re-implements the balancing or the NG test. Both come out of the
existing builders (see _load_funcs). Nothing writes into a checkpoint directory.

Usage:
  python3 build_armd_dataset.py --star_jsonl <file> [--variant swap|pure|both]
                                [--rule production|answer_format] [--register]
"""

import argparse
import ast
import json
import os
import random
import sys
import time
from pathlib import Path

# The phase-0 pools and the Arm C corpus are NOT in this repository: their records carry
# absolute Real-IAD image paths and we do not redistribute the pixels (README section 3).
# WORK_DIR is the workspace that holds Training/ and LlamaFactory/; it defaults to the
# parent of this repository. On another machine:  export WORK_DIR=/path/to/workspace
HERE = Path(__file__).resolve().parent
R = os.environ.get("WORK_DIR") or str(HERE.parents[1].parent)

POOLS = [
    f"{R}/Training/phase0_full_10k_20260529_015821",
    f"{R}/Training/phase0_heldout_20260601",
]
ARMC_PATH = f"{R}/Training/datasets_sft_iter2/sft_iter2_train.json"
OUT_DIR = f"{R}/Training/datasets_sft_armd"
DATASET_INFO = f"{R}/LlamaFactory/data/dataset_info.json"

# Arm C's user turn, verified byte identical for all 6,000 of its items:
# the product directory name with underscores turned into spaces.
USER_TPL = ("<image>\nAnalyze the provided image of the {p}. Determine if there "
            "are any anomalies present. If an anomaly is detected, specify its "
            "type and location, and provide a detailed reasoning for your "
            "conclusion.")


def _load_funcs(path):
    """Take the pure helpers out of an existing builder without running it.

    build_abc_datasets.py and build_llava_arms.py both do their work at module
    level, and build_llava_arms.py reads sys.argv on import, so a plain import
    would rebuild datasets or crash. Keeping only the imports and the function
    definitions gives the real functions from the real file, so the balancing
    rule and the NG test stay defined in exactly one place.
    """
    src = open(path).read()
    tree = ast.parse(src, path)
    tree.body = [n for n in tree.body
                 if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns = {"__name__": "_helpers"}
    exec(compile(tree, path, "exec"), ns)
    return ns


# Both builders sit next to this file in scripts/03_rollout_star/.
ABC = _load_funcs(str(HERE / "build_abc_datasets.py"))
LLA = _load_funcs(str(HERE / "build_llava_arms.py"))

is_ng = ABC["is_ng"]                                  # Arm A/B NG test
product_of = ABC["product_of"]                        # Arm A per-product key
load_corrected_paths = ABC["load_jsonl_corrected_paths"]
load_rewritten_paths = ABC["load_jsonl_rewritten_paths"]
balance = LLA["balance"]                              # the Arm C 50/50 cap


def user_turn(image_path):
    return USER_TPL.format(p=product_of(image_path).replace("_", " "))


def record(image_path, trace):
    return {"messages": [{"role": "user", "content": user_turn(image_path)},
                         {"role": "assistant", "content": trace}],
            "images": [image_path]}


def counts(items):
    ng = sum(1 for x in items if is_ng(x["images"][0]))
    return len(items), ng, len(items) - ng


def load_star(path, rule):
    """image_path -> the kept rationalisation, under the chosen keep rule.

    production     format == 1.0 and acc >= 1.5 (NG) / >= 1.0 (OK). The same bar
                   phase0_bucket.py used to put these items in the failure set,
                   and the same bar every kept trace in Arms A/B/C had to clear.
    answer_format  verdict correct and format correct. Looser. On the dry run about
                   half the NG traces kept this way named the wrong defect type and
                   the wrong place, so `production` is the default.
    """
    field = "kept_trace_production" if rule == "production" else "kept_trace"
    out, n_lines = {}, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            r = json.loads(line)
            t = r.get(field)
            if t:
                out[r["image_path"]] = t
    return out, n_lines


def load_kept_pool():
    kept = []
    for d in POOLS:
        p = f"{d}/good_traces.json"
        if os.path.exists(p):
            kept.extend(json.load(open(p)))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star_jsonl", required=True,
                    help="output of star_rationalise.py over the full failure pool")
    ap.add_argument("--variant", default="both", choices=["swap", "pure", "both"])
    ap.add_argument("--rule", default="production",
                    choices=["production", "answer_format"])
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--register", action="store_true",
                    help="add iad_sft_armd_swap / iad_sft_armd_pure to dataset_info.json")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- inputs -----------------------------------------------------------
    armc = json.load(open(ARMC_PATH))
    bad = [x for x in armc if x["messages"][0]["content"] != user_turn(x["images"][0])]
    if bad:
        sys.exit(f"[fail] user_turn() does not reproduce Arm C on {len(bad)} items. "
                 "Refusing to build a corpus whose prompt differs from the headline run.")
    print(f"[armc] {len(armc)} items, user turn reproduced byte identical on all of them")

    corrected_paths, rewritten_paths = set(), set()
    for d in POOLS:
        corrected_paths |= load_corrected_paths(f"{d}/gemini_corrected.jsonl")
        rewritten_paths |= load_rewritten_paths(f"{d}/gemini_rewritten.jsonl")

    c_corrected = [x for x in armc if x["images"][0] in corrected_paths]
    c_rewritten = [x for x in armc if x["images"][0] in rewritten_paths]
    c_kept = [x for x in armc if x["images"][0] not in corrected_paths
              and x["images"][0] not in rewritten_paths]
    print(f"[armc] kept={len(c_kept)} corrected={len(c_corrected)} "
          f"rewritten={len(c_rewritten)}")

    star, n_lines = load_star(args.star_jsonl, args.rule)
    print(f"[star] {n_lines} rationalised items on disk, "
          f"{len(star)} pass the '{args.rule}' rule "
          f"({100.0 * len(star) / max(n_lines, 1):.1f}%)")
    if n_lines < 1900:
        print(f"[warn] only {n_lines} lines in {args.star_jsonl}. The full failure "
              "pool is 1,940 rationalisable items. Is the pass complete?")

    kept_pool = load_kept_pool()
    print(f"[pool] kept traces available: {len(kept_pool)}")

    summary = {"star_jsonl": args.star_jsonl, "rule": args.rule, "seed": args.seed,
               "armc": {"n": len(armc), "kept": len(c_kept),
                        "corrected": len(c_corrected), "rewritten": len(c_rewritten)},
               "star": {"lines": n_lines, "passing": len(star)}}

    # ---- variant: swap ----------------------------------------------------
    if args.variant in ("swap", "both"):
        random.seed(args.seed)   # each variant reproducible on its own
        used = {x["images"][0] for x in armc}
        out, drop_ng, drop_ok, n_swapped = [], 0, 0, 0
        for it in armc:
            p = it["images"][0]
            if p in corrected_paths:
                t = star.get(p)
                if t is None:
                    if is_ng(p):
                        drop_ng += 1
                    else:
                        drop_ok += 1
                    continue
                out.append(record(p, t))
                n_swapped += 1
            else:
                out.append(it)

        spare = [x for x in kept_pool if x["images"][0] not in used]
        spare_ng = [x for x in spare if is_ng(x["images"][0])]
        spare_ok = [x for x in spare if not is_ng(x["images"][0])]
        random.shuffle(spare_ng)
        random.shuffle(spare_ok)
        if len(spare_ng) < drop_ng or len(spare_ok) < drop_ok:
            sys.exit(f"[fail] not enough spare kept traces to top up "
                     f"(need {drop_ng} NG / {drop_ok} OK, have "
                     f"{len(spare_ng)} / {len(spare_ok)})")
        # Re-render the top-ups through record(), because good_traces.json writes
        # the product with underscores and Arm C writes it with spaces. Every user
        # turn in the corpus must follow the Arm C convention.
        for x in spare_ng[:drop_ng] + spare_ok[:drop_ok]:
            out.append(record(x["images"][0], x["messages"][1]["content"]))
        random.shuffle(out)

        n, ng, ok = counts(out)
        if n != len(armc) or ng != ok:
            sys.exit(f"[fail] swap variant is {n} items, {ng} NG / {ok} OK. "
                     f"Expected {len(armc)} and a 50/50 split.")
        path = os.path.join(args.out_dir, "sft_D_star_swap.json")
        json.dump(out, open(path, "w"), indent=2)
        print(f"[swap] {n} items, {ng} NG / {ok} OK -> {path}")
        print(f"[swap] {n_swapped} self-rationalised, "
              f"{drop_ng + drop_ok} corrected items had no passing rationalisation "
              f"and were replaced by kept traces ({drop_ng} NG / {drop_ok} OK)")
        summary["swap"] = {"path": path, "n": n, "ng": ng, "ok": ok,
                           "self_rationalised": n_swapped,
                           "topped_up_ng": drop_ng, "topped_up_ok": drop_ok,
                           "kept": len(c_kept), "rewritten": len(c_rewritten)}

    # ---- variant: pure ----------------------------------------------------
    if args.variant in ("pure", "both"):
        random.seed(args.seed)
        pool = {}
        for x in kept_pool:
            pool[x["images"][0]] = x["messages"][1]["content"]
        for p, t in star.items():
            pool[p] = t
        items = sorted(pool.items())
        bal, n_ng, n_ok = balance(items, cap=len(armc))
        out = [record(p, t) for p, t in bal]
        n, ng, ok = counts(out)
        if n != len(armc) or ng != ok:
            sys.exit(f"[fail] pure variant is {n} items, {ng} NG / {ok} OK.")
        n_rat = sum(1 for x in out if x["images"][0] in star)
        path = os.path.join(args.out_dir, "sft_D_star_pure.json")
        json.dump(out, open(path, "w"), indent=2)
        print(f"[pure] {n} items, {ng} NG / {ok} OK -> {path}")
        print(f"[pure] {n_rat} self-rationalised, {n - n_rat} kept, "
              f"0 teacher-written (drawn from a pool of {len(items)}: "
              f"{n_ng} NG / {n_ok} OK)")
        summary["pure"] = {"path": path, "n": n, "ng": ng, "ok": ok,
                           "self_rationalised": n_rat, "kept": n - n_rat,
                           "pool": len(items)}

    # ---- register ---------------------------------------------------------
    if args.register:
        raw = open(DATASET_INFO).read()
        # 138 other datasets live in this file. Keep its formatting so the change
        # is two added keys and nothing else, and keep a backup either way.
        indent = len(raw.split("\n")[1]) - len(raw.split("\n")[1].lstrip()) or 2
        bak = DATASET_INFO + ".bak." + time.strftime("%Y%m%d_%H%M%S")
        open(bak, "w").write(raw)
        print(f"[register] backup of dataset_info.json -> {bak}")
        info = json.loads(raw)
        tags = {"role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant"}
        for key, fname in (("iad_sft_armd_swap", "sft_D_star_swap.json"),
                           ("iad_sft_armd_pure", "sft_D_star_pure.json")):
            full = os.path.join(args.out_dir, fname)
            if os.path.exists(full):
                info[key] = {"file_name": full, "formatting": "sharegpt",
                             "columns": {"messages": "messages", "images": "images"},
                             "tags": tags}
                print(f"[register] {key} -> {full}")
        with open(DATASET_INFO, "w") as f:
            f.write(json.dumps(info, indent=indent))
            if raw.endswith("\n"):
                f.write("\n")

    spath = os.path.join(args.out_dir, "armd_composition.json")
    json.dump(summary, open(spath, "w"), indent=2)
    print(f"[out] composition summary -> {spath}")


if __name__ == "__main__":
    main()
