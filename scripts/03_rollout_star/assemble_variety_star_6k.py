#!/usr/bin/env python3
"""
Assemble the FINAL Variety STaR SFT corpus: EXACTLY 6,000 traces, stratified by
product AND normal/anomaly (3,000 NG + 3,000 OK, spread as evenly as possible
across the 160 products via round-robin).

Source = the post-STaR pool (each rolled-out item has a final trace):
  good_traces_faithful.json   (kept, faithfulness-passed; sharegpt format)
  gemini_corrected.jsonl      (0/k-fail items, Gemini-corrected; {image_path, corrected_trace})
  gemini_rewritten.jsonl      (marginal/judge-failed, Gemini-rewritten; same schema)

Output = LlamaFactory sharegpt JSON ({messages, images}), drop-in for SFT.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42


def train_prompt(product: str) -> str:
    return (f"Analyze the provided image of the {product}. "
            "Determine if there are any anomalies present. "
            "If an anomaly is detected, specify its type and location, "
            "and provide a detailed reasoning for your conclusion.")


def product_of(path: str) -> str:
    if "/realiad-variety/" in path:
        return path.split("/realiad-variety/")[1].split("/")[0]
    if "/images/" in path:
        return path.split("/images/")[1].split("/")[0]
    return "unknown"


def is_ng(path: str) -> bool:
    return ("/NG/" in path) or ("_NG_" in path)


def sharegpt_entry(image_path: str, trace: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>\n" + train_prompt(product_of(image_path))},
            {"role": "assistant", "content": trace},
        ],
        "images": [image_path],
    }


def load_kept(d: Path):
    p = d / "good_traces_faithful.json"
    if not p.exists():
        p = d / "good_traces.json"
    return json.load(open(p)) if p.exists() else []


def load_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("corrected_trace"):
            out.append(sharegpt_entry(r["image_path"], r["corrected_trace"]))
    return out


def round_robin_pick(by_prod: dict, n: int, rng) -> list:
    """Pick n items spread as evenly as possible across products."""
    prods = sorted(by_prod)
    for p in prods:
        rng.shuffle(by_prod[p])
    picked, idx = [], {p: 0 for p in prods}
    while len(picked) < n:
        progressed = False
        for p in prods:
            if len(picked) >= n:
                break
            if idx[p] < len(by_prod[p]):
                picked.append(by_prod[p][idx[p]])
                idx[p] += 1
                progressed = True
        if not progressed:
            break  # exhausted before reaching n
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=6000)
    args = ap.parse_args()
    rng = random.Random(SEED)
    d = Path(args.phase0_dir)

    kept = load_kept(d)
    corrected = load_jsonl(d / "gemini_corrected.jsonl")
    rewritten = load_jsonl(d / "gemini_rewritten.jsonl")
    print(f"[pool] kept={len(kept)} corrected={len(corrected)} rewritten={len(rewritten)}")

    # union, dedup by image path; corrected/rewritten override a kept rollout
    by_path = {}
    for e in kept:
        by_path.setdefault(e["images"][0], e)
    for e in corrected:
        by_path[e["images"][0]] = e
    for e in rewritten:
        by_path[e["images"][0]] = e
    pool = list(by_path.values())

    ng, ok = defaultdict(list), defaultdict(list)
    for e in pool:
        ip = e["images"][0]
        (ng if is_ng(ip) else ok)[product_of(ip)].append(e)

    half = args.target // 2
    sel = round_robin_pick(ng, half, rng) + round_robin_pick(ok, half, rng)
    rng.shuffle(sel)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(sel, open(args.out, "w"), indent=2, ensure_ascii=False)

    n_ng = sum(1 for e in sel if is_ng(e["images"][0]))
    perprod = Counter(product_of(e["images"][0]) for e in sel)
    lo = min(perprod.values()) if perprod else 0
    hi = max(perprod.values()) if perprod else 0
    print(f"[6k] wrote {args.out}: {len(sel)} items ({n_ng} NG / {len(sel)-n_ng} OK), "
          f"{len(perprod)} products, per-product min/max = {lo}/{hi}")
    if len(sel) < args.target:
        print(f"[6k] WARNING: only {len(sel)} < target {args.target} (pool exhausted in some buckets)")


if __name__ == "__main__":
    main()
