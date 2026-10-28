"""
Sample product-stratified, 50/50 normal/anomaly SFT + GRPO splits from
Real-IAD Variety (C1 angle only), with zero overlap between SFT and GRPO.

Mirrors sample_realiad_v4.py but adapted to the Variety schema:
  - reads the per-category split JSONs (data/realiad-variety/Real-IAD_Variety_jsons/)
  - C1 view only
  - NORMAL  = anomaly_class == "OK"
  - ANOMALY = anomaly_class != "OK" AND mask_path is not null (visible defect)
              (maskless anomalies are excluded by default; --maskless-as-normal
               moves them into the normal pool instead)
  - per product the balanced pool is min(#normal, #anomaly) -- in Variety normals
    are the bottleneck (~50 C1 normals vs ~197 C1 anomalies per product), so the
    achievable balanced maximum is ~15,962 C1 images (7,981 N + 7,981 A).

Output (default variety_c1_compiled/):
  sft_train.json    messages/images format (assistant content empty -> filled by generator)
  grpo_train.json   question/answer format (answer empty -> filled by generator)

Image paths are emitted with LINUX_PREFIX so they are drop-in for the cluster;
generate_variety_traces.py maps them back to the local data dir automatically.

Usage:
  python sample_variety.py                 # max balanced set, 50/50 SFT/GRPO
  python sample_variety.py --per-product 40
  python sample_variety.py --sft-frac 0.5 --maskless-as-normal
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
VIEW = "C1"
ROOT        = Path(__file__).parent
DATA_ROOT   = ROOT / "data" / "realiad-variety"
JSON_DIR    = DATA_ROOT / "Real-IAD_Variety_jsons"
OUTPUT_DIR  = ROOT / "variety_c1_compiled"
LINUX_PREFIX = "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/realiad-variety"

# Normalise obvious label noise (casing / duplicates) seen in the metadata.
DEFECT_NORMALISE = {
    "porosity": "Porosity",
}


def norm_defect(name: str) -> str:
    return DEFECT_NORMALISE.get(name, name)


def rel_path(category: str, image_path: str) -> str:
    """Path relative to the data root, e.g. audio_jack_socket/OK/S0001/...png"""
    return f"{category}/{image_path}"


def linux_path(rel: str) -> str:
    return f"{LINUX_PREFIX}/{rel}"


def image_id_from_rel(rel: str) -> str:
    p = Path(rel)
    return "_".join(list(p.parts[:-1]) + [p.stem])


def question_for(product: str) -> str:
    return (f"Analyze the provided image of the {product.replace('_', ' ')}. "
            f"Determine if there are any anomalies present. If an anomaly is detected, "
            f"specify its type and location, and provide a detailed reasoning for your conclusion.")


def collect(maskless_as_normal: bool):
    """Return {category: {"ok": [rel...], "ng": [rel...]}} for the C1 view."""
    if not JSON_DIR.exists():
        raise SystemExit(
            f"Metadata not found: {JSON_DIR}\n"
            f"Run:  python download_variety.py --jsons-only")
    data = {}
    view_tag = f"_{VIEW}_"
    for jf in sorted(JSON_DIR.glob("*.json")):
        cat = jf.stem
        entries = json.loads(jf.read_text(encoding="utf-8"))
        ok, ng = [], []
        for split in ("train", "test"):
            for e in entries.get(split, []):
                ip = e["image_path"]
                if view_tag not in ip:
                    continue
                rel = rel_path(cat, ip)
                if e["anomaly_class"] == "OK":
                    ok.append(rel)
                elif e.get("mask_path"):
                    ng.append(rel)
                elif maskless_as_normal:
                    ok.append(rel)
                # else: maskless anomaly -> dropped
        data[cat] = {"ok": ok, "ng": ng}
    return data


def sample(data, rng, per_product, sft_frac):
    """Per product: take n = min(#ok, #ng) balanced pairs (capped by per_product),
    then split disjointly into SFT and GRPO by sft_frac."""
    sft, grpo = [], []   # each item: (rel, product, is_anomaly)
    for cat in sorted(data):
        ok = list(data[cat]["ok"]); ng = list(data[cat]["ng"])
        rng.shuffle(ok); rng.shuffle(ng)
        n = min(len(ok), len(ng))
        if per_product is not None:
            n = min(n, per_product)
        if n == 0:
            continue
        n_sft = int(round(n * sft_frac))
        n_grpo = n - n_sft
        # disjoint slices
        for i in range(n_sft):
            sft.append((ng[i], cat, True))
            sft.append((ok[i], cat, False))
        for i in range(n_sft, n_sft + n_grpo):
            grpo.append((ng[i], cat, True))
            grpo.append((ok[i], cat, False))
    rng.shuffle(sft); rng.shuffle(grpo)
    return sft, grpo


def make_grpo_entry(rel, product, is_anom):
    return {
        "image_id": image_id_from_rel(rel),
        "image_path": linux_path(rel),
        "product": product,
        "is_anomaly": is_anom,
        "question": question_for(product),
        "answer": "",
        "gt_label": "yes" if is_anom else "no",
    }


def make_sft_entry(rel, product):
    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question_for(product)}"},
            {"role": "assistant", "content": ""},
        ],
        "images": [linux_path(rel)],
    }


def main():
    ap = argparse.ArgumentParser(description="Sample C1 50/50 product-stratified SFT+GRPO from Real-IAD Variety.")
    ap.add_argument("--per-product", type=int, default=None,
                    help="Cap balanced pairs per product (default: max available = min(#ok,#ng))")
    ap.add_argument("--sft-frac", type=float, default=0.5,
                    help="Fraction of each product's balanced pairs assigned to SFT (rest -> GRPO)")
    ap.add_argument("--maskless-as-normal", action="store_true",
                    help="Count anomaly-class images with no mask as NORMAL (default: drop them)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    rng = random.Random(SEED)
    data = collect(args.maskless_as_normal)

    print(f"{'category':<30} {'OK':>5} {'NG':>6} {'min':>5}")
    tot_ok = tot_ng = tot_min = 0
    for cat in sorted(data):
        o, g = len(data[cat]["ok"]), len(data[cat]["ng"])
        m = min(o, g)
        tot_ok += o; tot_ng += g; tot_min += m
    print(f"{'TOTAL':<30} {tot_ok:>5} {tot_ng:>6} {tot_min:>5}")
    print(f"\nMax balanced (both splits combined): {2*tot_min}  ({tot_min} N + {tot_min} A)\n")

    sft, grpo = sample(data, rng, args.per_product, args.sft_frac)

    # stats + overlap check
    sft_ids = {image_id_from_rel(r) for r, _, _ in sft}
    grpo_ids = {image_id_from_rel(r) for r, _, _ in grpo}
    overlap = sft_ids & grpo_ids
    print(f"SFT : {len(sft):>6}  ({sum(a for _,_,a in sft)} anom / {sum(1 for _,_,a in sft if not a)} norm)")
    print(f"GRPO: {len(grpo):>6}  ({sum(a for _,_,a in grpo)} anom / {sum(1 for _,_,a in grpo if not a)} norm)")
    print(f"SFT/GRPO overlap: {len(overlap)} (must be 0)")
    print(f"Products in SFT: {len(set(p for _,p,_ in sft))}  GRPO: {len(set(p for _,p,_ in grpo))}")

    args.out.mkdir(parents=True, exist_ok=True)
    grpo_out = [make_grpo_entry(r, p, a) for r, p, a in grpo]
    sft_out  = [make_sft_entry(r, p) for r, p, _ in sft]
    (args.out / "grpo_train.json").write_text(json.dumps(grpo_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "sft_train.json").write_text(json.dumps(sft_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out / 'grpo_train.json'} ({len(grpo_out)})")
    print(f"-> {args.out / 'sft_train.json'} ({len(sft_out)})")


if __name__ == "__main__":
    main()
