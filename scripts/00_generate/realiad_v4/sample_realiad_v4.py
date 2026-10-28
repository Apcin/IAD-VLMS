"""
Sample 3K SFT + 3K GRPO entries from Real-IAD (C1 angle only).

Rules:
  - Only C1 camera angle images
  - Only NG images that have a ground-truth mask (visible defect)
  - 50/50 anomaly/normal target per product
  - Stratified across all 30 Real-IAD products
  - Zero overlap between SFT and GRPO splits
  - If a product has fewer than target anomalies, use what's available
    and redistribute the shortfall across other products

Output:
  3k_realiad_v4/grpo_train.json
  3k_realiad_v4/sft_train.json
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict

SEED = 42
TARGET_PER_SPLIT = 3000  # 3K per split
IMAGES_ROOT = Path(__file__).parent / "data" / "Real-IAD" / "images"
OUTPUT_DIR = Path(__file__).parent / "3k_realiad_v4"
LINUX_PREFIX = "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images"


def to_linux_path(local_path: Path) -> str:
    rel = local_path.relative_to(IMAGES_ROOT)
    return f"{LINUX_PREFIX}/{rel.as_posix()}"


def derive_image_id(img_path: Path) -> str:
    rel = img_path.relative_to(IMAGES_ROOT)
    return "_".join(list(rel.parts[:-1]) + [rel.stem])


def collect_c1_images():
    """Collect all C1 images grouped by product, split into OK and NG-with-mask."""
    products = sorted([p.name for p in IMAGES_ROOT.iterdir() if p.is_dir()])
    data = {}

    for prod in products:
        ok_images = []
        ng_images = []

        # OK images
        ok_dir = IMAGES_ROOT / prod / "OK"
        if ok_dir.exists():
            for f in ok_dir.rglob("*.jpg"):
                if "_C1_" in f.name:
                    ok_images.append(f)

        # NG images with mask only
        ng_dir = IMAGES_ROOT / prod / "NG"
        if ng_dir.exists():
            for f in ng_dir.rglob("*.jpg"):
                if "_C1_" in f.name:
                    mask = f.with_suffix(".png")
                    if mask.exists():
                        ng_images.append(f)

        data[prod] = {"ok": ok_images, "ng": ng_images}

    return data


def make_entry_grpo(img_path: Path, product: str, is_anomaly: bool) -> dict:
    image_id = derive_image_id(img_path)
    return {
        "image_id": image_id,
        "image_path": to_linux_path(img_path),
        "product": product,
        "is_anomaly": is_anomaly,
        "question": f"Analyze the provided image of the {product.replace('_', ' ')}. "
                    f"Determine if there are any anomalies present. If an anomaly is detected, "
                    f"specify its type and location, and provide a detailed reasoning for your conclusion.",
        "answer": "",
        "gt_label": "yes" if is_anomaly else "no",
    }


def make_entry_sft(img_path: Path, product: str) -> dict:
    linux_path = to_linux_path(img_path)
    question = (f"Analyze the provided image of the {product.replace('_', ' ')}. "
                f"Determine if there are any anomalies present. If an anomaly is detected, "
                f"specify its type and location, and provide a detailed reasoning for your conclusion.")
    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": ""},
        ],
        "images": [linux_path],
    }


def sample_splits(data: dict, rng: random.Random):
    """Sample SFT and GRPO splits with stratification."""
    products = sorted(data.keys())
    n_products = len(products)

    # Target per product (equal split)
    base_per_product = TARGET_PER_SPLIT // n_products  # 100
    # Half anomaly, half normal
    target_anom = base_per_product // 2  # 50
    target_norm = base_per_product // 2  # 50

    sft_entries_grpo = []  # (img_path, product, is_anomaly)
    grpo_entries = []
    shortfall_anom = 0
    shortfall_norm = 0
    overflow_products_anom = []
    overflow_products_norm = []

    # First pass: allocate per product
    product_pools = {}
    for prod in products:
        ok_pool = list(data[prod]["ok"])
        ng_pool = list(data[prod]["ng"])
        rng.shuffle(ok_pool)
        rng.shuffle(ng_pool)

        # Need target_anom * 2 from NG pool (for SFT + GRPO)
        # Need target_norm * 2 from OK pool
        avail_ng = len(ng_pool) // 2
        avail_ok = len(ok_pool) // 2

        actual_anom = min(target_anom, avail_ng)
        actual_norm = min(target_norm, avail_ok)

        if actual_anom < target_anom:
            shortfall_anom += target_anom - actual_anom
        else:
            overflow_products_anom.append(prod)

        if actual_norm < target_norm:
            shortfall_norm += target_norm - actual_norm
        else:
            overflow_products_norm.append(prod)

        product_pools[prod] = {
            "ok": ok_pool,
            "ng": ng_pool,
            "sft_anom": actual_anom,
            "sft_norm": actual_norm,
            "grpo_anom": actual_anom,
            "grpo_norm": actual_norm,
        }

    # Second pass: redistribute shortfall
    if shortfall_anom > 0 and overflow_products_anom:
        extra_each = shortfall_anom // len(overflow_products_anom)
        remainder = shortfall_anom % len(overflow_products_anom)
        rng.shuffle(overflow_products_anom)
        for i, prod in enumerate(overflow_products_anom):
            pool = product_pools[prod]
            avail = len(pool["ng"]) // 2
            extra = extra_each + (1 if i < remainder else 0)
            extra = min(extra, avail - pool["sft_anom"])
            pool["sft_anom"] += extra
            pool["grpo_anom"] += extra

    if shortfall_norm > 0 and overflow_products_norm:
        extra_each = shortfall_norm // len(overflow_products_norm)
        remainder = shortfall_norm % len(overflow_products_norm)
        rng.shuffle(overflow_products_norm)
        for i, prod in enumerate(overflow_products_norm):
            pool = product_pools[prod]
            avail = len(pool["ok"]) // 2
            extra = extra_each + (1 if i < remainder else 0)
            extra = min(extra, avail - pool["sft_norm"])
            pool["sft_norm"] += extra
            pool["grpo_norm"] += extra

    # Build entries
    sft_list = []
    grpo_list = []

    for prod in products:
        pool = product_pools[prod]
        ng = pool["ng"]
        ok = pool["ok"]

        # SFT anomaly
        sft_ng = ng[:pool["sft_anom"]]
        # GRPO anomaly (next slice, no overlap)
        grpo_ng = ng[pool["sft_anom"]:pool["sft_anom"] + pool["grpo_anom"]]

        # SFT normal
        sft_ok = ok[:pool["sft_norm"]]
        # GRPO normal
        grpo_ok = ok[pool["sft_norm"]:pool["sft_norm"] + pool["grpo_norm"]]

        for img in sft_ng:
            sft_list.append((img, prod, True))
        for img in sft_ok:
            sft_list.append((img, prod, False))
        for img in grpo_ng:
            grpo_list.append((img, prod, True))
        for img in grpo_ok:
            grpo_list.append((img, prod, False))

    rng.shuffle(sft_list)
    rng.shuffle(grpo_list)

    return sft_list, grpo_list


def main():
    rng = random.Random(SEED)
    print("Collecting C1 images...")
    data = collect_c1_images()

    for prod in sorted(data.keys()):
        print(f"  {prod:<25} OK={len(data[prod]['ok']):>4}  NG+mask={len(data[prod]['ng']):>4}")

    print(f"\nSampling {TARGET_PER_SPLIT} SFT + {TARGET_PER_SPLIT} GRPO...")
    sft_list, grpo_list = sample_splits(data, rng)

    # Stats
    print(f"\nSFT:  {len(sft_list)} total  "
          f"({sum(1 for _,_,a in sft_list if a)} anom, "
          f"{sum(1 for _,_,a in sft_list if not a)} norm)")
    print(f"GRPO: {len(grpo_list)} total  "
          f"({sum(1 for _,_,a in grpo_list if a)} anom, "
          f"{sum(1 for _,_,a in grpo_list if not a)} norm)")

    # Verify no overlap
    sft_ids = {derive_image_id(p) for p, _, _ in sft_list}
    grpo_ids = {derive_image_id(p) for p, _, _ in grpo_list}
    overlap = sft_ids & grpo_ids
    print(f"Overlap: {len(overlap)} (should be 0)")

    # Per-product stats
    from collections import Counter
    sft_prods = Counter(prod for _, prod, _ in sft_list)
    grpo_prods = Counter(prod for _, prod, _ in grpo_list)
    print(f"\nPer-product counts (SFT / GRPO):")
    for prod in sorted(data.keys()):
        s_a = sum(1 for _, p, a in sft_list if p == prod and a)
        s_n = sum(1 for _, p, a in sft_list if p == prod and not a)
        g_a = sum(1 for _, p, a in grpo_list if p == prod and a)
        g_n = sum(1 for _, p, a in grpo_list if p == prod and not a)
        print(f"  {prod:<25} SFT: {s_a:>3}a+{s_n:>3}n={s_a+s_n:>4}  "
              f"GRPO: {g_a:>3}a+{g_n:>3}n={g_a+g_n:>4}")

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grpo_out = [make_entry_grpo(p, prod, anom) for p, prod, anom in grpo_list]
    sft_out = [make_entry_sft(p, prod) for p, prod, _ in sft_list]

    with open(OUTPUT_DIR / "grpo_train.json", "w", encoding="utf-8") as f:
        json.dump(grpo_out, f, indent=2, ensure_ascii=False)
    print(f"\n-> {OUTPUT_DIR / 'grpo_train.json'} ({len(grpo_out)} entries)")

    with open(OUTPUT_DIR / "sft_train.json", "w", encoding="utf-8") as f:
        json.dump(sft_out, f, indent=2, ensure_ascii=False)
    print(f"-> {OUTPUT_DIR / 'sft_train.json'} ({len(sft_out)} entries)")


if __name__ == "__main__":
    main()
