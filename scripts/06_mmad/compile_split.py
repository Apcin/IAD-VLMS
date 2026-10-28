"""
Compile the MMAD traces into LlamaFactory sharegpt files and make the train/test split.

Split: stratified by (dataset, product, is_anomaly), seed 42, exactly TRAIN_N images in
training (OmniAD's SFT set has 1.6K images), allocated to the strata in proportion to their
size by largest remainder, the rest is the held-out test set. Two smaller nested training sets are
also written for the dose-response runs:
  * train_cat : one image per (dataset, product, defect folder), the "one example per
                category" reading of OmniAD's data section (146 images)
  * train_prod: one image per (dataset, product), the other reading (38 images)
Both are drawn from inside the 1,600-image training set, so the test set is the same for all runs.

Outputs (in this directory):
  split.json                      train/test keys, per-stratum counts
  mmad_sft_train1600.json         LlamaFactory sharegpt, 1,600 MMAD images
  mmad_sft_train_cat.json         146 images
  mmad_sft_train_prod.json        38 images
and the three datasets are registered in LlamaFactory/data/dataset_info.json.
"""
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
TRACES = HERE / "traces"
MMAD_ROOT = Path(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/reasoning_traces_gen/data/MMAD")
DATASET_INFO = Path(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/LlamaFactory/data/dataset_info.json")
TRAIN_N = 1600
SEED = 42

sys.path.insert(0, os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/Training")
from evaluate_qwen25vl_7b_trainprompt import make_train_prompt  # the exact SFT / eval prompt


def defect_folder(key: str) -> str:
    # DS-MVTec/bottle/image/broken_large/000.png -> broken_large ; VisA/candle/test/bad/000.jpg -> bad
    return key.split("/")[-2]


def main():
    recs = []
    for f in sorted(TRACES.glob("trace_*.json")):
        r = json.load(open(f))
        recs.append(r)
    print(f"{len(recs)} traces")

    strata = defaultdict(list)
    for r in recs:
        strata[(r["dataset"], r["product"], r["is_anomaly"])].append(r)

    # largest-remainder allocation of TRAIN_N over the strata
    total = sum(len(v) for v in strata.values())
    keys = sorted(strata)
    exact = {k: TRAIN_N * len(strata[k]) / total for k in keys}
    alloc = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: exact[k] - alloc[k], reverse=True)[:TRAIN_N - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(SEED)
    train, test = [], []
    for k in keys:
        items = sorted(strata[k], key=lambda r: r["key"])
        rng.shuffle(items)
        n_train = min(alloc[k], len(items) - 1)   # leave at least one test image per stratum
        train += items[:n_train]
        test += items[n_train:]
    train_keys = {r["key"] for r in train}

    # nested small sets from inside the training set
    rng2 = random.Random(SEED)
    by_cat, by_prod = defaultdict(list), defaultdict(list)
    for r in sorted(train, key=lambda r: r["key"]):
        by_cat[(r["dataset"], r["product"], defect_folder(r["key"]))].append(r)
        by_prod[(r["dataset"], r["product"])].append(r)
    train_cat = [rng2.choice(v) for k, v in sorted(by_cat.items())]
    train_prod = [rng2.choice(v) for k, v in sorted(by_prod.items())]

    def to_sharegpt(rs):
        out = []
        for r in rs:
            out.append({
                "messages": [
                    {"role": "user", "content": "<image>\n" + make_train_prompt(r["product"])},
                    {"role": "assistant", "content": r["reasoning"]},
                ],
                "images": [str(MMAD_ROOT / r["key"])],
            })
        return out

    files = {
        "mmad_sft_train1600": train,
        "mmad_sft_train_cat": train_cat,
        "mmad_sft_train_prod": train_prod,
    }
    for name, rs in files.items():
        json.dump(to_sharegpt(rs), open(HERE / f"{name}.json", "w"), indent=1, ensure_ascii=False)

    def summary(rs):
        s = defaultdict(lambda: [0, 0])
        for r in rs:
            s[r["dataset"]][int(r["is_anomaly"])] += 1
        return {k: {"normal": v[0], "anomalous": v[1]} for k, v in sorted(s.items())}

    split = {
        "seed": SEED, "train_n": TRAIN_N,
        "train_keys": sorted(train_keys),
        "test_keys": sorted(r["key"] for r in test),
        "train_cat_keys": sorted(r["key"] for r in train_cat),
        "train_prod_keys": sorted(r["key"] for r in train_prod),
        "counts": {"train1600": summary(train), "test": summary(test),
                   "train_cat": summary(train_cat), "train_prod": summary(train_prod)},
    }
    json.dump(split, open(HERE / "split.json", "w"), indent=1)
    print(json.dumps(split["counts"], indent=1))
    print("sizes:", {k: len(v) for k, v in files.items()}, "test", len(test))

    info = json.load(open(DATASET_INFO))
    for name in files:
        info[name] = {
            "file_name": str(HERE / f"{name}.json"),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        }
    json.dump(info, open(DATASET_INFO, "w"), indent=2)
    print("registered in", DATASET_INFO)


if __name__ == "__main__":
    main()
