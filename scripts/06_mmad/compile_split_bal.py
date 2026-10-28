"""
Balanced variant of the MMAD training split: 800 anomalous + 800 normal images, allocated over the
products in proportion to product size (largest remainder), each product half normal and half
anomalous where the normals allow (cap: at most 60 % of a product's normals or anomalies go to
training). Images already in the unbalanced run's training set are taken first, so the held-out
sets of the two runs overlap as much as possible. Same seed. Writes split_bal.json and
mmad_sft_train1600_bal.json and registers the dataset.
"""
import json
import os, random, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).parent
MMAD_ROOT = Path(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/reasoning_traces_gen/data/MMAD")
DATASET_INFO = Path(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/LlamaFactory/data/dataset_info.json")
SEED = 42; N_PER_CLASS = 800; CAP = 0.60
sys.path.insert(0, os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/Training")
from evaluate_qwen25vl_7b_trainprompt import make_train_prompt

recs = [json.load(open(f)) for f in sorted((HERE / "traces").glob("trace_*.json"))]
prev_train = set(json.load(open(HERE / "split.json"))["train_keys"])
by_prod = defaultdict(lambda: {True: [], False: []})
for r in recs:
    by_prod[(r["dataset"], r["product"])][r["is_anomaly"]].append(r)
prods = sorted(by_prod)
rng = random.Random(SEED)
train, alloc_info = [], {}
for is_anom in (True, False):
    sizes = {p: len(by_prod[p][is_anom]) for p in prods}
    total = sum(sizes.values())
    exact = {p: N_PER_CLASS * sizes[p] / total for p in prods}
    alloc = {p: min(int(exact[p]), int(CAP * sizes[p])) for p in prods}
    # largest remainder, respecting the cap
    order = sorted(prods, key=lambda p: exact[p] - alloc[p], reverse=True)
    i = 0
    while sum(alloc.values()) < N_PER_CLASS:
        p = order[i % len(order)]
        if alloc[p] < int(CAP * sizes[p]):
            alloc[p] += 1
        i += 1
        if i > 100000:
            raise SystemExit("cannot fill allocation under cap")
    for p in prods:
        items = sorted(by_prod[p][is_anom], key=lambda r: r["key"])
        rng.shuffle(items)
        first = [r for r in items if r["key"] in prev_train]
        rest = [r for r in items if r["key"] not in prev_train]
        chosen = (first + rest)[:alloc[p]]
        train += chosen
        alloc_info[f"{p[0]}/{p[1]}/{'anom' if is_anom else 'normal'}"] = f"{len(chosen)}/{sizes[p]}"
train_keys = {r["key"] for r in train}
test = [r for r in recs if r["key"] not in train_keys]
def to_sharegpt(rs):
    return [{"messages": [{"role": "user", "content": "<image>\n" + make_train_prompt(r["product"])},
                          {"role": "assistant", "content": r["reasoning"]}],
             "images": [str(MMAD_ROOT / r["key"])]} for r in rs]
json.dump(to_sharegpt(train), open(HERE / "mmad_sft_train1600_bal.json", "w"), indent=1, ensure_ascii=False)
def summary(rs):
    s = defaultdict(lambda: [0, 0])
    for r in rs: s[r["dataset"]][int(r["is_anomaly"])] += 1
    return {k: {"normal": v[0], "anomalous": v[1]} for k, v in sorted(s.items())}
prev_test = set(json.load(open(HERE / "split.json"))["test_keys"])
common = sorted(prev_test & {r["key"] for r in test})
split = {"seed": SEED, "n_per_class": N_PER_CLASS, "cap": CAP,
         "train_keys": sorted(train_keys), "test_keys": sorted(r["key"] for r in test),
         "common_test_keys_with_unbalanced_run": common,
         "counts": {"train": summary(train), "test": summary(test)}, "alloc": alloc_info}
json.dump(split, open(HERE / "split_bal.json", "w"), indent=1)
print(json.dumps(split["counts"], indent=1)); print("train", len(train), "test", len(test), "common test with run 1", len(common))
print("reused from run-1 train:", len(train_keys & prev_train))
info = json.load(open(DATASET_INFO))
info["mmad_sft_train1600_bal"] = {"file_name": str(HERE / "mmad_sft_train1600_bal.json"), "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
json.dump(info, open(DATASET_INFO, "w"), indent=2); print("registered")
