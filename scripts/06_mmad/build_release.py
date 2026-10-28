"""
Build the release package of the MMAD leakage experiment:
  1. hf_staging/anomalythink_mmad/   -> Hugging Face dataset (traces only, relative MMAD/... paths)
  2. repository_tu_delft_vlms/       -> scripts/06_mmad, traces/mmad, results/mmad

Image paths are normalised to "MMAD/<dataset>/<product>/..." relative to a data root, the same
convention as the AnomalyThink dataset's "Real-IAD/images/..." paths. No image is shipped.
"""
import json, os, re, shutil
from pathlib import Path

SRC = Path(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/mmad_leak_experiment")
MMAD_ABS = os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/reasoning_traces_gen/data/MMAD/"
HF = Path(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/hf_staging/anomalythink_mmad")
REPO = Path(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/repository_tu_delft_vlms")
DATASET_ID = os.environ.get("HF_DATASET_ID", "aacudad/AnomalyThink-MMAD")


def rel(p: str) -> str:
    p = p.replace("\\", "/")
    i = p.find(MMAD_ABS)
    return "MMAD/" + p[i + len(MMAD_ABS):] if i >= 0 else p


def norm_sharegpt(path: Path):
    out = []
    for e in json.load(open(path)):
        e = dict(e)
        e["images"] = [rel(x) for x in e["images"]]
        out.append(e)
    return out


def main():
    # ---------------- traces with provenance ----------------
    recs = []
    for f in sorted((SRC / "traces").glob("trace_*.json")):
        r = json.load(open(f))
        recs.append({
            "image": "MMAD/" + r["key"], "dataset": r["dataset"], "product": r["product"],
            "is_anomaly": r["is_anomaly"], "hints": r["hints"], "reasoning": r["reasoning"],
            "teacher": r["model"], "thinking_level": r["thinking_level"], "prompt_version": r["prompt_version"],
        })
    recs.sort(key=lambda r: r["image"])
    split1 = json.load(open(SRC / "split.json"))
    split2 = json.load(open(SRC / "split_bal.json"))
    s1_train, s1_test = set(split1["train_keys"]), set(split1["test_keys"])
    s2_train, s2_test = set(split2["train_keys"]), set(split2["test_keys"])
    for r in recs:
        k = r["image"][5:]
        r["run1_split"] = "train" if k in s1_train else "test"
        r["run2_split"] = "train" if k in s2_train else "test"

    # ---------------- HF staging ----------------
    if HF.exists():
        shutil.rmtree(HF)
    (HF / "splits").mkdir(parents=True)
    with open(HF / "traces_8293.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    all_sg = []
    for r in recs:
        all_sg.append({"messages": [
            {"role": "user", "content": "<image>\nAnalyze the provided image of the " + r["product"] +
             ". Determine if there are any anomalies present. If an anomaly is detected, specify its type and location, "
             "and provide a detailed reasoning for your conclusion."},
            {"role": "assistant", "content": r["reasoning"]}], "images": [r["image"]]})
    json.dump(all_sg, open(HF / "mmad_all_8293.json", "w"), indent=1, ensure_ascii=False)
    by_img = {e["images"][0]: e for e in all_sg}
    files = {
        "splits/run1_unbalanced_train_1600.json": [by_img["MMAD/" + k] for k in split1["train_keys"]],
        "splits/run1_unbalanced_test_6693.json": [by_img["MMAD/" + k] for k in split1["test_keys"]],
        "splits/run2_balanced_train_1600.json": [by_img["MMAD/" + k] for k in split2["train_keys"]],
        "splits/run2_balanced_test_6693.json": [by_img["MMAD/" + k] for k in split2["test_keys"]],
        "splits/one_per_product_and_defect_144.json": [by_img["MMAD/" + k] for k in split1["train_cat_keys"]],
        "splits/one_per_product_39.json": [by_img["MMAD/" + k] for k in split1["train_prod_keys"]],
    }
    for name, rows in files.items():
        json.dump(rows, open(HF / name, "w"), indent=1, ensure_ascii=False)
    json.dump({"run1_unbalanced": {"train": split1["train_keys"], "test": split1["test_keys"]},
               "run2_balanced": {"train": split2["train_keys"], "test": split2["test_keys"],
                                 "common_test_with_run1": split2["common_test_keys_with_unbalanced_run"]},
               "note": "keys are MMAD image paths relative to the MMAD root (mmad.json keys)"},
              open(HF / "splits" / "split_keys.json", "w"), indent=1)
    shutil.copy(SRC / "RESULTS.md", HF / "RESULTS.md")
    shutil.copy(SRC / "inspector_prompt_test_v2_mmad_run.txt", HF / "system_prompt_inspector_v2.txt")
    card = open(SRC / "hf_card_template.md").read().replace("{{DATASET_ID}}", DATASET_ID)
    open(HF / "README.md", "w").write(card)
    print("HF staging:", sum(p.stat().st_size for p in HF.rglob("*") if p.is_file()) // 2**20, "MiB")

    # ---------------- repo ----------------
    sdir = REPO / "scripts" / "06_mmad"
    tdir = REPO / "traces" / "mmad"
    rdir = REPO / "results" / "mmad"
    for d in (sdir, tdir, rdir):
        d.mkdir(parents=True, exist_ok=True)
    for name in ["generate_mmad_traces_v4.py", "compile_split.py", "compile_split_bal.py", "eval_heldout_vllm.py",
                 "score_heldout.py", "eval_all_checkpoints.sh", "eval_ckpts_on_gpu.sh", "eval_train_keys.sh",
                 "sft_mmad_train1600.yaml", "sft_mmad_train1600_bal_6ep.yaml", "run_sft_mmad_train1600.sh",
                 "run_sft_mmad_train1600_bal_6ep.sh", "build_release.py", "hf_card_template.md",
                 "inspector_prompt_test_v2_mmad_run.txt", "repo_readme_snippet.md"]:
        txt = open(SRC / name).read()
        # portabilise: the cluster root becomes an environment variable
        txt = txt.replace(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "", "${WORK_DIR}") if name.endswith((".sh", ".yaml")) \
            else txt.replace('os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "', 'os.environ.get("WORK_DIR", os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "") + "')
        if name.endswith(".py") and "import os" not in txt:
            txt = txt.replace("import json", "import json\nimport os", 1)
        open(sdir / name, "w").write(txt)
    shutil.copy(SRC / "RESULTS.md", sdir / "RESULTS.md")
    open(sdir / "README.md", "w").write(open(SRC / "repo_readme_snippet.md").read())
    for name, rows in files.items():
        json.dump(rows, open(tdir / Path(name).name, "w"), indent=1, ensure_ascii=False)
    shutil.copy(HF / "splits" / "split_keys.json", tdir / "split_keys.json")
    shutil.copy(HF / "traces_8293.jsonl", tdir / "traces_8293.jsonl")
    for run, ev in (("run1_unbalanced_4ep", "evals"), ("run2_balanced_6ep", "evals_bal")):
        for ck in sorted((SRC / ev).iterdir()):
            if not ck.is_dir():
                continue
            dst = rdir / run / ck.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ck.glob("eval_*.json"):
                d = json.load(open(f))
                for x in d["results"]:
                    x["absolute_path"] = rel(x["absolute_path"])
                json.dump(d, open(dst / f.name, "w"))
    kcr = SRC / "evals_thesis_kcr" / "checkpoint-376"
    if kcr.exists() and len(list(kcr.glob("eval_*_heldout.json"))) == 4:
        dst = rdir / "thesis_kcr_no_mmad_in_training" / "checkpoint-376"
        dst.mkdir(parents=True, exist_ok=True)
        for f in kcr.glob("eval_*.json"):
            d = json.load(open(f))
            for x in d["results"]:
                x["absolute_path"] = rel(x["absolute_path"])
            json.dump(d, open(dst / f.name, "w"))
    shutil.copy(SRC / "split.json", rdir / "split_run1_unbalanced.json")
    shutil.copy(SRC / "split_bal.json", rdir / "split_run2_balanced.json")
    print("repo results:", sum(p.stat().st_size for p in rdir.rglob("*") if p.is_file()) // 2**20, "MiB")
    print("done")


if __name__ == "__main__":
    main()
