#!/usr/bin/env python
"""Build the NO-REASONING (labels-only) ablation dataset from the exact 6K SFT file.

Control for: "do the reasoning traces actually help detection, or would plain yes/no labels on the
same 6K images do just as well?"

Source  : Training/datasets_small_new_v4/combined_6k_train.json  (what `iad_sft_6k_train` points at,
          i.e. the file the reported SFT-6K model (80.16/64.78) was trained on) -- 6000 examples.
Changes : (1) the user prompt drops the type/location/reasoning request and asks for a bare verdict;
          (2) the assistant target drops <think>/<location>/<type>, keeping ONLY <answer>Yes|No</answer>.
Images and the product-conditioned preamble are otherwise untouched.

The <answer> wrapper is deliberately kept: the eval harness's extract_tags() reads <answer>
independently of <think>, so the parser needs no change and output-format is not a confound.
The training prompt here MUST stay character-identical to make_noreason_prompt() in
scripts/04_eval/evaluate_qwen25vl_7b_trainprompt.py (--noreason-prompt).
"""
import argparse, json, os, re, sys
from pathlib import Path

# The 6K source corpus is NOT in this repository: its records point at raw Real-IAD images,
# which we do not redistribute (see README section 3). WORK_DIR is the workspace that holds
# Training/; it defaults to the parent of this repository. Override with --src / --dst or
# `export WORK_DIR=...`.
WORK_DIR = Path(os.environ.get("WORK_DIR") or Path(__file__).resolve().parents[2].parent)
SRC = WORK_DIR / "Training/datasets_small_new_v4/combined_6k_train.json"
DST = WORK_DIR / "Training/datasets_noreason/combined_6k_noreason_train.json"

OLD_TAIL = ("If an anomaly is detected, specify its type and location, "
            "and provide a detailed reasoning for your conclusion.")
NEW_TAIL = "Answer with yes or no."

# must equal make_noreason_prompt() in the eval harness
def expected_prompt(product: str) -> str:
    return (f"Analyze the provided image of the {product}. "
            "Determine if there are any anomalies present. "
            "Answer with yes or no.")

def main():
    global SRC, DST
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    args = ap.parse_args()
    SRC, DST = args.src, args.dst

    rows = json.load(open(SRC))
    out, n_yes, n_no = [], 0, 0
    for r in rows:
        user = r["messages"][0]["content"]
        asst = r["messages"][1]["content"]

        if OLD_TAIL not in user:
            sys.exit(f"FATAL: unexpected user prompt (no known tail):\n{user[:200]}")
        new_user = user.replace(OLD_TAIL, NEW_TAIL)

        m = re.search(r"<answer>\s*(Yes|No)\s*</answer>", asst, re.IGNORECASE)
        if not m:
            sys.exit(f"FATAL: no <answer> tag in target:\n{asst[-200:]}")
        verdict = m.group(1).capitalize()          # 'Yes' / 'No'
        n_yes += verdict == "Yes"; n_no += verdict == "No"

        out.append({
            "messages": [
                {"role": "user",      "content": new_user},
                {"role": "assistant", "content": f"<answer>{verdict}</answer>"},
            ],
            "images": r["images"],
        })

    DST.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(DST, "w"), indent=1)

    # --- verification ---
    assert len(out) == len(rows), "row count changed"
    assert n_yes == n_no == len(rows) // 2, f"class balance broke: {n_yes} yes / {n_no} no"
    for r in out:
        u = r["messages"][0]["content"]
        assert u.startswith("<image>\n"), "lost the <image> marker"
        body = u.split("\n", 1)[1]
        prod = body.split("image of the ", 1)[1].split(".", 1)[0]
        assert body == expected_prompt(prod), f"prompt != make_noreason_prompt:\n{body}"
        assert re.fullmatch(r"<answer>(Yes|No)</answer>", r["messages"][1]["content"]), "target not bare"
        assert "<think>" not in r["messages"][1]["content"]
        assert "<type>" not in r["messages"][1]["content"]
        assert "<location>" not in r["messages"][1]["content"]

    print(f"wrote {DST}")
    print(f"  n={len(out)}  yes={n_yes}  no={n_no}  (source n={len(rows)})")
    print(f"  user  : {out[0]['messages'][0]['content']}")
    print(f"  target: {out[0]['messages'][1]['content']}")
    print("  all assertions passed (prompt matches make_noreason_prompt; target is answer-only)")

if __name__ == "__main__":
    main()
