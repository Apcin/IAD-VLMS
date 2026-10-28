"""Arm D — STaR-style self-rationalisation on the items the policy got wrong.

The rationaliser is the SAME model that produced the k=8 rollouts for Arms A/B/C:
outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530.

Input pool = the two needs_correction.json files written by phase0_bucket.py, i.e.
exactly the items on which 0 of 8 rollouts passed. That is the same set that
phase1b_gemini_correct.py --mode correct consumed to build Arm B's corrected
traces, so Arm D and Arm B are drawn from the identical failure pool.

Difference from Arm B, and the whole point of this arm: the hint contains ONLY
the gold yes/no verdict. No defect type, no location, no mask, no teacher.
STaR (arXiv 2203.14465, section 3.2) also strips the hint before the trace is
stored, so every record keeps `user_prompt` as the UNHINTED training prompt.

Filtering reuses the production scorer (phase0_rollout_kscoring.score_one_detailed,
which calls the shipped accuracy_reward / consistency_reward). No new parser.

Generation is vLLM. Prompt template, sampling knobs and the 262144-pixel cap
match Training/evaluate_qwen25vl_7b_trainprompt.py and phase0_rollout_kscoring.py.

Usage:
  CUDA_VISIBLE_DEVICES=3 python star_rationalise.py \
      --output_jsonl <file> --limit 100 --k 1
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

# The needs_correction.json pools and the GRPO checkpoint are NOT in this repository:
# the pools carry absolute Real-IAD image paths and the checkpoint is on HuggingFace.
# WORK_DIR is the workspace that holds Training/ and outputs/; it defaults to the parent
# of this repository. On another machine:  export WORK_DIR=/path/to/workspace
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
R = os.environ.get("WORK_DIR") or str(REPO_ROOT.parent)
os.environ.setdefault("HF_HOME", f"{R}/hf_cache")
# The Nomic similarity used by the type reward is served over HTTP by
# gemini_judge_server_v2.py. 5300 is the default port of that server.
os.environ.setdefault("GEMINI_JUDGE_URL", "http://127.0.0.1:5300")
sys.path.insert(0, str(REPO_ROOT / "scripts/02_grpo/stage_rl"))   # reward.py
sys.path.insert(0, str(HERE))                                     # phase0_rollout_kscoring.py

CKPT_530 = os.environ.get(
    "GRPO_CKPT_530", f"{R}/outputs/grpo_qwen25vl_7b_6k_frozen_ep3_full_run2/checkpoint-530")
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_POOLS = [
    f"{R}/Training/phase0_full_10k_20260529_015821",
    f"{R}/Training/phase0_heldout_20260601",
]

# Production eval protocol: evaluate_qwen25vl_7b_trainprompt.py sets
# processor.image_processor.min_pixels = 256*28*28 and max_pixels = 262144.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 262144


# ---------------------------------------------------------------------------
# Hint. Answer only. No type, no location, no mask.
# ---------------------------------------------------------------------------
def make_hint(gt_answer: str) -> str:
    verdict = "is" if gt_answer == "yes" else "is not"
    return f"Ground truth: this image {verdict} anomalous. Explain why."


def gold_verdict(item: dict):
    """The verdict the reward function will score against.

    Take it from the <answer> tag of gt_trace, exactly like accuracy_reward and
    consistency_reward do, not from the item's own gt_answer / is_anomaly field.
    28 of the 1,942 failures carry an is_anomaly flag that disagrees with their
    own gold trace (all of them from the grpo_4k pool). Hinting off the flag
    would hand those items a verdict the scorer then marks wrong."""
    m = re.search(r"<answer>(.*?)</answer>", item.get("gt_trace") or "", re.DOTALL | re.IGNORECASE)
    v = m.group(1).strip().lower() if m else None
    if v in ("yes", "no"):
        return v
    v = (item.get("gt_answer") or "").strip().lower()
    return v if v in ("yes", "no") else None


# ---------------------------------------------------------------------------
# transformers-5.0 -> 4.57/vLLM compatibility mirror.
#
# Every Qwen checkpoint in outputs/ was saved by transformers 5.0. Three problems
# when vLLM (transformers 4.57.1) loads one:
#   1. the config is the 5.0 nested dialect (text_config + rope_parameters). Under
#      4.57 from_dict silently drops `architectures` and never sets rope_scaling,
#      so vLLM aborts with "No model architectures are specified" and the
#      mrope_section never reaches the engine.
#   2. the tokenizer / processor JSONs are in the 5.0 dialect and there is no
#      preprocessor_config.json at all.
#   3. the weights themselves are fine: the safetensors still use the classic
#      model.layers.* / visual.* / lm_head.weight names, so no remapping is needed.
# Same class of fix the LLaVA watcher applies (watch_grpo_llava_vllm_evals.sh),
# except we never write into the production checkpoint. We build a sibling
# directory that symlinks the weights and carries 4.x metadata taken from the
# base model, which is what the eval harness already does for the processor
# ("always load processor from base so tokenizer/image_processor are correct").
# ---------------------------------------------------------------------------

# Fields that must agree between the checkpoint and the base config, otherwise
# borrowing the base config would silently change the model.
_ARCH_CHECKS = [
    ("hidden_size", "hidden_size"),
    ("num_hidden_layers", "num_hidden_layers"),
    ("num_attention_heads", "num_attention_heads"),
    ("num_key_value_heads", "num_key_value_heads"),
    ("intermediate_size", "intermediate_size"),
    ("vocab_size", "vocab_size"),
    ("rms_norm_eps", "rms_norm_eps"),
    ("max_position_embeddings", "max_position_embeddings"),
]


def ensure_vllm_compat_dir(ckpt: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    for w in ("model.safetensors", "model.safetensors.index.json"):
        src = os.path.join(ckpt, w)
        dst = os.path.join(out_dir, w)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    for shard in sorted(f for f in os.listdir(ckpt) if re.match(r"model-\d+-of-\d+\.safetensors$", f)):
        dst = os.path.join(out_dir, shard)
        if not os.path.exists(dst):
            os.symlink(os.path.join(ckpt, shard), dst)

    from huggingface_hub import hf_hub_download, snapshot_download

    ck_cfg = json.load(open(os.path.join(ckpt, "config.json")))
    ck_tc = ck_cfg.get("text_config", ck_cfg)
    base_cfg = json.load(open(hf_hub_download(BASE_MODEL, "config.json")))

    for ck_key, base_key in _ARCH_CHECKS:
        a, b = ck_tc.get(ck_key), base_cfg.get(base_key)
        if a is not None and b is not None and a != b:
            raise SystemExit(f"[compat] refusing to reuse the base config: {ck_key} "
                             f"is {a} in the checkpoint but {b} in {BASE_MODEL}")
    ck_rope = ck_tc.get("rope_parameters", {})
    ck_theta = ck_rope.get("rope_theta", ck_tc.get("rope_theta"))
    if ck_theta is not None and float(ck_theta) != float(base_cfg["rope_theta"]):
        raise SystemExit(f"[compat] rope_theta mismatch: checkpoint {ck_theta} vs base "
                         f"{base_cfg['rope_theta']} (the transformers-5 rope trap)")
    ck_sec = ck_rope.get("mrope_section")
    if ck_sec is not None and list(ck_sec) != list(base_cfg["rope_scaling"]["mrope_section"]):
        raise SystemExit(f"[compat] mrope_section mismatch: {ck_sec} vs "
                         f"{base_cfg['rope_scaling']['mrope_section']}")
    for k, v in ck_cfg.get("vision_config", {}).items():
        if k in base_cfg["vision_config"] and base_cfg["vision_config"][k] != v:
            raise SystemExit(f"[compat] vision_config.{k}: {v} vs {base_cfg['vision_config'][k]}")

    base_cfg["torch_dtype"] = "bfloat16"
    base_cfg["use_cache"] = True
    json.dump(base_cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    gc_src = os.path.join(ckpt, "generation_config.json")
    if os.path.exists(gc_src):
        shutil.copy(gc_src, os.path.join(out_dir, "generation_config.json"))

    # Tokenizer + image processor from the base model. GRPO never touched them.
    base_dir = snapshot_download(
        BASE_MODEL,
        allow_patterns=["*.json", "merges.txt", "*.jinja"],
    )
    for f in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
              "vocab.json", "merges.txt", "chat_template.json", "chat_template.jinja",
              "added_tokens.json", "special_tokens_map.json", "processor_config.json"):
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, f))
    return out_dir


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------
def load_failure_pool(pool_dirs):
    """The Arm-B correction set: phase0_bucket.py's needs_correction.json, i.e.
    every item where 0 of the 8 rollouts passed (format==1.0 and acc>=1.5 NG /
    >=1.0 OK). Attaches the Gemini trace Arm B used for the same image so the
    two can be compared side by side."""
    items, gem = [], {}
    for d in pool_dirs:
        nc = os.path.join(d, "needs_correction.json")
        if os.path.exists(nc):
            for it in json.load(open(nc)):
                it["_pool_dir"] = d
                items.append(it)
        gpath = os.path.join(d, "gemini_corrected.jsonl")
        if os.path.exists(gpath):
            with open(gpath) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("corrected_trace"):
                        gem[r["image_path"]] = r["corrected_trace"]
    for it in items:
        it["_gemini_corrected_trace"] = gem.get(it["image_path"])
    return items


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=CKPT_530)
    p.add_argument("--pool_dir", action="append", default=None,
                   help="phase0 dir holding needs_correction.json (repeatable)")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--kept_sharegpt", default=None,
                   help="also write the kept rationalisations as LLaMA-Factory sharegpt")
    p.add_argument("--k", type=int, default=1, help="hinted samples per failed item")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    p.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    p.add_argument("--limit", type=int, default=None, help="dry-run subset size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=12288)
    p.add_argument("--compat_dir", default=None)
    # Production bucket rule from phase0_bucket.py, kept as flags so the
    # thresholds are never silently re-invented here.
    p.add_argument("--threshold_yes", type=float, default=1.5)
    p.add_argument("--threshold_no", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    pool_dirs = args.pool_dir or DEFAULT_POOLS

    import phase0_rollout_kscoring as P0
    from PIL import Image

    items = load_failure_pool(pool_dirs)
    n_all = len(items)
    n_flag_mismatch = 0
    usable = []
    for it in items:
        v = gold_verdict(it)
        if v is None:
            continue
        if v != (it.get("gt_answer") or "").strip().lower():
            n_flag_mismatch += 1
        it["_gold"] = v
        usable.append(it)
    print(f"[pool] needs_correction items: {n_all} "
          f"({sum(1 for x in items if gold_verdict(x)=='yes')} NG, "
          f"{sum(1 for x in items if gold_verdict(x)=='no')} OK, "
          f"{n_all-len(usable)} without a parsable gold verdict -> dropped)", flush=True)
    print(f"[pool] {n_flag_mismatch} items whose is_anomaly flag disagrees with their own "
          f"gold trace, hint follows the gold trace", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(usable)
    if args.num_shards > 1:
        usable = usable[args.shard_id::args.num_shards]
    if args.limit:
        usable = usable[:args.limit]
    print(f"[pool] this run: {len(usable)} items "
          f"({sum(1 for x in usable if x['_gold']=='yes')} NG, "
          f"{sum(1 for x in usable if x['_gold']=='no')} OK)", flush=True)

    done = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_path"])
                except Exception:
                    pass
        usable = [it for it in usable if it["image_path"] not in done]
        print(f"[resume] {len(done)} already done, {len(usable)} to go", flush=True)
    if not usable:
        print("[done] nothing to do", flush=True)
        return

    compat = args.compat_dir or (args.model_path.rstrip("/") + "_vllmcompat")
    t0 = time.time()
    compat = ensure_vllm_compat_dir(args.model_path, compat)
    print(f"[compat] {compat} ready in {time.time()-t0:.1f}s", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    t_load = time.time()
    llm = LLM(
        model=compat,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
    )
    load_s = time.time() - t_load
    print(f"[load] vLLM engine ready in {load_s:.1f}s", flush=True)

    proc = AutoProcessor.from_pretrained(compat)
    proc.image_processor.min_pixels = args.min_pixels
    proc.image_processor.max_pixels = args.max_pixels
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens, seed=args.seed)

    fout = open(args.output_jsonl, "a")
    kept_rows = []
    t_all = time.time()
    chunk_times = []
    n_done = 0

    for ci in range(0, len(usable), args.chunk):
        t_chunk = time.time()
        chunk = usable[ci:ci + args.chunk]
        reqs, metas = [], []
        for it in chunk:
            try:
                img = Image.open(it["image_path"]).convert("RGB")
            except Exception as e:
                print(f"[skip] {it['image_path']}: {e}", flush=True)
                continue
            product = it.get("product") or P0.extract_product_from_path(it["image_path"])
            unhinted = P0.make_train_prompt(product)
            hint = make_hint(it["_gold"])
            hinted = f"{unhinted}\n{hint}"
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": hinted},
            ]}]
            prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            reqs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
            metas.append((it, product, unhinted, hint, hinted))

        outs = llm.generate(reqs, sp)

        for (it, product, unhinted, hint, hinted), o in zip(metas, outs):
            th_yes, th_no = args.threshold_yes, args.threshold_no
            samples = []
            for c in o.outputs:
                s = P0.score_one_detailed(c.text, it["gt_trace"])
                passes_answer = s["answer_match"] == 1.0
                passes_format = s["format"] == 1.0
                th = th_yes if it["_gold"] == "yes" else th_no
                samples.append({
                    "text": c.text,
                    "len": len(c.text),
                    **s,
                    "passes_answer": passes_answer,
                    "passes_format": passes_format,
                    "passes_answer_and_format": bool(passes_answer and passes_format),
                    "passes_production_bucket": bool(passes_format and s["acc"] >= th),
                })

            def shortest(pred):
                ok = [s for s in samples if pred(s)]
                return min(ok, key=lambda s: s["len"])["text"] if ok else None

            kept = shortest(lambda s: s["passes_answer_and_format"])
            kept_bucket = shortest(lambda s: s["passes_production_bucket"])

            rec = {
                "image_path": it["image_path"],
                "product": product,
                "gt_answer": it["_gold"],
                "gt_answer_flag": it.get("gt_answer"),
                "is_anomaly": it.get("is_anomaly"),
                "source_pool": it.get("source_pool"),
                "pool_dir": it.get("_pool_dir"),
                "gt_trace": it["gt_trace"],
                # STaR section 3.2: the hint is stripped before the trace is stored.
                "user_prompt": unhinted,
                "hint": hint,
                "hinted_user_prompt": hinted,
                "best_failed_attempt": it.get("best_failed_attempt"),
                "gemini_corrected_trace": it.get("_gemini_corrected_trace"),
                "samples": samples,
                "n_answer": sum(1 for s in samples if s["passes_answer"]),
                "n_answer_and_format": sum(1 for s in samples if s["passes_answer_and_format"]),
                "n_production_bucket": sum(1 for s in samples if s["passes_production_bucket"]),
                "kept_trace": kept,
                "kept_trace_production": kept_bucket,
            }
            fout.write(json.dumps(rec) + "\n")
            if kept:
                kept_rows.append({
                    "messages": [
                        {"role": "user", "content": f"<image>\n{unhinted}"},
                        {"role": "assistant", "content": kept},
                    ],
                    "images": [it["image_path"]],
                })
        fout.flush()

        chunk_times.append((time.time() - t_chunk, len(chunk)))
        n_done += len(chunk)
        rate = (time.time() - t_all) / max(n_done, 1)
        print(f"[gen s{args.shard_id}] {n_done}/{len(usable)} | {rate:.2f}s/item overall | "
              f"chunk {chunk_times[-1][0]/max(chunk_times[-1][1],1):.2f}s/item | "
              f"ETA {(len(usable)-n_done)*rate/3600:.2f}h", flush=True)

    fout.close()
    if args.kept_sharegpt and kept_rows:
        json.dump(kept_rows, open(args.kept_sharegpt, "w"), indent=2)
        print(f"[out] {len(kept_rows)} kept traces -> {args.kept_sharegpt}", flush=True)

    total = time.time() - t_all
    steady = chunk_times[1:] if len(chunk_times) > 1 else chunk_times
    steady_s = sum(t for t, _ in steady) / max(sum(n for _, n in steady), 1)
    print(f"[timing] model load {load_s:.1f}s | generation {total:.1f}s for {n_done} items "
          f"| {total/max(n_done,1):.2f}s/item including first chunk "
          f"| {steady_s:.2f}s/item steady state", flush=True)
    print(f"[done] {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
