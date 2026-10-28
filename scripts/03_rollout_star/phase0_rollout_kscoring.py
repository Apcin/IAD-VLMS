#!/usr/bin/env python
"""Phase 0/1 rollout — k=8 rollouts per item with per-rollout scoring.

Pulls items from BOTH source pools (SFT 6k + GRPO 4k), normalizes to a common
schema, rolls out k attempts from run2/ckpt-530 with the GRPO prompt format,
scores each rollout with accuracy_reward + consistency_reward, and writes
ONE JSONL record per item with all k rollouts and their scores.

Output schema (one record per item):
    {
      "image_path":  "/bulk/.../Real-IAD/images/transistor1/.../...jpg",
      "product":     "transistor1",
      "gt_trace":    "<think>...</think><location>...</location><type>...</type><answer>yes/no</answer>",
      "gt_answer":   "yes" or "no",
      "is_anomaly":  true/false,
      "source_pool": "sft_6k" or "grpo_4k",
      "user_prompt": "<image>\\nAnalyze the provided image of the transistor1..." (the prompt the model saw),
      "rollouts": [
        {"text": "<think>...</think>...", "acc": 1.8, "format": 1.0, "len": 432},
        ...
      ],
    }

Downstream bucketer reads this and decides good/rewrite/correction.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, "/bulk/aacudad/reasoning_traces/Training/iad_r1_grpo_custom/stage_rl")
from reward import accuracy_reward, consistency_reward  # noqa: E402


def make_train_prompt(product: str) -> str:
    """Matches the eval script and SFT-Iter1 training prompt exactly."""
    return (
        f"Analyze the provided image of the {product}. "
        "Determine if there are any anomalies present. "
        "If an anomaly is detected, specify its type and location, "
        "and provide a detailed reasoning for your conclusion."
    )


def extract_product_from_path(image_path: str) -> str:
    """e.g. /bulk/.../Real-IAD/images/transistor1/OK/S0234/...jpg → 'transistor1'"""
    parts = image_path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "images" and i + 1 < len(parts):
            return parts[i + 1]
    return "object"


def normalize_sharegpt(row, source_pool):
    """combined_6k_train.json format → common schema."""
    image_path = row["images"][0]
    gt_trace = row["messages"][1]["content"]
    product = extract_product_from_path(image_path)
    m = re.search(r"<answer>(.*?)</answer>", gt_trace, re.DOTALL | re.IGNORECASE)
    gt_answer = m.group(1).strip().lower() if m else None
    return {
        "image_path": image_path,
        "product": product,
        "gt_trace": gt_trace,
        "gt_answer": gt_answer,
        "is_anomaly": (gt_answer == "yes"),
        "source_pool": source_pool,
    }


def normalize_per_item(row, source_pool):
    """grpo_train.json format → common schema."""
    return {
        "image_path": row["image_path"],
        "product": row.get("product") or extract_product_from_path(row["image_path"]),
        "gt_trace": row["answer"],
        "gt_answer": "yes" if row.get("is_anomaly") else "no",
        "is_anomaly": bool(row.get("is_anomaly")),
        "source_pool": source_pool,
    }


def load_pool(path, pool_name):
    with open(path) as f:
        rows = json.load(f)
    if not rows:
        return []
    if "messages" in rows[0]:
        return [normalize_sharegpt(r, pool_name) for r in rows]
    return [normalize_per_item(r, pool_name) for r in rows]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="GRPO checkpoint to roll out (run2/ckpt-530)")
    p.add_argument("--sft_pool",
                   default="/bulk/aacudad/reasoning_traces/Training/datasets_small_new_v4/combined_6k_train.json")
    p.add_argument("--grpo_pool",
                   default="/bulk/aacudad/reasoning_traces/Training/datasets_small_15k_c1_only/grpo_train.json")
    p.add_argument("--sft_label", default="sft_6k",
                   help="source_pool label for items from --sft_pool")
    p.add_argument("--grpo_label", default="grpo_4k",
                   help="source_pool label for items from --grpo_pool")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--k", type=int, default=8, help="rollouts per item")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_pixels", type=int, default=480000)
    p.add_argument("--limit", type=int, default=None,
                   help="Sample this many items total (stratified)")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_grpo_prompt", action="store_true", default=False,
                   help="Use GRPO 'You are an expert...' preamble (matches run2 training)")
    return p.parse_args()


def stratified_sample(items, n, seed):
    """Take n items balanced across (source_pool, is_anomaly)."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for it in items:
        buckets[(it["source_pool"], it["is_anomaly"])].append(it)
    n_per_bucket = n // len(buckets)
    leftover = n - n_per_bucket * len(buckets)
    out = []
    for key, lst in buckets.items():
        rng.shuffle(lst)
        out.extend(lst[:n_per_bucket])
    # Distribute leftover from largest bucket
    all_remaining = [it for key, lst in buckets.items() for it in lst[n_per_bucket:]]
    rng.shuffle(all_remaining)
    out.extend(all_remaining[:leftover])
    rng.shuffle(out)
    return out


@torch.inference_mode()
def generate_k(model, processor, image_path, user_text, k, temperature, top_p,
               max_new_tokens, max_pixels):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * s)), max(1, int(h * s))))
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": user_text}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=k,
        pad_token_id=pad_id,
    )
    input_len = inputs["input_ids"].shape[-1]
    return [processor.tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
            for i in range(k)]


def score_one(text, gt_trace):
    """Use the production reward functions on a single rollout."""
    fake = [[{"content": text}]]
    sol = [gt_trace]
    acc = accuracy_reward(fake, sol)[0]
    cons = consistency_reward(fake, sol)[0]
    return acc, cons


def score_one_detailed(text, gt_trace):
    """Detailed per-component breakdown that REPLICATES accuracy_reward's logic
    while exposing type_score, loc_score, answer_match separately. Returns the
    composite acc (same as score_one) plus the components for downstream analysis.

    For OK items (gt='no'):  acc = answer_match in {0,1}; type/loc are None
    For NG items (gt='yes'): acc = (type_score + loc_score)/2 + (1.0 if model says 'yes')
                              type_score in {0, 0.2, 0.5, 0.7, 0.9, 1.0}
                              loc_score in {0, 1}
    """
    # Extract GT answer
    sol_m = re.search(r"<answer>(.*?)</answer>", gt_trace, re.DOTALL | re.IGNORECASE)
    gt_answer = sol_m.group(1).strip().lower() if sol_m else None

    # Extract model answer
    ans_m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    model_answer = ans_m.group(1).strip().lower() if ans_m else None

    answer_match = 1.0 if (model_answer is not None and gt_answer is not None
                            and model_answer == gt_answer) else 0.0

    type_score = None
    loc_score = None

    if gt_answer == "no":
        acc = answer_match
    elif gt_answer == "yes":
        # Type score (Nomic similarity, bucketed)
        try:
            gpt_t = re.search(r"<type>(.*?)</type>", text, re.DOTALL | re.IGNORECASE)
            gt_t = re.search(r"<type>(.*?)</type>", gt_trace, re.DOTALL | re.IGNORECASE)
            if gpt_t and gt_t:
                from reward_process import type_reward as tr
                calc = tr.AnomalyRewardCalculator()
                type_score = float(calc.compute_reward(
                    gpt_t.group(1).strip().lower(),
                    gt_t.group(1).strip().lower(),
                ))
            else:
                type_score = 0.0
        except Exception:
            type_score = 0.0

        # Location score (3×3 grid match, binary)
        try:
            gpt_l = re.search(r"<location>(.*?)</location>", text, re.DOTALL | re.IGNORECASE)
            gt_l = re.search(r"<location>(.*?)</location>", gt_trace, re.DOTALL | re.IGNORECASE)
            if gpt_l and gt_l:
                from reward_process import location_reward as lr
                loc_score = float(lr.map_location_to_region(
                    gpt_l.group(1).strip().lower(),
                    gt_l.group(1).strip().lower(),
                ))
            else:
                loc_score = 0.0
        except Exception:
            loc_score = 0.0

        # Composite: same formula as accuracy_reward()
        # (type+loc)/2 + 1.0 if model says yes (== answer_match when gt is 'yes')
        acc = (type_score + loc_score) / 2.0 + answer_match
    else:
        acc = 0.0

    # Format/consistency reward via the production function
    cons = consistency_reward([[{"content": text}]], [gt_trace])[0]

    return {
        "acc": float(acc),
        "format": float(cons),
        "type_score": float(type_score) if type_score is not None else None,
        "loc_score": float(loc_score) if loc_score is not None else None,
        "answer_match": float(answer_match),
        "model_answer": model_answer,
        "gt_answer": gt_answer,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[init] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}", flush=True)
    print(f"[init] torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)

    print(f"[init] Loading pools …", flush=True)
    items = load_pool(args.sft_pool, args.sft_label) + load_pool(args.grpo_pool, args.grpo_label)
    print(f"[init] Total items: {len(items)} "
          f"({sum(1 for x in items if x['source_pool']==args.sft_label)} {args.sft_label} + "
          f"{sum(1 for x in items if x['source_pool']==args.grpo_label)} {args.grpo_label})", flush=True)

    if args.limit:
        items = stratified_sample(items, args.limit, args.seed)
        print(f"[init] Stratified sample → {len(items)} items", flush=True)

    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
        print(f"[init] Shard {args.shard_id}/{args.num_shards} → {len(items)} items", flush=True)

    print(f"[init] Loading model {args.model_path} …", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[init] Model loaded on {model.device}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)

    # Resume support: if the output file already exists, read which image_paths are done
    # and skip those items. Opens output in append mode.
    done_paths = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as fin:
            for line in fin:
                try:
                    rec = json.loads(line)
                    if rec.get("image_path"):
                        done_paths.add(rec["image_path"])
                except Exception:
                    pass
        print(f"[resume] Found {len(done_paths)} items already done in {args.output_jsonl}", flush=True)
        items = [it for it in items if it["image_path"] not in done_paths]
        print(f"[resume] {len(items)} items remaining", flush=True)
        fout = open(args.output_jsonl, "a")
    else:
        fout = open(args.output_jsonl, "w")

    stats = defaultdict(int)

    for it in tqdm(items, desc=f"Rollout k={args.k} shard{args.shard_id}"):
        # Build user prompt
        train_prompt = make_train_prompt(it["product"])
        if args.use_grpo_prompt:
            user_text = (
                "You are an expert in detecting defects in image. "
                "Your task is to detect if there are any defects in the test image."
                "Are there any defects in the query image?"
            )
        else:
            user_text = train_prompt  # matches v2 SFT recipe

        try:
            rollouts = generate_k(
                model, processor, it["image_path"], user_text,
                args.k, args.temperature, args.top_p,
                args.max_new_tokens, args.max_pixels,
            )
        except Exception as e:
            print(f"GEN ERROR on {it['image_path']}: {e}", flush=True)
            stats["gen_error"] += 1
            continue

        rollout_records = []
        for r in rollouts:
            scores = score_one_detailed(r, it["gt_trace"])
            rollout_records.append({
                "text": r,
                "len": len(r),
                **scores,  # acc, format, type_score, loc_score, answer_match, model_answer, gt_answer
            })

        record = {
            "image_path": it["image_path"],
            "product": it["product"],
            "gt_trace": it["gt_trace"],
            "gt_answer": it["gt_answer"],
            "is_anomaly": it["is_anomaly"],
            "source_pool": it["source_pool"],
            "user_prompt": user_text,
            "rollouts": rollout_records,
        }
        fout.write(json.dumps(record) + "\n")
        fout.flush()
        stats[f"done_{it['source_pool']}_{it['gt_answer']}"] += 1

    fout.close()
    print(f"[done] Wrote {args.output_jsonl}", flush=True)
    print(f"[stats] {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
