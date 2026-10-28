#!/usr/bin/env python
"""v2 rollout-and-filter for SFT-Iter2 data — CORRECTED prompt + SFT pool source.

Differences from v1:
  - Source data is the SFT pool (combined_6k_train.json, 6000 items),
    not the GRPO pool (4236). This matches v1 paper philosophy and the
    baseline's pipeline structure (SFT and GRPO trained on disjoint splits).
  - User message in the OUTPUT JSON matches SFT-Iter1's format exactly:
    `<image>\\nAnalyze the provided image of the {product}. Determine if there
    are any anomalies present. ...` (i.e. `make_train_prompt(product)`).
  - Rollout prompt sent to the model is identical to that (so model is asked
    the same thing it will be trained to answer).
  - Optional sharding via --shard_id and --num_shards for multi-GPU parallel
    rollout (3 GPUs cuts the 10-hour rollout to ~3.5h).
  - Source data is assumed to be LLaMA-Factory sharegpt format: each row has
    `messages: [{role:user, content:"<image>\\nAnalyze..."}, {role:assistant, content:"..."}]`
    and `images: ["/path/to/img.jpg"]`. The assistant message contains the
    GT trace with <answer>/<type>/<location> tags and is used as the
    reward `solution`.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="GRPO checkpoint to roll out, e.g. .../grpo_run2/checkpoint-530")
    p.add_argument("--source_data", required=True,
                   help="LLaMA-Factory sharegpt JSON (e.g. combined_6k_train.json)")
    p.add_argument("--output_path", required=True,
                   help="Output LLaMA-Factory sharegpt JSON for SFT-Iter2 training")
    p.add_argument("--k", type=int, default=4, help="Generations per prompt")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_pixels", type=int, default=480000)
    p.add_argument("--reward_threshold_yes", type=float, default=1.5,
                   help="Min accuracy_reward to accept a NG trace. Range [0,2].")
    p.add_argument("--reward_threshold_no", type=float, default=1.0,
                   help="Min accuracy_reward to accept an OK trace. Range [0,1].")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None,
                   help="Only process this many items (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    # GPU selection: set via CUDA_VISIBLE_DEVICES env var BEFORE invoking python.
    # (torch reads CUDA_VISIBLE_DEVICES at import time; setting it in main() is too late.)
    return p.parse_args()


def load_model(model_path):
    print(f"[init] Loading processor from {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"[init] Loading model from {model_path}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[init] Model loaded on device {model.device}", flush=True)
    return model, processor


@torch.inference_mode()
def generate_k(model, processor, image_path, user_text, k, temperature, top_p,
               max_new_tokens, max_pixels):
    """user_text is the FULL text after <image>\\n in the sharegpt messages."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    # Reconstruct the user message exactly as LLaMA-Factory would format it:
    # type=image content + the text from the source.
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    ).to(model.device)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=k,
        pad_token_id=pad_id,
    )
    input_len = inputs["input_ids"].shape[-1]
    return [
        processor.tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
        for i in range(k)
    ]


def score(completion, gt_solution_text):
    """gt_solution_text = the full assistant message from the source JSON,
    which contains <answer>/<type>/<location> tags accuracy_reward needs."""
    fake_completion = [[{"content": completion}]]
    fake_solution = [gt_solution_text]
    acc = accuracy_reward(fake_completion, fake_solution)[0]
    cons = consistency_reward(fake_completion, fake_solution)[0]
    return acc, cons


def extract_user_text(messages):
    """Strip the leading '<image>\\n' from the user content if present."""
    u = messages[0]["content"]
    if u.startswith("<image>\n"):
        return u[len("<image>\n"):]
    if u.startswith("<image>"):
        return u[len("<image>"):].lstrip("\n")
    return u


def extract_gt_label(assistant_content):
    """Extract yes/no from the GT assistant message's <answer> tag."""
    m = re.search(r"<answer>(.*?)</answer>", assistant_content, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return None


def main():
    args = parse_args()
    # CUDA device selected via CUDA_VISIBLE_DEVICES env var set by parent shell.
    print(f"[init] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}", flush=True)
    print(f"[init] torch.cuda.device_count() = {torch.cuda.device_count()}", flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.source_data) as f:
        items = json.load(f)

    # Sharding
    if args.num_shards > 1:
        items = [item for i, item in enumerate(items) if i % args.num_shards == args.shard_id]
        print(f"[init] Shard {args.shard_id}/{args.num_shards}: {len(items)} items", flush=True)

    if args.limit is not None:
        items = items[:args.limit]
    print(f"[init] Processing {len(items)} items from {args.source_data}", flush=True)

    model, processor = load_model(args.model_path)

    # Output accumulator (don't balance per-shard; balance at merge time)
    accepted_yes = []
    accepted_no = []
    stats = defaultdict(int)

    for item in tqdm(items, desc=f"Rollout shard{args.shard_id}"):
        # Source is sharegpt format
        user_text = extract_user_text(item["messages"])
        assistant_text = item["messages"][1]["content"]
        gt_label = extract_gt_label(assistant_text)
        image_path = item["images"][0]

        if gt_label not in ("yes", "no"):
            stats["bad_gt"] += 1
            continue
        stats[f"input_{gt_label}"] += 1

        try:
            comps = generate_k(
                model, processor,
                image_path, user_text,
                args.k, args.temperature, args.top_p,
                args.max_new_tokens, args.max_pixels,
            )
        except Exception as e:
            print(f"GEN ERROR on {image_path}: {e}", flush=True)
            stats["gen_error"] += 1
            continue

        threshold = args.reward_threshold_yes if gt_label == "yes" else args.reward_threshold_no
        passing = []
        for comp in comps:
            acc, cons = score(comp, assistant_text)
            if cons == 1.0 and acc >= threshold:
                passing.append((comp, acc))

        if not passing:
            stats[f"no_pass_{gt_label}"] += 1
            continue

        # Shortest correct (matches v1 paper claim, not v1 code quirk)
        passing.sort(key=lambda x: len(x[0]))
        chosen = passing[0][0]
        stats[f"kept_{gt_label}"] += 1

        # Output sharegpt entry — SAME user text as source so SFT-Iter2 trains
        # on identical prompt format as SFT-Iter1.
        entry = {
            "messages": [
                {"role": "user", "content": f"<image>\n{user_text}"},
                {"role": "assistant", "content": chosen},
            ],
            "images": [image_path],
        }
        (accepted_yes if gt_label == "yes" else accepted_no).append(entry)

    # Write per-shard output (merge & balance happens after all shards finish)
    out = {"yes": accepted_yes, "no": accepted_no, "stats": dict(stats)}

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[done] Shard {args.shard_id}: wrote {len(accepted_yes)} yes + {len(accepted_no)} no -> {args.output_path}", flush=True)
    print(f"[stats] {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
