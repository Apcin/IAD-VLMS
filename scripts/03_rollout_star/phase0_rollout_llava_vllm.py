"""Phase 0 rollout for LLaVA-OneVision via vLLM — drop-in producer of the exact
rollouts_raw.jsonl schema of phase0_rollout_kscoring.py (which is HF/Qwen-only).

Reuses load_pool / make_train_prompt / score_one_detailed from the original module,
swaps generation for vLLM (n=k in one pass, prefill shared). Same knobs as the 10k
run: k=8, temperature 0.7, top_p 0.9, max_new_tokens 512, 480000-px cap, seed 42.
Prompt: user-turn-only trainprompt (LLaVA SFT format). Resumable; shardable.

Usage:
  phase0_rollout_llava_vllm.py --model_path <ckpt> --output_jsonl <file> \
      [--shard_id 0 --num_shards 3] [--k 8] [--chunk 64]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Workspace root: the directory holding Training/, outputs/ and hf_cache/. Defaults to the
# parent of this repository (the cluster layout); override with `export WORK_DIR=...`.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
R = os.environ.get("WORK_DIR") or str(REPO_ROOT.parent)
os.environ.setdefault("HF_HOME", f"{R}/hf_cache")
os.environ.setdefault("GEMINI_JUDGE_URL", "http://127.0.0.1:5300")
sys.path.insert(0, str(REPO_ROOT / "scripts/02_grpo/stage_rl"))   # reward.py
sys.path.insert(0, str(HERE))                                     # phase0_rollout_kscoring.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--sft_pool", default=f"{R}/Training/datasets_small_new_v4/combined_6k_train.json")
    p.add_argument("--grpo_pool", default=f"{R}/Training/datasets_small_15k_c1_only/grpo_train.json")
    p.add_argument("--sft_label", default="sft_6k")
    p.add_argument("--grpo_label", default="grpo_4k")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_pixels", type=int, default=480000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    args = p.parse_args()

    import phase0_rollout_kscoring as P0
    from PIL import Image
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    items = P0.load_pool(args.sft_pool, args.sft_label) + P0.load_pool(args.grpo_pool, args.grpo_label)
    items = items[args.shard_id::args.num_shards]
    done = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_path"])
                except Exception:
                    pass
    items = [it for it in items if it["image_path"] not in done]
    print(f"[p0-llava shard {args.shard_id}/{args.num_shards}] {len(items)} to go ({len(done)} resumed)", flush=True)
    if not items:
        return

    def cap(img, mx=args.max_pixels):
        if img.width * img.height > mx:
            f = (mx / (img.width * img.height)) ** 0.5
            img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS)
        return img

    t0 = time.time()
    llm = LLM(model=args.model_path, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=12288, limit_mm_per_prompt={"image": 1}, seed=args.seed)
    print(f"[p0-llava] engine load {time.time()-t0:.0f}s", flush=True)
    proc = AutoProcessor.from_pretrained(args.model_path)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens)

    fout = open(args.output_jsonl, "a")
    t_all, n_done = time.time(), 0
    for ci in range(0, len(items), args.chunk):
        chunk = items[ci:ci + args.chunk]
        reqs, metas = [], []
        for it in chunk:
            try:
                img = cap(Image.open(it["image_path"]).convert("RGB"))
            except Exception as e:
                print(f"[p0-llava] skip {it['image_path']}: {e}", flush=True)
                continue
            user_text = P0.make_train_prompt(it["product"])
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ]}]
            prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            reqs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
            metas.append((it, user_text))
        outs = llm.generate(reqs, sp)
        for (it, user_text), o in zip(metas, outs):
            rollout_records = []
            for c in o.outputs:
                scores = P0.score_one_detailed(c.text, it["gt_trace"])
                rollout_records.append({"text": c.text, "len": len(c.text), **scores})
            fout.write(json.dumps({
                "image_path": it["image_path"],
                "product": it["product"],
                "gt_trace": it["gt_trace"],
                "gt_answer": it["gt_answer"],
                "is_anomaly": it["is_anomaly"],
                "source_pool": it["source_pool"],
                "user_prompt": user_text,
                "rollouts": rollout_records,
            }) + "\n")
        fout.flush()
        n_done = ci + len(chunk)
        rate = (time.time() - t_all) / max(n_done, 1)
        print(f"[p0-llava s{args.shard_id}] {n_done}/{len(items)} | {rate:.2f}s/item | "
              f"ETA {(len(items)-n_done)*rate/3600:.1f}h", flush=True)
    fout.close()
    print(f"[p0-llava s{args.shard_id}] DONE in {(time.time()-t_all)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
