"""KCR stage 1: k=8 rollouts from the GRPO checkpoint with vLLM (single GPU).

Faithful to GRPO training generation: same chat-template rendering (checkpoint's
processor = ckpt-188 files), same user-turn GRPO prompt (single_img=1 template),
same 480000-px pre-resize, temperature 1.0 / top_p 0.9 / top_k 50, max 512 tokens.

Chunked + resumable: appends JSONL, skips image_ids already present.
Usage: kcr_rollout_vllm.py --ckpt <dir> --out <jsonl> [--k 8] [--chunk 384]
"""
import argparse
import json
import os
import time
from pathlib import Path

# Workspace root: the directory holding Training/, outputs/ and hf_cache/. Defaults to the
# parent of this repository; override with `export WORK_DIR=...`.
WORK_DIR = os.environ.get("WORK_DIR") or str(Path(__file__).resolve().parents[2].parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=f"{WORK_DIR}/Training/datasets_small_15k_c1_only/grpo_train.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", f"{WORK_DIR}/hf_cache")
    from PIL import Image
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    QP = ("You are an expert in detecting defects in image. "
          "Your task is to detect if there are any defects in the test image."
          "{Question}")

    def cap(img, mx=480000):
        if img.width * img.height > mx:
            f = (mx / (img.width * img.height)) ** 0.5
            img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS)
        return img

    data = json.load(open(args.data))
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    todo = [d for d in data if d["image_id"] not in done]
    print(f"[rollout] {len(data)} items, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    t0 = time.time()
    llm = LLM(model=args.ckpt, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=12288, limit_mm_per_prompt={"image": 1})
    print(f"[rollout] engine load {time.time()-t0:.0f}s", flush=True)
    proc = AutoProcessor.from_pretrained(args.ckpt)
    sp = SamplingParams(n=args.k, temperature=1.0, top_p=0.9, top_k=50, max_tokens=512)

    t_all = time.time()
    for ci in range(0, len(todo), args.chunk):
        chunk = todo[ci:ci + args.chunk]
        reqs, metas = [], []
        for d in chunk:
            try:
                img = cap(Image.open(d["image_path"]).convert("RGB"))
            except Exception as e:
                print(f"[rollout] skip {d['image_id']}: {e}", flush=True)
                continue
            text = QP.format(Question=d["question"])
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text},
            ]}]
            prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            reqs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
            metas.append(d)
        t = time.time()
        outs = llm.generate(reqs, sp)
        with open(args.out, "a") as f:
            for d, o in zip(metas, outs):
                f.write(json.dumps({
                    "image_id": d["image_id"], "image_path": d["image_path"],
                    "product": d.get("product"), "is_anomaly": d.get("is_anomaly"),
                    "gt_label": d.get("gt_label"), "question": d["question"],
                    "gt_answer": d.get("answer"),
                    "completions": [c.text for c in o.outputs],
                }) + "\n")
        n_done = ci + len(chunk)
        rate = (time.time() - t_all) / max(n_done, 1)
        print(f"[rollout] {n_done}/{len(todo)} | chunk {time.time()-t:.0f}s | "
              f"{rate:.2f}s/prompt | ETA {(len(todo)-n_done)*rate/3600:.1f}h", flush=True)
    print(f"[rollout] DONE in {(time.time()-t_all)/3600:.2f}h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
