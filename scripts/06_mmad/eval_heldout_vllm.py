"""
vLLM evaluation of a Qwen2.5-VL checkpoint on the MMAD held-out keys of split.json,
for any of the four subsets (DS-MVTec, VisA, GoodsAD, MVTec-LOCO).

Same protocol as Training/evaluate_vllm_qwen25vl.py (which only knows DS-MVTec and VisA):
the canonical harness does GT extraction, prompt building, tag parsing and metrics, only the
data loader is swapped for one that returns the requested subset restricted to the given keys.

Usage:
  python eval_heldout_vllm.py --checkpoint <dir> --subset DS-MVTec --keys test --output out.json
  --keys test  (held-out images, default) | train | all
"""
import argparse
import importlib.util
import json
import os
import random
import time

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 262144
HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.environ.get("MMAD_SPLIT", os.path.join(HERE, "split.json"))
MMAD_JSON = os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/reasoning_traces_gen/data/MMAD/mmad.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--subset", required=True, choices=["DS-MVTec", "VisA", "GoodsAD", "MVTec-LOCO"])
    ap.add_argument("--keys", default="test", choices=["test", "train", "all"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=12288)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/hf_cache")
    os.environ["EVAL_MAX_IMAGE_PIXELS"] = "0"

    spec = importlib.util.spec_from_file_location(
        "evalmod", os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/Training/evaluate_qwen25vl_7b_trainprompt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.DATA_PATHS["realiad_images_root"] = ""   # skip the 150K-image Real-IAD rglob

    split = json.load(open(SPLIT))
    if args.keys == "test":
        keep = set(split["test_keys"])
    elif args.keys == "train":
        keep = set(split["train_keys"])
    else:
        keep = set(split["test_keys"]) | set(split["train_keys"])
    prefix = args.subset + "/"

    def load_subset(*a, **k):
        mmad = json.load(open(MMAD_JSON))
        out = []
        for image_id, entry in mmad.items():
            if image_id.startswith(prefix) and image_id in keep:
                entry["image_id"] = image_id
                entry["product"] = image_id.split("/")[1]
                out.append(entry)
        m.logger.info(f"{args.subset} {args.keys} keys: {len(out)} entries")
        random.seed(m.SEED)
        random.shuffle(out)
        return out

    m.load_balanced_test_data = load_subset

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    model_path = args.checkpoint or args.base_model
    t0 = time.time()
    engine = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
                 max_model_len=args.max_model_len, limit_mm_per_prompt={"image": 1},
                 mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS})
    print(f"[vllm-eval] engine load {time.time()-t0:.0f}s for {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.image_processor.min_pixels = MIN_PIXELS
    processor.image_processor.max_pixels = MAX_PIXELS
    sp = SamplingParams(temperature=0.0, max_tokens=1024)

    def load_model_stub(checkpoint, base_model, load_in_4bit=False):
        return None, processor

    def run_inference_batch_vllm(model, proc, batch_items, no_system_prompt=False, grpo_eval=False,
                                 bare_question=False, iadr1_native=False, yesno_user=False):
        reqs = []
        for img, product_name in batch_items:
            prompt = m.make_train_prompt(product_name)
            messages = [
                {"role": "system", "content": "Please answer by yes or no"},
                {"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": f"\n{prompt}"}]},
            ]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            reqs.append({"prompt": text, "multi_modal_data": {"image": img}})
        outs = engine.generate(reqs, sp)
        return [o.outputs[0].text for o in outs]

    m.load_model = load_model_stub
    m.run_inference_batch = run_inference_batch_vllm

    t = time.time()
    m.evaluate(checkpoint=args.checkpoint, base_model=args.base_model, num_samples=0,
               output_file=args.output, realiad_only=False, ds_mvtec_only=True,
               batch_size=args.batch_size)
    print(f"[vllm-eval] {args.subset} ({args.keys}) done in {time.time()-t:.0f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
