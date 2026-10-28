"""vLLM-backed drop-in for evaluate_qwen25vl_7b_trainprompt.py (LLaVA-OV protocol).

Reuses the canonical eval module for data loading, GT extraction, prompt building,
tag parsing and metrics; ONLY generation is swapped (HF model.generate -> vLLM
engine on one GPU). Protocol identity with the watcher evals:
EVAL_MAX_IMAGE_PIXELS=262144, greedy decoding, max 1024 new tokens, same
chat-template rendering (processor from --base-model), same output JSON schema.

Usage:
  evaluate_vllm_llava.py --checkpoint <dir> [--base-model ...] \
      (--ds-mvtec-only | --visa-only) [--yesno-user] --output out.json
Smoke: PROBE_SAMPLES=40 env truncates the (seed-42 deterministic) subset, so the
first N samples match the first N entries of an existing full HF eval JSON.
"""
import argparse
import importlib.util
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--base-model", default="llava-hf/llava-onevision-qwen2-7b-si-hf")
    ap.add_argument("--ds-mvtec-only", action="store_true")
    ap.add_argument("--visa-only", action="store_true")
    ap.add_argument("--yesno-user", action="store_true")
    ap.add_argument("--realiad-4k", action="store_true", help="score the 4,236-image held-out Real-IAD split instead of DS-MVTec / VisA")
    ap.add_argument("--out-realiad", default=None)
    ap.add_argument("--num-samples", type=int, default=0, help="0 = the whole split; a small number for a smoke test")
    ap.add_argument("--output", default=None,
                    help="Output JSON (single-dataset mode)")
    ap.add_argument("--out-ds", default=None,
                    help="Run DS-MVTec and write here (combinable with --out-visa: "
                         "both datasets share one engine load)")
    ap.add_argument("--out-visa", default=None,
                    help="Run VisA and write here")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("WORK_DIR", "."), "hf_cache"))
    os.environ["EVAL_MAX_IMAGE_PIXELS"] = "262144"

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "evalmod", os.path.join(here, "evaluate_qwen25vl_7b_trainprompt.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    model_path = args.checkpoint or args.base_model
    t0 = time.time()
    engine = LLM(model=model_path, dtype="bfloat16",
                 gpu_memory_utilization=args.gpu_mem_util,
                 max_model_len=12288, limit_mm_per_prompt={"image": 1})
    print(f"[vllm-eval] engine load {time.time()-t0:.0f}s for {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model)
    sp = SamplingParams(temperature=0.0, max_tokens=1024)

    def load_model_stub(checkpoint, base_model, load_in_4bit=False):
        return None, processor

    def run_inference_batch_vllm(model, proc, batch_items,
                                 no_system_prompt=False, grpo_eval=False,
                                 bare_question=False, iadr1_native=False,
                                 yesno_user=False):
        assert not (no_system_prompt or grpo_eval or bare_question or iadr1_native), \
            "prompt mode not wired for vLLM eval"
        reqs = []
        for img, product_name in batch_items:
            prompt = m.make_train_prompt(product_name)
            if yesno_user:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"Please answer by yes or no\n{prompt}"},
                ]}]
            else:
                messages = [
                    {"role": "system", "content": "Please answer by yes or no"},
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"\n{prompt}"},
                    ]},
                ]
            text = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
            reqs.append({"prompt": text, "multi_modal_data": {"image": img}})
        outs = engine.generate(reqs, sp)
        return [o.outputs[0].text for o in outs]

    m.load_model = load_model_stub
    m.run_inference_batch = run_inference_batch_vllm

    def run_one(ds_only, visa_only, out):
        t = time.time()
        m.evaluate(
            checkpoint=args.checkpoint,
            base_model=args.base_model,
            num_samples=0,
            output_file=out,
            realiad_only=False,
            ds_mvtec_only=ds_only,
            visa_only=visa_only,
            batch_size=args.batch_size,
            yesno_user=args.yesno_user,
        )
        print(f"[vllm-eval] {'DS-MVTec' if ds_only else 'VisA'} done in "
              f"{time.time()-t:.0f}s -> {out}", flush=True)

    if args.realiad_4k:
        if args.num_samples and args.num_samples > 0:
            # the held-out loader ignores num_samples; truncate its result for smoke tests
            _orig = m.load_balanced_test_data
            def _trunc(*a, **k):
                data = _orig(*a, **k)
                return data[: args.num_samples]
            m.load_balanced_test_data = _trunc
        out = args.out_realiad or os.path.join(args.checkpoint, "eval_realiad4k_full_trainprompt_vllm.json")
        t = time.time()
        m.evaluate(checkpoint=args.checkpoint, base_model=args.base_model, num_samples=args.num_samples, output_file=out,
                   realiad_only=False, ds_mvtec_only=False, visa_only=False, batch_size=args.batch_size,
                   yesno_user=args.yesno_user, realiad_4k=True)
        print(f"[vllm-eval] held-out Real-IAD done in {time.time()-t:.0f}s -> {out}", flush=True)
        return
    if args.out_ds or args.out_visa:
        if args.out_ds:
            run_one(True, False, args.out_ds)
        if args.out_visa:
            run_one(False, True, args.out_visa)
    else:
        assert args.output and (args.ds_mvtec_only or args.visa_only)
        run_one(args.ds_mvtec_only, args.visa_only, args.output)


if __name__ == "__main__":
    main()
