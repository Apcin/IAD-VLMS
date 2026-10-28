"""vLLM-backed drop-in for evaluate_qwen25vl_7b_trainprompt.py (Qwen3-VL protocol).

Same idea as evaluate_vllm_llava.py: the canonical eval module still does the data
loading, GT extraction, prompt building, tag parsing and metrics, and ONLY generation
is swapped from HF model.generate to a vLLM engine on one GPU.

Protocol parity with the HF path, which is what makes the numbers comparable:
  * processor from --base-model, min_pixels 256*28*28 and max_pixels 262144, the
    same two lines the HF harness sets at evaluate_qwen25vl_7b_trainprompt.py:551.
    vLLM re-runs the processor itself, so the caps are handed to it through
    mm_processor_kwargs instead of being set on our own processor object.
  * greedy decoding, max 1024 new tokens.
  * the same system turn "Please answer by yes or no", which Qwen does read
    (unlike LLaVA-OneVision, whose template drops it).
  * identical output JSON schema, so the BA helper and the thesis tables do not care.

Usage:
  evaluate_vllm_qwen3vl.py --checkpoint <dir> [--base-model Qwen/Qwen3-VL-8B-Instruct] \
      (--ds-mvtec-only | --visa-only) --output out.json
  evaluate_vllm_qwen3vl.py --checkpoint <dir> --out-ds a.json --out-visa b.json
PROBE_SAMPLES=N truncates the seed-42 subset, so the first N samples line up with
the first N entries of an existing HF eval JSON for a like-for-like engine check.
"""
import argparse
import importlib.util
import os
import time

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 262144


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--ds-mvtec-only", action="store_true")
    ap.add_argument("--visa-only", action="store_true")
    ap.add_argument("--yesno-user", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--out-ds", default=None)
    ap.add_argument("--out-visa", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=12288)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", (os.environ.get('WORK_DIR','/bulk/aacudad/reasoning_traces')+"/hf_cache"))
    # Qwen path: leave the pre-resize cap OFF, the processor enforces max_pixels.
    os.environ["EVAL_MAX_IMAGE_PIXELS"] = "0"

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
                 max_model_len=args.max_model_len,
                 limit_mm_per_prompt={"image": 1},
                 mm_processor_kwargs={"min_pixels": MIN_PIXELS,
                                      "max_pixels": MAX_PIXELS})
    print(f"[vllm-eval] engine load {time.time()-t0:.0f}s for {model_path}", flush=True)

    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.image_processor.min_pixels = MIN_PIXELS
    processor.image_processor.max_pixels = MAX_PIXELS
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
