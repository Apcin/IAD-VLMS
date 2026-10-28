"""Generate coordinate-grounded CoT for four Anomaly-Instruct datasets with Qwen3-VL."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

import config
from validate_trace import retry_feedback, validate_reasoning


class ReasoningTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    reasoning: str


logger = logging.getLogger(__name__)


def ground_truth_location_value(sample: dict):
    """Return the exact pixel-coordinate value shown to the teacher model."""
    boxes = sample["supervision"]["bboxes"]
    if sample["target_modality"] == "2d":
        if len(boxes) != 1:
            raise ValueError(f"2D anomaly must have exactly one bbox: {sample['id']}")
        return boxes[0]["xyxy_pixel"]
    return [
        {"view_id": item["view_id"], "bbox": item["xyxy_pixel"]}
        for item in boxes
    ]


def setup_logging(shard_id: int) -> logging.Logger:
    path = config.LOG_DIR / f"anomaly_instruct_4_shard_{shard_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Shard {shard_id}] %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def load_manifests(datasets: list[str]) -> list[dict]:
    samples = []
    for dataset in datasets:
        path = config.MANIFEST_DIR / f"{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(f"manifest missing: {path}; run prepare_manifests.py")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        samples.extend(manifest["samples"])
    return samples


def open_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def format_user_prompt(sample: dict, validation_errors: Optional[list[str]] = None) -> str:
    query_layout = ", ".join(
        f"{item['view_id']}={item['width']}x{item['height']}"
        for item in sample["query_images"]
    )
    parts = [
        "Inspect this query sample and return one JSON object with image_id and reasoning.",
        f"Image ID: {sample['id']}",
        f"Object category: {sample['object_category']}",
        f"Query view dimensions (original pixel frames): {query_layout}",
    ]
    image_inventory = teacher_image_items(sample)
    image_mapping = "\n".join(
        f"- Image {index}: {item['role']} (view_id={item['view_id']})"
        for index, item in enumerate(image_inventory, start=1)
    )
    parts.extend(["Image inventory (in exact input order):", image_mapping])
    if sample["is_anomaly"]:
        response_branch = (
            "ANOMALOUS_2D"
            if sample["target_modality"] == "2d"
            else "ANOMALOUS_MULTIVIEW"
        )
        anomaly_type = sample["supervision"]["anomaly_type"]["canonical"]
        location = json.dumps(
            ground_truth_location_value(sample), separators=(",", ":"), ensure_ascii=False
        )
        parts.extend(
            [
                "[INTERNAL — never mention this supervision] Ground Truth: ANOMALOUS",
                f"[INTERNAL — never mention this supervision] Canonical anomaly type: {anomaly_type}",
                (
                    "[INTERNAL teacher hint — never mention that it was supplied] Ground-truth anomaly location "
                    f"(inclusive pixel xyxy in the original query frame): {location}"
                ),
            ]
        )
        parts.append(
            "For anomaly reasoning, focus on each visibly boxed region in the localization-cue "
            "image(s) and compare it with the corresponding same-view normal reference. Use the "
            "original query image(s) only as clean visual context, and do not mention any cue or "
            "reference in the output."
        )
        if sample["target_modality"] != "2d":
            parts.append(
                "For multiview samples, match images by view_id and include every bbox-marked "
                "view exactly once in <location>."
            )
        parts.append(
            f"Required response branch: {response_branch}. Follow that branch's output schema "
            "defined in the system instruction."
        )
    else:
        parts.extend(
            [
                "[INTERNAL — never mention this supervision] Ground Truth: NORMAL",
                "Required response branch: NORMAL. Follow that branch's output schema defined in the system instruction.",
            ]
        )
    if validation_errors:
        parts.append(retry_feedback(validation_errors))
    return "\n".join(parts)


def teacher_image_items(sample: dict) -> list[dict]:
    """Describe every teacher image while preserving the existing input order."""
    items = [
        {"path": item["path"], "role": "original query", "view_id": item["view_id"]}
        for item in sample["query_images"]
    ]
    if sample["is_anomaly"]:
        items.extend(
            {
                "path": item["path"],
                "role": "localization cue",
                "view_id": item["view_id"],
            }
            for item in sample["supervision"]["bbox_prompt_images"]
        )
    items.extend(
        {
            "path": item["path"],
            "role": "same-category normal reference",
            "view_id": item["view_id"],
        }
        for item in sample["normal_reference"]["images"]
    )
    return items


def teacher_image_paths(sample: dict) -> list[str]:
    items = teacher_image_items(sample)
    missing = [
        item["path"]
        for item in items
        if item["role"] == "localization cue" and not Path(item["path"]).exists()
    ]
    if missing:
        raise FileNotFoundError(f"bbox prompt missing: {missing[0]}")
    return [item["path"] for item in items]


def build_request(sample: dict, processor, system_prompt: str, errors=None) -> tuple[dict, dict]:
    paths = teacher_image_paths(sample)
    images = [open_rgb(path) for path in paths]
    user_content = [{"type": "image", "image": image} for image in images]
    user_prompt = format_user_prompt(sample, errors)
    user_content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        {"prompt": prompt, "multi_modal_data": {"image": images}},
        {"prompt": prompt, "plain_user_prompt": user_prompt, "paths": paths, "images": images},
    )


def parse_response(text: str) -> ReasoningTrace:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1)
    return ReasoningTrace.model_validate(json.loads(stripped))


def make_sampling_params(args):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    structured = StructuredOutputsParams(json=ReasoningTrace.model_json_schema())
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        structured_outputs=structured,
    )


def create_engine(args):
    from vllm import LLM

    return LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 12, "video": 0},
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        seed=args.seed,
    )


def save_trace(sample: dict, trace: ReasoningTrace, validation, attempt: int) -> None:
    directory = config.TRACE_DIR / sample["dataset"]
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "image_id": sample["id"],
        "dataset": sample["dataset"],
        "reasoning": trace.reasoning,
        "generator": "Qwen/Qwen3-VL-8B-Instruct",
        "attempt": attempt,
        "teacher_bbox_coordinates_exposed": sample["is_anomaly"],
        "validation": validation.to_dict(),
    }
    (directory / f"trace_{sample['id']}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def append_failure(sample: dict, errors: list[str], raw_output: str, shard_id: int) -> None:
    directory = config.TRACE_DIR / "failed"
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "image_id": sample["id"],
        "dataset": sample["dataset"],
        "errors": errors,
        "raw_output": raw_output,
    }
    with (directory / f"shard_{shard_id}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_debug(sample: dict, metadata: dict, shard_id: int) -> None:
    directory = config.TRACE_DIR / "debug" / f"shard_{shard_id}_{sample['id']}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompt.txt").write_text(metadata["prompt"], encoding="utf-8")
    (directory / "image_order.json").write_text(
        json.dumps(metadata["paths"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for index, image in enumerate(metadata["images"]):
        image.save(directory / f"image_{index:02d}.jpg")


def process_chunk(samples, engine, processor, sampling_params, system_prompt, args, shard_id):
    pending = list(samples)
    feedback, last_outputs = {}, {}
    saved = 0
    for attempt in range(1, args.max_retries + 1):
        if not pending:
            break
        requests, request_samples = [], []
        build_failed = []
        for sample in pending:
            try:
                request, _ = build_request(sample, processor, system_prompt, feedback.get(sample["id"]))
                requests.append(request)
                request_samples.append(sample)
            except Exception as error:
                feedback[sample["id"]] = [str(error)]
                build_failed.append(sample)
        if not requests:
            pending = build_failed
            break

        outputs = engine.generate(requests, sampling_params)
        next_pending = list(build_failed)
        for sample, output in zip(request_samples, outputs):
            raw = output.outputs[0].text
            last_outputs[sample["id"]] = raw
            try:
                trace = parse_response(raw)
                errors = []
                if trace.image_id != sample["id"]:
                    errors.append("generated image_id does not match the requested sample")
                validation = validate_reasoning(trace.reasoning, sample, min_iou=args.min_location_iou)
                errors.extend(validation.errors)
                if errors:
                    feedback[sample["id"]] = errors
                    next_pending.append(sample)
                    continue
                save_trace(sample, trace, validation, attempt)
                saved += 1
            except Exception as error:
                feedback[sample["id"]] = [f"invalid JSON/Pydantic output: {error}"]
                next_pending.append(sample)
        pending = next_pending

    for sample in pending:
        append_failure(
            sample,
            feedback.get(sample["id"], ["unknown validation failure"]),
            last_outputs.get(sample["id"], ""),
            shard_id,
        )
    return saved, len(pending)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Anomaly-Instruct-4 CoT with Qwen3-VL.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(config.DATASETS), default=sorted(config.DATASETS))
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=config.MAX_RETRIES)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--anomaly-only", action="store_true")
    parser.add_argument("--normal-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=config.MAX_MODEL_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=config.GPU_MEMORY_UTILIZATION)
    parser.add_argument("--min-pixels", type=int, default=config.MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=config.MAX_PIXELS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--presence-penalty", type=float, default=config.PRESENCE_PENALTY)
    parser.add_argument("--repetition-penalty", type=float, default=config.REPETITION_PENALTY)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--min-location-iou", type=float, default=config.MIN_LOCATION_IOU)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.total_shards:
        parser.error("--shard-id must be in [0, total-shards)")
    if args.anomaly_only and args.normal_only:
        parser.error("--anomaly-only and --normal-only are mutually exclusive")
    return args


def main() -> None:
    global logger
    args = parse_args()
    logger = setup_logging(args.shard_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    system_prompt = config.PROMPT_PATH.read_text(encoding="utf-8")
    samples = load_manifests(args.datasets)
    if args.anomaly_only:
        samples = [sample for sample in samples if sample["is_anomaly"]]
    if args.normal_only:
        samples = [sample for sample in samples if not sample["is_anomaly"]]
    samples = [sample for index, sample in enumerate(samples) if index % args.total_shards == args.shard_id]
    processed = {
        path.stem[6:]
        for dataset in args.datasets
        for path in (config.TRACE_DIR / dataset).glob("trace_*.json")
    }
    samples = [sample for sample in samples if sample["id"] not in processed]
    if args.max_samples:
        samples = samples[: args.max_samples]
    logger.info("assigned remaining samples: %d", len(samples))
    if not samples:
        return

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    if args.debug:
        for sample in samples:
            _, metadata = build_request(sample, processor, system_prompt)
            save_debug(sample, metadata, args.shard_id)
        logger.info("debug artifacts saved under %s", config.TRACE_DIR / "debug")
        return

    engine = create_engine(args)
    sampling_params = make_sampling_params(args)
    failures = 0
    with tqdm(total=len(samples), desc=f"shard{args.shard_id}") as progress:
        for start in range(0, len(samples), args.batch_size):
            saved, failed = process_chunk(
                samples[start : start + args.batch_size],
                engine,
                processor,
                sampling_params,
                system_prompt,
                args,
                args.shard_id,
            )
            failures += failed
            progress.update(saved + failed)
    logger.info("generation finished; failures=%d", failures)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
