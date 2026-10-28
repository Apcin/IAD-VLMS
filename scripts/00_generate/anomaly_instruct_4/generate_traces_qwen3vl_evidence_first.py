"""Two-stage Qwen3-VL generation with one evidence/trace JSON per dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

import config
import generate_traces_qwen3vl as baseline
from dataset_json_store import merge_records, read_records
from validate_trace import retry_feedback, validate_reasoning


class RegionalObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_id: str
    region: str = Field(min_length=1)
    query_evidence: list[str]
    normal_comparison: str = Field(min_length=1)
    visibility: Literal["clear", "partial", "not_visible"]
    uncertainty: Optional[str]


class VisualEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_observation: str = Field(min_length=1)
    regional_observations: list[RegionalObservation] = Field(min_length=1)


logger = logging.getLogger(__name__)


def collection_path(stage: str, dataset: str) -> Path:
    if stage == "evidence":
        return config.EVIDENCE_DIR / f"evidence_{dataset}.json"
    return config.EVIDENCE_FIRST_TRACE_DIR / f"traces_{dataset}.json"


def image_description(sample: dict) -> str:
    dimensions = ", ".join(
        f"{item['view_id']}={item['width']}x{item['height']}"
        for item in sample["query_images"]
    )
    inventory = "\n".join(
        f"- Image {index}: {item['role']} (view_id={item['view_id']})"
        for index, item in enumerate(baseline.teacher_image_items(sample), start=1)
    )
    return f"Query view dimensions: {dimensions}\nImage inventory (exact input order):\n{inventory}"


def format_stage1_user_prompt(sample: dict, errors=None) -> str:
    # Only explicitly allowed fields reach the model. Never serialize the sample,
    # its ID, paths, source descriptions, labels, GT coordinates, or anomaly type.
    items = baseline.teacher_image_items(sample)
    if any(not re.fullmatch(r"query|view_\d+", item["view_id"]) for item in items):
        raise ValueError("Stage 1 requires neutral query/view_N view IDs")
    parts = [
        "Extract observable visual evidence and return the VisualEvidence JSON object.",
        f"Object category: {sample['object_category']}",
        image_description(sample),
        "Use original queries for visual facts, cues only for attention, and normal references for cautious comparison.",
    ]
    if errors:
        # Parser/IO errors may contain paths or internal identifiers. Do not relay
        # their raw text to this type-blind stage, even on a retry.
        parts.append("The previous response was invalid. Return only the required JSON fields and use supplied query view IDs.")
    return "\n".join(parts)


def format_stage2_user_prompt(sample: dict, evidence: VisualEvidence, errors=None) -> str:
    parts = [
        "Re-inspect the images using the candidate evidence and return image_id and reasoning.",
        f"Image ID: {sample['id']}",
        f"Object category: {sample['object_category']}",
        image_description(sample),
        "Candidate visual evidence (observations to check, not ground truth):",
        evidence.model_dump_json(),
    ]
    if sample["is_anomaly"]:
        branch = "ANOMALOUS_2D" if sample["target_modality"] == "2d" else "ANOMALOUS_MULTIVIEW"
        location = json.dumps(baseline.ground_truth_location_value(sample), separators=(",", ":"))
        parts.extend([
            "[INTERNAL supervision; never disclose] Ground Truth: ANOMALOUS",
            f"[INTERNAL supervision; never disclose] Canonical anomaly type: {sample['supervision']['anomaly_type']['canonical']}",
            f"[INTERNAL supervision; never disclose] Ground-truth location (inclusive original pixel xyxy): {location}",
        ])
    else:
        branch = "NORMAL"
        parts.append("[INTERNAL supervision; never disclose] Ground Truth: NORMAL")
    parts.append(f"Required response branch: {branch}. Follow the system instruction's output format.")
    if errors:
        parts.append(retry_feedback(errors))
    return "\n".join(parts)


def build_request(sample, processor, system_prompt, stage, evidence=None, errors=None):
    text = (
        format_stage1_user_prompt(sample, errors)
        if stage == "evidence"
        else format_stage2_user_prompt(sample, evidence, errors)
    )
    paths = baseline.teacher_image_paths(sample)
    images = [baseline.open_rgb(path) for path in paths]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": text},
        ]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"image": images}}, {"prompt": prompt, "paths": paths}


def parse_evidence(text: str, sample: dict) -> VisualEvidence:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    evidence = VisualEvidence.model_validate_json(fenced.group(1) if fenced else stripped)
    views = {item["view_id"] for item in sample["query_images"]}
    if any(item.view_id not in views for item in evidence.regional_observations):
        raise ValueError("evidence contains an unknown query view_id")
    return evidence


def load_sample_evidence(sample: dict, records: dict) -> VisualEvidence:
    record = records.get(sample["id"])
    if not record:
        raise ValueError("missing Stage 1 evidence")
    if record.get("image_id") != sample["id"] or record.get("dataset") != sample["dataset"]:
        raise ValueError("evidence identity mismatch")
    return parse_evidence(json.dumps(record["evidence"], ensure_ascii=False), sample)


def has_evidence(sample: dict, records: dict) -> bool:
    try:
        load_sample_evidence(sample, records)
        return True
    except (ValueError, KeyError, TypeError):
        return False


def has_trace(sample: dict, records: dict, min_iou: float) -> bool:
    record = records.get(sample["id"], {})
    return (
        record.get("image_id") == sample["id"]
        and record.get("dataset") == sample["dataset"]
        and isinstance(record.get("reasoning"), str)
        and validate_reasoning(record["reasoning"], sample, min_iou=min_iou).valid
    )


def make_sampling_params(args, stage):
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:
        raise RuntimeError(
            "A Qwen3-VL-capable vLLM with StructuredOutputsParams is required; "
            "use ../realiad_v4/requirements-qwen3vl.txt (vllm>=0.11.1)."
        ) from error

    schema = VisualEvidence if stage == "evidence" else baseline.ReasoningTrace
    return SamplingParams(
        temperature=args.evidence_temperature if stage == "evidence" else args.temperature,
        max_tokens=args.evidence_max_new_tokens if stage == "evidence" else args.max_new_tokens,
        top_p=args.top_p, top_k=args.top_k,
        presence_penalty=args.presence_penalty, repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        structured_outputs=StructuredOutputsParams(json=schema.model_json_schema()),
    )


def append_failure(sample, stage, errors, raw_output, shard_id):
    directory = config.EVIDENCE_FIRST_TRACE_DIR / "failed"
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "image_id": sample["id"], "dataset": sample["dataset"], "stage": stage,
        "errors": errors, "raw_output": raw_output,
    }
    with (directory / f"{stage}_shard_{shard_id}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_chunk(samples, engine, processor, sampling_params, system_prompt, args, stage, evidence_records):
    pending, feedback, last_outputs = list(samples), {}, {}
    saved = 0
    for attempt in range(1, args.max_retries + 1):
        if not pending:
            break
        requests, request_samples, next_pending = [], [], []
        for sample in pending:
            try:
                evidence = load_sample_evidence(sample, evidence_records) if stage == "cot" else None
                request, _ = build_request(
                    sample, processor, system_prompt, stage, evidence, feedback.get(sample["id"]),
                )
                requests.append(request)
                request_samples.append(sample)
            except Exception as error:
                feedback[sample["id"]] = [f"request construction failed: {error}"]
                next_pending.append(sample)
        if not requests:
            pending = next_pending
            break
        # As in the baseline, engine-level failures stop the run. Completed batches
        # are already on disk and the same command resumes their remaining work.
        outputs = engine.generate(requests, sampling_params)
        if len(outputs) != len(request_samples):
            raise RuntimeError("vLLM output count differs from request count")
        records = {}
        for sample, output in zip(request_samples, outputs):
            image_id = sample["id"]
            raw = output.outputs[0].text if output.outputs else ""
            last_outputs[image_id] = raw
            try:
                record = {
                    "image_id": image_id, "dataset": sample["dataset"],
                    "generator": args.model, "attempt": attempt,
                    "generation_method": "evidence_first", "stage": stage,
                }
                if stage == "evidence":
                    record["evidence"] = parse_evidence(raw, sample).model_dump()
                else:
                    trace = baseline.parse_response(raw)
                    if trace.image_id != image_id:
                        raise ValueError("generated image_id does not match the requested sample")
                    validation = validate_reasoning(trace.reasoning, sample, min_iou=args.min_location_iou)
                    if not validation.valid:
                        feedback[image_id] = validation.errors
                        next_pending.append(sample)
                        continue
                    record.update({
                        "reasoning": trace.reasoning, "validation": validation.to_dict(),
                        "teacher_bbox_coordinates_exposed": sample["is_anomaly"],
                        "evidence_file": str(collection_path("evidence", sample["dataset"])),
                    })
                records[image_id] = record
            except Exception as error:
                feedback[image_id] = [f"invalid JSON/Pydantic output: {error}"]
                next_pending.append(sample)
        # One dataset per chunk. Persist every accepted batch before any retry or
        # Stage 2 request; IO failures propagate rather than discard cached work.
        merge_records(collection_path(stage, samples[0]["dataset"]), samples[0]["dataset"], records)
        saved += len(records)
        pending = next_pending
    for sample in pending:
        append_failure(sample, stage, feedback[sample["id"]], last_outputs.get(sample["id"], ""), args.shard_id)
    return saved, len(pending)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--stage", choices=["all", "evidence", "cot"], default="all")
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=config.MAX_RETRIES)
    parser.add_argument("--max-samples", type=int, default=0, help="Remaining samples in this shard across selected datasets; 0=all")
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument("--anomaly-only", action="store_true")
    filters.add_argument("--normal-only", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Save model prompts/image paths without inference; cot requires cached evidence")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=config.MAX_MODEL_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--evidence-max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=config.GPU_MEMORY_UTILIZATION)
    parser.add_argument("--min-pixels", type=int, default=config.MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=config.MAX_PIXELS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--evidence-temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--presence-penalty", type=float, default=config.PRESENCE_PENALTY)
    parser.add_argument("--repetition-penalty", type=float, default=config.REPETITION_PENALTY)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--min-location-iou", type=float, default=config.MIN_LOCATION_IOU)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_id < args.total_shards:
        parser.error("--shard-id must be in [0, total-shards)")
    if min(args.batch_size, args.max_retries, args.max_new_tokens, args.evidence_max_new_tokens) < 1 or args.max_samples < 0:
        parser.error("batch size, retries, and token limits must be positive; max-samples must be nonnegative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s [Shard {args.shard_id}] %(levelname)s %(message)s",
        handlers=[logging.FileHandler(config.LOG_DIR / f"evidence_first_shard_{args.shard_id}.log"), logging.StreamHandler()],
        force=True,
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    samples = baseline.load_manifests(args.datasets)
    if args.anomaly_only:
        samples = [sample for sample in samples if sample["is_anomaly"]]
    if args.normal_only:
        samples = [sample for sample in samples if not sample["is_anomaly"]]
    samples = [sample for index, sample in enumerate(samples) if index % args.total_shards == args.shard_id]
    # Shard the stable manifest order BEFORE removing completed samples.
    resume_stage = "evidence" if args.stage == "evidence" else "cot"
    cached = {dataset: read_records(collection_path(resume_stage, dataset), dataset) for dataset in args.datasets}
    samples = [sample for sample in samples if not (
        has_evidence(sample, cached[sample["dataset"]]) if resume_stage == "evidence"
        else has_trace(sample, cached[sample["dataset"]], args.min_location_iou)
    )]
    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info("assigned remaining samples: %d; stage=%s", len(samples), args.stage)
    processor, engine, failures = None, None, 0
    for stage in (["evidence", "cot"] if args.stage == "all" else [args.stage]):
        system_prompt = (config.STAGE1_PROMPT_PATH if stage == "evidence" else config.STAGE2_PROMPT_PATH).read_text(encoding="utf-8")
        sampling_params = None
        for dataset in args.datasets:
            # Stage 2 deliberately reads B.json from disk, including after Stage 1
            # in the same process. This is also the independent cot-only path.
            evidence_records = read_records(collection_path("evidence", dataset), dataset)
            pending = []
            for sample in samples:
                if sample["dataset"] != dataset:
                    continue
                if stage == "evidence" and has_evidence(sample, evidence_records):
                    continue
                if stage == "cot" and not has_evidence(sample, evidence_records):
                    logger.warning("missing/invalid evidence: %s", sample["id"])
                    if not args.debug:
                        append_failure(sample, stage, ["missing/invalid Stage 1 evidence; run --stage evidence or --stage all"], "", args.shard_id)
                        failures += 1
                    continue
                pending.append(sample)
            if not pending:
                continue
            if processor is None:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(args.model)
            if args.debug:
                records = {}
                for sample in pending:
                    evidence = load_sample_evidence(sample, evidence_records) if stage == "cot" else None
                    _, metadata = build_request(sample, processor, system_prompt, stage, evidence)
                    records[sample["id"]] = {"image_id": sample["id"], "dataset": dataset, **metadata}
                path = config.EVIDENCE_FIRST_TRACE_DIR / "debug" / f"{stage}_{dataset}_shard_{args.shard_id}.json"
                merge_records(path, dataset, records)
                logger.info("debug requests saved: %s", path)
                continue
            if sampling_params is None:
                sampling_params = make_sampling_params(args, stage)
            if engine is None:
                engine = baseline.create_engine(args)
            with tqdm(total=len(pending), desc=f"{stage}/{dataset}/shard{args.shard_id}") as progress:
                for start in range(0, len(pending), args.batch_size):
                    saved, failed = process_chunk(
                        pending[start:start + args.batch_size], engine, processor, sampling_params,
                        system_prompt, args, stage, evidence_records,
                    )
                    failures += failed
                    progress.update(saved + failed)
    logger.info("finished; stage failures=%d", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
