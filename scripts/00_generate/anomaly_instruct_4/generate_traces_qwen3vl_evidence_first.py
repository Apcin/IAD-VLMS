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

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tqdm import tqdm

import config
import generate_traces_qwen3vl as baseline
from dataset_json_store import merge_records, read_records
from validate_trace import (
    has_repetition,
    retry_feedback,
    validate_evidence_first_reasoning,
)


class RegionalObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(description="Supplied query view identifier for this region.")
    region: str = Field(
        min_length=1,
        description="Prefer a short spatial locator; place detailed findings in query_evidence when possible.",
    )
    query_evidence: list[str] = Field(description="Distinct visible facts covering the relevant local morphology, surface condition, boundaries and continuity; preserve useful details and uncertainty.")
    normal_comparison: str = Field(
        min_length=1,
        description="Reliable visual similarities or differences from the corresponding normal structure.",
    )
    visibility: Literal["clear", "partial", "not_visible"]
    uncertainty: Optional[str]


class VisualEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_observation: str = Field(
        min_length=1,
        description="Overall visible form, arrangement and appearance, with enough context to identify the inspected structures; detailed local findings belong in regional_observations.",
    )
    regional_observations: list[RegionalObservation] = Field(
        min_length=1,
        description="Separate entries for spatially distinct inspected regions; multiple entries may share the same view_id.",
    )


class EvidenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_index: int = Field(ge=0)
    decision: Literal["keep", "revise", "uncertain"]
    verified_observation: str = Field(
        description="Verified local observations, corrections or additions, relevant comparisons and uncertainty. Cover the facts needed to explain the inspection judgment; do not compress away diagnostic details or infer morphology from a label.",
    )


class EvidenceAnchoredTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    evidence_review: list[EvidenceReview] = Field(min_length=1)
    inspection_text: str = Field(
        description="Evidence-based inspection assessment of the query: relevant overall condition, specific visible evidence, why it supports the judgment, and a conclusion with appropriate uncertainty. Aim for 80-120 English words; 130 is the hard maximum, not a requirement to pad. Omit auxiliary sources, coordinate commentary and XML tags.",
    )


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
        f"- Picture {index}: {item['role']} (view_id={item['view_id']})"
        for index, item in enumerate(baseline.teacher_image_items(sample), start=1)
    )
    return f"Query view dimensions: {dimensions}\nPicture inventory (exact input order):\n{inventory}"


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
        (
            "For each view, use its localization cue only to locate the target region in the clean "
            "original query when a cue is supplied. Inspect visual facts in the clean original query, "
            "prioritizing that region when present, then compare the inspected query region or "
            "structure with its counterpart in the same-view normal reference. Cover all relevant visible "
            "features, similarities, differences and uncertainty needed for a later inspection assessment; "
            "do not reduce the evidence to a generic mark or omit details merely to shorten the response."
        ),
    ]
    if errors:
        # Parser/IO errors may contain paths or internal identifiers. Do not relay
        # their raw text to this type-blind stage, even on a retry.
        parts.append("Return a complete valid JSON object using the required fields and supplied query view IDs. Preserve distinct visual details and uncertainty; remove repetition rather than useful evidence.")
    return "\n".join(parts)


def format_stage2_user_prompt(sample: dict, evidence: VisualEvidence, errors=None) -> str:
    parts = [
        "Return image_id, evidence_review, and inspection_text in that order. Recheck the images first, then write the final inspection from verified observations.",
        f"Image ID: {sample['id']}",
        f"Object category: {sample['object_category']}",
        image_description(sample),
        "Candidate visual evidence (observations to check, not ground truth):",
        evidence.model_dump_json(),
        "Region indices for evidence_review (zero-based regional_observations order):",
        json.dumps([{"region_index": index, "view_id": item.view_id, "region": item.region}
                    for index, item in enumerate(evidence.regional_observations)], ensure_ascii=False),
        "Recheck each region in its clean query. Keep supported observations, revise unsupported interpretations, "
        "or mark uncertain when the images cannot decide. Add relevant visible omissions in verified_observation. "
        "An accurate record may use keep for every region; do not invent corrections. "
        "Internal reference comparisons are allowed in evidence_review. Compose inspection_text from verified observations, "
        "retaining uncertainty but omitting auxiliary input sources. Explain the relevant overall condition, "
        "specific visible evidence, its inspection significance and how it supports the conclusion; do not stop "
        "at a feature list. Aim for 80-120 English words with a hard maximum of 130. This is a soft length "
        "target, so completeness matters more than padding. Do not generate XML tags, coordinates, types or answers as separate fields.",
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
    parts.append(f"Required response branch: {branch}. Supervision does not establish visible morphology; the program assembles the final tags from the supplied supervision.")
    if not sample["is_anomaly"]:
        parts.append("Explain why the inspected structures support a normal judgment and why relevant appearance variations do not establish a defect. Relate visible condition to expected structure or function where supported; keep unobserved properties uncertain.")
    else:
        parts.append("Explain how the verified morphology departs from expected structure or surface condition and supports the anomalous judgment. Connect the finding to the defect interpretation without inventing morphology or treating the label as visual proof; retain uncertainty about depth, cause or unseen effects.")
    if sample["target_modality"] != "2d":
        parts.append("Keep each finding tied to the view where it is visible; do not transfer findings across views or infer hidden 3D structure. Do not put view identifiers in inspection_text.")
    if errors:
        parts.append(retry_feedback(errors).replace("<think>", "inspection_text").replace("<location>", "the program-generated location tag"))
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
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
    )
    return {"prompt": prompt, "multi_modal_data": {"image": images}}, {"prompt": prompt, "paths": paths}


def parse_evidence(text: str, sample: dict) -> VisualEvidence:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    evidence = VisualEvidence.model_validate_json(fenced.group(1) if fenced else stripped)
    views = {item["view_id"] for item in sample["query_images"]}
    if any(item.view_id not in views for item in evidence.regional_observations):
        raise ValueError("evidence contains an unknown query view_id")
    descriptions = [evidence.object_observation]
    for item in evidence.regional_observations:
        descriptions.extend([item.region, *item.query_evidence, item.normal_comparison, item.uncertainty or ""])
    if any(has_repetition(text) for text in descriptions):
        raise ValueError("repetitive output in Stage 1 evidence")
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
        and isinstance(record.get("inspection_text"), str)
        and bool(record.get("evidence_review"))
        and record["reasoning"] == assemble_reasoning(record["inspection_text"], sample)
        and validate_evidence_first_reasoning(record["reasoning"], sample, min_iou=min_iou).valid
    )


def parse_stage2_response(text: str, evidence: VisualEvidence) -> EvidenceAnchoredTrace:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    trace = EvidenceAnchoredTrace.model_validate_json(fenced.group(1) if fenced else stripped)
    indices = [item.region_index for item in trace.evidence_review]
    if sorted(indices) != list(range(len(evidence.regional_observations))):
        raise ValueError("evidence_review must cover every region_index exactly once")
    if any(not item.verified_observation.strip() for item in trace.evidence_review):
        raise ValueError("evidence_review observations must not be empty")
    if any(has_repetition(item.verified_observation) for item in trace.evidence_review):
        raise ValueError("repetitive output in evidence_review")
    return trace


def assemble_reasoning(inspection_text: str, sample: dict) -> str:
    """Format verified prose and existing GT supervision; this is not a prediction."""
    reasoning = f"<think>{inspection_text.strip()}</think>"
    if not sample["is_anomaly"]:
        return reasoning + "<answer>No</answer>"
    location = json.dumps(baseline.ground_truth_location_value(sample), separators=(",", ":"))
    return (reasoning + f"<location>{location}</location>"
            f"<type>{sample['supervision']['anomaly_type']['canonical']}</type><answer>Yes</answer>")


def make_sampling_params(args, stage):
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:
        raise RuntimeError(
            "A Qwen3-VL-capable vLLM with StructuredOutputsParams is required; "
            "use ../realiad_v4/requirements-qwen3vl.txt (vllm>=0.11.1)."
        ) from error

    schema = (VisualEvidence if stage == "evidence" else EvidenceAnchoredTrace).model_json_schema()
    # This installed decoder turns string length keywords into regexes that reject
    # valid JSON escapes. Keep ordinary JSON string grammar; Pydantic still checks
    # nonempty evidence after parsing. Array sizes/required fields are unchanged.
    def plain_strings(value):
        if isinstance(value, dict):
            if value.get("type") == "string":
                value.pop("minLength", None)
                value.pop("maxLength", None)
            for child in value.values():
                plain_strings(child)
        elif isinstance(value, list):
            for child in value:
                plain_strings(child)
    plain_strings(schema)
    return SamplingParams(
        temperature=args.evidence_temperature if stage == "evidence" else args.temperature,
        max_tokens=args.evidence_max_new_tokens if stage == "evidence" else args.max_new_tokens,
        top_p=args.top_p, top_k=args.top_k,
        presence_penalty=args.presence_penalty, repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        structured_outputs=StructuredOutputsParams(json=schema),
    )


def append_failure(sample, stage, errors, raw_output, shard_id, generation_attempts=None, failure_log=None):
    path = (Path(failure_log).expanduser() if failure_log is not None
            else config.EVIDENCE_FIRST_TRACE_DIR / "failed" / f"{stage}_shard_{shard_id}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "image_id": sample["id"], "dataset": sample["dataset"], "stage": stage,
        "errors": errors, "raw_output": raw_output,
        "generation_attempts": generation_attempts or [],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_chunk(samples, engine, processor, sampling_params, system_prompt, args, stage, evidence_records):
    pending, feedback, last_outputs = list(samples), {}, {}
    histories = {}
    chunk_params = sampling_params
    # Use the configured budget for both full regional review and final assessment.
    # Repetition/truncation guards handle failures without shrinking normal prose.
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
        outputs = engine.generate(requests, chunk_params)
        if len(outputs) != len(request_samples):
            raise RuntimeError("vLLM output count differs from request count")
        records = {}
        for sample, output in zip(request_samples, outputs):
            image_id = sample["id"]
            completion = output.outputs[0] if output.outputs else None
            raw = completion.text if completion else ""
            last_outputs[image_id] = raw
            metadata = {
                "attempt": attempt,
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
                "output_tokens": len(getattr(completion, "token_ids", []) or []),
                "token_budget": getattr(chunk_params, "max_tokens", None),
            }
            histories.setdefault(image_id, []).append(metadata)
            try:
                if metadata["finish_reason"] == "length":
                    raise ValueError("truncated output: generation exhausted its token budget")
                record = {
                    "image_id": image_id, "dataset": sample["dataset"],
                    "generator": args.model, "attempt": attempt,
                    "generation_method": "evidence_first", "stage": stage,
                    "generation_attempts": histories[image_id],
                    "sampling": {
                        "temperature": args.evidence_temperature if stage == "evidence" else args.temperature,
                        "presence_penalty": args.presence_penalty,
                        "repetition_penalty": args.repetition_penalty,
                        "seed": args.seed,
                    },
                }
                if stage == "evidence":
                    record["evidence"] = parse_evidence(raw, sample).model_dump()
                else:
                    trace = parse_stage2_response(raw, load_sample_evidence(sample, evidence_records))
                    if trace.image_id != image_id:
                        raise ValueError("generated image_id does not match the requested sample")
                    reasoning = assemble_reasoning(trace.inspection_text, sample)
                    validation = validate_evidence_first_reasoning(reasoning, sample, min_iou=args.min_location_iou)
                    if not validation.valid:
                        feedback[image_id] = validation.errors
                        metadata["errors"] = validation.errors
                        next_pending.append(sample)
                        continue
                    record.update({
                        "inspection_text": trace.inspection_text.strip(),
                        "reasoning": reasoning, "validation": validation.to_dict(),
                        "quality_warnings": (["inspection_text outside the soft 80-120 word target; review explanatory completeness"]
                                             if not 80 <= validation.word_count <= 120 else []),
                        "evidence_review": [item.model_dump() for item in trace.evidence_review],
                        "supervision_fields_source": "manifest_gt",
                        "teacher_bbox_coordinates_exposed": sample["is_anomaly"],
                        "evidence_file": str(collection_path("evidence", sample["dataset"])),
                    })
                records[image_id] = record
                metadata["errors"] = []
            except Exception as error:
                if str(error).startswith(("truncated output", "repetitive output", "evidence_review")):
                    feedback[image_id] = [str(error)]
                elif isinstance(error, ValidationError) and any(item["type"] == "json_invalid" for item in error.errors()):
                    feedback[image_id] = ["repetitive output" if has_repetition(raw) else "invalid JSON output; return a complete JSON object with escaped string characters"]
                elif (stage == "cot" and isinstance(error, ValidationError)
                        and any(item["loc"] and item["loc"][0] == "evidence_review" for item in error.errors())):
                    feedback[image_id] = ["evidence_review must follow its required schema"]
                else:
                    feedback[image_id] = [f"invalid JSON/Pydantic output: {error}"]
                metadata["errors"] = feedback[image_id]
                next_pending.append(sample)
        # One dataset per chunk. Persist every accepted batch before any retry or
        # Stage 2 request; IO failures propagate rather than discard cached work.
        merge_records(collection_path(stage, samples[0]["dataset"]), samples[0]["dataset"], records)
        saved += len(records)
        pending = next_pending
    for sample in pending:
        append_failure(sample, stage, feedback[sample["id"]], last_outputs.get(sample["id"], ""), args.shard_id,
                       histories.get(sample["id"], []), failure_log=args.failure_log)
    return saved, len(pending)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    parser.add_argument("--stage", choices=["all", "evidence", "cot"], default="all")
    parser.add_argument(
        "--failure-log", type=Path,
        help="Append failures to this JSONL path (both stages for --stage all); default: failed/<stage>_shard_<id>.jsonl under the trace directory",
    )
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
    parser.add_argument("--max-new-tokens", type=int, default=max(config.MAX_NEW_TOKENS, 2048),
                        help="Stage 2 token budget for full evidence review plus final assessment; default at least 2048")
    parser.add_argument("--evidence-max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=config.GPU_MEMORY_UTILIZATION)
    parser.add_argument(
        "--mm-processor-cache-gb", type=float, default=0.0,
        help="vLLM multimodal processor cache size; 0 avoids mirrored-LRU desynchronization on long evidence-first runs",
    )
    parser.add_argument("--min-pixels", type=int, default=config.MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=config.MAX_PIXELS)
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("QWEN3VL_TEMPERATURE", "0.2")))
    parser.add_argument("--evidence-temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--presence-penalty", type=float, default=float(os.environ.get("QWEN3VL_PRESENCE_PENALTY", "0")))
    parser.add_argument("--repetition-penalty", type=float, default=float(os.environ.get("QWEN3VL_REPETITION_PENALTY", "1.05")))
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--min-location-iou", type=float, default=config.MIN_LOCATION_IOU)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_id < args.total_shards:
        parser.error("--shard-id must be in [0, total-shards)")
    if min(args.batch_size, args.max_retries, args.max_new_tokens, args.evidence_max_new_tokens) < 1 or args.max_samples < 0:
        parser.error("batch size, retries, and token limits must be positive; max-samples must be nonnegative")
    if args.mm_processor_cache_gb < 0:
        parser.error("--mm-processor-cache-gb must be nonnegative")
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
    logger.info("sampling: cot_temperature=%s evidence_temperature=%s presence_penalty=%s repetition_penalty=%s cot_max_tokens=%s",
                args.temperature, args.evidence_temperature, args.presence_penalty, args.repetition_penalty, args.max_new_tokens)
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
                        append_failure(sample, stage, ["missing/invalid Stage 1 evidence; run --stage evidence or --stage all"], "", args.shard_id,
                                       failure_log=args.failure_log)
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
