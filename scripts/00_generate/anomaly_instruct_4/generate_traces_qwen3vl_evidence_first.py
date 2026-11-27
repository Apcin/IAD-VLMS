"""Two-stage Qwen3-VL generation with one evidence/trace JSON per dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tqdm import tqdm

import config
import generate_traces_qwen3vl as baseline
from dataset_json_store import merge_records, read_records
from validate_trace import (
    BBOX_CUE_ERROR,
    BBOX_CUE_PATTERN,
    PICTURE_REFERENCE_ERROR,
    VIEW_IDENTIFIER_ERROR,
    contradicts_inspection_conclusion,
    has_repetition,
    retry_feedback,
    validate_evidence_first_reasoning,
)


FOCUS_COVERAGE_ERROR = "missing usable evidence for a marked query view"
REGION_LOCATOR_ERROR = "region needs a usable spatial locator"
COMPARISON_CONTENT_ERROR = "normal_comparison needs a visible comparison or a reason it is limited"
COMPARISON_VERDICT_ERROR = "normal_comparison contains a definitive defect judgment"
PROSE_SOURCE_ERRORS = {
    "reference leakage", BBOX_CUE_ERROR, "mask leakage", "annotation leakage",
    "coordinate leakage", "pipeline leakage", PICTURE_REFERENCE_ERROR, VIEW_IDENTIFIER_ERROR,
}
EVIDENCE_RETRY_GUIDANCE = {
    FOCUS_COVERAGE_ERROR: "Review every listed priority view. Give a visible fact for its marked physical part or state exactly what limits visibility.",
    REGION_LOCATOR_ERROR: "Name the inspected part or area in region. A quadrant is fine for a continuous surface.",
    COMPARISON_CONTENT_ERROR: "Compare the same part in query and reference, or state what prevents that comparison.",
    COMPARISON_VERDICT_ERROR: "Rewrite normal_comparison as visible similarities, differences or comparison limits. Leave the defect judgment to Stage 2.",
    f"{BBOX_CUE_ERROR} in region": "In region, give only the physical part or surface position, without how it was located.",
    f"{BBOX_CUE_ERROR} in query_evidence": "Rewrite query_evidence using the clean query for that view. Trace the object's own boundary and ignore the cue outline.",
    f"{BBOX_CUE_ERROR} in normal_comparison": "Redo normal_comparison using the clean query and the listed same-category normal reference. Compare the same part or explain what prevents comparison.",
    f"{BBOX_CUE_ERROR} in object_observation": "Rewrite object_observation using the overall form visible in the clean query.",
    f"{BBOX_CUE_ERROR} in uncertainty": "Rewrite uncertainty as a visibility or comparison limit of the images, or use null.",
}
REGION_CUE_SUFFIX = re.compile(
    r",\s+(?:as )?(?:indicated|marked|highlighted|shown|outlined) by "
    r"(?:the |a )?(?:red (?:box|rectangle)|bounding box)"
    r"(?: in (?:picture|image|view)\s+#?\d+)?[.!]?$",
    re.IGNORECASE,
)
UNLOCATED_REGION = re.compile(
    r"(?:the |an? )?(?:(?:highlighted|marked|indicated|selected|affected|target|visible) )?"
    r"(?:area|region|part|portion|surface)",
    re.IGNORECASE,
)
GRID_PLACEHOLDER = re.compile(
    r"(?:centered (?:near|in) )?(?:the )?(?:"
    r"(?:upper|middle|lower) row(?:,\s*(?:left|center|right) column)?|(?:left|center|right) column)"
    r"(?: of (?:the )?(?:full )?(?:query(?: image)?|image|frame))?",
    re.IGNORECASE,
)
QUERY_ONLY_COMPARISON = (
    r"(?:this is )?(?:an? )?(?:defect|anomaly) observation(?: in the query(?: view| image)?)?"
)
COMPARISON_PLACEHOLDER = re.compile(
    rf"(?:n/?a|not applicable|none)(?:[.;:,]\s*{QUERY_ONLY_COMPARISON})?|{QUERY_ONLY_COMPARISON}",
    re.IGNORECASE,
)
COMPARISON_VERDICT = re.compile(
    r"\b(?:confirms|proves|establishes|demonstrates)\s+(?:"
    r"(?:the presence of\s+)?(?:damage|defects|an? (?:surface |structural )?(?:defect|anomaly))\b"
    r"|[^.!?;\n]{1,120}?\b(?:is|are)\s+(?:damaged|defective|anomalous|"
    r"an? (?:surface |structural )?(?:defect|anomaly))\b)",
    re.IGNORECASE,
)


class EvidenceCueLeakage(ValueError):
    def __init__(self, fields: list[str]):
        super().__init__(BBOX_CUE_ERROR)
        self.fields = fields


class RegionalObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(description="Supplied query view identifier for this region.")
    region: str = Field(
        min_length=1,
        description="Only the inspected physical part or surface position. Omit the annotation source.",
    )
    query_evidence: list[str] = Field(description="Local features visible in the clean query for this view.")
    normal_comparison: str = Field(
        min_length=1,
        description="Same-part differences, similarities, or limits when comparing the clean query with the normal reference.",
    )
    visibility: Literal["clear", "partial", "not_visible"]
    uncertainty: Optional[str]


class VisualEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regional_observations: list[RegionalObservation] = Field(
        min_length=1,
        description="Local observations, starting with the actual marked areas when cues are supplied.",
    )
    object_observation: str = Field(
        min_length=1,
        description="Overall visible form, arrangement and appearance, with enough context to identify the inspected structures; detailed local findings belong in regional_observations.",
    )


class EvidenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_index: int = Field(ge=0)
    decision: Literal["keep", "revise", "uncertain"]
    verified_observation: str = Field(
        description="Key verified facts, corrections, or limits for this region.",
    )


class EvidenceAnchoredTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_review: list[EvidenceReview] = Field(min_length=1)
    inspection_conclusion: Literal["normal", "anomalous", "uncertain"]
    supporting_region_index: int | None = Field(
        description="Zero-based index of the verified local finding used in inspection_text for an anomalous conclusion. Null otherwise.",
    )
    inspection_text: str = Field(
        description="Explain how the verified observations support the inspection conclusion. At most 130 English words.",
    )


class ProseRepair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inspection_text: str


logger = logging.getLogger(__name__)


def collection_path(stage: str, dataset: str) -> Path:
    if stage == "evidence":
        return config.EVIDENCE_DIR / f"evidence_{dataset}.json"
    return config.EVIDENCE_FIRST_TRACE_DIR / f"traces_{dataset}.json"


def marked_query_views(sample: dict) -> set[str]:
    return {box["view_id"] for box in sample["supervision"]["bboxes"]}


def image_description(sample: dict) -> str:
    dimensions = ", ".join(
        f"{item['view_id']}={item['width']}x{item['height']}"
        for item in sample["query_images"]
    )
    inventory = "\n".join(
        f"- Picture {index}: {item['role']} (view_id={item['view_id']})"
        for index, item in enumerate(baseline.teacher_image_items(sample), start=1)
    )
    parts = [f"Query view dimensions: {dimensions}\nPicture inventory (exact input order):\n{inventory}"]
    marked_views = marked_query_views(sample)
    if marked_views:
        parts.append(
            f"Priority views with marked areas: {', '.join(sorted(marked_views))}. "
            "Use the cue to find the physical part, then inspect it and its boundary in the full clean query."
        )
    return "\n".join(parts)


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
    ]
    if errors:
        # Parser/IO errors may contain paths or internal identifiers. Do not relay
        # their raw text to this type-blind stage, even on a retry.
        tips = [tip for error, tip in EVIDENCE_RETRY_GUIDANCE.items() if error in errors]
        parts.extend(tips or ["Return one complete JSON object using the required fields and supplied query view IDs."])
    return "\n".join(parts)


def format_stage2_user_prompt(sample: dict, evidence: VisualEvidence, errors=None) -> str:
    # IDs can contain defect names. Associate responses with samples locally.
    marked_views = marked_query_views(sample)
    regions = [
        {"region_index": index, "priority_view": item.view_id in marked_views, **item.model_dump()}
        for index, item in enumerate(evidence.regional_observations)
    ]
    parts = [
        f"Object category: {sample['object_category']}",
        image_description(sample),
        "Stage 1 regional_observations. Review every listed region_index once. "
        "For each priority view, check the area indicated by its cue in the full clean query first.",
        json.dumps(regions, ensure_ascii=False),
    ]
    if errors:
        parts.append(retry_feedback(errors, evidence_first=True).replace("<think>", "inspection_text"))
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


def build_prose_repair_request(processor, trace: EvidenceAnchoredTrace, errors: list[str]) -> dict:
    messages = [
        {"role": "system", "content": (
            "Rewrite only the inspection_text as an English paragraph of at most 130 words. "
            "Use only confirmed observations, preserve the visibility limits and inspection conclusion, and describe "
            "the object itself without mentioning the sources of the evidence. Return one JSON object "
            "with inspection_text. Keep the observations and conclusion fixed."
        )},
        {"role": "user", "content": [
            {"type": "text", "text": (
                trace.model_dump_json() + "\n"
                + retry_feedback(errors, evidence_first=True).replace("<think>", "inspection_text")
            )},
        ]},
    ]
    return {"prompt": processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}


def json_payload(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else stripped


def parse_evidence(text: str, sample: dict) -> VisualEvidence:
    evidence = VisualEvidence.model_validate_json(json_payload(text))
    views = {item["view_id"] for item in sample["query_images"]}
    if any(item.view_id not in views for item in evidence.regional_observations):
        raise ValueError("evidence contains an unknown query view_id")
    inspected_views = {
        item.view_id for item in evidence.regional_observations
        if any(fact.strip() for fact in item.query_evidence)
        or (
            item.visibility != "clear"
            and (item.uncertainty or "").strip().casefold() not in {"", "none", "null", "n/a"}
        )
    }
    if marked_query_views(sample) - inspected_views:
        raise ValueError(FOCUS_COVERAGE_ERROR)
    descriptions = [("object_observation", evidence.object_observation)]
    for item in evidence.regional_observations:
        region = " ".join(item.region.split()).rstrip(".")
        suffix = REGION_CUE_SUFFIX.search(region)
        if suffix:
            locator = region[:suffix.start()].strip()
            if locator and not UNLOCATED_REGION.fullmatch(locator) and not BBOX_CUE_PATTERN.search(locator):
                item.region = region = locator
        if not region or GRID_PLACEHOLDER.fullmatch(region):
            raise ValueError(REGION_LOCATOR_ERROR)
        comparison = " ".join(item.normal_comparison.split()).rstrip(".")
        if not comparison or COMPARISON_PLACEHOLDER.fullmatch(comparison):
            raise ValueError(COMPARISON_CONTENT_ERROR)
        if COMPARISON_VERDICT.search(item.normal_comparison):
            raise ValueError(COMPARISON_VERDICT_ERROR)
        descriptions.extend([
            ("region", item.region),
            *(("query_evidence", fact) for fact in item.query_evidence),
            ("normal_comparison", item.normal_comparison),
            ("uncertainty", item.uncertainty or ""),
        ])
    leaked_fields = list(dict.fromkeys(field for field, text in descriptions if BBOX_CUE_PATTERN.search(text)))
    if leaked_fields:
        raise EvidenceCueLeakage(leaked_fields)
    if any(has_repetition(text) for _, text in descriptions):
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
    review = record.get("evidence_review")
    return (
        record.get("image_id") == sample["id"]
        and record.get("dataset") == sample["dataset"]
        and isinstance(record.get("reasoning"), str)
        and isinstance(record.get("inspection_text"), str)
        and isinstance(review, list)
        and bool(review)
        and all(
            isinstance(item, dict) and isinstance(item.get("verified_observation"), str)
            and not BBOX_CUE_PATTERN.search(item["verified_observation"])
            for item in review
        )
        and record["reasoning"] == assemble_reasoning(record["inspection_text"], sample)
        and validate_evidence_first_reasoning(record["reasoning"], sample, min_iou=min_iou).valid
    )


def parse_stage2_response(text: str, evidence: VisualEvidence) -> EvidenceAnchoredTrace:
    trace = EvidenceAnchoredTrace.model_validate_json(json_payload(text))
    indices = Counter(item.region_index for item in trace.evidence_review)
    expected = set(range(len(evidence.regional_observations)))
    problems = []
    for label, affected in (
        ("missing", expected - indices.keys()),
        ("duplicate", {index for index, count in indices.items() if count > 1}),
        ("unexpected", indices.keys() - expected),
    ):
        if affected:
            problems.append(f"{label} region_index {', '.join(map(str, sorted(affected)))}")
    if problems:
        raise ValueError("evidence_review must cover every region_index exactly once. " + ". ".join(problems))
    if any(not item.verified_observation.strip() for item in trace.evidence_review):
        raise ValueError("evidence_review observations must not be empty")
    if any(BBOX_CUE_PATTERN.search(item.verified_observation) for item in trace.evidence_review):
        raise ValueError(BBOX_CUE_ERROR)
    if any(has_repetition(item.verified_observation) for item in trace.evidence_review):
        raise ValueError("repetitive output in evidence_review")
    return trace


def inspection_support_error(trace: EvidenceAnchoredTrace, evidence: VisualEvidence, sample: dict) -> str | None:
    verified = set()
    for review in trace.evidence_review:
        region = evidence.regional_observations[review.region_index]
        if review.decision == "revise" or (
            review.decision == "keep" and region.visibility != "not_visible"
            and any(fact.strip() for fact in region.query_evidence)
        ):
            verified.add(review.region_index)
    boxed_views = marked_query_views(sample)
    if trace.inspection_conclusion == "normal":
        if not any(not boxed_views or evidence.regional_observations[i].view_id in boxed_views for i in verified):
            return "normal conclusion requires a verified observation in the query or marked area"
        return None

    index = trace.supporting_region_index
    if index is None or not 0 <= index < len(evidence.regional_observations):
        return "anomalous conclusion requires a valid supporting_region_index"
    region = evidence.regional_observations[index]
    if index not in verified:
        return "selected region lacks a verified query observation for the anomalous conclusion"
    if boxed_views and region.view_id not in boxed_views:
        return "selected region is outside the annotated query views"
    return None


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

    schema = {"evidence": VisualEvidence, "cot": EvidenceAnchoredTrace, "prose": ProseRepair}[stage].model_json_schema()
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
    fixed_traces, repair_params = {}, None
    # Use the configured budget for both full regional review and final assessment.
    # Repetition/truncation guards handle failures without shrinking normal prose.
    saved = held_for_review = 0
    for attempt in range(1, args.max_retries + 1):
        if not pending:
            break
        requests, request_samples, request_params, next_pending = [], [], [], []
        for sample in pending:
            try:
                image_id = sample["id"]
                params = sampling_params
                if image_id in fixed_traces:
                    if repair_params is None:
                        repair_params = make_sampling_params(args, "prose")
                    params = repair_params
                    request = build_prose_repair_request(processor, fixed_traces[image_id], feedback[image_id])
                else:
                    evidence = load_sample_evidence(sample, evidence_records) if stage == "cot" else None
                    request, _ = build_request(sample, processor, system_prompt, stage, evidence, feedback.get(image_id))
                requests.append(request)
                request_samples.append(sample)
                request_params.append(params)
            except Exception as error:
                feedback[sample["id"]] = [f"request construction failed: {error}"]
                next_pending.append(sample)
        if not requests:
            pending = next_pending
            break
        # As in the baseline, engine-level failures stop the run. Completed batches
        # are already on disk and the same command resumes their remaining work.
        outputs = engine.generate(requests, request_params if fixed_traces else sampling_params)
        if len(outputs) != len(request_samples):
            raise RuntimeError("vLLM output count differs from request count")
        records = {}
        for sample, output, params in zip(request_samples, outputs, request_params):
            image_id = sample["id"]
            completion = output.outputs[0] if output.outputs else None
            raw = completion.text if completion else ""
            last_outputs[image_id] = raw
            metadata = {
                "attempt": attempt,
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
                "output_tokens": len(getattr(completion, "token_ids", []) or []),
                "token_budget": getattr(params, "max_tokens", None),
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
                    evidence = load_sample_evidence(sample, evidence_records)
                    if image_id in fixed_traces:
                        repair = ProseRepair.model_validate_json(json_payload(raw))
                        trace = fixed_traces[image_id].model_copy(update={"inspection_text": repair.inspection_text})
                    else:
                        trace = parse_stage2_response(raw, evidence)
                    if contradicts_inspection_conclusion(trace.inspection_text, trace.inspection_conclusion):
                        errors = ["Make the whole-object conclusion in inspection_text agree with inspection_conclusion, using verified image evidence."]
                        feedback[image_id] = metadata["errors"] = errors
                        next_pending.append(sample)
                        continue
                    expected = "anomalous" if sample["is_anomaly"] else "normal"
                    review_error = None
                    if trace.inspection_conclusion != expected:
                        review_error = f"inspection_conclusion: {trace.inspection_conclusion}; dataset label: {expected}; review required"
                    else:
                        review_error = inspection_support_error(trace, evidence, sample)
                    if review_error:
                        # Preserve insufficient or conflicting evidence without retrying toward the label.
                        errors = [review_error]
                        metadata["errors"] = errors
                        append_failure(sample, stage, errors, raw, args.shard_id, histories[image_id],
                                       failure_log=args.failure_log)
                        held_for_review += 1
                        continue
                    reasoning = assemble_reasoning(trace.inspection_text, sample)
                    validation = validate_evidence_first_reasoning(reasoning, sample, min_iou=args.min_location_iou)
                    if not validation.valid:
                        if validation.errors and all(error in PROSE_SOURCE_ERRORS for error in validation.errors):
                            fixed_traces[image_id] = trace
                        feedback[image_id] = validation.errors
                        metadata["errors"] = validation.errors
                        next_pending.append(sample)
                        continue
                    record.update({
                        "inspection_text": trace.inspection_text.strip(),
                        "inspection_conclusion": trace.inspection_conclusion,
                        "supporting_region_index": trace.supporting_region_index if expected == "anomalous" else None,
                        "reasoning": reasoning, "validation": validation.to_dict(),
                        "evidence_review": [item.model_dump() for item in trace.evidence_review],
                        "supervision_fields_source": "manifest_gt",
                        "teacher_bbox_coordinates_exposed": False,
                        "evidence_file": str(collection_path("evidence", sample["dataset"])),
                    })
                records[image_id] = record
                metadata["errors"] = []
            except Exception as error:
                if isinstance(error, EvidenceCueLeakage):
                    feedback[image_id] = [BBOX_CUE_ERROR, *(f"{BBOX_CUE_ERROR} in {field}" for field in error.fields)]
                elif str(error) == BBOX_CUE_ERROR or str(error) in EVIDENCE_RETRY_GUIDANCE or str(error).startswith(
                    ("truncated output", "repetitive output", "evidence_review")
                ):
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
    return saved, held_for_review + len(pending)


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
    processor, engine, failures, blocked_on_evidence = None, None, 0, 0
    for stage in (["evidence", "cot"] if args.stage == "all" else [args.stage]):
        system_prompt = (config.STAGE1_PROMPT_PATH if stage == "evidence" else config.STAGE2_PROMPT_PATH).read_text(encoding="utf-8")
        sampling_params = None
        for dataset in args.datasets:
            # Stage 2 deliberately reads B.json from disk, including after Stage 1
            # in the same process. This is also the independent cot-only path.
            evidence_records = read_records(collection_path("evidence", dataset), dataset)
            pending = []
            blocked = 0
            for sample in samples:
                if sample["dataset"] != dataset:
                    continue
                if stage == "evidence" and has_evidence(sample, evidence_records):
                    continue
                if stage == "cot" and not has_evidence(sample, evidence_records):
                    blocked += 1
                    continue
                pending.append(sample)
            if blocked:
                logger.warning("%s: %d CoT samples waiting for valid Stage 1 evidence", dataset, blocked)
                blocked_on_evidence += blocked
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
    logger.info("finished; generation failures=%d; CoT samples waiting for Stage 1 evidence=%d",
                failures, blocked_on_evidence)
    return 1 if failures or (blocked_on_evidence and not args.debug) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
