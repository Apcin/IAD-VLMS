"""Validation for pixel-coordinate 2D and multiview anomaly reasoning traces."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Optional


NORMAL_PATTERN = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*<answer>No</answer>\s*$",
    re.DOTALL,
)
ANOMALY_PATTERN = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*"
    r"<location>\s*(?P<location>.*?)\s*</location>\s*"
    r"<type>\s*(?P<type>.*?)\s*</type>\s*"
    r"<answer>Yes</answer>\s*$",
    re.DOTALL,
)
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
FORBIDDEN_PATTERNS = {
    "annotation leakage": r"\b(?:ground[ -]?truth|annotation|metadata|supervision)\b",
    "mask leakage": r"\b(?:segmentation )?mask\b",
    "bbox-cue leakage": r"\b(?:red|drawn|provided|highlighted) (?:box|bounding box|rectangle)\b",
    "reference leakage": r"\b(?:reference image|provided normal|comparison image)\b",
    "multi-image leakage": r"\b(?:second|third|fifth|ninth) image\b",
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    word_count: int = 0
    answer: Optional[str] = None
    defect_type: Optional[str] = None
    predicted_locations: Optional[list[dict]] = None
    per_view_iou: Optional[dict[str, float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


def bbox_iou(left: list[int], right: list[int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    left_area = max(0, lx2 - lx1 + 1) * max(0, ly2 - ly1 + 1)
    right_area = max(0, rx2 - rx1 + 1) * max(0, ry2 - ry1 + 1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def parse_locations(value: str, target_modality: str) -> list[dict]:
    parsed = json.loads(value)
    if target_modality == "2d":
        if not isinstance(parsed, list) or len(parsed) != 4 or not all(isinstance(v, int) for v in parsed):
            raise ValueError("2D location must be one integer [x1,y1,x2,y2] list")
        return [{"view_id": "query", "bbox": parsed}]
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("multiview location must be a non-empty JSON list")
    locations = []
    for item in parsed:
        if not isinstance(item, dict) or set(item) != {"view_id", "bbox"}:
            raise ValueError("each multiview location needs exactly view_id and bbox")
        bbox = item["bbox"]
        if not isinstance(item["view_id"], str) or not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("invalid multiview view_id/bbox")
        if not all(isinstance(value, int) for value in bbox):
            raise ValueError("bbox coordinates must be integers")
        locations.append({"view_id": item["view_id"], "bbox": bbox})
    return locations


def _normalized_type(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.strip().lower())


def validate_reasoning(
    reasoning: str,
    sample: dict,
    *,
    min_iou: float = 0.5,
    min_words: int = 80,
    max_words: int = 220,
) -> ValidationResult:
    is_anomaly = sample["is_anomaly"]
    match = (ANOMALY_PATTERN if is_anomaly else NORMAL_PATTERN).fullmatch(reasoning or "")
    if not match:
        expected = (
            "<think>...</think><location>...</location><type>...</type><answer>Yes</answer>"
            if is_anomaly
            else "<think>...</think><answer>No</answer>"
        )
        return ValidationResult(False, [f"wrong tag structure; expected {expected}"])

    errors = []
    think = match.group("think").strip()
    word_count = len(WORD_PATTERN.findall(think))
    if word_count < min_words or word_count > max_words:
        errors.append(f"<think> word count {word_count} is outside {min_words}-{max_words}")
    if re.search(r"</?[A-Za-z][^>]*>", think):
        errors.append("nested XML/HTML inside <think>")
    for label, pattern in FORBIDDEN_PATTERNS.items():
        if re.search(pattern, think, flags=re.IGNORECASE):
            errors.append(label)

    if not is_anomaly:
        return ValidationResult(not errors, errors, word_count, answer="No")

    defect_type = match.group("type").strip()
    expected_type = sample["supervision"]["anomaly_type"]["canonical"]
    if _normalized_type(defect_type) != _normalized_type(expected_type):
        errors.append("defect type disagrees with the supplied type label")

    locations = None
    per_view_iou = {}
    try:
        locations = parse_locations(match.group("location"), sample["target_modality"])
        predicted = {item["view_id"]: item["bbox"] for item in locations}
        if len(predicted) != len(locations):
            errors.append("duplicate view_id in <location>")
        expected = {item["view_id"]: item["xyxy_pixel"] for item in sample["supervision"]["bboxes"]}
        sizes = {
            item["view_id"]: (item["width"], item["height"])
            for item in sample["query_images"]
        }
        if set(predicted) != set(expected):
            errors.append("predicted view IDs do not match all bbox-marked query views")
        for current_view in sorted(set(predicted) & set(expected)):
            box = predicted[current_view]
            width, height = sizes[current_view]
            x1, y1, x2, y2 = box
            if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
                errors.append(f"bbox for {current_view} is outside the original image")
                continue
            score = bbox_iou(box, expected[current_view])
            per_view_iou[current_view] = score
            if score <= min_iou:
                errors.append(
                    f"bbox for {current_view} must have IoU greater than {min_iou:.3f} with the anomaly"
                )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        errors.append(f"invalid <location>: {error}")

    return ValidationResult(
        not errors,
        errors,
        word_count,
        answer="Yes",
        defect_type=defect_type,
        predicted_locations=locations,
        per_view_iou=per_view_iou,
    )


def retry_feedback(errors: list[str]) -> str:
    """Return corrective feedback; the original prompt already contains the GT hint."""
    safe = []
    for error in errors:
        if "IoU greater than" in error or "view IDs" in error or "outside" in error:
            safe.append(
                "Re-inspect every boxed query view and use the supplied location hint to output a highly consistent bbox in the original image frame."
            )
        else:
            safe.append(error)
    return "Previous output failed validation. Correct these issues without discussing validation: " + "; ".join(dict.fromkeys(safe))
