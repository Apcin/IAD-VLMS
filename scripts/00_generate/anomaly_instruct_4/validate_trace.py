"""Validation for pixel-coordinate 2D and multiview anomaly reasoning traces."""

from __future__ import annotations

import json
import re
from collections import Counter
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
EVIDENCE_FIRST_MAX_WORDS = 130
PICTURE_REFERENCE_ERROR = "picture-number or input-order reference"
PICTURE_REFERENCE_PATTERN = re.compile(
    r"\b(?:picture|image)\s*#?\s*\d+\b"
    r"|\b(?:picture|image)\s+(?:number\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
    r"|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d+(?:st|nd|rd|th))\s+(?:picture|image)\b",
    re.IGNORECASE,
)
VIEW_IDENTIFIER_ERROR = "view identifier in descriptive text"
VIEW_IDENTIFIER_PATTERN = re.compile(r"\bview[_ -]?ids?\b|\bview_\d+\b", re.IGNORECASE)
BBOX_CUE_ERROR = "bbox-cue leakage"
BBOX_CUE_PATTERN = re.compile(
    r"\b(?:bounding (?:box(?:es)?|rectangles?)|bbox(?:es)?"
    r"|(?:annotation|localization) (?:cues?|overlays?|outlines?|box(?:es)?))\b"
    r"|\b(?:red|drawn|provided|highlighted|marked) (?:box(?:es)?|rectangles?)\b"
    r"|\b(?:outline|box) cues?\b|\b(?:drawn|overlaid) (?:outlines?|borders?)\b"
    r"|\b(?:lacks?|without|no) (?:the |a )?red outlines?\b",
    re.IGNORECASE,
)
FORBIDDEN_PATTERNS = {
    "annotation leakage": (
        r"\b(?:ground[ -]?truth|annotation|metadata|supervision)\b"
        r"|\b(?:canonical|supplied|provided)\s+(?:(?:anomaly|defect)\s+)?(?:type|label|category)\b"
    ),
    "mask leakage": (
        r"\b(?:segmentation|annotation|ground[ -]?truth|gt|binary|defect|anomaly|provided|supplied)\s+masks?\b"
        r"|\bmasks?\s+(?:images?|overlays?)\b"
    ),
    BBOX_CUE_ERROR: BBOX_CUE_PATTERN.pattern,
    "reference leakage": (
        r"\b(?:normal|reference|comparison)\s+(?:images?|pictures?|views?|samples?|photographs?)\b"
        r"|\bnormal\s+references?\b|\b(?:provided|supplied)\s+normal\b"
        r"|\b(?:provided|supplied|given)\s+(?:images?|pictures?|views?|samples?)\b"
        r"|\breference\s+comparisons?\b"
    ),
    "coordinate leakage": (
        r"[\[(]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*[\])]"
        r"|\b(?:pixel[ -]coordinates?|location[ -]coordinates?)\b"
        r"|\b(?:provided|supplied|given)\s+coordinates?\b"
    ),
    "pipeline leakage": r"\bevidence_review\b|\bevidence\s+(?:json|record)\b|\b(?:earlier|previous)\s+stage\b",
    PICTURE_REFERENCE_ERROR: PICTURE_REFERENCE_PATTERN.pattern,
    VIEW_IDENTIFIER_ERROR: VIEW_IDENTIFIER_PATTERN.pattern,
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


def has_repetition(text: str) -> bool:
    """Catch long loops within one field, not matching facts across regions."""
    words = WORD_PATTERN.findall(text.lower())
    sequences = Counter(tuple(words[index:index + 12]) for index in range(len(words) - 11))
    return bool(sequences and max(sequences.values()) >= 5)


def validate_evidence_first_reasoning(reasoning: str, sample: dict, *, min_iou: float = 0.5) -> ValidationResult:
    result = validate_reasoning(reasoning, sample, min_iou=min_iou, min_words=1, max_words=EVIDENCE_FIRST_MAX_WORDS)
    match = (ANOMALY_PATTERN if sample["is_anomaly"] else NORMAL_PATTERN).fullmatch(reasoning or "")
    if match:
        think = match.group("think")
        if re.search(r"\b(?:WRITE_|REPLACE_WITH)\w*|visually grounded inspection reasoning", think, re.IGNORECASE):
            result.errors.append("placeholder inspection text")
        if has_repetition(think):
            result.errors.append("repetitive output in inspection_text")
        result.valid = not result.errors
    return result


def contradicts_inspection_conclusion(inspection_text: str, conclusion: str) -> bool:
    """Catch direct whole-object contradictions in the final English sentence."""
    sentence = re.split(r"(?<=[.!?])\s+", inspection_text.strip())[-1].rstrip(".!?").lower()
    if re.search(r"\b(?:although|but|despite|except|however|otherwise|yet)\b", sentence):
        return False
    judgment_subject = r"(?:this|these (?:findings|observations|features)|the (?:findings|observations|features))"
    if conclusion == "anomalous":
        return bool(re.fullmatch(
            r"(?:overall,? )?(?:(?:there (?:is|are) )?no (?:visible )?(?:defects?|anomalies)"
            r"(?: (?:are|is) (?:visible|evident))?|"
            r"the (?:object|item|sample) (?:appears|is) (?:normal|undamaged|defect[- ]free)|"
            rf"{judgment_subject} (?:supports?|indicates?) (?:a )?normal (?:condition|classification|judg(?:e)?ment))",
            sentence,
        ) or re.match(
            r"(?:the )?overall condition (?:suggests?|indicates?) (?:an? )?(?:standard,\s*)?"
            r"undamaged (?:object|item|sample)\b", sentence,
        ))
    if conclusion == "normal":
        return bool(re.fullmatch(
            rf"{judgment_subject} (?:supports?|indicates?|confirms?) "
            r"(?:an? )?(?:anomalous (?:condition|classification|judg(?:e)?ment)|defect)",
            sentence,
        ))
    return False


def retry_feedback(errors: list[str], *, evidence_first: bool = False) -> str:
    """Return corrective feedback without leaking supervision to Stage 2."""
    safe = []
    for error in errors:
        if error == "reference leakage":
            safe.append("Remove all mentions of normal/reference/comparison inputs from <think>; retain supported observations and describe only the inspected object.")
        elif error == "annotation leakage":
            safe.append("Remove supervision sources, canonical-type wording and supplied-label references from <think>; describe visible morphology without citing labels.")
        elif error == BBOX_CUE_ERROR:
            safe.append("Use the localization cue only to find the area. Describe the object in the clean query and compare the same physical area in the reference.")
        elif error == "coordinate leakage":
            safe.append("Remove coordinate lists and coordinate commentary from <think>; keep required coordinates only in <location> and use spatial descriptions in the inspection prose.")
        elif error == "pipeline leakage":
            safe.append("Remove evidence records, review-field names and earlier-stage references from the final inspection; describe the query object directly.")
        elif "repetitive output" in error or "truncated output" in error:
            safe.append("Return a complete JSON object. Remove looping passages while preserving distinct visual facts and the explanation connecting them to the conclusion. State each point once.")
        elif error == "placeholder inspection text":
            safe.append("Replace placeholder wording with actual visible observations, an explanation of their inspection significance and a supported conclusion.")
        elif error.startswith("evidence_review must cover every region_index"):
            safe.append(error + ". Copy the listed region_index values and combine each region's findings into one entry.")
        elif error.startswith("evidence_review"):
            safe.append("Return exactly one evidence_review entry for every listed region_index, without duplicates or omissions; use keep, revise or uncertain and retain the verified facts needed for the inspection judgment.")
        elif "word count" in error:
            if evidence_first:
                safe.append(f"{error}. Keep the key evidence and conclusion within the word limit.")
            else:
                safe.append(f"{error}. Rewrite the inspection within that limit, aiming for 80-120 words where applicable; preserve key evidence, its significance and the conclusion, without padding or repetition.")
        elif error == PICTURE_REFERENCE_ERROR:
            safe.append("Remove picture numbers and input-order references from <think>; describe the query directly.")
        elif error == VIEW_IDENTIFIER_ERROR:
            safe.append("Remove view identifiers from the final prose." if evidence_first else
                        "Remove view identifiers from <think>; retain required view_id values only in <location>.")
        elif "IoU greater than" in error or "view IDs" in error or "outside" in error:
            if not evidence_first:
                safe.append("Re-inspect every boxed query view and use the supplied location hint to output a highly consistent bbox in the original image frame.")
        else:
            safe.append(error)
    return "Previous output failed validation. Correct these issues without discussing validation: " + "; ".join(dict.fromkeys(safe))
