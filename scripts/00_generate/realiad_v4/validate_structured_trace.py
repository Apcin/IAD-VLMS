"""Executable quality control for Real-IAD structured reasoning traces."""

import re
from dataclasses import asdict, dataclass
from typing import Optional


ALLOWED_LOCATIONS = {
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
}

ALLOWED_TYPES = {
    "Scratch", "Crack", "Dent", "Pit", "Abrasion", "Discoloration",
    "Contamination", "Corrosion", "Stain", "Burr", "Flash", "Broken",
    "Fracture", "Deformation", "Warping", "Chip", "Damage", "Bend",
    "Missing Parts", "Missing Component", "Extra Component",
    "Incomplete Assembly", "Misalignment", "Foreign Object", "Poor Finish",
    "Overmolding",
}

FORBIDDEN_LEAKAGE_PATTERNS = {
    "ground-truth leakage": r"\bground[ -]?truth\b",
    "metadata leakage": r"\bmetadata\b|\bannotation (?:says|indicates|shows)\b",
    "hint leakage": r"\b(?:the )?hint (?:says|indicates|shows)\b",
    "overlay leakage": r"\b(?:red )?overlay\b|\bhighlighted region\b|\bred-highlighted\b",
    "mask leakage": r"\b(?:segmentation )?mask\b",
    "reference leakage": r"\breference image\b|\bprovided normal image\b",
    "multi-image leakage": r"\b(?:second|third) image\b|\bimage [23]\b|\bmultiple images\b",
    "batch leakage": r"\bthis batch\b|\bprevious sample\b|\bnext sample\b",
}

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


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    think_word_count: int = 0
    answer: Optional[str] = None
    location: Optional[str] = None
    defect_type: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _location_set(value: Optional[str]) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def validate_reasoning(
    reasoning: str,
    *,
    is_anomaly: bool,
    expected_type: Optional[str] = None,
    expected_location: Optional[str] = None,
    min_words: int = 120,
    max_words: int = 200,
) -> ValidationResult:
    errors: list[str] = []
    pattern = ANOMALY_PATTERN if is_anomaly else NORMAL_PATTERN
    match = pattern.fullmatch(reasoning or "")

    if not match:
        expected = (
            "<think>...</think><location>...</location><type>...</type><answer>Yes</answer>"
            if is_anomaly
            else "<think>...</think><answer>No</answer>"
        )
        return ValidationResult(
            valid=False,
            errors=[f"wrong tag structure; expected exactly {expected}"],
        )

    think = match.group("think").strip()
    word_count = len(WORD_PATTERN.findall(think))
    if word_count < min_words or word_count > max_words:
        errors.append(
            f"<think> word count is {word_count}; required range is {min_words}-{max_words}"
        )
    if re.search(r"</?[a-zA-Z][^>]*>", think):
        errors.append("nested XML/HTML tag inside <think>")

    for label, pattern_text in FORBIDDEN_LEAKAGE_PATTERNS.items():
        if re.search(pattern_text, think, flags=re.IGNORECASE):
            errors.append(label)

    location = match.groupdict().get("location")
    defect_type = match.groupdict().get("type")
    answer = "Yes" if is_anomaly else "No"

    if is_anomaly:
        predicted_locations = _location_set(location)
        invalid_locations = predicted_locations - ALLOWED_LOCATIONS
        if not predicted_locations:
            errors.append("empty <location>")
        if invalid_locations:
            errors.append(
                "invalid <location> value(s): " + ", ".join(sorted(invalid_locations))
            )

        expected_locations = _location_set(expected_location)
        if expected_locations and predicted_locations != expected_locations:
            errors.append(
                f"location disagrees with mask: predicted={location}, expected={expected_location}"
            )

        if defect_type not in ALLOWED_TYPES:
            errors.append(f"invalid <type>: {defect_type}")
        if expected_type and defect_type != expected_type:
            errors.append(
                f"defect type disagrees with label: predicted={defect_type}, expected={expected_type}"
            )

    return ValidationResult(
        valid=not errors,
        errors=errors,
        think_word_count=word_count,
        answer=answer,
        location=location,
        defect_type=defect_type,
    )


def retry_feedback(errors: list[str]) -> str:
    joined = "\n".join(f"- {error}" for error in errors)
    return (
        "Your previous response failed executable validation for these reasons:\n"
        f"{joined}\n"
        "Regenerate the complete JSON object from scratch. Correct every listed issue, "
        "keep all claims visually grounded, and output JSON only."
    )
