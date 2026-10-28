"""
Verify reasoning traces using OpenAI's GPT-5-mini model via Responses API.
Processes traces in batches of 10 for efficiency.
"""
import json
import logging
import sys
import time
import argparse
import re
import random
import base64
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from PIL import Image
import os
import utils_realiad as utils

load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TRACES_DIR = DATA_DIR / "output" / "reasoning_traces_realiad_new"
VERIFICATION_OUTPUT_DIR = DATA_DIR / "output" / "verification_results"
LOGS_DIR = PROJECT_ROOT / "logs"
JSON_ROOT = DATA_DIR / "Real-IAD" / "json" / "realiad_jsons"
IMAGES_ROOT = DATA_DIR / "Real-IAD" / "images"

BATCH_SIZE = 15

VERIFICATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(run_id: str = "main"):
    log_file = LOGS_DIR / f"trace_verification_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# Predefined issue codes - model must pick from these
# NOTE: We only check FORMAT (tag existence), not CONTENT validity
# Content validation (location values, defect types) can be done programmatically
VALID_ISSUES = [
    "missing_think_tags",       # No <think>...</think> tags
    "missing_answer_tag",       # No <answer>Yes/No</answer> tag
    "missing_location_tag",     # Answer is Yes but no <location> tag
    "missing_type_tag",         # Answer is Yes but no <type> tag
    "extra_location_tag",       # Answer is No but has <location> tag
    "extra_type_tag",           # Answer is No but has <type> tag
    "has_batch_reference",      # Contains "another", "next", "previous", "tenth", etc.
    "illogical_reasoning",      # Reasoning doesn't support the conclusion
    "incoherent_content",       # Reasoning is nonsensical or off-topic
    "wrong_product_reference",  # Describes wrong product type
]


class VerificationStatus(Enum):
    COMPLETELY_CORRECT = "completely_correct"  # Perfect, no issues
    CORRECT = "correct"                        # Minor issues but acceptable
    WRONG = "wrong"                            # Significant issues
    COMPLETELY_WRONG = "completely_wrong"      # Major failures
    ERROR = "error"                            # Processing error


class SingleVerificationResult(BaseModel):
    image_id: str
    status: str  # completely_correct, correct, wrong, completely_wrong
    confidence_score: float
    issues: List[str] = Field(default_factory=list)  # From VALID_ISSUES only
    issue_elaboration: List[str] = Field(default_factory=list)  # Brief explanation for each issue


class BatchVerificationResponse(BaseModel):
    verifications: List[SingleVerificationResult]


@dataclass
class VerificationResult:
    image_id: str
    status: VerificationStatus
    issues: List[str]
    confidence_score: float
    image_path: str = ""
    issue_elaboration: List[str] = None

    def __post_init__(self):
        if self.issue_elaboration is None:
            self.issue_elaboration = []


BATCH_VERIFICATION_SYSTEM_PROMPT = f"""You are verifying industrial defect detection reasoning traces.

For EACH trace, you will receive:
1. The ORIGINAL image of the product
2. An OVERLAY image with the defect highlighted in RED
3. The reasoning trace text

Compare the reasoning trace against what you see in the images.

## FORMAT REQUIREMENTS (check tag EXISTENCE only)
1. Must have <think>...</think> tags
2. Must have <answer>Yes</answer> or <answer>No</answer>
3. If answer=Yes: MUST have <location> and <type> tags (any value inside is fine)
4. If answer=No: should NOT have <location> or <type> tags

## CONTENT REQUIREMENTS (compare with images)
1. No batch references ("another", "next", "previous", "final", "tenth", "across X samples")
2. Reasoning should logically lead to the conclusion
3. Should describe the correct product type (matches what you see in the image)
4. The defect description should match what you see in the overlay image
5. The location mentioned should roughly match where the red highlight is

## WHAT TO IGNORE - DO NOT FLAG THESE
- Word count / length is NOT an issue
- Minor stylistic differences are acceptable
- Specific wording choices don't matter
- Exact location wording (as long as it's roughly correct area)
- Exact defect type naming (as long as it describes what's visible)

## STATUS LEVELS
- "completely_correct": Perfect trace, accurately describes the image
- "correct": Acceptable, minor imperfections but describes the defect correctly
- "wrong": Has real issues - doesn't match image or has format problems
- "completely_wrong": Major failures - completely wrong description or format violations

## ISSUE CODES (pick ONLY from this list)
{json.dumps(VALID_ISSUES, indent=2)}

Return JSON:
{{
    "verifications": [
        {{
            "image_id": "exact_id",
            "status": "completely_correct" | "correct" | "wrong" | "completely_wrong",
            "confidence_score": 0.0-1.0,
            "issues": ["issue_code_1", "issue_code_2"],
            "issue_elaboration": ["brief explanation 1", "brief explanation 2"]
        }}
    ]
}}

IMPORTANT:
- Return results for ALL traces
- issues array should be EMPTY for completely_correct/correct status
- issue_elaboration array must match issues array (same length, one explanation per issue)
- Keep elaborations brief: "Image shows X, but reasoning says Y" format
- Only use issue codes from the list above
- Compare the reasoning against what you actually see in the images"""


def load_image_path_lookup() -> Dict[str, str]:
    """Load image paths from Real-IAD metadata."""
    lookup = {}
    if JSON_ROOT.exists():
        entries = utils.load_realiad_metadata(JSON_ROOT)
        for entry in entries:
            image_id = entry.get('image_id')
            image_rel_path = entry.get('image_rel_path', '')
            if image_id and image_rel_path:
                # Store full path
                full_path = IMAGES_ROOT / image_rel_path
                lookup[image_id] = str(full_path)
        logger.info(f"Loaded {len(lookup)} image paths from metadata")
    else:
        logger.warning(f"Real-IAD metadata not found at {JSON_ROOT}")
    return lookup


def load_mask_path_lookup() -> Dict[str, str]:
    """Load mask paths from Real-IAD metadata."""
    lookup = {}
    if JSON_ROOT.exists():
        entries = utils.load_realiad_metadata(JSON_ROOT)
        for entry in entries:
            image_id = entry.get('image_id')
            mask_rel_path = entry.get('mask_rel_path', '')
            if image_id and mask_rel_path:
                full_path = IMAGES_ROOT / mask_rel_path
                lookup[image_id] = str(full_path)
        logger.info(f"Loaded {len(lookup)} mask paths from metadata")
    else:
        logger.warning(f"Real-IAD metadata not found at {JSON_ROOT}")
    return lookup


def create_overlay(image_path: Path, mask_path: Path, color=(255, 0, 0), alpha=128) -> Image.Image:
    """Create an overlay of the mask on the image."""
    base_img = Image.open(image_path).convert("RGBA")
    mask_img = Image.open(mask_path).convert("L")

    # Resize mask to match image if needed
    if mask_img.size != base_img.size:
        mask_img = mask_img.resize(base_img.size, Image.NEAREST)

    # Create colored overlay
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    for x in range(base_img.width):
        for y in range(base_img.height):
            if mask_img.getpixel((x, y)) > 127:
                overlay.putpixel((x, y), (*color, alpha))

    return Image.alpha_composite(base_img, overlay).convert("RGB")


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_trace_files() -> List[Dict[str, Any]]:
    """Load all existing trace files."""
    traces = []
    trace_files = list(TRACES_DIR.glob("trace_*.json"))

    for trace_file in trace_files:
        try:
            with open(trace_file, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
                trace_data['_file_path'] = str(trace_file)
                traces.append(trace_data)
        except Exception as e:
            logger.warning(f"Failed to load {trace_file}: {e}")

    logger.info(f"Loaded {len(traces)} trace files")
    return traces


def format_batch_verification_prompt(batch: List[Dict[str, Any]]) -> str:
    """Format a batch of traces for verification."""
    prompt_parts = []
    prompt_parts.append(f"Verify these {len(batch)} reasoning traces:")
    prompt_parts.append("")

    for idx, item in enumerate(batch, 1):
        image_id = item.get('image_id', f'unknown_{idx}')
        reasoning = item.get('reasoning', '')

        prompt_parts.append(f"--- TRACE {idx} ---")
        prompt_parts.append(f"ID: {image_id}")
        prompt_parts.append(reasoning)
        prompt_parts.append("")

    return "\n".join(prompt_parts)


def parse_json_response(response_text: str) -> Dict:
    """Parse JSON from response."""
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text, re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response_text.strip()
    return json.loads(json_str)


def verify_batch(
    client: OpenAI,
    batch: List[Dict[str, Any]],
    image_path_lookup: Dict[str, str],
    mask_path_lookup: Dict[str, str],
    model: str = "gpt-5-mini",
    reasoning_effort: str = "low"
) -> Dict[str, VerificationResult]:
    """Verify a batch of traces using GPT-5-mini Responses API with images."""

    # Build content with images and text for each trace
    user_content = []
    user_content.append({
        "type": "input_text",
        "text": f"Verify these {len(batch)} reasoning traces. For each trace, I provide the original image, the overlay image (defect in red), and the reasoning text."
    })

    for idx, item in enumerate(batch, 1):
        image_id = item.get('image_id', f'unknown_{idx}')
        reasoning = item.get('reasoning', '')

        # Add trace header
        user_content.append({
            "type": "input_text",
            "text": f"\n--- TRACE {idx} ---\nID: {image_id}"
        })

        # Try to load and add images
        img_path = image_path_lookup.get(image_id)
        mask_path = mask_path_lookup.get(image_id)

        if img_path and mask_path and Path(img_path).exists() and Path(mask_path).exists():
            try:
                # Original image
                original_img = Image.open(img_path).convert("RGB")
                original_b64 = image_to_base64(original_img)
                user_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{original_b64}"
                })

                # Overlay image with mask
                overlay_img = create_overlay(Path(img_path), Path(mask_path))
                overlay_b64 = image_to_base64(overlay_img)
                user_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{overlay_b64}"
                })
            except Exception as e:
                logger.warning(f"Failed to load images for {image_id}: {e}")
                user_content.append({
                    "type": "input_text",
                    "text": "[Images could not be loaded]"
                })
        else:
            user_content.append({
                "type": "input_text",
                "text": "[Images not available]"
            })

        # Add reasoning text
        user_content.append({
            "type": "input_text",
            "text": f"\nReasoning:\n{reasoning}"
        })

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": BATCH_VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            reasoning={"effort": reasoning_effort},
            text={"format": {"type": "json_object"}}
        )

        # Extract text from response
        result_text = None
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        result_text = content.text
                        break

        if not result_text:
            raise ValueError("No text output in response")

        result_json = parse_json_response(result_text)

        # Handle response format variations
        if isinstance(result_json, list):
            result_json = {'verifications': result_json}
        if 'verifications' not in result_json and 'image_id' in result_json:
            result_json = {'verifications': [result_json]}

        validated = BatchVerificationResponse(**result_json)

        # Convert to results
        results = {}
        status_map = {
            "completely_correct": VerificationStatus.COMPLETELY_CORRECT,
            "correct": VerificationStatus.CORRECT,
            "wrong": VerificationStatus.WRONG,
            "completely_wrong": VerificationStatus.COMPLETELY_WRONG
        }

        for v in validated.verifications:
            # Filter to only valid issue codes and keep corresponding elaborations
            valid_issues = []
            valid_elaborations = []
            for idx, issue in enumerate(v.issues):
                if issue in VALID_ISSUES:
                    valid_issues.append(issue)
                    # Get corresponding elaboration if available
                    if idx < len(v.issue_elaboration):
                        valid_elaborations.append(v.issue_elaboration[idx])
                    else:
                        valid_elaborations.append("")

            results[v.image_id] = VerificationResult(
                image_id=v.image_id,
                status=status_map.get(v.status.lower(), VerificationStatus.ERROR),
                issues=valid_issues,
                confidence_score=v.confidence_score,
                issue_elaboration=valid_elaborations
            )

        return results

    except Exception as e:
        logger.error(f"Batch verification error: {e}")
        results = {}
        for item in batch:
            image_id = item.get('image_id', 'unknown')
            results[image_id] = VerificationResult(
                image_id=image_id,
                status=VerificationStatus.ERROR,
                issues=[],
                confidence_score=0.0,
                issue_elaboration=[]
            )
        return results


def save_verification_result(result: VerificationResult, original_reasoning: str, output_dir: Path):
    """Save a single verification result to disk."""
    output_file = output_dir / f"verify_{result.image_id}.json"

    result_dict = {
        "image_id": result.image_id,
        "image_path": result.image_path,
        "status": result.status.value,
        "confidence_score": result.confidence_score,
        "issues": result.issues,
        "issue_elaboration": result.issue_elaboration,
        "original_reasoning": original_reasoning
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)


def generate_summary_report(results: List[VerificationResult], output_dir: Path):
    """Generate a summary report."""
    total = len(results)

    counts = {
        "completely_correct": sum(1 for r in results if r.status == VerificationStatus.COMPLETELY_CORRECT),
        "correct": sum(1 for r in results if r.status == VerificationStatus.CORRECT),
        "wrong": sum(1 for r in results if r.status == VerificationStatus.WRONG),
        "completely_wrong": sum(1 for r in results if r.status == VerificationStatus.COMPLETELY_WRONG),
        "error": sum(1 for r in results if r.status == VerificationStatus.ERROR)
    }

    avg_confidence = sum(r.confidence_score for r in results) / total if total > 0 else 0

    # Count issues
    issue_counts = {}
    for r in results:
        for issue in r.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)

    report = {
        "summary": {
            "total": total,
            **counts,
            "pass_rate": (counts["completely_correct"] + counts["correct"]) / total if total > 0 else 0,
            "average_confidence": avg_confidence
        },
        "issue_breakdown": dict(sorted_issues),
        "problematic_traces": [
            {"image_id": r.image_id, "image_path": r.image_path, "status": r.status.value, "issues": r.issues}
            for r in results
            if r.status in [VerificationStatus.WRONG, VerificationStatus.COMPLETELY_WRONG]
        ]
    }

    report_file = output_dir / "verification_summary.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info("VERIFICATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {total}")
    logger.info(f"Completely Correct: {counts['completely_correct']} ({100*counts['completely_correct']/total:.1f}%)" if total > 0 else "")
    logger.info(f"Correct: {counts['correct']} ({100*counts['correct']/total:.1f}%)" if total > 0 else "")
    logger.info(f"Wrong: {counts['wrong']} ({100*counts['wrong']/total:.1f}%)" if total > 0 else "")
    logger.info(f"Completely Wrong: {counts['completely_wrong']} ({100*counts['completely_wrong']/total:.1f}%)" if total > 0 else "")
    logger.info(f"Errors: {counts['error']}")
    logger.info(f"Pass Rate: {100*(counts['completely_correct']+counts['correct'])/total:.1f}%" if total > 0 else "")
    logger.info(f"{'='*60}")

    if sorted_issues:
        logger.info("Top Issues:")
        for issue, count in sorted_issues[:5]:
            logger.info(f"  {issue}: {count}")

    return report


def create_batches(traces: List[Dict], processed_ids: set, batch_size: int) -> List[List[Dict]]:
    """Create batches of traces to process."""
    remaining = [t for t in traces if t.get('image_id') not in processed_ids]
    batches = []
    for i in range(0, len(remaining), batch_size):
        batches.append(remaining[i:i + batch_size])
    return batches


def run_verification(
    api_key: str,
    model: str = "gpt-5-mini",
    reasoning_effort: str = "low",
    limit: Optional[int] = None,
    skip_existing: bool = True,
    batch_size: int = BATCH_SIZE
):
    """Run verification pipeline with batch processing."""

    logger.info(f"Model: {model} | Reasoning effort: {reasoning_effort} | Batch size: {batch_size}")

    client = OpenAI(api_key=api_key)
    traces = load_trace_files()

    if limit:
        random.shuffle(traces)
        traces = traces[:limit]
        logger.info(f"Randomly selected {limit} traces")

    # Get already processed
    processed_ids = set()
    if skip_existing:
        for f in VERIFICATION_OUTPUT_DIR.glob("verify_*.json"):
            processed_ids.add(f.stem.replace("verify_", ""))
        logger.info(f"Skipping {len(processed_ids)} already verified")

    # Create lookup for original reasoning
    reasoning_lookup = {t['image_id']: t.get('reasoning', '') for t in traces}

    # Load image and mask path lookups from Real-IAD metadata
    image_path_lookup = load_image_path_lookup()
    mask_path_lookup = load_mask_path_lookup()

    batches = create_batches(traces, processed_ids, batch_size)
    total_remaining = sum(len(b) for b in batches)
    logger.info(f"To verify: {total_remaining} traces in {len(batches)} batches")

    if not batches:
        logger.info("Nothing to verify")
        return

    all_results = []
    errors = 0

    with tqdm(total=total_remaining, desc="Verifying") as pbar:
        for batch_idx, batch in enumerate(batches):
            try:
                batch_results = verify_batch(client, batch, image_path_lookup, mask_path_lookup, model, reasoning_effort)

                for result in batch_results.values():
                    # Add image path from lookup
                    result.image_path = image_path_lookup.get(result.image_id, '')
                    all_results.append(result)
                    original_reasoning = reasoning_lookup.get(result.image_id, '')
                    save_verification_result(result, original_reasoning, VERIFICATION_OUTPUT_DIR)

                    if result.status == VerificationStatus.ERROR:
                        errors += 1

                pbar.update(len(batch))

                correct_count = sum(1 for r in batch_results.values()
                                   if r.status in [VerificationStatus.COMPLETELY_CORRECT, VerificationStatus.CORRECT])
                pbar.set_postfix({
                    "batch": f"{batch_idx+1}/{len(batches)}",
                    "pass": f"{correct_count}/{len(batch)}"
                })

            except Exception as e:
                errors += len(batch)
                logger.error(f"Batch {batch_idx+1} failed: {e}")

                for item in batch:
                    img_id = item.get('image_id', 'unknown')
                    result = VerificationResult(
                        image_id=img_id,
                        status=VerificationStatus.ERROR,
                        issues=[],
                        confidence_score=0.0,
                        image_path=image_path_lookup.get(img_id, ''),
                        issue_elaboration=[]
                    )
                    all_results.append(result)
                    save_verification_result(result, item.get('reasoning', ''), VERIFICATION_OUTPUT_DIR)

                pbar.update(len(batch))

            time.sleep(1.0)

    generate_summary_report(all_results, VERIFICATION_OUTPUT_DIR)
    logger.info(f"Done. Results: {VERIFICATION_OUTPUT_DIR}")


def remove_error_results(output_dir: Path) -> int:
    """Remove all verification results with error status."""
    import gc

    # First pass: identify error files
    error_files = []
    for f in output_dir.glob("verify_*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            if data.get('status') == 'error':
                error_files.append(f)
        except Exception as e:
            logger.warning(f"Failed to read {f.name}: {e}")

    # Force garbage collection to release file handles
    gc.collect()
    time.sleep(0.5)

    # Second pass: delete error files
    removed = 0
    skipped = 0
    for f in error_files:
        try:
            f.unlink()
            removed += 1
        except PermissionError:
            skipped += 1
        except Exception as e:
            logger.warning(f"Failed to delete {f.name}: {e}")
            skipped += 1

    if skipped > 0:
        logger.warning(f"Could not delete {skipped} files (may be locked by OneDrive/antivirus)")

    return removed


def main():
    parser = argparse.ArgumentParser(description="Verify reasoning traces with GPT-5-mini")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--reasoning-effort", type=str, default="low", choices=["low", "medium", "high"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--remove-errors-first", action="store_true",
                        help="Remove all verification results with error status before running")

    args = parser.parse_args()

    if args.remove_errors_first:
        removed = remove_error_results(VERIFICATION_OUTPUT_DIR)
        logger.info(f"Removed {removed} error results")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.error("No API key. Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)

    run_verification(
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        limit=args.limit,
        skip_existing=not args.reprocess,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
