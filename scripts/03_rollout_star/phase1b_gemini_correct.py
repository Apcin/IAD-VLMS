#!/usr/bin/env python
"""Phase 0/1 Gemini trace correction client.

Reads needs_rewrite.json or needs_correction.json, sends each item to Gemini
via Vertex AI, receives a corrected reasoning trace in the same tag format,
writes a sharegpt-format JSON ready to merge into SFT data.

Two modes (--mode rewrite | correct):
  rewrite: trace is correct but vague → make grounding sharper, KEEP conclusion
  correct: trace is wrong → produce a correct trace using GT mask + answer

Inspired by gemini_judge_server_v2.py (Vertex AI client, image encoding,
retry cascade, JSON schema enforcement). This is a CLIENT (not a server).

Resumable: appends to output JSONL per item; re-runs skip done image_paths.

Usage:
    python phase0_gemini_correct.py \\
        --input out/needs_correction.json \\
        --output out/gemini_corrected.jsonl \\
        --mode correct \\
        --max_concurrent 4
"""

import argparse
import asyncio
import io
import json
import logging
import os
import re
import sys
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GEMINI-CORRECT] %(message)s")
logger = logging.getLogger(__name__)

KEY_FILE = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/bulk/aacudad/reasoning_traces/gemini key/vertex-ai-key.json",
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_FILE

GEMINI_MODELS = [
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
]
RETRY_DELAYS = [1, 1, 1, 5, 10]
VERTEX_LOCATION = "global"

# ── Prompts ──────────────────────────────────────────────────────────────────

REWRITE_SYSTEM = """You are reviewing a reasoning trace for an industrial anomaly-detection image. The trace already arrives at the CORRECT conclusion. Your job is to make edits of MINIMUM NECESSARY SCOPE so the trace is well-grounded.

SCOPE OF EDITS — choose the right level for each input:
- If the input trace is mostly fine (vague phrasing, minor speculation, small grounding gaps): make SURGICAL EDITS. Keep nearly all of the original prose; only touch the specific words/sentences that are weak.
- If the input trace is fundamentally broken (describes a wrong camera view, invents components, hallucinates geometry that contradicts the image, fabricates the wrong defect): REWRITE THE THINK BLOCK FROM SCRATCH. A fresh grounded trace is better than surgically patching a hallucination.
Pick surgical by default; only go to a full rewrite when surgical edits cannot make the trace honest.

PRESERVATION PRINCIPLE (for surgical edits):
Treat the input trace as the baseline. Keep every sentence, observation, claim, ordering, and phrasing that is already accurate and grounded. Only edit what is specifically broken: a vague claim that needs to become specific, a speculative claim that should become observational, a colour/shape descriptor that doesn't match what is actually in the image.

HARD CONSTRAINTS:
1. <answer>, <type>, and <location> tags MUST be copied verbatim from the input. Do not change them, do not change their case, do not change their order.
2. The output <think> block MUST be no longer than the input <think> block. Shorter is fine. Expanding is forbidden.
3. Preserve the overall structure: the same number of sentences (±1), the same flow, the same observation order. Do not add new "inspection sweeps" or summarising sentences.
4. Do not introduce NEW observations, NEW components, NEW colours, NEW pad counts, NEW spatial details that were not in the input trace. The point is to fix what is wrong, not to add new information.
5. Do not add stock language like "no other defects are visible" or "the part meets all quality requirements" if the input doesn't already have it.
6. If the input trace is already fine, return it unchanged (minus any spelling fix). It is acceptable for the output to be identical to the input.

EDIT EXAMPLES (the kinds of edits that ARE allowed):
- "the red mark" → "the bright red smudge" (only if "bright" / "smudge" is visible in the image)
- "a defect was found" → "a contamination mark was found" (only if the trace later says it's contamination)
- "This could mean a manufacturing issue" → DELETE this sentence (speculation, not observation)
- "I see a deep scratch" → "I see a fine scratch" (only if the depth is actually wrong)

Return a single JSON object: {"corrected_trace": "<think>...</think>...<answer>...</answer>"}"""

CORRECT_SYSTEM = """You are correcting a reasoning trace for an industrial anomaly-detection image. You will receive: the image, the GROUND TRUTH answer, the GT defect type and location (for NG samples), the GT defect mask (for NG samples), and the model's failed attempt — a trace whose conclusion is wrong.

SCOPE OF EDITS — choose the right level for each input:
- If the failed attempt's prose is mostly grounded in the image but reached the wrong conclusion (e.g., missed a subtle defect, or flagged a feature that isn't a defect): make SURGICAL EDITS. Keep the good observations, edit only what needs to change to reach the correct conclusion.
- If the failed attempt is fundamentally hallucinated (describes a wrong view, invents geometry/components/threads/cores that don't exist in the image, fabricates the wrong defect type, contradicts what is visible at almost every step): WRITE A FRESH TRACE. Do not try to salvage hallucinated prose.
Default to surgical; switch to a fresh trace only when the failed attempt is so wrong that surgical edits cannot fix it.

PRESERVATION PRINCIPLE (for surgical edits):
Treat the failed attempt as the baseline. Keep every sentence, observation, and claim from the failed attempt that is FACTUALLY CORRECT about the image, even if the attempt's conclusion is wrong. Edit only what must change so the trace ends with the correct conclusion.

WHAT TO CHANGE:
- The conclusion sentence (so the answer matches the GT).
- Any specific claim in the failed attempt that contradicts what is actually visible (e.g., the attempt says "no contamination" but a contamination mark is visible — flip that one sentence).
- For NG samples: if the failed attempt missed the defect, ADD one or two sentences describing the defect at the correct location (using the GT mask as your guide), in the same prose style as the rest of the trace. Don't write a full new inspection.
- For OK samples: if the failed attempt invented a defect, DELETE or correct only those sentences. Don't rewrite the rest.

HARD CONSTRAINTS:
1. The <answer> tag MUST match the GT answer (yes or no).
2. For NG samples, <location> MUST equal the GT location string and <type> MUST equal the GT type string, both verbatim and in the same case as provided.
3. For OK samples, the output MUST be just <think>...</think><answer>no</answer> with no <type> or <location> tags.
4. The output <think> block MUST be no longer than the failed attempt's <think> block by more than 30%. Try to match it.
5. Preserve as much of the failed attempt's prose, observations, and structure as is compatible with reaching the correct conclusion.
6. Don't reference the failed attempt or admit a previous mistake — the corrected trace should read as the first and only attempt.
7. Don't introduce stock language ("no other defects visible", "the part meets all standards") unless it was already in the failed attempt.

Return a single JSON object: {"corrected_trace": "<think>...</think>...<answer>...</answer>"}"""


# Strict response schema — Gemini will only return valid JSON matching this
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_trace": {"type": "string"},
    },
    "required": ["corrected_trace"],
}

# ── State ────────────────────────────────────────────────────────────────────
client = None
VERTEX_PROJECT = None


def init_client():
    global client, VERTEX_PROJECT
    from google import genai

    with open(KEY_FILE) as f:
        key_data = json.load(f)
    VERTEX_PROJECT = key_data["project_id"]
    client = genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
    )
    logger.info(f"Vertex client init: project={VERTEX_PROJECT}")


# ── Image helpers ────────────────────────────────────────────────────────────
def image_to_jpeg_bytes(img_path: str, max_dim: int = 1024) -> bytes:
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def find_gt_mask(image_path: str) -> str:
    """Real-IAD GT mask convention: same dir as image, same basename, .png instead of .jpg.
    e.g. /.../zipper_0101_NG_ZW_C1_20231008154826.jpg → /.../zipper_0101_NG_ZW_C1_20231008154826.png
    """
    candidate = image_path.rsplit(".", 1)[0] + ".png"
    if os.path.exists(candidate):
        return candidate
    return ""


# ── Per-item correction ─────────────────────────────────────────────────────
def correct_item_sync(item: dict, mode: str) -> dict:
    """Returns {'image_path', 'corrected_trace', 'model_used', 'success'} or
    {'image_path', 'error', 'success': False}"""
    from google.genai.types import Content, Part, GenerateContentConfig

    if mode == "rewrite":
        system = REWRITE_SYSTEM
        user_text_parts = [
            f"Original (correct but weak) trace:\n{item['original_trace']}\n",
            f"Ground truth answer: {item['gt_answer']}\n",
        ]
    elif mode == "correct":
        system = CORRECT_SYSTEM
        # Extract GT type/location from the gt_trace
        gt_type_match = re.search(r"<type>(.*?)</type>", item["gt_trace"], re.DOTALL)
        gt_loc_match = re.search(r"<location>(.*?)</location>", item["gt_trace"], re.DOTALL)
        gt_type = gt_type_match.group(1).strip() if gt_type_match else "unknown"
        gt_loc = gt_loc_match.group(1).strip() if gt_loc_match else "unknown"
        user_text_parts = [
            f"Ground truth answer: {item['gt_answer']}\n",
        ]
        if item["gt_answer"] == "yes":
            user_text_parts += [
                f"Ground truth defect type: {gt_type}\n",
                f"Ground truth defect location: {gt_loc}\n",
                "(A defect mask image is included after the source image: white pixels = defect region.)\n"
                if find_gt_mask(item["image_path"]) else "",
            ]
        user_text_parts.append(f"Model's failed attempt:\n{item['best_failed_attempt']}\n")
    else:
        return {"image_path": item["image_path"], "error": f"unknown mode {mode}", "success": False}

    # Build content parts: source image + (optional mask) + text
    parts = [Part.from_bytes(data=image_to_jpeg_bytes(item["image_path"]), mime_type="image/jpeg")]
    if mode == "correct" and item["gt_answer"] == "yes":
        mask_path = find_gt_mask(item["image_path"])
        if mask_path:
            parts.append(Part.from_bytes(data=image_to_jpeg_bytes(mask_path), mime_type="image/jpeg"))
    parts.append(Part.from_text(text="".join(user_text_parts)))
    contents = [Content(role="user", parts=parts)]

    cfg = GenerateContentConfig(
        system_instruction=system,
        temperature=0.3,  # low temperature for grounded outputs
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    # Try models in cascade with retries
    for model_name in GEMINI_MODELS:
        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                resp = client.models.generate_content(
                    model=model_name, contents=contents, config=cfg,
                )
                raw = resp.text.strip()
                obj = json.loads(raw)
                corrected = obj.get("corrected_trace", "").strip()
                if not corrected:
                    raise ValueError("empty corrected_trace")
                # Sanity check: tag structure
                if "<answer>" not in corrected or "<think>" not in corrected:
                    raise ValueError(f"missing required tags: {corrected[:200]}")
                return {
                    "image_path": item["image_path"],
                    "corrected_trace": corrected,
                    "model_used": model_name,
                    "mode": mode,
                    "success": True,
                }
            except Exception as e:
                logger.warning(f"[{model_name} att{attempt+1}/{len(RETRY_DELAYS)}] {item['image_path']}: {e}")
                if attempt < len(RETRY_DELAYS) - 1:
                    import time
                    time.sleep(delay)
        logger.warning(f"Cascading to next model after {model_name} exhausted")

    return {"image_path": item["image_path"], "error": "all retries+cascade exhausted", "success": False}


async def correct_item_async(item, mode):
    return await asyncio.get_event_loop().run_in_executor(None, correct_item_sync, item, mode)


async def main_async(args):
    init_client()

    with open(args.input) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items from {args.input}")

    if args.limit:
        items = items[:args.limit]
        print(f"Limited to {len(items)} items")

    # Resume: skip items already in output
    done_paths = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("success"):
                        done_paths.add(rec["image_path"])
                except: pass
    if done_paths:
        items = [it for it in items if it["image_path"] not in done_paths]
        print(f"Resuming: {len(done_paths)} already done, {len(items)} remaining")

    fout = open(args.output, "a")
    sem = asyncio.Semaphore(args.max_concurrent)

    async def process(item):
        async with sem:
            result = await correct_item_async(item, args.mode)
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            return result

    tasks = [process(it) for it in items]
    done_count = 0
    fail_count = 0
    for fut in asyncio.as_completed(tasks):
        res = await fut
        if res.get("success"):
            done_count += 1
        else:
            fail_count += 1
        if (done_count + fail_count) % 10 == 0:
            print(f"  Progress: {done_count} OK, {fail_count} FAIL", flush=True)

    fout.close()
    print(f"\nDone. {done_count} succeeded, {fail_count} failed. Output: {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="needs_rewrite.json or needs_correction.json from phase0_bucket.py")
    p.add_argument("--output", required=True,
                   help="JSONL — one record per processed item")
    p.add_argument("--mode", choices=["rewrite", "correct"], required=True)
    p.add_argument("--max_concurrent", type=int, default=4,
                   help="Concurrent Gemini calls (Vertex API rate limit aware)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only N items (pilot)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
