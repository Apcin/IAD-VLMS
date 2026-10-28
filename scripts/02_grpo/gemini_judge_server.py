"""
Gemini Vision Judge HTTP Server
================================
Uses Gemini 3 Flash Preview via Vertex AI to score reasoning traces
with actual image grounding. Sends all completions for a prompt together
so Gemini can score them relatively.

POST /batch_grouped — JSON: {groups: [{image_path, gt_label, gt_type, gt_location, product, completions: [{think_text, model_answer}, ...]}]}
POST /health      — health check
GET  /health      — health check

Env vars:
  JUDGE_PORT             — port (default 5200)
  GOOGLE_APPLICATION_CREDENTIALS — path to vertex-ai-key.json
"""

import io
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GEMINI-JUDGE] %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("JUDGE_PORT", "5200"))

# ── Vertex AI credentials ──────────────────────────────────────────────────
KEY_FILE = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/bulk/aacudad/reasoning_traces/gemini key/vertex-ai-key.json",
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_FILE

VERTEX_PROJECT = None  # read from key file
VERTEX_LOCATION = "global"
GEMINI_MODEL = "google/gemini-3-flash-preview"

# ── Globals ──────────────────────────────────────────────────────────────────
client = None

# ── Retry config ─────────────────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0  # seconds
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_PARSE_RETRIES = 3  # extra retries specifically for JSON parse failures

# ── Scoring rubric ──────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert evaluator of industrial anomaly detection reasoning. You will receive an image of a product along with ground truth information and multiple AI-generated reasoning traces for the SAME image.

Your job is to evaluate each reasoning trace by examining whether the described observations actually match what is visible in the image.

Score each completion on these criteria:

1. **Visual Grounding** (0.0-3.0): Examine the image carefully, then evaluate the reasoning trace. Use any value in the range, not just integers (e.g. 1.5, 2.3, 0.8).
- Does the model correctly identify what product it is looking at? Misidentifying the product (e.g. calling a screw a "capsule") should heavily reduce the score.
- Does the model describe observations that are actually visible in the image?
- If there IS a visible defect in the image: did the model notice it? Missing an obvious defect is a major failure.
- If the model claims to see defects that are NOT visible in the image, that is hallucination.
- Around 0: Model misidentifies the product, OR misses a clearly visible defect, OR hallucinates defects/features not in the image.
- Around 1: Model identifies the product but catches only some visual features, misses the main defect or makes significant errors.
- Around 2: Model correctly identifies the product, describes the image accurately and catches the defect if present, with minor omissions.
- Around 3: Excellent — model correctly identifies the product, precisely describes what is visible, catches all defects if present, and does not hallucinate.

2. **Completeness of Inspection** (0.0-1.0): Did the model systematically check all relevant areas of the product, or did it only glance at one spot? Use any value in the range (e.g. 0.3, 0.7).
- Around 0: Model only mentions one area or gives a superficial inspection.
- Around 1: Model checked multiple relevant areas of the product systematically.

IMPORTANT: Score the completions RELATIVE to each other. Use the full range of scores — do not cluster all scores together. If one completion is clearly better, its score should be noticeably higher. Be strict.

You MUST return EXACTLY ONE JSON object and NOTHING else — no markdown, no explanation, no extra text before or after. The JSON must be a single object on one line. Example for 4 completions:
{"scores": [{"id": 0, "visual_grounding": 2.1, "completeness": 0.8, "total": 2.9, "normalized": 0.725}, {"id": 1, "visual_grounding": 0.5, "completeness": 0.2, "total": 0.7, "normalized": 0.175}, {"id": 2, "visual_grounding": 2.8, "completeness": 1.0, "total": 3.8, "normalized": 0.95}, {"id": 3, "visual_grounding": 1.3, "completeness": 0.5, "total": 1.8, "normalized": 0.45}]}

The "id" field corresponds to the completion index (0-based). The "total" is the sum of both criteria (max 4.0). The "normalized" field is total/4 rounded to 3 decimal places. Return one score object per completion. Do NOT return multiple JSON objects."""


def load_client():
    """Initialize the Gemini/Vertex AI client."""
    global client, VERTEX_PROJECT
    from google import genai

    # Read project ID from the key file
    with open(KEY_FILE) as f:
        key_data = json.load(f)
    VERTEX_PROJECT = key_data["project_id"]

    client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    logger.info(f"Gemini client initialized: project={VERTEX_PROJECT}, model={GEMINI_MODEL}")


def image_to_jpeg_bytes(img_path: str) -> bytes:
    """Load an image and convert to JPEG bytes."""
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def score_group(group: dict) -> list:
    """
    Score a group of completions for the same image.
    
    group = {
        "image_path": str,
        "gt_label": str,         # "yes" or "no"
        "gt_type": str,
        "gt_location": str,
        "product": str,
        "completions": [
            {"think_text": str, "model_answer": str},
            ...
        ]
    }
    
    Returns: list of float scores (0.0-1.0), one per completion.
    """
    from google.genai.types import Content, Part, GenerateContentConfig, ThinkingConfig, ThinkingLevel

    image_path = group["image_path"]
    gt_label = group.get("gt_label", "no")
    gt_type = group.get("gt_type", "")
    gt_location = group.get("gt_location", "")
    product = group.get("product", "unknown")
    completions = group.get("completions", [])

    if not completions:
        return []

    n = len(completions)
    default_scores = [0.5] * n

    # Build the text prompt with all completions
    text_parts = []
    text_parts.append(f"Product type: {product}")
    text_parts.append(f"Ground truth: {'ANOMALOUS (defective)' if gt_label == 'yes' else 'NORMAL (no defects)'}")
    if gt_label == "yes":
        if gt_type:
            text_parts.append(f"Ground truth defect type: {gt_type}")
        if gt_location:
            text_parts.append(f"Ground truth defect location: {gt_location}")
    text_parts.append("")
    text_parts.append(f"Below are {n} AI-generated reasoning traces for the above image. Score each one.")
    text_parts.append("")

    for i, comp in enumerate(completions):
        think = comp.get("think_text", "").strip()
        answer = comp.get("model_answer", "")
        text_parts.append(f"--- Completion {i} ---")
        text_parts.append(f"Model answer: {answer}")
        text_parts.append(f"Reasoning: {think}")
        text_parts.append("")

    user_text = "\n".join(text_parts)

    # Build multimodal parts: image first, then text
    parts = []

    # Add the product image
    try:
        img_bytes = image_to_jpeg_bytes(image_path)
        parts.append(Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return default_scores

    parts.append(Part.from_text(text=user_text))

    cfg = GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[Part.from_text(text=JUDGE_SYSTEM)],
        temperature=0.1,  # near-deterministic for consistent scoring
        thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL),
    )

    # Retry loop with exponential backoff
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[Content(role="user", parts=parts)],
                config=cfg,
            )
            response_text = response.text or ""

            # Try parsing the JSON response
            scores = _parse_scores_json(response_text, n)
            if scores is not None:
                return scores

            # JSON parse failed — retry up to MAX_PARSE_RETRIES times
            last_error = f"JSON parse failed: {response_text[:300]}"
            if attempt < MAX_PARSE_RETRIES - 1:
                wait = 1.0 + attempt
                logger.warning(f"JSON parse error, retrying ({attempt+1}/{MAX_PARSE_RETRIES}), waiting {wait:.0f}s. Raw: {response_text[:200]}")
                time.sleep(wait)
                continue
            else:
                logger.error(f"JSON parse failed after {MAX_PARSE_RETRIES} attempts. Raw: {response_text[:300]}")
                return default_scores

        except Exception as e:
            last_error = e
            error_str = str(e)
            status_code = None
            
            # Try to extract HTTP status code
            code_match = re.search(r'(\d{3})', error_str)
            if code_match:
                status_code = int(code_match.group(1))

            if status_code in RETRYABLE_STATUS_CODES:
                wait = RETRY_BACKOFF_BASE ** attempt
                # For 429, check for Retry-After hint
                retry_after_match = re.search(r'retry.?after[:\s]*(\d+)', error_str, re.IGNORECASE)
                if retry_after_match:
                    wait = max(wait, int(retry_after_match.group(1)))
                logger.warning(f"Retryable error (status={status_code}), attempt {attempt+1}/{MAX_RETRIES}, waiting {wait:.1f}s: {error_str[:150]}")
                time.sleep(wait)
            else:
                logger.error(f"Non-retryable error scoring group: {error_str[:300]}")
                return default_scores

    logger.error(f"All {MAX_RETRIES} retries exhausted: {last_error}")
    return default_scores


def _parse_scores_json(response_text: str, n: int):
    """
    Robustly parse Gemini's JSON response into a list of float scores.
    Returns list of scores on success, None on failure.
    """
    # Strategy 1: Try direct json.loads (works when response_mime_type is respected)
    try:
        result = json.loads(response_text.strip())
        return _extract_scores(result, n)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract first complete JSON object using brace counting
    try:
        start = response_text.index('{')
        depth = 0
        for i in range(start, len(response_text)):
            if response_text[i] == '{':
                depth += 1
            elif response_text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = response_text[start:i+1]
                    result = json.loads(candidate)
                    return _extract_scores(result, n)
    except (ValueError, json.JSONDecodeError):
        pass

    # Strategy 3: Try each line individually (handles multi-line JSON objects separated by newlines)
    for line in response_text.strip().splitlines():
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                result = json.loads(line)
                if "scores" in result:
                    return _extract_scores(result, n)
            except json.JSONDecodeError:
                continue

    return None


def _extract_scores(result: dict, n: int) -> list:
    """Extract normalized scores from parsed JSON result."""
    scores_list = result.get("scores", [])
    scores = [0.5] * n
    for entry in scores_list:
        idx = entry.get("id", -1)
        if 0 <= idx < n:
            scores[idx] = float(entry.get("normalized", 0.5))
    return scores


class GeminiJudgeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "model": GEMINI_MODEL})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "model": GEMINI_MODEL})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception as e:
            self._respond(400, {"error": f"invalid JSON: {e}"})
            return

        if self.path == "/batch_grouped":
            self._handle_batch_grouped(body)
        else:
            self._respond(404, {"error": "not found"})

    def _handle_batch_grouped(self, body):
        """
        Score multiple groups. Each group = 1 image + N completions.
        Returns: {group_scores: [[score, ...], [score, ...], ...]}
        """
        groups = body.get("groups", [])
        if not groups:
            self._respond(400, {"error": "no groups provided"})
            return

        t0 = time.time()
        # Score all groups concurrently via thread pool
        with ThreadPoolExecutor(max_workers=min(len(groups), 8)) as pool:
            all_scores = list(pool.map(score_group, groups))

        elapsed = time.time() - t0
        logger.info(f"Scored {len(groups)} groups ({sum(len(s) for s in all_scores)} completions) in {elapsed:.1f}s")
        self._respond(200, {"group_scores": all_scores})

    def _respond(self, code, data):
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        except BrokenPipeError:
            logger.warning("Client disconnected before response was sent (BrokenPipeError)")

    def log_message(self, format, *args):
        # Suppress default access logs
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread so concurrent calls don't block."""
    daemon_threads = True


def main():
    load_client()

    # Quick connectivity test
    logger.info("Testing Gemini API connectivity...")
    try:
        from google.genai.types import Content, Part, GenerateContentConfig
        test_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[Content(role="user", parts=[Part.from_text(text="Reply with: OK")])],
            config=GenerateContentConfig(temperature=0.0),
        )
        logger.info(f"API test OK: {test_response.text.strip()[:50]}")
    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        sys.exit(1)

    server = ThreadedHTTPServer(("127.0.0.1", PORT), GeminiJudgeHandler)
    logger.info(f"Gemini Judge Server listening on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
