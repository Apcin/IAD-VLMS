"""
Gemini Vision Judge — FastAPI + Uvicorn
========================================
Async server using Gemini 3 Flash Preview via Vertex AI to score
reasoning traces with actual image grounding. All Gemini API calls
for a batch run concurrently via asyncio.

POST /batch_grouped — JSON body with groups to score
GET  /health       — health check

Env vars:
  JUDGE_PORT                     — port (default 5200)
  JUDGE_WORKERS                  — uvicorn worker count (default 4)
  GOOGLE_APPLICATION_CREDENTIALS — path to vertex-ai-key.json
"""

import asyncio
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GEMINI-JUDGE] %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("JUDGE_PORT", "5200"))
WORKERS = int(os.environ.get("JUDGE_WORKERS", "4"))

# ── Vertex AI credentials ──────────────────────────────────────────────────
KEY_FILE = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/bulk/aacudad/reasoning_traces/gemini key/vertex-ai-key.json",
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_FILE

VERTEX_PROJECT = None
VERTEX_LOCATION = "global"

# Model cascade: try primary first, fall back to 3.1-flash-lite on persistent failures
GEMINI_MODELS = [
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
]

# ── Globals ──────────────────────────────────────────────────────────────────
client = None  # google.genai.Client — initialized per-worker on startup
nomic_model = None  # Nomic embedding model — loaded once per worker
nomic_tokenizer = None

# ── Retry config ─────────────────────────────────────────────────────────────
# Per model: 3 fast retries (1s), then 5s, 10s backoff. Then cascade to next model.
RETRY_DELAYS = [1, 1, 1, 5, 10]  # delay AFTER each failed attempt
MAX_RETRIES = len(RETRY_DELAYS)
MAX_PARSE_RETRIES = 3

# ── Scoring rubric ──────────────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an expert evaluator of industrial anomaly detection reasoning. You will receive an image of a product along with ground truth information and multiple AI-generated reasoning traces for the SAME image.

For ANOMALOUS (defective) samples, you may also receive a ground truth segmentation mask image. In this mask, WHITE regions indicate the exact location and shape of the defect, and BLACK regions are normal. Use this mask to verify whether the model correctly identified the defect location and extent.

Your job is to evaluate each reasoning trace by examining whether the described observations actually match what is visible in the product image (and the ground truth mask when provided).

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
When a ground truth mask is provided, use it to verify defect location accuracy. If the model describes a defect in a location that does NOT match the white region in the mask, penalize accordingly. If the model misses the defect region shown in the mask, that is a major failure.

2. **Completeness of Inspection** (0.0-1.0): Did the model systematically check all relevant areas of the product, or did it only glance at one spot? Use any value in the range (e.g. 0.3, 0.7).
- Around 0: Model only mentions one area or gives a superficial inspection.
- Around 1: Model checked multiple relevant areas of the product systematically.

IMPORTANT: Score the completions RELATIVE to each other. Use the full range of scores — do not cluster all scores together. If one completion is clearly better, its score should be noticeably higher. Be strict.

You MUST return EXACTLY ONE JSON object and NOTHING else — no markdown, no explanation, no extra text before or after. The JSON must be a single object on one line. Example for 4 completions:
{"scores": [{"id": 0, "visual_grounding": 2.1, "completeness": 0.8, "total": 2.9, "normalized": 0.725}, {"id": 1, "visual_grounding": 0.5, "completeness": 0.2, "total": 0.7, "normalized": 0.175}, {"id": 2, "visual_grounding": 2.8, "completeness": 1.0, "total": 3.8, "normalized": 0.95}, {"id": 3, "visual_grounding": 1.3, "completeness": 0.5, "total": 1.8, "normalized": 0.45}]}

The "id" field corresponds to the completion index (0-based). The "total" is the sum of both criteria (max 4.0). The "normalized" field is total/4 rounded to 3 decimal places. Return one score object per completion. Do NOT return multiple JSON objects."""

# ── Response schema (enforces exact JSON structure, eliminates parse failures) ─
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "visual_grounding": {"type": "number"},
                    "completeness": {"type": "number"},
                    "total": {"type": "number"},
                    "normalized": {"type": "number"},
                },
                "required": ["id", "visual_grounding", "completeness", "total", "normalized"],
            },
        }
    },
    "required": ["scores"],
}

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Gemini Vision Judge")


@app.on_event("startup")
async def startup():
    global client, VERTEX_PROJECT, nomic_model, nomic_tokenizer
    import torch
    from google import genai
    from transformers import AutoModel, AutoTokenizer

    with open(KEY_FILE) as f:
        key_data = json.load(f)
    VERTEX_PROJECT = key_data["project_id"]
    client = genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
    )
    logger.info(f"Worker started: project={VERTEX_PROJECT}, models={GEMINI_MODELS}")

    # Load Nomic embedding model
    try:
        nomic_tokenizer = AutoTokenizer.from_pretrained(
            "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True,
        )
        nomic_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True, dtype=torch.float32,
        ).to("cpu")
        nomic_model.eval()
        logger.info("Nomic embedding model loaded")
    except Exception as e:
        logger.error(f"Failed to load Nomic model: {e}")

    # Quick connectivity test
    from google.genai.types import Content, Part, GenerateContentConfig
    for model_name in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[Content(role="user", parts=[Part.from_text(text="Reply with: OK")])],
                config=GenerateContentConfig(temperature=0.0),
            )
            logger.info(f"API test OK for {model_name}: {resp.text.strip()[:50]}")
        except Exception as e:
            logger.error(f"API test failed for {model_name}: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "models": GEMINI_MODELS, "workers": WORKERS, "nomic": nomic_model is not None}


@app.post("/embed_similarity")
async def embed_similarity(request: Request):
    """Compute cosine similarity between two texts using Nomic embeddings."""
    import torch
    body = await request.json()
    text1 = body.get("text1", "")
    text2 = body.get("text2", "")
    if not text1 or not text2:
        return {"similarity": 0.0}
    if nomic_model is None or nomic_tokenizer is None:
        return JSONResponse(status_code=503, content={"error": "Nomic model not loaded"})
    try:
        def _compute():
            t1 = f"search_query: {text1}"
            t2 = f"search_query: {text2}"
            inputs1 = nomic_tokenizer(t1, return_tensors="pt", truncation=True, max_length=512)
            inputs2 = nomic_tokenizer(t2, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out1 = nomic_model(**inputs1)
                out2 = nomic_model(**inputs2)
                # Mean pooling
                emb1 = (out1[0] * inputs1["attention_mask"].unsqueeze(-1)).sum(1) / inputs1["attention_mask"].sum(1, keepdim=True).clamp(min=1e-9)
                emb2 = (out2[0] * inputs2["attention_mask"].unsqueeze(-1)).sum(1) / inputs2["attention_mask"].sum(1, keepdim=True).clamp(min=1e-9)
                emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
                emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
                sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1).item()
            return sim
        sim = await asyncio.get_event_loop().run_in_executor(None, _compute)
        return {"similarity": sim}
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/batch_grouped")
async def batch_grouped(request: Request):
    body = await request.json()
    groups = body.get("groups", [])
    if not groups:
        return JSONResponse(status_code=400, content={"error": "no groups provided"})

    t0 = time.time()
    # Score ALL groups concurrently
    tasks = [asyncio.get_event_loop().run_in_executor(None, score_group, g) for g in groups]
    all_scores = await asyncio.gather(*tasks)
    all_scores = list(all_scores)

    elapsed = time.time() - t0
    total_completions = sum(len(s) for s in all_scores)
    logger.info(f"Scored {len(groups)} groups ({total_completions} completions) in {elapsed:.1f}s")
    return {"group_scores": all_scores}


# ── Scoring logic (sync, runs in thread pool) ───────────────────────────────

def image_to_jpeg_bytes(img_path: str) -> bytes:
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def score_group(group: dict) -> list:
    from google.genai.types import Content, Part, GenerateContentConfig, ThinkingConfig, ThinkingLevel

    image_path = group["image_path"]
    gt_mask_path = group.get("gt_mask_path", "")
    gt_label = group.get("gt_label", "no")
    gt_type = group.get("gt_type", "")
    gt_location = group.get("gt_location", "")
    product = group.get("product", "unknown")
    completions = group.get("completions", [])

    if not completions:
        return []

    n = len(completions)
    default_scores = [0.5] * n

    # Build text prompt
    text_parts = [f"Product type: {product}"]
    text_parts.append(f"Ground truth: {'ANOMALOUS (defective)' if gt_label == 'yes' else 'NORMAL (no defects)'}")
    if gt_label == "yes":
        if gt_type:
            text_parts.append(f"Ground truth defect type: {gt_type}")
        if gt_location:
            text_parts.append(f"Ground truth defect location: {gt_location}")
    text_parts.append("")

    # Indicate whether GT mask is provided
    has_gt_mask = False
    if gt_label == "yes" and gt_mask_path and os.path.isfile(gt_mask_path):
        text_parts.append("A ground truth segmentation mask is provided as the second image. White regions = defect, black regions = normal.")
        has_gt_mask = True

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

    # Multimodal parts: product image + optional GT mask + text
    parts = []
    try:
        img_bytes = image_to_jpeg_bytes(image_path)
        parts.append(Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return default_scores

    if has_gt_mask:
        try:
            mask_bytes = image_to_jpeg_bytes(gt_mask_path)
            parts.append(Part.from_bytes(data=mask_bytes, mime_type="image/jpeg"))
            logger.debug(f"Included GT mask: {gt_mask_path}")
        except Exception as e:
            logger.warning(f"Failed to load GT mask {gt_mask_path}: {e}")

    parts.append(Part.from_text(text=user_text))

    cfg = GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SCORE_SCHEMA,
        system_instruction=[Part.from_text(text=JUDGE_SYSTEM)],
        temperature=0.1,
        thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL),
    )

    # ── Retry loop with model cascade ──
    # Try each model in GEMINI_MODELS. Per model: 5 attempts with [1,1,1,5,10]s delays.
    # If all retries exhausted for one model, cascade to the next.
    last_error = None

    for model_idx, model_name in enumerate(GEMINI_MODELS):
        for attempt in range(MAX_RETRIES):
            try:
                response_text = ""
                t_start = time.time()
                for chunk in client.models.generate_content_stream(
                    model=model_name,
                    contents=[Content(role="user", parts=parts)],
                    config=cfg,
                ):
                    if chunk.text:
                        response_text += chunk.text
                elapsed = time.time() - t_start

                if not response_text:
                    last_error = f"Empty response from {model_name}"
                    logger.warning(f"{last_error}, attempt {attempt+1}/{MAX_RETRIES}")
                    time.sleep(RETRY_DELAYS[attempt])
                    continue

                scores = _parse_scores_json(response_text, n)
                if scores is not None:
                    if model_idx > 0:
                        logger.info(f"Succeeded with fallback model {model_name} in {elapsed:.1f}s")
                    return scores

                # JSON parse failed
                last_error = f"JSON parse failed on {model_name}: {response_text[:300]}"
                if attempt < MAX_PARSE_RETRIES - 1:
                    logger.warning(f"JSON parse error on {model_name}, attempt {attempt+1}/{MAX_RETRIES}. Raw: {response_text[:200]}")
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                else:
                    logger.error(f"JSON parse failed after {MAX_PARSE_RETRIES} attempts on {model_name}. Raw: {response_text[:300]}")
                    break  # cascade to next model

            except TypeError as e:
                last_error = e
                logger.warning(f"TypeError (likely 429 bug) on {model_name}, attempt {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAYS[attempt])

            except Exception as e:
                last_error = e
                error_str = str(e)
                error_type = type(e).__name__
                is_retryable = any(kw in error_str.lower() for kw in (
                    "429", "500", "502", "503", "504",
                    "resource_exhausted", "unavailable",
                    "timed out", "timeout", "too many",
                    "handshake", "read operation", "ssl",
                    "connection", "reset", "broken pipe",
                ))

                if is_retryable:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        f"Retryable [{error_type}] on {model_name}, "
                        f"attempt {attempt+1}/{MAX_RETRIES}, waiting {delay}s: {error_str[:300]}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Non-retryable [{error_type}] on {model_name}: {error_str[:500]}")
                    return default_scores

        # All retries exhausted for this model — cascade
        if model_idx < len(GEMINI_MODELS) - 1:
            logger.warning(f"All {MAX_RETRIES} retries exhausted on {model_name}, cascading to {GEMINI_MODELS[model_idx+1]}")

    logger.error(f"All models exhausted. Last error: {last_error}")
    return default_scores


def _parse_scores_json(response_text: str, n: int):
    try:
        result = json.loads(response_text.strip())
        return _extract_scores(result, n)
    except json.JSONDecodeError:
        pass

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
    scores_list = result.get("scores", [])
    scores = [0.5] * n
    for entry in scores_list:
        idx = entry.get("id", -1)
        if 0 <= idx < n:
            scores[idx] = float(entry.get("normalized", 0.5))
    return scores


if __name__ == "__main__":
    uvicorn.run(
        "gemini_judge_server_v2:app",
        host="127.0.0.1",
        port=PORT,
        workers=WORKERS,
        log_level="info",
    )
