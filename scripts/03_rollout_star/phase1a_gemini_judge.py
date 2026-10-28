#!/usr/bin/env python
"""Phase 1a — Gemini faithfulness judge (5-level rubric).

Reads judge_input.json (produced by phase0_bucket.py) — the items that
PASSED local reward — and asks Gemini whether each trace is faithful to
the actual image content or hallucinated.

Output:
  good_traces_faithful.json       — LLaMA-Factory sharegpt; passed the judge
                                    (score ≥ keep_threshold, default 2)
  needs_rewrite_from_judge.json   — items flagged for rewrite (score < threshold)
                                    Schema matches needs_rewrite.json from bucketer
                                    so Phase 1b can consume them uniformly
  judge_report.jsonl              — raw per-item Gemini outputs incl. rationale +
                                    grounded observations + hallucination flags

Resumable: append-mode JSONL; re-runs skip items already done.

Five-level rubric:
  0 HALLUCINATED   — describes things not in image / misidentifies product
                     / for NG: invents or completely mislocates the defect
  1 POOR           — generic placeholder reasoning, vague claims, weak grounding
                     / type or location significantly off from GT
  2 ACCEPTABLE     — correct product, some accurate observations, defect roughly OK
  3 GOOD           — multiple specific accurate observations grounded in image
                     / for NG: defect well-described with reasonable specificity
  4 EXCELLENT      — precise image-grounded descriptions; for NG: defect location
                     matches mask, type precise, reasoning chains observations
                     to conclusion; for OK: systematic inspection of all regions
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GEMINI-JUDGE] %(message)s")
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

# ── 5-level rubric prompt ────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator of industrial-anomaly-detection (IAD) reasoning traces.

You receive:
  - The product image (always)
  - The ground-truth defect mask image — WHITE pixels mark the defect region — ONLY for anomalous samples; if absent, the sample is normal
  - The ground-truth answer (yes = anomaly present, no = no anomaly), and for anomalous samples the ground-truth defect type and location
  - A reasoning trace produced by a vision-language model

The trace's conclusion may or may not match the ground truth. Your job is to judge whether the trace's <think> block is FAITHFUL to what is actually visible in the image — i.e., does it describe what is really there — independent of whether the final <answer> tag matches the ground truth. A trace can be faithful but wrong (genuinely looked at the image and inferred wrong), or unfaithful but accidentally right (hallucinated reasoning that happened to land on the right conclusion). Score on faithfulness ONLY.

Score on this 5-level rubric:

  0 = HALLUCINATED
      The trace describes things NOT present in the image, OR misidentifies
      the product (e.g. calls a screw a "capsule"), OR for anomalous samples
      claims a completely different defect than the GT mask shows.

  1 = POOR
      Generic placeholder reasoning ("there is something on the surface").
      Product loosely identified. For anomalous samples: defect is noted but
      type/location is significantly off from GT. Confident but ungrounded.

  2 = ACCEPTABLE
      Correctly identifies product. Some accurate observations of visible
      features. For anomalous samples: defect identified, type/location roughly
      correct. For normal samples: notes the inspected areas but somewhat
      superficial.

  3 = GOOD
      Multiple specific accurate observations grounded in the actual image.
      For anomalous samples: defect well-described with reasonable spatial
      and visual specificity, type matches GT. For normal samples: thorough
      inspection covering multiple regions with explicit evidence.

  4 = EXCELLENT
      Precise, image-grounded throughout. For anomalous samples: defect
      location exactly matches mask, type is precise, reasoning chains
      visible observations directly to the conclusion. For normal samples:
      systematic inspection of all relevant regions with explicit per-region
      "no defect" evidence.

Be strict. Most traces should fall in 1–3. Reserve 0 for genuine hallucinations and 4 for traces that are truly excellent.

Return ONE JSON object exactly matching this schema (no markdown, no extra text):
{
  "faithfulness_score": <integer 0-4>,
  "faithfulness_label": "HALLUCINATED" | "POOR" | "ACCEPTABLE" | "GOOD" | "EXCELLENT",
  "rationale": "<1-2 sentence explanation>",
  "hallucination_flags": ["<specific issue 1>", "<specific issue 2>"],
  "grounded_observations": ["<accurately-described feature 1>", "..."]
}

Limit each list to at most 3 items. Use empty arrays when not applicable."""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness_score": {"type": "integer"},
        "faithfulness_label": {"type": "string"},
        "rationale": {"type": "string"},
        "hallucination_flags": {"type": "array", "items": {"type": "string"}},
        "grounded_observations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["faithfulness_score", "faithfulness_label", "rationale",
                 "hallucination_flags", "grounded_observations"],
}

# ── Vertex client ────────────────────────────────────────────────────────────
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
    """Real-IAD convention: mask is same dir as image, same basename, .png instead of .jpg.
    e.g. /.../images/zipper/NG/ZW/S0101/zipper_0101_NG_ZW_C1_20231008154826.jpg
      → /.../images/zipper/NG/ZW/S0101/zipper_0101_NG_ZW_C1_20231008154826.png
    """
    candidate = image_path.rsplit(".", 1)[0] + ".png"
    if os.path.exists(candidate):
        return candidate
    # Fallback conventions (older datasets)
    for alt in [
        image_path.replace("/images/", "/masks/").rsplit(".", 1)[0] + ".png",
        image_path.rsplit(".", 1)[0] + "_mask.png",
    ]:
        if os.path.exists(alt):
            return alt
    return ""


# ── Per-item judge ──────────────────────────────────────────────────────────
def judge_item_sync(item: dict) -> dict:
    """Returns merged dict {input_meta..., judge_output...} on success,
    or {image_path, error, success: False}."""
    from google.genai.types import Content, Part, GenerateContentConfig

    # Build user-content parts
    parts = [Part.from_bytes(data=image_to_jpeg_bytes(item["image_path"]),
                             mime_type="image/jpeg")]
    if item["is_anomaly"]:
        mask_path = find_gt_mask(item["image_path"])
        if mask_path:
            parts.append(Part.from_bytes(data=image_to_jpeg_bytes(mask_path),
                                         mime_type="image/jpeg"))

    user_text_lines = [
        f"Product: {item.get('product','unknown')}",
        f"Ground truth answer: {item['gt_answer']}",
    ]
    if item["is_anomaly"]:
        user_text_lines.append(f"Ground truth defect type: {item.get('gt_type') or 'unknown'}")
        user_text_lines.append(f"Ground truth defect location: {item.get('gt_location') or 'unknown'}")
        mask_present = bool(find_gt_mask(item["image_path"]))
        if mask_present:
            user_text_lines.append("A defect mask image is included after the source image: WHITE = defect region.")
        else:
            user_text_lines.append("(No GT mask image available; judge based on type+location strings.)")
    user_text_lines.append("")
    user_text_lines.append("Reasoning trace to judge:")
    user_text_lines.append(item["chosen_trace"])
    parts.append(Part.from_text(text="\n".join(user_text_lines)))

    contents = [Content(role="user", parts=parts)]
    cfg = GenerateContentConfig(
        system_instruction=JUDGE_SYSTEM,
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    for model_name in GEMINI_MODELS:
        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                resp = client.models.generate_content(
                    model=model_name, contents=contents, config=cfg,
                )
                raw = resp.text.strip()
                obj = json.loads(raw)

                # Validate
                score = int(obj["faithfulness_score"])
                if score < 0 or score > 4:
                    raise ValueError(f"score out of range: {score}")
                label = obj["faithfulness_label"].upper()

                return {
                    "image_path":         item["image_path"],
                    "product":            item.get("product"),
                    "gt_answer":          item["gt_answer"],
                    "is_anomaly":         item["is_anomaly"],
                    "chosen_trace":       item["chosen_trace"],
                    "chosen_acc":         item.get("chosen_acc"),
                    "chosen_type_score":  item.get("chosen_type_score"),
                    "chosen_loc_score":   item.get("chosen_loc_score"),
                    "difficulty":         item.get("difficulty"),
                    "source_pool":        item.get("source_pool"),
                    "user_prompt_template": item.get("user_prompt_template"),
                    "gt_trace":           item.get("gt_trace"),
                    "gt_type":            item.get("gt_type"),
                    "gt_location":        item.get("gt_location"),
                    "local_bucket":       item.get("local_bucket", "good"),
                    "faithfulness_score": score,
                    "faithfulness_label": label,
                    "rationale":          obj.get("rationale", ""),
                    "hallucination_flags": obj.get("hallucination_flags", []),
                    "grounded_observations": obj.get("grounded_observations", []),
                    "model_used":         model_name,
                    "success":            True,
                }
            except Exception as e:
                logger.warning(f"[{model_name} att{attempt+1}/{len(RETRY_DELAYS)}] {item['image_path']}: {e}")
                if attempt < len(RETRY_DELAYS) - 1:
                    import time; time.sleep(delay)
        logger.warning(f"Cascading to next model after {model_name} exhausted")

    return {"image_path": item["image_path"], "error": "all retries+cascade exhausted", "success": False}


async def judge_item_async(item):
    return await asyncio.get_event_loop().run_in_executor(None, judge_item_sync, item)


# ── Output splitter ─────────────────────────────────────────────────────────
def to_sharegpt(item):
    """LLaMA-Factory sharegpt format using the chosen trace as assistant message."""
    return {
        "messages": [
            {"role": "user",      "content": f"<image>\n{item['user_prompt_template']}"},
            {"role": "assistant", "content": item["chosen_trace"]},
        ],
        "images": [item["image_path"]],
    }


def to_rewrite_record(item):
    """Schema compatible with phase0_bucket.py's needs_rewrite.json so that
    Phase 1b can consume both sources uniformly."""
    return {
        "image_path": item["image_path"],
        "product": item.get("product"),
        "gt_trace": item.get("gt_trace"),
        "gt_answer": item["gt_answer"],
        "gt_type": item.get("gt_type"),
        "gt_location": item.get("gt_location"),
        "is_anomaly": item["is_anomaly"],
        "source_pool": item.get("source_pool"),
        "user_prompt_template": item.get("user_prompt_template"),
        "original_trace": item["chosen_trace"],
        "original_acc": item.get("chosen_acc"),
        "flagged_by": "gemini_judge",
        "judge_score": item["faithfulness_score"],
        "judge_label": item["faithfulness_label"],
        "judge_rationale": item.get("rationale", ""),
        "hallucination_flags": item.get("hallucination_flags", []),
        "difficulty": item.get("difficulty"),
    }


# ── Main async loop ────────────────────────────────────────────────────────
async def main_async(args):
    init_client()

    with open(args.input) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items from {args.input}")

    if args.limit:
        items = items[:args.limit]
        print(f"Limited to {len(items)} items")

    # Resume: skip already-judged
    done_paths = set()
    if os.path.exists(args.report_jsonl):
        with open(args.report_jsonl) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("success"):
                        done_paths.add(rec["image_path"])
                except Exception:
                    pass
    if done_paths:
        items = [it for it in items if it["image_path"] not in done_paths]
        print(f"Resuming: {len(done_paths)} already judged, {len(items)} remaining")

    fout_report = open(args.report_jsonl, "a")
    sem = asyncio.Semaphore(args.max_concurrent)

    score_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, "fail": 0}

    async def process(it):
        async with sem:
            res = await judge_item_async(it)
            fout_report.write(json.dumps(res) + "\n")
            fout_report.flush()
            if res.get("success"):
                score_counts[res["faithfulness_score"]] += 1
            else:
                score_counts["fail"] += 1
            return res

    tasks = [process(it) for it in items]
    cnt = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        cnt += 1
        if cnt % 10 == 0:
            print(f"  Progress: {cnt}/{len(items)} | by score: {score_counts}", flush=True)

    fout_report.close()

    # Now split, taking local_bucket into account:
    #   local_bucket="good" + score >= threshold  → faithful (SFT-ready)
    #   local_bucket="good" + score <  threshold  → needs_rewrite (judge flagged correct-but-unfaithful)
    #   local_bucket="correction"                 → needs_correction_judged (regardless of score; conclusion is wrong)
    # Need to re-load judge_input to pull local_bucket if it's missing from older records.
    print("\nReading full report and splitting (local_bucket-aware)…")

    # Look up local_bucket by image_path from input (for backward compat with older report rows
    # that didn't carry local_bucket through)
    with open(args.input) as f:
        bucket_by_path = {it["image_path"]: it.get("local_bucket", "good") for it in json.load(f)}

    faithful_sharegpt = []
    rewrite_records = []
    correction_judged_records = []

    with open(args.report_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("success"):
                continue
            score = rec["faithfulness_score"]
            lb = rec.get("local_bucket") or bucket_by_path.get(rec["image_path"], "good")

            if lb == "correction":
                # Wrong-conclusion item — Gemini correction needed regardless of faithfulness
                # Attach judge rationale to the record so HTML / Phase 1b can use it
                correction_judged_records.append({
                    "image_path": rec["image_path"],
                    "product": rec.get("product"),
                    "gt_trace": rec.get("gt_trace"),
                    "gt_answer": rec["gt_answer"],
                    "gt_type": rec.get("gt_type"),
                    "gt_location": rec.get("gt_location"),
                    "is_anomaly": rec["is_anomaly"],
                    "source_pool": rec.get("source_pool"),
                    "user_prompt_template": rec.get("user_prompt_template"),
                    "best_failed_attempt": rec["chosen_trace"],
                    "best_failed_acc": rec.get("chosen_acc"),
                    "best_failed_format": None,  # we don't have it here; fine
                    "difficulty": rec.get("difficulty"),
                    "judge_score": score,
                    "judge_label": rec["faithfulness_label"],
                    "judge_rationale": rec.get("rationale", ""),
                    "hallucination_flags": rec.get("hallucination_flags", []),
                    "grounded_observations": rec.get("grounded_observations", []),
                })
                continue

            # local_bucket == "good"
            if score >= args.keep_threshold:
                faithful_sharegpt.append(to_sharegpt(rec))
            else:
                rewrite_records.append(to_rewrite_record(rec))

    with open(args.output_faithful, "w") as f:
        json.dump(faithful_sharegpt, f, indent=2)
    with open(args.output_rewrite, "w") as f:
        json.dump(rewrite_records, f, indent=2)
    # Additional output for correction items with judge attached
    correction_judged_path = args.output_rewrite.replace("needs_rewrite_from_judge", "needs_correction_judged")
    if correction_judged_path == args.output_rewrite:
        correction_judged_path = os.path.join(os.path.dirname(args.output_rewrite), "needs_correction_judged.json")
    with open(correction_judged_path, "w") as f:
        json.dump(correction_judged_records, f, indent=2)

    print(f"\nDone. Final counts:")
    print(f"  Faithful (local=good + score≥{args.keep_threshold}): {len(faithful_sharegpt)} → {args.output_faithful}")
    print(f"  Rewrite  (local=good + score<{args.keep_threshold}): {len(rewrite_records)} → {args.output_rewrite}")
    print(f"  Correction (local=correction, any score):           {len(correction_judged_records)} → {correction_judged_path}")
    print(f"  Full per-item report: {args.report_jsonl}")
    print(f"  Score distribution: {score_counts}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="judge_input.json from phase0_bucket.py")
    p.add_argument("--output_faithful", required=True,
                   help="Output JSON: items with judge_score ≥ keep_threshold "
                        "(LLaMA-Factory sharegpt, ready for SFT)")
    p.add_argument("--output_rewrite", required=True,
                   help="Output JSON: items with judge_score < keep_threshold "
                        "(merge into Phase 1b rewrite pile)")
    p.add_argument("--report_jsonl", required=True,
                   help="Per-item full Gemini outputs (rationale + flags + observations)")
    p.add_argument("--keep_threshold", type=int, default=2,
                   help="Score ≥ this → faithful; below → rewrite. Default 2.")
    p.add_argument("--max_concurrent", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
