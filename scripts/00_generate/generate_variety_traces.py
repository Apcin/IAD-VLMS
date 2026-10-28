"""
Real-IAD Variety reasoning-trace generation (C1 angle), Gemini via Vertex AI.

Adapted from generate_realiad_traces_v4_10k.py. Same prompt + trace format
(inspector_prompt_test_v2.txt, <think>/<location>/<type>/<answer>, 120-200 words),
but reads the Variety splits produced by sample_variety.py and handles the
Variety on-disk layout:
  - images are .png (not .jpg)
  - mask is a sibling  <image_stem>_mask.png   (not <stem>.png)
  - path layout: data/realiad-variety/<category>/{OK|NG/<DefectType>}/S####/...png

Usage:
  python generate_variety_traces.py --shard_id 0 --total_shards 4
  python generate_variety_traces.py --debug
"""
import argparse
import io
import json
import logging
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import (Content, GenerateContentConfig, Part,
                                ThinkingConfig, ThinkingLevel)
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

import config_mmad as config
import utils_realiad as utils

load_dotenv()

# ---------------------------------------------------------------------------
# Vertex AI credentials
# ---------------------------------------------------------------------------
# Both come from the environment, see .env.example. Nothing secret lives in this file.
_KEY_FILE = Path(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                 or Path(__file__).parent / "vertex-service-account.json")
if _KEY_FILE.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_KEY_FILE)
else:
    raise FileNotFoundError(f"Service account key not found: {_KEY_FILE}")

VERTEX_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
if not VERTEX_PROJECT:
    raise SystemExit("Set GOOGLE_CLOUD_PROJECT (and GOOGLE_APPLICATION_CREDENTIALS); see .env.example")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGES_ROOT = config.DATA_DIR / "realiad-variety"
SAMPLE_DIR  = Path(__file__).parent / "variety_c1_compiled"
BASE_OUT    = config.DATA_DIR / "output" / "reasoning_traces_variety"
GRPO_DIR    = BASE_OUT / "grpo"
SFT_DIR     = BASE_OUT / "sft"
GRPO_DIR.mkdir(parents=True, exist_ok=True)
SFT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE   = config.BATCH_SIZE          # 10
GEMINI_MODEL = "google/gemini-3-flash-preview"

DEFECT_NORMALISE = {"porosity": "Porosity"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(shard_id=None):
    suffix = str(shard_id) if shard_id is not None else "main"
    log_file = config.LOGS_DIR / f"variety_{suffix}.log"
    fmt = (f"%(asctime)s [Shard {shard_id}] %(levelname)s %(message)s"
           if shard_id is not None else "%(asctime)s %(levelname)s %(message)s")
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
                        force=True)
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ReasoningTrace(BaseModel):
    image_id: str
    reasoning: str


class BatchReasoningTraces(BaseModel):
    traces: List[ReasoningTrace]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_system_prompt() -> str:
    prompt_file = config.PROMPTS_DIR / "inspector_prompt_test_v2.txt"
    try:
        return prompt_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"Prompt not found: {prompt_file}")
        sys.exit(1)


_GEN_MARKER = "reasoning_traces_gen/"

def _linux_to_local(linux_path: str) -> Path:
    """Map a cluster path (.../reasoning_traces_gen/data/realiad-variety/...) to local."""
    idx = linux_path.find(_GEN_MARKER)
    if idx == -1:
        # already a local/relative path
        return Path(linux_path)
    rel = linux_path[idx + len(_GEN_MARKER):].replace("/", os.sep)
    return Path(__file__).parent / rel


def mask_for(img_path: Path) -> Path:
    """Variety masks are siblings named <image_stem>_mask.png."""
    return img_path.parent / f"{img_path.stem}_mask.png"


def defect_from_path(img_path: Path) -> str:
    """NG/<DefectType>/S####/file.png -> normalised <DefectType>."""
    try:
        parts = img_path.relative_to(IMAGES_ROOT).parts   # (category, NG, Defect, S####, file)
        defect = parts[2]
    except (ValueError, IndexError):
        return "Unknown"
    return DEFECT_NORMALISE.get(defect, defect)


def category_from_path(img_path: Path) -> str:
    try:
        return img_path.relative_to(IMAGES_ROOT).parts[0]
    except (ValueError, IndexError):
        return "unknown"


def _derive_image_id(img_path: Path) -> str:
    try:
        rel = img_path.relative_to(IMAGES_ROOT)
        return "_".join(list(rel.parts[:-1]) + [rel.stem])
    except ValueError:
        return img_path.stem


def get_normal_reference_image(category: str, camera_hint: Optional[int] = None) -> Optional[Path]:
    """A random OK image for this category, preferring the same camera angle."""
    ok_root = IMAGES_ROOT / category / "OK"
    if not ok_root.exists():
        return None
    all_ok = [p for p in ok_root.rglob("*.png") if "_mask" not in p.name]
    if not all_ok:
        return None
    if camera_hint is not None:
        cam_tag = f"_C{camera_hint}_"
        matching = [p for p in all_ok if cam_tag in p.name]
        if matching:
            return random.choice(matching)
    return random.choice(all_ok)


def load_grpo_questions():
    grpo_file = SAMPLE_DIR / "grpo_train.json"
    entries = json.loads(grpo_file.read_text(encoding="utf-8"))
    ng, ok, skipped = [], [], 0
    for entry in entries:
        img_path = _linux_to_local(entry["image_path"])
        if not img_path.exists():
            skipped += 1
            continue
        cam_match = re.search(r"_C(\d+)_", img_path.name)
        base = {
            "image_id": entry["image_id"],
            "image_path": img_path,
            "category": entry["product"],
            "camera_hint": int(cam_match.group(1)) if cam_match else None,
        }
        if entry["is_anomaly"]:
            mp = mask_for(img_path)
            if not mp.exists():
                skipped += 1
                continue
            ng.append({**base, "mask_path": mp, "defect_type": defect_from_path(img_path)})
        else:
            ok.append({**base, "mask_path": None, "defect_type": "None"})
    logger.info(f"grpo: {len(ng)} NG + {len(ok)} OK  (skipped {skipped})")
    return ng, ok


def load_sft_questions():
    sft_file = SAMPLE_DIR / "sft_train.json"
    entries = json.loads(sft_file.read_text(encoding="utf-8"))
    ng, ok, skipped = [], [], 0
    for entry in entries:
        linux_path = entry["images"][0]
        img_path = _linux_to_local(linux_path)
        if not img_path.exists():
            skipped += 1
            continue
        cam_match = re.search(r"_C(\d+)_", img_path.name)
        base = {
            "image_id": _derive_image_id(img_path),
            "image_path": img_path,
            "category": category_from_path(img_path),
            "camera_hint": int(cam_match.group(1)) if cam_match else None,
        }
        if "/NG/" in linux_path or f"{os.sep}NG{os.sep}" in str(img_path):
            mp = mask_for(img_path)
            if not mp.exists():
                ok.append({**base, "mask_path": None, "defect_type": "None"})
                continue
            ng.append({**base, "mask_path": mp, "defect_type": defect_from_path(img_path)})
        else:
            ok.append({**base, "mask_path": None, "defect_type": "None"})
    logger.info(f"sft:  {len(ng)} NG + {len(ok)} OK  (skipped {skipped})")
    return ng, ok


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def open_rgb(path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB") if img.mode != "RGB" else img


def save_trace(image_id: str, reasoning: str, output_dir: Path):
    out = output_dir / f"trace_{image_id}.json"
    out.write_text(json.dumps({"image_id": image_id, "reasoning": reasoning}, indent=2), encoding="utf-8")


def parse_json_response(text: str) -> Dict:
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return json.loads(m.group(1) if m else text)


def normalise(d) -> Dict:
    if isinstance(d, list):
        d = {"traces": d}
    if "traces" not in d and "image_id" in d:
        d = {"traces": [d]}
    return d


# ---------------------------------------------------------------------------
# Prompt builders (identical wording to v4 -> XML trace, 120-200 words)
# ---------------------------------------------------------------------------
def format_anomaly_prompt(batch: List[Dict]) -> str:
    parts = [
        f"You will receive {len(batch)} samples to inspect.",
        "For EACH sample, you are provided with THREE images in the order they appear.",
        "",
        "Inspect each sample visually and generate a complete inspection record "
        "following the specified format.",
        "",
        "IMPORTANT: Each <think> block must contain 120-200 words of detailed "
        "visual reasoning. Do not abbreviate.",
        "",
        "CRITICAL: Your reasoning must read as a single-image visual inspection. "
        "Do not mention, imply, or reference the existence of multiple images, "
        "reference images, overlays, highlighted regions, or comparisons with "
        "other views. Write as though you are inspecting one product image only.",
        "",
    ]
    for idx, item in enumerate(batch, 1):
        parts.append(f"--- SAMPLE {idx} ---")
        parts.append(f"Image ID: {item['image_id']}")
        parts.append(f"Product Type: {item['category']}")
        parts.append("[INTERNAL — do not reference in reasoning] Ground Truth: ANOMALOUS")
        if item.get("defect_type"):
            parts.append(f"[INTERNAL] Defect Type: {item['defect_type']}")
        parts.append(
            "[INTERNAL] The red overlay (second image) highlights the region of interest. "
            "For the <location> tag, use a 3x3 grid: top-left, top-center, top-right, "
            "middle-left, center, middle-right, bottom-left, bottom-center, bottom-right. "
            "If the defect spans multiple regions, separate them with commas "
            "(e.g. top-center, top-left)."
        )
        parts.append("")
    parts.append("Return a JSON object with a 'traces' array containing one entry per sample.")
    return "\n".join(parts)


def format_normal_prompt(batch: List[Dict]) -> str:
    parts = [
        f"You will receive {len(batch)} samples to inspect.",
        "For EACH sample, you are provided with ONE image: the original product image.",
        "",
        "Inspect each sample visually and generate a complete inspection record "
        "following the specified format.",
        "",
        "IMPORTANT: Each <think> block must contain 120-200 words of detailed "
        "visual reasoning. Do not abbreviate.",
        "",
    ]
    for idx, item in enumerate(batch, 1):
        parts.append(f"--- SAMPLE {idx} ---")
        parts.append(f"Image ID: {item['image_id']}")
        parts.append(f"Product Type: {item['category']}")
        parts.append("[INTERNAL — do not reference in reasoning] Ground Truth: NORMAL")
        parts.append("")
    parts.append("Return a JSON object with a 'traces' array containing one entry per sample.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Gemini batch call
# ---------------------------------------------------------------------------
def process_batch(batch, client, model, system_prompt, normal=False, debug=False, shard_id=0):
    batch_id = str(uuid.uuid4())[:8]
    text = format_normal_prompt(batch) if normal else format_anomaly_prompt(batch)
    parts = [Part.from_text(text=text)]

    debug_dir = Path("debug_output_variety")
    prefix = f"shard{shard_id}_{'normal' if normal else 'anomaly'}"
    if debug:
        debug_dir.mkdir(exist_ok=True)
        (debug_dir / f"{prefix}_batch_{batch_id}_prompt.txt").write_text(
            f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{text}", encoding="utf-8")

    for i, item in enumerate(batch):
        img = open_rgb(item["image_path"])
        parts.append(Part.from_bytes(data=image_to_jpeg_bytes(img), mime_type="image/jpeg"))

        if not normal and item.get("mask_path"):
            overlay = utils.create_overlay(item["image_path"], item["mask_path"]) or img
            parts.append(Part.from_bytes(data=image_to_jpeg_bytes(overlay), mime_type="image/jpeg"))

            ref_path = get_normal_reference_image(item["category"], item.get("camera_hint"))
            if ref_path:
                ref = open_rgb(ref_path)
                parts.append(Part.from_bytes(data=image_to_jpeg_bytes(ref), mime_type="image/jpeg"))
                if debug:
                    ref.save(debug_dir / f"{prefix}_batch_{batch_id}_s{i}_ref.jpg")
            else:
                logger.warning(f"No normal reference for category '{item['category']}'")
            if debug:
                overlay.save(debug_dir / f"{prefix}_batch_{batch_id}_s{i}_overlay.jpg")
        if debug:
            img.save(debug_dir / f"{prefix}_batch_{batch_id}_s{i}_orig.jpg")

    if debug:
        logger.info(f"[DEBUG] Batch {batch_id} saved (no API call).")
        return {item["image_id"]: "DEBUG_TRACE" for item in batch}

    cfg = GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[Part.from_text(text=system_prompt)],
        thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL),
    )
    full = ""
    for chunk in client.models.generate_content_stream(model=model,
                    contents=[Content(role="user", parts=parts)], config=cfg):
        if chunk.text:
            full += chunk.text

    validated = BatchReasoningTraces(**normalise(parse_json_response(full)))
    return {t.image_id: t.reasoning for t in validated.traces}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(shard_id=0, total_shards=1, normal_only=False, anomaly_only=False, debug=False):
    global logger
    logger = setup_logging(shard_id)
    logger.info(f"=== Variety Generation | Shard {shard_id+1}/{total_shards} ===")
    if debug:
        logger.warning("DEBUG mode — no API calls.")

    system_prompt = load_system_prompt()
    logger.info(f"Vertex project={VERTEX_PROJECT}, location={VERTEX_LOCATION}, model={GEMINI_MODEL}")

    def _run_type(questions, output_dir, normal):
        label = "normal" if normal else "anomaly"
        shard_questions = [q for i, q in enumerate(questions) if i % total_shards == shard_id]
        logger.info(f"[{label}] Shard assigned {len(shard_questions)} / {len(questions)}")

        processed = {f.stem[6:] for f in output_dir.glob("trace_*.json")}
        logger.info(f"[{label}] Already processed: {len(processed)}")
        remaining = [q for q in shard_questions if q["image_id"] not in processed]
        logger.info(f"[{label}] To process: {len(remaining)}")
        if not remaining:
            return

        client = genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        max_retries = config.MAX_RETRIES * 3
        attempt_counts: Dict[str, int] = {}
        permanently_skipped: set = set()

        with tqdm(total=len(remaining), desc=f"Shard{shard_id} {label}") as pbar:
            while True:
                todo = [q for q in remaining
                        if q["image_id"] not in processed
                        and q["image_id"] not in permanently_skipped]
                if not todo:
                    break
                for q in todo:
                    if attempt_counts.get(q["image_id"], 0) >= config.MAX_RETRIES:
                        if q["image_id"] not in permanently_skipped:
                            logger.warning(f"Skipping {q['image_id']} (max retries)")
                            permanently_skipped.add(q["image_id"])

                batch = []
                for q in todo:
                    if len(batch) >= BATCH_SIZE:
                        break
                    if q["image_id"] not in permanently_skipped:
                        batch.append(q)
                        attempt_counts[q["image_id"]] = attempt_counts.get(q["image_id"], 0) + 1
                if not batch:
                    break

                success, retry_count = False, 0
                _max = 1 if debug else max_retries
                while not success and retry_count < _max:
                    try:
                        traces = process_batch(batch, client, GEMINI_MODEL, system_prompt,
                                               normal=normal, debug=debug, shard_id=shard_id)
                        if not debug:
                            saved = 0
                            for item in batch:
                                if item["image_id"] in traces:
                                    save_trace(item["image_id"], traces[item["image_id"]], output_dir)
                                    processed.add(item["image_id"])
                                    saved += 1
                            pbar.update(saved)
                        else:
                            for item in batch:
                                processed.add(item["image_id"])
                            pbar.update(len(batch))
                        success = True
                    except Exception as e:
                        if debug:
                            logger.error(f"[DEBUG] Error: {e}")
                            break
                        err = str(e)
                        retry_count += 1
                        # Exponential backoff with jitter, capped at 5 min. Rate-limit /
                        # server-unavailable errors start higher; transient errors lower.
                        is_rate = ("429" in err or "RESOURCE_EXHAUSTED" in err
                                   or "503" in err or "UNAVAILABLE" in err
                                   or "Connection" in err or "timeout" in err.lower())
                        base = 10.0 if is_rate else 5.0
                        delay = min(base * (2 ** (retry_count - 1)), 300.0)
                        delay += random.uniform(0.0, 0.25 * delay)  # jitter
                        kind = "Rate limit / unavailable" if is_rate else "Batch error"
                        logger.warning(
                            f"{kind} (attempt {retry_count}/{_max}): {e} "
                            f"-> exp-backoff {delay:.1f}s")
                        time.sleep(delay)
        logger.info(f"[{label}] Done. Output: {output_dir}")

    grpo_ng, grpo_ok = load_grpo_questions()
    sft_ng,  sft_ok  = load_sft_questions()

    if not normal_only:
        _run_type(grpo_ng, GRPO_DIR, normal=False)
    if not anomaly_only:
        _run_type(grpo_ok, GRPO_DIR, normal=True)
    if not normal_only:
        _run_type(sft_ng, SFT_DIR, normal=False)
    if not anomaly_only:
        _run_type(sft_ok, SFT_DIR, normal=True)
    logger.info("=== Shard complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Real-IAD Variety reasoning traces (Gemini, inspector_prompt_test_v2, C1, 120-200 words).")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--anomaly_only", action="store_true")
    parser.add_argument("--normal_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(shard_id=args.shard_id, total_shards=args.total_shards,
        normal_only=args.normal_only, anomaly_only=args.anomaly_only, debug=args.debug)
