"""
MMAD reasoning-trace generation with the Real-IAD v4 recipe.

Side experiment (Sep 2026): generate one six-phase trace per MMAD image with the
same teacher recipe as the thesis corpus (inspector_prompt_test_v2 system prompt,
v4 user prompt, batch of 10, three images per anomalous sample: original, red
mask overlay, normal reference), but with the MMAD annotations as internal hints
(defect type, location, appearance, effect from the MMAD multiple-choice answers).

Teacher: gemini-3.6-flash on Vertex AI (Amir's service account), thinking MINIMAL.

Usage:
  python generate_mmad_traces_v4.py --shard_id 0 --total_shards 8
  python generate_mmad_traces_v4.py --debug --limit 20          # no API calls
  python generate_mmad_traces_v4.py --limit 10 --shard_id 0 --total_shards 1   # live smoke test
"""
import argparse
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from pydantic import BaseModel
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig, ThinkingConfig, ThinkingLevel

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
MMAD_ROOT = Path(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/reasoning_traces_gen/data/MMAD")
MMAD_JSON = MMAD_ROOT / "mmad.json"
PROMPT_FILE = HERE / "inspector_prompt_test_v2_mmad_run.txt"   # exact copy of the system prompt this run used
KEY_FILE = Path(os.environ.get("WORK_DIR", "/bulk/aacudad/reasoning_traces") + "/vertexai-amir-key.json")
OUT_DIR = HERE / "traces"
LOG_DIR = HERE / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL = "gemini-3.6-flash"
VERTEX_LOCATION = "global"
BATCH_SIZE = 10
MAX_RETRIES = 3            # per image
MAX_BATCH_RETRIES = 9      # per batch call

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_FILE)
VERTEX_PROJECT = json.load(open(KEY_FILE))["project_id"]

logger = logging.getLogger(__name__)


def setup_logging(shard_id: int):
    fmt = f"%(asctime)s [Shard {shard_id}] %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, force=True,
                        handlers=[logging.FileHandler(LOG_DIR / f"gen_shard{shard_id}.log"),
                                  logging.StreamHandler()])
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ReasoningTrace(BaseModel):
    image_id: str
    reasoning: str


class BatchReasoningTraces(BaseModel):
    traces: List[ReasoningTrace]


# ---------------------------------------------------------------------------
# MMAD loading
# ---------------------------------------------------------------------------
def _answer_text(q: Dict) -> Optional[str]:
    opts = q.get("Options") or {}
    a = q.get("Answer")
    return opts.get(a) if a in opts else None


def image_id_from_key(key: str) -> str:
    return re.sub(r"\.(png|jpg|jpeg|bmp)$", "", key, flags=re.I).replace("/", "_")


def load_mmad_items() -> List[Dict]:
    """One item per MMAD image, with hints pulled from the multiple-choice answers."""
    data = json.load(open(MMAD_JSON))
    items, dropped = [], 0
    for key, v in data.items():
        parts = key.split("/")
        ds, product = parts[0], parts[1]
        conv = v["conversation"]
        det = [q for q in conv if q["type"] == "Anomaly Detection"]
        if not det:
            dropped += 1
            continue
        det_ans = (_answer_text(det[0]) or "").strip().lower()
        is_anomaly = det_ans.startswith("yes")
        hints = {}
        for q in conv:
            t = q["type"]
            if t == "Defect Classification":
                hints["defect_type"] = _answer_text(q)
            elif t == "Defect Localization":
                hints["location"] = _answer_text(q)
            elif t == "Defect Description":
                hints["appearance"] = _answer_text(q)
            elif t == "Defect Analysis":
                hints["effect"] = _answer_text(q)
        item = {
            "image_id": image_id_from_key(key),
            "key": key,
            "dataset": ds,
            "product": product,
            "image_path": MMAD_ROOT / key,
            "is_anomaly": is_anomaly,
            "hints": hints,
            "mask_path": None,
            "ref_path": None,
        }
        if is_anomaly:
            m = v.get("mask_path")
            mp = MMAD_ROOT / ds / product / m if m else None
            if mp is None or not mp.exists():
                dropped += 1          # 4 MVTec-LOCO anomalies have no mask on disk
                continue
            item["mask_path"] = mp
            refs = [MMAD_ROOT / t for t in v.get("similar_templates", [])]
            refs = [r for r in refs if r.exists()]
            if not refs:
                dropped += 1
                continue
            item["ref_path"] = refs[0]
        items.append(item)
    logger.info(f"MMAD items: {len(items)} usable, {dropped} dropped")
    return items


# ---------------------------------------------------------------------------
# Image helpers (same as v4)
# ---------------------------------------------------------------------------
def open_rgb(path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB") if img.mode != "RGB" else img


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


MAX_SIDE = 1536   # longest side of the JPEGs sent to the teacher (GoodsAD images are 3000 px)


def _binary_mask(mask_path) -> Image.Image:
    """Any non-zero pixel is defect. Handles RGB red-on-black masks (DS-MVTec), 8-bit masks
    with values 255 (VisA, GoodsAD) or 242 (MVTec-LOCO), and MVTec-LOCO's directory of
    several mask files per image, which are unioned."""
    import numpy as np
    p = Path(mask_path)
    files = sorted(p.glob("*.png")) if p.is_dir() else [p]
    acc = None
    for f in files:
        a = np.array(Image.open(f))
        if a.ndim == 3:
            a = a[..., :3].max(axis=-1)
        b = a > 0
        acc = b if acc is None else (acc | b)
    return Image.fromarray((acc.astype("uint8") * 255), mode="L")


def _shrink(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if s <= MAX_SIDE:
        return img
    f = MAX_SIDE / s
    return img.resize((max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.LANCZOS)


def create_overlay(image_path, mask_path, color=(255, 0, 0), alpha=128) -> Image.Image:
    base = Image.open(image_path).convert("RGBA")
    mask = _binary_mask(mask_path)
    if base.size != mask.size:
        mask = mask.resize(base.size, Image.Resampling.NEAREST)
    solid = Image.new("RGBA", base.size, color + (alpha,))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.paste(solid, (0, 0), mask)
    return _shrink(Image.alpha_composite(base, overlay).convert("RGB"))


# ---------------------------------------------------------------------------
# Prompt builders (v4 wording, plus the MMAD hints as INTERNAL lines)
# ---------------------------------------------------------------------------
GRID_LINE = (
    "[INTERNAL] The red overlay (second image) highlights the region of interest. "
    "For the <location> tag, use a 3x3 grid: top-left, top-center, top-right, "
    "middle-left, center, middle-right, bottom-left, bottom-center, bottom-right. "
    "If the defect spans multiple regions, separate them with commas "
    "(e.g. top-center, top-left)."
)


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
        "The [INTERNAL] annotation lines below come from the dataset's own labels. "
        "Use them to make the trace correct, never quote or mention them. "
        "For the <type> tag pick the closest entry of the DEFECT TYPE VOCABULARY that "
        "matches the annotated defect type. If none fits, use the annotated wording, "
        "capitalised, without a trailing full stop.",
        "",
    ]
    for idx, item in enumerate(batch, 1):
        h = item["hints"]
        parts.append(f"--- SAMPLE {idx} ---")
        parts.append(f"Image ID: {item['image_id']}")
        parts.append(f"Product Type: {item['product'].replace('_', ' ')}")
        parts.append("[INTERNAL - do not reference in reasoning] Ground Truth: ANOMALOUS")
        if h.get("defect_type"):
            parts.append(f"[INTERNAL] Defect Type: {h['defect_type']}")
        if h.get("location"):
            parts.append(f"[INTERNAL] Annotated location: {h['location']}")
        if h.get("appearance"):
            parts.append(f"[INTERNAL] Annotated appearance: {h['appearance']}")
        if h.get("effect"):
            parts.append(f"[INTERNAL] Annotated effect: {h['effect']}")
        parts.append(GRID_LINE)
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
        parts.append(f"Product Type: {item['product'].replace('_', ' ')}")
        parts.append("[INTERNAL - do not reference in reasoning] Ground Truth: NORMAL")
        parts.append("")
    parts.append("Return a JSON object with a 'traces' array containing one entry per sample.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------
def parse_json_response(text: str) -> Dict:
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return json.loads(m.group(1) if m else text)


def normalise(d) -> Dict:
    if isinstance(d, list):
        d = {"traces": d}
    if "traces" not in d and "image_id" in d:
        d = {"traces": [d]}
    return d


def process_batch(batch: List[Dict], client, system_prompt: str, normal: bool,
                  debug: bool, shard_id: int) -> Dict[str, str]:
    batch_id = str(uuid.uuid4())[:8]
    text = format_normal_prompt(batch) if normal else format_anomaly_prompt(batch)
    parts = [Part.from_text(text=text)]
    dbg = HERE / "debug_output"
    if debug:
        dbg.mkdir(exist_ok=True)
        (dbg / f"shard{shard_id}_{'normal' if normal else 'anomaly'}_{batch_id}_prompt.txt").write_text(
            f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{text}")

    for i, item in enumerate(batch):
        img = _shrink(open_rgb(item["image_path"]))
        parts.append(Part.from_bytes(data=image_to_jpeg_bytes(img), mime_type="image/jpeg"))
        if not normal:
            overlay = create_overlay(item["image_path"], item["mask_path"])
            parts.append(Part.from_bytes(data=image_to_jpeg_bytes(overlay), mime_type="image/jpeg"))
            ref = _shrink(open_rgb(item["ref_path"]))
            parts.append(Part.from_bytes(data=image_to_jpeg_bytes(ref), mime_type="image/jpeg"))
            if debug:
                overlay.save(dbg / f"shard{shard_id}_{batch_id}_s{i}_overlay.jpg")
                ref.save(dbg / f"shard{shard_id}_{batch_id}_s{i}_ref.jpg")
        if debug:
            img.save(dbg / f"shard{shard_id}_{batch_id}_s{i}_orig.jpg")

    if debug:
        logger.info(f"[DEBUG] batch {batch_id} written, no API call")
        return {it["image_id"]: "DEBUG_TRACE" for it in batch}

    cfg = GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[Part.from_text(text=system_prompt)],
        thinking_config=ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL),
    )
    full = ""
    for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL, contents=[Content(role="user", parts=parts)], config=cfg):
        if chunk.text:
            full += chunk.text
    validated = BatchReasoningTraces(**normalise(parse_json_response(full)))
    return {t.image_id: t.reasoning for t in validated.traces}


# ---------------------------------------------------------------------------
# Trace validation (structure only, mirrors the auto-reject rule D)
# ---------------------------------------------------------------------------
ANOM_RE = re.compile(r"<think>[\s\S]+</think>\s*<location>[^<]+</location>\s*<type>[^<]+</type>\s*<answer>\s*Yes\s*</answer>\s*$", re.I)
NORM_RE = re.compile(r"<think>[\s\S]+</think>\s*<answer>\s*No\s*</answer>\s*$", re.I)


def trace_ok(reasoning: str, is_anomaly: bool) -> bool:
    r = reasoning.strip()
    return bool((ANOM_RE if is_anomaly else NORM_RE).match(r))


def save_trace(item: Dict, reasoning: str):
    out = OUT_DIR / f"trace_{item['image_id']}.json"
    rec = {
        "image_id": item["image_id"],
        "key": item["key"],
        "dataset": item["dataset"],
        "product": item["product"],
        "is_anomaly": item["is_anomaly"],
        "image_path": str(item["image_path"]),
        "hints": item["hints"],
        "model": GEMINI_MODEL,
        "thinking_level": "MINIMAL",
        "prompt_version": "inspector_prompt_test_v2 + v4 user prompt + MMAD hints",
        "reasoning": reasoning,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(shard_id: int, total_shards: int, debug: bool, limit: int, datasets: Optional[str] = None):
    global logger
    logger = setup_logging(shard_id)
    logger.info(f"=== MMAD v4-recipe generation | shard {shard_id + 1}/{total_shards} | model {GEMINI_MODEL} ===")
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
    items = load_mmad_items()
    if datasets:
        keep = set(datasets.split(","))
        items = [it for it in items if it["dataset"] in keep]
    mine = [it for i, it in enumerate(items) if i % total_shards == shard_id]
    if limit:
        mine = mine[:limit]
    done = {p.stem[6:] for p in OUT_DIR.glob("trace_*.json")}
    todo_all = [it for it in mine if it["image_id"] not in done]
    logger.info(f"shard items {len(mine)}, already done {len(mine) - len(todo_all)}, to do {len(todo_all)}")

    client = None if debug else genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION)
    attempts: Dict[str, int] = {}
    skipped = set()
    rejected_structure = 0
    t_start = time.time()
    n_done = 0

    for normal in (False, True):
        pool = [it for it in todo_all if it["is_anomaly"] != normal]
        label = "normal" if normal else "anomaly"
        logger.info(f"[{label}] {len(pool)} items")
        while True:
            todo = [it for it in pool if it["image_id"] not in done and it["image_id"] not in skipped]
            for it in todo:
                if attempts.get(it["image_id"], 0) >= MAX_RETRIES:
                    skipped.add(it["image_id"])
                    logger.warning(f"skip {it['image_id']} after {MAX_RETRIES} attempts")
            todo = [it for it in todo if it["image_id"] not in skipped]
            if not todo:
                break
            batch = todo[:BATCH_SIZE]
            for it in batch:
                attempts[it["image_id"]] = attempts.get(it["image_id"], 0) + 1

            ok, tries = False, 0
            while not ok and tries < (1 if debug else MAX_BATCH_RETRIES):
                try:
                    traces = process_batch(batch, client, system_prompt, normal, debug, shard_id)
                    for it in batch:
                        r = traces.get(it["image_id"])
                        if r is None:
                            continue
                        if not debug and not trace_ok(r, it["is_anomaly"]):
                            rejected_structure += 1
                            logger.warning(f"structure reject {it['image_id']}: {r[-120:]!r}")
                            continue
                        if not debug:
                            save_trace(it, r)
                        done.add(it["image_id"])
                        n_done += 1
                    ok = True
                except Exception as e:
                    err = str(e)
                    tries += 1
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        logger.warning("rate limit, waiting 10 s")
                        time.sleep(10)
                    else:
                        logger.error(f"batch error ({tries}): {err[:300]}")
                        time.sleep(5)
            if n_done and n_done % 50 < BATCH_SIZE:
                el = time.time() - t_start
                logger.info(f"progress {n_done}/{len(todo_all)} in {el/60:.1f} min, "
                            f"{el/max(n_done,1):.1f} s/trace, structure rejects {rejected_structure}")

    logger.info(f"=== shard done: {n_done} traces, {rejected_structure} structure rejects, "
                f"{len(skipped)} skipped, {(time.time()-t_start)/60:.1f} min ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--total_shards", type=int, default=1)
    ap.add_argument("--debug", action="store_true", help="write prompts and images, no API calls")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items of this shard")
    ap.add_argument("--datasets", default=None, help="comma list to restrict, e.g. VisA,GoodsAD (testing)")
    a = ap.parse_args()
    run(a.shard_id, a.total_shards, a.debug, a.limit, a.datasets)
