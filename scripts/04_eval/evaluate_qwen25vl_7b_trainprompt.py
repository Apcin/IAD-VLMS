"""
Evaluate Qwen2.5-VL-7B SFT model (single image, zero-shot)
============================================================

Evaluates the Qwen2.5-VL-7B model trained with a single-image (zero-shot) setup.
Uses unified_test_zeroshot_full.json — Real-IAD held-out products + MMAD test set.

Usage:
    # Evaluate a checkpoint (200 samples, 50/50 balanced)
    python Training/evaluate_qwen25vl_7b.py --checkpoint outputs/sft_qwen25vl_7b_zeroshot/checkpoint-XXX

    # Real-IAD only (in-distribution evaluation)
    python Training/evaluate_qwen25vl_7b.py --checkpoint outputs/sft_qwen25vl_7b_zeroshot/checkpoint-XXX --realiad-only

    # MMAD only (out-of-distribution)
    python Training/evaluate_qwen25vl_7b.py --checkpoint outputs/sft_qwen25vl_7b_zeroshot/checkpoint-XXX --mmad-only

    # Baseline — base model before finetuning
    python Training/evaluate_qwen25vl_7b.py --base-model Qwen/Qwen2.5-VL-7B-Instruct
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import torch

# Workspace root. WORK_DIR is the directory that holds Training/, outputs/, hf_cache/
# and the downloaded datasets. It defaults to the parent of this repository, which is
# how the cluster is laid out. On another machine:  export WORK_DIR=/path/to/workspace
REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(os.environ.get("WORK_DIR") or REPO_ROOT.parent)
os.environ.setdefault("HF_HOME", str(WORK_DIR / "hf_cache"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42
# Optional fast-probe cap for closed-loop testing. Set PROBE_SAMPLES env var (>0) to
# truncate the (seed-shuffled, hence deterministic) ds-mvtec-only / visa-only subset to
# that many samples. Default 0 = full subset (preserves watchdog / full-eval behaviour).
import os as _os
PROBE_SAMPLES = int(_os.environ.get("PROBE_SAMPLES", "0") or "0")

# Optional pre-resize cap in pixels (area), for models whose image processor has no
# max_pixels support (e.g. LLaVA-OneVision anyres). Set EVAL_MAX_IMAGE_PIXELS=262144
# to match the 512x512 training/eval protocol. Default 0 = off (Qwen path unchanged:
# its processor.max_pixels already enforces the cap).
EVAL_MAX_IMAGE_PIXELS = int(_os.environ.get("EVAL_MAX_IMAGE_PIXELS", "0") or "0")

def _cap_image_pixels(img):
    if EVAL_MAX_IMAGE_PIXELS and img.width * img.height > EVAL_MAX_IMAGE_PIXELS:
        f = (EVAL_MAX_IMAGE_PIXELS / (img.width * img.height)) ** 0.5
        img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS)
    return img

DATA_PATHS = {
    "test":  str(WORK_DIR / "Training/datasets/unified_test_zeroshot_full.json"),
    "train": str(WORK_DIR / "Training/datasets/unified_train_zeroshot_full.json"),
    "mmad_official": str(WORK_DIR / "MMAD_repo/dataset/MMAD/mmad.json"),
    "realiad_images_root": str(WORK_DIR / "reasoning_traces_gen/data/Real-IAD/images"),
    "mmad_images_root":    str(WORK_DIR / "reasoning_traces_gen/data/MMAD"),
}

# MMAD sub-datasets (first component of image_id path)
MMAD_SUBDATASETS = {"DS-MVTec", "GoodsAD", "MVTec-AD", "MVTec-LOCO", "VisA"}

def make_train_prompt(product_name: str) -> str:
    return (
        f"Analyze the provided image of the {product_name}. "
        "Determine if there are any anomalies present. "
        "If an anomaly is detected, specify its type and location, "
        "and provide a detailed reasoning for your conclusion."
    )


def make_noreason_prompt(product_name: str) -> str:
    """Native prompt of the no-reasoning (labels-only) SFT baseline: same product-conditioned
    preamble as make_train_prompt, but it asks for a bare verdict instead of type/location/reasoning.
    Must match the training data (Training/datasets_noreason/) character for character."""
    return (
        f"Analyze the provided image of the {product_name}. "
        "Determine if there are any anomalies present. "
        "Answer with yes or no."
    )


GRPO_EVAL_PROMPT = (
    "You are an expert in detecting defects in image. "
    "Your task is to detect if there are any defects in the test image."
    "Are there any defects in the query image?"
)


# ---------------------------------------------------------------------------
# Gemini-2.5-Flash benchmark mode (--gemini): identical data/GT/metrics, only the
# model call is swapped for a Gemini call (Vertex AI). Used to produce a directly
# comparable Gemini-2.5-Flash reference row on the SAME DS-MVTec / VisA eval set.
# ---------------------------------------------------------------------------
# Vertex settings come from the environment. Nothing secret is stored in this file:
# GOOGLE_APPLICATION_CREDENTIALS points at the service-account JSON (git-ignored).
GEMINI_VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GEMINI_VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
GEMINI_VERTEX_KEY = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

GEMINI_SYSTEM_PROMPT = (
    "You are an expert industrial visual-inspection system. You are given a single image of a "
    "manufactured product. Inspect it carefully and decide whether it contains any defect (anomaly).\n\n"
    "Respond with ONLY the following XML tags and nothing else:\n"
    "- If a defect IS present:\n"
    "<think>one or two sentences of reasoning grounded in the visible evidence</think>"
    "<type>the defect type</type><location>where in the image the defect is</location>"
    "<answer>Yes</answer>\n"
    "- If NO defect is present (the product looks normal):\n"
    "<answer>No</answer>"
)


def make_gemini_user_text(product_name: str) -> str:
    return f"Inspect this image of the {product_name} and report whether it has any defect."


def build_gemini_client(model_name: str):
    if GEMINI_VERTEX_KEY:
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GEMINI_VERTEX_KEY)
    if not GEMINI_VERTEX_PROJECT:
        raise SystemExit("--gemini needs GOOGLE_CLOUD_PROJECT (and GOOGLE_APPLICATION_CREDENTIALS) "
                         "in the environment; see .env.example")
    from google import genai
    client = genai.Client(vertexai=True, project=GEMINI_VERTEX_PROJECT, location=GEMINI_VERTEX_LOCATION)
    logger.info(f"Gemini client ready: model={model_name} (Vertex {GEMINI_VERTEX_PROJECT}/{GEMINI_VERTEX_LOCATION}), thinking_budget=0")
    return client


def gemini_predict_one(client, model_name: str, image, product_name: str, thinking_level: Optional[str] = None, max_retries: int = 6) -> str:
    """One Gemini call: structured-prompt, returns raw text (with <answer> tags). thinking_level
    (Gemini-3, e.g. 'minimal'/'low'/'high') takes precedence; else thinking_budget=0 (Gemini-2.5)."""
    import io
    import time as _time
    from google.genai.types import Content, Part, GenerateContentConfig, ThinkingConfig
    bio = io.BytesIO()
    image.save(bio, format="JPEG")
    parts = [
        Part.from_bytes(data=bio.getvalue(), mime_type="image/jpeg"),
        Part.from_text(text=make_gemini_user_text(product_name)),
    ]
    tcfg = ThinkingConfig(thinking_level=thinking_level) if thinking_level else ThinkingConfig(thinking_budget=0)
    cfg = GenerateContentConfig(
        system_instruction=[Part.from_text(text=GEMINI_SYSTEM_PROMPT)],
        temperature=0.0,
        max_output_tokens=1536,
        thinking_config=tcfg,
    )
    delay = 4.0
    for _ in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_name, contents=[Content(role="user", parts=parts)], config=cfg,
            )
            return resp.text or ""
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  gemini retry ({type(e).__name__}: {str(e)[:80]})")
            _time.sleep(delay)
            delay = min(delay * 1.8, 60.0)
    return ""


# ---------------------------------------------------------------------------
# OpenAI (GPT) benchmark mode (--openai): identical data/GT/metrics/prompt as
# --gemini, only the model call is swapped for an OpenAI Responses-API call.
# Produces a directly comparable GPT reference row on the SAME eval set.
# ---------------------------------------------------------------------------
def load_openai_key() -> str:
    """OPENAI_API_KEY from the environment, or from a git-ignored .env at the repo root."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for ln in env_path.read_text().splitlines():
            if ln.strip().startswith("OPENAI_API_KEY"):
                return ln.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENAI_API_KEY is not set and there is no OPENAI_API_KEY line in .env")


def build_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=load_openai_key())


def openai_predict_one(client, model_name: str, image, product_name: str, effort: str = "low", max_retries: int = 5) -> str:
    """One OpenAI Responses-API call: same structured prompt as Gemini, returns raw text with <answer> tags."""
    import io, base64
    import time as _time
    bio = io.BytesIO()
    image.convert("RGB").save(bio, format="JPEG", quality=85)
    b64 = base64.b64encode(bio.getvalue()).decode()
    delay = 4.0
    for _ in range(max_retries):
        try:
            r = client.responses.create(
                model=model_name,
                input=[{"role": "system", "content": GEMINI_SYSTEM_PROMPT},
                       {"role": "user", "content": [
                           {"type": "input_text", "text": make_gemini_user_text(product_name)},
                           {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}]}],
                reasoning={"effort": effort}, max_output_tokens=2000)
            txt = ""
            for o in r.output:
                if getattr(o, "type", None) == "message":
                    for c in o.content:
                        if getattr(c, "type", None) == "output_text":
                            txt = c.text
            return txt or ""
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  openai retry ({type(e).__name__}: {str(e)[:80]})")
            _time.sleep(delay)
            delay = min(delay * 1.8, 60.0)
    return ""


# ---------------------------------------------------------------------------
# Sub-dataset helper
# ---------------------------------------------------------------------------

def get_subdataset(image_id: str) -> str:
    """Extract MMAD sub-dataset from image_id prefix, e.g. 'GoodsAD/drink_bottle/...' -> 'GoodsAD'."""
    if "/" in image_id:
        prefix = image_id.split("/")[0]
        if prefix in MMAD_SUBDATASETS:
            return prefix
    return "unknown"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def build_image_map(realiad_root: str, mmad_root: str) -> Dict[str, str]:
    image_map: Dict[str, str] = {}
    if os.path.exists(realiad_root):
        p = Path(realiad_root)
        for ext in ["*.jpg", "*.jpeg", "*.JPG"]:
            for path in p.rglob(ext):
                image_map[path.stem] = str(path)
                try:
                    rel = path.relative_to(p)
                    image_map[str(rel.with_suffix("")).replace("/", "_")] = str(path)
                except ValueError:
                    pass
    if os.path.exists(mmad_root):
        p = Path(mmad_root)
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
            for path in p.rglob(ext):
                try:
                    image_map[str(path.relative_to(p))] = str(path)
                except ValueError:
                    pass
    return image_map


def resolve_image_path(image_id: str, image_map: Dict[str, str]) -> Optional[str]:
    if image_id in image_map:
        return image_map[image_id]
    if os.path.exists(image_id):
        return image_id
    base = image_id.split("/")[-1] if "/" in image_id else image_id
    for key, path in image_map.items():
        if image_id in key or key.endswith(base):
            return path
    return None


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------

def extract_tags(text: str) -> Dict[str, str]:
    result = {}
    for tag in ["think", "reasoning"]:
        m = re.search(f"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
        if m:
            result["reasoning"] = m.group(1).strip()
            break
    for tag in ["type", "location", "answer"]:
        m = re.search(f"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip().lower()
            if tag == "answer":
                if val in ["a", "yes", "y", "true"]:
                    val = "yes"
                elif val in ["b", "no", "n", "false"]:
                    val = "no"
            result[tag] = val
    return result


def normalize_answer(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ["a", "yes", "y", "true", "1"]:
        return "yes"
    if raw in ["b", "no", "n", "false", "0"]:
        return "no"
    return raw


def fuzzy_yes_no(text: str) -> str:
    """Fallback: match bare yes/no in a short response (like IAD-R1's get_ans)."""
    t = text.strip().lower()
    # Exact or starts-with match
    if t in ("yes", "yes.", "yes!") or t.startswith("yes"):
        return "yes"
    if t in ("no", "no.", "no!") or t.startswith("no"):
        return "no"
    # Substring match as last resort
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return ""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def is_realiad(image_id: str) -> bool:
    return "_OK_" in image_id or "_NG_" in image_id


def _entry_is_anomaly(entry: dict) -> bool:
    convs = entry.get("conversation", [])
    if not convs:
        return False
    first = convs[0]
    if "annotation" in first:
        return bool(first["annotation"])
    raw = first.get("Answer", "B")
    return "<answer>A</answer>" in raw or "answer>yes" in raw.lower()


def _mmad_product(image_id: str) -> str:
    """Return 'SubDataset/product' from MMAD image_id like 'DS-MVTec/wood/test/good/000.png'."""
    parts = image_id.replace("\\", "/").split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else image_id


def _stratified_balanced_sample(data: List[dict], num_samples: int) -> List[dict]:
    """Sample num_samples (50/50 anomaly/normal) stratified proportionally across MMAD products."""
    from collections import defaultdict
    prod_anomaly: dict = defaultdict(list)
    prod_normal:  dict = defaultdict(list)
    for entry in data:
        key = _mmad_product(entry.get("image_id", ""))
        (prod_anomaly if _entry_is_anomaly(entry) else prod_normal)[key].append(entry)

    products = sorted(set(prod_anomaly) | set(prod_normal))
    logger.info(f"Stratified sampling across {len(products)} MMAD products")

    per_class = (num_samples // 2) if num_samples else None

    if per_class is None:
        # All balanced samples
        a_all, n_all = [], []
        for p in products:
            n = min(len(prod_anomaly[p]), len(prod_normal[p]))
            a_all.extend(random.sample(prod_anomaly[p], n))
            n_all.extend(random.sample(prod_normal[p], n))
        combined = a_all + n_all
        random.shuffle(combined)
        logger.info(f"Stratified (all): {len(a_all)} anomaly + {len(n_all)} normal = {len(combined)} total")
        return combined

    # Weighted quota per product based on min(anomaly, normal) pool size
    weights  = {p: min(len(prod_anomaly[p]), len(prod_normal[p])) for p in products}
    total_w  = sum(weights.values()) or 1
    a_result, n_result = [], []
    for i, p in enumerate(products):
        if i == len(products) - 1:
            a_q = per_class - len(a_result)
            n_q = per_class - len(n_result)
        else:
            q   = max(1, round(per_class * weights[p] / total_w))
            a_q = n_q = q
        a_q = min(a_q, len(prod_anomaly[p]))
        n_q = min(n_q, len(prod_normal[p]))
        if a_q > 0:
            a_result.extend(random.sample(prod_anomaly[p], a_q))
        if n_q > 0:
            n_result.extend(random.sample(prod_normal[p], n_q))
    combined = a_result + n_result
    random.shuffle(combined)
    logger.info(f"Stratified: {len(a_result)} anomaly + {len(n_result)} normal = {len(combined)} total")
    return combined


def load_balanced_test_data(json_path: str, num_samples: Optional[int], realiad_only: bool, mmad_only: bool = False, mmad_all: bool = False, mmad_official: bool = False, mmad_official_iad_r1: bool = False, ds_mvtec_only: bool = False, visa_only: bool = False, realiad_4k: bool = False) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)

    if realiad_4k:
        # Load the new_sft_c1_train.json (4236 RealIAD samples) and convert SFT format to eval format
        realiad_4k_path = str(WORK_DIR / "Training/datasets_small_15k_c1_only_fixed/new_sft_c1_train.json")
        with open(realiad_4k_path, "r") as f:
            sft_data = json.load(f)
        result = []
        for item in sft_data:
            image_path = item["images"][0] if isinstance(item["images"], list) else item["images"]
            assistant_msg = item["messages"][1]["content"]
            # Extract product from path: .../images/{product}/...
            path_parts = image_path.split("/")
            product = "unknown"
            for pi, part in enumerate(path_parts):
                if part == "images" and pi + 1 < len(path_parts):
                    product = path_parts[pi + 1]
                    break
            # Extract GT tags from assistant response
            answer_m = re.search(r'<answer>(.*?)</answer>', assistant_msg, re.DOTALL)
            type_m = re.search(r'<type>(.*?)</type>', assistant_msg, re.DOTALL)
            location_m = re.search(r'<location>(.*?)</location>', assistant_msg, re.DOTALL)
            gt_answer = answer_m.group(1).strip().lower() if answer_m else "no"
            gt_type = type_m.group(1).strip() if type_m else ""
            gt_location = location_m.group(1).strip() if location_m else ""
            # Build a compatible entry (use image_path as image_id, store conversation for GT extraction)
            answer_tag = f"<answer>{gt_answer.capitalize()}</answer>"
            if gt_answer == "yes":
                answer_tag = f"<type>{gt_type}</type><location>{gt_location}</location>{answer_tag}"
            result.append({
                "image_id": image_path,
                "product": product,
                "conversation": [{"Answer": answer_tag}],
            })
        logger.info(f"RealIAD-4k: {len(result)} entries from {realiad_4k_path}")
        random.seed(SEED)
        random.shuffle(result)
        return result

    if visa_only:
        # Load all VisA entries directly from mmad.json (no sampling)
        with open(DATA_PATHS["mmad_official"], "r") as f:
            mmad_data = json.load(f)
        result = []
        for image_id, entry in mmad_data.items():
            if image_id.startswith("VisA/"):
                entry["image_id"] = image_id
                parts = image_id.split("/")
                entry["product"] = parts[1] if len(parts) > 1 else "unknown"
                result.append(entry)
        logger.info(f"VisA only: {len(result)} entries")
        random.seed(SEED)
        random.shuffle(result)
        if PROBE_SAMPLES > 0:
            result = result[:PROBE_SAMPLES]
            logger.info(f"VisA PROBE: truncated to {len(result)} samples (deterministic, seed={SEED})")
        return result

    if ds_mvtec_only:
        # Load all DS-MVTec entries directly from mmad.json (no sampling)
        with open(DATA_PATHS["mmad_official"], "r") as f:
            mmad_data = json.load(f)
        result = []
        for image_id, entry in mmad_data.items():
            if image_id.startswith("DS-MVTec/"):
                entry["image_id"] = image_id
                # Populate product field from image_id e.g. "DS-MVTec/zipper/test/..." -> "zipper"
                parts = image_id.split("/")
                entry["product"] = parts[1] if len(parts) > 1 else "unknown"
                result.append(entry)
        logger.info(f"DS-MVTec only: {len(result)} entries")
        random.seed(SEED)
        random.shuffle(result)
        if PROBE_SAMPLES > 0:
            result = result[:PROBE_SAMPLES]
            logger.info(f"DS-MVTec PROBE: truncated to {len(result)} samples (deterministic, seed={SEED})")
        return result

    if mmad_official or mmad_official_iad_r1 or mmad_all:
        # Merge MMAD entries from train split as well
        with open(DATA_PATHS["train"], "r") as f:
            train_data = json.load(f)
        mmad_from_train = [e for e in train_data if not is_realiad(e.get("image_id", ""))]
        mmad_from_test  = [e for e in data       if not is_realiad(e.get("image_id", ""))]
        data = mmad_from_train + mmad_from_test
        if mmad_official or mmad_official_iad_r1:
            with open(DATA_PATHS["mmad_official"], "r") as f:
                official_keys = set(json.load(f).keys())
            before = len(data)
            data = [e for e in data if e.get("image_id", "") in official_keys]
            logger.info(f"MMAD official filter: {len(data)}/{before} entries match mmad.json ({len(official_keys)} total)")
            if mmad_official_iad_r1:
                before = len(data)
                data = [e for e in data if e.get("image_id", "").startswith(("DS-MVTec", "VisA"))]
                logger.info(f"IAD-R1 subset filter (DS-MVTec + VisA only): {len(data)}/{before} entries")
        else:
            logger.info(f"MMAD all filter: {len(mmad_from_train)} train + {len(mmad_from_test)} test = {len(data)} entries")
        random.seed(SEED)
        return _stratified_balanced_sample(data, num_samples)
    elif realiad_only:
        data = [e for e in data if is_realiad(e.get("image_id", ""))]
        logger.info(f"Real-IAD only filter: {len(data)} entries")
    elif mmad_only:
        data = [e for e in data if not is_realiad(e.get("image_id", ""))]
        logger.info(f"MMAD only filter: {len(data)} entries")

    anomaly = [e for e in data if _entry_is_anomaly(e)]
    normal  = [e for e in data if not _entry_is_anomaly(e)]

    logger.info(f"Test set: {len(anomaly)} anomaly, {len(normal)} normal")
    per_class = (num_samples // 2) if num_samples else min(len(anomaly), len(normal))
    random.seed(SEED)
    anomaly = random.sample(anomaly, min(per_class, len(anomaly)))
    normal  = random.sample(normal,  min(per_class, len(normal)))
    combined = anomaly + normal
    random.shuffle(combined)
    logger.info(f"Balanced: {len(anomaly)} anomaly + {len(normal)} normal = {len(combined)} total")
    return combined


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: Optional[str], base_model: str, load_in_4bit: bool = False):
    from transformers import AutoProcessor, BitsAndBytesConfig

    model_path = checkpoint if checkpoint else base_model
    logger.info(f"Loading model from: {model_path}")

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if load_in_4bit else None
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if quant_config:
        load_kwargs["quantization_config"] = quant_config

    # Detect LoRA/PEFT adapter checkpoint (GRPO saves only adapter weights)
    is_peft = checkpoint and os.path.exists(os.path.join(checkpoint, "adapter_config.json"))
    if is_peft:
        logger.info(f"Detected LoRA adapter checkpoint. Loading base model from: {base_model}")
        from peft import PeftModel
        try:
            from transformers import AutoModelForImageTextToText
            base = AutoModelForImageTextToText.from_pretrained(base_model, **load_kwargs)
        except Exception:
            from transformers import Qwen2_5_VLForConditionalGeneration
            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model, **load_kwargs)
        logger.info(f"Applying LoRA adapter from: {checkpoint}")
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.merge_and_unload()
        logger.info("LoRA merged into base model.")
    else:
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        except Exception:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

    processor = AutoProcessor.from_pretrained(
        base_model,  # always load processor from base so tokenizer/image_processor are correct
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    # Match training resolution: sft_qwen25vl_7b_zeroshot.yaml uses image_max_pixels=262144 (512x512).
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 262144
    model.eval()
    logger.info("Model loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Inference (single image)
# ---------------------------------------------------------------------------

def run_inference_batch(model, processor, batch_items: list, no_system_prompt: bool = False, grpo_eval: bool = False, bare_question: bool = False, iadr1_native: bool = False, yesno_user: bool = False, noreason: bool = False) -> list:
    """batch_items: list of (image, product_name)"""
    texts = []
    all_images = []
    for img, product_name in batch_items:
        if noreason:
            # Native mode of the no-reasoning (labels-only) SFT baseline: no system message,
            # product-conditioned preamble that asks for a bare yes/no verdict.
            prompt = make_noreason_prompt(product_name)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": f"\n{prompt}"},
            ]}]
        elif iadr1_native:
            # Exact IAD-R1 native eval prompt (vLLM_Qwen_detect.py build_prompt):
            # system "Please answer by yes or no" + user "Are there any defects in the test image?"
            messages = [
                {"role": "system", "content": "Please answer by yes or no"},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": "Are there any defects in the test image?"},
                ]},
            ]
        elif bare_question:
            # Bare question only — matches SFT-Iter2 training prompt exactly
            # (rollout-and-filter wrote user message as "<image>\n{question}" where
            # question = "Are there any defects in the query image?")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": "Are there any defects in the query image?"},
            ]}]
        elif grpo_eval:
            # Exact GRPO training format: fixed IAD-R1 prompt, no system message
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": GRPO_EVAL_PROMPT},
            ]}]
        elif yesno_user:
            # Qwen-parity instruction relocated to the USER turn (for backbones that
            # ignore the system slot, e.g. LLaVA-OneVision): identical wording.
            prompt = make_train_prompt(product_name)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": f"Please answer by yes or no\n{prompt}"},
            ]}]
        elif no_system_prompt:
            # Exact SFT training format: no system message
            prompt = make_train_prompt(product_name)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": f"\n{prompt}"},
            ]}]
        else:
            prompt = make_train_prompt(product_name)
            messages = [
                {"role": "system", "content": "Please answer by yes or no"},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": f"\n{prompt}"},
                ]},
            ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        all_images.append(img)

    inputs = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    return [
        processor.decode(output_ids[i][input_len:], skip_special_tokens=True)
        for i in range(len(batch_items))
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    tp = sum(1 for r in results if r["gt_answer"] == "yes" and r["pred_answer"] == "yes")
    tn = sum(1 for r in results if r["gt_answer"] == "no"  and r["pred_answer"] == "no")
    fp = sum(1 for r in results if r["gt_answer"] == "no"  and r["pred_answer"] == "yes")
    fn = sum(1 for r in results if r["gt_answer"] == "yes" and r["pred_answer"] == "no")

    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    anomaly_results = [r for r in results if r["gt_answer"] == "yes"]
    type_acc = (
        sum(1 for r in anomaly_results if r.get("type_correct")) / len(anomaly_results)
        if anomaly_results else 0.0
    )
    loc_acc = (
        sum(1 for r in anomaly_results if r.get("location_correct")) / len(anomaly_results)
        if anomaly_results else 0.0
    )

    source_metrics = {}
    for src in ("realiad", "mmad"):
        src_results = [r for r in results if r.get("source") == src]
        if not src_results:
            continue
        _tp = sum(1 for r in src_results if r["gt_answer"] == "yes" and r["pred_answer"] == "yes")
        _tn = sum(1 for r in src_results if r["gt_answer"] == "no"  and r["pred_answer"] == "no")
        _fp = sum(1 for r in src_results if r["gt_answer"] == "no"  and r["pred_answer"] == "yes")
        _fn = sum(1 for r in src_results if r["gt_answer"] == "yes" and r["pred_answer"] == "no")
        _n  = len(src_results)
        _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _rec  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        source_metrics[src] = {
            "accuracy":  (_tp + _tn) / _n,
            "precision": _prec,
            "recall":    _rec,
            "f1":        2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0,
            "tp": _tp, "tn": _tn, "fp": _fp, "fn": _fn,
            "samples":   _n,
        }

    # Per-sub-dataset (DS-MVTec, GoodsAD, MVTec-AD, MVTec-LOCO, VisA)
    sub_dataset_metrics = {}
    all_subs = sorted({r.get("sub_dataset", "unknown") for r in results if r.get("sub_dataset") and r.get("sub_dataset") != "unknown"})
    for sub in all_subs:
        subset = [r for r in results if r.get("sub_dataset") == sub]
        _tp = sum(1 for r in subset if r["gt_answer"] == "yes" and r["pred_answer"] == "yes")
        _tn = sum(1 for r in subset if r["gt_answer"] == "no"  and r["pred_answer"] == "no")
        _fp = sum(1 for r in subset if r["gt_answer"] == "no"  and r["pred_answer"] == "yes")
        _fn = sum(1 for r in subset if r["gt_answer"] == "yes" and r["pred_answer"] == "no")
        _n  = len(subset)
        _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _rec  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        sub_dataset_metrics[sub] = {
            "accuracy":  (_tp + _tn) / _n,
            "precision": _prec,
            "recall":    _rec,
            "f1":        2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0,
            "tp": _tp, "tn": _tn, "fp": _fp, "fn": _fn,
            "samples":   _n,
        }

    by_product: Dict[str, dict] = {}
    for r in results:
        prod = r.get("product", "unknown")
        if prod not in by_product:
            by_product[prod] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        key = f"{'t' if r['pred_answer'] == r['gt_answer'] else 'f'}{'p' if r['gt_answer'] == 'yes' else 'n'}"
        by_product[prod][key] += 1

    product_metrics = {}
    for prod, counts in by_product.items():
        _tp, _tn, _fp, _fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
        _total = _tp + _tn + _fp + _fn
        _prec  = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _rec   = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        product_metrics[prod] = {
            "accuracy":  (_tp + _tn) / _total if _total > 0 else 0.0,
            "precision": _prec,
            "recall":    _rec,
            "f1":        2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else 0.0,
            "samples":   _total,
        }

    return {
        "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "type_accuracy":     type_acc,
        "location_accuracy": loc_acc,
        "per_source":        source_metrics,
        "per_sub_dataset":   sub_dataset_metrics,
        "per_product":       product_metrics,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint: Optional[str],
    base_model: str,
    num_samples: int,
    output_file: str,
    realiad_only: bool,
    mmad_only: bool = False,
    mmad_all: bool = False,
    mmad_official: bool = False,
    mmad_official_iad_r1: bool = False,
    ds_mvtec_only: bool = False,
    visa_only: bool = False,
    realiad_4k: bool = False,
    batch_size: int = 4,
    load_in_4bit: bool = False,
    no_system_prompt: bool = False,
    yesno_user: bool = False,
    grpo_eval: bool = False,
    bare_question: bool = False,
    iadr1_native: bool = False,
    noreason: bool = False,
    gemini: bool = False,
    gemini_model: str = "gemini-2.5-flash",
    gemini_thinking_level: Optional[str] = None,
    openai_bench: bool = False,
    openai_model: str = "gpt-5-mini",
    openai_effort: str = "low",
    shard: int = 0,
    num_shards: int = 1,
) -> None:
    label = (f"OpenAI {openai_model}" if openai_bench else
             (f"Gemini {gemini_model}" if gemini else
              ("SFT " + Path(checkpoint).name if checkpoint else "baseline")))
    logger.info("=" * 70)
    logger.info(f"Qwen2.5-VL-7B Evaluation — {label}")
    logger.info("=" * 70)
    logger.info(f"Checkpoint:      {checkpoint or '(base model)'}")
    logger.info(f"Base model:      {base_model}")
    logger.info(f"Samples:         {num_samples if num_samples else 'all'} (50/50 balanced)")
    logger.info(f"Real-IAD only:   {realiad_only}")
    logger.info(f"MMAD only:       {mmad_only}")
    logger.info(f"MMAD all:        {mmad_all}")
    logger.info(f"MMAD official:   {mmad_official}")
    logger.info(f"MMAD official (IAD-R1 subset): {mmad_official_iad_r1}")
    logger.info(f"DS-MVTec only:   {ds_mvtec_only}")
    logger.info(f"VisA only:       {visa_only}")
    logger.info(f"RealIAD-4k:      {realiad_4k}")
    logger.info(f"No system prompt:{no_system_prompt}")
    logger.info(f"GRPO eval prompt:{grpo_eval}")
    logger.info(f"Bare question:   {bare_question}")
    logger.info(f"No-reason prompt:{noreason}")
    logger.info(f"Batch size:      {batch_size}")

    if openai_bench:
        from concurrent.futures import ThreadPoolExecutor
        oaclient = build_openai_client()
        logger.info(f"OpenAI client ready: model={openai_model} effort={openai_effort}, shard {shard}/{num_shards}")
        model, processor = None, None
    elif gemini:
        from concurrent.futures import ThreadPoolExecutor
        gclient = build_gemini_client(gemini_model)
        model, processor = None, None
    else:
        model, processor = load_model(checkpoint, base_model, load_in_4bit=load_in_4bit)

    logger.info("Building image map...")
    # Gemini/OpenAI benchmark is MMAD-only (DS-MVTec / VisA) -> skip the slow 150K+ Real-IAD rglob
    realiad_root = "" if ((gemini or openai_bench) and (ds_mvtec_only or visa_only)) else DATA_PATHS["realiad_images_root"]
    image_map = build_image_map(realiad_root, DATA_PATHS["mmad_images_root"])
    logger.info(f"Found {len(image_map)} images")

    data = load_balanced_test_data(DATA_PATHS["test"], num_samples, realiad_only, mmad_only, mmad_all, mmad_official, mmad_official_iad_r1, ds_mvtec_only, visa_only, realiad_4k)

    resolved = []
    for idx, entry in enumerate(data):
        target_path = resolve_image_path(entry.get("image_id", ""), image_map)
        if not target_path:
            logger.warning(f"[{idx+1}] Skipping — could not resolve image: {entry.get('image_id','')}")
            continue

        convs      = entry.get("conversation", [])
        first_conv = convs[0] if convs else {}
        # Extract product name from image_id e.g. "VisA/candle/test/bad/000.JPG" -> "candle"
        image_id = entry.get("image_id", "")
        parts = image_id.replace("\\", "/").split("/")
        product_name = parts[1] if len(parts) > 1 else parts[0] if parts else "object"

        raw = first_conv.get("Answer", "")
        options = first_conv.get("Options", {})
        if options and raw in options:
            # mmad.json: Options vary per entry (A=Yes/A=No depends on question)
            gt_answer = "yes" if options[raw].strip().lower().strip(".") == "yes" else "no"
        elif "annotation" in first_conv:
            gt_answer = "yes" if first_conv["annotation"] else "no"
        elif "<answer>A</answer>" in raw or "answer>yes" in raw.lower():
            gt_answer = "yes"
        else:
            gt_answer = "no"

        gt_tags = extract_tags(first_conv.get("Answer", ""))
        gt_tags["answer"] = gt_answer

        resolved.append({
            "orig_idx":    idx,
            "entry":       entry,
            "target_path": target_path,
            "question":    product_name,  # repurposed as product_name for prompt building
            "gt_answer":   gt_answer,
            "gt_tags":     gt_tags,
        })

    if num_shards > 1:
        before = len(resolved)
        resolved = resolved[shard::num_shards]
        logger.info(f"Shard {shard}/{num_shards}: {len(resolved)}/{before} samples")

    results = []
    total_resolved = len(resolved)
    logger.info(f"Evaluating {total_resolved} samples...")

    for batch_start in range(0, total_resolved, batch_size):
        batch = resolved[batch_start:batch_start + batch_size]
        try:
            batch_items = [
                (_cap_image_pixels(Image.open(b["target_path"]).convert("RGB")), b["question"])  # question = product_name
                for b in batch
            ]
            if openai_bench:
                with ThreadPoolExecutor(max_workers=max(1, len(batch_items))) as _pool:
                    predictions = list(_pool.map(
                        lambda it: openai_predict_one(oaclient, openai_model, it[0], it[1], effort=openai_effort), batch_items))
            elif gemini:
                with ThreadPoolExecutor(max_workers=max(1, len(batch_items))) as _pool:
                    predictions = list(_pool.map(
                        lambda it: gemini_predict_one(gclient, gemini_model, it[0], it[1], thinking_level=gemini_thinking_level), batch_items))
            else:
                predictions = run_inference_batch(model, processor, batch_items, no_system_prompt=no_system_prompt, grpo_eval=grpo_eval, bare_question=bare_question, iadr1_native=iadr1_native, yesno_user=yesno_user, noreason=noreason)
        except Exception as e:
            import traceback
            logger.error(f"Batch {batch_start}-{batch_start+len(batch)} error: {e}\n{traceback.format_exc()}")
            predictions = [""] * len(batch)

        for i, (b, prediction) in enumerate(zip(batch, predictions)):
            idx        = b["orig_idx"]
            gt_answer  = b["gt_answer"]
            gt_tags    = b["gt_tags"]
            entry      = b["entry"]
            pred_tags  = extract_tags(prediction)
            pred_answer = normalize_answer(pred_tags.get("answer", ""))
            # Fallback: if no <answer> tag, try fuzzy yes/no on the raw response
            if not pred_answer:
                pred_answer = fuzzy_yes_no(prediction)

            type_correct = location_correct = False
            if gt_answer == "yes":
                gt_type = gt_tags.get("type", "").lower()
                pd_type = pred_tags.get("type", "").lower()
                type_correct = bool(gt_type and pd_type and (gt_type in pd_type or pd_type in gt_type))
                gt_loc = gt_tags.get("location", "").lower()
                pd_loc = pred_tags.get("location", "").lower()
                location_correct = bool(gt_loc and pd_loc and (gt_loc in pd_loc or pd_loc in gt_loc))

            correct  = pred_answer == gt_answer
            status   = "✓" if correct else "✗"
            image_id = entry.get("image_id", "")
            source   = "mmad" if "/" in image_id else "realiad"
            sub_dataset = get_subdataset(image_id)
            done     = batch_start + i + 1

            display_tag = sub_dataset if source == "mmad" else source
            logger.info(
                f"[{done:4d}/{total_resolved}] {status} | gt={gt_answer:3s} pred={pred_answer or '?':3s} "
                f"| {display_tag:<15s} | product={entry.get('product','?')[:20]}"
            )

            results.append({
                "sample_id":        idx,
                "image_id":         image_id,
                "absolute_path":    b["target_path"],
                "product":          entry.get("product", "unknown"),
                "source":           source,
                "sub_dataset":      sub_dataset,
                "gt_answer":        gt_answer,
                "pred_answer":      pred_answer,
                "correct":          correct,
                "type_correct":     type_correct     if gt_answer == "yes" else None,
                "location_correct": location_correct if gt_answer == "yes" else None,
                "pred_full":        prediction,
                "gt_tags":          gt_tags,
                "pred_tags":        pred_tags,
            })

            if done % 5 == 0:
                _tp = sum(1 for r in results if r["gt_answer"] == "yes" and r["pred_answer"] == "yes")
                _tn = sum(1 for r in results if r["gt_answer"] == "no"  and r["pred_answer"] == "no")
                _fp = sum(1 for r in results if r["gt_answer"] == "no"  and r["pred_answer"] == "yes")
                _fn = sum(1 for r in results if r["gt_answer"] == "yes" and r["pred_answer"] == "no")
                _total = _tp + _tn + _fp + _fn
                _acc   = (_tp + _tn) / _total if _total else 0.0
                _prec  = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
                _rec   = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
                _f1    = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) else 0.0
                _no_ans = sum(1 for r in results if not r["pred_answer"])
                logger.info(
                    f"  ── rolling [{done}/{total_resolved}] "
                    f"acc={_acc*100:.1f}%  F1={_f1*100:.1f}%  "
                    f"tp={_tp} tn={_tn} fp={_fp} fn={_fn}  no_answer={_no_ans}"
                )

    metrics = compute_metrics(results)

    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS — Qwen2.5-VL-7B {label}")
    logger.info("=" * 70)
    logger.info(f"Total samples:   {metrics['total']}")
    logger.info(f"TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
    logger.info(f"Accuracy:        {metrics['accuracy']*100:.2f}%")
    logger.info(f"Precision:       {metrics['precision']*100:.2f}%")
    logger.info(f"Recall:          {metrics['recall']*100:.2f}%")
    logger.info(f"F1 Score:        {metrics['f1']*100:.2f}%")
    logger.info(f"Type Accuracy:   {metrics['type_accuracy']*100:.2f}%  (anomalies only)")
    logger.info(f"Loc. Accuracy:   {metrics['location_accuracy']*100:.2f}%  (anomalies only)")

    if metrics.get("per_source"):
        logger.info("\nPer-source breakdown:")
        for src, m in sorted(metrics["per_source"].items()):
            logger.info(
                f"  {src:<10s}  acc={m['accuracy']*100:.1f}%  F1={m['f1']*100:.1f}%  "
                f"prec={m['precision']*100:.1f}%  rec={m['recall']*100:.1f}%  n={m['samples']}"
            )

    if metrics.get("per_sub_dataset"):
        logger.info("\nPer sub-dataset breakdown (MMAD):")
        for sub, m in sorted(metrics["per_sub_dataset"].items()):
            logger.info(
                f"  {sub:<15s}  acc={m['accuracy']*100:.1f}%  F1={m['f1']*100:.1f}%  "
                f"prec={m['precision']*100:.1f}%  rec={m['recall']*100:.1f}%  n={m['samples']}"
            )

    if metrics.get("per_product"):
        logger.info("\nPer-product breakdown:")
        for prod, m in sorted(metrics["per_product"].items(), key=lambda x: -x[1]["f1"]):
            logger.info(
                f"  {prod:<30s}  acc={m['accuracy']*100:.1f}%  F1={m['f1']*100:.1f}%  n={m['samples']}"
            )

    # Record exactly which prompt was used so the file is self-describing
    # (the filename suffix alone has been a source of confusion: a file named
    #  *_trainprompt.json may have been produced under --grpo-eval).
    if openai_bench:
        prompt_mode = "openai_structured"
        system_prompt = GEMINI_SYSTEM_PROMPT
        user_prompt = make_gemini_user_text("{product_name}")
    elif gemini:
        prompt_mode = "gemini25flash_structured"
        system_prompt = GEMINI_SYSTEM_PROMPT
        user_prompt = make_gemini_user_text("{product_name}")
    elif noreason:
        prompt_mode = "noreasonprompt"
        system_prompt = None
        user_prompt = make_noreason_prompt("{product_name}")
    elif iadr1_native:
        prompt_mode = "iadr1native"
        system_prompt = "Please answer by yes or no"
        user_prompt = "Are there any defects in the test image?"
    elif bare_question:
        prompt_mode = "bareprompt"
        system_prompt = None
        user_prompt = "Are there any defects in the query image?"
    elif grpo_eval:
        prompt_mode = "grpoprompt"
        system_prompt = None
        user_prompt = GRPO_EVAL_PROMPT
    elif yesno_user:
        prompt_mode = "trainprompt_yesno_user"
        system_prompt = None
        user_prompt = "Please answer by yes or no\n" + make_train_prompt("{product_name}")
    elif no_system_prompt:
        prompt_mode = "trainprompt_nosys"
        system_prompt = None
        user_prompt = make_train_prompt("{product_name}")
    else:
        prompt_mode = "trainprompt"
        system_prompt = "Please answer by yes or no"
        user_prompt = make_train_prompt("{product_name}")
    prompt_info = {
        "prompt_mode": prompt_mode,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "checkpoint": checkpoint,
        "base_model": base_model,
        "model": openai_model if openai_bench else (gemini_model if gemini else (checkpoint or base_model)),
        "thinking_budget": (0 if (gemini and not gemini_thinking_level) else None),
        "gemini_thinking_level": gemini_thinking_level if gemini else None,
        "openai_effort": openai_effort if openai_bench else None,
        "shard": f"{shard}/{num_shards}" if num_shards > 1 else None,
    }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"prompt_info": prompt_info, "metrics": metrics, "results": results}, f, indent=2)
    logger.info(f"\nResults saved → {out_path}  [prompt_mode={prompt_mode}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL-7B SFT model")
    parser.add_argument("--checkpoint",   type=str, default=None,
                        help="Path to SFT checkpoint directory")
    parser.add_argument("--base-model",   type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Base HF model (used as baseline when --checkpoint is omitted; "
                             "always used to load the processor)")
    parser.add_argument("--num-samples",  type=int, default=200,
                        help="Total samples, split 50/50 normal/anomaly (default: 200; 0 = all balanced)")
    parser.add_argument("--output",       type=str, default=None,
                        help="Output JSON path (default: <checkpoint|outputs/qwen25vl_baseline>/eval_results.json)")
    parser.add_argument("--realiad-only", action="store_true",
                        help="Evaluate only on Real-IAD test products (in-distribution)")
    parser.add_argument("--mmad-only",    action="store_true",
                        help="Evaluate only on MMAD test samples (out-of-distribution)")
    parser.add_argument("--mmad-all",     action="store_true",
                        help="Evaluate on ALL MMAD products (train+test splits combined)")
    parser.add_argument("--mmad-official", action="store_true",
                        help="Evaluate on official MMAD benchmark entries (mmad.json, ~8366 images)")
    parser.add_argument("--mmad-official-iad-r1", action="store_true",
                        help="Evaluate on official MMAD entries restricted to DS-MVTec + VisA only (matches IAD-R1 paper datasets)")
    parser.add_argument("--ds-mvtec-only", action="store_true",
                        help="Evaluate on all 1670 DS-MVTec entries from mmad.json (no sampling)")
    parser.add_argument("--visa-only", action="store_true",
                        help="Evaluate on all VisA entries from mmad.json (no sampling)")
    parser.add_argument("--realiad-4k", action="store_true",
                        help="Evaluate on RealIAD 4k new SFT samples (new_sft_c1_train.json, 4236 images)")
    parser.add_argument("--yesno-user", action="store_true",
                        help="Put 'Please answer by yes or no' + trainprompt in the USER turn, no system message (LLaVA-parity mode)")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Omit system message — use exact training format (for fine-tuned model eval)")
    parser.add_argument("--bare-question", action="store_true",
                        help="Use bare-question prompt only ('Are there any defects in the query image?') "
                             "with no system message and no preamble. Matches SFT-Iter2 training prompt.")
    parser.add_argument("--grpo-eval", action="store_true",
                        help="Use exact GRPO training prompt (IAD-R1 style, no system message)")
    parser.add_argument("--iadr1-native", action="store_true",
                        help="Use IAD-R1's own native eval prompt (vLLM_Qwen_detect.py): system "
                             "'Please answer by yes or no' + user 'Are there any defects in the test image?'")
    parser.add_argument("--noreason-prompt", action="store_true",
                        help="noreasonprompt: native mode of the no-reasoning (labels-only) SFT "
                             "baseline. No system message; product preamble ending 'Answer with yes "
                             "or no.' Must match Training/datasets_noreason/. Suffix: _noreasonprompt")
    parser.add_argument("--gemini", action="store_true",
                        help="Benchmark Gemini-2.5-Flash (Vertex AI) instead of a local Qwen model, using the "
                             "structured think/type/location/answer prompt + thinking_budget=0. Same data/GT/metrics.")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash",
                        help="Gemini model id for --gemini (default: gemini-2.5-flash)")
    parser.add_argument("--gemini-thinking-level", type=str, default=None,
                        help="Gemini-3 thinking_level (minimal|low|high); if set, overrides thinking_budget=0")
    parser.add_argument("--openai", action="store_true",
                        help="Benchmark an OpenAI model (Responses API) instead of a local Qwen model, using the "
                             "same structured prompt as --gemini. Same data/GT/metrics.")
    parser.add_argument("--openai-model", type=str, default="gpt-5-mini",
                        help="OpenAI model id for --openai (default: gpt-5-mini)")
    parser.add_argument("--openai-effort", type=str, default="low",
                        help="Reasoning effort for --openai (minimal|low|medium|high; default: low)")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based) for sharded API eval")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (bitsandbytes)")
    parser.add_argument("--batch-size",   type=int, default=4,
                        help="Inference batch size (default: 4)")
    args = parser.parse_args()

    if args.output is None:
        base = Path(args.checkpoint) if args.checkpoint else Path("outputs/qwen25vl_baseline_eval")
        args.output = str(base / "eval_qwen25vl_results.json")

    evaluate(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        num_samples=args.num_samples,
        output_file=args.output,
        realiad_only=args.realiad_only,
        mmad_only=args.mmad_only,
        mmad_all=args.mmad_all,
        mmad_official=args.mmad_official,
        mmad_official_iad_r1=args.mmad_official_iad_r1,
        ds_mvtec_only=args.ds_mvtec_only,
        visa_only=args.visa_only,
        realiad_4k=args.realiad_4k,
        no_system_prompt=args.no_system_prompt,
        yesno_user=args.yesno_user,
        grpo_eval=args.grpo_eval,
        bare_question=args.bare_question,
        iadr1_native=args.iadr1_native,
        noreason=args.noreason_prompt,
        gemini=args.gemini,
        gemini_model=args.gemini_model,
        gemini_thinking_level=args.gemini_thinking_level,
        openai_bench=args.openai,
        openai_model=args.openai_model,
        openai_effort=args.openai_effort,
        shard=args.shard,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
    )
