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

os.environ["HF_HOME"] = "/bulk/aacudad/reasoning_traces/hf_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42

DATA_PATHS = {
    "test":  "/bulk/aacudad/reasoning_traces/Training/datasets/unified_test_zeroshot_full.json",
    "train": "/bulk/aacudad/reasoning_traces/Training/datasets/unified_train_zeroshot_full.json",
    "mmad_official": "/bulk/aacudad/reasoning_traces/MMAD_repo/dataset/MMAD/mmad.json",
    "realiad_images_root": "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/Real-IAD/images",
    "mmad_images_root":    "/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/MMAD",
}

# MMAD sub-datasets (first component of image_id path)
MMAD_SUBDATASETS = {"DS-MVTec", "GoodsAD", "MVTec-AD", "MVTec-LOCO", "VisA"}

SYSTEM_PROMPT = (
    "You are an expert industrial anomaly detector. "
    "Inspect the provided image and determine if there is an anomaly (defect).\n\n"
    "OUTPUT FORMAT (required):\n"
    "<think>Your detailed reasoning about what you observe in the image</think>\n"
    "<answer>yes or no</answer>\n\n"
    "If anomaly detected (answer=yes), also include:\n"
    "<type>defect type (e.g., scratch, dent, stain, crack)</type>\n"
    "<location>location description (e.g., top-left, center, bottom-right)</location>"
)

# IAD-R1 baseline prompt style (matches Pretrain-Inference/vLLM_Qwen_detect.py)
IAD_R1_SYSTEM_PROMPT = "Please answer by yes or no"
IAD_R1_QUESTION = "Are there any defects in the test image?"


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


def load_balanced_test_data(json_path: str, num_samples: Optional[int], realiad_only: bool, mmad_only: bool = False, mmad_all: bool = False, mmad_official: bool = False, mmad_official_iad_r1: bool = False) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)

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

def parse_iad_r1_answer(response: str) -> str:
    """Fuzzy yes/no match matching IAD-R1's get_ans() logic."""
    text = response.strip()
    # Exact match variants
    for yes_form in ("yes.", "yes", "yes!"):
        if text.lower().strip('.').strip('!') == yes_form.strip('.'):
            return "yes"
    for no_form in ("no.", "no", "no!"):
        if text.lower().strip('.').strip('!') == no_form.strip('.'):
            return "no"
    # Substring match
    low = text.lower()
    if "yes" in low:
        return "yes"
    if "no" in low:
        return "no"
    return ""


def run_inference_batch(model, processor, batch_items: list, iad_r1_prompt: bool = False) -> list:
    """batch_items: list of (image, question)"""
    texts = []
    all_images = []
    for img, question in batch_items:
        if iad_r1_prompt:
            messages = [
                {"role": "system", "content": IAD_R1_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": IAD_R1_QUESTION},
                ]},
            ]
        else:
            messages = [{"role": "user", "content": [
                {"type": "text",  "text": SYSTEM_PROMPT + "\n\n"},
                {"type": "image", "image": img},
                {"type": "text",  "text": f"\nQuestion: {question}"},
            ]}]
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
    batch_size: int = 4,
    load_in_4bit: bool = False,
    iad_r1_prompt: bool = False,
) -> None:
    label = "SFT " + Path(checkpoint).name if checkpoint else "baseline"
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
    logger.info(f"IAD-R1 prompt:   {iad_r1_prompt}")
    logger.info(f"Batch size:      {batch_size}")

    model, processor = load_model(checkpoint, base_model, load_in_4bit=load_in_4bit)

    logger.info("Building image map...")
    image_map = build_image_map(DATA_PATHS["realiad_images_root"], DATA_PATHS["mmad_images_root"])
    logger.info(f"Found {len(image_map)} images")

    data = load_balanced_test_data(DATA_PATHS["test"], num_samples, realiad_only, mmad_only, mmad_all, mmad_official, mmad_official_iad_r1)

    resolved = []
    for idx, entry in enumerate(data):
        target_path = resolve_image_path(entry.get("image_id", ""), image_map)
        if not target_path:
            logger.warning(f"[{idx+1}] Skipping — could not resolve image: {entry.get('image_id','')}")
            continue

        convs      = entry.get("conversation", [])
        first_conv = convs[0] if convs else {}
        question   = first_conv.get("Question", "Are there any defects in this image?")

        if "annotation" in first_conv:
            gt_answer = "yes" if first_conv["annotation"] else "no"
        else:
            raw = first_conv.get("Answer", "B")
            gt_answer = "yes" if ("<answer>A</answer>" in raw or "answer>yes" in raw.lower()) else "no"

        gt_tags = extract_tags(first_conv.get("Answer", ""))
        gt_tags["answer"] = gt_answer

        resolved.append({
            "orig_idx":    idx,
            "entry":       entry,
            "target_path": target_path,
            "question":    question,
            "gt_answer":   gt_answer,
            "gt_tags":     gt_tags,
        })

    results = []
    total_resolved = len(resolved)
    logger.info(f"Evaluating {total_resolved} samples...")

    for batch_start in range(0, total_resolved, batch_size):
        batch = resolved[batch_start:batch_start + batch_size]
        try:
            batch_items = [
                (Image.open(b["target_path"]).convert("RGB"), b["question"])
                for b in batch
            ]
            predictions = run_inference_batch(model, processor, batch_items, iad_r1_prompt=iad_r1_prompt)
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
            if iad_r1_prompt:
                pred_answer = parse_iad_r1_answer(prediction)
            else:
                pred_answer = normalize_answer(pred_tags.get("answer", ""))

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

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    logger.info(f"\nResults saved → {out_path}")


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
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (bitsandbytes)")
    parser.add_argument("--batch-size",   type=int, default=4,
                        help="Inference batch size (default: 4)")
    parser.add_argument("--iad-r1-prompt", action="store_true",
                        help="Use IAD-R1 baseline prompt style: system='Please answer by yes or no', "
                             "question='Are there any defects in the test image?' with fuzzy yes/no parsing")
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
        iad_r1_prompt=args.iad_r1_prompt,
        mmad_official_iad_r1=args.mmad_official_iad_r1,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
    )
