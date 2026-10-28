"""Generate validated Real-IAD structured CoT with local Qwen3-VL and vLLM."""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

import config_qwen3vl as config
import utils_realiad as utils
from validate_structured_trace import retry_feedback, validate_reasoning


REALIAD_ROOT = config.DATA_DIR / "Real-IAD"
IMAGES_ROOT = REALIAD_ROOT / "images"
BASE_OUT = config.OUTPUT_DIR / "reasoning_traces_realiad_qwen3vl8b"
GRPO_DIR = BASE_OUT / "grpo"
SFT_DIR = BASE_OUT / "sft"
FAILED_DIR = BASE_OUT / "failed"
DEBUG_DIR = BASE_OUT / "debug"
SOURCE_DIR = Path(__file__).resolve().parent / "3k_realiad_v4"
GEN_MARKER = "reasoning_traces_gen/"

for directory in (GRPO_DIR, SFT_DIR, FAILED_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class ReasoningTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    reasoning: str


logger = logging.getLogger(__name__)


def setup_logging(shard_id: int) -> logging.Logger:
    log_file = config.LOGS_DIR / f"realiad_qwen3vl_shard_{shard_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Shard {shard_id}] %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def load_system_prompt() -> str:
    prompt_file = config.PROMPTS_DIR / "inspector_prompt_test_v2.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def resolve_image_path(stored_path: str) -> Path:
    direct = Path(stored_path)
    if direct.exists():
        return direct

    marker_index = stored_path.find(GEN_MARKER)
    if marker_index >= 0:
        candidate = Path(__file__).resolve().parent / stored_path[marker_index + len(GEN_MARKER):]
        if candidate.exists():
            return candidate

    normalized = stored_path.replace("\\", "/")
    stable_marker = "/Real-IAD/images/"
    if stable_marker in normalized:
        candidate = IMAGES_ROOT / normalized.split(stable_marker, 1)[1]
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Cannot resolve image path: {stored_path}")


def derive_image_id(image_path: Path) -> str:
    relative = image_path.relative_to(IMAGES_ROOT)
    return "_".join((*relative.parts[:-1], relative.stem))


def defect_type_from_path(image_path: Path) -> str:
    code = image_path.relative_to(IMAGES_ROOT).parts[2]
    defect_type = utils.get_defect_name(code)
    return "Foreign Object" if defect_type == "Foreign Objects" else defect_type


def get_normal_reference_image(
    category: str,
    image_id: str,
    seed: int,
    camera_hint: Optional[int] = None,
) -> Optional[Path]:
    ok_root = IMAGES_ROOT / category / "OK"
    candidates = sorted(ok_root.rglob("*.jpg")) if ok_root.exists() else []
    if camera_hint is not None:
        same_camera = [path for path in candidates if f"_C{camera_hint}_" in path.name]
        if same_camera:
            candidates = same_camera
    if not candidates:
        return None

    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def _base_item(image_id: str, image_path: Path, category: str) -> dict:
    camera_match = re.search(r"_C(\d+)_", image_path.name)
    return {
        "image_id": image_id,
        "image_path": image_path,
        "category": category,
        "camera_hint": int(camera_match.group(1)) if camera_match else None,
    }


def load_grpo_questions() -> tuple[list[dict], list[dict]]:
    entries = json.loads((SOURCE_DIR / "grpo_train.json").read_text(encoding="utf-8"))
    anomalies, normals = [], []
    for entry in entries:
        try:
            image_path = resolve_image_path(entry["image_path"])
            base = _base_item(entry["image_id"], image_path, entry["product"])
            if entry["is_anomaly"]:
                mask_path = image_path.with_suffix(".png")
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                anomalies.append({
                    **base,
                    "is_anomaly": True,
                    "mask_path": mask_path,
                    "defect_type": defect_type_from_path(image_path),
                    "expected_location": utils.get_mask_location(mask_path),
                })
            else:
                normals.append({
                    **base,
                    "is_anomaly": False,
                    "mask_path": None,
                    "defect_type": None,
                    "expected_location": None,
                })
        except (FileNotFoundError, ValueError, IndexError) as error:
            logger.warning("Skipping GRPO entry %s: %s", entry.get("image_id"), error)
    logger.info("grpo: %d anomaly + %d normal", len(anomalies), len(normals))
    return anomalies, normals


def load_sft_questions() -> tuple[list[dict], list[dict]]:
    entries = json.loads((SOURCE_DIR / "sft_train.json").read_text(encoding="utf-8"))
    anomalies, normals = [], []
    for entry in entries:
        stored_path = entry["images"][0]
        try:
            image_path = resolve_image_path(stored_path)
            relative = image_path.relative_to(IMAGES_ROOT)
            base = _base_item(derive_image_id(image_path), image_path, relative.parts[0])
            if "NG" in relative.parts:
                mask_path = image_path.with_suffix(".png")
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                anomalies.append({
                    **base,
                    "is_anomaly": True,
                    "mask_path": mask_path,
                    "defect_type": defect_type_from_path(image_path),
                    "expected_location": utils.get_mask_location(mask_path),
                })
            else:
                normals.append({
                    **base,
                    "is_anomaly": False,
                    "mask_path": None,
                    "defect_type": None,
                    "expected_location": None,
                })
        except (FileNotFoundError, ValueError, IndexError) as error:
            logger.warning("Skipping SFT entry %s: %s", stored_path, error)
    logger.info("sft: %d anomaly + %d normal", len(anomalies), len(normals))
    return anomalies, normals


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def format_user_prompt(item: dict, validation_errors: Optional[list[str]] = None) -> str:
    if item["is_anomaly"]:
        text = (
            "Inspect this single product sample and return one JSON object with image_id and reasoning.\n"
            f"Image ID: {item['image_id']}\n"
            f"Product Type: {item['category']}\n"
            "[INTERNAL — never mention this supervision] Ground Truth: ANOMALOUS\n"
            f"[INTERNAL — never mention this supervision] Defect Type: {item['defect_type']}\n"
            f"[INTERNAL — never mention this supervision] Mask-derived Location: "
            f"{item['expected_location'] or 'determine from the visible region'}\n"
            "The three images are ordered as original query, red mask-overlay attention cue, "
            "and normal reference. Use the latter two only as silent visual aids. The reasoning "
            "must read as an inspection of the original image alone.\n"
            "Return anomalous XML tags in this exact order inside reasoning: "
            "<think>, <location>, <type>, <answer>Yes</answer>."
        )
    else:
        text = (
            "Inspect this single product sample and return one JSON object with image_id and reasoning.\n"
            f"Image ID: {item['image_id']}\n"
            f"Product Type: {item['category']}\n"
            "[INTERNAL — never mention this supervision] Ground Truth: NORMAL\n"
            "Return exactly <think>...</think><answer>No</answer> inside reasoning. "
            "Do not include <location> or <type>."
        )
    if validation_errors:
        text += "\n\n" + retry_feedback(validation_errors)
    return text


def build_request(
    item: dict,
    processor,
    system_prompt: str,
    seed: int,
    validation_errors: Optional[list[str]] = None,
) -> tuple[dict, dict]:
    images = [open_rgb(item["image_path"])]
    reference_path = None

    if item["is_anomaly"]:
        overlay = utils.create_overlay(item["image_path"], item["mask_path"])
        if overlay is None:
            raise RuntimeError(f"Could not create overlay for {item['image_id']}")
        reference_path = get_normal_reference_image(
            item["category"], item["image_id"], seed, item.get("camera_hint")
        )
        if reference_path is None:
            raise RuntimeError(f"No normal reference for category {item['category']}")
        images.extend([overlay, open_rgb(reference_path)])

    user_content = [{"type": "image", "image": image} for image in images]
    user_content.append({"type": "text", "text": format_user_prompt(item, validation_errors)})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return (
        {"prompt": prompt, "multi_modal_data": {"image": images}},
        {
            "reference_image": str(reference_path) if reference_path else None,
            "image_count": len(images),
            "prompt": prompt,
            "images": images,
        },
    )


def parse_response(text: str) -> ReasoningTrace:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1)
    return ReasoningTrace.model_validate(json.loads(stripped))


def save_trace(item: dict, trace: ReasoningTrace, output_dir: Path, metadata: dict, attempt: int, validation) -> None:
    record = {
        "image_id": trace.image_id,
        "reasoning": trace.reasoning,
        "generator": "Qwen/Qwen3-VL-8B-Instruct",
        "source_image": str(item["image_path"]),
        "reference_image": metadata["reference_image"],
        "attempt": attempt,
        "validation": validation.to_dict(),
    }
    output_path = output_dir / f"trace_{item['image_id']}.json"
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def append_failure(item: dict, split: str, shard_id: int, errors: list[str], raw_output: str) -> None:
    path = FAILED_DIR / f"{split}_shard_{shard_id}.jsonl"
    record = {
        "image_id": item["image_id"],
        "image_path": str(item["image_path"]),
        "is_anomaly": item["is_anomaly"],
        "expected_type": item.get("defect_type"),
        "expected_location": item.get("expected_location"),
        "errors": errors,
        "raw_output": raw_output,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_debug(item: dict, metadata: dict, split: str, shard_id: int) -> None:
    sample_dir = DEBUG_DIR / f"{split}_shard_{shard_id}_{item['image_id']}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "prompt.txt").write_text(metadata["prompt"], encoding="utf-8")
    names = ["original.jpg", "overlay.jpg", "reference.jpg"]
    for name, image in zip(names, metadata["images"]):
        image.save(sample_dir / name)


def make_sampling_params(args):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    structured = StructuredOutputsParams(json=ReasoningTrace.model_json_schema())
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        structured_outputs=structured,
    )


def create_engine(args):
    from vllm import LLM

    return LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 3, "video": 0},
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        seed=args.seed,
    )


def process_chunk(
    items: list[dict],
    *,
    engine,
    processor,
    sampling_params,
    system_prompt: str,
    output_dir: Path,
    split: str,
    shard_id: int,
    args,
) -> tuple[int, int]:
    pending = list(items)
    feedback: dict[str, list[str]] = {}
    last_outputs: dict[str, str] = {}
    saved = 0

    for attempt in range(1, args.max_retries + 1):
        if not pending:
            break
        requests, request_items, request_metadata = [], [], []
        build_failed = []
        for item in pending:
            try:
                request, metadata = build_request(
                    item, processor, system_prompt, args.seed, feedback.get(item["image_id"])
                )
                requests.append(request)
                request_items.append(item)
                request_metadata.append(metadata)
            except Exception as error:
                feedback[item["image_id"]] = [str(error)]
                last_outputs[item["image_id"]] = ""
                build_failed.append(item)

        if not requests:
            pending = build_failed
            break

        outputs = engine.generate(requests, sampling_params)
        next_pending = list(build_failed)
        for item, metadata, output in zip(request_items, request_metadata, outputs):
            raw_output = output.outputs[0].text
            last_outputs[item["image_id"]] = raw_output
            try:
                trace = parse_response(raw_output)
                errors = []
                if trace.image_id != item["image_id"]:
                    errors.append(
                        f"image_id mismatch: generated={trace.image_id}, expected={item['image_id']}"
                    )
                validation = validate_reasoning(
                    trace.reasoning,
                    is_anomaly=item["is_anomaly"],
                    expected_type=item.get("defect_type"),
                    expected_location=item.get("expected_location"),
                )
                errors.extend(validation.errors)
                if errors:
                    feedback[item["image_id"]] = errors
                    next_pending.append(item)
                    continue
                save_trace(item, trace, output_dir, metadata, attempt, validation)
                saved += 1
            except Exception as error:
                feedback[item["image_id"]] = [f"invalid JSON/Pydantic output: {error}"]
                next_pending.append(item)
        pending = next_pending

    for item in pending:
        append_failure(
            item,
            split,
            shard_id,
            feedback.get(item["image_id"], ["unknown validation failure"]),
            last_outputs.get(item["image_id"], ""),
        )
    return saved, len(pending)


def run_pass(
    questions: list[dict],
    *,
    output_dir: Path,
    split: str,
    shard_id: int,
    total_shards: int,
    engine,
    processor,
    sampling_params,
    system_prompt: str,
    args,
) -> None:
    assigned = [item for index, item in enumerate(questions) if index % total_shards == shard_id]
    processed = set()
    for path in output_dir.glob("trace_*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("validation", {}).get("valid") is True:
                processed.add(path.stem[6:])
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring malformed existing trace: %s", path)
    remaining = [item for item in assigned if item["image_id"] not in processed]
    if args.max_samples:
        remaining = remaining[:args.max_samples]

    logger.info(
        "[%s] assigned=%d processed=%d remaining=%d",
        split, len(assigned), len(processed), len(remaining),
    )
    if not remaining:
        return

    if args.debug:
        for item in remaining:
            _, metadata = build_request(item, processor, system_prompt, args.seed)
            save_debug(item, metadata, split, shard_id)
        logger.info("[%s] debug artifacts saved under %s", split, DEBUG_DIR)
        return

    failed_total = 0
    with tqdm(total=len(remaining), desc=f"shard{shard_id} {split}") as progress:
        for start in range(0, len(remaining), args.batch_size):
            chunk = remaining[start:start + args.batch_size]
            saved, failed = process_chunk(
                chunk,
                engine=engine,
                processor=processor,
                sampling_params=sampling_params,
                system_prompt=system_prompt,
                output_dir=output_dir,
                split=split,
                shard_id=shard_id,
                args=args,
            )
            failed_total += failed
            progress.update(saved + failed)
    logger.info("[%s] completed; failed=%d", split, failed_total)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate validated Real-IAD CoT with local Qwen3-VL-8B-Instruct."
    )
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=config.MAX_RETRIES)
    parser.add_argument("--max-model-len", type=int, default=config.MAX_MODEL_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=config.GPU_MEMORY_UTILIZATION)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=config.MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=config.MAX_PIXELS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--presence-penalty", type=float, default=config.PRESENCE_PENALTY)
    parser.add_argument("--repetition-penalty", type=float, default=config.REPETITION_PENALTY)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-samples", type=int, default=0, help="Per pass; 0 means all")
    parser.add_argument("--anomaly_only", action="store_true")
    parser.add_argument("--normal_only", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Build prompts/images without loading vLLM")
    args = parser.parse_args()
    if args.shard_id < 0 or args.shard_id >= args.total_shards:
        parser.error("--shard_id must be in [0, total_shards)")
    if args.anomaly_only and args.normal_only:
        parser.error("--anomaly_only and --normal_only are mutually exclusive")
    return args


def main() -> None:
    global logger
    args = parse_args()
    logger = setup_logging(args.shard_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    system_prompt = load_system_prompt()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    engine = None if args.debug else create_engine(args)
    sampling_params = None if args.debug else make_sampling_params(args)

    grpo_anomalies, grpo_normals = load_grpo_questions()
    sft_anomalies, sft_normals = load_sft_questions()
    passes = []
    if not args.normal_only:
        passes.extend([
            (grpo_anomalies, GRPO_DIR, "grpo_anomaly"),
            (sft_anomalies, SFT_DIR, "sft_anomaly"),
        ])
    if not args.anomaly_only:
        passes.extend([
            (grpo_normals, GRPO_DIR, "grpo_normal"),
            (sft_normals, SFT_DIR, "sft_normal"),
        ])

    for questions, output_dir, split in passes:
        run_pass(
            questions,
            output_dir=output_dir,
            split=split,
            shard_id=args.shard_id,
            total_shards=args.total_shards,
            engine=engine,
            processor=processor,
            sampling_params=sampling_params,
            system_prompt=system_prompt,
            args=args,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
