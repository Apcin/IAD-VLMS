"""Compile only validated Qwen3-VL traces into SFT and GRPO datasets."""

import argparse
import json
import os
from pathlib import Path

import utils_realiad as utils
from validate_structured_trace import validate_reasoning

BASE = Path(__file__).parent
IMAGES_ROOT = BASE / "data" / "Real-IAD" / "images"

_GEN_MARKER = "reasoning_traces_gen/"


def _linux_to_local(linux_path: str) -> Path:
    direct = Path(linux_path)
    if direct.exists():
        return direct

    idx = linux_path.find(_GEN_MARKER)
    if idx >= 0:
        rel = linux_path[idx + len(_GEN_MARKER):].replace("/", os.sep)
        candidate = BASE / rel
        if candidate.exists():
            return candidate

    normalized = linux_path.replace("\\", "/")
    stable_marker = "/Real-IAD/images/"
    if stable_marker in normalized:
        candidate = IMAGES_ROOT / normalized.split(stable_marker, 1)[1]
        if candidate.exists():
            return candidate

    raise ValueError(f"Cannot resolve: {linux_path}")


def _derive_image_id(img_path: Path) -> str:
    rel = img_path.relative_to(IMAGES_ROOT)
    return "_".join(list(rel.parts[:-1]) + [rel.stem])


def load_traces(trace_dir: Path, subdir: str) -> dict:
    trace_path = trace_dir / subdir
    traces = {}
    rejected = 0
    for f in trace_path.glob("trace_*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not data.get("validation", {}).get("valid", False):
                rejected += 1
                continue
            traces[data["image_id"]] = data
    if rejected:
        print(f"  Rejected unvalidated traces in {subdir}: {rejected}")
    return traces


def validate_trace_record(record: dict, image_path: Path, is_anomaly: bool) -> tuple[bool, list[str]]:
    expected_type = None
    expected_location = None
    if is_anomaly:
        relative = image_path.relative_to(IMAGES_ROOT)
        defect_code = relative.parts[2]
        expected_type = utils.get_defect_name(defect_code)
        if expected_type == "Foreign Objects":
            expected_type = "Foreign Object"
        mask_path = image_path.with_suffix(".png")
        if not mask_path.exists():
            return False, [f"mask not found: {mask_path}"]
        expected_location = utils.get_mask_location(mask_path)
        if not expected_location:
            return False, [f"mask has no valid foreground: {mask_path}"]

    validation = validate_reasoning(
        record.get("reasoning", ""),
        is_anomaly=is_anomaly,
        expected_type=expected_type,
        expected_location=expected_location,
    )
    return validation.valid, validation.errors


def compile_grpo(source_dir: Path, trace_dir: Path, output_dir: Path) -> int:
    with open(source_dir / "grpo_train.json", "r", encoding="utf-8") as f:
        original = json.load(f)

    traces = load_traces(trace_dir, "grpo")
    print(f"GRPO: {len(traces)} traces, {len(original)} original entries")

    output = []
    missing = 0
    for entry in original:
        iid = entry["image_id"]
        if iid not in traces:
            missing += 1
            continue
        try:
            image_path = _linux_to_local(entry["image_path"])
            valid, errors = validate_trace_record(traces[iid], image_path, entry["is_anomaly"])
        except (OSError, ValueError, IndexError) as error:
            valid, errors = False, [str(error)]
        if not valid:
            print(f"  Rejected {iid}: {'; '.join(errors)}")
            missing += 1
            continue
        output.append({
            "image_id": iid,
            "image_path": str(image_path.resolve()),
            "product": entry["product"],
            "is_anomaly": entry["is_anomaly"],
            "question": entry["question"],
            "answer": traces[iid]["reasoning"],
            "gt_label": entry["gt_label"],
        })

    print(f"  Matched: {len(output)}, Missing: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "grpo_train.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  -> {output_dir / 'grpo_train.json'}")
    return missing


def compile_sft(source_dir: Path, trace_dir: Path, output_dir: Path) -> int:
    with open(source_dir / "sft_train.json", "r", encoding="utf-8") as f:
        original = json.load(f)

    traces = load_traces(trace_dir, "sft")
    print(f"SFT:  {len(traces)} traces, {len(original)} original entries")

    output = []
    missing = 0
    for entry in original:
        img_linux = entry["images"][0]
        try:
            img_local = _linux_to_local(img_linux)
            image_id = _derive_image_id(img_local)
        except (OSError, ValueError):
            missing += 1
            continue

        if image_id not in traces:
            missing += 1
            continue

        is_anomaly = "NG" in img_local.relative_to(IMAGES_ROOT).parts
        try:
            valid, errors = validate_trace_record(traces[image_id], img_local, is_anomaly)
        except (OSError, ValueError, IndexError) as error:
            valid, errors = False, [str(error)]
        if not valid:
            print(f"  Rejected {image_id}: {'; '.join(errors)}")
            missing += 1
            continue

        messages = []
        for msg in entry["messages"]:
            if msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": traces[image_id]["reasoning"]})
            else:
                messages.append(msg)

        output.append({
            "messages": messages,
            "images": [str(img_local.resolve())],
        })

    print(f"  Matched: {len(output)}, Missing: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sft_train.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  -> {output_dir / 'sft_train.json'}")
    return missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=BASE / "3k_realiad_v4")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=BASE / "data" / "output" / "reasoning_traces_realiad_qwen3vl8b",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "realiad_qwen3vl8b_compiled",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow a partial dataset; by default any missing trace makes compilation fail.",
    )
    args = parser.parse_args()

    grpo_missing = compile_grpo(args.source_dir, args.trace_dir, args.output_dir)
    print()
    sft_missing = compile_sft(args.source_dir, args.trace_dir, args.output_dir)
    print("\nDone!")
    if (grpo_missing or sft_missing) and not args.allow_missing:
        raise SystemExit(
            f"Compilation is incomplete: GRPO missing={grpo_missing}, SFT missing={sft_missing}. "
            "Regenerate failed traces or pass --allow-missing explicitly."
        )
