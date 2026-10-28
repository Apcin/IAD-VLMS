"""Fast structural audit for the four generated manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import config
from generate_traces_qwen3vl import format_user_prompt, ground_truth_location_value, teacher_image_paths


def audit_dataset(dataset: str) -> dict:
    path = config.MANIFEST_DIR / f"{dataset}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors, ids = [], set()
    stats = Counter()
    type_sources = Counter()

    for sample in manifest["samples"]:
        sample_id = sample["id"]
        if sample_id in ids:
            errors.append(f"duplicate id: {sample_id}")
        ids.add(sample_id)
        stats[sample["label"]] += 1

        for image in sample["query_images"] + sample["normal_reference"]["images"]:
            if not Path(image["path"]).exists():
                errors.append(f"missing image: {image['path']}")

        supervision = sample["supervision"]
        prompt = format_user_prompt(sample)

        if sample["is_anomaly"]:
            if supervision["bbox_coordinates_exposed_to_teacher"] is not True:
                errors.append(f"bbox coordinate exposure disabled: {sample_id}")
            expected_location = json.dumps(
                ground_truth_location_value(sample), separators=(",", ":"), ensure_ascii=False
            )
            if expected_location not in prompt:
                errors.append(f"GT bbox missing from teacher prompt: {sample_id}")
            anomaly_type = supervision.get("anomaly_type") or {}
            if not anomaly_type.get("canonical"):
                errors.append(f"empty anomaly type: {sample_id}")
            type_sources[anomaly_type.get("source", "missing")] += 1
            if not supervision["bboxes"]:
                errors.append(f"no GT bbox: {sample_id}")
            for item in supervision["mask_paths"] + supervision["bbox_prompt_images"]:
                if not Path(item["path"]).exists():
                    errors.append(f"missing supervision file: {item['path']}")
            teacher_paths = set(teacher_image_paths(sample))
            for item in supervision["mask_paths"]:
                if item["path"] in teacher_paths:
                    errors.append(f"source mask leaked into teacher images: {sample_id}")
        elif any(
            (
                supervision["mask_paths"],
                supervision["bbox_prompt_images"],
                supervision["bboxes"],
                supervision["anomaly_type"] is not None,
            )
        ):
            errors.append(f"normal sample has anomaly supervision: {sample_id}")
        elif supervision["bbox_coordinates_exposed_to_teacher"] is not False:
            errors.append(f"normal sample enables bbox coordinate exposure: {sample_id}")

    expected_counts = manifest["counts"]
    observed = {"total": len(ids), "anomaly": stats["Anomaly"], "normal": stats["Normal"]}
    if observed != expected_counts:
        errors.append(f"count mismatch: observed={observed}, manifest={expected_counts}")
    return {
        "dataset": dataset,
        "counts": observed,
        "type_sources": dict(type_sources),
        "errors": errors,
    }


def main() -> None:
    failed = False
    for dataset in config.DATASETS:
        result = audit_dataset(dataset)
        print(dataset, result["counts"], "type_sources=", result["type_sources"])
        if result["errors"]:
            failed = True
            for error in result["errors"][:20]:
                print("  ERROR", error)
            if len(result["errors"]) > 20:
                print(f"  ... {len(result['errors']) - 20} more errors")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
