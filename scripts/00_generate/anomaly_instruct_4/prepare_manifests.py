"""Build one deduplicated, read-only-source manifest for each supported dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import config
from dataset_utils import (
    anomaly_description,
    canonical_type,
    deduplicated_records,
    existing_bbox_path,
    image_paths,
    image_size,
    json_dump,
    mask_bbox,
    mask_paths,
    matching_reference_views,
    mvtec_type,
    normalize_bbox_1000,
    object_category,
    safe_id,
    stable_choice,
    view_id,
)


def _view_records(paths: list[Path], modality: str) -> list[dict]:
    output = []
    for path in paths:
        width, height = image_size(path)
        output.append(
            {
                "path": str(path),
                "view_id": view_id(path, modality),
                "width": width,
                "height": height,
            }
        )
    return output


def _normal_pools(selected: list[tuple[str, dict, list[str]]]) -> dict[str, list[tuple[str, dict]]]:
    pools: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, record, _ in selected:
        if record.get("metadata", {}).get("anomaly") is False:
            pools[object_category(key)].append((key, record))
    for category in pools:
        pools[category].sort(key=lambda item: item[0])
    return pools


def _type_annotation(record: dict, dataset: str) -> dict:
    description = anomaly_description(record)
    if dataset == "mvtec_ad":
        label = mvtec_type(record, dataset)
        return {
            "raw": label,
            "canonical": label,
            "source": "path_gt",
            "confidence": "high",
            "description": description,
        }
    return {
        "raw": description,
        "canonical": canonical_type(description),
        "source": "conversation_derived",
        "confidence": "medium",
        "description": description,
    }


def _spatial_supervision(record: dict, dataset: str, queries: list[Path]) -> tuple[list[dict], list[dict]]:
    masks = mask_paths(record, dataset)
    mask_by_name = {path.name: path for path in masks}
    mask_records, boxes = [], []
    modality = config.DATASETS[dataset]["target_modality"]
    for query in queries:
        mask = mask_by_name.get(query.name)
        if mask is None:
            if len(masks) == 1 and len(queries) == 1:
                mask = masks[0]
            else:
                raise ValueError(f"No mask matching query {query} in {record.get('id')}")
        bbox, mask_size = mask_bbox(mask, allow_empty=modality != "2d")
        query_size = image_size(query)
        if mask_size != query_size:
            raise ValueError(f"mask/query size mismatch: {mask} vs {query}")
        current_view = view_id(query, modality)
        mask_records.append(
            {"path": str(mask), "view_id": current_view, "has_foreground": bbox is not None}
        )
        if bbox is None:
            continue
        boxes.append(
            {
                "view_id": current_view,
                "xyxy_pixel": bbox,
                "xyxy_1000": normalize_bbox_1000(bbox, query_size),
                "coordinate_convention": "inclusive_xyxy",
                "source": "gt_mask",
            }
        )
    if not boxes:
        raise ValueError(f"No selected query view contains mask foreground for {record.get('id')}")
    return mask_records, boxes


def build_manifest(
    dataset: str, *, seed: int, require_bbox_prompts: bool, max_samples: int = 0
) -> dict:
    spec = config.DATASETS[dataset]
    selected = deduplicated_records(dataset)
    normal_pools = _normal_pools(selected)
    samples = []
    records_to_build = selected[:max_samples] if max_samples else selected

    for key, record, source_ids in records_to_build:
        is_anomaly = record.get("metadata", {}).get("anomaly") is True
        category = object_category(key)
        queries = image_paths(record, dataset)
        modality = spec["target_modality"]
        identity = safe_id(dataset, key)

        candidates = normal_pools.get(category, [])
        candidates = [(candidate_key, candidate) for candidate_key, candidate in candidates if candidate_key != key]
        reference_key, reference_record = stable_choice(candidates, identity, seed)
        references = matching_reference_views(reference_record, dataset, queries)

        sample = {
            "id": identity,
            "dataset": dataset,
            "sample_key": key,
            "source_instruction_ids": source_ids,
            "target_modality": modality,
            "object_category": category,
            "label": "Anomaly" if is_anomaly else "Normal",
            "is_anomaly": is_anomaly,
            "query_images": _view_records(queries, modality),
            "normal_reference": {
                "sample_key": reference_key,
                "selection": "same_category_sha256_seeded",
                "seed": seed,
                "images": _view_records(references, modality),
            },
            "question": (
                "Are there any defects in this query image?"
                if modality == "2d"
                else "Are there any defects in this query object shown from multiple views?"
            ),
            "supervision": {
                "bbox_coordinates_exposed_to_teacher": is_anomaly,
                "mask_usage": "program_side_bbox_derivation_and_audit_only",
                "mask_paths": [],
                "bbox_prompt_images": [],
                "bboxes": [],
                "anomaly_type": None,
            },
        }

        if is_anomaly:
            masks, boxes = _spatial_supervision(record, dataset, queries)
            bbox_prompts = []
            for query in queries:
                bbox_path = existing_bbox_path(query, dataset)
                if require_bbox_prompts and not bbox_path.exists():
                    raise FileNotFoundError(
                        f"bbox prompt missing: {bbox_path}; run build_mvtec3d_bboxes.py first"
                    )
                bbox_prompts.append(
                    {
                        "path": str(bbox_path),
                        "view_id": view_id(query, modality),
                        "source": "generated_from_mask" if dataset == "mvtec3d" else "provided",
                        "exists": bbox_path.exists(),
                    }
                )
            sample["supervision"] = {
                "bbox_coordinates_exposed_to_teacher": True,
                "mask_usage": "program_side_bbox_derivation_and_audit_only",
                "mask_paths": masks,
                "bbox_prompt_images": bbox_prompts,
                "bboxes": boxes,
                "anomaly_type": _type_annotation(record, dataset),
            }
        samples.append(sample)

    counts = Counter(sample["label"] for sample in samples)
    return {
        "schema_version": "anomaly-instruct-cot-manifest-v2-gt-guided",
        "dataset": dataset,
        "source_root": str(config.SOURCE_ROOT),
        "source_policy": "read_only",
        "teacher_policy": {
            "image_order": "query images, bbox prompt images for anomalies, normal reference images",
            "bbox_coordinates_in_prompt": True,
            "mask_images_in_prompt": False,
            "final_training_images": "query images only",
        },
        "counts": {"total": len(samples), "anomaly": counts["Anomaly"], "normal": counts["Normal"]},
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare four Anomaly-Instruct CoT manifests.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(config.DATASETS), default=sorted(config.DATASETS))
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-samples", type=int, default=0, help="Per dataset; 0 means all")
    parser.add_argument(
        "--allow-missing-mvtec3d-bboxes",
        action="store_true",
        help="Write planned derived paths before running build_mvtec3d_bboxes.py.",
    )
    args = parser.parse_args()
    if not config.SOURCE_ROOT.exists():
        parser.error(f"source root not found: {config.SOURCE_ROOT}")

    for dataset in args.datasets:
        manifest = build_manifest(
            dataset,
            seed=args.seed,
            require_bbox_prompts=not args.allow_missing_mvtec3d_bboxes,
            max_samples=args.max_samples,
        )
        output = config.MANIFEST_DIR / f"{dataset}.json"
        json_dump(output, manifest)
        print(dataset, manifest["counts"], "->", output)


if __name__ == "__main__":
    main()
