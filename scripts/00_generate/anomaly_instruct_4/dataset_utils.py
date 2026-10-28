"""Read-only source adapters and shared annotation helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

import config


TYPE_PATTERNS = (
    (r"missing (?:part|component|piece|section)", "missing part"),
    (r"foreign (?:object|material|particle|contaminant)", "foreign object"),
    (r"dark(?:er)?[^.,;]{0,32}(?:spot|patch|area|mark)", "dark spot"),
    (r"discolou?ration|discolou?red|off-color|color (?:change|variation|anomaly|difference)", "discoloration"),
    (r"concavit(?:y|ies)|concave area", "concavity"),
    (r"depression|sink(?: mark| area)?", "concavity"),
    (r"indentation|indented area", "indentation"),
    (r"protrusion|protruding area|raised (?:area|bump)", "protrusion"),
    (r"deformation|deformed area", "deformation"),
    (r"contamination|contaminant", "contamination"),
    (r"rough(?:ness| patch| area| texture)", "roughness"),
    (r"misalignment|misaligned", "misalignment"),
    (r"irregular (?:hole|opening)", "hole"),
    (r"gouge|cavit(?:y|ies)", "cavity"),
    (r"scratch(?:ed|es)?", "scratch"),
    (r"crack(?:ed|s)?", "crack"),
    (r"bulge|bulging", "bulge"),
    (r"dent(?:ed|s)?", "dent"),
    (r"hole|perforation", "hole"),
    (r"broken|breakage", "broken"),
    (r"chip(?:ped|ping)?", "chip"),
    (r"cut|tear|torn", "cut"),
    (r"bent|bend", "bent"),
    (r"stain(?:ed)?", "stain"),
    (r"burn(?:ed|t)?|scorch(?:ed|ing)?", "burn"),
    (r"material loss|missing material", "missing material"),
    (r"damage(?:d)?", "damage"),
)


def load_source_json(dataset: str) -> list[dict]:
    spec = config.DATASETS[dataset]
    path = config.SOURCE_ROOT / spec["directory"] / spec["json"]
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source_path(stored_path: str, dataset: str) -> Path:
    """Resolve stale release paths without modifying or symlinking the source tree."""
    direct = Path(stored_path)
    if direct.exists():
        return direct.resolve()

    spec = config.DATASETS[dataset]
    directory = spec["directory"]
    normalized = stored_path.replace("\\", "/")
    marker = f"/{directory}/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        candidate = config.SOURCE_ROOT / directory / relative
    elif normalized.startswith(f"{directory}/"):
        candidate = config.SOURCE_ROOT / normalized
    else:
        candidate = config.SOURCE_ROOT / directory / normalized
    if not candidate.exists():
        raise FileNotFoundError(f"Cannot resolve {dataset} source path: {stored_path}")
    return candidate.resolve()


def image_paths(record: dict, dataset: str) -> list[Path]:
    values = record["image"] if isinstance(record["image"], list) else [record["image"]]
    return [resolve_source_path(value, dataset) for value in values]


def mask_paths(record: dict, dataset: str) -> list[Path]:
    values = record.get("metadata", {}).get("mask")
    if not values:
        return []
    if not isinstance(values, list):
        values = [values]
    return [resolve_source_path(value, dataset) for value in values]


def assistant_answers(record: dict) -> list[str]:
    return [
        turn.get("value", "").strip()
        for turn in record.get("conversations", [])
        if turn.get("from") in {"gpt", "assistant"}
    ]


def descriptive_score(record: dict) -> tuple[int, int]:
    answers = assistant_answers(record)
    return len(answers), sum(len(answer) for answer in answers)


def relative_query_path(path: Path, dataset: str) -> Path:
    root = (config.SOURCE_ROOT / config.DATASETS[dataset]["directory"]).resolve()
    return path.resolve().relative_to(root)


def sample_key(record: dict, dataset: str) -> str:
    path = relative_query_path(image_paths(record, dataset)[0], dataset)
    parts = path.parts
    if dataset == "mvtec_ad":
        return path.as_posix()
    if dataset in {"anomaly_shapenet", "real3d"}:
        return Path(*parts[: parts.index("views")]).as_posix()
    if dataset == "mvtec3d":
        background_index = next(i for i, part in enumerate(parts) if part in {"black", "white"})
        return Path(*parts[:background_index], parts[background_index + 1]).as_posix()
    raise KeyError(dataset)


def deduplicated_records(dataset: str) -> list[tuple[str, dict, list[str]]]:
    """Choose the richest conversation for each physical image or 3D object."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in load_source_json(dataset):
        grouped[sample_key(record, dataset)].append(record)
    selected = []
    for key in sorted(grouped):
        records = grouped[key]
        best = max(records, key=descriptive_score)
        selected.append((key, best, [item.get("id", "") for item in records]))
    return selected


def object_category(key: str) -> str:
    return key.split("/", 1)[0]


def stable_choice(items: list, identity: str, seed: int):
    if not items:
        raise ValueError(f"No deterministic reference candidates for {identity}")
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    return items[int.from_bytes(digest[:8], "big") % len(items)]


def view_id(path: Path, target_modality: str) -> str:
    return "query" if target_modality == "2d" else path.stem


@lru_cache(maxsize=None)
def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def mask_bbox(mask_path: Path, *, allow_empty: bool = False) -> tuple[list[int] | None, tuple[int, int]]:
    """Return inclusive pixel xyxy around all non-zero mask pixels."""
    with Image.open(mask_path) as image:
        array = np.asarray(image.convert("L"))
    ys, xs = np.nonzero(array > 0)
    if not len(xs):
        if allow_empty:
            return None, (array.shape[1], array.shape[0])
        raise ValueError(f"Mask has no foreground: {mask_path}")
    height, width = array.shape
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], (width, height)


def normalize_bbox_1000(bbox: Iterable[int], size: tuple[int, int]) -> list[int]:
    width, height = size
    x1, y1, x2, y2 = bbox
    x_scale = 1000 / max(width - 1, 1)
    y_scale = 1000 / max(height - 1, 1)
    return [round(x1 * x_scale), round(y1 * y_scale), round(x2 * x_scale), round(y2 * y_scale)]


def anomaly_description(record: dict) -> str:
    conversations = record.get("conversations", [])
    candidates = []
    for index, turn in enumerate(conversations[:-1]):
        if turn.get("from") not in {"human", "user"}:
            continue
        question = turn.get("value", "").lower()
        for answer_turn in conversations[index + 1 :]:
            if answer_turn.get("from") in {"gpt", "assistant"}:
                answer = answer_turn.get("value", "").strip()
                if answer.lower() in {"yes", "no"}:
                    break
                question_relevant = any(
                    word in question
                    for word in ("anomal", "defect", "abnormal", "unusual", "irregular", "issue", "wrong")
                )
                question_speculative = any(
                    word in question
                    for word in ("cause", "impact", "affect", "prevent", "suggest", "future", "why")
                )
                pattern_hits = sum(bool(re.search(pattern, answer.lower())) for pattern, _ in TYPE_PATTERNS)
                score = pattern_hits * 10 + (4 if question_relevant else 0) - (8 if question_speculative else 0)
                score += 2 if any(word in answer.lower() for word in ("anomal", "defect")) else 0
                candidates.append((score, -index, answer))
                break
    if candidates:
        best = max(candidates)
        if best[0] > 0:
            return best[2]
    for answer in assistant_answers(record):
        if answer.lower() not in {"yes", "no"} and any(
            word in answer.lower() for word in ("anomal", "defect", "concav", "bulge", "spot")
        ):
            return answer
    raise ValueError(f"No anomaly description in source record {record.get('id')}")


def canonical_type(description: str) -> str:
    lowered = description.lower()
    for pattern, label in TYPE_PATTERNS:
        if re.search(pattern, lowered):
            return label
    return "surface anomaly"


def mvtec_type(record: dict, dataset: str) -> str:
    path = relative_query_path(image_paths(record, dataset)[0], dataset)
    parts = path.parts
    if "test" not in parts:
        raise ValueError(f"Anomalous MVTec path is not under test/: {path}")
    return parts[parts.index("test") + 1].replace("_", " ")


def existing_bbox_path(query_path: Path, dataset: str) -> Path:
    relative = relative_query_path(query_path, dataset)
    parts = list(relative.parts)
    if dataset == "mvtec_ad":
        test_index = parts.index("test")
        parts[test_index] = "ground_truth2"
        filename = Path(parts[-1])
        parts[-1] = f"{filename.stem}_bbox{filename.suffix}"
        candidate = config.SOURCE_ROOT / config.DATASETS[dataset]["directory"] / Path(*parts)
    elif dataset in {"anomaly_shapenet", "real3d"}:
        parts[parts.index("views")] = "bbox"
        candidate = config.SOURCE_ROOT / config.DATASETS[dataset]["directory"] / Path(*parts)
    elif dataset == "mvtec3d":
        candidate = config.DERIVED_BBOX_DIR / "mvtec3d" / relative
    else:
        raise KeyError(dataset)
    return candidate.resolve()


def matching_reference_views(reference_record: dict, dataset: str, query_paths: list[Path]) -> list[Path]:
    reference_selected = image_paths(reference_record, dataset)
    if config.DATASETS[dataset]["target_modality"] == "2d":
        return [reference_selected[0]]
    views_dir = reference_selected[0].parent
    matched = [views_dir / query_path.name for query_path in query_paths]
    if all(path.exists() for path in matched):
        return [path.resolve() for path in matched]
    return reference_selected


def safe_id(dataset: str, key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")
    return f"{dataset}__{slug}"


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
