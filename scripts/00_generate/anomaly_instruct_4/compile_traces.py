"""Compile validated traces while removing every teacher-only auxiliary input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from dataset_json_store import read_records
from validate_trace import validate_reasoning


def compile_dataset(
    dataset: str, *, allow_missing: bool, min_iou: float, trace_source: str = "baseline"
) -> tuple[int, int]:
    manifest_path = config.MANIFEST_DIR / f"{dataset}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_first = trace_source == "evidence-first"
    records = read_records(config.EVIDENCE_FIRST_TRACE_DIR / f"traces_{dataset}.json", dataset) if evidence_first else {}
    compiled_dir = config.EVIDENCE_FIRST_COMPILED_DIR if evidence_first else config.COMPILED_DIR
    output, evaluation = [], []
    missing = 0
    for sample in manifest["samples"]:
        if evidence_first:
            trace = records.get(sample["id"])
        else:
            trace_path = config.TRACE_DIR / dataset / f"trace_{sample['id']}.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else None
        if trace is None:
            missing += 1
            continue
        validation = validate_reasoning(trace.get("reasoning", ""), sample, min_iou=min_iou)
        if not validation.valid:
            missing += 1
            continue
        query_paths = [item["path"] for item in sample["query_images"]]
        image_tokens = "".join("<image>" for _ in query_paths)
        output.append(
            {
                "id": sample["id"],
                "images": query_paths,
                "messages": [
                    {"role": "user", "content": f"{image_tokens}\n{sample['question']}"},
                    {"role": "assistant", "content": trace["reasoning"]},
                ],
            }
        )
        evaluation.append(
            {
                "id": sample["id"],
                "dataset": dataset,
                "label": sample["label"],
                "object_category": sample["object_category"],
                "query_images": sample["query_images"],
                "ground_truth": sample["supervision"],
                "generated_validation": validation.to_dict(),
            }
        )

    if missing and not allow_missing:
        raise RuntimeError(
            f"{dataset}: {missing} missing/invalid traces; regenerate them or pass --allow-missing"
        )
    compiled_dir.mkdir(parents=True, exist_ok=True)
    (compiled_dir / f"{dataset}_cot_sft.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (compiled_dir / f"{dataset}_evaluation_sidecar.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(output), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile query-only Anomaly-Instruct CoT data.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(config.DATASETS), default=sorted(config.DATASETS))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--trace-source", choices=["baseline", "evidence-first"], default="baseline")
    parser.add_argument("--min-location-iou", type=float, default=config.MIN_LOCATION_IOU)
    args = parser.parse_args()
    for dataset in args.datasets:
        count, missing = compile_dataset(
            dataset, allow_missing=args.allow_missing, min_iou=args.min_location_iou, trace_source=args.trace_source
        )
        print(f"{dataset}: compiled={count} missing_or_invalid={missing}")


if __name__ == "__main__":
    main()
