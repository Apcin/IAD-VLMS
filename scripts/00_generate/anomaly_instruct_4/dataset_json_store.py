"""Small, atomic dataset JSON collections shared by generation and compilation."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path


def read_records(path: Path, dataset: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("dataset") != dataset or not isinstance(value.get("records"), dict):
        raise ValueError(f"invalid dataset collection: {path}")
    records = value["records"]
    for image_id, record in records.items():
        if not isinstance(record, dict) or record.get("image_id") != image_id or record.get("dataset") != dataset:
            raise ValueError(f"invalid record identity in {path}: {image_id}")
    return records


def merge_records(path: Path, dataset: str, records: dict[str, dict]) -> None:
    """Merge a completed batch under a stable lock; readers see old or new JSON.

    Lock files must stay in place: locking the replaced JSON inode is unsafe.
    A malformed existing collection raises instead of silently losing prior work.
    """
    if not records:
        return
    for image_id, record in records.items():
        if record.get("image_id") != image_id or record.get("dataset") != dataset:
            raise ValueError(f"record identity mismatch: {image_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = read_records(path, dataset)
        merged.update(records)
        value = {"schema_version": "anomaly-instruct-dataset-records-v1", "dataset": dataset, "records": merged}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
