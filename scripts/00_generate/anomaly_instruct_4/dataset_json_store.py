"""Append dataset JSON arrays; expose the latest record per ID to callers."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from pathlib import Path


FOOTER = b"]}\n"


def _header(dataset: str) -> bytes:
    metadata = {"schema_version": "anomaly-instruct-dataset-records-v2", "dataset": dataset}
    return (json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))[:-1] + ',"records":[\n').encode("utf-8")


def _check_record(record, dataset: str, image_id=None) -> str:
    if (not isinstance(record, dict) or not isinstance(record.get("image_id"), str)
            or record.get("dataset") != dataset
            or (image_id is not None and record["image_id"] != image_id)):
        raise ValueError(f"invalid record identity for dataset {dataset}")
    return record["image_id"]


def _read_unlocked(path: Path, dataset: str) -> dict[str, dict]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("dataset") != dataset:
        raise ValueError(f"invalid dataset collection: {path}")
    entries = value.get("records")
    if not isinstance(entries, (dict, list)):
        raise ValueError(f"invalid records in {path}")
    records = {}
    iterator = entries.items() if isinstance(entries, dict) else ((None, entry) for entry in entries)
    for key, record in iterator:
        records[_check_record(record, dataset, key)] = record
    return records


def _sync(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _encode_records(records: dict[str, dict], dataset: str) -> bytes:
    encoded = []
    for image_id, record in records.items():
        _check_record(record, dataset, image_id)
        encoded.append(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    # Each record occupies one physical line. Newlines inside strings are escaped.
    return b"\n,".join(encoded) + b"\n" if encoded else b""


def _replace_snapshot(path: Path, dataset: str, records: dict[str, dict]) -> None:
    """Atomic initialization / one-time conversion of a legacy collection."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_header(dataset) + _encode_records(records, dataset) + FOOTER)
            _sync(handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _recover_tail(handle, dataset: str) -> None:
    """Under LOCK_EX, retain valid lines and repair only an unfinished EOF tail.

    A complete malformed line or an identity mismatch raises before any write.
    This handles interrupted appends, not arbitrary corruption or power loss.
    """
    handle.seek(0)
    if handle.readline() != _header(dataset):
        raise ValueError(f"cannot recover a non-v2 collection: {handle.name}")
    last_good, count, needs_newline = handle.tell(), 0, False
    while True:
        line = handle.readline()
        if not line:
            break
        if line == FOOTER:
            if handle.read(1):
                raise ValueError(f"unexpected data after JSON footer: {handle.name}")
            return
        if FOOTER.startswith(line) and not line.endswith(b"\n"):
            break
        prefix = b",{" if count else b"{"
        candidate = line[1:] if count and line.startswith(b",") else line
        try:
            if not line.startswith(prefix):
                raise ValueError("invalid record separator")
            record = json.loads(candidate)
        except (ValueError, UnicodeDecodeError):
            plausible = line.startswith(prefix) or prefix.startswith(line)
            if line.endswith(b"\n") or not plausible:
                raise ValueError(f"malformed complete record in {handle.name}")
            # A non-newline-terminated readline consumed EOF. Only this partial
            # final record may be discarded; earlier invalid lines are not skipped.
            break
        _check_record(record, dataset)
        count += 1
        last_good = handle.tell()
        needs_newline = not line.endswith(b"\n")
    handle.seek(last_good)
    handle.write((b"\n" if needs_newline else b"") + FOOTER)
    handle.truncate()
    _sync(handle)
    logging.getLogger(__name__).warning("Recovered interrupted JSON tail: %s", handle.name)


def read_records(path: Path, dataset: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.with_suffix(path.suffix + ".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            return _read_unlocked(path, dataset)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Release before upgrading; another writer may repair the tail first.
            fcntl.flock(lock, fcntl.LOCK_UN)
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return _read_unlocked(path, dataset)
            except (json.JSONDecodeError, UnicodeDecodeError):
                with path.open("r+b") as handle:
                    _recover_tail(handle, dataset)
                return _read_unlocked(path, dataset)


def merge_records(path: Path, dataset: str, records: dict[str, dict]) -> None:
    """Append a batch; the last record for each image_id wins when reading.

    Normal appends only inspect the header/footer and write the new batch. Stable
    lock files coordinate readers, writers, recovery, and legacy conversion.
    """
    if not records:
        return
    payload = _encode_records(records, dataset)  # Serialize outside the lock.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            _replace_snapshot(path, dataset, records)
            return
        with path.open("r+b") as handle:
            if handle.readline() != _header(dataset):
                previous = _read_unlocked(path, dataset)
                previous.update(records)
                _replace_snapshot(path, dataset, previous)
                return
            handle.seek(-len(FOOTER), os.SEEK_END)
            if handle.read() != FOOTER:
                _recover_tail(handle, dataset)
            handle.seek(-len(FOOTER), os.SEEK_END)
            has_records = handle.tell() > len(_header(dataset))
            handle.write((b"," if has_records else b"") + payload + FOOTER)
            handle.truncate()
            _sync(handle)
