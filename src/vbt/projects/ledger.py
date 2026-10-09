"""Provenance of what the system creates in a project (docs/PROJECTS.md).

Every registration attempt is one line of ``provenance/ledger.jsonl`` (registered, pending review, refused,
approved, rejected), and every registered (or pending) item has a current record
``provenance/<kind>/<name>.json``::

    {"schema": "vbt.project.item/1", "kind": "utility", "name": "ic50_summary", "version": 2,
     "status": "registered" | "pending_review",
     "files": {"utilities/ic50_summary/utility.py": "sha256:...", ...},   # relative to the project
     "source_hash": "sha256:...",                                          # over the files, names included
     "who": {"agent", "run_id", "invocation_id", "tool_use_id"} | {"user": ...},
     "when": "2026-10-09T12:00:00Z", "why": "...",
     "validation": {"lint": [...], "check": {...}, "tests": {...}, "conformance": {...}},
     "review": {"mode", "verdict", "by", "notes", "at"},
     "previous": {"version", "source_hash", "when"} | null, ...}

:func:`verify` compares the files with the record: a registered utility whose files changed since registration
is not loaded as a tool (``vbt project check`` lists it).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model import Project, now_iso

__all__ = ["ITEM_SCHEMA", "ITEM_KINDS", "sha256_bytes", "sha256_file", "files_hash", "record_path", "read_record",
           "write_record", "append_ledger", "read_ledger", "records", "verify", "next_version"]

ITEM_SCHEMA = "vbt.project.item/1"
ITEM_KINDS = ("descriptor", "overlay", "plugin", "utility")
LEDGER = "ledger.jsonl"


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def files_hash(files: Mapping[str, str]) -> str:
    """One hash over ``{relative name: sha256}`` (names included, so a rename changes it)."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode("utf-8") + b"\0" + str(files[name]).encode("ascii") + b"\0")
    return "sha256:" + h.hexdigest()


def record_path(project: Project, kind: str, name: str) -> Path:
    if kind not in ITEM_KINDS:
        raise ValueError(f"unknown item kind {kind!r}")
    return project.provenance_dir / kind / f"{name}.json"


def read_record(project: Project, kind: str, name: str) -> dict[str, Any] | None:
    try:
        data = json.loads(record_path(project, kind, name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_record(project: Project, record: Mapping[str, Any]) -> Path:
    path = record_path(project, str(record["kind"]), str(record["name"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(record), indent=1, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def remove_record(project: Project, kind: str, name: str) -> None:
    record_path(project, kind, name).unlink(missing_ok=True)


def append_ledger(project: Project, event: Mapping[str, Any]) -> None:
    """Append one event (``at`` added) to ``provenance/ledger.jsonl`` under an exclusive lock."""
    path = project.provenance_dir / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"at": now_iso(), **dict(event)}, sort_keys=True, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_ledger(project: Project) -> list[dict[str, Any]]:
    path = project.provenance_dir / LEDGER
    out: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


def records(project: Project, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Every current record (of ``kinds``), sorted by kind and name."""
    out: list[dict[str, Any]] = []
    for kind in kinds or ITEM_KINDS:
        d = project.provenance_dir / kind
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("schema") == ITEM_SCHEMA:
                out.append(data)
    return out


def verify(project: Project, record: Mapping[str, Any]) -> list[str]:
    """Problems with a record's files: missing, or changed since the record was written."""
    problems = []
    for rel, digest in sorted((record.get("files") or {}).items()):
        p = project.root / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
        elif sha256_file(p) != digest:
            problems.append(f"{rel}: changed since it was registered")
    return problems


def next_version(project: Project, kind: str, name: str) -> tuple[int, dict[str, Any] | None]:
    """``(the next version number, the previous record's summary)``."""
    prev = read_record(project, kind, name)
    if not prev:
        return 1, None
    return int(prev.get("version") or 0) + 1, {"version": prev.get("version"), "source_hash": prev.get("source_hash"),
                                               "when": prev.get("when"), "status": prev.get("status")}
