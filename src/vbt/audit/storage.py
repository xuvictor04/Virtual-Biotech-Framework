"""Crash-safe storage and the run-directory layout rules shared by the audit code.

Everything here is stdlib-only, so a run directory can be audited on any machine.

Atomic writes
    ``write_text_atomic``/``write_json_atomic`` write to a temporary file in the
    destination directory, ``fsync`` it, then ``os.replace`` it over the target.
    A crash leaves either the old or the new file, never a torn one, and a failed
    write removes its temporary file.

Locking
    ``run_lock(run_dir)`` serialises read-modify-write updates of a run's records
    across threads (an ``RLock`` per run) and processes (``fcntl.flock`` on
    ``<run>/.audit.lock``; ``msvcrt`` on Windows; a no-op where neither exists).

Layout
    Paths are always classified by their *run-relative* parts, never by the
    absolute path, so a runs root that happens to live under a directory named
    ``logs`` is audited like any other.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

# ----------------------------------------------------------------- layout

#: Top-level directories the harness owns. Their contents describe the run
#: rather than resulting from it; agents write analysis output under work/.
RESERVED_DIRS = frozenset({"inputs", "evidence", "logs", "report", "memory"})

#: Directories whose contents are hashed as harness records (``harness_files``).
HARNESS_HASH_DIRS = ("inputs", "evidence", "report")

#: Root-level harness files that are hashed as harness records.
HARNESS_ROOT_HASHED = ("session_report.json", "README.md", "audit.html")

#: Root-level files the harness (or another harness component) writes. Never
#: analysis artifacts, never misplaced.
HARNESS_ROOT_FILES = frozenset({
    "MANIFEST.json", "session_report.json", "README.md", "audit.html",
    "replay_diff.json", "run.sh", "environment.yml", "environment_full.yml",
})

#: Files the harness writes inside the reserved directories. Anything else found
#: there was put there by something other than the harness and is reported as
#: misplaced rather than silently dropped.
HARNESS_FILES = set({
    "inputs/query.txt", "inputs/plan.json", "inputs/config.json", "inputs/turns.json",
    "inputs/environment.txt", "inputs/environment.yml", "inputs/environment_full.yml",
    "evidence/claims.json", "evidence/artifacts.json", "evidence/provenance.json",
    "report/FINAL_REPORT.md", "report/FINAL_REPORT.rendered.md",
    "report/plan_reconciliation.json", "report/plan_reconciliation.md",
    "report/chief_of_staff_brief.md", "report/scenario_score.json",
})

#: Directory names that never hold analysis output, at any depth.
IGNORED_DIR_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints", ".downloads", ".git", ".cache"})

#: Temporary/partial file suffixes that are never artifacts.
IGNORED_SUFFIXES = (".tmp", ".part", ".pyc")

#: Directories under work/ that the harness (not an agent) owns.
HARNESS_WORK_DIRS = frozenset({"_mcp", "_tool_outputs"})

#: Extension -> artifact kind. Drives grouping in reports and list_artifacts.
_KIND_BY_SUFFIX = {
    ".png": "figure", ".jpg": "figure", ".jpeg": "figure", ".svg": "figure",
    ".pdf": "figure", ".gif": "figure", ".webp": "figure",
    ".csv": "table", ".tsv": "table", ".parquet": "table", ".xlsx": "table", ".xls": "table",
    ".py": "code", ".sh": "code", ".ipynb": "code", ".r": "code", ".rmd": "code", ".sql": "code",
    ".md": "report", ".txt": "report", ".html": "report",
    ".json": "data", ".jsonl": "data", ".yaml": "data", ".yml": "data",
    ".h5ad": "data", ".h5": "data", ".hdf5": "data", ".npz": "data", ".npy": "data",
    ".pkl": "data", ".rds": "data", ".feather": "data", ".gz": "data", ".zip": "data",
    ".log": "log",
}


def classify(path: str | Path) -> str:
    """Best-effort artifact kind from the file extension."""
    return _KIND_BY_SUFFIX.get(Path(str(path)).suffix.lower(), "other")


def rel_parts(rel: str) -> tuple[str, ...]:
    return PurePosixPath(rel).parts


def is_ignored_rel(rel: str) -> bool:
    """True for run-relative paths that are never recorded (dot-dirs, caches, temp files)."""
    parts = rel_parts(rel)
    if not parts:
        return True
    for part in parts:
        if part.startswith(".") or part in IGNORED_DIR_NAMES:
            return True
    return parts[-1].endswith(IGNORED_SUFFIXES)


def is_logs_rel(rel: str) -> bool:
    parts = rel_parts(rel)
    return bool(parts) and parts[0] == "logs"


def is_work_rel(rel: str) -> bool:
    parts = rel_parts(rel)
    return len(parts) >= 2 and parts[0] == "work"


def work_owner(rel: str) -> str | None:
    """The agent directory a work/ path belongs to (``work/<owner>/...``), else None."""
    parts = rel_parts(rel)
    if len(parts) >= 3 and parts[0] == "work":
        return parts[1]
    return None


def is_harness_rel(rel: str) -> bool:
    """A file the harness itself writes (never an artifact, never misplaced)."""
    parts = rel_parts(rel)
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in HARNESS_ROOT_FILES
    if parts[0] in ("logs", "memory"):
        return True
    return rel in HARNESS_FILES


def is_harness_hashed_rel(rel: str) -> bool:
    parts = rel_parts(rel)
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in HARNESS_ROOT_HASHED
    return parts[0] in HARNESS_HASH_DIRS


def to_rel(path: str | Path, run_dir: Path) -> str | None:
    """Run-relative POSIX path for ``path`` (lexically normalised), or None if outside."""
    p = Path(os.path.normpath(str(path)))
    roots = {Path(os.path.normpath(str(run_dir)))}
    try:
        roots.add(Path(run_dir).resolve())
    except OSError:
        pass
    for root in roots:
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        s = rel.as_posix()
        return "" if s == "." else s
    try:
        rp = p.resolve()
    except OSError:
        return None
    for root in roots:
        try:
            s = rp.relative_to(root).as_posix()
            return "" if s == "." else s
        except ValueError:
            continue
    return None


# ----------------------------------------------------------------- snapshots

Snapshot = dict  # {rel: (mtime_ns, size)}


def snapshot_dir(run_dir: Path, exclude_top: frozenset[str] | set[str] = frozenset({"logs"})
                 ) -> dict[str, tuple[int, int]]:
    """Cheap ``{rel: (mtime_ns, size)}`` snapshot of a run directory.

    Skips the top-level directories in ``exclude_top`` (``logs/`` by default; by
    run-relative first component only, never by an ancestor's name), dot-dirs,
    caches and temporary files, and never follows symlinked directories out of
    the run. Runs after every capturing tool call, so it never hashes.
    """
    out: dict[str, tuple[int, int]] = {}
    root = str(run_dir)

    def walk(dirpath: str, prefix: str) -> None:
        try:
            it = os.scandir(dirpath)
        except OSError:
            return
        with it:
            for entry in it:
                name = entry.name
                rel = f"{prefix}{name}"
                if name.startswith(".") or name in IGNORED_DIR_NAMES:
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if not prefix and name in exclude_top:
                            continue
                        walk(entry.path, rel + "/")
                    elif entry.is_file(follow_symlinks=True):
                        if name.endswith(IGNORED_SUFFIXES):
                            continue
                        st = entry.stat(follow_symlinks=True)
                        out[rel] = (st.st_mtime_ns, st.st_size)
                except OSError:
                    continue

    walk(root, "")
    return out


def diff_snapshots(before: dict, after: dict) -> tuple[list[str], list[str]]:
    """(changed_or_new, deleted) run-relative paths between two snapshots."""
    changed = sorted(k for k, v in after.items() if before.get(k) != v)
    deleted = sorted(k for k in before if k not in after)
    return changed, deleted


# ----------------------------------------------------------------- hashing

def sha256_file(path: str | Path, _chunk: int = 1 << 20) -> str:
    """Streaming SHA-256, so multi-GB .h5ad files never land in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ----------------------------------------------------------------- atomic IO

def _default_mode(path: Path) -> int:
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return 0o644


def write_bytes_atomic(path: str | Path, data: bytes) -> Path:
    """Atomically replace ``path`` with ``data`` (mkstemp in the same dir, fsync, os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _default_mode(path)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp, mode)
        except OSError:
            pass
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    _fsync_dir(path.parent)
    return path


def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_text_atomic(path: str | Path, text: str) -> Path:
    return write_bytes_atomic(path, text.encode("utf-8"))


def dumps_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def write_json_atomic(path: str | Path, data: Any, *, compact: bool = False) -> Path:
    """Atomic JSON write. ``compact`` uses the C encoder (no indentation) for large,
    frequently rewritten records such as the MANIFEST during a session."""
    # Serialise first: an unserialisable payload must not leave a temp file.
    text = json.dumps(data, default=str, ensure_ascii=False, separators=(",", ":")) if compact else dumps_json(data)
    return write_text_atomic(path, text)


def read_json(path: str | Path, default: Any = None) -> Any:
    """Tolerant JSON read: ``default`` when missing or malformed."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return default


def read_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Read a JSONL file, skipping malformed or partial lines. Returns (rows, n_bad)."""
    rows: list[dict[str, Any]] = []
    bad = 0
    try:
        handle = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return rows, 0
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


# ----------------------------------------------------------------- locking

_locks: dict[str, threading.RLock] = {}
_depth: dict[str, int] = {}
_guard = threading.Lock()


def _os_lock(handle, lock: bool) -> bool:
    try:
        if os.name == "nt":  # pragma: no cover - windows only
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX if lock else fcntl.LOCK_UN)
        return True
    except (ImportError, OSError):
        return False


@contextmanager
def run_lock(run_dir: str | Path) -> Iterator[None]:
    """Serialise audit-record updates for one run across threads and processes.

    Re-entrant within a thread: only the outermost holder takes the file lock
    (a second ``flock`` on a new descriptor would block on the first).
    """
    root = Path(run_dir)
    try:
        key = str(root.resolve())
    except OSError:
        key = str(root)
    with _guard:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
        if _depth.get(key, 0) > 0:
            _depth[key] += 1
            try:
                yield
            finally:
                _depth[key] -= 1
            return
        handle = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            handle = open(root / ".audit.lock", "a+b")
        except OSError:
            handle = None
        locked = _os_lock(handle, True) if handle is not None else False
        _depth[key] = 1
        try:
            yield
        finally:
            _depth[key] = 0
            if locked:
                _os_lock(handle, False)
            if handle is not None:
                handle.close()
