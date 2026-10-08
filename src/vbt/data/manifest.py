"""The download manifest ``vbt data acquire`` writes: the upstream downloader's format, for any source.

``<downloads>/.download-manifest.json`` (the name is the descriptor's ``acquisition.manifest``) holds ``release``,
``base`` (the transport's base URL, the upstream downloader's ``BASE`` for Open Targets), ``expected_files``,
``complete`` and ``files`` (``{"<path>": {"bytes", "sha256", "url", <publisher algorithm>: hex}}``), so the upstream
doctor and downloader accept it and the data layer's readiness check (R2) finds bytes and sha256 for every file.
It also records ``tables`` (the download groups whose every file was verified), ``archive_files`` (the files the
source lists for every declared group), how the files were verified and the index they were verified against.

:func:`write_manifest` writes it for the groups whose files are all present and match the listing, without
downloading anything: a group missing a file, holding a file the listing does not name, or holding a changed file
is left out and reported. Entries of groups not looked at this time stay while their files keep their size.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..datalayer.plugins.acquisition import file_digests, matches, strongest
from ..datalayer.plugins.base import RemoteFile

__all__ = ["MANIFEST", "ManifestReport", "write_manifest", "load_manifest", "parquet_framed", "local_files",
           "group_patterns", "atomic_write"]

MANIFEST = ".download-manifest.json"
#: Names under a downloads directory that are never data files.
_SKIP_SUFFIXES = (".part", ".tmp", ".part.json")


def atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing to write through a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def parquet_framed(path: Path) -> bool:
    """``PAR1`` at both ends and a footer length that fits (the upstream downloader's check)."""
    size = path.stat().st_size
    if size < 12:
        return False
    with path.open("rb") as fh:
        if fh.read(4) != b"PAR1":
            return False
        fh.seek(-8, os.SEEK_END)
        footer = int.from_bytes(fh.read(4), "little")
        return 0 < footer <= size - 12 and fh.read(4) == b"PAR1"


def local_files(downloads: Path, *, skip_dirs: Iterable[str] = ()) -> list[str]:
    """Relative paths of the data files under ``downloads`` (no hidden files or directories, no ``.part`` or
    ``.tmp`` files, nothing under ``skip_dirs``)."""
    root = Path(downloads)
    if not root.is_dir():
        return []
    skip = {s.strip("/") for s in skip_dirs if s}
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")
                             and (f"{rel_dir}/{d}" if rel_dir else d) not in skip)
        for name in sorted(filenames):
            if name.startswith(".") or name.endswith(_SKIP_SUFFIXES):
                continue
            out.append(f"{rel_dir}/{name}" if rel_dir else name)
    return out


def group_patterns(spec: Any, name: str, release: str) -> tuple[list[str], list[str]]:
    """``(include, exclude)`` globs of download group ``name`` (a table or an ``extra`` entry), ``{release}``
    substituted."""
    from ..datalayer.plugins.acquisition import substitute

    entry = spec.tables.get(name) or spec.extra.get(name)
    if entry is None:
        return [], []
    v = {"release": release}
    return [substitute(p, v) for p in entry.files], [substitute(p, v) for p in entry.exclude]


@dataclass
class ManifestReport:
    path: Path
    tables: dict[str, int] = field(default_factory=dict)          # verified groups -> files
    skipped: dict[str, str] = field(default_factory=dict)         # group -> why it was left out
    files: int = 0
    bytes: int = 0
    seconds: float = 0.0
    verified_by: str = ""

    def summary(self) -> str:
        lines = [f"manifest {self.path}: {len(self.tables)} table(s), {self.files} file(s), "
                 f"{self.bytes / 1e9:.2f} GB verified against {self.verified_by or 'the listing'} in "
                 f"{self.seconds:.1f} s"]
        lines += [f"  left out {t}: {why}" for t, why in sorted(self.skipped.items())]
        return "\n".join(lines)


def _describe_skip(missing: Sequence[str], listed: int, extra: Sequence[str]) -> str:
    parts = []
    if missing:
        parts.append(f"{len(missing)} of {listed} file(s) missing")
    if extra:
        parts.append(f"{len(extra)} file(s) not in the release: {extra[0]}")
    return "; ".join(parts)


def write_manifest(spec: Any, downloads: Path, listing: Sequence[RemoteFile], *, release: str, base: str,
                   groups: Iterable[str] | None = None, known: Mapping[str, Mapping[str, Any]] | None = None,
                   about: Mapping[str, Any] | None = None, skip_dirs: Iterable[str] = (),
                   name: str | None = None, written_by: str = "vbt data acquire") -> ManifestReport:
    """Write the manifest for ``groups`` (default: every declared group with a file under ``downloads``) whose
    files are all present and match ``listing``. ``known`` (``{path: {bytes, sha256, <algo>}}``, e.g. just
    downloaded) spares hashing a file again while its size is unchanged."""
    t0 = time.monotonic()
    downloads = Path(downloads)
    mname = name or spec.manifest or MANIFEST
    path = downloads / mname
    report = ManifestReport(path=path)
    by_path = {f.path: f for f in listing}
    present_all = local_files(downloads, skip_dirs=skip_dirs)
    declared = [*spec.tables, *spec.extra]
    pats = {g: group_patterns(spec, g, release) for g in declared}
    if groups is None:
        names = sorted(g for g in declared if pats[g][0] and any(matches(p, *pats[g]) for p in present_all))
    else:
        names = sorted(groups)
    previous = load_manifest(path)
    old = previous.get("files") if previous.get("release") == release and previous.get("base") == base else None
    old = old if isinstance(old, dict) else {}
    algos_seen: set[str] = set()
    entries: dict[str, dict[str, Any]] = {}
    for g in names:
        include, exclude = pats.get(g, ([], []))
        if g not in pats:
            report.skipped[g] = "not a table of this release"
            continue
        if not include:
            report.skipped[g] = "written by a prepare step, not downloaded"
            continue
        listed = {p for p in by_path if matches(p, include, exclude)}
        present = {p for p in present_all if matches(p, include, exclude)}
        if not present:
            report.skipped[g] = "no files"
            continue
        missing, extra = sorted(listed - present), sorted(present - listed)
        if missing or extra or not listed:
            report.skipped[g] = _describe_skip(missing, len(listed), extra) or "the release lists no file for it"
            continue
        mine: dict[str, dict[str, Any]] = {}
        bad = ""
        for rel in sorted(listed):
            remote = by_path[rel]
            fpath = downloads / rel
            size = fpath.stat().st_size
            pub = strongest(remote.checksums)
            need = {"sha256"} | ({pub[0]} if pub else set())
            cached = dict((known or {}).get(rel) or {})
            if cached.get("bytes") == size and all(cached.get(a) for a in need):
                digests = {a: str(cached[a]) for a in need}
            else:
                size, digests = file_digests(fpath, need)
            if remote.size is not None and size != remote.size:
                bad = f"{rel}: {size} bytes, the listing says {remote.size}"
                break
            if pub and digests[pub[0]] != pub[1]:
                bad = f"{rel}: {pub[0]} {digests[pub[0]]} differs from the listing ({pub[1]})"
                break
            if pub:
                algos_seen.add(pub[0])
            entry = {"bytes": size, "sha256": digests["sha256"], "url": remote.url}
            if pub and pub[0] != "sha256":
                entry[pub[0]] = digests[pub[0]]
            mine[rel] = entry
        if bad:
            report.skipped[g] = bad
            continue
        entries.update(mine)
        report.tables[g] = len(mine)
        report.bytes += sum(e["bytes"] for e in mine.values())
    # entries of groups not looked at this time stay while their files keep their size
    for rel, entry in old.items():
        owner = next((g for g in declared if pats[g][0] and matches(rel, *pats[g])), None)
        if owner is None or owner in report.tables or owner in report.skipped or rel in entries \
                or rel not in by_path:
            continue
        fpath = downloads / rel
        if isinstance(entry, dict) and fpath.is_file() and fpath.stat().st_size == entry.get("bytes"):
            entries[rel] = dict(entry)
            report.tables[owner] = report.tables.get(owner, 0) + 1
    covered = sorted({g for g in declared for rel in entries if pats[g][0] and matches(rel, *pats[g])})
    algo = ", ".join(sorted(algos_seen)) or "size"
    index = dict(about or {})
    against = str(index.get("url") or "the listing").rstrip("/").rsplit("/", 1)[-1]
    report.verified_by = against
    listed_all = sum(1 for p in by_path if any(matches(p, *pats[g]) for g in declared if pats[g][0]))
    data = {"release": release, "base": base, "expected_files": len(entries), "complete": bool(entries),
            "files": dict(sorted(entries.items())), "tables": covered, "archive_files": listed_all,
            "verified": f"{algo} of every file against {against}", "integrity": index,
            "written_by": written_by, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write(path, (json.dumps(data, indent=2, default=str) + "\n").encode())
    report.files = len(entries)
    report.seconds = time.monotonic() - t0
    return report
