"""Export a run: a zip bundle, a Markdown chat transcript and a file listing (fig. S1 downloads).

``export_run`` writes the whole run directory as one zip whose root is
``<run_id>/`` (MANIFEST, README.md, audit.html, inputs, every work/ artifact,
evidence, report and logs), finalising the reports first when they are missing.
The zip is written to a temporary file and moved into place, so a concurrent
reader never sees a partial archive, and cached under ``<runs>/.downloads/``
keyed on the newest member's mtime plus the member count and total size; an
unchanged run is not re-zipped.

``--no-data`` leaves out large raw data (``*.h5ad``, ``*.h5``, ``*.hdf5``,
``*.loom``, ``*.zarr`` always; ``*.parquet``/``*.feather`` over the size cap);
``--max-file-mb N`` leaves out any analysis file larger than N MB. Harness
records are always included. Excluded files are listed, with their recorded
sha256, in ``<run_id>/logs/export_excluded.json`` inside the archive, so the
recipient knows what is missing (``vbt verify`` on the extracted copy reports
them as missing).

Never included: dot-files and dot-directories (``.env``, ``.audit.lock``,
``.downloads``), caches, temporary files and symlinks that leave the run.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import storage as st
from .claims import read_claims_file
from .render import number_refs, render_footnotes_md

DOWNLOADS_DIR = ".downloads"
#: Inside the archive only. Under logs/, which is never hashed or reported as misplaced,
#: so an extracted copy (or a resumed one) stays consistent.
EXCLUDED_RECORD = "logs/export_excluded.json"

#: Raw single-cell / array data: always excluded by ``no_data``.
RAW_DATA_SUFFIXES = (".h5ad", ".h5", ".hdf5", ".loom", ".zarr", ".mtx", ".mtx.gz")
#: Tabular data: excluded by ``no_data`` only above the size cap.
LARGE_TABLE_SUFFIXES = (".parquet", ".feather", ".arrow")
#: Size cap used by ``no_data`` when ``max_file_mb`` is not given.
NO_DATA_DEFAULT_CAP_MB = 25.0

FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


class ExportError(RuntimeError):
    pass


def _iso(epoch: float | None = None) -> str:
    return datetime.fromtimestamp(epoch if epoch is not None else datetime.now().timestamp(),
                                  timezone.utc).isoformat(timespec="seconds")


def _skip_member(rel: str) -> bool:
    parts = st.rel_parts(rel)
    if not parts:
        return True
    for part in parts:
        if part.startswith(".") or part in st.IGNORED_DIR_NAMES:
            return True
    name = parts[-1]
    return name.endswith(st.IGNORED_SUFFIXES) or name == ".env" or name.startswith(".env")


def iter_members(run_dir: Path) -> Iterable[tuple[str, Path, os.stat_result]]:
    """(run-relative path, file, stat) for every file that belongs in an export."""
    root = run_dir.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        keep = []
        for d in dirnames:
            full = Path(dirpath) / d
            if d.startswith(".") or d in st.IGNORED_DIR_NAMES:
                continue
            if full.is_symlink():
                try:
                    target = full.resolve()
                except OSError:
                    continue
                if target != root and root not in target.parents:
                    continue
                # a symlinked directory inside the run would duplicate content; skip it
                continue
            keep.append(d)
        dirnames[:] = sorted(keep)
        for name in sorted(filenames):
            rel = rel_dir + name
            if _skip_member(rel):
                continue
            p = Path(dirpath) / name
            try:
                if p.is_symlink():
                    target = p.resolve(strict=True)
                    if target != root and root not in target.parents:
                        continue
                stt = p.stat()
            except OSError:
                continue
            if not os.path.isfile(p):
                continue
            yield rel, p, stt


def _is_harness(rel: str) -> bool:
    return st.is_harness_rel(rel) or st.is_harness_hashed_rel(rel)


def _exclusion_reason(rel: str, size: int, *, no_data: bool, cap_bytes: float | None) -> str | None:
    if _is_harness(rel):
        return None
    low = rel.lower()
    if no_data and low.endswith(RAW_DATA_SUFFIXES):
        return "raw data (--no-data)"
    if no_data and low.endswith(LARGE_TABLE_SUFFIXES) and size > (cap_bytes if cap_bytes is not None
                                                                  else NO_DATA_DEFAULT_CAP_MB * 1024 * 1024):
        return "large table (--no-data)"
    if cap_bytes is not None and size > cap_bytes:
        return f"larger than {cap_bytes / (1024 * 1024):g} MB (--max-file-mb)"
    return None


def _cache_name(run_dir: Path, no_data: bool, max_file_mb: float | None) -> str:
    suffix = ""
    if no_data:
        suffix += ".nodata"
    if max_file_mb is not None:
        suffix += f".max{max_file_mb:g}mb"
    return f"{run_dir.name}{suffix}.zip"


def _manifest_hashes(run_dir: Path) -> dict[str, str]:
    m = st.read_json(run_dir / "MANIFEST.json", {})
    arts = m.get("artifacts") if isinstance(m, dict) else None
    out: dict[str, str] = {}
    if isinstance(arts, Mapping):
        for rel, e in arts.items():
            if isinstance(e, str):
                out[str(rel)] = e
            elif isinstance(e, Mapping) and e.get("sha256"):
                out[str(rel)] = str(e["sha256"])
    return out


def export_run(run_dir: str | Path, out: str | Path | None = None, *, no_data: bool = False,
               max_file_mb: float | None = None, finalize: bool = True) -> Path:
    """Zip a run directory (archive root ``<run_id>/``). Returns the zip path.

    Without ``out`` the zip is cached in ``<runs>/.downloads/`` and reused while
    the run is unchanged. ``finalize`` renders README.md/audit.html first when
    either is missing.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise ExportError(f"no run directory at {run_dir}")
    if max_file_mb is not None and max_file_mb <= 0:
        raise ExportError("--max-file-mb must be positive")
    if finalize and (run_dir / "MANIFEST.json").is_file():
        from .report import ensure_reports
        try:
            ensure_reports(run_dir)
        except Exception:  # noqa: BLE001 - an export never fails because a report could not render
            pass

    cap = max_file_mb * 1024 * 1024 if max_file_mb is not None else None
    out_path = Path(out).expanduser().resolve() if out is not None else None
    members: list[tuple[str, Path, os.stat_result]] = []
    excluded: list[dict[str, Any]] = []
    for rel, p, stt in iter_members(run_dir):
        if out_path is not None and p.resolve() == out_path:
            continue
        why = _exclusion_reason(rel, stt.st_size, no_data=no_data, cap_bytes=cap)
        if why:
            excluded.append({"path": rel, "bytes": stt.st_size, "reason": why})
        else:
            members.append((rel, p, stt))

    newest = max((s.st_mtime_ns for _r, _p, s in members), default=0)
    key = {"newest_mtime_ns": newest, "n_files": len(members), "total_bytes": sum(s.st_size for _r, _p, s in members),
           "n_excluded": len(excluded), "no_data": bool(no_data), "max_file_mb": max_file_mb}

    cache = run_dir.parent / DOWNLOADS_DIR / _cache_name(run_dir, no_data, max_file_mb)
    key_path = cache.with_name(cache.name + ".key.json")
    cache_valid = cache.is_file() and st.read_json(key_path, None) == key
    if out_path is None:
        if cache_valid:
            return cache
        target = cache
    else:
        target = out_path
        if cache_valid:
            _copy_atomic(cache, target)
            return target

    target.parent.mkdir(parents=True, exist_ok=True)
    root = run_dir.name
    hashes = _manifest_hashes(run_dir) if excluded else {}
    fd, tmp = tempfile.mkstemp(prefix="." + target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for rel, p, _s in members:
                try:
                    zf.write(p, arcname=f"{root}/{rel}")
                except (OSError, ValueError):
                    excluded.append({"path": rel, "bytes": _s.st_size, "reason": "unreadable during export"})
            if excluded:
                for e in excluded:
                    if hashes.get(e["path"]):
                        e["sha256"] = hashes[e["path"]]
                zf.writestr(f"{root}/{EXCLUDED_RECORD}", json.dumps({
                    "exported": _iso(), "no_data": bool(no_data), "max_file_mb": max_file_mb,
                    "note": "These files are part of the run but were left out of this archive. Their recorded "
                            "hashes are in MANIFEST.json, so `vbt verify` on the extracted copy reports them as "
                            "missing.", "excluded": excluded}, indent=2))
            note = f"Virtual Biotech run {root}; {len(members)} files"
            if excluded:
                note += f"; {len(excluded)} excluded (see {EXCLUDED_RECORD})"
            zf.comment = note.encode()[:60000]
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if out_path is None:
        st.write_json_atomic(key_path, key)
    return target


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + dst.name + ".", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def excluded_members(zip_path: str | Path) -> list[dict[str, Any]]:
    """The exclusion list recorded inside an export (empty when nothing was left out)."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/" + EXCLUDED_RECORD):
                return list(json.loads(zf.read(name)).get("excluded") or [])
    return []


# ----------------------------------------------------------------- chat export

def export_chat_markdown(run_dir: str | Path) -> str:
    """A self-contained Markdown transcript of the conversation with numbered claim references."""
    run_dir = Path(run_dir)
    m = st.read_json(run_dir / "MANIFEST.json", {}) or {}
    sr = st.read_json(run_dir / "session_report.json", {}) or {}
    turns = [t for t in (sr.get("turns") or []) if isinstance(t, dict)] if isinstance(sr, dict) else []
    claims, _err = read_claims_file(run_dir / "evidence" / "claims.json")
    run_id = str((m.get("run_id") if isinstance(m, dict) else None) or run_dir.name)
    lines = ["# The Virtual Biotech — conversation export", "",
             f"- **Run:** `{run_id}`",
             f"- **Started:** {m.get('created') or '—'}" if isinstance(m, dict) else "- **Started:** —",
             f"- **Status:** {m.get('status') or 'unknown'}" if isinstance(m, dict) else "- **Status:** unknown",
             f"- **Exported:** {_iso()}"]
    cost = (m.get("cost_usd") if isinstance(m, dict) else None)
    if cost is not None:
        try:
            lines.append(f"- **Cost:** ${float(cost):.2f}")
        except (TypeError, ValueError):
            pass
    lines += ["", "Claim references are numbered per answer; `†` marks a claim without verified evidence and "
                  "`[C7?]` a reference to a claim that was never filed. The full evidence record is in the run's "
                  "audit.html.", "", "---", ""]
    if not turns:
        lines += ["*No turns were recorded for this run.*", ""]
    for t in turns:
        n = t.get("turn")
        status = str(t.get("status") or "completed")
        lines += [f"## Turn {n} — User", "", str(t.get("prompt") or "").rstrip(), ""]
        body, footnotes = number_refs(str(t.get("response") or ""), claims)
        head = f"## Turn {n} — CSO"
        if status != "completed":
            head += f" ({status})"
        lines += [head, "", body.rstrip() or "*(no response recorded)*", ""]
        if footnotes:
            lines += ["**References**", "", render_footnotes_md(footnotes), ""]
        meta = []
        if t.get("agents"):
            meta.append("specialists: " + ", ".join(str(a) for a in t["agents"]))
        try:
            if t.get("cost_usd") is not None:
                meta.append(f"cost ${float(t['cost_usd']):.2f}")
        except (TypeError, ValueError):
            pass
        if t.get("duration_s") is not None:
            meta.append(f"{t['duration_s']}s")
        if meta:
            lines += ["*" + " · ".join(meta) + "*", ""]
        lines += ["---", ""]
    return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------- file listing

_GROUPS = (("figures", "figure"), ("tables", "table"), ("code", "code"), ("reports", "report"), ("data", "data"))


def list_session_files(run_dir: str | Path) -> dict[str, Any]:
    """Figures, tables, code (and reports, data, other) under work/, grouped by agent.

    Built from MANIFEST.json artifacts, plus any work/ file the MANIFEST has not
    recorded yet. Returns ``{group: {agent: [row]}, "counts": {group: n}}`` with
    rows ``{path, name, kind, bytes, produced_by, description, registered,
    cited_by, mtime, image}``.
    """
    run_dir = Path(run_dir)
    m = st.read_json(run_dir / "MANIFEST.json", {}) or {}
    arts = m.get("artifacts") if isinstance(m, dict) and isinstance(m.get("artifacts"), Mapping) else {}
    snap = st.snapshot_dir(run_dir)
    rows: dict[str, dict[str, Any]] = {}
    for rel, e in (arts or {}).items():
        rel = str(rel)
        if rel not in snap:
            continue
        e = e if isinstance(e, Mapping) else {"sha256": e}
        rows[rel] = {"path": rel, "kind": e.get("kind") or st.classify(rel),
                     "produced_by": e.get("produced_by") or st.work_owner(rel) or "unknown",
                     "description": e.get("description") or "", "registered": bool(e.get("registered")),
                     "cited_by": list(e.get("cited_by") or []), "sha256": e.get("sha256")}
    for rel in snap:
        if st.is_work_rel(rel) and rel not in rows and not st.is_ignored_rel(rel):
            rows[rel] = {"path": rel, "kind": st.classify(rel), "produced_by": st.work_owner(rel) or "unknown",
                         "description": "", "registered": False, "cited_by": [], "sha256": None}
    out: dict[str, Any] = {g: {} for g, _k in _GROUPS}
    out["other"] = {}
    kind_to_group = {k: g for g, k in _GROUPS}
    for rel, r in sorted(rows.items(), key=lambda kv: -snap[kv[0]][0]):
        mtime_ns, size = snap[rel]
        r["bytes"] = size
        r["mtime"] = _iso(mtime_ns / 1e9)
        r["name"] = Path(rel).name
        r["image"] = rel.lower().endswith(IMAGE_SUFFIXES)
        group = kind_to_group.get(r["kind"], "other")
        if rel.lower().endswith(FIGURE_SUFFIXES):
            group = "figures"
        out[group].setdefault(str(r["produced_by"]), []).append(r)
    out["counts"] = {g: sum(len(v) for v in out[g].values()) for g in [*(g for g, _k in _GROUPS), "other"]}
    return out


def format_file_listing(listing: Mapping[str, Any]) -> str:
    """Plain-text rendering of ``list_session_files`` for ``vbt show RUN --files``."""
    lines: list[str] = []
    for group in ("figures", "tables", "code", "reports", "data", "other"):
        by_agent = listing.get(group) or {}
        n = sum(len(v) for v in by_agent.values())
        if not n:
            continue
        lines.append(f"{group.capitalize()} ({n})")
        for agent in sorted(by_agent):
            lines.append(f"  {agent}")
            for r in by_agent[agent]:
                extra = []
                if r.get("cited_by"):
                    extra.append("cited by " + ", ".join(r["cited_by"]))
                if r.get("description"):
                    extra.append(str(r["description"])[:60])
                size = r.get("bytes") or 0
                size_s = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
                lines.append(f"    {r['path']}  ({size_s})" + (f"  [{'; '.join(extra)}]" if extra else ""))
    return "\n".join(lines) if lines else "No files under work/."


__all__ = ["export_run", "export_chat_markdown", "list_session_files", "format_file_listing", "iter_members",
           "excluded_members", "ExportError", "DOWNLOADS_DIR", "EXCLUDED_RECORD", "RAW_DATA_SUFFIXES"]
