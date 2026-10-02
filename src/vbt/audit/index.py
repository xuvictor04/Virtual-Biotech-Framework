"""Run discovery: the runs index (``INDEX.md``/``INDEX.json``) and run resolution.

``scan_runs`` summarises every run under a runs directory from its small records
only (``MANIFEST.json``, the ``stats`` block of ``evidence/claims.json`` and
``session_report.json``), never from artifacts, so it stays fast when runs hold
multi-GB data. ``update_index`` writes ``INDEX.md`` and ``INDEX.json`` beside the
runs; ``vbt.session.Run.close`` calls it, and so does ``vbt index``.

``resolve_run`` turns what a user types (a path, a run id, a unique id prefix,
the hex suffix of an id, or ``latest``) into a run directory, so every command
that takes a run accepts the same forms.

Dot-directories (``.downloads`` export cache, ``.audits``) are never runs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .storage import read_json, write_json_atomic, write_text_atomic

INDEX_MD = "INDEX.md"
INDEX_JSON = "INDEX.json"


class RunNotFound(LookupError):
    """No run matches the given path, id or prefix."""

    def __init__(self, arg: str, runs_dir: str | Path | None = None, recent: Iterable[str] = ()):
        self.arg = arg
        self.runs_dir = Path(runs_dir) if runs_dir else None
        self.recent = list(recent)
        msg = f"no run matches {arg!r}"
        if self.runs_dir is not None:
            msg += f" in {self.runs_dir}"
        if self.recent:
            msg += "; recent runs: " + ", ".join(self.recent[:5])
        super().__init__(msg)


class AmbiguousRunError(LookupError):
    """More than one run matches a prefix."""

    def __init__(self, arg: str, candidates: Iterable[str]):
        self.arg = arg
        self.candidates = sorted(candidates)
        shown = ", ".join(self.candidates[:10]) + (" ..." if len(self.candidates) > 10 else "")
        super().__init__(f"{arg!r} matches {len(self.candidates)} runs: {shown}; give more of the run id")


# ----------------------------------------------------------------- scanning

def _is_candidate_dir(d: Path) -> bool:
    if d.name.startswith(".") or not d.is_dir():
        return False
    return (d / "MANIFEST.json").is_file() or (d / "logs" / "trace.jsonl").is_file()


def _run_dirs(runs_dir: Path) -> list[Path]:
    try:
        entries = list(os.scandir(runs_dir))
    except OSError:
        return []
    out = []
    for e in entries:
        if e.name.startswith("."):
            continue
        try:
            if not e.is_dir():
                continue
        except OSError:
            continue
        d = Path(e.path)
        if _is_candidate_dir(d):
            out.append(d)
    return out


def _num(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _mtime_iso(p: Path) -> str | None:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return None


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """One index row for a run directory (tolerant of missing or malformed records)."""
    d = Path(run_dir)
    m = read_json(d / "MANIFEST.json", None)
    has_manifest = isinstance(m, dict)
    m = m if has_manifest else {}
    sr = read_json(d / "session_report.json", None)
    sr = sr if isinstance(sr, dict) else {}
    turns = [t for t in (sr.get("turns") or []) if isinstance(t, dict)]

    arts = m.get("artifacts") if isinstance(m.get("artifacts"), dict) else {}
    total_bytes = 0
    for e in arts.values():
        if isinstance(e, dict):
            try:
                total_bytes += int(e.get("bytes") or 0)
            except (TypeError, ValueError):
                pass

    n_claims, n_verified, n_unresolved = 0, 0, 0
    claims_doc = read_json(d / "evidence" / "claims.json", None)
    if isinstance(claims_doc, dict):
        stats = claims_doc.get("stats") if isinstance(claims_doc.get("stats"), dict) else {}
        claims = claims_doc.get("claims") if isinstance(claims_doc.get("claims"), list) else []
        n_claims = int(stats.get("n_claims", len(claims)) or 0)
        n_verified = int(stats.get("n_verified_evidence", 0) or 0)
        n_unresolved = int(stats.get("n_unresolved_evidence", 0) or 0)
    elif isinstance(claims_doc, list):
        n_claims = len(claims_doc)

    cost = _num(m.get("cost_usd"))
    if cost is None:
        cost = _num((sr.get("cost") or {}).get("total_usd") if isinstance(sr.get("cost"), dict) else None)
    if cost is None:
        cost = _num((m.get("config") or {}).get("total_cost_usd") if isinstance(m.get("config"), dict) else None)

    query = str(m.get("query") or sr.get("query") or (turns[0].get("prompt") if turns else "") or "")
    if not query:
        try:
            q = (d / "inputs" / "query.txt").read_text(encoding="utf-8", errors="replace")
            lines = [ln for ln in q.splitlines() if ln.strip() and not ln.startswith("--- turn")]
            query = lines[0] if lines else ""
        except OSError:
            pass
    status = str(m.get("status") or ("unknown" if has_manifest else "unrecorded"))
    if has_manifest and m.get("schema", 1) == 1 and not m.get("status"):
        status = "completed" if arts else "unknown"
    agents = [str(a) for a in (m.get("agents") or []) if a and not str(a).startswith("_") and a != "unknown"]
    if not agents:
        for t in turns:
            for a in t.get("agents") or []:
                if a not in agents:
                    agents.append(str(a))
    n_turns = m.get("turns") if isinstance(m.get("turns"), int) else len(turns)
    created = m.get("created") or m.get("started") or _mtime_iso(d)
    return {
        "run_id": str(m.get("run_id") or d.name),
        "dir": d.name,
        "path": str(d),
        "query": query[:2000],
        "status": status,
        "created": created,
        "completed": m.get("completed"),
        "updated": m.get("updated"),
        "agents": agents,
        "n_turns": int(n_turns or 0),
        "n_artifacts": len(arts),
        "total_bytes": total_bytes,
        "n_claims": n_claims,
        "n_verified_evidence": n_verified,
        "n_unresolved_evidence": n_unresolved,
        "cost_usd": round(cost, 6) if cost is not None else None,
        "has_readme": (d / "README.md").is_file(),
        "has_audit": (d / "audit.html").is_file(),
        "has_manifest": has_manifest,
    }


def scan_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Summaries of every run under ``runs_dir``, newest first.

    Reads only MANIFEST.json, the claims stats and session_report.json. Ignores
    dot-directories (``.downloads``) and directories that are not runs.
    """
    root = Path(runs_dir)
    rows = []
    for d in _run_dirs(root):
        try:
            rows.append(summarize_run(d))
        except Exception:  # noqa: BLE001 - one unreadable run never hides the others
            rows.append({"run_id": d.name, "dir": d.name, "path": str(d), "query": "", "status": "unreadable",
                         "created": _mtime_iso(d), "completed": None, "updated": None, "agents": [],
                         "n_turns": 0, "n_artifacts": 0, "total_bytes": 0, "n_claims": 0,
                         "n_verified_evidence": 0, "n_unresolved_evidence": 0, "cost_usd": None,
                         "has_readme": False, "has_audit": False, "has_manifest": False})
    rows.sort(key=lambda r: (str(r.get("created") or ""), r["dir"]), reverse=True)
    return rows


# ----------------------------------------------------------------- rendering

def _cell(text: Any, n: int = 80) -> str:
    s = " ".join(str(text or "").split())
    if len(s) > n:
        s = s[: n - 1] + "…"
    return s.replace("|", "\\|").replace("<", "&lt;") or "—"


def _human_bytes(n: int | float | None) -> str:
    v = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TB"


def render_index_md(rows: list[dict[str, Any]]) -> str:
    """Markdown table of runs (links use the directory name, which is what exists on disk)."""
    lines = ["# Runs", ""]
    if not rows:
        lines += ["*No runs recorded yet.*", ""]
        return "\n".join(lines)
    total_cost = sum(r.get("cost_usd") or 0 for r in rows)
    lines.append(f"{len(rows)} runs · {sum(r.get('n_artifacts') or 0 for r in rows)} artifacts · "
                 f"{sum(r.get('n_claims') or 0 for r in rows)} claims · ${total_cost:.2f} total")
    lines += ["", "| Run | Status | Query | Started | Turns | Specialists | Artifacts | Claims | Cost | Audit |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dirname = r.get("dir") or Path(str(r.get("path"))).name
        run = f"[`{r['run_id']}`]({dirname}/README.md)" if r.get("has_readme") else f"`{r['run_id']}`"
        audit = f"[audit.html]({dirname}/audit.html)" if r.get("has_audit") else "—"
        started = str(r.get("created") or "")[:16].replace("T", " ") or "—"
        cost = f"${r['cost_usd']:.2f}" if r.get("cost_usd") is not None else "—"
        claims = str(r.get("n_claims") or 0)
        if r.get("n_unresolved_evidence"):
            claims += f" ({r['n_unresolved_evidence']} unresolved)"
        lines.append(f"| {run} | {_cell(r.get('status'), 20)} | {_cell(r.get('query'), 70)} | {started} "
                     f"| {r.get('n_turns') or 0} | {len(r.get('agents') or [])} "
                     f"| {r.get('n_artifacts') or 0} ({_human_bytes(r.get('total_bytes'))}) | {claims} | {cost} "
                     f"| {audit} |")
    lines += ["", f"*Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `vbt index`. "
                  "Built from each run's MANIFEST.json, claims stats and session_report.json.*", ""]
    return "\n".join(lines)


def update_index(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Rebuild ``INDEX.md`` and ``INDEX.json`` under ``runs_dir`` (atomic writes)."""
    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = scan_runs(root)
    write_json_atomic(root / INDEX_JSON, {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                          "n_runs": len(rows), "runs": rows})
    write_text_atomic(root / INDEX_MD, render_index_md(rows))
    return rows


def load_index(runs_dir: str | Path, *, rebuild: bool = True) -> list[dict[str, Any]]:
    """The run rows: a fresh scan when ``rebuild`` (default), else the cached INDEX.json."""
    root = Path(runs_dir)
    if not rebuild:
        doc = read_json(root / INDEX_JSON, None)
        if isinstance(doc, dict) and isinstance(doc.get("runs"), list):
            return doc["runs"]
        if isinstance(doc, list):
            return doc
    return scan_runs(root)


# ----------------------------------------------------------------- resolution

def _newest(runs: list[Path]) -> Path | None:
    best, key = None, None
    for d in runs:
        m = read_json(d / "MANIFEST.json", {})
        k = (str((m or {}).get("created") or _mtime_iso(d) or ""), d.name)
        if key is None or k > key:
            best, key = d, k
    return best


def resolve_run(arg: str | Path, runs_dir: str | Path) -> Path:
    """Resolve a run argument to its directory.

    Accepted forms, in order: an existing directory path; a run id (directory
    name under ``runs_dir``); a unique prefix of a run id; a unique prefix of the
    hex suffix of a run id (``a1b2`` for ``20260101_120000_a1b2c3d4``);
    ``latest`` for the newest run. Raises :class:`AmbiguousRunError` (with the
    candidates) when a prefix matches several runs and :class:`RunNotFound`
    otherwise.
    """
    raw = str(arg).strip()
    if not raw:
        raise RunNotFound(raw, runs_dir)
    root = Path(runs_dir).expanduser()
    p = Path(raw).expanduser()
    if p.is_dir() and (p.is_absolute() or os.sep in raw or raw.startswith(".") or not (root / raw).is_dir()):
        return p.resolve()
    if (root / raw).is_dir() and not raw.startswith(".") and "/" not in raw and os.sep not in raw:
        return (root / raw).resolve()
    runs = _run_dirs(root)
    if raw in ("latest", "last"):
        newest = _newest(runs)
        if newest is None:
            raise RunNotFound(raw, root)
        return newest.resolve()
    if "/" in raw or os.sep in raw:
        raise RunNotFound(raw, root, [d.name for d in sorted(runs, reverse=True)])
    hits = [d for d in runs if d.name.startswith(raw)]
    if not hits:
        hits = [d for d in runs if "_" in d.name and d.name.rsplit("_", 1)[1].startswith(raw)]
    if not hits:
        # a run whose directory was renamed: match the run_id recorded in its MANIFEST
        for d in runs:
            m = read_json(d / "MANIFEST.json", {})
            rid = str((m or {}).get("run_id") or "") if isinstance(m, dict) else ""
            if rid and rid.startswith(raw):
                hits.append(d)
    if len(hits) == 1:
        return hits[0].resolve()
    if len(hits) > 1:
        raise AmbiguousRunError(raw, [d.name for d in hits])
    raise RunNotFound(raw, root, [d.name for d in sorted(runs, reverse=True)])


def runs_dir_from_config(config: dict[str, Any]) -> Path:
    """``config.paths.runs_dir`` resolved like every other configured path."""
    from ..config import resolve_path
    return resolve_path(((config or {}).get("paths") or {}).get("runs_dir") or "runs")


__all__ = ["AmbiguousRunError", "RunNotFound", "scan_runs", "summarize_run", "render_index_md",
           "update_index", "load_index", "resolve_run", "runs_dir_from_config", "INDEX_MD", "INDEX_JSON"]


if __name__ == "__main__":  # pragma: no cover
    import sys
    print(json.dumps(scan_runs(sys.argv[1] if len(sys.argv) > 1 else "runs"), indent=2))
