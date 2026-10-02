"""Run reports: ``README.md`` (a map of the run) and ``audit.html`` (one self-contained file).

Both are rendered from one ``collect(run_dir)`` dictionary built **only from the
on-disk records**: MANIFEST.json (schema 2 or 1), logs/trace.jsonl,
evidence/{artifacts,claims,provenance}.json, inputs/{plan,config}.json,
report/plan_reconciliation.json, session_report.json and logs/cost_report.json.
That keeps them provider-neutral, makes README and HTML agree by construction,
and lets ``vbt audit`` render them for a run that crashed before it saved
anything (every read is tolerant; a partial trace line is skipped and counted).

They answer, in order: is this record complete (the problems box comes from
``vbt.verify.verify_run``), how did the analysis flow (plan vs actual, dispatch
order, delegation prompts, timeline), what did each agent produce (artifacts with
hash, origin and the claims citing them), what supports each claim (evidence
with verified / external / unresolved status), and every tool call (errors
highlighted; paginated, never silently truncated).

``audit.html`` has no external assets: inline CSS, inline SVG and one small
inline script (pagination) allowed by a hash-pinned Content-Security-Policy, so
it opens from an email attachment and model text cannot run script. All model
and tool text is HTML-escaped.

``write_reports(run_or_dir)`` writes both atomically. ``vbt.session.Run`` calls
it on every save (after writing the MANIFEST and before hashing the reports);
for a bare directory it also refreshes the reports' hashes in the MANIFEST.

The problems box re-uses ``verify_run``. By default (``audit.report_verify:
fast``) artifact hashes whose recorded (mtime, size) still match the file are
trusted rather than re-read, so a per-turn save never re-hashes multi-GB data;
``vbt verify`` always re-hashes everything. ``full`` re-hashes; ``off`` skips it.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .claims import claim_stats, read_claims_file
from .plan import reconcile, render_plan_md
from .provenance import SUPPORT_AGENTS, build_index, read_trace
from .render import CLAIM_RE, EVIDENCE_STATUS_LABELS, evidence_status, number_refs
from .storage import (
    classify,
    is_ignored_rel,
    is_work_rel,
    read_json,
    run_lock,
    sha256_file,
    snapshot_dir,
    work_owner,
    write_json_atomic,
    write_text_atomic,
)

README_NAME = "README.md"
AUDIT_NAME = "audit.html"
REPORT_FILES = (README_NAME, AUDIT_NAME)

#: In-code default for ``audit.report_verify`` (fast | full | off).
DEFAULT_REPORT_VERIFY = "fast"

KIND_ORDER = ["report", "figure", "table", "data", "code", "log", "other"]
KIND_LABEL = {"report": "Reports", "figure": "Figures", "table": "Tables", "data": "Data", "code": "Code",
              "log": "Logs", "other": "Other"}

#: Categorical slots (validated reference palette), assigned to agents in order of
#: first dispatch and never cycled; later agents fall back to muted ink. Identity is
#: always carried by a direct label, colour only reinforces it.
_SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
_MUTED = ("#898781", "#898781")

TOOL_PAGE_SIZE = 100
MAX_TOOL_ROWS = 20000

_VERIFY_LOCK = threading.Lock()


# ----------------------------------------------------------------- small helpers

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def human_bytes(n: Any) -> str:
    v = _f(n) or 0.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TB"


def human_duration(seconds: Any) -> str:
    s = _f(seconds)
    if s is None:
        return "—"
    if s < 60:
        return f"{s:.0f}s" if s >= 10 else f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _usd(v: Any) -> str:
    x = _f(v)
    if x is None:
        return "—"
    return f"${x:.3f}" if 0 < x < 1 else f"${x:.2f}"


def _short_time(ts: Any) -> str:
    d = _parse_ts(ts)
    return d.strftime("%H:%M:%S") if d else ""


def agent_label(agent: Any) -> str:
    a = str(agent or "unknown")
    return {"cso": "CSO (orchestrator)", "_cso": "CSO (orchestrator)", "_mcp": "MCP tool outputs",
            "_tool_outputs": "Spilled tool outputs", "unknown": "Unattributed"}.get(a, a)


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _md_cell(text: Any, n: int = 160) -> str:
    """One markdown table cell: single line, pipes escaped, no raw HTML."""
    s = " ".join(str(text if text is not None else "").split())
    if len(s) > n:
        s = s[: n - 1] + "…"
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;") or "—"


def _md_inline(text: Any) -> str:
    return " ".join(str(text or "").split()).replace("<", "&lt;")


def _fence(text: str, lang: str = "") -> list[str]:
    """A fenced code block that cannot be closed early by the content."""
    runs = [len(m.group(0)) for m in re.finditer(r"`{3,}", text)]
    fence = "`" * max(3, (max(runs) + 1) if runs else 3)
    return [fence + lang, text.rstrip("\n"), fence]


def _clip(text: Any, n: int) -> tuple[str, bool]:
    s = str(text or "")
    return (s, False) if len(s) <= n else (s[:n], True)


def _safe_url(url: Any) -> str | None:
    u = str(url or "").strip()
    return u if re.match(r"^https?://[^\s<>\"']+$", u) else None


# ----------------------------------------------------------------- collect

def _normalize_artifacts(raw: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, Mapping):
        return out
    for rel, e in raw.items():
        rel = str(rel)
        if isinstance(e, str):  # schema 1: {rel: sha256}
            out[rel] = {"path": rel, "sha256": e}
        elif isinstance(e, Mapping):
            out[rel] = dict(e, path=rel)
    return out


def _scan_work_files(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Files under work/ (no hashing), for runs whose MANIFEST never recorded artifacts."""
    out = {}
    for rel, (mtime_ns, size) in snapshot_dir(run_dir).items():
        if is_work_rel(rel) and not is_ignored_rel(rel):
            out[rel] = {"path": rel, "bytes": size, "mtime_ns": mtime_ns, "sha256": None,
                        "unrecorded": True}
    return out


def _artifact_row(rel: str, e: Mapping[str, Any], registered: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reg = registered.get(rel) or {}
    owner = work_owner(rel)
    produced_by = e.get("produced_by") or reg.get("agent") or reg.get("produced_by") or owner or "unknown"
    try:
        size = int(e.get("bytes") if e.get("bytes") is not None else reg.get("bytes") or 0)
    except (TypeError, ValueError):
        size = 0
    return {
        "path": rel, "kind": e.get("kind") or reg.get("kind") or classify(rel), "bytes": size,
        "sha256": e.get("sha256") or reg.get("sha256"), "produced_by": str(produced_by),
        "created_by": e.get("created_by") or reg.get("created_by"),
        "tool_use_id": e.get("tool_use_id") or reg.get("tool_use_id"),
        "cited_by": [str(c) for c in (e.get("cited_by") or [])],
        "description": e.get("description") or reg.get("description") or "",
        "registered": bool(e.get("registered") or reg),
        "attribution": e.get("attribution"), "turn": e.get("turn"),
        "unrecorded": bool(e.get("unrecorded")),
    }


_TLS = threading.local()


def _install_hash_dispatch() -> bool:
    """Wrap ``vbt.verify._sha`` once with a dispatcher that trusts recorded hashes only
    inside ``_stat_trusting_hashes`` on the calling thread; everywhere else (and on
    every other thread) it re-hashes exactly as before."""
    from .. import verify as vmod

    with _VERIFY_LOCK:
        cur = getattr(vmod, "_sha", None)
        if cur is None:
            return False
        if getattr(cur, "_vbt_dispatch", False):
            return True

        def dispatch(p: Path, _orig=cur) -> str | None:
            known = getattr(_TLS, "known", None)
            if known:
                rec = known.get(str(p))
                if rec is not None:
                    try:
                        st = os.stat(p)
                    except OSError:
                        return None
                    if (st.st_mtime_ns, st.st_size) == rec[:2]:
                        return rec[2]
            return _orig(p)

        dispatch._vbt_dispatch = True  # type: ignore[attr-defined]
        vmod._sha = dispatch
        return True


@contextmanager
def _stat_trusting_hashes(run_dir: Path, artifacts: Mapping[str, Mapping[str, Any]]) -> Iterator[None]:
    """Make verify_run (on this thread) trust recorded hashes whose (mtime_ns, size) still match.

    Only the problems box of a report uses this; ``vbt verify`` always re-hashes.
    """
    if not _install_hash_dispatch():
        yield
        return
    known: dict[str, tuple[int, int, str]] = {}
    for rel, e in artifacts.items():
        try:
            if e.get("sha256") and e.get("mtime_ns") is not None and e.get("bytes") is not None:
                known[str(run_dir / rel)] = (int(e["mtime_ns"]), int(e["bytes"]), str(e["sha256"]))
        except (TypeError, ValueError):
            continue
    prev = getattr(_TLS, "known", None)
    _TLS.known = known
    try:
        yield
    finally:
        _TLS.known = prev


def _run_verify(run_dir: Path, artifacts: Mapping[str, Mapping[str, Any]], mode: Any) -> dict[str, Any] | None:
    mode = str(mode if mode not in (None, True) else DEFAULT_REPORT_VERIFY).lower()
    if mode in ("off", "false", "none", "0", "no"):
        return None
    try:
        from ..verify import verify_run
        if mode == "full":
            rep = verify_run(run_dir)
        else:
            with _stat_trusting_hashes(run_dir, artifacts):
                rep = verify_run(run_dir)
        rep["mode"] = "full" if mode == "full" else "fast"
        return rep
    except Exception as exc:  # noqa: BLE001 - a report must render even if verification breaks
        return {"status": "UNAVAILABLE", "problems": [], "mode": mode, "error": f"{type(exc).__name__}: {exc}",
                "integrity": {}, "evidence": {"warnings": []}}


def _agent_order(execution: list[dict[str, Any]], by_agent: Mapping[str, Any], manifest_agents: list[str]) -> list[str]:
    order: list[str] = []
    for e in execution:
        a = str(e.get("agent"))
        if a not in order:
            order.append(a)
    for a in manifest_agents:
        if a not in order and a in by_agent:
            order.append(a)
    for a in sorted(by_agent):
        if a not in order and not a.startswith("_") and a not in ("cso", "unknown"):
            order.append(a)
    for a in ("cso", "_cso", "_mcp", "_tool_outputs", "unknown"):
        if a in by_agent and a not in order:
            order.append(a)
    for a in by_agent:
        if a not in order:
            order.append(a)
    return order


def _agent_colors(agents: list[str]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    i = 0
    for a in agents:
        if a in out or a in ("cso", "_cso", "_mcp", "_tool_outputs", "unknown"):
            continue
        out[a] = (_SERIES_LIGHT[i], _SERIES_DARK[i]) if i < len(_SERIES_LIGHT) else _MUTED
        i += 1
    return out


def _turn_records(run_dir: Path, prov) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(recorded turns from session_report.json, turns seen only in the trace)."""
    sr = read_json(run_dir / "session_report.json", None)
    turns = [dict(t) for t in ((sr or {}).get("turns") or []) if isinstance(t, dict)] if isinstance(sr, dict) else []
    recorded = {t.get("turn") for t in turns}
    unrecorded = []
    for n, rec in sorted(prov.turns.items()):
        if n in recorded:
            continue
        unrecorded.append({"turn": n, "prompt": rec.get("prompt") or "", "status": rec.get("status") or "unfinished",
                           "start": rec.get("start"), "end": rec.get("end")})
    return turns, unrecorded


def _error_outputs(events: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ev in events:
        if ev.get("type") in ("tool_end", "tool_error") and (ev.get("is_error") or ev.get("type") == "tool_error"):
            tuid = ev.get("tool_use_id")
            if tuid:
                text = ev.get("output") if ev.get("output") is not None else ev.get("error")
                out[str(tuid)] = str(text or "")[:600]
    return out


def collect(run_dir: str | Path, *, verify: Any = DEFAULT_REPORT_VERIFY, notes: list[str] | None = None
            ) -> dict[str, Any]:
    """Everything README.md and audit.html show, from the run's on-disk records.

    Never raises for missing or malformed records; what could not be read is
    reported in ``data['read_errors']``.
    """
    run_dir = Path(run_dir)
    read_errors: list[str] = []
    raw_manifest = read_json(run_dir / "MANIFEST.json", None)
    if raw_manifest is None:
        read_errors.append("MANIFEST.json is missing or unreadable")
    manifest: dict[str, Any] = raw_manifest if isinstance(raw_manifest, dict) else {}
    schema = manifest.get("schema", 1 if manifest else None)

    # ---------------------------------------------------------- trace / provenance
    events, bad_lines = read_trace(run_dir / "logs" / "trace.jsonl")
    has_trace = (run_dir / "logs" / "trace.jsonl").is_file()
    prov = build_index(events, run_dir, malformed_lines=bad_lines)

    # ---------------------------------------------------------- artifacts
    arts = _normalize_artifacts(manifest.get("artifacts"))
    artifacts_recorded = isinstance(manifest.get("artifacts"), Mapping)
    if not artifacts_recorded:
        arts = _scan_work_files(run_dir)
    registered_rows = read_json(run_dir / "evidence" / "artifacts.json", [])
    registered = {str(r.get("path")): r for r in registered_rows if isinstance(r, Mapping) and r.get("path")} \
        if isinstance(registered_rows, list) else {}
    rows = [_artifact_row(rel, e, registered) for rel, e in sorted(arts.items())]
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_agent.setdefault(r["produced_by"], []).append(r)

    execution = prov.execution()
    manifest_agents = [str(a) for a in (manifest.get("agents") or [])]
    order = _agent_order(execution, by_agent, manifest_agents)
    specialists = [a for a in order if a not in SUPPORT_AGENTS and not a.startswith("_")]
    colors = _agent_colors([str(e.get("agent")) for e in execution] + order)

    # ---------------------------------------------------------- delegations
    calls_by_run: dict[str, int] = {}
    for c in prov.calls.values():
        rid = c.get("agent_run_id")
        if rid:
            calls_by_run[str(rid)] = calls_by_run.get(str(rid), 0) + 1
    delegations = []
    for e in execution:
        run = prov.runs.get(str(e.get("agent_run_id"))) or {}
        delegations.append({
            **e,
            "prompt": run.get("delegation_prompt") or "",
            "cost_usd": run.get("cost_usd"),
            "model_calls": run.get("model_calls"),
            "tool_calls": run.get("tool_calls") if run.get("tool_calls") is not None
            else calls_by_run.get(str(e.get("agent_run_id"))),
            "transcript_path": run.get("transcript_path"),
            "model": run.get("model"),
        })

    # ---------------------------------------------------------- tool calls
    err_out = _error_outputs(events)
    tool_rows = []
    for c in sorted(prov.calls.values(), key=lambda c: (c.get("start_t") is None, c.get("start_t") or 0,
                                                       str(c.get("started_at") or ""))):
        status = "error" if c.get("is_error") else ("running" if c.get("pending") else "ok")
        tool_rows.append({
            "tool_use_id": c.get("tool_use_id"), "tool": c.get("tool") or "?", "agent": c.get("agent") or "unknown",
            "turn": c.get("turn"), "started_at": c.get("started_at"), "duration_s": c.get("duration_s"),
            "status": status, "files_written": list(c.get("files_written") or []),
            "files_returned": list(c.get("files_returned") or []),
            "error": err_out.get(str(c.get("tool_use_id"))) if status == "error" else None,
            "exit_code": c.get("exit_code"),
        })

    # ---------------------------------------------------------- turns
    turns, unrecorded_turns = _turn_records(run_dir, prov)
    problem_turns = []
    for t in turns:
        s = str(t.get("status") or "completed")
        if s != "completed":
            problem_turns.append({"turn": t.get("turn"), "status": s, "prompt": t.get("prompt") or ""})
    for t in unrecorded_turns:
        problem_turns.append({"turn": t["turn"], "status": f"{t['status']} (no turn record)",
                              "prompt": t.get("prompt") or ""})
    for n in manifest.get("interrupted_turns") or []:
        if not any(p["turn"] == n for p in problem_turns):
            problem_turns.append({"turn": n, "status": "interrupted", "prompt": ""})

    # ---------------------------------------------------------- claims
    claims, claims_err = read_claims_file(run_dir / "evidence" / "claims.json")
    if claims_err:
        read_errors.append(f"evidence/claims.json: {claims_err}")
    cstats = claim_stats(claims)

    # ---------------------------------------------------------- plan
    plan_doc = read_json(run_dir / "inputs" / "plan.json", None)
    plan = plan_doc if isinstance(plan_doc, dict) and plan_doc.get("steps") else None
    plan_rec = read_json(run_dir / "report" / "plan_reconciliation.json", None)
    if plan is not None and not (isinstance(plan_rec, dict) and plan_rec.get("has_plan")):
        pt = plan.get("turn")
        execs = [e for e in execution if not e.get("orientation")
                 and (not isinstance(pt, int) or (e.get("turn") or 0) >= pt)]
        plan_rec = reconcile(plan, execs, arts)
    if not isinstance(plan_rec, dict):
        plan_rec = None

    # ---------------------------------------------------------- config, cost, time
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else None
    if not config:
        c2 = read_json(run_dir / "inputs" / "config.json", None)
        config = c2 if isinstance(c2, dict) else {}
    cost_report = read_json(run_dir / "logs" / "cost_report.json", None)
    cost_report = cost_report if isinstance(cost_report, dict) else {}
    cost = _f(manifest.get("cost_usd"))
    if cost is None:
        cost = _f(cost_report.get("total_usd"))
    created = manifest.get("created") or manifest.get("started")
    first_t = next((_f(ev.get("t")) for ev in events if _f(ev.get("t")) is not None), None)
    last_t = next((_f(ev.get("t")) for ev in reversed(events) if _f(ev.get("t")) is not None), None)
    if not created and first_t is not None:
        created = datetime.fromtimestamp(first_t, timezone.utc).isoformat(timespec="seconds")
    end = manifest.get("completed")
    d0, d1 = _parse_ts(created), _parse_ts(end)
    if d1 is None and last_t is not None:
        d1 = datetime.fromtimestamp(last_t, timezone.utc)
    duration_s = (d1 - d0).total_seconds() if d0 and d1 and d1 >= d0 else None

    status = str(manifest.get("status") or ("unknown" if manifest else "unrecorded"))
    if schema == 1 and not manifest.get("status"):
        status = "completed (schema 1 record)"

    # ---------------------------------------------------------- verification
    verification = _run_verify(run_dir, arts, verify) if manifest else None

    all_notes = list(notes or [])
    for n in manifest.get("audit_notes") or []:
        if isinstance(n, str) and n not in all_notes:
            all_notes.append(n)
    if not artifacts_recorded and arts:
        all_notes.append(f"MANIFEST.json has no artifact record; the {len(arts)} file(s) under work/ are listed "
                         "from disk without hashes. Run `vbt audit` to reconstruct the record.")
    if bad_lines:
        all_notes.append(f"logs/trace.jsonl has {bad_lines} malformed or partial line(s) (for example from a crash); "
                         "they were skipped.")
    if not has_trace:
        all_notes.append("No execution trace (logs/trace.jsonl) exists, so tool calls and the dispatch order "
                         "cannot be shown.")

    run_id = str(manifest.get("run_id") or run_dir.name)
    return {
        "run_id": run_id, "dir_name": run_dir.name, "run_dir": str(run_dir), "schema": schema,
        "query": str(manifest.get("query") or (turns[0].get("prompt") if turns else "")
                     or (unrecorded_turns[0]["prompt"] if unrecorded_turns else "") or ""),
        "status": status, "created": created, "completed": manifest.get("completed"),
        "updated": manifest.get("updated"), "duration_s": duration_s, "cost_usd": cost,
        "cost_report": cost_report, "config": config or {},
        "manifest": manifest, "reconstructed": manifest.get("reconstructed"),
        "turns": turns, "unrecorded_turns": unrecorded_turns, "problem_turns": problem_turns,
        "artifacts": rows, "artifacts_recorded": artifacts_recorded, "by_agent": by_agent,
        "agent_order": order, "specialists": specialists, "colors": colors,
        "n_artifacts": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
        "claims": claims, "claim_stats": cstats, "claims_error": claims_err,
        "plan": plan, "plan_reconciliation": plan_rec,
        "execution": execution, "delegations": delegations, "turn_windows": prov.turns,
        "tool_calls": tool_rows, "prov_summary": prov.summary(), "tool_counts": prov.tool_counts(),
        "has_trace": has_trace, "malformed_trace_lines": bad_lines, "trace_span": (first_t, last_t),
        "misplaced_files": [m if isinstance(m, dict) else {"path": str(m), "reason": "outside_work"}
                            for m in manifest.get("misplaced_files") or []],
        "deleted_artifacts": [d for d in manifest.get("deleted_artifacts") or [] if isinstance(d, dict)],
        "data_source_errors": list(manifest.get("data_source_errors") or []),
        "audit_errors": [str(e) for e in manifest.get("audit_errors") or []],
        "verify": verification, "notes": all_notes, "read_errors": read_errors,
        "generated": _now_iso(),
    }


# ----------------------------------------------------------------- shared pieces

def _problems(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(problems, warnings) for the problems box. ``unfinished_run`` is shown as status instead."""
    v = data.get("verify") or {}
    probs = [p for p in v.get("problems") or [] if isinstance(p, Mapping) and p.get("kind") != "unfinished_run"]
    warns = [w for w in (v.get("evidence") or {}).get("warnings") or [] if isinstance(w, Mapping)]
    return probs, warns


def _verify_status(data: Mapping[str, Any]) -> str:
    v = data.get("verify")
    if not v:
        return "not checked"
    return str(v.get("status") or "UNAVAILABLE")


def _reproduce_cmds(data: Mapping[str, Any]) -> list[tuple[str, str]]:
    rid = data.get("dir_name") or data["run_id"]
    return [
        (f"vbt verify {rid}", "re-hash every recorded file; check claims, report references and turn records"),
        (f"vbt verify {rid} --rerun", "also re-execute the agents' scripts in a scratch copy and compare outputs"),
        (f"vbt replay {rid}", "re-run the same turns with the pinned configuration into a new run"),
        (f"vbt export {rid}", "zip the whole run (add --no-data to leave out large raw data)"),
        (f"vbt audit {rid} --in-place", "rebuild this README, audit.html and the MANIFEST from the trace"),
    ]


_LAYOUT = [
    ("MANIFEST.json", "status, pinned config, every artifact (sha256, producing agent and tool call), "
                      "harness-record hashes, execution record"),
    ("README.md, audit.html", "these reports"),
    ("inputs/", "query.txt (every turn), plan.json (plan history), config.json (pinned configuration)"),
    ("work/<agent>/", "each specialist's code/, data/ and results/ (the analysis artifacts)"),
    ("work/_mcp/", "outputs written by MCP data servers, attributed to the calling agent"),
    ("evidence/", "claims.json (claim -> evidence), artifacts.json (registered artifacts), provenance.json"),
    ("report/", "FINAL_REPORT.md (raw [[claim:ID]] anchors), FINAL_REPORT.rendered.md, plan_reconciliation"),
    ("logs/", "trace.jsonl (every model and tool call), cost_report.json, transcript.md"),
    ("session_report.json", "turn records: prompt, response, status, agents, cost"),
]


def _evidence_text(ev: Mapping[str, Any]) -> str:
    kind = ev.get("kind") or "artifact"
    if kind == "tool_call":
        s = f"tool call {ev.get('tool_name') or ev.get('tool_use_id') or '?'}"
        if ev.get("agent"):
            s += f" by {ev['agent']}"
        if ev.get("tool_use_id") and ev.get("tool_name"):
            s += f" ({ev['tool_use_id']})"
        return s
    if kind == "citation":
        ref = (f"PMID {ev['pmid']}" if ev.get("pmid") else (f"doi:{ev['doi']}" if ev.get("doi")
                                                             else str(ev.get("url") or "?")))
        return f"citation {ref}" + (f" — {ev['title']}" if ev.get("title") else "")
    s = f"{kind} {ev.get('path', '?')}"
    if ev.get("line"):
        s += f" (line {ev['line']})"
    return s


# ----------------------------------------------------------------- README.md

def render_readme(data: Mapping[str, Any]) -> str:
    """A human map of the run, written to ``README.md`` in the run directory."""
    L: list[str] = []
    rid = data["run_id"]
    L += [f"# Run `{rid}`", ""]
    if data.get("query"):
        q = str(data["query"])
        L += ["> " + _md_inline(q[:1500]) + ("…" if len(q) > 1500 else ""), ""]

    ps = data.get("prov_summary") or {}
    cs = data.get("claim_stats") or {}
    vstatus = _verify_status(data)
    L += ["| | |", "|---|---|",
          f"| **Status** | {_md_cell(data.get('status'))} |",
          f"| **Started** | {_md_cell(data.get('created'))} |",
          f"| **Duration** | {human_duration(data.get('duration_s'))} |",
          f"| **Turns** | {len(data.get('turns') or [])}"
          + (f" (+{len(data['unrecorded_turns'])} without a turn record)" if data.get("unrecorded_turns") else "")
          + " |",
          f"| **Cost** | {_usd(data.get('cost_usd'))} |",
          f"| **Specialists** | {len(data.get('specialists') or [])}"
          + (f" ({_md_cell(', '.join(data['specialists']), 300)})" if data.get("specialists") else "") + " |",
          f"| **Artifacts** | {data.get('n_artifacts', 0)} ({human_bytes(data.get('total_bytes'))}) |"]
    if ps:
        L.append(f"| **Tool calls** | {ps.get('n_tool_calls', 0)} ({ps.get('n_attributed_to_specialist', 0)} by "
                 f"specialists, {ps.get('n_cso_tool_calls', 0)} by the CSO)"
                 + (f"; **{ps['n_tool_errors']} failed**" if ps.get("n_tool_errors") else "") + " |")
    L.append(f"| **Claims** | {cs.get('n_claims', 0)} ({cs.get('n_verified_evidence', 0)} verified, "
             f"{cs.get('n_external_evidence', 0)} external, {cs.get('n_unresolved_evidence', 0)} unresolved "
             "evidence items) |")
    L.append(f"| **Verification** | {vstatus} |")
    L += ["", "Reports: [audit.html](audit.html) · [final report with references](report/FINAL_REPORT.rendered.md) "
              "· [transcript](logs/transcript.md)", ""]

    # problems box
    probs, warns = _problems(data)
    status = str(data.get("status") or "")
    if status == "in_progress":
        L += ["> **Run in progress.** This record is not final; it is rewritten after every turn. A run that "
              "stopped (crash, kill) stays `in_progress` until `vbt audit` reconstructs it.", ""]
    if probs:
        L += ["> **Evidence audit incomplete.** File hashes alone do not establish that the report's conclusions "
              "have recorded support. Problems found by `vbt verify`:", ">"]
        for p in probs[:40]:
            L.append(f"> - **{_md_inline(p.get('kind'))}**: {_md_inline(str(p.get('detail', ''))[:400])}")
        if len(probs) > 40:
            L.append(f"> - … and {len(probs) - 40} more (run `vbt verify {data.get('dir_name')}`)")
        L.append("")
    elif data.get("verify") and vstatus == "COMPLETE":
        L += ["Verification passed: recorded files, claim evidence and report references are consistent.", ""]
    if warns:
        L.append("Warnings (these do not change the status):")
        L.append("")
        for w in warns[:20]:
            L.append(f"- *{_md_inline(w.get('kind'))}*: {_md_inline(str(w.get('detail', ''))[:300])}")
        L.append("")
    v = data.get("verify") or {}
    if v.get("error"):
        L += [f"*Verification could not run: {_md_inline(v['error'])}*", ""]
    L += ["These checks establish a recorded evidence trail; they do not verify scientific correctness or "
          "external citations." + (" Artifact hashes in this report were trusted when the file's size and "
                                   "modification time are unchanged; `vbt verify` re-hashes everything."
                                   if v.get("mode") == "fast" else ""), ""]

    if data.get("notes"):
        L += ["## Notes on this report", ""]
        L += [f"- {_md_inline(n)}" for n in data["notes"]]
        L.append("")

    if data.get("problem_turns"):
        L += ["## Interrupted or failed turns", "", "| Turn | Status | Prompt |", "|---|---|---|"]
        for p in data["problem_turns"]:
            L.append(f"| {p.get('turn')} | {_md_cell(p.get('status'), 120)} | {_md_cell(p.get('prompt'), 120)} |")
        L += ["", "An interrupted or failed turn's answer and evidence record may be incomplete.", ""]

    mis = data.get("misplaced_files") or []
    if mis:
        L += ["## Misplaced files", "",
              "Written outside `work/<agent>/` (or into harness-owned records). They are not artifacts and cannot "
              "be cited.", "", "| Path | Reason | Agent | Tool |", "|---|---|---|---|"]
        for m in mis[:200]:
            L.append(f"| `{_md_cell(m.get('path'), 200)}` | {_md_cell(m.get('reason'))} | "
                     f"{_md_cell(m.get('agent'))} | {_md_cell(m.get('tool'))} |")
        if len(mis) > 200:
            L.append(f"| … {len(mis) - 200} more in MANIFEST.json | | | |")
        L.append("")
    dele = data.get("deleted_artifacts") or []
    if dele:
        L += ["## Deleted artifacts", "", "| Path | Produced by | Cited by | Deleted by |", "|---|---|---|---|"]
        for d in dele[:200]:
            L.append(f"| `{_md_cell(d.get('path'), 200)}` | {_md_cell(d.get('produced_by'))} | "
                     f"{_md_cell(', '.join(d.get('cited_by') or []))} | {_md_cell(d.get('agent'))} |")
        L.append("")
    dse = data.get("data_source_errors") or []
    if dse:
        L += ["## Unavailable data sources", ""]
        for e in dse[:50]:
            if isinstance(e, Mapping):
                L.append(f"- turn {e.get('turn', '?')}: `{_md_inline(e.get('tool_name') or e.get('tool'))}` — "
                         f"{_md_inline(str(e.get('error') or '')[:300])}")
            else:
                L.append(f"- {_md_inline(str(e)[:300])}")
        L.append("")

    # turns
    turns = data.get("turns") or []
    if turns:
        L += ["## Turns", "", "| # | Status | Prompt | Specialists | Duration | Cost | Claims filed |",
              "|---|---|---|---|---|---|---|"]
        for t in turns:
            agents = ", ".join(str(a) for a in t.get("agents") or [])
            filed = t.get("claims_filed")
            L.append(f"| {t.get('turn')} | {_md_cell(t.get('status') or 'completed', 40)} | "
                     f"{_md_cell(t.get('prompt'), 100)} | {_md_cell(agents, 100)} | "
                     f"{human_duration(t.get('duration_s'))} | {_usd(t.get('cost_usd'))} | "
                     f"{_md_cell(', '.join(filed), 60) if filed else '—'} |")
        L += ["", "The CSO's answers, with claim references numbered and resolved, are in "
                  "[report/FINAL_REPORT.rendered.md](report/FINAL_REPORT.rendered.md).", ""]

    # plan
    plan = data.get("plan")
    if plan:
        L.append(render_plan_md(plan, data.get("plan_reconciliation")))
    elif data.get("plan_reconciliation") and len(data.get("specialists") or []) >= 2:
        L += ["## The analysis plan", "", "No plan was recorded (`mcp__provenance__write_plan`), so planned vs "
                                          "actual cannot be compared.", ""]

    # flow
    dels = data.get("delegations") or []
    if dels:
        L += ["## How the analysis flowed", "", "Specialists in dispatch order (from agent_start/agent_end in "
                                                "the trace):", "",
              "| # | Turn | Specialist | Task | Started | Duration | Tool calls | Cost | Status |",
              "|---|---|---|---|---|---|---|---|---|"]
        for i, d in enumerate(dels, 1):
            name = str(d.get("agent")) + (" (briefing)" if d.get("orientation") else "")
            L.append(f"| {i} | {d.get('turn') or '—'} | `{_md_cell(name, 60)}` | {_md_cell(d.get('description'), 60)} "
                     f"| {_short_time(d.get('start')) or '—'} | {human_duration(d.get('duration_s'))} "
                     f"| {d.get('tool_calls') if d.get('tool_calls') is not None else '—'} | {_usd(d.get('cost_usd'))} "
                     f"| {_md_cell(d.get('status'), 30)} |")
        L += ["", "<details><summary>Delegation prompts (what the CSO asked each specialist)</summary>", ""]
        for i, d in enumerate(dels, 1):
            prompt = str(d.get("prompt") or "").strip()
            if not prompt:
                continue
            shown, cut = _clip(prompt, 3000)
            L += [f"**{i}. `{_md_cell(d.get('agent'), 60)}`**", ""]
            L += _fence(shown + ("\n… [truncated; full prompt in logs/trace.jsonl]" if cut else ""))
            L.append("")
        L += ["</details>", ""]

    # artifacts
    L += ["## What each agent produced", ""]
    by_agent = data.get("by_agent") or {}
    if not by_agent:
        L += ["*No artifacts recorded.*", ""]
    for agent in data.get("agent_order") or []:
        arts = by_agent.get(agent) or []
        if not arts:
            continue
        L += [f"### {_md_inline(agent_label(agent))} · {len(arts)} artifact(s)", "",
              "| File | Kind | Size | SHA-256 | Origin | Cited by |", "|---|---|---|---|---|---|"]
        for e in sorted(arts, key=lambda e: (KIND_ORDER.index(e["kind"]) if e["kind"] in KIND_ORDER else 99,
                                             e["path"])):
            origin = e.get("created_by") or ""
            if e.get("tool_use_id"):
                origin = (origin + " " if origin else "") + f"`{_md_cell(str(e['tool_use_id'])[:24], 30)}`"
            sha = f"`{str(e['sha256'])[:12]}`" if e.get("sha256") else "*not hashed*"
            cited = ", ".join(e.get("cited_by") or []) or "—"
            L.append(f"| `{_md_cell(e['path'], 200)}` | {e['kind']} | {human_bytes(e['bytes'])} | {sha} | "
                     f"{_md_cell(origin, 80) if origin else '—'} | {_md_cell(cited, 80)} |")
        L.append("")

    # claims
    claims = data.get("claims") or []
    L += ["## Claims and their evidence", ""]
    if not claims:
        L += ["*No claims were filed.*", ""]
    for c in claims:
        nv = int(c.get("n_verified") or 0)
        badge = "verified" if nv else "no verified evidence"
        L.append(f"### `{_md_cell(c.get('id'), 64)}` — {_md_inline(str(c.get('text') or '')[:600])} ({badge})")
        L.append("")
        meta = [agent_label(c.get("agent") or "unattributed"), f"confidence {c.get('confidence') or 'moderate'}"]
        if c.get("turn") is not None:
            meta.append(f"turn {c['turn']}")
        L += ["*" + _md_inline(" · ".join(meta)) + "*", ""]
        for ev in c.get("evidence") or []:
            if not isinstance(ev, Mapping):
                continue
            st = evidence_status(ev)
            sha = f" `sha256:{str(ev['sha256'])[:12]}`" if ev.get("sha256") else ""
            note = f" — {_md_inline(str(ev['note'])[:300])}" if ev.get("note") else ""
            prob = f" ({_md_inline(str(ev['problem'])[:200])})" if ev.get("problem") and st == "unresolved" else ""
            L.append(f"- **{EVIDENCE_STATUS_LABELS[st]}** {_md_inline(_evidence_text(ev))}{sha}{note}{prob}")
        L.append("")
    bad = cs.get("claims_without_verified_evidence") or []
    if bad:
        L += [f"> **{len(bad)} claim(s) have no locally verified evidence:** {_md_inline(', '.join(map(str, bad)))}. "
              "External references need source review.", ""]

    # reproduce
    L += ["## Reproducing this run", "", "```bash"]
    for cmd, why in _reproduce_cmds(data):
        L.append(f"{cmd:<44} # {why}")
    L += ["```", "",
          "`verify` checks what was recorded; `--rerun` re-executes the scripts agents wrote. `replay` re-runs the "
          "agents with the same pinned models and prompts: LLM sampling makes it a comparison, not a "
          "bit-identical reproduction.", ""]

    cfg = data.get("config") or {}
    if cfg:
        text = json.dumps(cfg, indent=2, default=str)
        shown, cut = _clip(text, 20000)
        L += ["<details><summary>Pinned configuration</summary>", ""]
        L += _fence(shown + ("\n… [truncated; full configuration in inputs/config.json]" if cut else ""), "json")
        L += ["", "</details>", ""]

    L += ["## Directory layout", "", "| Path | Contents |", "|---|---|"]
    for p, what in _LAYOUT:
        L.append(f"| `{p}` | {_md_cell(what, 300)} |")
    L += ["", f"*Generated {data.get('generated')} by `vbt.audit.report` from the run's on-disk records.*", ""]
    return "\n".join(L)


# ----------------------------------------------------------------- audit.html

#: The only script in audit.html (tool-table pagination). Its hash is pinned in the
#: page's Content-Security-Policy, so nothing else can execute.
AUDIT_SCRIPT = """
(function(){
  var tbl=document.getElementById('tool-table'); if(!tbl) return;
  var rows=[].slice.call(tbl.tBodies[0].rows), size=PAGE, page=0;
  var only=document.getElementById('errors-only'), info=document.getElementById('tool-page');
  var prev=document.getElementById('tool-prev'), next=document.getElementById('tool-next');
  document.getElementById('tool-pager').hidden=false;
  function vis(){return rows.filter(function(r){return !only.checked||r.getAttribute('data-err')==='1';});}
  function draw(){
    var v=vis(), pages=Math.max(1,Math.ceil(v.length/size)); if(page>=pages) page=pages-1; if(page<0) page=0;
    rows.forEach(function(r){r.hidden=true;});
    v.slice(page*size,(page+1)*size).forEach(function(r){r.hidden=false;});
    info.textContent='Page '+(page+1)+' of '+pages+' \\u00b7 '+v.length+' call(s)';
    prev.disabled=page===0; next.disabled=page>=pages-1;
  }
  prev.onclick=function(){page--;draw();}; next.onclick=function(){page++;draw();};
  only.onchange=function(){page=0;draw();}; draw();
})();
""".replace("PAGE", str(TOOL_PAGE_SIZE))

AUDIT_SCRIPT_HASH = "sha256-" + base64.b64encode(hashlib.sha256(AUDIT_SCRIPT.encode("utf-8")).digest()).decode()

#: CSP for audit.html (also sent as a header by the web UI).
AUDIT_CSP = (f"default-src 'none'; style-src 'unsafe-inline'; img-src data:; script-src '{AUDIT_SCRIPT_HASH}'; "
             "base-uri 'none'; form-action 'none'")

_CSS = """
:root{color-scheme:light dark;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
  --grid:#e1e0d9;--rule:#c3c2b7;--border:rgba(11,11,11,.10);--good:#0ca30c;--goodtext:#006300;--warn:#fab219;
  --crit:#d03b3b;--link:#256abf}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;
  --ink2:#c3c2b7;--grid:#2c2c2a;--rule:#383835;--border:rgba(255,255,255,.10);--goodtext:#0ca30c;--link:#86b6ef}}
:root[data-theme=dark]{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--rule:#383835;
  --border:rgba(255,255,255,.10);--goodtext:#0ca30c;--link:#86b6ef}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--link)}
.wrap{max-width:1120px;margin:0 auto;padding:32px 16px 72px}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em}
.dim{color:var(--muted)}
.hdr{border-bottom:1px solid var(--rule);padding-bottom:18px;margin-bottom:20px}
.eyebrow{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
h1{font-size:24px;line-height:1.3;margin:8px 0 6px;font-weight:650;overflow-wrap:anywhere}
.sub{color:var(--ink2);font-size:13px}
h2{font-size:19px;margin:36px 0 4px;font-weight:620}
h3{font-size:15px;margin:20px 0 8px;font-weight:600;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.lede{color:var(--ink2);font-size:13.5px;margin:6px 0 14px;max-width:80ch}
section{border-top:1px solid var(--grid);margin-top:8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:18px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile-l{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.tile-v{font-size:22px;font-weight:640;margin:2px 0 1px;overflow-wrap:break-word;hyphens:auto}
.tile-s{font-size:11.5px;color:var(--ink2)}
.note{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--muted);border-radius:8px;
  padding:12px 16px;margin:16px 0;font-size:13.5px;color:var(--ink2)}
.note ul{margin:6px 0 0;padding-left:18px}
.note li{margin:2px 0;overflow-wrap:anywhere}
.warnbox{border-left-color:var(--warn)}
.critbox{border-left-color:var(--crit)}
.okbox{border-left-color:var(--good)}
.empty{color:var(--muted);font-style:italic}
table{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0 14px}
th{text-align:left;font-weight:600;color:var(--ink2);font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  border-bottom:1px solid var(--rule);padding:6px 8px;white-space:nowrap}
td{padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.err td{background:color-mix(in srgb,var(--crit) 8%,transparent)}
.scroll{overflow:auto;max-height:640px;border:1px solid var(--border);border-radius:8px}
.scroll table{margin:0}
.scroll th{position:sticky;top:0;background:var(--surface);z-index:1}
.xscroll{overflow-x:auto;max-width:100%}
.xscroll table{min-width:620px}
td code{overflow-wrap:anywhere}
.prompt{color:var(--ink2);min-width:24ch;max-width:52ch;overflow-wrap:anywhere}
.agent{margin:16px 0 24px}
.swatch{width:10px;height:10px;border-radius:3px;background:var(--ac,var(--muted));flex:none;display:inline-block}
.swatch.sm{width:8px;height:8px;margin-right:6px}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .swatch{background:var(--acd,var(--muted))}}
:root[data-theme=dark] .swatch{background:var(--acd,var(--muted))}
.count{margin-left:auto;font-weight:400;font-size:12px;color:var(--muted)}
.kind{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:10px 0 2px}
.tl{width:100%;height:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px;
  margin:8px 0 16px}
.tl .grid{stroke:var(--grid);stroke-width:1}
.tl .turn{stroke:var(--rule);stroke-width:1;stroke-dasharray:3 3}
.tl .tick,.tl .durlab,.tl .turnlab{font:11px system-ui,sans-serif;fill:var(--muted)}
.tl .rowlab{font:11.5px system-ui,sans-serif;fill:var(--ink2)}
.tl .bar{fill:var(--bar);stroke:var(--surface);stroke-width:2}
.tl .bar.open{fill-opacity:.45;stroke:var(--bar);stroke-dasharray:4 3}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .tl .bar{fill:var(--bard)}}
:root[data-theme=dark] .tl .bar{fill:var(--bard)}
.claim{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--good);border-radius:8px;
  padding:12px 16px;margin:10px 0}
.claim.warn{border-left-color:var(--warn)}
.claim:target{outline:2px solid var(--link)}
.claim-h{display:flex;gap:10px;align-items:baseline}
.cid{font:600 11px ui-monospace,monospace;color:var(--muted);flex:none}
.ctext{font-weight:550;overflow-wrap:anywhere}
.claim-m{font-size:12px;color:var(--muted);margin:3px 0 8px}
ul.ev{margin:0;padding-left:18px;font-size:13px}
ul.ev li{margin:3px 0;overflow-wrap:anywhere}
.ek{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-right:4px}
.vb{font-size:10.5px;padding:1px 7px;border-radius:20px;margin-left:6px;white-space:nowrap;
  border:1px solid color-mix(in srgb,currentColor 45%,transparent)}
.vb.verified,.vb.ok{color:var(--goodtext)}
.vb.external{color:var(--muted)}
.vb.unresolved,.vb.error{color:var(--crit)}
.vb.running{color:var(--ink2)}
.cref{color:inherit;text-decoration:none;border-bottom:1px dotted var(--muted);font:600 11px ui-monospace,monospace}
pre{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px;overflow-x:auto;
  font-size:12.5px;margin:8px 0;white-space:pre-wrap;overflow-wrap:anywhere}
.answer{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;background:var(--surface);
  border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin:8px 0}
details{margin:8px 0}
summary{cursor:pointer;font-size:13px;color:var(--ink2);padding:4px 0}
.pager{display:flex;gap:10px;align-items:center;font-size:13px;color:var(--ink2);margin:8px 0;flex-wrap:wrap}
.pager button{font:inherit;padding:3px 10px;border-radius:6px;border:1px solid var(--rule);background:var(--surface);
  color:var(--ink);cursor:pointer}
.pager button:disabled{opacity:.4;cursor:default}
.errtxt{color:var(--crit);font-size:12px;overflow-wrap:anywhere}
footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--grid);font-size:12px;color:var(--muted)}
"""


def _swatch(colors: Mapping[str, tuple[str, str]], agent: str, sm: bool = True) -> str:
    lc, dc = colors.get(agent, _MUTED)
    return f'<span class="swatch{" sm" if sm else ""}" style="--ac:{lc};--acd:{dc}"></span>'


def _ev_html(ev: Mapping[str, Any]) -> str:
    st = evidence_status(ev)
    badge = f'<span class="vb {st}">{_esc(EVIDENCE_STATUS_LABELS[st])}</span>'
    kind = ev.get("kind") or "artifact"
    if kind == "tool_call":
        body = (f'<span class="ek">tool call</span> <code>{_esc(ev.get("tool_name") or ev.get("tool_use_id"))}</code>'
                + (f' <span class="dim">by {_esc(ev["agent"])}</span>' if ev.get("agent") else "")
                + (f' <code class="dim">{_esc(ev["tool_use_id"])}</code>' if ev.get("tool_use_id") else ""))
    elif kind == "citation":
        if ev.get("pmid"):
            label, url = f"PMID {ev['pmid']}", f"https://pubmed.ncbi.nlm.nih.gov/{ev['pmid']}/"
        elif ev.get("doi"):
            label, url = f"doi:{ev['doi']}", "https://doi.org/" + str(ev["doi"])
        else:
            label, url = str(ev.get("url") or "?"), ev.get("url")
        url = _safe_url(url)
        ref = (f'<a href="{_esc(url)}" rel="noopener noreferrer" target="_blank">{_esc(label)}</a>' if url
               else _esc(label))
        body = f'<span class="ek">citation</span> {ref}' + (f' — {_esc(ev["title"])}' if ev.get("title") else "")
    else:
        body = f'<span class="ek">{_esc(kind)}</span> <code>{_esc(ev.get("path"))}</code>'
        if ev.get("line"):
            body += f' <span class="dim">line {_esc(ev["line"])}</span>'
        if ev.get("sha256"):
            body += f' <code class="dim" title="{_esc(ev["sha256"])}">{_esc(str(ev["sha256"])[:12])}</code>'
    if ev.get("note"):
        body += f' <span class="dim">— {_esc(ev["note"])}</span>'
    if ev.get("problem") and st == "unresolved":
        body += f' <span class="errtxt">({_esc(str(ev["problem"])[:300])})</span>'
    return body + " " + badge


def _answer_html(text: str, claims: list[Mapping[str, Any]]) -> str:
    """Escaped answer text with [[claim:ID]] anchors turned into numbered links to claim cards."""
    _, footnotes = number_refs(text, claims)
    labels = {fn["id"]: fn for fn in footnotes}
    out, pos = [], 0
    for m in CLAIM_RE.finditer(text):
        out.append(_esc(text[pos:m.start()]))
        fn = labels.get(m.group(1))
        if fn and not fn.get("missing"):
            out.append(f'<a class="cref" href="#claim-{_esc(fn["id"])}" title="{_esc(fn["id"])}">'
                       f'[{_esc(fn["label"])}]</a>')
        else:
            out.append(f'<span class="vb unresolved" title="no claim with this id was filed">'
                       f'{_esc(m.group(1))}?</span>')
        pos = m.end()
    out.append(_esc(text[pos:]))
    return "".join(out)


def _timeline_svg(data: Mapping[str, Any]) -> str:
    """One row per dispatched specialist run, directly labelled; colour only reinforces identity."""
    colors = data.get("colors") or {}
    dels = [d for d in data.get("delegations") or [] if d.get("start_t") is not None]
    if not dels:
        return ""
    last_t = (data.get("trace_span") or (None, None))[1]
    spans = []
    seen: dict[str, int] = {}
    for d in dels:
        st = float(d["start_t"])
        en = d.get("end_t")
        open_ = en is None
        en = float(en) if en is not None else float(last_t if last_t is not None else st)
        name = str(d.get("agent"))
        seen[name] = seen.get(name, 0) + 1
        label = name + (f" #{seen[name]}" if seen[name] > 1 else "") + (" (brief)" if d.get("orientation") else "")
        spans.append((name, label, st, max(en, st), open_, d.get("status")))
    t0 = min(s[2] for s in spans)
    t1 = max(s[3] for s in spans)
    total = max(t1 - t0, 1.0)
    row_h, gap, pad_l, pad_t, width = 22, 6, 220, 26, 960
    plot_w = width - pad_l - 64
    height = pad_t + len(spans) * (row_h + gap) + 28
    unit, div = ("m", 60.0) if total >= 120 else ("s", 1.0)
    out = [f'<svg class="tl" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="Timeline of {len(spans)} specialist runs over {human_duration(total)}">']
    for frac in (0, .25, .5, .75, 1.0):
        x = pad_l + plot_w * frac
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{pad_t - 8}" x2="{x:.1f}" y2="{height - 22}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{height - 7}" text-anchor="middle">'
                   f'+{total * frac / div:.1f}{unit}</text>')
    for n, w in sorted((data.get("turn_windows") or {}).items()):
        ts = w.get("start_t") if isinstance(w, Mapping) else None
        if ts is None or not (t0 <= float(ts) <= t1):
            continue
        x = pad_l + plot_w * ((float(ts) - t0) / total)
        out.append(f'<line class="turn" x1="{x:.1f}" y1="{pad_t - 14}" x2="{x:.1f}" y2="{height - 22}"/>')
        out.append(f'<text class="turnlab" x="{x + 3:.1f}" y="{pad_t - 16}">turn {_esc(n)}</text>')
    for i, (name, label, st, en, open_, status) in enumerate(spans):
        y = pad_t + i * (row_h + gap)
        x = pad_l + plot_w * ((st - t0) / total)
        w = max(plot_w * ((en - st) / total), 3)
        lc, dc = colors.get(name, _MUTED)
        secs = en - st
        tip = f"{label}: {human_duration(secs)}; {status or ''}" + (" (no agent_end recorded)" if open_ else "")
        out.append(f'<text class="rowlab" x="{pad_l - 10}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end">'
                   f'{_esc(label[:34])}</text>')
        out.append(f'<rect class="bar{" open" if open_ else ""}" x="{x:.1f}" y="{y}" width="{w:.1f}" '
                   f'height="{row_h}" rx="4" style="--bar:{lc};--bard:{dc}"><title>{_esc(tip)}</title></rect>')
        out.append(f'<text class="durlab" x="{min(x + w + 6, width - 4):.1f}" y="{y + row_h / 2 + 4:.1f}">'
                   f'{_esc(human_duration(secs))}{" …" if open_ else ""}</text>')
    out.append("</svg>")
    return "".join(out)


def render_audit_html(data: Mapping[str, Any]) -> str:
    """One self-contained HTML file: no external requests, all model text escaped."""
    colors = data.get("colors") or {}
    ps = data.get("prov_summary") or {}
    cs = data.get("claim_stats") or {}
    claims = data.get("claims") or []
    P: list[str] = []
    title = f"Audit · {data['run_id']}"
    P.append("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
             f'<meta http-equiv="Content-Security-Policy" content="{_esc(AUDIT_CSP)}">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><div class=\"wrap\">")
    counter = [0]

    def sec(name: str, anchor: str) -> str:
        counter[0] += 1
        return f'<section id="{anchor}"><h2>{counter[0]} · {_esc(name)}</h2>'

    # header + tiles
    P.append('<header class="hdr"><div class="eyebrow">The Virtual Biotech · run audit</div>'
             f'<h1>{_esc((data.get("query") or data["run_id"])[:400])}</h1>'
             f'<div class="sub"><code>{_esc(data["run_id"])}</code> · {_esc(data.get("created") or "")} · status '
             f'<strong>{_esc(data.get("status"))}</strong></div></header>')
    vstatus = _verify_status(data)
    tiles = [
        ("Status", str(data.get("status")).replace("_", " "), f"verification: {vstatus}"),
        ("Duration", human_duration(data.get("duration_s")), f"{len(data.get('turns') or [])} turn(s)"),
        ("Cost", _usd(data.get("cost_usd")), "model + tool fees"),
        ("Specialists", str(len(data.get("specialists") or [])), f"{len(data.get('delegations') or [])} dispatches"),
        ("Artifacts", str(data.get("n_artifacts", 0)), human_bytes(data.get("total_bytes"))),
        ("Tool calls", str(ps.get("n_tool_calls", 0)), f"{ps.get('n_tool_errors', 0)} failed"),
        ("Claims", str(cs.get("n_claims", 0)), f"{cs.get('n_verified_evidence', 0)} verified · "
                                               f"{cs.get('n_unresolved_evidence', 0)} unresolved evidence"),
    ]
    P.append('<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tile-l">{_esc(label)}</div><div class="tile-v">{_esc(v)}</div>'
        f'<div class="tile-s">{_esc(s)}</div></div>' for label, v, s in tiles) + "</div>")

    # problems box
    probs, warns = _problems(data)
    v = data.get("verify") or {}
    if data.get("status") == "in_progress":
        P.append('<div class="note warnbox"><strong>Run in progress.</strong> This record is not final; it is '
                 'rewritten after every turn. A run that stopped (crash, kill) stays <code>in_progress</code> until '
                 '<code>vbt audit</code> reconstructs it.</div>')
    if probs:
        box = "critbox" if vstatus == "FAIL" else "warnbox"
        P.append(f'<div class="note {box}"><strong>Evidence audit {"failed" if vstatus == "FAIL" else "incomplete"}'
                 f' ({len(probs)} problem{"s" if len(probs) != 1 else ""}).</strong> File hashes alone do not '
                 "establish that the report's conclusions have recorded support.<ul>")
        for p in probs:
            P.append(f'<li><strong>{_esc(p.get("kind"))}</strong>: {_esc(str(p.get("detail", ""))[:600])}</li>')
        P.append("</ul></div>")
    elif data.get("verify") and vstatus == "COMPLETE":
        P.append('<div class="note okbox"><strong>Verification passed.</strong> Recorded files, claim evidence and '
                 "report references are consistent.</div>")
    if warns:
        P.append('<div class="note"><strong>Warnings</strong> (do not change the status)<ul>'
                 + "".join(f'<li><em>{_esc(w.get("kind"))}</em>: {_esc(str(w.get("detail", ""))[:400])}</li>'
                           for w in warns) + "</ul></div>")
    if v.get("error"):
        P.append(f'<div class="note warnbox">Verification could not run: {_esc(v["error"])}</div>')
    P.append('<p class="lede">These checks establish a recorded evidence trail; they do not verify scientific '
             'correctness or external citations.'
             + (" Artifact hashes here were trusted when a file's size and modification time are unchanged; "
                "<code>vbt verify</code> re-hashes everything." if v.get("mode") == "fast" else "") + "</p>")
    if data.get("notes"):
        P.append('<div class="note"><strong>About this report.</strong><ul>'
                 + "".join(f"<li>{_esc(n)}</li>" for n in data["notes"]) + "</ul></div>")
    if data.get("problem_turns"):
        P.append('<div class="note warnbox"><strong>Interrupted or failed turns.</strong> Their answers and evidence '
                 'records may be incomplete.<ul>' + "".join(
                     f'<li>Turn {_esc(p.get("turn"))}: <strong>{_esc(p.get("status"))}</strong>'
                     + (f' — {_esc(str(p.get("prompt"))[:200])}' if p.get("prompt") else "") + "</li>"
                     for p in data["problem_turns"]) + "</ul></div>")
    mis = data.get("misplaced_files") or []
    if mis:
        P.append('<div class="note warnbox"><strong>Misplaced files.</strong> Written outside '
                 '<code>work/&lt;agent&gt;/</code> or into harness-owned records; they are not artifacts and cannot '
                 'be cited.<ul>' + "".join(
                     f'<li><code>{_esc(m.get("path"))}</code> — {_esc(m.get("reason"))}'
                     + (f' (by {_esc(m.get("agent"))}' + (f' via {_esc(m.get("tool"))}' if m.get("tool") else "")
                        + ")" if m.get("agent") else "") + "</li>" for m in mis[:500])
                 + (f"<li>… {len(mis) - 500} more in MANIFEST.json</li>" if len(mis) > 500 else "") + "</ul></div>")
    dele = data.get("deleted_artifacts") or []
    if dele:
        P.append('<div class="note warnbox"><strong>Deleted artifacts.</strong><ul>' + "".join(
            f'<li><code>{_esc(d.get("path"))}</code> (produced by {_esc(d.get("produced_by"))}'
            + (f'; cited by {_esc(", ".join(d.get("cited_by") or []))}' if d.get("cited_by") else "") + ")</li>"
            for d in dele[:500]) + "</ul></div>")
    dse = data.get("data_source_errors") or []
    if dse:
        P.append('<div class="note warnbox"><strong>Unavailable data sources.</strong><ul>' + "".join(
            (f'<li>turn {_esc(e.get("turn", "?"))}: <code>{_esc(e.get("tool_name") or e.get("tool"))}</code> — '
             f'{_esc(str(e.get("error") or "")[:300])}</li>') if isinstance(e, Mapping)
            else f"<li>{_esc(str(e)[:300])}</li>" for e in dse[:100]) + "</ul></div>")
    if data.get("audit_errors"):
        P.append('<div class="note critbox"><strong>Audit-capture errors.</strong><ul>' + "".join(
            f"<li>{_esc(e[:500])}</li>" for e in data["audit_errors"][:50]) + "</ul></div>")

    # turns
    turns = data.get("turns") or []
    if turns or data.get("unrecorded_turns"):
        P.append(sec("Conversation", "turns"))
        P.append('<p class="lede">Each user turn and the CSO\'s answer. Bracketed numbers link to the claim cards '
                 'below; a red id is a reference to a claim that was never filed.</p>')
        for t in turns:
            st = str(t.get("status") or "completed")
            agents = ", ".join(str(a) for a in t.get("agents") or [])
            meta = " · ".join(x for x in (
                f"status {st}", human_duration(t.get("duration_s")), _usd(t.get("cost_usd")),
                f"specialists: {agents}" if agents else "",
                f"claims filed: {', '.join(t['claims_filed'])}" if t.get("claims_filed") else "") if x)
            opened = " open" if len(turns) <= 3 else ""
            P.append(f'<details{opened}><summary><strong>Turn {_esc(t.get("turn"))}</strong>'
                     f' — {_esc(str(t.get("prompt") or "")[:200])} <span class="dim">({_esc(meta)})</span></summary>')
            P.append(f'<div class="answer">{_answer_html(str(t.get("response") or ""), claims)}</div></details>')
        for t in data.get("unrecorded_turns") or []:
            P.append(f'<details><summary><strong>Turn {_esc(t.get("turn"))}</strong> — '
                     f'{_esc(str(t.get("prompt") or "")[:200])} <span class="vb unresolved">'
                     f'{_esc(t.get("status"))}; no turn record</span></summary>'
                     '<p class="empty">The run stopped before this turn was recorded.</p></details>')
        P.append("</section>")

    # plan
    plan = data.get("plan")
    if plan:
        rec = data.get("plan_reconciliation") or {}
        P.append(sec("The analysis plan", "plan"))
        P.append('<p class="lede">The steps the CSO declared with <code>write_plan</code> and how the observed '
                 'dispatches compare. Deviations are recorded, not forbidden.</p>')
        if plan.get("goal"):
            P.append(f"<p><strong>Goal:</strong> {_esc(plan['goal'])}</p>")
        P.append('<div class="xscroll"><table><thead><tr><th>Step</th><th>Specialist</th><th>Depends on</th>'
                 '<th>Task</th><th>Expected outputs</th></tr></thead><tbody>')
        for s in plan.get("steps") or []:
            if not isinstance(s, Mapping):
                continue
            P.append(f'<tr><td><code>{_esc(s.get("id"))}</code></td><td>{_swatch(colors, str(s.get("agent")))}'
                     f'<code>{_esc(s.get("agent"))}</code></td>'
                     f'<td>{_esc(", ".join(map(str, s.get("depends_on") or [])) or "—")}</td>'
                     f'<td class="prompt">{_esc(s.get("task"))}</td>'
                     f'<td>{_esc(", ".join(map(str, s.get("expected_outputs") or [])) or "—")}</td></tr>')
        P.append("</tbody></table></div>")
        devs = rec.get("deviations") or []
        P.append(f'<div class="note {"warnbox" if devs else "okbox"}"><strong>Planned vs actual.</strong> '
                 f'{_esc(rec.get("summary", ""))}')
        if devs:
            P.append("<ul>" + "".join(f"<li><em>{_esc(d.get('kind'))}</em> — {_esc(d.get('detail'))}</li>"
                                      for d in devs) + "</ul>")
        P.append("</div></section>")

    # flow
    dels = data.get("delegations") or []
    if dels:
        P.append(sec("How the analysis flowed", "flow"))
        P.append('<p class="lede">Each bar is one specialist run, from agent_start to agent_end in the trace; '
                 'overlapping bars ran concurrently. Dashed bars never recorded an end.</p>')
        P.append(_timeline_svg(data))
        P.append('<div class="xscroll"><table><thead><tr><th>#</th><th>Turn</th><th>Specialist</th><th>Started</th>'
                 '<th>Duration</th><th>Model calls</th><th>Tool calls</th><th>Cost</th><th>Status</th>'
                 '<th>Task given by the CSO</th></tr></thead><tbody>')
        for i, d in enumerate(dels, 1):
            prompt = str(d.get("prompt") or "").strip()
            short, cut = _clip(prompt, 300)
            name = str(d.get("agent"))
            status = str(d.get("status") or "")
            P.append(
                f'<tr><td class="num">{i}</td><td class="num">{_esc(d.get("turn") or "—")}</td>'
                f'<td>{_swatch(colors, name)}<code>{_esc(name)}</code>'
                + (' <span class="dim">(briefing)</span>' if d.get("orientation") else "")
                + (f'<div class="dim">{_esc(d.get("description"))}</div>' if d.get("description") else "")
                + f'</td><td class="num">{_esc(_short_time(d.get("start")) or "—")}</td>'
                f'<td class="num">{_esc(human_duration(d.get("duration_s")))}</td>'
                f'<td class="num">{_esc(d.get("model_calls") if d.get("model_calls") is not None else "—")}</td>'
                f'<td class="num">{_esc(d.get("tool_calls") if d.get("tool_calls") is not None else "—")}</td>'
                f'<td class="num">{_esc(_usd(d.get("cost_usd")))}</td>'
                f'<td><span class="vb {"ok" if status == "completed" else "error"}">{_esc(status or "?")}</span></td>'
                f'<td class="prompt">' + (f"<details><summary>{_esc(short[:120])}…</summary><pre>{_esc(prompt)}</pre>"
                                          "</details>" if len(prompt) > 120 else _esc(prompt) or "—")
                + "</td></tr>")
        P.append("</tbody></table></div></section>")

    # artifacts
    P.append(sec("What each agent produced", "artifacts"))
    P.append('<p class="lede">Every file under <code>work/</code>, grouped by the agent that produced it, with the '
             'hash recorded when it was written, the call or script line that wrote it, and the claims citing it.</p>')
    by_agent = data.get("by_agent") or {}
    if not by_agent:
        P.append('<p class="empty">No artifacts recorded for this run.</p>')
    for agent in data.get("agent_order") or []:
        arts = by_agent.get(agent) or []
        if not arts:
            continue
        P.append(f'<div class="agent"><h3>{_swatch(colors, agent, sm=False)}{_esc(agent_label(agent))}'
                 f'<span class="count">{len(arts)} artifact(s) · '
                 f'{_esc(human_bytes(sum(a["bytes"] for a in arts)))}</span></h3>')
        for kind in KIND_ORDER + sorted({a["kind"] for a in arts} - set(KIND_ORDER)):
            group = [a for a in arts if a["kind"] == kind]
            if not group:
                continue
            P.append(f'<div class="kind">{_esc(KIND_LABEL.get(kind, kind))}</div>'
                     '<div class="xscroll"><table><thead><tr><th>File</th><th>Size</th><th>Origin</th>'
                     '<th>SHA-256</th><th>Cited by</th></tr></thead><tbody>')
            for e in sorted(group, key=lambda e: e["path"]):
                origin = _esc(e.get("created_by") or "")
                if e.get("tool_use_id"):
                    origin += (" " if origin else "") + f'<code class="dim">{_esc(str(e["tool_use_id"])[:24])}</code>'
                cited = " ".join(f'<a class="cref" href="#claim-{_esc(c)}">{_esc(c)}</a>'
                                 for c in e.get("cited_by") or []) or '<span class="dim">—</span>'
                desc = f'<div class="dim">{_esc(e["description"])}</div>' if e.get("description") else ""
                sha = (f'<code class="dim" title="{_esc(e["sha256"])}">{_esc(str(e["sha256"])[:12])}</code>'
                       if e.get("sha256") else '<span class="dim">not hashed</span>')
                P.append(f'<tr><td><code>{_esc(e["path"])}</code>{desc}</td>'
                         f'<td class="num">{_esc(human_bytes(e["bytes"]))}</td>'
                         f'<td>{origin or "<span class=dim>—</span>"}</td><td>{sha}</td><td>{cited}</td></tr>')
            P.append("</tbody></table></div>")
        P.append("</div>")
    P.append("</section>")

    # claims
    P.append(sec("Claims and their evidence", "claims"))
    P.append('<p class="lede">Each filed claim with the evidence behind it. <strong>verified</strong> means the '
             "pointer resolved against this run's records (not that the science was checked); <strong>external "
             "ref</strong> is a citation that cannot be checked locally; <strong>not on record</strong> is a local "
             "pointer that does not resolve.</p>")
    if not claims:
        P.append('<p class="empty">No claims were filed.</p>')
    for c in claims:
        nv = int(c.get("n_verified") or 0)
        cid = str(c.get("id"))
        meta = [agent_label(c.get("agent") or "unattributed"), f"confidence {c.get('confidence') or 'moderate'}"]
        if c.get("turn") is not None:
            meta.append(f"turn {c['turn']}")
        P.append(f'<div class="claim {"ok" if nv else "warn"}" id="claim-{_esc(cid)}"><div class="claim-h">'
                 f'<span class="cid">{_esc(cid)}</span><span class="ctext">{_esc(c.get("text"))}</span></div>'
                 f'<div class="claim-m">{_esc(" · ".join(meta))}'
                 + ("" if nv else ' · <span class="vb unresolved">no verified evidence</span>')
                 + '</div><ul class="ev">'
                 + "".join(f"<li>{_ev_html(ev)}</li>" for ev in c.get("evidence") or [] if isinstance(ev, Mapping))
                 + "</ul></div>")
    P.append("</section>")

    # tool provenance
    rows = data.get("tool_calls") or []
    P.append(sec("Full tool provenance", "tools"))
    P.append('<p class="lede">Every tool call in the run, attributed to the agent that made it, reconstructed from '
             "the execution trace rather than any agent's self-report. Failed calls are highlighted.</p>")
    if not rows:
        P.append('<p class="empty">No tool calls were recorded.</p>')
    else:
        shown = rows[:MAX_TOOL_ROWS]
        n_err = sum(1 for r in rows if r["status"] == "error")
        P.append('<div class="pager" id="tool-pager" hidden><button type="button" id="tool-prev">Previous</button>'
                 '<span id="tool-page"></span><button type="button" id="tool-next">Next</button>'
                 f'<label><input type="checkbox" id="errors-only"> errors only ({n_err})</label></div>')
        P.append('<div class="scroll"><table id="tool-table"><thead><tr><th>#</th><th>Time</th><th>Turn</th>'
                 '<th>Agent</th><th>Tool</th><th>Status</th><th>Duration</th><th>Files</th></tr></thead><tbody>')
        for i, r in enumerate(shown, 1):
            err = r["status"] == "error"
            files = list(r.get("files_written") or []) + [f"{p} (returned)" for p in r.get("files_returned") or []]
            files_html = "<br>".join(f"<code class=\"dim\">{_esc(f)}</code>" for f in files[:8])
            if len(files) > 8:
                files_html += f'<br><span class="dim">+{len(files) - 8} more</span>'
            detail = f'<div class="errtxt">{_esc(r["error"])}</div>' if err and r.get("error") else ""
            P.append(f'<tr{" class=err" if err else ""} data-err="{1 if err else 0}"><td class="num">{i}</td>'
                     f'<td class="num">{_esc(_short_time(r.get("started_at")))}</td>'
                     f'<td class="num">{_esc(r.get("turn") if r.get("turn") is not None else "—")}</td>'
                     f'<td>{_swatch(colors, str(r["agent"]))}<code>{_esc(r["agent"])}</code></td>'
                     f'<td><code>{_esc(r["tool"])}</code><div class="dim"><code>{_esc(r.get("tool_use_id"))}</code>'
                     f'</div>{detail}</td><td><span class="vb {_esc(r["status"])}">{_esc(r["status"])}</span></td>'
                     f'<td class="num">{_esc(human_duration(r.get("duration_s")))}</td><td>{files_html}</td></tr>')
        P.append("</tbody></table></div>")
        if len(rows) > len(shown):
            P.append(f'<p class="note warnbox">Showing the first {len(shown)} of {len(rows)} tool calls; the other '
                     f'{len(rows) - len(shown)} are in <code>evidence/provenance.json</code> and '
                     '<code>logs/trace.jsonl</code>.</p>')
        else:
            P.append(f'<p class="dim">{len(rows)} tool call(s), all listed.</p>')
    P.append("</section>")

    # reproduce + config + layout
    P.append(sec("Reproducing this run", "reproduce"))
    P.append("<pre>" + "\n".join(f"{_esc(cmd):<44} <span class=\"dim\"># {_esc(why)}</span>"
                                 for cmd, why in _reproduce_cmds(data)) + "</pre>")
    P.append('<p class="lede"><code>verify</code> checks what was recorded; <code>--rerun</code> re-executes the '
             "scripts agents wrote in a scratch copy. <code>replay</code> re-runs the agents with the same pinned "
             "models and prompts; LLM sampling makes it a comparison, not a bit-identical reproduction.</p>")
    cfg = data.get("config") or {}
    if cfg:
        text, cut = _clip(json.dumps(cfg, indent=2, default=str), 200000)
        P.append(f"<details><summary>Pinned configuration</summary><pre>{_esc(text)}"
                 + ("\n… [truncated; full configuration in inputs/config.json]" if cut else "") + "</pre></details>")
    P.append('<details><summary>Directory layout</summary><table><tbody>' + "".join(
        f"<tr><td><code>{_esc(p)}</code></td><td>{_esc(w)}</td></tr>" for p, w in _LAYOUT)
        + "</tbody></table></details>")
    P.append("</section>")
    P.append(f'<footer>Generated {_esc(data.get("generated"))} by vbt.audit.report from the run\'s on-disk records · '
             "self-contained, no external assets</footer></div>")
    P.append(f"<script>{AUDIT_SCRIPT}</script></body></html>")
    return "\n".join(P)


# ----------------------------------------------------------------- writing

def _report_verify_mode(run: Any) -> str:
    for source in (getattr(run, "audit_settings", None),
                   (getattr(run, "config", None) or {}).get("audit") if isinstance(getattr(run, "config", None),
                                                                                    dict) else None):
        if isinstance(source, Mapping) and source.get("report_verify") is not None:
            return str(source["report_verify"])
    return DEFAULT_REPORT_VERIFY


def _refresh_manifest_hashes(run_dir: Path, names: tuple[str, ...]) -> None:
    """Record the new report hashes in MANIFEST.harness_files (bare-directory mode)."""
    mpath = run_dir / "MANIFEST.json"
    if not mpath.is_file():
        return
    with run_lock(run_dir):
        m = read_json(mpath, None)
        if not isinstance(m, dict) or not isinstance(m.get("harness_files"), dict):
            return
        changed = False
        for name in names:
            p = run_dir / name
            if p.is_file():
                digest = sha256_file(p)
                if m["harness_files"].get(name) != digest:
                    m["harness_files"][name] = digest
                    changed = True
        if changed:
            m["harness_files"] = dict(sorted(m["harness_files"].items()))
            write_json_atomic(mpath, m, compact=len(m.get("artifacts") or {}) > 200)


def write_reports(run_or_dir: Any, *, notes: list[str] | None = None, verify: Any = None,
                  update_manifest: bool = True) -> dict[str, Path]:
    """Render and atomically write ``README.md`` and ``audit.html``. Returns their paths.

    ``run_or_dir`` is a ``vbt.session.Run`` (which re-hashes the reports itself
    after this returns) or a run directory, in which case the reports' hashes in
    MANIFEST.harness_files are refreshed so ``verify`` stays consistent.
    """
    is_run = hasattr(run_or_dir, "dir") and not isinstance(run_or_dir, (str, Path))
    run_dir = Path(run_or_dir.dir if is_run else run_or_dir)
    mode = verify if verify is not None else (_report_verify_mode(run_or_dir) if is_run else DEFAULT_REPORT_VERIFY)
    t0 = time.time()
    data = collect(run_dir, verify=mode, notes=notes)
    readme = write_text_atomic(run_dir / README_NAME, render_readme(data))
    page = write_text_atomic(run_dir / AUDIT_NAME, render_audit_html(data))
    if is_run:
        rec = getattr(run_or_dir, "_record_harness_stat", None)
        if callable(rec):
            for name in REPORT_FILES:
                rec(name)
    elif update_manifest:
        _refresh_manifest_hashes(run_dir, REPORT_FILES)
    data["render_s"] = round(time.time() - t0, 3)
    return {"readme": readme, "html": page}


def ensure_reports(run_dir: str | Path) -> dict[str, Path]:
    """Write the reports when either is missing; otherwise return the existing paths."""
    d = Path(run_dir)
    if all((d / n).is_file() for n in REPORT_FILES):
        return {"readme": d / README_NAME, "html": d / AUDIT_NAME}
    return write_reports(d)


__all__ = ["collect", "render_readme", "render_audit_html", "write_reports", "ensure_reports", "agent_label",
           "human_bytes", "human_duration", "AUDIT_CSP", "AUDIT_SCRIPT_HASH", "REPORT_FILES", "README_NAME",
           "AUDIT_NAME", "DEFAULT_REPORT_VERIFY"]
