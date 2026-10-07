"""Run directory, trace log, cost ledger and the crash-safe audit record.

Layout (mirrors the upstream Virtual Biotech run layout)::

    runs/<RUN_ID>/
      MANIFEST.json              schema 2: status, query, pinned config, artifact
                                 hashes + attribution, harness-file hashes,
                                 execution record, lifecycle problems
      inputs/query.txt           every turn, as '--- turn N ---' blocks
      inputs/plan.json           latest plan + plan history
      inputs/config.json         the pinned configuration (``set_config``)
      work/<agent>/...           each agent's scripts, data, results (artifacts)
      work/_mcp/data/processed/  MCP tool outputs (attributed to the calling agent)
      logs/trace.jsonl           every model call, tool call, delegation
      logs/cost_report.json      per-agent tokens and USD
      logs/transcript.md         conversation, CSO reasoning, sub-agent traces
      evidence/claims.json       {stats, claims}: claim -> validated evidence
      evidence/artifacts.json    registered (described) artifacts
      evidence/provenance.json   provenance index built from the trace
      report/FINAL_REPORT.md     CSO responses across turns (raw [[claim:ID]] anchors)
      report/FINAL_REPORT.rendered.md  numbered references + claims appendix
      report/plan_reconciliation.json  planned vs actual
      session_report.json        turn records

Crash safety: artifacts are hashed incrementally after every file-writing tool
call (``after_tool``) and the MANIFEST is rewritten atomically then and after
every turn, so a run killed mid-session still carries the hashes of everything
recorded so far and reads ``in_progress`` (never "complete") until ``close()``.
Every audit side effect runs inside a guard that records failures in
``audit_errors`` (and MANIFEST.audit_errors) instead of raising.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .audit import storage as st
from .audit.claims import (
    EvidenceContext,
    claims_payload,
    link_cited_by,
    merge_claims,
    read_claims_file,
    refresh_claims,
    validate_claims,
)
from .audit.plan import reconcile, render_plan_md, validate_plan
from .audit.provenance import (
    ROOT_WORKSPACE_AGENTS,
    build_index,
    data_call_fields,
    extract_returned_paths,
    is_capture_tool,
    parse_bash,
    research_turns,
)
from .audit.render import find_refs, render_report_md
from .providers.base import Usage

log = logging.getLogger(__name__)

MANIFEST_SCHEMA = 2
RUN_STATUSES = ("in_progress", "completed", "incomplete", "interrupted", "empty", "reconstructed")
#: A data provenance record id (``vbt.datalayer.record.prov_id``).
_PROV_ID = re.compile(r"^dp_[0-9a-f]{12}$")

#: In-code defaults for the ``audit`` config section.
AUDIT_DEFAULTS: dict[str, Any] = {
    "snapshot_after_tools": True,   # re-snapshot and hash after each file-writing tool call
    "capture_on_trace": True,       # also trigger capture from the tool_end trace event
    "render_reports": True,         # call vbt.audit.report.write_reports (when installed) each save
    "update_index": True,           # call vbt.audit.index.update_index (when installed) on close
    "reasoning_excerpt_chars": 2000,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def _reason_map(items: Mapping[str, Any] | Iterable[str] | None, reason: str) -> dict[str, str]:
    """``{name: reason}`` from a mapping (values kept) or a list of names (each gets ``reason``)."""
    if not items:
        return {}
    if isinstance(items, Mapping):
        return {str(k): str(v) for k, v in items.items()}
    if isinstance(items, str):
        return {items: reason}
    return {str(k): reason for k in items}


def write_json_atomic(path: Path, data: Any) -> None:
    """Atomic JSON write (kept for backwards compatibility; see ``vbt.audit.storage``)."""
    st.write_json_atomic(path, data)


def _sha256(p: Path) -> str:
    return st.sha256_file(p)


def _kind(p: Path) -> str:
    return st.classify(p)


# ----------------------------------------------------------------- costs


def _usage_from_dict(d: Mapping[str, Any]) -> Usage:
    """Inverse of ``Usage.as_dict`` for every field the installed Usage defines."""
    names = getattr(Usage, "__dataclass_fields__", {})
    kw: dict[str, Any] = {}
    for name in names:
        v = d.get(name)
        if name == "server_tool_requests":
            if isinstance(v, Mapping):
                kw[name] = {str(k): int(n or 0) for k, n in v.items()}
        elif v is not None:
            kw[name] = int(v or 0)
    return Usage(**kw)

@dataclass
class CostLedger:
    by_agent: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    usd_by_agent: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls_by_agent: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    extra_usd: float = 0.0  # tool-side costs that could not be attributed to an agent
    tool_usd_by_agent: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    extra_by_label: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, agent: str, usage: Usage, usd: float) -> None:
        """A model call by ``agent``."""
        self.by_agent[agent] = self.by_agent[agent] + usage
        self.usd_by_agent[agent] += usd
        self.calls_by_agent[agent] += 1

    def add_extra(self, agent: str | None = None, usd: float = 0.0, label: str | None = None) -> None:
        """A tool-side cost (e.g. web search fees). Attributed to ``agent`` without
        counting a model call; kept per ``label`` as well."""
        try:
            usd = float(usd or 0.0)
        except (TypeError, ValueError):
            return
        if not usd:
            return
        self.extra_by_label[label or "other"] += usd
        if agent:
            self.usd_by_agent[agent] += usd
            self.tool_usd_by_agent[agent] += usd
        else:
            self.extra_usd += usd

    @property
    def total_usd(self) -> float:
        return sum(self.usd_by_agent.values()) + self.extra_usd

    def report(self) -> dict[str, Any]:
        agents = sorted(set(self.by_agent) | set(self.usd_by_agent))
        return {
            "total_usd": round(self.total_usd, 6),
            "extra_usd": round(self.extra_usd, 6),
            "tool_usd": round(sum(self.tool_usd_by_agent.values()) + self.extra_usd, 6),
            "extra_by_label": {k: round(v, 6) for k, v in sorted(self.extra_by_label.items())},
            "agents": {
                a: {"usd": round(self.usd_by_agent[a], 6), "model_calls": self.calls_by_agent[a],
                    "tool_usd": round(self.tool_usd_by_agent.get(a, 0.0), 6),
                    **self.by_agent[a].as_dict()}
                for a in agents
            },
        }

    @classmethod
    def from_report(cls, data: Mapping[str, Any] | None) -> "CostLedger":
        ledger = cls()
        if not isinstance(data, Mapping):
            return ledger
        try:
            ledger.extra_usd = float(data.get("extra_usd") or 0.0)
            for label, usd in (data.get("extra_by_label") or {}).items():
                ledger.extra_by_label[label] = float(usd)
            for a, d in (data.get("agents") or {}).items():
                ledger.usd_by_agent[a] = float(d.get("usd") or 0.0)
                ledger.calls_by_agent[a] = int(d.get("model_calls") or 0)
                ledger.tool_usd_by_agent[a] = float(d.get("tool_usd") or 0.0)
                ledger.by_agent[a] = _usage_from_dict(d)
        except (TypeError, ValueError, AttributeError):
            return cls()
        return ledger


# ----------------------------------------------------------------- the run

def _new_manifest(run_id: str, created: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA, "run_id": run_id, "created": created, "started": created,
        "updated": created, "completed": None, "status": "in_progress", "query": "", "agents": [],
        "turns": 0, "cost_usd": 0.0, "config": config, "artifacts": {}, "harness_files": {},
        "execution": [], "research_turns": [], "interrupted_turns": [], "data_source_errors": [],
        "audit_errors": [], "misplaced_files": [], "deleted_artifacts": [], "plan": None,
        "plan_reconciliation": None, "degraded": None,
    }


def _upgrade_manifest(data: Any, run_id: str) -> dict[str, Any]:
    """Load a v1 or v2 MANIFEST into the v2 shape (tolerant)."""
    if not isinstance(data, dict):
        data = {}
    m = _new_manifest(str(data.get("run_id") or run_id), str(data.get("created") or data.get("started") or _now()),
                      data.get("config") if isinstance(data.get("config"), dict) else {})
    for k, v in data.items():
        if k in m and k != "artifacts":
            m[k] = v
    arts = data.get("artifacts") or {}
    if isinstance(arts, dict):
        for rel, e in arts.items():
            if isinstance(e, str):  # v1: {rel: sha256}
                m["artifacts"][rel] = {"path": rel, "sha256": e, "kind": st.classify(rel),
                                       "produced_by": st.work_owner(rel) or "unknown", "registered": False,
                                       "cited_by": []}
            elif isinstance(e, dict):
                m["artifacts"][rel] = dict(e, path=rel)
    m["schema"] = MANIFEST_SCHEMA
    for k in ("interrupted_turns", "data_source_errors", "audit_errors", "misplaced_files", "deleted_artifacts",
              "execution", "research_turns", "agents"):
        if not isinstance(m.get(k), list):
            m[k] = []
    if not isinstance(m.get("harness_files"), dict):
        m["harness_files"] = {}
    degraded = m.get("degraded")
    if isinstance(degraded, dict):  # older records carry {reason, servers, at} only
        m["degraded"] = degraded = dict(degraded)
        for k in ("servers", "tools", "tables"):
            if not isinstance(degraded.get(k), dict):
                degraded[k] = _reason_map(degraded.get(k), str(degraded.get("reason") or ""))
        degraded.setdefault("reason", None)
        degraded.setdefault("at", None)
    elif degraded is not None:
        m["degraded"] = None
    return m


class Run:
    """One research session (interactive or headless) and its audit record."""

    def __init__(self, root: Path, run_id: str | None = None, config: dict[str, Any] | None = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.dir = (Path(root) / self.run_id).resolve()
        for sub in ("inputs", "work/_mcp/data/processed", "logs", "evidence", "report"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self._init_state(config or {})
        self.started = _now()
        self.manifest = _new_manifest(self.run_id, self.started, self.config)
        self._trace = self._open_trace()
        self._snap = st.snapshot_dir(self.dir)
        with self._guard("init manifest"):
            self._write_manifest()

    def _init_state(self, config: dict[str, Any]) -> None:
        self.config = config
        self.audit_settings: dict[str, Any] = {}
        self.cost = CostLedger()
        self.todos: dict[str, list[dict[str, Any]]] = {}
        self.turns: list[dict[str, Any]] = []
        self.audit_errors: list[str] = []
        self.malformed_trace_lines = 0
        self._lock = threading.RLock()
        self._trace_lock = threading.Lock()
        self._calls: dict[str, dict[str, Any]] = {}
        self._captured: set[str] = set()
        self._harness_written: dict[str, tuple[int, int]] = {}
        self._active_turn: int | None = None
        self._closed = False
        self._noting = False

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def open_existing(cls, run_dir: str | Path, *, resume: bool = True) -> "Run":
        """Reopen a run directory (e.g. to resume a session). Tolerant of v1 records.

        With ``resume=True`` the run is marked ``in_progress`` again (an
        ``interrupted`` status is sticky) and new trace events are appended.
        """
        d = Path(run_dir).expanduser().resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"no run directory at {d}")
        self = cls.__new__(cls)
        raw = st.read_json(d / "MANIFEST.json", {})
        manifest = _upgrade_manifest(raw, d.name)
        self.run_id = manifest["run_id"] or d.name
        self.dir = d
        self._init_state(manifest.get("config") or {})
        self.started = manifest.get("created") or _now()
        self.manifest = manifest
        self.audit_errors = [str(e) for e in manifest.get("audit_errors") or []]
        report = st.read_json(d / "session_report.json", {})
        turns = report.get("turns") if isinstance(report, dict) else None
        self.turns = [t for t in turns or [] if isinstance(t, dict)]
        self.cost = CostLedger.from_report(st.read_json(d / "logs" / "cost_report.json", None))
        for sub in ("inputs", "logs", "evidence", "report"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        events = self.events()
        for ev in events:
            self._index_event(ev, live=False)
        self._active_turn = None
        self._trace = self._open_trace()
        self._snap = st.snapshot_dir(self.dir)
        if resume:
            if manifest.get("status") != "interrupted":
                manifest["status"] = "in_progress"
            manifest["completed"] = None
            with self._guard("reopen manifest"):
                self._write_manifest()
        return self

    def _open_trace(self):
        path = self.dir / "logs" / "trace.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.exists() and path.stat().st_size:
                with open(path, "rb") as f:
                    f.seek(-1, os.SEEK_END)
                    needs_newline = f.read(1) != b"\n"
                if needs_newline:  # a crash left a partial line; do not glue the next event to it
                    with open(path, "a") as f:
                        f.write("\n")
        except OSError:
            pass
        return open(path, "a", buffering=1, encoding="utf-8")

    def set_config(self, config: Mapping[str, Any]) -> None:
        """Pin the run configuration: rewrites MANIFEST.config and inputs/config.json."""
        with self._guard("set_config"), self._lock:
            self.config = dict(config or {})
            self.manifest["config"] = self.config
            self._write_harness_json("inputs/config.json", self.config)
            self._write_manifest()

    @property
    def current_turn(self) -> int:
        """Number of the turn in progress (or the next one, between turns)."""
        return self._active_turn if self._active_turn is not None else len(self.turns) + 1

    @property
    def status(self) -> str:
        return str(self.manifest.get("status") or "in_progress")

    def _audit_cfg(self, key: str, default: Any = None) -> Any:
        for source in (self.audit_settings, (self.config or {}).get("audit") if isinstance(self.config, dict) else None):
            if isinstance(source, Mapping) and key in source:
                return source[key]
        return AUDIT_DEFAULTS.get(key, default) if default is None else default

    # ------------------------------------------------------------------ guard

    @contextmanager
    def _guard(self, label: str) -> Iterator[None]:
        """Run an audit side effect; record any failure instead of raising."""
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - audit recording must never break a run
            self.note_audit_error(f"{label}: {type(exc).__name__}: {exc}")

    def note_audit_error(self, msg: str) -> None:
        """Record an audit-capture failure in ``audit_errors`` and MANIFEST.audit_errors."""
        msg = str(msg)[:2000]
        log.warning("[audit] %s", msg)
        if msg not in self.audit_errors:
            self.audit_errors.append(msg)
        if self._noting:
            return
        self._noting = True
        try:
            self.manifest["audit_errors"] = list(self.audit_errors)
            self._write_manifest()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._noting = False

    def mark_degraded(self, servers: Mapping[str, Any] | Iterable[str] | None, reason: str = "missing reference data",
                      *, tools: Mapping[str, Any] | Iterable[str] | None = None,
                      tables: Mapping[str, Any] | Iterable[str] | None = None) -> None:
        """Record that the run proceeds without some reference data (``--allow-missing-data``):
        MANIFEST.degraded = {servers, tools, tables, reason, at}. ``servers`` lists servers whose
        every tool is unready (kept for compatibility); ``tools`` and ``tables`` name the unready
        tools and tables of the data layer's tool-scoped readiness, each with its reason."""
        with self._guard("mark_degraded"), self._lock:
            servers_ = _reason_map(servers, reason)
            tools_ = _reason_map(tools, reason)
            tables_ = _reason_map(tables, reason)
            self.manifest["degraded"] = {"servers": servers_, "tools": tools_, "tables": tables_, "reason": reason,
                                         "at": _now()}
            self.trace("run_degraded", reason=reason, servers=servers_, tools=tools_, tables=tables_)
            self._write_manifest()

    # ------------------------------------------------------------------ paths

    def agent_dir(self, agent: str) -> Path:
        d = self.dir / "work" / agent
        for sub in ("code/scripts", "data/raw", "data/processed", "results/figures",
                    "results/tables", "results/reports"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    @property
    def mcp_output_dir(self) -> Path:
        return self.dir / "work" / "_mcp" / "data" / "processed"

    def rel(self, path: str | Path) -> str:
        p = Path(path).resolve()
        try:
            return p.relative_to(self.dir).as_posix()
        except ValueError:
            return str(p)

    def _workspace_for(self, agent: str | None) -> Path:
        if not agent or agent in ROOT_WORKSPACE_AGENTS:
            return self.dir
        return self.dir / "work" / agent

    # ------------------------------------------------------------------ trace

    def trace(self, type: str, **data: Any) -> None:
        event = {"ts": _now(), "t": round(time.time(), 3), "type": type, **data}
        try:
            line = json.dumps(event, default=str)
        except (TypeError, ValueError) as exc:
            line = json.dumps({"ts": event["ts"], "t": event["t"], "type": type,
                               "error": f"unserialisable event: {exc}"})
            self.note_audit_error(f"trace({type}): unserialisable event: {exc}")
        try:
            with self._trace_lock:
                if self._trace.closed:
                    with open(self.dir / "logs" / "trace.jsonl", "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                else:
                    self._trace.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 - ``type`` is the event type here, not the builtin
            self.note_audit_error(f"trace({type}): {exc.__class__.__name__}: {exc}")
        with self._guard(f"index {type}"):
            self._index_event(event)
        if type in ("tool_end", "tool_error") and self._audit_cfg("capture_on_trace", True):
            self._capture(event)

    def _index_event(self, ev: Mapping[str, Any], live: bool = True) -> None:
        t = ev.get("type")
        if t == "tool_start" and ev.get("tool_use_id"):
            tuid = str(ev["tool_use_id"])
            tool = ev.get("tool") or ev.get("tool_name")
            rec = {"tool": tool, "agent": ev.get("agent"), "agent_run_id": ev.get("agent_run_id"),
                   "t": ev.get("t"), "started_at": ev.get("ts"), "pending": True, "is_error": False}
            if live:
                # Unrounded clock reading taken before the tool runs: any file it writes is newer.
                rec["t_precise"] = time.time()
                if is_capture_tool(tool) and self._audit_cfg("snapshot_after_tools", True):
                    # State of everything outside work/ before the call, so a change made
                    # during the call is told apart from earlier harness writes exactly.
                    rec["outside_snap"] = st.snapshot_dir(self.dir, exclude_top={"logs", "work", "memory"})
            if tool == "Bash":
                inp = ev.get("input")
                cmd = inp.get("command") if isinstance(inp, Mapping) else None
                if cmd is None and isinstance(ev.get("input_preview"), Mapping):
                    cmd = ev["input_preview"].get("command")
                rec["command"] = str(cmd)[:4000] if cmd else None
            with self._lock:
                self._calls[tuid] = rec
        elif t in ("tool_end", "tool_error") and ev.get("tool_use_id"):
            tuid = str(ev["tool_use_id"])
            with self._lock:
                rec = self._calls.setdefault(tuid, {"tool": ev.get("tool") or ev.get("tool_name"),
                                                    "agent": ev.get("agent"), "t": None, "started_at": None})
                rec["pending"] = False
                rec["is_error"] = bool(ev.get("is_error")) or t == "tool_error"
                rec["end_t"] = ev.get("t")
                # result_status, error_kind, prov, coverage and the provenance summary (§15.2),
                # exactly as audit.provenance indexes them, so record_claims and verify agree.
                rec.update(data_call_fields(ev))
                if live:
                    rec["end_precise"] = time.time()
        elif t == "turn_start":
            try:
                self._active_turn = int(ev.get("turn"))
            except (TypeError, ValueError):
                self._active_turn = None
            if live and not self.manifest.get("query") and ev.get("prompt"):
                self.manifest["query"] = str(ev.get("prompt"))[:2000]
                with self._guard("manifest query"):
                    self._write_manifest()

    def events(self) -> list[dict[str, Any]]:
        """Trace events; malformed or partial lines are skipped and counted
        (``malformed_trace_lines``)."""
        try:
            with self._trace_lock:
                if getattr(self, "_trace", None) is not None and not self._trace.closed:
                    self._trace.flush()
        except (OSError, ValueError, AttributeError):
            pass
        events, bad = st.read_jsonl(self.dir / "logs" / "trace.jsonl")
        self.malformed_trace_lines = bad
        return events

    def tool_call_status(self, tool_use_id: str) -> dict[str, Any] | None:
        rec = self._calls.get(str(tool_use_id))
        return dict(rec) if rec else None

    # ------------------------------------------------------------------ harness writes

    def _record_harness_stat(self, rel: str) -> None:
        try:
            s = (self.dir / rel).stat()
            self._harness_written[rel] = (s.st_mtime_ns, s.st_size)
        except OSError:
            pass
        # A harness record rewritten between saves (plan, claims, artifacts, config) gets its
        # recorded hash refreshed now, so a run killed before the next save still verifies
        # (the next MANIFEST write carries the new hash) instead of reporting tampering.
        manifest = getattr(self, "manifest", None)
        if isinstance(manifest, dict) and st.is_harness_hashed_rel(rel) and not st.is_ignored_rel(rel):
            try:
                digest = st.sha256_file(self.dir / rel)
            except OSError:
                return
            with self._lock:
                hashes = manifest.get("harness_files")
                if not isinstance(hashes, dict):
                    hashes = manifest["harness_files"] = {}
                hashes[rel] = digest

    def _write_harness_text(self, rel: str, text: str) -> None:
        st.write_text_atomic(self.dir / rel, text)
        self._record_harness_stat(rel)

    def _write_harness_json(self, rel: str, data: Any, *, compact: bool = False) -> None:
        st.write_json_atomic(self.dir / rel, data, compact=compact)
        self._record_harness_stat(rel)

    def _write_manifest(self, *, compact: bool | None = None) -> None:
        """Atomically rewrite MANIFEST.json. Mid-turn rewrites of a large manifest are
        compact (C JSON encoder); turn and close saves are indented."""
        with self._lock:
            m = self.manifest
            m["updated"] = _now()
            m["turns"] = len(self.turns)
            m["cost_usd"] = round(self.cost.total_usd, 6)
            m["audit_errors"] = list(self.audit_errors)
            if compact is None:
                compact = len(m.get("artifacts") or {}) > 200
            with st.run_lock(self.dir):
                self._write_harness_json("MANIFEST.json", m, compact=compact)

    # ------------------------------------------------------------------ capture

    def after_tool(self, event: Mapping[str, Any]) -> None:
        """Incremental capture after a tool call (called by the runtime on tool_end).

        Diffs a cheap (mtime, size) snapshot of the run directory and records
        changed files with sha256 and attribution. Idempotent per tool_use_id;
        never raises.
        """
        self._capture(event)

    def _capture(self, event: Mapping[str, Any]) -> None:
        with self._guard("after_tool"):
            if not isinstance(event, Mapping):
                return
            tool = str(event.get("tool") or event.get("tool_name") or "")
            if not is_capture_tool(tool) or not self._audit_cfg("snapshot_after_tools", True):
                return
            tuid = str(event.get("tool_use_id") or "") or None
            with self._lock:
                if tuid:
                    if tuid in self._captured:
                        return
                    self._captured.add(tuid)
                after = st.snapshot_dir(self.dir)
                changed, deleted = st.diff_snapshots(self._snap, after)
                self._snap = after
                if not changed and not deleted:
                    return
                agent = str(event.get("agent") or "cso")
                call = self._calls.get(tuid or "", {})
                if call.get("t_precise") is not None:
                    # Precise clock readings bracket the call; file mtimes come from a coarser
                    # clock that can only lag, so no slack is needed at the start.
                    window = (float(call["t_precise"]), float(call.get("end_precise") or time.time()) + 0.01)
                else:
                    end_t = _f(event.get("t")) or _f(call.get("end_t")) or time.time()
                    start_t = _f(call.get("t"))
                    if start_t is None:
                        start_t = end_t - (_f(event.get("duration_s")) or 0.0)
                    window = (start_t - 0.002, end_t + 0.05)
                start_snap = call.pop("outside_snap", None) if call else None
                returned = self._returned_paths(event, tool)
                created_by = self._created_by(tool, call)
                dirty = False
                for rel in changed:
                    stat = after[rel]
                    if self._harness_written.get(rel) == stat:
                        continue
                    if st.is_work_rel(rel):
                        self._upsert_artifact(rel, stat, agent=agent, tool=tool, tool_use_id=tuid,
                                              returned=returned, created_by=created_by, window=window)
                        dirty = True
                    else:
                        if start_snap is not None:
                            during = start_snap.get(rel) != stat
                        else:
                            during = window[0] <= stat[0] / 1e9 <= window[1]
                        dirty |= self._note_outside_work(rel, stat, agent, tool, tuid, during)
                for rel in deleted:
                    if st.is_work_rel(rel):
                        dirty |= self._note_deleted(rel, agent=agent, tool=tool, tool_use_id=tuid)
                    elif st.is_harness_rel(rel) and st.rel_parts(rel)[0] not in ("logs", "memory") and (
                            start_snap is None or rel in start_snap):
                        self._add_misplaced({"path": rel, "reason": "harness_file_deleted", "agent": agent,
                                             "tool": tool, "tool_use_id": tuid, "bytes": 0,
                                             "detected_at": _now(), "turn": self.current_turn})
                        dirty = True
                if dirty:
                    self._write_manifest()

    def _returned_paths(self, event: Mapping[str, Any], tool: str) -> set[str]:
        if not tool.startswith("mcp__") or tool.startswith("mcp__provenance__"):
            return set()
        out: set[str] = set()
        for p in event.get("files_returned") or []:
            rel = st.to_rel(p, self.dir) if os.path.isabs(str(p)) else str(p)
            if rel:
                out.add(rel)
        texts = [event.get("output")] if isinstance(event.get("output"), str) else []
        spills = []
        if event.get("output_path"):
            spills.append(str(event["output_path"]))
        for text in texts:
            for rel in extract_returned_paths(tool, text, self.dir):
                out.add(rel)
                if st.rel_parts(rel)[:1] in (("logs",),) or "/_tool_outputs/" in f"/{rel}":
                    spills.append(rel)
        for sp in spills:
            p = Path(sp) if os.path.isabs(sp) else self.dir / sp
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    out.update(extract_returned_paths(tool, f.read(5_000_000), self.dir))
            except OSError:
                continue
        return out

    @staticmethod
    def _created_by(tool: str, call: Mapping[str, Any]) -> str:
        if tool == "Bash" and call.get("command"):
            scripts = parse_bash(str(call["command"]))["scripts"]
            names = list(dict.fromkeys(Path(s[2]).name for s in scripts))
            if len(names) == 1:
                return names[0]
        return tool

    def _concurrent_writer(self, tool_use_id: str | None, mtime: float) -> bool:
        """True when another file-writing call was running at ``mtime`` (parallel tool calls),
        so a file changed then cannot be credited to ``tool_use_id`` exactly."""
        with self._lock:
            calls = list(self._calls.items())
        for tuid, rec in calls:
            if tuid == tool_use_id or not is_capture_tool(rec.get("tool")):
                continue
            start = _f(rec.get("t_precise"))
            if start is None:
                start = _f(rec.get("t"))
            if start is None or mtime < start:
                continue
            if rec.get("pending"):
                if rec.get("t_precise") is not None:  # a live call still running
                    return True
                continue
            end = _f(rec.get("end_precise"))
            if end is None:
                end = _f(rec.get("end_t"))
            if end is not None and mtime <= end:
                return True
        return False

    def _upsert_artifact(self, rel: str, stat: tuple[int, int] | None = None, *, agent: str | None = None,
                         tool: str | None = None, tool_use_id: str | None = None,
                         returned: set[str] | frozenset = frozenset(), created_by: str | None = None,
                         window: tuple[float, float] | None = None, sha: str | None = None) -> dict | None:
        path = self.dir / rel
        try:
            if stat is None:
                s = path.stat()
                stat = (s.st_mtime_ns, s.st_size)
            sha = sha or st.sha256_file(path)
        except OSError:
            return None
        arts = self.manifest["artifacts"]
        prev = arts.get(rel) or {}
        owner = st.work_owner(rel)
        attributed = False
        if owner is None or owner in st.HARNESS_WORK_DIRS:
            stem = Path(rel).stem
            if tool_use_id and agent and (rel in returned or (owner == "_tool_outputs" and stem == tool_use_id)):
                produced_by, attributed = agent, True
            elif owner == "_tool_outputs" and stem in self._calls:
                # A spilled tool output is named after the call that produced it.
                produced_by = self._calls[stem].get("agent") or owner
                tool_use_id, tool, attributed = stem, self._calls[stem].get("tool"), True
                created_by = tool
            elif owner is None and tool_use_id and agent and window and window[0] <= stat[0] / 1e9 <= window[1] \
                    and not self._concurrent_writer(tool_use_id, stat[0] / 1e9):
                produced_by, attributed = agent, True
            else:
                produced_by = prev.get("produced_by") or owner or "unknown"
        else:
            produced_by = owner
            # Exact only when the file changed during this call and no other concurrent
            # call could have written it; otherwise provenance attribution decides later.
            attributed = bool(tool_use_id) and agent == owner and (
                window is None or (window[0] <= stat[0] / 1e9 <= window[1]
                                   and not self._concurrent_writer(tool_use_id, stat[0] / 1e9)))
        mtime = stat[0] / 1e9
        changed = sha != prev.get("sha256")
        entry = {
            "path": rel,
            "kind": prev.get("kind") or st.classify(rel),
            "bytes": stat[1],
            "sha256": sha,
            "produced_by": produced_by,
            "registered_by": prev.get("registered_by"),
            "tool_use_id": tool_use_id if attributed else prev.get("tool_use_id"),
            "created_by": (created_by or tool) if attributed else prev.get("created_by"),
            "produced_at": prev.get("produced_at") or _iso(mtime),
            "modified_at": _iso(mtime),
            "mtime_ns": stat[0],
            "description": prev.get("description"),
            "registered": bool(prev.get("registered")),
            "cited_by": list(prev.get("cited_by") or []),
            "turn": prev.get("turn") or self.current_turn,
            "modified_turn": self.current_turn if (changed or not prev) else prev.get("modified_turn"),
        }
        for k in ("registered_at", "attribution"):
            if prev.get(k) is not None:
                entry[k] = prev[k]
        arts[rel] = entry
        if produced_by and not str(produced_by).startswith("_") and produced_by not in self.manifest["agents"] \
                and produced_by != "unknown":
            self.manifest["agents"].append(produced_by)
        return entry

    def _add_misplaced(self, entry: dict[str, Any]) -> None:
        lst = self.manifest.setdefault("misplaced_files", [])
        for i, e in enumerate(lst):
            if isinstance(e, dict) and e.get("path") == entry["path"]:
                merged = dict(e)
                for k, v in entry.items():
                    if v is not None or k not in merged:
                        merged[k] = v
                lst[i] = merged
                return
            if e == entry["path"]:
                lst[i] = entry
                return
        lst.append(entry)

    def _note_outside_work(self, rel: str, stat: tuple[int, int], agent: str | None, tool: str | None,
                           tuid: str | None, in_window: bool) -> bool:
        """Record a file outside work/ that changed; ``in_window``: it changed during this call."""
        parts = st.rel_parts(rel)
        if not parts or parts[0] in ("logs", "memory"):
            return False
        if st.is_harness_rel(rel):
            if not in_window:
                return False  # written by harness code, not during this call
            reason = "harness_file_modified"
        else:
            reason = "outside_work"
        self._add_misplaced({
            "path": rel, "reason": reason, "agent": agent if in_window else None,
            "tool": tool if in_window else None, "tool_use_id": tuid if in_window else None,
            "bytes": stat[1], "detected_at": _now(), "turn": self.current_turn,
        })
        return True

    def _note_deleted(self, rel: str, *, agent: str | None = None, tool: str | None = None,
                      tool_use_id: str | None = None) -> bool:
        arts = self.manifest["artifacts"]
        if rel not in arts:
            return False
        e = arts.pop(rel)
        self.manifest.setdefault("deleted_artifacts", []).append({
            "path": rel, "sha256": e.get("sha256"), "produced_by": e.get("produced_by"),
            "cited_by": e.get("cited_by") or [], "deleted_at": _now(), "turn": self.current_turn,
            "agent": agent, "tool": tool, "tool_use_id": tool_use_id})
        return True

    def _scan_work(self) -> bool:
        """Record new/changed work/ files and drop deleted ones (no attribution)."""
        snap = st.snapshot_dir(self.dir)
        dirty = False
        arts = self.manifest["artifacts"]
        for rel, stat in snap.items():
            if not st.is_work_rel(rel):
                continue
            prev = arts.get(rel)
            if prev and prev.get("sha256") and prev.get("bytes") == stat[1] and prev.get("mtime_ns") == stat[0]:
                continue
            if self._upsert_artifact(rel, stat) is not None:
                dirty = True
        for rel in [r for r in arts if r not in snap]:
            dirty |= self._note_deleted(rel)
        return dirty

    def _rescan(self) -> None:
        """Full rescan: work/ artifacts, misplaced files, harness-file hashes."""
        self._scan_work()
        snap = st.snapshot_dir(self.dir)
        known = {e.get("path") if isinstance(e, dict) else e for e in self.manifest.get("misplaced_files") or []}
        for rel, stat in snap.items():
            if st.is_work_rel(rel) or st.is_harness_rel(rel) or rel in known:
                continue
            parts = st.rel_parts(rel)
            if parts and parts[0] in ("logs", "memory"):
                continue
            self._add_misplaced({"path": rel, "reason": "outside_work", "agent": None, "tool": None,
                                 "tool_use_id": None, "bytes": stat[1], "detected_at": _now(),
                                 "turn": self.current_turn})
        self._snap = snap

    def _hash_harness_files(self) -> None:
        # The harness never deletes its records, so a recorded file that vanished stays
        # listed (verify reports it missing) unless a later save regenerates it.
        hashes = dict(self.manifest.get("harness_files") or {})
        for sub in st.HARNESS_HASH_DIRS:
            base = self.dir / sub
            if not base.is_dir():
                continue
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.dir).as_posix()
                if st.is_ignored_rel(rel):
                    continue
                try:
                    hashes[rel] = st.sha256_file(p)
                except OSError:
                    continue
        for name in st.HARNESS_ROOT_HASHED:
            p = self.dir / name
            if p.is_file():
                try:
                    hashes[name] = st.sha256_file(p)
                except OSError:
                    pass
        self.manifest["harness_files"] = dict(sorted(hashes.items()))

    def refresh_harness_hashes(self) -> None:
        """Re-hash harness records and rewrite the MANIFEST (for code that writes reports later)."""
        with self._guard("refresh_harness_hashes"), self._lock:
            self._hash_harness_files()
            self._write_manifest()
            self._snap = st.snapshot_dir(self.dir)

    # ------------------------------------------------------------------ evidence

    def _load(self, name: str, default):
        return st.read_json(self.dir / "evidence" / name, default)

    def _derived_from(self, items: Any) -> tuple[list[dict[str, Any]] | None, list[str]]:
        """``derived_from`` entries (data provenance ids ``dp_...`` or tool_use ids) as records, each with
        whether this run holds its provenance record; ``(None, [error])`` for a malformed list."""
        if items is None:
            return [], []
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, (list, tuple)):
            return None, ["'derived_from' is a list of data provenance ids (dp_...) or tool_use ids"]
        out: list[dict[str, Any]] = []
        from .datalayer.replay import provenance_dirs

        prov_dirs = provenance_dirs(self.dir)            # data.provenance.dir as pinned, then the default
        for raw in items:
            ident = str(raw or "").strip()
            if not ident or len(ident) > 128 or any(c.isspace() or c in "/\\" for c in ident):
                return None, [f"'derived_from' entry {raw!r} is not a provenance or tool_use id"]
            if _PROV_ID.match(ident):
                found = any((d / "client" / f"{ident}.json").is_file() or any(
                    ident in p.read_text(encoding="utf-8", errors="replace")[:400]
                    for p in d.glob("*.json") if p.is_file()) for d in prov_dirs)
                out.append({"id": ident, "kind": "data_provenance", "found": found})
            else:
                out.append({"id": ident, "kind": "tool_use",
                            "found": any((d / f"{ident}.json").is_file() for d in prov_dirs)})
        return out, []

    def register_artifact(self, path: str, description: str, agent: str, kind: str | None = None,
                          workspace: str | Path | None = None, derived_from: Any = None) -> dict[str, Any]:
        """Describe a file under work/ as citable evidence. Returns ok:false payloads.

        ``derived_from`` names the inputs the file was built from: data provenance ids (``dp_...``, as
        returned by ``vbt.datalayer.client``) or tool_use ids of data tool calls; each is recorded with
        whether this run holds its provenance record (unknown ids are recorded and warned about)."""
        lineage, problems = self._derived_from(derived_from)
        if lineage is None:
            return {"ok": False, "errors": problems}
        try:
            raw = os.path.expanduser(str(path or "").strip())
            if not raw:
                return {"ok": False, "errors": ["'path' is required"]}
            cands: list[str] = []
            if os.path.isabs(raw):
                cands.append(os.path.normpath(raw))
            else:
                if workspace:
                    cands.append(os.path.normpath(str(Path(workspace) / raw)))
                cands.append(os.path.normpath(str(self.dir / raw)))
            target, inside = None, False
            for c in cands:
                rel = st.to_rel(c, self.dir)
                if rel is None or rel == "":
                    continue
                inside = True
                if (self.dir / rel).is_file():
                    target = rel
                    break
            if target is None:
                if not inside:
                    return {"ok": False, "errors": [f"Path is outside the run directory: {path}. Only files "
                                                    "within this run can be registered as evidence."]}
                return {"ok": False, "errors": [f"No such file: {path}. Write the file before registering it."]}
            real = (self.dir / target).resolve()
            if real != self.dir and self.dir not in real.parents:
                return {"ok": False, "errors": [f"{target} resolves outside the run directory"]}
            if not st.is_work_rel(target):
                first = st.rel_parts(target)[0]
                why = ("is a harness record" if first in st.RESERVED_DIRS or st.is_harness_rel(target)
                       else "is not under work/")
                return {"ok": False, "errors": [f"{target} {why}; only analysis outputs under work/<agent>/ "
                                                "can be registered."]}
            if st.is_ignored_rel(target):
                return {"ok": False, "errors": [f"{target} is a temporary or hidden file and cannot be registered"]}
            with self._lock:
                entry = self._upsert_artifact(target)
                if entry is None:
                    return {"ok": False, "errors": [f"Could not read {target}"]}
                entry["description"] = str(description or "")[:500]
                entry["registered"] = True
                entry["registered_by"] = agent
                entry["registered_at"] = _now()
                if kind:
                    entry["kind"] = str(kind).strip().lower()[:32]
                if lineage:
                    entry["derived_from"] = lineage
                with self._guard("register_artifact records"):
                    self._write_registered_list()
                    self._write_manifest()
            out: dict[str, Any] = {"ok": True, "artifact": {k: entry.get(k) for k in (
                "path", "kind", "bytes", "sha256", "produced_by", "registered_by", "description",
                "tool_use_id", "created_by", "derived_from") if k != "derived_from" or lineage}}
            missing = [d["id"] for d in lineage if not d["found"]]
            if missing:
                out["warnings"] = [f"derived_from: no provenance record of {', '.join(missing)} in this run"]
            return out
        except Exception as exc:  # noqa: BLE001
            self.note_audit_error(f"register_artifact: {type(exc).__name__}: {exc}")
            return {"ok": False, "errors": [f"internal error while registering {path}: {exc}"]}

    def _write_registered_list(self) -> None:
        rows = [dict(e, agent=e.get("produced_by")) for e in self.manifest["artifacts"].values() if e.get("registered")]
        rows.sort(key=lambda e: (str(e.get("registered_at") or ""), e["path"]))
        keep = ("path", "description", "agent", "produced_by", "registered_by", "kind", "sha256", "bytes",
                "registered_at", "tool_use_id", "created_by")
        self._write_harness_json("evidence/artifacts.json", [{k: r.get(k) for k in keep} for r in rows])

    def list_artifacts(self, agent: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        """Registered artifacts merged with unregistered work/ files (never logs/)."""
        with self._lock:
            with self._guard("list_artifacts scan"):
                if self._scan_work():
                    self._write_manifest()
            rows = []
            for e in self.manifest["artifacts"].values():
                if agent and e.get("produced_by") != agent and st.work_owner(e["path"]) != agent:
                    continue
                if kind and e.get("kind") != kind:
                    continue
                rows.append({
                    "path": e["path"], "kind": e.get("kind"), "bytes": e.get("bytes"),
                    "produced_by": e.get("produced_by"), "created_by": e.get("created_by"),
                    "tool_use_id": e.get("tool_use_id"), "description": e.get("description") or
                    ("(unregistered)" if not e.get("registered") else ""),
                    "registered": bool(e.get("registered")), "cited_by": list(e.get("cited_by") or []),
                })
        rows.sort(key=lambda r: (str(r.get("produced_by")), str(r.get("kind")), r["path"]))
        return rows

    def write_plan(self, goal: str, steps: Any, *, roster: Any = None, agent: str | None = None) -> dict[str, Any]:
        """Validate and record the analysis plan. Keeps the plan history."""
        try:
            res = validate_plan(steps, goal=goal or "", roster=roster)
            if not res.ok:
                return res.as_dict()
            with self._lock:
                doc = st.read_json(self.dir / "inputs" / "plan.json", {}) or {}
                history = [h for h in (doc.get("history") or []) if isinstance(h, dict)] if isinstance(doc, dict) else []
                plan = dict(res.plan or {})
                plan.update(written=_now(), written_t=round(time.time(), 3), turn=self.current_turn,
                            written_by=agent, version=len(history) + 1)
                self._write_harness_json("inputs/plan.json", {**plan, "history": history + [plan]})
                self.manifest["plan"] = {"version": plan["version"], "written": plan["written"],
                                         "turn": plan["turn"], "n_steps": len(plan.get("steps") or []),
                                         "goal": plan.get("goal", "")}
                with self._guard("write_plan manifest"):
                    self._write_manifest()
            out = res.as_dict()
            out["version"] = plan["version"]
            return out
        except Exception as exc:  # noqa: BLE001
            self.note_audit_error(f"write_plan: {type(exc).__name__}: {exc}")
            return {"ok": False, "errors": [f"internal error while recording the plan: {exc}"], "warnings": []}

    def _evidence_context(self, agent: str | None = None) -> EvidenceContext:
        return EvidenceContext(self.dir, self.manifest["artifacts"], self._calls,
                               workspace=self._workspace_for(agent) if agent else None)

    def record_claims(self, claims: Any, *, agent: str = "cso", turn: int | None = None,
                      strict: bool = True) -> dict[str, Any]:
        """Validate and file claim-evidence objects (port of upstream validate_claims)."""
        try:
            with self._lock:
                with self._guard("record_claims scan"):
                    self._scan_work()
                res = validate_claims(claims, self._evidence_context(agent), strict=strict,
                                      turn=turn if turn is not None else self.current_turn, filed_by=agent)
                if not res.ok:
                    return {"ok": False, "errors": res.errors, "warnings": res.warnings,
                            "hint": "Every artifact path must match a file under work/ in this run "
                                    "(call list_artifacts for exact paths), every tool_use_id must be a "
                                    "finished, successful call in this run's trace, and citations need a "
                                    "pmid, doi or url. Fix the listed problems and call again."}
                filed = _now()
                for c in res.claims:
                    c["filed"] = filed
                existing, err = read_claims_file(self.dir / "evidence" / "claims.json")
                if err:
                    self.note_audit_error(f"record_claims: existing evidence/claims.json unreadable ({err})")
                merged = merge_claims(existing, res.claims)
                link_cited_by(self.manifest["artifacts"], merged)
                self._write_harness_json("evidence/claims.json", claims_payload(merged))
                with self._guard("record_claims manifest"):
                    self._write_manifest()
            return {"ok": True, "recorded": len(res.claims), "total_claims": len(merged),
                    "claims": [{"id": c["id"], "n_evidence": len(c["evidence"]), "n_verified": c["n_verified"],
                                "turn": c.get("turn")} for c in res.claims],
                    "warnings": res.warnings}
        except Exception as exc:  # noqa: BLE001
            self.note_audit_error(f"record_claims: {type(exc).__name__}: {exc}")
            return {"ok": False, "errors": [f"internal error while recording claims: {exc}"]}

    # ------------------------------------------------------------------ turns and reports

    def finish_turn(self, turn: Mapping[str, Any]) -> None:
        """Record a finished (or interrupted/failed) turn and save the audit record."""
        with self._lock:
            with self._guard("finish_turn record"):
                rec = dict(turn or {})
                try:
                    n = int(rec.get("turn") or 0)
                except (TypeError, ValueError):
                    n = 0
                rec["turn"] = n if n > 0 else len(self.turns) + 1
                rec.setdefault("prompt", "")
                rec.setdefault("response", "")
                rec.setdefault("status", "completed")
                rec.setdefault("agents", [])
                for i, t in enumerate(self.turns):
                    if t.get("turn") == rec["turn"]:
                        self.turns[i] = rec
                        break
                else:
                    self.turns.append(rec)
                if not self.manifest.get("query") and self.turns:
                    self.manifest["query"] = str(self.turns[0].get("prompt") or "")[:2000]
                self._record_lifecycle(rec)
                # Files discovered while saving belong to the turn being finished.
                self._active_turn = rec["turn"]
            try:
                self._save(final=False)
            finally:
                self._active_turn = None

    def _record_lifecycle(self, rec: dict[str, Any]) -> None:
        n = rec["turn"]
        status = str(rec.get("status") or "")
        if status == "interrupted":
            lst = self.manifest.setdefault("interrupted_turns", [])
            if n not in lst:
                lst.append(n)
            self.manifest["status"] = "interrupted"
        errors = [e for e in self.manifest.get("data_source_errors") or []
                  if not (isinstance(e, dict) and e.get("turn") == n)]
        for f in rec.get("data_source_failures") or []:
            entry = dict(f) if isinstance(f, Mapping) else {"error": str(f)}
            entry.setdefault("tool_name", entry.get("tool"))
            entry["turn"] = n
            if isinstance(entry.get("error"), str):
                entry["error"] = entry["error"][:2000]
            errors.append(entry)
        self.manifest["data_source_errors"] = errors

    def _save(self, *, final: bool) -> None:
        """Rescan, rebuild provenance, refresh claims, write every report and the MANIFEST."""
        with self._lock:
            with self._guard("rescan"):
                self._rescan()
            prov = None
            with self._guard("provenance index"):
                events = self.events()
                prov = build_index(events, self.dir, malformed_lines=self.malformed_trace_lines)
            attribution: dict[str, Any] = {}
            execution: list[dict[str, Any]] = []
            if prov is not None:
                with self._guard("attribution"):
                    attribution = self._attribute(prov)
                with self._guard("execution record"):
                    execution = prov.execution()
                    self.manifest["execution"] = [{k: e.get(k) for k in (
                        "agent", "agent_run_id", "tool_use_id", "description", "start", "end", "duration_s",
                        "status", "turn", "orientation")} for e in execution]
                    for e in execution:
                        if e["agent"] not in self.manifest["agents"]:
                            self.manifest["agents"].append(e["agent"])
                with self._guard("research turns"):
                    research = research_turns(prov, self.manifest["artifacts"], self.turns)
                    self.manifest["research_turns"] = sorted(research)
                    for t in self.turns:
                        reasons = research.get(t.get("turn"))
                        if reasons or "research" not in t:
                            t["research"] = bool(reasons)
                            t["research_reasons"] = reasons or []
                with self._guard("provenance.json"):
                    self._write_harness_json("evidence/provenance.json", prov.to_dict(attribution))
                with self._guard("plan reconciliation"):
                    self._reconcile_plan(execution)
            claims: list[dict[str, Any]] = []
            with self._guard("claims refresh"):
                claims = self._refresh_claims()
            with self._guard("turn claim summary"):
                self._summarise_turn_claims(claims)
            with self._guard("reports"):
                self._write_reports(claims)
            with self._guard("manifest"):
                if final:
                    self.manifest["status"] = self._final_status()
                    self.manifest["completed"] = _now()
                elif self.manifest.get("status") not in ("interrupted",):
                    self.manifest["status"] = "in_progress"
                self._hash_harness_files()
                self._write_manifest(compact=False)
            if self._audit_cfg("render_reports", True):
                wrote = False
                with self._guard("write_reports"):
                    wrote = self._external_reports()
                if wrote:
                    with self._guard("manifest (reports)"):
                        self._hash_harness_files()
                        self._write_manifest(compact=False)
            self._snap = st.snapshot_dir(self.dir)

    def _final_status(self) -> str:
        if not self.turns:
            return "empty"
        if self.manifest.get("status") == "interrupted" or self.manifest.get("interrupted_turns"):
            return "interrupted"
        if any(str(t.get("status") or "completed") != "completed" for t in self.turns):
            return "incomplete"
        return "completed"

    def _external_reports(self) -> bool:
        """README.md / audit.html via vbt.audit.report.write_reports, when that module exists."""
        try:
            from .audit.report import write_reports  # type: ignore[import-not-found]
        except ImportError:
            return False
        write_reports(self)
        return True

    def _attribute(self, prov) -> dict[str, Any]:
        """Fill attribution gaps from the provenance index; returns the attribution map."""
        arts = self.manifest["artifacts"]
        keys = list(arts)
        prov.index_script_outputs(sorted(k for k, e in arts.items() if e.get("kind") == "code"))
        out: dict[str, Any] = {}
        for rel, e in arts.items():
            owner = st.work_owner(rel)
            unowned = owner is None or owner in st.HARNESS_WORK_DIRS
            mtime = (e.get("mtime_ns") or 0) / 1e9 or None
            a = prov.attribute_path(rel, mtime, keys)
            method = a.get("method")
            if method in ("tool_input", "mcp_return") and a.get("agent"):
                if unowned:
                    e["produced_by"] = a["agent"]
                if unowned or a["agent"] == owner:
                    if not e.get("tool_use_id"):
                        e["tool_use_id"] = a.get("tool_use_id")
                    if not e.get("created_by") or e.get("created_by") == "Bash" and a.get("created_by") != "Bash":
                        e["created_by"] = a.get("created_by")
            elif method == "script" and a.get("agent"):
                cur = e.get("created_by")
                script_name = str(a["created_by"]).split(":")[0]
                if (unowned or a["agent"] == owner) and (cur in (None, "Bash") or cur == script_name):
                    e["created_by"] = a["created_by"]
                    if unowned:
                        e["produced_by"] = a["agent"]
            elif method == "window" and a.get("agent") and unowned and e.get("produced_by") in (None, owner, "unknown"):
                e["produced_by"] = a["agent"]
            if e.get("tool_use_id") and e.get("created_by") and method not in ("tool_input", "mcp_return", "script"):
                a = dict(a, method="tool_capture", agent=e.get("produced_by"), tool_use_id=e.get("tool_use_id"),
                         created_by=e.get("created_by"), confidence="exact")
            out[rel] = a
            e["attribution"] = a.get("method")
        for m in self.manifest.get("misplaced_files") or []:
            if isinstance(m, dict) and not m.get("agent"):
                p = self.dir / m["path"]
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    mtime = None
                a = prov.attribute_path(m["path"], mtime)
                if a.get("agent"):
                    m["agent"], m["attribution"] = a["agent"], a.get("method")
                    m["tool_use_id"] = m.get("tool_use_id") or a.get("tool_use_id")
        return out

    def _reconcile_plan(self, execution: list[dict[str, Any]]) -> None:
        doc = st.read_json(self.dir / "inputs" / "plan.json", None)
        n = self.turns[-1]["turn"] if self.turns else None
        if isinstance(doc, dict) and doc.get("steps"):
            plan_turn = doc.get("turn")
            execs = [e for e in execution if not e.get("orientation")
                     and (not isinstance(plan_turn, int) or (e.get("turn") or 0) >= plan_turn)]
            rec = reconcile(doc, execs, self.manifest["artifacts"])
            rec.update(plan_version=doc.get("version"), plan_turn=plan_turn, plan_written=doc.get("written"))
        else:
            rec = reconcile(None, execution)
        rec["computed_at"] = _now()
        rec["turn"] = n
        self._write_harness_json("report/plan_reconciliation.json", rec)
        if isinstance(doc, dict) and doc.get("steps"):
            self._write_harness_text("report/plan_reconciliation.md", render_plan_md(doc, rec))
        summary = {"has_plan": rec.get("has_plan", False), "summary": rec.get("summary"),
                   "n_deviations": rec.get("n_deviations", 0),
                   "counts": {k: len(rec.get(k) or []) for k in ("not_run", "unplanned", "out_of_order",
                                                                   "missing_output")},
                   "plan_version": rec.get("plan_version")}
        self.manifest["plan_reconciliation"] = summary
        if self.turns:
            self.turns[-1]["plan_reconciliation"] = summary

    def _refresh_claims(self) -> list[dict[str, Any]]:
        path = self.dir / "evidence" / "claims.json"
        claims, err = read_claims_file(path)
        if err:
            # Never overwrite an unreadable record: it may be evidence of tampering.
            self.note_audit_error(f"evidence/claims.json: {err}")
            return claims
        if not path.exists():
            link_cited_by(self.manifest["artifacts"], [])
            return []
        ctx = self._evidence_context()
        ctx.fingerprints = self._current_fingerprints()
        refreshed = refresh_claims(claims, ctx)
        link_cited_by(self.manifest["artifacts"], refreshed)
        self._write_harness_json("evidence/claims.json", claims_payload(refreshed))
        return refreshed

    def _current_fingerprints(self) -> dict[str, str] | None:
        """Current reference-table fingerprints for the status refresh: the cache ``vbt ds fingerprint
        --write`` keeps (``<data.cache_dir>/fingerprints.json``). None without a data configuration or
        cache, so evidence is then not compared against reference data."""
        if not isinstance(self.config, dict) or not isinstance(self.config.get("data"), dict):
            return None
        try:
            from .verify import cached_fingerprints

            return cached_fingerprints(self.config) or None
        except Exception as exc:  # noqa: BLE001 - never fail a refresh over the fingerprint cache
            self.note_audit_error(f"fingerprints: {type(exc).__name__}: {exc}")
            return None

    def _summarise_turn_claims(self, claims: list[dict[str, Any]]) -> None:
        filed = {c.get("id"): c for c in claims}
        for t in self.turns:
            n = t.get("turn")
            mine = [c for c in claims if c.get("turn") == n]
            refs = list(dict.fromkeys(find_refs(str(t.get("response") or ""))))
            t["claims_filed"] = [c.get("id") for c in mine]
            t["claims_unresolved"] = [c.get("id") for c in mine if any(
                (ev.get("evidence_status") == "unresolved") for ev in c.get("evidence") or [])]
            t["claim_refs"] = refs
            t["dangling_claim_refs"] = [r for r in refs if r not in filed]

    def _write_reports(self, claims: list[dict[str, Any]]) -> None:
        turns = self.turns
        self._write_harness_text("inputs/query.txt", "\n\n".join(
            f"--- turn {t.get('turn')} ---\n{t.get('prompt', '')}" for t in turns) + ("\n" if turns else ""))
        st.write_json_atomic(self.dir / "logs" / "cost_report.json",
                             {**self.cost.report(), "run_id": self.run_id, "num_turns": len(turns)})
        self._write_harness_json("session_report.json", {
            "run_id": self.run_id, "status": self.manifest.get("status"), "query": self.manifest.get("query"),
            "updated": _now(), "turns": turns, "cost": self.cost.report()})
        st.write_text_atomic(self.dir / "logs" / "transcript.md", render_transcript(
            self.run_id, turns, self.cost.total_usd,
            excerpt_chars=int(self._audit_cfg("reasoning_excerpt_chars", 2000) or 2000)))
        raw = render_final_report(self.run_id, turns)
        self._write_harness_text("report/FINAL_REPORT.md", raw)
        self._write_harness_text("report/FINAL_REPORT.rendered.md", render_report_md(raw, claims))

    def close(self) -> None:
        """Finalise the record: status completed / incomplete / interrupted / empty."""
        with self._lock:
            if self._closed:
                return
            if self.turns:  # late files are attributed to the last recorded turn
                self._active_turn = self.turns[-1].get("turn")
            try:
                self._save(final=True)
            finally:
                self._active_turn = None
            self._closed = True
            if self._audit_cfg("update_index", True):
                with self._guard("update_index"):
                    try:
                        from .audit.index import update_index  # type: ignore[import-not-found]
                    except ImportError:
                        update_index = None
                    if update_index is not None:
                        update_index(self.dir.parent)
            try:
                with self._trace_lock:
                    self._trace.close()
            except Exception:  # noqa: BLE001
                pass


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------- report text

def _clip_block(text: str, n: int) -> tuple[str, int]:
    return (text if len(text) <= n else text[:n]), len(text)


def render_transcript(run_id: str, turns: list[Mapping[str, Any]], total_usd: float = 0.0,
                      *, excerpt_chars: int = 2000) -> str:
    lines = [f"# Virtual Biotech run {run_id}", "", f"Total cost: ${total_usd:.2f}", ""]
    for t in turns:
        agents = ", ".join(str(a) for a in t.get("agents") or []) or "(none)"
        lines += [f"## Turn {t.get('turn')}", f"**User:** {t.get('prompt', '')}", "",
                  f"**CSO:** {t.get('response', '')}", ""]
        meta = [f"*Status:* {t.get('status', 'completed')}", f"*Agents:* {agents}"]
        try:
            cost = f"*cost:* ${float(t.get('cost_usd') or 0):.2f}"
            if t.get("cumulative_cost_usd") is not None:
                cost += f" (cumulative ${float(t['cumulative_cost_usd']):.2f})"
            meta.append(cost)
        except (TypeError, ValueError):
            pass
        lines += [" · ".join(meta), ""]
        thinking = t.get("thinking_traces") or []
        if thinking:
            lines.append("### CSO Reasoning")
            for th in thinking:
                text = th if isinstance(th, str) else str((th or {}).get("text") or "")
                total = len(text) if isinstance(th, str) else int((th or {}).get("chars") or len(text))
                shown, _ = _clip_block(text, excerpt_chars)
                lines.append("> " + shown.replace("\n", "\n> "))
                if total > len(shown):
                    lines.append(f"> *... [{len(shown):,} of {total:,} chars shown]*")
                lines.append("")
        subs = t.get("subagent_traces") or []
        if subs:
            lines.append("### Sub-agent traces")
            for sa in subs:
                if not isinstance(sa, Mapping):
                    continue
                bits = []
                if sa.get("description"):
                    bits.append(str(sa["description"]))
                if sa.get("status"):
                    bits.append(str(sa["status"]))
                if sa.get("duration_s") is not None:
                    bits.append(f"{sa['duration_s']}s")
                if sa.get("model_calls") is not None:
                    bits.append(f"{sa['model_calls']} model calls")
                if sa.get("tool_calls") is not None:
                    bits.append(f"{sa['tool_calls']} tool calls")
                if sa.get("cost_usd") is not None:
                    try:
                        bits.append(f"${float(sa['cost_usd']):.2f}")
                    except (TypeError, ValueError):
                        pass
                if sa.get("transcript_path"):
                    bits.append(f"transcript: {sa['transcript_path']}")
                name = sa.get("agent") or sa.get("agent_type") or "?"
                lines.append(f"- **{name}**" + (" — " + " · ".join(bits) if bits else ""))
            lines += ["", "*Full sub-agent conversations: logs/agents/ and logs/trace.jsonl*", ""]
    return "\n".join(lines)


def render_final_report(run_id: str, turns: list[Mapping[str, Any]]) -> str:
    """The raw report (anchors kept) that ``verify`` checks."""
    report = [f"# Final report — {run_id}", ""]
    for t in turns:
        prompt = str(t.get("prompt") or "").replace("\n", " ")
        report += [f"## Turn {t.get('turn')}: {prompt[:120]}", "", str(t.get("response") or ""), ""]
    return "\n".join(report)
