"""``vbt ds retro-audit <run>``: re-classify a recorded run's data calls offline (§16, §22 step 1).

Every ``mcp__<server>__<tool>`` call in ``logs/trace.jsonl`` (the data child's own verbs excepted)
is run through what the gateway would have decided **without calling anything**: the tool's
binding (a ``serve: block`` binding is ``quarantined``), resolution of its identifier arguments
against the resolver sidecars already built in ``data.cache_dir`` (an unknown identifier is
``not_found``, a malformed one ``invalid_argument``; without a sidecar existence stays unknown), and
:func:`vbt.datalayer.gateway.classify.classify` on the recorded output. No witness runs, so a
lookup miss on a table that is not the identifier's universe is ``empty_unverified`` and a
contradiction (``tool_defect``) is found only where resolution proved the entity exists.

The report counts, per outcome, the calls that succeeded when recorded but would now be
``not_found``, ``empty``, ``empty_unverified``, ``quarantined``, ``tool_defect``, ``invalid_argument`` or
an error, and how many of them a filed claim cites (``evidence/claims.json`` tool_call evidence).
Those are the claims whose support the data layer would have refused or qualified. This module
imports no pyarrow and starts no server.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

__all__ = ["CallAudit", "retro_audit", "audit_calls", "format_report", "resolve_run", "OUTCOMES", "CHANGED"]

#: Outcomes a call can have under the gateway (the classify outcomes plus the prepare-time refusals).
OUTCOMES = ("ok", "partial", "empty", "empty_unverified", "not_found", "invalid_argument", "ambiguous",
            "quarantined", "tool_defect", "source_error", "oom", "interrupted")
#: Outcomes that mean a call recorded as a plain success would now be refused or qualified.
CHANGED = frozenset({"empty", "empty_unverified", "not_found", "invalid_argument", "ambiguous", "quarantined",
                     "tool_defect", "source_error", "oom"})
_ERROR_PREFIX = "Error: "


@dataclass
class CallAudit:
    tool_use_id: str
    tool: str
    agent: str | None
    recorded: dict[str, Any]
    outcome: str
    reason: str = ""
    cited_by: list[str] = field(default_factory=list)
    resolutions: dict[str, str] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return not self.recorded.get("is_error") and self.outcome in CHANGED

    def to_dict(self) -> dict[str, Any]:
        return {"tool_use_id": self.tool_use_id, "tool": self.tool, "agent": self.agent, "recorded": self.recorded,
                "outcome": self.outcome, "reason": self.reason, "changed": self.changed, "cited_by": self.cited_by,
                "resolutions": self.resolutions}


def resolve_run(arg: str | Path, config: Mapping[str, Any]) -> Path:
    """A run directory from an id, prefix, path or ``latest`` (``vbt.audit.index.resolve_run``)."""
    from ..audit.index import resolve_run as _resolve
    from ..config import resolve_path

    runs_dir = resolve_path(((config.get("paths") or {}).get("runs_dir")) or "runs")
    try:
        return _resolve(arg, runs_dir)
    except LookupError as exc:
        raise FileNotFoundError(str(exc)) from exc


# ---------------------------------------------------------------------------- reading a run


def _calls(run_dir: Path) -> list[dict[str, Any]]:
    """``[{tool_use_id, tool, agent, input, output, is_error, ...}]`` of the recorded tool calls."""
    from ..audit.provenance import read_trace

    events, _bad = read_trace(run_dir / "logs" / "trace.jsonl")
    starts: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for ev in events:
        tuid = str(ev.get("tool_use_id") or "")
        if ev.get("type") == "tool_start" and tuid:
            starts[tuid] = ev
        elif ev.get("type") == "tool_end" and tuid:
            start = starts.get(tuid, {})
            call = {**ev, "input": start.get("input")}
            if call["input"] is None and start.get("input_sha256"):
                path = run_dir / "logs" / "tool_inputs" / f"{tuid}.json"
                try:
                    call["input"] = json.loads(path.read_text()).get("input")
                except (OSError, ValueError):
                    call["input"] = start.get("input_preview")
            spill = ev.get("output_path")
            if spill:
                try:   # the full output when the trace kept only its head
                    call["output"] = (run_dir / spill).read_text(errors="replace")
                except OSError:
                    pass
            out.append(call)
    return out


def _citations(run_dir: Path) -> dict[str, list[str]]:
    """``{tool_use_id: [claim ids]}`` from the filed claims' tool_call evidence."""
    from ..audit.claims import load_claims

    out: dict[str, list[str]] = {}
    for claim in load_claims(run_dir):
        for ev in claim.get("evidence") or []:
            if not isinstance(ev, Mapping):
                continue
            tuid = str(ev.get("tool_use_id") or "").strip()
            if tuid and (ev.get("kind") in (None, "tool_call", "tool", "tool_use") or ev.get("tool_use_id")):
                out.setdefault(tuid, [])
                if claim.get("id") not in out[tuid]:
                    out[tuid].append(str(claim.get("id")))
    return out


# ---------------------------------------------------------------------------- classification


def _universe_tables(catalog: Any, contract: Any) -> dict[str, list[str]]:
    """``{argument: [source.table]}``: the identity universes of each identifier argument."""
    from .gateway.contracts import bound_column, column_spec

    out: dict[str, list[str]] = {}
    for name, a in contract.args.items():
        table, column = bound_column(contract, name, a)
        src = table.split(".")[0] if table else None
        kinds = list(a.accepts)
        spec = column_spec(contract, table, column)
        if getattr(spec, "id_type", None):
            kinds.append(spec.id_type)
        refs: list[str] = []
        for k in kinds:
            try:
                s, idt = catalog.id_type(k, src)
            except Exception:  # noqa: BLE001
                continue
            for u in (idt.universe if isinstance(idt.universe, list) else [idt.universe]):
                if u is None:
                    continue
                ut = getattr(u, "table", None) or str(u).split(".")[0]
                refs.append(ut if "." in ut and ut.split(".")[0] in catalog.sources else f"{s}.{ut}")
        if a.universe:
            refs.append(a.universe if a.universe.count(".") == 1 else ".".join(a.universe.split(".")[:2]))
        if refs:
            out[name] = list(dict.fromkeys(refs))
    return out


def _accepts(contract: Any, name: str, a: Any) -> tuple[list[str], str | None]:
    from .gateway.contracts import bound_column, column_spec

    table, column = bound_column(contract, name, a)
    kinds = list(a.accepts)
    spec = column_spec(contract, table, column)
    if not kinds and getattr(spec, "id_type", None):
        kinds.append(spec.id_type)
    return kinds, (table.split(".")[0] if table else None)


def _resolve_args(resolver: Any, contract: Any, args: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str],
                                                                                   tuple[str, str] | None]:
    """``(existence, resolutions, refusal)`` of the call's identifier arguments: ``existence``
    ``{arg: exists|absent|unknown}``, ``resolutions`` ``{arg: summary}``, ``refusal`` ``(outcome,
    reason)`` when resolution alone would have refused the call before it reached upstream."""
    existence: dict[str, str] = {}
    resolutions: dict[str, str] = {}
    refusal: tuple[str, str] | None = None
    for name, a in contract.identifier_args.items():
        value = args.get(name)
        if value is None or a.existence == "off":
            continue
        kinds, src = _accepts(contract, name, a)
        if not kinds:
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            if not isinstance(v, (str, int)):
                continue
            try:
                res = resolver.resolve(v, kinds, existence=a.existence, source_hint=src)
            except Exception as exc:  # noqa: BLE001 - a resolver configuration problem is not the call's
                resolutions[name] = f"{v!r}: unresolvable here ({type(exc).__name__})"
                continue
            resolutions[name] = res.summary() if hasattr(res, "summary") else str(res.status)
            if res.existence:
                existence[name] = str(res.existence)
            if refusal is None:
                if res.status == "rejected":
                    refusal = ("invalid_argument", f"{name}={v!r} is not a valid {'|'.join(kinds)}")
                elif res.status == "ambiguous":
                    refusal = ("ambiguous", f"{name}={v!r} matches several entities")
                elif res.status == "not_found" and res.existence == "absent" and a.existence != "upstream":
                    refusal = ("not_found", f"{name}={v!r} is not in its universe")
    return existence, resolutions, refusal


def _raw(call: Mapping[str, Any]) -> Any:
    """A :class:`~vbt.datalayer.api.RawResult`-like view of a recorded output."""
    from .gateway.classify import EMPTY_LOOKUP

    text = str(call.get("output") or "")
    if call.get("is_error"):
        body = text[len(_ERROR_PREFIX):] if text.startswith(_ERROR_PREFIX) else text
        envelope = "empty_lookup" if EMPTY_LOOKUP.fullmatch(body.strip()) else "is_error"
        return SimpleNamespace(text=body, structured=None, envelope=envelope, error_text=body)
    return SimpleNamespace(text=text, structured=None, envelope="ok", error_text=None)


def audit_calls(calls: Iterable[Mapping[str, Any]], catalog: Any, resolver: Any = None,
                citations: Mapping[str, list[str]] | None = None) -> list[CallAudit]:
    """Re-classify recorded MCP calls (see the module docstring)."""
    from .gateway.classify import classify
    from .gateway.service_client import DATA_SERVER

    citations = citations or {}
    out: list[CallAudit] = []
    for call in calls:
        name = str(call.get("tool") or "")
        if not name.startswith("mcp__"):
            continue
        parts = name.split("__", 2)
        if len(parts) != 3 or parts[1] in (DATA_SERVER, "provenance"):
            continue
        server, tool = parts[1], parts[2]
        tuid = str(call.get("tool_use_id") or "")
        recorded = {"is_error": bool(call.get("is_error"))}
        for k in ("result_status", "error_kind"):
            if call.get(k):
                recorded[k] = call[k]
        audit = CallAudit(tuid, name, call.get("agent"), recorded, "ok", cited_by=list(citations.get(tuid, [])))
        out.append(audit)
        if call.get("interrupted"):
            audit.outcome, audit.reason = "interrupted", "the call was interrupted"
            continue
        args = call.get("input") if isinstance(call.get("input"), Mapping) else {}
        try:
            contract = catalog.contract(server, tool)
        except Exception:  # noqa: BLE001 - unknown server: the generic guard's view
            contract = None
        b = getattr(contract, "binding", None)
        if b is not None and b.serve == "block":
            audit.outcome = "quarantined"
            audit.reason = (b.block.reason if b.block is not None and getattr(b.block, "reason", None)
                            else "the binding blocks this tool")
            continue
        existence: dict[str, str] = {}
        if b is not None and resolver is not None:
            existence, audit.resolutions, refusal = _resolve_args(resolver, contract, args)
            if refusal is not None:
                audit.outcome, audit.reason = refusal
                continue
        plan = SimpleNamespace(args_raw=dict(args), bound_table=getattr(contract, "bound_table", None),
                               existence=existence)
        universes = _universe_tables(catalog, contract) if b is not None else {}
        c = classify(_raw(call), contract, plan, universe_tables=universes, witness_total=None, tool=name)
        audit.outcome = "tool_defect" if c.outcome == "contradiction" else c.outcome
        audit.reason = c.reason
    return out


def retro_audit(run_dir: str | Path, config: Mapping[str, Any], *, resolver: Any = None, catalog: Any = None
                ) -> dict[str, Any]:
    """The retro-audit report of one run (JSON-able; :func:`format_report` renders it)."""
    run_dir = Path(run_dir)
    notes: list[str] = []
    if catalog is None or resolver is None:
        from .cli import resolver_for
        try:
            r, c = resolver_for(dict(config))
            resolver = resolver if resolver is not None else r
            catalog = catalog if catalog is not None else c
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"the data layer's catalog could not be loaded: {exc}") from exc
    calls = _calls(run_dir)
    citations = _citations(run_dir)
    audits = audit_calls(calls, catalog, resolver, citations)
    outcomes = Counter(a.outcome for a in audits)
    changed = [a for a in audits if a.changed]
    cited = [a for a in changed if a.cited_by]
    if not (run_dir / "logs" / "trace.jsonl").is_file():
        notes.append("no logs/trace.jsonl: nothing to audit")
    if any(not a.resolutions for a in audits if a.outcome == "ok"):
        notes.append("identifiers without a built resolver sidecar keep existence unknown "
                     "(`vbt ds index build` builds them)")
    notes.append("offline: no witness ran, so lookup misses off the identifier's universe are empty_unverified "
                 "and contradictions are found only where resolution proved the entity exists")
    return {
        "run": str(run_dir),
        "n_calls": len(calls),
        "n_data_calls": len(audits),
        "outcomes": dict(sorted(outcomes.items())),
        "changed": dict(sorted(Counter(a.outcome for a in changed).items())),
        "cited": dict(sorted(Counter(a.outcome for a in cited).items())),
        "claims_affected": sorted({cid for a in cited for cid in a.cited_by}),
        "calls": [a.to_dict() for a in audits],
        "notes": notes,
    }


def format_report(report: Mapping[str, Any]) -> list[str]:
    """Text lines of a retro-audit report: per-outcome counts, then the changed calls (cited first)."""
    lines = [f"retro-audit of {report['run']}",
             f"  {report['n_data_calls']} data calls of {report['n_calls']} tool calls"]
    lines.append(f"  {'outcome':<18} {'calls':>6} {'now changed':>12} {'cited':>6}")
    for outcome in OUTCOMES:
        n = report["outcomes"].get(outcome, 0)
        if not n:
            continue
        lines.append(f"  {outcome:<18} {n:>6} {report['changed'].get(outcome, 0):>12} "
                     f"{report['cited'].get(outcome, 0):>6}")
    changed = [c for c in report["calls"] if c["changed"]]
    if changed:
        lines.append("")
        lines.append("Calls that succeeded when recorded but would now be refused or qualified:")
        for c in sorted(changed, key=lambda c: (not c["cited_by"], c["tool"])):
            cite = f"  [cited by {', '.join(c['cited_by'])}]" if c["cited_by"] else ""
            lines.append(f"  {c['tool_use_id']} {c['tool']}: {c['outcome']} -- {c['reason'][:160]}{cite}")
    if report.get("claims_affected"):
        lines.append("")
        lines.append(f"Claims citing them: {', '.join(report['claims_affected'])}")
    for note in report.get("notes") or []:
        lines.append(f"note: {note}")
    return lines
