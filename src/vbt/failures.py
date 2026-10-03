"""Tool-failure semantics: which failures undermine the evidence of an answer.

Port of the upstream rules (``src/utils/trace_logger.unresolved_tool_failures``
and ``session_audit.data_failure_notice``):

* Every failed tool call is kept for the audit record (``all``).
* A failure is *resolved* only by a later successful call of the same tool with
  identical arguments (``canonical_key``). A success for a different target
  does not establish the failed query's result.
* Only data sources can make an answer's evidence incomplete: MCP data tools
  (``mcp__*``, excluding the in-process ``mcp__provenance__*`` record tools) and
  the web tools. Recovered Bash, Read or Edit errors are routine debugging and
  never trigger the user-facing warning.

Record shape (what ``Runtime`` collects and ``unresolved_from_trace`` builds)::

    {"tool": str, "tool_name": str, "tool_use_id": str | None, "input": dict,
     "is_error": bool, "error": str, "agent": str | None}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "WEB_TOOLS", "canonical_key", "input_sha256", "is_data_source", "unresolved_failures", "unresolved_from_trace",
    "data_failure_notice", "summarize", "records_from_trace",
]

WEB_TOOLS = ("WebSearch", "WebFetch")
_EXCLUDED_PREFIXES = ("mcp__provenance__",)

NOTICE_TEMPLATE = ("Data/evidence warning: these tools failed during this turn: {names}. Their results cannot "
                   "support this answer. Any alternative sources must be identified separately; evidence "
                   "coverage is incomplete.")


def _canonical_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, default=str, ensure_ascii=False)


def canonical_key(tool: str, input: Any) -> tuple[str, str]:
    """``(tool, canonical JSON of the input)``: identical calls share a key."""
    return str(tool or "unknown"), _canonical_json(input)


def input_sha256(input: Any) -> str:
    """sha256 of the canonical JSON of a tool input (key for spilled inputs)."""
    return hashlib.sha256(_canonical_json(input).encode("utf-8")).hexdigest()


def is_data_source(tool: str | None) -> bool:
    """MCP data tools (not the provenance record tools) and the web tools."""
    tool = str(tool or "")
    if tool.startswith(_EXCLUDED_PREFIXES):
        return False
    return tool.startswith("mcp__") or tool in WEB_TOOLS


def _tool_of(rec: Mapping[str, Any]) -> str:
    return str(rec.get("tool") or rec.get("tool_name") or "unknown")


def _key_of(rec: Mapping[str, Any]) -> tuple[str, str]:
    """Resolution key: the tool plus the sha256 of the canonical input (equivalent to
    ``canonical_key``, and comparable with spilled inputs known only by their sha)."""
    if "_key" in rec:
        return rec["_key"]  # type: ignore[return-value]
    return _tool_of(rec), input_sha256(rec.get("input"))


def _public(rec: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in rec.items() if not str(k).startswith("_")}
    tool = _tool_of(rec)
    out["tool"] = tool
    out["tool_name"] = tool
    if "error" in out and out["error"] is not None:
        out["error"] = str(out["error"])[:2000]
    return out


def unresolved_failures(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Failures without a later success of the same tool with identical input.

    ``records`` are in call order; each has ``tool``, ``input`` and ``is_error``
    (records without ``is_error`` count as failures, so a plain failure list
    works too). Returns the failure records, in first-failure order.
    """
    pending: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        key = _key_of(rec)
        if rec.get("is_error", True):
            pending.pop(key, None)  # keep the latest failure, ordered by its time
            pending[key] = _public(rec)
        else:
            pending.pop(key, None)
    return list(pending.values())


def _load_spilled_input(ref: Any, run_dir: str | Path | None) -> Any:
    if not ref:
        return None
    p = Path(str(ref))
    if not p.is_absolute():
        if run_dir is None:
            return None
        p = Path(run_dir) / p
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, Mapping) and "input" in data:
        return data["input"]
    return data


def records_from_trace(events: Iterable[Mapping[str, Any]], *, run_dir: str | Path | None = None,
                       agent: str | None = None) -> list[dict[str, Any]]:
    """Completed tool calls from trace events, joined by ``tool_use_id``.

    The input comes from ``tool_start`` (``input``, ``tool_input`` or the
    spilled ``input_ref`` file, resolved against ``run_dir``); success or
    failure from ``tool_end`` (``is_error``) or ``tool_error``. When a spilled
    input cannot be loaded, its ``input_sha256`` keys the call instead.
    ``agent`` restricts the records to one agent's calls.
    """
    started: dict[str, Mapping[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, Mapping):
            continue
        kind = ev.get("type")
        tuid = ev.get("tool_use_id")
        if kind == "tool_start":
            if tuid:
                started[str(tuid)] = ev
            continue
        if kind not in ("tool_end", "tool_error"):
            continue
        start = started.get(str(tuid), {}) if tuid else {}
        who = ev.get("agent") or start.get("agent")
        if agent is not None and who != agent:
            continue
        tool = ev.get("tool") or ev.get("tool_name") or start.get("tool") or start.get("tool_name") or "unknown"
        inp = ev.get("tool_input")
        if inp is None:
            inp = start.get("input", start.get("tool_input"))
        key = None
        if inp is None and (start.get("input_ref") or ev.get("input_ref")):
            inp = _load_spilled_input(start.get("input_ref") or ev.get("input_ref"), run_dir)
            if inp is None and (start.get("input_sha256") or ev.get("input_sha256")):
                key = (str(tool), str(start.get("input_sha256") or ev.get("input_sha256")))
        is_error = bool(ev.get("is_error")) or kind == "tool_error"
        rec: dict[str, Any] = {"tool": tool, "tool_use_id": tuid, "input": inp if inp is not None else {},
                               "is_error": is_error, "agent": who}
        if key is not None:
            rec["_key"] = key
        if is_error:
            rec["error"] = str(ev.get("error") or ev.get("output") or ev.get("tool_response") or "Tool failed")[:2000]
        out.append(rec)
    return out


def unresolved_from_trace(events: Iterable[Mapping[str, Any]], *, run_dir: str | Path | None = None,
                          data_only: bool = False, agent: str | None = None) -> list[dict[str, Any]]:
    """Unresolved failures in a trace slice (all tools; ``data_only`` keeps data sources).

    Pass the events of one turn (all agents: CSO, specialists, orientation) and
    the run directory so spilled inputs can be loaded.
    """
    failures = unresolved_failures(records_from_trace(events, run_dir=run_dir, agent=agent))
    if data_only:
        failures = [f for f in failures if is_data_source(f["tool"])]
    return failures


def data_failure_notice(failures: Iterable[Mapping[str, Any]]) -> str:
    """The upstream end-of-turn warning for unresolved data-source failures ('' if none)."""
    names = [_tool_of(f) for f in failures if isinstance(f, Mapping) and is_data_source(_tool_of(f))]
    if not names:
        return ""
    return NOTICE_TEMPLATE.format(names=", ".join(dict.fromkeys(names)))


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """``{unresolved_data, recovered_count, other_error_count, all}`` for call records.

    ``all`` is every failed call (audit record); ``unresolved_data`` the
    data-source failures no identical later call resolved (deduplicated by
    call key); ``recovered_count`` the failed calls that a later successful
    call of the same tool with identical input did resolve; and
    ``other_error_count`` the failed non-data calls (Bash, Read, Edit, ...)
    that were never resolved that way: routine debugging or worked-around
    errors, which are not evidence sources but did not *recover* either.
    Repeated unresolved data-source failures and interrupted calls count in
    neither number.
    """
    recs = [r for r in records if isinstance(r, Mapping)]
    all_failures = [_public(r) for r in recs if r.get("is_error", True)]
    unresolved_data = [f for f in unresolved_failures(recs) if is_data_source(f["tool"])]
    succeeded_later: set[tuple[str, str]] = set()
    recovered = other = 0
    for rec in reversed(recs):
        key = _key_of(rec)
        if not rec.get("is_error", True):
            succeeded_later.add(key)
        elif key in succeeded_later:
            recovered += 1
        elif not is_data_source(_tool_of(rec)):
            other += 1
    return {"unresolved_data": unresolved_data, "recovered_count": recovered,
            "other_error_count": other, "all": all_failures}
