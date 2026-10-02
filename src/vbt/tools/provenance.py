"""In-process provenance tools, exposed under the upstream MCP names.

Upstream runs these as an MCP server bound to its run manifest; here they write
straight into the harness ``Run`` so claims, artifacts and plans share one audit
record regardless of provider.

As upstream's prompts expect, every tool *returns* its outcome: a rejected claim,
plan or registration comes back as ``{"ok": false, "errors": [...]}`` for the
agent to fix and refile, not as a tool error (so a validation rejection is never
counted as a data-source failure).
"""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, schema


def _workspace(ctx: ToolContext):
    try:
        return ctx.workspace
    except Exception:  # noqa: BLE001 - registration must not fail on workspace lookup
        return None


def _emit(ctx: ToolContext, kind: str, **data: Any) -> None:
    emit = getattr(ctx.runtime, "emit", None)
    if callable(emit):
        try:
            emit(kind, **data)
        except Exception:  # noqa: BLE001 - UI events never break tools
            pass


def _register(ctx: ToolContext, a: dict[str, Any]) -> Any:
    r = ctx.run.register_artifact(a.get("path", ""), a.get("description", ""), ctx.agent, a.get("kind"),
                                  workspace=_workspace(ctx))
    if r.get("ok"):
        _emit(ctx, "artifact_registered", path=r["artifact"]["path"], agent=ctx.agent)
    return r


def _list(ctx: ToolContext, a: dict[str, Any]) -> Any:
    try:
        rows = ctx.run.list_artifacts(a.get("agent") or None, a.get("kind") or None)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "errors": [f"could not list artifacts: {exc}"]}
    return {"ok": True, "run_id": ctx.run.run_id, "n": len(rows), "artifacts": rows}


def _plan(ctx: ToolContext, a: dict[str, Any]) -> Any:
    roster = None
    agents = getattr(ctx.runtime, "agents", None)
    if agents:
        roster = set(agents)
    return ctx.run.write_plan(a.get("goal", "") or "", a.get("steps"), roster=roster, agent=ctx.agent)


def _claims(ctx: ToolContext, a: dict[str, Any]) -> Any:
    r = ctx.run.record_claims(a.get("claims"), agent=ctx.agent, turn=ctx.run.current_turn)
    if r.get("ok"):
        ids = [c["id"] for c in r.get("claims") or []]
        _emit(ctx, "claims_filed", ids=ids, n=len(ids))
    return r


_EVIDENCE_KINDS = ["artifact", "figure", "table", "code", "tool_call", "citation"]


def provenance_tools() -> list[Tool]:
    evidence = {"type": "array", "description": "At least one item per claim.", "items": {
        "type": "object", "properties": {
            "kind": {"type": "string", "enum": _EVIDENCE_KINDS,
                     "description": "artifact | figure | table | code: a file under work/ (give 'path'); "
                                    "tool_call: a finished, successful tool call (give 'tool_use_id'); "
                                    "citation: external literature (give 'pmid', 'doi' or 'url'; recorded as "
                                    "an external reference, not locally verified)."},
            "path": {"type": "string",
                     "description": "file under work/ — run-relative path from list_artifacts (an unambiguous "
                                    "workspace-relative path or file name also resolves)"},
            "tool_use_id": {"type": "string", "description": "id of a finished, successful tool call"},
            "pmid": {"type": "string", "description": "PubMed ID, e.g. '31234567'"},
            "doi": {"type": "string", "description": "DOI, e.g. '10.1038/s41586-020-2308-7'"},
            "url": {"type": "string", "description": "http(s) URL of the source"},
            "title": {"type": "string"},
            "note": {"type": "string", "description": "which row/column/panel supports the claim"},
            "sha256": {"type": "string", "description": "only when refiling: the hash the claim was filed with"},
        }}}
    step = {"type": "object", "properties": {
        "id": {"type": "string", "description": "short unique id, e.g. 's1' (max 32 chars)"},
        "agent": {"type": "string", "description": "specialist to dispatch"},
        "task": {"type": "string"},
        "depends_on": {"type": "array", "items": {"type": "string"},
                       "description": "ids of steps that must finish first; [] if none"},
        "expected_outputs": {"type": "array", "items": {"type": "string"}},
    }, "required": ["id", "agent"]}
    return [
        Tool("mcp__provenance__register_artifact",
             "Describe a file you produced under work/ so it can be cited: one line on what it shows. "
             "Returns {ok, artifact} or {ok: false, errors}.",
             schema({"path": {"type": "string", "description": "file under work/ (absolute, run-relative or "
                                                               "relative to your workspace)"},
                     "description": {"type": "string"},
                     "kind": {"type": "string", "description": "optional: figure | table | code | report | data"}},
                    ["path", "description"]), _register, source="provenance"),
        Tool("mcp__provenance__list_artifacts",
             "List what this run has produced so far — the exact citable paths — optionally by producing agent "
             "or kind (figure, table, code, report, data). Each row has path, kind, bytes, produced_by, "
             "created_by, tool_use_id, description and cited_by.",
             schema({"agent": {"type": "string"}, "kind": {"type": "string"}}), _list, source="provenance"),
        Tool("mcp__provenance__write_plan",
             "Record the analysis plan before dispatching two or more specialists: goal + steps "
             "[{id, agent, task, depends_on[], expected_outputs[]}]. Validated on write (missing agents, "
             "unknown/self dependencies and cycles are rejected); returns {ok, n_steps, order, parallel_groups, "
             "warnings} or {ok: false, errors}. Steps with no dependency between them run in parallel; "
             "deviations are recorded, not forbidden.",
             schema({"goal": {"type": "string"}, "steps": {"type": "array", "items": step}}, ["steps"]),
             _plan, source="provenance"),
        Tool("mcp__provenance__record_claims",
             "File claim-evidence objects for your synthesis: [{id, text, agent, confidence, evidence[]}]. "
             "Reference each claim inline as [[claim:ID]]. Evidence is validated: files must exist under work/ "
             "with unchanged content, tool_use_ids must be finished successful calls, citations need pmid/doi/url. "
             "Returns {ok, recorded, claims, warnings} or {ok: false, errors} — fix and call again.",
             schema({"claims": {"type": "array", "items": {"type": "object", "properties": {
                 "id": {"type": "string", "description": "unique id, e.g. 'C1' ([A-Za-z0-9_.-], max 64)"},
                 "text": {"type": "string", "description": "the assertion, one self-contained sentence"},
                 "agent": {"type": "string", "description": "specialist whose work supports it (optional)"},
                 "confidence": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                 "evidence": evidence}, "required": ["id", "text", "evidence"]}}}, ["claims"]),
             _claims, source="provenance"),
    ]
