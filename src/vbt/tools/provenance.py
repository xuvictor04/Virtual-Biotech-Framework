"""In-process provenance tools, exposed under the upstream MCP names.

Upstream runs these as an MCP server bound to its run manifest; here they
write straight into the harness ``Run`` so claims, artifacts and plans share
one audit record regardless of provider.
"""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolContext, ToolFailure, schema


def _register(ctx: ToolContext, a: dict[str, Any]) -> Any:
    r = ctx.run.register_artifact(a["path"], a.get("description", ""), ctx.agent, a.get("kind"))
    if not r["ok"]:
        raise ToolFailure(r["error"])
    return r


def _list(ctx: ToolContext, a: dict[str, Any]) -> Any:
    return {"artifacts": ctx.run.list_artifacts(a.get("agent"), a.get("kind"))}


def _plan(ctx: ToolContext, a: dict[str, Any]) -> Any:
    r = ctx.run.write_plan(a.get("goal", ""), a["steps"])
    if not r["ok"]:
        raise ToolFailure(r["error"])
    return r


def _claims(ctx: ToolContext, a: dict[str, Any]) -> Any:
    r = ctx.run.record_claims(a["claims"], ctx.runtime.tool_call_ids)
    if not r["ok"]:
        raise ToolFailure("; ".join(r["errors"]))
    return r


def provenance_tools() -> list[Tool]:
    evidence = {"type": "array", "items": {"type": "object", "properties": {
        "kind": {"type": "string", "description": "table | figure | code | report | tool_call | web"},
        "path": {"type": "string", "description": "run-relative path from list_artifacts"},
        "tool_use_id": {"type": "string"}, "url": {"type": "string"}, "note": {"type": "string"}}}}
    return [
        Tool("mcp__provenance__register_artifact",
             "Register a file you produced as citable evidence (with a one-line description).",
             schema({"path": {"type": "string"}, "description": {"type": "string"},
                     "kind": {"type": "string"}}, ["path", "description"]), _register, source="provenance"),
        Tool("mcp__provenance__list_artifacts",
             "List artifacts produced in this run (exact paths to cite), optionally by agent or kind.",
             schema({"agent": {"type": "string"}, "kind": {"type": "string"}}), _list, source="provenance"),
        Tool("mcp__provenance__write_plan",
             "Record the analysis plan: goal + steps [{id, agent, task, depends_on[], expected_outputs[]}]. "
             "Steps with no dependency between them run in parallel.",
             schema({"goal": {"type": "string"}, "steps": {"type": "array", "items": {"type": "object"}}},
                    ["steps"]), _plan, source="provenance"),
        Tool("mcp__provenance__record_claims",
             "File claim-evidence objects for your synthesis: [{id, text, agent, confidence, evidence[]}]. "
             "Evidence paths and tool ids are validated.",
             schema({"claims": {"type": "array", "items": {"type": "object", "properties": {
                 "id": {"type": "string"}, "text": {"type": "string"}, "agent": {"type": "string"},
                 "confidence": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                 "evidence": evidence}}}}, ["claims"]), _claims, source="provenance"),
    ]
