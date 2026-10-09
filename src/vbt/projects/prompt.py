"""What agents are told about the active project (the volatile part of their system prompt).

:func:`project_block` lists the project's registered data sources (with their tables), utilities (tool name and
description), plugins and skills, the items waiting for review, and the project notes kept for the agent's role
(``memory/<agent>/MEMORY.md``, first :data:`NOTES_LINES` lines). It is built from the provenance records, so it
names only what passed validation.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import ledger
from .authoring import UTILITY_TOOL_PREFIX, pending_items
from .model import Project, active_project

__all__ = ["project_block", "resources", "NOTES_LINES"]

NOTES_LINES = 100


def resources(project: Project) -> dict[str, list[dict[str, Any]]]:
    """``{kind: [{name, version, ...}]}`` of the registered items."""
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in ledger.ITEM_KINDS}
    for rec in ledger.records(project):
        if rec.get("status") != "registered":
            continue
        entry = {"name": rec.get("name"), "version": rec.get("version"), "when": rec.get("when"),
                 "why": str(rec.get("why") or "")[:200]}
        if rec.get("kind") == "descriptor":
            entry["tables"] = list(rec.get("tables") or [])
        if rec.get("kind") == "utility":
            entry["tool"] = UTILITY_TOOL_PREFIX + str(rec.get("name"))
            entry["description"] = str(rec.get("description") or "")[:300]
        if rec.get("kind") == "plugin":
            entry["plugin_kind"] = rec.get("plugin_kind")
        out.setdefault(str(rec.get("kind")), []).append(entry)
    skills = sorted(p.parent.name for p in project.skills_dir.glob("*/SKILL.md")) if project.skills_dir.is_dir() \
        else []
    out["skills"] = [{"name": s} for s in skills]
    return out


def _notes(project: Project, agent: str) -> str:
    path = project.memory_path(agent)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    text = "\n".join(lines[:NOTES_LINES]).strip()
    if len(lines) > NOTES_LINES:
        text += f"\n... ({len(lines) - NOTES_LINES} more lines in {path})"
    return text


def project_block(config: Mapping[str, Any] | None, agent: str) -> str:
    """The ``## Project`` section for ``agent`` ("" without an active project)."""
    project = active_project(config)
    if project is None:
        return ""
    res = resources(project)
    lines = [f"## Project `{project.name}`", "",
             f"This session runs in project `{project.name}` (`{project.root}`). What the project added was created "
             "by the system for it, validated before registration, and is recorded with its provenance in "
             f"`{project.provenance_dir}`. The core data tools and skills are unchanged; the project only adds."]
    if project.meta.get("description"):
        lines.append(f"Project description: {' '.join(str(project.meta['description']).split())}")
    src = [f"`{d['name']}` ({', '.join(f'`{t}`' for t in d.get('tables') or []) or 'no tables'})"
           for d in res["descriptor"]]
    lines.append("- Project data sources (read them with the `mcp__data__*` tools, like shipped tables): "
                 + ("; ".join(src) if src else "none yet"))
    utils = [f"`{u['tool']}` -- {u.get('description') or ''}" for u in res["utility"]]
    lines.append("- Project utilities (tools; call them like any tool when your role was granted them): "
                 + ("; ".join(utils) if utils else "none yet"))
    if res["plugin"]:
        lines.append("- Project plugins: " + ", ".join(f"{p.get('plugin_kind')}/{p['name']}" for p in res["plugin"]))
    if res["overlay"]:
        lines.append("- Project overlays: " + ", ".join(f"`{o['name']}`" for o in res["overlay"]))
    if res["skills"]:
        lines.append("- Project skills (load with the `Skill` tool): " + ", ".join(f"`{s['name']}`"
                                                                                    for s in res["skills"]))
    pending = pending_items(project)
    if pending:
        lines.append("- Waiting for human review (not usable yet): " + ", ".join(
            f"{p['kind']} `{p['name']}`" for p in pending))
    notes = _notes(project, agent)
    if notes:
        lines += ["", f"Project notes for `{agent}` (from `{project.memory_path(agent)}`; kept across runs):",
                  "<project-notes>", notes, "</project-notes>"]
    return "\n".join(lines)
