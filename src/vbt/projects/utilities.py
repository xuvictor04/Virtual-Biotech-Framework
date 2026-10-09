"""Registered project utilities as tools: ``util__<name>`` (docs/PROJECTS.md).

A utility registered by :meth:`vbt.projects.authoring.Author.register_utility` is a directory
``utilities/<name>/`` with ``utility.py``, ``test_utility.py`` and ``utility.json`` (description, entry, mode, JSON
schema). :func:`utility_tools` turns every registered utility whose files still match its provenance record into a
:class:`~vbt.tools.base.Tool`; a call runs the utility in the calling agent's sandbox
(:func:`vbt.projects.sandbox.run_sandboxed`: its own work directory is the working and only writable directory,
the reaper's memory limit applies, and the network follows ``bash.network_isolation`` as Bash does), with the
arguments validated against the declared schema by the runtime like any tool's.

The tools reach a session in two ways: the registration tool adds a new one to the running session's registry, and
the runtime lists the configuration's project's utilities at session start (``Runtime`` calls :func:`utility_tools`
with :func:`vbt.projects.active_project`). :func:`project_tools_for_roots` finds the project from skill roots
instead (a project's ``skills/`` directory), for callers that have only those. Agents get them through the ``util__*`` grant :func:`vbt.agents.load_roster` adds to
every agent that can run code while a project is active.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..tools.base import Tool, ToolContext, ToolFailure
from . import ledger
from .authoring import MANIFEST_FILE, UTILITY_TOOL_PREFIX
from .model import PROJECT_FILE, Project, ProjectError, ProjectSettings
from .sandbox import run_sandboxed, runner_argv

__all__ = ["utility_tools", "utility_tool", "project_tools_for_roots", "project_from_roots", "TOOL_GRANT"]

#: The allowlist pattern that grants every project utility.
TOOL_GRANT = UTILITY_TOOL_PREFIX + "*"


def project_from_roots(roots: Iterable[Any] | None) -> Project | None:
    """The project whose ``skills/`` directory is one of the skill ``roots`` (the active project), or None."""
    for r in roots or []:
        p = Path(str(r))
        if p.name == "skills" and (p.parent / PROJECT_FILE).is_file():
            try:
                return Project.load(p.parent)
            except ProjectError:
                continue
    return None


def project_tools_for_roots(roots: Iterable[Any] | None) -> list[Tool]:
    project = project_from_roots(roots)
    return utility_tools(project) if project is not None else []


def utility_tools(project: Project) -> list[Tool]:
    """A tool per registered utility whose files match its record (changed files: left out, ``vbt project
    check`` reports them)."""
    out = []
    for rec in ledger.records(project, ["utility"]):
        if rec.get("status") != "registered" or ledger.verify(project, rec):
            continue
        try:
            out.append(utility_tool(project, rec))
        except (OSError, ValueError):
            continue
    return out


def _manifest(project: Project, name: str) -> dict[str, Any]:
    data = json.loads((project.utilities_dir / name / MANIFEST_FILE).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{name}: {MANIFEST_FILE} is not an object")
    return data


def utility_tool(project: Project, record: dict[str, Any]) -> Tool:
    name = str(record["name"])
    manifest = _manifest(project, name)
    description = (f"{manifest.get('description') or name} [project utility {project.name}/{name} v"
                   f"{record.get('version')}, created by {((record.get('who') or {}).get('agent')) or 'the system'}; "
                   f"tests passed: {((record.get('validation') or {}).get('tests') or {}).get('passed', '?')}]")
    schema = dict(manifest.get("input_schema") or {"type": "object", "properties": {}})

    async def handler(ctx: ToolContext, args: dict[str, Any]) -> Any:
        return await _call(ctx, project, name, args)

    return Tool(UTILITY_TOOL_PREFIX + name, description, schema, handler, source=f"project:{project.name}",
                tags={"project_utility"})


async def _call(ctx: ToolContext, project: Project, name: str, args: dict[str, Any]) -> Any:
    from ..tools.policy import PathPolicy

    rec = ledger.read_record(project, "utility", name)
    if not rec or rec.get("status") != "registered":
        raise ToolFailure(f"utility {name} is not registered in project {project.name}")
    problems = ledger.verify(project, rec)
    if problems:
        raise ToolFailure(f"utility {name} changed since it was registered ({'; '.join(problems)}); it is not run. "
                          "Re-register it with RegisterUtility (its tests run again).")
    manifest = _manifest(project, name)
    config = dict(getattr(ctx.runtime, "config", None) or {})
    settings = ProjectSettings.from_config(config, project)
    timeout = float(manifest.get("timeout_s") or settings.call_timeout_s)
    network = ((config.get("bash") or {}).get("network_isolation")) != "unshare"
    directory = str(project.utilities_dir / name)
    mode = manifest.get("mode") or "function"
    argv = runner_argv(None, "script", directory) if mode == "script" else \
        runner_argv(None, "call", directory, str(manifest.get("entry") or "run"))
    try:
        res = await run_sandboxed(argv, policy=PathPolicy.from_ctx(ctx), cwd=Path(ctx.workspace), config=config,
                                  label=f"util_{name}_{ctx.tool_call_id or 'call'}", timeout_s=timeout,
                                  network=network, stdin=json.dumps(args).encode("utf-8"))
    except RuntimeError as exc:                        # projects.sandbox: bwrap without a working bwrap
        raise ToolFailure(str(exc)) from None
    ctx.trace("project_utility_call", utility=name, project=project.name, version=rec.get("version"),
              source_hash=rec.get("source_hash"), sandbox=res.sandbox, duration_s=res.duration_s,
              exit_code=res.exit_code, timed_out=res.timed_out)
    notes = "".join(f"[note: {n}]\n" for n in res.notes)
    if res.timed_out:
        raise ToolFailure(f"{notes}utility {name} timed out after {timeout:g}s\n{res.output[-4000:]}")
    result = res.result if isinstance(res.result, dict) else {}
    if not res.ok or not result.get("ok"):
        err = result.get("error") or f"exit code {res.exit_code}"
        tb = result.get("traceback") or ""
        raise ToolFailure(f"{notes}utility {name} failed: {err}\n{tb[-3000:]}\n{res.output[-3000:]}".rstrip())
    if mode == "script":
        return notes + (res.output or "(no output)")
    value = result.get("result")
    if res.output.strip():
        return {"result": value, "log": res.output[-8000:], **({"notes": res.notes} if res.notes else {})}
    return value if not res.notes else {"result": value, "notes": res.notes}
