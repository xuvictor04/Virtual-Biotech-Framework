"""A run's agent notes and the project's (docs/PROJECTS.md, "Notes").

Agents keep notes per run (``<run>/memory/<agent>/MEMORY.md``, ``UpdateMemory``); a project keeps notes per agent
(``<project>/memory/<agent>/MEMORY.md``) that every later session of the project puts in that agent's prompt. At the
close of a session in a project, :func:`offer_at_close` compares the two (``projects.notes_at_close``):

* ``offer`` (default): the lines the project does not have yet are counted and offered, with the command that adds
  them (``vbt project memory NAME --from-run RUN``): notes written by a model reach later sessions only when a
  person chooses;
* ``add``: they are added at once (a ledger event records the run);
* ``off``: nothing is compared.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import ledger
from .model import Project, ProjectSettings, active_project

__all__ = ["NOTES_AT_CLOSE", "new_notes", "add_notes", "offer_at_close"]

log = logging.getLogger(__name__)
NOTES_AT_CLOSE = ("offer", "add", "off")


def _run_note_file(run_dir: Path, agent: str) -> Path:
    return Path(run_dir) / "memory" / agent / "MEMORY.md"


def new_notes(project: Project, run_dir: str | Path, agents: Iterable[str] | None = None) -> dict[str, list[str]]:
    """``{agent: lines}``: the run's note lines the project's notes of that agent do not hold yet (blank lines and
    repeats left out), for ``agents`` (default: every agent with notes in the run)."""
    run_dir = Path(run_dir)
    names = list(agents) if agents is not None else sorted(
        d.name for d in (run_dir / "memory").glob("*") if d.is_dir())
    out: dict[str, list[str]] = {}
    for agent in names:
        src = _run_note_file(run_dir, agent)
        if not src.is_file():
            continue
        dest = project.memory_path(agent)
        seen = set(dest.read_text(encoding="utf-8").splitlines()) if dest.is_file() else set()
        lines: list[str] = []
        for ln in src.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.strip() and ln not in seen:
                lines.append(ln)
                seen.add(ln)
        if lines:
            out[agent] = lines
    return out


def add_notes(project: Project, run_dir: str | Path, agents: Iterable[str] | None = None, *,
              who: Mapping[str, Any] | None = None) -> int:
    """Append the run's new note lines to the project's notes (one ledger event per agent); the lines added."""
    run_dir = Path(run_dir)
    added = 0
    for agent, lines in new_notes(project, run_dir, agents).items():
        dest = project.memory_path(agent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        old = dest.read_text(encoding="utf-8") if dest.is_file() else ""
        dest.write_text(old + ("" if not old or old.endswith("\n") else "\n")
                        + f"<!-- from run {run_dir.name} -->\n" + "\n".join(lines) + "\n", encoding="utf-8")
        added += len(lines)
        ledger.append_ledger(project, {"event": "memory", "kind": "memory", "name": agent,
                                       "who": dict(who or {"user": "cli"}), "why": f"notes from run {run_dir.name}",
                                       "lines": len(lines)})
    return added


def offer_at_close(config: Mapping[str, Any], run_dir: str | Path, *, run_id: str | None = None
                   ) -> dict[str, Any] | None:
    """What :func:`vbt.orchestrator.CSOSession.close` does with the run's notes in a project (see the module
    docstring): None without a project or new notes, else ``{project, mode, lines, agents, command}`` (``added`` for
    ``add``). Never raises."""
    try:
        project = active_project(config)
        if project is None:
            return None
        mode = ProjectSettings.from_config(config, project).notes_at_close
        if mode == "off":
            return None
        pending = new_notes(project, run_dir)
        lines = sum(len(v) for v in pending.values())
        if not lines:
            return None
        run_name = run_id or Path(run_dir).name
        out: dict[str, Any] = {"project": project.name, "mode": mode, "lines": lines, "agents": sorted(pending),
                               "command": f"vbt project memory {project.name} --from-run {run_name}"}
        if mode == "add":
            out["added"] = add_notes(project, run_dir, who={"run": run_name})
        return out
    except Exception:  # noqa: BLE001 - closing a session never fails on its notes
        log.warning("comparing the run's notes with the project's failed", exc_info=True)
        return None
