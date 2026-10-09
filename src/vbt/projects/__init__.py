"""Projects: the system creates project-specific utilities as needed (docs/PROJECTS.md).

The core ships general mechanisms; anything specific to one dataset, source or project is created at run time by
the system's agents (the data engineer, ``configs/agents.yaml``), validated, and stored with the project:

* :mod:`.model` -- the project directory, ``vbt project init``, and activation of a project in a configuration
  (skills, read roots, runs, plugin paths and ``VBT_PROJECT_DIR``, which the data layer's catalog searches after
  the shipped descriptors and overlays);
* :mod:`.authoring` -- validation and registration: descriptors, overlays and acquisition specs (``vbt ds lint``
  and ``vbt ds check``), plugins of existing kinds (their conformance suite), utilities (their tests), the optional
  review, and :mod:`.ledger` provenance (who, when, why, hashes, tests run);
* :mod:`.sandbox` / :mod:`.runner` -- where project code runs (bwrap when available, the reaper's memory limit,
  no network for tests);
* :mod:`.utilities` -- registered utilities as ``util__<name>`` tools; :mod:`.prompt` -- what agents are told
  about the project; :mod:`.reload` -- a registration made visible to the running session;
* :mod:`.cli` -- ``vbt project ...`` (also ``python -m vbt.projects``) and ``--project``.

The agent-facing tools are in :mod:`vbt.tools.utilities`.
"""

from __future__ import annotations

from .model import (
    PROJECT_ENV,
    Project,
    ProjectError,
    ProjectSettings,
    activate,
    active_project,
    init_project,
    list_projects,
    projects_root,
    resolve_project,
    write_profile,
)

__all__ = ["PROJECT_ENV", "Project", "ProjectError", "ProjectSettings", "activate", "active_project", "init_project",
           "list_projects", "projects_root", "resolve_project", "write_profile"]
