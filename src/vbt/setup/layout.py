"""Where a deployment keeps its files: data, runs, projects, state and model weights.

``VBT_HOME`` (or ``vbt setup --home``) is the deployment root, laid out as::

    <home>/data       reference data, one directory per source and release (VBT_DATA_DIR)
    <home>/runs       run records (paths.runs_dir)
    <home>/projects   project files kept across runs (VBT_PROJECTS_DIR)
    <home>/state      host configuration and setup state written by `vbt setup` (VBT_STATE_DIR)
    <home>/models     the model server's Hugging Face cache (HF_CACHE)

The container image sets ``VBT_HOME=/srv/vbt``. Without a home (a clone used in place) the harness defaults
apply: ``data/`` and ``runs/`` in the checkout, the state under ``data/.vbt-setup`` and the model cache in
``~/.cache/huggingface``. Each directory can also be set on its own (flag, then variable).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import PROJECT_ROOT, resolve_path

__all__ = ["Layout", "resolve_layout", "HOME_ENV", "STATE_ENV", "PROJECTS_ENV", "DATA_ENV"]

HOME_ENV = "VBT_HOME"
STATE_ENV = "VBT_STATE_DIR"
DATA_ENV = "VBT_DATA_DIR"
PROJECTS_ENV = "VBT_PROJECTS_DIR"
MODELS_ENV = "HF_CACHE"


@dataclass(frozen=True)
class Layout:
    """The deployment's directories (absolute)."""

    home: Path | None
    state: Path
    data: Path
    runs: Path
    projects: Path
    models: Path

    def as_dict(self) -> dict[str, str | None]:
        return {"home": str(self.home) if self.home else None, "state": str(self.state), "data": str(self.data),
                "runs": str(self.runs), "projects": str(self.projects), "models": str(self.models)}

    def dirs(self) -> dict[str, Path]:
        """The directories setup creates (the model cache only matters where a model server runs)."""
        return {"state": self.state, "data": self.data, "runs": self.runs, "projects": self.projects}


def _pick(flag: Any, env: Mapping[str, str], name: str) -> str | None:
    if flag:
        return str(flag)
    value = (env.get(name) or "").strip()
    return value or None


def resolve_layout(config: Mapping[str, Any] | None = None, *, home: str | None = None, state: str | None = None,
                   data: str | None = None, runs: str | None = None, projects: str | None = None,
                   models: str | None = None, environ: Mapping[str, str] | None = None) -> Layout:
    """The layout from the flags, then the variables, then ``<home>/...``, then the harness defaults."""
    env = os.environ if environ is None else environ
    home_s = _pick(home, env, HOME_ENV)
    root = resolve_path(home_s) if home_s else None

    def under(name: str, default: Path) -> Path:
        return (root / name) if root is not None else default

    data_s = _pick(data, env, DATA_ENV)
    data_p = resolve_path(data_s) if data_s else under("data", PROJECT_ROOT / "data")
    paths = (config or {}).get("paths") or {}
    runs_default = resolve_path(paths.get("runs_dir") or "runs")
    runs_p = resolve_path(runs) if runs else under("runs", runs_default)
    state_s = _pick(state, env, STATE_ENV)
    state_p = resolve_path(state_s) if state_s else under("state", data_p / ".vbt-setup")
    projects_s = _pick(projects, env, PROJECTS_ENV)
    projects_p = resolve_path(projects_s) if projects_s else under("projects", data_p / "projects")
    models_s = _pick(models, env, MODELS_ENV)
    models_p = resolve_path(models_s) if models_s else under("models", Path("~/.cache/huggingface").expanduser())
    return Layout(home=root, state=state_p, data=data_p, runs=runs_p, projects=projects_p, models=models_p)
