"""Projects: a directory per project that holds what the system creates for it (docs/PROJECTS.md).

A project lives under the projects root (``projects.root``, else the deployment layout's ``projects``
directory: ``$VBT_PROJECTS_DIR``, ``<VBT_HOME>/projects``, or ``<data>/projects`` in a checkout) as::

    <root>/<name>/
      project.yaml        schema vbt.project/1: name, description, created, settings (review, sandbox, ...)
      profile.yaml        generated: activates the project for any vbt command (``vbt --profile <it> chat``)
      README.md           generated map of the directory
      descriptors/        <source>.yaml  data sources the project added (acquisition sections inside)
      overlays/           <server>.yaml  bindings of servers the core ships no overlay for
      plugins/<kind>/     <name>.py      plugins of existing kinds (format, layout, statistic, ...)
      utilities/<name>/   utility.py, test_utility.py, utility.json: callable tools (``util__<name>``)
      skills/<name>/      SKILL.md       project skills (searched after the shipped ones)
      memory/<agent>/     MEMORY.md      project notes injected into that agent's prompt
      runs/               the project's run records
      data/               project-owned data files (``${VBT_PROJECT_DIR}/data/<source>`` in descriptors)
      provenance/         ledger.jsonl (every registration and refusal) and <kind>/<name>.json (current records)

:func:`activate` turns a loaded configuration into the project's: ``project: {name, dir}``, the project's skills
after the shipped skill roots, the project directory as a read root, its runs directory, its plugin modules in
``data.plugins.paths`` and ``VBT_PROJECT_DIR`` in ``tool_env`` (the data layer's catalog, the data child, Bash
and the data client then search the project's descriptors and overlays after the shipped ones).
:func:`write_profile` writes the same layer as a profile file, so the unmodified CLI activates a project with
``--profile``. Nothing here writes outside the project directory.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ..config import CONFIG_DIR, resolve_path
from ..datalayer.descriptor.load import PROJECT_ENV, project_plugin_files

__all__ = [
    "PROJECT_SCHEMA", "PROJECT_FILE", "PROFILE_FILE", "PROJECT_ENV", "SUBDIRS", "NAME_RE", "ITEM_NAME_RE",
    "PROJECT_DEFAULTS", "REVIEW_MODES", "SANDBOX_MODES", "ProjectError", "Project", "ProjectSettings",
    "projects_root", "init_project", "resolve_project", "list_projects", "active_project", "activate",
    "profile_layer", "write_profile", "profile_is_current", "now_iso",
]

PROJECT_SCHEMA = "vbt.project/1"
PROJECT_FILE = "project.yaml"
PROFILE_FILE = "profile.yaml"
SUBDIRS = ("descriptors", "overlays", "plugins", "utilities", "skills", "memory", "runs", "data", "provenance")
#: Project names: a directory name and a profile label.
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
#: Utility, plugin and source names: lowercase identifiers (module and tool names are built from them).
ITEM_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
REVIEW_MODES = ("none", "reviewer", "human")
SANDBOX_MODES = ("auto", "bwrap", "none")

#: ``projects.*`` in-code defaults (a config's ``projects`` section is merged over them; a project's own
#: ``project.yaml`` ``settings`` can only make the review stricter).
PROJECT_DEFAULTS: dict[str, Any] = {
    "root": None,              # default: the deployment layout's projects directory
    "review": "none",          # none | reviewer (the scientific reviewer approves) | human (`vbt project approve`)
    # plugins run inside the harness and the data child (not in the sandbox): the stricter of this and `review`
    "plugin_review": "human",
    "sandbox": "auto",         # auto: bwrap when it works here, else the reaper's memory limit (+ no network)
    "test_timeout_s": 600,     # a utility's tests or a plugin's conformance suite
    "call_timeout_s": 1800,    # one call of a registered utility
    "test_network": False,     # tests and conformance suites run without network
    "max_source_bytes": 2_000_000,   # largest descriptor, overlay, plugin or utility file accepted
    "max_import_bytes": None,        # largest data import (null: bounded by the free disk space only)
    "check_depth": "standard",       # `vbt ds check` depth a data spec must pass
    "runs_in_project": True,   # runs of an active project go to <project>/runs
}

_REVIEW_RANK = {m: i for i, m in enumerate(REVIEW_MODES)}


class ProjectError(ValueError):
    """A project that does not exist, a bad name, or a request that would leave the project directory."""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ProjectSettings:
    """The effective ``projects`` settings for one project."""

    review: str = "none"
    plugin_review: str = "human"
    sandbox: str = "auto"
    test_timeout_s: float = 600
    call_timeout_s: float = 1800
    test_network: bool = False
    max_source_bytes: int = 2_000_000
    max_import_bytes: int | None = None
    check_depth: str = "standard"
    runs_in_project: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None, project: "Project | None" = None) -> "ProjectSettings":
        raw = {**PROJECT_DEFAULTS, **dict((config or {}).get("projects") or {})}
        review = str(raw.get("review") or "none")
        own = dict(((project.meta.get("settings") if project is not None else None) or {}))
        if str(own.get("review") or "none") in _REVIEW_RANK and \
                _REVIEW_RANK[str(own.get("review") or "none")] > _REVIEW_RANK.get(review, 0):
            review = str(own["review"])               # the project may ask for more review, never less
        if review not in REVIEW_MODES:
            raise ProjectError(f"projects.review must be one of {', '.join(REVIEW_MODES)} (got {review!r})")
        plugin_review = str(raw.get("plugin_review") or "human")
        if plugin_review not in REVIEW_MODES:
            raise ProjectError(f"projects.plugin_review must be one of {', '.join(REVIEW_MODES)} "
                               f"(got {plugin_review!r})")
        sandbox = str(raw.get("sandbox") or "auto")
        if sandbox not in SANDBOX_MODES:
            raise ProjectError(f"projects.sandbox must be one of {', '.join(SANDBOX_MODES)} (got {sandbox!r})")
        return cls(review=review, plugin_review=plugin_review, sandbox=sandbox,
                   test_timeout_s=float(raw.get("test_timeout_s") or 600),
                   call_timeout_s=float(raw.get("call_timeout_s") or 1800),
                   test_network=bool(raw.get("test_network")),
                   max_source_bytes=int(raw.get("max_source_bytes") or 2_000_000),
                   max_import_bytes=int(raw["max_import_bytes"]) if raw.get("max_import_bytes") else None,
                   check_depth=str(raw.get("check_depth") or "standard"),
                   runs_in_project=bool(raw.get("runs_in_project", True)))

    def review_for(self, kind: str) -> str:
        """The review an item of ``kind`` needs: ``review``, and for a plugin (code the harness and the data child
        import, outside the sandbox) at least ``plugin_review``."""
        if kind == "plugin" and _REVIEW_RANK[self.plugin_review] > _REVIEW_RANK[self.review]:
            return self.plugin_review
        return self.review


@dataclass
class Project:
    """One project directory (``root``) and its ``project.yaml`` (``meta``)."""

    name: str
    root: Path
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, root: str | os.PathLike) -> "Project":
        path = Path(root).expanduser().resolve()
        f = path / PROJECT_FILE
        if not f.is_file():
            raise ProjectError(f"{path} is not a project directory (no {PROJECT_FILE}); create one with "
                               f"`vbt project init NAME`")
        try:
            meta = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ProjectError(f"{f}: cannot read YAML: {exc}") from None
        if not isinstance(meta, dict) or meta.get("schema") != PROJECT_SCHEMA:
            raise ProjectError(f"{f}: not a {PROJECT_SCHEMA} file")
        name = str(meta.get("name") or path.name)
        if not NAME_RE.match(name):
            raise ProjectError(f"{f}: invalid project name {name!r}")
        return cls(name, path, meta)

    # -- directories -------------------------------------------------------------

    def sub(self, name: str) -> Path:
        return self.root / name

    @property
    def descriptors_dir(self) -> Path:
        return self.root / "descriptors"

    @property
    def overlays_dir(self) -> Path:
        return self.root / "overlays"

    @property
    def plugins_dir(self) -> Path:
        return self.root / "plugins"

    @property
    def utilities_dir(self) -> Path:
        return self.root / "utilities"

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def provenance_dir(self) -> Path:
        return self.root / "provenance"

    @property
    def staging_dir(self) -> Path:
        return self.root / ".staging"

    @property
    def profile_path(self) -> Path:
        return self.root / PROFILE_FILE

    def plugin_files(self) -> list[str]:
        return project_plugin_files(self.root)

    def inside(self, path: str | os.PathLike, *, sub: str | None = None) -> Path:
        """``path`` resolved, which must lie inside the project (or its ``sub`` directory); else ProjectError.
        Symlinks are resolved first, so a link cannot lead a write out of the project."""
        base = (self.root / sub) if sub else self.root
        p = Path(path)
        p = (p if p.is_absolute() else base / p).resolve()
        root = base.resolve()
        if p != root and root not in p.parents:
            raise ProjectError(f"{path} is outside the project directory {root}: a project writes only inside "
                               "itself")
        return p

    def memory_path(self, agent: str) -> Path:
        return self.memory_dir / agent / "MEMORY.md"

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "dir": str(self.root), "description": self.meta.get("description") or "",
                "created": self.meta.get("created"), "settings": dict(self.meta.get("settings") or {})}


# ---------------------------------------------------------------------------- locating projects


def projects_root(config: Mapping[str, Any] | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """``projects.root`` (resolved against the checkout), else the deployment layout's projects directory."""
    raw = ((config or {}).get("projects") or {}).get("root")
    if raw:
        return resolve_path(str(raw))
    from ..setup.layout import resolve_layout
    return resolve_layout(config, environ=environ).projects


def init_project(name: str, *, config: Mapping[str, Any] | None = None, root: str | os.PathLike | None = None,
                 description: str = "", review: str | None = None, exist_ok: bool = False) -> Project:
    """Create ``<root>/<name>`` with its subdirectories, ``project.yaml``, ``README.md`` and ``profile.yaml``."""
    if not NAME_RE.match(name or ""):
        raise ProjectError(f"invalid project name {name!r}: lowercase letters, digits, '-' and '_' (start with a "
                           "letter, at most 63 characters)")
    if review is not None and review not in REVIEW_MODES:
        raise ProjectError(f"review must be one of {', '.join(REVIEW_MODES)}")
    base = Path(root).expanduser().resolve() if root else projects_root(config)
    path = base / name
    if (path / PROJECT_FILE).exists():
        if not exist_ok:
            raise ProjectError(f"project {name!r} already exists at {path}")
        return Project.load(path)
    path.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (path / sub).mkdir(exist_ok=True)
    meta: dict[str, Any] = {"schema": PROJECT_SCHEMA, "name": name, "description": description or "",
                            "created": now_iso(), "settings": {}}
    if review:
        meta["settings"]["review"] = review
    (path / PROJECT_FILE).write_text(
        "# A Virtual Biotech project (docs/PROJECTS.md). `settings` may ask for more review than the host's\n"
        "# projects.review (none < reviewer < human), never less.\n" + yaml.safe_dump(meta, sort_keys=False),
        encoding="utf-8")
    (path / ".gitignore").write_text("runs/\n.staging/\n.lock\n", encoding="utf-8")
    project = Project(name, path.resolve(), meta)
    (path / "README.md").write_text(_readme(project), encoding="utf-8")
    write_profile(project, config)
    return project


def _readme(project: Project) -> str:
    return (f"# Project {project.name}\n\n{project.meta.get('description') or ''}\n\n"
            "Files here are created by the system (the data engineer's authoring tools), validated and recorded in\n"
            "`provenance/`; edit them only through `vbt project` so the records stay true. Layout: see\n"
            "docs/PROJECTS.md.\n\n"
            f"Activate the project for any command: `vbt --profile {project.profile_path} chat` (or `--project "
            f"{project.name}` where the CLI offers it).\n")


def resolve_project(ref: str | os.PathLike, config: Mapping[str, Any] | None = None) -> Project:
    """A project by name (under the projects root) or by directory."""
    text = str(ref or "").strip()
    if not text:
        raise ProjectError("a project name or directory is required")
    p = Path(text).expanduser()
    if (p / PROJECT_FILE).is_file():
        return Project.load(p)
    if NAME_RE.match(text):
        cand = projects_root(config) / text
        if (cand / PROJECT_FILE).is_file():
            return Project.load(cand)
        raise ProjectError(f"no project {text!r} under {projects_root(config)}; create it with "
                           f"`vbt project init {text}`")
    raise ProjectError(f"{text} is neither a project name nor a project directory")


def list_projects(config: Mapping[str, Any] | None = None) -> list[Project]:
    base = projects_root(config)
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if (d / PROJECT_FILE).is_file():
            try:
                out.append(Project.load(d))
            except ProjectError:
                continue
    return out


def active_project(config: Mapping[str, Any] | None) -> Project | None:
    """The project a configuration activates (``project.dir``), or None."""
    spec = (config or {}).get("project") or {}
    raw = spec.get("dir") if isinstance(spec, Mapping) else None
    if not raw:
        return None
    try:
        return Project.load(raw)
    except ProjectError:
        return None


# ---------------------------------------------------------------------------- activation


def _strip(values: Iterable[Any], drop: Iterable[str]) -> list[Any]:
    gone = {str(d) for d in drop}
    return [v for v in values or [] if str(v) not in gone]


def activate(config: Mapping[str, Any], project: Project, *, runs: bool | None = None) -> dict[str, Any]:
    """A copy of ``config`` with ``project`` active (see the module docstring). Shipped skill roots, read roots
    and plugin paths keep their order and precedence; the project's come after them. ``runs`` (default
    ``projects.runs_in_project``) moves the run records into the project."""
    cfg = copy.deepcopy(dict(config))
    root = str(project.root)
    cfg["project"] = {"name": project.name, "dir": root}
    paths = cfg.setdefault("paths", {})
    skills = str(project.skills_dir)
    paths["skills"] = [*_strip(paths.get("skills") or [], [skills]), skills]
    paths["read_roots"] = [*_strip(paths.get("read_roots") or [], [root]), root]
    if runs if runs is not None else ProjectSettings.from_config(config, project).runs_in_project:
        paths["runs_dir"] = str(project.runs_dir)
    data = cfg.setdefault("data", {})
    plugins = data.setdefault("plugins", {})
    own_prefix = str(project.plugins_dir) + os.sep
    plugins["paths"] = [*(p for p in plugins.get("paths") or [] if not str(p).startswith(own_prefix)),
                        *project.plugin_files()]
    tool_env = cfg.setdefault("tool_env", {})
    tool_env[PROJECT_ENV] = root
    return cfg


def _raw_layers(config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The raw (unexpanded) YAML of ``configs/default.yaml`` and the config's profiles, in merge order."""
    from ..config import _find

    layers: list[dict[str, Any]] = []
    files = [CONFIG_DIR / "default.yaml"]
    for prof in (config or {}).get("profiles") or []:
        try:
            files.append(_find(str(prof)))
        except FileNotFoundError:
            continue
    for f in files:
        try:
            data = yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict) and not data.get("project"):     # another project's profile never leaks in
            layers.append(data)
    return layers


def _raw(config: Mapping[str, Any] | None, *keys: str) -> Any:
    """The last layer's raw value of ``keys`` (``${...}`` kept), else the loaded config's value."""
    found: Any = None
    for layer in _raw_layers(config):
        cur: Any = layer
        for k in keys:
            cur = cur.get(k) if isinstance(cur, Mapping) else None
        if cur is not None:
            found = cur
    if found is not None:
        return found
    cur = config or {}
    for k in keys:
        cur = cur.get(k) if isinstance(cur, Mapping) else None
    return cur


def profile_layer(project: Project, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The configuration layer :func:`activate` applies, as a profile: the shipped lists as written in the
    configuration files (``${...}`` kept), then the project's entries under ``${vars.project_dir}``."""
    pd = "${vars.project_dir}"
    skills = [s for s in (_raw(config, "paths", "skills") or []) if "${vars.project_dir}" not in str(s)]
    roots = [s for s in (_raw(config, "paths", "read_roots") or []) if "${vars.project_dir}" not in str(s)]
    plugins = [s for s in (_raw(config, "data", "plugins", "paths") or [])
               if "${vars.project_dir}" not in str(s)]
    rel = [f"{pd}/{Path(p).relative_to(project.root).as_posix()}" for p in project.plugin_files()]
    layer: dict[str, Any] = {
        "vars": {"project_dir": str(project.root)},
        "project": {"name": project.name, "dir": pd},
        "paths": {"skills": [*skills, f"{pd}/skills"], "read_roots": [*roots, pd]},
        "data": {"plugins": {"paths": [*plugins, *rel]}},
        "tool_env": {PROJECT_ENV: pd},
    }
    if ProjectSettings.from_config(config, project).runs_in_project:
        layer["paths"]["runs_dir"] = f"{pd}/runs"
    return layer


def _profile_text(project: Project, config: Mapping[str, Any] | None) -> str:
    return ("# Generated by `vbt project` -- activates project " + project.name + " for any vbt command:\n"
            f"#   vbt --profile {project.profile_path} chat\n"
            "# Regenerated when a plugin is registered and by `vbt project profile " + project.name + "`; the\n"
            "# shipped lists are copied from the configuration files as written. Do not edit.\n"
            + yaml.safe_dump(profile_layer(project, config), sort_keys=False, width=120))


def write_profile(project: Project, config: Mapping[str, Any] | None = None) -> Path:
    path = project.profile_path
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(_profile_text(project, config), encoding="utf-8")
    tmp.replace(path)
    return path


def profile_is_current(project: Project, config: Mapping[str, Any] | None = None) -> bool:
    try:
        return project.profile_path.read_text(encoding="utf-8") == _profile_text(project, config)
    except OSError:
        return False
