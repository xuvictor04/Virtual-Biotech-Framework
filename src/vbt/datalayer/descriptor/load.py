"""Load descriptors and overlays from YAML, with ``vbt.config`` variable expansion and digests. No pyarrow.

Strings are expanded like ``vbt.config.load_config``: ``${VAR}``, ``${VAR:-x}``, ``${VAR-x}`` and
``${vars.name}`` (``${VAR}`` takes an ``env.VAR`` entry of the variables first, see
:func:`variables_from_config`); additionally ``${run.<name>}`` takes a value from the run mapping passed in
(``${run.mcp_output_dir}`` for tables a tool call materialises during a run, rev 2). A
``${run.*}`` variable without a value is left in place, so the table stays a template until
a run provides it. Overlay files whose name starts with ``_`` are generic overlays.

R8: :func:`load_descriptors` and :func:`load_overlays` raise on the first file that does not load,
unless they are given a ``quarantine`` list. Then a file that does not load (YAML, schema, a source
or server declared twice) is left out on its own and recorded as a :class:`Quarantined` entry with
the name it declares (read from the file when it parses, else from a ``source:``/``server:`` line),
so the catalog refuses only the tools that depend on it.

Projects (docs/PROJECTS.md): the active project's directory is ``$VBT_PROJECT_DIR`` (the harness puts it in
``tool_env`` when a project is active, so the data child, Bash and the data client see the same project; a
``env.VBT_PROJECT_DIR`` variable takes precedence). :func:`project_search_dirs` names its ``descriptors/`` and
``overlays/``, which the catalog searches after the shipped directories, and :func:`project_plugin_files` the
plugin modules under its ``plugins/<kind>/``; :func:`approved_project_plugins` those of them plugin discovery
imports: registered (approved when the review asked for it) and unchanged since.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from pydantic import BaseModel, ValidationError

from ... import config as _config
from .models import SourceDescriptor
from .overlay import Overlay

__all__ = [
    "DescriptorError", "Quarantined", "expand", "load_yaml", "load_descriptor", "load_descriptors", "load_overlay",
    "load_overlays", "digest", "variables_from_config", "YAML_SUFFIXES", "PROJECT_ENV", "PROJECT_DESCRIPTORS",
    "PROJECT_OVERLAYS", "PROJECT_PLUGINS", "project_dir", "project_search_dirs", "project_plugin_files",
    "approved_project_plugins",
]

YAML_SUFFIXES = (".yaml", ".yml")

#: The active project's directory (an absolute path); unset or empty: no project.
PROJECT_ENV = "VBT_PROJECT_DIR"
#: Subdirectories of a project directory the data layer searches.
PROJECT_DESCRIPTORS = "descriptors"
PROJECT_OVERLAYS = "overlays"
PROJECT_PLUGINS = "plugins"
_VAR = re.compile(r"\$\{([A-Za-z_][\w.]*)(?:(:?-)([^{}]*))?\}")


class DescriptorError(ValueError):
    """A descriptor or overlay file that cannot be read or does not validate."""

    def __init__(self, path: str | Path, message: str) -> None:
        self.path = str(path)
        self.detail = str(message)
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True)
class Quarantined:
    """A descriptor or overlay file left out of the catalog because it does not load (R8).

    ``kind`` is ``descriptor``, ``overlay`` or ``generic`` (an overlay file starting with ``_``);
    ``name`` the source or server it declares (None when that cannot be read; a reviewed overlay
    falls back to its file stem); ``error`` the loader's message. ``tables``/``id_types`` are the names
    a descriptor declares when its YAML parses (None: unknown, so any unresolved reference may be its).
    ``same_as`` are the ``server.tool`` names an overlay's bindings serve for other servers (from its YAML, or its
    ``same_as:`` lines; None: unknown, so any tool without a binding of its own may be bound there)."""

    path: str
    kind: str
    name: str | None
    error: str
    tables: tuple[str, ...] | None = None
    id_types: tuple[str, ...] | None = None
    same_as: tuple[str, ...] | None = ()

    @property
    def file(self) -> str:
        return Path(self.path).name

    @property
    def summary(self) -> str:
        """The error on one line (YAML errors span several), at most 300 characters."""
        return " ".join(self.error.split())[:300]

    @property
    def reason(self) -> str:
        """``DescriptorError: <path>: <error>`` on one line, as a load of the file alone raises it."""
        return f"DescriptorError: {self.path}: {self.summary}"

    def to_json(self) -> dict[str, Any]:
        return {"file": self.path, "kind": self.kind, "name": self.name, "error": self.summary}


def variables_from_config(config: Mapping[str, Any] | None) -> dict[str, str]:
    """The ``vars.*`` mapping of a loaded config (``project_root`` always defined), plus the
    config's ``tool_env`` as ``env.<NAME>`` entries: the environment the tool children get, which
    ``${NAME}`` prefers over this process's environment."""
    config = config or {}
    variables = {str(k): "" if v is None else str(v) for k, v in (config.get("vars") or {}).items()}
    variables.update({f"env.{k}": str(v) for k, v in (config.get("tool_env") or {}).items() if v})
    variables.setdefault("project_root", str(_config.PROJECT_ROOT))
    return variables


def expand(value: Any, variables: Mapping[str, str] | None = None, run: Mapping[str, Any] | None = None) -> Any:
    """Expand ``${...}`` in every string of ``value`` (see the module docstring)."""
    variables = dict(variables or {})
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            key, op, default = m.group(1), m.group(2), m.group(3)
            if key.startswith("run."):
                val = None if run is None else run.get(key[4:])
                if val is None:
                    return default if op in (":-", "-") else m.group(0)
                return str(val)
            if key.startswith("vars."):
                val = variables.get(key[5:])
            else:
                val = variables.get(f"env.{key}", os.environ.get(key))
            if op == ":-":
                return default if not val else val
            if op == "-":
                return default if val is None else val
            return val or ""

        prev = None
        while prev != value:
            prev, value = value, _VAR.sub(sub, value)
        return value
    if isinstance(value, dict):
        return {k: expand(v, variables, run) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, variables, run) for v in value]
    return value


def load_yaml(path: str | Path, variables: Mapping[str, str] | None = None,
              run: Mapping[str, Any] | None = None) -> dict[str, Any]:
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DescriptorError(p, f"cannot read YAML: {exc}") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise DescriptorError(p, "top level must be a mapping")
    return expand(data, variables, run)


def _validation_message(exc: ValidationError, limit: int = 12) -> str:
    lines = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        lines.append(f"{loc}: {err.get('msg')}")
    more = len(exc.errors()) - limit
    if more > 0:
        lines.append(f"... {more} more")
    return "; ".join(lines)


def load_descriptor(path: str | Path, variables: Mapping[str, str] | None = None,
                    run: Mapping[str, Any] | None = None) -> SourceDescriptor:
    data = load_yaml(path, variables, run)
    try:
        return SourceDescriptor.model_validate(data)
    except ValidationError as exc:
        raise DescriptorError(path, _validation_message(exc)) from None


def _yaml_files(directory: str | Path) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix in YAML_SUFFIXES and not p.name.startswith("."))


_DECLARED = {key: re.compile(rf"^{key}:[ \t]*[\"']?([^\"'\s#]+)", re.MULTILINE) for key in ("source", "server")}


def _declared(path: Path, key: str) -> tuple[str | None, dict[str, Any] | None]:
    """``(name, mapping)``: the ``key`` a file that does not load declares (from its YAML when that parses,
    else from a top-level ``key:`` line) and its parsed mapping (None when it does not parse)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict):
        value = data.get(key)
        return (value if isinstance(value, str) and value else None), data
    m = _DECLARED[key].search(text)
    return (m.group(1) if m else None), None


def _names(data: dict[str, Any] | None, key: str) -> tuple[str, ...] | None:
    value = (data or {}).get(key)
    return tuple(sorted(str(k) for k in value)) if isinstance(value, dict) else None


def _quarantined(path: Path, kind: str, exc: BaseException, name: str | None = None) -> Quarantined:
    """The :class:`Quarantined` entry of a file whose load raised ``exc``."""
    error = exc.detail if isinstance(exc, DescriptorError) else f"{type(exc).__name__}: {exc}"
    key = "source" if kind == "descriptor" else "server"
    declared, data = _declared(path, key)
    if name is None:
        name = declared
    if name is None and kind == "overlay":
        name = path.stem                               # reviewed overlays are named after their server
    if kind == "descriptor":
        return Quarantined(str(path), kind, name, error, _names(data, "tables"), _names(data, "id_types"))
    return Quarantined(str(path), kind, name, error, same_as=_same_as(path, data))


_SAME_AS = re.compile(r"same_as:[ \t]*\[([^\]\n]*)\]")
_TOOL_NAME = re.compile(r"^[A-Za-z_][\w-]*\.[A-Za-z_][\w-]*$")


def _same_as(path: Path, data: dict[str, Any] | None) -> tuple[str, ...] | None:
    """The other servers' tools an overlay binds with ``same_as`` (a tool whose reviewed binding sits in a file
    that does not load depends on that file, R8). From the parsed YAML, else from its ``same_as: [...]`` lines;
    None when the file mentions ``same_as`` in a form neither reads."""
    if isinstance(data, dict):
        out: list[str] = []
        for b in (data.get("tools") or {}).values() if isinstance(data.get("tools"), dict) else ():
            names = b.get("same_as") if isinstance(b, dict) else None
            out.extend(str(n) for n in (names if isinstance(names, list) else [names] if names else []))
        return tuple(sorted(set(out)))
    try:
        text = re.sub(r"#[^\n]*", "", path.read_text(encoding="utf-8", errors="replace"))   # comments never bind
    except OSError:
        return None
    keys = len(re.findall(r"\bsame_as\s*:", text))
    if not keys:
        return ()
    found = [n.strip().strip("'\"") for m in _SAME_AS.finditer(text) for n in m.group(1).split(",") if n.strip()]
    if len(_SAME_AS.findall(text)) != keys or not all(_TOOL_NAME.match(n) for n in found):
        return None
    return tuple(sorted(set(found)))


def _duplicates(kind: str, key: str, paths: dict[str, list[Path]], quarantine: list[Quarantined],
                loaded: dict[str, Any]) -> None:
    """Every file of a name declared more than once is quarantined (which one is meant is unknown)."""
    for name, files in paths.items():
        loaded.pop(name, None)
        for p in files:
            others = ", ".join(str(o) for o in files if o != p)
            q = _quarantined(p, kind, DescriptorError(p, f"{key} {name!r} is also declared in {others}"), name)
            quarantine.append(q)


def load_descriptors(directory: str | Path, variables: Mapping[str, str] | None = None,
                     run: Mapping[str, Any] | None = None, *,
                     quarantine: list[Quarantined] | None = None) -> dict[str, SourceDescriptor]:
    """Every ``*.yaml`` descriptor of ``directory`` by source name (a missing directory gives none).

    Without ``quarantine`` the first file that does not load raises :class:`DescriptorError`; with it,
    that file is appended to ``quarantine`` and the others load (a source declared by several files
    quarantines all of them)."""
    out: dict[str, SourceDescriptor] = {}
    origin: dict[str, Path] = {}
    twice: dict[str, list[Path]] = {}
    for p in _yaml_files(directory):
        try:
            desc = load_descriptor(p, variables, run)
        except Exception as exc:  # noqa: BLE001 - quarantined on its own when a list is given
            if quarantine is None:
                raise
            quarantine.append(_quarantined(p, "descriptor", exc))
            continue
        if desc.source in out or desc.source in twice:
            if quarantine is None:
                raise DescriptorError(p, f"source {desc.source!r} is also declared in {origin[desc.source]}")
            twice.setdefault(desc.source, [origin[desc.source]]).append(p)
            continue
        out[desc.source] = desc
        origin[desc.source] = p
    if quarantine is not None:
        _duplicates("descriptor", "source", twice, quarantine, out)
    return out


def load_overlay(path: str | Path, variables: Mapping[str, str] | None = None) -> Overlay:
    data = load_yaml(path, variables)
    try:
        return Overlay.model_validate(data)
    except ValidationError as exc:
        raise DescriptorError(path, _validation_message(exc)) from None


def load_overlays(directory: str | Path, variables: Mapping[str, str] | None = None, *,
                  quarantine: list[Quarantined] | None = None) -> tuple[dict[str, Overlay], list[Overlay]]:
    """``(reviewed overlays by server, generic overlays)``; files starting with ``_`` are generic.

    ``quarantine`` as in :func:`load_descriptors` (kind ``overlay`` or ``generic``)."""
    reviewed: dict[str, Overlay] = {}
    generic: list[Overlay] = []
    origin: dict[str, Path] = {}
    twice: dict[str, list[Path]] = {}
    for p in _yaml_files(directory):
        kind = "generic" if p.name.startswith("_") else "overlay"
        try:
            ov = load_overlay(p, variables)
        except Exception as exc:  # noqa: BLE001 - quarantined on its own when a list is given
            if quarantine is None:
                raise
            quarantine.append(_quarantined(p, kind, exc))
            continue
        if kind == "generic":
            generic.append(ov)
            continue
        if ov.server in reviewed or ov.server in twice:
            if quarantine is None:
                raise DescriptorError(p, f"server {ov.server!r} also has an overlay in {origin[ov.server]}")
            twice.setdefault(ov.server, [origin[ov.server]]).append(p)
            continue
        reviewed[ov.server] = ov
        origin[ov.server] = p
    if quarantine is not None:
        _duplicates("overlay", "server", twice, quarantine, reviewed)
    return reviewed, generic


def digest(model: BaseModel | Mapping[str, Any]) -> str:
    """``sha256:<hex>`` of the canonical JSON of a model (aliases applied) or mapping."""
    data = model.model_dump(mode="json", by_alias=True) if isinstance(model, BaseModel) else model
    text = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------- projects


def project_dir(variables: Mapping[str, str] | None = None, environ: Mapping[str, str] | None = None) -> Path | None:
    """The active project's directory: ``env.VBT_PROJECT_DIR`` of ``variables`` (a config's ``tool_env``), else
    ``$VBT_PROJECT_DIR``. None when neither names an existing absolute directory."""
    value = (variables or {}).get(f"env.{PROJECT_ENV}")
    if value is None:
        value = (os.environ if environ is None else environ).get(PROJECT_ENV)
    text = str(value or "").strip()
    if not text:
        return None
    p = Path(text).expanduser()
    return p if p.is_absolute() and p.is_dir() else None


def project_search_dirs(variables: Mapping[str, str] | None = None,
                        environ: Mapping[str, str] | None = None) -> tuple[Path, Path] | None:
    """``(descriptors dir, overlays dir)`` of the active project (searched after the shipped ones), or None."""
    root = project_dir(variables, environ)
    return None if root is None else (root / PROJECT_DESCRIPTORS, root / PROJECT_OVERLAYS)


def project_plugin_files(root: str | Path | None) -> list[str]:
    """The plugin modules of a project directory, ``plugins/<kind>/<name>.py``, sorted by kind and name: the
    ``data.plugins.paths`` entries that make a project's plugins discoverable after the shipped ones."""
    if not root:
        return []
    base = Path(root) / PROJECT_PLUGINS
    if not base.is_dir():
        return []
    return [str(p) for p in sorted(base.glob("*/*.py")) if p.is_file() and not p.name.startswith(("_", "."))]


#: The schema of a project's provenance records (``vbt.projects.ledger.ITEM_SCHEMA``; read here without importing
#: the projects package, which the data child never loads).
_ITEM_SCHEMA = "vbt.project.item/1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def approved_project_plugins(root: str | Path | None, kinds: Iterable[str] | None = None
                             ) -> tuple[list[str], list[str]]:
    """``(files, problems)``: the plugin modules of a project directory that discovery may import, and why the others
    may not. A module ``plugins/<kind>/<name>.py`` is imported only when its provenance record
    (``provenance/plugin/<name>.json``) says ``registered`` for that kind (a plugin waiting for review is not in
    ``plugins/``) and the file's hash is the one recorded: a module written or changed by hand, or by an agent's
    Bash, after the registration (its conformance suite and review) is never imported by the harness or the data
    child. ``kinds`` limits the kinds listed."""
    files: list[str] = []
    problems: list[str] = []
    if not root:
        return files, problems
    base = Path(root)
    wanted = set(kinds) if kinds is not None else None
    for f in project_plugin_files(base):
        p = Path(f)
        kind, name = p.parent.name, p.stem
        if wanted is not None and kind not in wanted:
            continue
        rel = f"{PROJECT_PLUGINS}/{kind}/{p.name}"
        try:
            record = json.loads((base / "provenance" / "plugin" / f"{name}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            problems.append(f"{rel}: no provenance record (not registered through RegisterPlugin): not imported")
            continue
        if not isinstance(record, dict) or record.get("schema") != _ITEM_SCHEMA or record.get("kind") != "plugin" \
                or record.get("plugin_kind") != kind or record.get("status") != "registered":
            problems.append(f"{rel}: its provenance record is not a registered {kind} plugin (status "
                            f"{record.get('status') if isinstance(record, dict) else None!r}): not imported")
            continue
        digest = (record.get("files") or {}).get(rel)
        try:
            actual = _sha256_file(p)
        except OSError as exc:
            problems.append(f"{rel}: unreadable ({exc}): not imported")
            continue
        if not digest or actual != digest:
            problems.append(f"{rel}: changed since it was registered (`vbt project check` lists it): not imported")
            continue
        files.append(str(p))
    return files, problems
