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
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ValidationError

from ... import config as _config
from .models import SourceDescriptor
from .overlay import Overlay

__all__ = [
    "DescriptorError", "Quarantined", "expand", "load_yaml", "load_descriptor", "load_descriptors", "load_overlay",
    "load_overlays", "digest", "variables_from_config", "YAML_SUFFIXES",
]

YAML_SUFFIXES = (".yaml", ".yml")
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
    a descriptor declares when its YAML parses (None: unknown, so any unresolved reference may be its)."""

    path: str
    kind: str
    name: str | None
    error: str
    tables: tuple[str, ...] | None = None
    id_types: tuple[str, ...] | None = None

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
    return Quarantined(str(path), kind, name, error)


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
