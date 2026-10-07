"""Load descriptors and overlays from YAML, with ``vbt.config`` variable expansion and digests. No pyarrow.

Strings are expanded like ``vbt.config.load_config``: ``${VAR}``, ``${VAR:-x}``, ``${VAR-x}`` and
``${vars.name}`` (``${VAR}`` takes an ``env.VAR`` entry of the variables first, see
:func:`variables_from_config`); additionally ``${run.<name>}`` takes a value from the run mapping passed in
(``${run.mcp_output_dir}`` for tables a tool call materialises during a run, rev 2). A
``${run.*}`` variable without a value is left in place, so the table stays a template until
a run provides it. Overlay files whose name starts with ``_`` are generic overlays.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ValidationError

from ... import config as _config
from .models import SourceDescriptor
from .overlay import Overlay

__all__ = [
    "DescriptorError", "expand", "load_yaml", "load_descriptor", "load_descriptors", "load_overlay", "load_overlays",
    "digest", "variables_from_config", "YAML_SUFFIXES",
]

YAML_SUFFIXES = (".yaml", ".yml")
_VAR = re.compile(r"\$\{([A-Za-z_][\w.]*)(?:(:?-)([^{}]*))?\}")


class DescriptorError(ValueError):
    """A descriptor or overlay file that cannot be read or does not validate."""

    def __init__(self, path: str | Path, message: str) -> None:
        self.path = str(path)
        super().__init__(f"{path}: {message}")


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


def load_descriptors(directory: str | Path, variables: Mapping[str, str] | None = None,
                     run: Mapping[str, Any] | None = None) -> dict[str, SourceDescriptor]:
    """Every ``*.yaml`` descriptor of ``directory`` by source name (a missing directory gives none)."""
    out: dict[str, SourceDescriptor] = {}
    origin: dict[str, Path] = {}
    for p in _yaml_files(directory):
        desc = load_descriptor(p, variables, run)
        if desc.source in out:
            raise DescriptorError(p, f"source {desc.source!r} is also declared in {origin[desc.source]}")
        out[desc.source] = desc
        origin[desc.source] = p
    return out


def load_overlay(path: str | Path, variables: Mapping[str, str] | None = None) -> Overlay:
    data = load_yaml(path, variables)
    try:
        return Overlay.model_validate(data)
    except ValidationError as exc:
        raise DescriptorError(path, _validation_message(exc)) from None


def load_overlays(directory: str | Path,
                  variables: Mapping[str, str] | None = None) -> tuple[dict[str, Overlay], list[Overlay]]:
    """``(reviewed overlays by server, generic overlays)``; files starting with ``_`` are generic."""
    reviewed: dict[str, Overlay] = {}
    generic: list[Overlay] = []
    origin: dict[str, Path] = {}
    for p in _yaml_files(directory):
        ov = load_overlay(p, variables)
        if p.name.startswith("_"):
            generic.append(ov)
            continue
        if ov.server in reviewed:
            raise DescriptorError(p, f"server {ov.server!r} also has an overlay in {origin[ov.server]}")
        reviewed[ov.server] = ov
        origin[ov.server] = p
    return reviewed, generic


def digest(model: BaseModel | Mapping[str, Any]) -> str:
    """``sha256:<hex>`` of the canonical JSON of a model (aliases applied) or mapping."""
    data = model.model_dump(mode="json", by_alias=True) if isinstance(model, BaseModel) else model
    text = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
