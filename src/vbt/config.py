"""Configuration loading: YAML files + profiles + environment expansion.

Resolution order (later wins): ``configs/default.yaml`` -> profile files given
with ``--profile`` -> explicit overrides. Strings may reference environment
variables as ``${VAR}`` or ``${VAR:-default}`` and top-level ``vars`` entries
as ``${vars.name}``. Relative paths are resolved against the project root.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

_VAR = re.compile(r"\$\{([A-Za-z_][\w.]*)(?::-([^{}]*))?\}")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            key, default = m.group(1), m.group(2)
            if key.startswith("vars."):
                return str(variables.get(key[5:], default or ""))
            return os.environ.get(key, default if default is not None else "")
        prev = None
        while prev != value:  # allow vars that reference env vars
            prev, value = value, _VAR.sub(sub, value)
        return value
    if isinstance(value, dict):
        return {k: _expand(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, variables) for v in value]
    return value


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _find(name: str) -> Path:
    p = Path(name)
    for cand in (p, CONFIG_DIR / p, CONFIG_DIR / "profiles" / p, CONFIG_DIR / "profiles" / f"{name}.yaml"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"config/profile not found: {name}")


def load_config(profiles: list[str] | None = None, overrides: dict | None = None) -> dict[str, Any]:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:  # pragma: no cover
        pass
    cfg = _load_yaml(CONFIG_DIR / "default.yaml")
    for prof in profiles or []:
        cfg = deep_merge(cfg, _load_yaml(_find(prof)))
    cfg = deep_merge(cfg, overrides or {})

    variables = {"project_root": str(PROJECT_ROOT)}
    variables.update({k: str(v) for k, v in (cfg.get("vars") or {}).items()})
    variables = {k: _expand(v, variables) for k, v in variables.items()}
    cfg = _expand(cfg, variables)
    cfg["vars"] = variables

    for key in ("agents_file", "mcp_servers_file"):
        if cfg.get(key):
            cfg[key.replace("_file", "")] = _expand(_load_yaml(resolve_path(cfg[key])), variables)
    return cfg


def resolve_path(p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
