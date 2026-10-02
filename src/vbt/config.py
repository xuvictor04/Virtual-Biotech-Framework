"""Configuration loading: YAML files + profiles + environment expansion.

Resolution order (later wins): in-code defaults -> ``configs/default.yaml`` ->
profile files given with ``--profile`` -> explicit overrides.

Strings may reference environment variables and top-level ``vars`` entries with
shell-style expansion:

* ``${VAR}`` -- the value, or "" when unset;
* ``${VAR:-default}`` -- ``default`` when VAR is unset **or empty** (so the
  blank entries of ``.env.example`` mean "use the default");
* ``${VAR-default}`` -- ``default`` only when VAR is unset;
* ``${vars.name}`` (with the same ``:-``/``-`` forms) -- a top-level ``vars``
  entry. ``vars.project_root``, ``vars.python`` (``sys.executable``) and
  ``vars.python_prefix`` (``sys.prefix``) are always defined.

Environment files: the project ``.env`` is loaded first, then
``<vars.upstream>/.env`` (the file the upstream README tells users to edit),
both without overriding variables that are already exported.

Relative paths are resolved against the project root.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

_VAR = re.compile(r"\$\{([A-Za-z_][\w.]*)(?:(:?-)([^{}]*))?\}")

#: Defaults for keys that may be missing from ``configs/default.yaml``.
#: Consumers also use ``.get(...)`` with the same values, so hand-built configs work.
CODE_DEFAULTS: dict[str, Any] = {
    "mcp": {
        "start_attempts": 3,          # startup attempts per server
        "start_timeout_s": 180,       # per attempt (connect + initialize + list_tools)
        "start_backoff_s": 3.0,       # delay before the 2nd attempt ...
        "start_backoff_factor": 1.5,  # ... multiplied by this for each further attempt
        "default_timeout_s": 1800,    # per tool call, unless the server sets timeout_s
        "max_restarts": 3,            # crash restarts per server per session
        "inherit_env": False,         # MCP children get an allow-listed env (envpolicy)
    },
    "web": {"literature_max_date": None},   # 'YYYY/MM/DD' ceiling keeps PubMed in no-web runs
    "orchestration": {
        "cso_tools": "restricted",          # restricted | upstream
        "require_reference_data": True,     # preflight before sessions and turns
    },
    "agent_overrides": {},                  # {agent: {effort, model, max_turns, tools_add}}
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _lookup(key: str, variables: dict[str, str]) -> str | None:
    if key.startswith("vars."):
        v = variables.get(key[5:])
    else:
        v = os.environ.get(key)
    return None if v is None else str(v)


def _expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            key, op, default = m.group(1), m.group(2), m.group(3)
            val = _lookup(key, variables)
            if op == ":-":
                return default if not val else val
            if op == "-":
                return default if val is None else val
            return val or ""
        prev = None
        while prev != value:  # allow vars that reference env vars / other vars
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


def load_env_file(path: Path) -> bool:
    """Load ``path`` into ``os.environ`` without overriding exported variables."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a core dependency
        return False
    if not Path(path).is_file():
        return False
    load_dotenv(path, override=False)
    return True


def _variables(cfg: dict[str, Any]) -> dict[str, str]:
    variables = {"project_root": str(PROJECT_ROOT), "python": sys.executable, "python_prefix": sys.prefix}
    variables.update({k: "" if v is None else str(v) for k, v in (cfg.get("vars") or {}).items()})
    # Expand until stable so vars may reference each other in any order.
    for _ in range(5):
        expanded = {k: _expand(v, variables) for k, v in variables.items()}
        if expanded == variables:
            break
        variables = expanded
    if not variables.get("mcp_python", "").strip():
        variables["mcp_python"] = sys.executable
    return variables


def env_files(config: dict[str, Any] | None = None) -> list[Path]:
    """The .env files load_config reads, in order (existing or not)."""
    files = [PROJECT_ROOT / ".env"]
    up = ((config or {}).get("vars") or {}).get("upstream")
    if up:
        files.append(resolve_path(up) / ".env")
    return files


def load_config(profiles: list[str] | None = None, overrides: dict | None = None) -> dict[str, Any]:
    load_env_file(PROJECT_ROOT / ".env")
    cfg = deep_merge(CODE_DEFAULTS, _load_yaml(CONFIG_DIR / "default.yaml"))
    for prof in profiles or []:
        cfg = deep_merge(cfg, _load_yaml(_find(prof)))
    cfg = deep_merge(cfg, overrides or {})

    variables = _variables(cfg)
    if variables.get("upstream") and load_env_file(resolve_path(variables["upstream"]) / ".env"):
        variables = _variables(cfg)  # the upstream .env may define variables the vars use
    cfg = _expand(cfg, variables)
    cfg["vars"] = variables

    for key in ("agents_file", "mcp_servers_file"):
        if cfg.get(key):
            cfg[key.replace("_file", "")] = _expand(_load_yaml(resolve_path(cfg[key])), variables)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Absolute, symlink-resolved path; relative paths are taken from the project root."""
    path = Path(p).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
