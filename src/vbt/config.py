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
    # "data": a copy of vbt.datalayer.settings.DATA_DEFAULTS (docs/DATA_LAYER.md section 17), added
    # by _install_data_defaults() at the end of this module (the settings module imports this one)
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


class ProfileError(ValueError):
    """A profile cannot be used with the resulting configuration (``requires_provider``)."""


#: Profile-only key: the provider a profile is written for (checked after every layer is merged).
REQUIRES_PROVIDER_KEY = "requires_provider"


def split_profile(raw: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """``(profile data, required provider)``: the ``requires_provider`` meta key
    is not configuration and is removed from the data."""
    data = dict(raw or {})
    req = data.pop(REQUIRES_PROVIDER_KEY, None)
    return data, (str(req) if req else None)


def check_profile_requirement(cfg: dict[str, Any], profile: str, required: str | None) -> None:
    """Raise :class:`ProfileError` when ``profile`` needs another provider than ``cfg`` has
    (e.g. ``upstream-web``, which only adjusts Claude's effort levels, used alone on the
    local default would silently run the local model)."""
    if not required:
        return
    name = (cfg.get("provider") or {}).get("name")
    if name != required:
        raise ProfileError(
            f"profile {profile!r} is written for the {required!r} provider, but the configuration uses "
            f"{name!r}: combine it with a profile that selects {required} (e.g. `--profile claude "
            f"--profile {profile}` or `--profile paper --profile {profile}`)")


def load_config(profiles: list[str] | None = None, overrides: dict | None = None) -> dict[str, Any]:
    load_env_file(PROJECT_ROOT / ".env")
    if "data" not in CODE_DEFAULTS:
        _install_data_defaults()
    cfg = deep_merge(CODE_DEFAULTS, _load_yaml(CONFIG_DIR / "default.yaml"))
    required: list[tuple[str, str | None]] = []
    for prof in profiles or []:
        data, req = split_profile(_load_yaml(_find(prof)))
        required.append((str(prof), req))
        cfg = deep_merge(cfg, data)
    cfg = deep_merge(cfg, overrides or {})
    for prof, req in required:
        check_profile_requirement(cfg, prof, req)

    variables = _variables(cfg)
    if variables.get("upstream") and load_env_file(resolve_path(variables["upstream"]) / ".env"):
        variables = _variables(cfg)  # the upstream .env may define variables the vars use
    cfg = _expand(cfg, variables)
    cfg["vars"] = variables

    for key in ("agents_file", "mcp_servers_file"):
        if cfg.get(key):
            cfg[key.replace("_file", "")] = _expand(_load_yaml(resolve_path(cfg[key])), variables)
    return cfg


LITERATURE_MAXDATE_ENV = "VBT_LITERATURE_MAXDATE"


def base_tool_env(config: dict[str, Any]) -> dict[str, str]:
    """The harness-level environment every MCP server / tool child gets.

    ``config['tool_env']`` (non-empty values) plus derived settings. The PubMed
    publication-date ceiling has a single source of truth,
    ``web.literature_max_date``: it is what keeps PubMed in no-web runs and what
    the agents' prompts announce, so it is always exported to the server as
    ``VBT_LITERATURE_MAXDATE`` (overriding a disagreeing tool_env value).
    """
    env = {k: str(v) for k, v in (config.get("tool_env") or {}).items() if v}
    ceiling = (config.get("web") or {}).get("literature_max_date")
    if ceiling:
        ceiling = str(ceiling).strip()
        prev = env.get(LITERATURE_MAXDATE_ENV)
        if prev and prev.strip() != ceiling:
            log.warning("tool_env.%s=%s disagrees with web.literature_max_date=%s; using the latter",
                        LITERATURE_MAXDATE_ENV, prev, ceiling)
        env[LITERATURE_MAXDATE_ENV] = ceiling
    return env


def resolve_path(p: str | Path) -> Path:
    """Absolute, symlink-resolved path; relative paths are taken from the project root."""
    path = Path(p).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _install_data_defaults() -> None:
    """``CODE_DEFAULTS["data"]``: the data layer's in-code defaults (``DataSettings`` carries the same
    values, so hand-built configs work). ``vbt.datalayer.settings`` imports this module; when it is
    the one being imported first, its defaults are not defined yet and ``load_config`` installs them."""
    mod = sys.modules.get("vbt.datalayer.settings")
    if mod is not None and not hasattr(mod, "DATA_DEFAULTS"):
        return
    from .datalayer.settings import DATA_DEFAULTS
    CODE_DEFAULTS["data"] = copy.deepcopy(DATA_DEFAULTS)


_install_data_defaults()
