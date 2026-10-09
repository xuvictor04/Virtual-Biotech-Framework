"""The pinned run configuration: everything needed to explain (and replay) a run.

``build_pinned_config`` is called by ``open_session`` after the MCP servers
started; the result goes to MANIFEST.config and inputs/config.json
(``Run.set_config``). ``vbt replay`` reads it back (profiles, models, web,
orchestration) and compares the commit and prompt hashes with the current
checkout to warn on drift.

Nothing here contacts the network; git and package lookups fail soft.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["build_pinned_config", "git_info", "prompt_hashes", "package_versions", "installed_distributions",
           "PINNED_PACKAGES", "PIN_SCHEMA", "default_model_pattern", "CLAUDE_MODEL_PATTERN", "SERVED_MODEL_PATTERN",
           "LOCAL_PROVIDER_NAMES", "REDACTED", "redact_config", "drop_redacted", "pinned_profiles",
           "served_model_conflict", "pinned_data", "pinned_project"]

PIN_SCHEMA = 1

#: ``--model`` ids accepted for the anthropic provider (when ``provider.model_pattern`` is unset).
CLAUDE_MODEL_PATTERN = r"^claude-[a-z0-9.-]+$"
#: Served model names of local OpenAI-compatible servers: a served name, an HF repo id or a file name.
SERVED_MODEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:@+/-]{0,199}$"
#: Providers that talk to a local OpenAI-compatible server (any served model name is accepted).
LOCAL_PROVIDER_NAMES = frozenset({"vllm", "sglang", "openai_compat", "llamacpp"})

PINNED_PACKAGES = ("anthropic", "httpx", "mcp", "fastmcp", "pydantic", "jsonschema", "numpy", "pandas", "scanpy",
                   "anndata", "statsmodels", "rpy2")

_SECRET_KEY = re.compile(r"(key|token|secret|password|auth)", re.IGNORECASE)
#: Placeholder for a secret in a pinned record (a value, or the user-info of a URL).
REDACTED = "<redacted>"
_URL_USERINFO = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]*@")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes((text or "").encode("utf-8"))


def _redact(value: Any) -> Any:
    """Drop secret-looking keys (provider options, ``web.search.brave_api_key``) and
    the ``user:password@`` part of URLs (a SearxNG or vLLM behind basic auth):
    run records are readable by every agent of the run and travel in audit bundles."""
    if isinstance(value, Mapping):
        return {str(k): (REDACTED if _SECRET_KEY.search(str(k)) and v not in (None, "", False) else _redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    if isinstance(value, str) and "@" in value and "://" in value:
        return _URL_USERINFO.sub(lambda m: m.group(1) + REDACTED + "@", value)
    if callable(value):
        return f"<{type(value).__name__}>"
    return value


def redact_config(value: Any) -> Any:
    """A config section as it may be written to a run record (see :func:`_redact`)."""
    return _redact(value)


def drop_redacted(value: Any) -> Any:
    """A pinned section without its redacted leaves, for re-use as config overrides
    (resume, replay): a ``<redacted>`` key or URL falls back to the current
    config's value (e.g. the Brave key from the environment) instead of being
    used literally."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, str) and (v == REDACTED or f"{REDACTED}@" in v):
                continue
            out[str(k)] = drop_redacted(v)
        return out
    return value


def pinned_profiles(provider_name: str | None, profiles: Iterable[str]) -> list[str]:
    """The profile list to rebuild a recorded run's configuration with.

    A run pinned on the ``anthropic`` provider without the ``claude`` or
    ``paper`` profile was made before the default switched to the local model;
    its provider options were redacted when pinned, so the ``claude`` profile is
    layered first to restore them and reset the local server's options (above
    all ``base_url``) and tier settings. Shared by ``--resume``
    (``cli.build_config``) and ``vbt replay`` so the two cannot drift."""
    out = [str(p) for p in profiles or []]
    if provider_name == "anthropic" and not any(p in ("claude", "paper") for p in out):
        out = ["claude", *out]
    return out


def served_model_conflict(config: Mapping[str, Any], model: str | None) -> str | None:
    """Why ``--model`` would be ignored on the wire, or None.

    For local providers ``provider.options.served_model_name`` is sent for every
    tier, so a different ``--model`` (or web model choice) would be recorded in
    the run while another model actually ran."""
    prov = config.get("provider") or {}
    if not model or prov.get("name") not in LOCAL_PROVIDER_NAMES:
        return None
    served = str(((prov.get("options") or {}).get("served_model_name") or "")).strip()
    if served and str(model) != served:
        return (f"model {model!r} would not be used: provider.options.served_model_name ({served!r}) is "
                "requested from the server for every tier. Remove served_model_name from the config (the tier "
                f"models then go on the wire) or choose {served!r}")
    return None


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


#: The commits an image without ``.git`` was built from (set by deploy/full/Dockerfile).
HARNESS_COMMIT_ENV = "VBT_HARNESS_COMMIT"
UPSTREAM_COMMIT_ENV = "VBT_UPSTREAM_COMMIT"


def git_info(upstream: str | Path | None = None) -> dict[str, Any]:
    """Harness commit and dirty flag, and the upstream submodule commit (without git: the commits the image
    recorded in ``VBT_HARNESS_COMMIT`` / ``VBT_UPSTREAM_COMMIT``, with ``dirty`` unknown)."""
    from .config import PROJECT_ROOT

    info: dict[str, Any] = {"commit": None, "dirty": None, "upstream_commit": None}
    commit = _git(["rev-parse", "HEAD"], PROJECT_ROOT)
    if commit:
        info["commit"] = commit
        status = _git(["status", "--porcelain", "--untracked-files=no"], PROJECT_ROOT)
        info["dirty"] = bool(status) if status is not None else None
    if upstream:
        up = Path(upstream)
        if (up / ".git").exists():
            info["upstream_commit"] = _git(["rev-parse", "HEAD"], up)
    if info["upstream_commit"] is None and commit:
        line = _git(["ls-tree", "HEAD", "third_party/TheVirtualBiotech"], PROJECT_ROOT)
        if line:
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "commit":
                info["upstream_commit"] = parts[2]
                info["upstream_commit_source"] = "submodule pointer"
    # the harness image has no .git: its build records both commits in the environment (deploy/full/Dockerfile)
    if info["commit"] is None and os.environ.get(HARNESS_COMMIT_ENV, "").strip():
        info["commit"] = os.environ[HARNESS_COMMIT_ENV].strip()
        info["commit_source"] = f"environment ({HARNESS_COMMIT_ENV})"
    if info["upstream_commit"] is None and os.environ.get(UPSTREAM_COMMIT_ENV, "").strip():
        info["upstream_commit"] = os.environ[UPSTREAM_COMMIT_ENV].strip()
        info["upstream_commit_source"] = f"environment ({UPSTREAM_COMMIT_ENV})"
    info["version"] = _harness_version(PROJECT_ROOT)
    return info


def _harness_version(root: Path) -> str | None:
    try:
        from importlib.metadata import version
        return version("vbt-harness")
    except Exception:  # noqa: BLE001 - not installed: read pyproject.toml
        pass
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.MULTILINE)
        return m.group(1) if m else None
    except OSError:
        return None


def package_versions(names: Iterable[str] = PINNED_PACKAGES) -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
        except Exception:  # noqa: BLE001
            out[n] = None
    return out


def installed_distributions() -> list[str]:
    """``name==version`` for every installed distribution, sorted (inputs/environment.txt)."""
    from importlib.metadata import distributions

    seen: dict[str, str] = {}
    for d in distributions():
        try:
            name = d.metadata["Name"]
        except Exception:  # noqa: BLE001
            name = None
        if name and name.lower() not in seen:
            seen[name.lower()] = f"{name}=={d.version}"
    return [seen[k] for k in sorted(seen)]


def prompt_hashes(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """sha256 of every upstream ``src/agents/**/*.md`` and every local ``src/vbt/prompts/*.md``."""
    from .config import resolve_path

    out: dict[str, dict[str, str]] = {"upstream": {}, "local": {}}
    up = (config.get("vars") or {}).get("upstream")
    if up:
        root = resolve_path(up) / "src" / "agents"
        if root.is_dir():
            for p in sorted(root.rglob("*.md")):
                try:
                    out["upstream"][p.relative_to(root).as_posix()] = _sha256_bytes(p.read_bytes())
                except OSError:
                    pass
    local = Path(__file__).resolve().parent / "prompts"
    for p in sorted(local.glob("*.md")):
        try:
            out["local"][p.name] = _sha256_bytes(p.read_bytes())
        except OSError:
            pass
    return out


def _tier_effective(tier: Mapping[str, Any]) -> dict[str, Any]:
    t = dict(tier or {})
    thinking = bool(t.get("thinking", True))
    budget = t.get("thinking_budget", t.get("budget_tokens"))
    return {"model": t.get("model"), "effort": t.get("effort"), "thinking": thinking,
            "thinking_budget": budget if thinking else None, "max_tokens": t.get("max_tokens")}


def _agent_entry(agent: Any, config: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
    try:
        s = agent.settings(config)
        model, effort, thinking = s.model, s.effort, s.thinking
        budget = (s.extra or {}).get("thinking_budget")
    except Exception:  # noqa: BLE001
        model = effort = thinking = budget = None
    try:
        tools = [t.name for t in runtime.tools_for(agent)]
    except Exception:  # noqa: BLE001
        tools = list(getattr(agent, "tools", []) or [])
    addenda = list(getattr(agent, "addenda", []) or [])
    return {
        "tier": getattr(agent, "tier", None), "model": model, "effort": effort, "thinking": thinking,
        "thinking_budget": budget if thinking else None, "tools": tools,
        "prompt_ref": getattr(agent, "prompt_ref", "") or None,
        "prompt_sha256": _sha256_text(getattr(agent, "prompt", "") or ""),
        "addenda_sha256": [_sha256_text(a) for a in addenda],
        "max_turns": getattr(agent, "max_turns", None),
        "workspace": getattr(agent, "workspace", None),
    }


def _bash_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    b = dict(config.get("bash") or {})
    pats = list(b.get("blocked_patterns") or [])
    out = {k: v for k, v in b.items() if k != "blocked_patterns" and not isinstance(v, (dict, list))}
    out["blocked_patterns"] = pats
    out["n_blocked_patterns"] = len(pats)
    for k, v in b.items():
        if isinstance(v, (dict, list)) and k != "blocked_patterns":
            out[k] = _redact(v)
    return out


def default_model_pattern(provider_name: str | None) -> str | None:
    """The ``--model`` check when ``provider.model_pattern`` is unset: Claude ids for
    ``anthropic``; any served-model-like name for local OpenAI-compatible servers
    (``qwen3.8-27b``, ``Qwen/Qwen3.8-27B-FP8``, ``model.gguf``); None (no check)
    for other providers."""
    if provider_name == "anthropic":
        return CLAUDE_MODEL_PATTERN
    if provider_name in LOCAL_PROVIDER_NAMES:
        return SERVED_MODEL_PATTERN
    return None


def pinned_data(config: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
    """``pinned["data"]`` (DATA_LAYER.md §15.5): the gateway's ``pinned()`` record (catalog and
    descriptor digests, plugins, sources, readiness, memory), redacted, always with its
    ``determinism`` and ``leakage`` blocks. Without a gateway (data layer disabled, mode
    ``off`` or the gateway failed to build): ``{enabled, mode, gateway: None}``. Never raises."""
    enabled: bool | None = None
    mode: str | None = None
    ceiling: Any = None
    try:
        from .datalayer.settings import DataSettings

        settings = DataSettings.from_config(dict(config or {}))
        enabled, ceiling = settings.enabled, settings.leakage.ceiling
        mode = settings.gateway.mode if settings.enabled else "off"
        prov_dir = settings.provenance.dir
    except Exception as exc:  # noqa: BLE001 - a pin record never blocks a session
        return {"enabled": None, "mode": None, "gateway": None, "error": f"{type(exc).__name__}: {exc}"}
    gateway = getattr(runtime, "gateway", None)
    if gateway is None:
        out: dict[str, Any] = {"enabled": enabled, "mode": mode, "gateway": None, "provenance_dir": prov_dir}
        why = getattr(runtime, "gateway_error", None)
        if why:
            out["reason"] = str(why)[:500]
        return out
    try:
        data = dict(gateway.pinned() or {})
    except Exception as exc:  # noqa: BLE001
        return {"enabled": enabled, "mode": getattr(gateway, "mode", mode), "gateway": None,
                "error": f"{type(exc).__name__}: {exc}"[:500]}
    data = _redact(data)
    data.setdefault("provenance_dir", prov_dir)     # where replay and derived_from find the call records
    data.setdefault("enabled", enabled)
    data.setdefault("mode", getattr(gateway, "mode", mode))
    if not isinstance(data.get("determinism"), Mapping):
        data["determinism"] = {"hash_seed": None, "flags_stripped": []}
    if not isinstance(data.get("leakage"), Mapping):
        data["leakage"] = {"ceiling": ceiling}
    return data


def pinned_project(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """``pinned["project"]`` (docs/PROJECTS.md): the active project's name and directory, and digests of what it
    adds: its catalog (``catalog_sha256`` over its descriptor and overlay files, each listed), its approved plugin
    modules and its registered utilities (their provenance ``source_hash``). None without a project. ``vbt replay``
    and a resumed session activate the same project; the digests say whether it changed since. Never raises."""
    spec = config.get("project")
    if not isinstance(spec, Mapping) or not spec.get("dir"):
        return None
    root = Path(str(spec["dir"]))
    out: dict[str, Any] = {"name": spec.get("name"), "dir": str(root)}
    try:
        from .datalayer.descriptor.load import approved_project_plugins

        files: dict[str, str] = {}
        for sub in ("descriptors", "overlays"):
            d = root / sub
            for f in sorted(d.glob("*.y*ml")) if d.is_dir() else []:
                files[f"{sub}/{f.name}"] = _sha256_bytes(f.read_bytes())
        out["catalog_files"] = files
        out["catalog_sha256"] = _sha256_text("".join(f"{k}\0{v}\0" for k, v in sorted(files.items())))
        plugins, problems = approved_project_plugins(root)
        out["plugins"] = {str(Path(f).relative_to(root)): _sha256_bytes(Path(f).read_bytes()) for f in plugins}
        if problems:
            out["plugins_not_imported"] = problems
        utilities: dict[str, Any] = {}
        udir = root / "provenance" / "utility"
        for f in sorted(udir.glob("*.json")) if udir.is_dir() else []:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(rec, Mapping) and rec.get("status") == "registered":
                utilities[str(rec.get("name") or f.stem)] = rec.get("source_hash")
        out["utilities"] = utilities
    except Exception as exc:  # noqa: BLE001 - a pin record never blocks a session
        out["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return out


def build_pinned_config(config: Mapping[str, Any], runtime: Any, *, interface: str = "chat",
                        profiles: Iterable[str] = (), server: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The pinned configuration of a session (see the module docstring).

    ``server``: the provider's ``server_info()`` (local servers: engine version,
    served models and roots, ``max_model_len``), pinned as ``provider.server``."""
    provider = dict(config.get("provider") or {})
    pname = provider.get("name")
    pattern = provider.get("model_pattern") or default_model_pattern(pname)
    models = {k: dict(v) for k, v in (config.get("models") or {}).items() if isinstance(v, Mapping)}
    orch = dict(config.get("orchestration") or {})
    orch.setdefault("review_policy", "research")
    orch.setdefault("enforce_plan", False)
    orch.setdefault("strategic_orientation", True)
    orch.setdefault("max_review_rounds", 2)

    agents: dict[str, Any] = {}
    cso = getattr(runtime, "cso", None)
    if cso is not None:
        agents["cso"] = _agent_entry(cso, config, runtime)
    for name, a in (getattr(runtime, "agents", None) or {}).items():
        agents[name] = _agent_entry(a, config, runtime)

    mcp = getattr(runtime, "mcp", None)
    if mcp is not None:
        mcp_rec: dict[str, Any] = {"started": sorted(getattr(mcp, "sessions", {}) or {}),
                                   "failures": dict(getattr(mcp, "failures", {}) or {})}
    else:
        mcp_rec = {"started": [], "failures": {}, "enabled": False}
    mcp_rec["configured"] = [s.get("name") for s in ((config.get("mcp_servers") or {}).get("servers") or [])
                             if isinstance(s, Mapping)]

    up = (config.get("vars") or {}).get("upstream")
    pinned: dict[str, Any] = {
        "schema": PIN_SCHEMA,
        "pinned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interface": interface,
        "profiles": list(profiles or config.get("profiles") or []),
        "provider": {"name": pname, "options": _redact(provider.get("options") or {}), "model_pattern": pattern},
        "models": models,
        "effective_models": {k: _tier_effective(v) for k, v in models.items()},
        "model_aliases": dict(config.get("model_aliases") or {}),
        "agent_overrides": dict(config.get("agent_overrides") or {}),
        "agents": agents,
        "limits": dict(config.get("limits") or {}),
        "orchestration": orch,
        "web": _redact(dict(config.get("web") or {})),
        "bash": _bash_summary(config),
        "skill_roots": [str(p) for p in getattr(runtime, "skill_roots", []) or []],
        "read_roots": [str(p) for p in getattr(runtime, "read_roots", []) or []],
        "mcp": mcp_rec,
        "context": dict(config.get("context") or {}),
        "retry": dict(config.get("retry") or {}),
        "audit": dict(config.get("audit") or {}),
        "harness": git_info(up),
        "environment": {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
                        "platform": platform.platform(), "packages": package_versions()},
        "prompt_hashes": prompt_hashes(config),
    }
    if server:
        pinned["provider"]["server"] = _redact(dict(server))
    pinned["data"] = pinned_data(config, runtime)
    project = pinned_project(config)
    if project is not None:
        pinned["project"] = project
    skill_hashes = getattr(runtime, "skill_hashes", None)
    if skill_hashes is not None:
        try:
            pinned["skill_hashes"] = dict(skill_hashes() if callable(skill_hashes) else skill_hashes)
        except Exception as exc:  # noqa: BLE001
            pinned["skill_hashes_error"] = f"{type(exc).__name__}: {exc}"
    return pinned
