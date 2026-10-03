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
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["build_pinned_config", "git_info", "prompt_hashes", "package_versions", "installed_distributions",
           "PINNED_PACKAGES", "PIN_SCHEMA"]

PIN_SCHEMA = 1

PINNED_PACKAGES = ("anthropic", "mcp", "fastmcp", "pydantic", "numpy", "pandas", "scanpy", "anndata",
                   "statsmodels", "rpy2")

_SECRET_KEY = re.compile(r"(key|token|secret|password|auth)", re.IGNORECASE)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes((text or "").encode("utf-8"))


def _redact(value: Any) -> Any:
    """Drop secret-looking keys (provider options never need them on record)."""
    if isinstance(value, Mapping):
        return {str(k): ("<redacted>" if _SECRET_KEY.search(str(k)) and v not in (None, "", False) else _redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    if callable(value):
        return f"<{type(value).__name__}>"
    return value


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info(upstream: str | Path | None = None) -> dict[str, Any]:
    """Harness commit and dirty flag, and the upstream submodule commit."""
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


def build_pinned_config(config: Mapping[str, Any], runtime: Any, *, interface: str = "chat",
                        profiles: Iterable[str] = ()) -> dict[str, Any]:
    """The pinned configuration of a session (see the module docstring)."""
    provider = dict(config.get("provider") or {})
    pname = provider.get("name")
    pattern = provider.get("model_pattern") or (r"^claude-[a-z0-9.-]+$" if pname == "anthropic" else None)
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
        "web": dict(config.get("web") or {}),
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
    skill_hashes = getattr(runtime, "skill_hashes", None)
    if skill_hashes is not None:
        try:
            pinned["skill_hashes"] = dict(skill_hashes() if callable(skill_hashes) else skill_hashes)
        except Exception as exc:  # noqa: BLE001
            pinned["skill_hashes_error"] = f"{type(exc).__name__}: {exc}"
    return pinned
