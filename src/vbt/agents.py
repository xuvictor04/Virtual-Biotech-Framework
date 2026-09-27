"""Agent definitions (roster) and system-prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, resolve_path
from .providers.base import ModelSettings

LOCAL_PROMPTS = Path(__file__).resolve().parent / "prompts"


@dataclass
class AgentDefinition:
    name: str
    description: str
    prompt: str                     # prompt text (already loaded)
    tier: str = "scientist"
    tools: list[str] = field(default_factory=list)
    division: str = ""
    role: str = ""
    effort: str | None = None
    model: str | None = None        # per-agent model override
    can_delegate: bool = False

    def settings(self, config: dict[str, Any]) -> ModelSettings:
        tier = dict(config["models"][self.tier])
        if self.effort and tier.get("effort") is not None:
            tier["effort"] = self.effort
        if self.model:
            tier["model"] = self.model
        extra = {k: v for k, v in tier.items() if k not in ("model", "effort", "max_tokens", "thinking", "temperature")}
        return ModelSettings(
            provider=config["provider"]["name"],
            model=tier["model"],
            max_tokens=int(tier.get("max_tokens", 32000)),
            effort=tier.get("effort"),
            thinking=bool(tier.get("thinking", True)),
            temperature=tier.get("temperature"),
            extra=extra,
        )


def _flatten(items: list[Any]) -> list[str]:
    out: list[str] = []
    for i in items or []:
        out.extend(_flatten(i) if isinstance(i, list) else [str(i)])
    return list(dict.fromkeys(out))


def load_prompt(ref: str, config: dict[str, Any]) -> str:
    if ref.startswith("local:"):
        path = LOCAL_PROMPTS / ref[6:]
    elif ref.startswith("file:"):
        path = resolve_path(ref[5:])
    else:
        path = resolve_path(config["paths"]["prompts_dir"]) / ref
    if not path.exists():
        raise FileNotFoundError(
            f"system prompt not found: {path}. Did you run `git submodule update --init`?")
    return path.read_text()


def load_roster(config: dict[str, Any]) -> tuple[AgentDefinition, dict[str, AgentDefinition]]:
    spec = config["agents"]
    web = config.get("web", {}).get("enabled", True)

    def build(name: str, d: dict[str, Any], can_delegate: bool = False) -> AgentDefinition:
        tools = _flatten(d.get("tools", []))
        if not web:
            tools = [t for t in tools if t not in ("WebSearch", "WebFetch")]
        return AgentDefinition(
            name=name, description=d.get("description", ""), prompt=load_prompt(d["prompt"], config),
            tier=d.get("tier", "scientist"), tools=tools, division=d.get("division", ""),
            role=d.get("role", name), effort=d.get("effort"), model=d.get("model"), can_delegate=can_delegate,
        )

    cso = build("cso", {**spec["cso"], "role": "Chief Scientific Officer (CSO)"}, can_delegate=True)
    agents = {n: build(n, d) for n, d in (spec.get("agents") or {}).items() if not d.get("disabled")}
    return cso, agents


def _render(template: str, values: dict[str, str]) -> str:
    for k, v in values.items():
        template = template.replace("{" + k + "}", v)
    return template


def system_prompt(agent: AgentDefinition, *, run_dir: Path, workspace: Path, config: dict[str, Any],
                  roster: dict[str, AgentDefinition] | None = None) -> str:
    values = {
        "agent": agent.name,
        "role": agent.role or agent.name,
        "division": agent.division or "n/a",
        "date": date.today().isoformat(),
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "skills": ", ".join(f"`{p}`" for p in config["paths"].get("skills", [])),
        "read_roots": ", ".join(f"`{p}`" for p in config["paths"].get("read_roots", []) if p) or "(none)",
        "web": "enabled" if config.get("web", {}).get("enabled", True) else
               "DISABLED for this run (do not attempt web searches)",
    }
    values["workspace_instruction"] = (
        "" if agent.name in ("cso", "scientific-reviewer")
        else _render((LOCAL_PROMPTS / "workspace_instruction.md").read_text(), values))
    parts = [_render((LOCAL_PROMPTS / "harness_preamble.md").read_text(), values), agent.prompt]
    if agent.can_delegate and roster:
        lines = "\n".join(f"  - `{n}` — {a.description}" for n, a in roster.items())
        parts.append(_render((LOCAL_PROMPTS / "cso_harness_addendum.md").read_text(), {"roster": lines}))
    return "\n\n".join(parts)


__all__ = ["AgentDefinition", "load_roster", "system_prompt", "load_prompt", "PROJECT_ROOT"]
