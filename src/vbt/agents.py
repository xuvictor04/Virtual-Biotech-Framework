"""Agent definitions (roster) and system-prompt assembly.

System prompts are built in two parts so provider prompt caches can reuse the
large, unchanging prefix across turns, delegations, bulk items and runs:

* **stable** -- the upstream prompt, per-agent addenda, the role-aware harness
  rules and (for the CSO) the CSO addendum. It contains no date and no
  run-specific paths.
* **volatile** -- the "Session" block: date, run directory, workspace,
  absolute skill and reference-data roots, the environment note, unavailable
  data servers and the agent's project memory.

``system_prompt_parts`` returns both; ``system_prompt`` joins them for callers
that send a single string.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import PROJECT_ROOT, resolve_path
from .providers.base import ModelSettings

log = logging.getLogger(__name__)

LOCAL_PROMPTS = Path(__file__).resolve().parent / "prompts"

WEB_TOOLS = ("WebSearch", "WebFetch")
PUBMED_PREFIX = "mcp__pubmed__"
#: Tools that let an agent create files; agents without any of them get no
#: workspace instruction and the read-only rules.
WRITE_TOOLS = ("Write", "Edit", "Bash", "NotebookEdit")
#: Added to the CSO when ``orchestration.cso_tools: upstream`` (upstream run.py parity).
UPSTREAM_CSO_EXTRA = ("Bash", "Write", "Edit", "NotebookEdit", "WebFetch", "WebSearch")
CSO_TOOL_MODES = ("restricted", "upstream")
MEMORY_TOOL = "UpdateMemory"
MEMORY_MODES = ("project", "none")
WORKSPACE_MODES = ("agent", "run")
MEMORY_LINES = 200
REVIEW_POLICIES = ("always", "research", "multi_specialist", "never")


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
    max_turns: int | None = None    # per-agent model-call cap (None: runtime default)
    memory: str = "project"         # 'project': per-run MEMORY.md injected + UpdateMemory; 'none'
    workspace: str = "agent"        # 'agent': work/<name>/ ; 'run': the run directory (CSO, reviewer)
    addenda: list[str] = field(default_factory=list)   # prompt texts appended after ``prompt``
    prompt_ref: str = ""            # where ``prompt`` came from (for pinning/provenance)

    def __post_init__(self) -> None:
        if self.memory not in MEMORY_MODES:
            raise ValueError(f"agent {self.name}: memory must be one of {MEMORY_MODES}, got {self.memory!r}")
        if self.workspace not in WORKSPACE_MODES:
            raise ValueError(f"agent {self.name}: workspace must be one of {WORKSPACE_MODES}, "
                             f"got {self.workspace!r}")

    def has_tool(self, name: str) -> bool:
        """True when ``name`` matches one of this agent's tool patterns."""
        return any(fnmatch.fnmatchcase(name, pat) for pat in self.tools)

    @property
    def can_write(self) -> bool:
        return any(self.has_tool(t) for t in WRITE_TOOLS)

    @property
    def uses_memory(self) -> bool:
        return self.memory == "project" and self.has_tool(MEMORY_TOOL)

    def settings(self, config: dict[str, Any]) -> ModelSettings:
        tier = dict(config["models"][self.tier])
        if self.effort and tier.get("effort") is not None:
            tier["effort"] = self.effort
        if self.model:
            tier["model"] = self.model
        extra = {k: v for k, v in tier.items() if k not in ("model", "effort", "max_tokens", "thinking", "temperature")}
        # Consumed locally by providers (the scripted mock routes on it); never sent to an API.
        extra["agent_name"] = self.name
        return ModelSettings(
            provider=config["provider"]["name"],
            model=tier["model"],
            max_tokens=int(tier.get("max_tokens", 32000)),
            effort=tier.get("effort"),
            thinking=bool(tier.get("thinking", True)),
            temperature=tier.get("temperature"),
            extra=extra,
        )


def _flatten(items: Any) -> list[str]:
    if items is None:
        return []
    if not isinstance(items, list):
        items = [items]
    out: list[str] = []
    for i in items:
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


def filter_web_tools(tools: Iterable[str], *, web: bool, literature_max_date: str | None = None) -> list[str]:
    """Remove web access from a tool allowlist when web is disabled.

    WebSearch/WebFetch are always removed. PubMed (``mcp__pubmed__*``) is a
    live literature source, so it is removed too unless a publication-date
    ceiling (``web.literature_max_date``) is configured, in which case the
    PubMed server enforces it (see ``mcp_servers/pubmed_server.py``).
    """
    tools = list(tools)
    if web:
        return tools
    out = [t for t in tools if t not in WEB_TOOLS]
    if not literature_max_date:
        out = [t for t in out if not t.startswith(PUBMED_PREFIX)]
    return out


def _int_or_none(v: Any) -> int | None:
    if v is None or v == "":
        return None
    return int(v)


def load_roster(config: dict[str, Any]) -> tuple[AgentDefinition, dict[str, AgentDefinition]]:
    spec = config["agents"]
    web_cfg = config.get("web") or {}
    web = web_cfg.get("enabled", True)
    lit_ceiling = web_cfg.get("literature_max_date")
    cso_mode = (config.get("orchestration") or {}).get("cso_tools", "restricted") or "restricted"
    if cso_mode not in CSO_TOOL_MODES:
        raise ValueError(f"orchestration.cso_tools must be one of {CSO_TOOL_MODES}, got {cso_mode!r}")
    overrides: Mapping[str, Any] = config.get("agent_overrides") or {}

    def build(name: str, d: dict[str, Any], can_delegate: bool = False) -> AgentDefinition:
        ov = overrides.get(name) or {}
        tools = _flatten(d.get("tools", []))
        if can_delegate and cso_mode == "upstream":
            tools += list(UPSTREAM_CSO_EXTRA)
        tools += _flatten(ov.get("tools_add", []))
        memory = d.get("memory", "project")
        if memory == "project" and MEMORY_TOOL not in tools:
            tools.append(MEMORY_TOOL)
        tools = list(dict.fromkeys(filter_web_tools(tools, web=web, literature_max_date=lit_ceiling)))
        addenda_refs = _flatten(d.get("addenda", []))
        return AgentDefinition(
            name=name, description=d.get("description", ""), prompt=load_prompt(d["prompt"], config),
            prompt_ref=d["prompt"],
            tier=ov.get("tier", d.get("tier", "scientist")), tools=tools, division=d.get("division", ""),
            role=d.get("role", name), effort=ov.get("effort", d.get("effort")),
            model=ov.get("model", d.get("model")), can_delegate=can_delegate,
            max_turns=_int_or_none(ov.get("max_turns", d.get("max_turns"))),
            memory=memory, workspace=d.get("workspace", "run" if can_delegate else "agent"),
            addenda=[load_prompt(r, config) for r in addenda_refs],
        )

    cso = build("cso", {**spec["cso"], "role": "Chief Scientific Officer (CSO)"}, can_delegate=True)
    agents = {n: build(n, d) for n, d in (spec.get("agents") or {}).items() if not d.get("disabled")}
    unknown = sorted(set(overrides) - set(agents) - {"cso"})
    if unknown:
        log.warning("agent_overrides for unknown agents ignored: %s", ", ".join(unknown))
    return cso, agents


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_ROLE_BLOCK = re.compile(r"<!-- role:([\w,\-]+) -->\n(.*?)<!-- /role -->\n?", re.DOTALL)


def _today() -> str:  # separate function so tests can pin the date
    return date.today().isoformat()


def _render(template: str, values: Mapping[str, str]) -> str:
    for k, v in values.items():
        template = template.replace("{" + k + "}", v)
    return template


def _select_role_blocks(template: str, kind: str) -> str:
    """Keep ``<!-- role:a,b -->...<!-- /role -->`` blocks whose list contains ``kind``."""
    return _ROLE_BLOCK.sub(lambda m: m.group(2) if kind in m.group(1).split(",") else "", template)


def role_kind(agent: AgentDefinition) -> str:
    """'cso' (delegates), 'specialist' (can write files) or 'readonly' (e.g. reviewer, bulk annotator)."""
    if agent.can_delegate:
        return "cso"
    return "specialist" if agent.can_write else "readonly"


def review_policy(config: Mapping[str, Any]) -> str:
    orch = config.get("orchestration") or {}
    if orch.get("enforce_review") is False:
        return "never"
    policy = orch.get("review_policy") or "research"
    return policy if policy in REVIEW_POLICIES else "research"


_REVIEW_TEXT = {
    "always": (
        "The harness enforces scientific review: whenever you delegated analyses this turn, the "
        "`scientific-reviewer` must evaluate their outputs before your final synthesis."),
    "research": (
        "The harness enforces scientific review on research turns -- two or more specialists, or any "
        "specialist that wrote files: the `scientific-reviewer` must evaluate those outputs before your "
        "final synthesis. Single-specialist lookups that produce no files do not need review."),
    "multi_specialist": (
        "The harness enforces scientific review when two or more specialists contributed this turn: the "
        "`scientific-reviewer` must evaluate their outputs before your final synthesis."),
    "never": (
        "The harness does not enforce review in this run; follow your instructions on when to call the "
        "`scientific-reviewer`."),
}

_CSO_TOOLS_TEXT = {
    "restricted": (
        "Your tool set is deliberately restricted in this harness: `Bash`, `Write`, `Edit`, "
        "`NotebookEdit`, `WebSearch` and `WebFetch` are unavailable to you even where your instructions "
        "mention them. Browse specialist outputs with `Read`, `Glob` and `Grep` (instead of `ls`/`head`), "
        "and delegate anything that needs code, data or the web."),
    "upstream": (
        "You have the upstream CSO tool set (including `Bash`, `Write`, `Edit` and the web tools where "
        "web access is enabled). Use them only to browse specialist outputs and manage session files -- "
        "never for analysis."),
}


def _abs_paths(paths: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for p in paths or []:
        if p is None or not str(p).strip():
            continue
        r = str(resolve_path(str(p)))
        if r not in out:
            out.append(r)
    return out


def _fmt_roots(paths: list[str]) -> str:
    if not paths:
        return "(none)"
    return ", ".join(f"`{p}`" + ("" if Path(p).exists() else " (not found)") for p in paths)


def memory_path(run_dir: Path, agent: str) -> Path:
    return Path(run_dir) / "memory" / agent / "MEMORY.md"


def _memory_block(agent: AgentDefinition, run_dir: Path) -> str:
    path = memory_path(run_dir, agent.name)
    notes = ""
    if path.is_file():
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        notes = "\n".join(lines[:MEMORY_LINES]).strip()
        if len(lines) > MEMORY_LINES:
            notes += f"\n... ({len(lines) - MEMORY_LINES} more lines in {path})"
    body = (f"Notes recorded by earlier `{agent.name}` instances in this run (first {MEMORY_LINES} lines of "
            f"`{path}`):\n<memory>\n{notes}\n</memory>" if notes else
            f"No notes yet: you are the first `{agent.name}` instance in this run to record any.")
    return (f"## Your memory\n\n{body}\n\nBefore you finish, call `{MEMORY_TOOL}` to record what a later "
            f"instance of your role should know: key findings (with numbers), the file paths you produced, "
            f"data quirks, and approaches that failed. Keep it brief and do not repeat existing notes.")


def _unavailable_text(unavailable: Mapping[str, Any] | Iterable[str] | None) -> str:
    if not unavailable:
        return ""
    items = unavailable.items() if isinstance(unavailable, Mapping) else ((n, "") for n in unavailable)
    parts = []
    for name, why in items:
        why = str(why or "").strip().splitlines()[0][:160] if str(why or "").strip() else ""
        parts.append(f"`{name}`" + (f" ({why})" if why else ""))
    return ("- Unavailable data servers: " + "; ".join(parts) + ". Their `mcp__<server>__*` tools are "
            "missing or failing in this run: report the gap rather than substituting silently.")


def system_prompt_parts(agent: AgentDefinition, *, run_dir: Path, workspace: Path, config: dict[str, Any],
                        roster: Mapping[str, AgentDefinition] | None = None,
                        unavailable_servers: Mapping[str, Any] | Iterable[str] | None = None) -> tuple[str, str]:
    """Return ``(stable, volatile)`` system-prompt parts for ``agent``."""
    kind = role_kind(agent)
    web_cfg = config.get("web") or {}
    if web_cfg.get("enabled", True):
        web = "enabled"
    else:
        web = "DISABLED for this run (do not attempt web searches)"
        if web_cfg.get("literature_max_date"):
            web += (f"; PubMed results are limited to publications up to "
                    f"{web_cfg['literature_max_date']}")
    values = {
        "agent": agent.name,
        "role": agent.role or agent.name,
        "division": agent.division or "n/a",
        "web": web,
    }

    # ----------------------------------------------------------- stable part
    preamble = _select_role_blocks((LOCAL_PROMPTS / "harness_preamble.md").read_text(), kind)
    values["workspace_instruction"] = (
        _render((LOCAL_PROMPTS / "workspace_instruction.md").read_text(), values)
        if kind == "specialist" and agent.workspace == "agent" else "")
    stable_parts = [agent.prompt.strip(), *(a.strip() for a in agent.addenda if a.strip()),
                    _render(preamble, values).strip()]
    if agent.can_delegate and roster:
        lines = "\n".join(f"  - `{n}` -- {a.description}" for n, a in roster.items())
        cso_mode = (config.get("orchestration") or {}).get("cso_tools", "restricted") or "restricted"
        stable_parts.append(_render((LOCAL_PROMPTS / "cso_harness_addendum.md").read_text(), {
            "roster": lines,
            "review_policy_text": _REVIEW_TEXT[review_policy(config)],
            "cso_tools_text": _CSO_TOOLS_TEXT.get(cso_mode, _CSO_TOOLS_TEXT["restricted"]),
        }).strip())
    stable = "\n\n".join(p for p in stable_parts if p)

    # --------------------------------------------------------- volatile part
    run_dir = Path(run_dir)
    paths = config.get("paths") or {}
    skills = _abs_paths(paths.get("skills", []))
    run_skills = run_dir / ".claude" / "skills"
    read_roots = _abs_paths(paths.get("read_roots", []))
    ws_line = (f"- Your workspace: `{workspace}`" + (" (the run directory)" if Path(workspace) == run_dir else ""))
    lines = [
        "# Session",
        "",
        f"- Today's date: {_today()}",
        f"- Run directory (shared by all agents in this run): `{run_dir}`",
        ws_line,
        f"- Skills (load one by name with the `Skill` tool; supporting files are readable here): "
        f"{_fmt_roots(skills)}; this run's copy: `{run_skills}`",
        f"- Reference data (read-only): {_fmt_roots(read_roots)}",
        f"- Installed packages: `{run_dir / 'inputs' / 'environment.txt'}` (the upstream prompts call this "
        f"file `environment_full.yml`; read this one instead)",
    ]
    unavailable = _unavailable_text(unavailable_servers)
    if unavailable:
        lines.append(unavailable)
    volatile_parts = ["\n".join(lines)]
    if agent.uses_memory:
        volatile_parts.append(_memory_block(agent, run_dir))
    if (config.get("provider") or {}).get("name") == "mock":
        # Routing tag for the scripted mock provider only; real prompts never carry it.
        volatile_parts.append(f"<agent-name>{agent.name}</agent-name>")
    volatile = "\n\n".join(volatile_parts)
    return stable, volatile


def system_prompt(agent: AgentDefinition, *, run_dir: Path, workspace: Path, config: dict[str, Any],
                  roster: Mapping[str, AgentDefinition] | None = None,
                  unavailable_servers: Mapping[str, Any] | Iterable[str] | None = None) -> str:
    """Single-string system prompt: the stable part, then the volatile part."""
    stable, volatile = system_prompt_parts(agent, run_dir=run_dir, workspace=workspace, config=config,
                                           roster=roster, unavailable_servers=unavailable_servers)
    return stable + "\n\n" + volatile


__all__ = [
    "AgentDefinition", "load_roster", "system_prompt", "system_prompt_parts", "load_prompt",
    "filter_web_tools", "role_kind", "memory_path", "PROJECT_ROOT",
]
