"""Runtime: the provider-neutral agent loop, delegation, and shared services.

One ``Runtime`` serves one ``Run``: it owns the provider, the tool registry
(built-ins + provenance + MCP-bridged tools + Task), and spawns agents with
isolated contexts, as the Claude Agent SDK did for the original system.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .agents import AgentDefinition, load_roster, system_prompt
from .config import resolve_path
from .providers import create_provider
from .providers.base import (
    LLMProvider,
    Message,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
)
from .session import Run
from .tools.base import Tool, ToolContext, ToolFailure, ToolRegistry, schema, to_text
from .tools.builtin import builtin_tools
from .tools.mcp_bridge import MCPBridge, MCPServerConfig
from .tools.provenance import provenance_tools

log = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], Any]


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class AgentResult:
    agent: str
    text: str
    messages: list[Message]
    cost_usd: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0
    tool_errors: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"
    delegations: list[str] = field(default_factory=list)
    duration_s: float = 0.0


class Runtime:
    def __init__(self, config: dict[str, Any], run: Run, provider: LLMProvider | None = None,
                 on_event: EventCallback | None = None):
        self.config = config
        self.run = run
        self.provider = provider or create_provider(config["provider"]["name"],
                                                    **(config["provider"].get("options") or {}))
        self.on_event = on_event
        self.registry = ToolRegistry()
        self.tool_call_ids: set[str] = set()
        self.cso, self.agents = load_roster(config)
        limits = config.get("limits", {})
        self.max_turns = int(limits.get("max_agent_turns", 100))
        self.max_depth = int(limits.get("max_delegation_depth", 1))
        self.tool_output_max = int(limits.get("tool_output_max_chars", 40000))
        self._parallel = asyncio.Semaphore(int(limits.get("max_parallel_agents", 8)))
        self.turn_budget_usd = float(limits.get("max_turn_cost_usd") or 0) or None
        self._turn_start_cost = 0.0
        self.read_roots = [resolve_path(p) for p in config["paths"].get("read_roots", []) if p]
        self.skill_roots = [resolve_path(p) for p in config["paths"].get("skills", []) if p]
        self.mcp: MCPBridge | None = None
        self.delegation_log: list[dict[str, Any]] = []
        self.registry.extend(builtin_tools())
        self.registry.extend(provenance_tools())
        self.registry.add(self._task_tool())
        self.registry.add(self._list_tools_tool())

    # ------------------------------------------------------------------ setup

    @property
    def search_backend(self):
        return self.provider.web_search if self.provider.supports_web_search() else None

    def tool_env(self, ctx: ToolContext | None = None) -> dict[str, str]:
        env = {k: str(v) for k, v in (self.config.get("tool_env") or {}).items() if v}
        env.update({"VBT_RUN_DIR": str(self.run.dir), "MCP_OUTPUT_DIR": str(self.run.mcp_output_dir)})
        if ctx is not None:
            env["VBT_AGENT"] = ctx.agent
            env["VBT_WORKSPACE"] = str(ctx.workspace)
        return env

    async def start_mcp(self, servers: list[str] | None = None) -> dict[str, str]:
        """Launch configured MCP servers (optionally a subset). Returns failures."""
        specs = [MCPServerConfig(**{k: v for k, v in s.items() if k in MCPServerConfig.__dataclass_fields__})
                 for s in (self.config.get("mcp_servers") or {}).get("servers", [])]
        self.mcp = MCPBridge(specs, extra_env=self.tool_env())
        tools = await self.mcp.start(set(servers) if servers else None)
        self.registry.extend(tools)
        self.run.trace("mcp_started", servers=sorted(self.mcp.sessions), failures=self.mcp.failures,
                       n_tools=len(tools))
        return self.mcp.failures

    async def aclose(self) -> None:
        if self.mcp:
            await self.mcp.aclose()
        await self.provider.aclose()

    def emit(self, kind: str, **data: Any) -> None:
        if self.on_event:
            try:
                self.on_event(kind, data)
            except Exception:  # noqa: BLE001 - UI callbacks must not break runs
                log.exception("event callback failed")

    def tools_for(self, agent: AgentDefinition) -> list[Tool]:
        tools = self.registry.select(agent.tools)
        if not agent.can_delegate:
            tools = [t for t in tools if t.name != "Task"]
        return tools

    def begin_turn(self) -> None:
        self._turn_start_cost = self.run.cost.total_usd

    def _check_budget(self) -> None:
        if self.turn_budget_usd and self.run.cost.total_usd - self._turn_start_cost > self.turn_budget_usd:
            raise BudgetExceeded(f"turn cost exceeded ${self.turn_budget_usd:.2f}")

    # ------------------------------------------------------------------ agent loop

    async def run_agent(
        self,
        agent: AgentDefinition,
        task: str | None,
        *,
        history: list[Message] | None = None,
        depth: int = 0,
        stream_text: bool = False,
        extra_tools: list[Tool] | None = None,
        allow_tools: bool = True,
        after_end_turn: Callable[[list[Message]], Awaitable[str | None]] | None = None,
    ) -> AgentResult:
        """Run ``agent`` until it answers without tool calls.

        ``history`` is extended in place (append-only). ``after_end_turn`` may
        return a harness message that re-opens the loop (used to enforce review).
        """
        t0 = time.time()
        messages = history if history is not None else []
        if task is not None:
            messages.append(Message.user(task))
        workspace = self.run.agent_dir(agent.name) if agent.name != "cso" else self.run.dir
        system = system_prompt(agent, run_dir=self.run.dir, workspace=workspace, config=self.config,
                               roster=self.agents if agent.can_delegate else None)
        tools = (self.tools_for(agent) + (extra_tools or [])) if allow_tools else []
        by_name = {t.name: t for t in tools}
        settings = agent.settings(self.config)
        result = AgentResult(agent.name, "", messages)
        self.run.trace("agent_start", agent=agent.name, depth=depth, model=settings.model,
                       task=(task or "")[:4000], tools=sorted(by_name))
        self.emit("agent_start", agent=agent.name, depth=depth)

        on_text = (lambda t: self.emit("text", agent=agent.name, text=t)) if stream_text else None
        turns = 0
        while True:
            self._check_budget()
            final_call = turns >= self.max_turns
            if final_call:
                messages.append(Message.user(
                    "[Harness] You have reached the turn limit. Do not call tools; write your final report now."))
            resp: ModelResponse = await self.provider.complete(
                settings=settings, system=system, messages=messages,
                tools=[t.spec for t in tools], on_text=on_text)
            turns += 1
            result.model_calls += 1
            result.cost_usd += resp.cost_usd
            self.run.cost.add(agent.name, resp.usage, resp.cost_usd)
            messages.append(resp.message)
            self.run.trace("model_call", agent=agent.name, model=resp.model, stop=resp.stop_reason.value,
                           usage=resp.usage.as_dict(), cost_usd=round(resp.cost_usd, 6),
                           text=resp.message.text[:20000],
                           thinking=[b.text[:8000] for b in resp.message.content if b.type == "thinking" and b.text])

            calls = resp.message.tool_calls
            if calls and not final_call and resp.stop_reason in (StopReason.TOOL_USE, StopReason.END_TURN):
                results = await asyncio.gather(*(self._execute(c, by_name, agent, depth) for c in calls))
                messages.append(Message("user", list(results)))
                result.tool_calls += len(calls)
                result.tool_errors += [{"tool": c.name, "error": r.content[:500]}
                                       for c, r in zip(calls, results) if r.is_error]
                result.delegations += [c.input.get("subagent_type", "?") for c in calls if c.name == "Task"]
                continue
            if calls and final_call:
                # Keep the history valid (every tool_use answered) without running anything.
                messages.append(Message("user", [ToolResult(c.id, "Not executed: turn limit reached.", True)
                                                 for c in calls]))
                result.stop_reason = "turn_limit"
                result.text = resp.message.text or "[Turn limit reached before a final report was written.]"
                break
            if resp.stop_reason is StopReason.PAUSE:
                continue
            if resp.stop_reason is StopReason.MAX_TOKENS and not final_call:
                if calls:  # truncated tool input: report and let the model retry
                    messages.append(Message("user", [ToolResult(c.id, "Tool input was truncated at max_tokens; "
                                                                  "retry with a shorter input.", True) for c in calls]))
                else:
                    messages.append(Message.user("[Harness] Your response hit the output limit. Continue "
                                                 "exactly where you stopped, concisely."))
                continue
            if resp.stop_reason is StopReason.REFUSAL:
                result.stop_reason = "refusal"
                result.text = (resp.message.text + f"\n\n[The model declined to continue: {resp.stop_detail}]").strip()
                break
            result.text = resp.message.text
            if after_end_turn is not None and not final_call:
                nudge = await after_end_turn(messages)
                if nudge:
                    messages.append(Message.user(nudge))
                    continue
            break

        result.duration_s = round(time.time() - t0, 1)
        self.run.trace("agent_end", agent=agent.name, depth=depth, cost_usd=round(result.cost_usd, 6),
                       model_calls=result.model_calls, tool_calls=result.tool_calls,
                       duration_s=result.duration_s, stop=result.stop_reason, text=result.text[:20000])
        self.emit("agent_end", agent=agent.name, depth=depth, cost_usd=result.cost_usd)
        return result

    async def _execute(self, call: ToolCall, by_name: dict[str, Tool], agent: AgentDefinition,
                       depth: int) -> ToolResult:
        self.tool_call_ids.add(call.id)
        tool = by_name.get(call.name)
        self.run.trace("tool_start", agent=agent.name, tool=call.name, tool_use_id=call.id,
                       input=_clip(call.input))
        self.emit("tool", agent=agent.name, tool=call.name, input=call.input)
        t0 = time.time()
        if tool is None:
            content, err = f"Tool {call.name!r} is not available to {agent.name}.", True
        else:
            ctx = ToolContext(agent=agent.name, run=self.run, runtime=self, tool_call_id=call.id, depth=depth)
            try:
                content, err = to_text(await tool(ctx, call.input)), False
            except ToolFailure as exc:
                content, err = f"Error: {exc}", True
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - surface any tool crash to the model
                log.debug("tool %s crashed", call.name, exc_info=True)
                content, err = f"Error: {type(exc).__name__}: {exc}", True
        if len(content) > self.tool_output_max:
            out_dir = self.run.dir / "work" / "_tool_outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{call.id}.txt"
            path.write_text(content)
            content = (content[: self.tool_output_max] +
                       f"\n\n[Output truncated: {len(content):,} chars. Full output saved to {path}; "
                       f"Read it with offset/limit or load it from code.]")
        self.run.trace("tool_end", agent=agent.name, tool=call.name, tool_use_id=call.id, is_error=err,
                       duration_s=round(time.time() - t0, 2), output=content[:4000])
        return ToolResult(call.id, content, err)

    # ------------------------------------------------------------------ delegation

    def _task_tool(self) -> Tool:
        async def handler(ctx: ToolContext, a: dict[str, Any]) -> str:
            name = a.get("subagent_type", "")
            agent = self.agents.get(name)
            if agent is None:
                raise ToolFailure(f"unknown subagent_type {name!r}; choose from {sorted(self.agents)}")
            if ctx.depth + 1 > self.max_depth:
                raise ToolFailure("delegation depth limit reached; scientists cannot delegate further")
            async with self._parallel:
                self.run.trace("delegation", parent=ctx.agent, agent=name, description=a.get("description", ""),
                               prompt=a.get("prompt", "")[:8000], tool_use_id=ctx.tool_call_id)
                self.emit("delegation", agent=name, description=a.get("description", ""))
                sub = await self.run_agent(agent, a.get("prompt", ""), depth=ctx.depth + 1)
            self.delegation_log.append({"agent": name, "description": a.get("description", ""),
                                        "cost_usd": sub.cost_usd, "tool_errors": sub.tool_errors,
                                        "ts": time.time()})
            failures = ""
            if sub.tool_errors:
                listed = "; ".join(f"{e['tool']}: {e['error'][:160]}" for e in sub.tool_errors[:8])
                failures = f"\n[Tool failures during this delegation: {listed}]"
            return (f"{sub.text}\n\n---\n[{name} finished: {sub.model_calls} model calls, "
                    f"{sub.tool_calls} tool calls, ${sub.cost_usd:.2f}, {sub.duration_s}s. "
                    f"Workspace: {self.run.dir / 'work' / name}]{failures}")

        return Tool(
            "Task",
            "Delegate a task to a specialist agent with its own isolated context and tools. Returns the "
            "specialist's final report. Multiple Task calls in one response run in parallel.",
            schema({"subagent_type": {"type": "string", "description": "specialist name"},
                    "description": {"type": "string", "description": "3-8 word summary"},
                    "prompt": {"type": "string", "description": "complete, self-contained instructions"}},
                   ["subagent_type", "prompt"]),
            handler, source="delegation")

    def _list_tools_tool(self) -> Tool:
        def handler(ctx: ToolContext, a: dict[str, Any]) -> Any:
            inventory: dict[str, list[str]] = {}
            for name in self.registry.names():
                t = self.registry.get(name)
                inventory.setdefault(t.source, []).append(f"{name}: {t.description.splitlines()[0][:160]}")
            failures = self.mcp.failures if self.mcp else {}
            return {"tools_by_source": inventory, "unavailable_servers": failures,
                    "agents": {n: a.description for n, a in self.agents.items()}}
        return Tool("ListTools", "Inventory every data tool, MCP server and specialist available in this run.",
                    schema({}), handler)


def _clip(obj: Any, n: int = 4000) -> Any:
    s = json.dumps(obj, default=str)
    return obj if len(s) <= n else s[:n] + "...(clipped)"
