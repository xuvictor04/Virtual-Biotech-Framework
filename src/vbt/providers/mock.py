"""Scripted provider for tests and dry runs (no network, no cost).

A script is a callable ``(agent_name, system, messages, tools) -> Message``.
``ScriptedProvider.from_rules`` builds one from simple per-agent queues, which
is enough to exercise delegation, review loops and bulk runs end to end.
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict, deque
from typing import Any, Callable

from .base import (
    LLMProvider,
    Message,
    ModelResponse,
    ModelSettings,
    StopReason,
    TextBlock,
    TextCallback,
    ToolCall,
    ToolSpec,
    Usage,
)

Script = Callable[[str, str, list[Message], list[ToolSpec]], Message]

_ids = itertools.count(1)


def call(name: str, **input: Any) -> ToolCall:
    """Helper to write scripted tool calls."""
    return ToolCall(id=f"call_{next(_ids)}", name=name, input=input)


def reply(*items: str | ToolCall) -> Message:
    return Message("assistant", [TextBlock(i) if isinstance(i, str) else i for i in items])


_AGENT_RE = re.compile(r"<agent-name>([\w\-]+)</agent-name>")


def agent_of(system: str) -> str:
    m = _AGENT_RE.search(system)
    return m.group(1) if m else "unknown"


class ScriptedProvider(LLMProvider):
    name = "mock"

    def __init__(self, script: Script):
        self.script = script
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def from_rules(cls, rules: dict[str, list[Message | Callable[[list[Message]], Message]]],
                   default: str = "Done.") -> "ScriptedProvider":
        """Per-agent FIFO queues of responses; falls back to ``default`` text."""
        queues = defaultdict(deque, {k: deque(v) for k, v in rules.items()})

        def script(agent, system, messages, tools):
            q = queues[agent]
            if not q:
                return reply(default)
            item = q.popleft()
            return item(messages) if callable(item) else item

        return cls(script)

    async def complete(self, *, settings: ModelSettings, system: str, messages: list[Message],
                       tools: list[ToolSpec], on_text: TextCallback | None = None) -> ModelResponse:
        agent = agent_of(system)
        self.calls.append({"agent": agent, "n_messages": len(messages), "tools": [t.name for t in tools]})
        msg = self.script(agent, system, messages, tools)
        if on_text and msg.text:
            r = on_text(msg.text)
            if hasattr(r, "__await__"):
                await r
        stop = StopReason.TOOL_USE if msg.tool_calls else StopReason.END_TURN
        usage = Usage(input_tokens=sum(len(str(m.content)) for m in messages) // 4, output_tokens=len(str(msg.content)) // 4)
        return ModelResponse(message=msg, stop_reason=stop, usage=usage, model=settings.model, cost_usd=0.0)

    def supports_web_search(self) -> bool:
        return True

    async def web_search(self, query: str, *, max_results: int = 8) -> dict[str, Any]:
        return {"results": [{"title": f"Mock result for {query}", "url": "https://example.org"}],
                "summary": f"(mock) no live search for: {query}"}
