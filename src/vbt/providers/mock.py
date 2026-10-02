"""Scripted provider for tests and dry runs (no network, no cost).

A script is a callable ``(agent_name, system, messages, tools) -> item`` where
``item`` is one of:

* a :class:`Message` (the assistant reply);
* a :class:`Turn` built with :func:`turn` (reply plus stop reason, usage,
  thinking text and cost overrides);
* a :class:`Fail` built with :func:`fail` (the call raises that exception, e.g.
  a :class:`RetryableProviderError` to exercise retry paths).

``ScriptedProvider.from_rules`` builds a script from per-agent FIFO queues whose
entries are any of the above or callables ``(messages) -> item``; that is
enough to exercise delegation, review loops and bulk runs end to end.

The agent is identified from ``settings.extra["agent_name"]`` first and falls
back to an ``<agent-name>`` tag in the system prompt.

``strict=True`` (the default) rejects histories that break tool_use/tool_result
pairing, mirroring the 400 the real API returns, so tests catch history
corruption.
"""

from __future__ import annotations

import inspect
import itertools
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Union

from .base import (
    Block,
    LLMProvider,
    Message,
    ModelResponse,
    ModelSettings,
    ProviderCapabilities,
    ProviderError,
    RetryableProviderError,
    StopReason,
    SystemSegment,
    TextBlock,
    TextCallback,
    ThinkingBlock,
    ThinkingCallback,
    ToolCall,
    ToolSpec,
    Usage,
    system_text,
    validate_tool_pairing,
)

_ids = itertools.count(1)


@dataclass
class Turn:
    """A scripted model turn with optional overrides."""

    message: Message
    stop: StopReason | None = None
    usage: Usage | None = None
    thinking: str | None = None
    cost_usd: float = 0.0


@dataclass
class Fail:
    """A scripted failure: the provider call raises ``exc``."""

    exc: BaseException | type[BaseException]

    def raise_(self) -> None:
        exc = self.exc
        if isinstance(exc, type):
            exc = exc()
        raise exc


ScriptItem = Union[Message, Turn, Fail]
Script = Callable[[str, str, list[Message], list[ToolSpec]], ScriptItem]
UsageFn = Callable[[str, list[Message], Message], Usage]


def call(name: str, **input: Any) -> ToolCall:
    """Helper to write scripted tool calls."""
    return ToolCall(id=f"call_{next(_ids)}", name=name, input=input)


def reply(*items: str | Block) -> Message:
    return Message("assistant", [TextBlock(i) if isinstance(i, str) else i for i in items])


def turn(*items: str | Block, stop: StopReason | str | None = None, usage: Usage | None = None,
         thinking: str | None = None, cost_usd: float = 0.0) -> Turn:
    """A scripted turn. ``thinking`` adds a leading ThinkingBlock (streamed via
    ``on_thinking``); ``stop`` overrides the inferred stop reason."""
    msg = reply(*items)
    if thinking:
        msg.content.insert(0, ThinkingBlock(thinking, provider="mock"))
    if isinstance(stop, str):
        stop = StopReason(stop)
    return Turn(msg, stop=stop, usage=usage, thinking=thinking, cost_usd=cost_usd)


def fail(exc: BaseException | type[BaseException] | None = None) -> Fail:
    """A scripted failure (default: a transient 529-style RetryableProviderError)."""
    return Fail(exc if exc is not None else RetryableProviderError("mock: overloaded", status=529))


_AGENT_RE = re.compile(r"<agent-name>([\w\-]+)</agent-name>")


def agent_of(system: str | list[SystemSegment]) -> str:
    m = _AGENT_RE.search(system_text(system))
    return m.group(1) if m else "unknown"


async def _emit(cb: TextCallback | None, text: str) -> None:
    if cb is None or not text:
        return
    r = cb(text)
    if inspect.isawaitable(r):
        await r


def _default_usage(agent: str, messages: list[Message], msg: Message) -> Usage:
    return Usage(input_tokens=sum(len(str(m.content)) for m in messages) // 4, output_tokens=len(str(msg.content)) // 4)


class ScriptedProvider(LLMProvider):
    name = "mock"

    def __init__(self, script: Script, *, strict: bool = True, usage_fn: UsageFn | None = None,
                 context_window: int | None = None, capabilities: ProviderCapabilities | None = None) -> None:
        self.script = script
        self.strict = strict
        self.usage_fn = usage_fn
        self._context_window = context_window
        self._capabilities = capabilities or ProviderCapabilities(
            images=True, documents=True, web_search=True, server_context_management=False)
        self.calls: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []

    @classmethod
    def from_rules(cls, rules: dict[str, list[Any]], default: str = "Done.", **kwargs: Any) -> "ScriptedProvider":
        """Per-agent FIFO queues of responses; falls back to ``default`` text.

        Entries: Message | Turn | Fail | callable(messages) -> one of those.
        ``kwargs`` go to the constructor (strict, usage_fn, context_window, capabilities).
        """
        queues: dict[str, deque] = defaultdict(deque, {k: deque(v) for k, v in rules.items()})

        def script(agent, system, messages, tools):
            q = queues[agent]
            if not q:
                return reply(default)
            item = q.popleft()
            return item(messages) if callable(item) else item

        return cls(script, **kwargs)

    # ------------------------------------------------------------------ interface

    def capabilities(self, model: str | None = None) -> ProviderCapabilities:
        return self._capabilities

    def context_window(self, model: str) -> int | None:
        return self._context_window

    def check_credentials(self) -> str | None:
        return None

    async def complete(self, *, settings: ModelSettings, system: str | list[SystemSegment], messages: list[Message],
                       tools: list[ToolSpec], on_text: TextCallback | None = None,
                       on_thinking: ThinkingCallback | None = None) -> ModelResponse:
        sys_text = system_text(system)
        agent = (settings.extra or {}).get("agent_name") or agent_of(sys_text)
        if self.strict:
            problems = validate_tool_pairing(messages)
            if problems:
                raise ProviderError("unpaired tool_use/tool_result in history (the API would reject this request "
                                    "with a 400): " + "; ".join(problems[:5]))
        self.calls.append({"agent": agent, "n_messages": len(messages), "tools": [t.name for t in tools],
                           "model": settings.model, "thinking": settings.thinking, "max_tokens": settings.max_tokens})
        item = self.script(agent, sys_text, messages, tools)
        if isinstance(item, Fail):
            item.raise_()
        t = item if isinstance(item, Turn) else Turn(item)
        msg = t.message
        if not isinstance(msg, Message):
            raise TypeError(f"mock script returned {type(msg).__name__}, expected Message, Turn or Fail")
        for b in msg.content:
            if isinstance(b, ThinkingBlock):
                await _emit(on_thinking, b.text)
        await _emit(on_text, msg.text)
        stop = t.stop or (StopReason.TOOL_USE if msg.tool_calls else StopReason.END_TURN)
        usage = t.usage or (self.usage_fn or _default_usage)(agent, messages, msg)
        return ModelResponse(message=msg, stop_reason=stop, usage=usage, model=settings.model, cost_usd=t.cost_usd,
                             served_model=settings.model)

    def supports_web_search(self) -> bool:
        return True

    async def web_search(self, query: str, *, max_results: int = 8, allowed_domains: list[str] | None = None,
                         blocked_domains: list[str] | None = None) -> dict[str, Any]:
        self.searches.append({"query": query, "max_results": max_results, "allowed_domains": allowed_domains,
                              "blocked_domains": blocked_domains})
        return {"results": [{"title": f"Mock result for {query}", "url": "https://example.org"}],
                "summary": f"(mock) no live search for: {query}", "cost_usd": 0.0}
