"""The runtime event stream: one schema for the CLI, the web UI and notebooks.

``Runtime.emit(kind, **data)`` publishes through an :class:`EventBus`. A
subscriber is any callable ``fn(kind, data)``; it may be a plain function or a
coroutine function. Awaitables are scheduled on the running loop with strong
references kept until they finish; :meth:`EventBus.drain` awaits them. A
subscriber that raises is logged and never breaks a run. Events published from
a worker thread (blocking tool handlers) are handed to the loop thread.

Every payload carries ``ts`` (epoch seconds). Kinds emitted by the runtime
(``EVENT_KINDS`` lists their payload keys):

==================  ================================================================
agent_start         invocation_id, parent_invocation_id, tool_use_id, agent, division,
                    depth, description
agent_end           invocation_id, agent, depth, status, stop_reason, cost_usd,
                    duration_s, model_calls, tool_calls
tool_start          invocation_id, tool_use_id, agent, tool, input_preview
tool_end            invocation_id, tool_use_id, agent, tool, is_error, duration_s,
                    output_preview
text                invocation_id, agent, depth, text          (streamed delta)
message_end         invocation_id, agent                       (after each assistant message)
thinking            invocation_id, agent, depth, text, streamed
delegation          invocation_id, parent_invocation_id, tool_use_id, agent,
                    description, prompt_preview
delegation_end      invocation_id, parent_invocation_id, tool_use_id, agent, status,
                    stop_reason, cost_usd, duration_s
compaction          invocation_id, agent, strategy, tokens_before, tokens_after
retry               invocation_id, agent, attempt, delay_s, error
cost                total_usd
warning             message
==================  ================================================================

Kinds emitted by other packages: ``turn_start``, ``turn_end``,
``review_enforced``, ``briefing`` (CSO session); ``artifact_registered``,
``claims_filed`` (provenance tools). The legacy kind ``tool`` (agent, tool,
input) is still emitted as an alias of ``tool_start`` until every consumer
reads ``tool_start``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

log = logging.getLogger(__name__)

__all__ = ["Event", "EventBus", "EventCallback", "EVENT_KINDS", "LEGACY_ALIASES", "preview", "to_json_line"]

EventCallback = Callable[[str, dict[str, Any]], Union[None, Awaitable[Any]]]

EVENT_KINDS: dict[str, tuple[str, ...]] = {
    "agent_start": ("invocation_id", "parent_invocation_id", "tool_use_id", "agent", "division", "depth",
                    "description"),
    "agent_end": ("invocation_id", "agent", "depth", "status", "stop_reason", "cost_usd", "duration_s",
                  "model_calls", "tool_calls"),
    "tool_start": ("invocation_id", "tool_use_id", "agent", "tool", "input_preview"),
    "tool_end": ("invocation_id", "tool_use_id", "agent", "tool", "is_error", "duration_s", "output_preview"),
    "text": ("invocation_id", "agent", "depth", "text"),
    "message_end": ("invocation_id", "agent"),
    "thinking": ("invocation_id", "agent", "depth", "text", "streamed"),
    "delegation": ("invocation_id", "parent_invocation_id", "tool_use_id", "agent", "description",
                   "prompt_preview"),
    "delegation_end": ("invocation_id", "parent_invocation_id", "tool_use_id", "agent", "status", "stop_reason",
                       "cost_usd", "duration_s"),
    "compaction": ("invocation_id", "agent", "strategy", "tokens_before", "tokens_after"),
    "retry": ("invocation_id", "agent", "attempt", "delay_s", "error"),
    "cost": ("total_usd",),
    "warning": ("message",),
    # emitted by other packages
    "turn_start": ("turn", "prompt"),
    "turn_end": ("turn", "status", "reply", "cost_usd", "cumulative_usd"),
    "review_enforced": ("agents", "round"),
    "briefing": ("text",),
    "artifact_registered": ("path", "agent"),
    "claims_filed": ("ids", "n"),
}

#: Old kind -> new kind it mirrors (emitted alongside the new kind).
LEGACY_ALIASES = {"tool": "tool_start"}


@dataclass
class Event:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ts(self) -> float:
        return float(self.data.get("ts") or 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.data}


def to_json_line(kind: str, data: dict[str, Any]) -> str:
    """One NDJSON line for an event (``{"kind": ..., **data}``)."""
    return json.dumps({"kind": kind, **data}, default=str, ensure_ascii=False)


def preview(value: Any, n: int = 500) -> Any:
    """A bounded copy of ``value`` for event payloads: long strings are clipped
    (with their length), long lists shortened, nesting capped."""

    def walk(v: Any, depth: int) -> Any:
        if isinstance(v, str):
            return v if len(v) <= n else v[:n] + f"...({len(v):,} chars)"
        if depth > 4:
            return "..."
        if isinstance(v, dict):
            return {str(k): walk(x, depth + 1) for k, x in list(v.items())[:50]}
        if isinstance(v, (list, tuple)):
            items = [walk(x, depth + 1) for x in list(v)[:20]]
            if len(v) > 20:
                items.append(f"...({len(v) - 20} more)")
            return items
        if v is None or isinstance(v, (bool, int, float)):
            return v
        return walk(str(v), depth)

    return walk(value, 0)


class EventBus:
    """Fan-out of runtime events to sync or async subscribers."""

    def __init__(self) -> None:
        self._subs: list[EventCallback] = []
        self._pending: set[asyncio.Future] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: int | None = None

    # ------------------------------------------------------------------ subscribers

    def subscribe(self, fn: EventCallback) -> Callable[[], None]:
        """Add a subscriber ``fn(kind, data)``; returns a function that removes it."""
        if fn not in self._subs:
            self._subs.append(fn)
        return lambda: self.unsubscribe(fn)

    def unsubscribe(self, fn: EventCallback) -> None:
        try:
            self._subs.remove(fn)
        except ValueError:
            pass

    @property
    def subscribers(self) -> list[EventCallback]:
        return list(self._subs)

    # ------------------------------------------------------------------ publish

    def publish(self, kind: str, data: dict[str, Any] | None = None) -> None:
        data = dict(data or {})
        data.setdefault("ts", round(time.time(), 3))
        if not self._subs:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            self._loop, self._thread = loop, threading.get_ident()
        elif self._loop is not None and not self._loop.is_closed() and self._thread != threading.get_ident():
            # A worker thread (blocking tool): deliver on the loop thread.
            try:
                self._loop.call_soon_threadsafe(self._deliver, kind, data)
                return
            except RuntimeError:  # loop closed meanwhile
                pass
        self._deliver(kind, data)

    def _deliver(self, kind: str, data: dict[str, Any]) -> None:
        for fn in list(self._subs):
            try:
                r = fn(kind, data)
            except Exception:  # noqa: BLE001 - consumers must never break a run
                log.exception("event subscriber %r failed on %s", fn, kind)
                continue
            if inspect.isawaitable(r):
                self._schedule(r, kind)

    def _schedule(self, aw: Awaitable[Any], kind: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            # No loop in this thread: run the coroutine to completion here.
            try:
                if inspect.iscoroutine(aw):
                    asyncio.run(aw)
            except Exception:  # noqa: BLE001
                log.exception("async event subscriber failed on %s", kind)
            return
        fut = asyncio.ensure_future(self._guard(aw, kind))
        self._pending.add(fut)
        fut.add_done_callback(self._pending.discard)

    @staticmethod
    async def _guard(aw: Awaitable[Any], kind: str) -> None:
        try:
            await aw
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("async event subscriber failed on %s", kind)

    @property
    def pending(self) -> int:
        return len(self._pending)

    async def drain(self, timeout: float | None = None) -> None:
        """Await every scheduled async delivery (including ones scheduled meanwhile)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._pending:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            await asyncio.wait(list(self._pending), timeout=remaining)
