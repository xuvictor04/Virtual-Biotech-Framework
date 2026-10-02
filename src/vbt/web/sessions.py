"""Browser sessions for the web UI: one CSO session per browser session, driven by events.

Pure asyncio (no web framework), so it can be tested and reused on its own.

* ``WebSession`` wraps one ``vbt.orchestrator.CSOSession``, created lazily on
  the first question (so opening the page never leaves an empty run behind).
  The runtime's ``on_event`` callback feeds an ``EventLog``: an append-only,
  bounded buffer with sequential ids that Server-Sent-Event streams read from
  (a reconnecting browser resumes from ``Last-Event-ID``). Events are reshaped
  for the browser on the way in: CSO text has ``[[claim:..]]`` anchors removed
  while it streams, tool inputs become short previews, and ``turn_end`` carries
  the reply with numbered references and footnotes resolved against the run's
  evidence/claims.json.
* A question runs as a background task; a second question while one is running
  is refused with ``SessionBusyError('previous query still processing')``.
  ``stop()`` calls ``CSOSession.cancel()`` (or cancels the task), and the turn is
  recorded as interrupted.
* ``WebSessionManager`` caps concurrent sessions (``max_sessions``) and closes
  idle ones after ``idle_timeout_s``, awaiting ``CSOSession.close()`` so their
  records are finalised.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

log = logging.getLogger(__name__)

BUSY_MESSAGE = "previous query still processing"
INTERRUPT_REASON = "interrupted by the user (Stop)"

#: Event kinds forwarded without reshaping (payload sanitised only).
PASSTHROUGH_KINDS = frozenset({
    "briefing", "review_enforced", "compaction", "retry", "warning", "cost", "artifact_registered", "claims_filed",
    "mcp_server_started", "mcp_start_failed", "mcp_crash", "mcp_restart", "mcp_timeout", "mcp_lazy_retry",
    "delegation_end", "session_resumed",
})

OpenSessionFn = Callable[[dict, Callable[..., None]], Awaitable[Any]]


class SessionBusyError(RuntimeError):
    def __init__(self, message: str = BUSY_MESSAGE):
        super().__init__(message)


class SessionLimitError(RuntimeError):
    pass


# ----------------------------------------------------------------- helpers

def _preview(value: Any, n: int = 300) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(value)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def jsonable(value: Any, *, max_str: int = 4000, depth: int = 0) -> Any:
    """A JSON-safe, size-bounded copy of an event payload."""
    if depth > 6:
        return _preview(value, 200)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_str else value[:max_str] + "…"
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, max_str=max_str, depth=depth + 1) for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v, max_str=max_str, depth=depth + 1) for v in list(value)[:200]]
    if isinstance(value, Path):
        return str(value)
    return _preview(value, max_str)


class EventLog:
    """Append-only, bounded event buffer with sequential ids and async waiting.

    ``publish`` may be called from the event loop thread or any other thread.
    """

    def __init__(self, maxlen: int = 5000, loop: asyncio.AbstractEventLoop | None = None):
        self._events: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._next = 1
        self._loop = loop
        self._changed = asyncio.Event() if loop is not None else None
        self._lock = threading.Lock()
        self.closed = False

    def _bind(self) -> None:
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._changed = asyncio.Event()

    @property
    def last_id(self) -> int:
        return self._next - 1

    def publish(self, kind: str, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            ev = {"id": self._next, "kind": kind, "ts": round(time.time(), 3), "data": dict(data or {})}
            self._next += 1
            self._events.append(ev)
        self._notify()
        return ev

    def _notify(self) -> None:
        self._bind()
        if self._loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._wake()
        elif not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._wake)

    def _wake(self) -> None:
        old, self._changed = self._changed, asyncio.Event()
        if old is not None:
            old.set()

    def since(self, after_id: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._events if e["id"] > after_id]

    def first_id_of_last(self, kind: str) -> int | None:
        with self._lock:
            for e in reversed(self._events):
                if e["kind"] == kind:
                    return e["id"]
        return None

    async def wait(self, after_id: int, timeout: float | None) -> list[dict[str, Any]]:
        """Events after ``after_id``; waits up to ``timeout`` seconds for new ones."""
        self._bind()
        evs = self.since(after_id)
        if evs or self.closed:
            return evs
        changed = self._changed
        try:
            await asyncio.wait_for(changed.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self.since(after_id)

    def close(self) -> None:
        self.closed = True
        self._notify()


def _repair_history_shim(messages: list[Any], reason: str) -> int:
    """Answer unanswered tool calls (used only when the runtime has no repair_history)."""
    from ..providers.base import Message, ToolResult, unanswered_tool_calls

    pending = unanswered_tool_calls(messages)
    by_idx: dict[int, list[Any]] = {}
    for idx, call in pending:
        by_idx.setdefault(idx, []).append(call)
    for idx in sorted(by_idx, reverse=True):
        results = [ToolResult(c.id, f"Not executed: {reason}", True) for c in by_idx[idx]]
        nxt = idx + 1
        if nxt < len(messages) and messages[nxt].role == "user":
            messages[nxt].content[0:0] = results
        else:
            messages.insert(nxt, Message("user", results))
    return len(pending)


# ----------------------------------------------------------------- one browser session

class WebSession:
    """One browser session: a lazily opened CSO session plus its event stream."""

    def __init__(self, sid: str, owner: str, config: dict[str, Any], open_session: OpenSessionFn, *,
                 profile: str | None = None, model: str | None = None,
                 divisions: Mapping[str, str] | None = None, event_buffer: int = 10000):
        self.id = sid
        self.owner = owner
        self.config = config
        self.profile = profile
        self.model = model
        self.created = time.time()
        self.last_active = self.created
        self.events = EventLog(maxlen=event_buffer)
        self.cso: Any = None
        self.task: asyncio.Task | None = None
        self.turns_started = 0
        self.closed = False
        self._open_session = open_session
        self._open_lock = asyncio.Lock()
        self._divisions = dict(divisions or {})
        self._strippers: dict[str, Any] = {}
        self._seen_tool_ids: set[str] = set()
        self._has_tool_start = False
        self._turn_end_seen = False
        self._stop_requested = False
        self._partial: list[str] = []
        self._prompt = ""
        self._legacy_tool_n = 0

    # ------------------------------------------------------------ state

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    @property
    def run_dir(self) -> Path | None:
        run = getattr(self.cso, "run", None)
        return Path(run.dir) if run is not None else None

    @property
    def run_id(self) -> str | None:
        run = getattr(self.cso, "run", None)
        return getattr(run, "run_id", None) if run is not None else None

    def touch(self) -> None:
        self.last_active = time.time()

    def status(self) -> dict[str, Any]:
        run = getattr(self.cso, "run", None)
        cost = None
        turns = 0
        if run is not None:
            try:
                cost = round(float(run.cost.total_usd), 6)
                turns = len(run.turns)
            except Exception:  # noqa: BLE001
                pass
        return {"id": self.id, "busy": self.busy, "run_id": self.run_id, "turns": turns, "cost_usd": cost,
                "profile": self.profile, "model": self.model, "created": self.created,
                "last_active": self.last_active, "last_event_id": self.events.last_id, "closed": self.closed}

    # ------------------------------------------------------------ events

    def on_event(self, kind: str, data: Mapping[str, Any] | None = None, **kw: Any) -> None:
        """Runtime event callback; accepts ``cb(kind, data)`` and ``cb(kind, **data)``."""
        try:
            payload = dict(data or {})
            payload.update(kw)
            self._ingest(str(kind), payload)
        except Exception:  # noqa: BLE001 - the UI must never break a run
            log.exception("web event %s could not be processed", kind)

    def _stripper(self, key: str):
        from ..audit.render import StreamingAnchorStripper
        if key not in self._strippers:
            self._strippers[key] = StreamingAnchorStripper()
        return self._strippers[key]

    def _flush_text(self, key: str, agent: Any, depth: Any, inv: Any) -> None:
        s = self._strippers.pop(key, None)
        if s is not None:
            rest = s.flush()
            if rest:
                self._publish_text(agent, depth, inv, rest)

    def _publish_text(self, agent: Any, depth: Any, inv: Any, text: str) -> None:
        if str(agent) == "cso":
            self._partial.append(text)
        self.events.publish("text", {"agent": agent, "depth": depth, "invocation_id": inv, "text": text})

    def _ingest(self, kind: str, d: dict[str, Any]) -> None:
        agent = d.get("agent")
        inv = d.get("invocation_id")
        key = f"{agent}|{inv or ''}"
        if kind == "text":
            out = self._stripper(key).feed(str(d.get("text") or ""))
            if out:
                self._publish_text(agent, d.get("depth"), inv, out)
            return
        if kind == "message_end":
            self._flush_text(key, agent, d.get("depth"), inv)
            self.events.publish("message_end", {"agent": agent, "invocation_id": inv})
            return
        if kind in ("tool_start", "tool"):
            tuid = d.get("tool_use_id")
            if kind == "tool_start":
                self._has_tool_start = True
            elif self._has_tool_start and not tuid:
                return  # legacy alias of a tool_start already shown
            if tuid:
                if str(tuid) in self._seen_tool_ids:
                    return
                self._seen_tool_ids.add(str(tuid))
            else:
                self._legacy_tool_n += 1
                tuid = f"legacy-{self._legacy_tool_n}"
            self.events.publish("tool_start", {
                "tool_use_id": tuid, "agent": agent, "tool": d.get("tool"), "invocation_id": inv,
                "input_preview": _preview(d.get("input_preview") if d.get("input_preview") is not None
                                          else d.get("input")),
                "legacy": kind == "tool" and not d.get("tool_use_id"),
            })
            return
        if kind == "tool_end":
            self.events.publish("tool_end", {
                "tool_use_id": d.get("tool_use_id"), "agent": agent, "tool": d.get("tool"), "invocation_id": inv,
                "is_error": bool(d.get("is_error")), "duration_s": d.get("duration_s"),
                "output_preview": _preview(d.get("output_preview") if d.get("output_preview") is not None
                                           else d.get("output"), 400)})
            return
        if kind in ("agent_start", "agent_end"):
            for k in [k for k in self._strippers if k.startswith(f"{agent}|")] if kind == "agent_end" else []:
                self._flush_text(k, agent, d.get("depth"), inv)
            self.events.publish(kind, jsonable({**{k: d.get(k) for k in (
                "agent", "depth", "invocation_id", "parent_invocation_id", "tool_use_id", "description", "status",
                "stop_reason", "cost_usd", "duration_s", "model_calls", "tool_calls") if d.get(k) is not None},
                "division": d.get("division") or self._divisions.get(str(agent), "")}))
            return
        if kind == "delegation":
            self.events.publish("delegation", jsonable({
                "agent": agent, "description": d.get("description"), "invocation_id": inv,
                "parent_invocation_id": d.get("parent_invocation_id"), "tool_use_id": d.get("tool_use_id"),
                "division": self._divisions.get(str(agent), ""),
                "prompt_preview": _preview(d.get("prompt_preview") or d.get("prompt"), 600)}))
            return
        if kind == "thinking":
            self.events.publish("thinking", jsonable({
                "agent": agent, "depth": d.get("depth"), "invocation_id": inv, "text": d.get("text") or "",
                "streamed": d.get("streamed")}, max_str=20000))
            return
        if kind == "turn_start":
            self.events.publish("turn_start", jsonable({"turn": d.get("turn"), "prompt": d.get("prompt")}))
            return
        if kind == "turn_end":
            for k in list(self._strippers):
                a, _, i = k.partition("|")
                self._flush_text(k, a, None, i or None)
            self._turn_end_seen = True
            self.events.publish("turn_end", self._enrich_turn_end(d))
            return
        if kind in PASSTHROUGH_KINDS:
            self.events.publish(kind, jsonable(d))
            return
        self.events.publish(kind, jsonable(d, max_str=2000))

    def _turn_record(self, n: Any) -> dict[str, Any] | None:
        run = getattr(self.cso, "run", None)
        for t in reversed(getattr(run, "turns", None) or []):
            if t.get("turn") == n:
                return t
        return None

    def _enrich_turn_end(self, d: Mapping[str, Any]) -> dict[str, Any]:
        from ..audit.claims import load_claims
        from ..audit.render import number_refs

        out = jsonable(dict(d), max_str=200000)
        reply = str(d.get("reply") if d.get("reply") is not None else d.get("response") or "")
        claims = load_claims(self.run_dir) if self.run_dir else []
        rendered, footnotes = number_refs(reply, claims)
        out["reply"] = reply
        out["rendered"] = rendered
        out["footnotes"] = jsonable(footnotes, max_str=4000)
        rec = self._turn_record(d.get("turn"))
        if rec is not None:
            out.setdefault("status", rec.get("status"))
            out["claims_filed"] = list(rec.get("claims_filed") or [])
            out["claims_unresolved"] = list(rec.get("claims_unresolved") or [])
            out["research"] = rec.get("research")
            out.setdefault("cost_usd", rec.get("cost_usd"))
        run_dir = self.run_dir
        if run_dir is not None:
            out["run_id"] = self.run_id
            out["has_audit"] = (run_dir / "audit.html").is_file()
        return out

    # ------------------------------------------------------------ lifecycle

    async def ensure_started(self) -> Any:
        async with self._open_lock:
            if self.cso is None:
                self.events.publish("status", {"message": "starting session"})
                self.cso = await self._open_session(self.config, self.on_event)
                self.events.publish("session_started", {"run_id": self.run_id,
                                                        "run_dir": str(self.run_dir) if self.run_dir else None})
        return self.cso

    def ask(self, prompt: str) -> int:
        """Start a turn in the background. Raises SessionBusyError while one runs."""
        if self.closed:
            raise RuntimeError("session is closed")
        if self.busy:
            raise SessionBusyError()
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("empty prompt")
        self.touch()
        self.turns_started += 1
        self._turn_end_seen = False
        self._stop_requested = False
        self._partial = []
        self._prompt = prompt
        self.events.publish("user", {"prompt": prompt, "n": self.turns_started})
        self.task = asyncio.ensure_future(self._run_turn(prompt))
        return self.turns_started

    async def _run_turn(self, prompt: str) -> None:
        status, error, reply = "completed", None, None
        try:
            cso = await self.ensure_started()
            reply = await cso.ask(prompt)
        except asyncio.CancelledError:
            status = "interrupted"
        except Exception as exc:  # noqa: BLE001 - reported to the browser
            if type(exc).__name__ == "SessionBusy":
                self.events.publish("error", {"message": BUSY_MESSAGE})
                return
            if self._stop_requested:
                status = "interrupted"
            else:
                status = f"failed: {type(exc).__name__}: {exc}"
                error = f"{type(exc).__name__}: {exc}"
                log.exception("web turn failed")
        finally:
            self.touch()
        if status == "interrupted":
            await asyncio.shield(asyncio.ensure_future(self._after_interrupt()))
        if error:
            self.events.publish("error", {"message": error})
        if not self._turn_end_seen:
            rec = self._turn_record(getattr(self.cso, "turn", None)) if self.cso is not None else None
            n = rec.get("turn") if rec else getattr(self.cso, "turn", None)
            text = reply if reply is not None else (rec or {}).get("response") or "".join(self._partial)
            run = getattr(self.cso, "run", None)
            self._ingest("turn_end", {
                "turn": n, "status": (rec or {}).get("status") or status, "reply": text,
                "cost_usd": (rec or {}).get("cost_usd"),
                "cumulative_usd": round(float(run.cost.total_usd), 6) if run is not None else None,
                "synthetic": True})

    async def _after_interrupt(self) -> None:
        """Make sure an interrupted turn is recorded and the history stays valid.

        ``CSOSession`` records interrupted turns and repairs its history itself;
        this only fills in when it did not (older orchestrators), so it is inert
        otherwise.
        """
        cso = self.cso
        if cso is None:
            return
        run = getattr(cso, "run", None)
        n = getattr(cso, "turn", None)
        try:
            if run is not None and isinstance(n, int) and n > 0 and not any(
                    t.get("turn") == n for t in getattr(run, "turns", [])):
                partial = "".join(self._partial).strip()
                response = (partial + "\n\n" if partial else "") + \
                    "[Turn interrupted by the user. Evidence coverage is incomplete.]"
                run.trace("turn_end", turn=n, status="interrupted")
                run.finish_turn({"turn": n, "prompt": self._prompt, "response": response, "status": "interrupted"})
            history = getattr(cso, "history", None)
            if isinstance(history, list):
                from ..providers.base import validate_tool_pairing
                if validate_tool_pairing(history):
                    repair = getattr(getattr(cso, "rt", None), "repair_history", None)
                    if callable(repair):
                        repair(history, INTERRUPT_REASON)
                    else:
                        _repair_history_shim(history, INTERRUPT_REASON)
        except Exception:  # noqa: BLE001
            log.exception("recording the interrupted turn failed")

    async def stop(self, timeout: float = 15.0) -> dict[str, Any]:
        """Cancel the running turn. Returns ``{ok, status}`` once it has wound down."""
        if not self.busy:
            return {"ok": False, "detail": "no turn is running"}
        self._stop_requested = True
        self.events.publish("status", {"message": "stopping"})
        cancel = getattr(self.cso, "cancel", None) if self.cso is not None else None
        handled = False
        if callable(cancel):
            try:
                r = cancel()
                if inspect.isawaitable(r):
                    r = await r
                handled = r is not False
            except Exception:  # noqa: BLE001
                handled = False
        task = self.task
        if task is None:
            return {"ok": True, "status": "interrupted"}
        if not handled:
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:  # the session's own cancel did not wind the turn down; force it
            task.cancel()
            await asyncio.wait({task}, timeout=timeout)
        return {"ok": True, "status": "interrupted"}

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.busy:
            try:
                await self.stop(timeout=10)
            except Exception:  # noqa: BLE001
                log.exception("stopping a turn during close failed")
        if self.cso is not None:
            try:
                await self.cso.close()
            except Exception:  # noqa: BLE001
                log.exception("closing a CSO session failed")
        self.events.publish("session_closed", {"run_id": self.run_id})
        self.events.close()


# ----------------------------------------------------------------- the manager

class WebSessionManager:
    """Owns the browser sessions: creation cap, lookup by (id, owner), idle cleanup."""

    def __init__(self, open_session: OpenSessionFn, *, max_sessions: int = 20, idle_timeout_s: float = 8 * 3600,
                 sweep_interval_s: float = 60.0, divisions: Mapping[str, str] | None = None,
                 event_buffer: int = 10000):
        self._open_session = open_session
        self.event_buffer = int(event_buffer)
        self.max_sessions = int(max_sessions)
        self.idle_timeout_s = float(idle_timeout_s)
        self.sweep_interval_s = float(sweep_interval_s)
        self.divisions = dict(divisions or {})
        self.sessions: dict[str, WebSession] = {}
        self._sweeper: asyncio.Task | None = None

    def __len__(self) -> int:
        return len(self.sessions)

    def start(self) -> None:
        """Start the idle sweeper (idempotent; needs a running loop)."""
        if self._sweeper is None or self._sweeper.done():
            try:
                self._sweeper = asyncio.get_running_loop().create_task(self._sweep_loop())
            except RuntimeError:
                self._sweeper = None

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.sweep_interval_s)
            try:
                await self.sweep()
            except Exception:  # noqa: BLE001
                log.exception("idle-session sweep failed")

    def create(self, owner: str, config: dict[str, Any], *, profile: str | None = None,
               model: str | None = None) -> WebSession:
        self.start()
        if len(self.sessions) >= self.max_sessions:
            raise SessionLimitError(f"too many active sessions ({self.max_sessions}); close one or try later")
        sid = secrets.token_urlsafe(16)
        s = WebSession(sid, owner, config, self._open_session, profile=profile, model=model,
                       divisions=self.divisions, event_buffer=self.event_buffer)
        self.sessions[sid] = s
        return s

    def get(self, sid: str, owner: str) -> WebSession:
        s = self.sessions.get(sid)
        if s is None or s.closed or not secrets.compare_digest(s.owner, owner):
            raise KeyError(sid)
        s.touch()
        return s

    async def close(self, sid: str) -> None:
        s = self.sessions.pop(sid, None)
        if s is not None:
            await s.close()

    async def sweep(self, now: float | None = None) -> list[str]:
        """Close sessions idle longer than ``idle_timeout_s`` (never one with a running turn)."""
        now = time.time() if now is None else now
        stale = [sid for sid, s in self.sessions.items()
                 if not s.busy and now - s.last_active > self.idle_timeout_s]
        for sid in stale:
            await self.close(sid)
        return stale

    async def aclose(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._sweeper = None
        for sid in list(self.sessions):
            await self.close(sid)


__all__ = ["WebSession", "WebSessionManager", "EventLog", "SessionBusyError", "SessionLimitError", "BUSY_MESSAGE",
           "jsonable"]
