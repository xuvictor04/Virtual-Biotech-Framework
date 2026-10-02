"""Bridge MCP servers (stdio or HTTP) into the harness tool registry.

Each server's tools are exposed as ``mcp__<server>__<tool>`` -- the naming the
upstream prompts use -- with their MCP input schemas passed through, so any
provider can call them. Servers are any MCP implementation (the upstream
FastMCP servers, your own, or third-party ones), configured in YAML.

Robustness (one supervisor per server):

* **Startup** -- each server runs in its own supervisor task that owns the
  transport and session contexts (its own ``AsyncExitStack``), so a failed or
  timed-out start is torn down completely. Startup is retried
  (``mcp.start_attempts``, ``start_timeout_s``, backoff ``start_backoff_s`` x
  ``start_backoff_factor``); a server that still fails gets one lazy retry on
  the first later call to it.
* **Diagnostics** -- server stderr goes to ``<log_dir>/<name>.log``; its tail is
  included in every startup-failure message.
* **Crash recovery** -- a transport failure (connection closed, broken/closed
  stream, EOF) tears the server down, restarts it (at most ``mcp.max_restarts``
  times per session) and retries the call once. MCP data tools are read-only
  queries, so a retry is safe.
* **Timeouts** -- per server ``timeout_s`` (else ``mcp.default_timeout_s``). On
  timeout the call fails with a readable message and a best-effort
  ``notifications/cancelled`` is sent so the server can stop working.
* **Concurrency** -- optional per-server ``max_concurrency`` semaphore.
* **Environment** -- children get an allow-listed environment
  (:func:`vbt.envpolicy.child_env`) plus ``env_passthrough`` names, the
  harness ``extra_env`` and the server's ``env``; provider credentials are
  never inherited unless ``inherit_env`` is set.
* **Results** -- text content becomes text, images become provider image
  parts (when available), embedded text resources become text. Legacy failure
  envelopes (``{"error": ...}``, ``{"success": false}``, ``{"ok": false}``) are
  reported as tool errors, while the upstream servers' explicit empty-lookup
  messages are not.

``on_event(kind, **data)`` (e.g. ``Runtime.emit`` or ``Run.trace``) receives
``mcp_server_started``, ``mcp_start_failed``, ``mcp_crash``, ``mcp_restart``,
``mcp_timeout`` and ``mcp_lazy_retry`` events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..envpolicy import child_env
from .base import Tool, ToolContext, ToolFailure, inline_refs

log = logging.getLogger(__name__)

DEFAULT_OPTIONS: dict[str, Any] = {
    "start_attempts": 3,
    "start_timeout_s": 180.0,
    "start_backoff_s": 3.0,
    "start_backoff_factor": 1.5,
    "default_timeout_s": 1800.0,
    "max_restarts": 3,
    "inherit_env": False,
}

STDERR_TAIL_CHARS = 2000
_CONNECTION_CLOSED = -32000  # JSON-RPC code the MCP SDK uses for a closed connection

TIMEOUT_HINT = ("the server may still be working — narrow the query (e.g. count_cells first, "
                "add value_filter/genes) before retrying")

FAILURE_INSTRUCTION = ("This call did not produce evidence. Report the failure and its effect on the "
                       "analysis. Identify any alternative source explicitly; do not attribute it to "
                       "this failed call.")


@dataclass
class MCPServerConfig:
    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None  # streamable-HTTP servers
    enabled: bool = True
    timeout_s: float | None = None          # per call; None -> options.default_timeout_s
    inherit_env: bool | None = None         # None -> options.inherit_env (default False)
    env_passthrough: list[str] = field(default_factory=list)
    max_concurrency: int | None = None
    start_timeout_s: float | None = None    # per startup attempt; None -> options.start_timeout_s


class _ConfigError(Exception):
    """A startup problem retrying cannot fix (missing interpreter or script)."""


@dataclass
class _Server:
    cfg: MCPServerConfig
    state: str = "stopped"            # stopped | starting | ready | failed | closed
    session: Any = None
    generation: int = 0               # bumped on every successful (re)start
    restarts: int = 0
    last_error: str | None = None
    in_flight: int = 0
    queued: int = 0
    calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    task: asyncio.Future | None = None
    stop: asyncio.Event | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sem: asyncio.Semaphore | None = None
    lazy_retry_used: bool = False
    log_path: Path | None = None
    log_offset: int = 0
    started_at: str | None = None
    start_duration_s: float | None = None


# ---------------------------------------------------------------------------
# Result-envelope semantics (port of upstream src/utils/tool_errors.py)
# ---------------------------------------------------------------------------

# The upstream tools' explicit empty-lookup messages. Arbitrary "not found"
# errors are not exempt: a missing dataset, file or column is a failed query.
_EMPTY_LOOKUP = re.compile(
    r"(?:"
    r"(?:Target|Drug|Disease|Biosample|Pathway|GO term|SO term|Study|Variant)"
    r"(?: [^\n]+)? not found"
    r"|Gene not found(?: in (?:expression|essentiality) dataset)?"
    r"|rsID '[^\n]+' not found in database"
    r"|Entity '[^\n]+' not found in literature vector dataset"
    r"|Trial [^\n]+ not found in ClinicalTrials\.gov"
    r"|No therapeutic area found matching '[^\n]+'"
    r"|No cells found for filter: [^\n]+"
    r"|No samples found"
    r"|No perturbation data found for this drug"
    r"|No data for cell line [^\n]+"
    r"|No comparison cell lines with data"
    r"|Insufficient data: drug_a has \d+ hits, drug_b has \d+ hits"
    r")\.?",
    re.IGNORECASE,
)


def tool_result_error(result: Any) -> str | None:
    """Return the failure message of a tool result, or None for success/empty lookups.

    Accepts a dict, serialized JSON, or MCP text-content blocks, and inspects
    only the result envelope (never nested evidence rows that may legitimately
    contain an ``error`` column).
    """
    if isinstance(result, str):
        try:
            return tool_result_error(json.loads(result))
        except (ValueError, TypeError):
            return None
    if isinstance(result, (list, tuple)):
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                error = tool_result_error(block.get("text"))
            elif getattr(block, "type", None) == "text":
                error = tool_result_error(getattr(block, "text", None))
            else:
                continue
            if error:
                return error
        return None
    if not isinstance(result, dict):
        return None
    if result.get("type") == "text":
        return tool_result_error(result.get("text"))
    if result.get("is_error") is True or result.get("isError") is True:
        return (tool_result_error(result.get("content"))
                or str(result.get("content") or result.get("error") or "Tool call failed"))
    if "structuredContent" in result:
        error = tool_result_error(result["structuredContent"])
        if error:
            return error
    if result.get("type") == "tool_result" or "isError" in result:
        return tool_result_error(result.get("content"))
    error = result.get("error") or result.get("errors")
    failed = result.get("success") is False or result.get("ok") is False
    if not error and not failed:
        return None
    if isinstance(error, list):
        message = "; ".join(str(item) for item in error)
    else:
        message = str(error or "Tool returned an unsuccessful result")
    if _EMPTY_LOOKUP.fullmatch(message.strip()):
        return None
    return message


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a field under its mcp 2.x (snake_case) or 1.x (camelCase) name."""
    fields = getattr(type(obj), "model_fields", None)
    if isinstance(fields, dict):  # pydantic model: read only declared fields (aliases may warn)
        for n in names:
            if n in fields:
                v = getattr(obj, n, None)
                return default if v is None else v
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _leaves(exc: BaseException) -> list[BaseException]:
    subs = getattr(exc, "exceptions", None)
    if isinstance(subs, (list, tuple)) and subs:
        out: list[BaseException] = []
        for s in subs:
            out.extend(_leaves(s))
        return out
    return [exc]


def _describe(exc: BaseException) -> str:
    leaves = _leaves(exc)
    return "; ".join(f"{type(e).__name__}: {e}".rstrip(": ") for e in leaves[:3])


def is_transport_error(exc: BaseException) -> bool:
    """True when ``exc`` means the connection to the server is gone (crash, EOF)."""
    try:
        import anyio
        transport = (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream)
    except ImportError:  # pragma: no cover - anyio ships with mcp
        transport = ()
    for e in _leaves(exc):
        if isinstance(e, transport) or isinstance(e, (ConnectionError, EOFError)):
            return True
        if type(e).__name__ in ("McpError", "MCPError"):
            code = getattr(getattr(e, "error", None), "code", None)
            if code is None:
                code = getattr(e, "code", None)
            if code == _CONNECTION_CLOSED or "connection closed" in str(e).lower():
                return True
    return False


def _image_part(media_type: str, data_b64: str, source: str) -> Any | None:
    try:
        from ..providers.base import ImagePart  # provided by the provider layer (content parts)
    except ImportError:
        return None
    try:
        return ImagePart(media_type=media_type, data_b64=data_b64, source=source)
    except TypeError:  # pragma: no cover - unexpected constructor
        return None


def _text_block(text: str) -> Any:
    from ..providers.base import TextBlock
    return TextBlock(text)


def _silence(fut: asyncio.Future) -> None:
    if not fut.cancelled():
        fut.exception()


class MCPBridge:
    def __init__(self, servers: list[MCPServerConfig], *, extra_env: dict[str, str] | None = None,
                 log_dir: str | os.PathLike | None = None, options: dict[str, Any] | None = None,
                 on_event: Callable[..., Any] | None = None):
        self.servers = [s for s in servers if s.enabled]
        self.extra_env = {k: str(v) for k, v in (extra_env or {}).items() if v is not None}
        self.options = {**DEFAULT_OPTIONS, **{k: v for k, v in (options or {}).items() if v is not None}}
        self.on_event = on_event
        self._tmp_log_dir: str | None = None
        self.log_dir = Path(log_dir) if log_dir else None
        self._servers: dict[str, _Server] = {}
        for s in self.servers:
            st = _Server(cfg=s)
            if s.max_concurrency:
                st.sem = asyncio.Semaphore(int(s.max_concurrency))
            self._servers[s.name] = st
        self.sessions: dict[str, Any] = {}     # live sessions (name -> ClientSession)
        self.failures: dict[str, str] = {}     # servers currently unavailable (name -> reason)
        self.tools: list[Tool] = []
        self._closed = False

    # ------------------------------------------------------------------ events / logs

    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event is None:
            return
        try:
            r = self.on_event(kind, **data)
            if asyncio.iscoroutine(r):
                asyncio.ensure_future(r)
        except Exception:  # noqa: BLE001 - observers must not break tool calls
            log.debug("MCP event callback failed", exc_info=True)

    def _log_file_for(self, st: _Server) -> Path:
        if self.log_dir is None:
            if self._tmp_log_dir is None:
                self._tmp_log_dir = tempfile.mkdtemp(prefix="vbt-mcp-logs-")
            base = Path(self._tmp_log_dir)
        else:
            base = self.log_dir
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{st.cfg.name}.log"

    def _open_errlog(self, st: _Server):
        path = self._log_file_for(st)
        f = open(path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115 - closed by the exit stack
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        f.write(f"\n===== {st.cfg.name}: start at {stamp} =====\n")
        f.flush()
        st.log_path, st.log_offset = path, f.tell()
        return f

    def stderr_tail(self, name: str, n: int = STDERR_TAIL_CHARS) -> str:
        """The last ``n`` characters the server wrote to stderr in its latest start."""
        st = self._servers.get(name)
        if st is None or st.log_path is None or not st.log_path.exists():
            return ""
        try:
            with open(st.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                start = max(st.log_offset, size - n)
                f.seek(start)
                return f.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _failure_message(self, st: _Server, exc: BaseException, timeout: float) -> str:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            msg = f"startup timed out after {timeout:g}s"
        else:
            msg = _describe(exc)
        tail = self.stderr_tail(st.cfg.name)
        if tail:
            msg += f"\n[stderr tail, {st.log_path}]\n{tail}"
        elif st.log_path:
            msg += f" (no stderr output; log: {st.log_path})"
        return msg

    # ------------------------------------------------------------------ environment / command

    def child_env(self, cfg: MCPServerConfig) -> dict[str, str]:
        inherit = cfg.inherit_env if cfg.inherit_env is not None else bool(self.options.get("inherit_env"))
        extra = {**self.extra_env, **{k: str(v) for k, v in (cfg.env or {}).items() if v is not None}}
        if inherit:
            return {**os.environ, **extra}
        return child_env(os.environ, passthrough=cfg.env_passthrough or (), extra=extra)

    @staticmethod
    def check_command(cfg: MCPServerConfig) -> None:
        """Raise ``_ConfigError`` with an actionable message for a bad stdio command."""
        if cfg.url:
            return
        cmd = (cfg.command or "").strip()
        if not cmd:
            raise _ConfigError(f"MCP server {cfg.name!r} has an empty command (is vars.mcp_python / "
                               f"VBT_MCP_PYTHON blank?)")
        if os.sep in cmd or (os.altsep and os.altsep in cmd):
            if not (os.path.isfile(cmd) and os.access(cmd, os.X_OK)):
                raise _ConfigError(f"MCP server {cfg.name!r}: interpreter {cmd!r} is missing or not executable")
        elif shutil.which(cmd) is None:
            raise _ConfigError(f"MCP server {cfg.name!r}: command {cmd!r} not found on PATH")
        for a in cfg.args or []:
            a = str(a)
            if not a.endswith(".py") or a.startswith("-"):
                continue
            path = a if os.path.isabs(a) or not cfg.cwd else os.path.join(cfg.cwd, a)
            if not os.path.isfile(path):
                raise _ConfigError(f"MCP server {cfg.name!r}: server script not found: {path} "
                                   f"(did you run `git submodule update --init`?)")

    # ------------------------------------------------------------------ server lifecycle

    async def _open(self, st: _Server, stack: AsyncExitStack):
        from mcp import ClientSession

        cfg = st.cfg
        if cfg.url:
            try:
                from mcp.client.streamable_http import streamable_http_client as http_client
            except ImportError:  # mcp 1.x
                from mcp.client.streamable_http import streamablehttp_client as http_client  # type: ignore
            streams = await stack.enter_async_context(http_client(cfg.url))
            read, write = streams[0], streams[1]
        else:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=cfg.command, args=[str(a) for a in cfg.args],
                                           env=self.child_env(cfg), cwd=cfg.cwd)
            errlog = self._open_errlog(st)
            stack.callback(errlog.close)
            read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _supervise(self, st: _Server, ready: asyncio.Future, stop: asyncio.Event) -> None:
        """Own one server process: open, warm up (initialize + list_tools), wait, close."""
        try:
            async with AsyncExitStack() as stack:
                session = await self._open(st, stack)
                listed = await session.list_tools()
                if not ready.done():
                    ready.set_result((session, listed))
                await stop.wait()
        except BaseException as exc:  # noqa: BLE001 - reported through `ready` or last_error
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception)
                                    else ConnectionError("server startup was cancelled"))
            elif isinstance(exc, Exception):
                st.last_error = _describe(exc)
                log.debug("MCP server %s transport ended: %s", st.cfg.name, st.last_error)
            if not isinstance(exc, Exception):
                raise

    async def _attempt(self, st: _Server, timeout: float):
        loop = asyncio.get_running_loop()
        ready: asyncio.Future = loop.create_future()
        ready.add_done_callback(_silence)
        stop = asyncio.Event()
        task = asyncio.ensure_future(self._supervise(st, ready, stop))
        try:
            session, listed = await asyncio.wait_for(asyncio.shield(ready), timeout)
        except BaseException:
            # A start that never got ready is cancelled at once (the transport's own
            # shutdown still terminates the process); no point waiting for `stop`.
            await self._stop_task(task, stop, grace=15.0 if ready.done() else 0.0)
            raise
        st.session, st.task, st.stop = session, task, stop
        return listed

    async def _stop_task(self, task: asyncio.Future | None, stop: asyncio.Event | None,
                         grace: float = 15.0) -> None:
        if stop is not None:
            stop.set()
        if task is None:
            return
        if not task.done():
            done, _ = await asyncio.wait({task}, timeout=grace)
            if not done:
                task.cancel()
                await asyncio.wait({task}, timeout=5.0)
        if task.done() and not task.cancelled():
            task.exception()  # retrieve so asyncio does not warn

    async def _teardown(self, st: _Server) -> None:
        task, stop = st.task, st.stop
        st.session, st.task, st.stop = None, None, None
        self.sessions.pop(st.cfg.name, None)
        await self._stop_task(task, stop)

    async def _start(self, st: _Server, *, timeout: float | None = None, reason: str = "startup") -> bool:
        """Start ``st`` with retries. Caller holds ``st.lock``. Returns success."""
        cfg = st.cfg
        attempts = max(1, int(self.options["start_attempts"]))
        timeout = float(timeout or cfg.start_timeout_s or self.options["start_timeout_s"])
        delay = float(self.options["start_backoff_s"])
        factor = float(self.options["start_backoff_factor"])
        st.state = "starting"
        err = "not started"
        for attempt in range(1, attempts + 1):
            t0 = time.monotonic()
            try:
                self.check_command(cfg)
                listed = await self._attempt(st, timeout)
            except _ConfigError as exc:
                err = str(exc)
                self._emit("mcp_start_failed", server=cfg.name, attempt=attempt, reason=reason, error=err,
                           retry=False)
                break
            except asyncio.CancelledError:
                st.state = "failed"
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, retried
                err = self._failure_message(st, exc, timeout)
                log.warning("MCP server %s failed to start (attempt %d/%d): %s", cfg.name, attempt, attempts,
                            err.splitlines()[0])
                self._emit("mcp_start_failed", server=cfg.name, attempt=attempt, reason=reason,
                           error=err[:2000], retry=attempt < attempts)
                if attempt < attempts:
                    await asyncio.sleep(delay)
                    delay *= factor
                continue
            st.generation += 1
            st.state, st.last_error, st.lazy_retry_used = "ready", None, False
            st.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            st.start_duration_s = round(time.monotonic() - t0, 3)
            self.sessions[cfg.name] = st.session
            self.failures.pop(cfg.name, None)
            new = self._register(st, listed)
            self._emit("mcp_server_started", server=cfg.name, attempt=attempt, reason=reason,
                       duration_s=st.start_duration_s, n_tools=len(st.tool_names), new_tools=len(new))
            return True
        st.state, st.last_error = "failed", err
        self.failures[cfg.name] = err
        self.sessions.pop(cfg.name, None)
        return False

    def _register(self, st: _Server, listed: Any) -> list[Tool]:
        known = {t.name for t in self.tools}
        new: list[Tool] = []
        names = []
        for t in getattr(listed, "tools", []) or []:
            names.append(t.name)
            full = f"mcp__{st.cfg.name}__{t.name}"
            if full in known:
                continue
            schema = _attr(t, "input_schema", "inputSchema") or {"type": "object"}
            tool = Tool(
                name=full,
                description=(t.description or t.name).strip(),
                input_schema=inline_refs(dict(schema)),
                handler=self._make_handler(st.cfg.name, t.name),
                source=f"mcp:{st.cfg.name}",
            )
            self.tools.append(tool)
            new.append(tool)
        st.tool_names = names
        return new

    async def start(self, only: set[str] | None = None, connect_timeout: float | None = None) -> list[Tool]:
        """Start the configured servers (optionally a subset) in parallel; returns all bridged tools."""
        wanted = [st for name, st in self._servers.items() if only is None or name in only]

        async def one(st: _Server) -> None:
            async with st.lock:
                if st.state != "ready":
                    await self._start(st, timeout=connect_timeout)

        await asyncio.gather(*(one(st) for st in wanted))
        return self.tools

    async def retry_failed(self) -> list[Tool]:
        """Retry every failed server now; returns tools that became available."""
        before = len(self.tools)
        failed = [st for st in self._servers.values() if st.state == "failed"]

        async def one(st: _Server) -> None:
            async with st.lock:
                if st.state == "failed":
                    await self._start(st, reason="retry")

        await asyncio.gather(*(one(st) for st in failed))
        return self.tools[before:]

    async def _ready_session(self, st: _Server) -> tuple[Any, int]:
        name = st.cfg.name
        async with st.lock:
            if self._closed:
                raise ToolFailure("MCP bridge is closed")
            if st.state == "ready" and st.session is not None:
                return st.session, st.generation
            if st.state == "failed" and st.lazy_retry_used:
                raise ToolFailure(f"MCP server {name!r} is unavailable: {st.last_error}")
            st.lazy_retry_used = True
            self._emit("mcp_lazy_retry", server=name, previous_error=(st.last_error or "")[:500])
            if not await self._start(st, reason="lazy"):
                raise ToolFailure(f"MCP server {name!r} is unavailable (restart failed): {st.last_error}")
            return st.session, st.generation

    async def _restart(self, st: _Server, generation: int, reason: str) -> None:
        name = st.cfg.name
        async with st.lock:
            if st.generation != generation:
                return  # another call already restarted it
            if st.state == "failed":  # another call's restart already failed
                raise ToolFailure(f"MCP server {name!r} is unavailable: {st.last_error}")
            max_restarts = int(self.options["max_restarts"])
            if st.restarts >= max_restarts:
                await self._teardown(st)
                st.state = "failed"
                st.lazy_retry_used = True  # no lazy retry after the restart budget is spent
                st.last_error = f"crashed {st.restarts + 1} times; restart limit ({max_restarts}) reached: {reason}"
                self.failures[name] = st.last_error
                raise ToolFailure(f"MCP server {name!r} crashed and its restart limit is reached: {reason}")
            st.restarts += 1
            self._emit("mcp_restart", server=name, restarts=st.restarts, reason=reason[:1000])
            log.warning("MCP server %s connection lost (%s); restarting (%d/%d)", name, reason,
                        st.restarts, max_restarts)
            await self._teardown(st)
            if not await self._start(st, reason="restart"):
                raise ToolFailure(f"MCP server {name!r} crashed and could not be restarted: {st.last_error}")

    # ------------------------------------------------------------------ calls

    def _make_handler(self, server: str, tool_name: str):
        async def handler(ctx: ToolContext, args: dict[str, Any]) -> Any:
            return await self.call(server, tool_name, args)
        return handler

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        """Call ``tool`` on ``server`` with crash recovery, timeout and concurrency limits."""
        st = self._servers.get(server)
        if st is None:
            raise ToolFailure(f"unknown MCP server {server!r}")
        st.queued += 1
        try:
            if st.sem is not None:
                await st.sem.acquire()
        finally:
            st.queued -= 1
        st.in_flight += 1
        st.calls += 1
        try:  # the semaphore (if any) is held from here on
            for attempt in (0, 1):
                session, generation = await self._ready_session(st)
                try:
                    result = await self._invoke(st, session, tool, args)
                except ToolFailure:
                    raise
                except Exception as exc:  # noqa: BLE001 - classified below
                    if not is_transport_error(exc):
                        raise ToolFailure(f"{server}.{tool} failed: {_describe(exc)}") from exc
                    reason = _describe(exc)
                    self._emit("mcp_crash", server=server, tool=tool, error=reason[:1000],
                               stderr_tail=self.stderr_tail(server, 1000))
                    if attempt == 0:
                        await self._restart(st, generation, reason)
                        continue
                    raise ToolFailure(f"{server}.{tool}: the server connection failed again after a "
                                      f"restart ({reason}); the server will be restarted on the next call"
                                      ) from exc
                return self._convert(server, tool, result)
            raise ToolFailure(f"{server}.{tool}: no result")  # pragma: no cover - loop always returns/raises
        finally:
            st.in_flight -= 1
            if st.sem is not None:
                st.sem.release()

    async def _invoke(self, st: _Server, session: Any, tool: str, args: dict[str, Any]) -> Any:
        import anyio

        timeout = float(st.cfg.timeout_s or self.options["default_timeout_s"])
        # mcp 1.x numbers requests with this counter and does not cancel abandoned
        # requests itself; mcp 2.x sends notifications/cancelled automatically.
        request_id = getattr(session, "_request_id", None)
        try:
            with anyio.fail_after(timeout):
                return await session.call_tool(tool, args)
        except TimeoutError:
            if isinstance(request_id, int):
                await self._send_cancel(session, request_id)
            self._emit("mcp_timeout", server=st.cfg.name, tool=tool, timeout_s=timeout)
            raise ToolFailure(f"{st.cfg.name}.{tool} timed out after {timeout:g}s; {TIMEOUT_HINT}") from None

    @staticmethod
    async def _send_cancel(session: Any, request_id: int) -> None:
        try:
            import anyio
            from mcp import types as mt

            note = mt.ClientNotification(mt.CancelledNotification(
                method="notifications/cancelled",
                params=mt.CancelledNotificationParams(requestId=request_id, reason="client timeout")))
            with anyio.move_on_after(5):
                await session.send_notification(note)
        except Exception:  # noqa: BLE001 - best effort
            log.debug("could not send notifications/cancelled", exc_info=True)

    def _convert(self, server: str, tool: str, result: Any) -> Any:
        source = f"mcp__{server}__{tool}"
        items: list[tuple[str, Any]] = []   # ("text", str) | ("part", content part)
        for c in _attr(result, "content", default=[]) or []:
            ctype = getattr(c, "type", None)
            if ctype == "text":
                items.append(("text", getattr(c, "text", "") or ""))
            elif ctype == "image":
                mime = _attr(c, "mime_type", "mimeType", default="image/png")
                data = getattr(c, "data", "") or ""
                part = _image_part(mime, data, source)
                items.append(("part", part) if part is not None else
                             ("text", f"[image omitted: {mime}, {len(data) * 3 // 4:,} bytes]"))
            elif ctype == "resource":
                res = getattr(c, "resource", None)
                uri = str(getattr(res, "uri", "") or "")
                mime = _attr(res, "mime_type", "mimeType", default="") or ""
                if getattr(res, "text", None) is not None:
                    items.append(("text", (f"[resource {uri}]\n" if uri else "") + res.text))
                elif getattr(res, "blob", None) and mime.startswith("image/"):
                    part = _image_part(mime, res.blob, uri or source)
                    items.append(("part", part) if part is not None else
                                 ("text", f"[image resource omitted: {uri} ({mime})]"))
                else:
                    items.append(("text", f"[binary resource omitted: {uri} ({mime or 'unknown type'})]"))
            elif ctype == "resource_link":
                items.append(("text", f"[resource link: {getattr(c, 'uri', '')}]"))
            else:
                items.append(("text", f"[{ctype or 'unknown'} content omitted]"))

        texts = [v for k, v in items if k == "text"]
        text = "\n".join(texts)
        structured = _attr(result, "structured_content", "structuredContent")
        if not text and structured is not None and not any(k == "part" for k, _ in items):
            text = json.dumps(structured, default=str)
        if _attr(result, "is_error", "isError", default=False):
            raise ToolFailure(text or "MCP tool reported an error")
        error = tool_result_error(text) if text else None
        if error is None and structured is not None:
            error = tool_result_error(structured if isinstance(structured, dict) else None)
        if error:
            raise ToolFailure(json.dumps({"status": "tool_error", "tool": tool, "error": error,
                                          "instruction": FAILURE_INSTRUCTION}))
        if any(k == "part" for k, _ in items):
            return [(_text_block(v) if k == "text" else v) for k, v in items]
        return text

    # ------------------------------------------------------------------ status / shutdown

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "state": st.state, "restarts": st.restarts, "last_error": st.last_error,
                "in_flight": st.in_flight, "queued": st.queued, "calls": st.calls,
                "tools": len(st.tool_names), "started_at": st.started_at,
                "start_duration_s": st.start_duration_s,
                "log": str(st.log_path) if st.log_path else None,
            }
            for name, st in self._servers.items()
        }

    async def aclose(self) -> None:
        self._closed = True
        try:
            await asyncio.gather(*(self._teardown(st) for st in self._servers.values()), return_exceptions=True)
        except Exception as exc:  # noqa: BLE001 - shutdown noise from child processes
            log.debug("MCP shutdown: %s", exc)
        for st in self._servers.values():
            st.state = "closed"
        if self._tmp_log_dir:
            shutil.rmtree(self._tmp_log_dir, ignore_errors=True)
            self._tmp_log_dir = None


__all__ = ["MCPBridge", "MCPServerConfig", "DEFAULT_OPTIONS", "tool_result_error", "is_transport_error"]
