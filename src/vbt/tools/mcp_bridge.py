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
  queries, so a retry is safe. When the retried call fails too, the session is
  torn down and the server marked ``broken``; the next call restarts it first
  instead of reusing the dead pipe.
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

* **Data gateway** (``docs/DATA_LAYER.md`` §11) -- with ``gateway=`` (a
  :class:`vbt.datalayer.api.GatewayProtocol`) every call goes through
  ``gateway.prepare`` once (a retry reuses its resolved arguments), runs upstream
  under the plan's cold-call lock, and is classified and completed by
  ``gateway.finish``; transport failures are put to ``gateway.on_crash`` (an OOM
  kill is never retried and counts against ``mcp.max_oom_kills`` instead of
  ``max_restarts``). Stdio servers are launched as ``gateway.launch_spec(cfg)``
  says (the reaper launcher), listings pass through ``gateway.rewrite_listing``
  (hidden tools are not registered) and ``recycle()`` restarts an idle server to
  free its memory. Internal ``_``-prefixed tools of the ``data`` server and
  :meth:`MCPBridge.call_raw` bypass the gateway. With ``gateway=None`` the
  bridge behaves exactly as before the data layer.

``on_event(kind, **data)`` (e.g. ``Runtime.emit`` or ``Run.trace``) receives
``mcp_server_started``, ``mcp_start_failed``, ``mcp_crash``, ``mcp_restart``,
``mcp_timeout``, ``mcp_lazy_retry`` and ``mcp_recycle`` events.
``on_tools_changed(tools)`` receives tools that were registered or whose listing
changed (a restarted or late-starting server), so they reach the agents' registry.
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
from typing import TYPE_CHECKING, Any, Callable

from ..datalayer.api import RawResult
from ..envpolicy import child_env
from .base import Tool, ToolContext, ToolFailure, inline_refs

if TYPE_CHECKING:  # pragma: no cover
    from ..datalayer.api import CallPlan, CrashDecision, GatewayProtocol

log = logging.getLogger(__name__)

DEFAULT_OPTIONS: dict[str, Any] = {
    "start_attempts": 3,
    "start_timeout_s": 180.0,
    "start_backoff_s": 3.0,
    "start_backoff_factor": 1.5,
    "default_timeout_s": 1800.0,
    "max_restarts": 3,
    "max_oom_kills": 3,
    "inherit_env": False,
}

STDERR_TAIL_CHARS = 2000
_CONNECTION_CLOSED = -32000  # JSON-RPC code the MCP SDK uses for a closed connection

DATA_SERVER = "data"              # the harness-owned data child; its "_" tools are internal verbs
EXIT_MARKER = "VBT_CHILD_EXIT"    # written to the server log by the reaper launcher when the child ends
EXIT_MARKER_WAIT_S = 0.5          # the pipe can close before the reaper writes the marker

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
    # Data layer (docs/DATA_LAYER.md §11.1, §14.2); runtime and preflight keep only declared fields.
    mem_limit_mb: int | str | None = None   # MB or "auto"; None -> data.memory.default_server_mb
    overlay: str | None = None              # None -> configs/data/overlays/<name>.yaml
    sources: list[str] = field(default_factory=list)
    launcher: bool | None = None            # False: launch without the reaper (no memory limit)


class _ConfigError(Exception):
    """A startup problem retrying cannot fix (missing interpreter or script)."""


@dataclass
class _Server:
    cfg: MCPServerConfig
    state: str = "stopped"            # stopped | starting | ready | broken | failed | closed
    session: Any = None
    generation: int = 0               # bumped on every successful (re)start
    restarts: int = 0
    oom_kills: int = 0                # memory kills (separate budget: options.max_oom_kills)
    broken_oom: bool = False          # the session broke because of a memory kill
    recycles: int = 0
    status_path: str | None = None    # reaper status file when launched under the launcher
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


def _lookup_miss(result: Any) -> bool:
    """True when the result envelope is one of the upstream's explicit empty-lookup messages
    (the envelopes :func:`tool_result_error` exempts), inspecting the same places it does."""
    if isinstance(result, str):
        try:
            return _lookup_miss(json.loads(result))
        except (ValueError, TypeError):
            return False
    if isinstance(result, (list, tuple)):
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
            elif getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
            else:
                continue
            if _lookup_miss(text):
                return True
        return False
    if not isinstance(result, dict):
        return False
    if result.get("type") == "text":
        return _lookup_miss(result.get("text"))
    if "structuredContent" in result and _lookup_miss(result["structuredContent"]):
        return True
    if result.get("type") == "tool_result" or "isError" in result:
        return _lookup_miss(result.get("content"))
    error = result.get("error") or result.get("errors")
    if not error and result.get("success") is not False and result.get("ok") is not False:
        return False
    message = "; ".join(str(item) for item in error) if isinstance(error, list) else str(error or "")
    return bool(_EMPTY_LOOKUP.fullmatch(message.strip()))


def legacy_result(raw: RawResult, tool: str) -> Any:
    """What the bridge returns (or raises) for ``raw`` without a gateway: text, content parts,
    or a ``ToolFailure`` for ``isError`` and legacy failure envelopes. Empty lookups are results.
    A gateway in ``observe`` mode returns this to stay byte-identical with no gateway."""
    if raw.envelope == "is_error":
        raise ToolFailure(raw.error_text or raw.text or "MCP tool reported an error")
    if raw.envelope == "legacy_error":
        raise ToolFailure(json.dumps({"status": "tool_error", "tool": tool, "error": raw.error_text,
                                      "instruction": FAILURE_INSTRUCTION}))
    if raw.parts is not None:
        return raw.parts
    return raw.text


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
                 on_event: Callable[..., Any] | None = None, gateway: "GatewayProtocol | None" = None,
                 on_tools_changed: Callable[[list[Tool]], Any] | None = None):
        self.servers = [s for s in servers if s.enabled]
        self.extra_env = {k: str(v) for k, v in (extra_env or {}).items() if v is not None}
        self.options = {**DEFAULT_OPTIONS, **{k: v for k, v in (options or {}).items() if v is not None}}
        self.on_event = on_event
        self.on_tools_changed = on_tools_changed
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
        self.gateway = gateway
        if gateway is not None:
            gateway.bind_bridge(self)

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

    def _tools_changed(self, tools: list[Tool]) -> None:
        if self.on_tools_changed is None or not tools:
            return
        try:
            r = self.on_tools_changed(list(tools))
            if asyncio.iscoroutine(r):
                asyncio.ensure_future(r)
        except Exception:  # noqa: BLE001 - observers must not break server starts
            log.warning("on_tools_changed callback failed", exc_info=True)

    def log_root(self) -> Path:
        """The directory server logs (and the launcher's status files) are written to."""
        if self.log_dir is None:
            if self._tmp_log_dir is None:
                self._tmp_log_dir = tempfile.mkdtemp(prefix="vbt-mcp-logs-")
            base = Path(self._tmp_log_dir)
        else:
            base = self.log_dir
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _log_file_for(self, st: _Server) -> Path:
        return self.log_root() / f"{st.cfg.name}.log"

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

            command, args, env = cfg.command, [str(a) for a in cfg.args], self.child_env(cfg)
            # The gateway may run the server under the reaper launcher (memory limit, status
            # file, exit marker, effective hash seed); HTTP servers are never relaunched.
            spec = self.gateway.launch_spec(cfg) if self.gateway is not None else None
            st.status_path = spec.status_path if spec is not None else None
            if spec is not None:
                command, args, env = spec.command, [str(a) for a in spec.args], {**env, **spec.env}
            params = StdioServerParameters(command=command, args=args, env=env, cwd=cfg.cwd)
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

    def _listing(self, server: str, tool: str, description: str, schema: dict[str, Any]) -> tuple[bool, str, dict]:
        """``(visible, description, input_schema)`` after ``gateway.rewrite_listing``."""
        if self.gateway is None:
            return True, description, schema
        try:
            decision = self.gateway.rewrite_listing(server, tool, description, schema)
        except Exception:  # noqa: BLE001 - keep the upstream listing, but never expose internal verbs
            log.warning("rewrite_listing failed for %s.%s", server, tool, exc_info=True)
            return not (server == DATA_SERVER and tool.startswith("_")), description, schema
        return bool(decision.visible), decision.description, decision.input_schema

    def _register(self, st: _Server, listed: Any) -> list[Tool]:
        """Register the listed tools; returns the new ones. A tool already registered (a restart,
        a late start) is updated in place, so the registry's object stays valid."""
        known = {t.name: t for t in self.tools}
        new: list[Tool] = []
        updated: list[Tool] = []
        names = []
        for t in getattr(listed, "tools", []) or []:
            names.append(t.name)
            full = f"mcp__{st.cfg.name}__{t.name}"
            schema = _attr(t, "input_schema", "inputSchema") or {"type": "object"}
            visible, description, input_schema = self._listing(
                st.cfg.name, t.name, (t.description or t.name).strip(), inline_refs(dict(schema)))
            if not visible:
                continue
            tool = known.get(full)
            if tool is not None:
                if tool.description != description or tool.input_schema != input_schema:
                    tool.description, tool.input_schema = description, input_schema
                    updated.append(tool)
                tool.handler = self._make_handler(st.cfg.name, t.name)
                continue
            tool = Tool(
                name=full,
                description=description,
                input_schema=input_schema,
                handler=self._make_handler(st.cfg.name, t.name),
                source=f"mcp:{st.cfg.name}",
            )
            self.tools.append(tool)
            known[full] = tool
            new.append(tool)
        st.tool_names = names
        self._tools_changed(new + updated)
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
            if st.state == "broken":
                await self._restart_broken(st)
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
            if st.state == "broken":  # another call gave up on this session; restart it under its budget
                await self._restart_broken(st)
                return
            max_restarts = int(self.options["max_restarts"])
            if st.restarts >= max_restarts:
                await self._teardown(st)
                self._give_up(st, f"crashed {st.restarts + 1} times; restart limit ({max_restarts}) reached: {reason}")
                raise ToolFailure(f"MCP server {name!r} crashed and its restart limit is reached: {reason}")
            st.restarts += 1
            self._emit("mcp_restart", server=name, restarts=st.restarts, reason=reason[:1000])
            log.warning("MCP server %s connection lost (%s); restarting (%d/%d)", name, reason,
                        st.restarts, max_restarts)
            await self._teardown(st)
            if not await self._start(st, reason="restart"):
                raise ToolFailure(f"MCP server {name!r} crashed and could not be restarted: {st.last_error}")

    def _give_up(self, st: _Server, error: str) -> None:
        """Mark ``st`` failed for good (no lazy retry). Caller holds ``st.lock``."""
        st.state = "failed"
        st.lazy_retry_used = True  # no lazy retry after the restart budget is spent
        st.last_error = error
        self.failures[st.cfg.name] = error

    async def _mark_broken(self, st: _Server, generation: int, reason: str, *, oom: bool = False) -> None:
        """Tear down a session whose call failed for good, so the next call restarts it first
        instead of hitting the dead pipe. A memory kill counts against ``max_oom_kills``."""
        async with st.lock:
            if st.generation != generation or st.state != "ready":
                return  # another call already restarted, broke or gave up on this session
            await self._teardown(st)
            st.last_error = reason
            st.broken_oom = oom
            if oom:
                st.oom_kills += 1
                limit = int(self.options["max_oom_kills"])
                if st.oom_kills > limit:
                    self._give_up(st, f"killed for memory {st.oom_kills} times; OOM kill limit ({limit}) "
                                      f"reached: {reason}")
                    return
            st.state = "broken"

    async def _restart_broken(self, st: _Server) -> None:
        """Restart a ``broken`` server. Caller holds ``st.lock``. Counts against ``max_restarts``
        unless the session broke because of a memory kill (that budget was charged already)."""
        name = st.cfg.name
        reason = st.last_error or "the previous call failed"
        if not st.broken_oom:
            max_restarts = int(self.options["max_restarts"])
            if st.restarts >= max_restarts:
                self._give_up(st, f"crashed {st.restarts + 1} times; restart limit ({max_restarts}) reached: {reason}")
                raise ToolFailure(f"MCP server {name!r} crashed and its restart limit is reached: {reason}")
            st.restarts += 1
        self._emit("mcp_restart", server=name, restarts=st.restarts, reason=reason[:1000], broken=True,
                   oom=st.broken_oom, oom_kills=st.oom_kills)
        log.warning("MCP server %s: restarting a broken session (%s)", name, reason.splitlines()[0][:200])
        st.broken_oom = False
        if not await self._start(st, reason="restart"):
            raise ToolFailure(f"MCP server {name!r} crashed and could not be restarted: {st.last_error}")

    async def recycle(self, server: str, wait_s: float = 30.0) -> bool:
        """Restart an idle server to free the memory it holds (upstream caches every table it
        loads). Waits up to ``wait_s`` for in-flight calls to finish; does not count toward
        ``max_restarts``. Returns True when nothing of the old process is resident any more."""
        st = self._servers.get(server)
        if st is None:
            raise ToolFailure(f"unknown MCP server {server!r}")
        t0 = time.monotonic()
        deadline = t0 + max(0.0, float(wait_s))
        while True:
            async with st.lock:
                if self._closed:
                    return False
                if st.in_flight == 0:
                    if st.state != "ready":  # stopped, broken or failed: no process holds memory
                        self._emit("mcp_recycle", server=server, ok=True, skipped=st.state,
                                   generation=st.generation, waited_s=round(time.monotonic() - t0, 3))
                        return True
                    st.recycles += 1
                    await self._teardown(st)
                    ok = await self._start(st, reason="recycle")
                    self._emit("mcp_recycle", server=server, ok=ok, generation=st.generation,
                               recycles=st.recycles, waited_s=round(time.monotonic() - t0, 3))
                    return ok
            if time.monotonic() >= deadline:
                self._emit("mcp_recycle", server=server, ok=False, reason="busy", in_flight=st.in_flight,
                           waited_s=round(time.monotonic() - t0, 3))
                return False
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------ calls

    def _make_handler(self, server: str, tool_name: str):
        async def handler(ctx: ToolContext, args: dict[str, Any]) -> Any:
            return await self.call(server, tool_name, args, ctx=ctx)
        return handler

    async def call(self, server: str, tool: str, args: dict[str, Any], ctx: Any = None) -> Any:
        """Call ``tool`` on ``server`` through the gateway (when there is one) with crash recovery,
        timeout and concurrency limits.

        The gateway prepares the call once (resolution, contracts, admission; ``GatewayError``
        propagates), the upstream attempts run under the plan's cold-call lock with the prepared
        arguments, and ``gateway.finish`` classifies and completes the raw result. A ``derived``
        or ``none`` route never reaches the upstream server. Without a gateway, and for the data
        child's internal ``_`` verbs, this is :meth:`call_raw`.
        """
        gw = self.gateway
        if gw is None or (server == DATA_SERVER and tool.startswith("_")):
            return await self.call_raw(server, tool, args)
        if server not in self._servers:
            raise ToolFailure(f"unknown MCP server {server!r}")
        plan = await gw.prepare(server, tool, args, ctx)
        raw: RawResult | None = None
        if plan.route == "upstream":
            async with plan.hold():
                raw = await self._attempts(server, tool, plan.args_sent, plan=plan, classify_only=True)
        return await gw.finish(plan, raw)

    async def call_raw(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        """Call ``tool`` on ``server`` without the gateway (internal data-child calls): crash
        recovery, timeout and concurrency limits, legacy result semantics."""
        return await self._attempts(server, tool, args)

    async def _attempts(self, server: str, tool: str, args: dict[str, Any], *, plan: "CallPlan | None" = None,
                        classify_only: bool = False) -> Any:
        """The attempt loop: a transport failure restarts the server and retries once, unless the
        crash decision says not to (a memory kill is never retried)."""
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
                    if st.status_path is not None:
                        await self._wait_exit_marker(st)
                    self._emit("mcp_crash", server=server, tool=tool, error=reason[:1000],
                               stderr_tail=self.stderr_tail(server, 1000))
                    decision = await self._crash_decision(st, plan, tool, reason, attempt)
                    if decision is not None and decision.oom:
                        await self._mark_broken(st, generation, reason, oom=True)
                        raise (decision.error or ToolFailure(
                            f"{server}.{tool}: the server was killed for memory ({reason}); the call is not "
                            f"retried and the server restarts on the next call")) from exc
                    if attempt == 0 and (decision is None or decision.retry):
                        await self._restart(st, generation, reason)
                        continue
                    await self._mark_broken(st, generation, reason)
                    if decision is not None and decision.error is not None:
                        raise decision.error from exc
                    if attempt == 0:
                        raise ToolFailure(f"{server}.{tool}: the server connection failed ({reason}); the call "
                                          f"is not retried and the server restarts on the next call") from exc
                    raise ToolFailure(f"{server}.{tool}: the server connection failed again after a "
                                      f"restart ({reason}); the server will be restarted on the next call"
                                      ) from exc
                return self._convert(server, tool, result, classify_only=classify_only)
            raise ToolFailure(f"{server}.{tool}: no result")  # pragma: no cover - loop always returns/raises
        finally:
            st.in_flight -= 1
            if st.sem is not None:
                st.sem.release()

    async def _wait_exit_marker(self, st: _Server) -> None:
        """Wait up to ``EXIT_MARKER_WAIT_S`` for the launcher's exit marker in the server log
        (the pipe can close before the reaper has reaped the child)."""
        deadline = time.monotonic() + EXIT_MARKER_WAIT_S
        while EXIT_MARKER not in self.stderr_tail(st.cfg.name) and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

    async def _crash_decision(self, st: _Server, plan: "CallPlan | None", tool: str, reason: str,
                              attempt: int) -> "CrashDecision | None":
        """The gateway's decision for a planned call; for unplanned calls to a launched server, the
        exit marker's (a memory kill is not retried); otherwise None (today's retry-once rule)."""
        name = st.cfg.name
        if self.gateway is not None and plan is not None:
            return await self.gateway.on_crash(name, plan, reason, self.stderr_tail(name))
        if st.status_path is not None:
            from ..datalayer.memory.crash import crash_decision
            decision = crash_decision(reason, self.stderr_tail(name), server=name, tool=tool, attempt=attempt)
            return decision if decision.oom else None
        return None

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

    def _convert(self, server: str, tool: str, result: Any, classify_only: bool = False) -> Any:
        """Convert an MCP result to text or content parts. Failures raise ``ToolFailure``, except
        with ``classify_only``: then a :class:`RawResult` is returned whose ``envelope`` says what
        the bridge would have done (``is_error``, ``legacy_error``, ``empty_lookup`` or ``ok``), so
        the gateway owns classification."""
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
        parts = ([(_text_block(v) if k == "text" else v) for k, v in items]
                 if any(k == "part" for k, _ in items) else None)
        if _attr(result, "is_error", "isError", default=False):
            raw = RawResult(text, structured, parts, "is_error", text or "MCP tool reported an error")
        else:
            error = tool_result_error(text) if text else None
            if error is None and structured is not None:
                error = tool_result_error(structured if isinstance(structured, dict) else None)
            if error:
                raw = RawResult(text, structured, parts, "legacy_error", error)
            elif (text and _lookup_miss(text)) or (isinstance(structured, dict) and _lookup_miss(structured)):
                raw = RawResult(text, structured, parts, "empty_lookup")
            else:
                raw = RawResult(text, structured, parts, "ok")
        if classify_only:
            return raw
        return legacy_result(raw, tool)

    # ------------------------------------------------------------------ status / shutdown

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "state": st.state, "restarts": st.restarts, "oom_kills": st.oom_kills, "recycles": st.recycles,
                "generation": st.generation, "last_error": st.last_error,
                "in_flight": st.in_flight, "queued": st.queued, "calls": st.calls,
                "tools": len(st.tool_names), "started_at": st.started_at,
                "start_duration_s": st.start_duration_s,
                "log": str(st.log_path) if st.log_path else None,
                "status_file": st.status_path,
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


__all__ = ["MCPBridge", "MCPServerConfig", "DEFAULT_OPTIONS", "DATA_SERVER", "EXIT_MARKER", "tool_result_error",
           "is_transport_error", "legacy_result"]
