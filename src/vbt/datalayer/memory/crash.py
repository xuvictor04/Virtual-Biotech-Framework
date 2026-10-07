"""Crash and out-of-memory classification (§14.4). No pyarrow.

| Signal | Classified as | Retry |
|---|---|---|
| error text with a memory signature | ``oom`` | no |
| transport error + ``VBT_CHILD_EXIT`` with signal 9, rc 137 or ``reason: memory_limit`` | ``oom`` (``oom_killed``) | no |
| transport error otherwise | ``server_crashed`` | once |

The exit marker is written by the reaper (:mod:`vbt.datalayer.launch.reaper`) to the server
log; the bridge waits up to 500 ms for it after a transport error and passes the log tail to
``gateway.on_crash``, which uses :func:`crash_decision`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from ..api import CrashDecision
from ..errors import ErrorKind, GatewayError

__all__ = ["MEMORY_SIGNATURES", "EXIT_MARKER", "OOM_EXIT_CODES", "OOM_SIGNALS", "classify_error_text",
           "parse_exit_marker", "is_memory_exit", "crash_decision"]

#: Error text that means an allocation failed (numpy/pandas, Arrow, C++, ENOMEM).
MEMORY_SIGNATURES: tuple[str, ...] = (
    "MemoryError", "Unable to allocate", "ArrowMemoryError", "bad_alloc", "Cannot allocate memory",
)
EXIT_MARKER = "VBT_CHILD_EXIT"
OOM_EXIT_CODES = frozenset({137})          # 128 + SIGKILL, as shells and container runtimes report it
OOM_SIGNALS = frozenset({9})               # SIGKILL: the kernel OOM killer or a memory watchdog

_SIGNATURE_RE = re.compile("|".join(re.escape(s) for s in MEMORY_SIGNATURES))
_MARKER_RE = re.compile(rf"^{EXIT_MARKER} (\{{.*\}})\s*$", re.MULTILINE)


def classify_error_text(text: str | None) -> Literal["oom"] | None:
    """``"oom"`` when ``text`` carries a memory signature, else None."""
    if text and _SIGNATURE_RE.search(str(text)):
        return "oom"
    return None


def parse_exit_marker(text: str | None) -> dict[str, Any] | None:
    """The last ``VBT_CHILD_EXIT {...}`` record in ``text`` (a server log tail), or None."""
    if not text:
        return None
    for raw in reversed(_MARKER_RE.findall(str(text))):
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def is_memory_exit(marker: dict[str, Any] | None) -> bool:
    """True when an exit record means the child was killed for memory (signal 9, rc 137, limit)."""
    if not marker:
        return False
    return (marker.get("signal") in OOM_SIGNALS or marker.get("code") in OOM_EXIT_CODES
            or marker.get("reason") == "memory_limit")


def crash_decision(reason: str, log_tail: str | None = None, *, server: str | None = None,
                   tool: str | None = None, attempt: int = 0) -> CrashDecision:
    """Decide what a transport failure means (§14.4).

    ``reason`` is the transport error text, ``log_tail`` the server's stderr since its latest
    start (with the reaper's exit marker when there is one) and ``attempt`` the 0-based attempt
    that failed. A memory kill is ``oom`` and never retried; any other crash is
    ``server_crashed``, retried once (the bridge never retries the second attempt).
    """
    name = f"mcp__{server}__{tool}" if server and tool else tool
    where = f"{server}.{tool}" if server and tool else (server or tool or "server")
    marker = parse_exit_marker(log_tail)
    if is_memory_exit(marker) or classify_error_text(reason):
        detail = {k: marker.get(k) for k in ("pid", "code", "signal", "maxrss_kb", "reason")} if marker else {}
        payload = {"server": server, "exit": detail} if detail else {"server": server}
        how = (f"was killed (signal {marker.get('signal')})" if marker and marker.get("signal")
               else "ran out of memory")
        return CrashDecision(retry=False, oom=True, error=GatewayError(
            ErrorKind.oom, f"{where}: the server {how} while answering this call; it is not retried and "
            f"the server restarts on the next call", tool=name, payload=payload, subkind="oom_killed"))
    retry = attempt == 0
    payload = {"server": server}
    if marker:
        payload["exit"] = {k: marker.get(k) for k in ("pid", "code", "signal", "maxrss_kb", "reason")}
    return CrashDecision(retry=retry, oom=False, error=GatewayError(
        ErrorKind.server_crashed, f"{where}: the server connection failed ({reason})", tool=name, payload=payload))
