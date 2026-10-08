"""Crash and out-of-memory classification (§14.4). No pyarrow.

| Signal | Classified as | Retry |
|---|---|---|
| error text with a memory signature | ``oom`` | no |
| transport error + ``VBT_CHILD_EXIT`` with signal 9, rc 137 or ``reason: memory_limit`` | ``oom`` (``oom_killed``) | no |
| transport error + a memory signature in the server's stderr before its exit (an uncaught ``MemoryError`` or ``std::bad_alloc`` under ``RLIMIT_DATA``, which fails the allocation instead of killing) | ``oom`` | no |
| transport error otherwise | ``server_crashed`` | once |

The exit marker is written by the reaper (:mod:`vbt.datalayer.launch.reaper`) to the server
log; the bridge waits up to 500 ms for it after a transport error and passes the log tail to
``gateway.on_crash``, which uses :func:`crash_decision`. The reaper tees the child's stderr and
names the cause of a memory exit itself (``cause``: ``watchdog``, ``cgroup_oom_kill``,
``memory_error``, ``kernel_oom_kill``, ``peak_rss``; INV-1); the error's payload and message carry it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from ..api import CrashDecision
from ..errors import ErrorKind, GatewayError

__all__ = ["MEMORY_SIGNATURES", "MEMORY_PATTERNS", "EXIT_MARKER", "OOM_EXIT_CODES", "OOM_SIGNALS",
           "EXIT_DETAIL", "classify_error_text", "parse_exit_marker", "is_memory_exit", "crash_decision"]

#: Error text that means an allocation failed (numpy/pandas, Arrow, C++, ENOMEM).
MEMORY_SIGNATURES: tuple[str, ...] = (
    "MemoryError", "Unable to allocate", "ArrowMemoryError", "bad_alloc", "Cannot allocate memory",
)
#: The signatures as patterns, plus errno ENOMEM, Rust's and Arrow's allocation failures and "out of memory".
#: The reaper (stdlib only) keeps an identical tuple to name a memory exit from the child's stderr.
MEMORY_PATTERNS: tuple[str, ...] = (
    r"MemoryError", r"Unable to allocate", r"bad_alloc", r"Cannot allocate memory", r"\bENOMEM\b",
    r"(?i:\bmemory allocation\b.{0,40}\bfailed\b)", r"\b(?:m|re|c)alloc of size \d+ failed", r"(?i:\bout of memory\b)",
)
EXIT_MARKER = "VBT_CHILD_EXIT"
OOM_EXIT_CODES = frozenset({137})          # 128 + SIGKILL, as shells and container runtimes report it
OOM_SIGNALS = frozenset({9})               # SIGKILL: the kernel OOM killer or a memory watchdog
#: Exit-marker fields an error payload carries.
EXIT_DETAIL = ("pid", "code", "signal", "maxrss_kb", "reason", "cause", "memory_error")
_CAUSES = {"watchdog": "was killed by the memory watchdog", "cgroup_oom_kill": "was killed at its cgroup's memory limit",
           "kernel_oom_kill": "was killed by the kernel's OOM killer",
           "memory_error": "ran out of memory (an allocation failed under its memory limit)",
           "peak_rss": "ran out of memory (it ended at its memory limit)"}

_SIGNATURE_RE = re.compile("|".join(MEMORY_PATTERNS))
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


def _before_exit(text: str | None) -> str:
    """The log tail up to its last exit marker: this start's stderr (an exit marker of an earlier start
    precedes it only when the tail spans a restart, and then the latest one ends the current start)."""
    if not text:
        return ""
    text = str(text)
    hits = list(_MARKER_RE.finditer(text))
    if not hits:
        return text
    end = hits[-1].start()
    start = hits[-2].end() if len(hits) > 1 else 0
    return text[start:end]


def died_of_memory(marker: dict[str, Any] | None, log_tail: str | None) -> bool:
    """The child died (rc != 0 or a signal) and the last stderr lines before its exit carry a memory
    signature: an uncaught ``MemoryError`` or ``std::bad_alloc`` (SIGABRT) under ``RLIMIT_DATA``, which
    fails the allocation instead of killing. A memory error a tool survived (more output followed it)
    is not the cause."""
    if not marker or (marker.get("code") in (0, None) and not marker.get("signal")):
        return False
    lines = [ln for ln in _before_exit(log_tail).splitlines() if ln.strip()]
    return bool(lines) and classify_error_text(lines[-1]) == "oom"   # the exception that ended the process


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
    killed = is_memory_exit(marker)
    if killed or classify_error_text(reason) or died_of_memory(marker, log_tail):
        detail = {k: marker.get(k) for k in EXIT_DETAIL if k in marker or k in EXIT_DETAIL[:5]} if marker else {}
        payload = {"server": server, "exit": detail} if detail else {"server": server}
        cause = marker.get("cause") if marker else None
        if cause in _CAUSES:
            how = _CAUSES[cause]
        elif killed and marker and marker.get("signal"):
            how = f"was killed (signal {marker.get('signal')})"
        else:
            how = "ran out of memory"
        return CrashDecision(retry=False, oom=True, error=GatewayError(
            ErrorKind.oom, f"{where}: the server {how} while answering this call; it is not retried and "
            f"the server restarts on the next call", tool=name, payload=payload, subkind="oom_killed"))
    retry = attempt == 0
    payload = {"server": server}
    if marker:
        payload["exit"] = {k: marker.get(k) for k in EXIT_DETAIL if k in marker or k in EXIT_DETAIL[:5]}
    return CrashDecision(retry=retry, oom=False, error=GatewayError(
        ErrorKind.server_crashed, f"{where}: the server connection failed ({reason})", tool=name, payload=payload))
