"""Contracts between ``MCPBridge`` and the data gateway (§11.2). No pyarrow.

The bridge depends only on these types and on :class:`GatewayProtocol`; with
``gateway=None`` it behaves exactly as before the data layer.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncContextManager, AsyncIterator, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from .errors import GatewayError
    from .record import DataProvenance

__all__ = [
    "GatewayMode", "Route", "Envelope", "ListingDecision", "RawResult", "CallPlan", "CrashDecision", "LaunchSpec",
    "GatewayProtocol",
]

GatewayMode = Literal["off", "observe", "enforce"]
Route = Literal["upstream", "derived", "none"]
Envelope = Literal["ok", "empty_lookup", "legacy_error", "is_error"]


@dataclass
class ListingDecision:
    """How one tool is listed to agents (description and schema rewritten from roles)."""

    visible: bool
    description: str
    input_schema: dict[str, Any]
    reason: str | None = None


@dataclass
class RawResult:
    """An upstream result before classification (``MCPBridge._convert(classify_only=True)``)."""

    text: str
    structured: Any
    parts: list[Any] | None
    envelope: Envelope
    error_text: str | None = None


@dataclass
class CallPlan:
    """What ``prepare`` decided for one call; ``finish`` completes it."""

    server: str
    tool: str
    args_raw: dict[str, Any]
    args_sent: dict[str, Any]
    route: Route
    contract: Any                                   # catalog.ToolContract
    resolutions: list[dict[str, Any]]
    witness: dict[str, Any] | None
    bound_table: str | None = None                  # after selector resolution (rev 2)
    scope: dict[str, Any] = field(default_factory=dict)        # fixed and listed scope values (rev 2)
    existence: dict[str, Literal["exists", "absent", "unknown"]] = field(default_factory=dict)   # per argument
    gateway_args: dict[str, Any] = field(default_factory=dict)
    cold_tables: tuple[str, ...] = ()
    cold_lock: asyncio.Lock | None = None
    record: "DataProvenance | None" = None
    notes: list[str] = field(default_factory=list)
    mode: GatewayMode = "enforce"

    def hold(self) -> AsyncContextManager[None]:
        """Async context that holds the per-server cold-call lock when one is assigned."""
        return _hold(self.cold_lock)


@contextlib.asynccontextmanager
async def _hold(lock: asyncio.Lock | None) -> AsyncIterator[None]:
    if lock is None:
        yield
        return
    async with lock:
        yield


@dataclass
class CrashDecision:
    retry: bool
    error: "GatewayError | None"
    oom: bool = False


@dataclass
class LaunchSpec:
    """A stdio server command rewritten to run under the reaper launcher."""

    command: str
    args: list[str]
    env: dict[str, str]
    status_path: str


@runtime_checkable
class GatewayProtocol(Protocol):
    mode: GatewayMode

    def bind_bridge(self, bridge: Any) -> None: ...

    def extra_servers(self) -> list[dict[str, Any]]: ...

    def launch_spec(self, cfg: Any) -> LaunchSpec | None: ...

    def rewrite_listing(self, server: str, tool: str, description: str,
                        input_schema: dict[str, Any]) -> ListingDecision: ...

    async def prepare(self, server: str, tool: str, args: dict[str, Any], ctx: Any) -> CallPlan:
        """Raises GatewayError (never in observe mode)."""
        ...

    async def finish(self, plan: CallPlan, raw: RawResult | None) -> Any:
        """Returns a DataResult; raises GatewayError."""
        ...

    async def on_crash(self, server: str, plan: CallPlan, reason: str, log_tail: str) -> CrashDecision: ...

    def pinned(self) -> dict[str, Any]: ...

    def readiness_snapshot(self) -> dict[str, Any]: ...
