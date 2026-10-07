"""Typed calls to the data child's hidden verbs (§11.8). No pyarrow.

Every verb is one ``MCPBridge.call_raw("data", verb, {"request": ...})``: the request is an
:mod:`vbt.datalayer.ipc` model and the JSON reply is validated with the verb's response model.
The gateway never reads data itself; witness scans, derived serving, readiness checks,
vocabulary snapshots and resolver index builds all go through this client.

A failed call (the child is down, crashed, timed out or answered with an error) raises
:class:`ServiceError`, a ``service_unavailable`` :class:`~vbt.datalayer.errors.GatewayError`.
Under ``data.gateway.when_service_down: strict`` the gateway lets it propagate for guarded
tools; ``lenient`` makes it fall back to the generic guard (§11.9).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..errors import ErrorKind, GatewayError
from ..ipc import (
    VERB_BUILD_INDEX,
    VERB_CHECK,
    VERB_RESOLVE_REMOTE,
    VERB_SERVE,
    VERB_STATS,
    VERB_VOCAB,
    VERB_WITNESS,
    BuildIndexRequest,
    BuildIndexResponse,
    CheckRequest,
    CheckResponse,
    IpcModel,
    ResolveRemoteRequest,
    ResolveRemoteResponse,
    ServeRequest,
    ServeResponse,
    StatsRequest,
    StatsResponse,
    VocabRequest,
    VocabResponse,
    WitnessRequest,
    WitnessResponse,
    parse_response,
    request_payload,
)

__all__ = ["DATA_SERVER", "ServiceError", "ServiceClient"]

DATA_SERVER = "data"


class ServiceError(GatewayError):
    """The data child could not answer (``service_unavailable``)."""

    def __init__(self, message: str, *, verb: str | None = None, tool: str | None = None) -> None:
        super().__init__(ErrorKind.service_unavailable, message, tool=tool, payload={"verb": verb} if verb else None)
        self.verb = verb


def _text_of(result: Any) -> Any:
    """The JSON body of a ``call_raw`` result: text, a dict, or MCP content parts."""
    if isinstance(result, list):
        texts = [p.get("text") if isinstance(p, Mapping) else getattr(p, "text", None) for p in result]
        return "\n".join(t for t in texts if t)
    return result


class ServiceClient:
    """Typed client of the data child. ``bridge`` is the :class:`~vbt.tools.mcp_bridge.MCPBridge`
    (bound later with :meth:`bind`); ``server`` is the data child's server name."""

    def __init__(self, bridge: Any = None, *, server: str = DATA_SERVER) -> None:
        self.bridge = bridge
        self.server = server
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None

    def bind(self, bridge: Any) -> None:
        self.bridge = bridge

    @property
    def available(self) -> bool:
        """The bridge knows the data child and it has not failed to start."""
        b = self.bridge
        if b is None:
            return False
        servers = getattr(b, "_servers", None)
        if isinstance(servers, Mapping) and self.server not in servers:
            return False
        failures = getattr(b, "failures", None) or {}
        return self.server not in failures

    async def call(self, verb: str, request: IpcModel) -> IpcModel:
        """Run ``verb`` with ``request``; the validated response, or :class:`ServiceError`."""
        self.calls += 1
        if self.bridge is None:
            self.failures += 1
            self.last_error = "no MCP bridge is bound to the gateway"
            raise ServiceError(f"data child unavailable for {verb}: {self.last_error}", verb=verb)
        try:
            result = await self.bridge.call_raw(self.server, verb, request_payload(request))
        except GatewayError as exc:
            self.failures += 1
            self.last_error = exc.message
            raise ServiceError(f"data child failed on {verb}: {exc.message}", verb=verb) from exc
        except Exception as exc:  # noqa: BLE001 - every child failure is service_unavailable
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:500]
            raise ServiceError(f"data child failed on {verb}: {self.last_error}", verb=verb) from exc
        body = _text_of(result)
        try:
            if isinstance(body, str):
                body = json.loads(body)
            return parse_response(verb, body)
        except Exception as exc:  # noqa: BLE001 - a malformed reply is a child defect
            self.failures += 1
            self.last_error = f"malformed {verb} reply: {exc}"[:500]
            raise ServiceError(f"data child returned a malformed {verb} reply: {exc}"[:500], verb=verb) from exc

    async def try_call(self, verb: str, request: IpcModel) -> IpcModel | None:
        """:meth:`call`, with None instead of :class:`ServiceError`."""
        try:
            return await self.call(verb, request)
        except ServiceError:
            return None

    # ------------------------------------------------------------------ typed verbs

    async def stats(self, tables: Sequence[str]) -> StatsResponse:
        return await self.call(VERB_STATS, StatsRequest(tables=list(tables)))  # type: ignore[return-value]

    async def check(self, tables: Sequence[str] = (), depth: str = "standard") -> CheckResponse:
        return await self.call(VERB_CHECK, CheckRequest(tables=list(tables), depth=depth))  # type: ignore

    async def witness(self, request: WitnessRequest) -> WitnessResponse:
        return await self.call(VERB_WITNESS, request)  # type: ignore[return-value]

    async def serve(self, request: ServeRequest) -> ServeResponse:
        return await self.call(VERB_SERVE, request)  # type: ignore[return-value]

    async def build_index(self, source: str, id_type: str, *, force: bool = False) -> BuildIndexResponse:
        req = BuildIndexRequest(source=source, id_type=id_type, force=force)
        return await self.call(VERB_BUILD_INDEX, req)  # type: ignore[return-value]

    async def resolve_remote(self, source: str, id_type: str, values: Sequence[str],
                             accepts: Sequence[str] = ()) -> ResolveRemoteResponse:
        req = ResolveRemoteRequest(source=source, id_type=id_type, values=[str(v) for v in values],
                                   accepts=list(accepts))
        return await self.call(VERB_RESOLVE_REMOTE, req)  # type: ignore[return-value]

    async def vocab(self, table: str, column: str, max_values: int | None = None) -> VocabResponse:
        return await self.call(VERB_VOCAB, VocabRequest(table=table, column=column,  # type: ignore[return-value]
                                                        max_values=max_values))
