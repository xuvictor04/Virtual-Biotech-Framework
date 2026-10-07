"""The gateway seam in MCPBridge (§11.1, §11.2): prepare once, classify-only results, listing
rewrites, update in place, internal calls, byte-identical behaviour without a gateway, the
dead-session fix and recycle. Uses a stdio FastMCP fixture server; no data, no network."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from vbt.datalayer.api import CallPlan, CrashDecision, GatewayProtocol, ListingDecision, RawResult
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.tools.base import ToolFailure
from vbt.tools.mcp_bridge import MCPBridge, MCPServerConfig, legacy_result

SERVER = str(Path(__file__).resolve().parent / "servers" / "gateway_fixture_server.py")

needs_fastmcp = pytest.mark.skipif(
    importlib.util.find_spec("fastmcp") is None or importlib.util.find_spec("mcp") is None,
    reason="fastmcp/mcp not installed (pip install -e '.[mcp]')")

pytestmark = needs_fastmcp

FAST = {"start_backoff_s": 0.01, "start_backoff_factor": 1.0, "start_timeout_s": 60}


def server(name, state, *args, **kw):
    return MCPServerConfig(name=name, command=sys.executable, args=["-E", SERVER, str(state), *map(str, args)], **kw)


def calls(state: Path) -> list[dict]:
    path = state / "calls"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class Events(list):
    def __call__(self, kind, **data):
        self.append((kind, data))

    def kinds(self):
        return [k for k, _ in self]


class FakeGateway:
    """Records every hook. ``resolve`` maps argument values (stands in for resolution);
    ``hidden`` tools are not listed; ``rename`` rewrites descriptions."""

    mode = "enforce"

    def __init__(self, *, hidden=(), rename=None, resolve=None, route="upstream", raise_on=None, retry=True):
        self.hidden = set(hidden)
        self.rename = dict(rename or {})
        self.resolve = dict(resolve or {})
        self.route = route
        self.raise_on = raise_on
        self.retry = retry
        self.bridge = None
        self.prepared: list[tuple] = []
        self.finished: list[tuple] = []
        self.crashes: list[tuple] = []
        self.listed: list[tuple] = []

    def bind_bridge(self, bridge):
        self.bridge = bridge

    def extra_servers(self):
        return []

    def launch_spec(self, cfg):
        return None

    def rewrite_listing(self, server, tool, description, input_schema):
        self.listed.append((server, tool))
        if tool in self.hidden or (server == "data" and tool.startswith("_")):
            return ListingDecision(False, description, input_schema, reason="hidden")
        return ListingDecision(True, self.rename.get(tool, description), input_schema)

    async def prepare(self, server, tool, args, ctx):
        self.prepared.append((server, tool, dict(args), ctx))
        if self.raise_on == tool:
            raise GatewayError(ErrorKind.not_found, "no such target", tool=f"mcp__{server}__{tool}",
                               argument="target_id", value=args.get("target_id"))
        sent = {k: self.resolve.get(v, v) if isinstance(v, str) else v for k, v in args.items()}
        return CallPlan(server, tool, dict(args), sent, self.route, None, [], None)

    async def finish(self, plan, raw):
        self.finished.append((plan, raw))
        return {"plan": plan, "raw": raw}

    async def on_crash(self, server, plan, reason, log_tail):
        self.crashes.append((server, plan, reason, log_tail))
        return CrashDecision(retry=self.retry, error=None if self.retry else GatewayError(
            ErrorKind.server_crashed, "crashed", tool=plan.tool))

    def pinned(self):
        return {}

    def readiness_snapshot(self):
        return {}


def test_fake_gateway_satisfies_the_protocol():
    assert isinstance(FakeGateway(), GatewayProtocol)


async def test_prepare_once_across_a_retry_and_classify_only(tmp_path):
    gw = FakeGateway(resolve={"PCSK9": "ENSG00000169174"})
    state = tmp_path / "s"
    bridge = MCPBridge([server("fx", state)], log_dir=tmp_path / "logs", options=FAST, gateway=gw)
    try:
        await bridge.start()
        assert gw.bridge is bridge

        # resolution is sent upstream; the raw result is classified, not converted
        out = await bridge.call("fx", "lookup", {"target_id": "PCSK9"})
        plan, raw = out["plan"], out["raw"]
        assert plan.args_raw == {"target_id": "PCSK9"} and plan.args_sent == {"target_id": "ENSG00000169174"}
        assert isinstance(raw, RawResult) and raw.envelope == "ok" and raw.error_text is None
        assert json.loads(raw.text)["approvedSymbol"] == "PCSK9"
        assert calls(state)[-1]["args"] == {"target_id": "ENSG00000169174"}

        envelopes = {}
        for tool, args in [("lookup", {"target_id": "ENSG00000999999"}), ("legacy_error", {}),
                           ("raise_error", {"message": "kaboom"}), ("rows", {"limit": 3}), ("empty", {})]:
            envelopes[tool] = (await bridge.call("fx", tool, args))["raw"]
        assert envelopes["lookup"].envelope == "empty_lookup"
        assert "ENSG00000999999 not found" in envelopes["lookup"].text
        assert envelopes["legacy_error"].envelope == "legacy_error"
        assert envelopes["legacy_error"].error_text == "Open Targets data could not be loaded"
        assert envelopes["raise_error"].envelope == "is_error" and "kaboom" in envelopes["raise_error"].error_text
        assert envelopes["rows"].envelope == "ok" and json.loads(envelopes["rows"].text)["count"] == 3
        assert envelopes["empty"].envelope == "ok", "structural empties are the gateway's to classify"

        # crash_once: one prepare, one on_crash, one finish with the retried attempt's result
        n_prepared = len(gw.prepared)
        out = await bridge.call("fx", "crash_once", {})
        assert len(gw.prepared) == n_prepared + 1
        assert len(gw.crashes) == 1 and gw.crashes[0][1] is out["plan"]
        assert out["raw"].text == "survived after restart (start #2)"
        assert [c["tool"] for c in calls(state)].count("crash_once") == 2
        st = bridge.status()["fx"]
        assert st["restarts"] == 1 and st["state"] == "ready" and st["generation"] == 2
    finally:
        await bridge.aclose()


async def test_gateway_error_and_derived_route_never_reach_upstream(tmp_path):
    state = tmp_path / "s"
    gw = FakeGateway(raise_on="lookup")
    bridge = MCPBridge([server("fx", state)], log_dir=tmp_path / "logs", options=FAST, gateway=gw)
    try:
        await bridge.start()
        with pytest.raises(GatewayError) as exc:
            await bridge.call("fx", "lookup", {"target_id": "PCSK99"})
        assert exc.value.kind is ErrorKind.not_found and gw.finished == []
        gw.route = "derived"
        out = await bridge.call("fx", "rows", {"limit": 2})
        assert out["raw"] is None and out["plan"].route == "derived"
        assert calls(state) == [] and bridge.status()["fx"]["calls"] == 0
        with pytest.raises(ToolFailure, match="unknown MCP server"):
            await bridge.call("nope", "x", {})
    finally:
        await bridge.aclose()


async def test_handler_passes_ctx_and_listing_is_rewritten(tmp_path):
    gw = FakeGateway(hidden={"crash_always"}, rename={"rows": "Rows (derived text)."})
    changed: list[list] = []
    bridge = MCPBridge([server("fx", tmp_path / "s"), server("data", tmp_path / "d")], log_dir=tmp_path / "logs",
                       options=FAST, gateway=gw, on_tools_changed=changed.append)
    try:
        await bridge.start()
        names = {t.name for t in bridge.tools}
        assert "mcp__fx__rows" in names and "mcp__fx__crash_always" not in names
        assert "mcp__fx___stats" in names, "only the data child's internal verbs are hidden"
        assert "mcp__data___stats" not in names and "mcp__data__rows" in names
        tool = next(t for t in bridge.tools if t.name == "mcp__fx__rows")
        assert tool.description == "Rows (derived text)."
        assert {t.name for batch in changed for t in batch} == names

        ctx = object()
        out = await tool.handler(ctx, {"limit": 1})
        assert gw.prepared[-1][3] is ctx and out["raw"].envelope == "ok"
    finally:
        await bridge.aclose()


async def test_internal_verbs_and_call_raw_bypass_the_gateway(tmp_path):
    gw = FakeGateway()
    bridge = MCPBridge([server("data", tmp_path / "d"), server("fx", tmp_path / "s")], log_dir=tmp_path / "logs",
                       options=FAST, gateway=gw)
    try:
        await bridge.start()
        out = await bridge.call("data", "_stats", {"request": json.dumps({"tables": ["a.b"]})})
        assert json.loads(out)["echo"] == {"tables": ["a.b"]}
        assert json.loads(await bridge.call_raw("fx", "rows", {"limit": 1}))["count"] == 1
        with pytest.raises(ToolFailure, match="could not be loaded"):
            await bridge.call_raw("fx", "legacy_error", {})
        assert gw.prepared == [] and gw.finished == []
    finally:
        await bridge.aclose()


async def test_no_gateway_is_byte_identical(tmp_path):
    """Without a gateway, results and failures are exactly those of the legacy conversion, and
    ``legacy_result`` reproduces them from the classify-only form (observe mode)."""
    plain = MCPBridge([server("fx", tmp_path / "a")], log_dir=tmp_path / "la", options=FAST)
    gw = FakeGateway()
    seam = MCPBridge([server("fx", tmp_path / "b")], log_dir=tmp_path / "lb", options=FAST, gateway=gw)
    cases = [("lookup", {"target_id": "ENSG00000169174"}), ("lookup", {"target_id": "ENSG00000999999"}),
             ("rows", {"limit": 4}), ("empty", {}), ("legacy_error", {}), ("raise_error", {"message": "x"}),
             ("ping", {})]
    try:
        await plain.start()
        await seam.start()
        assert plain.gateway is None
        assert [(t.name, t.description, t.input_schema) for t in plain.tools] == \
               [(t.name, t.description, t.input_schema) for t in seam.tools]
        for tool, args in cases:
            try:
                expected = ("ok", await plain.call("fx", tool, args))
            except ToolFailure as exc:
                expected = ("error", str(exc))
            assert expected == await _outcome(plain.call_raw("fx", tool, args))
            raw = (await seam.call("fx", tool, args))["raw"]
            assert expected == await _outcome(_legacy(raw, tool)), tool
        assert isinstance(expected[1], str)
    finally:
        await plain.aclose()
        await seam.aclose()


async def _legacy(raw, tool):
    return legacy_result(raw, tool)


async def _outcome(coro):
    try:
        return ("ok", await coro)
    except ToolFailure as exc:
        return ("error", str(exc))


async def test_update_in_place_after_restart_and_late_tools(tmp_path):
    state = tmp_path / "s"
    changed: list[list] = []
    events = Events()
    bridge = MCPBridge([server("fx", state)], log_dir=tmp_path / "logs", options=FAST, gateway=FakeGateway(),
                       on_tools_changed=changed.append, on_event=events)
    try:
        await bridge.start()
        lookup = next(t for t in bridge.tools if t.name == "mcp__fx__lookup")
        assert "variant base" in lookup.description
        changed.clear()
        # The server's listing changes; a crash restarts it and the new listing is registered.
        (state / "variant").write_text("v2")
        await bridge.call("fx", "crash_once", {})
        same = next(t for t in bridge.tools if t.name == "mcp__fx__lookup")
        assert same is lookup, "updated in place: the registry's Tool object stays valid"
        assert "variant v2" in lookup.description
        late = next(t for t in bridge.tools if t.name == "mcp__fx__late_tool")
        assert {t.name for t in changed[-1]} == {"mcp__fx__lookup", "mcp__fx__late_tool"}
        started = [d for k, d in events if k == "mcp_server_started"]
        assert started[-1]["new_tools"] == 1
        out = await late.handler(None, {})
        assert out["raw"].text == "late tool on start #2"

        # A failing callback never breaks registration.
        bridge.on_tools_changed = lambda tools: 1 / 0
        (state / "variant").write_text("v3")
        await bridge.recycle("fx", wait_s=5)
        assert "variant v3" in lookup.description
    finally:
        await bridge.aclose()


async def test_dead_session_is_restarted_before_the_next_call(tmp_path):
    """After a retry fails too, the session is torn down and marked broken; the next call
    restarts first instead of hitting the dead pipe (no second mcp_crash)."""
    events = Events()
    state = tmp_path / "s"
    bridge = MCPBridge([server("fx", state)], log_dir=tmp_path / "logs", options={**FAST, "max_restarts": 3},
                       on_event=events)
    try:
        await bridge.start()
        with pytest.raises(ToolFailure, match="failed again after a restart"):
            await bridge.call("fx", "crash_always", {})
        st = bridge.status()["fx"]
        assert st["state"] == "broken" and st["restarts"] == 1 and "fx" not in bridge.sessions
        crashes = events.kinds().count("mcp_crash")
        assert await bridge.call("fx", "ping", {}) == "pong from start #3"
        assert events.kinds().count("mcp_crash") == crashes, "the dead pipe was not used again"
        st = bridge.status()["fx"]
        assert st["state"] == "ready" and st["restarts"] == 2
        restart = [d for k, d in events if k == "mcp_restart"][-1]
        assert restart["broken"] is True and restart["oom"] is False
    finally:
        await bridge.aclose()


async def test_no_retry_decision_breaks_the_session(tmp_path):
    state = tmp_path / "s"
    gw = FakeGateway(retry=False)
    bridge = MCPBridge([server("fx", state)], log_dir=tmp_path / "logs", options=FAST, gateway=gw)
    try:
        await bridge.start()
        with pytest.raises(GatewayError) as exc:
            await bridge.call("fx", "crash_once", {})
        assert exc.value.kind is ErrorKind.server_crashed
        assert [c["tool"] for c in calls(state)].count("crash_once") == 1, "not retried"
        assert bridge.status()["fx"]["state"] == "broken" and gw.finished == []
        out = await bridge.call("fx", "crash_once", {})
        assert out["raw"].text == "survived after restart (start #2)"
    finally:
        await bridge.aclose()


async def test_recycle_waits_for_in_flight_calls_and_is_not_a_restart(tmp_path):
    events = Events()
    bridge = MCPBridge([server("fx", tmp_path / "s")], log_dir=tmp_path / "logs", options=FAST, on_event=events)
    try:
        await bridge.start()
        slow = asyncio.ensure_future(bridge.call("fx", "slow", {"seconds": 0.8}))
        await asyncio.sleep(0.2)
        t0 = time.monotonic()
        assert await bridge.recycle("fx", wait_s=10) is True
        assert time.monotonic() - t0 >= 0.4, "recycle waited for the in-flight call"
        assert await slow == "slept 0.8s on start #1"
        st = bridge.status()["fx"]
        assert st["restarts"] == 0 and st["recycles"] == 1 and st["generation"] == 2 and st["state"] == "ready"
        assert await bridge.call("fx", "ping", {}) == "pong from start #2"
        rec = [d for k, d in events if k == "mcp_recycle"]
        assert rec and rec[-1]["ok"] is True and rec[-1]["generation"] == 2

        # A server that stays busy is not recycled within the wait.
        busy = asyncio.ensure_future(bridge.call("fx", "slow", {"seconds": 1.0}))
        await asyncio.sleep(0.2)
        assert await bridge.recycle("fx", wait_s=0.2) is False
        assert [d for k, d in events if k == "mcp_recycle"][-1]["reason"] == "busy"
        await busy
        assert bridge.status()["fx"]["generation"] == 2
    finally:
        await bridge.aclose()
