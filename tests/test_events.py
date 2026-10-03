"""The runtime event stream: schema, async consumers, parallel delegations."""

import asyncio
import json
import threading

from vbt.events import EVENT_KINDS, EventBus, preview, to_json_line
from vbt.orchestrator import open_session
from vbt.providers.mock import ScriptedProvider, call, reply, turn


# --------------------------------------------------------------------------- EventBus


async def test_sync_and_async_subscribers_and_drain():
    bus = EventBus()
    got_sync, got_async = [], []

    def sync(kind, data):
        got_sync.append((kind, data))

    async def slow(kind, data):
        await asyncio.sleep(0.01)
        got_async.append(kind)

    def broken(kind, data):
        raise RuntimeError("consumer bug")

    bus.subscribe(sync)
    bus.subscribe(broken)
    unsubscribe = bus.subscribe(slow)
    bus.publish("tool_start", {"tool": "Read"})
    bus.publish("tool_end", {"tool": "Read", "is_error": False})
    assert [k for k, _ in got_sync] == ["tool_start", "tool_end"]
    assert all("ts" in d for _, d in got_sync)
    assert got_async == [] and bus.pending == 2
    await bus.drain()
    assert got_async == ["tool_start", "tool_end"] and bus.pending == 0
    unsubscribe()
    bus.publish("cost", {"total_usd": 1})
    await bus.drain()
    assert got_async == ["tool_start", "tool_end"]


async def test_publish_from_a_worker_thread_is_delivered_on_the_loop():
    bus = EventBus()
    seen = []

    def sub(kind, data):
        seen.append((kind, threading.get_ident()))

    bus.subscribe(sub)
    bus.publish("warning", {"message": "prime"})  # remembers the loop
    loop_thread = threading.get_ident()
    await asyncio.to_thread(bus.publish, "cost", {"total_usd": 2.0})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert ("cost", loop_thread) in seen


def test_event_helpers():
    p = preview({"content": "x" * 5000, "items": list(range(50))}, 100)
    assert len(p["content"]) < 200 and "5,000 chars" in p["content"]
    assert len(p["items"]) == 21
    line = json.loads(to_json_line("text", {"agent": "cso", "text": "hi"}))
    assert line == {"kind": "text", "agent": "cso", "text": "hi"}
    for kind in ("agent_start", "agent_end", "tool_start", "tool_end", "text", "message_end", "thinking",
                 "delegation", "delegation_end", "compaction", "retry", "cost", "warning"):
        assert kind in EVENT_KINDS


# --------------------------------------------------------------------------- runtime stream


async def test_parallel_delegations_to_the_same_agent_are_distinguishable(config):
    config["orchestration"]["enforce_review"] = False
    t1 = call("Task", subagent_type="genomics-analyst", description="EGFR", prompt="EGFR genetics")
    t2 = call("Task", subagent_type="genomics-analyst", description="KRAS", prompt="KRAS genetics")

    def genomics(messages):
        target = "EGFR" if "EGFR" in messages[0].text else "KRAS"
        return reply(f"{target}: L2G 0.8")

    provider = ScriptedProvider.from_rules({
        "cso": [turn("Dispatching.", t1, t2, thinking="plan: two lookups"),
                reply("Both done.")],
        "genomics-analyst": [lambda m: reply(call("TodoWrite", todos=[])), lambda m: reply(call("TodoWrite", todos=[])),
                             genomics, genomics],
    })
    events, async_events = [], []

    def on_event(kind, data):
        events.append((kind, data))

    async def consumer(kind, data):
        await asyncio.sleep(0)
        async_events.append(kind)

    session = await open_session(config, provider=provider, start_mcp=False, on_event=on_event)
    rt = session.rt
    rt.bus.subscribe(consumer)
    res = await rt.run_agent(rt.cso, "Compare EGFR and KRAS", history=session.history, stream_text=True)
    await rt.bus.drain()
    assert res.status == "completed" and res.text == "Both done."

    starts = [d for k, d in events if k == "agent_start" and d["depth"] == 1]
    assert {d["invocation_id"] for d in starts} == {t1.id, t2.id}
    assert all(d["parent_invocation_id"] == res.invocation_id for d in starts)
    assert all(d["tool_use_id"] in (t1.id, t2.id) for d in starts)
    sub_tools = [d for k, d in events if k == "tool_start" and d["agent"] == "genomics-analyst"]
    assert {d["invocation_id"] for d in sub_tools} == {t1.id, t2.id}
    tool_ends = [d for k, d in events if k == "tool_end"]
    assert {d["tool_use_id"] for d in tool_ends} >= {t1.id, t2.id}
    assert all("is_error" in d and "duration_s" in d for d in tool_ends)
    deleg_end = [d for k, d in events if k == "delegation_end"]
    assert {d["invocation_id"] for d in deleg_end} == {t1.id, t2.id}
    assert all(d["status"] == "completed" for d in deleg_end)
    kinds = [k for k, _ in events]
    for k in ("agent_start", "agent_end", "tool_start", "tool_end", "text", "message_end", "thinking",
              "delegation", "delegation_end", "cost"):
        assert k in kinds, k
    assert "tool" not in kinds, "the legacy 'tool' alias is retired (consumers read tool_start)"
    assert all("ts" in d for _, d in events)
    thinking = [d for k, d in events if k == "thinking"]
    assert thinking[0]["text"] == "plan: two lookups" and thinking[0]["streamed"] is True
    assert sorted(async_events) == sorted(kinds), "the async subscriber saw every event"

    # the trace carries agent_run_id/parent_run_id on events inside each invocation
    trace = rt.run.events()
    sub_trace = [e for e in trace if e["type"] in ("tool_start", "tool_end", "model_call")
                 and e.get("agent") == "genomics-analyst"]
    assert sub_trace and {e["agent_run_id"] for e in sub_trace} == {t1.id, t2.id}
    assert all(e["parent_run_id"] == res.invocation_id for e in sub_trace)
    await session.close()


async def test_non_streamed_thinking_is_emitted_after_the_message(config):
    config["orchestration"]["enforce_review"] = False
    provider = ScriptedProvider.from_rules({"genomics-analyst": [turn("answer", thinking="hidden reasoning")]})
    events = []
    session = await open_session(config, provider=provider, start_mcp=False,
                                 on_event=lambda k, d: events.append((k, d)))
    rt = session.rt
    await rt.run_agent(rt.agents["genomics-analyst"], "q", depth=1)
    th = [d for k, d in events if k == "thinking"]
    assert th and th[0]["streamed"] is False and th[0]["text"] == "hidden reasoning"
    assert not [d for k, d in events if k == "text"], "specialist text is not streamed"
    await session.close()
