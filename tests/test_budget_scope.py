"""Budget scopes: per-turn caps limit turns, not bulk runs; spend survives failures."""

import asyncio

import pytest

from vbt import budget
from vbt.agents import AgentDefinition
from vbt.budget import BudgetExceeded, CostScope, InvocationCost, open_scope
from vbt.orchestrator import open_session
from vbt.providers.base import ProviderError, validate_tool_pairing
from vbt.providers.mock import ScriptedProvider, call, fail, reply, turn
from vbt.runtime import BudgetExceeded as RuntimeBudgetExceeded
from vbt.tools.base import Tool, schema


# --------------------------------------------------------------------------- unit


def test_runtime_reexports_budget_exceeded():
    assert RuntimeBudgetExceeded is BudgetExceeded
    assert issubclass(BudgetExceeded, RuntimeError)


def test_scopes_nest_charge_the_chain_and_check_the_innermost():
    assert budget.current_scope() is None
    budget.charge(5.0)  # no scope: no-op, never raises
    budget.check()
    with open_scope("bulk", 10) as bulk:
        with open_scope("item:1", 2) as item:
            assert item.parent is bulk and budget.current_scope() is item
            budget.charge(1.5)
            budget.check()
            budget.charge(1.0)
            with pytest.raises(BudgetExceeded) as exc:
                budget.check()
            assert exc.value.scope is item and "item:1" in str(exc.value)
        assert bulk.spent == pytest.approx(2.5) and item.spent == pytest.approx(2.5)
        assert budget.current_scope() is bulk
        budget.check()
        with open_scope("item:2", None) as item2:
            budget.charge(8.0)
            with pytest.raises(BudgetExceeded) as exc:
                budget.check()
            assert exc.value.scope is bulk, "an unlimited item inside an exceeded bulk scope stops at bulk"
            assert item2.path == "bulk/item:2"
    assert budget.current_scope() is None


def test_zero_or_missing_limit_means_unlimited():
    for lim in (None, 0, "0", -1, "nope"):
        s = CostScope("x", lim)
        s.charge(1e9)
        assert s.limit_usd is None and not s.exceeded


def test_invocation_accumulator_and_contextvar_isolation():
    acc = InvocationCost("a")
    tok = budget.use_invocation(acc)
    try:
        assert budget.current_invocation() is acc
        acc.add(1.0, model_call=True)
        acc.add(0.25, model_call=False)
    finally:
        budget.reset_invocation(tok)
    assert budget.current_invocation() is None
    assert (acc.usd, acc.model_usd, acc.tool_usd, acc.model_calls) == (1.25, 1.0, 0.25, 1)


async def test_tasks_inherit_the_scope_of_their_creator():
    with open_scope("turn", 100) as turn_scope:
        async def child():
            budget.charge(2.0)
            return budget.current_scope()
        seen = await asyncio.gather(asyncio.ensure_future(child()), asyncio.ensure_future(child()))
    assert all(s is turn_scope for s in seen) and turn_scope.spent == 4.0


# --------------------------------------------------------------------------- runtime


def _probe(name="probe", tools=()):
    return AgentDefinition(name=name, description="test agent", prompt="You are a test agent.", tier="scientist",
                           tools=list(tools))


async def _session(config, provider, **limits):
    config["orchestration"]["enforce_review"] = False
    config.setdefault("limits", {}).update(limits)
    return await open_session(config, provider=provider, start_mcp=False)


async def test_ten_item_scopes_with_a_dollar_per_call_mock_all_succeed(config):
    """Bulk-style code with no turn scope is never stopped by max_turn_cost_usd."""
    provider = ScriptedProvider(lambda agent, system, messages, tools: (
        turn(call("TodoWrite", todos=[]), cost_usd=1.0) if len(messages) == 1 else turn("ok", cost_usd=1.0)))
    session = await _session(config, provider, max_turn_cost_usd=3)
    rt, agent = session.rt, _probe(tools=["TodoWrite"])
    results = []
    for i in range(10):
        with rt.cost_scope(f"item:{i}") as scope:
            res = await rt.run_agent(agent, f"item {i}", depth=1)
        results.append((res, scope.spent))
    assert all(r.status == "completed" for r, _ in results)
    assert [s for _, s in results] == [2.0] * 10
    assert all(r.cost_usd == 2.0 for r, _ in results)
    # and with no scope at all
    res = await rt.run_agent(agent, "unscoped", depth=1)
    assert res.status == "completed"
    assert rt.run.cost.total_usd == pytest.approx(22.0)
    await session.close()


async def test_item_scope_limit_stops_only_that_item(config):
    provider = ScriptedProvider(lambda agent, system, messages, tools: (
        turn(call("TodoWrite", todos=[]), cost_usd=1.0) if len(messages) == 1 else turn("ok", cost_usd=1.0)))
    session = await _session(config, provider)
    rt, agent = session.rt, _probe(tools=["TodoWrite"])
    history = []
    with rt.cost_scope("bulk", 100) as bulk:
        with rt.cost_scope("item:a", 0.5):
            with pytest.raises(BudgetExceeded) as exc:
                await rt.run_agent(agent, "a", depth=1, history=history)
        assert exc.value.scope.name == "item:a"
        assert validate_tool_pairing(history) == []
        with rt.cost_scope("item:b", 5):
            assert (await rt.run_agent(agent, "b", depth=1)).status == "completed"
    assert bulk.spent == pytest.approx(3.0)
    await session.close()


async def test_cost_of_a_failed_attempt_stays_in_the_scope(config):
    script = iter([turn(call("TodoWrite", todos=[]), cost_usd=1.0), fail(ProviderError("authentication failed"))])
    provider = ScriptedProvider(lambda *a: next(script))
    session = await _session(config, provider)
    rt = session.rt
    with rt.cost_scope("item:x") as scope:
        with pytest.raises(ProviderError):
            await rt.run_agent(_probe(tools=["TodoWrite"]), "go", depth=1)
    assert scope.spent == pytest.approx(1.0)
    assert rt.run.cost.usd_by_agent["probe"] == pytest.approx(1.0)
    ends = [e for e in rt.run.events() if e["type"] == "agent_end"]
    assert ends[-1]["status"] == "error" and ends[-1]["cost_usd"] == pytest.approx(1.0)
    await session.close()


async def test_tool_side_cost_reaches_agent_result_ledger_and_scope(config):
    async def search(ctx, a):
        ctx.add_cost(0.25, "web_search")
        return {"results": []}

    script = iter([turn(call("paid_search", q="x"), cost_usd=1.0), turn("done", cost_usd=1.0)])
    provider = ScriptedProvider(lambda *a: next(script))
    session = await _session(config, provider)
    rt = session.rt
    tool = Tool("paid_search", "search", schema({"q": {"type": "string"}}), search)
    with rt.cost_scope("item") as scope:
        res = await rt.run_agent(_probe(), "go", depth=1, extra_tools=[tool])
    assert res.cost_usd == pytest.approx(2.25)
    assert scope.spent == pytest.approx(2.25)
    report = rt.run.cost.report()
    assert report["agents"]["probe"]["usd"] == pytest.approx(2.25)
    assert report["agents"]["probe"]["tool_usd"] == pytest.approx(0.25)
    assert report["agents"]["probe"]["model_calls"] == 2
    assert report["extra_by_label"]["web_search"] == pytest.approx(0.25)
    await session.close()


async def test_blocking_tool_cost_from_a_worker_thread(config):
    def handler(ctx, a):  # runs in a thread (Tool.blocking)
        ctx.add_cost(0.5, "extract")
        return "ok"

    script = iter([reply(call("slow_sync")), reply("done")])
    session = await _session(config, ScriptedProvider(lambda *a: next(script)))
    tool = Tool("slow_sync", "x", schema({}), handler, blocking=True)
    with session.rt.cost_scope("s") as scope:
        res = await session.rt.run_agent(_probe(), "go", depth=1, extra_tools=[tool])
    assert res.cost_usd == pytest.approx(0.5) and scope.spent == pytest.approx(0.5)
    await session.close()


async def test_begin_turn_compat_shim_resets_its_scope(config):
    provider = ScriptedProvider(lambda *a: turn("x", cost_usd=2.0))
    session = await _session(config, provider, max_turn_cost_usd=3)
    rt = session.rt
    s1 = rt.begin_turn()
    await rt.run_agent(_probe(), "a", depth=1)
    await rt.run_agent(_probe(), "b", depth=1)  # 4 > 3 after this call
    with pytest.raises(BudgetExceeded):
        await rt.run_agent(_probe(), "c", depth=1)
    s2 = rt.begin_turn()
    assert s2 is s1 and s2.spent == 0.0 and not s2.grace_used
    assert (await rt.run_agent(_probe(), "d", depth=1)).status == "completed"
    await session.close()
