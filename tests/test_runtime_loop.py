"""Agent-loop core: reply text, history invariants, stop statuses, delegation,
trace fidelity, tool output handling, context and retry wiring (strict mock)."""

import asyncio
import hashlib
import json
import re
from pathlib import Path

import pytest

from vbt import failures as fl
from vbt.agents import AgentDefinition
from vbt.budget import BudgetExceeded
from vbt.context import CompactionResult
from vbt.providers.base import (
    ContextOverflowError,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    validate_tool_pairing,
)
from vbt.providers.mock import ScriptedProvider, call, fail, reply, turn
from vbt.providers.retry import RetryPolicy
from vbt.runtime import BUDGET_GRACE_MSG, repair_history
from vbt.tools.base import Tool, ToolContext, ToolFailure, schema
from conftest import open_scripted_session


# --------------------------------------------------------------------------- helpers


def _probe(name="probe", tools=(), **kw):
    return AgentDefinition(name=name, description="test agent", prompt="You are a test agent.", tier="scientist",
                           tools=list(tools), **kw)


async def _session(config, provider, on_event=None, **limits):
    return await open_scripted_session(config, provider, on_event=on_event, **limits)


def _results(msg):
    return [b for b in msg.content if isinstance(b, ToolResult)]


def _trace(rt, kind):
    return [e for e in rt.run.events() if e["type"] == kind]


class _Slow:
    """A tool that waits until cancelled and records it."""

    def __init__(self, name="slow_wait"):
        self.started = asyncio.Event()
        self.cancelled = False
        self.tool = Tool(name, "wait for a long time", schema({}), self.handler)

    async def handler(self, ctx, a):
        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "finished"


# --------------------------------------------------------------------------- (A) text


async def test_full_text_keeps_synthesis_emitted_with_record_claims(config):
    synthesis = "Synthesis: EGFR is a strong target [[claim:C1]]."
    claims = [{"id": "C1", "text": "EGFR is supported", "evidence": [{"kind": "citation", "pmid": "31234567"}]}]
    provider = ScriptedProvider.from_rules({"cso": [
        reply(synthesis, call("mcp__provenance__record_claims", claims=claims)), reply("Done.")]})
    session = await _session(config, provider)
    res = await session.rt.run_agent(session.rt.cso, "Is EGFR a target?", history=session.history)
    assert res.status == "completed"
    assert res.text == "Done."
    assert res.full_text == synthesis + "\n\nDone."
    assert res.drafts == []
    await session.close()


async def test_max_tokens_chunks_are_concatenated(config):
    provider = ScriptedProvider.from_rules({"probe": [
        reply("Looking up.", call("TodoWrite", todos=[])),
        turn("Part A, ", stop="max_tokens"), turn("part B, ", stop="max_tokens"), reply("end.")]})
    session = await _session(config, provider)
    history = []
    res = await session.rt.run_agent(_probe(tools=["TodoWrite"]), "write a long report", history=history, depth=1)
    assert res.text == "Part A, part B, end."
    assert res.full_text == "Looking up.\n\nPart A, part B, end."
    assert sum("Continue exactly where you stopped" in m.text for m in history if m.role == "user") == 2
    await session.close()


async def test_review_nudge_resets_full_text_and_keeps_drafts(config):
    provider = ScriptedProvider.from_rules({"cso": [reply("Draft synthesis."), reply("Final synthesis.")]})
    session = await _session(config, provider)
    nudges = []

    async def after(messages):
        if nudges:
            return None
        nudges.append(1)
        return "[Harness - review policy] dispatch the reviewer first."

    res = await session.rt.run_agent(session.rt.cso, "q", history=session.history, after_end_turn=after)
    assert res.drafts == ["Draft synthesis."]
    assert res.full_text == "Final synthesis." and res.text == "Final synthesis."
    await session.close()


# --------------------------------------------------------------------------- (B) history invariants


def test_repair_history_answers_every_unanswered_call():
    a, b, c = ToolCall("a", "Bash", {}), ToolCall("b", "Read", {}), ToolCall("c", "Task", {})
    msgs = [Message.user("q"), Message("assistant", [TextBlock("x"), a, b]),
            Message("user", [TextBlock("follow-up"), ToolResult("b", "ok")]),
            Message("assistant", [c]), Message("assistant", [TextBlock("again")]),
            Message("assistant", [ToolCall("d", "Glob", {})]), Message("user", [])]
    n = repair_history(msgs, "turn interrupted")
    assert n >= 4
    assert validate_tool_pairing(msgs) == []
    first = _results(msgs[2])
    assert [r.tool_call_id for r in first] == ["b", "a"] and first[1].is_error
    assert first[1].content == "Not executed: turn interrupted"
    assert msgs[2].content[-1].text == "follow-up"
    assert msgs[4].role == "user" and _results(msgs[4])[0].tool_call_id == "c"
    assert msgs[-1].role == "user" and _results(msgs[-1])[0].tool_call_id == "d"


def test_repair_history_drops_orphans_and_leaves_valid_history_alone():
    good = [Message.user("q"), Message("assistant", [ToolCall("a", "Bash", {})]),
            Message("user", [ToolResult("a", "ok")])]
    snapshot = list(good)
    assert repair_history(good, "x") == 0 and all(x is y for x, y in zip(good, snapshot))
    bad = [Message.user("q"), Message("assistant", [TextBlock("hi")]),
           Message("user", [ToolResult("zz", "stray"), TextBlock("next")])]
    assert repair_history(bad, "x") >= 1
    assert validate_tool_pairing(bad) == []
    assert "orphan tool result zz" in bad[2].text


async def test_cancel_during_delegation_leaves_paired_history(config):
    provider = ScriptedProvider.from_rules({
        "cso": [reply("Delegating.", call("Task", subagent_type="genomics-analyst", description="genetics",
                                          prompt="EGFR genetics")),
                reply("Recovered after the interruption.")],
        "genomics-analyst": [reply(call("Bash", command="sleep 1"))],
    })
    started = asyncio.Event()

    def on_event(kind, data):
        if kind == "tool_start" and data.get("tool") == "Bash":
            started.set()

    session = await _session(config, provider, on_event=on_event)
    rt, history = session.rt, session.history
    task = asyncio.ensure_future(rt.run_agent(rt.cso, "Is EGFR a target?", history=history, stream_text=True))
    await asyncio.wait_for(started.wait(), 10)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert validate_tool_pairing(history) == []
    res = _results(history[-1])
    assert len(res) == 1 and res[0].is_error and res[0].content.startswith("Not executed")
    entry = rt.delegation_log[-1]
    assert entry["agent"] == "genomics-analyst" and entry["status"] == "cancelled" and entry["end_ts"]
    ends = {e["agent"]: e for e in _trace(rt, "agent_end")}
    assert ends["cso"]["status"] == "cancelled" and ends["genomics-analyst"]["status"] == "cancelled"
    assert [e["status"] for e in _trace(rt, "delegation_end")] == ["cancelled"]
    sub_transcript = rt.run.dir / ends["genomics-analyst"]["transcript_path"]
    sub_msgs = [json.loads(line) for line in sub_transcript.read_text().splitlines()]
    assert sub_msgs[-1]["role"] == "user" and sub_msgs[-1]["content"][0]["is_error"]
    bash_end = [e for e in _trace(rt, "tool_end") if e["tool"] == "Bash"][0]
    assert bash_end["is_error"] and bash_end["interrupted"]

    # the strict mock would reject a dangling tool_use: the next turn works
    nxt = await rt.run_agent(rt.cso, "Please continue.", history=history)
    assert nxt.status == "completed" and nxt.text == "Recovered after the interruption."
    await session.close()
    await asyncio.sleep(1.2)  # let the orphaned `sleep` exit while the loop is alive (no child-watcher noise)


async def test_budget_in_delegation_becomes_task_error_and_cso_gets_one_grace_call(config):
    grace_seen = []

    def cso_grace(messages):
        grace_seen.append(messages[-1].text)
        return reply("Synthesis from the partial evidence; genetics is missing.")

    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="genetics", prompt="x")), cso_grace],
        "genomics-analyst": [turn("Found L2G 0.8 so far.", call("TodoWrite", todos=[]), cost_usd=2.0),
                             turn(call("TodoWrite", todos=[]), cost_usd=2.0)],
    })
    session = await _session(config, provider)
    rt, history = session.rt, session.history
    with rt.cost_scope("turn", 3) as scope:
        res = await rt.run_agent(rt.cso, "q", history=history)
        assert res.status == "budget" and res.stop_reason == "budget"
        assert res.text == "Synthesis from the partial evidence; genetics is missing."
        assert grace_seen == [BUDGET_GRACE_MSG] and scope.grace_used
        task_result = _results(history[2])[0]
        assert task_result.is_error
        assert task_result.content.startswith("Error: [incomplete: budget] budget exhausted")
        assert "Found L2G 0.8 so far." in task_result.content
        entry = rt.delegation_log[-1]
        assert entry["status"] == "budget" and entry["cost_usd"] == pytest.approx(4.0)
        assert [c["agent"] for c in provider.calls].count("cso") == 2
        assert validate_tool_pairing(history) == []
        # the grace is used: a further call in the same exhausted scope stops
        with pytest.raises(BudgetExceeded):
            await rt.run_agent(rt.cso, "more", history=history)
        assert validate_tool_pairing(history) == []
    await session.close()


async def test_sibling_delegation_is_cancelled_when_one_raises(config):
    slow = _Slow()

    async def explode(ctx, a):
        await slow.started.wait()
        raise BudgetExceeded("tool-side budget hit")

    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", prompt="x"), call("explode"))],
        "genomics-analyst": [reply(call("slow_wait"))],
    })
    session = await _session(config, provider)
    rt, history = session.rt, session.history
    rt.registry.add(slow.tool)
    rt.agents["genomics-analyst"].tools.append("slow_wait")
    with pytest.raises(BudgetExceeded):
        await rt.run_agent(rt.cso, "q", history=history,
                           extra_tools=[Tool("explode", "x", schema({}), explode)])
    assert slow.cancelled
    assert validate_tool_pairing(history) == []
    assert all(r.is_error and "Not executed" in r.content for r in _results(history[-1]))
    assert rt.delegation_log[-1]["status"] == "cancelled"
    assert not [t for t in rt._outstanding if not t.done()]
    await session.close()


@pytest.mark.parametrize("stop", ["refusal", "other", "context_exceeded"])
async def test_abnormal_stops_with_tool_calls_answer_the_calls(config, stop):
    provider = ScriptedProvider.from_rules({"cso": [turn("Partial.", call("TodoWrite", todos=[]), stop=stop)]})
    session = await _session(config, provider)
    session.rt.context.recover_overflow = _no_recovery
    res = await session.rt.run_agent(session.rt.cso, "q", history=session.history)
    assert res.status == {"refusal": "refusal", "other": "error", "context_exceeded": "context_exceeded"}[stop]
    assert validate_tool_pairing(session.history) == []
    assert res.text.startswith("Partial.")
    await session.close()


async def _no_recovery(*a, **k):
    raise ContextOverflowError("nothing to compact")


async def test_pause_on_the_final_call_ends_the_loop(config):
    provider = ScriptedProvider(lambda *a: turn("thinking...", stop="pause_turn"))
    session = await _session(config, provider)
    res = await session.rt.run_agent(_probe(max_turns=1), "q", depth=1)
    assert res.status == "turn_limit" and len(provider.calls) == 2
    await session.close()


async def test_broken_history_is_repaired_before_the_provider_call(config):
    provider = ScriptedProvider.from_rules({"cso": [reply("fine")]})
    session = await _session(config, provider)
    history = [Message.user("q1"), Message("assistant", [ToolCall("dangling", "Task", {"prompt": "x"})])]
    res = await session.rt.run_agent(session.rt.cso, "q2", history=history)
    assert res.text == "fine" and validate_tool_pairing(history) == []
    assert _trace(session.rt, "history_repaired")
    await session.close()


# --------------------------------------------------------------------------- (C) delegation statuses


async def test_refusal_and_turn_limit_are_task_errors_with_partial_reports(config):
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", prompt="g"),
                      call("Task", subagent_type="single-cell-analyst", prompt="s")),
                reply("Synthesis with gaps.")],
        "genomics-analyst": [turn("Partial: EGFR L2G 0.8.", call("TodoWrite", todos=[]), stop="refusal")],
        "single-cell-analyst": [reply("Notes so far.", call("TodoWrite", todos=[])),
                                reply("Partial cell-type report.")],
    })
    session = await _session(config, provider)
    rt, history = session.rt, session.history
    rt.agents["single-cell-analyst"].max_turns = 1
    res = await rt.run_agent(rt.cso, "q", history=history)
    assert res.status == "completed"
    by_id = {r.tool_call_id: r for r in _results(history[2])}
    calls = history[1].tool_calls
    refusal, limit = by_id[calls[0].id], by_id[calls[1].id]
    assert refusal.is_error and "[incomplete: refusal]" in refusal.content
    assert "Partial: EGFR L2G 0.8." in refusal.content and "declined" in refusal.content
    assert limit.is_error and "[incomplete: turn_limit]" in limit.content
    assert "Partial cell-type report." in limit.content and "status turn_limit" in limit.content
    statuses = {d["agent"]: d["status"] for d in rt.delegation_log}
    assert statuses == {"genomics-analyst": "refusal", "single-cell-analyst": "turn_limit"}
    assert {e["status"] for e in _trace(rt, "delegation_end")} == {"refusal", "turn_limit"}
    await session.close()


async def test_delegation_timeout_is_a_task_error_with_the_partial_report(config):
    slow = _Slow()
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", prompt="g")), reply("ok")],
        "genomics-analyst": [reply("Started the analysis.", call("slow_wait"))],
    })
    session = await _session(config, provider, delegation_timeout_s=0.3)
    rt, history = session.rt, session.history
    assert rt.delegation_timeout_s == pytest.approx(0.3)
    rt.registry.add(slow.tool)
    rt.agents["genomics-analyst"].tools.append("slow_wait")
    res = await rt.run_agent(rt.cso, "q", history=history)
    assert res.status == "completed" and slow.cancelled
    r = _results(history[2])[0]
    assert r.is_error and "[incomplete: timeout]" in r.content and "Started the analysis." in r.content
    entry = rt.delegation_log[-1]
    assert entry["status"] == "timeout"
    sub_end = [e for e in _trace(rt, "agent_end") if e["agent"] == "genomics-analyst"][0]
    assert sub_end["status"] == "timeout"
    sub_msgs = [json.loads(x) for x in (rt.run.dir / sub_end["transcript_path"]).read_text().splitlines()]
    assert validate_tool_pairing([Message(m["role"], [_block(b) for b in m["content"]]) for m in sub_msgs]) == []
    await session.close()


def _block(b):
    if b["type"] == "tool_call":
        return ToolCall(b["id"], b["name"], b["input"])
    if b["type"] == "tool_result":
        return ToolResult(b["tool_call_id"], b["content"], b["is_error"])
    return TextBlock(b.get("text", ""))


async def test_task_footer_separates_unresolved_data_failures_from_recovered_errors(config):
    async def lookup(ctx, a):
        raise ToolFailure("Open Targets data could not be loaded")

    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", prompt="g")), reply("ok")],
        "genomics-analyst": [reply(call("Bash", command="exit 3")), reply(call("Bash", command="echo fixed")),
                             reply(call("mcp__fake__lookup", gene="EGFR")), reply("Report: partial.")],
    })
    session = await _session(config, provider)
    rt, history = session.rt, session.history
    rt.registry.add(Tool("mcp__fake__lookup", "lookup", schema({"gene": {"type": "string"}}), lookup, source="mcp:fake"))
    rt.agents["genomics-analyst"].tools.append("mcp__fake__*")
    await rt.run_agent(rt.cso, "q", history=history)
    r = _results(history[2])[0]
    assert not r.is_error and r.content.startswith("Report: partial.")
    assert "[Unresolved data-source failures: mcp__fake__lookup: Error: Open Targets data could not be loaded" \
        in r.content
    assert "[1 code/tool errors were encountered and recovered]" in r.content
    assert "Bash" not in r.content.split("Unresolved data-source failures:")[1]
    entry = rt.delegation_log[-1]
    assert [e["tool"] for e in entry["tool_errors"]] == ["Bash", "mcp__fake__lookup"]
    assert entry["tool_errors"][0]["input"] == {"command": "exit 3"}
    assert [f["tool"] for f in entry["unresolved_data_failures"]] == ["mcp__fake__lookup"]
    assert entry["recovered_errors"] == 1
    for k in ("start_ts", "end_ts", "status", "stop_reason", "cost_usd", "duration_s", "invocation_id",
              "tool_use_id", "description"):
        assert k in entry
    await session.close()


async def test_delegate_api_logs_even_on_errors(config):
    provider = ScriptedProvider.from_rules({"chief-of-staff": [fail(RuntimeError("boom"))]})
    session = await _session(config, provider)
    rt = session.rt
    with pytest.raises(RuntimeError) as exc:
        await rt.delegate("chief-of-staff", "brief please", description="strategic briefing")
    assert exc.value.agent_result.status == "error"
    entry = rt.delegation_log[-1]
    assert entry["agent"] == "chief-of-staff" and entry["status"] == "error" and entry["tool_use_id"] is None
    assert entry["invocation_id"].startswith("chief-of-staff:")
    assert [e["status"] for e in _trace(rt, "delegation_end")] == ["error"]
    with pytest.raises(ToolFailure):
        await rt.delegate("no-such-agent", "x")
    await session.close()


# --------------------------------------------------------------------------- (D) caps


async def test_specialist_and_cso_caps_are_separate(config):
    provider = ScriptedProvider(lambda agent, system, messages, tools: (
        reply(call("TodoWrite", todos=[])) if sum(m.role == "assistant" for m in messages) < 4 else reply("done")))
    session = await _session(config, provider, max_cso_turns=2, max_specialist_turns=5)
    rt = session.rt
    assert rt.max_turns_for(rt.cso, 0) == 2
    assert rt.max_turns_for(_probe(), 1) == 5
    assert rt.max_turns_for(_probe(max_turns=7), 1) == 7
    assert rt.max_turns_for(rt.agents["single-cell-analyst"], 1) == rt.agents["single-cell-analyst"].max_turns
    spec = await rt.run_agent(_probe(tools=["TodoWrite"]), "q", depth=1)
    assert spec.status == "completed" and spec.model_calls == 5
    top = await rt.run_agent(_probe(tools=["TodoWrite"]), "q", depth=0)
    assert top.status == "turn_limit" and top.model_calls == 3
    await session.close()


# --------------------------------------------------------------------------- (F) terminal tools


async def test_terminal_tool_ends_the_loop_without_another_model_call(config):
    def submit(ctx, a):
        if not a.get("label"):
            raise ToolFailure("label is required")
        return "Result recorded."

    script = iter([reply("Submitting.", call("submit_result")), reply(call("submit_result", label="POSITIVE"))])
    provider = ScriptedProvider(lambda *a: next(script))
    session = await _session(config, provider)
    history = []
    tool = Tool("submit_result", "submit", schema({"label": {"type": "string"}}), submit, terminal=True)
    res = await session.rt.run_agent(_probe(), "annotate", history=history, depth=1, extra_tools=[tool])
    assert len(provider.calls) == 2
    assert res.status == "completed" and res.stop_reason == "terminal_tool"
    assert validate_tool_pairing(history) == [] and history[-1].role == "user"
    assert _results(history[-1])[0].content == "Result recorded."
    await session.close()


# --------------------------------------------------------------------------- (H) trace fidelity


async def test_long_write_is_spilled_with_sha_and_transcript_code_versions_exist(config):
    code = "import pandas as pd\n" + "\n".join(f"x_{i} = {i}  # step {i}" for i in range(2000)) + "\n"
    heredoc = "python - <<'EOF'\nprint('inline analysis')\nEOF"
    write = call("Write", file_path="code/scripts/analysis.py", content=code)
    bash = call("Bash", command=heredoc)
    provider = ScriptedProvider.from_rules({"genomics-analyst": [reply(write), reply(bash), reply("done")]})
    session = await _session(config, provider)
    rt = session.rt
    seen = []
    orig = rt.run.after_tool
    rt.run.after_tool = lambda ev: (seen.append(ev), orig(ev))
    res = await rt.run_agent(rt.agents["genomics-analyst"], "write the script", depth=1)
    assert res.status == "completed"

    start = next(e for e in _trace(rt, "tool_start") if e["tool_use_id"] == write.id)
    assert "input" not in start and start["input_chars"] > 16000
    assert start["input_sha256"] == fl.input_sha256(write.input)
    spilled = json.loads((rt.run.dir / start["input_ref"]).read_text())
    assert spilled["input"] == write.input and spilled["input_sha256"] == start["input_sha256"]
    assert start["agent_run_id"] == res.invocation_id

    end = next(e for e in _trace(rt, "tool_end") if e["tool_use_id"] == write.id)
    sha = hashlib.sha256(code.encode()).hexdigest()
    assert end["code_sha256"] == sha
    assert (rt.run.dir / end["code_version"]).read_text() == code
    assert end["code_version"] == f"logs/code_versions/{sha}.py"

    bstart = next(e for e in _trace(rt, "tool_start") if e["tool_use_id"] == bash.id)
    assert heredoc in (rt.run.dir / bstart["bash_script"]).read_text()

    agent_end = [e for e in _trace(rt, "agent_end") if e["agent"] == "genomics-analyst"][-1]
    lines = (rt.run.dir / agent_end["transcript_path"]).read_text().splitlines()
    assert len(lines) == len(res.messages) and json.loads(lines[1])["content"][0]["input"] == write.input
    assert agent_end["transcript_path"] == res.transcript_path
    mc = _trace(rt, "model_call")[-1]
    assert {"served_model", "retries", "fallback_used"} <= set(mc)

    # (M) after_tool receives the full input
    hooked = next(ev for ev in seen if ev["tool_use_id"] == write.id)
    assert hooked["input"] == write.input and hooked["tool"] == "Write"
    await session.close()


async def test_audit_failure_is_reported_in_the_tool_result(config):
    provider = ScriptedProvider.from_rules({"genomics-analyst": [
        reply(call("Write", file_path="results/tables/t.csv", content="a,b\n")), reply("done")]})
    session = await _session(config, provider)
    rt = session.rt
    rt.run.after_tool = lambda ev: rt.run.audit_errors.append("after_tool: disk full")
    history = []
    await rt.run_agent(rt.agents["genomics-analyst"], "go", depth=1, history=history)
    content = _results(history[2])[0].content
    assert "[Harness] Audit recording failed for this call: after_tool: disk full" in content
    assert "do not invent citations" in content
    await session.close()


# --------------------------------------------------------------------------- (I) tool output


def _big_trial():
    return {"protocolSection": {"briefSummary": "s" * 60_000, "phase": "PHASE2"},
            "participantFlow": [{"group": f"g{i}"} for i in range(200)],
            "outcomeMeasures": [{"title": f"o{i}", "value": i} for i in range(150)],
            "adverseEvents": {"serious": [{"term": f"t{i}", "n": i} for i in range(300)]},
            "hasResults": True}


async def test_json_truncation_keeps_top_level_keys_and_query_reads_a_nested_slice(config):
    async def ct_lookup(ctx, a):
        return json.dumps(_big_trial())

    checks = {}

    def after_lookup(messages):
        text = _results(messages[-1])[0].content
        checks["truncated"] = text
        path = re.search(r"Full output saved to (\S+);", text).group(1)
        return reply(call("QueryToolOutput", path=path, json_path="$.adverseEvents.serious[20:40]"))

    def after_query(messages):
        checks["query"] = _results(messages[-1])[0].content
        return reply("done")

    provider = ScriptedProvider.from_rules({"probe": [reply(call("mcp__ctgov__get_trial", nct_id="NCT1")),
                                                      after_lookup, after_query]})
    session = await _session(config, provider, tool_output_max_chars=6000)
    rt = session.rt
    tool = Tool("mcp__ctgov__get_trial", "trial", schema({"nct_id": {"type": "string"}}), ct_lookup)
    res = await rt.run_agent(_probe(), "annotate NCT1", depth=1, extra_tools=[tool])
    assert res.status == "completed"
    text = checks["truncated"]
    assert len(text) < 7000
    for key in _big_trial():
        assert f'"{key}"' in text
    assert "QueryToolOutput path=" in text and "json_path=$.adverseEvents.serious[" in text
    assert "Read" not in text.split("[Output truncated")[1], "the probe agent has no Read tool"
    spill = Path(re.search(r"Full output saved to (\S+);", text).group(1))
    assert spill.parent == rt.run.dir / "logs" / "tool_outputs" and spill.suffix == ".json"
    assert json.loads(spill.read_text()) == _big_trial() and "\n" in spill.read_text()
    q = json.loads(checks["query"])
    assert [x["term"] for x in q] == [f"t{i}" for i in range(20, 40)]
    end = next(e for e in _trace(rt, "tool_end") if e["tool"] == "mcp__ctgov__get_trial")
    assert end["output_path"] == f"logs/tool_outputs/{spill.name}" and end["output_chars"] > 60_000
    assert len(end["output"]) == rt.trace_output_chars
    assert not (rt.run.dir / "work" / "_tool_outputs").exists()
    await session.close()


async def test_text_truncation_note_mentions_read_only_when_available(config):
    async def noisy(ctx, a):
        return "line\n" * 5000 + "Traceback: the real error"

    script = iter([reply(call("noisy")), reply(call("noisy")), reply("done")])
    provider = ScriptedProvider(lambda *a: next(script))
    session = await _session(config, provider, tool_output_max_chars=3000)
    tool = Tool("noisy", "x", schema({}), noisy)
    history = []
    await session.rt.run_agent(_probe(tools=["Read"]), "go", depth=1, extra_tools=[tool], history=history)
    text = _results(history[2])[0].content
    assert "Traceback: the real error" in text, "the tail of long output is kept"
    assert "or Read with offset/limit" in text and "QueryToolOutput(path, offset, limit)" in text
    path = re.search(r"Full output saved to (\S+);", text).group(1)
    assert Path(path).suffix == ".txt"
    await session.close()


async def test_query_tool_output_is_restricted_and_auto_added(config, tmp_path):
    session = await _session(config, ScriptedProvider(lambda *a: reply("x")))
    rt = session.rt
    assert all(any(t.name == "QueryToolOutput" for t in rt.tools_for(a)) for a in [rt.cso, *rt.agents.values()])
    q = rt.registry.get("QueryToolOutput")
    ctx = ToolContext(agent="genomics-analyst", run=rt.run, runtime=rt)
    with pytest.raises(ToolFailure):
        await q(ctx, {"path": str(rt.run.dir / "MANIFEST.json")})
    out = rt.run.dir / "logs" / "tool_outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t1.json").write_text(json.dumps({"a": {"b": [1, 2, 3]}, "odd key": "v"}, indent=1))
    assert await q(ctx, {"path": "t1.json", "json_path": "$.a.keys()"}) == ["b"]
    assert await q(ctx, {"path": "logs/tool_outputs/t1.json", "json_path": '$["odd key"]'}) == "v"
    assert json.loads(await q(ctx, {"path": "t1.json", "json_path": "a.b", "offset": 1, "limit": 1})) == [2]
    with pytest.raises(ToolFailure, match="not found"):
        await q(ctx, {"path": "t1.json", "json_path": "$.zzz"})
    paged = await q(ctx, {"path": "t1.json", "offset": 2, "limit": 2})
    assert paged.splitlines()[0].strip().startswith("2\t") and "more lines" in paged
    await session.close()


# --------------------------------------------------------------------------- (L) workspace


async def test_cso_workspace_is_the_run_dir_and_no_work_cso_exists(config, tmp_path):
    real = tmp_path / "real_data"
    (real / "target").mkdir(parents=True)
    (tmp_path / "data_link").symlink_to(real)
    config["paths"]["read_roots"] = [str(tmp_path / "data_link")]
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="scientific-reviewer", prompt="review")), reply("ok")],
        "scientific-reviewer": [reply(call("Read", file_path="work/genomics-analyst/results/tables/x.csv")),
                                reply("APPROVED")],
    })
    session = await _session(config, provider)
    rt, run = session.rt, session.run
    assert rt.read_roots == [real.resolve()]
    table = run.agent_dir("genomics-analyst") / "results" / "tables" / "x.csv"
    table.write_text("gene,score\nEGFR,0.8\n")
    await session.ask("q")
    assert rt.workspace_for("cso") == run.dir
    assert rt.workspace_for("scientific-reviewer") == run.dir
    assert rt.workspace_for("genomics-analyst") == run.dir / "work" / "genomics-analyst"
    assert ToolContext(agent="cso", run=run, runtime=rt).workspace == run.dir
    read_end = next(e for e in _trace(rt, "tool_end") if e["tool"] == "Read")
    assert not read_end["is_error"] and "EGFR" in read_end["output"]
    assert not (run.dir / "work" / "cso").exists()
    await session.close()
    assert not (run.dir / "work" / "cso").exists()


# --------------------------------------------------------------------------- (J) context and retry


class _FakeContext:
    def __init__(self):
        self.recoveries = 0

    def settings_for(self, settings):
        return settings

    async def maybe_compact(self, messages, **kw):
        return None

    async def recover_overflow(self, messages, **kw):
        self.recoveries += 1
        return CompactionResult("summarize", tokens_before=210_000, tokens_after_estimate=60_000)


async def test_injected_context_manager_recovers_an_overflow_once(config):
    script = iter([fail(ContextOverflowError("prompt is too long")), reply("answer after compaction")])
    events = []
    session = await _session(config, ScriptedProvider(lambda *a: next(script)),
                             on_event=lambda k, d: events.append((k, d)))
    rt = session.rt
    rt.context = fake = _FakeContext()
    res = await rt.run_agent(_probe(), "q", depth=1)
    assert res.status == "completed" and res.text == "answer after compaction"
    assert fake.recoveries == 1 and res.compactions == 1
    comp = _trace(rt, "compaction")
    assert comp and comp[0]["overflow"] is True and comp[0]["strategy"] == "summarize"
    assert any(k == "compaction" and d["tokens_after"] == 60_000 for k, d in events)
    await session.close()


async def test_second_overflow_ends_with_context_exceeded(config):
    provider = ScriptedProvider(lambda *a: fail(ContextOverflowError("prompt is too long")))
    session = await _session(config, provider)
    rt = session.rt
    rt.context = fake = _FakeContext()
    history = []
    res = await rt.run_agent(_probe(), "q", depth=1, history=history)
    assert res.status == "context_exceeded" and fake.recoveries == 1
    assert validate_tool_pairing(history) == []
    await session.close()


async def test_context_exceeded_stop_is_recovered_and_the_request_resent(config):
    script = iter([turn("partial", call("TodoWrite", todos=[]), stop="context_exceeded"), reply("complete answer")])
    session = await _session(config, ScriptedProvider(lambda *a: next(script)))
    rt = session.rt
    rt.context = fake = _FakeContext()
    history = []
    res = await rt.run_agent(_probe(tools=["TodoWrite"]), "q", depth=1, history=history)
    assert res.status == "completed" and res.text == "complete answer" and fake.recoveries == 1
    assert [m.text for m in history if m.role == "assistant"] == ["complete answer"]
    assert res.model_calls == 2
    await session.close()


async def test_retryable_provider_error_is_retried(config):
    script = iter([fail(), reply("ok after retry")])
    events = []
    session = await _session(config, ScriptedProvider(lambda *a: next(script)),
                             on_event=lambda k, d: events.append((k, d)))
    rt = session.rt
    rt.retry_policy = RetryPolicy(attempts=3, base_delay_s=0, max_delay_s=0, jitter=0)
    res = await rt.run_agent(_probe(), "q", depth=1)
    assert res.text == "ok after retry" and res.retries == 1
    rec = _trace(rt, "provider_retry")
    assert rec and rec[0]["attempt"] == 1 and "overloaded" in rec[0]["error"]
    assert [d["attempt"] for k, d in events if k == "retry"] == [1]
    await session.close()


# --------------------------------------------------------------------------- (N) system prompt


async def test_cso_system_prompt_is_built_once_and_split_for_caching(config):
    systems = []

    def script(agent, system, messages, tools):
        systems.append(system)
        return reply("ok")

    session = await _session(config, ScriptedProvider(script))
    rt = session.rt
    await rt.run_agent(rt.cso, "q1", history=session.history)
    await rt.run_agent(rt.cso, "q2", history=session.history)
    cached = rt._system_cache["cso"]
    assert cached[0].cache is True and cached[1].cache is False
    assert "Today's date" not in cached[0].text and "Today's date" in cached[1].text
    assert systems[0] == systems[1]
    await session.close()


async def test_cancel_outstanding_cancels_running_tool_tasks(config):
    slow = _Slow()
    provider = ScriptedProvider.from_rules({"probe": [reply(call("slow_wait")), reply("continued")]})
    session = await _session(config, provider)
    rt = session.rt
    history = []
    task = asyncio.ensure_future(rt.run_agent(_probe(), "q", depth=1, history=history, extra_tools=[slow.tool]))
    await asyncio.wait_for(slow.started.wait(), 5)
    await rt.cancel_outstanding()
    res = await task
    assert slow.cancelled and res.text == "continued"
    assert _results(history[2])[0].is_error
    await session.close()


async def test_real_context_manager_compacts_charges_and_traces(config):
    from vbt.providers.base import Usage

    def usage(agent, messages, msg):
        if agent == "context-compactor":
            return Usage(input_tokens=1_000, output_tokens=100)
        return Usage(input_tokens=180_000, output_tokens=50)  # 90% of the default 200k window

    provider = ScriptedProvider.from_rules({
        "probe": [reply(call("TodoWrite", todos=[])), reply(call("TodoWrite", todos=[])),
                  reply(call("TodoWrite", todos=[])), reply("done")],
        "context-compactor": [reply("SUMMARY: three todo updates, nothing else.")] * 5,
    }, usage_fn=usage)  # the mock keeps reporting 90% usage, so later rounds compact again
    session = await _session(config, provider)
    rt = session.rt
    history = []
    res = await rt.run_agent(_probe(tools=["TodoWrite"]), "keep working", depth=1, history=history)
    assert res.status == "completed" and res.compactions >= 1
    comp = _trace(rt, "compaction")
    assert comp and "summarize" in comp[0]["strategy"] and comp[0]["overflow"] is False
    assert len(comp) == res.compactions
    assert "SUMMARY: three todo updates" in history[0].text
    assert validate_tool_pairing(history) == []
    assert "context-compactor" in [c["agent"] for c in provider.calls]
    assert rt.run.cost.calls_by_agent["probe"] == res.model_calls + res.compactions
    await session.close()


async def test_failed_compaction_still_charges_the_summariser(config, monkeypatch):
    import vbt.context as ctxmod
    from vbt.providers.base import Usage

    def usage(agent, messages, msg):
        if agent == "context-compactor":
            return Usage(input_tokens=1_000, output_tokens=100)
        return Usage(input_tokens=180_000, output_tokens=50)

    real = ctxmod.validate_tool_pairing

    def flaky(messages, **kw):  # any summarised history is reported broken
        if any(ctxmod.SUMMARY_HEADER in m.text for m in messages[:1]):
            return ["invented pairing problem"]
        return real(messages, **kw)

    monkeypatch.setattr(ctxmod, "validate_tool_pairing", flaky)
    provider = ScriptedProvider.from_rules({
        "probe": [reply(call("TodoWrite", todos=[])), reply(call("TodoWrite", todos=[])),
                  reply(call("TodoWrite", todos=[])), reply("done")],
        "context-compactor": [reply("SUMMARY: x")] * 5,
    }, usage_fn=usage)
    session = await _session(config, provider)
    rt = session.rt
    res = await rt.run_agent(_probe(tools=["TodoWrite"]), "keep working", depth=1, history=[])
    summariser_calls = sum(1 for c in provider.calls if c["agent"] == "context-compactor")
    assert res.status == "completed" and res.compactions == 0 and summariser_calls >= 1
    assert _trace(rt, "compaction_failed") and len(_trace(rt, "compaction_failed_spend")) == summariser_calls
    assert rt.run.cost.calls_by_agent["probe"] == res.model_calls + summariser_calls
    await session.close()


async def test_tool_context_trace_carries_the_invocation_ids(config):
    def tracer(ctx, a):
        ctx.trace("custom_tool_event", detail="x")
        return "ok"

    script = iter([reply(call("tracer")), reply("done")])
    session = await _session(config, ScriptedProvider(lambda *a: next(script)))
    rt = session.rt
    res = await rt.run_agent(_probe(), "q", depth=1, extra_tools=[Tool("tracer", "x", schema({}), tracer)])
    ev = _trace(rt, "custom_tool_event")[0]
    assert ev["agent_run_id"] == res.invocation_id and ev["agent"] == "probe" and ev["tool_use_id"]
    await session.close()
