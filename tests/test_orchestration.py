"""End-to-end orchestration with the scripted provider (no network)."""

import asyncio
import json

import pytest

from vbt.orchestrator import META_QUERY, NO_CLARIFY, CSOSession, SessionBusy, open_session  # noqa: F401
from vbt.providers.mock import ScriptedProvider, call, fail, reply, turn
from vbt.tools.base import Tool, ToolFailure, schema


def _results(messages):
    """Tool-result texts in the last user message."""
    return [b.content for b in messages[-1].content if b.type == "tool_result"]


async def test_delegation_review_and_claims(config):
    def genomics(messages):
        return reply("L2G score 0.82 for EGFR in lung adenocarcinoma. Evidence: strong.")

    rules = {
        "cso": [
            reply("Dispatching two specialists in parallel.",
                  call("Task", subagent_type="genomics-analyst", description="genetics", prompt="EGFR genetics"),
                  call("Task", subagent_type="single-cell-analyst", description="expr", prompt="EGFR expression")),
            reply("EGFR is supported."),  # tries to finish without review -> harness nudges
            reply(call("Task", subagent_type="scientific-reviewer", description="review", prompt="review all")),
            reply("Final synthesis: EGFR is a strong target."),
        ],
        "genomics-analyst": [
            reply(call("Write", file_path="results/tables/l2g.csv", content="gene,l2g\nEGFR,0.82\n")),
            reply(call("mcp__provenance__register_artifact", path="work/genomics-analyst/results/tables/l2g.csv",
                       description="L2G scores")),
            genomics,
        ],
        "single-cell-analyst": [reply("EGFR highest in alveolar type 2 cells.")],
        "scientific-reviewer": [reply("APPROVED: conclusions supported.")],
    }
    provider = ScriptedProvider.from_rules(rules)
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("Is EGFR a good target in lung cancer?")
    assert out == "Final synthesis: EGFR is a strong target."

    events = session.run.events()
    kinds = [e["type"] for e in events]
    assert kinds.count("delegation") == 3
    assert "review_enforced" in kinds
    arts = json.loads((session.run.dir / "evidence" / "artifacts.json").read_text())
    assert arts[0]["path"] == "work/genomics-analyst/results/tables/l2g.csv"
    report = json.loads((session.run.dir / "session_report.json").read_text())
    assert set(report["turns"][0]["agents"]) == {"genomics-analyst", "single-cell-analyst", "scientific-reviewer"}
    await session.close()
    assert (session.run.dir / "MANIFEST.json").exists()


async def test_cso_cannot_use_data_tools_and_scientists_cannot_delegate(config):
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Bash", command="echo hi"),
                      call("Task", subagent_type="genomics-analyst", prompt="x")), reply("ok")],
        "genomics-analyst": [reply(call("Task", subagent_type="single-cell-analyst", prompt="y")), reply("done")],
    })
    config["orchestration"]["enforce_review"] = False
    session = await open_session(config, provider=provider, start_mcp=False)
    await session.ask("q")
    ends = [e for e in session.run.events() if e["type"] == "tool_end"]
    by_tool = {(e["agent"], e["tool"]): e for e in ends}
    assert by_tool[("cso", "Bash")]["is_error"]
    assert by_tool[("genomics-analyst", "Task")]["is_error"]
    await session.close()


async def test_orientation_runs_brief_and_clarification(config):
    config["orchestration"]["strategic_orientation"] = True
    provider = ScriptedProvider.from_rules({
        "chief-of-staff": [reply("BRIEF: B7-H3 is an emerging checkpoint target.")],
        "cso": [reply("1) Which lung cancer subtypes?"),
                lambda msgs: reply("Saw brief" if "BRIEF" in msgs[-1].text else "no brief")],
    })
    config["orchestration"]["enforce_review"] = False
    session = await open_session(config, provider=provider, start_mcp=False)
    first = await session.ask("Evaluate B7-H3 in lung cancer")
    assert "BRIEF" in first and "subtypes" in first
    second = await session.ask("LUAD and SCLC")
    assert second == "Saw brief"
    await session.close()


async def test_write_outside_run_is_blocked(config, tmp_path):
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", prompt="x")), reply("ok")],
        "genomics-analyst": [reply(call("Write", file_path=str(tmp_path / "escape.txt"), content="x")),
                             reply("done")],
    })
    config["orchestration"]["enforce_review"] = False
    session = await open_session(config, provider=provider, start_mcp=False)
    await session.ask("q")
    assert not (tmp_path / "escape.txt").exists()
    await session.close()


# ---------------------------------------------------------------- P3: session lifecycle

WARNING_FMT = ("Data/evidence warning: these tools failed during this turn: {names}. Their results cannot "
               "support this answer. Any alternative sources must be identified separately; evidence "
               "coverage is incomplete.")


class BlockingProvider(ScriptedProvider):
    """Scripted provider whose calls for ``blocked`` agents never return (until cancelled)."""

    def __init__(self, script, blocked=(), **kw):
        super().__init__(script, **kw)
        self.blocked = set(blocked)
        self.started = asyncio.Event()

    async def complete(self, *, settings, **kw):
        if (settings.extra or {}).get("agent_name") in self.blocked:
            self.started.set()
            await asyncio.Event().wait()
        return await super().complete(settings=settings, **kw)


def blocking(rules, blocked):
    return BlockingProvider(ScriptedProvider.from_rules(rules).script, blocked=blocked)


def _turn(session, n=None):
    turns = session.run.turns
    return turns[-1] if n is None else next(t for t in turns if t["turn"] == n)


def _types(session):
    return [e["type"] for e in session.run.events()]


async def test_reply_keeps_synthesis_emitted_with_record_claims(config):
    config["orchestration"]["enforce_review"] = False
    claim = {"id": "C1", "text": "EGFR is amplified in LUAD.", "confidence": "high",
             "evidence": [{"kind": "citation", "pmid": "12345678"}]}
    provider = ScriptedProvider.from_rules({"cso": [
        reply("Synthesis: EGFR is a strong target [[claim:C1]].",
              call("mcp__provenance__record_claims", claims=[claim])),
        reply("Done."),
    ]})
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("Is EGFR a target?")
    assert "Synthesis: EGFR is a strong target [[claim:C1]]." in out and out.endswith("Done.")
    rec = _turn(session)
    assert rec["response"] == out and rec["status"] == "completed"
    assert rec["claims_filed"] == ["C1"]
    await session.close()
    report = (session.run.dir / "report" / "FINAL_REPORT.md").read_text()
    assert "[[claim:C1]]" in report and "Synthesis" in report


async def test_cancel_mid_delegation_records_interrupted_turn_and_session_recovers(config):
    config["orchestration"]["enforce_review"] = False
    holder = {}

    def specialist(messages):
        holder["session"].cancel()  # Ctrl+C while the specialist works
        return reply(call("Read", file_path="nothing.txt"))

    provider = ScriptedProvider.from_rules({
        "cso": [reply("Dispatching genetics.", call("Task", subagent_type="genomics-analyst",
                                                     description="genetics", prompt="EGFR genetics")),
                reply("Recovered: here is a fresh answer.")],
        "genomics-analyst": [specialist],
    })  # strict mock: an unpaired tool_use in the history would fail the next call
    session = await open_session(config, provider=provider, start_mcp=False)
    holder["session"] = session
    with pytest.raises(asyncio.CancelledError):
        await session.ask("q1")
    rec = _turn(session, 1)
    assert rec["status"] == "interrupted"
    assert rec["response"].startswith("Dispatching genetics.")
    assert "Turn interrupted: cancelled by the user. Evidence coverage is incomplete." in rec["response"]
    assert "genomics-analyst" in rec["agents"]
    assert rec["delegations"][0]["status"] == "cancelled"
    assert not session.busy
    out = await session.ask("q2")
    assert out == "Recovered: here is a fresh answer."
    assert _turn(session, 2)["status"] == "completed"
    await session.close()
    manifest = json.loads((session.run.dir / "MANIFEST.json").read_text())
    assert manifest["status"] == "interrupted" and 1 in manifest["interrupted_turns"]


async def test_concurrent_ask_is_rejected(config):
    config["orchestration"]["enforce_review"] = False
    provider = blocking({"cso": [reply("never")]}, blocked={"cso"})
    session = await open_session(config, provider=provider, start_mcp=False)
    first = asyncio.ensure_future(session.ask("one"))
    await provider.started.wait()
    with pytest.raises(SessionBusy, match="previous query still processing"):
        await session.ask("two")
    assert session.cancel() is True
    with pytest.raises(asyncio.CancelledError):
        await first
    assert _turn(session)["status"] == "interrupted"
    assert session.cancel() is False
    await session.close()


async def test_chief_of_staff_failure_degrades_and_is_recorded(config):
    config["orchestration"].update(strategic_orientation=True, enforce_review=False)
    provider = ScriptedProvider.from_rules({
        "chief-of-staff": [fail(RuntimeError("search backend exploded"))],
        "cso": [reply("1) Which subtypes matter most?")],
    })
    events = []
    session = await open_session(config, provider=provider, start_mcp=False,
                                 on_event=lambda k, d: events.append((k, d)))
    out = await session.ask("Evaluate B7-H3")
    assert "[Chief of Staff briefing unavailable: RuntimeError: search backend exploded]" in out
    assert "Which subtypes" in out
    rec = _turn(session)
    assert rec["status"] == "completed" and "chief-of-staff" in rec["agents"]
    assert rec["delegations"][0]["agent"] == "chief-of-staff" and rec["delegations"][0]["status"] == "error"
    assert session.oriented
    briefings = [d for k, d in events if k == "briefing"]
    assert briefings and "unavailable" in briefings[0]["text"]
    end = [d for k, d in events if k == "turn_end"][-1]
    assert any("Which subtypes" in u for u in end["unstreamed"])
    await session.close()


async def test_clarification_failure_cancels_the_brief(config):
    config["orchestration"].update(strategic_orientation=True, enforce_review=False)
    provider = blocking({"cso": [fail(RuntimeError("clarification crashed"))]}, blocked={"chief-of-staff"})
    session = await open_session(config, provider=provider, start_mcp=False)
    with pytest.raises(RuntimeError, match="clarification crashed"):
        await session.ask("Evaluate B7-H3")
    cos = [d for d in session.rt.delegation_log if d["agent"] == "chief-of-staff"]
    assert cos and cos[0]["status"] == "cancelled"
    rec = _turn(session)
    assert rec["status"].startswith("failed: RuntimeError: clarification crashed")
    assert not session.oriented
    await session.close()


async def test_meta_query_skips_orientation_until_first_scientific_query(config):
    config["orchestration"].update(strategic_orientation=True, enforce_review=False)
    provider = ScriptedProvider.from_rules({
        "cso": [reply(META_QUERY), reply("I am the CSO of the Virtual Biotech."),
                reply(NO_CLARIFY), lambda msgs: reply("Brief seen" if "BRIEF" in msgs[-1].text else "no brief")],
        "chief-of-staff": [reply("BRIEF: field overview.")] * 2,  # the meta turn's brief is discarded
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("Hi, who are you?")
    assert out == "I am the CSO of the Virtual Biotech."
    assert not session.oriented
    out2 = await session.ask("Evaluate B7-H3 in lung cancer")
    assert out2 == "Brief seen" and session.oriented
    await session.close()


async def test_concurrently_finishing_reviewer_is_not_counted(config):
    config["orchestration"]["review_policy"] = "always"
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x"),
                      call("Task", subagent_type="scientific-reviewer", description="r", prompt="review")),
                reply("Synthesis without a real review."),
                reply(call("Task", subagent_type="scientific-reviewer", description="r2", prompt="review")),
                reply("Final synthesis.")],
        "genomics-analyst": [reply("L2G 0.8")],
        "scientific-reviewer": [reply("nothing to review yet"), reply("APPROVED")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("q")
    assert out == "Final synthesis."
    assert _types(session).count("review_enforced") == 1
    rec = _turn(session)
    assert rec["reviewed"] is True and rec["review_rounds"] == 1
    await session.close()


async def test_question_ending_synthesis_after_specialists_is_still_reviewed(config):
    # The upstream CSO prompt ends every substantive answer with "Would you like me to
    # pursue any of these?": that must not switch the review policy off (F1).
    config["orchestration"]["review_policy"] = "always"
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x")),
                reply("EGFR is supported.\n\nSuggested Next Steps: ... Would you like me to pursue any of these?"),
                reply(call("Task", subagent_type="scientific-reviewer", description="r", prompt="review")),
                reply("EGFR is supported (reviewed). Would you like me to pursue any of these?")],
        "genomics-analyst": [reply("done")],
        "scientific-reviewer": [reply("APPROVED")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("q")
    assert out.endswith("pursue any of these?")
    assert _types(session).count("review_enforced") == 1
    rec = _turn(session)
    assert rec["reviewed"] is True and rec["review_rounds"] == 1 and rec["review_exempt"] is None
    await session.close()


async def test_question_ending_plan_check_runs_after_specialists(config):
    config["orchestration"].update(enforce_review=False, review_policy="never", enforce_plan=True)
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x"),
                      call("Task", subagent_type="single-cell-analyst", description="s", prompt="y")),
                reply("Synthesis. Would you like me to pursue any of these?"),
                reply(call("mcp__provenance__write_plan", goal="EGFR", steps=[
                    {"id": "s1", "agent": "genomics-analyst", "task": "genetics"}])),
                reply("Synthesis restated. Would you like me to pursue any of these?")],
        "genomics-analyst": [reply("g")], "single-cell-analyst": [reply("s")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    await session.ask("q")
    assert "plan_nudge" in _types(session)
    assert _turn(session)["plan_nudged"] is True
    await session.close()


async def test_clarification_without_specialists_is_exempt_and_recorded(config):
    config["orchestration"]["review_policy"] = "always"
    provider = ScriptedProvider.from_rules({"cso": [reply("Which lung cancer subtype do you mean?")]})
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("q")
    assert out.endswith("?")
    types = _types(session)
    assert "review_enforced" not in types and "review_exempt" in types
    rec = _turn(session)
    assert rec["reviewed"] is False and rec["review_exempt"] == "awaiting_user"
    await session.close()


async def test_research_policy_skips_single_lookup_without_files(config):
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x")),
                reply("Quick answer.")],
        "genomics-analyst": [reply("L2G 0.8")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    assert await session.ask("q") == "Quick answer."
    assert "review_enforced" not in _types(session)
    await session.close()


async def test_review_skipped_when_rounds_are_exhausted(config):
    config["orchestration"].update(review_policy="always", max_review_rounds=1)
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x")),
                reply("Synthesis 1."), reply("Synthesis 2, still unreviewed.")],
        "genomics-analyst": [reply("done")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("q")
    assert out == "Synthesis 2, still unreviewed."
    types = _types(session)
    assert types.count("review_enforced") == 1 and "review_skipped" in types
    rec = _turn(session)
    assert rec["reviewed"] is False and rec["review_rounds"] == 1 and rec["review_skipped"] is True
    await session.close()


def _failing_mcp_tool(name="mcp__x__lookup"):
    def handler(ctx, a):
        raise ToolFailure(f"upstream service unavailable for {a.get('gene')}")
    return Tool(name, "test data tool", schema({"gene": {"type": "string"}}, ["gene"]), handler, source="mcp")


async def test_recovered_code_error_gives_no_data_warning(config):
    config["orchestration"]["enforce_review"] = False
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x")),
                reply("Answer.")],
        "genomics-analyst": [reply(call("Read", file_path="missing.csv")),
                             reply(call("Bash", command="echo recovered")), reply("done")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("q")
    assert out == "Answer."
    assert "Data/evidence warning" not in out
    rec = _turn(session)
    failed = [f for f in rec["tool_failures"] if f["tool"] == "Read"]
    assert failed and failed[0]["input"] == {"file_path": "missing.csv"}
    assert rec["data_source_failures"] == []
    assert rec["recovered_errors"] == 0 and rec["other_errors"] == 1  # Read was never retried identically
    await session.close()


async def test_unresolved_mcp_failure_warns_with_upstream_wording(config):
    config["orchestration"]["enforce_review"] = False
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x")),
                reply("Answer.")],
        "genomics-analyst": [reply(call("mcp__x__lookup", gene="EGFR")), reply("done")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    session.rt.registry.add(_failing_mcp_tool())
    session.rt.agents["genomics-analyst"].tools.append("mcp__x__lookup")
    events = []
    session.rt.bus.subscribe(lambda k, d: events.append((k, d)))
    out = await session.ask("q")
    wording = WARNING_FMT.format(names="mcp__x__lookup")
    assert out.startswith("Answer.") and wording in out
    rec = _turn(session)
    assert [f["tool"] for f in rec["data_source_failures"]] == ["mcp__x__lookup"]
    assert rec["data_source_failures"][0]["agent"] == "genomics-analyst"
    end = [d for k, d in events if k == "turn_end"][-1]
    assert wording in end["unstreamed"]
    await session.close()


async def test_chief_of_staff_web_failure_is_in_the_warning(config):
    config["orchestration"].update(strategic_orientation=True, enforce_review=False)
    provider = ScriptedProvider.from_rules({
        "chief-of-staff": [reply(call("WebSearch", query="B7-H3 2026")), reply("BRIEF: offline brief.")],
        "cso": [reply(NO_CLARIFY), reply("Answer after brief.")],
    })

    async def broken_search(query, **kw):
        raise RuntimeError("search quota exceeded")
    provider.web_search = broken_search
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("Evaluate B7-H3")
    assert out.startswith("Answer after brief.")
    assert WARNING_FMT.format(names="WebSearch") in out
    rec = _turn(session)
    assert rec["data_source_failures"][0]["agent"] == "chief-of-staff"
    assert "chief-of-staff" in rec["agents"]
    assert NO_CLARIFY not in out
    await session.close()


async def test_plan_nudge_fires_for_multi_specialist_turn_without_plan(config):
    config["orchestration"].update(enforce_review=False, enforce_plan=True)
    seen = {}

    def after_nudge(messages):
        seen["nudge"] = messages[-1].text
        return reply(call("mcp__provenance__write_plan", goal="EGFR", steps=[
            {"id": "s1", "agent": "genomics-analyst", "task": "genetics"},
            {"id": "s2", "agent": "single-cell-analyst", "task": "expression"}]))

    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x"),
                      call("Task", subagent_type="single-cell-analyst", description="s", prompt="y")),
                reply("Synthesis: EGFR [[claim:C1]]."), after_nudge, reply("Final.")],
        "genomics-analyst": [reply("g done")], "single-cell-analyst": [reply("s done")],
    })
    events = []
    session = await open_session(config, provider=provider, start_mcp=False,
                                 on_event=lambda k, d=None, **kw: events.append(k))
    out = await session.ask("q")
    # The CSO acknowledged the plan without restating: its synthesis stays in the reply.
    assert out == "Synthesis: EGFR [[claim:C1]].\n\nFinal."
    assert "restate" in seen["nudge"] and "draft_superseded" in events
    assert "mcp__provenance__write_plan" in seen["nudge"]
    assert "deviations are recorded, not forbidden" in seen["nudge"]
    assert "plan_nudge" in _types(session)
    rec = _turn(session)
    assert rec["plan_nudged"] is True and rec["plan_written"] is True and rec["plan_missing"] is False
    assert rec["drafts"] == ["Synthesis: EGFR [[claim:C1]]."] and rec["draft_kept"] is True
    assert "Synthesis: EGFR [[claim:C1]]." in (session.run.dir / "report" / "FINAL_REPORT.md").read_text()
    await session.close()


async def test_plan_nudge_with_short_ack_keeps_unanchored_synthesis(config):
    config["orchestration"].update(enforce_review=False, review_policy="never", enforce_plan=True)
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x"),
                      call("Task", subagent_type="single-cell-analyst", description="s", prompt="y")),
                reply("A long synthesis of genetics and expression evidence for EGFR."),
                reply(call("mcp__provenance__write_plan", goal="EGFR", steps=[
                    {"id": "s1", "agent": "genomics-analyst", "task": "genetics"}])),
                reply("Plan recorded.")],
        "genomics-analyst": [reply("g")], "single-cell-analyst": [reply("s")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("q")
    assert out == "A long synthesis of genetics and expression evidence for EGFR.\n\nPlan recorded."
    await session.close()


async def test_restated_synthesis_after_nudge_is_not_duplicated(config):
    config["orchestration"].update(review_policy="always")
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="g", prompt="x")),
                reply("Draft [[claim:C1]]."),
                reply(call("Task", subagent_type="scientific-reviewer", description="r", prompt="review")),
                reply("Reviewed synthesis [[claim:C1]].")],
        "genomics-analyst": [reply("g")], "scientific-reviewer": [reply("APPROVED")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("q")
    assert out == "Reviewed synthesis [[claim:C1]]."
    rec = _turn(session)
    assert rec["drafts"] == ["Draft [[claim:C1]]."] and rec["draft_kept"] is False
    await session.close()


async def test_turn_record_fields_state_file_and_pinned_config(config):
    config["orchestration"]["enforce_review"] = False
    provider = ScriptedProvider.from_rules({
        "cso": [turn("Thinking about EGFR", call("Task", subagent_type="genomics-analyst", description="g",
                                                  prompt="x"), thinking="considering the genetics evidence"),
                reply("Answer.")],
        "genomics-analyst": [reply("done")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    await session.ask("q")
    rec = _turn(session)
    for key in ("turn", "prompt", "response", "status", "started", "ended", "agents", "delegations", "cost_usd",
                "cumulative_cost_usd", "duration_s", "tool_failures", "data_source_failures", "reviewed",
                "review_rounds", "plan_written", "plan_missing", "compactions", "thinking_traces",
                "subagent_traces", "mcp_tools_used"):
        assert key in rec, key
    assert rec["thinking_traces"][0]["text"] == "considering the genetics evidence"
    sub = rec["subagent_traces"][0]
    assert sub["agent"] == "genomics-analyst" and sub["transcript_path"].startswith("logs/agents/")
    assert rec["delegations"][0]["tool_use_id"]
    state = json.loads((session.run.dir / "logs" / "cso_state.json").read_text())
    assert state["turn"] == 1 and state["history"] and state["delegation_log"][0]["agent"] == "genomics-analyst"
    await session.close()
    cfg = json.loads((session.run.dir / "inputs" / "config.json").read_text())
    manifest = json.loads((session.run.dir / "MANIFEST.json").read_text())
    assert cfg["agents"]["genomics-analyst"]["prompt_sha256"] and "commit" in cfg["harness"]
    assert cfg["orchestration"]["review_policy"] == "research" and cfg["interface"] == "chat"
    assert manifest["config"]["prompt_hashes"] == cfg["prompt_hashes"]
    assert (session.run.dir / "inputs" / "environment.txt").read_text().strip()


def _readiness_config(config, monkeypatch, problem):
    """A non-mock provider whose own check_credentials hook decides (F2)."""
    import vbt.preflight as pf
    import vbt.providers as providers

    config["provider"] = {"name": "fakeprov", "options": {"api_key": "from-profile"}}
    config["orchestration"]["require_reference_data"] = True
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(pf, "check_reference_data", lambda cfg: [])
    monkeypatch.setattr(pf, "check_mcp_commands", lambda cfg: [])
    built = {}

    def create_provider(name, **opts):
        p = ScriptedProvider.from_rules({"cso": [reply("ok")]})
        p.name = name
        p.check_credentials = lambda: problem
        built["opts"] = opts
        built["provider"] = p
        return p

    monkeypatch.setattr(providers, "create_provider", create_provider)
    return built


async def test_session_preflight_uses_the_configured_providers_credential_check(config, monkeypatch):
    built = _readiness_config(config, monkeypatch, None)
    session = await open_session(config, start_mcp=False)  # provider=None, as the CLI does
    assert built["opts"] == {"api_key": "from-profile"}
    assert session.rt.provider is built["provider"]
    await session.close()


async def test_session_preflight_reports_the_providers_own_problem(config, monkeypatch):
    from vbt.preflight import DataReadinessError

    _readiness_config(config, monkeypatch, "token expired")
    with pytest.raises(DataReadinessError, match="token expired"):
        await open_session(config, start_mcp=False)
