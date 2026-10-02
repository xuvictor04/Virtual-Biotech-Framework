"""Context-window management: tool-result clearing, summarisation, overflow recovery."""

import base64
from pathlib import Path

import pytest

from vbt.context import (
    CLEARED_PREFIX,
    SUMMARIZER_AGENT,
    SUMMARY_HEADER,
    ContextManager,
    ContextPolicy,
    estimate_tokens,
)
from vbt.providers.base import (
    ContextOverflowError,
    ImagePart,
    Message,
    ModelSettings,
    ProviderCapabilities,
    ProviderError,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    Usage,
    validate_tool_pairing,
)
from vbt.providers.mock import ScriptedProvider, fail, reply

WINDOW = 100_000
TASK = "TASK: Evaluate EGFR as a target in lung adenocarcinoma."
CLARIFICATION = "Answer: focus on LUAD and SCLC; ignore mesothelioma."


def settings(agent="genomics-analyst", **extra):
    return ModelSettings("mock", "mock-model", effort="high", extra={"agent_name": agent, **extra})


def history(rounds=10, chars=8000, thinking=True, clarification_at=3):
    """[task, (assistant tool call, user tool result) x rounds] with one plain user turn."""
    msgs = [Message.user(TASK)]
    for r in range(rounds):
        blocks = [ThinkingBlock(f"think {r}", "mock")] if thinking else []
        if r == clarification_at:
            msgs.append(Message("assistant", blocks + [TextBlock("Which subtypes should I cover?")]))
            msgs.append(Message.user(CLARIFICATION))
            continue
        msgs.append(Message("assistant", blocks + [TextBlock(f"step {r}"),
                                                   ToolCall(f"t{r}", "Bash", {"command": f"analysis {r}"})]))
        msgs.append(Message("user", [ToolResult(f"t{r}", f"R{r}:" + "x" * chars)]))
    return msgs


def provider(summary="SUMMARY: plan step 4 of 6; L2G 0.82 for EGFR; artifact work/genomics-analyst/l2g.csv",
             caps=None, **kw):
    return ScriptedProvider.from_rules({SUMMARIZER_AGENT: [summary] if not isinstance(summary, str)
                                        else [reply(summary)]},
                                       context_window=WINDOW, capabilities=caps, **kw)


def manager(p, tmp_path, **policy):
    return ContextManager(p, ContextPolicy(**policy), spill_dir=tmp_path / "spill")


async def test_below_soft_threshold_does_nothing(tmp_path):
    msgs = history()
    before = list(msgs)
    res = await manager(provider(), tmp_path).maybe_compact(
        msgs, last_usage=Usage(input_tokens=20_000), settings=settings(), system="sys", agent="genomics-analyst")
    assert res is None and msgs == before


async def test_clears_old_tool_results_and_spills_them(tmp_path):
    msgs = history()
    originals = {b.tool_call_id: b.content for m in msgs for b in m.content if isinstance(b, ToolResult)}
    head = msgs[0]
    cm = manager(provider(), tmp_path)
    res = await cm.maybe_compact(msgs, last_usage=Usage(input_tokens=60_000, cache_read_tokens=12_000),
                                 settings=settings(), system="sys", agent="genomics-analyst")
    assert res is not None and res.strategy == "clear_tool_results"
    assert res.tokens_after_estimate < res.tokens_before and res.cost_usd == 0.0
    results = {b.tool_call_id: b.content for m in msgs for b in m.content if isinstance(b, ToolResult)}
    cleared = sorted(k for k, v in results.items() if v.startswith(CLEARED_PREFIX))
    assert cleared == ["t0", "t1", "t2"]  # older than the last 6 model calls
    for k in cleared:
        assert "use QueryToolOutput or Read" in results[k]
        path = Path(results[k].split("full output at ")[1].split(";")[0])
        assert path.parent == tmp_path / "spill" and path.read_text() == originals[k]
    assert all(results[k] == originals[k] for k in results if k not in cleared)
    assert msgs[0] is head and validate_tool_pairing(msgs) == []
    # thinking blocks stay: the mock's thinking is not bound to the history
    assert sum(isinstance(b, ThinkingBlock) for m in msgs for b in m.content) == 10


async def test_summarizes_with_large_reported_usage(tmp_path):
    p = provider(usage_fn=lambda agent, msgs, msg: Usage(input_tokens=88_000 if agent != SUMMARIZER_AGENT else 900,
                                                        output_tokens=50))
    msgs = history()
    resp = await p.complete(settings=settings(), system="sys", messages=msgs, tools=[])  # large usage reported
    msgs.append(resp.message)  # "Done." final answer
    msgs.append(Message.user("Follow-up: what about SCLC?"))
    cm = manager(p, tmp_path)
    res = await cm.maybe_compact(msgs, last_usage=resp.usage, settings=settings(), system="sys",
                                 agent="genomics-analyst")
    assert res is not None and "summarize" in res.strategy
    assert res.usage.input_tokens == 900 and res.summarized_messages > 0 and res.thinking_dropped > 0
    # the summariser call: same model, thinking off, no tools, its own agent name
    summ_call = p.calls[-1]
    assert summ_call["agent"] == SUMMARIZER_AGENT and summ_call["tools"] == [] and summ_call["thinking"] is False
    assert summ_call["model"] == "mock-model"
    head = msgs[0]
    assert head.role == "user" and head.content[0].text == TASK  # head kept verbatim
    summary = head.content[-1].text
    assert summary.startswith(SUMMARY_HEADER) and "L2G 0.82" in summary
    assert CLARIFICATION in summary  # user turns preserved verbatim
    assert msgs[1].role == "assistant"  # tail starts with an assistant message
    assert msgs[-1].text == "Follow-up: what about SCLC?"
    assert validate_tool_pairing(msgs) == []
    assert estimate_tokens(msgs) < WINDOW * 0.5
    assert list((tmp_path / "spill").glob("*.txt"))  # cleared outputs left on disk


async def test_history_bound_thinking_is_stripped_after_the_edit(tmp_path):
    caps = ProviderCapabilities(history_bound_thinking=True)
    msgs = history()
    cm = manager(provider(caps=caps), tmp_path)
    await cm.maybe_compact(msgs, last_usage=Usage(input_tokens=72_000), settings=settings(), system="s",
                           agent="genomics-analyst")
    first_edit = next(i for i, m in enumerate(msgs) for b in m.content
                      if isinstance(b, ToolResult) and str(b.content).startswith(CLEARED_PREFIX))
    for i, m in enumerate(msgs):
        has_thinking = any(isinstance(b, ThinkingBlock) for b in m.content)
        if m.role == "assistant":
            assert has_thinking == (i < first_edit)
    assert validate_tool_pairing(msgs) == []


async def test_already_spilled_output_is_referenced_not_rewritten(tmp_path):
    full = tmp_path / "work" / "_tool_outputs" / "t0.txt"
    full.parent.mkdir(parents=True)
    full.write_text("FULL OUTPUT " * 10_000)
    msgs = history()
    msgs[2] = Message("user", [ToolResult("t0", "partial " * 1000 + f"\n\n[Output truncated: 120,000 chars. Full "
                                                                  f"output saved to {full}; Read it with offset/limit.]")])
    await manager(provider(), tmp_path).maybe_compact(msgs, last_usage=Usage(input_tokens=72_000),
                                                      settings=settings(), system="s", agent="a")
    stub = msgs[2].content[0].content
    assert stub.startswith(CLEARED_PREFIX) and f"full output at {full};" in stub
    assert not (tmp_path / "spill" / "t0.txt").exists()


async def test_images_in_old_results_are_spilled_as_files(tmp_path):
    msgs = history(rounds=10, chars=8000)
    png = base64.b64encode(b"\x89PNG\r\n fake image").decode()
    msgs[2] = Message("user", [ToolResult("t0", [TextBlock("umap.png"), ImagePart("image/png", png, "umap.png")])])
    await manager(provider(), tmp_path).maybe_compact(msgs, last_usage=Usage(input_tokens=72_000),
                                                      settings=settings(), system="s", agent="a")
    stub = msgs[2].content[0].content
    assert isinstance(stub, str) and "image part(s)" in stub
    assert (tmp_path / "spill" / "t0.part0.png").read_bytes() == b"\x89PNG\r\n fake image"


async def test_mid_round_compaction_keeps_pending_calls(tmp_path):
    msgs = history()
    msgs.append(Message("assistant", [ThinkingBlock("pending", "mock"), ToolCall("tp", "Bash", {"command": "x"})]))
    res = await manager(provider(), tmp_path).maybe_compact(msgs, last_usage=Usage(input_tokens=90_000),
                                                            settings=settings(), system="s", agent="a")
    assert res is not None and "summarize" in res.strategy
    assert msgs[-1].tool_calls[0].id == "tp"
    assert validate_tool_pairing(msgs, allow_pending=True) == []


async def test_repeated_compaction_carries_verbatim_user_turns(tmp_path):
    prompts = []

    def second(msgs):
        prompts.append(msgs[0].text)
        return reply("second summary")

    p = ScriptedProvider.from_rules({SUMMARIZER_AGENT: [reply("first summary"), second]}, context_window=WINDOW)
    cm = manager(p, tmp_path)
    msgs = history()
    await cm.maybe_compact(msgs, last_usage=Usage(input_tokens=90_000), settings=settings(), system="s", agent="a")
    # the agent keeps working; a second compaction folds the first summary in
    for r in range(20, 30):
        msgs.append(Message("assistant", [ToolCall(f"t{r}", "Bash", {"command": str(r)})]))
        msgs.append(Message("user", [ToolResult(f"t{r}", "y" * 8000)]))
    await cm.maybe_compact(msgs, last_usage=Usage(input_tokens=90_000), settings=settings(), system="s", agent="a")
    summaries = [b.text for b in msgs[0].content if isinstance(b, TextBlock) and b.text.startswith(SUMMARY_HEADER)]
    assert len(summaries) == 1 and "second summary" in summaries[0] and CLARIFICATION in summaries[0]
    assert "first summary" in prompts[0]  # the earlier summary is folded into the new one
    assert "first summary" not in summaries[0]
    assert validate_tool_pairing(msgs) == []


async def test_summariser_failure_falls_back_to_a_mechanical_digest(tmp_path):
    p = provider(summary=fail(ProviderError("summariser down")))
    msgs = history()
    res = await manager(p, tmp_path).maybe_compact(msgs, last_usage=Usage(input_tokens=90_000),
                                                   settings=settings(), system="s", agent="a")
    assert "summarize" in res.strategy and "mechanical" in res.note
    assert "Mechanical digest" in msgs[0].content[-1].text and validate_tool_pairing(msgs) == []


async def test_recover_overflow_compacts_aggressively(tmp_path):
    msgs = history(rounds=12, chars=20_000)
    cm = manager(provider(), tmp_path)
    res = await cm.recover_overflow(msgs, settings=settings(context_window_tokens=60_000), system="s", agent="a")
    assert "summarize" in res.strategy
    assert sum(1 for m in msgs if m.role == "assistant") == 1  # only the last round is kept verbatim
    assert estimate_tokens(msgs) < 60_000 and validate_tool_pairing(msgs) == []
    with pytest.raises(ContextOverflowError):
        await cm.recover_overflow([Message.user("x" * 1000)], settings=settings(), system="s", agent="a")


async def test_failed_compaction_restores_history(tmp_path):
    p = provider(summary=fail(KeyboardInterrupt))
    msgs = history()
    before = list(msgs)
    with pytest.raises(KeyboardInterrupt):
        await manager(p, tmp_path).maybe_compact(msgs, last_usage=Usage(input_tokens=90_000), settings=settings(),
                                                 system="s", agent="a")
    assert msgs == before


def test_window_policy_and_server_side_settings():
    p = ScriptedProvider.from_rules({}, context_window=123_000,
                                    capabilities=ProviderCapabilities(server_context_management=True))
    cm = ContextManager(p, ContextPolicy.from_config({"context": {"soft_ratio": 0.6, "server_side": "true",
                                                                  "keep_recent_calls": 4, "bogus": 1}}))
    assert (cm.policy.soft_ratio, cm.policy.server_side, cm.policy.keep_recent_calls) == (0.6, True, 4)
    assert cm.window_for(settings()) == 123_000
    assert cm.window_for(settings(context_window_tokens=200_000)) == 200_000
    assert cm.settings_for(settings()).extra["context_management"] is True
    assert ContextManager(ScriptedProvider.from_rules({})).window_for(settings()) == 200_000
    assert ContextManager.used_tokens(Usage(10, 99, 20, 30)) == 60
    assert ContextPolicy.from_config(None) == ContextPolicy()
    off = ContextManager(ScriptedProvider.from_rules({}), ContextPolicy(server_side=True))
    assert "context_management" not in off.settings_for(settings()).extra  # mock lacks the capability
