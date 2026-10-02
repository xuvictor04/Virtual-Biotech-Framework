"""Scripted provider contract and the neutral history helpers (no network)."""

import json

import pytest

from vbt.providers import create_provider
from vbt.providers.base import (
    DocumentPart,
    ImagePart,
    Message,
    ModelSettings,
    OpaqueBlock,
    ProviderCapabilities,
    ProviderError,
    RetryableProviderError,
    StopReason,
    SystemSegment,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    Usage,
    content_text,
    message_from_dict,
    message_to_dict,
    messages_from_dicts,
    messages_to_dicts,
    system_text,
    unanswered_tool_calls,
    validate_tool_pairing,
)
from vbt.providers.mock import ScriptedProvider, call, fail, reply, turn


def S(agent="cso", **kw):
    return ModelSettings("mock", "mock-model", extra={"agent_name": agent}, **kw)


# ---------------------------------------------------------------- pairing


def _paired():
    c = ToolCall("t1", "Read", {"file_path": "a"})
    return [Message.user("q"), Message("assistant", [TextBlock("look"), c]),
            Message("user", [ToolResult("t1", "content"), TextBlock("[harness] note after results")])]


def test_validate_tool_pairing_accepts_valid_history():
    assert validate_tool_pairing(_paired()) == []
    pending = _paired()[:2]
    assert validate_tool_pairing(pending, allow_pending=True) == []
    assert any("unpaired tool_use" in p for p in validate_tool_pairing(pending))


@pytest.mark.parametrize("case", ["dangling_then_user_text", "orphan", "results_after_text", "duplicate",
                                  "assistant_follows"])
def test_validate_tool_pairing_rejects(case):
    c = ToolCall("t1", "Read", {})
    if case == "dangling_then_user_text":
        msgs = [Message.user("q"), Message("assistant", [c]), Message.user("next turn")]
    elif case == "orphan":
        msgs = [Message.user("q"), Message("assistant", [TextBlock("hi")]), Message("user", [ToolResult("zz", "x")])]
    elif case == "results_after_text":
        msgs = [Message.user("q"), Message("assistant", [c]), Message("user", [TextBlock("x"), ToolResult("t1", "r")])]
    elif case == "duplicate":
        msgs = [Message.user("q"), Message("assistant", [c]),
                Message("user", [ToolResult("t1", "r"), ToolResult("t1", "r")])]
    else:
        msgs = [Message.user("q"), Message("assistant", [c]), Message("assistant", [TextBlock("x")])]
    assert validate_tool_pairing(msgs)


def test_unanswered_tool_calls():
    c1, c2 = ToolCall("t1", "Read", {}), ToolCall("t2", "Bash", {})
    msgs = [Message.user("q"), Message("assistant", [c1, c2]), Message("user", [ToolResult("t1", "r")])]
    assert [(i, c.id) for i, c in unanswered_tool_calls(msgs)] == [(1, "t2")]


# ---------------------------------------------------------------- strict mock


async def test_strict_mock_rejects_unpaired_history():
    p = ScriptedProvider.from_rules({"cso": [reply("ok")]})
    bad = [Message.user("q"), Message("assistant", [ToolCall("t1", "Task", {})]), Message.user("second turn")]
    with pytest.raises(ProviderError, match="unpaired tool_use"):
        await p.complete(settings=S(), system="", messages=bad, tools=[])
    lenient = ScriptedProvider.from_rules({"cso": [reply("ok")]}, strict=False)
    assert (await lenient.complete(settings=S(), system="", messages=bad, tools=[])).message.text == "ok"


async def test_agent_name_from_settings_then_tag():
    p = ScriptedProvider.from_rules({"genomics-analyst": [reply("from extra")], "cso": [reply("from tag")]})
    r1 = await p.complete(settings=S("genomics-analyst"), system="<agent-name>cso</agent-name>",
                          messages=[Message.user("q")], tools=[])
    r2 = await p.complete(settings=ModelSettings("mock", "m"),
                          system=[SystemSegment("static", cache=True), SystemSegment("<agent-name>cso</agent-name>")],
                          messages=[Message.user("q")], tools=[])
    assert (r1.message.text, r2.message.text) == ("from extra", "from tag")
    assert [c["agent"] for c in p.calls] == ["genomics-analyst", "cso"]


async def test_turn_overrides_and_thinking_stream():
    usage = Usage(input_tokens=150_000, output_tokens=10)
    p = ScriptedProvider.from_rules({"cso": [
        turn("partial answer", stop="max_tokens", usage=usage, thinking="Considering L2G.", cost_usd=0.5),
        turn(call("Read", file_path="x"), stop=StopReason.REFUSAL),
    ]})
    thoughts, texts = [], []
    r = await p.complete(settings=S(), system="", messages=[Message.user("q")], tools=[],
                         on_text=texts.append, on_thinking=thoughts.append)
    assert r.stop_reason is StopReason.MAX_TOKENS and r.usage is usage and r.cost_usd == 0.5
    assert thoughts == ["Considering L2G."] and texts == ["partial answer"]
    assert isinstance(r.message.content[0], ThinkingBlock)
    r2 = await p.complete(settings=S(), system="", messages=[Message.user("q")], tools=[])
    assert r2.stop_reason is StopReason.REFUSAL and r2.message.tool_calls


async def test_fail_items_and_callables():
    p = ScriptedProvider.from_rules({"cso": [fail(), fail(ProviderError("bad")), lambda msgs: fail(KeyError),
                                             lambda msgs: turn(f"saw {len(msgs)}")]})
    with pytest.raises(RetryableProviderError):
        await p.complete(settings=S(), system="", messages=[Message.user("q")], tools=[])
    with pytest.raises(ProviderError, match="bad"):
        await p.complete(settings=S(), system="", messages=[Message.user("q")], tools=[])
    with pytest.raises(KeyError):
        await p.complete(settings=S(), system="", messages=[Message.user("q")], tools=[])
    r = await p.complete(settings=S(), system="", messages=[Message.user("q")], tools=[])
    assert r.message.text == "saw 1"


async def test_usage_fn_context_window_capabilities_and_search():
    caps = ProviderCapabilities(images=False, history_bound_thinking=True)
    p = ScriptedProvider.from_rules({}, usage_fn=lambda agent, msgs, msg: Usage(input_tokens=999),
                                    context_window=50_000, capabilities=caps)
    r = await p.complete(settings=S(), system="", messages=[Message.user("q")], tools=[])
    assert r.usage.input_tokens == 999 and r.message.text == "Done."
    assert p.context_window("x") == 50_000 and p.capabilities("x") is caps and p.check_credentials() is None
    out = await p.web_search("EGFR", allowed_domains=["nih.gov"])
    assert out["cost_usd"] == 0.0 and p.searches[0]["allowed_domains"] == ["nih.gov"]
    factory = create_provider("mock", context_window=1234, strict=False)
    assert factory.context_window("m") == 1234 and factory.strict is False


# ---------------------------------------------------------------- serialisation & parts


def test_message_to_dict_round_trip_is_lossless():
    native_thinking = {"type": "thinking", "thinking": "hmm", "signature": "abc=="}
    msgs = [
        Message("user", [TextBlock("Evaluate EGFR"), TextBlock("cited", native={"citations": [{"x": 1}]})]),
        Message("assistant", [ThinkingBlock("hmm", "anthropic", native=native_thinking),
                              ThinkingBlock("", "anthropic", native={"type": "redacted_thinking", "data": "zz"}),
                              TextBlock("Calling tools"),
                              ToolCall("t1", "Read", {"file_path": "a.png", "nested": {"k": [1, 2]}}),
                              ToolCall("t2", "Read", {"file_path": "b.pdf"}),
                              OpaqueBlock("anthropic", {"type": "fallback", "from": {"model": "x"}})]),
        Message("user", [ToolResult("t1", [TextBlock("a.png"), ImagePart("image/png", "aGk=", "a.png")]),
                         ToolResult("t2", [DocumentPart(data_b64="cGRm", title="b.pdf")], is_error=True)]),
    ]
    data = json.loads(json.dumps(messages_to_dicts(msgs)))
    back = messages_from_dicts(data)
    assert back == msgs
    assert message_from_dict(message_to_dict(msgs[1])) == msgs[1]
    back[1].content[0].native["signature"] = "mutated"
    assert msgs[1].content[0].native["signature"] == "abc=="  # deep copies, no aliasing


def test_content_text_and_system_text():
    parts = [TextBlock("header"), ImagePart("image/png", "aGk=", "plots/umap.png"), ImagePart("image/png", "aGk="),
             DocumentPart(data_b64="cGRm", title="label.pdf")]
    assert content_text(parts) == "header\n[image: plots/umap.png]\n[image: image/png]\n[document: label.pdf]"
    assert content_text("plain") == "plain" and content_text(None) == ""
    assert system_text([SystemSegment("a", cache=True), SystemSegment("b")]) == "a\n\nb"
    assert system_text("x") == "x"


def test_usage_add_and_dict():
    a = Usage(1, 2, 3, 4, 1, {"web_search": 1})
    b = Usage(10, 20, 30, 40, 10, {"web_search": 2, "web_fetch": 1})
    s = a + b
    assert (s.input_tokens, s.cache_write_1h_tokens, s.server_tool_requests) == (11, 11, {"web_search": 3, "web_fetch": 1})
    assert s.as_dict()["cache_write_1h_tokens"] == 11 and s.context_tokens == 11 + 33 + 44
    assert a.server_tool_requests == {"web_search": 1}  # inputs untouched
