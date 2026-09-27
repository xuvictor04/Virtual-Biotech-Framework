"""Request encoding of the Claude adapter (no network)."""

from vbt.providers.anthropic_provider import AnthropicProvider, cost_usd
from vbt.providers.base import Message, ModelSettings, TextBlock, ThinkingBlock, ToolCall, ToolResult, ToolSpec, Usage


def _p():
    return AnthropicProvider(api_key="test-key")


def test_adaptive_model_request():
    req = _p()._request(ModelSettings("anthropic", "claude-opus-5", effort="high"), "sys",
                        [Message.user("hi")], [ToolSpec("Read", "read", {"type": "object"})])
    assert req["thinking"]["type"] == "adaptive"
    assert req["output_config"] == {"effort": "high"}
    assert req["tools"][0]["name"] == "Read"
    assert "temperature" not in req


def test_paper_models_use_budget_thinking_and_no_effort():
    req = _p()._request(ModelSettings("anthropic", "claude-sonnet-4-5", effort=None, thinking=True,
                                      max_tokens=64000), "sys", [Message.user("hi")], [])
    assert req["thinking"]["type"] == "enabled" and req["thinking"]["budget_tokens"] < 64000
    assert "output_config" not in req
    req = _p()._request(ModelSettings("anthropic", "claude-haiku-4-5", effort=None, thinking=False,
                                      max_tokens=16000), "sys", [Message.user("hi")], [])
    assert "thinking" not in req


def test_history_encoding_drops_foreign_thinking_and_keeps_own():
    own = ThinkingBlock("x", "anthropic", native={"type": "thinking", "thinking": "x", "signature": "sig"})
    foreign = ThinkingBlock("y", "other", native={"type": "reasoning"})
    msgs = [Message.user("q"),
            Message("assistant", [own, foreign, TextBlock("t"), ToolCall("id1", "Read", {"file_path": "a"})]),
            Message("user", [ToolResult("id1", "", False)])]
    enc = _p()._encode_messages(msgs)
    types = [b["type"] for b in enc[1]["content"]]
    assert types == ["thinking", "text", "tool_use"]
    assert enc[2]["content"][0]["content"] == "(empty result)"


def test_cost():
    assert abs(cost_usd("claude-opus-5", Usage(1_000_000, 1_000_000)) - 30.0) < 1e-9
    assert abs(cost_usd("claude-haiku-4-5", Usage(0, 0, cache_read_tokens=1_000_000)) - 0.1) < 1e-9
