"""Model-family dialects for OpenAI-compatible servers (no network)."""

import pytest

from vbt.providers.families import (
    DEEPSEEK_V4,
    FAMILIES,
    GENERIC,
    GENERIC_WITH_EFFORT,
    HARNESS_EFFORTS,
    QWEN3,
    QWEN3_6,
    QWEN3_8,
    resolve_family,
)


@pytest.mark.parametrize("model_id, family", [
    ("qwen3.8-27b", "qwen3_8"),
    ("Qwen/Qwen3.8-27B-FP8", "qwen3_8"),
    ("RedHatAI/Qwen3.8-27B-NVFP4", "qwen3_8"),
    ("nvidia/Qwen3.8-27B-NVFP4", "qwen3_8"),
    ("RedHatAI/Qwen3.8-27B-INT4", "qwen3_8"),
    ("Qwen/Qwen3.6-35B-A3B-FP8", "qwen3_6"),
    ("qwen3.5-9b", "qwen3_6"),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek_v4"),
    ("dsv4-flash", "deepseek_v4"),
    ("Qwen/Qwen3-0.6B", "qwen3"),
    ("Qwen/Qwen3-8B", "qwen3"),          # the 2025 Qwen3-8B is not Qwen3.8
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "generic"),
    ("gpt-oss-20b", "generic"),
    ("local-model", "generic"),
    ("", "generic"),
    (None, "generic"),
])
def test_resolve_family_from_model_id(model_id, family):
    assert resolve_family(model_id).name == family


def test_explicit_family_wins_and_unknown_raises():
    assert resolve_family("Qwen/Qwen3-8B", "qwen3_8") is QWEN3_8
    assert resolve_family("anything", "deepseek_v4") is DEEPSEEK_V4
    assert resolve_family("qwen3.8-27b", "auto") is QWEN3_8
    assert resolve_family("x", "qwen3.6") is QWEN3_6  # alias
    with pytest.raises(ValueError, match="unknown model family"):
        resolve_family("x", "llama9")
    assert set(FAMILIES) == {"qwen3_8", "qwen3_6", "qwen3", "deepseek_v4", "generic"}


@pytest.mark.parametrize("effort, thinking, wire", [
    ("low", True, "low"),
    ("medium", True, "medium"),
    ("high", True, "xhigh"),
    ("xhigh", True, "xhigh"),
    ("max", True, "xhigh"),
    ("minimal", True, "low"),
    (None, True, "none"),       # no effort -> thinking off (CONTEXT tier mapping)
    ("high", False, "none"),
    ("medium", False, "none"),
])
def test_qwen38_effort_mapping(effort, thinking, wire):
    on, eff = QWEN3_8.reasoning_mode(effort, thinking)
    assert eff == wire and on is (wire != "none")
    fields = QWEN3_8.reasoning_fields(on, eff)
    assert fields["reasoning_effort"] == wire
    if wire == "none":
        # portable thinking-off for servers that ignore reasoning_effort (SGLang, llama.cpp)
        assert fields["chat_template_kwargs"] == {"enable_thinking": False}
    else:
        assert "chat_template_kwargs" not in fields  # preserve_thinking stays at the template default


def test_qwen38_never_sends_high_max_or_minimal():
    for effort in list(HARNESS_EFFORTS) + [None, "bogus"]:
        for thinking in (True, False):
            on, eff = QWEN3_8.reasoning_mode(effort, thinking)
            fields = QWEN3_8.reasoning_fields(on, eff)
            assert fields.get("reasoning_effort") in {"xhigh", "medium", "low", "none"}, (effort, thinking, fields)
    # an explicit override is mapped too, never passed through when disallowed
    for override, wire in (("high", "xhigh"), ("max", "xhigh"), ("low", "low"), ("xhigh", "xhigh")):
        on, eff = QWEN3_8.reasoning_mode("medium", True, override=override)
        assert (on, eff) == (True, wire)
    assert QWEN3_8.reasoning_mode("high", True, override="none") == (False, "none")


def test_qwen38_sampling_matches_the_model_card():
    assert QWEN3_8.sampling(True) == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                      "presence_penalty": 0.0, "repetition_penalty": 1.0}
    assert QWEN3_8.sampling(False) == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                                       "presence_penalty": 1.5, "repetition_penalty": 1.0}
    assert QWEN3_8.supports_thinking_budget and QWEN3_8.replay_reasoning_field == "reasoning"
    assert QWEN3_8.max_tool_name_len == 64
    # sampling() returns a copy
    QWEN3_8.sampling(True)["temperature"] = 9
    assert QWEN3_8.sampling(True)["temperature"] == 1.0


def test_qwen36_toggles_thinking_and_preserves_reasoning():
    on, eff = QWEN3_6.reasoning_mode("high", True)
    assert on and eff is None
    assert QWEN3_6.reasoning_fields(on, eff) == {
        "chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": True}}
    on, eff = QWEN3_6.reasoning_mode("high", False)
    assert QWEN3_6.reasoning_fields(on, eff) == {
        "chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": False}}
    assert QWEN3_6.reasoning_mode(None, True)[0] is True  # no effort levels: thinking flag decides
    assert QWEN3_6.sampling(True)["presence_penalty"] == 1.5
    assert QWEN3.reasoning_fields(*QWEN3.reasoning_mode(None, False)) == {
        "chat_template_kwargs": {"enable_thinking": False}}
    assert QWEN3.sampling(True)["temperature"] == 0.6


@pytest.mark.parametrize("effort, wire", [("minimal", "low"), ("low", "low"), ("medium", "low"), ("high", "high"),
                                          ("xhigh", "high"), ("max", "high")])
def test_deepseek_v4_effort_vocabulary(effort, wire):
    on, eff = DEEPSEEK_V4.reasoning_mode(effort, True)
    assert on and eff == wire
    assert DEEPSEEK_V4.reasoning_fields(on, eff) == {"reasoning_effort": wire,
                                                     "chat_template_kwargs": {"thinking": True}}


def test_deepseek_v4_sends_max_only_when_requested_explicitly():
    """P4: the harness 'max' (agents.yaml single-cell-analyst) must not become Think-Max, which needs
    max_tokens >= 128K; an explicit extra['reasoning_effort'] = 'max' still reaches the wire."""
    assert DEEPSEEK_V4.reasoning_mode("max", True) == (True, "high")
    assert "max" not in {DEEPSEEK_V4.map_effort(e) for e in HARNESS_EFFORTS}
    assert DEEPSEEK_V4.reasoning_mode("high", True, override="max") == (True, "max")


def test_deepseek_v4_thinking_off_and_sampling():
    on, eff = DEEPSEEK_V4.reasoning_mode("high", False)
    assert DEEPSEEK_V4.reasoning_fields(on, eff) == {"reasoning_effort": "none",
                                                     "chat_template_kwargs": {"thinking": False}}
    assert DEEPSEEK_V4.sampling(True) == {"temperature": 1.0, "top_p": 0.95}
    assert not DEEPSEEK_V4.supports_thinking_budget
    for effort in HARNESS_EFFORTS:
        assert DEEPSEEK_V4.map_effort(effort) in {"low", "high", "max"}


def test_generic_sends_nothing_unless_effort_is_supported():
    on, eff = GENERIC.reasoning_mode("xhigh", True)
    assert GENERIC.reasoning_fields(on, eff) == {}
    assert GENERIC.sampling(True) == {} and GENERIC.sampling(False) == {}
    on, eff = GENERIC_WITH_EFFORT.reasoning_mode("xhigh", True)
    assert GENERIC_WITH_EFFORT.reasoning_fields(on, eff) == {"reasoning_effort": "high"}
    on, eff = GENERIC_WITH_EFFORT.reasoning_mode("medium", False)
    assert GENERIC_WITH_EFFORT.reasoning_fields(on, eff) == {}
