"""Live probes of the OpenAI-compatible adapter against a REAL inference server.

Skipped unless ``VBT_LIVE_LOCAL_URL`` is set. Recipe (CPU, no GPU needed; see
docs/LOCAL_LLM_VERIFICATION.md):

    scripts/dev/cpu_server.sh --install        # once: vllm-cpu in its own venv
    scripts/dev/cpu_server.sh &                # Qwen/Qwen3.5-0.8B on 127.0.0.1:8011
    VBT_LIVE_LOCAL_URL=http://127.0.0.1:8011/v1 VBT_LIVE_LOCAL_MODEL=qwen3.5-0.8b \\
        python3 -m pytest -q tests/test_live_local.py

Environment:

* ``VBT_LIVE_LOCAL_URL`` - server base URL (``.../v1``).
* ``VBT_LIVE_LOCAL_MODEL`` - served model name (default: the first served model).
* ``VBT_LIVE_LOCAL_PROVIDER`` - registered provider name: ``vllm`` (default),
  ``sglang``, ``llamacpp`` or ``openai_compat``.
* ``VBT_LIVE_LOCAL_FAMILY`` - model family (default ``auto``).
* ``VBT_LIVE_LOCAL_TIMEOUT`` - read timeout in seconds (default 600; CPU is slow).
* ``VBT_LIVE_LOCAL_HARNESS=1`` - also run one CSO turn through the harness
  runtime (several model calls with a large system prompt: minutes on CPU).
* ``VBT_LIVE_LOCAL_LOG`` - file to append one JSON line per probe with raw
  excerpts (used to write the verification report).

Tiny CPU models are weak: the probes check plumbing (request fields the server
accepts, response fields, streaming, tool-call parsing, ordering, usage, error
mapping), not answer quality. Probes that need the model to *decide* to call a
tool use greedy sampling and retry a few times before failing.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pytest

from vbt.providers.base import (
    ContextOverflowError,
    Message,
    ModelSettings,
    ProviderError,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from vbt.providers.openai_compat import OpenAICompatProvider

URL = os.environ.get("VBT_LIVE_LOCAL_URL", "").strip()
pytestmark = pytest.mark.skipif(not URL, reason="set VBT_LIVE_LOCAL_URL to run the live local-server probes")

PROVIDER = os.environ.get("VBT_LIVE_LOCAL_PROVIDER", "vllm").strip() or "vllm"
FAMILY = os.environ.get("VBT_LIVE_LOCAL_FAMILY", "auto").strip() or "auto"
TIMEOUT = float(os.environ.get("VBT_LIVE_LOCAL_TIMEOUT", "600") or 600)
LOG = os.environ.get("VBT_LIVE_LOCAL_LOG", "").strip()
GREEDY = {"temperature": 0.0, "top_p": 1.0}
ATTEMPTS = 3

WEATHER = ToolSpec("get_weather", "Get the current weather for one city.", {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}},
    "required": ["city"],
})
SUBMIT = ToolSpec("submit_result", "Submit the final structured trial annotation.", {
    "type": "object",
    "properties": {
        "nct_id": {"type": "string", "pattern": "^NCT[0-9]{8}$"},
        "outcome": {"type": "string", "enum": ["success", "failure", "unknown"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["nct_id", "outcome", "confidence", "evidence", "notes"],
    "additionalProperties": False,
}, strict=True)


def record(probe: str, **data: Any) -> None:
    if not LOG:
        return
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"probe": probe, "provider": PROVIDER, "t": time.time(), **data}, default=str) + "\n")


class Recording(OpenAICompatProvider):
    """Keeps every request body (to check what was actually sent)."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.bodies: list[dict[str, Any]] = []

    async def _send(self, api, headers, body, **kw):  # noqa: ANN001
        self.bodies.append(json.loads(json.dumps(body)))
        return await super()._send(api, headers, body, **kw)


def make(**kw: Any) -> Recording:
    kw.setdefault("base_url", URL)
    kw.setdefault("family", FAMILY)
    kw.setdefault("read_timeout_s", TIMEOUT)
    kw.setdefault("name", PROVIDER)
    return Recording(**kw)


_MODEL: str | None = None


def served_model() -> str:
    global _MODEL
    if _MODEL is None:
        explicit = os.environ.get("VBT_LIVE_LOCAL_MODEL", "").strip()
        if explicit:
            _MODEL = explicit
        else:
            async def first() -> str:
                p = make()
                try:
                    return str((await p.list_models())[0]["id"])
                finally:
                    await p.aclose()
            _MODEL = asyncio.run(first())
    return _MODEL


def st(*, thinking: bool = False, effort: str | None = None, max_tokens: int = 256, greedy: bool = True,
       **extra: Any) -> ModelSettings:
    ex: dict[str, Any] = {"agent_name": "live-probe", **extra}
    if greedy:
        ex.setdefault("sampling", dict(GREEDY))
    return ModelSettings(PROVIDER, served_model(), max_tokens=max_tokens, effort=effort, thinking=thinking, extra=ex)


def thinking_text(msg: Message) -> str:
    return "".join(b.text for b in msg.content if isinstance(b, ThinkingBlock))


async def until_tool_call(p: OpenAICompatProvider, settings: ModelSettings, messages: list[Message],
                          tools: list[ToolSpec], *, want: int = 1, system: str = "You are a helpful assistant."):
    """The first response with at least ``want`` tool calls (greedy, then sampled retries)."""
    last = None
    for attempt in range(ATTEMPTS):
        s = settings if attempt == 0 else ModelSettings(
            settings.provider, settings.model, settings.max_tokens, settings.effort, settings.thinking,
            None, {k: v for k, v in settings.extra.items() if k != "sampling"})
        last = await p.complete(settings=s, system=system, messages=messages, tools=tools)
        if len(last.message.tool_calls) >= want:
            return last, attempt + 1
    return last, ATTEMPTS


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


async def test_prepare_health_models_and_window():
    p = make()
    model = served_model()
    try:
        await p.prepare(models=[model], wait_s=60)
        assert await p.health() is True
        info = await p.server_info()
        ids = [m["id"] for m in info["models"]]
        record("prepare", info=info)
        assert model in ids
        assert isinstance(info["max_model_len"], int) and info["max_model_len"] > 0
        assert p.context_window(model) == info["max_model_len"]
        if PROVIDER == "vllm":
            assert info["version"], "vLLM serves /version"
        with pytest.raises(ProviderError, match="not served"):
            await p.prepare(models=["no-such-model-xyz"])
    finally:
        await p.aclose()


async def test_plain_completion_with_thinking_off():
    p = make()
    try:
        r = await p.complete(settings=st(max_tokens=64), system="Answer in one word.",
                             messages=[Message.user("Reply with the single word: pong")], tools=[])
        body = p.bodies[-1]
        record("plain", text=r.message.text, stop=r.stop_reason.value, usage=vars(r.usage), request={
            k: body.get(k) for k in ("reasoning_effort", "chat_template_kwargs", "max_tokens", "temperature",
                                     "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")})
        assert r.message.text.strip()
        assert r.stop_reason in (StopReason.END_TURN, StopReason.MAX_TOKENS)
        assert r.usage.input_tokens + r.usage.cache_read_tokens > 0 and r.usage.output_tokens > 0
        assert r.usage.cache_write_tokens == 0 and r.cost_usd == 0.0
        assert not thinking_text(r.message), "thinking off must not produce reasoning"
        assert r.served_model == served_model()
    finally:
        await p.aclose()


async def test_streaming_with_reasoning():
    p = make()
    thoughts: list[str] = []
    texts: list[str] = []
    try:
        r = await p.complete(settings=st(thinking=True, effort="medium", max_tokens=768, thinking_budget=384,
                                         greedy=False),
                             system="You are a careful assistant.",
                             messages=[Message.user("What is 17 * 23? Think briefly, then answer with the number.")],
                             tools=[], on_text=texts.append, on_thinking=thoughts.append)
        body = p.bodies[-1]
        tb = [b for b in r.message.content if isinstance(b, ThinkingBlock)]
        record("stream_reasoning", thinking_chunks=len(thoughts), text_chunks=len(texts),
               reasoning=thinking_text(r.message)[:400], text=r.message.text[:400], stop=r.stop_reason.value,
               usage=vars(r.usage), native=tb[0].native if tb else None,
               request={k: body.get(k) for k in ("reasoning_effort", "chat_template_kwargs", "thinking_token_budget",
                                                 "max_tokens", "temperature", "stream", "stream_options")})
        assert body["stream"] is True
        assert tb and tb[0].text, "thinking on must return reasoning"
        assert tb[0].provider == PROVIDER and tb[0].native["field"] in ("reasoning", "reasoning_content")
        assert len(thoughts) > 1, "reasoning must arrive as several streamed deltas"
        assert "".join(thoughts).strip() == tb[0].text
        if r.stop_reason is not StopReason.MAX_TOKENS:
            assert "".join(texts).strip() == r.message.text
            assert "<think>" not in r.message.text and "</think>" not in r.message.text
    finally:
        await p.aclose()


async def test_single_tool_call_auto():
    p = make()
    try:
        r, attempts = await until_tool_call(
            p, st(max_tokens=256), [Message.user("What is the weather in Paris right now? Use the tool.")], [WEATHER])
        record("tool_call", attempts=attempts, calls=[vars(c) for c in r.message.tool_calls], text=r.message.text,
               stop=r.stop_reason.value, request={k: p.bodies[-1].get(k) for k in ("tool_choice",
                                                                                   "parallel_tool_calls")})
        calls = r.message.tool_calls
        assert calls, f"no tool call in {ATTEMPTS} attempts: {r.message.text[:300]!r}"
        assert r.stop_reason is StopReason.TOOL_USE
        assert calls[0].name == "get_weather" and calls[0].native is None
        assert isinstance(calls[0].input.get("city"), str) and "paris" in calls[0].input["city"].lower()
        assert calls[0].id
        assert p.bodies[-1]["tool_choice"] == "auto" and p.bodies[-1]["parallel_tool_calls"] is True
    finally:
        await p.aclose()


async def test_parallel_tool_calls_in_one_turn():
    p = make()
    try:
        r, attempts = await until_tool_call(
            p, st(max_tokens=384),
            [Message.user("Get the current weather for Paris, Tokyo and Lima. Call get_weather once for EACH city, "
                          "all three calls in this one reply.")], [WEATHER], want=2)
        calls = r.message.tool_calls
        record("parallel", attempts=attempts, calls=[vars(c) for c in calls], stop=r.stop_reason.value)
        assert len(calls) >= 2, f"expected parallel calls, got {[c.input for c in calls]}"
        assert len({c.id for c in calls}) == len(calls)
        assert {c.name for c in calls} == {"get_weather"}
    finally:
        await p.aclose()


async def test_tool_result_round_trip_multi_turn():
    p = make()
    try:
        q = Message.user("What is the weather in Paris and in Tokyo? Use the tool for each city.")
        first, _ = await until_tool_call(p, st(max_tokens=384), [q], [WEATHER])
        calls = first.message.tool_calls
        assert calls, "the model made no tool call"
        weather = {"paris": "Paris: 17 C, light rain.", "tokyo": "Tokyo: 24 C, sunny."}
        # results deliberately in REVERSE order: the adapter must re-order them to the call order
        results = [ToolResult(c.id, weather.get(str(c.input.get("city", "")).lower(), "Sunny, 20 C."))
                   for c in reversed(calls)]
        follow = Message("user", [*results, TextBlock("Now answer the question in one sentence.")])
        r = await p.complete(settings=st(max_tokens=256), system="You are a helpful assistant.",
                             messages=[q, first.message, follow], tools=[WEATHER])
        body = p.bodies[-1]
        roles = [m["role"] for m in body["messages"]]
        record("round_trip", roles=roles, n_calls=len(calls), answer=r.message.text[:400], stop=r.stop_reason.value,
               assistant=body["messages"][2])
        assert roles[:3] == ["system", "user", "assistant"]
        assert roles[3:] == ["tool"] * len(calls) + ["user"]
        assert [m["tool_call_id"] for m in body["messages"][3:3 + len(calls)]] == [c.id for c in calls]
        asst = body["messages"][2]
        assert [tc["function"]["name"] for tc in asst["tool_calls"]] == ["get_weather"] * len(calls)
        assert all(isinstance(json.loads(tc["function"]["arguments"]), dict) for tc in asst["tool_calls"])
        assert r.message.text.strip() or r.message.tool_calls
    finally:
        await p.aclose()


async def test_forced_named_tool_choice_with_strict_schema():
    p = make()
    try:
        r = await p.complete(
            settings=st(max_tokens=384, tool_choice={"name": "submit_result"}),
            system="You annotate clinical trials.",
            messages=[Message.user("Trial NCT01234567 met its primary endpoint (overall survival HR 0.71, p=0.002). "
                                   "Submit the annotation.")],
            tools=[WEATHER, SUBMIT])
        body = p.bodies[-1]
        calls = r.message.tool_calls
        record("forced_strict", calls=[vars(c) for c in calls], stop=r.stop_reason.value,
               tool_choice=body.get("tool_choice"), strict=[t["function"].get("strict") for t in body["tools"]])
        assert body["tool_choice"] == {"type": "function", "function": {"name": "submit_result"}}
        assert [t["function"].get("strict") for t in body["tools"]] == [None, True]
        assert len(calls) >= 1 and calls[0].name == "submit_result", calls
        args = calls[0].input
        assert calls[0].native is None, calls[0].native
        assert re.fullmatch(r"NCT[0-9]{8}", args["nct_id"]) and args["outcome"] in {"success", "failure", "unknown"}
        assert 0 <= float(args["confidence"]) <= 1 and 1 <= len(args["evidence"]) <= 3
        assert set(args) <= set(SUBMIT.input_schema["properties"])
    finally:
        await p.aclose()


async def test_tool_choice_required_and_none():
    p = make()
    try:
        req = await p.complete(settings=st(max_tokens=256, tool_choice="required"), system="Be brief.",
                               messages=[Message.user("Hello! How are you?")], tools=[WEATHER])
        none = await p.complete(settings=st(max_tokens=64, tool_choice="none"), system="Be brief.",
                                messages=[Message.user("What is the weather in Paris? Use the tool.")],
                                tools=[WEATHER])
        record("tool_choice", required=[vars(c) for c in req.message.tool_calls], required_text=req.message.text[:200],
               none_text=none.message.text[:200], none_calls=len(none.message.tool_calls))
        if PROVIDER != "llamacpp":  # llama-server does not enforce 'required' for the Qwen XML tool format
            assert req.message.tool_calls and req.message.tool_calls[0].name == "get_weather"
        assert not none.message.tool_calls
    finally:
        await p.aclose()


async def test_prefix_cache_hits_on_a_repeated_prefix():
    p = make()
    try:
        facts = " ".join(f"Fact {i}: protein P{i} binds ligand L{i * 7 % 101} with affinity {i % 13} nM." for i in
                         range(300))
        system = "You are a biochemistry assistant. Reference sheet follows.\n" + facts
        usages = []
        for q in ("Which ligand does P5 bind?", "Which ligand does P9 bind?", "Which ligand does P12 bind?"):
            r = await p.complete(settings=st(max_tokens=16), system=system, messages=[Message.user(q)], tools=[])
            usages.append(vars(r.usage))
        record("prefix_cache", usages=usages)
        assert usages[0]["input_tokens"] + usages[0]["cache_read_tokens"] > 2000
        assert max(u["cache_read_tokens"] for u in usages[1:]) > 0, usages
        for u in usages:
            assert u["cache_write_tokens"] == 0
    finally:
        await p.aclose()


async def test_prefix_cache_in_a_growing_agent_loop():
    """An agent loop (user -> call -> result -> call -> result -> new user text) is append-only on the
    wire; record how much of each follow-up prompt the server served from its prefix cache. Hybrid
    (Gated DeltaNet) models cache recurrent state only at block-aligned positions ('align' mode), so
    hits are partial; at least one follow-up must hit."""
    p = make()
    try:
        facts = " ".join(f"Entry {i}: gene G{i} maps to locus L{i * 11 % 97}; score {i % 17}." for i in range(150))
        system = "You are a genetics assistant. Index follows.\n" + facts
        tool = ToolSpec("lookup", "Look up a gene.", {"type": "object", "properties": {"gene": {"type": "string"}},
                                                       "required": ["gene"]})
        messages = [Message.user("Use the lookup tool for G5, then for G9.")]
        usages = []
        for step in range(3):
            r = await p.complete(settings=st(max_tokens=48), system=system, messages=messages, tools=[tool])
            usages.append({"prompt": r.usage.input_tokens + r.usage.cache_read_tokens,
                           "cached": r.usage.cache_read_tokens, "calls": len(r.message.tool_calls)})
            messages.append(r.message)
            results = [ToolResult(c.id, f"{c.input.get('gene', '?')} maps to locus L42.")
                       for c in r.message.tool_calls]
            messages.append(Message("user", [*results, TextBlock("Continue.")] if step == 1 else
                                    (results or [TextBlock("Continue.")])))
        record("prefix_cache_loop", usages=usages)
        assert usages[0]["prompt"] > 2000
        assert any(u["cached"] > 0 for u in usages[1:]), usages
    finally:
        await p.aclose()


async def test_context_overflow_without_room_raises():
    p = make()
    try:
        await p.prepare()
        window = p.context_window(served_model())
        assert window
        words = " ".join(f"w{i}" for i in range(int(window * 0.75) + 1000))  # well over the window
        with pytest.raises(ContextOverflowError) as ei:
            await p.complete(settings=st(max_tokens=64), system="s", messages=[Message.user(words)], tools=[])
        record("overflow_no_room", error=str(ei.value)[:600], window=window, learned=p._learned_window)
        assert str(window) in str(ei.value)
        assert p._learned_window == window
    finally:
        await p.aclose()


@pytest.mark.parametrize("slack", [0, 256])
async def test_context_overflow_with_room_retries_with_smaller_max_tokens(slack):
    """max_tokens = window - slack with a ~3K-token prompt: vLLM rejects it (slack 0: character
    pre-check, "upper bound for 0 input tokens"; slack 256: "at least 257 input tokens"), neither
    of which is the prompt size; the adapter counts the prompt and retries once."""
    # auto_discover off and no configured window: the client cannot clamp max_tokens itself, so
    # the server's overflow error must be parsed and the request retried once.
    p = make(auto_discover=False, extra_body={"stop": ["\n"]})
    try:
        probe = make()
        await probe.prepare()
        window = probe.context_window(served_model())
        await probe.aclose()
        filler = " ".join(f"w{i}" for i in range(600))  # ~3K tokens ("w123" is several tokens)
        r = await p.complete(settings=st(max_tokens=window - slack), system="Reply with OK.",
                             messages=[Message.user(filler + "\n\nReply with the single word OK.")], tools=[])
        record("overflow_retry", slack=slack, sent_max_tokens=[b["max_tokens"] for b in p.bodies], window=window,
               learned=p._learned_window, text=r.message.text[:100], stop=r.stop_reason.value, usage=vars(r.usage))
        if PROVIDER == "llamacpp":
            # llama-server does not reject prompt + max_tokens > n_ctx; it generates until the window is full
            assert len(p.bodies) == 1 and r.stop_reason in (StopReason.END_TURN, StopReason.MAX_TOKENS,
                                                            StopReason.CONTEXT_EXCEEDED)
            return
        assert len(p.bodies) == 2, "expected exactly one retry"
        assert p.bodies[0]["max_tokens"] == window - slack
        prompt = r.usage.input_tokens + r.usage.cache_read_tokens
        assert window - prompt - 512 < p.bodies[1]["max_tokens"] <= window - prompt - 256
        assert p._learned_window == window
    finally:
        await p.aclose()


async def test_unknown_model_is_a_clear_provider_error():
    if PROVIDER == "llamacpp":
        pytest.skip("llama-server answers for any model name")
    p = make()
    try:
        bad = ModelSettings(PROVIDER, "no-such-model-xyz", max_tokens=16, effort=None, thinking=False,
                            extra={"agent_name": "live-probe"})
        with pytest.raises(ProviderError) as ei:
            await p.complete(settings=bad, system="s", messages=[Message.user("hi")], tools=[])
        record("unknown_model", error=str(ei.value)[:600])
        assert served_model() in str(ei.value)
    finally:
        await p.aclose()


@pytest.mark.parametrize("effort, thinking", [("xhigh", True), ("medium", True), (None, False)])
async def test_qwen3_8_request_shape_is_accepted_by_the_server(effort, thinking):
    """The production Qwen3.8 dialect (top-level reasoning_effort xhigh/medium/none, enable_thinking,
    thinking_token_budget, card sampling) must pass the server's request validation, whatever the
    served model's template does with it."""
    if PROVIDER not in ("vllm", "sglang"):
        pytest.skip("reasoning_effort dialect is for vLLM/SGLang")
    p = make(family="qwen3_8")
    try:
        r = await p.complete(settings=st(thinking=thinking, effort=effort, max_tokens=128, thinking_budget=32,
                                         greedy=False),
                             system="Be brief.", messages=[Message.user("Say hi.")], tools=[WEATHER])
        body = p.bodies[-1]
        record("qwen3_8_shape", effort=effort, sent={k: body.get(k) for k in (
            "reasoning_effort", "chat_template_kwargs", "thinking_token_budget", "top_k", "min_p",
            "presence_penalty", "repetition_penalty")}, n_requests=len(p.bodies), effort_in_kwargs=p._effort_in_kwargs,
            stop=r.stop_reason.value)
        assert r.stop_reason in (StopReason.END_TURN, StopReason.MAX_TOKENS, StopReason.TOOL_USE)
        wire = body.get("reasoning_effort") or (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
        assert wire == {"xhigh": "xhigh", "medium": "medium", None: "none"}[effort] or (effort is None and wire is None)
    finally:
        await p.aclose()


async def test_concurrent_requests_with_a_client_side_cap():
    p = make(max_concurrency=2)
    try:
        rs = await asyncio.gather(*[
            p.complete(settings=st(max_tokens=8, session_key=f"s{i}"), system="Be brief.",
                       messages=[Message.user(f"Say the number {i}.")], tools=[]) for i in range(4)])
        record("concurrency", texts=[r.message.text for r in rs])
        assert all(r.usage.output_tokens > 0 for r in rs)
    finally:
        await p.aclose()


@pytest.mark.skipif(os.environ.get("VBT_LIVE_LOCAL_HARNESS", "") not in ("1", "true", "yes"),
                    reason="set VBT_LIVE_LOCAL_HARNESS=1 for a full CSO turn through the harness (slow on CPU)")
async def test_harness_cso_turn(tmp_path: Path):
    """One CSO turn through the real runtime: system prompt + tools + Task delegation plumbing."""
    from vbt.config import load_config
    from vbt.orchestrator import open_session

    model = served_model()
    window = int(os.environ.get("VBT_LIVE_LOCAL_WINDOW", "0") or 0)
    if not window:
        probe = make()
        await probe.prepare()
        window = probe.context_window(model) or 32768
        await probe.aclose()
    tier = {"model": model, "context_window_tokens": window}
    config = load_config(["local-h100"], overrides={
        "provider": {"name": PROVIDER, "options": {"base_url": URL, "served_model_name": model, "family": FAMILY,
                                                    "read_timeout_s": TIMEOUT, "max_concurrency": 2}},
        "models": {
            "orchestrator": {**tier, "effort": "medium", "thinking": False, "max_tokens": 1024},
            "scientist": {**tier, "effort": "medium", "thinking": False, "max_tokens": 1024},
            "support": {**tier, "effort": None, "thinking": False, "max_tokens": 1024},
            "bulk": {**tier, "effort": None, "thinking": False, "max_tokens": 1024},
        },
        "paths": {"runs_dir": str(tmp_path / "runs")},
        "limits": {"max_cso_turns": 6, "max_specialist_turns": 4, "max_parallel_agents": 1},
        "orchestration": {"strategic_orientation": False, "enforce_review": False},
        "preflight": {"skip": True},
        "web": {"enabled": False},
        "context": {"default_window_tokens": window},
    })
    events: list[tuple[str, dict[str, Any]]] = []
    session = await open_session(config, start_mcp=False, on_event=lambda kind, data: events.append((kind, data)))
    t0 = time.time()
    try:
        out = await session.ask("Delegate to exactly one specialist: ask the genomics-analyst for one sentence on "
                                "whether PCSK9 is a genetically supported target for LDL cholesterol. Then "
                                "summarise its answer in one sentence.")
    finally:
        await session.close()
    kinds: dict[str, int] = {}
    for kind, _ in events:
        kinds[kind] = kinds.get(kind, 0) + 1
    tools_used = [d.get("tool") for k, d in events if k == "tool_start"]
    ends = [(d.get("agent"), d.get("status")) for k, d in events if k == "agent_end"]
    turn_end = [d for k, d in events if k == "turn_end"]
    record("harness_turn", seconds=round(time.time() - t0, 1), answer=str(out)[:800], events=kinds,
           tools=tools_used, agent_ends=ends, turn_status=[d.get("status") for d in turn_end],
           run_dir=str(session.run.dir) if getattr(session, "run", None) is not None else None)
    # A tiny model may wander until a turn limit ("incomplete: turn_limit"); the plumbing must not fail.
    assert isinstance(out, str) and out.strip()
    assert turn_end, kinds
    status = str(turn_end[-1].get("status"))
    assert "error" not in status and "fail" not in status, (status, ends)
    assert kinds.get("message_end", 0) >= 1 and any(agent == "cso" for agent, _ in ends)
