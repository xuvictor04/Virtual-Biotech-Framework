"""OpenAI-compatible local provider (vLLM / SGLang / llama.cpp) against a fake server.

Offline: a threaded ``http.server`` plays the vLLM OpenAI API (``/v1/chat/
completions`` with SSE or JSON, ``/v1/models``, ``/health``, ``/version``).
"""

import asyncio
import base64
import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vbt.providers import create_provider
from vbt.providers.base import (
    ContextOverflowError,
    ImagePart,
    Message,
    ModelSettings,
    OpaqueBlock,
    ProviderError,
    RetryableProviderError,
    StopReason,
    SystemSegment,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from vbt.providers.openai_compat import (
    OpenAICompatProvider,
    extract_text_tool_calls,
    parse_overflow,
    stable_hash,
    tool_alias,
)
from vbt.providers.retry import RetryPolicy, complete_with_retry

MODEL = "qwen3.8-27b"


# ---------------------------------------------------------------------------
# Fake server
# ---------------------------------------------------------------------------


def chunk(delta=None, finish=None, usage=None, model=MODEL):
    c = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model, "choices": []}
    if delta is not None or finish is not None:
        c["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        c["usage"] = usage
    return c


def usage(prompt=100, completion=20, cached=0):
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached}}


def text_turn(text="Done.", finish="stop", u=None):
    return [chunk({"role": "assistant", "content": ""}), chunk({"content": text}), chunk(finish=finish),
            chunk(usage=u or usage())]


def tc_delta(index, id=None, name=None, args=None):
    d = {"index": index}
    if id:
        d["id"] = id
        d["type"] = "function"
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    d["function"] = fn
    return chunk({"tool_calls": [d]})


def sse(chunks, done=True, comments=False):
    out = ""
    for c in chunks:
        if comments:
            out += ": ping\n\n"
        out += f"data: {json.dumps(c)}\n\n"
    if done:
        out += "data: [DONE]\n\n"
    return out.encode()


def api_error(status, message, etype="BadRequestError", headers=None):
    return ("json", status, {"error": {"message": message, "type": etype, "param": None, "code": status}},
            headers or {})


class Fake:
    def __init__(self):
        self.requests = []
        self.responses = deque()
        self.models = [{"id": MODEL, "object": "model", "owned_by": "vllm", "root": "Qwen/Qwen3.8-27B-FP8",
                        "max_model_len": 262144}]
        self.health_status = 200
        self.delay = 0.0
        self.inflight = 0
        self.max_inflight = 0
        self.lock = threading.Lock()
        self.url = ""
        self.tokenize_count = None  # POST /tokenize answer ({"count": N}); None -> 404 (no such endpoint)
        self.tokenize_requests = []  # kept apart from `requests` so `posts` stays chat completions only

    @property
    def posts(self):
        return [r for r in self.requests if r["method"] == "POST"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status, payload, ctype="application/json", headers=None, length=None):
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(length if length is not None else len(payload)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def do_GET(self):  # noqa: N802
        fake = self.server.fake
        fake.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
        if self.path == "/health":
            self._send(fake.health_status, b"")
        elif self.path == "/v1/models":
            self._send(200, json.dumps({"object": "list", "data": fake.models}).encode())
        elif self.path == "/version":
            self._send(200, json.dumps({"version": "0.31.0"}).encode())
        else:
            self._send(404, json.dumps({"detail": "Not Found"}).encode())

    def do_POST(self):  # noqa: N802
        fake = self.server.fake
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path == "/tokenize":
            fake.tokenize_requests.append(body)
            if fake.tokenize_count is None:
                self._send(404, json.dumps({"detail": "Not Found"}).encode())
            else:
                self._send(200, json.dumps({"count": fake.tokenize_count, "max_model_len": 32768,
                                            "tokens": [1] * fake.tokenize_count}).encode())
            return
        fake.requests.append({"method": "POST", "path": self.path, "headers": dict(self.headers), "body": body})
        with fake.lock:
            fake.inflight += 1
            fake.max_inflight = max(fake.max_inflight, fake.inflight)
        try:
            if fake.delay:
                time.sleep(fake.delay)
            spec = fake.responses.popleft() if fake.responses else ("sse", text_turn())
            kind = spec[0]
            if kind == "sse":
                self._send(200, sse(spec[1]), "text/event-stream")
            elif kind == "sse_comments":
                self._send(200, sse(spec[1], comments=True), "text/event-stream")
            elif kind == "sse_raw":
                self._send(200, spec[1], "text/event-stream")
            elif kind == "sse_nodone":
                self._send(200, sse(spec[1], done=False), "text/event-stream")
            elif kind == "drop":  # promise more bytes than we send, then close
                payload = sse(spec[1], done=False)
                self._send(200, payload, "text/event-stream", length=len(payload) + 5000)
                self.close_connection = True
            elif kind == "json":
                _, status, obj, headers = spec
                self._send(status, json.dumps(obj).encode(), "application/json", headers)
            else:
                raise AssertionError(kind)
        finally:
            with fake.lock:
                fake.inflight -= 1

    def log_message(self, *a):
        pass


def _clear_env(monkeypatch):
    for v in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy",
              "VBT_LLM_BASE_URL", "VBT_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


@pytest.fixture
def fake(monkeypatch):
    _clear_env(monkeypatch)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    srv.fake = Fake()
    srv.fake.url = f"http://127.0.0.1:{srv.server_port}"
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield srv.fake
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def make(fake):
    made = []

    def make(**kw):
        kw.setdefault("base_url", fake.url + "/v1")
        p = OpenAICompatProvider(**kw)
        made.append(p)
        return p

    yield make


def settings(effort="medium", thinking=True, max_tokens=32768, temperature=None, **extra):
    return ModelSettings("vllm", MODEL, max_tokens=max_tokens, effort=effort, thinking=thinking,
                         temperature=temperature, extra={"agent_name": "scientist", **extra})


Q = [Message.user("EGFR in NSCLC?")]
TOOLS = [ToolSpec("mcp__genetics__query_l2g", "L2G", {"type": "object", "properties": {"gene": {"type": "string"}}}),
         ToolSpec("Read", "read", {"type": "object", "properties": {"file_path": {"type": "string"}}}),
         ToolSpec("WebSearch", "search", {"type": "object"})]


def offline(**kw):
    """A provider for pure request-building tests (never connects)."""
    kw.setdefault("base_url", "http://127.0.0.1:9/v1")
    return OpenAICompatProvider(**kw)


# ---------------------------------------------------------------------------
# Request encoding (pure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier, s, effort, budget, max_tokens, presence", [
    ("orchestrator", settings("high", True, 40960, thinking_budget=24576), "xhigh", 24576, 40960, 0.0),
    ("scientist", settings("medium", True, 32768, thinking_budget=8192), "medium", 8192, 32768, 0.0),
    ("support", settings(None, False, 16000), "none", None, 16000, 1.5),
    ("bulk", settings("medium", True, 8192, thinking_budget=3072), "medium", 3072, 8192, 0.0),
])
def test_tier_mapping_request_fields(tier, s, effort, budget, max_tokens, presence):
    body = offline()._request(s, "sys", Q, TOOLS)
    assert body["reasoning_effort"] == effort
    assert body.get("thinking_token_budget") == budget
    assert body["max_tokens"] == max_tokens
    assert body["presence_penalty"] == presence and body["top_k"] == 20 and body["min_p"] == 0.0
    if effort == "none":
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert (body["temperature"], body["top_p"]) == (0.7, 0.8)
    else:
        assert "chat_template_kwargs" not in body
        assert (body["temperature"], body["top_p"], body["repetition_penalty"]) == (1.0, 0.95, 1.0)
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["parallel_tool_calls"] is True and body["tool_choice"] == "auto"
    assert body["model"] == MODEL


def test_never_sends_high_max_or_minimal_to_qwen38():
    p = offline()
    for effort in ("minimal", "low", "medium", "high", "xhigh", "max", None):
        for thinking in (True, False):
            body = p._request(settings(effort, thinking), "s", Q, [])
            assert body["reasoning_effort"] in {"xhigh", "medium", "low", "none"}
    body = p._request(settings("medium", reasoning_effort="max"), "s", Q, [])
    assert body["reasoning_effort"] == "xhigh"


def test_thinking_budget_defaults_and_bounds():
    p = offline()
    assert p._request(settings(max_tokens=8192), "s", Q, [])["thinking_token_budget"] == 4096
    assert p._request(settings(max_tokens=40960), "s", Q, [])["thinking_token_budget"] == 16384
    assert p._request(settings(max_tokens=4000, thinking_budget=9000), "s", Q, [])["thinking_token_budget"] == 2000
    assert "thinking_token_budget" not in p._request(settings(thinking=False), "s", Q, [])
    # deepseek_v4 has no thinking budget
    ds = offline(family="deepseek_v4")
    body = ds._request(settings("max"), "s", Q, [])
    assert "thinking_token_budget" not in body and body["reasoning_effort"] == "max"
    assert body["chat_template_kwargs"] == {"thinking": True} and "top_k" not in body


def test_sampling_overrides_and_temperature():
    p = offline()
    body = p._request(settings(temperature=0.3, sampling={"presence_penalty": 0.5, "top_k": None, "seed": 7}),
                      "s", Q, [])
    assert body["temperature"] == 0.3 and body["presence_penalty"] == 0.5 and body["seed"] == 7
    assert "top_k" not in body
    body = p._request(settings(sampling={"top_k": -1}), "s", Q, [])
    assert "top_k" not in body  # never top_k=-1


def test_strict_tools_and_forced_tool_choice():
    tools = TOOLS + [ToolSpec("submit_result", "submit", {"type": "object", "properties": {}}, strict=True)]
    p = offline()
    body = p._request(settings(tool_choice={"name": "submit_result"}), "s", Q, tools)
    assert body["tool_choice"] == {"type": "function", "function": {"name": "submit_result"}}
    fns = {t["function"]["name"]: t["function"] for t in body["tools"]}
    assert fns["submit_result"]["strict"] is True and "strict" not in fns["Read"]
    assert all(t["type"] == "function" for t in body["tools"])
    assert fns["Read"]["parameters"] == TOOLS[1].input_schema
    assert p._request(settings(tool_choice="required"), "s", Q, tools)["tool_choice"] == "required"
    assert p._request(settings(force_tool="submit_result"), "s", Q, tools)["tool_choice"]["function"]["name"] == \
        "submit_result"
    with pytest.raises(ProviderError, match="not among"):
        p._request(settings(tool_choice={"name": "nope"}), "s", Q, tools)
    body = p._request(settings(tool_choice={"name": "submit_result"}), "s", Q, [])
    assert "tools" not in body and "tool_choice" not in body and "parallel_tool_calls" not in body


def test_extra_keys_are_never_forwarded_and_extra_body_merges_last():
    p = offline(extra_body={"chat_template_kwargs": {"custom": 1}, "seed": 3, "top_p": 0.9})
    body = p._request(settings(thinking=False, session_key="inv-1", foo="bar", prompt_cache=False,
                               context_window_tokens=100000, context_management=True), "s", Q, [])
    for k in ("agent_name", "session_key", "foo", "prompt_cache", "context_window_tokens", "context_management",
              "thinking_budget", "sampling", "tool_choice"):
        assert k not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "custom": 1}
    assert body["seed"] == 3 and body["top_p"] == 0.9


def test_served_model_name_overrides_settings_model():
    body = offline(served_model_name="my-qwen", family="qwen3_8")._request(settings(), "s", Q, [])
    assert body["model"] == "my-qwen" and body["reasoning_effort"] == "medium"


def test_message_encoding_order_single_system_and_reasoning_replay():
    calls = [ToolCall("c1", "mcp__genetics__query_l2g", {"gene": "EGFR"}), ToolCall("c2", "Read", {"file_path": "a"}),
             ToolCall("c3", "WebSearch", {"q": "x"})]
    history = [
        Message.user("EGFR?"),
        Message("assistant", [ThinkingBlock("plan the lookups", provider="vllm", native={"field": "reasoning"}),
                              *calls]),
        Message("user", [ToolResult("c3", "web"), ToolResult("c1", "l2g"),
                         ToolResult("c2", "boom", is_error=True), TextBlock("[Harness] reminder")]),
        Message("assistant", [ThinkingBlock("anthropic thought", provider="anthropic", native={"signature": "s"}),
                              OpaqueBlock("anthropic", {"type": "server_tool_use"}),
                              ThinkingBlock("compat thought", provider="openai_compat",
                                            native={"field": "reasoning", "family": "qwen3_8"}),
                              TextBlock("Summary so far.")]),
        Message.user("Continue."),
    ]
    system = [SystemSegment("static prompt", cache=True), SystemSegment("date: today")]
    body = offline()._request(settings(), system, history, TOOLS)
    msgs = body["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "tool", "tool", "user", "assistant",
                                         "user"]
    assert msgs[0]["content"] == "static prompt\n\ndate: today"
    assert sum(m["role"] == "system" for m in msgs) == 1
    a1 = msgs[2]
    assert a1["content"] is None and a1["reasoning"] == "plan the lookups"
    assert [tc["id"] for tc in a1["tool_calls"]] == ["c1", "c2", "c3"]
    assert json.loads(a1["tool_calls"][0]["function"]["arguments"]) == {"gene": "EGFR"}
    assert isinstance(a1["tool_calls"][0]["function"]["arguments"], str)
    # tool messages in the order of the assistant's calls, then the trailing text as a user message
    assert [m["tool_call_id"] for m in msgs[3:6]] == ["c1", "c2", "c3"]
    assert [m["content"] for m in msgs[3:6]] == ["l2g", "Error: boom", "web"]
    assert msgs[6] == {"role": "user", "content": "[Harness] reminder"}
    # foreign (Anthropic) thinking and opaque blocks dropped; family-compatible reasoning replayed
    assert msgs[7]["reasoning"] == "compat thought" and msgs[7]["content"] == "Summary so far."
    assert "anthropic thought" not in json.dumps(body)


def test_conversation_without_user_message_is_an_adapter_error():
    p = offline()
    with pytest.raises(ProviderError, match="no user message"):
        p._request(settings(), "sys", [], [])
    with pytest.raises(ProviderError, match="no user message"):
        p._request(settings(), "sys", [Message("assistant", [TextBlock("hi")])], [])


def test_images_go_to_a_user_message_only_with_vision():
    png = base64.b64encode(b"\x89PNG fake").decode()
    history = [Message.user("Check the UMAP"),
               Message("assistant", [ToolCall("t0", "Read", {"file_path": "umap.png"})]),
               Message("user", [ToolResult("t0", [TextBlock("umap.png"), ImagePart("image/png", png, "umap.png")])])]
    body = offline()._request(settings(), "s", history, TOOLS)
    assert body["messages"][-1] == {"role": "tool", "tool_call_id": "t0", "content": "umap.png\n[image: umap.png]"}
    assert "image_url" not in json.dumps(body)
    body = offline(vision=True)._request(settings(), "s", history, TOOLS)
    tool_msg, user_msg = body["messages"][-2:]
    assert tool_msg["role"] == "tool" and isinstance(tool_msg["content"], str)
    assert user_msg["role"] == "user"
    assert user_msg["content"][1] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}
    assert offline(vision=True).capabilities(MODEL).images and not offline().capabilities(MODEL).images


def test_tool_alias_is_stable_and_valid():
    long = "mcp__clinicaltrials__" + "x" * 60
    a = tool_alias(long)
    assert a == tool_alias(long) and len(a) <= 64 and a != long
    assert tool_alias("ns.tool") != "ns.tool" and tool_alias("ns.tool").startswith("ns_tool_")
    assert tool_alias("mcp__clinicaltrials__get_clinical_trial_details") == \
        "mcp__clinicaltrials__get_clinical_trial_details"
    import re
    for n in (long, "ns.tool", "a b/c", ""):
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool_alias(n))


def test_max_tokens_clamped_to_the_known_window():
    p = offline(context_window=20000)
    body = p._request(settings(max_tokens=32768), "s", [Message.user("x" * 40000)], [])
    assert 0 < body["max_tokens"] < 20000 - 10000
    assert body["thinking_token_budget"] < body["max_tokens"]
    huge = p._request(settings(max_tokens=32768), "s", [Message.user("x" * 400000)], [])
    assert huge["max_tokens"] == 1024  # never <= 0: the server reports the real overflow
    assert offline()._request(settings(max_tokens=32768), "s", Q, [])["max_tokens"] == 32768  # window unknown


def test_check_credentials_and_registration(monkeypatch):
    _clear_env(monkeypatch)
    p = OpenAICompatProvider()
    assert "VBT_LLM_BASE_URL" in p.check_credentials()
    monkeypatch.setenv("VBT_LLM_BASE_URL", "http://gpu-box:8000/v1")
    assert OpenAICompatProvider().check_credentials() is None
    assert "invalid base_url" in OpenAICompatProvider(base_url="gpu-box:8000").check_credentials()
    from vbt.providers.openai_compat import create
    assert create("sglang", base_url="http://h:8000/v1", name="ignored").name == "sglang"
    for name in ("vllm", "sglang", "openai_compat", "llamacpp"):
        prov = create_provider(name, base_url="http://h:8000/v1")
        assert isinstance(prov, OpenAICompatProvider) and prov.name == name
        caps = prov.capabilities(MODEL)
        assert caps.tool_choice and caps.replays_reasoning and not caps.web_search and not caps.documents
    p = OpenAICompatProvider(base_url="http://h:8000")
    assert p.api_bases == ["http://h:8000/v1"]
    with pytest.raises(ProviderError, match="unknown model family"):
        OpenAICompatProvider(base_url="http://h:8000/v1", family="nope")


# ---------------------------------------------------------------------------
# Streaming decode
# ---------------------------------------------------------------------------


def _parallel_turn():
    return [
        chunk({"role": "assistant", "content": ""}),
        chunk({"reasoning": "Weigh L2G "}),
        chunk({"reasoning": "vs coloc."}),
        chunk({"content": "Checking three sources."}),
        tc_delta(0, id="call_a", name="mcp__genetics__query_l2g", args=""),
        tc_delta(1, id="call_b", name="Read", args=""),
        tc_delta(0, args='{"gene": '),
        tc_delta(2, id="call_c", name="WebSearch", args='{"q"'),
        tc_delta(1, args='{"file_path": "a.txt"}'),
        tc_delta(0, args='"EGFR"}'),
        tc_delta(2, args=': "EGFR trials"}'),
        chunk(finish="tool_calls"),
        chunk(usage=usage(1200, 80, cached=1024)),
    ]


async def test_stream_decodes_text_reasoning_parallel_calls_and_usage(fake, make):
    fake.responses.append(("sse_comments", _parallel_turn()))
    p = make()
    texts, thoughts = [], []
    resp = await p.complete(settings=settings(), system="sys", messages=Q, tools=TOOLS, on_text=texts.append,
                            on_thinking=thoughts.append)
    await p.aclose()
    assert "".join(thoughts) == "Weigh L2G vs coloc." and "".join(texts) == "Checking three sources."
    content = resp.message.content
    assert isinstance(content[0], ThinkingBlock) and content[0].text == "Weigh L2G vs coloc."
    assert content[0].provider == "vllm" and content[0].native["field"] == "reasoning"
    assert resp.message.text == "Checking three sources."
    calls = resp.message.tool_calls
    assert [(c.id, c.name, c.input) for c in calls] == [
        ("call_a", "mcp__genetics__query_l2g", {"gene": "EGFR"}),
        ("call_b", "Read", {"file_path": "a.txt"}),
        ("call_c", "WebSearch", {"q": "EGFR trials"})]
    assert resp.stop_reason is StopReason.TOOL_USE
    u = resp.usage
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens, u.cache_write_tokens) == (176, 1024, 80, 0)
    assert resp.cost_usd == 0.0 and resp.served_model == MODEL and resp.request_id == "chatcmpl-1"
    post = fake.posts[-1]
    assert post["path"] == "/v1/chat/completions" and post["body"]["stream"] is True
    # model discovery ran once before the first call
    assert [r["path"] for r in fake.requests if r["method"] == "GET"] == ["/v1/models"]


async def test_reasoning_replays_on_the_next_request(fake, make):
    fake.responses.append(("sse", _parallel_turn()))
    fake.responses.append(("sse", text_turn("Final.")))
    p = make()
    r1 = await p.complete(settings=settings(), system="s", messages=Q, tools=TOOLS)
    history = Q + [r1.message, Message("user", [ToolResult(c.id, "ok") for c in r1.message.tool_calls])]
    await p.complete(settings=settings(), system="s", messages=history, tools=TOOLS)
    await p.aclose()
    sent = fake.posts[-1]["body"]["messages"]
    assistant = [m for m in sent if m["role"] == "assistant"][0]
    assert assistant["reasoning"] == "Weigh L2G vs coloc." and assistant["content"] == "Checking three sources."
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "tool", "tool"]


async def test_sglang_reasoning_content_field(fake, make):
    fake.responses.append(("sse", [chunk({"reasoning_content": "Think "}), chunk({"reasoning_content": "hard."}),
                                   chunk({"content": "Answer."}), chunk(finish="stop"), chunk(usage=usage())]))
    p = make(name="sglang")
    thoughts = []
    resp = await p.complete(settings=settings(), system="s", messages=Q, tools=[], on_thinking=thoughts.append)
    assert "".join(thoughts) == "Think hard." and resp.message.content[0].text == "Think hard."
    assert resp.message.content[0].native["field"] == "reasoning_content"
    assert resp.stop_reason is StopReason.END_TURN
    body = p._request(settings(), "s", Q + [resp.message, Message.user("more")], [])
    assert body["messages"][2]["reasoning_content"] == "Think hard." and "reasoning" not in body["messages"][2]
    await p.aclose()


async def test_vllm_duplicate_reasoning_fields_are_not_doubled(fake, make):
    fake.responses.append(("sse", [chunk({"reasoning": "abc", "reasoning_content": "abc"}),
                                   chunk({"content": "x"}), chunk(finish="stop")]))
    p = make()
    resp = await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    await p.aclose()
    assert resp.message.content[0].text == "abc"


async def test_non_stream_json_response(fake, make):
    fake.responses.append(("json", 200, {
        "id": "chatcmpl-9", "object": "chat.completion", "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "Looking it up.", "reasoning": "need data",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "Read", "arguments": "{\"file_path\": \"x\"}"}},
                           {"id": "call_2", "type": "function",
                            "function": {"name": "WebSearch", "arguments": "{}"}}]}}],
        "usage": usage(50, 10, cached=32)}, {}))
    p = make()
    texts = []
    resp = await p.complete(settings=settings(), system="s", messages=Q, tools=TOOLS, on_text=texts.append)
    await p.aclose()
    assert texts == ["Looking it up."] and resp.request_id == "chatcmpl-9"
    assert [(c.id, c.name, c.input) for c in resp.message.tool_calls] == [("call_1", "Read", {"file_path": "x"}),
                                                                          ("call_2", "WebSearch", {})]
    assert resp.stop_reason is StopReason.TOOL_USE and resp.usage.cache_read_tokens == 32
    assert resp.message.content[0].text == "need data"


async def test_invalid_arguments_are_kept_raw(fake, make):
    fake.responses.append(("sse", [tc_delta(0, id="c1", name="Read", args='{"file_path": "a'),
                                   tc_delta(1, id="c2", name="WebSearch", args="[1, 2]"),
                                   chunk(finish="tool_calls")]))
    p = make()
    resp = await p.complete(settings=settings(), system="s", messages=Q, tools=TOOLS)
    await p.aclose()
    c1, c2 = resp.message.tool_calls
    assert c1.input == {} and c1.native["invalid_arguments"] == '{"file_path": "a'
    assert "invalid JSON" in c1.native["error"]
    assert c2.input == {} and "JSON object" in c2.native["error"]
    assert resp.stop_reason is StopReason.TOOL_USE
    # replayed as an empty JSON object (vLLM would coerce it to {} anyway)
    body = p._request(settings(), "s", Q + [resp.message, Message("user", [ToolResult("c1", "bad", True),
                                                                           ToolResult("c2", "bad", True)])], TOOLS)
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == "{}"


async def test_length_finish_maps_to_max_tokens_or_context_exceeded(fake, make):
    p = make()
    await p.prepare()  # max_model_len 262144
    fake.responses.append(("sse", text_turn("partial", finish="length", u=usage(1000, 500))))
    r = await p.complete(settings=settings(max_tokens=500), system="s", messages=Q, tools=[])
    assert r.stop_reason is StopReason.MAX_TOKENS
    fake.responses.append(("sse", text_turn("partial", finish="length", u=usage(260000, 2144))))
    r = await p.complete(settings=settings(max_tokens=500), system="s", messages=Q, tools=[])
    assert r.stop_reason is StopReason.CONTEXT_EXCEEDED
    # truncated tool call: MAX_TOKENS, so the runtime reports the truncated input
    fake.responses.append(("sse", [tc_delta(0, id="c1", name="Read", args='{"file'), chunk(finish="length"),
                                   chunk(usage=usage(100, 500))]))
    r = await p.complete(settings=settings(max_tokens=500), system="s", messages=Q, tools=TOOLS)
    assert r.stop_reason is StopReason.MAX_TOKENS and r.message.tool_calls
    await p.aclose()


async def test_text_tool_call_fallback_json_and_xml(fake, make):
    tools = TOOLS + [ToolSpec("mcp__x__query", "q", {"type": "object", "properties": {
        "gene": {"type": "string"}, "limit": {"type": "integer"}, "flags": {"type": "array"}}})]
    fake.responses.append(("sse", text_turn(
        'Let me check.\n<tool_call>\n{"name": "Read", "arguments": {"file_path": "a.txt"}}\n</tool_call>')))
    fake.responses.append(("sse", text_turn(
        "<tool_call>\n<function=mcp__x__query>\n<parameter=gene>\n123\n</parameter>\n<parameter=limit>\n5\n"
        "</parameter>\n<parameter=flags>\n[\"a\"]\n</parameter>\n</function>\n</tool_call>")))
    p = make()
    r1 = await p.complete(settings=settings(), system="s", messages=Q, tools=tools)
    assert r1.message.text == "Let me check." and r1.stop_reason is StopReason.TOOL_USE
    assert [(c.name, c.input) for c in r1.message.tool_calls] == [("Read", {"file_path": "a.txt"})]
    r2 = await p.complete(settings=settings(), system="s", messages=Q, tools=tools)
    assert [(c.name, c.input) for c in r2.message.tool_calls] == [
        ("mcp__x__query", {"gene": "123", "limit": 5, "flags": ["a"]})]
    assert r2.message.text == ""
    # without tools the text is left alone
    fake.responses.append(("sse", text_turn("<tool_call>{\"name\": \"Read\", \"arguments\": {}}</tool_call>")))
    r3 = await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    assert not r3.message.tool_calls and "<tool_call>" in r3.message.text
    await p.aclose()


def test_extract_text_tool_calls_unit():
    calls, rest = extract_text_tool_calls("a <tool_call>{\"name\": \"Read\", \"arguments\": \"{\\\"x\\\": 1}\"}"
                                          "</tool_call> b <tool_call>garbage</tool_call>", TOOLS)
    assert [(c.name, c.input) for c in calls] == [("Read", {"x": 1})]
    assert rest == "a  b <tool_call>garbage</tool_call>"
    calls, _ = extract_text_tool_calls("<tool_call>{\"name\": \"Read\", \"arguments\": {\"x\": 1</tool_call>")
    assert calls[0].name == "Read" and calls[0].native["invalid_arguments"]
    calls, rest = extract_text_tool_calls("<function=Read>\n<parameter=file_path>\na b\n</parameter>\n</function>",
                                          TOOLS)
    assert [(c.name, c.input) for c in calls] == [("Read", {"file_path": "a b"})] and rest == ""


async def test_think_tags_in_content_become_reasoning(fake, make):
    fake.responses.append(("sse", text_turn("<think>\nconsider\n</think>\n\nAnswer.")))
    p = make()
    r = await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    await p.aclose()
    assert r.message.content[0].text == "consider" and r.message.text == "Answer."


async def test_stray_think_tag_with_thinking_off_stays_answer_text(fake, make):
    """Live llama.cpp + Qwen3.5-0.8B, thinking off: the model wrote 'I'm doing well ...</think>...'; the
    first sentence was moved into a hidden ThinkingBlock. With thinking off it is answer text."""
    content = "I'm doing well, thanks!</think>\n\nHow can I help?"
    fake.responses.append(("sse", text_turn(content)))
    p = make()
    r = await p.complete(settings=settings(None, False, 1024), system="s", messages=Q, tools=[])
    assert not [b for b in r.message.content if isinstance(b, ThinkingBlock)]
    assert r.message.text == content
    fake.responses.append(("sse", text_turn("<think>\nplan\n</think>\n\nAnswer.")))  # explicit tags still split
    r = await p.complete(settings=settings(None, False, 1024), system="s", messages=Q, tools=[])
    assert r.message.content[0].text == "plan" and r.message.text == "Answer."
    await p.aclose()


async def test_tool_choice_none_never_extracts_text_tool_calls(fake, make):
    """Live llama.cpp: with tool_choice 'none' the model's attempted call comes back as content text."""
    xml = "<tool_call>\n<function=Read>\n<parameter=file_path>\n/x\n</parameter>\n</function>\n</tool_call>"
    fake.responses.append(("sse", text_turn(xml)))
    p = make()
    r = await p.complete(settings=settings(tool_choice="none"), system="s", messages=Q, tools=TOOLS)
    assert fake.posts[-1]["body"]["tool_choice"] == "none"
    assert not r.message.tool_calls and r.message.text == xml and r.stop_reason is StopReason.END_TURN
    await p.aclose()


async def test_tool_alias_round_trip(fake, make):
    long = "mcp__clinicaltrials__" + "y" * 60
    tools = [ToolSpec(long, "long tool", {"type": "object"}), ToolSpec("ns.tool", "dotted", {"type": "object"})]
    a_long, a_dot = tool_alias(long), tool_alias("ns.tool")
    fake.responses.append(("sse", [tc_delta(0, id="c1", name=a_long, args="{}"),
                                   tc_delta(1, id="c2", name=a_dot, args='{"k": 1}'), chunk(finish="tool_calls")]))
    p = make()
    r = await p.complete(settings=settings(), system="s", messages=Q, tools=tools)
    sent = fake.posts[-1]["body"]
    assert [t["function"]["name"] for t in sent["tools"]] == [a_long, a_dot]
    assert sent["tools"][1]["function"]["description"].startswith("[harness tool ns.tool]")
    assert [(c.name, c.input) for c in r.message.tool_calls] == [(long, {}), ("ns.tool", {"k": 1})]
    body = p._request(settings(), "s", Q + [r.message, Message("user", [ToolResult("c1", "x"),
                                                                        ToolResult("c2", "y")])], tools)
    assert [tc["function"]["name"] for tc in body["messages"][2]["tool_calls"]] == [a_long, a_dot]
    await p.aclose()


# ---------------------------------------------------------------------------
# Errors, overflow, retries
# ---------------------------------------------------------------------------

VLLM_OVERFLOW = ("This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens "
                 "and your prompt contains 20000 input tokens, for a total of 52768 tokens. Please reduce the length "
                 "of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=20000)")


@pytest.mark.parametrize("message, err, expected", [
    (VLLM_OVERFLOW, None, (32768, 20000, 32768)),
    ("'max_tokens' or 'max_completion_tokens' is too large: 40960. This model's maximum context length is 262144 "
     "tokens and your request has 230000 input tokens (40960 > 262144 - 230000).", None, (262144, 230000, 40960)),
    ("This model's maximum context length is 32768 tokens. However, your request has 40000 input tokens. Please "
     "reduce the length of the input messages.", None, (32768, 40000, None)),
    ("Requested token count exceeds the model's maximum context length of 32768 tokens. You requested a total of "
     "40000 tokens: 30000 tokens from the input messages and 10000 tokens for the completion.", None,
     (32768, 30000, 10000)),
    ("The input (40000 tokens) is longer than the model's context length (32768 tokens).", None,
     (32768, 40000, None)),
    ("the request exceeds the available context size, try increasing it", {"n_ctx": 4096, "n_prompt_tokens": 5000},
     (4096, 5000, None)),
])
def test_parse_overflow_formats(message, err, expected):
    assert parse_overflow(message, err) == expected


async def test_overflow_retries_once_with_smaller_max_tokens(fake, make):
    fake.responses.append(api_error(400, VLLM_OVERFLOW))
    fake.responses.append(("sse", text_turn("ok")))
    p = make(auto_discover=False)
    r = await p.complete(settings=settings(max_tokens=32768, thinking_budget=16384), system="s", messages=Q, tools=[])
    assert r.message.text == "ok"
    first, second = fake.posts[-2]["body"], fake.posts[-1]["body"]
    assert first["max_tokens"] == 32768 and second["max_tokens"] == 32768 - 20000 - 256
    assert second["thinking_token_budget"] < second["max_tokens"]
    assert p.context_window(MODEL) == 32768  # learned from the error
    await p.aclose()


async def test_overflow_without_room_raises_context_overflow(fake, make):
    fake.responses.append(api_error(400, VLLM_OVERFLOW.replace("20000", "31000")))
    p = make(auto_discover=False)
    with pytest.raises(ContextOverflowError, match="limit 32768"):
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    assert len(fake.posts) == 1
    # a second overflow after the retry is final too
    fake.responses.extend([api_error(400, VLLM_OVERFLOW), api_error(400, VLLM_OVERFLOW.replace("20000", "20500"))])
    with pytest.raises(ContextOverflowError):
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    assert len(fake.posts) == 3
    # llama.cpp style (structured fields)
    fake.responses.append(("json", 400, {"error": {"code": 400, "message": "the request exceeds the available context "
                                                   "size, try increasing it", "type": "exceed_context_size_error",
                                                   "n_prompt_tokens": 5000, "n_ctx": 4096}}, {}))
    with pytest.raises(ContextOverflowError):
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    await p.aclose()


# Verbatim vLLM 0.30.0 messages (live CPU run, docs/LOCAL_LLM_VERIFICATION.md). vLLM tokenizes at
# most limit - max_tokens + 1 prompt tokens, so "at least N" is a lower bound, never the prompt size, and a
# character pre-check can fire before any tokenization ("upper bound for 0 input tokens").
VLLM030_AT_LEAST = ("This model's maximum context length is 32768 tokens. However, you requested 32512 output "
                    "tokens and your prompt contains at least 257 input tokens, for a total of at least 32769 "
                    "tokens. Please reduce the length of the input prompt or the number of requested output tokens. "
                    "(parameter=input_tokens, value=257)")
VLLM030_CHARS = ("This model's maximum context length is 32768 tokens. However, you requested 32768 output tokens "
                 "and your prompt contains 31800 characters (more than 0 characters, which is the upper bound for 0 "
                 "input tokens). Please reduce the length of the input prompt or the number of requested output "
                 "tokens. (parameter=input_text, value=31800)")


def test_vllm_lower_bound_overflow_messages_are_not_prompt_sizes():
    from vbt.providers.openai_compat import overflow_info

    assert parse_overflow(VLLM030_AT_LEAST) == (32768, None, 32512)
    assert overflow_info(VLLM030_AT_LEAST).prompt_min == 257
    assert parse_overflow(VLLM030_CHARS) == (32768, None, 32768)
    assert overflow_info(VLLM030_CHARS).prompt_min == 1
    exact = overflow_info(VLLM_OVERFLOW)
    assert (exact.prompt, exact.prompt_min) == (20000, 20000)


@pytest.mark.parametrize("message", [VLLM030_AT_LEAST, VLLM030_CHARS])
async def test_lower_bound_overflow_counts_the_prompt_via_tokenize_then_retries(fake, make, message):
    """Regression (live vLLM 0.30): the retry used the lower bound as the prompt size, so it either
    retried with max_tokens=limit-256 (characters form: 'for 0 input tokens') or limit-(limit-M+1)-256,
    which overflowed again. The prompt is now counted with POST /tokenize."""
    fake.tokenize_count = 26207
    fake.responses.append(api_error(400, message))
    fake.responses.append(("sse", text_turn("ok")))
    p = make(auto_discover=False)
    s = settings(effort=None, thinking=False, max_tokens=32768)
    r = await p.complete(settings=s, system="s", messages=Q, tools=TOOLS)
    assert r.message.text == "ok"
    assert [b["max_tokens"] for b in (fake.posts[-2]["body"], fake.posts[-1]["body"])] == \
        [32768, 32768 - 26207 - 256]
    tok = fake.tokenize_requests[-1]
    assert tok["messages"] == fake.posts[-2]["body"]["messages"] and tok["add_generation_prompt"] is True
    assert tok["tools"] == fake.posts[-2]["body"]["tools"]
    assert tok["chat_template_kwargs"] == {"enable_thinking": False}
    assert p.context_window(MODEL) == 32768
    await p.aclose()


async def test_lower_bound_overflow_without_room_after_counting(fake, make):
    fake.tokenize_count = 31900
    fake.responses.append(api_error(400, VLLM030_AT_LEAST))
    p = make(auto_discover=False)
    with pytest.raises(ContextOverflowError, match=r"limit 32768, prompt 31900 counted via /tokenize"):
        await p.complete(settings=settings(max_tokens=32512), system="s", messages=Q, tools=[])
    assert len(fake.posts) == 1
    await p.aclose()


async def test_lower_bound_overflow_without_tokenize_estimates_and_retries_once(fake, make):
    fake.tokenize_count = None  # e.g. a server without /tokenize
    fake.responses.extend([api_error(400, VLLM030_AT_LEAST), api_error(400, VLLM030_AT_LEAST)])
    p = make(auto_discover=False)
    with pytest.raises(ContextOverflowError, match="limit 32768, prompt 257 estimated"):
        await p.complete(settings=settings(max_tokens=32512), system="s", messages=Q, tools=[])
    first, second = fake.posts[-2]["body"], fake.posts[-1]["body"]
    assert second["max_tokens"] == 32768 - 257 - 256 < first["max_tokens"]
    assert len(fake.tokenize_requests) == 1
    await p.aclose()


async def test_tokenize_request_carries_the_effort_the_server_would_apply(fake, make):
    fake.tokenize_count = 1000
    fake.responses.extend([api_error(400, VLLM030_AT_LEAST), ("sse", text_turn("ok"))])
    p = make(auto_discover=False)
    await p.complete(settings=settings("high", True, 32512), system="s", messages=Q, tools=[])
    assert fake.tokenize_requests[-1]["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    await p.aclose()


async def test_data_parallel_rank_out_of_range_names_the_option(fake, make):
    fake.responses.append(api_error(400, "data_parallel_rank 1 is out of range [0, 1)."))
    p = make(auto_discover=False, data_parallel_size=2)
    with pytest.raises(ProviderError, match="data_parallel_size \\(2\\) must equal the server's --data-parallel-size"):
        await p.complete(settings=settings(session_key="k"), system="s", messages=Q, tools=[])
    await p.aclose()


@pytest.mark.parametrize("status", [503, 429, 502, 500])
async def test_transient_http_errors_are_retryable_with_retry_after(fake, make, status):
    fake.responses.append(api_error(status, "busy", "ServiceUnavailable", {"Retry-After": "7"}))
    p = make(auto_discover=False)
    with pytest.raises(RetryableProviderError) as ei:
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    await p.aclose()
    assert ei.value.retry_after == 7.0 and ei.value.status == status


async def test_connection_refused_is_retryable_with_a_hint(monkeypatch):
    _clear_env(monkeypatch)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = OpenAICompatProvider(base_url=f"http://127.0.0.1:{port}/v1")
    with pytest.raises(RetryableProviderError, match="vbt local serve"):
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    assert await p.health() is False
    with pytest.raises(ProviderError, match="not ready"):
        await p.prepare()
    info = await p.server_info()
    assert info["error"] and info["provider"] == "vllm"
    await p.aclose()


async def test_404_lists_served_models(fake, make):
    fake.responses.append(api_error(404, "The model `qwen-x` does not exist.", "NotFoundError"))
    p = make(auto_discover=False)
    with pytest.raises(ProviderError) as ei:
        await p.complete(settings=ModelSettings("vllm", "qwen-x"), system="s", messages=Q, tools=[])
    await p.aclose()
    assert not isinstance(ei.value, RetryableProviderError)
    assert MODEL in str(ei.value) and "served models" in str(ei.value)


async def test_client_errors_are_not_retryable(fake, make):
    fake.responses.append(api_error(400, "Unexpected reasoning effort 'high'"))
    fake.responses.append(api_error(401, "Unauthorized", "AuthenticationError"))
    fake.responses.append(api_error(400, "tools[0]: invalid schema"))
    p = make(auto_discover=False)
    with pytest.raises(ProviderError, match="adapter bug") as e1:
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    with pytest.raises(ProviderError, match="VBT_LLM_API_KEY") as e2:
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    with pytest.raises(ProviderError, match="invalid schema") as e3:
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    await p.aclose()
    for e in (e1, e2, e3):
        assert not isinstance(e.value, (RetryableProviderError, ContextOverflowError))


async def test_rejected_top_level_effort_falls_back_to_chat_template_kwargs(fake, make):
    fake.responses.append(api_error(400, "1 validation error for ChatCompletionRequest\nreasoning_effort\n  Input "
                                         "should be 'low', 'medium' or 'high'"))
    p = make(name="sglang", auto_discover=False)
    r = await p.complete(settings=settings("high"), system="s", messages=Q, tools=[])
    assert r.message.text == "Done."
    first, second = fake.posts[0]["body"], fake.posts[1]["body"]
    assert first["reasoning_effort"] == "xhigh"
    assert "reasoning_effort" not in second
    assert second["chat_template_kwargs"] == {"reasoning_effort": "xhigh", "enable_thinking": True}
    await p.complete(settings=settings(None, False), system="s", messages=Q, tools=[])  # remembered
    third = fake.posts[2]["body"]
    assert "reasoning_effort" not in third and third["chat_template_kwargs"] == {"enable_thinking": False}
    # a second rejection is final
    fake.responses.append(api_error(400, "reasoning_effort: still invalid"))
    with pytest.raises(ProviderError, match="still invalid"):
        await p.complete(settings=settings("high"), system="s", messages=Q, tools=[])
    await p.aclose()


async def test_mid_stream_error_and_dropped_streams_are_retryable(fake, make):
    err = {"error": {"message": "EngineCore encountered an issue", "type": "InternalServerError", "code": 500}}
    fake.responses.append(("sse_raw", sse([chunk({"content": "par"})], done=False)
                           + f"data: {json.dumps(err)}\n\n".encode()))
    fake.responses.append(("drop", [chunk({"content": "par"})]))
    fake.responses.append(("sse_nodone", [chunk({"content": "par"})]))
    p = make(auto_discover=False)
    texts = []
    with pytest.raises(RetryableProviderError, match="mid-stream") as ei:
        await p.complete(settings=settings(), system="s", messages=Q, tools=[], on_text=texts.append)
    assert ei.value.status == 500 and texts == ["par"]
    with pytest.raises(RetryableProviderError):
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    with pytest.raises(RetryableProviderError, match="ended before"):
        await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    await p.aclose()


async def test_complete_with_retry_recovers(fake, make):
    fake.responses.append(api_error(503, "loading", headers={"Retry-After": "2"}))
    fake.responses.append(("drop", [chunk({"content": "x"})]))
    fake.responses.append(("sse", text_turn("ok")))
    p = make(auto_discover=False)
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    resp = await complete_with_retry(p, policy=RetryPolicy(attempts=5, base_delay_s=1.0, jitter=0.0),
                                     sleep=fake_sleep, settings=settings(), system="s", messages=Q, tools=TOOLS)
    await p.aclose()
    assert resp.retries == 2 and resp.message.text == "ok" and sleeps == [2.0, 2.0]
    bodies = [json.dumps(r["body"], sort_keys=True) for r in fake.posts]
    assert len(set(bodies)) == 1  # identical re-sends


# ---------------------------------------------------------------------------
# Discovery, routing, concurrency, pricing
# ---------------------------------------------------------------------------


async def test_prepare_discovers_models_and_window(fake, make):
    p = make()
    assert p.context_window(MODEL) is None
    await p.prepare(models=[MODEL])
    assert p.context_window(MODEL) == 262144
    assert await p.health() is True
    info = await p.server_info()
    assert info["version"] == "0.31.0" and info["max_model_len"] == 262144 and info["family"] == "qwen3_8"
    assert info["models"][0]["id"] == MODEL and info["models"][0]["root"] == "Qwen/Qwen3.8-27B-FP8"
    assert (await p.list_models())[0]["max_model_len"] == 262144
    with pytest.raises(ProviderError, match="not served") as ei:
        await p.prepare(models=[MODEL, "claude-opus-5"])
    assert MODEL in str(ei.value)
    assert make(context_window=131072).context_window(MODEL) == 131072  # configured window wins
    fake.health_status = 503
    with pytest.raises(ProviderError, match="not ready"):
        await make().prepare()
    for prov in (p,):
        await prov.aclose()


async def test_llamacpp_window_from_meta_n_ctx(fake, make):
    """llama-server's /v1/models has no max_model_len; its per-slot window is meta.n_ctx (live llama.cpp run)."""
    fake.models = [{"id": "qwen3.5-0.8b-q8_0", "aliases": [], "object": "model", "owned_by": "llamacpp",
                    "meta": {"vocab_type": 2, "n_vocab": 248320, "n_ctx": 16384, "n_ctx_train": 262144}}]
    p = make(name="llamacpp")
    await p.prepare()
    assert p.context_window("qwen3.5-0.8b-q8_0") == 16384
    assert (await p.list_models())[0]["max_model_len"] == 16384
    assert (await p.server_info())["max_model_len"] == 16384
    await p.aclose()


async def test_family_resolved_from_the_served_root(fake, make):
    fake.models = [{"id": "local-model", "object": "model", "root": "/models/Qwen3.8-27B-FP8", "max_model_len": 65536}]
    p = make(served_model_name="local-model")
    assert p.family_for("local-model").name == "generic"
    await p.prepare()
    assert p.family_for("local-model").name == "qwen3_8"
    assert p._request(settings(), "s", Q, [])["reasoning_effort"] == "medium"
    with pytest.raises(ProviderError, match="not served"):
        await make(served_model_name="other").prepare()
    await p.aclose()


async def test_sticky_data_parallel_header(fake, make):
    p = make(data_parallel_size=4, auto_discover=False)
    for key in ("inv-cso-1", "inv-genetics-7", "inv-genetics-7"):
        await p.complete(settings=settings(session_key=key), system="s", messages=Q, tools=[])
    await p.aclose()
    ranks = [r["headers"].get("X-data-parallel-rank") for r in fake.posts]
    assert ranks[1] == ranks[2] == str(stable_hash("inv-genetics-7") % 4)
    assert ranks[0] == str(stable_hash("inv-cso-1") % 4)
    # without data parallelism no header is sent
    q = make(auto_discover=False)
    await q.complete(settings=settings(session_key="x"), system="s", messages=Q, tools=[])
    await q.aclose()
    assert "X-data-parallel-rank" not in fake.posts[-1]["headers"]


def test_sticky_url_routing():
    p = OpenAICompatProvider(base_urls=["http://a:8000/v1", "http://b:8001/v1", "http://c:8002"])
    assert p.api_bases == ["http://a:8000/v1", "http://b:8001/v1", "http://c:8002/v1"]
    for key in ("inv-1", "inv-2", "inv-3", "agent-x"):
        api, _ = p._route({"session_key": key})
        assert api == p.api_bases[stable_hash(key) % 3]
        assert p._route({"session_key": key})[0] == api
    api, _ = p._route({"agent_name": "genetics"})  # falls back to agent_name
    assert api == p.api_bases[stable_hash("genetics") % 3]
    assert OpenAICompatProvider(base_urls="http://a:1/v1, http://b:2/v1").api_bases == ["http://a:1/v1",
                                                                                        "http://b:2/v1"]


async def test_api_key_header_and_max_concurrency(fake, make, monkeypatch):
    monkeypatch.setenv("VBT_LLM_API_KEY", "secret-token")
    fake.delay = 0.15
    p = make(max_concurrency=1, auto_discover=False)
    await asyncio.gather(*(p.complete(settings=settings(), system="s", messages=Q, tools=[]) for _ in range(3)))
    await p.aclose()
    assert fake.max_inflight == 1
    assert fake.posts[-1]["headers"]["Authorization"] == "Bearer secret-token"
    fake.max_inflight = 0
    q = make(auto_discover=False)
    await asyncio.gather(*(q.complete(settings=settings(), system="s", messages=Q, tools=[]) for _ in range(3)))
    await q.aclose()
    assert fake.max_inflight > 1


async def test_pricing_cost(fake, make):
    fake.responses.append(("sse", text_turn("ok", u=usage(1_000_000, 1_000_000, cached=500_000))))
    p = make(pricing={"input_per_mtok": 1.0, "cached_per_mtok": 0.1, "output_per_mtok": 2.0}, auto_discover=False)
    r = await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    await p.aclose()
    assert r.cost_usd == pytest.approx(0.5 * 1.0 + 0.5 * 0.1 + 2.0)


async def test_loopback_servers_bypass_proxies(fake, make, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{dead}")  # a proxy that cannot answer
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    p = make(auto_discover=False)
    assert p.trust_env is False
    assert (await p.complete(settings=settings(), system="s", messages=Q, tools=[])).message.text == "Done."
    await p.aclose()
    assert OpenAICompatProvider(base_url="http://gpu-box:8000/v1").trust_env is True
    assert OpenAICompatProvider(base_url="http://localhost:8000/v1", trust_env=True).trust_env is True


async def test_harness_turn_through_the_runtime(fake, config):
    """A real CSO turn: tool call -> runtime executes it -> results re-sent -> final answer."""
    from vbt.orchestrator import open_session

    fake.responses.extend([
        ("sse", [chunk({"reasoning": "Look around first."}), chunk({"content": "Checking files."}),
                 tc_delta(0, id="call_g1", name="Glob", args='{"pattern": "*.nothing-matches"}'),
                 tc_delta(1, id="call_g2", name="Glob", args='{"pattern": "*.also-nothing"}'),
                 chunk(finish="tool_calls"), chunk(usage=usage(3000, 40, cached=2048))]),
        ("sse", text_turn("EGFR is a plausible target.", u=usage(3500, 30, cached=3000))),
    ])
    config.setdefault("orchestration", {})["enforce_review"] = False
    provider = OpenAICompatProvider(base_url=fake.url + "/v1", family="qwen3_8")
    session = await open_session(config, provider=provider, start_mcp=False)
    try:
        out = await session.ask("Is EGFR a target in NSCLC?")
    finally:
        await session.close()
    assert out.endswith("EGFR is a plausible target.")  # the answer joins the turn's text segments
    first, second = fake.posts[0]["body"], fake.posts[1]["body"]
    assert first["messages"][0]["role"] == "system"
    assert [m["role"] for m in first["messages"]].count("system") == 1
    assert first["reasoning_effort"] in {"xhigh", "medium", "low", "none"}
    assert {"Glob", "Task"} <= {t["function"]["name"] for t in first["tools"]}
    roles = [m["role"] for m in second["messages"]]
    assert roles[0] == "system" and roles.count("system") == 1 and roles[-2:] == ["tool", "tool"]
    asst = [m for m in second["messages"] if m["role"] == "assistant"][-1]
    assert asst["reasoning"] == "Look around first." and asst["content"] == "Checking files."
    assert [tc["id"] for tc in asst["tool_calls"]] == ["call_g1", "call_g2"]
    assert [m["tool_call_id"] for m in second["messages"][-2:]] == ["call_g1", "call_g2"]
    # the second request repeats the first one's prefix (append-only history)
    assert second["messages"][:len(first["messages"])] == first["messages"]


def test_openai_compat_does_not_import_anthropic():
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    code = ("import sys; sys.path.insert(0, %r); import vbt.providers.openai_compat, vbt.providers.families; "
            "print('anthropic' in sys.modules)" % str(src))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_works_across_event_loops(fake):
    """``vbt doctor`` / CLI flows may call the provider from several asyncio.run() loops."""
    p = OpenAICompatProvider(base_url=fake.url + "/v1", max_concurrency=2)

    async def one():
        return await p.complete(settings=settings(), system="s", messages=Q, tools=[])

    assert asyncio.run(p.prepare()) is None
    assert asyncio.run(one()).message.text == "Done."
    assert asyncio.run(one()).message.text == "Done."  # a fresh client and semaphore for the new loop
    asyncio.run(p.aclose())
