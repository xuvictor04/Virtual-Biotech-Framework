"""OpenAI-compatible Chat Completions adapter for local open-weight models.

Serves the harness from a self-hosted inference server: vLLM (primary,
``/v1/chat/completions``), SGLang or llama.cpp's ``llama-server``. Registered
as provider names ``vllm``, ``sglang``, ``openai_compat`` and ``llamacpp``.
The default deployment is Qwen3.8-27B on vLLM 0.31 (see
``docs/PROVIDERS.md`` and ``configs/local_models.yaml``); per-model dialects
(reasoning controls, sampling, response fields) live in
:mod:`vbt.providers.families`.

What the adapter guarantees (from the verified harness requirements):

* exactly one ``system`` message, at index 0 (all ``SystemSegment`` joined);
* assistant turns as ``content`` (or null) + ``tool_calls`` with JSON-object
  ``arguments`` + the turn's reasoning replayed in ``reasoning`` (only
  reasoning produced by this provider / model family);
* tool results as ``role: tool`` messages in the order of the assistant's
  tool calls (the Qwen template pairs them positionally), any further text of
  the harness user message as a separate ``user`` message after them;
* tool names outside ``^[A-Za-z0-9_-]{1,64}$`` aliased reversibly;
* ``strict`` tools, ``tool_choice`` (``extra['tool_choice']``), parallel calls;
* reasoning effort mapped per family (never ``high``/``max`` to Qwen3.8),
  ``thinking_token_budget``, card sampling sent explicitly;
* streaming with tool-call deltas accumulated by index; usage with cached
  prompt tokens; ``length`` -> ``MAX_TOKENS`` or ``CONTEXT_EXCEEDED``;
* context overflow (HTTP 400 "maximum context length ...") retried once with a
  smaller ``max_tokens`` when there is room, else :class:`ContextOverflowError`;
* transient failures (connect errors, timeouts, 429, 5xx, error events
  mid-stream, dropped streams) as :class:`RetryableProviderError`.

Only ``httpx`` is used (already a harness dependency); no vendor SDK.
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import hashlib
import inspect
import ipaddress
import itertools
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterable
from urllib.parse import urlsplit

import httpx

from .base import (
    ContextOverflowError,
    DocumentPart,
    ImagePart,
    LLMProvider,
    Message,
    ModelResponse,
    ModelSettings,
    ProviderCapabilities,
    ProviderError,
    RetryableProviderError,
    StopReason,
    SystemSegment,
    TextBlock,
    TextCallback,
    ThinkingBlock,
    ThinkingCallback,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    content_text,
    system_text,
)
from ..envpolicy import redact_url, register_secret, url_secrets
from .families import FAMILIES, GENERIC, GENERIC_WITH_EFFORT, ModelFamily, resolve_family

log = logging.getLogger(__name__)

#: Provider names this adapter is registered under.
PROVIDER_NAMES = ("vllm", "sglang", "openai_compat", "llamacpp")

#: ``ModelSettings.extra`` keys this adapter understands. None of them (and no
#: other ``extra`` key) is ever forwarded verbatim to the server.
LOCAL_EXTRA_KEYS = frozenset({
    "agent_name", "session_key", "tool_choice", "force_tool", "thinking_budget", "context_window_tokens",
    "reasoning_effort", "sampling", "prompt_cache", "context_management",
})

TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

DEFAULT_THINKING_BUDGET_CAP = 16384
OVERFLOW_MIN_ROOM = 2048       # retry an overflow with a smaller max_tokens only with this much room
OVERFLOW_RETRY_MARGIN = 256    # max_tokens = limit - prompt - margin on that retry
CLAMP_MARGIN = 256             # max_tokens clamp: window - estimated prompt - margin
MIN_CLAMPED_MAX_TOKENS = 1024  # never clamp below this (the server reports a real overflow instead)
IMAGE_TOKENS = 1600

SERVE_HINT = ("is the inference server running? Start it with `vbt local serve --profile <h100|h200|rtxpro6000|5090>` "
              "(see deploy/local/README.md) and check provider.options.base_url / VBT_LLM_BASE_URL")
#: ``vbt local serve`` starts vLLM: a llama.cpp provider is told how to start its own server instead.
LLAMACPP_SERVE_HINT = ("is the inference server running? Start llama-server with the model and --jinja "
                       "(docs/E2E_RUN.md; scripts/dev/cpu_server.sh --engine llamacpp) and check "
                       "provider.options.base_url / VBT_LLM_BASE_URL")


def serve_hint(provider_name: str) -> str:
    """How to start the server a provider of this name talks to."""
    return LLAMACPP_SERVE_HINT if provider_name == "llamacpp" else SERVE_HINT

_TEMPLATE_BUG_RE = re.compile(
    r"Unexpected reasoning effort|System message must be at the beginning|No user query found|"
    r"Conversation roles must alternate|raise_exception", re.I)
_OVERFLOW_RE = re.compile(
    r"maximum context length|model'?s context length|context window|exceed_context_size|available context size|"
    r"maximum allowed length|too many (?:input )?tokens|prompt is too long|max_(?:completion_)?tokens'?.{0,40}too large|"
    r"maximum model length|longer than the model", re.I)
_LIMIT_RES = (r"maximum context length (?:is|of) (\d+)", r"model'?s context length \((\d+)",
              r"maximum allowed length \((\d+)", r"context length of (\d+)", r"maximum model length (?:is )?(\d+)",
              r"n_ctx\W{0,4}(\d+)", r"context size \((\d+)")
_PROMPT_RES = (r"prompt contains (\d+) input tokens", r"(?:request|prompt) has (\d+) input tokens",
               r"\((\d+) in the messages", r"(\d+) tokens from the input", r"[Ii]nput (?:length )?\((\d+) tokens\)",
               r"n_prompt_tokens\W{0,4}(\d+)", r"(?<!\d)(?<!at least )(?<!bound for )(\d+) input tokens")
# Recent vLLM (observed on 0.30.0) tokenizes at most (limit - max_tokens + 1) prompt tokens before rejecting
# a request, so its overflow message carries only a LOWER bound ("your prompt contains at least N input
# tokens", where N is always limit - max_tokens + 1) or, from a character pre-check, "your prompt contains C
# characters (more than X characters, which is the upper bound for Y input tokens)". Neither is the real
# prompt size.
_PROMPT_MIN_RES = (r"contains at least (\d+) input tokens",)
_PROMPT_CHARS_RE = re.compile(r"contains (\d+) characters \(more than \d+ characters, which is the upper bound for "
                              r"(\d+) input tokens\)")
_REQUEST_RES = (r"requested (\d+) output tokens", r"(\d+) in the completion", r"(\d+) tokens for the completion",
                r"too large: (\d+)")

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FUNCTION_XML_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAMETER_XML_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)

_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(msg, *args)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def stable_hash(key: str) -> int:
    """Process-independent hash (``hash()`` is salted per interpreter)."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16)


def tool_alias(name: str, max_len: int = 64) -> str:
    """``name`` if it is a valid OpenAI function name, else a stable alias:
    the sanitized name truncated, plus ``_`` and the first 8 hex chars of its SHA-1."""
    max_len = max(16, min(64, int(max_len or 64)))
    if TOOL_NAME_RE.match(name or "") and len(name) <= max_len:
        return name
    digest = hashlib.sha1((name or "").encode("utf-8")).hexdigest()[:8]
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_") or "tool"
    return f"{base[:max_len - 9]}_{digest}"


#: Environment variable the adapter reads its bearer token from when no ``api_key`` option is set.
API_KEY_ENV = "VBT_LLM_API_KEY"


#: ``engine="k"`` label of vLLM's per-engine metrics (one per data-parallel rank).
_DP_ENGINE_RE = re.compile(r'^vllm:num_requests_running\{[^}]*\bengine="(\d+)"', re.MULTILINE)


def resolve_api_key(api_key: str | None = None, api_key_env: str | None = None) -> str | None:
    """The bearer token for the inference server: the ``api_key`` option, else
    ``$VBT_LLM_API_KEY``, else the variable named by ``api_key_env`` (an explicit
    opt-in such as ``OPENAI_API_KEY`` for a hosted OpenAI-compatible API).

    ``OPENAI_API_KEY`` is never read implicitly: developers commonly have it
    exported, and a self-hosted server (or whatever listens on its port) must
    not receive the user's OpenAI billing credential. A key given in the config
    is registered for redaction in tool output."""
    key = (api_key or "").strip()
    if key:
        register_secret(key, "provider.options.api_key")
        return key
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key and api_key_env and str(api_key_env).strip():
        key = os.environ.get(str(api_key_env).strip(), "").strip()
    return key or None


def _split_base_url(url: str) -> tuple[str, str]:
    """``(api_base, server_root)``: ``http://h:8000/v1`` -> (``.../v1``, ``http://h:8000``).
    A bare ``http://h:8000`` gets ``/v1`` appended; any other path is used as is."""
    u = url.strip().rstrip("/")
    path = urlsplit(u).path.rstrip("/")
    if path.endswith("/v1"):
        return u, u[: -len("/v1")]
    if path in ("", "/"):
        return u + "/v1", u
    return u, u


def _is_loopback(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _retry_after(headers: Any) -> float | None:
    if not headers:
        return None
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000.0)
        except ValueError:
            pass
    ra = headers.get("retry-after")
    if not ra:
        return None
    try:
        return max(0.0, float(ra))
    except ValueError:
        try:
            return max(0.0, email.utils.parsedate_to_datetime(ra).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _error_info(payload: Any) -> tuple[str, str | None, int | None, dict[str, Any]]:
    """``(message, type, code, error dict)`` from an OpenAI/vLLM/SGLang/llama.cpp error body."""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("msg") or json.dumps(err)[:2000]
            return str(msg), err.get("type"), _int(err.get("code")), err
        if isinstance(err, str):
            return err, payload.get("type"), _int(payload.get("code") or payload.get("status")), payload
        if payload.get("object") == "error" or "message" in payload:
            return str(payload.get("message") or ""), payload.get("type"), _int(payload.get("code")), payload
        if "detail" in payload:
            return str(payload["detail"]), None, None, payload
        return json.dumps(payload)[:2000], None, None, payload
    return str(payload)[:2000], None, None, {}


def _first_int(patterns: Iterable[str], text: str) -> int | None:
    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1))
    return None


@dataclass(frozen=True)
class OverflowInfo:
    """Facts from a context-overflow error. ``prompt`` is the exact prompt size
    when the server reported it; ``prompt_min`` is a lower bound when it did not
    (recent vLLM, e.g. 0.30, stops tokenizing at ``limit - requested + 1``)."""

    limit: int | None = None
    prompt: int | None = None
    prompt_min: int | None = None
    requested: int | None = None


def overflow_info(message: str, err: dict[str, Any] | None = None) -> OverflowInfo:
    """:class:`OverflowInfo` from a context-overflow error (vLLM, SGLang or
    llama.cpp wording); unknown parts are None."""
    err = err or {}
    message = message or ""
    limit = _int(err.get("n_ctx")) or _first_int(_LIMIT_RES, message)
    requested = _first_int(_REQUEST_RES, message)
    exact = _int(err.get("n_prompt_tokens"))
    if exact is not None:
        return OverflowInfo(limit, exact, exact, requested)
    lower = _first_int(_PROMPT_MIN_RES, message)
    if lower is not None:
        return OverflowInfo(limit, None, lower, requested)
    chars = _PROMPT_CHARS_RE.search(message)
    if chars:
        return OverflowInfo(limit, None, int(chars.group(2)) + 1, requested)
    prompt = _first_int(_PROMPT_RES, message)
    return OverflowInfo(limit, prompt, prompt, requested)


def parse_overflow(message: str, err: dict[str, Any] | None = None) -> tuple[int | None, int | None, int | None]:
    """``(limit, prompt_tokens, requested_output)`` from a context-overflow error
    (vLLM, SGLang or llama.cpp wording); unknown parts are None. ``prompt_tokens``
    is None when the server reported only a lower bound (see :func:`overflow_info`)."""
    info = overflow_info(message, err)
    return info.limit, info.prompt, info.requested


def is_overflow_message(message: str, etype: str | None = None) -> bool:
    return bool(_OVERFLOW_RE.search(message or "")) or (etype or "") == "exceed_context_size_error"


async def _emit(cb: TextCallback | None, text: str) -> None:
    if cb is None or not text:
        return
    r = cb(text)
    if inspect.isawaitable(r):
        await r


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Merge ``src`` into ``dst``; dict values are merged one level deep."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = {**dst[k], **v}
        else:
            dst[k] = v
    return dst


def estimate_prompt_tokens(body: dict[str, Any]) -> int:
    """Rough prompt size of a request body (chars / 4; images ~1600 tokens)."""
    chars = 0
    images = 0
    for m in body.get("messages") or []:
        chars += 8
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    chars += len(p.get("text") or "")
                else:
                    images += 1
        for f in ("reasoning", "reasoning_content"):
            v = m.get(f)
            if isinstance(v, str):
                chars += len(v)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            chars += 16 + len(fn.get("name") or "") + len(fn.get("arguments") or "")
    for t in body.get("tools") or []:
        chars += len(json.dumps(t, ensure_ascii=False))
    return chars // 4 + images * IMAGE_TOKENS


# ---------------------------------------------------------------------------
# Text tool-call fallback (server started without a tool-call parser)
# ---------------------------------------------------------------------------


def _schema_types(schema: Any) -> set[str]:
    if not isinstance(schema, dict):
        return set()
    t = schema.get("type")
    out = set(t) if isinstance(t, list) else ({t} if isinstance(t, str) else set())
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or []:
            out |= _schema_types(sub)
    return out


def _coerce_param(value: str, schema: Any) -> Any:
    types = _schema_types(schema)
    if types and types <= {"string"}:
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        if "boolean" in types and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return value


def _parse_xml_function(name: str, body: str, schemas: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    props = ((schemas.get(name) or {}).get("properties") or {}) if isinstance(schemas.get(name), dict) else {}
    args: dict[str, Any] = {}
    for pm in _PARAMETER_XML_RE.finditer(body):
        key = pm.group(1).strip()
        args[key] = _coerce_param(pm.group(2), props.get(key))
    return name, args, body


def _parse_tool_call_body(inner: str, schemas: dict[str, Any]) -> tuple[str, Any, str] | None:
    """``(name, arguments, raw)`` for one ``<tool_call>`` body (Hermes JSON or Qwen XML)."""
    s = inner.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            m = re.search(r'"name"\s*:\s*"([^"]+)"', s)
            return (m.group(1), None, s) if m else None
        if not isinstance(obj, dict) or not obj.get("name"):
            return None
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return str(obj["name"]), None, args
        return str(obj["name"]), args, s
    m = _FUNCTION_XML_RE.search(s)
    if m:
        return _parse_xml_function(m.group(1).strip(), m.group(2), schemas)
    return None


def extract_text_tool_calls(text: str, tools: list[ToolSpec] | None = None,
                            aliases: dict[str, str] | None = None) -> tuple[list[ToolCall], str]:
    """Tool calls written into the text (``<tool_call>{json}</tool_call>`` or
    the Qwen XML ``<tool_call><function=name><parameter=k>v</parameter>...``
    form), and the text with those blocks removed. ``aliases`` maps wire names
    back to harness names. Unparseable blocks stay in the text."""
    aliases = aliases or {}
    schemas: dict[str, Any] = {}
    for t in tools or []:
        schemas[t.name] = t.input_schema
        schemas[tool_alias(t.name)] = t.input_schema
    calls: list[ToolCall] = []

    def make(parsed: tuple[str, Any, str]) -> None:
        name, args, raw = parsed
        name = aliases.get(name, name)
        cid = f"call_{uuid.uuid4().hex[:24]}"
        if isinstance(args, dict):
            calls.append(ToolCall(cid, name, args))
        else:
            what = "invalid JSON" if args is None else f"arguments must be a JSON object, got {type(args).__name__}"
            calls.append(ToolCall(cid, name, {}, native={"invalid_arguments": raw, "error": what}))

    def repl(m: re.Match) -> str:
        parsed = _parse_tool_call_body(m.group(1), schemas)
        if parsed is None:
            return m.group(0)
        make(parsed)
        return ""

    remaining = _TOOL_CALL_BLOCK_RE.sub(repl, text)
    if not calls and "<function=" in remaining:
        def repl_fn(m: re.Match) -> str:
            make(_parse_xml_function(m.group(1).strip(), m.group(2), schemas))
            return ""
        remaining = _FUNCTION_XML_RE.sub(repl_fn, remaining)
    return calls, remaining.strip()


# ---------------------------------------------------------------------------
# Stream state
# ---------------------------------------------------------------------------


class _Overflow(Exception):
    """Internal: the server rejected the request as too long for its context."""

    def __init__(self, message: str, info: OverflowInfo | None = None) -> None:
        super().__init__(message)
        info = info or OverflowInfo()
        self.limit, self.prompt, self.prompt_min, self.requested = (info.limit, info.prompt, info.prompt_min,
                                                                    info.requested)


class _EffortRejected(Exception):
    """Internal: request validation rejected the top-level ``reasoning_effort``."""


@dataclass
class _ToolSlot:
    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamState:
    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    reasoning_field: str | None = None
    slots: dict[int, _ToolSlot] = field(default_factory=dict)
    finish: str | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    request_id: str | None = None
    done: bool = False
    chunks: int = 0


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAICompatProvider(LLMProvider):
    """Chat Completions client for vLLM / SGLang / llama.cpp servers.

    Options (``provider.options`` in the config):

    * ``base_url`` — e.g. ``http://localhost:8000/v1`` (env ``VBT_LLM_BASE_URL``
      when unset); ``base_urls`` — several replicas (sticky per agent session).
    * ``model`` / ``served_model_name`` — the served model name to request
      (``served_model_name`` overrides ``ModelSettings.model``).
    * ``api_key`` — optional (else env ``VBT_LLM_API_KEY``, or the variable
      named by ``api_key_env``). ``OPENAI_API_KEY`` is never read implicitly:
      a self-hosted server must not receive the user's OpenAI credential.
    * ``family`` — ``auto`` (from the model id / served root) or a
      :mod:`~vbt.providers.families` name (``qwen3_8``, ``qwen3_6``, ``qwen3``,
      ``deepseek_v4``, ``generic``).
    * ``timeout_s`` (connect/write, 30) and ``read_timeout_s`` (max silence
      between streamed chunks, 600); there is no total cap.
    * ``max_concurrency`` — client-side cap on in-flight requests.
    * ``data_parallel_size`` + ``routing='header'`` — pin each agent session to
      one vLLM DP rank via ``X-data-parallel-rank``; ``routing='urls'`` /
      several ``base_urls`` — pick a replica URL per session.
    * ``pricing`` — ``{input_per_mtok, cached_per_mtok, output_per_mtok}`` USD
      (default: all 0; local runs are budgeted in tokens).
    * ``extra_body`` — merged into every request last.
    * ``context_window`` — window in tokens (else ``max_model_len`` from
      ``/v1/models``); ``vision`` — the server accepts images (default False:
      served with ``--language-model-only``).
    * ``reasoning_effort_supported`` — send ``reasoning_effort`` for the
      ``generic`` family; ``replay_reasoning_field`` — override the assistant
      field used to replay reasoning; ``parallel_tool_calls`` (True);
      ``auto_discover`` (True: read ``/v1/models`` once before the first call);
      ``wait_ready_s`` (0: how long ``prepare()`` waits for ``/health``);
      ``trust_env`` (honour HTTP(S)_PROXY / NO_PROXY; default: yes unless every
      base URL is a loopback address, which no proxy can reach); ``headers``.
    """

    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        family: str | None = "auto",
        name: str = "vllm",
        timeout_s: float = 30.0,
        read_timeout_s: float | None = 600.0,
        max_concurrency: int | None = None,
        data_parallel_size: int = 1,
        routing: str = "header",
        base_urls: list[str] | str | None = None,
        pricing: dict[str, float] | None = None,
        extra_body: dict[str, Any] | None = None,
        served_model_name: str | None = None,
        *,
        context_window: int | None = None,
        vision: bool = False,
        reasoning_effort_supported: bool = False,
        replay_reasoning_field: str | None = None,
        parallel_tool_calls: bool | None = True,
        auto_discover: bool = True,
        wait_ready_s: float = 0.0,
        trust_env: bool | None = None,
        headers: dict[str, str] | None = None,
        api_key_env: str | None = None,
        **unknown: Any,
    ) -> None:
        if unknown:
            _warn_once(f"opts:{sorted(unknown)}", "OpenAICompatProvider: ignoring unknown provider options %s",
                       sorted(unknown))
        self.name = str(name or "openai_compat")
        urls: list[str] = []
        if isinstance(base_urls, str):
            base_urls = [u for u in re.split(r"[,\s]+", base_urls) if u]
        for u in base_urls or []:
            if u and str(u).strip():
                urls.append(str(u).strip())
        primary = (base_url or "").strip() or os.environ.get("VBT_LLM_BASE_URL", "").strip()
        if not urls and primary:
            urls = [primary]
        self.base_url: str | None = urls[0] if urls else None
        for u in urls:  # a user:password@ URL: never echo the password into tool output
            register_secret(url_secrets(u), "provider.options.base_url credentials")
        self._bases: list[tuple[str, str]] = [_split_base_url(u) for u in urls]
        self.model = (model or "").strip() or None
        self.served_model_name = (served_model_name or "").strip() or None
        self._api_key = resolve_api_key(api_key, api_key_env)
        fam_opt = (family or "auto").strip() if isinstance(family, str) else "auto"
        if fam_opt.lower() not in ("", "auto"):
            try:
                resolve_family(None, fam_opt)
            except ValueError as exc:
                raise ProviderError(f"provider {self.name!r}: {exc}") from None
        self.family_option = fam_opt or "auto"
        self.timeout_s = float(timeout_s or 30.0)
        self.read_timeout_s = float(read_timeout_s) if read_timeout_s else None
        self.max_concurrency = int(max_concurrency) if max_concurrency else None
        self.data_parallel_size = max(1, int(data_parallel_size or 1))
        routing = (routing or "header").strip().lower()
        if routing not in ("header", "urls"):
            raise ProviderError(f"provider {self.name!r}: routing must be 'header' or 'urls', got {routing!r}")
        self.routing = routing
        self.pricing = self._check_pricing(pricing)
        self.extra_body = dict(extra_body or {})
        self.context_window_option = int(context_window) if context_window else None
        self.vision = bool(vision)
        self.reasoning_effort_supported = bool(reasoning_effort_supported)
        self.replay_reasoning_field = replay_reasoning_field or None
        self.parallel_tool_calls = parallel_tool_calls
        self.auto_discover = bool(auto_discover)
        self.wait_ready_s = float(wait_ready_s or 0.0)
        if trust_env is None:
            trust_env = not (self._bases and all(_is_loopback(api) for api, _ in self._bases))
        self.trust_env = bool(trust_env)
        self.extra_headers = {str(k): str(v) for k, v in (headers or {}).items()}
        # discovery cache (from /v1/models, or learned from overflow errors)
        self._max_len: dict[str, int] = {}
        self._roots: dict[str, str] = {}
        self._learned_window: int | None = None
        self._discovered = False
        self._discovering = False
        self._prepared = False
        # set when the server's request validation rejected top-level reasoning_effort:
        # the effort then travels as chat_template_kwargs.reasoning_effort
        self._effort_in_kwargs = False
        # per-event-loop client / semaphore
        self._http: httpx.AsyncClient | None = None
        self._http_loop: asyncio.AbstractEventLoop | None = None
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        self._rr = itertools.count()

    # ------------------------------------------------------------------ config helpers

    @staticmethod
    def _check_pricing(pricing: dict[str, Any] | None) -> dict[str, float]:
        keys = ("input_per_mtok", "cached_per_mtok", "output_per_mtok")
        out = {k: 0.0 for k in keys}
        for k, v in (pricing or {}).items():
            if k not in keys:
                _warn_once(f"pricing:{k}", "provider pricing key %r is not used (known: %s)", k, ", ".join(keys))
                continue
            out[k] = float(v or 0.0)
        return out

    @property
    def api_bases(self) -> list[str]:
        return [a for a, _ in self._bases]

    def family_for(self, model: str | None) -> ModelFamily:
        """The dialect used for ``model`` (explicit option, else matched from the
        served name, the requested model id or the served model's root path)."""
        if self.family_option.lower() not in ("", "auto"):
            fam = resolve_family(None, self.family_option)
        else:
            fam = GENERIC
            wire = self.served_model_name or model
            for cand in (self.served_model_name, model, self._roots.get(wire or ""), self._roots.get(model or "")):
                if cand:
                    f = resolve_family(cand)
                    if f is not GENERIC:
                        fam = f
                        break
        if fam is GENERIC and self.reasoning_effort_supported:
            fam = GENERIC_WITH_EFFORT
        return fam

    def _replay_field(self, fam: ModelFamily) -> str | None:
        if self.replay_reasoning_field:
            return self.replay_reasoning_field
        if fam.replay_reasoning_field is None:
            return None
        if self.name in ("sglang", "llamacpp"):
            return "reasoning_content"
        return fam.replay_reasoning_field

    def _wire_model(self, settings: ModelSettings) -> str:
        return self.served_model_name or settings.model or self.model or ""

    # ------------------------------------------------------------------ facts

    def capabilities(self, model: str | None = None) -> ProviderCapabilities:
        fam = self.family_for(model or self.model)
        return ProviderCapabilities(
            images=self.vision, documents=False, web_search=False, server_context_management=False,
            history_bound_thinking=False, replays_reasoning=self._replay_field(fam) is not None, tool_choice=True,
        )

    def context_window(self, model: str) -> int | None:
        if self.context_window_option:
            return self.context_window_option
        return self._server_window(model)

    def server_context_window(self, model: str | None = None) -> int | None:
        """The window the server enforces (``max_model_len`` from ``/v1/models``,
        or learned from an overflow error), None until discovered. The context
        manager never compacts against a configured window larger than this."""
        return self._server_window(model)

    def _server_window(self, model: str | None) -> int | None:
        for cand in (self.served_model_name, model):
            if cand and cand in self._max_len:
                return self._max_len[cand]
        values = set(self._max_len.values())
        if len(values) == 1:
            return values.pop()
        return self._learned_window

    def _clamp_window(self, settings: ModelSettings) -> int | None:
        """Window used to clamp ``max_tokens``: the server's real limit when known,
        else the configured one."""
        w = self._server_window(settings.model) or self.context_window_option
        if not w:
            w = _int((settings.extra or {}).get("context_window_tokens"))
        return w or None

    def check_credentials(self) -> str | None:
        if not self._bases:
            return (f"no base_url configured for provider {self.name!r}: set provider.options.base_url (or the "
                    "VBT_LLM_BASE_URL environment variable), e.g. http://localhost:8000/v1")
        for api, _ in self._bases:
            parts = urlsplit(api)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                return f"invalid base_url {api!r} for provider {self.name!r}: expected http(s)://host:port/v1"
        return None

    # ------------------------------------------------------------------ encode

    def _own_thinking(self, b: ThinkingBlock, fam: ModelFamily) -> bool:
        if b.provider == self.name:
            return True
        native = b.native or {}
        return b.provider in PROVIDER_NAMES and native.get("family") == fam.name

    @staticmethod
    def _tool_text(r: ToolResult) -> str:
        text = content_text(r.content)
        if not text.strip():
            text = "(empty result)"
        if r.is_error and not text.lstrip().lower().startswith(("error", "not executed", "[error")):
            text = "Error: " + text
        return text

    @staticmethod
    def _image_part(p: ImagePart) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": f"data:{p.media_type};base64,{p.data_b64}"}}

    def _encode_assistant(self, m: Message, fam: ModelFamily, alias: Callable[[str], str],
                          replay_field: str | None) -> dict[str, Any]:
        texts = [b.text for b in m.content if isinstance(b, TextBlock) and b.text]
        reasoning = [b.text for b in m.content
                     if isinstance(b, ThinkingBlock) and b.text and self._own_thinking(b, fam)]
        calls = [b for b in m.content if isinstance(b, ToolCall)]
        out: dict[str, Any] = {"role": "assistant",
                               "content": "\n\n".join(texts) if texts else (None if calls else "")}
        if reasoning and replay_field:
            out[replay_field] = "\n\n".join(reasoning)
        if calls:
            out["tool_calls"] = [
                {"id": c.id or f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                 "function": {"name": alias(c.name),
                              "arguments": json.dumps(c.input if isinstance(c.input, dict) else {},
                                                      ensure_ascii=False)}}
                for c in calls
            ]
        return out

    def _encode_user(self, m: Message, call_order: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        results = [b for b in m.content if isinstance(b, ToolResult)]
        tool_images: list[ImagePart] = []
        if results:
            rank = {cid: i for i, cid in enumerate(call_order)}
            ordered = sorted(enumerate(results), key=lambda t: (rank.get(t[1].tool_call_id, len(rank) + t[0])))
            for _, r in ordered:
                out.append({"role": "tool", "tool_call_id": r.tool_call_id, "content": self._tool_text(r)})
                if self.vision and not isinstance(r.content, str):
                    tool_images.extend(p for p in r.content if isinstance(p, ImagePart))
        parts: list[dict[str, Any]] = []
        if tool_images:
            names = ", ".join(p.source or p.media_type for p in tool_images)
            parts.append({"type": "text", "text": f"[Images returned by the tool results above: {names}]"})
            parts.extend(self._image_part(p) for p in tool_images)
        for b in m.content:
            if isinstance(b, TextBlock):
                if b.text:
                    parts.append({"type": "text", "text": b.text})
            elif isinstance(b, ImagePart):
                parts.append(self._image_part(b) if self.vision else {"type": "text", "text": content_text([b])})
            elif isinstance(b, DocumentPart):
                parts.append({"type": "text", "text": content_text([b])})
        if parts:
            if all(p["type"] == "text" for p in parts):
                out.append({"role": "user", "content": "\n\n".join(p["text"] for p in parts)})
            else:
                out.append({"role": "user", "content": parts})
        elif not results:
            out.append({"role": "user", "content": "(no content)"})
        return out

    def _encode_messages(self, system: str | list[SystemSegment] | None, messages: list[Message],
                         fam: ModelFamily, alias: Callable[[str], str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        sys_text = system_text(system)
        if sys_text:
            out.append({"role": "system", "content": sys_text})
        replay_field = self._replay_field(fam)
        call_order: list[str] = []
        for m in messages:
            if m.role == "assistant":
                out.append(self._encode_assistant(m, fam, alias, replay_field))
                call_order = [c.id for c in m.tool_calls]
            else:
                out.extend(self._encode_user(m, call_order))
                call_order = []
        if not any(x["role"] == "user" for x in out):
            raise ProviderError(f"{self.name}: the conversation has no user message (the chat template would raise "
                                "'No user query found'); this is a harness bug in the caller")
        return out

    def _aliaser(self, fam: ModelFamily) -> Callable[[str], str]:
        return lambda n: tool_alias(n, fam.max_tool_name_len)

    def _reverse_aliases(self, tools: list[ToolSpec], messages: list[Message], fam: ModelFamily) -> dict[str, str]:
        alias = self._aliaser(fam)
        rev: dict[str, str] = {}
        for m in messages:
            for c in m.tool_calls:
                rev.setdefault(alias(c.name), c.name)
        for t in tools or []:
            rev[alias(t.name)] = t.name
        return rev

    def _encode_tool(self, t: ToolSpec, alias: Callable[[str], str]) -> dict[str, Any]:
        params = t.input_schema if isinstance(t.input_schema, dict) and t.input_schema else \
            {"type": "object", "properties": {}}
        wire = alias(t.name)
        desc = t.description or ""
        if wire != t.name:
            desc = f"[harness tool {t.name}] {desc}".strip()
        fn: dict[str, Any] = {"name": wire, "description": desc, "parameters": params}
        if getattr(t, "strict", False):
            fn["strict"] = True
        return {"type": "function", "function": fn}

    def _tool_choice(self, extra: dict[str, Any], tools: list[ToolSpec], alias: Callable[[str], str]) -> Any:
        tc = extra.get("tool_choice")
        if tc is None and extra.get("force_tool"):
            tc = {"name": extra["force_tool"]}
        if tc is None:
            return "auto"
        if isinstance(tc, str):
            if tc in ("auto", "required", "none"):
                return tc
            name = tc
        elif isinstance(tc, dict):
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            if not name:
                raise ProviderError(f"{self.name}: invalid tool_choice {tc!r} (expected 'auto', 'required', 'none' "
                                    "or {'name': <tool>})")
        else:
            raise ProviderError(f"{self.name}: invalid tool_choice {tc!r}")
        if name not in {t.name for t in tools}:
            raise ProviderError(f"{self.name}: tool_choice forces tool {name!r}, which is not among this request's "
                                f"tools {[t.name for t in tools]}")
        return {"type": "function", "function": {"name": alias(name)}}

    def _max_tokens(self, settings: ModelSettings, body: dict[str, Any], override: int | None) -> int:
        if override is not None:
            return max(1, int(override))
        mt = max(1, int(settings.max_tokens or 1))
        window = self._clamp_window(settings)
        if window:
            room = window - estimate_prompt_tokens(body) - CLAMP_MARGIN
            if room < mt:
                mt = max(room, min(mt, MIN_CLAMPED_MAX_TOKENS))
        return max(1, mt)

    def _request(self, settings: ModelSettings, system: str | list[SystemSegment] | None,
                 messages: list[Message], tools: list[ToolSpec], *, max_tokens: int | None = None) -> dict[str, Any]:
        """The ``/chat/completions`` JSON body (no I/O). ``max_tokens`` overrides
        the window clamp (used by the overflow retry)."""
        extra = dict(settings.extra or {})
        unknown = set(extra) - LOCAL_EXTRA_KEYS
        if unknown:
            _warn_once(f"extra:{sorted(unknown)}", "ModelSettings.extra keys %s are not used by the %s adapter and are "
                       "not sent to the server", sorted(unknown), self.name)
        fam = self.family_for(settings.model)
        alias = self._aliaser(fam)
        body: dict[str, Any] = {"model": self._wire_model(settings),
                                "messages": self._encode_messages(system, messages, fam, alias)}
        if tools:
            body["tools"] = [self._encode_tool(t, alias) for t in tools]
            body["tool_choice"] = self._tool_choice(extra, tools, alias)
            if self.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = bool(self.parallel_tool_calls)
        thinking_on, wire_effort = fam.reasoning_mode(settings.effort, settings.thinking,
                                                      override=extra.get("reasoning_effort"))
        reasoning = fam.reasoning_fields(thinking_on, wire_effort)
        if self._effort_in_kwargs and "reasoning_effort" in reasoning:
            # Equivalent chat-template form; 'none' never goes into the kwargs.
            effort = reasoning.pop("reasoning_effort")
            kwargs = dict(reasoning.get("chat_template_kwargs") or {})
            if thinking_on:
                kwargs["reasoning_effort"] = effort
            if fam.thinking_kwarg:
                kwargs[fam.thinking_kwarg] = bool(thinking_on)
            reasoning["chat_template_kwargs"] = kwargs
        body.update(reasoning)
        mt = self._max_tokens(settings, body, max_tokens)
        body["max_tokens"] = mt
        if thinking_on and fam.supports_thinking_budget:
            budget = _int(extra.get("thinking_budget")) or 0
            if budget <= 0:
                budget = min(mt // 2, DEFAULT_THINKING_BUDGET_CAP)
            if budget >= mt:
                budget = mt // 2
            body["thinking_token_budget"] = max(1, budget)
        sampling = fam.sampling(thinking_on)
        override = extra.get("sampling")
        if isinstance(override, dict):
            for k, v in override.items():
                if v is None:
                    sampling.pop(k, None)
                else:
                    sampling[k] = v
        if settings.temperature is not None:
            sampling["temperature"] = float(settings.temperature)
        top_k = sampling.get("top_k")
        if top_k is not None and (_int(top_k) or 0) <= 0:
            _warn_once("top_k", "sampling top_k=%s disables top-k sampling (implicated in Qwen reasoning loops); "
                       "not sent", top_k)
            sampling.pop("top_k")
        body.update(sampling)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        if self.extra_body:
            _merge(body, self.extra_body)
        return body

    # ------------------------------------------------------------------ transport

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._http is None or self._http_loop is not loop or self._http.is_closed:
            limits = httpx.Limits(max_connections=(self.max_concurrency + 8) if self.max_concurrency else None,
                                  max_keepalive_connections=64)
            timeout = httpx.Timeout(connect=self.timeout_s, read=self.read_timeout_s,
                                    write=max(60.0, self.timeout_s), pool=None)
            self._http = httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=self.trust_env)
            self._http_loop = loop
        return self._http

    @contextlib.asynccontextmanager
    async def _limiter(self) -> AsyncIterator[None]:
        if not self.max_concurrency:
            yield
            return
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.max_concurrency)
            self._sem_loop = loop
        async with self._sem:
            yield

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "text/event-stream, application/json", **self.extra_headers}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _route(self, extra: dict[str, Any]) -> tuple[str, dict[str, str]]:
        """``(api_base, headers)`` for a request: sticky per session key
        (``extra['session_key']``, else ``agent_name``)."""
        key = extra.get("session_key") or extra.get("agent_name")
        headers = self._headers()
        apis = self.api_bases
        n_urls = len(apis)
        h = stable_hash(str(key)) if key else None
        if n_urls > 1:
            idx = (h if h is not None else next(self._rr)) % n_urls
            api = apis[idx]
        else:
            api = apis[0]
        if self.data_parallel_size > 1 and self.routing == "header" and h is not None:
            rank = (h // n_urls if n_urls > 1 else h) % self.data_parallel_size
            headers["X-data-parallel-rank"] = str(rank)
        return api, headers

    async def _get(self, url: str, *, timeout: float | None = None) -> httpx.Response:
        t = httpx.Timeout(timeout or self.timeout_s, connect=min(self.timeout_s, timeout or self.timeout_s))
        return await self._client().get(url, headers=self._headers(), timeout=t)

    async def _count_prompt_tokens(self, api: str, headers: dict[str, str], body: dict[str, Any]) -> int | None:
        """Exact prompt size of a chat request via the server's ``POST /tokenize``
        (vLLM; same messages, tools and chat-template kwargs), or None when the
        server has no such endpoint. Used when an overflow error carries only a
        lower bound."""
        kwargs = dict(body.get("chat_template_kwargs") or {})
        effort = body.get("reasoning_effort")
        if effort is not None:  # what vLLM itself passes to the template for top-level reasoning_effort
            kwargs.setdefault("enable_thinking", effort != "none")
            if effort != "none":
                kwargs.setdefault("reasoning_effort", effort)
        req: dict[str, Any] = {"model": body.get("model"), "messages": body.get("messages") or [],
                               "add_generation_prompt": True}
        if body.get("tools"):
            req["tools"] = body["tools"]
        if kwargs:
            req["chat_template_kwargs"] = kwargs
        try:
            r = await self._client().post(self._root_of(api) + "/tokenize", json=req, headers=headers,
                                          timeout=httpx.Timeout(60.0, connect=self.timeout_s))
        except httpx.HTTPError:
            return None
        if r.status_code >= 400:
            return None
        d = _safe_json(r.content)
        if not isinstance(d, dict):
            return None
        n = _int(d.get("count"))
        if n is None and isinstance(d.get("tokens"), list):
            n = len(d["tokens"])
        return n if n and n > 0 else None

    # ------------------------------------------------------------------ discovery

    def _root_of(self, api: str) -> str:
        for a, r in self._bases:
            if a == api:
                return r
        return _split_base_url(api)[1]

    async def _discover(self, api: str, *, raise_errors: bool) -> list[dict[str, Any]]:
        """GET ``{api}/models`` and cache served ids, ``max_model_len`` and roots."""
        try:
            resp = await self._get(api + "/models", timeout=30.0)
        except httpx.TransportError as exc:
            if raise_errors:
                raise ProviderError(f"{self.name}: cannot reach {redact_url(api)}/models "
                                    f"({type(exc).__name__}: {exc}); "
                                    f"{serve_hint(self.name)}") from None
            raise
        if resp.status_code >= 400:
            if raise_errors:
                msg = _error_info(_safe_json(resp.content))[0]
                raise ProviderError(f"{self.name}: GET {redact_url(api)}/models returned HTTP {resp.status_code}: "
                                    f"{msg}")
            return []
        payload = _safe_json(resp.content)
        data = payload.get("data") if isinstance(payload, dict) else None
        models = [d for d in (data or []) if isinstance(d, dict) and d.get("id")]
        for d in models:
            mid = str(d["id"])
            meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
            # vLLM: max_model_len; llama.cpp: meta.n_ctx (the per-slot window, not n_ctx_train)
            mlen = _int(d.get("max_model_len") or d.get("context_length") or d.get("max_context_length")
                        or meta.get("n_ctx"))
            if mlen:
                self._max_len[mid] = mlen
            if d.get("root"):
                self._roots[mid] = str(d["root"])
        self._discovered = True
        return models

    async def _auto_discover_once(self) -> None:
        if self._discovered or self._discovering or not self.auto_discover:
            return
        self._discovering = True  # concurrent first calls do not all ask
        try:
            await self._discover(self.api_bases[0], raise_errors=False)
            self._discovered = True  # any HTTP answer: do not ask again
        except httpx.TransportError:
            pass  # server down: the request itself reports it
        except Exception:  # noqa: BLE001 - discovery is best effort
            log.debug("model discovery failed", exc_info=True)
            self._discovered = True
        finally:
            self._discovering = False

    async def prepare(self, models: Iterable[str] | None = None, *, wait_s: float | None = None) -> None:
        """Check every server (``GET /health``, waiting up to ``wait_s`` /
        ``wait_ready_s`` for it to come up) and discover the served models.

        ``models`` (e.g. the configured tier models) must be served; without it
        the ``served_model_name`` / ``model`` option is checked when set.
        Raises :class:`ProviderError` with a fix hint.
        """
        problem = self.check_credentials()
        if problem:
            raise ProviderError(problem)
        wait = self.wait_ready_s if wait_s is None else float(wait_s)
        deadline = time.monotonic() + max(0.0, wait)
        for api, root in self._bases:
            while True:
                try:
                    h = await self._get(root + "/health", timeout=30.0)
                    if h.status_code < 400 or h.status_code in (404, 405):
                        break  # healthy, or a server without /health
                    why = f"GET {redact_url(root)}/health returned HTTP {h.status_code}"
                except httpx.TransportError as exc:
                    why = f"cannot connect to {redact_url(root)} ({type(exc).__name__}: {exc})"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderError(f"{self.name} server not ready: {why}; {serve_hint(self.name)}")
                await asyncio.sleep(min(5.0, remaining))
            served = await self._discover(api, raise_errors=True)
            ids = [str(d["id"]) for d in served]
            if self.served_model_name:
                wanted = [self.served_model_name]
            else:
                wanted = [m for m in (models if models is not None else [self.model]) if m]
            missing = sorted({m for m in wanted if m not in ids})
            if missing and ids:
                raise ProviderError(
                    f"{self.name}: model(s) {missing} are not served by {redact_url(api)}; served: {ids}. Set "
                    f"models.<tier>.model (or --model) to a served name, or start the server with "
                    f"--served-model-name {missing[0]}")
            await self._check_data_parallel(root)
        self._prepared = True

    async def _dp_engines(self, root: str) -> int | None:
        """Data-parallel engines behind ``root``: the distinct ``engine="k"`` labels of
        vLLM's ``/metrics`` (None when the server exposes no such metrics)."""
        try:
            r = await self._get(root + "/metrics", timeout=10.0)
        except httpx.HTTPError:
            return None
        if r.status_code >= 400:
            return None
        engines = set(_DP_ENGINE_RE.findall(r.text or ""))
        return len(engines) or None

    async def _check_data_parallel(self, root: str) -> None:
        """Fail at startup (not mid-run) when ``data_parallel_size`` exceeds the server's
        data-parallel engines: vLLM rejects ``X-data-parallel-rank`` >= its
        ``--data-parallel-size`` with HTTP 400, i.e. for every session hashed there."""
        if self.routing != "header":
            return
        n = await self._dp_engines(root)
        if n is None or n == self.data_parallel_size:
            return
        where = redact_url(root)
        if self.data_parallel_size > n:
            raise ProviderError(
                f"{self.name}: provider.options.data_parallel_size is {self.data_parallel_size}, but the server at "
                f"{where} runs {n} data-parallel engine(s) (/metrics), so X-data-parallel-rank values >= {n} would be "
                f"rejected (HTTP 400) for those agent sessions. Set data_parallel_size to {n} (the local-* profiles "
                f"read it from VBT_LLM_DP_SIZE: export VBT_LLM_DP_SIZE={n}) or restart the server with "
                f"--data-parallel-size {self.data_parallel_size}")
        _warn_once(f"dp:{where}:{n}", "%s: the server at %s runs %d data-parallel engines but "
                   "provider.options.data_parallel_size is %d: sessions are pinned to ranks below %d only (set "
                   "VBT_LLM_DP_SIZE=%d for sticky routing over every replica)", self.name, where, n,
                   self.data_parallel_size, self.data_parallel_size, n)

    async def list_models(self) -> list[dict[str, Any]]:
        """Models served by every configured server (``GET /v1/models``)."""
        problem = self.check_credentials()
        if problem:
            raise ProviderError(problem)
        out: list[dict[str, Any]] = []
        for api in self.api_bases:
            for d in await self._discover(api, raise_errors=True):
                out.append({"id": d.get("id"), "root": d.get("root"),
                            "max_model_len": self._max_len.get(str(d.get("id"))) or d.get("max_model_len"),
                            "owned_by": d.get("owned_by"), "base_url": api})
        return out

    async def health(self) -> bool:
        """True when every configured server answers ``GET /health`` with 2xx."""
        if self.check_credentials():
            return False
        for _, root in self._bases:
            try:
                r = await self._get(root + "/health", timeout=10.0)
            except httpx.TransportError:
                return False
            if r.status_code >= 300:
                return False
        return True

    async def reachable(self, timeout_s: float = 3.0) -> list[tuple[str, str | None]]:
        """``(server, problem)`` per configured server: one bounded ``GET /health``
        (``vbt doctor`` without ``--smoke``). ``problem`` is None when the server
        answered at all (any HTTP status), else why it could not be reached."""
        out: list[tuple[str, str | None]] = []
        for _, root in self._bases:
            try:
                await self._get(root + "/health", timeout=timeout_s)
                out.append((redact_url(root), None))
            except httpx.TransportError as exc:
                out.append((redact_url(root), f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__))
        return out

    async def server_info(self) -> dict[str, Any]:
        """Engine version, served models and context length (never raises)."""
        info: dict[str, Any] = {"provider": self.name, "base_urls": [redact_url(u) for u in self.api_bases],
                                "family": self.family_for(self.served_model_name or self.model).name,
                                "served_model_name": self.served_model_name, "models": [], "max_model_len": None,
                                "version": None}
        if self.check_credentials():
            info["error"] = self.check_credentials()
            return info
        try:
            info["models"] = await self.list_models()
        except Exception as exc:  # noqa: BLE001
            info["error"] = f"{type(exc).__name__}: {exc}"
        model = self.served_model_name or self.model
        if not model and len(info["models"]) == 1:
            model = info["models"][0].get("id")
        info["family"] = self.family_for(model).name
        info["max_model_len"] = self._server_window(model)
        try:
            r = await self._get(self._bases[0][1] + "/version", timeout=10.0)
            if r.status_code < 400:
                v = _safe_json(r.content)
                info["version"] = v.get("version") if isinstance(v, dict) else (str(v)[:200] if v else None)
        except Exception:  # noqa: BLE001 - optional endpoint
            pass
        return info

    async def aclose(self) -> None:
        client, self._http, self._http_loop = self._http, None, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()

    # ------------------------------------------------------------------ errors

    async def _http_error(self, status: int, raw: bytes, headers: Any, api: str, model: str) -> Exception:
        payload = _safe_json(raw)
        msg, etype, _code, err = _error_info(payload if payload is not None else raw.decode("utf-8", "replace"))
        where = f"{self.name} server at {redact_url(api)}"
        if status in RETRYABLE_STATUS:
            return RetryableProviderError(f"{where} returned HTTP {status} ({etype or 'transient'}): {msg}",
                                          retry_after=_retry_after(headers), status=status)
        if status == 413 or is_overflow_message(msg, etype):
            return _Overflow(f"HTTP {status}: {msg}", overflow_info(msg, err))
        if status == 404:
            ids: list[str] = []
            with contextlib.suppress(Exception):
                ids = [str(d["id"]) for d in await self._discover(api, raise_errors=False)]
            if ids:
                hint = (f"; served models: {ids}. Set models.<tier>.model (or provider.options.served_model_name) to "
                        "one of them")
            else:
                hint = "; check provider.options.base_url (it should end in /v1)"
            return ProviderError(f"{where}: model {model!r} not found (HTTP 404: {msg}){hint}")
        if status in (401, 403):
            return ProviderError(f"{where} refused the request (HTTP {status}: {msg}); set provider.options.api_key "
                                 "or VBT_LLM_API_KEY to the server's --api-key")
        if status == 400 and "data_parallel_rank" in msg:
            return ProviderError(f"{where} rejected the X-data-parallel-rank header (HTTP 400: {msg}); "
                                 f"provider.options.data_parallel_size ({self.data_parallel_size}) must equal the "
                                 "server's --data-parallel-size")
        if status in (400, 422) and "reasoning_effort" in msg and not _TEMPLATE_BUG_RE.search(msg):
            return _EffortRejected(f"{where} rejected the request (HTTP {status}): {msg}")
        if _TEMPLATE_BUG_RE.search(msg):
            return ProviderError(f"{where}: the chat template rejected the request (HTTP {status}: {msg}); this is "
                                 "a harness adapter bug, please report it with the request log")
        return ProviderError(f"{where} rejected the request (HTTP {status}{', ' + etype if etype else ''}): {msg}")

    def _stream_error(self, chunk: Any) -> Exception:
        msg, etype, code, err = _error_info(chunk)
        if is_overflow_message(msg, etype):
            return _Overflow(f"error event: {msg}", overflow_info(msg, err))
        if code is not None and 400 <= code < 500 and code not in RETRYABLE_STATUS:
            if _TEMPLATE_BUG_RE.search(msg):
                return ProviderError(f"{self.name}: the chat template rejected the request ({msg}); this is a "
                                     "harness adapter bug")
            return ProviderError(f"{self.name}: error event in the stream ({etype or code}): {msg}")
        return RetryableProviderError(f"{self.name}: error event mid-stream ({etype or code or 'error'}): {msg}",
                                      status=code if code and code >= 400 else None)

    # ------------------------------------------------------------------ decode

    @staticmethod
    def _reasoning_of(delta: dict[str, Any], fam: ModelFamily, st: _StreamState) -> str:
        fields = list(fam.reasoning_response_fields) + [f for f in ("reasoning", "reasoning_content")
                                                        if f not in fam.reasoning_response_fields]
        if st.reasoning_field and st.reasoning_field in fields:
            fields.remove(st.reasoning_field)
            fields.insert(0, st.reasoning_field)
        for f in fields:
            v = delta.get(f)
            if isinstance(v, str) and v:
                st.reasoning_field = st.reasoning_field or f
                return v
        return ""

    @staticmethod
    def _content_of(c: Any) -> str:
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
        return ""

    @staticmethod
    def _absorb_tool_delta(tc: dict[str, Any], st: _StreamState, position: int | None = None) -> None:
        idx = _int(tc.get("index"))
        if idx is None:
            if position is not None:
                idx = position
            elif tc.get("id") and all(s.id != tc["id"] for s in st.slots.values()):
                idx = (max(st.slots) + 1) if st.slots else 0
            else:
                idx = max(st.slots) if st.slots else 0
        slot = st.slots.setdefault(idx, _ToolSlot(idx))
        if tc.get("id") and not slot.id:
            slot.id = str(tc["id"])
        fn = tc.get("function") or {}
        name = fn.get("name")
        if name:
            if not slot.name or name.startswith(slot.name):
                slot.name = name
            elif name != slot.name:
                slot.name += name
        args = fn.get("arguments")
        if args is not None:
            slot.arguments += args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)

    def _absorb_chunk(self, chunk: dict[str, Any], st: _StreamState, fam: ModelFamily) -> tuple[str, str]:
        """Fold one streamed chunk into ``st``; returns ``(text_delta, reasoning_delta)``."""
        st.chunks += 1
        st.model = chunk.get("model") or st.model
        st.request_id = st.request_id or chunk.get("id")
        if isinstance(chunk.get("usage"), dict):
            st.usage = chunk["usage"]
        text, reasoning = "", ""
        for ch in chunk.get("choices") or []:
            if not isinstance(ch, dict) or (_int(ch.get("index")) or 0) != 0:
                continue
            delta = ch.get("delta")
            if not isinstance(delta, dict):
                delta = ch.get("message") if isinstance(ch.get("message"), dict) else {}
            r = self._reasoning_of(delta, fam, st)
            if r:
                st.reasoning.append(r)
                reasoning += r
            c = self._content_of(delta.get("content"))
            if c:
                st.text.append(c)
                text += c
            for tc in delta.get("tool_calls") or []:
                if isinstance(tc, dict):
                    self._absorb_tool_delta(tc, st)
            if ch.get("finish_reason"):
                st.finish = str(ch["finish_reason"])
        return text, reasoning

    def _absorb_message(self, payload: dict[str, Any], st: _StreamState, fam: ModelFamily) -> None:
        """A non-streamed ``chat.completion`` body."""
        st.model = payload.get("model") or st.model
        st.request_id = payload.get("id") or st.request_id
        if isinstance(payload.get("usage"), dict):
            st.usage = payload["usage"]
        choices = payload.get("choices") or []
        ch = choices[0] if choices and isinstance(choices[0], dict) else {}
        msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
        r = self._reasoning_of(msg, fam, st)
        if r:
            st.reasoning.append(r)
        c = self._content_of(msg.get("content"))
        if c:
            st.text.append(c)
        for i, tc in enumerate(msg.get("tool_calls") or []):
            if isinstance(tc, dict):
                self._absorb_tool_delta({k: v for k, v in tc.items() if k != "index"}, st, position=i)
        st.finish = str(ch.get("finish_reason")) if ch.get("finish_reason") else None
        st.done = True

    @staticmethod
    def _decode_call(slot: _ToolSlot, reverse: dict[str, str], seen: set[str]) -> ToolCall | None:
        name = slot.name.strip()
        if not name:
            _warn_once("noname", "a streamed tool call had no function name; dropped")
            return None
        name = reverse.get(name, name)
        cid = slot.id or f"call_{uuid.uuid4().hex[:24]}"
        if cid in seen:
            cid = f"{cid}_{uuid.uuid4().hex[:6]}"
        seen.add(cid)
        raw = slot.arguments
        if not raw.strip():
            return ToolCall(cid, name, {})
        try:
            val = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolCall(cid, name, {}, native={"invalid_arguments": raw, "error": f"invalid JSON: {exc}"})
        if isinstance(val, str):  # double-encoded arguments
            with contextlib.suppress(json.JSONDecodeError):
                inner = json.loads(val)
                if isinstance(inner, dict):
                    val = inner
        if not isinstance(val, dict):
            return ToolCall(cid, name, {}, native={
                "invalid_arguments": raw, "error": f"arguments must be a JSON object, got {type(val).__name__}"})
        return ToolCall(cid, name, val)

    def _cost(self, usage: Usage) -> float:
        p = self.pricing
        return (usage.input_tokens * p["input_per_mtok"] + usage.cache_read_tokens * p["cached_per_mtok"]
                + usage.output_tokens * p["output_per_mtok"]) / 1e6

    @staticmethod
    def _usage(raw: dict[str, Any] | None) -> Usage:
        u = raw or {}
        prompt = _int(u.get("prompt_tokens")) or 0
        completion = _int(u.get("completion_tokens")) or 0
        details = u.get("prompt_tokens_details")
        cached = (_int(details.get("cached_tokens")) or 0) if isinstance(details, dict) else 0
        cached = max(0, min(cached, prompt))
        return Usage(input_tokens=prompt - cached, output_tokens=completion, cache_read_tokens=cached)

    def _build_response(self, st: _StreamState, body: dict[str, Any], settings: ModelSettings, fam: ModelFamily,
                        tools: list[ToolSpec], reverse: dict[str, str]) -> ModelResponse:
        text = "".join(st.text)
        reasoning = "".join(st.reasoning)
        kwargs = body.get("chat_template_kwargs") or {}
        thinking_off = (body.get("reasoning_effort") == "none" or kwargs.get("enable_thinking") is False
                        or kwargs.get("thinking") is False)
        # When the request switched thinking off, the template already closed an empty think block, so a
        # stray </think> in the answer is model noise (seen with a 0.8B model on llama.cpp), not reasoning.
        if not reasoning and "</think>" in text and (not thinking_off or text.lstrip().startswith("<think>")):
            _warn_once(f"think:{self.name}", "%s returned <think> reasoning inside the content; start the server with "
                       "a reasoning parser (vLLM: --reasoning-parser qwen3)", self.name)
            head, _, tail = text.partition("</think>")
            reasoning, text = head.replace("<think>", "", 1), tail
        seen: set[str] = set()
        calls = [c for c in (self._decode_call(s, reverse, seen) for _, s in sorted(st.slots.items())) if c]
        # tool_choice 'none' forbids calls: llama.cpp then returns the model's attempted call as plain
        # text, which must stay text rather than become a call the caller did not allow.
        if (not calls and tools and body.get("tool_choice") != "none"
                and ("<tool_call>" in text or "<function=" in text)):
            extracted, rest = extract_text_tool_calls(text, tools, reverse)
            if extracted:
                _warn_once(f"textcalls:{self.name}", "%s returned tool calls inside the text; start vLLM with "
                           "--enable-auto-tool-choice --tool-call-parser qwen3_coder (extracted them)", self.name)
                calls, text = extracted, rest
        text = text.lstrip("\n").rstrip()
        reasoning = reasoning.strip()
        blocks: list[Any] = []
        if reasoning:
            blocks.append(ThinkingBlock(reasoning, provider=self.name,
                                        native={"field": st.reasoning_field or "reasoning", "family": fam.name}))
        if text:
            blocks.append(TextBlock(text))
        blocks.extend(calls)
        usage = self._usage(st.usage)
        fr = st.finish
        detail = None
        if fr in ("abort", "error"):
            # Checked before the tool calls: an aborted stream (engine shutdown/restart, KV transfer
            # failure) can hold half-streamed calls with truncated arguments, which must not become a
            # TOOL_USE turn (a bogus call/result pair in the history); the whole request is re-sent.
            raise RetryableProviderError(f"{self.name}: the server aborted the request (finish_reason={fr})")
        if fr == "length":
            window = self._server_window(settings.model) or self._clamp_window(settings)
            total = (_int((st.usage or {}).get("prompt_tokens")) or 0) + usage.output_tokens
            window_limited = int(body.get("max_tokens") or 0) < int(settings.max_tokens or 0)
            stop = StopReason.CONTEXT_EXCEEDED if (window and total >= window) or window_limited else \
                StopReason.MAX_TOKENS
        elif calls:
            stop = StopReason.TOOL_USE
        elif fr in (None, "stop", "eos", "end_turn", "stop_sequence", "tool_calls", "function_call"):
            stop = StopReason.END_TURN
        elif fr == "content_filter":
            stop, detail = StopReason.REFUSAL, "content_filter"
        else:
            stop, detail = StopReason.OTHER, f"finish_reason={fr}"
        served = st.model or body.get("model") or settings.model
        return ModelResponse(message=Message("assistant", blocks), stop_reason=stop, usage=usage, model=served,
                             cost_usd=self._cost(usage), stop_detail=detail, request_id=st.request_id,
                             served_model=served)

    # ------------------------------------------------------------------ calls

    async def _consume_sse(self, resp: httpx.Response, st: _StreamState, fam: ModelFamily,
                           on_text: TextCallback | None, on_thinking: ThinkingCallback | None) -> None:
        data_lines: list[str] = []
        event = ""

        async def dispatch() -> bool:
            nonlocal data_lines, event
            data, ev = "\n".join(data_lines), event
            data_lines, event = [], ""
            if not data.strip():
                return False
            if data.strip() == "[DONE]":
                st.done = True
                return True
            try:
                chunks = [json.loads(data)]
            except json.JSONDecodeError:
                chunks = []
                for piece in data.split("\n"):
                    with contextlib.suppress(json.JSONDecodeError):
                        chunks.append(json.loads(piece))
                if not chunks:
                    log.debug("%s: unparseable SSE data: %r", self.name, data[:200])
                    return False
            for chunk in chunks:
                if ev == "error" or (isinstance(chunk, dict) and (chunk.get("error") is not None
                                                                  or chunk.get("object") == "error")):
                    raise self._stream_error(chunk)
                if isinstance(chunk, dict):
                    t, r = self._absorb_chunk(chunk, st, fam)
                    await _emit(on_thinking, r)
                    await _emit(on_text, t)
            return False

        async for line in resp.aiter_lines():
            if line == "":
                if data_lines and await dispatch():
                    return
                continue
            if line.startswith(":"):
                continue  # comment / keep-alive
            if line.startswith("data:"):
                data_lines.append(line[5:][1:] if line[5:6] == " " else line[5:])
            elif line.startswith("event:"):
                event = line[6:].strip()
        if data_lines and await dispatch():
            return
        if not st.done and st.finish is None:
            raise RetryableProviderError(f"{self.name}: the stream ended before the response was complete "
                                         f"({st.chunks} chunks; connection dropped?)")

    async def _send(self, api: str, headers: dict[str, str], body: dict[str, Any], *, settings: ModelSettings,
                    fam: ModelFamily, tools: list[ToolSpec], reverse: dict[str, str],
                    on_text: TextCallback | None, on_thinking: ThinkingCallback | None) -> ModelResponse:
        st = _StreamState()
        url = api + "/chat/completions"
        try:
            async with self._client().stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    raw = await resp.aread()
                    raise await self._http_error(resp.status_code, raw, resp.headers, api, str(body.get("model")))
                ctype = (resp.headers.get("content-type") or "").lower()
                if "event-stream" in ctype or ("json" not in ctype and body.get("stream")):
                    await self._consume_sse(resp, st, fam, on_text, on_thinking)
                else:
                    raw = await resp.aread()
                    payload = _safe_json(raw)
                    if not isinstance(payload, dict):
                        raise RetryableProviderError(f"{self.name}: unparseable response body from {redact_url(url)}: "
                                                     f"{raw[:200]!r}")
                    if payload.get("error") is not None or payload.get("object") == "error":
                        raise self._stream_error(payload)
                    self._absorb_message(payload, st, fam)
                    await _emit(on_thinking, "".join(st.reasoning))
                    await _emit(on_text, "".join(st.text))
        except (ProviderError, _Overflow, _EffortRejected):
            raise
        except httpx.ConnectError as exc:
            raise RetryableProviderError(f"{self.name}: cannot connect to {redact_url(api)} "
                                         f"({type(exc).__name__}: {exc}); {serve_hint(self.name)}") from None
        except httpx.TimeoutException as exc:
            raise RetryableProviderError(f"{self.name}: timed out talking to {redact_url(api)} ({type(exc).__name__}; "
                                         f"read timeout {self.read_timeout_s}s between chunks)") from None
        except httpx.TransportError as exc:
            raise RetryableProviderError(f"{self.name}: connection to {redact_url(api)} dropped ({type(exc).__name__}: "
                                         f"{exc})") from None
        return self._build_response(st, body, settings, fam, tools, reverse)

    async def complete(
        self,
        *,
        settings: ModelSettings,
        system: str | list[SystemSegment],
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback | None = None,
        on_thinking: ThinkingCallback | None = None,
    ) -> ModelResponse:
        problem = self.check_credentials()
        if problem:
            raise ProviderError(problem)
        tools = list(tools or [])
        await self._auto_discover_once()
        fam = self.family_for(settings.model)
        body = self._request(settings, system, messages, tools)
        api, headers = self._route(settings.extra or {})
        reverse = self._reverse_aliases(tools, messages, fam)
        retried = False
        known_prompt: tuple[int, str] | None = None
        async with self._limiter():
            while True:
                try:
                    return await self._send(api, headers, body, settings=settings, fam=fam, tools=tools,
                                            reverse=reverse, on_text=on_text, on_thinking=on_thinking)
                except _EffortRejected as rej:
                    if self._effort_in_kwargs or "reasoning_effort" not in body:
                        raise ProviderError(str(rej)) from None
                    _warn_once(f"effort-kwargs:{self.name}", "%s: the server rejected top-level reasoning_effort "
                               "(%s); sending it as chat_template_kwargs.reasoning_effort from now on", self.name,
                               str(rej)[:300])
                    self._effort_in_kwargs = True
                    body = self._request(settings, system, messages, tools, max_tokens=body.get("max_tokens"))
                    continue
                except _Overflow as ov:
                    if ov.limit:
                        self._learned_window = ov.limit
                    if ov.prompt is not None:
                        prompt, how = ov.prompt, ""
                    elif known_prompt is not None:
                        prompt, how = known_prompt
                    elif ov.limit and not retried:
                        # vLLM (0.30) reports only a lower bound: count the prompt exactly via
                        # /tokenize, else estimate it (never below the server's bound).
                        counted = await self._count_prompt_tokens(api, headers, body)
                        if counted is not None:
                            prompt, how = counted, "counted via /tokenize"
                        else:
                            prompt, how = max(ov.prompt_min or 0, estimate_prompt_tokens(body)), "estimated"
                        known_prompt = (prompt, how)
                    else:
                        prompt, how = None, ""
                    room = (ov.limit - prompt) if (ov.limit and prompt is not None) else None
                    if not retried and room is not None and room >= OVERFLOW_MIN_ROOM:
                        new_max = room - OVERFLOW_RETRY_MARGIN
                        if new_max < int(body.get("max_tokens") or 0):
                            log.info("%s: request overflowed the %d-token window (prompt %d%s); retrying with "
                                     "max_tokens=%d", self.name, ov.limit, prompt, f", {how}" if how else "",
                                     new_max)
                            body = self._request(settings, system, messages, tools, max_tokens=new_max)
                            retried = True
                            continue
                    parts = [f"limit {ov.limit}" if ov.limit else ""]
                    if prompt is not None:
                        parts.append(f"prompt {prompt}{' ' + how if how else ''}")
                    elif ov.prompt_min:
                        parts.append(f"prompt >= {ov.prompt_min}")
                    facts = ", ".join(p for p in parts if p)
                    raise ContextOverflowError(f"{self.name}: request does not fit the model's context window"
                                               f"{' (' + facts + ')' if facts else ''}: {ov}") from None


def _safe_json(raw: bytes | str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        return None


def create(registered_name: str = "vllm", /, **options: Any) -> OpenAICompatProvider:
    """Factory used by the provider registry: the registered name (``vllm``,
    ``sglang``, ...) is the provider's name; a ``name`` option is ignored."""
    options.pop("name", None)
    return OpenAICompatProvider(name=registered_name, **options)


__all__ = [
    "FAMILIES", "LOCAL_EXTRA_KEYS", "OpenAICompatProvider", "OverflowInfo", "PROVIDER_NAMES", "create",
    "estimate_prompt_tokens", "extract_text_tool_calls", "is_overflow_message", "overflow_info", "parse_overflow",
    "stable_hash", "tool_alias",
]
