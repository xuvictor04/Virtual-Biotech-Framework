"""Claude adapter (Anthropic Messages API) for the provider interface.

The paper ran on the Claude Agent SDK; this harness owns its own agent loop so
it stays provider-agnostic, and only this file knows about the Anthropic API.

Sources for the model facts below (prices, context windows, beta names,
thinking rules): the ``claude-api`` skill's model and pricing reference
(cached 2026-09-25) and its model-migration, prompt-caching, error-code and
context-editing notes. Prices can be overridden per model with
``provider.options.prices``.
"""

from __future__ import annotations

import asyncio
import email.utils
import importlib
import inspect
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from .base import (
    ContextOverflowError,
    DocumentPart,
    ImagePart,
    LLMProvider,
    Message,
    ModelResponse,
    ModelSettings,
    OpaqueBlock,
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
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model facts
# ---------------------------------------------------------------------------

# USD per million tokens. ``cache_write_5m`` / ``cache_write_1h`` are prompt-cache
# writes with the 5-minute / 1-hour TTL (1.25x / 2x input); ``cache_read`` is
# 0.1x input except Claude Fable 5.1 / Mythos 5.1 (0.025x) and Claude Opus 5.5
# (0.05x). Values from the claude-api skill pricing reference (2026-09-25).
PRICES: dict[str, dict[str, float]] = {
    "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_read": 0.25, "cache_write_5m": 12.5, "cache_write_1h": 20.0},
    "claude-mythos-5-1": {"input": 10.0, "output": 50.0, "cache_read": 0.25, "cache_write_5m": 12.5, "cache_write_1h": 20.0},
    "claude-fable-5": {"input": 10.0, "output": 50.0, "cache_read": 1.00, "cache_write_5m": 12.5, "cache_write_1h": 20.0},
    "claude-mythos-5": {"input": 10.0, "output": 50.0, "cache_read": 1.00, "cache_write_5m": 12.5, "cache_write_1h": 20.0},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write_5m": 5.0, "cache_write_1h": 8.0},
    "claude-opus-5": {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write_5m": 6.25, "cache_write_1h": 10.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write_5m": 6.25, "cache_write_1h": 10.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write_5m": 6.25, "cache_write_1h": 10.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "cache_read": 0.50, "cache_write_5m": 6.25, "cache_write_1h": 10.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20, "cache_write_5m": 2.5, "cache_write_1h": 4.0},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20, "cache_write_5m": 2.5, "cache_write_1h": 4.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write_5m": 3.75, "cache_write_1h": 6.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write_5m": 3.75, "cache_write_1h": 6.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.10, "cache_write_5m": 1.25, "cache_write_1h": 2.0},
}
PRICE_KEYS = ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")
# Server-side web search: $10 per 1,000 searches, on top of tokens.
WEB_SEARCH_USD_PER_REQUEST = 10.0 / 1000

# Context windows (tokens), from the skill's model table. Claude Sonnet 4.5 is
# not listed there; profiles set ``context_window_tokens`` for it explicitly.
CONTEXT_WINDOWS: dict[str, int] = {
    "claude-fable-5-1": 1_000_000, "claude-mythos-5-1": 1_000_000,
    "claude-fable-5": 1_000_000, "claude-mythos-5": 1_000_000,
    "claude-opus-5-5": 1_000_000, "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000, "claude-opus-4-7": 1_000_000, "claude-opus-4-6": 1_000_000,
    "claude-sonnet-5-5": 1_000_000, "claude-sonnet-5": 1_000_000, "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5": 200_000,
}

# Models that take adaptive thinking + the effort parameter.
_ADAPTIVE = ("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
             "claude-opus-4-6", "claude-sonnet-5", "claude-sonnet-4-6")
# Models where thinking cannot be disabled (omit the parameter to get adaptive).
_THINKING_ALWAYS_ON = ("claude-fable-5", "claude-mythos-5", "claude-opus-5-5")
# Families that accept the server-side refusal fallback (``fallbacks: "default"``).
FALLBACK_FAMILIES = ("claude-fable-5", "claude-opus-5", "claude-sonnet-5-5")
# Models on which sampling parameters (temperature, ...) are rejected.
_NO_SAMPLING = ("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
                "claude-sonnet-5")
# Models whose thinking blocks are bound to the conversation prefix ("preserved
# thinking"): editing earlier turns invalidates every later thinking block.
_HISTORY_BOUND_THINKING = ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5")
# Pre-adaptive Claude 4.x models (budget thinking): need the interleaved-thinking
# beta to think between tool calls.
_BUDGET_THINKING_4X = ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4")

FALLBACK_BETA = "server-side-fallback-2026-07-01"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"
CLEAR_TOOL_USES_EDIT = "clear_tool_uses_20250919"

# ModelSettings.extra keys this adapter consumes locally; nothing in ``extra``
# is ever forwarded to the API.
LOCAL_EXTRA_KEYS = frozenset({"agent_name", "context_window_tokens", "thinking_budget", "context_management",
                              "prompt_cache", "session_key"})  # session_key: replica routing of local servers

_STOP_MAP = {
    "end_turn": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
    "pause_turn": StopReason.PAUSE,
    "model_context_window_exceeded": StopReason.CONTEXT_EXCEEDED,
}

_CONTEXT_RE = re.compile(
    r"prompt is too long|context window|context limit|context length|exceeds? the (?:maximum )?context|"
    r"too many (?:input )?tokens|input length and `?max_tokens`? exceed", re.I)
_RETRYABLE_TYPES = {"overloaded_error": 529, "rate_limit_error": 429, "api_error": 500, "timeout_error": 504}
_FATAL_TYPES = {"invalid_request_error", "authentication_error", "permission_error", "not_found_error",
                "billing_error", "request_too_large"}

_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(msg, *args)


def normalize_model(model: str) -> str:
    """Strip platform prefixes and snapshot/date suffixes: ``anthropic.claude-x``,
    ``claude-x@20250929``, ``claude-x-20250929``, ``claude-x[1m]`` -> ``claude-x``."""
    m = (model or "").strip()
    m = re.sub(r"^(?:[a-z]+\.)?anthropic\.", "", m)
    m = re.sub(r"\[[^\]]*\]$", "", m)
    m = re.sub(r"@.*$", "", m)
    m = re.sub(r"-v\d+(?::\d+)?$", "", m)
    m = re.sub(r"-\d{8}$", "", m)
    return m


def _family(model: str, prefixes: tuple[str, ...]) -> bool:
    """True if ``model`` is one of ``prefixes`` or a variant of one (``p-...``)."""
    m = normalize_model(model)
    return any(m == p or m.startswith(p + "-") for p in prefixes)


def _complete_price(entry: dict[str, Any], base: dict[str, float] | None = None) -> dict[str, float]:
    """Fill a (possibly partial) price entry: missing keys come from ``base`` or
    from the standard multipliers on ``input``."""
    out = dict(base or {})
    out.update({k: float(v) for k, v in entry.items() if k in PRICE_KEYS and v is not None})
    if "input" not in out:
        raise ValueError(f"price entry needs an 'input' price: {entry}")
    if "output" not in out:
        out["output"] = out["input"] * 5
    out.setdefault("cache_read", out["input"] * 0.1)
    out.setdefault("cache_write_5m", out["input"] * 1.25)
    out.setdefault("cache_write_1h", out["input"] * 2.0)
    return out


def price_for(model: str, prices: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    """Price entry for ``model``: exact (normalised) match, else the longest
    family prefix, else the most expensive known model. Warns once per model
    that is not an exact match."""
    table = prices if prices is not None else PRICES
    m = normalize_model(model)
    if m in table:
        return table[m]
    for key in sorted(table, key=len, reverse=True):
        if m.startswith(key + "-"):
            _warn_once(f"price:{model}", "no price listed for model %r; using %r prices (set "
                       "provider.options.prices to override)", model, key)
            return table[key]
    worst = max(table, key=lambda k: (table[k]["output"], table[k]["input"]))
    _warn_once(f"price:{model}", "unknown model %r: pricing it as %r (the most expensive listed) so budgets stay "
               "conservative; set provider.options.prices to override", model, worst)
    return table[worst]


def cost_usd(model: str, usage: Usage, prices: dict[str, dict[str, float]] | None = None) -> float:
    """USD cost of ``usage`` on ``model`` including cache reads/writes and the
    per-request web-search fee."""
    p = price_for(model, prices)
    w1h = max(0, usage.cache_write_1h_tokens)
    w5m = max(0, usage.cache_write_tokens - w1h)
    tokens = (usage.input_tokens * p["input"] + usage.output_tokens * p["output"]
              + usage.cache_read_tokens * p["cache_read"] + w5m * p["cache_write_5m"]
              + w1h * p["cache_write_1h"]) / 1e6
    searches = (usage.server_tool_requests or {}).get("web_search", 0)
    return tokens + searches * WEB_SEARCH_USD_PER_REQUEST


def _usage_from(u: Any) -> Usage:
    """Normalise an SDK usage object (Usage, BetaUsage or an iteration entry)."""
    if u is None:
        return Usage()
    cc = getattr(u, "cache_creation", None)
    w1h = int(getattr(cc, "ephemeral_1h_input_tokens", 0) or 0) if cc is not None else 0
    w5m = int(getattr(cc, "ephemeral_5m_input_tokens", 0) or 0) if cc is not None else 0
    writes = int(getattr(u, "cache_creation_input_tokens", 0) or 0) or (w5m + w1h)
    servers: dict[str, int] = {}
    stu = getattr(u, "server_tool_use", None)
    if stu is not None:
        for key, attr in (("web_search", "web_search_requests"), ("web_fetch", "web_fetch_requests")):
            n = int(getattr(stu, attr, 0) or 0)
            if n:
                servers[key] = n
    return Usage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=max(writes, w1h),
        cache_write_1h_tokens=w1h,
        server_tool_requests=servers,
    )


def _retry_after(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
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
            dt = email.utils.parsedate_to_datetime(ra)
            return max(0.0, dt.timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def _error_message(exc: Any) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(getattr(exc, "message", None) or exc)


def _error_type(exc: Any) -> str | None:
    t = getattr(exc, "type", None)
    if t:
        return str(t)
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"].get("type")
    return None


def _transport_errors() -> tuple[type[BaseException], ...]:
    errs: list[type[BaseException]] = []
    for name in ("httpx2", "httpx"):
        try:
            errs.append(importlib.import_module(name).TransportError)
        except Exception:  # noqa: BLE001 - optional
            pass
    return tuple(errs)


async def _emit(cb: TextCallback | None, text: str) -> None:
    if cb is None or not text:
        return
    r = cb(text)
    if inspect.isawaitable(r):
        await r


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 4,
        refusal_fallback: bool = True,
        web_search_model: str = "claude-sonnet-5",
        prompt_caching: bool = True,
        prices: dict[str, dict[str, float]] | None = None,
        fallback_families: list[str] | tuple[str, ...] | None = None,
        context_editing: bool | dict[str, Any] | None = None,
        system_cache_ttl: str | None = None,
        timeout: float | None = None,
        **unknown: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise ProviderError("pip install 'vbt-harness[anthropic]' to use the Claude provider") from exc
        if unknown:
            log.warning("AnthropicProvider: ignoring unknown provider options %s", sorted(unknown))
        self._anthropic = anthropic
        self._api_key = (api_key or "").strip() or None
        kwargs: dict[str, Any] = {"max_retries": int(max_retries)}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if base_url:
            kwargs["base_url"] = base_url
        if timeout:
            kwargs["timeout"] = float(timeout)
        self.client = anthropic.AsyncAnthropic(**kwargs)
        self.refusal_fallback = refusal_fallback
        self.fallback_families = tuple(fallback_families) if fallback_families else FALLBACK_FAMILIES
        self.web_search_model = web_search_model
        self.prompt_caching = prompt_caching
        self.context_editing = context_editing
        self.system_cache_ttl = system_cache_ttl
        self.prices: dict[str, dict[str, float]] = {k: dict(v) for k, v in PRICES.items()}
        for model, entry in (prices or {}).items():
            key = normalize_model(model)
            self.prices[key] = _complete_price(entry, self.prices.get(key))
        self._transport_errors = _transport_errors()

    # ------------------------------------------------------------------ facts

    def cost(self, model: str, usage: Usage) -> float:
        return cost_usd(model, usage, self.prices)

    def capabilities(self, model: str | None = None) -> ProviderCapabilities:
        return ProviderCapabilities(
            images=True, documents=True, web_search=True, server_context_management=True,
            history_bound_thinking=bool(model) and _family(model, _HISTORY_BOUND_THINKING),
        )

    def context_window(self, model: str) -> int | None:
        return CONTEXT_WINDOWS.get(normalize_model(model))

    def check_credentials(self) -> str | None:
        """Problem text when no usable credential is configured (no network)."""
        if self._api_key:
            return None
        token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if token and token.strip():
            return None
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key is not None:
            if key.strip():
                return None
            return ("ANTHROPIC_API_KEY is set but blank (an empty value, e.g. from .env, counts as missing and "
                    "shadows other credential sources); set a real key or remove the empty entry")
        wif = ("ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_SERVICE_ACCOUNT_ID")
        if all(os.environ.get(v) for v in wif) and (os.environ.get("ANTHROPIC_IDENTITY_TOKEN_FILE")
                                                    or os.environ.get("ANTHROPIC_IDENTITY_TOKEN")):
            return None
        if os.environ.get("ANTHROPIC_PROFILE"):
            return None
        cfg_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        if (Path(cfg_home) / "anthropic").is_dir():
            return None  # `ant auth login` profile on disk
        return ("no Anthropic credentials found: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN, or run "
                "`ant auth login`)")

    # ------------------------------------------------------------------ encode

    def _encode_part(self, p: Any) -> dict[str, Any] | None:
        if isinstance(p, TextBlock):
            return {"type": "text", "text": p.text} if p.text else None
        if isinstance(p, ImagePart):
            return {"type": "image", "source": {"type": "base64", "media_type": p.media_type, "data": p.data_b64}}
        if isinstance(p, DocumentPart):
            d: dict[str, Any] = {"type": "document",
                                 "source": {"type": "base64", "media_type": p.media_type, "data": p.data_b64}}
            if p.title:
                d["title"] = p.title
            return d
        if isinstance(p, str):
            return {"type": "text", "text": p} if p else None
        raise TypeError(f"unknown tool-result part {p!r}")

    def _encode_block(self, block) -> dict[str, Any] | None:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text} if block.text else None
        if isinstance(block, ThinkingBlock):
            # Only our own thinking blocks, unchanged; others are dropped.
            return block.native if block.provider == self.name and block.native else None
        if isinstance(block, ToolCall):
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        if isinstance(block, ToolResult):
            if isinstance(block.content, str):
                content: Any = block.content or "(empty result)"
            else:
                content = [c for c in (self._encode_part(p) for p in block.content) if c is not None]
                if not content:
                    content = "(empty result)"
            return {"type": "tool_result", "tool_use_id": block.tool_call_id, "content": content,
                    "is_error": block.is_error}
        if isinstance(block, OpaqueBlock):
            return block.native if block.provider == self.name else None
        raise TypeError(f"unknown block {block!r}")

    def _encode_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            content = [c for c in (self._encode_block(b) for b in m.content) if c is not None]
            if not content:
                content = [{"type": "text", "text": "(no content)"}]
            out.append({"role": m.role, "content": content})
        return out

    def _encode_system(self, system: str | list[SystemSegment] | None,
                       cache: bool = True) -> str | list[dict[str, Any]] | None:
        if system is None:
            return None
        if isinstance(system, str):
            return system or None
        segs = [s if isinstance(s, SystemSegment) else SystemSegment(str(s)) for s in system]
        blocks: list[dict[str, Any]] = []
        last_cached = -1
        for s in segs:
            if not s.text:
                continue
            blocks.append({"type": "text", "text": s.text})
            if s.cache:
                last_cached = len(blocks) - 1
        if cache and self.prompt_caching and last_cached >= 0:
            cc: dict[str, Any] = {"type": "ephemeral"}
            if self.system_cache_ttl:
                cc["ttl"] = self.system_cache_ttl
            blocks[last_cached]["cache_control"] = cc
        return blocks or None

    def _context_management(self, settings: ModelSettings) -> dict[str, Any] | None:
        spec = (settings.extra or {}).get("context_management", self.context_editing)
        if not spec:
            return None
        if spec is True:
            return {"edits": [{"type": CLEAR_TOOL_USES_EDIT}]}
        if isinstance(spec, list):
            return {"edits": list(spec)}
        if isinstance(spec, dict):
            return dict(spec) if "edits" in spec else {"edits": [{"type": CLEAR_TOOL_USES_EDIT, **spec}]}
        return None

    def _request(self, settings: ModelSettings, system: str | list[SystemSegment] | None, messages,
                 tools) -> dict[str, Any]:
        """Keyword arguments for ``messages.stream``; ``betas`` (if any) selects
        the beta endpoint."""
        model = settings.model
        extra = settings.extra or {}
        unknown = set(extra) - LOCAL_EXTRA_KEYS
        if unknown:
            _warn_once(f"extra:{sorted(unknown)}", "ModelSettings.extra keys %s are not used by the Anthropic "
                       "adapter and are not sent to the API", sorted(unknown))
        req: dict[str, Any] = {"model": model, "max_tokens": settings.max_tokens,
                               "messages": self._encode_messages(messages)}
        use_cache = self.prompt_caching and extra.get("prompt_cache", True) is not False
        sys_payload = self._encode_system(system, cache=use_cache)
        if sys_payload is not None:
            req["system"] = sys_payload
        if tools:
            req["tools"] = [
                {"name": t.name, "description": t.description[:10000], "input_schema": t.input_schema}
                for t in tools
            ]
        if use_cache:
            req["cache_control"] = {"type": "ephemeral"}

        betas: list[str] = []
        adaptive = _family(model, _ADAPTIVE) or _family(model, _THINKING_ALWAYS_ON)
        if adaptive:
            # With thinking off we simply omit the parameter: newer models then
            # run their default, and lower effort is the recommended cost lever.
            if settings.thinking or _family(model, _THINKING_ALWAYS_ON):
                req["thinking"] = {"type": "adaptive", "display": "summarized"}
            if settings.effort:
                req["output_config"] = {"effort": settings.effort}
        elif settings.thinking:
            budget = int(extra.get("thinking_budget") or 8000)
            budget = max(1024, min(budget, settings.max_tokens - 1024))
            req["thinking"] = {"type": "enabled", "budget_tokens": budget}
            if tools and _family(model, _BUDGET_THINKING_4X):
                # Claude 4.x budget thinking only reasons between tool calls with this beta.
                betas.append(INTERLEAVED_THINKING_BETA)
        if settings.temperature is not None and not _family(model, _NO_SAMPLING) and "thinking" not in req:
            req["temperature"] = settings.temperature
        if self.refusal_fallback and _family(model, self.fallback_families):
            req["fallbacks"] = "default"
            betas.append(FALLBACK_BETA)
        cm = self._context_management(settings)
        if cm:
            req["context_management"] = cm
            betas.append(CONTEXT_MANAGEMENT_BETA)
        if betas:
            req["betas"] = betas
        return req

    # ------------------------------------------------------------------ decode

    def _decode(self, final, request_model: str) -> tuple[Message, Usage, list[dict[str, Any]] | None, bool]:
        content = list(final.content or [])
        last_fallback = max((i for i, b in enumerate(content) if b.type == "fallback"), default=-1)
        blocks = []
        for i, b in enumerate(content):
            t = b.type
            if i < last_fallback and t != "text":
                # A declined partial before the final fallback marker: only its text is
                # continuation context; its thinking/tool_use blocks must not be replayed
                # (or executed).
                continue
            if t == "text":
                blocks.append(TextBlock(b.text))
            elif t == "thinking":
                blocks.append(ThinkingBlock(b.thinking or "", self.name, native=b.to_dict()))
            elif t == "redacted_thinking":
                blocks.append(ThinkingBlock("", self.name, native=b.to_dict()))
            elif t == "tool_use":
                blocks.append(ToolCall(b.id, b.name, dict(b.input or {})))
            else:  # server tool results, fallback markers, compaction, ...
                blocks.append(OpaqueBlock(self.name, b.to_dict()))
        usage = _usage_from(final.usage)
        iterations = None
        raw_iters = getattr(final.usage, "iterations", None)
        if raw_iters:
            iterations = []
            for it in raw_iters:
                it_model = getattr(it, "model", None) or request_model
                u = _usage_from(it)
                u.server_tool_requests = {}
                iterations.append({"type": getattr(it, "type", "message"), "model": str(it_model),
                                   **u.as_dict(), "cost_usd": self.cost(str(it_model), u)})
        fallback_used = last_fallback >= 0 or any(it["type"] == "fallback_message" for it in iterations or [])
        return Message("assistant", blocks), usage, iterations, fallback_used

    @staticmethod
    def _stop(reason: str | None) -> StopReason:
        if not reason:
            return StopReason.END_TURN
        return _STOP_MAP.get(reason, StopReason.OTHER)

    def _response(self, final, settings: ModelSettings) -> ModelResponse:
        message, usage, iterations, fallback_used = self._decode(final, settings.model)
        served = getattr(final, "model", None) or settings.model
        if iterations:
            searches = Usage(server_tool_requests=dict(usage.server_tool_requests))
            cost = sum(it["cost_usd"] for it in iterations) + self.cost(served, searches)
        else:
            cost = self.cost(served, usage)
        stop = self._stop(final.stop_reason)
        detail = None
        if stop is StopReason.REFUSAL and getattr(final, "stop_details", None):
            sd = final.stop_details
            detail = f"{getattr(sd, 'category', None)}: {getattr(sd, 'explanation', '')}"
        return ModelResponse(
            message=message, stop_reason=stop, usage=usage, model=served, cost_usd=cost, stop_detail=detail,
            request_id=getattr(final, "_request_id", None), served_model=served, fallback_used=fallback_used,
            iterations=iterations,
        )

    # ------------------------------------------------------------------ errors

    def _map_error(self, exc: BaseException) -> ProviderError | None:
        """Translate SDK/transport exceptions to the neutral error types."""
        a = self._anthropic
        if isinstance(exc, ProviderError):
            return None
        if isinstance(exc, a.APIStatusError):
            status = int(getattr(exc, "status_code", 0) or 0)
            etype = _error_type(exc)
            msg = _error_message(exc)
            mid_stream = status < 400  # an `event: error` after a 200 response started streaming
            if etype in _RETRYABLE_TYPES or status in (408, 409, 429) or status >= 500 or (
                    mid_stream and etype not in _FATAL_TYPES):
                sem = _RETRYABLE_TYPES.get(etype or "", status) if mid_stream else status
                where = " mid-stream" if mid_stream else ""
                return RetryableProviderError(
                    f"Anthropic API transient error{where} ({etype or status}): {msg}",
                    retry_after=_retry_after(getattr(exc, "response", None)), status=sem or None)
            if status == 413 or etype == "request_too_large" or _CONTEXT_RE.search(msg):
                return ContextOverflowError(f"Anthropic API: request does not fit the context window "
                                            f"({etype or status}): {msg}")
            return ProviderError(f"Anthropic API rejected the request ({etype or status}): {msg}")
        if isinstance(exc, a.APIConnectionError):  # includes APITimeoutError
            return RetryableProviderError(f"Anthropic API connection error: {type(exc).__name__}: {exc}")
        if self._transport_errors and isinstance(exc, self._transport_errors):
            return RetryableProviderError(f"connection dropped: {type(exc).__name__}: {exc}")
        if isinstance(exc, a.AnthropicError):
            return ProviderError(f"Anthropic SDK error: {type(exc).__name__}: {exc}")
        return None

    # ------------------------------------------------------------------ calls

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
        req = self._request(settings, system, messages, tools)
        betas = req.pop("betas", None)
        try:
            if betas:
                stream_cm = self.client.beta.messages.stream(betas=betas, **req)
            else:
                stream_cm = self.client.messages.stream(**req)
            async with stream_cm as stream:
                if on_text is not None or on_thinking is not None:
                    async for event in stream:
                        if event.type != "content_block_delta":
                            continue
                        delta = event.delta
                        if delta.type == "text_delta":
                            await _emit(on_text, delta.text)
                        elif delta.type == "thinking_delta":
                            await _emit(on_thinking, delta.thinking)
                final = await stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - mapped below; anything unknown re-raises unchanged
            mapped = self._map_error(exc)
            if mapped is None:
                raise
            raise mapped from exc
        return self._response(final, settings)

    def supports_web_search(self) -> bool:
        return True

    #: Bounded re-sends when the server-side search loop returns ``pause_turn``.
    WEB_SEARCH_MAX_CONTINUATIONS = 2

    async def web_search(self, query: str, *, max_results: int = 8, allowed_domains: list[str] | None = None,
                         blocked_domains: list[str] | None = None) -> dict[str, Any]:
        """Use Claude's server-side web search tool and return cited results.

        Cost covers tokens (including cache) plus $10 per 1,000 searches.
        """
        if allowed_domains and blocked_domains:
            raise ProviderError("web_search: pass allowed_domains or blocked_domains, not both")
        model = self.web_search_model
        dynamic = _family(model, ("claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
                                  "claude-sonnet-5", "claude-sonnet-4-6"))
        tool: dict[str, Any] = {"type": "web_search_20260209" if dynamic else "web_search_20250305",
                                "name": "web_search", "max_uses": 3}
        if allowed_domains:
            tool["allowed_domains"] = list(allowed_domains)
        if blocked_domains:
            tool["blocked_domains"] = list(blocked_domains)
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": (
                "Search the web for the query below. Reply with a concise factual summary "
                "of what the top sources say, citing each source's URL and date where given. "
                f"Return at most {max_results} sources.\n\nQuery: {query}"
            ),
        }]
        results, summary, errors = [], [], []
        cost = 0.0
        final = None
        assistant: list[Any] = []  # blocks of a paused turn, re-sent to resume it
        for _ in range(self.WEB_SEARCH_MAX_CONTINUATIONS + 1):
            try:
                final = await self.client.messages.create(
                    model=model, max_tokens=4000, tools=[tool], messages=messages,
                )
            except Exception as exc:  # noqa: BLE001
                mapped = self._map_error(exc)
                if mapped is None:
                    raise
                raise mapped from exc
            for b in final.content:
                if b.type == "web_search_tool_result":
                    if isinstance(b.content, list):
                        for r in b.content:
                            results.append({
                                "title": getattr(r, "title", ""),
                                "url": getattr(r, "url", ""),
                                "page_age": getattr(r, "page_age", None),
                            })
                    else:  # server-tool errors arrive as an object, not an exception
                        errors.append(str(getattr(b.content, "error_code", b.content)))
                elif b.type == "text":
                    summary.append(b.text)
            cost += self.cost(getattr(final, "model", None) or model, _usage_from(final.usage))
            stop = getattr(final, "stop_reason", None)
            if stop != "pause_turn":
                break
            # The server-side tool loop hit its iteration limit: re-send with the
            # partial assistant turn so the server resumes where it stopped.
            assistant.extend(_block_dict(b) for b in final.content)
            messages = [messages[0], {"role": "assistant", "content": list(assistant)}]
        stop = getattr(final, "stop_reason", None)
        if stop == "pause_turn":
            errors.append(f"incomplete: server tool loop still paused after "
                          f"{self.WEB_SEARCH_MAX_CONTINUATIONS} continuation(s)")
        elif stop == "max_tokens":
            errors.append("incomplete: output truncated at max_tokens")
        elif stop == "refusal":
            sd = getattr(final, "stop_details", None)
            detail = f" ({getattr(sd, 'category', None)}: {getattr(sd, 'explanation', '')})" if sd else ""
            errors.append(f"refusal{detail}")
        out: dict[str, Any] = {"results": results[:max_results], "summary": "".join(summary), "cost_usd": cost}
        if stop is not None and stop != "end_turn":
            out["stop_reason"] = stop
        if errors:
            out["errors"] = errors
        return out

    async def aclose(self) -> None:
        await self.client.close()


def _block_dict(b: Any) -> Any:
    """Serialise an SDK content block for re-sending (pause_turn continuation)."""
    for attr in ("to_dict", "model_dump"):
        fn = getattr(b, attr, None)
        if callable(fn):
            return fn()
    return dict(vars(b))


def _run_sync(coro):  # pragma: no cover - convenience for scripts
    return asyncio.run(coro)
