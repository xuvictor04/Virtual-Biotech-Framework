"""Claude adapter (Anthropic Messages API) for the provider interface.

The paper ran on the Claude Agent SDK; this harness owns its own agent loop so
it stays provider-agnostic, and only this file knows about the Anthropic API.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from .base import (
    LLMProvider,
    Message,
    ModelResponse,
    ModelSettings,
    OpaqueBlock,
    ProviderError,
    StopReason,
    TextBlock,
    TextCallback,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

log = logging.getLogger(__name__)

# USD per million tokens: (input, output). Cache reads bill at 0.1x input and
# 5-minute cache writes at 1.25x input.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Models that take adaptive thinking + the effort parameter.
_ADAPTIVE = ("claude-fable-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
             "claude-opus-4-6", "claude-sonnet-5", "claude-sonnet-4-6")
# Models where thinking cannot be disabled (omit the parameter to get adaptive).
_THINKING_ALWAYS_ON = ("claude-fable-5", "claude-opus-5-5")
# Models that accept the server-side refusal fallback ("default" routing).
_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5")
# Models on which sampling parameters (temperature, ...) are rejected.
_NO_SAMPLING = ("claude-fable-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5")


def _family(model: str, prefixes: tuple[str, ...]) -> bool:
    """True if ``model`` is one of ``prefixes`` or a dated/suffixed variant of one."""
    return any(model.startswith(p) for p in prefixes)


def _exact(model: str, names: tuple[str, ...]) -> bool:
    return model in names


def price_for(model: str) -> tuple[float, float]:
    for key in sorted(PRICES, key=len, reverse=True):
        if model.startswith(key):
            return PRICES[key]
    return PRICES["claude-opus-5"]


def cost_usd(model: str, usage: Usage) -> float:
    pin, pout = price_for(model)
    return (
        usage.input_tokens * pin
        + usage.cache_read_tokens * pin * 0.1
        + usage.cache_write_tokens * pin * 1.25
        + usage.output_tokens * pout
    ) / 1e6


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
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise ProviderError("pip install 'vbt-harness[anthropic]' to use the Claude provider") from exc
        self._anthropic = anthropic
        kwargs: dict[str, Any] = {"max_retries": max_retries}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.AsyncAnthropic(**kwargs)
        self.refusal_fallback = refusal_fallback
        self.web_search_model = web_search_model
        self.prompt_caching = prompt_caching

    # ------------------------------------------------------------------ encode

    def _encode_block(self, block) -> dict[str, Any] | None:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text} if block.text else None
        if isinstance(block, ThinkingBlock):
            # Only our own thinking blocks, unchanged; others are dropped.
            return block.native if block.provider == self.name and block.native else None
        if isinstance(block, ToolCall):
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        if isinstance(block, ToolResult):
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_call_id,
                "content": block.content or "(empty result)",
                "is_error": block.is_error,
            }
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

    def _request(self, settings: ModelSettings, system: str, messages, tools) -> dict[str, Any]:
        model = settings.model
        req: dict[str, Any] = {
            "model": model,
            "max_tokens": settings.max_tokens,
            "system": system,
            "messages": self._encode_messages(messages),
        }
        if tools:
            req["tools"] = [
                {"name": t.name, "description": t.description[:10000], "input_schema": t.input_schema}
                for t in tools
            ]
        if self.prompt_caching:
            req["cache_control"] = {"type": "ephemeral"}

        adaptive = _family(model, _ADAPTIVE) or _family(model, _THINKING_ALWAYS_ON)
        if adaptive:
            # With thinking off we simply omit the parameter: newer models then
            # run their default, and lower effort is the recommended cost lever.
            if settings.thinking or _family(model, _THINKING_ALWAYS_ON):
                req["thinking"] = {"type": "adaptive", "display": "summarized"}
            if settings.effort:
                req["output_config"] = {"effort": settings.effort}
        elif settings.thinking:
            budget = int(settings.extra.get("thinking_budget", 8000))
            budget = max(1024, min(budget, settings.max_tokens - 1024))
            req["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if settings.temperature is not None and not _family(model, _NO_SAMPLING) and "thinking" not in req:
            req["temperature"] = settings.temperature
        return req

    # ------------------------------------------------------------------ decode

    def _decode(self, final) -> tuple[Message, Usage]:
        blocks = []
        for b in final.content:
            t = b.type
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
        u = final.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return Message("assistant", blocks), usage

    @staticmethod
    def _stop(reason: str | None) -> StopReason:
        try:
            return StopReason(reason or "end_turn")
        except ValueError:
            return StopReason.END_TURN if reason == "stop_sequence" else StopReason.OTHER

    # ------------------------------------------------------------------ calls

    async def complete(
        self,
        *,
        settings: ModelSettings,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback | None = None,
    ) -> ModelResponse:
        req = self._request(settings, system, messages, tools)
        use_fallback = self.refusal_fallback and _exact(settings.model, _FALLBACK_MODELS)
        a = self._anthropic
        try:
            if use_fallback:
                stream_cm = self.client.beta.messages.stream(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **req
                )
            else:
                stream_cm = self.client.messages.stream(**req)
            async with stream_cm as stream:
                if on_text is not None:
                    async for text in stream.text_stream:
                        r = on_text(text)
                        if inspect.isawaitable(r):
                            await r
                final = await stream.get_final_message()
        except (a.BadRequestError, a.AuthenticationError, a.PermissionDeniedError, a.NotFoundError) as exc:
            raise ProviderError(f"Anthropic API rejected the request: {exc}") from exc

        message, usage = self._decode(final)
        stop = self._stop(final.stop_reason)
        detail = None
        if stop is StopReason.REFUSAL and getattr(final, "stop_details", None):
            sd = final.stop_details
            detail = f"{getattr(sd, 'category', None)}: {getattr(sd, 'explanation', '')}"
        return ModelResponse(
            message=message,
            stop_reason=stop,
            usage=usage,
            model=final.model,
            cost_usd=cost_usd(final.model or settings.model, usage),
            stop_detail=detail,
            request_id=getattr(final, "_request_id", None),
        )

    def supports_web_search(self) -> bool:
        return True

    async def web_search(self, query: str, *, max_results: int = 8) -> dict[str, Any]:
        """Use Claude's server-side web search tool and return cited results."""
        model = self.web_search_model
        tool_type = "web_search_20260209" if _family(model, _ADAPTIVE) else "web_search_20250305"
        final = await self.client.messages.create(
            model=model,
            max_tokens=4000,
            tools=[{"type": tool_type, "name": "web_search", "max_uses": 3}],
            messages=[{
                "role": "user",
                "content": (
                    "Search the web for the query below. Reply with a concise factual summary "
                    "of what the top sources say, citing each source's URL and date where given. "
                    f"Return at most {max_results} sources.\n\nQuery: {query}"
                ),
            }],
        )
        results, summary = [], []
        for b in final.content:
            if b.type == "web_search_tool_result" and isinstance(b.content, list):
                for r in b.content[:max_results]:
                    results.append({
                        "title": getattr(r, "title", ""),
                        "url": getattr(r, "url", ""),
                        "page_age": getattr(r, "page_age", None),
                    })
            elif b.type == "text":
                summary.append(b.text)
        usage = Usage(final.usage.input_tokens or 0, final.usage.output_tokens or 0)
        return {"results": results, "summary": "".join(summary), "cost_usd": cost_usd(model, usage)}

    async def aclose(self) -> None:
        await self.client.close()


def _run_sync(coro):  # pragma: no cover - convenience for scripts
    return asyncio.run(coro)
