"""Model providers. Add a backend by subclassing LLMProvider and registering it here."""

from __future__ import annotations

from typing import Any, Callable

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
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
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
from .retry import RetryPolicy, complete_with_retry

_FACTORIES: dict[str, Callable[..., LLMProvider]] = {}


def register_provider(name: str, factory: Callable[..., LLMProvider]) -> None:
    _FACTORIES[name] = factory


def create_provider(name: str, **options: Any) -> LLMProvider:
    if name not in _FACTORIES:
        raise ProviderError(f"unknown provider {name!r}; registered: {sorted(_FACTORIES)}")
    return _FACTORIES[name](**options)


def _anthropic(**options: Any) -> LLMProvider:
    from .anthropic_provider import AnthropicProvider  # lazy: the SDK is an optional extra
    return AnthropicProvider(**options)


def _mock(**options: Any) -> LLMProvider:
    from .mock import ScriptedProvider, reply
    kwargs = {k: options[k] for k in ("strict", "usage_fn", "context_window", "capabilities") if k in options}
    return ScriptedProvider(options.get("script") or (lambda *a: reply("(mock provider) no script configured.")),
                            **kwargs)


register_provider("anthropic", _anthropic)
register_provider("mock", _mock)

__all__ = [
    "ContextOverflowError", "DocumentPart", "ImagePart", "LLMProvider", "Message", "ModelResponse", "ModelSettings",
    "OpaqueBlock", "ProviderCapabilities", "ProviderError", "RetryPolicy", "RetryableProviderError", "StopReason",
    "SystemSegment", "TextBlock", "ThinkingBlock", "ToolCall", "ToolResult", "ToolSpec", "Usage",
    "complete_with_retry", "content_text", "create_provider", "message_from_dict", "message_to_dict",
    "messages_from_dicts", "messages_to_dicts", "register_provider", "system_text", "unanswered_tool_calls",
    "validate_tool_pairing",
]
