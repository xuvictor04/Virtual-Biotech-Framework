"""Model providers. Add a backend by subclassing LLMProvider and registering it here."""

from __future__ import annotations

from typing import Any, Callable

from .base import (
    LLMProvider,
    Message,
    ModelResponse,
    ModelSettings,
    OpaqueBlock,
    ProviderError,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

_FACTORIES: dict[str, Callable[..., LLMProvider]] = {}


def register_provider(name: str, factory: Callable[..., LLMProvider]) -> None:
    _FACTORIES[name] = factory


def create_provider(name: str, **options: Any) -> LLMProvider:
    if name not in _FACTORIES:
        raise ProviderError(f"unknown provider {name!r}; registered: {sorted(_FACTORIES)}")
    return _FACTORIES[name](**options)


def _anthropic(**options: Any) -> LLMProvider:
    from .anthropic_provider import AnthropicProvider
    return AnthropicProvider(**options)


def _mock(**options: Any) -> LLMProvider:
    from .mock import ScriptedProvider, reply
    return ScriptedProvider(options.get("script") or (lambda *a: reply("(mock provider) no script configured.")))


register_provider("anthropic", _anthropic)
register_provider("mock", _mock)

__all__ = [
    "LLMProvider", "Message", "ModelResponse", "ModelSettings", "OpaqueBlock", "ProviderError",
    "StopReason", "TextBlock", "ThinkingBlock", "ToolCall", "ToolResult", "ToolSpec", "Usage",
    "create_provider", "register_provider",
]
