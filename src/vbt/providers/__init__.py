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


#: Options of the OpenAI-compatible adapter. ``provider.options`` is one namespace that profiles
#: deep-merge onto ``configs/default.yaml`` (which configures the local server), so a profile that
#: switches to ``anthropic`` without nulling them inherits these; the Anthropic factory drops them.
LOCAL_SERVER_OPTIONS = frozenset({
    "base_url", "base_urls", "served_model_name", "family", "model", "api_key_env", "timeout_s", "read_timeout_s",
    "max_concurrency", "data_parallel_size", "routing", "pricing", "extra_body", "context_window", "vision",
    "reasoning_effort_supported", "replay_reasoning_field", "parallel_tool_calls", "auto_discover", "wait_ready_s",
    "trust_env", "headers",
})


def anthropic_options(options: dict[str, Any]) -> dict[str, Any]:
    """The options the Anthropic adapter receives from ``provider.options``.

    The generic ``base_url`` belongs to the local server and is never used for
    Claude: inheriting it would send every request, with the ``x-api-key``
    header carrying ``ANTHROPIC_API_KEY``, to whatever listens there (in
    cleartext for ``http://``). A different Anthropic endpoint (a gateway or
    proxy) is set with ``anthropic_base_url`` (or the SDK's own
    ``ANTHROPIC_BASE_URL`` environment variable)."""
    from ..envpolicy import redact_url, register_secret

    opts = dict(options)
    endpoint = opts.pop("anthropic_base_url", None)
    inherited = opts.get("base_url")
    for key in LOCAL_SERVER_OPTIONS:
        opts.pop(key, None)
    if isinstance(opts.get("api_key"), str):
        register_secret(opts["api_key"], "provider.options.api_key")  # masked in tool output like an env key
    if inherited and not endpoint:
        import logging

        logging.getLogger(__name__).warning(
            "provider anthropic: ignoring provider.options.base_url %s (the local inference server's option, "
            "e.g. inherited from configs/default.yaml); Claude requests go to the Anthropic API. Set "
            "provider.options.anthropic_base_url to use another Anthropic endpoint.", redact_url(str(inherited)))
    if endpoint:
        opts["base_url"] = endpoint
    return opts


def _anthropic(**options: Any) -> LLMProvider:
    from .anthropic_provider import AnthropicProvider  # lazy: the SDK is an optional extra
    return AnthropicProvider(**anthropic_options(options))


def _mock(**options: Any) -> LLMProvider:
    from .mock import ScriptedProvider, reply
    kwargs = {k: options[k] for k in ("strict", "usage_fn", "context_window", "capabilities") if k in options}
    return ScriptedProvider(options.get("script") or (lambda *a: reply("(mock provider) no script configured.")),
                            **kwargs)


def _openai_compat_factory(registered_name: str) -> Callable[..., LLMProvider]:
    def factory(**options: Any) -> LLMProvider:
        from .openai_compat import create  # lazy: keeps `import vbt.providers` light
        return create(registered_name, **options)
    return factory


register_provider("anthropic", _anthropic)
register_provider("mock", _mock)
# Local / self-hosted OpenAI-compatible servers (default: vLLM serving Qwen3.8).
for _name in ("vllm", "sglang", "openai_compat", "llamacpp"):
    register_provider(_name, _openai_compat_factory(_name))
del _name

__all__ = [
    "ContextOverflowError", "DocumentPart", "ImagePart", "LLMProvider", "Message", "ModelResponse", "ModelSettings",
    "OpaqueBlock", "ProviderCapabilities", "ProviderError", "RetryPolicy", "RetryableProviderError", "StopReason",
    "SystemSegment", "TextBlock", "ThinkingBlock", "ToolCall", "ToolResult", "ToolSpec", "Usage",
    "LOCAL_SERVER_OPTIONS", "anthropic_options", "complete_with_retry", "content_text", "create_provider",
    "message_from_dict", "message_to_dict", "messages_from_dicts", "messages_to_dicts", "register_provider",
    "system_text", "unanswered_tool_calls", "validate_tool_pairing",
]
