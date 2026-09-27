"""Provider-neutral message types and the LLMProvider interface.

Everything above this module (agents, tools, orchestration) speaks only these
types. A provider adapter translates them to and from one vendor's API, so
swapping Claude for another model family means writing one adapter file and
changing ``provider:`` in the config.

Design notes
------------
* Conversations are append-only. Adapters may attach vendor-native payloads to
  blocks (``native``) so that opaque items such as thinking signatures are
  replayed byte-for-byte to the same provider; other providers ignore them.
* Stop reasons are normalised to a small vocabulary (``StopReason``).
* Usage is normalised to input/output/cache token counts; cost is computed by
  the provider because only it knows its price table.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Union


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"
    native: dict[str, Any] | None = None


@dataclass
class ThinkingBlock:
    """Model reasoning. ``text`` may be empty when the provider hides it.

    ``native`` holds the provider's exact block (e.g. with its signature) so it
    can be echoed back unchanged on the next request to the same provider.
    """

    text: str = ""
    provider: str = ""
    type: Literal["thinking"] = "thinking"
    native: dict[str, Any] | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"
    native: dict[str, Any] | None = None


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass
class OpaqueBlock:
    """A provider-native block the harness does not interpret (server-tool
    results, fallback markers, ...). Replayed to the same provider only."""

    provider: str
    native: dict[str, Any]
    type: Literal["opaque"] = "opaque"


Block = Union[TextBlock, ThinkingBlock, ToolCall, ToolResult, OpaqueBlock]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: list[Block]

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls("user", [TextBlock(text)])

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.content if isinstance(b, TextBlock) and b.text)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]


# ---------------------------------------------------------------------------
# Requests / responses
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """A tool as the model sees it: name, description, JSON schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    PAUSE = "pause_turn"  # provider-side tool loop paused; resend to continue
    OTHER = "other"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass
class ModelSettings:
    """Per-agent model settings, resolved from config tiers."""

    provider: str
    model: str
    max_tokens: int = 32000
    effort: str | None = "high"  # low | medium | high | xhigh | max (if supported)
    thinking: bool = True
    temperature: float | None = None  # only honoured where the model allows it
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    message: Message
    stop_reason: StopReason
    usage: Usage
    model: str
    cost_usd: float = 0.0
    stop_detail: str | None = None
    request_id: str | None = None


TextCallback = Callable[[str], Union[None, Awaitable[None]]]


class ProviderError(RuntimeError):
    """Raised for non-retryable provider failures (bad request, auth, ...)."""


class LLMProvider(abc.ABC):
    """Interface every model backend implements."""

    name: str = "base"

    @abc.abstractmethod
    async def complete(
        self,
        *,
        settings: ModelSettings,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback | None = None,
    ) -> ModelResponse:
        """Run one model turn and return the assistant message."""

    async def web_search(self, query: str, *, max_results: int = 8) -> dict[str, Any]:
        """Optional provider-native web search. Return {"results": [...], "summary": str}."""
        raise NotImplementedError(f"provider {self.name!r} has no native web search")

    def supports_web_search(self) -> bool:
        return False

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
