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
* Tool results may carry content parts (text, images, PDF documents). Providers
  without vision render them as text via :func:`content_text`.
* Transient failures (overload, rate limit, dropped connection) raise
  :class:`RetryableProviderError`; a request that no longer fits the model's
  context window raises :class:`ContextOverflowError`. Everything else that the
  provider rejects is a plain :class:`ProviderError`.

This module must stay free of vendor SDK imports.
"""

from __future__ import annotations

import abc
import copy
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
class ImagePart:
    """An image inside a tool result (base64, no data-URI prefix).

    ``source`` names where it came from (a path or URL) for logs and for the
    text fallback of providers without vision.
    """

    media_type: str
    data_b64: str
    source: str = ""
    type: Literal["image"] = "image"


@dataclass(kw_only=True)
class DocumentPart:
    """A document (PDF) inside a tool result. Keyword-only to avoid swapping
    ``media_type`` and ``data_b64`` by position."""

    data_b64: str
    media_type: str = "application/pdf"
    title: str = ""
    type: Literal["document"] = "document"


ContentPart = Union[TextBlock, ImagePart, DocumentPart]
ToolResultContent = Union[str, list[ContentPart]]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str | list[ContentPart]
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


def content_text(content: str | list[Any] | None) -> str:
    """Flatten tool-result content to text.

    Text parts are joined with newlines; images render as ``[image: <source>]``
    and documents as ``[document: <title>]`` so that logs, error summaries and
    providers without vision still get a readable stand-in.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out: list[str] = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, TextBlock):
            out.append(part.text)
        elif isinstance(part, ImagePart):
            out.append(f"[image: {part.source or part.media_type}]")
        elif isinstance(part, DocumentPart):
            out.append(f"[document: {part.title or part.media_type}]")
        else:  # unknown part type: keep something readable
            out.append(str(getattr(part, "text", "") or f"[{getattr(part, 'type', 'part')}]"))
    return "\n".join(s for s in out if s)


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

    @property
    def tool_results(self) -> list[ToolResult]:
        return [b for b in self.content if isinstance(b, ToolResult)]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


@dataclass
class SystemSegment:
    """One piece of a system prompt.

    ``cache=True`` marks the end of a stable prefix that providers with prompt
    caching may cache (e.g. the static upstream agent prompt); segments after
    the last cached one are treated as volatile (date, run paths, ...).
    """

    text: str
    cache: bool = False


SystemPrompt = Union[str, list[SystemSegment]]


def system_text(system: str | list[Any] | None) -> str:
    """The system prompt as one string (segments joined by blank lines)."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = [s.text if isinstance(s, SystemSegment) else str(s) for s in system]
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Requests / responses
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """A tool as the model sees it: name, description, JSON schema.

    ``strict=True`` asks providers that support it to constrain the call's
    arguments to ``input_schema`` (OpenAI-compatible servers: ``"strict": true``
    on the function, grammar-enforced by vLLM >= 0.30). Providers without
    strict tool calling ignore it.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = False


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    PAUSE = "pause_turn"  # provider-side tool loop paused; resend to continue
    CONTEXT_EXCEEDED = "context_exceeded"  # the model ran out of context window mid-generation
    OTHER = "other"


@dataclass
class Usage:
    """Token usage of one model call.

    ``cache_write_tokens`` counts every token written to the prompt cache (all
    TTLs); ``cache_write_1h_tokens`` is the subset written with the 1-hour TTL
    (priced higher by some providers). ``server_tool_requests`` counts
    provider-side tool invocations, e.g. ``{"web_search": 2}``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    server_tool_requests: dict[str, int] = field(default_factory=dict)

    @property
    def context_tokens(self) -> int:
        """Prompt size of the call: uncached input plus cache reads and writes."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def __add__(self, other: "Usage") -> "Usage":
        servers = dict(self.server_tool_requests)
        for k, v in (other.server_tool_requests or {}).items():
            servers[k] = servers.get(k, 0) + v
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cache_write_1h_tokens + other.cache_write_1h_tokens,
            servers,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_write_1h_tokens": self.cache_write_1h_tokens,
            "server_tool_requests": dict(self.server_tool_requests),
        }


@dataclass
class ModelSettings:
    """Per-agent model settings, resolved from config tiers.

    ``extra`` holds harness-side hints. Providers consume the keys they know
    (e.g. ``thinking_budget``, ``context_window_tokens``, ``agent_name``,
    ``context_management``) and never forward unknown keys to a vendor API.
    """

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
    served_model: str | None = None      # model that produced the message (differs after a fallback)
    fallback_used: bool = False          # a provider-side fallback model served (part of) the turn
    retries: int = 0                     # transient failures retried before this response (retry.py)
    iterations: list[dict[str, Any]] | None = None  # per-attempt usage/cost when the provider reports it


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider (and, where it matters, a given model) supports.

    ``history_bound_thinking``: thinking blocks are bound to the exact
    conversation prefix that produced them, so any client-side edit of earlier
    turns invalidates every later thinking block (the context manager then
    strips them instead of replaying them).

    ``replays_reasoning``: the provider re-sends the text of earlier
    reasoning (``ThinkingBlock``) on every request, so reasoning counts
    against the context window; the context manager should clear old
    reasoning together with old tool results when it compacts.

    ``tool_choice``: the provider honours ``ModelSettings.extra['tool_choice']``
    (``'auto'``, ``'required'``, ``'none'`` or ``{'name': <tool>}`` to force
    one tool).
    """

    images: bool = False
    documents: bool = False
    web_search: bool = False
    server_context_management: bool = False
    history_bound_thinking: bool = False
    replays_reasoning: bool = False
    tool_choice: bool = False


TextCallback = Callable[[str], Union[None, Awaitable[None]]]
ThinkingCallback = TextCallback


class ProviderError(RuntimeError):
    """Raised for non-retryable provider failures (bad request, auth, ...)."""


class RetryableProviderError(ProviderError):
    """A transient failure (overload, rate limit, 5xx, dropped connection).

    Safe to retry by re-sending the same request: a failed response is never
    appended to the history. ``retry_after`` is the server's requested delay in
    seconds when it sent one; ``status`` the (semantic) HTTP status if known.
    """

    def __init__(self, message: str = "transient provider error", *, retry_after: float | None = None,
                 status: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status = status


class ContextOverflowError(ProviderError):
    """The request does not fit the model's context window (or request size limit)."""


class LLMProvider(abc.ABC):
    """Interface every model backend implements."""

    name: str = "base"

    @abc.abstractmethod
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
        """Run one model turn and return the assistant message.

        ``on_text`` / ``on_thinking`` receive streamed text and reasoning deltas
        when the provider streams; providers without reasoning never call
        ``on_thinking``.
        """

    def capabilities(self, model: str | None = None) -> ProviderCapabilities:
        """Feature flags; ``model`` narrows model-dependent flags when given."""
        return ProviderCapabilities(web_search=self.supports_web_search())

    def context_window(self, model: str) -> int | None:
        """Context window of ``model`` in tokens, or None when unknown."""
        return None

    def check_credentials(self) -> str | None:
        """Return a human-readable problem if credentials are missing, else None.

        Must not make network calls; used before any billable work starts.
        """
        return None

    async def prepare(self) -> None:
        """Optional readiness check before the first call (e.g. a local
        server's health and model discovery). May raise ``ProviderError`` with
        a fix hint. Default: nothing to do."""
        return None

    async def server_info(self) -> dict[str, Any]:
        """Optional facts about the serving backend (engine version, served
        models, context length) for run records. Default: ``{}``."""
        return {}

    async def web_search(self, query: str, *, max_results: int = 8, allowed_domains: list[str] | None = None,
                         blocked_domains: list[str] | None = None) -> dict[str, Any]:
        """Optional provider-native web search.

        Return ``{"results": [{"title", "url", ...}], "summary": str, "cost_usd": float}``.
        """
        raise NotImplementedError(f"provider {self.name!r} has no native web search")

    def supports_web_search(self) -> bool:
        return False

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


# ---------------------------------------------------------------------------
# History invariants
# ---------------------------------------------------------------------------


def validate_tool_pairing(messages: list[Message], *, allow_pending: bool = False) -> list[str]:
    """Check the tool_use/tool_result pairing rules vendor APIs enforce.

    * every assistant ``ToolCall`` is answered by a ``ToolResult`` in the very
      next message, which must be a user message;
    * the results come first in that user message (before any text);
    * no orphan results (a result whose call is not in the preceding assistant
      message) and no duplicate results;
    * tool results never appear in assistant messages.

    ``allow_pending=True`` tolerates a trailing assistant message whose calls
    have not been executed yet (mid tool round). Returns a list of problems
    (empty when the history is valid).
    """
    problems: list[str] = []
    n = len(messages)
    for i, m in enumerate(messages):
        if m.role == "assistant":
            stray = [b.tool_call_id for b in m.content if isinstance(b, ToolResult)]
            if stray:
                problems.append(f"messages[{i}]: assistant message contains tool_result(s) {stray}")
            calls = m.tool_calls
            if not calls:
                continue
            ids = [c.id for c in calls]
            if len(set(ids)) != len(ids):
                problems.append(f"messages[{i}]: duplicate tool_use ids {ids}")
            if i == n - 1:
                if not allow_pending:
                    problems.append(f"messages[{i}]: unpaired tool_use {ids} at the end of the history "
                                    "(no tool_result message follows)")
                continue
            nxt = messages[i + 1]
            if nxt.role != "user":
                problems.append(f"messages[{i}]: unpaired tool_use {ids}: the next message is not a user "
                                "message with tool_results")
                continue
            leading: set[str] = set()
            for b in nxt.content:
                if not isinstance(b, ToolResult):
                    break
                leading.add(b.tool_call_id)
            missing = [f"{c.id} ({c.name})" for c in calls if c.id not in leading]
            if missing:
                problems.append(f"messages[{i}]: unpaired tool_use {missing}: no tool_result at the start of "
                                f"messages[{i + 1}]")
        else:
            results = [b for b in m.content if isinstance(b, ToolResult)]
            if not results:
                continue
            prev = messages[i - 1] if i > 0 else None
            expected = {c.id for c in prev.tool_calls} if prev is not None and prev.role == "assistant" else set()
            seen: set[str] = set()
            for r in results:
                if r.tool_call_id not in expected:
                    problems.append(f"messages[{i}]: orphan tool_result {r.tool_call_id} (no matching tool_use in "
                                    "the preceding assistant message)")
                if r.tool_call_id in seen:
                    problems.append(f"messages[{i}]: duplicate tool_result {r.tool_call_id}")
                seen.add(r.tool_call_id)
            other_seen = False
            for b in m.content:
                if not isinstance(b, ToolResult):
                    other_seen = True
                elif other_seen:
                    problems.append(f"messages[{i}]: tool_result {b.tool_call_id} follows other content; "
                                    "tool_results must come first")
                    break
    return problems


def unanswered_tool_calls(messages: list[Message]) -> list[tuple[int, ToolCall]]:
    """``(assistant message index, call)`` for every call without a result in
    the following user message (anywhere in it). Used to repair histories."""
    out: list[tuple[int, ToolCall]] = []
    for i, m in enumerate(messages):
        if m.role != "assistant" or not m.tool_calls:
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        answered = {r.tool_call_id for r in nxt.tool_results} if nxt is not None and nxt.role == "user" else set()
        out.extend((i, c) for c in m.tool_calls if c.id not in answered)
    return out


# ---------------------------------------------------------------------------
# Lossless (de)serialisation, for resuming sessions
# ---------------------------------------------------------------------------


def _part_to_dict(p: Any) -> dict[str, Any]:
    if isinstance(p, TextBlock):
        d: dict[str, Any] = {"type": "text", "text": p.text}
        if p.native is not None:
            d["native"] = copy.deepcopy(p.native)
        return d
    if isinstance(p, ImagePart):
        return {"type": "image", "media_type": p.media_type, "data_b64": p.data_b64, "source": p.source}
    if isinstance(p, DocumentPart):
        return {"type": "document", "media_type": p.media_type, "data_b64": p.data_b64, "title": p.title}
    raise TypeError(f"unknown content part {p!r}")


def _part_from_dict(d: dict[str, Any]) -> ContentPart:
    t = d.get("type")
    if t == "text":
        return TextBlock(d.get("text", ""), native=copy.deepcopy(d.get("native")))
    if t == "image":
        return ImagePart(d["media_type"], d["data_b64"], d.get("source", ""))
    if t == "document":
        return DocumentPart(data_b64=d["data_b64"], media_type=d.get("media_type", "application/pdf"),
                            title=d.get("title", ""))
    raise ValueError(f"unknown content part type {t!r}")


def block_to_dict(b: Block) -> dict[str, Any]:
    """JSON-serialisable form of a block, including native payloads."""
    if isinstance(b, TextBlock):
        return _part_to_dict(b)
    if isinstance(b, ThinkingBlock):
        return {"type": "thinking", "text": b.text, "provider": b.provider, "native": copy.deepcopy(b.native)}
    if isinstance(b, ToolCall):
        return {"type": "tool_call", "id": b.id, "name": b.name, "input": copy.deepcopy(b.input),
                "native": copy.deepcopy(b.native)}
    if isinstance(b, ToolResult):
        content: Any = b.content if isinstance(b.content, str) else [_part_to_dict(p) for p in b.content]
        return {"type": "tool_result", "tool_call_id": b.tool_call_id, "content": content, "is_error": b.is_error}
    if isinstance(b, OpaqueBlock):
        return {"type": "opaque", "provider": b.provider, "native": copy.deepcopy(b.native)}
    raise TypeError(f"unknown block {b!r}")


def block_from_dict(d: dict[str, Any]) -> Block:
    t = d.get("type")
    if t == "text":
        return TextBlock(d.get("text", ""), native=copy.deepcopy(d.get("native")))
    if t == "thinking":
        return ThinkingBlock(d.get("text", ""), d.get("provider", ""), native=copy.deepcopy(d.get("native")))
    if t == "tool_call":
        return ToolCall(d["id"], d["name"], copy.deepcopy(d.get("input") or {}), native=copy.deepcopy(d.get("native")))
    if t == "tool_result":
        raw = d.get("content", "")
        content = raw if isinstance(raw, str) else [_part_from_dict(p) for p in raw]
        return ToolResult(d["tool_call_id"], content, bool(d.get("is_error", False)))
    if t == "opaque":
        return OpaqueBlock(d.get("provider", ""), copy.deepcopy(d.get("native") or {}))
    raise ValueError(f"unknown block type {t!r}")


def message_to_dict(m: Message) -> dict[str, Any]:
    return {"role": m.role, "content": [block_to_dict(b) for b in m.content]}


def message_from_dict(d: dict[str, Any]) -> Message:
    role = d.get("role")
    if role not in ("user", "assistant"):
        raise ValueError(f"invalid message role {role!r}")
    return Message(role, [block_from_dict(b) for b in d.get("content") or []])


def messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    return [message_to_dict(m) for m in messages]


def messages_from_dicts(items: list[dict[str, Any]]) -> list[Message]:
    return [message_from_dict(d) for d in items]
