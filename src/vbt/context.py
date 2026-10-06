"""Provider-neutral context-window management for long agent loops.

The CSO keeps one conversation across a whole session and specialists can make
hundreds of model calls with large tool outputs, so histories outgrow the
model's context window. :class:`ContextManager` keeps them inside it:

1. **Tool-result clearing** (above ``soft_ratio`` of the window): the content of
   tool results older than the last ``keep_recent_calls`` model calls is
   persisted to ``spill_dir`` (unless the runtime's truncation note already names
   its spill file under ``logs/tool_outputs``) and
   replaced by a stub naming the file, so the agent can re-read it on demand
   with QueryToolOutput or Read.
2. **Summarisation** (above ``hard_ratio``, or still above ``soft_ratio`` after
   clearing): one provider call (same model, thinking off, no tools) condenses
   the oldest span of the conversation. The head user message (the task) is kept
   verbatim and the summary is appended to it as a
   ``[Conversation summary — harness compaction]`` block; the kept tail starts
   with an assistant message, so tool_use/tool_result pairs are never split.
   User messages from the span are carried verbatim inside the summary block.
3. **Overflow recovery**: after a provider reports a context overflow,
   :meth:`ContextManager.recover_overflow` compacts aggressively.

The history list is mutated in place; on any failure it is restored. After a
compaction ``validate_tool_pairing(messages, allow_pending=True)`` is ``[]``.

Thinking blocks: blocks inside a summarised span are dropped. For providers
whose thinking blocks are bound to the exact conversation prefix
(``ProviderCapabilities.history_bound_thinking``), every thinking block after the
first edited message is stripped as well, since replaying it would be rejected.
For providers that re-send earlier reasoning as text on every request
(``ProviderCapabilities.replays_reasoning``: local OpenAI-compatible servers,
whose chat template keeps all past reasoning), the clearing pass also empties
the reasoning of assistant turns older than ``keep_recent_calls`` (text ``""``,
``native['cleared_by_harness']``), in the same pass as the tool results, so the
prompt cache misses once per compaction rather than on every turn.

No vendor SDK imports here.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .providers.base import (
    ContextOverflowError,
    DocumentPart,
    ImagePart,
    LLMProvider,
    Message,
    ModelSettings,
    OpaqueBlock,
    ProviderCapabilities,
    ProviderError,
    SystemSegment,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    Usage,
    content_text,
    system_text,
    validate_tool_pairing,
)
from .providers.retry import RetryPolicy, complete_with_retry

log = logging.getLogger(__name__)

SUMMARY_HEADER = "[Conversation summary — harness compaction]"
VERBATIM_MARKER = "### User messages from the summarized conversation (verbatim, oldest first)"
CLEARED_PREFIX = "[cleared by harness:"
SUMMARIZER_AGENT = "context-compactor"

CHARS_PER_TOKEN = 4          # rough estimate used only when the provider reported no usage
IMAGE_TOKENS = 1600          # a ~1568 px image
# The runtime's own truncation note (vbt.tools.base.truncation_note), appended as
# the last thing in a truncated tool output. Only this exact phrase, naming a file
# under ``logs/tool_outputs``, counts as "already spilled": any other "saved to
# <file>" in a tool's output (a figure, a CSV) is the tool talking, not the harness.
_SPILLED_RE = re.compile(r"\[Output truncated: [\d,]+ chars(?: \([^()\n]*\))?\. Full output saved to (\S+?); "
                         r"[^\n]*\]\s*\Z")
_SPILL_DIR_PARTS = ("logs", "tool_outputs")


def _harness_spill_path(text: str) -> Path | None:
    """The runtime's spill file named by ``text``'s truncation note, if any."""
    m = _SPILLED_RE.search(text)
    if not m:
        return None
    cand = Path(m.group(1))
    parts = cand.parts
    under = any(parts[i:i + 2] == _SPILL_DIR_PARTS for i in range(len(parts) - 2))
    return cand if under and cand.is_file() else None

SUMMARIZER_SYSTEM = """\
You are the context-compaction step of a multi-agent drug-discovery research harness. \
An agent's conversation has grown too long for its context window. You receive the \
oldest part of that conversation as a transcript (and possibly an earlier summary). \
Write a faithful, dense summary that lets the agent continue its work without the \
original messages.

Preserve, quoting exactly (never paraphrase numbers, identifiers or paths):
- the research plan and the current status of each step, and what remains to do;
- every clarification answer and decision the user gave;
- each delegation: which specialist, the task, its key findings with numbers \
(effect sizes, p-values, scores, counts, units) and every artifact/file path it reported;
- reviewer verdicts and any revisions they requested;
- filed claim IDs, evidence references (PMIDs, NCT IDs, dataset IDs) and registered artifacts;
- tool failures, unavailable data sources and their consequences;
- open questions and caveats.

Do not invent facts. Omit pleasantries and reasoning that led nowhere. Cleared tool \
outputs appear as "[cleared by harness: ...]" stubs: keep the file paths they name so \
the agent can re-read them. The user's own messages are appended verbatim by the \
harness after your summary, so do not repeat them. Output only the summary."""


# ---------------------------------------------------------------------------
# Policy and result
# ---------------------------------------------------------------------------


@dataclass
class ContextPolicy:
    enabled: bool = True
    soft_ratio: float = 0.70           # start clearing old tool results
    hard_ratio: float = 0.85           # summarise regardless
    keep_recent_calls: int = 6         # tool results of the last N model calls are never cleared
    min_clear_chars: int = 2000        # smaller tool results are not worth clearing
    summary_max_tokens: int = 4000     # target length of a summary
    server_side: bool = False          # ask providers that support it to clear tool results server-side
    min_clear_fraction: float = 0.05   # a clearing pass must free this share of the window, else summarise
    default_window_tokens: int = 200_000  # used when neither settings nor provider know the window

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "ContextPolicy":
        """Build from the ``context:`` config section (or a full config holding one)."""
        cfg = dict(cfg or {})
        if isinstance(cfg.get("context"), dict):
            cfg = dict(cfg["context"])
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in cfg and cfg[f.name] is not None:
                v = cfg[f.name]
                if f.name in ("enabled", "server_side"):
                    kwargs[f.name] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                elif f.name in ("keep_recent_calls", "min_clear_chars", "summary_max_tokens", "default_window_tokens"):
                    kwargs[f.name] = int(v)
                else:
                    kwargs[f.name] = float(v)
        return cls(**kwargs)


@dataclass
class CompactionResult:
    """What a compaction did, for cost accounting and tracing."""

    strategy: str                       # e.g. "clear_tool_results", "summarize", "clear_tool_results+summarize"
    tokens_before: int
    tokens_after_estimate: int
    usage: Usage = field(default_factory=Usage)   # usage of the summariser call (if any)
    cost_usd: float = 0.0
    cleared_results: int = 0
    summarized_messages: int = 0
    thinking_dropped: int = 0
    spill_paths: list[str] = field(default_factory=list)
    model: str | None = None
    note: str = ""
    thinking_cleared: int = 0           # reasoning blocks emptied (replays_reasoning providers)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = self.usage.as_dict()
        return d


# ---------------------------------------------------------------------------
# Size estimates
# ---------------------------------------------------------------------------


def content_chars(content: Any) -> int:
    """Approximate size of tool-result content in characters (images and
    documents converted to a character equivalent)."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    n = 0
    for p in content:
        if isinstance(p, TextBlock):
            n += len(p.text)
        elif isinstance(p, ImagePart):
            n += IMAGE_TOKENS * CHARS_PER_TOKEN
        elif isinstance(p, DocumentPart):
            n += max(IMAGE_TOKENS * CHARS_PER_TOKEN, len(p.data_b64) // 2)
        else:
            n += len(str(p))
    return n


def _block_chars(b: Any) -> int:
    if isinstance(b, TextBlock):
        return len(b.text)
    if isinstance(b, ThinkingBlock):
        return len(b.text)
    if isinstance(b, ToolCall):
        return len(b.name) + len(json.dumps(b.input, default=str))
    if isinstance(b, ToolResult):
        return content_chars(b.content) + 20
    if isinstance(b, OpaqueBlock):
        return len(json.dumps(b.native, default=str))
    return len(str(b))


def message_chars(m: Message) -> int:
    return sum(_block_chars(b) for b in m.content) + 8


def estimate_tokens(obj: Any) -> int:
    """Rough token estimate (chars / 4) of a message, list of messages, system
    prompt or string."""
    if obj is None:
        return 0
    if isinstance(obj, str):
        return len(obj) // CHARS_PER_TOKEN
    if isinstance(obj, Message):
        return message_chars(obj) // CHARS_PER_TOKEN
    if isinstance(obj, list):
        if obj and all(isinstance(x, SystemSegment) for x in obj):
            return len(system_text(obj)) // CHARS_PER_TOKEN
        return sum(estimate_tokens(x) for x in obj)
    return len(str(obj)) // CHARS_PER_TOKEN


def _clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n] + f"\n[... {len(text) - n:,} more chars]"


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:120] or "result"


_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp",
        "application/pdf": ".pdf"}


@dataclass
class _Stats:
    removed_tokens: int = 0
    cleared: int = 0
    summarized: int = 0
    thinking_dropped: int = 0
    thinking_cleared: int = 0
    paths: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    model: str | None = None
    notes: list[str] = field(default_factory=list)


def _attach_spend(exc: BaseException, st: _Stats) -> None:
    """Record on ``exc`` what a failed compaction already spent (the summariser
    call), so the caller can charge it: see ``compaction_spend``."""
    if st.cost or st.usage.input_tokens or st.usage.output_tokens:
        try:
            exc.compaction_usage = st.usage  # type: ignore[attr-defined]
            exc.compaction_cost_usd = st.cost  # type: ignore[attr-defined]
        except (AttributeError, TypeError):  # exceptions with __slots__
            log.warning("compaction failed after spending $%.4f that cannot be attached to %s",
                        st.cost, type(exc).__name__)


def compaction_spend(exc: BaseException) -> tuple[Usage, float] | None:
    """(usage, cost) a failed compaction spent before raising ``exc``, if any."""
    usage = getattr(exc, "compaction_usage", None)
    if usage is None:
        return None
    return usage, float(getattr(exc, "compaction_cost_usd", 0.0) or 0.0)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ContextManager:
    """Keeps an agent's history inside its model's context window."""

    def __init__(self, provider: LLMProvider, policy: ContextPolicy | None = None,
                 spill_dir: str | Path | None = None, *, retry: RetryPolicy | None = None) -> None:
        self.provider = provider
        self.policy = policy or ContextPolicy()
        self.spill_dir = Path(spill_dir) if spill_dir else None
        self.retry = retry or RetryPolicy(attempts=3)

    # ------------------------------------------------------------------ sizing

    def window_for(self, settings: ModelSettings) -> int:
        """Context window for ``settings``: ``extra['context_window_tokens']``
        override, else the provider's table, else ``policy.default_window_tokens``."""
        override = (settings.extra or {}).get("context_window_tokens")
        if override:
            return int(override)
        try:
            w = self.provider.context_window(settings.model)
        except Exception:  # noqa: BLE001 - a broken table must not break the loop
            w = None
        return int(w) if w else int(self.policy.default_window_tokens)

    @staticmethod
    def used_tokens(usage: Usage | None) -> int:
        """Prompt size reported by a call: input + cache reads + cache writes."""
        if usage is None:
            return 0
        return int(usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens)

    def _capabilities(self, model: str | None) -> ProviderCapabilities:
        try:
            return self.provider.capabilities(model)
        except TypeError:  # providers written against capabilities() without a model argument
            return self.provider.capabilities()

    def settings_for(self, settings: ModelSettings) -> ModelSettings:
        """Settings with server-side context editing requested, when the policy
        enables it and the provider supports it; otherwise unchanged."""
        if not (self.policy.enabled and self.policy.server_side):
            return settings
        if "context_management" in (settings.extra or {}):
            return settings
        if not self._capabilities(settings.model).server_context_management:
            return settings
        return replace(settings, extra={**(settings.extra or {}), "context_management": True})

    def _projected_tokens(self, messages: list[Message], last_usage: Usage | None,
                          system: str | list[SystemSegment] | None) -> int:
        used = self.used_tokens(last_usage)
        if used > 0 and last_usage is not None:
            last_asst = max((i for i, m in enumerate(messages) if m.role == "assistant"), default=-1)
            since = estimate_tokens(messages[last_asst + 1:]) if last_asst >= 0 else 0
            return used + int(last_usage.output_tokens) + since
        return estimate_tokens(system_text(system)) + estimate_tokens(messages)

    # ------------------------------------------------------------------ public API

    async def maybe_compact(self, messages: list[Message], *, last_usage: Usage | None, settings: ModelSettings,
                            system: str | list[SystemSegment] | None, agent: str) -> CompactionResult | None:
        """Compact ``messages`` in place if the next request would exceed the soft
        threshold. Returns None when nothing was needed (or possible)."""
        if not self.policy.enabled or len(messages) < 2:
            return None
        window = self.window_for(settings)
        projected = self._projected_tokens(messages, last_usage, system)
        if projected < self.policy.soft_ratio * window:
            return None
        return await self._compact(messages, settings=settings, system=system, agent=agent, window=window,
                                   tokens_before=projected, aggressive=False)

    async def recover_overflow(self, messages: list[Message], *, settings: ModelSettings,
                               system: str | list[SystemSegment] | None, agent: str) -> CompactionResult:
        """Compact aggressively after the provider reported a context overflow.

        Raises ContextOverflowError when nothing can be compacted any further,
        or when context management is disabled (``context.enabled: false``).
        """
        window = self.window_for(settings)
        estimate = estimate_tokens(system_text(system)) + estimate_tokens(messages)
        if not self.policy.enabled:
            raise ContextOverflowError(f"context overflow for {agent} (~{estimate:,} tokens estimated, window "
                                       f"{window:,}) and context management is disabled")
        tokens_before = max(estimate, window)  # it did not fit
        res = await self._compact(messages, settings=settings, system=system, agent=agent, window=window,
                                  tokens_before=tokens_before, aggressive=True)
        if res is None:
            raise ContextOverflowError(f"context overflow for {agent}: the history cannot be compacted further "
                                       f"(~{estimate:,} tokens estimated, window {window:,})")
        return res

    # ------------------------------------------------------------------ core

    async def _compact(self, messages: list[Message], *, settings: ModelSettings,
                       system: str | list[SystemSegment] | None, agent: str, window: int, tokens_before: int,
                       aggressive: bool) -> CompactionResult | None:
        snapshot = list(messages)
        pre_problems = validate_tool_pairing(messages, allow_pending=True)
        pol = self.policy
        soft, hard = pol.soft_ratio * window, pol.hard_ratio * window
        st = _Stats()
        steps: list[str] = []
        first_edit: int | None = None
        keep = 1 if aggressive else max(0, pol.keep_recent_calls)
        min_chars = min(pol.min_clear_chars, 500) if aggressive else pol.min_clear_chars
        # providers that replay reasoning text: old reasoning is cleared with old tool results
        thinking = bool(self._capabilities(settings.model).replays_reasoning)

        def clear(keep_recent: int, label: str) -> None:
            nonlocal first_edit
            e = self._clear_tool_results(messages, keep_recent=keep_recent, min_chars=min_chars, stats=st,
                                         clear_thinking=thinking)
            if e is not None:
                steps.append(label)
                first_edit = e if first_edit is None else min(first_edit, e)

        try:
            # 1. clear old tool results - only when it frees a worthwhile share of the
            #    window (every edit restarts the prompt cache from the edited message)
            probe = _Stats()
            clearable = self._clear_tool_results(list(messages), keep_recent=keep, min_chars=min_chars,
                                                 stats=probe, dry_run=True, clear_thinking=thinking) is not None
            cleared_now = clearable and (aggressive or probe.removed_tokens >= pol.min_clear_fraction * window)
            if cleared_now:
                clear(keep, "clear_tool_results")

            # 2. summarise the oldest span: above the hard ratio, or still above soft
            summarized = False
            if aggressive or tokens_before >= hard or tokens_before - st.removed_tokens >= soft:
                summarized = await self._summarize(messages, settings=settings, agent=agent, window=window,
                                                   aggressive=aggressive, stats=st)
                if summarized:
                    steps.append("summarize")
                    first_edit = 0
            if not summarized and clearable and not cleared_now:
                clear(keep, "clear_tool_results")  # small gain, but it is all we can do

            # 3. overflow last resort: clear the latest tool results as well
            if aggressive and tokens_before - st.removed_tokens >= hard:
                clear(0, "clear_recent_tool_results")

            if not steps:
                messages[:] = snapshot
                log.warning("context for %s is at ~%d tokens (window %d) but nothing could be compacted",
                            agent, tokens_before, window)
                return None

            if first_edit is not None:
                st.thinking_dropped += self._drop_stale_thinking(messages, first_edit, settings.model)

            problems = validate_tool_pairing(messages, allow_pending=True)
            if problems and not pre_problems:
                raise ProviderError("context compaction produced an invalid history: " + "; ".join(problems[:3]))
            if problems:
                st.notes.append("history had tool-pairing problems before compaction: " + "; ".join(pre_problems[:2]))
        except BaseException as exc:
            messages[:] = snapshot
            _attach_spend(exc, st)
            raise

        after = max(0, tokens_before - st.removed_tokens)
        return CompactionResult(
            strategy="+".join(steps), tokens_before=int(tokens_before), tokens_after_estimate=int(after),
            usage=st.usage, cost_usd=st.cost, cleared_results=st.cleared, summarized_messages=st.summarized,
            thinking_dropped=st.thinking_dropped, spill_paths=st.paths, model=st.model, note="; ".join(st.notes),
            thinking_cleared=st.thinking_cleared)

    # ------------------------------------------------------------------ clearing

    def _spill_root(self) -> Path:
        if self.spill_dir is None:
            self.spill_dir = Path(tempfile.mkdtemp(prefix="vbt-context-"))
            log.warning("ContextManager has no spill_dir; cleared tool outputs go to %s", self.spill_dir)
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        return self.spill_dir

    @staticmethod
    def _is_cleared(content: Any) -> bool:
        return isinstance(content, str) and content.startswith(CLEARED_PREFIX)

    @staticmethod
    def _clear_thinking(m: Message, stats: _Stats) -> Message | None:
        """``m`` with the text of its reasoning blocks emptied (None if it had none)."""
        new_blocks: list[Any] = []
        changed = False
        for b in m.content:
            if isinstance(b, ThinkingBlock) and b.text:
                stats.removed_tokens += len(b.text) // CHARS_PER_TOKEN
                stats.thinking_cleared += 1
                native = {**(b.native or {}), "cleared_by_harness": True, "cleared_chars": len(b.text)}
                new_blocks.append(ThinkingBlock("", b.provider, native=native))
                changed = True
            else:
                new_blocks.append(b)
        return Message(m.role, new_blocks) if changed else None

    def _clear_tool_results(self, messages: list[Message], *, keep_recent: int, min_chars: int, stats: _Stats,
                            dry_run: bool = False, clear_thinking: bool = False) -> int | None:
        """Replace old tool-result content with stubs (and, with
        ``clear_thinking``, empty the reasoning of old assistant turns). Returns
        the index of the first edited message (None if nothing qualified)."""
        assistants = [i for i, m in enumerate(messages) if m.role == "assistant"]
        if keep_recent > 0:
            if len(assistants) <= keep_recent:
                return None
            boundary = assistants[-keep_recent]
        else:
            boundary = len(messages)
        first: int | None = None
        for i in range(boundary):
            m = messages[i]
            if m.role == "assistant":
                if clear_thinking:
                    cleared = self._clear_thinking(m, stats)
                    if cleared is not None:
                        messages[i] = cleared
                        first = i if first is None else first
                continue
            if m.role != "user" or not any(isinstance(b, ToolResult) for b in m.content):
                continue
            new_blocks: list[Any] = []
            changed = False
            for b in m.content:
                if (isinstance(b, ToolResult) and not self._is_cleared(b.content)
                        and content_chars(b.content) >= min_chars):
                    before = content_chars(b.content)
                    stub = f"{CLEARED_PREFIX} {before:,} chars]" if dry_run else self._spill(b, stats)
                    stats.removed_tokens += max(0, (before - len(stub)) // CHARS_PER_TOKEN)
                    stats.cleared += 1
                    new_blocks.append(ToolResult(b.tool_call_id, stub, b.is_error))
                    changed = True
                else:
                    new_blocks.append(b)
            if changed:
                messages[i] = Message(m.role, new_blocks)
                first = i if first is None else first
        return first

    def _spill(self, result: ToolResult, stats: _Stats) -> str:
        """Persist ``result``'s content (unless already spilled) and return the stub."""
        content = result.content
        text = content if isinstance(content, str) else "\n".join(
            p.text for p in content if isinstance(p, TextBlock))
        parts = [] if isinstance(content, str) else [p for p in content if isinstance(p, (ImagePart, DocumentPart))]
        n_chars = len(text)
        path = _harness_spill_path(text)  # the runtime already saved the full output
        root = None
        if path is None:
            root = self._spill_root()
            path = root / f"{_safe_name(result.tool_call_id)}.txt"
            k = 1
            while path.exists() and path.read_text(errors="replace") != text:
                path = root / f"{_safe_name(result.tool_call_id)}-{k}.txt"
                k += 1
            if not path.exists():
                path.write_text(text)
        stats.paths.append(str(path))
        part_paths: list[str] = []
        for j, p in enumerate(parts):
            root = root or self._spill_root()
            ext = _EXT.get(p.media_type, ".bin")
            pp = root / f"{_safe_name(result.tool_call_id)}.part{j}{ext}"
            try:
                pp.write_bytes(base64.b64decode(p.data_b64))
                part_paths.append(str(pp))
                stats.paths.append(str(pp))
            except (ValueError, OSError) as exc:  # undecodable payload: keep going, say so
                part_paths.append(f"(part {j} not saved: {exc})")
        extra = ""
        if part_paths:
            kinds = ", ".join(sorted({p.type for p in parts}))
            extra = f" + {len(parts)} {kinds} part(s) at {', '.join(part_paths)}"
        return (f"{CLEARED_PREFIX} {n_chars:,} chars{extra}; full output at {path}; "
                "use QueryToolOutput or Read]")

    # ------------------------------------------------------------------ summarising

    def _choose_tail(self, messages: list[Message], window: int, aggressive: bool) -> int | None:
        """Index where the kept tail starts (always an assistant message, >= 2)."""
        cands = [i for i in range(2, len(messages)) if messages[i].role == "assistant"]
        if not cands:
            return None
        max_rounds = 1 if aggressive else max(1, self.policy.keep_recent_calls // 2)
        budget = window * (0.10 if aggressive else 0.25)
        best = cands[-1]
        for i in reversed(cands):
            rounds = sum(1 for m in messages[i:] if m.role == "assistant")
            if rounds > max_rounds or estimate_tokens(messages[i:]) > budget:
                break
            best = i
        return best

    @staticmethod
    def _split_summary(text: str) -> tuple[str, str]:
        """(model summary, verbatim user section) of an earlier summary block."""
        body = text[len(SUMMARY_HEADER):].strip()
        if VERBATIM_MARKER in body:
            summ, verb = body.split(VERBATIM_MARKER, 1)
            return summ.strip(), verb.strip()
        return body, ""

    def _render_span(self, span: list[Message], earlier: str, budget_chars: int) -> str:
        lines: list[str] = []
        if earlier:
            lines.append("## Earlier summary (from a previous compaction)\n" + earlier)
        lines.append("## Transcript")
        for m in span:
            for b in m.content:
                if isinstance(b, TextBlock) and b.text:
                    who = "USER" if m.role == "user" else "ASSISTANT"
                    lines.append(f"{who}: {_clip(b.text, 20000 if m.role == 'user' else 8000)}")
                elif isinstance(b, ToolCall):
                    args = _clip(json.dumps(b.input, default=str, ensure_ascii=False), 2000)
                    lines.append(f"ASSISTANT CALLED {b.name} [{b.id}]: {args}")
                elif isinstance(b, ToolResult):
                    tag = "TOOL ERROR" if b.is_error else "TOOL RESULT"
                    lines.append(f"{tag} [{b.tool_call_id}]: {_clip(content_text(b.content), 3000)}")
                # thinking and provider-opaque blocks are dropped
        text = "\n\n".join(lines)
        if len(text) > budget_chars:
            head = int(budget_chars * 0.4)
            tail = budget_chars - head
            text = (text[:head] + f"\n\n[... {len(text) - budget_chars:,} chars of transcript omitted to fit the "
                    "summariser's window ...]\n\n" + text[-tail:])
        return text

    @staticmethod
    def _mechanical_summary(span: list[Message], earlier: str, reason: str, max_chars: int) -> str:
        out = [f"(Model summary unavailable: {reason}. Mechanical digest of the summarized span follows.)"]
        if earlier:
            out.append("Earlier summary: " + earlier)
        for m in span:
            for b in m.content:
                if m.role == "assistant" and isinstance(b, TextBlock) and b.text:
                    out.append("- assistant: " + _clip(b.text, 600))
                elif isinstance(b, ToolCall):
                    out.append(f"- called {b.name}: " + _clip(json.dumps(b.input, default=str), 300))
                elif isinstance(b, ToolResult):
                    out.append(f"- result [{b.tool_call_id}]{' (error)' if b.is_error else ''}: "
                               + _clip(content_text(b.content), 300))
        return _clip("\n".join(out), max_chars)

    async def _summarize(self, messages: list[Message], *, settings: ModelSettings, agent: str, window: int,
                         aggressive: bool, stats: _Stats) -> bool:
        if len(messages) < 3 or messages[0].role != "user" or messages[0].tool_results:
            return False
        tail_start = self._choose_tail(messages, window, aggressive)
        if tail_start is None:
            return False
        head, span = messages[0], messages[1:tail_start]
        if not span:
            return False

        earlier_summaries: list[str] = []
        earlier_verbatim: list[str] = []
        head_blocks: list[Any] = []
        for b in head.content:
            if isinstance(b, TextBlock) and b.text.startswith(SUMMARY_HEADER):
                s, v = self._split_summary(b.text)
                earlier_summaries.append(s)
                if v:
                    earlier_verbatim.append(v)
            else:
                head_blocks.append(b)
        earlier = "\n\n".join(s for s in earlier_summaries if s)

        pol = self.policy
        budget_chars = int(max(20_000, min(400_000, (window - 2 * pol.summary_max_tokens - 4000) * 3)))
        transcript = self._render_span(span, earlier, budget_chars)
        words = max(200, int(pol.summary_max_tokens * 0.7))
        prompt = (f"Agent: {agent}\nSummarize the following conversation span in at most ~{words} words.\n\n"
                  + transcript)
        # one-off request: caching its prefix would only add a cache-write surcharge
        extra: dict[str, Any] = {"agent_name": SUMMARIZER_AGENT, "prompt_cache": False}
        if (settings.extra or {}).get("context_window_tokens"):
            extra["context_window_tokens"] = settings.extra["context_window_tokens"]
        s_settings = replace(settings, thinking=False, temperature=None, extra=extra,
                             effort="low" if settings.effort is not None else None,
                             max_tokens=max(1024, pol.summary_max_tokens * 2))
        summary = ""
        try:
            resp = await complete_with_retry(self.provider, policy=self.retry, settings=s_settings,
                                             system=SUMMARIZER_SYSTEM, messages=[Message.user(prompt)], tools=[])
            stats.usage = stats.usage + resp.usage
            stats.cost += resp.cost_usd
            stats.model = resp.served_model or resp.model
            summary = resp.message.text.strip()
            if not summary:
                stats.notes.append("summariser returned no text; used a mechanical digest")
                summary = self._mechanical_summary(span, earlier, "empty summariser reply", pol.summary_max_tokens * 4)
        except ProviderError as exc:  # includes retry exhaustion and overflow of the summariser itself
            log.warning("context summariser failed for %s: %s", agent, exc)
            stats.notes.append(f"summariser failed ({type(exc).__name__}); used a mechanical digest")
            summary = self._mechanical_summary(span, earlier, type(exc).__name__, pol.summary_max_tokens * 4)

        user_turns = []
        for m in span:
            if m.role != "user":
                continue
            t = "\n\n".join(b.text for b in m.content if isinstance(b, TextBlock) and b.text)
            if t:
                user_turns.append(_clip(t, 20000))
        verbatim = [v for v in earlier_verbatim if v] + [f"[user message]\n{t}" for t in user_turns]

        block = (f"{SUMMARY_HEADER}\nThe earlier part of this conversation was condensed by the harness to fit the "
                 f"context window. Cleared tool outputs remain on disk at the paths named below.\n\n{summary}")
        if verbatim:
            block += f"\n\n{VERBATIM_MARKER}\n" + "\n\n".join(verbatim)

        removed_chars = sum(message_chars(m) for m in span) + sum(len(s) for s in earlier_summaries)
        stats.removed_tokens += max(0, (removed_chars - len(block)) // CHARS_PER_TOKEN)
        stats.summarized += len(span)
        stats.thinking_dropped += sum(1 for m in span for b in m.content if isinstance(b, ThinkingBlock))
        messages[:] = [Message("user", head_blocks + [TextBlock(block)])] + messages[tail_start:]
        return True

    # ------------------------------------------------------------------ thinking

    def _drop_stale_thinking(self, messages: list[Message], first_edit: int, model: str) -> int:
        if not self._capabilities(model).history_bound_thinking:
            return 0
        dropped = 0
        for i in range(max(0, first_edit), len(messages)):
            m = messages[i]
            if m.role != "assistant":
                continue
            kept = [b for b in m.content if not isinstance(b, ThinkingBlock)]
            if len(kept) != len(m.content):
                dropped += len(m.content) - len(kept)
                messages[i] = Message(m.role, kept or [TextBlock("(reasoning removed by harness compaction)")])
        return dropped


__all__ = ["CLEARED_PREFIX", "CompactionResult", "ContextManager", "ContextPolicy", "SUMMARIZER_AGENT",
           "compaction_spend",
           "SUMMARY_HEADER", "content_chars", "estimate_tokens"]
