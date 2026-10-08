"""Runtime: the provider-neutral agent loop, delegation, and shared services.

One ``Runtime`` serves one ``Run``: it owns the provider, the tool registry
(built-ins + provenance + MCP-bridged tools + Task + QueryToolOutput), and
spawns agents with isolated contexts, as the Claude Agent SDK did for the
original system.

Invariants the loop keeps on every exit path (normal end, refusal, context
overflow, budget stop, turn limit, cancellation, crash):

* **History pairing** -- every assistant tool call in a persistent history is
  answered by a tool result (``repair_history``); the history is validated
  before every provider call.
* **Accounting** -- spend is charged when it happens, to the run ledger, every
  enclosing budget scope (``vbt.budget``) and the invocation's accumulator, so
  failed attempts are never lost.
* **Records** -- every invocation traces ``agent_start``/``agent_end`` with an
  ``agent_run_id``; tool inputs are traced inline or spilled with a sha256;
  long outputs are spilled to ``logs/tool_outputs``; the full message list is
  written to ``logs/agents/<agent_run_id>.jsonl``.

Text of a run: ``AgentResult.text`` is the final-report buffer (the last
assistant message plus any max-tokens continuations; reset when tools are
called) -- what a delegating agent receives. ``AgentResult.full_text`` is every
assistant text since the last harness nudge -- what the CSO said this turn.

Config (``limits:``; in-code defaults): ``max_cso_turns`` (``max_agent_turns`` or
100), ``max_specialist_turns`` (400; an agent's own ``max_turns`` wins),
``delegation_timeout_s`` (None), ``trace_output_chars`` (10000),
``trace_input_inline_chars`` (16000), ``tool_output_max_chars`` (40000),
``max_turn_cost_usd`` / ``max_turn_tokens`` (caps of the ``turn`` scope; either
one stops it), ``cached_token_weight`` (0.1: weight of cached input tokens in
token budgets). ``retry:`` and ``context:`` sections configure ``RetryPolicy``
and ``ContextPolicy``.

Model-facing guards (local open-weight models need them more than Claude):

* every request carries ``extra['session_key']`` (the invocation id; the CSO's
  is stable for the whole session) so a data-parallel server keeps an agent on
  one replica and its prefix cache;
* tool calls whose arguments were not valid JSON
  (``ToolCall.native['invalid_arguments']``) or miss required properties / have
  wrong top-level types are not run: the model gets an error result asking it to
  re-issue the call (recorded as a model error, never a data-source failure);
* an empty reply (no text, no tool call) right after tool results gets one
  harness nudge per invocation;
* ``run_agent(force_tool=...)`` makes the final allowed call a forced call of
  that tool (``extra['tool_choice']`` when the provider supports it); the final
  no-tool call (turn limit, budget grace) sends ``tool_choice 'none'``. When
  the agent's innermost budget scope runs out (e.g. a bulk item's
  ``limits.max_item_tokens``), such an agent still gets that one forced call
  (budget-exempt) before ``BudgetExceeded`` propagates, so the evidence it
  gathered is submitted instead of lost;
* ``BudgetExceeded`` raised out of ``run_agent`` carries the partial
  :class:`AgentResult` as ``exc.agent_result`` (model calls made so far).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from . import budget, failures
from .agents import AgentDefinition, load_roster, system_prompt_parts
from .budget import BudgetExceeded, CostScope, InvocationCost, open_scope  # noqa: F401  (re-exported)
from .config import base_tool_env, resolve_path
from .context import ContextManager, ContextPolicy, CompactionResult, compaction_spend
from .events import EventBus, EventCallback, preview
from .providers import create_provider
from .providers.base import (
    ContextOverflowError,
    DocumentPart,
    ImagePart,
    LLMProvider,
    Message,
    ModelResponse,
    ModelSettings,
    StopReason,
    SystemSegment,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    content_text,
    unanswered_tool_calls,
    validate_tool_pairing,
)
from .providers.retry import RetryPolicy, complete_with_retry
from .session import Run, write_json_atomic
from .tools.base import (
    QUERY_TOOL,
    Tool,
    ToolContext,
    ToolFailure,
    ToolRegistry,
    maybe_parse_json,
    query_tool_output_tool,
    schema,
    shrink_json,
    to_text,
    truncate_text,
    truncation_note,
)
from .tools.builtin import builtin_tools
from .tools.mcp_bridge import DATA_SERVER, MCPBridge, MCPServerConfig
from .tools.provenance import provenance_tools

log = logging.getLogger(__name__)

__all__ = ["Runtime", "AgentResult", "BudgetExceeded", "repair_history", "AGENT_STATUSES"]

AGENT_STATUSES = ("completed", "turn_limit", "refusal", "context_exceeded", "timeout", "cancelled", "budget", "error")

TURN_LIMIT_MSG = "[Harness] You have reached the turn limit. Do not call tools; write your final report now."
FORCE_TOOL_MSG = ("[Harness] You have reached the turn limit. Call {tool} now with your final result; no other tool "
                  "will run.")
BUDGET_GRACE_MSG = ("[Harness] The turn budget is exhausted. Do not call tools; write your final synthesis from the "
                    "evidence gathered so far and state what is missing.")
BUDGET_FORCE_MSG = ("[Harness] The budget for this task is exhausted. Call {tool} now with your final result, filled "
                    "from the evidence gathered so far (use the schema's unknown/null values where evidence is "
                    "missing); no other tool will run.")
CONTINUE_MSG = "[Harness] Your response hit the output limit. Continue exactly where you stopped, concisely."
EMPTY_REPLY_MSG = ("[Harness] Your last reply was empty. Continue: call the next tool or write your final report.")
TRUNCATED_INPUT_MSG = "Tool input was truncated at max_tokens; retry with a shorter input."
INVALID_JSON_MSG = ("Your arguments for {tool} were not valid JSON ({error}). Re-issue the call with a single JSON "
                    "object matching the schema.")
INVALID_ARGS_MSG = ("Your arguments for {tool} do not match its schema: {problems}. Re-issue the call with a single "
                    "JSON object matching the schema.")
AUDIT_NOTE = ("[Harness] Audit recording failed for this call: {msg}. Report incomplete evidence and do not invent "
              "citations.")

#: Harness tools declared ``strict`` (schema-constrained arguments where the provider supports it).
STRICT_TOOLS = frozenset({"mcp__provenance__record_claims"})

#: Run-relative prefixes that resolve against the run directory, not the agent workspace.
RUN_PREFIXES = ("work", "inputs", "evidence", "report", "logs", ".claude")
CODE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
CODE_EXTS = frozenset({".py", ".r", ".sh", ".ipynb", ".sql"})
_INLINE_CODE_RE = re.compile(
    r"<<-?\s*['\"]?[A-Za-z_][\w]*"                          # heredoc
    r"|\bpython[0-9.]*\s+(?:-[A-Za-z]+\s+)*-c\b"             # python -c
    r"|\bRscript\s+(?:-{1,2}[\w-]+(?:=\S+)?\s+)*-e\b"        # Rscript -e
    r"|(?:^|[\s;&|(])R\s+(?:-{1,2}[\w-]+\s+)*-e\b"           # R -e
)
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")
#: A trailing ``[Output truncated: ...]`` note (``truncation_note``). It must stay the last thing in a
#: tool result: ``context._SPILLED_RE`` anchors on it to find the spill file.
_TRUNCATION_NOTE_RE = re.compile(r"\[Output truncated: [^\n]*\]\s*\Z")

#: (agent_run_id, parent_run_id) of the invocation whose code is running.
_RUN_IDS: ContextVar[tuple[str, str | None] | None] = ContextVar("vbt_agent_run_ids", default=None)


def _safe(name: str) -> str:
    return _SAFE_RE.sub("_", str(name))[:160] or "x"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _add_note(text: str, note: str) -> str:
    """``text`` with ``note`` added, before a trailing truncation note (never after it)."""
    m = _TRUNCATION_NOTE_RE.search(text)
    if m is None:
        return text + "\n\n" + note
    return text[:m.start()].rstrip("\n") + "\n\n" + note + "\n\n" + text[m.start():]


def _accepted_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments ``fn`` accepts (all of them when it takes ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _is_content_parts(result: Any) -> bool:
    """A tool returned provider content parts (e.g. MCP text + images)."""
    return (isinstance(result, list) and bool(result)
            and all(isinstance(x, (TextBlock, ImagePart, DocumentPart)) for x in result))


def _last_assistant_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "assistant" and m.text:
            return m.text
    return ""


# ---------------------------------------------------------------------------
# Tool-argument checks (before a tool runs)
# ---------------------------------------------------------------------------

_JSON_TYPE_NAMES = ("string", "integer", "number", "boolean", "array", "object", "null")


def _json_type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _type_matches(value: Any, t: str) -> bool:
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or \
            (isinstance(value, float) and value.is_integer())
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "array":
        return isinstance(value, (list, tuple))
    if t == "object":
        return isinstance(value, Mapping)
    if t == "null":
        return value is None
    return True  # unknown type keyword: do not judge


def _declared_types(prop: Any) -> list[str] | None:
    """The top-level JSON types a property schema declares, or None when it
    declares none (anyOf/oneOf/$ref/...: not checked here)."""
    if not isinstance(prop, Mapping):
        return None
    t = prop.get("type")
    types = [t] if isinstance(t, str) else list(t) if isinstance(t, list) else []
    types = [x for x in types if isinstance(x, str) and x in _JSON_TYPE_NAMES]
    return types or None


def _reduced_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Top-level ``required`` and per-property ``type`` of a tool's input schema:
    the part the harness enforces (the tool and the server check the rest)."""
    props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
    required = [str(r) for r in (schema.get("required") or []) if isinstance(r, str)]
    out_props: dict[str, Any] = {}
    for name, prop in props.items():
        types = _declared_types(prop)
        if types:
            if str(name) not in required and "null" not in types:
                types = types + ["null"]   # models often send null for an omitted optional argument
            out_props[str(name)] = {"type": types if len(types) > 1 else types[0]}
    return {"type": "object", "required": required, "properties": out_props}


def _jsonschema_problems(reduced: dict[str, Any], args: Mapping[str, Any]) -> list[str] | None:
    """Problems found by ``jsonschema`` (None when it is not installed)."""
    try:
        import jsonschema  # optional (vbt-harness[tools])
    except ImportError:
        return None
    try:
        validator_cls = jsonschema.validators.validator_for(reduced)
        validator = validator_cls(reduced)
        out = []
        for e in sorted(validator.iter_errors(dict(args)), key=lambda e: list(e.path)):
            where = ".".join(str(p) for p in e.path)
            if e.validator == "required":
                out.append(e.message.replace("is a required property", "is required"))
            elif e.validator == "type" and where:
                out.append(f"{where!r} must be {_type_phrase(e.validator_value)}, got "
                           f"{_json_type_of(e.instance)}")
            else:
                out.append(f"{where + ': ' if where else ''}{e.message}")
        return out
    except Exception:  # noqa: BLE001 - a validator problem must never block a tool call
        log.debug("jsonschema validation failed", exc_info=True)
        return None


def _type_phrase(types: Any) -> str:
    ts = [t for t in ([types] if isinstance(types, str) else list(types or [])) if t != "null"] or ["null"]
    return " or ".join(ts)


def tool_argument_problems(schema: Mapping[str, Any] | None, args: Any) -> list[str]:
    """Missing required properties and wrong top-level JSON types of ``args``
    against a tool's input ``schema`` ([] when fine or when the schema declares
    nothing checkable). Uses ``jsonschema`` when installed, else a minimal
    required/type check with the same scope. Nested values are not checked
    (the tool and, for strict tools, the server's grammar do that)."""
    if not isinstance(schema, Mapping):
        return []
    if not isinstance(args, Mapping):
        return [f"arguments must be a JSON object, got {_json_type_of(args)}"]
    if schema.get("type") not in (None, "object"):
        return []
    reduced = _reduced_schema(schema)
    found = _jsonschema_problems(reduced, args)
    if found is not None:
        return found
    problems = [f"{r!r} is required" for r in reduced["required"] if r not in args]
    for name, prop in reduced["properties"].items():
        if name not in args:
            continue
        types = prop["type"] if isinstance(prop["type"], list) else [prop["type"]]
        if not any(_type_matches(args[name], t) for t in types):
            problems.append(f"{name!r} must be {_type_phrase(types)}, got {_json_type_of(args[name])}")
    return problems


# ---------------------------------------------------------------------------
# History repair
# ---------------------------------------------------------------------------


def repair_history(messages: list[Message], reason: str = "interrupted") -> int:
    """Make ``messages`` satisfy the tool_use/tool_result pairing rules, in place.

    * drops empty messages;
    * moves tool results to the front of their user message, turns duplicate
      or orphan results (no matching call in the preceding assistant message)
      and results inside assistant messages into text;
    * answers every unanswered tool call with
      ``ToolResult(id, 'Not executed: <reason>', is_error=True)`` at the start of
      the following user message, or in a new user message.

    A valid history is left untouched. Returns the number of fixes.
    """
    fixes = 0
    if not validate_tool_pairing(messages) and all(m.content for m in messages):
        return 0
    kept = [m for m in messages if m.content]
    fixes += len(messages) - len(kept)
    if fixes:
        messages[:] = kept
    for i, m in enumerate(messages):
        if m.role == "assistant":
            stray = [b for b in m.content if isinstance(b, ToolResult)]
            if stray:
                rest = [b for b in m.content if not isinstance(b, ToolResult)]
                note = TextBlock("[harness: removed tool result(s) from an assistant message: "
                                 + ", ".join(b.tool_call_id for b in stray) + "]")
                messages[i] = Message("assistant", rest or [note])
                fixes += len(stray)
            continue
        results = [b for b in m.content if isinstance(b, ToolResult)]
        if not results:
            continue
        prev = messages[i - 1] if i > 0 else None
        expected = {c.id for c in prev.tool_calls} if prev is not None and prev.role == "assistant" else set()
        seen: set[str] = set()
        keep: list[Any] = []
        dropped: list[str] = []
        for r in results:
            if r.tool_call_id in expected and r.tool_call_id not in seen:
                keep.append(r)
                seen.add(r.tool_call_id)
            else:
                kind = "duplicate" if r.tool_call_id in seen else "orphan"
                dropped.append(f"[harness: {kind} tool result {r.tool_call_id}: {content_text(r.content)[:2000]}]")
        others = [b for b in m.content if not isinstance(b, ToolResult)]
        new = keep + others + ([TextBlock("\n".join(dropped))] if dropped else [])
        if [id(b) for b in new] != [id(b) for b in m.content]:
            messages[i] = Message("user", new)
            fixes += len(dropped) or 1
    groups: dict[int, list[ToolCall]] = {}
    for idx, c in unanswered_tool_calls(messages):
        groups.setdefault(idx, []).append(c)
    for idx in sorted(groups, reverse=True):
        stubs = [ToolResult(c.id, f"Not executed: {reason}", True) for c in groups[idx]]
        nxt = idx + 1
        if nxt < len(messages) and messages[nxt].role == "user":
            m = messages[nxt]
            res = [b for b in m.content if isinstance(b, ToolResult)]
            oth = [b for b in m.content if not isinstance(b, ToolResult)]
            messages[nxt] = Message("user", res + stubs + oth)
        else:
            messages.insert(nxt, Message("user", stubs))
        fixes += len(stubs)
    return fixes


# ---------------------------------------------------------------------------
# Results and loop state
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    agent: str
    text: str                        # final-report buffer (what a delegating agent receives)
    messages: list[Message]
    cost_usd: float = 0.0            # this invocation's own spend (model calls + tool-side costs)
    model_calls: int = 0
    tool_calls: int = 0
    tool_errors: list[dict[str, Any]] = field(default_factory=list)   # every failed call: {tool, input, error}
    stop_reason: str = "end_turn"
    delegations: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    full_text: str = ""              # all assistant text since the last harness nudge
    drafts: list[str] = field(default_factory=list)                    # text that preceded a harness nudge
    status: str = "completed"        # one of AGENT_STATUSES
    invocation_id: str = ""
    parent_invocation_id: str | None = None
    transcript_path: str | None = None
    unresolved_data_failures: list[dict[str, Any]] = field(default_factory=list)
    recovered_errors: int = 0
    other_errors: int = 0
    compactions: int = 0
    retries: int = 0
    fallback_count: int = 0
    error: str | None = None
    tokens: float = 0.0              # budget tokens of this invocation's own model calls (vbt.budget)
    nudges: list[str] = field(default_factory=list)   # harness nudges sent (empty_reply, ...)


def _has_tool_blocks(messages: list[Message]) -> bool:
    return any(isinstance(b, (ToolCall, ToolResult)) for m in messages for b in m.content)


def _catalog_fault(exc: BaseException) -> bool:
    """A descriptor/overlay/catalog error (configuration), as opposed to a missing package."""
    try:
        from .datalayer.catalog import CatalogError
        from .datalayer.descriptor.load import DescriptorError
    except Exception:  # noqa: BLE001
        return False
    return isinstance(exc, (DescriptorError, CatalogError))


@dataclass
class _Loop:
    agent: AgentDefinition
    depth: int
    messages: list[Message]
    tools: list[Tool]
    by_name: dict[str, Tool]
    settings: Any
    system: Any
    result: AgentResult
    inv: str
    parent: str | None
    stream_text: bool
    after_end_turn: Callable[[list[Message]], Awaitable[str | None]] | None
    max_turns: int
    acc: InvocationCost
    segments: list[str] = field(default_factory=list)
    report: str = ""
    continuing: bool = False
    turns: int = 0
    last_usage: Any = None
    calls: list[dict[str, Any]] = field(default_factory=list)   # call records for failure resolution
    pending_budget: BudgetExceeded | None = None
    # allow_tools=False over a history that already holds tool_use/tool_result
    # blocks: the specs are still sent (the API rejects such a request without
    # tool definitions) but no call is executed.
    inert_tools: list[Tool] = field(default_factory=list)
    refused_rounds: int = 0
    force_tool: str | None = None      # forced on the final allowed call (run_agent(force_tool=...))
    task_given: bool = False           # run_agent got a task message (it is the instruction of call 1)
    empty_nudged: bool = False         # the empty-reply nudge was sent (once per invocation)
    call_mode: str | None = None       # None | 'force' | 'none': tool_choice of the next call
    budget_forced: BudgetExceeded | None = None  # the budget-exempt forced call was granted for this

    @property
    def specs(self):
        return [t.spec for t in (self.tools or self.inert_tools)]

    @property
    def has_read(self) -> bool:
        return "Read" in self.by_name


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class Runtime:
    def __init__(self, config: dict[str, Any], run: Run, provider: LLMProvider | None = None,
                 on_event: EventCallback | None = None):
        self.config = config
        self.run = run
        # null options are unset (a profile switching provider resets the other adapter's options to null)
        self.provider = provider or create_provider(
            config["provider"]["name"],
            **{k: v for k, v in (config["provider"].get("options") or {}).items() if v is not None})
        self.bus = EventBus()
        self.on_event = on_event
        if on_event is not None:
            self.bus.subscribe(on_event)
        self.registry = ToolRegistry()
        self.cso, self.agents = load_roster(config)
        limits = config.get("limits") or {}
        self.max_turns = int(limits.get("max_agent_turns", 100))
        self.max_depth = int(limits.get("max_delegation_depth", 1))
        self.tool_output_max = int(limits.get("tool_output_max_chars", 40000))
        self.trace_output_chars = int(limits.get("trace_output_chars") or 10000)
        self.trace_input_inline_chars = int(limits.get("trace_input_inline_chars") or 16000)
        timeout = limits.get("delegation_timeout_s")
        self.delegation_timeout_s = float(timeout) if timeout not in (None, "", 0) else None
        self._parallel = asyncio.Semaphore(int(limits.get("max_parallel_agents", 8)))
        self.turn_budget_usd = float(limits.get("max_turn_cost_usd") or 0) or None
        self.turn_budget_tokens = float(limits.get("max_turn_tokens") or 0) or None
        self.cached_token_weight = budget.cached_token_weight(config)
        paths = config.get("paths") or {}
        self.read_roots = [Path(resolve_path(p)).resolve() for p in paths.get("read_roots", []) if p]
        self.skill_roots = [Path(resolve_path(p)).resolve() for p in paths.get("skills", []) if p]
        self.mcp: MCPBridge | None = None
        #: Servers that started but lack their reference data (preflight with
        #: --allow-missing-data); told to every agent next to the MCP failures.
        self.degraded_servers: dict[str, str] = {}
        #: The data-layer gateway in front of every MCP call (``start_mcp``), or None
        #: (``data.enabled: false``, mode ``off``, or it could not be built: ``gateway_error``).
        self.gateway: Any = None
        self.gateway_error: str | None = None
        self.gateway_refused: str | None = None     # set when strict refuses servers over a broken catalog
        self._data_settings: Any = None
        #: Tool-scoped readiness from the data layer: {tool: reason} of unready tools.
        self.tool_readiness: dict[str, Any] = {}
        #: {skill name: sha256} of the skills materialised into <run>/.claude/skills.
        self.skill_hashes: dict[str, str] = {}
        self.delegation_log: list[dict[str, Any]] = []
        self.retry_policy = RetryPolicy.from_config(config.get("retry"))
        self.context = ContextManager(self.provider, ContextPolicy.from_config(config.get("context")),
                                      spill_dir=run.dir / "logs" / "tool_outputs" / "cleared",
                                      retry=self.retry_policy)
        self._outstanding: set[asyncio.Future] = set()
        self._live: dict[str, AgentResult | None] = {}
        self._cancel_reasons: dict[str, str] = {}
        self._system_cache: dict[str, Any] = {}
        self._known_agents: dict[str, AgentDefinition] = {}
        self.registry.extend(builtin_tools(skill_roots=self.skill_roots))
        self._materialize_skills()
        for t in provenance_tools():
            if t.name in STRICT_TOOLS:
                t.strict = True
            self.registry.add(t)
        self.registry.add(self._task_tool())
        self.registry.add(self._list_tools_tool())
        self.registry.add(query_tool_output_tool())
        try:
            from .bulk import bulk_settings
            if bulk_settings(config).get("dispatch_enabled"):
                from .bulk_dispatch import bulk_dispatch_tools
                self.registry.extend(bulk_dispatch_tools(self))
        except Exception:  # noqa: BLE001 - optional CSO tools must never block a runtime
            log.exception("registering bulk dispatch tools failed")

    # ------------------------------------------------------------------ setup

    def _materialize_skills(self) -> None:
        """Link every skill into <run>/.claude/skills and record {name: sha256}."""
        try:
            from .tools.skills import materialize
            self.skill_hashes = materialize(self.run.dir, self.skill_roots)
        except Exception as exc:  # noqa: BLE001 - skills are optional; never block a runtime
            self._note_audit_error(f"skill materialize: {type(exc).__name__}: {exc}")

    @property
    def search_backend(self):
        """The WebSearch backend (``web.search``: provider-native, SearxNG, Brave or
        None); resolved on every call (cheap, no network)."""
        from .tools.search_backends import resolve_search_backend
        return resolve_search_backend(self.config, self.provider)

    def supports_tool_choice(self, model: str | None = None) -> bool:
        """The provider honours ``extra['tool_choice']`` (forced / disabled tool calls)."""
        try:
            return bool(getattr(self.provider.capabilities(model), "tool_choice", False))
        except TypeError:  # providers written against capabilities() without a model argument
            try:
                return bool(getattr(self.provider.capabilities(), "tool_choice", False))
            except Exception:  # noqa: BLE001
                return False
        except Exception:  # noqa: BLE001
            return False

    def tool_env(self, ctx: ToolContext | None = None) -> dict[str, str]:
        env = base_tool_env(self.config)
        env.update({"VBT_RUN_DIR": str(self.run.dir), "MCP_OUTPUT_DIR": str(self.run.mcp_output_dir)})
        if ctx is not None:
            env["VBT_AGENT"] = ctx.agent
            env["VBT_WORKSPACE"] = str(ctx.workspace)
        return env

    def run_variables(self) -> dict[str, str]:
        """``${run.*}`` values for data-layer descriptors (tables a tool materialises in this run)."""
        return {"dir": str(self.run.dir), "run_id": str(getattr(self.run, "run_id", "") or ""),
                "mcp_output_dir": str(self.run.mcp_output_dir)}

    def _gateway_unavailable(self, reason: str) -> None:
        self.gateway, self.gateway_error = None, reason
        log.warning("data gateway unavailable; MCP calls run unguarded: %s", reason)
        try:
            self.run.trace("data_gateway_unavailable", reason=reason)
        except Exception:  # noqa: BLE001 - tracing must not block MCP startup
            log.debug("tracing data_gateway_unavailable failed", exc_info=True)

    def data_settings(self) -> Any:
        """``vbt.datalayer.settings.DataSettings`` of this run's config (parsed once)."""
        if self._data_settings is None:
            from .datalayer.settings import DataSettings
            self._data_settings = DataSettings.from_config(self.config)
        return self._data_settings

    def _build_gateway(self) -> Any:
        """The data gateway when ``data.enabled`` and the mode is not ``off``; None otherwise.
        A gateway that cannot be built is reported (warning + ``data_gateway_unavailable``
        trace event) and the MCP servers start without it, except that a catalog fault under
        ``when_service_down: strict`` refuses the servers the gateway would enforce
        (``gateway_refused``). A descriptor or overlay file that does not load is quarantined on its own
        (R8): the gateway is built from the rest, each file is traced (``data_catalog_quarantined``),
        and only the servers and tools that depend on it are refused. A missing or empty
        descriptors/overlays directory is warned about (``data_catalog_missing``): every server would run
        on the generic guard."""
        self.gateway, self.gateway_error, self._data_settings = None, None, None
        self.gateway_refused = None
        try:
            settings = self.data_settings()
        except Exception as exc:  # noqa: BLE001
            self._gateway_unavailable(f"data settings: {type(exc).__name__}: {exc}")
            return None
        if not settings.enabled or settings.gateway.mode == "off":
            return None
        for what, d in (("descriptors_dir", settings.descriptors_dir), ("overlays_dir", settings.overlays_dir)):
            # an empty catalog builds, but leaves every server on the generic guard: say so
            if not Path(d).is_dir() or not any(Path(d).glob("*.y*ml")):
                reason = f"data.{what} {d} is missing or holds no YAML files: no tool has a reviewed binding"
                log.warning("data catalog: %s", reason)
                with contextlib.suppress(Exception):
                    self.run.trace("data_catalog_missing", setting=what, path=str(d), reason=reason)
        try:
            from . import datalayer
            gateway = datalayer.build_gateway(self.config, run=self.run_variables())
            for q in getattr(getattr(gateway, "catalog", None), "quarantined", None) or []:
                log.warning("data catalog: %s %s quarantined: %s", q.kind, q.path, q.summary)
                with contextlib.suppress(Exception):
                    self.run.trace("data_catalog_quarantined", **q.to_json())
            return gateway
        except Exception as exc:  # noqa: BLE001 - e.g. the gateway package is not installed yet
            reason = f"{type(exc).__name__}: {exc}"[:1000]
            self._gateway_unavailable(reason)
            if settings.gateway.when_service_down == "strict" and _catalog_fault(exc):
                # a malformed descriptor or overlay is a configuration fault: under strict the servers the
                # gateway would enforce are refused, never started unguarded
                self.gateway_refused = reason
            return None

    async def start_mcp(self, servers: list[str] | None = None) -> dict[str, str]:
        """Launch configured MCP servers (optionally a subset) behind the data gateway
        (when enabled; its own servers, such as the ``data`` child, are added unless the
        config names them). Returns failures."""
        raw_specs = [dict(s) for s in (self.config.get("mcp_servers") or {}).get("servers", [])
                     if isinstance(s, Mapping)]
        gateway = self._build_gateway()
        if gateway is not None and "gateway" not in _accepted_kwargs(MCPBridge, {"gateway": gateway}):
            self._gateway_unavailable("MCPBridge has no gateway seam in this checkout")
            gateway = None
        enabled = [str(s.get("name")) for s in raw_specs if s.get("enabled", True) is not False]
        refused = self._refused_servers(enabled, gateway)
        starting = [n for n in enabled if n not in refused and (not servers or n in set(servers))]
        if gateway is not None and refused and not starting:
            # every server the gateway would guard is refused (their overlays are quarantined): there is
            # nothing to guard, so neither the gateway nor its data child is started
            self._gateway_unavailable(f"no server left to guard: {self.gateway_refused}")
            gateway = None
        extra_names: set[str] = set()
        guarded = bool(starting) if refused else bool(enabled)
        if gateway is not None and guarded:          # the data child serves the gateway of real servers
            try:
                configured = {s.get("name") for s in raw_specs}
                for s in gateway.extra_servers() or []:
                    if isinstance(s, Mapping) and s.get("name") not in configured:  # an explicit entry wins
                        raw_specs.append(dict(s))
                        extra_names.add(str(s.get("name")))
            except Exception as exc:  # noqa: BLE001
                log.warning("data gateway extra_servers failed: %s", exc)
        # a refused server is not handed to the bridge at all: a direct call would otherwise start it lazily,
        # and without a gateway it would then be served unguarded
        specs = [MCPServerConfig(**{k: v for k, v in s.items() if k in MCPServerConfig.__dataclass_fields__})
                 for s in raw_specs if str(s.get("name")) not in refused]
        seam: dict[str, Any] = {"on_tools_changed": self._on_tools_changed}
        if gateway is not None:
            seam["gateway"] = gateway
        seam = _accepted_kwargs(MCPBridge, seam)  # a bridge without the seam gets neither
        self.gateway = gateway
        self.mcp = MCPBridge(specs, extra_env=self.tool_env(), log_dir=self.run.dir / "logs" / "mcp",
                             options=self.config.get("mcp") or {}, on_event=self._mcp_event, **seam)
        wanted = (set(servers) | extra_names) if servers else None
        if refused:
            wanted = (wanted or {s.name for s in specs}) - set(refused)
        tools = await self.mcp.start(wanted) if wanted != set() else []
        if refused:
            self.mcp.failures.update(refused)
            self.run.trace("data_gateway_refused", servers=sorted(refused), reason=self.gateway_refused)
        self.registry.extend(tools)
        self.run.trace("mcp_started", servers=sorted(self.mcp.sessions), failures=self.mcp.failures,
                       n_tools=len(tools), data_gateway=getattr(gateway, "mode", None) if gateway else None)
        self._system_cache.clear()  # the unavailable-server list may have changed
        return self.mcp.failures

    def _refused_servers(self, names: list[str], gateway: Any = None) -> dict[str, str]:
        """Of the enabled servers ``names``, those not started under ``when_service_down: strict`` and an
        enforcing gateway: every enforced server when the catalog could not be built at all
        (``gateway_refused`` without a gateway), else only those whose own overlay is quarantined (R8; the
        data child is never refused: its public tools are refused per call). Sets ``gateway_refused`` to
        the reason when a quarantine refuses one."""
        try:
            settings = self.data_settings()
        except Exception:  # noqa: BLE001
            return {}
        if settings.gateway.mode != "enforce":
            return {}
        if gateway is None:
            if not getattr(self, "gateway_refused", None):
                return {}
            why = (f"not started: the data catalog could not be loaded ({self.gateway_refused}); fix the "
                   "descriptor or overlay (`vbt ds lint`), or set data.gateway.when_service_down: lenient")
            return {name: why[:1200] for name in names if settings.gateway.enforces(name)}
        if settings.gateway.when_service_down != "strict":
            return {}
        try:
            quarantined = gateway.catalog.quarantined_servers()
        except Exception:  # noqa: BLE001 - a catalog without quarantine support refuses nothing
            return {}
        reasons = {name: "; ".join(q.reason for q in quarantined[name]) for name in names
                   if quarantined.get(name) and name != DATA_SERVER and settings.gateway.enforces(name)}
        if reasons:
            self.gateway_refused = "; ".join(f"{name}: {why}" for name, why in sorted(reasons.items()))[:1000]
        return {name: (f"not started: the data catalog could not be loaded for this server ({why}); the file is "
                       "quarantined: fix it (`vbt ds lint`), or set data.gateway.when_service_down: lenient")[:1200]
                for name, why in reasons.items()}

    def _on_tools_changed(self, tools: list[Tool]) -> None:
        """Tools the bridge registered or updated after start (a restarted server, a late
        listing): put them in the registry and rebuild prompts that list tools."""
        try:
            self.registry.extend(list(tools or []))
        finally:
            self._system_cache.clear()

    @property
    def data_layer_enforcing(self) -> bool:
        """The gateway enforces its contract (agents get ``data_layer_addendum.md``)."""
        return self.gateway is not None and getattr(self.gateway, "mode", None) == "enforce"

    def set_tool_readiness(self, tools: Mapping[str, Any] | None) -> None:
        """Record the data layer's unready tools ``{tool: reason}`` (reason: text or
        ``{table, column, check, ...}``); ListTools and the prompts list them by table."""
        self.tool_readiness = {str(k): v for k, v in (tools or {}).items()}
        self._system_cache.clear()

    def _prompt_unavailable(self) -> Any:
        """Unavailable servers for the prompt; with unready tools, ``{servers, tools}``."""
        servers = self.unavailable_servers()
        if not self.tool_readiness:
            return servers or None
        return {"servers": servers, "tools": dict(self.tool_readiness)}

    def _mcp_event(self, kind: str, **data: Any) -> None:
        """MCP bridge lifecycle events (start, crash, restart, timeout): traced and emitted."""
        try:
            self._trace(kind, **data)
        except Exception:  # noqa: BLE001 - observers must not break tool calls
            log.debug("tracing %s failed", kind, exc_info=True)
        self.emit(kind, **data)

    def unavailable_servers(self) -> dict[str, str]:
        """MCP servers agents cannot rely on: failed starts plus servers without their data."""
        out = dict(self.degraded_servers or {})
        if self.mcp:
            out.update(self.mcp.failures or {})
        return out

    def set_degraded(self, servers: Mapping[str, str]) -> None:
        """Record servers that lack reference data (prompts are rebuilt with the notice)."""
        self.degraded_servers = {str(k): str(v) for k, v in (servers or {}).items()}
        self._system_cache.clear()

    async def aclose(self) -> None:
        await self.cancel_outstanding()
        try:
            await self.bus.drain(timeout=10)
        except Exception:  # noqa: BLE001
            log.debug("event drain failed", exc_info=True)
        close = getattr(self.gateway, "aclose", None)
        if callable(close):
            try:
                await close()                          # the gateway's background check, before its bridge
            except Exception:  # noqa: BLE001
                log.debug("data gateway close failed", exc_info=True)
        if self.mcp:
            await self.mcp.aclose()
        await self.provider.aclose()

    # ------------------------------------------------------------------ events and trace

    def emit(self, kind: str, **data: Any) -> None:
        """Publish an event to every subscriber (never raises)."""
        data.setdefault("ts", round(time.time(), 3))
        try:
            self.bus.publish(kind, data)
        except Exception:  # noqa: BLE001 - UI callbacks must not break runs
            log.exception("event publish failed")

    def _trace(self, type_: str, **data: Any) -> None:
        """Trace an event tagged with the running invocation's agent_run_id/parent_run_id."""
        ids = _RUN_IDS.get()
        if ids is not None:
            data.setdefault("agent_run_id", ids[0])
            data.setdefault("parent_run_id", ids[1])
        self.run.trace(type_, **data)

    def _note_audit_error(self, msg: str) -> None:
        fn = getattr(self.run, "note_audit_error", None)
        if callable(fn):
            try:
                fn(msg)
                return
            except Exception:  # noqa: BLE001
                pass
        log.warning("[audit] %s", msg)

    # ------------------------------------------------------------------ roster helpers

    def tools_for(self, agent: AgentDefinition) -> list[Tool]:
        tools = self.registry.select(agent.tools)
        if not agent.can_delegate:
            tools = [t for t in tools if t.name != "Task"]
        if not any(t.name == QUERY_TOOL for t in tools):
            q = self.registry.get(QUERY_TOOL)
            if q is not None:
                tools.append(q)
        return tools

    def _definition(self, name: str) -> AgentDefinition | None:
        if name == getattr(self.cso, "name", "cso"):
            return self.cso
        return self.agents.get(name) or self._known_agents.get(name)

    def workspace_for(self, name: str) -> Path:
        """Base directory for ``name``'s relative paths and Bash cwd.

        The run directory for the CSO and for agents with ``workspace: run``
        (the scientific reviewer by default); ``work/<name>/`` otherwise.
        """
        if name == "cso":
            return self.run.dir
        agent = self._definition(name)
        ws = agent.workspace if agent is not None else None
        if ws == "run" or (ws is None and name == "scientific-reviewer"):
            return self.run.dir
        return self.run.agent_dir(name)

    def max_turns_for(self, agent: AgentDefinition, depth: int) -> int:
        """Model-call cap for one invocation: the agent's own ``max_turns``, else
        ``limits.max_cso_turns`` at depth 0, else ``limits.max_specialist_turns``."""
        mt = agent.max_turns
        if mt:
            return int(mt)
        limits = self.config.get("limits") or {}
        if depth == 0:
            return int(limits.get("max_cso_turns") or limits.get("max_agent_turns") or 100)
        return int(limits.get("max_specialist_turns") or 400)

    def _agent_settings(self, agent: AgentDefinition, depth: int, inv: str,
                        session_key: str | None = None) -> ModelSettings:
        """The agent's tier settings (tier extras such as thinking_budget passed
        through unchanged) plus ``extra['session_key']`` for sticky routing: the
        invocation id, or for the depth-0 agent (the CSO, whose conversation spans
        every turn) one key per run and agent."""
        s = self.context.settings_for(agent.settings(self.config))
        extra = dict(s.extra or {})
        if not extra.get("session_key"):
            if session_key:
                extra["session_key"] = str(session_key)
            elif depth == 0:
                extra["session_key"] = f"{getattr(self.run, 'run_id', '') or 'run'}:{agent.name}"
            else:
                extra["session_key"] = inv
        return dataclasses.replace(s, extra=extra)

    def _system_for(self, agent: AgentDefinition, workspace: Path) -> list[SystemSegment]:
        """[stable (cached), volatile] system prompt. The delegating agent's (CSO's)
        prompt is built once per Runtime; others once per invocation."""
        if agent.can_delegate and agent.name in self._system_cache:
            return self._system_cache[agent.name]
        stable, volatile = system_prompt_parts(agent, run_dir=self.run.dir, workspace=workspace, config=self.config,
                                               roster=self.agents if agent.can_delegate else None,
                                               unavailable_servers=self._prompt_unavailable(),
                                               data_layer=self.data_layer_enforcing)
        system = [SystemSegment(stable, cache=True), SystemSegment(volatile)]
        if agent.can_delegate:
            self._system_cache[agent.name] = system
        return system

    # ------------------------------------------------------------------ budget and cost

    def cost_scope(self, name: str, limit: float | None = None, *, limit_tokens: float | None = None):
        """Context manager opening a budget scope nested in the current one::

            with rt.cost_scope("turn", limits.max_turn_cost_usd, limit_tokens=limits.max_turn_tokens): ...
        """
        return open_scope(name, limit, limit_tokens=limit_tokens)

    def begin_turn(self) -> CostScope:
        """Compat shim: open (or reset) a ``turn`` scope in the current context."""
        cur = budget.current_scope()
        if cur is not None and cur.meta.get("compat_turn"):
            cur.reset()
            cur.limit_usd = self.turn_budget_usd
            cur.limit_tokens = self.turn_budget_tokens
            return cur
        scope = CostScope("turn", self.turn_budget_usd, parent=cur, meta={"compat_turn": True},
                          limit_tokens=self.turn_budget_tokens)
        budget.set_scope(scope)
        return scope

    def _check_budget(self) -> None:
        """Raise BudgetExceeded for the innermost exceeded scope (scopes only)."""
        budget.check()

    def _charge(self, agent: str, usage: Any, usd: float, label: str | None = None) -> None:
        """Charge a cost now: run ledger, every enclosing scope, the invocation accumulator.

        ``usage`` given: a model call (``run.cost.add``); otherwise a tool-side
        cost (``run.cost.add_extra(agent=, usd=, label=)``). A model call also
        charges its budget tokens (``budget.usage_tokens``) to every scope, so
        token limits work when the provider costs 0 USD.
        """
        try:
            usd = float(usd or 0.0)
        except (TypeError, ValueError):
            usd = 0.0
        cost = self.run.cost
        tokens = 0.0
        if usage is not None:
            cost.add(agent, usage, usd)
            tokens = budget.usage_tokens(usage, self.cached_token_weight)
        elif usd:
            cost.add_extra(agent=agent, usd=usd, label=label)
        budget.charge(usd, tokens)
        acc = budget.current_invocation()
        if acc is not None:
            acc.add(usd, model_call=usage is not None, tokens=tokens)
        if usd or usage is not None:
            self.emit("cost", total_usd=round(cost.total_usd, 6), total_tokens=self.total_tokens())

    def total_tokens(self) -> int:
        """Input (incl. cached) + output tokens of every model call of the run so far."""
        try:
            return int(sum(u.input_tokens + u.cache_read_tokens + u.cache_write_tokens + u.output_tokens
                           for u in self.run.cost.by_agent.values()))
        except Exception:  # noqa: BLE001 - a display number
            return 0

    # ------------------------------------------------------------------ history

    def repair_history(self, messages: list[Message], reason: str = "interrupted") -> int:
        """``repair_history`` plus a ``history_repaired`` trace event when it fixed anything."""
        before = validate_tool_pairing(messages)
        n = repair_history(messages, reason)
        if n:
            self._trace("history_repaired", reason=reason, fixes=n, problems=before[:5])
        return n

    # ------------------------------------------------------------------ cancellation

    def _track(self, fut: asyncio.Future) -> None:
        self._outstanding.add(fut)
        fut.add_done_callback(self._outstanding.discard)

    async def cancel_outstanding(self) -> None:
        """Cancel and await every outstanding tool/agent task (except the caller's)."""
        me = asyncio.current_task()
        tasks = [t for t in list(self._outstanding) if not t.done() and t is not me]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ agent loop

    async def run_agent(
        self,
        agent: AgentDefinition,
        task: str | None,
        *,
        history: list[Message] | None = None,
        depth: int = 0,
        stream_text: bool = False,
        extra_tools: list[Tool] | None = None,
        allow_tools: bool = True,
        after_end_turn: Callable[[list[Message]], Awaitable[str | None]] | None = None,
        invocation_id: str | None = None,
        parent_invocation_id: str | None = None,
        description: str = "",
        tool_use_id: str | None = None,
        force_tool: str | None = None,
        max_turns: int | None = None,
        session_key: str | None = None,
    ) -> AgentResult:
        """Run ``agent`` until it answers without tool calls (or a stop condition).

        ``history`` is extended in place and stays valid on every exit path.
        ``after_end_turn`` may return a harness message that re-opens the loop
        (used to enforce review); text before it moves to ``drafts``.

        ``force_tool``: on the final allowed call (after ``max_turns`` model calls)
        the agent is told to call this tool, the call is forced with
        ``extra['tool_choice']`` when the provider supports it, and the tool runs
        (a terminal tool then completes the invocation). ``max_turns`` overrides
        the agent's cap (0: the first call is the final one, e.g. a bulk agent
        continued once to submit its result). ``session_key`` pins the requests to
        one server replica (default: the invocation id; the depth-0 agent's key
        is stable for the session).
        """
        t0 = time.time()
        if agent.name not in self.agents and agent is not self.cso:
            self._known_agents[agent.name] = agent  # ad-hoc agents (bulk, judges): workspace_for needs them
        outer = _RUN_IDS.get()
        inv = invocation_id or f"{agent.name}:{uuid.uuid4().hex[:8]}"
        parent = parent_invocation_id if parent_invocation_id is not None else (outer[0] if outer else None)
        messages = history if history is not None else []
        self.repair_history(messages, "the previous turn was interrupted before this call ran")
        if task is not None:
            messages.append(Message.user(task))
        workspace = self.workspace_for(agent.name)
        system = self._system_for(agent, workspace)
        by_name: dict[str, Tool] = {}
        if allow_tools:
            for t in self.tools_for(agent) + list(extra_tools or []):
                by_name[t.name] = t
        inert: list[Tool] = []
        if not allow_tools and _has_tool_blocks(messages):
            inert = self.tools_for(agent) + list(extra_tools or [])
        settings = self._agent_settings(agent, depth, inv, session_key)
        result = AgentResult(agent.name, "", messages, invocation_id=inv, parent_invocation_id=parent)
        if inv in self._live:
            self._live[inv] = result
        acc = InvocationCost(inv)
        cap = self.max_turns_for(agent, depth) if max_turns is None else max(0, int(max_turns))
        st = _Loop(agent=agent, depth=depth, messages=messages, tools=list(by_name.values()), by_name=by_name,
                   settings=settings, system=system, result=result, inv=inv, parent=parent, stream_text=stream_text,
                   after_end_turn=after_end_turn, max_turns=cap, acc=acc, inert_tools=inert,
                   force_tool=force_tool if force_tool and force_tool in by_name else None,
                   task_given=task is not None)
        ids_token = _RUN_IDS.set((inv, parent))
        acc_token = budget.use_invocation(acc)
        self._trace("agent_start", agent=agent.name, depth=depth, agent_run_id=inv, parent_run_id=parent,
                    tool_use_id=tool_use_id, description=description, model=settings.model,
                    task=(task or "")[:4000], task_chars=len(task or ""), tools=sorted(by_name),
                    max_turns=st.max_turns)
        self.emit("agent_start", invocation_id=inv, parent_invocation_id=parent, tool_use_id=tool_use_id,
                  agent=agent.name, division=getattr(agent, "division", ""), depth=depth, description=description)
        failure: BaseException | None = None
        try:
            await self._loop(st)
        except asyncio.CancelledError as exc:
            failure = exc
            result.status = self._cancel_reasons.get(inv, "cancelled")
            raise
        except BudgetExceeded as exc:
            failure = exc
            result.status, result.stop_reason, result.error = "budget", "budget", str(exc)
            exc.agent_result = result  # the caller can still count this invocation's model calls (bulk)
            raise
        except KeyboardInterrupt as exc:
            failure = exc
            result.status = "cancelled"
            raise
        except BaseException as exc:
            failure = exc
            result.status, result.error = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                self._finish(st, t0, failure)
            finally:
                budget.reset_invocation(acc_token)
                try:
                    _RUN_IDS.reset(ids_token)
                except ValueError:
                    _RUN_IDS.set(outer)
        return result

    async def _loop(self, st: _Loop) -> None:
        r = st.result
        while True:
            # ---- budget: the depth-0 agent gets one final no-tool call per scope
            grace = False
            exc = st.pending_budget
            st.pending_budget = None
            if exc is None:
                try:
                    self._check_budget()
                except BudgetExceeded as e:
                    exc = e
            budget_force = False
            if exc is not None:
                if st.depth == 0 and exc.scope is not None and not exc.scope.grace_used:
                    exc.scope.grace_used = True
                    grace = True
                    self._trace("budget_grace", agent=st.agent.name, scope=exc.scope.as_dict(), error=str(exc))
                    self.emit("warning", message=f"{st.agent.name}: {exc}; asking for a final synthesis")
                    st.messages.append(Message.user(BUDGET_GRACE_MSG))
                elif self._budget_force_allowed(st, exc):
                    # A force_tool agent (bulk: submit_result) whose own scope ran out gets one forced,
                    # budget-exempt call, so the evidence it gathered is submitted instead of lost.
                    st.budget_forced = exc
                    budget_force = True
                    self._trace("budget_force_tool", agent=st.agent.name, tool=st.force_tool,
                                scope=exc.scope.as_dict() if exc.scope is not None else None, error=str(exc))
                    st.messages.append(Message.user(BUDGET_FORCE_MSG.format(tool=st.force_tool)))
                else:
                    raise exc
            final_call = grace or budget_force or st.turns >= st.max_turns
            forced = bool(final_call and not grace and st.force_tool)
            if final_call and not grace and not budget_force:
                if not forced:
                    st.messages.append(Message.user(TURN_LIMIT_MSG))
                elif not (st.turns == 0 and st.task_given):  # else the caller's task is this call's instruction
                    st.messages.append(Message.user(FORCE_TOOL_MSG.format(tool=st.force_tool)))
            if forced:
                st.call_mode = "force"
            elif final_call or (st.inert_tools and not st.by_name):
                st.call_mode = "none"   # no tool may run on this call
            else:
                st.call_mode = None

            # ---- keep the context inside the window, then call the model
            await self._maybe_compact(st)
            resp = await self._call_model(st)
            st.turns += 1
            if resp is None:  # overflow that compaction could not fix
                r.status = r.stop_reason = "context_exceeded"
                r.text = st.report or (st.segments[-1] if st.segments else "")
                if not r.text:
                    r.text = "[The conversation no longer fits the model's context window; no report was written.]"
                return
            st.messages.append(resp.message)
            self._absorb_text(st, resp.message.text)
            self.emit("message_end", invocation_id=st.inv, agent=st.agent.name)
            calls = resp.message.tool_calls
            stop = resp.stop_reason
            r.stop_reason = stop.value

            if stop is StopReason.CONTEXT_EXCEEDED:
                if calls:
                    self._answer(st, calls, "the context window was exceeded")
                r.status = "context_exceeded"
                r.text = st.report
                return
            if stop is StopReason.REFUSAL:
                if calls:
                    self._answer(st, calls, "the model declined to continue")
                note = f"[The model declined to continue: {resp.stop_detail}]"
                st.segments.append(note)
                r.status = "refusal"
                r.text = (st.report + "\n\n" + note).strip()
                return
            if final_call:
                why = "budget exhausted" if (grace or budget_force) else "turn limit reached"
                if forced and calls:
                    terminal = await self._tool_round(st, calls, only=st.force_tool,
                                                      reason=f"{why}; only {st.force_tool} runs")
                    if terminal:
                        r.status, r.stop_reason, r.text = "completed", "terminal_tool", st.report
                        return
                elif calls:
                    self._answer(st, calls, why)
                if budget_force and st.budget_forced is not None:
                    raise st.budget_forced  # the forced call did not complete: the scope's limit stands
                r.status = r.stop_reason = "budget" if grace else "turn_limit"
                r.text = st.report or ("[The budget ran out before a final report was written.]" if grace else
                                       "[Turn limit reached before a final report was written.]")
                return
            if calls and not st.by_name:
                # Tools are disabled for this call (they are only declared so the
                # tool blocks already in the history stay valid): answer without
                # executing, give the model one chance to reply in text.
                self._answer(st, calls, "tools are not available for this call; answer directly in text")
                st.refused_rounds += 1
                if st.refused_rounds <= 1:
                    st.report = ""
                    continue
                r.status, r.text = "completed", st.report
                return
            if calls and stop in (StopReason.TOOL_USE, StopReason.END_TURN, StopReason.PAUSE):
                report = st.report
                try:
                    terminal = await self._tool_round(st, calls)
                except BudgetExceeded as e:
                    if st.depth == 0 and e.scope is not None and not e.scope.grace_used:
                        st.pending_budget = e  # history is answered; the next iteration makes the grace call
                        st.report = ""
                        continue
                    raise
                st.report = ""
                if terminal:
                    r.status, r.stop_reason, r.text = "completed", "terminal_tool", report
                    return
                continue
            if stop is StopReason.PAUSE:
                continue  # provider-side tool loop paused: resend to continue
            if stop is StopReason.MAX_TOKENS:
                if calls:  # truncated tool input: report it and let the model retry
                    st.messages.append(Message("user", [ToolResult(c.id, TRUNCATED_INPUT_MSG, True) for c in calls]))
                    st.report = ""
                else:
                    st.messages.append(Message.user(CONTINUE_MSG))
                    st.continuing = bool(resp.message.text)
                continue
            if calls:  # OTHER (or another abnormal stop) with pending calls
                self._answer(st, calls, f"the provider stopped the turn ({resp.stop_detail or stop.value})")
                r.status, r.error = "error", f"provider stop {stop.value}: {resp.stop_detail or ''}".strip()
                r.text = st.report
                return
            if (stop is StopReason.END_TURN and not st.empty_nudged and not resp.message.text.strip()
                    and self._follows_tool_results(st.messages)):
                # An empty reply right after tool results (seen with local models): nudge once;
                # a second empty reply is accepted as the final answer.
                st.empty_nudged = True
                r.nudges.append("empty_reply")
                self._trace("empty_reply_nudge", agent=st.agent.name, turns=st.turns)
                st.messages.append(Message.user(EMPTY_REPLY_MSG))
                continue
            r.text = st.report
            if st.after_end_turn is not None:
                nudge = await st.after_end_turn(st.messages)
                if nudge:
                    st.messages.append(Message.user(nudge))
                    r.drafts.append("\n\n".join(s for s in st.segments if s))
                    st.segments, st.report, st.continuing = [], "", False
                    continue
            r.status = "completed"
            return

    def _absorb_text(self, st: _Loop, text: str) -> None:
        if st.continuing:
            if text:
                if st.segments:
                    st.segments[-1] += text
                else:
                    st.segments.append(text)
                st.report += text
            st.continuing = False
        else:
            if text:
                st.segments.append(text)
            st.report = text

    @staticmethod
    def _follows_tool_results(messages: list[Message]) -> bool:
        """The last message (a reply) directly follows a user message with tool results."""
        return len(messages) >= 2 and messages[-2].role == "user" and bool(messages[-2].tool_results)

    @staticmethod
    def _answer(st: _Loop, calls: list[ToolCall], reason: str) -> None:
        st.messages.append(Message("user", [ToolResult(c.id, f"Not executed: {reason}.", True) for c in calls]))

    def _finish(self, st: _Loop, t0: float, failure: BaseException | None) -> None:
        r = st.result
        if failure is not None:
            reason = {"cancelled": "the call was cancelled (turn interrupted)",
                      "timeout": "the delegation timed out",
                      "budget": "budget exhausted"}.get(r.status, f"interrupted by {type(failure).__name__}")
            try:
                self.repair_history(st.messages, reason)
            except Exception:  # noqa: BLE001
                log.exception("history repair failed")
        if not r.full_text:
            r.full_text = "\n\n".join(s for s in st.segments if s)
        if failure is not None and not r.text:
            r.text = st.report or (st.segments[-1] if st.segments else "")
        r.cost_usd = st.acc.usd
        r.tokens = st.acc.tokens
        r.duration_s = round(time.time() - t0, 1)
        summary = failures.summarize(st.calls)
        r.unresolved_data_failures = summary["unresolved_data"]
        r.recovered_errors = summary["recovered_count"]
        r.other_errors = summary["other_error_count"]
        r.transcript_path = self._write_transcript(st)
        self._trace("agent_end", agent=r.agent, depth=st.depth, agent_run_id=st.inv, parent_run_id=st.parent,
                    status=r.status, stop=r.stop_reason, cost_usd=round(r.cost_usd, 6),
                    budget_tokens=round(r.tokens, 1), nudges=list(r.nudges), model_calls=r.model_calls,
                    tool_calls=r.tool_calls, duration_s=r.duration_s, text=r.text[:20000],
                    full_text_chars=len(r.full_text), drafts=len(r.drafts), transcript_path=r.transcript_path,
                    compactions=r.compactions, retries=r.retries, fallback_count=r.fallback_count, error=r.error,
                    unresolved_data_failures=[{"tool": f.get("tool"), "error": str(f.get("error") or "")[:300]}
                                              for f in r.unresolved_data_failures],
                    recovered_errors=r.recovered_errors, other_errors=r.other_errors)
        self.emit("agent_end", invocation_id=st.inv, agent=r.agent, depth=st.depth, status=r.status,
                  stop_reason=r.stop_reason, cost_usd=r.cost_usd, duration_s=r.duration_s,
                  model_calls=r.model_calls, tool_calls=r.tool_calls)

    def _write_transcript(self, st: _Loop) -> str | None:
        path = self.run.dir / "logs" / "agents" / f"{_safe(st.inv)}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for m in st.messages:
                    f.write(json.dumps(dataclasses.asdict(m), default=str, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 - records must not break the run
            self._note_audit_error(f"agent transcript {path.name}: {type(exc).__name__}: {exc}")
            return None
        return self.run.rel(path)

    @staticmethod
    def _budget_force_allowed(st: _Loop, exc: BudgetExceeded) -> bool:
        """One forced call is granted when the agent has a ``force_tool``, has not had
        it yet, and the exceeded scope is its own innermost one (a bulk item's
        scope, not the bulk job's or an enclosing turn's)."""
        if not st.force_tool or st.budget_forced is not None or exc.scope is None:
            return False
        return exc.scope is budget.current_scope()

    # ------------------------------------------------------------------ model calls

    def _call_settings(self, st: _Loop) -> ModelSettings:
        """``st.settings`` plus the ``tool_choice`` of this call when the provider
        honours it: ``{'name': force_tool}`` on a forced final call, ``'none'`` on a
        call where no tool may run (turn limit, budget grace, inert tools)."""
        mode = st.call_mode
        if mode is None or not st.specs or "tool_choice" in (st.settings.extra or {}):
            return st.settings
        if not self.supports_tool_choice(st.settings.model):
            return st.settings
        choice: Any = {"name": st.force_tool} if mode == "force" and st.force_tool else "none"
        return dataclasses.replace(st.settings, extra={**(st.settings.extra or {}), "tool_choice": choice})

    def _ensure_valid(self, st: _Loop) -> None:
        problems = validate_tool_pairing(st.messages)
        if problems:
            n = repair_history(st.messages, "history repaired by the harness")
            self._trace("history_repaired", agent=st.agent.name, problems=problems[:5], fixes=n)

    async def _call_model(self, st: _Loop) -> ModelResponse | None:
        """One provider call with transient-error retry and one overflow recovery.

        Returns None when the request overflowed and could not be compacted
        enough (status context_exceeded)."""
        overflowed = False
        while True:
            self._ensure_valid(st)
            streamed = {"thinking": False}
            kwargs: dict[str, Any] = {}
            if st.stream_text:
                def on_text(t: str) -> None:
                    self.emit("text", invocation_id=st.inv, agent=st.agent.name, depth=st.depth, text=t)

                def on_thinking(t: str) -> None:
                    streamed["thinking"] = True
                    self.emit("thinking", invocation_id=st.inv, agent=st.agent.name, depth=st.depth, text=t,
                              streamed=True)
                kwargs = {"on_text": on_text, "on_thinking": on_thinking}

            def on_retry(attempt: int, exc: Exception, delay: float) -> None:
                st.result.retries += 1
                err = f"{type(exc).__name__}: {exc}"[:500]
                self._trace("provider_retry", agent=st.agent.name, attempt=attempt, delay_s=round(delay, 2),
                            error=err, status=getattr(exc, "status", None))
                self.emit("retry", invocation_id=st.inv, agent=st.agent.name, attempt=attempt,
                          delay_s=round(delay, 2), error=err)

            try:
                resp = await complete_with_retry(self.provider, policy=self.retry_policy, on_retry=on_retry,
                                                 settings=self._call_settings(st), system=st.system,
                                                 messages=st.messages, tools=st.specs, **kwargs)
            except ContextOverflowError as exc:
                self._trace("context_overflow", agent=st.agent.name, error=str(exc)[:500], recovered=not overflowed)
                if overflowed or not await self._recover_overflow(st):
                    return None
                overflowed = True
                continue
            self._account(st, resp, streamed["thinking"])
            if resp.stop_reason is StopReason.CONTEXT_EXCEEDED and not overflowed:
                self._trace("context_overflow", agent=st.agent.name, stop=resp.stop_reason.value, recovered=True)
                if await self._recover_overflow(st):
                    overflowed = True
                    self.emit("message_end", invocation_id=st.inv, agent=st.agent.name)
                    continue  # the truncated response is discarded and the request re-sent
            return resp

    def _account(self, st: _Loop, resp: ModelResponse, thinking_streamed: bool) -> None:
        r = st.result
        r.model_calls += 1
        if resp.fallback_used:
            r.fallback_count += 1
        self._charge(st.agent.name, resp.usage, resp.cost_usd)
        st.last_usage = resp.usage
        thinking = [b.text for b in resp.message.content if isinstance(b, ThinkingBlock) and b.text]
        self._trace("model_call", agent=st.agent.name, model=resp.model, served_model=resp.served_model,
                    fallback_used=resp.fallback_used, retries=resp.retries, stop=resp.stop_reason.value,
                    stop_detail=resp.stop_detail, request_id=resp.request_id, usage=resp.usage.as_dict(),
                    cost_usd=round(resp.cost_usd, 6), text=resp.message.text[:20000],
                    thinking=[t[:20000] for t in thinking], thinking_chars=sum(len(t) for t in thinking),
                    tool_calls=[c.name for c in resp.message.tool_calls])
        if not thinking_streamed:
            for t in thinking:
                self.emit("thinking", invocation_id=st.inv, agent=st.agent.name, depth=st.depth, text=t,
                          streamed=False)

    async def _maybe_compact(self, st: _Loop) -> None:
        try:
            res = await self.context.maybe_compact(st.messages, last_usage=st.last_usage, settings=st.settings,
                                                   system=st.system, agent=st.agent.name)
        except BaseException as exc:
            self._charge_failed_compaction(st, exc)
            if not isinstance(exc, Exception):
                raise
            log.warning("context compaction failed for %s: %s", st.agent.name, exc)
            self._trace("compaction_failed", agent=st.agent.name, error=f"{type(exc).__name__}: {exc}"[:500])
            return
        if res is not None:
            self._record_compaction(st, res, overflow=False)

    async def _recover_overflow(self, st: _Loop) -> bool:
        try:
            res = await self.context.recover_overflow(st.messages, settings=st.settings, system=st.system,
                                                      agent=st.agent.name)
        except BaseException as exc:  # ContextOverflowError: nothing left to compact
            self._charge_failed_compaction(st, exc)
            if not isinstance(exc, Exception):
                raise
            self._trace("context_overflow_unrecoverable", agent=st.agent.name,
                        error=f"{type(exc).__name__}: {exc}"[:500])
            return False
        self._record_compaction(st, res, overflow=True)
        return True

    def _charge_failed_compaction(self, st: _Loop, exc: BaseException) -> None:
        """Charge what a compaction spent (its summariser call) before failing:
        the spend of failed attempts is never lost."""
        spent = compaction_spend(exc)
        if spent is None:
            return
        usage, usd = spent
        self._charge(st.agent.name, usage, usd, label="context_compaction")
        self._trace("compaction_failed_spend", agent=st.agent.name, usage=usage.as_dict(),
                    cost_usd=round(usd, 6))

    def _record_compaction(self, st: _Loop, res: CompactionResult, *, overflow: bool) -> None:
        st.result.compactions += 1
        u = res.usage
        if res.cost_usd or u.input_tokens or u.output_tokens:
            self._charge(st.agent.name, u, res.cost_usd, label="context_compaction")
        self._trace("compaction", agent=st.agent.name, overflow=overflow, **res.as_dict())
        self.emit("compaction", invocation_id=st.inv, agent=st.agent.name, strategy=res.strategy,
                  tokens_before=res.tokens_before, tokens_after=res.tokens_after_estimate)
        st.last_usage = None  # sizes changed: estimate until the next response reports usage

    # ------------------------------------------------------------------ tools

    async def _tool_round(self, st: _Loop, calls: list[ToolCall], *, only: str | None = None,
                          reason: str = "") -> bool:
        """Run ``calls`` in parallel; returns True when a terminal tool succeeded.

        ``only``: run just the calls of that tool; the others are answered
        ``Not executed: <reason>`` (a forced final call).

        On any exception (including cancellation and BudgetExceeded) the pending
        siblings are cancelled and awaited, every call is answered, and the
        exception propagates."""
        async def skip(c: ToolCall) -> ToolResult:
            return ToolResult(c.id, f"Not executed: {reason or 'not allowed on this call'}.", True)

        tasks = [asyncio.ensure_future(self._execute(c, st) if only is None or c.name == only else skip(c))
                 for c in calls]
        for t in tasks:
            self._track(t)
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
                for t in done:
                    if not t.cancelled() and t.exception() is not None:
                        raise t.exception()  # type: ignore[misc]
        except BaseException as exc:
            reason = self._interrupt_reason(st, exc)
            for t in tasks:
                if not t.done():
                    t.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self._append_round(st, calls, tasks, reason)
            raise
        results = self._append_round(st, calls, tasks, None)
        st.result.delegations += [c.input.get("subagent_type", "?") for c in calls
                                  if c.name == "Task" and (only is None or only == "Task")]
        return any(not res.is_error and st.by_name.get(c.name) is not None and st.by_name[c.name].terminal
                   for c, res in zip(calls, results))

    def _interrupt_reason(self, st: _Loop, exc: BaseException) -> str:
        if isinstance(exc, asyncio.CancelledError):
            return "the delegation timed out" if self._cancel_reasons.get(st.inv) == "timeout" else \
                "the turn was cancelled"
        if isinstance(exc, BudgetExceeded):
            return "budget exhausted"
        if isinstance(exc, KeyboardInterrupt):
            return "the turn was interrupted"
        return f"a sibling tool call failed ({type(exc).__name__})"

    @staticmethod
    def _append_round(st: _Loop, calls: list[ToolCall], tasks: list[asyncio.Future],
                      reason: str | None) -> list[ToolResult]:
        blocks: list[ToolResult] = []
        for c, t in zip(calls, tasks):
            res = None
            if t.done() and not t.cancelled() and t.exception() is None:
                res = t.result()
            if isinstance(res, ToolResult):
                blocks.append(res)
            elif reason is None:
                blocks.append(ToolResult(c.id, "Error: the tool call was cancelled.", True))
            else:
                blocks.append(ToolResult(c.id, f"Not executed: {reason}.", True))
        st.messages.append(Message("user", list(blocks)))
        return blocks

    def _resolve_agent_path(self, agent: str, raw: str) -> Path:
        p = Path(os.path.expanduser(str(raw)))
        if p.is_absolute():
            return p.resolve()
        ws = self.workspace_for(agent)
        cands = ([self.run.dir / p] if p.parts and p.parts[0] in RUN_PREFIXES else []) + [ws / p, self.run.dir / p]
        for c in cands:
            if c.exists():
                return c.resolve()
        return cands[0].resolve()

    def _tool_start_event(self, call: ToolCall, st: _Loop) -> dict[str, Any]:
        ev: dict[str, Any] = {"agent": st.agent.name, "tool": call.name, "tool_use_id": call.id}
        try:
            text = json.dumps(call.input, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(call.input)
        if len(text) <= self.trace_input_inline_chars:
            ev["input"] = call.input
        else:
            sha = failures.input_sha256(call.input)
            path = self.run.dir / "logs" / "tool_inputs" / f"{_safe(call.id)}.json"
            ev.update(input_sha256=sha, input_chars=len(text), input_preview=preview(call.input, 2000))
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"tool_use_id": call.id, "tool": call.name, "agent": st.agent.name,
                                            "agent_run_id": st.inv, "input_sha256": sha, "input": call.input},
                                           indent=1, default=str, ensure_ascii=False), encoding="utf-8")
                ev["input_ref"] = self.run.rel(path)
            except Exception as exc:  # noqa: BLE001 - never silently clip: keep the input inline instead
                self._note_audit_error(f"tool input spill {call.id}: {type(exc).__name__}: {exc}")
                ev["input"] = call.input
        if call.name == "Bash":
            cmd = call.input.get("command") if isinstance(call.input, dict) else None
            if isinstance(cmd, str) and _INLINE_CODE_RE.search(cmd):
                path = self.run.dir / "logs" / "bash" / f"{_safe(call.id)}.sh"
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"#!/bin/bash\n# vbt: agent={st.agent.name} tool_use_id={call.id}\n{cmd}\n",
                                    encoding="utf-8")
                    ev["bash_script"] = self.run.rel(path)
                    ev["bash_sha256"] = _sha256_bytes(cmd.encode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    self._note_audit_error(f"bash script save {call.id}: {type(exc).__name__}: {exc}")
        return ev

    def _spill_output(self, call: ToolCall, text: str, obj: Any) -> Path | None:
        out_dir = self.run.dir / "logs" / "tool_outputs"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            if obj is not None:
                path = out_dir / f"{_safe(call.id)}.json"
                path.write_text(json.dumps(obj, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
            else:
                path = out_dir / f"{_safe(call.id)}.txt"
                path.write_text(text, encoding="utf-8")
            return path
        except Exception as exc:  # noqa: BLE001
            self._note_audit_error(f"tool output spill {call.id}: {type(exc).__name__}: {exc}")
            return None

    def _data_provenance_dir(self) -> Path:
        """``<run>/<data.provenance.dir>``; the setting is always relative to the run (an absolute or
        ``..`` path falls back to the default), so replay and ``derived_from`` find what is written."""
        rel = "logs/data_provenance"
        try:
            rel = self.data_settings().provenance.dir or rel
        except Exception:  # noqa: BLE001 - fall back to the documented default
            log.debug("data settings unreadable; provenance under %s", rel, exc_info=True)
        return self.run.dir / rel

    def _record_data_result(self, call: ToolCall, data: Any) -> dict[str, Any]:
        """``tool_end`` fields of a data-layer result (§15.1): ``result_status`` and the
        ``data_provenance`` summary (status, coverage and its statement, source@release,
        table fingerprints, counts, row-key hash, evidence nature and caveat, leakage risk,
        whether the order was verified). The full ``vbt.dataprov/1`` record, stamped with
        this ``tool_use_id``, is written atomically to ``logs/data_provenance/<tool_use_id>.json``."""
        header = getattr(data, "header", None)
        header = header if isinstance(header, Mapping) else {}
        status = str(getattr(data, "status", None) or header.get("status") or "unknown")
        prov = getattr(data, "provenance", None)
        summary: dict[str, Any] = {}
        record: dict[str, Any] | None = None
        if prov is not None:
            try:
                if hasattr(prov, "tool_use_id"):
                    prov.tool_use_id = call.id
                if getattr(prov, "id", "") is None and callable(getattr(prov, "finalize", None)):
                    prov.finalize()
                record = dict(prov.to_dict()) if callable(getattr(prov, "to_dict", None)) else dict(prov)
                record["tool_use_id"] = call.id
                summary = dict(prov.summary()) if callable(getattr(prov, "summary", None)) else {}
                path = self._data_provenance_dir() / f"{_safe(call.id)}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, record)
                summary["record"] = self.run.rel(path)
            except Exception as exc:  # noqa: BLE001 - a provenance failure is an audit error, not a tool error
                self._note_audit_error(f"data provenance {call.id}: {type(exc).__name__}: {exc}")
        result = (record or {}).get("result") if isinstance((record or {}).get("result"), Mapping) else {}
        for key, value in (("prov", (record or {}).get("id")), ("status", result.get("status")),
                           ("coverage", result.get("coverage")),
                           ("coverage_statement", result.get("coverage_statement")),
                           ("returned", result.get("returned")), ("total", result.get("total")),
                           ("truncated", result.get("truncated"))):
            if summary.get(key) is None and value is not None:
                summary[key] = value
        for key in ("prov", "status", "coverage", "coverage_statement", "source", "returned", "total", "truncated"):
            if summary.get(key) is None and header.get(key) is not None:
                summary[key] = header[key]
        nature = (record or {}).get("evidence_nature")
        caveat = header.get("evidence") or (nature.get("caveat") if isinstance(nature, Mapping) else None)
        if caveat:
            summary["evidence_caveat"] = caveat
        order = result.get("order") if isinstance(result.get("order"), Mapping) else {}
        verified = order.get("verified")
        if verified is None and isinstance(header.get("order"), str):
            verified = True if "(verified)" in header["order"] else None
        summary["order_verified"] = verified
        summary.setdefault("status", status)
        return {"result_status": status, "data_provenance": summary}

    def _shape_output(self, call: ToolCall, text: str, raw: Any, st: _Loop,
                      full_text: str | None = None) -> tuple[str, Path | None]:
        """(text for the model, spill path). Outputs longer than the trace limit are
        spilled; longer than ``tool_output_max`` are truncated for the model
        (structure-aware for JSON) with a note naming the recovery tools.

        ``full_text``: the unshrunk payload of a data-layer result whose ``text`` the
        gateway already shortened. It is what gets spilled (before any cap), and the
        model text then ends with the truncation note naming the spill."""
        full = full_text if full_text and full_text != text else None
        if full is None and len(text) <= self.trace_output_chars and len(text) <= self.tool_output_max:
            return text, None
        obj = raw if isinstance(raw, (dict, list)) else maybe_parse_json(text)
        if full is not None:
            path = self._spill_output(call, full, maybe_parse_json(full))
            total = len(full)
        else:
            path = self._spill_output(call, text, obj)
            total = len(text)
        if len(text) <= self.tool_output_max:
            if full is None or path is None:
                return text, path
            return text + "\n\n" + truncation_note(total, str(path), structured=obj is not None,
                                                   has_read=st.has_read), path
        if path is None:  # could not save: keep what fits, say so
            return truncate_text(text, self.tool_output_max) + (
                f"\n\n[Output truncated: {total:,} chars; the full output could not be saved.]"), None
        if obj is not None:
            body = shrink_json(obj, max(2000, self.tool_output_max - 700), spill_path=str(path))
            note = truncation_note(total, str(path), structured=True, has_read=st.has_read)
        else:
            body = truncate_text(text, self.tool_output_max)
            note = truncation_note(total, str(path), structured=False, has_read=st.has_read)
        return body + "\n\n" + note, path

    def _files_returned(self, text: str) -> list[str]:
        """Absolute paths under the run directory named in an MCP output that exist."""
        root = str(self.run.dir)
        out: list[str] = []
        for m in re.finditer(re.escape(root) + r"/[^\s\"'<>|`,;)\]}]+", text):
            p = m.group(0).rstrip(".:")
            if p not in out and os.path.isfile(p):
                out.append(p)
        for m in re.finditer(r"(?<![\w/])(work/_mcp/[^\s\"'<>|`,;)\]}]+)", text):
            p = str(self.run.dir / m.group(1).rstrip(".:"))
            if p not in out and os.path.isfile(p):
                out.append(p)
        return out[:500]

    def _snapshot_code(self, call: ToolCall, st: _Loop) -> dict[str, Any] | None:
        inp = call.input if isinstance(call.input, dict) else {}
        raw = inp.get("file_path") or inp.get("notebook_path") or inp.get("path")
        if not raw:
            return None
        try:
            p = self._resolve_agent_path(st.agent.name, str(raw))
            if p.suffix.lower() not in CODE_EXTS or not p.is_file() or p.stat().st_size > 50_000_000:
                return None
            data = p.read_bytes()
            sha = _sha256_bytes(data)
            dest = self.run.dir / "logs" / "code_versions" / f"{sha}{p.suffix}"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            return {"code_sha256": sha, "code_path": self.run.rel(p), "code_version": self.run.rel(dest)}
        except Exception as exc:  # noqa: BLE001
            self._note_audit_error(f"code version {call.id}: {type(exc).__name__}: {exc}")
            return None

    def _after_tool(self, event: dict[str, Any]) -> None:
        try:
            self.run.after_tool(event)
        except Exception as exc:  # noqa: BLE001 - audit hooks never break a call
            log.warning("after_tool hook failed: %s", exc)

    async def _execute(self, call: ToolCall, st: _Loop) -> ToolResult:
        agent = st.agent
        tool = st.by_name.get(call.name)
        n_audit0 = len(getattr(self.run, "audit_errors", None) or [])
        self._trace("tool_start", **self._tool_start_event(call, st))
        self.emit("tool_start", invocation_id=st.inv, tool_use_id=call.id, agent=agent.name, tool=call.name,
                  input_preview=preview(call.input))
        t0 = time.time()
        raw: Any = None
        parts: list[Any] | None = None  # multimodal result (text + image/document parts)
        model_error: str | None = None  # the harness rejected the call before running it
        data: Any = None                # a data-layer result (DataResult): status, provenance, unshrunk text
        error_kind: str | None = None   # a data-layer error's kind (GatewayError.kind)
        try:
            rejected = self._argument_problem(call, tool) if tool is not None else None
            if tool is None:
                content, err = f"Tool {call.name!r} is not available to {agent.name}.", True
            elif rejected is not None:
                model_error, content = rejected
                err = True
            else:
                ctx = ToolContext(agent=agent.name, run=self.run, runtime=self, tool_call_id=call.id, depth=st.depth,
                                  invocation_id=st.inv)
                try:
                    raw = await tool(ctx, call.input)
                    if getattr(raw, "is_data_result", False):
                        # The model sees the gateway's text (header first); shaping uses its object.
                        data, raw = raw, getattr(raw, "obj", None)
                        content, err = str(data.text), False
                    elif _is_content_parts(raw):
                        content, err = content_text(raw), False
                        parts = self._supported_parts(agent, raw)
                    else:
                        content, err = to_text(raw), False
                except ToolFailure as exc:
                    content, err = f"Error: {exc}", True
                    kind = getattr(exc, "kind", None)  # GatewayError
                    if kind is not None:
                        error_kind = str(getattr(kind, "value", kind))
                except BudgetExceeded:
                    raise
                except Exception as exc:  # noqa: BLE001 - surface any tool crash to the model
                    log.debug("tool %s crashed", call.name, exc_info=True)
                    content, err = f"Error: {type(exc).__name__}: {exc}", True
        except BaseException as exc:
            # Interrupted (cancellation, budget): close the call in the audit trail, then propagate.
            why = self._interrupt_reason(st, exc)
            end = {"agent": agent.name, "tool": call.name, "tool_use_id": call.id, "is_error": True,
                   "interrupted": True, "duration_s": round(time.time() - t0, 2), "output": f"Interrupted: {why}",
                   "output_chars": 0}
            self._trace("tool_end", **end)
            self._after_tool({**end, "input": call.input, "agent_run_id": st.inv, "parent_run_id": st.parent})
            self.emit("tool_end", invocation_id=st.inv, tool_use_id=call.id, agent=agent.name, tool=call.name,
                      is_error=True, duration_s=end["duration_s"], output_preview=f"Interrupted: {why}")
            st.calls.append({"tool": call.name, "input": call.input, "is_error": True, "tool_use_id": call.id,
                             "agent": agent.name, "error": f"Interrupted: {why}"})
            raise
        duration = round(time.time() - t0, 2)
        model_text, spill = self._shape_output(call, content, None if parts is not None else raw, st,
                                               full_text=getattr(data, "full_text", None))
        if parts is not None and model_text != content:  # keep images/documents, truncate the text
            parts = [TextBlock(model_text)] + [x for x in parts if isinstance(x, (ImagePart, DocumentPart))]
        end: dict[str, Any] = {"agent": agent.name, "tool": call.name, "tool_use_id": call.id, "is_error": err,
                               "duration_s": duration, "output": content[: self.trace_output_chars],
                               "output_chars": len(content)}
        if model_error:
            end["model_error"] = model_error
        if data is not None:
            end.update(self._record_data_result(call, data))
        if error_kind:
            end["error_kind"] = error_kind
        if spill is not None:
            end["output_path"] = self.run.rel(spill)
        if call.name.startswith("mcp__") and not err:
            files = self._files_returned(content)
            if files:
                end["files_returned"] = files
        if not err and call.name in CODE_TOOLS:
            code = self._snapshot_code(call, st)
            if code:
                end.update(code)
        self._trace("tool_end", **end)
        self._after_tool({**end, "input": call.input, "agent_run_id": st.inv, "parent_run_id": st.parent})
        self.emit("tool_end", invocation_id=st.inv, tool_use_id=call.id, agent=agent.name, tool=call.name,
                  is_error=err, duration_s=duration, output_preview=content[:500])
        errs = getattr(self.run, "audit_errors", None) or []
        if len(errs) > n_audit0:
            note = AUDIT_NOTE.format(msg=str(errs[-1])[:500])
            if parts is not None:
                if parts and isinstance(parts[0], TextBlock) and _TRUNCATION_NOTE_RE.search(parts[0].text):
                    parts = [TextBlock(_add_note(parts[0].text, note))] + parts[1:]
                else:
                    parts = parts + [TextBlock(note)]
            else:
                model_text = _add_note(model_text, note)  # a truncation note stays last
        rec = {"tool": call.name, "input": call.input, "is_error": err, "tool_use_id": call.id, "agent": agent.name}
        if data is not None:
            rec["result_status"] = end.get("result_status")
        if err:
            rec["error"] = content[:2000]
            entry = {"tool": call.name, "input": preview(call.input, 2000), "error": content[:500],
                     "tool_use_id": call.id}
            if model_error:
                rec["model_error"] = entry["model_error"] = model_error
            if error_kind:
                rec["error_kind"] = entry["error_kind"] = error_kind
            st.result.tool_errors.append(entry)
        st.calls.append(rec)
        st.result.tool_calls += 1
        return ToolResult(call.id, parts if parts is not None else model_text, err)

    @staticmethod
    def _argument_problem(call: ToolCall, tool: Tool) -> tuple[str, str] | None:
        """``(kind, message)`` when the call must not run: its arguments were not
        valid JSON (the provider kept them raw in ``native['invalid_arguments']``)
        or miss required properties / have wrong top-level types for the tool's
        schema. None when the call may run."""
        native = call.native if isinstance(call.native, Mapping) else {}
        if "invalid_arguments" in native:
            error = str(native.get("error") or "unparseable arguments")[:300]
            return "invalid_arguments", INVALID_JSON_MSG.format(tool=call.name, error=error)
        try:
            problems = tool_argument_problems(tool.input_schema, call.input)
        except Exception:  # noqa: BLE001 - a checker bug must never block a tool call
            log.debug("argument check failed for %s", call.name, exc_info=True)
            return None
        if problems:
            listed = "; ".join(problems[:8]) + (f"; ... ({len(problems) - 8} more)" if len(problems) > 8 else "")
            return "schema_mismatch", INVALID_ARGS_MSG.format(tool=call.name, problems=listed)
        return None

    def _supported_parts(self, agent: AgentDefinition, raw: list[Any]) -> list[Any] | None:
        """Keep the image/document parts the agent's model accepts; unsupported
        ones become their text stand-ins. None when nothing but text remains."""
        try:
            caps = self.provider.capabilities(agent.settings(self.config).model)
        except Exception:  # noqa: BLE001 - be conservative: fall back to text
            return None
        out: list[Any] = []
        for x in raw:
            if (isinstance(x, ImagePart) and caps.images) or (isinstance(x, DocumentPart) and caps.documents):
                out.append(x)
            elif isinstance(x, TextBlock):
                out.append(x)
            else:
                out.append(TextBlock(content_text([x])))
        return out if any(isinstance(x, (ImagePart, DocumentPart)) for x in out) else None

    # ------------------------------------------------------------------ delegation

    async def delegate(self, agent_name: str, prompt: str, *, description: str = "", parent_agent: str = "cso",
                       parent_invocation_id: str | None = None, tool_use_id: str | None = None,
                       depth: int = 1) -> AgentResult:
        """Run specialist ``agent_name`` on ``prompt`` in a fresh context.

        Returns its AgentResult (status completed, turn_limit, refusal,
        context_exceeded, budget or timeout, with partial text). Cancellation
        and unexpected errors propagate (an Exception carries the partial result
        as ``.agent_result``). Always appended to ``delegation_log`` and traced
        as ``delegation`` / ``delegation_end``.
        """
        agent = self.agents.get(agent_name)
        if agent is None:
            raise ToolFailure(f"unknown subagent_type {agent_name!r}; choose from {sorted(self.agents)}")
        inv = tool_use_id or f"{agent_name}:{uuid.uuid4().hex[:8]}"
        if parent_invocation_id is None:
            ids = _RUN_IDS.get()
            parent_invocation_id = ids[0] if ids else None
        entry: dict[str, Any] = {
            "agent": agent_name, "description": description, "tool_use_id": tool_use_id, "invocation_id": inv,
            "parent_invocation_id": parent_invocation_id, "parent_agent": parent_agent, "depth": depth,
            "start_ts": None, "end_ts": None, "status": "error", "stop_reason": None, "cost_usd": 0.0,
            "duration_s": 0.0, "model_calls": 0, "tool_calls": 0, "tool_errors": [],
            "unresolved_data_failures": [], "recovered_errors": 0, "other_errors": 0, "transcript_path": None, "error": None,
            "budget_tokens": 0.0,
        }
        history: list[Message] = []
        self._live[inv] = None
        me = asyncio.current_task()
        added = me is not None and me not in self._outstanding
        if added:
            self._outstanding.add(me)  # type: ignore[arg-type]
        result: AgentResult | None = None
        scope: CostScope | None = None
        t_req = time.time()
        try:
            async with self._parallel:
                entry["start_ts"] = time.time()
                self._trace("delegation", parent=parent_agent, agent=agent_name, description=description,
                            prompt=prompt[:8000], prompt_chars=len(prompt), tool_use_id=tool_use_id,
                            invocation_id=inv, parent_invocation_id=parent_invocation_id, depth=depth)
                self.emit("delegation", invocation_id=inv, parent_invocation_id=parent_invocation_id,
                          tool_use_id=tool_use_id, agent=agent_name, description=description,
                          prompt_preview=prompt[:500])
                with open_scope(f"delegation:{inv}") as scope:
                    try:
                        result = await self._run_delegated(agent, prompt, history, depth, inv, parent_invocation_id,
                                                           description, tool_use_id)
                    except BudgetExceeded as exc:
                        result = self._partial_result(inv, agent, history, "budget", f"budget exhausted: {exc}")
        except BaseException as exc:
            status = ("cancelled" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "error")
            result = self._partial_result(inv, agent, history, status, f"{type(exc).__name__}: {exc}")
            if isinstance(exc, Exception):
                try:
                    exc.agent_result = result  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            self._live.pop(inv, None)
            self._cancel_reasons.pop(inv, None)
            if added:
                self._outstanding.discard(me)  # type: ignore[arg-type]
            end = time.time()
            r = result
            entry.update(
                end_ts=end, ts=end, duration_s=round(end - (entry["start_ts"] or t_req), 1),
                cost_usd=round(scope.spent if scope is not None else (r.cost_usd if r else 0.0), 6),
                budget_tokens=round(scope.tokens if scope is not None else (r.tokens if r else 0.0), 1))
            if r is not None:
                entry.update(status=r.status, stop_reason=r.stop_reason, model_calls=r.model_calls,
                             tool_calls=r.tool_calls, tool_errors=list(r.tool_errors),
                             unresolved_data_failures=list(r.unresolved_data_failures),
                             recovered_errors=r.recovered_errors, other_errors=r.other_errors,
                             transcript_path=r.transcript_path, error=r.error)
            self.delegation_log.append(entry)
            self._trace("delegation_end", agent=agent_name, tool_use_id=tool_use_id, invocation_id=inv,
                        parent_invocation_id=parent_invocation_id, status=entry["status"],
                        stop_reason=entry["stop_reason"], cost_usd=entry["cost_usd"], duration_s=entry["duration_s"],
                        model_calls=entry["model_calls"], tool_calls=entry["tool_calls"],
                        n_tool_errors=len(entry["tool_errors"]), recovered_errors=entry["recovered_errors"],
                        other_errors=entry["other_errors"],
                        unresolved_data_failures=[{"tool": f.get("tool"), "error": str(f.get("error") or "")[:300]}
                                                  for f in entry["unresolved_data_failures"]],
                        transcript_path=entry["transcript_path"], error=entry["error"])
            self.emit("delegation_end", invocation_id=inv, parent_invocation_id=parent_invocation_id,
                      tool_use_id=tool_use_id, agent=agent_name, status=entry["status"],
                      stop_reason=entry["stop_reason"], cost_usd=entry["cost_usd"], duration_s=entry["duration_s"])
        return result

    async def _run_delegated(self, agent: AgentDefinition, prompt: str, history: list[Message], depth: int, inv: str,
                             parent: str | None, description: str, tool_use_id: str | None) -> AgentResult:
        kwargs = dict(history=history, depth=depth, invocation_id=inv, parent_invocation_id=parent,
                      description=description, tool_use_id=tool_use_id)
        timeout = self.delegation_timeout_s
        if not timeout:
            return await self.run_agent(agent, prompt, **kwargs)
        task = asyncio.ensure_future(self.run_agent(agent, prompt, **kwargs))
        self._track(task)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        if task in done:
            return task.result()
        self._cancel_reasons[inv] = "timeout"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._trace("delegation_timeout", agent=agent.name, invocation_id=inv, timeout_s=timeout)
        return self._partial_result(inv, agent, history, "timeout", f"delegation timed out after {timeout:g}s")

    def _partial_result(self, inv: str, agent: AgentDefinition, history: list[Message], status: str,
                        error: str) -> AgentResult:
        live = self._live.get(inv)
        r = live if isinstance(live, AgentResult) else AgentResult(agent.name, "", history, invocation_id=inv)
        r.status = status
        r.error = r.error or error
        if not r.text:
            r.text = _last_assistant_text(history)
        if not r.full_text:
            r.full_text = "\n\n".join(m.text for m in history if m.role == "assistant" and m.text)
        return r

    def _footer(self, name: str, sub: AgentResult) -> str:
        stop = f" ({sub.stop_reason})" if sub.stop_reason not in (None, "", "end_turn", sub.status) else ""
        spend = f"${sub.cost_usd:.2f}"
        if sub.tokens and not sub.cost_usd:   # local models: 0 USD, the token count is the cost
            spend = f"{budget.format_tokens(sub.tokens)} tokens"
        out = (f"[{name} finished: {sub.model_calls} model calls, {sub.tool_calls} tool calls, {spend}, "
               f"{sub.duration_s}s, status {sub.status}{stop}. Workspace: {self.workspace_for(name)}]")
        if sub.unresolved_data_failures:
            listed = "; ".join(f"{f.get('tool')}: {str(f.get('error') or '')[:160]}"
                               for f in sub.unresolved_data_failures[:8])
            out += f"\n[Unresolved data-source failures: {listed}]"
        if sub.recovered_errors:
            out += f"\n[{sub.recovered_errors} code/tool errors were encountered and recovered]"
        if sub.other_errors:
            out += (f"\n[{sub.other_errors} other code/tool errors (not data sources) were encountered "
                    f"and not resolved by an identical retry]")
        return out

    def _task_tool(self) -> Tool:
        async def handler(ctx: ToolContext, a: dict[str, Any]) -> str:
            name = a.get("subagent_type", "")
            if name not in self.agents:
                raise ToolFailure(f"unknown subagent_type {name!r}; choose from {sorted(self.agents)}")
            if ctx.depth + 1 > self.max_depth:
                raise ToolFailure("delegation depth limit reached; scientists cannot delegate further")
            try:
                sub = await self.delegate(name, a.get("prompt", ""), description=a.get("description", ""),
                                          parent_agent=ctx.agent, parent_invocation_id=ctx.invocation_id or None,
                                          tool_use_id=ctx.tool_call_id or None, depth=ctx.depth + 1)
            except ToolFailure:
                raise
            except Exception as exc:  # noqa: BLE001 - becomes this Task's error result
                partial = getattr(exc, "agent_result", None)
                body = (partial.text or partial.full_text) if partial is not None else ""
                footer = self._footer(name, partial) if partial is not None else ""
                raise ToolFailure(f"[incomplete: error] {type(exc).__name__}: {exc}"
                                  + (f"\n{body}" if body else "") + (f"\n---\n{footer}" if footer else "")) from None
            footer = self._footer(name, sub)
            if sub.status != "completed":
                body = sub.text or sub.full_text or "(no report was written)"
                if sub.status == "budget":
                    body = f"budget exhausted ({sub.error or 'cost limit reached'}).\n{body}"
                elif sub.status == "timeout":
                    body = f"{sub.error or 'timed out'}.\n{body}"
                raise ToolFailure(f"[incomplete: {sub.status}] {body}\n---\n{footer}")
            return f"{sub.text}\n\n---\n{footer}"

        return Tool(
            "Task",
            "Delegate a task to a specialist agent with its own isolated context and tools. Returns the "
            "specialist's final report. Multiple Task calls in one response run in parallel.",
            schema({"subagent_type": {"type": "string", "description": "specialist name"},
                    "description": {"type": "string", "description": "3-8 word summary"},
                    "prompt": {"type": "string", "description": "complete, self-contained instructions"}},
                   ["subagent_type", "prompt"]),
            handler, source="delegation")

    def _list_tools_tool(self) -> Tool:
        def handler(ctx: ToolContext, a: dict[str, Any]) -> Any:
            inventory: dict[str, list[str]] = {}
            for name in self.registry.names():
                t = self.registry.get(name)
                inventory.setdefault(t.source, []).append(f"{name}: {t.description.splitlines()[0][:160]}")
            failures_ = self.unavailable_servers()
            out: dict[str, Any] = {"tools_by_source": inventory, "unavailable_servers": failures_,
                                   "tools_not_ready": dict(self.tool_readiness),
                                   "agents": {n: a.description for n, a in self.agents.items()}}
            if self.mcp is not None:
                try:
                    out["servers"] = self.mcp.status()
                except Exception:  # noqa: BLE001
                    log.debug("MCP status failed", exc_info=True)
            if self.degraded_servers:
                out["servers_without_reference_data"] = dict(self.degraded_servers)
            return out
        return Tool("ListTools", "Inventory every data tool, MCP server and specialist available in this run.",
                    schema({}), handler)
