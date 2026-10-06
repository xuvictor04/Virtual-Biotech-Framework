"""Cost scopes: budgets that apply to what they are meant to limit.

A :class:`CostScope` is a named spend counter with an optional USD limit and an
optional token limit (a local model costs 0 USD, so its budgets are in tokens).
A scope is exceeded when EITHER limit is hit. Scopes
nest through a :class:`~contextvars.ContextVar`: opening a scope inside another
makes the outer one its ``parent``, and :func:`charge` adds a cost to every
scope in the chain. Because asyncio tasks copy the context when they are
created, a CSO turn scope covers the CSO's own model calls, every tool call and
every delegated specialist started inside it, while code that never opens a
turn scope (bulk runs, scoring judges) is never limited by the per-turn cap.

Typical layout::

    with open_scope("turn", limits.max_turn_cost_usd,
                    limit_tokens=limits.max_turn_tokens):            # CSO turn
        ...                                                          # delegations inherit it
    with open_scope("bulk", budget_usd, limit_tokens=budget_tokens): # a bulk run
        with open_scope(f"item:{id}", max_item_cost_usd,
                        limit_tokens=max_item_tokens):               # one item
            ...

Spend is charged when it happens (after each model call or paid tool call), so
the cost of an attempt that later raises stays in every enclosing scope.
:func:`check` (called before each model call) raises :class:`BudgetExceeded`
for the innermost exceeded scope.

Tokens: every model call charges its *budget tokens* (:func:`usage_tokens`):
uncached input + cache writes + output, plus cached input (prompt-cache reads)
at a weight (default :data:`CACHED_TOKEN_WEIGHT` = 0.1, configurable as
``limits.cached_token_weight``), since a prefix-cache hit costs a local server a
small fraction of a recomputed prompt.

:class:`InvocationCost` is a second, independent accumulator for one
``run_agent`` invocation (its own model calls plus tool-side costs such as web
search fees), used for ``AgentResult.cost_usd``.

No provider or runtime imports here.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

__all__ = [
    "BudgetExceeded", "CACHED_TOKEN_WEIGHT", "CostScope", "InvocationCost", "open_scope", "current_scope", "charge",
    "check", "exceeded_scope", "current_invocation", "use_invocation", "reset_invocation", "usage_tokens",
    "cached_token_weight", "format_tokens",
]

#: Weight of cached input tokens (prompt-cache reads) in token budgets.
CACHED_TOKEN_WEIGHT = 0.1

_LOCK = threading.Lock()  # tool handlers may charge from worker threads (Tool.blocking)


class BudgetExceeded(RuntimeError):
    """A cost scope's limit was exceeded. ``scope`` is the exceeded scope (or None)."""

    def __init__(self, message: str = "budget exceeded", *, scope: "CostScope | None" = None) -> None:
        super().__init__(message)
        self.scope = scope


def _norm_limit(limit: Any) -> float | None:
    """``None``, ``0`` or a non-positive/invalid value mean "unlimited"."""
    if limit is None:
        return None
    try:
        v = float(limit)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def cached_token_weight(config: Mapping[str, Any] | None = None) -> float:
    """``limits.cached_token_weight`` of a config (default :data:`CACHED_TOKEN_WEIGHT`)."""
    limits = (config or {}).get("limits") if isinstance(config, Mapping) else None
    v = (limits or {}).get("cached_token_weight") if isinstance(limits, Mapping) else None
    try:
        w = float(v)
    except (TypeError, ValueError):
        return CACHED_TOKEN_WEIGHT
    return w if w >= 0 else CACHED_TOKEN_WEIGHT


def usage_tokens(usage: Any, cached_weight: float = CACHED_TOKEN_WEIGHT) -> float:
    """Budget tokens of one model call's ``Usage``: uncached input + cache writes +
    output, plus cache reads at ``cached_weight``. 0 for None."""
    if usage is None:
        return 0.0

    def n(name: str) -> float:
        try:
            return float(getattr(usage, name, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    return (n("input_tokens") + n("cache_write_tokens") + n("output_tokens")
            + float(cached_weight) * n("cache_read_tokens"))


def format_tokens(n: Any) -> str:
    """``1234567`` -> ``1.23M``; ``12345`` -> ``12.3K``; ``None`` -> ``0``."""
    try:
        v = float(n or 0)
    except (TypeError, ValueError):
        return str(n)
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:.0f}"


@dataclass(eq=False)
class CostScope:
    """A named spend counter with optional USD and token limits, chained to its parent."""

    name: str
    limit_usd: float | None = None
    parent: "CostScope | None" = None
    spent: float = 0.0
    grace_used: bool = False  # the depth-0 agent got its one final no-tool call
    meta: dict[str, Any] = field(default_factory=dict)
    limit_tokens: float | None = None
    tokens: float = 0.0       # budget tokens charged (see usage_tokens)

    def __post_init__(self) -> None:
        self.limit_usd = _norm_limit(self.limit_usd)
        self.limit_tokens = _norm_limit(self.limit_tokens)

    # ------------------------------------------------------------------ queries

    @property
    def usd_exceeded(self) -> bool:
        return self.limit_usd is not None and self.spent > self.limit_usd

    @property
    def tokens_exceeded(self) -> bool:
        return self.limit_tokens is not None and self.tokens > self.limit_tokens

    @property
    def exceeded(self) -> bool:
        """Either limit is exceeded."""
        return self.usd_exceeded or self.tokens_exceeded

    @property
    def exhausted(self) -> bool:
        """Either limit is reached (spend >= limit): nothing new should start."""
        return ((self.limit_usd is not None and self.spent >= self.limit_usd)
                or (self.limit_tokens is not None and self.tokens >= self.limit_tokens))

    @property
    def remaining_usd(self) -> float | None:
        return None if self.limit_usd is None else self.limit_usd - self.spent

    @property
    def remaining_tokens(self) -> float | None:
        return None if self.limit_tokens is None else self.limit_tokens - self.tokens

    def describe(self) -> str:
        """Which limit is exceeded, with the numbers (for messages)."""
        bits = []
        if self.usd_exceeded:
            bits.append(f"cost exceeded ${self.limit_usd:.2f} (spent ${self.spent:.2f})")
        if self.tokens_exceeded:
            bits.append(f"tokens exceeded {format_tokens(self.limit_tokens)} (used {format_tokens(self.tokens)})")
        return "; ".join(bits) or "within budget"

    def chain(self) -> list["CostScope"]:
        """This scope and its ancestors, innermost first."""
        out: list[CostScope] = []
        s: CostScope | None = self
        while s is not None and s not in out:
            out.append(s)
            s = s.parent
        return out

    @property
    def path(self) -> str:
        """``outer/inner`` names, for messages and traces."""
        return "/".join(s.name for s in reversed(self.chain()))

    def find(self, name: str) -> "CostScope | None":
        """Nearest scope in the chain called ``name``."""
        return next((s for s in self.chain() if s.name == name), None)

    # ------------------------------------------------------------------ updates

    def charge(self, usd: float, tokens: float = 0.0) -> None:
        """Add ``usd`` (and budget ``tokens``) to this scope and every ancestor."""
        if not usd and not tokens:
            return
        with _LOCK:
            for s in self.chain():
                s.spent += usd or 0.0
                s.tokens += tokens or 0.0

    def reset(self) -> None:
        self.spent = 0.0
        self.tokens = 0.0
        self.grace_used = False

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "limit_usd": self.limit_usd, "spent": round(self.spent, 6),
                "limit_tokens": self.limit_tokens, "tokens": round(self.tokens, 1), "grace_used": self.grace_used}


_SCOPE: ContextVar[CostScope | None] = ContextVar("vbt_cost_scope", default=None)


def current_scope() -> CostScope | None:
    return _SCOPE.get()


@contextmanager
def open_scope(name: str, limit_usd: float | None = None, *, limit_tokens: float | None = None,
               **meta: Any) -> Iterator[CostScope]:
    """Open a scope nested in the current one for the duration of the block."""
    scope = CostScope(name, limit_usd, parent=_SCOPE.get(), meta=dict(meta), limit_tokens=limit_tokens)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        try:
            _SCOPE.reset(token)
        except ValueError:  # reset from another context (e.g. generator closed elsewhere)
            _SCOPE.set(scope.parent)


def set_scope(scope: CostScope | None) -> None:
    """Make ``scope`` current in this context without a ``with`` block (compat shims)."""
    _SCOPE.set(scope)


def charge(usd: float, tokens: float = 0.0) -> None:
    """Charge ``usd`` (and budget ``tokens``) to every scope in the current chain
    (no-op without a scope)."""
    scope = _SCOPE.get()
    if scope is not None and (usd or tokens):
        scope.charge(float(usd or 0.0), float(tokens or 0.0))


def exceeded_scope() -> CostScope | None:
    """Innermost exceeded scope of the current chain, if any."""
    scope = _SCOPE.get()
    if scope is None:
        return None
    return next((s for s in scope.chain() if s.exceeded), None)


def check() -> None:
    """Raise :class:`BudgetExceeded` for the innermost exceeded scope."""
    s = exceeded_scope()
    if s is not None:
        raise BudgetExceeded(f"{s.name} {s.describe()}; scope {s.path}", scope=s)


# ---------------------------------------------------------------------------
# Per-invocation accumulator
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class InvocationCost:
    """Spend of one ``run_agent`` invocation (not its delegated sub-agents)."""

    invocation_id: str = ""
    usd: float = 0.0
    model_usd: float = 0.0
    tool_usd: float = 0.0
    model_calls: int = 0
    tokens: float = 0.0       # budget tokens of its model calls

    def add(self, usd: float, *, model_call: bool, tokens: float = 0.0) -> None:
        usd = float(usd or 0.0)
        with _LOCK:
            self.usd += usd
            self.tokens += float(tokens or 0.0)
            if model_call:
                self.model_usd += usd
                self.model_calls += 1
            else:
                self.tool_usd += usd


_INVOCATION: ContextVar[InvocationCost | None] = ContextVar("vbt_invocation_cost", default=None)


def current_invocation() -> InvocationCost | None:
    return _INVOCATION.get()


def use_invocation(acc: InvocationCost | None):
    """Make ``acc`` the current accumulator; returns a token for :func:`reset_invocation`."""
    return _INVOCATION.set(acc)


def reset_invocation(token) -> None:
    try:
        _INVOCATION.reset(token)
    except ValueError:
        _INVOCATION.set(None)
