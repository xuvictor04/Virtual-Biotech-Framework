"""Massively parallel, schema-validated, resumable per-item agent runs.

The paper dispatched 37,075 clinical trialist agents in parallel — one per NCT
ID — each using its full context window on a single trial and returning JSON
validated against a Pydantic schema (Methods, "Benefits of multi-agent
architecture"). ``BulkRunner`` generalises that pattern:

* one fresh agent per item (isolated context), bounded concurrency
  (``concurrency``, default ``bulk.default_concurrency``);
* a terminal ``submit_result`` tool whose input schema *is* the Pydantic model,
  so any provider with tool calling returns structured output; it is declared
  ``strict`` (grammar-constrained arguments on vLLM); validation errors go back
  to the agent to fix, and a successful submit ends the item with no further
  model call;
* the agent's final allowed call is a forced ``submit_result`` call, and an
  agent that ends without submitting is continued ONCE in the same
  conversation with a harness nudge and a forced ``submit_result``
  (``tool_choice`` where the provider supports it) instead of re-running the
  item from scratch (record field ``continued``); fresh re-runs (``retries``)
  are kept for errors;
* results appended to JSONL as they finish (crash-safe, resumable: finished
  IDs are skipped on restart).

Budgets (``vbt.budget`` cost scopes)
------------------------------------
The whole run is one ``bulk`` scope (limits ``budget_usd`` and
``budget_tokens``); each item runs in its own ``item:<id>`` scope (limits
``limits.max_item_cost_usd`` and ``limits.max_item_tokens``); either limit stops
a scope. Local models cost 0 USD, so their bulk runs are budgeted in tokens.
The bulk scope is *detached* from any enclosing CSO ``turn`` scope, so the
per-turn caps never limit a bulk run; its spend is charged to the enclosing
scope afterwards for accounting only.

* An item's cost is its scope's spend: every attempt (including failed ones)
  and tool-side costs such as web-search fees; ``budget_tokens`` likewise.
* When the item scope runs out, the agent still makes one forced, budget-exempt
  ``submit_result`` call in its own conversation (``run_agent(force_tool=...)``),
  so the evidence gathered so far is submitted; only if that call does not
  submit does ``BudgetExceeded`` fail the item (status ``item_budget``, no retry).
* ``BudgetExceeded`` of the bulk scope (or the bulk spend reaching
  ``budget_usd`` / ``budget_tokens``) stops the run; items that never ran are
  not written.
* A non-retryable :class:`~vbt.providers.base.ProviderError` (authentication,
  permission, unknown model, bad request) is batch-fatal: the run stops, the
  summary is written with ``fatal_error`` and the error is re-raised. Transient
  errors are retried inside ``run_agent`` (P2); once retries are exhausted
  they are an ordinary per-item failure.

Prompt caching: with ``bulk.prewarm: first_item`` (default) one item runs
alone before the fan-out, so the shared tools+system prefix is written to the
provider cache once and read by every later item.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ValidationError

from . import budget as _budget
from .agents import AgentDefinition
from .budget import BudgetExceeded, CostScope
from .providers.base import ContextOverflowError, Message, ProviderError, RetryableProviderError
from .runtime import Runtime
from .tools.base import Tool, ToolContext, ToolFailure, inline_refs

__all__ = ["BulkItem", "BulkStats", "BulkRunner", "BULK_DEFAULTS", "bulk_settings", "max_item_cost",
           "max_item_tokens", "default_concurrency", "is_fatal_provider_error", "load_results", "add_bulk_parser",
           "SUBMIT_NUDGE", "ANNOTATE_CONCURRENCY", "unpriced_provider", "usd_only_budget_problem"]

#: In-code defaults for the ``bulk:`` config section. Also read (with in-code
#: defaults): ``default_concurrency`` (:data:`DEFAULT_CONCURRENCY`; concurrent
#: items when a caller does not say: vbt bulk, case1 annotate) and
#: ``dispatch_max_budget_tokens`` (cap on BulkDispatch budget_tokens; None = no cap; the local
#: default.yaml sets a finite one).
BULK_DEFAULTS: dict[str, Any] = {
    "dispatch_enabled": False,        # register BulkDispatch/BulkStatus for the CSO
    "dispatch_max_budget_usd": 50.0,  # upper bound on BulkDispatch budget_usd
    "pilot_size": 5,                  # items run by an unconfirmed BulkDispatch
    "prewarm": "first_item",          # 'first_item' | 'none'
}
DEFAULT_CONCURRENCY = 32
#: ``vbt case1 annotate`` when ``bulk.default_concurrency`` is unset (the Claude-era default; the local
#: profiles set default_concurrency to what their server's --max-num-seqs leaves room for).
ANNOTATE_CONCURRENCY = 64

SUBMIT_TOOL = "submit_result"
SUBMIT_REPLY = "Result recorded."
SUBMIT_NUDGE = (f"[Harness] You ended without calling {SUBMIT_TOOL}. Call {SUBMIT_TOOL} now with your final "
                "structured result, filled from the evidence you gathered (use the schema's unknown/null values "
                "where the evidence is missing).")
#: Agent statuses after which a missing submit is worth one continued, forced call.
_CONTINUABLE = ("completed", "turn_limit")


def bulk_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """``bulk:`` config section merged over :data:`BULK_DEFAULTS`."""
    out = dict(BULK_DEFAULTS)
    out.update({k: v for k, v in ((config or {}).get("bulk") or {}).items() if v is not None})
    return out


def _positive(v: Any) -> float | None:
    try:
        v = float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return v if v and v > 0 else None


def max_item_cost(config: dict[str, Any] | None) -> float | None:
    """``limits.max_item_cost_usd`` (default None = unlimited)."""
    return _positive(((config or {}).get("limits") or {}).get("max_item_cost_usd"))


def max_item_tokens(config: dict[str, Any] | None) -> float | None:
    """``limits.max_item_tokens`` (default None = unlimited)."""
    return _positive(((config or {}).get("limits") or {}).get("max_item_tokens"))


max_item_tokens_cfg = max_item_tokens  # BulkRunner's keyword argument shadows the function name


def default_concurrency(config: dict[str, Any] | None, fallback: int = DEFAULT_CONCURRENCY) -> int:
    """``bulk.default_concurrency``, else ``fallback`` (:data:`DEFAULT_CONCURRENCY` for
    ``vbt bulk``, :data:`ANNOTATE_CONCURRENCY` for ``vbt case1 annotate``): the
    Claude profiles leave the key unset, so each command keeps its own default."""
    try:
        return max(1, int(bulk_settings(config).get("default_concurrency") or fallback))
    except (TypeError, ValueError):
        return fallback


def unpriced_provider(provider: Any = None, config: dict[str, Any] | None = None) -> bool:
    """True when every model call costs 0 USD: a local OpenAI-compatible server
    (vllm, sglang, llamacpp, openai_compat) without a nominal ``pricing``. A USD
    budget can then never stop anything. Judged from the provider object when
    given, else from ``config['provider']``."""
    from .pinning import LOCAL_PROVIDER_NAMES

    if provider is not None:
        if getattr(provider, "name", None) not in LOCAL_PROVIDER_NAMES:
            return False
        pricing = getattr(provider, "pricing", None)
    else:
        prov = (config or {}).get("provider") or {}
        if prov.get("name") not in LOCAL_PROVIDER_NAMES:
            return False
        pricing = (prov.get("options") or {}).get("pricing")
    if not isinstance(pricing, dict):
        return True
    try:
        return not any(float(v or 0) > 0 for v in pricing.values())
    except (TypeError, ValueError):
        return True


def usd_only_budget_problem(budget_usd: Any, budget_tokens: Any, *, provider: Any = None,
                            config: dict[str, Any] | None = None, usd_name: str = "budget_usd",
                            tokens_name: str = "budget_tokens") -> str | None:
    """Why a bulk job budgeted only in USD would have no effective cap (the provider
    costs 0 USD), or None."""
    if not budget_usd or budget_tokens or not unpriced_provider(provider, config):
        return None
    return (f"{usd_name} alone cannot stop this job: the local model costs 0 USD (provider.options.pricing is "
            f"unset), so the spend never reaches it. Pass {tokens_name} (input + output tokens) instead, or "
            "set a nominal provider.options.pricing")


def is_fatal_provider_error(exc: BaseException) -> bool:
    """Non-retryable provider failure (auth, permission, not found, bad request):
    every further item would fail the same way."""
    return (isinstance(exc, ProviderError)
            and not isinstance(exc, (RetryableProviderError, ContextOverflowError)))


@dataclass
class BulkItem:
    id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _cost_block(values: list[float]) -> dict[str, Any]:
    r = lambda x: None if x is None else round(x, 6)  # noqa: E731
    return {"n": len(values), "median_usd": r(statistics.median(values)) if values else None,
            "p90_usd": r(_quantile(values, 0.9)), "total_usd": round(sum(values), 6)}


@dataclass
class BulkStats:
    done: int = 0
    failed: int = 0
    skipped: int = 0
    costs: list[float] = field(default_factory=list)      # every written item
    ok_costs: list[float] = field(default_factory=list)   # successful items only
    refusals: int = 0
    fallbacks: int = 0
    refused_items: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    tokens: list[float] = field(default_factory=list)     # budget tokens of every written item
    ok_tokens: list[float] = field(default_factory=list)  # successful items only
    continued: int = 0                                    # items continued once to submit
    continued_ok: int = 0                                 # ... that then submitted

    def add(self, rec: dict[str, Any]) -> None:
        c = float(rec.get("cost_usd") or 0.0)
        t = float(rec.get("budget_tokens") or 0.0)
        self.costs.append(c)
        self.tokens.append(t)
        if rec.get("ok"):
            self.done += 1
            self.ok_costs.append(c)
            self.ok_tokens.append(t)
        else:
            self.failed += 1
        self.refusals += int(rec.get("refusals") or 0)
        self.fallbacks += int(rec.get("fallbacks") or 0)
        if rec.get("refusals"):
            self.refused_items.append(str(rec.get("id")))
        if rec.get("continued"):
            self.continued += 1
            self.continued_ok += int(bool(rec.get("ok")))

    def summary(self) -> dict[str, Any]:
        """Medians/p90 over successful items (the paper's per-trial metric) and over
        all written items, separately; in USD and in budget tokens."""
        el = time.time() - self.started
        ok = _cost_block(self.ok_costs)
        med_tok = statistics.median(self.ok_tokens) if self.ok_tokens else None
        p90_tok = _quantile(self.ok_tokens, 0.9)
        return {
            "completed": self.done, "failed": self.failed, "skipped_existing": self.skipped,
            "total_cost_usd": round(sum(self.costs), 4),
            "median_cost_usd": None if ok["median_usd"] is None else round(ok["median_usd"], 4),
            "p90_cost_usd": None if ok["p90_usd"] is None else round(ok["p90_usd"], 4),
            "cost_successful": ok, "cost_all": _cost_block(self.costs),
            "total_tokens": round(sum(self.tokens)),
            "median_tokens": None if med_tok is None else round(med_tok),
            "p90_tokens": None if p90_tok is None else round(p90_tok),
            "continued": self.continued, "continued_submitted": self.continued_ok,
            "refusals": self.refusals, "fallbacks": self.fallbacks, "refused_items": self.refused_items[:50],
            "elapsed_s": round(el, 1),
            "items_per_hour": round(3600 * (self.done + self.failed) / el, 1) if el > 0 else None,
        }


def _submit_tool(model: type[BaseModel], sink: dict[str, Any]) -> Tool:
    schema = inline_refs(model.model_json_schema())

    def handler(ctx: ToolContext, args: dict[str, Any]) -> str:
        try:
            obj = model.model_validate(args)
        except (ValidationError, ValueError) as exc:  # JSON-Schema-backed models raise ValueError
            raise ToolFailure(f"schema validation failed; fix and resubmit:\n{exc}")
        sink["result"] = obj.model_dump(mode="json")
        return SUBMIT_REPLY

    return Tool(SUBMIT_TOOL,
                "Submit your final structured result (validated against the required schema). "
                "Call exactly once, when all fields are filled from evidence; this ends your task.",
                schema, handler, source="bulk", terminal=True, strict=True)


class _Fatal(Exception):
    def __init__(self, exc: BaseException):
        super().__init__(str(exc))
        self.exc = exc


_DEFAULT = object()


class BulkRunner:
    def __init__(self, runtime: Runtime, agent: AgentDefinition, output_model: type[BaseModel],
                 out_path: Path, *, concurrency: int | None = None, retries: int = 1,
                 budget_usd: float | None = None, max_item_cost_usd: Any = _DEFAULT,
                 prewarm: str | None = None, on_progress: Callable[[BulkStats], None] | None = None,
                 detach: bool = True, account_to_parent: bool = True, job_id: str | None = None,
                 stop_event: asyncio.Event | None = None, budget_tokens: float | None = None,
                 max_item_tokens: Any = _DEFAULT, continue_unsubmitted: bool = True):
        self.rt = runtime
        self.agent = agent
        self.model = output_model
        self.out_path = Path(out_path)
        self.concurrency = max(1, int(concurrency)) if concurrency else default_concurrency(runtime.config)
        self.retries = max(0, int(retries))
        self.budget_usd = float(budget_usd) if budget_usd else None
        self.budget_tokens = float(budget_tokens) if budget_tokens else None
        self.max_item_cost_usd = (max_item_cost(runtime.config) if max_item_cost_usd is _DEFAULT
                                  else max_item_cost_usd)
        self.max_item_tokens = (max_item_tokens_cfg(runtime.config) if max_item_tokens is _DEFAULT
                                else max_item_tokens)
        self.continue_unsubmitted = bool(continue_unsubmitted)
        self.prewarm = prewarm if prewarm is not None else bulk_settings(runtime.config)["prewarm"]
        self.on_progress = on_progress
        self.detach = detach
        self.account_to_parent = account_to_parent
        self.job_id = job_id
        self._write_lock = asyncio.Lock()
        self.stats = BulkStats()
        self.stop = stop_event or asyncio.Event()
        self.budget_stop = False
        self.fatal: BaseException | None = None
        self.scope: CostScope | None = None
        self.records: list[dict[str, Any]] = []
        self.queued = 0

    # ------------------------------------------------------------------ state

    def completed_ids(self) -> set[str]:
        if not self.out_path.exists():
            return set()
        ids = set()
        for line in self.out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ok"):
                ids.add(rec["id"])
        return ids

    def _bulk_exhausted(self) -> bool:
        s = self.scope
        if s is None:
            return False
        return bool((self.budget_usd and s.spent >= self.budget_usd)
                    or (self.budget_tokens and s.tokens >= self.budget_tokens))

    # ------------------------------------------------------------------ one item

    async def _attempt(self, item: BulkItem, sink: dict[str, Any], key: str, counts: dict[str, Any]):
        """One attempt: the agent run, plus (when it ended without submitting) one
        continuation of the same conversation with a forced ``submit_result``.
        ``counts`` accumulates model calls / fallbacks / refusals even when the
        continuation raises."""
        history: list[Message] = []
        submit = _submit_tool(self.model, sink)

        def tally(r: Any) -> None:
            counts["model_calls"] += int(getattr(r, "model_calls", 0) or 0)
            counts["fallbacks"] += int(getattr(r, "fallback_count", 0) or 0)
            if getattr(r, "status", "") == "refusal":
                counts["refusals"] += 1

        async def run(task: str, **kw: Any) -> Any:
            try:
                r = await self.rt.run_agent(self.agent, task, history=history, depth=1, extra_tools=[submit],
                                            force_tool=SUBMIT_TOOL, session_key=key, **kw)
            except BudgetExceeded as exc:
                # The item's budget ran out: the runtime already granted the forced submit_result call
                # (budget-exempt); count the calls this invocation made before re-raising.
                partial = getattr(exc, "agent_result", None)
                if partial is not None:
                    tally(partial)
                raise
            tally(r)
            return r

        res = await run(item.prompt, description=f"bulk item {item.id}")
        if "result" in sink or not self.continue_unsubmitted or getattr(res, "status", "") not in _CONTINUABLE:
            return res
        counts["continued"] = True
        self.rt.run.trace("bulk_continue", item=item.id, agent=self.agent.name, status=res.status)
        return await run(SUBMIT_NUDGE, description=f"bulk item {item.id} (submit)", max_turns=0)

    async def _one(self, item: BulkItem) -> dict[str, Any] | None:
        """Run one item. Returns its record, or None when it never ran (bulk budget)."""
        t0 = time.time()
        last_err, status = "", "error"
        attempts = 0
        counts: dict[str, Any] = {"model_calls": 0, "fallbacks": 0, "refusals": 0, "continued": False}
        with self.rt.cost_scope(f"item:{item.id}", self.max_item_cost_usd,
                                limit_tokens=self.max_item_tokens) as iscope:

            def record(ok: bool, **fields: Any) -> dict[str, Any]:
                rec = {"id": item.id, "ok": ok, **fields, "cost_usd": round(iscope.spent, 6),
                       "budget_tokens": round(iscope.tokens), "model_calls": counts["model_calls"],
                       "attempts": attempts, "fallbacks": counts["fallbacks"], "refusals": counts["refusals"],
                       "elapsed_s": round(time.time() - t0, 2), "meta": item.meta}
                if counts["continued"]:
                    rec["continued"] = True
                return rec

            for attempt in range(self.retries + 1):
                attempts = attempt + 1
                sink: dict[str, Any] = {}
                key = f"bulk:{self.job_id or self.agent.name}:{item.id}:{attempt}"
                try:
                    res = await self._attempt(item, sink, key, counts)
                except BudgetExceeded as exc:
                    sc = exc.scope
                    if sc is iscope or (sc is not None and sc.name.startswith("item:")):
                        status, last_err = "item_budget", f"BudgetExceeded: {exc}"
                        break
                    # bulk (or an outer) scope: stop the batch
                    self.budget_stop = True
                    self.stop.set()
                    if iscope.spent <= 0 and iscope.tokens <= 0 and attempt == 0:
                        return None
                    status, last_err = "bulk_budget", f"BudgetExceeded: {exc}"
                    break
                except asyncio.CancelledError:
                    raise
                except ProviderError as exc:
                    if is_fatal_provider_error(exc):
                        raise _Fatal(exc) from exc
                    status, last_err = "provider_error", f"{type(exc).__name__}: {exc}"
                    continue
                except Exception as exc:  # noqa: BLE001 - one bad item must not stop the batch
                    status, last_err = "error", f"{type(exc).__name__}: {exc}"
                    continue
                if "result" in sink:
                    return record(True, result=sink["result"], status="ok", agent_status=res.status)
                status = getattr(res, "status", "completed") or "completed"
                last_err = f"agent finished ({status}) without {SUBMIT_TOOL}: {(res.text or '')[:300]}"
                if status == "refusal" or counts["continued"]:
                    # a refusal is not worth a paid retry; an agent that ignored a forced submit in its own
                    # conversation is not re-run from scratch
                    break
            return record(False, error=last_err, status=status)

    async def _process(self, item: BulkItem) -> None:
        try:
            rec = await self._one(item)
        except _Fatal as f:
            if self.fatal is None:
                self.fatal = f.exc
            self.stop.set()
            return
        if rec is None:
            return
        async with self._write_lock:
            with open(self.out_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            self.stats.add(rec)
            self.records.append({k: rec.get(k) for k in ("id", "ok", "status", "cost_usd", "budget_tokens",
                                                         "attempts", "model_calls", "fallbacks", "refusals",
                                                         "continued", "error")})
            if self.on_progress:
                try:
                    self.on_progress(self.stats)
                except Exception:  # noqa: BLE001
                    pass
            self.rt.emit("bulk_progress", job_id=self.job_id, agent=self.agent.name,
                         done=self.stats.done, failed=self.stats.failed, queued=self.queued,
                         spent_usd=round(self.scope.spent, 6) if self.scope else None,
                         spent_tokens=round(self.scope.tokens) if self.scope else None)
            if self._bulk_exhausted():
                self.budget_stop = True
                self.stop.set()

    # ------------------------------------------------------------------ run

    async def run(self, items: Iterable[BulkItem]) -> dict[str, Any]:
        done_ids = self.completed_ids()
        pending: list[BulkItem] = []
        for it in items:
            if it.id in done_ids:
                self.stats.skipped += 1
                continue
            pending.append(it)
        self.queued = n = len(pending)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        queue: asyncio.Queue[BulkItem] = asyncio.Queue()
        for it in pending:
            queue.put_nowait(it)

        ledger = self.rt.run.cost
        before = ledger.by_agent.get(self.agent.name)
        before = before.as_dict() if before is not None else {}
        outer = _budget.current_scope()
        _budget.set_scope(None if self.detach else outer)
        try:
            with self.rt.cost_scope("bulk", self.budget_usd, limit_tokens=self.budget_tokens) as bscope:
                self.scope = bscope

                async def worker() -> None:
                    while not self.stop.is_set():
                        if self._bulk_exhausted():
                            self.budget_stop = True
                            self.stop.set()
                            return
                        try:
                            item = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        await self._process(item)

                prewarmed = False
                if (self.prewarm == "first_item" and n > 1 and self.concurrency > 1 and not queue.empty()):
                    # one item alone writes the cached tools+system prefix before the fan-out
                    await self._process(queue.get_nowait())
                    prewarmed = True
                if not self.stop.is_set():
                    await asyncio.gather(*(worker() for _ in range(min(self.concurrency, max(queue.qsize(), 1)))))
        finally:
            _budget.set_scope(outer)
            if self.detach and self.account_to_parent and outer is not None and self.scope is not None:
                # accounting only: the turn caps never stopped this run
                outer.charge(self.scope.spent, self.scope.tokens)

        after = ledger.by_agent.get(self.agent.name)
        after = after.as_dict() if after is not None else {}
        delta = {k: int(after.get(k, 0) or 0) - int(before.get(k, 0) or 0)
                 for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
        prompt_tokens = delta["input_tokens"] + delta["cache_read_tokens"] + delta["cache_write_tokens"]
        summary = {**self.stats.summary(), "queued": n, "budget_usd": self.budget_usd,
                   "budget_spent": round(self.scope.spent, 6) if self.scope else 0.0,
                   "budget_tokens": self.budget_tokens,
                   "budget_tokens_spent": round(self.scope.tokens) if self.scope else 0,
                   "budget_stop": self.budget_stop, "max_item_cost_usd": self.max_item_cost_usd,
                   "max_item_tokens": self.max_item_tokens, "concurrency": self.concurrency,
                   "input_tokens": delta["input_tokens"], "output_tokens": delta["output_tokens"],
                   "prewarm": self.prewarm if prewarmed else "none",
                   "cache_read_tokens": delta["cache_read_tokens"], "cache_write_tokens": delta["cache_write_tokens"],
                   "cache_hit_rate": round(delta["cache_read_tokens"] / prompt_tokens, 4) if prompt_tokens else None,
                   "unrun": queue.qsize(), "output": str(self.out_path), "agent": self.agent.name}
        if self.job_id:
            summary["job_id"] = self.job_id
        if self.fatal is not None:
            summary["fatal_error"] = f"{type(self.fatal).__name__}: {self.fatal}"
        try:
            self.rt.run.trace("bulk_summary", **summary)
        except Exception:  # noqa: BLE001 - the summary is also returned / raised with
            pass
        self.summary = summary
        if self.fatal is not None:
            try:
                self.fatal.bulk_summary = summary  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            raise self.fatal
        return summary


def load_results(path: Path) -> list[dict[str, Any]]:
    """Latest record per ID (a later success supersedes an earlier failure)."""
    latest: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line from a crash
            if rec["id"] not in latest or rec.get("ok") or not latest[rec["id"]].get("ok"):
                latest[rec["id"]] = rec
    return list(latest.values())


# ---------------------------------------------------------------- CLI

def add_bulk_parser(sub) -> None:
    p = sub.add_parser("bulk", help="generic bulk run: one agent per line of an input JSONL")
    p.add_argument("input", help="JSONL with {id, prompt, ...}")
    p.add_argument("--agent", required=True, help="agent name from configs/agents.yaml")
    p.add_argument("--schema", required=True, help="dotted path to a Pydantic model, e.g. pkg.mod:Model")
    p.add_argument("--out", required=True)
    add_budget_arguments(p)
    p.add_argument("--prewarm", choices=["first_item", "none"])
    p.set_defaults(handler=_cli)


def add_budget_arguments(p, *, what: str = "item") -> None:
    """``--concurrency``, ``--budget``, ``--budget-tokens``, ``--max-item-cost`` and
    ``--max-item-tokens`` (shared by ``vbt bulk`` and ``vbt case1 annotate``)."""
    p.add_argument("--concurrency", type=int, default=None,
                   help="concurrent agents (default: bulk.default_concurrency, 32; size it to the server's "
                        "--max-num-seqs for a local model)")
    p.add_argument("--budget", type=float, help="stop the bulk run when its spend reaches this many USD "
                   "(the per-turn caps never apply to bulk runs); refused alone with a 0-USD local model - "
                   "use --budget-tokens")
    p.add_argument("--budget-tokens", type=float,
                   help="stop the bulk run when it has used this many tokens (input + output, cached input "
                        "at limits.cached_token_weight); local models cost 0 USD, so budget them in tokens")
    p.add_argument("--max-item-cost", type=float, help=f"per-{what} cost cap in USD (limits.max_item_cost_usd)")
    p.add_argument("--max-item-tokens", type=float, help=f"per-{what} token cap (limits.max_item_tokens)")


def budget_kwargs(args) -> dict[str, Any]:
    """BulkRunner keyword arguments from :func:`add_budget_arguments` options."""
    kw: dict[str, Any] = {}
    if getattr(args, "max_item_cost", None):
        kw["max_item_cost_usd"] = args.max_item_cost
    if getattr(args, "max_item_tokens", None):
        kw["max_item_tokens"] = args.max_item_tokens
    if getattr(args, "budget_tokens", None):
        kw["budget_tokens"] = args.budget_tokens
    if getattr(args, "concurrency", None):
        kw["concurrency"] = args.concurrency
    return kw


def _cli(args, config) -> int:
    import importlib
    import sys

    from .orchestrator import open_session

    problem = usd_only_budget_problem(args.budget, getattr(args, "budget_tokens", None), config=config,
                                      usd_name="--budget", tokens_name="--budget-tokens")
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    mod, _, name = args.schema.partition(":")
    model = getattr(importlib.import_module(mod), name)
    items = [BulkItem(str(r["id"]), r["prompt"], r) for r in
             (json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip())]

    async def main() -> int:
        session = await open_session(config, start_mcp=not args.no_mcp)
        try:
            runner = BulkRunner(session.rt, session.rt.agents[args.agent], model, Path(args.out),
                                budget_usd=args.budget, prewarm=getattr(args, "prewarm", None),
                                **budget_kwargs(args))
            try:
                print(json.dumps(await runner.run(items), indent=2))
            except ProviderError as exc:
                print(f"bulk run stopped by a fatal provider error: {exc}", file=sys.stderr)
                print(json.dumps(getattr(exc, "bulk_summary", {}), indent=2))
                return 1
        finally:
            await session.close()
        return 0

    return asyncio.run(main())
