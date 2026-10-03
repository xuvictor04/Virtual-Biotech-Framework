"""Massively parallel, schema-validated, resumable per-item agent runs.

The paper dispatched 37,075 clinical trialist agents in parallel — one per NCT
ID — each using its full context window on a single trial and returning JSON
validated against a Pydantic schema (Methods, "Benefits of multi-agent
architecture"). ``BulkRunner`` generalises that pattern:

* one fresh agent per item (isolated context), bounded concurrency;
* a terminal ``submit_result`` tool whose input schema *is* the Pydantic model,
  so any provider with tool calling returns structured output; validation
  errors go back to the agent to fix, and a successful submit ends the item
  with no further model call;
* results appended to JSONL as they finish (crash-safe, resumable: finished
  IDs are skipped on restart).

Budgets (``vbt.budget`` cost scopes)
------------------------------------
The whole run is one ``bulk`` scope (limit ``budget_usd``); each item runs in
its own ``item:<id>`` scope (limit ``limits.max_item_cost_usd``). The bulk
scope is *detached* from any enclosing CSO ``turn`` scope, so the per-turn cap
(``limits.max_turn_cost_usd``) never limits a bulk run; its spend is charged
to the enclosing scope afterwards for accounting only.

* An item's cost is its scope's spend: every attempt (including failed ones)
  and tool-side costs such as web-search fees.
* ``BudgetExceeded`` of the item scope fails that item (no retry).
* ``BudgetExceeded`` of the bulk scope (or the bulk spend reaching
  ``budget_usd``) stops the run; items that never ran are not written.
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
from .providers.base import ContextOverflowError, ProviderError, RetryableProviderError
from .runtime import Runtime
from .tools.base import Tool, ToolContext, ToolFailure, inline_refs

__all__ = ["BulkItem", "BulkStats", "BulkRunner", "BULK_DEFAULTS", "bulk_settings", "max_item_cost",
           "is_fatal_provider_error", "load_results", "add_bulk_parser"]

#: In-code defaults for the ``bulk:`` config section.
BULK_DEFAULTS: dict[str, Any] = {
    "dispatch_enabled": False,        # register BulkDispatch/BulkStatus for the CSO
    "dispatch_max_budget_usd": 50.0,  # upper bound on BulkDispatch budget_usd
    "pilot_size": 5,                  # items run by an unconfirmed BulkDispatch
    "prewarm": "first_item",          # 'first_item' | 'none'
}

SUBMIT_TOOL = "submit_result"
SUBMIT_REPLY = "Result recorded."


def bulk_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """``bulk:`` config section merged over :data:`BULK_DEFAULTS`."""
    out = dict(BULK_DEFAULTS)
    out.update({k: v for k, v in ((config or {}).get("bulk") or {}).items() if v is not None})
    return out


def max_item_cost(config: dict[str, Any] | None) -> float | None:
    """``limits.max_item_cost_usd`` (default None = unlimited)."""
    v = ((config or {}).get("limits") or {}).get("max_item_cost_usd")
    try:
        v = float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return v if v and v > 0 else None


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

    def add(self, rec: dict[str, Any]) -> None:
        c = float(rec.get("cost_usd") or 0.0)
        self.costs.append(c)
        if rec.get("ok"):
            self.done += 1
            self.ok_costs.append(c)
        else:
            self.failed += 1
        self.refusals += int(rec.get("refusals") or 0)
        self.fallbacks += int(rec.get("fallbacks") or 0)
        if rec.get("refusals"):
            self.refused_items.append(str(rec.get("id")))

    def summary(self) -> dict[str, Any]:
        """Medians/p90 over successful items (the paper's per-trial metric) and over
        all written items, separately."""
        el = time.time() - self.started
        ok = _cost_block(self.ok_costs)
        return {
            "completed": self.done, "failed": self.failed, "skipped_existing": self.skipped,
            "total_cost_usd": round(sum(self.costs), 4),
            "median_cost_usd": None if ok["median_usd"] is None else round(ok["median_usd"], 4),
            "p90_cost_usd": None if ok["p90_usd"] is None else round(ok["p90_usd"], 4),
            "cost_successful": ok, "cost_all": _cost_block(self.costs),
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
                schema, handler, source="bulk", terminal=True)


class _Fatal(Exception):
    def __init__(self, exc: BaseException):
        super().__init__(str(exc))
        self.exc = exc


_DEFAULT = object()


class BulkRunner:
    def __init__(self, runtime: Runtime, agent: AgentDefinition, output_model: type[BaseModel],
                 out_path: Path, *, concurrency: int = 32, retries: int = 1,
                 budget_usd: float | None = None, max_item_cost_usd: Any = _DEFAULT,
                 prewarm: str | None = None, on_progress: Callable[[BulkStats], None] | None = None,
                 detach: bool = True, account_to_parent: bool = True, job_id: str | None = None,
                 stop_event: asyncio.Event | None = None):
        self.rt = runtime
        self.agent = agent
        self.model = output_model
        self.out_path = Path(out_path)
        self.concurrency = max(1, int(concurrency))
        self.retries = max(0, int(retries))
        self.budget_usd = float(budget_usd) if budget_usd else None
        self.max_item_cost_usd = (max_item_cost(runtime.config) if max_item_cost_usd is _DEFAULT
                                  else max_item_cost_usd)
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
        return bool(self.budget_usd and s is not None and s.spent >= self.budget_usd)

    # ------------------------------------------------------------------ one item

    async def _one(self, item: BulkItem) -> dict[str, Any] | None:
        """Run one item. Returns its record, or None when it never ran (bulk budget)."""
        t0 = time.time()
        last_err, status = "", "error"
        attempts = model_calls = fallbacks = refusals = 0
        with self.rt.cost_scope(f"item:{item.id}", self.max_item_cost_usd) as iscope:
            for attempt in range(self.retries + 1):
                attempts = attempt + 1
                sink: dict[str, Any] = {}
                try:
                    res = await self.rt.run_agent(self.agent, item.prompt, depth=1,
                                                  extra_tools=[_submit_tool(self.model, sink)],
                                                  description=f"bulk item {item.id}")
                except BudgetExceeded as exc:
                    sc = exc.scope
                    if sc is iscope or (sc is not None and sc.name.startswith("item:")):
                        status, last_err = "item_budget", f"BudgetExceeded: {exc}"
                        break
                    # bulk (or an outer) scope: stop the batch
                    self.budget_stop = True
                    self.stop.set()
                    if iscope.spent <= 0 and attempt == 0:
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
                model_calls += res.model_calls
                fallbacks += int(getattr(res, "fallback_count", 0) or 0)
                if getattr(res, "status", "") == "refusal":
                    refusals += 1
                if "result" in sink:
                    return {"id": item.id, "ok": True, "result": sink["result"], "cost_usd": round(iscope.spent, 6),
                            "model_calls": model_calls, "attempts": attempts, "status": "ok",
                            "agent_status": res.status, "fallbacks": fallbacks, "refusals": refusals,
                            "elapsed_s": round(time.time() - t0, 2), "meta": item.meta}
                status = getattr(res, "status", "completed") or "completed"
                last_err = f"agent finished ({status}) without {SUBMIT_TOOL}: {(res.text or '')[:300]}"
                if status == "refusal":
                    break  # a refusal is not worth a paid retry
        return {"id": item.id, "ok": False, "error": last_err, "status": status, "cost_usd": round(iscope.spent, 6),
                "model_calls": model_calls, "attempts": attempts, "fallbacks": fallbacks, "refusals": refusals,
                "elapsed_s": round(time.time() - t0, 2), "meta": item.meta}

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
            self.records.append({k: rec.get(k) for k in ("id", "ok", "status", "cost_usd", "attempts",
                                                         "model_calls", "fallbacks", "refusals", "error")})
            if self.on_progress:
                try:
                    self.on_progress(self.stats)
                except Exception:  # noqa: BLE001
                    pass
            self.rt.emit("bulk_progress", job_id=self.job_id, agent=self.agent.name,
                         done=self.stats.done, failed=self.stats.failed, queued=self.queued,
                         spent_usd=round(self.scope.spent, 6) if self.scope else None)
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
            with self.rt.cost_scope("bulk", self.budget_usd) as bscope:
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
                outer.charge(self.scope.spent)  # accounting only: the turn cap never stopped this run

        after = ledger.by_agent.get(self.agent.name)
        after = after.as_dict() if after is not None else {}
        delta = {k: int(after.get(k, 0) or 0) - int(before.get(k, 0) or 0)
                 for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
        prompt_tokens = delta["input_tokens"] + delta["cache_read_tokens"] + delta["cache_write_tokens"]
        summary = {**self.stats.summary(), "queued": n, "budget_usd": self.budget_usd,
                   "budget_spent": round(self.scope.spent, 6) if self.scope else 0.0,
                   "budget_stop": self.budget_stop, "max_item_cost_usd": self.max_item_cost_usd,
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
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--budget", type=float, help="stop the bulk run when its spend reaches this many USD")
    p.add_argument("--max-item-cost", type=float, help="per-item cost cap in USD (limits.max_item_cost_usd)")
    p.add_argument("--prewarm", choices=["first_item", "none"])
    p.set_defaults(handler=_cli)


def _cli(args, config) -> int:
    import importlib
    import sys

    from .orchestrator import open_session

    mod, _, name = args.schema.partition(":")
    model = getattr(importlib.import_module(mod), name)
    items = [BulkItem(str(r["id"]), r["prompt"], r) for r in
             (json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip())]

    async def main() -> int:
        session = await open_session(config, start_mcp=not args.no_mcp)
        try:
            kw: dict[str, Any] = {}
            if getattr(args, "max_item_cost", None):
                kw["max_item_cost_usd"] = args.max_item_cost
            runner = BulkRunner(session.rt, session.rt.agents[args.agent], model, Path(args.out),
                                concurrency=args.concurrency, budget_usd=args.budget,
                                prewarm=getattr(args, "prewarm", None), **kw)
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
