"""Massively parallel, schema-validated, resumable per-item agent runs.

The paper dispatched 37,075 clinical trialist agents in parallel — one per NCT
ID — each using its full context window on a single trial and returning JSON
validated against a Pydantic schema (Methods, "Benefits of multi-agent
architecture"). ``BulkRunner`` generalises that pattern:

* one fresh agent per item (isolated context), bounded concurrency;
* a ``submit_result`` tool whose input schema *is* the Pydantic model, so any
  provider with tool calling returns structured output; validation errors go
  back to the agent to fix;
* results appended to JSONL as they finish (crash-safe, resumable: finished
  IDs are skipped on restart), with per-item cost, retries and a budget cap.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ValidationError

from .agents import AgentDefinition
from .runtime import Runtime
from .tools.base import Tool, ToolContext, ToolFailure, inline_refs


@dataclass
class BulkItem:
    id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BulkStats:
    done: int = 0
    failed: int = 0
    skipped: int = 0
    costs: list[float] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def summary(self) -> dict[str, Any]:
        el = time.time() - self.started
        return {
            "completed": self.done, "failed": self.failed, "skipped_existing": self.skipped,
            "total_cost_usd": round(sum(self.costs), 4),
            "median_cost_usd": round(statistics.median(self.costs), 4) if self.costs else None,
            "elapsed_s": round(el, 1),
            "items_per_hour": round(3600 * (self.done + self.failed) / el, 1) if el > 0 else None,
        }


def _submit_tool(model: type[BaseModel], sink: dict[str, Any]) -> Tool:
    schema = inline_refs(model.model_json_schema())

    def handler(ctx: ToolContext, args: dict[str, Any]) -> str:
        try:
            obj = model.model_validate(args)
        except ValidationError as exc:
            raise ToolFailure(f"schema validation failed; fix and resubmit:\n{exc}")
        sink["result"] = obj.model_dump(mode="json")
        return "Result recorded and validated. You are done: reply with one short sentence, no more tool calls."

    return Tool("submit_result",
                "Submit your final structured result (validated against the required schema). "
                "Call exactly once, when all fields are filled from evidence.",
                schema, handler, source="bulk")


class BulkRunner:
    def __init__(self, runtime: Runtime, agent: AgentDefinition, output_model: type[BaseModel],
                 out_path: Path, *, concurrency: int = 32, retries: int = 1,
                 budget_usd: float | None = None, on_progress: Callable[[BulkStats], None] | None = None):
        self.rt = runtime
        self.agent = agent
        self.model = output_model
        self.out_path = Path(out_path)
        self.concurrency = concurrency
        self.retries = retries
        self.budget_usd = budget_usd
        self.on_progress = on_progress
        self._write_lock = asyncio.Lock()
        self.stats = BulkStats()

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

    async def _one(self, item: BulkItem) -> dict[str, Any]:
        last_err = ""
        cost = 0.0
        for attempt in range(self.retries + 1):
            sink: dict[str, Any] = {}
            try:
                res = await self.rt.run_agent(self.agent, item.prompt, depth=1,
                                              extra_tools=[_submit_tool(self.model, sink)])
                cost += res.cost_usd
                if "result" in sink:
                    return {"id": item.id, "ok": True, "result": sink["result"], "cost_usd": round(cost, 6),
                            "model_calls": res.model_calls, "attempts": attempt + 1, "meta": item.meta}
                last_err = f"agent finished without submit_result: {res.text[:300]}"
            except Exception as exc:  # noqa: BLE001 - one bad item must not stop the batch
                last_err = f"{type(exc).__name__}: {exc}"
        return {"id": item.id, "ok": False, "error": last_err, "cost_usd": round(cost, 6), "meta": item.meta}

    async def run(self, items: Iterable[BulkItem]) -> dict[str, Any]:
        done_ids = self.completed_ids()
        queue: asyncio.Queue[BulkItem | None] = asyncio.Queue()
        n = 0
        for it in items:
            if it.id in done_ids:
                self.stats.skipped += 1
                continue
            queue.put_nowait(it)
            n += 1
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        stop = asyncio.Event()

        async def worker() -> None:
            while not stop.is_set():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                rec = await self._one(item)
                async with self._write_lock:
                    with open(self.out_path, "a") as f:
                        f.write(json.dumps(rec, default=str) + "\n")
                    self.stats.costs.append(rec["cost_usd"])
                    if rec["ok"]:
                        self.stats.done += 1
                    else:
                        self.stats.failed += 1
                    if self.on_progress:
                        self.on_progress(self.stats)
                    if self.budget_usd and sum(self.stats.costs) >= self.budget_usd:
                        stop.set()

        await asyncio.gather(*(worker() for _ in range(min(self.concurrency, max(n, 1)))))
        summary = {**self.stats.summary(), "queued": n, "budget_stop": stop.is_set(),
                   "output": str(self.out_path)}
        self.rt.run.trace("bulk_summary", **summary)
        return summary


def load_results(path: Path) -> list[dict[str, Any]]:
    """Latest record per ID (a later success supersedes an earlier failure)."""
    latest: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
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
    p.add_argument("--budget", type=float)
    p.set_defaults(handler=_cli)


def _cli(args, config) -> int:
    import importlib

    from .orchestrator import open_session

    mod, _, name = args.schema.partition(":")
    model = getattr(importlib.import_module(mod), name)
    items = [BulkItem(str(r["id"]), r["prompt"], r) for r in
             (json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip())]

    async def main() -> int:
        session = await open_session(config, start_mcp=not args.no_mcp)
        runner = BulkRunner(session.rt, session.rt.agents[args.agent], model, Path(args.out),
                            concurrency=args.concurrency, budget_usd=args.budget)
        print(json.dumps(await runner.run(items), indent=2))
        await session.close()
        return 0

    return asyncio.run(main())
