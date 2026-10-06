"""CSO tools for massively parallel per-item runs: ``BulkDispatch`` and ``BulkStatus``.

The paper's CSO dispatched one clinical trialist agent per NCT ID (37,075
agents) following a protocol the trialist had designed (Supplementary Text IV).
These tools let the CSO do the same from conversation:

``BulkDispatch``
    Items come from a CSV, parquet or JSONL file under the run directory or a
    read root (optionally filtered with a pandas ``query``) or from an explicit
    ``ids`` list. Each item's prompt is ``prompt_template`` filled with the
    item's fields (``{field}`` placeholders). The agent is ``subagent_type``
    from the roster with an optional ``protocol_path`` (markdown written by a
    specialist) appended to its prompt, and each item must call a terminal
    ``submit_result`` validated against ``schema`` — a registered model name
    (``trial_annotation``) or a JSON Schema file.

    Without ``confirm`` a pilot of ``pilot_size`` items runs now and the tool
    returns the pilot results with a cost and time projection for the full set.
    With ``confirm: true`` (after a pilot of the same job) the full run starts
    as a background task writing ``work/<subagent>/results/bulk/<job_id>.jsonl``
    (pilot items are reused, not rerun) and the tool returns the job id and paths.
    ``budget_usd`` (capped by ``bulk.dispatch_max_budget_usd``) and/or
    ``budget_tokens`` (capped by ``bulk.dispatch_max_budget_tokens``, if set) is
    required; local models cost 0 USD, so their jobs need ``budget_tokens`` (a
    USD-only job is refused: its spend would never reach the cap, and the
    projection's ``fits_budget`` ignores a USD budget there). Both cap the job's
    cumulative spend (every pilot and full run).

``BulkStatus``
    Progress of a job (done/failed/queued, spend) and its summary when finished.

The lead registers both tools only when ``bulk.dispatch_enabled`` is true
(false by default)::

    if bulk_settings(config)["dispatch_enabled"]:
        rt.registry.extend(bulk_dispatch_tools(rt))
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib
import json
import math
import statistics
import string
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, create_model

from .bulk import BulkItem, BulkRunner, bulk_settings, load_results, unpriced_provider, usd_only_budget_problem
from .tools.base import Tool, ToolContext, ToolFailure

__all__ = ["bulk_dispatch_tools", "SCHEMA_REGISTRY", "resolve_schema", "load_items", "BulkJob"]

#: Registered output models for ``schema``: name -> "module:Class".
SCHEMA_REGISTRY: dict[str, str] = {
    "trial_annotation": "vbt.case_studies.trial_outcomes.schema:TrialAnnotation",
}

_JSON_TYPES: dict[str, Any] = {"string": str, "number": float, "integer": int, "boolean": bool,
                               "array": list, "object": dict}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def _jsonschema_model(schema: dict[str, Any], name: str = "BulkResult") -> type[BaseModel]:
    """A model class the bulk ``submit_result`` tool can use for a JSON Schema.

    With ``jsonschema`` installed the schema is enforced exactly; otherwise a
    pydantic model is generated from the schema's top-level properties.
    """
    try:
        import jsonschema  # type: ignore
    except ImportError:
        jsonschema = None  # type: ignore[assignment]
    if jsonschema is None:
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        fields: dict[str, Any] = {}
        for k, spec in props.items():
            t = _JSON_TYPES.get((spec or {}).get("type") if isinstance((spec or {}).get("type"), str) else "", Any)
            fields[k] = (t, ...) if k in required else (Optional[t], None)
        return create_model(name, __config__=ConfigDict(extra="allow"), **fields)  # type: ignore[call-overload]

    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)
    frozen = json.loads(json.dumps(schema))

    class _JsonSchemaResult(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
            return json.loads(json.dumps(frozen))

        @classmethod
        def model_validate(cls, obj: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
            errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
            if errors:
                msg = "\n".join(f"- {'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:20])
                raise ValueError(f"{len(errors)} JSON Schema violation(s):\n{msg}")
            return cls(**obj) if isinstance(obj, dict) else cls()

    _JsonSchemaResult.__name__ = name
    return _JsonSchemaResult


def resolve_schema(spec: str, *, base: Path | None = None) -> type[BaseModel]:
    """Registered name, ``module:Class`` dotted path, or a JSON Schema file."""
    if spec in SCHEMA_REGISTRY:
        spec = SCHEMA_REGISTRY[spec]
    p = Path(spec)
    if not p.is_absolute() and base is not None:
        p = base / p
    if p.suffix == ".json" and p.exists():
        return _jsonschema_model(json.loads(p.read_text()), name=p.stem.title().replace("_", ""))
    if ":" in spec:
        mod, _, cls = spec.partition(":")
        try:
            model = getattr(importlib.import_module(mod), cls)
        except (ImportError, AttributeError) as exc:
            raise ToolFailure(f"schema {spec!r} not importable: {exc}") from exc
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise ToolFailure(f"schema {spec!r} is not a pydantic model")
        return model
    raise ToolFailure(f"unknown schema {spec!r}: use one of {sorted(SCHEMA_REGISTRY)} or a JSON Schema .json file")


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def load_items(path: Path | None, *, ids: list[str] | None = None, query: str | None = None,
               id_column: str | None = None, max_items: int | None = None) -> tuple[list[dict[str, Any]], str]:
    """Rows (dicts) and the id column, from a table file and/or an id list."""
    import pandas as pd

    if path is None:
        if not ids:
            raise ToolFailure("give items_path or ids")
        df = pd.DataFrame({"id": [str(i) for i in ids]})
    else:
        suf = path.suffix.lower()
        if suf == ".csv":
            df = pd.read_csv(path)
        elif suf in (".tsv", ".tab"):
            df = pd.read_csv(path, sep="\t")
        elif suf in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        elif suf in (".jsonl", ".ndjson"):
            df = pd.read_json(path, lines=True)
        else:
            raise ToolFailure(f"items_path must be .csv, .tsv, .parquet or .jsonl, got {path.name}")
        if query:
            try:
                df = df.query(query)
            except Exception as exc:  # noqa: BLE001
                raise ToolFailure(f"query {query!r} failed: {type(exc).__name__}: {exc}") from exc
    col = id_column or next((c for c in ("id", "nct_id", "item_id") if c in df.columns), None)
    if col is None or col not in df.columns:
        raise ToolFailure(f"no id column (pass id_column); columns: {list(df.columns)[:30]}")
    if path is not None and ids:
        df = df[df[col].astype(str).isin(set(map(str, ids)))]
    df = df.drop_duplicates(subset=[col])
    if max_items:
        df = df.head(int(max_items))
    rows = [{k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v) for k, v in r.items()}
            for r in df.to_dict(orient="records")]
    return rows, col


class _Fields(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def render_prompt(template: str, row: dict[str, Any]) -> str:
    try:
        return string.Formatter().vformat(template, (), _Fields({k: ("" if v is None else v) for k, v in row.items()}))
    except KeyError as exc:
        raise ToolFailure(f"prompt_template placeholder {exc} is not an item field; fields: {sorted(row)}") from exc
    except (ValueError, IndexError) as exc:
        raise ToolFailure(f"bad prompt_template: {exc} (escape literal braces as {{{{ }}}})") from exc


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BulkJob:
    job_id: str
    agent: str
    out_path: Path
    summary_path: Path
    n_items: int
    budget_usd: float | None
    state: str = "pending"           # pending | piloted | running | completed | stopped | failed | cancelled
    runner: BulkRunner | None = None
    task: asyncio.Task | None = None
    pilot: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None
    started: float = dataclasses.field(default_factory=time.time)
    #: Cumulative spend of every finished pilot and full run of this job (the
    #: run in progress, if any, is added by :meth:`total_spent`). Persisted in
    #: ``summary_path`` so a resumed job cannot reset it.
    spent_usd: float = 0.0
    budget_tokens: float | None = None
    spent_tokens: float = 0.0         # cumulative budget tokens, like spent_usd

    def total_spent(self) -> float:
        r = self.runner
        live = r.scope.spent if (self.state == "running" and r is not None and r.scope is not None) else 0.0
        return self.spent_usd + live

    def total_tokens(self) -> float:
        r = self.runner
        live = r.scope.tokens if (self.state == "running" and r is not None and r.scope is not None) else 0.0
        return self.spent_tokens + live

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"job_id": self.job_id, "state": self.state, "agent": self.agent,
                               "n_items": self.n_items, "budget_usd": self.budget_usd,
                               "spent_usd": round(self.total_spent(), 6),
                               "budget_tokens": self.budget_tokens, "spent_tokens": round(self.total_tokens()),
                               "results_path": str(self.out_path), "summary_path": str(self.summary_path)}
        r = self.runner
        if r is not None:
            out["progress"] = {"done": r.stats.done, "failed": r.stats.failed, "skipped_existing": r.stats.skipped,
                               "queued": r.queued, "spent_usd": round(r.scope.spent, 4) if r.scope else 0.0,
                               "spent_tokens": round(r.scope.tokens) if r.scope else 0,
                               "elapsed_s": round(time.time() - self.started, 1)}
        if self.summary is not None:
            out["summary"] = self.summary
        if self.error:
            out["error"] = self.error
        return out

    def persist(self) -> None:
        try:
            self.summary_path.write_text(json.dumps(self.status(), indent=2, default=str))
        except Exception:  # noqa: BLE001
            pass

    def add_spend(self, runner: BulkRunner) -> None:
        """Fold a finished pilot/full run's spend into the job total."""
        if runner.scope is not None:
            self.spent_usd += float(runner.scope.spent or 0.0)
            self.spent_tokens += float(runner.scope.tokens or 0.0)


def _persisted_spend(summary_path: Path, key: str = "spent_usd") -> float:
    try:
        return float(json.loads(summary_path.read_text()).get(key) or 0.0)
    except Exception:  # noqa: BLE001 - missing or unreadable summary: nothing spent yet
        return 0.0


def _positive_number(value: Any, name: str) -> float | None:
    """None when absent; a positive float, else ToolFailure."""
    if value in (None, ""):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ToolFailure(f"{name} must be a positive number") from None
    if v <= 0:
        raise ToolFailure(f"{name} must be positive")
    return v


def _jobs(runtime: Any) -> dict[str, BulkJob]:
    jobs = getattr(runtime, "_bulk_jobs", None)
    if jobs is None:
        jobs = {}
        runtime._bulk_jobs = jobs
    return jobs


def _within(p: Path, roots: list[Path]) -> bool:
    for r in roots:
        try:
            p.relative_to(r)
            return True
        except ValueError:
            continue
    return False


def _resolve_input(ctx: ToolContext, raw: str, what: str) -> Path:
    rt = ctx.runtime
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        cands = [ctx.workspace / p, ctx.run.dir / p]
        p = next((c for c in cands if c.exists()), cands[0])
    p = p.resolve()
    roots = [Path(ctx.run.dir).resolve(), *[Path(r).resolve() for r in getattr(rt, "read_roots", []) or []]]
    if not _within(p, roots):
        raise ToolFailure(f"{what} must be under the run directory or a read root; got {p}")
    if not p.exists():
        raise ToolFailure(f"{what} not found: {p}")
    return p


_DISPATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subagent_type": {"type": "string", "description": "Roster agent that handles one item each."},
        "items_path": {"type": "string",
                       "description": "CSV/TSV/parquet/JSONL table of items (run dir or read roots)."},
        "query": {"type": "string", "description": "Optional pandas DataFrame.query filter on items_path."},
        "ids": {"type": "array", "items": {"type": "string"},
                "description": "Explicit item ids (alone, or to subset items_path)."},
        "id_column": {"type": "string", "description": "Id column (default: id, nct_id or item_id)."},
        "max_items": {"type": "integer", "description": "Cap on the number of items."},
        "prompt_template": {"type": "string",
                            "description": "Per-item prompt with {field} placeholders from the item row."},
        "protocol_path": {"type": "string",
                          "description": "Markdown protocol (e.g. written by a specialist) appended to the "
                                         "agent's prompt."},
        "schema": {"type": "string",
                   "description": "Output schema: a registered name (trial_annotation) or a JSON Schema file."},
        "concurrency": {"type": "integer", "description": "Parallel agents (default 16)."},
        "budget_usd": {"type": "number", "description": "Spend cap in USD for this job (pilot included); this "
                                                        "or budget_tokens is required (budget_tokens with a "
                                                        "local model, which costs 0 USD)."},
        "budget_tokens": {"type": "number",
                          "description": "Token cap for this job (pilot included; input + output tokens, cached "
                                         "input at a reduced weight). Use it with a local model (0 USD)."},
        "pilot_size": {"type": "integer", "description": "Items in the pilot (default bulk.pilot_size)."},
        "confirm": {"type": "boolean",
                    "description": "false (default): run the pilot and return a projection. true: start the "
                                   "full background run (after a pilot of the same job)."},
        "job_id": {"type": "string", "description": "Optional; defaults to a hash of the job definition."},
    },
    "required": ["subagent_type", "prompt_template", "schema"],
}


def bulk_dispatch_tools(runtime: Any) -> list[Tool]:
    """``[BulkDispatch, BulkStatus]`` bound to ``runtime`` (register for the CSO when
    ``bulk.dispatch_enabled`` is true)."""

    async def dispatch(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
        rt = runtime
        cfg = bulk_settings(rt.config)
        sub = str(a.get("subagent_type") or "")
        base = rt.agents.get(sub)
        if base is None:
            raise ToolFailure(f"unknown subagent_type {sub!r}; roster: {sorted(rt.agents)}")
        budget_usd = _positive_number(a.get("budget_usd"), "budget_usd")
        budget_tokens = _positive_number(a.get("budget_tokens"), "budget_tokens")
        unpriced = unpriced_provider(getattr(rt, "provider", None), rt.config)
        if budget_usd is None and budget_tokens is None:
            raise ToolFailure("budget_tokens is required with a local model (0 USD); budget_usd with a priced "
                              "provider" if unpriced else
                              "budget_usd is required (a positive number); with a local model (0 USD) pass "
                              "budget_tokens instead")
        problem = usd_only_budget_problem(budget_usd, budget_tokens, provider=getattr(rt, "provider", None),
                                          config=rt.config)
        if problem:  # a 0-USD model never reaches a USD cap: the job would run unbounded
            raise ToolFailure(problem)
        cap = float(cfg.get("dispatch_max_budget_usd") or 0) or None
        if budget_usd is not None and cap is not None and budget_usd > cap:
            raise ToolFailure(f"budget_usd {budget_usd} exceeds bulk.dispatch_max_budget_usd={cap}")
        tcap = float(cfg.get("dispatch_max_budget_tokens") or 0) or None
        if budget_tokens is not None and tcap is not None and budget_tokens > tcap:
            raise ToolFailure(f"budget_tokens {budget_tokens:g} exceeds bulk.dispatch_max_budget_tokens={tcap:g}")
        items_path = _resolve_input(ctx, a["items_path"], "items_path") if a.get("items_path") else None
        rows, id_col = load_items(items_path, ids=a.get("ids"), query=a.get("query"),
                                  id_column=a.get("id_column"), max_items=a.get("max_items"))
        if not rows:
            raise ToolFailure("no items selected")
        template = str(a["prompt_template"])
        items = [BulkItem(str(r[id_col]), render_prompt(template, r), {"row_id": str(r[id_col])}) for r in rows]
        protocol = ""
        protocol_path = None
        if a.get("protocol_path"):
            protocol_path = _resolve_input(ctx, a["protocol_path"], "protocol_path")
            protocol = protocol_path.read_text()
        model = resolve_schema(str(a["schema"]), base=ctx.workspace)
        agent = dataclasses.replace(
            base, name=f"{base.name}-bulk", memory="none", can_delegate=False,
            prompt=base.prompt + (f"\n\n## Protocol (from {protocol_path.name})\n\n{protocol}" if protocol else "")
            + "\n\nYou handle exactly one item. When every field is filled from evidence, call "
              "`submit_result` once; that ends your task.",
            tools=[t for t in base.tools if t not in ("Task", "BulkDispatch", "BulkStatus")])
        key = json.dumps({"sub": sub, "items": a.get("items_path"), "q": a.get("query"), "ids": a.get("ids"),
                          "max": a.get("max_items"), "tpl": template, "schema": a.get("schema"),
                          "protocol": a.get("protocol_path")}, sort_keys=True)
        job_id = str(a.get("job_id") or f"{sub}-{hashlib.sha256(key.encode()).hexdigest()[:10]}")
        out_dir = rt.run.agent_dir(sub) / "results" / "bulk"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{job_id}.jsonl"
        summary_path = out_dir / f"{job_id}.summary.json"
        jobs = _jobs(rt)
        job = jobs.get(job_id)
        if job is not None and job.state == "running":
            raise ToolFailure(f"job {job_id} is already running; use BulkStatus")
        if job is None:
            job = BulkJob(job_id, agent.name, out_path, summary_path, len(items), budget_usd,
                          spent_usd=_persisted_spend(summary_path),
                          spent_tokens=_persisted_spend(summary_path, "spent_tokens"))
            jobs[job_id] = job
        job.n_items, job.budget_usd, job.budget_tokens = len(items), budget_usd, budget_tokens
        concurrency = int(a.get("concurrency") or 16)
        pilot_size = int(a["pilot_size"]) if a.get("pilot_size") is not None else int(cfg.get("pilot_size") or 5)
        # the budgets cap the job's cumulative spend: every pilot and every full run counts against them
        remaining_budget = None if budget_usd is None else budget_usd - job.spent_usd
        if remaining_budget is not None and remaining_budget <= 0:
            raise ToolFailure(f"job {job_id} already spent ${job.spent_usd:.2f} of budget_usd={budget_usd}; "
                              "raise budget_usd (within bulk.dispatch_max_budget_usd) to continue")
        remaining_tokens = None if budget_tokens is None else budget_tokens - job.spent_tokens
        if remaining_tokens is not None and remaining_tokens <= 0:
            raise ToolFailure(f"job {job_id} already used {job.spent_tokens:,.0f} of budget_tokens="
                              f"{budget_tokens:,.0f}; raise budget_tokens to continue")

        if not a.get("confirm"):
            pilot_items = items[:max(1, pilot_size)]
            pilot_conc = max(1, min(concurrency, len(pilot_items)))
            runner = BulkRunner(rt, agent, model, out_path, concurrency=pilot_conc,
                                budget_usd=remaining_budget, budget_tokens=remaining_tokens, job_id=job_id)
            t0 = time.time()
            try:
                summary = await runner.run(pilot_items)
            finally:
                job.add_spend(runner)
                job.persist()
            elapsed = time.time() - t0
            results = [r for r in load_results(out_path) if r["id"] in {i.id for i in pilot_items}]
            ok_costs = [float(r.get("cost_usd") or 0) for r in results if r.get("ok")]
            all_costs = [float(r.get("cost_usd") or 0) for r in results]
            per_item = (sum(all_costs) / len(all_costs)) if all_costs else None
            done_ids = {r["id"] for r in results if r.get("ok")}
            remaining = sum(1 for i in items if i.id not in done_ids)
            ran = max(1, summary.get("completed", 0) + summary.get("failed", 0))
            # per-item latency: median of the pilot records' own elapsed_s (the pilot ran in parallel,
            # so elapsed/ran is already divided by its parallelism); fall back to undoing that division
            lat = [float(r["elapsed_s"]) for r in results if r.get("elapsed_s") is not None]
            latency = statistics.median(lat) if lat else elapsed / ran * pilot_conc
            wall_s = math.ceil(remaining / max(1, concurrency)) * latency
            left = None if budget_usd is None else budget_usd - job.spent_usd
            all_tokens = [float(r.get("budget_tokens") or 0) for r in results]
            tok_item = (sum(all_tokens) / len(all_tokens)) if all_tokens else None
            left_tokens = None if budget_tokens is None else budget_tokens - job.spent_tokens
            fits = []
            # a 0-USD model gives no cost signal: its USD budget says nothing about fitting
            if left is not None and per_item is not None and not unpriced:
                fits.append(per_item * remaining <= left)
            if left_tokens is not None and tok_item is not None:
                fits.append(tok_item * remaining <= left_tokens)
            projection = {
                "n_items": len(items), "n_remaining": remaining,
                "pilot_cost_usd": round(sum(all_costs), 4),
                "mean_cost_per_item_usd": None if per_item is None else round(per_item, 4),
                "median_cost_per_success_usd": summary.get("median_cost_usd"),
                "projected_remaining_cost_usd": None if per_item is None else round(per_item * remaining, 2),
                "projected_total_cost_usd": None if per_item is None else round(per_item * len(items), 2),
                "pilot_tokens": round(sum(all_tokens)),
                "mean_tokens_per_item": None if tok_item is None else round(tok_item),
                "projected_remaining_tokens": None if tok_item is None else round(tok_item * remaining),
                "projected_wall_time_s": round(wall_s, 1),
                "budget_usd": budget_usd, "job_spent_usd": round(job.spent_usd, 6),
                "budget_remaining_usd": None if left is None else round(left, 6),
                "budget_tokens": budget_tokens, "job_spent_tokens": round(job.spent_tokens),
                "budget_remaining_tokens": None if left_tokens is None else round(left_tokens),
                "fits_budget": all(fits) if fits else None,
                "pilot_success_rate": round(len(ok_costs) / len(results), 3) if results else None,
            }
            job.state, job.pilot = "piloted", {"summary": summary, "projection": projection}
            job.persist()
            return {"job_id": job_id, "state": "piloted", "pilot_results": results[:max(1, pilot_size)],
                    "pilot_summary": summary, "projection": projection, "results_path": str(out_path),
                    "next": "Review the pilot. To run every item call BulkDispatch again with the same "
                            "arguments and confirm: true (pilot items are reused)."}

        if pilot_size > 0 and job.pilot is None:
            raise ToolFailure("run a pilot first: call BulkDispatch with confirm: false, review the projection, "
                              "then confirm (or pass pilot_size: 0 to skip the pilot)")
        runner = BulkRunner(rt, agent, model, out_path, concurrency=concurrency, budget_usd=remaining_budget,
                            budget_tokens=remaining_tokens, job_id=job_id, account_to_parent=False)
        job.runner, job.state, job.summary, job.error, job.started = runner, "running", None, None, time.time()

        async def background() -> None:
            try:
                s = await runner.run(items)
                job.summary = s
                job.state = "stopped" if s.get("budget_stop") else "completed"
            except asyncio.CancelledError:
                job.state = "cancelled"
                raise
            except BaseException as exc:  # noqa: BLE001
                job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
                job.summary = getattr(exc, "bulk_summary", None)
            finally:
                job.add_spend(runner)
                if job.state == "running":  # cancelled/failed before a terminal state was set
                    job.state = "cancelled"
                job.persist()
                rt.emit("bulk_end", job_id=job_id, agent=agent.name, state=job.state, error=job.error,
                        summary=job.summary)

        job.task = asyncio.create_task(background())
        track = getattr(rt, "_track", None)
        if callable(track):
            track(job.task)
        rt.emit("bulk_start", job_id=job_id, agent=agent.name, n_items=len(items), budget_usd=budget_usd,
                budget_tokens=budget_tokens)
        return {"job_id": job_id, "state": "running", "n_items": len(items), "budget_usd": budget_usd,
                "budget_tokens": budget_tokens, "results_path": str(out_path), "summary_path": str(summary_path),
                "next": "Follow progress with BulkStatus(job_id)."}

    def status(ctx: ToolContext, a: dict[str, Any]) -> dict[str, Any]:
        jobs = _jobs(runtime)
        jid = a.get("job_id")
        if not jid:
            return {"jobs": [j.status() for j in jobs.values()]}
        job = jobs.get(str(jid))
        if job is None:
            for p in runtime.run.dir.glob(f"work/*/results/bulk/{jid}.summary.json"):
                return json.loads(p.read_text())
            raise ToolFailure(f"unknown job_id {jid!r}; known: {sorted(jobs)}")
        return job.status()

    return [
        Tool("BulkDispatch",
             "Massively parallel per-item run (one fresh agent per item, schema-validated JSON results). "
             "Without confirm: runs a pilot and returns results plus a cost/time projection. With confirm: "
             "starts the full run in the background. budget_usd (or, with a local model, budget_tokens) is "
             "required.",
             _DISPATCH_SCHEMA, dispatch, source="harness"),
        Tool("BulkStatus", "Progress and summary of a BulkDispatch job (omit job_id to list jobs).",
             {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": []},
             status, source="harness"),
    ]
