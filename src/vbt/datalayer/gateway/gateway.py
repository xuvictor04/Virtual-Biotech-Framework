"""The data gateway in front of every MCP call (§11; implements :class:`~vbt.datalayer.api.GatewayProtocol`).
No pyarrow: every data read goes through the data child (:mod:`.service_client`).

``prepare(server, tool, args, ctx)`` runs §11.3 in order: contract lookup (a selector chooses the
bound table), block, call-scoped readiness (an accepted kind whose resolver index is not ready
is dropped with a note), argument contracts, identifier resolution (existence modes; ``bound``
costs one witness count, ``unknown`` proceeds with a note), the witness pre-scan, the
scope-completeness rule, leakage, memory admission (upstream route only) and limit inflation,
then the route (``upstream`` | ``derived`` | ``none``).

``finish(plan, raw)`` runs §11.4: classification, rows through the field map, transforms T1-T6,
witness checks W1-W6 (including one re-call of a short page), T7-T14, sections and files,
status and coverage (§6.3), the ``_vbt`` header and the ``vbt.dataprov/1`` record. A
contradiction is repaired from the data child when the binding declares ``derived`` and
``on_contradiction: derived`` (``served_by: repaired``), otherwise it is a ``tool_defect``.

Modes: ``observe`` never raises and returns upstream results unchanged, tracing
``data_observe`` events with the decision it would have taken; ``enforce`` applies everything.
Profile ``fidelity`` serves derived tools ``pass`` behind the witness and refuses only on a
contradiction (repairs become errors).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...config import base_tool_env
from ..api import CallPlan, CrashDecision, LaunchSpec, ListingDecision, RawResult
from ..catalog import Catalog, CatalogError, ToolContract, build_catalog
from ..descriptor.columns import is_container
from ..descriptor.load import variables_from_config
from ..errors import (
    ErrorKind,
    GatewayError,
    json_value,
    not_found_payload,
    not_ready_payload,
    tool_defect_payload,
    too_large_payload,
    unsupported_combination_payload,
)
from ..ipc import RankKeyModel, ServeRequest, ServeResponse, WitnessRequest, WitnessResponse
from ..launch import build_launch_spec
from ..memory import AdmissionController, MemoryEstimator, ResidencyLedger, TableRead, crash_decision, read_status
from ..predicate import And, Eq, IsNull, Not, Predicate, map_columns, to_json
from ..record import (
    DataProvenance,
    OrderInfo,
    RequestInfo,
    ResultInfo,
    SourceInfo,
    TableInfo,
    UpstreamInfo,
)
from ..resolve import IndexMissing, IndexStore, Resolver, ResolverConfigError, error_for, list_error
from ..result import DataResult, Header
from ..rowkey import canonical
from ..settings import DataSettings
from . import soma_filter
from .classify import classify
from .contracts import (
    PreparedArgs,
    VocabSnapshot,
    apply_arg_contracts,
    bound_column,
    build_predicate,
    column_spec,
    is_present,
    vocab_key,
)
from .fields import FieldMapper, extract_rows, get_path, jp_first, jp_get, jp_set, jp_test, parse_payload, place_rows
from .files import MaterializedRegistry, reconcile
from .leakage import LeakagePlan, ceiling_of, leakage_record, prepare_leakage
from .readiness import ReadinessCache, call_readiness, degraded_tools, tables_read
from .scope import ScopeDecision, grain_columns, scope_completeness
from .service_client import DATA_SERVER, ServiceClient, ServiceError
from .transforms import (
    Counters,
    honour_arguments,
    on_item_rows,
    rank_keys,
    row_key,
    split_container,
    t1_leakage,
    t2_unknowns,
    t3_existence,
    t6_negation,
    t7_duplicates,
    t8_pooled_over,
    t9_levels,
    t10_order_cut,
    t11_counts,
    t12_trim,
    t13_flag_partition,
    t14_validity,
)

__all__ = ["DataGateway", "build_gateway", "GATEWAY_VERSION"]

log = logging.getLogger(__name__)

GATEWAY_VERSION = "1.0"
_STATE = "_vbt_state"
_INCLUDE_NEGATED = "include_negated"
_INCLUDE_DUPLICATES = "include_duplicates"
_SERVICE_SCRIPT = ("src", "vbt", "datalayer", "service", "server.py")


@dataclass
class _CallState:
    """Everything one call carries from ``prepare`` to ``finish``."""

    name: str
    t0: float
    mode: str = "enforce"
    generic: bool = False
    unguarded: bool = False                       # no overlay at all for this server: legacy semantics
    lenient: bool = False
    prepared: PreparedArgs | None = None
    values: dict[str, Any] = field(default_factory=dict)          # resolved values (predicate inputs)
    results: dict[str, Any] = field(default_factory=dict)         # ResolutionResult per argument
    resolved: dict[str, str] = field(default_factory=dict)        # _vbt.resolved
    resolution_summary: dict[str, Any] = field(default_factory=dict)
    per_arg: dict[str, Predicate] = field(default_factory=dict)
    predicate: Predicate | None = None
    search_text: str | None = None                                  # derived search: the free-text argument
    witness: WitnessResponse | None = None
    witness_reason: str | None = None
    decision: ScopeDecision | None = None
    leakage: LeakagePlan | None = None
    admission: Any = None
    requested_limit: int | None = None
    sent_limit: int | None = None
    inflated: bool = False
    order: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    checks: list[tuple[str, bool | None, str | None]] = field(default_factory=list)
    transforms: list[str] = field(default_factory=list)
    undefined: dict[str, Any] = field(default_factory=dict)
    anchors: dict[str, tuple[str, Any]] = field(default_factory=dict)
    auto_fixed: dict[str, Any] = field(default_factory=dict)
    soft_sections: dict[str, str] = field(default_factory=dict)
    storage_types: dict[str, str | None] = field(default_factory=dict)
    attempts: int = 0
    t_ms: dict[str, float] = field(default_factory=dict)
    t_prepared: float | None = None                                 # end of prepare (the upstream call starts)
    t_finish: float | None = None                                   # start of finish
    soma_added: bool = False
    soma_predicate: Predicate | None = None
    resolved_columns: dict[str, str] = field(default_factory=dict)   # identifier argument -> bound column
    force_partial: bool = False
    in_universe: bool | None = None
    null_container: bool = False                                    # the rows' list is null (not assessed)
    derived_withheld: int = 0                                       # derived rows dropped by T4/T6 in the gateway
    engine_matched: bool = False                                    # the source's engine matches an argument
    order_verified: bool | None = None
    not_found_items: list[Any] | None = None
    short_page: str | None = None
    serve_excluded_unknown: dict[str, int] = field(default_factory=dict)
    serve_negated: int = 0                                          # groups the data child dropped as negated
    materialized: list[dict[str, Any]] = field(default_factory=list)


def _state(plan: CallPlan) -> _CallState:
    st = getattr(plan, _STATE, None)
    if st is None:
        st = _CallState(name=f"mcp__{plan.server}__{plan.tool}", t0=time.monotonic())
        setattr(plan, _STATE, st)
    return st


def _ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000.0, 1)


def _last(path: str) -> str:
    return str(path).lstrip("/").split(".")[-1].replace("[]", "")


class DataGateway:
    """The harness-side gateway (see the module docstring)."""

    def __init__(self, settings: DataSettings, catalog: Catalog, registry: Any = None, *,
                 service: ServiceClient | None = None, index_store: IndexStore | None = None,
                 admission: AdmissionController | None = None, readiness: ReadinessCache | None = None,
                 run: Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.settings = settings
        self.catalog = catalog
        self.registry = registry if registry is not None else getattr(catalog, "registry", None)
        self.config = dict(config or {})
        self.run = dict(run or {})
        self.mode = settings.gateway.mode
        self.profile = settings.gateway.profile
        self.service = service or ServiceClient()
        self.index_store = index_store or IndexStore(settings.cache_dir)
        # measured calibrations and feedback under cache_dir sharpen the estimates; commits record feedback
        self.admission = admission or AdmissionController(
            settings, MemoryEstimator.from_settings(settings, load_calibrations=True), ResidencyLedger(),
            feedback_dir=settings.cache_dir)
        self.readiness = readiness or ReadinessCache(settings.cache_dir, catalog, self.registry)
        self.readiness.load()
        self.resolver = Resolver(self.registry, catalog, self._index_provider, remote=self._remote, settings=settings)
        self.bridge: Any = None
        self.materialized = MaterializedRegistry()
        self.observed: list[dict[str, Any]] = []
        self._schemas: dict[tuple[str, str], dict[str, Any]] = {}
        self._vocab: dict[str, Any] = {}
        self._stats: dict[str, Any] = {}
        self._index_fp: dict[str, str] = {}
        self._index_failed: dict[str, str] = {}
        self._soma_vocab = soma_filter.VocabCache(settings.resolution.remote_ttl_s)
        self._check_task: asyncio.Task[Any] | None = None
        self._readiness_supplied = False
        self._upstream_commit: str | None = None
        self._commit_warned: set[str] = set()
        self._ceiling = ceiling_of(settings)

    # ================================================================== wiring

    def bind_bridge(self, bridge: Any) -> None:
        self.bridge = bridge
        self.service.bind(bridge)
        with contextlib.suppress(Exception):
            self.admission.bind_bridge(bridge)

    def _mode_for(self, server: str) -> str:
        if self.mode == "enforce" and not self.settings.gateway.enforces(server):
            return "observe"
        return self.mode

    def extra_servers(self) -> list[dict[str, Any]]:
        """The data child's server spec (``VBT_DATA_SETTINGS`` in its environment); none when the
        service script is not in this checkout (the gateway then runs without a data child)."""
        variables = variables_from_config(self.config)
        python = variables.get("mcp_python") or variables.get("python") or sys.executable
        script = Path(self.settings.project_root).joinpath(*_SERVICE_SCRIPT)
        if not script.is_file():
            log.warning("data child script %s is missing; data-layer checks run without the data child", script)
            return []
        svc = self.settings.service
        # the harness tool env (data paths such as OPEN_TARGETS_DATA_PATH) expands descriptor roots
        env = {**base_tool_env(dict(self.config or {})), "VBT_DATA_SETTINGS": self.settings.to_json()}
        return [{"name": DATA_SERVER, "command": python, "args": ["-E", str(script)],
                 "env": env, "timeout_s": float(svc.timeout_s),
                 "max_concurrency": int(svc.max_concurrency), "mem_limit_mb": int(svc.mem_limit_mb)}]

    def launch_spec(self, cfg: Any) -> LaunchSpec | None:
        log_root = getattr(self.bridge, "log_root", None) if self.bridge is not None else None
        try:
            log_dir = log_root() if callable(log_root) else log_root
            spec = build_launch_spec(cfg, self.settings, log_dir)
        except Exception:  # noqa: BLE001 - launch unguarded rather than not at all
            log.warning("launch_spec failed for %s", getattr(cfg, "name", cfg), exc_info=True)
            return None
        if spec is not None:
            with contextlib.suppress(Exception):
                self.admission.ledger.set_status_path(str(cfg.name), spec.status_path)
        return spec

    # ================================================================== listing

    def rewrite_listing(self, server: str, tool: str, description: str,
                        input_schema: dict[str, Any]) -> ListingDecision:
        if server == DATA_SERVER and tool.startswith("_"):
            if tool == "_check":
                self._schedule_check()
            return ListingDecision(False, description, input_schema, reason="internal data-child verb")
        self._schemas[(server, tool)] = dict(input_schema or {})
        try:
            contract = self.catalog.contract(server, tool)
        except Exception:  # noqa: BLE001
            return ListingDecision(True, description, input_schema)
        b = contract.binding
        if b is not None and (b.hidden or (b.block is not None and b.block.hidden and b.serve == "block")):
            return ListingDecision(False, description, input_schema, reason="hidden by the overlay")
        if contract.generic or b is None or self._mode_for(server) != "enforce":
            return ListingDecision(True, description, input_schema)
        from ..derive import annotate_schema, describe_tool
        try:
            schema = annotate_schema(contract, input_schema, catalog=self.catalog, registry=self.registry,
                                     vocab=self._vocab, enum_max=self.settings.derive.enum_max)
            unready = degraded_tools(_OneTool(self.catalog, server, tool), self.readiness).get(
                f"mcp__{server}__{tool}") if self.readiness.tables else None
            text = describe_tool(contract, description, catalog=self.catalog,
                                 max_chars=self.settings.derive.description_max_chars, unready=unready)
        except Exception:  # noqa: BLE001 - never lose a tool over its description
            log.warning("deriving the listing of %s.%s failed", server, tool, exc_info=True)
            return ListingDecision(True, description, input_schema)
        return ListingDecision(True, text, schema)

    # ================================================================== readiness

    def _schedule_check(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._readiness_supplied:
            return                                     # preflight's results stand; calls refresh what is unchecked
        tables = self._session_tables()
        if tables and (self._check_task is None or self._check_task.done()):
            self._check_task = loop.create_task(self.refresh_readiness(tables))

    def _session_tables(self) -> list[str]:
        """The tables the started servers' bound tools read: the session never checks a table no tool reads
        (an unbound archive of millions of rows would hold the first call for minutes)."""
        started = {getattr(s, "name", None) for s in getattr(self.bridge, "servers", None) or []}
        out: set[str] = set()
        for server in self.catalog.servers():
            if started and server not in started:
                continue
            for tool in self.catalog.tools(server):
                try:
                    out.update(self.readiness.physical(ref)[0] for ref in self.catalog.contract(server, tool).tables)
                except Exception:  # noqa: BLE001 - a broken contract is reported by lint, not here
                    continue
        return sorted(out)

    async def refresh_readiness(self, tables: Sequence[str] = (), depth: str | None = None) -> bool:
        """Run the data child's ``_check`` (all tables by default) and cache the results."""
        try:
            resp = await self.service.check(list(tables), depth or self.settings.readiness.session_depth)
        except ServiceError as exc:
            log.info("readiness check unavailable: %s", exc.message)
            return False
        self.readiness.load_check_results(resp)
        return True

    async def aclose(self) -> None:
        """Stop background work (the session's readiness check) before the bridge closes."""
        task, self._check_task = self._check_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait({task}, timeout=5.0)

    async def wait_readiness(self, timeout_s: float | None = None) -> bool:
        """Wait for the session's readiness check (the one the listing started, else a new one)."""
        task = self._check_task
        try:
            if task is not None:
                await asyncio.wait_for(asyncio.shield(task), timeout_s)
                return bool(task.result())
            tables = self._session_tables()
            return await asyncio.wait_for(self.refresh_readiness(tables), timeout_s) if tables else True
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - readiness stays unchecked
            return False

    def set_readiness(self, results: Any) -> list[str]:
        """Store ``_check`` results computed elsewhere (preflight). The listing-triggered check would
        repeat that work: it is cancelled, and not scheduled again on a data-child restart."""
        self._readiness_supplied = True
        task, self._check_task = self._check_task, None
        if task is not None and not task.done():
            task.cancel()
        return self.readiness.load_check_results(results)

    def readiness_snapshot(self) -> dict[str, Any]:
        snap = self.readiness.snapshot()
        snap["pending"] = bool(self._check_task is not None and not self._check_task.done())
        snap["service_available"] = self.service.available
        return snap

    def degraded_tools(self) -> dict[str, str]:
        return degraded_tools(self.catalog, self.readiness)

    # ================================================================== resolver plumbing

    def _index_provider(self, source: str, id_type: str) -> Any:
        key = f"{source}:{id_type}"
        fps = [self._index_fp.get(key)]
        entry = self.readiness.indexes.get(key) or {}
        fps.append(entry.get("fingerprint"))
        for fp in fps:
            if not fp:
                continue
            try:
                return self.index_store.load(source, fp, id_type)
            except (IndexMissing, ValueError, OSError):
                continue
        return None

    async def _remote(self, source: str, id_type: str, values: Sequence[str]) -> Any:
        resp = await self.service.resolve_remote(source, id_type, values)
        return resp.model_dump()

    def _index_needed(self, qualified: str, seen: set[str] | None = None) -> list[str]:
        """The id_types whose local index ``qualified`` resolves through."""
        seen = seen or set()
        if qualified in seen:
            return []
        seen.add(qualified)
        try:
            src, spec = self.catalog.id_type(qualified)
        except Exception:  # noqa: BLE001
            return []
        out: list[str] = []
        if spec.index == "remote" or spec.universe_via is not None:
            return out
        if spec.universe is not None or spec.resolve_via:
            out.append(f"{src}:{qualified.partition(':')[2] or qualified}")
        if spec.label_of:
            out.extend(self._index_needed(self.catalog.qualify_id_type(spec.label_of, src), seen))
        for m in spec.union:
            out.extend(self._index_needed(self.catalog.qualify_id_type(m, src), seen))
        return out

    async def _ensure_index(self, qualified: str) -> bool:
        """Load (or have the data child build) the sidecar of ``qualified``; False when unavailable."""
        src, _, bare = qualified.partition(":")
        if self._index_provider(src, bare) is not None:
            self.readiness.set_index(qualified, "ready", fingerprint=self._index_fp.get(qualified))
            return True
        if qualified in self._index_failed:
            return False
        try:
            resp = await self.service.build_index(src, bare)
        except ServiceError as exc:
            self._index_failed[qualified] = exc.message
            self.readiness.set_index(qualified, "missing", detail=exc.message)
            return False
        self._index_fp[qualified] = resp.fingerprint
        ok = self._index_provider(src, bare) is not None
        self.readiness.set_index(qualified, "ready" if ok else "missing", fingerprint=resp.fingerprint,
                                 detail="" if ok else f"index file not found at {resp.path}")
        if not ok:
            self._index_failed[qualified] = f"no index at {resp.path}"
        return ok

    # ================================================================== prepare

    async def prepare(self, server: str, tool: str, args: dict[str, Any], ctx: Any) -> CallPlan:
        args = dict(args or {})
        contract = self.catalog.contract(server, tool)
        mode = self._mode_for(server)
        plan = CallPlan(server=server, tool=tool, args_raw=dict(args), args_sent=dict(args), route="upstream",
                        contract=contract, resolutions=[], witness=None, mode=mode)  # type: ignore[arg-type]
        st = _state(plan)
        st.mode = mode
        setattr(plan, "_vbt_ctx", ctx)
        if mode != "enforce":
            try:
                await self._prepare(plan, st)
                self._observe(plan, "would_route", route=plan.route, args_sent=plan.args_sent,
                              resolved=st.resolved, notes=st.notes)
            except GatewayError as exc:
                self._observe(plan, f"would_{exc.kind.value}", error=exc.envelope())
            except Exception as exc:  # noqa: BLE001 - observe mode never changes a call
                self._observe(plan, "observe_failed", error=f"{type(exc).__name__}: {exc}")
            plan.args_sent = self._with_agent(server, tool, dict(args), ctx)
            plan.route = "upstream"
            plan.cold_lock = None
            return plan
        try:
            await self._prepare(plan, st)
        except GatewayError as exc:
            exc.with_tool(st.name)
            raise
        plan.args_sent = self._with_agent(server, tool, plan.args_sent, ctx)
        st.t_ms["prepare"] = _ms(st.t0)
        st.t_prepared = time.monotonic()
        return plan

    @staticmethod
    def _with_agent(server: str, tool: str, args: dict[str, Any], ctx: Any) -> dict[str, Any]:
        """A data-child public verb's payload with the calling agent, so the child applies the tables'
        ``expose.withhold_from`` per call (the agent's own ``agent`` argument never counts)."""
        agent = getattr(ctx, "agent", None)
        if server != DATA_SERVER or tool.startswith("_") or not agent:
            return args
        out = dict(args)
        if set(out) == {"request"} and isinstance(out["request"], Mapping):
            out["request"] = {**out["request"], "agent": str(agent)}
        else:
            out["agent"] = str(agent)
        return out

    async def _prepare(self, plan: CallPlan, st: _CallState) -> None:
        contract: ToolContract = plan.contract
        b = contract.binding
        name = st.name
        if b is None or contract.generic:
            st.generic = True
            st.unguarded = self._unguarded(contract)
            self._generic_lint(plan, st)
            return
        # 1. selector -> bound table
        selected = contract.selected_table(plan.args_raw)
        plan.bound_table = selected
        # 2. block
        if b.serve == "block":
            blk = b.block
            raise GatewayError(ErrorKind.quarantined, blk.reason if blk else "this tool is blocked", tool=name,
                               payload={"reason": blk.reason if blk else None,
                                        "alternatives": list(blk.alternatives) if blk else [],
                                        "until_phase": blk.until_phase if blk else None})
        # 3. readiness
        await self._check_readiness(plan, st, contract, selected)
        # auto-derived gateway-only scope arguments
        from ..derive import auto_scope_args
        schema = self._schemas.get((plan.server, plan.tool))
        args = dict(plan.args_raw)
        for arg, column in auto_scope_args(contract, schema).items():
            if arg in args:
                st.auto_fixed[column] = args.pop(arg)
        # 4. argument contracts
        vocab = await self._vocab_for(contract, args, selected)
        for column, value in list(st.auto_fixed.items()):
            snap = vocab.get(vocab_key(selected or contract.bound_table or "", column))
            st.auto_fixed[column] = self._snap_auto(contract, column, value, snap, st)
        prepared = apply_arg_contracts(contract, args, vocab, fixed_scope=st.auto_fixed, schema=schema,
                                       output_dir=self.run.get("mcp_output_dir"), registry=self.registry,
                                       enum_max=self.settings.derive.enum_max)
        st.prepared = prepared
        st.notes.extend(prepared.notes)
        plan.args_sent = prepared.args_sent
        plan.gateway_args = dict(prepared.gateway_args)
        plan.scope.update(prepared.scope)
        for column, value in st.auto_fixed.items():
            plan.scope[_last(column)] = value
        await self._soma(plan, st, contract)
        # 5. resolution
        await self._resolve(plan, st, contract, prepared, selected)
        # values for predicates: resolved identifiers, contract-checked values, gateway-only values
        values = dict(prepared.values)                 # in the table's terms (before send_map and quoting)
        values.update(st.values)
        for column, value in st.auto_fixed.items():
            st.per_arg[f"@{column}"] = Eq(column, value)
        pred, per_arg = build_predicate(contract, values, selected=selected, registry=self.registry,
                                        confirmed=self._confirmed(selected), fixed_scope=self._fixed(st, prepared),
                                        flags=prepared.flags, defaults=set(prepared.defaults))
        st.per_arg.update(per_arg)
        self._search_text(plan, st, contract)
        conjuncts = list(st.per_arg.values())
        for arg, (column, value) in st.anchors.items():
            conjuncts.append(Not(Eq(column, value)))
        st.predicate = conjuncts[0] if len(conjuncts) == 1 else (And(tuple(conjuncts)) if conjuncts else None)
        self._unique_within(plan, st, contract, selected)
        # route
        derived = b.serve == "derived" and b.derived is not None and self.profile != "fidelity"
        plan.route = "derived" if derived else "upstream"
        if st.undefined:
            plan.route = "none"
            return
        st.order = self._effective_order(contract, prepared)
        st.requested_limit = self._requested_limit(plan, contract, schema)
        # 6. witness pre-scan
        await self._witness(plan, st, contract, selected, derived=derived)
        if st.witness is not None and st.witness.total == 0 and st.witness.total_method != "unknown":
            st.in_universe = await self._in_coverage_universe(st, self._rows_table(plan, contract))
            st.null_container = await self._null_container(st, contract)
        # 7. scope completeness
        inflatable = self._inflatable(plan, st, contract)
        st.decision = scope_completeness(
            contract, plan, st.witness, fixed=self._fixed(st, prepared), order=st.order,
            verb=(b.derived.verb if b.derived is not None and derived else
                  ("count" if b.result.kind == "count" else None)),
            route=plan.route, inflated=inflatable or derived, values=self._dim_values(vocab, contract, selected),
            storage_types=st.storage_types, units=self._units(st),
            vocab_values=self._dim_values(vocab, contract, selected, every=True))
        plan.scope.update(st.decision.disclosure())
        st.notes.extend(st.decision.notes)
        # 8. leakage
        st.leakage = prepare_leakage(contract, plan.args_sent, self._ceiling, tool=name)
        st.notes.extend(st.leakage.notes)
        # 9. admission (upstream only; observe mode reserves nothing and never recycles a server)
        if plan.route == "upstream" and st.mode == "enforce":
            await self._admit(plan, st, contract)
        # 10. limit inflation
        if plan.route == "upstream":
            self._inflate(plan, st, contract, inflatable)

    # ---------------------------------------------------------------- generic

    def _unguarded(self, contract: ToolContract) -> bool:
        """A server with neither an overlay nor a generic overlay: the generic guard (§11.9) has no
        configuration to apply, so results keep the bridge's legacy semantics."""
        return (contract.binding is None and contract.generic_spec is None
                and contract.server not in self.catalog.overlays)

    def _generic_lint(self, plan: CallPlan, st: _CallState) -> None:
        spec = plan.contract.generic_spec
        if spec is None or not spec.param_kinds or self.registry is None:
            return
        for arg, value in plan.args_raw.items():
            if not isinstance(value, str):
                continue
            declared = [k for glob, kinds in spec.param_kinds.items() if fnmatch(arg, glob) for k in kinds]
            if not declared:
                continue
            for p in self.registry.all("identifier"):
                try:
                    score = float(p.looks_like(value))
                except Exception:  # noqa: BLE001
                    continue
                if score >= 0.8 and p.id_type not in declared:
                    st.notes.append(f"{arg}={value!r} looks like {p.id_type}; the parameter suggests "
                                    f"{', '.join(declared)}")
                    break

    # ---------------------------------------------------------------- readiness

    def _partition_values(self, contract: ToolContract, args: Mapping[str, Any]) -> dict[str, dict[str, list[Any]]]:
        out: dict[str, dict[str, list[Any]]] = {}
        for name, a in contract.args.items():
            v = args.get(name)
            if v is None or a.op not in ("eq", "in"):
                continue
            for table, column in contract.arg_columns(name):
                t = contract.tables.get(table)
                if t is not None and column in t.physical_spec.partitions:
                    out.setdefault(table, {})[column] = list(v) if isinstance(v, list) else [v]
        return out

    async def _check_readiness(self, plan: CallPlan, st: _CallState, contract: ToolContract,
                               selected: str | None) -> None:
        tables = tables_read(contract, selected, plan.args_raw)
        self.readiness.shallow_refresh(tables)
        parts = self._partition_values(contract, plan.args_raw)
        r = call_readiness(contract, self.readiness, bound_table=selected, args=plan.args_raw, partition_values=parts)
        if r.unchecked and st.mode != "enforce":
            # observe mode adds no latency: unchecked tables are noted, never waited for
            st.lenient = True
            st.notes.append("observe mode: readiness of " + ", ".join(r.unchecked) + " not checked in the call path")
            return
        if r.unchecked and self._check_task is not None and not self._check_task.done():
            # only a call reading unchecked tables waits for the session's check (it may cover them)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._check_task), timeout=self.settings.service.timeout_s)
            r = call_readiness(contract, self.readiness, bound_table=selected, args=plan.args_raw,
                               partition_values=parts)
        if r.unchecked:
            ok = await self.refresh_readiness(r.unchecked)
            if not ok:
                if self.settings.gateway.when_service_down == "strict":
                    down = self.service.down_for_session
                    raise ServiceError("the data-layer service is down" + (" for the rest of this session"
                                                                           if down else "") +
                                       f", so this tool cannot be guarded ({self.service.last_error})",
                                       tool=st.name, subkind="down" if down else None)
                st.lenient = True
                st.notes.append("data-layer service unavailable: readiness and witness checks skipped")
            r = call_readiness(contract, self.readiness, bound_table=selected, args=plan.args_raw,
                               partition_values=parts)
        for x in r.soft:                               # section tables mark their sections unavailable
            st.soft_sections[x["name"]] = f"{x['name']} not ready: {x['check']}"
        hard = r.hard
        if hard:
            first = hard[0]
            raise GatewayError(ErrorKind.not_ready, f"{first['name']} is not ready ({first['check']}: "
                               f"{first['detail']})", tool=st.name, payload=not_ready_payload(hard))
        st.notes.extend(r.notes)

    # ---------------------------------------------------------------- vocabularies

    async def _vocab_for(self, contract: ToolContract, args: Mapping[str, Any], selected: str | None) -> dict[str, Any]:
        """Vocabulary snapshots for category/scope arguments with values (and free-text substring
        arguments, and the auto-derived scope arguments)."""
        wanted: list[tuple[str, str]] = []
        for name, a in contract.args.items():
            if args.get(name) is None and not a.gateway_only:
                continue
            table, column = bound_column(contract, name, a, selected)
            if not table or not column:
                continue
            spec = column_spec(contract, table, column)
            role = getattr(spec, "role", None)
            if a.accepts or role in ("identifier", "endpoint"):
                continue
            if role in ("category", "scope") or getattr(spec, "scope", None) is not None or \
                    a.interpreted_as in ("substring", "casefold_substring"):
                if isinstance(getattr(spec, "vocab", None), list) and a.role != "free_text":
                    continue
                wanted.append((table, column))
                af = getattr(spec, "aliases_from", None)
                if af is not None:
                    wanted.append((af.table if "." in af.table else f"{table.split('.')[0]}.{af.table}", af.column))
        bound = selected or contract.bound_table
        for column in self._scope_dims(contract, bound):
            wanted.append((bound, column))
        out: dict[str, Any] = {}
        for table, column in dict.fromkeys(wanted):
            key = vocab_key(table, column)
            if key not in self._vocab:
                resp = await self.service.try_call("_vocab", _vocab_request(table, column))
                if resp is None:
                    continue
                self._vocab[key] = VocabSnapshot(values=list(resp.values), rendered=list(resp.rendered),
                                                 storage_type=resp.storage_type, complete=resp.complete,
                                                 fingerprint=resp.fingerprint)
            out[key] = self._vocab[key]
        return out

    def _scope_dims(self, contract: ToolContract, bound: str | None) -> list[str]:
        t = contract.tables.get(bound) if bound else None
        if t is None:
            return []
        scope = t.scope_columns()
        names = {_last(c) for c in scope}
        return [k for k in t.key if k in scope or _last(k) in names]

    def _dim_values(self, vocab: Mapping[str, Any], contract: ToolContract, selected: str | None, *,
                    every: bool = False) -> dict[str, list[Any]]:
        """Values of each scope dimension from the vocabulary: only a dimension with one value in the whole
        table (certainly one here), or with ``every`` all stored values (for an error's retry list)."""
        bound = selected or contract.bound_table
        out = {}
        for dim in self._scope_dims(contract, bound):
            snap = vocab.get(vocab_key(bound or "", dim))
            if snap is not None and (every or len(snap.values) <= 1):
                out[dim] = list(snap.values)
        return out

    def _snap_auto(self, contract: ToolContract, column: str, value: Any, snap: Any, st: _CallState) -> Any:
        from .contracts import snap_number
        if snap is None:
            return value
        values = list(snap.values)
        if value in values:
            return value
        hit = snap_number(value, values, snap.storage_type) if isinstance(value, (int, float, str)) else None
        if hit is not None:
            st.storage_types.setdefault(column, snap.storage_type)
            return hit
        from ..errors import invalid_argument_payload
        name = _last(column)
        raise GatewayError(ErrorKind.invalid_argument, f"{name}={value!r} is not a value of {column}", tool=st.name,
                           payload=invalid_argument_payload(name, value,
                                                            [json_value(v, snap.storage_type) for v in values],
                                                            enum_max=self.settings.derive.enum_max))

    def _confirmed(self, table: str | None) -> dict[str, Any]:
        m = self.readiness.get(table) if table else None
        return dict(m.confirmed) if m is not None else {}

    def _fixed(self, st: _CallState, prepared: PreparedArgs | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if prepared is not None:
            out.update(prepared.fixed)
        for arg, col in st.resolved_columns.items():
            v = st.values.get(arg)
            if v is not None and not isinstance(v, list):
                out[col] = v
        out.update(st.auto_fixed)
        return out

    def _units(self, st: _CallState) -> dict[str, str]:
        out: dict[str, str] = {}
        w = st.witness
        if w is None:
            return out
        for col, vals in (w.distinct or {}).items():
            if col.endswith("unit") and len(vals) == 1:
                out[col[:-len("_unit")] if col.endswith("_unit") else col] = str(vals[0])
        return out

    # ---------------------------------------------------------------- SOMA filters

    async def _soma(self, plan: CallPlan, st: _CallState, contract: ToolContract) -> None:
        for name, a in contract.args.items():
            if a.escape != soma_filter.LANGUAGE:
                continue
            text = plan.args_raw.get(name)
            include_dup = bool(plan.args_raw.get(_INCLUDE_DUPLICATES) or plan.gateway_args.get(_INCLUDE_DUPLICATES))
            try:
                pred = soma_filter.parse(text) if is_present(text) else None
                if pred is not None:
                    cols = soma_filter.filter_columns(pred)
                    vocab = await self._soma_vocab.fetch(cols, lambda c: self._soma_values(plan.server, c))
                    pred, notes = soma_filter.resolve_filter(pred, vocab)
                    st.notes.extend(notes)
                pred, added = soma_filter.enforce_primary(pred, include_dup)
            except soma_filter.SomaFilterError as exc:
                from ..errors import invalid_argument_payload
                raise GatewayError(ErrorKind.invalid_argument, f"{name}: {exc}", tool=st.name, argument=name,
                                   value=text, payload={**invalid_argument_payload(
                                       name, exc.value if exc.value is not None else text, exc.valid_values,
                                       enum_max=self.settings.derive.enum_max), "reason": str(exc)}) from None
            if pred is not None:
                plan.args_sent[name] = soma_filter.compile_filter(pred)
                st.soma_added = added
                if added:
                    st.notes.append(f"{name}: {soma_filter.PRIMARY_COLUMN} == True added (duplicate cells "
                                    f"excluded; pass {_INCLUDE_DUPLICATES}=true to include them)")
                st.soma_predicate = pred

    async def _soma_values(self, server: str, column: str) -> list[Any] | None:
        if self.bridge is None:
            return None
        try:
            out = await self.bridge.call_raw(server, soma_filter.VOCAB_TOOL, {soma_filter.VOCAB_ARG: column,
                                                                             "limit": 100000})
        except Exception:  # noqa: BLE001
            return None
        obj, ok = parse_payload(out if isinstance(out, str) else json.dumps(out, default=str), None)
        return list(jp_get(obj, soma_filter.VOCAB_PATH)) if ok else None

    def _unique_within(self, plan: CallPlan, st: _CallState, contract: ToolContract, selected: str | None) -> None:
        """Columns unique only within others (``donor_id`` within ``dataset_id``) that the call reads
        need their qualifier fixed (I7)."""
        b = contract.binding
        if b is not None:
            for rq in b.requires_fixed:
                pred = st.soma_predicate if rq.arg in contract.args else None
                for col in rq.columns:
                    if pred is not None and soma_filter.fixes_single(pred, col):
                        continue
                    raise GatewayError(ErrorKind.unsupported_combination,
                                       f"{rq.arg} must fix one {col}: {rq.reason}", tool=st.name, argument=rq.arg,
                                       payload=unsupported_combination_payload(
                                           [rq.arg], rq.reason, alternative=rq.alternatives[0]
                                           if rq.alternatives else None))
        t = contract.tables.get(selected or contract.bound_table or "")
        if b is None or t is None:
            return
        read = {c for rs in b.reads.values() for c in rs.columns}
        read |= {c for n in contract.args for tb, c in contract.arg_columns(n) if is_present(plan.args_raw.get(n))}
        soma_pred = st.soma_predicate
        fixed = {_last(k) for k in self._fixed(st, st.prepared)}
        for col in read:
            spec = t.columns.get(_last(col))
            for q in getattr(spec, "unique_within", []) or []:
                qn = _last(q)
                if qn in fixed or (soma_pred is not None and soma_filter.fixes_single(soma_pred, qn)):
                    continue
                from ..errors import incomplete_key_payload
                raise GatewayError(ErrorKind.incomplete_key,
                                   f"{_last(col)} is unique only within {qn}; fix {qn} to one value",
                                   tool=st.name, payload=incomplete_key_payload(qn, None, []))

    # ---------------------------------------------------------------- resolution

    async def _resolve(self, plan: CallPlan, st: _CallState, contract: ToolContract, prepared: PreparedArgs,
                       selected: str | None) -> None:
        name = st.name
        for req in prepared.list_resolution_requests:
            a = req.binding
            table, column = req.table, req.column
            spec = column_spec(contract, table, column)
            src = table.split(".")[0] if table else None
            bound_type = getattr(spec, "id_type", None)
            accepts = list(a.accepts) or ([bound_type] if bound_type else [])
            if not accepts:
                continue
            try:
                usable = []
                for k in accepts:
                    q = self.catalog.qualify_id_type(k, src)
                    needed = self._index_needed(q)
                    ok = True
                    for n in needed:
                        if not await self._ensure_index(n):
                            ok = False
                    if ok:
                        usable.append(k)
                    else:
                        st.notes.append(f"{req.arg}: {q} not tried (resolver index not ready)")
            except CatalogError as exc:
                raise GatewayError(ErrorKind.not_ready, f"{req.arg}: {exc}", tool=name,
                                   payload=not_ready_payload([{"name": name, "check": "binding",
                                                               "detail": str(exc)}])) from None
            if not usable:
                raise GatewayError(ErrorKind.not_ready, f"no resolver index is ready for {req.arg}", tool=name,
                                   payload=not_ready_payload([{"name": f"{src}:{accepts[0]}", "check": "index",
                                                               "detail": self._index_failed.get(
                                                                   self.catalog.qualify_id_type(accepts[0], src), ""),
                                                               "hint": "build it with `vbt ds index build`"}]))
            bound_q = self.catalog.qualify_id_type(bound_type, src) if bound_type else None
            existence = a.existence
            try:
                if req.is_list:
                    mrf = a.min_resolved_fraction
                    results, summary = await self.resolver.aresolve_list(
                        req.values, usable, bound_id_type=bound_q, existence=existence,
                        universe_where=a.universe_where, where_params=plan.args_raw, source_hint=src,
                        on_missing=a.on_missing, min_resolved_fraction=mrf, dedupe=a.dedupe)
                    err = list_error(summary, req.arg, tool=name)
                    if err is not None:
                        raise err
                    st.resolution_summary[req.arg] = {k: json_value(summary[k]) for k in
                                                      ("requested", "resolved", "unresolved", "ambiguous",
                                                       "outside_universe", "duplicates")}
                    sent, canon = [], []
                    for r in results:
                        if r.ok and r.canonical is not None and r.canonical in summary["canonicals"] and \
                                r.canonical not in canon:
                            canon.append(r.canonical)
                            sent.append(self.resolver.send_value(r, a.send_as, table))
                            plan.resolutions.append(_record(r, req.arg))
                    if any(r.status == "unknown" for r in results):
                        plan.existence[req.arg] = "unknown"
                        st.notes.append(f"{req.arg}: existence of some values could not be decided")
                    else:
                        plan.existence[req.arg] = "exists"
                    by_canon = {r.canonical: r for r in results if r.canonical is not None}
                    st.values[req.arg] = [self._stored(by_canon[c], table) for c in canon]
                    plan.args_sent[req.arg] = sent
                    if a.gateway_only:
                        plan.gateway_args[req.arg] = sent
                        plan.args_sent.pop(req.arg, None)
                    continue
                res = await self.resolver.aresolve(req.values[0], usable, bound_id_type=bound_q,
                                                   existence=existence, universe_where=a.universe_where,
                                                   where_params=plan.args_raw, source_hint=src)
            except ResolverConfigError as exc:
                raise GatewayError(ErrorKind.not_ready, f"{req.arg}: the binding cannot be resolved ({exc})",
                                   tool=name, payload=not_ready_payload([{"name": name, "check": "resolver_config",
                                                                          "detail": str(exc)[:300]}])) from None
            err = error_for(res, req.arg, tool=name, enum_max=self.settings.derive.enum_max)
            if err is not None:
                raise err
            st.results[req.arg] = res
            canonical_value = res.canonical
            if res.status == "resolved_unverified":
                found = await self._count(table, Eq(column, canonical_value), st) if table and column else None
                if found is None:
                    plan.existence[req.arg] = "unknown"
                    st.notes.append(f"{req.arg}: {res.raw!r} is not in the identity universe and the bound column "
                                    "could not be counted; existence unknown")
                elif found == 0:
                    payload = not_found_payload(req.arg, res.raw, res.id_type, list(res.accepts), list(res.tried),
                                                [], res.source, table)
                    raise GatewayError(ErrorKind.not_found, f"{req.arg}={res.raw!r} is in neither the identity "
                                       f"universe nor {table}.{column}", tool=name, payload=payload)
                else:
                    plan.existence[req.arg] = "exists"
                    st.notes.append(f"{req.arg}: {canonical_value} is not in the identity universe but occurs in "
                                    f"{_last(table or '')}.{column} ({found} rows)")
            elif res.status == "unknown":
                plan.existence[req.arg] = "unknown"
                st.notes.append(f"{req.arg}: existence of {res.raw!r} could not be decided; an empty result is not "
                                "citable")
            else:
                plan.existence[req.arg] = "unknown" if res.existence == "unknown" else "exists"
            if canonical_value is None:
                canonical_value = res.raw
            send = self.resolver.send_value(res, a.send_as, table)
            plan.resolutions.append(_record(res, req.arg))
            if res.rule not in (None, "exact", "raw_member") or str(res.raw) != str(canonical_value):
                st.resolved[req.arg] = res.summary()
            for n in res.notes or ():
                st.notes.append(f"{req.arg}: {n}")
            if a.family == "include" and len(res.family) > 1:
                st.values[req.arg] = list(res.family)
                st.notes.append(f"{req.arg}: family members {', '.join(res.family)} included")
            else:
                st.values[req.arg] = self._stored(res, table) if res.canonical is not None else canonical_value
            if a.role == "anchor":
                if table and column:
                    present = await self._count(table, Eq(column, canonical_value), st)
                    if present == 0:
                        st.undefined[req.arg] = {"argument": req.arg, "value": json_value(canonical_value),
                                                 "reason": "not_in_table"}
                    elif present is None:
                        st.notes.append(f"{req.arg}: presence in {table} could not be checked")
                    st.anchors[req.arg] = (column, canonical_value)
                st.values.pop(req.arg, None)
            if a.gateway_only:
                plan.gateway_args[req.arg] = send
                plan.args_sent.pop(req.arg, None)
            else:
                plan.args_sent[req.arg] = send
            if a.op == "eq" and column and a.role != "anchor":
                st.resolved_columns[req.arg] = column

    def _stored(self, res: Any, table: str | None) -> Any:
        """The bound table's own spelling of a resolved key (predicates read the table as stored)."""
        if not table:
            return res.canonical
        return self.resolver.send_value(res, "stored", table)

    async def _count(self, table: str | None, pred: Predicate, st: _CallState) -> int | None:
        """One witness count (existence ``bound``, anchors, coverage universes); None when unknown."""
        if not table or st.lenient:
            return None
        req = WitnessRequest(table=table, predicate=to_json(pred), key_set_max=0,
                             budget_bytes=self.settings.witness.max_scan_bytes)
        try:
            resp = await self.service.witness(req)
        except ServiceError:
            return None
        if resp.total_method == "unknown" or resp.total is None:
            return None
        return int(resp.total)

    # ---------------------------------------------------------------- order, limits, witness

    def _effective_order(self, contract: ToolContract, prepared: PreparedArgs) -> list[Any]:
        b = contract.binding
        if prepared.order and prepared.order.get("column"):
            return [{"column": prepared.order["column"], "direction": prepared.order.get("direction", "desc"),
                     "nulls": "last"}]
        return list(b.result.order) if b is not None else []

    def _requested_limit(self, plan: CallPlan, contract: ToolContract, schema: Mapping[str, Any] | None) -> int | None:
        name = contract.limit_arg
        if not name:
            return None
        v = plan.args_sent.get(name, plan.gateway_args.get(name))
        if v is None:
            prop = ((schema or {}).get("properties") or {}).get(name) or {}
            v = prop.get("default") if isinstance(prop, Mapping) else None
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _scannable(self, contract: ToolContract, table: str | None) -> bool:
        t = contract.tables.get(table) if table else None
        if t is None or t.descriptor.kind == "remote" or (t.format or "") == "none":
            return False
        rs = contract.binding.reads.get(table) if contract.binding is not None else None
        return not (rs is not None and rs.access in ("remote", "upstream"))

    def _search_text(self, plan: CallPlan, st: _CallState, contract: ToolContract) -> None:
        """A derived ``search`` matches its free-text argument itself (ranked by match class), so that
        argument becomes ``search_text`` instead of a substring conjunct."""
        b = contract.binding
        d = b.derived if b is not None else None
        if d is None or d.verb != "search" or b.serve != "derived" or self.profile == "fidelity":
            return
        for name, a in contract.args.items():
            value = plan.args_raw.get(name)
            if a.role == "free_text" and isinstance(value, str) and value.strip():
                st.search_text = value
                st.per_arg.pop(name, None)
                return

    def _inexpressible(self, plan: CallPlan, contract: ToolContract) -> str | None:
        for name, a in contract.args.items():
            if not is_present(plan.args_raw.get(name)):
                continue
            # text matched over several columns (binds_any) is the source's matching, not one column's
            if a.role == "free_text" and (a.interpreted_as in ("engine", "regex") or not a.binds):
                return f"{name} is matched by the source's engine"
            if a.role == "unbound":
                return f"{name} is not bound to a column"
        return None

    async def _witness(self, plan: CallPlan, st: _CallState, contract: ToolContract, selected: str | None, *,
                       derived: bool) -> None:
        b = contract.binding
        table = selected or contract.bound_table
        t = contract.tables.get(table) if table else None
        if st.mode != "enforce":
            st.witness_reason = "observe mode: no witness scan in the call path"
            return
        if t is None or st.lenient or not self.settings.witness.enabled or not b.witness:
            st.witness_reason = "no witness for this binding"
            return
        if not self._scannable(contract, table):
            st.witness_reason = f"{table} is not scannable by the data child"
            return
        why = self._inexpressible(plan, contract)
        if why:
            st.witness_reason = why
            st.engine_matched = True
            return
        dims = [d for d in self._scope_dims(contract, table)
                if _last(d) not in {_last(k) for k in self._fixed(st, st.prepared)}]
        if derived and not dims:
            return                                     # derived serving counts itself (one scan)
        predicate = st.predicate
        it = contract.tables.get(b.result.rows_of) if b.result.rows_of else None
        if it is not None and it.is_item_table and str(it.physical) == table:
            # the rows are items of the bound table: count those
            table, t = str(b.result.rows_of), it
            predicate = _on_items(predicate, contract, table, selected)
        await self._storage_types(table, st)
        # an item table's parent key parts are table-level ("/id"): a bare name is the item's own field
        key = [("/" + k if t.is_item_table and "[]" not in k and not k.startswith("/") else k)
               for k in t.key if not k.endswith("#")]
        order = [RankKeyModel(**_rank_json(o)) for o in st.order] if not derived else []
        within = sorted({w for o in st.order for w in (_rank_json(o).get("within") or [])})
        grains = {g: (list(spec) if isinstance(spec, list) else spec.model_dump(exclude_defaults=True))
                  for g, spec in t.spec.grains.items()}
        unknown_cols = [c for n, a in contract.args.items() if a.op in ("ge", "gt", "le", "lt", "range") or
                        a.op.endswith("_abs") for _, c in contract.arg_columns(n) if is_present(plan.args_raw.get(n))]
        k = st.requested_limit if self.settings.witness.topk and not derived else None
        distinct = list(dims)
        for d in dims:
            col = t.scope_columns().get(d)
            unit_from = getattr(col, "unit_from", None)
            if unit_from:
                distinct.append(unit_from)
        grain = None if b.result.grain in (None, "row", "rows") else b.result.grain   # "row": count rows
        req = WitnessRequest(table=table, grain=grain, predicate=to_json(predicate) if predicate
                             else None, key=key, order=order, k=k, group_by=within, distinct=distinct, grains=grains,
                             unknown_columns=unknown_cols, key_set_max=self.settings.witness.max_key_set,
                             budget_bytes=t.spec.max_scan_bytes or self.settings.witness.max_scan_bytes,
                             params={n: v for n, v in plan.args_raw.items() if isinstance(v, (str, int, float, bool))})
        try:
            st.witness = await self.service.witness(req)
        except ServiceError as exc:
            if self.settings.gateway.when_service_down == "strict":
                raise
            st.witness_reason = exc.message
            st.lenient = True
            return
        plan.witness = st.witness.model_dump(mode="json")
        if st.witness.total_method == "unknown":
            st.witness_reason = st.witness.reason or "the witness could not count this call"
        st.checks.append(("witness_count", True if st.witness.total is not None else None,
                          f"total={st.witness.total} ({st.witness.total_method})"))

    async def _storage_types(self, table: str | None, st: _CallState) -> None:
        if not table:
            return
        if table not in self._stats:
            try:
                resp = await self.service.stats([table])
                self._stats[table] = resp.tables.get(table)
            except ServiceError:
                self._stats[table] = None
        stats = self._stats.get(table)
        if stats is not None:
            for col, cs in stats.columns.items():
                st.storage_types.setdefault(col, cs.storage_type)
        for key, snap in self._vocab.items():
            if key.startswith(table + "."):
                st.storage_types.setdefault(key[len(table) + 1:], getattr(snap, "storage_type", None))

    def _inflatable(self, plan: CallPlan, st: _CallState, contract: ToolContract) -> bool:
        b = contract.binding
        w = st.witness
        if plan.route != "upstream" or not b.witness or w is None or w.total is None or w.total_method == "unknown":
            return False
        if contract.limit_arg is None or st.requested_limit is None:
            return False
        if any(a.role == "output_path" and is_present(plan.args_raw.get(n)) for n, a in contract.args.items()):
            return False
        n = int(w.total) + int(w.unknown_total or 0)
        if n > self.settings.witness.max_inflate_rows:
            return False
        stats = self._stats.get(plan.bound_table or contract.bound_table or "")
        row_bytes = None
        if stats is not None and stats.row_bytes_p99:
            row_bytes = stats.row_bytes_p99.get(b.result.grain or "row") or max(stats.row_bytes_p99.values())
        return row_bytes is None or n * row_bytes <= self.settings.witness.max_inflate_bytes

    def _inflate(self, plan: CallPlan, st: _CallState, contract: ToolContract, inflatable: bool) -> None:
        b = contract.binding
        name = contract.limit_arg
        st.sent_limit = st.requested_limit
        if name is None or st.requested_limit is None:
            return
        w = st.witness
        if inflatable and w is not None:
            n = int(w.total or 0) + int(w.unknown_total or 0)
            if n > st.requested_limit:
                target = max(n, st.requested_limit)
                if name in plan.gateway_args:
                    plan.gateway_args[name] = target
                else:
                    plan.args_sent[name] = target
                st.sent_limit = target
                st.inflated = True
                st.transforms.append(f"limit_inflation {st.requested_limit}->{target}")
            return
        ordered = bool(st.order) and b.result.order_source == "witness" and b.witness
        if not ordered or st.lenient or st.engine_matched:
            return                                    # engine-matched text: the source's order, disclosed unverified
        if w is not None and w.total is not None and w.total_method != "unknown" and \
                int(w.total) + int(w.unknown_total or 0) <= st.requested_limit:
            return                                    # nothing can be truncated
        topk_ok = w is not None and self.settings.witness.topk and bool(w.topk) and w.total_method != "unknown"
        if not topk_ok:
            raise GatewayError(ErrorKind.too_large,
                               "the ranking cannot be verified: the result may be truncated before it is ranked",
                               tool=st.name, subkind="unranked_truncation",
                               payload=too_large_payload("unranked_truncation",
                                                         hint="narrow the query or use mcp__data__find",
                                                         alternative="mcp__data__find"))

    # ---------------------------------------------------------------- admission

    async def _admit(self, plan: CallPlan, st: _CallState, contract: ToolContract) -> None:
        b = contract.binding
        reads: list[TableRead] = []
        tables = [t for t, rs in b.reads.items() if rs.access in ("full_table", "bounded_scan")
                  and not (rs.when and any(plan.args_raw.get(k) != v for k, v in rs.when.items()))]
        missing = [t for t in tables if t not in self._stats]
        if missing and not st.lenient:
            try:
                resp = await self.service.stats(missing)
                for t in missing:
                    self._stats[t] = resp.tables.get(t)
            except ServiceError:
                for t in missing:
                    self._stats.setdefault(t, None)
        for t in tables:
            reads.append(TableRead(t, b.reads[t].access, self._stats.get(t)))
        for t, rs in b.reads.items():
            if rs.access == "remote" and rs.count_via and rs.est_row_bytes:
                total = await self._remote_count(rs.count_via, plan.args_sent)
                self.admission.admit_remote(total, rs.est_row_bytes, self.settings.memory.max_result_bytes,
                                            tool=st.name, table=t)
        await self._admit_sized(plan, st, contract)
        adm = await self.admission.admit(plan.server, reads, "upstream", tool=st.name)
        st.admission = adm
        plan.cold_tables = tuple(adm.cold_tables)
        plan.cold_lock = adm.lock

    async def _admit_sized(self, plan: CallPlan, st: _CallState, contract: ToolContract) -> None:
        """Count-first admission of reads whose table declares ``size_from`` with a ``via`` tool (§14.1):
        the scoping record's count x ``row_bytes`` over ``memory.max_result_bytes`` is ``too_large``
        before the download (a whole cBioPortal study for get_clinical_data)."""
        b = contract.binding
        seen: set[tuple[str, str]] = set()
        for ref in b.reads:
            t = contract.tables.get(ref)
            sf = getattr(getattr(t, "spec", None), "size_from", None) if t is not None else None
            if sf is None or not sf.via or not sf.row_bytes:
                continue
            key = [k for k in t.key if not k.endswith("#")]
            arg = next((n for n in contract.args for tb, c in contract.arg_columns(n)
                        if tb == ref and key and _last(c) == _last(key[0]) and is_present(plan.args_sent.get(n))),
                       None)
            if arg is None or (sf.via, str(plan.args_sent[arg])) in seen:
                continue
            seen.add((sf.via, str(plan.args_sent[arg])))
            total = await self._remote_value(sf.via, {sf.arg or arg: plan.args_sent[arg]}, sf.path)
            self.admission.admit_remote(total, sf.row_bytes, self.settings.memory.max_result_bytes,
                                        tool=st.name, table=ref)

    async def _remote_value(self, ref: str, args: Mapping[str, Any], path: str | None) -> int | None:
        """The count a ``server.tool`` call returns at ``path`` (the largest of a mapping of counts)."""
        server, _, tool = ref.partition(".")
        if not tool or self.bridge is None or not path:
            return None
        try:
            out = await self.bridge.call_raw(server, tool, dict(args))
        except Exception:  # noqa: BLE001 - no count: admission stays lenient
            return None
        obj, ok = parse_payload(out if isinstance(out, str) else json.dumps(out, default=str), None)
        value = jp_first(obj, path) if ok else None
        if isinstance(value, Mapping):
            nums = [v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            value = max(nums) if nums else None
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    async def _remote_count(self, ref: str, args: Mapping[str, Any]) -> int | None:
        server, _, tool = ref.partition(".")
        if not tool or self.bridge is None:
            return None
        try:
            out = await self.bridge.call_raw(server, tool, dict(args))
        except Exception:  # noqa: BLE001
            return None
        obj, ok = parse_payload(out if isinstance(out, str) else json.dumps(out, default=str), None)
        if not ok or not isinstance(obj, Mapping):
            return None
        for k in ("count", "total", "total_count", "totalCount"):
            if isinstance(obj.get(k), int):
                return int(obj[k])
        return None

    # ================================================================== observe

    def _observe(self, plan: CallPlan, decision: str, **detail: Any) -> None:
        event = {"tool": f"mcp__{plan.server}__{plan.tool}", "decision": decision, **detail}
        self.observed.append(event)
        emit = getattr(self.bridge, "_emit", None)
        if callable(emit):
            with contextlib.suppress(Exception):
                emit("data_observe", **json.loads(json.dumps(event, default=str)))

    # ================================================================== crash

    async def on_crash(self, server: str, plan: CallPlan, reason: str, log_tail: str) -> CrashDecision:
        st = _state(plan)
        decision = crash_decision(reason, log_tail, server=server, tool=plan.tool, attempt=st.attempts)
        st.attempts += 1
        if decision.oom:
            adm = st.admission
            if adm is not None:
                self.admission.commit(adm, ok=False, oom=True)
            elif plan.cold_tables:
                self.admission.learn_refusal(server, plan.cold_tables)
        return decision

    # ================================================================== finish

    async def finish(self, plan: CallPlan, raw: RawResult | None) -> Any:
        st = _state(plan)
        t1 = time.monotonic()
        if st.mode != "enforce":
            from ...tools.mcp_bridge import legacy_result
            try:
                cls = classify(raw, plan.contract, plan, universe_tables=self._universe_tables(plan.contract),
                               registry=self.registry)
                if cls.outcome != "ok":
                    self._observe(plan, f"would_{cls.outcome}", reason=cls.reason)
            except Exception:  # noqa: BLE001
                pass
            if raw is None:
                raise GatewayError(ErrorKind.source_error, "no upstream result", tool=st.name)
            return legacy_result(raw, plan.tool)
        if st.t_prepared is not None:
            st.t_ms["call"] = _ms(st.t_prepared)
        st.t_finish = t1
        try:
            result = await self._finish(plan, st, raw)
        except GatewayError as exc:
            exc.with_tool(st.name)
            oom = exc.kind == ErrorKind.oom
            self._release(st, ok=not oom, oom=oom)
            if oom and plan.route == "upstream":
                await self._recycle_after_oom(plan.server)
            raise
        except BaseException:
            self._release(st, ok=False)
            raise
        self._release(st, ok=True)
        return result

    def _release(self, st: _CallState, *, ok: bool, oom: bool = False) -> None:
        """Commit (or release) the call's admission reservation, once."""
        adm, st.admission = st.admission, None
        if adm is not None:
            self.admission.commit(adm, ok=ok, oom=oom)

    def abandon(self, plan: CallPlan) -> None:
        """The upstream attempt raised before :meth:`finish`: release the call's memory reservation so
        it does not count as resident until its TTL."""
        self._release(_state(plan), ok=False)

    async def _recycle_after_oom(self, server: str) -> None:
        """§14.4: an in-tool memory error leaves the server holding a partial cache near its limit;
        restart it proactively (thrash-guarded by admission)."""
        with contextlib.suppress(Exception):
            await self.admission.recycle_after_oom(server)

    def _universe_tables(self, contract: ToolContract) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, a in contract.args.items():
            refs: list[str] = []
            table, column = bound_column(contract, name, a)
            src = table.split(".")[0] if table else None
            kinds = list(a.accepts)
            spec = column_spec(contract, table, column)
            if getattr(spec, "id_type", None):
                kinds.append(spec.id_type)
            for k in kinds:
                try:
                    s, idt = self.catalog.id_type(k, src)
                except Exception:  # noqa: BLE001
                    continue
                universes: list[Any] = idt.universe if isinstance(idt.universe, list) else [idt.universe]
                for u in universes:
                    if u is None:
                        continue
                    ut = getattr(u, "table", None) or str(u).split(".")[0]
                    refs.append(ut if "." in ut and ut.split(".")[0] in self.catalog.sources else f"{s}.{ut}")
            if a.universe:
                refs.append(a.universe if a.universe.count(".") == 1 else ".".join(a.universe.split(".")[:2]))
            if refs:
                out[name] = list(dict.fromkeys(refs))
        return out

    async def _finish(self, plan: CallPlan, st: _CallState, raw: RawResult | None) -> DataResult:
        contract: ToolContract = plan.contract
        if st.unguarded:
            from ...tools.mcp_bridge import legacy_result
            if raw is None:
                raise GatewayError(ErrorKind.source_error, "no upstream result", tool=st.name)
            return legacy_result(raw, plan.tool)
        if st.generic:
            return self._finish_generic(plan, st, raw)
        counters = Counters()
        served_by = "upstream"
        if plan.route == "none":
            return self._result(plan, st, obj={}, rows=[], counters=counters, served_by="derived",
                                total=0, total_method="anchor", status="empty")
        if plan.route == "derived":
            obj, rows, resp = await self._serve(plan, st, contract)
            if not rows:
                st.in_universe = await self._in_coverage_universe(st, self._rows_table(plan, contract))
            return self._finish_rows(plan, st, contract, obj, rows, counters, served_by="derived", serve=resp)
        if raw is None:
            raise GatewayError(ErrorKind.source_error, "no upstream result", tool=st.name)
        w = st.witness
        cls = classify(raw, contract, plan, universe_tables=self._universe_tables(contract),
                       witness_total=w.total if w is not None and w.total_method != "unknown" else None,
                       registry=self.registry)
        if cls.outcome == "oom":
            raise cls.error  # type: ignore[misc]
        if cls.outcome in ("not_found", "source_error"):
            raise cls.error  # type: ignore[misc]
        if cls.outcome == "contradiction":
            return await self._contradiction(plan, st, contract, "W1", 0, cls.reason)
        if cls.outcome in ("empty", "empty_unverified"):
            st.notes.append(cls.reason)
            return self._result(plan, st, obj=cls.obj if cls.is_json else {}, rows=[], counters=counters,
                                served_by=served_by, total=0 if cls.outcome == "empty" else None,
                                total_method="witness_scan" if cls.outcome == "empty" else "unknown",
                                status=cls.outcome)
        if cls.outcome == "partial":
            st.notes.append(cls.reason)
            st.force_partial = True
        obj = cls.obj
        if not cls.is_json:
            return self._result(plan, st, obj=obj, rows=[], counters=counters, served_by=served_by, total=None,
                                total_method="unknown", status="partial" if cls.outcome == "partial" else "ok",
                                text_rows=True)
        return await self._process_upstream(plan, st, contract, obj, raw, counters)

    # ---------------------------------------------------------------- upstream rows

    def _mapper(self, contract: ToolContract) -> FieldMapper:
        r = contract.binding.result
        it = contract.tables.get(r.rows_of) if r.rows_of else None
        prefix = it.items_path if it is not None and it.is_item_table else None
        return FieldMapper(r.fields, parent_key=r.parent_key, key_from_args=r.key_from_args, item_prefix=prefix)

    def _rows_table(self, plan: CallPlan, contract: ToolContract) -> Any:
        b = contract.binding
        ref = b.result.rows_of or plan.bound_table or contract.bound_table
        return contract.tables.get(ref) if ref else None

    async def _process_upstream(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any,
                                raw: RawResult, counters: Counters) -> DataResult:
        b = contract.binding
        mapper = self._mapper(contract)
        paths = b.result.row_paths
        record = b.result.kind == "record"
        if b.result.record_when and jp_test(obj, b.result.record_when):
            paths, record = ["$"], True                # a by-key reply: the payload is the one record
        groups = extract_rows(obj, paths)
        anchor_vec = None
        rows_by_path: list[tuple[str, list[Any]]] = []
        for path, rows in groups:
            rows_by_path.append((path, mapper.logical_rows(rows, payload=obj, args=plan.args_sent,
                                                           anchor_vector=anchor_vec)))
        all_rows = [r for _, rs in rows_by_path for r in rs]
        returned_raw = len(all_rows)
        t = self._rows_table(plan, contract)
        # T1-T6
        processed: list[tuple[str, list[Any]]] = []
        phantoms: list[Any] = []
        for path, rows in rows_by_path:
            rows = self._t1_to_t6(plan, st, contract, t, rows, counters, mapper, phantoms)
            processed.append((path, rows))
        if phantoms:
            items = [_phantom_label(r, contract) for r in phantoms]
            if b.result.on_unknown_items == "not_found":
                payload = not_found_payload(self._phantom_arg(contract) or "", items, None, [], ["upstream: "
                                            "rows for records that do not exist"], [], None,
                                            plan.bound_table)
                payload["items"] = items
                raise GatewayError(ErrorKind.not_found, f"{len(items)} requested item(s) do not exist: "
                                   f"{', '.join(map(str, items[:5]))}", tool=st.name, payload=payload)
            st.notes.append(f"{len(items)} unknown item(s) withheld")
            st.not_found_items = items
        rows_now = [r for _, rs in processed for r in rs]
        # witness checks W1-W6
        w = st.witness
        violated = {a: n for a, n in counters.excluded.items() if n}
        if violated and w is not None and (b.on_contradiction == "tool_defect" or self.profile == "fidelity"):
            # W2 after T4: upstream returned rows its own arguments exclude (rows outside the witness)
            detail = ", ".join(f"{n} row(s) violating {a}" for a, n in sorted(violated.items()))
            return await self._contradiction(plan, st, contract, "W2", len(rows_now) + sum(violated.values()),
                                             f"returned rows are not in the witness key set: {detail}")
        if not rows_now and returned_raw and b.result.rows_of and violated:
            # every item failed the argument that names its parent: the check, not the answer, is wrong
            parent = contract.tables.get(contract.bound_table) if contract.bound_table else None
            keyed = [a for a in violated if a in contract.args and parent is not None and
                     bound_column(contract, a, contract.args[a])[1] in parent.key]
            if keyed:
                return await self._contradiction(plan, st, contract, "W2", returned_raw,
                                                 f"every returned item failed {', '.join(keyed)}, the record's own key")
        defect = self._witness_checks(plan, st, contract, t, rows_now, returned_raw, obj)
        if defect is not None and defect[0] == "W5" and not self._keyed_lookup(plan, contract):
            self._one_to_many(plan, st, contract, t)
        if defect is not None:
            check, detail = defect
            if check == "W6" and st.attempts == 0 and contract.limit_arg:
                again = await self._recall(plan, st, contract)
                if again is not None:
                    return again
            if check == "W6":
                if b.derived is not None and b.on_contradiction == "derived" and self.profile != "fidelity":
                    return await self._repair(plan, st, contract, detail)
                st.notes.append(f"short page: {detail}")
                st.short_page = detail
            else:
                return await self._contradiction(plan, st, contract, check, len(rows_now), detail)
        # T7-T14 and placement
        return self._finish_rows(plan, st, contract, obj, rows_now, counters, served_by="upstream",
                                 groups=processed, mapper=mapper, record=record, witness=w)

    def _phantom_arg(self, contract: ToolContract) -> str | None:
        for n, a in contract.args.items():
            if a.each or a.op == "in":
                return n
        return None

    def _t1_to_t6(self, plan: CallPlan, st: _CallState, contract: ToolContract, t: Any, rows: list[Any],
                  counters: Counters, mapper: FieldMapper, phantoms: list[Any]) -> list[Any]:
        b = contract.binding
        # T1 leakage
        lk = st.leakage
        if lk is not None and lk.ceiling is not None and lk.specs:
            spec = next(iter(lk.specs.values()))
            rows = t1_leakage(rows, spec, lk.ceiling, counters)
        # T2 in-band unknowns and placeholders
        if t is not None:
            key0 = next((k for k in t.key if "[" not in k), None)
            rows = t2_unknowns(rows, t.columns, counters, key_column=key0)
        for bucket, d in mapper.counters.items():
            if bucket == "dangling" and d:
                counters.notes.append("dangling references: " + ", ".join(f"{k} ({v})" for k, v in d.items()))
        # T3 existence per row (W6)
        refs = self._nonnull_refs(t)
        rows, phantom = t3_existence(rows, b.result.exists_when, refs, counters)
        phantoms.extend(phantom)
        # T4 + T5 honour bound arguments; unknown never passes
        present: set[str] = set()
        for r in rows:
            if isinstance(r, Mapping):
                present.update(r.keys())
        from ..predicate import columns as pred_columns
        preds: dict[str, tuple[Predicate, Any]] = {}
        items_path = t.items_path if t is not None and t.is_item_table and b.result.rows_of else None
        parents: dict[str, str | None] = {}
        alias = {k.lstrip("/"): k for k in b.result.parent_key}
        for arg, pred in st.per_arg.items():
            binding = contract.args.get(arg.split("+")[0]) if not arg.startswith("@") else None
            if items_path:
                # rows are items: item columns are read on the item, parent columns from its parent key
                prefix = items_path + "."
                for c in pred_columns(pred):
                    if "[" not in c and not c.startswith(prefix):
                        parents[c] = alias.get(c) if alias.get(c) in present or not rows else None
                pred = on_item_rows(pred, items_path)
            tops = {c.lstrip("/").split(".")[0].split("[")[0] for c in pred_columns(pred)} - set(parents)
            if rows and not tops <= present:
                st.checks.append((f"honour:{arg}", False, f"{', '.join(sorted(tops - present))} not in the rows"))
                continue
            preds[arg] = (pred, binding)
        def spec_of(col: str) -> Any:
            return t.columns.get(col) if t is not None else None
        rows = honour_arguments(rows, preds, spec_of, counters, params=plan.args_raw, parents=parents)
        # T6 negation
        include_neg = bool(plan.args_raw.get(_INCLUDE_NEGATED) or plan.gateway_args.get(_INCLUDE_NEGATED))
        drop_empty = any(a.drop_empty_parents for a in contract.args.values())
        rows = t6_negation(rows, self._qualifier_paths(t, "negate"), include_neg, counters,
                           drop_empty_parents=drop_empty)
        return rows

    def _derived_items(self, plan: CallPlan, st: _CallState, contract: ToolContract, t: Any, rows: list[Any],
                       counters: Counters) -> list[Any]:
        """Derived rows: the data child selected parents; per-item filters (``item_filter``) and negated
        items (T4, T6) are applied here, so nested lists hold only qualifying, non-negated items."""
        preds = {arg: (pred, contract.args[arg]) for arg, pred in st.per_arg.items()
                 if arg in contract.args and contract.args[arg].item_filter}
        dt = contract.tables.get(contract.binding.derived.table) if contract.binding.derived is not None else None
        if dt is not None and dt.is_item_table and dt.items_path:
            # the derived rows are flat items: item filters read the item's (renamed) fields
            rename = dict(contract.binding.derived.rename or {})
            preds = {a: (on_item_rows(p, dt.items_path, rename), b) for a, (p, b) in preds.items()}
        if preds:
            before = len(rows)
            rows = honour_arguments(rows, preds, lambda col: t.columns.get(col) if t is not None else None, counters,
                                    params=plan.args_raw)
            st.derived_withheld += before - len(rows)
        include_neg = bool(plan.args_raw.get(_INCLUDE_NEGATED) or plan.gateway_args.get(_INCLUDE_NEGATED))
        drop_empty = any(a.drop_empty_parents for a in contract.args.values())
        before = len(rows)
        rows = t6_negation(rows, [p for p in self._qualifier_paths(t, "negate") if "[]" in p], include_neg, counters,
                           drop_empty_parents=drop_empty)
        st.derived_withheld += before - len(rows)
        return rows

    def _nonnull_refs(self, t: Any) -> list[str]:
        if t is None:
            return []
        nullable = set(t.nullable_key)
        out = []
        for name, c in t.columns.items():
            if getattr(c, "role", None) == "identifier" and getattr(c, "ref", None) and name not in nullable and \
                    getattr(c, "missing", None) not in ("non_entity", "not_applicable", "absent") and \
                    name in t.key:
                out.append(name)
        return out

    def _qualifier_paths(self, t: Any, effect: str) -> list[str]:
        if t is None:
            return []
        out: list[str] = []

        def walk(cols: Mapping[str, Any], prefix: str) -> None:
            for name, c in cols.items():
                if getattr(c, "role", None) == "qualifier" and getattr(c, "effect", None) == effect:
                    out.append(prefix + name)
                if is_container(c):
                    walk(c.fields, f"{prefix}{name}[].")

        walk(t.columns, "")
        return out

    def _key_columns(self, contract: ToolContract, t: Any) -> list[str]:
        r = contract.binding.result
        if isinstance(r.row_key, list):
            return [k[2:] if k.startswith("$.") else k for k in r.row_key]
        if t is None:
            return []
        if t.is_item_table:
            # item parts by their field name; parent parts by theirs, or as "/<path>" when an item field has
            # that name (the data child's output rows use the same names)
            keys = [k for k in t.key if not k.endswith("#")]
            item = {k.split("[].")[-1] for k in keys if "[]." in k}
            return [k.split("[].")[-1] if "[]." in k else ("/" + k if k.split(".")[-1] in item else k)
                    for k in keys]
        return [k for k in t.key if not k.endswith("#")]

    def _keys_of(self, rows: Sequence[Any], cols: Sequence[str], st: _CallState) -> list[str]:
        types = [st.storage_types.get(c) for c in cols]
        return [row_key(r, cols, types) for r in rows if isinstance(r, Mapping)]

    def _witness_checks(self, plan: CallPlan, st: _CallState, contract: ToolContract, t: Any, rows: list[Any],
                        returned_raw: int, obj: Any) -> tuple[str, str] | None:
        b = contract.binding
        w = st.witness
        # W4 echo (independent of the witness)
        for arg, spec in b.result.echo_specs().items():
            want = st.values.get(arg)
            if want is None or isinstance(want, list):
                continue
            got = jp_first(obj, spec.path)
            if got is None or str(got) == str(want):
                st.checks.append((f"echo:{arg}", True if got is not None else None, None))
                continue
            accepted = False
            for acc in spec.accept:
                if acc == "redirect" or acc.startswith("synonym:"):
                    accepted = True
                    st.resolved[arg] = f"{plan.args_raw.get(arg)} -> {got} (alias_redirect)"
                    st.notes.append(f"{arg}: the source answered for {got} (alias of {want})")
                    break
            if not accepted:
                st.checks.append((f"echo:{arg}", False, f"requested {want}, returned {got}"))
                return "W4", f"requested {want}, the record is {got}"
            st.checks.append((f"echo:{arg}", True, "alias_redirect"))
        if w is None or w.total is None or w.total_method == "unknown":
            return None
        total = int(w.total)
        cols = self._key_columns(contract, t)
        if self._parent_counted(contract):
            return None                                # the witness counts the parent record of a nested section
        # W1 empty
        if not rows and total > 0 and returned_raw == 0:
            return "W1", f"0 rows returned, the witness counted {total}"
        # W5 one-to-many
        if b.result.kind == "record" and total > 1 and len(rows) == 1:
            return "W5", f"one record returned, the witness found {total}"
        keys = self._keys_of(rows, cols, st) if cols else []
        # W2 outside
        if w.key_set is not None and keys:
            # key parts no returned row carries (a tissue id upstream keeps on the enclosing group) are
            # compared on the parts the rows do carry, never as null
            idx = [i for i, c in enumerate(cols) if any(isinstance(r, Mapping) and get_path(r, c) is not None
                                                        for r in rows)]
            pcols = [cols[i] for i in idx]
            ptypes = [st.storage_types.get(c) for c in pcols]
            witness_keys = {canonical([list(k)[i] for i in idx], ptypes) for k in w.key_set}
            pkeys = keys if len(idx) == len(cols) else self._keys_of(rows, pcols, st)
            outside = [k for k in pkeys if k not in witness_keys]
            st.checks.append(("witness_keys", not outside, f"{len(outside)} returned row(s) outside the witness"))
            if outside:
                return "W2", f"{len(outside)} returned row(s) are not in the witness key set (e.g. {outside[0]})"
        # W3 top-k (not inflated)
        if not st.inflated and st.order and w.topk and isinstance(w.topk, list) and keys and \
                st.requested_limit is not None and b.result.order_source == "witness":
            k = min(st.requested_limit, len(w.topk))
            want = [canonical(list(x), [st.storage_types.get(c) for c in cols]) for x in w.topk[:k]]
            got = keys[:k]
            ok = set(got) == set(want) or self._ties_explain(rows[:k], set(got) - set(want), cols, st, contract, t,
                                                             want)
            st.checks.append(("topk_order", ok, None if ok else f"returned {got[:3]}, witness {want[:3]}"))
            if not ok:
                return "W3", "the returned top-k differs from the witness top-k"
            st.order_verified = True
        # W6 short page
        limit = st.sent_limit
        expected = total if limit is None else min(total, limit)
        if returned_raw < expected and b.result.kind != "record":
            return "W6", f"{returned_raw} of {expected} expected rows returned"
        return None

    def _keyed_lookup(self, plan: CallPlan, contract: ToolContract) -> bool:
        """The given arguments bind every key column of the bound table (a lookup by the table's key)."""
        t = contract.tables.get(plan.bound_table or contract.bound_table or "")
        if t is None:
            return False
        bound = {_last(c) for n in contract.args if is_present(plan.args_raw.get(n))
                 for _tb, c in contract.arg_columns(n)}
        return all(_last(k) in bound for k in t.key if not k.endswith("#"))

    def _one_to_many(self, plan: CallPlan, st: _CallState, contract: ToolContract, t: Any) -> None:
        """A record lookup by a non-key (a position without alleles) that matches several records:
        ``ambiguous`` with the matching keys as candidates, so the agent can retry with one (C11)."""
        w = st.witness
        cols = self._key_columns(contract, t)
        keys = [list(k) for k in (w.key_set or [])][: self.settings.resolution.max_candidates] if w else []
        given = [n for n in contract.args if is_present(plan.args_raw.get(n))]
        cands = [{"id": k[0] if len(k) == 1 else k, "label": None, "via": "+".join(given)} for k in keys]
        key_arg = next((n for n in contract.args for _tb, c in contract.arg_columns(n)
                        if cols and _last(c) == _last(cols[0])), None)
        from ..errors import ambiguous_payload
        payload = ambiguous_payload(given[0] if given else "", {n: plan.args_raw.get(n) for n in given}, cands)
        if key_arg:
            payload["disambiguate_with"] = [key_arg]
        raise GatewayError(ErrorKind.ambiguous, f"{' and '.join(given)} match {w.total if w else 'several'} "
                           f"records; call again with one {key_arg or 'key'}", tool=st.name, payload=payload)

    def _parent_counted(self, contract: ToolContract) -> bool:
        """A record read from a nested section of the bound table (``rows: $.tep``, no item table): the
        witness counts parent records, so a null section is no contradiction and the total is the record."""
        r = contract.binding.result
        return r.kind == "record" and not r.rows_of and any(p not in ("$", "") for p in r.row_paths)

    def _ties_explain(self, rows: Sequence[Any], extra: set[str], cols: Sequence[str], st: _CallState,
                      contract: ToolContract, t: Any, want: Sequence[str]) -> bool:
        """Do ties at the top-k boundary explain the difference? Every returned row outside the witness
        top-k must have the rank value of the boundary: the last witness top-k row that was returned."""
        keys = rank_keys(st.order, t, self.registry)
        if not keys or not rows:
            return False
        from ..plugins.statistics import rank_value_of
        rk, plugin, spec = keys[0]
        types = [st.storage_types.get(c) for c in cols]
        by_key = {row_key(r, cols, types): r for r in rows}

        def value(r: Any) -> Any:
            return rank_value_of(plugin, r, rk.column, spec) if plugin is not None else get_path(r, rk.column)

        boundary = next((by_key[k] for k in reversed(list(want)) if k in by_key), None)
        if boundary is None:
            return False
        b = value(boundary)
        return all(value(by_key[k]) == b for k in extra)

    async def _recall(self, plan: CallPlan, st: _CallState, contract: ToolContract) -> DataResult | None:
        """W6: one re-call with the corrected limit."""
        st.attempts += 1
        w = st.witness
        name = contract.limit_arg
        if self.bridge is None or w is None or w.total is None or name is None:
            return None
        target = int(w.total) + int(w.unknown_total or 0)
        if st.sent_limit is not None and target <= st.sent_limit:
            target = st.sent_limit * 2
        if target > self.settings.witness.max_inflate_rows:
            return None
        args = dict(plan.args_sent)
        args[name] = target
        try:
            out = await self.bridge.call_raw(plan.server, plan.tool, args)
        except Exception:  # noqa: BLE001 - a failed re-call keeps the first answer (partial)
            return None
        text = out if isinstance(out, str) else json.dumps(out, default=str)
        st.sent_limit = target
        st.inflated = True
        st.transforms.append(f"short_page_recall limit {target}")
        obj, ok = parse_payload(text, None)
        if not ok:
            return None
        raw = RawResult(text, None, None, "ok")
        return await self._process_upstream(plan, st, contract, obj, raw, Counters())

    async def _contradiction(self, plan: CallPlan, st: _CallState, contract: ToolContract, check: str, returned: int,
                             detail: str) -> DataResult:
        b = contract.binding
        st.checks.append((check, False, detail))
        if b.derived is not None and b.on_contradiction == "derived" and self.profile != "fidelity":
            return await self._repair(plan, st, contract, detail)
        w = st.witness
        witness = {"total": w.total if w is not None else None,
                   "topk": (w.topk[:10] if isinstance(w.topk, list) else None) if w is not None else None}
        raise GatewayError(ErrorKind.tool_defect, f"the tool's answer contradicts the data ({check}: {detail}); "
                           "it was withheld", tool=st.name,
                           payload=tool_defect_payload(check, witness, returned, [d.id for d in b.defects]))

    async def _repair(self, plan: CallPlan, st: _CallState, contract: ToolContract, detail: str) -> DataResult:
        st.notes.append(f"repaired from the data: {detail}")
        obj, rows, resp = await self._serve(plan, st, contract)
        return self._finish_rows(plan, st, contract, obj, rows, Counters(), served_by="repaired", serve=resp)

    # ---------------------------------------------------------------- derived serving

    async def _serve(self, plan: CallPlan, st: _CallState, contract: ToolContract
                     ) -> tuple[Any, list[Any], ServeResponse]:
        b = contract.binding
        d = b.derived
        if d is None:
            raise GatewayError(ErrorKind.tool_defect, "no derived serving declared", tool=st.name)
        within = list((st.decision.per_group if st.decision else []) or [])
        limit = st.requested_limit
        limit_grain = None
        lname = contract.limit_arg
        if lname and contract.args[lname].limit_grain:
            limit_grain = contract.args[lname].limit_grain
        sections: dict[str, dict[str, Any]] = {}
        for sname, sec in d.sections.items():
            key = {col: (self._stored(st.results[arg], sec.table) if arg in st.results else
                         st.values.get(arg, plan.args_sent.get(arg, plan.gateway_args.get(arg))))
                   for col, arg in sec.key_from_args.items()}
            sections[sname] = {"path": sec.path, "table": sec.table, "verb": sec.verb, "single": sec.single,
                               "key": json_value(key), "value": sec.value}
        anchor = None
        anchors = [{"name": arg, "column": column, "value": json_value(value)}
                   for arg, (column, value) in st.anchors.items()]     # every anchor (a pair for a similarity)
        if anchors:
            anchor = {"column": anchors[0]["column"], "value": anchors[0]["value"]}
        pred = _on_items(st.predicate, contract, d.table, plan.bound_table)
        nest = dict(d.nest) if d.nest else None
        if nest is not None:
            # item filters and negation apply to the nested items before groups are counted and cut
            items = nest.get("items") or "items"
            name = items if isinstance(items, str) else str(items.get("name", "items"))
            inner = []
            for arg, p in st.per_arg.items():
                a = contract.args.get(arg)
                split = split_container(p) if a is not None and a.item_filter else None
                if split is not None and split[0].split(".")[-1] == name:
                    inner.append(to_json(split[1]))
            nest["item_filter"] = inner
            nest["include_negated"] = bool(plan.args_raw.get(_INCLUDE_NEGATED) or
                                           plan.gateway_args.get(_INCLUDE_NEGATED))
        # a search tool called with only its key argument (biosample_id, no query) is a lookup by key
        verb = "find" if d.verb == "search" and not st.search_text else d.verb
        req = ServeRequest(table=d.table, verb=verb, predicate=to_json(pred) if pred else None,
                           columns=list(d.columns), order=[RankKeyModel(**_rank_json(o)) for o in st.order],
                           limit=limit, limit_grain=limit_grain, group_by=within + list(d.group_by),
                           explode=list(d.explode), carry=list(d.carry), rename=dict(d.rename), split=d.split,
                           nest=nest, aggregate=dict(d.aggregate), sections=sections, anchor=anchor, anchors=anchors,
                           search_text=st.search_text,
                           params={n: v for n, v in plan.args_raw.items() if isinstance(v, (str, int, float, bool))},
                           budget_bytes=self.settings.witness.repair_max_bytes)
        resp = await self.service.serve(req)
        if resp.error:                                 # the handler refused the request: its own kind, not an outage
            raise GatewayError.from_envelope(resp.error).with_tool(st.name)
        first = resp.rows[0] if isinstance(resp.rows, list) and resp.rows else None
        match = first.get("_match") if isinstance(first, Mapping) else None
        if st.search_text and isinstance(match, Mapping) and match.get("class") != "exact":
            st.notes.append(f"{st.search_text!r} matched the first row by {match.get('rule') or match.get('class')} "
                            f"on {match.get('column')} ({match.get('value')!r})")
        obj: dict[str, Any] = {}
        for k, v in d.envelope.items():
            obj[k] = _template(v, plan.args_raw)
        rows: list[Any] = []
        paths = b.result.row_paths or ["$"]
        if isinstance(resp.rows, dict):
            into = dict((d.split or {}).get("into") or {})
            for part_key, part in resp.rows.items():
                target = str(part_key) if str(part_key).startswith("$") else str(into.get(part_key, f"$.{part_key}"))
                obj = jp_set(obj, target, list(part))
                rows.extend(part)
        else:
            rows = list(resp.rows)
            path = paths[0]
            if path != "$":
                obj = place_rows(obj, path, rows, record=b.result.kind == "record")
            elif b.result.kind == "record":
                obj = {**obj, **rows[0]} if rows and isinstance(rows[0], Mapping) else obj
            elif obj:
                obj["rows"] = rows                     # an envelope is declared: rows under "rows"
            else:
                obj = rows  # type: ignore[assignment]
        for sname, sec in d.sections.items():
            value = resp.sections.get(sname)
            unavailable = st.soft_sections.get(sec.table)
            if unavailable:
                value = {"_vbt_unavailable": unavailable}
            if sec.path and isinstance(obj, dict):
                jp_set(obj, sec.path, value)
        if resp.excluded_unknown:
            st.serve_excluded_unknown = dict(resp.excluded_unknown)
        st.serve_negated = int((resp.sections.get("_excluded") or {}).get("negated") or 0)
        st.transforms.append(f"served {d.verb} on {d.table}")
        return obj, rows, resp

    # ---------------------------------------------------------------- shared tail

    def _finish_rows(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any, rows: list[Any],
                     counters: Counters, *, served_by: str, serve: ServeResponse | None = None,
                     groups: list[tuple[str, list[Any]]] | None = None, mapper: FieldMapper | None = None,
                     record: bool = False, witness: WitnessResponse | None = None) -> DataResult:
        b = contract.binding
        t = self._rows_table(plan, contract)
        cols = self._key_columns(contract, t)
        types = [st.storage_types.get(c) for c in cols]
        mapper = mapper or self._mapper(contract)
        upstream = serve is None
        groups = groups if groups is not None else [(b.result.row_paths[0] if b.result.row_paths else "$", rows)]
        full_rows: list[Any] = []
        new_groups: list[tuple[str, list[Any]]] = []
        truncated = False
        returned_total = 0
        sections_out: dict[str, list[Any]] = {}
        level_grains: dict[str, int] = {}
        if not upstream:
            counters.excluded_negated += st.serve_negated   # groups the derived nest dropped as negated
        for path, grows in groups:
            if not upstream:
                grows = self._derived_items(plan, st, contract, t, grows, counters)
            if upstream:
                # T7 duplicates
                include_dup = bool(plan.args_raw.get(_INCLUDE_DUPLICATES) or plan.gateway_args.get(_INCLUDE_DUPLICATES))
                grows = t7_duplicates(grows, self._qualifier_paths(t, "duplicate") if t is not None else [],
                                      include_dup, counters)
            # T9 levels
            if b.result.levels:
                keys = {lvl: grain_columns(t, lvl) or [f"{lvl}Id"] for lvl in b.result.levels}
                grows, secs, lg = t9_levels(grows, b.result.levels, keys, counters)
                for sec_name, sec_rows in secs.items():
                    sections_out.setdefault(sec_name, []).extend(sec_rows)
                for lvl, n_keys in lg.items():
                    level_grains[lvl] = level_grains.get(lvl, 0) + n_keys
            full_rows.extend(grows)
            # T10 order and cut
            if upstream and (st.inflated or st.order) and not record:
                within = list(st.decision.per_group) if st.decision else []
                lname = contract.limit_arg
                lgrain = contract.args[lname].limit_grain if lname else None
                lcols = grain_columns(t, lgrain) if lgrain else None
                boundary = None
                if b.result.levels:
                    first = next(iter(b.result.levels))
                    boundary = grain_columns(t, first) or [f"{first}Id"]
                limit = st.requested_limit if (st.inflated or (st.requested_limit is not None and
                                                                len(grows) > st.requested_limit)) else None
                if st.order or limit is not None:
                    grows, info = t10_order_cut(grows, rank_keys(st.order, t, self.registry), cols, limit,
                                                within=within, limit_grain=lcols, boundary=boundary,
                                                storage_types=types)
                    truncated = truncated or bool(info.get("truncated"))
                    if info.get("truncated"):
                        st.transforms.append(f"cut {limit}")
            new_groups.append((path, grows))
            returned_total += len(grows)
        rows_out = [r for _, rs in new_groups for r in rs]
        # T8 pooled over
        pooled = t8_pooled_over(rows_out, cols) if upstream else []
        # T13, T14, T12
        if t is not None:
            flags = {}
            for name, c in t.columns.items():
                if is_container(c):
                    for fname, f in c.fields.items():
                        if getattr(f, "role", None) == "flag" and getattr(f, "partition_items", False):
                            flags[name] = fname
            if flags:
                t13_flag_partition(rows_out, flags)
            measures = {n: c for n, c in t.columns.items() if getattr(c, "role", None) == "measure"}
            computed = [n for n, fm in b.result.fields.items() if fm.computed is not None]
            t14_validity(rows_out, measures, counters, computed)
            trims: dict[str, Any] = dict(b.result.trim)
            orders: dict[str, Any] = {}
            item_keys: dict[str, list[str]] = {}
            for name, c in t.columns.items():
                if is_container(c):
                    if c.rank:
                        orders[name] = list(c.rank)
                    if c.item_key is not None:
                        item_keys[name] = list(c.item_key.columns)
                    if record and name not in trims and getattr(c, "role", None) in ("nested", "member"):
                        trims[name] = self.settings.results.relation_list_max
                elif record and getattr(c, "role", None) == "hierarchy" and name not in trims:
                    trims[name] = self.settings.results.relation_list_max
            if trims:
                t12_trim(rows_out, trims, counters, orders=orders, item_keys=item_keys)
        # place rows back (derived rows are already placed by _serve, unless the gateway withheld some)
        if (upstream or st.derived_withheld) and isinstance(obj, (dict, list)):
            for path, grows in new_groups:
                out_rows = mapper.output_rows(grows) if upstream else grows
                if path == "$" and not isinstance(obj, dict):
                    obj = out_rows
                elif path == "$" and record:
                    obj = out_rows[0] if out_rows else obj
                elif path != "$":
                    obj = place_rows(obj, path, out_rows, record=record)
        if isinstance(obj, dict):
            for name, srows in sections_out.items():
                obj[name] = srows
            for sname, sec in b.result.sections.items():
                reason = st.soft_sections.get(sec.table)
                if reason and sec.path and isinstance(obj, dict):
                    jp_set(obj, sec.path, {"_vbt_unavailable": reason})
        # T11 counts, summaries, removed fields, arg echo
        vetoed: dict[str, str] = {}
        for fname, fm in b.result.fields.items():
            if fm.computed and self.registry is not None:
                plugin = self.registry.find("statistic", str(fm.computed.get("statistic", "")))
                fn = getattr(plugin, "vetoed_companions", None) if plugin is not None else None
                if callable(fn):
                    with contextlib.suppress(Exception):
                        for v in fn(None) or ():
                            vetoed[str(v)] = f"vetoed by the {plugin.name} statistic"
        complete = (not upstream) or (st.inflated and not truncated) or (witness is not None and
                                                                         witness.total is not None and
                                                                         witness.total == len(full_rows))
        if isinstance(obj, (dict, list)):
            obj = t11_counts(obj, rows_out, count_fields=b.result.count_fields, summary_fields=b.result.summary_fields,
                             full_rows=full_rows if complete else None, drop_fields=b.result.drop_fields,
                             vetoed=vetoed, arg_echo=b.result.arg_echo, args_raw=plan.args_raw, counters=counters)
        # files
        if b.result.files:
            checks = reconcile(b.result.files, obj, output_dir=self.run.get("mcp_output_dir"))
            for c in checks:
                st.checks.append((c.name, c.ok, c.detail))
            bad = [c for c in checks if c.ok is False]
            if bad:
                raise GatewayError(ErrorKind.tool_defect, f"the returned file failed {bad[0].name}: {bad[0].detail}",
                                   tool=st.name, payload=tool_defect_payload(bad[0].name, {"total": None}, None,
                                                                             [d.id for d in b.defects]))
            self._register_materialized(plan, st, contract, obj)
        # totals
        w = witness if witness is not None else st.witness
        total: int | None
        if serve is not None:
            total, total_method = serve.total, "data_child"
            if total is not None and st.derived_withheld:
                total = max(int(total) - st.derived_withheld, 0)   # parents the item filters or negation emptied
            truncated = bool(serve.truncated) or (total is not None and total > len(rows_out))
        elif w is not None and w.total is not None and w.total_method != "unknown" and \
                not self._parent_counted(contract):
            total, total_method = int(w.total), "witness_scan"
            truncated = truncated or total > len(rows_out)
        elif record:
            total, total_method = len(rows_out), "record"
        else:
            total, total_method = self._upstream_total(obj, b, counters, len(rows_out))
            if total is not None and total > len(rows_out):
                truncated = True
            elif total is None and contract.limit_arg is None and not st.short_page:
                # no limit can cut the answer: every row upstream returned is the whole answer
                total, total_method = len(rows_out), "returned"
        status = "ok" if rows_out else "empty"
        if b.result.kind in ("count", "file"):
            # no rows: a count or a written file; the payload itself is the answer
            if b.result.kind == "count":
                total, total_method = self._upstream_total(obj, b, counters, 0)
            status = "partial" if (st.soft_sections or st.force_partial) else "ok"
            return self._result(plan, st, obj=obj, rows=[], counters=counters, served_by=served_by, total=total,
                                total_method=total_method, status=status, cols=cols, text_rows=True)
        if rows_out and (truncated or (total is None and not record) or st.soft_sections or pooled
                         or st.not_found_items or st.short_page or st.force_partial):
            status = "partial"
        if not rows_out:
            proven = serve is not None or (w is not None and w.total is not None and w.total_method != "unknown")
            if any(v == "unknown" for v in plan.existence.values()) or not proven:
                status = "empty_unverified"
        return self._result(plan, st, obj=obj, rows=rows_out, counters=counters, served_by=served_by, total=total,
                            total_method=total_method, status=status, truncated=truncated, pooled=pooled,
                            serve=serve, level_grains=level_grains, cols=cols)

    def _upstream_total(self, obj: Any, b: Any, counters: Counters, returned: int) -> tuple[int | None, str]:
        spec = b.result.total
        if spec is None:
            return None, "unknown"
        path = spec if isinstance(spec, str) else spec.path
        val = jp_first(obj, path)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return None, "unknown"
        if not isinstance(spec, str):
            for cond in spec.valid_unless:
                with contextlib.suppress(Exception):
                    from .fields import jp_test
                    if jp_test(obj, cond):
                        return None, "unknown"
        dropped = sum(counters.excluded.values()) + sum(counters.excluded_unknown.values()) + \
            sum(counters.withheld.values())
        method = "upstream_upper_bound" if dropped or (not isinstance(spec, str) and
                                                       spec.method == "upstream_upper_bound") else "upstream"
        return int(val), method

    def _register_materialized(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any) -> None:
        name = f"{plan.server}.{plan.tool}"
        for src, desc in self.catalog.sources.items():
            for tname, spec in desc.tables.items():
                mb = spec.materialized_by
                if mb is None or mb.tool not in (name, contract.alias_of):
                    continue
                path = jp_first(obj, mb.path_from)
                if path:
                    p = Path(str(path))
                    if not p.is_absolute() and self.run.get("mcp_output_dir"):
                        p = Path(str(self.run["mcp_output_dir"])) / p
                    entry = self.materialized.register(f"{src}.{tname}", p, prov=None, tool=st.name)
                    st.materialized.append(entry)

    # ---------------------------------------------------------------- generic

    def _finish_generic(self, plan: CallPlan, st: _CallState, raw: RawResult | None) -> DataResult:
        if raw is None:
            raise GatewayError(ErrorKind.source_error, "no upstream result", tool=st.name)
        cls = classify(raw, plan.contract, plan, registry=self.registry)
        if cls.outcome in ("oom", "not_found", "source_error"):
            raise cls.error  # type: ignore[misc]
        status = "empty_unverified" if cls.outcome == "empty_unverified" else "ok"
        if status == "empty_unverified" and self.settings.gateway.unbound_empty == "error":
            raise GatewayError(ErrorKind.not_found, "the tool returned nothing and cannot be checked", tool=st.name)
        self._stamp_times(st)
        prov = DataProvenance(tool=st.name, server=plan.server, mode=st.mode, profile=self.profile,
                              gateway_version=GATEWAY_VERSION, served_by="upstream",
                              request=RequestInfo(args_raw=dict(plan.args_raw), args_sent=dict(plan.args_sent)),
                              result=ResultInfo(status=status, coverage="unknown"),
                              memory=st.admission.to_record() if st.admission is not None
                              else {"admission": "not_applicable"},
                              upstream=self._upstream_info(plan), t_ms=dict(st.t_ms))
        prov.result.output_sha256 = hashlib.sha256((raw.text or "").encode("utf-8")).hexdigest()
        prov.tool_use_id = getattr(getattr(plan, "_vbt_ctx", None), "tool_call_id", None) or None
        prov.finalize()
        header = Header(status=status, served_by="upstream", notes=st.notes,  # type: ignore[arg-type]
                        cite="not citable" if status == "empty_unverified" else None, prov=prov.id)
        if raw.parts is not None:
            return DataResult(obj=raw.parts, text=raw.text, status=status, provenance=prov,  # type: ignore[arg-type]
                              full_text=raw.text, header=header.to_dict())
        obj = cls.obj if cls.is_json else (raw.text or "")
        return DataResult.build(obj, header, prov)

    # ---------------------------------------------------------------- result, header, record

    def _coverage(self, t: Any, status: str, counters: Counters, in_universe: bool | None, *,
                  extra_unknown: int = 0) -> tuple[str, str | None]:
        cov = getattr(getattr(t, "spec", None), "coverage", None) if t is not None else None
        if cov is None:
            return "unknown", None
        statement = cov.statement
        if cov.absence_means == "unknown":
            return "unknown", statement
        if cov.absence_means == "censored":
            return "censored", statement
        excluded = max(sum(counters.excluded_unknown.values()), extra_unknown) + counters.excluded_negated + \
            counters.items_removed.get("negated", 0)
        if cov.universe is not None:
            if in_universe is False:
                return "not_covered", statement
            if in_universe is None:
                return "unknown", statement
        if excluded:
            return "partial_unknown", statement
        return "covered", statement

    async def _null_container(self, st: _CallState, contract: ToolContract) -> bool:
        """For rows that are items of a list (``rows_of``) declared ``null_means: unknown``: is the list
        null in the matching parent row? Then an empty answer means not assessed, not none listed."""
        b = contract.binding
        it = contract.tables.get(b.result.rows_of) if b.result.rows_of else None
        if it is None or not it.is_item_table or st.predicate is None or not it.items_path:
            return False
        if getattr(it.container, "null_means", None) not in ("unknown", "not_assessed"):
            return False
        path = it.items_path[:-2] if it.items_path.endswith("[]") else it.items_path
        n = await self._count(str(it.physical), And((st.predicate, IsNull(path))), st)
        return bool(n)

    async def _in_coverage_universe(self, st: _CallState, t: Any) -> bool | None:
        cov = getattr(getattr(t, "spec", None), "coverage", None) if t is not None else None
        if cov is None or cov.universe is None:
            return None
        u = cov.universe
        table = u.table if "." in u.table else f"{t.ref.source}.{u.table}"
        value = next((v for v in st.values.values() if v is not None and not isinstance(v, list)), None)
        if value is None or not u.keys:
            return None
        n = await self._count(table, Eq(u.keys[0], value), st)
        return None if n is None else n > 0

    def _result(self, plan: CallPlan, st: _CallState, *, obj: Any, rows: list[Any], counters: Counters,
                served_by: str, total: int | None, total_method: str, status: str, truncated: bool = False,
                pooled: Sequence[str] = (), serve: ServeResponse | None = None,
                level_grains: Mapping[str, int] | None = None, cols: Sequence[str] | None = None,
                text_rows: bool = False) -> DataResult:
        contract: ToolContract = plan.contract
        b = contract.binding
        t = self._rows_table(plan, contract)
        cols = list(cols if cols is not None else self._key_columns(contract, t))
        types = [st.storage_types.get(c) for c in cols]
        desc = t.descriptor if t is not None else None
        release = desc.release.expect if desc is not None else None
        source = f"{desc.source}@{release}" if desc is not None and release else (desc.source if desc else None)
        excluded_unknown = dict(counters.excluded_unknown)
        w = st.witness
        if w is not None:
            for k, v in (w.excluded_unknown or {}).items():
                if k != "_rows":
                    excluded_unknown[k] = max(int(v), excluded_unknown.get(k, 0))
        for k, v in st.serve_excluded_unknown.items():
            if k != "_rows":
                excluded_unknown[k] = max(int(v), excluded_unknown.get(k, 0))
        in_universe = st.in_universe
        # rows the witness or the data child excluded as unknown count like the transforms' own (partial_unknown)
        coverage, statement = self._coverage(t, status, counters, in_universe,
                                             extra_unknown=sum(excluded_unknown.values()))
        if b.result.parent_key and not b.result.rows_of:
            # items of a container with no item table: the table's coverage is not theirs, none is declared
            coverage, statement = "unknown", None
        text = [n for n, a in contract.args.items() if a.role == "free_text" and is_present(plan.args_raw.get(n))
                and a.interpreted_as != "exact"]
        if text and status == "empty" and coverage in ("covered", "partial_unknown"):
            # a text miss shows that no stored string matched, not that no such entity exists
            coverage = "unknown"
            st.notes.append(f"no stored text matched {', '.join(text)}; a text miss is not evidence of absence")
        if st.null_container and status == "empty" and coverage != "not_covered":
            coverage = "unknown"
            st.notes.append("the list is null in the source (not assessed); empty does not mean none listed")
        if t is not None and t.spec.coverage is not None and t.spec.coverage.universe is not None and \
                in_universe is None and status == "empty":
            st.notes.append("coverage universe not checked")
        excluded_na = dict(counters.excluded_not_applicable)
        if w is not None:
            for k, v in (w.excluded_not_applicable or {}).items():
                excluded_na[k] = max(int(v), excluded_na.get(k, 0))
        grains: dict[str, dict[str, int | None]] = {}
        if t is not None:
            for g, spec in t.spec.grains.items():
                gcols = list(spec) if isinstance(spec, list) else list(spec.columns or [])
                if not gcols:
                    continue
                returned_g = len({canonical([get_path(r, c) for c in gcols]) for r in rows if isinstance(r, Mapping)})
                total_g = None
                if serve is not None and g in serve.grains:
                    total_g = serve.grains[g].get("total")
                elif w is not None and g in (w.distinct_counts or {}):
                    total_g = int(w.distinct_counts[g])
                grains[g] = {"returned": returned_g, "total": total_g}
        for lvl, n in (level_grains or {}).items():
            grains[lvl] = {"returned": n, "total": n if total is not None and total == len(rows) else None}
        order_text = None
        verified = st.order_verified
        if st.order:
            o = _rank_json(st.order[0])
            within = (st.decision.per_group if st.decision else []) or list(o.get("within") or [])
            how = {"witness": "verified" if (st.inflated or verified or serve is not None) else "unverified",
                   "upstream_full_sort": "upstream full sort", "source_server_side": "ranked by source, not verified"
                   }[b.result.order_source]
            order_text = f"{o['column']} {o.get('direction', 'desc')}" + \
                (f" within {', '.join(within)}" if within else "") + f" ({how})"
            if st.inflated or serve is not None:
                verified = True
        returned = len(rows) if not text_rows else None
        if b is not None and b.result.kind == "record" and rows:
            returned = len(rows)
        cite = {"ok": "cite rows by key", "partial": (f"partial: cite as 'top {returned} of {total}'" if total
                                                      else "partial: cite only the rows shown"),
                "empty": ("citable only as an absence (supports: absence)" if coverage == "covered" else
                          "not citable as support; absence claim requires covered coverage"),
                "empty_unverified": "not citable"}.get(status)
        evidence = t.spec.evidence_nature if t is not None else None
        withheld = dict(counters.withheld)
        if st.soma_added:
            withheld.setdefault("duplicates", 0)
        notes = list(dict.fromkeys(st.notes + counters.notes))
        if counters.excluded:
            notes.append("rows failing a bound argument removed: " + ", ".join(f"{k} ({v})"
                                                                                for k, v in counters.excluded.items()))
        if counters.dropped_parents:
            notes.append(f"{counters.dropped_parents} record(s) left without qualifying items removed")
        if counters.items_removed.get("negated"):
            notes.append(f"{counters.items_removed['negated']} negated item(s) removed (pass include_negated=true "
                         "to keep them)")
        records = _derived_records(serve)
        if records:
            notes.append("the derived " + ", ".join(sorted(records)) + " record(s) are in this call's provenance")
        # provenance
        prov = self._record(plan, st, contract, t, status=status, rows=rows, cols=cols, types=types, total=total,
                            total_method=total_method, truncated=truncated, coverage=coverage, statement=statement,
                            served_by=served_by, order=st.order[0] if st.order else None, verified=verified,
                            obj=obj, withheld_leakage=counters.withheld.get("leakage", 0),
                            leakage_unchecked=counters.leakage_unchecked,
                            derived=records)
        for entry in st.materialized:
            entry["prov"] = prov.id
        header = Header(
            status=status, source=source, tables=[_last(str(t.ref))] if t is not None else None,  # type: ignore
            key=cols or None, returned=returned, total=total, total_method=total_method,
            truncated=truncated if not text_rows else None, order=order_text, grains=grains or None,
            scope=json_value(plan.scope) or None, resolved=st.resolved or None,
            resolution_summary=st.resolution_summary or None, excluded_unknown=excluded_unknown or None,
            excluded_not_applicable=excluded_na or None, excluded_negated=counters.excluded_negated or None,
            excluded=dict(counters.excluded) or None, withheld=withheld or None,
            not_found_items=st.not_found_items, pooled_over=list(pooled) or None,
            removed_fields=dict(counters.removed_fields) or None, trimmed=dict(counters.trimmed) or None,
            undefined=(next(iter(st.undefined.values())) if st.undefined else None),
            coverage=coverage,  # type: ignore[arg-type]
            coverage_statement=statement, evidence=evidence.caveat if evidence is not None else None,
            served_by=served_by, hash_seed=self._hash_seed(plan.server), notes=notes, cite=cite, prov=prov.id)
        if st.soft_sections:
            header.extra["unavailable_sections"] = dict(st.soft_sections)
        return DataResult.build(obj, header, prov)

    def _record(self, plan: CallPlan, st: _CallState, contract: ToolContract, t: Any, *, status: str,
                rows: Sequence[Any], cols: Sequence[str], types: Sequence[str | None], total: int | None,
                total_method: str, truncated: bool, coverage: str, statement: str | None, served_by: str,
                order: Any, verified: bool | None, obj: Any, withheld_leakage: int = 0,
                leakage_unchecked: int = 0, derived: dict[str, Any] | None = None) -> DataProvenance:
        b = contract.binding
        desc = t.descriptor if t is not None else None
        self._stamp_times(st)
        tables = []
        for ref in tables_read(contract, plan.bound_table, plan.args_raw):
            ct = contract.tables.get(ref)
            phys, _ = self.readiness.physical(ref)
            m = self.readiness.get(phys)
            rs = b.reads.get(ref) if b is not None else None
            access = ("derived_scan" if plan.route == "derived" else
                      f"upstream_{rs.access}" if rs is not None else "witness_scan")
            tables.append(TableInfo(name=_last(ref), layout=ct.layout if ct is not None else None,
                                    fingerprint=m.fingerprint if m is not None else None, access=access,
                                    lineage=(ct.spec.lineage.model_dump() if ct is not None and ct.spec.lineage
                                             else None)))
        o = _rank_json(order) if order is not None else {}
        prov = DataProvenance(
            tool=st.name, server=plan.server, mode=st.mode, profile=self.profile, gateway_version=GATEWAY_VERSION,
            served_by=served_by,
            source=SourceInfo(name=desc.source if desc else None, release=desc.release.expect if desc else None,
                              descriptor_sha256=self._safe_digest(desc.source) if desc else None,
                              overlay_sha256=self.catalog.overlay_digest(plan.server),
                              serving_source=str(t.served_from) if t is not None and t.served_from else None),
            plugins=self._plugins_used(contract, t), tables=tables,
            request=RequestInfo(args_raw=json_value(dict(plan.args_raw)), args_sent=json_value(dict(plan.args_sent)),
                                resolutions=[_resolution_record(r) for r in plan.resolutions],
                                scope=json_value(dict(plan.scope))),
            result=ResultInfo(status=status, returned=len(rows), total=total, total_method=total_method,
                              coverage=coverage, coverage_statement=statement, truncated=truncated,
                              order=OrderInfo(by=o.get("column"), direction=o.get("direction"),
                                              within=list((st.decision.per_group if st.decision else []) or []),
                                              verified=verified,
                                              source=b.result.order_source if b is not None else None)
                              if order is not None else None,
                              key_columns=list(cols), transforms=list(st.transforms)),
            memory=st.admission.to_record() if st.admission is not None else {"admission": "not_applicable"},
            leakage=leakage_record(st.leakage.ceiling, withheld_leakage, st.leakage.risk or leakage_unchecked > 0)
            if st.leakage is not None and st.leakage.active else None,
            evidence_nature=(t.spec.evidence_nature.model_dump() if t is not None and t.spec.evidence_nature
                             else None),
            upstream=self._upstream_info(plan), t_ms=dict(st.t_ms), derived=derived or None)
        for name, ok, detail in st.checks:
            prov.add_check(name, ok, detail)
        if rows and cols:
            prov.set_row_keys([[get_path(r, c) for c in cols] for r in rows if isinstance(r, Mapping)],
                              self.settings.provenance.row_keys_max, types)
        text = json.dumps(obj, default=str, sort_keys=True, ensure_ascii=False)
        prov.result.output_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        prov.result.output_rows_sha256 = hashlib.sha256(json.dumps(
            rows, default=str, sort_keys=True).encode("utf-8")).hexdigest()
        prov.tool_use_id = getattr(getattr(plan, "_vbt_ctx", None), "tool_call_id", None) or None
        return prov.finalize()

    def _stamp_times(self, st: _CallState) -> None:
        """``t_ms`` {prepare, call, finish, total} as of building the record (§15.1)."""
        if st.t_finish is not None:
            st.t_ms["finish"] = _ms(st.t_finish)
        st.t_ms["total"] = _ms(st.t0)

    def _upstream_info(self, plan: CallPlan) -> UpstreamInfo:
        reviewed = getattr(getattr(plan.contract, "overlay", None), "upstream_commit", None) \
            if plan.contract is not None else None
        served = self._served_commit()
        if served and reviewed and served != reviewed and plan.server not in self._commit_warned:
            self._commit_warned.add(plan.server)
            log.warning("upstream %s is at %s but its overlay was reviewed against %s: the bindings may not "
                        "match the code that answers", plan.server, served[:12], str(reviewed)[:12])
        return UpstreamInfo(commit=served or reviewed, server=plan.server, hash_seed=self._hash_seed(plan.server),
                            flags_stripped=self._flags_stripped(plan.server), reviewed_commit=reviewed)

    def _served_commit(self) -> str | None:
        """The commit of the upstream checkout that serves the calls (``vars.upstream``), read once."""
        if self._upstream_commit is None:
            commit = ""
            with contextlib.suppress(Exception):
                from ...pinning import git_info
                upstream = variables_from_config(self.config).get("upstream")
                commit = git_info(upstream).get("upstream_commit") or ""
            self._upstream_commit = commit
        return self._upstream_commit or None

    def _safe_digest(self, source: str) -> str | None:
        try:
            return self.catalog.descriptor_digest(source)
        except Exception:  # noqa: BLE001
            return None

    def _plugins_used(self, contract: ToolContract, t: Any) -> dict[str, str]:
        if self.registry is None:
            return {}
        out: dict[str, str] = {}
        if t is not None:
            for kind, name in (("format", t.format), ("layout", t.layout)):
                p = self.registry.find(kind, name) if name else None
                if p is not None:
                    out[f"{kind}/{name}"] = str(p.version)
        for a in contract.args.values():
            for k in a.accepts:
                try:
                    _, spec = self.catalog.id_type(k, t.ref.source if t is not None else None)
                except Exception:  # noqa: BLE001
                    continue
                p = self.registry.find("identifier", spec.plugin)
                if p is not None:
                    out[f"identifier/{spec.plugin}"] = str(p.version)
        return dict(sorted(out.items()))

    def _status_of(self, server: str) -> dict[str, Any] | None:
        if self.bridge is None:
            return None
        try:
            path = (self.bridge.status().get(server) or {}).get("status_file")
        except Exception:  # noqa: BLE001
            return None
        return read_status(path)

    def _hash_seed(self, server: str) -> int | None:
        s = self._status_of(server)
        v = s.get("hash_seed") if s else None
        return int(v) if isinstance(v, int) else None

    def _flags_stripped(self, server: str) -> list[str]:
        s = self._status_of(server)
        v = s.get("flags_stripped") if s else None
        return list(v) if isinstance(v, list) else []

    # ================================================================== pinning

    def pinned(self) -> dict[str, Any]:
        """``pinned["data"]`` (§15.5): catalog digests, plugins, sources and table fingerprints,
        readiness, memory limits, determinism (hash seeds from the reaper status files) and leakage."""
        descriptors = {s: self._safe_digest(s) for s in sorted(self.catalog.sources)}
        overlays = {s: self.catalog.overlay_digest(s) for s in self.catalog.servers()}
        sources: dict[str, Any] = {}
        for s, d in sorted(self.catalog.sources.items()):
            tables = {ref.split(".", 1)[1]: m.fingerprint for ref, m in sorted(self.readiness.tables.items())
                      if ref.split(".")[0] == s}
            sources[s] = {"release": d.release.expect, "tables": tables}
        columns_not_ready: dict[str, str] = {}
        for ref, m in self.readiness.tables.items():
            for col, col_status in m.columns.items():
                if col_status not in ("ready", "awaiting_producer", "unbound"):
                    columns_not_ready[f"{ref}.{col}"] = col_status
        seeds: dict[str, Any] = {}
        stripped: set[str] = set()
        limits: dict[str, Any] = {}
        if self.bridge is not None:
            with contextlib.suppress(Exception):
                for server in self.bridge.status():
                    status = self._status_of(server)
                    if status:
                        seeds[server] = status.get("hash_seed")
                        stripped.update(status.get("flags_stripped") or [])
                        limits[server] = status.get("limit_mb")
        values = {v for v in seeds.values() if v is not None}
        hash_seed = next(iter(values)) if len(values) == 1 else (None if not values else sorted(values))
        try:
            plugins = self.registry.versions() if self.registry is not None else {}
        except Exception:  # noqa: BLE001
            plugins = {}
        return {
            "mode": self.mode, "profile": self.profile, "gateway_version": GATEWAY_VERSION,
            "catalog_sha256": self.catalog.digest(), "descriptors": descriptors, "overlays": overlays,
            "plugins": plugins, "sources": sources,
            "readiness": {"tools_not_ready": self.degraded_tools(), "columns_not_ready": columns_not_ready,
                          "hash_randomization": self.readiness.hash_randomization},
            "memory": {"server_limits_mb": limits, "containment": self.settings.memory.limit_kind},
            "determinism": {"hash_seed": hash_seed, "per_server": seeds, "flags_stripped": sorted(stripped)},
            "leakage": {"ceiling": self._ceiling.isoformat() if self._ceiling else None},
        }


# ====================================================================== helpers

class _OneTool:
    """A catalog view with one tool, for per-tool readiness in listings."""

    def __init__(self, catalog: Catalog, server: str, tool: str) -> None:
        self._c, self._server, self._tool = catalog, server, tool

    def servers(self) -> list[str]:
        return [self._server]

    def tools(self, server: str) -> list[str]:
        return [self._tool]

    def contract(self, server: str, tool: str) -> ToolContract:
        return self._c.contract(server, tool)


def _vocab_request(table: str, column: str) -> Any:
    from ..ipc import VocabRequest
    return VocabRequest(table=table, column=column)


def _rank_json(o: Any) -> dict[str, Any]:
    if isinstance(o, Mapping):
        out = {k: o[k] for k in ("column", "direction", "nulls", "statistic", "within") if o.get(k) is not None}
    else:
        out = {"column": o.column, "direction": o.direction, "nulls": o.nulls}
        if getattr(o, "statistic", None):
            out["statistic"] = o.statistic
        if getattr(o, "within", None):
            out["within"] = list(o.within)
    out.setdefault("direction", "desc")
    return out


#: ServeResponse sections that carry a derived handler's record (§10.6, F16, F18): kept in provenance.
DERIVED_RECORDS = ("_expansion", "_propagation", "_statistics", "_network", "_essentiality", "_specificity",
                   "_selectivity")


def _derived_records(serve: ServeResponse | None) -> dict[str, Any]:
    """``{expansion: ..., statistics: ...}``: the records a derived handler returned beside its rows."""
    if serve is None:
        return {}
    return {k[1:]: json_value(v) for k, v in serve.sections.items() if k in DERIVED_RECORDS and v}


def _record(res: Any, arg: str) -> dict[str, Any]:
    r = res.record(arg)
    return {k: json_value(v) for k, v in r.__dict__.items()}


def _resolution_record(d: Mapping[str, Any]) -> Any:
    from ..record import ResolutionRecord
    fields_ = ResolutionRecord.__dataclass_fields__
    return ResolutionRecord(**{k: v for k, v in d.items() if k in fields_})


def _template(value: Any, args: Mapping[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("{") and value.endswith("}") and value[1:-1] in args:
        return args[value[1:-1]]
    if isinstance(value, str):
        try:
            return value.format(**{k: v for k, v in args.items()})
        except (KeyError, IndexError, ValueError):
            return value
    return value


def _phantom_label(row: Any, contract: ToolContract) -> Any:
    if not isinstance(row, Mapping):
        return row
    rk = contract.binding.result.row_key if contract.binding is not None else None
    if isinstance(rk, list) and rk:
        return get_path(row, rk[0][2:] if rk[0].startswith("$.") else rk[0])
    return next((v for v in row.values() if isinstance(v, (str, int))), None)


def _on_items(pred: Predicate | None, contract: ToolContract, table: str, bound: str | None) -> Predicate | None:
    """``pred`` (over the bound table's columns) for reading ``table``: when ``table`` is an item table of
    the bound table, a bare name there is the item's field, so bound columns become parent paths ``/c``
    (§6.4)."""
    t = contract.tables.get(table)
    bound = bound or contract.bound_table
    if pred is None or t is None or not t.is_item_table or table == bound or str(t.physical) != bound:
        return pred
    return map_columns(pred, lambda c: c if c.startswith(("/", "^")) else "/" + c)


def build_gateway(config: Mapping[str, Any] | None = None, run: Mapping[str, Any] | None = None, *,
                  registry: Any = None, service: ServiceClient | None = None) -> DataGateway:
    """Wire a :class:`DataGateway` from a loaded config: settings, plugin registry, catalog (with
    ``data.sources.alias``), sidecar index store, resolver, memory admission, readiness cache and
    the data-child client."""
    from ..plugins.registry import discover
    config = dict(config or {})
    settings = DataSettings.from_config(config)
    if registry is None:
        registry = discover(settings)
    catalog = build_catalog(settings, registry, variables=variables_from_config(config), run=run)
    return DataGateway(settings, catalog, registry, service=service, run=run, config=config)
