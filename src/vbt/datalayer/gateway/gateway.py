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
Observe mode adds no call to the data child, the source or the watched server in the call path (no
index build, vocabulary or SOMA vocabulary fetch, witness or existence count, remote resolution) and
renames no file. Profile ``fidelity`` serves derived tools ``pass`` behind the witness and refuses only
on a contradiction (repairs become errors).

A tool that depends on a quarantined catalog file (R8: an overlay or descriptor that does not load,
``ToolContract.quarantined``) is refused ``quarantined`` (subkind ``catalog_file``, naming the file and
its error) under ``when_service_down: strict``; ``lenient`` runs it unguarded. Every other tool is
served as usual.

Live and remote sources (round 3):

* ``count_first.sample`` (``{grain, stratify, seed, columns_arg, total_path, max_read}``, read from the binding
  when its model carries it) serves a Census pull as the derived sample: the data child draws upstream's
  cell-type-stratified, donor-balanced sample with donors keyed by the descriptor grain (``(dataset_id,
  donor_id)``), the unmodified server is asked for exactly those cells (``soma_joinid in [...]``,
  ``max_cells`` = the sample's size), the columns argument is completed with the keys, the payload's total is
  the counted one, ``derived_sample`` describes the draw, and the written file must hold the sample
  (``sample_cells``). Without a sample the call is refused unless the filter fixes the donor key's qualifier.
* ``derived.compose`` joins sub-reads (each on its own table's key) into the derived rows, sections over live
  tables are read here, and keys the call names that a live table's derived rows lack are ``not_found``
  (get_clinical_data served from cBioPortal's live tables). A derived handler's typed error arrives as
  ``ServeResponse.error`` (on the wire ``refusal``: MCPBridge reads a top-level ``error`` key as a failure).
* every enforced call of a live source records what the source reported (``_live_release``): the data release,
  the API and software versions (``source.versions``) and the per-record releases (record versions).
* under the evidence ceiling an upstream count (first posting bounded) also gets the count with the last update
  bounded (``_vbt.ceiling_totals``): the rows a find may return under ``rows: withhold``.
* a SOMA filter's key column (``soma_joinid``) is never resolved against a vocabulary, and "No cells found" with
  a count-first count of 0 is an empty answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...config import base_tool_env
from ..api import CallPlan, CrashDecision, LaunchSpec, ListingDecision, RawResult
from ..catalog import Catalog, CatalogError, ToolContract, build_catalog
from ..derive.tools import WHERE_POINTER
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
from ..ipc import (
    VERB_CENSUS_COUNT,
    VERB_RELEASE,
    CensusCountRequest,
    RankKeyModel,
    ReleaseRequest,
    ServeRequest,
    ServeResponse,
    WitnessRequest,
    WitnessResponse,
)
from ..launch import build_launch_spec
from ..memory import AdmissionController, MemoryEstimator, ResidencyLedger, TableRead, crash_decision, read_status
from ..memory.host import host_budget_mb
from ..memory.sizing import is_auto
from ..predicate import And, Cmp, Contains, Eq, In, IsNull, Not, Predicate, TextMatch, map_columns, to_json
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
from .fields import (FieldMapper, extract_rows, get_path, grain_values, jp_first, jp_get, jp_set, jp_test,
                     parse_payload, place_rows)
from .files import MaterializedRegistry, reconcile
from .leakage import LeakagePlan, ceiling_of, leakage_record, prepare_leakage
from .readiness import (
    ReadinessCache,
    call_readiness,
    degraded_tools,
    derived_dependencies,
    partitions_selected,
    tables_read,
)
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
    trim_column,
)

__all__ = ["DataGateway", "build_gateway", "GATEWAY_VERSION", "compact_native_listing", "WHERE_POINTER"]

log = logging.getLogger(__name__)

GATEWAY_VERSION = "1.0"
_STATE = "_vbt_state"
_INCLUDE_NEGATED = "include_negated"
_INCLUDE_DUPLICATES = "include_duplicates"
_SERVICE_SCRIPT = ("src", "vbt", "datalayer", "service", "server.py")
#: Seconds before a resolver index build that failed on an outage (not a rejection) is tried again.
INDEX_RETRY_S = 60.0
#: Family members whose rows a ``family: exact`` call counts (``_vbt.family_rows``) at most.
MAX_FAMILY_COUNTS = 10
#: Seconds a failed readiness check of the same tables answers False without asking the child again.
CHECK_FAILURE_TTL_S = 5.0
#: Set while an observe-mode call is prepared: the resolver's remote questions are not asked in its path.
_OBSERVING: contextvars.ContextVar[bool] = contextvars.ContextVar("vbt_gateway_observing", default=False)
QUARANTINE_INSTRUCTION = ("This tool is unavailable: a data-layer configuration file it depends on does not load and "
                          "is quarantined until it is fixed (`vbt ds lint` names the file and the error). This call "
                          "did not produce evidence; report the outage and do not treat it as a negative result.")


def quarantine_error(contract: ToolContract, tool: str | None = None) -> GatewayError:
    """The ``quarantined`` error of a tool that depends on a catalog file that does not load (R8)."""
    files = [q.to_json() for q in contract.quarantined]
    first = contract.quarantined[0]
    what = "its server's overlay" if first.kind == "overlay" else f"the {first.kind} {first.name or first.file}"
    return GatewayError(ErrorKind.quarantined,
                        f"{contract.name} is quarantined: {what} ({first.path}) does not load: {first.summary}",
                        tool=tool or f"mcp__{contract.server}__{contract.tool}", subkind="catalog_file",
                        instruction=QUARANTINE_INSTRUCTION,
                        payload={"reason": f"catalog file does not load: {contract.quarantine_reason}",
                                 "alternatives": [], "files": files})


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
    section_meta: dict[str, dict[str, Any]] = field(default_factory=dict)   # derived sections' own status
    count_first: dict[str, Any] | None = None                                # the count-first admission
    sample: dict[str, Any] | None = None                                     # the derived sample upstream fetched
    storage_types: dict[str, str | None] = field(default_factory=dict)
    attempts: int = 0
    t_ms: dict[str, float] = field(default_factory=dict)
    t_prepared: float | None = None                                 # end of prepare (the upstream call starts)
    t_finish: float | None = None                                   # start of finish
    soma_added: bool = False
    native_header: dict[str, Any] | None = None                     # a native data tool: the child's own _vbt
    soma_predicate: Predicate | None = None
    census_release: str | None = None                               # the dated release a SOMA alias names
    live_release: dict[str, dict[str, Any]] = field(default_factory=dict)   # source -> {resolved, versions, per}
    remote_parts: list[Any] = field(default_factory=list)          # the conjuncts the remote witness counted
    ceiling_totals: dict[str, Any] | None = None                    # an upstream count's totals under the ceiling
    resolved_columns: dict[str, str] = field(default_factory=dict)   # identifier argument -> bound column
    force_partial: bool = False
    in_universe: bool | None = None
    null_container: bool = False                                    # the rows' list is null (not assessed)
    derived_withheld: int = 0                                       # derived rows dropped by T4/T6 in the gateway
    engine_matched: bool = False                                    # the source's engine matches an argument
    engine_text: dict[str, str] = field(default_factory=dict)       # engine_param -> the text sent upstream
    serve_as_of: str | None = None                                  # derived rows of a live table: its release
    order_verified: bool | None = None
    rows_path: str | None = None                       # the rows path a derived reply used (rows_when)
    redirected: bool = False                                        # an echo accepted a redirected record
    listed_checked: bool = False                                    # result.not_found_list was read
    self_counted: bool = False                                      # result.withheld_list was counted
    not_found_items: list[Any] | None = None
    short_page: str | None = None
    serve_excluded_unknown: dict[str, int] = field(default_factory=dict)
    serve_negated: int = 0                                          # groups the data child dropped as negated
    materialized: list[dict[str, Any]] = field(default_factory=list)
    family_rows: dict[str, int] = field(default_factory=dict)     # rows stored under other family members


def _state(plan: CallPlan) -> _CallState:
    st = getattr(plan, _STATE, None)
    if st is None:
        st = _CallState(name=f"mcp__{plan.server}__{plan.tool}", t0=time.monotonic())
        setattr(plan, _STATE, st)
    return st


def _ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000.0, 1)



def _sample_alternative(contract: Any) -> str | None:
    """The tool the overlay's ``count_first.sample.alternative`` names for a refused sample, else none."""
    cf = getattr(getattr(contract, "binding", None), "count_first", None)
    sample = getattr(cf, "sample", None) if cf is not None else None
    return getattr(sample, "alternative", None) if sample is not None else None

def _last(path: str) -> str:
    return str(path).lstrip("/").split(".")[-1].replace("[]", "")


def _one_source(contract: Any) -> Any:
    """The descriptor of the one source every table a tool reads belongs to (a record over several tables of one
    source, such as get_census_info: its release is that source's), else None."""
    descs = {id(t.descriptor): t.descriptor for t in (getattr(contract, "tables", None) or {}).values()}
    return next(iter(descs.values())) if len(descs) == 1 else None


def _record_versions(t: Any, rows: Sequence[Any], cols: Sequence[str]) -> dict[str, dict[str, str]] | None:
    """Live record versions ``{source.table: {key: version}}`` of the returned rows, for a table whose key
    declares a ``version`` column (replay reports rows whose version moved as ``source_updated``)."""
    try:
        version = t.spec.key.version
    except AttributeError:   # no table, or a key without a record-version column
        return None
    if not version or len(cols) != 1:
        return None
    out: dict[str, str] = {}
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        k = get_path(r, cols[0])
        v = get_path(r, version)
        if v is None:
            v = get_path(r, _last(version))
        if k not in (None, "") and v not in (None, ""):
            out[str(k)] = str(v)
    return {str(t.physical): out} if out else None



def _nested_coverage(column: Any, table_coverage: str) -> tuple[str, str | None]:
    """Coverage of a record read from a nested column: its own declared coverage, else unknown."""
    cov = getattr(column, "coverage", None)
    if cov is None or getattr(column, "null_means", None) in ("unknown", "not_assessed"):
        return "unknown", getattr(cov, "statement", None)
    if cov.absence_means == "absent":
        return ("covered" if table_coverage in ("covered", "partial_unknown") else table_coverage), cov.statement
    return ("censored" if cov.absence_means == "censored" else "unknown"), cov.statement

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
        if admission is None and host_budget_mb(settings) is not None:
            # data.memory.host_budget_mb (auto: a share of host RAM) caps resident memory over all servers;
            # bind_bridge wires its LRU idle recycle (§14.3)
            self.admission.enable_host_budget()
        self.readiness = readiness or ReadinessCache(settings.cache_dir, catalog, self.registry,
                                                     acquisition=settings.acquisition.policy())
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
        self._index_retry_at: dict[str, float] = {}     # monotonic time a failed build may be retried (inf: never)
        self._check_inflight: dict[tuple[tuple[str, ...], str], asyncio.Task[bool]] = {}
        self._index_builds: dict[str, asyncio.Task[bool]] = {}   # observe mode's background builds
        self._check_failed_until: dict[tuple[tuple[str, ...], str], float] = {}
        self._soma_vocab = soma_filter.VocabCache(settings.resolution.remote_ttl_s)
        self._check_task: asyncio.Task[Any] | None = None
        self._readiness_supplied = False
        self._upstream_commit: str | None = None
        self._commit_warned: set[str] = set()
        self._ceiling = ceiling_of(settings)

    # ================================================================== wiring

    def reload_catalog(self, catalog: Catalog, registry: Any = None, *, settings: DataSettings | None = None) -> None:
        """Serve ``catalog`` (and ``registry``) from now on: a project registered a descriptor, an overlay or a
        plugin in the running session (docs/PROJECTS.md). The readiness cache and the resolver take the new
        catalog; what was derived from the old one (tool schemas, vocabularies, table statistics, index
        fingerprints and failed builds) is dropped and derived again on use. Resident tables, admission and the
        recorded check results of unchanged tables stay: a table's results are keyed by its fingerprint."""
        if settings is not None:
            self.settings = settings
        self.catalog = catalog
        self.registry = registry if registry is not None else getattr(catalog, "registry", None)
        self.readiness.catalog = catalog
        self.readiness.registry = self.registry
        self.resolver = Resolver(self.registry, catalog, self._index_provider, remote=self._remote,
                                 settings=self.settings)
        self._schemas.clear()
        self._vocab.clear()
        self._stats.clear()
        self._index_fp.clear()
        self._index_failed.clear()
        self._index_retry_at.clear()
        self._check_failed_until.clear()

    def bind_bridge(self, bridge: Any) -> None:
        self.bridge = bridge
        self.service.bind(bridge)
        with contextlib.suppress(Exception):
            # the host budget covers the upstream servers only: it leaves out the data child itself
            # (memory/host.py HARNESS_SERVERS), whatever the bridge lists
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
        configured = ((self.settings.raw or {}).get("service") or {}).get("mem_limit_mb")
        # an 'auto' limit stays 'auto' for the launcher, which plans it from the host (launch.server_limit_mb)
        limit: int | str = "auto" if is_auto(configured) else int(svc.mem_limit_mb)
        return [{"name": DATA_SERVER, "command": python, "args": ["-E", str(script)],
                 "env": env, "timeout_s": float(svc.timeout_s),
                 "max_concurrency": int(svc.max_concurrency), "mem_limit_mb": limit}]

    def launch_spec(self, cfg: Any) -> LaunchSpec | None:
        log_root = getattr(self.bridge, "log_root", None) if self.bridge is not None else None
        from ..launch import LIMIT_KINDS

        own = getattr(cfg, "limit_kind", None)
        if own and own not in LIMIT_KINDS and dataclasses.is_dataclass(cfg):
            # a typo never launches the server without containment: data.memory.limit_kind applies instead
            log.warning("server %s: limit_kind %r is not one of %s; launching under data.memory.limit_kind %r",
                        getattr(cfg, "name", cfg), own, LIMIT_KINDS, self.settings.memory.limit_kind)
            cfg = dataclasses.replace(cfg, limit_kind=None)
        try:
            log_dir = log_root() if callable(log_root) else log_root
            spec = build_launch_spec(cfg, self.settings, log_dir)
        except Exception:  # noqa: BLE001 - launch unguarded rather than not at all
            log.warning("launch_spec failed for %s", getattr(cfg, "name", cfg), exc_info=True)
            return None
        if spec is not None and str(cfg.name) != DATA_SERVER:     # admission and the host budget are upstream's
            with contextlib.suppress(Exception):
                self.admission.ledger.set_status_path(str(cfg.name), spec.status_path)
        return spec

    # ================================================================== listing

    def rewrite_listing(self, server: str, tool: str, description: str,
                        input_schema: dict[str, Any]) -> ListingDecision:
        if server == DATA_SERVER and tool.startswith("_"):
            if tool == "_check":
                self._forget_transient_index_failures()   # a (re)started data child gets another try
                self._schedule_check()
            return ListingDecision(False, description, input_schema, reason="internal data-child verb")
        self._schemas[(server, tool)] = dict(input_schema or {})
        try:
            contract = self.catalog.contract(server, tool)
        except Exception:  # noqa: BLE001
            return ListingDecision(True, description, input_schema)
        if contract.quarantined:
            if self._mode_for(server) == "enforce" and self.settings.gateway.when_service_down == "strict":
                return ListingDecision(True, f"UNAVAILABLE: quarantined ({contract.quarantine_reason}).\n"
                                             f"{description}", input_schema)
            return ListingDecision(True, description, input_schema)
        b = contract.binding
        if b is not None and (b.hidden or (b.block is not None and b.block.hidden and b.serve == "block")):
            return ListingDecision(False, description, input_schema, reason="hidden by the overlay")
        if contract.generic or b is None or self._mode_for(server) != "enforce":
            return ListingDecision(True, description, input_schema)
        from ..derive import annotate_schema, describe_tool
        try:
            # a native verb is listed without every table's column map (derive/tools.py, column_maps)
            schema = annotate_schema(contract, input_schema, catalog=self.catalog, registry=self.registry,
                                     vocab=self._vocab, enum_max=self.settings.derive.enum_max, column_maps=False)
            unready = degraded_tools(_OneTool(self.catalog, server, tool), self.readiness).get(
                f"mcp__{server}__{tool}") if self.readiness.tables else None
            text = describe_tool(contract, description, catalog=self.catalog,
                                 max_chars=self.settings.derive.description_max_chars, unready=unready)
        except Exception:  # noqa: BLE001 - never lose a tool over its description
            log.warning("deriving the listing of %s.%s failed", server, tool, exc_info=True)
            return ListingDecision(True, description, input_schema)
        if server == DATA_SERVER:
            schema = compact_native_listing(schema)
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
                    contract = self.catalog.contract(server, tool)
                    required, optional = derived_dependencies(contract, self.catalog)
                    # a derived serve's dependencies too (an enrichment's universe and hierarchy)
                    out.update(self.readiness.physical(ref)[0] for ref in (*contract.tables, *required, *optional))
                except Exception:  # noqa: BLE001 - a broken contract is reported by lint, not here
                    continue
        return sorted(out)

    async def refresh_readiness(self, tables: Sequence[str] = (), depth: str | None = None) -> bool:
        """Run the data child's ``_check`` (all tables by default) and cache the results. Single-flight:
        concurrent calls for the same tables and depth (parallel specialists reading one unchecked table)
        await one check, and a failed check answers False for :data:`CHECK_FAILURE_TTL_S` seconds."""
        depth = depth or self.settings.readiness.session_depth
        key = (tuple(sorted(set(tables))), depth)
        if time.monotonic() < self._check_failed_until.get(key, 0.0):
            return False
        task = self._check_inflight.get(key)
        if task is None or task.done():
            task = asyncio.get_running_loop().create_task(self._run_check(list(tables), depth))
            self._check_inflight[key] = task

            def _done(t: asyncio.Task[bool], key: tuple[tuple[str, ...], str] = key) -> None:
                if self._check_inflight.get(key) is t:
                    self._check_inflight.pop(key, None)
                if not t.cancelled() and t.exception() is None and not t.result():
                    self._check_failed_until[key] = time.monotonic() + CHECK_FAILURE_TTL_S

            task.add_done_callback(_done)
        return await asyncio.shield(task)

    async def _run_check(self, tables: list[str], depth: str) -> bool:
        try:
            resp = await self.service.check(tables, depth)
        except ServiceError as exc:
            log.info("readiness check unavailable: %s", exc.message)
            return False
        self.readiness.load_check_results(resp)
        return True

    async def aclose(self) -> None:
        """Stop background work (the session's readiness check) before the bridge closes."""
        task, self._check_task = self._check_task, None
        tasks = {t for t in [task, *self._check_inflight.values(), *self._index_builds.values()]
                 if t is not None and not t.done()}
        self._check_inflight.clear()
        self._index_builds.clear()
        for t in tasks:
            t.cancel()
        if tasks:
            with contextlib.suppress(BaseException):
                await asyncio.wait(tasks, timeout=5.0)

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
        """Unready tools ``{mcp__server__tool: reason}``: readiness, and (under ``strict``) the tools that
        depend on a quarantined catalog file."""
        out = degraded_tools(self.catalog, self.readiness)
        if getattr(self.catalog, "quarantined", None) and self.settings.gateway.when_service_down == "strict":
            for name, files in self.catalog.quarantined_tools().items():
                server, _, tool = name.partition(".")
                if self._mode_for(server) == "enforce":
                    out.setdefault(f"mcp__{server}__{tool}", "quarantined: " + "; ".join(
                        f"{q.file}: {q.summary}" for q in files))
        return out

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
        if _OBSERVING.get():
            # the resolver records existence unknown; the remote question is never asked in an observed call
            raise RuntimeError("observe mode: remote resolution is not done in the call path")
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
        if qualified in self._index_failed and time.monotonic() < self._index_retry_at.get(qualified, math.inf):
            return False
        try:
            resp = await self.service.build_index(src, bare)
        except ServiceError as exc:
            # a rejected build (a configuration fault) or a child the bridge gave up on stays failed; an
            # outage (timeout, crash, restart) is retried after a back-off, or when the child is relisted
            self._index_failed[qualified] = exc.message
            self._index_retry_at[qualified] = math.inf if exc.subkind in ("rejected", "down") else \
                time.monotonic() + INDEX_RETRY_S
            # keep a fingerprint already known: an index built later (``vbt ds index build``) is still found
            known = (self.readiness.indexes.get(qualified) or {}).get("fingerprint")
            self.readiness.set_index(qualified, "missing", fingerprint=known, detail=exc.message)
            return False
        self._index_fp[qualified] = resp.fingerprint
        ok = self._index_provider(src, bare) is not None
        self.readiness.set_index(qualified, "ready" if ok else "missing", fingerprint=resp.fingerprint,
                                 detail="" if ok else f"index file not found at {resp.path}")
        if ok:
            self._index_failed.pop(qualified, None)
            self._index_retry_at.pop(qualified, None)
        else:
            self._index_failed[qualified] = f"no index at {resp.path}"
            self._index_retry_at[qualified] = math.inf
        return ok

    def _build_in_background(self, qualified: str) -> None:
        """Observe mode: build ``qualified`` off the call path, so later calls trace full decisions."""
        task = self._index_builds.get(qualified)
        if task is not None and not task.done():
            return
        try:
            self._index_builds[qualified] = asyncio.get_running_loop().create_task(self._ensure_index(qualified))
        except RuntimeError:
            pass

    def _forget_transient_index_failures(self) -> None:
        for q in [q for q, at in self._index_retry_at.items() if at != math.inf]:
            self._index_failed.pop(q, None)
            self._index_retry_at.pop(q, None)

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
            token = _OBSERVING.set(True)
            try:
                await self._prepare(plan, st)
                self._observe(plan, "would_route", route=plan.route, args_sent=plan.args_sent,
                              resolved=st.resolved, notes=st.notes)
            except GatewayError as exc:
                self._observe(plan, f"would_{exc.kind.value}", error=exc.envelope())
            except Exception as exc:  # noqa: BLE001 - observe mode never changes a call
                self._observe(plan, "observe_failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                _OBSERVING.reset(token)
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
        if contract.quarantined:
            # R8: a catalog file this tool depends on does not load. strict refuses (observe traces it);
            # lenient runs the call unguarded, as a broken catalog did before quarantine
            err = quarantine_error(contract, name)
            if st.mode == "enforce" and self.settings.gateway.when_service_down == "lenient":
                st.generic = st.unguarded = True
                st.notes.append(f"unguarded: {contract.quarantine_reason}")
                self._observe(plan, "quarantined_unguarded", error=err.envelope())
                return
            raise err
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
        # observe mode adds no latency and no side effects: only cached vocabularies, and output paths
        # are confined and recorded but an existing file is never renamed (the call goes upstream unchanged)
        observe = st.mode != "enforce"
        vocab = await self._vocab_for(contract, args, selected, fetch=not observe)
        for column, value in list(st.auto_fixed.items()):
            snap = vocab.get(vocab_key(selected or contract.bound_table or "", column))
            st.auto_fixed[column] = self._snap_auto(contract, column, value, snap, st)
        prepared = apply_arg_contracts(contract, args, vocab, fixed_scope=st.auto_fixed, schema=schema,
                                       output_dir=self.run.get("mcp_output_dir"), registry=self.registry,
                                       enum_max=self.settings.derive.enum_max, dry=observe)
        st.prepared = prepared
        st.notes.extend(prepared.notes)
        for n, a in contract.args.items():
            if a.engine_param and isinstance(prepared.args_sent.get(n), str):
                # the text as sent upstream (escaped, wrapped), before the leakage filter joins it: the
                # remote witness adds the evidence ceiling itself
                st.engine_text[a.engine_param] = prepared.args_sent[n]
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
        # a pass binding is served derived when an argument it lists in derived_when is set (upstream cannot
        # honour include_descendants: the gateway answers that call itself)
        wants = any(plan.args_raw.get(a) or plan.gateway_args.get(a) for a in b.derived_when)
        derived = (b.serve == "derived" or wants) and b.derived is not None and self.profile != "fidelity"
        plan.route = "derived" if derived else "upstream"
        if st.undefined:
            plan.route = "none"
            return
        st.order = self._effective_order(contract, prepared, plan.args_raw)
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
        await self._family_rows(plan, st, contract, selected)
        # 8. leakage
        st.leakage = prepare_leakage(contract, plan.args_sent, self._ceiling, tool=name,
                                     environ={**os.environ, **base_tool_env(dict(self.config or {}))})
        st.notes.extend(st.leakage.notes)
        await self._ceiling_totals(plan, st, contract, selected)
        # 9. admission (upstream only; observe mode reserves nothing and never recycles a server)
        if plan.route == "upstream" and st.mode == "enforce":
            if b.count_first is not None:
                await self._count_first(plan, st, contract)
            await self._census_release(plan, st, contract)
            await self._admit(plan, st, contract)
        elif plan.route == "derived" and st.mode == "enforce":
            # a derived read of a live table pulls what upstream would (a whole cBioPortal study): sized first
            await self._admit_sized(plan, st, contract)
        if st.mode == "enforce" and not st.lenient and plan.route != "none" and plan.server != DATA_SERVER:
            await self._live_release(plan, st, contract)
        # 10. limit inflation
        if plan.route == "upstream":
            self._inflate(plan, st, contract, inflatable)

    async def _count_first(self, plan: CallPlan, st: _CallState, contract: ToolContract) -> None:
        """Count-first admission (``count_first``, F20): the data child counts the cells the filter selects
        (against the release ``stable`` resolves to) and the pull is refused ``too_large`` before upstream
        fetches anything when its estimate (``count_first.estimate``: the server's footprint plus the cells' rows
        and genes, calibrated on real pulls) would exceed the server's memory limit."""
        cf = contract.binding.count_first
        args = plan.args_sent
        # the genes the pull reads: every gene argument counts (upstream ORs symbols and Ensembl IDs); none named
        # reads every gene of the cell
        named = [args.get(a) for a in cf.genes_args if isinstance(args.get(a), list)]
        n_genes = sum(len(g) for g in named) if named else None
        max_cells = args.get(cf.max_cells_arg) if cf.max_cells_arg else None
        limit_mb = 0.0
        with contextlib.suppress(Exception):
            limit_mb = float(self.admission.limit_mb(plan.server) or 0)
        facet = _sample_facet(cf)
        sample_req = self._sample_request(plan, contract, cf, facet) if facet is not None else None
        if facet is not None and sample_req is None:
            # no sample can be asked for (no sample size, no two-column donor key): upstream would balance donor
            # labels alone, right only when the filter fixes the donor key's qualifier
            key = list(facet.get("key") or grain_columns(contract.tables.get(cf.table), str(facet.get("grain")
                                                                                            or "donor")) or [])
            if len(key) == 2:
                self._sample_unavailable(plan, st, contract, {"key": key}, "no sample size, seed or donor key")
        est = cf.estimate
        req = CensusCountRequest(table=cf.table, value_filter=args.get(cf.filter_arg) if cf.filter_arg else None,
                                 n_genes=n_genes, max_cells=max_cells if isinstance(max_cells, int) else None,
                                 cap_bytes=int(limit_mb * 1024 * 1024) if limit_mb > 0 else None, sample=sample_req,
                                 **(_estimate_fields(est) if est is not None else {}))
        try:
            resp = await self.service.call(VERB_CENSUS_COUNT, req)
        except ServiceError as exc:
            if sample_req is not None:
                self._sample_unavailable(plan, st, contract, sample_req, f"the count failed ({exc.message[:200]})")
            st.notes.append(f"count-first admission unavailable ({exc.message[:200]}); the memory admission applies")
            return
        # the sample (every drawn cell id) is applied below, not recorded with the count
        info = resp.counted()
        sample = resp.sample if sample_req is not None and isinstance(resp.sample, Mapping) else None
        st.count_first = info
        if resp.release:
            st.notes.append(f"{cf.table} release: {resp.release.get('resolved') or resp.release}")
        if resp.admissible is False:
            raise GatewayError(ErrorKind.too_large, f"{cf.table}: the filter selects {resp.n_cells} cells; the pull "
                               f"needs about {(resp.need_bytes or 0) // (1024 * 1024)} MB, over the "
                               f"{int(limit_mb)} MB limit of {plan.server}: narrow value_filter or lower max_cells",
                               tool=st.name, payload={"n_cells": resp.n_cells, "need_bytes": resp.need_bytes,
                                                      "cap_bytes": resp.cap_bytes, "release": resp.release},
                               subkind="count_first")
        if resp.n_cells is None:
            st.notes.append(f"count-first admission could not count: {resp.reason}")
        else:
            need = f", about {resp.need_bytes // (1024 * 1024)} MB to pull" if resp.need_bytes is not None else ""
            st.notes.append(f"count-first: the filter selects {resp.n_cells} cells{need}")
        if sample_req is not None:
            self._apply_sample(plan, st, contract, cf, facet or {}, sample_req, resp.n_cells, sample)

    # ---------------------------------------------------------------- the derived sample (count_first.sample)

    def _sample_request(self, plan: CallPlan, contract: ToolContract, cf: Any, facet: Mapping[str, Any]
                        ) -> dict[str, Any] | None:
        """The ``sample`` of the count-first request: ``max_cells`` (the argument, else the tool's schema
        default), upstream's seed, the stratum and the donor key (the descriptor grain ``facet.grain``)."""
        want = plan.args_sent.get(cf.max_cells_arg) if cf.max_cells_arg else None
        if not isinstance(want, int) or isinstance(want, bool):
            prop = ((self._schemas.get((plan.server, plan.tool)) or {}).get("properties") or {}).get(
                cf.max_cells_arg or "") or {}
            want = prop.get("default") if isinstance(prop, Mapping) else None
            want = want if isinstance(want, int) and not isinstance(want, bool) else facet.get("max_cells_default")
        if not isinstance(want, int) or want <= 0:
            return None
        t = contract.tables.get(cf.table)
        key = list(facet.get("key") or grain_columns(t, str(facet.get("grain") or "donor")) or [])
        if len(key) != 2:
            return None
        if facet.get("seed") is None:
            return None                                # the overlay declares the generator's seed (ASN-4)
        id_column = next((_last(k) for k in (t.key if t is not None else []) if not k.endswith("#")), None)
        out = {"max_cells": int(want), "seed": int(facet["seed"]), "key": key, "id_column": id_column}
        if facet.get("stratify"):
            out["stratify"] = str(facet["stratify"])
        if facet.get("max_read"):
            out["max_read"] = int(facet["max_read"])
        return out

    def _sample_unavailable(self, plan: CallPlan, st: _CallState, contract: ToolContract, req: Mapping[str, Any],
                            why: str) -> None:
        """No derived sample: upstream balances donors by ``donor_id`` alone, which is right only within one
        ``dataset_id`` (the qualifier of the donor key). Served upstream when the filter fixes it, else refused."""
        qualifier = str(req["key"][0])
        if st.soma_predicate is not None and soma_filter.fixes_single(st.soma_predicate, qualifier):
            st.notes.append(f"no derived sample ({why}); the filter fixes one {qualifier}, so upstream's own "
                            "donor balancing is served")
            return
        raise GatewayError(ErrorKind.unsupported_combination,
                           f"no derived sample ({why}); without one, donors of different {qualifier} values that share "
                           f"a {req['key'][1]} label would be balanced as one donor: fix one {qualifier} in the filter "
                           "or narrow it", tool=st.name, argument=contract.binding.count_first.filter_arg,
                           payload=unsupported_combination_payload(
                               [contract.binding.count_first.filter_arg or ""], why,
                               alternative=_sample_alternative(contract)))

    def _apply_sample(self, plan: CallPlan, st: _CallState, contract: ToolContract, cf: Any, facet: Mapping[str, Any],
                      req: Mapping[str, Any], n_cells: int | None, sample: Mapping[str, Any] | None) -> None:
        """Serve the pull as the derived sample: upstream is asked for exactly the sampled cells (a ``soma_joinid in
        [...]`` filter and ``max_cells`` = its size: upstream keeps every cell a filter selects when they are at most
        ``max_cells``). A filter selecting at most ``max_cells`` cells is passed as it is (every cell is fetched, no
        donor is balanced). The columns argument is completed with the cell key, the donor key and the stratum so
        the written file shows which cells it holds."""
        if n_cells is None:
            self._sample_unavailable(plan, st, contract, req, "the cells could not be counted")
            return
        self._complete_columns(plan, st, contract, cf, facet, req)
        max_cells = int(req["max_cells"])
        if n_cells <= max_cells:
            st.notes.append(f"the filter selects {n_cells} cells, at most max_cells={max_cells}: every cell is "
                            "fetched (nothing is sampled)")
            return
        ids = list((sample or {}).get("ids") or [])
        if not ids or not (sample or {}).get("value_filter"):
            self._sample_unavailable(plan, st, contract, req, str((sample or {}).get("reason") or "no sample returned"))
            return
        plan.args_sent[cf.filter_arg] = str(sample["value_filter"])  # type: ignore[index]
        if cf.max_cells_arg:
            plan.args_sent[cf.max_cells_arg] = int(sample.get("max_cells") or len(ids))  # type: ignore[union-attr]
        keep = ("n_sampled", "n_total", "n_donors", "n_donors_sampled", "n_datasets", "n_strata", "donor_key",
                "stratified_by", "seed", "method", "per_dataset", "per_donor", "per_stratum")
        t = contract.tables.get(cf.table)
        id_column = next((_last(k) for k in (t.key if t is not None else []) if not k.endswith("#")), None)
        st.sample = {"ids": ids, "n_total": int(n_cells), "max_cells": max_cells, "facet": dict(facet),
                     "id_column": id_column,
                     "summary": {k: sample[k] for k in keep if k in sample}}  # type: ignore[index]
        st.transforms.append(f"derived sample {len(ids)} of {n_cells} cells")
        how = f"stratified by {sample.get('stratified_by')}, " if sample.get("stratified_by") else ""  # type: ignore
        st.notes.append(f"served as the derived sample: {len(ids)} of the {n_cells} cells the filter selects, {how}"
                        f"balanced over ({', '.join(req['key'])}) donors (seed {sample.get('seed')}); upstream was "
                        f"asked for exactly these cells ({cf.filter_arg} = soma_joinid in [...], "
                        f"{cf.max_cells_arg} = {len(ids)})")

    def _complete_columns(self, plan: CallPlan, st: _CallState, contract: ToolContract, cf: Any,
                          facet: Mapping[str, Any], req: Mapping[str, Any]) -> None:
        """``facet.columns_arg`` names the obs columns to write: the cell key, the donor key and the stratum are
        added to the caller's list (when none is given, the table's declared columns are asked for)."""
        arg = facet.get("columns_arg")
        t = contract.tables.get(cf.table)
        if not arg or t is None:
            return
        need = [*[k for k in t.key if not k.endswith("#")], *req["key"], *([req["stratify"]] if req.get("stratify")
                                                                          else [])]
        given = plan.args_sent.get(arg)
        if isinstance(given, list):
            added = [c for c in dict.fromkeys(need) if c not in given]
            if added:
                plan.args_sent[arg] = [*given, *added]
                st.notes.append(f"{arg}: {', '.join(added)} added (the file must show its cells and their donors)")
        elif given is None:
            plan.args_sent[arg] = list(dict.fromkeys([*need, *t.columns]))
            st.notes.append(f"{arg}: the declared obs columns are asked for (the file must show its cells and donors)")

    async def _sampled_result(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any) -> None:
        """A pull served as the derived sample: the payload's total (``facet.total_path``) is the counted total of
        the caller's filter, not the sample's size upstream counted; ``derived_sample`` describes the draw; and the
        written file must hold exactly the sampled cells (``sample_cells`` check, else ``tool_defect``; the data
        child reads the file's cell keys, as it reads its genes for ``recompute_genes``)."""
        sm = st.sample
        if not sm or not isinstance(obj, dict):
            return
        tp = sm["facet"].get("total_path")
        if tp:
            before = jp_first(obj, tp)
            jp_set(obj, tp, sm["n_total"])
            st.notes.append(f"{_last(tp)} is the {sm['n_total']} cells the filter selects (upstream counted the "
                            f"{before} cells of the sample)")
        obj["derived_sample"] = json_value(sm["summary"])
        b = contract.binding
        spec = b.result.files[0] if b.result.files else None
        raw = jp_first(obj, spec.path_from) if spec is not None else None
        if not raw:
            return
        path = Path(str(raw))
        if not path.is_absolute() and self.run.get("mcp_output_dir"):
            path = Path(str(self.run["mcp_output_dir"])) / path
        got = await self._file_cells(contract, path, sm.get("id_column"))
        if got is None:
            st.checks.append(("sample_cells", None, f"the cells of {path.name} could not be read"))
            return
        want = set(int(i) for i in sm["ids"])
        missing, extra = len(want - got), len(got - want)
        ok = not missing and not extra
        st.checks.append(("sample_cells", ok, f"file holds {len(got)} cells; {missing} sampled cell(s) missing, "
                                              f"{extra} unsampled"))
        if not ok:
            raise GatewayError(ErrorKind.tool_defect, f"the written file does not hold the sampled cells ({missing} "
                               f"missing, {extra} not sampled)", tool=st.name,
                               payload=tool_defect_payload("sample_cells", {"total": sm["n_total"]}, None,
                                                           [d.id for d in b.defects]))

    def _resolve_release_alias(self, st: _CallState, contract: ToolContract, obj: Any) -> None:
        """``result.release_alias``: the payload names the release by the alias the server opened (Census
        ``stable``); the dated release this call resolved replaces it, so the body cites what the header does."""
        path = contract.binding.result.release_alias
        if not path or not isinstance(obj, dict):
            return
        cf = (st.count_first or {}).get("release") if isinstance(st.count_first, Mapping) else None
        resolved = st.census_release or (str(cf["resolved"]) if isinstance(cf, Mapping) and cf.get("resolved")
                                         else None)
        before = jp_first(obj, path)
        if not resolved or before is None or str(before) == resolved:
            return
        jp_set(obj, path, resolved)
        st.transforms.append(f"{_last(path)} {before} -> {resolved}")
        st.notes.append(f"{_last(path)}: the server opened {before!r}, which named the {resolved} release for this "
                        "call")

    async def _file_cells(self, contract: ToolContract, path: Path, column: Any) -> set[int] | None:
        """The cell keys (``column``) a written file holds, read by the data child; None when it cannot."""
        cf = contract.binding.count_first
        if cf is None or not column:
            return None
        try:
            resp = await self.service.call(VERB_CENSUS_COUNT, CensusCountRequest(
                table=cf.table, cells_file=str(path), cells_column=str(column)))
        except ServiceError:
            return None
        cells = resp.file_cells
        if not isinstance(cells, list):
            return None
        with contextlib.suppress(TypeError, ValueError):
            return {int(v) for v in cells}
        return None

    def _soma_tables(self, contract: ToolContract) -> list[str]:
        """The tables the tool reads whose layout resolves a moving release alias (SOMA: ``stable``)."""
        out = []
        for ref, t in contract.tables.items():
            try:
                layout = self.registry.find("layout", t.layout) if self.registry is not None else None
            except Exception:  # noqa: BLE001
                layout = None
            if layout is not None and callable(getattr(layout, "resolve", None)):
                out.append(ref)
        return out

    async def _census_release(self, plan: CallPlan, st: _CallState, contract: ToolContract) -> None:
        """Every call of a tool that reads the Census records the dated release ``stable`` names (the witness or
        count-first may have resolved it already): when the alias moves, the same cited count changes and the
        provenance shows why (get_census_info and count_cells recorded no release)."""
        w = st.witness
        cf = (st.count_first or {}).get("release") if isinstance(st.count_first, Mapping) else None
        if (w is not None and getattr(w, "as_of", None)) or (isinstance(cf, Mapping) and cf.get("resolved")):
            return
        tables = self._soma_tables(contract)
        if not tables:
            return
        try:
            resp = await self.service.call(VERB_RELEASE, ReleaseRequest(table=tables[0]))
        except ServiceError as exc:
            st.notes.append(f"{tables[0]} release not resolved ({exc.message[:200]})")
            return
        rel = resp.release or {}
        if rel.get("resolved"):
            st.census_release = str(rel["resolved"])
            if rel.get("drift"):
                st.notes.append(f"{tables[0]}: {rel.get('requested')} moved from {rel['drift'].get('before')} to "
                                f"{rel['drift'].get('now')} during this session")
        else:
            st.notes.append(f"{tables[0]} release unknown: {rel.get('reason') or resp.reason or 'not resolved'}")

    def _release_tables(self, contract: ToolContract) -> list[str]:
        """One table per live source the tool reads whose descriptor names a release (``layout.options.release``:
        data release and software versions; ``release.per``: per-record releases): CT.gov, cBioPortal, PubMed."""
        out: dict[str, str] = {}
        for ref, t in contract.tables.items():
            src = t.descriptor.source
            try:
                layout = self.registry.find("layout", t.layout) if self.registry is not None else None
            except Exception:  # noqa: BLE001
                layout = None
            if src in out or not callable(getattr(layout, "release_info", None)):
                continue
            lay = getattr(t.physical_spec, "layout", None)
            opts = lay.get("options") if isinstance(lay, Mapping) else getattr(lay, "options", None)
            per = getattr(t.descriptor.release, "per", None)
            if isinstance(opts, Mapping) and opts.get("release") or per:
                out[src] = ref
        return list(out.values())

    async def _live_release(self, plan: CallPlan, st: _CallState, contract: ToolContract) -> None:
        """An upstream- or derived-served call of a live source records what the source says about the data it read
        (CR7): the data release (CT.gov ``dataTimestamp``; one cBioPortal study's ``importDate``), the API and software
        versions (CT.gov ``apiVersion``, cBioPortal ``portalVersion``/``dbVersion``, the PubMed build) and the
        per-record releases of the records the call names (cBioPortal studies). Provenance only: a failed request
        leaves them unrecorded, never the call."""
        for ref in self._release_tables(contract):
            pred = st.predicate if st.predicate is not None and ref == (plan.bound_table or contract.bound_table) \
                else None
            try:
                resp = await self.service.call(VERB_RELEASE, ReleaseRequest(
                    table=ref, predicate=to_json(pred) if pred is not None else None))
            except ServiceError:
                continue
            rel = dict(resp.release or {})
            if rel:
                st.live_release[str(ref).split(".")[0]] = rel

    async def _row_releases(self, plan: CallPlan, st: _CallState, contract: ToolContract, t: Any,
                            rows: list[Any]) -> None:
        """Upstream-served records of a source whose release is per record (``release.per``: a cBioPortal study's
        importDate) that come back without that version (search_studies lists studies without importDate): their
        releases are read for the record versions of the provenance (LIVE3-11). Provenance only."""
        if t is None or st.mode != "enforce" or not rows:
            return
        desc = t.descriptor
        per = getattr(desc.release, "per", None) or {}
        version = t.spec.key.version
        key = [k for k in t.key if not str(k).endswith("#")]
        if per.get("table") != _last(str(t.physical)) or not version or len(key) != 1:
            return
        if all(isinstance(r, Mapping) and get_path(r, str(version)) not in (None, "") for r in rows):
            return                                     # the rows carry their own versions
        ids = sorted({str(get_path(r, key[0])) for r in rows if isinstance(r, Mapping) and get_path(r, key[0])})
        if not ids:
            return
        try:
            resp = await self.service.call(VERB_RELEASE, ReleaseRequest(
                table=str(t.ref), predicate=to_json(In(key[0], tuple(ids)))))
        except ServiceError:
            return
        got = dict((resp.release or {}).get("per") or {}) if resp.release else {}
        if got:
            lr = st.live_release.setdefault(desc.source, {})
            lr["per"] = {**dict(lr.get("per") or {}), **got}

    async def _recompute_genes(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any) -> None:
        """``count_first.recompute_genes``: genes_found/genes_not_found of the written file, from its
        ``var.feature_name``; when the file cannot be read they are dropped (upstream's are wrong)."""
        cf = contract.binding.count_first
        if cf is None or not cf.recompute_genes or not isinstance(obj, dict):
            return
        symbols = next(iter(cf.genes_args), None)          # the first gene argument names genes by feature_name
        genes = plan.args_sent.get(symbols) if symbols else None
        paths = list((st.prepared.output_paths if st.prepared is not None else {}).values())
        resp = None
        if isinstance(genes, list) and paths:
            req = CensusCountRequest(table=cf.table, genes_file=str(paths[0]), genes=[str(g) for g in genes])
            with contextlib.suppress(ServiceError):
                resp = await self.service.call(VERB_CENSUS_COUNT, req)
        if resp is not None and resp.genes_found is not None:
            obj["genes_found"], obj["genes_not_found"] = list(resp.genes_found), list(resp.genes_not_found or [])
            st.notes.append("genes_found/genes_not_found recomputed from the file's var.feature_name")
        else:
            for k in ("genes_found", "genes_not_found"):
                obj.pop(k, None)
            st.notes.append("genes_found/genes_not_found removed: upstream compares positional var_names")

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
        tables = tables_read(contract, selected, plan.args_raw, catalog=self.catalog)
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

    async def _vocab_for(self, contract: ToolContract, args: Mapping[str, Any], selected: str | None, *,
                         fetch: bool = True) -> dict[str, Any]:
        """Vocabulary snapshots for category/scope arguments with values (and free-text substring
        arguments, and the auto-derived scope arguments). ``fetch=False`` (observe mode): cached only."""
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
                    (a.interpreted_as in ("substring", "casefold_substring") and not a.pooled):
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
                if not fetch:
                    continue
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
                    # a key column (soma_joinid) has one value per cell: its "vocabulary" is the whole table (217 M
                    # values on the real Census), so its values are never listed or checked
                    keys = {_last(k) for t in contract.tables.values() for k in t.key if not k.endswith("#")}
                    cols = [c for c in soma_filter.filter_columns(pred) if c not in keys]
                    if st.mode != "enforce":
                        # observe mode never asks the watched server for a vocabulary in the call path
                        vocab = {c: self._soma_vocab.get(c) for c in cols}
                        if any(v is None for v in vocab.values()):
                            st.notes.append("observe mode: SOMA vocabularies not fetched in the call path")
                    else:
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
                        if st.mode != "enforce":
                            # observe mode never builds an index inside the agent's call
                            n_src, _, n_bare = n.partition(":")
                            if self._index_provider(n_src, n_bare) is None:
                                st.notes.append(f"observe mode: {n} index not built in the call path")
                                self._build_in_background(n)
                                ok = False
                            continue
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
                # a list column (drug_warning.chemblIds) holds the ID as one of its items
                probe = Contains(column, canonical_value) if a.op in ("contains", "overlaps") else \
                    Eq(column, canonical_value)
                found = await self._count(table, probe, st) if table and column else None
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

    async def _family_rows(self, plan: CallPlan, st: _CallState, contract: ToolContract,
                           selected: str | None) -> None:
        """``family: exact`` (§11.5): the call reads the resolved ID only, so rows stored under its other family
        members (a salt's 61 adverse-event rows when the name resolved to the parent) are counted under the same
        filters and reported as ``_vbt.family_rows`` (status partial), never left behind an empty answer."""
        if st.mode != "enforce" or st.lenient or not st.predicate:
            return
        table = selected or contract.bound_table
        for arg, res in st.results.items():
            a = contract.args.get(arg)
            members = [m for m in (getattr(res, "family", None) or ()) if m != res.canonical]
            if a is None or a.family == "include" or not members or arg not in st.per_arg or not table:
                continue
            tb, column = bound_column(contract, arg, a, selected)
            spec = column_spec(contract, tb, column)
            if getattr(spec, "self_", False):
                continue                               # the entity table itself: its other members are other rows
            for member in members[:MAX_FAMILY_COUNTS]:
                _p, per = build_predicate(contract, {arg: member}, selected=selected, registry=self.registry,
                                          confirmed=self._confirmed(selected))
                if arg not in per:
                    continue
                parts = [per[arg] if k == arg else p for k, p in st.per_arg.items()]
                # a list column names the parent and its salts in one row (drug_mechanism_of_action.chemblIds):
                # rows the requested ID already matched are in the answer, not left behind (OT-RV3-05)
                parts.append(Not(st.per_arg[arg]))
                parts += [Not(Eq(c, v)) for c, v in st.anchors.values()]
                n = await self._count(table, parts[0] if len(parts) == 1 else And(tuple(parts)), st)
                if n:
                    st.family_rows[str(member)] = int(n)
            if st.family_rows:
                st.notes.append(f"{arg}: {sum(st.family_rows.values())} matching row(s) are stored under other family "
                                f"members of {res.canonical} ({', '.join(f'{m}: {n}' for m, n in st.family_rows.items())}); "
                                "call again with that ID for them")

    def _stored(self, res: Any, table: str | None) -> Any:
        """The bound table's own spelling of a resolved key (predicates read the table as stored)."""
        if not table:
            return res.canonical
        return self.resolver.send_value(res, "stored", table)

    async def _count(self, table: str | None, pred: Predicate, st: _CallState) -> int | None:
        """One witness count (existence ``bound``, anchors, coverage universes); None when unknown (and in
        observe mode, which asks the data child nothing in the call path)."""
        if not table or st.lenient or st.mode != "enforce":
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

    def _effective_order(self, contract: ToolContract, prepared: PreparedArgs,
                         args: Mapping[str, Any] | None = None) -> list[Any]:
        b = contract.binding
        if prepared.order and prepared.order.get("column"):
            return [{"column": prepared.order["column"], "direction": prepared.order.get("direction", "desc"),
                     "nulls": "last"}]
        if b is None:
            return []
        for arg, order in b.result.order_when.items():
            if is_present((args or {}).get(arg)):
                return list(order)                     # upstream's own ranking for this argument (RV-OT-08)
        return list(b.result.order)

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

    def _remote_countable(self, contract: ToolContract, table: str | None) -> bool:
        """The table's layout answers an independent count request (capability ``count`` without ``scan``:
        live APIs, SOMA): its witness is the remote witness (§11.6, F20), whatever the read access."""
        t = contract.tables.get(table) if table else None
        if t is None or self.registry is None:
            return False
        try:
            layout = self.registry.find("layout", t.layout)
        except Exception:  # noqa: BLE001
            return False
        caps = set(getattr(layout, "capabilities", ()) or ())
        return "count" in caps and "scan" not in caps

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

    def _inexpressible(self, plan: CallPlan, contract: ToolContract, *, remote: bool = False,
                       parsed: Sequence[str] = ()) -> str | None:
        """Why the witness cannot count this call (None: it can). A remote count request sends an engine
        argument with an ``engine_param`` to that parameter, so the source's engine matches it there too; an
        argument the gateway parsed into the predicate IR itself (``parsed``: a SOMA ``value_filter``) is counted
        from that predicate."""
        for name, a in contract.args.items():
            if not is_present(plan.args_raw.get(name)):
                continue
            if remote and (a.engine_param or name in parsed or _engine_column(a)):
                continue
            # text matched over several columns (binds_any) is the source's matching, not one column's
            if a.role == "free_text" and (a.interpreted_as in ("engine", "regex") or not a.binds):
                return f"{name} is matched by the source's engine"
            if a.role == "unbound":
                return f"{name} is not bound to a column"
        return None

    @staticmethod
    def _engine_matches(plan: CallPlan, contract: ToolContract, table: str | None) -> list[Predicate]:
        """Engine text bound to one of the table's columns (``eligibility_text`` binds ``eligibilityCriteria``), as the
        remote witness sends it: one phrase per value, matched by the source's engine on that column's field
        (``AREA[EligibilityCriteria]"MGMT"``, as upstream quotes it). None of it is re-checked on the rows."""
        out: list[Predicate] = []
        for name, a in contract.args.items():
            value = plan.args_sent.get(name)
            col = _engine_column(a)
            if not col or not is_present(value) or not table or not col.startswith(table + "."):
                continue
            column = col[len(table) + 1:]
            for v in (value if isinstance(value, list) else [value]):
                if isinstance(v, str) and v.strip():
                    out.append(TextMatch(column, v.strip(), "exact"))
        return out

    @staticmethod
    def _soma_parsed(contract: ToolContract, st: _CallState) -> list[str]:
        """The SOMA filter arguments the gateway parsed into ``st.soma_predicate`` that a remote witness can count:
        only for a count, or a written file with a declared total (cells); a tool whose rows are value counts or
        datasets has no total in cells to compare."""
        b = contract.binding
        if st.soma_predicate is None or b is None or b.result.total is None or b.result.kind not in ("count", "file"):
            return []
        return [n for n, a in contract.args.items() if a.escape == soma_filter.LANGUAGE]

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
        remote = self._remote_countable(contract, table)
        if not remote and not self._scannable(contract, table):
            st.witness_reason = f"{table} is not scannable by the data child and declares no count capability"
            return
        why = self._inexpressible(plan, contract, remote=remote, parsed=self._soma_parsed(contract, st))
        if why:
            st.witness_reason = why
            st.engine_matched = True
            return
        if st.engine_text:
            st.engine_matched = True                   # counted by the source's engine; its order stays unverified
        dims = [d for d in self._scope_dims(contract, table)
                if _last(d) not in {_last(k) for k in self._fixed(st, st.prepared)}]
        if derived and not dims:
            return                                     # derived serving counts itself (one scan)
        if remote:
            # one independent count request under the bound predicate: only a total, no ranking or keys.
            # Engine text goes to its request parameter (TextMatch on "@query.cond"), never into per_arg:
            # the returned rows carry no such column to re-check
            parts = [] if st.predicate is None else list(st.predicate.preds if isinstance(st.predicate, And)
                                                          else (st.predicate,))
            parts.extend(TextMatch(f"@{param}", text) for param, text in st.engine_text.items())
            parts.extend(self._engine_matches(plan, contract, table))
            if self._soma_parsed(contract, st):
                # the SOMA filter as the gateway parsed and sent it (is_primary_data == True included)
                sp = st.soma_predicate
                parts.extend(sp.preds if isinstance(sp, And) else (sp,))
            pred = None if not parts else (parts[0] if len(parts) == 1 else And(tuple(parts)))
            st.remote_parts = list(parts)
            req = WitnessRequest(table=table, predicate=to_json(pred) if pred is not None else None,
                                 key=[], params=self._params(plan, st))
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
                st.witness_reason = st.witness.reason or "the remote count request could not count this call"
            st.checks.append(("witness_count", True if st.witness.total is not None else None,
                              f"total={st.witness.total} (remote count request)"))
            return
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
                             params=self._params(plan, st))
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

    async def _ceiling_totals(self, plan: CallPlan, st: _CallState, contract: ToolContract, selected: str | None
                              ) -> None:
        """Under the evidence ceiling an upstream count bounds the availability date only (the overlay's
        ``leakage_filter``: CT.gov ``StudyFirstPostDate``), while rows changed after the ceiling are withheld and a
        native find counts only the records also changed by it (``rows: withhold``). The two totals of one filter
        differ (a real RECRUITING count: 2,323 first posted by 2017-12-31, 87 also last updated by then); one more
        remote count, with the change date bounded too, puts both in this call's header."""
        lk = st.leakage
        w = st.witness
        table = selected or contract.bound_table
        spec = lk.specs.get(table) if lk is not None and table else None
        if lk is None or not lk.active or not lk.injected or spec is None or not spec.changed_at or \
                spec.rows != "withhold" or w is None or w.total is None or not self._remote_countable(contract, table):
            return
        changed = str(spec.changed_at)
        parts = [*st.remote_parts, Cmp(changed, "<=", lk.ceiling.isoformat())]      # type: ignore[union-attr]
        req = WitnessRequest(table=table, predicate=to_json(And(tuple(parts)) if len(parts) > 1 else parts[0]),
                             key=[], params=self._params(plan, st))
        try:
            both = await self.service.witness(req)
        except ServiceError:
            return
        if both.total is None:
            return
        st.ceiling_totals = {"available": int(w.total), "available_and_unchanged": int(both.total),
                             "ceiling": lk.ceiling.isoformat()}                     # type: ignore[union-attr]
        avail = _last(str(spec.available_at).rsplit(".date", 1)[0])
        st.notes.append(f"under the evidence ceiling {lk.ceiling} the total counts records available "
                        f"({avail}) by then ({w.total}); {both.total} of them were also last changed "
                        f"({_last(changed.rsplit('.date', 1)[0])}) by then: the rows a find returns under this "
                        "ceiling (the others are withheld from rows)")

    async def _storage_types(self, table: str | None, st: _CallState) -> None:
        """The storage types of ``table``'s leaves (they type witness keys and values): from ``_stats`` when this
        session already has them, else from the session check the call waited for (the footers it read; an item
        table's are its physical table's), and only without either from a ``_stats`` request. That request
        samples the table and every item table over it: 14.5 s for the 25.09 target tables on a first call."""
        if not table:
            return
        checked = None
        if table not in self._stats:
            m = self.readiness.get(self.readiness.physical(table)[0])
            checked = m.storage_types if m is not None and m.storage_types else None
        if checked is not None:
            for col, typ in checked.items():
                st.storage_types.setdefault(col, typ)
        elif table not in self._stats:
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
        if self._writes_rows(contract):
            return False
        if self._remote_countable(contract, plan.bound_table or contract.bound_table):
            return False                               # a remote count has no keys to check more rows against,
                                                       # and more pages spend the source's request budget
        n = int(w.total) + int(w.unknown_total or 0)
        if n > self.settings.witness.max_inflate_rows:
            return False
        stats = self._stats.get(plan.bound_table or contract.bound_table or "")
        row_bytes = None
        if stats is not None and stats.row_bytes_p99:
            row_bytes = stats.row_bytes_p99.get(b.result.grain or "row") or max(stats.row_bytes_p99.values())
        return row_bytes is None or n * row_bytes <= self.settings.witness.max_inflate_bytes

    @staticmethod
    def _writes_rows(contract: ToolContract) -> bool:
        """The tool writes its rows to a file (an ``output_path`` argument, declared ``files``; upstream names a file
        itself when the agent gives none): a larger limit, inflated or re-called, would write rows the agent never
        asked for into that file."""
        b = contract.binding
        return any(a.role == "output_path" for a in contract.args.values()) or bool(b is not None and b.result.files)

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
        if decision.oom and self._mode_for(server) != "enforce":
            # observe mode keeps the reaper's containment but not its crash policy: the memory kill is
            # traced, and the call is retried once as it would be without a gateway
            self._observe(plan, "would_oom", error=decision.error.envelope() if decision.error else None)
            return CrashDecision(retry=st.attempts == 1, oom=False, error=GatewayError(
                ErrorKind.server_crashed, f"{server}.{plan.tool}: the server connection failed ({reason})",
                tool=st.name, payload={"server": server}))
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
            st.serve_as_of = getattr(resp, "as_of", None)
            if not rows:
                st.in_universe = await self._in_coverage_universe(st, self._rows_table(plan, contract))
            return self._finish_rows(plan, st, contract, obj, rows, counters, served_by="derived", serve=resp)
        if raw is None:
            raise GatewayError(ErrorKind.source_error, "no upstream result", tool=st.name)
        if plan.server == DATA_SERVER:
            st.native_header = self._native_envelope(raw, st)
            missing = (st.native_header or {}).get("not_found_items")
            if missing:
                st.not_found_items = list(missing)     # keys the source does not hold: never a complete answer
        w = st.witness
        cls = classify(raw, contract, plan, universe_tables=self._universe_tables(contract),
                       witness_total=w.total if w is not None and w.total_method != "unknown" else None,
                       registry=self.registry)
        if cls.outcome == "oom":
            raise cls.error  # type: ignore[misc]
        counted = (st.count_first or {}).get("n_cells") if isinstance(st.count_first, Mapping) else None
        if cls.outcome == "not_found" and cls.explicit_not_found and isinstance(counted, int) and \
                "cell" in str(cls.reason).lower():
            # "No cells found for filter: ..." names the filter, not an identifier argument: the count-first count
            # of that filter (same release) decides. 0 cells is an empty answer; more is a contradiction
            if counted > 0:
                return await self._contradiction(plan, st, contract, "W1", 0, f"{cls.reason}, but the count-first "
                                                 f"count found {counted} cells")
            st.notes.append(f"{cls.reason}: the count-first count found 0 cells for the filter in this release")
            return self._result(plan, st, obj=cls.obj if cls.is_json else {}, rows=[], counters=counters,
                                served_by=served_by, total=0, total_method="count_first", status="empty",
                                text_rows=True)
        if cls.outcome in ("not_found", "source_error"):
            raise cls.error  # type: ignore[misc]
        if cls.outcome == "contradiction":
            return await self._contradiction(plan, st, contract, "W1", 0, cls.reason)
        if cls.outcome in ("empty", "empty_unverified"):
            if cls.is_json:
                self._echo_set_none_found(plan, st, contract, cls.obj)
                self._listed_missing(plan, st, contract, cls.obj, rows_returned=False)
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

    @staticmethod
    def _native_envelope(raw: RawResult, st: _CallState) -> dict[str, Any] | None:
        """A native data tool answered: its typed error envelope is raised as that error (not an
        ``empty_unverified`` success around a not_found), and its own ``_vbt`` header is kept for the result
        (the child is the source of truth: its source, totals and ``served_by``)."""
        obj = raw.structured if isinstance(raw.structured, Mapping) else None
        if obj is None:
            try:
                obj = json.loads(raw.text) if raw.text else None
            except ValueError:
                obj = None
        if not isinstance(obj, Mapping):
            return None
        if obj.get("status") == "tool_error" and obj.get("kind"):
            raise GatewayError.from_envelope(obj).with_tool(st.name)
        vbt = obj.get("_vbt")
        return dict(vbt) if isinstance(vbt, Mapping) else None

    # ---------------------------------------------------------------- upstream rows

    def _mapper(self, contract: ToolContract) -> FieldMapper:
        r = contract.binding.result
        it = contract.tables.get(r.rows_of) if r.rows_of else None
        prefix = it.items_path if it is not None and it.is_item_table else None
        return FieldMapper(r.fields, parent_key=r.parent_key, key_from_args=r.key_from_args, item_prefix=prefix)

    @staticmethod
    def _params(plan: CallPlan, st: _CallState) -> dict[str, Any]:
        """Scalar parameters for the data child: the caller's arguments over the disclosed schema defaults
        (``min_cell_lines`` defaults to 3), so the computation applies the default ``_vbt.scope`` reports."""
        prepared = st.prepared
        defaults = dict(getattr(prepared, "defaults", None) or {}) if prepared is not None else {}
        merged = {**defaults, **{n: v for n, v in plan.args_raw.items() if v is not None}}
        for n, v in plan.args_raw.items():
            if v is None:
                merged.pop(n, None)                    # an explicit null lifts the default (no restriction)
        return {n: v for n, v in merged.items() if isinstance(v, (str, int, float, bool))}

    @staticmethod
    def _nested_record_column(contract: ToolContract, t: Any) -> Any:
        """The nested column a ``kind: record`` result reads (``rows: $.<column>``) without an item table."""
        b = contract.binding
        if t is None or b.result.kind != "record" or b.result.rows_of or len(b.result.row_paths) != 1:
            return None
        path = b.result.row_paths[0]
        if not path.startswith("$.") or path == "$":
            return None
        c = t.columns.get(path[2:].split(".")[0].split("[")[0])
        return c if c is not None and is_container(c) else None

    def _rows_table(self, plan: CallPlan, contract: ToolContract) -> Any:
        b = contract.binding
        ref = b.result.rows_of or plan.bound_table or contract.bound_table
        return contract.tables.get(ref) if ref else None

    async def _process_upstream(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any,
                                raw: RawResult, counters: Counters) -> DataResult:
        b = contract.binding
        await self._recompute_genes(plan, st, contract, obj)
        await self._sampled_result(plan, st, contract, obj)
        self._resolve_release_alias(st, contract, obj)
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
        await self._row_releases(plan, st, contract, t, rows_now)
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
        # T7-T14 and placement (a pull of the derived sample is served derived: the gateway chose its cells)
        return self._finish_rows(plan, st, contract, obj, rows_now, counters,
                                 served_by="derived" if st.sample else "upstream",
                                 groups=processed, mapper=mapper, record=record, witness=w, returned_raw=returned_raw)

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
        # an argument whose echo accepts a redirect (a merged NCT ID answered by its surviving record) is checked by
        # that echo (W4), not re-applied to the row: T4 dropped the redirected record and answered empty (LIVE3-02)
        redirected = {arg for arg, spec in b.result.echo_specs().items()
                      if any(acc == "redirect" or acc.startswith("synonym:") for acc in spec.accept)}
        for arg, pred in st.per_arg.items():
            binding = contract.args.get(arg.split("+")[0]) if not arg.startswith("@") else None
            if arg in redirected:
                continue
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
        d = contract.binding.derived
        if d is not None and d.verb == "aggregate" and d.group_by:
            # one output row per group: the group is the row's key (under its renamed output names)
            rename = dict(d.rename or {})
            return [rename.get(c, c) for c in d.group_by]
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

    def _echo_set_missing(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any) -> list[Any] | None:
        """``result.echo_set``: the requested members of a set argument that came back neither returned nor
        withheld (an unknown PMID among known ones); None without an echo set or a list value."""
        es = contract.binding.result.echo_set
        if es is None:
            return None
        want = st.values.get(es.arg, plan.args_sent.get(es.arg))
        if not isinstance(want, list):
            return None
        got = {str(v) for v in jp_get(obj, es.path) if v is not None}
        held = {str(v) for v in jp_get(obj, es.withheld_path) if v is not None} if es.withheld_path else set()
        return [v for v in want if str(v) not in got and str(v) not in held]

    def _echo_set_none_found(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any) -> None:
        """Every member of an echoed set is missing from a source that decides existence (``existence: upstream``):
        ``not_found`` naming them, never an empty answer."""
        es = contract.binding.result.echo_set
        missing = self._echo_set_missing(plan, st, contract, obj)
        want = st.values.get(es.arg, plan.args_sent.get(es.arg)) if es is not None else None
        if not missing or not isinstance(want, list) or len(missing) < len(want) or \
                contract.args[es.arg].existence != "upstream":
            return
        payload = not_found_payload(es.arg, missing, None, list(contract.args[es.arg].accepts),
                                    ["upstream: no record returned"], [], None, plan.bound_table)
        payload["items"] = missing
        raise GatewayError(ErrorKind.not_found, f"{len(missing)} requested {es.arg} returned no record: "
                           f"{', '.join(map(str, missing[:5]))}", tool=st.name, payload=payload)

    def _listed_missing(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any, *,
                        rows_returned: bool) -> None:
        """``result.not_found_list``: the requested values upstream itself lists as found nowhere (search_genes
        ``genes_not_found``) are ``not_found_items`` and leave the resolution summary's resolved count; every
        requested value listed, with no rows, is ``not_found`` (``existence: upstream`` decides by upstream's own
        not-found; an unknown ENSG00000999999 was 'resolved' and answered as a successful empty, LIVE3-04)."""
        path = contract.binding.result.not_found_list
        if not path or st.listed_checked:
            return
        st.listed_checked = True
        listed = [x for v in jp_get(obj, path) for x in (v if isinstance(v, list) else [v]) if x is not None]
        if not listed:
            return
        asked: dict[str, list[Any]] = {}
        for arg, a in contract.args.items():
            if a.existence != "upstream":
                continue
            vals = plan.args_sent.get(arg, plan.args_raw.get(arg))
            if isinstance(vals, list) and vals:
                asked[arg] = list(vals)
        names = {str(v): arg for arg, vals in asked.items() for v in vals}
        missing = [v for v in listed if str(v) in names]
        if not missing:
            return
        for arg in {names[str(v)] for v in missing}:
            gone = [v for v in missing if names[str(v)] == arg]
            rs = st.resolution_summary.get(arg)
            if isinstance(rs, dict):
                rs["resolved"] = max(0, int(rs.get("resolved") or 0) - len(gone))
                rs["unresolved"] = list(rs.get("unresolved") or []) + gone
        requested = sum(len(v) for v in asked.values())
        if not rows_returned and len(missing) >= requested:
            arg = names[str(missing[0])]
            payload = not_found_payload(arg, missing, None, list(contract.args[arg].accepts),
                                        [f"upstream: listed under {path}"], [], None, plan.bound_table)
            payload["items"] = missing
            raise GatewayError(ErrorKind.not_found, f"{len(missing)} requested value(s) found nowhere: "
                               f"{', '.join(map(str, missing[:5]))}", tool=st.name, payload=payload)
        st.not_found_items = list(st.not_found_items or []) + [v for v in missing
                                                               if v not in (st.not_found_items or [])]
        st.notes.append(f"{len(missing)} requested value(s) found nowhere (upstream's {path}): "
                        f"{', '.join(map(str, missing[:10]))}")

    def _witness_checks(self, plan: CallPlan, st: _CallState, contract: ToolContract, t: Any, rows: list[Any],
                        returned_raw: int, obj: Any) -> tuple[str, str] | None:
        b = contract.binding
        w = st.witness
        es = b.result.echo_set
        missing = self._echo_set_missing(plan, st, contract, obj)
        if missing is not None and not rows:
            self._echo_set_none_found(plan, st, contract, obj)
        self._listed_missing(plan, st, contract, obj, rows_returned=bool(rows))
        if missing is not None:
            got = [str(v) for v in jp_get(obj, es.path) if v is not None]
            want = {str(v) for v in st.values.get(es.arg, plan.args_sent.get(es.arg)) or []}
            extra = [g for g in got if g not in want]
            st.checks.append((f"echo_set:{es.arg}", not missing and not extra,
                              f"{len(missing)} requested not returned, {len(extra)} returned unrequested"))
            if extra and es.mode in ("subset", "equal"):
                return "W4", f"returned {es.arg} that were not requested: {', '.join(extra[:5])}"
            if missing:
                st.not_found_items = list(st.not_found_items or []) + missing
                st.notes.append(f"{len(missing)} requested {es.arg} returned no record: "
                                f"{', '.join(map(str, missing[:10]))}")
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
                    st.redirected = True
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
        if b.result.kind == "count":
            # a count answer has no rows: its total was compared with the witness's (classify, contradiction)
            return None
        # W1 empty
        if not rows and total > 0 and returned_raw == 0:
            return "W1", f"0 rows returned, the witness counted {total}"
        # W5 one-to-many
        if b.result.kind == "record" and total > 1 and len(rows) == 1:
            return "W5", f"one record returned, the witness found {total}"
        keys = self._keys_of(rows, cols, st) if cols else []
        # W2 outside (a record the echo accepted as a redirect is keyed by its surviving ID, which the witness of the
        # requested ID does not hold)
        if w.key_set is not None and keys and not st.redirected:
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
        # W6 short page (a declared preview lists at most ``preview`` rows of what the tool wrote)
        limit = st.sent_limit
        expected = total if limit is None else min(total, limit)
        cap = _preview_cap(b.result.preview, obj)
        if cap is not None:
            expected = min(expected, cap)
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
        """W6: one re-call with the corrected limit (never for a tool that writes its rows to a file: the re-call
        would overwrite the agent's file with every match). The limit sent is recorded in ``args_sent``."""
        st.attempts += 1
        w = st.witness
        name = contract.limit_arg
        if self.bridge is None or w is None or w.total is None or name is None or self._writes_rows(contract):
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
        plan.args_sent = args
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
        live_sections: dict[str, dict[str, Any]] = {}
        for sname, sec in d.sections.items():
            key = {col: (self._stored(st.results[arg], sec.table) if arg in st.results else
                         st.values.get(arg, plan.args_sent.get(arg, plan.gateway_args.get(arg))))
                   for col, arg in sec.key_from_args.items()}
            spec = {"path": sec.path, "table": sec.table, "verb": sec.verb, "single": sec.single,
                    "key": json_value(key), "value": sec.value}
            # a live table's section is one more read of that table (the data child serves no sections on it)
            (live_sections if self._live_table(sec.table) else sections)[sname] = spec
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
        ranks = [_rank_json(o) for o in st.order]
        made = _nest_outputs(nest)
        if nest is not None and any(r["column"] in made for r in ranks):
            # the order ranks the groups by what the nest computes (evidence_count desc, then disease): the data
            # child ranks the groups before it cuts them to the limit, and reads the rows by the stored columns only
            nest["order"] = [{**r, "column": d.rename.get(r["column"], r["column"])} for r in ranks]
            ranks = [r for r in ranks if r["column"] not in made]
        # a search tool called with only its key argument (biosample_id, no query) is a lookup by key
        verb = "find" if d.verb == "search" and not st.search_text else d.verb
        req = ServeRequest(table=d.table, verb=verb, predicate=to_json(pred) if pred else None,
                           columns=list(d.columns), order=[RankKeyModel(**r) for r in ranks],
                           limit=limit, limit_grain=limit_grain, group_by=within + list(d.group_by),
                           explode=list(d.explode), carry=list(d.carry), rename=dict(d.rename), split=d.split,
                           nest=nest, aggregate=dict(d.aggregate), sections=sections, anchor=anchor, anchors=anchors,
                           search_text=st.search_text,
                           params=self._params(plan, st),
                           budget_bytes=self.settings.witness.repair_max_bytes)
        resp = await self.service.serve(req)
        if _serve_error(resp):                         # the handler refused the request: its own kind, not an outage
            raise GatewayError.from_envelope(self._as_argument(contract, d.table, _serve_error(resp))).with_tool(
                st.name)
        if str(resp.reason or "").startswith("too_large"):
            # over the scan budget the child answers no rows and no total: a refusal, never an empty answer
            why = str(resp.reason).partition(":")[2].strip() or str(resp.reason)
            raise GatewayError(ErrorKind.too_large, f"{d.table}: {why}", tool=st.name, subkind="scan_budget",
                               payload=too_large_payload("scan_budget", limit_mb=round(
                                   self.settings.witness.repair_max_bytes / 1e6, 1),
                                   hint="narrow the query (a more specific filter, a smaller limit), or read the "
                                        "rows with mcp__data__find", alternative="mcp__data__find"))
        for name in ("_statistics", "_propagation"):
            # what the handler did differently from its contract (annotations not propagated for want of the
            # hierarchy, a caller's universe) is said in the header, not only in the provenance record
            rec = (resp.sections or {}).get(name)
            for note in (rec.get("notes") or [] if isinstance(rec, Mapping) else []):
                if isinstance(note, str) and note not in st.notes:
                    st.notes.append(note)
        if isinstance(resp.rows, list) and verb == "find" and self._live_table(d.table) and not (
                d.aggregate or d.split or d.nest or d.group_by or d.explode):
            # rows of a live table as the source holds them: a key the call names and the rows lack is unknown there
            self._requested_missing(plan, st, contract, d, pred, resp)   # before the other reads: refused early
        if d.compose and isinstance(resp.rows, list):
            await self._compose(plan, st, contract, d, pred, resp)
        if live_sections:
            await self._live_sections(st, live_sections, resp)
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
            for arg, when in d.rows_when.items():
                if _truthy_arg(plan.args_raw.get(arg)):
                    path = when                        # upstream's key for this call (therapeutic_areas)
                    break
            st.rows_path = path
            if path != "$":
                obj = place_rows(obj, path, rows, record=b.result.kind == "record")
            elif b.result.kind == "record":
                obj = {**obj, **rows[0]} if rows and isinstance(rows[0], Mapping) else obj
            elif obj:
                obj["rows"] = rows                     # an envelope is declared: rows under "rows"
            else:
                obj = rows  # type: ignore[assignment]
        if isinstance(obj, dict):
            _declare_counts(obj, b.result.count_fields)
        st.section_meta = {k: dict(v) for k, v in dict(resp.sections.get("_section_meta") or {}).items()
                           if k in d.sections}
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

    @staticmethod
    def _as_argument(contract: ToolContract, table: str, env: Mapping[str, Any]) -> dict[str, Any]:
        """A derived read's error names a column of its table (``studyId``); the caller passed an argument
        (``study_id``): the envelope names the argument that binds that column."""
        out = dict(env)
        col = out.get("argument")
        arg = next((n for n in contract.args for tb, c in contract.arg_columns(n) if tb == table and c == col),
                   None) if col else None
        if arg is not None:
            out["argument"] = arg
        return out

    def _live_table(self, ref: str | None) -> bool:
        """The table's layout reads a live source page by page (``live`` without ``scan``: CT.gov, cBioPortal)."""
        if not ref or self.registry is None:
            return False
        try:
            t = self.catalog.table(ref)
            layout = self.registry.find("layout", t.layout)
        except Exception:  # noqa: BLE001
            return False
        caps = set(getattr(layout, "capabilities", ()) or ())
        return "live" in caps and "scan" not in caps

    async def _compose(self, plan: CallPlan, st: _CallState, contract: ToolContract, d: Any, pred: Predicate | None,
                       resp: ServeResponse) -> None:
        """``derived.compose``: each sub-derivation reads its own table with the conjuncts of the call that name its
        columns (and the key values of the rows read so far), and its rows are merged into the main rows on the sub
        table's key (a later sub-read wins a name both carry, as upstream's ``data.update``). get_clinical_data: the
        study's samples, then their sample-level attributes on ``(studyId, sampleId)``, then their patients'
        attributes on ``(studyId, patientId)``."""
        rows = [r for r in resp.rows if isinstance(r, dict)]    # type: ignore[union-attr]
        for sub in d.compose:
            try:
                t = self.catalog.table(sub.table)
            except Exception as exc:  # noqa: BLE001
                raise GatewayError(ErrorKind.tool_defect, f"compose: {sub.table} is not in the catalog ({exc})",
                                   tool=st.name) from None
            key = [k for k in t.key if not k.endswith("#")]
            names = set(key) | set(t.columns) | set(getattr(t.spec.pivot, "index", None) or [])
            parts = [p for p in (list(pred.preds) if isinstance(pred, And) else [pred] if pred is not None else [])
                     if set(_columns_of(p)) <= names]
            fixed = {c for p in parts for c in _columns_of(p)}
            for k in key:
                values = sorted({str(r.get(k)) for r in rows if r.get(k) not in (None, "")})
                if k not in fixed and values:
                    parts.append(In(k, tuple(values)))
            sub_pred = None if not parts else (parts[0] if len(parts) == 1 else And(tuple(parts)))
            req = ServeRequest(table=sub.table, verb="find", predicate=to_json(sub_pred) if sub_pred else None,
                               columns=list(sub.columns), budget_bytes=self.settings.witness.repair_max_bytes)
            got = await self.service.serve(req)
            if _serve_error(got):
                raise GatewayError.from_envelope(_serve_error(got)).with_tool(st.name)
            if got.truncated or str(got.reason or "").startswith("too_large"):
                st.force_partial = True
                st.notes.append(f"{sub.table}: not every row was read within the source's budget; some rows lack its "
                                "fields")
            by_key = {tuple(str(r.get(k)) for k in key): r for r in got.rows if isinstance(r, Mapping)}
            for r in rows:
                hit = by_key.get(tuple(str(r.get(k)) for k in key))
                if hit is not None:
                    r.update({c: v for c, v in hit.items() if c not in key})
            st.transforms.append(f"composed {sub.table} on ({', '.join(key)})")

    async def _live_sections(self, st: _CallState, live: Mapping[str, Mapping[str, Any]], resp: ServeResponse) -> None:
        """Sections over live tables, each read on its own (a ``value`` keeps that field of each row: get_clinical_data's
        ``clinical_attributes`` is the list of the study's attribute ids, as upstream lists them)."""
        meta = resp.sections.setdefault("_section_meta", {})
        for name, sec in live.items():
            parts = [Eq(str(c), v) for c, v in dict(sec.get("key") or {}).items()]
            pred = None if not parts else (parts[0] if len(parts) == 1 else And(tuple(parts)))
            try:
                got = await self.service.serve(ServeRequest(table=str(sec["table"]), verb="find",
                                                            predicate=to_json(pred) if pred else None))
            except ServiceError as exc:
                resp.sections[name] = {"_vbt_unavailable": f"{sec['table']} not read: {exc.message[:200]}"}
                meta[name] = {"status": "not_ready", "table": sec["table"]}
                continue
            err = _serve_error(got)
            if err:
                resp.sections[name] = {"_vbt_unavailable": str(err.get("message") or err)[:300]}
                meta[name] = {"status": "not_ready", "table": sec["table"]}
                continue
            rows = [r for r in got.rows if isinstance(r, Mapping)]
            field = sec.get("value")
            resp.sections[name] = [r.get(field) for r in rows] if field else rows
            meta[name] = {"status": "ok" if rows else "empty", "table": sec["table"], "total": got.total,
                          "truncated": bool(got.truncated)}

    def _requested_missing(self, plan: CallPlan, st: _CallState, contract: ToolContract, d: Any,
                           pred: Predicate | None, resp: ServeResponse) -> None:
        """Keys the call names (every key column of the derived table fixed by ``eq``/``in``) that the derived rows
        do not hold: unknown to the source. Refused ``not_found`` naming them (``on_unknown_items``), as upstream's
        phantom rows are, unless the read was cut by the source's budget (then they are only unread)."""
        try:
            t = self.catalog.table(d.table)
        except Exception:  # noqa: BLE001
            return
        key = [k for k in t.key if not k.endswith("#")]
        asked: dict[str, list[Any]] = {}
        for p in (list(pred.preds) if isinstance(pred, And) else [pred] if pred is not None else []):
            col = getattr(p, "column", None)
            if col in key and isinstance(p, Eq):
                asked[col] = [p.value]
            elif col in key and isinstance(p, In):
                asked[col] = list(getattr(p, "values", ()))
        if set(asked) != set(key) or resp.truncated:
            return
        combos: list[tuple[Any, ...]] = [()]
        for k in key:
            combos = [(*c, v) for c in combos for v in asked[k]]
        seen = {tuple(str(r.get(k)) for k in key) for r in resp.rows if isinstance(r, Mapping)}  # type: ignore[union-attr]
        missing = [c for c in combos if tuple(str(v) for v in c) not in seen]
        if not missing:
            return
        varying = [i for i, k in enumerate(key) if len(asked[k]) > 1] or [len(key) - 1]
        items = [c[varying[0]] for c in missing]
        arg = next((n for n in contract.args for tb, col in contract.arg_columns(n)
                    if tb == d.table and col == key[varying[0]]), None)
        if contract.binding.result.on_unknown_items == "not_found":
            payload = not_found_payload(arg or "", items, None, [], ["derived: no such record at the source"], [],
                                        None, d.table)
            payload["items"] = items
            raise GatewayError(ErrorKind.not_found, f"{len(items)} requested item(s) do not exist: "
                               f"{', '.join(map(str, items[:5]))}", tool=st.name, payload=payload)
        st.notes.append(f"{len(items)} requested item(s) not at the source")
        st.not_found_items = items

    # ---------------------------------------------------------------- shared tail

    def _finish_rows(self, plan: CallPlan, st: _CallState, contract: ToolContract, obj: Any, rows: list[Any],
                     counters: Counters, *, served_by: str, serve: ServeResponse | None = None,
                     groups: list[tuple[str, list[Any]]] | None = None, mapper: FieldMapper | None = None,
                     record: bool = False, witness: WitnessResponse | None = None,
                     returned_raw: int | None = None) -> DataResult:
        b = contract.binding
        t = self._rows_table(plan, contract)
        cols = self._key_columns(contract, t)
        types = [st.storage_types.get(c) for c in cols]
        mapper = mapper or self._mapper(contract)
        upstream = serve is None
        groups = groups if groups is not None else [(st.rows_path or (b.result.row_paths[0] if b.result.row_paths
                                                                      else "$"), rows)]
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
            # one cap per column: the overlay's (``$.descendants``), else relation_list_max for a record's lists
            trims: dict[str, Any] = {}
            for path, spec in b.result.trim.items():
                trims.setdefault(trim_column(path), spec)
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
                    # a record whose one row was withheld (T1 leakage, T3 existence) or removed (T4) is gone: the
                    # upstream payload would show it in the text the model reads
                    obj = out_rows[0] if out_rows else {}
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
                             vetoed=vetoed, arg_echo=b.result.arg_echo, args_raw=plan.args_raw, counters=counters,
                             upstream_returned=returned_raw if upstream else None)
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
        if st.family_rows and status in ("ok", "empty", "empty_unverified"):
            status = "partial"                         # rows elsewhere in the family: never a complete or empty answer
        self._self_withheld(st, contract, obj, counters)
        if st.leakage is not None and st.leakage.current and total is not None:
            total, total_method, status = self._ceiling_safe_total(st, contract, obj, total, total_method, status)
        t = self._rows_table(plan, contract)
        cols = list(cols if cols is not None else self._key_columns(contract, t))
        types = [st.storage_types.get(c) for c in cols]
        desc = t.descriptor if t is not None else _one_source(contract)
        release = self._release_of(desc, st)
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
        nested = self._nested_record_column(contract, t)
        if nested is not None:
            # a record read from a nested column (``rows: $.tep``) never inherits its table's coverage (rev 2:
            # nested containers default to unknown); a coverage the column declares applies instead
            coverage, statement = _nested_coverage(nested, coverage)
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
        # derived rows carry the renamed columns (``rename: {disease: disease_id}``): read the grain there
        rename = dict(getattr(b.derived, "rename", None) or {}) if b is not None and b.derived is not None and \
            served_by == "derived" else {}
        # an aggregate's rows are its groups: the table's grains are not counted on them
        grouped = b is not None and b.derived is not None and serve is not None and b.derived.verb == "aggregate"
        if t is not None and not grouped:
            for g, spec in t.spec.grains.items():
                gcols = list(spec) if isinstance(spec, list) else list(spec.columns or [])
                if not gcols:
                    continue
                rcols = [rename.get(c, c) for c in gcols]
                served_g = serve.grains.get(g) if serve is not None else None
                if served_g is not None and served_g.get("returned") is not None and not st.derived_withheld:
                    # the data child counted the returned matches as it counts the total (canonicalised, list
                    # elements, nulls excluded); its output rows may be renamed or projected
                    returned_g = int(served_g["returned"])
                elif not isinstance(spec, list) and spec.canonicalize and rows:
                    returned_g = None                  # parent families are known to the data child only
                else:
                    returned_g = len({v for r in rows if isinstance(r, Mapping) for v in grain_values(r, rcols)})
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
            # ties are broken by the declared keys that follow (search: match_class, then approvedSymbol)
            then = "".join(f", then {x['column']} {x.get('direction', 'desc')}"
                           for x in (_rank_json(r) for r in st.order[1:]))
            order_text = f"{o['column']} {o.get('direction', 'desc')}" + \
                (f" within {', '.join(within)}" if within else "") + then + f" ({how})"
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
        native = self._native_provenance(st, obj) if st.native_header else None
        # provenance
        prov = self._record(plan, st, contract, t if native is None else native["table"], status=status, rows=rows,
                            cols=cols if native is None else native["key"],
                            types=types if native is None else [st.storage_types.get(c) for c in native["key"]],
                            total=total,
                            total_method=total_method, truncated=truncated, coverage=coverage, statement=statement,
                            served_by=served_by, order=st.order[0] if st.order else None, verified=verified,
                            obj=obj, withheld_leakage=counters.withheld.get("leakage", 0),
                            leakage_unchecked=counters.leakage_unchecked,
                            derived=records, native=native)
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
            family_rows=dict(st.family_rows) or None, not_found_items=st.not_found_items,
            pooled_over=list(pooled) or None,
            removed_fields=dict(counters.removed_fields) or None, trimmed=dict(counters.trimmed) or None,
            undefined=(next(iter(st.undefined.values())) if st.undefined else None),
            coverage=coverage,  # type: ignore[arg-type]
            coverage_statement=statement, evidence=evidence.caveat if evidence is not None else None,
            served_by=served_by, hash_seed=self._hash_seed(plan.server), notes=notes, cite=cite, prov=prov.id)
        if st.native_header:
            # a native data tool: the data child's source, totals and serving, not "upstream"/"unknown"
            nh = st.native_header
            for name in ("source", "total", "total_method", "served_by", "truncated", "coverage", "coverage_statement",
                         "tables", "key"):
                if nh.get(name) is not None:
                    setattr(header, name, nh[name])
            if isinstance(nh.get("withheld"), Mapping):
                # rows the child withheld (the evidence ceiling of a live source) are this answer's withheld rows
                header.withheld = {**dict(header.withheld or {}), **{str(k): int(v) for k, v in nh["withheld"].items()}}
            if nh.get("notes"):
                # the child's disclosures (a default filter it added, an engine match, a ceiling, missing keys)
                header.notes = list(dict.fromkeys([*map(str, nh["notes"]), *header.notes]))
            if isinstance(nh.get("leakage"), Mapping):
                header.extra["leakage"] = dict(nh["leakage"])
        if st.ceiling_totals:
            header.extra["ceiling_totals"] = dict(st.ceiling_totals)
        if st.soft_sections:
            header.extra["unavailable_sections"] = dict(st.soft_sections)
        if st.section_meta:
            # each section read on its own: its status, total and coverage, never the main table's
            header.extra["sections"] = json_value(st.section_meta)
            header.tables = list(dict.fromkeys((header.tables or []) + [
                _last(str(m["table"])) for m in st.section_meta.values() if m.get("table")]))
            blind = sorted(n for n, m in st.section_meta.items() if m.get("status") == "empty" and
                           m.get("coverage") != "covered")
            if blind and header.cite:
                header.cite += (f"; empty section(s) {', '.join(blind)} are not evidence of absence "
                                "(coverage unknown)")
        return DataResult.build(obj, header, prov)

    def _native_provenance(self, st: _CallState, obj: Any) -> dict[str, Any]:
        """What a native data tool's provenance names, from the child's own ``_vbt`` and body: the table it read,
        its key, the source and release (``clinicaltrials_gov@<dataTimestamp>``), the totals' method, the
        serving, and the live record versions (replay reports a cited row whose version moved)."""
        nh = st.native_header or {}
        ref = next(iter(nh.get("tables") or []), None)
        table = None
        if isinstance(ref, str):
            with contextlib.suppress(Exception):
                table = self.catalog.table(ref)
        name, _, release = str(nh.get("source") or "").partition("@")
        versions = obj.get("record_versions") if isinstance(obj, Mapping) else None
        return {"table": table, "ref": ref, "key": [str(k) for k in nh.get("key") or [] if not str(k).endswith("#")],
                "source": name or None, "release": release or None,
                "total_method": nh.get("total_method"), "served_by": nh.get("served_by"),
                "record_versions": {str(k): {str(i): str(v) for i, v in dict(m).items()}
                                    for k, m in dict(versions).items() if isinstance(m, Mapping)}
                if isinstance(versions, Mapping) else None,
                "leakage": dict(nh["leakage"]) if isinstance(nh.get("leakage"), Mapping) else None}

    @staticmethod
    def _release_of(desc: Any, st: _CallState) -> str | None:
        """The release a result names: the descriptor's pinned one, else what the call observed of a live
        source (the remote witness's ``as_of``: CT.gov ``dataTimestamp``; the release of derived rows read
        from a live table; the Census release count-first resolved), never an alias such as ``stable``."""
        if desc is None:
            return None
        if desc.release.expect:
            return str(desc.release.expect)
        w = st.witness
        if w is not None and getattr(w, "as_of", None):
            return str(w.as_of)
        if st.serve_as_of:
            return str(st.serve_as_of)
        cf = (st.count_first or {}).get("release") if isinstance(st.count_first, Mapping) else None
        if isinstance(cf, Mapping) and cf.get("resolved"):
            return str(cf["resolved"])
        if st.census_release:
            return st.census_release
        lr = (st.live_release.get(str(getattr(desc, "source", None))) or {}) if st.live_release else {}
        if lr.get("resolved"):
            return str(lr["resolved"])                 # CT.gov dataTimestamp; the one cBioPortal study's importDate
        return None

    def _record(self, plan: CallPlan, st: _CallState, contract: ToolContract, t: Any, *, status: str,
                rows: Sequence[Any], cols: Sequence[str], types: Sequence[str | None], total: int | None,
                total_method: str, truncated: bool, coverage: str, statement: str | None, served_by: str,
                order: Any, verified: bool | None, obj: Any, withheld_leakage: int = 0,
                leakage_unchecked: int = 0, derived: dict[str, Any] | None = None,
                native: Mapping[str, Any] | None = None) -> DataProvenance:
        b = contract.binding
        desc = t.descriptor if t is not None else _one_source(contract)
        self._stamp_times(st)
        tables = []
        refs = tables_read(contract, plan.bound_table, plan.args_raw)
        if native is not None and native.get("ref"):
            refs = [str(native["ref"])]                # a native tool reads the table it names
        for ref in refs:
            ct = contract.tables.get(ref) or (native.get("table") if native is not None else None)
            phys, _ = self.readiness.physical(ref)
            m = self.readiness.get(phys)
            rs = b.reads.get(ref) if b is not None else None
            access = ("derived_scan" if plan.route == "derived" else "native" if native is not None else
                      f"upstream_{rs.access}" if rs is not None else "witness_scan")
            pfs = dict(m.partition_fingerprints) if m is not None else {}
            read = (partitions_selected(plan.scope, pfs) or sorted(pfs)) if pfs else None
            tables.append(TableInfo(name=_last(ref), layout=ct.layout if ct is not None else None,
                                    fingerprint=m.fingerprint if m is not None else None, access=access,
                                    partitions_read=read,
                                    partition_fingerprints={p: pfs[p] for p in read} if read else None,
                                    lineage=(ct.spec.lineage.model_dump() if ct is not None and ct.spec.lineage
                                             else None)))
        o = _rank_json(order) if order is not None else {}
        versions = _record_versions(t, rows, cols)
        lr = st.live_release.get(desc.source) or {} if desc is not None else {}
        if isinstance(lr.get("per"), Mapping):
            # the records the call depends on (a cBioPortal study's importDate): replay reports a moved one
            versions = {**dict(versions or {}), **{str(k): {str(i): str(v) for i, v in dict(m).items()}
                                                   for k, m in lr["per"].items() if isinstance(m, Mapping)}}
        if native is not None:
            # the child's own header is the truth for a native tool: its source and release, totals and serving
            served_by = str(native.get("served_by") or served_by)
            total_method = str(native.get("total_method") or total_method)
            versions = native.get("record_versions") or versions
        prov = DataProvenance(
            tool=st.name, server=plan.server, mode=st.mode, profile=self.profile, gateway_version=GATEWAY_VERSION,
            served_by=served_by,
            source=SourceInfo(name=(native or {}).get("source") or (desc.source if desc else None),
                              release=(native or {}).get("release") or self._release_of(desc, st),
                              versions={str(k): str(v) for k, v in dict(lr.get("versions") or {}).items()},
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
                              key_columns=list(cols),
                              key_storage_types=list(types) if cols and any(types) else None,
                              transforms=list(st.transforms), record_versions=versions),
            memory=st.admission.to_record() if st.admission is not None else {"admission": "not_applicable"},
            leakage=(dict(native["leakage"]) if native is not None and native.get("leakage") else
                     self._leakage_record(st.leakage, withheld_leakage, leakage_unchecked)
                     if st.leakage is not None and st.leakage.recorded else None),
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

    @staticmethod
    def _leakage_record(lk: LeakagePlan, withheld: int, unchecked: int) -> dict[str, Any] | None:
        """The provenance ``leakage`` block: the data ceiling when it bounds (or cannot date) a source the tool
        reads, else the ceiling the source's own server applied (PubMed's literature ceiling)."""
        data = lk.active or bool(lk.ceiling is not None and lk.undated)
        return leakage_record(lk.ceiling if data else lk.self_ceiling, withheld, (lk.risk or unchecked > 0) and data,
                              lk.reason if data else None)

    def _self_withheld(self, st: _CallState, contract: ToolContract, obj: Any, counters: Counters) -> None:
        """``result.withheld_list``: the rows the source's server withheld under its own ceiling (PubMed
        ``withheld``) are counted as ``withheld.leakage`` in the header and the provenance record (LIVE3-09)."""
        path = contract.binding.result.withheld_list if contract.binding is not None else None
        lk = st.leakage
        if not path or lk is None or lk.self_ceiling is None or not isinstance(obj, (dict, list)) or st.self_counted:
            return
        st.self_counted = True
        n = len([x for v in jp_get(obj, path) for x in (v if isinstance(v, list) else [v]) if x is not None])
        if n:
            counters.withheld["leakage"] = counters.withheld.get("leakage", 0) + n
            st.notes.append(f"{n} record(s) withheld by the source's own evidence ceiling {lk.self_ceiling} "
                            f"(listed in {path.removeprefix('$.')})")

    def _ceiling_safe_total(self, st: _CallState, contract: ToolContract, obj: Any, total: int, total_method: str,
                            status: str) -> tuple[int | None, str, str]:
        """A call that selects on the record as it is today (a status, a later date, posted results) under the
        evidence ceiling (LIVE3-01): a first-posted bound does not bound it. The records also unchanged since the
        ceiling (``ceiling_totals.available_and_unchanged``) held the same values then, so they are the total (a
        lower bound: partial); without that count the answer is partial with ``leakage.risk``."""
        lk = st.leakage
        b = contract.binding
        fields = ", ".join(lk.current)
        safe = (st.ceiling_totals or {}).get("available_and_unchanged")
        status = "partial" if status in ("ok", "empty") and (total or safe) else status
        if safe is None:
            lk.risk = True
            lk.reason = "selects on the current record"
            st.notes.insert(0, f"the call selects on {fields}, which the source holds only as it is today: its total "
                            f"({total}) is not bounded by the evidence ceiling {lk.ceiling} (leakage risk)")
            return total, total_method, status
        st.notes.insert(0, f"the call selects on {fields}, which the source holds only as it is today: under the "
                        f"evidence ceiling {lk.ceiling} the total counts the {safe} matching record(s) also last "
                        f"changed by then (their value then is their value now); the {total} first posted by then "
                        "include records whose value changed after it")
        spec = b.result.total if b is not None else None
        path = spec if isinstance(spec, str) else getattr(spec, "path", None)
        if path and isinstance(obj, dict) and isinstance(jp_first(obj, path), (int, float)):
            jp_set(obj, path, int(safe))               # the reply's own count, not only the header's
        return int(safe), "ceiling_unchanged", status

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
            **({"quarantined": [q.to_json() for q in self.catalog.quarantined]}
               if getattr(self.catalog, "quarantined", None) else {}),
        }


# ====================================================================== helpers

def compact_native_listing(schema: Mapping[str, Any]) -> dict[str, Any]:
    """A native verb's listed schema without the per-table column map of its ``where`` (``x-vbt-where``): every
    table's columns with their roles, identifier types and operators. The model reads a tool's whole schema on every
    request; on the shipped catalog (94 tables, 791 columns) the map was 128 KB in each of ``find`` and
    ``aggregate``, about 70,000 tokens per specialist request with ``similar`` and ``neighbors``, more than a 32K
    window holds and a large share of a 262K one on every agent. ``mcp__data__describe`` returns one table's columns
    (with each column's ``ops``) on demand; the ``where`` description names it. Nothing else reads the listed map
    (the data child resolves ``where`` against the catalog). The listing builds the schema without the map
    (``annotate_schema(column_maps=False)``); this strips one built elsewhere."""
    out = dict(schema)
    props = out.get("properties")
    if not isinstance(props, Mapping):
        return out
    if set(props) == {"request"} and isinstance(props["request"], Mapping):     # the hidden-verb form
        return {**out, "properties": {"request": compact_native_listing(props["request"])}}
    where = props.get("where")
    if isinstance(where, Mapping) and "x-vbt-where" in where:
        slim = {k: v for k, v in where.items() if k != "x-vbt-where"}
        text = str(slim.get("description") or "")
        slim["description"] = text.replace("; per column: x-vbt-where)", ")").replace(
            " (per column: x-vbt-where)", "") + WHERE_POINTER
        out["properties"] = {**props, "where": slim}
    return out


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


def _declare_counts(obj: dict[str, Any], count_fields: Sequence[str] | Mapping[str, str]) -> None:
    """A derived reply carries upstream's top-level count fields (``count``, ``total_found``): each one absent from
    the envelope is declared here and T11 sets it (a ``len(<path>)`` field only when the reply holds that path)."""
    targets = count_fields.items() if isinstance(count_fields, Mapping) else ((p, None) for p in count_fields)
    for target, expr in targets:
        name = str(target)[2:] if str(target).startswith("$.") else None
        if not name or not name.isidentifier() or name in obj:
            continue
        m = re.fullmatch(r"len\(\$\.(\w+)\)", str(expr)) if expr is not None else None
        if expr is not None and (m is None or m.group(1) not in obj):
            continue
        obj[name] = None


def _truthy_arg(v: Any) -> bool:
    """A flag argument as upstream reads it (``if list_therapeutic_areas:``)."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _nest_outputs(nest: Mapping[str, Any] | None) -> set[str]:
    """The group fields a derived ``nest`` computes (``count_as``, ``first_item``): an order may rank by them."""
    if not nest:
        return set()
    out = {str(k) for k in (nest.get("first_item") or {})}
    if nest.get("count_as"):
        out.add(str(nest["count_as"]))
    return out


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


def _preview_cap(preview: Any, obj: Any) -> int | None:
    """The rows a declared preview lists at most: an int, or per rows path the cap of the path the reply used."""
    if isinstance(preview, int):
        return preview
    if isinstance(preview, Mapping):
        caps = [int(n) for path, n in preview.items() if isinstance(jp_first(obj, path), list)]
        return min(caps) if caps else None
    return None


def _serve_error(resp: Any) -> dict[str, Any] | None:
    """The typed error a ``_serve`` reply carries (``ServeResponse.error``, ``refusal`` on the wire: MCPBridge reads
    a reply with a top-level ``error`` key as a failed call)."""
    err = getattr(resp, "error", None)
    return dict(err) if isinstance(err, Mapping) else None


def _columns_of(p: Any) -> set[str]:
    from ..predicate import columns as predicate_columns

    try:
        return set(predicate_columns(p))
    except Exception:  # noqa: BLE001
        return set()


def _engine_column(a: Any) -> str | None:
    """The column (``source.table.path``) a free-text argument matched by the source's engine binds, else None."""
    if a.role == "free_text" and a.interpreted_as == "engine" and not a.engine_param and isinstance(a.binds, str) \
            and not a.escape:
        return a.binds
    return None


def _estimate_fields(est: Any) -> dict[str, Any]:
    """``count_first.estimate`` as the count request's estimate fields (bytes)."""
    return {"base_bytes": int(est.base_mb * 1024 * 1024), "row_bytes": int(est.cell_bytes),
            "value_bytes": float(est.value_bytes), "all_genes_cell_bytes": est.all_genes_cell_bytes,
            "read_all_bytes": est.read_all_bytes}


def _sample_facet(cf: Any) -> dict[str, Any] | None:
    """``count_first.sample`` as a mapping (None when the binding declares none)."""
    facet = None
    with contextlib.suppress(AttributeError):
        facet = cf.sample
    if facet is None:
        return None
    if hasattr(facet, "model_dump"):
        return dict(facet.model_dump(exclude_none=True))
    return dict(facet) if isinstance(facet, Mapping) else {}


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
