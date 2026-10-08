"""Request and response models of the data child's hidden verbs (§11.8). No pyarrow.

Hidden FastMCP tools on the ``data`` server, called through ``MCPBridge.call_raw("data",
verb, {"request": payload})``. Every verb takes exactly one argument named ``request``
(:data:`REQUEST_ARG`); payloads are these pydantic models serialised as JSON. Tables are
named ``"source.table"`` (item tables included); predicates use the compact JSON form of
:mod:`vbt.datalayer.predicate`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .settings import SETTINGS_ENV

__all__ = [
    "SETTINGS_ENV", "REQUEST_ARG", "TABLE_ERRORS", "VERB_STATS", "VERB_CHECK", "VERB_WITNESS", "VERB_SERVE", "VERB_BUILD_INDEX",
    "VERB_RESOLVE_REMOTE", "VERB_VOCAB", "VERB_AGGREGATE", "VERB_SIMILAR", "VERB_EXPAND", "VERB_ENRICH",
    "PHASE1_VERBS", "ServeVerb", "IpcModel", "RankKeyModel", "StatsRequest", "ColumnStatsModel", "TableStatsModel",
    "StatsResponse", "CheckRequest", "CheckItemModel", "KeyCheckModel", "TableCheckModel", "CheckResponse",
    "WitnessRequest", "WitnessResponse", "ServeRequest", "ServeResponse", "BuildIndexRequest", "BuildIndexResponse",
    "ResolveRemoteRequest", "ResolveRemoteResponse", "VocabRequest", "VocabResponse", "VERB_CENSUS_COUNT",
    "CensusCountRequest", "CensusCountResponse", "VERB_MODELS",
    "request_payload", "parse_response",
]

REQUEST_ARG = "request"
# Per-table failures of _stats and _check go on the wire under this name: the bridge reads a
# top-level ``errors`` key as a failed call (legacy envelopes), which would hide every other table.
TABLE_ERRORS = "table_errors"

VERB_STATS = "_stats"
VERB_CHECK = "_check"
VERB_WITNESS = "_witness"
VERB_SERVE = "_serve"
VERB_BUILD_INDEX = "_build_index"
VERB_RESOLVE_REMOTE = "_resolve_remote"
VERB_VOCAB = "_vocab"
VERB_AGGREGATE = "_aggregate"     # phase 2
VERB_SIMILAR = "_similar"         # phase 2
VERB_EXPAND = "_expand"           # phase 3
VERB_ENRICH = "_enrich"           # phase 3
PHASE1_VERBS = (VERB_STATS, VERB_CHECK, VERB_WITNESS, VERB_SERVE, VERB_BUILD_INDEX, VERB_RESOLVE_REMOTE, VERB_VOCAB)

ServeVerb = Literal["lookup", "find", "search", "members", "count", "aggregate", "similar", "compare", "view",
                    "expand"]
TotalMethod = Literal["scan", "footer", "index", "unknown"]
Existence = Literal["exists", "absent", "unknown"]
Depth = Literal["shallow", "standard", "deep"]


class IpcModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RankKeyModel(IpcModel):
    column: str
    direction: Literal["asc", "desc", "asc_abs", "desc_abs"] = "desc"
    nulls: Literal["last", "first"] = "last"
    statistic: str | None = None
    within: list[str] = []


# --------------------------------------------------------------------------- _stats

class StatsRequest(IpcModel):
    tables: list[str]


class ColumnStatsModel(IpcModel):
    uncompressed_bytes: int = 0
    num_values: int | None = None
    max_rep_level: int = 0
    max_def_level: int = 0
    kind: Literal["flat", "string", "nested", "dense_matrix", "sparse_matrix"] = "flat"
    storage_type: str | None = None
    null_count: int | None = None
    min: Any = None
    max: Any = None


class TableStatsModel(IpcModel):
    fingerprint: str
    signature: str | None = None
    partition_fingerprints: dict[str, str] = {}
    rows: int | None = None
    fragments: int = 0
    bytes_on_disk: int = 0
    columns: dict[str, ColumnStatsModel] = {}          # keyed by §6.4 leaf path
    row_bytes_p99: dict[str, int] = {}                 # per output grain ("row", item table, "cell")


class StatsResponse(IpcModel):
    tables: dict[str, TableStatsModel] = {}
    errors: dict[str, str] = Field(default_factory=dict, alias=TABLE_ERRORS)


# --------------------------------------------------------------------------- _check

class CheckRequest(IpcModel):
    tables: list[str] = []                             # empty: every table of every loaded source
    depth: Depth = "standard"


class CheckItemModel(IpcModel):
    name: str                                          # R1..R10, R4b, R5b, or a named sub-check
    ok: bool
    level: Literal["error", "warning", "info"] = "error"
    detail: str = ""
    hint: str = ""
    column: str | None = None
    partition: str | None = None


class KeyCheckModel(IpcModel):
    method: Literal["full", "sampled", "none"] = "sampled"
    ok: bool | None = None
    at: str | None = None
    duplicates: int | None = None
    null_counts: dict[str, int] = {}                   # nullable key parts
    detail: str = ""


class TableCheckModel(IpcModel):
    status: str                                        # ready | missing | partial | schema_drift | ... (§13)
    columns: dict[str, str] = {}                       # per column / nested leaf path
    containers: dict[str, str] = {}
    partitions: dict[str, str] = {}
    item_tables: dict[str, str] = {}
    checks: list[CheckItemModel] = []
    fingerprint: str | None = None
    partition_fingerprints: dict[str, str] = {}        # partition label -> fingerprint (partitioned layouts)
    signature: str | None = None
    confirmed: dict[str, Any] = {}                     # verified:false facts confirmed (True) or refuted (False)
    vocab: dict[str, str] = {}                         # column -> vocabulary snapshot id
    key_check: KeyCheckModel | None = None
    #: §6.4 leaf path -> storage type, from the footers the check read (an item table: its physical table's). The
    #: gateway types witness keys with them instead of asking ``_stats`` (14.5 s for the 25.09 target tables).
    storage_types: dict[str, str | None] = {}


class CheckResponse(IpcModel):
    tables: dict[str, TableCheckModel] = {}
    depth: Depth = "standard"
    hash_randomization: int | None = None              # sys.flags.hash_randomization of the child
    errors: dict[str, str] = Field(default_factory=dict, alias=TABLE_ERRORS)
    #: R8: descriptor and overlay files the child's catalog left out ({file, kind, name, error})
    quarantined: list[dict[str, Any]] = []


# --------------------------------------------------------------------------- _witness

class WitnessRequest(IpcModel):
    table: str                                         # a table or an item table (counts items)
    grain: str | None = None
    predicate: dict[str, Any] | None = None
    key: list[str] = []
    order: list[RankKeyModel] = []
    k: int | None = None
    group_by: list[str] = []                           # comparable groups: top-k per group
    distinct: list[str] = []
    grains: dict[str, list[str] | dict[str, Any]] = {}  # {name: [cols] | GrainSpec}
    unknown_columns: list[str] = []                    # columns whose unknown values are counted
    key_set_max: int | None = None
    budget_bytes: int | None = None
    params: dict[str, Any] = {}                        # values of Param() nodes in the predicate


class WitnessResponse(IpcModel):
    total: int | None = None
    total_method: TotalMethod = "unknown"
    topk: list[list[Any]] | dict[str, list[list[Any]]] = []   # canonical keys; per group when grouped
    key_set: list[list[Any]] | None = None
    distinct: dict[str, list[Any]] = {}
    excluded_unknown: dict[str, int] = {}              # per column, "_rows" for the total
    excluded_not_applicable: dict[str, int] = {}
    unknown_total: int | None = None
    distinct_counts: dict[str, int] = {}               # per grain over all matching rows
    group_totals: dict[str, int] = {}                  # canonical group key -> rows
    one_to_many: dict[str, int] = {}                   # canonical key -> rows sharing it
    scanned_bytes: int | None = None
    reason: str | None = None
    as_of: str | None = None                           # remote witness: the source's data release when it has one


# --------------------------------------------------------------------------- _serve

class ServeRequest(IpcModel):
    table: str
    verb: ServeVerb = "find"
    predicate: dict[str, Any] | None = None
    columns: list[str] = []
    order: list[RankKeyModel] = []
    limit: int | None = None
    limit_grain: str | None = None
    group_by: list[str] = []
    distinct: list[str] = []
    explode: list[str] = []
    carry: list[str] = []
    rename: dict[str, str] = {}
    split: dict[str, Any] | None = None
    nest: dict[str, Any] | None = None
    aggregate: dict[str, dict[str, str]] = {}          # aggregate: {output: {count_distinct|count|first|distinct: column}}
    sections: dict[str, dict[str, Any]] = {}
    anchor: dict[str, Any] | None = None               # {column, value} for similar
    anchors: list[dict[str, Any]] = []                 # every anchor argument: [{name, column, value}]
    search_text: str | None = None
    id_type: str | None = None
    params: dict[str, Any] = {}
    budget_bytes: int | None = None


class ServeResponse(IpcModel):
    rows: list[dict[str, Any]] | dict[str, list[dict[str, Any]]] = []   # dict when split into lists
    total: int | None = None
    truncated: bool = False
    key_columns: list[str] = []
    grains: dict[str, dict[str, int | None]] = {}      # {grain: {returned, total}}
    excluded_unknown: dict[str, int] = {}
    excluded_not_applicable: dict[str, int] = {}
    served_by: str = "derived"
    sections: dict[str, Any] = {}
    row_keys: list[list[Any]] = []
    reason: str | None = None
    error: dict[str, Any] | None = None                # a §12.1 envelope: the derived handler refused the request
    as_of: str | None = None                           # a live table's release (CT.gov dataTimestamp)


# --------------------------------------------------------------------------- _build_index

class BuildIndexRequest(IpcModel):
    source: str | None = None
    id_type: str | None = None
    table: str | None = None
    access_path: list[str] | None = None               # the columns of one declared access path
    force: bool = False

    @model_validator(mode="after")
    def _one_target(self) -> "BuildIndexRequest":
        resolver = self.source is not None and self.id_type is not None
        access = self.table is not None and self.access_path is not None
        if resolver == access:
            raise ValueError("give either {source, id_type} (resolver sidecar) or {table, access_path} "
                             "(row-group value index)")
        return self


class BuildIndexResponse(IpcModel):
    path: str
    rows: int
    fingerprint: str


# --------------------------------------------------------------------------- _resolve_remote

class ResolveRemoteRequest(IpcModel):
    source: str
    id_type: str
    values: list[str]
    accepts: list[str] = []


class ResolveRemoteResponse(IpcModel):
    resolutions: list[dict[str, Any]] = []
    existence: Existence = "unknown"


# --------------------------------------------------------------------------- _vocab

class VocabRequest(IpcModel):
    table: str
    column: str
    max_values: int | None = None


class VocabResponse(IpcModel):
    values: list[Any] = []                             # storage-typed values
    rendered: list[str] = []                           # shortest storage-typed rendering of each value
    fingerprint: str | None = None
    complete: bool = True
    storage_type: str | None = None
    counts: dict[str, int] | None = Field(default=None)


# --------------------------------------------------------------------------- _census_count (phase 4)

VERB_CENSUS_COUNT = "_census_count"


class CensusCountRequest(IpcModel):
    """Count-first admission of a single-cell pull (``service/verbs/census_count.py``), or, with
    ``genes_file``, the genes found in a written h5ad's ``var.feature_name``."""

    table: str
    value_filter: str | None = None
    predicate: dict[str, Any] | None = None
    n_genes: int | None = None
    max_cells: int | None = None
    row_bytes: int | None = None
    value_bytes: float | None = None
    cap_bytes: int | None = None
    sample: dict[str, Any] | None = None               # {max_cells, seed}
    genes_file: str | None = None
    genes: list[str] | None = None
    release_only: bool = False                         # only the dated release the alias names (no count)


class CensusCountResponse(IpcModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    table: str | None = None
    n_cells: int | None = None
    admissible: bool | None = None
    need_bytes: int | None = None
    cap_bytes: int | None = None
    release: dict[str, Any] | None = None
    reason: str | None = None
    genes_found: list[str] | None = None
    genes_not_found: list[str] | None = None


VERB_MODELS: dict[str, tuple[type[IpcModel], type[IpcModel]]] = {
    VERB_STATS: (StatsRequest, StatsResponse),
    VERB_CHECK: (CheckRequest, CheckResponse),
    VERB_WITNESS: (WitnessRequest, WitnessResponse),
    VERB_SERVE: (ServeRequest, ServeResponse),
    VERB_BUILD_INDEX: (BuildIndexRequest, BuildIndexResponse),
    VERB_RESOLVE_REMOTE: (ResolveRemoteRequest, ResolveRemoteResponse),
    VERB_VOCAB: (VocabRequest, VocabResponse),
    VERB_CENSUS_COUNT: (CensusCountRequest, CensusCountResponse),
}


def request_payload(request: IpcModel) -> dict[str, Any]:
    """Tool arguments for a verb call: ``{"request": <model JSON>}``."""
    return {REQUEST_ARG: request.model_dump(mode="json", by_alias=True, exclude_none=True)}


def parse_response(verb: str, data: Any) -> IpcModel:
    """Validate a verb's JSON response (a dict or JSON text) with its response model."""
    _, response = VERB_MODELS[verb]
    if isinstance(data, (str, bytes)):
        return response.model_validate_json(data)
    return response.model_validate(data)
