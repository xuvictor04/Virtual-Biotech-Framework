"""The ``vbt.dataprov/1`` provenance record of one gateway call (§15.1). No pyarrow.

The runtime writes the full record to ``<run>/logs/data_provenance/<tool_use_id>.json`` and
puts :meth:`DataProvenance.summary` (status, **coverage** and its statement, source@release,
table fingerprints, counts, ``row_keys_sha256``, evidence nature, leakage risk) into the
trace's ``tool_end.data_provenance``. The record id is a content hash, so the same record
always has the same ``dp_...`` id.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from .rowkey import canonical, canonical_sha256

__all__ = [
    "SCHEMA", "GATEWAY_VERSION", "SourceInfo", "TableInfo", "ResolutionRecord", "RequestInfo", "OrderInfo",
    "ResultInfo", "CheckRecord", "UpstreamInfo", "DataProvenance", "prov_id", "cap_row_keys", "utc_now",
]

SCHEMA = "vbt.dataprov/1"
GATEWAY_VERSION = "1.0"
ROW_KEY_SAMPLE = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SourceInfo:
    name: str | None = None
    release: str | None = None
    release_verified: bool | None = None
    descriptor_sha256: str | None = None
    overlay_sha256: str | None = None
    manifest_sha256: str | None = None
    scope_versions: dict[str, Any] = field(default_factory=dict)   # interactionResources.databaseVersion per source
    serving_source: str | None = None                               # data.sources.alias: the source that served it


@dataclass
class TableInfo:
    name: str
    layout: str | None = None
    fragments: int | None = None
    fingerprint: str | None = None
    access: str | None = None                      # upstream_full_table | witness_scan | derived_scan | remote
    partitions_read: list[str] | None = None
    partition_fingerprints: dict[str, str] | None = None
    lineage: dict[str, Any] | None = None          # {source, release, table} of an `implements` table
    key_check: dict[str, Any] | None = None        # {method, at}


@dataclass
class ResolutionRecord:
    arg: str
    raw: Any
    canonical: Any = None
    matched_id_type: str | None = None
    canonical_id_type: str | None = None
    rule: str | None = None
    family: list[str] | None = None
    hops: list[str] = field(default_factory=list)
    index_fingerprint: str | None = None
    existence: str | None = None                   # exists | absent | unknown


@dataclass
class RequestInfo:
    args_raw: dict[str, Any] = field(default_factory=dict)
    args_sent: dict[str, Any] = field(default_factory=dict)
    resolutions: list[ResolutionRecord] = field(default_factory=list)
    expansions: list[dict[str, Any]] = field(default_factory=list)   # {arg, term, relation, predicates, n_terms, terms_sha256, ...}
    scope: dict[str, Any] = field(default_factory=dict)
    selector: dict[str, Any] | None = None


@dataclass
class OrderInfo:
    by: str | None = None
    direction: str | None = None
    within: list[str] = field(default_factory=list)
    verified: bool | None = None
    source: str | None = None                      # witness | upstream_full_sort | source_server_side


@dataclass
class ResultInfo:
    status: str | None = None
    returned: int | None = None
    total: int | None = None
    total_method: str | None = None
    coverage: str | None = None
    coverage_statement: str | None = None
    truncated: bool | None = None
    order: OrderInfo | None = None
    key_columns: list[str] = field(default_factory=list)
    key_storage_types: list[str | None] | None = None   # storage types of the key columns, in key order
    row_keys: list[Any] = field(default_factory=list)
    row_keys_complete: bool = True
    row_keys_sha256: str | None = None
    row_keys_by_grain: dict[str, dict[str, Any]] | None = None   # levels: {patient: {key_columns, row_keys_sha256, ...}}
    output_sha256: str | None = None
    output_rows_sha256: str | None = None
    computed: dict[str, Any] | None = None         # {metric, anchor_keys, input_row_keys_sha256, values_sha256}
    statistics: dict[str, Any] | None = None       # {test, correction, family_m, universe, query, propagation}
    transforms: list[str] = field(default_factory=list)
    record_versions: dict[str, dict[str, str]] | None = None   # live tables: {source.table: {key: version}}


@dataclass
class CheckRecord:
    name: str
    ok: bool | None
    detail: str | None = None


@dataclass
class UpstreamInfo:
    commit: str | None = None
    server: str | None = None
    hash_seed: int | None = None
    flags_stripped: list[str] = field(default_factory=list)
    reviewed_commit: str | None = None                 # the commit the overlay's bindings were reviewed against


@dataclass
class DataProvenance:
    """One call's ``vbt.dataprov/1`` record (field order is the JSON key order)."""

    tool: str
    server: str
    schema: str = SCHEMA
    id: str | None = None
    tool_use_id: str | None = None
    mode: str | None = None                        # off | observe | enforce
    profile: str | None = None                     # safe | fidelity
    gateway_version: str = GATEWAY_VERSION
    served_by: str | None = None                   # upstream | derived | repaired | blocked
    source: SourceInfo = field(default_factory=SourceInfo)
    plugins: dict[str, str] = field(default_factory=dict)
    tables: list[TableInfo] = field(default_factory=list)
    request: RequestInfo = field(default_factory=RequestInfo)
    result: ResultInfo = field(default_factory=ResultInfo)
    checks: list[CheckRecord] = field(default_factory=list)
    memory: dict[str, Any] | None = None
    as_of: str | None = None
    retrieved_at: str | None = None
    leakage: dict[str, Any] | None = None          # {ceiling, withheld, risk}
    evidence_nature: dict[str, Any] | None = None  # {kind, caveat}
    upstream: UpstreamInfo = field(default_factory=UpstreamInfo)
    t_ms: dict[str, float] = field(default_factory=dict)
    derived: dict[str, Any] | None = None          # derived handlers' records: expansion, statistics, network, ...

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("derived"):
            d.pop("derived", None)                     # absent unless a derived handler recorded something
        # Key order of §15.1: schema, id, tool_use_id first.
        head = {k: d.pop(k) for k in ("schema", "id", "tool_use_id", "tool", "server")}
        return {**head, **d}

    def finalize(self) -> "DataProvenance":
        """Stamp ``retrieved_at`` (when unset) and the content id ``dp_<sha256[:12]>``."""
        if self.retrieved_at is None:
            self.retrieved_at = utc_now()
        self.id = prov_id(self.to_dict())
        return self

    def add_check(self, name: str, ok: bool | None, detail: str | None = None) -> None:
        self.checks.append(CheckRecord(name=name, ok=ok, detail=detail))

    def set_row_keys(self, keys: Sequence[Sequence[Any]], max_n: int,
                     storage_types: Sequence[str | None] | None = None) -> None:
        """Store canonical row keys (all up to ``max_n``, else a sample) and their hash."""
        kept, complete, sha = cap_row_keys(keys, max_n, storage_types)
        self.result.row_keys = kept
        self.result.row_keys_complete = complete
        self.result.row_keys_sha256 = sha

    @property
    def source_label(self) -> str | None:
        if not self.source.name:
            return None
        return f"{self.source.name}@{self.source.release}" if self.source.release else self.source.name

    def summary(self) -> dict[str, Any]:
        """The trace summary (``tool_end.data_provenance``)."""
        leakage_risk = bool(self.leakage.get("risk")) if isinstance(self.leakage, dict) else None
        return {
            "prov": self.id,
            "status": self.result.status,
            "coverage": self.result.coverage,
            "coverage_statement": self.result.coverage_statement,
            "source": self.source_label,
            "tables": [{"name": t.name, "fingerprint": t.fingerprint} for t in self.tables],
            "returned": self.result.returned,
            "total": self.result.total,
            "truncated": self.result.truncated,
            "row_keys_sha256": self.result.row_keys_sha256,
            "served_by": self.served_by,
            "evidence_nature": (self.evidence_nature or {}).get("kind") if self.evidence_nature else None,
            "leakage_risk": leakage_risk,
            "order_verified": self.result.order.verified if self.result.order else None,
            "evidence_caveat": (self.evidence_nature or {}).get("caveat") if self.evidence_nature else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False, indent=1)


def prov_id(record: dict[str, Any]) -> str:
    """``dp_`` + the first 12 hex digits of the sha256 of the record's canonical JSON, without
    ``id`` and ``tool_use_id`` (so the id does not depend on itself or on the caller's id)."""
    body = {k: v for k, v in record.items() if k not in ("id", "tool_use_id")}
    text = json.dumps(body, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))
    return "dp_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def cap_row_keys(keys: Sequence[Sequence[Any]], max_n: int, storage_types: Sequence[str | None] | None = None,
                 *, sample_n: int = ROW_KEY_SAMPLE) -> tuple[list[list[Any]], bool, str]:
    """``(kept keys, complete, sha256)``: every key when there are at most ``max_n``, else the first
    ``min(sample_n, max_n)`` as a sample. The hash always covers every key and uses the
    canonical encoding, so it compares across witness, upstream and replay."""
    rendered = [canonical(list(k), storage_types) for k in keys]
    sha = canonical_sha256(rendered)
    complete = len(rendered) <= max_n
    kept_n = len(rendered) if complete else min(sample_n, max_n)
    return [json.loads(r) for r in rendered[:kept_n]], complete, sha
