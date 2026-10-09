"""Source descriptors (``configs/data/sources/<source>.yaml``, schema ``vbt.datasource/1``; §6.1, §6.7).

Every model forbids unknown keys. Column roles are in ``descriptor/columns.py``; the small
models nested containers share with tables (``ItemKey``, ``CoverageSpec``, ``RankSpec``,
``UniverseSpec``, ``ParseSpec``, ``EnrichmentSpec`` ...) are defined there and re-exported here.
Cross-object rules (references, keys against columns, id_types) are lint rules
(``descriptor/lint.py``); the validators here check only what one model can see.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .columns import (
    AliasesFrom,
    CensorSpec,
    ColumnSpec,
    CompositeRef,
    CoverageSpec,
    CutoffSpec,
    EnrichmentFamily,
    EnrichmentSpec,
    IdTypeFrom,
    ItemKey,
    KindFrom,
    MemberEnd,
    MembershipSpec,
    MissingKind,
    ParseSpec,
    RankSpec,
    ScopeFacet,
    Strict,
    TimeFallback,
    UniverseSpec,
)

__all__ = [
    "SCHEMA", "Strict", "SourceDescriptor", "ReleaseSpec", "ManifestSpec", "TableDefaults", "LayoutRef", "FormatRef",
    "RemoteBudget", "IdTypeSpec", "RemoteUniverse", "RetiredSpec", "XrefSpec", "CrosswalkSpec", "MapsTo",
    "CanonicalizeSpec", "HierarchyRef", "UniverseSpec", "TableSpec", "ItemsOf", "KeySpec", "ItemKey", "GrainSpec",
    "CoverageSpec", "CensorSpec", "ConstraintSpec", "Sentinel", "SentinelSpec", "RankSpec", "PartitionSpec",
    "AccessPath", "ConditionsSpec", "AggregatedOver", "EdgeSpec", "PivotSpec", "RolesFrom", "FragmentKey",
    "SizeFrom", "EvidenceNature", "RemoteProbe", "ExposeSpec", "Lineage", "LeakageSpec", "MaterializedBy",
    "ViewSpec", "SectionSpec", "ParseSpec", "AxisSpec", "MatrixSpec", "EnrichmentSpec", "EnrichmentFamily",
    "MembershipSpec", "MemberEnd", "ScopeFacet", "CompositeRef", "IdTypeFrom", "KindFrom", "AliasesFrom",
    "CutoffSpec", "TimeFallback", "MissingKind", "ColumnSpec", "SENTINEL_WORDS", "SENTINEL_OPS", "plugin_name",
    "id_type_identity", "AcquisitionSpec", "AcquisitionTransport", "AcquisitionFiles", "PrepareStep",
]

SCHEMA = "vbt.datasource/1"


class LayoutRef(Strict):
    plugin: str
    options: dict[str, Any] = {}


class FormatRef(Strict):
    plugin: str
    options: dict[str, Any] = {}


def plugin_name(ref: str | LayoutRef | FormatRef | None) -> str | None:
    """The plugin name of a layout/format reference (a plain string is the name)."""
    if ref is None or isinstance(ref, str):
        return ref
    return ref.plugin


class TableDefaults(Strict):
    format: str | FormatRef | None = None
    layout: str | LayoutRef | None = None
    missing: MissingKind | None = "unknown"


class RemoteBudget(Strict):
    requests_per_min: int | None = None
    max_pages: int | None = None
    page_size: int | None = None
    max_requests_per_call: int | None = None
    timeout_s: float | None = None


class ReleaseSpec(Strict):
    expect: str | None = None                          # the release the files must have (R2 compares)
    from_: str | list[str] = Field(alias="from")       # literal | as_of | manifest.<jsonpath> | format.<key> | ...
    resolve: dict[str, str] | None = None              # {result: "$.<field>"} or {table, column}
    per: dict[str, str] | None = None                  # per-record release: {table, column}
    # text the release read through `from: format.<key>` must contain (the pinned release, e.g. "2026-06-08" in
    # data-version cl/releases/2026-06-08/cl-basic.owl): a file of another release is stale (R2:release, DEP-3)
    match: str | None = None

    @field_validator("expect", mode="before")
    @classmethod
    def _expect_text(cls, v: Any) -> Any:
        return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


class ManifestSpec(Strict):
    path: str | None = None                            # relative to root
    inline: dict[str, dict[str, Any]] | None = None    # {"<file>": {bytes, md5 | sha256}}
    entries: str = "$.files"
    required: bool = False
    require: dict[str, Any] = {}                       # {complete: true}
    checks: dict[str, str] = {}                        # {rows: "$.<jsonpath>"}: compared with the data;
                                                       # "<table>.rows" scopes a check to one table

    @model_validator(mode="after")
    def _has_source(self) -> "ManifestSpec":
        if self.path is None and self.inline is None:
            raise ValueError("a manifest needs `path` or `inline`")
        return self


class RemoteUniverse(Strict):
    tool: str                                          # "<server>.<tool>" listing the universe
    args: dict[str, Any] = {}
    path: str                                          # JSONPath of the listed keys
    ttl_s: float | None = None


class RetiredSpec(Strict):
    listed_in: list[str] = []                          # [table.column] holding retired IDs
    flag: str | None = None                            # column flagging obsolete terms
    label_prefix: str | None = None                    # "obsolete "
    replaced_by: str | None = None                     # column naming the replacement
    consider: str | None = None                        # column naming candidate replacements


class XrefSpec(Strict):
    column: str                                        # "table.column" holding CURIEs
    namespaces: dict[str, str] = {}                    # {<CURIE prefix>: <id_type>}
    id_type_from: IdTypeFrom | None = None


class CrosswalkSpec(Strict):
    name: str
    table: str
    from_: str = Field(alias="from")
    to: str
    cardinality: Literal["one", "many"] = "one"


class MapsTo(Strict):
    id_type: str                                       # "<source>:<id_type>"
    via: str | None = None                             # crosswalk name or source.table
    cardinality: Literal["one", "many"] = "one"


class CanonicalizeSpec(Strict):
    parent: str                                        # "table.column" of the parent key


class HierarchyRef(Strict):
    table: str
    columns: list[str] = []
    predicates: list[str] = []
    reflexive: bool = False


class IdTypeSpec(Strict):
    plugin: str                                        # identifier plugin = identity space
    options: dict[str, Any] = {}                       # {prefixes: from_universe}, {canonical: "<regex>"}
    universe: str | list[str] | UniverseSpec | None = None   # identity universe (decides not_found, I2)
    universe_via: RemoteUniverse | None = None
    label_of: str | None = None                        # a label kind: existence and candidates from that type
    union: list[str] = []                              # multi-kind id_type (embedding vocabulary words)
    resolve_via: list[str] = []                        # label/synonym columns
    rules: list[str] | None = None                     # ordered resolver rules (§11.5)
    retired: RetiredSpec | None = None
    xref_via: list[XrefSpec] = []
    crosswalks: list[CrosswalkSpec] = []
    maps_to: list[MapsTo] = []
    stored_forms: dict[str, str] = {}                  # {"table.column": "as_stored"}
    canonicalize: CanonicalizeSpec | None = None
    disambiguate_with: list[str] = []
    prefer: list[str] = []
    hierarchy: HierarchyRef | None = None
    extends: str | None = None                         # "<source>:<id_type>" to add hierarchy/xrefs to
    index: Literal["local", "remote"] = "local"
    resolvable: bool = True
    authority: Literal["current", "release_snapshot"] = "current"


class ItemsOf(Strict):
    """An item table over a nested container (rev 2)."""

    table: str                                         # parent table
    path: str                                          # "items[]", "a[].b[].c[]"

    @field_validator("path")
    @classmethod
    def _ends_in_list(cls, v: str) -> str:
        if not v.rstrip().endswith("]"):
            raise ValueError("items_of.path must end at a list container ('...[]')")
        return v


class KeySpec(Strict):
    columns: list[str]                                 # ordered; nested paths and partition columns allowed
    nullable: list[str] = []                           # parts that may be null (NULLS NOT DISTINCT)
    check: Literal["full", "sampled", "none"] = "sampled"
    sample_prefix: list[str] = []
    # key: the columns identify a row; content_hash: they group rows that no two are equal;
    # none: no identity, exact copies of a row occur in the release and are counted as stored
    row_identity: Literal["key", "content_hash", "none"] = "key"
    version: str | None = None                         # live records: the record-version column
    # live records: a list column naming the record's former keys (CT.gov nctIdAliases: a merged NCT ID the
    # registry answers with the surviving record); a requested key found there is the record, not a miss
    aliases: str | None = None
    verified: bool = True

    @model_validator(mode="after")
    def _nullable_subset(self) -> "KeySpec":
        extra = [c for c in self.nullable if c not in self.columns]
        if extra:
            raise ValueError(f"key.nullable names parts that are not key columns: {extra}")
        bad_prefix = [c for c in self.sample_prefix if c not in self.columns]
        if bad_prefix:
            raise ValueError(f"key.sample_prefix names parts that are not key columns: {bad_prefix}")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("key.columns repeats a column")
        return self


class GrainSpec(Strict):
    columns: list[str] = []
    canonicalize: str | None = None                    # "<id_type>.parent": count parent families
    unordered: list[str] = []                          # [side_a, side_b]: unordered pairs
    by: list[str] = []


class ConstraintSpec(Strict):
    column: str
    op: Literal["<", "<=", ">", ">=", "==", "!=", "in", "is_finite", "not_null", "equals_expr", "len_eq",
                "subset_of"]
    value: Any = None
    expr: str | None = None                            # relation: "len(children) == 0", "l2(vector)"
    tolerance: float | None = None
    origin: str
    verified: bool = True
    on_refute: Literal["not_ready", "drop_field", "recompute"] = "not_ready"

    @model_validator(mode="after")
    def _operands(self) -> "ConstraintSpec":
        if self.op in ("equals_expr", "len_eq", "subset_of") and not self.expr:
            raise ValueError(f"constraint op {self.op!r} needs `expr`")
        if self.op in ("<", "<=", ">", ">=", "==", "!=", "in") and self.value is None:
            raise ValueError(f"constraint op {self.op!r} needs `value`")
        return self


#: Sentinel expectation words (§6.3) and the comparison operators of ``{col: {op: value}}``.
SENTINEL_WORDS = frozenset({"nonempty", "contains", "items", "is_null", "is_empty", "min_rows"})
SENTINEL_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "in"})
_ITEMS_COND = re.compile(r"^\s*(>=|<=|==|>|<)\s*\d+\s*$")


class Sentinel(Strict):
    key: dict[str, Any]                                # {col: value}; {id: X} for a one-column key
    orientation: Literal["as_stored", "any"] = "as_stored"
    expect: dict[str, Any] = {}
    via: dict[str, Any] | None = None                  # remote: {tool, args}

    @field_validator("key")
    @classmethod
    def _non_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("a sentinel key names at least one column")
        return v

    @field_validator("expect")
    @classmethod
    def _grammar(cls, v: dict[str, Any]) -> dict[str, Any]:
        for k, val in v.items():
            if k in ("nonempty", "is_null", "is_empty"):
                if not (isinstance(val, list) and all(isinstance(x, str) for x in val)):
                    raise ValueError(f"expect.{k} is a list of column paths")
            elif k in ("contains", "items"):
                if not isinstance(val, dict):
                    raise ValueError(f"expect.{k} maps column paths to values")
                if k == "items":
                    for path, cond in val.items():
                        if not (isinstance(cond, int) and not isinstance(cond, bool)
                                or isinstance(cond, str) and _ITEMS_COND.match(cond)):
                            raise ValueError(f"expect.items.{path} is a count or '>=N'/'<=N'/'==N'/'>N'/'<N'")
            elif k == "min_rows":
                if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                    raise ValueError("expect.min_rows is a non-negative integer")
            elif isinstance(val, dict):
                if len(val) != 1 or next(iter(val)) not in SENTINEL_OPS:
                    raise ValueError(f"expect.{k} is a value or one {{op: value}} with op in {sorted(SENTINEL_OPS)}")
        return v


class SentinelSpec(Strict):
    present: list[Sentinel] = []
    absent: list[Sentinel] = []


class PartitionSpec(Strict):
    """A partition column (``key=value`` directories): a logical column of the table (§6.3)."""

    column: ColumnSpec
    type: Literal["string", "int64", "date"] = "string"
    expect: Literal["declared", "manifest", "any"] = "declared"
    mirrored_by: list[str] = []


class AccessPath(Strict):
    columns: list[str]
    via: Literal["partition", "row_group_stats", "sidecar_index"]
    build: Literal["readiness", "on_demand"] = "on_demand"


class ConditionsSpec(Strict):
    columns: list[str]
    from_: str = Field("self", alias="from")
    value_map: dict[str, dict[str, Any]] = {}


class AggregatedOver(Strict):
    dimension: str
    how: str | None = None
    origin: str | None = None


class EdgeRoles(Strict):
    """An edge's direction read from per-side role columns: the side whose role is ``source`` is the edge's
    source and the side whose role is ``target`` its target, on rows matching ``when``. Rows stored in both
    orientations with the roles swapped (25.09 SIGNOR: "regulator" / "regulator target") are one directed edge."""

    columns: dict[Literal["a", "b"], str]
    source: str
    target: str
    when: dict[str, list[Any]] = {}
    verified: bool = False

    @model_validator(mode="after")
    def _both_sides(self) -> "EdgeRoles":
        if set(self.columns) != {"a", "b"}:
            raise ValueError("edge.direction_from_roles.columns names the role column of side a and of side b")
        if self.source == self.target:
            raise ValueError("edge.direction_from_roles: source and target roles must differ")
        return self


class EdgeSpec(Strict):
    a: str
    b: str
    directed: bool = False
    directed_when: dict[str, list[Any]] = {}
    direction_from_roles: EdgeRoles | None = None
    orientation: Literal["canonical", "as_reported", "both"] = "as_reported"
    verified: bool = False
    sides: dict[Literal["a", "b"], list[str]] = {}


class PivotSpec(Strict):
    index: list[str]
    name_column: str
    value_column: str
    level_from: str | None = None


class RolesFrom(Strict):
    table: str
    name: str
    level_from: str | None = None
    type_from: str | None = None
    defaults: dict[str, Any] = {}


class FragmentKey(Strict):
    name: str                                          # the virtual column holding the fragment key
    from_: Literal["filename_regex", "path_regex", "directory"] = Field(alias="from")
    pattern: str | None = None


class SizeFrom(Strict):
    """Count-first admission of a remote read (§14.1): the rows a call reads number ``column`` of the
    ``table`` record it is scoped to. ``via`` (``server.tool``, called with ``{arg: <scope value>}``) reads
    that count at ``path`` (a number, or a mapping of counts whose largest is taken) before the call."""

    table: str
    column: str
    row_bytes: int | None = None
    via: str | None = None
    arg: str | None = None
    path: str | None = None


class EvidenceNature(Strict):
    kind: str
    caveat: str


class RemoteProbe(Strict):
    tool: str
    arg: str | None = None
    from_sentinel: bool = False


class ExposeSpec(Strict):
    native: bool = True
    withhold_from: list[str] = []
    reason: str | None = None


class Lineage(Strict):
    source: str
    release: str | None = None
    table: str | None = None


#: Where a source's evidence ceiling comes from (``LeakageSpec.ceiling_from``): the run's ``data.leakage.ceiling``,
#: or the literature ceiling ``web.literature_max_date`` (exported as ``VBT_LITERATURE_MAXDATE``), which the PubMed
#: server applies to every search itself.
DATA_CEILING = "data.leakage.ceiling"
LITERATURE_CEILING = "web.literature_max_date"
CEILING_SOURCES = (DATA_CEILING, LITERATURE_CEILING)


class LeakageSpec(Strict):
    """The source can return facts dated after the evidence ceiling (rev 2)."""

    ceiling_from: str = DATA_CEILING
    available_at: str
    changed_at: str | None = None
    partial_dates: Literal["latest", "earliest"] = "latest"
    rows: Literal["withhold", "redact", "stamp"] = "withhold"
    redact: list[str] = []
    counts: Literal["inject_filter", "block", "stamp"] = "block"
    # engine terms that select on the record as it is today and name no declared column (CT.gov
    # AREA[ResultsFirstPostDate]); the redact columns and the time columns other than available_at count as such
    # by themselves. A count that selects on any of them under the ceiling is not bounded by it.
    current_terms: list[str] = []


class MaterializedBy(Strict):
    """Tables written by a tool call during a run (files a tool materialises)."""

    tool: str
    path_from: str
    value_from_arg: dict[str, dict[str, str]] = {}
    release_from: Literal["producer", "file"] = "producer"


class SectionSpec(Strict):
    """One section of a multi-table result or view: ``{path, table, verb, single, key_from_args}``."""

    path: str | None = None
    table: str
    verb: Literal["lookup", "find", "search", "members", "count", "aggregate", "similar", "expand"] = "lookup"
    single: bool = False                               # an entity_detail section is a list, never the first of many
    key_from_args: dict[str, str] = {}
    value: str | None = None                           # single: the section holds this field of the record


class ViewSpec(Strict):
    sections: dict[str, SectionSpec]


class AxisSpec(Strict):
    name: str                                          # long-view name of the axis
    from_: Literal["column", "header", "index", "table", "positional", "file"] = Field(alias="from")
    column: str | dict[str, int] | None = None
    aliases: list[str] = []
    exclude: list[str] = []
    parse: ParseSpec | None = None
    ids_from: str | None = None
    index_name: str | None = None
    key: KeySpec
    columns: dict[str, ColumnSpec] = {}
    attributes_from: dict[str, Any] | None = None


class MatrixSpec(Strict):
    axes: dict[Literal["row", "col"], AxisSpec]       # "obs"/"var" accepted as aliases of row/col
    values: dict[str, ColumnSpec]
    storage: Literal["dense", "csr", "csc", "text"] = "dense"
    implicit: Literal["none", "zero", "zero_if_measured"] = "none"
    measured_by: dict[str, Any] | None = None
    sections: dict[str, ColumnSpec] = {}

    @field_validator("axes", mode="before")
    @classmethod
    def _axis_aliases(cls, v: Any) -> Any:
        if isinstance(v, dict):
            alias = {"obs": "row", "var": "col"}
            out: dict[str, Any] = {}
            for k, axis in v.items():
                name = alias.get(str(k), str(k))
                if name in out:
                    raise ValueError(f"axis {name!r} given twice (obs/var are aliases of row/col)")
                out[name] = axis
            return out
        return v

    @model_validator(mode="after")
    def _complete(self) -> "MatrixSpec":
        if set(self.axes) != {"row", "col"}:
            raise ValueError("a matrix declares both axes (row/obs and col/var)")
        if not self.values:
            raise ValueError("a matrix declares at least one value")
        if self.implicit == "zero_if_measured" and not self.measured_by:
            raise ValueError("implicit: zero_if_measured needs measured_by")
        return self


class TableSpec(Strict):
    kind: Literal["entity", "fact", "crosswalk", "ontology", "edges", "sets", "matrix", "vectors", "records",
                  "entity_detail"]
    path: str | None = None
    items_of: ItemsOf | None = None
    implements: str | None = None                      # "<source>:<table>@<release>": the same logical table
    lineage: Lineage | None = None
    layout: str | LayoutRef | None = None
    format: str | FormatRef | None = None              # "none" for remote tables served only upstream
    grain: str
    key: KeySpec
    alternate_keys: list[KeySpec] = []
    grains: dict[str, list[str] | GrainSpec] = {}
    rank: list[RankSpec] = []
    constraints: list[ConstraintSpec] = []
    coverage: CoverageSpec | None = None
    sentinels: SentinelSpec | None = None
    partitions: dict[str, PartitionSpec] = {}
    access_paths: list[AccessPath] = []
    conditions: ConditionsSpec | None = None
    aggregated_over: list[AggregatedOver] = []
    edge: EdgeSpec | None = None
    pivot: PivotSpec | None = None
    column_patterns: dict[str, ColumnSpec] = {}
    roles_from: RolesFrom | None = None
    fragment_key: FragmentKey | None = None
    fragment_overrides: dict[str, dict[str, dict[str, Any]]] = {}
    size_class: Literal["auto", "small", "large", "huge"] = "auto"
    size_from: SizeFrom | None = None
    max_scan_bytes: int | None = None
    evidence_nature: EvidenceNature | None = None
    materialized_by: MaterializedBy | None = None
    remote_probe: RemoteProbe | None = None
    expose: ExposeSpec = ExposeSpec()
    columns: dict[str, ColumnSpec] = {}
    matrix: MatrixSpec | None = None
    strict: bool | None = None

    def pattern_column(self, name: str) -> ColumnSpec | None:
        """The ``column_patterns`` spec a physical column matches (``ae_serious_{organ}_pct`` matches
        ``ae_serious_cardiac_pct``), or None. Pattern columns count as declared under ``strict``."""
        import re

        for template, spec in self.column_patterns.items():
            parts = re.split(r"\{[A-Za-z_][A-Za-z0-9_]*\}", template)
            rx = "^" + ".+?".join(re.escape(p) for p in parts) + "$"
            if re.match(rx, name):
                return spec
        return None

    @model_validator(mode="after")
    def _shape(self) -> "TableSpec":
        if self.items_of is not None:
            if self.key.columns:
                raise ValueError("an item table's key is composed from its parent: key.columns must be []")
            if self.path or self.layout or self.format:
                raise ValueError("an item table shares its parent's fragments: no path, layout or format")
            if self.columns:
                raise ValueError("an item table's columns are the container's fields (declared on the parent)")
        if self.kind == "matrix" and self.matrix is None:
            raise ValueError("kind: matrix needs a `matrix` block (§6.7)")
        if self.matrix is not None and self.kind != "matrix":
            raise ValueError("a `matrix` block requires kind: matrix")
        if not self.columns and self.items_of is None and self.kind != "matrix" and self.roles_from is None:
            raise ValueError("columns are required unless kind is matrix, items_of is set or roles_from is given")
        return self

    @property
    def is_item_table(self) -> bool:
        return self.items_of is not None


class AcquisitionTransport(Strict):
    """The acquisition plugin that lists and reads the source's files, and its options (``{release}`` is
    substituted in every string)."""

    plugin: str                                        # http | huggingface | s3 | gcs | json_index | zip_member | ...
    options: dict[str, Any] = {}


class AcquisitionFiles(Strict):
    """What one table (or one named download group, ``extra``) needs from the source: the listed files matching
    ``files`` (path globs: ``**`` spans directories), or a ``prepare`` step's output."""

    files: list[str] = []
    exclude: list[str] = []
    bytes: int | None = None                           # known size of the matched files (plan without a listing)
    count: int | None = None                           # known number of files
    prepared_by: str | None = None                     # the prepare step that writes this table from downloads
    optional: bool = False                             # acquired only when named (extra groups)
    note: str | None = None

    @model_validator(mode="after")
    def _has_files(self) -> "AcquisitionFiles":
        if not self.files and not self.prepared_by:
            raise ValueError("an acquisition entry names `files` or the `prepared_by` step that writes it")
        return self


class PrepareStep(Strict):
    """An unmodified upstream script that turns downloaded files into what the data layer reads. ``command`` is an
    argv; ``{python}`` (this interpreter), ``{upstream}`` (``vars.upstream``), ``{home}``, ``{downloads}``,
    ``{output}`` (the home's ``output`` directory) and ``{release}`` are substituted."""

    command: list[str]
    needs: list[str]                                   # download groups (tables or extra) the step reads
    output: str                                        # the directory the step writes, relative to the home
    fresh: bool = True                                 # the step creates `output` whole and refuses an existing one
    manifest: str | None = None                        # the manifest the step writes into `output`
    complete_key: str = "complete"                     # that manifest's completeness flag
    output_bytes: int | None = None                    # disk the output needs (an estimate), for the plan
    note: str | None = None

    @field_validator("command")
    @classmethod
    def _argv(cls, v: list[str]) -> list[str]:
        if not v or not all(isinstance(x, str) and x for x in v):
            raise ValueError("prepare.command is a non-empty argv list of strings")
        return v


class AcquisitionSpec(Strict):
    """How the source's files are acquired (``vbt data acquire``): a transport plugin, the files of each table,
    optional prepare steps, where they land, and what a person must know (licence, login).

    ``mode``: ``download`` (the default), ``remote`` (read live; nothing to download) or ``manual`` (a person
    must fetch the files: ``login`` says how). The home of a source is ``<acquisition root>/<dir>``; downloads
    land in ``<home>/<downloads>`` and keep their listed paths; ``env`` names the variables that point the data
    layer at the files (``{home}``, ``{downloads}`` and ``{release}`` substituted)."""

    mode: Literal["download", "remote", "manual"] = "download"
    release: str | None = None                         # the release acquired and pinned (default: release.expect)
    transport: AcquisitionTransport | None = None
    dir: str = "{source}/{release}"
    downloads: str = "."
    env: dict[str, str] = {}
    index_cache: str | None = None                     # where the transport keeps its index (relative to downloads)
    tables: dict[str, AcquisitionFiles] = {}
    extra: dict[str, AcquisitionFiles] = {}            # named download groups that are not tables
    prepare: dict[str, PrepareStep] = {}
    verify: list[Literal["size", "checksum", "parquet_framing"]] = ["size", "checksum"]
    manifest: str | None = ".download-manifest.json"  # written in the downloads directory (upstream format)
    manifest_base: str | None = None                   # the manifest's `base` (default: the transport's `base`)
    licence: str | None = None
    login: str | None = None
    notes: str | None = None
    homepage: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "AcquisitionSpec":
        if self.mode == "download" and self.transport is None:
            raise ValueError("acquisition mode download needs a `transport`")
        clash = sorted(set(self.tables) & set(self.extra))
        if clash:
            raise ValueError(f"acquisition.extra names tables: {clash}")
        groups = set(self.tables) | set(self.extra)
        for name, step in self.prepare.items():
            unknown = [n for n in step.needs if n not in groups]
            if unknown:
                raise ValueError(f"acquisition.prepare.{name}.needs names unknown download groups: {unknown}")
        for name, entry in {**self.tables, **self.extra}.items():
            if entry.prepared_by is not None and entry.prepared_by not in self.prepare:
                raise ValueError(f"acquisition entry {name}: prepared_by {entry.prepared_by!r} is not a prepare step")
        return self


class SourceDescriptor(Strict):
    schema_: Literal["vbt.datasource/1"] = Field(alias="schema")
    source: str
    title: str
    kind: Literal["local", "remote"] = "local"
    root: str | None = None
    release: ReleaseSpec
    manifests: list[ManifestSpec] = []
    defaults: TableDefaults = TableDefaults()
    strict: bool = False
    budget: RemoteBudget | None = None
    id_types: dict[str, IdTypeSpec] = {}
    tables: dict[str, TableSpec]
    views: dict[str, ViewSpec] = {}
    leakage: LeakageSpec | None = None
    concepts: dict[str, list[str]] = {}
    acquisition: AcquisitionSpec | None = None

    @model_validator(mode="after")
    def _acquisition_tables(self) -> "SourceDescriptor":
        if self.acquisition is None:
            return self
        # a descriptor narrowed to fewer tables keeps the others' files as optional download groups (acquired only
        # when named), so a prepare step that reads them still resolves
        for name in [t for t in self.acquisition.tables if t not in self.tables]:
            entry = self.acquisition.tables.pop(name)
            self.acquisition.extra[name] = entry.model_copy(update={"optional": True})
        items = [t for t in self.acquisition.tables if self.tables[t].items_of is not None]
        if items:
            raise ValueError(f"acquisition.tables names item tables (they share their parent's files): {items}")
        return self

    @field_validator("source")
    @classmethod
    def _source_name(cls, v: str) -> str:
        if not v or not v.replace("_", "a").isalnum() or ":" in v or "." in v:
            raise ValueError("source names are identifiers (letters, digits, underscores)")
        return v

    @field_validator("id_types", "tables")
    @classmethod
    def _plain_names(cls, v: dict[str, Any]) -> dict[str, Any]:
        bad = [k for k in v if ":" in k or "." in k]
        if bad:
            raise ValueError(f"names are declared bare (qualification is by source): {bad}")
        return v

    def table_layout(self, table: str) -> str | None:
        t = self.tables[table]
        return plugin_name(t.layout) or plugin_name(self.defaults.layout)

    def table_format(self, table: str) -> str | None:
        t = self.tables[table]
        return plugin_name(t.format) or plugin_name(self.defaults.format)

    def table_strict(self, table: str) -> bool:
        t = self.tables[table]
        return self.strict if t.strict is None else bool(t.strict)


def id_type_identity(desc: SourceDescriptor, spec: IdTypeSpec) -> tuple[Any, ...]:
    """What decides existence for an id_type, with every name qualified: plugin, identity universe
    (or remote listing), ``label_of`` and ``union``. Two sources declaring one bare name with the
    same identity are interchangeable; with different identities a bare ``accepts`` is ambiguous."""
    src = desc.source

    def q(name: str) -> str:
        return name if ":" in name else f"{src}:{name}"

    def table(ref: str) -> str:
        return ref if ref.split(".")[0] not in desc.tables else f"{src}.{ref}"

    def uni(u: Any) -> Any:
        if u is None:
            return None
        if isinstance(u, list):
            return tuple(uni(x) for x in u)
        if isinstance(u, UniverseSpec):
            where = None if u.where is None else repr(sorted(u.where.items()))
            return (table(u.table), tuple(u.keys), u.mode, where, tuple(u.per_scope), u.per_fragment)
        return table(str(u))

    via = None if spec.universe_via is None else (spec.universe_via.tool, spec.universe_via.path)
    return (spec.plugin, uni(spec.universe), via, q(spec.label_of) if spec.label_of else None,
            tuple(sorted(q(m) for m in spec.union)))
