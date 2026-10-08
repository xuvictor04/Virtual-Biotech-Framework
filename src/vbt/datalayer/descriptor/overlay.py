"""Overlays: bindings of MCP tools we cannot edit (``configs/data/overlays/<server>.yaml``,
schema ``vbt.overlay/1``; §8.1). No pyarrow.

An overlay says, per tool, what the tool reads, what each argument means in descriptor
terms, which result fields are which descriptor columns, and how the gateway serves it
(``pass``, ``derived`` or ``block``). Overlays never contain code. Files whose name starts
with ``_`` are generic overlays (``_generic.yaml``): they apply to servers and tools without
a reviewed binding (§11.9).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .columns import RankSpec, Strict, _yaml_word
from .models import SectionSpec

__all__ = [
    "SCHEMA", "Overlay", "GenericSpec", "ToolBinding", "ReadSpec", "ArgBinding", "ResultSpec", "FieldMap",
    "EchoSpec", "EchoSet", "TotalSpec", "TrimSpec", "RecomputeSpec", "StatisticsSpec", "FileCheckSpec",
    "LeakageFilter", "DerivedSpec", "BlockSpec", "DefectSpec", "TextSpec", "SectionSpec", "CountFirst", "SampleSpec",
    "EstimateSpec", "ARG_ROLES", "ARG_OPS", "FILTER_ROLES",
]

SCHEMA = "vbt.overlay/1"

ARG_ROLES = ("filter", "limit", "free_text", "output_path", "projection", "flag", "selector", "order_by",
             "order_direction", "anchor", "family_param", "universe_override", "unbound")
ARG_OPS = ("eq", "in", "contains", "overlaps", "ge", "gt", "le", "lt", "ne", "range", "ge_abs", "gt_abs", "le_abs",
           "lt_abs", "nonempty", "has_kind")
#: Argument roles that bind a column and are re-applied to rows (T4).
FILTER_ROLES = ("filter", "flag", "anchor")


class ReadSpec(Strict):
    access: Literal["full_table", "bounded_scan", "projection", "remote", "upstream"] = "full_table"
    columns: list[str] = []                            # paths; default: containers bound or returned
    when: dict[str, Any] | None = None                 # argument-dependent reads ({<arg>: <value>})
    est_row_bytes: int | None = None                   # remote reads
    count_via: str | None = None                       # remote: a count tool run first


class ArgBinding(Strict):
    binds: str | dict[str, str] | None = None          # column path; or {selector value: column}
    binds_any: list[str] = []                          # match on any of these columns -> Or
    role: Literal["filter", "limit", "free_text", "output_path", "projection", "flag", "selector", "order_by",
                  "order_direction", "anchor", "family_param", "universe_override", "unbound"] = "filter"
    accepts: list[str] = []                            # id_types (bare or "source:id_type"), in order
    op: Literal["eq", "in", "contains", "overlaps", "ge", "gt", "le", "lt", "ne", "range", "ge_abs", "gt_abs",
                "le_abs", "lt_abs", "nonempty", "has_kind"] = "eq"
    values: dict[Any, Any] = {}                        # selector/order_by/order_direction value maps
    match: Literal["exact", "casefold"] = "exact"
    when_true: dict[str, Any] | None = None            # flag args: a positive predicate ({in: [none_recorded]})
    when_false: dict[str, Any] | None = None
    interpreted_as: Literal["exact", "regex", "substring", "casefold_substring", "engine"] = "exact"
    # a substring argument that searches: every stored value it matches is pooled into the answer (search_drugs
    # 'statin'), never refused as a collision that asks for one exact value
    pooled: bool = False
    engine_doc: str | None = None
    engine_param: str | None = None                    # engine args: the source's request parameter the value
                                                       # fills (query.cond): the remote witness counts that search
    escape: str | None = None                          # format plugin whose quote() escapes the value
    forbid: list[str] = []
    pattern: str | None = None
    send_as: Literal["canonical", "label", "raw", "stored", "native_label", "alias"] = "canonical"
    send_map: dict[str, Any] = {}                      # {<value>: <form upstream expects>}
    snap: bool = True                                  # numeric scope/category values snapped to the vocabulary
    min: float | None = None
    max: float | None = None
    each: bool = False                                 # list argument: every element resolved
    max_items: int | None = None
    min_items: int | None = None
    on_missing: Literal["error", "partial", "drop_disclosed"] = "error"
    min_resolved_fraction: float | None = None
    dedupe: bool = True
    default_disclosed: bool = True
    gateway_only: bool = False                         # x-gateway arg: stripped before the upstream call
    wrap: str | None = None                            # "({value})": parenthesise an engine fragment
    existence: Literal["universe", "bound", "upstream", "off"] = "universe"
    universe: str | None = None                        # override the id_type's identity universe
    universe_where: dict[str, Any] | None = None       # subset universe (a predicate on the universe table)
    qualified_by: list[str] = []                       # arguments that qualify the key
    item_filter: bool = False                          # filter nested items, not rows
    drop_empty_parents: bool = False
    family: Literal["exact", "include"] = "exact"      # salt/parent family expansion (§11.5)
    limit_grain: str | None = None                     # the limit counts this grain, not rows
    limit_mode: Literal["per_group", "refuse"] = "per_group"

    @field_validator("existence", mode="before")
    @classmethod
    def _existence_word(cls, v: Any) -> Any:
        return _yaml_word(v, false="off")              # YAML 1.1 reads `off` as false

    @field_validator("wrap")
    @classmethod
    def _wrap_has_value(cls, v: str | None) -> str | None:
        if v is not None and "{value}" not in v:
            raise ValueError("wrap must contain '{value}'")
        return v

    @model_validator(mode="after")
    def _list_bounds(self) -> "ArgBinding":
        if self.min_items is not None and self.max_items is not None and self.min_items > self.max_items:
            raise ValueError("min_items > max_items")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min > max")
        return self

    @model_validator(mode="after")
    def _engine_param(self) -> "ArgBinding":
        if self.engine_param is not None and (self.role != "free_text" or self.interpreted_as != "engine"):
            raise ValueError("engine_param is for free_text arguments interpreted_as: engine")
        return self

    @model_validator(mode="after")
    def _text_match(self) -> "ArgBinding":
        # a scalar argument upstream matches as text (substring, regex) is free text whatever the role
        # says: it is checked for substring collisions and re-applied as a text match, never as an
        # exact filter or a vocabulary value (a list argument keeps its role; its items are checked)
        if self.role == "filter" and self.interpreted_as in ("substring", "casefold_substring", "regex") and \
                self.op == "eq" and not self.each:
            self.role = "free_text"
        return self

    @property
    def bound_columns(self) -> list[str]:
        """Every column path this argument binds (selector maps and ``binds_any`` included)."""
        out: list[str] = []
        if isinstance(self.binds, str):
            out.append(self.binds)
        elif isinstance(self.binds, dict):
            out.extend(self.binds.values())
        out.extend(self.binds_any)
        return out


class FieldMap(Strict):
    """A result field mapped to a descriptor column, or computed."""

    column: str | None = None                          # "col", "items[].field", "/id"
    computed: dict[str, Any] | None = None             # {statistic, from, anchor}
    placeholders: list[Any] = []
    on_placeholder: Literal["null", "dangling_ref"] = "null"

    @model_validator(mode="after")
    def _one_source(self) -> "FieldMap":
        if (self.column is None) == (self.computed is None):
            raise ValueError("a field maps exactly one of `column` or `computed`")
        return self


class EchoSpec(Strict):
    path: str
    accept: list[str] = ["canonical"]                  # canonical, synonym:<col>, redirect
    source: Literal["record", "request"] = "record"    # request is a lint error (an echo must read the record)


class EchoSet(Strict):
    arg: str
    path: str
    mode: Literal["subset", "equal"] = "subset"
    withheld_path: str | None = None


class TotalSpec(Strict):
    path: str
    method: Literal["upstream", "upstream_upper_bound"] = "upstream"
    valid_unless: list[str] = []
    partial_when: list[str] = []


class TrimSpec(Strict):
    max: int
    order: Literal["depth", "key", "declared"] = "declared"


class RecomputeSpec(Strict):
    # count_true: rows whose ``of`` is true (num_high_quality: probes with isHighQuality)
    agg: Literal["count", "count_distinct", "count_true", "sum", "mean", "median", "min", "max"]
    of: str | None = None
    group_by: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, data: Any) -> Any:
        # The YAML form is {recompute: {agg, of, group_by}}.
        if isinstance(data, dict) and set(data) == {"recompute"}:
            return data["recompute"]
        return data


class StatisticsSpec(Strict):
    test: str | None = None
    correction: str | None = None
    family: str | dict[str, Any] | None = None
    universe: str | dict[str, Any] | None = None


class FileCheckSpec(Strict):
    path_from: str
    must_exist: bool = True
    echo_checks: dict[str, str] = {}                   # {n_obs: $.n_cells}
    key_columns: list[str] = []                        # obs columns the file must hold
    var_key_columns: list[str] = []                    # var columns the file must hold (Census: feature_id)
    forbid_positional_index: bool = False
    write_once: bool = True


class LeakageFilter(Strict):
    arg: str
    template: str                                      # a query fragment with "{ceiling}"


class SampleSpec(Strict):
    """``count_first.sample``: the pull is served as the derived donor-balanced sample. The data child draws
    upstream's own sample (``stratify``: cells allocated over that column first; ``seed``: upstream's generator)
    with donors keyed by the descriptor grain ``grain`` (or ``key``: a qualifier and a donor column), and the
    server is asked for exactly the drawn cells. ``columns_arg`` names the argument listing the columns to write
    (completed with the cell key, the donor key and the stratum); ``total_path`` is the payload's total, restored
    to the counted one; ``max_read`` caps the cells whose key columns a sample reads; ``max_cells_default`` is
    the sample size when neither the call nor the tool's schema gives one."""

    grain: str = "donor"
    key: list[str] | None = None
    stratify: str | None = None
    seed: int | None = None
    columns_arg: str | None = None
    total_path: str | None = None
    max_read: int | None = Field(default=None, ge=1)
    max_cells_default: int | None = Field(default=None, ge=1)


class EstimateSpec(Strict):
    """``count_first.estimate``: the server memory a pull needs, calibrated on real pulls. ``base_mb`` is the
    footprint of any pull (the client opened, its read buffers); each pulled cell adds ``cell_bytes`` (its obs row
    and its share of the matrix read) plus ``value_bytes`` per named gene; a pull that names no gene reads every
    gene of the cell (``all_genes_cell_bytes`` per cell instead); ``read_all_bytes`` per matching cell is a
    metadata read of every cell the filter selects before the pull (a sample drawn by the server itself)."""

    base_mb: float = Field(default=0.0, ge=0)
    cell_bytes: int = Field(default=200, ge=0)
    value_bytes: float = Field(default=4.0, ge=0)
    all_genes_cell_bytes: int | None = Field(default=None, ge=0)
    read_all_bytes: int | None = Field(default=None, ge=0)


class CountFirst(Strict):
    """Count-first admission of a pull from a remote table (phase 4, F20): before the upstream call the
    data child counts the rows ``filter_arg`` selects (``_census_count``) and the call is refused
    ``too_large`` when the pull's estimate (``estimate``, from the counted cells and the genes the ``genes_arg``
    arguments name; the default is rows x (row bytes + genes x value bytes)) exceeds the server's limit.
    ``recompute_genes``: the written file's ``genes_found``/``genes_not_found`` are recomputed from its
    ``var.feature_name`` (upstream compares against positional var_names). ``sample``: served as the derived
    donor-balanced sample (:class:`SampleSpec`)."""

    table: str
    filter_arg: str | None = None
    genes_arg: str | list[str] | None = None           # every argument naming genes (symbols and Ensembl IDs)
    max_cells_arg: str | None = None
    recompute_genes: bool = False
    sample: SampleSpec | None = None
    estimate: EstimateSpec | None = None

    @property
    def genes_args(self) -> list[str]:
        """The gene arguments, as a list."""
        if self.genes_arg is None:
            return []
        return [self.genes_arg] if isinstance(self.genes_arg, str) else list(self.genes_arg)


class RequiresFixed(Strict):
    """The tool is accepted only when ``arg`` (a SOMA filter) fixes each of ``columns`` to one value; else
    ``unsupported_combination`` naming ``alternatives`` (a column unique only within another)."""

    arg: str
    columns: list[str]
    reason: str
    alternatives: list[str] = []


class ResultSpec(Strict):
    kind: Literal["rows", "record", "count", "file"] = "rows"
    rows: str | list[str] | None = "$"
    rows_of: str | None = None                         # item table the rows are items of
    record_when: str | None = None                     # payload test: a by-key reply is one record at "$"
    grain: str | None = None
    fields: dict[str, FieldMap] = {}
    parent_key: dict[str, str] = {}
    row_key: list[str] | Literal["from_descriptor"] = "from_descriptor"
    key_from_args: dict[str, str] = {}
    exists_when: str | None = None
    on_unknown_items: Literal["not_found", "partial"] = "not_found"
    echo: dict[str, str | EchoSpec] = {}
    echo_set: EchoSet | None = None
    order: list[RankSpec] = []
    order_from_arg: str | None = None
    # the order when an argument is given and no order_by argument decides (upstream ranks most constrained first
    # when min_genetic_constraint is set): {argument: [rank spec, ...]}, the first given argument wins
    order_when: dict[str, list[RankSpec]] = {}
    order_source: Literal["witness", "upstream_full_sort", "source_server_side"] = "witness"
    total: str | TotalSpec | None = None
    as_of: str | None = None
    # the payload field naming the release by the alias the server opened (Census census_version 'stable'): the
    # gateway writes the dated release the call resolved there (the alias is disclosed in a note)
    release_alias: str | None = None
    count_fields: list[str] | dict[str, str] = []
    # the reply lists at most this many of the rows it returns or writes (association tools write ``limit`` rows
    # to output_path and list head(10)): a preview is not a short page (W6), so it is never re-called. Per rows
    # path ({$.top_regions: 10}) when only one of the reply's shapes is a preview
    preview: int | dict[str, int] | None = None
    summary_fields: dict[str, Literal["drop"] | RecomputeSpec] = {}
    drop_fields: dict[str, str] = {}
    arg_echo: dict[str, str] = {}
    trim: dict[str, TrimSpec | int] = {}
    sections: dict[str, SectionSpec] = {}
    levels: dict[str, list[str]] = {}
    nested_errors: list[str] = []
    not_found_when: list[str] = []
    statistics: StatisticsSpec | None = None
    files: list[FileCheckSpec] = []
    codec: str | None = None                           # envelope plugin decoding the reply (default jsonpath)
    codec_options: dict[str, Any] = {}

    @property
    def row_paths(self) -> list[str]:
        if self.rows is None:
            return []
        return [self.rows] if isinstance(self.rows, str) else list(self.rows)

    def echo_specs(self) -> dict[str, EchoSpec]:
        """Echo entries with the string shorthand expanded (``{arg: "$.path"}`` reads the record)."""
        return {k: (EchoSpec(path=v) if isinstance(v, str) else v) for k, v in self.echo.items()}


class DerivedSpec(Strict):
    verb: Literal["lookup", "find", "search", "members", "count", "aggregate", "similar", "expand", "enrich",
                  "compare"]
    table: str                                         # table or item table ("source.table")
    columns: list[str] = []
    explode: list[str] = []
    carry: list[str] = []
    rename: dict[str, str] = {}
    split: dict[str, Any] | None = None                # {by_sign: <column>, into: {"+": <path>, "-": <path>}}
    nest: dict[str, Any] | None = None
    group_by: list[str] = []
    aggregate: dict[str, Any] = {}
    sections: dict[str, SectionSpec] = {}
    envelope: dict[str, Any] = {}
    compose: list["DerivedSpec"] = []


class BlockSpec(Strict):
    reason: str
    alternatives: list[str] = []
    until_phase: int | None = None
    hidden: bool = False


class DefectSpec(Strict):
    """Documentation plus a detector test per entry; never drives code (§8.3)."""

    id: str
    what: str
    where: str | None = None                           # "<file>:<line>[,<line>|-<line>]"
    test: str | None = None
    effect: str | None = None


class TextSpec(Strict):
    summary: str | None = None
    drop_promises: list[str] = []
    notes: list[str] = []


class ToolBinding(Strict):
    status: Literal["reviewed", "unreviewed"] = "reviewed"
    same_as: list[str] = []                            # other "server.tool" names served by this binding
    reads: dict[str, ReadSpec] = {}                    # "source.table" -> ReadSpec
    args: dict[str, ArgBinding] = {}
    require_any: list[list[str]] = []
    exclusive: list[list[str]] = []
    result: ResultSpec = ResultSpec()
    serve: Literal["pass", "derived", "block"] = "pass"
    derived: DerivedSpec | None = None
    # gateway-only arguments that, when set, serve a `pass` binding through `derived` (include_descendants)
    derived_when: list[str] = []
    count_first: CountFirst | None = None
    block: BlockSpec | None = None
    on_contradiction: Literal["derived", "tool_defect"] = "derived"
    witness: bool = True
    leakage_filter: LeakageFilter | None = None
    requires_fixed: list[RequiresFixed] = []
    defects: list[DefectSpec] = []
    text: TextSpec = TextSpec()
    hidden: bool = False


class GenericSpec(Strict):
    """Settings of a generic overlay (§11.9): which result shapes mean not found or structurally
    empty, and which parameter names suggest identifier kinds (observe-mode lint only)."""

    not_found_when: list[str] = []
    empty_when: list[str] = []                         # JSONPath predicates of a structural empty
    param_kinds: dict[str, list[str]] = {}             # {"<param glob>": [<id_type>, ...]}
    notes: list[str] = []
    codec: str | None = None                           # envelope plugin decoding replies (default jsonpath)
    codec_options: dict[str, Any] = {}


class Overlay(Strict):
    schema_: Literal["vbt.overlay/1"] = Field(alias="schema")
    server: str                                        # name in configs/mcp_servers.yaml; "*" for a generic overlay
    match: dict[str, str] | None = None                # {url_prefix: ...} for HTTP servers
    sources: list[str] = []
    upstream_commit: str | None = None
    tools: dict[str, ToolBinding] = {}
    generic: GenericSpec | None = None

    def binding(self, tool: str) -> ToolBinding | None:
        return self.tools.get(tool)


DerivedSpec.model_rebuild()
