"""Column roles as pydantic models (§7). No pyarrow.

:data:`ColumnSpec` is a union discriminated by ``role`` with one model per role carrying
exactly the role's facets (:data:`vbt.datalayer.roles.ROLE_FACETS`) plus the common facets
(``_Common``), so validation rejects any facet not listed for the role. ``NestedCol`` and
``MemberCol`` hold ``fields`` recursively.

The small models nested containers need (``ItemKey``, ``CoverageSpec``, ``RankSpec``,
``UniverseSpec``, ``ParseSpec``, membership and enrichment specs) live here as well and are
re-exported by ``descriptor/models.py``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

__all__ = [
    "Strict", "MissingKind", "ScopeFacet", "CompositeRef", "IdTypeFrom", "KindFrom", "AliasesFrom", "CutoffSpec",
    "TimeFallback", "MemberEnd", "EnrichmentFamily", "EnrichmentSpec", "MembershipSpec", "UniverseSpec",
    "CensorSpec", "CoverageSpec", "RankSpec", "ItemKey", "ParseSpec",
    "IdentifierCol", "LabelCol", "SynonymCol", "CategoryCol", "ScopeCol", "QualifierCol", "MeasureCol", "CountCol",
    "FlagCol", "TimeCol", "PositionCol", "HierarchyCol", "MemberCol", "EndpointCol", "VectorCol", "TextCol",
    "ReferenceCol", "NestedCol", "PayloadCol", "IgnoreCol", "ColumnSpec", "ROLE_MODELS", "validate_column",
    "facet_names", "is_container", "container_fields", "item_key_of", "DEFAULT_POOLING",
    "DEFAULT_STATISTIC",
]


class Strict(BaseModel):
    """Base of every descriptor and overlay model: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


MissingKind = Literal["unknown", "zero", "absent", "not_applicable", "non_entity", "false"]
NullMeaning = Literal["absent", "unknown", "not_assessed"]

#: Pooling policy by scope kind when the facet does not set one (§6.3).
DEFAULT_POOLING = {"condition": "forbid", "replicate": "list", "batch": "list", "stratum": "group",
                   "partition": "group"}


def _yaml_word(value: Any, *, false: str | None = None, true: str | None = None) -> Any:
    """YAML 1.1 reads ``off``/``on``/``false`` as booleans; map them back to the intended word."""
    if value is False and false is not None:
        return false
    if value is True and true is not None:
        return true
    return value


class ScopeFacet(Strict):
    kind: Literal["condition", "replicate", "batch", "stratum", "partition"] = "condition"
    pooling: Literal["forbid", "group", "list"] | None = None
    pool_ok_for: list[Literal["exists", "count", "distinct"]] = []
    determined_by: list[str] = []

    @model_validator(mode="after")
    def _default_pooling(self) -> "ScopeFacet":
        if self.pooling is None:
            self.pooling = DEFAULT_POOLING[self.kind]  # type: ignore[assignment]
        return self


class CompositeRef(Strict):
    """``{table, on: {target column: local column}}`` for composite foreign keys."""

    table: str
    on: dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def _yaml_on(cls, data: Any) -> Any:
        if isinstance(data, dict) and True in data and "on" not in data:  # YAML 1.1: `on:` is True
            data = {("on" if k is True else k): v for k, v in data.items()}
        return data


class IdTypeFrom(Strict):
    """The id_type of each value comes from a sibling field: ``{field: ^.source, map: {<namespace>: <id_type>}}``."""

    field: str
    map: dict[str, str]


class KindFrom(Strict):
    """Which member kind of a union id_type each stored value must satisfy."""

    column: str
    map: dict[str, str]
    otherwise: str | None = None


class AliasesFrom(Strict):
    table: str
    column: str
    strip_prefix: str | None = None


class CutoffSpec(Strict):
    value: float
    op: Literal["lt", "le", "gt", "ge"]
    meaning: str | None = None
    origin: str | None = None


class TimeFallback(Strict):
    column: str
    precision: Literal["year", "month", "day", "second", "variable"] | None = None


class MemberEnd(Strict):
    """One side of a membership: the set or the member (``{path, id_type}``, ``{parent: id}``)."""

    path: str | None = None
    parent: str | None = None
    id_type: str | None = None
    maps_to: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> "MemberEnd":
        if (self.path is None) == (self.parent is None):
            raise ValueError("a membership end names exactly one of `path` (item-relative) or `parent` (row column)")
        return self


class EnrichmentFamily(Strict):
    include_zero_overlap: bool = True
    size_bounds: dict[str, Any] = {}                   # {min_arg, max_arg, counted_within}


class UniverseSpec(Strict):
    table: str                                         # "table" or "source.table"
    keys: list[str]                                    # columns or paths; ["@row.<key>"] for matrix axes
    mode: Literal["tuple", "union"] = "tuple"
    where: dict[str, Any] | None = None                # predicate JSON; may use {"param": "<argument>"}
    per_scope: list[str] = []
    per_fragment: bool = False


class EnrichmentSpec(Strict):
    universe: UniverseSpec | None = None
    family: EnrichmentFamily = EnrichmentFamily()
    test: str
    correction: str | None = None


class MembershipSpec(Strict):
    set: MemberEnd
    member: MemberEnd
    propagation: Literal["direct", "propagated", "mixed"] = "direct"
    propagate_via: dict[str, Any] | None = None        # {id_type, relation}
    count_grain: str | None = None
    min_resolved_fraction: float | None = None
    enrichment: EnrichmentSpec | None = None


class CensorSpec(Strict):
    column: str
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    value: Any = None
    plus: list[str] = []                               # other reasons a row is absent (value missing, ...)


class CoverageSpec(Strict):
    statement: str                                     # agent-facing scope sentence (required)
    absence_means: Literal["absent", "unknown", "censored"] = "unknown"
    censor: CensorSpec | None = None
    universe: UniverseSpec | None = None               # coverage universe: was the entity measured here
    per_scope: dict[str, str] = {}
    verified: bool = True
    applies_to: Literal["rows", "entity"] = "rows"


class RankSpec(Strict):
    column: str                                        # column path, "match_class" (search) or "similarity"
    direction: Literal["asc", "desc", "asc_abs", "desc_abs"] = "desc"
    nulls: Literal["last", "first"] = "last"
    statistic: str | None = None
    within: list[str] = []
    verified_by: Literal["witness", "source_server_side", "upstream_full_sort", "none"] = "witness"


class ItemKey(Strict):
    """The key of one item of a nested or member container (a bare list is shorthand for ``columns``)."""

    columns: list[str] = []
    identity: Literal["key", "value", "position"] = "key"
    check: Literal["full", "sampled", "none"] = "sampled"
    max_items: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            return {"columns": list(data)}
        return data

    @model_validator(mode="after")
    def _consistent(self) -> "ItemKey":
        if self.identity == "key" and not self.columns and self.max_items != 1:
            raise ValueError("item_key with identity 'key' needs columns (use identity value/position otherwise)")
        return self


class ParseSpec(Strict):
    """Header- or cell-encoded identifiers (``"A1BG (1)"``): named regex groups become virtual fields."""

    pattern: str
    fields: dict[str, "ColumnSpec"]
    on_mismatch: Literal["error", "warn"] = "error"

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        import re
        try:
            groups = re.compile(v).groupindex
        except re.error as exc:
            raise ValueError(f"pattern does not compile: {exc}") from None
        if not groups:
            raise ValueError("pattern needs named groups (?P<name>...)")
        return v

    @model_validator(mode="after")
    def _groups_have_fields(self) -> "ParseSpec":
        import re
        missing = set(self.fields) - set(re.compile(self.pattern).groupindex)
        if missing:
            raise ValueError(f"parse fields without a named group: {sorted(missing)}")
        return self


class _Common(Strict):
    path: str | None = None
    missing: MissingKind | None = None
    missing_values: list[Any] = []
    unknown_when: list[dict[str, Any]] = []
    placeholders: list[Any] = []
    placeholder_when: Literal["equals_key"] | None = None
    optional: bool = False
    present_in: list[str] | None = None
    applies_when: dict[str, list[Any]] | None = None
    by_partition: dict[str, dict[str, Any]] = {}
    verified: bool = True
    verified_by: Literal["data", "manifest", "upstream_doc", "none"] = "data"
    equals: str | None = None
    level: str | None = None
    scope: ScopeFacet | None = None
    unique_within: list[str] = []
    side: Literal["a", "b"] | None = None
    list_delimiter: str | None = None
    parse: Literal["number", "boolean", "date"] | ParseSpec | None = None
    stored_as: str | None = None
    integrity: Literal["full", "partial"] = "full"
    remote_name: str | None = None
    description: str | None = None

    @field_validator("missing", mode="before")
    @classmethod
    def _missing_word(cls, v: Any) -> Any:
        return _yaml_word(v, false="false")


class IdentifierCol(_Common):
    role: Literal["identifier"]
    id_type: str | None = None
    self_: bool = Field(False, alias="self")
    ref: str | CompositeRef | None = None
    maps_to: str | None = None
    cardinality: Literal["one", "many"] = "one"
    form: Literal["parent", "as_stored", "any"] | None = None
    alternate_key: bool = False
    retired_into: str | None = None
    xref: bool = False
    id_type_from: Literal["prefix"] | IdTypeFrom | None = None
    kind_from: KindFrom | None = None
    resolvable: bool | None = None
    authority: Literal["current", "release_snapshot"] | None = None
    #: The hierarchy the rows are already propagated over (OT indirect associations: ``disease.descendants``);
    #: ``include_descendants`` on such a column is refused (§11.5).
    propagated_over: str | list[str] | None = None


class LabelCol(_Common):
    role: Literal["label"]
    of: str | None = None
    id_type: str | None = None
    unique: bool | None = None
    authority: Literal["current", "release_snapshot"] | None = None


class SynonymCol(_Common):
    role: Literal["synonym"]
    of: str | None = None
    synonym_kind: Literal["exact", "alias", "previous", "obsolete", "related", "broad", "narrow"] | None = None
    synonym_kind_from: str | None = None


class _CategoryFacets(_Common):
    vocab: Literal["data"] | list[Any] = "data"
    match: Literal["exact", "casefold"] = "exact"
    aliases: dict[str, Any] = {}
    aliases_from: AliasesFrom | None = None
    values: dict[str, str] = {}
    hierarchy_via: str | None = None
    negative_values: list[Any] = []
    projection_of: str | None = None
    lossy: bool = False

    @field_validator("values", mode="before")
    @classmethod
    def _value_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): val for k, val in v.items()}
        return v


class CategoryCol(_CategoryFacets):
    role: Literal["category"]


class ScopeCol(_CategoryFacets):
    """A category that is a condition of the measurement (shorthand for category + scope facet)."""

    role: Literal["scope"]
    unit_from: str | None = None
    statistic: str | None = None

    @model_validator(mode="after")
    def _implied_scope(self) -> "ScopeCol":
        if self.scope is None:
            self.scope = ScopeFacet(kind="condition")
        return self


class QualifierCol(_Common):
    role: Literal["qualifier"]
    effect: Literal["negate", "direction", "evidence_code", "estimated_actual", "duplicate"]
    default_filter: bool = False


#: The statistic of a measure column that names none.
DEFAULT_STATISTIC = "numeric"


class MeasureCol(_Common):
    role: Literal["measure"]
    statistic: str = DEFAULT_STATISTIC
    fallback: str | None = None                        # a phase-1 statistic used until the named one is registered
    scale: list[float] | None = None
    unit: str | None = None
    unit_from: str | None = None
    direction: Literal["higher_is_stronger", "lower_is_stronger", "signed", "none"] | None = None
    encoding: dict[Any, Any] | None = None
    cutoff: CutoffSpec | None = None
    comparable_within: list[str] = []
    significant_above: float | None = None
    of: str | None = None
    part_of: str | None = None
    columns: dict[str, str] | None = None              # composite measures: {mantissa, exponent}
    family: list[str] | None = None                    # multiple-testing family
    undefined_when: list[dict[str, Any]] = []
    levels: list[str] | None = None                    # ordered string ordinals ("1A" < "1B" ...)

    @field_validator("scale")
    @classmethod
    def _scale_pair(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and (len(v) != 2 or v[0] > v[1]):
            raise ValueError("scale is [low, high] with low <= high")
        return v


class CountCol(_Common):
    role: Literal["count"]
    counts: str | None = None                          # free text: what is counted
    distinct_by: list[str] | str | None = None
    length_of: str | None = None
    comparable_within: list[str] = []


class FlagCol(_Common):
    role: Literal["flag"]
    true_means: str | None = None
    encoding: dict[Any, bool | None] | None = None
    event_of: str | None = None
    partition_items: bool = False


class TimeCol(_Common):
    role: Literal["time"]
    precision: Literal["year", "month", "day", "second", "variable"] | None = None
    unit: str | None = None
    as_of: bool | str | None = None
    fallback: list[TimeFallback] = []
    partial_dates: Literal["latest", "earliest"] | None = None


class PositionCol(_Common):
    role: Literal["position"]
    part: Literal["chrom", "start", "end", "pos"]
    build: str | None = None
    chrom_style: Literal["bare", "chr"] | None = None
    coordinate_base: int | str | None = None


class HierarchyCol(_Common):
    role: Literal["hierarchy"]
    relation: Literal["parent", "child", "ancestor", "descendant"]
    of: str | None = None
    closure: Literal["direct", "transitive"] | None = None
    reflexive: bool | None = None
    closure_of: str | None = None
    inverse_of: str | None = None
    predicate: str | list[str] | None = None
    target_id_type: str | None = None


class _ContainerFacets(_Common):
    fields: dict[str, "ColumnSpec"] = {}
    item_key: ItemKey | None = None
    null_means: NullMeaning | None = None
    empty_means: NullMeaning | None = None
    null_items: Literal["unknown", "skip"] | None = None
    coverage: CoverageSpec | None = None
    rank: list[RankSpec] = []
    grain: str | None = None


class MemberCol(_ContainerFacets):
    role: Literal["member"]
    membership: MembershipSpec
    propagated_over: str | list[str] | None = None
    ref: str | CompositeRef | None = None


class EndpointCol(_Common):
    role: Literal["endpoint"]
    directed: bool | None = None
    id_type: str | None = None
    ref: str | CompositeRef | None = None


class VectorCol(_Common):
    role: Literal["vector"]
    dim: int | None = None
    metric: str | None = None
    normalized: bool | None = None
    norm_column: str | None = None
    element: Literal["float32", "float64"] | None = None


class TextCol(_Common):
    role: Literal["text"]
    searchable: bool | None = None


class ReferenceCol(_Common):
    role: Literal["reference"]
    ref_kinds: list[str] = []


class NestedCol(_ContainerFacets):
    role: Literal["nested"]


class PayloadCol(Strict):
    """Returned verbatim, never filtered or ranked (existence facets only)."""

    role: Literal["payload"]
    optional: bool = False
    present_in: list[str] | None = None
    description: str | None = None


class IgnoreCol(Strict):
    role: Literal["ignore"]
    reason: str | None = None
    optional: bool = False
    present_in: list[str] | None = None
    description: str | None = None


ColumnSpec = Annotated[
    Union[IdentifierCol, LabelCol, SynonymCol, CategoryCol, ScopeCol, QualifierCol, MeasureCol, CountCol, FlagCol,
          TimeCol, PositionCol, HierarchyCol, MemberCol, EndpointCol, VectorCol, TextCol, ReferenceCol, NestedCol,
          PayloadCol, IgnoreCol],
    Field(discriminator="role"),
]

ROLE_MODELS: dict[str, type[BaseModel]] = {
    "identifier": IdentifierCol, "label": LabelCol, "synonym": SynonymCol, "category": CategoryCol,
    "scope": ScopeCol, "qualifier": QualifierCol, "measure": MeasureCol, "count": CountCol, "flag": FlagCol,
    "time": TimeCol, "position": PositionCol, "hierarchy": HierarchyCol, "member": MemberCol,
    "endpoint": EndpointCol, "vector": VectorCol, "text": TextCol, "reference": ReferenceCol, "nested": NestedCol,
    "payload": PayloadCol, "ignore": IgnoreCol,
}

for _model in (ParseSpec, MemberCol, NestedCol, _ContainerFacets):
    _model.model_rebuild()

_ADAPTER: TypeAdapter[Any] = TypeAdapter(ColumnSpec)


def validate_column(data: Any) -> Any:
    """Validate one column spec (a dict with ``role``) into its role model."""
    return _ADAPTER.validate_python(data)


def facet_names(model: type[BaseModel]) -> frozenset[str]:
    """The YAML facet names of a role model (aliases applied), without ``role``."""
    return frozenset((f.alias or name) for name, f in model.model_fields.items() if name != "role")


def is_container(col: Any) -> bool:
    return isinstance(col, (NestedCol, MemberCol))


def container_fields(col: Any) -> dict[str, Any]:
    return dict(col.fields) if is_container(col) else {}


def item_key_of(col: Any) -> ItemKey | None:
    return col.item_key if is_container(col) else None
