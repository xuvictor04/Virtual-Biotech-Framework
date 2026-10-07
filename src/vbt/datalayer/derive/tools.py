"""Native tool specs derived from roles (§10.6, phase 2). No pyarrow.

The data child serves the public verbs of ``service/verbs/public.py``; this module builds what agents
see for them: one :class:`NativeTool` per verb (``mcp__data__<verb>``) with

* a ``table`` enum of the tables and item tables whose roles support the verb (``find``/``lookup``/
  ``aggregate``: every table; ``search``: label or synonym columns; ``vocab``: category or scope
  columns; ``members``: a member column; ``similar``: a vector column; ``neighbors``: an ``edge``
  spec), limited to **ready** tables (``ready``), without ``expose.native: false`` tables and without
  tables whose ``expose.withhold_from`` names the agent;
* a ``where`` schema per table and long-view column derived from the roles (``x-vbt-id-type`` and the
  accepted kinds for identifiers, ``enum`` for small declared vocabularies, ``minimum``/``maximum``
  from a measure's ``scale``, the operators a role supports), under ``x-vbt-where``;
* a description that states what the verb guarantees (resolution, honest totals, coverage).

:func:`native_tool_names` gives the ``mcp__data__*`` names an agent may call (``configs/agents.yaml``
grants them by glob; a verb with no table left for the agent is not listed).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Mapping

from ..roles import Role

__all__ = [
    "NATIVE_SERVER", "NATIVE_VERBS", "NativeTool", "native_tools", "native_tool", "native_tool_names",
    "tables_for", "visible_to", "long_columns", "where_schema", "native_input_schema", "wrap_request",
    "VERB_DESCRIPTIONS",
]

NATIVE_SERVER = "data"
NATIVE_VERBS = ("resolve", "describe", "lookup", "find", "search", "vocab", "members", "aggregate", "similar",
                "neighbors", "expand", "enrich")
#: The most hops ``neighbors`` takes (the data child's ``network.MAX_HOPS``).
MAX_HOPS = 4

VERB_DESCRIPTIONS = {
    "resolve": "Resolve identifiers (symbols, aliases, retired IDs, cross-references) to one canonical key of an "
               "id_type by a recorded rule; unknown values are not_found, several matches are ambiguous.",
    "describe": "Describe a data source or one of its tables: grain, complete key, every column's role and facets, "
                "rank, coverage and the verbs that apply.",
    "lookup": "The record of one complete key of a table (every key column; identifiers resolved).",
    "find": "Rows of a table matching `where` (each entry bound to a long-view column and resolved like a tool "
            "argument), ranked by `rank_by` (else the table's rank; nulls last, ties by key), cut to `limit` "
            "(per `group_by` group), one per `distinct` combination. `_vbt.total` counts every match.",
    "search": "Rows whose key, label or synonym matches `text`, ordered by match class (exact, casefold, previous, "
              "alias, synonym, prefix, word, substring), then rank, then key; each row carries `match`.",
    "vocab": "The distinct values of a category or scope column, in their storage type, with counts when scanned.",
    "members": "The members of one set of a sets table; propagate=true adds the members of every descendant set "
               "once each, with `via` naming the set it was annotated to.",
    "aggregate": "Grouped aggregation over a table's long view (matrix row attributes joined): count, "
                 "count_distinct or the measure's statistic (mean, median, ...); unknown values excluded and "
                 "counted; per-group n; groups below `min_n` dropped and counted.",
    "similar": "Cosine top-k of a vectors table around an anchor; the anchor is excluded from the results and the "
               "total; candidates without a finite score are excluded and counted.",
    "neighbors": "Edges of an edges table around `node` (or `nodes`) over 1 to 4 hops, frontier expanded in key "
                 "order, sources never pooled; each row names the partner; `max_nodes` cuts the last hop (partial).",
    "expand": "The descendants (or ancestors) of terms under an id_type's hierarchy, with each term's depth; the "
              "closure follows the declared predicates only (GO is_a and part_of, never regulates). Exact top-level "
              "ancestors come from direction=ancestors.",
    "enrich": "Over-representation of a gene list in the sets of a member column (hypergeometric p and BH q over "
              "the declared universe; sets below min_size or above max_size dropped and counted).",
}
_TAIL = ("Unknown identifiers are errors, never empty results; an empty result states its coverage and is citable "
         "only as an absence.")
_RANKABLE = (Role.measure.value, Role.count.value, Role.time.value)
_COUNT = Role.count.value                              # the default aggregation shares the role's name


@dataclass
class NativeTool:
    """What one public verb is listed as."""

    verb: str
    name: str                                          # mcp__data__<verb>
    description: str
    input_schema: dict[str, Any]
    tables: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- tables


def visible_to(table: Any, agent: str | None) -> bool:
    """Served by the native tools and not withheld from ``agent``."""
    expose = table.spec.expose
    if not expose.native:
        return False
    return not (agent and agent in expose.withhold_from)


def long_columns(table: Any) -> dict[str, Any]:
    """The long-view columns of a table (a matrix: axis keys and fields, values, row attributes)."""
    matrix = table.physical_spec.matrix if table.kind == "matrix" else None
    if matrix is None:
        return dict(table.columns)
    out: dict[str, Any] = {}
    for ax in (matrix.axes["row"], matrix.axes["col"]):
        out.update(ax.columns)
        if ax.parse is not None:
            out.update(ax.parse.fields)
    out.update(matrix.values)
    return out


def _roles(table: Any) -> set[str]:
    return {str(getattr(c, "role", None)) for c in long_columns(table).values()}


def _supports(verb: str, table: Any) -> bool:
    roles = _roles(table)
    if verb in ("find", "lookup", "aggregate", "describe"):
        return True
    if verb == "search":
        return table.kind != "matrix" and bool(roles & {"label", "synonym"})
    if verb == "vocab":
        return bool(roles & {"category", "scope"})
    if verb == "members":
        return "member" in roles
    if verb == "similar":
        return "vector" in roles
    if verb == "neighbors":
        return table.spec.edge is not None
    if verb == "expand":
        return table.kind == "ontology" or "hierarchy" in roles
    if verb == "enrich":
        return any(getattr(getattr(c, "membership", None), "enrichment", None) is not None
                   for c in table.columns.values() if getattr(c, "role", None) == "member")
    return False


def tables_for(catalog: Any, verb: str, *, agent: str | None = None,
               ready: Collection[str] | Callable[[str], bool] | None = None) -> list[str]:
    """The tables a verb may name for ``agent`` (``ready``: the ready table refs, or a predicate; None: all)."""
    out = []
    for ref in catalog.table_refs():
        name = str(ref)
        try:
            t = catalog.table(name)
        except Exception:  # noqa: BLE001 - a table the catalog cannot resolve is not offered
            continue
        if not visible_to(t, agent) or not _supports(verb, t):
            continue
        if ready is not None and not (ready(name) if callable(ready) else name in ready):
            continue
        out.append(name)
    return sorted(out)


# ---------------------------------------------------------------------------- schemas


def _qualify(catalog: Any, table: Any, id_type: Any) -> str | None:
    if not isinstance(id_type, str) or not id_type:
        return None
    try:
        return catalog.qualify_id_type(id_type, table.descriptor.source)
    except Exception:  # noqa: BLE001
        return None


def _column_schema(catalog: Any, table: Any, name: str, spec: Any, enum_max: int) -> dict[str, Any]:
    role = getattr(spec, "role", None)
    out: dict[str, Any] = {"x-vbt-role": role}
    qualified = _qualify(catalog, table, getattr(spec, "id_type", None))
    if qualified:
        out["x-vbt-id-type"] = qualified
        out["x-vbt-ops"] = ["eq", "in", "ne"]
        out["description"] = (f"{qualified.split(':')[-1]}; symbols, aliases and other accepted kinds are resolved "
                              "(unknown -> not_found)")
        return out
    vocab = getattr(spec, "vocab", None)
    if role in ("category", "scope"):
        out["x-vbt-ops"] = ["eq", "in", "ne"]
        if isinstance(vocab, list) and len(vocab) <= enum_max:
            out["enum"] = list(vocab)
        else:
            out["description"] = f"values of {name} (see mcp__data__vocab)"
        return out
    if role in _RANKABLE:
        out["type"] = "number"
        out["x-vbt-ops"] = ["eq", "ge", "gt", "le", "lt", "ne"]
        scale = getattr(spec, "scale", None)
        if scale:
            out["minimum"], out["maximum"] = scale[0], scale[1]
        out["description"] = "thresholds as {ge: x}; unknown values never pass"
        return out
    out["x-vbt-ops"] = ["eq", "in", "ne", "contains"]
    return out


def where_schema(catalog: Any, ref: str, *, enum_max: int = 64) -> dict[str, Any]:
    """``{column: schema}`` of one table's long view (``x-vbt-where`` entries)."""
    t = catalog.table(ref)
    return {name: _column_schema(catalog, t, name, spec, enum_max) for name, spec in long_columns(t).items()
            if getattr(spec, "role", None) not in ("payload", "ignore", "nested", "vector")}


def native_input_schema(catalog: Any, verb: str, tables: list[str], *, enum_max: int = 64) -> dict[str, Any]:
    """The argument schema of one verb, with the ``table`` enum and the ``where`` schema per table."""
    table = {"type": "string", "enum": tables, "description": "source.table (see mcp__data__describe)"}
    where = {"type": "object", "description": "{column: value | [values] | {op: value}}; ops eq, in, ne, ge, gt, "
                                              "le, lt, contains (per column: x-vbt-where)",
             "x-vbt-where": {ref: where_schema(catalog, ref, enum_max=enum_max) for ref in tables}}
    limit = {"type": "integer", "minimum": 1, "maximum": 1000}
    props: dict[str, Any]
    required: list[str]
    if verb == "resolve":
        names = sorted(f"{s}:{n}" for s, d in catalog.sources.items() for n in d.id_types)
        props = {"id_type": {"type": "string", "enum": names}, "values": {"type": "array", "items": {"type": "string"},
                                                                           "minItems": 1}}
        required = ["id_type", "values"]
    elif verb == "describe":
        sources = sorted({r.split(".")[0] for r in tables})
        props = {"source": {"type": "string", "enum": sources}, "table": table}
        required = ["source"]
    elif verb == "lookup":
        props = {"table": table, "key": {"type": "object", "description": "every key column of the table"}}
        required = ["table", "key"]
    elif verb == "find":
        props = {"table": table, "where": where,
                 "rank_by": {"description": "measure, count or time columns: 'col desc' or {column, direction}"},
                 "limit": {**limit, "default": 50}, "distinct": {"type": "array", "items": {"type": "string"}},
                 "group_by": {"type": "array", "items": {"type": "string"},
                              "description": "rank and limit within each group"},
                 "columns": {"type": "array", "items": {"type": "string"}}}
        required = ["table"]
    elif verb == "search":
        props = {"table": table, "text": {"type": "string", "minLength": 1}, "limit": {**limit, "default": 20}}
        required = ["table", "text"]
    elif verb == "vocab":
        props = {"table": table, "column": {"type": "string"}, "max_values": {"type": "integer", "minimum": 1}}
        required = ["table", "column"]
    elif verb == "members":
        props = {"table": table, "set_id": {"type": "string"},
                 "propagate": {"type": "boolean", "default": False,
                               "description": "true: also the members of every descendant set (each once, `via` "
                                              "names its set)"}}
        required = ["table", "set_id"]
    elif verb == "expand":
        names = sorted(_hierarchy_id_types(catalog, tables))
        props = {"id_type": {"type": "string", **({"enum": names} if names else {})},
                 "values": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                 "direction": {"type": "string", "enum": ["descendants", "ancestors"], "default": "descendants"},
                 "include_self": {"type": "boolean", "default": False},
                 "max_expand": {"type": "integer", "minimum": 1}}
        required = ["id_type", "values"]
    elif verb == "enrich":
        props = {"table": table, "column": {"type": "string", "description": "the member column (default: the "
                                                                               "one with an enrichment contract)"},
                 "genes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                 "universe": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                              "description": "a background list instead of the declared universe"},
                 "scope": {"type": "object"}, "min_size": {"type": "integer", "minimum": 1},
                 "max_size": {"type": "integer", "minimum": 1},
                 "alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                 "propagate": {"description": "auto, true or false"}, "limit": {**limit, "default": 50}}
        required = ["table", "genes"]
    elif verb == "aggregate":
        props = {"table": table, "where": where, "group_by": {"type": "array", "items": {"type": "string"},
                                                              "minItems": 1},
                 "measure": {"type": "string"},
                 "how": {"type": "string", "default": _COUNT,
                         "description": "count, count_distinct, or the measure statistic's aggregation (mean, "
                                        "median, min, max, sum)"},
                 "min_n": {"type": "integer", "minimum": 1, "default": 1}, "limit": {**limit, "default": 100}}
        required = ["table", "group_by"]
    elif verb == "similar":
        props = {"table": table, "anchor": {"type": "string", "description": "the reference entity; excluded from "
                                                                            "the results"},
                 "where": where, "top_k": {**limit, "default": 10}}
        required = ["table", "anchor"]
    else:                                              # neighbors
        props = {"table": table, "node": {"type": "string"},
                 "nodes": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                 "where": where, "limit": {**limit, "default": 50},
                 "hops": {"type": "integer", "minimum": 1, "maximum": MAX_HOPS, "default": 1},
                 "max_nodes": {"type": "integer", "minimum": 1},
                 "score_order": {"type": "boolean", "default": False,
                                 "description": "order each node's edges by score within one source"}}
        required = ["table"]
        out = {"type": "object", "properties": props, "required": required, "additionalProperties": False,
               "anyOf": [{"required": ["node"]}, {"required": ["nodes"]}]}
        return out
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def _hierarchy_id_types(catalog: Any, tables: list[str]) -> set[str]:
    """``source:id_type`` names whose hierarchy (declared, or their universe table's) ``expand`` can walk."""
    out: set[str] = set()
    names = {t.split(".", 1)[1] for t in tables if "." in t}
    for src, desc in (getattr(catalog, "sources", None) or {}).items():
        for name, spec in (getattr(desc, "id_types", None) or {}).items():
            uni = getattr(spec, "universe", None)
            uni = uni[0] if isinstance(uni, list) and uni else uni
            table = getattr(uni, "table", None) if uni is not None and not isinstance(uni, str) else \
                (str(uni).split(".")[0] if uni else None)
            if getattr(spec, "hierarchy", None) is not None or (table and table in names):
                out.add(f"{src}:{name}")
    return out


def native_tool(catalog: Any, verb: str, *, agent: str | None = None,
                ready: Collection[str] | Callable[[str], bool] | None = None, enum_max: int = 64) -> NativeTool | None:
    """The listing of one verb for ``agent`` (None when no table is left for it)."""
    if verb not in NATIVE_VERBS:
        raise ValueError(f"unknown native verb {verb!r}")
    tables = tables_for(catalog, "find" if verb == "resolve" else verb, agent=agent, ready=ready)
    if not tables:
        return None
    schema = native_input_schema(catalog, verb, tables, enum_max=enum_max)
    return NativeTool(verb=verb, name=f"mcp__{NATIVE_SERVER}__{verb}", description=f"{VERB_DESCRIPTIONS[verb]} {_TAIL}",
                      input_schema=schema, tables=tables)


def native_tools(catalog: Any, *, agent: str | None = None,
                 ready: Collection[str] | Callable[[str], bool] | None = None, enum_max: int = 64) -> list[NativeTool]:
    """Every listed native tool for ``agent``."""
    out = []
    for verb in NATIVE_VERBS:
        tool = native_tool(catalog, verb, agent=agent, ready=ready, enum_max=enum_max)
        if tool is not None:
            out.append(tool)
    return out


def native_tool_names(catalog: Any, *, agent: str | None = None,
                      ready: Collection[str] | Callable[[str], bool] | None = None) -> list[str]:
    return [t.name for t in native_tools(catalog, agent=agent, ready=ready)]


def wrap_request(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The schema of a data-child tool that takes its payload as one ``request`` argument (the form of
    the hidden verbs): the native schema under ``request``."""
    return {"type": "object", "properties": {"request": copy.deepcopy(dict(schema))}, "required": ["request"]}
