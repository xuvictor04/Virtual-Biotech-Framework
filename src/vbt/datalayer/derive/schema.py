"""Argument JSON schemas derived from roles (§10.1). No pyarrow.

The upstream schema from ``list_tools`` is kept and annotated (never loosened):

* identifier arguments: ``x-vbt-id-type`` (the bound column's qualified id_type),
  ``x-vbt-accepts``, examples and one sentence from the identifier plugins; no strict
  ``pattern`` for resolvable kinds (symbols must reach the resolver), ``pattern`` for
  non-resolvable kinds; ``minItems``/``maxItems`` for list arguments;
* category and scope arguments: ``enum`` from the vocabulary snapshot when it has at most
  ``data.derive.enum_max`` values (numbers rendered in their storage type, ``0.05``), else
  ``x-vbt-vocabulary: table.column``;
* scope key columns the upstream signature lacks: an auto-derived gateway-only property
  (``x-gateway: true``) with the vocabulary as enum (:func:`auto_scope_args`); declared
  ``gateway_only`` arguments likewise;
* ``selector`` and ``order_by``/``order_direction`` enums; measure thresholds with
  ``minimum``/``maximum`` from the binding and the column's scale, the ``abs`` wording and
  "unknown values never pass"; limits with ``minimum: 1`` and the binding's (or the memory
  derived) maximum and the ``limit_grain`` wording; regex arguments as literal text; engine
  arguments with their ``engine_doc``; ``require_any``/``exclusive`` as ``x-vbt-*`` plus one sentence;
  schema defaults on filters disclosed; anchors described as excluded from the results.

Generic (unbound) tools are returned unchanged. The data child's public verbs (``data.find`` ...) get
their native schema from :mod:`.tools` (table enums, ``where`` schemas per long-view column); a listing
that takes the payload as one ``request`` argument (the hidden verbs' form) gets it under ``request``.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from ..errors import json_value

__all__ = ["annotate_schema", "auto_scope_args", "bound_id_type", "native_schema", "THRESHOLD_OPS"]

THRESHOLD_OPS = frozenset({"ge", "gt", "le", "lt", "range", "ge_abs", "gt_abs", "le_abs", "lt_abs"})
_OP_WORDS = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "ge_abs": "|x| >=", "gt_abs": "|x| >", "le_abs": "|x| <=",
             "lt_abs": "|x| <"}


def _split(binds: str) -> tuple[str | None, str | None]:
    parts = str(binds).split(".")
    if len(parts) < 3:
        return None, str(binds)
    return f"{parts[0]}.{parts[1]}", ".".join(parts[2:])


def _column(contract: Any, binding: Any) -> tuple[str | None, str | None, Any]:
    binds = binding.binds
    if isinstance(binds, dict):
        binds = next(iter(binds.values()), None)
    if binds is None and binding.binds_any:
        binds = binding.binds_any[0]
    if not binds:
        return None, None, None
    table, col = _split(binds)
    spec = None
    if table and col:
        try:
            spec = contract._column(table, col)
        except Exception:  # noqa: BLE001
            spec = None
    return table, col, spec


def bound_id_type(contract: Any, binding: Any, catalog: Any = None) -> str | None:
    """The qualified id_type of the column an identifier argument binds."""
    table, _, spec = _column(contract, binding)
    id_type = getattr(spec, "id_type", None)
    if not id_type or not table:
        return None
    if ":" in id_type:
        return id_type
    src = table.split(".")[0]
    if catalog is not None:
        try:
            return catalog.qualify_id_type(id_type, src)
        except Exception:  # noqa: BLE001
            pass
    return f"{src}:{id_type}"


def auto_scope_args(contract: Any, schema: Mapping[str, Any] | None = None) -> dict[str, str]:
    """``{argument: column}`` for scope key columns of the bound table that no argument binds and
    the upstream schema does not name: the gateway accepts them as ``x-gateway`` arguments,
    fixes the dimension with them and strips them before the upstream call."""
    b = getattr(contract, "binding", None)
    bound = contract.bound_table if b is not None else None
    t = contract.tables.get(bound) if bound else None
    if t is None:
        return {}
    props = set(((schema or {}).get("properties") or {}).keys())
    bound_cols = {c for n in contract.args for tb, c in contract.arg_columns(n) if tb == bound}
    scope = t.scope_columns()
    out: dict[str, str] = {}
    for k in t.key:
        if k not in scope or k in bound_cols:
            continue
        name = k.split(".")[-1].replace("[]", "")
        if name in props or name in contract.args:
            continue
        out[name] = k
    return out


def _append(prop: dict[str, Any], sentence: str) -> None:
    text = str(prop.get("description") or "").strip()
    if not sentence or sentence in text:
        return
    if text and not text.endswith((".", ":", "!", "?")):
        text += "."
    prop["description"] = f"{text} {sentence}".strip()


def _vocab_enum(snap: Any, enum_max: int) -> list[Any] | None:
    if snap is None:
        return None
    values = list(getattr(snap, "values", []) or [])
    if not values or len(values) > enum_max or not getattr(snap, "complete", True):
        return None
    st = getattr(snap, "storage_type", None)
    return [json_value(v, st) for v in values]


def _plugin(registry: Any, catalog: Any, qualified: str | None) -> Any:
    if registry is None or catalog is None or not qualified:
        return None
    try:
        _, spec = catalog.id_type(qualified)
    except Exception:  # noqa: BLE001
        return None
    return registry.find("identifier", spec.plugin), spec


def native_schema(contract: Any, schema: Mapping[str, Any] | None, catalog: Any, *, enum_max: int = 64,
                  agent: str | None = None, ready: Any = None) -> dict[str, Any] | None:
    """The schema of a data-child public verb (None for any other tool); under ``request`` when the
    listed schema takes the payload as one ``request`` argument."""
    from .tools import NATIVE_SERVER, NATIVE_VERBS, native_tool, wrap_request

    if catalog is None or getattr(contract, "server", None) != NATIVE_SERVER or contract.tool not in NATIVE_VERBS:
        return None
    tool = native_tool(catalog, contract.tool, agent=agent, ready=ready, enum_max=enum_max)
    if tool is None:
        return None
    props = (schema or {}).get("properties") or {}
    return wrap_request(tool.input_schema) if set(props) == {"request"} else tool.input_schema


def annotate_schema(contract: Any, schema: Mapping[str, Any] | None, *, catalog: Any = None, registry: Any = None,
                    vocab: Mapping[str, Any] | None = None, enum_max: int = 64,
                    limit_max: Mapping[str, int] | None = None) -> dict[str, Any]:
    """The annotated schema (a copy; see the module docstring). ``vocab`` maps
    ``"source.table.column"`` to vocabulary snapshots; ``limit_max`` the memory-derived maximum
    per limit argument."""
    out: dict[str, Any] = copy.deepcopy(dict(schema or {"type": "object"}))
    b = getattr(contract, "binding", None)
    if b is None or getattr(contract, "generic", False):
        return out
    native = native_schema(contract, schema, catalog, enum_max=enum_max)
    if native is not None:
        return native
    vocab = dict(vocab or {})
    props: dict[str, Any] = out.setdefault("properties", {})
    for name, a in contract.args.items():
        prop = props.get(name)
        if prop is None:
            if not a.gateway_only:
                continue
            prop = props[name] = {}
        table, col, spec = _column(contract, a)
        role = getattr(spec, "role", None)
        if a.gateway_only:
            prop["x-gateway"] = True
        if a.accepts or a.role == "anchor":
            qualified = bound_id_type(contract, a, catalog)
            if qualified:
                prop["x-vbt-id-type"] = qualified
            accepts = []
            for k in a.accepts:
                try:
                    accepts.append(catalog.qualify_id_type(k, table.split(".")[0] if table else None)
                                   if catalog is not None else k)
                except Exception:  # noqa: BLE001
                    accepts.append(k)
            prop["x-vbt-accepts"] = accepts
            examples: list[str] = []
            resolvable = True
            for k in accepts or ([qualified] if qualified else []):
                got = _plugin(registry, catalog, k)
                if not got or got[0] is None:
                    continue
                plugin, spec_t = got
                examples.extend(list(getattr(plugin, "examples", ()) or ())[:1])
                if not spec_t.resolvable:
                    resolvable = False
                    if getattr(plugin, "canonical", None):
                        target = prop.get("items") if prop.get("type") == "array" else prop
                        if isinstance(target, dict):
                            target["pattern"] = plugin.canonical
            if resolvable:
                prop.pop("pattern", None)
                if isinstance(prop.get("items"), dict):
                    prop["items"].pop("pattern", None)
            if examples:
                prop["examples"] = list(dict.fromkeys(examples))[:3]
            names = ", ".join(k.split(":")[-1] for k in accepts) or (qualified or "").split(":")[-1]
            _append(prop, f"Accepts {names}; resolved to one {(qualified or 'identifier').split(':')[-1]} "
                          "(unknown values are errors, never empty results).")
            if a.role == "anchor":
                _append(prop, "The reference entity; excluded from the results.")
        if a.each or prop.get("type") == "array":
            if a.min_items is not None:
                prop["minItems"] = a.min_items
            if a.max_items is not None:
                prop["maxItems"] = a.max_items
        if role in ("category", "scope") or getattr(spec, "scope", None) is not None:
            snap = vocab.get(f"{table}.{col}") if table and col else None
            declared = getattr(spec, "vocab", None)
            enum = list(declared) if isinstance(declared, list) and len(declared) <= enum_max else \
                _vocab_enum(snap, enum_max)
            if enum and a.op in ("eq", "in") and not a.accepts:
                target = prop["items"] if isinstance(prop.get("items"), dict) else prop
                if target is prop and "default" in prop and prop["default"] is None:
                    enum = enum + [None]                   # a null default stays valid
                target["enum"] = enum
            elif table and col and not a.accepts:
                prop["x-vbt-vocabulary"] = f"{table}.{col}"
                _append(prop, f"Values of {col} (see mcp__data__vocab).")
            meanings = getattr(spec, "values", None) or {}
            if meanings:
                _append(prop, "; ".join(f"{k}: {v}" for k, v in list(meanings.items())[:8]) + ".")
        if a.role == "selector":
            keys = [str(k) for k in (a.values or (a.binds if isinstance(a.binds, dict) else {}))]
            if keys:
                prop["enum"] = keys
        if a.role == "order_by":
            if a.values:
                prop["enum"] = [str(k) for k in a.values]
            else:
                t = contract.tables.get(table) if table else None
                cols = sorted(n for n, c in (t.columns.items() if t is not None else [])
                              if getattr(c, "role", None) in ("measure", "count", "time"))
                if cols:
                    prop["enum"] = cols
        if a.role == "order_direction" and not a.values:
            prop.setdefault("type", "boolean")
        if a.op in THRESHOLD_OPS and role == "measure":
            lo, hi = a.min, a.max
            scale = getattr(spec, "scale", None)
            if scale and not a.op.endswith("_abs"):
                lo = scale[0] if lo is None else max(lo, scale[0])
                hi = scale[1] if hi is None else min(hi, scale[1])
            if lo is not None:
                prop["minimum"] = lo
            if hi is not None:
                prop["maximum"] = hi
            words = f"Keeps rows with {col} {_OP_WORDS.get(a.op, a.op)} this value"
            if a.op.endswith("_abs"):
                words += " (compared on the absolute value)"
            _append(prop, words + "; unknown values never pass.")
        if a.role == "limit":
            prop["minimum"] = max(1, int(a.min) if a.min is not None else 1)
            cap = a.max
            mem = (limit_max or {}).get(name)
            if mem is not None:
                cap = mem if cap is None else min(cap, mem)
            if cap is not None:
                prop["maximum"] = int(cap)
            if a.limit_grain:
                _append(prop, f"Counts {a.limit_grain}s, not rows.")
        if a.role == "free_text" and a.interpreted_as == "regex":
            _append(prop, "Literal text, not a pattern.")
        if a.interpreted_as == "engine" and a.engine_doc:
            _append(prop, f"Matched by the source's search engine: {a.engine_doc}.")
        if "default" in prop and prop.get("default") is not None and a.role == "filter" and a.default_disclosed:
            from ..gateway.contracts import default_note
            _append(prop, default_note(contract, a, prop, prop["default"]))
    for name, column in auto_scope_args(contract, schema).items():
        bound = contract.bound_table
        t = contract.tables.get(bound) if bound else None
        spec = t.scope_columns().get(column) if t is not None else None
        prop = {"x-gateway": True}
        enum = _vocab_enum(vocab.get(f"{bound}.{column}"), enum_max)
        declared = getattr(spec, "vocab", None)
        if isinstance(declared, list):
            enum = list(declared)
        if enum:
            prop["enum"] = enum
        pooling = getattr(getattr(spec, "scope", None), "pooling", "forbid")
        if pooling == "forbid":
            prop["description"] = (f"Results are per {name}; pass one value, or the call fails with incomplete_key "
                                   "when several apply.")
        else:
            prop["description"] = f"Results are listed per {name}; pass one value to restrict them."
        props[name] = prop
    from ..gateway.contracts import qualifier_args
    for name, effect in qualifier_args(contract).items():
        if name in props or name in contract.args:
            continue
        what = "duplicate records (cells counted in several datasets)" if effect == "duplicate" else \
            "negative findings (evidence that the relation does NOT hold)"
        props[name] = {"type": "boolean", "default": False, "x-gateway": True,
                       "description": f"Include {what}, which are excluded by default; their number is disclosed."}
    if b.require_any:
        out["x-vbt-require-any"] = [list(g) for g in b.require_any]
        _append(out, "Pass at least one of: " + "; ".join(", ".join(g) for g in b.require_any) + ".")
    if b.exclusive:
        out["x-vbt-exclusive"] = [list(g) for g in b.exclusive]
        _append(out, "At most one of: " + "; ".join(", ".join(g) for g in b.exclusive) + ".")
    return out
