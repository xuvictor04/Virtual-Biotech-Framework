"""The scope-completeness rule (§11.3 step 7, I7; rev 2: one rule for every serve mode). No pyarrow.

For each key column of the bound table (or item table) that is a scope dimension (role ``scope``
or a ``scope`` facet), that no equality-bound argument fixes (``gateway_only`` arguments
included) and that has more than one distinct value under the call's predicate:

* if rows would be **merged** across it -- the result ``grain`` is coarser than the key and omits
  it, a ``summary_fields`` recompute or a ``limit_grain`` aggregates across it, or the rows do not
  carry it (pass mode, via the field map) -- the pooling policy decides: ``forbid`` ->
  ``incomplete_key`` with ``{dimension, argument, values, unit}`` (values rendered in the storage
  type: ``[0.05, 0.5, 5.0]``); ``group`` -> per-value groups; ``list`` -> rows listed with the
  value, disclosed in ``_vbt.scope``;
* if the order ranks by a measure whose ``comparable_within`` includes it: rank within groups
  (``limit_mode: per_group``; the cut keeps k rows per group) or ``incomplete_key`` with
  ``subkind: incomparable_order`` (``limit_mode: refuse``, or a pass-mode cut that cannot be applied
  per group because the limit was not inflated);
* otherwise the rows carry the dimension and are comparable across it: listed and disclosed.

``pool_ok_for`` lets ``exists``, ``count`` and ``distinct`` verbs pool across the dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..errors import ErrorKind, GatewayError, incomplete_key_payload, json_value
from .fields import get_path

__all__ = ["ScopeDecision", "scope_completeness", "per_group_cut", "grain_columns", "dimension_arg"]


@dataclass
class ScopeDecision:
    listed: dict[str, list[Any]] = field(default_factory=dict)
    grouped: dict[str, list[Any]] = field(default_factory=dict)
    per_group: list[str] = field(default_factory=list)
    fixed: dict[str, Any] = field(default_factory=dict)
    unverified: list[str] = field(default_factory=list)   # dimensions whose values could not be counted
    notes: list[str] = field(default_factory=list)

    def disclosure(self) -> dict[str, Any]:
        """The ``_vbt.scope`` entries this decision adds."""
        out: dict[str, Any] = {}
        for dim, v in self.fixed.items():
            out[dim] = v
        for dim, vals in self.listed.items():
            out[dim] = vals[0] if len(vals) == 1 else {"listed": vals}
        for dim, vals in self.grouped.items():
            out[dim] = {"grouped": vals}
        return out


def _last(name: str) -> str:
    return str(name).lstrip("/").split(".")[-1].replace("[]", "")


def grain_columns(table: Any, grain: str | None) -> list[str] | None:
    """The columns of a declared grain (None: the row grain or unknown)."""
    if not grain or grain in ("row", "rows") or table is None:
        return None
    g = table.spec.grains.get(grain)
    if g is None and table.is_item_table:
        g = table.physical_spec.grains.get(grain)
    if g is None:
        return None
    if isinstance(g, list):
        return list(g)
    return list(g.columns or [])


def dimension_arg(contract: Any, table: str, dim: str) -> str | None:
    """The argument that binds ``dim`` (a ``gateway_only`` argument counts)."""
    for name in contract.args:
        for t, c in contract.arg_columns(name):
            if t == table and (c == dim or _last(c) == _last(dim)):
                return name
    return None


def _carried(contract: Any, dim: str, route: str) -> bool:
    b = contract.binding
    if route == "derived" and b.derived is not None and b.derived.columns:
        return any(_last(c) == _last(dim) for c in b.derived.columns)
    fields = b.result.fields
    if not fields:
        return True                                    # rows keep their own names
    return any((fm.column and _last(fm.column) == _last(dim)) or name == _last(dim) for name, fm in fields.items())


def _order_measures(contract: Any, table: Any, order: Sequence[Any]) -> list[tuple[str, list[str]]]:
    out = []
    for rk in order:
        col = getattr(rk, "column", None) or (rk.get("column") if isinstance(rk, Mapping) else None)
        if not col:
            continue
        spec = table.columns.get(str(col).split(".")[0]) if table is not None else None
        within = list(getattr(spec, "comparable_within", []) or [])
        within += list(getattr(rk, "within", []) or [])
        out.append((str(col), within))
    return out


def _distinct(witness: Any, dim: str) -> list[Any] | None:
    if witness is None:
        return None
    distinct = getattr(witness, "distinct", None)
    if distinct is None and isinstance(witness, Mapping):
        distinct = witness.get("distinct")
    if not distinct:
        return None
    for k, v in distinct.items():
        if k == dim or _last(k) == _last(dim):
            return list(v)
    return None


def scope_completeness(contract: Any, plan: Any, witness: Any, *, fixed: Mapping[str, Any] | None = None,
                       order: Sequence[Any] | None = None, verb: str | None = None, route: str = "upstream",
                       inflated: bool = True, values: Mapping[str, Sequence[Any]] | None = None,
                       storage_types: Mapping[str, str | None] | None = None,
                       units: Mapping[str, str] | None = None,
                       vocab_values: Mapping[str, Sequence[Any]] | None = None) -> ScopeDecision:
    """Apply the rule (module docstring). ``fixed`` maps columns fixed by equality arguments to
    their values; ``values`` gives distinct values per dimension when the witness has none (a
    vocabulary snapshot); ``inflated`` says whether a pass-mode limit was inflated so the gateway can
    cut per group. ``vocab_values`` (every stored value of a dimension) fill an error's ``values`` and
    ``retry_with`` when the values under the call are unknown; they are never disclosed as this call's."""
    decision = ScopeDecision()
    b = contract.binding
    bound = getattr(plan, "bound_table", None) or contract.bound_table
    table = contract.tables.get(bound) if bound else None
    if b is None or table is None:
        return decision
    fixed = dict(fixed or {})
    fixed_names = {_last(k) for k in fixed}
    scope_cols = table.scope_columns()
    dims = [k for k in table.key if k in scope_cols or _last(k) in {_last(s) for s in scope_cols}]
    grain_cols = grain_columns(table, b.result.grain)
    limit_name = contract.limit_arg
    limit_binding = contract.args.get(limit_name) if limit_name else None
    limit_grain_cols = grain_columns(table, limit_binding.limit_grain) if limit_binding is not None else None
    order = list(order if order is not None else b.result.order)
    measures = _order_measures(contract, table, order)
    for dim in dims:
        col = scope_cols.get(dim) or next((v for k, v in scope_cols.items() if _last(k) == _last(dim)), None)
        facet = getattr(col, "scope", None)
        if facet is None:
            continue
        if _last(dim) in fixed_names:
            decision.fixed[_last(dim)] = next(json_value(v, (storage_types or {}).get(dim))
                                              for k, v in fixed.items() if _last(k) == _last(dim))
            continue
        vals = _distinct(witness, dim)
        if vals is None and values is not None:
            vals = list(values.get(dim) or values.get(_last(dim)) or []) or None
        st = (storage_types or {}).get(dim) or (storage_types or {}).get(_last(dim))
        if vals is None:
            decision.unverified.append(dim)
            decision.notes.append(f"the values of {dim} under this call could not be counted")
            vals_known = False
            vals = []
        else:
            vals_known = True
            vals = sorted({repr(json_value(v, st)): json_value(v, st) for v in vals}.values(),
                          key=lambda v: (v is None, str(type(v)), v if v is not None else 0))
            if len(vals) <= 1:
                if vals:
                    decision.listed[_last(dim)] = vals
                continue
        if verb and verb in (facet.pool_ok_for or []):
            decision.notes.append(f"{verb} pools across {dim} (pool_ok_for)")
            continue
        merged = False
        why = ""
        if grain_cols is not None and not any(_last(c) == _last(dim) for c in grain_cols):
            merged, why = True, f"the result grain {b.result.grain} omits {dim}"
        for path, spec in b.result.summary_fields.items():
            if spec != "drop" and not any(_last(g) == _last(dim) for g in getattr(spec, "group_by", []) or []):
                merged, why = True, f"{path} would be computed across {dim}"
                break
        if limit_grain_cols is not None and not any(_last(c) == _last(dim) for c in limit_grain_cols):
            merged, why = True, f"the limit counts {limit_binding.limit_grain}, across {dim}"
        if not merged and not _carried(contract, dim, route):
            merged, why = True, f"the rows do not carry {dim}"
        arg = dimension_arg(contract, bound, dim)
        unit = (units or {}).get(dim) or (units or {}).get(_last(dim))
        if merged:
            if not vals_known:
                continue
            if facet.pooling == "forbid":
                payload = incomplete_key_payload(_last(dim), arg, vals, unit=unit, storage_type=st)
                raise GatewayError(ErrorKind.incomplete_key,
                                   f"results differ by {_last(dim)} ({len(vals)} values) and {why}; pass "
                                   f"{arg or _last(dim)}", tool=f"mcp__{contract.server}__{contract.tool}",
                                   argument=arg, payload=payload)
            if facet.pooling == "group":
                decision.grouped[_last(dim)] = vals
            else:
                decision.listed[_last(dim)] = vals
            continue
        ranked = [m for m, within in measures if any(_last(w) == _last(dim) for w in within)]
        if ranked:
            mode = limit_binding.limit_mode if limit_binding is not None else "per_group"
            if mode == "refuse" or (route == "upstream" and not inflated):
                offered = vals
                if not vals_known:
                    vv = (vocab_values or {}).get(dim) or (vocab_values or {}).get(_last(dim)) or []
                    offered = sorted({repr(json_value(v, st)): json_value(v, st) for v in vv}.values(),
                                     key=lambda v: (v is None, str(type(v)), v if v is not None else 0))
                payload = incomplete_key_payload(_last(dim), arg, offered, unit=unit, subkind="incomparable_order",
                                                 storage_type=st)
                across = f"across {len(vals)} groups" if vals_known else "across its groups"
                raise GatewayError(ErrorKind.incomplete_key,
                                   f"{ranked[0]} is comparable only within {_last(dim)}; a single top-k {across} "
                                   f"is not meaningful; pass {arg or _last(dim)}",
                                   tool=f"mcp__{contract.server}__{contract.tool}", argument=arg, payload=payload)
            decision.per_group.append(_last(dim))
            decision.notes.append(f"ranked by {ranked[0]} within each {_last(dim)}")
            continue
        if vals_known:
            decision.listed[_last(dim)] = vals
    return decision


def _group_key(row: Any, within: Sequence[str]) -> tuple[str, ...]:
    return tuple(repr(json_value(get_path(row, c))) for c in within)


def per_group_cut(rows: Iterable[Any], within: Sequence[str], limit: int | None) -> tuple[list[Any], dict[str, int]]:
    """Keep the first ``limit`` rows of each group (rows already ordered). Returns the kept rows and
    ``{group label: total rows}``."""
    rows = list(rows)
    if not within:
        kept = rows if limit is None else rows[:limit]
        return kept, {"": len(rows)}
    counts: dict[tuple[str, ...], int] = {}
    totals: dict[str, int] = {}
    kept = []
    for row in rows:
        k = _group_key(row, within)
        counts[k] = counts.get(k, 0) + 1
        label = "/".join(k)
        totals[label] = totals.get(label, 0) + 1
        if limit is None or counts[k] <= limit:
            kept.append(row)
    return kept, totals
