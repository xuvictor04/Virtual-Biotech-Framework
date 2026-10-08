"""Argument contracts (§11.3 step 4, §8.1 ``ArgBinding``; F7). No pyarrow.

:func:`apply_arg_contracts` checks and rewrites one call's arguments before anything is
resolved or read, raising a typed :class:`~vbt.datalayer.errors.GatewayError` for every rule
it enforces (I8: every argument is honoured or rejected):

* unknown argument names when the upstream schema forbids them; ``require_any``; ``exclusive``;
* limits (``<= 0``, below ``min`` or above ``max`` -> ``invalid_argument``; never clamped);
  list arguments: ``min_items``/``max_items`` reject instead of truncating; duplicates removed;
* selectors choose the bound table (unknown value -> ``invalid_argument`` with the valid values);
* enum and vocabulary resolution for category and scope columns: exact, then casefold where the
  facet allows it, then ``aliases`` and ``aliases_from``; numeric values are snapped to the
  storage-typed vocabulary snapshot (relative tolerance 1e-6) and the **stored** value is echoed
  in ``_vbt.scope`` (C20);
* ``order_by`` and ``order_direction`` against their enums; ``send_map`` and ``send_as: alias``;
* free text: ``forbid`` and ``pattern``; the substring-collision check for ``interpreted_as:
  substring`` (``PC-3`` also matches ``BxPC-3``); ``re.escape`` for ``interpreted_as: regex``;
  ``escape`` through the named format plugin's ``quote()``; ``wrap``;
* flags compile to their positive ``when_true``/``when_false`` predicates;
* an ``unbound`` argument with a non-default value -> ``unsupported_filter``; a threshold on a
  ``comparable_within`` measure whose group no argument fixes -> ``unsupported_combination``;
* ``output_path`` confinement and write-once renames (disclosed); disclosed schema defaults are
  materialised; ``projection`` arguments get the key, ``unique_within`` and default-filter
  columns injected (or are refused when the output could not carry the complete key);
  ``gateway_only`` arguments are stripped before the upstream call.

:func:`arg_predicate` and :func:`build_predicate` compile bound arguments (after resolution) to
the predicate IR: ``abs`` ops to :class:`~vbt.datalayer.predicate.CmpAbs`, ``binds_any`` to
``Or``, pair bindings (two arguments over the same two endpoint columns) to ``Or(And, And)``,
measure thresholds through the column's statistic plugin (unknown never passes, I6).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..descriptor.columns import DEFAULT_STATISTIC
from ..errors import (
    ErrorKind,
    GatewayError,
    invalid_argument_payload,
    json_value,
    nearest,
    unsupported_combination_payload,
    unsupported_filter_payload,
)
from ..plugins.base import UnsupportedFilter
from ..predicate import (
    And,
    Cmp,
    CmpAbs,
    Contains,
    Eq,
    In,
    KindMatch,
    NonEmpty,
    Not,
    Or,
    Predicate,
    Range,
    TextMatch,
    facet_predicate,
    merge_container_predicates,
)
from ..rowkey import render_float
from . import soma_filter
from .files import confine, write_once

__all__ = [
    "VocabSnapshot", "ArgRequest", "PreparedArgs", "apply_arg_contracts", "arg_predicate", "build_predicate",
    "column_spec", "bound_column", "qualifier_args", "THRESHOLD_OPS", "snap_number", "REL_TOL", "is_present",
    "vocab_key",
]

THRESHOLD_OPS = frozenset({"ge", "gt", "le", "lt", "range", "ge_abs", "gt_abs", "le_abs", "lt_abs"})
EQUALITY_OPS = frozenset({"eq", "in"})
REL_TOL = 1e-6
_ROLE_VALUE = ("category", "scope")


@dataclass
class VocabSnapshot:
    """A column's vocabulary as the data child's ``_vocab`` reports it (``VocabResponse`` fits)."""

    values: list[Any] = field(default_factory=list)
    rendered: list[str] = field(default_factory=list)
    storage_type: str | None = None
    complete: bool = True
    fingerprint: str | None = None


@dataclass
class ArgRequest:
    """An identifier argument the resolver must resolve (``each`` lists included)."""

    arg: str
    binding: Any
    values: list[Any]
    is_list: bool
    table: str | None
    column: str | None


@dataclass
class PreparedArgs:
    args_sent: dict[str, Any]
    scope: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    gateway_args: dict[str, Any] = field(default_factory=dict)
    selected_table: str | None = None
    list_resolution_requests: list[ArgRequest] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    projection: dict[str, list[str]] = field(default_factory=dict)
    renamed_outputs: dict[str, dict[str, str]] = field(default_factory=dict)
    flags: dict[str, Predicate] = field(default_factory=dict)
    order: dict[str, Any] | None = None
    fixed: dict[str, Any] = field(default_factory=dict)        # {column: value} fixed by equality arguments
    stored: dict[str, Any] = field(default_factory=dict)       # {arg: stored value} after snapping
    output_paths: dict[str, Path] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)       # checked values in the table's terms (predicates)


def vocab_key(table: str, column: str) -> str:
    return f"{table}.{column}"


def is_present(value: Any) -> bool:
    return value is not None and not (isinstance(value, (list, tuple, str)) and len(value) == 0)


def _tool(contract: Any) -> str:
    return f"mcp__{contract.server}__{contract.tool}"


def _invalid(contract: Any, arg: str, value: Any, message: str, valid: Sequence[Any] | None = None, *,
             enum_max: int = 64, near: Sequence[Any] = (), **extra: Any) -> GatewayError:
    payload = invalid_argument_payload(arg, value, valid, near=near, enum_max=enum_max,
                                       vocabulary_ref=extra.pop("vocabulary_ref", None))
    payload.update(extra)
    return GatewayError(ErrorKind.invalid_argument, message, tool=_tool(contract), payload=payload)


# ---------------------------------------------------------------------------
# Column lookups
# ---------------------------------------------------------------------------

def bound_column(contract: Any, name: str, binding: Any, selected: str | None = None,
                 selector_value: Any = None) -> tuple[str | None, str | None]:
    """``(source.table, column path)`` the argument binds; a ``{selector value: column}`` map is
    resolved with the selector's value; ``binds_any`` gives the first column."""
    binds = binding.binds
    if isinstance(binds, dict):
        target = None
        for k, v in binds.items():
            if selector_value is not None and (k == selector_value or str(k).casefold() == str(selector_value).casefold()):
                target = v
        if target is None and selected is not None:
            target = next((v for v in binds.values() if str(v).startswith(selected + ".")), None)
        binds = target
    if binds is None and binding.binds_any:
        binds = binding.binds_any[0]
    if not binds:
        return None, None
    parts = str(binds).split(".")
    if len(parts) < 3:
        return None, str(binds)
    return f"{parts[0]}.{parts[1]}", ".".join(parts[2:])


def column_spec(contract: Any, table: str | None, column: str | None) -> Any:
    if not table or not column:
        return None
    try:
        return contract._column(table, column)
    except Exception:  # noqa: BLE001
        return None


def _role(spec: Any) -> str | None:
    return getattr(spec, "role", None)


def _chromosome(contract: Any, name: str, value: Any) -> str:
    """A chromosome name in the bare style of the position role, or ``invalid_argument`` with the valid names."""
    from ..roles import BARE_CHROMOSOMES, bare_chromosome

    got = bare_chromosome(value)
    if got is None:
        why = "; the mitochondrial chromosome is named MT" if str(value).strip().upper() in ("M", "CHRM") else ""
        raise _invalid(contract, name, value, f"{name}={value!r} names no human chromosome{why}",
                       list(BARE_CHROMOSOMES), reason="position")
    return got


_REGION = re.compile(r"^\s*([^:\s]+):([0-9][0-9,]*)-([0-9][0-9,]*)\s*$")


def _region(contract: Any, name: str, value: str) -> dict[str, Any]:
    """``chrom:start-end`` (chr prefix allowed) as ``{chrom, min, max}``; ``invalid_argument`` otherwise."""
    m = _REGION.match(value)
    if m is None:
        raise _invalid(contract, name, value, f"{name} must be a region chrom:start-end (e.g. 1:55000000-55100000)",
                       reason="region")
    start, end = (int(g.replace(",", "")) for g in m.group(2, 3))
    if start > end:
        raise _invalid(contract, name, value, f"{name}: the start {start} is after the end {end}", reason="region")
    return {"chrom": _chromosome(contract, name, m.group(1)), "min": start, "max": end}


def _is_identifier(contract: Any, name: str, binding: Any, table: str | None, column: str | None) -> bool:
    if binding.accepts or binding.role == "anchor":
        return True
    return _role(column_spec(contract, table, column)) in ("identifier", "endpoint")


def _storage_type(snap: Any) -> str | None:
    return getattr(snap, "storage_type", None) if snap is not None else None


def snap_number(value: Any, vocab: Sequence[Any], storage_type: str | None = None) -> Any | None:
    """The vocabulary value equal to ``value`` within ``REL_TOL`` (compared in the storage type),
    or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(str(value))
        except (TypeError, ValueError):
            return None
    x = float(value)
    for v in vocab:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        y = float(v)
        if x == y or abs(x - y) <= REL_TOL * max(abs(x), abs(y)):
            return v
        if storage_type and float(render_float(x, storage_type)) == float(render_float(y, storage_type)):
            return v
    return None


_MISSING = object()


def _in_vocab(value: Any, vocab: Sequence[Any]) -> Any:
    """The vocabulary element equal to ``value`` with the same JSON type (``True`` is not ``1``)."""
    for v in vocab:
        if v == value and isinstance(v, bool) == isinstance(value, bool) and \
                isinstance(v, str) == isinstance(value, str):
            return v
    return _MISSING


def _render(value: Any, storage_type: str | None) -> Any:
    if isinstance(value, float) and not isinstance(value, bool):
        return json_value(value, storage_type or "double")
    return json_value(value)


# ---------------------------------------------------------------------------
# Vocabulary resolution
# ---------------------------------------------------------------------------

def _resolve_value(contract: Any, name: str, binding: Any, spec: Any, value: Any, snap: Any,
                   aliases_snap: Any, *, enum_max: int, notes: list[str]) -> tuple[Any, Any]:
    """``(value to send, stored value to echo)`` for one category/scope value."""
    placeholders = list(getattr(spec, "placeholders", []) or []) + list(getattr(spec, "missing_values", []) or [])
    declared = getattr(spec, "vocab", "data")
    vocab: list[Any] | None
    if isinstance(declared, list):
        vocab = list(declared)
    elif snap is not None and getattr(snap, "values", None) is not None:
        vocab = list(snap.values)
    else:
        vocab = None
    if vocab is not None:
        vocab = [v for v in vocab if not any(v == p for p in placeholders)]
    storage = _storage_type(snap)
    if vocab is None:
        notes.append(f"{name}: vocabulary not available; value not checked")
        return value, _render(value, storage)
    hit = _in_vocab(value, vocab)
    if hit is not _MISSING:
        return hit, _render(hit, storage)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and binding.snap:
        hit = snap_number(value, vocab, storage)
        if hit is not None:
            if float(hit) != float(value):
                notes.append(f"{name}: {value!r} snapped to the stored value {_render(hit, storage)!r}")
            return hit, _render(hit, storage)
    if isinstance(value, str):
        numeric = [v for v in vocab if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric and binding.snap:
            hit = snap_number(value, numeric, storage)
            if hit is not None:
                notes.append(f"{name}: {value!r} snapped to the stored value {_render(hit, storage)!r}")
                return hit, _render(hit, storage)
        casefold = getattr(spec, "match", "exact") == "casefold" or binding.match == "casefold"
        if casefold:
            folded = [v for v in vocab if isinstance(v, str) and v.casefold() == value.casefold()]
            if len(set(folded)) == 1:
                notes.append(f"{name}: {value!r} -> {folded[0]!r} (casefold)")
                return folded[0], folded[0]
        aliases = dict(getattr(spec, "aliases", {}) or {})
        for alias, target in aliases.items():
            if str(alias) == value or (casefold and str(alias).casefold() == value.casefold()):
                notes.append(f"{name}: {value!r} -> {target!r} (alias)")
                return target, _render(target, storage)
        af = getattr(spec, "aliases_from", None)
        if af is not None and aliases_snap is not None:
            prefix = af.strip_prefix or ""
            for stored in aliases_snap.values:
                short = str(stored)[len(prefix):] if prefix and str(stored).startswith(prefix) else str(stored)
                if value == short or value == str(stored) or (casefold and value.casefold() == short.casefold()):
                    target = stored if stored in vocab else short if short in vocab else stored
                    notes.append(f"{name}: {value!r} -> {target!r} (aliases_from {af.table}.{af.column})")
                    return target, _render(target, storage)
    table, col = bound_column(contract, name, binding)
    ref = vocab_key(table, col) if table and col and len(vocab) > enum_max else None
    raise _invalid(contract, name, value, f"{name}={value!r} is not a value of {col or name}",
                   [_render(v, storage) for v in vocab], enum_max=enum_max,
                   near=[_render(v, storage) for v in nearest(value, vocab, 5)], vocabulary_ref=ref)


# ---------------------------------------------------------------------------
# Free text
# ---------------------------------------------------------------------------

def _quote_with(contract: Any, name: str, binding: Any, value: str, registry: Any, notes: list[str]) -> str:
    plugin = registry.find("format", binding.escape) if registry is not None else None
    quote = getattr(plugin, "quote", None) if plugin is not None else None
    if callable(quote):
        try:
            return str(quote(value))
        except Exception as exc:  # noqa: BLE001 - an unquotable value is the caller's input error
            raise _invalid(contract, name, value, f"{name} cannot be quoted safely for {binding.escape}: {exc}",
                           reason="unquotable") from None
    if any(c in value for c in "\"\\\x00"):
        raise _invalid(contract, name, value,
                       f"{name} contains characters that cannot be quoted safely for {binding.escape}",
                       reason="unquotable")
    notes.append(f"{name}: no {binding.escape} quoting plugin; value sent as given (no special characters)")
    return value


def _derived_search(contract: Any) -> bool:
    """A derived ``search`` ranks every match (exact first), so several substring matches are expected."""
    b = getattr(contract, "binding", None)
    d = getattr(b, "derived", None)
    return b is not None and b.serve == "derived" and d is not None and d.verb == "search"


def substring_hits(contract: Any, name: str, binding: Any, value: Any, snap: Any) -> list[str]:
    """The stored values ``value`` matches when upstream matches it as a substring; more than one is a
    collision (``invalid_argument``: upstream would pool them). ``[]`` for exact matching or no snapshot."""
    if binding.interpreted_as not in ("substring", "casefold_substring") or snap is None or \
            not isinstance(value, str):
        return []
    fold = binding.interpreted_as == "casefold_substring"
    needle = value.casefold() if fold else value
    hits = sorted({str(v) for v in snap.values if isinstance(v, str) and needle in (v.casefold() if fold else v)})
    if len(hits) > 1:
        raise _invalid(contract, name, value,
                       f"{name}={value!r} matches several values as a substring ({', '.join(hits[:5])}); "
                       "pass the exact value", list(snap.values), near=hits[:10], reason="substring_collision")
    return hits


def _free_text(contract: Any, name: str, binding: Any, value: Any, snap: Any, registry: Any,
               notes: list[str]) -> Any:
    if not isinstance(value, str):
        return value
    for ch in binding.forbid:
        if ch in value:
            raise _invalid(contract, name, value, f"{name} may not contain {ch!r}", reason="forbidden_character")
    if binding.pattern and not re.fullmatch(binding.pattern, value):
        raise _invalid(contract, name, value, f"{name} does not match {binding.pattern}", reason="pattern")
    out = value
    if not _derived_search(contract):
        substring_hits(contract, name, binding, value, snap)
    if binding.interpreted_as == "regex":
        out = re.escape(out)
        if out != value:
            notes.append(f"{name}: matched as literal text (regex characters escaped)")
    if binding.escape and binding.escape != soma_filter.LANGUAGE:   # SOMA filters are parsed and recompiled
        out = _quote_with(contract, name, binding, out, registry, notes)
    if binding.wrap:
        if not _balanced(out):
            # a ")" that closes the wrap early would let the rest of the value escape it
            raise _invalid(contract, name, value, f"{name} has unbalanced parentheses; it is wrapped as "
                           f"{binding.wrap!r} and may not close the wrap", reason="unbalanced_parentheses")
        out = binding.wrap.replace("{value}", out)
    return out


def _balanced(text: str) -> bool:
    """Parentheses outside double-quoted phrases never close more than they opened, and all close."""
    depth = 0
    quoted = False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch == "(":
            depth += 1
        elif not quoted and ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


# ---------------------------------------------------------------------------
# The contract pass
# ---------------------------------------------------------------------------

def _schema_props(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict((schema or {}).get("properties") or {})


def _nullable(prop: Any) -> bool:
    """The upstream schema accepts null (``type`` lists ``null``, or an ``anyOf``/``oneOf`` branch is null)."""
    if not isinstance(prop, Mapping):
        return False
    t = prop.get("type")
    if t == "null" or (isinstance(t, list) and "null" in t):
        return True
    return any(_nullable(b) for k in ("anyOf", "oneOf") for b in prop.get(k) or [])


def null_is_unrestricted(contract: Any, prop: Any) -> bool:
    """An explicit null on this argument means "no restriction": the gateway serves the call itself
    (derived), or the upstream signature accepts None."""
    b = getattr(contract, "binding", None)
    return (b is not None and b.serve == "derived") or _nullable(prop)


def default_note(contract: Any, binding: Any, prop: Any, default: Any) -> str:
    """How to lift a disclosed default: null where null means no restriction, else the loosest value."""
    if null_is_unrestricted(contract, prop):
        return f"Defaults to {default!r}; pass null for no restriction."
    loosest = binding.min if binding.op in ("ge", "gt", "ge_abs", "gt_abs") else \
        binding.max if binding.op in ("le", "lt", "le_abs", "lt_abs") else None
    if loosest is not None:
        return f"Defaults to {default!r}; pass {loosest:g} for no restriction (null is not accepted)."
    return f"Defaults to {default!r} (null is not accepted)."


def _default(schema: Mapping[str, Any] | None, name: str) -> tuple[bool, Any]:
    prop = _schema_props(schema).get(name)
    if isinstance(prop, Mapping) and "default" in prop:
        return True, prop["default"]
    return False, None


def _measure_guard(contract: Any, name: str, binding: Any, table: str | None, column: str | None,
                   args: Mapping[str, Any], fixed_scope: Mapping[str, Any], selected: str | None) -> None:
    """``unsupported_combination`` for a threshold on a ``comparable_within`` measure whose group is
    not fixed by an equality argument (or ``fixed_scope``)."""
    spec = column_spec(contract, table, column)
    within = list(getattr(spec, "comparable_within", []) or [])
    if not within or binding.op not in THRESHOLD_OPS:
        return
    for group in within:
        g = group.lstrip("/").split(".")[-1]
        if g in fixed_scope or group in fixed_scope:
            continue
        fixer = None
        for other, ob in contract.args.items():
            t, c = bound_column(contract, other, ob, selected)
            if c is not None and (c == group or c.split(".")[-1] == g) and (t == table or t is None):
                fixer = other
                if ob.op == "eq" and is_present(args.get(other)) and not isinstance(args.get(other), list):
                    break
                if ob.op == "in" and isinstance(args.get(other), list) and len(args.get(other)) == 1:
                    break
        else:
            raise GatewayError(ErrorKind.unsupported_combination,
                               f"{name} thresholds {column}, which is comparable only within {group}; fix "
                               f"{fixer or group} to one value, or make separate calls per {group}",
                               tool=_tool(contract), argument=name,
                               payload=unsupported_combination_payload([name], f"{column} is comparable only "
                                                                       f"within {group}", group_argument=fixer))


def _key_columns(contract: Any, table: str | None) -> list[str]:
    t = contract.tables.get(table) if table else None
    return [c for c in (t.key if t is not None else ()) if not c.endswith("#")]


def _projection(contract: Any, name: str, value: Any, table: str | None, schema: Mapping[str, Any] | None,
                notes: list[str]) -> tuple[Any, list[str]]:
    if value is None:
        return value, []
    t = contract.tables.get(table) if table else None
    if t is None:
        return value, []
    is_text = isinstance(value, str)
    current = [c.strip() for c in value.split(",")] if is_text else list(value)
    needed: list[str] = []
    for k in _key_columns(contract, table):
        top = k.split(".")[0].split("[")[0]
        needed.append(top)
        spec = t.columns.get(top)
        for q in getattr(spec, "unique_within", []) or []:
            needed.append(str(q).lstrip("/"))
    for cname, spec in t.columns.items():
        if _role(spec) == "qualifier" and getattr(spec, "default_filter", False):
            needed.append(cname)
    added = [c for c in dict.fromkeys(needed) if c not in current]
    if not added:
        return value, []
    prop = _schema_props(schema).get(name) or {}
    allowed = ((prop.get("items") or {}).get("enum") if isinstance(prop, Mapping) else None)
    if allowed:
        missing = [c for c in added if c not in allowed]
        if missing:
            raise GatewayError(ErrorKind.unsupported_combination,
                               f"{name} cannot include {', '.join(missing)}, so the output could not carry the "
                               "complete key of each row", tool=_tool(contract), argument=name,
                               payload=unsupported_combination_payload([name], "output would lack key columns "
                                                                       + ", ".join(missing)))
    out = current + added
    notes.append(f"{name}: added {', '.join(added)} (key and default-filter columns)")
    return (", ".join(out) if is_text else out), added


def _order_columns(contract: Any, table: str | None) -> list[str]:
    t = contract.tables.get(table) if table else None
    if t is None:
        return []
    return sorted(n for n, c in t.columns.items() if _role(c) in ("measure", "count", "time"))


def apply_arg_contracts(contract: Any, args: Mapping[str, Any], vocab: Mapping[str, Any] | None = None,
                        fixed_scope: Mapping[str, Any] | None = None, *, schema: Mapping[str, Any] | None = None,
                        output_dir: str | Path | None = None, registry: Any = None,
                        enum_max: int = 64, dry: bool = False) -> PreparedArgs:
    """Check and rewrite ``args`` (see the module docstring). ``vocab`` maps ``"source.table.column"``
    to :class:`VocabSnapshot`-like objects; ``fixed_scope`` holds scope values fixed elsewhere.
    ``dry`` (observe mode): output paths are confined and recorded, but no existing file is renamed."""
    vocab = dict(vocab or {})
    fixed_scope = dict(fixed_scope or {})
    args = dict(args or {})
    bindings = contract.args
    out = PreparedArgs(args_sent=dict(args))
    tool = _tool(contract)

    # unknown names
    props = _schema_props(schema)
    if (schema or {}).get("additionalProperties") is False and props:
        allowed = set(props) | {n for n, b in bindings.items() if b.gateway_only} | set(qualifier_args(contract))
        for name in args:
            if name not in allowed:
                raise _invalid(contract, name, args[name], f"unknown argument {name!r}", sorted(allowed),
                               reason="unknown_argument")

    selected = contract.selected_table(args)
    out.selected_table = selected
    selector_value = next((args.get(n) for n in contract.selector_args if args.get(n) is not None), None)

    for group in contract.binding.require_any if contract.binding is not None else ():
        if not any(is_present(args.get(a)) for a in group):
            raise _invalid(contract, group[0], None, f"pass at least one of {', '.join(group)}", list(group),
                           reason="require_any")
    for group in contract.binding.exclusive if contract.binding is not None else ():
        given = [a for a in group if is_present(args.get(a))]
        if len(given) > 1:
            raise GatewayError(ErrorKind.unsupported_combination,
                               f"{' and '.join(given)} cannot be combined in one call", tool=tool,
                               payload=unsupported_combination_payload(given, "exclusive arguments"))

    for name, binding in bindings.items():
        value = args.get(name)
        table, column = bound_column(contract, name, binding, selected, selector_value)
        spec = column_spec(contract, table, column)
        snap = vocab.get(vocab_key(table, column)) if table and column else None

        if binding.role == "limit":
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                    isinstance(value, float) and not value.is_integer()):
                raise _invalid(contract, name, value, f"{name} must be a positive integer", reason="limit")
            lo = max(1, int(binding.min) if binding.min is not None else 1)
            if value < lo or (binding.max is not None and value > binding.max):
                hi = f" and at most {int(binding.max)}" if binding.max is not None else ""
                raise _invalid(contract, name, value, f"{name} must be at least {lo}{hi}", reason="limit_bounds",
                               bounds=[lo, binding.max])
            out.args_sent[name] = int(value)
            continue

        if binding.role == "family_param" and isinstance(value, (int, float)) and not isinstance(value, bool):
            # a parameter of the derived computation (max_hops, min_cell_lines, ...): its declared bounds
            if (binding.min is not None and value < binding.min) or (binding.max is not None and value > binding.max):
                text = " and ".join(t for t in (f"at least {binding.min:g}" if binding.min is not None else "",
                                                f"at most {binding.max:g}" if binding.max is not None else "") if t)
                raise _invalid(contract, name, value, f"{name} must be {text}", reason="bounds",
                               bounds=[binding.min, binding.max])

        if binding.role == "unbound":
            has_default, default = _default(schema, name)
            if name in args and (not has_default or value != default) and value is not None:
                raise GatewayError(ErrorKind.unsupported_filter,
                                   f"{name} cannot be honoured faithfully on this data; drop it", tool=tool,
                                   argument=name, value=value,
                                   payload=unsupported_filter_payload(name, column, "unbound_argument"))
            continue

        if value is None and name in args:
            has_default, default = _default(schema, name)
            if has_default and default is not None and binding.role == "filter":
                if not null_is_unrestricted(contract, props.get(name)):
                    # upstream's signature does not take None: say so before the call, never a source error
                    raise _invalid(contract, name, value, f"{name} cannot be null; "
                                   + default_note(contract, binding, props.get(name), default).replace(
                                       "Defaults to", "omit it for the default"), reason="null")
                tool_binding = getattr(contract, "binding", None)
                if tool_binding is not None and tool_binding.serve == "derived":
                    out.args_sent.pop(name, None)   # derived: null is "no restriction", nothing is sent
                else:
                    # upstream takes None: send it explicitly, or upstream applies the very default the note
                    # says is lifted (CT.gov country=null counted US trials only: 300 of 705)
                    out.args_sent[name] = None
                out.notes.append(f"{name}=null: no restriction (the default {default!r} is not applied)")
        if value is None:
            continue

        if binding.role == "output_path":
            target = confine(value, output_dir, argument=name, tool=tool)
            out.output_paths[name] = target
            if output_dir:
                # upstream resolves a relative path under its own dated folder: send the confined absolute
                # path, so the file that is confined and kept write-once is the file upstream writes
                out.args_sent[name] = str(target)
            if dry:
                if output_dir and target.exists():
                    out.notes.append(f"{name}: the existing {target.name} would be kept as a write-once copy")
                continue
            renamed = write_once(target) if output_dir else None
            if renamed is not None:
                out.renamed_outputs[name] = {"from": str(target), "to": str(renamed)}
                out.notes.append(f"{name}: the existing {target.name} was kept as {renamed.name} (write-once)")
            continue

        if binding.role == "projection":
            new, added = _projection(contract, name, value, selected or table, schema, out.notes)
            out.args_sent[name] = new
            if added:
                out.projection[name] = added
            continue

        if binding.role == "selector":
            continue

        if binding.role == "order_by":
            if binding.values:
                hit = next((k for k in binding.values if k == value or (binding.match == "casefold" and
                            str(k).casefold() == str(value).casefold())), None)
                if hit is None:
                    raise _invalid(contract, name, value, f"unknown {name} value {value!r}",
                                   [str(k) for k in binding.values], reason="order_by")
                target = binding.values[hit]
                out.order = dict(target) if isinstance(target, Mapping) else {"column": str(target)}
                out.args_sent[name] = hit
            else:
                valid = _order_columns(contract, selected or table)
                if valid and value not in valid:
                    raise _invalid(contract, name, value, f"{name} must name a measure, count or time column",
                                   valid, reason="order_by")
                out.order = {"column": value}
            continue

        if binding.role == "order_direction":
            if binding.values:
                keys = {str(k).casefold(): k for k in binding.values}
                k = keys.get(str(value).casefold())
                if k is None:
                    raise _invalid(contract, name, value, f"unknown {name} value {value!r}",
                                   [str(k) for k in binding.values], reason="order_direction")
                direction = binding.values[k]
            elif isinstance(value, bool):
                direction = "asc" if value else "desc"
            elif str(value).lower() in ("asc", "desc"):
                direction = str(value).lower()
            else:
                raise _invalid(contract, name, value, f"{name} must be a boolean or asc/desc",
                               [True, False], reason="order_direction")
            out.order = {**(out.order or {}), "direction": direction}
            continue

        if binding.role == "free_text":
            out.values[name] = value
            out.args_sent[name] = _free_text(contract, name, binding, value, snap, registry, out.notes)
            continue

        if binding.role == "flag":
            pred = None
            if value is True and binding.when_true:
                pred = facet_predicate(binding.when_true, column or name)
            elif value is False and binding.when_false:
                pred = facet_predicate(binding.when_false, column or name)
            if pred is not None:
                out.flags[name] = pred
            continue

        # filters, anchors and the remaining roles
        if isinstance(value, (list, tuple)):
            items = list(value)
            if binding.min_items is not None and len(items) < binding.min_items:
                raise _invalid(contract, name, value, f"{name} needs at least {binding.min_items} values",
                               reason="min_items")
            if binding.max_items is not None and len(items) > binding.max_items:
                raise _invalid(contract, name, value, f"{name} takes at most {binding.max_items} values "
                               f"({len(items)} given); split the call", reason="max_items")
            if binding.dedupe:
                seen: set[str] = set()
                deduped = []
                for v in items:
                    k = repr(json_value(v))
                    if k not in seen:
                        seen.add(k)
                        deduped.append(v)
                if len(deduped) != len(items):
                    out.notes.append(f"{name}: {len(items) - len(deduped)} duplicate value(s) removed")
                items = deduped
            value = items
            out.args_sent[name] = items
            for v in items:                            # each element is matched by upstream on its own
                substring_hits(contract, name, binding, v, snap)

        if isinstance(value, str) and (binding.forbid or binding.pattern):
            for ch in binding.forbid:
                if ch in value:
                    raise _invalid(contract, name, value, f"{name} may not contain {ch!r}",
                                   reason="forbidden_character")
            if binding.pattern and not re.fullmatch(binding.pattern, value):
                raise _invalid(contract, name, value, f"{name} does not match {binding.pattern}", reason="pattern")

        if _role(spec) == "position" and getattr(spec, "part", None) == "chrom" and \
                getattr(spec, "chrom_style", "bare") == "bare" and isinstance(value, (str, int)) and \
                not isinstance(value, bool):
            # the position role stores bare names: chr19 -> 19, chrx -> X; M is refused (named MT)
            chrom = _chromosome(contract, name, value)
            if chrom != value:
                out.notes.append(f"{name}: {value!r} normalized to {chrom!r} (bare chromosome names)")
            value = chrom
            out.args_sent[name] = chrom

        if binding.op == "range" and isinstance(value, str) and _role(spec) == "position":
            # a genomic region 'chr1:1000-2000': the chromosome and the position range, checked here; upstream
            # receives the region string it parses itself
            out.values[name] = _region(contract, name, value)
            continue

        if _is_identifier(contract, name, binding, table, column):
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            out.list_resolution_requests.append(ArgRequest(name, binding, values, isinstance(value, (list, tuple)),
                                                           table, column))
            continue

        if binding.op in THRESHOLD_OPS:
            _measure_guard(contract, name, binding, table, column, args, fixed_scope, selected)
            nums = value if isinstance(value, (list, tuple)) else [value]
            for v in nums:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float)
                                                                              and not math.isfinite(v)):
                    raise _invalid(contract, name, value, f"{name} must be a number", reason="number")
                x = abs(v) if binding.op.endswith("_abs") else v
                if (binding.min is not None and x < binding.min) or (binding.max is not None and x > binding.max):
                    raise GatewayError(ErrorKind.unsupported_filter,
                                       f"{name}={v} is outside the range the data holds "
                                       f"([{binding.min}, {binding.max}])", tool=tool, argument=name, value=v,
                                       payload=unsupported_filter_payload(name, column, "scale",
                                                                          confirmed_range=[binding.min, binding.max]))
            out.values[name] = value
            continue

        if _role(spec) in _ROLE_VALUE or getattr(spec, "scope", None) is not None or binding.values:
            if binding.values and _role(spec) not in _ROLE_VALUE:
                keys = {str(k): k for k in binding.values}
                if str(value) not in keys and not isinstance(value, list):
                    raise _invalid(contract, name, value, f"unknown {name} value {value!r}", list(keys),
                                   reason="enum")
            elif binding.op in EQUALITY_OPS or binding.op in ("contains", "overlaps"):
                aliases_snap = None
                af = getattr(spec, "aliases_from", None)
                if af is not None:
                    src = table.split(".")[0] if table else ""
                    at = af.table if "." in af.table else f"{src}.{af.table}"
                    aliases_snap = vocab.get(vocab_key(at, af.column))
                if isinstance(value, list):
                    resolved = [_resolve_value(contract, name, binding, spec, v, snap, aliases_snap,
                                               enum_max=enum_max, notes=out.notes) for v in value]
                    out.args_sent[name] = [r[0] for r in resolved]
                    out.stored[name] = [r[1] for r in resolved]
                else:
                    sent, stored = _resolve_value(contract, name, binding, spec, value, snap, aliases_snap,
                                                  enum_max=enum_max, notes=out.notes)
                    out.args_sent[name] = sent
                    out.stored[name] = stored
                    if binding.op == "eq" and column:
                        out.fixed[column] = sent
                if getattr(spec, "scope", None) is not None and column:
                    out.scope[name] = out.stored[name]
        elif binding.op == "eq" and column and not isinstance(value, list):
            out.fixed[column] = value

        out.values[name] = out.args_sent.get(name, value)
        if binding.send_map:
            sent = out.args_sent.get(name)
            mapped = binding.send_map.get(sent, binding.send_map.get(str(sent)))
            if mapped is not None:
                out.args_sent[name] = mapped
        elif binding.send_as == "alias" and spec is not None and getattr(spec, "aliases", None):
            reverse = {v: k for k, v in spec.aliases.items()}
            sent = out.args_sent.get(name)
            if sent in reverse:
                out.args_sent[name] = reverse[sent]

    # disclosed schema defaults
    for name, prop in props.items():
        if name in args or not isinstance(prop, Mapping) or "default" not in prop:
            continue
        binding = bindings.get(name)
        if binding is None or not binding.default_disclosed or binding.role in ("limit", "output_path", "unbound",
                                                                                 "projection"):
            continue
        default = prop["default"]
        out.defaults[name] = default
        if default is None:
            continue
        out.args_sent[name] = default
        out.values[name] = default
        out.scope[name] = json_value(default)
        out.notes.append(default_note(contract, binding, prop, default).rstrip(".").replace(
            "Defaults to", f"{name} defaults to"))
        table, column = bound_column(contract, name, binding, selected, selector_value)
        if binding.op == "eq" and column:
            out.fixed[column] = default

    _overlapping_groups(contract, args, vocab, selected, selector_value)
    for name, binding in bindings.items():
        if binding.gateway_only and name in out.args_sent:
            out.gateway_args[name] = out.args_sent.pop(name)
    for name in qualifier_args(contract):
        if name in out.args_sent and name not in bindings:
            value = out.args_sent.pop(name)
            if not isinstance(value, bool):
                raise _invalid(contract, name, value, f"{name} must be true or false", [True, False],
                               reason="boolean")
            out.gateway_args[name] = value
    return out


#: Gateway-only override of a qualifier the gateway enforces by default (§7, §11.3).
QUALIFIER_ARGS = {"duplicate": "include_duplicates", "negate": "include_negated"}


def qualifier_args(contract: Any) -> dict[str, str]:
    """``{argument: effect}`` for the default qualifier filters this tool's tables carry (a ``duplicate`` or
    ``negate`` qualifier, at the top or in a container): ``include_duplicates`` / ``include_negated`` are then
    gateway-only arguments, advertised ``x-gateway`` and never sent upstream (upstream rejects them)."""
    out: dict[str, str] = {}
    b = getattr(contract, "binding", None)
    if b is None or getattr(contract, "generic", False):
        return out
    refs = {contract.bound_table, b.result.rows_of, b.derived.table if b.derived is not None else None,
            *b.reads.keys()}

    def walk(cols: Mapping[str, Any]) -> None:
        for col in cols.values():
            effect = getattr(col, "effect", None)
            if getattr(col, "role", None) == "qualifier" and effect in QUALIFIER_ARGS:
                out[QUALIFIER_ARGS[effect]] = effect
            fields = getattr(col, "fields", None)
            if fields:
                walk(fields)

    for ref in refs:
        t = contract.tables.get(ref) if ref else None
        if t is not None:
            walk(t.physical_spec.columns if t.is_item_table else t.spec.columns)
    return out


def _group_args(contract: Any) -> list[str]:
    """Arguments whose values each select a group of rows that are compared or pooled separately: the
    ``groups`` of a computed field and list arguments upstream matches element by element as text."""
    names: list[str] = []
    b = getattr(contract, "binding", None)
    for fm in (b.result.fields.values() if b is not None else ()):
        for g in (fm.computed or {}).get("groups") or ():
            if g in contract.args and g not in names:
                names.append(str(g))
    # a derived handler that compares groups names them in its split options ({<handler>: {groups: [...]}})
    split = (b.derived.split if b is not None and b.derived is not None else None) or {}
    for opts in split.values():
        for g in (opts.get("groups") or () if isinstance(opts, Mapping) else ()):
            if g in contract.args and g not in names:
                names.append(str(g))
    for n, a in contract.args.items():
        if a.each and a.interpreted_as in ("substring", "casefold_substring") and n not in names:
            names.append(n)
    return names


def _overlapping_groups(contract: Any, args: Mapping[str, Any], vocab: Mapping[str, Any], selected: str | None,
                        selector_value: Any) -> None:
    """Two groups (or two elements of one list) whose text matches share a stored value would count the
    same rows twice, or in the first group only (upstream's if/elif): ``invalid_argument``."""
    groups: list[tuple[str, Any, set[str]]] = []
    for name in _group_args(contract):
        binding = contract.args[name]
        value = args.get(name)
        if not is_present(value):
            continue
        table, column = bound_column(contract, name, binding, selected, selector_value)
        snap = vocab.get(vocab_key(table, column)) if table and column else None
        fold = binding.interpreted_as == "casefold_substring" or binding.match == "casefold"
        sub = binding.interpreted_as in ("substring", "casefold_substring")
        for v in (value if isinstance(value, (list, tuple)) else [value]):
            if not isinstance(v, str):
                continue
            needle = v.casefold() if fold else v
            if snap is not None:
                stored = [str(x) for x in snap.values if isinstance(x, str)]
                hits = {x for x in stored if sub and needle in (x.casefold() if fold else x)} or \
                    {x for x in stored if (x.casefold() if fold else x) == needle}
            else:
                hits = {needle}
            groups.append((name, v, {h.casefold() if fold else h for h in hits} or {needle}))
    for i, (n1, v1, h1) in enumerate(groups):
        for n2, v2, h2 in groups[i + 1:]:
            shared = sorted(h1 & h2)
            if shared:
                raise _invalid(contract, n2, v2,
                               f"{n1}={v1!r} and {n2}={v2!r} select overlapping groups ({', '.join(shared[:5])}); "
                               "pass groups that share no value", list(shared), reason="overlapping_groups")


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def _leaf(op: str, column: str, value: Any) -> Predicate | None:
    if op == "eq":
        if isinstance(value, (list, tuple)):
            return In(column, tuple(value))
        return Eq(column, value)
    if op == "in":
        return In(column, tuple(value) if isinstance(value, (list, tuple)) else (value,))
    if op == "contains":
        if isinstance(value, (list, tuple)):
            return Or(tuple(Contains(column, v) for v in value)) if len(value) > 1 else Contains(column, value[0])
        return Contains(column, value)
    if op == "overlaps":
        vals = value if isinstance(value, (list, tuple)) else [value]
        return Or(tuple(Contains(column, v) for v in vals)) if len(vals) > 1 else Contains(column, vals[0])
    if op in ("ge", "gt", "le", "lt", "ne"):
        return Cmp(column, op, value)
    if op.endswith("_abs"):
        return CmpAbs(column, op, value)
    if op == "range":
        if isinstance(value, Mapping):
            return Range(column, value.get("min", value.get("lo")), value.get("max", value.get("hi")))
        lo, hi = (list(value) + [None, None])[:2] if isinstance(value, (list, tuple)) else (value, None)
        return Range(column, lo, hi)
    if op == "nonempty":
        return NonEmpty(column) if value not in (False, None) else Not(NonEmpty(column))
    if op == "has_kind":
        return KindMatch(column, str(value))
    return None


def arg_predicate(contract: Any, name: str, binding: Any, value: Any, *, selected: str | None = None,
                  selector_value: Any = None, registry: Any = None,
                  confirmed: Mapping[str, Any] | None = None,
                  fixed_scope: Mapping[str, Any] | None = None, default: bool = False) -> Predicate | None:
    """The predicate one bound argument contributes (None: no row predicate, e.g. an anchor, a
    limit or an argument without a column). A ``default`` (a schema default the caller did not
    choose) is the tool's own per-row filter: the ``comparable_within`` guard applies only to
    thresholds the caller passed."""
    if value is None or binding.role not in ("filter", "flag", "free_text"):
        return None
    if binding.role == "free_text":
        if not (binding.binds or binding.binds_any) or binding.interpreted_as in ("engine", "regex"):
            return None
        mode = {"exact": "exact", "substring": "substring", "casefold_substring": "casefold_substring"}[
            binding.interpreted_as]
        texts: list[str] = []
        if binding.binds:
            _table, column = bound_column(contract, name, binding, selected, selector_value)
            texts.extend([column] if column else [])
        texts.extend(".".join(str(c).split(".")[2:]) or str(c) for c in binding.binds_any)   # any column matches
        matches = [TextMatch(c, str(value), mode) for c in dict.fromkeys(texts)]
        return (matches[0] if len(matches) == 1 else Or(tuple(matches))) if matches else None
    columns: list[str] = []
    if binding.binds is not None:
        table, column = bound_column(contract, name, binding, selected, selector_value)
        if column:
            columns.append(column)
    else:
        table = None
    for c in binding.binds_any:
        parts = str(c).split(".")
        table = table or (f"{parts[0]}.{parts[1]}" if len(parts) >= 3 else None)
        columns.append(".".join(parts[2:]) if len(parts) >= 3 else str(c))
    if not columns:
        return None
    preds: list[Predicate] = []
    for column in dict.fromkeys(columns):
        spec = column_spec(contract, table, column)
        p: Predicate | None = None
        if binding.op in THRESHOLD_OPS and _role(spec) == "measure" and registry is not None:
            plugin = registry.find("statistic", getattr(spec, "statistic", DEFAULT_STATISTIC)) or \
                registry.find("statistic", getattr(spec, "fallback", None) or DEFAULT_STATISTIC)
            if plugin is not None:
                try:
                    p = plugin.predicate(column, binding.op, value, spec, (confirmed or {}).get(column),
                                         dict(fixed_scope or {}))
                except UnsupportedFilter as exc:
                    if exc.reason == "group" and default:
                        p = None                       # the plain comparison below
                    elif exc.reason == "group":
                        group = list(getattr(exc, "group", None) or [])
                        raise GatewayError(ErrorKind.unsupported_combination, str(exc), tool=_tool(contract),
                                           argument=name,
                                           payload=unsupported_combination_payload(
                                               [name], str(exc), group_argument=group[0] if group else None)) from None
                    else:
                        raise GatewayError(ErrorKind.unsupported_filter, str(exc), tool=_tool(contract), argument=name,
                                           value=value, payload=unsupported_filter_payload(
                                               name, column, exc.reason or "scale",
                                               confirmed_range=getattr(exc, "confirmed_range", None))) from None
        if p is None:
            p = _leaf(binding.op, column, value)
        if p is not None and binding.op == "range" and isinstance(value, Mapping) and value.get("chrom") is not None:
            chrom_col = _chrom_column(contract, table)
            if chrom_col:
                p = And((Eq(chrom_col, value["chrom"]), p))
        if p is not None:
            preds.append(p)
    if not preds:
        return None
    return preds[0] if len(preds) == 1 else Or(tuple(preds))


def _chrom_column(contract: Any, table: str | None) -> str | None:
    """The chromosome column (``role: position, part: chrom``) of ``table``, for a region's chromosome."""
    t = (getattr(contract, "tables", None) or {}).get(table) if table else None
    cols = getattr(getattr(t, "spec", None), "columns", None) or {}
    return next((n for n, c in cols.items() if _role(c) == "position" and getattr(c, "part", None) == "chrom"), None)


def _pairs(contract: Any) -> list[tuple[str, str, list[str]]]:
    """Pairs of arguments that both bind the same two columns with ``binds_any`` (endpoint pairs)."""
    out = []
    names = [n for n, b in contract.args.items() if len(b.binds_any) == 2]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sorted(contract.args[a].binds_any) == sorted(contract.args[b].binds_any):
                out.append((a, b, list(contract.args[a].binds_any)))
    return out


def _compile_flag(contract: Any, name: str, binding: Any, pred: Predicate, *, selected: str | None,
                  registry: Any, confirmed: Mapping[str, Any] | None,
                  fixed_scope: Mapping[str, Any] | None) -> Predicate:
    """A flag's ``when_true``/``when_false`` names meanings (``none_recorded``); on a measure column its
    statistic turns them into stored codes, and refuses an encoding not confirmed from the data (I9)."""
    if registry is None or not isinstance(pred, (Eq, In)):
        return pred
    table, column = bound_column(contract, name, binding, selected)
    spec = column_spec(contract, table, column) if column else None
    if _role(spec) != "measure":
        return pred
    plugin = registry.find("statistic", getattr(spec, "statistic", None) or DEFAULT_STATISTIC)
    if plugin is None:
        return pred
    op, value = ("in", list(pred.values)) if isinstance(pred, In) else ("eq", pred.value)
    try:
        return plugin.predicate(pred.column, op, value, spec, (confirmed or {}).get(column), dict(fixed_scope or {}))
    except UnsupportedFilter as exc:
        raise GatewayError(ErrorKind.unsupported_filter, str(exc), tool=_tool(contract), argument=name, value=True,
                           payload=unsupported_filter_payload(name, column, exc.reason or "unconfirmed_encoding")
                           ) from None


def build_predicate(contract: Any, values: Mapping[str, Any], *, selected: str | None = None,
                    registry: Any = None, confirmed: Mapping[str, Any] | None = None,
                    fixed_scope: Mapping[str, Any] | None = None,
                    flags: Mapping[str, Predicate] | None = None,
                    defaults: Iterable[str] = ()) -> tuple[Predicate | None, dict[str, Predicate]]:
    """``(predicate, {arg: predicate})`` for the bound arguments in ``values`` (resolved values).
    Conjuncts on one container are merged into one ``Any`` (C21)."""
    per_arg: dict[str, Predicate] = {}
    selector_value = next((values.get(n) for n in contract.selector_args if values.get(n) is not None), None)
    paired: set[str] = set()
    for a, b, cols in _pairs(contract):
        va, vb = values.get(a), values.get(b)
        if va is None or vb is None:
            continue
        c1, c2 = (".".join(str(c).split(".")[2:]) for c in cols)
        per_arg[f"{a}+{b}"] = Or((And((Eq(c1, va), Eq(c2, vb))), And((Eq(c1, vb), Eq(c2, va)))))
        paired.update((a, b))
    for name, binding in contract.args.items():
        if name in paired:
            continue
        if flags and name in flags:
            per_arg[name] = _compile_flag(contract, name, binding, flags[name], selected=selected, registry=registry,
                                          confirmed=confirmed, fixed_scope=fixed_scope)
            continue
        if binding.role == "flag":
            continue
        p = arg_predicate(contract, name, binding, values.get(name), selected=selected, selector_value=selector_value,
                          registry=registry, confirmed=confirmed, fixed_scope=fixed_scope, default=name in defaults)
        if p is not None:
            per_arg[name] = p
    if not per_arg:
        return None, per_arg
    merged = merge_container_predicates(list(per_arg.values()))
    return (merged[0] if len(merged) == 1 else And(tuple(merged))), per_arg
