"""The predicate IR and its three-valued (Kleene) evaluation oracle (§9.2). No pyarrow.

Bound arguments compile to these frozen dataclasses; format plugins compile them to native
filters and return what they cannot push down as a residual, which the reader evaluates
with :func:`evaluate`. :func:`evaluate` is also the oracle of the format conformance suite
and of the gateway transforms, so pushdown can never change a result.

Semantics (I6, rev 2):

* a comparison with null, NaN (or a declared unknown code, mapped to null before
  evaluation) is unknown (``None``); ``Not(unknown)`` is unknown; ``And``/``Or`` are Kleene;
* ``Any([]) = False``, ``Any(null list) = None``; ``Any`` with no true item and at least one
  unknown item is ``None`` (null items count as unknown unless ``skip_null_items``);
  ``All([]) = True``, ``All(null) = None``;
* inside ``Any``/``All`` paths are item-relative: ``x`` names a field of the item, ``^.x``
  a field of the enclosing item (or row), ``/x`` a table-level column; multi-level paths
  (``a[].b[]``) nest one quantifier per level;
* a leaf comparison on a path that crosses a list (``tissues[].name``) is existential
  over the leaves; several bindings on one container are merged into **one** ``Any`` by
  :func:`merge_container_predicates` so their conditions apply to the same item (C21);
* ``CensoredCmp(time, event, "<=", t)`` is true when ``time <= t`` and the event occurred,
  false when ``time > t``, unknown when ``time <= t`` and the case is censored.

JSON form (descriptors, ``universe.where``, ipc): ``{"eq": [col, v]}``, ``{"in": [col, [v...]]}``,
``{"lt"|"le"|"gt"|"ge"|"ne": [col, v]}``, ``{"lt_abs"|...: [col, v]}``, ``{"range": [col, lo, hi]}``,
``{"contains": [path, v]}``, ``{"any"|"all": [path, p]}``, ``{"nonempty": path}``,
``{"has_kind": [path, id_type]}``, ``{"censored": [time, event, op, v]}``,
``{"text": [col, text, mode]}``, ``{"is_null": col}``, ``{"not": p}``, ``{"and"|"or": [p...]}``;
a value ``{"param": name}`` is a call-time :class:`Param`.
"""

from __future__ import annotations

import math
import re
import typing as t
from collections import OrderedDict
from dataclasses import dataclass, field

from .roles import LIST, ItemCond, Path, Segment, format_path, parse_path

__all__ = [
    "Predicate", "Eq", "In", "Cmp", "CmpAbs", "Range", "Contains", "Any", "All", "NonEmpty", "KindMatch",
    "CensoredCmp", "TextMatch", "IsNull", "Not", "And", "Or", "Param", "RankKey", "CMP_OPS", "TEXT_MODES",
    "evaluate", "columns", "to_json", "from_json", "facet_predicate", "merge_container_predicates",
    "kleene_and", "kleene_or", "kleene_not", "is_null", "PredicateError", "map_columns",
]

Truth = t.Optional[bool]

_OP_ALIASES = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "ne": "!=", "eq": "==",
               "<": "<", "<=": "<=", ">": ">", ">=": ">=", "!=": "!=", "==": "==", "=": "=="}
_OP_WORDS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "!=": "ne", "==": "eq"}
CMP_OPS = ("<", "<=", ">", ">=", "!=")
TEXT_MODES = ("exact", "casefold", "substring", "casefold_substring", "word")


class PredicateError(ValueError):
    """A malformed predicate or predicate JSON."""


def _op(op: str, *, allow_eq: bool = False) -> str:
    norm = _OP_ALIASES.get(str(op).strip())
    if norm is None or (norm == "==" and not allow_eq):
        raise PredicateError(f"unknown comparison operator {op!r}")
    return norm


@dataclass(frozen=True)
class Param:
    """A call-time parameter inside a universe or overlay predicate (``{"param": "<argument>"}``)."""

    name: str


@dataclass(frozen=True)
class Eq:
    column: str
    value: t.Any


@dataclass(frozen=True)
class In:
    column: str
    values: tuple[t.Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            object.__setattr__(self, "values", tuple(self.values) if isinstance(self.values, (list, set, frozenset))
                               else (self.values,))


@dataclass(frozen=True)
class Cmp:
    column: str
    op: str                                            # "<", "<=", ">", ">=", "!=" (words accepted)
    value: t.Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", _op(self.op))


@dataclass(frozen=True)
class CmpAbs:
    """``|column| op value`` (upstream's strict ``>`` is kept by binding ``gt_abs``)."""

    column: str
    op: str
    value: t.Any

    def __post_init__(self) -> None:
        op = str(self.op)
        object.__setattr__(self, "op", _op(op[:-4] if op.endswith("_abs") else op))


@dataclass(frozen=True)
class Range:
    column: str
    lo: t.Any = None
    hi: t.Any = None
    lo_inclusive: bool = True
    hi_inclusive: bool = True


@dataclass(frozen=True)
class Contains:
    """The list at ``path`` (or the leaves of a path crossing lists) contains ``value``."""

    path: str
    value: t.Any


@dataclass(frozen=True)
class Any:
    """Some item of the container at ``path`` satisfies ``pred`` (item-relative paths)."""

    path: str
    pred: "Predicate"
    skip_null_items: bool = False


@dataclass(frozen=True)
class All:
    """Every item of the container at ``path`` satisfies ``pred``."""

    path: str
    pred: "Predicate"
    skip_null_items: bool = False


@dataclass(frozen=True)
class NonEmpty:
    path: str


@dataclass(frozen=True)
class KindMatch:
    """The identifier at ``path`` is of ``id_type`` (decided by the caller's ``kind_of``)."""

    path: str
    id_type: str


@dataclass(frozen=True)
class CensoredCmp:
    """A right-censored time compared with a threshold (time + event pair)."""

    time: str
    event: str
    op: str
    value: t.Any

    def __post_init__(self) -> None:
        op = _op(self.op)
        if op == "!=":
            raise PredicateError("CensoredCmp supports <, <=, > and >= only")
        object.__setattr__(self, "op", op)


@dataclass(frozen=True)
class TextMatch:
    column: str
    text: str
    mode: str = "exact"                                # exact | casefold | substring | casefold_substring | word

    def __post_init__(self) -> None:
        if self.mode not in TEXT_MODES:
            raise PredicateError(f"unknown text match mode {self.mode!r} ({', '.join(TEXT_MODES)})")


@dataclass(frozen=True)
class IsNull:
    column: str


@dataclass(frozen=True)
class Not:
    pred: "Predicate"


@dataclass(frozen=True)
class And:
    preds: tuple["Predicate", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "preds", tuple(self.preds))


@dataclass(frozen=True)
class Or:
    preds: tuple["Predicate", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "preds", tuple(self.preds))


Predicate = t.Union[Eq, In, Cmp, CmpAbs, Range, Contains, Any, All, NonEmpty, KindMatch, CensoredCmp, TextMatch,
                    IsNull, Not, And, Or]
_PRED_TYPES = (Eq, In, Cmp, CmpAbs, Range, Contains, Any, All, NonEmpty, KindMatch, CensoredCmp, TextMatch, IsNull,
               Not, And, Or)


@dataclass(frozen=True)
class RankKey:
    """One ordering key: nulls last by default, ties broken by the canonical key (by the caller)."""

    column: str
    direction: t.Literal["asc", "desc", "asc_abs", "desc_abs"] = "desc"
    nulls: t.Literal["last", "first"] = "last"
    statistic: str | None = None
    within: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.direction not in ("asc", "desc", "asc_abs", "desc_abs"):
            raise PredicateError(f"unknown rank direction {self.direction!r}")
        if self.nulls not in ("last", "first"):
            raise PredicateError(f"unknown nulls placement {self.nulls!r}")
        object.__setattr__(self, "within", tuple(self.within or ()))

    def to_json(self) -> dict[str, t.Any]:
        out: dict[str, t.Any] = {"column": self.column, "direction": self.direction, "nulls": self.nulls}
        if self.statistic:
            out["statistic"] = self.statistic
        if self.within:
            out["within"] = list(self.within)
        return out

    @classmethod
    def from_json(cls, data: t.Mapping[str, t.Any]) -> "RankKey":
        return cls(column=data["column"], direction=data.get("direction", "desc"), nulls=data.get("nulls", "last"),
                   statistic=data.get("statistic"), within=tuple(data.get("within") or ()))


# ---------------------------------------------------------------------------
# Kleene helpers
# ---------------------------------------------------------------------------

def is_null(x: t.Any) -> bool:
    """None or NaN (unknown)."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def kleene_not(x: Truth) -> Truth:
    return None if x is None else not x


def kleene_and(values: t.Iterable[Truth]) -> Truth:
    unknown = False
    for v in values:
        if v is False:
            return False
        if v is None:
            unknown = True
    return None if unknown else True


def kleene_or(values: t.Iterable[Truth]) -> Truth:
    unknown = False
    for v in values:
        if v is True:
            return True
        if v is None:
            unknown = True
    return None if unknown else False


# ---------------------------------------------------------------------------
# Path selection
# ---------------------------------------------------------------------------

_ABSENT = object()


def _get(value: t.Any, name: str, axis: str | None) -> t.Any:
    if not isinstance(value, t.Mapping):
        return None
    if name in value:
        return value[name]
    if axis is not None:  # long-view rows may keep the prefixed name
        return value.get(f"@{axis}.{name}")
    return None


def _base(path: Path, stack: t.Sequence[t.Any]) -> t.Any:
    if path.absolute:
        return stack[0]
    if path.axis is not None:
        return stack[0]
    idx = len(stack) - 1 - path.up
    return stack[idx] if idx >= 0 else _ABSENT


def _expand(values: list[t.Any], bracket: t.Any) -> list[t.Any]:
    out: list[t.Any] = []
    for v in values:
        if v is None:
            out.append(None)                           # a null list: unknown
            continue
        items = v if isinstance(v, (list, tuple)) else [v]
        for item in items:
            if bracket == LIST:
                out.append(item)
            elif isinstance(bracket, ItemCond) and bracket.matches(item):
                out.append(item)
    return out


def _select(path: Path, stack: t.Sequence[t.Any]) -> tuple[list[t.Any], bool]:
    """``(values, multi)``: the value(s) at ``path``; ``multi`` when a list was crossed, in which
    case a null list contributes one ``None`` and an empty list contributes nothing."""
    base = _base(path, stack)
    if base is _ABSENT:
        raise PredicateError(f"path {format_path(path)!r} goes above the row")
    vals = [base]
    multi = False
    for seg in path.segments:
        if seg.name:
            vals = [_get(v, seg.name, path.axis) for v in vals]
        for b in seg.brackets:
            multi = True
            vals = _expand(vals, b)
    return vals, multi


def _resolve(value: t.Any, params: t.Mapping[str, t.Any] | None) -> t.Any:
    if isinstance(value, Param):
        if params is None or value.name not in params:
            raise PredicateError(f"no value for parameter {value.name!r}")
        return params[value.name]
    return value


def _is_number(x: t.Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _equal(x: t.Any, v: t.Any) -> bool:
    if isinstance(x, bool) or isinstance(v, bool):
        return isinstance(x, bool) and isinstance(v, bool) and x == v
    if _is_number(x) and _is_number(v):
        return float(x) == float(v)
    return bool(x == v)


def _compare(x: t.Any, op: str, v: t.Any) -> Truth:
    if is_null(x) or is_null(v):
        return None
    if op == "!=":
        return not _equal(x, v)
    if op == "==":
        return _equal(x, v)
    if _is_number(x) and _is_number(v):
        a, b = float(x), float(v)
    elif isinstance(x, str) and isinstance(v, str):
        a, b = x, v  # type: ignore[assignment]
    else:
        return None  # incomparable types: unknown, never a silent pass or fail
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


def _leaf(column: str, stack: t.Sequence[t.Any], test: t.Callable[[t.Any], Truth]) -> Truth:
    """Apply ``test`` to the value at ``column``; existential (Kleene OR) over list leaves."""
    vals, multi = _select(parse_path(column), stack)
    if not multi:
        v = vals[0] if vals else None
        if isinstance(v, (list, tuple)):               # a role on list<T> applies per element (§6.4)
            return kleene_or(None if is_null(e) else test(e) for e in v)
        return None if is_null(v) else test(v)
    return kleene_or(None if is_null(v) else test(v) for v in vals)


def _word_match(text: str, needle: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(needle.casefold()) + r"(?!\w)", text.casefold()) is not None


def _text(x: t.Any, needle: str, mode: str) -> Truth:
    if not isinstance(x, str):
        x = str(x)
    if mode == "exact":
        return x == needle
    if mode == "casefold":
        return x.casefold() == needle.casefold()
    if mode == "substring":
        return needle in x
    if mode == "casefold_substring":
        return needle.casefold() in x.casefold()
    return _word_match(x, needle)


def _quantify(is_all: bool, path_text: str, pred: Predicate, stack: list[t.Any],
              params: t.Mapping[str, t.Any] | None, kind_of: t.Any, skip_null: bool) -> Truth:
    path = parse_path(path_text)
    split = path.split_container()
    if split is None:  # no list in the path: the value itself is the single item
        vals, _ = _select(path, stack)
        v = vals[0] if vals else None
        if v is None:
            return None
        items = v if isinstance(v, (list, tuple)) else [v]
        rest = None
    else:
        container, rest = split
        last = container.segments[-1]
        # The container value before expansion: a null container is unknown, not a list of nulls.
        head = container.segments[:-1] + ((Segment(last.name),) if last.name else ())
        vals, _ = _select(Path(head, container.axis, container.up, container.absolute), stack)
        value = vals[0] if vals else None
        if value is None:
            return None
        items = [value]
        for b in last.brackets:
            items = _expand(items, b)
    results: list[Truth] = []
    for item in items:
        if is_null(item):
            if skip_null:
                continue
            results.append(None)
            continue
        inner = stack + [item]
        if rest is not None and rest.segments:
            if rest.crosses_list:
                results.append(_quantify(is_all, format_path(rest), pred, inner, params, kind_of, skip_null))
                continue
            sub, _ = _select(rest, inner)
            sv = sub[0] if sub else None
            if is_null(sv):
                results.append(None)
                continue
            inner = inner + [sv]
        results.append(_eval(pred, inner, params, kind_of))
    if is_all:
        return kleene_and(results)
    return kleene_or(results)


def _eval(p: Predicate, stack: list[t.Any], params: t.Mapping[str, t.Any] | None, kind_of: t.Any) -> Truth:
    if isinstance(p, And):
        return kleene_and(_eval(q, stack, params, kind_of) for q in p.preds)
    if isinstance(p, Or):
        return kleene_or(_eval(q, stack, params, kind_of) for q in p.preds)
    if isinstance(p, Not):
        return kleene_not(_eval(p.pred, stack, params, kind_of))
    if isinstance(p, Eq):
        v = _resolve(p.value, params)
        return _leaf(p.column, stack, lambda x: None if is_null(v) else _equal(x, v))
    if isinstance(p, In):
        wanted: list[t.Any] = []
        for item in p.values:
            r = _resolve(item, params)
            wanted.extend(r if isinstance(r, (list, tuple, set, frozenset)) else [r])
        has_null = any(is_null(w) for w in wanted)
        concrete = [w for w in wanted if not is_null(w)]
        return _leaf(p.column, stack,
                     lambda x: True if any(_equal(x, w) for w in concrete) else (None if has_null else False))
    if isinstance(p, Cmp):
        v = _resolve(p.value, params)
        return _leaf(p.column, stack, lambda x: _compare(x, p.op, v))
    if isinstance(p, CmpAbs):
        v = _resolve(p.value, params)
        return _leaf(p.column, stack, lambda x: _compare(abs(x), p.op, v) if _is_number(x) else None)
    if isinstance(p, Range):
        lo, hi = _resolve(p.lo, params), _resolve(p.hi, params)

        def in_range(x: t.Any) -> Truth:
            parts: list[Truth] = []
            if lo is not None:
                parts.append(_compare(x, ">=" if p.lo_inclusive else ">", lo))
            if hi is not None:
                parts.append(_compare(x, "<=" if p.hi_inclusive else "<", hi))
            return kleene_and(parts)

        return _leaf(p.column, stack, in_range)
    if isinstance(p, Contains):
        v = _resolve(p.value, params)
        vals, multi = _select(parse_path(p.path), stack)
        if not multi:
            container = vals[0] if vals else None
            if container is None:
                return None
            vals = list(container) if isinstance(container, (list, tuple, set, frozenset)) else [container]
        if is_null(v):
            return None
        return kleene_or(None if is_null(x) else _equal(x, v) for x in vals)
    if isinstance(p, (Any, All)):
        return _quantify(isinstance(p, All), p.path, p.pred, stack, params, kind_of, p.skip_null_items)
    if isinstance(p, NonEmpty):
        vals, multi = _select(parse_path(p.path), stack)
        if not multi:
            v = vals[0] if vals else None
            if v is None:
                return None
            return len(v) > 0 if isinstance(v, (list, tuple, dict, str, set)) else True
        return kleene_or(None if x is None else True for x in vals)
    if isinstance(p, IsNull):
        vals, multi = _select(parse_path(p.column), stack)
        if not multi:
            return is_null(vals[0] if vals else None)
        return all(is_null(x) for x in vals)
    if isinstance(p, KindMatch):
        if kind_of is None:
            return _leaf(p.path, stack, lambda x: None)
        return _leaf(p.path, stack, lambda x: kind_of(x, p.id_type))
    if isinstance(p, TextMatch):
        return _leaf(p.column, stack, lambda x: _text(x, p.text, p.mode))
    if isinstance(p, CensoredCmp):
        tv, _ = _select(parse_path(p.time), stack)
        ev, _ = _select(parse_path(p.event), stack)
        time_v = tv[0] if tv else None
        event_v = ev[0] if ev else None
        cmp = _compare(time_v, p.op, _resolve(p.value, params))
        if cmp is None:
            return None
        event = None if is_null(event_v) else bool(event_v)
        if p.op in ("<", "<="):
            if not cmp:
                return False                           # survived (event-free or censored) past t
            return True if event is True else None     # censored before t: unknown
        if cmp:
            return True                                # time beyond t: the event (if any) came later
        return False if event is True else None
    raise PredicateError(f"not a predicate: {p!r}")


def evaluate(p: Predicate | None, row: t.Any, params: t.Mapping[str, t.Any] | None = None, *,
             kind_of: t.Callable[[t.Any, str], Truth] | None = None) -> Truth:
    """Evaluate ``p`` on ``row`` with three-valued logic: ``True``, ``False`` or ``None`` (unknown).
    ``None`` predicate is ``True``. ``kind_of(value, id_type)`` decides :class:`KindMatch`."""
    if p is None:
        return True
    return _eval(p, [row], params, kind_of)


# ---------------------------------------------------------------------------
# Introspection, JSON, merging
# ---------------------------------------------------------------------------

def _join(prefix: str, inner: str) -> str:
    if inner.startswith(("/", "^.")):
        return inner
    if inner.startswith("[]"):
        return prefix + inner[2:] if prefix.endswith("[]") else prefix + inner
    return f"{prefix}.{inner}" if inner else prefix


def columns(p: Predicate | None) -> set[str]:
    """Column paths ``p`` reads (quantifier bodies joined to their container path)."""
    if p is None:
        return set()
    if isinstance(p, (And, Or)):
        return set().union(*(columns(q) for q in p.preds)) if p.preds else set()
    if isinstance(p, Not):
        return columns(p.pred)
    if isinstance(p, (Any, All)):
        return {_join(p.path, c) for c in columns(p.pred)} or {p.path}
    if isinstance(p, (Contains, NonEmpty, KindMatch)):
        return {p.path}
    if isinstance(p, CensoredCmp):
        return {p.time, p.event}
    return {p.column}  # type: ignore[union-attr]


def map_columns(p: Predicate, fn: t.Callable[[str], str]) -> Predicate:
    """``p`` with every top-level column path passed through ``fn`` (quantifier bodies untouched)."""
    if isinstance(p, And):
        return And(tuple(map_columns(q, fn) for q in p.preds))
    if isinstance(p, Or):
        return Or(tuple(map_columns(q, fn) for q in p.preds))
    if isinstance(p, Not):
        return Not(map_columns(p.pred, fn))
    if isinstance(p, (Any, All)):
        return type(p)(fn(p.path), p.pred, p.skip_null_items)
    if isinstance(p, (Contains,)):
        return Contains(fn(p.path), p.value)
    if isinstance(p, NonEmpty):
        return NonEmpty(fn(p.path))
    if isinstance(p, KindMatch):
        return KindMatch(fn(p.path), p.id_type)
    if isinstance(p, CensoredCmp):
        return CensoredCmp(fn(p.time), fn(p.event), p.op, p.value)
    if isinstance(p, Eq):
        return Eq(fn(p.column), p.value)
    if isinstance(p, In):
        return In(fn(p.column), p.values)
    if isinstance(p, Cmp):
        return Cmp(fn(p.column), p.op, p.value)
    if isinstance(p, CmpAbs):
        return CmpAbs(fn(p.column), p.op, p.value)
    if isinstance(p, Range):
        return Range(fn(p.column), p.lo, p.hi, p.lo_inclusive, p.hi_inclusive)
    if isinstance(p, TextMatch):
        return TextMatch(fn(p.column), p.text, p.mode)
    if isinstance(p, IsNull):
        return IsNull(fn(p.column))
    return p


def _value_json(v: t.Any) -> t.Any:
    if isinstance(v, Param):
        return {"param": v.name}
    if isinstance(v, tuple):
        return [_value_json(x) for x in v]
    return v


def _value_from(v: t.Any) -> t.Any:
    if isinstance(v, dict) and set(v) == {"param"}:
        return Param(str(v["param"]))
    if isinstance(v, list):
        return tuple(_value_from(x) for x in v)
    return v


def to_json(p: Predicate) -> dict[str, t.Any]:
    """The compact JSON form (see the module docstring)."""
    if isinstance(p, Eq):
        return {"eq": [p.column, _value_json(p.value)]}
    if isinstance(p, In):
        return {"in": [p.column, [_value_json(v) for v in p.values]]}
    if isinstance(p, Cmp):
        return {_OP_WORDS[p.op]: [p.column, _value_json(p.value)]}
    if isinstance(p, CmpAbs):
        return {_OP_WORDS[p.op] + "_abs": [p.column, _value_json(p.value)]}
    if isinstance(p, Range):
        args = [p.column, _value_json(p.lo), _value_json(p.hi)]
        if not (p.lo_inclusive and p.hi_inclusive):
            args += [p.lo_inclusive, p.hi_inclusive]
        return {"range": args}
    if isinstance(p, Contains):
        return {"contains": [p.path, _value_json(p.value)]}
    if isinstance(p, (Any, All)):
        out: dict[str, t.Any] = {"any" if isinstance(p, Any) else "all": [p.path, to_json(p.pred)]}
        if p.skip_null_items:
            out["skip_null_items"] = True
        return out
    if isinstance(p, NonEmpty):
        return {"nonempty": p.path}
    if isinstance(p, KindMatch):
        return {"has_kind": [p.path, p.id_type]}
    if isinstance(p, CensoredCmp):
        return {"censored": [p.time, p.event, _OP_WORDS[p.op], _value_json(p.value)]}
    if isinstance(p, TextMatch):
        return {"text": [p.column, p.text, p.mode]}
    if isinstance(p, IsNull):
        return {"is_null": p.column}
    if isinstance(p, Not):
        return {"not": to_json(p.pred)}
    if isinstance(p, And):
        return {"and": [to_json(q) for q in p.preds]}
    if isinstance(p, Or):
        return {"or": [to_json(q) for q in p.preds]}
    raise PredicateError(f"not a predicate: {p!r}")


def _args(data: t.Mapping[str, t.Any], key: str, n: int | tuple[int, ...]) -> list[t.Any]:
    args = data[key]
    ns = (n,) if isinstance(n, int) else n
    if not isinstance(args, (list, tuple)) or len(args) not in ns:
        raise PredicateError(f"{key!r} takes a list of {' or '.join(map(str, ns))} arguments, got {args!r}")
    return list(args)


def from_json(data: t.Any) -> Predicate:
    """Parse the compact JSON form."""
    if isinstance(data, _PRED_TYPES):
        return data  # type: ignore[return-value]
    if not isinstance(data, t.Mapping) or not data:
        raise PredicateError(f"predicate JSON must be a non-empty object, got {data!r}")
    keys = [k for k in data if k != "skip_null_items"]
    if len(keys) != 1:
        raise PredicateError(f"predicate JSON must have exactly one operator, got {sorted(data)}")
    key = keys[0]
    if key == "eq":
        c, v = _args(data, key, 2)
        return Eq(str(c), _value_from(v))
    if key == "in":
        c, vs = _args(data, key, 2)
        if not isinstance(vs, (list, tuple)):
            vs = [vs]
        return In(str(c), tuple(_value_from(v) for v in vs))
    if key in ("lt", "le", "gt", "ge", "ne"):
        c, v = _args(data, key, 2)
        return Cmp(str(c), key, _value_from(v))
    if key in ("lt_abs", "le_abs", "gt_abs", "ge_abs"):
        c, v = _args(data, key, 2)
        return CmpAbs(str(c), key, _value_from(v))
    if key == "range":
        args = _args(data, key, (3, 5))
        lo_inc, hi_inc = (args[3], args[4]) if len(args) == 5 else (True, True)
        return Range(str(args[0]), _value_from(args[1]), _value_from(args[2]), bool(lo_inc), bool(hi_inc))
    if key == "contains":
        c, v = _args(data, key, 2)
        return Contains(str(c), _value_from(v))
    if key in ("any", "all"):
        path, inner = _args(data, key, 2)
        cls = Any if key == "any" else All
        return cls(str(path), from_json(inner), bool(data.get("skip_null_items", False)))
    if key == "nonempty":
        return NonEmpty(str(data[key]))
    if key == "has_kind":
        c, k = _args(data, key, 2)
        return KindMatch(str(c), str(k))
    if key == "censored":
        tcol, ecol, op, v = _args(data, key, 4)
        return CensoredCmp(str(tcol), str(ecol), str(op), _value_from(v))
    if key == "text":
        args = _args(data, key, (2, 3))
        return TextMatch(str(args[0]), str(args[1]), str(args[2]) if len(args) == 3 else "exact")
    if key == "is_null":
        return IsNull(str(data[key]))
    if key == "not":
        return Not(from_json(data[key]))
    if key in ("and", "or"):
        parts = data[key]
        if not isinstance(parts, (list, tuple)):
            raise PredicateError(f"{key!r} takes a list of predicates")
        return (And if key == "and" else Or)(tuple(from_json(q) for q in parts))
    raise PredicateError(f"unknown predicate operator {key!r}")


_FACET_OPS = ("eq", "ne", "in", "lt", "le", "gt", "ge", "is_null", "nonempty", "contains")


def facet_predicate(data: t.Mapping[str, t.Any], column: str) -> Predicate:
    """Parse the implied-column form used by facets (``unknown_when: [{eq: -1}]``, ``[{column: unit,
    eq: ""}]``), flag bindings (``when_true: {in: [none_recorded]}``) and sentinel expectations
    (``{lt: -0.5}``). ``column`` is the column the facet sits on unless ``column:`` overrides it."""
    if not isinstance(data, t.Mapping):
        return Eq(column, data)                        # a bare value means equality
    col = str(data.get("column", column))
    ops = [k for k in data if k != "column"]
    if len(ops) != 1 or ops[0] not in _FACET_OPS:
        raise PredicateError(f"facet predicate needs exactly one of {_FACET_OPS}, got {sorted(data)}")
    op = ops[0]
    v = data[op]
    if op == "eq":
        return Eq(col, _value_from(v))
    if op == "in":
        return In(col, tuple(_value_from(x) for x in (v if isinstance(v, (list, tuple)) else [v])))
    if op == "is_null":
        return IsNull(col) if v in (True, None) else Not(IsNull(col))
    if op == "nonempty":
        return NonEmpty(col) if v in (True, None) else Not(NonEmpty(col))
    if op == "contains":
        return Contains(col, _value_from(v))
    return Cmp(col, op, _value_from(v))


def _container_form(p: Predicate) -> tuple[str | None, Predicate]:
    """``(container path, item-relative predicate)`` when ``p`` reads through a list, else ``(None, p)``."""
    if isinstance(p, (Any, All)):
        if isinstance(p, All):
            return None, p                             # All is not merged into an existential
        path = parse_path(p.path)
        split = path.split_container()
        if split is None:
            return None, p
        container, rest = split
        if not rest.segments:
            return format_path(container), p.pred
        return format_path(container), Any(format_path(rest), p.pred, p.skip_null_items)
    col = None
    if isinstance(p, (Eq, In, Cmp, CmpAbs, Range, TextMatch, KindMatch)):
        col = p.path if isinstance(p, KindMatch) else p.column
    elif isinstance(p, Contains):
        col = p.path
    if col is None or col == LIST:                     # the item itself (a re-rooted Contains) is a leaf
        return None, p
    path = parse_path(col)
    if path.absolute or path.up or path.axis:
        return None, p
    split = path.split_container()
    if split is None:
        return None, p
    container, rest = split
    inner_col = format_path(rest) if rest.segments else LIST
    if isinstance(p, Contains):
        return format_path(container), Eq(inner_col, p.value)
    if isinstance(p, KindMatch):
        return format_path(container), KindMatch(inner_col, p.id_type)
    return format_path(container), _with_column(p, inner_col)


def _with_column(p: Predicate, col: str) -> Predicate:
    if isinstance(p, Eq):
        return Eq(col, p.value)
    if isinstance(p, In):
        return In(col, p.values)
    if isinstance(p, Cmp):
        return Cmp(col, p.op, p.value)
    if isinstance(p, CmpAbs):
        return CmpAbs(col, p.op, p.value)
    if isinstance(p, Range):
        return Range(col, p.lo, p.hi, p.lo_inclusive, p.hi_inclusive)
    if isinstance(p, TextMatch):
        return TextMatch(col, p.text, p.mode)
    raise PredicateError(f"cannot re-root {p!r}")


def merge_container_predicates(preds: t.Iterable[Predicate]) -> list[Predicate]:
    """Merge conjuncts that read through the same container into one ``Any`` per container (C21),
    recursively for deeper containers. The result is a list of conjuncts in first-seen order."""
    out: list[t.Any] = []
    groups: "OrderedDict[str, list[Predicate]]" = OrderedDict()
    for p in preds:
        container, inner = _container_form(p)
        if container is None:
            out.append(p)
            continue
        if container not in groups:
            groups[container] = []
            out.append(container)                      # placeholder keeps the first-seen position
        groups[container].append(inner)
    result: list[Predicate] = []
    for item in out:
        if isinstance(item, str):
            inner_preds = merge_container_predicates(groups[item])
            body = inner_preds[0] if len(inner_preds) == 1 else And(tuple(inner_preds))
            result.append(Any(item, body))
        else:
            result.append(item)
    return result
