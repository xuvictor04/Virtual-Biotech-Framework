"""Builtin statistic plugins (phase 1, §9.4) and their shared semantics. Stdlib only.

Phase 1 registers ``numeric``, ``percent``, ``score_0_1``, ``score_0_100``, ``ordinal``,
``clinical_phase``, ``signed_factor``, ``binary_factor``, ``pvalue``,
``pvalue_mantissa_exponent``, ``llr_critval`` and ``count``. :class:`StatisticBase` holds what
every one of them shares:

* **unknown never passes (I6)**: null, NaN, the spec's ``missing_values`` (plus a plugin's
  default codes), ``unknown_when`` and ``undefined_when`` rows are unknown. ``predicate``
  returns the comparison AND one guard per code and condition, merged per container
  (:func:`~vbt.datalayer.predicate.merge_container_predicates`), so no op (``ne`` included)
  ever passes an unknown value; an ``unknown_when`` condition that is itself unknown (its column
  is null or absent) makes the value unknown too; ``missing: zero`` makes null (and NaN) mean 0:
  ``OR IsNull`` when 0 satisfies the op, ``AND NOT IsNull`` when it does not;
* **I9**: thresholds on a measure whose facts are ``verified: false`` need ``confirmed`` facts
  (the ``facts`` of :meth:`StatisticBase.validate`, which always carry ``"confirmed"``);
  thresholds outside the effective scale raise :class:`UnsupportedFilter` (``reason="scale"``);
* **comparable_within (rev 2)**: a value filter on a measure comparable only within some
  columns raises :class:`UnsupportedFilter` (``reason="group"``) unless ``fixed_scope`` fixes
  every one of them; :meth:`StatisticBase.aggregate` with ``groups=`` refuses to pool values
  from several groups;
* **ordering**: ``sort_key`` returns a :class:`RankKey` with nulls last; the caller breaks ties
  by the canonical key. :func:`order_rows` sorts rows with a plugin's ``value_of`` and
  ``rank_value`` (ordinal levels rank by position; composite p-values by log10), unknown
  values last whatever the direction;
* **aggregation**: empty or all-unknown input gives ``None`` (never 0); with level ``keys``
  each key counts once (I15).

``UnsupportedFilter.reason`` values used here: ``scale`` (outside the scale, or the scale is
not confirmed), ``unconfirmed_encoding``, ``group``, ``negated_form`` (factor labels accept only
positive forms), ``invalid_value`` (a label or level the spec does not declare), ``op`` (an op
the statistic does not support) and ``unbound_argument`` (a value the call must supply).
"""

from __future__ import annotations

import math
import statistics as _stats
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence

from ...predicate import (
    And,
    Cmp,
    CmpAbs,
    Eq,
    In,
    IsNull,
    Not,
    Or,
    Param,
    Predicate,
    Range,
    RankKey,
    evaluate,
    facet_predicate,
    is_null,
    merge_container_predicates,
)
from ...roles import parse_path
from ...rowkey import render_value
from ..base import AggResult, ColumnStats, ConfirmResult, FamilySpec, PluginBase, UnsupportedFilter, ValueSnapshot

__all__ = [
    "StatisticBase", "OPS", "THRESHOLD_OPS", "DIRECTION_ORDER", "normalize_op", "is_number", "same_value",
    "order_rows", "rank_value_of", "spec_get",
]

#: Accepted ``op`` spellings -> canonical op.
OPS: dict[str, str] = {
    "eq": "eq", "==": "eq", "=": "eq", "ne": "ne", "!=": "ne", "lt": "lt", "<": "lt", "le": "le", "<=": "le",
    "gt": "gt", ">": "gt", "ge": "ge", ">=": "ge", "in": "in", "range": "range", "between": "range",
    "lt_abs": "lt_abs", "le_abs": "le_abs", "gt_abs": "gt_abs", "ge_abs": "ge_abs",
    "significant": "significant",
}
THRESHOLD_OPS = frozenset({"lt", "le", "gt", "ge", "lt_abs", "le_abs", "gt_abs", "ge_abs", "range", "significant"})
_CMP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "ne": "!="}
_COMPLEMENT = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "!=": "==", "==": "!="}
_SYMBOL = {"lt": "<", "le": "≤", "gt": ">", "ge": "≥"}

#: Measure direction -> default RankKey direction.
DIRECTION_ORDER: dict[str, str | None] = {"higher_is_stronger": "desc", "lower_is_stronger": "asc",
                                          "signed": "desc_abs", "none": None}
_DIRECTION_TEXT = {"higher_is_stronger": "higher is stronger", "lower_is_stronger": "lower is stronger",
                   "signed": "signed (ranked by absolute value)", "none": "no direction"}


def normalize_op(op: str) -> str:
    norm = OPS.get(str(op).strip().lower())
    if norm is None:
        raise UnsupportedFilter(f"unknown filter operator {op!r} (known: {', '.join(sorted(set(OPS.values())))})",
                                reason="op")
    return norm


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_number(x: Any) -> float | None:
    """A float for numbers and numeric strings ("1", "-1.5"), else None."""
    if is_number(x):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


def same_value(a: Any, b: Any) -> bool:
    """Code equality across YAML and storage spellings: numbers compare as floats (``1`` == ``1.0`` ==
    ``"1"``), booleans only with booleans, everything else with ``==``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None and (is_number(a) or is_number(b)):
        return na == nb
    return bool(a == b)


def spec_get(spec: Any, name: str, default: Any = None) -> Any:
    """A facet of a column model or a plain mapping (count columns lack measure facets)."""
    if spec is None:
        return default
    if isinstance(spec, Mapping):
        value = spec.get(name, default)
    else:
        value = getattr(spec, name, default)
    return default if value is None else value


def _sibling(column: str, name: str) -> str:
    """Resolve a facet reference next to ``column`` (§6.4 scoping: sibling first; ``/x`` table level)."""
    if name.startswith("/"):
        return name[1:]
    up = 0
    while name.startswith("^."):
        up += 1
        name = name[2:]
    path = parse_path(column)
    if path.absolute or path.axis or len(path.segments) <= 1:
        return name
    text = column.rsplit(".", 1)[0] if "." in column else ""
    for _ in range(up):
        text = text.rsplit(".", 1)[0] if "." in text else ""
    return f"{text}.{name}" if text else name


def _negate(p: Predicate) -> Predicate:
    """The guard ``NOT p`` in a positive, mergeable form where one exists."""
    if isinstance(p, Eq):
        return Cmp(p.column, "!=", p.value)
    if isinstance(p, Cmp):
        return Cmp(p.column, _COMPLEMENT[p.op], p.value) if p.op != "!=" else Eq(p.column, p.value)
    if isinstance(p, Not):
        return p.pred
    return Not(p)


def _conjoin(parts: Sequence[Predicate]) -> Predicate:
    merged = merge_container_predicates(parts)
    return merged[0] if len(merged) == 1 else And(tuple(merged))


class StatisticBase(PluginBase):
    """Shared phase-1 semantics; subclasses set the class defaults and override the hooks."""

    kind: ClassVar[str] = "statistic"
    #: Scale when the spec declares none (``None``: unbounded on that side).
    default_scale: ClassVar[tuple[float | None, float | None] | None] = None
    default_direction: ClassVar[str | None] = None
    #: In-band unknown codes every use of this statistic shares (``clinical_phase``: -1).
    default_missing: ClassVar[tuple[Any, ...]] = ()
    #: Aggregations that make sense for this statistic.
    aggregations: ClassVar[frozenset[str]] = frozenset({"mean", "median", "min", "max", "sum", "count", "max_abs"})
    #: Thresholds need a confirmed scale even when the spec is verified (scales that differ by release).
    scale_needs_confirmation: ClassVar[bool] = False

    # -- facts -------------------------------------------------------------

    def missing_codes(self, spec: Any) -> tuple[Any, ...]:
        codes = list(self.default_missing)
        for c in spec_get(spec, "missing_values", []) or []:
            if not any(same_value(c, d) for d in codes):
                codes.append(c)
        return tuple(codes)

    def direction(self, spec: Any) -> str:
        return str(spec_get(spec, "direction", None) or self.default_direction or "none")

    def declared_scale(self, spec: Any) -> tuple[float | None, float | None] | None:
        scale = spec_get(spec, "scale", None)
        if scale:
            return float(scale[0]), float(scale[1])
        return self.default_scale

    def unverified(self, spec: Any) -> bool:
        return spec_get(spec, "verified", True) is False

    @staticmethod
    def confirmed_ok(confirmed: Mapping[str, Any] | None) -> bool:
        return bool(confirmed) and confirmed.get("confirmed") is True  # type: ignore[union-attr]

    def effective_scale(self, spec: Any, confirmed: Mapping[str, Any] | None
                        ) -> tuple[tuple[float | None, float | None] | None, bool]:
        """``(scale, known)``: the scale thresholds are checked against, and whether it may be used
        to filter (I9: a ``verified: false`` scale only after readiness confirmed it)."""
        declared = self.declared_scale(spec)
        if self.confirmed_ok(confirmed):
            assert confirmed is not None
            if confirmed.get("scale"):
                lo, hi = confirmed["scale"]
                return (lo, hi), True
            return declared, True
        if declared is None:
            return None, True
        if self.unverified(spec) or self.scale_needs_confirmation:
            return declared, False
        return declared, True

    def is_unknown(self, value: Any, spec: Any, row: Mapping[str, Any] | None = None,
                   column: str | None = None) -> bool:
        """Null, NaN, a missing code, or a row matching ``unknown_when``/``undefined_when``."""
        if is_null(value):
            return True
        if any(same_value(value, c) for c in self.missing_codes(spec)):
            return True
        if row is not None:
            return any(evaluate(p, row) is not False for p in self.conditions(column or "value", spec))
        return False

    def conditions(self, column: str, spec: Any) -> list[Predicate]:
        """The ``unknown_when`` and ``undefined_when`` conditions, their columns resolved next to ``column``."""
        out: list[Predicate] = []
        for cond in list(spec_get(spec, "unknown_when", []) or []) + list(spec_get(spec, "undefined_when", []) or []):
            target = column
            if isinstance(cond, Mapping) and cond.get("column") is not None:
                cond = dict(cond)
                target = _sibling(column, str(cond.pop("column")))
            out.append(facet_predicate(cond, target))
        return out

    def value_of(self, row: Mapping[str, Any], column: str, spec: Any) -> Any:
        """The measure value of ``row`` (top-level or dotted struct path), ``None`` when unknown;
        ``missing: zero`` maps null to 0."""
        value: Any = row
        for name in parse_path(column).strip_brackets().names():
            value = value.get(name) if isinstance(value, Mapping) else None
        if is_null(value) and spec_get(spec, "missing", None) == "zero":
            return 0                                       # NaN is null (I6), so it means 0 here too
        if self.is_unknown(value, spec, row, column):
            return None
        return value

    def rank_value(self, value: Any, spec: Any) -> float | None:
        """A float that orders values (``None``: unknown, sorted last)."""
        if value is None or is_null(value) or any(same_value(value, c) for c in self.missing_codes(spec)):
            return None
        n = _as_number(value) if not isinstance(value, bool) else float(value)
        return None if n is None or math.isnan(n) else n

    # -- protocol ----------------------------------------------------------

    def sort_key(self, column: str, spec: Any, direction: str | None) -> RankKey:
        if direction is None:
            direction = DIRECTION_ORDER.get(self.direction(spec)) or "desc"
        return RankKey(column=column, direction=direction, nulls="last", statistic=self.name,  # type: ignore[arg-type]
                       within=tuple(spec_get(spec, "comparable_within", []) or ()))

    def check_group(self, column: str, spec: Any, fixed_scope: Mapping[str, Any] | None) -> None:
        within = [str(g) for g in spec_get(spec, "comparable_within", []) or []]
        fixed = {_ref_name(k) for k, v in (fixed_scope or {}).items() if _single(v)}
        missing = [g for g in within if _ref_name(g) not in fixed]
        if missing:
            raise UnsupportedFilter(
                f"{column} is comparable only within {', '.join(within)}; fix {', '.join(missing)} before "
                f"filtering on its value", reason="group", column=column, group=missing)

    def check_scale(self, column: str, op: str, values: Sequence[Any], spec: Any,
                    confirmed: Mapping[str, Any] | None) -> None:
        scale, known = self.effective_scale(spec, confirmed)
        if scale is None:
            return
        if not known:
            raise UnsupportedFilter(
                f"the scale of {column} ({_fmt_scale(scale)}) is not confirmed from the data; a threshold "
                f"cannot be applied faithfully yet", reason="scale", column=column)
        lo, hi = scale
        for v in values:
            if isinstance(v, Param) or not is_number(v):
                continue
            x = abs(float(v)) if op.endswith("_abs") else float(v)
            if op.endswith("_abs"):
                bound = max(abs(lo) if lo is not None else math.inf, abs(hi) if hi is not None else math.inf)
                bad = x < 0 or x > bound
            else:
                bad = (lo is not None and x < lo) or (hi is not None and x > hi)
            if bad:
                raise UnsupportedFilter(f"{op} {v} is outside the scale of {column} ({_fmt_scale(scale)})",
                                        reason="scale", column=column, confirmed_range=[lo, hi])

    def guards(self, column: str, spec: Any) -> list[Predicate]:
        """Conjuncts that make in-band codes and unknown/undefined conditions fail (never pass)."""
        out: list[Predicate] = [Cmp(column, "!=", code) for code in self.missing_codes(spec)]
        return out + [_negate(p) for p in self.conditions(column, spec)]

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        """The bare comparison for a canonical op (hook: ordinal, factor and composite plugins override)."""
        if op == "eq":
            return Eq(column, value)
        if op == "ne":
            return Cmp(column, "!=", value)
        if op == "in":
            vals = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
            return In(column, tuple(vals))
        if op == "range":
            lo, hi = _pair(value)
            return Range(column, lo, hi)
        if op.endswith("_abs"):
            return CmpAbs(column, op, value)
        if op == "significant":
            raise UnsupportedFilter(f"{self.name} has no significance rule", reason="op", column=column)
        return Cmp(column, _CMP[op], value)

    def predicate(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None,
                  fixed_scope: Mapping[str, Any]) -> Predicate:
        op = normalize_op(op)
        self.check_group(column, spec, fixed_scope)
        self.check_values(column, op, _operands(op, value), spec, confirmed)
        main = self.compare(column, op, value, spec, confirmed)
        if spec_get(spec, "missing", None) == "zero":
            probe = _row_with(column, 0)
            if probe is not None and evaluate(main, probe) is True:
                main = Or((main, IsNull(column)))          # null means 0 here, and 0 satisfies the op
            elif probe is not None:
                main = And((main, Not(IsNull(column))))    # null means 0, which fails: false, not unknown
        return _conjoin([main, *self.guards(column, spec)])

    def check_values(self, column: str, op: str, values: Sequence[Any], spec: Any,
                     confirmed: Mapping[str, Any] | None) -> None:
        """Hook: thresholds against the effective scale (value ops only when a scale is declared)."""
        if op in THRESHOLD_OPS or self.declared_scale(spec) is not None:
            self.check_scale(column, op, values, spec, confirmed)

    def bounds(self, spec: Any, constraints: Sequence[Any]) -> dict[str, Any]:
        scale = self.declared_scale(spec)
        lo, hi = scale if scale else (None, None)
        out: dict[str, Any] = {}
        origins: list[str] = []
        for c in constraints or ():
            op = spec_get(c, "op", None)
            v = spec_get(c, "value", None)
            if op not in ("<", "<=", ">", ">=", "==") or not is_number(v):
                continue
            v = float(v)
            if op in ("<", "<=", "==") and (hi is None or v < hi):
                hi = v
                out["exclusiveMaximum" if op == "<" else "maximum"] = v
                out.pop("maximum" if op == "<" else "exclusiveMaximum", None)
                origins.append(str(spec_get(c, "origin", "")))
            if op in (">", ">=", "==") and (lo is None or v > lo):
                lo = v
                out["exclusiveMinimum" if op == ">" else "minimum"] = v
                out.pop("minimum" if op == ">" else "exclusiveMinimum", None)
                origins.append(str(spec_get(c, "origin", "")))
        if lo is not None and "minimum" not in out and "exclusiveMinimum" not in out:
            out["minimum"] = lo
        if hi is not None and "maximum" not in out and "exclusiveMaximum" not in out:
            out["maximum"] = hi
        if origins:
            out["constraints"] = [o for o in origins if o]
        return out

    def observed(self, stats: ColumnStats | None, snapshot: ValueSnapshot | None, spec: Any
                 ) -> tuple[list[Any] | None, float | None, float | None, int]:
        """``(known distinct values or None, min, max, nan_count)`` with codes, nulls and NaN excluded."""
        nan = int(snapshot.nan_count) if snapshot is not None else 0
        if snapshot is not None and snapshot.complete:
            known = []
            for v in snapshot.values:
                if isinstance(v, float) and math.isnan(v):
                    nan += 1
                    continue
                if v is None or any(same_value(v, c) for c in self.missing_codes(spec)):
                    continue
                known.append(v)
            nums = [float(v) for v in known if is_number(v)]
            return known, (min(nums) if nums else None), (max(nums) if nums else None), nan
        mn = stats.min if stats is not None else None
        mx = stats.max if stats is not None else None
        codes = self.missing_codes(spec)
        if any(same_value(mn, c) for c in codes) or any(same_value(mx, c) for c in codes):
            return None, None, None, nan                 # an in-band code hides the real range
        return None, (float(mn) if is_number(mn) else None), (float(mx) if is_number(mx) else None), nan

    def validate(self, stats: ColumnStats, snapshot: ValueSnapshot | None, spec: Any) -> ConfirmResult:
        known, mn, mx, nan = self.observed(stats, snapshot, spec)
        facts: dict[str, Any] = {"min": mn, "max": mx, "nan_count": nan}
        problems: list[str] = []
        undecided: list[str] = []
        self.check_facts(stats, snapshot, spec, known, mn, mx, facts, problems, undecided)
        confirmed = False if problems else (None if undecided else True)
        facts["confirmed"] = confirmed
        return ConfirmResult(confirmed, facts, "; ".join(problems + undecided))

    def check_facts(self, stats: ColumnStats | None, snapshot: ValueSnapshot | None, spec: Any,
                    known: list[Any] | None, mn: float | None, mx: float | None, facts: dict[str, Any],
                    problems: list[str], undecided: list[str]) -> None:
        """Hook: confirm the scale from min/max and encodings from the snapshot."""
        scale = self.declared_scale(spec)
        if scale is not None:
            lo, hi = scale
            if mn is None or mx is None:
                undecided.append("no min/max to confirm the scale")
            elif (lo is not None and mn < lo) or (hi is not None and mx > hi):
                problems.append(f"observed range [{_num(mn)}, {_num(mx)}] is outside the scale {_fmt_scale(scale)}")
            else:
                facts["scale"] = [lo, hi]
        encoding = spec_get(spec, "encoding", None)
        if encoding:
            confirm_codes(list(encoding), known, snapshot, self.missing_codes(spec), facts, problems, undecided)

    def unknown_filter(self, values: Iterable[Any], spec: Any, keys: Sequence[Any] | None,
                       groups: Sequence[Mapping[str, Any]] | None) -> tuple[list[Any], int, list[Any], str]:
        """``(known values, n_excluded, their groups, detail)``; with level ``keys`` each key counts
        once (its first known value; a key with only unknown values is excluded once)."""
        zero = spec_get(spec, "missing", None) == "zero"
        slots: dict[Any, list[Any]] = {}                   # key -> [value, group, known]
        conflicts = 0
        for i, v in enumerate(values):
            if zero and is_null(v):
                v = 0
            k = render_value(keys[i]) if keys is not None else i
            slot = slots.setdefault(k, [None, None, False])
            if self.is_unknown(v, spec):
                continue
            if slot[2]:
                conflicts += not same_value(slot[0], v)
                continue
            slot[:] = [v, groups[i] if groups is not None else None, True]
        known = [s[0] for s in slots.values() if s[2]]
        known_groups = [s[1] for s in slots.values() if s[2]]
        excluded = sum(1 for s in slots.values() if not s[2])
        detail = f"{conflicts} level key(s) with conflicting values; first kept" if conflicts else ""
        return known, excluded, known_groups, detail

    def aggregate(self, values: Sequence[Any], how: str, spec: Any, keys: Sequence[Any] | None = None, *,
                  groups: Sequence[Mapping[str, Any]] | None = None) -> AggResult:
        how = str(how).lower()
        if how not in self.aggregations:
            raise UnsupportedFilter(f"aggregation {how!r} is not meaningful for {self.name} "
                                    f"(supported: {', '.join(sorted(self.aggregations))})", reason="op")
        if keys is not None and len(keys) != len(values):
            raise ValueError(f"{len(values)} values but {len(keys)} level keys")
        known, excluded, known_groups, detail = self.unknown_filter(values, spec, keys, groups)
        within = list(spec_get(spec, "comparable_within", []) or [])
        if groups is not None and within:
            distinct = {tuple(str(g.get(c)) if isinstance(g, Mapping) else str(g) for c in within)
                        for g in known_groups}
            if len(distinct) > 1:
                raise UnsupportedFilter(f"{self.name} values are comparable only within {', '.join(within)}; "
                                        f"{len(distinct)} groups cannot be aggregated together", reason="group",
                                        group=within)
        if not known:
            return AggResult(None, 0, excluded, detail or "no known values")
        return AggResult(self.combine(known, how, spec), len(known), excluded, detail)

    def combine(self, known: list[Any], how: str, spec: Any) -> Any:
        nums = [self.rank_value(v, spec) for v in known]
        nums = [n for n in nums if n is not None]
        if not nums:
            return None
        if how == "count":
            return len(nums)
        if how == "sum":
            return math.fsum(nums)
        if how == "mean":
            return math.fsum(nums) / len(nums)
        if how == "median":
            return _stats.median(nums)
        if how == "min":
            return min(nums)
        if how == "max":
            return max(nums)
        if how == "max_abs":
            return max(nums, key=abs)
        raise UnsupportedFilter(f"aggregation {how!r} is not supported by {self.name}", reason="op")

    def comparable(self, a: Mapping[str, Any], b: Mapping[str, Any], spec: Any) -> bool:
        for col in spec_get(spec, "comparable_within", []) or []:
            name = str(col).lstrip("/^.")
            va, vb = _lookup(a, name), _lookup(b, name)
            if is_null(va) or is_null(vb) or not same_value(va, vb):
                return False
        unit_from = spec_get(spec, "unit_from", None)
        if unit_from:
            ua, ub = _lookup(a, unit_from), _lookup(b, unit_from)
            if is_null(ua) or is_null(ub) or ua != ub:
                return False
        return True

    def family(self, spec: Any) -> FamilySpec | None:
        fam = spec_get(spec, "family", None)
        if not fam:
            return None
        return FamilySpec(columns=tuple(str(c) for c in fam), method=None,
                          origin=spec_get(spec, "verified_by", None))

    # -- text --------------------------------------------------------------

    def scale_text(self, spec: Any) -> str:
        scale = self.declared_scale(spec)
        text = f"scale {_fmt_scale(scale)}" if scale is not None else "scale unbounded"
        if self.unverified(spec) or self.scale_needs_confirmation:
            text += " (not yet confirmed from the data)"
        return text

    def describe(self, spec: Any) -> str:
        parts = [f"{self.name} measure" + (f" in {spec_get(spec, 'unit')}" if spec_get(spec, "unit") else "")]
        parts.append(_DIRECTION_TEXT.get(self.direction(spec), self.direction(spec)))
        parts.append(self.scale_text(spec))
        cutoff = spec_get(spec, "cutoff", None)
        if cutoff is not None:
            cop, cval = spec_get(cutoff, "op"), spec_get(cutoff, "value")
            inclusive = "inclusive" if cop in ("le", "ge") else "exclusive"
            meaning = spec_get(cutoff, "meaning", None)
            parts.append(f"cutoff {_SYMBOL.get(cop, cop)} {_num(cval)} ({inclusive})" + (f": {meaning}" if meaning
                                                                                         else ""))
        codes = self.missing_codes(spec)
        unknown = ["null", "NaN"] + [repr(c) for c in codes]
        if spec_get(spec, "unknown_when", None) or spec_get(spec, "undefined_when", None):
            unknown.append("unknown_when rows")
        parts.append(f"unknown values ({', '.join(unknown)}) never pass a filter")
        within = spec_get(spec, "comparable_within", None)
        if within:
            parts.append(f"comparable only within {', '.join(map(str, within))}")
        fam = spec_get(spec, "family", None)
        if fam:
            parts.append(f"multiple-testing family: {', '.join(map(str, fam))}")
        return "; ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def confirm_codes(codes: Sequence[Any], known: list[Any] | None, snapshot: ValueSnapshot | None,
                  missing: Sequence[Any], facts: dict[str, Any], problems: list[str], undecided: list[str]) -> None:
    """Encodings are confirmed only from a complete distinct-value snapshot: every declared code is
    observed and the observed values are a subset of the codes (plus null, NaN and missing codes)."""
    if snapshot is None or not snapshot.complete or known is None:
        undecided.append("encodings are confirmed only from a complete distinct-value snapshot")
        return
    never = [c for c in codes if not any(same_value(c, v) for v in known)
             and not any(same_value(c, m) for m in missing)]
    extra = [v for v in known if not any(same_value(v, c) for c in codes)]
    if never:
        problems.append(f"declared code(s) {', '.join(map(repr, never))} never observed")
    if extra:
        problems.append(f"observed value(s) {', '.join(map(repr, extra[:10]))} are not declared codes")
    facts["codes"] = list(known)
    facts["null_count"] = int(snapshot.null_count)


def _pair(value: Any) -> tuple[Any, Any]:
    if isinstance(value, Mapping):
        return value.get("lo", value.get("min")), value.get("hi", value.get("max"))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]
    raise UnsupportedFilter(f"a range needs [lo, hi], got {value!r}", reason="invalid_value")


def _operands(op: str, value: Any) -> list[Any]:
    if op == "range":
        return [v for v in _pair(value) if v is not None]
    if op == "in":
        return list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return [value]


def _ref_name(ref: Any) -> str:
    """The column name a scope reference names (``^.unit``, ``/unit`` and ``a.unit`` -> ``unit``)."""
    return str(ref).lstrip("/^.").split(".")[-1]


def _single(value: Any) -> bool:
    """A scope value that fixes one group (a list of several values does not)."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) == 1
    return value is not None


def _row_with(column: str, value: Any) -> dict[str, Any] | None:
    """A row holding ``value`` at a top-level or struct path (None for paths through lists)."""
    path = parse_path(column)
    if path.crosses_list or path.axis or path.up:
        return None
    row: Any = value
    for name in reversed(path.names()):
        row = {name: row}
    return row


def _lookup(row: Mapping[str, Any], name: str) -> Any:
    value: Any = row
    for part in str(name).split("."):
        value = value.get(part) if isinstance(value, Mapping) else None
    return value


def _num(x: Any) -> str:
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return repr(x) if isinstance(x, float) else str(x)


def _fmt_scale(scale: tuple[float | None, float | None] | Sequence[Any] | None) -> str:
    if scale is None:
        return "unbounded"
    lo, hi = scale
    return f"[{'-inf' if lo is None else _num(lo)}, {'inf' if hi is None else _num(hi)}]"


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def rank_value_of(plugin: Any, row: Mapping[str, Any], column: str, spec: Any) -> float | None:
    """The ordering value of ``row`` under ``plugin`` (falls back to the plain number for plugins
    without ``value_of``/``rank_value``)."""
    value_of = getattr(plugin, "value_of", None)
    value = value_of(row, column, spec) if callable(value_of) else _lookup(row, column)
    rank = getattr(plugin, "rank_value", None)
    if callable(rank):
        return rank(value, spec)
    if not is_number(value) or is_null(value):
        return None
    return float(value)


def order_rows(rows: Iterable[Mapping[str, Any]], keys: Sequence[tuple[RankKey, Any, Any]], *,
               tie_key: Callable[[Mapping[str, Any]], Any] | None = None) -> list[Mapping[str, Any]]:
    """Sort rows by ``keys`` (``(rank_key, plugin, spec)`` each): unknown values last whatever the
    direction (``nulls="first"`` puts them first), then by ``tie_key`` (the canonical row key)."""

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        parts: list[Any] = []
        for rk, plugin, spec in keys:
            v = rank_value_of(plugin, row, rk.column, spec)
            if v is None:
                parts.append((1 if rk.nulls == "last" else -1, 0.0))
                continue
            if rk.direction.endswith("_abs"):
                v = abs(v)
            if rk.direction.startswith("desc"):
                v = -v
            parts.append((0, v + 0.0))
        parts.append(tie_key(row) if tie_key is not None else "")
        return tuple(parts)

    return sorted(rows, key=key)
