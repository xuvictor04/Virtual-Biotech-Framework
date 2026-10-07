"""``pvalue`` and ``pvalue_mantissa_exponent`` (§9.4).

``pvalue`` is a probability on ``[0, 1]`` where lower is stronger (ranked ascending). It also
serves as the phase-1 ``fallback`` of measures that declare ``statistic: fdr_bh`` (q-values order
the same way); the family-aware ``fdr_bh`` plugin arrives in phase 3. :func:`fdr_bh` is the basic
Benjamini-Hochberg adjustment over the values given (``m`` defaults to their count).

``pvalue_mantissa_exponent`` reads a p-value stored as two columns, ``spec.columns =
{mantissa, exponent}`` (GWAS ``pValueMantissa`` x 10^``pValueExponent``), so values below the
float64 range (1e-400) still order correctly: the rank value is ``log10(mantissa) + exponent``.
Thresholds compile without arithmetic on the normalised form (mantissa in ``[1, 10)``):
``p < m·10^e`` is ``exponent < e OR (exponent == e AND mantissa < m)``; a row whose mantissa is
not normalised (nor exactly 0), or is null, is never selected. A threshold may be a float, ``(mantissa,
exponent)`` or ``{mantissa, exponent}``.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, ClassVar, Mapping, Sequence

from ...predicate import And, Cmp, Eq, In, Or, Param, Predicate, Range, is_null
from ...roles import parse_path
from ..base import UnsupportedFilter
from ..registry import register
from . import StatisticBase, _conjoin, _operands, _pair, _sibling, is_number, normalize_op, spec_get

__all__ = ["PValueStatistic", "PValueMantissaExponentStatistic", "fdr_bh", "split_pvalue"]


def fdr_bh(pvalues: Sequence[float | None], m: int | None = None) -> list[float | None]:
    """Benjamini-Hochberg q-values in input order (unknown stays ``None``; ``m`` defaults to the
    number of known p-values and may be larger, e.g. a declared family with untested members)."""
    known = [(p, i) for i, p in enumerate(pvalues) if p is not None and not is_null(p)]
    total = m if m is not None else len(known)
    if total < len(known):
        raise ValueError(f"family size m={total} is smaller than the {len(known)} p-values given")
    out: list[float | None] = [None] * len(pvalues)
    running = 1.0
    for rank, (p, i) in sorted(((r, pi) for r, pi in enumerate(sorted(known), start=1)), reverse=True):
        running = min(running, float(p) * total / rank)
        out[i] = min(1.0, running)
    return out


def split_pvalue(value: Any) -> tuple[float, int]:
    """``(mantissa, exponent)`` with the mantissa normalised to ``[1, 10)`` (``0`` -> ``(0.0, 0)``)."""
    if isinstance(value, Mapping):
        return _normalise(float(value["mantissa"]), int(value["exponent"]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return _normalise(float(value[0]), int(value[1]))
    if not is_number(value):
        raise UnsupportedFilter(f"a p-value threshold must be a number or (mantissa, exponent), got {value!r}",
                                reason="invalid_value")
    if value == 0:
        return 0.0, 0
    d = Decimal(repr(float(value)))
    e = d.adjusted()
    return float(d.scaleb(-e)), int(e)


def _normalise(mantissa: float, exponent: int) -> tuple[float, int]:
    if mantissa == 0:
        return 0.0, 0
    d = Decimal(repr(mantissa))
    shift = d.adjusted()
    return float(d.scaleb(-shift)), int(exponent + shift)


@register
class PValueStatistic(StatisticBase):
    name: ClassVar[str] = "pvalue"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, 1.0)
    default_direction: ClassVar[str | None] = "lower_is_stronger"
    aggregations = frozenset({"min", "max", "median", "count"})

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("pvalue", {"role": "measure", "statistic": cls.name},
                          values=(0.05, 1e-8, 0.5, 1e-8, 1.0), best=1e-8,
                          thresholds=(("lt", 0.05), ("le", 1e-8), ("ge", 0.5)),
                          refused=(("lt", 5.0), ("le", -0.1)), confirmed={"confirmed": True, "scale": [0.0, 1.0]},
                          agg=("min", "max", "count"), group=("studyId", "GCST1", "GCST2")),
        )


@register
class PValueMantissaExponentStatistic(PValueStatistic):
    name: ClassVar[str] = "pvalue_mantissa_exponent"

    def parts(self, column: str, spec: Any) -> tuple[str, str]:
        cols = spec_get(spec, "columns", None) or {}
        if "mantissa" not in cols or "exponent" not in cols:
            raise UnsupportedFilter(f"{column}: pvalue_mantissa_exponent needs columns {{mantissa, exponent}}",
                                    reason="op", column=column)
        return _sibling(column, str(cols["mantissa"])), _sibling(column, str(cols["exponent"]))

    def value_of(self, row: Mapping[str, Any], column: str, spec: Any) -> Any:
        m_col, e_col = self.parts(column, spec)
        m, e = _get(row, m_col), _get(row, e_col)
        if is_null(m) or is_null(e) or not is_number(m) or not is_number(e):
            return None
        if self.is_unknown(m, spec, row, m_col) or float(m) < 0:
            return None
        return (float(m), int(e))

    def rank_value(self, value: Any, spec: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, tuple):
            m, e = value
            return -math.inf if m == 0 else math.log10(m) + e
        if is_number(value) and not is_null(value):
            return -math.inf if value == 0 else math.log10(float(value))
        return None

    def combine(self, known: list[Any], how: str, spec: Any) -> Any:
        ranked = sorted((r, v) for v in known if (r := self.rank_value(v, spec)) is not None)
        if not ranked:
            return None
        if how == "count":
            return len(ranked)
        if how == "median":
            return ranked[(len(ranked) - 1) // 2][1]
        return ranked[0][1] if how == "min" else ranked[-1][1]

    def check_values(self, column: str, op: str, values: Sequence[Any], spec: Any,
                     confirmed: Mapping[str, Any] | None) -> None:
        if op.endswith("_abs") or op == "significant":
            raise UnsupportedFilter(f"{op} is not defined for p-values", reason="op", column=column)
        for v in values:
            if isinstance(v, Param):
                raise UnsupportedFilter(f"{column}: a composite p-value threshold must be bound before compiling",
                                        reason="unbound_argument", column=column)
            m, e = split_pvalue(v)
            if m < 0 or (m > 0 and (e > 0 or (e == 0 and m > 1))):
                raise UnsupportedFilter(f"{op} {v} is outside the scale of {column} ([0, 1])", reason="scale",
                                        column=column, confirmed_range=[0.0, 1.0])
        if self.unverified(spec) and not self.confirmed_ok(confirmed):
            raise UnsupportedFilter(f"the composite p-value facts of {column} are not confirmed from the data",
                                    reason="scale", column=column)

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        m_col, e_col = self.parts(column, spec)
        if op == "in":
            vals = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
            return Or(tuple(self._cmp(m_col, e_col, "eq", v) for v in vals))
        if op == "range":
            lo, hi = _pair(value)
            parts = [self._cmp(m_col, e_col, "ge", lo)] if lo is not None else []
            parts += [self._cmp(m_col, e_col, "le", hi)] if hi is not None else []
            return And(tuple(parts)) if len(parts) > 1 else parts[0]
        return self._cmp(m_col, e_col, op, value)

    @staticmethod
    def _cmp(m_col: str, e_col: str, op: str, value: Any) -> Predicate:
        tm, te = split_pvalue(value)
        known_e = Range(e_col)                               # unknown for a null exponent, else true
        zero = Eq(m_col, 0.0)
        # only normalised mantissas (or an exact zero) are compared; others are never selected
        valid = Or((zero, Range(m_col, 1.0, 10.0, True, False)))
        if tm == 0:
            body: Predicate = {"lt": In(m_col, ()), "le": zero, "eq": zero, "gt": Cmp(m_col, ">", 0.0),
                               "ge": Range(m_col, 0.0, None), "ne": Cmp(m_col, ">", 0.0)}[op]
            return And((body, known_e, valid))
        strict = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}
        if op in strict:
            outer = "<" if op in ("lt", "le") else ">"
            body = Or((And((Cmp(e_col, outer, te), Cmp(m_col, ">", 0.0))),
                       And((Eq(e_col, te), Cmp(m_col, strict[op], tm)))))
            if op in ("lt", "le"):                          # p = 0 is below every positive threshold
                body = Or((body, And((zero, known_e))))
            return And((body, valid))
        if op == "eq":
            return And((Eq(e_col, te), Eq(m_col, tm), valid))
        if op == "ne":
            return And((Or((Cmp(e_col, "!=", te), Cmp(m_col, "!=", tm))), known_e, valid))
        raise UnsupportedFilter(f"{op} is not supported for composite p-values", reason="op")

    def predicate(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None,
                  fixed_scope: Mapping[str, Any]) -> Predicate:
        op = normalize_op(op)
        self.check_group(column, spec, fixed_scope)
        self.check_values(column, op, _operands(op, value), spec, confirmed)
        m_col, _ = self.parts(column, spec)
        return _conjoin([self.compare(column, op, value, spec, confirmed), *self.guards(m_col, spec)])

    def scale_text(self, spec: Any) -> str:
        cols = spec_get(spec, "columns", None) or {}
        return (f"scale [0, 1] stored as {cols.get('mantissa', 'mantissa')} x 10^{cols.get('exponent', 'exponent')} "
                f"(normalised mantissa in [1, 10))")

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        spec = {"role": "measure", "statistic": cls.name, "columns": {"mantissa": "mant", "exponent": "exp"}}
        return (
            StatisticCase("pvalue_mantissa_exponent", spec,
                          values=(5e-8, 3e-300, 0.04, 3e-300, 1.0), best=3e-300,
                          row=_composite_row,
                          unknown_rows=({"mant": None, "exp": -8}, {"mant": 5.0, "exp": None},
                                        {"mant": float("nan"), "exp": -3}),
                          thresholds=(("lt", 5e-8), ("le", 3e-300), ("ge", 0.04), ("gt", 5e-8), ("eq", 0.04),
                                      ("lt", (1.0, -400))),
                          refused=(("lt", 2.0), ("gt_abs", 0.1)), confirmed={"confirmed": True},
                          agg=("min", "count")),
        )


def _composite_row(value: Any) -> dict[str, Any]:
    if value is None or is_null(value):
        return {"mant": None, "exp": None}
    m, e = split_pvalue(value)
    return {"mant": m, "exp": e}


def _get(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for name in parse_path(path).strip_brackets().names():
        value = value.get(name) if isinstance(value, Mapping) else None
    return value

