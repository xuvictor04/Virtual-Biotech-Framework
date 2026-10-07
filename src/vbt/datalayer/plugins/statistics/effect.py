"""Effect-size statistics (§9.4, phase 3): ``log2fc``, ``std_error``, ``wald``, ``effect_beta`` and ``odds_ratio``.

* ``log2fc``: a signed log2 fold change, ranked by magnitude (``desc_abs``); ``gt_abs``-style
  thresholds compile to :class:`~vbt.datalayer.predicate.CmpAbs` (upstream's strict ``>`` is kept by
  binding ``gt_abs``). Fold changes are comparable only within one contrast, so descriptors declare
  ``comparable_within`` (drug, cell line, dose, plate for Tahoe).
* ``std_error``: the standard error of an estimate, on ``[0, inf)``; it has no direction of
  strength (a small error is precise, not strong), so it ranks ascending only when asked.
* ``wald``: a signed z statistic (estimate / standard error), ranked by magnitude;
  ``op="significant"`` is ``|z| >= z_crit`` with ``z_crit`` from the call, ``significant_above`` or
  the two-sided 5% critical value 1.959964 (inclusive).
* ``effect_beta``: a signed regression effect (GWAS beta), ranked by magnitude; aggregations stay on
  the signed values (``mean``, ``median``, ``max_abs``), never on absolute values.
* ``odds_ratio``: a positive ratio on ``[0, inf)`` where 1 is no effect; it ranks by ``|log OR|``
  (strongest deviation from 1 first), thresholds compare the ratio itself, ``*_abs`` thresholds are
  refused (``|OR|`` is meaningless), and the mean is the geometric mean.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Mapping, Sequence

from ...predicate import CmpAbs, Predicate
from ..base import UnsupportedFilter
from ..registry import register
from . import StatisticBase, _num, is_number, spec_get

__all__ = ["Log2FcStatistic", "StdErrorStatistic", "WaldStatistic", "EffectBetaStatistic", "OddsRatioStatistic",
           "Z_CRIT_95"]

#: Two-sided 5% critical value of the standard normal.
Z_CRIT_95 = 1.959963984540054


class _SignedEffect(StatisticBase):
    """A signed effect: ranked by magnitude, aggregated on its signed values."""

    default_direction: ClassVar[str | None] = "signed"
    aggregations = frozenset({"mean", "median", "min", "max", "count", "max_abs"})
    unit_word: ClassVar[str] = "effect"

    def scale_text(self, spec: Any) -> str:
        return f"{super().scale_text(spec)}; signed {self.unit_word} (0 = no effect)"


@register
class Log2FcStatistic(_SignedEffect):
    name: ClassVar[str] = "log2fc"
    version: ClassVar[str] = "1.0"
    unit_word: ClassVar[str] = "log2 fold change"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("log2fc", {"role": "measure", "statistic": cls.name, "comparable_within": ["concentration"]},
                          values=(2.0, -3.5, 0.4, 2.0, 0.0), best=-3.5,
                          thresholds=(("gt_abs", 1.0), ("ge", 0.4), ("lt", 0.0), ("le_abs", 0.4)),
                          group=("concentration", 0.05, 5.0), agg=("mean", "max_abs"),
                          fixed_scope={"concentration": 0.05}),
        )


@register
class StdErrorStatistic(StatisticBase):
    name: ClassVar[str] = "std_error"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, None)
    default_direction: ClassVar[str | None] = "none"
    aggregations = frozenset({"median", "min", "max", "count"})

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        if op.endswith("_abs"):
            raise UnsupportedFilter(f"{op} is not defined for a standard error (it is never negative)", reason="op",
                                    column=column)
        return super().compare(column, op, value, spec, confirmed)

    def scale_text(self, spec: Any) -> str:
        return f"{super().scale_text(spec)}; standard error of an estimate (smaller is more precise)"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("std_error", {"role": "measure", "statistic": cls.name},
                          values=(0.2, 0.05, 1.3, 0.2),
                          thresholds=(("le", 0.2), ("lt", 1.3), ("ge", 0.05)),
                          refused=(("lt", -0.1), ("gt_abs", 0.1)), agg=("median", "count")),
        )


@register
class WaldStatistic(_SignedEffect):
    name: ClassVar[str] = "wald"
    version: ClassVar[str] = "1.0"
    unit_word: ClassVar[str] = "Wald z statistic"

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        if op == "significant":
            return CmpAbs(column, "ge_abs", self.critical(value, spec))
        return super().compare(column, op, value, spec, confirmed)

    @staticmethod
    def critical(value: Any, spec: Any) -> float:
        if is_number(value) and not isinstance(value, bool):
            if value <= 0:
                raise UnsupportedFilter(f"a critical |z| must be positive, got {value!r}", reason="invalid_value")
            return float(value)
        return float(spec_get(spec, "significant_above", None) or Z_CRIT_95)

    def check_values(self, column: str, op: str, values: Sequence[Any], spec: Any,
                     confirmed: Mapping[str, Any] | None) -> None:
        if op == "significant":
            return
        super().check_values(column, op, values, spec, confirmed)

    def scale_text(self, spec: Any) -> str:
        crit = spec_get(spec, "significant_above", None) or Z_CRIT_95
        return f"{super().scale_text(spec)}; significant when |z| ≥ {_num(round(float(crit), 6))} (inclusive)"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("wald", {"role": "measure", "statistic": cls.name},
                          values=(10.0, -12.5, 1.2, 0.0, 10.0), best=-12.5,
                          thresholds=(("significant", None), ("ge_abs", 1.96), ("lt", 0.0), ("gt", 1.2)),
                          agg=("median", "max_abs")),
        )


@register
class EffectBetaStatistic(_SignedEffect):
    name: ClassVar[str] = "effect_beta"
    version: ClassVar[str] = "1.0"
    unit_word: ClassVar[str] = "regression effect (beta)"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("effect_beta", {"role": "measure", "statistic": cls.name, "comparable_within": ["studyId"]},
                          values=(0.12, -0.31, 0.05, 0.12), best=-0.31,
                          thresholds=(("gt_abs", 0.1), ("lt", 0.0), ("ge", 0.05)),
                          group=("studyId", "GCST1", "GCST2"), fixed_scope={"studyId": "GCST1"},
                          agg=("mean", "median", "max_abs")),
        )


@register
class OddsRatioStatistic(StatisticBase):
    name: ClassVar[str] = "odds_ratio"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, None)
    default_direction: ClassVar[str | None] = "signed"
    aggregations = frozenset({"mean", "median", "min", "max", "count"})

    def rank_value(self, value: Any, spec: Any) -> float | None:
        n = super().rank_value(value, spec)
        if n is None or n < 0:
            return None
        return -math.inf if n == 0 else math.log(n)

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        if op.endswith("_abs"):
            raise UnsupportedFilter(f"{op} is not defined for an odds ratio (compare the ratio, 1 = no effect)",
                                    reason="op", column=column)
        return super().compare(column, op, value, spec, confirmed)

    def combine(self, known: list[Any], how: str, spec: Any) -> Any:
        nums = [float(v) for v in known if is_number(v) and float(v) >= 0]
        if not nums:
            return None
        if how == "mean":                              # geometric mean: the mean of log odds, back-transformed
            if any(v == 0 for v in nums):
                return 0.0
            return math.exp(math.fsum(math.log(v) for v in nums) / len(nums))
        if how == "median":
            s = sorted(nums)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 else math.sqrt(s[mid - 1] * s[mid])
        if how == "count":
            return len(nums)
        return min(nums) if how == "min" else max(nums)

    def scale_text(self, spec: Any) -> str:
        return f"{super().scale_text(spec)}; odds ratio (1 = no effect; ranked by |log OR|; mean is geometric)"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("odds_ratio", {"role": "measure", "statistic": cls.name},
                          values=(1.5, 0.2, 1.0, 3.0, 1.5), best=0.2,
                          thresholds=(("gt", 1.0), ("le", 0.2), ("ge", 1.5)),
                          refused=(("lt", -1.0), ("gt_abs", 2.0)), agg=("mean", "median", "count")),
        )
