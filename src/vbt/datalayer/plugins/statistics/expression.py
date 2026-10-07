"""Expression levels (§9.4, phase 3): ``tpm`` (with a unit guard) and ``log2_intensity``.

* ``tpm``: a non-negative abundance where higher is stronger. Expression tables mix units (TPM,
  normalised counts, ``rna.level`` codes), named per row by the measure's ``unit_from`` column, so:
  values with different units are not :meth:`~StatisticBase.comparable`; a threshold needs the unit
  fixed (``unit_from`` acts like ``comparable_within``); aggregation over ``groups`` refuses rows
  whose unit differs **or is blank** (a blank unit cannot be pooled with anything, it is not "the
  same unit"). No in-band sentinel is assumed: a ``999.9`` placeholder must be declared in
  ``missing_values`` to be unknown, and then it never passes a filter nor ranks first.
* ``log2_intensity``: microarray log2 intensities, unbounded, comparable only within one cohort and
  platform. A spec without ``comparable_within`` gets ``[cohort]``: an intensity of 8 in one GEO
  series and 8 in another are not the same expression level.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

from ..base import AggResult, UnsupportedFilter
from ..registry import register
from . import StatisticBase, _lookup, _ref_name, _single, is_null, spec_get

__all__ = ["TpmStatistic", "Log2IntensityStatistic"]


def _blank(v: Any) -> bool:
    return is_null(v) or (isinstance(v, str) and not v.strip())


@register
class TpmStatistic(StatisticBase):
    name: ClassVar[str] = "tpm"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, None)
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    aggregations = frozenset({"mean", "median", "min", "max", "count"})

    def check_group(self, column: str, spec: Any, fixed_scope: Mapping[str, Any] | None) -> None:
        super().check_group(column, spec, fixed_scope)
        unit_from = spec_get(spec, "unit_from", None)
        if not unit_from:
            return
        fixed = {_ref_name(k): v for k, v in (fixed_scope or {}).items() if _single(v)}
        value = fixed.get(_ref_name(unit_from))
        if isinstance(value, (list, tuple)):
            value = value[0]
        if value is None or _blank(value):
            raise UnsupportedFilter(f"{column} mixes units ({unit_from}); fix one unit before filtering on its value",
                                    reason="group", column=column, group=[str(unit_from)])

    def comparable(self, a: Mapping[str, Any], b: Mapping[str, Any], spec: Any) -> bool:
        unit_from = spec_get(spec, "unit_from", None)
        if unit_from and (_blank(_lookup(a, unit_from)) or _blank(_lookup(b, unit_from))):
            return False
        return super().comparable(a, b, spec)

    def aggregate(self, values: Sequence[Any], how: str, spec: Any, keys: Sequence[Any] | None = None, *,
                  groups: Sequence[Mapping[str, Any]] | None = None) -> AggResult:
        unit_from = spec_get(spec, "unit_from", None)
        if unit_from and groups is not None:
            units = []
            for v, g in zip(values, groups):
                if self.is_unknown(v, spec):
                    continue
                u = _lookup(g, unit_from) if isinstance(g, Mapping) else None
                if _blank(u):
                    raise UnsupportedFilter(f"a {self.name} value without a unit ({unit_from} is blank) cannot be "
                                            "pooled", reason="group", group=[str(unit_from)])
                units.append(str(u))
            if len(set(units)) > 1:
                raise UnsupportedFilter(f"{self.name} values in {len(set(units))} units ({', '.join(sorted(set(units)))}) "
                                        "cannot be aggregated together", reason="group", group=[str(unit_from)])
        elif unit_from and groups is None and len([v for v in values if not self.is_unknown(v, spec)]) > 1:
            raise UnsupportedFilter(f"{self.name} aggregation needs each value's {unit_from} (pass groups)",
                                    reason="group", group=[str(unit_from)])
        return super().aggregate(values, how, spec, keys, groups=groups)

    def scale_text(self, spec: Any) -> str:
        unit = spec_get(spec, "unit", None) or (f"the unit named by {spec_get(spec, 'unit_from')}"
                                                 if spec_get(spec, "unit_from", None) else "TPM")
        return f"{super().scale_text(spec)}; abundance in {unit} (never compared across units)"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("tpm", {"role": "measure", "statistic": cls.name, "missing_values": [999.9]},
                          values=(0.0, 12.5, 3.2, 12.5), best=12.5, unknown=(None, float("nan"), 999.9),
                          thresholds=(("ge", 1.0), ("gt", 0.0), ("lt", 12.5)),
                          refused=(("ge", -1.0),), agg=("mean", "median", "max"),
                          group=("tissue", "liver", "lung"), fixed_scope={}),
        )


@register
class Log2IntensityStatistic(StatisticBase):
    name: ClassVar[str] = "log2_intensity"
    version: ClassVar[str] = "1.0"
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    default_within: ClassVar[tuple[str, ...]] = ("cohort",)
    aggregations = frozenset({"mean", "median", "min", "max", "count"})

    def within(self, spec: Any) -> list[str]:
        return [str(c) for c in (spec_get(spec, "comparable_within", None) or self.default_within)]

    def _with_within(self, spec: Any) -> Any:
        if spec_get(spec, "comparable_within", None):
            return spec
        if isinstance(spec, Mapping):
            return {**spec, "comparable_within": list(self.default_within)}
        return spec.model_copy(update={"comparable_within": list(self.default_within)})

    def check_group(self, column: str, spec: Any, fixed_scope: Mapping[str, Any] | None) -> None:
        super().check_group(column, self._with_within(spec), fixed_scope)

    def comparable(self, a: Mapping[str, Any], b: Mapping[str, Any], spec: Any) -> bool:
        return super().comparable(a, b, self._with_within(spec))

    def aggregate(self, values: Sequence[Any], how: str, spec: Any, keys: Sequence[Any] | None = None, *,
                  groups: Sequence[Mapping[str, Any]] | None = None) -> AggResult:
        return super().aggregate(values, how, self._with_within(spec), keys, groups=groups)

    def sort_key(self, column: str, spec: Any, direction: str | None) -> Any:
        return super().sort_key(column, self._with_within(spec), direction)

    def scale_text(self, spec: Any) -> str:
        return f"{super().scale_text(spec)}; log2 intensity (comparable only within {', '.join(self.within(spec))})"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("log2_intensity", {"role": "measure", "statistic": cls.name, "comparable_within": ["cohort"]},
                          values=(7.5, 12.1, 3.0, 7.5), best=12.1,
                          thresholds=(("ge", 7.5), ("lt", 3.0)), group=("cohort", "GSE12251", "GSE73661"),
                          fixed_scope={"cohort": "GSE12251"}, agg=("mean", "median")),
        )
