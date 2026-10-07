"""``numeric`` and ``percent``: plain numbers with the declared direction and scale (§9.4).

``numeric`` is the default statistic of a measure. Direction comes from the spec
(``higher_is_stronger`` -> ``desc``, ``lower_is_stronger`` -> ``asc``, ``signed`` ->
``desc_abs``, so a signed effect ranks by magnitude); ``abs`` thresholds (``gt_abs``) compile
to :class:`~vbt.datalayer.predicate.CmpAbs`. ``percent`` is ``numeric`` on ``[0, 100]`` unless
the spec declares another scale.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..registry import register
from . import StatisticBase


@register
class NumericStatistic(StatisticBase):
    name: ClassVar[str] = "numeric"
    version: ClassVar[str] = "1.0"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("numeric_signed", {"role": "measure", "statistic": cls.name, "direction": "signed"},
                          values=(-3.0, 2.0, -2.0, 0.5, 0.0, 2.0), best=-3.0,
                          thresholds=(("gt", 1.0), ("le", -0.5), ("ge_abs", 2.0), ("eq", 2.0)),
                          group=("unit", "uM", "nM")),
            StatisticCase("numeric_codes", {"role": "measure", "statistic": cls.name,
                                            "direction": "higher_is_stronger", "missing_values": [-1, 999.9],
                                            "unknown_when": [{"column": "unit", "eq": ""}],
                                            "cutoff": {"value": 0.5, "op": "ge", "meaning": "expressed"}},
                          values=(0.5, 3.0, 0.0, 12.0), best=12.0, row=_with_unit,
                          unknown_rows=({"value": 4.0, "unit": ""}, {"value": 4.0, "unit": None}),
                          thresholds=(("ge", 0.5), ("lt", 3.0), ("ne", 3.0), ("in", [0.5, 3.0]))),
        )


@register
class PercentStatistic(NumericStatistic):
    name: ClassVar[str] = "percent"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, 100.0)
    default_direction: ClassVar[str | None] = "higher_is_stronger"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("percent", {"role": "measure", "statistic": cls.name},
                          values=(0.0, 12.5, 100.0, 50.0, 12.5), best=100.0,
                          thresholds=(("ge", 50.0), ("lt", 12.5), ("range", [10.0, 60.0])),
                          refused=(("gt", 150.0), ("lt", -1.0), ("range", [-5.0, 50.0])),
                          confirmed={"confirmed": True, "scale": [0.0, 100.0]}),
        )


def _with_unit(value: Any) -> dict[str, Any]:
    return {"value": value, "unit": "uM"}
