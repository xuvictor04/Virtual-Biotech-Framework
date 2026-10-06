"""``score_0_1`` and ``score_0_100``: bounded scores, higher is stronger (§9.4).

Open Targets association and evidence scores live on ``[0, 1]``; a threshold of ``50`` on such
a column is a unit mistake and raises :class:`~vbt.datalayer.plugins.base.UnsupportedFilter`
(``reason="scale"``) instead of returning an empty success.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..registry import register
from . import StatisticBase


@register
class Score01Statistic(StatisticBase):
    name: ClassVar[str] = "score_0_1"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, 1.0)
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    aggregations = frozenset({"mean", "median", "min", "max", "count"})

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        hi = cls.default_scale[1] if cls.default_scale else 1.0
        return (
            StatisticCase(cls.name, {"role": "measure", "statistic": cls.name},
                          values=(0.1 * hi, 0.87 * hi, 0.5 * hi, 0.87 * hi, 0.0), best=0.87 * hi,
                          thresholds=(("ge", 0.5 * hi), ("gt", 0.1 * hi), ("le", 0.87 * hi), ("lt", 0.5 * hi)),
                          refused=(("gt", 50.0 * hi), ("ge", -0.5 * hi), ("in", [2.0 * hi])),
                          unconfirmed=(("ge", 0.5 * hi),),
                          confirmed={"confirmed": True, "scale": [0.0, hi]},
                          group=("datatypeId", "literature", "genetic_association")),
        )


@register
class Score100Statistic(Score01Statistic):
    name: ClassVar[str] = "score_0_100"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, 100.0)
