"""``survival_time``: right-censored times with their event flags (§9.4, phase 3; capability ``paired``).

A survival time is only half a fact: a patient alive at 30 months (censored) did not "survive 30
months" in the sense an event at 30 months did. So:

* thresholds compile to :class:`~vbt.datalayer.predicate.CensoredCmp` over the time and the event
  column (``columns: {event: <flag column>}``, resolved next to the time): ``time <= t`` is true only
  when the event occurred, and unknown for a case censored before ``t`` (I6, counted in
  ``excluded_unknown``); without a declared event column a threshold raises ``UnsupportedFilter``;
* :meth:`SurvivalTimeStatistic.aggregate` refuses ``mean``/``sum`` (a mean of censored times
  understates survival) and allows ``count``, ``min`` and ``max`` of the observed times only;
* :meth:`SurvivalTimeStatistic.aggregate_pair` (capability ``paired``) takes times and events
  together: ``median`` is the Kaplan-Meier median (the first time the survival estimate drops to
  0.5 or below; ``None`` when it never does), ``events`` counts events, ``n`` counts cases and
  ``km`` returns the survival curve ``[[t, S(t)], ...]``; ``mean`` raises. With level ``keys`` each
  key (patient) counts once (I15: patient survival copied onto every sample row is one patient).
  A case with an unknown time or event is excluded and counted.

:func:`kaplan_meier` is the product-limit estimator (events before censorings at tied times).
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

from ...predicate import CensoredCmp, Predicate
from ...rowkey import render_value
from ..base import AggResult, UnsupportedFilter
from ..registry import register
from . import StatisticBase, _sibling, is_null, is_number, spec_get

__all__ = ["SurvivalTimeStatistic", "kaplan_meier", "km_median"]

_SYMBOL = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def kaplan_meier(times: Sequence[float], events: Sequence[bool]) -> list[tuple[float, float]]:
    """``[(t, S(t))]`` at each distinct event time (product-limit; censored cases leave the risk set
    after the events at the same time)."""
    pairs = sorted(zip((float(t) for t in times), (bool(e) for e in events)))
    at_risk = len(pairs)
    s = 1.0
    out: list[tuple[float, float]] = []
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        d = c = 0
        while i < len(pairs) and pairs[i][0] == t:
            d += pairs[i][1]
            c += not pairs[i][1]
            i += 1
        if d:
            s *= 1.0 - d / at_risk
            out.append((t, s))
        at_risk -= d + c
    return out


def km_median(times: Sequence[float], events: Sequence[bool]) -> float | None:
    """The Kaplan-Meier median: the first event time with ``S(t) <= 0.5`` (``None`` when never reached)."""
    for t, s in kaplan_meier(times, events):
        if s <= 0.5 + 1e-12:
            return t
    return None


def _event(v: Any) -> bool | None:
    if is_null(v):
        return None
    if isinstance(v, bool):
        return v
    if is_number(v):
        return bool(v)
    text = str(v).strip().lower()
    if text in ("1", "true", "yes", "dead", "deceased", "event", "1:deceased", "progressed", "1:progression",
                "1:recurred/progressed"):
        return True
    if text in ("0", "false", "no", "alive", "living", "censored", "0:living", "0:censored", "0:diseasefree",
                "0:progressionfree"):
        return False
    return None


@register
class SurvivalTimeStatistic(StatisticBase):
    name: ClassVar[str] = "survival_time"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"paired"})
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, None)
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    aggregations = frozenset({"count", "min", "max"})
    pair_aggregations: ClassVar[frozenset[str]] = frozenset({"median", "events", "n", "km"})

    def event_column(self, column: str, spec: Any) -> str | None:
        ev = (spec_get(spec, "columns", None) or {}).get("event")
        return _sibling(column, str(ev)) if ev else None

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        event = self.event_column(column, spec)
        if op not in _SYMBOL:
            raise UnsupportedFilter(f"{op} is not defined for a censored time (use lt, le, gt or ge)", reason="op",
                                    column=column)
        if event is None:
            raise UnsupportedFilter(f"{column}: a threshold on a censored time needs its event column "
                                    "(columns: {event: ...})", reason="unbound_argument", column=column)
        return CensoredCmp(column, event, _SYMBOL[op], value)

    def aggregate(self, values: Sequence[Any], how: str, spec: Any, keys: Sequence[Any] | None = None, *,
                  groups: Sequence[Mapping[str, Any]] | None = None) -> AggResult:
        if str(how).lower() in ("mean", "sum", "median"):
            raise UnsupportedFilter(f"{how} of censored times without their events is biased; use aggregate_pair "
                                    "(Kaplan-Meier)", reason="censored")
        return super().aggregate(values, how, spec, keys, groups=groups)

    def aggregate_pair(self, times: Sequence[Any], events: Sequence[Any], how: str, spec: Any = None,
                       keys: Sequence[Any] | None = None) -> AggResult:
        how = str(how).lower()
        if how in ("mean", "sum"):
            raise UnsupportedFilter(f"{how} of censored survival times is biased (censored cases are not deaths); "
                                    "use the Kaplan-Meier median", reason="censored")
        if how not in self.pair_aggregations:
            raise UnsupportedFilter(f"aggregation {how!r} is not defined for paired survival data "
                                    f"(supported: {', '.join(sorted(self.pair_aggregations))})", reason="op")
        if len(times) != len(events) or (keys is not None and len(keys) != len(times)):
            raise ValueError("times, events and keys must have the same length")
        slots: dict[Any, tuple[float, bool] | None] = {}
        conflicts = 0
        for i, (t, e) in enumerate(zip(times, events)):
            k = render_value(keys[i]) if keys is not None else i
            ev = _event(e)
            known = None if (is_null(t) or not is_number(t) or self.is_unknown(t, spec) or ev is None or float(t) < 0) \
                else (float(t), ev)
            if k in slots and slots[k] is not None:
                conflicts += known is not None and known != slots[k]
                continue
            slots[k] = known
        cases = [v for v in slots.values() if v is not None]
        excluded = sum(1 for v in slots.values() if v is None)
        detail = f"{conflicts} level key(s) with conflicting values; first kept" if conflicts else ""
        if not cases:
            return AggResult(None, 0, excluded, detail or "no known cases")
        ts = [c[0] for c in cases]
        es = [c[1] for c in cases]
        if how == "n":
            value: Any = len(cases)
        elif how == "events":
            value = sum(es)
        elif how == "km":
            value = [[t, s] for t, s in kaplan_meier(ts, es)]
        else:
            value = km_median(ts, es)
            if value is None:
                detail = (detail + "; " if detail else "") + "the survival estimate never reaches 0.5 (median not reached)"
        return AggResult(value, len(cases), excluded, detail)

    def scale_text(self, spec: Any) -> str:
        unit = spec_get(spec, "unit", None)
        return (f"{super().scale_text(spec)}; right-censored time{f' in {unit}' if unit else ''} "
                "(thresholds need the event; medians are Kaplan-Meier; means are refused)")

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase
        from ..conformance.statistic import PairedCase

        spec = {"role": "measure", "statistic": cls.name, "unit": "months", "columns": {"event": "os_event"}}
        return (
            StatisticCase("survival_time", spec, values=(12.0, 48.5, 3.0, 12.0), best=48.5,
                          row=_with_event, thresholds=(("le", 12.0), ("gt", 3.0), ("ge", 48.5)),
                          refused=(("lt", -1.0), ("eq", 12.0)), agg=("count", "max")),
            PairedCase("km_textbook", times=(6, 6, 6, 7, 10, 13, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34,
                                             35),
                       events=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
            PairedCase("not_reached", times=(5, 8, 12, 20), events=(1, 0, 0, 0)),
            PairedCase("patient_level", times=(10, 10, 4, 30), events=(1, 1, 1, 0), keys=("P1", "P1", "P2", "P3"),
                       expected_n=3),
        )


def _with_event(value: Any) -> dict[str, Any]:
    return {"value": value, "os_event": True}
