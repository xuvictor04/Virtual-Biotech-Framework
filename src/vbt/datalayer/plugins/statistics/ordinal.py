"""``ordinal`` and ``clinical_phase`` (§9.4).

``ordinal`` orders values by the spec's ``levels`` (ordered string ordinals, ``"1A" < "1B"``)
or, without levels, by the declared order of its ``encoding`` codes. Thresholds compile to a
positive ``In`` over the qualifying levels, so a value the spec does not declare (or an
in-band unknown code) never passes; ``eq``/``in`` values may be stored codes or encoding
labels. A value that is not a declared level raises ``UnsupportedFilter`` (``invalid_value``).
Without levels or an encoding it behaves like ``numeric``.

``clinical_phase``: Open Targets stores phases on ``[0, 4]`` in some tables and normalised to
``[0, 1]`` in others, and ChEMBL uses ``-1`` for unknown. The scale is ``[0, 4]`` unless the
confirmed maximum is at most 1; until readiness confirmed the range (or the spec declares a
verified ``scale``) a threshold raises ``UnsupportedFilter`` (``reason="scale"``). ``-1`` is a
default in-band unknown code, merged with the spec's ``missing_values``.
"""

from __future__ import annotations

import statistics as _stats
from typing import Any, ClassVar, Mapping, Sequence

from ...predicate import In, Predicate
from ..base import ColumnStats, UnsupportedFilter, ValueSnapshot
from ..registry import register
from . import StatisticBase, _fmt_scale, confirm_codes, is_number, same_value, spec_get


@register
class OrdinalStatistic(StatisticBase):
    name: ClassVar[str] = "ordinal"
    version: ClassVar[str] = "1.0"
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    aggregations = frozenset({"min", "max", "median", "count"})

    # -- levels ------------------------------------------------------------

    def levels(self, spec: Any) -> list[Any] | None:
        """Stored values in order, lowest first (``levels`` or the encoding's code order)."""
        levels = spec_get(spec, "levels", None)
        if levels:
            return list(levels)
        encoding = spec_get(spec, "encoding", None)
        if encoding:
            return list(encoding)
        return None

    def level_index(self, value: Any, spec: Any) -> int | None:
        levels = self.levels(spec)
        if levels is None or value is None:
            return None
        for i, level in enumerate(levels):
            if same_value(level, value):
                return i
        encoding = spec_get(spec, "encoding", None) or {}
        for i, code in enumerate(levels):                  # an encoding label names its code
            label = encoding.get(code) if isinstance(encoding, Mapping) else None
            if isinstance(label, str) and isinstance(value, str) and label.casefold() == value.casefold():
                return i
        return None

    def rank_value(self, value: Any, spec: Any) -> float | None:
        if self.levels(spec) is None:
            return super().rank_value(value, spec)
        if value is None or self.is_unknown(value, spec):
            return None
        i = self.level_index(value, spec)
        return None if i is None else float(i)

    def _stored(self, indexes: Sequence[int], spec: Any) -> tuple[Any, ...]:
        levels = self.levels(spec) or []
        out: list[Any] = []
        for i in indexes:
            level = levels[i]
            out.append(level)
            # an unquoted YAML code may be stored as a number or as text: match either spelling
            if isinstance(level, str) and is_number(_num_or_none(level)):
                out.append(_num_or_none(level))
        return tuple(out)

    def _index_or_raise(self, column: str, value: Any, spec: Any) -> int:
        i = self.level_index(value, spec)
        if i is None:
            raise UnsupportedFilter(f"{value!r} is not a declared level of {column} "
                                    f"({', '.join(map(str, self.levels(spec) or []))})", reason="invalid_value",
                                    column=column)
        return i

    # -- protocol hooks ----------------------------------------------------

    def check_values(self, column: str, op: str, values: Sequence[Any], spec: Any,
                     confirmed: Mapping[str, Any] | None) -> None:
        if self.levels(spec) is None:
            super().check_values(column, op, values, spec, confirmed)
            return
        if op.endswith("_abs") or op == "significant":
            raise UnsupportedFilter(f"{op} is not defined for the ordinal {column}", reason="op", column=column)
        if self.unverified(spec) and not self.confirmed_ok(confirmed):
            raise UnsupportedFilter(f"the levels of {column} are not confirmed from the data; filtering on them "
                                    f"would be a guess", reason="unconfirmed_encoding", column=column)
        for v in values:
            self._index_or_raise(column, v, spec)

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        levels = self.levels(spec)
        if levels is None:
            return super().compare(column, op, value, spec, confirmed)
        n = len(levels)
        if op in ("eq", "in"):
            vals = value if op == "in" and isinstance(value, (list, tuple, set, frozenset)) else [value]
            keep = sorted({self._index_or_raise(column, v, spec) for v in vals})
        elif op == "ne":
            i = self._index_or_raise(column, value, spec)
            keep = [j for j in range(n) if j != i]
        elif op == "range":
            lo, hi = (value[0], value[1]) if isinstance(value, (list, tuple)) else (value.get("lo"), value.get("hi"))
            a = 0 if lo is None else self._index_or_raise(column, lo, spec)
            b = n - 1 if hi is None else self._index_or_raise(column, hi, spec)
            keep = list(range(a, b + 1))
        else:
            i = self._index_or_raise(column, value, spec)
            keep = {"lt": range(0, i), "le": range(0, i + 1), "gt": range(i + 1, n), "ge": range(i, n)}[op]
        codes = self.missing_codes(spec)
        keep = [j for j in keep if not any(same_value(levels[j], c) for c in codes)]
        return In(column, self._stored(keep, spec))

    def check_facts(self, stats: ColumnStats | None, snapshot: ValueSnapshot | None, spec: Any,
                    known: list[Any] | None, mn: float | None, mx: float | None, facts: dict[str, Any],
                    problems: list[str], undecided: list[str]) -> None:
        levels = self.levels(spec)
        if levels is None:
            super().check_facts(stats, snapshot, spec, known, mn, mx, facts, problems, undecided)
            return
        if spec_get(spec, "encoding", None):
            confirm_codes(levels, known, snapshot, self.missing_codes(spec), facts, problems, undecided)
            return
        if snapshot is None or not snapshot.complete or known is None:
            undecided.append("levels are confirmed only from a complete distinct-value snapshot")
            return
        extra = [v for v in known if self.level_index(v, spec) is None]
        if extra:
            problems.append(f"observed value(s) {', '.join(map(repr, extra[:10]))} are not declared levels")
        facts["levels_seen"] = [lv for lv in levels if any(same_value(lv, v) for v in known)]

    def combine(self, known: list[Any], how: str, spec: Any) -> Any:
        levels = self.levels(spec)
        if levels is None:
            return super().combine(known, how, spec)
        idx = [i for i in (self.level_index(v, spec) for v in known) if i is not None]
        if not idx:
            return None
        if how == "count":
            return len(idx)
        pick = {"min": min(idx), "max": max(idx), "median": _stats.median_low(idx)}[how]
        return levels[pick]

    def scale_text(self, spec: Any) -> str:
        levels = self.levels(spec)
        if levels is None:
            return super().scale_text(spec)
        encoding = spec_get(spec, "encoding", None) or {}
        names = [f"{lv} ({encoding[lv]})" if isinstance(encoding, Mapping) and lv in encoding else str(lv)
                 for lv in levels]
        text = "ordinal scale " + " < ".join(names)
        if self.unverified(spec):
            text += " (not yet confirmed from the data)"
        return text

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        levels = ["1A", "1B", "2A", "2B", "3", "4"]
        return (
            StatisticCase("ordinal_levels", {"role": "measure", "statistic": cls.name, "levels": levels,
                                             "direction": "lower_is_stronger", "missing_values": ["N/A"]},
                          values=("2A", "1A", "3", "1A", "4"), best="1A",
                          unknown=(None, "N/A"),
                          thresholds=(("le", "1B"), ("lt", "2A"), ("ge", "3"), ("gt", "2B"), ("eq", "2B"),
                                      ("in", ["1A", "4"]), ("ne", "3")),
                          refused=(("le", "5A"), ("in", ["1A", "zz"])),
                          unconfirmed=(("le", "1B"),), confirmed={"confirmed": True},
                          agg=("max", "min", "median", "count"), group=("drugId", "CHEMBL1", "CHEMBL2")),
            StatisticCase("ordinal_encoding", {"role": "measure", "statistic": cls.name,
                                               "encoding": {0: "none", 1: "low", 2: "high"}, "missing_values": [-1]},
                          values=(0, 2, 1, 2), best=2, unknown=(None, float("nan"), -1),
                          thresholds=(("ge", 1), ("lt", 2), ("eq", "low"), ("in", ["high"])),
                          refused=(("ge", 7),), unconfirmed=(("ge", 1),), confirmed={"confirmed": True},
                          agg=("max", "count")),
        )


@register
class ClinicalPhaseStatistic(StatisticBase):
    name: ClassVar[str] = "clinical_phase"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, 4.0)
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    default_missing: ClassVar[tuple[Any, ...]] = (-1,)
    scale_needs_confirmation: ClassVar[bool] = True
    aggregations = frozenset({"min", "max", "median", "count"})

    def effective_scale(self, spec: Any, confirmed: Mapping[str, Any] | None
                        ) -> tuple[tuple[float | None, float | None] | None, bool]:
        declared = spec_get(spec, "scale", None)
        if declared and not self.unverified(spec):
            return (float(declared[0]), float(declared[1])), True
        if self.confirmed_ok(confirmed):
            assert confirmed is not None
            if confirmed.get("scale"):
                lo, hi = confirmed["scale"]
                return (lo, hi), True
            mx = confirmed.get("max")
            if is_number(mx):
                return ((0.0, 1.0) if mx <= 1 else (0.0, 4.0)), True
        return self.default_scale, False

    def check_facts(self, stats: ColumnStats | None, snapshot: ValueSnapshot | None, spec: Any,
                    known: list[Any] | None, mn: float | None, mx: float | None, facts: dict[str, Any],
                    problems: list[str], undecided: list[str]) -> None:
        if mn is None or mx is None:
            undecided.append("no min/max (after in-band codes) to decide between the 0-4 and 0-1 phase scales")
            return
        if mn < 0:
            problems.append(f"phase {mn} below 0 is not a declared unknown code")
        if mx > 4:
            problems.append(f"phase {mx} above 4")
        if not problems:
            facts["scale"] = [0.0, 1.0] if mx <= 1 else [0.0, 4.0]

    def scale_text(self, spec: Any) -> str:
        declared = spec_get(spec, "scale", None)
        if declared and not self.unverified(spec):
            return f"scale {_fmt_scale(declared)}"
        return "scale [0, 4] (phases) or [0, 1] (normalised), decided by the confirmed range"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("clinical_phase", {"role": "measure", "statistic": cls.name},
                          values=(0.5, 4.0, 2.0, 3.0, 4.0), best=4.0, unknown=(None, float("nan"), -1),
                          thresholds=(("ge", 3.0), ("lt", 2.0), ("eq", 4.0)),
                          refused=(("ge", 5.0), ("lt", -1.0)),
                          unconfirmed=(("ge", 3.0),), confirmed={"confirmed": True, "max": 4.0},
                          agg=("max", "min", "count")),
        )


def _num_or_none(text: str) -> Any:
    try:
        f = float(text)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() and "." not in text else f
