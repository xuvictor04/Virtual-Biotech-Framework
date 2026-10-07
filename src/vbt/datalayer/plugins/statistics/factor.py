"""``signed_factor`` and ``binary_factor``: prioritisation factors (§9.4, CT-6).

With an ``encoding`` (stored code -> label), a factor is filtered by labels, and only in the
**positive** form: ``{in: [label]}`` or ``eq`` compiles to ``In(column, codes)``. A negated
form (``ne``, a threshold) raises ``UnsupportedFilter`` (``reason="negated_form"``), because
``hasSafetyEvent != 1`` passes the null and ``-1`` rows (VERIFIED, CT-6) while ``{in:
[none_recorded]}`` passes only the rows that say so. Codes whose label is null are in-band
unknowns. An encoding with ``verified: false`` needs confirmed facts (a distinct-value
snapshot in which every code was observed, :func:`~.confirm_codes`) before it may filter (I9);
otherwise ``UnsupportedFilter`` (``reason="unconfirmed_encoding"``).

``signed_factor`` without an encoding is a number on ``[-1, 1]`` (higher is more favourable);
``binary_factor`` defaults to the encoding ``{0: no, 1: yes}`` and matches ``0``/``False`` and
``1``/``True`` spellings alike.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

from ...predicate import In, Predicate
from ..base import ColumnStats, UnsupportedFilter, ValueSnapshot
from ..registry import register
from . import StatisticBase, _as_number, confirm_codes, is_number, same_value, spec_get

_UNKNOWN_LABELS = frozenset({"unknown", "missing", "not_assessed", "na", "n/a"})


class _FactorBase(StatisticBase):
    default_encoding: ClassVar[Mapping[Any, Any] | None] = None
    aggregations = frozenset({"min", "max", "count", "mean"})

    def encoding(self, spec: Any) -> Mapping[Any, Any] | None:
        enc = spec_get(spec, "encoding", None)
        return dict(enc) if enc else (dict(self.default_encoding) if self.default_encoding else None)

    def missing_codes(self, spec: Any) -> tuple[Any, ...]:
        codes = list(super().missing_codes(spec))
        for code, label in (self.encoding(spec) or {}).items():
            if label is None and not any(same_value(code, c) for c in codes):
                codes.append(code)
        return tuple(codes)

    def spellings(self, code: Any) -> list[Any]:
        """Every stored spelling a declared code may have (``"1"`` -> ``"1"``, ``1``)."""
        out = [code]
        n = _as_number(code)
        if isinstance(code, str) and n is not None:
            out.append(int(n) if n.is_integer() else n)
        return out

    def codes_for(self, column: str, value: Any, spec: Any) -> list[Any]:
        encoding = self.encoding(spec) or {}
        matched = []
        for code, label in encoding.items():
            hit = same_value(code, value)
            if not hit and isinstance(value, str) and isinstance(label, str):
                hit = label.casefold() == value.casefold()
            if hit:
                if label is None or (isinstance(label, str) and label.casefold() in _UNKNOWN_LABELS):
                    raise UnsupportedFilter(f"{value!r} means unknown for {column} and cannot be selected",
                                            reason="invalid_value", column=column)
                matched.append(code)
        if not matched:
            labels = ", ".join(f"{label} ({code})" for code, label in encoding.items() if label is not None)
            raise UnsupportedFilter(f"{value!r} is not a declared value of {column}: {labels}",
                                    reason="invalid_value", column=column)
        return matched

    def check_values(self, column: str, op: str, values: Sequence[Any], spec: Any,
                     confirmed: Mapping[str, Any] | None) -> None:
        if self.encoding(spec) is None:
            super().check_values(column, op, values, spec, confirmed)
            return
        if op not in ("eq", "in"):
            raise UnsupportedFilter(
                f"{column} is an encoded factor: filter it with the positive form {{in: [label]}} naming the "
                f"values to keep, not {op}", reason="negated_form" if op == "ne" else "op", column=column)
        if confirmed is not None and confirmed.get("confirmed") is False:
            raise UnsupportedFilter(f"the encoding of {column} was refuted by the data", reason="unconfirmed_encoding",
                                    column=column)
        if self.unverified(spec) and not self.confirmed_ok(confirmed):
            raise UnsupportedFilter(f"the encoding of {column} is not confirmed from the data; filtering on its "
                                    f"labels would be a guess", reason="unconfirmed_encoding", column=column)
        for v in values:
            self.codes_for(column, v, spec)

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        if self.encoding(spec) is None:
            return super().compare(column, op, value, spec, confirmed)
        vals = value if op == "in" and isinstance(value, (list, tuple, set, frozenset)) else [value]
        stored: list[Any] = []
        for v in vals:
            for code in self.codes_for(column, v, spec):
                for s in self.spellings(code):
                    if not any(type(s) is type(t) and same_value(s, t) for t in stored):
                        stored.append(s)
        return In(column, tuple(stored))

    def rank_value(self, value: Any, spec: Any) -> float | None:
        if isinstance(value, bool):
            value = int(value)
        return super().rank_value(value, spec)

    def check_facts(self, stats: ColumnStats | None, snapshot: ValueSnapshot | None, spec: Any,
                    known: list[Any] | None, mn: float | None, mx: float | None, facts: dict[str, Any],
                    problems: list[str], undecided: list[str]) -> None:
        encoding = self.encoding(spec)
        if encoding is None:
            super().check_facts(stats, snapshot, spec, known, mn, mx, facts, problems, undecided)
            return
        codes = [c for c, label in encoding.items() if label is not None]
        confirm_codes(codes, known, snapshot, self.missing_codes(spec), facts, problems, undecided)

    def scale_text(self, spec: Any) -> str:
        encoding = self.encoding(spec)
        if encoding is None:
            return super().scale_text(spec)
        text = "scale of codes " + ", ".join(f"{c} = {label if label is not None else 'unknown'}"
                                             for c, label in encoding.items())
        if self.unverified(spec):
            text += " (not yet confirmed from the data)"
        return text + "; filter with {in: [label]} only"


@register
class SignedFactorStatistic(_FactorBase):
    name: ClassVar[str] = "signed_factor"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (-1.0, 1.0)
    default_direction: ClassVar[str | None] = "higher_is_stronger"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("signed_factor", {"role": "measure", "statistic": cls.name},
                          values=(-1.0, 0.0, 0.5, 1.0, 0.5), best=1.0,
                          thresholds=(("ge", 0.5), ("lt", 0.0), ("le", -1.0)),
                          refused=(("gt", 2.0), ("lt", -3.0))),
            StatisticCase("signed_factor_encoded", {"role": "measure", "statistic": cls.name,
                                                    "encoding": {-1: "has_safety_event", 0: "none_recorded",
                                                                 1: None}},
                          values=(-1, 0, 0, -1), best=0, unknown=(None, float("nan"), 1),
                          thresholds=(("in", ["none_recorded"]), ("eq", "has_safety_event"), ("in", [0])),
                          refused=(("ne", "has_safety_event"), ("lt", 0), ("in", ["no_such_label"]),
                                   ("eq", 1)),
                          unconfirmed=(("in", ["none_recorded"]),), confirmed={"confirmed": True, "codes": [-1, 0]},
                          agg=("max", "count")),
        )


@register
class BinaryFactorStatistic(_FactorBase):
    name: ClassVar[str] = "binary_factor"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, 1.0)
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    default_encoding: ClassVar[Mapping[Any, Any] | None] = {0: "no", 1: "yes"}

    def spellings(self, code: Any) -> list[Any]:
        out = super().spellings(code)
        n = _as_number(code)
        if n in (0.0, 1.0) and not isinstance(code, bool):
            out.append(bool(n))
        return out

    def codes_for(self, column: str, value: Any, spec: Any) -> list[Any]:
        if isinstance(value, bool):
            value = int(value)
        return super().codes_for(column, value, spec)

    def rank_value(self, value: Any, spec: Any) -> float | None:
        if is_number(value) or isinstance(value, bool):
            return super().rank_value(value, spec)
        return None

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("binary_factor", {"role": "measure", "statistic": cls.name},
                          values=(0, 1, 1, 0), best=1,
                          thresholds=(("eq", "yes"), ("in", [1]), ("eq", True), ("in", ["no"])),
                          refused=(("gt", 0), ("ne", "yes"), ("eq", "maybe")),
                          unconfirmed=(("eq", "yes"),), confirmed={"confirmed": True, "codes": [0, 1]},
                          agg=("max", "count", "mean")),
        )
