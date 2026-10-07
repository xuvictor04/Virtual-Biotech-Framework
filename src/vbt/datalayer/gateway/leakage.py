"""Evidence-date ceiling (§6.1 ``LeakageSpec``, §11.3 step 8, §11.7 T1). No pyarrow.

When ``data.leakage.ceiling`` is set (no-web scenarios) and a table a tool reads belongs to a
source that declares ``leakage``, facts dated after the ceiling must not reach the agent:

* **prepare**: the overlay's ``leakage_filter`` is injected into the named argument
  (``AREA[StudyFirstPostDate]RANGE[MIN,{ceiling}]``), so counts and searches are computed
  upstream under the ceiling; a counting tool without a ``leakage_filter`` is ``quarantined``
  under ``counts: block`` (its count cannot be corrected row by row);
* **finish (T1)**: rows whose ``available_at`` is after the ceiling are withheld; rows whose
  ``changed_at`` is after it are withheld, or have the ``redact`` columns nulled, or are only
  stamped, per ``LeakageSpec.rows``. Partial dates follow ``partial_dates``: under ``latest``
  ``2004-01`` counts as 2004-01-31 and ``2004`` as 2004-12-31, so a record that might postdate the
  ceiling is treated as if it did. A row whose date is missing or unparsable cannot be checked: it
  is kept and ``provenance.leakage.risk`` is set.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping

from ..errors import ErrorKind, GatewayError
from .fields import get_path, set_path

__all__ = ["LeakagePlan", "ceiling_of", "parse_date", "leakage_specs", "is_counting", "prepare_leakage",
           "withhold_rows", "leakage_record"]

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_ISO = re.compile(r"^\s*(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(?:[T ].*)?\s*$")
_WORDY = re.compile(r"^\s*(?:(\d{1,2})\s+)?([A-Za-z]+)\.?\s+(?:(\d{1,2}),\s*)?(\d{4})\s*$")


def parse_date(value: Any, partial: str = "latest") -> date | None:
    """A date from ``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD`` (time ignored) or ``Month YYYY`` /
    ``Month D, YYYY`` text; partial dates complete to the last (``latest``) or first
    (``earliest``) day of their period. None when the value is missing or unparsable."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, int) and 1000 <= value <= 9999:
        value = str(value)
    text = str(value).strip()
    m = _ISO.match(text)
    year: int | None = None
    month: int | None = None
    day: int | None = None
    if m is not None:
        year = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else None
        day = int(m.group(3)) if m.group(3) else None
    else:
        w = _WORDY.match(text)
        if w is None or w.group(2).lower() not in _MONTHS:
            return None
        year, month = int(w.group(4)), _MONTHS[w.group(2).lower()]
        d = w.group(1) or w.group(3)
        day = int(d) if d else None
    try:
        if month is None:
            return date(year, 12, 31) if partial == "latest" else date(year, 1, 1)
        if day is None:
            last = calendar.monthrange(year, month)[1]
            return date(year, month, last if partial == "latest" else 1)
        return date(year, month, day)
    except ValueError:
        return None


def ceiling_of(settings: Any) -> date | None:
    """The configured ceiling (``data.leakage.ceiling``) as a date; a partial ceiling counts as
    its earliest day (the conservative end)."""
    raw = getattr(getattr(settings, "leakage", None), "ceiling", None)
    if raw in (None, ""):
        return None
    return parse_date(raw, "earliest")


def leakage_specs(contract: Any) -> dict[str, Any]:
    """``{source.table: LeakageSpec}`` for the tables the tool reads whose source declares ``leakage``."""
    out: dict[str, Any] = {}
    for ref, t in getattr(contract, "tables", {}).items():
        spec = getattr(t.descriptor, "leakage", None)
        if spec is not None:
            out[ref] = spec
    return out


def is_counting(contract: Any) -> bool:
    """A tool whose answer is a count or carries an upstream total (searches report totals)."""
    b = getattr(contract, "binding", None)
    if b is None:
        return False
    r = b.result
    return r.kind == "count" or r.total is not None


@dataclass
class LeakagePlan:
    ceiling: date | None = None
    specs: dict[str, Any] = field(default_factory=dict)
    injected: str | None = None                        # the argument the filter was injected into
    notes: list[str] = field(default_factory=list)
    risk: bool = False

    @property
    def active(self) -> bool:
        return self.ceiling is not None and bool(self.specs)


def prepare_leakage(contract: Any, args_sent: dict[str, Any], ceiling: date | None, *,
                    tool: str | None = None) -> LeakagePlan:
    """Inject the overlay's ``leakage_filter`` or refuse a counting tool (``quarantined``).
    ``args_sent`` is updated in place."""
    specs = leakage_specs(contract)
    plan = LeakagePlan(ceiling=ceiling, specs=specs)
    if ceiling is None or not specs:
        return plan
    binding = contract.binding
    text = ceiling.isoformat()
    lf = getattr(binding, "leakage_filter", None) if binding is not None else None
    if lf is not None:
        fragment = lf.template.replace("{ceiling}", text)
        existing = args_sent.get(lf.arg)
        # both sides parenthesised: AND binds tighter than OR, so "x OR y AND <ceiling>" would leave x unbounded
        args_sent[lf.arg] = f"({existing}) AND ({fragment})" if existing not in (None, "") else fragment
        plan.injected = lf.arg
        plan.notes.append(f"evidence ceiling {text} applied upstream through {lf.arg}")
        return plan
    if is_counting(contract):
        modes = {getattr(s, "counts", "block") for s in specs.values()}
        if "block" in modes:
            raise GatewayError(
                ErrorKind.quarantined,
                f"this tool counts records that may postdate the evidence ceiling {text} and has no way to "
                "apply the ceiling upstream", tool=tool,
                payload={"reason": "leakage", "ceiling": text, "alternatives": []}, subkind="leakage")
        plan.risk = True
        plan.notes.append(f"counts may include records after the evidence ceiling {text}")
    return plan


def withhold_rows(rows: Iterable[Any], spec: Any, ceiling: date | None) -> tuple[list[Any], dict[str, int]]:
    """T1 on logical rows: ``(kept rows, {withheld, redacted, stamped, unchecked})``."""
    rows = list(rows)
    counts = {"withheld": 0, "redacted": 0, "stamped": 0, "unchecked": 0}
    if ceiling is None or spec is None:
        return rows, counts
    partial = getattr(spec, "partial_dates", "latest")
    kept: list[Any] = []
    for row in rows:
        if not isinstance(row, Mapping):
            kept.append(row)
            continue
        avail = parse_date(get_path(row, spec.available_at), partial)
        if avail is None:
            counts["unchecked"] += 1
        elif avail > ceiling:
            counts["withheld"] += 1
            continue
        if spec.changed_at:
            changed_raw = get_path(row, spec.changed_at)
            changed = parse_date(changed_raw, partial)
            if changed is not None and changed > ceiling:
                if spec.rows == "withhold":
                    counts["withheld"] += 1
                    continue
                if spec.rows == "redact":
                    for col in spec.redact:
                        if get_path(row, col) is not None:
                            set_path(row, col, None)  # type: ignore[arg-type]
                    counts["redacted"] += 1
                else:
                    counts["stamped"] += 1
            elif changed is None and changed_raw is not None:
                counts["unchecked"] += 1
        kept.append(row)
    return kept, counts


def leakage_record(ceiling: date | None, withheld: int, risk: bool) -> dict[str, Any] | None:
    """The provenance ``leakage`` block (None without a ceiling)."""
    if ceiling is None:
        return None
    return {"ceiling": ceiling.isoformat(), "withheld": int(withheld), "risk": bool(risk)}
