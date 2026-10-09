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

A source whose tool applies a ceiling itself declares where it comes from (``ceiling_from``): the PubMed
server bounds every search by ``VBT_LITERATURE_MAXDATE`` (``web.literature_max_date``), so that source's
``leakage`` names ``web.literature_max_date``. The gateway's prepare and T1 act on ``data.leakage.ceiling``
sources only; the data child bounds its own counts and live reads by each source's ceiling
(:func:`ceiling_for`), so a witness counts what the tool counted. The rows such a server withheld itself
(``result.withheld_list``) are the header's ``withheld.leakage`` and the provenance ``leakage`` record.

Under the ceiling, two more cases are disclosed rather than served as if bounded:

* a call that selects on the record as it is today (:func:`current_selection`: a ``redact`` column, a date
  other than ``available_at``, a declared ``current_terms`` engine term such as CT.gov's
  ``AREA[ResultsFirstPostDate]``): its count is not bounded by a first-posted filter, so the gateway reports the
  records also unchanged since the ceiling as the total, or marks the answer partial with ``risk`` when it
  cannot count them;
* a live source that declares no ``leakage`` (cBioPortal, the Census: :func:`undated_sources`): its records
  carry no date the gateway can check, so provenance records ``leakage {ceiling, withheld: 0, risk: true,
  reason: 'source not dated'}`` and the header says so.
"""

from __future__ import annotations

import calendar
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping

from ..descriptor.models import CEILING_SOURCES, DATA_CEILING, LITERATURE_CEILING
from ..errors import ErrorKind, GatewayError
from .fields import get_path, set_path

__all__ = ["LeakagePlan", "ceiling_of", "ceiling_for", "parse_date", "leakage_specs", "is_counting",
           "prepare_leakage", "withhold_rows", "leakage_record", "current_selection", "undated_sources",
           "self_bounded_specs", "DATA_CEILING", "LITERATURE_CEILING", "CEILING_SOURCES"]

LITERATURE_MAXDATE_ENV = "VBT_LITERATURE_MAXDATE"     # web.literature_max_date, as the harness exports it

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


def ceiling_for(spec: Any, settings: Any, environ: Mapping[str, str] | None = None) -> date | None:
    """The ceiling of one source's ``leakage`` (its ``ceiling_from``): ``data.leakage.ceiling``, or the
    literature ceiling ``VBT_LITERATURE_MAXDATE`` (``YYYY/MM/DD``) the harness exports to every server and to
    the data child; None when that ceiling is unset."""
    src = getattr(spec, "ceiling_from", DATA_CEILING) or DATA_CEILING
    if src == LITERATURE_CEILING:
        raw = str((os.environ if environ is None else environ).get(LITERATURE_MAXDATE_ENV) or "").strip()
        return parse_date(raw.replace("/", "-"), "earliest") if raw else None
    return ceiling_of(settings)


def leakage_specs(contract: Any) -> dict[str, Any]:
    """``{source.table: LeakageSpec}`` for the tables the tool reads whose source declares ``leakage`` bounded
    by ``data.leakage.ceiling`` (a source whose server applies its own ceiling is not the gateway's to bound)."""
    out: dict[str, Any] = {}
    for ref, t in getattr(contract, "tables", {}).items():
        spec = getattr(t.descriptor, "leakage", None)
        if spec is not None and (getattr(spec, "ceiling_from", DATA_CEILING) or DATA_CEILING) == DATA_CEILING:
            out[ref] = spec
    return out


def self_bounded_specs(contract: Any) -> dict[str, Any]:
    """``{source.table: LeakageSpec}`` of the tables whose server applies its own ceiling (``ceiling_from`` other
    than ``data.leakage.ceiling``: PubMed's literature ceiling)."""
    out: dict[str, Any] = {}
    for ref, t in getattr(contract, "tables", {}).items():
        spec = getattr(t.descriptor, "leakage", None)
        if spec is not None and (getattr(spec, "ceiling_from", DATA_CEILING) or DATA_CEILING) != DATA_CEILING:
            out[ref] = spec
    return out


def undated_sources(contract: Any) -> list[str]:
    """The remote sources a tool reads that declare no ``leakage`` (cBioPortal, the Census): their records carry no
    date to check against the evidence ceiling."""
    out: list[str] = []
    for t in getattr(contract, "tables", {}).values():
        d = t.descriptor
        if getattr(d, "kind", None) == "remote" and getattr(d, "leakage", None) is None and d.source not in out:
            out.append(d.source)
    return out


def _leaves(cols: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for name, col in (cols or {}).items():
        path = f"{prefix}{name}"
        fields = getattr(col, "fields", None)
        if fields:
            yield from _leaves(fields, path + ".")
        else:
            yield path, col


def current_selection(contract: Any, args: Mapping[str, Any], specs: Mapping[str, Any]) -> list[str]:
    """The arguments of a call that select on the record as the source holds it today (LIVE3-01): a bound
    argument on a ``redact`` column or a date column other than ``available_at`` (CT.gov ``status`` on
    overallStatus), or a free-text engine argument naming such a column's ``remote_name`` or a declared
    ``current_terms`` term (``advanced_filter='AREA[ResultsFirstPostDate]RANGE[2018-01-01,MAX]'``). Each entry
    reads ``argument (field)``."""
    out: list[str] = []
    for ref, spec in specs.items():
        t = getattr(contract, "tables", {}).get(ref)
        if t is None:
            continue
        redact = {str(c) for c in spec.redact}
        current: set[str] = set(redact)
        terms: dict[str, str] = {str(x).casefold(): str(x) for x in getattr(spec, "current_terms", []) or []}
        for path, col in _leaves(t.columns):
            if path in redact or (getattr(col, "role", None) == "time" and path != spec.available_at):
                current.add(path)
                rn = getattr(col, "remote_name", None)
                if rn:
                    terms[str(rn).casefold()] = str(rn)
        for name, a in getattr(contract, "args", {}).items():
            v = args.get(name)
            if v is None or v == "" or v == []:
                continue
            for tb, col in contract.arg_columns(name):
                if tb == ref and str(col).replace("[]", "") in current:
                    out.append(f"{name} ({str(col).rsplit('.', 1)[-1]})")
            if getattr(a, "role", None) == "free_text" and isinstance(v, str):
                low = v.casefold()
                out.extend(f"{name} ({shown})" for key, shown in sorted(terms.items()) if key in low)
    return list(dict.fromkeys(out))


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
    current: list[str] = field(default_factory=list)   # arguments selecting on the record as it is today
    undated: list[str] = field(default_factory=list)   # remote sources read that carry no date
    reason: str | None = None                          # why risk is set without a row-level check
    # a source whose server applies its own ceiling (PubMed): that ceiling, recorded with the rows it withheld
    self_ceiling: date | None = None

    @property
    def active(self) -> bool:
        return self.ceiling is not None and bool(self.specs)

    @property
    def recorded(self) -> bool:
        """Provenance carries a ``leakage`` record: a ceiling bounds a dated source, or an undated one was read
        under it, or the source's server applied its own ceiling."""
        return self.active or (self.ceiling is not None and bool(self.undated)) or self.self_ceiling is not None


def prepare_leakage(contract: Any, args_sent: dict[str, Any], ceiling: date | None, *,
                    tool: str | None = None, environ: Mapping[str, str] | None = None) -> LeakagePlan:
    """Inject the overlay's ``leakage_filter`` or refuse a counting tool (``quarantined``).
    ``args_sent`` is updated in place. ``environ`` holds the literature ceiling the servers get."""
    specs = leakage_specs(contract)
    plan = LeakagePlan(ceiling=ceiling, specs=specs)
    for spec in self_bounded_specs(contract).values():
        plan.self_ceiling = plan.self_ceiling or ceiling_for(spec, None, environ)
    if ceiling is not None:
        plan.undated = undated_sources(contract)
        if plan.undated:
            plan.risk = True
            plan.reason = "source not dated"
            plan.notes.append(f"evidence ceiling {ceiling.isoformat()}: {', '.join(plan.undated)} records carry no "
                              "date the gateway can check against it; they may postdate the ceiling (leakage risk)")
    if ceiling is None or not specs:
        return plan
    binding = contract.binding
    text = ceiling.isoformat()
    lf = getattr(binding, "leakage_filter", None) if binding is not None else None
    if lf is not None:
        plan.current = current_selection(contract, args_sent, specs)   # before the ceiling filter is added
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


def leakage_record(ceiling: date | None, withheld: int, risk: bool, reason: str | None = None
                   ) -> dict[str, Any] | None:
    """The provenance ``leakage`` block (None without a ceiling); ``reason`` says why ``risk`` holds without a
    row-level check (``source not dated``, ``selects on the current record``)."""
    if ceiling is None:
        return None
    out: dict[str, Any] = {"ceiling": ceiling.isoformat(), "withheld": int(withheld), "risk": bool(risk)}
    if reason:
        out["reason"] = reason
    return out
