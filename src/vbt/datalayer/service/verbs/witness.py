"""``_witness``: an independent bounded count, top-k and key set of a bound table (§11.6).

One scan of the table or item table computes everything the request asks for: ``total`` (rows,
items, or distinct values of ``grain``), the top-k canonical keys under ``order`` (per comparable
group: ``group_by``, else the ``within`` groups of the order), the key set when the total is at most
``key_set_max`` (``data.witness.max_key_set``), distinct values, grain counts, group totals, keys
shared by several rows (``one_to_many``) and the unknown attribution (``excluded_unknown`` per
column with ``_rows``, ``excluded_not_applicable``, ``unknown_total``).

When the witness cannot express the request (a parameter without a value, a table served upstream
only) or the scan would exceed the budget, the response is ``total_method: unknown`` with a reason,
and the gateway makes no count or ranking claim from it.

**Remote witness** (phase 4, F20): a table whose layout declares the ``count`` capability (``live_api``,
``soma``) is not scanned. The bound predicate is compiled into one independent count request (CT.gov
``countTotal=true&pageSize=0``, E-utilities ``esearch`` ``rettype=count``, a SOMA ``value_filter`` read of
``soma_joinid``) within the source's request budget (``budget`` of the descriptor). Only the total is
answered: no top-k, no key set. A predicate that does not compile completely, a request that fails or
a source without a total is ``unknown`` (never a guess). The method is reported as ``scan`` with the
reason ``remote count request``. When the descriptor injects the evidence ceiling into counts
(``leakage.counts: inject_filter``), the ceiling is one more conjunct of the counted predicate
(``available_at <= ceiling``), so the remote total counts what the find may return: CT.gov
``AREA[StudyFirstPostDate]RANGE[MIN,2017-12-31]`` under a 2017-12-31 ceiling, not every trial registered since.

``_live_find`` (phase 4) is the derived ``lookup``/``find`` of live tables: the pages of one request
within the source's budget (``max_pages`` and ``max_requests_per_call``; more pages are ``truncated``,
never presented as complete), ``pivot`` tables reshaped to one row per index (cBioPortal patient-level
clinical data: one row per ``(studyId, patientId)`` with one column per attribute), and record versions
(``key.version``) observed so that a record the source changed since an earlier call is listed under
``source_updated`` for provenance. A ``limit`` sizes the pages (a find for 2 trials asks for a page of 2,
not ten pages of 100) and stops at ``limit`` rows; a cut answer without a total makes one count request
so that the gateway still knows how many rows matched. The source's release (``release`` of the layout,
cached for ``live_api.RELEASE_TTL_S``) is the ``as_of`` of the answer. A failed request is a typed error
(:func:`remote_failure`): a key the source does not know (HTTP 404 on a filled path) is ``not_found``, a
rejected query ``invalid_argument``, an unreachable or busy source a retryable ``source_error``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...errors import ErrorKind, GatewayError
from ...ipc import VERB_WITNESS, WitnessRequest, WitnessResponse
from ...plugins.base import FormatError
from ...predicate import from_json, to_json
from .. import ServiceContext, ServiceError, layout_spec
from ..reader import BudgetExceeded, TableUnavailable, UnboundParameter

__all__ = ["witness", "remote_witness", "remote_names", "remote_kinds", "is_remote", "live_find", "pivot_rows",
           "remote_failure", "requested_keys", "leakage_conjunct", "leakage_bounds", "leakage_ceiling", "REMOTE_REASON",
           "LIVE_FIND", "VERBS"]

REMOTE_REASON = "remote count request"
LIVE_FIND = "_live_find"


def remote_names(columns: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """``{column path: remote_name}`` of a table's columns (nested fields included)."""
    out: dict[str, str] = {}
    for name, col in (columns or {}).items():
        path = f"{prefix}{name}"
        rn = getattr(col, "remote_name", None)
        if rn:
            out[path] = str(rn)
        fields = getattr(col, "fields", None)
        if fields:
            out.update(remote_names(fields, path + "."))
    return out


def is_remote(ctx: ServiceContext, ref: str) -> bool:
    """True when ``ref``'s layout answers counts remotely (capability ``count`` without ``scan``)."""
    try:
        t = ctx.table(ref)
        layout = ctx.plugin("layout", t.layout)
    except Exception:  # noqa: BLE001 - unknown tables are the reader's to report
        return False
    caps = set(getattr(layout, "capabilities", ()) or ())
    return "count" in caps and "scan" not in caps


def remote_witness(ctx: ServiceContext, req: WitnessRequest) -> WitnessResponse:
    """The remote witness: one independent count request (see the module docstring)."""
    t = ctx.table(req.table)
    layout = ctx.plugin("layout", t.layout)
    if req.grain not in (None, "row", "rows") or req.group_by:
        return WitnessResponse(total_method="unknown",
                               reason=f"{REMOTE_REASON}s count rows only (grain {req.grain!r}, groups {req.group_by})")
    try:
        predicate = from_json(req.predicate) if req.predicate else None
    except Exception as exc:  # noqa: BLE001
        return WitnessResponse(total_method="unknown", reason=f"predicate: {exc}")
    reason = REMOTE_REASON
    ceiling = leakage_conjunct(ctx, t)
    if ceiling is not None:
        # the tool's count is asked with the evidence ceiling injected (counts: inject_filter): so is this one
        from ...predicate import And

        parts = () if predicate is None else (predicate.preds if isinstance(predicate, And) else (predicate,))
        predicate = And((*parts, ceiling)) if parts else ceiling     # flat: each conjunct compiles on its own
        reason = f"{REMOTE_REASON} (evidence ceiling {ceiling.value})"
    kwargs: dict[str, Any] = {"predicate": predicate, "budget": t.descriptor.budget,
                              **_compile_kwargs(layout.count, t)}
    try:
        total = layout.count(layout_spec(t), **kwargs)
    except Exception as exc:  # noqa: BLE001 - an outage leaves the count unknown, never zero
        return WitnessResponse(total_method="unknown", reason=f"{REMOTE_REASON} failed: {exc}"[:500])
    if total is None:
        return WitnessResponse(total_method="unknown",
                               reason="the bound predicate cannot be expressed as a count request to the source")
    return WitnessResponse(total=int(total), total_method="scan", reason=reason, scanned_bytes=0,
                           as_of=_release(layout, t))


def _release(layout: Any, t: Any) -> str | None:
    """The source's data release (CT.gov ``dataTimestamp``), cached by the layout; None when it has none or the
    request fails (provenance then names no release rather than a guess)."""
    if not callable(getattr(layout, "release", None)):
        return None
    from ...plugins.layouts.live_api import RELEASE_TTL_S, Budget

    try:
        got = layout.release(layout_spec(t), Budget.of(t.descriptor.budget), max_age_s=RELEASE_TTL_S)
    except Exception:  # noqa: BLE001 - provenance only: the count stands without a release
        return None
    return str(got) if got else None


def leakage_conjunct(ctx: ServiceContext, t: Any) -> Any:
    """``available_at <= ceiling`` for a source whose ``leakage.counts`` is ``inject_filter`` when the run has
    that source's ceiling (``ceiling_from``: ``data.leakage.ceiling``, or ``VBT_LITERATURE_MAXDATE`` for PubMed,
    whose server bounds every search by it), else None. The tool's own count is bounded the same way (the
    overlay's ``leakage_filter``, or the server itself): without it here, every count under a ceiling
    contradicted the witness (a real Germany phase-3 count: 3,846 under a 2017-12-31 ceiling against 4,700;
    PubMed 'PCSK9 AND evolocumab' 1,003 by 2025/01/31 against 1,212)."""
    spec = getattr(t.descriptor, "leakage", None)
    if spec is None or getattr(spec, "counts", None) != "inject_filter":
        return None
    from ...predicate import Cmp

    ceiling = leakage_ceiling(ctx, t)                  # a partial ceiling counts as its earliest day, as upstream's
    return None if ceiling is None else Cmp(str(spec.available_at), "<=", ceiling.isoformat())


def leakage_bounds(ctx: ServiceContext, t: Any) -> list[Any]:
    """The conjuncts a native read under the ceiling sends: ``available_at <= ceiling``, and with ``rows: withhold``
    also ``changed_at <= ceiling`` (a row updated after it would be withheld: sending the bound fills the page with
    rows that can be returned, and the count counts exactly those). Empty without a ceiling."""
    spec = getattr(t.descriptor, "leakage", None)
    ceiling = leakage_ceiling(ctx, t)
    if spec is None or ceiling is None:
        return []
    from ...predicate import Cmp

    out = [Cmp(str(spec.available_at), "<=", ceiling.isoformat())]
    if spec.changed_at and spec.rows == "withhold":
        out.append(Cmp(str(spec.changed_at), "<=", ceiling.isoformat()))
    return out


def leakage_ceiling(ctx: ServiceContext, t: Any) -> Any:
    """The ceiling (a date) that bounds ``t``'s source in this run, or None."""
    spec = getattr(t.descriptor, "leakage", None)
    if spec is None:
        return None
    from ...gateway.leakage import ceiling_for

    return ceiling_for(spec, ctx.settings)


def witness(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = WitnessRequest.model_validate(dict(payload))
    if is_remote(ctx, req.table):
        return remote_witness(ctx, req).model_dump(mode="json")
    try:
        reader = ctx.reader(req.table)
    except TableUnavailable:
        raise
    except ServiceError as exc:
        return WitnessResponse(total_method="unknown", reason=str(exc)).model_dump(mode="json")
    settings = ctx.settings.witness
    group_by = list(req.group_by)
    if not group_by:
        for o in req.order:
            for w in o.within:
                if w not in group_by:
                    group_by.append(w)
    grains: dict[str, Any] = dict(req.grains)
    grain = None if req.grain in ("row", "rows") else req.grain   # the reserved row grain counts rows
    if grain:
        grains.setdefault("__grain__", grain)
    k = req.k if settings.topk and req.order else None
    try:
        agg = reader.aggregate(req.predicate, order=[o.model_dump() for o in req.order], k=k, group_by=group_by,
                               key=req.key, distinct=req.distinct, grains=grains,
                               key_set_max=req.key_set_max if req.key_set_max is not None else settings.max_key_set,
                               one_to_many=bool(req.key), params=req.params, unknown_columns=req.unknown_columns,
                               budget_bytes=req.budget_bytes)
    except BudgetExceeded as exc:
        return WitnessResponse(total_method="unknown", reason=f"over budget: {exc.reason}").model_dump(mode="json")
    except UnboundParameter as exc:
        return WitnessResponse(total_method="unknown", reason=str(exc)).model_dump(mode="json")
    except FormatError:
        raise
    st = agg.stats
    total = agg.distinct_counts.pop("__grain__") if grain else st.total
    resp = WitnessResponse(
        total=total, total_method="footer" if st.footer else ("index" if st.used_sidecar else "scan"),
        topk=agg.topk if k else [],
        key_set=agg.key_set, distinct=agg.distinct, excluded_unknown=dict(st.excluded_unknown),
        excluded_not_applicable=dict(st.excluded_not_applicable), unknown_total=st.unknown_total,
        distinct_counts=agg.distinct_counts, group_totals=agg.group_totals, one_to_many=agg.one_to_many,
        scanned_bytes=st.scanned_bytes,
        reason=None if all(agg.distinct_complete.values()) else "distinct values truncated at the value cap")
    return resp.model_dump(mode="json")


def pivot_rows(rows: Sequence[Mapping[str, Any]], index: Sequence[str], name_column: str,
               value_column: str) -> list[dict[str, Any]]:
    """Long ``(index..., name, value)`` rows -> one row per index with one column per name (first value
    kept; a second, different value for one name is listed under ``_conflicts``)."""
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        key = tuple(r.get(c) for c in index)
        row = out.setdefault(key, {c: r.get(c) for c in index})
        name = r.get(name_column)
        if name is None:
            continue
        value = r.get(value_column)
        if name in row and row[name] != value:
            row.setdefault("_conflicts", {}).setdefault(str(name), [row[name]]).append(value)
            continue
        row[str(name)] = value
    return [out[k] for k in sorted(out, key=lambda k: tuple(str(x) for x in k))]


def remote_failure(ref: str, exc: Any) -> GatewayError:
    """The typed error of a failed live request (``RemoteError``): a 404 on a path the filter filled
    (``studies/{studyId}/...``) is ``not_found`` naming that column and value; a 400 is the source rejecting
    the arguments; anything else (429 after the retries, 5xx, a transport error) is ``source_error``."""
    status = getattr(exc, "status", None)
    filled = dict(getattr(exc, "filled", None) or {})
    if status == 404 and filled:
        column, value = next(iter(filled.items()))
        return GatewayError(ErrorKind.not_found, f"{ref}: {column} {value!r} is not known to the source (HTTP 404)",
                            argument=column, value=value, payload={"table": ref, "http_status": 404})
    if status == 400:
        return GatewayError(ErrorKind.invalid_argument, f"{ref}: the source rejected the request: {exc}"[:500],
                            payload={"table": ref, "http_status": 400})
    return GatewayError(ErrorKind.source_error, f"{ref}: {exc}"[:500], payload={"table": ref, "http_status": status},
                        retryable="later" if status in (None, 429) or (status or 0) >= 500 else None)


def remote_kinds(columns: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """``{column path: kind}`` for the request compiler (``rest_json``): ``integer`` for counts (a strict bound is
    sent as the next whole number), ``text`` for text matched by the source's engine (never sent as equality)."""
    out: dict[str, str] = {}
    for name, col in (columns or {}).items():
        path = f"{prefix}{name}"
        role = getattr(col, "role", None)
        if role == "count":
            out[path] = "integer"
        elif role == "text" and getattr(col, "remote_name", None):
            out[path] = "text"
        fields = getattr(col, "fields", None)
        if fields:
            out.update(remote_kinds(fields, path + "."))
    return out


def _compile_kwargs(fn: Any, t: Any) -> dict[str, Any]:
    """The descriptor facets a layout's ``request``/``count`` takes: ``remote_names`` and ``kinds``."""
    params = inspect.signature(fn).parameters
    out: dict[str, Any] = {}
    columns = getattr(t.spec, "columns", {}) or {}
    names = remote_names(columns)
    if names and "remote_names" in params:
        out["remote_names"] = names
    kinds = remote_kinds(columns)
    if kinds and "kinds" in params:
        out["kinds"] = kinds
    return out


def _conjuncts(p: Any) -> list[Any]:
    from ...predicate import And

    return [] if p is None else (list(p.preds) if isinstance(p, And) else [p])


def _and(parts: Sequence[Any]) -> Any:
    from ...predicate import And

    return None if not parts else (parts[0] if len(parts) == 1 else And(tuple(parts)))


def requested_keys(predicate: Any, key: Sequence[str], cap: int = 1000) -> list[tuple[Any, ...]] | None:
    """The key tuples a predicate asks for when it consists only of ``Eq``/``In`` on the key columns and fixes
    every one of them (``nctId in [a, b]``; ``studyId = s and patientId = p``), else None."""
    from ...predicate import Eq, In

    fixed: dict[str, list[Any]] = {}
    for p in _conjuncts(predicate):
        col = getattr(p, "column", None)
        if col not in key or col in fixed:
            return None
        if isinstance(p, Eq):
            fixed[col] = [p.value]
        elif isinstance(p, In):
            fixed[col] = list(p.values)
        else:
            return None
    if not key or set(fixed) != set(key):
        return None
    out: list[tuple[Any, ...]] = [()]
    for col in key:
        out = [(*k, v) for k in out for v in fixed[col]]
        if len(out) > cap:
            return None
    return list(dict.fromkeys(out))


def _projection(t: Any, layout: Any, columns: Sequence[str], predicate: Any) -> list[str]:
    """The paths a live read keeps: the requested ones plus the key, the record version and the evidence dates
    (provenance and the ceiling need them); none requested reads every column, except on a layout whose column
    reads reserve memory (``reads_columns``: SOMA), which reads the declared columns."""
    from ...predicate import columns as predicate_columns

    want = list(columns)
    if not want:
        if not getattr(layout, "reads_columns", False):
            return []
        want = list(getattr(t.spec, "columns", {}) or {})
    extra = [c for c in t.spec.key.columns if not str(c).endswith("#")]
    if t.spec.key.version:
        extra.append(str(t.spec.key.version))
    if getattr(t.spec.key, "aliases", None):
        extra.append(str(t.spec.key.aliases))         # a requested alias is found on the record that lists it
    spec = getattr(t.descriptor, "leakage", None)
    if spec is not None:
        extra.extend(str(c) for c in (spec.available_at, spec.changed_at) if c)
    if getattr(layout, "reads_columns", False) and predicate is not None:
        extra.extend(sorted(predicate_columns(predicate)))    # a residual is evaluated on the rows read
    return list(dict.fromkeys([*want, *extra]))


_STUDY_RELEASES: dict[str, tuple[float, str | None]] = {}


def _per_record_releases(ctx: ServiceContext, t: Any, rows: Sequence[Any], predicate: Any,
                         max_records: int = 5) -> tuple[str | None, dict[str, str]]:
    """The release of the records the rows depend on (``release.per: {table, column}``: a cBioPortal study's
    ``importDate``): ``(table ref, {record key: release})``, read from that table (cached for
    ``live_api.RELEASE_TTL_S``); ``(None, {})`` when the source declares none. At most ``max_records`` records
    are read; a failed read leaves that record out (provenance only)."""
    import time

    from ...plugins.layouts.live_api import RELEASE_FAILURE_TTL_S, RELEASE_TTL_S, Budget, RemoteError, fetch_all
    from ...predicate import Eq, In

    per = getattr(t.descriptor.release, "per", None) or {}
    if not per.get("table") or not per.get("column"):
        return None, {}
    ref = f"{t.physical.source}.{per['table']}"
    try:
        pt = ctx.table(ref)
    except Exception:  # noqa: BLE001 - an undeclared per table names no release
        return None, {}
    key = [c for c in pt.spec.key.columns if not str(c).endswith("#")]
    if len(key) != 1:
        return None, {}
    kcol, vcol = key[0], str(per["column"])
    if str(pt.physical) == str(t.physical) and rows:
        # the rows are the records themselves (without rows, a release request reads the records the predicate names)
        return ref, {str(r.get(kcol)): str(r.get(vcol)) for r in rows
                     if isinstance(r, Mapping) and r.get(kcol) not in (None, "") and r.get(vcol) not in (None, "")}
    ids: list[str] = []
    for p in _conjuncts(predicate):
        if isinstance(p, Eq) and p.column == kcol:
            ids.append(str(p.value))
        elif isinstance(p, In) and p.column == kcol:
            ids.extend(str(v) for v in p.values)
    ids.extend(str(r.get(kcol)) for r in rows if isinstance(r, Mapping) and r.get(kcol) not in (None, ""))
    ids = list(dict.fromkeys(ids))
    if not ids and not rows:
        # an empty answer depends on the record its filter names through a reference (molecular_data's
        # molecularProfileId -> molecular_profile.studyId): its release, not the fetch time (LIVE3-11)
        ids = _ids_through_refs(ctx, t, predicate, per["table"], kcol)
    if not ids:
        return ref, {}
    if len(ids) > max_records:
        listed = _listed_releases(ctx, pt, kcol, vcol)   # a listing of 20 studies: one read of /studies
        return ref, {rid: listed[rid] for rid in ids if rid in listed}
    layout = ctx.plugin("layout", pt.layout)
    out: dict[str, str] = {}
    down = False
    for rid in ids:
        cache = f"{ref}:{rid}"
        hit = _STUDY_RELEASES.get(cache)
        if hit is not None and time.monotonic() - hit[0] <= RELEASE_TTL_S:
            if hit[1]:
                out[rid] = hit[1]
            continue
        if down:
            continue
        try:
            got = fetch_all(layout, layout_spec(pt), Eq(kcol, rid), Budget.of(pt.descriptor.budget),
                            projection=[kcol, vcol], max_rows=1)
        except Exception as exc:  # noqa: BLE001 - provenance only: the rows stand without a release
            # not repeated for RELEASE_FAILURE_TTL_S (the entry reads as fresh that long), and a source that does not
            # answer at all is not asked for the other records of this call either (RR-3)
            _STUDY_RELEASES[cache] = (time.monotonic() - RELEASE_TTL_S + RELEASE_FAILURE_TTL_S, None)
            down = down or (isinstance(exc, RemoteError) and exc.status is None)
            continue
        value = next((str(r.get(vcol)) for r in got["rows"] if isinstance(r, Mapping) and r.get(vcol)), None)
        _STUDY_RELEASES[cache] = (time.monotonic(), value)
        if value:
            out[rid] = value
    return ref, out


_LISTED_RELEASES: dict[str, tuple[float, dict[str, str]]] = {}


def _listed_releases(ctx: ServiceContext, pt: Any, kcol: str, vcol: str) -> dict[str, str]:
    """``{record key: release}`` of every record of a listable live table (``/studies`` holds each study's
    importDate), read once within the source's budget and kept for ``RELEASE_TTL_S``; {} when it cannot be read
    (a path with placeholders, a failed or cut read)."""
    import time

    from ...plugins.layouts.live_api import RELEASE_TTL_S, Budget, fetch_all

    ref = str(pt.physical)
    hit = _LISTED_RELEASES.get(ref)
    if hit is not None and time.monotonic() - hit[0] <= RELEASE_TTL_S:
        return hit[1]
    if "{" in str(pt.spec.path or ""):
        return {}
    try:
        got = fetch_all(ctx.plugin("layout", pt.layout), layout_spec(pt), None, Budget.of(pt.descriptor.budget),
                        projection=[kcol, vcol])
    except Exception:  # noqa: BLE001 - provenance only
        return {}
    if got.get("truncated"):
        return {}
    out = {str(r.get(kcol)): str(r.get(vcol)) for r in got["rows"]
           if isinstance(r, Mapping) and r.get(kcol) not in (None, "") and r.get(vcol) not in (None, "")}
    _LISTED_RELEASES[ref] = (time.monotonic(), out)
    return out


def _ids_through_refs(ctx: ServiceContext, t: Any, predicate: Any, per_table: str, kcol: str) -> list[str]:
    """The ``per_table`` keys a filter names through a reference: an ``Eq`` on a column that refers to another
    table's key (``molecularProfileId`` -> ``molecular_profile``), whose record holds a column referring to
    ``per_table.kcol`` (``studyId`` -> ``study.studyId``); that record is read by its key."""
    from ...predicate import Eq

    want = f"{per_table}.{kcol}"
    for p in _conjuncts(predicate):
        if not isinstance(p, Eq):
            continue
        col = (t.columns or {}).get(str(p.column))
        ref = getattr(col, "ref", None) if col is not None else None
        if not isinstance(ref, str) or "." not in ref:
            continue
        rtable, _, rkey = ref.partition(".")
        try:
            rt = ctx.table(f"{t.physical.source}.{rtable}")
        except Exception:  # noqa: BLE001 - an undeclared referenced table names nothing
            continue
        link = next((n for n, c in (rt.columns or {}).items() if getattr(c, "ref", None) == want), None)
        if link is None:
            continue
        try:
            got = live_find(ctx, {"table": str(rt.ref), "predicate": to_json(Eq(rkey, p.value)),
                                  "columns": [link], "limit": 1})
        except Exception:  # noqa: BLE001 - provenance only
            continue
        ids = [str(r.get(link)) for r in got.get("rows") or [] if isinstance(r, Mapping) and r.get(link)]
        if ids:
            return ids
    return []


def live_find(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``{table, predicate?, columns?, limit?}`` -> ``{rows, total, as_of, fetched_at, truncated, pages,
    source_updated, record_versions, leakage, withheld_keys}``.

    A ``limit`` sizes the pages and stops the reading once that many rows are in hand (a find of 2 trials
    reads one page of 2, not ten pages of 100); a pivoted table is read whole (its long rows cannot be cut).
    When the rows are cut and the page carries no total, one count request (the layout's ``count``) supplies
    it. ``as_of`` is the source's data release when the layout declares one (CT.gov ``dataTimestamp``, the
    Census release ``stable`` names, read at most every ``RELEASE_TTL_S`` seconds; a failed release request
    leaves the page's own time, never fails the rows), or the release of the one record the rows depend on
    (``release.per``: a cBioPortal study's ``importDate``, also recorded as that record's version); ``fetched_at``
    is the page's own time. ``columns`` keep their dotted paths, plus the key, the record version and the dates
    the ceiling is checked on.

    **Evidence ceiling**: a source with ``leakage`` is read under its ceiling (``ceiling_from``) as the tool's
    own path is: ``available_at <= ceiling`` (and ``changed_at <= ceiling`` under ``rows: withhold``,
    :func:`leakage_bounds`) are conjuncts of the request and of the count (a find of recruiting trials under
    2017-12-31 returned trials first posted in 2025 and the uncapped total, 64,639), unless the predicate names the
    keys (a lookup): those records are read and withheld visibly. Every row is then checked as T1 checks upstream
    rows (``rows: withhold|redact|stamp``; a bound the source cannot take is checked here) and ``leakage`` reports
    ``{ceiling, withheld, redacted, unchecked}``. A failed request is a typed error (:func:`remote_failure`),
    never an empty answer."""
    from ...plugins.layouts.live_api import Budget, RecordVersions, RemoteError, fetch_all, project_row

    ref = str(payload["table"])
    t = ctx.table(ref)
    layout = ctx.plugin("layout", t.layout)
    if "live" not in (getattr(layout, "capabilities", ()) or ()):
        raise ServiceError(f"{ref} is not a live table (layout {t.layout!r})")
    predicate = from_json(payload["predicate"]) if payload.get("predicate") else None
    columns = list(payload.get("columns") or [])
    lspec = layout_spec(t)
    kwargs = _compile_kwargs(layout.request, t)
    pivot = t.spec.pivot
    key = [c for c in t.spec.key.columns if not str(c).endswith("#")]
    asked = requested_keys(predicate, key) if pivot is None else None
    spec = getattr(t.descriptor, "leakage", None)
    ceiling = leakage_ceiling(ctx, t)
    bounds = leakage_bounds(ctx, t) if asked is None else []
    sent = _and([*_conjuncts(predicate), *bounds])
    if pivot is not None:
        # the source holds long rows: only conjuncts on the index columns can be sent; the rest is applied
        # to the pivoted rows
        from ...predicate import columns as predicate_columns

        sent = _and([p for p in _conjuncts(sent) if predicate_columns(p) <= set(pivot.index)])
    limit = payload.get("limit")
    cut = int(limit) if limit is not None and pivot is None else None
    budget = Budget.of(t.descriptor.budget).pages_of(cut)
    projection = [] if pivot else _projection(t, layout, columns, sent)
    try:
        got = fetch_all(layout, lspec, sent, budget, projection=projection, max_rows=cut, **kwargs)
    except RemoteError as exc:
        raise remote_failure(ref, exc) from None
    release = _release(layout, t)                      # provenance only: a failed release request keeps the rows
    rows = list(got["rows"])
    if pivot is None and len(key) == 1:
        # a listing of bare keys (E-utilities esearch idlist: ["28304224"]) is one row per key
        rows = [r if isinstance(r, Mapping) else {key[0]: r} for r in rows]
    if pivot is not None:
        from ...predicate import evaluate

        rows = pivot_rows(rows, pivot.index, pivot.name_column, pivot.value_column)
        if predicate is not None:
            rows = [r for r in rows if evaluate(predicate, r) is True]
        if columns:
            rows = [{k: v for k, v in r.items() if k in columns or k in pivot.index} for r in rows]
    leakage: dict[str, Any] | None = None
    withheld_keys: list[Any] = []
    if spec is not None and ceiling is not None:
        from ...gateway.fields import get_path
        from ...gateway.leakage import withhold_rows

        before = [tuple(get_path(r, c) for c in key) for r in rows if isinstance(r, Mapping)]
        rows, counts = withhold_rows(rows, spec, ceiling)
        kept = {tuple(get_path(r, c) for c in key) for r in rows if isinstance(r, Mapping)}
        withheld_keys = [k[0] if len(k) == 1 else dict(zip(key, k)) for k in before if k not in kept]
        leakage = {"ceiling": ceiling.isoformat(), "ceiling_from": spec.ceiling_from,
                   "sent": bool(bounds), **counts,
                   # the dates the request and the count were bounded on (available, and changed under withhold)
                   "bounded_on": [str(getattr(b, "column", "")) for b in bounds]}
    if columns and pivot is None:
        # what was asked for, with the key and the record version (the dates the ceiling needed are dropped)
        keep = [*columns, *key, *([str(t.spec.key.version)] if t.spec.key.version else [])]
        rows = [project_row(r, keep) if isinstance(r, Mapping) else r for r in rows]
    flags: dict[str, Any] = {"source_updated": [], "versions": {}}
    version = t.spec.key.version
    store = None
    if version and len(key) == 1:
        store = RecordVersions(Path(ctx.settings.cache_dir) / t.physical.source / "record_versions.json")
        flags = store.observe_rows(str(t.physical), rows, key[0], version)
    versions = {str(t.physical): flags["versions"]} if flags["versions"] else {}
    per_ref, per = _per_record_releases(ctx, t, rows, predicate)
    if per_ref is not None and per and per_ref != str(t.physical):
        if store is None:
            store = RecordVersions(Path(ctx.settings.cache_dir) / t.physical.source / "record_versions.json")
        for rid, v in per.items():
            if store.observe(per_ref, rid, v) == "source_updated":
                flags["source_updated"].append(f"{per_ref}:{rid}")
        versions[per_ref] = dict(per)
    if store is not None:
        store.save()
    if len(per) == 1 and not release:
        release = next(iter(per.values()))            # the one record the rows depend on names the release
    if got["total"] is not None and pivot is None and not withheld_keys:
        total = got["total"]
    elif got["truncated"]:
        total = None
    else:
        total = len(rows)
    if total is None and got["truncated"] and pivot is None and "count" in (getattr(layout, "capabilities", ()) or ()):
        try:
            total = layout.count(lspec, predicate=sent, budget=t.descriptor.budget, **_compile_kwargs(layout.count, t))
        except RemoteError:
            total = None                               # the rows stand; the total stays unknown
    shown = rows[: int(limit)] if limit is not None else rows
    return {"table": ref, "rows": shown, "total": total, "total_method": "unknown" if total is None else "remote",
            "as_of": release or got["as_of"], "fetched_at": got["as_of"],
            "truncated": bool(got["truncated"]) or len(shown) < len(rows),
            # every matching row was read and the answer is cut to ``limit`` afterwards (a SOMA read is one
            # unpaged request): the rows are the first ``limit`` of the whole match, not a page prefix
            "cut_after_read": not got["truncated"] and len(shown) < len(rows),
            "pages": got["pages"], "requests": budget.used, "source_updated": flags["source_updated"],
            "record_versions": versions, "leakage": leakage, "withheld_keys": withheld_keys,
            "requested_keys": [k[0] if len(k) == 1 else dict(zip(key, k)) for k in asked] if asked is not None
            else None}


VERBS = {VERB_WITNESS: witness, LIVE_FIND: live_find}
