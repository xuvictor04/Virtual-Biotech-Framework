"""``rest_json``: JSON pages of a REST API (phase 4, F20). No pyarrow at import.

The format of ``live_api`` tables (ClinicalTrials.gov v2, E-utilities, cBioPortal). It does two jobs:

* **compile** a :mod:`~vbt.datalayer.predicate` IR into request parameters. ``schema`` is the request
  map (a mapping, not an Arrow schema; the ``live_api`` layout builds it from the layout options and the
  descriptor columns' ``remote_name``)::

      {"filters":     {<column>: {param, join: ",", quote: none|essie,
                                  range: {lo, hi, lo_default, hi_default, format, extra}}},
       "remote_names": {<column>: "AREA[OverallStatus]"},     # Essie fields (CT.gov)
       "essie_param": "filter.advanced",                      # where Essie fragments go (ANDed)
       "text_params": {<column> | "*": <param>}}              # TextMatch: the source's own search engine
                                                              # (a TextMatch on "@<param>" sets <param>)

  ``Eq``/``In`` (and ``Contains`` on a list column, and an ``Or`` of those on one column) on a filtered column
  become one parameter (values joined); ``Range``/``Cmp`` with a ``range`` entry become the lo/hi parameters; a column with a
  ``remote_name`` becomes an Essie fragment (``AREA[Phase](PHASE2 OR PHASE3)``, ``AREA[StartDate]RANGE[2020-01-01,MAX]``)
  with every literal quoted by :func:`essie_quote`. Anything else (``Not``, ``Or`` across columns,
  ``IsNull``, an unmapped column) is returned as the residual, evaluated on the returned rows; a
  residual makes a count request impossible (the remote witness then says ``unknown``).
* **decode** a page (:meth:`RestJsonFormat.decode_page`): rows, the upstream total, the next page token
  (``next_path``), page index (``page_number_param``) or offset, and ``as_of`` from the payload
  (``as_of_path``) or the ``Date`` header.

``scan`` reads recorded responses saved as ``.json`` files (fixtures, replays): each file is one page.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, ClassVar, Iterator, Mapping, Sequence

from ...predicate import All, And, Any as AnyP, Cmp, Contains, Eq, In, Or, Predicate, Range, TextMatch
from ..base import Fragment, FormatError, FragmentStats, Page, PluginBase
from ..registry import register

__all__ = ["RestJsonFormat", "essie_quote", "eutils_quote", "iso_date", "jp_values"]

_TOKEN = re.compile(r"\.([^.\[\]]+)|\[\s*(\*|-?\d+)\s*\]")
_ESSIE_PLAIN = re.compile(r"^[A-Za-z0-9_\-.]+$")
_ESSIE_WORDS = frozenset({"AND", "OR", "NOT", "RANGE", "AREA", "SEARCH", "EXPANSION", "COVERAGE", "MIN", "MAX"})


def jp_values(obj: Any, path: str | None) -> list[Any]:
    """Values at a ``$.a.b[*].c`` path (``[N]`` and ``[*]`` supported); ``[]`` when absent."""
    if not path:
        return []
    text = str(path).strip()
    text = text[1:] if text.startswith("$") else text
    values = [obj]
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ValueError(f"unsupported JSONPath {path!r}")
        name, idx = m.groups()
        nxt: list[Any] = []
        for v in values:
            if name is not None:
                if isinstance(v, Mapping) and name in v:
                    nxt.append(v[name])
            elif idx == "*":
                if isinstance(v, list):
                    nxt.extend(v)
            elif isinstance(v, list):
                i = int(idx)
                if -len(v) <= i < len(v):
                    nxt.append(v[i])
        values = nxt
        pos = m.end()
    return values


def essie_quote(value: Any) -> str:
    """An Essie literal: plain tokens as they are, everything else double-quoted with ``\\`` and ``"``
    escaped (so ``a) OR (b`` stays one phrase, never an operator)."""
    text = str(value)
    if _ESSIE_PLAIN.fullmatch(text) and text.upper() not in _ESSIE_WORDS:
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def eutils_quote(value: Any) -> str:
    """An E-utilities term literal: double-quoted with internal quotes removed (PubMed has no escape)."""
    return '"' + str(value).replace('"', " ").strip() + '"'


_QUOTES: dict[str, Callable[[Any], str]] = {"none": str, "essie": essie_quote, "eutils": eutils_quote}


def iso_date(value: Any) -> str | None:
    """An ISO-8601 UTC timestamp from a payload date (ISO or RFC 2822 ``Date`` header), else the text."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        dt = None
    if dt is not None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return text


@register
class RestJsonFormat(PluginBase):
    kind: ClassVar[str] = "format"
    name: ClassVar[str] = "rest_json"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"string_compile"})

    # ------------------------------------------------------------------ compile

    def quote(self, value: Any) -> str:
        """An Essie phrase literal: always double-quoted, ``\\`` and ``"`` escaped."""
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def compile(self, predicate: Predicate | None, schema: Any) -> tuple[dict[str, str], Predicate | None]:
        """``(params, residual)`` for ``predicate`` under the request map ``schema`` (module docstring)."""
        spec = dict(schema or {})
        params: dict[str, str] = {}
        essie: list[str] = []
        residual: list[Predicate] = []
        if predicate is None:
            return params, None
        parts = list(predicate.preds) if isinstance(predicate, And) else [predicate]
        for p in parts:
            if not self._one(p, spec, params, essie):
                residual.append(p)
        if essie:
            target = spec.get("essie_param") or "filter.advanced"
            existing = params.get(target)
            joined = " AND ".join(essie)
            params[target] = f"({existing}) AND {joined}" if existing else joined
        if not residual:
            return params, None
        return params, residual[0] if len(residual) == 1 else And(tuple(residual))

    def _one(self, p: Predicate, spec: Mapping[str, Any], params: dict[str, str], essie: list[str]) -> bool:
        filters = dict(spec.get("filters") or {})
        remote = dict(spec.get("remote_names") or {})
        texts = dict(spec.get("text_params") or {})
        if isinstance(p, TextMatch):
            # "@<param>" names the request parameter itself (a free-text argument the source's engine
            # matches, which has no column)
            param = p.column[1:] if str(p.column).startswith("@") else (texts.get(p.column) or texts.get("*"))
            if not param or param in params:
                return False
            params[param] = str(p.text)
            return True
        if isinstance(p, Or):
            # one column equal to (or a list holding) any of several values: the same request as In
            # (phase PHASE2 or PHASE3 -> AREA[Phase](PHASE2 OR PHASE3))
            cols = {getattr(q, "column", None) or getattr(q, "path", None) for q in p.preds}
            if len(cols) != 1 or None in cols or not all(isinstance(q, (Eq, Contains)) for q in p.preds):
                return False
            p = In(str(next(iter(cols))), tuple(q.value for q in p.preds))  # type: ignore[union-attr]
        column = getattr(p, "column", None) or getattr(p, "path", None)
        if isinstance(p, (AnyP, All)):
            return False
        if column is None:
            return False
        base = str(column).replace("[]", "")
        f = filters.get(column) or filters.get(base)
        if f is not None:
            return self._param(p, dict(f), params)
        name = remote.get(column) or remote.get(base)
        if name:
            frag = self._essie(p, str(name))
            if frag is None:
                return False
            essie.append(frag)
            return True
        return False

    def _param(self, p: Predicate, f: Mapping[str, Any], params: dict[str, str]) -> bool:
        quote = _QUOTES.get(str(f.get("quote") or "none"), str)
        if isinstance(p, (Eq, In, Contains)) and f.get("param"):
            values = [p.value] if isinstance(p, (Eq, Contains)) else list(p.values)
            if not values or str(f["param"]) in params:
                return False
            params[str(f["param"])] = str(f.get("join", ",")).join(quote(_text(v)) for v in values)
            return True
        rng = f.get("range")
        if isinstance(p, (Range, Cmp)) and isinstance(rng, Mapping):
            lo = hi = None
            if isinstance(p, Range):
                lo, hi = p.lo, p.hi
            elif p.op in (">=", ">"):
                lo = p.value
            elif p.op in ("<=", "<"):
                hi = p.value
            else:
                return False
            fmt = rng.get("format")
            for key, v in (("lo", lo), ("hi", hi)):
                if v is None:
                    v = rng.get(f"{key}_default")              # APIs that need both bounds (E-utilities)
                if v is not None and rng.get(key):
                    params[str(rng[key])] = _date_text(v, fmt)
            for k, v in (rng.get("extra") or {}).items():
                params[str(k)] = str(v)
            return True
        return False

    def _essie(self, p: Predicate, area: str) -> str | None:
        if isinstance(p, (Eq, Contains)):
            return f"{area}{essie_quote(_text(p.value))}"
        if isinstance(p, In):
            if not p.values:
                return None
            if len(p.values) == 1:
                return f"{area}{essie_quote(_text(p.values[0]))}"
            return f"{area}(" + " OR ".join(essie_quote(_text(v)) for v in p.values) + ")"
        lo = hi = None
        if isinstance(p, Range):
            lo, hi = p.lo, p.hi
        elif isinstance(p, Cmp) and p.op in (">=", ">"):
            lo = p.value
        elif isinstance(p, Cmp) and p.op in ("<=", "<"):
            hi = p.value
        else:
            return None
        return f"{area}RANGE[{_text(lo) if lo is not None else 'MIN'},{_text(hi) if hi is not None else 'MAX'}]"

    # ------------------------------------------------------------------ decode

    def decode_page(self, payload: Any, options: Mapping[str, Any], headers: Mapping[str, str] | None = None, *,
                    offset: int = 0, page_size: int | None = None, page_number: int | None = None) -> Page:
        """One :class:`~vbt.datalayer.plugins.base.Page` of a JSON response (module docstring). The next page
        is the payload's token (``next_path``), the next page index (``page_number_param``: a full page means
        another may follow) or the next offset (``offset_param``)."""
        opts = dict(options or {})
        rows_path = opts.get("rows_path")
        if rows_path:
            got = jp_values(payload, rows_path)
            if "[*]" in str(rows_path):
                rows = list(got)
            elif len(got) == 1 and isinstance(got[0], list):
                rows = list(got[0])
            elif not got:
                rows = []
            else:
                raise FormatError(f"{rows_path} is not a list in the response")
        elif isinstance(payload, list):
            rows = list(payload)
        else:
            rows = [payload] if isinstance(payload, Mapping) else []
        total = _int(next(iter(jp_values(payload, opts.get("total_path"))), None)) if opts.get("total_path") else None
        nxt = None
        if opts.get("next_path"):
            token = next(iter(jp_values(payload, opts.get("next_path"))), None)
            nxt = str(token) if token not in (None, "") else None
        elif opts.get("page_number_param") and page_size:
            index = int(page_number or 0)
            if rows and len(rows) >= page_size and (total is None or (index + 1) * page_size < total):
                nxt = str(index + 1)
        elif opts.get("offset_param") and page_size:
            after = offset + len(rows)
            if rows and len(rows) >= page_size and (total is None or after < total):
                nxt = str(after)
        as_of = None
        if opts.get("as_of_path"):
            as_of = iso_date(next(iter(jp_values(payload, opts["as_of_path"])), None))
        if as_of is None and headers:
            hdr = {str(k).lower(): v for k, v in headers.items()}
            as_of = iso_date(hdr.get("last-modified") or hdr.get("date"))
        if as_of is None:
            as_of = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return Page(rows=rows, total=total, next=nxt, as_of=as_of)

    # ------------------------------------------------------------------ protocol (recorded pages)

    def _load(self, frag: Fragment) -> Any:
        path = frag.uri[len("file://"):] if frag.uri.startswith("file://") else frag.uri
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise FormatError(f"cannot read {frag.uri}: {exc}", fragment=frag.uri) from exc
        if not data.strip():
            raise FormatError(f"{frag.uri} is empty", fragment=frag.uri)
        try:
            return json.loads(data)
        except ValueError as exc:
            raise FormatError(f"{frag.uri} is not JSON: {exc}", fragment=frag.uri) from exc

    def _rows(self, frag: Fragment, options: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        page = self.decode_page(self._load(frag), options or getattr(self, "options", {}) or {})
        return [dict(r) for r in page.rows if isinstance(r, Mapping)]

    def logical_schema(self, frag: Fragment) -> Any:
        import pyarrow as pa

        rows = self._rows(frag)
        return pa.Table.from_pylist(rows).schema if rows else pa.schema([])

    def leaf_path(self, path: str, schema: Any) -> str:
        return str(path).replace("[]", ".list.element")

    def stats(self, frag: Fragment) -> FragmentStats:
        return FragmentStats(rows=len(self._rows(frag)), row_groups=1, columns={}, method="scan")

    def metadata(self, frag: Fragment) -> Mapping[str, str]:
        page = self.decode_page(self._load(frag), getattr(self, "options", {}) or {})
        return {"as_of": page.as_of or "", "total": "" if page.total is None else str(page.total)}

    def scan(self, frags: Sequence[Fragment], *, columns: list[str], predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None) -> Iterator[Any]:
        import pyarrow as pa

        from ...predicate import evaluate

        pages = [self._rows(f) for f in frags]                 # every page decodes, or the scan fails
        for rows in pages:
            if predicate is not None:
                rows = [r for r in rows if evaluate(predicate, r) is True]
            if columns:
                rows = [{c: r.get(c) for c in columns} for r in rows]
            for i in range(0, len(rows), max(1, batch_rows)):
                yield pa.RecordBatch.from_pylist(rows[i:i + batch_rows])

    def to_native(self, table: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in table.to_pylist()]

    def configure(self, options: Mapping[str, Any] | None = None, matrix: Any = None) -> "RestJsonFormat":
        import copy

        other = copy.copy(self)
        other.options = dict(options or {})                  # type: ignore[attr-defined]
        return other


def _text(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _int(v: Any) -> int | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _date_text(v: Any, fmt: str | None) -> str:
    """A date literal for a range parameter (``format`` is a strftime pattern for date values)."""
    if fmt and isinstance(v, str):
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(v, pattern).strftime(fmt)
            except ValueError:
                continue
    return _text(v)
