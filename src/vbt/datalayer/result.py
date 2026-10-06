"""Successful gateway results and the ``_vbt`` header (§12.2). No pyarrow.

Every gateway result is a :class:`DataResult`: the payload with the ``_vbt`` header as the
**first** JSON key (or the first line of a text result), its status, the provenance record
and the **unshrunk** payload (``full_text``) that the runtime spills before the
model-facing size cap applies (rev 2).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Mapping, cast

__all__ = [
    "HEADER_KEY", "HEADER_VERSION", "HEADER_MAX_CHARS", "ResultStatus", "Coverage", "RESULT_STATUSES",
    "COVERAGE_VALUES", "Header", "inject_header", "header_line", "DataResult", "build", "dumps",
]

HEADER_KEY = "_vbt"
HEADER_VERSION = 1
HEADER_MAX_CHARS = 1200

ResultStatus = Literal["ok", "partial", "empty", "empty_unverified"]
Coverage = Literal["covered", "unknown", "not_covered", "partial_unknown", "censored"]
RESULT_STATUSES: tuple[str, ...] = ("ok", "partial", "empty", "empty_unverified")
COVERAGE_VALUES: tuple[str, ...] = ("covered", "unknown", "not_covered", "partial_unknown", "censored")

#: Long strings shortened (after notes) when a header does not fit its budget.
_SHORTEN_AT = 240


def dumps(obj: Any) -> str:
    """The JSON rendering used for every model-facing and spilled payload."""
    return json.dumps(obj, default=str, ensure_ascii=False)


@dataclass
class Header:
    """The ``_vbt`` header. ``None`` fields are omitted from :meth:`to_dict`; field order is the
    rendered key order."""

    status: ResultStatus
    v: int = HEADER_VERSION
    source: str | None = None                          # "open_targets@25.09"
    tables: list[str] | None = None
    key: list[str] | None = None
    returned: int | None = None
    total: int | None = None
    total_method: str | None = None                    # witness_scan | data_child | upstream | upstream_upper_bound | unknown
    truncated: bool | None = None
    order: str | None = None                           # "phase desc (verified)", "score desc within sourceDatabase (verified)"
    grains: dict[str, dict[str, int | None]] | None = None   # {grain: {returned, total}}
    scope: dict[str, Any] | None = None                # fixed and listed scope values
    resolved: dict[str, str] | None = None             # {"target_id": "PCSK9 -> ENSG... (label_exact:approvedSymbol)"}
    resolution_summary: dict[str, Any] | None = None
    excluded_unknown: dict[str, int] | None = None
    excluded_not_applicable: dict[str, int] | None = None
    excluded_negated: int | dict[str, int] | None = None
    excluded: dict[str, int] | None = None             # rows failing a re-applied bound argument (T4)
    withheld: dict[str, int] | None = None             # {leakage, phantom, duplicates}
    family_rows: dict[str, int] | None = None
    not_found_items: list[Any] | None = None
    pooled_over: list[str] | None = None
    removed_fields: dict[str, str] | None = None
    trimmed: dict[str, dict[str, int]] | None = None   # {path: {returned, total}}
    undefined: dict[str, Any] | None = None            # anchor arguments without a value in the table
    coverage: Coverage | None = None
    coverage_statement: str | None = None
    evidence: str | None = None                        # the table's evidence_nature caveat
    served_by: str | None = None                       # upstream | derived | repaired
    hash_seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)   # further keys, rendered before notes
    notes: list[str] = field(default_factory=list)
    cite: str | None = None
    prov: str | None = None

    def to_dict(self, max_chars: int = HEADER_MAX_CHARS) -> dict[str, Any]:
        """The header as a dict without ``None`` fields, at most ``max_chars`` of JSON where possible:
        notes are dropped from the end first (replaced by one count note), then long strings
        are shortened."""
        out: dict[str, Any] = {}
        for name in ["v"] + [f.name for f in fields(self) if f.name != "v"]:
            value = getattr(self, name)
            if name == "extra":
                out.update((k, copy.deepcopy(v)) for k, v in value.items() if v is not None and k not in out)
                continue
            if value is None or (name == "notes" and not value):
                continue
            out[name] = copy.deepcopy(value)
        if len(dumps(out)) <= max_chars:
            return out
        notes = list(out.get("notes") or [])
        dropped = 0
        while notes and len(dumps(out)) > max_chars:
            notes.pop()
            dropped += 1
            out["notes"] = notes + [f"+{dropped} more notes in provenance"]
        if len(dumps(out)) > max_chars:
            for k, v in list(out.items()):
                if isinstance(v, str) and len(v) > _SHORTEN_AT and k not in ("status", "prov"):
                    out[k] = v[:_SHORTEN_AT - 1] + "…"
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Header":
        """The inverse of :meth:`to_dict`: keys that are not fields go to ``extra``."""
        names = {f.name for f in fields(cls)} - {"extra"}
        extra = dict(data.get("extra") or {})
        extra.update((k, v) for k, v in data.items() if k not in names and k != "extra")
        return cls(**{k: v for k, v in data.items() if k in names}, extra=extra)


def _header_dict(header: Header | Mapping[str, Any], max_chars: int = HEADER_MAX_CHARS) -> dict[str, Any]:
    if isinstance(header, Header):
        return header.to_dict(max_chars)
    return Header.from_dict(header).to_dict(max_chars) if "status" in header else dict(header)


def header_line(header: Header | Mapping[str, Any]) -> str:
    """The one-line header of a text result:
    ``[vbt-data dp_7c1e0942 · ok · open_targets@25.09/known_drug · 20 of 61 · ranked phase desc]``."""
    h = _header_dict(header)
    parts = [f"vbt-data {h['prov']}" if h.get("prov") else "vbt-data", str(h.get("status", "?"))]
    where = h.get("source") or ""
    if h.get("tables"):
        where = f"{where}/{','.join(h['tables'])}" if where else ",".join(h["tables"])
    if where:
        parts.append(where)
    if h.get("returned") is not None:
        total = h.get("total")
        parts.append(f"{h['returned']} of {total if total is not None else '?'}")
    if h.get("order"):
        parts.append(f"ranked {h['order']}")
    if h.get("coverage") and h.get("status") in ("empty", "empty_unverified"):
        parts.append(f"coverage {h['coverage']}")
    return "[" + " · ".join(parts) + "]"


def inject_header(obj: Any, header: Header | Mapping[str, Any], *, max_chars: int = HEADER_MAX_CHARS) -> Any:
    """Return ``obj`` with the header first: a dict gets ``_vbt`` as its first key (replacing an
    existing one), a list is wrapped as ``{"_vbt": ..., "rows": [...]}``, text gets the
    header line as its first line. Other values are wrapped as ``{"_vbt": ..., "value": v}``."""
    h = _header_dict(header, max_chars)
    if isinstance(obj, Mapping):
        out = {HEADER_KEY: h}
        out.update((k, v) for k, v in obj.items() if k != HEADER_KEY)
        return out
    if isinstance(obj, list):
        return {HEADER_KEY: h, "rows": obj}
    if isinstance(obj, str):
        return f"{header_line(h)}\n{obj}" if obj else header_line(h)
    return {HEADER_KEY: h, "value": obj}


@dataclass
class DataResult:
    """A gateway success. ``obj`` already carries the header; ``text`` is what the model sees;
    ``full_text`` is the unshrunk payload for the runtime spill."""

    obj: Any
    text: str
    status: ResultStatus
    provenance: Any = None                             # record.DataProvenance
    full_text: str | None = None
    header: dict[str, Any] | None = None

    #: Duck-typing marker the runtime checks (no import of this module needed there).
    is_data_result = True

    def __str__(self) -> str:
        return self.text

    @classmethod
    def build(cls, obj: Any, header: Header | Mapping[str, Any], provenance: Any = None,
              full_obj: Any = None, *, max_chars: int = HEADER_MAX_CHARS) -> "DataResult":
        """Inject the header into ``obj`` (and into ``full_obj``, the unshrunk payload, when the
        gateway shrank ``obj``) and render both texts."""
        h = _header_dict(header, max_chars)
        shown = inject_header(obj, h)
        text = shown if isinstance(shown, str) else dumps(shown)
        if full_obj is None:
            full_text = text
        else:
            full = inject_header(full_obj, h)
            full_text = full if isinstance(full, str) else dumps(full)
        status = cast(ResultStatus, h.get("status") or "ok")
        return cls(obj=shown, text=text, status=status, provenance=provenance, full_text=full_text, header=h)


build = DataResult.build
