"""``jsonpath``: the builtin envelope, a result read through the overlay JSONPath subset (F21).

Options (``result.codec_options``; an overlay without a codec gets them from its own result spec,
:func:`~vbt.datalayer.plugins.envelopes.codec_of`)::

    rows:            "$.results" | ["$.up", "$.down"]   paths of row lists ("[*]" paths list items)
    total:           "$.count"                          an integer total the source reports
    found:           "$.found == true"                  a predicate that means found
    not_found_when:  ["$.error =~ '(?i)not found'"]      predicates that mean not found (checked first)
    message:         "$.message"                        the human message (default $.message, $.error)
    errors:          ["$.errors[*]", "$.warnings"]      paths whose non-empty presence is a nested error
    status:          "$.status"                         the HTTP status (default: STATUS_KEYS, the text)

A payload that is not JSON, or whose named row paths are all absent, is ``unparsed``: no rows are
invented from a shape the spec does not describe. A payload with an explicit not-found is never
unparsed (its rows are simply absent).
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

from ..base import EnvelopeBase, ParsedResult
from ..registry import register
from . import http_status

__all__ = ["JsonPathEnvelope", "decode_with"]

_MESSAGE_KEYS = ("message", "error", "detail", "msg")


def _paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _text(value: Any) -> str:
    import json

    return value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)


def _test(obj: Any, expr: str) -> bool:
    from ...gateway.fields import jp_test

    try:
        return jp_test(obj, expr)
    except Exception:  # noqa: BLE001 - a bad predicate never matches
        return False


def _rows_at(obj: Any, path: str) -> list[Any] | None:
    """The row list at ``path``: a list value, or the items a ``[*]`` path selects; None when absent."""
    from ...gateway.fields import jp_get

    try:
        values = jp_get(obj, path)
    except Exception:  # noqa: BLE001 - an unreadable path is an absent one
        return None
    if "[*]" in path:
        return list(values) if values else ([] if _parent_present(obj, path) else None)
    if len(values) == 1 and isinstance(values[0], list):
        return list(values[0])
    return None


def _parent_present(obj: Any, path: str) -> bool:
    from ...gateway.fields import jp_get

    head = path.split("[*]", 1)[0]
    try:
        return any(isinstance(v, list) for v in jp_get(obj, head))
    except Exception:  # noqa: BLE001
        return False


def decode_with(raw_text: str | None, structured: Any, spec: Mapping[str, Any]) -> ParsedResult:
    """Decode one result with the ``jsonpath`` options in ``spec`` (see the module docstring)."""
    from ...gateway.fields import jp_get, parse_payload

    spec = dict(spec or {})
    try:
        obj, is_json = parse_payload(raw_text, structured)
    except Exception:  # noqa: BLE001 - an unreadable payload is unparsed, never an error here
        obj, is_json = raw_text, False
    if not is_json:
        text = None if obj is None else str(obj)
        return ParsedResult(rows=None, unparsed=True, message=(text or "")[:500] or None,
                            http_status=http_status(None, text),
                            found=False if text and any(_test({"error": text, "message": text}, p)
                                                        for p in _paths(spec.get("not_found_when"))) else None)

    # found / not found
    found: bool | None = None
    if any(_test(obj, p) for p in _paths(spec.get("not_found_when"))):
        found = False
    elif spec.get("found"):
        found = _test(obj, str(spec["found"]))

    # nested errors: a path whose value is present and not empty
    errors: list[str] = []
    for p in _paths(spec.get("errors")):
        try:
            values = jp_get(obj, p)
        except Exception:  # noqa: BLE001
            values = []
        for v in values:
            if v in (None, "", [], {}):
                continue
            if isinstance(v, list):
                errors.extend(_text(x) for x in v if x not in (None, "", [], {}))
            else:
                errors.append(_text(v))

    # message and status
    message = None
    for p in _paths(spec.get("message")) or [f"$.{k}" for k in _MESSAGE_KEYS]:
        try:
            values = [v for v in jp_get(obj, p) if v not in (None, "")]
        except Exception:  # noqa: BLE001
            values = []
        if values:
            message = _text(values[0])[:500]
            break
    status = None
    for p in _paths(spec.get("status")):
        try:
            got = jp_get(obj, p)
        except Exception:  # noqa: BLE001
            got = []
        status = next((_as_int(v) for v in got if _as_int(v) is not None), None)
        if status is not None:
            break
    if status is None:
        status = http_status(obj, message)

    # rows and total
    rows: list[Any] | None = None
    unparsed = False
    row_paths = _paths(spec.get("rows"))
    if row_paths:
        found_lists = [r for r in (_rows_at(obj, p) for p in row_paths) if r is not None]
        if found_lists:
            rows = [row for lst in found_lists for row in lst]
        elif found is not False and not errors and status is None:
            unparsed = True                            # the named row lists are absent: an unknown shape
    elif isinstance(obj, list):
        rows = list(obj)
    total = None
    if spec.get("total"):
        try:
            total = next((_as_int(v) for v in jp_get(obj, str(spec["total"])) if _as_int(v) is not None), None)
        except Exception:  # noqa: BLE001
            total = None
    return ParsedResult(rows=rows, total=total, found=found, message=message, unparsed=unparsed,
                        errors=tuple(errors), http_status=status)


@register
class JsonPathEnvelope(EnvelopeBase):
    name: ClassVar[str] = "jsonpath"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"rows", "totals", "nested_errors", "http_status"})

    def decode(self, raw_text: str | None, structured: Any, spec: Mapping[str, Any]) -> ParsedResult:
        return decode_with(raw_text, structured, spec)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.envelope import EnvelopeCase, EnvelopeCases

        spec = {"rows": "$.results", "total": "$.count", "not_found_when": ["$.error =~ '(?i)not found'"],
                "errors": ["$.errors"]}
        cases: Sequence[EnvelopeCase] = (
            EnvelopeCase("rows_and_total", '{"count": 3, "results": [{"id": "a"}, {"id": "b"}]}', None, spec,
                         rows=2, total=3, found=None),
            EnvelopeCase("structured", None, {"count": 0, "results": []}, spec, rows=0, total=0),
            EnvelopeCase("not_found", '{"error": "Study NCT00000000 not found"}', None, spec, found=False,
                         rows=None, message="Study NCT00000000 not found"),
            EnvelopeCase("nested_error", '{"count": 1, "results": [{"id": "a"}], "errors": ["upstream timeout"]}',
                         None, spec, rows=1, total=1, errors=("upstream timeout",), requires=("nested_errors",)),
            EnvelopeCase("http_503", '{"error": "Failed to retrieve data: status code: 503"}', None, spec,
                         http_status=503, requires=("http_status",)),
            EnvelopeCase("http_404_text", "Client error '404 Not Found' for url 'https://example.org/x'", None,
                         spec, http_status=404, unparsed=True, requires=("http_status",)),
            EnvelopeCase("item_paths", '{"data": {"items": [{"k": 1}, {"k": 2}]}}', None,
                         {"rows": "$.data.items[*]"}, rows=2),
            EnvelopeCase("two_lists", '{"up": [{"g": "A"}], "down": [{"g": "B"}, {"g": "C"}]}', None,
                         {"rows": ["$.up", "$.down"]}, rows=3),
            EnvelopeCase("top_level_list", '[{"id": 1}, {"id": 2}]', None, {}, rows=2),
        )
        return EnvelopeCases(recorded=tuple(cases), spec=spec)
