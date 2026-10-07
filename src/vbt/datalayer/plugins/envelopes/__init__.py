"""Builtin envelope plugins (phase 4, F21: the fifth plugin kind, §9.5). No pyarrow.

An envelope plugin decodes one tool result into :class:`~vbt.datalayer.plugins.base.ParsedResult`
(rows, total, found, message, nested errors, HTTP status, or ``unparsed``). Overlays name one with
``result.codec`` (default :data:`DEFAULT_ENVELOPE`, the JSONPath decoder every overlay already speaks)
and pass its options as ``result.codec_options``; a third-party result shape that JSONPaths cannot
describe (XML, a text table, a paged envelope) gets its own plugin instead of core code.

Shared helpers: :func:`http_status` reads an HTTP status from a payload or an error text
(``status code: 404``, ``HTTP 503``, ``503 Server Error``), :func:`default_envelope` returns the
builtin decoder without a registry, :func:`codec_of` the codec an overlay result spec names.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

__all__ = ["DEFAULT_ENVELOPE", "STATUS_KEYS", "http_status", "default_envelope", "codec_of"]

#: The envelope used when an overlay names none.
DEFAULT_ENVELOPE = "jsonpath"
#: Payload keys that may carry an HTTP status.
STATUS_KEYS = ("status", "status_code", "statusCode", "http_status", "code")

_STATUS_RE = re.compile(
    r"(?i)(?:status(?:[ _-]?code)?\s*[:=]?\s*|\bHTTP(?:/\d(?:\.\d)?)?\s+|\berror\s+)([1-5]\d\d)\b"
    r"|\b([1-5]\d\d)\s+(?:Client\s+Error|Server\s+Error|Not\s+Found|Bad\s+Request|Unauthorized|Forbidden|"
    r"Too\s+Many\s+Requests|Internal\s+Server\s+Error|Bad\s+Gateway|Service\s+Unavailable|Gateway\s+Time-?out)\b")


def _status_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 100 <= value <= 599:
        return value
    if isinstance(value, str) and value.strip().isdigit() and 100 <= int(value.strip()) <= 599:
        return int(value.strip())
    return None


def http_status(obj: Any = None, text: str | None = None) -> int | None:
    """The HTTP status a payload (``status``/``status_code``/... at the top level) or an error text
    reports, or None."""
    if isinstance(obj, Mapping):
        for key in STATUS_KEYS:
            n = _status_number(obj.get(key))
            if n is not None:
                return n
        for key in ("error", "message", "detail"):
            v = obj.get(key)
            if isinstance(v, str):
                n = http_status(None, v)
                if n is not None:
                    return n
    if text:
        m = _STATUS_RE.search(str(text))
        if m:
            return int(m.group(1) or m.group(2))
    return None


_DEFAULT: list[Any] = []


def default_envelope() -> Any:
    """The builtin :data:`DEFAULT_ENVELOPE` plugin instance (no registry needed)."""
    if not _DEFAULT:
        from .jsonpath import JsonPathEnvelope

        _DEFAULT.append(JsonPathEnvelope())
    return _DEFAULT[0]


def codec_of(result_spec: Any) -> tuple[str, dict[str, Any]]:
    """``(codec name, options)`` an overlay result spec names (``result.codec``/``codec_options``,
    else the default codec reading the spec's own ``rows``, ``total`` and ``not_found_when``)."""
    name = getattr(result_spec, "codec", None) or DEFAULT_ENVELOPE
    options = dict(getattr(result_spec, "codec_options", None) or {})
    if name == DEFAULT_ENVELOPE and result_spec is not None:
        rows = getattr(result_spec, "row_paths", None)
        if rows and getattr(result_spec, "kind", "rows") == "rows":
            options.setdefault("rows", list(rows))
        total = getattr(result_spec, "total", None)
        path = total if isinstance(total, str) else getattr(total, "path", None)
        if path:
            options.setdefault("total", path)
        nf = list(getattr(result_spec, "not_found_when", []) or [])
        if nf:
            options.setdefault("not_found_when", nf)
        nested = list(getattr(result_spec, "nested_errors", []) or [])
        if nested:
            options.setdefault("errors", nested)
    return name, options
