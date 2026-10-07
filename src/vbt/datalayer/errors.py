"""Typed gateway errors returned to agents (§12.1). No pyarrow.

A :class:`GatewayError` is a :class:`~vbt.tools.base.ToolFailure`, so the runtime shows
``Error: <json>`` with ``is_error=True`` exactly as for any failed tool. ``str(err)`` is the
JSON envelope, which extends the bridge's ``{"status": "tool_error", ...}`` envelope with
``contract: vbt.data/1``, the error ``kind`` and a fixed per-kind payload.

Payload builders (``*_payload``) produce exactly the fields of the §12.1 payload table, so
the correctness tests can assert them and agents always see the same shape.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ..tools.base import ToolFailure
from .rowkey import render_float

__all__ = [
    "CONTRACT", "ErrorKind", "MODEL_SIDE_KINDS", "DATA_SIDE_KINDS", "INSTRUCTIONS", "DEFAULT_RETRYABLE",
    "GatewayError", "error_envelope", "nearest", "edit_distance", "json_value",
    "not_found_payload", "ambiguous_payload", "invalid_argument_payload", "unsupported_combination_payload",
    "incomplete_key_payload", "unsupported_filter_payload", "insufficient_resolution_payload",
    "not_ready_payload", "too_large_payload", "tool_defect_payload",
]

CONTRACT = "vbt.data/1"


class ErrorKind(str, Enum):
    not_found = "not_found"
    ambiguous = "ambiguous"
    invalid_argument = "invalid_argument"
    unsupported_combination = "unsupported_combination"
    incomplete_key = "incomplete_key"
    unsupported_filter = "unsupported_filter"
    insufficient_resolution = "insufficient_resolution"
    not_ready = "not_ready"
    too_large = "too_large"
    quarantined = "quarantined"
    tool_defect = "tool_defect"
    oom = "oom"
    server_crashed = "server_crashed"
    source_error = "source_error"
    service_unavailable = "service_unavailable"

    def __str__(self) -> str:  # "not_found", never "ErrorKind.not_found"
        return self.value


#: Lookup misses and argument problems are the model's input errors, not data outages
#: (``failures._data_failure`` exempts them; they stay ``is_error`` and uncitable).
MODEL_SIDE_KINDS: frozenset[ErrorKind] = frozenset({
    ErrorKind.not_found, ErrorKind.ambiguous, ErrorKind.invalid_argument, ErrorKind.unsupported_combination,
    ErrorKind.incomplete_key, ErrorKind.unsupported_filter, ErrorKind.insufficient_resolution,
})
DATA_SIDE_KINDS: frozenset[ErrorKind] = frozenset(set(ErrorKind) - MODEL_SIDE_KINDS)

#: The "retryable" column of the §12.1 table, as tokens.
DEFAULT_RETRYABLE: dict[ErrorKind, str] = {
    ErrorKind.not_found: "after_fixing_input",
    ErrorKind.ambiguous: "with_candidate",
    ErrorKind.invalid_argument: "yes",
    ErrorKind.unsupported_combination: "as_separate_calls",
    ErrorKind.incomplete_key: "with_dimension",
    ErrorKind.unsupported_filter: "after_changing_filter",
    ErrorKind.insufficient_resolution: "after_fixing_list",
    ErrorKind.not_ready: "no",
    ErrorKind.too_large: "with_narrower_arguments",
    ErrorKind.quarantined: "no",
    ErrorKind.tool_defect: "no",
    ErrorKind.oom: "no",
    ErrorKind.server_crashed: "once",
    ErrorKind.source_error: "maybe",
    ErrorKind.service_unavailable: "later",
}

_NOT_EVIDENCE = "This is not evidence about biology and must not be cited."

INSTRUCTIONS: dict[ErrorKind, str] = {
    ErrorKind.not_found: "No record with this identifier exists in this source. " + _NOT_EVIDENCE,
    ErrorKind.ambiguous: ("The value matches several records. Call again with one of the candidate IDs. "
                          + _NOT_EVIDENCE),
    ErrorKind.invalid_argument: ("An argument is not valid for this tool (see valid_values, near and looks_like). "
                                 "Fix it and call again. " + _NOT_EVIDENCE),
    ErrorKind.unsupported_combination: ("These arguments cannot be combined in one call. Make separate calls. "
                                        + _NOT_EVIDENCE),
    ErrorKind.incomplete_key: ("Results differ by a dimension the arguments do not fix. Call again with one of the "
                               "listed values (retry_with). " + _NOT_EVIDENCE),
    ErrorKind.unsupported_filter: ("This filter cannot be applied faithfully to this data. Change or drop it. "
                                   + _NOT_EVIDENCE),
    ErrorKind.insufficient_resolution: ("Too few elements of the list resolved to identifiers of this source. Fix "
                                        "the listed elements and call again. " + _NOT_EVIDENCE),
    ErrorKind.not_ready: ("The data this tool reads failed its readiness check. This call did not produce evidence; "
                          "report the outage and do not treat it as a negative result."),
    ErrorKind.too_large: ("The call cannot be answered within the memory or verification limits. Narrow the query "
                          "(fewer rows, a more specific filter) or use the named alternative. "
                          "This call did not produce evidence."),
    ErrorKind.quarantined: ("This tool is not available because it gives wrong answers on this data. Use the named "
                            "alternative. This call did not produce evidence."),
    ErrorKind.tool_defect: ("The tool's answer contradicted an independent read of the data and was withheld. "
                            "This call did not produce evidence; do not cite it."),
    ErrorKind.oom: ("The server ran out of memory. Do not retry the same call; narrow it. "
                    "This call did not produce evidence."),
    ErrorKind.server_crashed: ("The server crashed. This call did not produce evidence. Report the failure and its "
                               "effect on the analysis."),
    ErrorKind.source_error: ("The data source reported an error. This call did not produce evidence. Report the "
                             "failure and its effect on the analysis."),
    ErrorKind.service_unavailable: ("The data-layer service is down, so this tool cannot be guarded. This call did "
                                    "not produce evidence; try again later."),
}

_ENVELOPE_FIXED = ("status", "contract", "kind", "subkind", "tool", "argument", "value", "message", "citable",
                   "retryable", "instruction")


def json_value(value: Any, storage_type: str | None = None) -> Any:
    """A JSON-typed value: floats rendered in their storage type (``0.05`` for float32 0.05),
    NaN as None, tuples and sets as lists."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return value
        return float(render_float(value, storage_type or "double"))
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_value(v, storage_type) for v in value]
    return str(value)


def _sort_key(value: Any) -> tuple[int, Any]:
    if value is None:
        return (2, "")
    if isinstance(value, bool):
        return (0, float(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value))


class GatewayError(ToolFailure):
    """A typed data-layer error (§12.1). ``str()`` is the JSON envelope the agent sees."""

    def __init__(self, kind: ErrorKind | str, message: str, *, tool: str | None = None, argument: str | None = None,
                 value: Any = None, payload: Mapping[str, Any] | None = None, retryable: str | None = None,
                 subkind: str | None = None, instruction: str | None = None) -> None:
        self.kind = ErrorKind(kind)
        self.message = str(message)
        self.tool = tool
        self.argument = argument
        self.value = value
        self.payload: dict[str, Any] = dict(payload or {})
        if subkind is None and "subkind" in self.payload:
            subkind = self.payload.pop("subkind")
        # argument/value live at the envelope's top level; a payload builder's copies fill them in.
        if self.argument is None and self.payload.get("argument") is not None:
            self.argument = self.payload["argument"]
        if self.value is None and self.payload.get("value") is not None:
            self.value = self.payload["value"]
        self.subkind = subkind
        self.retryable = retryable if retryable is not None else DEFAULT_RETRYABLE[self.kind]
        self.instruction = instruction or INSTRUCTIONS[self.kind]
        super().__init__(self.message)

    @property
    def model_side(self) -> bool:
        """True for the model's input errors (not a data-source failure)."""
        return self.kind in MODEL_SIDE_KINDS

    def envelope(self) -> dict[str, Any]:
        env: dict[str, Any] = {"status": "tool_error", "contract": CONTRACT, "kind": self.kind.value}
        if self.subkind:
            env["subkind"] = self.subkind
        env["tool"] = self.tool
        if self.argument is not None:
            env["argument"] = self.argument
        if self.value is not None:
            env["value"] = json_value(self.value)
        env["message"] = self.message
        for k, v in self.payload.items():
            if k in _ENVELOPE_FIXED:
                continue  # the fixed fields above cannot be overridden by a payload
            env[k] = v
        env["citable"] = False
        env["retryable"] = self.retryable
        env["instruction"] = self.instruction
        return env

    def __str__(self) -> str:
        return json.dumps(self.envelope(), default=str, ensure_ascii=False)

    @classmethod
    def from_envelope(cls, env: Mapping[str, Any]) -> "GatewayError":
        """The error an :meth:`envelope` describes (an envelope the data child sent back)."""
        payload = {k: v for k, v in env.items() if k not in _ENVELOPE_FIXED}
        return cls(str(env.get("kind")), str(env.get("message") or ""), tool=env.get("tool"),
                   argument=env.get("argument"), value=env.get("value"), payload=payload,
                   retryable=env.get("retryable"), subkind=env.get("subkind"), instruction=env.get("instruction"))

    def with_tool(self, tool: str) -> "GatewayError":
        """Set the tool name when the raising code did not know it; returns self."""
        if not self.tool:
            self.tool = tool
        return self


def error_envelope(kind: ErrorKind | str, message: str, **kwargs: Any) -> dict[str, Any]:
    """The envelope dict of a :class:`GatewayError` without raising it."""
    return GatewayError(kind, message, **kwargs).envelope()


# ---------------------------------------------------------------------------
# Near matches for invalid values
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (small strings; vocabularies and identifiers)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nearest(value: Any, vocabulary: Iterable[Any], n: int = 10) -> list[Any]:
    """The ``n`` vocabulary values nearest to ``value`` by casefold edit distance, then
    casefold prefix/substring relation, then the value itself (deterministic)."""
    target = str(value).casefold()

    def score(v: Any) -> tuple[int, int, str]:
        s = str(v).casefold()
        related = 0 if (target and (s.startswith(target) or target in s or s in target)) else 1
        return (edit_distance(target, s), related, str(v))

    return sorted(vocabulary, key=score)[:n]


# ---------------------------------------------------------------------------
# Per-kind payloads (§12.1 payload table)
# ---------------------------------------------------------------------------

def not_found_payload(argument: str, value: Any, id_type: str | None, accepts: Sequence[str], tried: Sequence[str],
                      suggestions: Sequence[Mapping[str, Any]], source: str | None, table: str | None, *,
                      subkind: str | None = None, replacement: Any = None,
                      profiled_values: Mapping[str, Any] | Sequence[Any] | None = None) -> dict[str, Any]:
    """``argument, value, id_type, accepts, tried[], suggestions[{id, label, why}], source, table,
    subkind?, replacement?, profiled_values?``. Subkinds: ``obsolete`` (replacement named),
    ``combination_not_profiled`` (profiled values listed), ``not_measured_in`` (fragments listed)."""
    out: dict[str, Any] = {
        "argument": argument, "value": json_value(value), "id_type": id_type, "accepts": list(accepts),
        "tried": list(tried),
        "suggestions": [{"id": s.get("id"), "label": s.get("label"), "why": s.get("why")} for s in suggestions],
        "source": source, "table": table,
    }
    if subkind is not None:
        out["subkind"] = subkind
    if replacement is not None:
        out["replacement"] = json_value(replacement)
    if profiled_values is not None:
        out["profiled_values"] = json_value(profiled_values)
    return out


def ambiguous_payload(argument: str, value: Any, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``argument, value, candidates[{id, label, via, <disambiguate_with columns>}]``."""
    cands = []
    for c in candidates:
        entry = {"id": c.get("id"), "label": c.get("label"), "via": c.get("via")}
        entry.update({str(k): json_value(v) for k, v in c.items() if k not in entry})
        cands.append(entry)
    return {"argument": argument, "value": json_value(value), "candidates": cands}


def invalid_argument_payload(argument: str, value: Any, valid_values: Sequence[Any] | None, *,
                             vocabulary_ref: str | None = None, near: Sequence[Any] = (),
                             looks_like: Sequence[str] = (), items: Sequence[Mapping[str, Any]] | None = None,
                             enum_max: int = 64) -> dict[str, Any]:
    """``argument, value, valid_values, vocabulary_ref?, near[], looks_like[], items?[{index, value, reason}]``.

    ``valid_values`` is the full vocabulary when it has at most ``enum_max`` values, else the
    10 values nearest to ``value`` by casefold and edit distance (and ``vocabulary_ref``
    names where the full list lives)."""
    vocab = list(valid_values or [])
    shown = vocab if len(vocab) <= enum_max else nearest(value, vocab, 10)
    out: dict[str, Any] = {"argument": argument, "value": json_value(value), "valid_values": json_value(shown)}
    if vocabulary_ref is not None:
        out["vocabulary_ref"] = vocabulary_ref
    out["near"] = json_value(list(near))
    out["looks_like"] = list(looks_like)
    if items is not None:
        out["items"] = [{"index": i.get("index"), "value": json_value(i.get("value")), "reason": i.get("reason")}
                        for i in items]
    return out


def unsupported_combination_payload(arguments: Sequence[str], reason: str, *,
                                    group_argument: str | None = None,
                                    alternative: str | None = None) -> dict[str, Any]:
    """``arguments, reason, group_argument?`` (the argument that would fix a comparable group)
    ``, alternative?`` (a tool that answers the call)."""
    out: dict[str, Any] = {"arguments": list(arguments), "reason": reason}
    if group_argument is not None:
        out["group_argument"] = group_argument
    if alternative is not None:
        out["alternative"] = alternative
    return out


def incomplete_key_payload(dimension: str, argument: str | None, values: Iterable[Any], *, unit: str | None = None,
                           subkind: str | None = None, storage_type: str | None = None,
                           retry_with: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """``dimension, argument, values`` (sorted, JSON-typed, rendered in the storage type) ``, unit?,
    subkind?, retry_with``. ``retry_with`` defaults to one ``{argument: value}`` per value."""
    rendered = sorted((json_value(v, storage_type) for v in values), key=_sort_key)
    out: dict[str, Any] = {"dimension": dimension, "argument": argument, "values": rendered}
    if unit is not None:
        out["unit"] = unit
    if subkind is not None:
        out["subkind"] = subkind
    if retry_with is None:
        retry_with = [{argument or dimension: v} for v in rendered]
    out["retry_with"] = [dict(r) for r in retry_with]
    return out


def unsupported_filter_payload(argument: str, column: str | None, reason: str, *,
                               confirmed_range: Sequence[Any] | None = None) -> dict[str, Any]:
    """``argument, column, reason`` (``scale``, ``unconfirmed_encoding``, ``unbound_argument``) ``, confirmed_range?``."""
    out: dict[str, Any] = {"argument": argument, "column": column, "reason": reason}
    if confirmed_range is not None:
        out["confirmed_range"] = json_value(list(confirmed_range))
    return out


def insufficient_resolution_payload(argument: str, requested: int, resolved: int, unresolved: Sequence[Any],
                                    ambiguous: Sequence[Any], outside_universe: Sequence[Any],
                                    min_resolved_fraction: float) -> dict[str, Any]:
    """``argument, requested, resolved, unresolved[], ambiguous[], outside_universe[], min_resolved_fraction``."""
    return {"argument": argument, "requested": int(requested), "resolved": int(resolved),
            "unresolved": json_value(list(unresolved)), "ambiguous": json_value(list(ambiguous)),
            "outside_universe": json_value(list(outside_universe)),
            "min_resolved_fraction": float(min_resolved_fraction)}


def not_ready_payload(tables: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``tables[{name, column?, partition?, check, detail, hint}]``."""
    out = []
    for t in tables:
        entry: dict[str, Any] = {"name": t.get("name")}
        for opt in ("column", "partition"):
            if t.get(opt) is not None:
                entry[opt] = t.get(opt)
        entry.update({"check": t.get("check"), "detail": t.get("detail"), "hint": t.get("hint")})
        out.append(entry)
    return {"tables": out}


def too_large_payload(reason: str, *, need_mb: float | None = None, limit_mb: float | None = None,
                      alternative: str | None = None, hint: str | None = None,
                      learned: bool = False) -> dict[str, Any]:
    """``reason, need_mb?, limit_mb?, alternative?, hint?, learned?`` (subkind passed to the error)."""
    out: dict[str, Any] = {"reason": reason}
    for k, v in (("need_mb", need_mb), ("limit_mb", limit_mb), ("alternative", alternative), ("hint", hint)):
        if v is not None:
            out[k] = v
    if learned:
        out["learned"] = True
    return out


def tool_defect_payload(check: str, witness: Mapping[str, Any], returned: Any,
                        defect_ids: Sequence[str] = ()) -> dict[str, Any]:
    """``check (W1..W6), witness{total, topk?}, returned, defect_ids[]``."""
    w: dict[str, Any] = {"total": witness.get("total")}
    if witness.get("topk") is not None:
        w["topk"] = json_value(witness.get("topk"))
    return {"check": check, "witness": w, "returned": json_value(returned), "defect_ids": list(defect_ids)}
