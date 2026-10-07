"""Envelope and structural classification of upstream results (§11.4 step 1, §11.9). No pyarrow.

The bridge returns a :class:`~vbt.datalayer.api.RawResult` whose ``envelope`` says what it
would have done without a gateway; :func:`classify` decides what the result **means**, in
this order (rev 2 fixes the precedence):

1. memory signatures in the error text (``MemoryError``, ``Unable to allocate``,
   ``ArrowMemoryError``, ``bad_alloc``) -> ``oom``;
2. explicit not-found: the binding's (or the generic overlay's) ``not_found_when`` predicates and
   the bridge's ``_EMPTY_LOOKUP`` messages, **before** legacy envelopes become ``source_error``
   (cBioPortal's "Failed to retrieve ... status code: 404" is ``not_found``). For a reviewed
   binding the meaning depends on where the identifier lives:

   * the bound table is the id_type's universe table and existence is ``upstream`` -> ``not_found``;
   * resolution proved the entity exists and the bound table is the universe table ->
     ``contradiction`` (repair or ``tool_defect``);
   * the bound table is another table -> ``empty`` when the witness counted 0 rows,
     ``contradiction`` when it counted more, ``empty_unverified`` when it could not count;
   * a generic (unbound) tool -> ``not_found``;

3. ``nested_errors`` present -> ``source_error``, unless ``total.partial_when`` matches (``partial``);
4. remaining ``legacy_error`` / ``is_error`` -> ``source_error``;
5. otherwise ``ok``; for generic tools a structural empty (``count == 0``, ``num_results == 0``,
   an empty top-level list, or the generic ``empty_when``) is ``empty_unverified``: a success that
   cannot be cited, either as support or as absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

from ..errors import ErrorKind, GatewayError, not_found_payload
from ...tools.mcp_bridge import _EMPTY_LOOKUP
from ..memory.crash import classify_error_text
from .fields import jp_test, parse_payload

__all__ = ["Outcome", "Classification", "classify", "structural_empty", "EMPTY_LOOKUP"]

Outcome = Literal["ok", "not_found", "empty", "empty_unverified", "contradiction", "partial", "source_error", "oom"]

#: The upstream tools' explicit empty-lookup messages: the bridge owns the pattern, this module its meaning.
EMPTY_LOOKUP = _EMPTY_LOOKUP

_COUNT_KEYS = ("count", "num_results", "total", "total_count", "n_results", "numResults", "totalCount")


@dataclass
class Classification:
    outcome: Outcome
    reason: str = ""
    obj: Any = None                                    # the parsed payload (or text)
    is_json: bool = False
    error: GatewayError | None = None                  # for not_found / source_error / oom
    explicit_not_found: bool = False
    matched: list[str] = field(default_factory=list)   # predicates that matched


def _texts(raw: Any) -> list[str]:
    out = [t for t in (getattr(raw, "error_text", None), getattr(raw, "text", None)) if t]
    return [str(t) for t in out]


def _error_message(obj: Any) -> str | None:
    if isinstance(obj, Mapping):
        err = obj.get("error") or obj.get("errors") or obj.get("message")
        if isinstance(err, list):
            return "; ".join(str(e) for e in err)
        return str(err) if err else None
    return None


def structural_empty(obj: Any, empty_when: Iterable[str] = ()) -> bool:
    """True for a structurally empty success: an empty list, a ``count``-like field equal to 0
    (with every list field empty), or a generic ``empty_when`` predicate."""
    if isinstance(obj, list):
        return not obj
    if not isinstance(obj, Mapping):
        return False
    if any(jp_test(obj, e) for e in empty_when):
        return True
    counts = [obj[k] for k in _COUNT_KEYS if k in obj and isinstance(obj[k], (int, float))
              and not isinstance(obj[k], bool)]
    lists = [v for v in obj.values() if isinstance(v, list)]
    if counts and all(c == 0 for c in counts) and all(not v for v in lists):
        return True
    # no counts: every list empty and no non-empty nested record
    records = [v for v in obj.values() if isinstance(v, Mapping) and v]
    return not counts and bool(lists) and all(not v for v in lists) and not records


def classify(raw: Any, contract: Any, plan: Any = None, *,
             universe_tables: Mapping[str, Iterable[str]] | None = None,
             witness_total: int | None = None, tool: str | None = None) -> Classification:
    """Classify ``raw`` (see the module docstring).

    ``universe_tables`` maps each identifier argument to the ``source.table`` names of its
    identity universe; ``witness_total`` is the witness count under the bound predicate
    (None: not counted)."""
    name = tool or (f"mcp__{contract.server}__{contract.tool}" if contract is not None else None)
    envelope = getattr(raw, "envelope", "ok")
    obj, is_json = parse_payload(getattr(raw, "text", None), getattr(raw, "structured", None))

    # 1. memory signatures in any error text
    if envelope in ("is_error", "legacy_error"):
        for t in _texts(raw):
            if classify_error_text(t):
                return Classification("oom", "memory error inside the server", obj, is_json,
                                      GatewayError(ErrorKind.oom, f"the server ran out of memory: {t[:300]}",
                                                   tool=name, subkind="memory_error"))

    # 2. explicit not-found, before legacy envelopes
    binding = getattr(contract, "binding", None)
    preds: list[str] = []
    if binding is not None:
        preds.extend(binding.result.not_found_when)
    gspec = getattr(contract, "generic_spec", None)
    if gspec is not None:
        preds.extend(gspec.not_found_when)
    probe = obj if is_json else {"error": obj, "message": obj}
    if envelope == "is_error" and not is_json:
        probe = {"error": getattr(raw, "error_text", None) or obj, "message": obj}
    matched = [p for p in preds if _safe_test(probe, p)]
    message = getattr(raw, "error_text", None) or _error_message(obj) or (obj if isinstance(obj, str) else None)
    lookup_miss = envelope == "empty_lookup" or bool(message and EMPTY_LOOKUP.fullmatch(str(message).strip()))
    if matched or lookup_miss:
        reason = f"explicit not-found ({matched[0] if matched else str(message or 'empty lookup')[:200]})"
        return _not_found(contract, plan, obj, is_json, reason, matched, universe_tables, witness_total, name)

    # 3. nested errors
    if binding is not None and binding.result.nested_errors and is_json:
        hits = [p for p in binding.result.nested_errors if _safe_test(obj, p)]
        if hits:
            total = binding.result.total
            partial_when = list(getattr(total, "partial_when", []) or []) if total is not None else []
            if any(_safe_test(obj, p) for p in partial_when):
                return Classification("partial", f"nested error with a partial result ({hits[0]})", obj, is_json,
                                      matched=hits)
            return Classification("source_error", f"nested error at {hits[0]}", obj, is_json,
                                  GatewayError(ErrorKind.source_error, f"the source reported a nested error ({hits[0]})",
                                               tool=name, payload={"path": hits[0]}), matched=hits)

    # 4. remaining error envelopes
    if envelope in ("legacy_error", "is_error"):
        text = getattr(raw, "error_text", None) or getattr(raw, "text", None) or "the tool reported an error"
        return Classification("source_error", str(text)[:300], obj, is_json,
                              GatewayError(ErrorKind.source_error, f"the source reported an error: {str(text)[:500]}",
                                           tool=name))

    # 5. success; generic structural empties are uncitable
    if contract is None or getattr(contract, "generic", False) or binding is None:
        empty_when = list(gspec.empty_when) if gspec is not None else []
        if is_json and structural_empty(obj, empty_when):
            return Classification("empty_unverified", "structural empty from an unbound tool", obj, is_json)
    return Classification("ok", "", obj, is_json)


def _safe_test(obj: Any, expr: str) -> bool:
    try:
        return jp_test(obj, expr)
    except (ValueError, TypeError, Exception):  # noqa: BLE001 - a bad overlay predicate never matches
        return False


def _not_found(contract: Any, plan: Any, obj: Any, is_json: bool, reason: str, matched: list[str],
               universe_tables: Mapping[str, Iterable[str]] | None, witness_total: int | None,
               name: str | None) -> Classification:
    binding = getattr(contract, "binding", None)
    generic = contract is None or getattr(contract, "generic", False) or binding is None
    args = dict(getattr(plan, "args_raw", {}) or {})
    if generic:
        return Classification("not_found", reason, obj, is_json,
                              _nf_error(name, args, None, reason), explicit_not_found=True, matched=matched)
    bound = getattr(plan, "bound_table", None) or contract.bound_table
    existence = dict(getattr(plan, "existence", {}) or {})
    id_args = [a for a in contract.identifier_args if args.get(a) is not None]
    universe_tables = universe_tables or {}
    on_universe = [a for a in id_args if bound and bound in set(universe_tables.get(a, ()))]
    # existence: upstream decides on the bound universe, or wherever no local universe table exists
    # (a remote universe listed only by an upstream tool, cBioPortal studies)
    for a in [a for a in id_args if a in on_universe or not universe_tables.get(a)]:
        if contract.args[a].existence == "upstream" and existence.get(a) != "exists":
            return Classification("not_found", reason, obj, is_json, _nf_error(name, args, a, reason, bound),
                                  explicit_not_found=True, matched=matched)
    proven = [a for a in on_universe if existence.get(a) == "exists"]
    if proven:
        return Classification("contradiction", f"{reason}, but {proven[0]} was resolved in {bound}", obj, is_json,
                              explicit_not_found=True, matched=matched)
    if id_args and not on_universe:
        if witness_total is None:
            return Classification("empty_unverified", f"{reason}; the witness could not count {bound}", obj, is_json,
                                  explicit_not_found=True, matched=matched)
        if witness_total > 0:
            return Classification("contradiction", f"{reason}, but the witness counted {witness_total} rows",
                                  obj, is_json, explicit_not_found=True, matched=matched)
        return Classification("empty", reason, obj, is_json, explicit_not_found=True, matched=matched)
    arg = on_universe[0] if on_universe else (id_args[0] if id_args else None)
    return Classification("not_found", reason, obj, is_json, _nf_error(name, args, arg, reason, bound),
                          explicit_not_found=True, matched=matched)


def _nf_error(tool: str | None, args: Mapping[str, Any], arg: str | None, reason: str,
              table: str | None = None) -> GatewayError:
    value = args.get(arg) if arg else None
    payload = not_found_payload(arg or "", value, None, [], [f"upstream: {reason}"], [], None, table)
    if arg is None:
        payload.pop("argument", None)
        payload.pop("value", None)
    return GatewayError(ErrorKind.not_found, f"the source reports no such record ({reason})", tool=tool,
                        argument=arg, value=value, payload=payload)
