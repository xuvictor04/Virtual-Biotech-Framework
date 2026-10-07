"""The resolver (§11.5): an argument value -> one canonical key of the bound column's id_type, by a recorded
rule, or a typed outcome. Harness side; no pyarrow (indexes are stdlib sidecars, ``resolve/index.py``).

``Resolver.resolve(value, accepts, bound_id_type=...)`` tries the accepted kinds in order. For each
kind it applies that id_type's ordered rule list (``IdTypeSpec.rules`` or the defaults of
``resolve/rules.py``), filtered by ``data.resolution.allow``:

* membership rules first: ``raw_member`` (the value as given is a universe key, checked before
  normalisation), ``exact`` and ``normalized:<steps>`` (the plugin's normalisation, steps recorded);
* label rules on the index: ``label_exact``/``label_casefold`` and ``synonym:<kind>`` (casefold
  and synonyms resolve only to a unique canonical; ``related`` adds a warning; ``narrow``/``broad``
  never resolve), then ``retired`` (one replacement resolves, several are ambiguous, none is
  ``not_found`` with subkind ``obsolete``), ``xref`` (namespace maps; one xref shared by several
  terms is ambiguous), ``crosswalk`` and ``stored_form``.

A label kind (``label_of``) resolves through its parent type's index; when a key kind and its
label kind are both accepted, the key kind leaves the label rules to the label kind, so the label
kind is recorded as ``matched_id_type`` (the §4 walk-through). A union kind tries every member and
is ambiguous when members give different canonicals. Several candidates are collapsed to their
common parent when the id_type declares ``canonicalize`` (rule ``parent_family``), resolved by a
declared ``prefer`` order (disclosed), or returned as ``ambiguous`` with their ``disambiguate_with``
values. Substring and prefix matching are never used.

The result is always of the bound column's id_type (I1): the matched kind reaches it through
declared edges (``label_of``, union membership, ``maps_to``/``extends``, crosswalks), at most
``data.resolution.max_hops`` of them, each recorded in ``hops``; a one-to-many hop is ambiguous;
an accepted kind with no edge is a :class:`ResolverConfigError`.

Statuses: ``resolved``; ``resolved_unverified`` (existence ``bound``: absent from the universe, the
gateway confirms with one witness count on the bound column); ``ambiguous``; ``not_found``
(syntactically valid, absent; with suggestions and the ``tried`` list); ``rejected`` (every
accepted kind rejected the syntax and no label, synonym, retired or xref rule applied, or the
value is outside ``universe_where``; the gateway raises ``invalid_argument``); ``unknown``
(existence cannot be decided: a remote universe failed or is over budget, or an index is missing).
Existence modes ``upstream`` and ``off`` check syntax only.
"""

from __future__ import annotations

import inspect
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Collection, Iterable, Mapping, Sequence, cast

from ..errors import (
    ErrorKind,
    GatewayError,
    ambiguous_payload,
    insufficient_resolution_payload,
    invalid_argument_payload,
    not_found_payload,
)
from ..plugins.base import Candidate, Normalized, Rejected, Resolution
from ..plugins.identifiers import as_text
from ..predicate import PredicateError, evaluate, from_json
from ..record import ResolutionRecord
from ..settings import ResolutionSettings
from .index import Entry, ResolverIndex
from .rules import LABEL_HEADS, MEMBERSHIP_HEADS, SEARCH_ONLY_SYNONYMS, Rule, allowed, parse_rule, rules_for

__all__ = [
    "Resolver", "ResolutionResult", "ResolverConfigError", "EXISTENCE_MODES", "HINT_THRESHOLD", "OK_STATUSES",
    "error_for", "list_error",
]

EXISTENCE_MODES = ("universe", "bound", "upstream", "off")
#: Plugins whose ``looks_like`` scores at least this are named in a rejection's ``looks_like``.
HINT_THRESHOLD = 0.8
#: Statuses with which the call proceeds.
OK_STATUSES = frozenset({"resolved", "resolved_unverified", "unknown"})

IndexProvider = Callable[[str, str], "ResolverIndex | None"]
#: ``universe_where``: allowed canonical keys, a ``canonical -> bool`` callable, or a predicate (Predicate JSON
#: or IR) over the bound id_type's index attributes (``attr:<column>`` rows, e.g. ``ontology.isTherapeuticArea``).
WherePredicate = Callable[[str], bool] | Collection[str] | Mapping[str, Any] | Any


class ResolverConfigError(Exception):
    """The catalog cannot resolve this binding (an accepted kind has no declared edge to the bound id_type
    within ``max_hops``, or an id_type is unknown). A configuration defect, never the agent's input error."""


@dataclass(frozen=True)
class ResolutionResult(Resolution):
    """A :class:`~vbt.datalayer.plugins.base.Resolution` with what the gateway needs for errors and headers."""

    label: str | None = None                           # the canonical key's label (for _vbt.resolved)
    existence: str | None = None                       # exists | absent | unknown; None: not checked (syntax only)
    suggestions: tuple[Candidate, ...] = ()            # not_found: same-id_type keys within one edit
    looks_like: tuple[str, ...] = ()                   # rejected: kinds the value looks like
    reasons: tuple[str, ...] = ()                      # rejected: each accepted kind's reason
    subkind: str | None = None                         # not_found: obsolete; rejected: outside_universe
    replacement: tuple[str, ...] = ()                  # obsolete: replacement candidates
    valid_values: tuple[str, ...] = ()                 # outside_universe: the restricted universe (when listable)
    accepts: tuple[str, ...] = ()                      # qualified accepted kinds
    source: str | None = None                          # "<source>@<release>" of the canonical id_type
    table: str | None = None                           # its universe table
    index_fingerprint: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES

    def summary(self) -> str:
        """``PCSK9 -> ENSG00000169174 (label_exact:approvedSymbol)`` for ``_vbt.resolved``."""
        if self.canonical is None:
            return f"{self.raw} ({self.status})"
        if self.rule in (None, "exact", "raw_member") and str(self.raw) == self.canonical:
            return str(self.canonical)
        return f"{self.raw} -> {self.canonical} ({self.rule})"

    def record(self, arg: str) -> ResolutionRecord:
        """The provenance record of this resolution (``request.resolutions[]``)."""
        return ResolutionRecord(arg=arg, raw=self.raw, canonical=self.canonical, matched_id_type=self.matched_id_type,
                                canonical_id_type=self.id_type, rule=self.rule,
                                family=list(self.family) if len(self.family) > 1 else None, hops=list(self.hops),
                                index_fingerprint=self.index_fingerprint, existence=self.existence)


@dataclass
class _Hit:
    """One accepted kind's outcome before mapping to the bound id_type."""

    status: str                                        # resolved | ambiguous | obsolete | outside | absent | rejected | unknown
    kind: str                                          # the id_type whose index (or plugin) produced it
    matched: str                                       # the accepted kind
    canonical: str | None = None
    rule: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    family: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)
    reason: str | None = None
    looks_like: tuple[str, ...] = ()
    replacement: tuple[str, ...] = ()
    existence: str | None = None
    hops: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Edge:
    kind: str                                          # label_of | label_rev | union_member | identity |
    #                                                    crosswalk_fwd | crosswalk_rev | remote_map
    name: str | None = None                            # crosswalk name
    owner: str | None = None                           # id_type whose index holds the crosswalk rows


class _NeedRemote(Exception):
    def __init__(self, key: tuple[str, str, tuple[str, ...]]) -> None:
        super().__init__(key)
        self.key = key


def _where_ok(where: "_Where | None", canonical: str) -> bool:
    return where is None or where(canonical)


def _attr_value(text: str) -> Any:
    """Index attribute values are text; JSON scalars (``true``, ``3``) are compared as such."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def _nest(attrs: Mapping[str, str]) -> dict[str, Any]:
    """``{"ontology.isTherapeuticArea": "true"}`` -> ``{"ontology": {"isTherapeuticArea": True}}``."""
    row: dict[str, Any] = {}
    for col, text in attrs.items():
        parts = col.split(".")
        node = row
        for part in parts[:-1]:
            nxt = node.setdefault(part, {})
            if not isinstance(nxt, dict):
                break
            node = nxt
        else:
            node[parts[-1]] = _attr_value(text)
    return row


class _Where:
    """``universe_where`` (``ArgBinding.universe_where``) as a test on the bound id_type's canonical keys."""

    def __init__(self, where: WherePredicate, index: ResolverIndex | None,
                 params: Mapping[str, Any] | None = None) -> None:
        self.index, self.params = index, dict(params or {})
        self.allowed: frozenset[str] | None = None
        self.call: Callable[[str], bool] | None = None
        self.pred: Any = None
        if isinstance(where, Mapping):
            try:
                self.pred = from_json(dict(where))
            except PredicateError as exc:
                raise ResolverConfigError(f"universe_where is not a predicate: {exc}") from None
        elif callable(where):
            self.call = where
        elif isinstance(where, (set, frozenset, list, tuple)):
            self.allowed = frozenset(str(v) for v in where)
        else:
            self.pred = where                          # a Predicate IR object

    def __call__(self, canonical: str) -> bool:
        if self.allowed is not None:
            return canonical in self.allowed
        if self.call is not None:
            return bool(self.call(canonical))
        if self.index is None:
            return False                               # cannot be shown to satisfy it (I6)
        return evaluate(self.pred, _nest(self.index.attributes(canonical)), self.params) is True

    def values(self) -> tuple[str, ...]:
        """The restricted universe when it can be listed (for ``invalid_argument.valid_values``)."""
        if self.allowed is not None:
            return tuple(sorted(self.allowed))
        if self.index is None:
            return ()
        return tuple(c for c in self.index.canonicals() if self(c))


def _distinct(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        if v and v not in out:
            out.append(v)
    return out


class Resolver:
    """Resolves identifier arguments against the catalog's id_types and their sidecar indexes.

    ``index_provider(source, id_type) -> ResolverIndex | None`` (or a ``{"source:id_type": index}``
    mapping) supplies local indexes; ``remote(source, id_type, values) -> {"existence", "resolutions":
    [{"value", "canonical", "rule", "label"}]}`` answers ``index: remote`` and ``universe_via`` id_types
    (it may be async: use :meth:`aresolve`); an exception from it (budget, outage) gives ``unknown``."""

    def __init__(self, registry: Any, catalog: Any, index_provider: IndexProvider | Mapping[str, ResolverIndex] | None,
                 remote: Callable[..., Any] | None = None, settings: Any = None) -> None:
        self.registry = registry
        self.catalog = catalog
        if isinstance(index_provider, Mapping):
            table = dict(index_provider)
            self.index_provider: IndexProvider = lambda s, t: table.get(f"{s}:{t}")
        else:
            self.index_provider = index_provider or (lambda s, t: None)
        self.remote = remote
        res = getattr(settings, "resolution", settings)
        self.settings: ResolutionSettings = res if res is not None else ResolutionSettings()
        self._plugins: dict[str, tuple[ResolverIndex | None, Any]] = {}   # per id_type: (index, configured plugin)
        self._graph: dict[str, list[tuple[str, _Edge]]] | None = None

    # ------------------------------------------------------------------ catalog helpers

    def qualify(self, name: str, source_hint: str | None = None) -> str:
        try:
            return self.catalog.qualify_id_type(name, source_hint)
        except LookupError as exc:
            raise ResolverConfigError(str(exc)) from None

    def spec(self, qualified: str) -> tuple[str, Any]:
        try:
            return self.catalog.id_type(qualified)
        except LookupError as exc:
            raise ResolverConfigError(str(exc)) from None

    def index(self, qualified: str) -> ResolverIndex | None:
        src, _, bare = qualified.partition(":")
        return self.index_provider(src, bare)

    def plugin(self, qualified: str) -> Any:
        """The id_type's identifier plugin, configured with its options and the index's universe sample."""
        src, spec = self.spec(qualified)
        index = self.index(qualified)
        hit = self._plugins.get(qualified)
        if hit is not None and hit[0] is index:
            return hit[1]
        base = self.registry.find("identifier", spec.plugin) if self.registry is not None else None
        if base is None:
            raise ResolverConfigError(f"id_type {qualified} names identifier plugin {spec.plugin!r}, which is not "
                                      "registered")
        sample = index.canonicals() if index is not None else None
        try:
            configured = base.configure(dict(spec.options), sample) if spec.options else base
        except Exception as exc:  # noqa: BLE001 - bad options are a configuration defect
            raise ResolverConfigError(f"id_type {qualified}: plugin {spec.plugin} rejects its options: {exc}") from None
        self._plugins[qualified] = (index, configured)
        return configured

    def _universe_state(self, qualified: str) -> tuple[str, ResolverIndex | None]:
        """``remote`` | ``local`` | ``missing`` (declared universe, index not built) | ``none`` (no universe)."""
        _, spec = self.spec(qualified)
        if spec.index == "remote" or spec.universe_via is not None:
            return "remote", None
        index = self.index(qualified)
        if index is not None:
            return "local", index
        return ("none" if spec.universe is None else "missing"), None

    def _source_info(self, qualified: str) -> tuple[str | None, str | None]:
        src, spec = self.spec(qualified)
        desc = self.catalog.source(src)
        rel = desc.release.expect
        source = f"{src}@{rel}" if rel else src
        u = spec.universe
        if isinstance(u, list):
            u = u[0] if u else None
        table = getattr(u, "table", None) if u is not None and not isinstance(u, str) else \
            (str(u).split(".")[0] if u else None)
        return source, table

    # ------------------------------------------------------------------ edges

    def _edges(self) -> dict[str, list[tuple[str, _Edge]]]:
        if self._graph is not None:
            return self._graph
        graph: dict[str, list[tuple[str, _Edge]]] = {}

        def add(a: str, b: str, edge: _Edge) -> None:
            graph.setdefault(a, [])
            if all(n != b for n, _ in graph[a]):
                graph[a].append((b, edge))

        def q(ref: str, src: str) -> str:
            return ref if ":" in ref else f"{src}:{ref}"

        sources = self.catalog.sources
        for src in sorted(sources):
            for name, spec in sorted(sources[src].id_types.items()):
                me = f"{src}:{name}"
                if spec.label_of:
                    add(me, q(spec.label_of, src), _Edge("label_of"))
                    add(q(spec.label_of, src), me, _Edge("label_rev"))
                for m in spec.union:                  # a member's key is a union key; not the reverse
                    add(q(m, src), me, _Edge("union_member"))
                if spec.extends:
                    add(me, q(spec.extends, src), _Edge("identity"))
                    add(q(spec.extends, src), me, _Edge("identity"))
                for mt in spec.maps_to:
                    other = q(mt.id_type, src)
                    own = {c.name for c in spec.crosswalks}
                    theirs: set[str] = set()
                    try:
                        _, ospec = self.catalog.id_type(other)
                        theirs = {c.name for c in ospec.crosswalks}
                    except LookupError:
                        pass
                    if mt.via and mt.via in own:          # rows in my index: their keys -> my keys
                        add(other, me, _Edge("crosswalk_fwd", mt.via, me))
                        add(me, other, _Edge("crosswalk_rev", mt.via, me))
                    elif mt.via and mt.via in theirs:     # rows in their index: my keys -> their keys
                        add(me, other, _Edge("crosswalk_fwd", mt.via, other))
                        add(other, me, _Edge("crosswalk_rev", mt.via, other))
                    elif mt.via and spec.index == "remote" and mt.via.split(".")[-1] in sources[src].tables:
                        # a huge universe mapped through a source table (rsid -> ot_variant via variant): the
                        # data child translates each value by a bounded scan; several targets are ambiguous
                        add(me, other, _Edge("remote_map", mt.via, other))
                        add(other, me, _Edge("identity"))
                    else:
                        add(me, other, _Edge("identity"))
                        add(other, me, _Edge("identity"))
        self._graph = graph
        return graph

    def path(self, start: str, goal: str) -> list[tuple[str, str, _Edge]] | None:
        """The shortest declared edge path ``[(from, to, edge), ...]`` from ``start`` to ``goal`` (None: none)."""
        if start == goal:
            return []
        graph = self._edges()
        prev: dict[str, tuple[str, _Edge]] = {}
        seen = {start}
        todo = deque([start])
        while todo:
            node = todo.popleft()
            for nxt, edge in graph.get(node, ()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = (node, edge)
                if nxt == goal:
                    out = []
                    cur = goal
                    while cur != start:
                        p, e = prev[cur]
                        out.append((p, cur, e))
                        cur = p
                    return out[::-1]
                todo.append(nxt)
        return None

    # ------------------------------------------------------------------ public API

    def resolve(self, value: Any, accepts: Sequence[str], *, bound_id_type: str | None = None,
                existence: str = "universe", universe_where: WherePredicate | None = None,
                where_params: Mapping[str, Any] | None = None, source_hint: str | None = None) -> ResolutionResult:
        """Resolve one value (see the module docstring). ``where_params`` supplies ``{"param": name}`` values of a
        ``universe_where`` predicate."""
        return self._resolve(value, accepts, bound_id_type, existence, universe_where, where_params, source_hint,
                             None)

    async def aresolve(self, value: Any, accepts: Sequence[str], *, bound_id_type: str | None = None,
                       existence: str = "universe", universe_where: WherePredicate | None = None,
                       where_params: Mapping[str, Any] | None = None,
                       source_hint: str | None = None) -> ResolutionResult:
        """:meth:`resolve` with an async ``remote`` callable: each remote question is awaited once, then the
        resolution runs again with the answer (resolution itself never blocks)."""
        cache: dict[tuple[str, str, tuple[str, ...]], Any] = {}
        while True:
            try:
                return self._resolve(value, accepts, bound_id_type, existence, universe_where, where_params,
                                     source_hint, cache)
            except _NeedRemote as need:
                cache[need.key] = await self._remote_async(need.key)

    def resolve_list(self, values: Sequence[Any], accepts: Sequence[str], *, bound_id_type: str | None = None,
                     existence: str = "universe", universe_where: WherePredicate | None = None,
                     where_params: Mapping[str, Any] | None = None, source_hint: str | None = None,
                     on_missing: str = "error",
                     min_resolved_fraction: float | None = None,
                     dedupe: bool = True) -> tuple[list[ResolutionResult], dict[str, Any]]:
        """Resolve each element (``each: true``). Returns one result per input value and the
        ``_vbt.resolution_summary``: ``{status, requested, resolved, unresolved[], ambiguous[],
        outside_universe[], duplicates[], unknown[], items[{index, value, reason, status}], canonicals[]}``.

        ``status`` is ``insufficient`` when fewer than ``min_resolved_fraction`` of the distinct requested
        entities resolved, ``failed`` when an element failed under ``on_missing: error`` (no partial call is
        made), ``partial`` when failed elements are dropped (``partial``/``drop_disclosed``), else ``ok``.
        A symbol and its Ensembl ID count once (``duplicates``); ``canonicals`` are the deduplicated keys to send."""
        results = [self.resolve(v, accepts, bound_id_type=bound_id_type, existence=existence,
                                universe_where=universe_where, where_params=where_params, source_hint=source_hint)
                   for v in values]
        return results, self.summarize(values, results, on_missing=on_missing,
                                       min_resolved_fraction=min_resolved_fraction, dedupe=dedupe)

    async def aresolve_list(self, values: Sequence[Any], accepts: Sequence[str], *, bound_id_type: str | None = None,
                            existence: str = "universe", universe_where: WherePredicate | None = None,
                            where_params: Mapping[str, Any] | None = None, source_hint: str | None = None,
                            on_missing: str = "error",
                            min_resolved_fraction: float | None = None,
                            dedupe: bool = True) -> tuple[list[ResolutionResult], dict[str, Any]]:
        results = [await self.aresolve(v, accepts, bound_id_type=bound_id_type, existence=existence,
                                       universe_where=universe_where, where_params=where_params,
                                       source_hint=source_hint) for v in values]
        return results, self.summarize(values, results, on_missing=on_missing,
                                       min_resolved_fraction=min_resolved_fraction, dedupe=dedupe)

    def summarize(self, values: Sequence[Any], results: Sequence[ResolutionResult], *, on_missing: str = "error",
                  min_resolved_fraction: float | None = None, dedupe: bool = True) -> dict[str, Any]:
        canonicals: list[str] = []
        seen_raw: set[str] = set()
        duplicates: list[Any] = []
        unresolved: list[Any] = []
        ambiguous: list[Any] = []
        outside: list[Any] = []
        unknown: list[Any] = []
        items: list[dict[str, Any]] = []
        resolved = 0
        for i, (v, r) in enumerate(zip(values, results)):
            raw_key = as_text(v).strip()
            if dedupe and raw_key in seen_raw:
                duplicates.append(v)
                continue
            seen_raw.add(raw_key)
            if r.ok and r.canonical is not None:
                if dedupe and r.canonical in canonicals:
                    duplicates.append(v)
                    continue
                canonicals.append(r.canonical)
                resolved += 1
                if r.status == "unknown":
                    unknown.append(v)
                continue
            if r.status == "unknown":
                unknown.append(v)
                unresolved.append(v)
                items.append({"index": i, "value": v, "reason": "existence could not be decided", "status": r.status})
                continue
            if r.status == "ambiguous":
                ambiguous.append(v)
                reason = "ambiguous: " + ", ".join(c.id for c in r.candidates[:5])
            elif r.subkind == "outside_universe":
                outside.append(v)
                reason = r.reasons[0] if r.reasons else "outside the allowed values"
            else:
                unresolved.append(v)
                reason = (r.reasons[0] if r.reasons else None) or (r.tried[-1] if r.tried else r.status)
                reason = f"{r.status}: {reason}"
            items.append({"index": i, "value": v, "reason": reason, "status": r.status,
                          "looks_like": list(r.looks_like)})
        requested = len(values) - len(duplicates)
        fraction = resolved / requested if requested else 1.0
        if min_resolved_fraction is not None and fraction < float(min_resolved_fraction):
            status = "insufficient"
        elif items and on_missing == "error":
            status = "failed"
        elif items:
            status = "partial"
        else:
            status = "ok"
        return {"status": status, "requested": requested, "resolved": resolved, "unresolved": unresolved,
                "ambiguous": ambiguous, "outside_universe": outside, "duplicates": duplicates, "unknown": unknown,
                "items": items, "canonicals": canonicals, "on_missing": on_missing,
                "min_resolved_fraction": min_resolved_fraction}

    def send_value(self, res: ResolutionResult, send_as: str = "canonical", table: str | None = None) -> Any:
        """The value to send upstream (``ArgBinding.send_as``): ``canonical``; ``raw``; ``label`` (the key's
        label); ``stored`` (``table``'s own spelling, ``'Erdafitinib '``); ``native_label`` (``table``'s own
        label, ``SEPT9`` for ``SEPTIN9``). Falls back to the canonical key when the index has no such form."""
        if send_as == "raw" or res.canonical is None:
            return res.raw
        if send_as in ("canonical", "alias") or res.id_type is None:
            return res.canonical
        index = self.index(res.id_type)
        if index is None:
            return res.canonical
        if send_as == "stored" and table:
            stored = index.stored_value(res.canonical, table)
            return stored if stored is not None else res.canonical
        if send_as == "native_label" and table:
            native = index.native_label(res.canonical, table)
            if native is not None:
                return native
            return index.label(res.canonical) or res.canonical
        if send_as == "label":
            return index.label(res.canonical) or res.canonical
        return res.canonical

    # ------------------------------------------------------------------ core

    def _resolve(self, value: Any, accepts: Sequence[str], bound_id_type: str | None, existence: str,
                 universe_where: WherePredicate | None, where_params: Mapping[str, Any] | None,
                 source_hint: str | None,
                 cache: dict[tuple[str, str, tuple[str, ...]], Any] | None) -> ResolutionResult:
        if existence not in EXISTENCE_MODES:
            raise ValueError(f"existence must be one of {EXISTENCE_MODES}, got {existence!r}")
        text = as_text(value)
        bound = self.qualify(bound_id_type, source_hint) if bound_id_type else None
        hint = bound.partition(":")[0] if bound else source_hint
        kinds = _distinct(self.qualify(a, hint) for a in accepts) or ([bound] if bound else [])
        if not kinds:
            raise ResolverConfigError("nothing to resolve against: no accepted kinds and no bound id_type")
        tried: list[str] = []
        paths: dict[str, list[tuple[str, str, _Edge]]] = {}
        for k in kinds:
            p = [] if bound is None else self.path(k, bound)
            if p is None:
                tried.append(f"{k}: no declared edge to {bound}")
            elif len(p) > self.settings.max_hops:
                tried.append(f"{k}: reaching {bound} takes {len(p)} hops (data.resolution.max_hops = "
                             f"{self.settings.max_hops})")
            else:
                paths[k] = p
        if not paths:
            raise ResolverConfigError(f"no accepted kind of {kinds} reaches the bound id_type {bound}: "
                                      + "; ".join(tried))
        usable = [k for k in kinds if k in paths]
        label_parents = {self.qualify(self.spec(k)[1].label_of, k.partition(":")[0])
                         for k in usable if self.spec(k)[1].label_of}
        target = bound or usable[0]
        where = None if universe_where is None else _Where(universe_where, self.index(target), where_params)
        ctx = _Ctx(text=text, value=value, existence=existence, where=where, cache=cache, tried=tried,
                   accepts=tuple(usable), bound=target)

        rejected: list[_Hit] = []
        absent: tuple[_Hit, list[tuple[str, str, _Edge]]] | None = None
        unknown: _Hit | None = None
        for k in usable:
            # universe_where restricts the bound id_type's keys: applied while matching only when no hop follows
            hit = self._attempt(ctx, k, defer_labels=k in label_parents, where=where if not paths[k] else None)
            if hit.status in ("resolved", "ambiguous", "obsolete", "outside"):
                return self._finish(ctx, hit, paths[k])
            if hit.status == "rejected":
                rejected.append(hit)
            elif hit.status == "absent" and absent is None:
                absent = (hit, paths[k])
            elif hit.status == "unknown" and unknown is None:
                unknown = hit
        if unknown is not None:                        # a kind that could not decide outranks a miss (I2)
            return self._result(ctx, "unknown", canonical=unknown.canonical, matched=unknown.matched,
                                existence="unknown", notes=unknown.notes)
        if absent is not None:
            return self._absent(ctx, *absent)
        return self._rejected(ctx, rejected)

    # -- one accepted kind --------------------------------------------------

    def _attempt(self, ctx: "_Ctx", kind: str, *, defer_labels: bool, where: "_Where | None") -> _Hit:
        src, spec = self.spec(kind)
        if spec.union:
            return self._attempt_union(ctx, kind, where)
        if spec.label_of:
            return self._attempt_label(ctx, kind, self.qualify(spec.label_of, src), where)
        plugin = self.plugin(kind)
        n = plugin.normalize(ctx.text)
        state, index = self._universe_state(kind)
        if isinstance(n, Rejected):
            ctx.tried.append(f"{kind}: {n.reason}")
        if state == "remote":
            return self._attempt_remote(ctx, kind, n)
        rules = rules_for(spec, plugin, self.settings.allow)
        if defer_labels:                               # the accepted label kind applies them (matched_id_type)
            rules = [r for r in rules if r.head not in LABEL_HEADS]
        absent_value: str | None = None
        unknown = False
        for rule in rules:
            if rule.head in MEMBERSHIP_HEADS:
                if rule.head == "raw_member":
                    if state != "local":
                        continue
                    same = isinstance(n, Normalized) and n.value == ctx.text and not n.steps
                    candidate, rule_text = ctx.text, "exact" if same else "raw_member"
                elif not isinstance(n, Normalized):
                    continue
                elif rule.head == "exact":
                    if n.steps:
                        continue
                    candidate, rule_text = n.value, "exact"
                else:
                    if not n.steps or (rule.steps and not set(n.steps) <= set(rule.steps)):
                        continue
                    candidate, rule_text = n.value, "normalized:" + "+".join(n.steps)
                if state == "none":                    # no universe: syntax decides
                    return self._resolved(kind, kind, candidate, rule_text, where, existence=None)
                if state == "missing":
                    unknown = True
                    continue
                if index is not None and index.contains(candidate):
                    return self._resolved(kind, kind, candidate, rule_text, where, existence="exists", index=index)
                if rule.head != "raw_member":
                    absent_value = candidate
                    ctx.tried.append(f"{kind} {rule_text}: {candidate} is not in the universe")
                continue
            if index is None:
                continue
            hit = self._apply(ctx, kind, kind, rule, n, plugin, index, spec, where)
            if hit is not None:
                return hit
        needs_index = unknown or any(r.head not in MEMBERSHIP_HEADS for r in rules)
        if state == "missing" and (isinstance(n, Normalized) or needs_index):
            return _Hit("unknown", kind, kind, canonical=n.value if isinstance(n, Normalized) else None,
                        notes=[f"the resolver index of {kind} is not ready; {ctx.text!r} cannot be checked"])
        if isinstance(n, Rejected):
            return _Hit("rejected", kind, kind, reason=f"{kind}: {n.reason}", looks_like=n.looks_like)
        return _Hit("absent", kind, kind, canonical=absent_value or n.value)

    def _attempt_label(self, ctx: "_Ctx", kind: str, parent: str, where: "_Where | None") -> _Hit:
        _, spec = self.spec(kind)
        n = self.plugin(kind).normalize(ctx.text)
        if isinstance(n, Rejected):
            ctx.tried.append(f"{kind}: {n.reason}")
            return _Hit("rejected", parent, kind, reason=f"{kind}: {n.reason}", looks_like=n.looks_like)
        _, pspec = self.spec(parent)
        state, index = self._universe_state(parent)
        if index is None:
            note = f"the resolver index of {parent} is not ready; {ctx.text!r} cannot be resolved as a {kind}"
            return _Hit("unknown", parent, kind, notes=[note])
        pplugin = self.plugin(parent)
        rules = [parse_rule(r) for r in spec.rules] if spec.rules else \
            [r for r in rules_for(pspec, pplugin, self.settings.allow) if r.head in LABEL_HEADS]
        for rule in rules:
            if rule.head not in LABEL_HEADS or not allowed(rule, self.settings.allow) or rule.search_only:
                continue
            hit = self._apply(ctx, parent, kind, rule, n, pplugin, index, pspec, where)
            if hit is not None:
                return hit
        return _Hit("absent", parent, kind)

    def _attempt_union(self, ctx: "_Ctx", kind: str, where: "_Where | None") -> _Hit:
        src, spec = self.spec(kind)
        plugin = self.plugin(kind)
        n = plugin.normalize(ctx.text)
        state, index = self._universe_state(kind)
        if index is not None:                          # the value is itself a word of the union's universe
            options = [(ctx.text, "raw_member")]
            if isinstance(n, Normalized):
                options.insert(0, (n.value, "normalized:" + "+".join(n.steps) if n.steps else "exact"))
            for value, rule_text in options:
                if index.contains(value):
                    return self._resolved(kind, kind, value, rule_text, where, existence="exists", index=index)
        hits: list[_Hit] = []
        for m in spec.union:
            member = self.qualify(m, src)
            sub = _Ctx(text=ctx.text, value=ctx.value, existence=ctx.existence, where=None, cache=ctx.cache,
                       tried=ctx.tried, accepts=ctx.accepts, bound=kind)
            hits.append(self._attempt(sub, member, defer_labels=False, where=None))
        pool: list[Candidate] = []
        for h in hits:
            if h.status == "resolved" and h.canonical:
                pool.append(Candidate(h.canonical, self._label(h.kind, h.canonical), f"{h.kind} {h.rule}"))
            elif h.status == "ambiguous":
                pool.extend(h.candidates)
        ids = _distinct(c.id for c in pool if _where_ok(where, c.id))
        if len(ids) == 1:
            first = next(h for h in hits if h.canonical == ids[0] or any(c.id == ids[0] for c in h.candidates))
            return _Hit("resolved", kind, first.matched, canonical=ids[0], rule=first.rule or "exact",
                        family=first.family,
                        notes=first.notes + [f"{ctx.text!r} is a {first.kind} value of the union {kind}"])
        if len(ids) > 1:
            cands = [c for c in pool if c.id in ids]
            return _Hit("ambiguous", kind, kind, candidates=_unique_candidates(cands),
                        notes=[f"{ctx.text!r} names different entities in the member kinds of {kind}"])
        if pool:
            return _Hit("outside", kind, kind, canonical=pool[0].id)
        if any(h.status == "unknown" for h in hits):
            return _Hit("unknown", kind, kind, notes=[n for h in hits for n in h.notes])
        if isinstance(n, Rejected) and all(h.status == "rejected" for h in hits):
            return _Hit("rejected", kind, kind, reason=f"{kind}: {n.reason}",
                        looks_like=tuple(x for h in hits for x in h.looks_like))
        return _Hit("absent", kind, kind, canonical=n.value if isinstance(n, Normalized) else None)

    def _attempt_remote(self, ctx: "_Ctx", kind: str, n: Normalized | Rejected) -> _Hit:
        if isinstance(n, Rejected):
            return _Hit("rejected", kind, kind, reason=f"{kind}: {n.reason}", looks_like=n.looks_like)
        src, _, bare = kind.partition(":")
        try:
            resp = self._remote(ctx, src, bare, (n.value,))
        except _NeedRemote:
            raise
        except Exception as exc:  # noqa: BLE001 - any remote failure (budget, outage) leaves existence unknown
            return _Hit("unknown", kind, kind, canonical=n.value,
                        notes=[f"existence of {n.value} in {kind} is unknown: remote universe failed ({exc})"])
        resp = dict(resp or {})
        state = str(resp.get("existence") or "unknown")
        rows = [r for r in resp.get("resolutions") or () if r.get("value") in (None, n.value, ctx.text)]
        canon = _distinct(str(r.get("canonical")) for r in rows if r.get("canonical"))
        rule_text = "exact" if not n.steps else "normalized:" + "+".join(n.steps)
        if state == "exists":
            if len(canon) > 1:
                cands = [Candidate(c, r.get("label"), r.get("rule") or rule_text) for c, r in
                         zip(canon, rows)]
                return _Hit("ambiguous", kind, kind, candidates=cands,
                            notes=[f"{n.value} names {len(canon)} records in {kind} (cardinality many)"])
            rule = str(rows[0].get("rule") or rule_text) if rows else rule_text
            return _Hit("resolved", kind, kind, canonical=canon[0] if canon else n.value, rule=rule,
                        existence="exists")
        if state == "absent":
            ctx.tried.append(f"{kind} {rule_text}: {n.value} is not in the remote universe")
            return _Hit("absent", kind, kind, canonical=n.value)
        return _Hit("unknown", kind, kind, canonical=n.value,
                    notes=[f"existence of {n.value} in {kind} could not be decided remotely"])

    # -- rules ----------------------------------------------------------------

    def _apply(self, ctx: "_Ctx", kind: str, matched: str, rule: Rule, n: Normalized | Rejected, plugin: Any,
               index: ResolverIndex, spec: Any, where: "_Where | None") -> _Hit | None:
        """Apply one non-membership rule on ``kind``'s index; None when nothing matched."""
        rows = self._match(rule, ctx.text, n, plugin, index, spec)
        if not rows:
            ctx.tried.append(f"{matched} {rule.text}: none")
            return None
        if rule.head == "retired":
            return self._retired(ctx, kind, matched, rows, index, spec, where)
        groups: dict[str, Entry] = {}
        for e in rows:
            groups.setdefault(e.canonical, e)
        ids = sorted(groups)
        inside = [c for c in ids if _where_ok(where, c)]
        if not inside:
            return _Hit("outside", kind, matched, canonical=ids[0], rule=self._rule_text(rule, groups[ids[0]]))
        if len(inside) == 1:
            c = inside[0]
            hit = self._resolved(kind, matched, c, self._rule_text(rule, groups[c]), None, index=index,
                                 existence="exists" if index.contains(c) else None)
            if rule.head == "xref":
                ns = groups[c].arg
                hit.notes.append(f"{ctx.text!r} is a {self._xref_type(spec, ns) or ns} cross-reference of {c}")
            return hit
        rule_text = self._rule_text(rule, groups[inside[0]])
        collapsed = self._collapse(ctx, kind, matched, inside, rule_text, index, spec)
        if collapsed is not None:
            return collapsed
        cands = [Candidate(c, index.label(c) or groups[c].label or None, self._rule_text(rule, groups[c]),
                           self._attrs(index, spec, c)) for c in inside]
        return _Hit("ambiguous", kind, matched, rule=rule_text, candidates=cands,
                    notes=[f"{ctx.text!r} matches {len(inside)} {kind} keys by {rule_text}"])

    def _match(self, rule: Rule, text: str, n: Normalized | Rejected, plugin: Any, index: ResolverIndex,
               spec: Any) -> list[Entry]:
        head = rule.head
        keys = [plugin.label_key(text)]
        if isinstance(n, Normalized) and head in ("retired", "xref", "crosswalk", "stored_form"):
            keys.append(plugin.label_key(n.value))
        if head == "xref":
            keys.extend(plugin.label_key(v) for v in _curie_variants(text))
        rows: list[Entry] = []
        for key in _distinct(keys):
            rows.extend(index.lookup(key))
        out: list[Entry] = []
        stripped = text.strip()
        for e in rows:
            eh = e.head
            if head in ("label_exact", "label_casefold"):
                if eh not in ("label_exact", "label_casefold") or (rule.arg and e.arg != rule.arg):
                    continue
                if head == "label_exact" and e.label.strip() != stripped:
                    continue
            elif head == "synonym":
                kind = e.arg or ""
                if eh != "synonym" or kind in SEARCH_ONLY_SYNONYMS:
                    continue
                if rule.arg and kind != rule.arg:
                    continue
                if not rule.arg and not allowed(Rule("synonym", kind), self.settings.allow):
                    continue
            elif head == "stored_form":
                if eh != "stored_form" or (e.stored_value != text and e.stored_value.strip() != stripped):
                    continue
            elif head == "xref":
                if eh != "xref" or (rule.arg and (e.arg or "").casefold() != rule.arg.casefold()):
                    continue
                if not self._xref_declared(spec, e.arg):
                    continue
            elif head in ("retired", "crosswalk"):
                if eh != head:
                    continue
                if rule.arg and e.arg not in (rule.chain or (rule.arg,)):
                    continue
            else:
                continue
            if head != "retired" and not e.canonical:
                continue
            if e not in out:
                out.append(e)
        return out

    @staticmethod
    def _xref_declared(spec: Any, namespace: str | None) -> bool:
        declared: set[str] = set()
        for x in getattr(spec, "xref_via", ()) or ():
            declared.update(k.casefold() for k in x.namespaces)
            if x.id_type_from is not None:
                declared.update(k.casefold() for k in x.id_type_from.map)
        return not declared or (namespace or "").casefold() in declared

    @staticmethod
    def _xref_type(spec: Any, namespace: str | None) -> str | None:
        for x in getattr(spec, "xref_via", ()) or ():
            for k, v in list(x.namespaces.items()) + (list(x.id_type_from.map.items()) if x.id_type_from else []):
                if k.casefold() == (namespace or "").casefold():
                    return v
        return None

    @staticmethod
    def _rule_text(rule: Rule, e: Entry) -> str:
        if rule.head in ("label_exact", "label_casefold"):
            return f"{rule.head}:{rule.arg or e.arg}" if (rule.arg or e.arg) else rule.head
        if rule.head in ("synonym", "retired", "xref", "crosswalk"):
            return f"{rule.head}:{e.arg}" if e.arg else rule.text
        return rule.text

    def _retired(self, ctx: "_Ctx", kind: str, matched: str, rows: list[Entry], index: ResolverIndex, spec: Any,
                 where: "_Where | None") -> _Hit:
        consider_col = getattr(getattr(spec, "retired", None), "consider", None)
        consider_col = consider_col.rpartition(".")[2] if consider_col else None
        definitive = [e for e in rows if e.canonical and e.arg != consider_col]
        consider = [e for e in rows if e.canonical and e.arg == consider_col]
        rule_text = f"retired:{rows[0].arg}" if rows[0].arg else "retired"
        repl = _distinct(e.canonical for e in definitive if _where_ok(where, e.canonical))
        if len(repl) == 1:
            e = next(x for x in definitive if x.canonical == repl[0])
            note = f"{ctx.text!r} is a retired ID; it was replaced by {repl[0]} ({e.rule})"
            hit = self._resolved(kind, matched, repl[0], e.rule, None, index=index,
                                 existence="exists" if index.contains(repl[0]) else None)
            hit.notes.insert(0, note)
            return hit
        options = repl or _distinct(e.canonical for e in consider if _where_ok(where, e.canonical))
        if len(options) > 1:
            return _Hit("ambiguous", kind, matched, rule=rule_text,
                        candidates=[Candidate(c, index.label(c), rule_text, self._attrs(index, spec, c))
                                    for c in options],
                        notes=[f"{ctx.text!r} is a retired ID with several replacement candidates"])
        return _Hit("obsolete", kind, matched, rule=rule_text, replacement=tuple(options),
                    notes=[f"{ctx.text!r} is a retired ID without a single replacement"
                           + (f"; consider {options[0]}" if options else "")])

    def _collapse(self, ctx: "_Ctx", kind: str, matched: str, ids: list[str], rule_text: str, index: ResolverIndex,
                  spec: Any) -> _Hit | None:
        """Parent families, then a declared ``prefer`` order; None when the candidates stay ambiguous."""
        if spec.canonicalize is not None and allowed(Rule("parent_family"), self.settings.allow):
            parents = {index.parent(c) or c for c in ids}
            if len(parents) == 1:
                parent = parents.pop()
                if index.contains(parent):
                    hit = self._resolved(kind, matched, parent, "parent_family", None, index=index, existence="exists")
                    hit.notes.insert(0, f"{ctx.text!r} matched {', '.join(ids)} by {rule_text}; they are forms of "
                                        f"one parent, {parent}")
                    return hit
                ctx.tried.append(f"{matched} parent_family: parent {parent} is not in the universe")
        for prefix in spec.prefer:
            chosen = [c for c in ids if c.startswith(prefix)]
            if len(chosen) == 1:
                hit = self._resolved(kind, matched, chosen[0], rule_text, None, index=index,
                                     existence="exists" if index.contains(chosen[0]) else None)
                others = ", ".join(c for c in ids if c != chosen[0])
                hit.notes.insert(0, f"{ctx.text!r} also matches {others}; {chosen[0]} chosen by the declared prefer "
                                    f"order {list(spec.prefer)}")
                return hit
            if chosen:
                break
        return None

    def _resolved(self, kind: str, matched: str, canonical: str, rule: str, where: "_Where | None", *,
                  existence: str | None, index: ResolverIndex | None = None) -> _Hit:
        if not _where_ok(where, canonical):
            return _Hit("outside", kind, matched, canonical=canonical, rule=rule)
        family: tuple[str, ...] = ()
        _, spec = self.spec(kind)
        if spec.canonicalize is not None and index is not None:
            fam = index.family(canonical)
            family = fam if len(fam) > 1 else ()
        return _Hit("resolved", kind, matched, canonical=canonical, rule=rule, family=family, existence=existence)

    @staticmethod
    def _attrs(index: ResolverIndex, spec: Any, canonical: str) -> dict[str, Any]:
        attrs = index.attributes(canonical)
        return {col: attrs.get(col) for col in spec.disambiguate_with}

    def _label(self, kind: str, canonical: str) -> str | None:
        index = self.index(kind)
        return index.label(canonical) if index is not None else None

    # -- after a decisive hit -------------------------------------------------

    def _finish(self, ctx: "_Ctx", hit: _Hit, path: list[tuple[str, str, _Edge]]) -> ResolutionResult:
        notes = list(hit.notes)
        if hit.status == "ambiguous":
            return self._result(ctx, "ambiguous", matched=hit.matched, rule=hit.rule,
                                candidates=hit.candidates[: self.settings.max_candidates], notes=notes)
        if hit.status == "obsolete":
            return self._result(ctx, "not_found", matched=hit.matched, rule=hit.rule, subkind="obsolete",
                                replacement=hit.replacement, notes=notes)
        if hit.status == "outside":
            return self._outside(ctx, hit)
        kind, hops = hit.kind, list(hit.hops)
        first, first_rule = str(hit.canonical), hit.rule or "exact"
        mapped = self._follow(ctx, first, first_rule, path, hops)   # path starts at the accepted kind
        if isinstance(mapped, ResolutionResult):
            return mapped
        canonical, rule = mapped
        if canonical != first:
            notes.insert(0, f"{ctx.text!r} -> {first} ({first_rule}) -> {canonical} ({rule})")
        target = ctx.bound
        family = hit.family if target == kind else ()
        if target != kind:
            _, tspec = self.spec(target)
            tindex = self.index(target)
            if tspec.canonicalize is not None and tindex is not None:
                fam = tindex.family(canonical)
                family = fam if len(fam) > 1 else ()
        if ctx.where is not None and path and not _where_ok(ctx.where, canonical):
            return self._outside(ctx, _Hit("outside", target, hit.matched, canonical=canonical, rule=rule))
        existence = hit.existence if not path else None
        status = "resolved"
        if existence is None or path:
            existence, status, extra = self._existence(ctx, target, canonical)
            notes.extend(extra)
            if status == "not_found":
                return self._result(ctx, "not_found", canonical=None, matched=hit.matched, rule=rule,
                                    hops=hops, notes=notes, existence=existence,
                                    suggestions=self._suggest(target, ctx.text))
        if rule not in ("exact", "raw_member") and canonical == first:
            notes.insert(0, f"{ctx.text!r} resolved to {canonical} ({target}) by {rule}")
        if rule == "synonym:related":
            notes.append(f"warning: {ctx.text!r} is a related (not exact) synonym of {canonical}; check that it is "
                         "the intended entity")
        if len(family) > 1:
            notes.append(f"{canonical} has family members {', '.join(f for f in family if f != canonical)}")
        return self._result(ctx, status, canonical=canonical, matched=hit.matched, rule=rule,
                            family=family, hops=hops, notes=notes, existence=existence)

    def _follow(self, ctx: "_Ctx", canonical: str, rule: str, path: list[tuple[str, str, _Edge]],
                hops: list[str]) -> tuple[str, str] | ResolutionResult:
        """Map ``canonical`` along ``path``; returns ``(canonical, rule)`` or a terminal result."""
        crosswalks = list(Rule("crosswalk", rule.partition(":")[2]).chain) if rule.startswith("crosswalk:") else []
        for a, b, edge in path:
            if edge.kind in ("label_of", "union_member", "identity"):
                hops.append(f"{a}>{b} ({edge.kind})")
                continue
            if edge.kind == "remote_map":
                mapped = self._remote_map(ctx, a, b, canonical, rule, hops)
                if isinstance(mapped, ResolutionResult):
                    return mapped
                canonical, rule = mapped
                continue
            if edge.kind == "label_rev":
                label = self._label(a, canonical)
                if label is None:
                    return self._result(ctx, "not_found", matched=a, rule=rule,
                                        notes=[f"{canonical} has no label for {b}"])
                hops.append(f"{a}>{b} (label)")
                canonical = label
                continue
            owner = edge.owner or b
            index = self.index(owner)
            if index is None:
                return self._result(ctx, "unknown", canonical=None, matched=a, existence="unknown",
                                    notes=[f"the resolver index of {owner} (crosswalk {edge.name}) is not ready"])
            if edge.kind == "crosswalk_fwd":
                key = self.plugin(owner).label_key(canonical)
                rows = [e for e in index.lookup(key) if e.head == "crosswalk" and e.arg == edge.name]
                targets = _distinct(e.canonical for e in rows)
            else:
                rows = [e for e in index.rows_for(canonical, "crosswalk") if e.arg == edge.name]
                targets = _distinct(e.label for e in rows)
            if not targets:
                return self._result(ctx, "not_found", matched=a, rule=rule,
                                    notes=[f"{canonical} has no {edge.name} crosswalk row to {b}"],
                                    tried_extra=[f"{a}>{b} crosswalk:{edge.name}: no row for {canonical}"])
            if len(targets) > 1:
                cands = [Candidate(t, self._label(b, t), f"crosswalk:{edge.name}") for t in targets]
                return self._result(ctx, "ambiguous", matched=a, rule=f"crosswalk:{edge.name}",
                                    candidates=cands[: self.settings.max_candidates],
                                    notes=[f"{canonical} maps to {len(targets)} {b} keys through {edge.name}"])
            hops.append(f"{a}>{b} (crosswalk:{edge.name}: {canonical} -> {targets[0]})")
            canonical = targets[0]
            crosswalks.append(str(edge.name))
        if crosswalks:
            rule = "crosswalk:" + ">".join(_distinct(crosswalks))
        return canonical, rule

    def _remote_map(self, ctx: "_Ctx", a: str, b: str, canonical: str, rule: str, hops: list[str]
                    ) -> tuple[str, str] | ResolutionResult:
        """``a`` -> ``b`` through the data child (``_resolve_remote`` with id_type ``a>b``): one target maps,
        none is not_found, several are ambiguous (``cardinality: many``), an undecidable scan is unknown."""
        src, _, a_bare = a.partition(":")
        b_bare = b.partition(":")[2] or b
        try:
            resp = dict(self._remote(ctx, src, f"{a_bare}>{b_bare}", (canonical,)) or {})
        except _NeedRemote:
            raise
        except Exception as exc:  # noqa: BLE001 - a remote failure never reads as not found
            return self._result(ctx, "unknown", canonical=None, matched=a, existence="unknown",
                                notes=[f"{canonical} could not be mapped to {b}: {exc}"])
        state = str(resp.get("existence") or "unknown")
        targets = _distinct(str(r.get("canonical")) for r in resp.get("resolutions") or () if r.get("canonical"))
        if state == "unknown" and not targets:
            return self._result(ctx, "unknown", canonical=None, matched=a, existence="unknown",
                                notes=[f"{canonical} could not be mapped to {b} (the scan was undecided)"])
        if not targets:
            return self._result(ctx, "not_found", matched=a, rule=rule,
                                notes=[f"{canonical} names no {b} record"],
                                tried_extra=[f"{a}>{b}: no record for {canonical}"])
        if len(targets) > 1:
            cands = [Candidate(t, None, f"maps_to:{b_bare}") for t in targets]
            return self._result(ctx, "ambiguous", matched=a, rule=f"maps_to:{b_bare}",
                                candidates=cands[: self.settings.max_candidates],
                                notes=[f"{canonical} names {len(targets)} {b} records (cardinality many); pass one"])
        hops.append(f"{a}>{b} (maps_to: {canonical} -> {targets[0]})")
        return targets[0], (rule if rule != "exact" else f"maps_to:{b_bare}")

    def _existence(self, ctx: "_Ctx", target: str, canonical: str) -> tuple[str | None, str, list[str]]:
        """``(existence, status, notes)`` of ``canonical`` in ``target``'s universe under ``ctx.existence``."""
        state, index = self._universe_state(target)
        if state == "none":
            return None, "resolved", []
        if state == "remote":
            if ctx.existence in ("upstream", "off"):
                return None, "resolved", []
            src, _, bare = target.partition(":")
            try:
                resp = dict(self._remote(ctx, src, bare, (canonical,)) or {})
                found = str(resp.get("existence") or "unknown")
            except _NeedRemote:
                raise
            except Exception as exc:  # noqa: BLE001
                return "unknown", "unknown", [f"existence of {canonical} in {target} is unknown ({exc})"]
        elif state == "missing":
            if ctx.existence in ("upstream", "off"):
                return None, "resolved", []
            return "unknown", "unknown", [f"the resolver index of {target} is not ready; existence of {canonical} is "
                                          "unknown"]
        else:
            assert index is not None
            found = "exists" if index.contains(canonical) else "absent"
        if found == "exists":
            return "exists", "resolved", []
        if found == "unknown":
            return "unknown", "unknown", [f"existence of {canonical} in {target} could not be decided"]
        if ctx.existence == "universe":
            ctx.tried.append(f"{target}: {canonical} is not in the universe")
            return "absent", "not_found", []
        if ctx.existence == "bound":
            return "absent", "resolved_unverified", [f"{canonical} is not in the {target} universe; existence is "
                                                     "decided by the bound column"]
        return None, "resolved", [f"{canonical} is not in the {target} universe (existence checked upstream)"]

    def _absent(self, ctx: "_Ctx", hit: _Hit, path: list[tuple[str, str, _Edge]]) -> ResolutionResult:
        """No rule resolved the value, but some kind accepted its syntax."""
        _, spec = self.spec(hit.matched)
        if hit.canonical is not None and not spec.label_of and ctx.existence != "universe":
            hops: list[str] = []
            mapped = self._follow(ctx, hit.canonical, "exact", path, hops)
            if isinstance(mapped, ResolutionResult) and mapped.status != "not_found":
                return mapped                          # ambiguous hop or an index that is not ready
            if not isinstance(mapped, ResolutionResult):
                canonical, rule = mapped
                if ctx.existence == "bound":
                    return self._result(ctx, "resolved_unverified", canonical=canonical, matched=hit.matched,
                                        rule=rule, hops=hops, existence="absent",
                                        notes=[f"{canonical} is not in the {ctx.bound} universe; existence is decided "
                                               "by the bound column"])
                return self._result(ctx, "resolved", canonical=canonical, matched=hit.matched,
                                    rule=rule, hops=hops, existence=None,
                                    notes=[f"{canonical}: syntax checked; existence is decided upstream"
                                           if ctx.existence == "upstream" else f"{canonical}: syntax checked only"])
        return self._result(ctx, "not_found", matched=hit.matched, existence="absent",
                            suggestions=self._suggest(hit.kind, ctx.text), notes=hit.notes)

    def _rejected(self, ctx: "_Ctx", hits: list[_Hit]) -> ResolutionResult:
        reasons = tuple(h.reason for h in hits if h.reason)
        looks: list[str] = []
        for h in hits:
            looks.extend(x for x in h.looks_like if x not in looks)
        accepted_plugins = set()
        for k in ctx.accepts:
            try:
                accepted_plugins.add(self.spec(k)[1].plugin)
            except ResolverConfigError:
                pass
        if self.registry is not None:
            scored = []
            for p in self.registry.all("identifier"):
                if p.name in accepted_plugins or p.name in looks:
                    continue
                try:
                    score = float(p.looks_like(ctx.text))
                except Exception:  # noqa: BLE001 - a broken third-party hint never breaks resolution
                    continue
                if score >= HINT_THRESHOLD:
                    scored.append((-score, p.name))
            looks.extend(name for _, name in sorted(scored))
        notes = [f"looks like {', '.join(looks)}"] if looks else []
        return self._result(ctx, "rejected", matched=None, reasons=reasons, looks_like=tuple(looks),
                            notes=notes)

    def _outside(self, ctx: "_Ctx", hit: _Hit) -> ResolutionResult:
        valid = ctx.where.values() if ctx.where is not None else ()
        reason = f"{hit.canonical} is outside the values this argument allows"
        return self._result(ctx, "rejected", canonical=None, matched=hit.matched, rule=hit.rule,
                            subkind="outside_universe", valid_values=valid, reasons=(reason,),
                            notes=[f"{ctx.text!r} resolved to {hit.canonical}, which is outside the allowed values"])

    def _suggest(self, kind: str, text: str) -> tuple[Candidate, ...]:
        index = self.index(kind)
        if index is None:
            return ()
        try:
            plugin = self.plugin(kind)
        except ResolverConfigError:
            return ()
        key = plugin.label_key(text)
        out: list[Candidate] = []
        for e in index.lookup(key):
            if e.canonical and e.head != "retired" and all(c.id != e.canonical for c in out):
                out.append(Candidate(e.canonical, index.label(e.canonical) or e.label or None, "shared label key"))
        for e in index.suggest(key, max_n=self.settings.max_candidates):
            if all(c.id != e.canonical for c in out):
                out.append(Candidate(e.canonical, index.label(e.canonical) or e.label or None, "edit distance 1"))
        return tuple(out[: self.settings.max_candidates])

    def _result(self, ctx: "_Ctx", status: str, *, canonical: str | None = None, matched: str | None = None, rule: str | None = None, candidates: Sequence[Candidate] = (),
                family: Sequence[str] = (), hops: Sequence[str] = (), notes: Sequence[str] = (),
                existence: str | None = None, suggestions: Sequence[Candidate] = (), looks_like: Sequence[str] = (),
                reasons: Sequence[str] = (), subkind: str | None = None, replacement: Sequence[str] = (),
                valid_values: Sequence[str] = (), tried_extra: Sequence[str] = ()) -> ResolutionResult:
        target = ctx.bound
        source = table = fingerprint = label = None
        try:
            source, table = self._source_info(target)
        except (ResolverConfigError, LookupError):
            pass
        index = self.index(target)
        if index is not None:
            fingerprint = index.fingerprint
            if canonical is not None and status in OK_STATUSES:
                label = index.label(canonical)
        return ResolutionResult(
            status=cast(Any, status), canonical=canonical, rule=rule, tried=tuple(ctx.tried) + tuple(tried_extra),
            candidates=tuple(candidates), id_type=target, matched_id_type=matched, family=tuple(family),
            hops=tuple(hops), notes=tuple(_distinct(notes)), raw=ctx.value, label=label, existence=existence,
            suggestions=tuple(suggestions), looks_like=tuple(looks_like), reasons=tuple(reasons), subkind=subkind,
            replacement=tuple(replacement), valid_values=tuple(valid_values), accepts=ctx.accepts, source=source,
            table=table, index_fingerprint=fingerprint)

    # -- remote ---------------------------------------------------------------

    def _remote(self, ctx: "_Ctx", source: str, id_type: str, values: tuple[str, ...]) -> Any:
        key = (source, id_type, values)
        if ctx.cache is not None:
            if key not in ctx.cache:
                raise _NeedRemote(key)
            got = ctx.cache[key]
            if isinstance(got, BaseException):
                raise got
            return got
        if self.remote is None:
            raise RuntimeError(f"no remote resolver for {source}:{id_type} (index: remote)")
        got = self.remote(source, id_type, list(values))
        if inspect.isawaitable(got):
            _close(got)
            raise RuntimeError("the remote resolver is async; use aresolve")
        return got

    async def _remote_async(self, key: tuple[str, str, tuple[str, ...]]) -> Any:
        if self.remote is None:
            return RuntimeError(f"no remote resolver for {key[0]}:{key[1]} (index: remote)")
        try:
            got = self.remote(key[0], key[1], list(key[2]))
            if inspect.isawaitable(got):
                got = await got
            return got
        except Exception as exc:  # noqa: BLE001 - stored and reported as existence unknown
            return exc


@dataclass
class _Ctx:
    text: str
    value: Any
    existence: str
    where: "_Where | None"
    cache: dict[tuple[str, str, tuple[str, ...]], Any] | None
    tried: list[str]
    accepts: tuple[str, ...]
    bound: str


def _close(aw: Awaitable[Any]) -> None:
    close = getattr(aw, "close", None)
    if callable(close):
        close()


def _curie_variants(text: str) -> list[str]:
    """``DOID_9352`` -> ``DOID:9352`` and back (xref columns store CURIEs with a colon)."""
    t = text.strip()
    out = []
    for a, b in ((":", "_"), ("_", ":")):
        head, sep, tail = t.partition(a)
        if sep and head and tail and head.replace("-", "").isalnum():
            out.append(f"{head}{b}{tail}")
    return out


def _unique_candidates(cands: Iterable[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    for c in cands:
        if all(o.id != c.id for o in out):
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# errors (§12.1 payloads)
# ---------------------------------------------------------------------------

def error_for(res: ResolutionResult, argument: str, *, tool: str | None = None,
              enum_max: int = 64) -> GatewayError | None:
    """The typed error of an unsuccessful resolution (None when the call proceeds)."""
    if res.status == "not_found":
        sugg = [{"id": c.id, "label": c.label, "why": c.via} for c in res.suggestions]
        payload = not_found_payload(argument, res.raw, res.id_type, list(res.accepts), list(res.tried), sugg,
                                    res.source, res.table, subkind=res.subkind,
                                    replacement=list(res.replacement) if res.subkind == "obsolete" else None)
        what = "is retired without a single replacement" if res.subkind == "obsolete" else \
            f"was not found in {res.id_type}"
        return GatewayError(ErrorKind.not_found, f"{argument}={res.raw!r} {what}", tool=tool, payload=payload)
    if res.status == "ambiguous":
        cands = [{"id": c.id, "label": c.label, "via": c.via, **dict(c.extra)} for c in res.candidates]
        return GatewayError(ErrorKind.ambiguous, f"{argument}={res.raw!r} matches {len(cands)} entities; choose one",
                            tool=tool, payload=ambiguous_payload(argument, res.raw, cands))
    if res.status == "rejected":
        payload = invalid_argument_payload(argument, res.raw, list(res.valid_values), looks_like=res.looks_like,
                                           enum_max=enum_max)
        if res.reasons:
            payload["reasons"] = list(res.reasons)
        if res.subkind:
            payload["subkind"] = res.subkind
        hint = f" (looks like {', '.join(res.looks_like)})" if res.looks_like else ""
        return GatewayError(ErrorKind.invalid_argument, f"{argument}={res.raw!r} is not an accepted identifier{hint}",
                            tool=tool, payload=payload)
    return None


def list_error(summary: Mapping[str, Any], argument: str, *, tool: str | None = None) -> GatewayError | None:
    """The typed error of a list argument's resolution summary (None when the call proceeds)."""
    status = summary.get("status")
    if status == "insufficient":
        payload = insufficient_resolution_payload(argument, summary["requested"], summary["resolved"],
                                                  summary["unresolved"], summary["ambiguous"],
                                                  summary["outside_universe"],
                                                  float(summary.get("min_resolved_fraction") or 0.0))
        return GatewayError(ErrorKind.insufficient_resolution,
                            f"{argument}: {summary['resolved']} of {summary['requested']} values resolved", tool=tool,
                            payload=payload)
    if status == "failed":
        items = list(summary.get("items", ()))
        looks = list(dict.fromkeys(k for i in items for k in i.get("looks_like") or ()))   # what the values look like
        payload = invalid_argument_payload(argument, [i["value"] for i in items], None, looks_like=looks,
                                           items=items)
        return GatewayError(ErrorKind.invalid_argument,
                            f"{argument}: {len(summary.get('items', ()))} element(s) did not resolve", tool=tool,
                            payload=payload)
    return None
