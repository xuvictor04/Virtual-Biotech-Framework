"""``neighbors`` over several hops and the interaction network tools (§6.8 S8, §10.6; phase 3, F17).

Upstream ``get_interaction_network`` expanded a breadth-first frontier held in a Python ``set``: the
iteration order followed the hash seed, so the edges kept under ``max_network_size`` differed between
runs (2, 2, 4, 4, 3, 4 edges over six runs, VERIFIED), and one ``min_confidence`` was applied to
STRING, IntAct, Reactome and SIGNOR scores that are not comparable. :func:`expand_network` is the
specified traversal over an ``edges`` table (``TableSpec.edge``):

* hop 0 is the seeds; each hop reads the edges touching the frontier (one bounded scan per hop) and
  the **frontier is expanded in canonical key order** (sorted node IDs), so the result never depends
  on the hash seed; ``score_order: true`` orders each node's edges by score within one source only;
* an edge is identified by the table's **full key** (``sourceDatabase, intA, intB, targetA, targetB``):
  sources are never pooled, and the same gene pair reported by two sources is two edges;
* undirected edges join their endpoints both ways; with ``orientation: both`` storage (every edge
  stored A-B and B-A) the two stored rows are one edge (an unordered canonical pair of the endpoint
  sides); ``directed`` is reported per edge (``directed_when``, or ``direction_from_roles``: the two stored
  rows of a SIGNOR relation, roles swapped, are one directed edge with ``source_node`` and ``target_node``);
* a row whose endpoint is null (``missing: non_entity``: an interactor not mapped to a gene) is
  excluded and counted (``excluded_non_entity``), never a node named ``None``;
* ``max_nodes`` cuts the last hop deterministically (candidates in key order): the result is
  ``partial`` with per-hop frontier counts (``{hop, expanded, added, not_added}``); a read over the scan
  budget is ``too_large``, never a smaller network presented as complete;
* provenance carries the hops, the node-set and edge-set hashes and the frontier counts.

The public ``neighbors`` verb (``{table, node | nodes, hops?, where?, max_nodes?, score_order?, limit?}``) uses it for
more than one hop or more than one node (one node, one hop keeps the phase-2 answer). Derived bindings
reach it through ``_serve`` with ``split: {network: {mode: network | common, ...}}``:
``get_interaction_network`` (``mode: network``: nodes, edges and statistics; ``max_hops``,
``max_network_size``) and ``find_common_interactors`` (``mode: common``: per-source partners of each
interactor that touches at least ``min_targets`` inputs; scores are reported per source, never averaged
across sources). ``min_confidence`` other than 0 is ``unsupported_filter``: one threshold cannot apply
to scores that are comparable only within a source.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from ...errors import ErrorKind, GatewayError, json_value, unsupported_filter_payload
from ...predicate import And, Eq, In, Or, Predicate, from_json
from ...result import inject_header
from ...rowkey import canonical
from .. import ServiceContext
from . import public as _public
from .hierarchy import serve_extension
from .public import LongView, _invalid, _limit, compile_where, guarded, header, key_label, long_view, table_access

__all__ = ["Network", "expand_network", "node_hash", "neighbors", "serve_network", "common_interactors"]

#: The deepest traversal a call may ask for.
MAX_HOPS = 4


def node_hash(nodes: Sequence[Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(sorted(str(n) for n in nodes)).encode()).hexdigest()[:16]


class Network:
    """The result of :func:`expand_network`."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.frontier: list[dict[str, Any]] = []
        self.excluded_non_entity = 0
        self.truncated = False

    def record(self, hops: int) -> dict[str, Any]:
        return {"hops": hops, "nodes": len(self.nodes), "edges": len(self.edges), "node_set": node_hash(list(self.nodes)),
                "edge_set": node_hash(list(self.edges)), "frontier": self.frontier, "truncated": self.truncated,
                "excluded_non_entity": self.excluded_non_entity}


def _edge_spec(view: LongView) -> Any:
    edge = view.table.spec.edge
    if edge is None:
        raise _invalid("table", view.ref, f"{view.ref} is not an edges table")
    return edge


def _role_direction(row: Mapping[str, Any], edge: Any) -> tuple[str, str] | None:
    """``("a", "b")`` or ``("b", "a")``: the source and the target side when ``direction_from_roles`` gives the
    row a direction, else None."""
    spec = getattr(edge, "direction_from_roles", None)
    if spec is None or any(row.get(col) not in values for col, values in spec.when.items()):
        return None
    roles = (row.get(spec.columns["a"]), row.get(spec.columns["b"]))
    if roles == (spec.source, spec.target):
        return ("a", "b")
    if roles == (spec.target, spec.source):
        return ("b", "a")
    return None


def _directed(row: Mapping[str, Any], edge: Any) -> bool:
    if edge.directed or _role_direction(row, edge) is not None:
        return True
    return any(row.get(col) in values for col, values in (edge.directed_when or {}).items())


def _identity(row: Mapping[str, Any], key: Sequence[str], edge: Any) -> str:
    """The edge's identity: its full key; for ``orientation: both`` the two endpoint sides are unordered, or
    ordered source first when the roles give the direction (the two stored rows of one directed edge)."""
    roles = _role_direction(row, edge)
    if edge.orientation != "both" or (roles is None and _directed(row, edge)):
        return canonical([row.get(k) for k in key])
    side_a = [edge.a, *[c for c in edge.sides.get("a", []) if c in key]]
    side_b = [edge.b, *[c for c in edge.sides.get("b", []) if c in key]]
    rest = [k for k in key if k not in side_a and k not in side_b]
    va, vb = canonical([row.get(c) for c in side_a]), canonical([row.get(c) for c in side_b])
    sides = sorted([va, vb]) if roles is None else [va, vb] if roles == ("a", "b") else [vb, va]
    return canonical([*[row.get(k) for k in rest], *sides])


def _direction_fields(row: Mapping[str, Any], edge: Any) -> dict[str, Any]:
    """``source_node`` and ``target_node`` of an edge whose roles give its direction ({} otherwise)."""
    roles = _role_direction(row, edge)
    if roles is None:
        return {}
    ends = {"a": row.get(edge.a), "b": row.get(edge.b)}
    return {"source_node": ends[roles[0]], "target_node": ends[roles[1]]}


def _score_column(view: LongView) -> tuple[str | None, list[str]]:
    for r in view.table.spec.rank:
        spec = view.column(r.column)
        if getattr(spec, "role", None) == "measure":
            return r.column, list(r.within)
    return None, []


def expand_network(view: LongView, seeds: Sequence[Any], hops: int, *, pred: Predicate | None = None,
                   max_nodes: int | None = None, budget: int | None = None, score_order: bool = False) -> Network:
    """The deterministic traversal (see the module docstring)."""
    edge = _edge_spec(view)
    key = list(view.key)
    score, within = _score_column(view)
    net = Network()
    for s in sorted({str(s) for s in seeds}):
        net.nodes[s] = {"node": s, "hop": 0, "is_seed": True}
    frontier = sorted(net.nodes)
    for hop in range(1, hops + 1):
        if not frontier:
            break
        where: Predicate = Or((In(edge.a, tuple(frontier)), In(edge.b, tuple(frontier))))
        if pred is not None:
            where = And((where, pred))
        rows, _total, _eu = view.rows(where, order=[], limit=None, budget=budget)
        by_node: dict[str, list[dict[str, Any]]] = {}
        front = set(frontier)
        for r in rows:
            a, b = r.get(edge.a), r.get(edge.b)
            if a in (None, "") or b in (None, ""):
                net.excluded_non_entity += 1
                continue
            for end in (str(a), str(b)):
                if end in front:
                    by_node.setdefault(end, []).append(r)
        candidates: list[str] = []
        seen_new: set[str] = set()
        kept: list[tuple[str, dict[str, Any]]] = []
        for node in frontier:                          # canonical key order, never set iteration order
            edges = by_node.get(node, [])
            edges.sort(key=lambda r: (_score_key(r, score, within) if score_order else (), canonical([r.get(k)
                                                                                                         for k in key])))
            for r in edges:
                a, b = str(r[edge.a]), str(r[edge.b])
                partner = b if a == node else a
                kept.append((partner, r))
                if partner not in net.nodes and partner not in seen_new:
                    seen_new.add(partner)
                    candidates.append(partner)
        candidates.sort()
        room = None if max_nodes is None else max(0, int(max_nodes) - len(net.nodes))
        added = candidates if room is None else candidates[:room]
        for n in added:
            net.nodes[n] = {"node": n, "hop": hop, "is_seed": False}
        not_added = len(candidates) - len(added)
        if not_added:
            net.truncated = True
        for partner, r in kept:
            if partner in net.nodes:
                ident = _identity(r, key, edge)
                if ident not in net.edges:
                    net.edges[ident] = {**{k: r.get(k) for k in key},
                                        **({score: r.get(score)} if score else {}),
                                        **{c: r.get(c) for c in view.columns if c not in key and c != score and
                                           getattr(view.column(c), "role", None) in ("count", "measure")},
                                        "directed": _directed(r, edge), **_direction_fields(r, edge)}
        net.frontier.append({"hop": hop, "expanded": len(frontier), "added": len(added), "not_added": not_added})
        if net.truncated:
            break
        frontier = sorted(added)
    return net


def _score_key(row: Mapping[str, Any], score: str | None, within: Sequence[str]) -> tuple[Any, ...]:
    v = row.get(score) if score else None
    return (*[str(row.get(w)) for w in within], 1 if v is None else 0, -(v or 0.0))


def _degrees(net: Network, edge: Any) -> dict[str, int]:
    deg: dict[str, int] = {}
    for e in net.edges.values():
        for end in (e.get(edge.a), e.get(edge.b)):
            deg[str(end)] = deg.get(str(end), 0) + 1
    return deg


def _ordered_edges(net: Network, key: Sequence[str]) -> list[dict[str, Any]]:
    return [net.edges[k] for k in sorted(net.edges)]


# ---------------------------------------------------------------------------- the public verb


def neighbors(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Edges within ``hops`` of ``node`` (or ``nodes``), with the nodes reached (see the module docstring)."""
    hops = payload.get("hops", 1)
    nodes_arg = payload.get("nodes")
    if (hops == 1 or hops is None) and nodes_arg is None:
        return _PHASE2_NEIGHBORS(ctx, payload)
    if isinstance(hops, bool) or not isinstance(hops, int) or not 1 <= hops <= MAX_HOPS:
        raise _invalid("hops", hops, f"hops is an integer from 1 to {MAX_HOPS}", list(range(1, MAX_HOPS + 1)))
    table = table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    view = long_view(ctx, str(table.ref))
    edge = _edge_spec(view)
    values = nodes_arg if nodes_arg is not None else [payload.get("node")]
    if not isinstance(values, list) or not values:
        raise _invalid("nodes", values, "nodes is a non-empty list of node identifiers")
    notes: list[str] = []
    resolved: dict[str, str] = {}
    _p, keys = compile_where(view, {edge.a: values}, argument="nodes", notes=notes, resolved=resolved)
    seeds = list(keys.get(edge.a) or values)
    extra, _k = compile_where(view, payload.get("where"), notes=notes, resolved=resolved)
    max_nodes = payload.get("max_nodes")
    if max_nodes is not None and (isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1):
        raise _invalid("max_nodes", max_nodes, "max_nodes is a positive integer")
    net = expand_network(view, seeds, hops, pred=extra, max_nodes=max_nodes, budget=payload.get("budget_bytes"),
                         score_order=bool(payload.get("score_order")))
    edges = _ordered_edges(net, view.key)
    limit = _limit(payload, 200, _public._max_rows(ctx))
    shown = edges[:limit]
    deg = _degrees(net, edge)
    nodes = [{**n, "degree": deg.get(k, 0)} for k, n in sorted(net.nodes.items(), key=lambda kv: (kv[1]["hop"], kv[0]))]
    if net.truncated:
        notes.append(f"the network was cut at max_nodes={max_nodes} (candidates in key order); frontier counts are in "
                     "_vbt.network.frontier")
    excluded = {edge.b: net.excluded_non_entity} if net.excluded_non_entity else {}
    hdr = header(view, rows=shown, total=len(edges), truncated=len(shown) < len(edges) or net.truncated,
                 order="full edge key (canonical)", resolved=resolved, excluded_unknown=excluded, notes=notes,
                 extra={"network": net.record(hops)})
    if net.truncated and shown:
        hdr.status = "partial"
    return inject_header({"rows": json_value(shown), "nodes": json_value(nodes)}, hdr)


_current = _public.VERBS["neighbors"]
_PHASE2_NEIGHBORS = getattr(_current, "_phase2", _current)          # the one-hop verb of public.py
if not getattr(_current, "_phase3", False):
    _wrapped = guarded("neighbors", neighbors)
    _wrapped._phase3 = True  # type: ignore[attr-defined]
    _wrapped._phase2 = _current  # type: ignore[attr-defined]
    _public.VERBS["neighbors"] = _wrapped


# ---------------------------------------------------------------------------- derived bindings (_serve)


def _endpoint_values(pred: Predicate | None, columns: Sequence[str]) -> list[Any]:
    """The node values a predicate fixes on any of ``columns`` (``Eq``/``In``, through ``And``/``Or``)."""
    out: list[Any] = []

    def rec(p: Any) -> None:
        if isinstance(p, Eq) and p.column in columns:
            out.append(p.value)
        elif isinstance(p, In) and p.column in columns:
            out.extend(p.values)
        elif isinstance(p, (And, Or)):
            for q in p.preds:
                rec(q)

    rec(pred)
    return list(dict.fromkeys(out))


def _strip_endpoints(pred: Predicate | None, columns: Sequence[str]) -> Predicate | None:
    """``pred`` without its endpoint conditions (the seeds are applied by the traversal)."""
    if pred is None:
        return None
    if isinstance(pred, (Eq, In)) and pred.column in columns:
        return None
    if isinstance(pred, Or) and all(isinstance(q, (Eq, In)) and q.column in columns for q in pred.preds):
        return None
    if isinstance(pred, And):
        kept = [q for q in (_strip_endpoints(p, columns) for p in pred.preds) if q is not None]
        return None if not kept else kept[0] if len(kept) == 1 else And(tuple(kept))
    return pred


def _param(params: Mapping[str, Any], opts: Mapping[str, Any], name: str, default: Any) -> Any:
    arg = (opts.get("args") or {}).get(name, name)
    v = params.get(arg)
    return default if v is None else v


def _refuse_confidence(params: Mapping[str, Any], opts: Mapping[str, Any], view: LongView) -> None:
    v = _param(params, opts, "min_confidence", 0)
    if v not in (0, 0.0, None):
        score, within = _score_column(view)
        raise GatewayError(ErrorKind.unsupported_filter, f"min_confidence={v} would apply one threshold to "
                           f"{score} values comparable only within {', '.join(within)}; filter one source with "
                           "interaction.search_interactions or interaction.get_interactions instead",
                           argument=str((opts.get("args") or {}).get("min_confidence", "min_confidence")), value=v,
                           payload=unsupported_filter_payload("min_confidence", score, "group"))


@serve_extension("network", lambda req: bool((req.get("split") or {}).get("network")))
def serve_network(ctx: ServiceContext, req: Mapping[str, Any]) -> dict[str, Any]:
    """``split: {network: {mode, args?}}``: ``get_interaction_network`` or ``find_common_interactors``."""
    from ...ipc import ServeResponse

    opts = dict((req.get("split") or {}).get("network") or {})
    params = dict(req.get("params") or {})
    view = long_view(ctx, str(req.get("table")))
    edge = _edge_spec(view)
    pred = from_json(req["predicate"]) if req.get("predicate") else None
    seeds = [str(s) for s in _endpoint_values(pred, [edge.a, edge.b])]
    rest = _strip_endpoints(pred, [edge.a, edge.b])
    _refuse_confidence(params, opts, view)
    if opts.get("mode") == "common":
        rows, stats = common_interactors(view, seeds, int(_param(params, opts, "min_targets", 2)), pred=rest,
                                         budget=req.get("budget_bytes"))
        total = len(rows)
        limit = req.get("limit")
        shown = rows[: int(limit)] if limit is not None else rows
        sections = {name: json_value(stats) for name in (req.get("sections") or {})}
        sections["_network"] = json_value(stats)
        return ServeResponse(rows=json_value(shown), total=total, truncated=len(shown) < total,
                             key_columns=["interactor_id"], sections=sections, served_by="derived",
                             row_keys=[[r["interactor_id"]] for r in shown]).model_dump(mode="json")
    hops = int(_param(params, opts, "max_hops", 2))
    if not 1 <= hops <= MAX_HOPS:
        raise GatewayError(ErrorKind.invalid_argument, f"max_hops is an integer from 1 to {MAX_HOPS}",
                           argument="max_hops", value=hops)
    max_nodes = _param(params, opts, "max_network_size", 500)
    net = expand_network(view, seeds, hops, pred=rest, max_nodes=int(max_nodes), budget=req.get("budget_bytes"))
    deg = _degrees(net, edge)
    nodes = [{"target_id": k, "hop_distance": n["hop"], "is_seed": n["is_seed"], "interaction_count": deg.get(k, 0)}
             for k, n in sorted(net.nodes.items(), key=lambda kv: (kv[1]["hop"], kv[0]))]
    edges = _ordered_edges(net, view.key)
    degrees = [n["interaction_count"] for n in nodes]
    stats = {"seeds_found": sum(1 for n in nodes if n["is_seed"] and n["interaction_count"] > 0),
             "avg_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0,
             "max_degree": max(degrees, default=0), "network_truncated": net.truncated,
             "network": net.record(hops)}
    sections = {name: json_value(stats) for name in (req.get("sections") or {})}
    sections["_network"] = json_value(stats)
    return ServeResponse(rows={"$.nodes": json_value(nodes), "$.edges": json_value(edges)},
                         total=len(nodes) + len(edges), truncated=net.truncated, key_columns=[key_label(k) for k in view.key],
                         sections=sections, served_by="derived").model_dump(mode="json")


def common_interactors(view: LongView, inputs: Sequence[str], min_targets: int, *, pred: Predicate | None = None,
                       budget: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Interactors (not among ``inputs``) touching at least ``min_targets`` inputs, with their partners per
    source; ordered by connection count, then ID."""
    edge = _edge_spec(view)
    score, within = _score_column(view)
    source_col = within[0] if within else None
    net = expand_network(view, inputs, 1, pred=pred, budget=budget)
    by: dict[str, dict[str, Any]] = {}
    inset = set(inputs)
    for e in _ordered_edges(net, view.key):
        a, b = str(e.get(edge.a)), str(e.get(edge.b))
        if a in inset and b in inset:
            continue
        target, partner = (a, b) if a in inset else (b, a)
        rec = by.setdefault(partner, {"interactor_id": partner, "connects_to": set(), "sources": {},
                                      "total_interactions": 0})
        rec["connects_to"].add(target)
        rec["total_interactions"] += 1
        src = str(e.get(source_col)) if source_col else "all"
        s = rec["sources"].setdefault(src, {"connects_to": set(), "max_" + (score or "score"): None})
        s["connects_to"].add(target)
        if score and e.get(score) is not None:
            k = "max_" + score
            s[k] = e[score] if s[k] is None else max(s[k], e[score])
    rows = []
    for rec in by.values():
        n = len(rec["connects_to"])
        if n < min_targets:
            continue
        rows.append({"interactor_id": rec["interactor_id"], "connects_to": sorted(rec["connects_to"]),
                     "connection_count": n, "total_interactions": rec["total_interactions"],
                     "sources": {s: {**v, "connects_to": sorted(v["connects_to"])}
                                 for s, v in sorted(rec["sources"].items())}})
    rows.sort(key=lambda r: (-r["connection_count"], r["interactor_id"]))
    stats = {"input_targets": sorted(inset), "min_targets": min_targets, "candidates": len(by),
             "network": net.record(1), "scores": f"{score} reported per {source_col}, never averaged across sources"
             if score else None}
    return rows, stats


VERBS: dict[str, Any] = {}
