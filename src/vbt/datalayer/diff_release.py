"""``vbt ds diff-release --from A --to B``: what changed between two releases of a source (§16, §22 step 5, F24).

Both sides are read with the same descriptor; only the source's ``root`` differs. ``--from`` and
``--to`` are data roots, or release labels resolved as siblings of the configured root (a root
``/data/ot/25.09`` and ``--to 25.12`` read ``/data/ot/25.12``). Per table:

* ``table_added`` / ``table_removed``: the table is readable on one side only;
* ``column_removed``: a column the descriptor roles (a **role column**) is missing on the new side,
  so tools that read it become ``not_ready`` (``schema_drift``) until the descriptor is updated;
  ``column_added``: a column on the new side the descriptor does not declare;
* ``type_changed``: the Arrow storage type of a leaf changed (``int64`` -> ``double``, ``string`` ->
  ``large_string``): literals compile in the storage type, so float keys and scope values change;
* ``range_changed``: the row-group minimum or maximum of a measure, flag or count column moved
  (the encoding signal: ``hasSafetyEvent`` -1/1 versus null/1, a 0-1 scale becoming 0-4);
* ``vocab_changed``: values added to or removed from the distinct snapshot of a category, scope or
  flag column (the vocabularies argument contracts snap to);
* ``axis_changed``: members added to or removed from a matrix axis (genes or models of a DepMap
  matrix, cells of an AnnData cohort);
* ``rows_changed``: the row count moved (informational).

Runs inside the data child's interpreter (``vbt ds diff-release`` calls :func:`diff_release_child`
through the data child's command line); :func:`diff_release` is the same with a context in hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["CHANGE_KINDS", "ROLE_RANGES", "VOCAB_ROLES", "side_root", "diff_release", "diff_release_child",
           "format_report"]

CHANGE_KINDS = ("table_added", "table_removed", "column_removed", "column_added", "type_changed", "range_changed",
                "vocab_changed", "axis_changed", "rows_changed", "unreadable")
#: Roles whose row-group ranges are compared (encodings and scales live in them).
ROLE_RANGES = frozenset({"measure", "flag", "count", "time"})
#: Roles whose distinct snapshot is compared.
VOCAB_ROLES = frozenset({"category", "scope", "flag"})
#: Changes that make some tool not_ready or change its answers (rows_changed and column_added do not).
BREAKING = frozenset({"table_removed", "column_removed", "type_changed", "range_changed", "vocab_changed",
                      "axis_changed", "unreadable"})
_SAMPLE = 20


def side_root(configured: str | None, release: str | None, value: str) -> Path:
    """The data root of one side: ``value`` when it is a directory, else the sibling of the configured
    root named ``value`` (when the configured root ends with the descriptor's release)."""
    p = Path(str(value)).expanduser()
    if p.is_dir():
        return p
    if configured:
        root = Path(configured)
        if release and root.name == str(release):
            return root.parent / str(value)
        if root.name == str(value):
            return root
    raise ValueError(f"{value!r} is not a directory, and the configured root {configured!r} does not end with the "
                     f"release {release!r} to resolve it as a sibling")


def _side_context(ctx: Any, source: str, root: Path) -> Any:
    from .catalog import Catalog
    from .service import ServiceContext

    cat = ctx.catalog
    sources = dict(cat.sources)
    sources[source] = cat.source(source).model_copy(update={"root": str(root)})
    side = Catalog(sources, cat.overlays, cat.generic, aliases=cat.aliases, registry=ctx.registry)
    return ServiceContext(ctx.settings, catalog=side, registry=ctx.registry)


def _readable_tables(ctx: Any, source: str, only: Iterable[str] = ()) -> list[str]:
    """Physical tables of ``source`` with files (item tables share their parent; live and upstream-only
    layouts have no files to compare)."""
    desc = ctx.catalog.source(source)
    wanted = {t.split(".", 1)[-1] for t in only}
    out = []
    for name, spec in sorted(desc.tables.items()):
        if spec.items_of is not None or (wanted and name not in wanted):
            continue
        try:
            t = ctx.table(f"{source}.{name}")
            plugin = ctx.registry.find("layout", t.layout) if t.layout else None
        except Exception:  # noqa: BLE001 - an unloadable table is reported by readiness, not here
            continue
        caps = set(getattr(plugin, "capabilities", ()) or ())
        if t.format == "none" or caps & {"live", "upstream_only"}:
            continue
        out.append(f"{source}.{name}")
    return out


def _roles(ctx: Any, ref: str) -> dict[str, str]:
    t = ctx.table(ref)
    return {str(name): str(getattr(col, "role", "")) for name, col in (t.spec.columns or {}).items()}


def _stats(ctx: Any, refs: Sequence[str]) -> tuple[dict[str, Any], dict[str, str]]:
    from .ipc import TABLE_ERRORS
    from .service.verbs import load_verbs

    body = load_verbs()["_stats"](ctx, {"tables": list(refs)}) if refs else {}
    errors = dict(body.get(TABLE_ERRORS) or body.get("errors") or {})
    return dict(body.get("tables") or {}), {str(k): str(v) for k, v in errors.items()}


def _vocab(ctx: Any, ref: str, column: str, max_values: int) -> tuple[list[str] | None, bool]:
    from .service.verbs import load_verbs

    try:
        body = load_verbs()["_vocab"](ctx, {"table": ref, "column": column, "max_values": max_values})
    except Exception:  # noqa: BLE001 - not every column has a snapshot (nested, unreadable)
        return None, False
    return [str(v) for v in body.get("rendered") or []], bool(body.get("complete", True))


def _axis(ctx: Any, ref: str, axis: str) -> list[str] | None:
    from .rowkey import canonical
    from .service.verbs.public import LongView

    try:
        view = LongView(ctx, ref)
        keys = view.row_key if axis == "row" else view.col_key
        return sorted(canonical([r.get(k) for k in keys]) for r in view.axis(axis))
    except Exception:  # noqa: BLE001
        return None


def _top(name: str) -> str:
    return name.split(".", 1)[0].split("[", 1)[0]


def _change(kind: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "detail": detail, **{k: v for k, v in extra.items() if v is not None}}


def _set_change(kind: str, what: str, before: Sequence[str], after: Sequence[str], **extra: Any) -> dict | None:
    a, b = set(before), set(after)
    added, removed = sorted(b - a), sorted(a - b)
    if not added and not removed:
        return None
    detail = f"{what}: +{len(added)} -{len(removed)}"
    if added:
        detail += f"; added {', '.join(added[:5])}" + (" ..." if len(added) > 5 else "")
    if removed:
        detail += f"; removed {', '.join(removed[:5])}" + (" ..." if len(removed) > 5 else "")
    return _change(kind, detail, added=added[:_SAMPLE], removed=removed[:_SAMPLE], n_added=len(added),
                   n_removed=len(removed), **extra)


def _table_diff(ref: str, ctx_a: Any, ctx_b: Any, sa: Mapping[str, Any], sb: Mapping[str, Any], *,
                vocab: bool, axes: bool, max_values: int) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    roles = _roles(ctx_a, ref)
    ca, cb = dict(sa.get("columns") or {}), dict(sb.get("columns") or {})
    tops_a, tops_b = {_top(c) for c in ca}, {_top(c) for c in cb}
    table = ctx_a.table(ref)
    partition_cols = set(table.physical_spec.partitions)
    for col in sorted(set(roles) - tops_b - partition_cols):
        if col in tops_a:
            changes.append(_change("column_removed", f"role column {col} ({roles[col]}) is missing in the new release",
                                   column=col, role=roles[col]))
    for col in sorted(tops_b - tops_a - set(roles) - partition_cols):
        changes.append(_change("column_added", f"{col} is new and not declared by the descriptor", column=col))
    for path in sorted(set(ca) & set(cb)):
        ta, tb = ca[path].get("storage_type"), cb[path].get("storage_type")
        if ta and tb and ta != tb:
            changes.append(_change("type_changed", f"{path}: {ta} -> {tb}", column=path, before=ta, after=tb))
        if roles.get(_top(path)) in ROLE_RANGES:
            ra = (ca[path].get("min"), ca[path].get("max"))
            rb = (cb[path].get("min"), cb[path].get("max"))
            if ra != rb and None not in ra + rb:
                changes.append(_change("range_changed", f"{path}: row-group range {list(ra)} -> {list(rb)}",
                                       column=path, before=list(ra), after=list(rb)))
    if sa.get("rows") != sb.get("rows") and sa.get("rows") is not None and sb.get("rows") is not None:
        changes.append(_change("rows_changed", f"rows {sa.get('rows')} -> {sb.get('rows')}", before=sa.get("rows"),
                               after=sb.get("rows")))
    if vocab:
        for col in sorted([c for c, r in roles.items() if r in VOCAB_ROLES] + sorted(partition_cols)):
            if col not in tops_a & tops_b and col not in partition_cols:
                continue
            va, complete_a = _vocab(ctx_a, ref, col, max_values)
            vb, complete_b = _vocab(ctx_b, ref, col, max_values)
            if va is None or vb is None:
                continue
            ch = _set_change("vocab_changed", f"{col} vocabulary", va, vb, column=col,
                             complete=complete_a and complete_b)
            if ch is not None:
                changes.append(ch)
    if axes and table.kind == "matrix":
        for axis in ("row", "col"):
            aa, ab = _axis(ctx_a, ref, axis), _axis(ctx_b, ref, axis)
            if aa is None or ab is None:
                continue
            ch = _set_change("axis_changed", f"{axis} axis members", aa, ab, axis=axis)
            if ch is not None:
                changes.append(ch)
    return changes


def diff_release(ctx: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``{from, to, sources?, tables?, vocab?, axes?, max_values?}`` -> the report (see the module docstring)."""
    frm, to = str(payload["from"]), str(payload["to"])
    tables = list(payload.get("tables") or [])
    sources = list(payload.get("sources") or sorted({t.split(".", 1)[0] for t in tables}))
    if not sources:
        sources = [s for s, d in sorted(ctx.catalog.sources.items())
                   if d.root and str(d.release.expect) == frm and not Path(frm).is_dir()]
    if not sources:
        raise ValueError("name the source(s) to compare (--source), or give --from as the configured release label")
    vocab, axes = bool(payload.get("vocab", True)), bool(payload.get("axes", True))
    max_values = int(payload.get("max_values") or 500)
    report: dict[str, Any] = {"from": frm, "to": to, "sources": {}, "summary": {}}
    counts: dict[str, int] = {}
    for source in sources:
        desc = ctx.catalog.source(source)
        root_a = side_root(desc.root, desc.release.expect, frm)
        root_b = side_root(desc.root, desc.release.expect, to)
        ctx_a, ctx_b = _side_context(ctx, source, root_a), _side_context(ctx, source, root_b)
        refs = _readable_tables(ctx_a, source, [t for t in tables if t.startswith(source + ".")])
        stats_a, err_a = _stats(ctx_a, refs)
        stats_b, err_b = _stats(ctx_b, refs)
        out: dict[str, Any] = {}
        for ref in refs:
            sa, sb = stats_a.get(ref), stats_b.get(ref)
            if sa is None and sb is None:
                continue
            if sa is None or sb is None:
                kind = "table_added" if sa is None else "table_removed"
                why = err_a.get(ref) if sa is None else err_b.get(ref)
                changes = [_change(kind, f"{ref} is readable only in the {'new' if sa is None else 'old'} release"
                                   + (f" ({str(why)[:200]})" if why else ""))]
                fps = {"from": (sa or {}).get("fingerprint"), "to": (sb or {}).get("fingerprint")}
            else:
                fps = {"from": sa.get("fingerprint"), "to": sb.get("fingerprint")}
                try:
                    changes = _table_diff(ref, ctx_a, ctx_b, sa, sb, vocab=vocab, axes=axes, max_values=max_values)
                except Exception as exc:  # noqa: BLE001 - one table's failure is reported, the rest continue
                    changes = [_change("unreadable", f"{type(exc).__name__}: {exc}"[:500])]
            for c in changes:
                counts[c["kind"]] = counts.get(c["kind"], 0) + 1
            breaking = any(c["kind"] in BREAKING for c in changes)
            out[ref] = {"status": "changed" if changes else "same", "breaking": breaking, "fingerprint": fps,
                        "changes": changes}
        report["sources"][source] = {"from_root": str(root_a), "to_root": str(root_b), "tables": out}
    report["summary"] = {"changes": dict(sorted(counts.items())),
                         "breaking_tables": sorted(ref for s in report["sources"].values()
                                                   for ref, t in s["tables"].items() if t["breaking"])}
    return report


def diff_release_child(ctx: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """:func:`diff_release` as a data-child command (``module:function(ctx, payload)``); errors as ``{error}``."""
    try:
        return diff_release(ctx, payload)
    except (ValueError, LookupError) as exc:
        return {"error": str(exc)}


def format_report(report: Mapping[str, Any]) -> list[str]:
    """Text lines of a diff-release report: per source and table, the changes (breaking ones marked)."""
    lines = [f"diff-release {report.get('from')} -> {report.get('to')}"]
    for source, s in (report.get("sources") or {}).items():
        lines.append(f"{source}: {s.get('from_root')} -> {s.get('to_root')}")
        same = [ref for ref, t in s["tables"].items() if t["status"] == "same"]
        for ref, t in s["tables"].items():
            if t["status"] == "same":
                continue
            lines.append(f"  {ref}{' (breaking)' if t['breaking'] else ''}")
            for c in t["changes"]:
                lines.append(f"    {c['kind']:<15} {c['detail']}")
        if same:
            lines.append(f"  unchanged: {len(same)} table(s)")
    summary = report.get("summary") or {}
    if summary.get("breaking_tables"):
        lines.append("Tools reading these tables become not_ready (schema_drift) or answer differently until the "
                     "descriptor is reviewed: " + ", ".join(summary["breaking_tables"]))
    lines.append("changes: " + (json.dumps(summary.get("changes") or {}, sort_keys=True)))
    return lines
