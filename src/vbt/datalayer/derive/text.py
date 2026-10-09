"""Agent-facing tool descriptions derived from roles (§10.2). No pyarrow.

Description = upstream's first sentence (unless an overlay ``text.drop_promises`` entry appears
in it) plus a generated block, capped at ``data.derive.description_max_chars``. Upstream
``Returns:`` and ``Example:`` sections are dropped because several promise fields that do not
exist. The block states the source and release, the table and what one record is (the item grain
for item tables), the key, what each identifier argument accepts, the order and whether the
harness verifies it, what ``_vbt.total`` and ``_vbt.grains`` count, level columns, cutoffs,
propagation, multiple-testing families, censoring, lossy projections, the evidence nature, unknown
exclusions, default filters that are **not** applied, and what an empty result means.

Blocked tools are listed as ``UNAVAILABLE: <reason>; use <alternative>``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

__all__ = ["describe_tool", "unavailable_text", "first_sentence"]

_SECTION = re.compile(r"\n\s*(Returns?|Examples?|Args?|Arguments|Parameters|Raises|Note)s?\s*:", re.IGNORECASE)
_SENTENCE = re.compile(r"(.+?[.!?])(\s|$)", re.DOTALL)


def first_sentence(description: str | None) -> str:
    """The first sentence of the first paragraph (before any ``Args:``/``Returns:`` section), ending in a
    full stop."""
    text = _SECTION.split(str(description or ""), maxsplit=1)[0].strip()
    text = re.split(r"\n\s*\n", text, maxsplit=1)[0]
    text = " ".join(text.split())
    m = _SENTENCE.match(text)
    out = (m.group(1) if m else text).strip()
    return out + "." if out and not out.endswith((".", "!", "?")) else out


def unavailable_text(reason: str, alternatives: list[str] | None = None, description: str | None = None) -> str:
    alt = f"; use {', '.join(alternatives)}" if alternatives else ""
    head = f"UNAVAILABLE: {reason}{alt}."
    first = first_sentence(description)
    return f"{head} {first}".strip() if first else head


def _table_name(ref: str | None) -> str:
    return (ref or "").split(".", 1)[-1]


def _source_label(t: Any) -> str:
    desc = t.descriptor
    rel = desc.release.expect
    return f"{desc.title} {rel}" if rel else desc.title


def _cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    stop = max(cut.rfind(". "), cut.rfind("\n"))
    if stop > max_chars // 2:
        cut = cut[: stop + 1]
    return cut.rstrip() + "…"


def _col(t: Any, name: str) -> Any:
    return t.columns.get(str(name).split(".")[0]) if t is not None else None


def _order_text(b: Any) -> str | None:
    order = list(b.result.order)
    if b.result.order_from_arg:
        when = "; ".join(f"without it, with {arg}: by {spec[0].column} {spec[0].direction}"
                         for arg, spec in b.result.order_when.items() if spec)
        return f"ordered by the {b.result.order_from_arg} argument" + (f" ({when})" if when else "")
    if not order:
        return None
    first = order[0]
    within = f" within {', '.join(first.within)}" if first.within else ""
    how = {"witness": "verified by the harness", "upstream_full_sort": "upstream sorts every match",
           "source_server_side": "ranked by the source, not verified"}[b.result.order_source]
    then = "".join(f", then {r.column} {r.direction}" for r in order[1:])
    return f"ranked by {first.column} {first.direction}{within}{then} ({how})"


def _propagation(name: str, prop: str, fraction: float | None) -> str:
    if prop == "mixed" and fraction is not None:
        return f"{name}: mixed: {fraction:.0%} of items also list an ancestor."
    return {"direct": f"{name}: direct annotations only.",
            "propagated": f"{name}: annotations propagated to ancestors.",
            "mixed": f"{name}: mixed direct and propagated annotations."}[prop]


def describe_tool(contract: Any, description: str | None, *, catalog: Any = None, max_chars: int = 1200,
                  unready: str | None = None, measured: Mapping[str, float] | None = None) -> str:
    """The rewritten description (generic tools: unchanged). ``unready`` is the reason every call
    would be unready (listed with an ``UNAVAILABLE:`` prefix). ``measured`` maps
    ``"source.table.column"`` of a ``mixed`` member column to the fraction of items that also list an
    ancestor (measured at readiness)."""
    b = getattr(contract, "binding", None)
    if b is None or getattr(contract, "generic", False):
        return description or ""
    if b.serve == "block" and b.block is not None:
        return _cap(unavailable_text(b.block.reason, list(b.block.alternatives), description), max_chars)
    from .tools import NATIVE_SERVER, NATIVE_VERBS, native_tool
    if catalog is not None and getattr(contract, "server", None) == NATIVE_SERVER and contract.tool in NATIVE_VERBS:
        tool = native_tool(catalog, contract.tool, column_maps=False)      # the description only
        if tool is not None:
            head = f"UNAVAILABLE: {unready}.\n" if unready else ""
            return _cap(head + tool.description, max_chars)
    first = first_sentence(description)
    if any(p.lower() in first.lower() for p in b.text.drop_promises):
        first = ""
    lines: list[str] = []
    if unready:
        lines.append(f"UNAVAILABLE: {unready}.")
    if b.text.summary:
        lines.append(b.text.summary.strip())
    elif first:
        lines.append(first)
    bound = (b.result.rows_of or (b.derived.table if b.derived is not None else None) or contract.bound_table)
    t = contract.tables.get(bound) if bound else None
    if t is not None:
        line = f"Data: {_source_label(t)} · table {_table_name(bound)} — one record = {t.grain.rstrip('.')}."
        lines.append(line)
        key = [k for k in t.key if not k.endswith("#")]
        if key:
            nullable = [k for k in key if k in t.nullable_key]
            extra = f" ({' and '.join(nullable)} may be null)" if nullable else ""
            lines.append(f"Key: {', '.join(key)}{extra}.")
    arg_bits = []
    for name, a in contract.args.items():
        if a.accepts:
            kinds = " or ".join(k.split(":")[-1].replace("_", " ") for k in a.accepts)
            arg_bits.append(f"{name} accepts {kinds} (resolved; unknown -> error)")
        elif a.role == "anchor":
            arg_bits.append(f"{name} is the reference entity, excluded from the results")
        elif a.role == "limit" and a.limit_grain:
            arg_bits.append(f"{name} counts {a.limit_grain}s, not rows")
    if arg_bits:
        lines.append("Arguments: " + "; ".join(arg_bits) + ".")
    results = []
    order = _order_text(b)
    if order:
        results.append(order)
    if t is not None and t.is_item_table:
        results.append(f"one row = one item ({t.grain.rstrip('.')}); `_vbt.total` counts items")
    else:
        results.append("`_vbt.total` counts all matches")
    if t is not None:
        for g in t.spec.grains:
            results.append(f"`_vbt.grains.{g}` counts {g}s")
    lines.append("Results: " + "; ".join(results) + ".")
    thresholds = [a for a in contract.args.values() if a.op in ("ge", "gt", "le", "lt", "range", "ge_abs", "gt_abs",
                                                                 "le_abs", "lt_abs")]
    unknown_cols = []
    for a in thresholds:
        if isinstance(a.binds, str):
            unknown_cols.append(a.binds.split(".")[-1])
    if unknown_cols:
        lines.append(f"Rows with unknown {', '.join(dict.fromkeys(unknown_cols))} are excluded and counted in "
                     "`_vbt.excluded_unknown`.")
    for level, cols in b.result.levels.items():
        lines.append(f"{', '.join(cols[:4])}{' ...' if len(cols) > 4 else ''} are {level} values repeated on each "
                     f"row; count {level}s with `_vbt.grains.{level}`.")
    if t is not None:
        for name, c in t.columns.items():
            role = getattr(c, "role", None)
            cutoff = getattr(c, "cutoff", None)
            if cutoff is not None:
                op = {"lt": "<", "le": "≤", "gt": ">", "ge": "≥"}[cutoff.op]
                incl = "inclusive" if cutoff.op in ("le", "ge") else "exclusive"
                meaning = f"{cutoff.meaning} = " if cutoff.meaning else ""
                lines.append(f"Cutoff: {meaning}{name} {op} {cutoff.value:g}, {incl}.")
            fam = getattr(c, "family", None)
            if role == "measure" and fam:
                lines.append(f"{name} is adjusted within one {', '.join(fam)} family.")
            if role == "member":
                lines.append(_propagation(name, c.membership.propagation, (measured or {}).get(f"{bound}.{name}")))
            level = getattr(c, "level", None)
            if level and not any(name in cols for cols in b.result.levels.values()):
                lines.append(f"{name} is a {level} value repeated on each row; count {level}s, not rows.")
            if role == "flag" and getattr(c, "event_of", None):
                lines.append(f"{c.event_of} is right-censored when {name} is false; medians need Kaplan-Meier.")
            if getattr(c, "projection_of", None) and getattr(c, "lossy", False):
                lines.append(f"{name} keeps one value of {c.projection_of}; records under several are missed.")
            if role == "qualifier" and getattr(c, "effect", None) == "negate":
                lines.append(f"{name} = true marks a negative finding (the relation does NOT hold); such records are "
                             "excluded and counted in `_vbt.excluded_negated` unless include_negated is set.")
            if role == "qualifier" and getattr(c, "default_filter", False):
                applied = b.result.kind != "file"
                lines.append(f"{name}: {'rows excluded by default and counted' if applied else 'NOT applied by this tool'}"
                             f"{'' if applied else '; such rows are included'}.")
        ev = t.spec.evidence_nature
        if ev is not None:
            lines.append(f"Evidence: {ev.caveat.rstrip('.')}.")
        cov = t.spec.coverage
        nested = None
        if b.result.kind == "record" and not b.result.rows_of and len(b.result.row_paths) == 1 and \
                b.result.row_paths[0].startswith("$."):
            nested = t.columns.get(b.result.row_paths[0][2:].split(".")[0].split("[")[0])
            nested = nested if getattr(nested, "role", None) == "nested" else None
        if nested is not None:
            # a record read from a nested column has that column's coverage, never the table's (rev 2)
            cov = getattr(nested, "coverage", None)
            if cov is None or cov.absence_means != "absent":
                statement = cov.statement.rstrip(".") if cov is not None else None
                lines.append("Empty vs not found: unknown identifiers are errors; an empty (null) result is not "
                             "evidence of absence." + (f" {statement}." if statement else ""))
                lines.extend(n.strip() for n in b.text.notes if n.strip())
                return _cap("\n".join(lines), max_chars)
        if cov is not None and cov.absence_means == "censored" and cov.censor is not None:
            lines.append(f"Missing rows are censored ({cov.censor.column} {cov.censor.op} {cov.censor.value}): "
                         "not significant or not tested.")
        statement = cov.statement.rstrip(".") if cov is not None else None
        tail = f" {statement}." if statement else ""
        lines.append(f"Empty vs not found: unknown identifiers are errors; an empty result is citable only as an "
                     f"absence.{tail}")
    lines.extend(n.strip() for n in b.text.notes if n.strip())
    return _cap("\n".join(lines), max_chars)

