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
from typing import Any

__all__ = ["describe_tool", "unavailable_text", "first_sentence"]

_SECTION = re.compile(r"\n\s*(Returns?|Examples?|Args?|Arguments|Parameters|Raises|Note)s?\s*:", re.IGNORECASE)
_SENTENCE = re.compile(r"(.+?[.!?])(\s|$)", re.DOTALL)


def first_sentence(description: str | None) -> str:
    text = _SECTION.split(str(description or ""), maxsplit=1)[0].strip()
    text = " ".join(text.split())
    m = _SENTENCE.match(text)
    return (m.group(1) if m else text).strip()


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
        return f"ordered by the {b.result.order_from_arg} argument"
    if not order:
        return None
    first = order[0]
    within = f" within {', '.join(first.within)}" if first.within else ""
    how = {"witness": "verified by the harness", "upstream_full_sort": "upstream sorts every match",
           "source_server_side": "ranked by the source, not verified"}[b.result.order_source]
    return f"ranked by {first.column} {first.direction}{within} ({how})"


def describe_tool(contract: Any, description: str | None, *, catalog: Any = None, max_chars: int = 1200,
                  unready: str | None = None) -> str:
    """The rewritten description (generic tools: unchanged). ``unready`` is the reason every call
    would be unready (listed with an ``UNAVAILABLE:`` prefix)."""
    b = getattr(contract, "binding", None)
    if b is None or getattr(contract, "generic", False):
        return description or ""
    if b.serve == "block" and b.block is not None:
        return _cap(unavailable_text(b.block.reason, list(b.block.alternatives), description), max_chars)
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
                prop = c.membership.propagation
                lines.append({"direct": f"{name}: direct annotations only.",
                              "propagated": f"{name}: annotations propagated to ancestors.",
                              "mixed": f"{name}: mixed direct and propagated annotations."}[prop])
            if role == "flag" and getattr(c, "event_of", None):
                lines.append(f"{c.event_of} is right-censored when {name} is false; medians need Kaplan-Meier.")
            if getattr(c, "projection_of", None) and getattr(c, "lossy", False):
                lines.append(f"{name} keeps one value of {c.projection_of}; records under several are missed.")
            if role == "qualifier" and getattr(c, "default_filter", False):
                applied = b.result.kind != "file"
                lines.append(f"{name}: {'rows excluded by default and counted' if applied else 'NOT applied by this tool'}"
                             f"{'' if applied else '; such rows are included'}.")
        ev = t.spec.evidence_nature
        if ev is not None:
            lines.append(f"Evidence: {ev.caveat.rstrip('.')}.")
        cov = t.spec.coverage
        if cov is not None and cov.absence_means == "censored" and cov.censor is not None:
            lines.append(f"Missing rows are censored ({cov.censor.column} {cov.censor.op} {cov.censor.value}): "
                         "not significant or not tested.")
        statement = cov.statement.rstrip(".") if cov is not None else None
        tail = f" {statement}." if statement else ""
        lines.append(f"Empty vs not found: unknown identifiers are errors; an empty result is citable only as an "
                     f"absence.{tail}")
    lines.extend(n.strip() for n in b.text.notes if n.strip())
    return _cap("\n".join(lines), max_chars)

