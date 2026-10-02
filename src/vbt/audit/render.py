"""Rendering ``[[claim:ID]]`` anchors and claim evidence for readers.

The CSO cites its filed claims inline as ``[[claim:C3]]``. Raw anchors are noise
to a reader, so every surface (CLI, rendered report, web UI) goes through here:

* ``StreamingAnchorStripper`` hides anchors while text streams, including an
  anchor split across chunks (claims are filed at the end of a turn, so the ids
  do not resolve yet mid-stream).
* ``number_refs`` turns anchors into footnote numbers in order of first
  appearance. A claim with no verified evidence is marked ``†``; an anchor whose
  claim was never filed is shown as ``[C7?]``, visibly broken rather than dropped.
* ``render_claims_appendix_md`` lists every claim with each evidence item's
  status: ``verified`` (resolved against this run's records), ``external`` (a
  citation, not checkable locally) or ``unresolved`` (a local pointer that is not
  on record; a defect).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

CLAIM_RE = re.compile(r"\[\[claim:([A-Za-z0-9_.-]+)\]\]")
_ANCHOR_OPEN = "[[claim:"
_PARTIAL_RE = re.compile(r"\[\[claim:[A-Za-z0-9_.-]*\]?")
_MAX_ID = 64

UNVERIFIED_MARK = "†"

EVIDENCE_STATUS_LABELS = {
    "verified": "verified",
    "external": "external ref",
    "unresolved": "not on record",
}


def find_refs(text: str | None) -> list[str]:
    """Claim ids referenced inline, in order (with repeats)."""
    return CLAIM_RE.findall(text or "")


def strip_refs(text: str | None) -> str:
    return CLAIM_RE.sub("", text or "")


def _is_partial_anchor(s: str) -> bool:
    """True when ``s`` could still grow into a complete ``[[claim:ID]]`` anchor."""
    if len(s) <= len(_ANCHOR_OPEN):
        return _ANCHOR_OPEN.startswith(s)
    if len(s) > len(_ANCHOR_OPEN) + _MAX_ID + 1:
        return False
    return _PARTIAL_RE.fullmatch(s) is not None


class StreamingAnchorStripper:
    """Remove ``[[claim:..]]`` anchors from streamed text, across chunk boundaries.

    ``feed(chunk)`` returns the text that is safe to display now; a trailing
    fragment that may be the start of an anchor is held back until the next
    chunk decides it. ``flush()`` releases whatever is still held at the end.
    """

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str | None) -> str:
        buf = self._pending + (chunk or "")
        buf = CLAIM_RE.sub("", buf)
        hold = len(buf)
        start = max(0, len(buf) - (len(_ANCHOR_OPEN) + _MAX_ID + 2))
        i = buf.find("[", start)
        while i != -1:
            if _is_partial_anchor(buf[i:]):
                hold = i
                break
            i = buf.find("[", i + 1)
        self._pending = buf[hold:]
        return buf[:hold]

    def flush(self) -> str:
        out, self._pending = CLAIM_RE.sub("", self._pending), ""
        return out


# ----------------------------------------------------------------- evidence


def evidence_status(ev: Mapping[str, Any]) -> str:
    """``verified`` / ``external`` / ``unresolved`` for one evidence entry."""
    status = ev.get("evidence_status")
    if status in EVIDENCE_STATUS_LABELS:
        return status
    if ev.get("verified"):
        return "verified"
    return "external" if ev.get("kind") == "citation" else "unresolved"


def describe_evidence(ev: Mapping[str, Any]) -> str:
    """One-line description of an evidence item, e.g. ``table work/a/x.csv``."""
    kind = ev.get("kind") or "artifact"
    if kind == "tool_call":
        what = ev.get("tool_name") or ev.get("tool") or "tool call"
        s = f"tool_call {what} ({ev.get('tool_use_id', '?')})"
        if ev.get("agent"):
            s += f" by {ev['agent']}"
        return s
    if kind == "citation":
        if ev.get("pmid"):
            ref = f"PMID {ev['pmid']}"
        elif ev.get("doi"):
            ref = f"doi:{ev['doi']}"
        else:
            ref = str(ev.get("url") or "?")
        if ev.get("title"):
            ref += f" — {ev['title']}"
        return f"citation {ref}"
    s = f"{kind} {ev.get('path', '?')}"
    if ev.get("line"):
        s += f":{ev['line']}"
    return s


def _claims_by_id(claims: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not claims:
        return {}
    if isinstance(claims, Mapping):
        if "claims" in claims and isinstance(claims["claims"], list):
            claims = claims["claims"]
        else:
            return {str(k): v for k, v in claims.items() if isinstance(v, Mapping)}
    out: dict[str, Mapping[str, Any]] = {}
    for c in claims:  # type: ignore[union-attr]
        if isinstance(c, Mapping) and c.get("id"):
            out[str(c["id"])] = c
    return out


def _n_verified(claim: Mapping[str, Any]) -> int:
    if "n_verified" in claim:
        try:
            return int(claim.get("n_verified") or 0)
        except (TypeError, ValueError):
            return 0
    return sum(1 for e in claim.get("evidence") or [] if isinstance(e, Mapping) and evidence_status(e) == "verified")


def number_refs(text: str | None, claims) -> tuple[str, list[dict[str, Any]]]:
    """Replace anchors with numbered markers in order of first appearance.

    Returns ``(rendered_text, footnotes)``. Filed claims become ``[n]`` (``[n†]``
    when none of their evidence is verified); unfiled ids become ``[C7?]``.
    Each footnote is ``{n, id, label, marker, missing, verified, n_verified,
    text, confidence, agent, evidence}``; unfiled ids have ``n=None`` and
    ``missing=True`` and are listed after the numbered ones.
    """
    by_id = _claims_by_id(claims)
    order: dict[str, int] = {}
    numbered: list[dict[str, Any]] = []
    missing: dict[str, dict[str, Any]] = {}

    def repl(m: re.Match) -> str:
        cid = m.group(1)
        claim = by_id.get(cid)
        if claim is None:
            if cid not in missing:
                missing[cid] = {"n": None, "id": cid, "label": f"{cid}?", "marker": f"[{cid}?]",
                                "missing": True, "verified": False, "n_verified": 0,
                                "text": "", "confidence": None, "agent": None, "evidence": []}
            return f"[{cid}?]"
        if cid not in order:
            order[cid] = len(order) + 1
            nv = _n_verified(claim)
            label = f"{order[cid]}" if nv else f"{order[cid]}{UNVERIFIED_MARK}"
            numbered.append({
                "n": order[cid], "id": cid, "label": label, "marker": f"[{label}]",
                "missing": False, "verified": nv > 0, "n_verified": nv,
                "text": str(claim.get("text") or ""), "confidence": claim.get("confidence"),
                "agent": claim.get("agent"), "evidence": list(claim.get("evidence") or []),
            })
        fn = numbered[order[cid] - 1]
        return fn["marker"]

    rendered = CLAIM_RE.sub(repl, text or "")
    return rendered, numbered + list(missing.values())


def render_footnotes_md(footnotes: list[dict[str, Any]]) -> str:
    """Markdown footnote list for the output of ``number_refs``."""
    if not footnotes:
        return ""
    lines = []
    for fn in footnotes:
        if fn.get("missing"):
            lines.append(f"- [{fn['label']}] **{fn['id']}** — no claim with this id was filed "
                         "(dangling reference).")
            continue
        ev = "; ".join(f"{describe_evidence(e)} ({evidence_status(e)})"
                       for e in fn.get("evidence") or [] if isinstance(e, Mapping))
        flag = "" if fn.get("verified") else " — *no verified evidence*"
        lines.append(f"- [{fn['label']}] **{fn['id']}**: {fn.get('text', '')}{flag}"
                     + (f"  \n  Evidence: {ev}" if ev else ""))
    return "\n".join(lines)


def render_claims_appendix_md(claims, title: str = "Claims and evidence") -> str:
    """A markdown appendix listing every filed claim and each evidence item's status."""
    items = list(_claims_by_id(claims).values())
    lines = [f"## {title}", ""]
    if not items:
        lines += ["No claims were filed.", ""]
        return "\n".join(lines)
    n_ver = sum(1 for c in items if _n_verified(c))
    lines += [f"{len(items)} claim(s) on record; {n_ver} with at least one verified evidence item. "
              "*verified* = resolved against this run's artifacts or trace; *external ref* = a citation "
              "that cannot be checked locally; *not on record* = a local pointer that does not resolve.", ""]
    for c in items:
        meta = ", ".join(str(x) for x in (c.get("confidence"), c.get("agent"),
                                          f"turn {c['turn']}" if c.get("turn") else None) if x)
        flag = "" if _n_verified(c) else " — **no verified evidence**"
        lines.append(f"### {c.get('id')}{flag}")
        lines.append("")
        lines.append(f"{c.get('text', '')}" + (f"  \n*{meta}*" if meta else ""))
        lines.append("")
        for e in c.get("evidence") or []:
            if not isinstance(e, Mapping):
                continue
            status = evidence_status(e)
            note = f" — {e['note']}" if e.get("note") else ""
            sha = f" `sha256:{str(e['sha256'])[:12]}`" if e.get("sha256") else ""
            lines.append(f"- `{EVIDENCE_STATUS_LABELS[status]}` {describe_evidence(e)}{sha}{note}")
        lines.append("")
    return "\n".join(lines)


def render_report_md(text: str | None, claims, *, title: str | None = None) -> str:
    """Rendered copy of a report: numbered anchors, footnotes and the claims appendix."""
    body, footnotes = number_refs(text or "", claims)
    parts = []
    if title:
        parts += [f"# {title}", ""]
    parts.append(body.rstrip())
    if footnotes:
        parts += ["", "---", "", "### References", "", render_footnotes_md(footnotes)]
    parts += ["", render_claims_appendix_md(claims)]
    return "\n".join(parts).rstrip() + "\n"
