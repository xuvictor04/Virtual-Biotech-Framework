"""Claim-evidence objects and their validator (port of upstream ``src/utils/claims.py``).

A claim is one assertion in the CSO's synthesis. Its evidence entries must point
at things that exist *in this run*:

* local evidence (``artifact``/``figure``/``table``/``code``) resolves against the
  artifact registry (MANIFEST ``artifacts``, which also holds unregistered
  ``work/`` files) by exact path, then unique suffix, then unique basename; an
  ambiguous match is rejected. It is stored as the run-relative key, must sit
  under ``work/`` (harness records and anything outside the run are refused), and
  its hash is recorded. Refiling with a stale ``sha256`` is rejected.
* ``tool_call`` evidence must name a ``tool_use_id`` in the trace whose call
  finished without error (a pending call, including ``record_claims``'s own, is
  rejected).
* ``citation`` evidence (PMID, DOI or URL) is external: format-checked, stored
  ``verified=False`` with ``evidence_status='external'``.

Rejection is the point: an unvalidated evidence link looks like provenance while
being a guess. In strict mode any problem rejects the whole batch; non-strict
mode (status refresh, retrofit) downgrades problems to warnings and marks the
evidence ``unresolved``.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from .storage import (
    is_harness_rel,
    is_work_rel,
    read_json,
    rel_parts,
    sha256_file,
    to_rel,
)

CLAIM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

#: Evidence kinds. Local kinds resolve against the registry, tool_call against the
#: trace; citation is external and cannot be checked from here.
EVIDENCE_KINDS = frozenset({"artifact", "figure", "table", "code", "tool_call", "citation"})
LOCAL_KINDS = frozenset({"artifact", "figure", "table", "code"})
KIND_ALIASES = {
    "report": "artifact", "data": "artifact", "file": "artifact", "result": "artifact",
    "web": "citation", "url": "citation", "literature": "citation", "paper": "citation",
    "pmid": "citation", "doi": "citation", "reference": "citation",
    "tool": "tool_call", "tool_use": "tool_call", "toolcall": "tool_call",
}
CONFIDENCE_LEVELS = ("strong", "moderate", "weak")

_PMID_RE = re.compile(r"^\d{1,9}$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


@dataclass
class ValidationResult:
    claims: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "n_claims": len(self.claims), "errors": list(self.errors),
                "warnings": list(self.warnings)}


@dataclass
class EvidenceContext:
    """What evidence is resolved against."""

    run_dir: Path
    artifacts: Mapping[str, Mapping[str, Any]]
    calls: Mapping[str, Mapping[str, Any]]
    workspace: Path | None = None
    sha_cache: dict[str, str | None] = field(default_factory=dict)

    def current_sha(self, rel: str) -> tuple[str | None, str | None]:
        """(sha256, problem) for a run-relative file, cached per validation."""
        if rel in self.sha_cache:
            sha = self.sha_cache[rel]
            return (sha, None) if sha else (None, "is missing or unreadable")
        path = self.run_dir / rel
        try:
            actual = path.resolve(strict=True)
            root = self.run_dir.resolve()
            if actual != root and root not in actual.parents:
                self.sha_cache[rel] = None
                return None, "resolves outside this run directory"
            if not actual.is_file():
                self.sha_cache[rel] = None
                return None, "is not a file"
            sha = sha256_file(actual)
        except OSError as exc:
            self.sha_cache[rel] = None
            return None, f"is missing or unreadable ({exc.__class__.__name__})"
        self.sha_cache[rel] = sha
        return sha, None


# ----------------------------------------------------------------- citations

def normalize_pmid(value: Any) -> str | None:
    s = str(value or "").strip()
    s = re.sub(r"^(?:pmid|pubmed)\s*:?\s*", "", s, flags=re.I)
    return s if _PMID_RE.match(s) else None


def normalize_doi(value: Any) -> str | None:
    s = str(value or "").strip()
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", s, flags=re.I)
    return s if _DOI_RE.match(s) else None


def normalize_url(value: Any) -> str | None:
    s = str(value or "").strip()
    if not s or any(c.isspace() for c in s):
        return None
    u = urlparse(s)
    return s if u.scheme in ("http", "https") and u.netloc else None


# ----------------------------------------------------------------- path resolution

def _strip_dot_slash(p: str) -> str:
    while p.startswith("./"):
        p = p[2:]
    return p


def resolve_evidence_path(path: str, ctx: EvidenceContext) -> tuple[str | None, str | None]:
    """Resolve an evidence path to a registry key under work/. Returns (rel, problem)."""
    raw = str(path or "").strip()
    if not raw:
        return None, "missing 'path'"
    p = os.path.expanduser(raw)
    keys = [k for k in ctx.artifacts if is_work_rel(k)]

    def check(rel: str | None) -> tuple[str | None, str | None] | None:
        """A definitive answer for a resolved rel, or None to keep looking."""
        if rel is None:
            return None, f"{raw!r} is outside this run directory"
        if rel in ctx.artifacts:
            if not is_work_rel(rel):
                return None, f"{rel!r} is not under work/; only analysis outputs can be cited"
            return rel, None
        exists = (ctx.run_dir / rel).is_file() if rel else False
        if exists:
            if is_harness_rel(rel) or (rel_parts(rel) and rel_parts(rel)[0] in ("logs", "evidence", "inputs", "report")):
                return None, (f"{rel!r} is a harness record, not an analysis artifact; "
                              "cite the files under work/ that support the claim")
            if not is_work_rel(rel):
                return None, f"{rel!r} is not under work/; only analysis outputs can be cited"
            return None, f"{rel!r} is not a recorded artifact of this run (temporary or ignored file)"
        return None

    def match(norm: str) -> tuple[str | None, str | None]:
        norm = _strip_dot_slash(norm)
        hits = [k for k in keys if k == norm or k.endswith("/" + norm)]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, (f"{raw!r} is ambiguous: it matches {len(hits)} artifacts "
                          f"({', '.join(sorted(hits)[:4])}{', ...' if len(hits) > 4 else ''}); cite the full path")
        base = PurePosixPath(norm).name
        pool = keys
        parts = PurePosixPath(norm).parts
        if len(parts) >= 3 and parts[0] == "work":
            # A fully qualified path names its owner; never re-point it at another agent's file.
            pool = [k for k in keys if PurePosixPath(k).parts[1:2] == (parts[1],)]
        hits = [k for k in pool if PurePosixPath(k).name == base]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, (f"{raw!r} is ambiguous: {len(hits)} artifacts are named {base!r} "
                          f"({', '.join(sorted(hits)[:4])}{', ...' if len(hits) > 4 else ''}); cite the full path")
        return None, (f"{raw!r} is not a registered artifact of this run; call list_artifacts "
                      "for the exact paths")

    if os.path.isabs(p):
        rel = to_rel(p, ctx.run_dir)
        got = check(rel)
        if got is not None:
            return got
        return match(rel or "")
    norm = os.path.normpath(_strip_dot_slash(p)).replace(os.sep, "/")
    if norm.startswith("../") or norm == "..":
        # May still land inside the run when taken relative to a workspace.
        if ctx.workspace is not None:
            rel = to_rel(os.path.normpath(str(ctx.workspace / norm)), ctx.run_dir)
            if rel is None:
                return None, f"{raw!r} is outside this run directory"
            got = check(rel)
            if got is not None:
                return got
            return match(rel)
        return None, f"{raw!r} is outside this run directory"
    if ctx.workspace is not None:
        ws_rel = to_rel(os.path.normpath(str(ctx.workspace / norm)), ctx.run_dir)
        if ws_rel and (ws_rel in ctx.artifacts or (ctx.run_dir / ws_rel).is_file()):
            got = check(ws_rel)
            if got is not None:
                return got
    got = check(norm)
    if got is not None:
        return got
    return match(norm)


# ----------------------------------------------------------------- evidence checks

def _canonical_kind(ev: Mapping[str, Any]) -> str:
    kind = str(ev.get("kind") or "").strip().lower()
    if not kind:
        if ev.get("tool_use_id"):
            return "tool_call"
        if ev.get("path"):
            return "artifact"
        if ev.get("pmid") or ev.get("doi") or ev.get("url"):
            return "citation"
        return "artifact"
    return KIND_ALIASES.get(kind, kind)


def check_evidence(ev: Mapping[str, Any], ctx: EvidenceContext, *, stored: bool = False
                   ) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Validate one evidence item. Returns (entry, problems, warnings).

    ``entry`` is None only when the item cannot be represented at all (not an
    object or an unknown kind). With ``stored=True`` the item comes from
    claims.json: its path must be an exact registry key with a recorded sha256,
    and that hash is preserved.
    """
    if not isinstance(ev, Mapping):
        return None, ["is not an object"], []
    kind = _canonical_kind(ev)
    if kind not in EVIDENCE_KINDS:
        return None, [f"unknown kind {ev.get('kind')!r} (expected one of {sorted(EVIDENCE_KINDS)})"], []
    entry: dict[str, Any] = {"kind": kind}
    problems: list[str] = []
    warnings: list[str] = []
    if ev.get("note"):
        entry["note"] = str(ev["note"])[:500]

    if kind in LOCAL_KINDS:
        if stored:
            rel = str(ev.get("path") or "").strip()
            if not rel:
                problems.append("missing 'path'")
            elif rel not in ctx.artifacts:
                rel2, why = resolve_evidence_path(rel, ctx)
                problems.append(f"cites {rel!r}, which is not a registered artifact of this run"
                                + (f" ({why})" if why and rel2 is None else ""))
            elif not is_work_rel(rel):
                problems.append(f"cites {rel!r}, which is not under work/")
        else:
            rel, why = resolve_evidence_path(str(ev.get("path") or ""), ctx)
            if rel is None:
                problems.append(f"({kind}) {why}")
                rel = str(ev.get("path") or "")
        entry["path"] = rel
        if ev.get("line"):
            entry["line"] = ev["line"]
        if not problems:
            rec = ctx.artifacts.get(rel) or {}
            entry["produced_by"] = rec.get("produced_by")
            cur, problem = ctx.current_sha(rel)
            reg = rec.get("sha256")
            if problem:
                problems.append(f"artifact {rel!r} {problem}")
            elif reg and cur != reg:
                problems.append(f"artifact {rel!r} has changed since it was registered")
            elif ev.get("sha256") and str(ev["sha256"]) != cur:
                problems.append(f"artifact {rel!r} has changed since this claim was filed; "
                                "review the updated artifact and refile the claim")
            elif stored and not ev.get("sha256"):
                problems.append(f"artifact {rel!r} has no recorded sha256 (the claim was not filed "
                                "through record_claims)")
            entry["sha256"] = str(ev.get("sha256")) if (stored or ev.get("sha256")) and ev.get("sha256") else cur
        elif stored and ev.get("sha256"):
            entry["sha256"] = str(ev["sha256"])
    elif kind == "tool_call":
        tuid = str(ev.get("tool_use_id") or "").strip()
        entry["tool_use_id"] = tuid
        if not tuid:
            problems.append("(tool_call) missing 'tool_use_id'")
        else:
            call = ctx.calls.get(tuid)
            if call is None:
                problems.append(f"cites tool call {tuid!r}, which does not appear in this run's trace")
            else:
                entry["tool_name"] = call.get("tool") or call.get("tool_name")
                entry["agent"] = call.get("agent")
                entry["ts"] = call.get("started_at") or call.get("ts")
                if call.get("is_error"):
                    problems.append(f"cites failed tool call {tuid!r}; a failed query cannot support a finding")
                elif call.get("pending"):
                    problems.append(f"cites unfinished tool call {tuid!r}; wait for its result before citing it")
                elif str(entry.get("tool_name") or "").startswith("mcp__provenance__"):
                    warnings.append(f"tool call {tuid!r} is a provenance bookkeeping call, not evidence")
    else:  # citation
        raw = {k: ev.get(k) for k in ("pmid", "doi", "url") if ev.get(k)}
        if not raw:
            problems.append("(citation) needs pmid, doi or url")
        for k, v in raw.items():
            norm = {"pmid": normalize_pmid, "doi": normalize_doi, "url": normalize_url}[k](v)
            if norm is None:
                problems.append(f"(citation) {k} {str(v)[:80]!r} is not a valid {k.upper()}")
            else:
                entry[k] = norm
        if ev.get("title"):
            entry["title"] = str(ev["title"])[:300]
        if not problems:
            ref = entry.get("pmid") and f"PMID {entry['pmid']}" or entry.get("doi") or entry.get("url")
            warnings.append(f"external citation {ref} is not locally verifiable")

    if kind == "citation":
        entry["verified"] = False
        entry["evidence_status"] = "external" if not problems else "unresolved"
    else:
        entry["verified"] = not problems
        entry["evidence_status"] = "verified" if not problems else "unresolved"
    return entry, problems, warnings


# ----------------------------------------------------------------- claims

def _confidence(raw: Mapping[str, Any], cid: str, warnings: list[str]) -> str:
    confidence = str(raw.get("confidence") or "moderate").strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        warnings.append(f"claim {cid}: unknown confidence {confidence!r}, using 'moderate'")
        confidence = "moderate"
    return confidence


def validate_claims(claims: Any, ctx: EvidenceContext, *, strict: bool = True, stored: bool = False,
                    turn: int | None = None, filed_by: str | None = None) -> ValidationResult:
    """Validate a batch of claims. Input objects are never mutated."""
    res = ValidationResult()
    if isinstance(claims, Mapping):
        claims = claims.get("claims", [])
    if isinstance(claims, str):
        import json
        try:
            claims = json.loads(claims)
        except ValueError:
            res.errors.append("claims must be a list of claim objects")
            return res
        if isinstance(claims, Mapping):
            claims = claims.get("claims", [])
    if not isinstance(claims, list):
        res.errors.append("claims must be a list of claim objects")
        return res
    if not claims and not stored:
        res.errors.append("no claims given")
        return res
    seen: set[str] = set()
    for i, raw in enumerate(claims):
        where = f"claim[{i}]"
        if not isinstance(raw, Mapping):
            res.errors.append(f"{where}: not an object")
            continue
        cid = str(raw.get("id") or "").strip()
        if not cid:
            res.errors.append(f"{where}: missing 'id'")
            continue
        if not CLAIM_ID_RE.match(cid):
            res.errors.append(f"{where}: id {cid[:70]!r} must match {CLAIM_ID_RE.pattern}")
            continue
        if cid in seen:
            res.errors.append(f"claim {cid}: duplicate id in this batch")
            continue
        seen.add(cid)
        text = str(raw.get("text") or "").strip()
        if not text:
            res.errors.append(f"claim {cid}: missing 'text'")
            continue
        confidence = _confidence(raw, cid, res.warnings)
        raw_ev = raw.get("evidence")
        if isinstance(raw_ev, Mapping):
            raw_ev = [raw_ev]
        if not isinstance(raw_ev, list) or not raw_ev:
            res.errors.append(f"claim {cid}: must cite at least one piece of evidence")
            continue
        evidence: list[dict[str, Any]] = []
        for j, ev in enumerate(raw_ev):
            entry, problems, warns = check_evidence(ev, ctx, stored=stored)
            res.warnings.extend(f"claim {cid}: {w}" for w in warns)
            msgs = [f"claim {cid}: evidence[{j}] {p}" for p in problems]
            if msgs:
                (res.errors if strict else res.warnings).extend(msgs)
            if entry is None or (strict and problems):
                continue
            evidence.append(entry)
        if not evidence:
            if not strict or not any(e.startswith(f"claim {cid}:") for e in res.errors):
                res.errors.append(f"claim {cid}: no evidence survived validation")
            continue
        agent = str(raw.get("agent") or "").strip() or None
        if not agent:
            producers = [e.get("produced_by") or e.get("agent") for e in evidence]
            producers = [p for p in producers if p and not str(p).startswith("_")]
            agent = producers[0] if producers else None
        claim = {
            "id": cid, "text": text[:4000], "agent": agent, "confidence": confidence,
            "turn": turn if turn is not None else raw.get("turn"),
            "evidence": evidence,
            "n_verified": sum(1 for e in evidence if e.get("verified")),
        }
        if filed_by or raw.get("filed_by"):
            claim["filed_by"] = filed_by or raw.get("filed_by")
        if stored:
            for k in ("filed", "filed_by"):
                if raw.get(k) is not None:
                    claim[k] = raw[k]
        res.claims.append(claim)
    return res


def refresh_claims(stored: Iterable[Mapping[str, Any]], ctx: EvidenceContext) -> list[dict[str, Any]]:
    """Refresh verified/evidence_status/n_verified of filed claims without changing pointers or hashes.

    Later turns may rewrite or delete artifacts; only an explicit refile may
    update a claim's evidence, so stale pointers become ``unresolved``.
    """
    out = []
    for c in stored or []:
        if not isinstance(c, Mapping):
            continue
        claim = copy.deepcopy(dict(c))
        evs = []
        for ev in claim.get("evidence") or []:
            if not isinstance(ev, Mapping):
                continue
            entry, problems, _w = check_evidence(ev, ctx, stored=True)
            new = dict(ev)
            if entry is None:
                new["verified"], new["evidence_status"] = False, "unresolved"
            else:
                new["verified"] = entry["verified"]
                new["evidence_status"] = entry["evidence_status"]
                for k in ("tool_name", "agent", "produced_by"):
                    if entry.get(k) and not new.get(k):
                        new[k] = entry[k]
            if problems:
                new["problem"] = "; ".join(problems)[:500]
            else:
                new.pop("problem", None)
            evs.append(new)
        claim["evidence"] = evs
        claim["n_verified"] = sum(1 for e in evs if e.get("verified"))
        out.append(claim)
    return out


def merge_claims(existing: Iterable[Mapping[str, Any]], new: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge newly filed claims; a re-filed id replaces the earlier version in place."""
    out = [dict(c) for c in existing or [] if isinstance(c, Mapping) and c.get("id")]
    index = {c["id"]: i for i, c in enumerate(out)}
    for c in new or []:
        c = dict(c)
        if c["id"] in index:
            out[index[c["id"]]] = c
        else:
            index[c["id"]] = len(out)
            out.append(c)
    return out


def link_cited_by(artifacts: dict[str, dict[str, Any]], claims: Iterable[Mapping[str, Any]]) -> None:
    """Populate each artifact's ``cited_by`` so the registry reads both ways."""
    for e in artifacts.values():
        if isinstance(e, dict):
            e["cited_by"] = []
    for c in claims or []:
        for ev in c.get("evidence") or []:
            p = ev.get("path") if isinstance(ev, Mapping) else None
            if p and p in artifacts and isinstance(artifacts[p], dict):
                cited = artifacts[p].setdefault("cited_by", [])
                if c.get("id") not in cited:
                    cited.append(c.get("id"))


def _count(items: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        k = str(i)
        out[k] = out.get(k, 0) + 1
    return out


def claim_stats(claims: list[Mapping[str, Any]]) -> dict[str, Any]:
    evs = [e for c in claims for e in c.get("evidence") or [] if isinstance(e, Mapping)]

    def status(e: Mapping[str, Any]) -> str:
        s = e.get("evidence_status")
        if s:
            return str(s)
        return "verified" if e.get("verified") else ("external" if e.get("kind") == "citation" else "unresolved")

    return {
        "n_claims": len(claims),
        "n_evidence": len(evs),
        "n_verified_evidence": sum(1 for e in evs if status(e) == "verified"),
        "n_external_evidence": sum(1 for e in evs if status(e) == "external"),
        "n_unresolved_evidence": sum(1 for e in evs if status(e) == "unresolved"),
        "claims_without_verified_evidence": [c.get("id") for c in claims if not c.get("n_verified")],
        "by_agent": _count(c.get("agent") or "unattributed" for c in claims),
        "by_confidence": _count(c.get("confidence") or "moderate" for c in claims),
        "by_turn": _count(c.get("turn") if c.get("turn") is not None else "?" for c in claims),
    }


def claims_payload(claims: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"stats": claim_stats(claims), "claims": list(claims)}


def read_claims_file(path: str | Path) -> tuple[list[dict[str, Any]], str | None]:
    """(claims, error). Reads ``{stats, claims}`` and the legacy bare-list format."""
    p = Path(path)
    if not p.exists():
        return [], None
    data = read_json(p, _SENTINEL)
    if data is _SENTINEL:
        return [], f"{p.name} is not valid JSON"
    claims = data.get("claims") if isinstance(data, Mapping) else data
    if not isinstance(claims, list):
        return [], f"{p.name}: claims must be a list of claim objects"
    bad = [i for i, c in enumerate(claims) if not isinstance(c, Mapping) or not c.get("id")]
    if bad:
        return [c for c in claims if isinstance(c, Mapping) and c.get("id")], \
            f"{p.name}: {len(bad)} entr{'y is' if len(bad) == 1 else 'ies are'} not claim objects"
    return [dict(c) for c in claims], None


_SENTINEL = object()


def load_claims(run_dir: str | Path) -> list[dict[str, Any]]:
    """Filed claims of a run (empty when missing or unreadable)."""
    claims, _err = read_claims_file(Path(run_dir) / "evidence" / "claims.json")
    return claims


__all__ = [
    "CLAIM_ID_RE", "EVIDENCE_KINDS", "LOCAL_KINDS", "KIND_ALIASES", "ValidationResult", "EvidenceContext",
    "validate_claims", "refresh_claims", "check_evidence", "resolve_evidence_path", "merge_claims",
    "link_cited_by", "claim_stats", "claims_payload", "read_claims_file", "load_claims",
    "normalize_pmid", "normalize_doi", "normalize_url",
]
