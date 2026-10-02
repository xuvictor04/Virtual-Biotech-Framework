"""Claim anchor rendering: streaming stripping, numbering, evidence status, appendix."""

from vbt.audit.render import (
    CLAIM_RE,
    StreamingAnchorStripper,
    evidence_status,
    find_refs,
    number_refs,
    render_claims_appendix_md,
    render_report_md,
    strip_refs,
)

CLAIMS = [
    {"id": "C1", "text": "IL1RL1 is highest in mast cells", "n_verified": 1, "confidence": "strong",
     "agent": "single-cell-analyst", "turn": 1,
     "evidence": [{"kind": "table", "path": "work/sc/results/tables/x.csv", "verified": True,
                   "evidence_status": "verified", "sha256": "ab" * 32}]},
    {"id": "C2.b", "text": "Literature reports asthma association", "n_verified": 0, "confidence": "weak",
     "evidence": [{"kind": "citation", "pmid": "12345678", "verified": False, "evidence_status": "external"}]},
]


def _stream(chunks):
    s = StreamingAnchorStripper()
    return "".join(s.feed(c) for c in chunks) + s.flush()


def test_anchor_regex_allows_dots_and_finds_in_order():
    assert CLAIM_RE.pattern == r"\[\[claim:([A-Za-z0-9_.-]+)\]\]"
    assert find_refs("a[[claim:C1]] b[[claim:C2.b]] c[[claim:C1]]") == ["C1", "C2.b", "C1"]
    assert strip_refs("x[[claim:C1]]y") == "xy"
    assert find_refs(None) == []


def test_streaming_stripper_handles_anchor_split_across_chunks():
    text = "IL1RL1 is high[[claim:C1]] and safe[[claim:C2.b]]."
    expected = "IL1RL1 is high and safe."
    # every possible split point, and character-by-character streaming
    for i in range(len(text) + 1):
        assert _stream([text[:i], text[i:]]) == expected, i
    assert _stream(list(text)) == expected
    assert _stream(["mast cells[[cla", "im:C", "1]], next"]) == "mast cells, next"


def test_streaming_stripper_releases_non_anchor_brackets():
    s = StreamingAnchorStripper()
    out = s.feed("see [1] and [[x]] and [[clai")
    assert out == "see [1] and [[x]] and "  # the possible anchor prefix is held back
    out += s.feed("med]] ok")  # turned out not to be an anchor
    out += s.flush()
    assert out == "see [1] and [[x]] and [[claimed]] ok"
    s2 = StreamingAnchorStripper()
    assert s2.feed("ends with [") == "ends with "
    assert s2.flush() == "["


def test_number_refs_orders_by_first_appearance_and_marks_status():
    text = "A[[claim:C2.b]] then B[[claim:C1]] again A[[claim:C2.b]] and C[[claim:C7]]."
    rendered, notes = number_refs(text, CLAIMS)
    assert rendered == "A[1†] then B[2] again A[1†] and C[C7?]."
    assert [n["id"] for n in notes] == ["C2.b", "C1", "C7"]
    assert notes[0]["n"] == 1 and notes[0]["verified"] is False and notes[0]["label"] == "1†"
    assert notes[1]["n"] == 2 and notes[1]["verified"] is True
    assert notes[2]["missing"] is True and notes[2]["n"] is None and notes[2]["label"] == "C7?"
    # also accepts the {stats, claims} payload and a dict keyed by id
    assert number_refs(text, {"stats": {}, "claims": CLAIMS})[0] == rendered
    assert number_refs("x[[claim:C1]]", {"C1": CLAIMS[0]})[0] == "x[1]"


def test_evidence_status_values():
    assert evidence_status({"kind": "table", "verified": True}) == "verified"
    assert evidence_status({"kind": "citation", "verified": False}) == "external"
    assert evidence_status({"kind": "table", "verified": False}) == "unresolved"
    assert evidence_status({"kind": "table", "verified": False, "evidence_status": "unresolved"}) == "unresolved"


def test_appendix_and_rendered_report():
    md = render_claims_appendix_md(CLAIMS)
    assert "## Claims and evidence" in md
    assert "### C1" in md and "work/sc/results/tables/x.csv" in md
    assert "external ref" in md and "PMID 12345678" in md
    assert "no verified evidence" in md  # C2.b has only an external citation
    out = render_report_md("# R\n\nClaim[[claim:C1]] and[[claim:C9]].", CLAIMS)
    assert "Claim[1] and[C9?]." in out
    assert "### References" in out and "dangling reference" in out
    assert "## Claims and evidence" in out
    assert "No claims were filed." in render_claims_appendix_md([])
