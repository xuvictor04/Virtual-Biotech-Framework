"""Tool-failure semantics (port of the upstream unresolved_tool_failures rules)."""

import json

from vbt import failures as fl

TOOL = "mcp__drug__search_known_drugs"


def _rec(tool, inp, ok, err="Data unavailable", tuid=None):
    r = {"tool": tool, "input": inp, "is_error": not ok, "tool_use_id": tuid}
    if not ok:
        r["error"] = err
    return r


def test_successful_retry_resolves_only_the_same_query():
    recs = [_rec(TOOL, {"target_id": "ENSG_PCSK9", "min_phase": 4}, False)]
    assert TOOL in fl.data_failure_notice(fl.unresolved_failures(recs))
    recs.append(_rec(TOOL, {"target_id": "ENSG_LPA", "min_phase": 4}, True))
    assert len(fl.unresolved_failures(recs)) == 1, "a different target does not resolve the failure"
    recs.append(_rec(TOOL, {"min_phase": 4, "target_id": "ENSG_PCSK9"}, True))  # key order does not matter
    assert fl.unresolved_failures(recs) == []
    assert fl.data_failure_notice(fl.unresolved_failures(recs)) == ""
    assert len(fl.summarize(recs)["all"]) == 1, "failed attempts stay in the audit list"


def test_a_failure_after_a_success_is_unresolved():
    recs = [_rec(TOOL, {"q": 1}, True), _rec(TOOL, {"q": 1}, False)]
    assert [f["tool"] for f in fl.unresolved_failures(recs)] == [TOOL]


def test_data_source_classification():
    assert fl.is_data_source("mcp__genetics__l2g")
    assert fl.is_data_source("WebSearch") and fl.is_data_source("WebFetch")
    assert not fl.is_data_source("mcp__provenance__record_claims")
    for t in ("Bash", "Read", "Edit", "Write", "Task", "QueryToolOutput", "", None):
        assert not fl.is_data_source(t)


def test_notice_wording_is_upstream_exact_and_lists_only_data_tools():
    unresolved = [_rec("Bash", {"command": "x"}, False), _rec(TOOL, {"a": 1}, False),
                  _rec("WebFetch", {"url": "u"}, False), _rec(TOOL, {"a": 2}, False),
                  _rec("mcp__provenance__write_plan", {}, False)]
    assert fl.data_failure_notice(unresolved) == (
        "Data/evidence warning: these tools failed during this turn: mcp__drug__search_known_drugs, WebFetch. "
        "Their results cannot support this answer. Any alternative sources must be identified separately; "
        "evidence coverage is incomplete.")
    assert fl.data_failure_notice([_rec("Bash", {}, False), _rec("Edit", {}, False)]) == ""


def test_summarize_separates_unresolved_data_from_recovered():
    recs = [_rec("Bash", {"command": "python a.py"}, False, "exit code 1"),
            _rec("Bash", {"command": "python a.py"}, True),
            _rec("Edit", {"file_path": "x"}, False, "old_string not found"),
            _rec(TOOL, {"t": 1}, False),
            _rec("mcp__other__x", {"t": 1}, False),
            _rec("mcp__other__x", {"t": 1}, True)]
    s = fl.summarize(recs)
    assert [f["tool"] for f in s["unresolved_data"]] == [TOOL]
    assert len(s["all"]) == 4
    assert s["recovered_count"] == 2
    assert s["other_error_count"] == 1
    assert s["unresolved_data"][0]["tool_name"] == TOOL and s["unresolved_data"][0]["error"]


def test_summarize_does_not_count_repeated_or_interrupted_failures_as_recovered():
    q = {"g": "X"}
    recs = [_rec(TOOL, q, False, "timeout"), _rec(TOOL, q, False, "timeout"), _rec(TOOL, q, False, "timeout"),
            _rec("Bash", {"command": "a"}, False, "exit 1"), _rec("Bash", {"command": "a"}, False, "exit 1"),
            _rec("mcp__other__x", {"t": 2}, False, "Interrupted")]
    s = fl.summarize(recs)
    assert len(s["all"]) == 6
    assert [f["tool"] for f in s["unresolved_data"]] == [TOOL, "mcp__other__x"]
    assert s["recovered_count"] == 0
    assert s["other_error_count"] == 2
    # A success with different input does not recover; an identical later success recovers every earlier failure.
    s = fl.summarize(recs + [_rec(TOOL, {"g": "Y"}, True), _rec("Bash", {"command": "a"}, True)])
    assert s["recovered_count"] == 2 and s["other_error_count"] == 0
    assert [f["tool"] for f in s["unresolved_data"]] == [TOOL, "mcp__other__x"]
    assert fl.summarize([])["recovered_count"] == 0


def test_unresolved_from_trace_joins_start_input_and_spilled_inputs(tmp_path):
    big = {"query": "x" * 50}
    (tmp_path / "logs" / "tool_inputs").mkdir(parents=True)
    (tmp_path / "logs" / "tool_inputs" / "b.json").write_text(json.dumps({"tool_use_id": "b", "input": big}))
    events = [
        {"type": "tool_start", "tool_use_id": "a", "tool": TOOL, "agent": "g", "input": {"target_id": "T1"}},
        {"type": "tool_end", "tool_use_id": "a", "tool": TOOL, "agent": "g", "is_error": True, "output": "Error: down"},
        {"type": "tool_start", "tool_use_id": "b", "tool": "WebSearch", "agent": "cso",
         "input_ref": "logs/tool_inputs/b.json", "input_sha256": fl.input_sha256(big)},
        {"type": "tool_end", "tool_use_id": "b", "tool": "WebSearch", "agent": "cso", "is_error": True,
         "output": "Error: 503"},
        {"type": "tool_start", "tool_use_id": "c", "tool": "WebSearch", "agent": "cso", "input": big},
        {"type": "tool_end", "tool_use_id": "c", "tool": "WebSearch", "agent": "cso", "is_error": False},
        {"type": "tool_start", "tool_use_id": "d", "tool": "Bash", "agent": "g", "input": {"command": "false"}},
        {"type": "tool_end", "tool_use_id": "d", "tool": "Bash", "agent": "g", "is_error": True, "output": "exit 1"},
        {"type": "tool_start", "tool_use_id": "e", "tool": TOOL, "agent": "g", "input": {"target_id": "T1"}},
        # e never ended (interrupted): not a completed call, ignored
    ]
    all_unresolved = fl.unresolved_from_trace(events, run_dir=tmp_path)
    assert [(f["tool"], f["tool_use_id"]) for f in all_unresolved] == [(TOOL, "a"), ("Bash", "d")]
    assert all_unresolved[0]["error"] == "Error: down"
    data = fl.unresolved_from_trace(events, run_dir=tmp_path, data_only=True)
    assert [f["tool_use_id"] for f in data] == ["a"]
    # without the run dir the spilled input is keyed by its sha256 and still resolved by the identical call
    assert [f["tool_use_id"] for f in fl.unresolved_from_trace(events, data_only=True)] == ["a"]
    assert [f["tool_use_id"] for f in fl.unresolved_from_trace(events, agent="cso")] == []


def test_canonical_key_and_sha_are_order_insensitive():
    assert fl.canonical_key("t", {"a": 1, "b": [1, 2]}) == fl.canonical_key("t", {"b": [1, 2], "a": 1})
    assert fl.input_sha256({"a": 1, "b": 2}) == fl.input_sha256({"b": 2, "a": 1})
    assert fl.canonical_key("t", None) == fl.canonical_key("t", {})
