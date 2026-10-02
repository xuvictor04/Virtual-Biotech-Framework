"""README.md and audit.html: rendered from on-disk records, for finished and crashed runs."""

import base64
import hashlib
import html
import json
import os
import re

from vbt.audit.report import (
    AUDIT_CSP,
    AUDIT_SCRIPT,
    AUDIT_SCRIPT_HASH,
    TOOL_PAGE_SIZE,
    collect,
    render_audit_html,
    render_readme,
    write_reports,
)
from vbt.orchestrator import open_session
from vbt.providers.mock import ScriptedProvider, call, reply
from vbt.session import Run
from vbt.verify import verify_run

TABLE = "work/genomics-analyst/results/tables/l2g.csv"
XSS = "<script>alert(1)</script>"


def tool_call(run, agent, tool, tuid, inp=None, *, action=None, is_error=False, output="ok"):
    run.trace("tool_start", agent=agent, tool=tool, tool_use_id=tuid, input=inp or {})
    if action:
        action()
    run.trace("tool_end", agent=agent, tool=tool, tool_use_id=tuid, is_error=is_error, duration_s=0.1, output=output)


def write(run, rel, text="x\n"):
    p = run.dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def dispatch(run, agent, tuid, body=None, prompt="go"):
    run.trace("tool_start", agent="cso", tool="Task", tool_use_id=tuid, input={"subagent_type": agent})
    run.trace("delegation", parent="cso", agent=agent, description=f"{agent} task", prompt=prompt, tool_use_id=tuid)
    run.trace("agent_start", agent=agent, depth=1)
    if body:
        body()
    run.trace("agent_end", agent=agent, depth=1, stop="end_turn", cost_usd=0.12, model_calls=3, tool_calls=1,
              duration_s=1.5)
    run.trace("tool_end", agent="cso", tool="Task", tool_use_id=tuid, is_error=False, output="report")


async def test_reports_for_a_mock_session(config):
    config["orchestration"]["enforce_review"] = False
    provider = ScriptedProvider.from_rules({
        "cso": [
            reply("Delegating.", call("Task", subagent_type="genomics-analyst", description="genetics",
                                      prompt="Assess EGFR genetics in LUAD")),
            reply(call("mcp__provenance__write_plan", goal="EGFR", steps=[
                {"id": "s1", "agent": "genomics-analyst", "task": "genetics"},
                {"id": "s2", "agent": "single-cell-analyst", "task": "expression", "depends_on": ["s1"]}])),
            reply(call("mcp__provenance__record_claims", claims=[
                {"id": "C1", "text": "EGFR L2G is 0.82.", "evidence": [{"kind": "table", "path": TABLE},
                                                                    {"kind": "citation", "pmid": "15118073"}]}])),
            reply("EGFR is supported [[claim:C1]]."),
        ],
        "genomics-analyst": [
            reply(call("Write", file_path="results/tables/l2g.csv", content="gene,l2g\nEGFR,0.82\n")),
            reply(call("mcp__nope__missing", q=1)),
            reply("done"),
        ],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    await session.ask("Is EGFR a target?")
    run_dir = session.run.dir
    # written on every save (vbt.session.Run calls write_reports)
    assert (run_dir / "README.md").is_file() and (run_dir / "audit.html").is_file()
    await session.close()

    readme = (run_dir / "README.md").read_text()
    for heading in ("## Turns", "## How the analysis flowed", "## What each agent produced",
                    "## Claims and their evidence", "## Reproducing this run", "## Directory layout",
                    "## The analysis plan"):
        assert heading in readme, heading
    assert f"vbt verify {run_dir.name} --rerun" in readme and f"vbt replay {run_dir.name}" in readme
    assert "Assess EGFR genetics in LUAD" in readme            # delegation prompt
    assert TABLE in readme and "Write" in readme                # artifact with its origin
    assert "**verified** table" in readme and "**external ref** citation PMID 15118073" in readme
    assert "not_run" in readme                                  # planned single-cell step never ran
    assert "Pinned configuration" in readme

    page = (run_dir / "audit.html").read_text()
    assert 'id="claim-C1"' in page and 'href="#claim-C1"' in page
    assert 'id="tool-table"' in page and "mcp__nope__missing" in page
    assert re.search(r'<tr class=err data-err="1">', page)
    assert "<svg class=\"tl\"" in page                          # specialist timeline
    assert "Assess EGFR genetics in LUAD" in page
    # self-contained: no external scripts, stylesheets or images
    assert not re.search(r'<(script|img|link)[^>]+(src|href)="https?:', page)
    assert "<link" not in page
    m = json.loads((run_dir / "MANIFEST.json").read_text())
    assert {"README.md", "audit.html"} <= set(m["harness_files"])
    v = verify_run(run_dir)
    assert not v["integrity"]["harness_changed"], v["integrity"]
    assert v["integrity"]["status"] == "passed"


def test_reports_for_a_crashed_run(tmp_path):
    run = Run(tmp_path / "runs", run_id="CRASHED")
    run.trace("turn_start", turn=1, prompt="Is EGFR a target?")
    dispatch(run, "genomics-analyst", "t1", lambda: tool_call(
        run, "genomics-analyst", "Write", "w1", {"file_path": "results/tables/l2g.csv"},
        action=lambda: write(run, TABLE, "gene,l2g\nEGFR,0.82\n")))
    run.trace("agent_start", agent="single-cell-analyst", depth=1)  # killed mid-delegation: no agent_end
    with open(run.dir / "logs" / "trace.jsonl", "a") as f:
        f.write('{"type": "tool_start", "agent": "single-cell-an')  # partial line from the crash
    # no finish_turn, no close
    paths = write_reports(run.dir)
    readme = paths["readme"].read_text()
    assert "Run in progress" in readme
    assert "malformed or partial line" in readme
    assert "Turn 1" not in readme or "no turn record" in readme
    assert "Interrupted or failed turns" in readme and "unfinished (no turn record)" in readme
    page = paths["html"].read_text()
    assert "unreadable_trace_lines" in page
    assert "Dashed bars never recorded an end" in page and 'class="bar open"' in page
    assert "genomics-analyst" in page and TABLE in page
    # the MANIFEST now records the reports' hashes, so verify does not see them as tampered
    v = verify_run(run.dir)
    assert "README.md" not in v["integrity"]["harness_changed"]
    assert v["status"] != "COMPLETE"


def test_html_escapes_model_text(tmp_path):
    run = Run(tmp_path / "runs", run_id="XSS")
    run.trace("turn_start", turn=1, prompt=f"Is {XSS} a target?")
    dispatch(run, "genomics-analyst", "t1", lambda: tool_call(
        run, "genomics-analyst", "Write", "w1", {"file_path": "results/tables/l2g.csv"},
        action=lambda: write(run, TABLE, "x\n")), prompt=f"Investigate {XSS}")
    tool_call(run, "genomics-analyst", "mcp__x__q", "bad", {"q": XSS}, is_error=True, output=f"Error: {XSS}")
    assert run.record_claims([{"id": "C1", "text": f"Claim {XSS}", "evidence": [
        {"kind": "table", "path": TABLE, "note": XSS}, {"kind": "citation", "url": "https://example.org/a\"b"}]}])["ok"]
    run.trace("turn_end", turn=1)
    run.finish_turn({"turn": 1, "prompt": f"Is {XSS} a target?", "response": f"Answer {XSS} [[claim:C1]]",
                     "status": "completed", "agents": ["genomics-analyst"]})
    run.close()
    page = (run.dir / "audit.html").read_text()
    assert "<script>alert" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert page.count("<script>") == 1  # only the pagination script, pinned by hash in the CSP
    assert f'content="{html.escape(AUDIT_CSP)}"' in page and f"script-src '{AUDIT_SCRIPT_HASH}'" in AUDIT_CSP
    digest = base64.b64encode(hashlib.sha256(AUDIT_SCRIPT.encode()).digest()).decode()
    assert AUDIT_SCRIPT_HASH == f"sha256-{digest}"
    assert f"<script>{AUDIT_SCRIPT}</script>" in page
    readme = (run.dir / "README.md").read_text()
    outside_code = re.sub(r"(`{3,})[^\n]*\n.*?\n\1", "", readme, flags=re.S)  # fenced blocks are literal
    assert "<script>" not in outside_code and "Investigate <script>" in readme


def test_tool_table_lists_every_call_and_highlights_errors(tmp_path):
    run = Run(tmp_path / "runs", run_id="MANY")
    run.trace("turn_start", turn=1, prompt="q")
    n = TOOL_PAGE_SIZE * 2 + 17
    for i in range(n):
        tool_call(run, "cso", "Read", f"r{i}", {"file_path": "x"}, is_error=(i % 50 == 0), output=f"Error {i}")
    data = collect(run.dir, verify="off")
    page = render_audit_html(data)
    body = page.split('id="tool-table"', 1)[1].split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    assert body.count("<tr") == n                        # paginated client-side, never truncated
    assert body.count('data-err="1"') == len(range(0, n, 50))
    assert 'id="tool-pager" hidden' in page and f"{n} tool call(s), all listed" in page
    assert "Error 0" in body


def test_schema1_manifest_and_missing_records(tmp_path):
    d = tmp_path / "runs" / "OLD"
    (d / "work" / "genomics-analyst").mkdir(parents=True)
    (d / "work" / "genomics-analyst" / "a.csv").write_text("x\n")
    (d / "MANIFEST.json").write_text(json.dumps({"run_id": "OLD", "started": "2025-01-01T00:00:00+00:00",
                                                 "artifacts": {"work/genomics-analyst/a.csv": "ab" * 32}}))
    data = collect(d)
    assert data["n_artifacts"] == 1 and data["by_agent"]["genomics-analyst"][0]["sha256"] == "ab" * 32
    readme = render_readme(data)
    assert "work/genomics-analyst/a.csv" in readme and "No execution trace" in readme
    render_audit_html(data)
    # no MANIFEST at all: listed from disk, without hashes
    (d / "MANIFEST.json").unlink()
    data = collect(d)
    assert data["status"] == "unrecorded" and data["artifacts"][0]["sha256"] is None
    assert "MANIFEST.json is missing or unreadable" in data["read_errors"]
    assert "not hashed" in render_readme(data)


def test_fast_report_verification_never_weakens_vbt_verify(tmp_path):
    run = Run(tmp_path / "runs", run_id="FAST")
    run.trace("turn_start", turn=1, prompt="q")
    tool_call(run, "genomics-analyst", "Write", "w1", {"file_path": "results/tables/l2g.csv"},
              action=lambda: write(run, TABLE, "gene,l2g\nEGFR,0.82\n"))
    run.finish_turn({"turn": 1, "prompt": "q", "response": "a", "status": "completed"})
    run.close()
    p = run.dir / TABLE
    st = p.stat()
    p.write_text("gene,l2g\nEGFR,0.99\n")              # same size ...
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))  # ... and mtime
    assert collect(run.dir)["verify"]["integrity"]["changed"] == []          # report: stat-trusting
    assert collect(run.dir, verify="full")["verify"]["integrity"]["changed"] == [TABLE]
    assert verify_run(run.dir)["integrity"]["changed"] == [TABLE]            # vbt verify always re-hashes
    assert collect(run.dir, verify="off")["verify"] is None
