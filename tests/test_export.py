"""Run export: the zip bundle (archive root, --no-data, size cap, cache), chat Markdown and file listing."""

import argparse
import json
import os
import zipfile

from vbt.audit.cli import add_audit_parsers
from vbt.audit.export import (
    EXCLUDED_RECORD,
    excluded_members,
    export_chat_markdown,
    export_run,
    list_session_files,
)
from vbt.session import Run
from vbt.verify import verify_run

TABLE = "work/genomics-analyst/results/tables/l2g.csv"
FIG = "work/genomics-analyst/results/figures/manhattan.png"
H5AD = "work/_mcp/data/processed/census.h5ad"


def make_run(tmp_path, run_id="EXP1"):
    run = Run(tmp_path / "runs", run_id=run_id)
    run.trace("turn_start", turn=1, prompt="Is EGFR a target?")
    run.trace("agent_start", agent="genomics-analyst", depth=1)
    for rel, data in ((TABLE, b"gene,l2g\nEGFR,0.82\n"), (FIG, b"\x89PNG fake"), (H5AD, os.urandom(300_000)),
                      ("work/genomics-analyst/code/scripts/run.py", b"print(1)\n")):
        tuid = "w" + str(abs(hash(rel)))
        run.trace("tool_start", agent="genomics-analyst", tool="Write", tool_use_id=tuid, input={"file_path": rel})
        p = run.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        run.trace("tool_end", agent="genomics-analyst", tool="Write", tool_use_id=tuid, is_error=False, output="ok")
    run.trace("agent_end", agent="genomics-analyst", depth=1, stop="end_turn")
    assert run.record_claims([{"id": "C1", "text": "EGFR L2G 0.82",
                               "evidence": [{"kind": "table", "path": TABLE}]}])["ok"]
    run.trace("turn_end", turn=1)
    run.finish_turn({"turn": 1, "prompt": "Is EGFR a target?", "response": "Yes [[claim:C1]], and [[claim:C9]].",
                     "status": "completed", "agents": ["genomics-analyst"], "cost_usd": 0.5})
    run.close()
    (run.dir / ".env").write_text("ANTHROPIC_API_KEY=sk-secret\n")
    return run


def names(z):
    return zipfile.ZipFile(z).namelist()


def test_zip_has_run_id_root_and_reports(tmp_path):
    run = make_run(tmp_path)
    (run.dir / "README.md").unlink()  # missing reports are rendered before zipping
    z = export_run(run.dir)
    assert z.parent.name == ".downloads" and z.parent.parent == run.dir.parent
    n = names(z)
    assert "EXP1/MANIFEST.json" in n and "EXP1/README.md" in n and "EXP1/audit.html" in n
    assert f"EXP1/{TABLE}" in n and f"EXP1/{H5AD}" in n
    assert all(x.startswith("EXP1/") for x in n)
    assert not any(x.endswith(".env") or "/." in x for x in n)  # no .env, .audit.lock, dot-dirs
    assert excluded_members(z) == []
    # the extracted bundle verifies like the original
    out = tmp_path / "extracted"
    zipfile.ZipFile(z).extractall(out)
    assert verify_run(out / "EXP1")["integrity"]["status"] == "passed"


def test_no_data_excludes_raw_data_and_records_it(tmp_path):
    run = make_run(tmp_path)
    z = export_run(run.dir, no_data=True)
    assert z.name == "EXP1.nodata.zip"
    n = names(z)
    assert f"EXP1/{H5AD}" not in n and f"EXP1/{TABLE}" in n and f"EXP1/{FIG}" in n
    assert f"EXP1/{EXCLUDED_RECORD}" in n
    ex = excluded_members(z)
    assert [e["path"] for e in ex] == [H5AD] and ex[0]["sha256"] and "raw data" in ex[0]["reason"]
    # a size cap leaves out big analysis files but never harness records
    capped = export_run(run.dir, tmp_path / "small.zip", max_file_mb=0.1)
    n2 = names(capped)
    assert f"EXP1/{H5AD}" not in n2 and "EXP1/MANIFEST.json" in n2 and "EXP1/logs/trace.jsonl" in n2


def test_cache_is_reused_until_the_run_changes(tmp_path):
    run = make_run(tmp_path)
    z1 = export_run(run.dir)
    m1 = z1.stat().st_mtime_ns
    z2 = export_run(run.dir)
    assert z2 == z1 and z2.stat().st_mtime_ns == m1
    p = run.dir / TABLE
    p.write_text("gene,l2g\nEGFR,0.9\n")
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 5_000_000_000))
    z3 = export_run(run.dir)
    assert z3 == z1 and z3.stat().st_mtime_ns != m1
    with zipfile.ZipFile(z3) as zf:
        assert zf.read(f"EXP1/{TABLE}") == b"gene,l2g\nEGFR,0.9\n"
    out = export_run(run.dir, tmp_path / "copy.zip")  # an explicit output reuses the valid cache
    assert names(out) == names(z3)
    assert not list((run.dir.parent / ".downloads").glob("*.tmp"))


def test_chat_markdown_and_file_listing(tmp_path):
    run = make_run(tmp_path)
    md = export_chat_markdown(run.dir)
    assert "## Turn 1 — User" in md and "Is EGFR a target?" in md
    assert "Yes [1], and [C9?]." in md and "**C1**: EGFR L2G 0.82" in md and "dangling reference" in md
    listing = list_session_files(run.dir)
    assert listing["counts"]["figures"] == 1 and listing["counts"]["tables"] == 1
    assert listing["counts"]["code"] == 1 and listing["counts"]["data"] == 1
    fig = listing["figures"]["genomics-analyst"][0]
    assert fig["path"] == FIG and fig["image"]
    assert listing["tables"]["genomics-analyst"][0]["cited_by"] == ["C1"]


def test_cli_export_and_show(tmp_path, config, capsys):
    run = make_run(tmp_path)
    config["paths"]["runs_dir"] = str(run.dir.parent)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    add_audit_parsers(sub)

    def cli(*argv):
        args = p.parse_args(list(argv))
        return args.handler(args, config)

    assert cli("export", "EXP", "--no-data") == 0
    out = capsys.readouterr().out
    assert "EXP1.nodata.zip" in out and "Left out 1 file" in out and H5AD in out
    assert cli("export", "EXP1", "--chat") == 0
    assert "## Turn 1 — CSO" in capsys.readouterr().out
    target = tmp_path / "chat.md"
    assert cli("export", "EXP1", "--chat", "-o", str(target)) == 0 and target.is_file()
    assert cli("show", "EXP1", "--files") == 0
    out = capsys.readouterr().out
    assert "Figures (1)" in out and FIG in out and "cited by C1" in out
    assert cli("show", "EXP1", "--files", "--json") == 0
    assert json.loads(capsys.readouterr().out)["files"]["counts"]["tables"] == 1
