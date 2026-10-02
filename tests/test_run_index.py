"""Run index (INDEX.md/INDEX.json, `vbt list`) and run resolution by path, id or prefix."""

import argparse
import json

import pytest

from vbt.audit.cli import add_audit_parsers
from vbt.audit.index import AmbiguousRunError, RunNotFound, resolve_run, scan_runs, update_index
from vbt.session import Run


def finished_run(root, run_id, prompt="Is EGFR a target?"):
    run = Run(root, run_id=run_id)
    run.trace("turn_start", turn=1, prompt=prompt)
    run.trace("turn_end", turn=1, status="completed")
    run.finish_turn({"turn": 1, "prompt": prompt, "response": "Which subtype?", "status": "completed"})
    run.close()
    return run


def cli(config, *argv):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    add_audit_parsers(sub)
    args = p.parse_args(list(argv))
    return args.handler(args, config)


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    finished_run(root, "20260101_100000_aaaa1111", "first question")
    finished_run(root, "20260101_100500_aaaa2222", "second question")
    finished_run(root, "20260102_090000_bbbb3333", "third question")
    (root / ".downloads").mkdir()
    (root / ".downloads" / "MANIFEST.json").write_text("{}")  # a dot-dir is never a run
    (root / "notes").mkdir()                                   # nor is a directory without records
    return root


def test_scan_ignores_downloads_and_non_runs(runs):
    rows = scan_runs(runs)
    # newest first (same creation second here, so by directory name)
    assert [r["run_id"] for r in rows] == ["20260102_090000_bbbb3333", "20260101_100500_aaaa2222",
                                           "20260101_100000_aaaa1111"]
    r = next(r for r in rows if r["run_id"].endswith("bbbb3333"))
    assert r["query"] == "third question" and r["status"] == "completed" and r["n_turns"] == 1
    assert r["has_audit"] and r["has_readme"] and r["cost_usd"] == 0.0
    for key in ("created", "agents", "n_artifacts", "n_claims"):
        assert key in r


def test_close_updates_the_index(runs):
    doc = json.loads((runs / "INDEX.json").read_text())  # written by Run.close()
    assert doc["n_runs"] == 3
    md = (runs / "INDEX.md").read_text()
    assert "[`20260102_090000_bbbb3333`](20260102_090000_bbbb3333/README.md)" in md
    assert "(20260102_090000_bbbb3333/audit.html)" in md
    (runs / "20260101_100000_aaaa1111" / "MANIFEST.json").write_text("not json")
    rows = update_index(runs)
    assert len(rows) == 3  # an unreadable MANIFEST still lists the run


def test_resolve_by_path_id_prefix_and_suffix(runs, monkeypatch, tmp_path):
    target = runs / "20260102_090000_bbbb3333"
    assert resolve_run("20260102_090000_bbbb3333", runs) == target.resolve()
    assert resolve_run("20260102", runs) == target.resolve()          # unique prefix
    assert resolve_run("bbbb", runs) == target.resolve()              # hex suffix
    assert resolve_run(str(target), runs) == target.resolve()         # absolute path
    monkeypatch.chdir(tmp_path)
    assert resolve_run("runs/20260102_090000_bbbb3333", runs) == target.resolve()
    assert resolve_run("latest", runs) == target.resolve()
    with pytest.raises(AmbiguousRunError) as exc:
        resolve_run("20260101", runs)
    assert exc.value.candidates == ["20260101_100000_aaaa1111", "20260101_100500_aaaa2222"]
    assert "20260101_100000_aaaa1111" in str(exc.value)
    with pytest.raises(AmbiguousRunError):
        resolve_run("aaaa", runs)
    with pytest.raises(RunNotFound):
        resolve_run("zzzz", runs)
    with pytest.raises(RunNotFound):
        resolve_run(".downloads", runs)
    with pytest.raises(RunNotFound):
        resolve_run("../etc", runs)


def test_cli_list_index_verify_by_prefix(runs, config, capsys):
    config["paths"]["runs_dir"] = str(runs)
    assert cli(config, "list", "--json", "--limit", "2") == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2
    assert cli(config, "list") == 0
    out = capsys.readouterr().out
    assert "third question" in out and "completed" in out
    (runs / "INDEX.md").unlink()
    assert cli(config, "index") == 0 and (runs / "INDEX.md").is_file()
    capsys.readouterr()
    assert cli(config, "verify", "bbbb") == 0
    out = capsys.readouterr().out
    assert "COMPLETE" in out and out.rstrip().endswith("audit.html")
    assert cli(config, "verify", "bbbb", "--json") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"
    assert cli(config, "verify", "20260101") == 2
    assert "matches 2 runs" in capsys.readouterr().err
    assert cli(config, "verify", "nope") == 2
    # an interrupted run exits non-zero
    live = Run(runs, run_id="20260103_000000_cccc4444")
    live.finish_turn({"turn": 1, "prompt": "q", "response": "partial", "status": "interrupted"})
    live.close()
    assert cli(config, "verify", "cccc") == 1
    assert cli(config, "show", "cccc") == 0
    assert "interrupted" in capsys.readouterr().out
