"""``vbt datasource`` / ``vbt ds`` (§16): parser wiring, and lint, describe, explain, resolve and check
on the shipped configs (resolve and check against the OT fixture)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dl_upstream import needs_arrow
from vbt import cli
from vbt.datalayer import cli as ds


def _parse(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def test_parser_wiring() -> None:
    for name in ("datasource", "ds"):
        assert _parse(name, "list").handler is ds.cmd_list
    args = _parse("ds", "check", "--depth", "shallow", "--table", "open_targets.target", "--table",
                  "open_targets.disease", "--column", "open_targets.target.id", "--tool", "target.get_target_info",
                  "--json")
    assert args.handler is ds.cmd_check and args.depth == "shallow" and args.json
    assert args.table == ["open_targets.target", "open_targets.disease"] and args.tool == "target.get_target_info"
    assert _parse("ds", "lint", "--strict").strict
    assert _parse("ds", "describe", "open_targets.target").target == "open_targets.target"
    args = _parse("ds", "resolve", "ensembl_gene", "PCSK9", "TP53")
    assert args.handler is ds.cmd_resolve and args.values == ["PCSK9", "TP53"]
    args = _parse("ds", "explain", "--all")
    assert args.handler is ds.cmd_explain and args.all
    assert _parse("ds", "fingerprint", "--write").write
    args = _parse("ds", "index", "build", "--id-type", "ensembl_gene", "--access-paths")
    assert args.handler is ds.cmd_index_build and args.id_type == ["ensembl_gene"] and args.access_paths
    assert _parse("ds", "estimate", "--tool", "drug.search_known_drugs").handler is ds.cmd_estimate
    args = _parse("ds", "retro-audit", "latest", "--json")
    assert args.handler is ds.cmd_retro_audit and args.run == "latest"
    # `vbt data` stays the Zenodo fetcher
    assert _parse("data", "zenodo", "list").cmd == "data"
    with pytest.raises(SystemExit):
        _parse("ds")


def test_lint_is_clean_on_the_shipped_configs(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--profile", "mock", "ds", "lint", "-q"]) == 0
    out = capsys.readouterr().out
    assert " 0 error(s)" in out and "bound tools" in out


def test_describe(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--profile", "mock", "ds", "describe", "open_targets.known_drug"]) == 0
    out = capsys.readouterr().out
    assert "open_targets.known_drug  [fact]" in out
    assert "key: drugId, targetId, diseaseId, phase, status" in out
    assert "phase: measure" in out and "coverage:" in out
    assert cli.main(["--profile", "mock", "ds", "describe", "tahoe_100m"]) == 0
    out = capsys.readouterr().out
    assert "tahoe_100m.de_permissive" in out and "id_type tahoe_drug" in out
    assert cli.main(["--profile", "mock", "ds", "describe", "no_such_source"]) == 2


def test_explain(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--profile", "mock", "ds", "explain", "drug.search_known_drugs"]) == 0
    out = capsys.readouterr().out
    assert "serve: pass" in out and "bound table: open_targets.known_drug" in out
    assert "arg target_id:" in out and "accepts ensembl_gene" in out
    assert "derived text:" in out and "derived schema:" in out
    assert "defect OT-DRUG-003" in out
    assert cli.main(["--profile", "mock", "ds", "explain", "mcp__association__search_literature"]) == 0
    assert "serve: derived" in capsys.readouterr().out
    assert cli.main(["--profile", "mock", "ds", "explain", "clinicaltrials.clear_trial_cache"]) == 0
    assert "serve: block" in capsys.readouterr().out
    assert cli.main(["--profile", "mock", "ds", "explain", "--all"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n  status: ") >= 100
    assert cli.main(["--profile", "mock", "ds", "explain", "not-a-tool"]) == 2


@pytest.fixture
def fixture_env(ot_root: Path, tahoe_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(ot_root))
    monkeypatch.setenv("TAHOE_DATA_PATH", str(tahoe_root))
    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path))
    return tmp_path / ".vbt-datalayer"


@needs_arrow
def test_index_build_resolve_and_check(fixture_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--profile", "mock", "ds", "index", "build", "--id-type", "ensembl_gene"]) == 0
    out = capsys.readouterr().out
    assert "built open_targets:ensembl_gene" in out
    assert list(fixture_env.glob("open_targets/*/index/ensembl_gene.tsv.gz"))

    rc = cli.main(["--profile", "mock", "ds", "resolve", "ensembl_gene", "PCSK9", "ENSG00000169174.12", "narc1",
                   "--json"])
    rows = {r["value"]: r for r in json.loads(capsys.readouterr().out)}
    assert rc == 0
    assert {r["canonical"] for r in rows.values()} == {"ENSG00000169174"}
    assert rows["PCSK9"]["rule"].startswith("label_exact")
    assert rows["ENSG00000169174.12"]["rule"].startswith("normalized")
    assert cli.main(["--profile", "mock", "ds", "resolve", "ensembl_gene", "ENSG00000999999"]) == 1
    assert "not_found" in capsys.readouterr().out

    rc = cli.main(["--profile", "mock", "ds", "check", "--tool", "drug.search_known_drugs", "--json"])
    body = json.loads(capsys.readouterr().out)
    assert rc == 0 and body["tool"]["ready"] is True, body["tool"]
    assert body["tables"]["open_targets.known_drug"]["status"] == "ready"
    assert body["indexes"]["open_targets:ensembl_gene"]["status"] == "ready"
    assert cli.main(["--profile", "mock", "ds", "check", "--table", "open_targets.known_drug"]) == 0
    assert "open_targets.known_drug: ready" in capsys.readouterr().out


def test_unknown_names_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    """A typo is an error (rc 2), never a check, explanation or estimate of nothing (R7)."""
    assert cli.main(["--profile", "mock", "ds", "check", "--tool", "target.nosuch"]) == 2
    assert "no binding" in capsys.readouterr().err
    assert cli.main(["--profile", "mock", "ds", "check", "--table", "nosuch.table"]) == 2
    assert "unknown table" in capsys.readouterr().err
    assert cli.main(["--profile", "mock", "ds", "explain", "nosuch.tool"]) == 2
    assert "unknown server" in capsys.readouterr().err
    assert cli.main(["--profile", "mock", "ds", "estimate", "--tool", "target.nosuch"]) == 2
    assert cli.main(["--profile", "mock", "ds", "estimate", "--table", "open_targets.nosuch"]) == 2


@needs_arrow
def test_estimate_without_data_is_not_admissible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    """A table the tool loads that cannot be measured gives no verdict and rc 1 (R7)."""
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path))
    assert cli.main(["--profile", "mock", "ds", "estimate", "--tool", "target.get_target_info"]) == 1
    out = capsys.readouterr().out
    assert "not estimable" in out and "admissible" not in out


@needs_arrow
def test_check_and_resolve_report_missing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    """check exits 1 when a requested table is not ready; resolve says no local index exists (R7)."""
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path))
    assert cli.main(["--profile", "mock", "ds", "check", "--table", "open_targets.known_drug"]) == 1
    capsys.readouterr()
    cli.main(["--profile", "mock", "ds", "resolve", "ensembl_gene", "ENSG00000141510"])
    err = capsys.readouterr().err
    assert "no local index" in err and "vbt ds index build" in err
