"""``vbt data``: the acquisition commands beside the Zenodo and Open Targets ones, on the shipped configuration.

Offline: the parsers of ``acquire``, ``status``, ``ot`` and ``zenodo`` coexist; ``vbt data acquire --all --plan
--offline`` plans every shipped download source from its declared sizes alone (no request is made); ``--for-tools``
and ``--for-agents`` resolve through the shipped overlays and roster; ``vbt data status --json`` lists every declared
table; ``vbt data zenodo presets`` still works. The engine itself is tested in
``tests/datalayer/test_dl_acquisition.py``.
"""

from __future__ import annotations

import json
import socket
from typing import Any

import pytest

from vbt import cli


def _parse(*argv: str) -> Any:
    return cli.build_parser().parse_args(list(argv))


def test_the_data_subcommands_coexist():
    a = _parse("data", "acquire", "open_targets.target", "--for-tools", "target.*", "--plan", "--max-gb", "2")
    assert (a.cmd, a.data_source, a.targets, a.for_tools, a.plan, a.max_gb) == (
        "data", "acquire", ["open_targets.target"], ["target.*"], True, 2.0)
    s = _parse("data", "status", "depmap", "--check", "--json")
    assert (s.data_source, s.sources, s.check, s.json) == ("status", ["depmap"], True, True)
    assert _parse("data", "ot", "fetch", "so", "--dry-run").ot_action == "fetch"
    assert _parse("data", "zenodo", "list").zenodo_action == "list"


@pytest.fixture
def no_network(monkeypatch):
    """Any connection attempt fails the test: the offline plan must not touch the network."""

    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("the offline plan opened a connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_an_offline_plan_of_every_shipped_source(tmp_path, capsys, no_network):
    rc = cli.main(["--profile", "mock", "data", "acquire", "--all", "--plan", "--offline", "--json",
                   "--dest", str(tmp_path)])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    by = {s["source"]: s for s in body["sources"]}
    assert sorted(by) == ["cell_ontology", "depmap", "gene_ontology", "msigdb", "open_targets", "tahoe_100m",
                          "zenodo_vbt"]
    assert (by["open_targets"]["files"], by["open_targets"]["bytes_remaining"]) == (3508, 31_131_380_890)
    assert by["tahoe_100m"]["bytes_remaining"] == 88_859_715_303 + 1_451_950
    assert by["tahoe_100m"]["prepare"] == {"prepare_tahoe": "pending"}
    assert by["depmap"]["bytes_remaining"] == 428_678_699 + 421_115_594 + 645_696 + 20_795
    assert all(s["offline"] and s["licence"] for s in body["sources"])
    assert by["open_targets"]["home"] == str(tmp_path / "open_targets" / "25.09")
    assert not any(tmp_path.iterdir())                 # a plan writes nothing


def test_tools_and_agents_resolve_offline(tmp_path, capsys, no_network):
    rc = cli.main(["--profile", "mock", "data", "acquire", "--for-tools", "functional_genomics.query_drug_perturbation",
                   "--for-agents", "genomics-analyst", "--plan", "--offline", "--json", "--dest", str(tmp_path)])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    want = {s["source"]: set(s["requested"]) for s in body["sources"]}
    assert {"de_permissive", "drug_metadata"} <= want["tahoe_100m"]
    assert {"variant", "credible_set", "l2g_prediction", "target"} <= want["open_targets"]
    assert "mcp__functional_genomics__query_drug_perturbation" in body["why"]["tahoe_100m.de_permissive"]


def test_status_lists_every_declared_table(tmp_path, capsys, monkeypatch):
    for var in ("OPEN_TARGETS_DATA_PATH", "TAHOE_DATA_PATH"):
        monkeypatch.delenv(var, raising=False)
    profile = tmp_path / "p.yaml"
    profile.write_text(f"data:\n  cache_dir: {tmp_path / 'cache'}\n  acquisition:\n    root: {tmp_path / 'acq'}\n")
    assert cli.main(["--profile", "mock", "--profile", str(profile), "data", "status", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    by = {s["source"]: s for s in body}
    assert by["cellxgene_census"]["kind"] == "remote"
    assert all(t["present"] == "remote" for t in by["clinicaltrials_gov"]["tables"])
    ot = {t["table"]: t for t in by["open_targets"]["tables"]}
    assert len(ot) >= 38 and ot["open_targets.target"]["present"] == "absent" and ot["open_targets.target"]["tools"]
    assert cli.main(["--profile", "mock", "data", "status", "no_such_source"]) == 2


def test_zenodo_presets_still_list(capsys):
    assert cli.main(["--profile", "mock", "data", "zenodo", "presets"]) == 0
    assert "case1" in capsys.readouterr().out
