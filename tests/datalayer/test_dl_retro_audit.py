"""``vbt ds retro-audit`` (§16, §22): a recorded run's data calls re-classified offline.

The run is synthetic: a trace with an empty-lookup *success* (upstream's ``Target X not found`` with
``is_error: false``) that a filed claim cites, a quarantined tool, an unknown identifier that the
resolver sidecar proves absent, an unbound server's structural empty, a genuine row and an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vbt import cli
from vbt.config import load_config
from vbt.datalayer.retro_audit import format_report, retro_audit

PCSK9 = "ENSG00000169174"
UNKNOWN = "ENSG00000999999"

CALLS: list[tuple[str, str, dict[str, Any], str, bool]] = [
    # (tool_use_id, tool, input, output, is_error)
    ("tu_nf", "mcp__target__get_target_info", {"target_id": UNKNOWN}, f"Target {UNKNOWN} not found", False),
    ("tu_q", "mcp__association__search_literature", {"keyword_id": PCSK9},
     json.dumps({"publications": [], "count": 0}), False),
    ("tu_abs", "mcp__drug__search_known_drugs", {"target_id": UNKNOWN},
     json.dumps({"drugs": [], "num_drugs": 0}), False),
    ("tu_gen", "mcp__thirdparty__lookup", {"q": "x"}, json.dumps({"results": [], "count": 0}), False),
    ("tu_ok", "mcp__target__get_target_info", {"target_id": PCSK9},
     json.dumps({"id": PCSK9, "approvedSymbol": "PCSK9"}), False),
    ("tu_err", "mcp__drug__get_drug_info", {"drug_id": "CHEMBL25"}, "Error: upstream exploded", True),
    ("tu_data", "mcp__data___witness", {"request": {}}, "{}", False),
    ("tu_bash", "Bash", {"command": "ls"}, "a\nb", False),
]


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "20260101_000000_retroaud"
    (run / "logs").mkdir(parents=True)
    (run / "evidence").mkdir()
    events = []
    for tuid, tool, args, output, is_error in CALLS:
        events.append({"type": "tool_start", "agent": "target-biologist", "tool": tool, "tool_use_id": tuid,
                       "input": args})
        events.append({"type": "tool_end", "agent": "target-biologist", "tool": tool, "tool_use_id": tuid,
                       "is_error": is_error, "output": output, "output_chars": len(output)})
    (run / "logs" / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    claims = [
        {"id": "C1", "text": f"{UNKNOWN} is not a known target", "confidence": "strong",
         "evidence": [{"kind": "tool_call", "tool_use_id": "tu_nf"}]},
        {"id": "C2", "text": "PCSK9 has no known drugs", "confidence": "moderate",
         "evidence": [{"kind": "tool_call", "tool_use_id": "tu_q"}, {"kind": "tool_call", "tool_use_id": "tu_ok"}]},
    ]
    (run / "evidence" / "claims.json").write_text(json.dumps({"stats": {}, "claims": claims}))
    return run


def _config(tmp_path: Path) -> dict[str, Any]:
    return load_config(["mock"], overrides={"paths": {"runs_dir": str(tmp_path / "runs")},
                                            "data": {"cache_dir": str(tmp_path / "dl")}})


def _sidecar(config: dict[str, Any]) -> None:
    """A tiny ensembl_gene resolver sidecar (what `vbt ds index build` writes)."""
    from vbt.datalayer.plugins.registry import discover
    from vbt.datalayer.resolve import Entry, IndexStore
    from vbt.datalayer.settings import DataSettings

    plugin = discover(entry_points=False).get("identifier", "ensembl_gene")
    rows = [Entry(plugin.label_key(PCSK9), PCSK9, "exact", "PCSK9"),
            Entry(plugin.label_key("PCSK9"), PCSK9, "label_exact:approvedSymbol", "PCSK9")]
    IndexStore(DataSettings.from_config(config).cache_dir).write_sidecar("open_targets", "fp1:test", "ensembl_gene",
                                                                          rows)


def _by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["tool_use_id"]: c for c in report["calls"]}


def test_cited_empty_lookup_success_is_reported(tmp_path: Path) -> None:
    run = _run(tmp_path)
    config = _config(tmp_path)
    report = retro_audit(run, config)
    calls = _by_id(report)
    assert set(calls) == {"tu_nf", "tu_q", "tu_abs", "tu_gen", "tu_ok", "tu_err"}   # data verbs and Bash skipped
    nf = calls["tu_nf"]
    assert nf["outcome"] == "not_found" and nf["changed"] and nf["cited_by"] == ["C1"]
    assert calls["tu_q"]["outcome"] == "quarantined" and calls["tu_q"]["cited_by"] == ["C2"]
    assert calls["tu_gen"]["outcome"] == "empty_unverified"
    assert calls["tu_ok"]["outcome"] == "ok" and not calls["tu_ok"]["changed"]
    assert calls["tu_err"]["outcome"] == "source_error" and not calls["tu_err"]["changed"]
    assert report["cited"] == {"not_found": 1, "quarantined": 1}
    assert report["claims_affected"] == ["C1", "C2"]
    assert report["n_data_calls"] == 6 and report["n_calls"] == len(CALLS)
    text = "\n".join(format_report(report))
    assert "tu_nf mcp__target__get_target_info: not_found" in text and "[cited by C1]" in text
    assert "Claims citing them: C1, C2" in text


def test_resolution_from_sidecars(tmp_path: Path) -> None:
    run = _run(tmp_path)
    config = _config(tmp_path)
    before = _by_id(retro_audit(run, config))
    assert before["tu_abs"]["outcome"] != "not_found"      # no sidecar: existence stays unknown
    _sidecar(config)
    after = _by_id(retro_audit(run, config))
    absent = after["tu_abs"]
    assert absent["outcome"] == "not_found" and absent["changed"], absent
    assert "not in its universe" in absent["reason"]
    assert after["tu_ok"]["outcome"] == "ok" and after["tu_ok"]["resolutions"]


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run(tmp_path)
    rc = cli.main(["--profile", "mock", "--runs-dir", str(tmp_path / "runs"), "ds", "retro-audit", run.name, "--json"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["cited"].get("not_found") == 1
    assert cli.main(["--profile", "mock", "--runs-dir", str(tmp_path / "runs"), "ds", "retro-audit", "latest"]) == 0
    assert "retro-audit of" in capsys.readouterr().out
    assert cli.main(["--profile", "mock", "--runs-dir", str(tmp_path / "runs"), "ds", "retro-audit", "nope"]) == 2
