"""``vbt validate`` (src/vbt/validate): the report, the independent oracle, the six correctness tests generated from
descriptor roles, the step skips, the CLI and one end-to-end run on the OT-25.09-shaped fixtures through the
unmodified servers (skipped without the upstream checkout).

The real-data run is in the D3 report; ``VBT_DL_REAL_DATA=<dir>`` runs :func:`test_real_data_validation` here."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "datalayer"))

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from vbt.validate import STEPS, Options, run_validate  # noqa: E402
from vbt.validate.cases import Case, complete_cases, judge, judge_off, oracle_queries, plan_cases, table_source  # noqa: E402
from vbt.validate.contained import Outcome, run_oracle  # noqa: E402
from vbt.validate.report import FAIL, PASS, SKIPPED, StepResult, ValidationReport, percentile  # noqa: E402

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the reaper is Linux-only")
DEAD = {"VBT_CBIOPORTAL_BASE": "http://127.0.0.1:9/cbioportal/api", "VBT_CTGOV_BASE": "http://127.0.0.1:9/ctgov/api/v2",
        "VBT_EUTILS_BASE": "http://127.0.0.1:9/eutils"}


def _config(**overrides):
    from vbt.config import load_config

    return load_config(["mock"], overrides=overrides or None)


# --------------------------------------------------------------------------- the report


def test_percentile_and_the_report(tmp_path):
    assert percentile([], 50) is None and percentile([3.0], 95) == 3.0
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile(range(1, 101), 95) == pytest.approx(95.05)
    rep = ValidationReport(started="t0", host={"hostname": "h", "platform": "p", "cpus": 4, "plan_mb": 13680})
    rep.steps += [StepResult("host", "Host", PASS, "plans with 13,680 MB", rows=[{"setting": "a|b", "value": 1.5}]),
                  StepResult.skipped("model", "Model server", "no model server answers at http://x")]
    assert rep.ok and rep.counts() == {"pass": 1, "skipped": 1}
    md, js = rep.write(tmp_path)
    text = md.read_text()
    assert "# vbt validate: PASS" in text and "| Host | PASS |" in text and "a\\|b" in text
    assert "Skipped: no model server answers" in text
    data = json.loads(js.read_text())
    assert data["schema"] == "vbt.validate/1" and data["ok"] is True and data["steps"][1]["status"] == SKIPPED
    rep.steps.append(StepResult("check", "Check", FAIL, "1 present table not ready"))
    assert not rep.ok and "# vbt validate: FAIL" in rep.to_markdown()


# --------------------------------------------------------------------------- the oracle


@linux_only
def test_the_oracle_answers_from_the_files_under_the_reaper(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    pq.write_table(pa.table({"id": ["A1", "A2", "A2"], "score": [0.5, None, 0.9], "kind": ["x", "y", "x"]}),
                   d / "part-0.parquet")
    pq.write_table(pa.table({"id": ["A3", "A2"], "score": [0.7, 0.9], "kind": ["y", "z"]}), d / "part-1.parquet")
    (tmp_path / "t.csv").write_text("id,score\nB1,1\nB2,\nB1,3\n")
    src = {"source": str(d), "format": "parquet"}
    queries = [{"id": "s", "kind": "sample", "column": "id", "n": 2, **src},
               {"id": "c", "kind": "count", "column": "id", "values": ["A2", "A9"], **src},
               {"id": "k", "kind": "topk", "filter": {"column": "id", "value": "A2"},
                "order": [{"column": "score", "direction": "desc", "nulls": "last"}], "k": 2, **src},
               {"id": "n", "kind": "nulls", "column": "score", **src},
               {"id": "d", "kind": "distinct", "column": "kind", **src},
               {"id": "l", "kind": "load", **src},
               {"id": "csv", "kind": "count", "column": "id", "values": ["B1"], "source": str(tmp_path / "t.csv"),
                "format": "csv"},
               {"id": "bad", "kind": "count", "column": "nope", "values": ["x"], **src}]
    answers, status = run_oracle(_config(), queries)
    assert len(answers["s"]["values"]) == 2 and set(answers["s"]["values"]) <= {"A1", "A2", "A3"}
    assert answers["c"] == {"counts": {"A2": 3, "A9": 0}, "rows": 5}
    assert answers["k"] == {"top": [0.9, 0.9], "total": 3, "nulls": 1}
    assert answers["n"]["nulls"] == 1 and answers["n"]["non_null"] == 4 and answers["n"]["median"] == 0.8
    assert answers["d"] == {"values": ["x", "y", "z"], "complete": True}
    assert answers["l"]["rows"] == 5 and answers["l"]["peak_mb"] > 0
    assert answers["csv"]["counts"] == {"B1": 2}
    assert "error" in answers["bad"], "a failed query answers an error; the others still run"
    assert status["containment"] in ("cgroup_v1", "cgroup_v2", "systemd_scope", "watchdog", "rlimit_data")
    assert status["peak_rss_mb"] > 0 and status["rc"] == 0


@linux_only
def test_the_oracle_counts_the_items_of_nested_containers(tmp_path):
    """Tools whose rows are items of a container (``rows_of``) are asked for an identifier whose row holds items,
    and the oracle counts the items, not the parent rows."""
    rows = [{"id": "G1", "probes": [], "hall": {"marks": [{"m": 1}], "attrs": None}},
            {"id": "G2", "probes": [{"p": "a"}, {"p": "b"}], "hall": {"marks": [], "attrs": None}},
            {"id": "G3", "probes": None, "hall": None},
            {"id": "G4", "probes": [{"p": "c"}], "hall": {"marks": [{"m": 2}, {"m": 3}], "attrs": "x"}}]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "t.parquet")
    src = {"source": str(tmp_path / "t.parquet"), "format": "parquet"}
    queries = [{"id": "s", "kind": "sample", "column": "id", "n": 3, "items": "probes[]", **src},
               {"id": "h", "kind": "sample", "column": "id", "n": 3, "items": "hall.marks[]", **src},
               {"id": "a", "kind": "sample", "column": "id", "n": 3, "items": "hall", **src},
               {"id": "c", "kind": "count", "column": "id", "values": ["G1", "G2", "G4"], "items": "hall.marks[]", **src}]
    answers, _ = run_oracle(_config(), queries)
    assert sorted(answers["s"]["values"]) == ["G2", "G4"] and answers["s"]["items"] == {"G2": 2, "G4": 1}
    assert sorted(answers["h"]["values"]) == ["G1", "G4"] and answers["h"]["items"]["G4"] == 2
    assert sorted(answers["a"]["values"]) == ["G1", "G2", "G4"], "a struct holds one item when it is not null"
    assert answers["c"]["counts"] == {"G1": 1, "G2": 1, "G4": 1} and answers["c"]["items"] == {"G1": 1, "G2": 0, "G4": 2}


# --------------------------------------------------------------------------- the cases


@pytest.fixture(scope="module")
def ot_fixture(tmp_path_factory):
    import dl_fixtures

    return dl_fixtures.build_ot_fixture(tmp_path_factory.mktemp("ot") / "25.09")


def _catalog(monkeypatch, ot: Path):
    from vbt.preflight import data_catalog

    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(ot))
    return data_catalog(_config())


def _schemas(catalog, server):
    """Signatures as the server would list them: every bound argument, the identifier one required."""
    out = {}
    for tool in catalog.tools(server):
        c = catalog.contract(server, tool)
        if c.binding is None:
            continue
        ids = list(c.identifier_args)
        out[tool] = {"properties": {a: {} for a in c.binding.args if not c.binding.args[a].gateway_only},
                     "required": ids[:1]}
    return out


@linux_only
def test_cases_come_from_the_roles_and_the_oracle(monkeypatch, ot_fixture):
    import dl_fixtures as F

    settings, catalog, registry = _catalog(monkeypatch, ot_fixture)
    ready = {str(t) for t in catalog.table_refs()}
    plan = plan_cases(catalog, registry, "target", _schemas(catalog, "target"), ready)
    kinds = {c.ct for c in plan.cases}
    assert {"CT-1", "CT-2", "CT-3", "CT-4", "CT-5"} <= kinds, kinds
    info = [c for c in plan.cases if c.tool == "get_target_info"]
    assert [c.ct for c in info] == ["CT-3", "CT-1", "CT-2"]
    assert info[0].table == "open_targets.target" and info[0].column == "id"
    src, why = table_source(catalog, "open_targets.target")
    assert why is None and src["source"] == str(ot_fixture / "target") and src["format"] == "parquet"
    assert table_source(catalog, "open_targets.target_go")[0] is None, "item tables are not read by the oracle"
    config = _config()
    answers, _ = run_oracle(config, plan.queries)
    more, _ = run_oracle(config, oracle_queries(plan, answers, catalog))
    complete_cases(plan, {**answers, **more})
    info = {c.ct: c for c in plan.cases if c.tool == "get_target_info"}
    value = info["CT-3"].value
    assert value in (F.PCSK9, F.TP53) or str(value).startswith("ENSG")
    assert info["CT-3"].oracle["count"] == 1 and info["CT-3"].args == {"target_id": value}
    assert info["CT-1"].args["target_id"] != value and info["CT-1"].oracle["canonical"] == value
    absent = info["CT-2"].args["target_id"]
    assert absent.startswith("ENSG") and absent != value and info["CT-2"].oracle["absent"] == absent
    topk = next(c for c in plan.cases if c.ct == "CT-4" and not c.skip)
    assert len(topk.oracle["top"]) <= 3 and topk.oracle["total"] >= len(topk.oracle["top"])
    probes = {c.ct: c for c in plan.cases if c.tool == "get_chemical_probes"}
    assert probes["CT-3"].items == "chemicalProbes[]", "rows that are items of a container are counted as items"
    assert probes["CT-3"].skip or probes["CT-3"].oracle["items"] >= 1


def _out(obj=None, *, kind=None, text=""):
    return Outcome(kind is not None, kind, obj if obj is not None else {}, text or json.dumps(obj or {}), 0.01)


def test_judges_read_the_typed_answers():
    from types import SimpleNamespace

    binding = SimpleNamespace(result=SimpleNamespace(rows="$.rows", rows_of=None, kind="rows"), derived=None)
    unknown = Case("CT-2", "s", "t", {"id": "X9"}, "not_found")
    assert judge(unknown, _out(kind="not_found"), binding)[0] == "correct"
    assert judge(unknown, _out({"found": False, "error": "X9 not found"}), binding)[0] == "wrong"
    assert judge(unknown, _out(kind="too_large"), binding)[0] == "refused"
    assert "success" in judge_off(unknown, _out({"found": False}), binding)
    present = Case("CT-3", "s", "t", {"id": "X1"}, "rows", oracle={"count": 2}, value="X1")
    assert judge(present, _out({"_vbt": {"status": "empty"}, "rows": []}), binding)[0] == "wrong"
    assert judge(present, _out({"_vbt": {"status": "ok", "total": 2}, "rows": [{}, {}]}), binding)[0] == "correct"
    assert judge(present, _out({"_vbt": {"status": "ok", "total": 3}, "rows": []}), binding)[0] == "wrong"
    wrong_form = Case("CT-1", "s", "t", {"id": "x1"}, "rows", oracle={"canonical": "X1"}, value="X1")
    assert judge(wrong_form, _out({"_vbt": {"status": "ok"}, "id": "X1"}), binding)[0] == "correct"
    assert judge(wrong_form, _out({"_vbt": {"status": "ok"}, "id": "X2"}), binding)[0] == "wrong"
    bad_arg = Case("CT-5", "s", "t", {"limit": 0}, "invalid_argument")
    assert judge(bad_arg, _out(kind="invalid_argument"), binding)[0] == "correct"
    assert judge(bad_arg, _out({"rows": []}), binding)[0] == "wrong" and "accepted" in judge_off(bad_arg, _out({"rows": []}),
                                                                                                    binding)
    top = Case("CT-4", "s", "t", {"id": "X1", "limit": 2}, "topk", oracle={"top": [9.0, 5.0]},
               order={"field": "score", "k": 2})
    assert judge(top, _out({"rows": [{"score": 9.0}, {"score": 5.0}]}), binding)[0] == "correct"
    assert judge(top, _out({"rows": [{"score": 5.0}, {"score": 1.0}]}), binding)[0] == "wrong"
    thr = Case("CT-6", "s", "t", {"id": "X1", "min": 0.5}, "threshold", oracle={"median": 0.5},
               threshold={"field": "score", "op": "ge", "arg": "min"})
    assert judge(thr, _out({"rows": [{"score": 0.7}, {"score": 0.5}]}), binding)[0] == "correct"
    assert judge(thr, _out({"rows": [{"score": 0.7}, {"score": None}]}), binding)[0] == "wrong"
    assert "1 with an unknown" in judge_off(thr, _out({"rows": [{"score": None}]}), binding)
    items = SimpleNamespace(result=SimpleNamespace(rows="$.probes", rows_of="s.t_probes", kind="rows"), derived=None)
    probes = Case("CT-3", "s", "t", {"id": "X1"}, "rows", oracle={"count": 1, "items": 2}, value="X1",
                  items="probes[]")
    assert judge(probes, _out({"_vbt": {"status": "ok", "total": 2}, "probes": [{}, {}]}), items)[0] == "correct"
    verdict, detail = judge(probes, _out({"_vbt": {"status": "ok", "total": 1}, "probes": [{}]}), items)
    assert verdict == "wrong" and "oracle 2 items" in detail
    assert "items (2)" in judge(probes, _out({"_vbt": {"status": "empty"}, "probes": []}), items)[1]


# --------------------------------------------------------------------------- the steps


def test_steps_without_their_prerequisite_are_skipped_with_the_reason(tmp_path, monkeypatch):
    for k, v in DEAD.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("VBT_ZENODO_DIR", str(tmp_path / "no-zenodo"))
    report, out = run_validate(_config(), Options(steps=["correctness", "latency", "memory", "replication", "model"],
                                                  out=tmp_path / "out"))
    reasons = {s.name: (s.status, s.reason) for s in report.steps}
    assert reasons["correctness"] == (SKIPPED, "needs the check step (which tables are ready)")
    assert reasons["latency"][0] == SKIPPED and "no correctness calls" in reasons["latency"][1]
    assert reasons["memory"][0] == SKIPPED and reasons["replication"][0] == SKIPPED
    assert "Zenodo archive is not at" in reasons["replication"][1]
    assert reasons["model"] == (SKIPPED, "the provider is 'mock': no local model server to check")
    assert report.ok and (out / "validate.md").is_file() and (out / "validate.json").is_file()


async def test_off_calls_never_load_what_the_server_cannot_hold():
    """Without the gateway nothing admits upstream's whole-table loads: validate skips an off call whose tables (not
    yet resident) would take the server over its limit, with the admission's own estimate, and counts the tables an
    allowed off call loads."""
    from types import SimpleNamespace

    from vbt.datalayer.memory.estimate import MB
    from vbt.validate.contained import Calls

    resident: dict[str, set[str]] = {"s": {"t.warm"}}
    asked: list[list[str]] = []

    async def stats(tables):
        asked.append(list(tables))
        return SimpleNamespace(tables={t: {"mb": {"t.big": 4000, "t.small": 500, "t.mid": 1500}[t]} for t in tables})

    adm = SimpleNamespace(est=SimpleNamespace(safety=1.3, peak_upstream=lambda st, t: st["mb"] * MB),
                          limit_mb=lambda server: 4400.0, is_learned=lambda server, cold: False,
                          ledger=SimpleNamespace(resident=lambda server: frozenset(resident[server]),
                                                 resident_mb=lambda server: 800.0))
    gw = SimpleNamespace(admission=adm, _stats={}, service=SimpleNamespace(stats=stats))
    calls = Calls({}, ["s"], log_dir=Path("."), gateway=gw)
    why = await calls.off_guard("s", ["t.big"])
    assert why and "t.big whole" in why and "4,400 MB limit" in why
    assert await calls.off_guard("s", ["t.warm"]) is None, "a table the server holds costs nothing more"
    assert await calls.off_guard("s", ["t.small", "t.mid"]) is None              # 800 + 2000 x 1.3 = 3,400
    assert await calls.off_guard("s", ["t.mid"]) is None, "already loaded by an earlier off call"
    assert asked == [["t.big"], ["t.small", "t.mid"]], "statistics are asked once per table"
    assert await calls.off_guard("s", []) is None
    # an off load that killed the server is not tried again (its estimate fitted: the estimate was wrong)
    calls.after_off("s", ["t.mid"], _out(kind="oom", text="the server ran out of memory"))
    why = await calls.off_guard("s", ["t.mid"])
    assert why and "killed at its memory limit loading t.mid" in why
    calls.after_off("s", ["t.small"], _out({"rows": []}))
    assert await calls.off_guard("s", ["t.small"]) is None


def test_the_check_step_reuses_an_earlier_check(tmp_path, monkeypatch):
    for k, v in DEAD.items():
        monkeypatch.setenv(k, v)
    earlier = {"depth": "deep", "tables": {"open_targets.target": {"status": "ready", "checks": []},
                                           "open_targets.go": {"status": "missing", "checks": []}}}
    path = tmp_path / "check.json"
    path.write_text(json.dumps(earlier))
    report, out = run_validate(_config(), Options(steps=["check"], out=tmp_path / "out", check_from=path))
    step = report.steps[0]
    assert step.status == PASS and "reused from" in step.summary and "depth deep: 1 ready" in step.summary
    assert step.details["absent"] == ["open_targets.go"]
    assert json.loads((out / "check.json").read_text()) == earlier, "the run keeps the check it used"


def test_the_cli_runs_the_host_and_lint_steps(tmp_path, capsys):
    from vbt import cli

    rc = cli.main(["--profile", "mock", "validate", "--only", "host,lint", "--out", str(tmp_path / "v")])
    out = capsys.readouterr()
    assert rc == 0, out.err[-2000:]
    report = json.loads((tmp_path / "v" / "validate.json").read_text())
    assert [s["name"] for s in report["steps"]] == ["host", "lint"]
    host = report["steps"][0]
    assert host["status"] == PASS and host["details"]["sizing"]["settings"]["memory.default_server_mb"]["configured"] \
        == "auto"
    assert "Host and host-scaled limits" in out.out
    assert cli.main(["--profile", "mock", "validate", "--only", "nope"]) == 2
    assert set(STEPS) >= {"host", "lint", "check", "correctness", "latency", "memory", "live", "replication", "model"}


@linux_only
def test_validate_end_to_end_on_the_fixtures(tmp_path, monkeypatch, ot_fixture):
    """check + correctness + latency for target and drug on the OT-25.09-shaped fixtures, through the unmodified
    servers: every generated case is answered correctly in enforce mode, and the off answers show what upstream
    does with wrong-form and unknown identifiers."""
    from dl_upstream import upstream_missing

    if upstream_missing():
        pytest.skip(upstream_missing())
    pytest.importorskip("fastmcp")
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(ot_fixture))
    for k, v in DEAD.items():
        monkeypatch.setenv(k, v)
    config = _config(data={"cache_dir": str(tmp_path / "cache")}, mcp_servers_file="configs/mcp_servers.yaml",
                     vars={"upstream": os.environ.get("VBT_UPSTREAM") or str(ROOT / "third_party" / "TheVirtualBiotech")})
    report, out = run_validate(config, Options(steps=["host", "lint", "check", "correctness", "latency"],
                                                servers=["target", "drug"], depth="standard", max_tools=8,
                                                out=tmp_path / "out", timeout_s=180))
    steps = {s.name: s for s in report.steps}
    assert steps["check"].status == PASS, steps["check"].rows
    corr = steps["correctness"]
    assert corr.status == PASS, [r for r in corr.rows if r["enforce"] == "wrong"]
    by_test = corr.details["by_test"]
    assert by_test["CT-3"].get("correct", 0) >= 4 and by_test["CT-2"].get("correct", 0) >= 4
    assert by_test["CT-1"].get("correct", 0) >= 4 and by_test.get("CT-5", {}).get("correct", 0) >= 1
    off_unknown = [r["off"] for r in corr.rows if r["test"] == "CT-2" and r["enforce"] == "correct"]
    assert any(text.startswith("success") for text in off_unknown), "upstream answers unknown ids as successes"
    lat = {r["server"]: r for r in steps["latency"].rows}
    assert set(lat) == {"target", "drug"} and lat["target"]["calls"] > 0
    assert lat["target"]["enforce p95 ms"] is not None and lat["target"]["data-child requests"] > 0
    assert "correctness" in (out / "validate.md").read_text()


@pytest.mark.skipif(not os.environ.get("VBT_DL_REAL_DATA"), reason="set VBT_DL_REAL_DATA=<dir> to validate on real data")
@linux_only
def test_real_data_validation(tmp_path, monkeypatch):
    """The correctness step on the real Open Targets release (``VBT_DL_REAL_DATA`` = the shared real-data root or the
    25.09 directory) for the target and drug servers."""
    real = Path(os.environ["VBT_DL_REAL_DATA"])
    ot = real if (real / "target").is_dir() else real / "open_targets" / "25.09"
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(ot))
    for k, v in DEAD.items():
        monkeypatch.setenv(k, v)
    report, _out = run_validate(_config(data={"cache_dir": str(tmp_path / "cache")},
                                        mcp_servers_file="configs/mcp_servers.yaml"),
                                Options(steps=["check", "correctness", "latency"], servers=["target", "drug"],
                                        depth="standard", out=tmp_path / "out"))
    steps = {s.name: s for s in report.steps}
    assert steps["correctness"].status == PASS, [r for r in steps["correctness"].rows if r["enforce"] == "wrong"]
