"""Calibration, soak and enforce graduation (§18 F23, §21 "Soak and calibration"). Slow: ``VBT_DL_SLOW=1``.

Runs on the real Open Targets 25.09 release under ``VBT_DL_REAL_DATA`` when it is set (a release directory with
a download manifest, found as every opt-in real-data module finds it), else on the generated fixture; the memory test needs real data,
because on a few hundred rows the interpreter's own allocations dwarf the table.

* Memory: a sample-and-scale calibration (three row groups) predicts the peak RSS of an upstream-style
  whole-table pandas load within +/-30%, flat and nested tables included (``target`` and ``expression``
  carry nested lists).
* Witness: the p95 of the witness scan the gateway adds per call stays under 300 ms on the
  composite-key tables (``known_drug``: drug, target, disease, phase, status, urls; ``interaction``:
  source, both partners).
* Graduation: every server of ``data.gateway.enforce_servers`` in ``configs/default.yaml`` passes the
  observe-to-enforce checklist; with recorded observe runs (``VBT_SOAK_RUNS``, paths separated by
  ``os.pathsep``) its retro-audit evidence is required too.
* Fidelity: under ``profile: fidelity`` a derived tool on the paper scenarios is served ``pass`` behind
  the witness: an upstream answer that agrees with the data is returned as upstream's, and only a
  contradiction is an error (``tool_defect``).
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from dl_upstream import REPO, needs_arrow, real_ot_dir

pytestmark = [
    pytest.mark.slow, needs_arrow,
    pytest.mark.skipif(os.environ.get("VBT_DL_SLOW") != "1", reason="soak tests run with VBT_DL_SLOW=1"),
]

PCSK9 = "ENSG00000169174"
#: Tables of the memory soak: (table, nested) — nested tables carry list or struct columns.
MEMORY_TABLES = (("known_drug", False), ("target", True), ("expression", True), ("disease", True))
COMPOSITE = {"known_drug": "targetId", "interaction": "targetA"}
WITNESS_P95_S = 0.300
TOLERANCE = 0.30


def _real_root() -> Path | None:
    root = real_ot_dir()        # VBT_DL_REAL_DATA, never the shell's OPEN_TARGETS_DATA_PATH (the suite pops it: DEP-8)
    if root is None:
        return None
    return root if (root / "target").is_dir() and (root / ".download-manifest.json").is_file() else None


#: Read once at import.
REAL_ROOT = _real_root()


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if REAL_ROOT is not None:
        return REAL_ROOT
    import dl_fixtures as F

    return F.build_ot_fixture(tmp_path_factory.mktemp("soak") / "25.09")


@pytest.fixture(scope="module")
def ctx(root: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(root))
        settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                           "overlays_dir": str(REPO / "configs" / "data" / "overlays"),
                                           "cache_dir": str(tmp_path_factory.mktemp("soak-cache"))},
                                          project_root=REPO)
        yield ServiceContext(settings)


# ---------------------------------------------------------------------------- memory

_LOAD = r"""
import json, resource, sys
import pandas, pyarrow, pyarrow.dataset as ds          # imported before the baseline: only the load counts
def rss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0
before = rss_kb()
frame = ds.dataset(sys.argv[1], format="parquet").to_table().to_pandas()
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"rows": len(frame), "peak_bytes": (peak - before) * 1024}))
"""


def _measured_peak(path: Path) -> dict[str, Any]:
    out = subprocess.run([sys.executable, "-c", _LOAD, str(path)], capture_output=True, text=True, timeout=3600,
                         check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("table,nested", MEMORY_TABLES)
def test_calibrated_estimate_is_within_30_percent_of_measured_rss(ctx: Any, root: Path, table: str,
                                                                  nested: bool) -> None:
    if REAL_ROOT is None:
        pytest.skip("needs the real 25.09 subset (OPEN_TARGETS_DATA_PATH with a download manifest)")
    from vbt.datalayer.memory.calibrate import calibrate_table
    from vbt.datalayer.memory.estimate import MemoryEstimator
    from vbt.datalayer.service.verbs import load_verbs

    ref = f"open_targets.{table}"
    if not (root / table).is_dir():
        pytest.skip(f"{table} is not in this subset")
    stats = load_verbs()["_stats"](ctx, {"tables": [ref]})["tables"][ref]
    est = MemoryEstimator.from_settings(ctx.settings)
    est.add_calibration(calibrate_table(ctx, ref, write=False), stats["fingerprint"])
    estimate = est.peak_upstream(stats)
    measured = _measured_peak(root / table)
    ratio = estimate / max(1, measured["peak_bytes"])
    if nested:
        assert any("." in p or "[]" in p for p in stats["columns"]), f"{table}: expected nested leaves"
    assert abs(ratio - 1) <= TOLERANCE, (f"{table}: estimate {estimate / 2**20:.1f} MiB vs measured "
                                         f"{measured['peak_bytes'] / 2**20:.1f} MiB (ratio {ratio:.2f})")


# ---------------------------------------------------------------------------- witness overhead


@pytest.mark.parametrize("table", sorted(COMPOSITE))
def test_witness_p95_overhead_on_composite_key_tables(ctx: Any, table: str) -> None:
    from vbt.datalayer.predicate import Eq, to_json
    from vbt.datalayer.service.verbs import load_verbs

    ref = f"open_targets.{table}"
    verbs = load_verbs()
    t = ctx.table(ref)
    assert len(t.key) >= 3, f"{ref} is not a composite-key table"
    column = COMPOSITE[table]
    targets = [r[column] for r in verbs["_serve"](ctx, {"verb": "find", "table": ref, "columns": [column],
                                                         "limit": 50}).get("rows") or []][:20] or [PCSK9]
    request = {"table": ref, "key": list(t.key), "k": 20, "key_set_max": 20000}
    verbs["_witness"](ctx, {**request, "predicate": to_json(Eq(column, targets[0]))})       # warm the caches
    times = []
    for value in targets * max(1, 40 // len(targets)):
        t0 = time.perf_counter()
        verbs["_witness"](ctx, {**request, "predicate": to_json(Eq(column, value))})
        times.append(time.perf_counter() - t0)
    p95 = statistics.quantiles(times, n=20)[-1] if len(times) >= 20 else max(times)
    assert p95 < WITNESS_P95_S, f"{ref}: witness p95 {p95 * 1000:.0f} ms over {len(times)} calls"


# ---------------------------------------------------------------------------- graduation


def test_enforced_servers_pass_the_graduation_checklist() -> None:
    from vbt.datalayer.cli import graduation_checklist

    config = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    enforced = config["data"]["gateway"]["enforce_servers"]
    assert isinstance(enforced, list) and enforced, "enforce_servers lists the graduated servers"
    runs = [Path(p) for p in os.environ.get("VBT_SOAK_RUNS", "").split(os.pathsep) if p]
    report = graduation_checklist({}, enforced, runs)
    for server in enforced:
        items = report[server]["items"]
        static = {k: v for k, v in items.items() if k != "observe_evidence"}
        assert all(v["ok"] is True for v in static.values()), (server, static)
        if runs:
            assert items["observe_evidence"]["ok"] is True, (server, items["observe_evidence"])


# ---------------------------------------------------------------------------- fidelity


def _gateway(ctx: Any, tmp: Path, profile: str) -> Any:
    from vbt.datalayer.gateway import DataGateway
    from vbt.datalayer.replay import InProcessDataBridge
    from vbt.datalayer.settings import DataSettings

    settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                       "overlays_dir": str(REPO / "configs" / "data" / "overlays"),
                                       "cache_dir": str(ctx.settings.cache_dir),
                                       "gateway": {"profile": profile}}, project_root=REPO)
    tmp.mkdir(parents=True, exist_ok=True)
    gw = DataGateway(settings, ctx.catalog, ctx.registry, run={"mcp_output_dir": str(tmp)})
    gw.bind_bridge(InProcessDataBridge(ctx))
    return gw


#: Derived tools of the paper scenarios (target assessment of PCSK9), with their rows path.
PAPER_CALLS = (("association", "query_evidence", {"target_id": "PCSK9", "limit": 5}, "evidence"),)


@pytest.mark.parametrize("server,tool,args,rows", PAPER_CALLS)
def test_fidelity_serves_derived_tools_pass_and_refuses_only_contradictions(ctx: Any, tmp_path: Path, server: str,
                                                                           tool: str, args: dict[str, Any],
                                                                           rows: str) -> None:
    from vbt.datalayer.api import RawResult
    from vbt.datalayer.errors import ErrorKind, GatewayError

    async def run() -> None:
        safe = _gateway(ctx, tmp_path / "safe", "safe")
        plan = await safe.prepare(server, tool, dict(args), None)
        assert plan.route == "derived"
        derived = await safe.finish(plan, None)
        answer = {k: v for k, v in derived.obj.items() if k != "_vbt"}
        assert answer[rows], "the scenario has rows on this data"

        fidelity = _gateway(ctx, tmp_path / "fidelity", "fidelity")
        plan = await fidelity.prepare(server, tool, dict(args), None)
        assert plan.route == "upstream", "fidelity serves the tool pass, behind the witness"
        agree = await fidelity.finish(plan, RawResult(json.dumps(answer), answer, None, "ok"))
        assert agree.header["served_by"] == "upstream" and agree.status in ("ok", "partial")
        plan = await fidelity.prepare(server, tool, dict(args), None)
        empty = {**answer, rows: []}
        with pytest.raises(GatewayError) as exc:
            await fidelity.finish(plan, RawResult(json.dumps(empty), empty, None, "ok"))
        assert exc.value.kind == ErrorKind.tool_defect

    asyncio.run(run())
