"""Memory at scale (phase 4, F19): calibration tiers, the host budget, reaper containment (cgroups, the RSS
watchdog), the stdout relay cap, huge-table sidecars within time and byte budgets, and ``vbt ds status``.

Reaper tests run real children on Linux. cgroup containment is exercised against a fake delegated cgroup v2
tree (``VBT_REAPER_CGROUP_ROOT``), since this host mounts cgroup v1 on tmpfs; the watchdog is the real one.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import yaml  # noqa: E402

from vbt.datalayer.cli import memory_status  # noqa: E402
from vbt.datalayer.errors import ErrorKind, GatewayError  # noqa: E402
from vbt.datalayer.launch import REAPER  # noqa: E402
from vbt.datalayer.launch import reaper as reaper_mod  # noqa: E402
from vbt.datalayer.memory.admission import AdmissionController, TableRead  # noqa: E402
from vbt.datalayer.memory.calibrate import (  # noqa: E402
    calibrate_fragments,
    calibration_path,
    deep_bytes,
    feedback_from_status,
    fit,
    load_calibrations,
    load_feedback,
    measure_table,
    pick_samples,
    record_feedback,
    stats_of_fragments,
    write_calibration,
)
from vbt.datalayer.memory.crash import crash_decision, parse_exit_marker  # noqa: E402
from vbt.datalayer.memory.estimate import MB, MemoryEstimator  # noqa: E402
from vbt.datalayer.memory.host import HostBudget, host_budget_mb  # noqa: E402
from vbt.datalayer.memory.ledger import ResidencyLedger  # noqa: E402
from vbt.datalayer.plugins.base import Fragment  # noqa: E402
from vbt.datalayer.plugins.formats.parquet import ParquetFormat  # noqa: E402
from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service.sidecar import build_access_index, load_access_index  # noqa: E402
from vbt.datalayer.service.verbs import load_verbs  # noqa: E402
from vbt.datalayer.service.verbs.index_huge import build_huge_index  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the reaper needs Linux")
PARQUET = ParquetFormat()


# --------------------------------------------------------------------------- fixtures


def _flat(n: int, rng: random.Random) -> pa.Table:
    return pa.table({"id": [f"ENSG{i:011d}" for i in range(n)], "score": [rng.random() for _ in range(n)],
                     "count": [rng.randint(0, 10_000) for _ in range(n)]})


def _nested(n: int, rng: random.Random) -> pa.Table:
    """The target_essentiality shape: list<struct<screens: list<struct<cells: list<struct<...>>>>>>."""
    rows = []
    for i in range(n):
        rows.append({"id": f"ENSG{i:011d}", "geneEssentiality": [
            {"isEssential": bool(rng.getrandbits(1)), "depMapEssentiality": [
                {"tissueName": f"tissue{t}", "screens": [
                    {"cellLineName": f"CL{c}", "geneEffect": rng.random(), "expression": rng.random()}
                    for c in range(rng.randint(1, 4))]}
                for t in range(rng.randint(1, 3))]}]})
    return pa.Table.from_pylist(rows)


def _write(path: Path, table: pa.Table, rg: int) -> Fragment:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=rg)
    st = path.stat()
    return Fragment(uri=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns)


def _measured_total(table: pa.Table) -> int:
    return sum(v["pandas_bytes"] for v in measure_table(table).values())


# --------------------------------------------------------------------------- calibration


@pytest.mark.parametrize("shape, n, rg", [("flat", 6000, 500), ("nested", 3000, 250)])
def test_sample_calibration_is_within_tolerance(tmp_path: Path, shape: str, n: int, rg: int) -> None:
    rng = random.Random(7)
    table = _flat(n, rng) if shape == "flat" else _nested(n, rng)
    frag = _write(tmp_path / f"{shape}.parquet", table, rg)
    stats = stats_of_fragments(PARQUET, [frag])
    assert stats["rows"] == n
    seed = MemoryEstimator()
    cal = calibrate_fragments(PARQUET, [frag], table_stats=stats, row_groups=3, seed=seed, table=f"t.{shape}",
                              fingerprint=f"fp1:{shape}")
    assert cal["rows_sampled"] == 3 * rg and len(cal["sampled"]) == 3
    actual = _measured_total(table)
    est = MemoryEstimator(fragmentation=1.0)
    est.add_calibration(cal)
    fstats = {**stats, "fingerprint": f"fp1:{shape}"}
    assert est.tier(fstats) == "sample"
    calibrated = est.peak_upstream(fstats)
    assert abs(calibrated - actual) / actual < 0.15, (calibrated, actual)
    seed_est = MemoryEstimator(fragmentation=1.0).peak_upstream(fstats)
    assert MemoryEstimator(fragmentation=1.0).tier(fstats) == "seed"
    if shape == "nested":
        assert cal["columns"]["geneEssentiality"]["kind"] == "nested"
        assert abs(calibrated - actual) <= abs(seed_est - actual)   # the sample beats the seed factors
        assert cal["object_overhead_bytes"]["nested_item"] != seed.object_overhead["nested_item"]
    else:
        assert cal["columns"]["id"]["kind"] == "string" and cal["columns"]["score"]["kind"] == "flat"


def test_fitted_factors_reproduce_the_sample() -> None:
    stats = {"rows": 100, "columns": {"a": {"uncompressed_bytes": 800, "num_values": 100, "kind": "flat",
                                            "storage_type": "double"},
                                      "s": {"uncompressed_bytes": 1000, "num_values": 100, "kind": "string",
                                            "storage_type": "string"}}}
    measured = {"a": {"arrow_bytes": 400, "pandas_bytes": 400}, "s": {"arrow_bytes": 700, "pandas_bytes": 4000}}
    cal = fit(stats, measured, 50)
    assert cal["scale"] == 2.0 and cal["bytes_per_row"] == (800 + 8000) / 100
    assert cal["expansion"]["flat"] == 1.0 and cal["decode"]["flat"] == 1.0
    refit = MemoryEstimator(expansion=cal["expansion"], fragmentation=1.0, object_overhead_bytes=cal[
        "object_overhead_bytes"])
    assert abs(refit.peak_upstream(stats) - 8800) / 8800 < 0.05
    assert pick_samples(10, 3) == [0, 4, 9] and pick_samples(1, 3) == [0] and pick_samples(0) == []


def test_deep_bytes_counts_nested_objects() -> None:
    shallow = sys.getsizeof([{"a": [1, 2, 3]}])
    assert deep_bytes([{"a": [1, 2, 3]}]) > 3 * shallow
    shared = "x" * 1000
    assert deep_bytes([shared, shared]) < 2 * sys.getsizeof(shared)


def test_calibrations_and_feedback_are_stored_and_loaded(tmp_path: Path) -> None:
    cal = fit({"rows": 10, "columns": {"a": {"uncompressed_bytes": 80, "num_values": 10}}},
              {"a": {"arrow_bytes": 80, "pandas_bytes": 800}}, 10)
    cal["fingerprint"] = "fp1:sha256:abc"
    path = write_calibration(tmp_path, "src", "fp1:sha256:abc", cal)
    assert path == calibration_path(tmp_path, "src", "fp1:sha256:abc") and path.name == "calibration.json"
    assert load_calibrations(tmp_path)["fp1:sha256:abc"]["bytes_per_row"] == 80.0
    record_feedback(tmp_path, {"src.t": {"estimated_mb": 100.0, "fingerprint": "fp1:sha256:abc"}}, 250.0,
                    server="s1")
    record_feedback(tmp_path, {"src.t": {"estimated_mb": 100.0, "fingerprint": "fp1:sha256:abc"}}, 180.0)
    fb = load_feedback(tmp_path)
    assert fb["fp1:sha256:abc"]["factor"] == 2.5 and len(fb["fp1:sha256:abc"]["observations"]) == 2
    settings = DataSettings.from_dict({"cache_dir": str(tmp_path)})
    est = MemoryEstimator.from_settings(settings, load_calibrations=True)
    stats = {"rows": 10, "fingerprint": "fp1:sha256:abc", "columns": {"a": {"uncompressed_bytes": 80}}}
    assert est.tier(stats) == "measured"
    assert est.peak_upstream(stats) == math.ceil(80 * 10 * 2.5 * est.fragmentation)
    assert feedback_from_status({"peak_rss_mb": 900.0}, 300.0) == 600.0
    assert feedback_from_status({"peak_rss_mb": 200.0}, 300.0) is None and feedback_from_status(None, 1) is None


async def test_admission_records_measured_feedback(tmp_path: Path) -> None:
    status = tmp_path / "t.status.json"
    ledger = ResidencyLedger()
    ledger.set_status_path("t", status)
    settings = DataSettings.from_dict({})
    ac = AdmissionController(settings, MemoryEstimator(fragmentation=1.0), ledger, None, {"t": 100000},
                             feedback_dir=tmp_path)
    stats = {"rows": 1000, "fingerprint": "fp1:x", "columns": {"a": {"uncompressed_bytes": 100 * MB}}}
    adm = await ac.admit("t", [TableRead("src.big", "full_table", stats)], "upstream")
    assert adm.tiers == {"src.big": "seed"} and adm.fingerprints == {"src.big": "fp1:x"}
    status.write_text(json.dumps({"rss_mb": 900.0, "peak_rss_mb": 900.0 + 300.0, "ts": time.time(),
                                  "limit_mb": 100000}))
    ac.commit(adm, ok=True)
    fb = load_feedback(tmp_path)
    assert "fp1:x" in fb and fb["fp1:x"]["factor"] > 1.0
    assert ac.est.tier(stats) == "measured"


# --------------------------------------------------------------------------- host budget


def _ledger_with(servers: dict[str, float]) -> ResidencyLedger:
    ledger = ResidencyLedger(baseline_mb=100.0)
    for name, mb in servers.items():
        ledger.add(name, {f"{name}.table": mb})
    return ledger


async def test_host_budget_evicts_the_lru_idle_server() -> None:
    ledger = _ledger_with({"target": 3000.0, "drug": 3000.0, "pathway": 3000.0})
    recycled: list[str] = []
    clock = [0.0]

    async def recycle(server: str, wait_s: float = 30.0) -> bool:
        recycled.append(server)
        return True

    host = HostBudget(10_000.0, ledger, recycle=recycle, clock=lambda: clock[0])
    for name, t in (("pathway", 1.0), ("target", 2.0), ("drug", 3.0)):
        clock[0] = t
        host.touch(name)
    host.begin("target")                                  # busy: never recycled
    assert host.total_mb() == pytest.approx(9300.0)
    evicted = await host.reserve("drug", 2000.0)
    assert evicted == ["pathway"] and recycled == ["pathway"] and not ledger.resident("pathway")
    with pytest.raises(GatewayError) as e:                # only the busy target and the asker hold memory
        await host.reserve("drug", 6000.0)
    assert e.value.kind == ErrorKind.too_large and e.value.subkind == "host_busy"
    assert e.value.payload["busy_servers"] == ["target"] and e.value.retryable == "later"
    host.end("target")
    assert await host.reserve("drug", 6000.0) == ["target"]


async def test_admission_uses_the_host_budget() -> None:
    ledger = _ledger_with({"a": 4000.0})
    calls: list[str] = []

    async def recycle(server: str, wait_s: float = 30.0) -> bool:
        calls.append(server)
        return True

    settings = DataSettings.from_dict({"memory": {"host_budget_mb": 5000}})
    ac = AdmissionController(settings, MemoryEstimator(fragmentation=1.0, safety=1.0), ledger, recycle,
                             {"a": 20000, "b": 20000})
    host = ac.enable_host_budget()
    assert host.budget_mb == 5000.0
    stats = {"rows": 10, "columns": {"x": {"uncompressed_bytes": int(1000 * MB)}}}
    adm = await ac.admit("b", [TableRead("s.t", "full_table", stats)], "upstream")
    assert adm.host_evicted == ("a",) and calls == ["a"] and host.busy("b")
    assert adm.to_record()["host_recycled"] == ["a"]
    ac.commit(adm, ok=True)
    assert not host.busy("b")


def test_host_budget_setting() -> None:
    assert host_budget_mb(DataSettings.from_dict({"memory": {"host_budget_mb": "off"}})) is None
    assert host_budget_mb(DataSettings.from_dict({"memory": {"host_budget_mb": 4096}})) == 4096.0
    auto = DataSettings.from_dict({"memory": {"host_budget_mb": "auto", "harness_reserve_mb": 1024}})
    assert host_budget_mb(auto, total_mb=16000.0) == 0.75 * 16000 - 1024
    assert host_budget_mb(DataSettings.from_dict({}), total_mb=16000.0) == 0.75 * 16000 - 2048


# --------------------------------------------------------------------------- reaper: watchdog, cgroups, relay


def _reaper(args: list[str], child: list[str], env: dict[str, str] | None = None, **kw: Any
            ) -> subprocess.CompletedProcess:
    full = {**os.environ, **(env or {})}
    return subprocess.run([sys.executable, "-E", str(REAPER), *args, "--", *child], capture_output=True,
                          env=full, timeout=120, **kw)


@linux_only
def test_watchdog_kill_is_oom_and_never_retried(tmp_path: Path) -> None:
    status = tmp_path / "w.status.json"
    child = "import time\nb = b'x' * (400 * 1024 * 1024)\ntime.sleep(30)\n"
    proc = _reaper(["--limit-mb", "600", "--status", str(status), "--server", "w", "--containment", "watchdog"],
                   [sys.executable, "-c", child])
    assert proc.returncode == 128 + 9
    marker = parse_exit_marker(proc.stderr.decode())
    assert marker["reason"] == "memory_limit" and marker["watchdog"] is True and marker["signal"] == 9
    data = json.loads(status.read_text())
    assert data["containment"] == "watchdog" and data["watchdog_kill_mb"] == 300.0 and data["rlimit_data"]
    assert data["watchdog_rss_mb"] >= 300.0
    decision = crash_decision("connection closed", proc.stderr.decode(), server="w", tool="load", attempt=0)
    assert decision.oom and not decision.retry and decision.error.kind == ErrorKind.oom


@linux_only
def test_rss_containment_sets_no_data_limit(tmp_path: Path) -> None:
    """``limit_kind: rss`` (single_cell): TileDB reserves 1 GiB read buffers per column and fails with
    std::bad_alloc under any RLIMIT_DATA while its RSS stays low, so the child is contained by its resident
    memory (cgroup, else the watchdog) with no data limit."""
    status = tmp_path / "r.status.json"
    probe = [sys.executable, "-c", "import resource; print(resource.getrlimit(resource.RLIMIT_DATA)[0])"]
    env = {"VBT_REAPER_CGROUP_ROOT": str(tmp_path / "none"), "VBT_REAPER_CGROUP_V1_ROOT": str(tmp_path / "none-v1"),
           "XDG_RUNTIME_DIR": ""}
    proc = _reaper(["--limit-mb", "1000", "--status", str(status), "--server", "r", "--containment", "rss"], probe,
                   env=env)
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout.decode().split()[-1]) == reaper_mod.resource.RLIM_INFINITY
    data = json.loads(status.read_text())
    assert data["containment"] == "watchdog" and data["rlimit_data"] is False and data["watchdog_kill_mb"] == 500.0
    # a reservation far beyond the limit succeeds (RSS stays low); under RLIMIT_DATA the same mmap fails
    reserve = [sys.executable, "-c", "import mmap; m = mmap.mmap(-1, 3 << 30, flags=mmap.MAP_PRIVATE | "
               "mmap.MAP_ANONYMOUS); print('reserved')"]          # private writable: what RLIMIT_DATA counts
    ok = _reaper(["--limit-mb", "1000", "--status", str(status), "--server", "r", "--containment", "rss"], reserve,
                 env=env)
    assert ok.returncode == 0 and b"reserved" in ok.stdout, ok.stderr
    limited = _reaper(["--limit-mb", "1000", "--status", str(status), "--server", "r"], reserve)
    assert limited.returncode != 0 and b"reserved" not in limited.stdout


def test_watchdog_threshold() -> None:
    assert reaper_mod.watchdog_threshold_mb(12000) == 12000 - 600
    assert reaper_mod.watchdog_threshold_mb(4000) == 4000 - 512
    assert reaper_mod.watchdog_threshold_mb(600) == 300


@linux_only
def test_e_is_stripped_under_cgroup_containment(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu memory pids\n")
    (root / "cgroup.subtree_control").write_text("memory\n")
    status = tmp_path / "c.status.json"
    probe = tmp_path / "probe.json"
    code = ("import json, os, sys; json.dump({'seed': os.environ.get('PYTHONHASHSEED'), "
            "'ignore_env': sys.flags.ignore_environment, 'pid': os.getpid(), "
            f"'path': os.environ.get('PYTHONPATH')}}, open({str(probe)!r}, 'w'))")
    proc = _reaper(["--limit-mb", "512", "--status", str(status), "--server", "c", "--containment", "cgroup"],
                   [sys.executable, "-E", "-c", code],
                   env={"VBT_REAPER_CGROUP_ROOT": str(root), "PYTHONPATH": "/nowhere"})
    assert proc.returncode == 0, proc.stderr
    seen = json.loads(probe.read_text())
    assert seen["seed"] == "0" and seen["ignore_env"] == 0 and seen["path"] is None
    data = json.loads(status.read_text())
    assert data["containment"] == "cgroup_v2" and data["flags_stripped"] == ["-E"] and data["hash_seed"] == 0
    cg = Path(data["cgroup"])
    # the fake tree keeps what the reaper wrote (on a real cgroupfs the empty cgroup is removed afterwards)
    assert cg.parent == root and (cg / "memory.max").read_text() == str(512 * 1024 * 1024)
    assert (cg / "memory.swap.max").read_text() == "0" and (cg / "cgroup.procs").read_text() == str(seen["pid"])
    # fallback: no delegation anywhere -> the watchdog contains the child, flags still stripped
    proc = _reaper(["--limit-mb", "2048", "--status", str(status), "--server", "c", "--containment", "cgroup"],
                   [sys.executable, "-E", "-c", code],
                   env={"VBT_REAPER_CGROUP_ROOT": str(tmp_path / "none"),
                        "VBT_REAPER_CGROUP_V1_ROOT": str(tmp_path / "none-v1"), "XDG_RUNTIME_DIR": ""})
    assert proc.returncode == 0, proc.stderr
    data = json.loads(status.read_text())
    assert data["containment"] == "watchdog" and data["flags_stripped"] == ["-E"]
    assert json.loads(probe.read_text())["seed"] == "0"


def test_cgroup_v2_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "cgroup.controllers").write_text("memory\n")
    monkeypatch.setenv("VBT_REAPER_CGROUP_ROOT", str(tmp_path))
    cg = reaper_mod.make_cgroup_v2("srv/x", 256)
    assert cg is not None and Path(cg.path).name.startswith("vbt-srv_x-")
    assert (Path(cg.path) / "memory.max").read_text() == str(256 * 1024 * 1024)
    assert (Path(cg.path) / "memory.oom.group").read_text() == "1"
    assert cg.add(4242) and (Path(cg.path) / "cgroup.procs").read_text() == "4242"
    (Path(cg.path) / "memory.events").write_text("low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\n")
    assert cg.oom_kills() == 1


def _relay_bytes(data: bytes, cap: int) -> tuple[bytes, dict[str, int]]:
    r_in, w_in = os.pipe()
    r_out, w_out = os.pipe()
    chunks: list[bytes] = []

    def drain() -> None:
        while True:
            b = os.read(r_out, 1 << 16)
            if not b:
                return
            chunks.append(b)

    t = threading.Thread(target=drain)
    t.start()

    def feed() -> None:
        for i in range(0, len(data), 7919):
            os.write(w_in, data[i:i + 7919])
        os.close(w_in)

    f = threading.Thread(target=feed)
    f.start()
    counts = reaper_mod.relay(r_in, w_out, cap)
    f.join()
    os.close(w_out)
    t.join()
    os.close(r_in)
    os.close(r_out)
    return b"".join(chunks), counts


def test_relay_replaces_an_oversized_message_with_an_error_of_the_same_id() -> None:
    small = json.dumps({"jsonrpc": "2.0", "id": 6, "result": {"ok": True}}) + "\n"
    big = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"rows": ["x" * 100] * 30000}}) + "\n"
    note = json.dumps({"jsonrpc": "2.0", "method": "notifications/log", "params": {"d": "y" * 3_000_000}}) + "\n"
    last = json.dumps({"jsonrpc": "2.0", "id": "req-8", "result": {}}) + "\n"
    out, counts = _relay_bytes((small + big + note + last).encode(), 1024 * 1024)
    lines = [json.loads(ln) for ln in out.decode().splitlines()]
    assert lines[0] == json.loads(small) and lines[-1] == json.loads(last)
    err = lines[1]
    assert err["id"] == 7 and err["error"]["code"] == reaper_mod.RELAY_ERROR_CODE
    assert err["error"]["data"]["bytes"] == len(big) and "relay cap" in err["error"]["message"]
    assert len(lines) == 3 and counts == {"messages": 4, "replaced": 1, "dropped": 1}
    # under the cap everything passes byte for byte
    out, counts = _relay_bytes((small + last).encode(), 1024 * 1024)
    assert out == (small + last).encode() and counts["replaced"] == 0


@linux_only
def test_relay_through_the_reaper(tmp_path: Path) -> None:
    child = ("import json, sys\n"
             "print(json.dumps({'jsonrpc': '2.0', 'id': 1, 'result': 'a'}), flush=True)\n"
             "print(json.dumps({'jsonrpc': '2.0', 'id': 2, 'result': 'b' * (3 * 1024 * 1024)}), flush=True)\n"
             "print(json.dumps({'jsonrpc': '2.0', 'id': 3, 'result': 'c'}), flush=True)\n")
    status = tmp_path / "r.status.json"
    proc = _reaper(["--limit-mb", "0", "--status", str(status), "--server", "r", "--relay-max-mb", "1"],
                   [sys.executable, "-E", "-c", child])
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(ln) for ln in proc.stdout.decode().splitlines()]
    assert [m["id"] for m in lines] == [1, 2, 3] and lines[0]["result"] == "a" and lines[2]["result"] == "c"
    assert lines[1]["error"]["code"] == -32001
    assert json.loads(status.read_text())["relay"]["replaced"] == 1
    # off by default: the same child's output passes untouched
    proc = _reaper(["--limit-mb", "0", "--server", "r"], [sys.executable, "-E", "-c", child])
    assert len(proc.stdout.decode().splitlines()[1]) > 3 * 1024 * 1024


# --------------------------------------------------------------------------- huge sidecars


@pytest.fixture
def lit_ctx(tmp_path: Path) -> ServiceContext:
    root = tmp_path / "ot"
    rng = random.Random(3)
    words = [f"ENSG{i:011d}" for i in range(40)]
    rows = [{"pmid": str(1000 + i), "keywordId": rng.choice(words), "year": 2000 + i % 20} for i in range(4000)]
    for shard in range(3):
        _write(root / "literature" / f"part-{shard:05d}.parquet", pa.Table.from_pylist(rows[shard::3]), 100)
    desc = {"schema": "vbt.datasource/1", "source": "ot", "title": "ot", "root": str(root),
            "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
            "tables": {"literature": {"kind": "fact", "path": "literature", "size_class": "huge",
                                      "grain": "one co-mention", "key": {"columns": ["pmid", "keywordId"]},
                                      "access_paths": [{"columns": ["keywordId"], "via": "sidecar_index",
                                                        "build": "on_demand"}],
                                      "columns": {"pmid": {"role": "label"}, "keywordId": {"role": "label"},
                                                  "year": {"role": "time", "precision": "year"}}}}}
    (tmp_path / "sources").mkdir()
    (tmp_path / "overlays").mkdir()
    (tmp_path / "sources" / "ot.yaml").write_text(yaml.safe_dump(desc))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp_path / "sources"),
                                       "overlays_dir": str(tmp_path / "overlays"),
                                       "cache_dir": str(tmp_path / "cache")}, project_root=tmp_path)
    return ServiceContext(settings)


def test_huge_index_is_resumable_within_budgets(lit_ctx: ServiceContext) -> None:
    reader = lit_ctx.reader("ot.literature")
    ticks = iter(range(10**6))
    first = build_huge_index(reader, "keywordId", time_budget_s=5, max_entries=50, clock=lambda: next(ticks))
    assert first["status"] == "incomplete" and "time budget" in first["reason"]
    assert 0 < first["units_done"] < first["units_total"] == 42
    second = build_huge_index(reader, "keywordId", budget_bytes=1, max_entries=50)
    assert second["status"] == "incomplete" and "byte budget" in second["reason"]
    assert second["units_done"] == first["units_done"]           # resumed: nothing redone, nothing lost
    done = load_verbs()["_build_index_huge"](lit_ctx, {"table": "ot.literature", "max_entries": 50})
    assert done["status"] == "complete" and done["units_done"] == 42 and done["rows"] == 40
    huge = load_access_index(done["path"])
    # the same index the one-pass builder writes
    one_pass_path, _ = build_access_index(reader, "keywordId", force=True)
    assert huge.entries == load_access_index(one_pass_path).entries
    assert not Path(done["path"] + ".work").exists()
    again = load_verbs()["_build_index_huge"](lit_ctx, {"table": "ot.literature"})
    assert again["status"] == "complete" and again["scanned_bytes"] == 0


def test_huge_index_refuses_undeclared_columns(lit_ctx: ServiceContext) -> None:
    with pytest.raises(Exception, match="no sidecar_index access path"):
        load_verbs()["_build_index_huge"](lit_ctx, {"table": "ot.literature", "column": "year"})


# --------------------------------------------------------------------------- vbt ds status


def test_status_reads_status_files_and_calibrations(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "target.status.json").write_text(json.dumps({"pid": 1, "rss_mb": 812.5, "peak_rss_mb": 900.0,
                                                         "limit_mb": 7168, "containment": "watchdog",
                                                         "hash_seed": 0, "ts": time.time()}))
    (logs / "drug.status.json").write_text(json.dumps({"pid": 2, "rss_mb": None, "peak_rss_mb": 50.0,
                                                       "exit": {"code": 0}}))
    cache = tmp_path / "cache"
    cal = fit({"rows": 4, "columns": {"a": {"uncompressed_bytes": 32}}}, {"a": {"pandas_bytes": 64}}, 4)
    cal.update({"fingerprint": "fp1:z", "table": "s.t"})
    write_calibration(cache, "s", "fp1:z", cal)
    config = {"data": {"cache_dir": str(cache), "memory": {"host_budget_mb": 9000}}}
    body = memory_status(config, logs)
    assert body["host"]["budget_mb"] == 9000.0 and body["host"]["resident_mb"] == 812.5
    assert set(body["servers"]) == {"target", "drug"} and body["calibrations"]["fp1:z"]["table"] == "s.t"
    from vbt.datalayer import cli

    ns = type("NS", (), {"log_dir": str(logs), "run": None, "json": False})()
    assert cli.cmd_status(ns, config) == 0


def test_status_and_calibrate_commands_are_registered() -> None:
    import argparse

    from vbt.datalayer.cli import COMMANDS, add_datasource_parsers

    parser = argparse.ArgumentParser()
    add_datasource_parsers(parser.add_subparsers(dest="cmd"))
    ns = parser.parse_args(["ds", "status", "--log-dir", "/tmp/x", "--json"])
    assert ns.handler is COMMANDS["status"]
    ns = parser.parse_args(["ds", "calibrate", "--table", "open_targets.target", "--row-groups", "2"])
    assert ns.handler is COMMANDS["calibrate"] and ns.row_groups == 2
    ns = parser.parse_args(["ds", "index", "build", "--access-paths", "--huge", "--time-budget-s", "60", "--once"])
    assert ns.huge and ns.time_budget_s == 60 and ns.once
    ns = parser.parse_args(["ds", "overlay", "init", "uniprot", "--from-json", "x.json", "--out", "-"])
    assert ns.handler is COMMANDS["overlay init"]


async def test_gateway_enables_the_host_budget(tmp_path: Path) -> None:
    """F19: a gateway built from settings caps host memory (auto by default) and binds its LRU recycle to
    the bridge; ``off`` leaves it unset."""
    from test_dl_gateway_flow import world

    gw = world(tmp_path, data={"memory": {"host_budget_mb": 5000}})
    host = gw.admission.host
    assert host is not None and host.enabled and host.budget_mb == 5000.0
    assert host.recycle == gw.bridge.recycle and host.servers is not None
    gw.admission.ledger.add("pathway", {"pathway.t": 4000.0})
    host.touch("pathway")
    assert await host.reserve("target", 2000.0) == ["pathway"]          # the idle server is recycled
    assert world(tmp_path / "off", data={"memory": {"host_budget_mb": "off"}}).admission.host is None
    auto = world(tmp_path / "auto").admission.host
    assert auto is None or auto.enabled == (host_budget_mb(DataSettings.from_dict({})) is not None)
