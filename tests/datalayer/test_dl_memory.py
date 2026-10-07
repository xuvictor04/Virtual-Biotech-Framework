"""Memory estimates, the residency ledger, admission and crash classification (§10.4, §14).
Pure unit tests: no servers, no data."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.ipc import TableStatsModel
from vbt.datalayer.memory import (
    MB, AdmissionController, MemoryEstimator, ResidencyLedger, TableRead, classify_error_text, crash_decision,
    is_memory_exit, parse_exit_marker, read_status,
)
from vbt.datalayer.memory.estimate import leaves_of
from vbt.datalayer.settings import DataSettings

# --------------------------------------------------------------------------- estimates

#: A three-level nested table shaped like target_essentiality: 11.6 MB of footer leaf bytes.
#: The §10.4 measurement on such a fixture: pandas peak RSS +1,001 MB, revision-1 estimate 107 MB.
THREE_LEVEL = {
    "fingerprint": "sha256:x", "rows": 5000,
    "columns": {
        "id": {"uncompressed_bytes": 100_000, "num_values": 5_000, "kind": "string", "storage_type": "large_string"},
        "geneEssentiality[].isEssential": {"uncompressed_bytes": 1_250, "num_values": 5_000, "max_rep_level": 1,
                                           "kind": "nested", "storage_type": "bool"},
        "geneEssentiality[].depMapEssentiality[].tissueId": {
            "uncompressed_bytes": 1_500_000, "num_values": 150_000, "max_rep_level": 2, "kind": "nested",
            "storage_type": "large_string"},
        "geneEssentiality[].depMapEssentiality[].screens[].geneEffect": {
            "uncompressed_bytes": 4_800_000, "num_values": 600_000, "max_rep_level": 3, "kind": "nested",
            "storage_type": "double"},
        "geneEssentiality[].depMapEssentiality[].screens[].depmapId": {
            "uncompressed_bytes": 5_200_000, "num_values": 600_000, "max_rep_level": 3, "kind": "nested",
            "storage_type": "large_string"},
    },
    "row_bytes_p99": {"row": 310_000, "target_essentiality_screens": 420},
}
MEASURED_PEAK = 1001 * MB


def settings(**memory):
    return DataSettings.from_dict({"memory": memory} if memory else {})


def stats(mb: float, rows: int = 1000) -> dict:
    """A flat table whose upstream peak is ``mb`` MB with the default factors (1.5 x 1.15)."""
    return {"fingerprint": "f", "rows": rows,
            "columns": {"x": {"uncompressed_bytes": int(mb * MB / (1.5 * 1.15)), "num_values": rows}}}


def test_estimator_three_level_num_values_term():
    est = MemoryEstimator.from_settings(settings())
    model = TableStatsModel.model_validate(THREE_LEVEL)
    leaf_bytes = sum(c["uncompressed_bytes"] for c in THREE_LEVEL["columns"].values())
    assert leaf_bytes == 11_601_250

    bytes_term = 100_000 * 3.5 + (1_250 + 1_500_000 + 4_800_000 + 5_200_000) * 8.0
    per_value = 5_000 * 50 + 5_000 * 120 + 150_000 * (50 + 240) + 600_000 * 360 + 600_000 * (50 + 360)
    structs = (5_000 + 150_000 + 600_000) * 240      # one dict per item of each struct container
    expected = (bytes_term + per_value + structs) * 1.15
    assert est.bytes_term(model) == pytest.approx(bytes_term)
    leaves = leaves_of(model)
    assert [leaf.depth for leaf in leaves] == [3, 3, 2, 1, 0]   # sorted by path
    assert est.struct_items(leaves) == {
        "geneEssentiality[]": 5_000, "geneEssentiality[].depMapEssentiality[]": 150_000,
        "geneEssentiality[].depMapEssentiality[].screens[]": 600_000}
    peak = est.peak_upstream(model)
    assert peak == pytest.approx(expected, rel=1e-9)
    assert est.peak_upstream(THREE_LEVEL) == peak, "the JSON form of _stats works too"

    # Revision 1 (bytes only) was about 9x low; the num_values term lands in the measured band.
    rev1 = leaf_bytes * 8.0 * 1.15
    assert rev1 == pytest.approx(107e6, rel=0.01) and rev1 / MEASURED_PEAK < 0.15
    assert 0.7 <= peak / MEASURED_PEAK <= 1.3

    # Arrow scan: projected leaves only
    one = est.peak_arrow_scan(model, ["geneEssentiality[].depMapEssentiality[].screens[].geneEffect"])
    assert one == pytest.approx(4_800_000 * est.decode["nested"])
    assert est.peak_arrow_scan(model, ["geneEssentiality"]) > one
    assert est.peak_arrow_scan(model) >= est.peak_arrow_scan(model, ["geneEssentiality"])
    assert est.peak_arrow_scan(model, ["id"]) == pytest.approx(100_000 * est.decode["string"])


def test_estimator_flat_rows_transient_densify():
    est = MemoryEstimator(expansion={"flat": 1.5, "string": 3.5, "nested": 8.0, "fragmentation": 1.15},
                          object_overhead_bytes={"string": 50}, safety=1.3)
    flat = {"rows": 1000, "columns": {"a": {"uncompressed_bytes": 8000, "storage_type": "double"},
                                      "s": {"uncompressed_bytes": 20000, "storage_type": "string"}}}
    # num_values defaults to rows for top-level leaves: 1000 strings x 50 B
    assert est.peak_upstream(flat) == pytest.approx((8000 * 1.5 + 20000 * 3.5 + 1000 * 50) * 1.15, abs=1)
    assert est.peak_upstream(None) == 0
    # bounded scans hold the matching share, and at least one 1024-row batch
    big = {"rows": 1_000_000, "columns": {"a": {"uncompressed_bytes": 8_000_000, "num_values": 1_000_000}}}
    peak = est.peak_upstream(big)
    assert est.transient(big, 0.1) == pytest.approx(peak * 0.1, rel=1e-6)
    assert est.transient(big, 0.0) == pytest.approx(peak * 1024 / 1_000_000, rel=1e-3)
    assert est.transient(big, None) == peak
    # row bytes per grain come from row_bytes_p99, never the stored-row mean
    assert est.row_bytes("row", THREE_LEVEL) == 310_000
    assert est.row_bytes("target_essentiality_screens", THREE_LEVEL) == 420
    assert est.row_bytes("cell", THREE_LEVEL) is None
    assert est.max_rows("target_essentiality_screens", THREE_LEVEL, 2_000_000) == 2_000_000 // 420
    assert est.max_rows("row", THREE_LEVEL, 100) == 1
    assert est.max_rows("cell", THREE_LEVEL, 2_000_000) is None
    assert est.densify(17_000, 18_000, 8) == 17_000 * 18_000 * 8
    assert est.peak_all([flat, None, flat]) == 2 * est.peak_upstream(flat)


# --------------------------------------------------------------------------- ledger


def test_ledger_generations_status_and_reservations(tmp_path):
    now = [1000.0]
    ledger = ResidencyLedger(baseline_mb=200, clock=lambda: now[0])
    assert not ledger.sync("t", 1)
    ledger.add("t", {"ot.target": 3500.0})
    assert ledger.resident("t") == {"ot.target"} and ledger.resident_mb("t") == 3700.0
    rid = ledger.reserve("t", {"ot.known_drug": 500.0})
    assert ledger.pending("t") == {"ot.known_drug"} and ledger.resident_mb("t") == 4200.0
    ledger.release("t", rid)
    assert ledger.resident_mb("t") == 3700.0

    # the reaper's status file wins while it is fresh and from this generation
    status = tmp_path / "t.status.json"
    ledger.set_status_path("t", status)
    status.write_text(json.dumps({"pid": 1, "rss_mb": 5100.0, "limit_mb": 12000, "ts": now[0]}))
    assert read_status(status)["rss_mb"] == 5100.0
    assert ledger.resident_mb("t") == 5100.0 and ledger.limit_mb("t") == 12000.0
    now[0] += 10
    assert ledger.resident_mb("t") == 3700.0, "a stale status file falls back to estimates"

    # a new generation forgets what the old process held, and ignores its last status file
    status.write_text(json.dumps({"pid": 1, "rss_mb": 5100.0, "limit_mb": 12000, "ts": now[0]}))
    now[0] += 1
    assert ledger.sync("t", 2)
    assert ledger.resident("t") == frozenset() and ledger.resident_mb("t") == 200.0
    status.write_text(json.dumps({"pid": 2, "rss_mb": 310.0, "limit_mb": 12000, "ts": now[0]}))
    assert ledger.resident_mb("t") == 310.0 and ledger.baseline_mb("t") == 310.0
    ledger.add("t", {"x": 1.0}, generation=1)
    assert ledger.resident("t") == frozenset(), "a past generation's load is ignored"
    snap = ledger.snapshot("t")["t"]
    assert snap["generation"] == 2 and snap["tables"] == {}
    status.write_text(json.dumps({"pid": 2, "rss_mb": 310.0, "ts": now[0], "exit": {"code": 0}}))
    assert ledger.status("t") is None, "an exited child holds nothing"


def test_ledger_reservations_lapse():
    now = [0.0]
    ledger = ResidencyLedger(reservation_ttl_s=60, clock=lambda: now[0])
    ledger.reserve("t", {"a": 100.0})
    assert ledger.pending("t") == {"a"}
    now[0] = 61.0
    assert ledger.pending("t") == frozenset() and ledger.reserved_mb("t") == 0


# --------------------------------------------------------------------------- admission


class Recycler:
    def __init__(self, ok=True):
        self.calls: list[tuple] = []
        self.ok = ok

    async def __call__(self, server, wait_s=30.0):
        self.calls.append((server, wait_s))
        return self.ok


def controller(limit_mb=4000, *, recycle=None, baseline_mb=300.0, **memory):
    s = settings(**memory)
    ledger = ResidencyLedger(baseline_mb=baseline_mb)
    return AdmissionController(s, MemoryEstimator.from_settings(s), ledger, recycle, {"t": limit_mb})


async def test_admission_refuses_before_any_call():
    rec = Recycler()
    ac = controller(4000, recycle=rec)
    with pytest.raises(GatewayError) as exc:
        await ac.admit("t", {"ot.evidence": stats(5000)}, "upstream", tool="mcp__t__x", alternative="mcp__data__find")
    err = exc.value
    assert err.kind is ErrorKind.too_large and err.subkind == "over_limit"
    assert err.payload["permanent"] is True and err.payload["alternative"] == "mcp__data__find"
    assert err.payload["limit_mb"] == 4000 and err.payload["need_mb"] == pytest.approx(5000 * 1.3, rel=1e-3)
    assert err.retryable == "with_narrower_arguments" and json.loads(str(err))["kind"] == "too_large"
    assert rec.calls == [], "a permanent refusal never recycles"
    assert ac.ledger.pending("t") == frozenset()


async def test_no_admission_for_other_routes():
    ac = controller(10)
    for route in ("derived", "none"):
        adm = await ac.admit("t", {"ot.evidence": stats(50_000)}, route)
        assert not adm.admitted and adm.lock is None and adm.cold_tables == ()
        assert adm.to_record() == {"admission": "not_applicable"}


async def test_warm_and_cold_calls_commit_and_unestimated():
    ac = controller(8000)
    adm = await ac.admit("t", [TableRead("ot.target", "full_table", stats(1000)),
                               TableRead("ot.go", "full_table", None)], "upstream")
    assert adm.cold_tables == ("ot.go", "ot.target") and adm.lock is ac.cold_lock("t")
    assert adm.unestimated == ("ot.go",) and adm.need_mb == pytest.approx(1300, rel=1e-3)
    assert ac.ledger.pending("t") == {"ot.go", "ot.target"}
    ac.commit(adm)
    ac.commit(adm)  # idempotent
    assert ac.ledger.resident("t") == {"ot.go", "ot.target"} and ac.ledger.pending("t") == frozenset()
    warm = await ac.admit("t", [TableRead("ot.target", "full_table", stats(1000))], "upstream")
    assert warm.cold_tables == () and warm.lock is None and warm.need_mb == 0
    rec = warm.to_record()
    assert rec["admission"] == "admitted" and rec["cold_tables"] == [] and rec["limit_mb"] == 8000
    # bounded scans add their transient share but take no cold lock
    scan = await ac.admit("t", [TableRead("ot.target", "full_table", stats(1000)),
                                TableRead("ot.evidence", "bounded_scan", stats(2000, rows=10_000_000), 0.01)],
                          "upstream")
    assert scan.lock is None and scan.need_mb == pytest.approx(2000 * 0.01 * 1.3, rel=1e-2)


async def test_recycle_when_resident_plus_need_exceeds_the_limit():
    rec = Recycler()
    ac = controller(5000, recycle=rec, baseline_mb=300)
    ac.ledger.add("t", {"ot.target": 3000.0})
    adm = await ac.admit("t", {"ot.known_drug": stats(1000)}, "upstream")
    # 3300 resident + 1300 needed fits under 5000: no recycle
    assert rec.calls == [] and adm.cold_tables == ("ot.known_drug",)
    ac.commit(adm)
    adm = await ac.admit("t", {"ot.disease": stats(1000)}, "upstream")
    # 4300 resident + 1300 > 5000: recycled, the ledger starts over
    assert rec.calls == [("t", 30.0)] and adm.recycled
    assert ac.ledger.resident("t") == frozenset() and adm.cold_tables == ("ot.disease",)
    assert adm.to_record()["recycled"] is True

    # recycling refused (busy): too_large, retryable later
    rec.ok = False
    ac.ledger.add("t", {"ot.target": 4000.0})
    with pytest.raises(GatewayError) as exc:
        await ac.admit("t", {"ot.drug": stats(1000)}, "upstream")
    assert exc.value.subkind == "resident_memory" and exc.value.retryable == "later"
    assert exc.value.payload["resident_mb"] > 4000


async def test_recycle_thrash_guard():
    rec = Recycler()
    clock = [0.0]
    s = settings(max_recycles_per_10min=2)
    ac = AdmissionController(s, None, ResidencyLedger(), rec, {"t": 2000}, clock=lambda: clock[0])
    for _ in range(2):
        ac.ledger.add("t", {"big": 1500.0})
        await ac.admit("t", {"other": stats(400)}, "upstream")
    assert len(rec.calls) == 2 and not ac.can_recycle("t")
    ac.ledger.add("t", {"big": 1500.0})
    with pytest.raises(GatewayError, match="could not be recycled"):
        await ac.admit("t", {"other": stats(400)}, "upstream")
    assert len(rec.calls) == 2
    clock[0] = 601.0
    assert ac.can_recycle("t")
    off = AdmissionController(settings(recycle_idle_servers=False), None, ResidencyLedger(), rec, {"t": 2000})
    assert not off.can_recycle("t")


async def test_learned_refusals_after_oom():
    ac = controller(8000)
    adm = await ac.admit("t", {"ot.target": stats(1000), "ot.go": stats(500)}, "upstream")
    ac.commit(adm, ok=False, oom=True)
    assert ac.ledger.pending("t") == frozenset() and ac.ledger.resident("t") == frozenset()
    with pytest.raises(GatewayError) as exc:
        await ac.admit("t", {"ot.go": stats(500), "ot.target": stats(1000), "ot.x": stats(1)}, "upstream")
    assert exc.value.subkind == "learned_refusal" and exc.value.payload["learned"] is True
    # a smaller cold set, or the same tables on another server, is still admitted
    assert (await ac.admit("t", {"ot.target": stats(1000)}, "upstream")).cold_tables == ("ot.target",)
    ac.limits["u"] = 8000
    assert (await ac.admit("u", {"ot.target": stats(1000), "ot.go": stats(500)}, "upstream")).admitted
    ac.learn_refusal("t", [])
    assert ("t", frozenset()) not in ac.learned_refusals, "a warm OOM never refuses every call"
    assert ac.snapshot()["learned_refusals"] == [["t", ["ot.go", "ot.target"]]]


async def test_concurrent_cold_calls_are_serialised_and_warm_calls_are_not():
    ac = controller(100_000)
    ac.ledger.add("t", {"warm": 10.0})
    spans: dict[str, tuple[float, float]] = {}

    async def call(name, reads):
        adm = await ac.admit("t", reads, "upstream")
        lock = adm.lock
        if lock is None:
            t0 = time.monotonic()
            await asyncio.sleep(0.15)
        else:
            async with lock:
                t0 = time.monotonic()
                await asyncio.sleep(0.15)
        spans[name] = (t0, time.monotonic())
        ac.commit(adm)
        return adm

    a, b, w1, w2 = await asyncio.gather(call("a", {"cold_a": stats(10)}), call("b", {"cold_b": stats(10)}),
                                        call("w1", {"warm": stats(10)}), call("w2", {"warm": stats(10)}))
    assert a.lock is b.lock is ac.cold_lock("t") and w1.lock is None and w2.lock is None
    first, second = sorted([spans["a"], spans["b"]])
    assert second[0] >= first[1] - 1e-3, "cold calls on one server never overlap"
    assert spans["w2"][0] < spans["w1"][1] and spans["w1"][0] < spans["w2"][1], "warm calls overlap"
    assert ac.ledger.resident("t") == {"warm", "cold_a", "cold_b"}
    assert ac.cold_lock("u") is not ac.cold_lock("t")


async def test_generation_reset_through_the_bridge_generation():
    gen = {"t": 1}
    s = settings()
    ac = AdmissionController(s, None, ResidencyLedger(), None, {"t": 8000}, generation=lambda server: gen[server])
    adm = await ac.admit("t", {"ot.target": stats(1000)}, "upstream")
    ac.commit(adm)
    assert ac.ledger.resident("t") == {"ot.target"}
    gen["t"] = 2                                   # the server restarted (crash or recycle)
    adm = await ac.admit("t", {"ot.target": stats(1000)}, "upstream")
    assert adm.cold_tables == ("ot.target",) and adm.generation == 2
    gen["t"] = 3                                   # ... and again during the call
    ac.commit(adm)
    assert ac.ledger.resident("t") == frozenset(), "a load from a past generation is not resident"


def test_limits_fall_back_to_status_file_and_settings(tmp_path):
    s = DataSettings.from_dict({"memory": {"default_server_mb": 6000}, "service": {"mem_limit_mb": 2500}})
    ledger = ResidencyLedger()
    ac = AdmissionController(s, None, ledger)
    assert ac.limit_mb("target") == 6000 and ac.limit_mb("data") == 2500
    status = tmp_path / "target.status.json"
    status.write_text(json.dumps({"limit_mb": 512, "ts": time.time()}))
    ledger.set_status_path("target", status)
    assert ac.limit_mb("target") == 512
    ac.limits = lambda server: 777 if server == "target" else None
    assert ac.limit_mb("target") == 777
    assert AdmissionController(DataSettings.from_dict({"memory": {"limit_kind": "none"}})).limit_mb("x") == 0


def test_admit_remote_count_first():
    ac = controller(8000)
    assert ac.admit_remote(1000, 500, 2_000_000) == 500_000
    assert ac.admit_remote(None, 500, 10) is None and ac.admit_remote(10, None, 10) is None
    with pytest.raises(GatewayError) as exc:
        ac.admit_remote(2_000_000, 1200, 50 * MB, tool="mcp__clinicaltrials__get_clinical_data",
                        table="cbioportal.clinical_sample")
    err = exc.value
    assert err.kind is ErrorKind.too_large and err.subkind == "remote_size"
    assert err.payload["need_bytes"] == 2_400_000_000 and err.payload["cap_bytes"] == 50 * MB
    assert err.payload["total"] == 2_000_000 and err.payload["est_row_bytes"] == 1200
    assert err.tool == "mcp__clinicaltrials__get_clinical_data"


# --------------------------------------------------------------------------- crash classification


def test_crash_classification():
    for text in ("MemoryError", "Unable to allocate 2.00 GiB for an array with shape (268435456,)",
                 "pyarrow.lib.ArrowMemoryError: malloc of size 123 failed", "std::bad_alloc",
                 "OSError: [Errno 12] Cannot allocate memory"):
        assert classify_error_text(text) == "oom", text
    assert classify_error_text("Target X not found") is None and classify_error_text(None) is None

    tail = ('serving...\nVBT_CHILD_EXIT {"code": 1, "maxrss_kb": 10, "pid": 5, "reason": "exit_code", '
            '"signal": null}\n===== restarted =====\nVBT_CHILD_EXIT {"code": null, "maxrss_kb": 900000, '
            '"pid": 6, "reason": "signal", "signal": 9}\n')
    assert parse_exit_marker(tail)["pid"] == 6
    assert parse_exit_marker("no marker") is None and parse_exit_marker('VBT_CHILD_EXIT {bad') is None
    assert is_memory_exit({"signal": 9}) and is_memory_exit({"code": 137}) and is_memory_exit(
        {"reason": "memory_limit", "code": 1})
    assert not is_memory_exit({"code": 1, "signal": None, "reason": "exit_code"}) and not is_memory_exit(None)

    d = crash_decision("McpError: Connection closed", tail, server="genetics", tool="get_variant_info")
    assert d.oom and not d.retry and d.error.kind is ErrorKind.oom and d.error.subkind == "oom_killed"
    assert d.error.tool == "mcp__genetics__get_variant_info" and d.error.payload["exit"]["signal"] == 9

    plain = 'VBT_CHILD_EXIT {"code": 1, "maxrss_kb": 10, "pid": 5, "reason": "exit_code", "signal": null}\n'
    d = crash_decision("McpError: Connection closed", plain, server="s", tool="t")
    assert not d.oom and d.retry and d.error.kind is ErrorKind.server_crashed
    d = crash_decision("McpError: Connection closed", None, server="s", tool="t", attempt=1)
    assert not d.oom and not d.retry and d.error.retryable == "once"
    # a MemoryError logged earlier by a tool that survived it (more output followed) is not a crash cause
    d = crash_decision("EndOfStream", "Traceback ...\nMemoryError\nINFO handled request 7\n" + plain,
                       server="s", tool="t")
    assert not d.oom
    # RLIMIT_DATA fails the allocation instead of killing: an uncaught MemoryError (rc 1) or a C++
    # std::bad_alloc (SIGABRT) that ends the process is oom, never retried (INV-1)
    d = crash_decision("EndOfStream", "Traceback ...\nMemoryError\n" + plain, server="s", tool="t")
    assert d.oom and not d.retry and d.error.kind is ErrorKind.oom
    abrt = 'VBT_CHILD_EXIT {"code": null, "maxrss_kb": 10, "pid": 5, "reason": "signal", "signal": 6}\n'
    d = crash_decision("Connection closed", "terminate called after throwing an instance of 'std::bad_alloc'\n"
                       "  what():  std::bad_alloc\n" + abrt, server="s", tool="t")
    assert d.oom and not d.retry
    # a clean exit is never a memory death, whatever was logged
    clean = 'VBT_CHILD_EXIT {"code": 0, "maxrss_kb": 10, "pid": 5, "reason": "exit_code", "signal": null}\n'
    assert not crash_decision("EndOfStream", "MemoryError\n" + clean, server="s", tool="t").oom
