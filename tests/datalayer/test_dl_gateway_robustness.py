"""Gateway robustness (review findings R5, R6, R7, INV-1): resolver index builds that fail on an
outage are retried, readiness checks are single-flight, and observe mode adds no latency and no side
effects to the call it watches."""

from __future__ import annotations

import asyncio
import copy
import time

import pytest

from test_dl_gateway_flow import DRUG_OVERLAY, PCSK9, world
from vbt.datalayer.gateway import gateway as gwmod
from vbt.datalayer.gateway.service_client import ServiceError

pytest.importorskip("pydantic")


def _with_output_arg() -> dict:
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs"]["args"]["out"] = {"role": "output_path"}
    return ov


async def test_observe_mode_never_renames_an_existing_output(tmp_path):
    """INV-1: observe mode confines and records an output path, but a file earlier claims cite is kept."""
    gw = world(tmp_path, overlay=_with_output_arg(), data={"gateway": {"mode": "observe"}})
    cited = tmp_path / "out" / "cited.csv"
    cited.write_text("a,b\n1,2\n")
    plan = await gw.prepare("drug", "search_known_drugs", {"target_id": PCSK9, "out": "cited.csv"}, None)
    assert plan.args_sent["out"] == "cited.csv"
    assert cited.read_text() == "a,b\n1,2\n"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["cited.csv"]
    # enforce mode keeps it write-once as before
    gw = world(tmp_path / "e", overlay=_with_output_arg())
    (tmp_path / "e" / "out" / "cited.csv").write_text("x\n")
    await gw.prepare("drug", "search_known_drugs", {"target_id": PCSK9, "out": "cited.csv"}, None)
    assert len(list((tmp_path / "e" / "out").glob("cited.*.csv"))) == 1


async def test_observe_mode_builds_no_index_and_fetches_no_vocab_in_the_call(tmp_path):
    """R6: an index that is not built yet is noted, never built inside the agent's call."""
    gw = world(tmp_path, data={"gateway": {"mode": "observe"}})
    plan = await gw.prepare("drug", "search_known_drugs", {"target_id": "PCSK9"}, None)
    assert plan.args_sent == {"target_id": "PCSK9"}
    assert not gw.service.verbs("_build_index") and not gw.service.verbs("_vocab")
    assert any(e[0] == "data_observe" for e in gw.bridge.events)
    await asyncio.gather(*gw._index_builds.values())          # built off the call path for later calls
    assert gw.service.verbs("_build_index")
    # enforce builds it on first use
    gw = world(tmp_path / "e")
    await gw.prepare("drug", "search_known_drugs", {"target_id": "PCSK9"}, None)
    assert gw.service.verbs("_build_index")


async def test_index_build_outage_is_retried_and_rejection_is_not(tmp_path):
    """R5: one transient failure does not disable an id_type for the session."""
    gw = world(tmp_path)
    gw.service.fail = {"_build_index"}
    assert not await gw._ensure_index("open_targets:ensembl_gene")
    assert not await gw._ensure_index("open_targets:ensembl_gene")          # within the back-off
    assert len(gw.service.verbs("_build_index")) == 1
    gw.service.fail = set()
    gw._index_retry_at["open_targets:ensembl_gene"] = time.monotonic() - 1  # the back-off has passed
    assert await gw._ensure_index("open_targets:ensembl_gene")
    assert "open_targets:ensembl_gene" not in gw._index_failed

    # a relisted (restarted) data child clears transient failures at once
    gw = world(tmp_path / "b")
    gw.service.fail = {"_build_index"}
    assert not await gw._ensure_index("open_targets:ensembl_gene")
    gw.service.fail = set()
    gw.rewrite_listing("data", "_check", "", {})
    assert await gw._ensure_index("open_targets:ensembl_gene")

    # a rejection (configuration fault) stays cached
    gw = world(tmp_path / "c")

    async def rejected(*a, **k):
        raise ServiceError("bad descriptor", verb="_build_index", subkind="rejected")

    gw.service.build_index = rejected  # type: ignore[method-assign]
    assert not await gw._ensure_index("open_targets:ensembl_gene")
    gw._forget_transient_index_failures()
    assert "open_targets:ensembl_gene" in gw._index_failed


async def test_concurrent_readiness_refreshes_share_one_check(tmp_path, monkeypatch):
    """R7: parallel calls on one unchecked table send one ``_check``; a failure is briefly cached."""
    gw = world(tmp_path)
    results = await asyncio.gather(*[gw.refresh_readiness(["open_targets.target"], "standard") for _ in range(8)])
    assert all(results) and len(gw.service.verbs("_check")) == 1
    gw.service.fail = {"_check"}
    results = await asyncio.gather(*[gw.refresh_readiness(["open_targets.known_drug"]) for _ in range(5)])
    assert not any(results) and len(gw.service.verbs("_check")) == 2
    assert not await gw.refresh_readiness(["open_targets.known_drug"])     # negative-cached
    assert len(gw.service.verbs("_check")) == 2
    monkeypatch.setattr(gwmod, "CHECK_FAILURE_TTL_S", 0.0)
    gw._check_failed_until.clear()
    gw.service.fail = set()
    assert await gw.refresh_readiness(["open_targets.known_drug"])


async def test_observe_mode_keeps_the_plain_crash_policy(tmp_path):
    """R6: under observe, a memory kill is traced and retried once, as without a gateway."""
    gw = world(tmp_path, data={"gateway": {"mode": "observe"}})
    plan = await gw.prepare("drug", "search_known_drugs", {"target_id": PCSK9}, None)
    d = await gw.on_crash("drug", plan, "connection closed", 'VBT_CHILD_EXIT {"signal": 9, "code": null}')
    assert d.retry and not d.oom
    assert any(e[0] == "data_observe" and e[1]["decision"] == "would_oom" for e in gw.bridge.events)
    d2 = await gw.on_crash("drug", plan, "connection closed", 'VBT_CHILD_EXIT {"signal": 9, "code": null}')
    assert not d2.retry
