"""Determinism (§21, rev 2): results do not depend on the harness's hash seed or on the server launch.

* The data child's ``_serve`` and the gateway's transforms, run on the S8 interaction fixture (an
  undirected edge table, two source databases, ranked per source) in two harness processes with
  ``PYTHONHASHSEED`` 1 and 2, give byte-identical headers, rows and ``row_keys_sha256``.
* The reaper-launched upstream ``interaction`` server, started twice, returns the identical raw result
  (the reaper pins ``PYTHONHASHSEED=0`` in the child despite ``-E``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import HAVE_ARROW, needs_arrow, needs_fastmcp

pytestmark = [needs_arrow, needs_fastmcp]

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

#: Header fields that differ between runs by construction: ``prov`` is the provenance record's id, a hash
#: over the record including its timings and retrieval time. Everything else must be byte-identical.
VOLATILE = {"prov"}

_PROBE = r"""
import asyncio, json, sys
from pathlib import Path
src, here, ot, tmp = sys.argv[1:5]
sys.path[:0] = [src, here]
from dl_upstream import DataEnv, call, start_bridge
from vbt.datalayer.predicate import Eq, Or, to_json
from vbt.datalayer.service import ServiceContext
from vbt.datalayer.service.verbs import load_verbs
from vbt.datalayer.settings import DataSettings

PCSK9 = "ENSG00000169174"
VOLATILE = set(sys.argv[5].split(","))

def scrub(obj):
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj

out = {}
ctx = ServiceContext(DataSettings.from_dict({"cache_dir": str(Path(tmp) / "child-cache")}, project_root=None,
                                            variables={}))
serve = load_verbs()["_serve"]
out["serve"] = serve(ctx, {"table": "open_targets.interaction", "verb": "find",
                           "predicate": to_json(Or([Eq("targetA", PCSK9), Eq("targetB", PCSK9)])),
                           "order": [{"column": "scoring", "direction": "desc", "within": ["sourceDatabase"]},
                                     {"column": "targetA", "direction": "asc"},
                                     {"column": "targetB", "direction": "asc"}],
                           "limit": 3, "group_by": ["sourceDatabase"]})

async def gateway():
    bridge = await start_bridge(["interaction"], env=DataEnv(ot_root=Path(ot)), gateway=True,
                                tmp_path=Path(tmp) / "bridge")
    try:
        await bridge.gateway.wait_readiness(600)
        res = {}
        for label, args in (("by_symbol", {"target_id": "PCSK9", "limit": 3}),
                            ("by_id_source", {"target_id": PCSK9, "source_database": "intact", "limit": 2})):
            r = await call(bridge, "interaction", "get_interactions", args)
            prov = r.provenance.to_dict() if r.provenance is not None else {}
            res[label] = {"is_error": r.is_error, "header": scrub(r.header), "rows": scrub(r.rows("interactions")),
                          "row_keys_sha256": ((prov.get("result") or {}).get("row_keys_sha256")),
                          "row_keys": ((prov.get("result") or {}).get("row_keys"))}
        raw = await bridge.call_raw("interaction", "get_interactions", {"target_id": PCSK9, "limit": 50})
        res["raw"] = getattr(raw, "text", None) if not isinstance(raw, str) else raw
        return res
    finally:
        await bridge.gateway.aclose()
        await bridge.aclose()

out["gateway"] = asyncio.run(gateway())
print("VBT_DETERMINISM" + json.dumps(out, sort_keys=True, default=str))
"""


@pytest.fixture(scope="module")
def s8_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The OT fixture with the S8 interaction table over two source databases (IntAct and STRING)."""
    if not HAVE_ARROW:
        pytest.skip("pyarrow and pandas are needed for the fixtures")
    import dl_fixtures as F
    return F.build_ot_fixture(tmp_path_factory.mktemp("s8") / "25.09", interaction_sources=("intact", "string"))


def _probe(seed: int, ot: Path, tmp: Path) -> dict[str, Any]:
    env = dict(os.environ, PYTHONHASHSEED=str(seed), OPEN_TARGETS_DATA_PATH=str(ot), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", _PROBE, str(REPO / "src"), str(HERE), str(ot), str(tmp),
                           ",".join(sorted(VOLATILE))], capture_output=True, text=True, timeout=900, env=env, cwd=REPO)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("VBT_DETERMINISM")), None)
    assert proc.returncode == 0 and line is not None, proc.stderr[-4000:]
    return json.loads(line[len("VBT_DETERMINISM"):])


@pytest.fixture(scope="module")
def runs(s8_root: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], dict[str, Any]]:
    return (_probe(1, s8_root, tmp_path_factory.mktemp("seed1")),
            _probe(2, s8_root, tmp_path_factory.mktemp("seed2")))


def test_serve_is_independent_of_the_hash_seed(runs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    one, two = runs
    assert one["serve"]["rows"], one["serve"]
    assert {r.get("sourceDatabase") for r in one["serve"]["rows"]} == {"intact", "string"}
    assert json.dumps(one["serve"], sort_keys=True) == json.dumps(two["serve"], sort_keys=True)


def test_gateway_results_are_independent_of_the_hash_seed(runs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    one, two = runs
    for label in ("by_symbol", "by_id_source"):
        a, b = one["gateway"][label], two["gateway"][label]
        assert not a["is_error"] and a["rows"] and a["row_keys_sha256"], a
        assert json.dumps(a["header"], sort_keys=True) == json.dumps(b["header"], sort_keys=True)
        assert json.dumps(a["rows"], sort_keys=True) == json.dumps(b["rows"], sort_keys=True)
        assert a["row_keys_sha256"] == b["row_keys_sha256"] and a["row_keys"] == b["row_keys"]


def test_reaper_launched_server_returns_identical_results(runs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    one, two = runs
    assert one["gateway"]["raw"] and "interactions" in one["gateway"]["raw"]
    assert one["gateway"]["raw"] == two["gateway"]["raw"]
