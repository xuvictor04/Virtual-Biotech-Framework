"""Deterministic networks over an edges table (§6.8 S8, §10.6; phase 3, F17).

A small Open Targets-shaped interaction table: IntAct and STRING both report the A-B pair (two edges:
sources are never pooled), a two-hop neighbourhood, and an IntAct row whose second interactor maps to
no gene (``targetB`` null, ``missing: non_entity``).

* the same network under different ``PYTHONHASHSEED`` values (upstream's set iteration gave 2, 2, 4,
  4, 3, 4 edges over six runs);
* edges are keyed by the full key (source database included); null endpoints are excluded and counted;
* ``max_nodes`` cuts the last hop in key order and the result is ``partial`` with frontier counts;
* ``neighbors`` with ``hops`` > 1 and the derived ``get_interaction_network`` and
  ``find_common_interactors`` (per-source partners; ``min_confidence`` other than 0 refused).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import REPO, needs_arrow

pytestmark = needs_arrow

PCSK9 = "ENSG00000169174"
N = {c: f"ENSG000009990{i:02d}" for i, c in enumerate("ABCDEFGH", start=1)}
N["A"] = PCSK9                                         # the descriptor's readiness sentinel is a node
EDGES = [  # (source, a, b, score)
    ("intact", "A", "B", 0.9), ("intact", "A", "C", 0.5), ("intact", "B", "D", 0.7), ("intact", "C", "E", 0.6),
    ("intact", "D", "F", 0.4), ("string", "A", "B", 0.95), ("string", "B", "G", 0.8), ("intact", "A", None, 0.3),
]


def build(root: Path) -> Path:
    import dl_fixtures as F

    ot = root / "ot" / "25.09"
    sp = {"mnemonic": "human", "scientific_name": "Homo sapiens", "taxon_id": 9606}
    rows = [{"sourceDatabase": s, "targetA": N[a], "targetB": N[b] if b else None, "intA": f"P{a}{s}",
             "intB": f"P{b or 'X'}{s}", "intABiologicalRole": "unspecified role", "intBBiologicalRole": "unspecified role",
             "speciesA": sp, "speciesB": sp, "count": 1, "scoring": score} for s, a, b, score in EDGES]
    F.write_table(ot, "interaction", F.table("interaction", rows))
    targets = [{"id": g, "approvedSymbol": f"SYM{c}" if g != PCSK9 else "PCSK9", "biotype": "protein_coding",
                "approvedName": c,
                "pathways": [{"pathwayId": "R-HSA-1", "pathway": "p", "topLevelTerm": "t"}] if g == PCSK9 else None,
                "go": [{"id": "GO:0000001", "source": "x", "evidence": "IDA", "aspect": "P", "geneProduct": "P1",
                        "ecoId": "ECO"}] if g == PCSK9 else None} for c, g in N.items()]
    F.write_table(ot, "target", F.table("target", targets))
    return ot


def make_ctx(ot: Path, cache: Path) -> Any:
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    os.environ["OPEN_TARGETS_DATA_PATH"] = str(ot)
    settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                       "overlays_dir": str(REPO / "configs" / "data" / "overlays"),
                                       "cache_dir": str(cache)}, project_root=REPO)
    return ServiceContext(settings)


@pytest.fixture(scope="module")
def ot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build(tmp_path_factory.mktemp("network"))


@pytest.fixture(scope="module")
def ctx(ot: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(ot))
        yield make_ctx(ot, tmp_path_factory.mktemp("network-cache"))


def call(ctx: Any, name: str, /, **payload: Any) -> dict[str, Any]:
    from vbt.datalayer.service.verbs import load_verbs

    out = load_verbs()[name](ctx, payload)
    json.dumps(out)
    return out


def ok(out: dict[str, Any]) -> dict[str, Any]:
    assert out.get("status") != "tool_error", json.dumps(out)[:1500]
    return out


def pair(e: dict[str, Any]) -> tuple[str, str, str]:
    inv = {v: k for k, v in N.items()}
    return e["sourceDatabase"], inv[e["targetA"]], inv[e["targetB"]]


# ---------------------------------------------------------------------------- traversal


def test_two_hops_keep_sources_apart_and_exclude_null_endpoints(ctx) -> None:
    out = ok(call(ctx, "neighbors", table="open_targets.interaction", node=N["A"], hops=2))
    edges = {pair(e) for e in out["rows"]}
    assert edges == {("intact", "A", "B"), ("string", "A", "B"), ("intact", "A", "C"), ("intact", "B", "D"),
                     ("intact", "C", "E"), ("string", "B", "G")}
    hops = {n["node"]: n["hop"] for n in out["nodes"]}
    assert hops == {N["A"]: 0, N["B"]: 1, N["C"]: 1, N["D"]: 2, N["E"]: 2, N["G"]: 2}
    h = out["_vbt"]
    assert h["excluded_unknown"] == {"targetB": 1}, "the unmapped interactor is excluded and counted"
    rec = h["network"]
    assert rec["hops"] == 2 and rec["nodes"] == 6 and rec["edges"] == 6 and rec["node_set"].startswith("sha256:")
    assert [f["expanded"] for f in rec["frontier"]] == [1, 2]
    # one hop from one node keeps the phase-2 answer (partner per row)
    one = ok(call(ctx, "neighbors", table="open_targets.interaction", node="PCSK9"))
    assert {N["B"], N["C"]} <= {r["partner"] for r in one["rows"]} and "network" not in one["_vbt"]


def test_truncation_is_partial_in_key_order(ctx) -> None:
    out = ok(call(ctx, "neighbors", table="open_targets.interaction", node=N["A"], hops=2, max_nodes=4))
    nodes = {n["node"] for n in out["nodes"]}
    assert nodes == {N["A"], N["B"], N["C"], N["D"]}, "the first candidate in key order (D < E < G) is kept"
    h = out["_vbt"]
    assert h["status"] == "partial" and h["truncated"] is True
    assert h["network"]["frontier"][-1] == {"hop": 2, "expanded": 2, "added": 1, "not_added": 2}
    assert all({e["targetA"], e["targetB"]} <= nodes for e in out["rows"])


def test_hops_and_nodes_are_checked(ctx) -> None:
    bad = call(ctx, "neighbors", table="open_targets.interaction", node=N["A"], hops=9)
    assert bad["status"] == "tool_error" and bad["kind"] == "invalid_argument"
    multi = ok(call(ctx, "neighbors", table="open_targets.interaction", nodes=["SYMD", "SYMG"], hops=1))
    assert {pair(e) for e in multi["rows"]} == {("intact", "B", "D"), ("intact", "D", "F"), ("string", "B", "G")}


_SCRIPT = """
import json, os, sys
sys.path[:0] = [{src!r}, {tests!r}]
from test_dl_network import make_ctx
from pathlib import Path
from vbt.datalayer.service.verbs import load_verbs
ctx = make_ctx(Path({ot!r}), Path({cache!r}))
out = load_verbs()["neighbors"](ctx, {{"table": "open_targets.interaction", "node": {node!r}, "hops": 3,
                                       "max_nodes": 5}})
print(json.dumps({{"rows": out["rows"], "nodes": out["nodes"], "net": out["_vbt"]["network"]}}, sort_keys=True))
"""


def test_the_network_does_not_depend_on_the_hash_seed(ot, tmp_path) -> None:
    script = _SCRIPT.format(src=str(REPO / "src"), tests=str(REPO / "tests" / "datalayer"), ot=str(ot),
                            cache=str(tmp_path / "cache"), node=N["A"])
    outs = set()
    for seed in ("0", "1", "2", "17", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120)
        assert res.returncode == 0, res.stderr[-2000:]
        outs.add(res.stdout.strip().splitlines()[-1])
    assert len(outs) == 1, "the same call gives the same network whatever the hash seed"
    net = json.loads(outs.pop())
    assert net["net"]["truncated"] is True and len(net["nodes"]) == 5


# ---------------------------------------------------------------------------- derived tools


async def test_interaction_network_tools_are_derived(ctx, tmp_path) -> None:
    from test_dl_native_tools import _derived, _gateway

    from vbt.datalayer.errors import GatewayError

    gw = _gateway(ctx, tmp_path)
    res = await _derived(gw, "interaction", "get_interaction_network", {"seed_targets": ["PCSK9"], "max_hops": 1})
    obj = res.obj
    assert {pair(e) for e in obj["edges"]} == {("intact", "A", "B"), ("string", "A", "B"), ("intact", "A", "C")}
    assert {n["target_id"] for n in obj["nodes"]} == {N["A"], N["B"], N["C"]}
    assert obj["network_size"] == 3 and obj["edge_count"] == 3 and obj["seed_targets"] == ["PCSK9"]
    assert obj["statistics"]["network"]["node_set"].startswith("sha256:")
    assert obj["statistics"]["max_degree"] == 3
    with pytest.raises(GatewayError) as e:
        await _derived(gw, "interaction", "get_interaction_network", {"seed_targets": ["PCSK9"], "min_confidence": 0.5})
    assert e.value.kind.value == "unsupported_filter"
    with pytest.raises(GatewayError) as e:                     # a family_param's bounds hold at the gateway
        await gw.prepare("interaction", "get_interaction_network", {"seed_targets": ["PCSK9"], "max_hops": 9}, None)
    assert e.value.kind.value == "invalid_argument" and e.value.envelope()["reason"] == "bounds"
    common = await _derived(gw, "interaction", "find_common_interactors",
                            {"target_ids": ["PCSK9", "SYMD"], "min_targets": 2})
    rows = common.obj["common_interactors"]
    assert [r["interactor_id"] for r in rows] == [N["B"]]
    assert rows[0]["connects_to"] == sorted([N["A"], N["D"]]) and rows[0]["connection_count"] == 2
    assert rows[0]["sources"]["intact"]["connects_to"] == sorted([N["A"], N["D"]])
    assert rows[0]["sources"]["string"] == {"connects_to": [N["A"]], "max_scoring": 0.95}
    assert "avg_confidence" not in rows[0] and common.obj["count"] == 1
