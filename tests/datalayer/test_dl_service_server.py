"""The data child's entry point (service/server.py; §4, §11.8, §13 preflight integration).

The server is launched exactly as the gateway launches it: ``<python> -E .../service/server.py`` with
the settings in ``VBT_DATA_SETTINGS`` and without ``PYTHONPATH``, so the ``sys.path`` bootstrap is
what makes ``vbt`` importable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import yaml  # noqa: E402

from vbt.datalayer.ipc import PHASE1_VERBS, REQUEST_ARG, WitnessRequest, parse_response, request_payload  # noqa: E402
from vbt.datalayer.service.verbs.public import PUBLIC_VERBS  # noqa: E402
from vbt.datalayer.settings import SETTINGS_ENV, DataSettings  # noqa: E402

SERVER = Path(__file__).resolve().parents[2] / "src" / "vbt" / "datalayer" / "service" / "server.py"
HAVE_FASTMCP = importlib.util.find_spec("fastmcp") is not None


@pytest.fixture
def data_env(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "data"
    (root / "t").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": f"r{i}", "score": i / 10} for i in range(5)]),
                   root / "t" / "part-00000.parquet")
    desc = {"schema": "vbt.datasource/1", "source": "s", "title": "s", "root": str(root),
            "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
            "tables": {"t": {"kind": "fact", "path": "t", "grain": "row", "key": {"columns": ["id"]},
                             "columns": {"id": {"role": "identifier"},
                                         "score": {"role": "measure", "statistic": "score_0_1"}}},
                       "gone": {"kind": "fact", "path": "gone", "grain": "row", "key": {"columns": ["id"]},
                                "columns": {"id": {"role": "identifier"}}}}}
    (tmp_path / "sources").mkdir()
    (tmp_path / "overlays").mkdir()
    (tmp_path / "sources" / "s.yaml").write_text(yaml.safe_dump(desc))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp_path / "sources"),
                                       "overlays_dir": str(tmp_path / "overlays"),
                                       "cache_dir": str(tmp_path / "cache")}, project_root=tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env[SETTINGS_ENV] = settings.to_json()
    return env


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-E", str(SERVER), *args], env=env, capture_output=True, text=True,
                          timeout=120, cwd="/")


def test_check_json_prints_valid_json(data_env):
    out = _run(data_env, "--check", "--json")
    assert out.returncode == 0, out.stderr
    resp = parse_response("_check", out.stdout)
    assert resp.tables["s.t"].status == "ready" and resp.tables["s.gone"].status == "missing"
    assert resp.hash_randomization in (0, 1)
    one = parse_response("_check", _run(data_env, "--check", "--json", "--table", "s.t", "--depth", "shallow").stdout)
    assert list(one.tables) == ["s.t"] and one.depth == "shallow"


def test_list_verbs_and_build_index_cli(data_env):
    out = _run(data_env, "--list-verbs")
    assert out.returncode == 0 and set(PHASE1_VERBS) <= set(json.loads(out.stdout))
    bad = _run(data_env, "--build-index")
    assert bad.returncode == 2 and "--id-type" in bad.stderr
    none = _run(data_env, "--build-index", "--access-paths")
    assert none.returncode == 0 and json.loads(none.stdout) == {"built": [], "errors": {}}


@pytest.mark.needs_fastmcp
@pytest.mark.skipif(not HAVE_FASTMCP, reason="fastmcp is needed to launch the data child")
async def test_launch_with_dash_E_lists_the_hidden_verbs_and_serves_them(data_env):
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    transport = StdioTransport(command=sys.executable, args=["-E", str(SERVER)], env=data_env, cwd="/")
    async with Client(transport) as client:
        tools = {t.name: t for t in await client.list_tools()}
        public = {name for name in tools if not name.startswith("_")}
        assert set(PHASE1_VERBS) <= set(tools) and public == set(PUBLIC_VERBS)
        schema = getattr(tools["_witness"], "input_schema", None) or tools["_witness"].inputSchema
        assert list(schema["properties"]) == [REQUEST_ARG]
        payload = request_payload(WitnessRequest(table="s.t", predicate={"ge": ["score", 0.2]},
                                                 order=[{"column": "score", "direction": "desc"}], k=2))
        result = await client.call_tool("_witness", payload)
        data = result.structured_content if result.structured_content is not None else \
            json.loads(result.content[0].text)
        w = parse_response("_witness", data)
        assert (w.total, w.topk) == (3, [["r4"], ["r3"]])
        text = await client.call_tool("_witness", {REQUEST_ARG: json.dumps({"table": "s.t"})})
        assert parse_response("_witness", text.structured_content).total == 5
        # a public verb takes its own arguments (the payload), open to the agent the gateway adds
        find = getattr(tools["find"], "input_schema", None) or tools["find"].inputSchema
        assert {"table", "where", "limit"} <= set(find["properties"]) and find["additionalProperties"] is True
        got = await client.call_tool("find", {"table": "s.t", "where": {"score": {"ge": 0.2}}, "limit": 2,
                                              "rank_by": "score desc", "agent": "someone"})
        body = got.structured_content
        assert body["_vbt"]["total"] == 3 and [r["id"] for r in body["rows"]] == ["r4", "r3"], body
