"""Per-server table faults (§13): with a table deleted from a fixture copy, every bound tool that reads
it is ``not_ready`` (or an error), never ``ok`` or ``empty``.

The local tables the bound tools read are deleted in groups chosen so that no tool reads two tables of
the same group (deleting a group therefore attributes each tool's failure to exactly one table), and
the data child's ``--check`` runs once per group. For each table, every tool whose call reads it, with
its argument-dependent reads (``ReadSpec.when``) switched on, must be unready. One group is then served
live: the unmodified upstream servers start behind the gateway and every affected tool whose arguments
can be filled from the descriptors' sentinels is called; each call must fail.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import DataEnv, call, harness_config, needs_arrow, needs_fastmcp, start_bridge
from vbt import preflight

pytestmark = needs_arrow

OT_SERVERS = ("target", "disease", "drug", "association", "genetics", "interaction", "pathway", "expression",
              "functional_genomics")
#: A value of each identifier kind for the live calls (the descriptors' sentinels and fixture keys).
SAMPLE = {"ensembl_gene": "ENSG00000169174", "hgnc_symbol": "PCSK9", "ot_disease": "MONDO_0005148",
          "disease_name": "type 2 diabetes mellitus", "chembl_molecule": "CHEMBL25", "drug_name": "aspirin",
          "tahoe_drug": "Bortezomib", "depmap_cell_line": "ACH-000681", "go_term": "GO:0008150",
          "hpo": "HP_0001250", "gwas_study": "GCST90000001", "rsid": "rs123", "reactome_pathway": "R-HSA-1",
          "so_term": "SO:0001583", "study_locus_id": "abc", "ot_variant": "1_1000_A_G", "biosample": "UBERON_0002048"}


def _config(tmp_path: Path, ot: Path, tahoe: Path) -> dict[str, Any]:
    cfg = harness_config(gateway=True, tmp_path=tmp_path, env=DataEnv(ot_root=ot, tahoe_root=tahoe))
    cfg["tool_env"] = {"OPEN_TARGETS_DATA_PATH": str(ot), "TAHOE_DATA_PATH": str(tahoe),
                       "VBT_ZENODO_DIR": os.environ.get("VBT_ZENODO_DIR", "")}
    cfg["mcp_servers"]["servers"] = [s for s in cfg["mcp_servers"]["servers"] if s["name"] in OT_SERVERS]
    return cfg


def _plan(catalog: Any) -> tuple[dict[str, list[tuple[str, str, dict[str, Any]]]], list[list[str]]]:
    """``({physical table: [(server, tool, args)]}, groups)``: the calls that read each local table (args switch
    on its conditional reads) and the deletion groups (no call reads two tables of one group)."""
    readers: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for server in OT_SERVERS:
        for tool in catalog.tools(server):
            contract = catalog.contract(server, tool)
            b = contract.binding
            if b is None or b.serve == "block" or b.hidden:
                continue
            for ref in contract.tables:
                t = catalog.table(ref)
                if t.descriptor.kind != "local" or t.layout == "upstream_only":
                    continue
                rs = b.reads.get(ref)
                args = {k: (v if not isinstance(v, dict) else "x") for k, v in ((rs.when if rs else None) or {}).items()}
                phys = str(t.physical)
                if (server, tool, args) not in readers.setdefault(phys, []):
                    readers[phys].append((server, tool, args))
    tools_of = {phys: {(s, t) for s, t, _ in calls} for phys, calls in readers.items()}
    groups: list[list[str]] = []
    for phys in sorted(readers):
        for g in groups:
            if not any(tools_of[phys] & tools_of[other] for other in g):
                g.append(phys)
                break
        else:
            groups.append([phys])
    return readers, groups


def _delete(ot: Path, tahoe: Path, catalog: Any, refs: list[str]) -> None:
    import shutil

    for ref in refs:
        t = catalog.table(ref)
        root = ot if ref.startswith("open_targets.") else tahoe
        path = root / t.physical_spec.path
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


@pytest.fixture(scope="module")
def faults(ot_root: Path, tahoe_root: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    import dl_fixtures as F
    from vbt.datalayer.gateway.readiness import call_readiness

    base = tmp_path_factory.mktemp("faults-base")
    _settings, catalog, _registry = preflight.data_catalog(_config(base, ot_root, tahoe_root))
    readers, groups = _plan(catalog)
    outcome: dict[tuple[str, str, str], dict[str, Any]] = {}
    roots: list[tuple[Path, Path]] = []
    for i, group in enumerate(groups):
        d = tmp_path_factory.mktemp(f"faults-{i}")
        ot = F.copy_fixture(ot_root, d / "ot" / ot_root.name)
        tahoe = F.copy_fixture(tahoe_root, d / "tahoe")
        _delete(ot, tahoe, catalog, group)
        roots.append((ot, tahoe))
        cfg = _config(d / "cfg", ot, tahoe)
        dr, cache, cat = preflight.load_data_readiness(cfg)
        for phys in group:
            for server, tool, args in readers[phys]:
                contract = cat.contract(server, tool)
                # as the gateway decides a call: the bound table after selector resolution
                r = call_readiness(contract, cache, bound_table=contract.selected_table(args), args=args)
                outcome[(phys, server, tool)] = {"ready": r.ready, "reasons": r.reasons, "unchecked": r.unchecked,
                                                 "args": args, "group": i}
    return {"readers": readers, "groups": groups, "outcome": outcome, "roots": roots, "catalog": catalog}


def test_groups_attribute_each_failure_to_one_table(faults: dict[str, Any]) -> None:
    readers, groups = faults["readers"], faults["groups"]
    assert sum(len(g) for g in groups) == len(readers) >= 30
    for g in groups:
        tools = [(s, t) for phys in g for s, t, _ in readers[phys]]
        assert len(tools) == len(set(tools)), g


def test_every_reader_of_a_deleted_table_is_not_ready(faults: dict[str, Any]) -> None:
    bad = []
    for (phys, server, tool), r in sorted(faults["outcome"].items()):
        named = any(x["name"] == phys for x in r["reasons"])
        if r["ready"] or not named:
            bad.append(f"{server}.{tool}{r['args'] or ''} with {phys} deleted: ready={r['ready']} "
                       f"reasons={[x['name'] for x in r['reasons']]} unchecked={r['unchecked']}")
    assert not bad, "\n".join(bad)


@needs_fastmcp
def test_deleted_table_calls_fail_live(faults: dict[str, Any], tmp_path: Path) -> None:
    """The group holding ``known_drug``, served by the real upstream servers behind the gateway."""
    catalog = faults["catalog"]
    i = next(n for n, g in enumerate(faults["groups"]) if "open_targets.known_drug" in g)
    ot, tahoe = faults["roots"][i]
    calls = []
    for phys in faults["groups"][i]:
        for server, tool, args in faults["readers"][phys]:
            filled = _fill_args(catalog, server, tool, args)
            if filled is not None:
                calls.append((phys, server, tool, filled))
    servers = sorted({s for _, s, _, _ in calls})
    assert calls and "drug" in servers

    async def run() -> list[Any]:
        bridge = await start_bridge(servers, env=DataEnv(ot_root=ot, tahoe_root=tahoe), gateway=True,
                                    tmp_path=tmp_path)
        try:
            await bridge.gateway.wait_readiness(600)
            return [await call(bridge, s, t, a) for _, s, t, a in calls]
        finally:
            await bridge.gateway.aclose()
            await bridge.aclose()

    results = asyncio.run(run())
    bad = [f"{s}.{t}({a}) with {phys} deleted: {r.status} {r.text[:200]}"
           for (phys, s, t, a), r in zip(calls, results) if not r.is_error and not _marked_partial(r)]
    assert not bad, "\n".join(bad)
    assert any(r.kind == "not_ready" for r in results), [r.kind for r in results]


def _marked_partial(r: Any) -> bool:
    """An honest partial answer: a view whose section over the deleted table is marked unavailable."""
    obj = r.obj if isinstance(r.obj, dict) else {}
    return (r.status or r.header.get("status")) == "partial" and \
        any(isinstance(v, dict) and v.get("_vbt_unavailable") for v in obj.values())


def _fill_args(catalog: Any, server: str, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """The call's arguments: its conditional-read switches plus a sentinel value for each identifier argument
    of the bound table (None when a required-looking argument cannot be filled)."""
    contract = catalog.contract(server, tool)
    out = dict(args)
    for name, a in contract.identifier_args.items():
        kinds = [k.split(":")[-1] for k in a.accepts]
        for table, column in contract.arg_columns(name):
            spec = contract.tables.get(table)
            col = getattr(getattr(spec, "spec", None), "columns", {}).get(column.split(".")[0]) if spec else None
            if getattr(col, "id_type", None):
                kinds.append(str(col.id_type).split(":")[-1])
        value = next((SAMPLE[k] for k in kinds if k in SAMPLE), None)
        if value is None:
            continue
        out.setdefault(name, [value] if a.op == "in" else value)
        if contract.bound_table and contract.arg_columns(name) and len(out) > len(args):
            break
    for group in contract.binding.require_any:
        if not any(g in out for g in group):
            return None
    return out if out or not contract.identifier_args else None
