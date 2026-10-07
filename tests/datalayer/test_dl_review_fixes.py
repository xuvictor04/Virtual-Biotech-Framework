"""Regressions of the phase-1 review findings that the shipped catalog shows without live servers.

The gateway runs on the shipped descriptors and overlays (``configs/data``) with the in-memory data
child and fake bridge of :mod:`test_dl_gateway_flow`; upstream replies are canned. The enforce-mode
answers of the real upstream servers on the fixtures are in :mod:`test_dl_review_live`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from test_dl_gateway_flow import REGISTRY, FakeBridge, FakeService, call
from vbt.config import load_config
from vbt.datalayer.catalog import load_catalog
from vbt.datalayer.descriptor.overlay import ArgBinding
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.gateway import DataGateway
from vbt.datalayer.ipc import VocabResponse
from vbt.datalayer.resolve import IndexStore
from vbt.datalayer.settings import DataSettings, GatewaySettings

_CATALOG: list[Any] = []


def shipped_catalog() -> Any:
    if not _CATALOG:
        _CATALOG.append(load_catalog(load_config(), REGISTRY))
    return _CATALOG[0]


def shipped(tmp_path: Path, tables: dict[str, list[dict[str, Any]]] | None = None, *,
            upstream: Any = None, **data: Any) -> DataGateway:
    """A gateway over the shipped catalog with an in-memory data child holding ``tables``."""
    settings = DataSettings.from_dict({"cache_dir": str(tmp_path / "cache"), "gateway": {"mode": "enforce"}, **data},
                                      project_root=tmp_path)
    store = IndexStore(settings.cache_dir)
    service = FakeService(tables or {}, store=store)
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    gw = DataGateway(settings, shipped_catalog(), REGISTRY, service=service, index_store=store,
                     run={"mcp_output_dir": str(tmp_path / "out")})
    gw.bind_bridge(FakeBridge(upstream))
    return gw


def soma_vocab(gw: DataGateway, values: dict[str, list[Any]]) -> None:
    """Census vocabularies come from upstream's list_metadata_values."""
    gw.bridge.upstream = lambda tool, a: {"value_counts": [{"value": v, "count": 1}  # type: ignore[attr-defined]
                                                           for v in values.get(a.get("column_name"), [])]}


ASSOC = {"open_targets.association_by_datatype_direct": [
    {"targetId": "T1", "diseaseId": "D1", "datatypeId": "literature", "score": 0.9},
    {"targetId": "T2", "diseaseId": "D1", "datatypeId": "genetic_association", "score": 0.5}]}
BY_DATATYPE = {"datatype": "literature", "output_path": "a.parquet"}
EXPRESSION = {"open_targets.expression": [{"id": "ENSG00000169174", "tissues": [
    {"efo_code": "UBERON_0002107", "label": "liver", "rna": {"value": 120.0}}]}]}
TISSUES = {"num_tissues": 1, "tissues": [{"efo_code": "UBERON_0002107", "label": "liver"}]}


def vocab(gw: DataGateway, values: dict[str, list[Any]]) -> None:
    """Answer ``_vocab`` for any column whose name ends with a key of ``values``."""
    def answer(req: Any) -> VocabResponse:
        hit = next((v for k, v in values.items() if req.column.endswith(k)), [])
        return VocabResponse(values=hit, rendered=[str(v) for v in hit], complete=True)
    gw.service._vocab = answer  # type: ignore[attr-defined,method-assign]


async def refused(gw: DataGateway, server: str, tool: str, args: dict[str, Any]) -> GatewayError:
    with pytest.raises(GatewayError) as e:
        await gw.prepare(server, tool, args, None)
    return e.value


# ---------------------------------------------------------------------------- arguments (F2, SC1-SC4, SC7)


def test_text_matched_filter_is_free_text() -> None:
    """interpreted_as substring on a filter argument means a text match, never an exact filter (F2)."""
    b = ArgBinding.model_validate({"binds": "s.t.c", "interpreted_as": "casefold_substring"})
    assert b.role == "free_text"
    listed = ArgBinding.model_validate({"binds": "s.t.c", "interpreted_as": "casefold_substring", "op": "in",
                                        "each": True})
    assert listed.role == "filter"                     # a list keeps its role; its items are checked
    assert ArgBinding.model_validate({"binds": "s.t.c"}).role == "filter"
    cat = shipped_catalog()
    for server, tool, arg in [("genetics", "get_study_metadata", "trait"),
                              ("genetics", "query_regulatory_regions", "biosample"),
                              ("drug", "get_drug_mechanisms", "mechanism"),
                              ("target", "get_homologues", "species_filter")]:
        a = cat.contract(server, tool).args[arg]
        assert a.role in ("free_text", "unbound"), (server, tool, arg, a.role)


async def test_essie_wrap_cannot_be_closed_early(tmp_path: Path) -> None:
    """A ')' in a wrapped Essie argument would close the wrap and OR past the server's filters (SC1)."""
    gw = shipped(tmp_path, leakage={"ceiling": "2023-01-01"})
    for args in ({"condition": "lung cancer) OR (melanoma"}, {"condition": "c", "advanced_filter": "x) OR (y"}):
        e = await refused(gw, "clinicaltrials", "count_clinical_trials", args)
        assert e.kind == ErrorKind.invalid_argument and "unbalanced parentheses" in e.message
    plan = await gw.prepare("clinicaltrials", "count_clinical_trials",
                            {"condition": "lung cancer", "advanced_filter": "AREA[Phase]PHASE3 OR AREA[Phase]PHASE2"},
                            None)
    sent = plan.args_sent["advanced_filter"]
    assert sent.startswith("((AREA[Phase]PHASE3 OR AREA[Phase]PHASE2)) AND (AREA[StudyFirstPostDate]"), sent


async def test_donor_balanced_needs_one_dataset(tmp_path: Path) -> None:
    """donor_id is unique only within dataset_id: an unfixed dataset is unsupported_combination naming
    get_anndata (SC2)."""
    gw = shipped(tmp_path)
    soma_vocab(gw, {"cell_type": ["T cell", "B cell"], "dataset_id": ["d1", "d2"], "is_primary_data": ["True", "False"]})
    e = await refused(gw, "single_cell", "get_anndata_donor_balanced",
                      {"value_filter": "cell_type == 'T cell'", "output_path": "c.h5ad"})
    assert e.kind == ErrorKind.unsupported_combination
    assert e.envelope()["alternative"] == "single_cell.get_anndata"
    plan = await gw.prepare("single_cell", "get_anndata_donor_balanced",
                            {"value_filter": "cell_type == 'T cell' and dataset_id == 'd1'", "output_path": "c.h5ad"},
                            None)
    assert plan.route == "upstream" and "dataset_id == 'd1'" in plan.args_sent["value_filter"]


async def test_qualifier_overrides_are_gateway_only(tmp_path: Path) -> None:
    """include_duplicates / include_negated are advertised x-gateway and never sent upstream (SC3)."""
    from vbt.datalayer.derive import annotate_schema

    gw = shipped(tmp_path)
    soma_vocab(gw, {"cell_type": ["T cell"], "is_primary_data": ["True", "False"]})
    plan = await gw.prepare("single_cell", "count_cells",
                            {"value_filter": "cell_type == 'T cell'", "include_duplicates": True}, None)
    assert plan.args_sent == {"value_filter": "cell_type == 'T cell'"}
    assert plan.gateway_args.get("include_duplicates") is True
    cat = shipped_catalog()
    schema = annotate_schema(cat.contract("single_cell", "count_cells"),
                             {"type": "object", "properties": {"value_filter": {"type": "string"}}}, catalog=cat)
    assert schema["properties"]["include_duplicates"]["x-gateway"] is True
    schema = annotate_schema(cat.contract("disease", "find_diseases_by_phenotype"),
                             {"type": "object", "properties": {"phenotype_id": {"type": "string"}}}, catalog=cat)
    assert schema["properties"]["include_negated"]["x-gateway"] is True


async def test_depmap_groups_collide_or_overlap(tmp_path: Path) -> None:
    """DepMap disease arguments are substring-matched upstream: a value matching several stored values,
    or two groups sharing a value, is invalid_argument (SC4)."""
    gw = shipped(tmp_path)
    vocab(gw, {"diseaseFromSource": ["Lung Cancer", "Melanoma", "Small Cell Lung Cancer"],
               "tissueName": ["Lung"]})
    e = await refused(gw, "functional_genomics", "find_essential_genes", {"disease": "Lung Cancer"})
    assert e.kind == ErrorKind.invalid_argument and e.envelope().get("reason") == "substring_collision"
    e = await refused(gw, "functional_genomics", "find_selective_dependencies",
                      {"target_disease": "Melanoma", "comparison_disease": "Melanoma"})
    assert e.kind == ErrorKind.invalid_argument and e.envelope().get("reason") == "overlapping_groups"
    plan = await gw.prepare("functional_genomics", "find_essential_genes", {"disease": "Melanoma"}, None)
    assert plan.args_sent["disease"] == "Melanoma"


async def test_incomparable_order_offers_the_stored_values(tmp_path: Path) -> None:
    """With the witness off the error still lists the dimension's values to retry with (SC7)."""
    gw = shipped(tmp_path, ASSOC)
    e = await refused(gw, "association", "filter_by_datatype", {"limit": 5})
    env = e.envelope()
    assert e.kind == ErrorKind.incomplete_key and env["subkind"] == "incomparable_order"
    assert env["values"] == ["genetic_association", "literature"]
    assert env["retry_with"] and "0 groups" not in e.message


# ---------------------------------------------------------------------------- outcomes (SC5, SC6, F11)


async def test_position_matching_several_variants_is_ambiguous(tmp_path: Path) -> None:
    """chromosome+position without alleles matching two variants: ambiguous with both IDs (SC5)."""
    gw = shipped(tmp_path, {"open_targets.variant": [
        {"variantId": "1_100_A_G", "chromosome": "1", "position": 100},
        {"variantId": "1_100_A_T", "chromosome": "1", "position": 100}]})
    gw.service.keys["open_targets.variant"] = ["variantId"]  # type: ignore[attr-defined]
    with pytest.raises(GatewayError) as e:
        await call(gw, "genetics", "get_variant_annotation", {"chromosome": "1", "position": 100},
                   lambda a: {"variantId": "1_100_A_G", "chromosome": "1", "position": 100})
    assert e.value.kind == ErrorKind.ambiguous
    assert sorted(c["id"] for c in e.value.envelope()["candidates"]) == ["1_100_A_G", "1_100_A_T"]
    from vbt.failures import MODEL_SIDE_KINDS
    assert "ambiguous" in {getattr(k, "value", k) for k in MODEL_SIDE_KINDS}


async def test_clinical_data_is_sized_before_the_download(tmp_path: Path) -> None:
    """get_clinical_data pulls a whole study: its sample count x row bytes over the cap is too_large
    before the call (size_from, SC6)."""
    def upstream(tool: str, args: dict[str, Any]) -> Any:
        assert tool == "get_study_details" and args == {"study_id": "big_study"}, (tool, args)
        return {"studyId": "big_study", "sample_counts": {"sequenced": 9000, "cna": 50_000}}

    gw = shipped(tmp_path, upstream=upstream)
    e = await refused(gw, "clinicaltrials", "get_clinical_data", {"study_id": "big_study"})
    assert e.kind == ErrorKind.too_large and "50,000 rows" in e.message
    gw.bridge.upstream = lambda tool, args: {"studyId": "small", "sample_counts": {"sequenced": 20}}
    plan = await gw.prepare("clinicaltrials", "get_clinical_data", {"study_id": "small"}, None)
    assert plan.route == "upstream"


async def test_registry_not_found_is_not_found(tmp_path: Path) -> None:
    """An NCT ID the registry reports as not found is kind not_found, not an unverified empty (F11)."""
    gw = shipped(tmp_path)
    with pytest.raises(GatewayError) as e:
        await call(gw, "clinicaltrials", "get_clinical_trial_details", {"nct_id": "NCT99999999"},
                   lambda a: {"success": False, "error": "Trial NCT99999999 not found in ClinicalTrials.gov"})
    assert e.value.kind == ErrorKind.not_found


# ---------------------------------------------------------------------------- files (INV-2)


async def test_output_path_sent_is_the_confined_write_once_file(tmp_path: Path) -> None:
    """Upstream puts a relative output_path under its dated folder: the gateway sends the absolute confined
    path, so the file it keeps write-once is the file upstream writes (INV-2)."""
    gw = shipped(tmp_path, ASSOC)
    out = tmp_path / "out"
    plan = await gw.prepare("association", "filter_by_datatype", BY_DATATYPE, None)
    target = out / "a.parquet"
    assert plan.args_sent["output_path"] == str(target)
    target.write_bytes(b"first answer")
    gw.abandon(plan)
    plan = await gw.prepare("association", "filter_by_datatype", BY_DATATYPE, None)
    assert plan.args_sent["output_path"] == str(target) and not target.exists()
    kept = [p for p in out.iterdir() if p.name.startswith("a.") and p.name != "a.parquet"]
    assert len(kept) == 1 and kept[0].read_bytes() == b"first answer"


# ---------------------------------------------------------------------------- mode, readiness, memory (R2, R3, R5, INV-3)


def test_gateway_mode_is_validated() -> None:
    """YAML reads a bare `mode: off` as false: it means off; any other value is refused (R3)."""
    assert GatewaySettings(mode=False).mode == "off"  # type: ignore[arg-type]
    for bad in ("enforced", "Enforce", True):
        with pytest.raises(ValueError):
            GatewaySettings(mode=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GatewaySettings(profile="fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DataSettings.from_dict({"gateway": {"when_service_down": "loose"}})


def test_yaml_off_turns_the_data_layer_off(tmp_path: Path) -> None:
    from vbt.preflight import data_layer_active

    prof = tmp_path / "off.yaml"
    prof.write_text("data:\n  gateway:\n    mode: off\n")
    cfg = load_config([str(prof)])
    assert DataSettings.from_config(cfg).gateway.mode == "off"
    assert data_layer_active(cfg) is False


async def test_supplied_readiness_cancels_the_listing_check(tmp_path: Path) -> None:
    """Preflight's results stand: the listing-triggered full check is cancelled and never rescheduled (R2)."""
    gw = shipped(tmp_path)
    gw._check_task = asyncio.get_running_loop().create_task(asyncio.sleep(60))
    task = gw._check_task
    gw.set_readiness({"tables": {}})
    await asyncio.sleep(0)
    assert task.cancelled() and gw._check_task is None
    gw._schedule_check()
    assert gw._check_task is None


async def test_observe_mode_never_waits_reserves_or_witnesses(tmp_path: Path) -> None:
    """Observe adds no latency and changes nothing: no wait on the session's check, no memory
    reservation, no witness scan (R2, R5)."""
    gw = shipped(tmp_path, EXPRESSION, gateway={"mode": "observe"})
    gw._check_task = asyncio.get_running_loop().create_task(asyncio.sleep(60))
    for _ in range(3):
        await asyncio.wait_for(call(gw, "expression", "list_available_tissues", {}, lambda a: TISSUES), timeout=10)
    assert not gw.admission.ledger.pending("expression")
    assert not [v for v, _r in gw.service.log if v == "_witness"]  # type: ignore[attr-defined]
    gw._check_task.cancel()


async def test_abandoned_call_releases_its_reservation(tmp_path: Path) -> None:
    """An upstream attempt that raises before finish() releases what prepare reserved (R5)."""
    gw = shipped(tmp_path, EXPRESSION)
    reserved: list[str] = []
    real = gw.admission.ledger.reserve

    def spy(server: str, tables: Any) -> Any:
        reserved.append(server)
        return real(server, tables)

    gw.admission.ledger.reserve = spy  # type: ignore[method-assign]
    plan = await gw.prepare("expression", "list_available_tissues", {}, None)
    assert reserved == ["expression"] and gw.admission.ledger.pending("expression")
    gw.abandon(plan)
    assert not gw.admission.ledger.pending("expression")


async def test_in_tool_memory_error_recycles_the_server(tmp_path: Path) -> None:
    """isError with a memory signature: oom, and the server is recycled proactively (§14.4, INV-3)."""
    gw = shipped(tmp_path)
    recycled: list[str] = []

    async def recycle(server: str, wait_s: float = 30.0) -> bool:
        recycled.append(server)
        return True

    gw.admission.recycle = recycle
    with pytest.raises(GatewayError) as e:
        await call(gw, "pubmed", "search_pubmed", {"query": "pcsk9"},
                   lambda a: "MemoryError: Unable to allocate 2.00 GiB", is_error=True)
    assert e.value.kind == ErrorKind.oom
    assert recycled == ["pubmed"]


# ---------------------------------------------------------------------------- provenance (INV-5, INV-7)


async def test_provenance_records_the_serving_commit_and_timings(tmp_path: Path) -> None:
    gw = shipped(tmp_path, EXPRESSION)
    gw._upstream_commit = "f" * 40                    # the checkout that serves the calls
    plan, res = await call(gw, "expression", "list_available_tissues", {}, lambda a: TISSUES)
    prov = res.provenance
    assert prov.upstream.commit == "f" * 40 and prov.upstream.reviewed_commit
    assert prov.upstream.reviewed_commit != prov.upstream.commit
    assert {"prepare", "call", "finish", "total"} <= set(prov.t_ms), prov.t_ms
    assert prov.upstream.server == "expression" and prov.memory is not None


async def test_generic_guard_provenance_is_complete(tmp_path: Path) -> None:
    """A tool with no reviewed binding still records upstream (hash seed, stripped flags) and memory (INV-7)."""
    gw = shipped(tmp_path)
    plan, res = await call(gw, "nosuch", "tool", {"x": 1}, lambda a: {"answer": 1})
    prov = res.provenance
    assert prov.upstream is not None and prov.upstream.server == "nosuch"
    assert prov.memory is not None and {"prepare", "total"} <= set(prov.t_ms)


def test_preflight_reports_an_unset_ceiling_in_no_web_runs(tmp_path: Path) -> None:
    """§22 item 7: until no-web runs set the evidence ceiling, readiness says it is unset (SC10)."""
    from vbt.preflight import _leakage_ceiling

    cfg = load_config()
    cfg.setdefault("web", {})["enabled"] = False
    found = _leakage_ceiling(cfg)
    assert found and not found[0].ok and not found[0].required
    assert "clinicaltrials_gov" in found[0].detail
    cfg.setdefault("data", {}).setdefault("leakage", {})["ceiling"] = "2025-01-31"
    assert _leakage_ceiling(cfg) == []
    cfg["web"]["enabled"] = True
    assert _leakage_ceiling(load_config()) == []


def test_addendum_points_at_a_served_tool() -> None:
    """The burden addendum names no tool that is quarantined in phase 1 (SC11)."""
    text = (Path(__file__).resolve().parents[2] / "src" / "vbt" / "prompts" / "genomics_burden_addendum.md").read_text()
    cat = shipped_catalog()
    for server in cat.servers():
        for tool in cat.tools(server):
            b = cat.contract(server, tool).binding
            if b is not None and b.serve == "block":
                assert f"mcp__{server}__{tool}" not in text, tool
    assert "query_associations` with `include_indirect=false`" in text


# ---------------------------------------------------------------------------- preflight (R1, R4)


def test_session_check_reads_only_the_enabled_tools_tables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The session-start check covers the tables the enabled servers' tools read, not every catalog table
    (no unbound Zenodo archive tables), and reuses persisted results that still match (R1)."""
    from vbt import preflight

    seen: list[Any] = []
    monkeypatch.setattr(preflight, "run_data_check",
                        lambda config, tables=None, depth=None, **kw: seen.append(tables) or {"tables": {}})
    cfg = load_config()
    cfg["data"]["cache_dir"] = str(tmp_path / "cache")          # no persisted results to reuse
    cfg["mcp_servers"]["servers"] = [s for s in cfg["mcp_servers"]["servers"] if s.get("name") == "target"]
    preflight.check_reference_data(cfg)
    assert seen and seen[0], seen
    assert all(t.startswith("open_targets.") for t in seen[0]), seen[0]      # no Zenodo, Tahoe or cBioPortal
    assert "open_targets.target" in seen[0] and "open_targets.study" not in seen[0]


def test_r5_sampling_skips_rows_before_rendering_their_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Sampled key checks decide on the prefix block before the full key is canonicalised, so a sampled
    check renders only the sampled rows' keys (R1)."""
    from types import SimpleNamespace

    from vbt.datalayer.service import checks

    rows = [SimpleNamespace(key=[f"G{i // 10}", f"D{i}"]) for i in range(20_000)]
    full = {"n": 0}
    real = checks.canonical

    def counting(values: Any, types: Any = None) -> Any:
        if types is not None:
            full["n"] += 1
        return real(values, types)

    monkeypatch.setattr(checks, "canonical", counting)
    spec = SimpleNamespace(key=SimpleNamespace(check="sampled", sample_prefix=["g"]))
    table = SimpleNamespace(spec=spec, nullable_key=[], is_item_table=False, ref="s.t")
    reader = SimpleNamespace(table=table, key=["g", "d"], partitions={}, storage_types=lambda k: [None, None],
                             scan=lambda *a, **kw: iter(rows))
    run = SimpleNamespace(ctx=SimpleNamespace(reader=lambda ref: reader, settings=SimpleNamespace(
        readiness=SimpleNamespace(key_check_full_max_rows=10**9), cache_dir=tmp_path)), ref="s.t",
        rows=len(rows), depth="standard", stats={}, sample_rows=1000, add=lambda *a, **kw: None)
    model = checks.r5_keys(run)  # type: ignore[arg-type]
    assert model.ok and model.method == "sampled"
    assert 0 < full["n"] < len(rows) / 5, full


def test_missing_descriptors_dir_is_not_all_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mis-set descriptors_dir is a failed data check: the legacy whole-release checks decide, and nothing
    reports every data tool ready (R4)."""
    from vbt import preflight

    monkeypatch.delenv("OPEN_TARGETS_DATA_PATH", raising=False)
    cfg = load_config()
    cfg["data"]["descriptors_dir"] = str(tmp_path / "nonexistent")
    with pytest.raises(preflight.DataCheckUnavailable):
        preflight.data_catalog(cfg)
    results = preflight.check_reference_data(cfg)
    lines = [r.line() for r in results]
    assert not any(r.ok for r in results if "data tools ready" in r.label), lines
    ot = next(r for r in results if r.label.startswith("Open Targets reference data"))
    assert not ot.ok and "OPEN_TARGETS_DATA_PATH is not set" in ot.line(), lines


# ---------------------------------------------------------------------------- transforms and field maps (F1, F3, F6)


def test_parent_bound_argument_reads_the_parent_key() -> None:
    """A probe has its own `id`: target_id (bound to the parent's id) is checked on its `/id` part, and an
    item without that part is not re-checked (upstream scoped it to its parent) (F1)."""
    from vbt.datalayer.gateway.transforms import Counters, honour_arguments
    from vbt.datalayer.predicate import Eq

    rows = [{"id": "PROBE-1", "/id": "ENSG00000169174"}, {"id": "PROBE-2", "/id": "ENSG00000141510"}]
    preds = {"target_id": (Eq("id", "ENSG00000169174"), None)}
    c = Counters()
    kept = honour_arguments(rows, preds, lambda col: None, c, parents={"id": "/id"})
    assert [r["id"] for r in kept] == ["PROBE-1"] and c.excluded == {"target_id": 1}
    c = Counters()
    assert len(honour_arguments([{"id": "Approved Drug"}], preds, lambda col: None, c, parents={"id": None})) == 1
    assert not c.excluded


def test_item_filter_on_flat_item_rows() -> None:
    """Derived rows of an item table are the items: go[].aspect is read as the item's aspect (F6, F7)."""
    from vbt.datalayer.gateway.transforms import on_item_rows
    from vbt.datalayer.predicate import Eq, Or, evaluate

    flat = on_item_rows(Eq("go[].aspect", "P"), "go[]")
    assert evaluate(flat, {"id": "GO:1", "aspect": "P"}) is True
    assert evaluate(flat, {"id": "GO:2", "aspect": "F"}) is False
    both = on_item_rows(Or((Eq("tissues[].label", "liver"), Eq("tissues[].efo_code", "liver"))), "tissues[]")
    assert evaluate(both, {"label": "liver", "efo_code": "UBERON_0002107"}) is True


def test_field_maps_never_add_or_null_upstream_values() -> None:
    """A per-item spec (children[].name) checks the items and adds no key; an unchanged mapped value is
    returned as upstream gave it, a list-crossing column of an item row is item-relative (F3, hierarchy)."""
    from vbt.datalayer.descriptor.overlay import FieldMap
    from vbt.datalayer.gateway.fields import FieldMapper

    fields = {"disease_id": FieldMap(column="id"), "disease_name": FieldMap(column="name"),
              "parents[].name": FieldMap(column="name", placeholders=["Unknown"], on_placeholder="dangling_ref"),
              "children[].name": FieldMap(column="name", placeholders=["Unknown"], on_placeholder="dangling_ref")}
    m = FieldMapper(fields)
    row = {"disease_id": "X", "disease_name": "x", "parents": [{"id": "P", "name": "Unknown"}], "children": []}
    logical = m.to_logical(row)
    assert logical["name"] == "x"                       # disease_name keeps its column
    out = m.to_output(logical)
    assert set(out) == set(row) and out["disease_name"] == "x"
    assert out["parents"] == [{"id": "P", "name": None}] and row["parents"][0]["name"] == "Unknown"
    assert m.counters["dangling"] == {"parents[].name": 1}

    item = FieldMapper({"disease": FieldMap(column="geneEssentiality[].depMapEssentiality[].screens[].diseaseFromSource")},
                       item_prefix="geneEssentiality[].depMapEssentiality[].screens[]")
    row = {"disease": "Lung Cancer", "num_cell_lines": 1}
    assert item.to_output(item.to_logical(row)) == row


async def test_listing_check_covers_only_the_started_servers(tmp_path: Path) -> None:
    """The listing-triggered check reads the started servers' tables, never every catalog table (R1, R2)."""
    from types import SimpleNamespace

    gw = shipped(tmp_path)
    gw.bridge.servers = [SimpleNamespace(name="expression")]  # type: ignore[attr-defined]
    tables = gw._session_tables()
    assert tables and all(t.startswith("open_targets.") for t in tables), tables
    assert "open_targets.expression" in tables and "open_targets.known_drug" not in tables
    gw._schedule_check()
    assert gw._check_task is not None
    await gw._check_task
    checked = [r for v, r in gw.service.log if v == "_check"]  # type: ignore[attr-defined]
    assert checked and sorted(checked[0].tables) == tables
