"""The catalog (§11.3 step 1, §11.5, §6.2): contract lookup, same_as, the generic fallback, item tables,
qualified id_types and ambiguity, selector table selection, implements aliases and digests."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import yaml

from vbt.datalayer.catalog import (
    POSITION_MARK,
    AmbiguousIdType,
    Catalog,
    TableRef,
    UnknownIdType,
    UnknownTable,
    build_catalog,
    load_catalog,
)
from vbt.datalayer.descriptor.lint import errors
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.settings import DataSettings

OT: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "ot", "title": "OT", "release": {"expect": "25.09", "from": "literal"},
    "defaults": {"format": "parquet", "layout": "sharded_dir"},
    "id_types": {"gene": {"plugin": "ensembl_gene", "universe": "target.id"},
                 "symbol": {"plugin": "hgnc_symbol", "label_of": "gene"},
                 "disease": {"plugin": "ot_disease", "universe": "disease.id"}},
    "tables": {
        "target": {"kind": "entity", "grain": "one gene", "key": {"columns": ["id"]}, "columns": {
            "id": {"role": "identifier", "id_type": "gene", "self": True},
            "go": {"role": "member", "item_key": ["termId", "aspect"],
                   "membership": {"set": {"path": "termId"}, "member": {"parent": "id"}},
                   "fields": {"termId": {"role": "category"}, "aspect": {"role": "category"}}},
            "ess": {"role": "nested", "item_key": {"identity": "position", "max_items": 1}, "fields": {
                "dep": {"role": "nested", "item_key": ["tissueId"], "fields": {
                    "tissueId": {"role": "category"},
                    "screens": {"role": "nested", "item_key": ["depmapId"], "fields": {
                        "depmapId": {"role": "identifier"}, "geneEffect": {"role": "measure"}}}}}}},
            "syn": {"role": "member", "item_key": {"identity": "value"},
                    "membership": {"set": {"parent": "id"}, "member": {"path": "[]"}}},
            "slots": {"role": "nested", "item_key": {"identity": "position"}, "fields": {"v": {"role": "payload"}}}}},
        "target_go": {"kind": "sets", "items_of": {"table": "target", "path": "go[]"}, "grain": "one annotation",
                      "key": {"columns": []}},
        "target_screens": {"kind": "fact", "items_of": {"table": "target", "path": "ess[].dep[].screens[]"},
                           "grain": "one screen", "key": {"columns": []}},
        "target_syn": {"kind": "sets", "items_of": {"table": "target", "path": "syn[]"}, "grain": "one synonym",
                       "key": {"columns": []}},
        "target_slots": {"kind": "fact", "items_of": {"table": "target", "path": "slots[]"}, "grain": "one slot",
                         "key": {"columns": []}},
        "disease": {"kind": "ontology", "grain": "one term", "key": {"columns": ["id"]},
                    "columns": {"id": {"role": "identifier", "id_type": "disease", "self": True}}},
        "coloc": {"kind": "fact", "grain": "one coloc", "layout": "single_file", "key": {"columns": ["a", "b"]},
                  "columns": {"a": {"role": "identifier", "id_type": "gene"}, "b": {"role": "identifier"}}},
        "ecaviar": {"kind": "fact", "grain": "one ecaviar", "key": {"columns": ["a", "b"]},
                    "columns": {"a": {"role": "identifier", "id_type": "gene"}, "b": {"role": "identifier"}}},
    },
}

ZEN: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "zen", "title": "Zenodo extract", "release": {"from": "literal"},
    "id_types": {"gene": {"plugin": "ensembl_gene", "universe": "target_ext.id"}},
    "tables": {
        "disease_ext": {"kind": "ontology", "grain": "one term", "implements": "ot:disease@25.09",
                        "lineage": {"source": "ot", "release": "25.09", "table": "disease"},
                        "key": {"columns": ["id"]}, "columns": {"id": {"role": "identifier", "self": True}}},
        "target_ext": {"kind": "entity", "grain": "one gene", "key": {"columns": ["id"]},
                       "columns": {"id": {"role": "identifier", "self": True}}},
    },
}

DRUG_OV: dict[str, Any] = {
    "schema": "vbt.overlay/1", "server": "drug", "sources": ["ot"], "tools": {
        "get_go": {"same_as": ["target.get_go", "pathway.get_go"],
                   "reads": {"ot.target": {"access": "full_table"}},
                   "args": {"target_id": {"binds": "ot.target.id", "accepts": ["gene", "symbol"]},
                            "limit": {"role": "limit", "min": 1, "max": 50}},
                   "result": {"rows": "$.go", "rows_of": "ot.target_go"}},
        "coloc": {"reads": {"ot.coloc": {"access": "bounded_scan"}, "ot.ecaviar": {"access": "bounded_scan"}},
                  "args": {"method": {"role": "selector", "match": "casefold",
                                      "values": {"coloc": "ot.coloc", "ecaviar": "ot.ecaviar"}},
                           "gene": {"binds": "ot.coloc.a"}},
                  "result": {"rows": "$.rows"}},
        "draft": {"status": "unreviewed", "args": {"q": {"role": "free_text"}}},
        "screens": {"reads": {"ot.target": {"access": "projection"}}, "serve": "derived",
                    "args": {"gene": {"binds": "ot.target.id", "accepts": ["gene"]}},
                    "derived": {"verb": "find", "table": "ot.target_screens",
                                "sections": {"info": {"table": "ot.disease", "verb": "lookup"}}},
                    "result": {"rows": "$.rows", "rows_of": "ot.target_screens"}},
    }}

GENERIC: dict[str, Any] = {"schema": "vbt.overlay/1", "server": "*",
                           "generic": {"not_found_when": ["$.error =~ '(?i)not found'"], "empty_when": ["$.count == 0"],
                                       "param_kinds": {"*_id": ["gene"]}}}


def catalog(aliases: dict[str, str] | None = None, ot: dict | None = None) -> Catalog:
    sources = {"ot": SourceDescriptor.model_validate(ot or OT), "zen": SourceDescriptor.model_validate(ZEN)}
    return Catalog(sources, {"drug": Overlay.model_validate(DRUG_OV)}, [Overlay.model_validate(GENERIC)],
                   aliases=aliases)


def test_fixtures_lint_clean():
    assert not errors(catalog().lint())


def test_contract_lookup_reviewed_and_tables():
    c = catalog().contract("drug", "get_go")
    assert not c.generic and c.alias_of is None and c.binding.result.rows_of == "ot.target_go"
    assert set(c.tables) == {"ot.target", "ot.target_go"} and set(c.descriptors) == {"ot"}
    assert c.full_table_reads == ("ot.target",)
    assert set(c.identifier_args) == {"target_id"} and c.limit_arg == "limit"
    assert c.bound_table == "ot.target" and c.selected_table({}) == "ot.target"
    assert c.item_table_of().ref == TableRef("ot", "target_go")
    assert c.arg_columns("target_id") == [("ot.target", "id")]
    assert c.name == "drug.get_go"


def test_contract_same_as_alias():
    cat = catalog()
    c = cat.contract("target", "get_go")
    assert not c.generic and c.alias_of == "drug.get_go" and c.overlay.server == "drug"
    assert c.server == "target" and c.binding is cat.overlays["drug"].tools["get_go"]
    assert cat.contract("pathway", "get_go").alias_of == "drug.get_go"
    assert cat.contract("target", "other").generic


def test_contract_generic_fallback_and_unreviewed():
    g = catalog().contract("uniprot", "search")
    assert g.generic and g.binding is None and g.tables == {}
    assert g.generic_spec.not_found_when == ["$.error =~ '(?i)not found'"] and g.generic_spec.param_kinds
    u = catalog().contract("drug", "draft")
    assert u.generic and u.binding is not None and u.binding.status == "unreviewed"
    no_generic = Catalog({}, {}, [])
    assert no_generic.contract("x", "y").generic_spec is None


def test_selector_table_selection():
    c = catalog().contract("drug", "coloc")
    assert c.selector_args == ["method"]
    assert c.selected_table({"method": "eCAVIAR"}) == "ot.ecaviar"     # casefold match
    assert c.selected_table({"method": "coloc"}) == "ot.coloc"
    assert c.selected_table({}) == "ot.coloc"                           # no selector value: the bound table
    with pytest.raises(GatewayError) as exc:
        c.selected_table({"method": "moloc"})
    env = exc.value.envelope()
    assert exc.value.kind == ErrorKind.invalid_argument and env["argument"] == "method"
    assert env["valid_values"] == ["coloc", "ecaviar"]


def test_derived_contract_tables_include_sections():
    c = catalog().contract("drug", "screens")
    assert {"ot.target", "ot.target_screens", "ot.disease"} == set(c.tables)
    assert c.full_table_reads == () and c.bound_table == "ot.target"


def test_item_tables_resolve_to_parent_fragments_with_composed_keys():
    cat = catalog()
    go = cat.table("ot.target_go")
    assert go.is_item_table and go.physical == TableRef("ot", "target") and go.items_path == "go[]"
    assert go.key == ("id", "go[].termId", "go[].aspect")
    assert set(go.columns) == {"termId", "aspect"} and go.container.role == "member"
    assert go.layout == "sharded_dir" and go.format == "parquet"
    screens = cat.table("ot.target_screens")
    assert screens.key == ("id", "ess[].dep[].tissueId", "ess[].dep[].screens[].depmapId")   # singleton ess[] adds nothing
    assert set(screens.columns) == {"depmapId", "geneEffect"}
    assert cat.table("ot.target_syn").key == ("id", "syn[]")                                   # identity: value
    assert cat.table("ot.target_slots").key == ("id", "slots[]" + POSITION_MARK)               # identity: position
    plain = cat.table(TableRef("ot", "coloc"))
    assert not plain.is_item_table and plain.key == ("a", "b") and plain.layout == "single_file"
    assert cat.table("ot.coloc") is plain                                                      # cached
    for bad in ("ot.nope", "nope.target", "ot", "ot.a.b"):
        with pytest.raises(UnknownTable):
            cat.table(bad)


def test_id_type_qualification_and_ambiguity():
    cat = catalog()
    assert cat.id_type("ot:gene")[0] == "ot" and cat.id_type("zen:gene")[0] == "zen"
    assert cat.id_type("gene", "zen")[0] == "zen" and cat.id_type("gene", "ot")[0] == "ot"
    assert cat.id_type("symbol")[0] == "ot"                              # unique across sources
    with pytest.raises(AmbiguousIdType):
        cat.id_type("gene")                                              # two sources, different universes
    with pytest.raises(UnknownIdType):
        cat.id_type("ot:nope")
    with pytest.raises(UnknownIdType):
        cat.id_type("nope")
    assert cat.qualify_id_type("symbol") == "ot:symbol" and cat.qualify_id_type("gene", "zen") == "zen:gene"
    # the same identity under one bare name in two sources is not ambiguous
    zen_same = copy.deepcopy(ZEN)
    zen_same["id_types"]["gene"] = {"plugin": "ensembl_gene", "universe": "ot.target.id"}
    same = Catalog({"ot": SourceDescriptor.model_validate(OT), "zen": SourceDescriptor.model_validate(zen_same)})
    assert same.id_type("gene")[0] == "ot"


def test_implements_alias():
    plain = catalog()
    assert plain.table("ot.disease").served_from is None
    aliased = catalog(aliases={"ot": "zen"})
    t = aliased.table("ot.disease")
    assert t.ref == TableRef("ot", "disease") and t.served_from == TableRef("zen", "disease_ext")
    assert t.physical == TableRef("zen", "disease_ext") and t.descriptor.source == "zen"
    assert t.spec.lineage.release == "25.09"
    assert aliased.table("ot.target").served_from is None                # not implemented: served by ot itself
    with pytest.raises(UnknownTable):
        Catalog({"ot": SourceDescriptor.model_validate(OT)}, aliases={"ot": "missing"}).table("ot.disease")


def test_digest_stability():
    a, b = catalog(), catalog()
    assert a.digest() == b.digest() and a.digest().startswith("sha256:")
    assert a.overlay_digest("drug") == b.overlay_digest("drug") and a.overlay_digest("nope") is None
    assert a.descriptor_digest("ot") == b.descriptor_digest("ot")
    ot2 = copy.deepcopy(OT)
    ot2["title"] = "changed"
    assert catalog(ot=ot2).digest() != a.digest() and catalog(ot=ot2).descriptor_digest("ot") != a.descriptor_digest("ot")
    assert catalog(aliases={"ot": "zen"}).digest() != a.digest()
    assert a.servers() == ["drug"] and "get_go" in a.tools("drug") and a.tools("x") == []
    assert TableRef("ot", "target") in a.table_refs()


def test_plugin_names():
    names = catalog().plugin_names()
    assert names["identifier"] == {"ensembl_gene", "hgnc_symbol", "ot_disease"}
    assert names["layout"] == {"sharded_dir", "single_file"} and names["format"] == {"parquet"}


def test_build_and_load_catalog(tmp_path):
    src, ov = tmp_path / "sources", tmp_path / "overlays"
    src.mkdir()
    ov.mkdir()
    (src / "ot.yaml").write_text(yaml.safe_dump(OT))
    (src / "zen.yaml").write_text(yaml.safe_dump(ZEN))
    (ov / "drug.yaml").write_text(yaml.safe_dump(DRUG_OV))
    (ov / "_generic.yaml").write_text(yaml.safe_dump(GENERIC))
    data = {"descriptors_dir": str(src), "overlays_dir": str(ov), "sources": {"alias": {"ot": "zen"}}}
    cat = build_catalog(DataSettings.from_config({"data": data}))
    assert set(cat.sources) == {"ot", "zen"} and cat.servers() == ["drug"] and len(cat.generic) == 1
    assert cat.table("ot.disease").served_from == TableRef("zen", "disease_ext")
    loaded = load_catalog({"data": data, "vars": {}})
    assert loaded.digest() == cat.digest()
    empty = load_catalog({"data": {"descriptors_dir": str(tmp_path / "none"), "overlays_dir": str(tmp_path / "none")}})
    assert empty.sources == {} and empty.contract("a", "b").generic
