"""Item tables in the data child (service/items.py; §6.2): composed keys, correlated item predicates,
null vs empty vs null-item lists and item-key violations, over one, three and nested levels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import yaml  # noqa: E402

from vbt.datalayer.predicate import And, Any as AnyP, Cmp, Eq  # noqa: E402
from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service import items as I  # noqa: E402
from vbt.datalayer.service.checks import check_table  # noqa: E402
from vbt.datalayer.service.reader import ScanStats  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402

GENE_A, GENE_B, GENE_C, GENE_D = "ENSG00000000001", "ENSG00000000002", "ENSG00000000003", "ENSG00000000004"


def make_ctx(tmp: Path, desc: dict[str, Any]) -> ServiceContext:
    (tmp / "sources").mkdir(exist_ok=True)
    (tmp / "overlays").mkdir(exist_ok=True)
    (tmp / "sources" / f"{desc['source']}.yaml").write_text(yaml.safe_dump(desc, sort_keys=False))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp / "sources"), "overlays_dir": str(tmp / "overlays"),
                                       "cache_dir": str(tmp / "cache")}, project_root=tmp)
    return ServiceContext(settings)


GO = pa.struct([("id", pa.string()), ("aspect", pa.string()), ("evidence", pa.string()), ("source", pa.string())])
TISSUE = pa.struct([("label", pa.string()), ("rna", pa.struct([("value", pa.float64()), ("unit", pa.string())]))])
SCREEN = pa.struct([("depmapId", pa.string()), ("cellLineName", pa.string()), ("geneEffect", pa.float32())])
DEPMAP = pa.struct([("tissueId", pa.string()), ("screens", pa.list_(SCREEN))])
ESSENTIALITY = pa.struct([("isEssential", pa.bool_()), ("depMapEssentiality", pa.list_(DEPMAP))])


def go(i: str, aspect: str = "P") -> dict[str, Any]:
    return {"id": i, "aspect": aspect, "evidence": "IDA", "source": "UniProt"}


def screens(*pairs: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"depmapId": d, "cellLineName": f"line {d}", "geneEffect": e} for d, e in pairs]


@pytest.fixture
def ot(tmp_path: Path) -> ServiceContext:
    root = tmp_path / "ot"
    rows = [
        {"id": GENE_A, "go": [go("GO:1"), go("GO:2", "F"), None],
         "tissues": [{"label": "liver", "rna": {"value": 5.0, "unit": "TPM"}},
                     {"label": "heart", "rna": {"value": 20.0, "unit": "TPM"}}]},
        {"id": GENE_B, "go": [], "tissues": [{"label": "liver", "rna": {"value": 15.0, "unit": "TPM"}}]},
        {"id": GENE_C, "go": None, "tissues": None},
        {"id": GENE_D, "go": [go("GO:3"), go("GO:3")], "tissues": []},
    ]
    schema = pa.schema([("id", pa.string()), ("go", pa.list_(GO)), ("tissues", pa.list_(TISSUE))])
    (root / "target").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / "target" / "part-00000.parquet")
    ess = [
        {"id": GENE_A, "geneEssentiality": [{"isEssential": True, "depMapEssentiality": [
            {"tissueId": "UBERON_1", "screens": screens(("ACH-1", -1.5), ("ACH-2", -0.2))},
            {"tissueId": "UBERON_2", "screens": screens(("ACH-3", -2.5))}]}]},
        {"id": GENE_B, "geneEssentiality": [{"isEssential": False, "depMapEssentiality": [
            {"tissueId": "UBERON_1", "screens": screens(("ACH-1", 0.1), ("ACH-9", None))}]}]},
    ]
    (root / "target_essentiality").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(ess, schema=pa.schema([("id", pa.string()),
                                                               ("geneEssentiality", pa.list_(ESSENTIALITY))])),
                   root / "target_essentiality" / "part-00000.parquet")
    desc = {
        "schema": "vbt.datasource/1", "source": "ot", "title": "ot", "root": str(root),
        "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
        "tables": {
            "target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["id"]},
                       "columns": {
                           "id": {"role": "identifier", "self": True},
                           "go": {"role": "member", "item_key": ["id", "aspect", "evidence", "source"],
                                  "null_means": "not_assessed", "empty_means": "unknown",
                                  "membership": {"set": {"path": "id"}, "member": {"parent": "id"}},
                                  "fields": {"id": {"role": "identifier"}, "aspect": {"role": "category"},
                                             "evidence": {"role": "qualifier", "effect": "evidence_code"},
                                             "source": {"role": "category"}}},
                           "tissues": {"role": "nested", "item_key": ["label"],
                                       "fields": {"label": {"role": "label"},
                                                  "rna": {"role": "nested",
                                                          "fields": {"value": {"role": "measure"},
                                                                     "unit": {"role": "category"}}}}}}},
            "target_go": {"kind": "sets", "items_of": {"table": "target", "path": "go[]"}, "grain": "annotation",
                          "key": {"columns": [], "check": "full"},
                          "sentinels": {"present": [{"key": {"id": GENE_A}, "expect": {"min_rows": 2}}]}},
            "expression_tissues": {"kind": "fact", "items_of": {"table": "target", "path": "tissues[]"},
                                   "grain": "tissue of a gene", "key": {"columns": []}},
            "target_essentiality": {"kind": "fact", "path": "target_essentiality", "grain": "gene",
                                    "key": {"columns": ["id"]},
                                    "columns": {"id": {"role": "identifier", "self": True},
                                                "geneEssentiality": {
                                                    "role": "nested", "item_key": {"identity": "position", "max_items": 1},
                                                    "fields": {"isEssential": {"role": "flag"},
                                                               "depMapEssentiality": {
                                                                   "role": "nested", "item_key": ["tissueId"],
                                                                   "fields": {"tissueId": {"role": "identifier"},
                                                                              "screens": {
                                                                                  "role": "nested", "item_key": ["depmapId"],
                                                                                  "fields": {"depmapId": {"role": "identifier"},
                                                                                             "cellLineName": {"role": "label"},
                                                                                             "geneEffect": {"role": "measure"}}}}}}}}},
            "target_essentiality_screens": {
                "kind": "fact", "grain": "screen",
                "items_of": {"table": "target_essentiality",
                             "path": "geneEssentiality[].depMapEssentiality[].screens[]"},
                "key": {"columns": []}, "rank": [{"column": "geneEffect", "direction": "asc"}]},
        },
    }
    return make_ctx(tmp_path, desc)


def test_composed_keys_and_item_rows(ot):
    r = ot.reader("ot.target_go")
    assert r.key == ("id", "go[].id", "go[].aspect", "go[].evidence", "go[].source")
    rows, keys, stats = r.rows(None)
    assert stats.total == 4
    assert keys[0] == [GENE_A, "GO:1", "P", "IDA", "UniProt"]
    assert rows[0]["id"] == "GO:1" and rows[0]["/id"] == GENE_A, "the parent key is injected; the item keeps its id"
    # a bare name is the item's field; /id (or ^.id) is the parent's
    assert r.count(Eq("id", "GO:3"))[0] == 2
    assert r.count(Eq("/id", GENE_A))[0] == 2
    assert r.count(Eq("^.id", GENE_D))[0] == 2


def test_three_level_item_table(ot):
    r = ot.reader("ot.target_essentiality_screens")
    assert r.key == ("id", "geneEssentiality[].depMapEssentiality[].tissueId",
                     "geneEssentiality[].depMapEssentiality[].screens[].depmapId")
    assert r.count()[0] == 5
    top = r.top_k(None, [{"column": "geneEffect", "direction": "asc"}], 3)
    assert top == [[GENE_A, "UBERON_2", "ACH-3"], [GENE_A, "UBERON_1", "ACH-1"], [GENE_A, "UBERON_1", "ACH-2"]]
    total, unknown, _na, unknown_total = r.count(Cmp("geneEffect", "<", 0))
    assert (total, unknown_total, unknown) == (3, 1, {"geneEffect": 1, "_rows": 1})
    rows, _, _ = r.rows(Eq("depmapId", "ACH-1"), order=[{"column": "geneEffect", "direction": "asc"}])
    assert [(x["/id"] if "/id" in x else x["id"], x["tissueId"], x["geneEffect"]) for x in rows] == \
        [(GENE_A, "UBERON_1", -1.5), (GENE_B, "UBERON_1", pytest.approx(0.1))]


def test_correlated_any_applies_both_conditions_to_one_item(ot):
    parent = ot.reader("ot.target")
    pred = AnyP("tissues[]", And((Eq("label", "liver"), Cmp("rna.value", ">=", 10))))
    assert parent.key_set(pred) == [[GENE_B]], "liver (5) and heart (20) of GENE_A are different items"
    items = ot.reader("ot.expression_tissues")
    rows, keys, _ = items.rows(And((Eq("label", "liver"), Cmp("rna.value", ">=", 10))))
    assert keys == [[GENE_B, "liver"]] and rows[0]["rna"]["value"] == 15.0


def test_null_empty_and_null_items_are_counted_apart(ot):
    r = ot.reader("ot.target_go")
    st = ScanStats()
    assert sum(1 for _ in r.scan(None, stats=st)) == 4
    c = st.counts
    assert (c.rows, c.null, c.empty, c.nonempty, c.items, c.null_items) == (4, 1, 1, 2, 4, 1)
    direct = I.container_counts([{"go": None}, {"go": []}, {"go": [None, {"id": 1}]}], "go[]")
    assert (direct.null, direct.empty, direct.null_items, direct.items) == (1, 1, 1, 1)


def test_item_key_violations(ot):
    dup = I.item_key_duplicates({"go": [go("GO:3"), go("GO:3"), go("GO:4")]}, "go[]", ["id", "aspect", "evidence",
                                                                                          "source"])
    assert len(dup) == 1 and "GO:3" in dup[0]
    model = check_table(ot, "ot.target_go")
    assert model.key_check is not None and model.key_check.duplicates == 1
    assert model.status == "key_violation"
    assert any(c.name == "R5b" and not c.ok and "duplicate" in c.detail for c in model.checks)
    assert any(c.name == "R8" and c.ok for c in model.checks), "the item-table sentinel counts items of its parent"
    parent = check_table(ot, "ot.target")
    assert any(c.name == "R5b:items" and not c.ok for c in parent.checks)
    assert parent.containers.get("go") == "key_violation"


def test_levels_and_paths():
    lv = I.levels("hallmarks.attributes[]")
    assert [(x.names, x.text) for x in lv] == [(("hallmarks", "attributes"), "hallmarks.attributes[]")]
    lv3 = I.levels("a[].b[].c[]")
    assert [x.text for x in lv3] == ["a[]", "a[].b[]", "a[].b[].c[]"]
    row = {"k": 1, "a": [{"b": [{"c": [10, 20]}, {"c": None}]}, {"b": []}]}
    views = list(I.explode(row, lv3))
    assert [p for _, p in views] == [(0, 0, 0), (0, 0, 1)]
    assert I.path_value(views[1][0], "a[].b[].c[]") == 20
    assert I.view_at(row, lv3, (0, 0, 1)) == views[1][0]
    with pytest.raises(ValueError):
        I.levels("a[].b")
