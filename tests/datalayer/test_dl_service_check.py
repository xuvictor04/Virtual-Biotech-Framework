"""Readiness checks R1-R10 in the data child (service/checks.py, ``_check``; §13).

Each test builds the smallest table that shows one status and asserts the status of the part it
concerns (table, column, container, partition or item table) and the finding that explains it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import yaml  # noqa: E402

from vbt.datalayer.ipc import parse_response  # noqa: E402
from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service.checks import check_table, eval_expr, worst  # noqa: E402
from vbt.datalayer.service.verbs import load_verbs  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402


def make_ctx(tmp: Path, tables: dict[str, Any], *, id_types: dict[str, Any] | None = None, **extra: Any
             ) -> ServiceContext:
    desc = {"schema": "vbt.datasource/1", "source": "s", "title": "s", "root": str(tmp / "data"),
            "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
            "id_types": id_types or {}, "tables": tables, **extra}
    (tmp / "sources").mkdir(parents=True, exist_ok=True)
    (tmp / "overlays").mkdir(parents=True, exist_ok=True)
    (tmp / "sources" / "s.yaml").write_text(yaml.safe_dump(desc, sort_keys=False))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp / "sources"), "overlays_dir": str(tmp / "overlays"),
                                       "cache_dir": str(tmp / "cache")}, project_root=tmp)
    return ServiceContext(settings)


def write(tmp: Path, name: str, rows: list[dict[str, Any]], schema: Any = None, *, part: str = "part-00000",
          **kw: Any) -> Path:
    d = tmp / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist(rows, schema=schema) if schema is not None else pa.Table.from_pylist(rows)
    pq.write_table(tbl, d / f"{part}.parquet", **kw)
    return d


def table(columns: dict[str, Any], key: list[str], **extra: Any) -> dict[str, Any]:
    return {"kind": "fact", "path": extra.pop("path", "t"), "grain": "row", "key": {"columns": key, **extra.pop(
        "key_extra", {})}, "columns": columns, **extra}


def finding(model: Any, name: str, ok: bool = False) -> Any:
    hits = [c for c in model.checks if c.name == name and c.ok == ok]
    assert hits, f"no {'passing' if ok else 'failing'} {name} in {[(c.name, c.ok, c.detail) for c in model.checks]}"
    return hits[0]


ROWS = [{"id": f"r{i}", "v": float(i)} for i in range(6)]
COLS = {"id": {"role": "identifier"}, "v": {"role": "measure"}}


def test_missing_table_and_partial_file(tmp_path):
    ctx = make_ctx(tmp_path, {"t": table(COLS, ["id"]), "gone": table(COLS, ["id"], path="gone")})
    d = write(tmp_path, "t", ROWS)
    assert check_table(ctx, "s.t").status == "ready"
    assert check_table(ctx, "s.gone").status == "missing"
    (d / "part-00001.parquet.part").write_bytes(b"PAR1")
    model = check_table(ctx, "s.t")
    assert model.status == "partial" and "part-00001.parquet.part" in finding(model, "R1:partial_files").detail


def _hive(tmp: Path, dirs: list[str], vocab: list[str]) -> ServiceContext:
    for v in dirs:
        write(tmp, f"ev/sourceId={v}", [{"id": f"{v}-1", "datasourceId": v}])
    ev = {"kind": "fact", "path": "ev", "layout": "hive", "grain": "row", "key": {"columns": ["id"]},
          "partitions": {"sourceId": {"column": {"role": "category", "vocab": vocab}, "mirrored_by": ["datasourceId"]}},
          "columns": {"id": {"role": "identifier"}, "datasourceId": {"role": "category"}}}
    return make_ctx(tmp, {"ev": ev})


def test_missing_and_extra_partitions(tmp_path):
    ctx = _hive(tmp_path / "a", ["chembl", "europepmc"], ["chembl", "europepmc", "eva"])
    model = check_table(ctx, "s.ev")
    assert model.status == "partial" and model.partitions == {"sourceId=eva": "partial"}
    ctx2 = _hive(tmp_path / "b", ["chembl", "europepmc", "surprise"], ["chembl", "europepmc"])
    model2 = check_table(ctx2, "s.ev")
    assert model2.status == "schema_drift" and model2.partitions.get("sourceId=surprise") == "schema_drift"
    assert finding(model2, "R7:mirror", ok=True)


def test_mirror_column_differing_from_its_partition(tmp_path):
    write(tmp_path, "ev/sourceId=chembl", [{"id": "x", "datasourceId": "europepmc"}])
    ev = {"kind": "fact", "path": "ev", "layout": "hive", "grain": "row", "key": {"columns": ["id"]},
          "partitions": {"sourceId": {"column": {"role": "category"}, "expect": "any", "mirrored_by": ["datasourceId"]}},
          "columns": {"id": {"role": "identifier"}, "datasourceId": {"role": "category"}}}
    model = check_table(make_ctx(tmp_path, {"ev": ev}), "s.ev")
    assert model.columns.get("datasourceId") == "schema_drift" and finding(model, "R7:mirror")


def test_missing_column_versus_optional_column_and_type_mismatch(tmp_path):
    write(tmp_path, "t", [{"id": "a", "v": 1.0, "label": "x"}])
    cols = {**COLS, "ghost": {"role": "category"}, "ctIds": {"role": "reference", "optional": True}}
    model = check_table(make_ctx(tmp_path, {"t": table(cols, ["id"])}), "s.t")
    assert model.columns["ghost"] == "schema_drift" and model.columns["ctIds"] == "missing"
    assert model.status == "schema_drift"
    assert finding(model, "R4").column == "ghost"
    assert any(c.column == "ctIds" and c.level == "warning" for c in model.checks)
    optional_only = check_table(make_ctx(tmp_path / "o", {"t": table({**COLS, "ctIds": {"role": "reference",
                                                                                         "optional": True}},
                                                                     ["id"], path=str(tmp_path / "data" / "t"))}),
                                "s.t")
    assert optional_only.status == "ready", "a missing optional column is a warning only"
    bad = check_table(make_ctx(tmp_path / "m", {"t": table({"id": {"role": "identifier"}, "label": {"role": "measure"}},
                                                           ["id"], path=str(tmp_path / "data" / "t"))}), "s.t")
    assert bad.columns["label"] == "schema_drift" and "does not fit role measure" in finding(bad, "R4").detail


def test_r4b_names_the_failing_prefixes(tmp_path):
    ids = ["EFO_0000001", "MONDO_0000002", "OBA_VT0000047", "OBA_VT0000048", "GO_0005737"]
    write(tmp_path, "disease", [{"id": i} for i in ids])
    t = {"kind": "ontology", "path": "disease", "grain": "term", "key": {"columns": ["id"]},
         "columns": {"id": {"role": "identifier", "id_type": "ot_disease", "self": True}}}
    ctx = make_ctx(tmp_path, {"disease": t},
                   id_types={"ot_disease": {"plugin": "ot_disease", "options": {"prefixes": ["EFO", "MONDO"]},
                                            "universe": "disease.id"}})
    model = check_table(ctx, "s.disease")
    f = finding(model, "R4b")
    assert model.status == "encoding_drift" and "OBA (2)" in f.detail and "GO (1)" in f.detail and "60.0%" in f.detail
    ok = make_ctx(tmp_path / "ok", {"disease": {**t, "path": str(tmp_path / "data" / "disease")}},
                  id_types={"ot_disease": {"plugin": "ot_disease", "options": {"prefixes": "from_universe"},
                                           "universe": "disease.id"}})
    assert check_table(ok, "s.disease").status == "ready"


def test_nested_null_keys_are_not_flagged_by_footer_counts(tmp_path):
    item = pa.struct([("id", pa.string()), ("aspect", pa.string())])
    schema = pa.schema([("gene", pa.string()), ("go", pa.list_(item))])
    write(tmp_path, "target", [{"gene": "g1", "go": [{"id": "GO:1", "aspect": "P"}]}, {"gene": "g2", "go": []},
                               {"gene": "g3", "go": None}], schema)
    footer = pq.ParquetFile(tmp_path / "data" / "target" / "part-00000.parquet").metadata.row_group(0).column(1)
    assert footer.path_in_schema == "go.list.element.id" and footer.statistics.null_count >= 2, \
        "the trap: footer null counts include empty and null lists"
    tables = {"target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["gene"]},
                         "columns": {"gene": {"role": "identifier", "self": True},
                                     "go": {"role": "nested", "item_key": ["id"], "null_means": "not_assessed",
                                            "fields": {"id": {"role": "identifier"}, "aspect": {"role": "category"}}}}},
              "target_go": {"kind": "sets", "items_of": {"table": "target", "path": "go[]"}, "grain": "annotation",
                            "key": {"columns": []}}}
    ctx = make_ctx(tmp_path, tables)
    model = check_table(ctx, "s.target_go")
    assert model.status == "ready" and finding(model, "R5", ok=True)
    parent = check_table(ctx, "s.target")
    assert parent.item_tables == {"target_go": "ready"}
    assert parent.confirmed["go[]"] == {"null": 1, "empty": 1, "nonempty": 1, "items": 1, "null_items": 0}
    write(tmp_path, "target", [{"gene": "g4", "go": [{"id": None, "aspect": "F"}]}], schema, part="part-00001")
    bad = check_table(make_ctx(tmp_path / "again", tables | {"target": {**tables["target"],
                                                                         "path": str(tmp_path / "data" / "target")}}),
                      "s.target_go")
    assert bad.status == "key_violation" and "go[].id=1" in finding(bad, "R5").detail


def test_key_duplicates_found_by_sampled_prefix_blocks(tmp_path):
    rows = [{"block": f"b{b:02d}", "k": f"k{i}", "v": 1.0} for b in range(40) for i in range(10)]
    write(tmp_path, "t", rows[:200])
    dups = [{"block": f"b{b:02d}", "k": "k3", "v": 2.0} for b in range(40)]     # every block repeats one key
    write(tmp_path, "t", rows[200:] + dups, part="part-00001")
    cols = {"block": {"role": "identifier"}, "k": {"role": "identifier"}, "v": {"role": "measure"}}
    ctx = make_ctx(tmp_path, {"t": table(cols, ["block", "k"], key_extra={"check": "sampled"})})
    model = check_table(ctx, "s.t", sample_rows=60)
    assert model.status == "key_violation" and model.key_check.method == "sampled"
    f = finding(model, "R5b")
    assert "sampled" in f.detail and model.key_check.duplicates and model.key_check.duplicates < 40
    full = check_table(make_ctx(tmp_path / "f", {"t": table(cols, ["block", "k"], key_extra={"check": "full"},
                                                            path=str(tmp_path / "data" / "t"))}), "s.t")
    assert full.key_check.method == "full" and full.key_check.duplicates == 40


def test_unconfirmed_and_refuted_encodings(tmp_path):
    write(tmp_path, "t", [{"id": "a", "safety": -1.0, "flag": "Y", "phase": 2.0},
                          {"id": "b", "safety": 2.0, "flag": "N", "phase": None}])
    cols = {"id": {"role": "identifier"},
            "safety": {"role": "measure", "statistic": "signed_factor", "encoding": {-1: "unfavourable", 0: "none"},
                       "verified": False},
            "flag": {"role": "flag", "encoding": {"Y": True, "N": False, "U": None}, "verified": False},
            "phase": {"role": "measure", "statistic": "clinical_phase", "scale": [0, 4], "verified": False}}
    model = check_table(make_ctx(tmp_path, {"t": table(cols, ["id"])}), "s.t")
    assert model.columns["safety"] == "encoding_drift" and model.confirmed["safety"]["confirmed"] is False
    assert model.columns["flag"] == "encoding_drift" and "'U'" in [c for c in model.checks
                                                                   if c.column == "flag"][0].detail
    assert model.confirmed["phase"]["confirmed"] is True and model.confirmed["phase"]["scale"] == [0, 4]
    assert model.status == "encoding_drift"


def test_literal_and_relation_constraints(tmp_path):
    write(tmp_path, "t", [{"id": "a", "padj": 0.01, "leaf": True, "children": [], "n": 2, "rows": ["x", "y"]},
                          {"id": "b", "padj": 0.09, "leaf": True, "children": ["a"], "n": 1, "rows": ["x"]}])
    cols = {"id": {"role": "identifier"}, "padj": {"role": "measure"}, "leaf": {"role": "flag"},
            "children": {"role": "hierarchy", "relation": "child"}, "n": {"role": "count"}, "rows": {"role": "payload"}}
    constraints = [{"column": "padj", "op": "<", "value": 0.10, "origin": "prepare.py:72"},
                   {"column": "leaf", "op": "equals_expr", "expr": "len(children) == 0", "origin": "doc",
                    "verified": False, "on_refute": "drop_field"},
                   {"column": "n", "op": "len_eq", "expr": "len(rows)", "origin": "doc"}]
    model = check_table(make_ctx(tmp_path, {"t": table(cols, ["id"], constraints=constraints)}), "s.t")
    assert model.confirmed["constraint:padj < 0.1"]["confirmed"] is True
    leaf = model.confirmed["constraint:leaf equals_expr len(children) == 0"]
    assert leaf == {"confirmed": False, "on_refute": "drop_field", "checked": 2}
    assert model.confirmed["constraint:n len_eq len(rows)"]["confirmed"] is True
    assert model.status == "ready", "on_refute: drop_field is recorded for the gateway, not an outage"
    strict = [dict(constraints[1], on_refute="not_ready"), {"column": "padj", "op": "<", "value": 0.05,
                                                             "origin": "x"}]
    bad = check_table(make_ctx(tmp_path / "b", {"t": table(cols, ["id"], constraints=strict,
                                                           path=str(tmp_path / "data" / "t"))}), "s.t")
    assert bad.status == "encoding_drift" and bad.columns["leaf"] == "encoding_drift"
    assert bad.columns["padj"] == "encoding_drift"
    assert eval_expr("l2(v) == 5.0", {"v": [3.0, 4.0]}) is True


def test_sentinels(tmp_path):
    write(tmp_path, "t", [{"id": "a", "v": 1.0, "tags": ["x"]}, {"id": "b", "v": None, "tags": []}])
    cols = {"id": {"role": "identifier"}, "v": {"role": "measure"}, "tags": {"role": "payload"}}
    good = {"present": [{"key": {"id": "a"}, "expect": {"v": 1.0, "nonempty": ["tags"], "contains": {"tags[]": "x"}}},
                        {"key": {"id": "b"}, "expect": {"is_null": ["v"], "is_empty": ["tags"]}}],
            "absent": [{"key": {"id": "zzz"}}]}
    assert check_table(make_ctx(tmp_path, {"t": table(cols, ["id"], sentinels=good)}), "s.t").status == "ready"
    bad = {"present": [{"key": {"id": "missing-one"}}], "absent": [{"key": {"id": "a"}}]}
    model = check_table(make_ctx(tmp_path / "b", {"t": table(cols, ["id"], sentinels=bad,
                                                             path=str(tmp_path / "data" / "t"))}), "s.t")
    assert model.status == "partial"
    fails = [c.detail for c in model.checks if c.name == "R8" and not c.ok]
    assert any("is missing" in d for d in fails) and any("found 1 row" in d for d in fails)
    wrong = {"present": [{"key": {"id": "a"}, "expect": {"v": {"gt": 5}}}]}
    w = check_table(make_ctx(tmp_path / "c", {"t": table(cols, ["id"], sentinels=wrong,
                                                         path=str(tmp_path / "data" / "t"))}), "s.t")
    assert w.status == "partial" and "v is not gt 5" in finding(w, "R8").detail


def test_dangling_references_under_integrity_partial(tmp_path):
    write(tmp_path, "drug", [{"id": "CHEMBL1"}, {"id": "CHEMBL2"}])
    write(tmp_path, "kd", [{"drugId": "CHEMBL1", "x": 1}, {"drugId": "CHEMBL9", "x": 2}])
    drug = {"kind": "entity", "path": "drug", "grain": "drug", "key": {"columns": ["id"]},
            "columns": {"id": {"role": "identifier", "self": True}}}

    def kd(integrity: str) -> dict[str, Any]:
        return {"kind": "fact", "path": "kd", "grain": "row", "key": {"columns": ["drugId", "x"]},
                "columns": {"drugId": {"role": "identifier", "ref": "drug.id", "integrity": integrity},
                            "x": {"role": "count"}}}

    partial = check_table(make_ctx(tmp_path, {"drug": drug, "kd": kd("partial")}), "s.kd")
    assert partial.status == "ready" and partial.confirmed["ref:drugId"] == {"dangling": 1, "checked": 2}
    assert any(c.name == "R9" and c.level == "warning" and "CHEMBL9" in c.detail for c in partial.checks)
    full = check_table(make_ctx(tmp_path / "f", {"drug": {**drug, "path": str(tmp_path / "data" / "drug")},
                                                 "kd": {**kd("full"), "path": str(tmp_path / "data" / "kd")}}), "s.kd")
    assert full.status == "key_violation" and full.columns["drugId"] == "key_violation"


def test_cyclic_hierarchy(tmp_path):
    write(tmp_path, "onto", [{"id": "A", "parents": ["B"]}, {"id": "B", "parents": ["C"]}, {"id": "C", "parents": ["A"]},
                             {"id": "D", "parents": []}])
    t = {"kind": "ontology", "path": "onto", "grain": "term", "key": {"columns": ["id"]},
         "columns": {"id": {"role": "identifier", "self": True},
                     "parents": {"role": "hierarchy", "relation": "parent", "of": "id", "reflexive": False}}}
    model = check_table(make_ctx(tmp_path, {"onto": t}), "s.onto")
    f = finding(model, "R10:hierarchy")
    assert model.status == "encoding_drift" and "cycle" in f.detail and model.confirmed["parents.acyclic"] == \
        {"confirmed": False}
    write(tmp_path / "ok", "onto", [{"id": "A", "parents": ["B"]}, {"id": "B", "parents": []}])
    ok = check_table(make_ctx(tmp_path / "ok", {"onto": t}), "s.onto")
    assert ok.status == "ready" and ok.confirmed["parents.acyclic"] == {"confirmed": True}


def test_check_verb_reports_hash_randomization_and_depths(tmp_path):
    write(tmp_path, "t", ROWS)
    ctx = make_ctx(tmp_path, {"t": table(COLS, ["id"], key_extra={"check": "full"})})
    verb = load_verbs()["_check"]
    shallow = parse_response("_check", verb(ctx, {"tables": ["s.t"], "depth": "shallow"}))
    assert shallow.tables["s.t"].status == "ready" and shallow.depth == "shallow"
    assert {c.name for c in shallow.tables["s.t"].checks} <= {"R1:location", "R1:fragments"}
    deep = parse_response("_check", verb(ctx, {"depth": "deep"}))
    assert deep.hash_randomization in (0, 1) and deep.tables["s.t"].key_check.method == "full"
    assert deep.tables["s.t"].fingerprint and deep.tables["s.t"].signature
    assert worst(["ready", "partial", "schema_drift", "key_violation"]) == "schema_drift"


def test_inline_manifest_hashes_and_manifest_row_checks(tmp_path):
    import hashlib
    import json

    d = write(tmp_path, "t", ROWS)
    data = (d / "part-00000.parquet").read_bytes()
    good = {"t/part-00000.parquet": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
    ok = check_table(make_ctx(tmp_path, {"t": table(COLS, ["id"])}, manifests=[{"inline": good}]), "s.t")
    assert ok.status == "ready" and finding(ok, "R2:inline", ok=True)
    bad = {"t/part-00000.parquet": {"bytes": len(data), "md5": "0" * 32}}
    model = check_table(make_ctx(tmp_path, {"t": table(COLS, ["id"])}, manifests=[{"inline": bad}]), "s.t")
    assert model.status == "partial" and "md5" in finding(model, "R2:inline").detail
    (tmp_path / "data" / "prep.json").write_text(json.dumps({"complete": True, "rows": {"t": 7},
                                                             "filters": {"t": "v < 10"}, "files": {}}))
    rows = check_table(make_ctx(tmp_path, {"t": table(COLS, ["id"])},
                                manifests=[{"path": "prep.json", "required": True, "require": {"complete": True},
                                            "checks": {"rows": "$.rows.t", "constraints": "$.filters.t"}}]), "s.t")
    assert rows.status == "partial" and "says 7 rows, the data has 6" in finding(rows, "R2:checks.rows").detail
    assert finding(rows, "R2:checks.constraints", ok=True).detail.endswith("v < 10")


def test_inverse_hierarchy_links(tmp_path):
    write(tmp_path, "onto", [{"id": "A", "parents": [], "children": ["B"]}, {"id": "B", "parents": ["A"], "children": []},
                             {"id": "C", "parents": ["A"], "children": []}])
    t = {"kind": "ontology", "path": "onto", "grain": "term", "key": {"columns": ["id"]},
         "columns": {"id": {"role": "identifier", "self": True},
                     "parents": {"role": "hierarchy", "relation": "parent", "of": "id", "inverse_of": "children"},
                     "children": {"role": "hierarchy", "relation": "child", "of": "id"}}}
    model = check_table(make_ctx(tmp_path, {"onto": t}), "s.onto")
    assert model.confirmed["parents.inverse_of"] == {"confirmed": False}
    assert "1 link(s) without the inverse" in finding(model, "R10:hierarchy").detail


def test_key_check_spill_is_removed_on_every_exit(tmp_path, monkeypatch):
    """The uniqueness pass spills to <cache>/keycheck.<pid>.*; an unfinished scan, a completed one or a
    stale directory of a killed child never leaves those files behind (R2)."""
    import os

    import vbt.datalayer.service.checks as checks
    from vbt.datalayer.service.reader import BudgetExceeded, TableReader

    rows = [{"k": f"k{i}", "v": 1.0} for i in range(40)]
    write(tmp_path, "t", rows + [{"k": "k3", "v": 2.0}])
    cols = {"k": {"role": "identifier"}, "v": {"role": "measure"}}
    ctx = make_ctx(tmp_path, {"t": table(cols, ["k"], key_extra={"check": "full"})})
    cache = Path(ctx.settings.cache_dir)
    monkeypatch.setattr(checks, "SPILL_KEYS", 5)
    model = check_table(ctx, "s.t")
    assert model.key_check.duplicates == 1                      # found through the spill
    assert not list(cache.glob("keycheck.*"))

    real_scan = TableReader.scan

    def failing_scan(self, *a, **k):
        for i, m in enumerate(real_scan(self, *a, **k)):
            if i == 20:
                raise BudgetExceeded("scan budget exceeded")
            yield m

    monkeypatch.setattr(TableReader, "scan", failing_scan)
    model = check_table(ctx, "s.t")
    assert model.key_check.ok is None and "not completed" in model.key_check.detail
    assert not list(cache.glob("keycheck.*"))
    # a killed child's directory (dead pid) and a pre-pid one are swept; a live child's is kept
    for name in ("keycheck.999999999.abc", "keycheck.zz3xggm8", f"keycheck.{os.getpid()}.live"):
        (cache / name).mkdir()
        (cache / name / "00").write_text("k\n")
    removed = checks.sweep_spills(cache)
    assert sorted(Path(p).name for p in removed) == ["keycheck.999999999.abc", "keycheck.zz3xggm8"]
    assert [p.name for p in cache.glob("keycheck.*")] == [f"keycheck.{os.getpid()}.live"]
