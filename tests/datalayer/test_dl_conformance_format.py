"""Format conformance suite (§9.3 F-1..F-11) over the registered format plugins, plus explicit
parquet cases: float32 keys, leaf reads inside lists, the correlated three-level Any, NaN and
signed-zero pushdown rules, residual rules, footer stats and legacy leaf paths."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from vbt.datalayer.plugins.base import FormatError, Fragment  # noqa: E402
from vbt.datalayer.plugins.conformance import golden as g  # noqa: E402
from vbt.datalayer.plugins.conformance.format import *  # noqa: E402,F401,F403  (collects F-1..F-11)
from vbt.datalayer.plugins.conformance.format import oracle_rows, scan_rows, scan_schema, write_golden  # noqa: E402
from vbt.datalayer.plugins.formats import round_to_storage, storage_typed  # noqa: E402
from vbt.datalayer.plugins.formats.parquet import PATHS_KEY, ParquetFormat  # noqa: E402
from vbt.datalayer.predicate import (  # noqa: E402
    All,
    And,
    Any,
    Cmp,
    CmpAbs,
    Contains,
    Eq,
    In,
    IsNull,
    Not,
    TextMatch,
    evaluate,
)
from vbt.datalayer.rowkey import render_float  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "src"
PARQUET = ParquetFormat()


def _write(tmp_path, name: str, table=None) -> Fragment:
    case = g.golden(name)
    return write_golden(PARQUET, case, tmp_path, table=table)


def _ids(rows):
    return [r["id"] for r in rows]


# ---------------------------------------------------------------------------
# F-10: float32 equality is storage-typed
# ---------------------------------------------------------------------------

def test_float32_eq_pushdown_equals_the_oracle(tmp_path):
    frag = _write(tmp_path, "scalars")
    schema = PARQUET.logical_schema(frag)
    pushdown, residual = PARQUET.compile(Eq("concentration", 0.05), schema)
    assert residual is None and "concentration" in str(pushdown)
    rows = scan_rows(PARQUET, [frag], predicate=Eq("concentration", 0.05))
    full = scan_rows(PARQUET, [frag])
    assert _ids(rows) == _ids(oracle_rows(PARQUET, schema, full, Eq("concentration", 0.05)))
    assert _ids(rows) == ["ENSG00000169174", 'say "hi"', "a(b)", "zz"]
    # the untyped float64 literal matches nothing: this is the C20 bug the cast fixes
    assert [r for r in full if evaluate(Eq("concentration", 0.05), r) is True] == []
    assert {render_float(r["concentration"], "float") for r in rows} == {"0.05"}


def test_storage_typed_rewrites_literals_inside_quantifiers():
    def type_of(path):
        return {"drugs[].x": "float", "f32": "float32", "f64": "double"}.get(path)

    p = And((Eq("f32", 0.05), Any("drugs[]", Cmp("x", "<", 0.1)), Eq("f64", 0.05), In("f32", (0.5, 0.05))))
    typed = storage_typed(p, type_of)
    assert typed.preds[0].value == round_to_storage(0.05, "float") != 0.05
    assert typed.preds[1].pred.value == round_to_storage(0.1, "float")
    assert typed.preds[2].value == 0.05
    assert typed.preds[3].values == (0.5, round_to_storage(0.05, "float"))
    assert round_to_storage(1e300, "float") == 1e300 and round_to_storage(True, "float") is True


# ---------------------------------------------------------------------------
# F-2 / F-11: leaf projection inside lists
# ---------------------------------------------------------------------------

def test_read_leaves_reads_drug_ids_inside_lists(tmp_path):
    frag = _write(tmp_path, "drugs")
    schema = PARQUET.logical_schema(frag)
    leaf = PARQUET.leaf_path("drugs[].drugId", schema)
    assert leaf == "drugs.list.element.drugId"
    table = PARQUET.read_leaves(frag, [leaf], None)
    item = table.schema.field("drugs").type.value_type
    assert [item.field(i).name for i in range(item.num_fields)] == ["drugId"]
    assert PARQUET.to_native(table) == [
        {"drugs": [{"drugId": "CHEMBL1"}, {"drugId": "CHEMBL2"}]}, {"drugs": []}, {"drugs": None}, {"drugs": [None]},
        {"drugs": [{"drugId": None}]}, {"drugs": [{"drugId": "CHEMBL1"}, {"drugId": "CHEMBL3"}]}]
    assert PARQUET.to_native(PARQUET.read_leaves(frag, ["drugs[].drugId"], [0])) == PARQUET.to_native(table)
    stats = PARQUET.stats(frag).columns[leaf]
    assert stats.kind == "nested" and stats.max_rep_level == 1 and stats.num_values >= 6


def test_leaf_paths_three_levels_and_groups(tmp_path):
    frag = _write(tmp_path, "essentiality")
    schema = PARQUET.logical_schema(frag)
    deep = "geneEssentiality[].depMapEssentiality[].screens[].geneEffect"
    assert PARQUET.leaf_path(deep, schema) == \
        "geneEssentiality.list.element.depMapEssentiality.list.element.screens.list.element.geneEffect"
    assert PARQUET.leaf_path("geneEssentiality", schema) == "geneEssentiality"
    assert PARQUET.leaf_path("geneEssentiality[]", schema) == "geneEssentiality.list.element"
    with pytest.raises(ValueError):
        PARQUET.leaf_path("geneEssentiality[].nope", schema)
    with pytest.raises(ValueError):
        PARQUET.leaf_path("@row.x", schema)
    level3 = PARQUET.read_leaves(frag, [PARQUET.leaf_path(deep, schema)], None)
    native = PARQUET.to_native(level3)
    assert native[0] == {"geneEssentiality": [{"depMapEssentiality": [
        {"screens": [{"geneEffect": -1.5}, {"geneEffect": -0.2}]}, {"screens": [{"geneEffect": -2.0}]}]}]}
    assert native[6]["geneEssentiality"][0]["depMapEssentiality"][0]["screens"] == [None, {"geneEffect": None}]


def test_legacy_list_encodings_map_from_the_footer(tmp_path):
    table = g.build_drugs()
    path = os.path.join(str(tmp_path), "legacy.parquet")
    pq.write_table(table, path, use_compliant_nested_type=False)       # writes drugs.list.item.drugId
    frag = Fragment(path, None, None)
    schema = PARQUET.logical_schema(frag)
    assert PARQUET.leaf_path("drugs[].drugId", schema) == "drugs.list.item.drugId"
    assert PARQUET.to_native(PARQUET.read_leaves(frag, ["drugs[].drugId"], None))[0] == \
        {"drugs": [{"drugId": "CHEMBL1"}, {"drugId": "CHEMBL2"}]}
    assert PARQUET.leaf_path("drugs[].drugId", schema.with_metadata({})) == "drugs.list.element.drugId"


# ---------------------------------------------------------------------------
# F-3 / F-4: the correlated Any, NaN, signed zero, residual rules
# ---------------------------------------------------------------------------

def test_correlated_any_over_three_levels_matches_evaluate(tmp_path):
    frag = _write(tmp_path, "essentiality")
    full = scan_rows(PARQUET, [frag])
    verdicts = {r["id"]: evaluate(g.CORRELATED_ANY, r) for r in full}
    # g1 has a strong liver screen and an expressed liver screen, but not one screen with both
    assert verdicts == {"g1": False, "g2": True, "g3": None, "g4": False, "g5": None, "g6": None, "g7": None}
    assert _ids(scan_rows(PARQUET, [frag], predicate=g.CORRELATED_ANY)) == ["g2"]
    uncorrelated = And((Any("geneEssentiality[].depMapEssentiality[]", Eq("tissueName", "liver")),
                        Cmp("geneEssentiality[].depMapEssentiality[].screens[].geneEffect", "<=", -1.0),
                        Cmp("geneEssentiality[].depMapEssentiality[].screens[].expression", ">=", 10.0)))
    assert "g1" in _ids(scan_rows(PARQUET, [frag], predicate=uncorrelated))


def test_nan_is_null_in_pushdown(tmp_path):
    # one row group, no nulls: footer statistics ignore NaN, so IS NULL must stay residual
    table = pa.table({"id": ["a", "b", "c"], "v": pa.array([0.05, float("nan"), 0.05], pa.float32())})
    frag = write_golden(PARQUET, None, tmp_path, table=table, name="nan")
    schema = PARQUET.logical_schema(frag)
    assert PARQUET.compile(IsNull("v"), schema)[0] is None
    assert _ids(scan_rows(PARQUET, [frag], predicate=IsNull("v"))) == ["b"]
    for p, want in ((Not(IsNull("v")), ["a", "c"]), (Cmp("v", "!=", 0.05), []), (Not(Eq("v", 0.05)), []),
                    (Not(In("v", (0.05,))), []), (Cmp("v", "<", 1.0), ["a", "c"]), (Not(Cmp("v", "<", 1.0)), []),
                    (CmpAbs("v", ">", 0.01), ["a", "c"])):
        assert _ids(scan_rows(PARQUET, [frag], predicate=p)) == want, str(p)
        assert want == [r["id"] for r in PARQUET.to_native(table) if evaluate(storage_typed(
            p, PARQUET.storage_type_of(schema)), r) is True], str(p)


def test_signed_zero_and_int_literals(tmp_path):
    table = pa.table({"id": ["a", "b", "c", "d"], "f": pa.array([-0.0, 0.0, 1.0, None], pa.float64()),
                      "i": pa.array([2, 3, 2 ** 40, None], pa.int64())})
    frag = write_golden(PARQUET, None, tmp_path, table=table, name="zero")
    assert _ids(scan_rows(PARQUET, [frag], predicate=In("f", (0,)))) == ["a", "b"]
    assert _ids(scan_rows(PARQUET, [frag], predicate=Not(In("f", (0,))))) == ["c"]
    assert _ids(scan_rows(PARQUET, [frag], predicate=Eq("i", 2.5))) == []           # never truncated to 2
    assert _ids(scan_rows(PARQUET, [frag], predicate=Cmp("i", ">", 2.5))) == ["b", "c"]
    assert _ids(scan_rows(PARQUET, [frag], predicate=In("i", (2.0, 2.5)))) == ["a"]
    assert _ids(scan_rows(PARQUET, [frag], predicate=Cmp("i", "!=", "2"))) == ["a", "b", "c"]  # str never equal
    assert _ids(scan_rows(PARQUET, [frag], predicate=Eq("i", True))) == []


def test_residual_rules(tmp_path):
    frag = _write(tmp_path, "scalars")
    schema = PARQUET.logical_schema(frag)
    for p in (Contains("tags", "a"), Any("drugs[]", Eq("drugId", "X")), All("x[]", Eq("[]", 1)), IsNull("f32"),
              TextMatch("name", "brca1", "word"), CmpAbs("name", ">", 1), Cmp("flag", "<", True),
              Cmp("ts", ">", 0), Eq("missing_column", 1)):
        pushdown, residual = PARQUET.compile(p, schema)
        assert pushdown is None and residual == p, str(p)
    # casefold text pushes a superset and keeps the exact test as residual
    p = TextMatch("name", "brca1", "casefold_substring")
    pushdown, residual = PARQUET.compile(p, schema)
    assert pushdown is not None and residual == p
    assert _ids(scan_rows(PARQUET, [frag], predicate=p)) == ["ENSG00000169174", "Straße", "back\\slash"]
    p = TextMatch("name", "ss", "casefold_substring")                   # Straße casefolds to strasse
    assert _ids(scan_rows(PARQUET, [frag], predicate=p)) == ["it's", 'say "hi"']
    # one conjunct pushed, one residual: the residual's column joins the projection
    p = And((Eq("grp", "a"), TextMatch("name", "BRCA1", "word")))
    pushdown, residual = PARQUET.compile(p, schema)
    assert pushdown is not None and residual == TextMatch("name", "BRCA1", "word")
    rows = []
    for batch in PARQUET.scan([frag], columns=["id"], predicate=p, partitions={}):
        rows.extend(PARQUET.to_native(batch))
    assert set(rows[0]) == {"id", "name"} and all(r["name"] is not None or True for r in rows)
    assert _ids(scan_rows(PARQUET, [frag], predicate=p)) == ["ENSG00000169174", "back\\slash"]


def test_scan_row_groups_and_projection(tmp_path):
    table = g.build_scalars()
    frag = write_golden(PARQUET, None, tmp_path, table=table, name="rg", row_group_size=3)
    assert PARQUET.stats(frag).row_groups == 4
    rows = scan_rows(PARQUET, [frag], columns=["id", "i32"], row_groups={frag.uri: [1, 3]})
    want = table.to_pylist()
    assert rows == [{"id": r["id"], "i32": r["i32"]} for r in want[3:6] + want[9:]]
    with pytest.raises(ValueError):
        list(PARQUET.scan([frag], columns=["nope"], predicate=None, partitions={}))
    assert list(PARQUET.scan([], columns=None, predicate=None, partitions={})) == []
    sfrag = _write(tmp_path, "struct_nulls")
    assert scan_rows(PARQUET, [sfrag], columns=["id", "s.a"]) == [
        {"id": "s1", "s.a": "x"}, {"id": "s2", "s.a": None}, {"id": "s3", "s.a": None}, {"id": "s4", "s.a": "y"}]
    assert _ids(scan_rows(PARQUET, [sfrag], predicate=IsNull("s.a"))) == ["s2", "s3"]


# ---------------------------------------------------------------------------
# Footers, native conversion, failures
# ---------------------------------------------------------------------------

def test_footer_stats_metadata_and_schema(tmp_path):
    frag = _write(tmp_path, "scalars")
    stats = PARQUET.stats(frag)
    assert stats.rows == 10 and stats.row_groups == 1 and stats.method == "footer"
    i32 = stats.columns["i32"]
    assert i32.storage_type == "int32" and i32.null_count == 1 and (i32.min, i32.max) == (-2147483648, 2147483647)
    assert stats.columns["id"].kind == "string" and stats.columns["f32"].storage_type == "float"
    assert stats.columns["concentration"].max_def_level == 1 and stats.columns["i32"].max_rep_level == 0
    large = PARQUET.stats(_write(tmp_path, "large"))
    assert large.columns["lname"].storage_type == "large_string"
    assert large.columns["ltags.list.element"].storage_type == "large_string"
    meta = PARQUET.metadata(frag)
    assert meta["num_rows"] == "10" and "ARROW:schema" not in meta and meta["created_by"]
    schema = PARQUET.logical_schema(Fragment(frag.uri, None, None, partition={"sourceId": "x", "year": 2020}))
    assert schema.field("year").type == pa.int64() and schema.field("sourceId").type == pa.string()
    assert PATHS_KEY in schema.metadata


def test_to_native_maps_nan_and_no_numpy(tmp_path):
    table = pa.table({"m": pa.array([[("k", 1.5)], None, [("z", float("nan"))]], pa.map_(pa.string(), pa.float64())),
                      "l": pa.array([[float("nan"), 1.0], [], None], pa.list_(pa.float32())),
                      "d": pa.array(["a", "b", "a"]).dictionary_encode()})
    rows = PARQUET.to_native(table)
    assert rows == [{"m": [{"key": "k", "value": 1.5}], "l": [None, 1.0], "d": "a"},
                    {"m": None, "l": [], "d": "b"}, {"m": [{"key": "z", "value": None}], "l": None, "d": "a"}]
    frag = write_golden(PARQUET, None, tmp_path, table=table, name="maps")
    assert PARQUET.leaf_path("m[].key", PARQUET.logical_schema(frag)) == "m.key_value.key"
    assert scan_rows(PARQUET, [frag]) == rows


def test_unreadable_fragment_names_its_partition(tmp_path):
    frag = _write(tmp_path, "scalars")
    g.truncate(frag.uri)
    bad = Fragment(frag.uri, None, None, partition={"sourceId": "europepmc"})
    with pytest.raises(FormatError) as err:
        scan_rows(PARQUET, [bad])
    assert err.value.fragment == frag.uri and err.value.partition == {"sourceId": "europepmc"}
    with pytest.raises(FormatError):
        PARQUET.read_leaves(bad, ["id"], None)
    with pytest.raises(FormatError):
        PARQUET.metadata(Fragment(str(tmp_path / "missing.parquet"), None, None))


def test_scan_schema_uses_declared_partition_types(tmp_path):
    frag = _write(tmp_path, "scalars")
    f = Fragment(frag.uri, None, None, partition={"year": 2020, "sourceId": None})
    schema = scan_schema(PARQUET, [f], {"sourceId": "string", "year": "int64"})
    assert schema.names[-2:] == ["sourceId", "year"] and schema.field("year").type == pa.int64()
    rows = scan_rows(PARQUET, [f], columns=["id", "sourceId", "year"], partitions={"sourceId": "string",
                                                                                     "year": "int64"})
    assert rows[0] == {"id": "ENSG00000169174", "sourceId": None, "year": 2020}
    assert scan_rows(PARQUET, [f], predicate=Eq("sourceId", "x"), partitions={"sourceId": "string"}) == []


def test_write_parquet_helper_round_trips(tmp_path):
    table = g.golden("large").build()
    path = os.path.join(str(tmp_path), "w.parquet")
    g.write_parquet(table, path, row_group_size=2)
    back = pq.read_table(path)
    assert back.schema.remove_metadata() == table.schema and back.num_rows == table.num_rows
    assert pq.ParquetFile(path).metadata.num_row_groups == 3


def test_parquet_module_imports_without_pyarrow():
    code = (
        "import sys\n"
        "sys.modules['pyarrow'] = None\n"
        "sys.modules['pandas'] = None\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from vbt.datalayer.plugins.formats.parquet import ParquetFormat\n"
        "from vbt.datalayer.plugins.conformance import golden\n"
        "p = ParquetFormat()\n"
        "assert p.capabilities == {'tabular', 'pushdown', 'stats', 'nested', 'leaf_projection'}\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
