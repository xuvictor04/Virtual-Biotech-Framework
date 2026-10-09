"""Item keys at real-data scale: nullable item-key parts, item-key checks in Arrow, lists nested in lists, and the
Open Targets download manifest.

Offline (always), each on the smallest fixture with the real shape:

* :func:`items.arrow_items` yields the items :func:`items.explode` yields (null containers, empty lists and null
  items skipped, positions kept), including a null list slot that still spans values;
* the composed key of an item table (R5, R5b) counted in Arrow equals the row scan's count, with nulls not
  distinct, nullable parts, a repeated parent key (items compared across rows), list-valued parts compared as
  canonical lists (target chemicalProbes ``urls``) and three list levels (target_essentiality screens); the Arrow
  path converts no item to Python, and a part rewritten by cleaning falls back to the scan;
* R5b:items of a list in a list is unique per enclosing item, over every row at depth deep;
* ``vbt data ot fetch``: sha1 verification against ``release_data_integrity``, resumable downloads, and the
  ``.download-manifest.json`` the upstream doctor and readiness R2 read.

``VBT_DL_REAL_DATA=<dir>`` (the ``open_targets/25.09`` directory): deep item-key checks of the real target,
target_essentiality and l2g_prediction item tables, and the manifest of the downloaded tables.
``VBT_DL_NETWORK=1``: the release's integrity list and one small table fetched from the EBI FTP over HTTPS.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import yaml  # noqa: E402

from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service import checks as _checks  # noqa: E402
from vbt.datalayer.service import items as _items  # noqa: E402
from vbt.datalayer.service.checks import check_table  # noqa: E402
from vbt.datalayer.rowkey import canonical  # noqa: E402
from vbt.datalayer.service.reader import TableReader  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402


def make_ctx(tmp: Path, tables: dict[str, Any]) -> ServiceContext:
    desc = {"schema": "vbt.datasource/1", "source": "s", "title": "s", "root": str(tmp / "data"),
            "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
            "id_types": {}, "tables": tables}
    (tmp / "sources").mkdir(parents=True, exist_ok=True)
    (tmp / "overlays").mkdir(parents=True, exist_ok=True)
    (tmp / "sources" / "s.yaml").write_text(yaml.safe_dump(desc, sort_keys=False))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp / "sources"), "overlays_dir": str(tmp / "overlays"),
                                       "cache_dir": str(tmp / "cache")}, project_root=tmp)
    return ServiceContext(settings)


def write(tmp: Path, name: str, rows: list[dict[str, Any]], schema: Any, part: str = "part-00000",
          row_group_size: int | None = None) -> None:
    d = tmp / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), d / f"{part}.parquet", row_group_size=row_group_size)


def checks(model: Any, name: str) -> list[Any]:
    return [c for c in model.checks if c.name == name]


def failing(model: Any, name: str) -> list[Any]:
    return [c for c in checks(model, name) if not c.ok and c.level == "error"]


# ---------------------------------------------------------------------------- arrow_items == explode

LEAF = pa.struct([("c", pa.string()), ("tags", pa.list_(pa.string()))])
ITEM = pa.struct([("k", pa.string()), ("b", pa.list_(LEAF))])
NESTED = pa.schema([("id", pa.string()), ("a", pa.list_(ITEM))])


def _maybe(rng: random.Random, value: Any, p: float = 0.15) -> Any:
    return None if rng.random() < p else value


def _nested_rows(rng: random.Random, n: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(n):
        a = None
        if rng.random() > 0.15:
            a = []
            for _ in range(rng.randrange(0, 4)):
                if rng.random() < 0.1:
                    a.append(None)                                   # a null item
                    continue
                b = None if rng.random() < 0.2 else [
                    _maybe(rng, {"c": _maybe(rng, rng.choice("xyz")),
                                 "tags": _maybe(rng, [_maybe(rng, rng.choice("pq")) for _ in range(rng.randrange(3))])})
                    for _ in range(rng.randrange(0, 4))]
                a.append({"k": _maybe(rng, rng.choice("KLM")), "b": b})
        rows.append({"id": f"r{i}", "a": a})
    return rows


def test_arrow_items_yield_the_items_explode_yields():
    rng = random.Random(7)
    rows = _nested_rows(rng, 300)
    tbl = pa.Table.from_pylist(rows, schema=NESTED)
    for path, parts in (("a[]", ["a[].k", "a[]#"]),
                        ("a[].b[]", ["a[].k", "a[].b[].c", "a[].b[]#", "a[]#", "a[].b[].tags"])):
        lvls = _items.levels(path)
        got = _items.arrow_items(tbl, lvls, parts)
        want_rows, want_parts = [], {p: [] for p in parts}
        for r, row in enumerate(rows):
            for view, pos in _items.explode(row, lvls):
                want_rows.append(r)
                for p, v in zip(parts, _items.key_values(view, parts, pos, lvls)):
                    want_parts[p].append(v)
        assert got.rows.tolist() == want_rows, path
        for p in parts:
            assert got.parts[p].to_pylist() == want_parts[p], (path, p)
        # the parent of an item: the row for a list in the row, the enclosing item for a list in a list
        if len(lvls) == 1:
            assert got.parents.tolist() == want_rows
        else:
            outer = _items.arrow_items(tbl, lvls[:1], [])
            assert [int(outer.rows[p]) for p in got.parents] == want_rows


def test_a_null_list_slot_spanning_values_is_skipped():
    """``list_parent_indices`` counts the values a null list slot still spans; ``flatten`` skips them."""
    values = pa.array([{"k": "a"}, {"k": "b"}, {"k": "c"}, {"k": "d"}])
    lists = pa.ListArray.from_arrays(pa.array([0, 2, 3, 4], pa.int32()), values, mask=pa.array([False, True, False]))
    tbl = pa.table({"id": ["r0", "r1", "r2"], "a": lists})
    got = _items.arrow_items(tbl, _items.levels("a[]"), ["a[].k", "a[]#"])
    assert got.rows.tolist() == [0, 0, 2]
    assert got.parts["a[].k"].to_pylist() == ["a", "b", "d"] and got.parts["a[]#"].to_pylist() == [0, 1, 0]
    sliced = _items.arrow_items(tbl.slice(1), _items.levels("a[]"), ["a[].k"])
    assert sliced.rows.tolist() == [1] and sliced.parts["a[].k"].to_pylist() == ["d"]


def test_key_parts_compare_in_arrow_as_canonical_keys_render_them():
    """Two values are one key part in Arrow exactly when ``canonical`` renders them alike: NaN is null, lists are
    multisets, structs compare field by field, in-band codes are null after cleaning."""
    from vbt.datalayer.rowkey import render_value

    cases = [pa.array([1.5, float("nan"), None, 1.5]),
             pa.array([["a", "b"], ["b", "a"], [], None, ["a", None], [None, "a"], ["a"]]),
             pa.array([[{"n": "A", "u": "1"}, {"n": "B", "u": None}], [{"n": "B", "u": None}, {"n": "A", "u": "1"}],
                       [None], [{"n": "A", "u": "1"}]]),
             pa.array([{"x": 1, "ids": ["p", "q"]}, {"x": 1, "ids": ["q", "p"]}, None, {"x": None, "ids": None}])]
    for arr in cases:
        codes, _ = _checks._codes(_checks._comparable(arr))
        rendered = [render_value(v) for v in arr.to_pylist()]
        for i in range(len(arr)):
            for j in range(len(arr)):
                assert (codes[i] == codes[j]) == (rendered[i] == rendered[j]), (arr.type, i, j, rendered)
    cleaned = _checks._cleaned(pa.array([[-1, 2], [3, -1], None]), frozenset({"-1"}))
    assert cleaned.to_pylist() == [[None, 2], [3, None], None]
    assert _checks._cleaned(pa.array([-1.0, -1, 2.0]), frozenset({"-1"})).to_pylist() == [-1.0, -1.0, 2.0]


def test_a_part_off_the_container_path_is_not_read_in_arrow():
    lvls = _items.levels("a[].b[]")
    assert _items.part_level("a[].b[].c", lvls) == (1, "c")
    assert _items.part_level("a[].k", lvls) == (0, "k")
    assert _items.part_level("a[].b[]#", lvls) == (1, "#")
    with pytest.raises(_items.ArrowUnsupported):
        _items.part_level("x[].k", lvls)
    with pytest.raises(_items.ArrowUnsupported):
        _items.part_level("a[].b[].tags[].z", lvls)


# ---------------------------------------------------------------------------- composed keys: Arrow == scan


def _scan_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_checks, "_arrow_item_keys", lambda *a, **k: None)


def _no_item_scan(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fail the test if an item table's key check scans rows (the Arrow path converts no item to Python)."""
    real = TableReader.scan
    seen: list[str] = []

    def scan(self, predicate=None, **kw):
        if self.levels and kw.get("columns") == []:
            seen.append(self.ref)
        return real(self, predicate, **kw)

    monkeypatch.setattr(TableReader, "scan", scan)
    return seen


def _probes(tmp: Path, rows: list[dict[str, Any]], *, nullable: list[str], parent_key: list[str] | None = None,
            row_group_size: int | None = 4, missing: list[str] | None = None) -> ServiceContext:
    """target chemicalProbes: (id, origin, drugId, urls), drugId nullable (25.09: null in 141 of 5,090 items)."""
    item = pa.struct([("id", pa.string()), ("origin", pa.string()), ("drugId", pa.string()),
                      ("urls", pa.list_(pa.string())), ("score", pa.float64())])
    schema = pa.schema([("gene", pa.string()), ("probes", pa.list_(item))])
    half = len(rows) // 2
    write(tmp, "target", rows[:half], schema, part="part-00000", row_group_size=row_group_size)
    write(tmp, "target", rows[half:], schema, part="part-00001", row_group_size=row_group_size)
    fields = {"id": {"role": "identifier"}, "origin": {"role": "category"}, "drugId": {"role": "identifier"},
              "urls": {"role": "payload"}, "score": {"role": "payload"}}
    if missing:
        fields["origin"] = {"role": "category", "missing_values": missing}
    return make_ctx(tmp, {
        "target": {"kind": "entity", "path": "target", "grain": "gene",
                   "key": {"columns": parent_key if parent_key is not None else ["gene"]},
                   "columns": {"gene": {"role": "identifier"},
                               "probes": {"role": "nested",
                                          "item_key": {"columns": ["id", "origin", "drugId", "urls"],
                                                       "nullable": nullable},
                                          "fields": fields}}},
        "target_probes": {"kind": "fact", "items_of": {"table": "target", "path": "probes[]"}, "grain": "probe",
                          "key": {"columns": [], "check": "sampled"}}})


def _probe_rows(rng: random.Random, n: int, *, repeat_genes: bool = False) -> list[dict[str, Any]]:
    rows = []
    for g in range(n):
        probes = []
        for _ in range(rng.randrange(0, 6)):
            urls = rng.choice([None, [], ["u1"], ["u1", "u2"], ["u2", "u1"], ["u2", None]])
            probes.append({"id": rng.choice(["P1", "P2", "P3"]), "origin": rng.choice(["CP", "GL", None]),
                           "drugId": rng.choice(["C1", None]), "urls": urls, "score": rng.random()})
        gene = f"g{g % (n // 3) if repeat_genes else g}"
        rows.append({"gene": gene, "probes": probes if rng.random() > 0.1 else None})
    return rows


def _expected(rows: list[dict[str, Any]]) -> tuple[int, set[str]]:
    """Repeats of the composed key (gene, id, origin, drugId, urls) and the repeated keys, rendered as rows are."""
    seen: dict[str, int] = {}
    for row in rows:
        for p in row["probes"] or []:
            k = canonical([row["gene"], p["id"], p["origin"], p["drugId"], p["urls"]],
                          ["string", "string", "string", "string", "list<element: string>"])
            seen[k] = seen.get(k, 0) + 1
    return sum(n - 1 for n in seen.values()), {k for k, n in seen.items() if n > 1}


@pytest.mark.parametrize("seed,repeat_genes", [(1, False), (2, False), (3, True), (4, True)])
def test_the_item_key_check_in_arrow_counts_what_the_scan_counts(tmp_path, monkeypatch, seed, repeat_genes):
    rows = _probe_rows(random.Random(seed), 60, repeat_genes=repeat_genes)
    repeats, repeated = _expected(rows)
    assert repeats > 0
    for nullable in (["drugId"], []):
        ctx = _probes(tmp_path / f"a{len(nullable)}", rows, nullable=nullable)
        seen = _no_item_scan(monkeypatch)
        fast = check_table(ctx, "s.target_probes", "deep")
        assert seen == [], "the Arrow path scanned rows"
        monkeypatch.undo()
        _scan_only(monkeypatch)
        slow = check_table(make_ctx_from(ctx), "s.target_probes", "deep")
        monkeypatch.undo()
        assert fast.key_check.duplicates == slow.key_check.duplicates == repeats
        assert fast.key_check.null_counts == slow.key_check.null_counts
        assert [c.detail for c in checks(fast, "R5")] == [c.detail for c in checks(slow, "R5")]
        assert fast.status == slow.status == "key_violation"
        (r5b,) = failing(fast, "R5b")
        examples = r5b.detail.split("(every key): ", 1)[1].split(", [")
        assert examples and all((e if e.startswith("[") else "[" + e) in repeated for e in examples), r5b.detail


def make_ctx_from(ctx: ServiceContext) -> ServiceContext:
    """A fresh context over the same descriptors (no cached footers or readers)."""
    return ServiceContext(ctx.settings)


def test_list_parts_compare_as_canonical_lists(tmp_path, monkeypatch):
    """Canonical keys render a list part order-insensitively: ["u1", "u2"] and ["u2", "u1"] are one key, while an
    empty list and a null list differ."""
    rows = [{"gene": "g1", "probes": [{"id": "P1", "origin": "CP", "drugId": None, "urls": ["u1", "u2"], "score": 1.0},
                                      {"id": "P1", "origin": "CP", "drugId": None, "urls": ["u2", "u1"], "score": 2.0}]},
            {"gene": "g2", "probes": [{"id": "P1", "origin": "CP", "drugId": None, "urls": [], "score": 1.0},
                                      {"id": "P1", "origin": "CP", "drugId": None, "urls": None, "score": 2.0}]}]
    ctx = _probes(tmp_path / "a", rows, nullable=["drugId"])
    seen = _no_item_scan(monkeypatch)
    model = check_table(ctx, "s.target_probes", "deep")
    assert seen == []
    assert model.key_check.duplicates == 1 and failing(model, "R5b")
    assert '["u1","u2"]' in failing(model, "R5b")[0].detail
    monkeypatch.undo()
    _scan_only(monkeypatch)
    assert check_table(make_ctx_from(ctx), "s.target_probes", "deep").key_check.duplicates == 1


def test_list_of_struct_parts_compare_as_canonical_lists(tmp_path, monkeypatch):
    """25.09 chemicalProbes ``urls`` is a list of ``{niceName, url}`` structs and ``origin`` a list of strings: the
    Arrow path compares them as the scan's canonical keys do (lists as multisets, structs field by field)."""
    url = pa.struct([("niceName", pa.string()), ("url", pa.string())])
    item = pa.struct([("id", pa.string()), ("origin", pa.list_(pa.string())), ("urls", pa.list_(url))])
    schema = pa.schema([("gene", pa.string()), ("probes", pa.list_(item))])
    a, b = {"niceName": "A", "url": "u1"}, {"niceName": "B", "url": "u2"}
    rows = [{"gene": "g1", "probes": [{"id": "P", "origin": ["x", "y"], "urls": [a, b]},
                                      {"id": "P", "origin": ["y", "x"], "urls": [b, a]}]},        # one key twice
            {"gene": "g2", "probes": [{"id": "P", "origin": ["x"], "urls": [a]},
                                      {"id": "P", "origin": ["x"], "urls": [{"niceName": "A", "url": None}]},
                                      {"id": "P", "origin": ["x"], "urls": [None]},
                                      {"id": "P", "origin": ["x"], "urls": []}]}]                  # four keys
    write(tmp_path / "a", "target", rows, schema)
    tables = {"target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["gene"]},
                         "columns": {"gene": {"role": "identifier"},
                                     "probes": {"role": "nested", "item_key": ["id", "origin", "urls"],
                                                "fields": {"id": {"role": "identifier"},
                                                           "origin": {"role": "category", "path": "[]"},
                                                           "urls": {"role": "payload"}}}}},
              "target_probes": {"kind": "fact", "items_of": {"table": "target", "path": "probes[]"},
                                "grain": "probe", "key": {"columns": [], "check": "sampled"}}}
    ctx = make_ctx(tmp_path / "a", tables)
    seen = _no_item_scan(monkeypatch)
    fast = check_table(ctx, "s.target_probes", "deep")
    assert seen == [] and fast.key_check.duplicates == 1
    monkeypatch.undo()
    _scan_only(monkeypatch)
    assert check_table(make_ctx_from(ctx), "s.target_probes", "deep").key_check.duplicates == 1


def test_in_band_codes_of_a_part_read_as_null_in_arrow(tmp_path, monkeypatch):
    """origin "NA" reads as null after cleaning (25.09 drug_indications: maxPhaseForIndication -1), so ("NA") and
    (null) are one key in Arrow as in rows; a part cleaned by a condition is counted on rows."""
    rows = [{"gene": "g1", "probes": [{"id": "P1", "origin": "NA", "drugId": None, "urls": None, "score": 1.0},
                                      {"id": "P1", "origin": None, "drugId": None, "urls": None, "score": 1.0}]}]
    ctx = _probes(tmp_path / "a", rows, nullable=["drugId", "origin", "urls"], missing=["NA"])
    seen = _no_item_scan(monkeypatch)
    model = check_table(ctx, "s.target_probes", "deep")
    assert seen == [] and model.key_check.duplicates == 1
    assert '["g1","P1",null,null,null]' in failing(model, "R5b")[0].detail
    monkeypatch.undo()
    _scan_only(monkeypatch)
    assert check_table(make_ctx_from(ctx), "s.target_probes", "deep").key_check.duplicates == 1
    monkeypatch.undo()
    cond = _probes(tmp_path / "b", rows, nullable=["drugId", "origin", "urls"])
    desc = yaml.safe_load((tmp_path / "b" / "sources" / "s.yaml").read_text())
    desc["tables"]["target"]["columns"]["probes"]["fields"]["origin"]["unknown_when"] = [{"column": "id", "eq": "P9"}]
    (tmp_path / "b" / "sources" / "s.yaml").write_text(yaml.safe_dump(desc, sort_keys=False))
    seen = _no_item_scan(monkeypatch)
    check_table(make_ctx_from(cond), "s.target_probes", "deep")
    assert seen == ["s.target_probes"], "a conditional cleaning is decided on rows"


def test_items_of_rows_sharing_a_parent_key_are_compared_across_rows(tmp_path, monkeypatch):
    rows = [{"gene": "g1", "probes": [{"id": "P1", "origin": "CP", "drugId": "C1", "urls": None, "score": 1.0}]},
            {"gene": "g2", "probes": [{"id": "P1", "origin": "CP", "drugId": "C1", "urls": None, "score": 1.0}]},
            {"gene": "g1", "probes": [{"id": "P1", "origin": "CP", "drugId": "C1", "urls": None, "score": 1.0},
                                      {"id": "P2", "origin": "CP", "drugId": "C1", "urls": None, "score": 1.0}]},
            {"gene": "g3", "probes": [{"id": "P2", "origin": "CP", "drugId": "C1", "urls": None, "score": 1.0},
                                      {"id": "P2", "origin": "CP", "drugId": "C1", "urls": None, "score": 1.0}]}]
    ctx = _probes(tmp_path / "a", rows, nullable=["drugId"], row_group_size=1)
    seen = _no_item_scan(monkeypatch)
    model = check_table(ctx, "s.target_probes", "deep")
    assert seen == []
    assert model.key_check.duplicates == 2          # g1/P1 in two rows (two files), g3/P2 twice in one row
    details = failing(model, "R5b")[0].detail
    assert '["g1","P1","CP","C1",null]' in details and '["g3","P2","CP","C1",null]' in details


def test_a_sampled_item_key_check_keeps_whole_parent_rows_in_arrow(tmp_path, monkeypatch):
    rows = [{"gene": f"g{g}", "probes": [{"id": f"P{i}", "origin": "CP", "drugId": "C1", "urls": ["u"], "score": 1.0}
                                         for i in range(10)]} for g in range(300)]
    ctx = _probes(tmp_path / "a", rows, nullable=["drugId"], row_group_size=50)
    seen = _no_item_scan(monkeypatch)
    model = check_table(ctx, "s.target_probes", "standard", sample_rows=600)
    assert seen == []
    (r5,) = checks(model, "R5")
    scanned = int(r5.detail.rsplit("(", 1)[1].split()[0])
    assert 0 < scanned < 3000 and scanned % 10 == 0, r5.detail
    assert "of parent rows" in checks(model, "R5b")[0].detail
    again = check_table(make_ctx_from(ctx), "s.target_probes", "standard", sample_rows=600)
    assert checks(again, "R5")[0].detail == r5.detail                 # the sample is decided by the key values


ESS = pa.schema([("id", pa.string()), ("geneEssentiality", pa.list_(pa.struct([
    ("isEssential", pa.bool_()),
    ("depMapEssentiality", pa.list_(pa.struct([
        ("tissueId", pa.string()), ("tissueName", pa.string()),
        ("screens", pa.list_(pa.struct([("depmapId", pa.string()), ("geneEffect", pa.float64())])))])))])))])


def _essentiality(tmp: Path, rows: list[dict[str, Any]]) -> ServiceContext:
    write(tmp, "ess", rows, ESS, row_group_size=3)
    screens = {"role": "nested", "item_key": ["depmapId"],
               "fields": {"depmapId": {"role": "identifier"}, "geneEffect": {"role": "payload"}}}
    tissues = {"role": "nested", "item_key": {"columns": ["tissueId"], "nullable": ["tissueId"]},
               "fields": {"tissueId": {"role": "category"}, "tissueName": {"role": "label", "of": "tissueId"},
                          "screens": screens}}
    return make_ctx(tmp, {
        "ess": {"kind": "entity_detail", "path": "ess", "grain": "gene", "key": {"columns": ["id"], "check": "full"},
                "columns": {"id": {"role": "identifier"},
                            "geneEssentiality": {"role": "nested", "item_key": {"identity": "position", "max_items": 1},
                                                 "fields": {"isEssential": {"role": "flag"},
                                                            "depMapEssentiality": tissues}}}},
        "ess_screens": {"kind": "fact", "grain": "screen", "key": {"columns": [], "check": "sampled"},
                        "items_of": {"table": "ess", "path": "geneEssentiality[].depMapEssentiality[].screens[]"}}})


def _ess_rows(rng: random.Random, n: int) -> list[dict[str, Any]]:
    rows = []
    for g in range(n):
        tissues = [{"tissueId": t, "tissueName": t or "other",
                    "screens": [{"depmapId": f"ACH-{rng.randrange(6)}", "geneEffect": rng.random()}
                                for _ in range(rng.randrange(0, 5))]}
                   for t in rng.sample(["UBERON_1", "UBERON_2", None, "UBERON_3"], rng.randrange(1, 4))]
        rows.append({"id": f"ENSG{g}", "geneEssentiality": [{"isEssential": False, "depMapEssentiality": tissues}]})
    return rows


def test_three_list_levels_with_a_null_tissue_group(tmp_path, monkeypatch):
    """target_essentiality screens: (id, tissueId, depmapId) with tissueId null for the "other" group (25.09:
    1,593,364 of 20,955,265 screens); the null group is one key value (NULLS NOT DISTINCT)."""
    rows = _ess_rows(random.Random(11), 40)
    ctx = _essentiality(tmp_path / "a", rows)
    seen = _no_item_scan(monkeypatch)
    fast = check_table(ctx, "s.ess_screens", "deep")
    assert seen == []
    monkeypatch.undo()
    _scan_only(monkeypatch)
    slow = check_table(make_ctx_from(ctx), "s.ess_screens", "deep")
    assert fast.key_check.duplicates == slow.key_check.duplicates > 0         # random ACH- ids repeat in a tissue
    assert fast.key_check.null_counts == slow.key_check.null_counts == {}     # tissueId is a nullable part
    assert fast.status == slow.status == "key_violation"


# ---------------------------------------------------------------------------- R5b:items, a list in a list


INDICATIONS = pa.schema([("id", pa.string()), ("indications", pa.list_(pa.struct([
    ("disease", pa.string()), ("references", pa.list_(pa.struct([("source", pa.string())])))])))])


def _indications(tmp: Path, rows: list[dict[str, Any]]) -> ServiceContext:
    write(tmp, "drug_indication", rows, INDICATIONS, row_group_size=2)
    refs = {"role": "nested", "item_key": ["source"], "fields": {"source": {"role": "category"}}}
    return make_ctx(tmp, {"drug_indication": {
        "kind": "entity_detail", "path": "drug_indication", "grain": "drug", "key": {"columns": ["id"]},
        "columns": {"id": {"role": "identifier"},
                    "indications": {"role": "nested", "item_key": ["disease"],
                                    "fields": {"disease": {"role": "identifier"}, "references": refs}}}}})


def test_item_keys_of_a_list_in_a_list_are_unique_per_enclosing_item_over_every_row(tmp_path):
    """drug_indication (25.09: 61,629 indications): one source cited under two indications of a drug is not a repeat
    (R1 had 1,130 of 2,000 sampled rows flagged when the references of a row were pooled)."""
    rows = [{"id": f"CHEMBL{i}", "indications": [
        {"disease": "D1", "references": [{"source": "ClinicalTrials"}, {"source": "DailyMed"}]},
        {"disease": "D2", "references": [{"source": "ClinicalTrials"}]}]} for i in range(7)]
    ok = check_table(_indications(tmp_path / "ok", rows), "s.drug_indication", "deep")
    assert not failing(ok, "R5b:items"), [c.detail for c in failing(ok, "R5b:items")]
    rows[5]["indications"][1]["references"].append({"source": "ClinicalTrials"})
    rows[6]["indications"].append({"disease": "D1", "references": []})
    bad = check_table(_indications(tmp_path / "bad", rows), "s.drug_indication", "deep")
    found = {c.detail.split(":")[0]: c.detail for c in failing(bad, "R5b:items")}
    assert "1 row(s)" in found["indications.references"] and "every row" in found["indications.references"]
    assert '["ClinicalTrials"]' in found["indications.references"]
    assert "1 row(s)" in found["indications"] and '["D1"]' in found["indications"]
    for row in rows:                                     # the Python rule agrees row by row
        assert bool(_items.item_key_duplicates(row, "indications[].references", ["source"])) == \
            (row["id"] == "CHEMBL5")


def test_r5b_items_at_standard_depth_reads_sampled_row_groups_in_arrow(tmp_path, monkeypatch):
    """The standard check reads whole row groups in Arrow until the relation sample's rows are covered (the 2,000
    sampled 25.09 target_essentiality genes converted to Python held 2.3 M screens: 1.97 GB); a container the
    Arrow path cannot read is still sampled as rows."""
    rows = [{"id": f"CHEMBL{i}", "indications": [{"disease": "D1", "references": [{"source": "A"}, {"source": "B"}]}]}
            for i in range(30)]
    rows[7]["indications"][0]["references"].append({"source": "A"})
    monkeypatch.setattr(_checks, "_RELATION_SAMPLE", 3)
    ctx = _indications(tmp_path / "a", rows)                      # row groups of 2 rows
    seen = []
    real = TableReader.sample_rows
    monkeypatch.setattr(TableReader, "sample_rows", lambda self, *a, **k: seen.append(1) or real(self, *a, **k))
    model = check_table(ctx, "s.drug_indication", "standard")
    assert seen == [], "no row was sampled as Python"
    details = [c.detail for c in checks(model, "R5b:items")]
    found = _checks._arrow_nested_repeats(ctx.reader("s.drug_indication"), "indications[].references", ["source"],
                                          "key", 3)
    assert found is not None and found[3:] == (4, 2)            # two row groups of 2 rows cover 3 rows
    assert (found[0] == 1) == any("4 rows in 2 sampled row group(s)" in d for d in details), details
    full = check_table(make_ctx_from(ctx), "s.drug_indication", "deep")
    assert "1 row(s) repeat an item key under one parent (every row" in failing(full, "R5b:items")[0].detail
    monkeypatch.setattr(_checks, "_arrow_nested_repeats", lambda *a, **k: None)
    model = check_table(make_ctx_from(ctx), "s.drug_indication", "standard")
    assert seen, "a container the Arrow path cannot read is sampled as rows"


def test_a_struct_container_is_counted_as_null_or_present(tmp_path):
    """25.09 target ``tep`` is a struct (41 of 78,726 genes have one); the list path read no column for ``tep[]`` and
    reported every gene's TEP as null."""
    tep = pa.struct([("description", pa.string()), ("url", pa.string())])
    schema = pa.schema([("id", pa.string()), ("tep", tep), ("refs", pa.list_(pa.string()))])
    rows = [{"id": f"g{i}", "tep": {"description": "d", "url": None} if i % 4 == 0 else None,
             "refs": None if i % 3 else ["r"]} for i in range(12)]
    write(tmp_path, "target", rows, schema, row_group_size=5)
    ctx = make_ctx(tmp_path, {"target": {
        "kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["id"]},
        "columns": {"id": {"role": "identifier"},
                    "tep": {"role": "nested", "null_means": "not_assessed",
                            "fields": {"description": {"role": "text"}, "url": {"role": "payload"}}},
                    "refs": {"role": "nested", "null_means": "unknown", "item_key": {"identity": "value"},
                             "fields": {}}}}})
    model = check_table(ctx, "s.target", "standard")
    details = {c.detail.split(":")[0]: c.detail for c in checks(model, "R6:containers")}
    assert details["tep"] == "tep: 9 null, 3 present (a struct, not a list)", details
    assert details["refs"].startswith("refs: 8 null, 0 empty, 4 non-empty list(s)"), details


# ---------------------------------------------------------------------------- vbt data ot: fetch and manifest


class _Release:
    """A fake release site: ``<release>/output/<table>/...parquet`` and ``release_data_integrity`` (+ ``.sha1``),
    served with ``Range`` support. ``corrupt`` paths are served with a flipped byte; ``cut`` paths are cut short
    once (the next request resumes)."""

    def __init__(self, root: Path, release: str = "25.09") -> None:
        import hashlib
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.root = root
        self.release = release
        self.files: dict[str, bytes] = {}
        self.requests: list[tuple[str, str]] = []
        self.corrupt: set[str] = set()
        self.cut: set[str] = set()
        out = root / release / "output"
        for rel, rows in (("so/so.parquet", 3), ("go/part-00000-a.parquet", 5), ("go/part-00001-a.parquet", 4),
                          ("evidence/sourceId=chembl/part-00000-b.parquet", 6)):
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"id": [f"{rel}-{i}" for i in range(rows)]}), path)
            self.files[rel] = path.read_bytes()
        lines = [f"{hashlib.sha1(b).hexdigest()}  ./output/{rel}" for rel, b in self.files.items()]
        lines += ["0" * 40 + "  ./output/go/_SUCCESS", "1" * 40 + "  ./input/target/x.json", "2" * 40 + "  ./webapp"]
        text = ("\n".join(lines) + "\n").encode()
        (root / release / "release_data_integrity").write_bytes(text)
        (root / release / "release_data_integrity.sha1").write_text(
            f"{hashlib.sha1(text).hexdigest()}  release_data_integrity\n")
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, head: bool) -> None:
                path = root / self.path.lstrip("/")
                rng = self.headers.get("Range")
                outer.requests.append((self.command, self.path + (f" {rng}" if rng else "")))
                if not path.is_file():
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                blob = path.read_bytes()
                rel = self.path.split("/output/", 1)[-1]
                if rel in outer.corrupt:
                    blob = blob[:20] + bytes([blob[20] ^ 1]) + blob[21:]
                start = int(rng.split("=")[1].split("-")[0]) if rng else 0
                body = blob[start:]
                self.send_response(206 if rng else 200)
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{len(blob) - 1}/{len(blob)}")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if head:
                    return
                if rel in outer.cut and not rng:
                    outer.cut.discard(rel)
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.close_connection = True               # cut short: the client resumes from here
                    return
                self.wfile.write(body)

            def do_HEAD(self) -> None:  # noqa: N802 - http.server API
                self._send(True)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                self._send(False)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.site = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def release(tmp_path):
    rel = _Release(tmp_path / "site")
    yield rel
    rel.close()


def _manifest(dest: Path) -> dict[str, Any]:
    import json

    return json.loads((dest / ".download-manifest.json").read_text())


def test_fetch_verifies_sha1_and_writes_the_upstream_manifest(release, tmp_path):
    from vbt.data.opentargets import OpenTargetsRelease, output_base

    dest = tmp_path / "ot"
    with OpenTargetsRelease("25.09", site=release.site, retry_wait=0) as ot:
        rep = ot.fetch(["so", "go", "evidence"], dest, workers=2)
    assert not rep.failed and sorted(r.rel for r in rep.results) == sorted(release.files)
    for rel, blob in release.files.items():
        assert (dest / rel).read_bytes() == blob
    assert not list(dest.rglob("*.part"))
    data = _manifest(dest)
    assert data["release"] == "25.09" and data["base"] == output_base("25.09", release.site)
    assert data["complete"] is True and data["expected_files"] == len(data["files"]) == 4
    assert data["tables"] == ["evidence", "go", "so"] and data["archive_files"] == 4
    entry = data["files"]["evidence/sourceId=chembl/part-00000-b.parquet"]
    assert entry["bytes"] == len(release.files["evidence/sourceId=chembl/part-00000-b.parquet"])
    assert len(entry["sha256"]) == 64 and len(entry["sha1"]) == 40
    assert entry["url"] == data["base"] + "evidence/sourceId=chembl/part-00000-b.parquet"
    assert (dest / "_release" / "release_data_integrity").is_file()
    # a second fetch downloads nothing: the files are hashed in place
    release.requests.clear()
    with OpenTargetsRelease("25.09", site=release.site, retry_wait=0) as ot:
        again = ot.fetch(["go"], dest)
    assert {r.status for r in again.results} == {"present"}
    assert not [r for r in release.requests if ".parquet" in r[1]]
    assert _manifest(dest)["tables"] == ["evidence", "go", "so"]


def test_a_file_that_fails_its_sha1_never_takes_its_name(release, tmp_path):
    from vbt.data.opentargets import OpenTargetsRelease

    release.corrupt.add("go/part-00001-a.parquet")
    dest = tmp_path / "ot"
    with OpenTargetsRelease("25.09", site=release.site, retry_wait=0) as ot:
        rep = ot.fetch(["so", "go"], dest)
    (bad,) = rep.failed
    assert bad.rel == "go/part-00001-a.parquet" and "sha1" in bad.error
    assert not (dest / bad.rel).exists() and not (dest / (bad.rel + ".part")).exists()
    data = _manifest(dest)
    assert data["tables"] == ["so"] and all(not k.startswith("go/") for k in data["files"])


def test_an_interrupted_download_resumes_with_a_range_request(release, tmp_path):
    from vbt.data.opentargets import OpenTargetsRelease

    rel = "evidence/sourceId=chembl/part-00000-b.parquet"
    blob = release.files[rel]
    dest = tmp_path / "ot"
    (dest / rel).parent.mkdir(parents=True)
    (dest / (rel + ".part")).write_bytes(blob[: len(blob) // 2])          # left by an interrupted run
    with OpenTargetsRelease("25.09", site=release.site, retry_wait=0) as ot:
        rep = ot.fetch(["evidence"], dest)
    assert not rep.failed and (dest / rel).read_bytes() == blob
    assert [path for _m, path in release.requests if rel in path] == \
        [f"/25.09/output/{rel} bytes={len(blob) // 2}-"]
    # a connection cut mid-file is retried, and the file is complete
    release.cut.add("so/so.parquet")
    with OpenTargetsRelease("25.09", site=release.site, retry_wait=0) as ot:
        rep = ot.fetch(["so"], dest)
    assert not rep.failed and (dest / "so" / "so.parquet").read_bytes() == release.files["so/so.parquet"]


def test_unknown_tables_and_the_size_limit_are_refused(release, tmp_path):
    from vbt.data.opentargets import OpenTargetsRelease

    with OpenTargetsRelease("25.09", site=release.site, retry_wait=0) as ot:
        with pytest.raises(ValueError, match="not a table of release 25.09: targets"):
            ot.fetch(["targets"], tmp_path / "ot")
        with pytest.raises(ValueError, match="over the"):
            ot.fetch(["go"], tmp_path / "ot", max_bytes=10)
        plan = ot.fetch(["go"], tmp_path / "ot", dry_run=True)
    assert sorted(r.rel for r in plan.results) == ["go/part-00000-a.parquet", "go/part-00001-a.parquet"]
    assert not (tmp_path / "ot" / "go").exists()


def _local_release(tmp: Path, site: _Release) -> Path:
    """A downloaded directory as R1 left it: tables, ``_release/`` with the integrity list, ``_samples/``."""
    import shutil

    dest = tmp / "ot"
    for rel, blob in site.files.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_bytes(blob)
    (dest / "go" / "_SUCCESS").write_text("")
    (dest / "_release").mkdir()
    for name in ("release_data_integrity", "release_data_integrity.sha1"):
        shutil.copy(site.root / "25.09" / name, dest / "_release" / name)
    (dest / "_samples" / "interval").mkdir(parents=True)
    (dest / "_samples" / "interval" / "part-0.parquet").write_bytes(site.files["so/so.parquet"])
    return dest


def test_the_manifest_of_tables_already_downloaded(release, tmp_path):
    """Nothing is downloaded: a table whose files all match the list is recorded; one missing a file, holding a file
    the release does not list, or with a changed file is left out (R1's directory: 31 tables, _samples, _footers)."""
    from vbt.data.opentargets import load_integrity, write_manifest

    dest = _local_release(tmp_path, release)
    (dest / "go" / "part-00001-a.parquet").unlink()
    (dest / "evidence" / "sourceId=chembl" / "part-00099-x.parquet").write_bytes(release.files["so/so.parquet"])
    integrity, about = load_integrity("25.09", dest=dest, site="http://127.0.0.1:9")    # no request: local copy
    assert about["from"].endswith("_release/release_data_integrity")
    rep = write_manifest(dest, integrity, about=about)
    assert rep.tables == {"so": 1}
    assert "1 of 2 file(s) missing" in rep.skipped["go"] and "not in the release" in rep.skipped["evidence"]
    data = _manifest(dest)
    assert list(data["files"]) == ["so/so.parquet"] and data["complete"] is True and data["expected_files"] == 1
    (dest / "so" / "so.parquet").write_bytes(release.files["go/part-00000-a.parquet"])
    rep = write_manifest(dest, integrity)
    assert "sha1" in rep.skipped["so"] and not rep.tables
    assert _manifest(dest)["complete"] is False


def test_the_cli_writes_the_manifest_offline(release, tmp_path, capsys):
    from vbt.cli import main

    dest = _local_release(tmp_path, release)
    assert main(["data", "ot", "manifest", "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "3 table(s), 4 file(s)" in out and (dest / ".download-manifest.json").is_file()


def _ot_ctx(tmp: Path, dest: Path) -> ServiceContext:
    desc = {"schema": "vbt.datasource/1", "source": "ot", "title": "ot", "root": str(dest),
            "release": {"expect": "25.09", "from": "manifest.release"},
            "manifests": [{"path": ".download-manifest.json", "required": False, "require": {"complete": True}}],
            "defaults": {"format": "parquet", "layout": "sharded_dir"}, "id_types": {},
            "tables": {"go": {"kind": "fact", "path": "go", "grain": "row", "key": {"columns": ["id"]},
                              "columns": {"id": {"role": "identifier"}}}}}
    (tmp / "sources").mkdir(parents=True, exist_ok=True)
    (tmp / "overlays").mkdir(parents=True, exist_ok=True)
    (tmp / "sources" / "ot.yaml").write_text(yaml.safe_dump(desc, sort_keys=False))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp / "sources"), "overlays_dir": str(tmp / "overlays"),
                                       "cache_dir": str(tmp / "cache")}, project_root=tmp)
    return ServiceContext(settings)


def test_readiness_finds_the_manifest_and_its_byte_counts(release, tmp_path):
    from vbt.data.opentargets import load_integrity, write_manifest

    dest = _local_release(tmp_path, release)
    before = check_table(_ot_ctx(tmp_path / "c0", dest), "ot.go", "standard")
    assert [c.name for c in before.checks if not c.ok] == ["R2:manifest_absent"]
    integrity, _ = load_integrity("25.09", dest=dest)
    write_manifest(dest, integrity)
    after = check_table(_ot_ctx(tmp_path / "c1", dest), "ot.go", "standard")
    assert not [c for c in after.checks if not c.ok], [c.detail for c in after.checks if not c.ok]
    assert any(c.name == "R2:manifest" and "2 manifest entries match" in c.detail for c in after.checks)
    with (dest / "go" / "part-00000-a.parquet").open("ab") as fh:
        fh.write(b"x")
    changed = check_table(_ot_ctx(tmp_path / "c2", dest), "ot.go", "standard")
    assert any(c.name == "R2:manifest_bytes" and not c.ok for c in changed.checks)


def test_the_upstream_doctor_and_downloader_accept_the_manifest(release, tmp_path, monkeypatch):
    """The unmodified upstream checks (tools/doctor.py reference_files, tools/download_open_targets.py
    load_manifest and metadata_matches) read the manifest; the doctor's dataset list is narrowed to the tables
    here (the real directory lacks the tables the doctor then names)."""
    import importlib
    import sys

    from dl_upstream import upstream_root
    from vbt.data.opentargets import load_integrity, write_manifest

    up = upstream_root()
    if not (up / "tools" / "doctor.py").is_file() or not (up / "src" / "config" / "datasets.py").is_file():
        pytest.skip(f"the upstream checkout is not at {up}")
    dest = _local_release(tmp_path, release)
    integrity, _ = load_integrity("25.09", dest=dest)
    write_manifest(dest, integrity)                           # the real 25.09 base: what upstream expects
    monkeypatch.syspath_prepend(str(up))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    def upstream_modules() -> list[str]:
        return [m for m in sys.modules if m in ("tools", "src") or m.startswith(("tools.", "src."))]

    for name in upstream_modules():
        monkeypatch.delitem(sys.modules, name)
    try:
        doctor = importlib.import_module("tools.doctor")
        downloader = importlib.import_module("tools.download_open_targets")
        monkeypatch.setattr(doctor, "OPEN_TARGETS_DATASETS", ("so", "go", "evidence"))
        assert sorted(doctor.reference_files(str(dest))) == ["evidence", "go", "so"]
        entries = downloader.load_manifest(dest / ".download-manifest.json")
        assert set(entries) == set(release.files)
        assert all(downloader.metadata_matches(entries[rel], downloader.BASE + rel) for rel in release.files)
        (dest / "go" / "part-00000-a.parquet").write_bytes(b"PAR1" + b"0" * 30 + b"PAR1")
        with pytest.raises(ValueError, match="File size differs"):
            doctor.reference_files(str(dest))
    finally:
        for name in upstream_modules():                # upstream packages never stay imported (test_preflight)
            del sys.modules[name]


SNAP = Path(__file__).resolve().parent / "real" / "ot_25_09"


def test_the_recorded_integrity_list_parses_as_the_release_lists_its_files():
    """An excerpt of the 25.09 ``release_data_integrity`` (recorded 2026-10-08): only ``./output/`` lines are files
    of the tables, ``_SUCCESS`` markers are not Parquet files, and hive partitions stay in the path."""
    import json

    from vbt.data.opentargets import inventory, parse_integrity

    rec = json.loads((SNAP / "_release_data_integrity.excerpt.json").read_text())
    assert rec["output_parquet_files"] == sum(rec["parquet_files_by_table"].values()) == 3508
    release = json.loads((SNAP / "_release.json").read_text())
    assert rec["sha1"] == release["release_data_integrity"]["sha1"] and rec["lines"] == release[
        "release_data_integrity"]["lines"]
    assert {t: n for t, n in rec["parquet_files_by_table"].items()} == \
        {t: v["shards"] for t, v in release["tables"].items()}
    integrity = parse_integrity("\n".join(rec["excerpt"]))
    assert integrity["so/so.parquet"] == "d9e26aa8547366b410d292cbddf75df511db7d9d"
    assert "go/_SUCCESS" in integrity and not any(k.startswith(("disk_images", "input", "etc")) for k in integrity)
    inv = inventory(integrity)
    assert {t: len(f) for t, f in inv.items()} == {"drug_warning": 1, "evidence": 2, "go": 8, "reactome": 1, "so": 1}
    assert all(f.startswith("evidence/sourceId=chembl/") for f in inv["evidence"])


# ---------------------------------------------------------------------------- opt-in: real data and the network


def _real_ot() -> Path | None:
    from dl_upstream import real_ot_dir

    d = real_ot_dir()
    return d if d is not None and (d / "target").is_dir() else None


REAL = _real_ot()
needs_real = pytest.mark.skipif(REAL is None, reason="VBT_DL_REAL_DATA=<dir> with the Open Targets 25.09 tables needed")
needs_network = pytest.mark.skipif(not __import__("netgate").network_enabled(),
                                   reason="VBT_DL_NETWORK=1 needed (EBI FTP over HTTPS)")


def _real_ctx(tmp: Path) -> ServiceContext:
    repo = Path(__file__).resolve().parents[2]
    settings = DataSettings.from_dict({"descriptors_dir": str(repo / "configs" / "data" / "sources"),
                                       "overlays_dir": str(repo / "configs" / "data" / "overlays"),
                                       "cache_dir": str(tmp / "cache")}, project_root=repo)
    return ServiceContext(settings)                     # the descriptor root is ${OPEN_TARGETS_DATA_PATH}


@needs_real
@pytest.mark.parametrize("table,items", [
    ("target_go", 821_377), ("target_chemical_probes", 5_090), ("target_safety_liabilities", 4_484),
    ("target_essentiality_screens", 20_955_265), ("l2g_features", 44_578_431)])
def test_real_item_keys_hold_over_every_item_in_arrow(tmp_path, monkeypatch, table, items):
    """Deep checks of the 25.09 item tables whose keys have nullable parts (go ecoId, probes drugId, liabilities
    eventId/url, the "other" tissue group) or tens of millions of items; no item is converted to Python."""
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(REAL))
    seen = _no_item_scan(monkeypatch)
    model = check_table(_real_ctx(tmp_path), f"open_targets.{table}", "deep")
    assert seen == []
    (r5,) = checks(model, "R5")[-1:]
    assert r5.ok and f"({items} keys)" in r5.detail, r5.detail
    assert model.key_check.duplicates == 0 and model.status == "ready", [c.detail for c in model.checks if not c.ok]


@needs_real
def test_real_manifest_matches_the_release_integrity_list():
    """``vbt data ot manifest`` was run on the downloaded 25.09 tables: every listed file is present with its size,
    and its sha1 is the release's."""
    import json

    from vbt.data.opentargets import MANIFEST, inventory, load_integrity

    path = REAL / MANIFEST
    if not path.is_file():
        pytest.skip(f"no {MANIFEST} under {REAL} (run `vbt data ot manifest --dest {REAL}`)")
    data = json.loads(path.read_text())
    integrity, _ = load_integrity("25.09", dest=REAL)
    inv = inventory(integrity)
    assert data["release"] == "25.09" and data["complete"] is True and data["expected_files"] == len(data["files"])
    for rel, entry in data["files"].items():
        assert entry["sha1"] == integrity[rel] and (REAL / rel).stat().st_size == entry["bytes"], rel
    for table in data["tables"]:
        assert sorted(r for r in data["files"] if r.startswith(table + "/")) == inv[table], table


@needs_network
def test_live_release_integrity_and_a_small_table(tmp_path):
    """The 25.09 integrity list from the EBI FTP matches its .sha1 and the recorded inventory; ``so`` (one file) is
    fetched, verified and recorded in the manifest."""
    import json

    from vbt.data.opentargets import OpenTargetsRelease, inventory

    rec = json.loads((SNAP / "_release_data_integrity.excerpt.json").read_text())
    with OpenTargetsRelease("25.09") as ot:
        integrity, about = ot.integrity()
        assert about["sha1"] == rec["sha1"]
        assert {t: len(f) for t, f in inventory(integrity).items()} == rec["parquet_files_by_table"]
        rep = ot.fetch(["so"], tmp_path / "ot")
    assert not rep.failed and rep.manifest is not None and rep.manifest.tables == {"so": 1}
    assert _manifest(tmp_path / "ot")["files"]["so/so.parquet"]["sha1"] == integrity["so/so.parquet"]
