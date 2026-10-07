"""Role-driven result transforms T1-T14 (§11.7) and result field maps (§8.1 ``fields``), as pure functions."""

from __future__ import annotations

import math
from datetime import date

from test_dl_gateway_flow import REGISTRY
from vbt.datalayer.descriptor.columns import validate_column
from vbt.datalayer.descriptor.models import LeakageSpec
from vbt.datalayer.descriptor.overlay import FieldMap, RecomputeSpec, TrimSpec
from vbt.datalayer.gateway.fields import FieldMapper, cosine, extract_rows, jp_first, place_rows, recount
from vbt.datalayer.gateway.transforms import (
    Counters,
    honour_arguments,
    rank_keys,
    t1_leakage,
    t2_unknowns,
    t3_existence,
    t6_negation,
    t7_duplicates,
    t8_pooled_over,
    t9_levels,
    t10_order_cut,
    t11_counts,
    t12_trim,
    t13_flag_partition,
    t14_validity,
)
from vbt.datalayer.predicate import Any as AnyItem
from vbt.datalayer.predicate import Cmp, Eq


def cols(**specs):
    return {k: validate_column(v) for k, v in specs.items()}


# --------------------------------------------------------------------------- T1-T3


def test_t1_leakage_withholds_and_redacts_with_partial_dates():
    spec = LeakageSpec(available_at="posted", changed_at="updated", rows="redact", redact=["status"])
    rows = [{"id": 1, "posted": "2019-05", "updated": "2019-06-01", "status": "done"},
            {"id": 2, "posted": "2020-01", "updated": None, "status": "x"},          # Jan 2020 counts as 2020-01-31
            {"id": 3, "posted": "2019", "updated": "2021-02-03", "status": "stopped"},
            {"id": 4, "posted": None, "status": "?"}]
    c = Counters()
    kept = t1_leakage(rows, spec, date(2020, 1, 15), c)
    assert [r["id"] for r in kept] == [1, 3, 4]
    assert c.withheld == {"leakage": 1}
    assert kept[1]["status"] is None                         # changed after the ceiling: redacted


def test_t2_in_band_unknowns_placeholders_and_nested():
    spec = cols(phase={"role": "measure", "missing_values": [-1]},
                level={"role": "measure", "unknown_when": [{"column": "unit", "eq": ""}]},
                drugType={"role": "category", "placeholders": ["Unknown"]},
                name={"role": "label", "placeholder_when": "equals_key"},
                id={"role": "identifier"},
                tissues={"role": "nested", "item_key": ["t"],
                         "fields": {"t": {"role": "label"}, "v": {"role": "measure", "missing_values": [-1]}}})
    rows = [{"id": "X1", "name": "X1", "phase": -1, "level": 5, "unit": "", "drugType": "Unknown",
             "tissues": [{"t": "liver", "v": -1}, {"t": "lung", "v": 3}]},
            {"id": "X2", "name": "Two", "phase": float("nan"), "level": 5, "unit": "TPM", "drugType": "SM"}]
    c = Counters()
    out = t2_unknowns(rows, spec, c, key_column="id")
    a, b = out
    assert a["phase"] is None and a["level"] is None and a["drugType"] is None and a["name"] is None
    assert a["tissues"][0]["v"] is None and a["tissues"][1]["v"] == 3
    assert b["phase"] is None and b["level"] == 5 and b["name"] == "Two"
    assert c.nulled == {"phase": 2, "level": 1, "drugType": 1, "name": 1, "tissues[].v": 1}


def test_t3_phantom_rows():
    rows = [{"sampleId": "S1", "patientId": "P1"}, {"sampleId": "S9", "patientId": None}, {"sampleId": "S2"}]
    c = Counters()
    kept, phantom = t3_existence(rows, "$.patientId != null", [], c)
    assert [r["sampleId"] for r in kept] == ["S1"] and len(phantom) == 2 and c.withheld == {"phantom": 2}
    kept, phantom = t3_existence(rows, None, ["patientId"], Counters())
    assert [r["sampleId"] for r in kept] == ["S1", "S2"]


# --------------------------------------------------------------------------- T4-T6


class B:
    def __init__(self, item_filter=False, drop_empty_parents=False):
        self.item_filter, self.drop_empty_parents = item_filter, drop_empty_parents


def test_t4_t5_honour_arguments_unknown_beats_false_and_not_applicable():
    spec = cols(year={"role": "time"}, score={"role": "measure", "applies_when": {"kind": ["drug"]}})
    rows = [{"year": 2020, "score": 1, "kind": "drug"}, {"year": 2001, "score": 1, "kind": "drug"},
            {"year": None, "score": 0, "kind": "drug"}, {"year": 2021, "score": None, "kind": "gene"},
            {"year": 2022, "score": None, "kind": "drug"}]
    preds = {"min_year": (Cmp("year", ">=", 2010), B()), "min_score": (Cmp("score", ">=", 1), B())}
    c = Counters()
    kept = honour_arguments(rows, preds, lambda col: spec.get(col), c)
    assert kept == [rows[0]]
    assert c.excluded == {"min_year": 1}
    assert c.excluded_unknown == {"year": 1, "score": 1}
    assert c.excluded_not_applicable == {"score": 1}


def test_t4_item_filter_with_drop_empty_parents():
    rows = [{"d": "D1", "pheno": [{"id": "P1"}, {"id": "P2"}]}, {"d": "D2", "pheno": [{"id": "P2"}]},
            {"d": "D3", "pheno": [{"id": "P1"}]}]
    pred = AnyItem("pheno[]", Eq("id", "P1"))
    c = Counters()
    kept = honour_arguments(rows, {"pid": (pred, B(True, True))}, lambda col: None, c)
    assert [r["d"] for r in kept] == ["D1", "D3"] and kept[0]["pheno"] == [{"id": "P1"}]
    assert c.dropped_parents == 1 and c.items_removed == {"pid": 2}


def test_t6_negation_rows_and_items():
    rows = [{"id": 1, "negated": True}, {"id": 2, "negated": False},
            {"id": 3, "ev": [{"x": 1, "isNeg": True}]}, {"id": 4, "ev": [{"x": 1, "isNeg": True}, {"x": 2}]}]
    c = Counters()
    kept = t6_negation(rows, ["negated", "ev[].isNeg"], False, c, drop_empty_parents=True)
    assert [r["id"] for r in kept] == [2, 4] and kept[1]["ev"] == [{"x": 2}]
    assert c.excluded_negated == 3 and c.dropped_parents == 1
    assert len(t6_negation(rows, ["negated"], True, Counters())) == 4


# --------------------------------------------------------------------------- T7-T10


def test_t7_duplicates_and_t8_pooled_over():
    c = Counters()
    rows = t7_duplicates([{"c": 1, "dup": False}, {"c": 2, "dup": True}], ["dup"], False, c)
    assert rows == [{"c": 1, "dup": False}] and c.withheld == {"duplicates": 1}
    assert t8_pooled_over([{"a": 1}, {"a": 2}], ["a", "plate"]) == ["plate"]


def test_t9_levels_split_into_sections():
    rows = [{"sampleId": "S1", "patientId": "P1", "OS_MONTHS": 10, "AGE": 50},
            {"sampleId": "S2", "patientId": "P1", "OS_MONTHS": 10, "AGE": 50},
            {"sampleId": "S3", "patientId": "P2", "OS_MONTHS": 3, "AGE": 70}]
    c = Counters()
    out, sections, grains = t9_levels(rows, {"patient": ["OS_MONTHS", "AGE"]}, {"patient": ["patientId"]}, c)
    assert all("OS_MONTHS" not in r for r in out)
    assert sections["patients"] == [{"patientId": "P1", "OS_MONTHS": 10, "AGE": 50},
                                    {"patientId": "P2", "OS_MONTHS": 3, "AGE": 70}]
    assert grains == {"patient": 2}


def test_t10_order_nulls_last_ties_by_key_and_cut():
    rows = [{"id": "b", "s": 1}, {"id": "a", "s": 1}, {"id": "c", "s": None}, {"id": "d", "s": 5}]
    keys = rank_keys([{"column": "s", "direction": "desc"}], None, REGISTRY)
    out, info = t10_order_cut(rows, keys, ["id"], 3)
    assert [r["id"] for r in out] == ["d", "a", "b"] and info["truncated"] and info["total"] == 4


def test_t10_per_group_limit_grain_and_boundary():
    rows = [{"src": "x", "g": "A", "s": 9}, {"src": "x", "g": "A", "s": 8}, {"src": "x", "g": "B", "s": 1},
            {"src": "y", "g": "C", "s": 700}, {"src": "y", "g": "D", "s": 100}]
    keys = rank_keys([{"column": "s", "direction": "desc"}], None, REGISTRY)
    out, _ = t10_order_cut(rows, keys, ["src", "g"], 1, within=["src"])
    assert [(r["src"], r["s"]) for r in out] == [("y", 700), ("x", 9)]
    out, info = t10_order_cut(rows[:3], keys, ["src", "g", "s"], 2, limit_grain=["g"])
    assert [r["g"] for r in out] == ["A", "B"] and info["grain_total"] == 2
    patients = [{"sid": "1", "pid": "P1", "s": 3}, {"sid": "2", "pid": "P1", "s": 2}, {"sid": "3", "pid": "P2", "s": 1}]
    out, _ = t10_order_cut(patients, keys, ["sid"], 1, boundary=["pid"])
    assert [r["sid"] for r in out] == ["1", "2"]                 # a patient's samples stay together


# --------------------------------------------------------------------------- T11-T14


def test_t11_counts_summaries_removed_fields_and_arg_echo():
    obj = {"count": 99, "limit": 61, "num_genes": 7, "interpretation": "strong", "summary": {"median": 3},
           "diseases": [{"evidence_count": 5, "evidence": [1, 2]}, {"evidence_count": 1, "evidence": []}],
           "rows": [{"g": "A", "interpretation": "x"}, {"g": "A"}, {"g": "B"}]}
    rows = obj["rows"]
    c = Counters()
    t11_counts(obj, rows, count_fields=["$.count"],
               summary_fields={"$.num_genes": RecomputeSpec(agg="count_distinct", of="g"), "$.summary": "drop"},
               full_rows=rows, drop_fields={"$.interpretation": "not supported by the statistic"},
               vetoed={"interpretation": "vetoed by cosine"}, arg_echo={"limit": "$.limit"},
               args_raw={"limit": 20}, counters=c)
    assert obj["count"] == 3 and obj["num_genes"] == 2 and obj["limit"] == 20
    assert "summary" not in obj and "interpretation" not in obj and "interpretation" not in rows[0]
    assert set(c.removed_fields) == {"$.summary", "$.interpretation", "interpretation"}
    c = Counters()
    t11_counts(obj, rows, summary_fields={"$.num_genes": RecomputeSpec(agg="count", of="g")}, full_rows=None,
               counters=c)
    assert "num_genes" not in obj and "$.num_genes" in c.removed_fields
    changed = recount(obj, {"$.diseases[*].evidence_count": "len($.diseases[*].evidence)"}, 0)
    assert [d["evidence_count"] for d in obj["diseases"]] == [2, 0] and len(changed) == 2


def test_t12_trim_in_declared_order_with_counts():
    rows = [{"id": 1, "children": [{"k": c} for c in "dcba"]}, {"id": 2, "children": [{"k": "x"}]}]
    c = Counters()
    t12_trim(rows, {"children": TrimSpec(max=2, order="key")}, c, item_keys={"children": ["k"]})
    assert rows[0]["children"] == [{"k": "a"}, {"k": "b"}]
    assert c.trimmed == {"children": {"returned": 3, "total": 5}}
    c = Counters()
    rows = [{"t": [{"v": 1}, {"v": 5}, {"v": 3}]}]
    from vbt.datalayer.descriptor.columns import RankSpec
    t12_trim(rows, {"t": 2}, c, orders={"t": [RankSpec(column="v", direction="desc")]})
    assert rows[0]["t"] == [{"v": 5}, {"v": 3}]


def test_t13_flag_partition_and_t14_validity():
    rows = [{"tract": [{"m": "SM", "value": True}, {"m": "AB", "value": False}, {"m": "PR", "value": None}]}]
    assert t13_flag_partition(rows, {"tract": "value"}) == 1
    assert rows[0]["tract"] == [{"m": "SM", "value": True}]
    assert rows[0]["tract_not_met"] == [{"m": "AB", "value": False}]
    assert rows[0]["tract_unknown"] == [{"m": "PR", "value": None}]
    spec = cols(sparsity={"role": "measure", "undefined_when": [{"column": "n", "eq": 0}]})
    rows = [{"sparsity": 0.5, "n": 0, "sim": float("inf")}, {"sparsity": 0.2, "n": 4, "sim": 0.3}]
    c = Counters()
    t14_validity(rows, spec, c, computed=["sim"])
    assert rows[0]["sparsity"] is None and rows[0]["sim"] is None and rows[1]["sparsity"] == 0.2
    assert c.nulled == {"sparsity": 1, "sim": 1}


# --------------------------------------------------------------------------- field maps


def test_field_mapper_renames_computed_placeholders_and_keys():
    fields = {"entity_id": FieldMap(column="word"),
              "startDate": FieldMap(column="protocolSection.statusModule.startDateStruct.date"),
              "parent": FieldMap(column="parentId", placeholders=["Unknown"], on_placeholder="dangling_ref"),
              "similarity": FieldMap(computed={"statistic": "cosine", "from": "vector", "anchor": "entity_id"})}
    m = FieldMapper(fields, parent_key={"id": "$.target_id"}, key_from_args={"studyId": "study_id"})
    payload = {"target_id": "T1", "rows": [{"entity_id": "w1", "startDate": "2020", "parent": "Unknown",
                                            "vector": [1.0, 0.0]},
                                           {"entity_id": "w2", "startDate": None, "parent": "P", "vector": [0, 0]}]}
    rows = m.logical_rows(extract_rows(payload, ["$.rows"])[0][1], payload=payload, args={"study_id": "S"},
                          anchor_vector=[1.0, 0.0])
    a, b = rows
    assert a["word"] == "w1" and a["protocolSection"]["statusModule"]["startDateStruct"]["date"] == "2020"
    assert a["id"] == "T1" and a["studyId"] == "S" and a["parentId"] is None and a["similarity"] == 1.0
    assert b["similarity"] is None                      # zero-norm vector: undefined, never a number
    assert m.counters["dangling"] == {"parent": 1}
    a["word"] = "w1b"
    out = m.to_output(a)
    assert out["entity_id"] == "w1b" and "word" not in out and "protocolSection" not in out and "id" not in out
    assert math.isclose(cosine([1, 1], [1, 1]), 1.0) and math.isclose(cosine([1, 0], [0, 1]), 0.0)


def test_extract_and_place_rows():
    obj = {"data": {"rows": [{"a": 1}]}, "rec": {"x": 1}}
    assert extract_rows(obj, ["$.data.rows", "$.rec", "$.none"]) == [("$.data.rows", [{"a": 1}]),
                                                                     ("$.rec", [{"x": 1}]), ("$.none", [])]
    place_rows(obj, "$.data.rows", [{"a": 2}])
    assert jp_first(obj, "$.data.rows[0].a") == 2
