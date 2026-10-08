"""Wave-1 contracts: errors and payloads (§12.1), results and the header (§12.2), provenance (§15.1),
the predicate oracle (§9.2), ipc models (§11.8), settings (§17), the bridge API (§11.2) and the
no-pyarrow rule for harness-side modules (I12)."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from vbt.datalayer import errors as E
from vbt.datalayer import ipc
from vbt.datalayer.api import CallPlan, CrashDecision, GatewayProtocol, LaunchSpec, ListingDecision, RawResult
from vbt.datalayer.predicate import (
    All,
    And,
    Any,
    CensoredCmp,
    Cmp,
    CmpAbs,
    Contains,
    Eq,
    In,
    IsNull,
    KindMatch,
    NonEmpty,
    Not,
    Or,
    Param,
    PredicateError,
    Range,
    RankKey,
    TextMatch,
    columns,
    evaluate,
    facet_predicate,
    from_json,
    merge_container_predicates,
    to_json,
)
from vbt.datalayer.record import DataProvenance, TableInfo, cap_row_keys, prov_id
from vbt.datalayer.result import HEADER_KEY, DataResult, Header, header_line, inject_header
from vbt.datalayer.settings import DATA_DEFAULTS, DataSettings
from vbt.tools.base import ToolFailure

SRC = Path(__file__).resolve().parents[2] / "src"


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


# ---------------------------------------------------------------------------
# errors (§12.1)
# ---------------------------------------------------------------------------

def test_error_kinds_and_model_side():
    assert {k.value for k in E.ErrorKind} == {
        "not_found", "ambiguous", "invalid_argument", "unsupported_combination", "incomplete_key",
        "unsupported_filter", "insufficient_resolution", "not_ready", "too_large", "quarantined", "tool_defect",
        "oom", "server_crashed", "source_error", "service_unavailable"}
    assert {k.value for k in E.MODEL_SIDE_KINDS} == {
        "not_found", "ambiguous", "invalid_argument", "unsupported_combination", "incomplete_key",
        "unsupported_filter", "insufficient_resolution"}
    assert E.MODEL_SIDE_KINDS.isdisjoint(E.DATA_SIDE_KINDS)
    assert set(E.INSTRUCTIONS) == set(E.ErrorKind) == set(E.DEFAULT_RETRYABLE)


def test_gateway_error_is_tool_failure_and_str_is_envelope():
    payload = E.not_found_payload("target_id", "PCSK99", "ensembl_gene", ["ensembl_gene", "hgnc_symbol"],
                                  ["hgnc_symbol exact: none"], [{"id": "ENSG00000169174", "label": "PCSK9",
                                                                 "why": "edit distance 1"}],
                                  "open_targets@25.09", "target")
    err = E.GatewayError("not_found", "no such target", tool="mcp__target__get_target_info", payload=payload)
    assert isinstance(err, ToolFailure)
    env = json.loads(str(err))
    assert env == err.envelope()
    assert env["status"] == "tool_error" and env["contract"] == "vbt.data/1" and env["kind"] == "not_found"
    assert env["citable"] is False and env["retryable"] == "after_fixing_input"
    assert env["argument"] == "target_id" and env["value"] == "PCSK99"   # lifted from the payload
    assert env["tool"] == "mcp__target__get_target_info" and env["message"] == "no such target"
    assert env["instruction"].startswith("No record with this identifier exists in this source.")
    assert env["suggestions"][0]["id"] == "ENSG00000169174"
    assert err.model_side and not E.GatewayError("not_ready", "x").model_side
    assert str(E.ErrorKind.oom) == "oom"


def test_envelope_fixed_fields_cannot_be_overridden_and_subkind_is_top_level():
    err = E.GatewayError("incomplete_key", "pass concentration", payload=E.incomplete_key_payload(
        "concentration", "concentration", [5.0, 0.05, 0.5], unit="uM", subkind="incomparable_order",
        storage_type="float"))
    env = err.envelope()
    assert env["subkind"] == "incomparable_order" and env["kind"] == "incomplete_key"
    forged = E.GatewayError("not_found", "m", payload={"citable": True, "status": "ok", "kind": "x"}).envelope()
    assert forged["citable"] is False and forged["status"] == "tool_error" and forged["kind"] == "not_found"
    assert E.GatewayError("oom", "m", retryable="once").retryable == "once"


def test_not_found_payload_fields():
    base = E.not_found_payload("a", "v", "t", ["t"], [], [], "s@1", "tab")
    assert set(base) == {"argument", "value", "id_type", "accepts", "tried", "suggestions", "source", "table"}
    full = E.not_found_payload("a", "v", "t", ["t"], [], [], "s@1", "tab", subkind="obsolete", replacement="MONDO_1",
                               profiled_values={"concentration": [0.5]})
    assert set(full) - set(base) == {"subkind", "replacement", "profiled_values"}


def test_ambiguous_payload_carries_disambiguators():
    p = E.ambiguous_payload("disease_id", "rheumatoid arthritis", [
        {"id": "HP_0001370", "label": "Rheumatoid arthritis", "via": "label_exact:name", "isTherapeuticArea": False},
        {"id": "EFO_0000685", "label": "rheumatoid arthritis", "via": "label_exact:name"}])
    assert set(p) == {"argument", "value", "candidates"}
    assert p["candidates"][0]["isTherapeuticArea"] is False and set(p["candidates"][1]) == {"id", "label", "via"}


def test_invalid_argument_payload_full_or_nearest_vocabulary():
    small = E.invalid_argument_payload("aspect", "PX", ["C", "F", "P"], looks_like=["go_aspect"])
    assert set(small) == {"argument", "value", "valid_values", "near", "looks_like"}
    assert small["valid_values"] == ["C", "F", "P"]
    vocab = [f"tissue_{i:03d}" for i in range(200)] + ["liver", "Liver lobe", "kidney"]
    big = E.invalid_argument_payload("tissue", "livr", vocab, vocabulary_ref="expression.tissues", enum_max=64,
                                     items=[{"index": 0, "value": "livr", "reason": "unknown"}])
    assert len(big["valid_values"]) == 10 and big["valid_values"][0] == "liver"
    assert big["vocabulary_ref"] == "expression.tissues" and big["items"][0]["index"] == 0


def test_incomplete_key_values_sorted_json_typed_storage_rendered():
    p = E.incomplete_key_payload("concentration", "concentration", {_f32(5.0), _f32(0.05), _f32(0.5)}, unit="uM",
                                 storage_type="float")
    assert set(p) == {"dimension", "argument", "values", "unit", "retry_with"}
    assert p["values"] == [0.05, 0.5, 5.0]
    assert json.dumps(p["values"]) == "[0.05, 0.5, 5.0]"
    assert p["retry_with"][0] == {"concentration": 0.05}
    mixed = E.incomplete_key_payload("plate", None, ["plate2", None, 1, "plate1"])
    assert mixed["values"] == [1, "plate1", "plate2", None] and mixed["retry_with"][0] == {"plate": 1}


def test_other_payload_fields():
    assert set(E.unsupported_filter_payload("max_phase", "phase", "scale")) == {"argument", "column", "reason"}
    assert E.unsupported_filter_payload("a", "c", "scale", confirmed_range=[0, 1])["confirmed_range"] == [0, 1]
    ir = E.insufficient_resolution_payload("genes", 10, 3, ["x"], ["y"], ["z"], 0.95)
    assert set(ir) == {"argument", "requested", "resolved", "unresolved", "ambiguous", "outside_universe",
                       "min_resolved_fraction"}
    nr = E.not_ready_payload([{"name": "open_targets.target", "column": "homologues.speciesId", "check": "R4",
                               "detail": "str vs int", "hint": "update the descriptor"},
                              {"name": "open_targets.evidence", "partition": "sourceId=chembl", "check": "R1",
                               "detail": "missing", "hint": "re-download"}])
    assert set(nr) == {"tables"}
    assert set(nr["tables"][0]) == {"name", "column", "check", "detail", "hint"}
    assert set(nr["tables"][1]) == {"name", "partition", "check", "detail", "hint"}
    td = E.tool_defect_payload("W1", {"total": 7, "topk": [["a"]]}, 0, ["OT-DRUG-009"])
    assert td == {"check": "W1", "witness": {"total": 7, "topk": [["a"]]}, "returned": 0, "defect_ids": ["OT-DRUG-009"]}
    assert set(E.tool_defect_payload("W2", {"total": 1}, 3)["witness"]) == {"total"}


def test_nearest_and_edit_distance():
    assert E.edit_distance("kitten", "sitting") == 3
    assert E.nearest("PCSK", ["PCSK9", "APOB", "PCSK1"], 2) == ["PCSK1", "PCSK9"]


# ---------------------------------------------------------------------------
# result (§12.2)
# ---------------------------------------------------------------------------

def test_header_drops_none_and_v_first():
    h = Header(status="ok", source="open_targets@25.09", tables=["known_drug"], returned=2, total=2)
    d = h.to_dict()
    assert list(d)[:2] == ["v", "status"] and "coverage" not in d and "notes" not in d
    assert Header.from_dict(d).to_dict() == d


def test_header_extra_keys_survive():
    d = Header(status="partial", extra={"unavailable_partitions": ["sourceId=europepmc"]}, notes=["n"]).to_dict()
    assert list(d)[-2:] == ["unavailable_partitions", "notes"]
    assert Header.from_dict(d).to_dict() == d
    shown = inject_header({}, {"status": "ok", "custom": 1})
    assert shown[HEADER_KEY]["custom"] == 1


def test_header_trims_notes_to_budget():
    h = Header(status="partial", notes=[f"note {i} " + "x" * 80 for i in range(40)], prov="dp_1")
    d = h.to_dict(max_chars=1200)
    assert len(json.dumps(d, ensure_ascii=False)) <= 1200
    assert d["notes"][-1].endswith("more notes in provenance") and d["prov"] == "dp_1"


def test_rev2_header_fields_present():
    names = set(Header.__dataclass_fields__)
    assert {"excluded_not_applicable", "excluded_negated", "withheld", "family_rows", "resolution_summary",
            "not_found_items", "trimmed", "undefined", "evidence", "hash_seed", "scope", "grains"} <= names


def test_inject_header_dict_list_str():
    h = Header(status="ok", source="s@1", tables=["t"], returned=1, total=3, order="phase desc (verified)",
               prov="dp_7c1e0942")
    out = inject_header({"success": True, HEADER_KEY: {"stale": 1}, "drugs": [1]}, h)
    assert list(out) == [HEADER_KEY, "success", "drugs"] and out[HEADER_KEY]["status"] == "ok"
    wrapped = inject_header([1, 2], h)
    assert list(wrapped) == [HEADER_KEY, "rows"] and wrapped["rows"] == [1, 2]
    text = inject_header("plain text", h)
    assert text.splitlines()[0] == "[vbt-data dp_7c1e0942 · ok · s@1/t · 1 of 3 · ranked phase desc (verified)]"
    assert text.splitlines()[1] == "plain text"
    assert header_line({"status": "empty", "coverage": "unknown"}) == "[vbt-data · empty · coverage unknown]"


def test_data_result_header_first_and_full_text_unshrunk():
    full = {"success": True, "drugs": [{"id": i} for i in range(50)]}
    shown = {"success": True, "drugs": full["drugs"][:2], "truncated": True}
    rec = DataProvenance(tool="mcp__drug__x", server="drug")
    r = DataResult.build(shown, Header(status="partial", returned=2, total=50), rec, full_obj=full)
    assert r.is_data_result is True and DataResult.is_data_result is True
    assert r.status == "partial" and r.provenance is rec
    assert r.text.startswith('{"_vbt": {"v": 1, "status": "partial"')
    assert len(json.loads(r.full_text)["drugs"]) == 50 and len(json.loads(r.text)["drugs"]) == 2
    assert next(iter(json.loads(r.full_text))) == HEADER_KEY
    same = DataResult.build({"a": 1}, {"status": "ok"})
    assert same.full_text == same.text and str(same) == same.text
    txt = DataResult.build("rows as text", {"status": "empty_unverified"})
    assert txt.text.startswith("[vbt-data · empty_unverified]")


# ---------------------------------------------------------------------------
# record (§15.1)
# ---------------------------------------------------------------------------

def _record() -> DataProvenance:
    rec = DataProvenance(tool="mcp__drug__search_known_drugs", server="drug", mode="enforce", served_by="upstream",
                         tool_use_id="toolu_01", retrieved_at="2026-10-06T09:14:03Z")
    rec.source.name, rec.source.release, rec.source.release_verified = "open_targets", "25.09", True
    rec.tables.append(TableInfo(name="known_drug", fingerprint="fp1:x", key_check={"method": "full"}))
    rec.result.status, rec.result.returned, rec.result.total = "partial", 20, 61
    rec.result.coverage, rec.result.coverage_statement = "unknown", "ChEMBL-curated only"
    rec.upstream.hash_seed, rec.upstream.flags_stripped = 0, ["-E"]
    return rec


def test_prov_id_stable_and_content_addressed():
    a, b = _record().finalize(), _record().finalize()
    assert a.id == b.id and a.id.startswith("dp_") and len(a.id) == 15
    c = _record()
    c.tool_use_id = "toolu_other"
    assert c.finalize().id == a.id                     # the caller's id is not part of the content
    d = _record()
    d.result.total = 62
    assert d.finalize().id != a.id
    assert prov_id(a.to_dict()) == a.id


def test_record_fields_and_summary():
    rec = _record().finalize()
    d = rec.to_dict()
    assert list(d)[:3] == ["schema", "id", "tool_use_id"] and d["schema"] == "vbt.dataprov/1"
    for key in ("release_verified", "scope_versions"):
        assert key in d["source"]
    for key in ("partitions_read", "partition_fingerprints", "lineage", "key_check"):
        assert key in d["tables"][0]
    for key in ("expansions", "selector", "resolutions"):
        assert key in d["request"]
    for key in ("coverage", "coverage_statement", "output_rows_sha256", "computed", "statistics", "row_keys_complete"):
        assert key in d["result"]
    assert d["upstream"]["hash_seed"] == 0 and d["upstream"]["flags_stripped"] == ["-E"]
    assert "evidence_nature" in d and "retrieved_at" in d
    s = rec.summary()
    assert set(s) == {"prov", "status", "coverage", "coverage_statement", "source", "tables", "returned", "total",
                      "truncated", "row_keys_sha256", "served_by", "evidence_nature", "leakage_risk",
                      "order_verified", "evidence_caveat"}
    assert s["source"] == "open_targets@25.09" and s["tables"] == [{"name": "known_drug", "fingerprint": "fp1:x"}]
    assert json.loads(rec.to_json())["id"] == rec.id


def test_cap_row_keys():
    keys = [["CHEMBL1", _f32(0.05)], ["CHEMBL2", None]]
    kept, complete, sha = cap_row_keys(keys, 10, ["string", "float"])
    assert complete and kept == [["CHEMBL1", 0.05], ["CHEMBL2", None]]
    many = [[f"k{i}"] for i in range(500)]
    kept2, complete2, sha2 = cap_row_keys(many, 300)
    assert not complete2 and len(kept2) == 100
    assert sha2 == cap_row_keys(list(reversed(many)), 300)[2]
    rec = DataProvenance(tool="t", server="s")
    rec.set_row_keys(keys, 10, ["string", "float"])
    assert rec.result.row_keys_sha256 == sha and rec.result.row_keys_complete


# ---------------------------------------------------------------------------
# predicate (§9.2): the Kleene oracle
# ---------------------------------------------------------------------------

ROW = {
    "id": "G1",
    "tissues": [{"name": "liver", "rna": {"value": 3}}, {"name": "brain", "rna": {"value": 20}}],
    "empty": [], "nothing": None, "with_null": [None, {"name": "x"}],
    "ge": [{"dep": [{"tissueId": "T1", "screens": [{"depmapId": "A", "geneEffect": -1.0},
                                                   {"depmapId": "B", "geneEffect": None}]},
                    {"tissueId": "T2", "screens": []}]}],
    "nan": float("nan"), "lfc": -2.5, "tags": ["a", "b"], "phase": None,
}


def ev(p, row=ROW, **kw):
    return evaluate(p, row, **kw)


def test_comparisons_with_null_and_nan_are_unknown():
    assert ev(Cmp("nan", ">", 0)) is None and ev(Cmp("phase", ">=", 0)) is None and ev(Eq("phase", 1)) is None
    assert ev(Not(Cmp("nan", ">", 0))) is None
    assert ev(IsNull("nan")) is True and ev(IsNull("phase")) is True and ev(IsNull("lfc")) is False
    assert ev(Cmp("missing_column", "<", 3)) is None
    assert ev(Eq("lfc", float("nan"))) is None
    assert ev(Cmp("id", "<", 3)) is None               # incomparable types never silently pass or fail


def test_kleene_connectives():
    t, f, u = Eq("id", "G1"), Eq("id", "G2"), Eq("phase", 1)
    assert ev(And((t, u))) is None and ev(And((f, u))) is False and ev(And((t, t))) is True
    assert ev(Or((f, u))) is None and ev(Or((t, u))) is True and ev(Or((f, f))) is False
    assert ev(Not(f)) is True and ev(None) is True


def test_quantifiers_on_empty_null_and_null_items():
    p = Eq("name", "x")
    assert ev(Any("empty[]", p)) is False and ev(All("empty[]", p)) is True
    assert ev(Any("nothing[]", p)) is None and ev(All("nothing[]", p)) is None
    assert ev(Any("with_null[]", Eq("name", "y"))) is None     # no true item, one unknown item
    assert ev(Any("with_null[]", Eq("name", "x"))) is True
    assert ev(Any("with_null[]", Eq("name", "y"), skip_null_items=True)) is False
    assert ev(All("with_null[]", Eq("name", "x"))) is None
    assert ev(NonEmpty("empty")) is False and ev(NonEmpty("nothing")) is None and ev(NonEmpty("tags")) is True


def test_correlated_any_versus_uncorrelated_existentials():
    uncorrelated = And((Eq("tissues[].name", "liver"), Cmp("tissues[].rna.value", ">=", 10)))
    assert ev(uncorrelated) is True                    # C21: the wrong answer two existentials give
    merged = merge_container_predicates([Eq("tissues[].name", "liver"), Cmp("tissues[].rna.value", ">=", 10)])
    assert merged == [Any("tissues[]", And((Eq("name", "liver"), Cmp("rna.value", ">=", 10))))]
    assert ev(And(tuple(merged))) is False
    assert ev(Any("tissues[]", And((Eq("name", "brain"), Cmp("rna.value", ">=", 10))))) is True


def test_three_level_correlated_any_with_parent_and_table_paths():
    screens = "ge[].dep[].screens[]"
    assert ev(Any(screens, And((Eq("^.tissueId", "T1"), Cmp("geneEffect", "<", -0.5))))) is True
    assert ev(Any(screens, And((Eq("^.tissueId", "T2"), Cmp("geneEffect", "<", -0.5))))) is False
    assert ev(Any(screens, And((Eq("depmapId", "B"), Cmp("geneEffect", "<", -0.5))))) is None
    assert ev(Any(screens, And((Eq("/id", "G1"), Eq("depmapId", "A"))))) is True
    assert ev(All("ge[].dep[]", NonEmpty("screens"))) is False
    nested = Any("ge[]", Any("dep[]", Any("screens[]", Cmp("geneEffect", "<", 0))))
    assert ev(nested) is True
    assert merge_container_predicates([Eq("ge[].dep[].tissueId", "T1"), Cmp("ge[].dep[].screens[].geneEffect", "<", 0)]) \
        == [Any("ge[]", Any("dep[]", And((Eq("tissueId", "T1"), Any("screens[]", Cmp("geneEffect", "<", 0))))))]


def test_contains_in_list_elements_and_item_predicates():
    assert ev(Contains("tags", "a")) is True and ev(Contains("tags", "z")) is False
    assert ev(Contains("tissues[].name", "brain")) is True and ev(Contains("nothing", "a")) is None
    assert ev(Eq("tags", "b")) is True                 # a role on list<T> applies per element
    assert ev(In("id", ("G1", "G2"))) is True and ev(In("id", ("G3", None))) is None
    assert ev(Eq("tissues[name=brain].rna.value", 20)) is True
    assert ev(Eq("tissues[name=liver].rna.value", 20)) is False


def test_cmpabs_range_text_kind():
    assert ev(CmpAbs("lfc", "gt_abs", 2)) is True and ev(CmpAbs("lfc", ">", 3)) is False
    assert ev(CmpAbs("nan", ">", 0)) is None
    assert ev(Range("lfc", -3, -2)) is True and ev(Range("lfc", -2.5, 0, lo_inclusive=False)) is False
    row = {"name": "Non-Cancerous tissue", "cl": "BxPC-3"}
    assert evaluate(TextMatch("name", "cancer", "word"), row) is False
    assert evaluate(TextMatch("name", "cancerous", "word"), row) is True
    assert evaluate(TextMatch("cl", "PC-3", "exact"), row) is False
    assert evaluate(TextMatch("cl", "bxpc-3", "casefold"), row) is True
    assert evaluate(TextMatch("cl", "pc-3", "casefold_substring"), row) is True
    assert evaluate(KindMatch("cl", "depmap_cell_line"), row) is None
    assert evaluate(KindMatch("cl", "x"), row, kind_of=lambda v, k: v.startswith("Bx")) is True


def test_censored_comparison():
    p = CensoredCmp("os", "dead", "<=", 12)
    assert evaluate(p, {"os": 10, "dead": True}) is True
    assert evaluate(p, {"os": 10, "dead": False}) is None     # censored before t: unknown
    assert evaluate(p, {"os": 14, "dead": False}) is False
    assert evaluate(p, {"os": None, "dead": True}) is None
    q = CensoredCmp("os", "dead", ">", 12)
    assert evaluate(q, {"os": 14, "dead": True}) is True and evaluate(q, {"os": 10, "dead": False}) is None
    assert evaluate(q, {"os": 10, "dead": True}) is False
    with pytest.raises(PredicateError):
        CensoredCmp("os", "dead", "!=", 1)


def test_params_json_round_trip_and_columns():
    preds = [
        Eq("a", 1), In("a", (1, 2)), Cmp("a", "ge", 0.5), CmpAbs("lfc", "gt_abs", 1), Range("a", 0, 1, True, False),
        Contains("tags", "x"), Any("t[]", And((Eq("name", Param("tissue")), Cmp("v", "<", 3))), True),
        All("t[]", NonEmpty("x")), KindMatch("w", "ensembl_gene"), CensoredCmp("os", "dead", "<=", 12),
        TextMatch("c", "x", "casefold"), IsNull("a"), Not(Or((Eq("a", 1), Eq("b", 2)))),
    ]
    for p in preds:
        assert from_json(json.loads(json.dumps(to_json(p)))) == p
    p = Any("tissues[]", And((Eq("name", Param("t")), Cmp("rna.value", ">=", 10))))
    assert evaluate(p, ROW, {"t": "brain"}) is True and evaluate(p, ROW, {"t": "liver"}) is False
    with pytest.raises(PredicateError):
        evaluate(p, ROW)
    assert columns(p) == {"tissues[].name", "tissues[].rna.value"}
    assert columns(CensoredCmp("os", "dead", "<", 1)) == {"os", "dead"}
    assert from_json({"in": ["proteinIds[].source", ["uniprot_swissprot", "uniprot_trembl"]]}) == \
        In("proteinIds[].source", ("uniprot_swissprot", "uniprot_trembl"))
    for bad in ({}, {"eq": [1]}, {"zz": [1, 2]}, {"eq": ["a", 1], "ne": ["a", 2]}):
        with pytest.raises(PredicateError):
            from_json(bad)


def test_facet_predicates():
    assert facet_predicate({"eq": -1}, "phase") == Eq("phase", -1)
    assert facet_predicate({"column": "unit", "eq": ""}, "value") == Eq("unit", "")
    assert facet_predicate({"in": ["none_recorded"]}, "flag") == In("flag", ("none_recorded",))
    assert facet_predicate({"lt": -0.5}, "gene_effect") == Cmp("gene_effect", "<", -0.5)
    with pytest.raises(PredicateError):
        facet_predicate({"eq": 1, "ne": 2}, "x")


def test_rank_key():
    rk = RankKey("padj", "asc", within=["drug", "plate"])
    assert rk.within == ("drug", "plate") and RankKey.from_json(rk.to_json()) == rk
    with pytest.raises(PredicateError):
        RankKey("x", "up")


def test_nan_float_is_not_equal_and_bool_not_number():
    assert evaluate(Eq("v", 1), {"v": True}) is False and evaluate(Eq("v", True), {"v": True}) is True
    assert evaluate(Eq("v", 0.05), {"v": _f32(0.05)}) is False   # the reader compiles in the storage type
    assert math.isnan(ROW["nan"])


# ---------------------------------------------------------------------------
# ipc (§11.8)
# ---------------------------------------------------------------------------

def _roundtrip(model):
    data = json.loads(model.model_dump_json(by_alias=True))
    assert type(model).model_validate(data) == model
    return data


def test_ipc_round_trips():
    _roundtrip(ipc.StatsRequest(tables=["open_targets.target"]))
    stats = ipc.TableStatsModel(fingerprint="fp1:x", signature="sig", partition_fingerprints={"sourceId=chembl": "p"},
                                rows=None, fragments=3, bytes_on_disk=10,
                                columns={"go.list.element.id": ipc.ColumnStatsModel(
                                    uncompressed_bytes=5, num_values=9, max_rep_level=1, max_def_level=3,
                                    kind="nested", storage_type="large_string", null_count=0, min="GO:1", max="GO:9")},
                                row_bytes_p99={"row": 2000, "open_targets.target_go": 120})
    _roundtrip(ipc.StatsResponse(tables={"open_targets.target": stats}))
    check = ipc.CheckResponse(tables={"t.x": ipc.TableCheckModel(
        status="partial", partitions={"sourceId=europepmc": "partial"},
        checks=[ipc.CheckItemModel(name="R1", ok=False, detail="stray .part", hint="re-download",
                                   partition="sourceId=europepmc")],
        key_check=ipc.KeyCheckModel(method="full", ok=True, null_counts={"status": 6021}))}, hash_randomization=0)
    _roundtrip(check)
    w = ipc.WitnessRequest(table="tahoe_100m.de_permissive", grain="gene", predicate={"eq": ["drug", "X"]},
                           key=["drug", "gene_name"], order=[ipc.RankKeyModel(column="padj", direction="asc",
                                                                             within=["plate"])],
                           k=20, group_by=["plate"], distinct=["concentration"],
                           grains={"gene": ["gene_name"], "pair": {"unordered": ["a", "b"]}},
                           unknown_columns=["padj"], key_set_max=1000, budget_bytes=10**9)
    _roundtrip(w)
    wr = ipc.WitnessResponse(total=61, total_method="scan", topk={'["1"]': [["a", "b"]]}, key_set=[["a", "b"]],
                             distinct={"concentration": [0.05, 0.5]}, excluded_unknown={"padj": 2, "_rows": 2},
                             excluded_not_applicable={"clinicalPhase": 1}, unknown_total=2,
                             distinct_counts={"gene": 40}, group_totals={'["1"]': 61}, one_to_many={'["x"]': 2},
                             scanned_bytes=100, reason=None)
    assert _roundtrip(wr)["total_method"] == "scan"
    sr = ipc.ServeRequest(table="open_targets.pharmacogenomics", verb="find", predicate={"eq": ["x", 1]},
                          columns=["x"], order=[], limit=10, limit_grain="drug", explode=["a[]"], carry=["id"],
                          rename={"x": "y"}, split={"by_sign": "lfc"}, nest=None, sections={}, anchor=None,
                          search_text=None, id_type="chembl_molecule")
    _roundtrip(sr)
    _roundtrip(ipc.ServeResponse(rows={"+": [{"a": 1}], "-": []}, total=1, truncated=False, key_columns=["a"],
                                 grains={"drug": {"returned": 1, "total": 1}}, served_by="derived"))
    _roundtrip(ipc.BuildIndexRequest(source="open_targets", id_type="ensembl_gene"))
    _roundtrip(ipc.BuildIndexRequest(table="tahoe_100m.de_permissive", access_path=["gene_name"]))
    _roundtrip(ipc.ResolveRemoteResponse(resolutions=[{"raw": "rs1", "canonical": "1_1_A_G"}], existence="exists"))
    _roundtrip(ipc.VocabResponse(values=[_f32(0.05)], rendered=["0.05"], fingerprint="fp", storage_type="float"))


def test_ipc_validation_and_helpers():
    with pytest.raises(ValueError):
        ipc.BuildIndexRequest(source="s", table="t")
    with pytest.raises(ValueError):
        ipc.WitnessRequest(table="t", bogus=1)
    with pytest.raises(ValueError):
        ipc.ServeRequest(table="t", verb="drop")
    payload = ipc.request_payload(ipc.VocabRequest(table="t.x", column="c"))
    assert list(payload) == [ipc.REQUEST_ARG] == ["request"] and payload["request"]["column"] == "c"
    resp = ipc.parse_response(ipc.VERB_WITNESS, json.dumps({"total": 3, "total_method": "index"}))
    assert isinstance(resp, ipc.WitnessResponse) and resp.total == 3
    assert ipc.SETTINGS_ENV == "VBT_DATA_SETTINGS"
    assert set(ipc.PHASE1_VERBS) == {"_stats", "_check", "_witness", "_serve", "_build_index", "_resolve_remote",
                                     "_vocab"} == set(ipc.VERB_MODELS) - {ipc.VERB_CENSUS_COUNT,
                                                                          ipc.VERB_RELEASE}   # phase 4: typed too


# ---------------------------------------------------------------------------
# settings (§17) and api (§11.2)
# ---------------------------------------------------------------------------

def test_settings_defaults_merge_and_dirs(tmp_path):
    s = DataSettings.from_config({"data": {"gateway": {"mode": "observe", "enforce_servers": ["target"]},
                                           "cache_dir": str(tmp_path / "c"), "descriptors_dir": "x/y"},
                                  "vars": {"project_root": str(tmp_path)}})
    assert s.gateway.mode == "observe" and s.gateway.enforce_servers == ("target",)
    assert not s.gateway.enforces("target")            # observe never enforces
    assert s.cache_dir == tmp_path / "c" and s.descriptors_dir == tmp_path / "x" / "y"
    assert s.witness.max_inflate_rows == 5000 and s.resolution.max_hops == 2 and s.resolution.max_expand == 5000
    assert s.resolution.min_resolved_fraction == 0.95 and s.resolution.remote_ttl_s == 3600
    assert s.witness.topk is True and s.memory.object_overhead_bytes == {"string": 50, "nested_item": 120,
                                                                        "struct_item": 240}
    assert s.readiness.vocab_budget_bytes == 500_000_000 and s.readiness.remote_ttl_s == 3600
    assert s.leakage.ceiling is None and dict(s.sources.alias) == {}
    assert DataSettings.from_json(s.to_json()) == s
    env = DataSettings.from_env({"VBT_DATA_SETTINGS": s.to_json()})
    assert env == s
    default = DataSettings.from_config({})
    assert default.cache_dir.name == ".vbt-datalayer" and "${" not in str(default.cache_dir)
    assert default.gateway.enforces("anything")
    assert DATA_DEFAULTS["plugins"] == {"paths": [], "entry_points": True, "disabled": [], "override": {},
                                        "require_conformance": False}


def test_call_plan_hold_and_protocol():
    lock = asyncio.Lock()
    plan = CallPlan(server="s", tool="t", args_raw={}, args_sent={}, route="upstream", contract=None, resolutions=[],
                    witness=None, cold_lock=lock, bound_table="s.t", gateway_args={"plate": "1"})

    async def go():
        async with plan.hold():
            assert lock.locked()
        assert not lock.locked()
        async with CallPlan("s", "t", {}, {}, "derived", None, [], None).hold():
            pass

    asyncio.run(go())
    assert plan.scope == {} and plan.existence == {}

    class Fake:
        mode = "observe"

        def bind_bridge(self, bridge): ...
        def extra_servers(self): return []
        def launch_spec(self, cfg): return LaunchSpec("python", [], {}, "/tmp/s")
        def rewrite_listing(self, server, tool, description, input_schema):
            return ListingDecision(True, description, input_schema)
        async def prepare(self, server, tool, args, ctx): ...
        async def finish(self, plan, raw): ...
        async def on_crash(self, server, plan, reason, log_tail): return CrashDecision(False, None, oom=True)
        def pinned(self): return {}
        def readiness_snapshot(self): return {}

    assert isinstance(Fake(), GatewayProtocol)
    assert RawResult("t", None, None, "empty_lookup").error_text is None


def test_lazy_package_attributes():
    import vbt.datalayer as dl
    assert dl.GatewayError is E.GatewayError and dl.ErrorKind is E.ErrorKind
    assert dl.DataResult is DataResult and dl.DataSettings is DataSettings
    assert callable(dl.load_catalog)
    with pytest.raises(AttributeError):
        dl.nope  # noqa: B018
    try:
        dl.build_gateway
    except ImportError as exc:
        assert "gateway" in str(exc)


# ---------------------------------------------------------------------------
# I12: harness-side modules never import pyarrow or pandas
# ---------------------------------------------------------------------------

HARNESS_MODULES = [
    "vbt.datalayer", "vbt.datalayer.errors", "vbt.datalayer.result", "vbt.datalayer.record", "vbt.datalayer.settings",
    "vbt.datalayer.api", "vbt.datalayer.ipc", "vbt.datalayer.roles", "vbt.datalayer.predicate", "vbt.datalayer.rowkey",
    "vbt.datalayer.catalog", "vbt.datalayer.descriptor", "vbt.datalayer.descriptor.models",
    "vbt.datalayer.descriptor.columns", "vbt.datalayer.descriptor.overlay", "vbt.datalayer.descriptor.scoping",
    "vbt.datalayer.descriptor.load", "vbt.datalayer.descriptor.lint", "vbt.datalayer.plugins",
    "vbt.datalayer.plugins.base", "vbt.datalayer.plugins.registry", "vbt.datalayer.plugins.conformance",
]


def test_harness_modules_import_without_pyarrow_or_pandas():
    code = (
        "import sys\n"
        "sys.modules['pyarrow'] = None\n"
        "sys.modules['pandas'] = None\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import importlib\n"
        f"for m in {HARNESS_MODULES!r}:\n"
        "    importlib.import_module(m)\n"
        "import vbt.datalayer as dl\n"
        "dl.GatewayError, dl.DataResult, dl.DataSettings, dl.load_catalog\n"
        "from vbt.datalayer.plugins.registry import discover\n"
        "discover(entry_points=False)\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('pyarrow', 'pandas') and sys.modules[m] is not None]\n"
        "assert not loaded, loaded\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
