"""Argument contracts (§11.3 step 4, F7): every rule of ``gateway/contracts.py`` and the predicates
bound arguments compile to."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from test_dl_gateway_flow import REGISTRY, f32
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.gateway.contracts import VocabSnapshot, apply_arg_contracts, build_predicate
from vbt.datalayer.predicate import And, Any as AnyItem, CmpAbs, Eq, In, Or, evaluate

SRC: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "s", "title": "S", "release": {"from": "literal"},
    "defaults": {"format": "parquet", "layout": "single_file"},
    "tables": {
        "facts": {
            "kind": "fact", "grain": "one measurement", "key": {"columns": ["gene", "contrast", "concentration"]},
            "columns": {
                "gene": {"role": "label"},
                "contrast": {"role": "category", "scope": {"kind": "stratum"}},
                "concentration": {"role": "scope"},
                "cell_line": {"role": "category", "match": "casefold", "aliases": {"HELA": "HeLa"}},
                "disease": {"role": "text"},
                "padj": {"role": "measure", "statistic": "numeric", "comparable_within": ["contrast"]},
                "lfc": {"role": "measure", "statistic": "numeric"},
                "score": {"role": "measure", "statistic": "numeric", "scale": [0, 1]},
                "safety": {"role": "flag"},
                "aspect": {"role": "category", "vocab": ["biological_process", "molecular_function"]},
                "dup": {"role": "qualifier", "effect": "duplicate", "default_filter": True},
                "donor": {"role": "category", "unique_within": ["dataset"]},
                "dataset": {"role": "category"},
                "go": {"role": "nested", "item_key": ["id"],
                       "fields": {"id": {"role": "label"}, "evidence": {"role": "category"}}},
                "a": {"role": "label"}, "b": {"role": "label"}}},
        "coloc": {"kind": "fact", "grain": "one coloc", "key": {"columns": ["l"]}, "columns": {"l": {"role": "label"}}},
        "ecaviar": {"kind": "fact", "grain": "one ecaviar", "key": {"columns": ["l"]},
                    "columns": {"l": {"role": "label"}}},
    },
}
T = "s.facts"


def tool(args: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"schema": "vbt.overlay/1", "server": "srv", "tools": {"t": {"reads": {T: {}}, "args": args, **extra}}}


def contract(args: dict[str, Any], **extra: Any):
    cat = Catalog({"s": SourceDescriptor.model_validate(copy.deepcopy(SRC))},
                  {"srv": Overlay.model_validate(tool(args, **extra))}, registry=REGISTRY)
    return cat.contract("srv", "t")


def err(fn, kind: ErrorKind) -> dict[str, Any]:
    with pytest.raises(GatewayError) as e:
        fn()
    assert e.value.kind == kind, e.value.envelope()
    return e.value.envelope()


def test_unknown_argument_when_schema_forbids_extra():
    c = contract({"gene": {"binds": f"{T}.gene"}})
    schema = {"type": "object", "properties": {"gene": {"type": "string"}}, "additionalProperties": False}
    env = err(lambda: apply_arg_contracts(c, {"gene": "X", "genes": "Y"}, schema=schema), ErrorKind.invalid_argument)
    assert env["argument"] == "genes" and "gene" in env["valid_values"]
    assert apply_arg_contracts(c, {"gene": "X"}, schema=schema).args_sent == {"gene": "X"}


def test_require_any_and_exclusive():
    c = contract({"a": {"binds": f"{T}.a"}, "b": {"binds": f"{T}.b"}}, require_any=[["a", "b"]], exclusive=[["a", "b"]])
    err(lambda: apply_arg_contracts(c, {}), ErrorKind.invalid_argument)
    env = err(lambda: apply_arg_contracts(c, {"a": "1", "b": "2"}), ErrorKind.unsupported_combination)
    assert env["arguments"] == ["a", "b"]
    assert apply_arg_contracts(c, {"a": "1"}).args_sent == {"a": "1"}


def test_limit_bounds_reject_never_clamp():
    c = contract({"limit": {"role": "limit", "min": 1, "max": 50}})
    err(lambda: apply_arg_contracts(c, {"limit": 0}), ErrorKind.invalid_argument)
    err(lambda: apply_arg_contracts(c, {"limit": -3}), ErrorKind.invalid_argument)
    env = err(lambda: apply_arg_contracts(c, {"limit": 51}), ErrorKind.invalid_argument)
    assert env["bounds"] == [1, 50]
    assert apply_arg_contracts(c, {"limit": 50}).args_sent["limit"] == 50


def test_list_bounds_and_dedupe():
    c = contract({"genes": {"binds": f"{T}.gene", "op": "in", "min_items": 1, "max_items": 3}})
    err(lambda: apply_arg_contracts(c, {"genes": []}), ErrorKind.invalid_argument)
    env = err(lambda: apply_arg_contracts(c, {"genes": ["a", "b", "c", "d"]}), ErrorKind.invalid_argument)
    assert env["reason"] == "max_items"
    out = apply_arg_contracts(c, {"genes": ["a", "b", "a"]})
    assert out.args_sent["genes"] == ["a", "b"] and any("duplicate" in n for n in out.notes)


def test_regex_escaping_of_literal_text():
    c = contract({"disease": {"role": "free_text", "binds": f"{T}.disease", "interpreted_as": "regex"}})
    out = apply_arg_contracts(c, {"disease": "mellitus (T2D)"})
    assert out.args_sent["disease"] == r"mellitus\ \(T2D\)"
    import re
    assert re.search(out.args_sent["disease"], "diabetes mellitus (T2D)")
    assert not re.search(out.args_sent["disease"], "mellitus T2D")


def test_substring_collision():
    c = contract({"line": {"role": "free_text", "binds": f"{T}.cell_line", "interpreted_as": "substring"}})
    snap = VocabSnapshot(values=["PC-3", "BxPC-3", "HeLa"])
    env = err(lambda: apply_arg_contracts(c, {"line": "PC-3"}, {f"{T}.cell_line": snap}), ErrorKind.invalid_argument)
    assert env["reason"] == "substring_collision" and env["near"] == ["BxPC-3", "PC-3"]
    assert apply_arg_contracts(c, {"line": "HeLa"}, {f"{T}.cell_line": snap}).args_sent["line"] == "HeLa"


def test_forbid_pattern_wrap_and_escape_plugin():
    class Quoter:
        def quote(self, v):
            if "\n" in v:
                raise ValueError("newline")
            return '"' + v.replace('"', '\\"') + '"'

    class Reg:
        def find(self, kind, name):
            return Quoter() if (kind, name) == ("format", "essie") else None

    c = contract({"q": {"role": "free_text", "interpreted_as": "engine", "escape": "essie", "wrap": "({value})",
                        "forbid": ["%"]},
                  "code": {"binds": f"{T}.gene", "pattern": "^[A-Z]+$"}})
    out = apply_arg_contracts(c, {"q": 'lung "cancer"'}, registry=Reg())
    assert out.args_sent["q"] == '("lung \\"cancer\\"")'
    err(lambda: apply_arg_contracts(c, {"q": "a\nb"}, registry=Reg()), ErrorKind.invalid_argument)
    err(lambda: apply_arg_contracts(c, {"q": "100%"}, registry=Reg()), ErrorKind.invalid_argument)
    err(lambda: apply_arg_contracts(c, {"code": "abc"}), ErrorKind.invalid_argument)
    # no quoting plugin registered: only values without special characters pass
    err(lambda: apply_arg_contracts(c, {"q": 'say "x"'}, registry=None), ErrorKind.invalid_argument)


def test_output_path_confinement_and_write_once(tmp_path: Path):
    c = contract({"out": {"role": "output_path"}})
    for bad in ("../x.h5ad", "/etc/passwd", "a/../../b"):
        err(lambda bad=bad: apply_arg_contracts(c, {"out": bad}, output_dir=tmp_path), ErrorKind.invalid_argument)
    (tmp_path / "cells.h5ad").write_bytes(b"first")
    out = apply_arg_contracts(c, {"out": "cells.h5ad"}, output_dir=tmp_path)
    renamed = Path(out.renamed_outputs["out"]["to"])
    assert renamed.name.startswith("cells.") and renamed.name.endswith(".h5ad") and renamed.read_bytes() == b"first"
    assert not (tmp_path / "cells.h5ad").exists() and any("write-once" in n for n in out.notes)


def test_float32_snapping_echoes_stored_value():
    c = contract({"concentration": {"binds": f"{T}.concentration", "gateway_only": True}})
    snap = VocabSnapshot(values=[f32(0.05), f32(0.5), f32(5.0)], storage_type="float")
    out = apply_arg_contracts(c, {"concentration": 0.05}, {f"{T}.concentration": snap})
    assert out.gateway_args["concentration"] == f32(0.05) and "concentration" not in out.args_sent
    assert out.scope["concentration"] == 0.05                    # rendered in the storage type
    assert out.fixed["concentration"] == f32(0.05)
    out = apply_arg_contracts(c, {"concentration": "0.5"}, {f"{T}.concentration": snap})
    assert out.gateway_args["concentration"] == f32(0.5)
    env = err(lambda: apply_arg_contracts(c, {"concentration": 0.07}, {f"{T}.concentration": snap}),
              ErrorKind.invalid_argument)
    assert env["valid_values"] == [0.05, 0.5, 5.0]


def test_casefold_alias_and_send_map():
    c = contract({"line": {"binds": f"{T}.cell_line"},
                  "aspect": {"binds": f"{T}.aspect", "send_map": {"biological_process": "P"}}})
    snap = VocabSnapshot(values=["HeLa", "A549"])
    v = {f"{T}.cell_line": snap}
    assert apply_arg_contracts(c, {"line": "hela"}, v).args_sent["line"] == "HeLa"
    assert apply_arg_contracts(c, {"line": "HELA"}, v).args_sent["line"] == "HeLa"
    env = err(lambda: apply_arg_contracts(c, {"line": "Hela-S3"}, v), ErrorKind.invalid_argument)
    assert "HeLa" in env["valid_values"] and env["near"]
    out = apply_arg_contracts(c, {"aspect": "biological_process"})
    assert out.args_sent["aspect"] == "P"                         # the form upstream expects
    assert out.values["aspect"] == "biological_process"           # predicates keep the table's value
    err(lambda: apply_arg_contracts(c, {"aspect": "process"}), ErrorKind.invalid_argument)


def test_selector_chooses_table_and_rejects_unknown_value():
    c = contract({"method": {"role": "selector", "values": {"coloc": "s.coloc", "ecaviar": "s.ecaviar"}}})
    assert apply_arg_contracts(c, {"method": "ecaviar"}).selected_table == "s.ecaviar"
    env = err(lambda: apply_arg_contracts(c, {"method": "colc"}), ErrorKind.invalid_argument)
    assert sorted(env["valid_values"]) == ["coloc", "ecaviar"]


def test_order_by_and_direction():
    c = contract({"sort": {"role": "order_by", "values": {"score": {"column": "score", "direction": "desc"}}},
                  "asc": {"role": "order_direction"}})
    out = apply_arg_contracts(c, {"sort": "score", "asc": True})
    assert out.order == {"column": "score", "direction": "asc"}
    err(lambda: apply_arg_contracts(c, {"sort": "relevance"}), ErrorKind.invalid_argument)
    c = contract({"sort": {"role": "order_by", "binds": f"{T}.score"}})
    env = err(lambda: apply_arg_contracts(c, {"sort": "gene"}), ErrorKind.invalid_argument)
    assert env["valid_values"] == ["lfc", "padj", "score"]


def test_unbound_argument_with_non_default_value():
    c = contract({"min_sample_size": {"role": "unbound"}})
    schema = {"properties": {"min_sample_size": {"type": "integer", "default": 0}}}
    env = err(lambda: apply_arg_contracts(c, {"min_sample_size": 50}, schema=schema), ErrorKind.unsupported_filter)
    assert env["reason"] == "unbound_argument"
    apply_arg_contracts(c, {"min_sample_size": 0}, schema=schema)       # the default is fine


def test_threshold_on_comparable_within_measure_needs_its_group():
    c = contract({"max_padj": {"binds": f"{T}.padj", "op": "le"}, "contrast": {"binds": f"{T}.contrast"}})
    env = err(lambda: apply_arg_contracts(c, {"max_padj": 0.05}), ErrorKind.unsupported_combination)
    assert env["group_argument"] == "contrast"
    snap = VocabSnapshot(values=["c1", "c2"])
    apply_arg_contracts(c, {"max_padj": 0.05, "contrast": "c1"}, {f"{T}.contrast": snap})


def test_threshold_bounds_are_unsupported_filter():
    c = contract({"max_padj": {"binds": f"{T}.lfc", "op": "le", "max": 0.10}})
    env = err(lambda: apply_arg_contracts(c, {"max_padj": 0.2}), ErrorKind.unsupported_filter)
    assert env["reason"] == "scale" and env["confirmed_range"] == [None, 0.1]


def test_projection_injects_key_unique_within_and_default_filter_columns():
    c = contract({"cols": {"role": "projection", "binds": f"{T}.gene"}})
    out = apply_arg_contracts(c, {"cols": ["lfc"]})
    assert out.args_sent["cols"][:1] == ["lfc"]
    assert {"gene", "contrast", "concentration", "dup"} <= set(out.args_sent["cols"])
    schema = {"properties": {"cols": {"type": "array", "items": {"enum": ["lfc", "gene"]}}}}
    err(lambda: apply_arg_contracts(c, {"cols": ["lfc"]}, schema=schema), ErrorKind.unsupported_combination)


def test_disclosed_defaults_and_gateway_only_stripping():
    c = contract({"country": {"binds": f"{T}.dataset"}, "plate": {"binds": f"{T}.donor", "gateway_only": True}})
    schema = {"properties": {"country": {"type": "string", "default": "United States"}}}
    out = apply_arg_contracts(c, {"plate": "p1"}, schema=schema)
    assert out.args_sent == {"country": "United States"} and out.gateway_args == {"plate": "p1"}
    assert out.scope["country"] == "United States" and any("defaults to" in n for n in out.notes)


def test_flags_compile_to_positive_predicates():
    c = contract({"no_events": {"role": "flag", "binds": f"{T}.safety", "when_true": {"in": ["none_recorded"]}}})
    out = apply_arg_contracts(c, {"no_events": True})
    p = out.flags["no_events"]
    assert evaluate(p, {"safety": "none_recorded"}) is True
    assert evaluate(p, {"safety": None}) is None                 # unknown never counts as the favourable state
    assert evaluate(p, {"safety": "recorded"}) is False


def test_predicates_abs_binds_any_and_pairs():
    c = contract({"min_abs": {"binds": f"{T}.lfc", "op": "gt_abs"},
                  "gene": {"binds_any": [f"{T}.a", f"{T}.b"]},
                  "other": {"binds_any": [f"{T}.a", f"{T}.b"]}})
    pred, per_arg = build_predicate(c, {"min_abs": 1.0}, registry=REGISTRY)
    assert any(isinstance(q, CmpAbs) for q in (pred.preds if isinstance(pred, And) else [pred]))
    assert evaluate(pred, {"lfc": -2.0}) is True and evaluate(pred, {"lfc": 0.5}) is False
    assert evaluate(pred, {"lfc": None}) is None
    pred, per_arg = build_predicate(c, {"gene": "X"})
    assert isinstance(per_arg["gene"], Or)
    assert evaluate(pred, {"a": "Y", "b": "X"}) is True
    pred, per_arg = build_predicate(c, {"gene": "X", "other": "Y"})
    p = per_arg["gene+other"]
    assert isinstance(p, Or) and all(isinstance(q, And) for q in p.preds)
    assert evaluate(p, {"a": "Y", "b": "X"}) is True and evaluate(p, {"a": "X", "b": "Z"}) is False


def test_container_predicates_merge_into_one_any():
    c = contract({"go_id": {"binds": f"{T}.go[].id"}, "ev": {"binds": f"{T}.go[].evidence"}})
    pred, _ = build_predicate(c, {"go_id": "GO:1", "ev": "IDA"})
    assert isinstance(pred, AnyItem) and isinstance(pred.pred, And)
    row = {"go": [{"id": "GO:1", "evidence": "IEA"}, {"id": "GO:2", "evidence": "IDA"}]}
    assert evaluate(pred, row) is False                   # no single item satisfies both (C21)
    assert evaluate(pred, {"go": [{"id": "GO:1", "evidence": "IDA"}]}) is True
    pred, _ = build_predicate(c, {"go_id": ["GO:1", "GO:2"]})
    assert isinstance(pred, AnyItem) and isinstance(pred.pred, In)
    assert Eq("x", 1) != pred
