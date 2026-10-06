"""Canonical row keys (§6.3): storage-typed float rendering, nulls, order-insensitive parts, hashes."""

from __future__ import annotations

import json
import struct

import pytest

from vbt.datalayer.rowkey import canonical, canonical_sha256, content_hash, element_type, render_float, render_value


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def test_float32_renders_shortest_roundtrip():
    stored = _f32(0.05)
    assert stored != 0.05                              # 0.05000000074505806 in float64
    assert render_float(stored, "float") == "0.05"
    assert render_float(stored, "float32") == "0.05"
    assert render_float(0.05, "double") == "0.05"
    assert render_float(stored, "double") == repr(stored)   # in a float64 column it is a different value
    for v in (0.5, 5.0, 1e-5, 123.456, -2.75, 3.4e38):
        assert float(render_float(_f32(v), "float")) == pytest.approx(v, rel=1e-6)
        assert _f32(float(render_float(_f32(v), "float"))) == _f32(v)


def test_float16_and_integral_floats():
    assert render_float(4.0, "float") == "4.0"
    assert render_float(4, "double") == "4.0"
    assert render_float(0.1, "halffloat") == "0.1"
    assert render_float(float("inf"), "double") == "Infinity"


def test_null_and_nan_are_explicit_null():
    assert canonical([None, float("nan")], ["double", "double"]) == "[null,null]"
    assert render_float(float("nan"), "float") == "null"
    assert canonical([1, None, "x"]) == '[1,null,"x"]'


def test_float32_key_parts_match_across_sides():
    # witness side: the float32 value read from Parquet; upstream side: the JSON float 0.05
    types = ["string", "float", "string"]
    witness = canonical(["Erdafitinib", _f32(0.05), "uM"], types)
    upstream = canonical(["Erdafitinib", 0.05, "uM"], types)
    assert witness == upstream == '["Erdafitinib",0.05,"uM"]'


def test_list_parts_are_order_insensitive():
    a = canonical([["b", "a", "c"], 1], ["list<element: string>", "int64"])
    b = canonical([["c", "b", "a"], 1], ["list<element: string>", "int64"])
    assert a == b == '[["a","b","c"],1]'
    assert canonical([{"x", "y"}]) == canonical([("y", "x")])
    floats = canonical([[_f32(0.5), _f32(0.05)]], ["large_list<item: float>"])
    assert floats == "[[0.05,0.5]]"


def test_dict_keys_sorted():
    assert render_value({"b": 1, "a": [2, 1]}) == '{"a":[1,2],"b":1}'
    assert canonical([{"z": None, "a": 1.5}]) == canonical([{"a": 1.5, "z": None}])


def test_unicode_kept_and_bool_not_int():
    assert canonical(["Ünïcode", True, 1]) == '["Ünïcode",true,1]'
    assert json.loads(canonical(["α", None, 2.5])) == ["α", None, 2.5]


def test_storage_types_must_match_length():
    with pytest.raises(ValueError):
        canonical([1, 2], ["int64"])


def test_sha_is_stable_and_order_insensitive():
    keys = [["ENSG1", 1.0], ["ENSG2", None]]
    h1 = canonical_sha256(keys, ["string", "double"])
    h2 = canonical_sha256(list(reversed(keys)), ["string", "double"])
    h3 = canonical_sha256([canonical(k, ["string", "double"]) for k in keys])
    assert h1 == h2 == h3
    assert len(h1) == 64
    assert canonical_sha256([["ENSG1", 1.0]], ["string", "double"]) != h1


def test_content_hash_uses_declared_columns():
    row = {"a": 1, "b": _f32(0.05), "c": "ignored"}
    h = content_hash(row, ["a", "b"], {"b": "float"})
    assert h == content_hash({"b": 0.05, "a": 1, "c": "other"}, ["a", "b"], {"b": "float"})
    assert h != content_hash(row, ["a", "b", "c"], {"b": "float"})
    assert content_hash({"x": 1, "y": 2}) == content_hash({"y": 2, "x": 1})


def test_element_type():
    assert element_type("list<element: float>") == "float"
    assert element_type("large_list<item: double>") == "double"
    assert element_type("fixed_size_list<element: float>[3]") == "float"
    assert element_type("string") is None
