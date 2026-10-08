"""Descriptor and overlay models (§6.1, §6.7, §7, §8.1), the path grammar and reference scoping (§6.4),
loading (§6.1) and every lint rule of §6.9 with a positive and a negative case."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from vbt.datalayer.descriptor.columns import ROLE_MODELS, facet_names, validate_column
from vbt.datalayer.descriptor.lint import DEFECT_WHERE, Finding, errors, lint_descriptor, lint_overlay, warnings
from vbt.datalayer.descriptor.load import (
    DescriptorError,
    digest,
    expand,
    load_descriptor,
    load_descriptors,
    load_overlays,
)
from vbt.datalayer.descriptor.models import KeySpec, MatrixSpec, Sentinel, SourceDescriptor, TableSpec
from vbt.datalayer.descriptor.overlay import ArgBinding, Overlay, RecomputeSpec, ResultSpec
from vbt.datalayer.descriptor.scoping import AmbiguousReference, TableScope, UnknownReference, resolve_reference
from vbt.datalayer.plugins.base import IdentifierBase, PluginBase
from vbt.datalayer.plugins.registry import PluginRegistry
from vbt.datalayer.roles import (
    COMMON_FACETS,
    KEYABLE_ROLES,
    ROLE_FACETS,
    ItemCond,
    PathError,
    Role,
    arrow_compatible,
    facets_for,
    format_path,
    parse_path,
    type_family,
)

# ---------------------------------------------------------------------------
# A small valid source to mutate per test
# ---------------------------------------------------------------------------

BASE: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "src", "title": "Test source", "release": {"from": "literal"},
    "id_types": {
        "gene": {"plugin": "gene_plugin", "universe": "genes.id", "resolve_via": ["genes.symbol"]},
        "symbol": {"plugin": "symbol_plugin", "label_of": "gene"},
        "drug": {"plugin": "drug_plugin", "universe": "drugs.id", "canonicalize": {"parent": "drugs.parentId"}},
    },
    "tables": {
        "genes": {"kind": "entity", "grain": "one gene", "key": {"columns": ["id"], "check": "full"},
                  "columns": {"id": {"role": "identifier", "id_type": "gene", "self": True},
                              "symbol": {"role": "label", "of": "id", "id_type": "symbol"},
                              "go": {"role": "nested", "item_key": ["termId", "aspect"],
                                     "fields": {"termId": {"role": "category"},
                                                "aspect": {"role": "category", "vocab": ["C", "F", "P"],
                                                           "scope": {"kind": "stratum", "determined_by": ["termId"]}},
                                                "label": {"role": "label", "of": "termId"}}}}},
        "genes_go": {"kind": "sets", "items_of": {"table": "genes", "path": "go[]"}, "grain": "one annotation",
                     "key": {"columns": []}},
        "drugs": {"kind": "entity", "grain": "one drug", "key": {"columns": ["id"]},
                  "columns": {"id": {"role": "identifier", "id_type": "drug", "self": True},
                              "parentId": {"role": "identifier", "id_type": "drug", "ref": "drugs.id"}}},
        "facts": {"kind": "fact", "grain": "one effect", "key": {"columns": ["geneId", "drugId", "dose"],
                                                                 "nullable": ["dose"]},
                  "rank": [{"column": "effect", "direction": "asc"}],
                  "columns": {"geneId": {"role": "identifier", "id_type": "gene", "ref": "genes.id"},
                              "drugId": {"role": "identifier", "id_type": "drug", "ref": "drugs.id"},
                              "dose": {"role": "scope", "unit_from": "unit", "vocab": "data"},
                              "unit": {"role": "scope", "scope": {"kind": "condition", "determined_by": ["dose"]}},
                              "effect": {"role": "measure", "comparable_within": ["dose"], "missing": "unknown"}}},
    },
}


def mk(mutate=None) -> dict[str, Any]:
    d = copy.deepcopy(BASE)
    if mutate is not None:
        mutate(d)
    return d


def model(d: dict[str, Any]) -> SourceDescriptor:
    return SourceDescriptor.model_validate(d)


def lint(d: dict[str, Any], **kw) -> list[Finding]:
    return lint_descriptor(model(d), **kw)


def rules(findings: list[Finding], level: str = "error") -> set[str]:
    return {f.rule for f in findings if f.level == level}


def test_base_descriptor_lints_clean():
    findings = lint(mk())
    assert not findings, findings


# ---------------------------------------------------------------------------
# Roles and facets (§7)
# ---------------------------------------------------------------------------

def test_role_models_carry_exactly_the_role_facets():
    assert len(Role) == 20 and set(ROLE_MODELS) == {r.value for r in Role}
    for role, m in ROLE_MODELS.items():
        assert facet_names(m) == facets_for(role), role
    assert "path" in COMMON_FACETS and "scope" in COMMON_FACETS and "level" in COMMON_FACETS
    assert ROLE_FACETS[Role.payload] == frozenset() and {Role.text, Role.payload, Role.vector,
                                                          Role.ignore}.isdisjoint(KEYABLE_ROLES)


@pytest.mark.parametrize("spec", [
    {"role": "label", "scale": [0, 1]},
    {"role": "measure", "vocab": "data"},
    {"role": "category", "statistic": "numeric"},
    {"role": "identifier", "of": "x"},
    {"role": "flag", "levels": ["a"]},
    {"role": "payload", "missing": "unknown"},
    {"role": "ignore", "path": "[]"},
    {"role": "nested", "id_type": "x"},
    {"role": "count", "unit": "x"},
    {"role": "qualifier"},                                  # effect is required
    {"role": "nope"},
    {"role": "measure", "scale": [1, 0]},
])
def test_wrong_facet_for_role_rejected(spec):
    with pytest.raises(ValidationError):
        validate_column(spec)


def test_column_model_details():
    assert validate_column({"role": "scope", "vocab": "data"}).scope.pooling == "forbid"
    assert validate_column({"role": "category", "scope": {"kind": "replicate"}}).scope.pooling == "list"
    assert validate_column({"role": "category", "scope": {"kind": "partition"}}).scope.pooling == "group"
    ident = validate_column({"role": "identifier", "self": True, "ref": {"table": "patient", True: {"a": "b"}}})
    assert ident.self_ and ident.ref.on == {"a": "b"}           # YAML 1.1 reads `on:` as True
    assert validate_column({"role": "flag", "missing": False}).missing == "false"
    nested = validate_column({"role": "nested", "item_key": ["a"], "fields": {
        "a": {"role": "nested", "item_key": {"identity": "position", "max_items": 1},
              "fields": {"b": {"role": "measure"}}}}})
    assert nested.item_key.columns == ["a"] and nested.fields["a"].item_key.max_items == 1
    member = validate_column({"role": "member", "item_key": {"identity": "value"},
                              "membership": {"set": {"parent": "set_id"}, "member": {"path": "[]", "id_type": "g"},
                                             "enrichment": {"test": "hypergeom_enrichment",
                                                            "universe": {"table": "o.t", "keys": ["id"]}}}})
    assert member.membership.member.path == "[]" and member.membership.enrichment.family.include_zero_overlap
    with pytest.raises(ValidationError):
        validate_column({"role": "member", "membership": {"set": {"parent": "a", "path": "b"},
                                                          "member": {"path": "[]"}}})
    with pytest.raises(ValidationError):
        validate_column({"role": "nested", "item_key": {"identity": "key"}})
    parse = validate_column({"role": "category", "parse": {"pattern": r"^(?P<sym>\S+) \((?P<id>\d+)\)$",
                                                           "fields": {"id": {"role": "identifier"}}}})
    assert set(parse.parse.fields) == {"id"}
    with pytest.raises(ValidationError):
        validate_column({"role": "category", "parse": {"pattern": "(x)", "fields": {}}})


def test_arrow_compat():
    assert type_family("large_string") == "string" and type_family("int32") == "int"
    assert type_family("large_list<item: struct<a: int32>>") == "list"
    assert type_family("dictionary<values=string, indices=int32, ordered=0>") == "string"
    assert arrow_compatible("measure", "float") and arrow_compatible("measure", "list<element: double>")
    assert not arrow_compatible("measure", "string") and arrow_compatible("measure", "string", parse="number")
    assert arrow_compatible("count", "int32") and not arrow_compatible("count", "double")
    assert arrow_compatible("count", "double", stored_as="float64")
    assert arrow_compatible("flag", "bool") and not arrow_compatible("flag", "int64")
    assert arrow_compatible("flag", "int64", encoding={1: True})
    assert arrow_compatible("time", "string", stored_as="iso8601_string")
    assert arrow_compatible("vector", "fixed_size_list<element: float>[100]")
    assert not arrow_compatible("vector", "list<element: string>")
    assert arrow_compatible("identifier", "large_list<item: large_string>")
    assert not arrow_compatible("identifier", "null") and arrow_compatible("payload", "null")
    assert arrow_compatible("nested", "struct<a: int64>") and not arrow_compatible("nested", "string")


# ---------------------------------------------------------------------------
# Path grammar (§6.4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "id", "genomicLocation.chromosome", "go[].id", "path[][]",
    "geneEssentiality[].depMapEssentiality[].screens[].geneEffect", "tissues[].protein.cell_type[].name",
    "dbXrefs[source=NCBI_Gene].id", 'x[name="a b"].y', "a[k=1.5].b", "a[flag=true]", "a[x=null].y",
    "@row.ModelID", "@col.entrez_id", "^.source", "^.^.id", "/id", "[].label", "[]", "`odd name`.x",
    "members[]",
])
def test_path_round_trip(text):
    p = parse_path(text)
    assert parse_path(format_path(p)) == p


def test_path_details():
    p = parse_path("path[][]")
    assert p.segments[0].brackets == ("[]", "[]") and p.crosses_list
    q = parse_path("dbXrefs[source=NCBI_Gene].id")
    assert q.segments[0].brackets == (ItemCond("source", "NCBI_Gene"),) and q.segments[1].name == "id"
    assert parse_path("@obs.patient_id").axis == "row" and parse_path("@var.g").axis == "col"
    assert format_path(parse_path("@obs.x")) == "@row.x"
    assert parse_path("^.^.x").up == 2 and parse_path("/x").absolute
    assert parse_path("[].label").is_item_relative
    container, rest = parse_path("a.b[].c[].d").split_container()
    assert format_path(container) == "a.b[]" and format_path(rest) == "c[].d"
    assert parse_path("a[k=1].b").segments[0].brackets[0].value == 1
    assert ItemCond("source", "x").matches({"source": "x"}) and not ItemCond("s", 1).matches(3)


@pytest.mark.parametrize("bad", ["", "a.", "a..b", ".a", "@foo.x", "@row", "a[b]", "a b", "a[x=]", "a[x=1", "`a",
                                 "/^.a", "a.[]"])
def test_bad_paths(bad):
    with pytest.raises(PathError):
        parse_path(bad)


# ---------------------------------------------------------------------------
# Reference scoping (§6.4)
# ---------------------------------------------------------------------------

SCOPED: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "s", "title": "t", "release": {"from": "literal"},
    "tables": {
        "t": {"kind": "entity", "grain": "g", "key": {"columns": ["id"]}, "columns": {
            "id": {"role": "identifier", "self": True},
            "source": {"role": "category"},
            "xr": {"role": "nested", "item_key": ["source"], "fields": {
                "source": {"role": "category"},
                "ids": {"role": "identifier", "path": "[]", "xref": True,
                        "id_type_from": {"field": "^.source", "map": {}}},
                "inner": {"role": "nested", "item_key": ["k"],
                          "fields": {"k": {"role": "category"}, "id": {"role": "category"}}}}}}},
        "u": {"kind": "fact", "grain": "g", "key": {"columns": ["tid"]},
              "columns": {"tid": {"role": "identifier", "ref": "t.id"}}},
        "t_items": {"kind": "sets", "items_of": {"table": "t", "path": "xr[]"}, "grain": "g", "key": {"columns": []}},
        "m": {"kind": "matrix", "grain": "cell", "key": {"columns": ["@row.a", "@col.b"]}, "matrix": {
            "axes": {"obs": {"name": "r", "from": "index", "key": {"columns": ["a"]},
                             "columns": {"a": {"role": "identifier"}, "dup": {"role": "category"}}},
                     "var": {"name": "c", "from": "index", "key": {"columns": ["b"]},
                             "columns": {"b": {"role": "identifier"}, "dup": {"role": "category"}}}},
            "values": {"X": {"role": "measure"}}}},
    },
}


def test_scoping_sibling_then_enclosing_then_table():
    d = model(SCOPED)
    s = TableScope(d, "t")
    assert s.resolve("source", ["xr"]).path == "xr.source"              # sibling first
    assert s.resolve("^.source", ["xr"]).path == "source"               # ^. skips the item level
    assert s.resolve("^.source", ["xr", "[]"]).path == "xr.source"      # element level of a per-element column
    assert s.resolve("id", ["xr", "inner"]).path == "xr.inner.id"
    assert s.resolve("^.^.id", ["xr", "inner"]).path == "id"
    assert s.resolve("/id", ["xr", "inner"]).path == "id"
    assert s.resolve("k", ["xr", "inner"]).level == 2
    assert s.resolve("source", ["xr", "inner"]).path == "xr.source"     # enclosing item outward
    assert s.resolve("xr[].inner[].k").path == "xr.inner.k"
    assert s.resolve("[]", ["xr"]).kind == "element"
    items = TableScope(d, "t_items")
    assert items.resolve("source").path == "xr.source" and items.resolve("^.source").path == "source"


def test_scoping_qualified_and_errors():
    d = model(SCOPED)
    s = TableScope(d, "t")
    r = s.resolve("u.tid", mode="table_column")
    assert (r.kind, r.table, r.path) == ("table_column", "u", "tid")
    assert s.resolve("s.u.tid", mode="table_column").table == "u"
    unloaded = s.resolve("other.tab.col", mode="table_column")
    assert unloaded.kind == "foreign" and not unloaded.loaded
    for bad, cont in (("nope", ["xr"]), ("xr[].nope", []), ("^.^.^.id", ["xr"]), ("u.nope", [])):
        with pytest.raises(UnknownReference):
            s.resolve(bad, cont, mode="table_column" if bad.startswith("u.") else "column")
    with pytest.raises(UnknownReference):
        s.resolve("id", mode="table_column")                            # a bare name is not table.col
    m = TableScope(d, "m")
    assert m.resolve("@row.dup").kind == "axis" and m.resolve("a").path == "@row.a"
    with pytest.raises(AmbiguousReference):
        m.resolve("dup")
    # resolve_reference with a lone TableSpec: scoped names only
    spec = d.tables["t"]
    assert resolve_reference(spec, ["xr"], "source").path == "xr.source"
    assert resolve_reference(spec, ["xr"], "^.source", descriptor=d, table_name="t").path == "source"


def test_scoping_ambiguity_between_column_path_and_table_ref():
    d = mk(lambda d: d["tables"]["facts"]["columns"].update(
        {"genes": {"role": "nested", "fields": {"id": {"role": "category"}}}}))
    scope = TableScope(model(d), "facts")
    with pytest.raises(AmbiguousReference):
        scope.resolve("genes.id", mode="any")
    assert scope.resolve("genes.id").path == "genes.id"                 # column mode: scoped first
    assert scope.resolve("genes.id", mode="table_column").table == "genes"


# ---------------------------------------------------------------------------
# Models: keys, item tables, matrices, sentinels
# ---------------------------------------------------------------------------

def test_item_table_key_rule():
    with pytest.raises(ValidationError, match="composed"):
        TableSpec.model_validate({"kind": "sets", "items_of": {"table": "genes", "path": "go[]"}, "grain": "g",
                                  "key": {"columns": ["termId"]}})
    with pytest.raises(ValidationError):
        TableSpec.model_validate({"kind": "sets", "items_of": {"table": "genes", "path": "go[]"}, "grain": "g",
                                  "key": {"columns": []}, "path": "x"})
    with pytest.raises(ValidationError):
        TableSpec.model_validate({"kind": "sets", "items_of": {"table": "genes", "path": "go"}, "grain": "g",
                                  "key": {"columns": []}})


def test_key_nullable_subset_and_matrix_required():
    with pytest.raises(ValidationError, match="nullable"):
        KeySpec(columns=["a"], nullable=["b"])
    with pytest.raises(ValidationError):
        KeySpec(columns=["a", "a"])
    with pytest.raises(ValidationError, match="matrix"):
        TableSpec.model_validate({"kind": "matrix", "grain": "g", "key": {"columns": ["@row.a"]}})
    with pytest.raises(ValidationError, match="columns are required"):
        TableSpec.model_validate({"kind": "fact", "grain": "g", "key": {"columns": ["a"]}})
    with pytest.raises(ValidationError):
        MatrixSpec.model_validate({"axes": {"obs": {"name": "r", "from": "index", "key": {"columns": ["a"]}},
                                            "row": {"name": "r", "from": "index", "key": {"columns": ["a"]}}},
                                   "values": {"X": {"role": "measure"}}})


def test_sentinel_expect_grammar():
    Sentinel(key={"id": "X"}, expect={"approvedSymbol": "PCSK9", "nonempty": ["go"], "contains": {"go[].id": "GO:1"},
                                      "items": {"tissues[]": ">=1"}, "is_null": ["a"], "min_rows": 1,
                                      "gene_effect": {"lt": -0.5}})
    for bad in ({"nonempty": "go"}, {"items": {"x[]": "lots"}}, {"min_rows": -1}, {"x": {"lt": 1, "gt": 0}},
                {"x": {"near": 1}}):
        with pytest.raises(ValidationError):
            Sentinel(key={"id": "X"}, expect=bad)
    with pytest.raises(ValidationError):
        Sentinel(key={})


def test_overlay_model_details():
    a = ArgBinding.model_validate({"binds": "s.t.c", "existence": False})     # YAML `off`
    assert a.existence == "off"
    with pytest.raises(ValidationError):
        ArgBinding.model_validate({"wrap": "(x)"})
    with pytest.raises(ValidationError):
        ArgBinding.model_validate({"binds": "s.t.c", "op": "regex"})
    r = ResultSpec.model_validate({"summary_fields": {"$.n": {"recompute": {"agg": "count_distinct", "of": "g"}},
                                                      "$.m": "drop"}, "echo": {"id": "$.id"}})
    assert isinstance(r.summary_fields["$.n"], RecomputeSpec) and r.summary_fields["$.m"] == "drop"
    assert r.echo_specs()["id"].source == "record"
    assert ArgBinding.model_validate({"binds": {"coloc": "a.b.c"}, "binds_any": ["a.b.d"]}).bound_columns == \
        ["a.b.c", "a.b.d"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_expand_variables_and_run(monkeypatch):
    monkeypatch.setenv("DL_TEST_ROOT", "/data/ot")
    monkeypatch.delenv("DL_TEST_UNSET", raising=False)
    data = {"root": "${DL_TEST_ROOT}", "x": "${DL_TEST_UNSET:-fallback}", "v": "${vars.up}/y",
            "out": "${run.mcp_output_dir}", "nested": ["${run.other:-d}"]}
    assert expand(data, {"up": "/u"}) == {"root": "/data/ot", "x": "fallback", "v": "/u/y",
                                          "out": "${run.mcp_output_dir}", "nested": ["d"]}
    assert expand(data, {"up": "/u"}, {"mcp_output_dir": "/run/out"})["out"] == "/run/out"


def test_load_descriptors_and_overlays(tmp_path, monkeypatch):
    import yaml
    monkeypatch.setenv("DL_TEST_ROOT", "/data/src")
    d = mk()
    d["root"] = "${DL_TEST_ROOT}"
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "src.yaml").write_text(yaml.safe_dump(d))
    (tmp_path / "src" / ".hidden.yaml").write_text("not: [valid")
    loaded = load_descriptors(tmp_path / "src")
    assert set(loaded) == {"src"} and loaded["src"].root == "/data/src"
    assert digest(loaded["src"]) == digest(load_descriptor(tmp_path / "src" / "src.yaml"))
    other = mk(lambda x: x.update(title="changed"))
    assert digest(model(other)) != digest(loaded["src"])
    (tmp_path / "src" / "dup.yaml").write_text(yaml.safe_dump(d))
    with pytest.raises(DescriptorError, match="also declared"):
        load_descriptors(tmp_path / "src")
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({**d, "tables": {"x": {"kind": "fact"}}}))
    with pytest.raises(DescriptorError, match="tables.x"):
        load_descriptor(tmp_path / "bad.yaml")
    ov = tmp_path / "ov"
    ov.mkdir()
    (ov / "drug.yaml").write_text(yaml.safe_dump({"schema": "vbt.overlay/1", "server": "drug", "tools": {}}))
    (ov / "_generic.yaml").write_text(yaml.safe_dump({"schema": "vbt.overlay/1", "server": "*",
                                                      "generic": {"not_found_when": ["$.error =~ 'not found'"]}}))
    reviewed, generic = load_overlays(ov)
    assert set(reviewed) == {"drug"} and len(generic) == 1 and generic[0].generic.not_found_when
    assert load_descriptors(tmp_path / "missing") == {}


# ---------------------------------------------------------------------------
# Lint rules (§6.9): each with a positive (finding) and a negative (clean) case
# ---------------------------------------------------------------------------

def test_rule_strict_roles():
    def bare_nested(d):
        d["tables"]["genes"]["columns"]["extra"] = {"role": "nested"}

    assert "strict_roles" not in rules(lint(mk(bare_nested)), "warning")     # non-strict descriptor: nothing
    assert "strict_roles" in rules(lint(mk(bare_nested), strict=True))     # --strict: error
    strict_desc = mk(bare_nested)
    strict_desc["strict"] = True
    assert "strict_roles" in rules(lint(strict_desc), "warning")            # descriptor strict, phase 1: warning
    assert "strict_roles" not in rules(lint(strict_desc))
    assert not lint(mk(), strict=True)


def test_rule_key_columns_exist_and_keyable():
    def missing(d):
        d["tables"]["drugs"]["key"]["columns"] = ["nope"]

    def text_key(d):
        d["tables"]["drugs"]["columns"]["notes"] = {"role": "text"}
        d["tables"]["drugs"]["key"]["columns"] = ["id", "notes"]

    def empty(d):
        d["tables"]["drugs"]["key"]["columns"] = []

    for mutate in (missing, text_key, empty):
        assert "key" in rules(lint(mk(mutate))), mutate.__name__


def test_rule_key_nullable_parts():
    def all_nullable(d):
        d["tables"]["facts"]["key"]["nullable"] = ["geneId", "drugId", "dose"]

    def nullable_self(d):
        d["tables"]["drugs"]["key"] = {"columns": ["id"], "nullable": ["id"]}

    assert "key" in rules(lint(mk(all_nullable)))
    msgs = [f.message for f in errors(lint(mk(nullable_self)))]
    assert any("self identifier" in m for m in msgs) and any("every key part is nullable" in m for m in msgs)

    def alt_key(d):
        d["tables"]["drugs"]["columns"]["inchiKey"] = {"role": "identifier", "alternate_key": True}
        d["tables"]["drugs"]["alternate_keys"] = [{"columns": ["inchiKey"], "nullable": ["inchiKey"]}]

    assert not errors(lint(mk(alt_key)))                                    # an alternate key may be one nullable part


def test_rule_item_table_key_composition():
    def no_item_key(d):
        del d["tables"]["genes"]["columns"]["go"]["item_key"]

    def bad_container(d):
        d["tables"]["genes_go"]["items_of"]["path"] = "nothere[]"

    def missing_parent(d):
        d["tables"]["genes_go"]["items_of"]["table"] = "ghost"

    for mutate in (no_item_key, bad_container, missing_parent):
        assert "key" in rules(lint(mk(mutate))), mutate.__name__


def test_rule_reference_item_field_shadowing_a_column():
    """25.09 target_go declared ``gene: [id]``: inside the item ``id`` is the GO item's id, so the grain counted GO
    terms. A bare name that resolves to an item field while the table has a column of that name is a warning."""
    def shadowing(grain):
        def mutate(d):
            d["tables"]["genes"]["columns"]["go"]["fields"]["id"] = {"role": "identifier"}
            d["tables"]["genes_go"]["grains"] = {"gene": grain, "term": ["go[].termId"]}
        return mutate

    found = [f for f in lint(mk(shadowing(["id"]))) if f.level == "warning"]
    assert [f.rule for f in found] == ["reference"] and "'/id'" in found[0].message, found
    assert "grains.gene" in found[0].where
    assert not lint(mk(shadowing(["/id"])))
    assert not lint(mk(shadowing(["go[].id"])))


def test_rule_scope_key():
    def stray_scope(d):
        d["tables"]["facts"]["columns"]["batch"] = {"role": "category", "scope": {"kind": "batch"}}

    def determined(d):
        stray_scope(d)
        d["tables"]["facts"]["columns"]["batch"]["scope"]["determined_by"] = ["drugId"]

    def aggregated(d):
        stray_scope(d)
        d["tables"]["facts"]["aggregated_over"] = [{"dimension": "batch", "how": "mean"}]

    def nested_scope_out_of_item_key(d):
        d["tables"]["genes"]["columns"]["go"]["fields"]["aspect"]["scope"] = {"kind": "stratum"}
        d["tables"]["genes"]["columns"]["go"]["item_key"] = ["termId"]

    assert "scope_key" in rules(lint(mk(stray_scope)))
    assert "scope_key" not in rules(lint(mk(determined)))
    assert "scope_key" not in rules(lint(mk(aggregated)))
    assert "scope_key" in rules(lint(mk(nested_scope_out_of_item_key)))
    assert "scope_key" not in rules(lint(mk()))                             # dose in key; aspect in item key


def test_rule_references():
    def bad_of(d):
        d["tables"]["genes"]["columns"]["symbol"]["of"] = "nope"

    def bad_ref(d):
        d["tables"]["facts"]["columns"]["geneId"]["ref"] = "genes.nope"

    def bad_comparable(d):
        d["tables"]["facts"]["columns"]["effect"]["comparable_within"] = ["nope"]

    def bad_item_key(d):
        d["tables"]["genes"]["columns"]["go"]["item_key"] = ["symbol"]    # a table column, not an item field

    def parent_ref(d):
        d["tables"]["genes"]["columns"]["go"]["fields"]["label"]["of"] = "^.id"

    def unloaded(d):
        d["tables"]["facts"]["columns"]["geneId"]["ref"] = "elsewhere.genes.id"

    def release_dimension(d):
        d["tables"]["facts"]["columns"]["effect"]["comparable_within"] = ["release"]

    for mutate in (bad_of, bad_ref, bad_comparable, bad_item_key):
        assert "reference" in rules(lint(mk(mutate))), mutate.__name__
    assert not errors(lint(mk(parent_ref)))
    findings = lint(mk(unloaded))
    assert not errors(findings) and "reference" in rules(findings, "warning")
    assert not lint(mk(release_dimension))


def test_rule_soft_references_follow_strictness():
    def constraint(d):
        d["tables"]["facts"]["constraints"] = [{"column": "unrolled", "op": "<", "value": 1, "origin": "x.py:1"}]

    assert "reference_soft" in rules(lint(mk(constraint)), "warning")
    assert "reference_soft" in rules(lint(mk(constraint), strict=True))


def test_rule_id_types():
    def unknown(d):
        d["tables"]["facts"]["columns"]["geneId"]["id_type"] = "nope"

    def qualified_missing(d, other):
        d["tables"]["facts"]["columns"]["geneId"]["id_type"] = "other:nope"

    assert "id_type" in rules(lint(mk(unknown)))
    other = model({"schema": "vbt.datasource/1", "source": "other", "title": "o", "release": {"from": "literal"},
                   "id_types": {"gene": {"plugin": "gene_plugin"}},
                   "tables": {"t": {"kind": "entity", "grain": "g", "key": {"columns": ["id"]},
                                    "columns": {"id": {"role": "identifier", "self": True}}}}})
    d = mk(lambda x: qualified_missing(x, other))
    assert "id_type" in rules(lint_descriptor(model(d), loaded_sources={"other": other}))
    assert "id_type" in rules(lint_descriptor(model(d)), "warning")         # source not loaded: not checked
    ok = mk(lambda x: x["tables"]["facts"]["columns"]["geneId"].update(id_type="other:gene"))
    assert not errors(lint_descriptor(model(ok), loaded_sources={"other": other}))


class _Ident(IdentifierBase):
    name = id_type = "gene_plugin"
    canonical = r"G\d+"
    examples = ("G1",)


class _Digits(IdentifierBase):
    name = id_type = "digits"
    canonical = r"\d+"
    examples = ("123", "4567")


class _Stat(PluginBase):
    kind = "statistic"
    name = "numeric"

    def sort_key(self, *a): ...
    def predicate(self, *a): ...
    def bounds(self, *a): ...
    def validate(self, *a): ...
    def aggregate(self, *a): ...
    def comparable(self, *a): ...
    def describe(self, *a): ...
    def family(self, *a): ...


def _registry(*plugins) -> PluginRegistry:
    reg = PluginRegistry()
    for p in plugins:
        reg.add(p)
    return reg


def test_rule_plugins_only_with_registry():
    d = mk()
    assert not rules(lint(d))                                               # registry=None: names not checked
    reg = _registry(_Ident, _Stat)
    found = lint(d, registry=reg)
    assert "plugin" in rules(found) and any("symbol_plugin" in f.message for f in errors(found))

    def gene_effect(x):
        x["tables"]["facts"]["columns"]["effect"]["statistic"] = "gene_effect"

    def with_fallback(x):
        gene_effect(x)
        x["tables"]["facts"]["columns"]["effect"]["fallback"] = "numeric"

    stat_errors = [f for f in errors(lint(mk(gene_effect), registry=reg)) if "statistic" in f.message]
    assert stat_errors
    assert not [f for f in errors(lint(mk(with_fallback), registry=reg)) if "statistic" in f.message]


def test_rule_verified_has_confirmation_path():
    def family(d):
        d["tables"]["facts"]["columns"]["effect"].update(family=["drugId", "dose"], verified=False)

    def documented(d):
        family(d)
        d["tables"]["facts"]["columns"]["effect"]["verified_by"] = "upstream_doc"

    assert "verified" in rules(lint(mk(family)), "warning")
    assert "verified" not in rules(lint(mk(documented)), "warning")


def test_rule_coverage():
    def censored(d):
        d["tables"]["facts"]["coverage"] = {"statement": "only padj < 0.1 stored", "absence_means": "censored"}

    def censored_ok(d):
        censored(d)
        d["tables"]["facts"]["coverage"]["censor"] = {"column": "effect", "op": ">=", "value": 0.1}

    assert "coverage" in rules(lint(mk(censored)))
    assert not errors(lint(mk(censored_ok)))
    with pytest.raises(ValidationError):
        model(mk(lambda d: d["tables"]["facts"].update(coverage={"absence_means": "absent"})))


def test_rule_partitions():
    def partition(d, vocab):
        d["tables"]["facts"]["partitions"] = {"sourceId": {"column": {"role": "category", "vocab": vocab,
                                                                      "scope": {"kind": "partition"}},
                                                           "expect": "declared"}}
        d["tables"]["facts"]["key"]["columns"].append("sourceId")
        d["tables"]["facts"]["access_paths"] = [{"columns": ["sourceId"], "via": "partition"}]

    assert "partition" in rules(lint(mk(lambda d: partition(d, "data"))))
    assert not errors(lint(mk(lambda d: partition(d, ["chembl", "europepmc"]))))

    def not_a_partition(d):
        d["tables"]["facts"]["access_paths"] = [{"columns": ["drugId"], "via": "partition"}]

    assert "partition" in rules(lint(mk(not_a_partition)))


def test_rule_resolver_columns():
    def canonicalize(d):
        d["id_types"]["drug"]["canonicalize"] = {"parent": "drugs.nope"}

    def crosswalk(d):
        d["id_types"]["gene"]["crosswalks"] = [{"name": "cw", "table": "ghost", "from": "a", "to": "b"}]

    def stored_forms(d):
        d["id_types"]["drug"]["stored_forms"] = {"facts.nope": "as_stored"}

    def resolve_via(d):
        d["id_types"]["gene"]["resolve_via"] = ["genes.nope"]

    def retired(d):
        d["id_types"]["gene"]["retired"] = {"listed_in": ["genes.obsolete"]}

    def universe(d):
        d["id_types"]["gene"]["universe"] = {"table": "genes", "keys": ["nope"]}

    for mutate in (canonicalize, crosswalk, stored_forms, resolve_via, retired, universe):
        assert "resolver_columns" in rules(lint(mk(mutate))), mutate.__name__

    def all_ok(d):
        d["id_types"]["gene"]["crosswalks"] = [{"name": "cw", "table": "genes", "from": "id", "to": "symbol"}]
        d["id_types"]["drug"]["stored_forms"] = {"facts.drugId": "as_stored"}
        d["id_types"]["gene"]["maps_to"] = [{"id_type": "drug", "via": "cw"}]

    assert not errors(lint(mk(all_ok)))


def test_rule_edges_and_matrix():
    def edges(d):
        d["tables"]["pairs"] = {"kind": "edges", "grain": "pair", "key": {"columns": ["a", "b"], "nullable": ["b"]},
                                "edge": {"a": "a", "b": "nope"},
                                "columns": {"a": {"role": "endpoint", "side": "a", "id_type": "gene"},
                                            "b": {"role": "endpoint", "side": "b", "id_type": "gene"}}}

    assert "reference" in rules(lint(mk(edges)))

    def matrix(d, key):
        d["tables"]["m"] = {"kind": "matrix", "grain": "cell", "key": {"columns": ["@row.a", "@col.b"]},
                            "matrix": {"axes": {"row": {"name": "r", "from": "index", "key": {"columns": [key]},
                                                        "columns": {"a": {"role": "identifier"}}},
                                                "col": {"name": "c", "from": "header", "key": {"columns": ["b"]},
                                                        "parse": {"pattern": "^(?P<b>\\d+)$",
                                                                  "fields": {"b": {"role": "identifier"}}}}},
                                       "values": {"X": {"role": "measure", "comparable_within": ["release"]}}}}

    assert not errors(lint(mk(lambda d: matrix(d, "a"))))
    assert "key" in rules(lint(mk(lambda d: matrix(d, "zz"))))


# ---------------------------------------------------------------------------
# Overlay lint
# ---------------------------------------------------------------------------

def _overlay(tools: dict[str, Any], server: str = "srv") -> Overlay:
    return Overlay.model_validate({"schema": "vbt.overlay/1", "server": server, "tools": tools})


GOOD_TOOL: dict[str, Any] = {
    "reads": {"src.facts": {"access": "full_table"}},
    "args": {"gene": {"binds": "src.facts.geneId", "accepts": ["gene", "symbol"]},
             "dose": {"binds": "src.facts.dose", "gateway_only": True},
             "limit": {"role": "limit", "min": 1, "max": 100}},
    "result": {"rows": "$.rows", "order": [{"column": "effect", "direction": "asc"}], "echo": {"gene": "$.gene"}},
    "defects": [{"id": "X-1", "what": "head before sort", "where": "x_mcp/tools.py:10,12"}],
}


def olint(tool: dict[str, Any], *, sources: dict[str, SourceDescriptor] | None = None, registry=None,
          catalog=None) -> list[Finding]:
    srcs = sources or {"src": model(mk())}
    return lint_overlay(_overlay({"t": tool}), catalog if catalog is not None else srcs, registry)


def test_overlay_good_tool_clean():
    assert not olint(GOOD_TOOL)


def mut(**changes) -> dict[str, Any]:
    tool = copy.deepcopy(GOOD_TOOL)
    for path, value in changes.items():
        cur = tool
        keys = path.split("__")
        for k in keys[:-1]:
            cur = cur[k]
        if value is None:
            cur.pop(keys[-1], None)
        else:
            cur[keys[-1]] = value
    return tool


def test_overlay_binding_rules():
    cases = {
        "unknown column": mut(args__gene__binds="src.facts.nope"),
        "unknown table": mut(reads={"src.ghost": {}}),
        "filter without binds": mut(args__x={"role": "filter"}),
        "require_any names": mut(require_any=[["gene", "ghost"]]),
        "derived without spec": mut(serve="derived"),
        "block without alternative": mut(serve="block", block={"reason": "wrong answers"}),
        "rows_of not an item table": mut(result__rows_of="src.facts"),
        "order column": mut(result__order=[{"column": "nope"}]),
        "field column": mut(result__fields={"x": {"column": "nope"}}),
        "order_from_arg": mut(result__order_from_arg="gene"),
        "dict binds without selector": mut(args__gene__binds={"a": "src.facts.geneId"}),
    }
    for label, tool in cases.items():
        assert "binding" in rules(olint(tool)), label
    assert not errors(olint(mut(serve="block", block={"reason": "r", "alternatives": ["mcp__data__find"]})))
    assert not errors(olint(mut(serve="block", block={"reason": "r", "hidden": True})))
    assert not errors(olint(mut(serve="derived", derived={"verb": "find", "table": "src.facts"})))
    assert not errors(olint(mut(result__rows_of="src.genes_go", result__order=[])))


def test_overlay_accepts_rules():
    assert "accepts" in rules(olint(mut(args__gene__accepts=["ghost"])))
    # a resolvable kind that reaches the bound kind only through a declared edge
    assert "accepts" in rules(olint(mut(args__gene__accepts=["gene", "drug"])))
    assert not errors(olint(mut(args__gene__accepts=["symbol"])))           # label_of gene

    other = model({"schema": "vbt.datasource/1", "source": "other", "title": "o", "release": {"from": "literal"},
                   "id_types": {"symbol": {"plugin": "symbol_plugin", "universe": "t.id"}},
                   "tables": {"t": {"kind": "entity", "grain": "g", "key": {"columns": ["id"]},
                                    "columns": {"id": {"role": "identifier", "self": True}}}}})
    srcs = {"src": model(mk()), "other": other}
    # bound source first: "symbol" means src:symbol, no ambiguity
    assert not errors(olint(GOOD_TOOL, sources=srcs))
    three = dict(srcs, third=model({**other.model_dump(by_alias=True), "source": "third"}))
    tool = mut(args__gene__binds="src.facts.drugId", args__gene__accepts=["drug", "symbol"])
    # "symbol" is not in the bound source (drugs are) and two others declare it with different universes
    msgs = [f.message for f in errors(olint(tool, sources={**three, "src": model(mk(
        lambda d: d["id_types"].pop("symbol") and d["tables"]["genes"]["columns"]["symbol"].pop("id_type")))}))]
    assert any("qualify it" in m for m in msgs)


def test_overlay_flag_codes_echo_defects():
    def flags(d):
        d["tables"]["facts"]["columns"]["safety"] = {"role": "flag", "encoding": {-1: False, 1: True}}

    srcs = {"src": model(mk(flags))}
    ne = mut(args__safe={"binds": "src.facts.safety", "role": "flag", "when_true": {"ne": 1}})
    assert "flag_codes" in rules(olint(ne, sources=srcs))
    pos = mut(args__safe={"binds": "src.facts.safety", "role": "flag", "when_true": {"in": [-1]}})
    assert not errors(olint(pos, sources=srcs))
    assert "echo" in rules(olint(mut(result__echo={"gene": {"path": "$.gene", "source": "request"}})))
    assert "echo" in rules(olint(mut(result__echo={"ghost": "$.x"})))
    assert "defect" in rules(olint(mut(defects=[{"id": "X", "what": "w", "where": "somewhere in tools.py"}])))
    assert DEFECT_WHERE.match("drug_mcp/tools.py:583-588") and DEFECT_WHERE.match("a/b.py:1,2,30-31")
    assert not DEFECT_WHERE.match("drug_mcp/tools.py") and not DEFECT_WHERE.match("tools.py:12a")


def test_overlay_scope_binding_rule():
    assert "scope_binding" in rules(olint(mut(args__dose={"binds": "src.facts.dose", "op": "ge"})))
    assert not errors(olint(mut(args__dose=None)))      # unbound: the auto-derived gateway-only argument fixes it
    assert not errors(olint(mut(args__dose={"binds": "src.facts.dose", "op": "in"})))


def test_overlay_digits_kind_rule():
    def digits(d):
        d["id_types"]["num"] = {"plugin": "digits"}
        d["tables"]["genes"]["columns"]["num"] = {"role": "identifier", "id_type": "num"}

    def digits_prefixed(d):
        digits(d)
        d["id_types"]["num"]["options"] = {"input_requires_prefix": True}

    reg = _registry(_Digits, _Ident)
    tool = mut(args__gene__accepts=["gene", "num"])
    assert "digits_kind" in rules(olint(tool, sources={"src": model(mk(digits))}, registry=reg))
    assert "digits_kind" not in rules(olint(tool, sources={"src": model(mk(digits_prefixed))}, registry=reg))
    assert "digits_kind" not in rules(olint(tool, sources={"src": model(mk(digits))}))   # needs the registry


def test_overlay_same_as():
    class Cat:
        def __init__(self, sources, overlays):
            self.sources, self.overlays = sources, overlays

    srcs = {"src": model(mk())}
    target = _overlay({"get_pgx": copy.deepcopy(GOOD_TOOL)}, server="target")
    cat = Cat(srcs, {"target": target})
    clash = mut(same_as=["target.get_pgx"])
    assert "binding" in rules(olint(clash, catalog=cat))
    assert not errors(olint(mut(same_as=["target.other_tool"]), catalog=cat))
    assert "binding" in rules(olint(mut(same_as=["not-a-tool-name"]), catalog=cat))
    assert warnings(olint(mut(same_as=["unknown.tool"]), catalog=cat))
