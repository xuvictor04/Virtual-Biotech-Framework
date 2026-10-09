"""The shipped Open Targets descriptor against the real 25.09 release (docs/DATA_LAYER.md §23).

``tests/datalayer/real/ot_25_09/`` holds what the release showed on 2026-10-08
(https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.09/output/): one JSON per table with the
Parquet footer of every shard aggregated (schema, rows, row groups, bytes, per-leaf Arrow type, null
count and min/max), read with HTTP range requests through the harness's own
:func:`~vbt.datalayer.plugins.layouts.http_range.read_footer` and
:func:`~vbt.datalayer.plugins.formats.parquet.footer_stats` (no row group transferred), plus
``values.json``, the value-level facts measured on the tables small enough to download.

* Offline (always): every physical column and nested field of the release has a role and every
  declared non-optional one exists; the Arrow types fit the roles; flat key parts have no footer
  nulls; declared scales hold on the footer min/max; the evidence partitions are the declared
  vocabulary; verified vocabularies, codes, keys and item keys equal what was measured; and every
  known drift that needs a change outside the descriptor is still drift (fixing one means removing
  it from :data:`KNOWN_DRIFT`). Remote footers and the parquet plugin's remote reads are also checked
  against a local Range-capable server.
* ``VBT_DL_NETWORK=1``: the live listing and the first and last shard footers of every table match
  the snapshots (``VBT_DL_NETWORK=full``: every shard; with ``VBT_UPDATE_REAL_SNAPSHOT=1`` the table
  snapshots are rewritten from the live footers).
* ``VBT_DL_REAL_DATA=<dir>`` (the shared real-data root holding ``open_targets/25.09``, or that ``25.09``
  output directory itself): the local copy of every table present there has the recorded schema and rows, and the
  code facts hold on its values.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from dl_upstream import real_ot_dir  # noqa: E402
from netgate import network_mode  # noqa: E402
from vbt.datalayer.descriptor.columns import is_container  # noqa: E402
from vbt.datalayer.descriptor.load import load_descriptors  # noqa: E402
from vbt.datalayer.plugins.base import Fragment  # noqa: E402
from vbt.datalayer.plugins.registry import discover  # noqa: E402
from vbt.datalayer.roles import arrow_compatible  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SNAP = REPO / "tests" / "datalayer" / "real" / "ot_25_09"
BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.09/output/"
RETRIEVED = "2026-10-08"
NETWORK = network_mode()
REAL = real_ot_dir()
UPDATE = os.environ.get("VBT_UPDATE_REAL_SNAPSHOT", "") == "1"
_TEXT = 40                                             # recorded string bounds are cut to this length

#: Physical nested fields the descriptor leaves undeclared on purpose: {table: {path: why}}.
KNOWN_UNDECLARED: dict[str, dict[str, str]] = {}

#: Facts the release refutes whose correction needs files other than the descriptor (contract requests).
#: Each entry is asserted to be still drift, so a fix must remove it here.
KNOWN_DRIFT = {
    "ontology.leaf": "false for every disease term; the constraint stays verified: false with on_refute: drop_field",
}
#: Item keys the release refutes whose correction needs files other than the descriptor (none left: the
#: six-part target.go key with a nullable ecoId replaced the five-part one together with the CT-3 oracle).
REFUTED_ITEM_KEYS: dict[str, str] = {}


# ---------------------------------------------------------------------------- snapshot form


def type_tree(t: pa.DataType) -> Any:
    if pa.types.is_struct(t):
        return {"struct": [field_tree(f) for f in t]}
    if pa.types.is_large_list(t):
        return {"large_list": field_tree(t.value_field)}
    if pa.types.is_list(t):
        return {"list": field_tree(t.value_field)}
    if pa.types.is_map(t):
        return {"map": [field_tree(t.key_field), field_tree(t.item_field)]}
    return str(t)


def field_tree(f: pa.Field) -> dict[str, Any]:
    return {"name": f.name, "type": type_tree(f.type), "nullable": f.nullable}


def to_type(tree: Any) -> pa.DataType:
    if isinstance(tree, str):
        return pa.type_for_alias(tree)
    if "struct" in tree:
        return pa.struct([to_field(f) for f in tree["struct"]])
    if "large_list" in tree:
        return pa.large_list(to_field(tree["large_list"]))
    if "list" in tree:
        return pa.list_(to_field(tree["list"]))
    key, item = tree["map"]
    return pa.map_(to_field(key), to_field(item))


def to_field(f: dict[str, Any]) -> pa.Field:
    return pa.field(f["name"], to_type(f["type"]), nullable=f["nullable"])


def to_schema(tree: list[dict[str, Any]]) -> pa.Schema:
    return pa.schema([to_field(f) for f in tree])


def _plain(v: Any) -> Any:
    if isinstance(v, float) and v != v:
        return "NaN"
    if v is None or isinstance(v, (bool, int, float)):
        return v
    text = v.isoformat() if isinstance(v, (datetime.date, datetime.datetime)) else str(v)
    return text[:_TEXT]


def footer_record(url: str, size: int | None = None) -> dict[str, Any]:
    """One remote shard's footer: schema tree, rows, row groups, writer and per-leaf statistics."""
    from vbt.datalayer.plugins.formats.parquet import footer_stats
    from vbt.datalayer.plugins.layouts.http_range import read_footer

    schema, md = read_footer(url, size=size)
    st = footer_stats(schema, md)
    leaves = {leaf: {"type": c.storage_type, "nulls": c.null_count, "values": c.num_values, "min": c.min, "max": c.max}
              for leaf, c in st.columns.items()}
    return {"rows": st.rows, "row_groups": st.row_groups, "created_by": md.created_by,
            "schema": [field_tree(f) for f in schema], "leaves": leaves}


def merge(table: str, files: list[dict[str, Any]], other: list[str]) -> dict[str, Any]:
    """The table snapshot from per-shard records (``path``, ``size``, ``rows``, ``row_groups``,
    ``created_by``, ``schema``, ``leaves``), in path order."""
    files = sorted(files, key=lambda f: f["path"])
    schemas = {json.dumps(f["schema"]) for f in files}
    leaves: dict[str, dict[str, Any]] = {}
    for f in files:
        for leaf, s in f["leaves"].items():
            a = leaves.setdefault(leaf, {"type": s["type"], "nulls": 0, "values": 0, "min": None, "max": None})
            a["nulls"] = None if a["nulls"] is None or s["nulls"] is None else a["nulls"] + s["nulls"]
            a["values"] += s["values"]
            for k, better in (("min", lambda x, y: x < y), ("max", lambda x, y: x > y)):
                v = s[k]
                if v is None or (isinstance(v, float) and v != v):
                    continue
                try:
                    if a[k] is None or better(v, a[k]):
                        a[k] = v
                except TypeError:
                    pass
    snap: dict[str, Any] = {
        "table": table, "release": "25.09", "source": BASE + table + "/", "retrieved": RETRIEVED,
        "method": "Parquet footer of every shard over HTTP range requests (vbt http_range.read_footer + "
                  "parquet.footer_stats); no row group transferred",
        "shards": len(files), "other_files": sorted(other), "rows": sum(f["rows"] for f in files),
        "row_groups": sum(f["row_groups"] for f in files), "bytes": sum(f.get("size") or 0 for f in files),
        "created_by": sorted({str(f["created_by"]).split(" (build")[0] for f in files}),
        "schema_variants": len(schemas), "file_names_sha256": hashlib.sha256(
            "\n".join(f["path"] for f in files).encode()).hexdigest(),
        "shard_rows": [f["rows"] for f in files], "schema": files[0]["schema"] if files else [],
        "leaves": {k: [v["type"], v["nulls"], v["values"], _plain(v["min"]), _plain(v["max"])] for k, v in leaves.items()},
    }
    parts: dict[str, dict[str, Any]] = {}
    for f in files:
        if "/" not in f["path"]:
            continue
        p = parts.setdefault(f["path"].split("/")[0], {"shards": 0, "rows": 0, "bytes": 0, "filled": set(), "mirror": set()})
        p["shards"] += 1
        p["rows"] += f["rows"]
        p["bytes"] += f.get("size") or 0
        p["filled"] |= {leaf for leaf, s in f["leaves"].items() if s["values"] - (s["nulls"] or 0) > 0}
        ds_leaf = f["leaves"].get("datasourceId")
        if ds_leaf is not None and f["rows"]:
            p["mirror"] |= {ds_leaf["min"], ds_leaf["max"]}
    if parts:
        snap["partitions"] = {k: {"shards": v["shards"], "rows": v["rows"], "bytes": v["bytes"],
                                  "datasourceId": sorted(str(x) for x in v["mirror"]), "filled": sorted(v["filled"])}
                              for k, v in sorted(parts.items())}
    return snap


def listing(table: str, client: Any = None) -> list[tuple[str, str]]:
    """``(relative path, url)`` of every file under the table's directory (hive partitions walked)."""
    import httpx

    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=120)
    href = re.compile(r'href="([^"?/][^"]*)"')
    out, stack = [], [""]
    try:
        while stack:
            rel = stack.pop()
            r = client.get(BASE + table + "/" + rel)
            r.raise_for_status()
            for h in href.findall(r.text):
                if h.startswith("/"):
                    continue
                if h.endswith("/"):
                    stack.append(rel + h)
                else:
                    out.append((rel + h, BASE + table + "/" + rel + h))
    finally:
        if own:
            client.close()
    return sorted(out)


# ---------------------------------------------------------------------------- fixtures and helpers


@pytest.fixture(scope="module")
def ot() -> Any:
    return load_descriptors(REPO / "configs" / "data" / "sources", {"project_root": str(REPO)})["open_targets"]


@pytest.fixture(scope="module")
def snaps() -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(SNAP.glob("*.json")):
        if not p.name.startswith("_") and p.name != "values.json":
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


@pytest.fixture(scope="module")
def values() -> dict[str, Any]:
    return json.loads((SNAP / "values.json").read_text(encoding="utf-8"))


def physical_tables(ot: Any) -> dict[str, Any]:
    return {spec.path: spec for spec in ot.tables.values() if spec.items_of is None}


def walk(columns: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    for name, col in columns.items():
        path = f"{prefix}.{name}" if prefix else name
        yield path, col
        if is_container(col):
            yield from walk(col.fields, path)


def physical_paths(schema: pa.Schema) -> list[str]:
    out: list[str] = []

    def rec(t: pa.DataType, prefix: str) -> None:
        while pa.types.is_list(t) or pa.types.is_large_list(t):
            t = t.value_type
        if pa.types.is_struct(t):
            for f in t:
                out.append(f"{prefix}.{f.name}")
                rec(f.type, f"{prefix}.{f.name}")

    for f in schema:
        out.append(f.name)
        rec(f.type, f.name)
    return out


def type_at(schema: pa.Schema, path: str) -> pa.DataType | None:
    names = path.split(".")
    if names[0] not in schema.names:
        return None
    t = schema.field(names[0]).type
    for name in names[1:]:
        while pa.types.is_list(t) or pa.types.is_large_list(t):
            t = t.value_type
        if not pa.types.is_struct(t) or t.get_field_index(name) < 0:
            return None
        t = t.field(t.get_field_index(name)).type
    return t


def dotted(leaf: str) -> str:
    """A footer leaf path as a declared dotted path (``go.list.element.aspect`` -> ``go.aspect``)."""
    return re.sub(r"\.(list\.element|list\.item|element|array)(?=\.|$)", "", leaf)


def leaf_stats(snap: dict[str, Any]) -> dict[str, list[Any]]:
    return {dotted(k): v for k, v in snap["leaves"].items()}


def unroled(spec: Any, schema: pa.Schema) -> tuple[list[str], list[str]]:
    """``(physical paths without a role, declared non-optional paths absent from the data)``."""
    declared = dict(walk(spec.columns))

    def covered(path: str) -> bool:
        parts = path.split(".")
        return any((c := declared.get(".".join(parts[:i]))) is not None and not is_container(c)
                   for i in range(1, len(parts)))

    def optional(path: str) -> bool:
        parts = path.split(".")
        return any(getattr(declared.get(".".join(parts[:i])), "optional", False) for i in range(1, len(parts) + 1))

    physical = physical_paths(schema)
    missing = [p for p in physical if p not in declared and not covered(p)]
    absent = [p for p in declared if p not in physical and p not in spec.partitions and not optional(p) and not covered(p)]
    return missing, absent


# ---------------------------------------------------------------------------- offline: descriptor vs snapshots


def test_snapshots_cover_every_table_of_the_release(ot, snaps) -> None:
    release = json.loads((SNAP / "_release.json").read_text(encoding="utf-8"))
    assert set(snaps) == set(physical_tables(ot)) == set(release["tables"]) and len(snaps) == 38
    for name, snap in snaps.items():
        assert snap["source"] == BASE + name + "/" and snap["retrieved"] == RETRIEVED
        assert snap["shards"] == len(snap["shard_rows"]) > 0 and snap["rows"] == sum(snap["shard_rows"])
        assert snap["schema_variants"] == 1, f"{name}: shards disagree on the schema"
        assert release["tables"][name] == {"shards": snap["shards"], "rows": snap["rows"], "bytes": snap["bytes"]}
    assert sum(s["shards"] for s in snaps.values()) == release["parquet_shards"] == 3508


def test_every_real_column_and_field_has_a_role(ot, snaps) -> None:
    problems = []
    for path, spec in physical_tables(ot).items():
        missing, absent = unroled(spec, to_schema(snaps[path]["schema"]))
        known = KNOWN_UNDECLARED.get(path, {})
        problems += [f"{path}.{p}: no role" for p in missing if p not in known]
        problems += [f"{path}.{p}: declared, not in the 25.09 files" for p in absent]
        problems += [f"{path}.{p}: KNOWN_UNDECLARED but declared or gone" for p in known if p not in missing]
    assert not problems, "\n".join(problems)


def test_real_arrow_types_fit_the_roles(ot, snaps) -> None:
    problems = []
    for path, spec in physical_tables(ot).items():
        schema = to_schema(snaps[path]["schema"])
        for col_path, col in walk(spec.columns):
            t = type_at(schema, col_path)
            role = getattr(col, "role", None)
            if t is None or role is None:
                continue
            if not arrow_compatible(role, str(t), parse=getattr(col, "parse", None),
                                    stored_as=getattr(col, "stored_as", None), encoding=getattr(col, "encoding", None),
                                    list_delimiter=getattr(col, "list_delimiter", None)):
                problems.append(f"{path}.{col_path}: {t} does not fit role {role}")
    assert not problems, "\n".join(problems)


def test_flat_key_parts_have_no_footer_nulls(ot, snaps) -> None:
    problems = []
    for path, spec in physical_tables(ot).items():
        if spec.key.check == "none":
            continue
        stats = leaf_stats(snaps[path])
        for part in spec.key.columns:
            if part in spec.key.nullable or part in spec.partitions or part not in stats:
                continue
            nulls = stats[part][1]
            if nulls:
                problems.append(f"{path}.{part}: {nulls} null(s) in a non-nullable key part")
    assert not problems, "\n".join(problems)


def test_declared_scales_hold_on_the_footer_bounds(ot, snaps) -> None:
    from vbt.datalayer.plugins.statistics import spec_get

    reg = discover(entry_points=False)
    problems, checked = [], 0
    for path, spec in physical_tables(ot).items():
        stats = leaf_stats(snaps[path])
        for col_path, col in walk(spec.columns):
            scale = getattr(col, "scale", None)
            if getattr(col, "role", None) != "measure" or not scale or col_path not in stats:
                continue
            _t, _n, _v, lo, hi = stats[col_path]
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                continue
            plugin = reg.get("statistic", col.statistic)
            codes = {float(c) for c in plugin.missing_codes(col) if isinstance(c, (int, float))}
            codes |= {float(c) for c in (spec_get(col, "missing_values", []) or []) if isinstance(c, (int, float))}
            checked += 1
            low_ok = lo >= float(scale[0]) - 1e-9 or float(lo) in codes
            if not (low_ok and hi <= float(scale[1]) + 1e-9):
                problems.append(f"{path}.{col_path}: footer range [{lo}, {hi}] outside the scale {scale}")
    assert not problems, "\n".join(problems)
    assert checked >= 30


def test_evidence_partitions_are_the_declared_sources(ot, snaps) -> None:
    ev = physical_tables(ot)["evidence"]
    part = ev.partitions["sourceId"]
    parts = snaps["evidence"]["partitions"]
    assert {k.split("=", 1)[1] for k in parts} == set(part.column.vocab) and len(parts) == 23
    assert all(k.startswith("sourceId=") for k in parts)
    for label, p in parts.items():
        assert p["datasourceId"] == [label.split("=", 1)[1]], f"{label}: datasourceId does not mirror the partition"
    filled = {k.split("=", 1)[1]: set(p["filled"]) for k, p in parts.items()}
    # applies_when facets: a source-specific field is filled only where the descriptor says
    for col_path, col in walk(ev.columns):
        sources = (getattr(col, "applies_when", None) or {}).get("sourceId")
        if not sources:
            continue
        leaf_sources = {s for s, leaves in filled.items()
                        if any(dotted(x) == col_path or dotted(x).startswith(col_path + ".") for x in leaves)}
        assert leaf_sources <= set(sources), f"evidence.{col_path} is filled in {sorted(leaf_sources - set(sources))}"


def test_measured_values_match_the_verified_facts(ot, values) -> None:
    tables = ot.tables
    # vocabularies: a verified vocab is exactly the set of stored values
    for ref, counts in values["vocab"].items():
        table, _, path = ref.partition(".")
        col = dict(walk(tables[table].columns)).get(path)
        if col is None or not isinstance(getattr(col, "vocab", None), list):
            continue
        assert getattr(col, "verified", True), f"{ref}: measured, still verified: false"
        assert set(map(str, col.vocab)) == set(counts), f"{ref}: vocab {col.vocab} vs stored {sorted(counts)}"
    # codes: every stored code is a declared code and every declared code is stored
    for ref, counts in values["codes"].items():
        table, _, path = ref.partition(".")
        col = dict(walk(tables[table].columns))[path]
        stored = {k for k in counts if k != "null"}
        if getattr(col, "encoding", None):
            declared = {str(k) for k, label in col.encoding.items() if label is not None}
            unknown = {str(k) for k, label in col.encoding.items() if label is None}
        elif getattr(col, "statistic", None) == "binary_factor":
            declared, unknown = {"0", "1"}, set()
        else:
            continue
        if ref in values["refuted"]:
            assert stored != declared, f"{ref}: listed as refuted but the codes now match"
            continue
        assert getattr(col, "verified", True), f"{ref}: measured, still verified: false"
        assert stored - unknown == declared, f"{ref}: declared codes {sorted(declared)} vs stored {sorted(stored)}"
    # keys and item keys: the descriptor declares the column set that was measured unique
    for ref, fact in values["item_keys"].items():
        table, _, path = ref.partition(".")
        col = dict(walk(tables[table].columns))[path]
        assert fact["unique_duplicates"] == 0
        if ref in REFUTED_ITEM_KEYS:   # still the former key: measured not unique
            assert list(col.item_key.columns) == fact["former"] != fact["unique"], REFUTED_ITEM_KEYS[ref]
            assert fact["former_duplicate_items"] > 0, ref
            continue
        assert list(col.item_key.columns) == fact["unique"], f"{ref}: item_key {col.item_key.columns}"
        # parts measured null in some items are declared nullable (NULLS NOT DISTINCT), and only those
        assert set(col.item_key.nullable) == set(fact.get("null_parts") or ()), f"{ref}: nullable {col.item_key.nullable}"
    for table, fact in values["keys"].items():
        key = tables[table].key
        assert list(key.columns) == fact["columns"], f"{table}: key {key.columns} vs measured {fact['columns']}"
        if fact.get("content_duplicates") is not None:
            assert key.row_identity == "content_hash" and fact["content_duplicates"] == 0 and key.verified
        else:
            assert fact["duplicates"] == 0 and key.verified, table
    # identities the release refutes (identical rows exist): the rows have none, copies are counted as stored
    for table, fact in values["refuted_keys"].items():
        assert fact["content_duplicates"] > 0 and tables[table].key.row_identity == "none", table


def test_known_drift_is_still_drift(ot, values) -> None:
    leaf = [c for c in ot.tables["disease"].constraints if c.column == "ontology.leaf"]
    assert leaf and leaf[0].verified is False and values["ids"]["disease.ontology.leaf"] == {"False": 39530}, \
        KNOWN_DRIFT["ontology.leaf"]


def test_so_ids_are_canonical_with_a_colon(ot, values) -> None:
    """25.09 so.id is SO:NNNNNNN (2,611 terms); variant.mostSevereConsequenceId stores SO_NNNNNNN as a stored form."""
    so = ot.id_types["so_term"]
    assert "separator" not in (so.options or {}) and values["ids"]["so.id"] == {"SO:": 2611}
    assert so.stored_forms == {"variant.mostSevereConsequenceId": "as_stored"}
    col = ot.tables["variant"].columns["mostSevereConsequenceId"]
    assert col.id_type == "so_term" and col.form == "as_stored"


def test_universe_filters_keep_the_canonical_ids(ot, values) -> None:
    """The universes whose column also holds other ids filter them with ``where`` (readiness R4b samples
    what the filter keeps): disease_hpo.id holds 12,081 non-HP terms, target.proteinIds[].id 112,428 ENSP ids."""
    hpo = ot.id_types["hpo"]
    assert hpo.universe.where == {"text": ["id", "HP_", "substring"]} and set(values["ids"]["disease_hpo.id"]) - {"HP"}
    uni = ot.id_types["uniprot_accession"]
    sources = values["ids"]["target.proteinIds.source"]
    canonical = values["ids"]["target.proteinIds.uniprot_canonical"]
    kept = set(uni.universe.where["in"][1])
    assert kept < set(sources) and all(canonical.get(s) == sources[s] for s in kept) and "ensembl_PRO" not in canonical


def test_item_tables_with_null_item_key_parts_are_checked(ot, values) -> None:
    """Item keys unique only with NULLS NOT DISTINCT declare their null parts, so the item tables keep a key check."""
    for table, container in (("target_chemical_probes", "target.chemicalProbes"),
                             ("target_safety_liabilities", "target.safetyLiabilities"),
                             ("target_essentiality_screens", "target_essentiality.geneEssentiality.depMapEssentiality")):
        nullable = set(values["item_keys"][container]["null_parts"])
        assert nullable and nullable <= set(values["item_keys"][container]["unique"]), table
        assert ot.tables[table].key.check == "sampled", table


def test_interaction_rows_are_stored_in_both_orientations(ot, values) -> None:
    for table in ("interaction", "interaction_evidence"):
        edge = ot.tables[table].edge
        seen = values["orientation"][table]
        assert edge.orientation == "both" and not edge.directed and not edge.directed_when and edge.verified, table
        assert seen["ordered_pairs"] == seen["with_reverse"] and set(seen["ordered_pairs"]) == {
            "intact", "reactome", "signor", "string"}, table


def test_every_interaction_evidence_record_has_its_interaction_row(ot, values, snaps) -> None:
    fact = values["refs"]["interaction_evidence.intA"]
    ref = ot.tables["interaction_evidence"].columns["intA"].ref
    assert ref.table == fact["table"] and sorted(ref.on) == sorted(fact["on"])
    assert fact["rows"] == snaps["interaction_evidence"]["rows"] and fact["target_rows"] == snaps["interaction"]["rows"]
    assert fact["matched"] == fact["rows"] and not fact["unmatched_by_source"]


def test_literature_sample_pmid_counts_are_labelled_as_measured(values) -> None:
    """The literature comment quotes pmids seen in two or more sampled shards, not shard-pair co-occurrences."""
    fact = values["samples"]["literature"]
    by_k = {int(k): n for k, n in fact["pmids_by_shard_count"].items()}
    assert sum(by_k.values()) == fact["pmids"] and set(by_k) <= set(range(1, fact["shards"] + 1))
    assert sum(n for k, n in by_k.items() if k >= 2) == fact["pmids_in_two_or_more_sample_shards"]
    assert sum(n * k * (k - 1) // 2 for k, n in by_k.items()) == fact["pmid_shard_pair_cooccurrences"]
    text = (REPO / "configs" / "data" / "sources" / "open_targets.yaml").read_text(encoding="utf-8")
    two = f"{fact['pmids_in_two_or_more_sample_shards']:,} of the {fact['pmids']:,} pmids"
    assert two in text and f"{fact['pmid_shard_pair_cooccurrences']:,}" not in text
    assert 'every shard spans "100"' not in text


@pytest.mark.skipif(REAL is None, reason="set VBT_DL_REAL_DATA=<dir> to check the real files")
def test_real_literature_sample_and_footers_hold(values) -> None:
    import collections
    import pyarrow.compute as pc

    fact = values["samples"]["literature"]
    shards = sorted((REAL / "_samples" / "literature").glob("*.parquet"))
    if len(shards) != fact["shards"]:
        pytest.skip(f"{REAL / '_samples' / 'literature'} does not hold the {fact['shards']} sampled shards")
    seen: collections.Counter[str] = collections.Counter()
    rows = 0
    for path in shards:
        col = pq.read_table(path, columns=["pmid"])["pmid"]
        rows += len(col)
        seen.update(pc.unique(col).to_pylist())
    by_k = collections.Counter(seen.values())
    assert rows == fact["rows"] and len(seen) == fact["pmids"]
    assert {str(k): n for k, n in sorted(by_k.items())} == fact["pmids_by_shard_count"]
    footers = REAL / "_footers" / "literature.json"
    if not footers.exists():
        pytest.skip(f"{footers} is not there")
    files = json.loads(footers.read_text(encoding="utf-8"))["files"]
    bounds = [(f["leaves"]["pmid"]["min"], f["leaves"]["pmid"]["max"]) for f in files]
    assert len(bounds) == 334
    # every shard's range holds numeric PMIDs, so statistics on pmid prune no shard
    assert all(lo <= p <= hi for lo, hi in bounds for p in ("2", "9", "28304224", "39000000"))
    assert sum(hi.startswith("PPR") for _lo, hi in bounds) == 326 and sum(hi.startswith("c8") for _lo, hi in bounds) == 8


# ---------------------------------------------------------------------------- offline: remote footers through the plugins


class _RangeServer:
    """A local Range-capable HTTP server over a dict of path -> bytes."""

    def __init__(self, files: dict[str, bytes]) -> None:
        outer = self
        self.files = files
        self.bytes = 0

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, head: bool) -> None:
                body = outer.files.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                if rng:
                    a, b = rng.split("=")[1].split("-")
                    lo, hi = int(a), min(int(b), len(body) - 1)
                    part = body[lo:hi + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(body)}")
                else:
                    part = body
                    self.send_response(200)
                self.send_header("Content-Length", str(len(part)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if not head:
                    outer.bytes += len(part)
                    self.wfile.write(part)

            def do_HEAD(self) -> None:
                self._send(True)

            def do_GET(self) -> None:
                self._send(False)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def no_proxy(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_remote_footers_report_what_local_footers_report(tmp_path, no_proxy) -> None:
    """The release's shapes (large_string, list<struct>, int32 codes with nulls) over HTTP ranges: the
    layout's footer statistics and the parquet plugin's remote reads equal the local file's."""
    from vbt.datalayer.plugins.layouts.http_range import footer_stats

    n = 50_000
    tbl = pa.table({
        "id": pa.array([f"SO:{i:07d}" for i in range(n)], pa.large_string()),
        "code": pa.array([(-1 if i % 7 == 0 else None) for i in range(n)], pa.int32()),
        "go": pa.array([[{"id": f"GO:{i:07d}", "aspect": "CFP"[i % 3]}] if i % 2 else None for i in range(n)],
                       pa.list_(pa.struct([("id", pa.string()), ("aspect", pa.string())]))),
        "score": pa.array([i / n for i in range(n)], pa.float64()),
    })
    local = tmp_path / "t.parquet"
    pq.write_table(tbl, local, row_group_size=20_000)
    blob = local.read_bytes()
    server = _RangeServer({"/t.parquet": blob})
    try:
        reg = discover(entry_points=False)
        fmt = reg.get("format", "parquet")
        remote = Fragment(uri=f"{server.base}/t.parquet", size=None, mtime_ns=None)
        here = Fragment(uri=str(local), size=len(blob), mtime_ns=None)
        before = server.bytes
        got = footer_stats(remote)
        assert server.bytes - before < len(blob) / 4, "only the footer is transferred"
        want = fmt.stats(here)
        assert (got.rows, got.row_groups) == (want.rows, want.row_groups) == (n, 3)
        assert got.columns == want.columns
        assert got.columns["id"].storage_type == "large_string" and got.columns["id"].min == "SO:0000000"
        assert got.columns["code"].storage_type == "int32" and got.columns["code"].null_count == n - len(range(0, n, 7))
        assert got.columns["code"].min == got.columns["code"].max == -1
        assert got.columns["go.list.element.aspect"].kind == "nested"
        assert fmt.stats(remote).columns == want.columns
        assert fmt.logical_schema(remote) == fmt.logical_schema(here)
        assert fmt.metadata(remote)["num_rows"] == str(n)
        leaves = fmt.read_leaves(remote, ["go[].aspect"], [1])
        assert leaves.equals(fmt.read_leaves(here, ["go[].aspect"], [1]))
    finally:
        server.close()


def test_remote_scans_and_sidecar_footers_read_what_local_ones_read(tmp_path, no_proxy) -> None:
    """Value scans and sidecar footers of an http_range table: ParquetFormat.scan reads the remote row groups
    (pruned and filtered as locally) and the sidecar footer reader parses the remote footer."""
    from vbt.datalayer.predicate import Eq
    from vbt.datalayer.service.sidecar import read_footer

    n = 300_000
    tbl = pa.table({"pmid": pa.array([str(10_000_000 + i) for i in range(n)], pa.large_string()),
                    "code": pa.array([(-1 if i % 7 == 0 else None) for i in range(n)], pa.int32())})
    local = tmp_path / "t.parquet"
    pq.write_table(tbl, local, row_group_size=100_000)
    blob = local.read_bytes()
    server = _RangeServer({"/t.parquet": blob})
    try:
        fmt = discover(entry_points=False).get("format", "parquet")
        remote = Fragment(uri=f"{server.base}/t.parquet", size=None, mtime_ns=None)
        here = Fragment(uri=str(local), size=len(blob), mtime_ns=None)

        def scanned(frag: Fragment, **kw: Any) -> pa.Table:
            batches = list(fmt.scan([frag], columns=["pmid", "code"], predicate=Eq("code", -1), partitions={}, **kw))
            return pa.Table.from_batches(batches) if batches else pa.table({})

        want, got = scanned(here), scanned(remote)
        assert got.num_rows == len(range(0, n, 7)) and got.equals(want)
        assert scanned(remote, row_groups={remote.uri: [2]}).equals(scanned(here, row_groups={here.uri: [2]}))
        before = server.bytes
        rf, lf = read_footer(remote), read_footer(here)
        assert server.bytes - before < len(blob) / 4, "only the footer is transferred"
        assert (rf.leaves, rf.row_groups) == (lf.leaves, lf.row_groups) and len(rf.row_groups) == 3
    finally:
        server.close()


# ---------------------------------------------------------------------------- VBT_DL_NETWORK: live footers


def _compare_live(name: str, snap: dict[str, Any], every: bool) -> list[str]:
    files = [(p, u) for p, u in listing(name) if p.endswith(".parquet")]
    problems = []
    names_sha = hashlib.sha256("\n".join(p for p, _u in files).encode()).hexdigest()
    if len(files) != snap["shards"] or names_sha != snap["file_names_sha256"]:
        return [f"{name}: {len(files)} live shards (names {names_sha[:12]}) vs {snap['shards']} recorded"]
    picks = list(range(len(files))) if every else sorted({0, len(files) - 1})
    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(lambda i: (i, footer_record(files[i][1])), picks))
    for i, rec in recs:
        if rec["schema"] != snap["schema"]:
            problems.append(f"{name}: shard {files[i][0]} has another schema")
        if rec["rows"] != snap["shard_rows"][i]:
            problems.append(f"{name}: shard {files[i][0]} has {rec['rows']} rows, recorded {snap['shard_rows'][i]}")
    return problems


@pytest.mark.skipif(not NETWORK, reason="live Open Targets FTP reads need VBT_DL_NETWORK=1 (or full)")
def test_live_release_matches_the_snapshots(snaps) -> None:
    every = NETWORK == "full"
    if UPDATE and every:
        _rewrite_snapshots(list(snaps))
        return
    problems = []
    for name, snap in sorted(snaps.items()):
        problems += _compare_live(name, snap, every)
    assert not problems, "\n".join(problems)


def _rewrite_snapshots(tables: list[str]) -> None:
    from vbt.datalayer.plugins.layouts.http_range import head

    total = {}
    for name in tables:
        entries = listing(name)
        shards = [(p, u) for p, u in entries if p.endswith(".parquet")]

        def one(e: tuple[str, str]) -> dict[str, Any]:
            size = head(e[1])["size"]
            return {"path": e[0], "size": size, **footer_record(e[1], size=size)}

        with ThreadPoolExecutor(8) as ex:
            files = list(ex.map(one, shards))
        snap = merge(name, files, [p for p, _u in entries if not p.endswith(".parquet")])
        (SNAP / f"{name}.json").write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        total[name] = {"shards": snap["shards"], "rows": snap["rows"], "bytes": snap["bytes"]}
    release = json.loads((SNAP / "_release.json").read_text(encoding="utf-8"))
    release["tables"].update(total)
    release["parquet_shards"] = sum(t["shards"] for t in release["tables"].values())
    (SNAP / "_release.json").write_text(json.dumps(release, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- VBT_DL_REAL_DATA: downloaded files


@pytest.mark.skipif(not REAL, reason="checks on downloaded 25.09 files need VBT_DL_REAL_DATA=<dir>")
def test_downloaded_tables_match_the_snapshots(snaps) -> None:
    root = Path(REAL)                                  # <root>/open_targets/25.09, or the 25.09 directory named
    fmt = discover(entry_points=False).get("format", "parquet")
    checked, problems = [], []
    for name, snap in sorted(snaps.items()):
        files = sorted(p for p in (root / name).rglob("*.parquet")) if (root / name).is_dir() else []
        if len(files) != snap["shards"]:
            continue                                       # absent or partial: nothing to compare
        rows = 0
        for f in files:
            frag = Fragment(uri=str(f), size=f.stat().st_size, mtime_ns=None)
            schema = fmt.logical_schema(frag)
            if [field_tree(x) for x in schema] != snap["schema"]:
                problems.append(f"{name}: {f.name} has another schema")
            rows += fmt.stats(frag).rows
        if rows != snap["rows"]:
            problems.append(f"{name}: {rows} rows, recorded {snap['rows']}")
        checked.append(name)
    if (root / "target_prioritisation").is_dir():
        t = pq.read_table(root / "target_prioritisation", columns=["hasSafetyEvent", "isCancerDriverGene", "hasTEP"])
        expected = {"hasSafetyEvent": {"-1"}, "isCancerDriverGene": {"-1"}, "hasTEP": {"1"}}
        for col, want in expected.items():
            codes = {str(v) for v in t[col].to_pylist() if v is not None}
            assert codes == want, (col, codes)
    if (root / "so").is_dir():
        ids = pq.read_table(root / "so", columns=["id"])["id"].to_pylist()
        assert all(i.startswith("SO:") for i in ids) and len(ids) == snaps["so"]["rows"]
    assert not problems, "\n".join(problems)
    if not checked:
        pytest.skip(f"no complete 25.09 table under {root}")
