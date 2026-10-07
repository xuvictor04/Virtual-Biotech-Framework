"""Phase-2 format, layout and identifier plugins (§9.3, §9.4; F12).

* The conformance suites run for every new plugin (the parametrized F-, L- and I- cases list them).
* ``obo``: the published projection on ``tests/fixtures/mini_cl.obo`` (is_a, part_of relationships,
  obsolete terms, ``data-version``), size-only stats, unreadable files.
* ``gmt``: set_id, description, members; the file-stem fragment key; malformed lines.
* ``csv``/``tsv``: the cached ``stats_scan`` pass (one per fingerprint), quoted empty strings versus
  nulls, gzip, a cut-short file, ``options.column_types``.
* ``jsonl``: types inferred from the whole file (a field null in the first rows), bad lines.
* ``npy``/``safetensors``: ``ids_from`` vocabularies, misaligned vocabularies.
* ``requires``: a missing module is reported per plugin (``plugin_unavailable`` with an install hint).
* ``zip_member`` and ``http_range``: archive members read in place (listing by name, fingerprints from
  the central directory, stat-only signatures, manifests), footer-only Parquet stats over HTTP ranges
  and remote CSV reads through a local Range-capable server.
* Identifiers ``ncbi_gene`` (prefix required for agent input), ``geo_gsm``, ``census_joinid``,
  ``cell_barcode``, ``cell_ontology``/``uberon`` (hierarchy through ``vbt.analysis.ontology``).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from vbt.datalayer.plugins.base import FormatError, Fragment, LayoutSpec, Manifest, Normalized, Rejected  # noqa: E402
from vbt.datalayer.plugins.registry import discover  # noqa: E402
from vbt.datalayer.predicate import Eq  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MINI_CL = REPO / "tests" / "fixtures" / "mini_cl.obo"
NEW_FORMATS = ("csv", "tsv", "jsonl", "h5ad", "zarr", "obo", "gmt", "npy", "safetensors")
NEW_LAYOUTS = ("zip_member", "http_range")
NEW_IDENTIFIERS = ("ncbi_gene", "geo_gsm", "census_joinid", "cell_barcode", "cell_ontology", "uberon")


@pytest.fixture(scope="module")
def reg():
    return discover(entry_points=False)


def frag(path: Path | str, **kw: Any) -> Fragment:
    st = os.stat(path)
    return Fragment(uri=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns, **kw)


def rows_of(plugin: Any, frags: list[Fragment], **kw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for batch in plugin.scan(frags, columns=kw.pop("columns", None), predicate=None, partitions={}, **kw):
        out.extend(plugin.to_native(pa.Table.from_batches([batch])))
    return out


# ---------------------------------------------------------------------------- registration and suites


def test_every_phase2_plugin_is_registered_with_its_capabilities(reg) -> None:
    caps = {name: set(reg.get("format", name).capabilities) for name in NEW_FORMATS}
    assert caps["csv"] == caps["tsv"] == {"tabular", "stats_scan", "matrix"}
    assert caps["jsonl"] == {"tabular", "nested", "stats_scan"}
    assert caps["h5ad"] == caps["zarr"] == {"matrix", "stats"}
    assert caps["obo"] == caps["gmt"] == {"tabular", "nested"}
    assert caps["npy"] == caps["safetensors"] == {"tabular", "vectors"}
    assert set(reg.get("format", "h5ad").requires) == {"h5py", "anndata", "scipy"}
    for name in NEW_LAYOUTS:
        assert reg.has("layout", name)
    for name in NEW_IDENTIFIERS:
        assert reg.get("identifier", name).id_type == name


def test_conformance_suites_run_for_every_new_plugin() -> None:
    """The parametrized suites list each new plugin (no test edits were needed to cover them)."""
    from vbt.datalayer.plugins.conformance import format as fsuite, identifier as isuite, layout as lsuite

    format_ids = {p.id for p in fsuite.GOLDEN_PARAMS} | {p.id for p in fsuite.PROJECTION_PARAMS} | \
        {p.id for p in fsuite._matrix_params()}
    for name in NEW_FORMATS:
        assert any(i == name or i.startswith(name + "-") for i in format_ids), name
    matrix_ids = {p.id.split("-")[0] for p in fsuite._matrix_params()}
    assert {"csv", "tsv", "h5ad", "zarr"} <= matrix_ids
    assert set(NEW_LAYOUTS) <= {p.name for p in lsuite.LAYOUT_PLUGINS}
    assert set(NEW_IDENTIFIERS) <= set(isuite.IDS)


def test_requires_reported_when_a_module_is_missing(reg, monkeypatch) -> None:
    import importlib.util

    real = importlib.util.find_spec

    def find_spec(name: str, *a: Any, **k: Any) -> Any:
        return None if name in ("h5py", "zarr") else real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    report = reg.requires_report()
    assert report["format/h5ad"] == ["h5py"]
    assert report["format/zarr"] == ["zarr"]
    assert "format/csv" not in report and "format/obo" not in report


# ---------------------------------------------------------------------------- obo


def test_obo_projection_of_mini_cl(reg) -> None:
    obo = reg.get("format", "obo")
    f = frag(MINI_CL)
    schema = obo.logical_schema(f)
    assert schema.names == ["id", "name", "namespace", "def", "synonyms", "alt_id", "is_obsolete", "replaced_by",
                            "consider", "is_a", "relationship", "xref", "subset"]
    rows = {r["id"]: r for r in rows_of(obo, [f])}
    assert len(rows) == 14 and "part_of" not in rows                # the [Typedef] stanza is not a term
    lung_fibro = rows["CL:0002553"]
    assert lung_fibro["is_a"] == ["CL:0000057"]                       # the "! fibroblast" comment is dropped
    assert lung_fibro["relationship"] == [{"type": "part_of", "target": "UBERON:0002048"}]
    assert rows["CL:0000000"]["is_a"] == [] and rows["CL:0000000"]["name"] == "cell"
    assert rows["CL:0009999"]["is_obsolete"] is True and rows["CL:0000057"]["is_obsolete"] is False
    assert rows["CL:0000057"]["def"] is None and rows["CL:0000057"]["synonyms"] == []
    meta = obo.metadata(f)
    assert meta["data-version"] == "mini-cl/test" and meta["format-version"] == "1.2" and meta["ontology"] == "cl"
    stats = obo.stats(f)
    assert stats.rows is None and stats.method == "size_only"
    assert sum(c.uncompressed_bytes for c in stats.columns.values()) <= f.size


def test_obo_synonym_scopes_and_unreadable_files(reg, tmp_path) -> None:
    obo = reg.get("format", "obo")
    good = tmp_path / "go.obo"
    good.write_text('format-version: 1.2\ndata-version: releases/2024-06-17\n\n[Term]\nid: GO:0000001\n'
                    'name: mitochondrion inheritance\nsynonym: "mitochondrial inheritance" EXACT []\n'
                    'synonym: "mito \\"inherit\\" ! x" NARROW [GOC:x]\nis_a: GO:0048308 ! organelle inheritance\n'
                    'relationship: part_of GO:0000002 ! x\nalt_id: GO:0019952\nxref: Wikipedia:Mito\n\n'
                    '[Term]\nid: GO:0000005\nname: obsolete x\nis_obsolete: true\nconsider: GO:0042254\n'
                    'consider: GO:0044183\n')
    (row, old) = rows_of(obo, [frag(good)])
    assert row["synonyms"] == [{"text": "mitochondrial inheritance", "scope": "exact"},
                               {"text": 'mito "inherit" ! x', "scope": "narrow"}]
    assert row["alt_id"] == ["GO:0019952"] and row["xref"] == ["Wikipedia:Mito"]
    assert old["is_obsolete"] is True and old["consider"] == ["GO:0042254", "GO:0044183"]
    for name, text in (("noheader.obo", "[Term]\nid: GO:1\n"), ("noid.obo", "format-version: 1.2\n\n[Term]\nname: x\n"),
                       ("cut.obo", "format-version: 1.2\n\n[Term]\nid: GO:0000001\nname: mito"),
                       ("empty.obo", "")):
        bad = tmp_path / name
        bad.write_text(text)
        with pytest.raises(FormatError) as err:
            obo.logical_schema(frag(bad))
        assert err.value.fragment == str(bad)


# ---------------------------------------------------------------------------- gmt


def test_gmt_projection_and_fragment_key(reg, tmp_path) -> None:
    gmt = reg.get("format", "gmt")
    path = tmp_path / "h.all.v2024.1.Hs.symbols.gmt"
    path.write_text("HALLMARK_A\thttp://x/A\tTP53\tMDM2\n\nHALLMARK_B\thttp://x/B\n")
    f = frag(path)
    assert gmt.logical_schema(f).names == ["set_id", "description", "members"]
    assert rows_of(gmt, [f]) == [{"set_id": "HALLMARK_A", "description": "http://x/A", "members": ["TP53", "MDM2"]},
                                 {"set_id": "HALLMARK_B", "description": "http://x/B", "members": []}]
    assert gmt.metadata(f)["fragment_key"] == "h.all.v2024.1.Hs.symbols"
    assert gmt.metadata(frag(path, fragment_key="hallmark"))["fragment_key"] == "hallmark"
    assert gmt.stats(f).rows == 2
    bad = tmp_path / "bad.gmt"
    bad.write_text("ONLY_A_NAME\n")
    with pytest.raises(FormatError):
        gmt.logical_schema(frag(bad))


# ---------------------------------------------------------------------------- csv / tsv


def test_csv_stats_scan_is_one_pass_per_fingerprint(reg, tmp_path) -> None:
    from vbt.datalayer.plugins.formats.csv import STATS_CACHE

    csv = reg.get("format", "csv")
    path = tmp_path / "t.csv"
    path.write_text('id,score,name\na,1.5,"x"\nb,,""\nc,nan,\n')
    f = frag(path)
    st = csv.stats(f)
    assert st.method == "scan" and st.rows == 3
    assert st.columns["score"].null_count == 1 and st.columns["score"].min == 1.5    # NaN never a min/max
    assert st.columns["name"].null_count == 1                                     # quoted "" is a value
    assert csv.stats(f) is st
    assert sum(n for k, n in STATS_CACHE.passes.items() if str(path) in k) == 1
    with open(path, "a") as fh:
        fh.write("d,2.0,y\n")
    assert csv.stats(frag(path)).rows == 4
    assert sum(n for k, n in STATS_CACHE.passes.items() if str(path) in k) == 2
    assert rows_of(csv, [f]) == [{"id": "a", "score": 1.5, "name": "x"}, {"id": "b", "score": None, "name": ""},
                                 {"id": "c", "score": None, "name": None}, {"id": "d", "score": 2.0, "name": "y"}]


def test_csv_gzip_tsv_types_and_cut_files(reg, tmp_path) -> None:
    csv, tsv = reg.get("format", "csv"), reg.get("format", "tsv")
    gz = tmp_path / "t.csv.gz"
    with gzip.open(gz, "wt") as fh:
        fh.write("id,n\nx,1\ny,2\n")
    assert rows_of(csv, [frag(gz)]) == [{"id": "x", "n": 1}, {"id": "y", "n": 2}]
    t = tmp_path / "t.tsv"
    t.write_text("gene\tcode\nA\t007\nB\t010\n")
    assert rows_of(tsv, [frag(t)]) == [{"gene": "A", "code": 7}, {"gene": "B", "code": 10}]
    typed = tsv.configure({"column_types": {"code": "string"}})
    assert rows_of(typed, [frag(t)]) == [{"gene": "A", "code": "007"}, {"gene": "B", "code": "010"}]
    cut = tmp_path / "cut.csv"
    cut.write_text("id,n\nx,1\ny,2")
    with pytest.raises(FormatError, match="newline"):
        rows_of(csv, [frag(cut)])
    assert rows_of(csv.configure({"allow_unterminated": True}), [frag(cut)])[-1] == {"id": "y", "n": 2}
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("id,n\nx,1\ny\n")
    with pytest.raises(FormatError):
        rows_of(csv, [frag(ragged)])
    dup = tmp_path / "dup.csv"
    dup.write_text("id,v,v\nx,1,2\n")
    assert csv.logical_schema(frag(dup)).names == ["id", "v#1", "v#2"]
    assert rows_of(csv, [frag(dup)]) == [{"id": "x", "v#1": 1, "v#2": 2}]


# ---------------------------------------------------------------------------- jsonl


def test_jsonl_infers_types_from_the_whole_file(reg, tmp_path) -> None:
    jsonl = reg.get("format", "jsonl")
    path = tmp_path / "t.jsonl"
    lines = [{"id": f"r{i}", "tags": None, "s": None} for i in range(5)]
    lines.append({"id": "r5", "tags": ["a", "b"], "s": {"x": 1, "y": NAN_JSON}})
    path.write_text("\n".join(json.dumps(r).replace('"__nan__"', "NaN") for r in lines) + "\n")
    rows = rows_of(jsonl, [frag(path)], batch_rows=2)
    assert rows[0] == {"id": "r0", "tags": None, "s": None}
    assert rows[-1] == {"id": "r5", "tags": ["a", "b"], "s": {"x": 1, "y": None}}
    assert str(jsonl.logical_schema(frag(path)).field("tags").type) == "list<item: string>"
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": 1}\n[1, 2]\n')
    with pytest.raises(FormatError, match="line 2"):
        jsonl.logical_schema(frag(bad))


NAN_JSON = "__nan__"


# ---------------------------------------------------------------------------- npy / safetensors


def test_embeddings_need_an_aligned_vocabulary(reg, tmp_path) -> None:
    from vbt.datalayer.plugins.formats.npy import embedding_projection_golden, write_npy, write_safetensors

    table = embedding_projection_golden()
    for name, writer in (("npy", write_npy), ("safetensors", write_safetensors)):
        plugin = reg.get("format", name)
        path = tmp_path / f"vec.{name}"
        writer(table, str(path))
        rows = rows_of(plugin, [frag(path)])
        assert [r["id"] for r in rows] == table.column("id").to_pylist()
        assert rows[3]["vector"][1] is None and len(rows[0]["vector"]) == 4
        assert plugin.stats(frag(path)).shape == (5, 4)
        ids = tmp_path / "vec.ids.txt"
        ids.write_text("\n".join(table.column("id").to_pylist()[:-1]) + "\n")
        os.utime(path)
        with pytest.raises(FormatError, match="vocabulary"):
            plugin.logical_schema(frag(path, sha256=f"changed-{name}"))
        other = tmp_path / "words.txt"
        other.write_text("\n".join(f"w{i}" for i in range(5)) + "\n")
        configured = plugin.configure({"ids_from": "words.txt"})
        assert [r["id"] for r in rows_of(configured, [frag(path)])] == [f"w{i}" for i in range(5)]


# ---------------------------------------------------------------------------- zip_member


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / "golden.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("top/", b"")
        z.writestr("top/data/GSE1.csv", "id,v\na,1\nb,2\n")
        z.writestr("top/data/GSE2.csv", "id,v\nc,3\n")
        z.writestr("top/data/GSE3.csv.part", "id,v\n")
        z.writestr("top/data/.hidden.csv", "id,v\n")
        z.writestr("top/data/_SUCCESS", "")
        z.writestr("top/sets/h.all.gmt", "HALLMARK_A\turl\tTP53\n")
        z.writestr("top/onto/mini_cl.obo", MINI_CL.read_bytes())
    return path


def test_zip_members_are_listed_by_name_and_read_in_place(reg, tmp_path) -> None:
    from vbt.datalayer.plugins.layouts.zip_member import clear_archive_cache, split_uri

    layout = reg.get("layout", "zip_member")
    archive = _archive(tmp_path)
    spec = LayoutSpec(table="z.t", path="data/GSE*.csv", options={"archive": "golden.zip", "prefix": "top"},
                      fragment_key={"name": "cohort", "from": "filename_regex", "pattern": r"(GSE\d+)"}, format="csv")
    frags = layout.fragments(str(tmp_path), spec)
    assert [split_uri(f.uri)[1] for f in frags] == ["top/data/GSE1.csv", "top/data/GSE2.csv"]
    assert [f.fragment_key for f in frags] == ["GSE1", "GSE2"] and all(f.size for f in frags)
    csv = reg.get("format", "csv")
    assert rows_of(csv, frags) == [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "c", "v": 3}]
    obo = reg.get("format", "obo")
    (cl,) = layout.fragments(str(tmp_path), LayoutSpec(table="z.o", path="golden.zip!top/onto/*.obo", format="obo"))
    assert len(rows_of(obo, [cl])) == 14
    gmt = reg.get("format", "gmt")
    (h,) = layout.fragments(str(tmp_path), LayoutSpec(table="z.g", path="golden.zip!top/sets/*.gmt"))
    assert gmt.metadata(h)["fragment_key"] == "h.all"
    # fingerprints from the central directory; signature from the archive's stat
    fp = layout.fingerprint(frags, None)
    sig = layout.signature(str(tmp_path), spec)
    assert fp.startswith("fp1:zipcrc:") and layout.fingerprint(frags, None) == fp
    manifest = Manifest(path=None, entries={"top/data/GSE1.csv": {"sha256": "a" * 64}, "top/data/GSE2.csv": {
        "sha256": "b" * 64}})
    assert layout.fingerprint(frags, manifest).startswith("fp1:manifest:")
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("top/data/GSE4.csv", "id,v\nd,4\n")
    clear_archive_cache()
    assert layout.signature(str(tmp_path), spec) != sig
    frags2 = layout.fragments(str(tmp_path), spec)
    assert len(frags2) == 3 and layout.fingerprint(frags2, None) != fp
    items = {i.name: i for i in layout.probe(str(tmp_path), spec, Manifest(path=None, entries={
        "top/data/GSE1.csv": {"bytes": 1}}))}
    assert items["location"].ok and items["fragments"].ok and not items["manifest_bytes"].ok
    assert not layout.probe(str(tmp_path / "nowhere"), spec, None)[0].ok


# ---------------------------------------------------------------------------- http_range


class _RangeServer:
    """A local Range-capable HTTP server over a dict of path -> bytes; counts requests and bytes served."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests = 0
        self.bytes = 0
        outer = self

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
                outer.requests += 1
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
                self.send_header("ETag", f'"{len(body)}"')
                self.end_headers()
                if not head:
                    outer.bytes += len(part)
                    self.wfile.write(part)

            def do_HEAD(self) -> None:
                self._send(True)

            def do_GET(self) -> None:
                self._send(False)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def no_proxy(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_http_range_footer_only_stats_and_remote_reads(reg, tmp_path, no_proxy) -> None:
    big = pa.table({"id": [f"r{i:06d}" for i in range(300_000)], "x": [float(i) for i in range(300_000)]})
    buf = io.BytesIO()
    pq.write_table(big, buf, row_group_size=60_000)
    blob = buf.getvalue()
    server = _RangeServer({"/t.parquet": blob, "/t.csv": b"id,v\na,1\nb,2\n"})
    try:
        layout = reg.get("layout", "http_range")
        spec = LayoutSpec(table="h.t", path=None, options={"urls": [f"{server.base}/t.parquet"]})
        (f,) = layout.fragments("", spec)
        before = server.bytes
        stats = layout.footer_stats(f)
        assert stats.rows == 300_000 and stats.row_groups == 5 and stats.method == "footer"
        assert server.bytes - before < len(blob) / 10, "only the footer is transferred"
        sig = layout.signature("", spec)
        assert sig == layout.signature("", spec) and server.requests > 0
        assert layout.fingerprint([f], None).startswith("fp1:http:")
        assert all(i.ok for i in layout.probe("", spec, None) if i.level == "error")
        csv = reg.get("format", "csv")
        remote = Fragment(uri=f"{server.base}/t.csv", size=None, mtime_ns=None)
        assert rows_of(csv, [remote]) == [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
        missing = LayoutSpec(table="h.m", path=f"{server.base}/nope.csv")
        assert not layout.probe("", missing, None)[0].ok
        with pytest.raises(FormatError):
            layout.footer_stats(Fragment(uri=f"{server.base}/t.csv", size=None, mtime_ns=None))
    finally:
        server.close()


# ---------------------------------------------------------------------------- identifiers


def _n(reg: Any, kind: str, raw: Any, *, stored: bool = False, options: Any = None) -> Any:
    p = reg.get("identifier", kind)
    if options is not None:
        p = p.configure(options, None)
    return p.normalize_stored(raw) if stored else p.normalize(raw)


def test_ncbi_gene_needs_a_prefix_for_agent_input(reg) -> None:
    opts = {"input_requires_prefix": True}
    assert isinstance(_n(reg, "ncbi_gene", "3845", options=opts), Rejected)
    assert _n(reg, "ncbi_gene", "NCBIGene:3845", options=opts) == Normalized("3845", ("strip_prefix",))
    assert _n(reg, "ncbi_gene", "entrez:3845", options=opts).value == "3845"
    assert _n(reg, "ncbi_gene", "3845", stored=True, options=opts).value == "3845"      # stored digits overlap PMIDs
    assert isinstance(_n(reg, "pmid", "NCBIGene:3845"), Rejected)
    assert "pmid" in reg.get("identifier", "ncbi_gene").overlaps


def test_geo_census_barcode_and_ontology_terms(reg) -> None:
    assert _n(reg, "geo_gsm", "gsm1234567").value == "GSM1234567"
    assert "GSE" in _n(reg, "geo_gsm", "GSE73661").reason
    assert _n(reg, "census_joinid", 0).value == "0" and isinstance(_n(reg, "census_joinid", "-3"), Rejected)
    assert _n(reg, "cell_barcode", "aaacctgagaaaccat-1").value == "AAACCTGAGAAACCAT-1"
    assert _n(reg, "cell_ontology", "http://purl.obolibrary.org/obo/CL_0000057").value == "CL:0000057"
    assert _n(reg, "uberon", "UBERON_0002048").value == "UBERON:0002048"
    assert isinstance(_n(reg, "cell_ontology", "UBERON:0002048"), Rejected)
    cl = reg.get("identifier", "cell_ontology").configure({"obo": str(MINI_CL)}, None)
    assert cl.ancestors("CL_0002553") == {"CL:0000057", "CL:0000499", "CL:0000003", "CL:0000000"}
    assert cl.label("CL:0000057") == "fibroblast" and cl.ancestors("not a term") == frozenset()


def test_zarr_zip_store(reg, tmp_path) -> None:
    pytest.importorskip("zarr")
    pytest.importorskip("anndata")
    from vbt.datalayer.plugins.conformance.golden import matrix_golden
    from vbt.datalayer.plugins.formats.zarr import write_matrix_zarr

    case = matrix_golden("anndata_dense")
    store = tmp_path / "c.zarr"
    cfg = write_matrix_zarr(case, str(store))
    zipped = shutil.make_archive(str(tmp_path / "c.zarr"), "zip", root_dir=store)
    plugin = reg.get("format", "zarr").configure(cfg["options"], cfg["matrix"])
    rows = plugin.to_native(plugin.axis_values(frag(zipped), "row"))
    assert [r["sample_id"] for r in rows] == list(case.row_ids)
    cells = []
    for batch in plugin.slice(frag(zipped), "X", row_predicate=Eq("sample_id", case.row_ids[0]), col_keys=None,
                              budget_bytes=None):
        cells.extend(plugin.to_native(pa.Table.from_batches([batch])))
    assert len(cells) == len(case.col_ids)


@pytest.mark.parametrize("layout_name", ["single_file", "sharded_dir"])
def test_zarr_directory_stores_are_fragments(reg, tmp_path, layout_name) -> None:
    """A ``*.zarr`` directory store is one fragment (never walked into) and ``*.zarr.zip`` archives are
    listed too; the fingerprint hashes the store's files, so editing a chunk changes it."""
    pytest.importorskip("zarr")
    pytest.importorskip("anndata")
    from vbt.datalayer.plugins.base import LayoutSpec
    from vbt.datalayer.plugins.conformance.golden import matrix_golden
    from vbt.datalayer.plugins.formats.zarr import write_matrix_zarr

    case = matrix_golden("anndata_dense")
    data = tmp_path / "stores"
    data.mkdir()
    cfg = write_matrix_zarr(case, str(data / "a.zarr"))
    shutil.make_archive(str(data / "b.zarr"), "zip", root_dir=data / "a.zarr")
    layout = reg.get("layout", layout_name)
    path = "stores/*" if layout_name == "single_file" else "stores"
    lspec = LayoutSpec(table="t.m", path=path, format="zarr")
    frags = layout.fragments(str(tmp_path), lspec)
    assert [os.path.basename(f.uri) for f in frags] == ["a.zarr", "b.zarr.zip"]
    plugin = reg.get("format", "zarr").configure(cfg["options"], cfg["matrix"])
    for f in frags:
        rows = plugin.to_native(plugin.axis_values(f, "row"))
        assert [r["sample_id"] for r in rows] == list(case.row_ids)
    one = LayoutSpec(table="t.m", path="stores/a.zarr", format="zarr")
    assert [f.uri for f in reg.get("layout", "single_file").fragments(str(tmp_path), one)] == \
        [str(data / "a.zarr")]
    fp = layout.fingerprint(frags, None)
    sig = layout.signature(str(tmp_path), lspec)
    chunk = next(p for p in sorted((data / "a.zarr").rglob("*")) if p.is_file() and p.stat().st_size > 0
                 and p.name not in (".zattrs", ".zgroup", ".zarray", "zarr.json"))
    chunk.write_bytes(chunk.read_bytes() + b"\0")
    assert layout.signature(str(tmp_path), lspec) != sig
    assert layout.fingerprint(layout.fragments(str(tmp_path), lspec), None) != fp
