"""Generic, declarative acquisition (``vbt data acquire | status``): the acquisition plugin kind, the descriptors'
``acquisition`` sections, the engine, readiness hints and the ``data.acquisition.auto`` policy.

Offline (always), every transfer from a local HTTP fixture server:

* the kind's wiring: a harness-side kind beside the five data-child kinds, discovered and validated like them, a
  third-party transport added without a core change; the A-1..A-7 conformance suite runs over every builtin
  (collected here);
* the descriptor model refuses inconsistent sections; every shipped section names a registered transport, declares
  its licence, points the data layer at the files through variables the data child receives, and the Open Targets
  section declares the 38 tables with the counts and bytes of the 25.09 footer scan;
* recorded real index documents (``tests/datalayer/real/acquisition/``: the Hugging Face tree, the figshare file
  list, the Zenodo record and an S3 listing, retrieved 2026-10-08) decode through the plugins to what the shipped
  descriptors declare;
* the engine: the plan (states, declared and listed sizes, time at a rate), parallel downloads that take their
  name only after size, checksum and framing match, resume from ``.part``, the size and disk refusals, the
  manifest in the upstream format, a rerun that transfers nothing, the pinned release, prepare steps (a scratch
  script; and the unmodified upstream ``tools/prepare_tahoe.py`` on generated shards through the shipped tahoe
  section when the upstream checkout is present), the env file and the provenance record;
* overlays -> tools -> tables (``--for-tools``, ``--for-agents``), ``vbt data status``, the not_ready reason that
  says how to acquire a table, and :func:`vbt.data.ondemand.between_turns` under ``off``, ``ask`` and
  ``under_budget``; the CLI end to end.

``VBT_DL_NETWORK=1``: every shipped source is listed live (sizes equal the declared ones) and two small tables
are acquired. ``VBT_DL_REAL_DATA=<data/real>``: the manifests ``vbt data acquire`` wrote there still match the files.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Iterator, Mapping

import pytest
import yaml

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from netgate import network_enabled  # noqa: E402
from vbt.data import acquire as A  # noqa: E402
from vbt.data import ondemand, targets  # noqa: E402
from vbt.data.manifest import load_manifest, local_files  # noqa: E402
from vbt.datalayer.catalog import build_catalog  # noqa: E402
from vbt.datalayer.descriptor.load import load_descriptor  # noqa: E402
from vbt.datalayer.descriptor.models import AcquisitionSpec, SourceDescriptor  # noqa: E402
from vbt.datalayer.gateway.readiness import ReadinessCache, acquisition_hint, auto_decision, call_readiness  # noqa: E402
from vbt.datalayer.ipc import CheckItemModel, TableCheckModel  # noqa: E402
from vbt.datalayer.plugins import HARNESS_KINDS, KINDS  # noqa: E402
from vbt.datalayer.plugins.acquisition import HttpSession, glob_match  # noqa: E402
from vbt.datalayer.plugins.base import (  # noqa: E402
    AcquisitionBase,
    AcquisitionError,
    PluginBase,
    PluginError,
    RemoteFile,
)
from vbt.datalayer.plugins.conformance.acquisition import (  # noqa: E402,F401  (the A-1..A-7 suite is collected here)
    SAMPLE,
    _site_cleanup,
    test_a1_a2_listing_is_exactly_the_published_files,
    test_a3_fetch_returns_the_bytes,
    test_a4_a_missing_or_damaged_file_raises,
    test_a5_an_index_that_fails_raises,
    test_a6_names_outside_the_destination_never_list,
    test_a7_describe,
    AcquisitionCases,
    FixtureSite,
    check_listing,
    fetch_all,
)
from vbt.datalayer.plugins.registry import discover, discover_harness, validate_plugin  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "configs" / "data" / "sources"
RECORDED = Path(__file__).resolve().parent / "real" / "acquisition"
OT_SNAP = Path(__file__).resolve().parent / "real" / "ot_25_09" / "_release.json"
BUILTINS = ["gcs", "http", "huggingface", "json_index", "s3", "zip_member"]


# ---------------------------------------------------------------------------- the kind


def test_acquisition_is_a_harness_side_kind():
    assert set(KINDS) == {"format", "layout", "statistic", "identifier", "envelope"}
    assert list(HARNESS_KINDS) == ["acquisition"]
    reg = discover_harness(entry_points=False)
    assert reg.names("acquisition") == BUILTINS
    for p in reg.all("acquisition"):
        assert {"listing", "fetch", "describe"} <= set(dir(p))
    # the data child's registry holds none of them, and neither registry trips over the other's plugins
    data = discover(entry_points=False, extra=[type(reg.get("acquisition", "http"))])
    assert "acquisition" not in data.kinds and data.find("acquisition", "http") is None


def test_validation_of_acquisition_plugins():
    class NoListing(PluginBase):
        kind = "acquisition"
        name = "dl_bad_transport"
        capabilities = frozenset()

        def fetch(self, *a: Any, **k: Any) -> Iterator[bytes]:
            yield b""

        def describe(self, options: Mapping[str, Any]) -> str:
            return "x"

    with pytest.raises(PluginError, match="listing"):
        validate_plugin(NoListing(), HARNESS_KINDS)
    bad = type("Teleport", (LocalDirTransport,), {"capabilities": frozenset({"teleport"})})
    with pytest.raises(PluginError, match="unknown acquisition capabilities"):
        validate_plugin(bad(), HARNESS_KINDS)


class LocalDirTransport(AcquisitionBase):
    """A third-party transport: the files of a local directory (``options.dir``)."""

    name = "dl_test_localdir"
    version = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"range", "sizes", "checksums"})

    def listing(self, options: Mapping[str, Any], session: Any, *, index_cache: Any = None) -> list[RemoteFile]:
        root = Path(options["dir"])
        if not root.is_dir():
            raise AcquisitionError(f"{root} is not a directory")
        out = []
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root).as_posix()
            if p.is_file() and ".." not in rel:
                out.append(RemoteFile(rel, str(p), p.stat().st_size,
                                      {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}))
        return out

    def fetch(self, file: RemoteFile, session: Any, *, offset: int = 0) -> Iterator[bytes]:
        with open(file.url, "rb") as fh:
            fh.seek(offset)
            yield fh.read()

    def describe(self, options: Mapping[str, Any]) -> str:
        return f"the local directory {options.get('dir')}"

    @classmethod
    def conformance_cases(cls) -> Any:
        def publish(files: Mapping[str, bytes], root: str) -> dict[str, tuple[bytes, str]]:
            d = Path(os.environ["DL_TEST_LOCALDIR"])
            for rel, body in files.items():
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / rel).write_bytes(body)
            return {}

        return AcquisitionCases(publish=publish, options=lambda root: {"dir": os.environ["DL_TEST_LOCALDIR"]})


def test_a_third_party_transport_needs_no_core_change(tmp_path, monkeypatch):
    monkeypatch.setenv("DL_TEST_LOCALDIR", str(tmp_path / "src"))
    reg = discover_harness(entry_points=False, extra=[LocalDirTransport])
    plugin = reg.get("acquisition", "dl_test_localdir")
    cases = plugin.conformance_cases()
    cases.publish(SAMPLE, "")
    with HttpSession(retry_wait=0) as s:
        files = plugin.listing(cases.options(""), s)
        check_listing(plugin, files, SAMPLE)
        assert all(fetch_all(plugin, f, s) == SAMPLE[f.path] for f in files)


# ---------------------------------------------------------------------------- the descriptor model


def _base_desc(**acq: Any) -> dict[str, Any]:
    return {"schema": "vbt.datasource/1", "source": "s", "title": "s", "release": {"from": "literal"},
            "defaults": {"format": "parquet", "layout": "sharded_dir"},
            "tables": {"t": {"kind": "fact", "path": "t", "grain": "row", "key": {"columns": ["id"]},
                             "columns": {"id": {"role": "identifier"}}},
                       "items": {"kind": "fact", "items_of": {"table": "t", "path": "xs[]"}, "grain": "item",
                                 "key": {"columns": []}}},
            "acquisition": {"transport": {"plugin": "http", "options": {"base": "https://x/"}}, **acq}}


@pytest.mark.parametrize("acq, message", [
    ({"tables": {"items": {"files": ["a"]}}}, "item tables"),
    ({"tables": {"t": {}}}, "names `files` or"),
    ({"tables": {"t": {"prepared_by": "missing"}}}, "is not a prepare step"),
    ({"tables": {"t": {"files": ["a"]}}, "extra": {"t": {"files": ["b"]}}}, "extra names tables"),
    ({"prepare": {"p": {"command": ["x"], "needs": ["ghost"], "output": "o"}}}, "unknown download groups"),
    ({"transport": None}, "needs a `transport`"),
    ({"verify": ["md5sum"]}, "verify"),
])
def test_inconsistent_acquisition_sections_are_refused(acq, message):
    with pytest.raises(ValueError, match=message):
        SourceDescriptor.model_validate(_base_desc(**acq))


def test_a_narrowed_descriptor_keeps_the_other_tables_files_as_optional_groups():
    """A copy of a descriptor that declares fewer tables (a test, an operator's narrowed source) stays valid: the
    files of the tables it dropped become optional download groups, so prepare steps that read them resolve."""
    desc = SourceDescriptor.model_validate(_base_desc(
        tables={"t": {"files": ["t/*"]}, "dropped": {"files": ["d/*"]}},
        prepare={"p": {"command": ["x"], "needs": ["dropped"], "output": "o"}}))
    acq = desc.acquisition
    assert list(acq.tables) == ["t"]
    assert acq.extra["dropped"].optional and acq.extra["dropped"].files == ["d/*"]


def test_the_shipped_sections():
    tool_env = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())["tool_env"]
    reg = discover_harness(entry_points=False)
    seen = {}
    for path in sorted(SOURCES.glob("*.yaml")):
        desc = load_descriptor(path)
        acq = desc.acquisition
        if desc.kind == "local":
            assert acq is not None, f"{desc.source}: a local source declares how its files are acquired"
        if acq is None:
            continue
        seen[desc.source] = acq
        raw = yaml.safe_load(path.read_text())
        assert set(raw["acquisition"].get("tables", {})) <= set(raw["tables"]), f"{desc.source}: table names"
        assert acq.licence, desc.source
        assert reg.find("acquisition", acq.transport.plugin) is not None, desc.source
        for var in acq.env:
            assert var in tool_env, f"{var} must reach the data child (configs/default.yaml tool_env)"
        if acq.mode == "download":
            covered = set(acq.tables) | {t for t, s in desc.tables.items() if s.items_of is not None}
            assert covered >= {t for t, s in desc.tables.items() if s.materialized_by is None}, desc.source
            for name, entry in {**acq.tables, **acq.extra}.items():
                if entry.files:
                    assert entry.bytes is not None and entry.count is not None, f"{desc.source}.{name}: sizes known"
    assert sorted(seen) == ["cell_ontology", "cellxgene_census", "depmap", "gene_ontology", "msigdb",
                            "open_targets", "tahoe_100m", "zenodo_vbt"]
    assert seen["cellxgene_census"].mode == "remote"
    ot = seen["open_targets"]
    rel = json.loads(OT_SNAP.read_text())["tables"]
    assert {t: (e.count, e.bytes) for t, e in ot.tables.items()} == {t: (v["shards"], v["bytes"]) for t, v in rel.items()}
    assert sum(e.count for e in ot.tables.values()) == 3508
    assert sum(e.bytes for e in ot.tables.values()) == 31_131_380_890
    assert ot.index_cache == "_release" and "parquet_framing" in ot.verify and ot.manifest == ".download-manifest.json"
    step = seen["tahoe_100m"].prepare["prepare_tahoe"]
    assert step.command[:3] == ["{python}", "-B", "{upstream}/tools/prepare_tahoe.py"] and step.fresh
    assert step.manifest == "preparation_manifest.json" and step.needs == ["de_shards", "metadata"]
    assert all(e.prepared_by == "prepare_tahoe" for e in seen["tahoe_100m"].tables.values())
    assert seen["tahoe_100m"].extra["de_shards"].count == 1026


def test_acquired_files_land_where_the_descriptors_read_them():
    """With the variables an acquisition sets, each table's root + path (relative to the source's home) is what its
    acquisition patterns name: a directory they fill, or the file (glob) itself."""
    import re

    home = Path("/H")
    checked = 0
    for path in sorted(SOURCES.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        desc = load_descriptor(path)
        acq = desc.acquisition
        if acq is None or acq.mode != "download":
            continue
        env = A.source_env(desc, home, A.source_release(desc))

        def expand(text: str) -> str:
            return re.sub(r"\$\{([A-Za-z_]\w*)(?::?-(?:[^{}]|\{[^{}]*\})*)?\}",      # one nested ${...} default
                          lambda m: env.get(m.group(1), m.group(0)), text)

        for name, entry in acq.tables.items():
            if not entry.files:
                continue
            table_path = expand(str(raw["tables"][name]["path"]))
            where = Path(table_path) if table_path.startswith("/") else Path(expand(str(raw.get("root") or ""))) / table_path
            rel = where.relative_to(home).as_posix()
            assert any(p == rel or p.startswith(rel + "/") for p in entry.files), (desc.source, name, rel, entry.files)
            checked += 1
    assert checked == 38 + 4 + 1 + 1 + 1 + 6          # OT, DepMap, GO, CL, MSigDB, Zenodo


def test_globs_span_directories_only_with_two_stars():
    assert glob_match("go/part-0.parquet", "go/**/*.parquet")
    assert glob_match("evidence/sourceId=chembl/part-0.parquet", "evidence/**/*.parquet")
    assert not glob_match("_samples/go/part-0.parquet", "go/**/*.parquet")
    assert not glob_match("metadata/pseudobulk_differential_expression/x.parquet", "metadata/*.parquet")


# ---------------------------------------------------------------------------- recorded real index documents


def _recorded(name: str) -> dict[str, Any]:
    return json.loads((RECORDED / name).read_text())


@pytest.fixture
def site() -> Iterator[FixtureSite]:
    s = FixtureSite()
    yield s
    s.close()


def test_the_recorded_hugging_face_tree_lists_what_tahoe_declares(site):
    rec = _recorded("hf_tahoe_metadata_tree.json")
    desc = load_descriptor(SOURCES / "tahoe.yaml")
    rev = desc.acquisition.release
    site.publish({f"/api/datasets/tahoebio/Tahoe-100M/tree/{rev}/metadata?recursive=true":
                  (json.dumps(rec["entries"]).encode(), "application/json")})
    opts = {**A.transport_options(desc, rev), "endpoint": site.root}
    plugin = discover_harness(entry_points=False).get("acquisition", "huggingface")
    with HttpSession(retry_wait=0) as s:
        files = {f.path: f for f in plugin.listing(opts, s)}
    meta = desc.acquisition.extra["metadata"]
    mine = [files[p] for p in meta.files]
    assert sum(f.size for f in mine) == meta.bytes and all(len(f.checksums["sha256"]) == 64 for f in mine)
    assert files["metadata/gene_vocabulary.json"].checksums.keys() == {"git_sha1"}     # not stored with LFS
    de = rec["de_shards"]
    assert (de["files"], de["bytes"], de["lfs_sha256"]) == (desc.acquisition.extra["de_shards"].count,
                                                            desc.acquisition.extra["de_shards"].bytes, True)


def test_the_recorded_figshare_list_and_zenodo_record_match_the_descriptors(site):
    from vbt.datalayer.plugins.acquisition.json_index import index_files

    fig = _recorded("figshare_27993248_files.json")
    desc = load_descriptor(SOURCES / "depmap.yaml")
    files = {f.path: f for f in index_files(fig["excerpt"], A.transport_options(desc, "24Q4"), url=fig["source"])}
    for table, entry in desc.acquisition.tables.items():
        (name,) = entry.files
        assert files[name].size == entry.bytes and len(files[name].checksums["md5"]) == 32, table
    zen = _recorded("zenodo_22259123_record.json")
    zdesc = load_descriptor(SOURCES / "zenodo.yaml")
    spec = A.transport_options(zdesc, "22259123")["archive_index"]
    (archive,) = [f for f in index_files(zen["record"], spec, url=zen["source"]) if f.path == spec["select"]]
    assert archive.size == 2_909_966_318 and archive.checksums == {"md5": "9e30309e22cf269431af573fd732fb49"}
    assert zen["resolved"].endswith("/records/22259124")


def test_the_recorded_s3_listing_decodes(site):
    rec = _recorded("s3_census_obs_listing.json")
    xml = rec["xml"].replace("<IsTruncated>true</IsTruncated>", "<IsTruncated>false</IsTruncated>")
    prefix = "cell-census/2025-11-08/soma/census_data/homo_sapiens/obs/"
    from urllib.parse import urlencode

    site.publish({f"/?{urlencode({'list-type': '2', 'prefix': prefix})}": (xml.encode(), "application/xml")})
    plugin = discover_harness(entry_points=False).get("acquisition", "s3")
    with HttpSession(retry_wait=0) as s:
        files = plugin.listing({"bucket": "cellxgene-census-public-us-west-2", "prefix": prefix,
                                "endpoint": site.root}, s)
    assert len(files) == 5 and all(f.size is not None and len(f.checksums["md5"]) == 32 for f in files)
    assert files[0].path.startswith("__commits/")


# ---------------------------------------------------------------------------- the engine


def _parquet(rows: int, seed: int) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.table({"id": [f"r{seed}-{i}" for i in range(rows)]}), buf)
    return buf.getvalue()


class DemoRelease:
    """A release site like Open Targets': ``/<release>/out/<table>/...`` and a sha1 list with its ``.sha1``."""

    FILES = {"alpha/part-0.parquet": 5, "alpha/part-1.parquet": 7, "beta/x=1/part-0.parquet": 4,
             "gamma/g.parquet": 3}

    def __init__(self, site: FixtureSite, release: str = "1.0") -> None:
        self.site = site
        self.release = release
        self.files = {rel: _parquet(n, i) for i, (rel, n) in enumerate(self.FILES.items())}
        self.publish()

    def publish(self, *, corrupt: str | None = None) -> None:
        docs: dict[str, tuple[Any, ...]] = {}
        lines = []
        for rel, body in self.files.items():
            served = body if rel != corrupt else body[:20] + bytes([body[20] ^ 1]) + body[21:]
            docs[f"/{self.release}/out/{rel}"] = (served, "application/octet-stream")
            lines.append(f"{hashlib.sha1(body).hexdigest()}  ./out/{rel}")
        lines.append("0" * 40 + "  ./out/alpha/_SUCCESS")
        text = ("\n".join(lines) + "\n").encode()
        docs[f"/{self.release}/list"] = (text, "text/plain")
        docs[f"/{self.release}/list.sha1"] = (f"{hashlib.sha1(text).hexdigest()}  list\n".encode(), "text/plain")
        self.site.publish(docs)

    def descriptor(self, root: Path | None = None, **acq: Any) -> dict[str, Any]:
        site = self.site.root
        tables = {t: {"kind": "fact", "path": t, "grain": "row", "key": {"columns": ["id"]},
                      "columns": {"id": {"role": "identifier"}}} for t in ("alpha", "beta", "gamma")}
        spec = {"release": self.release, "licence": "CC0 1.0 (test)",
                "transport": {"plugin": "http", "options": {
                    "base": f"{site}/{{release}}/out/", "listing": "checksums",
                    "checksums": {"url": f"{site}/{{release}}/list", "algo": "sha1", "prefix": "./out/",
                                  "verify": f"{site}/{{release}}/list.sha1"}}},
                "dir": "demo/{release}", "env": {"DEMO_DATA_PATH": "{home}"}, "index_cache": "_release",
                "verify": ["size", "checksum", "parquet_framing"],
                "tables": {"alpha": {"files": ["alpha/**/*.parquet"], "count": 2,
                                     "bytes": len(self.files["alpha/part-0.parquet"]) + len(self.files["alpha/part-1.parquet"])},
                           "beta": {"files": ["beta/**/*.parquet"], "count": 1,
                                    "bytes": len(self.files["beta/x=1/part-0.parquet"])},
                           "gamma": {"files": ["gamma/*.parquet"], "count": 1,
                                     "bytes": len(self.files["gamma/g.parquet"])}}}
        spec.update(acq)
        return {"schema": "vbt.datasource/1", "source": "demo", "title": "demo",
                "root": str(root) if root else "${DEMO_DATA_PATH}",
                "release": {"expect": self.release, "from": "manifest.release"},
                "manifests": [{"path": ".download-manifest.json", "required": False, "require": {"complete": True}}],
                "defaults": {"format": "parquet", "layout": "sharded_dir"}, "tables": tables, "acquisition": spec}


DEMO_OVERLAY: dict[str, Any] = {
    "schema": "vbt.overlay/1", "server": "demo", "sources": ["demo"],
    "tools": {"get_alpha": {"reads": {"demo.alpha": {"access": "full_table"}}, "args": {},
                            "result": {"rows": "$.rows"}}}}


def _catalog(tmp: Path, descs: list[dict[str, Any]], overlays: list[dict[str, Any]] = (DEMO_OVERLAY,)) -> Any:
    (tmp / "sources").mkdir(parents=True, exist_ok=True)
    (tmp / "overlays").mkdir(parents=True, exist_ok=True)
    for d in descs:
        (tmp / "sources" / f"{d['source']}.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    for o in overlays:
        (tmp / "overlays" / f"{o['server']}.yaml").write_text(yaml.safe_dump(o, sort_keys=False))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp / "sources"), "overlays_dir": str(tmp / "overlays"),
                                       "cache_dir": str(tmp / "cache")}, project_root=tmp)
    return build_catalog(settings, discover(settings, entry_points=False))


def _settings(tmp: Path, **kw: Any) -> A.AcquisitionSettings:
    base = dict(root=tmp / "acq", workers=3, retries=1, reserve_bytes=0, provenance_dir=tmp / "prov",
                cache_dir=tmp / "cache")
    base.update(kw)
    return A.AcquisitionSettings(**base)


def test_plan_execute_manifest_and_rerun(site, tmp_path):
    rel = DemoRelease(site)
    cat = _catalog(tmp_path / "c", [rel.descriptor()])
    st = _settings(tmp_path)
    offline = A.plan_acquisition(cat, {"demo": ["alpha", "beta"]}, st, offline=True)
    (sp,) = offline.sources
    assert sp.offline and sp.declared_files == 3 and sp.bytes_remaining == sp.declared_bytes
    assert sp.home == tmp_path / "acq" / "demo" / "1.0" and sp.env == {"DEMO_DATA_PATH": str(sp.home)}
    plan = A.plan_acquisition(cat, {"demo": ["alpha", "beta"]}, st)
    (sp,) = plan.sources
    assert [f.state for f in sp.files] == ["missing"] * 3 and sp.bytes_remaining == sp.declared_bytes
    assert (sp.home / "_release" / "list").is_file()          # the checksum list is kept beside the data
    text = "\n".join(plan.lines())
    assert "licence: CC0 1.0 (test)" in text and "env: DEMO_DATA_PATH=" in text and "MB/s (assumed)" in text
    rep = A.execute(plan, st, retry_wait=0)
    assert rep.ok and {r.status for r in rep.sources[0].files} == {"downloaded"}
    for p, body in rel.files.items():
        if not p.startswith("gamma/"):
            assert (sp.home / p).read_bytes() == body
    assert not (sp.home / "gamma").exists() and not list(sp.home.rglob("*.part"))
    man = load_manifest(sp.home / ".download-manifest.json")
    assert man["release"] == "1.0" and man["base"] == f"{site.root}/1.0/out/" and man["complete"] is True
    assert man["tables"] == ["alpha", "beta"] and man["expected_files"] == len(man["files"]) == 3
    assert man["archive_files"] == 4 and man["verified"] == "sha1 for 3 files against list"
    entry = man["files"]["beta/x=1/part-0.parquet"]
    assert entry["verified_by"] == "sha1"                    # per file, what it was checked against (ACC-3)
    assert entry["sha1"] == hashlib.sha1(rel.files["beta/x=1/part-0.parquet"]).hexdigest()
    assert entry["sha256"] == hashlib.sha256(rel.files["beta/x=1/part-0.parquet"]).hexdigest()
    lock = load_manifest(sp.home / A.LOCK)
    assert lock["release"] == "1.0" and sorted(lock["groups"]) == ["alpha", "beta"]
    # a rerun lists, finds every file verified by the manifest and transfers nothing
    site.requests.clear()
    again = A.plan_acquisition(cat, {"demo": ["alpha", "beta", "gamma"]}, st)
    assert [f.state for f in again.sources[0].files] == ["verified"] * 3 + ["missing"]
    rep2 = A.execute(again, st, retry_wait=0)
    assert [r.status for r in rep2.sources[0].files].count("present") == 3
    assert [p for _m, p, _r in site.requests if "/out/" in p] == ["/1.0/out/gamma/g.parquet"]
    assert load_manifest(sp.home / ".download-manifest.json")["tables"] == ["alpha", "beta", "gamma"]


def test_a_corrupt_file_never_takes_its_name_and_resumes_from_part(site, tmp_path):
    rel = DemoRelease(site)
    rel.publish(corrupt="alpha/part-1.parquet")
    cat = _catalog(tmp_path / "c", [rel.descriptor()])
    st = _settings(tmp_path)
    rep = A.execute(A.plan_acquisition(cat, {"demo": ["alpha", "gamma"]}, st), st, retry_wait=0)
    (bad,) = rep.sources[0].failed
    home = tmp_path / "acq" / "demo" / "1.0"
    assert bad.rel == "alpha/part-1.parquet" and "sha1" in bad.error and not rep.ok
    assert not (home / bad.rel).exists() and not (home / (bad.rel + ".part")).exists()
    assert load_manifest(home / ".download-manifest.json")["tables"] == ["gamma"]
    # an interrupted transfer: the next run asks for the rest only
    rel.publish()
    body = rel.files["alpha/part-1.parquet"]
    (home / "alpha" / "part-1.parquet.part").write_bytes(body[: len(body) // 2])
    site.requests.clear()
    rep = A.execute(A.plan_acquisition(cat, {"demo": ["alpha"]}, st), st, retry_wait=0)
    assert rep.ok and (home / "alpha" / "part-1.parquet").read_bytes() == body
    assert [(p, r) for m, p, r in site.requests if p.endswith("part-1.parquet") and m == "GET"] == \
        [("/1.0/out/alpha/part-1.parquet", f"bytes={len(body) // 2}-")]


def test_size_disk_and_release_refusals(site, tmp_path, monkeypatch):
    rel = DemoRelease(site)
    cat = _catalog(tmp_path / "c", [rel.descriptor()])
    st = _settings(tmp_path)
    plan = A.plan_acquisition(cat, {"demo": ["alpha"]}, st)
    with pytest.raises(ValueError, match="over the"):
        A.execute(plan, st, max_bytes=10)
    monkeypatch.setattr(A.shutil, "disk_usage", lambda p: SimpleNamespace(free=100, total=1000, used=900))
    with pytest.raises(ValueError, match="is free"):
        A.execute(plan, st)
    monkeypatch.undo()
    home = tmp_path / "acq" / "demo" / "1.0"
    home.mkdir(parents=True, exist_ok=True)
    (home / A.LOCK).write_text(json.dumps({"release": "0.9"}))
    rep = A.execute(plan, st, retry_wait=0)
    assert "holds release 0.9" in rep.sources[0].error and not rep.ok
    assert not (home / "alpha").exists()


def _prepare_desc(rel: DemoRelease, script: Path) -> dict[str, Any]:
    d = rel.descriptor()
    d["tables"]["prepped"] = {"kind": "fact", "path": "prepped", "grain": "row", "key": {"columns": ["id"]},
                              "columns": {"id": {"role": "identifier"}}}
    d["acquisition"]["tables"]["prepped"] = {"prepared_by": "make"}
    d["acquisition"]["env"] = {"DEMO_DATA_PATH": "{home}/out"}
    d["acquisition"]["prepare"] = {"make": {"command": ["{python}", str(script), "{downloads}", "{output}",
                                                       "{release}"],
                                            "needs": ["alpha", "gamma"], "output": "out", "manifest": "done.json"}}
    return d


PREPARE_SCRIPT = """
import json, shutil, sys
from pathlib import Path
src, out, release = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if out.exists():
    sys.exit("destination exists")
(out / "prepped").mkdir(parents=True)
for p in sorted((src / "alpha").glob("*.parquet")):
    shutil.copy(p, out / "prepped" / p.name)
(out / "done.json").write_text(json.dumps({"complete": True, "release": release}))
print("prepared", release)
"""


def test_prepare_steps_run_after_their_inputs_verify(site, tmp_path):
    script = tmp_path / "prep.py"
    script.write_text(PREPARE_SCRIPT)
    rel = DemoRelease(site)
    cat = _catalog(tmp_path / "c", [_prepare_desc(rel, script)])
    st = _settings(tmp_path)
    plan = A.plan_acquisition(cat, {"demo": ["prepped"]}, st)
    (sp,) = plan.sources
    assert sp.groups == ["alpha", "gamma"] and sp.prepare == ["make"] and sp.prepared == {"make": "pending"}
    rep = A.execute(plan, st, retry_wait=0)
    step = rep.sources[0].prepare["make"]
    assert step["status"] == "ran" and step["returncode"] == 0 and "prepared 1.0" in step["stdout_tail"], step
    home = sp.home
    assert sorted(p.name for p in (home / "out" / "prepped").iterdir()) == ["part-0.parquet", "part-1.parquet"]
    assert rep.ok and load_manifest(home / A.LOCK)["prepare"]["make"]["status"] == "ran"
    # done once: a rerun does not run it again; an output left incomplete blocks it (the step creates it whole)
    rep = A.execute(A.plan_acquisition(cat, {"demo": ["prepped"]}, st), st, retry_wait=0)
    assert rep.sources[0].prepare["make"]["status"] == "done"
    (home / "out" / "done.json").write_text(json.dumps({"complete": False}))
    rep = A.execute(A.plan_acquisition(cat, {"demo": ["prepped"]}, st), st, retry_wait=0)
    assert rep.sources[0].prepare["make"]["status"] == "blocked" and not rep.ok
    # failed inputs block it
    import shutil

    shutil.rmtree(home)
    rel.publish(corrupt="gamma/g.parquet")
    rep = A.execute(A.plan_acquisition(cat, {"demo": ["prepped"]}, st), st, retry_wait=0)
    assert rep.sources[0].prepare["make"] == {"status": "blocked", "detail": "downloads failed for gamma"}


def _de_shard() -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.table({
        "gene_name": ["A", "B", "C", "D", "E"], "baseMean": pa.array([1.0, 2, 3, 4, 5], pa.float32()),
        "log2FoldChange": pa.array([1.0, 0.1, -2.0, None, 0.7], pa.float32()), "lfcSE": pa.array([0.1] * 5, pa.float32()),
        "stat": pa.array([1.0] * 5, pa.float32()), "pvalue": pa.array([0.01] * 5, pa.float32()),
        "padj": pa.array([0.01, 0.04, 0.2, 0.01, 0.07], pa.float32()), "plate": ["1"] * 5,
        "n_cells_trt": [10] * 5, "n_cells_ctrl": [10] * 5, "Cell_ID_Cellosaur": ["CVCL_0023"] * 5,
        "Cell_ID_DepMap": ["ACH-000681"] * 5, "drug": ["Bortezomib"] * 5,
        "concentration": pa.array([0.05] * 5, pa.float32()), "concentration_unit": ["uM"] * 5,
        "Cell_Name_Vevo": ["A549"] * 5}), buf)
    return buf.getvalue()


def test_the_shipped_tahoe_section_runs_the_unmodified_upstream_preparation(site, tmp_path):
    """The shipped tahoe acquisition against a generated Hugging Face repository (one DE shard, the four metadata
    tables): every file verified by its sha256, then upstream tools/prepare_tahoe.py, unmodified."""
    from dl_upstream import upstream_root

    up = upstream_root()
    if not (up / "tools" / "prepare_tahoe.py").is_file():
        pytest.skip(f"the upstream checkout is not at {up}")
    from vbt.datalayer.plugins.acquisition.huggingface import HuggingFaceTransport

    files = {"metadata/pseudobulk_differential_expression/train-00000-of-00001.parquet": _de_shard()}
    for name in ("gene", "drug", "cell_line", "sample"):
        files[f"metadata/{name}_metadata.parquet"] = _parquet(2, hash(name) % 97)
    site.publish(HuggingFaceTransport.conformance_cases().publish(files, site.root))
    shipped = load_descriptor(SOURCES / "tahoe.yaml")
    acq = shipped.acquisition
    transport = acq.transport.model_copy(update={"options": {"repo": "org/repo", "revision": "abc123",
                                                             "endpoint": site.root}})
    desc = shipped.model_copy(update={"acquisition": acq.model_copy(update={"transport": transport})})
    st = _settings(tmp_path)
    sp = A.plan_source(desc, ["de_permissive"], st, home=tmp_path / "tahoe")
    assert sp.groups == ["de_shards", "metadata"] and len(sp.files) == 5 and not sp.listing_error
    rep = A.execute(A.AcquisitionPlan([sp], tmp_path), st, config={"vars": {"upstream": str(up)}}, retry_wait=0)
    step = rep.sources[0].prepare["prepare_tahoe"]
    assert step["status"] == "ran", step
    assert step["argv"][1:3] == ["-B", str(up / "tools" / "prepare_tahoe.py")]
    man = load_manifest(tmp_path / "tahoe" / "prepared" / "preparation_manifest.json")
    assert man["complete"] is True and man["source_revision"] == acq.release
    assert man["rows"] == {"source": 5, "permissive": 3, "significant": 2, "high_quality": 1}
    assert rep.sources[0].env == {"TAHOE_DATA_PATH": str(tmp_path / "tahoe" / "prepared")}


def test_zip_members_through_a_json_index(site, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("top/data/t/part-0.parquet", _parquet(6, 1))
        zf.writestr("top/data/other.csv", b"a,b\n")
    blob = buf.getvalue()
    site.publish({"/files/a.zip": (blob, "application/zip"),
                  "/api/records/7": (json.dumps({"id": 8, "files": [
                      {"key": "a.zip", "size": len(blob), "checksum": f"md5:{hashlib.md5(blob).hexdigest()}",
                       "links": {"self": f"{site.root}/files/a.zip"}}]}).encode(), "application/json")})
    d = {"schema": "vbt.datasource/1", "source": "z", "title": "z", "root": "${Z_DIR}/top",
         "release": {"expect": "7", "from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
         "tables": {"t": {"kind": "fact", "path": "data/t", "grain": "row", "key": {"columns": ["id"]},
                          "columns": {"id": {"role": "identifier"}}}},
         "acquisition": {"release": "7", "licence": "CC BY 4.0 (test)", "env": {"Z_DIR": "{home}"},
                         "transport": {"plugin": "zip_member", "options": {"archive_index": {
                             "index": f"{site.root}/api/records/{{release}}", "files": "$.files[*]", "path": "$.key",
                             "url": "$.links.self", "size": "$.size", "checksums": {"md5": "$.checksum"},
                             "select": "a.zip", "record": {"version_record": "$.id"}}}},
                         "tables": {"t": {"files": ["top/data/t/**"], "count": 1, "bytes": 1}}}}
    cat = _catalog(tmp_path / "c", [d])
    st = _settings(tmp_path)
    rep = A.execute(A.plan_acquisition(cat, {"z": ["t"]}, st), st, retry_wait=0)
    home = tmp_path / "acq" / "z" / "7"
    assert rep.ok and (home / "top/data/t/part-0.parquet").is_file() and not (home / "top/data/other.csv").exists()
    man = load_manifest(home / ".download-manifest.json")
    assert man["integrity"]["version_record"] == 8 and len(man["files"]["top/data/t/part-0.parquet"]["crc32"]) == 8


def test_env_file_and_provenance(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# keep\nOTHER=1\nDEMO_DATA_PATH="/old"\n')
    A.write_env_file(env, {"DEMO_DATA_PATH": "/new", "X_PATH": "/x"})
    assert env.read_text() == '# keep\nOTHER=1\nDEMO_DATA_PATH="/new"\nX_PATH="/x"\n'
    st = _settings(tmp_path)
    rep = A.AcquisitionReport(sources=[A.SourceResult("demo", "1.0", tmp_path, files=[
        A.FileResult("a", "downloaded", bytes=3), A.FileResult("b", "failed", error="boom")])], bytes_downloaded=3)
    rec = A.record_provenance(st, rep, by="auto", run_dir=tmp_path / "run", extra={"wanted": {"demo": ["a"]}})
    line = json.loads((tmp_path / "prov" / "acquisitions.jsonl").read_text())
    assert line == json.loads(json.dumps(rec, default=str)) and line["by"] == "auto" and line["ok"] is False
    assert line["sources"][0]["failed"] == ["b"] and (tmp_path / "run" / "data_acquisitions.jsonl").is_file()


def test_settings_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path / "d"))
    st = A.AcquisitionSettings.from_config({"vars": {"project_root": str(tmp_path)}})
    assert st.root == tmp_path / "d" / "sources" and st.auto == "off" and st.workers >= 1
    st = A.AcquisitionSettings.from_config({"vars": {"project_root": str(tmp_path)}, "data": {"acquisition": {
        "root": "rel/root", "auto": "under_budget", "budget_bytes": "2 GB", "workers": 3}}})
    assert (st.root, st.auto, st.budget_bytes, st.workers) == (tmp_path / "rel/root", "under_budget", 2 * 10**9, 3)
    with pytest.raises(ValueError, match="off, ask or under_budget"):
        A.AcquisitionSettings.from_config({"data": {"acquisition": {"auto": "always"}}})


# ---------------------------------------------------------------------------- tools -> tables, status, readiness


@pytest.fixture(scope="module")
def shipped() -> Any:
    settings = DataSettings.from_dict({})
    return build_catalog(settings, discover(settings, entry_points=False))


def test_tools_and_agents_map_to_the_tables_they_read(shipped):
    assert "open_targets.target" in targets.tool_tables(shipped, "target", "get_target_info")
    found, missing = targets.expand_tools(shipped, ["target.*", "mcp__nope__x"])
    assert ("target", "get_target_info") in found and len(found) > 3 and missing == ["mcp__nope__x"]
    config = {"agents": {"agents": {"geneticist": {"tools": [["mcp__functional_genomics__query_drug_perturbation"],
                                                              "mcp__data__*", "Read"]}}}}
    t = targets.resolve_targets(shipped, config, ["open_targets.target_go", "cellxgene_census.obs"],
                                agents=["geneticist", "ghost"])
    assert t.wanted["open_targets"] == ["target"] and "tahoe_100m" in t.wanted
    assert set(t.wanted["tahoe_100m"]) >= {"de_permissive", "drug_metadata"}
    assert "item table" in t.why["open_targets.target"][0]
    assert any("read live" in n for n in t.notes) and any("mcp__data__*" in n for n in t.notes)
    assert t.errors == ["ghost: unknown agent (configs/agents.yaml)"]
    t = targets.resolve_targets(shipped, {}, ["open_targets.no_such", "nosource"])
    assert len(t.errors) == 2
    t = targets.resolve_targets(shipped, {}, ["cell_ontology"], include_optional=True)
    assert t.wanted == {"cell_ontology": ["term", "uberon_basic"]}
    assert targets.resolve_targets(shipped, {}, ["cell_ontology"]).wanted == {"cell_ontology": ["term"]}
    unlocks = targets.tools_by_table(shipped)
    assert "mcp__target__get_target_info" in unlocks["open_targets.target"]


def _missing(name: str = "R1:location") -> TableCheckModel:
    return TableCheckModel(status="missing", checks=[CheckItemModel(name=name, ok=False, detail="no files")])


def test_a_not_ready_reason_says_how_to_acquire_the_table(shipped, tmp_path):
    cache = ReadinessCache(tmp_path / "cache", shipped)
    cache.tables["open_targets.target"] = _missing()
    contract = shipped.contract("target", "get_target_info")
    r = call_readiness(contract, cache, bound_table=contract.bound_table)
    (reason,) = [x for x in r.reasons if x["name"] == "open_targets.target"]
    acq = reason["acquire"]
    assert acq["command"] == "vbt data acquire open_targets.target" and acq["files"] == 10
    assert acq["bytes"] == 75_568_567 and acq["release"] == "25.09" and acq["licence"].startswith("CC0 1.0")
    assert "policy" not in acq
    assert "`vbt data acquire open_targets.target` (75.57 MB in 10 file(s), release 25.09" in reason["hint"]
    # the policy, when the cache knows it
    cache.acquisition = {"auto": "under_budget", "budget_bytes": "1 GB"}
    r = call_readiness(contract, cache, bound_table=contract.bound_table)
    (reason,) = [x for x in r.reasons if x["name"] == "open_targets.target"]
    assert reason["acquire"]["policy"] == "auto" and "between turns" in reason["hint"]
    # a prepared table names the prepare step and its inputs' size; a remote table has no acquisition hint
    h = acquisition_hint(shipped, "tahoe_100m.de_permissive", {"auto": "under_budget", "budget_bytes": 10**9})
    assert h["prepare"] == ["prepare_tahoe"] and h["bytes"] == 88_859_715_303 + 1_451_950 and h["files"] == 1030
    assert h["policy"] == "over_budget"
    assert acquisition_hint(shipped, "cellxgene_census.obs") is None
    # schema drift is not fixed by acquiring files
    cache.tables["open_targets.target"] = TableCheckModel(
        status="schema_drift", checks=[CheckItemModel(name="R4", ok=False, detail="drift")])
    r = call_readiness(contract, cache, bound_table=contract.bound_table)
    assert all("acquire" not in x for x in r.reasons)


@pytest.mark.parametrize("policy, nbytes, decision", [
    (None, 5, "off"), ({"auto": "off"}, 5, "off"), ({"auto": False}, 5, "off"), ({"auto": "ask"}, 5, "ask"),
    ({"auto": "under_budget", "budget_bytes": 10}, 5, "auto"), ({"auto": "under_budget", "budget_bytes": 4}, 5,
                                                                "over_budget"),
    ({"auto": "under_budget", "budget": "1 KB"}, 1000, "auto"), ({"auto": "under_budget", "budget_bytes": 10}, None,
                                                                 "over_budget"),
])
def test_auto_decision(policy, nbytes, decision):
    assert auto_decision(policy, nbytes)[0] == decision


def test_status_shows_present_verified_ready_and_unlocks(site, tmp_path, monkeypatch):
    from vbt.data import status as S

    rel = DemoRelease(site)
    desc = rel.descriptor()
    cat = _catalog(tmp_path / "c", [desc])
    st = _settings(tmp_path)
    A.execute(A.plan_acquisition(cat, {"demo": ["alpha"]}, st), st, retry_wait=0)
    home = tmp_path / "acq" / "demo" / "1.0"
    cfg = {"vars": {"project_root": str(tmp_path)}, "data": {
        "descriptors_dir": str(tmp_path / "c" / "sources"), "overlays_dir": str(tmp_path / "c" / "overlays"),
        "cache_dir": str(tmp_path / "cache"), "acquisition": {"root": str(tmp_path / "acq")}}}
    monkeypatch.delenv("DEMO_DATA_PATH", raising=False)
    (src,) = S.collect_status(cfg)
    assert src.acquired == ["alpha"] and "DEMO_DATA_PATH=" in src.hint and src.tables[0].present == "absent"
    monkeypatch.setenv("DEMO_DATA_PATH", str(home))
    from vbt import preflight

    preflight._CATALOGS.clear()                        # the per-process catalog expanded the variable unset
    (src,) = S.collect_status(cfg)
    by = {t.table: t for t in src.tables}
    assert (by["demo.alpha"].present, by["demo.alpha"].verified, by["demo.alpha"].ready) == ("2 file(s)", "yes",
                                                                                            "unchecked")
    assert by["demo.beta"].present == "absent" and by["demo.alpha"].acquirable
    assert by["demo.alpha"].tools == ["mcp__demo__get_alpha"] and by["demo.beta"].tools == []
    (home / "alpha" / "part-0.parquet").write_bytes(b"PAR1" + b"0" * 40 + b"PAR1")
    assert {t.table: t.verified for t in S.collect_status(cfg)[0].tables}["demo.alpha"] == "no"
    text = "\n".join(S.status_lines(S.collect_status(cfg)))
    assert "demo [local] release 1.0" in text and "demo.alpha" in text


# ---------------------------------------------------------------------------- on demand


def test_between_turns_under_each_policy(site, tmp_path):
    rel = DemoRelease(site)
    cat = _catalog(tmp_path / "c", [rel.descriptor()])
    refusal = {"tables": [{"name": "demo.alpha", "check": "R1:location", "detail": "no files",
                           "acquire": {"table": "demo.alpha"}}, {"name": "demo.nope"}]}
    assert ondemand.refused_tables(cat, [refusal]) == {"demo": ["alpha"]}
    out = ondemand.between_turns({}, refusals=[refusal], catalog=cat, settings=_settings(tmp_path, auto="off"))
    assert out["decision"].startswith("off") and not (tmp_path / "acq").exists()
    st = _settings(tmp_path, auto="ask")
    out = ondemand.between_turns({}, refusals=[refusal], catalog=cat, settings=st)
    assert out["decision"].startswith("queued for approval") and ondemand.pending(st) == {"demo": ["alpha"]}
    st = _settings(tmp_path, auto="under_budget", budget_bytes=10)
    out = ondemand.between_turns({}, refusals=[refusal], catalog=cat, settings=st)
    assert "over the budget" in out["decision"] and not (tmp_path / "acq").exists()
    st = _settings(tmp_path, auto="under_budget", budget_bytes=10**6)
    out = ondemand.between_turns({}, refusals=[refusal], catalog=cat, settings=st, run_dir=tmp_path / "run")
    assert out["decision"] == "acquired" and (tmp_path / "acq" / "demo" / "1.0" / "alpha" / "part-0.parquet").is_file()
    prov = [json.loads(x) for x in (tmp_path / "prov" / "acquisitions.jsonl").read_text().splitlines()]
    assert prov[-1]["by"] == "auto" and prov[-1]["policy"] == "under_budget" and prov[-1]["wanted"] == {"demo": ["alpha"]}
    assert (tmp_path / "run" / "data_acquisitions.jsonl").is_file() and ondemand.pending(st) == {}


# ---------------------------------------------------------------------------- the CLI


def test_the_cli_plans_acquires_and_reports(site, tmp_path, capsys, monkeypatch):
    from vbt.cli import main

    rel = DemoRelease(site)
    _catalog(tmp_path / "c", [rel.descriptor()])
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"data": {
        "descriptors_dir": str(tmp_path / "c" / "sources"), "overlays_dir": str(tmp_path / "c" / "overlays"),
        "cache_dir": str(tmp_path / "cache"), "provenance": {"dir": str(tmp_path / "prov")},
        "acquisition": {"root": str(tmp_path / "acq"), "retries": 0}}}))
    base = ["--profile", "mock", "--profile", str(profile), "data"]
    assert main([*base, "acquire", "demo.alpha", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "demo 1.0 (download)" in out and "for: alpha" in out and not (tmp_path / "acq" / "demo").exists()
    assert main([*base, "acquire", "demo.alpha", "--plan", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["sources"][0]["states"]["missing"] == 2 and body["why"] == {"demo.alpha": ["requested"]}
    env = tmp_path / "vbt.env"
    assert main([*base, "acquire", "demo", "--env-file", str(env)]) == 0
    out = capsys.readouterr().out
    assert "demo 1.0: 4 file(s) downloaded" in out and env.read_text() == f'DEMO_DATA_PATH="{tmp_path / "acq" / "demo" / "1.0"}"\n'
    # the host-wide log sits in the acquisition root (data.provenance.dir is relative to a run)
    assert json.loads((tmp_path / "acq" / "acquisitions.jsonl").read_text().splitlines()[-1])["by"] == "cli"
    assert not (tmp_path / "prov").exists()
    monkeypatch.setenv("DEMO_DATA_PATH", str(tmp_path / "acq" / "demo" / "1.0"))
    assert main([*base, "status", "demo"]) == 0
    assert "demo.gamma" in capsys.readouterr().out
    assert main([*base, "acquire", "demo.nope"]) == 2
    assert "has no table or download group" in capsys.readouterr().err
    assert main([*base, "acquire"]) == 2


# ---------------------------------------------------------------------------- opt-in: the network and real data

NETWORK = network_enabled()


#: Groups with more files than this are checked on a sample (first, last and three between) instead of every file.
HEAD_ALL_FILES = 120


def _remote_size(url: str) -> int | None:
    """The server's own size of ``url``: HEAD Content-Length, else the total of a one-byte range read. The bytes on
    the wire are not the file's when the server compresses them (the MSigDB GMT: 20,551 gzip bytes for a
    48,690-byte file), so no encoding is accepted."""
    import httpx

    headers = {"Accept-Encoding": "identity"}
    with httpx.Client(follow_redirects=True, timeout=60, headers=headers) as client:
        r = client.head(url)
        if r.status_code < 400 and r.headers.get("content-length") and not r.headers.get("content-encoding"):
            return int(r.headers["content-length"])
        r = client.get(url, headers={"Range": "bytes=0-0"})
        total = (r.headers.get("content-range") or "").rpartition("/")[2]
        return int(total) if total.isdigit() else None


@pytest.mark.skipif(not NETWORK, reason="set VBT_DL_NETWORK=1 to list the real sources")
def test_live_listings_match_the_declared_sizes(shipped, tmp_path):
    """Every shipped download source listed live; the files of each declared group are what it declares, and their
    sizes are the servers' own (ACC-2): the Open Targets listing (a checksum list) carries no sizes and a per-file
    http transport reports the descriptor's declared bytes, so each file is sized by HEAD (every file of a group up
    to HEAD_ALL_FILES, whose sum must be the declared bytes; a sample beyond). An unknown size fails the test."""
    from concurrent.futures import ThreadPoolExecutor

    st = _settings(tmp_path, retries=2)
    wanted = {s: list(d.acquisition.tables) + list(d.acquisition.extra)
              for s, d in shipped.sources.items() if d.acquisition is not None and d.acquisition.mode == "download"}
    plan = A.plan_acquisition(shipped, wanted, st, root=tmp_path / "acq", sizes=False)
    from vbt.data.manifest import group_patterns

    for sp in plan.sources:
        assert not sp.listing_error, (sp.source, sp.listing_error)
        acq = sp.desc.acquisition
        for g in sp.groups:
            entry = acq.tables.get(g) or acq.extra.get(g)
            mine = sorted((f for f in sp.files if g in f.groups), key=lambda f: f.remote.path)
            assert len(mine) == entry.count, (sp.source, g, len(mine))
            assert group_patterns(acq, g, sp.release)[0]
            if not mine or mine[0].remote.member:          # members of an archive have no URL of their own
                continue
            every = len(mine) <= HEAD_ALL_FILES
            picked = mine if every else [mine[i] for i in sorted({0, len(mine) // 4, len(mine) // 2,
                                                                   3 * len(mine) // 4, len(mine) - 1})]
            with ThreadPoolExecutor(8) as pool:
                sizes = list(pool.map(lambda f: _remote_size(f.remote.url), picked))
            unknown = [f.remote.path for f, s in zip(picked, sizes) if not s]
            assert not unknown, (sp.source, g, unknown[:3])
            for f, size in zip(picked, sizes):
                if f.remote.size is not None:
                    assert size == f.remote.size, (sp.source, f.remote.path, size, f.remote.size)
            if every:
                assert sum(sizes) == entry.bytes, (sp.source, g, sum(sizes), entry.bytes)


@pytest.mark.skipif(not NETWORK, reason="set VBT_DL_NETWORK=1 to acquire two small real tables")
def test_live_small_acquisition(shipped, tmp_path):
    st = _settings(tmp_path, retries=2)
    rep = A.execute(A.plan_acquisition(shipped, {"open_targets": ["so"], "msigdb": ["hallmark"]}, st), st)
    assert rep.ok, rep.lines()
    man = load_manifest(tmp_path / "acq" / "open_targets" / "25.09" / ".download-manifest.json")
    assert man["tables"] == ["so"] and man["integrity"]["sha1"] == "872decc7ef350306d0843ba6279019662b004a42"
    assert load_manifest(tmp_path / "acq" / "msigdb" / "2024.1.Hs" / ".download-manifest.json")["files"][
        "h.all.v2024.1.Hs.symbols.gmt"]["sha256"] == "ee2463540042078bfa3f67828e1e223bb354446d9fbb4d22845866835ba5c772"


REAL = os.environ.get("VBT_DL_REAL_DATA", "").strip()


@pytest.mark.skipif(not REAL, reason="set VBT_DL_REAL_DATA=<data/real> to check the acquired real directories")
def test_real_acquired_directories_still_match_their_manifests():
    """Every ``.download-manifest.json`` ``vbt data acquire`` wrote under the real-data root lists files that are
    present with the recorded size (no file changed or vanished since)."""
    root = Path(REAL)
    found = 0
    for lock in sorted(root.glob("*/*/" + A.LOCK)):
        man = load_manifest(lock.parent / ".download-manifest.json")
        if not man:
            continue
        found += 1
        for rel, entry in man["files"].items():
            assert (lock.parent / rel).stat().st_size == entry["bytes"], (lock.parent, rel)
        assert set(man["files"]) <= set(local_files(lock.parent, skip_dirs=["_release"])), lock.parent
    if not found:
        pytest.skip(f"no acquired directory under {root}")


def test_no_pyarrow_in_the_acquisition_modules():
    """The engine, the plugins and the CLI run in the harness: none imports pyarrow at import time."""
    code = ("import sys; import vbt.data.acquire, vbt.data.cli, vbt.data.status, vbt.data.ondemand, "
            "vbt.datalayer.plugins.acquisition.http, vbt.datalayer.plugins.acquisition.zip_member; "
            "print('pyarrow' in sys.modules)")
    import subprocess

    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert out.stdout.strip() == "False", out.stderr


def test_acquisition_spec_defaults():
    spec = AcquisitionSpec.model_validate({"transport": {"plugin": "http"}})
    assert (spec.mode, spec.dir, spec.downloads, spec.manifest, spec.verify) == (
        "download", "{source}/{release}", ".", ".download-manifest.json", ["size", "checksum"])
