"""Zenodo archive fetcher against a local Range-capable fake server (offline)."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vbt.data import zenodo as Z

MEMBERS = {
    "virtualbiotech_submission/README.md": b"# archive\n",
    "virtualbiotech_submission/clinical_trials/code/a.py": b"print('a')\n" * 50,
    "virtualbiotech_submission/clinical_trials/data/labels.csv": b"nct_id,phase\nNCT1,2\n" * 2000,
    "virtualbiotech_submission/clinical_trials/results/fig.png": b"\x89PNG" + b"0" * 100,
    "virtualbiotech_submission/clinical_trials/benchmarks/Biomni/tdc_annotations.csv": b"nct_id\nNCT1\n",
    "virtualbiotech_submission/clinical_trials/benchmarks/Biomni/notes/deep.csv": b"x\n",
    "virtualbiotech_submission/b7-h3/data/inputs/b7h3_lung_scrnaseq.h5ad": b"H" * 3000,
    "virtualbiotech_submission/b7-h3/data/survival_outputs/results/os.json": b"{}",
    "virtualbiotech_submission/osmr/traces/agent_reports/r.md": b"report",
}


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("virtualbiotech_submission/", b"")
        for k, v in MEMBERS.items():
            z.writestr(k, v)
    return buf.getvalue()


class _Server:
    def __init__(self, blob: bytes, *, fail_first: int = 0):
        self.blob = blob
        self.md5 = hashlib.md5(blob).hexdigest()
        self.range_requests = 0
        self.fail_left = fail_first
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _file(self, head: bool):
                rng = self.headers.get("Range")
                if outer.fail_left > 0 and rng:
                    outer.fail_left -= 1
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if rng:
                    outer.range_requests += 1
                    a, b = rng.split("=")[1].split("-")
                    a, b = int(a), min(int(b), len(outer.blob) - 1)
                    body = outer.blob[a:b + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{b}/{len(outer.blob)}")
                else:
                    body = outer.blob
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if not head:
                    self.wfile.write(body)

            def do_HEAD(self):
                self._file(True)

            def do_GET(self):
                if self.path.startswith("/api/records/"):
                    base = f"http://127.0.0.1:{self.server.server_address[1]}"
                    rec = {"id": 1, "files": [{"key": "virtualbiotech_submission.zip", "size": len(outer.blob),
                                               "checksum": f"md5:{outer.md5}",
                                               "links": {"self": f"{base}/files/v.zip/content"}}]}
                    body = json.dumps(rec).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._file(False)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.api = f"http://127.0.0.1:{self.httpd.server_address[1]}/api"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = _Server(_zip_bytes())
    yield s
    s.close()


def _arch(server, **kw):
    return Z.ZenodoArchive(1, api_base=server.api, backoff=0.01, buffer_size=4096, **kw)


def test_resolve_and_list(server):
    with _arch(server) as a:
        info = a.resolve()
        assert info["md5"] == server.md5 and info["url"].endswith("/content")
        names = [i.filename for i in a.list()]
        assert set(names) == set(MEMBERS)
        assert [i.filename for i in a.list("clinical_trials/code/**")] == \
            ["virtualbiotech_submission/clinical_trials/code/a.py"]
    assert server.range_requests > 0


def test_glob_and_presets():
    assert Z.glob_match("clinical_trials/code/x/y.py", "clinical_trials/code/**")
    assert Z.glob_match("a/traces/b/c.md", "*/traces/**")
    assert not Z.glob_match("clinical_trials/benchmarks/B/notes/deep.csv", "clinical_trials/benchmarks/*/*.csv")
    infos = zipfile.ZipFile(io.BytesIO(_zip_bytes())).infolist()
    case1 = {Z._rel(i.filename) for i in Z.select_members(infos, preset="case1")}
    assert "clinical_trials/benchmarks/Biomni/tdc_annotations.csv" in case1
    assert "clinical_trials/benchmarks/Biomni/notes/deep.csv" not in case1
    assert "clinical_trials/data/labels.csv" in case1 and "README.md" in case1
    assert not any(n.startswith("b7-h3") for n in case1)
    res = {Z._rel(i.filename) for i in Z.select_members(infos, preset="b7h3-results")}
    assert res == {"b7-h3/data/survival_outputs/results/os.json"}  # h5ad excluded
    small = {Z._rel(i.filename) for i in Z.select_members(infos, preset="small")}
    assert "clinical_trials/results/fig.png" not in small
    capped = Z.select_members(infos, preset="all", max_member_mb=0.002)
    assert all(i.file_size <= 0.002 * 1024 * 1024 for i in capped)
    assert Z.select_members(infos) == []
    with pytest.raises(KeyError):
        Z.select_members(infos, preset="nope")


def test_fetch_skip_and_atomic(server, tmp_path):
    with _arch(server) as a:
        rep = a.fetch(preset="case1", dest=tmp_path)
    assert len(rep.fetched) == rep.selected == 5 and rep.bytes_fetched > 0 and rep.bytes_transferred > 0
    f = tmp_path / "virtualbiotech_submission/clinical_trials/data/labels.csv"
    assert f.read_bytes() == MEMBERS["virtualbiotech_submission/clinical_trials/data/labels.csv"]
    assert not list(tmp_path.rglob("*.part*"))
    # re-run: everything skipped by size + CRC
    with _arch(server) as a:
        rep2 = a.fetch(preset="case1", dest=tmp_path)
    assert rep2.fetched == [] and len(rep2.skipped) == 5
    # a corrupted local copy (same size) is re-fetched
    data = bytearray(f.read_bytes())
    data[0] ^= 0xFF
    f.write_bytes(bytes(data))
    with _arch(server) as a:
        rep3 = a.fetch(["clinical_trials/data/*.csv"], dest=tmp_path)
    assert rep3.fetched == ["virtualbiotech_submission/clinical_trials/data/labels.csv"]
    assert f.read_bytes() == MEMBERS["virtualbiotech_submission/clinical_trials/data/labels.csv"]


def test_retry_on_transient_errors(tmp_path):
    s = _Server(_zip_bytes(), fail_first=2)
    try:
        with Z.ZenodoArchive(1, api_base=s.api, backoff=0.01, buffer_size=4096) as a:
            rep = a.fetch(["README.md"], dest=tmp_path)
        assert rep.fetched == ["virtualbiotech_submission/README.md"]
    finally:
        s.close()


def test_full_download_md5(server, tmp_path):
    with _arch(server) as a:
        p = a.download(tmp_path)
    assert p.name == "virtualbiotech_submission.zip" and hashlib.md5(p.read_bytes()).hexdigest() == server.md5


def test_unsafe_member_rejected(tmp_path):
    with pytest.raises(ValueError):
        Z._safe_target(tmp_path, "../evil.txt")


def test_zenodo_root(monkeypatch, tmp_path):
    monkeypatch.setenv("VBT_ZENODO_DIR", str(tmp_path))
    assert Z.zenodo_root() == tmp_path / "virtualbiotech_submission"
    monkeypatch.delenv("VBT_ZENODO_DIR")
    assert Z.zenodo_root({"zenodo": {"dir": str(tmp_path / "virtualbiotech_submission")}}) == \
        tmp_path / "virtualbiotech_submission"


def test_cli_fetch_case1(server, tmp_path, monkeypatch, capsys):
    from vbt import cli

    monkeypatch.setenv("VBT_ZENODO_DIR", str(tmp_path))
    rc = cli.main(["data", "zenodo", "fetch", "--preset", "case1", "--record", "1", "--api-base", server.api])
    assert rc == 0
    assert (tmp_path / "virtualbiotech_submission/clinical_trials/code/a.py").exists()
    assert "5 fetched" in capsys.readouterr().out
    rc = cli.main(["data", "zenodo", "list", "--pattern", "b7-h3/**", "--record", "1", "--api-base", server.api])
    out = capsys.readouterr().out
    assert rc == 0 and "b7h3_lung_scrnaseq.h5ad" in out and "clinical_trials" not in out
