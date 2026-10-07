"""Small harness fixes of phase 1 (DATA_LAYER.md §11.1): the PubMed server's E-utilities base can
be pointed at a stub with ``VBT_EUTILS_BASE`` (so tests can stub E-utilities and count requests),
and the trial-outcome label row joins ``pubmed_ids`` with ``|`` like the released labels file."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
DEFAULT_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _python(code: str, **env: str) -> subprocess.CompletedProcess:
    full = {k: v for k, v in os.environ.items() if k != "VBT_EUTILS_BASE"}
    full.update(PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", NO_PROXY="127.0.0.1,localhost",
                no_proxy="127.0.0.1,localhost", **env)
    return subprocess.run([sys.executable, "-c", code], env=full, capture_output=True, text=True, timeout=120)


def test_eutils_base_defaults_to_ncbi():
    pytest.importorskip("fastmcp")
    pytest.importorskip("httpx")
    out = _python("import vbt.mcp_servers.pubmed_server as p; print(p.EUTILS)")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == DEFAULT_EUTILS


def test_eutils_base_override_is_honoured_and_requests_reach_the_stub():
    pytest.importorskip("fastmcp")
    pytest.importorskip("httpx")
    seen: list[str] = []

    class Stub(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            seen.append(self.path)
            body = json.dumps({"esearchresult": {"count": "0", "idlist": []}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep test output quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/entrez/eutils"
    try:
        out = _python("import json, vbt.mcp_servers.pubmed_server as p; "
                      "print(p.EUTILS); print(json.dumps(p.search('NCT01234567[si]')))", VBT_EUTILS_BASE=base)
    finally:
        server.shutdown()
        server.server_close()
    assert out.returncode == 0, out.stderr
    first, second = out.stdout.strip().splitlines()
    assert first == base
    assert json.loads(second)["count"] == 0
    assert len(seen) == 1 and seen[0].startswith("/entrez/eutils/esearch.fcgi?")
    assert "NCT01234567" in seen[0]


def _annotation(**kw):
    from vbt.case_studies.trial_outcomes.schema import TrialAnnotation

    rec = dict(nct_id="NCT12345678", overall_status="Completed", primary_endpoint_result="POSITIVE",
               primary_endpoint_rationale="met", secondary_endpoint_result="UNKNOWN",
               secondary_endpoint_rationale="n/a", results_source="ClinicalTrials.gov",
               ae_source="ClinicalTrials.gov", tiers_consulted=["ClinicalTrials.gov"], confidence="high")
    rec.update(kw)
    return TrialAnnotation.model_validate(rec)


def test_pubmed_ids_use_the_released_pipe_delimiter_and_round_trip():
    pytest.importorskip("pydantic")
    ann = _annotation(pubmed_ids=["2688428", "8420383", "8604728"])
    row = ann.to_label_row()
    assert row["pubmed_ids"] == "2688428|8420383|8604728"         # as in clinical_trial_labels_reconciled.csv
    back = _annotation(pubmed_ids=row["pubmed_ids"].split("|"))
    assert back.pubmed_ids == ann.pubmed_ids and back.to_label_row() == row
    assert _annotation(pubmed_ids=["31234567"]).to_label_row()["pubmed_ids"] == "31234567"
    assert _annotation().to_label_row()["pubmed_ids"] is None
