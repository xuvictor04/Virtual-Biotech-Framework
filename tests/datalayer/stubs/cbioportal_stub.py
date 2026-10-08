"""A local cBioPortal REST stub serving the ``pybioportal`` stub's study, for the data child's live tables.

The upstream clinicaltrials server reads cBioPortal through the ``pybioportal`` stub (no network); the data child
reads the REST API itself (``live_api`` layout), so a derived ``get_clinical_data`` needs the same study over HTTP.
``start(log_path)`` serves on 127.0.0.1 in a daemon thread and returns a :class:`CbioportalStub` whose ``base`` is
what ``VBT_CBIOPORTAL_BASE`` should be set to. Every request is appended to the log as one JSON line
``{"path", "params"}``.

The answers have the shapes the real API gave on 2026-10-08 (``tests/datalayer/real/live/round3``): ``/info``
names the portal and database versions, a study carries its ``importDate``, ``projection=META`` answers an empty
body with the total in the ``total-count`` header, ``pageNumber`` is a page index from 0, and an unknown study is
HTTP 404 with ``{"message": "... not found: <id>"}``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from stubs.pybioportal import ATTRIBUTES, PATIENT_DATA, PATIENTS, SAMPLE_DATA, SAMPLES, STUDY

PATH = "/cbioportal/api"
IMPORT_DATE = "2026-01-07 15:46:33"
INFO = {"portalVersion": "fixture-7.1.2", "dbVersion": "3.0.0"}


def study() -> dict[str, Any]:
    return {"studyId": STUDY, "name": "Fixture study", "description": "two patients", "cancerTypeId": "luad",
            "pmid": "1", "citation": "Fixture 2026", "allSampleCount": len(SAMPLES), "sequencedSampleCount": 3,
            "cnaSampleCount": 3, "mrnaRnaSeqV2SampleCount": 0, "rppaSampleCount": 0, "completeSampleCount": 3,
            "importDate": IMPORT_DATE}


def samples() -> list[dict[str, Any]]:
    return [{"sampleId": s, "patientId": p, "studyId": STUDY, "sampleType": "Primary Solid Tumor"}
            for s, p in SAMPLES.items()]


def clinical(kind: str) -> list[dict[str, Any]]:
    if kind == "PATIENT":
        return [{"patientId": p, "studyId": STUDY, "clinicalAttributeId": a, "value": v}
                for p, attrs in PATIENT_DATA.items() for a, v in attrs.items()]
    return [{"sampleId": s, "patientId": SAMPLES[s], "studyId": STUDY, "clinicalAttributeId": a, "value": v}
            for s, attrs in SAMPLE_DATA.items() for a, v in attrs.items()]


def attributes() -> list[dict[str, Any]]:
    return [{"clinicalAttributeId": a, "displayName": d, "description": d, "datatype": t, "patientAttribute": p,
             "priority": "1", "studyId": STUDY} for a, d, t, p in ATTRIBUTES]


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def log_message(self, *args: Any) -> None:   # noqa: D102 - quiet
        return

    def do_GET(self) -> None:   # noqa: N802 - BaseHTTPRequestHandler API
        url = urlparse(self.path)
        params = {k: v[-1] for k, v in parse_qs(url.query).items()}
        path = url.path[len(PATH):] if url.path.startswith(PATH) else url.path
        self.server.record({"path": path, "params": params})
        parts = [p for p in path.split("/") if p]
        if parts == ["info"]:
            return self._send(INFO)
        if parts == ["studies"]:
            return self._rows([study()], params)
        if len(parts) >= 2 and parts[0] == "studies":
            if parts[1] != STUDY:
                return self._send({"message": f"Study not found: {parts[1]}"}, status=404)
            rest = parts[2:]
            if not rest:
                return self._send(study())
            if rest == ["samples"]:
                return self._rows(samples(), params)
            if rest == ["patients"]:
                return self._rows([{"patientId": p, "studyId": STUDY} for p in PATIENTS], params)
            if rest == ["clinical-data"]:
                return self._rows(clinical(params.get("clinicalDataType", "SAMPLE")), params)
            if rest == ["clinical-attributes"]:
                return self._rows(attributes(), params)
        return self._send({"message": f"no such endpoint: {path}"}, status=404)

    def _rows(self, rows: list[dict[str, Any]], params: dict[str, str]) -> None:
        if params.get("projection") == "META":
            return self._send(None, headers={"total-count": str(len(rows))})
        size = int(params.get("pageSize") or 10_000_000)
        page = int(params.get("pageNumber") or 0)
        return self._send(rows[page * size:(page + 1) * size])

    def _send(self, body: Any, *, status: int = 200, headers: dict[str, str] | None = None) -> None:
        data = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, log_path: Path) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.log_path = log_path
        self._lock = threading.Lock()

    def record(self, entry: dict[str, Any]) -> None:
        with self._lock, self.log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")


@dataclass
class CbioportalStub:
    server: _Server
    thread: threading.Thread
    log_path: Path

    @property
    def base(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}{PATH}"

    def requests(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines() if line.strip()]

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def start(log_path: str | Path) -> CbioportalStub:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    server = _Server(log)
    thread = threading.Thread(target=server.serve_forever, name="cbioportal-stub", daemon=True)
    thread.start()
    return CbioportalStub(server, thread, log)
