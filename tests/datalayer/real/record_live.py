"""Re-record the live-source exchanges the offline suite replays (DEP-18).

The offline tests replay real exchanges with ClinicalTrials.gov, cBioPortal and NCBI E-utilities: every request a
scenario makes through the shipped descriptors, in order. When a source changes (or a descriptor changes what it
sends), run the scenario again against the live API with a transport that keeps each exchange::

    python tests/datalayer/real/record_live.py real_live pubmed_count_ceiling     # -> live/replay/<name>.json
    python tests/datalayer/real/record_live.py round3 release_pubmed ceiling_count  # -> live/round3/<name>.json
    python tests/datalayer/real/record_live.py round3 --all --dry-run              # what would be written

``real_live`` runs ``tests/datalayer/test_dl_real_live.py``'s ``SCENARIOS``, ``round3`` those of
``test_dl_round3_live.py``. Requests go one at a time at the descriptors' own rate limits. A file is rewritten with
its ``exchanges``, ``retrieved``, ``seconds`` and ``source``; any other top-level key the file holds (a recorded
count a test compares with) is kept and listed, because it may need a manual update. Bodies are kept whole (not
trimmed): check the size before committing. Afterwards run the offline module to see what the new answers change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[1]
REPO = TESTS.parent
MODULES = {"real_live": ("test_dl_real_live", HERE / "live" / "replay"),
           "round3": ("test_dl_round3_live", HERE / "live" / "round3")}
KEPT_HEADERS = ("date", "content-type")


class Recorder:
    """The live transport, keeping each exchange the way ``Replay`` reads it."""

    def __init__(self, http_get: Any) -> None:
        self.http_get = http_get
        self.exchanges: list[dict[str, Any]] = []

    def __call__(self, url: str, params: Any = None, *, timeout: float = 30.0, headers: Any = None
                 ) -> tuple[int, dict[str, str], bytes]:
        status, got, body = self.http_get(url, params, timeout=timeout, headers=headers)
        ex: dict[str, Any] = {"url": url,
                              "params": {str(k): str(v) for k, v in (params or {}).items() if v is not None},
                              "status": status,
                              "headers": {k: v for k, v in got.items() if k.lower() in KEPT_HEADERS}}
        try:
            ex["body"] = json.loads(body)
        except ValueError:
            ex["text"] = body.decode("utf-8", errors="replace")
        ex["bytes"] = len(body)
        self.exchanges.append(ex)
        return status, got, body


def _import(module: str) -> Any:
    os.environ["VBT_DL_NETWORK"] = "1"               # the scenarios' contexts keep the real base URLs
    for p in (REPO / "src", TESTS, TESTS / "datalayer"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return __import__(module)


def record(kind: str, names: list[str], *, dry_run: bool = False) -> list[Path]:
    import pytest

    module, out_dir = MODULES[kind]
    mod = _import(module)
    from vbt.datalayer.plugins.layouts import live_api
    from vbt.datalayer.service.verbs import witness

    written: list[Path] = []
    for name in names:
        rec = Recorder(live_api.http_get)
        with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(prefix="vbt-record-") as tmp:
            for target, attr in ((live_api, "_RELEASES"), (live_api, "_RELEASE_FAILURES"),
                                 (witness, "_STUDY_RELEASES")):
                mp.setattr(target, attr, {})
            mp.setattr(live_api.LiveApiLayout, "transport", staticmethod(rec))
            started = time.monotonic()
            out = mod.SCENARIOS[name](Path(tmp))
            if hasattr(out, "__await__"):
                out = asyncio.run(out)
            seconds = round(time.monotonic() - started, 2)
        path = out_dir / f"{name}.json"
        old = json.loads(path.read_text()) if path.is_file() else {}
        doc = {"scenario": name, "retrieved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": seconds,
               "source": f"recorded by tests/datalayer/real/record_live.py {kind} {name} through the shipped "
                         f"descriptors",
               "exchanges": rec.exchanges}
        if "result" in old:
            doc["result"] = json.loads(json.dumps(out, default=str))
        kept = {k: v for k, v in old.items() if k not in doc and k not in ("result",)}
        doc.update(kept)
        size = len(json.dumps(doc))
        print(f"{name}: {len(rec.exchanges)} request(s) in {seconds} s, {size:,} bytes -> {path}"
              + (f"; kept {sorted(kept)} from the old file (check them)" if kept else ""))
        for ex in rec.exchanges:
            print(f"  {ex['status']} {ex['url']} {ex['params']}")
        if not dry_run:
            path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=sorted(MODULES))
    ap.add_argument("names", nargs="*", help="scenario names (see the module's SCENARIOS)")
    ap.add_argument("--all", action="store_true", help="every scenario of the module")
    ap.add_argument("--dry-run", action="store_true", help="make the requests, write nothing")
    args = ap.parse_args(argv)
    mod = _import(MODULES[args.kind][0])
    names = sorted(mod.SCENARIOS) if args.all else args.names
    unknown = [n for n in names if n not in mod.SCENARIOS]
    if not names or unknown:
        ap.error(f"name one or more of {', '.join(sorted(mod.SCENARIOS))}" + (f" (unknown: {unknown})" if unknown
                                                                              else ""))
    record(args.kind, names, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
