"""The §19 fixture precondition through the CLI: ``vbt ds index build`` then ``vbt ds check --json`` on the
OT and Tahoe fixtures report every table the six correctness tests read, and the ``ensembl_gene``,
``ot_disease`` and ``chembl_molecule`` resolver indexes, ready. A failure here is a fixture (or
readiness-check) error, not a correctness failure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import INDEXES, READ_TABLES, needs_arrow
from vbt import cli

pytestmark = needs_arrow

TAHOE_TABLES = ("de_permissive", "drug_metadata", "cell_line_metadata", "gene_metadata")


def _cases() -> list[Any]:
    refs = [f"open_targets.{t}" for t in READ_TABLES] + [f"tahoe_100m.{t}" for t in TAHOE_TABLES]
    return [pytest.param(ref, id=ref) for ref in refs]


@pytest.fixture(scope="module")
def check(ot_root: Path, tahoe_root: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    cache = tmp_path_factory.mktemp("fixture-readiness")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(ot_root))
        mp.setenv("TAHOE_DATA_PATH", str(tahoe_root))
        mp.setenv("VBT_DATA_DIR", str(cache))
        argv = ["--profile", "mock", "ds", "index", "build", "--json"]
        for name in INDEXES:
            argv += ["--id-type", f"open_targets:{name}"]
        assert cli.main(argv) == 0, "FIXTURE: `vbt ds index build` failed"
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cli.main(["--profile", "mock", "ds", "check", "--json"]) == 0
    return json.loads(buf.getvalue().strip().splitlines()[-1])


@pytest.mark.parametrize("ref", _cases())
def test_fixture_table_is_ready(check: dict[str, Any], ref: str) -> None:
    from vbt.datalayer.gateway.readiness import table_status
    from vbt.datalayer.ipc import TableCheckModel

    assert ref not in check["table_errors"], f"FIXTURE: the readiness check of {ref} failed: {check['table_errors'][ref]}"
    entry = check["tables"].get(ref)
    assert entry is not None, f"FIXTURE: `vbt ds check` did not report {ref}"
    status = table_status(TableCheckModel.model_validate(entry))
    failed = [f"{c['name']} {c.get('column') or ''}: {c['detail']}" for c in entry["checks"]
              if not c["ok"] and c["level"] == "error"]
    assert status == "ready", f"FIXTURE: {ref} is {status}: {failed[:5]}"


@pytest.mark.parametrize("name", INDEXES)
def test_fixture_index_is_ready(check: dict[str, Any], name: str) -> None:
    entry = check["indexes"].get(f"open_targets:{name}")
    assert entry is not None and entry["status"] == "ready", f"FIXTURE: resolver index {name}: {entry}"
