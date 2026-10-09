"""What real project data showed the data layer (package D5's reports).

* A descriptor column key is a literal top-level name, which may hold a dot (HGNC's ``pseudogene.org``). The reader
  already quoted such a name as one §6.4 path segment, but the R4 type check split the declared path at its dots and
  reported the column absent (``schema_drift``), and a scan of a file format without footers (CSV) asked the format
  for the column ``pseudogene``. A project's drafts left such columns out. Both formats now check ``ready``, and the
  column is read and filtered on under its name.
* The layouts' directory walk followed directory links without remembering where it had been: a link loop never
  ended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import yaml  # noqa: E402

from vbt.datalayer.predicate import Eq  # noqa: E402
from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service.checks import check_table  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402

ROWS = [{"hgnc_id": f"HGNC:{i}", "pseudogene.org": None if i % 3 else f"PGOHUM{i:05d}"} for i in range(50)]


def _ctx(tmp: Path, fmt: str) -> ServiceContext:
    d = tmp / "data" / "genes"
    d.mkdir(parents=True)
    defaults: dict[str, Any]
    if fmt == "parquet":
        pq.write_table(pa.Table.from_pylist(ROWS), d / "part-00000.parquet")
        defaults = {"format": "parquet", "layout": "sharded_dir"}
    else:
        (d / "genes.tsv").write_text("hgnc_id\tpseudogene.org\n" + "".join(
            f"{r['hgnc_id']}\t{r['pseudogene.org'] or ''}\n" for r in ROWS))
        defaults = {"format": {"plugin": "csv", "options": {"delimiter": "\t"}},
                    "layout": {"plugin": "sharded_dir", "options": {"pattern": "*.tsv"}}}
    desc = {"schema": "vbt.datasource/1", "source": "s", "title": "s", "root": str(tmp / "data"),
            "release": {"from": "literal"}, "defaults": defaults, "id_types": {},
            "tables": {"genes": {"kind": "entity", "path": "genes", "grain": "gene",
                                 "key": {"columns": ["hgnc_id"], "check": "full"},
                                 "columns": {"hgnc_id": {"role": "identifier"},
                                             "pseudogene.org": {"role": "label"}}}}}
    (tmp / "sources").mkdir()
    (tmp / "overlays").mkdir()
    (tmp / "sources" / "s.yaml").write_text(yaml.safe_dump(desc, sort_keys=False))
    return ServiceContext(DataSettings.from_dict({"descriptors_dir": str(tmp / "sources"),
                                                  "overlays_dir": str(tmp / "overlays"),
                                                  "cache_dir": str(tmp / "cache")}, project_root=tmp))


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_a_dotted_top_level_column_is_checked_read_and_filtered(tmp_path, fmt):
    ctx = _ctx(tmp_path, fmt)
    assert not ctx.catalog.quarantined and not ctx.catalog.lint()
    model = check_table(ctx, "s.genes", "standard")
    assert model.status == "ready", [c.detail for c in model.checks if not c.ok]
    reader = ctx.reader("s.genes")
    col = reader.physical_path("pseudogene.org")
    assert col == "`pseudogene.org`"
    rows = [m.row for m in reader.scan(None, columns=[col])]
    assert len(rows) == 50 and rows[0]["pseudogene.org"] == "PGOHUM00000" and rows[1]["pseudogene.org"] is None
    found = [m.row for m in reader.scan(Eq(col, "PGOHUM00003"), columns=["*"])]
    assert found == [{"hgnc_id": "HGNC:3", "pseudogene.org": "PGOHUM00003"}]


def test_the_layout_walk_ends_on_a_directory_link_loop(tmp_path):
    """``sharded_dir`` and the listings walk directory links; a link back to an ancestor walked forever (D5)."""
    import os

    from vbt.datalayer.plugins.layouts import _walk

    base = tmp_path / "t"
    (base / "a" / "b").mkdir(parents=True)
    (base / "a" / "b" / "part-0.parquet").write_bytes(b"x")
    os.symlink(base, base / "a" / "b" / "loop")                     # back to the root
    os.symlink(base / "a", base / "a" / "again")                    # back to a parent
    files = sorted(rel for rel, _e in _walk(str(base)))
    assert files == ["a/b/part-0.parquet"]
