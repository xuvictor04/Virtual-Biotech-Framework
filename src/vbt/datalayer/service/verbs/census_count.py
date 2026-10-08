"""``_census_count``: count-first admission of Census pulls and donor-balanced sampling (phase 4, F20).

Upstream ``get_anndata``, ``get_expression_for_genes`` and ``get_anndata_donor_balanced`` fetch first
and check the size after (``single_cell_mcp/tools.py:483-490``). This verb counts first: the obs filter
(a SOMA ``value_filter`` or the predicate IR) is compiled by the ``soma`` format and counted by the
``soma`` layout (one read of ``soma_joinid``), against the release ``stable`` resolves to (recorded with
its ``release_confidence`` and any drift). The estimate is ``cells x (row_bytes + genes x value_bytes)``
(a dense upper bound unless the caller passes a measured value size); over ``cap_bytes`` the pull is
not admissible and the gateway refuses it ``too_large`` before upstream allocates anything.

``sample: {max_cells, seed, max_read}`` is the derived replacement of ``get_anndata_donor_balanced``:
donors are keyed by ``(dataset_id, donor_id)`` (``donor_id`` is unique only within a dataset), the cell
budget is split evenly over datasets and then over each dataset's donors (a donor with fewer cells gives
its share back), and cells are drawn per donor with a seeded generator. Only the three key columns are
read, column by column, and only after the count: a filter selecting more than ``max_read`` cells
(default 5,000,000) is not sampled (``sample.reason``). The response lists the sampled ``soma_joinid``
values, the counts per donor and per dataset, and ``value_filter``/``max_cells``: the arguments that make
upstream ``get_anndata_donor_balanced`` (or ``get_anndata``) fetch exactly the sampled cells (upstream
keeps every cell when the filter selects at most ``max_cells``; ``single_cell_mcp/tools.py:1078``).
:func:`donor_balanced` is the pure part.

On the real Census (2025-11-08) the key matters: in ``tissue_general == 'spleen'`` (577,677 primary
cells, 7 datasets) six ``donor_id`` labels (``582C``, ``621B``, ``637C``, ``640C``, ``D496``, ``D503``)
occur in two datasets each: 58 labels, 64 ``(dataset_id, donor_id)`` donors.

:func:`genes_from_h5ad` recomputes ``genes_found``/``genes_not_found`` of a written h5ad from its
``var.feature_name`` column (upstream compares against ``var_names``, which Census writes as positional
digits).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ...ipc import VERB_CENSUS_COUNT, CensusCountRequest
from ...predicate import Predicate, from_json
from .. import ServiceContext, ServiceError, layout_spec

__all__ = ["VERB", "CensusCountRequest", "census_count", "donor_balanced", "donor_balanced_columns",
           "sample_filter", "genes_from_h5ad", "DONOR_KEY", "VERBS"]

VERB = VERB_CENSUS_COUNT
DONOR_KEY = ("dataset_id", "donor_id")
DEFAULT_ROW_BYTES = 200
DEFAULT_VALUE_BYTES = 4                        # float32 X, dense upper bound
DEFAULT_MAX_READ = 5_000_000                   # cells whose key columns a sample may read


def _predicate(req: CensusCountRequest) -> Predicate | None:
    parts: list[Predicate] = []
    if req.value_filter:
        from ...gateway.soma_filter import SomaFilterError, parse

        try:
            parts.append(parse(req.value_filter))
        except SomaFilterError as exc:
            raise ServiceError(f"value_filter: {exc}") from exc
    if req.predicate:
        parts.append(from_json(req.predicate))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    from ...predicate import And

    return And(tuple(parts))


def donor_balanced(rows: Iterable[Mapping[str, Any]], max_cells: int, *, seed: int = 0,
                   key: Sequence[str] = DONOR_KEY, id_column: str = "soma_joinid") -> dict[str, Any]:
    """Sample at most ``max_cells`` cells balanced over datasets, then over donors within each dataset."""
    rows = list(rows)
    return donor_balanced_columns([r.get(id_column) for r in rows], [r.get(key[0]) for r in rows],
                                  [r.get(key[1]) for r in rows], max_cells, seed=seed, key=key)


def donor_balanced_columns(ids: Sequence[Any], datasets: Sequence[Any], donors: Sequence[Any], max_cells: int, *,
                           seed: int = 0, key: Sequence[str] = DONOR_KEY) -> dict[str, Any]:
    """:func:`donor_balanced` over three parallel columns (the cell id, ``key[0]`` and ``key[1]``)."""
    by_dataset: dict[Any, dict[Any, list[Any]]] = {}
    for cid, ds, donor in zip(ids, datasets, donors):
        by_dataset.setdefault(ds, {}).setdefault(donor, []).append(cid)
    rng = random.Random(seed)
    picked: list[Any] = []
    per_donor: dict[str, int] = {}
    per_dataset: dict[str, int] = {}
    datasets_sorted = sorted(by_dataset, key=lambda d: str(d))
    remaining = max(0, int(max_cells))
    for i, ds in enumerate(datasets_sorted):
        share = remaining // (len(datasets_sorted) - i)
        cells = by_dataset[ds]
        names = sorted(cells, key=lambda d: str(d))
        alloc: dict[Any, int] = {d: 0 for d in names}
        left = min(share, sum(len(v) for v in cells.values()))
        # water-fill: equal shares, a small donor's unused share goes to the others
        while left > 0:
            open_ = [d for d in names if alloc[d] < len(cells[d])]
            if not open_:
                break
            each = max(1, left // len(open_))
            for d in open_:
                give = min(each, len(cells[d]) - alloc[d], left)
                alloc[d] += give
                left -= give
                if left <= 0:
                    break
        taken = 0
        for d in names:
            pool = sorted(cells[d], key=lambda x: (str(type(x)), x))
            n = alloc[d]
            chosen = pool if n >= len(pool) else rng.sample(pool, n)
            picked.extend(chosen)
            if n:
                per_donor[f"{ds}/{d}"] = n
            taken += n
        per_dataset[str(ds)] = taken
        remaining -= taken
    return {"soma_joinids": sorted(picked, key=lambda x: (str(type(x)), x)), "per_donor": per_donor,
            "per_dataset": per_dataset, "donor_key": list(key), "n_sampled": len(picked),
            "n_donors": sum(len(v) for v in by_dataset.values()), "n_datasets": len(by_dataset)}


def sample_filter(joinids: Sequence[Any]) -> str:
    """The SOMA ``value_filter`` selecting exactly ``joinids`` (``soma_joinid in [...]``)."""
    return "soma_joinid in [" + ", ".join(str(int(i)) for i in joinids) + "]"


def genes_from_h5ad(path: str, requested: Sequence[str], *, column: str = "feature_name") -> dict[str, Any]:
    """``{genes_found, genes_not_found}`` of ``requested`` against the h5ad's ``var[column]`` (h5py only)."""
    import h5py

    with h5py.File(str(path), "r") as f:
        var = f["var"]
        ds = var[column]
        if hasattr(ds, "keys") and "categories" in ds:          # categorical encoding
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ds["categories"][()]]
            codes = ds["codes"][()]
            values = [cats[int(c)] if int(c) >= 0 else None for c in codes]
        else:
            values = [v.decode() if isinstance(v, bytes) else str(v) for v in ds[()]]
    present = {v for v in values if v}
    found = [g for g in requested if g in present]
    return {"genes_found": found, "genes_not_found": [g for g in requested if g not in present],
            "checked_against": f"var.{column}"}


def census_count(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    if "request" in payload and len(payload) == 1:
        payload = dict(payload["request"])
    req = CensusCountRequest.model_validate({"table": "cellxgene_census.obs", **payload})
    if req.genes_file:
        # the written file's genes, by feature_name (the gateway replaces upstream's var_names comparison)
        try:
            return {"table": req.table, **genes_from_h5ad(req.genes_file, list(req.genes or []))}
        except (OSError, KeyError, ImportError) as exc:
            return {"table": req.table, "reason": f"genes not checked: {exc}"[:500]}
    t = ctx.table(req.table)
    layout = ctx.plugin("layout", t.layout)
    if "count" not in (getattr(layout, "capabilities", ()) or ()):
        raise ServiceError(f"{req.table} has no remote count (layout {t.layout!r})")
    predicate = _predicate(req)
    lspec = layout_spec(t)
    if hasattr(layout, "record") and layout.record is None:
        from ...plugins.layouts.soma import VERSION_RECORD

        layout.record = str(Path(ctx.settings.cache_dir) / t.physical.source / VERSION_RECORD)
    describe = getattr(layout, "describe_read", None)
    sent = describe(lspec, predicate) if callable(describe) else {}
    release = layout.resolve(lspec) if callable(getattr(layout, "resolve", None)) else None
    out: dict[str, Any] = {"table": req.table, "value_filter": sent.get("value_filter"),
                           "release": dict(release) if release is not None else None}
    if req.release_only:
        return out
    try:
        n = layout.count(lspec, predicate=predicate, budget=t.descriptor.budget)
    except Exception as exc:  # noqa: BLE001 - an outage is not a count of zero
        return {**out, "n_cells": None, "admissible": None, "reason": f"count failed: {exc}"[:500]}
    if n is None:
        return {**out, "n_cells": None, "admissible": None,
                "reason": "the filter cannot be compiled into one SOMA value_filter"}
    sample_max = int((req.sample or {}).get("max_cells") or req.max_cells or n) if req.sample else None
    cells = n if req.max_cells is None else min(n, int(req.max_cells))
    if sample_max is not None:
        cells = min(n, sample_max)
    genes = int(req.n_genes or 0)
    row_bytes = int(req.row_bytes or DEFAULT_ROW_BYTES)
    value_bytes = float(req.value_bytes if req.value_bytes is not None else DEFAULT_VALUE_BYTES)
    need = int(cells * (row_bytes + genes * value_bytes))
    cap = req.cap_bytes
    out.update({"n_cells": int(n), "n_cells_pulled": int(cells), "n_genes": genes, "need_bytes": need,
                "cap_bytes": cap, "admissible": None if cap is None else need <= int(cap),
                "truncated_by_max_cells": req.max_cells is not None and n > int(req.max_cells)})
    if req.sample:
        out["sample"] = _sample(layout, lspec, predicate, t, n, dict(req.sample), sample_max or n)
    return out


def _sample(layout: Any, lspec: Any, predicate: Predicate | None, t: Any, n: int, spec: Mapping[str, Any],
            max_cells: int) -> dict[str, Any]:
    """The donor-balanced sample of the ``n`` counted cells (module docstring)."""
    max_read = int(spec.get("max_read") or DEFAULT_MAX_READ)
    seed = int(spec.get("seed") or 0)
    if n > max_read:
        return {"reason": f"the filter selects {n} cells; a sample reads the key columns of at most {max_read}: "
                          "narrow value_filter", "n_sampled": None}
    cols = ["soma_joinid", *DONOR_KEY]
    try:
        if callable(getattr(layout, "columns", None)):
            got = layout.columns(lspec, predicate=predicate, columns=cols)
            out = donor_balanced_columns(got["soma_joinid"], got[DONOR_KEY[0]], got[DONOR_KEY[1]], max_cells,
                                         seed=seed)
        else:
            page = layout.request(lspec, predicate=predicate, projection=cols, page_token=None,
                                  budget=t.descriptor.budget)
            out = donor_balanced(page.rows, max_cells, seed=seed)
    except Exception as exc:  # noqa: BLE001 - a failed read is no sample, never an empty one
        return {"reason": f"sample failed: {exc}"[:500], "n_sampled": None}
    out["seed"] = seed
    out["value_filter"] = sample_filter(out["soma_joinids"])
    out["max_cells"] = out["n_sampled"]
    return out


VERBS = {VERB: census_count}
