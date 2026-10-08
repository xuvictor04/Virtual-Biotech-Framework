"""``_census_count``: count-first admission of Census pulls and donor-balanced sampling (phase 4, F20).

Upstream ``get_anndata``, ``get_expression_for_genes`` and ``get_anndata_donor_balanced`` fetch first
and check the size after (``single_cell_mcp/tools.py:483-490``). This verb counts first: the obs filter
(a SOMA ``value_filter`` or the predicate IR) is compiled by the ``soma`` format and counted by the
``soma`` layout (one read of ``soma_joinid``), against the release ``stable`` resolves to (recorded with
its ``release_confidence`` and any drift). The estimate (:func:`estimate_bytes`) is the server's footprint of
any pull (``base_bytes``) plus, per pulled cell, its row and its genes' values (``row_bytes + genes x
value_bytes``; every gene of the cell, ``all_genes_cell_bytes``, when the call names none), plus
``read_all_bytes`` per cell whose metadata the server reads before pulling (every matching cell when the server
draws its own sample). The overlay's ``count_first.estimate`` carries values calibrated on real pulls; over
``cap_bytes`` the pull is not admissible and the gateway refuses it ``too_large`` before upstream allocates
anything.

``sample: {max_cells, seed, max_read, key, stratify}`` is the derived replacement of
``get_anndata_donor_balanced``. Donors are keyed by ``key`` (default ``(dataset_id, donor_id)``: ``donor_id``
is unique only within a dataset). With ``stratify: cell_type`` the draw is upstream's own
(:func:`stratified_sample`: the cells are allocated over cell types, then balanced over each type's donors,
with ``numpy.random.RandomState(seed)``, default 42 as upstream) on a frame read as upstream reads obs
(:meth:`SomaLayout.frame`), so with one dataset it picks upstream's cells and across datasets it keeps donors
that share a label apart. Without ``stratify`` (the first version, :func:`donor_balanced`) the budget is split
evenly over datasets and then over each dataset's donors (a donor with fewer cells gives its share back).
Only the key columns (and the stratum) are read, and only after the count: a filter selecting more than
``max_read`` cells (default 5,000,000) is not sampled (``sample.reason``). The response lists the sampled
``soma_joinid`` values, the counts per donor, dataset (and stratum), and ``value_filter``/``max_cells``: the
arguments that make upstream ``get_anndata_donor_balanced`` (or ``get_anndata``) fetch exactly the sampled
cells (upstream keeps every cell when the filter selects at most ``max_cells``;
``single_cell_mcp/tools.py:1078``). On the real Census (2025-11-08) a 200,000-cell spleen sample's filter is
2,104,665 characters; counting it took 4.7 s and upstream's obs read of it 6.3-8.1 s (S1, 2026-10-08).

On the real Census (2025-11-08) the key matters: in ``tissue_general == 'spleen'`` (577,677 primary
cells, 7 datasets) six ``donor_id`` labels (``582C``, ``621B``, ``637C``, ``640C``, ``D496``, ``D503``)
occur in two datasets each: 58 labels, 64 ``(dataset_id, donor_id)`` donors.

:func:`genes_from_h5ad` recomputes ``genes_found``/``genes_not_found`` of a written h5ad from its
``var.feature_name`` column (upstream compares against ``var_names``, which Census writes as positional
digits). ``cells_file`` (with ``cells_column``) answers ``file_cells``, the ``obs[column]`` values of a written
h5ad (:func:`cells_from_h5ad`): the gateway checks that a pull served as the derived sample wrote exactly the drawn
cells (``obs.soma_joinid``; the file's ``var`` holds a ``soma_joinid`` column too, the genes').

``_release`` (:func:`release`) answers only what the source says about the data a call reads: the dated release
a moving alias names, or a live source's release, versions and per-record releases (:func:`remote_release`).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ...ipc import VERB_CENSUS_COUNT, VERB_RELEASE, CensusCountRequest, ReleaseRequest
from ...predicate import Predicate, from_json
from .. import ServiceContext, ServiceError, layout_spec

__all__ = ["VERB", "CensusCountRequest", "census_count", "release", "estimate_bytes", "donor_balanced",
           "donor_balanced_columns", "stratified_sample", "sample_filter", "genes_from_h5ad", "cells_from_h5ad",
           "DONOR_KEY", "VERBS"]

VERB = VERB_CENSUS_COUNT
DONOR_KEY = ("dataset_id", "donor_id")
DEFAULT_ROW_BYTES = 200
DEFAULT_VALUE_BYTES = 4                        # float32 X, dense upper bound
DEFAULT_MAX_READ = 5_000_000                   # cells whose key columns a sample may read
UPSTREAM_SEED = 42                             # upstream's generator: np.random.RandomState(42) (tools.py:1073)
STRATUM = "cell_type"                          # upstream allocates the cells over cell types first


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


def _donor_categorical(frame: Any, key: Sequence[str]) -> Any:
    """``(dataset_id, donor_id)`` as one categorical labelled ``"<dataset_id>/<donor_id>"``. Its categories run in
    the order of the donor categories, then the dataset categories: with one dataset they are upstream's donor
    categories one for one, so value_counts orders donors (ties included) exactly as upstream does."""
    import numpy as np
    import pandas as pd

    def cat(s: Any) -> Any:
        return s.cat.remove_unused_categories() if hasattr(s, "cat") else pd.Series(pd.Categorical(s), index=s.index)

    ds, dn = cat(frame[key[0]]), cat(frame[key[1]])
    ds_codes = ds.cat.codes.to_numpy(dtype="int64")
    dn_codes = dn.cat.codes.to_numpy(dtype="int64")
    n_ds = max(len(ds.cat.categories), 1)
    combo = dn_codes * n_ds + ds_codes
    uniq = np.unique(combo)
    labels = [f"{ds.cat.categories[c % n_ds]}/{dn.cat.categories[c // n_ds]}" for c in uniq]
    return pd.Categorical.from_codes(np.searchsorted(uniq, combo), categories=labels)


def stratified_sample(frame: Any, max_cells: int, *, seed: int = UPSTREAM_SEED, stratum: str = STRATUM,
                      key: Sequence[str] = DONOR_KEY, id_column: str = "soma_joinid") -> dict[str, Any]:
    """Upstream's donor-balanced, cell-type-stratified sample (``single_cell_mcp/tools.py:1073-1124``) with donors
    keyed by ``(dataset_id, donor_id)``: every cell when the filter selects at most ``max_cells``; otherwise each
    cell type gets ``min(20, max_cells // types)`` cells plus its share of the rest in proportion to its cells,
    split evenly over its donors (a donor with fewer cells gives the remainder to a draw from the type's other
    cells), all drawn with ``numpy.random.RandomState(seed)`` in upstream's order. ``frame`` holds ``id_column``,
    ``stratum`` and the ``key`` columns (a pandas frame read as upstream reads obs, or a mapping of columns).
    With one dataset the draw is upstream's own; across datasets, donors that share a label stay apart (SC-001)."""
    import numpy as np
    import pandas as pd

    df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(dict(frame))
    for col in (stratum, *key):
        if hasattr(df[col], "cat"):
            df[col] = df[col].cat.remove_unused_categories()
    donors_all = _donor_categorical(df, key)
    n_total = len(df)
    rng = np.random.RandomState(seed)
    ids = df[id_column].to_numpy()
    if n_total <= max_cells:
        selected = ids
    else:
        donor_col = pd.Series(donors_all, index=df.index)
        ct_counts = df[stratum].value_counts()
        ct_counts = ct_counts[ct_counts > 0]
        n_types = len(ct_counts)
        min_per_type = min(20, max_cells // max(n_types, 1))
        remaining = max(0, max_cells - min_per_type * n_types)
        picked: list[Any] = []
        for ct, ct_count in ct_counts.items():
            in_ct = (df[stratum] == ct).to_numpy()
            ct_ids = ids[in_ct]
            ct_donor = donor_col[in_ct]
            n_proportional = int(remaining * (ct_count / n_total))
            budget = min(ct_count, min_per_type + n_proportional)
            budget = max(budget, min(ct_count, min_per_type))
            donors = ct_donor.value_counts()
            donors = donors[donors > 0]
            if len(donors) == 0 or budget == 0:
                continue
            per_donor = max(1, budget // len(donors))
            parts = []
            for donor, _n in donors.items():
                pool = ct_ids[(ct_donor == donor).to_numpy()]
                parts.append(rng.choice(pool, size=min(len(pool), per_donor), replace=False))
            combined = np.concatenate(parts)
            if len(combined) < budget:
                rest = np.setdiff1d(ct_ids, combined)
                n_extra = min(budget - len(combined), len(rest))
                if n_extra > 0:
                    combined = np.concatenate([combined, rng.choice(rest, size=n_extra, replace=False)])
            if len(combined) > budget:
                combined = rng.choice(combined, size=budget, replace=False)
            picked.append(combined)
        selected = np.concatenate(picked) if picked else ids[:0]
        if len(selected) > max_cells:
            selected = rng.choice(selected, size=max_cells, replace=False)
    chosen = set(int(i) for i in selected)
    mask = np.fromiter((int(i) in chosen for i in ids), dtype=bool, count=len(ids))
    per_donor = {str(k): int(v) for k, v in pd.Series(donors_all[mask]).value_counts(sort=False).items() if v}
    per_dataset = {str(k): int(v) for k, v in df[key[0]][mask].value_counts(sort=False).items() if v}
    per_stratum = {str(k): int(v) for k, v in df[stratum][mask].value_counts(sort=False).items() if v}
    return {"soma_joinids": sorted(chosen), "n_sampled": len(chosen), "n_total": n_total,
            "per_donor": per_donor, "per_dataset": per_dataset, "per_stratum": per_stratum,
            "n_donors": int(len(set(donors_all))), "n_donors_sampled": len(per_donor),
            "n_datasets": int(df[key[0]].nunique()), "n_strata": int(df[stratum].nunique()),
            "donor_key": list(key), "stratified_by": stratum, "seed": seed,
            "method": "donor_balanced_cell_type_stratified (upstream's), donors keyed by (dataset_id, donor_id)"}


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


def cells_from_h5ad(path: str, column: str = "soma_joinid") -> list[int]:
    """The ``obs[column]`` values of a written h5ad (h5py only): the cells the file holds."""
    import h5py

    with h5py.File(str(path), "r") as f:
        return [int(v) for v in f["obs"][column][()]]


def remote_release(ctx: ServiceContext, t: Any, layout: Any, lspec: Any, predicate: Predicate | None
                   ) -> dict[str, Any] | None:
    """What a live table's source says about the data a call reads (``release_only`` on a ``live_api`` table, for an
    upstream-served call): ``resolved`` (the data release, CT.gov ``dataTimestamp``), ``versions`` (the API and
    software versions the layout's ``release.versions`` names: CT.gov ``apiVersion``, cBioPortal ``portalVersion``
    and ``dbVersion``, the PubMed build) and ``per`` (``release.per``: the importDate of each cBioPortal study the
    predicate names, at most five; one study is the call's release). Each is cached for ``RELEASE_TTL_S``; a failed
    request leaves it out (``reason``), never the call."""
    from ...plugins.layouts.live_api import RELEASE_TTL_S, Budget
    from .witness import _per_record_releases

    out: dict[str, Any] = {}
    reasons: list[str] = []
    info = getattr(layout, "release_info", None)
    if callable(info):
        got: dict[str, Any] = {}
        try:
            got = dict(info(lspec, Budget.of(t.descriptor.budget), max_age_s=RELEASE_TTL_S) or {})
        except Exception as exc:  # noqa: BLE001 - provenance only
            reasons.append(f"release request failed: {exc}"[:300])
        if got.get("release"):
            out.update(requested=None, resolved=str(got["release"]), release_confidence="exact")
        if got.get("versions"):
            out["versions"] = dict(got["versions"])
    try:
        ref, per = _per_record_releases(ctx, t, [], predicate)
    except Exception as exc:  # noqa: BLE001 - provenance only
        ref, per = None, {}
        reasons.append(f"record releases not read: {exc}"[:300])
    if ref and per:
        out["per"] = {ref: dict(per)}
        if len(per) == 1 and not out.get("resolved"):
            out.update(requested=None, resolved=next(iter(per.values())), release_confidence="per_record")
    if reasons:
        out["reason"] = "; ".join(reasons)
    return out or None


def _open(ctx: ServiceContext, table: str, value_filter: str | None, predicate: dict[str, Any] | None
          ) -> tuple[Any, Any, Any, Predicate | None, dict[str, Any], dict[str, Any] | None]:
    """The table, its layout and layout spec, the request's predicate, what the layout would send for it and the
    release a moving alias names (None for a layout without one)."""
    t = ctx.table(table)
    layout = ctx.plugin("layout", t.layout)
    predicate_ = _predicate(CensusCountRequest(table=table, value_filter=value_filter, predicate=predicate))
    lspec = layout_spec(t)
    if hasattr(layout, "record") and layout.record is None:
        from ...plugins.layouts.soma import VERSION_RECORD

        layout.record = str(Path(ctx.settings.cache_dir) / t.physical.source / VERSION_RECORD)
    describe = getattr(layout, "describe_read", None)
    sent = describe(lspec, predicate_) if callable(describe) else {}
    release = layout.resolve(lspec) if callable(getattr(layout, "resolve", None)) else None
    return t, layout, lspec, predicate_, dict(sent or {}), dict(release) if release is not None else None


def release(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``_release``: what a remote table's source says about the data a call reads (no count): the dated release
    a moving alias names (Census ``stable``), else a live source's release, versions and per-record releases
    (:func:`remote_release`)."""
    payload = dict(payload)
    if "request" in payload and len(payload) == 1:
        payload = dict(payload["request"])
    req = ReleaseRequest.model_validate(payload)
    t, layout, lspec, predicate, sent, rel = _open(ctx, req.table, req.value_filter, req.predicate)
    if rel is None:
        rel = remote_release(ctx, t, layout, lspec, predicate)
    return {"table": req.table, "value_filter": sent.get("value_filter"), "release": rel}


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
    if req.cells_file:
        # the cells a written file holds: the gateway checks a derived sample's file against the drawn cells
        try:
            return {"table": req.table, "file_cells": cells_from_h5ad(req.cells_file, req.cells_column)}
        except (OSError, KeyError, ImportError, TypeError, ValueError) as exc:
            return {"table": req.table, "reason": f"cells not read: {exc}"[:500]}
    t, layout, lspec, predicate, sent, rel = _open(ctx, req.table, req.value_filter, req.predicate)
    if "count" not in (getattr(layout, "capabilities", ()) or ()):
        raise ServiceError(f"{req.table} has no remote count (layout {t.layout!r})")
    out: dict[str, Any] = {"table": req.table, "value_filter": sent.get("value_filter"), "release": rel}
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
    sample = _sample(layout, lspec, predicate, t, n, dict(req.sample), sample_max or n) if req.sample else None
    # the server reads the metadata of every matching cell before it pulls (it draws its own sample), unless it
    # is asked for exactly the sample's cells
    read = cells if (sample or {}).get("soma_joinids") else n
    need = estimate_bytes(req, cells, read)
    cap = req.cap_bytes
    out.update({"n_cells": int(n), "n_cells_pulled": int(cells), "n_genes": req.n_genes, "need_bytes": need,
                "cap_bytes": cap, "admissible": None if cap is None else need <= int(cap),
                "truncated_by_max_cells": req.max_cells is not None and n > int(req.max_cells)})
    if sample is not None:
        out["sample"] = sample
    return out


def estimate_bytes(req: CensusCountRequest, cells: int, read: int) -> int:
    """The memory a pull of ``cells`` cells needs (``CensusCountRequest``): ``base_bytes``, then per pulled cell
    ``row_bytes`` plus the named genes' values (or ``all_genes_cell_bytes`` when no gene is named), plus
    ``read_all_bytes`` per cell whose metadata the server reads first (``read``)."""
    row_bytes = int(req.row_bytes if req.row_bytes is not None else DEFAULT_ROW_BYTES)
    value_bytes = float(req.value_bytes if req.value_bytes is not None else DEFAULT_VALUE_BYTES)
    if req.n_genes is None and req.all_genes_cell_bytes is not None:
        per_cell = row_bytes + int(req.all_genes_cell_bytes)
    else:
        per_cell = row_bytes + int(req.n_genes or 0) * value_bytes
    return int(int(req.base_bytes or 0) + cells * per_cell + read * int(req.read_all_bytes or 0))


def _sample(layout: Any, lspec: Any, predicate: Predicate | None, t: Any, n: int, spec: Mapping[str, Any],
            max_cells: int) -> dict[str, Any]:
    """The donor-balanced sample of the ``n`` counted cells (module docstring)."""
    max_read = int(spec.get("max_read") or DEFAULT_MAX_READ)
    stratum = spec.get("stratify")
    seed = int(spec["seed"]) if spec.get("seed") is not None else (UPSTREAM_SEED if stratum else 0)
    if n > max_read:
        return {"reason": f"the filter selects {n} cells; a sample reads the key columns of at most {max_read}: "
                          "narrow value_filter", "n_sampled": None}
    key = tuple(str(k) for k in (spec.get("key") or DONOR_KEY))
    if len(key) != 2:
        return {"reason": f"a donor key has two columns (the qualifier and the donor), not {list(key)}", "n_sampled": None}
    cols = ["soma_joinid", *key]
    try:
        if stratum:
            if not callable(getattr(layout, "frame", None)):
                return {"reason": f"layout {t.layout!r} cannot read a frame for a stratified sample", "n_sampled": None}
            frame = layout.frame(lspec, predicate=predicate, columns=list(dict.fromkeys([*cols, str(stratum)])))
            out = stratified_sample(frame, max_cells, seed=seed, stratum=str(stratum), key=key)
            if out["n_total"] != n:
                # the count and the read must see the same cells (a moved alias, a failed residual)
                return {"reason": f"the sample read {out['n_total']} cells where the count found {n}", "n_sampled": None}
        elif callable(getattr(layout, "columns", None)):
            got = layout.columns(lspec, predicate=predicate, columns=cols)
            out = donor_balanced_columns(got["soma_joinid"], got[key[0]], got[key[1]], max_cells, seed=seed, key=key)
        else:
            page = layout.request(lspec, predicate=predicate, projection=cols, page_token=None,
                                  budget=t.descriptor.budget)
            out = donor_balanced(page.rows, max_cells, seed=seed, key=key)
    except Exception as exc:  # noqa: BLE001 - a failed read is no sample, never an empty one
        return {"reason": f"sample failed: {exc}"[:500], "n_sampled": None}
    out["seed"] = seed
    out["value_filter"] = sample_filter(out["soma_joinids"])
    out["max_cells"] = out["n_sampled"]
    return out


VERBS = {VERB: census_count, VERB_RELEASE: release}
