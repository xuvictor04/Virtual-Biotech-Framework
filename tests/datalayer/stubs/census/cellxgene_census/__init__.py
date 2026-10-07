"""A network-free ``cellxgene_census`` for the single_cell defect detectors (imported by the unmodified
upstream ``single_cell_mcp.tools`` when ``stubs/census`` is first on ``sys.path``).

``VBT_CENSUS_DATA`` names a JSON file ``{"obs": [rows], "var": [rows]}``; ``value_filter`` strings are
applied with ``pandas.DataFrame.query`` (the SOMA subset upstream uses: ``==``, ``<``, ``in``, ``and``).
Every read and ``get_anndata`` call is appended to ``VBT_CENSUS_LOG`` (JSONL).
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

__all__ = ["open_soma", "get_anndata"]


def _log(kind: str, **data: Any) -> None:
    path = os.environ.get("VBT_CENSUS_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": kind, **data}, default=str) + "\n")


def _data() -> dict[str, pd.DataFrame]:
    with open(os.environ["VBT_CENSUS_DATA"], encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: pd.DataFrame(raw.get(k) or []) for k in ("obs", "var")}


def _filter(df: pd.DataFrame, value_filter: str | None) -> pd.DataFrame:
    return df.query(value_filter) if value_filter else df


class _Table:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)


class _Read:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def concat(self) -> _Table:
        return _Table(self._df)


class _Soma:
    def __init__(self, name: str, df: pd.DataFrame) -> None:
        self.name = name
        self._df = df
        self.count = len(df)

    def read(self, column_names: list[str] | None = None, value_filter: str | None = None, **_: Any) -> _Read:
        _log("read", table=self.name, value_filter=value_filter, column_names=column_names)
        df = _filter(self._df, value_filter)
        cols = [c for c in (column_names or list(df.columns)) if c in df.columns]
        return _Read(df[cols])


class _Node(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def open_soma(census_version: str | None = None, **_: Any) -> _Node:
    d = _data()
    human = _Node(obs=_Soma("obs", d["obs"]), ms=_Node(RNA=_Node(var=_Soma("var", d["var"]))))
    return _Node(census_data=_Node(homo_sapiens=human))


def get_anndata(census: Any = None, organism: str | None = None, measurement_name: str = "RNA", X_name: str = "raw",
                obs_value_filter: str | None = None, var_value_filter: str | None = None,
                obs_coords: list[int] | None = None, column_names: dict[str, list[str]] | None = None,
                **_: Any) -> Any:
    import anndata
    import numpy as np

    _log("get_anndata", obs_value_filter=obs_value_filter, var_value_filter=var_value_filter, obs_coords=obs_coords)
    d = _data()
    obs = _filter(d["obs"], obs_value_filter)
    if obs_coords is not None:
        obs = obs[obs["soma_joinid"].isin(obs_coords)]
    var = _filter(d["var"], var_value_filter)
    obs = obs.copy()
    obs.index = pd.Index(obs["soma_joinid"].astype(str).tolist(), name="obs_id")
    var = var.copy()
    var.index = pd.Index(var["feature_id"].astype(str).tolist(), name="var_id")
    return anndata.AnnData(X=np.zeros((len(obs), len(var)), dtype=np.float32), obs=obs, var=var)
