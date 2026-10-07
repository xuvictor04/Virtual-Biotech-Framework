"""Files a tool writes: output-path confinement, write-once outputs and ``FileCheckSpec``
reconciliation (§8.1, §11.3 step 4, §11.4 step 6; rev 2, phase 1). No pyarrow.

* :func:`confine` rejects absolute and ``..`` output paths (``invalid_argument``), so a tool can
  only write under the run's output directory.
* :func:`write_once` keeps an existing output: it is renamed to ``<stem>.<token>.<ext>`` before the
  call writes the same name, and the rename is disclosed, so a file a claim cites is never
  overwritten in place.
* :func:`reconcile` checks a returned file against ``ResultSpec.files``: it exists, its header
  agrees with the payload (``echo_checks: {n_obs: $.n_cells}``), the declared key columns are
  present, and the var index is neither positional digits nor ``-N`` uniquified names. The h5ad
  header is read with h5py when it is installed (never pandas or anndata); a check that cannot be
  made is recorded with ``ok: None``, never as passed.
* :class:`MaterializedRegistry` records files that tables declare ``materialized_by`` a tool
  (fragment, sha256, producing provenance id).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..errors import ErrorKind, GatewayError, invalid_argument_payload
from .fields import jp_first

__all__ = ["FileCheck", "confine", "write_once", "file_token", "file_sha256", "read_h5ad_header", "reconcile",
           "MaterializedRegistry", "positional_index", "uniquified_names"]

_SUFFIX = re.compile(r"^(.+)-(\d+)$")


@dataclass
class FileCheck:
    name: str
    ok: bool | None
    detail: str = ""


def confine(value: Any, output_dir: str | Path | None, *, argument: str, tool: str | None = None) -> Path:
    """The output path for ``value`` under ``output_dir``; absolute paths and ``..`` components are
    ``invalid_argument``."""
    text = str(value or "").strip()
    p = Path(text)
    if not text or p.is_absolute() or text.startswith("~") or any(part == ".." for part in p.parts):
        raise GatewayError(ErrorKind.invalid_argument,
                           f"{argument} must be a relative path inside the run's output directory",
                           tool=tool, argument=argument, value=text,
                           payload={**invalid_argument_payload(argument, text, None),
                                    "reason": "output paths are relative and may not leave the output directory"})
    base = Path(output_dir) if output_dir else Path(".")
    target = (base / p).resolve() if output_dir else p
    if output_dir:
        root = Path(output_dir).resolve()
        if root != target and root not in target.parents:
            raise GatewayError(ErrorKind.invalid_argument, f"{argument} leaves the output directory",
                               tool=tool, argument=argument, value=text,
                               payload=invalid_argument_payload(argument, text, None))
    return target


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_token(path: str | Path) -> str:
    """A short content token for a renamed output (``sha256[:12]``)."""
    return file_sha256(path)[:12]


def write_once(path: str | Path, token: str | None = None) -> Path | None:
    """Rename an existing ``path`` to ``<stem>.<token><suffixes>`` (the token defaults to the
    file's content hash; the producing provenance id when known). Returns the new path, or None
    when nothing existed."""
    p = Path(path)
    if not p.exists():
        return None
    tok = token or file_token(p)
    suffixes = "".join(p.suffixes)
    stem = p.name[: -len(suffixes)] if suffixes else p.name
    target = p.with_name(f"{stem}.{tok}{suffixes}")
    n = 1
    while target.exists():
        target = p.with_name(f"{stem}.{tok}-{n}{suffixes}")
        n += 1
    p.rename(target)
    return target


def _decode(values: Any) -> list[str]:
    out = []
    for v in list(values):
        out.append(v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v))
    return out


def read_h5ad_header(path: str | Path, max_index: int = 200_000) -> dict[str, Any] | None:
    """``{n_obs, n_vars, obs_columns, var_index}`` from an h5ad file's HDF5 groups (h5py only,
    imported lazily). None when h5py is missing or the layout is not the group encoding."""
    try:
        import h5py  # noqa: PLC0415 - optional, never pandas/anndata
    except Exception:  # noqa: BLE001
        return None
    try:
        with h5py.File(str(path), "r") as f:
            out: dict[str, Any] = {}
            for axis, n_key in (("obs", "n_obs"), ("var", "n_vars")):
                grp = f.get(axis)
                if grp is None or not hasattr(grp, "attrs") or not hasattr(grp, "keys"):
                    return None
                index_name = grp.attrs.get("_index", "_index")
                if isinstance(index_name, bytes):
                    index_name = index_name.decode()
                idx = grp.get(index_name)
                if idx is None:
                    return None
                out[n_key] = int(idx.shape[0])
                if axis == "obs":
                    order = grp.attrs.get("column-order", [])
                    out["obs_columns"] = _decode(order) if len(order) else [k for k in grp.keys() if k != index_name]
                else:
                    out["var_index"] = _decode(idx[: min(int(idx.shape[0]), max_index)])
            return out
    except Exception:  # noqa: BLE001 - an unreadable header is an unmade check
        return None


def positional_index(values: Sequence[str]) -> bool:
    """True when the index is ``0, 1, 2, ...`` as strings (a lost var index)."""
    vals = list(values)
    return bool(vals) and all(v.isdigit() for v in vals) and vals[: min(len(vals), 50)] == \
        [str(i) for i in range(min(len(vals), 50))]


def uniquified_names(values: Sequence[str]) -> list[str]:
    """Names that look ``-N`` uniquified (``GENE-1`` beside ``GENE``)."""
    names = set(values)
    out = []
    for v in values:
        m = _SUFFIX.match(v)
        if m and m.group(1) in names:
            out.append(v)
    return out


def reconcile(specs: Sequence[Any], obj: Any, *, output_dir: str | Path | None = None,
              header_reader: Callable[[Path], Mapping[str, Any] | None] = read_h5ad_header) -> list[FileCheck]:
    """Check returned files against ``ResultSpec.files`` (see the module docstring)."""
    checks: list[FileCheck] = []
    for spec in specs:
        raw = jp_first(obj, spec.path_from)
        if raw in (None, ""):
            if spec.must_exist:
                checks.append(FileCheck("file_exists", False, f"no path at {spec.path_from}"))
            continue
        path = Path(str(raw))
        if not path.is_absolute() and output_dir:
            path = Path(output_dir) / path
        if not path.exists():
            checks.append(FileCheck("file_exists", False if spec.must_exist else None, f"{path} does not exist"))
            continue
        checks.append(FileCheck("file_exists", True, str(path)))
        needs_header = bool(spec.echo_checks or spec.key_columns or spec.forbid_positional_index)
        header = header_reader(path) if needs_header else None
        if needs_header and header is None:
            checks.append(FileCheck("file_header", None, f"the header of {path.name} could not be read"))
            continue
        for field_name, jpath in spec.echo_checks.items():
            want = jp_first(obj, jpath)
            got = (header or {}).get(field_name)
            if want is None or got is None:
                checks.append(FileCheck(f"file_echo:{field_name}", None, f"{field_name} or {jpath} unavailable"))
                continue
            ok = _num_equal(got, want)
            checks.append(FileCheck(f"file_echo:{field_name}", ok, f"file {field_name}={got}, payload {jpath}={want}"))
        if spec.key_columns:
            cols = set((header or {}).get("obs_columns") or [])
            missing = [c for c in spec.key_columns if c not in cols]
            checks.append(FileCheck("file_key_columns", not missing,
                                    f"missing key columns {missing}" if missing else "key columns present"))
        if spec.forbid_positional_index:
            index = list((header or {}).get("var_index") or [])
            if positional_index(index):
                checks.append(FileCheck("file_var_index", False, "the var index is positional digits"))
            else:
                dupes = uniquified_names(index)
                checks.append(FileCheck("file_var_index", not dupes,
                                        f"-N uniquified names: {dupes[:5]}" if dupes else "var index has names"))
    return checks


def _num_equal(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


@dataclass
class MaterializedRegistry:
    """Files registered as fragments of ``materialized_by`` tables during a run."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def register(self, table: str, path: str | Path, *, prov: str | None, tool: str,
                 fragment: str | None = None) -> dict[str, Any]:
        p = Path(path)
        entry = {"table": table, "path": str(p), "fragment": fragment or p.name, "tool": tool, "prov": prov,
                 "sha256": file_sha256(p) if p.is_file() else None}
        self.entries = [e for e in self.entries if e["path"] != entry["path"]] + [entry]
        return entry

    def lookup(self, path: str | Path) -> dict[str, Any] | None:
        text = str(Path(path))
        for e in self.entries:
            if e["path"] == text:
                return e
        return None

    def for_table(self, table: str) -> list[dict[str, Any]]:
        return [e for e in self.entries if e["table"] == table]
