"""The paper's Case 1 replication as a ``vbt validate`` extension (ASN-6: paper-replication logic is not part of the
core command). ``configs/default.yaml`` lists this module under ``validate.extensions``; importing it registers the
``replication`` step, which runs the Case 1 statistics on the authors' inputs when the Zenodo archive is present and
compares them row by row with the authors' tables (``--replicate quick|full|off``)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ...validate import register_step
from ...validate.report import FAIL, PASS, WARN, StepResult

NAME = "replication"
TITLE = "Paper replication (Case 1 on the Zenodo archive)"


def step_replication(ctx: Any) -> StepResult:
    mode = getattr(ctx.opts, "replicate", "quick")
    if mode == "off":
        return StepResult.skipped(NAME, TITLE, "--replicate off")
    try:
        from ...data.zenodo import zenodo_root

        root = zenodo_root(ctx.config)
    except Exception as exc:  # noqa: BLE001
        return StepResult.skipped(NAME, TITLE, f"no Zenodo root ({exc})")
    if not root or not Path(root, "clinical_trials").is_dir():
        return StepResult.skipped(NAME, TITLE, f"the Zenodo archive is not at {root} (VBT_ZENODO_DIR; "
                                  "`vbt data zenodo fetch --preset case1`)")
    from .replicate import replicate_case1

    quick = mode != "full"
    t0 = time.monotonic()
    try:
        rep = replicate_case1(root, ctx.out / "case1_replication", n_perm=0 if quick else 1000,
                              gene_perm=0 if quick else 200, mixed=not quick, expr=not quick, progress=None)
    except FileNotFoundError as exc:
        return StepResult.skipped(NAME, TITLE, f"the archive lacks a Case 1 input: {exc}")
    agree = rep.agreement()
    rows = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in agree.to_dict("records")]
    rows_total = int(agree["rows"].sum()) if len(agree) else 0
    matched = int(agree["matched"].sum()) if len(agree) else 0
    status = PASS if rows_total and matched == rows_total else (WARN if matched else FAIL)
    return StepResult(NAME, TITLE, status, f"{matched} of {rows_total} compared rows match the authors' "
                      f"tables ({'quick: no permutations, GLMMs or expression models' if quick else 'full'})",
                      rows=rows, seconds=time.monotonic() - t0, details={"zenodo_root": str(root)})


register_step(NAME, TITLE, step_replication)
