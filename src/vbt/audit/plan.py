"""Plan validation and planned-vs-actual reconciliation (port of upstream plan_runner).

The CSO declares a DAG of steps with ``mcp__provenance__write_plan`` before
dispatching specialists. The DAG is validated on write (missing/duplicate ids,
missing agents, self or dangling dependencies and cycles are rejected); after
each turn the dispatches actually observed in the trace are reconciled against
the latest plan. Deviation is *recorded*, not forbidden: the CSO is expected to
adapt when a specialist's findings change what should happen next.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

STEP_ID_MAX = 32

#: Agents that never count as "unplanned" when they run outside the plan.
DEFAULT_IGNORE_UNPLANNED = frozenset({"cso", "_cso", "chief-of-staff"})


@dataclass
class PlanResult:
    plan: dict[str, Any] | None = None
    order: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        if not self.ok:
            return {"ok": False, "errors": list(self.errors), "warnings": list(self.warnings)}
        return {"ok": True, "n_steps": len((self.plan or {}).get("steps", [])), "order": list(self.order),
                "parallel_groups": [list(g) for g in self.parallel_groups], "warnings": list(self.warnings)}


def _as_list(value: Any) -> list[Any] | None:
    """Coerce a string to a one-item list; None/'' to []; non-lists to None."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
            except ValueError:
                pass
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def validate_plan(steps: Any, goal: str = "", roster: Iterable[str] | None = None) -> PlanResult:
    """Validate an analysis DAG and compute a valid order and parallel groups.

    ``steps`` may be a list of step dicts, a ``{"goal", "steps"}`` dict, or a JSON
    string of either. ``roster`` (agent names) only produces warnings.
    """
    res = PlanResult()
    if isinstance(steps, str):
        try:
            steps = json.loads(steps)
        except ValueError:
            res.errors.append("steps must be a list of step objects (got an unparseable string)")
            return res
    if isinstance(steps, Mapping):
        goal = goal or str(steps.get("goal") or "")
        steps = steps.get("steps", [])
    if not isinstance(steps, list) or not steps:
        res.errors.append("plan must be a non-empty list of steps")
        return res
    roster_set = set(roster) if roster is not None else None

    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(steps):
        where = f"step[{i}]"
        if not isinstance(raw, Mapping):
            res.errors.append(f"{where}: not an object")
            continue
        sid = str(raw.get("id") or "").strip()
        if not sid:
            res.errors.append(f"{where}: missing 'id'")
            continue
        if len(sid) > STEP_ID_MAX:
            res.errors.append(f"{where}: id {sid[:40]!r} too long (max {STEP_ID_MAX})")
            continue
        if sid in seen:
            res.errors.append(f"step {sid}: duplicate id")
            continue
        seen.add(sid)
        agent = str(raw.get("agent") or "").strip()
        if not agent:
            res.errors.append(f"step {sid}: missing 'agent'")
            continue
        if roster_set is not None and agent not in roster_set:
            res.warnings.append(f"step {sid}: agent {agent!r} is not in this run's roster "
                                f"({', '.join(sorted(roster_set)) or 'empty'})")
        task = str(raw.get("task") or "").strip()
        if not task:
            res.warnings.append(f"step {sid}: no 'task' description")
        deps = _as_list(raw.get("depends_on"))
        if deps is None:
            res.errors.append(f"step {sid}: 'depends_on' must be a list of step ids")
            continue
        outs = _as_list(raw.get("expected_outputs"))
        if outs is None:
            res.errors.append(f"step {sid}: 'expected_outputs' must be a list")
            continue
        clean.append({
            "id": sid, "agent": agent, "task": task,
            "depends_on": list(dict.fromkeys(str(d).strip() for d in deps if str(d).strip())),
            "expected_outputs": [str(o) for o in outs if str(o).strip()],
        })
    if res.errors:
        return res

    ids = {s["id"] for s in clean}
    for s in clean:
        for d in s["depends_on"]:
            if d == s["id"]:
                res.errors.append(f"step {s['id']}: depends on itself")
            elif d not in ids:
                res.errors.append(f"step {s['id']}: depends on unknown step {d!r}")
    if res.errors:
        return res

    order = _topo_sort(clean)
    if order is None:
        res.errors.append("plan contains a dependency cycle; no execution order exists. "
                          "Check depends_on: every chain must terminate.")
        return res
    res.order = order
    res.parallel_groups = _parallel_groups(clean, order)
    res.plan = {
        "goal": goal, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps": clean, "valid_order": order, "parallel_groups": res.parallel_groups,
    }
    return res


def _topo_sort(steps: list[dict[str, Any]]) -> list[str] | None:
    """Kahn's algorithm, preserving declared order among ready steps. None on a cycle."""
    indeg = {s["id"]: 0 for s in steps}
    adj: dict[str, list[str]] = {s["id"]: [] for s in steps}
    for s in steps:
        for d in s["depends_on"]:
            adj[d].append(s["id"])
            indeg[s["id"]] += 1
    ready = [s["id"] for s in steps if indeg[s["id"]] == 0]
    out: list[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    return out if len(out) == len(steps) else None


def _parallel_groups(steps: list[dict[str, Any]], order: list[str]) -> list[list[str]]:
    """Steps grouped into waves that may run concurrently."""
    by_id = {s["id"]: s for s in steps}
    depth: dict[str, int] = {}
    for sid in order:
        depth[sid] = 1 + max((depth[d] for d in by_id[sid]["depends_on"]), default=-1)
    groups: dict[int, list[str]] = {}
    for sid in order:
        groups.setdefault(depth[sid], []).append(sid)
    return [groups[k] for k in sorted(groups)]


def _artifact_keys(artifacts: Any) -> set[str]:
    if not artifacts:
        return set()
    if isinstance(artifacts, Mapping):
        return {str(k) for k in artifacts}
    keys = set()
    for a in artifacts:
        if isinstance(a, Mapping) and a.get("path"):
            keys.add(str(a["path"]))
        elif isinstance(a, str):
            keys.add(a)
    return keys


def _output_produced(expected: str, keys: set[str], names: set[str]) -> bool:
    e = expected.strip()
    if not e:
        return True
    norm = e.lstrip("/")
    while norm.startswith("./"):
        norm = norm[2:]
    if norm in keys or any(k.endswith("/" + norm) for k in keys):
        return True
    return PurePosixPath(norm).name in names


def _step_order(steps: list[Mapping[str, Any]], declared: Any) -> list[Any]:
    """The plan's valid_order when it names exactly the steps, else a fresh topological order."""
    ids = [s.get("id") for s in steps]
    if isinstance(declared, list) and sorted(map(str, declared)) == sorted(map(str, ids)) \
            and len(set(ids)) == len(ids):
        return list(declared)
    known = set(ids)
    clean = [{"id": s.get("id"), "depends_on": [d for d in s.get("depends_on") or [] if d in known
                                                and d != s.get("id")]} for s in steps]
    if len(set(ids)) == len(ids):
        order = _topo_sort(clean)
        if order is not None:
            return order
    return ids


def _match_steps(steps: list[Mapping[str, Any]], actual: list[str], declared_order: Any = None
                 ) -> dict[Any, int | None]:
    """Assign each plan step at most one dispatch index, one to one.

    Steps are visited in dependency order; each takes the earliest unused
    dispatch of its agent that comes after every dispatch already matched to
    its dependencies. If no such dispatch exists, the earliest unused dispatch
    of that agent is taken (the caller then reports the step out of order).
    A step left with no dispatch of its agent is ``None`` (not run).
    """
    by_id = {s.get("id"): s for s in steps}
    used: set[int] = set()
    match: dict[Any, int | None] = {}
    for sid in _step_order(steps, declared_order):
        s = by_id.get(sid)
        if s is None:
            continue
        agent = str(s.get("agent"))
        floor = max((match[d] for d in s.get("depends_on") or [] if match.get(d) is not None), default=-1)
        free = [i for i, a in enumerate(actual) if a == agent and i not in used]
        pick = next((i for i in free if i > floor), free[0] if free else None)
        match[sid] = pick
        if pick is not None:
            used.add(pick)
    return match


def reconcile(plan: Mapping[str, Any] | None, execution: Iterable[Mapping[str, Any]] | None,
              artifacts: Any = None, *, ignore_unplanned: Iterable[str] = DEFAULT_IGNORE_UNPLANNED
              ) -> dict[str, Any]:
    """Compare the declared plan with what actually ran.

    ``execution`` is a list of ``{agent, start|start_t, ...}`` dispatch records;
    each plan step is matched to one
    dispatch of its agent (see :func:`_match_steps`). ``artifacts`` is the MANIFEST artifact map (or a
    list of paths) used to check ``expected_outputs``.

    Returns ``{planned, actual, not_run, unplanned, out_of_order, missing_output,
    deviations, n_deviations, summary}``.
    """
    out: dict[str, Any] = {"has_plan": bool(plan), "planned": [], "actual": [], "not_run": [], "unplanned": [],
                           "out_of_order": [], "missing_output": [], "deviations": [], "n_deviations": 0,
                           "summary": ""}
    if not plan:
        out["summary"] = "No plan was recorded."
        return out
    steps = [s for s in plan.get("steps") or [] if isinstance(s, Mapping)]
    records = [e for e in execution or [] if isinstance(e, Mapping) and e.get("agent")]

    def _key(e: Mapping[str, Any]) -> tuple:
        t = e.get("start_t")
        return (0, float(t)) if isinstance(t, (int, float)) else (1, str(e.get("start") or ""))

    records = sorted(records, key=_key)
    actual = [str(e["agent"]) for e in records]
    out["planned"] = [s.get("agent") for s in steps]
    out["actual"] = actual
    planned_set = {str(s.get("agent")) for s in steps}
    ignore = set(ignore_unplanned or ())

    by_id = {s.get("id"): s for s in steps}
    match = _match_steps(steps, actual, plan.get("valid_order"))
    for s in steps:
        if match.get(s.get("id")) is None:
            d = {"kind": "not_run", "step": s.get("id"), "agent": s.get("agent"),
                 "detail": f"Planned step {s.get('id')} ({s.get('agent')}) never ran."}
            out["not_run"].append(d)
    reported: set[str] = set()
    for a in actual:
        if a not in planned_set and a not in ignore and a not in reported:
            reported.add(a)
            out["unplanned"].append({"kind": "unplanned", "agent": a,
                                     "detail": f"{a} ran but was not in the plan."})
    for s in steps:
        mine = match.get(s.get("id"))
        if mine is None:
            continue
        for dep_id in s.get("depends_on") or []:
            dep = by_id.get(dep_id)
            theirs = match.get(dep_id)
            if not dep or theirs is None or mine >= theirs:
                continue
            a, b = s.get("agent"), dep.get("agent")
            out["out_of_order"].append({
                "kind": "out_of_order", "step": s.get("id"), "depends_on": dep_id,
                "detail": f"{a} (step {s.get('id')}) was dispatched before {b} (step {dep_id}), "
                          "which it depends on."})
    if artifacts is not None:
        keys = _artifact_keys(artifacts)
        names = {PurePosixPath(k).name for k in keys}
        for s in steps:
            for o in s.get("expected_outputs") or []:
                if not _output_produced(str(o), keys, names):
                    out["missing_output"].append({
                        "kind": "missing_output", "step": s.get("id"), "output": o,
                        "detail": f"Step {s.get('id')} declared output {o!r}, which this run never produced."})
    out["deviations"] = out["not_run"] + out["unplanned"] + out["out_of_order"] + out["missing_output"]
    n = out["n_deviations"] = len(out["deviations"])
    out["summary"] = (f"Plan followed: {len(steps)} steps, no deviations." if n == 0 else
                      f"{len(steps)} steps planned, {len(actual)} dispatches observed, {n} deviation(s) "
                      f"(not_run {len(out['not_run'])}, unplanned {len(out['unplanned'])}, out_of_order "
                      f"{len(out['out_of_order'])}, missing_output {len(out['missing_output'])}).")
    return out


def render_plan_md(plan: Mapping[str, Any] | None, report: Mapping[str, Any] | None = None) -> str:
    """Markdown section describing the plan and any deviations."""
    if not plan:
        return ""
    lines = ["## The analysis plan", ""]
    if plan.get("goal"):
        lines += [f"**Goal:** {plan['goal']}", ""]
    lines += ["| Step | Specialist | Depends on | Task |", "|---|---|---|---|"]
    for s in plan.get("steps") or []:
        deps = ", ".join(s.get("depends_on") or []) or "—"
        task = str(s.get("task") or "").replace("|", "\\|").replace("\n", " ")
        if len(task) > 90:
            task = task[:90] + "…"
        lines.append(f"| `{s.get('id')}` | `{s.get('agent')}` | {deps} | {task} |")
    lines.append("")
    groups = plan.get("parallel_groups") or []
    if len(groups) > 1:
        lines += ["Intended waves (steps in a wave may run concurrently): "
                  + " → ".join("[" + ", ".join(g) + "]" for g in groups), ""]
    if report:
        lines += ["**Planned vs actual:** " + str(report.get("summary", "")), ""]
        devs = report.get("deviations") or []
        if devs:
            lines += [f"- *{d.get('kind')}* — {d.get('detail')}" for d in devs]
            lines += ["", "> Deviations are expected when a specialist's findings change what should happen "
                          "next; they are recorded so the actual sequence is auditable, not to flag an error.", ""]
    return "\n".join(lines)
