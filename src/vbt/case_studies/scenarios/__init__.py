"""Scripted case-study conversations, rubric scoring and numeric checks.

A scenario is a YAML file with:

* ``turns`` — synthetic steering turns (the paper's first prompt, then the kind
  of lightweight steering the paper describes);
* ``paper_turns`` — the verbatim user turns from Supplementary Text V/VI when
  available (only the initial queries are quoted in the main text; a full
  transcript can be supplied with ``paper_turns_file``, loaded when present);
  ``--turns-source paper|synthetic`` selects which list is replayed;
* ``profiles`` it needs (e.g. ``no-web``) and ``paper_cost_usd``;
* ``expected_findings`` — the paper's findings as a rubric for an LLM judge;
* ``expected_values`` — the paper's reported numbers, checked deterministically
  against the run's ``evidence/claims.json`` texts and artifact tables::

      - {id, description, value, tolerance, source: claims|artifact,
         pattern, column, query}

  ``tolerance`` is absolute, or relative when written as ``"15%"``. For
  ``source: claims`` ``pattern`` is a regex selecting claim texts whose numbers
  are candidates; for ``source: artifact`` it is a glob (relative to the run
  directory) of CSV/TSV tables, ``column`` the column read and ``query`` an
  optional pandas filter. The candidate closest to ``value`` is reported with
  its deviation.

``run_scenario`` drives a CSO session through the turns; ``score_run`` asks a
judge agent to grade the run against the rubric; ``score_scenario_run`` does
that on an ephemeral runtime (no new run directory) and adds the numeric
checks and the cost ratio, writing ``<run>/report/scenario_score.json``.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ...agents import AgentDefinition
from ...bulk import _submit_tool

HERE = Path(__file__).resolve().parent

TURN_SOURCES = ("synthetic", "paper")


def list_scenarios() -> dict[str, dict[str, Any]]:
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted(HERE.glob("*.yaml"))}


def load_scenario(name_or_path: str) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.exists():
        p = HERE / f"{name_or_path}.yaml"
    scen = yaml.safe_load(p.read_text())
    ref = scen.get("paper_turns_file")
    if ref:
        f = Path(ref)
        if not f.is_absolute():
            from ...config import resolve_path
            f = resolve_path(f)
        if f.exists():  # a full Supplementary Text transcript supersedes the quoted turns
            data = yaml.safe_load(f.read_text())
            turns = data.get("turns") if isinstance(data, dict) else data
            if isinstance(turns, list) and turns:
                scen["paper_turns"] = [str(t) for t in turns]
                scen["paper_turns_loaded_from"] = str(f)
    return scen


def scenario_turns(scenario: dict[str, Any], source: str = "synthetic") -> list[str]:
    """The user turns to replay: ``synthetic`` (default) or ``paper``."""
    if source not in TURN_SOURCES:
        raise ValueError(f"turns source must be one of {TURN_SOURCES}")
    if source == "paper":
        turns = scenario.get("paper_turns")
        if not turns:
            raise ValueError(f"scenario {scenario.get('id')!r} has no paper_turns")
        return list(turns)
    return list(scenario["turns"])


def live_source_warnings(config: dict[str, Any], scenario: dict[str, Any] | None = None) -> list[str]:
    """Live data sources still enabled in a no-web (leakage-controlled) setting."""
    profiles = list((scenario or {}).get("profiles") or [])
    no_web = "no-web" in profiles or not (config.get("web") or {}).get("enabled", True)
    if not no_web:
        return []
    servers = {s.get("name"): s for s in ((config.get("mcp_servers") or {}).get("servers") or [])
               if s.get("enabled", True)}
    out = []
    if "clinicaltrials" in servers:
        out.append("ClinicalTrials.gov/cBioPortal (mcp__clinicaltrials__*) are live and may return post-cutoff "
                   "records; disable the server to remove them.")
    ceiling = ((config.get("web") or {}).get("literature_max_date")
               or (config.get("tool_env") or {}).get("VBT_LITERATURE_MAXDATE"))
    if "pubmed" in servers and not ceiling:
        out.append("PubMed is enabled without a publication-date ceiling (web.literature_max_date / "
                   "VBT_LITERATURE_MAXDATE): post-cutoff literature can leak.")
    if (config.get("bash") or {}).get("network", True):
        out.append("Bash network access is enabled (bash.network): agents can download post-cutoff data.")
    return out


async def run_scenario(config: dict[str, Any], scenario: dict[str, Any], *, turns: int | None = None,
                       on_event=None, start_mcp: bool = True, provider=None, turns_source: str = "synthetic"):
    from ...orchestrator import open_session

    session = await open_session(config, on_event=on_event, start_mcp=start_mcp, provider=provider,
                                 run_id=None)
    (session.run.dir / "inputs" / "scenario.json").write_text(
        json.dumps({**scenario, "turns_source": turns_source}, indent=2))
    replies = []
    try:
        for q in scenario_turns(scenario, turns_source)[: turns or None]:
            replies.append(await session.ask(q))
    finally:
        await session.close()
    return session.run, replies


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class FindingVerdict(BaseModel):
    id: str
    verdict: Literal["reproduced", "partial", "absent", "contradicted"]
    evidence: str = Field(description="Quote or paraphrase from the run supporting the verdict")


class ScenarioScore(BaseModel):
    findings: list[FindingVerdict]
    novel_findings: list[str] = Field(default_factory=list,
                                      description="Substantive, supported findings not in the rubric")
    overall_comment: str


JUDGE_PROMPT = """You grade a Virtual Biotech research run against the findings reported in the
paper for the same case study. Read the run's report and, where needed, its specialist reports
under work/*/results/reports/ and evidence/claims.json. For each rubric finding decide:
reproduced (same conclusion, supported by the run's own analysis), partial (direction or
part of it), absent (not addressed), contradicted (run reached the opposite conclusion with
evidence). Numbers are checked separately by a deterministic checker; here conclusions and
direction matter. Do not reward claims that the run asserts without analysis. Then call
submit_result."""


async def score_run(runtime, run_dir: Path, scenario: dict[str, Any], *, write: bool = True) -> dict[str, Any]:
    """Judge verdicts plus numeric checks and the cost ratio; written to
    ``<run_dir>/report/scenario_score.json`` when ``write``."""
    run_dir = Path(run_dir)
    judge = AgentDefinition(name="scenario-judge", description="rubric grader", prompt=JUDGE_PROMPT,
                            tier="scientist", tools=["Read", "Glob", "Grep"], role="Evaluation judge",
                            memory="none", workspace="run")
    runtime.read_roots.append(run_dir.resolve())
    rubric = "\n".join(f"- [{f['id']}] {f['text']}" for f in scenario["expected_findings"])
    sink: dict[str, Any] = {}
    task = (f"Run directory: {run_dir}\nFinal report: {run_dir / 'report' / 'FINAL_REPORT.md'}\n\n"
            f"Rubric ({scenario['title']}):\n{rubric}")
    with runtime.cost_scope("scenario_score"):
        res = await runtime.run_agent(judge, task, depth=1, extra_tools=[_submit_tool(ScenarioScore, sink)])
    if "result" not in sink:
        raise RuntimeError("judge did not submit a score")
    score = sink["result"]
    weights = {"reproduced": 1.0, "partial": 0.5, "absent": 0.0, "contradicted": 0.0}
    score["score"] = round(sum(weights[f["verdict"]] for f in score["findings"]) / max(len(score["findings"]), 1), 3)
    score["judge_cost_usd"] = round(getattr(res, "cost_usd", 0.0) or 0.0, 6)
    if scenario.get("expected_values"):
        checks = check_expected_values(run_dir, scenario)
        score["expected_values"] = checks
        n = [c for c in checks if c["within_tolerance"] is not None]
        score["numeric_score"] = round(sum(bool(c["within_tolerance"]) for c in n) / len(n), 3) if n else None
    score["cost"] = cost_check(run_dir, scenario)
    if write:
        (run_dir / "report").mkdir(parents=True, exist_ok=True)
        (run_dir / "report" / "scenario_score.json").write_text(json.dumps(score, indent=2, default=str))
    return score


async def score_scenario_run(config: dict[str, Any], run_dir: Path, scenario: dict[str, Any], *,
                             provider=None) -> dict[str, Any]:
    """Score ``run_dir`` with a judge on an ephemeral runtime: no new run directory is created
    under ``paths.runs_dir``; the score is written into the scored run."""
    from ...runtime import Runtime
    from ...session import Run

    tmp = Path(tempfile.mkdtemp(prefix="vbt-score-"))
    rt = Runtime(config, Run(tmp), provider=provider)
    try:
        return await score_run(rt, Path(run_dir), scenario)
    finally:
        try:
            await rt.aclose()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Deterministic numeric checks
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text or ""):
        s = m.group(0).replace(",", "").replace("−", "-")
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def _tolerance(spec: dict[str, Any], value: float) -> float:
    tol = spec.get("tolerance", 0)
    if isinstance(tol, str) and tol.strip().endswith("%"):
        return abs(value) * float(tol.strip()[:-1]) / 100.0
    try:
        return float(tol or 0)
    except (TypeError, ValueError):
        return 0.0


def _claim_candidates(run_dir: Path, pattern: str | None) -> list[tuple[float, str]]:
    from ...audit.claims import read_claims_file

    claims, _err = read_claims_file(run_dir / "evidence" / "claims.json")
    rx = re.compile(pattern, re.I | re.S) if pattern else None
    out = []
    for c in claims:
        text = str(c.get("text") or "")
        if rx is not None and not rx.search(text):
            continue
        out += [(v, f"claim {c.get('id')}") for v in _numbers(text)]
    return out


def _artifact_candidates(run_dir: Path, spec: dict[str, Any]) -> list[tuple[float, str]]:
    import pandas as pd

    pattern = spec.get("pattern") or "work/**/*.csv"
    col = spec.get("column")
    out: list[tuple[float, str]] = []
    for p in sorted(run_dir.glob(pattern)):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".tsv", ".txt"):
            continue
        try:
            df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
        except Exception:  # noqa: BLE001 - not a table
            continue
        if spec.get("query"):
            try:
                df = df.query(spec["query"])
            except Exception:  # noqa: BLE001 - the filter's columns are absent in this table
                continue
        if spec.get("count_rows"):
            out.append((float(len(df)), f"{p.relative_to(run_dir)} (rows)"))
            continue
        if not col or col not in df.columns:
            continue
        for v in pd.to_numeric(df[col], errors="coerce").dropna():
            out.append((float(v), f"{p.relative_to(run_dir)}:{col}"))
    return out


def check_expected_values(run_dir: Path, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare the paper's reported numbers with what the run produced."""
    run_dir = Path(run_dir)
    rows = []
    for spec in scenario.get("expected_values") or []:
        value = float(spec["value"])
        tol = _tolerance(spec, value)
        source = spec.get("source", "claims")
        if source == "artifact":
            cands = _artifact_candidates(run_dir, spec)
        elif source == "claims":
            cands = _claim_candidates(run_dir, spec.get("pattern"))
        else:
            cands = []
        best = min(cands, key=lambda c: abs(c[0] - value)) if cands else None
        row = {"id": spec["id"], "description": spec.get("description", ""), "expected": value,
               "tolerance": tol, "source": source, "n_candidates": len(cands),
               "found": None if best is None else best[0], "where": None if best is None else best[1],
               "deviation": None if best is None else round(best[0] - value, 6),
               "relative_deviation": (None if best is None or value == 0
                                      else round((best[0] - value) / abs(value), 4)),
               "within_tolerance": None if best is None else abs(best[0] - value) <= tol + 1e-12}
        rows.append(row)
    return rows


def cost_check(run_dir: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    """Run cost vs the paper's cost, as a ratio (1.0 = same spend)."""
    run_dir = Path(run_dir)
    total = None
    for p in (run_dir / "logs" / "cost_report.json", run_dir / "report" / "cost_report.json"):
        if p.exists():
            try:
                total = float(json.loads(p.read_text()).get("total_usd"))
                break
            except Exception:  # noqa: BLE001
                continue
    paper = scenario.get("paper_cost_usd")
    return {"run_cost_usd": total, "paper_cost_usd": paper,
            "ratio": round(total / float(paper), 4) if (total is not None and paper) else None}
