"""Scripted case-study conversations and rubric scoring.

A scenario is a YAML file with the paper's initial prompt, follow-up steering
turns, the profiles it needs (e.g. ``no-web``), and the paper's findings as a
rubric. ``run_scenario`` drives a CSO session through the turns; ``score_run``
asks a judge agent to grade the run's final report against the rubric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ...agents import AgentDefinition
from ...bulk import _submit_tool

HERE = Path(__file__).resolve().parent


def list_scenarios() -> dict[str, dict[str, Any]]:
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted(HERE.glob("*.yaml"))}


def load_scenario(name_or_path: str) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.exists():
        p = HERE / f"{name_or_path}.yaml"
    return yaml.safe_load(p.read_text())


async def run_scenario(config: dict[str, Any], scenario: dict[str, Any], *, turns: int | None = None,
                       on_event=None, start_mcp: bool = True, provider=None):
    from ...orchestrator import open_session

    session = await open_session(config, on_event=on_event, start_mcp=start_mcp, provider=provider,
                                 run_id=None)
    (session.run.dir / "inputs" / "scenario.json").write_text(json.dumps(scenario, indent=2))
    replies = []
    try:
        for q in scenario["turns"][: turns or None]:
            replies.append(await session.ask(q))
    finally:
        await session.close()
    return session.run, replies


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
evidence). Numbers need not match exactly; conclusions and direction matter. Do not reward
claims that the run asserts without analysis. Then call submit_result."""


async def score_run(runtime, run_dir: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    judge = AgentDefinition(name="scenario-judge", description="rubric grader", prompt=JUDGE_PROMPT,
                            tier="scientist", tools=["Read", "Glob", "Grep"], role="Evaluation judge")
    runtime.read_roots.append(Path(run_dir).resolve())
    rubric = "\n".join(f"- [{f['id']}] {f['text']}" for f in scenario["expected_findings"])
    sink: dict[str, Any] = {}
    task = (f"Run directory: {run_dir}\nFinal report: {Path(run_dir) / 'report' / 'FINAL_REPORT.md'}\n\n"
            f"Rubric ({scenario['title']}):\n{rubric}")
    await runtime.run_agent(judge, task, depth=1, extra_tools=[_submit_tool(ScenarioScore, sink)])
    if "result" not in sink:
        raise RuntimeError("judge did not submit a score")
    score = sink["result"]
    weights = {"reproduced": 1.0, "partial": 0.5, "absent": 0.0, "contradicted": 0.0}
    score["score"] = round(sum(weights[f["verdict"]] for f in score["findings"]) / max(len(score["findings"]), 1), 3)
    (Path(run_dir) / "report" / "scenario_score.json").write_text(json.dumps(score, indent=2))
    return score
