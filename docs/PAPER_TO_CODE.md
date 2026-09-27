# Paper → code map

Zhang, Eckmann, Miao, Mahon, Zou. *The Virtual Biotech: A multi-agent AI framework for
therapeutic discovery and development.* Science (2026), doi:10.1126/science.aeg6779.
Reference implementation: <https://github.com/harrisongzhang/TheVirtualBiotech> (MIT),
pinned as the `third_party/TheVirtualBiotech` submodule.

## Architecture (Fig. 1)

| Paper | Harness | Notes |
|---|---|---|
| Claude Agent SDK (parent spawns child agents with isolated contexts, MCP, persistent sessions) | `src/vbt/runtime.py` (`Runtime.run_agent`, `Task` tool), `src/vbt/orchestrator.py` | Own agent loop so any provider works; `Task` calls in one response run in parallel. |
| Model: Sonnet 4.5 (CSO + scientists), Haiku 4.5 (Chief of Staff, Reviewer) | `configs/profiles/paper.yaml`; tiers in `configs/default.yaml` | Default profile uses current Claude models; `--profile paper` pins the paper's. |
| Virtual CSO — orchestrates, never touches data | `configs/agents.yaml: cso` (tools: Task, provenance, read-only file tools) | Upstream CSO prompt + `src/vbt/prompts/cso_harness_addendum.md`. |
| 8 scientist agents in 4 divisions + Chief of Staff + Scientific Reviewer | `configs/agents.yaml` | Prompts are the upstream originals (Supplementary Text X). |
| Strategic orientation: CoS briefing ∥ CSO clarification interview (Fig. 1C) | `CSOSession._orientation` | Run concurrently on turn 1. |
| Scientific reviewer → CSO re-delegates for refinement | `CSOSession._work` (`enforce_review`, `max_review_rounds`) | Harness re-opens the CSO loop if specialist output was not reviewed. |
| 10 MCP servers (FastMCP), >100 tools; summaries + previews | `configs/mcp_servers.yaml`, `src/vbt/tools/mcp_bridge.py` | 11 upstream data servers + PubMed server = 103 data tools. |
| Provenance MCP (plans, artifacts, claims) | `src/vbt/tools/provenance.py`, `src/vbt/session.py` | Native tools with the same `mcp__provenance__*` names. |
| Skills with progressive disclosure | `Skill` tool (`src/vbt/tools/builtin.py`), upstream `.claude/skills`, local `skills/` | |
| File ops + code execution | `Read/Write/Edit/Glob/Grep/Bash` built-ins | Writes confined to the run directory; package installs blocked. |
| UI showing reasoning, tools in use, downloadable data/code/reports | `vbt chat` (streamed), run directory (`logs/trace.jsonl`, `work/`, `report/`) | |
| Evidence strength weak/strong; claim–evidence records | preamble rules + `record_claims`; `vbt verify` | |

## Case study 1 — target prioritization (Fig. 2-3)

| Paper step | Harness |
|---|---|
| Clinical trialist proposes outcome fields, evidence hierarchy, source tracking | `vbt scenario run trial_curation` (agentic design step) |
| 37,075 agents, one per NCT ID, 3-tier cascade, Pydantic JSON | `vbt case1 annotate` → `bulk.BulkRunner` + `trial_outcomes/schema.py` + `annotator_prompt.md` |
| Phase I success = progression to Phase II (algorithmic) | `vbt case1 phase1` (`trial_outcomes/phase1.py`; reproduces 99.8% of released labels) |
| Manual review (50 Ph II + 50 Ph III) and TDC agreement | `vbt case1 validate` (`--sample-manual`, `--ref tdc:<csv>`) |
| τ specificity and bimodality coefficient on Tabula Sapiens | `vbt case1 features` (`trial_outcomes/features.py`) |
| Logistic GLM, beta regression, permutation, mixed-effects, genetic adjustment, BH, k-means τ=0.69 | `vbt case1 stats` (`trial_outcomes/stats.py`, `pipeline.py`) |
| Released curated dataset | `third_party/TheVirtualBiotech/datasets/clinical_trials/` (used as `--labels released`) |

## Case studies 2-3 (Fig. 4-5)

| Paper | Harness |
|---|---|
| B7-H3 in LUAD/SCLC, no web search | `vbt scenario run b7h3` (applies `no-web` profile) |
| OSMR / MOONGLOW failure analysis | `vbt scenario run osmr` |
| Expert re-implementation of 8 analyses (Fig. 4C-F, 5B-D) | `src/vbt/analysis/*` reference implementations |
| Comparison of conclusions | `vbt scenario score <name> <run_dir>` (rubric from the paper's findings) |
