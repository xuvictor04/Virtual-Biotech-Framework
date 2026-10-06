# Paper → code map

Zhang, Eckmann, Miao, Mahon, Zou. *The Virtual Biotech: A multi-agent AI framework for
therapeutic discovery and development.* Science (2026), doi:10.1126/science.aeg6779.
Reference implementation: <https://github.com/harrisongzhang/TheVirtualBiotech> (MIT),
pinned as the `third_party/TheVirtualBiotech` submodule.

## Architecture (Fig. 1)

| Paper | Harness | Notes |
|---|---|---|
| Claude Agent SDK (parent spawns child agents with isolated contexts, MCP, persistent sessions) | `src/vbt/runtime.py` (`Runtime.run_agent`, `Task` tool), `src/vbt/orchestrator.py`, `src/vbt/providers/` | Own agent loop so any provider works (`vllm` by default, `anthropic`, `sglang`, `llamacpp`, `openai_compat`, `mock`; [PROVIDERS.md](PROVIDERS.md)); `Task` calls in one response run in parallel. |
| Model: Sonnet 4.5 (CSO + scientists), Haiku 4.5 (Chief of Staff, Reviewer) | Tiers in `configs/default.yaml` (`orchestrator`, `scientist`, `support`, `bulk`); `configs/profiles/paper.yaml`; local server: `configs/local_models.yaml`, `src/vbt/providers/openai_compat.py` | The default is a local open-weight model, Qwen3.8-27B on vLLM 0.31 on one 80-96 GB GPU ([LOCAL_LLM.md](LOCAL_LLM.md)). One model serves every tier, and the tiers differ in reasoning effort and a hard reasoning budget: CSO `xhigh` (24,576 tokens), scientists `medium` (8,192), Chief of Staff and Reviewer (support tier) with thinking off, bulk annotators `medium` (3,072). `--profile paper` pins the paper's Sonnet 4.5 / Haiku 4.5, and `--profile claude` uses current Claude models. Runs on the local model are a separate configuration, not a replication of the paper. |
| Virtual CSO — orchestrates, never touches data | `configs/agents.yaml: cso` (tools: Task, provenance, read-only file tools; BulkDispatch when enabled) | Upstream CSO prompt + `src/vbt/prompts/cso_harness_addendum.md` (review policy, plan, claim filing, restricted tools). `orchestration.cso_tools: upstream` gives the upstream CSO tool set. |
| 8 scientist agents in 4 divisions + Chief of Staff + Scientific Reviewer | `configs/agents.yaml` | Prompts are the upstream originals (Supplementary Text X), followed by per-agent addenda and role-aware harness rules (`src/vbt/agents.py: system_prompt_parts`). Tool lists are checked against the upstream registry (`tests/test_roster.py`). |
| Strategic orientation: CoS briefing ∥ CSO clarification interview (Fig. 1C) | `CSOSession._orientation` | Run concurrently on turn 1. |
| Scientific reviewer → CSO re-delegates for refinement | `CSOSession._work` (`enforce_review`, `review_policy`, `max_review_rounds`) | Harness re-opens the CSO loop if specialist output was not reviewed by a reviewer dispatched after the specialist finished (`review_policy`: always / research / multi_specialist / never); a turn whose rounds run out is recorded `reviewed: false`. |
| 10 MCP servers (FastMCP), >100 tools; summaries + previews | `configs/mcp_servers.yaml`, `src/vbt/tools/mcp_bridge.py` | 11 upstream data servers + PubMed server = 103 data tools. The bridge retries startup, restarts crashed servers, applies per-server timeouts, logs server stderr and passes an allow-listed environment (no provider secrets). |
| Provenance MCP (plans, artifacts, claims) | `src/vbt/tools/provenance.py`, `src/vbt/session.py` | Native tools with the same `mcp__provenance__*` names. |
| Skills with progressive disclosure | `Skill` tool (`src/vbt/tools/builtin.py`), upstream `.claude/skills`, local `skills/` | Local `skills/run-organization` overrides the upstream skill with this harness's run layout. |
| File ops + code execution | `Read/Write/Edit/NotebookEdit/Glob/Grep/Bash` built-ins, `src/vbt/tools/policy.py` | One path policy for every file tool: reads limited to the run directory and the configured read roots, with a blocklist (`.env`, `.git`, credentials) that wins over every allow; writes limited to the agent's own `work/<agent>/` (`work/_cso/` for the CSO and reviewer; every agent may also use the run's `.tmp`/`.home`), and harness records (`MANIFEST.json`, `README.md`, `audit.html`, `evidence/`, `inputs/`, `logs/`, `report/`, `memory/`, `.claude/`) are protected. `Bash` gets the same path checks on every path argument, redirect and `cp`/`tee`/`mkdir` target; `rm`/`mv`/`ln`/`chmod`/`chown` (also via `xargs`/`find -delete`) only inside the agent's own workspace; the upstream guardrail groups (package installs, destructive file-system and database commands, system commands) with the upstream guidance messages, heredoc bodies scanned only for installs; network commands blocked with `bash.network: false`. Commands run in their own process group (TERM then KILL on timeout or interrupt), `HOME`/`TMPDIR` inside the run, an allow-listed environment without provider secrets (`src/vbt/envpolicy.py`), output above 2 MB spilled to `logs/tool_outputs/`; the `timeout` argument is in milliseconds as in Claude Code. Command substitutions, `bash -c`/`eval` strings and heredocs fed to a shell are re-checked, variable assignments are tracked, and `sed -i`/`perl -i`/`tar -C`/`unzip -d`/`cp -t`/`dd of=` are write targets. By default it is a policy layer on the host, not an OS sandbox (interpreter code can still open files); `bash.sandbox.os: bwrap` adds OS enforcement (read-only filesystem except the agent's own workspace, `.tmp` and `.home`, blocked paths hidden), or run untrusted workloads in a container or VM. |
| UI showing reasoning, tools in use, downloadable data/code/reports (fig. S1) | `vbt web` (`src/vbt/web/`, `web` extra: live CSO reasoning, agent and tool activity, claim-evidence panel, figures, file and run downloads); `vbt chat` (terminal) | Both read the same run directory (`logs/trace.jsonl`, `work/`, `report/`, `audit.html`). |
| Evidence strength weak/strong; claim–evidence records | preamble rules + `record_claims`; `vbt verify` | Claims must cite files in `work/` (hash-checked) or external citations (PMID/DOI/URL). `vbt verify` separates artifact integrity (FAIL) from evidence coverage (INCOMPLETE); `--rerun` re-executes the recorded scripts in a scratch copy. |
| Session audit record (harness addition) | `src/vbt/session.py`, `src/vbt/audit/` | Crash-safe MANIFEST (schema 2), per-run README.md and self-contained audit.html, run index (`vbt list`), bundle export (`vbt export`), retrofit of older runs (`vbt audit`). |
| Reproducibility (harness addition) | `src/vbt/pinning.py`, `src/vbt/replay.py` | `inputs/config.json` pins models, effective thinking, per-agent tools and prompt hashes, commits and package versions; `--resume` continues a session, `vbt replay` re-runs its turns and diffs the result. |
| Readiness before billable calls (harness addition) | `src/vbt/preflight.py`, `vbt doctor` | Session and per-turn checks of credentials and reference data; `--allow-missing-data` runs degraded and tells the agents which servers lack data. |
| Persistent agent memory (Agent SDK `memory='project'`) | `memory/<agent>/MEMORY.md` injected into later delegations of the same role; `UpdateMemory` tool | Per run. |
| Web search (the Agent SDK's `WebSearch`, run server-side by Claude) | `WebSearch` tool, `src/vbt/tools/search_backends.py` | Pluggable backend ([WEB_SEARCH.md](WEB_SEARCH.md)): Claude's native search with the Anthropic provider; a self-hosted SearxNG for the local model (`deploy/local/docker-compose.yml`), or the Brave Search API. |
| Case study 2 "without web search to prevent information leakage" | `configs/profiles/no-web.yaml` | Removes WebSearch/WebFetch; PubMed limited to a publication-date ceiling; Bash network commands blocked. ClinicalTrials.gov/cBioPortal remain live. |

## Case study 1 — target prioritization (Fig. 2-3)

| Paper step | Harness |
|---|---|
| Clinical trialist proposes outcome fields, evidence hierarchy, source tracking | `vbt scenario run trial_curation` (agentic design step) |
| 37,075 agents, one per NCT ID, 3-tier cascade, Pydantic JSON | `vbt case1 annotate` → `bulk.BulkRunner` + `trial_outcomes/schema.py` + `annotator_prompt.md` (the `submit_result` tool is `strict`, so a local server constrains its arguments to the schema, and an agent that has not submitted by its last allowed call is forced to; budgeted with `--budget-tokens` on the local model) |
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

## Tool-surface deviations from the reference implementation

The agent tool lists in `configs/agents.yaml` reproduce the upstream registry
(`src/agents/registry.py`, plus the CSO options in `run.py`). `tests/test_roster.py`
parses the registry and fails if an upstream tool is neither granted nor listed in
`x-parity-exceptions`. Deliberate differences:

| Agent | Difference | Why |
|---|---|---|
| CSO | No `Bash`, `Write`, `Edit`, `NotebookEdit`, `WebFetch`, `WebSearch` (restricted mode, default) | The paper's CSO "never directly accesses data"; it reads specialist outputs with `Read`/`Glob`/`Grep` and writes the plan and claims through provenance tools. `orchestration.cso_tools: upstream` restores the upstream set. `Agent` (an SDK alias of `Task`) is not exposed. |
| All | `WebFetch`/`WebSearch` removed when `web.enabled: false` | No-web leakage control (Case study 2). PubMed (`mcp__pubmed__*`) is removed too unless `web.literature_max_date` is set. |
| CSO | `BulkDispatch`, `BulkStatus` (when `bulk.dispatch_enabled`) | The paper's CSO dispatches the 37,075-agent trial annotation. |
| Genomics analyst | `mcp__association__{query_evidence, filter_by_datasource, filter_by_datatype, get_associations_for_target, query_associations, compare_direct_indirect}` + `genomics_burden_addendum.md` | The association server (gene-burden, ClinVar and other genetic evidence) is started but no upstream agent can reach it. |
| Chief of Staff | `ListTools`, `mcp__pubmed__*` | Data-landscape inventory for the briefing; literature search. |
| Single-cell analyst, FDA safety officer, clinical trialist | `mcp__pubmed__*` | Harness PubMed server (literature step of the trial-annotation cascade). |
| Agents with `memory: project` | `UpdateMemory` | Implements the SDK's per-agent project memory. |
| Scientific reviewer | `workspace: run` | Read-only; run-relative paths from `list_artifacts` resolve to the run directory. |
| Turn caps | `max_turns`: single-cell analyst 600, Chief of Staff 60, reviewer 40; others `limits.max_specialist_turns` | Upstream caps only the CSO loop (100 turns). |
