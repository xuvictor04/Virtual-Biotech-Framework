# Virtual Biotech Framework

A provider-agnostic harness that replicates and extends **The Virtual Biotech**
(Zhang *et al.*, *Science* 2026, [doi:10.1126/science.aeg6779](https://doi.org/10.1126/science.aeg6779)):
a multi-agent AI organization in which a virtual Chief Scientific Officer (CSO)
coordinates specialist scientist agents across target discovery, safety, modality
selection and clinical development.

- **Faithful:** the agent system prompts, skills and FastMCP data servers are the
  authors' originals. They are pinned as a git submodule
  ([harrisongzhang/TheVirtualBiotech](https://github.com/harrisongzhang/TheVirtualBiotech), MIT license).
- **Swappable model provider:** the harness runs its own agent loop behind a small
  `LLMProvider` interface. Claude is the default; supporting another vendor means
  writing one adapter (see [docs/PROVIDERS.md](docs/PROVIDERS.md)).
- **Reproducible:** includes the paper's three case studies, a bulk runner that assigns
  one agent per trial, statistics code, reference re-implementations of the expert-reviewed
  analyses, and an audit record for every run.

The paper is summarized in [docs/ARTICLE_BREAKDOWN.md](docs/ARTICLE_BREAKDOWN.md), and each
part of it is mapped to its module in [docs/PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md).

## Architecture

```
            user ─┬─► CSO (orchestrator; no data tools) ◄──── Chief of Staff briefing (parallel, turn 1)
                  │      │ Task(...) ×N in parallel, isolated contexts
                  │      ▼
                  │   scientist agents ──► MCP data servers (Open Targets, CELLxGENE, DepMap/Tahoe,
                  │      │                  Reactome/GO, PPI, ClinicalTrials.gov/cBioPortal, PubMed)
                  │      │             ──► Read/Write/Edit/Glob/Grep/Bash, Skills, WebSearch/WebFetch
                  │      ▼
                  │   Scientific Reviewer ── gaps? ──► CSO re-delegates (enforced by the harness)
                  └──◄ synthesis + claim→evidence records (runs/<id>/)
```

| Layer | Module |
|---|---|
| Provider interface, Claude adapter, scripted mock | `src/vbt/providers/` |
| Transient-error retry (429/5xx/overloaded, Retry-After) | `src/vbt/providers/retry.py` |
| Context-window management (tool-result clearing, summarisation, overflow recovery) | `src/vbt/context.py` |
| Agent loop, delegation (`Task`), budget scopes, events, trace | `src/vbt/runtime.py`, `budget.py`, `events.py`, `failures.py` |
| CSO session: orientation, review policy, plan nudge, interruption, resume | `src/vbt/orchestrator.py` |
| Pinned run configuration, replay | `src/vbt/pinning.py`, `src/vbt/replay.py` |
| Readiness checks (`vbt doctor`, preflight before sessions and turns) | `src/vbt/preflight.py` |
| Tools: Claude-Code-compatible built-ins, path/Bash policy, MCP bridge, skills, web, provenance | `src/vbt/tools/` |
| Audit record (MANIFEST, claims, provenance, README.md/audit.html, run index, export, verify) | `src/vbt/session.py`, `src/vbt/audit/`, `src/vbt/verify.py` |
| Browser UI (fig. S1) | `src/vbt/web/` |
| Agent roster, tool allowlists, model tiers | `configs/agents.yaml`, `configs/default.yaml` |
| Bulk runner (one agent per item, Pydantic output, resumable) and CSO `BulkDispatch` | `src/vbt/bulk.py`, `src/vbt/bulk_dispatch.py` |
| Case studies, scenarios, Zenodo archive | `src/vbt/case_studies/` |
| Reference analyses (single-cell, spatial, survival, TF, Shapley, biomarkers) | `src/vbt/analysis/` |

## Setup

```bash
git clone --recursive <this repo> && cd Virtual-Biotech-Framework
# or, in an existing clone:
git submodule update --init

# Supported: one conda env for the harness, the MCP data servers and agents' Bash
# (Python + R statistics stack: scanpy, LIANA, decoupler, PyDESeq2, gseapy, lme4,
# lmerTest, glmmTMB, betareg, MuMIn, rpy2, CELLxGENE Census, FastMCP).
conda env create -f environment.yml && conda activate vbt-harness
pip install -e ".[anthropic,web,tools,dev]"

# pip only (no R): `all` covers the provider, every MCP server's imports, the
# analysis/single-cell/survival stacks and the web UI; `full` adds rpy2 and Cell2Location.
pip install -e ".[all]"

cp .env.example .env    # set ANTHROPIC_API_KEY and OPEN_TARGETS_DATA_PATH
python third_party/TheVirtualBiotech/tools/download_open_targets.py /data/open_targets --workers 8
vbt doctor --smoke      # credentials, data, MCP imports, then one live call per MCP server
vbt --profile mock run "hello"   # offline dry run: scripted provider, no MCP, no cost
vbt doctor --analysis   # also the Python/R analysis stack agents use from Bash
```

**Environment variables.** `.env` is read first, then `third_party/TheVirtualBiotech/.env`
(the file the upstream README tells you to edit); exported variables win over both. A
blank value such as `VBT_MCP_PYTHON=""` means *unset*: the default from
`configs/default.yaml` is used (`${VAR:-default}` treats empty as unset; `${VAR-default}`
only replaces an unset variable). The MCP servers run with `vars.mcp_python` (default:
the interpreter running `vbt`; set `VBT_MCP_PYTHON` to the conda env's python if `vbt`
runs elsewhere). The conda env from `environment.yml` is the supported interpreter for
both the MCP servers and agents' `Bash`. Provider keys and other secrets are never passed
to MCP servers or agent `Bash` commands (`src/vbt/envpolicy.py`); a server gets a secret
only when it lists it in `env_passthrough` (the PubMed server: `NCBI_API_KEY`, `NCBI_EMAIL`).

**Readiness checks (preflight).** Before a session starts and before every turn, the
harness checks the provider credentials (a blank key counts as missing) and the reference
data the enabled MCP servers need (Open Targets layout via the upstream doctor; Tahoe-100M
when `TAHOE_DATA_PATH` is set); the session check also confirms every MCP server command
exists. A failing session check refuses to create the run (exit code 2); a failing per-turn
check records the turn as `not_sent` ("This turn has not been sent to the model") *before
any billable model call*. It is skipped for the mock provider, with
`orchestration.require_reference_data: false`, and with `--skip-preflight`.
`--allow-missing-data` starts anyway when only reference data is missing (credentials are
always required): the run is marked degraded (`MANIFEST.degraded`) and every agent's prompt
names the servers that lack data. `vbt doctor` runs the same checks plus the
MCP interpreter's imports (`--smoke`: one cheap call per server, failing on any tool error;
`--analysis`: scanpy, PyDESeq2, gseapy, LIANA, lifelines, rpy2 and the R packages).

**Data requirements:**
- Open Targets 25.09: about 40 GB, required by most data tools.
- Tahoe-100M: optional, about 83 GB.
- CELLxGENE Census: streamed on demand.
- Case-study datasets (Tabula Sapiens, Visium LUAD, TAURUS, GEO): see each scenario's `reference_data` list.

## Usage

```bash
vbt chat                                   # interactive CSO session (recommended)
vbt run "Evaluate PCSK9 as a target for lowering LDL cholesterol."   # headless; one arg per turn
vbt run -f turns.txt --events ndjson       # one turn per line ('#' comments); JSON event stream
vbt --profile paper chat                   # the paper's models (Sonnet 4.5 + Haiku 4.5)
vbt --model sonnet chat                    # a model_aliases label or a model id for CSO/scientists/bulk
vbt --no-web chat                          # no web search (information-leakage control)
vbt web                                    # browser UI (fig. S1); needs VBT_WEB_PASSWORD or --no-auth on localhost
vbt tools                                  # each agent's resolved tool list
vbt doctor [--smoke] [--analysis]          # installation, credentials, reference data, MCP servers
```

Global flags (before the subcommand): `--profile NAME` (repeatable), `--model`, `--no-web`,
`--no-mcp`, `--runs-dir`, `--skip-preflight`, `--allow-missing-data`, `-v`.

**Interactive sessions.** `vbt chat` streams the CSO's answer with claim anchors replaced by
numbered footnotes, shows a live line of running specialists (`-v`: every tool call;
`--show-reasoning`: the CSO's thinking), the Chief of Staff briefing, review/retry/compaction
notices and data-source warnings. Commands: `/help`, `/summary`, `/claims`, `/evidence C3`,
`/done`; three double quotes start and end multi-line input. Ctrl+C during a turn interrupts
that turn (recorded as `interrupted`, the history is repaired and the session continues);
Ctrl+C at the prompt ends the session and writes the reports; a second Ctrl+C within 2 s
force-exits. `vbt run` exits 1 when a turn did not complete or `verify` is not COMPLETE
(`--no-strict` turns this off).

**Runs, audit and reproducibility.**

```bash
vbt list                                   # runs, newest first (also runs/INDEX.md)
vbt show <RUN> [--files]                   # summary; figures/tables/code by agent
vbt verify <RUN> [--rerun]                 # artifact integrity + claim-evidence coverage; --rerun re-executes scripts in a scratch copy
vbt export <RUN> [-o run.zip] [--no-data] [--chat]   # zip bundle (or the chat as Markdown)
vbt audit <RUN> [--in-place] | --all       # rebuild MANIFEST/provenance/reports from the trace (older or crashed runs)
vbt chat --resume <RUN>                    # continue a recorded session (also `vbt run --resume`)
vbt replay <RUN> [--model ...]             # re-run the recorded turns into a new run and diff (replay_diff.json)
```

`<RUN>` is a run directory, a run id, a unique id prefix or hex suffix, or `latest`. A replay
is a comparison, not a reproduction: model outputs differ between runs; `vbt verify --rerun`
re-executes the recorded scripts instead.

Every run writes `runs/<RUN_ID>/`:
- `README.md` and `audit.html` (self-contained): status, cost, dispatch order with
  delegation prompts, plan vs. actual, artifacts by agent with hashes, claims with evidence
  badges, interrupted turns, problems found by `verify`; rewritten every turn.
- `MANIFEST.json` (schema 2): status, query, agents, pinned configuration, artifacts with
  sha256/attribution/citing claims, harness-file hashes, interrupted turns, data-source
  errors, audit errors, misplaced files, `degraded`. Written atomically on every capture.
- `inputs/`: `query.txt` (every turn), `config.json` (pinned: models with effective
  thinking/effort, per-agent tools and prompt hashes, harness and upstream commits, package
  versions, skill hashes, preflight result), `environment.txt`/`environment.yml`, `plan.json`.
- `work/<agent>/`: code, data, figures, tables and reports (`work/_cso/` for the CSO and
  reviewer, `work/_mcp/` for MCP outputs).
- `evidence/`: `artifacts.json`, `claims.json`, `provenance.json`.
- `report/`: `FINAL_REPORT.md`, `FINAL_REPORT.rendered.md` (numbered anchors + claims
  appendix), `plan_reconciliation.json`, `chief_of_staff_brief.md`.
- `logs/`: `trace.jsonl` (every model call, tool call and delegation, with agent run ids),
  `agents/<agent_run_id>.jsonl` (full transcripts), `transcript.md`, `cost_report.json`,
  `cso_state.json` (for `--resume`), `tool_outputs/` (spilled long outputs), `mcp/<server>.log`.
- `memory/<agent>/MEMORY.md` (`UpdateMemory`) and `.claude/skills/` (the skills, linked).

`runs/INDEX.md` and `INDEX.json` list every run and are updated when a session closes.

**Web UI.** `vbt web [--host 127.0.0.1] [--port 7860]` (`pip install -e ".[web]"`) serves the
fig. S1 interface: the CSO's streamed answer with footnoted claims, reasoning, the briefing,
agents by division with running/finished tool rows, a claim-evidence panel, a figure
gallery, the run's files, past runs and downloads (zip bundle, chat as Markdown,
audit.html). It requires `VBT_WEB_PASSWORD` (HMAC-signed cookie, login lockout);
`--no-auth` is accepted only on a loopback address. Each browser session is one CSO session;
a second query while one runs is refused ("previous query still processing") and Stop
interrupts the running turn. Limits are under `web_ui` in `configs/default.yaml`.

### Case study 1: target prioritization

```bash
vbt scenario run trial_curation            # agentic: CSO + clinical trialist design the curation protocol
vbt case1 annotate --sample 200 --budget 60   # one clinical-trialist agent per NCT ID (resumable)
vbt case1 annotate                         # all Phase II/III trials (paper: 37,075 trials, ~$0.23 median each)
vbt case1 phase1                           # algorithmic Phase I→II labels (99.8% match to released labels)
vbt case1 validate --ref released          # or --sample-manual 50, or --ref tdc:<tdc.csv>
vbt case1 features --h5ad TabulaSapiens.h5ad
vbt case1 stats --labels released          # OR / beta regression / permutation / mixed effects / genetic adjustment
```

The authors' curated trial dataset is included in the submodule, so the statistical analysis can be re-run
without paying for annotation (`--labels released`). You can also recompute everything from scratch.

### CSO-dispatched bulk runs

With `bulk.dispatch_enabled: true` the CSO gets `BulkDispatch`/`BulkStatus` (the paper's CSO
dispatching the trial annotation): items from a CSV/TSV/parquet/JSONL file in the run (with an
optional pandas query) or an id list, a prompt template, the trialist's protocol and a JSON
Schema. An unconfirmed call runs a pilot (`bulk.pilot_size` items) and returns the results
with a cost/time projection; `confirm: true` runs the rest in the background
(`work/<agent>/results/bulk/<job_id>.jsonl`, progress events, `BulkStatus`). A job's
`budget_usd` (at most `bulk.dispatch_max_budget_usd`) is its own budget scope, so it is not
stopped by the per-turn cap (`limits.max_turn_cost_usd`); `limits.max_item_cost_usd` caps each
item.

### Case studies 2 and 3

```bash
vbt scenario list
vbt scenario run b7h3 --score              # B7-H3 in lung cancer (auto-applies the no-web profile)
vbt scenario run osmr --score              # OSMR / MOONGLOW trial failure analysis
vbt scenario score osmr runs/<RUN_ID>      # grade an existing run against the paper's findings
```

Each scenario replays the paper's initial prompt followed by light steering turns.
`--score` asks a judge agent to grade the final report against the paper's findings
(reproduced / partial / absent / contradicted). The deterministic re-implementations in
`vbt.analysis` mirror the paper's expert review of Fig. 4C–F and 5B–D.

### Zenodo case-study archive

The authors' Zenodo archive (the case-study inputs and outputs behind the figures) can be
fetched and used to re-run the Case 1 statistics against the published intermediate
tables; see [docs/ZENODO_REPLICATION.md](docs/ZENODO_REPLICATION.md).

```bash
vbt data zenodo fetch --preset case1       # download the Case 1 part of the archive
vbt case1 replicate                        # re-run the Case 1 statistics on the archive and compare
```

## Configuration

- `configs/default.yaml`:
  - `provider`: which model backend to use.
  - `models`: model tiers `orchestrator`, `scientist`, `support` and `bulk`.
  - `paths`: prompts, skills and read-only data roots.
  - `model_aliases`: labels accepted by `--model` (`provider.model_pattern` validates ids).
  - `limits`: CSO/specialist turn caps, parallelism, delegation timeout, output truncation,
    trace sizes, per-turn and per-bulk-item budgets.
  - `retry`: transient provider errors (attempts, exponential backoff with jitter,
    Retry-After ceiling).
  - `context`: context-window management (clearing old tool results above `soft_ratio` of the
    window, summarising older spans above `hard_ratio`, one overflow recovery per error).
  - `orchestration`: strategic orientation, enforced review and `review_policy`
    (`always`, `research`, `multi_specialist`, `never`), `enforce_plan`, `cso_tools`
    (`restricted` (default) or `upstream`) and `require_reference_data`.
  - `preflight`: `skip`, `allow_missing_data` (also the global flags).
  - `mcp`: server start attempts/timeouts, crash restarts, default call timeout.
  - `bulk`: CSO `BulkDispatch` (enabled, budget cap, pilot size, prewarm).
  - `bash`, `read`, `memory` and `web`: tool policies (sandbox paths, guardrail groups,
    network, output caps, WebFetch extraction, domain filters).
  - `audit` and `web_ui`: the run record and the browser UI.
  - `agent_overrides`: per-run tweaks per agent (`effort`, `model`, `max_turns`, `tools_add`).

  Every key has an in-code default, so older profiles and hand-built configs keep working.
- `configs/agents.yaml`: the organization. It lists each agent's division, prompt, model tier,
  tool allowlist (glob patterns such as `mcp__genetics__*` are allowed), prompt `addenda`,
  `max_turns`, `memory` (`project`: notes in `memory/<agent>/MEMORY.md` are injected into later
  delegations of the same role) and `workspace` (`agent` or `run`). `x-parity-exceptions` lists
  the deliberate differences from the upstream tool lists.
- `configs/mcp_servers.yaml`: any MCP server, either stdio or HTTP. Its tools appear to agents
  as `mcp__<server>__<tool>`. Per server: `timeout_s` (single_cell 7200 s, functional_genomics
  3600 s, others 1800 s), `max_concurrency`, `env_passthrough`. The bridge retries failed starts,
  restarts crashed servers and retries the call once, and writes server stderr to a per-server
  log (`logs/mcp/<name>.log` in the run).
- Profiles in `configs/profiles/` are layered in order: `paper`, `no-web`, `mock`, or your own.
  `no-web` removes WebSearch/WebFetch, keeps PubMed only below a publication-date ceiling
  (`web.literature_max_date`), and blocks network commands in Bash; ClinicalTrials.gov and
  cBioPortal stay live.

System prompts are assembled as a stable, cacheable prefix (upstream prompt, addenda,
role-aware harness rules, CSO addendum) followed by a short volatile Session block (date,
run paths, skill and data roots, unavailable servers, agent memory).

## Tests

```bash
python -m pytest -q      # offline: scripted provider, synthetic data, no API calls
```

## Differences from the reference implementation

- **Own agent loop.** The paper and upstream repo use the Claude Agent SDK. This harness
  reimplements that tool surface (Read, Write, Edit, Glob, Grep, Bash, Skill, TodoWrite,
  WebSearch, WebFetch, Task) on its own loop, so the original prompts run on any provider.
- **Provenance tools.** The upstream provenance MCP server is replaced by native tools with
  the same names.
- **Tool surface.** Agent tool lists match the upstream registry except where
  `configs/agents.yaml: x-parity-exceptions` says otherwise. The main deviation: the CSO has no
  `Bash`, `Write`, `Edit`, `NotebookEdit` or web tools (the paper's CSO "never directly accesses
  data"); `orchestration.cso_tools: upstream` restores them. Harness additions: PubMed for
  literature-using agents, Open Targets association tools (gene-burden evidence) for the
  genomics analyst, `ListTools` for the Chief of Staff, `UpdateMemory`, and `BulkDispatch` for
  the CSO when bulk dispatch is enabled.
- **PubMed server.** A PubMed MCP server was added (`src/vbt/mcp_servers/pubmed_server.py`)
  for the literature step of the trial-annotation cascade.
- **Review is enforced.** The harness enforces the Scientific Reviewer step before synthesis.
  Upstream leaves this to the CSO prompt.
- **Web search.** Web search is provided by the provider (Claude's server-side search tool).
  Other providers must implement `web_search` or run with `--no-web`.
- **Code execution.** `Bash` runs on the host, in the agent's workspace, under a policy
  layer (`src/vbt/tools/policy.py`) shared with the file tools: every path argument must
  be readable (run directory, read roots, system roots) and every redirect/`cp`/`mkdir`
  target writable (the agent's own `work/<agent>/`); `rm`/`mv`/`chmod`/... only inside its
  own workspace; package installs, destructive file-system/database commands and system
  commands are blocked with the upstream guidance messages (heredoc bodies are scanned only
  for installs); `bash.network: false` blocks network commands. Commands run in their own
  process group (killed on timeout or interrupt), with `HOME`/`TMPDIR` inside the run and an
  allow-listed environment without provider keys; output above 2 MB is spilled to a file.
  The `timeout` argument is in milliseconds, as in Claude Code. These are guardrails, not an
  OS sandbox (interpreter code can still open arbitrary files): run the harness in a
  container or VM.
- **Context and retries.** The Agent SDK manages context and retries itself; here
  `context.py` clears old tool results (spilled to files the agent can query with
  `QueryToolOutput`) and summarises older spans as the window fills, and `retry.py` retries
  transient provider errors with backoff. Both are traced and shown as notices.

## Status and known gaps

- The offline test suite covers the agent loop, delegation, review enforcement, bulk runs, Case 1
  statistics and the pure-Python parts of every reference analysis. It uses a scripted provider,
  synthetic data, and a fake Messages API for the Claude streaming path.
- End-to-end integration tests (`tests/test_integration_e2e.py`) drive the CLI entry point and
  `open_session` with the scripted provider: a multi-turn research run with parallel
  delegations, an enforced review and filed claims that `vbt verify` reports COMPLETE, with
  README.md/audit.html, the run index and an export bundle; an interrupted turn followed by a
  successful one; a CSO `BulkDispatch` job that runs past the per-turn cap; the preflight
  gate (refused session, degraded run, `not_sent` turn); and construction of the web app.
  Ctrl+C in `vbt chat` was checked by hand with real SIGINTs; the tests drive the same
  `cancel()` path the signal handler calls.
- Security guardrails (path policy, Bash command policy, WebFetch SSRF checks, web UI
  authentication and download confinement) have regression tests, but they are a policy
  layer, not an OS sandbox.
- The MCP bridge is tested against local FastMCP fixture servers (environment policy, error
  envelopes, crash restart and retry, timeouts, startup retries, stderr logs), with both the
  upstream-pinned fastmcp 3.2 / mcp 1.26 and fastmcp 4 / mcp 2. The roster test checks every
  agent's tool list against the upstream registry and every allowlist entry against the tools the
  servers register. `vbt doctor --smoke` was run against the real servers without Open Targets
  data: they start, and the data tools fail the smoke test as expected.
- Not yet exercised end to end: live Claude runs, the Open Targets–backed MCP tools, and the
  wrappers around optional heavy dependencies (PyDESeq2, LIANA, decoupler, lifelines,
  Cell2Location, CELLxGENE Census, rpy2/lme4/glmmTMB). The build environment had no API key
  and no network access to the data sources.
- Case 1 calibration (harness addition): the paper takes the minimum feature value across a
  drug's targets, which makes the trial-level feature depend on the number of targets.
  - With *random* gene features, the null odds ratios are about 1.08–1.20, not 1.0.
  - `vbt case1 stats` therefore also reports a target-count-adjusted analysis.
    `--gene-perm N` adds a gene-label permutation null. Compare real effects against that
    null, not against OR = 1.

## License and citation

This harness is released under the MIT License. The upstream prompts, servers and datasets are © the
Virtual Biotech authors (MIT; Open Targets data CC0). If you use this work, please cite the paper.
