# Virtual Biotech Framework

A provider-agnostic harness that replicates and extends **The Virtual Biotech**
(Zhang *et al.*, *Science* 2026, [doi:10.1126/science.aeg6779](https://doi.org/10.1126/science.aeg6779)):
a multi-agent AI organization in which a virtual Chief Scientific Officer (CSO)
coordinates specialist scientist agents across target discovery, safety, modality
selection and clinical development.

- **Faithful:** the agent system prompts, skills and FastMCP data servers are the
  authors' originals. They are pinned as a git submodule
  ([harrisongzhang/TheVirtualBiotech](https://github.com/harrisongzhang/TheVirtualBiotech), MIT license).
- **Local model by default, Claude optional:** the harness runs its own agent loop behind a
  small `LLMProvider` interface. The default is a **local open-weight model**, Qwen3.8-27B
  served by vLLM 0.31 on one 80-96 GB NVIDIA GPU. [docs/LOCAL_LLM.md](docs/LOCAL_LLM.md)
  explains why this model, which GPUs work, and how to set it up. `--profile claude` uses Claude
  through the Anthropic API instead, and `--profile paper` pins the paper's Claude Sonnet/Haiku
  4.5 setup. Supporting another vendor means writing one adapter (see
  [docs/PROVIDERS.md](docs/PROVIDERS.md)).
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

  every agent's model call ──► LLMProvider ─┬─► vLLM 0.31, OpenAI API: Qwen3.8-27B on one GPU (default)
                                            └─► Anthropic API: Claude (--profile claude | paper)
```

| Layer | Module |
|---|---|
| Provider interface; OpenAI-compatible local adapter (vLLM, SGLang, llama.cpp; Qwen3.8, Qwen3.5/3.6, Qwen3 and DeepSeek-V4 dialects); Claude adapter; scripted mock | `src/vbt/providers/` (`openai_compat.py`, `families.py`, `anthropic_provider.py`, `mock.py`) |
| Local model server: serving profiles per GPU, `vbt local profiles / serve / check / bench`, Docker Compose with SearxNG | `src/vbt/local/`, `configs/local_models.yaml`, `configs/profiles/local-*.yaml`, `deploy/local/` |
| Transient-error retry (429/5xx/overloaded, Retry-After) | `src/vbt/providers/retry.py` |
| Context-window management (tool-result clearing, summarisation, overflow recovery) | `src/vbt/context.py` |
| Agent loop, delegation (`Task`), budget scopes, events, trace | `src/vbt/runtime.py`, `budget.py`, `events.py`, `failures.py` |
| CSO session: orientation, review policy, plan nudge, interruption, resume | `src/vbt/orchestrator.py` |
| Pinned run configuration, replay | `src/vbt/pinning.py`, `src/vbt/replay.py` |
| Readiness checks (`vbt doctor`, preflight before sessions and turns) | `src/vbt/preflight.py` |
| Tools: Claude-Code-compatible built-ins, path/Bash policy, MCP bridge, skills, web (WebSearch backends: SearxNG, Brave, provider-native), provenance | `src/vbt/tools/` |
| Audit record (MANIFEST, claims, provenance, README.md/audit.html, run index, export, verify) | `src/vbt/session.py`, `src/vbt/audit/`, `src/vbt/verify.py` |
| Browser UI (fig. S1) | `src/vbt/web/` |
| Agent roster, tool allowlists, model tiers | `configs/agents.yaml`, `configs/default.yaml` |
| Bulk runner (one agent per item, Pydantic output, resumable) and CSO `BulkDispatch` | `src/vbt/bulk.py`, `src/vbt/bulk_dispatch.py` |
| Case studies, scenarios, Zenodo archive | `src/vbt/case_studies/` |
| Reference analyses (single-cell, spatial, survival, TF, Shapley, biomarkers) | `src/vbt/analysis/` |
| Data layer: source descriptors, tool overlays, gateway, data child, `vbt ds` | `src/vbt/datalayer/`, `configs/data/` |

## Setup

**What you need.** The default model runs on your own hardware:
- **One NVIDIA GPU with 80-96 GB or more**: H100 80GB, H200 141GB, RTX PRO 6000 Blackwell
  96GB, or B200/B300. Linux x86_64, NVIDIA driver 575 or newer (580 or newer for the default
  `vllm/vllm-openai:v0.31.0` image), Docker with the NVIDIA Container Toolkit (or a separate
  Python environment for vLLM), at least 128 GB of RAM and 500 GB of NVMe.
- A 32 GB card (RTX 5090) runs a reduced development profile (about 3 parallel specialists,
  128K context). Several GPUs run data-parallel replicas or the larger DeepSeek-V4-Flash model.
- The harness itself needs no GPU. It can run on the GPU host or on any machine that reaches
  the server (for example through an SSH tunnel).
- No suitable GPU: use Claude (`--profile claude`, an Anthropic API key), or the scripted
  `mock` provider for offline dry runs.

[docs/LOCAL_LLM.md](docs/LOCAL_LLM.md) has the per-GPU profiles, their expected capacity and
the full setup; [deploy/local/README.md](deploy/local/README.md) is the operations reference.

```bash
git clone --recursive <this repo> && cd Virtual-Biotech-Framework
# or, in an existing clone:
git submodule update --init

# 1. The harness. Supported: one conda env for the harness, the MCP data servers and agents'
#    Bash (Python + R statistics stack: scanpy, LIANA, decoupler, PyDESeq2, gseapy, lme4,
#    lmerTest, glmmTMB, betareg, MuMIn, rpy2, CELLxGENE Census, FastMCP).
conda env create -f environment.yml && conda activate vbt-harness
pip install -e ".[web,tools,dev]"
# pip only (no R): `all` covers the providers, every MCP server's imports, the
# analysis/single-cell/survival stacks and the web UI; `full` adds rpy2 and Cell2Location.
#   pip install -e ".[all]"

# 2. Configuration and reference data.
cp .env.example .env    # set OPEN_TARGETS_DATA_PATH (and VBT_LLM_BASE_URL if vLLM runs elsewhere)
python third_party/TheVirtualBiotech/tools/download_open_targets.py /data/open_targets --workers 8

# 3. The model server: Qwen3.8-27B on vLLM 0.31, plus SearxNG for WebSearch.
vbt local profiles --detect                                            # recommends a serving profile
docker compose -f deploy/local/docker-compose.yml --profile h100 up -d # or bare metal: vbt local serve --profile h100
#   first start: downloads ~25-31 GB of weights and compiles CUDA graphs (10-30 minutes)

# 4. Checks.
vbt local check         # capability probe of the running server (tools, strict schema, reasoning, ...)
vbt doctor --smoke      # server /health + served model, data, MCP imports, one live call per MCP server
vbt doctor --analysis   # also the Python/R analysis stack agents use from Bash
vbt --profile mock run "hello"   # offline dry run: scripted provider, no MCP, no model server

# Optional: Claude instead of the local model. ANTHROPIC_API_KEY in .env, then
pip install -e ".[anthropic]"
vbt --profile claude doctor --smoke     # (or --profile paper for the paper's Sonnet/Haiku 4.5)
```

Without `--profile`, the harness uses `configs/default.yaml`, which matches the `h100` serving
profile. On other hardware, pass the harness profile that matches the server:
`--profile local-h200` (H200; also B200), `local-rtxpro6000`, `local-5090`, `local-dp` or
`local-deepseek-v4`.

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
vbt chat                                   # interactive CSO session on the local model (recommended)
vbt run "Evaluate PCSK9 as a target for lowering LDL cholesterol."   # headless; one arg per turn
vbt run -f turns.txt --events ndjson       # one turn per line ('#' comments); JSON event stream
vbt --profile local-h200 chat              # the harness profile matching your serving profile
vbt --profile claude chat                  # Claude through the Anthropic API instead
vbt --profile paper chat                   # the paper's models (Sonnet 4.5 + Haiku 4.5)
vbt --profile claude --model sonnet chat   # a model_aliases label or a model id for CSO/scientists/bulk
vbt --no-web chat                          # no web search (information-leakage control)
vbt web                                    # browser UI (fig. S1); needs VBT_WEB_PASSWORD or --no-auth on localhost
vbt tools                                  # each agent's resolved tool list
vbt doctor [--smoke] [--analysis]          # installation, credentials, reference data, MCP servers
vbt local profiles | serve | check | bench # the local model server (docs/LOCAL_LLM.md)
```

Global flags (before the subcommand): `--profile NAME` (repeatable), `--model`, `--no-web`,
`--no-mcp`, `--runs-dir`, `--skip-preflight`, `--allow-missing-data`, `-v`. The `--model`
aliases depend on the profile: the local default has `qwen`; `--profile claude` and
`--profile paper` have `opus`, `sonnet`, `haiku` and `paper`.

**Budgets.** A local model costs 0 USD, so its runs are budgeted in tokens
(`limits.max_turn_tokens`, `limits.max_item_tokens`; cached prompt tokens count at 0.1). The USD
caps apply to Claude, or to a local model when `provider.options.pricing` sets a price.

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
vbt case1 annotate --sample 200 --budget-tokens 40000000   # one clinical-trialist agent per NCT ID (resumable)
vbt --profile claude case1 annotate --sample 200 --budget 60   # with Claude: a USD budget
vbt case1 annotate --budget-tokens <N>    # all Phase II/III trials (paper: 37,075 trials, ~$0.23 median each on Claude)
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
(`work/<agent>/results/bulk/<job_id>.jsonl`, progress events, `BulkStatus`). A job's budget is
its own budget scope, so it is not stopped by the per-turn cap: `budget_tokens` for the local
model (required there, at most `bulk.dispatch_max_budget_tokens`, 20M), or `budget_usd` for
Claude (at most `bulk.dispatch_max_budget_usd`). `limits.max_item_tokens` /
`limits.max_item_cost_usd` cap each item.

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

### Open Targets release tables

`vbt data ot` downloads tables of an Open Targets Platform release (default 25.09) into the directory the
upstream servers read (`$OPEN_TARGETS_DATA_PATH`). The file list and checksums come from the release's
`release_data_integrity` file, itself checked against its `.sha1`; no directory listing is walked. Every
downloaded file's sha1 is checked, a partial download resumes, and `.download-manifest.json` is written in
the upstream downloader's format (with sha1 added), which the data layer's readiness check (R2) and the
upstream `tools/doctor.py` read.

```bash
vbt data ot list                                   # the release's tables and file counts
vbt data ot fetch target go reactome --dry-run     # what would be downloaded, and its size
vbt data ot fetch target go reactome --max-gb 2    # download, verify, write the manifest
vbt data ot manifest                               # verify tables already on disk and write the manifest
```

## Data layer

The upstream MCP servers read Parquet files and live APIs directly. In places they answer wrongly
without saying so: an identifier in another form is "not found", an unknown ID is an empty success, a
top-k list is cut in file order, an argument is ignored. The harness adds a data layer between the agents
and those servers ([docs/DATA_LAYER.md](docs/DATA_LAYER.md)), and the servers are not edited.

- **Descriptors and overlays.** `configs/data/sources/*.yaml` describe each source: its tables, keys,
  coverage and the role of every column. `configs/data/overlays/*.yaml` bind each upstream tool to the
  tables it reads.
- **What it guarantees** (gateway in `enforce` mode):
  - identifiers are resolved or refused, never answered as "not found";
  - an unknown ID is an error, and an empty result says what it covers;
  - ranked results are the global top-k, with a total checked by an independent count (the witness);
  - every argument is honoured or refused.

  Every result starts with a `_vbt` header (status, source and release, total, coverage). It also leaves a
  provenance record that claims cite by row key. A separate data child process answers these checks under
  a memory limit, and readiness is checked per tool before a session starts.
- **Gateway modes** (`data.gateway.mode` in `configs/default.yaml`):
  - `enforce` (default) guards the servers in `enforce_servers` and observes the rest;
  - `observe` traces every decision and returns the upstream answers unchanged;
  - `"off"` turns the gateway off.
- **Native tools.** `mcp__data__*` (resolve, lookup, find, search, aggregate, neighbors, enrich, ...) read
  the declared tables directly. Every data agent is granted them.

```bash
vbt ds lint [--strict]                    # validate descriptors and overlays
vbt ds check [--table S.T | --tool s.t]   # readiness R1-R10, run in the data child under the reaper
vbt ds explain target.get_target_info     # binding, serve mode, derived schema and text of a tool
vbt ds resolve ensembl_gene PCSK9         # resolver rules and candidates
vbt ds index build                        # resolver and access-path indexes
vbt ds estimate | calibrate | status      # memory estimates, calibration, per-server memory
vbt ds replay <RUN> --all                 # re-execute a run's recorded data calls
vbt verify <RUN> --data                   # fresh fingerprints and replays of the cited data
```

The layer was run against Open Targets 25.09, Tahoe-100M, DepMap, GO, the Cell Ontology, MSigDB and the
live ClinicalTrials.gov, cBioPortal, E-utilities and Census APIs. On the real release, the unmodified
servers fail the six correctness tests: for example, `get_target_info("PCSK9")` answers "not found", and
the top five known drugs for PCSK9 are phase 3 although 23 phase-4 rows exist. With the gateway, the
answers match an independent pyarrow oracle, and invalid arguments are refused.

- [docs/DATA_LAYER_REAL_DATA.md](docs/DATA_LAYER_REAL_DATA.md): what was checked, what was corrected,
  the memory and latency measured, and what is still unchecked.
- [docs/DATA_LAYER_STATUS.md](docs/DATA_LAYER_STATUS.md): the status of each fix and how to write a
  descriptor or overlay.
- [docs/DATA_LAYER_RUNBOOK.md](docs/DATA_LAYER_RUNBOOK.md): from a symptom to the command that
  diagnoses it.

## Configuration

- `configs/default.yaml`:
  - `provider`: which model backend to use (default `vllm`: the local Qwen3.8-27B server at
    `$VBT_LLM_BASE_URL`; `--profile claude` / `--profile paper` switch to `anthropic`).
  - `models`: model tiers `orchestrator`, `scientist`, `support` and `bulk`. With the local
    model all four use `qwen3.8-27b` and differ in reasoning effort (`xhigh`, `medium`, off,
    `medium`) and `thinking_budget` (the hard reasoning cap sent as `thinking_token_budget`).
  - `paths`: prompts, skills and read-only data roots.
  - `model_aliases`: labels accepted by `--model` (`provider.model_pattern` validates ids).
  - `limits`: CSO/specialist turn caps, parallelism, delegation timeout, output truncation,
    trace sizes, per-turn and per-bulk-item budgets (in tokens and in USD; either one stops a
    scope).
  - `retry`: transient provider errors (attempts, exponential backoff with jitter,
    Retry-After ceiling).
  - `context`: context-window management (clearing old tool results above `soft_ratio` of the
    window, summarising older spans above `hard_ratio`, one overflow recovery per error).
  - `orchestration`: strategic orientation, enforced review and `review_policy`
    (`always`, `research`, `multi_specialist`, `never`), `enforce_plan`, `cso_tools`
    (`restricted` (default) or `upstream`) and `require_reference_data`.
  - `preflight`: `skip`, `allow_missing_data` (also the global flags).
  - `mcp`: server start attempts/timeouts, crash restarts, default call timeout.
  - `bulk`: CSO `BulkDispatch` (enabled, budget caps, pilot size, prewarm) and the default
    bulk concurrency.
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
- `configs/local_models.yaml`: the serving profiles for the local model server (`h100`, `h200`,
  `rtxpro6000`, `b200`, `5090`, `dp`, `deepseek-v4`): checkpoint, `vllm serve` arguments,
  variants, Docker tags, expected capacity and caveats. `vbt local serve` renders them, and
  `deploy/local/docker-compose.yml` mirrors them.
- Profiles in `configs/profiles/` are layered in order (`--profile A --profile B`):
  - `local-h100` (the same values as `default.yaml`), `local-h200`, `local-rtxpro6000`,
    `local-5090`, `local-dp`, `local-deepseek-v4`: the harness side of each serving profile
    (tier efforts and reasoning budgets, concurrency, token budgets, context policy);
  - `claude` (current Claude models) and `paper` (the paper's Claude Sonnet/Haiku 4.5): the
    Anthropic provider, with USD budgets and the Claude context policy; `upstream-web` (the
    upstream web app's effort levels, on top of `claude` or `paper`);
  - `no-web` removes WebSearch/WebFetch, keeps PubMed only below a publication-date ceiling
    (`web.literature_max_date`), and blocks network commands in Bash; ClinicalTrials.gov and
    cBioPortal stay live;
  - `mock`: the scripted offline provider, no MCP servers.

System prompts are assembled as a stable, cacheable prefix (upstream prompt, addenda,
role-aware harness rules, CSO addendum) followed by a short volatile Session block (date,
run paths, skill and data roots, unavailable servers, agent memory).

## Tests

```bash
python -m pytest -q      # offline: scripted provider, fake inference servers, synthetic data, no API calls

# live probes of the local adapter against a running server (skipped unless the URL is set)
VBT_LIVE_LOCAL_URL=http://localhost:8000/v1 VBT_LIVE_LOCAL_MODEL=qwen3.8-27b \
  VBT_LIVE_LOCAL_HARNESS=1 python -m pytest -v tests/test_live_local.py

# the data layer on downloaded releases and on the live APIs (skipped unless set; docs/DATA_LAYER_REAL_DATA.md)
VBT_DL_REAL_DATA=data/real python -m pytest -q tests/datalayer/test_dl_real_ot_servers.py
VBT_DL_NETWORK=1 python -m pytest -q tests/datalayer/test_dl_real_live.py
```

## Differences from the reference implementation

- **Own agent loop.** The paper and upstream repo use the Claude Agent SDK. This harness
  reimplements that tool surface (Read, Write, Edit, Glob, Grep, Bash, Skill, TodoWrite,
  WebSearch, WebFetch, Task) on its own loop, so the original prompts run on any provider.
- **Model.** The paper ran every agent on Claude: Sonnet 4.5, and Haiku 4.5 for the Chief of
  Staff and the Scientific Reviewer. This harness defaults to a local open-weight model,
  Qwen3.8-27B, with one model for every tier (the tiers differ in reasoning effort and budget).
  Runs on it are a different configuration, not a replication: `--profile paper` reproduces
  the published model setup. The local default also budgets runs in tokens (the model costs
  0 USD), searches the web through SearxNG instead of Claude's server-side search, and serves
  the model text-only, so images in tool results reach it as text placeholders.
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
- **Web search.** `WebSearch` goes through a pluggable backend (`web.search.backend`,
  [docs/WEB_SEARCH.md](docs/WEB_SEARCH.md)): the provider's native search where it has one
  (Claude's server-side search tool), else a self-hosted SearxNG (the local default:
  `docker compose -f deploy/local/docker-compose.yml up -d searxng`) or the Brave Search API;
  `--no-web` disables web access.
- **Code execution.** `Bash` runs on the host, in the agent's workspace, under a policy
  layer (`src/vbt/tools/policy.py`) shared with the file tools: every path argument must
  be readable (run directory, read roots, system roots) and every redirect/`cp`/`mkdir`
  target writable (the agent's own `work/<agent>/`); `rm`/`mv`/`chmod`/... only inside its
  own workspace; package installs, destructive file-system/database commands and system
  commands are blocked with the upstream guidance messages (heredoc bodies are scanned only
  for installs); `bash.network: false` blocks network commands. Commands run in their own
  process group (killed on timeout or interrupt), with `HOME`/`TMPDIR` inside the run and an
  allow-listed environment without provider keys; output above 2 MB is spilled to a file.
  The `timeout` argument is in milliseconds, as in Claude Code. Command substitutions,
  `bash -c`/`eval` strings and heredocs fed to a shell are checked as commands too, paths held
  in variables are tracked, and in-place editors (`sed -i`, `perl -i`), `tar -C`, `unzip -d`,
  `cp -t` and `dd of=` count as writes. By default this is a guardrail on the command text, not
  an OS sandbox: interpreter code (`python -c`, scripts, heredocs) can still open arbitrary
  files. Set `bash.sandbox.os: bwrap` to run every command under bubblewrap (filesystem
  read-only except the agent's own `work/<agent>/`, `.tmp` and `.home`; `paths.blocked_read`
  hidden), or run the harness in a container or VM.
- **Context and retries.** The Agent SDK manages context and retries itself; here
  `context.py` clears old tool results (spilled to files the agent can query with
  `QueryToolOutput`) and summarises older spans as the window fills, and `retry.py` retries
  transient provider errors with backoff. Both are traced and shown as notices.

## Status and known gaps

- **Local model (the default path): not yet run on a GPU.** The OpenAI-compatible adapter, the
  model-family dialects, the serving profiles and the `vbt local` commands have offline tests
  against fake servers. The adapter was also run against real inference engines on CPU: vLLM
  0.30.0's CPU build and llama.cpp, serving the tiny Qwen3.5-0.8B from the same model family
  ([docs/LOCAL_LLM_VERIFICATION.md](docs/LOCAL_LLM_VERIFICATION.md)). Every Qwen3.8 request
  field passed vLLM's validation; single, parallel, strict-schema and forced tool calls,
  streamed reasoning, context-overflow recovery and one full CSO turn with a delegation worked;
  five adapter bugs were found and fixed. Not exercised yet: vLLM 0.31.0, Qwen3.8-27B itself
  (its chat template, answer quality, reasoning loops), the GPU kernels (FP8, NVFP4, INT4, FP8
  KV cache, MTP), 262K-token contexts and data-parallel routing. The runtime paths added when
  the local model became the default (the forced final `submit_result`, `tool_choice: none` on
  final calls, the empty-reply nudge, token budgets) have offline tests only. The capacity
  numbers in the serving profiles are estimates from the model's geometry and the vLLM source.
- **Model quality is unvalidated for this domain.** No biomedical benchmark exists for
  Qwen3.8-27B (or for any model that was considered), and the case studies have not been run
  on it. Its bulk throughput (a dense 27B model; the paper annotated 37,075 trials) is
  unmeasured. [docs/LOCAL_LLM.md](docs/LOCAL_LLM.md#5-validation-plan-and-open-risks) has the
  validation plan and the open risks (KV-cache precision, a vLLM release from the day of the
  decision, throughput).
- The offline test suite covers the agent loop, delegation, review enforcement, bulk runs, Case 1
  statistics and the pure-Python parts of every reference analysis. It uses a scripted provider,
  synthetic data, fake OpenAI-compatible servers for the local adapter and a fake Messages API
  for the Claude streaming path.
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
- The data layer and the unmodified MCP servers behind it were run on real data, through the
  bridge but without a model ([docs/DATA_LAYER_REAL_DATA.md](docs/DATA_LAYER_REAL_DATA.md)):
  - 31 of the 38 Open Targets 25.09 tables, through the Open Targets-backed servers, with the
    gateway off and on;
  - the live ClinicalTrials.gov, cBioPortal and PubMed tools;
  - the CELLxGENE Census tools (`count_cells`, `get_anndata`).

  The evidence, variant, credible-set and colocalisation tables were not downloaded, so the tools
  that read them were seen only refusing (`not_ready`).
- Not yet exercised end to end: live Claude runs, agent sessions on real data, and the wrappers
  around optional heavy dependencies (PyDESeq2, LIANA, decoupler, lifelines, Cell2Location,
  rpy2/lme4/glmmTMB). The build environment had no GPU and no API key.
- Case 1 calibration (harness addition): the paper takes the minimum feature value across a
  drug's targets, which makes the trial-level feature depend on the number of targets.
  - With *random* gene features, the null odds ratios are about 1.08–1.20, not 1.0.
  - `vbt case1 stats` therefore also reports a target-count-adjusted analysis.
    `--gene-perm N` adds a gene-label permutation null. Compare real effects against that
    null, not against OR = 1.

## License and citation

This harness is released under the MIT License. The upstream prompts, servers and datasets are © the
Virtual Biotech authors (MIT; Open Targets data CC0). If you use this work, please cite the paper.
