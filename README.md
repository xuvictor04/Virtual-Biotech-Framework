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
| Agent loop, delegation (`Task`), budgets, tool truncation | `src/vbt/runtime.py` |
| CSO workflow: orientation, review enforcement, multi-turn | `src/vbt/orchestrator.py` |
| Tools: Claude-Code-compatible built-ins, MCP bridge, provenance | `src/vbt/tools/` |
| Agent roster, tool allowlists, model tiers | `configs/agents.yaml`, `configs/default.yaml` |
| Bulk runner (one agent per item, Pydantic output, resumable) | `src/vbt/bulk.py` |
| Case studies | `src/vbt/case_studies/` |
| Reference analyses (single-cell, spatial, survival, TF, Shapley, biomarkers) | `src/vbt/analysis/` |

## Setup

```bash
git clone --recursive <this repo> && cd Virtual-Biotech-Framework
# or, in an existing clone:
git submodule update --init

# Option A: the upstream conda environment (complete scientific stack incl. R, scanpy, LIANA, decoupler)
conda env create -f third_party/TheVirtualBiotech/environment.yml && conda activate vbt
pip install -e ".[anthropic,dev]"

# Option B: pip only
pip install -e ".[all]"

cp .env.example .env    # set ANTHROPIC_API_KEY and OPEN_TARGETS_DATA_PATH
python third_party/TheVirtualBiotech/tools/download_open_targets.py /data/open_targets --workers 8
vbt doctor --smoke      # checks prompts, keys, data, and starts all 12 MCP servers
```

**Data requirements:**
- Open Targets 25.09: about 40 GB, required by most data tools.
- Tahoe-100M: optional, about 83 GB.
- CELLxGENE Census: streamed on demand.
- Case-study datasets (Tabula Sapiens, Visium LUAD, TAURUS, GEO): see each scenario's `reference_data` list.

## Usage

```bash
vbt chat                                   # interactive CSO session (recommended)
vbt run "Evaluate PCSK9 as a target for lowering LDL cholesterol."   # headless; one arg per turn
vbt --profile paper chat                   # the paper's models (Sonnet 4.5 + Haiku 4.5)
vbt --no-web chat                          # no web search (information-leakage control)
vbt tools                                  # each agent's resolved tool list
vbt verify runs/<RUN_ID>                   # artifact integrity + claim-evidence coverage
```

Every run writes `runs/<RUN_ID>/`, containing:
- `inputs/`: the query and `plan.json`
- `work/<agent>/`: code, data, figures, tables and reports
- `logs/trace.jsonl`: every model call, tool call and delegation
- `logs/cost_report.json`: token and USD cost per agent
- `evidence/`: `artifacts.json` and `claims.json`
- `report/FINAL_REPORT.md`

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

## Configuration

- `configs/default.yaml`:
  - `provider`: which model backend to use.
  - `models`: model tiers `orchestrator`, `scientist`, `support` and `bulk`.
  - `paths`: prompts, skills and read-only data roots.
  - `limits`: turns, parallelism, output truncation and per-turn budget.
  - `orchestration`: strategic orientation and enforced review.
  - `bash` and `web`: tool policies.
- `configs/agents.yaml`: the organization. It lists each agent's division, prompt, model tier
  and tool allowlist; glob patterns such as `mcp__genetics__*` are allowed. You can add or remove
  agents here.
- `configs/mcp_servers.yaml`: any MCP server, either stdio or HTTP. Its tools appear to agents
  as `mcp__<server>__<tool>`.
- Profiles in `configs/profiles/` are layered in order: `paper`, `no-web`, `mock`, or your own.

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
- **PubMed server.** A PubMed MCP server was added (`src/vbt/mcp_servers/pubmed_server.py`)
  for the literature step of the trial-annotation cascade.
- **Review is enforced.** The harness enforces the Scientific Reviewer step before synthesis.
  Upstream leaves this to the CSO prompt.
- **Web search.** Web search is provided by the provider (Claude's server-side search tool).
  Other providers must implement `web_search` or run with `--no-web`.
- **Code execution.** `Bash` runs on the host inside the run directory and has a command
  blocklist, but it is not a security sandbox. Run the harness in a container or VM.

## Status and known gaps

- The offline test suite covers the agent loop, delegation, review enforcement, bulk runs, Case 1
  statistics and the pure-Python parts of every reference analysis. It uses a scripted provider,
  synthetic data, and a fake Messages API for the Claude streaming path.
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
