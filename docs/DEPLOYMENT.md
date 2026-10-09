# Deploying the harness

This page is for the people who run the harness on their own infrastructure: a GPU host (or cluster) that serves
the local model, enough memory to hold the reference data the upstream servers load, and disk for the full data
releases. Everything here is a command in the repository; nothing depends on the machine the harness was
developed on.

* [1. What gets deployed](#1-what-gets-deployed)
* [2. Hardware for the full paper configuration](#2-hardware-for-the-full-paper-configuration)
* [3. Bring-up with Docker (GPU host)](#3-bring-up-with-docker-gpu-host)
* [4. Bring-up without containers](#4-bring-up-without-containers)
* [5. HPC with Apptainer](#5-hpc-with-apptainer)
* [6. What `vbt setup` does](#6-what-vbt-setup-does)
* [7. Operations](#7-operations)
* [8. Continuous integration](#8-continuous-integration)
* [9. What was verified, and what was not](#9-what-was-verified-and-what-was-not)
* [10. Reference](#10-reference)

## 1. What gets deployed

```
                       GPU host                                              $VBT_HOME (host directory)
  ┌──────────────────────────────────────────────────────────┐            data/      sources/<source>/<release>/...
  │ vllm      vllm/vllm-openai, the serving profile picked    │◄─ models/  models/    Hugging Face cache (weights)
  │           for the GPUs (compose.vllm.yaml, written by     │            runs/      run records
  │           `vbt setup`), 127.0.0.1:8000                    │            projects/  project files kept across runs
  │ vbt       harness image: `vbt web` + the unmodified       │◄─ data/    state/     host.yaml, host.env,
  │           upstream MCP servers + the data child, as       │   runs/               compose.vllm.yaml,
  │           children of the web process, 127.0.0.1:7860     │   projects/           setup-state.json, logs/
  │ searxng   web search for the local model (internal only)  │   state/   secrets.env  (operator's, mode 600)
  └──────────────────────────────────────────────────────────┘
```

| Piece | File | What it is |
|---|---|---|
| Harness image | `deploy/full/Dockerfile` | Ubuntu 22.04 + one micromamba env (`/opt/conda/envs/vbt`) with everything the harness, the agents' Bash and the unmodified upstream MCP servers import, pinned by `deploy/full/conda-linux-64.lock` (384 conda packages, URL + md5) and `deploy/full/requirements.lock` (87 pip packages); the harness installed editable at `/opt/vbt`; the upstream submodule at the commit the repository pins. Entrypoint `deploy/full/vbt-host`. |
| HPC image | `deploy/full/vbt-harness.def` | The same environment for Apptainer, built with the same two install scripts (`install-env.sh`, `install-harness.sh`). |
| Compose file | `deploy/full/compose.yaml` | `vbt` (web UI), `setup` (one-shot `vbt setup`), `searxng`; the `vllm` service comes from `$VBT_HOME/state/compose.vllm.yaml`. |
| Driver script | `deploy/full/deploy.sh` | `build`, `lock`, `setup`, `up`, `smoke`, `down`/`ps`/`logs`/`config`, `backup`, `upgrade`. |
| Host bring-up | `vbt setup` (`src/vbt/setup/`) | Probes the host, picks the serving profile, writes the host configuration, then acquires data, sizes memory, builds indexes, checks readiness, calibrates and smoke-tests, with resume. |
| Production profile | `configs/profiles/production.yaml` | Policies of an operated installation (strict preflight, gateway in enforce mode, no environment inheritance by MCP servers). |

The environment is the upstream one where upstream pins a version (`third_party/TheVirtualBiotech/environment.yml`:
numpy 1.26.3, pandas 2.3.3, pyarrow 20.0.0, scanpy 1.11.5, cellxgene-census 1.17.0, fastmcp 3.2.0, mcp 1.26.0,
R 4.4.3 with lme4/lmerTest, ...; `tests/test_setup.py` checks every pin). Three deliberate differences keep the image
a server image: `matplotlib-base` instead of `matplotlib` (no Qt), OpenBLAS instead of MKL, and PyDESeq2 from pip
(its conda package pulls in Jupyter). The upstream agent framework (`claude-agent-sdk`) and its Gradio app are not
installed: the harness replaces both.

## 2. Hardware for the full paper configuration

"Full" means: every agent enabled, the default local model, all 38 Open Targets 25.09 tables, Tahoe-100M, the
Census and the live APIs. The numbers below are measured where they say so; everything else is labelled an estimate.

### 2.1 GPUs for the model

| Option | GPUs | Weights | Notes |
|---|---|---|---|
| Default local model, one card | 1x H100 80GB (`h100`), H200 141GB (`h200`), RTX PRO 6000 96GB (`rtxpro6000`), B200/B300 (`b200`) | 30.9 GB (FP8) or 24.7 GB (NVFP4) | ~8.5 (H100) to ~13 (RTX PRO 6000) concurrent 128K-token agent contexts, estimated from the model geometry ([LOCAL_LLM.md](LOCAL_LLM.md) §2) |
| More throughput | 2-8x H200/B200 (`dp`) | the BF16 checkpoint | one data-parallel replica per GPU; per H200 ~10 agents at 100K tokens (estimate) |
| Largest local model | 4x H200 / 4x B200 (`deepseek-v4`) | 166.9 GB | ~49 GB of weights per GPU |
| Development only | 1x 32 GB card (`5090`) | INT4, 19.45 GB | ~2.7 x 128K contexts; 3 parallel specialists |
| Claude instead | none | - | `--harness-profile claude` (or `paper` for the paper's model configuration); `ANTHROPIC_API_KEY` in the secrets file; the harness host needs no GPU |

`vbt setup` reads `nvidia-smi` and picks the profile (`vbt local profiles --detect` does the same); `--serving-profile`
overrides it. NVIDIA driver 580 or newer runs the default `vllm/vllm-openai:v0.31.0` image (575-579: the `-cu129`
tag, chosen automatically). Plan 40-200 GB of disk for model weights (`$VBT_HOME/models`).

### 2.2 Memory for the upstream servers

The upstream servers load whole tables into pandas and keep them for the life of the process; each server is its
own process with its own copy (three servers each hold `target`). What a server needs is the sum of the tables its
tools load whole (`access: full_table` in its overlay). Measured loads come from
[DATA_LAYER_REAL_DATA.md §5.1](DATA_LAYER_REAL_DATA.md#51-whole-table-loads-and-the-memory-estimate) (peak RSS of a
pandas load of the real 25.09 table). Every other figure is the shipped memory estimator (`vbt ds estimate`; fitted
factors, 0.70-1.46 of the measured load on the 18 measured tables, median 1.00) applied to the Parquet footers of
every shard of the release; for the 31 downloaded tables the footer-derived estimate equals `vbt ds estimate` on the
files exactly. Above about 3 GB no estimate has been checked against a real load: the five tables marked
"killed" did not fit a 5.9 GB test child. Memory here is in MiB (MB) and GiB (GB), as the limits, admission and
the reaper count it (`vbt ds estimate` reports the same units); sizes on disk are decimal.

Per server, the tables loaded whole (estimates unless marked "measured"):

| Server | Tables loaded whole | Whole-table loads | On disk |
|---|---|---:|---:|
| genetics | l2g_prediction (11,564 MB; killed at 5.9 GB), study (measured 2,556 MB), colocalisation_coloc (24,268 MB), colocalisation_ecaviar (37,136 MB) | ~73.8 GB | 9.5 GB |
| interaction | interaction (9,849 MB; killed at 5.9 GB) | ~9.6 GB | 0.09 GB |
| association | the six association tables (measured 4,800 MB together), literature_vector (measured 119 MB) | ~4.9 GB | 0.43 GB |
| expression | expression (5,408 MB; killed at 5.9 GB), biosample (measured 97 MB) | ~5.4 GB | 0.05 GB |
| target | target (measured 3,031 MB), mouse_phenotype (measured 323 MB), target_prioritisation (measured 51 MB), openfda target reactions (7 MB) | ~3.4 GB | 0.09 GB |
| drug | known_drug (measured 606 MB), drug_indication (measured 75 MB), drug_molecule (measured 53 MB), mechanism of action, warnings, openfda drug reactions (34 MB) | ~0.8 GB | 0.02 GB |
| disease | disease (measured 199 MB), disease_phenotype (measured 285 MB), disease_hpo (32 MB) | ~0.5 GB | 0.01 GB |
| pathway | go, reactome, so (18 MB) | ~0.02 GB | <0.01 GB |
| **all** | | **~98 GB** | |

Other tables are read in bounded scans (filters pushed into pyarrow), but each of them is large when loaded:
evidence (estimate 183 GB; 30.4 M rows), variant (185 GB), interval (46 GB), literature (45 GB), credible_set
(41 GB), interaction_evidence (52 GB). If every table a server reads were loaded whole, genetics would need
~348 GB and association ~234 GB (estimates): these are upper bounds, not the expected use. functional_genomics
loads `target_essentiality` whole (10,765 MB; 14,294 MB with the safety factor and the idle baseline).

Recommended RAM for the harness host with the full release:

| Part | RAM |
|---|---|
| Upstream servers holding every whole-table load at once (above, x1.3 admission safety) | ~128 GB |
| Data child (readiness checks, witnesses, native tools): `vbt setup` gives it 5% of RAM, 3-32 GB | 16-32 GB |
| Agents' Bash commands (8 in parallel at `data.memory.workspace_mb`; `auto` gives each what the plan leaves after the host budget, the data child and the reserve: 13,107 MB on 512 GB, 28,672 MB on 1 TB, at least 8,000 MB) | 64-230 GB |
| `single_cell` server (Census pulls: 1.8-4.5 GB peak measured for 1,842-149,759 cells through the unmodified upstream functions, DATA_LAYER_REAL_DATA.md §8.1; the donor-balanced pull's estimate is 3,900 MB plus 4,500 bytes per cell; a 200,000-cell pull exceeded 4.7 GB) | 16-64 GB |
| Harness, web UI, model client | ~4 GB |
| **Total** | **256 GB minimum; 512 GB to 1 TB** for the full configuration with headroom |

The host budget (`data.memory.host_budget_mb`) caps the sum of the servers' resident memory: when a load would cross
it, the least recently used idle server is recycled, so a smaller host still works with more reloads.

**Memory settings scale with the host (`auto`).** `configs/default.yaml` ships every memory budget as `auto`, resolved
by one rule (`src/vbt/datalayer/memory/sizing.py`) from `plan`, the memory the harness may plan with: the smaller of
MemTotal and the memory cgroup limit of the process and its ancestors (a container's `--memory`), or
`data.memory.host_mb` when it is a number (set it on a host the harness shares, e.g. with the model server's RAM), or
`$VBT_HOST_MEMORY_MB`.

| Setting | `auto` means | Floor |
|---|---|---|
| `data.memory.host_budget_mb` (sum over the upstream servers) | 0.75 x plan - max(`harness_reserve_mb` 2,048, 5% of plan) | 1,024 |
| `data.memory.default_server_mb` (one upstream server) | 0.8 x the host budget | 2,048 |
| `data.service.mem_limit_mb` (the data child) | 5% of plan, at most 32,768 | 3,000 |
| `data.service.max_resident_mb` | 2/3 of the data child's limit (what one request holds resident: the readiness key pass) | |
| `data.witness.max_scan_bytes`, `max_inflate_bytes`, `max_key_set`, `repair_max_bytes`; `data.readiness.vocab_budget_bytes` | the shipped floor x data child / 3,000 | the shipped floor |

| plan | host budget | one server | data child |
|---:|---:|---:|---:|
| 16 GB (16,384 MB) | 10,240 MB | 8,192 MB | 3,000 MB |
| 64 GB | 45,875 MB | 36,700 MB | 3,276 MB |
| 128 GB | 91,750 MB | 73,400 MB | 6,553 MB |
| 512 GB | 367,002 MB | 293,601 MB | 26,214 MB |

On a 512 GB host every server loads the Open Targets tables it reads whole, as upstream does (genetics' ~73.8 GB x1.3
safety is about 96 GB). On a 16 GB host a load that cannot fit is refused before the call with `too_large`; the
refusal names the host (`host_mb`), whether the limit was `auto` or configured (`limit_source`) and, for `auto`, the
smallest host that would admit it (`host_mb_needed`). A number anywhere stays as configured, and a server's own
`mem_limit_mb` in `configs/mcp_servers.yaml` wins over `default_server_mb`.

**Containment is `rss` by default** (`data.memory.limit_kind`): a server is held to its limit by resident memory, in a
memory cgroup when one can be created, else by the reaper's RSS watchdog. `RLIMIT_DATA` is set only when a server or
the host asks for `rlimit_data`: TileDB's Census reads reserve large virtual buffers and fail under it with
`std::bad_alloc` while their resident memory stays under 3 GB (`single_cell` is pinned to `rss`).

### 2.3 Disk per source

| Source | On disk | Notes |
|---|---:|---|
| Open Targets 25.09 (38 tables, 3,508 Parquet shards) | 31.1 GB | the 7 large tables are 29.3 GB of it (evidence 8.99 GB, colocalisation_ecaviar 5.16, colocalisation_coloc 3.92, interval 3.23, variant 3.18, credible_set 2.59, literature 2.27) |
| Tahoe-100M (`tahoebio/Tahoe-100M`) | 82.76 GiB of DE shards (1,026 files) + 2.29 GB `obs_metadata.parquet` | the prepared DE file of the whole release has not been built yet: plan the same again |
| DepMap 24Q4 | 0.81 GB | four CSV files |
| Gene Ontology, Cell Ontology, MSigDB Hallmark | 35 MB | |
| Zenodo case-study archive | 2.9 GB | only for the case-study replications |
| CELLxGENE Census | 0 | read remotely; pulled AnnData files land in each run |
| Data-layer sidecars (`data/.vbt-datalayer`) | 37 MB for the 31 downloaded OT tables (measured) | resolver indexes, readiness cache, calibrations |
| Model weights | 25-170 GB per checkpoint | `$VBT_HOME/models` |
| Runs and projects | grows with use | back up (section 7.3) |

Plan **500 GB** of fast disk (NVMe) for data, weights and the first year of runs; **1-2 TB** to keep a previous data
release next to the current one during an upgrade.

## 3. Bring-up with Docker (GPU host)

Prerequisites: Linux x86_64, Docker Engine with Compose v2.24 or newer, the NVIDIA Container Toolkit, NVIDIA driver
575 or newer, git.

```bash
git clone --recursive <this repository> vbt && cd vbt          # or: git submodule update --init
export VBT_HOME=/srv/vbt
install -d -m 700 "$VBT_HOME"                                     # the secrets file below needs the directory
install -m 600 /dev/null "$VBT_HOME/secrets.env"
printf 'VBT_WEB_PASSWORD=%s\n' "$(openssl rand -base64 24)" >> "$VBT_HOME/secrets.env"
#   optional in the same file: ANTHROPIC_API_KEY, HF_TOKEN, VLLM_API_KEY, NCBI_API_KEY, NCBI_EMAIL

deploy/full/deploy.sh up
```

`up` builds the image if it is missing (`deploy.sh build`, about 4 GB), saves the host's `nvidia-smi` output in the
state directory (the harness container sees no GPU), runs `vbt setup` in the image (section 6) with every step but
the smoke test, starts the services, waits for the model server's health check (the first start downloads the
weights and compiles CUDA graphs: 10-30 minutes; `VBT_MODEL_WAIT_S`, default 3600) and runs the smoke step.

Before the first `up`, see what it will do (`--plan`, `--probe` and `--status` start, pull and change nothing):

```bash
deploy/full/deploy.sh setup --plan          # steps, download sizes, time estimates, memory sizing, warnings
deploy/full/deploy.sh setup --probe         # what the host offers: CPUs, RAM, GPUs, disk, containment, network
```

Then:

```bash
deploy/full/deploy.sh ps                     # services and health
deploy/full/deploy.sh logs vllm              # the model server's startup (look for "GPU KV cache size")
deploy/full/deploy.sh smoke                  # again, any time
ssh -N -L 7860:127.0.0.1:7860 gpu-host       # the web UI from a workstation: http://localhost:7860
```

Everything is published on 127.0.0.1 only. For remote users put an authenticating reverse proxy with TLS in front of
port 7860 (`VBT_WEB_BIND` / `VBT_WEB_PORT` move it). vLLM's API key protects only its `/v1` routes, so never publish
port 8000 ([deploy/local/README.md](../deploy/local/README.md), Security). SearxNG is never published.

Behind a proxy that re-signs TLS, build with `HTTPS_PROXY=... VBT_BUILD_CA_FILE=/path/proxy-ca.pem deploy.sh build`
(the CA is a build secret, not part of the image; add `VBT_BUILD_NETWORK=host` when the proxy listens on the host's
loopback). A registry mirror is set with `BASE_IMAGE` and `MICROMAMBA_IMAGE`.

## 4. Bring-up without containers

```bash
git clone --recursive <this repository> vbt && cd vbt
micromamba create -y -p /opt/vbt-env -f deploy/full/conda-linux-64.lock   # the image's exact environment
/opt/vbt-env/bin/python -m pip install --no-deps -r deploy/full/requirements.lock
deploy/full/install-harness.sh --prefix /opt/vbt-env --src "$PWD"
#   the lock is the image's exact environment; the root environment.yml (README step 1) gives version ranges and
#   resolves to whatever conda-forge and PyPI hold on the day it is created

export PATH=/opt/vbt-env/bin:$PATH VBT_HOME=/srv/vbt
vbt --profile production setup --plan
vbt --profile production setup                 # writes $VBT_HOME/state/host.{yaml,env}
vbt local serve --profile "$(. $VBT_HOME/state/host.env; echo $VBT_SERVING_PROFILE)" --docker --detach
deploy/full/vbt-host setup --only smoke
deploy/full/vbt-host web                         # = vbt with the host's profiles and host.env
```

`deploy/full/vbt-host` is `vbt` with the host configuration applied: it loads `host.env` (a variable already set in
the environment wins) and passes the profiles `vbt setup` recorded (`VBT_PROFILES`, `host.yaml` last). A plain `vbt`
does the same on its own (`vbt.cli.apply_host_config`, from `$VBT_STATE_DIR`, else `$VBT_HOME/state`, else
`data/.vbt-setup`): the recorded profiles come before the command's own `--profile` flags, a resumed session or a
replay without `--profile` keeps its pinned profiles, and `VBT_NO_HOST_ENV=1` turns it off. The wrapper stays the
image's entrypoint (it also gives the container a writable `HOME`).

## 5. HPC with Apptainer

```bash
apptainer build vbt-harness.sif deploy/full/vbt-harness.def          # or: ... docker-daemon://vbt-harness:latest
export SIF=$PWD/vbt-harness.sif HOME_DIR=/scratch/$USER/vbt
mkdir -p $HOME_DIR
apptainer run --bind $HOME_DIR:/srv/vbt $SIF setup --plan
apptainer run --bind $HOME_DIR:/srv/vbt $SIF setup --serving-profile h200 --llm-url http://gpu-node:8000/v1
apptainer run --bind $HOME_DIR:/srv/vbt $SIF web --host 127.0.0.1
```

The model server runs as a GPU job, e.g. `apptainer run --nv --bind $HOME_DIR/models:/root/.cache/huggingface
docker://vllm/vllm-openai:v0.31.0 <arguments of vbt local serve --profile h200 --dry-run>`. Point the harness at it
with `--llm-url` (written to `host.env`). Login nodes usually forbid memory cgroups: the shipped `rss` containment
then falls back to the reaper's RSS watchdog (never `RLIMIT_DATA`, under which TileDB's Census reads crash), and
`vbt setup` reports which one applies.

## 6. What `vbt setup` does

```
vbt [--profile P ...] setup [--plan | --probe | --status] [--only S,..] [--from S] [--skip S,..] [--force]
                            [--home DIR] [--serving-profile NAME] [--harness-profile NAME] [--llm-url URL] ...
```

| Step | What it runs | Writes |
|---|---|---|
| `probe` | CPUs (affinity, cgroup quota), RAM (MemTotal and the container's memory limit), GPUs (`nvidia-smi`, or `state/nvidia-smi.csv`), free disk under each directory, whether a memory cgroup can be created (the reaper's own functions), whether bubblewrap works, the container runtime, and reachability of every URL the descriptors name plus the model server and SearxNG | `setup-state.json` |
| `configure` | serving profile (from the GPUs), harness profile, memory sizing (below), data roots | `host.yaml`, `host.env`, `compose.vllm.yaml` |
| `acquire` | `vbt data acquire --for-tools <every tool the enabled agents may call>` (`--plan --json` for the plan, `--offline` with `--no-network`); the variables it reports (`OPEN_TARGETS_DATA_PATH=...`) replace the defaults | the data under `data/sources/<source>/<release>`, then `host.yaml` and `host.env` again |
| `size` | `vbt ds estimate --json` of the tables the enabled servers load whole; when the largest server's estimate x1.3 is above the rule's server limit (0.8 x the host budget) it raises `data.memory.default_server_mb`, up to the host budget | `host.yaml` again |
| `index` | `vbt ds index build --table ...` for every table the enabled tools read: only the id types those tables hold | `data/.vbt-datalayer/` |
| `check` | `vbt ds check --json --table ...` for every table the enabled tools read | readiness cache |
| `calibrate` | `vbt ds calibrate --table ...` for the local tables loaded whole | calibrations |
| `smoke` | an offline mock session; `vbt doctor` (`--smoke` once the model server answers, else the step is `deferred`); `vbt doctor --analysis` (on a pip-only install without `Rscript`, failures that are only about R are a warning, not a failed step; `--no-analysis` skips it) | logs |

Nothing in setup names a dataset: what to fetch, check and calibrate comes from the descriptors, the overlays and the
enabled roster (agents in `configs/agents.yaml`, servers enabled in `configs/mcp_servers.yaml`). Disable a server or
an agent and setup stops fetching and checking what only it reads (a derived serve's optional dependencies, such as
the Gene Ontology behind `get_go_enrichment`, count as read). Data already on disk elsewhere: set its root variable
(e.g. `OPEN_TARGETS_DATA_PATH`) and run with `--skip acquire`. A source no bound tool reads is not fetched by setup;
the Cell Ontology the native data tools resolve cell types with is fetched with `vbt data acquire cell_ontology
--env-file <file>` (until then `cell_ontology.term` is `missing`, never served from the test fixture).

Each step after `configure` is a `vbt` command run as a child under the host configuration, logged to
`state/logs/<step>.log`. A step that finished is skipped next time while its inputs are unchanged (its command, the
host configuration, the files under each data root); `--force` runs it anyway. A failed step stops the run
(`--keep-going` continues). `--plan` prints the steps with download sizes and time estimates and changes nothing; the
time of the data steps is this host's own measured seconds per GB after its first run (the reference rates below
before that). The commands are configuration: `setup.steps.<step>.run` / `.plan` in a profile replace a step's
arguments.

**Sizing** (MB; `ram` = the smaller of MemTotal and the container limit, or `data.memory.host_mb`): the rule `auto`
applies at run time (section 2.2), so `host.yaml` holds the numbers `auto` gives on this host: host budget `0.75 x ram
- max(2048, 0.05 x ram)`; server limit `0.8 x budget` (at least 2,048), raised by step `size` when the largest
server's whole-table loads x1.3 need more (at most the budget); data child `clamp(0.05 x ram, 3000, 32768)`, its
`max_resident_mb` 2/3 of that; agents' Bash `(ram - budget - data child - reserve) / max_parallel_agents`, within
8,000-65,536 and at most half of `ram` (the limits share one budget; `vbt validate --only host` prints the sum at
full load); `limit_kind:
cgroup` where a memory cgroup can be created (else the shipped `rss`); `bash.sandbox.os: bwrap` where bubblewrap works
(Docker's default seccomp profile refuses it; `compose.yaml` shows the opt-in). `vbt setup --project NAME` also
fetches, checks and indexes the sources of that project (docs/PROJECTS.md).

**Reference rates** (seconds per GB of local data, measured with Open Targets 25.09's 31 downloaded tables, 1.76 GB,
on 4 CPUs; section 9): size 10.8, index 194.5, check 81.2, calibrate 34.0. They extrapolate linearly, which is an
assumption: a host's own rates replace them after its first run.

## 7. Operations

### 7.1 Upgrades

```bash
deploy/full/deploy.sh upgrade        # git pull --recurse-submodules, build, setup (resumes), up, smoke
```

* **Harness.** A new commit rebuilds the image; `vbt setup` reruns only the steps whose inputs changed.
* **Upstream servers.** The submodule is pinned; `deploy.sh build` refuses a submodule that is not at the pinned
  commit. Move the pin in a commit of its own, after `pytest` (the roster and upstream-untouched tests compare the
  tool registry with `configs/agents.yaml` and the overlays).
* **Python environment.** Edit `deploy/full/environment.yml` or `requirements.in`, then `deploy/full/deploy.sh lock`
  (resolves them in a container and writes both locks; pip is constrained to the conda versions and the build fails
  if a pip package would replace a conda one), build, test, commit the locks.
* **Model server.** The vLLM image tag and the serving profiles live in `configs/local_models.yaml`; `vbt setup`
  rewrites `compose.vllm.yaml` from them. `VLLM_IMAGE` overrides the image for a trial.

### 7.2 New data releases

A release is a new directory, never an in-place update, so runs stay verifiable against the release they used.

1. Pin the new release in the descriptor's acquisition section (`acquisition.release: "25.12"` in
   `configs/data/sources/open_targets.yaml`) and fetch it next to the old one: `vbt data acquire open_targets`
   writes the acquisition home `$VBT_HOME/data/sources/open_targets/25.12` (`<acquisition.dir>`; the engine
   fetches only the release the descriptor pins). Without `--env-file` the servers keep reading 25.09.
2. `vbt ds diff-release --from $VBT_HOME/data/sources/open_targets/25.09 --to
   $VBT_HOME/data/sources/open_targets/25.12`: role columns, types, encodings, vocabularies and matrix axes that
   changed.
3. Update the descriptor (`release.expect`, any column the diff names) and lint it: `vbt ds lint --strict`.
4. Point the root at the new release (`vbt data acquire open_targets --env-file <file>` rewrites
   `OPEN_TARGETS_DATA_PATH`, or set it in the environment or the secrets file) and run `vbt setup --from
   configure`: `size`, `index`, `check` and `calibrate` rerun because the data changed.
5. Keep the old release until the runs that cite it no longer need re-verification: `vbt verify <run> --data`
   replays a run's data calls against the fingerprints it pinned.

### 7.3 Backups

| What | How | Why |
|---|---|---|
| `runs/`, `projects/`, `state/` | `deploy/full/deploy.sh backup /backup/vbt` (a dated `tar.gz`), or a filesystem snapshot | the records and the host configuration cannot be recreated |
| `secrets.env` | your secret store | not in the backup on purpose |
| `data/` | re-acquirable (`vbt setup` fetches and verifies checksums); keep an archive copy of releases cited by runs you must reproduce | large |
| `models/` | re-downloadable | large |

Restore: unpack into `$VBT_HOME`, then `deploy.sh up` (setup resumes; the data steps rerun only if data changed).

### 7.4 Secrets

Secrets live only in `$VBT_HOME/secrets.env` (mode 600; `VBT_SECRETS_FILE` moves it): `VBT_WEB_PASSWORD`,
`ANTHROPIC_API_KEY`, `HF_TOKEN`, `VLLM_API_KEY` (and the same value as `VBT_LLM_API_KEY`), `NCBI_API_KEY`,
`NCBI_EMAIL`. The harness containers read the whole file; the `vllm` service gets only `HF_TOKEN` and `VLLM_API_KEY`
(`deploy.sh` reads those two from it, unless the shell already sets them).
`vbt setup` never writes a secret; images, `host.env`, `host.yaml` and backups contain none. Inside the
harness, provider keys never reach the MCP servers or the agents' Bash (`src/vbt/envpolicy.py`); only the PubMed
server gets the NCBI variables. Rotate by editing the file and `deploy.sh up` (containers are recreated).

### 7.5 Certifying a host: `vbt validate`

```
vbt [--profile P ...] validate [--only|--skip STEP,..] [--depth standard|deep] [--servers S,..] [--max-tools N]
                               [--memory-tables N] [--replicate quick|full|off] [--timeout-s S] [--out DIR]
                               [--check-from check.json] [--json]
```

`vbt validate` (also `python -m vbt.validate`) runs on the host's real data and writes `validate.md` and
`validate.json` (schema `vbt.validate/1`) to `--out` (default `<state>/validate/<UTC time>`); it exits 1 when a step
fails. Each step reports PASS, FAIL, WARN or SKIPPED with the reason, and nothing in it names a dataset or a server:
cases, oracles and limits come from the descriptors, the overlays and the upstream signatures.

| Step | What it does |
|---|---|
| `host` | the plan, MemTotal, the cgroup limit, the containment, and every memory setting: configured, on this host, the rule |
| `lint` | `vbt ds lint` |
| `check` | the session check at `--depth` (deep by default), large tables each in a data child of their own; writes `check.json` for `--check-from` |
| `correctness` | the six correctness tests per server, gateway enforcing vs off, against an oracle that reads the files with pyarrow alone |
| `latency` | p50/p95 per server: enforce, off, first calls, warm overhead, witness and data-child requests |
| `memory` | the largest present tables that fit the server limit: the upstream loader's measured peak vs the estimates |
| `live` | endpoint reachability and the live servers' checks; runs whenever a descriptor declares a remote source (no `VBT_DL_NETWORK` needed), and is skipped with the reason when no remote endpoint answers |
| `replication` | an extension (`validate.extensions`, shipped: `vbt.case_studies.trial_outcomes.validate_step`): Case 1 on the Zenodo archive against the authors' tables (skipped without the archive) |
| `model` | `vbt local check` (skipped when no model server answers) |

On the 16 GB development machine (2026-10-09, a 3,000 MB server profile to stay under a 6 GB per-process-tree
cap shared with other jobs) a deep run took 829 s and its verdict was **FAIL** (exit 1). The correctness step
answered 145 cases on 9 servers with 111 correct, 17 typed refusals and 0 wrong in enforce mode, but 4 calls
(`target.get_chemical_probes`, `target.get_genetic_constraint`) were admitted and then killed at the 3,000 MB server
limit, and 13 were not answered because the `target` server had reached its OOM kill limit. Host, lint, check,
latency, memory, live and replication passed; the model step was skipped (no model server). That run used the
overlays from before Wave C, which changed how those `target` tables are read; it has not been re-run since.

### 7.6 Monitoring

`deploy.sh ps` (health checks of `vbt` and `vllm`), `vbt-host ds status <run>` (per-server memory, host budget,
calibrations), `vbt setup --status` (last outcome of every step), `state/logs/`, `runs/INDEX.md`.

## 8. Continuous integration

| Workflow | When | What |
|---|---|---|
| `.github/workflows/ci.yml` | every push and pull request | `ruff check`, shellcheck and hadolint of the deployment scripts, `docker compose config` of both compose files, and the offline suite in two shards (`tests/datalayer`, the rest) with a pip cache |
| `.github/workflows/real-data.yml` | by hand | `VBT_DL_NETWORK=1` on a hosted runner (live APIs) and/or `VBT_DL_REAL_DATA=<dir>` on a self-hosted runner that holds the releases (labels and directory are inputs) |
| `.github/workflows/image.yml` | by hand, version tags, and changes of the image's inputs on `main` | builds the image, smoke-tests it (`setup --plan`, a mock session, `doctor --analysis`), and publishes version tags to the GitHub container registry |

## 9. What was verified, and what was not

Built and run on a 4-CPU, 16 GB development machine without a GPU, on 2026-10-08/09 (another job shared the
machine, so times are approximate):

* **Image.** `deploy.sh build` from a clean clone with the submodule at its pinned commit: the locked environment
  (384 conda + 87 pip packages) installed in 82 s with a warm network path, `pip check` clean, no pip package
  replacing a conda one, 31 imports of the harness, the upstream servers and the analysis stack passing; image
  3.66 GB. Inside it, `vbt doctor --analysis` imports scanpy, PyDESeq2, gseapy, decoupler, LIANA, harmonypy,
  lifelines and rpy2, and R loads lme4, lmerTest, glmmTMB, betareg and MuMIn.
* **`vbt setup`** ran end to end on the real Open Targets 25.09 release (the 31 downloaded tables, 1.76 GB), on the
  host and inside the image (`docker run --memory 6g`): probe, configure, size, index, check, calibrate and the
  offline part of smoke; the next run re-ran `size` (42 s), because `configure` had raised the recorded server limit
  from 6,569 to 8,212 MB, and a run after that skipped every data step (18 s on the host, 24 s in the image). In that
  build (D1's branch, before the acquisition engine was merged) `acquire` reported `vbt data acquire` unavailable;
  the engine is part of the harness now (next item).

  | Step (host run) | Time | Seconds per GB | Result |
  |---|---:|---:|---|
  | size (`vbt ds estimate`, 25 tables) | 19 s | 10.8 | genetics ~16 GB of whole-table loads (the colocalisation tables absent) |
  | index (`vbt ds index build`) | 343 s | 194.5 | 20 indexes built, 7 without data on this host |
  | check (`vbt ds check`, 48 tables) | 143 s | 81.2 | 38 ready, 10 missing (not downloaded); 14 tools unready |
  | calibrate (25 tables) | 60 s | 34.0 | |
  | whole run | 585 s | | peak RSS of the process tree 2,663 MB |

* **Compose.** `deploy.sh up` without a GPU, with an empty `$VBT_HOME` and the mock profile standing in for the
  model: SearxNG up and healthy, setup in the image, `compose up`, then the smoke step, 36 s in all, exit 0 (smoke
  deferred: no model server). The `vbt` container reported healthy, `/api/auth` answered 200, a wrong password 401
  and the right one 200, `/api/runs` 200 with the session cookie; `deploy.sh backup` and `down` worked. Inside the
  image, `tests/test_setup.py` passes (41 passed, 2 skipped: no `.github`, no Docker in the container). Two
  problems found on the way are fixed here: the SearxNG image listens on `[::]` and exits on a host without IPv6
  (compose sets `GRANIAN_HOST=0.0.0.0`), and a fixed `ulimits:` above the Docker daemon's own limit stopped the
  containers from starting (removed). `vbt doctor --analysis` reported every R package after the first as missing
  (its check read one `cat` of a vector; it now prints one line per package); the smoke step still loads each
  reported package with its own `Rscript` call and fails only on one R cannot load (all five load in the image).
* **With `vbt data acquire`.** In a scratch merge with the acquisition change (then a separate branch, merged since),
  `vbt setup --plan`
  took its acquire row from `vbt data acquire --for-tools <85 tools> --plan --json`: 117.68 GB to fetch for the
  default roster (Open Targets 25.09: 28.82 GB in 2,916 files; Tahoe-100M: 88.86 GB in 1,030 files). A run limited
  to two small tables (`setup.steps.acquire` in a profile) downloaded 229.58 KB from the EBI FTP, verified against
  the release's checksums, wrote `OPEN_TARGETS_DATA_PATH` into `host.env`, and the next run found the step up to date.
* **Not verified here:** vLLM in compose (no GPU), the Apptainer build (no Apptainer), the five largest whole-table
  loads and the seven tables that were not downloaded (estimates only, section 2.2), a published image.

## 10. Reference

| Variable | Set by | Meaning |
|---|---|---|
| `VBT_HOME` | operator | deployment root; `/srv/vbt` in the image |
| `VBT_STATE_DIR`, `VBT_DATA_DIR`, `VBT_PROJECTS_DIR`, `HF_CACHE` | operator (optional) | move one directory out of `VBT_HOME` |
| `VBT_BASE_PROFILES` | operator (compose: `production`) | profiles of every command |
| `VBT_PROFILES` | `vbt setup` (host.env) | the session profiles: harness profile, base profiles, `host.yaml` |
| `VBT_SERVING_PROFILE`, `VBT_LLM_DP_SIZE`, `VBT_LLM_BASE_URL`, `SEARXNG_URL` | `vbt setup` (host.env) | the model server and search |
| `OPEN_TARGETS_DATA_PATH`, `TAHOE_DATA_PATH`, ... | `vbt setup` (host.env), from each descriptor's `root` | data roots: what `vbt data acquire` reported, else the value already set, else `<data>/sources/<source>/<release>` (its default home) |
| `VBT_SECRETS_FILE`, `VBT_IMAGE`, `VBT_WEB_BIND`, `VBT_WEB_PORT`, `VBT_UID`, `VBT_GID`, `SEARXNG_IMAGE`, `VLLM_IMAGE`, `VLLM_GPU`, `VLLM_BIND`, `VLLM_PORT` | operator | compose and `deploy.sh` |

Related: [LOCAL_LLM.md](LOCAL_LLM.md) (model choice and serving profiles), [deploy/local/README.md](../deploy/local/README.md)
(the model server alone), [DATA_LAYER_REAL_DATA.md](DATA_LAYER_REAL_DATA.md) (measured memory and latency),
[DATA_LAYER_RUNBOOK.md](DATA_LAYER_RUNBOOK.md) (from a symptom to the command).
