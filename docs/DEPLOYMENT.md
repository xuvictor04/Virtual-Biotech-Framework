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
"killed" did not fit a 5.9 GB test child.

Per server, the tables loaded whole (estimates unless marked "measured"):

| Server | Tables loaded whole | Whole-table loads | On disk |
|---|---|---:|---:|
| genetics | l2g_prediction (12,126 MB; killed at 5.9 GB), study (measured 2,556 MB), colocalisation_coloc (25,447 MB), colocalisation_ecaviar (38,940 MB) | ~79.5 GB | 9.5 GB |
| interaction | interaction (10,327 MB; killed at 5.9 GB) | ~10.3 GB | 0.09 GB |
| association | the six association tables (measured 4,800 MB together), literature_vector (measured 119 MB) | ~4.9 GB | 0.43 GB |
| expression | expression (5,671 MB; killed at 5.9 GB), biosample (measured 97 MB) | ~5.8 GB | 0.05 GB |
| target | target (measured 3,031 MB), mouse_phenotype (measured 323 MB), target_prioritisation (measured 51 MB), openfda target reactions (7 MB) | ~3.4 GB | 0.09 GB |
| drug | known_drug (measured 606 MB), drug_indication (measured 75 MB), drug_molecule (measured 53 MB), mechanism of action, warnings, openfda drug reactions (36 MB) | ~0.8 GB | 0.02 GB |
| disease | disease (measured 199 MB), disease_phenotype (measured 285 MB), disease_hpo (34 MB) | ~0.5 GB | 0.01 GB |
| pathway | go, reactome, so (19 MB) | ~0.02 GB | <0.01 GB |
| **all** | | **~105 GB** | |

Other tables are read in bounded scans (filters pushed into pyarrow), but each of them is large when loaded:
evidence (estimate 192 GB; 30.4 M rows), variant (194 GB), interval (48 GB), literature (47 GB), credible_set
(43 GB), interaction_evidence (54 GB), target_essentiality (11 GB). If every table a server reads were loaded whole,
genetics would need ~365 GB and association ~245 GB (estimates): these are upper bounds, not the expected use.

Recommended RAM for the harness host with the full release:

| Part | RAM |
|---|---|
| Upstream servers holding every whole-table load at once (above, x1.3 admission safety) | ~137 GB |
| Data child (readiness checks, witnesses, native tools): `vbt setup` gives it 5% of RAM, 3-32 GB | 16-32 GB |
| Agents' Bash commands (8 in parallel at `data.memory.workspace_mb`, 8-64 GB each) | 64-256 GB |
| `single_cell` server (Census pulls; 2.6-3.2 GB measured for 4,771-19,782 cells; a 200,000-cell pull exceeded 4.7 GB) | 16-64 GB |
| Harness, web UI, model client | ~4 GB |
| **Total** | **256 GB minimum; 512 GB to 1 TB** for the full configuration with headroom |

The host budget (`data.memory.host_budget_mb`) caps the sum of the servers' resident memory: when a load would cross
it, the least recently used idle server is recycled, so a smaller host still works with more reloads. `vbt setup`
sizes the budget (75% of RAM minus a reserve) and the per-server limit (the largest server's whole-table loads x1.3,
step `size`) from the host it runs on and the data it finds.

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
export VBT_HOME=/srv/vbt                                          # created if missing
install -m 600 /dev/null "$VBT_HOME/secrets.env"
printf 'VBT_WEB_PASSWORD=%s\n' "$(openssl rand -base64 24)" >> "$VBT_HOME/secrets.env"
#   optional in the same file: ANTHROPIC_API_KEY, HF_TOKEN, VLLM_API_KEY, NCBI_API_KEY, NCBI_EMAIL

deploy/full/deploy.sh up
```

`up` builds the image if it is missing (`deploy.sh build`, about 4 GB), saves the host's `nvidia-smi` output in the
state directory (the harness container sees no GPU), runs `vbt setup` in the image (section 6) with every step but
the smoke test, starts the services, waits for the model server's health check (the first start downloads the
weights and compiles CUDA graphs: 10-30 minutes; `VBT_MODEL_WAIT_S`, default 3600) and runs the smoke step.

Before the first `up`, see what it will do:

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
#   the root environment.yml (version ranges) does not solve on 2026-10-08: conda-forge has no `tiledbsoma`
#   (the package is tiledbsoma-py) and no decoupler >= 2.0; use the lock above

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
with `--llm-url` (written to `host.env`). Login nodes usually forbid memory cgroups: `vbt setup` then keeps
`RLIMIT_DATA` containment, which it reports.

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
| `size` | `vbt ds estimate --json` of the tables the enabled servers load whole; the largest server's estimate x1.3 becomes `data.memory.default_server_mb` | `host.yaml` again |
| `index` | `vbt ds index build --table ...` for every table the enabled tools read: only the id types those tables hold | `data/.vbt-datalayer/` |
| `check` | `vbt ds check --json --table ...` for every table the enabled tools read | readiness cache |
| `calibrate` | `vbt ds calibrate --table ...` for the local tables loaded whole | calibrations |
| `smoke` | an offline mock session; `vbt doctor` (`--smoke` once the model server answers, else the step is `deferred`); `vbt doctor --analysis` | logs |

Nothing in setup names a dataset: what to fetch, check and calibrate comes from the descriptors, the overlays and the
enabled roster (agents in `configs/agents.yaml`, servers enabled in `configs/mcp_servers.yaml`). Disable a server or
an agent and setup stops fetching and checking what only it reads. Data already on disk elsewhere: set its root
variable (e.g. `OPEN_TARGETS_DATA_PATH`) and run with `--skip acquire`.

Each step after `configure` is a `vbt` command run as a child under the host configuration, logged to
`state/logs/<step>.log`. A step that finished is skipped next time while its inputs are unchanged (its command, the
host configuration, the files under each data root); `--force` runs it anyway. A failed step stops the run
(`--keep-going` continues). `--plan` prints the steps with download sizes and time estimates and changes nothing; the
time of the data steps is this host's own measured seconds per GB after its first run (the reference rates below
before that). The commands are configuration: `setup.steps.<step>.run` / `.plan` in a profile replace a step's
arguments.

**Sizing** (MB; `ram` = the smaller of MemTotal and the container limit): host budget `0.75 x ram - max(2048, 0.05 x
ram)`; server limit `clamp(0.25 x ram, 12000, budget)` before `size`, then the largest server's whole-table loads x1.3
(at least 12000, at most the budget); data child `clamp(0.05 x ram, 3000, 32768)`; agents' Bash `clamp(0.25 x ram /
max_parallel_agents, 8000, 65536)`; `limit_kind: cgroup` where a memory cgroup can be created; `bash.sandbox.os:
bwrap` where bubblewrap works (Docker's default seccomp profile refuses it; `compose.yaml` shows the opt-in). On a
host under ~16 GB every value stays at its shipped default.

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

1. Fetch the new release next to the old one, e.g. `$VBT_HOME/data/open_targets/25.12` (`vbt data acquire`, or
   `vbt data ot fetch ... --release 25.12 --dest ...`).
2. `vbt ds diff-release --from $VBT_HOME/data/open_targets/25.09 --to $VBT_HOME/data/open_targets/25.12`: role
   columns, types, encodings, vocabularies and matrix axes that changed.
3. Update the descriptor (`release.expect`, any column the diff names) and lint it: `vbt ds lint --strict`.
4. Point the root at the new release (`OPEN_TARGETS_DATA_PATH` in the environment or the secrets file) and run
   `vbt setup --from configure`: `size`, `index`, `check` and `calibrate` rerun because the data changed.
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

### 7.5 Monitoring

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
  offline part of smoke; a second run skipped every data step (18 s on the host, 24 s in the image). `acquire`
  reported `vbt data acquire` unavailable in this build (the acquisition engine is a separate change).

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
* **With `vbt data acquire`.** In a scratch merge with the acquisition change (a separate branch), `vbt setup --plan`
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
