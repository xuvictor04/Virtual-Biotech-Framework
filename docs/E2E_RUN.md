# End-to-end run of the whole stack on CPU

This page records the first run of the **whole** harness with a real model: a small open model from the default
model's family served OpenAI-compatibly by llama.cpp on CPU, driving real sessions through the unmodified upstream
MCP servers that have data, the data gateway in enforce mode, the data child, the reaper, a project, the audit
record and the post-run commands. It proves the mechanics before the owners run the production model on their GPUs;
a 2B model's answers are weak by design, and every place where it could not drive a step is listed with how that
step was driven instead.

Everything below is a command in this repository. Nothing depends on the machine it was first run on: the server URL,
the model name, the data paths and the host's memory come from the environment, and memory limits stay `auto`.

* [1. What ran](#1-what-ran)
* [2. Commands](#2-commands)
* [3. Session 1: a research question on the real Open Targets release](#3-session-1-a-research-question-on-the-real-open-targets-release)
* [4. Session 2: a dataset the harness has no descriptor for](#4-session-2-a-dataset-the-harness-has-no-descriptor-for)
* [5. After the run: verify, retro-audit, graduate](#5-after-the-run-verify-retro-audit-graduate)
* [6. Harness bugs the run found](#6-harness-bugs-the-run-found)
* [7. Tests](#7-tests)
* [8. For a production host](#8-for-a-production-host)

## 1. What ran

| Piece | What |
|---|---|
| Model server | `llama-server` 0.5.0-dev (llama.cpp commit `0c1e570`, the copy vendored in the `llama-cpp-python` 0.3.36 sdist, built with CMake/GCC 13, `-DGGML_NATIVE=ON`), 2 slots of 32,768 tokens, `--jinja --reasoning-format deepseek --cache-ram 2048` |
| Model | `unsloth/Qwen3.5-2B-GGUF` at revision `f6d5376be1edb4d416d56da11e5397a961aca8ae`, file `Qwen3.5-2B-Q4_K_M.gguf` (1,280,835,840 bytes, sha256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`, the Hub's LFS id), served as `qwen3.5-2b`; the adapter resolves it to the `qwen3_6` family (the Qwen3.5/3.6 dialect, same `qwen3_coder` tool-call format and chat template structure as the default Qwen3.8 model) |
| Harness profile | `configs/profiles/e2e-cpu.yaml` (provider `llamacpp`, every tier on the small model with thinking off, 32K windows, small turn limits, orientation and review on, web off, the seven Open Targets servers) |
| Upstream servers | `target`, `disease`, `drug`, `association`, `genetics`, `interaction`, `pathway` from `third_party/TheVirtualBiotech` at `71f9da68`, unmodified, each launched through the reaper |
| Data | Open Targets 25.09: the 31 tables already downloaded (697 Parquet files, 1.9 GB on disk), `OPEN_TARGETS_DATA_PATH` pointing at them; the 7 large tables not downloaded (`evidence`, `variant`, `credible_set`, `colocalisation_coloc`, `colocalisation_ecaviar`, `interval`, `literature`) make 11 of the 49 granted data tools unready |
| Data layer | gateway `enforce` for every server, the data child (`src/vbt/datalayer/service/server.py`) under the reaper, memory limits `auto` scaled from `VBT_HOST_MEMORY_MB=6000` (the harness's share of a host whose model server holds ~3 GB) |
| Host | 4 vCPUs (AVX-512), 15.7 GB RAM in a 13.4 GB memory cgroup, no GPU, no swap |

## 2. Commands

```bash
# 1. the model server (any OpenAI-compatible server with tool calls works; llama.cpp is the CPU choice)
pip download --no-deps --no-binary :all: llama-cpp-python==0.3.36 && tar xzf llama_cpp_python-0.3.36.tar.gz
cmake -S llama_cpp_python-0.3.36/vendor/llama.cpp -B llama-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build llama-build --target llama-server            # or any llama.cpp build with --jinja and --cache-ram
curl -L -o Qwen3.5-2B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf
LLAMA_SERVER=llama-build/bin/llama-server scripts/dev/cpu_server.sh --engine llamacpp \
  --model Qwen3.5-2B-Q4_K_M.gguf --served-name qwen3.5-2b --port 8012 --slots 2 --max-model-len 65536 \
  -- --cache-ram 2048 -t 4
#   == llama-server -m Qwen3.5-2B-Q4_K_M.gguf --alias qwen3.5-2b --host 127.0.0.1 --port 8012 -c 65536 --jinja \
#      --reasoning-format deepseek -np 2 --metrics --cache-ram 2048 -t 4

# 2. the harness on this host
export OPEN_TARGETS_DATA_PATH=/data/open_targets/25.09     # the downloaded release (vbt data acquire open_targets)
export VBT_LLM_BASE_URL=http://127.0.0.1:8012/v1           # the profile's default
export VBT_HOST_MEMORY_MB=6000                             # only where the model server shares the host's RAM
vbt --profile e2e-cpu ds index build                       # resolver indexes (once per release)
vbt --profile e2e-cpu doctor --data                        # readiness of every granted tool (cached for sessions)
vbt --profile e2e-cpu project init e2e --description "End-to-end CPU smoke project"

# 3. session 1 (two turns: the CSO's clarification interview, then the answer to it)
vbt --profile e2e-cpu run --project e2e --events ndjson "<question>" "<answer to the clarification>" > s1.ndjson

# 4. session 2 (the user's file is in the project's incoming/ directory, a read root of every agent)
curl -L -o <projects>/e2e/incoming/gene_condition_source_id.tsv \
  https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id
vbt --profile e2e-cpu run --project e2e --events ndjson "<question>" "<answer>" > s2.ndjson

# 5. after the run
vbt --profile e2e-cpu --profile <projects>/e2e/profile.yaml verify --data <run>
vbt --profile e2e-cpu ds retro-audit --project e2e <run>
vbt --profile e2e-cpu ds graduate --project e2e target pathway --run <run>
vbt --profile e2e-cpu ds status --project e2e <run>        # per-server memory from the reaper's status files
```

Preparation timings on this host: the index build took 4 min 40 s (16 Open Targets indexes; peak process-tree
RSS 1,864 MB; sources without data, such as DepMap here, are reported failed and exit 1), `doctor --data` 3 min 21 s
(1,186 MB). Both are cached under `$VBT_DATA_DIR/.vbt-datalayer`, so a session starts in seconds. `vbt setup` does
the same steps on a production host.

Model speed on this host (llama-server timings): prefill 108 tokens/s and decode 11.7 tokens/s for one request; two
concurrent requests share the four cores (~60 tokens/s prefill each). A specialist's first call carries 8-16K
tokens of system prompt and tools, so a fresh prefill takes 1.5-4 minutes; follow-up calls of the same agent reuse
the slot's prompt cache (`cache_read_tokens` in the trace) and cost seconds.
