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
| Model server | `llama-server` 0.5.0-dev (llama.cpp commit `0c1e570`, the copy vendored in the `llama-cpp-python` 0.3.36 sdist, built with CMake/GCC 13, `-DGGML_NATIVE=ON`), 2 slots of 65,536 tokens (`-c 131072 -np 2`), `--jinja --reasoning-format deepseek --cache-ram 2048` |
| Model | `unsloth/Qwen3.5-2B-GGUF` at revision `f6d5376be1edb4d416d56da11e5397a961aca8ae`, file `Qwen3.5-2B-Q4_K_M.gguf` (1,280,835,840 bytes, sha256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`, the Hub's LFS id), served as `qwen3.5-2b`; the adapter resolves it to the `qwen3_6` family (the Qwen3.5/3.6 dialect, same `qwen3_coder` tool-call format and chat template structure as the default Qwen3.8 model) |
| Harness profile | `configs/profiles/e2e-cpu.yaml` (provider `llamacpp`, every tier on the small model with thinking off, 64K windows, small turn limits, orientation and review on, web off, the seven Open Targets servers) |
| Upstream servers | `target`, `disease`, `drug`, `association`, `genetics`, `interaction`, `pathway` from `third_party/TheVirtualBiotech` at `71f9da68`, unmodified, each launched through the reaper |
| Data | Open Targets 25.09: the 31 tables already downloaded (697 Parquet files, 1.9 GB on disk), `OPEN_TARGETS_DATA_PATH` pointing at them; the 7 large tables not downloaded (`evidence`, `variant`, `credible_set`, `colocalisation_coloc`, `colocalisation_ecaviar`, `interval`, `literature`) make 11 of the 49 granted data tools unready |
| Data layer | gateway `enforce` for every server, the data child (`src/vbt/datalayer/service/server.py`) under the reaper, memory limits `auto` scaled from `VBT_HOST_MEMORY_MB=6000` (the harness's share of a host whose model server holds ~3 GB) |
| Host | 4 vCPUs (AVX-512), 15.7 GiB RAM (16,094 MiB) in a 13,680 MiB memory cgroup, no GPU, no swap |

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
  --model Qwen3.5-2B-Q4_K_M.gguf --served-name qwen3.5-2b --port 8012 --slots 2 --max-model-len 131072 \
  -- --cache-ram 2048 -t 4
#   == llama-server -m Qwen3.5-2B-Q4_K_M.gguf --alias qwen3.5-2b --host 127.0.0.1 --port 8012 -c 131072 --jinja \
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

## 6. Harness bugs the run found

Each was fixed at its root and has a regression test in `tests/test_e2e_stack.py` (offline, on fixtures).

| What the run showed | Cause | Fix | Test |
|---|---|---|---|
| At every session start the data child computed a second readiness check of every table (first session: the data child at 81% CPU and 1,454 MB RSS, still computing after the harness had stopped waiting for it) | the orchestrator handed the preflight's readiness to the gateway only after `start_mcp()` had listed the tools, and listing the data child's tools without readiness asks the child for a check | `Runtime.start_mcp(readiness=...)` sets it on the gateway before the bridge lists; the orchestrator passes the preflight's (after the fix the data child stayed at 114 MB through session start) | `test_the_preflight_check_reaches_the_gateway_before_the_data_child_lists_its_tools` |
| Every Task to the genomics analyst ended `context_exceeded` with no tool call (three times in one turn) | its 49 granted tools were 345,188 characters of definitions (llama-server counted 117,265 tokens for its first request): `mcp__data__find` and `mcp__data__aggregate` each listed every table's column map (`x-vbt-where`, 131,733 characters each on Open Targets 25.09) | the gateway lists the native verbs without the map (`compact_native_listing`): the `where` description points at `mcp__data__describe(source, table)`, which returns one table's columns, roles, identifier types and operators; the data child still resolves `where` against the catalog. The analyst's tools went from 345,188 to 84,230 characters, the pathways analyst's from 315,988 to 55,030 | `test_the_native_data_tools_are_listed_without_every_tables_column_map` |
| That failure was reported as "The conversation no longer fits the model's context window" for an agent that had made no call | the overflow text did not distinguish a long conversation from a fixed part (system prompt and tool definitions) that compaction cannot shrink | the report names the system prompt's and the tool definitions' estimated tokens against the window and the two remedies (a larger window, fewer tools), and the trace gets a `context_fixed_part` event | `test_an_agent_whose_tools_fill_the_window_says_so` |
| When the model server was down, the `llamacpp` provider told the user to run `vbt local serve` (which starts vLLM on a GPU) | one hint for every OpenAI-compatible provider | `serve_hint(provider)`: the llama.cpp provider names llama-server and `scripts/dev/cpu_server.sh --engine llamacpp` | `test_a_server_that_is_down_is_named_with_the_command_that_starts_it` |
| `vbt ds retro-audit <run>` did not find a project's run | the data commands that read a recorded run had no `--project`, so they looked under the default runs directory | `--project` on `ds retro-audit`, `ds replay`, `ds graduate` and `ds status`, activating the project as `vbt run --project` does | `test_the_commands_that_read_a_project_run_take_the_project` |
| The data engineer's `RegisterDataSpec` of `InspectDataset`'s own draft failed lint: `'#GeneID' is not a valid path: empty segment at 0` (ClinVar's header) | the draft wrote column references bare, and a reference is parsed as a path | references that are not plain words are written backtick-quoted (`_ref`); the column keeps its literal name. The real file then registered as drafted and checked `ready` in 4.6 s | `test_a_header_that_is_not_a_word_registers_as_drafted` |
| `vbt ds retro-audit` reported the gateway's recorded `too_large` refusal as `source_error`, and `vbt ds graduate ... data` always failed `observe_evidence` ("no recorded call of this server") | retro-audit re-classified the refusal's text instead of keeping the typed kind the enforcing gateway recorded, and it skips the data child's own verbs (served from the descriptors, nothing upstream to re-check), so graduation never saw them | a recorded typed refusal keeps its kind; the report counts the native calls and lists outcomes beyond the classify set; graduation's observe item is not applicable for the data child | `test_an_enforced_runs_refusals_and_native_calls_audit_as_recorded` |

## 7. Tests

`tests/test_e2e_stack.py` has three layers:

* **Unit regressions** for each bug above (seconds; no servers).
* **The stack on fixtures** (`test_session_1_research_through_the_whole_stack`,
  `test_session_2_the_engineer_creates_what_a_specialist_then_uses`): the same two sessions driven by a scripted model
  (the mock provider, named so that the session preflight runs as for a served model) through the real stack: the
  `e2e-cpu` profile, a project made by `init_project`, the unmodified upstream servers launched by the reaper, the
  gateway in enforce mode and the data child, on the Open Targets fixture of `tests/datalayer/dl_fixtures.py` and a
  10-row ClinVar fixture. Session 1 asserts the delegations, a symbol resolved to its Ensembl id through the
  resolver, a typed `not_found`, review, claims with tool-call evidence, the MANIFEST, `audit.html`,
  `verify_run(data=True)` COMPLETE, retro-audit and graduation. Session 2 asserts that the data engineer's draft
  registers as is, the utility's tests pass and it registers as `util__condition_summary`, the specialist's
  `mcp__data__find` on the new table and its `util__condition_summary` call return the fixture's counts, and the
  project's provenance ledger and the run's receipts record both. They skip without `fastmcp`/`mcp`, `pyarrow` or
  the upstream checkout.
* **A served model** (`test_a_served_model_drives_the_stack`), opt-in: `VBT_E2E_MODEL_URL=http://127.0.0.1:8012/v1`
  (and `VBT_E2E_MODEL`, default `qwen3.5-2b`) runs session 1's question with the served model over the fixture stack
  (and, when the model only asked its clarification questions, a second turn telling it to proceed) and asserts only
  what the harness owes whatever the model does: every tool call has its end in the trace, no turn failed, the run
  record has no audit error and `audit.html` is written. A weak model's own failures (no delegation, a refused call,
  a turn limit) are not test failures.

```bash
python -m pytest -q -p no:cacheprovider tests/test_e2e_stack.py                       # offline
VBT_E2E_MODEL_URL=http://127.0.0.1:8012/v1 python -m pytest -q -p no:cacheprovider tests/test_e2e_stack.py \
  -k served_model                                                                    # with a model server
```
