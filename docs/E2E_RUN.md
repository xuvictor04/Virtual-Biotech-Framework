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
| Model server | `llama-server` 0.5.0-dev (llama.cpp commit `0c1e570`, the copy vendored in the `llama-cpp-python` 0.3.36 sdist, built with CMake/GCC 13, `-DGGML_NATIVE=ON`), 2 slots of 65,536 tokens (`-c 131072 -np 2`), `--jinja --reasoning-format deepseek --cache-ram 2048` (1024 after the restart of section 3) |
| Model | `unsloth/Qwen3.5-2B-GGUF` at revision `f6d5376be1edb4d416d56da11e5397a961aca8ae`, file `Qwen3.5-2B-Q4_K_M.gguf` (1,280,835,840 bytes, sha256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`, the Hub's LFS id), served as `qwen3.5-2b`; the adapter resolves it to the `qwen3_6` family (the Qwen3.5/3.6 dialect, same `qwen3_coder` tool-call format and chat template structure as the default Qwen3.8 model) |
| Harness profile | `configs/profiles/e2e-cpu.yaml` (provider `llamacpp`, every tier on the small model with thinking off, 64K windows, small turn limits, orientation and review on, web off, the seven Open Targets servers) |
| Upstream servers | `target`, `disease`, `drug`, `association`, `genetics`, `interaction`, `pathway` from `third_party/TheVirtualBiotech` at `71f9da68`, unmodified, each launched through the reaper |
| Data | Open Targets 25.09: the 31 tables already downloaded (697 Parquet files, 1.9 GB on disk), `OPEN_TARGETS_DATA_PATH` pointing at them; the 7 large tables not downloaded (`evidence`, `variant`, `credible_set`, `colocalisation_coloc`, `colocalisation_ecaviar`, `interval`, `literature`) make 11 of the 49 granted data tools unready |
| Data layer | gateway `enforce` for every server, the data child (`src/vbt/datalayer/service/server.py`) under the reaper, memory limits `auto` scaled from `VBT_HOST_MEMORY_MB=6000` (the harness's share of a host whose model server held 3.5 to 5 GB) |
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

Model speed on this host (llama-server timings): one request alone prefilled 4,244 tokens at 108 tokens/s and
decoded at 11.7 tokens/s. With both slots busy the four cores are shared: concurrent prefills ran at 22 to 64
tokens/s, and a slot decoding while the other prefilled a long prompt slowed to 0.4 to 2 tokens/s (llama.cpp
processes the other slot's prompt in the same batches). An agent's first request carries 8K to 30K tokens of
system prompt and tool definitions (section 8), so a fresh prefill takes 2 to 11 minutes here; the same agent's
later calls reuse the slot's prompt cache (`cache_read_tokens` in the trace) and cost seconds to a minute.

## 3. Session 1: a research question on the real Open Targets release

The question (turn 1) asked whether PCSK9 is a genetically supported and tractable target for
hypercholesterolaemia, named the genomics analyst and the bio-pathways analyst, asked for a check of a gene called
PCSK99, and asked for review and claims. Turn 2 answered the CSO's clarification interview ("1) a, d, e. 2) a.
3) No constraints. Proceed now: use the Task tool to delegate ...").

**Served-model attempts.** Three runs drove the stack with the 2B model; each found something, fixed before the next.

| Run | What happened | Wall time |
|---|---|---|
| `20261009_085747_87706d42` | Orientation ran: the Chief of Staff's brief (7 calls, stopped at its 6-turn limit with its brief) and the CSO's clarification interview. Turn 2 answered "proceed with the analysis exactly as specified"; the CSO repeated its interview instead of delegating. The data child spent the session start computing a second readiness check (section 6). | 753 s |
| `20261009_091119_39e04cc2` | With the interview answered option by option, the CSO wrote a 3-step plan (`write_plan`) and called `Task` for the genomics analyst three times; each ended `context_exceeded` before any call: the analyst's first request was 117,265 tokens against a 32,768-token slot (section 6). Earlier in this run the model server had stopped (this sandbox's 30-minute limit on that background command); the provider's 8 retries carried turn 1 across its restart. Stopped (SIGTERM) after the third `context_exceeded`. | 1,576 s |
| `20261009_093814_0235427f` | The full run, below. | 9,339 s |

The full served-model run (`20261009_093814_0235427f`, after the listing fix, with 64K slots):

* **Orientation** (turn 1, 8 min 22 s): the Chief of Staff's brief (7 model calls; first request 8,148 tokens) and the
  CSO's clarification interview (one call, 11,260-token request).
* **Plan and delegation** (turn 2): `write_plan` with four steps, then one CSO call issuing three `Task` calls:
  genomics analyst, bio-pathways analyst and clinical trialist (the profile's two parallel agents: the trialist
  started when the pathways analyst finished).
* **Tool calls through the enforcing gateway**: 80 tool calls in all, 44 to the unmodified upstream servers and 10
  to the native data tools. The model resolved PCSK9 with `target.search_targets_by_name("PCSK9")` and
  `mcp__data__search` (ENSG00000169174) and read its Reactome pathways (`pathway.get_gene_pathways`, 4 rows, `ok`).
  The gateway's typed refusals: `not_found` for `drug.search_known_drugs(disease_id="hypercholesterolaemia")` (the
  British spelling is neither an Open Targets disease id nor a disease name; the refusal lists the ten
  resolutions it tried and suggests HP_0003124 "Hypercholesterolemia", edit distance 1); `too_large` 22 times
  (11 `over_limit`: whole-table loads such as `target.get_target_info` needing ~3,930 MB,
  `genetics.query_l2g_predictions` ~15,034 MB and `interaction.get_interactions` ~12,803 MB against the 2,048 MB
  server limit of a 6,000 MB share; 11 `host_busy`: the bug of section 6, fixed after this run);
  `not_ready` for `genetics.query_gwas_associations` (`credible_set` is not downloaded here); `service_unavailable`
  for `pathway.get_go_enrichment` (the derived serve needs the GO ontology file, which is not on this host); `empty`
  and `partial` results with their totals ("top 10 of 16"). Four calls the harness refused before they ran: the
  model passed `include_indirect` as a string (`"false"`), and the schema check said so.
* **The model server restart**: at 10:06 the llama-server process passed the 5,000 MB memory cap this host's run gave
  it and was killed; the provider's retries (`provider_retry`, then HTTP 503 "Loading model" while it reloaded)
  carried both specialists across the restart.
* **Review**: each specialist stopped at its 8-turn limit with a report. The CSO then tried to end the turn
  without review; the harness enforced it (`review_enforced`, round 1), the CSO delegated to the scientific reviewer
  (one call, 5,915-token request, a 1,280-token review naming the missing genetic metrics as a technical gap), and
  the CSO sent the genomics analyst and the clinical trialist back for a second round, as the review asked. In that
  round the genomics analyst's context was summarised four times (section 8).
* **The end**: the CSO's second review request was in flight when the model server was stopped at 12:07 by this
  sandbox's two-hour limit on background commands. The provider retried for about six minutes with the llama-server
  start command in its message (section 6) and the turn ended `failed`; the run record was written as such
  (MANIFEST `incomplete`, `audit.html`). The CSO had used 13 of its 14 turns; no claims had been filed.

Peak memory of the harness's process tree: 4,056 MB (data child 2,089 MB at most, each idle upstream server about
172 MB, the association server 739 MB after its one admitted whole-table call).

**The same session driven by the scripted model over the real stack** (run `20261009_094151_882f50d7`): the steps
the small model did not reach, on the same release, servers, gateway, data child and project. A small script outside
the repository opened the session exactly as `vbt --profile e2e-cpu run --project e2e` does, with the scripted
provider of `tests/test_e2e_stack.py` (named so that the session preflight runs as for a served model) and that
file's session rules adapted to the real release; section 7's fixture tests are its repository form.
Session start 7.8 s, the turn 26.7 s, peak tree RSS 2,380 MB.

* `target.get_target_info(target_id="PCSK99")`: `not_found`, with what was tried (`ensembl_gene`, then the
  approved symbol exactly and case-folded, previous symbols, aliases) and a suggestion (PCSK9, edit distance 1).
* `pathway.get_gene_pathways(target_id="PCSK9")`: `ok`, 4 Reactome pathways; the gateway resolved the symbol
  (`PCSK9 -> ENSG00000169174 (label_exact:approvedSymbol)`, recorded in the result's `_vbt.resolved`).
* `target.get_target_info(target_id="PCSK9")`: `too_large` / `over_limit` (3,930 MB needed), so the analyst read
  the target row with `mcp__data__find` (`ok`) and its direct associations with `mcp__data__find(rank_by="score
  desc", limit=5)`: `partial`, "top 5 of 992", MONDO_0011369 first (score 0.816).
* Review, then `record_claims` with two claims, each citing a tool call (`ok: true`); the answer cites
  `[[claim:C1]]` and `[[claim:C2]]`, and the harness appended its data warning naming the refused tool.

## 4. Session 2: a dataset the harness has no descriptor for

The user downloaded ClinVar's `gene_condition_source_id`
(https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id, Last-Modified 2026-10-07; 1,323,448 bytes,
sha256 `36ea1e52...a1206`, 14,211 rows x 9 columns, 484 exact duplicate rows) into the project's `incoming/`
directory and asked which conditions ClinVar associates with PCSK9, asking for the data engineer to register it with
a small tested helper utility and the genomics analyst to answer with both.

**Served model** (run `20261009_121711_f6caab8d`, a fresh project `e2e-b` made with `vbt project init` so that
the scripted run's registrations below could not pre-empt it; 3,130 s, peak tree RSS 1,594 MB). The two test suites
ran at low priority on the same cores during part of it.

* Orientation (turn 1, 17 min): the Chief of Staff's brief (7 calls; it read the file and counted its rows with
  `Bash`) and the CSO's interview.
* Turn 2: the CSO delegated to the data engineer (`Task`, after a `TodoWrite`). The engineer (first request 16,435
  tokens) never called `InspectDataset`: over its 24-turn budget it read the file, wrote YAML files of its own
  invention (`kind: acquisition`, `kind: descriptor` with made-up fields) under its work directory, and called
  `RegisterDataSpec` once with a data path it had invented (`work/data-engineer/incoming/...`), which the tool
  refused ("no such file or directory to import"). It stopped at its turn limit; the CSO sent it back with the same
  task, and the second attempt repeated the exploration. The turn was then interrupted with one SIGINT (Ctrl-C), 35
  minutes into turn 2: it was recorded `interrupted`, the agents `cancelled`, and the run record and `audit.html`
  were written (`verify --data`: `interrupted_turn`, `degraded_run`, no claims; 27 s, 1,061 MB).

The 2B model cannot drive the data engineer's steps, so they were driven by the scripted model over the same real
stack:

**Driven by the scripted model over the real stack** (the same script with the test file's session-2 rules, run
`20261009_104536_5d78baac`, project `e2e`; session start 8.6 s, the turn 12.2 s, peak tree RSS 1,778 MB):

* The data engineer: `ProjectInfo`, `InspectDataset` on the real file (14,211 rows profiled, `#GeneID` int64 with
  5,179 distinct values, `SourceName` 8 values, key drafted as (`#GeneID`, `DiseaseName`, `SourceID`) with
  `row_identity: none` for the repeated rows), `RegisterDataSpec` of the draft as is: registered (version 1) and
  checked `ready` (`clinvar_gcs.gene_conditions`), "available now: the data tools serve it in this session".
* The utility: `Write` of `utilities/condition_summary/utility.py` and its test file, then `RegisterUtility`: its 2
  tests passed in the sandbox (`bwrap+netns`, 0.11 s), registered as `util__condition_summary`.
* The genomics analyst, in the same turn: `mcp__data__find(table="clinvar_gcs.gene_conditions",
  where={"AssociatedGenes": "PCSK9"})`: 2 of 2 rows (Familial hypercholesterolemia, MONDO:0005439;
  Hypercholesterolemia, autosomal dominant, 3, MONDO:0011369), then `util__condition_summary` on them:
  `n_conditions 2`, `by_source {"MONDO": 2}`.
* Review, one claim citing the find, the answer citing `[[claim:C1]]`. The project's provenance ledger
  (`provenance/ledger.jsonl`) records both registrations with the agent, run and tool call; the trace has
  `project_registration` (2) and `project_utility_call` (1) events.

## 5. After the run: verify, retro-audit, graduate

| Command | Run | Result | Time, peak RSS |
|---|---|---|---|
| `verify --data` | served-model session 1 | `INCOMPLETE`, 19 problems: the failed turn, 14 refused data calls (`data_source_unavailable`), `degraded_run`, no claims; 29 tables re-checked, 0 changed | 26 s, 1,071 MB |
| `verify --data` | scripted session 1 | `INCOMPLETE`, 2 problems: the refused `get_target_info` and `degraded_run`; 30 tables checked, 0 changed; replays: match 2 (both cited calls replayed with the same rows) | 74 s, 1,302 MB |
| `verify --data` | scripted session 2 | `INCOMPLETE`, 1 problem: `degraded_run`; 29 tables checked, 0 changed; replays: match 1 | 33 s, 1,002 MB |
| `ds retro-audit` | served-model session 1 | 44 upstream calls: ok 14, not_found 1, invalid_argument 5, not_ready 1, service_unavailable 1, too_large 22; none would now be refused or qualified; 10 native calls counted | 5.2 s, 322 MB |
| `ds retro-audit` | scripted session 1 | ok 1, not_found 1, too_large 1; 2 native calls | 4.6 s, 266 MB |
| `ds graduate ... data` | served-model session 1 | target, pathway, drug, association, genetics, interaction and data graduated (every overlay lints, every tool reviewed, observe evidence from the run) | 6.6 s, 362 MB |

`degraded_run` is this host's partial release (section 8); every other problem is a real gap of that run. The
numbers in the retro-audit row are after the retro-audit fixes of section 6; before them the same run read 4
`source_error`, and one empty search was reported as now refused.

## 6. Harness bugs the run found

Each was fixed at its root and has a regression test in `tests/test_e2e_stack.py` (offline, on fixtures).

| What the run showed | Cause | Fix | Test |
|---|---|---|---|
| At every session start the data child computed a second readiness check of every table (first session: the data child at 81% CPU and 1,454 MB RSS, still computing after the harness had stopped waiting for it) | the orchestrator handed the preflight's readiness to the gateway only after `start_mcp()` had listed the tools, and listing the data child's tools without readiness asks the child for a check | `Runtime.start_mcp(readiness=...)` sets it on the gateway before the bridge lists; the orchestrator passes the preflight's (after the fix the data child stayed at 114 MB through session start) | `test_the_preflight_check_reaches_the_gateway_before_the_data_child_lists_its_tools` |
| Every Task to the genomics analyst ended `context_exceeded` with no tool call (three times in one turn) | its 49 granted tools were 345,188 characters of definitions (llama-server counted 117,265 tokens for its first request): `mcp__data__find` and `mcp__data__aggregate` each listed every table's column map (`x-vbt-where`, 131,733 characters each on Open Targets 25.09) | the gateway lists the native verbs without the map (`compact_native_listing`): the `where` description points at `mcp__data__describe(source, table)`, which returns one table's columns, roles, identifier types and operators; the data child still resolves `where` against the catalog. The analyst's tools went from 345,188 to 84,230 characters, the pathways analyst's from 315,988 to 55,030 | `test_the_native_data_tools_are_listed_without_every_tables_column_map` |
| That failure was reported as "The conversation no longer fits the model's context window" for an agent that had made no call | the overflow text did not distinguish a long conversation from a fixed part (system prompt and tool definitions) that compaction cannot shrink | the report names the system prompt's and the tool definitions' estimated tokens against the window and the two remedies (a larger window, fewer tools), and the trace gets a `context_fixed_part` event | `test_an_agent_whose_tools_fill_the_window_says_so` |
| Once the data child held 1,574 MB (after a few native finds), every upstream call, even one needing 3 MB, was refused `too_large` / `host_busy` ("servers hold about 2,761 MB ... no idle server is left to recycle") | the host budget (`0.75 x share - harness reserve`, 2,452 MB here) is the upstream servers' share and the reserve already covers the data child, but the gateway registered its own child with the admission ledger and the budget's server list, so the child's memory was charged twice and it can never be recycled as an idle upstream server | the gateway keeps the data child out of the upstream admission ledger and the host budget's server list (it stays under its own limit, `data.service.mem_limit_mb`) | `test_the_data_childs_memory_is_not_charged_to_the_upstream_host_budget` |
| When the model server was down, the `llamacpp` provider told the user to run `vbt local serve` (which starts vLLM on a GPU) | one hint for every OpenAI-compatible provider | `serve_hint(provider)`: the llama.cpp provider names llama-server and `scripts/dev/cpu_server.sh --engine llamacpp` | `test_a_server_that_is_down_is_named_with_the_command_that_starts_it` |
| `vbt ds retro-audit <run>` did not find a project's run | the data commands that read a recorded run had no `--project`, so they looked under the default runs directory | `--project` on `ds retro-audit`, `ds replay`, `ds graduate` and `ds status`, activating the project as `vbt run --project` does | `test_the_commands_that_read_a_project_run_take_the_project` |
| The data engineer's `RegisterDataSpec` of `InspectDataset`'s own draft failed lint: `'#GeneID' is not a valid path: empty segment at 0` (ClinVar's header) | the draft wrote column references bare, and a reference is parsed as a path | references that are not plain words are written backtick-quoted (`_ref`); the column keeps its literal name. The real file then registered as drafted and checked `ready` in 4.6 s | `test_a_header_that_is_not_a_word_registers_as_drafted` |
| `vbt ds retro-audit` of the served-model run reported four calls the harness had refused before they ran (an argument that did not match the tool's schema) as `source_error`, and `search_go_terms(query="cholesterol")`, an empty search when recorded, as now refused `invalid_argument` ("not a valid go_term") | retro-audit read the harness's refusal text as the source's error, and it resolved every argument bound to an identifier column, free-text search arguments included, which the gateway matches as text and never resolves | a call recorded with `model_error` is `invalid_argument` ("refused before the call ran"); free-text arguments are not resolved | `test_retro_audit_blames_neither_the_source_nor_a_search_text` |
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

On this host the offline file ran 16 passed, 1 skipped in 103 s (another test shared the cores; peak 1,788 MB). The
served-model test passed against the 2B model in 582 s (peak 1,644 MB): the Chief of Staff's brief (7 calls) and the
CSO's interview, then the second turn, where the CSO answered without delegating, as in the first served run of
section 3.

```bash
python -m pytest -q -p no:cacheprovider tests/test_e2e_stack.py                       # offline
VBT_E2E_MODEL_URL=http://127.0.0.1:8012/v1 python -m pytest -q -p no:cacheprovider tests/test_e2e_stack.py \
  -k served_model                                                                    # with a model server
```

## 8. For a production host

What this run measured that matters when the owners deploy on their GPUs:

* **Context per request.** An agent's first request carries its system prompt and every granted tool's definition
  before any conversation: on this roster (after the listing fix of section 6) the CSO's was 15,724 tokens (with
  the brief; 11,260 for its tool-less clarification interview), the
  Chief of Staff's 8,148, the pathways analyst's 21,016 and the genomics analyst's 29,699. Serve every tier with a
  window of at least 64K (the production profiles' 262K windows hold them easily); a window that cannot hold an
  agent's fixed part now says so by name instead of failing as an overflow.
* **Whole-table upstream loads and the memory share.** The unmodified Open Targets servers load whole tables for
  several tools (`target.get_target_info`, `get_target_tractability`, `interaction.get_interactions`,
  `genetics.query_l2g_predictions` ...). Under a 6,000 MB share the per-server limit is 2,048 MB and admission
  refused them `too_large` before they ran ("this call needs about 3,930 MB ... a host with about 10 GB admits it"),
  which is the designed behaviour: the agents then used the native data tools (`mcp__data__find`, `search`,
  `describe`), which read with filters. Give the harness its real share with `VBT_HOST_MEMORY_MB` or
  `data.memory.host_mb` on a host whose model server holds RAM, and leave the limits `auto`.
* **Readiness on a partial release.** With 7 of the 38 Open Targets tables not downloaded, 11 tools are unready
  (refused `not_ready` with how to acquire the data) and `vbt verify --data` reports the run `INCOMPLETE`
  (`degraded_run`). A host with the full release (`vbt data acquire open_targets`) has neither.
* **Prompt caching.** The Chief of Staff's and each specialist's later calls reuse the server's prompt cache (seconds
  each here). The CSO's clarification interview is a call without tools, so the CSO's first call of the next turn
  shares no prefix with it and is prefilled in full (15,724 tokens, about three minutes on this CPU; a fraction of
  a second on a GPU).
* **Compaction against a large fixed part.** The context thresholds count the whole request, system prompt and tool
  definitions included, which compaction cannot shrink. At the first profile's 0.55 / 0.7 of a 64K window the
  genomics analyst (about 30K of fixed part) was summarised four times in one task, each summary removing at most
  ~2,100 tokens and costing a model call of minutes on this CPU. The profile now uses 0.75 / 0.85; the production
  windows leave room either way. The thresholds would better count only what compaction can remove (contract
  request in the D4 report).
* **Slots and prompt caches.** With two slots and three or more agents working, a slot's cache belongs to whichever
  agent ran there last: the genomics analyst's first call of its second task prefilled its whole 37,482-token
  prompt again. On CPU a long prefill in one slot also slows the other slot's decoding to 0.4 to 2 tokens/s. A
  production server with more slots (vLLM's prefix cache is shared across requests) does not have either cost.
* **Model server memory.** llama-server's resident memory grew with both 64K slots in use and the prompt cache:
  it reached 5,017 MB, over the 5,000 MB cap this host's run gave it, and was restarted mid-session; the provider's
  retries (`provider_retry` events, HTTP 503 "Loading model" while it reloaded) carried the session across the
  restart without losing a call. Size the model server's memory for `slots x window` plus `--cache-ram`.
* **llama.cpp specifics.** Start llama-server with `--jinja` (tool calls) and, for the Qwen models' reasoning
  stream, `--reasoning-format deepseek`; it ignores the per-request thinking budget (only `--reasoning-budget`
  applies), so the CPU profile turns thinking off.
