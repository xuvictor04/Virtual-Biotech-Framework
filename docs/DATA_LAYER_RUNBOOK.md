# Data layer runbook: symptom → command

Operator guide for the vbt data layer (design: [DATA_LAYER.md](DATA_LAYER.md)). Each entry starts
from what you see in an agent's tool result, in `vbt verify`, `vbt doctor` or `vbt validate`, and
names the command that explains it and the fix. `vbt ds` is short for `vbt datasource`; every command
takes `--help`. Commands that read data (`check`, `fingerprint`, `index build`, `estimate`,
`calibrate`, `diff-release`) run the data child's interpreter (`${vars.mcp_python}`), so they need the
same environment as the MCP servers (`OPEN_TARGETS_DATA_PATH`, `TAHOE_DATA_PATH`, ...). On a host
brought up with `vbt setup`, every `vbt` command loads the host configuration (`host.env`, the
recorded profiles), which sets them. `vbt data` fetches and verifies the files
([DATA_SETUP.md](DATA_SETUP.md)); `vbt validate` certifies the whole host
([DEPLOYMENT.md §7.5](DEPLOYMENT.md#75-certifying-a-host-vbt-validate)).

The error kinds and payloads are those of DATA_LAYER.md §12.1. A model-side kind (`not_found`,
`ambiguous`, `invalid_argument`, `unsupported_combination`, `incomplete_key`, `unsupported_filter`,
`insufficient_resolution`) is the agent's input to fix; a data-side kind (`not_ready`, `too_large`,
`quarantined`, `tool_defect`, `oom`, `server_crashed`, `source_error`, `service_unavailable`) is
yours.

## Quick reference

| Symptom | First command |
|---|---|
| `not_ready` naming a table, column or partition | `vbt ds check --tool <server>.<tool>` |
| `not_ready`, status `missing` or `partial` (files absent or truncated) | `vbt data acquire <source>.<table>` (the refusal's `acquire` entry names it, with size and licence) |
| `not_ready`, status `stale` (`R2:release`: another release on disk) | `vbt data acquire <source> --env-file <file>` |
| `R2:manifest_absent`, files not verified | `vbt data status --check` |
| `too_large` (memory estimate; `over_limit`) | `vbt ds estimate --tool <server>.<tool>` |
| `too_large` with `limit_source: auto` | `vbt validate --only host` (the limits this host gets) |
| `too_large` subkind `host_busy` | `vbt ds status <run>` |
| `too_large` subkind `scan_budget` | `vbt ds explain <server>.<tool>` |
| `too_large` subkind `unranked_truncation` | `vbt ds explain <server>.<tool>` |
| `too_large` subkind `expansion` | `vbt ds describe <source>.<table>` |
| `oom`, `server_crashed`, recycles | `vbt ds status <run>` |
| `tool_defect` (W1-W6) | `vbt ds explain <server>.<tool>`, then `vbt ds retro-audit <run>` |
| `incomplete_key` | `vbt ds describe <source>.<table>` |
| rows `withheld` (leakage, phantom, duplicates) | `vbt ds describe <source>.<table>` |
| `data_version_drift` in `vbt verify` | `vbt ds fingerprint`, `vbt ds replay <run> --all` |
| `replay_mismatch` / `source_updated` | `vbt ds replay <run> <tool_use_id>` |
| new release to install | `vbt ds diff-release --from 25.09 --to 25.12` |
| slow witness | `vbt ds index build --access-paths --table <source>.<table>` |
| sidecar index stale or missing | `vbt ds index build --id-type <source>:<id_type> --force` |
| adding a third-party server | `vbt ds overlay init <server>` |
| switching a server from observe to enforce | `vbt ds graduate <server> --run <run>` |
| is this host fit to run the enabled agents? | `vbt validate` |

## not_ready: a table, column or partition failed readiness

The payload is `tables[{name, column?, partition?, check, detail, hint}]`. Readiness is scoped: only
the tools that read the failing part are refused; every other tool keeps working.

1. `vbt ds check --tool <server>.<tool>` shows the call-scoped readiness: the tables, columns,
   containers and partitions the tool reads, and the check (R1-R10) each failed.
2. `vbt ds check --table <source>.<table> --depth standard -V` gives the per-column and per-partition
   statuses; `--column <source>.<table>.<column>` narrows to one column.
3. By status:
   - `missing` / `partial` (R1, R2, R3): files absent, truncated, or not in the manifest. A partition
     listed but unreadable is never skipped (I14). The reason carries `acquire` (command, bytes,
     files, preparation steps, licence, and the `data.acquisition.auto` decision) and its `hint` says
     the same in words: run `vbt data acquire <source>.<table>` (next section), then rerun the check;
     a session re-checks at the next turn. A `.part` file left by an interrupted download fails its
     partition only, and the next `vbt data acquire` resumes it. Files already on disk elsewhere: point
     the source's variable at them (`OPEN_TARGETS_DATA_PATH`, ...).
   - `stale` (`R2:release`): the files are of another release than the descriptor pins: a manifest's
     `release`, or a file's own header (the Cell Ontology's `data-version`). Fetch the pinned release
     next to the old one (`vbt data acquire <source> --env-file <file>` also points the variable at it)
     or move the descriptor to the release you have (DEPLOYMENT.md §7.2).
   - `schema_drift` (R4): a declared column is missing or has an incompatible type, or (strict
     descriptors) an undeclared column appeared. Usually a new release: run
     `vbt ds diff-release --from <old> --to <new>` and update the descriptor.
   - `encoding_drift` (R6): a `verified: false` fact was refuted by the data (a flag's codes, a
     scale). The descriptor's fact is wrong for this release; fix it and rerun `vbt ds check`.
   - `key_violation` (R5, R5b): nulls in a non-nullable key part or duplicate keys. Check whether the
     key is complete (`vbt ds describe <source>.<table>`); declare the missing part or `nullable`.
   - `plugin_unavailable`: a format or layout plugin cannot load in the data child's interpreter (a
     missing optional dependency such as h5py or anndata; `vbt doctor` probes the imports).
4. The readiness cache lives in `<data.cache_dir>/readiness/`; it is keyed by signatures, so a fixed
   file is picked up at the next turn. Deleting the cache directory only costs a recheck.

## too_large

- **Memory estimate over the server's limit** (subkind `over_limit`). `vbt ds estimate --tool
  <server>.<tool>` prints the peak estimate per table in MiB, its tier (`measured`, `sample`, `seed`),
  the server's limit and whether the load is admissible, by admission's own rule (the estimate x
  `data.memory.estimate_safety` plus the server's 300 MB idle baseline): the command and the gateway
  give the same answer. If the tier is `seed`, `vbt ds calibrate --table <source>.<table>` replaces it
  with a sampled measurement. The payload says where the limit came from: `host_mb` (the memory the
  limits are planned from), `limit_source` and, for `auto`, `host_mb_needed` (the smallest host whose
  auto limit admits the load with its baseline). With `limit_source: auto` the fix is a larger plan
  (more host memory, a larger container limit, or `data.memory.host_mb` / `VBT_HOST_MEMORY_MB` where
  the harness has a known share of a shared host; next section), or a number for that server's
  `mem_limit_mb` in `configs/mcp_servers.yaml`. With `configured`, raise the number. Otherwise narrow
  the call (a filter on the partition column, a smaller limit) or use the alternative the payload names.
- **`host_busy`**: the load fits the server's limit, but the upstream servers together would pass the
  host budget (`data.memory.host_budget_mb`) and no idle server is left to recycle. `vbt ds status
  <run>` shows each server's resident memory against the budget; the data child is never counted in
  it. Retry when a server is idle, or give the budget more memory.
- **`scan_budget`**: a derived answer or a native `find` would decode more than its scan budget (the
  table's `max_scan_bytes`, else `data.witness.max_scan_bytes`, which `auto` scales with the data
  child). The payload names the alternative (usually `mcp__data__find` with a narrower `where`);
  `vbt ds explain <server>.<tool>` shows what the binding scans.
- **`unranked_truncation`**: the tool cuts before it ranks and the witness cannot verify the order
  within `data.witness.max_inflate_rows` / `max_inflate_bytes`. `vbt ds explain <server>.<tool>` shows
  the binding's `order` and `limit_mode`. The agent should narrow the query or use the native
  `mcp__data__find`, which ranks globally. Raising the inflation caps is the operator's option when
  the table is small enough.
- **`expansion`**: a descendant expansion (`include_descendants`, hierarchy `members`) exceeded
  `data.resolution.max_expand` (default 5000 terms). `vbt ds describe <source>.<table>` shows the
  hierarchy and its predicates; ask for a narrower term or raise `max_expand`.
- **learned refusal**: a previous identical call was killed for memory; see the next section.

## oom and server_crashed

`vbt ds status <run>` (or `--log-dir <dir>` for a live session) reads the reaper status files: RSS,
limit, containment (`cgroup_v1`, `cgroup_v2`, `watchdog`, `rlimit_data`), OOM kills, recycles and the
measured feedback the estimator learned. Its first line gives the memory the limits are planned from and
where that number comes from (`data.memory.host_mb`, `VBT_HOST_MEMORY_MB`, else MemTotal and the cgroup
limit), the upstream budget, the upstream servers' resident sum it caps, and the data child's own RSS (not
in the budget). An `oom` is never retried; the same call is refused with `too_large` afterwards. Check the
server's log under `<run>/logs/mcp/`; the exit marker's `cause` says what ended the child (`cgroup_oom_kill`,
`watchdog`, `memory_error`, `kernel_oom_kill`, `possible_kernel_oom`). The shipped containment is `rss`
(`data.memory.limit_kind`): a memory cgroup when one can be created, else the RSS watchdog, never
`RLIMIT_DATA` (under which TileDB's Census reads crash with `std::bad_alloc`; it applies only to a server or
host that asks for `rlimit_data`). A server killed under an `auto` limit needs a larger plan (next section)
or a number for its `mem_limit_mb`; a load the estimate under-sized needs `vbt ds calibrate` of the tables it
loads. `max_oom_kills` per session stops a server that keeps dying.

## Host-scaled limits

Every memory budget ships as `auto` and is computed from the **plan**: the smaller of MemTotal and the memory
cgroup limit of the harness and its ancestors (a container's `--memory`), or `data.memory.host_mb` (a number
or a size such as `"480 GB"`), or `$VBT_HOST_MEMORY_MB`. One rule (`src/vbt/datalayer/memory/sizing.py`,
DATA_LAYER.md §14.2) gives:

| Setting | `auto` means |
|---|---|
| `data.memory.host_budget_mb` (all upstream servers) | 0.75 x plan - max(`harness_reserve_mb` 2,048, 5% of plan), at least 1,024 |
| `data.memory.default_server_mb` (one upstream server) | 0.8 x the host budget, at least 2,048 |
| `data.service.mem_limit_mb` (the data child) | 5% of plan within 3,000-32,768; `max_resident_mb` 2/3 of it |
| `data.witness.*` budgets, `data.readiness.vocab_budget_bytes` | the shipped value x data child / 3,000 |
| `data.memory.workspace_mb` (each agent command, notebook, utility test) | (plan - host budget - data child - reserve) / `limits.max_parallel_agents` within 8,000-65,536, at most half the plan |

`vbt validate --only host` prints every configured and effective value and the sum at full load (the host
budget, the data child and `max_parallel_agents` agent commands at their limit) against the plan. Values it
printed for simulated plans (`VBT_HOST_MEMORY_MB`):

| Plan | Host budget | One server | Data child | Agent command | Full load | `host` step |
|---:|---:|---:|---:|---:|---:|---|
| 13,680 MB | 8,212 | 6,569 | 3,000 | 6,840 | 65,932 | WARN (over the plan) |
| 16 GB | 10,240 | 8,192 | 3,000 | 8,000 | 77,240 | WARN |
| 64 GB | 45,875 | 36,700 | 3,276 | 8,000 | 113,151 | WARN |
| 128 GB | 91,750 | 73,400 | 6,553 | 8,000 | 162,303 | WARN |
| 512 GB | 367,002 | 293,601 | 26,214 | 13,107 | 498,072 | PASS |
| 1 TB | 734,003 | 587,202 | 32,768 | 28,672 | 996,147 | PASS |

Below about 512 GB the 8,000 MB floor of an agent command makes eight parallel agents at full load add up to
more than the plan: the `host` step warns, and `vbt setup --plan` says so. Lower
`limits.max_parallel_agents` or `data.memory.workspace_mb` there, or accept that the limits are ceilings, not
reservations. A number anywhere stays as configured, and a server's own `mem_limit_mb` in
`configs/mcp_servers.yaml` wins over `default_server_mb`. A size that does not parse or an unknown
`limit_kind` is an error in `vbt ds lint`, `vbt doctor` and `vbt validate` (an unknown `limit_kind` runs under
`rss` with a warning). `vbt setup` writes the numbers this rule gives into `host.yaml`; its `size` step raises
the server limit, up to the host budget, when the largest server's whole-table loads (x1.3) need more, and
records where its plan came from. On a host the harness shares with the model server, set
`data.memory.host_mb` to the harness's share before `vbt setup`.

## Fetching data and download manifests

[DATA_SETUP.md](DATA_SETUP.md) is the guide; these are the symptoms.

- **What is missing.** `vbt data status [--check]` lists every declared table with its home, whether its
  files are present and verified, its readiness and the tools it unlocks; `--json` for scripts. `vbt data
  acquire --for-agents <agent> --plan` (or `--for-tools`, `--all`, `--missing`) prints what a fetch would
  take (files, bytes, time, licence) and writes nothing.
- **A fetch refused before any transfer**: the bytes left exceed `--max-gb`, or the free disk minus
  `data.acquisition.reserve_bytes`, or the home holds another release (`.vbt-acquisition.json`). Free
  space, raise `--max-gb`, or fetch the new release into its own home (`acquisition.dir` names the release).
- **A file that will not verify** never takes its name: on a size, checksum or Parquet framing mismatch its
  partial download is discarded and fetched again, and after four attempts the
  file is reported failed and its table is left out of the manifest's verified tables. An interrupted
  transfer keeps its `.part`; the next `vbt data acquire` resumes it with a range request, or starts it
  again when the server ignores ranges.
- **`R2:manifest_absent`** (files present, no manifest) or **`R2:manifest`** (a file's size differs from its
  manifest entry): `vbt data acquire <source>` verifies what is present against the release's checksums,
  downloads only what is missing and writes `.download-manifest.json` (the upstream downloader's format;
  each entry records what it was verified against: the publisher's checksum, or only its size where the
  publisher gives none). For Open Targets tables already on disk, `vbt data ot manifest` writes it without
  downloading anything. A manifest must say `complete: true`.
- **Acquired but still `missing`**: the data child reads the directories its descriptors' variables named
  when it started. Point the variable at the home (`--env-file <file>`, or the `host.env` that `vbt setup`
  writes) and start a new session; a between-turns acquisition lists those variables as `env_needed` in its
  `data_acquisition` event.
- **"unknown source" or "quarantined"** from `vbt data acquire`: the descriptor did not load (`vbt ds lint`
  names the file and the error, including an acquisition transport plugin that does not exist).
- **On-demand policy.** `data.acquisition.auto`: `"off"` (default; the refusal says how and an operator runs
  it), `ask` (queued; `vbt data acquire --pending` fetches the queue), `under_budget` (the session fetches
  between turns what fits `budget_bytes`). Every acquisition is logged in
  `<data.acquisition.root>/acquisitions.jsonl`, and during a run in the run's `data_acquisitions.jsonl`.

## Certifying a host: `vbt validate`

`vbt validate` writes `validate.md` and `validate.json` under `<state>/validate/<time>` (`--out` moves
them) and exits 0 only for **PASS**.

- **FAIL**: a step failed. `correctness` lists each case (enforce vs off vs the oracle). A server
  `killed` there was admitted and then died at its limit: give it more memory (previous sections) or
  calibrate the table it loads. `wrong` is an enforce answer that differs from the oracle: report it with
  the case file `correctness/<server>/cases.json`.
- **INCOMPLETE**: nothing failed, but the host is not certified. A `correctness`, `live` or `model` step
  that applies was skipped (no data, no network, no model server), or a step found that the enabled
  agents' tools read tables absent on this host or servers that got no case. The report lists each
  reason. `vbt data acquire --for-agents <agent>` fetches what is absent; `--only`/`--skip` leaves out
  steps on purpose, and those do not count against the verdict.
- `--check-from <check.json>` reuses an earlier deep check so the later steps can be rerun quickly.

## tool_defect: the witness contradicts the upstream answer

The payload names `check` (W1-W6), the witness counts and the defect ids of the binding.

| check | meaning | usual cause |
|---|---|---|
| W1 | upstream returned 0 rows, the witness found rows | nested-type trap, wrong column, wrong identifier form upstream |
| W2 | returned rows outside the witness key set | an argument ignored, the wrong column matched |
| W3 | top-k differs from the witness top-k | head before sort, ranking across groups |
| W4 | echo: the record returned is not the one requested | upstream substituted another entity |
| W5 | one-to-many: a lookup returned one row of several | `iloc[0]` on a non-unique match |
| W6 | phantom rows or short pages | rows for records that do not exist; a page shorter than the limit |

1. `vbt ds explain <server>.<tool>` lists the binding's `defects` (with `file:line` and the detector
   test), its `on_contradiction` (derived repair or refusal) and its witness settings.
2. A contradiction without a declared defect is new upstream behaviour: reproduce it with the
   detector pattern in `tests/datalayer/test_dl_defect_detectors.py` (import the unmodified upstream
   function against a fixture), then add the defect entry, and a `derived` serve when the data child
   can answer the tool.
3. `vbt ds retro-audit <run>` re-classifies a recorded run offline and counts the calls (and claims)
   the gateway would now refuse.

## incomplete_key

Rows would be merged, ranked or limited across a scope dimension with several values (`dimension`,
`values`, `retry_with`), or ranked across incomparable groups (subkind `incomparable_order`). The agent
must fix the dimension (for example `concentration` and `plate` for Tahoe DE, `sourceDatabase` for
interactions) or make one call per value. `vbt ds describe <source>.<table>` shows the key, the scope
columns and their pooling policy (`forbid`, `group`, `pool_ok_for`).

## withheld rows: leakage, phantom records, duplicates

`_vbt.withheld` counts rows the gateway removed: `leakage` (rows dated after
`data.leakage.ceiling`, for no-web scenarios), `phantom` (W6: records that do not exist) and
`duplicates` (the same key twice). A claim citing a call with `leakage.risk: true` is refused.
`vbt ds describe <source>.<table>` shows the table's `leakage` policy and date columns. To change the
ceiling, set `data.leakage.ceiling` in the profile (never per call).

## Drift: data_version_drift, replay_mismatch, source_updated, schema drift

- **`data_version_drift`** (`vbt verify`): a table's current fingerprint differs from the one the run
  pinned or the one a cited call recorded. It is a warning from the cached fingerprints
  (`vbt ds fingerprint --write` refreshes `<data.cache_dir>/fingerprints.json`) and a problem under
  `vbt verify --data`, which reads them fresh. Run `vbt ds replay <run> --all` to see whether the
  cited answers changed.
- **`replay_mismatch`**: `vbt ds replay <run> <tool_use_id>` re-executed the call on current data and
  the canonical row keys, output rows (enrichment as set, k, K, p, q), computed values or statistics
  differ. The `data changed` lines name the tables and partitions read that changed; a partition the
  call never read is listed as `ignored`. Re-run the analysis on the current release and refile the
  claims; `refresh_claims` already marks evidence on changed tables `unresolved`.
- **`source_updated`**: a live record (ClinicalTrials.gov, PubMed) has a newer version than when the
  call cited it. Records with unchanged versions are still compared. Decide whether the claim needs
  the new record; the old answer stays attributable through `as_of`.
- **New release** (schema drift before it bites): `vbt ds diff-release --from 25.09 --to 25.12`
  (labels resolve as siblings of the configured root; directories work too; `--quick` reads footers
  only) reports removed role columns, Arrow type changes, row-group range changes (encodings),
  vocabulary changes and matrix axis membership. Exit 1 means some tool will be `not_ready` or answer
  differently until the descriptor is reviewed.
- **Upstream code drift**: CI fails `test_upstream_input_schemas_match_the_reviewed_snapshot` when an
  upstream tool or parameter changes. Review the overlay, then regenerate the snapshot with
  `vbt ds explain --all --json --schemas > tests/datalayer/upstream_tool_schemas.json`.

## Slow witness

The witness should add well under 300 ms (p95) per call on composite-key tables (soak test). When it
does not:

1. `vbt ds explain <server>.<tool>` shows which table the witness scans and whether the binding has an
   access path.
2. Declare `access_paths` for the filtered column in the descriptor and build its row-group sidecar:
   `vbt ds index build --access-paths --table <source>.<table>`. Huge tables build resumably with
   `--huge --time-budget-s 600`.
3. `data.witness.max_scan_bytes` bounds every scan; a scan over it makes the count `unknown` (and an
   empty result `empty_unverified`) rather than slow.

## Sidecar index rebuild

Resolver sidecars (`<data.cache_dir>/index/`) and row-group value indexes are keyed by fingerprint and
always safe to delete.

- `vbt ds index build` builds every local resolver index; `--id-type <source>:<id_type> --force`
  rebuilds one (after a release update or a changed `resolve_via`).
- `vbt ds index build --access-paths [--table <source>.<table>]` builds the row-group indexes.
- `vbt ds resolve <id_type> <value>` shows how one identifier resolves (rule, candidates,
  suggestions) from the sidecars, without starting anything.
- `vbt ds check` reports an index that is missing or built for an older fingerprint.

## Adding a third-party server

1. Add it to `configs/mcp_servers.yaml`. Until an overlay reviews its tools, the generic guard applies:
   explicit not-found shapes are `not_found`, structural empties `empty_unverified`, HTTP 5xx
   `source_error`.
2. `vbt ds overlay init <server>` lists its tools and writes `configs/data/overlays/<server>.yaml`
   with every tool `status: unreviewed`; the header explains each step.
3. Bind arguments and results, add a descriptor if it reads a dataset the catalog lacks, then
   `vbt ds lint` and `vbt ds explain <server>.<tool>`.
4. Run it in observe mode (`data.gateway.mode: observe`, or leave it out of
   `data.gateway.enforce_servers`) for at least one recorded scenario.

## Observe → enforce graduation

`vbt ds graduate <server> --run <run> [--run <run> ...]` prints the checklist of DATA_LAYER.md §22:
overlay present, lint clean, every tool reviewed, every blocked tool names an alternative, and
retro-audit evidence from recorded runs (calls observed, calls the gateway would now refuse or
qualify, and how many of them claims cited). Review the changed calls (`vbt ds retro-audit <run>`)
for false positives, then add the server to `data.gateway.enforce_servers` in
`configs/default.yaml`. Kill switches, in increasing scope: `serve: pass` on one tool, removing the
server from `enforce_servers`, `data.gateway.mode: observe`, `data.enabled: false`.
