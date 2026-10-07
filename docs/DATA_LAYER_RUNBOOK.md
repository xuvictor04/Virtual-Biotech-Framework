# Data layer runbook: symptom → command

Operator guide for the vbt data layer (design: [DATA_LAYER.md](DATA_LAYER.md)). Each entry starts
from what you see in an agent's tool result, in `vbt verify`, or in `vbt doctor`, and names the
command that explains it and the fix. `vbt ds` is short for `vbt datasource`; every command takes
`--help`. Commands that read data (`check`, `fingerprint`, `index build`, `estimate`, `calibrate`,
`diff-release`) run the data child's interpreter (`${vars.mcp_python}`), so they need the same
environment as the MCP servers (`OPEN_TARGETS_DATA_PATH`, `TAHOE_DATA_PATH`, ...).

The error kinds and payloads are those of DATA_LAYER.md §12.1. A model-side kind (`not_found`,
`ambiguous`, `invalid_argument`, `unsupported_combination`, `incomplete_key`, `unsupported_filter`,
`insufficient_resolution`) is the agent's input to fix; a data-side kind (`not_ready`, `too_large`,
`quarantined`, `tool_defect`, `oom`, `server_crashed`, `source_error`, `service_unavailable`) is
yours.

## Quick reference

| Symptom | First command |
|---|---|
| `not_ready` naming a table, column or partition | `vbt ds check --tool <server>.<tool>` |
| `too_large` (memory estimate) | `vbt ds estimate --tool <server>.<tool>` |
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

## not_ready: a table, column or partition failed readiness

The payload is `tables[{name, column?, partition?, check, detail, hint}]`. Readiness is scoped: only
the tools that read the failing part are refused; every other tool keeps working.

1. `vbt ds check --tool <server>.<tool>` shows the call-scoped readiness: the tables, columns,
   containers and partitions the tool reads, and the check (R1-R10) each failed.
2. `vbt ds check --table <source>.<table> --depth standard -V` gives the per-column and per-partition
   statuses; `--column <source>.<table>.<column>` narrows to one column.
3. By status:
   - `missing` / `partial` (R1, R2, R3): files absent, truncated, or not in the manifest. A partition
     listed but unreadable is never skipped (I14). Re-download the affected partition, then rerun the
     check. A `.part` file left by an interrupted download fails its partition only.
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

- **Memory estimate over budget** (no subkind). `vbt ds estimate --tool <server>.<tool>` prints the
  peak estimate per table, its tier (`measured`, `sample`, `seed`) and the server's limit. If the
  estimate tier is `seed`, `vbt ds calibrate --table <source>.<table>` replaces it with a sampled
  measurement. Otherwise narrow the call (a filter on the partition column, a smaller limit) or raise
  `mem_limit_mb` for that server in `configs/mcp_servers.yaml`.
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
limit, containment (`rlimit_data`, `cgroup_v2`, `watchdog`), OOM kills, recycles and the measured
feedback the estimator learned. An `oom` is never retried; the same call is refused with `too_large`
afterwards. Check the server's log under `<run>/logs/mcp/`. Raise the server's `mem_limit_mb`, enable
`data.memory.limit_kind: cgroup` where the host delegates cgroups, or calibrate the tables it loads
(`vbt ds calibrate`). `max_oom_kills` per session stops a server that keeps dying.

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
