# Data layer: implementation status

Status of the 24 fixes of [DATA_LAYER.md §18](DATA_LAYER.md#18-the-24-fixes-in-five-phases-140-engineer-weeks)
after phases 1-5 and the phases 2-5 review pass. Each fix is **implemented**, **partial** (the core is in
place, with named gaps) or **deferred**. Paths are relative to `src/vbt/datalayer/` unless noted.
The last part of this page explains how to use the layer.

No real Open Targets, Tahoe or Census release is on the development machine. Everything here is tested
on generated fixtures (`tests/datalayer/dl_fixtures.py`) and on stubs of the live APIs. The one
exception is the Zenodo case-study-1 subset: when `data/zenodo` is extracted, it is checked for real.

## The 24 fixes

| Fix | Status | Where | Notes |
|---|---|---|---|
| F1 Typed outcome channel | implemented | `errors.py`, `result.py`, `record.py`, `src/vbt/runtime.py` (`_record_data_result`) | `_vbt` header first; `vbt.dataprov/1` records under `data.provenance.dir`, which is always relative to the run and pinned in `MANIFEST.config.data.provenance_dir`. |
| F2 Gateway seam | implemented | `gateway/gateway.py`, `src/vbt/tools/mcp_bridge.py` | off/observe/enforce. Observe mode adds no latency and no side effects: no index build or vocabulary fetch in the call path, no write-once rename, and the plain crash policy. A broken catalog under `when_service_down: strict` refuses the servers the gateway would enforce (`Runtime.gateway_refused`). |
| F3 Data child and bounded reader | implemented | `service/reader.py`, `service/server.py`, `service/verbs/` | Key-check spill files are removed on every exit and swept when a child starts (`checks.sweep_spills`). |
| F4 Identifier plugins and resolver | implemented | `plugins/identifiers/`, `resolve/resolver.py` | `CHEMBL:25` normalizes to `CHEMBL25`. An rsID is mapped to the variant it names through the declared `maps_to ... via: variant` (`remote_map` edge, the data child's `_resolve_remote` with `rsid>ot_variant`); several alleles are ambiguous. |
| F5 Phase-1 descriptors and overlays | implemented | `configs/data/sources/`, `configs/data/overlays/` | `vbt ds lint` is clean. |
| F6 Not-found contract and witness | implemented | `gateway/classify.py`, `service/verbs/witness.py` | A plain-text success that mentions an HTTP status is not a failure. Count results skip the row-based W1. |
| F7 Argument contracts and transforms | implemented | `gateway/contracts.py`, `gateway/transforms.py` | Chromosomes bound to a bare-style position column are normalized (`roles.bare_chromosome`: `chr19` becomes `19`; `chrM` is invalid). A region `chrom:start-end` is a chromosome plus a position range. A null on a defaulted filter is refused before the call unless null means "no restriction" (a derived tool, or an upstream signature that takes None). Disclosed defaults reach the data child (`DataGateway._params`). A pair whose every evidence item is negated is withheld and counted (T6). |
| F8 Memory and process safety | implemented | `launch/reaper.py`, `launch/__init__.py`, `memory/` | The Bash workspace limit honours `data.memory.limit_kind`. `vbt ds` and `vbt verify --data` run data-child code under the reaper (`cli._run_reaped`). |
| F9 Tool-scoped readiness | implemented | `service/checks.py`, `gateway/readiness.py`, `src/vbt/preflight.py` | Concurrent refreshes share one `_check`. A malformed catalog is a required preflight failure. |
| F10 Claims, provenance, correctness tests | implemented | `tests/datalayer/test_dl_correctness_six.py`, `src/vbt/audit/claims.py` | New CT-1/2/6 cases: `CHEMBL:25`, `chr19`, the null `tep`, negated-only phenotype pairs, and grains. CT-3 compares exact item keys and full row keys. |
| F11 Strict descriptors | implemented | `configs/data/sources/*.yaml`, `service/checks.py` (`r4_types`) | `column_patterns` count as declared under `strict`, checked against the pattern's role. The Zenodo descriptor matches the real archive: `chembl_clinical_nct` is keyed `(nct_id, drugId, targetId, diseaseId)`. |
| F12 Format and layout plugins v2 | implemented | `plugins/formats/`, `plugins/layouts/` | |
| F13 Derivation engine | implemented | `derive/`, `tests/datalayer/golden/` | A record read from a nested column takes that column's coverage in both the text and the header. |
| F14 Native tools and the client | implemented | `service/verbs/public.py`, `derive/tools.py`, `client.py` | Derived view sections report their own status, total, table and coverage under `_vbt.sections`. `vbt.analysis.survival` reads through the client (`client.open_file`, live `find` on the cBioPortal clinical tables). |
| F15 Statistic plugins v2 | implemented | `plugins/statistics/` | |
| F16 Native enrichment and aggregation | implemented | `service/verbs/enrich.py` | `mcp__data__enrich` is public. A gene with no screens is empty, never `found: true`. |
| F17 Composite-key completion | implemented | `service/verbs/network.py`, `service/verbs/pairs.py` | The listed `neighbors` schema offers `nodes`, `hops` 1-4, `max_nodes` and `score_order`. |
| F18 Ontology semantics | implemented | `service/verbs/hierarchy.py` | `mcp__data__expand` is public. `drug.search_known_drugs` takes `include_descendants`, served derived through `known_drug.ancestors` (overlay facet `derived_when`). |
| F19 Memory at scale | implemented | `memory/host.py`, `memory/calibrate.py`, `launch/reaper.py`, `service/verbs/index_huge.py` | The host budget with LRU idle recycle is on by default (`data.memory.host_budget_mb: auto`). The stdout relay is configured by `data.memory.relay_max_message_mb`. `build: readiness` sidecar indexes are built by the check (`checks.access_indexes`); `on_demand` ones are built on first read, and huge ones offline (`vbt ds index build --access-paths --huge`). |
| F20 Live and remote sources | partial | `plugins/layouts/live_api.py`, `plugins/layouts/soma.py`, `service/verbs/witness.py`, `service/verbs/census_count.py` | Done: the remote witness runs for every table whose layout declares `count` (including the shipped `access: upstream` CT.gov and cBioPortal tools); `mcp__data__find`/`lookup` on live tables go to `_live_find` (cBioPortal patient pivot, record versions; unpaged reads are admitted count-first); Census count-first admission for `get_anndata`, `get_expression_for_genes` and `get_anndata_donor_balanced` (overlay facet `count_first`; the resolved release is disclosed); `genes_found`/`genes_not_found` recomputed from `var.feature_name`. Gaps: `get_anndata_donor_balanced` is still served upstream behind `requires_fixed: dataset_id`, not as the derived `(dataset_id, donor_id)` sample (`census_count.donor_balanced` exists, but no tool serves it); `clinicaltrials.get_clinical_data` stays `serve: pass` (its text points to `mcp__data__find` on `cbioportal.patient_clinical`). The survival script's expression download still uses the REST API (no table declares the molecular profile). |
| F21 Third-party servers, envelope kind | implemented | `plugins/envelopes/`, `cli.py` (`overlay init`) | |
| F22 Replay, drift, row-level citations | implemented | `replay.py`, `src/vbt/verify.py` | Under `--data`, a pinned or cited table that is gone, a failed fingerprint read and a lost provenance record are problems (INCOMPLETE). Replay `auto` uses the guarded bridge; `inprocess` is opt-in. |
| F23 Calibration, soak, graduation | partial | `memory/calibrate.py`, `tests/datalayer/test_dl_soak.py`, `cli.py` (`graduate`) | Tooling and fixture-scale soak are in place. Calibration on real 25.09 subsets, the witness p95 target and fidelity validation on the paper scenarios need the real releases. |
| F24 Coverage and drift guards | implemented | `tests/datalayer/test_dl_coverage_guards.py`, `diff_release.py`, `docs/DATA_LAYER_RUNBOOK.md` | |

## Review findings closed in this pass

| Finding | Status | Regression test |
|---|---|---|
| F20 remote witness not dispatched | fixed | `test_dl_live_sources.py::test_shipped_count_tool_gets_the_remote_witness` (the monkeypatch is gone) |
| F20 Census count-first unwired | partial (see F20 above) | `test_dl_live_sources.py::test_census_pulls_are_admitted_count_first`, `::test_genes_found_are_recomputed_from_feature_name` |
| F20 live find unreachable | fixed | `test_dl_live_sources.py::test_public_find_and_lookup_serve_live_tables` |
| F17F18 native verbs not upgraded | fixed | `test_dl_native_tools.py::test_listed_schemas_offer_the_phase3_verbs`, `::test_expand_is_a_public_verb` |
| F19 host budget never enabled | fixed | `test_dl_memory_v2.py::test_gateway_enables_the_host_budget` |
| F18 include_descendants on known_drug | fixed | `test_dl_ontology.py::test_include_descendants_on_search_known_drugs` |
| F19 access-path build ignored | fixed (`readiness`); `on_demand` unchanged, as specified | `test_dl_service_check.py::test_readiness_builds_declared_access_indexes` |
| F19 relay unconfigurable | fixed | `test_dl_launcher.py::test_build_launch_spec` |
| S10 survival not on the client | fixed (expression download excepted) | `test_dl_client.py::test_survival_reads_go_through_the_client` |
| CT6-V1, SW-3 negated-only pairs | fixed | CT-6 `pheno_x`/`pheno_w`/`pheno_z`, `test_ct6_negated_pair_is_not_presence` |
| CT2-V1, SW-5 `tep` coverage | fixed | CT-2 `tep_tp53`, `test_ct2_tep_absence_claim_is_rejected` |
| CT6-V2 grains after rename | fixed | `_ct6_phenotype` grain assertions |
| GW-NULL | fixed | `test_dl_review_live_p25.py::test_null_on_a_defaulted_filter_is_refused_before_the_call` |
| CT1-V1, SW-7 `CHEMBL:25`, `chr19` | fixed | CT-1 `drug_curie`/`chrom_prefixed`, `test_dl_review_live_p25.py` |
| CT-TEST-1 weak CT assertions | fixed | `test_dl_correctness_six.py` (`row_keys`, canonical id_type, `served_full` on every CT-3 case) |
| SW-1 `min_cell_lines` default | fixed | `test_dl_review_live_p25.py::test_min_cell_lines_default_is_applied` |
| SW-2 unscreened gene found | fixed | `::test_unscreened_gene_is_never_found` |
| SW-4 section status | fixed | `::test_view_sections_carry_their_own_status_and_coverage` |
| SW-6 rsID | fixed | `test_dl_resolver.py::test_rsid_maps_to_a_variant_through_the_data_child`, `::test_rsids_resolve_to_the_variant_they_name` |
| SW-8 region | fixed | `::test_gwas_region_queries_are_served` |
| R2 spill leak | fixed | `test_dl_service_check.py::test_key_check_spill_is_removed_on_every_exit` |
| R3 `column_patterns`, Zenodo | fixed | `::test_column_patterns_count_as_declared_under_strict`, `::test_shipped_zenodo_descriptor_is_ready_on_the_real_archive` |
| R4 plain-text successes | fixed | `test_dl_gateway_classify.py::test_generic_guard` |
| R5 index failures cached | fixed | `test_dl_gateway_robustness.py::test_index_build_outage_is_retried_and_rejection_is_not` |
| R6 observe latency | fixed | `::test_observe_mode_builds_no_index_and_fetches_no_vocab_in_the_call`, `::test_observe_mode_keeps_the_plain_crash_policy` |
| R7 single-flight readiness | fixed | `::test_concurrent_readiness_refreshes_share_one_check` |
| R8 broken catalog | fixed (refusal under strict; no per-file quarantine) | `test_dl_runtime_plumbing.py::test_a_broken_overlay_refuses_enforced_servers_under_strict` and two more |
| R9 hygiene | fixed | `test_dl_cli.py::test_status_and_explain_text_forms` |
| INV-1 observe renames files | fixed | `test_dl_gateway_robustness.py::test_observe_mode_never_renames_an_existing_output` |
| INV-2 verify `--data` passes on missing data | fixed | `test_dl_verify_data.py::test_data_mode_reports_missing_tables_unreadable_fingerprints_and_lost_records` |
| INV-3 custom provenance dir | fixed | `test_dl_replay.py::test_a_custom_provenance_dir_is_found_by_replay_and_derived_from` |
| INV-4 in-process decoding | fixed | `test_dl_verify_data.py::test_fresh_fingerprints_never_decode_data_in_the_harness`, `test_dl_launcher.py::test_cli_data_child_commands_run_under_the_reaper` |
| INV-6 workspace `limit_kind` | fixed | `test_dl_client.py::test_workspace_limit_follows_limit_kind` |

## How to use the layer

### Gateway modes

`data.gateway.mode` in `configs/default.yaml`:

* `enforce` (default): servers listed in `enforce_servers` are guarded. Identifiers are resolved,
  arguments checked, results witnessed and transformed, and each result carries a `_vbt` header and a
  provenance record. Every other server runs in observe mode.
* `observe`: every decision is traced as a `data_observe` event, and the upstream answer is returned
  unchanged. Observe mode never builds an index or fetches a vocabulary in the call path (they are
  built in the background), never renames a file, and keeps the plain one-retry crash policy. The
  reaper's memory limit still applies.
* `off` (quote it in YAML): no gateway.

`when_service_down: strict` (default) refuses guarded calls when the data child is down. It also
refuses guarded servers when a descriptor or overlay cannot be loaded. `lenient` lets those calls run
unguarded.

### Authoring a descriptor or overlay

1. Describe the files in `configs/data/sources/<source>.yaml` (§6). Give each table its grain, key,
   coverage and one role per column (§7). `column_patterns` declares families of columns such as
   `ae_serious_{organ}_pct`. `access_paths` with `via: sidecar_index` and `build: readiness` are built by
   the readiness check; `on_demand` ones are built on first read.
2. Bind each tool in `configs/data/overlays/<server>.yaml` (§8). Set `reads`, `args` (`binds`,
   `accepts`, `op`, roles) and `result`. Set `serve` to `pass`, `derived` or `block`. Phase-4/5 facets:
   * `derived_when: [arg]` serves a `pass` tool derived when a gateway-only argument is set.
   * `count_first: {table, filter_arg, genes_arg, max_cells_arg, recompute_genes}` admits remote pulls
     count-first.
   * A record read from a nested column (`rows: $.tep`) takes that column's coverage. Declare
     `coverage`/`null_means` on the column.
3. `vbt ds lint` (add `--strict` for strict sources), then `vbt ds check --table S.T` on the data.
   Use `vbt ds explain server.tool` to see the derived schema and text, and regenerate the golden
   snapshots with `VBT_UPDATE_GOLDEN=1 pytest tests/datalayer/test_dl_derive_all_tools.py`.
4. For a third-party server, start from `vbt ds overlay init <server>`.

### `vbt ds` commands

| Command | What it does |
|---|---|
| `vbt ds list`, `describe S[.T]` | sources, tables, roles, coverage |
| `vbt ds lint [--strict]` | validate descriptors and overlays |
| `vbt ds check [--depth D] [--table S.T] [--tool s.t]` | readiness R1-R10 (plus `build: readiness` sidecars), run in the data child under the reaper |
| `vbt ds resolve <id_type> <value...>` | resolver rules and candidates |
| `vbt ds explain s.t \| --all [--json]` | binding, serve mode, derived schema (from the upstream source, offline) and text |
| `vbt ds fingerprint [--write]` | table fingerprints (what runs pin) |
| `vbt ds index build [--id-type T] [--access-paths [--huge]]` | resolver and access-path sidecars |
| `vbt ds estimate`, `calibrate`, `status` | memory estimates, calibration, per-server memory and the host budget |
| `vbt ds replay <run> <tool_use_id...> \| --all [--backend bridge\|inprocess]` | re-execute recorded calls (the default `auto` is the guarded bridge) |
| `vbt ds diff-release`, `graduate`, `retro-audit` | release drift, the observe-to-enforce checklist, offline re-classification |
| `vbt verify <run> --data` | fresh fingerprints and replays; missing tables, unreadable fingerprints and lost records leave the run INCOMPLETE |

### Native tools for agents

`mcp__data__{resolve, describe, lookup, find, search, vocab, members, aggregate, similar, neighbors,
expand, enrich}` are granted to every data agent (`configs/agents.yaml`). `find` and `lookup` on a
live table (CT.gov, cBioPortal) are answered within the source's request budget. A result cut by the
source's page budget is `partial`.

### Memory settings

`data.memory.*` keys:

* `limit_kind`: `rlimit_data`, `cgroup`, `watchdog` or `none`. It applies to MCP servers and to the
  Bash workspace.
* `host_budget_mb`: `auto`, a number, or `off`. The host budget with LRU idle recycle.
* `relay_max_message_mb`: 0 means off. A server message larger than this is replaced by an error with
  the same id.
* `workspace_mb`: the memory limit of agent Bash commands.
