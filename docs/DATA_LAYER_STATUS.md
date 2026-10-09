# Data layer: implementation status

Status of the 24 fixes of [DATA_LAYER.md §18](DATA_LAYER.md#18-the-24-fixes-in-five-phases-140-engineer-weeks)
after phases 1-5, the phases 2-5 review pass and the real-data pass of 2026-10-08. Each fix is
**implemented**, **partial** (the core is in place, with named gaps) or **deferred**. Paths are relative to
`src/vbt/datalayer/` unless noted.
The last part of this page explains how to use the layer.

The default suite runs offline on generated fixtures (`tests/datalayer/dl_fixtures.py`), on stubs of the
live APIs and on recorded responses. On 2026-10-08 the layer was also run against the real releases and the
live APIs: Open Targets 25.09, Tahoe-100M, DepMap 24Q4, GO, the Cell Ontology, MSigDB, the Zenodo
case-study-1 subset, ClinicalTrials.gov, cBioPortal, E-utilities and the Census.
[DATA_LAYER_REAL_DATA.md](DATA_LAYER_REAL_DATA.md) records those runs:

* which releases were checked;
* which descriptor facts held and which were corrected;
* the six correctness tests on the real Open Targets release, with the gateway off and on;
* the measured memory and latency;
* what still needs data that could not be obtained.

The real-data checks are opt-in: `VBT_DL_REAL_DATA=<dir>` for downloaded data, `VBT_DL_NETWORK=1` for the
network. Small snapshots of what they measured (footers, value facts, recorded exchanges, each with its
source URL and retrieval date) are under `tests/datalayer/real/`.

## The 24 fixes

| Fix | Status | Where | Notes |
|---|---|---|---|
| F1 Typed outcome channel | implemented | `errors.py`, `result.py`, `record.py`, `src/vbt/runtime.py` (`_record_data_result`) | `_vbt` header first; `vbt.dataprov/1` records under `data.provenance.dir`, which is always relative to the run and pinned in `MANIFEST.config.data.provenance_dir`. |
| F2 Gateway seam | implemented | `gateway/gateway.py`, `src/vbt/tools/mcp_bridge.py` | off/observe/enforce. Observe mode adds no latency and no side effects: no index build or vocabulary fetch in the call path, no write-once rename, and the plain crash policy. A descriptor or overlay file that does not load is quarantined on its own. Only the servers whose own overlay is quarantined, and the tools that depend on the file, are refused (`Runtime.gateway_refused`, typed `quarantined` errors). |
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
| F20 Live and remote sources | implemented | `plugins/layouts/live_api.py`, `plugins/layouts/soma.py`, `service/verbs/witness.py`, `service/verbs/census_count.py`, `service/verbs/serve.py` | Done: the remote witness runs for every table whose layout declares `count` (including the shipped `access: upstream` CT.gov and cBioPortal tools). Free text the source's engine matches is sent to the request parameter it fills (`engine_param`: CT.gov `query.cond`/`query.term`/`query.intr`/`filter.advanced`, PubMed `term`), so realistic CT.gov and PubMed calls get an independent count of the same search (several phases compile into one `AREA[Phase](... OR ...)` fragment); a remote count never inflates the page. `mcp__data__find`/`lookup` on live tables go to `_live_find` (cBioPortal patient pivot, record versions; unpaged reads are admitted count-first), and so do `_serve` lookups and finds, so a `serve: derived` binding can read a live table. Census count-first admission for `get_anndata`, `get_expression_for_genes` and `get_anndata_donor_balanced` (overlay facet `count_first`): every gene argument counts (`genes_arg` lists `gene_symbols` and `ensembl_ids`; a pull naming no gene reads every gene), and the estimate (`count_first.estimate`: a base plus a per-cell slope) is calibrated on real pulls, so the 200,000-cell spleen pull that was admitted and killed at 4,500 MB is now `too_large` before upstream is called. `get_anndata_donor_balanced` is served as the derived `(dataset_id, donor_id)` sample (`count_first.sample`): the data child draws upstream's own cell-type-stratified sample (same generator and order) with donors keyed by dataset, the unmodified server is asked for exactly those cells (`soma_joinid in [...]`), the payload's total is the counted one, `derived_sample` describes the draw and the written file must hold the drawn cells (`sample_cells`); without a sample the filter must fix one `dataset_id`. `clinicaltrials.get_clinical_data` is `serve: derived` from the live cBioPortal tables (`derived.compose`: samples, sample attributes, patient attributes; the attribute ids from a live section), sized first like the upstream pull (`size_from`). A derived handler's typed refusal crosses MCPBridge as `ServeResponse.error`, on the wire `refusal` (a top-level `error` is a failed call to the bridge); release lookups have their own verb (`_release`). Results of live sources name the release the call observed (CT.gov `dataTimestamp`; the Census release `stable` names, recorded for every `single_cell` call and written into `get_census_info`'s body through `result.release_alias`; a cBioPortal study's `importDate` through `release.per`, also recorded as that study's version) and the API and software versions (`source.versions`: CT.gov `apiVersion`, cBioPortal `portalVersion`/`dbVersion`, the PubMed build). Under the evidence ceiling an upstream CT.gov count also gets the count of records last updated by the ceiling (`_vbt.ceiling_totals`). `eligibility_text` binds the criteria column, so its phrases reach the remote witness as `AREA[EligibilityCriteria]"..."`. Native finds on live tables apply each source's evidence ceiling (CT.gov `data.leakage.ceiling` in the request and the count, PubMed `VBT_LITERATURE_MAXDATE`, which the remote witness also counts under), the Census `is_primary_data` default filter (disclosed; `where` overrides it), exact strict bounds, `search` for engine-matched text (equality refused), dotted `columns`, and report keys the source does not hold as `not_found_items` (a live `lookup` of one is `not_found`); `count_cells` is witnessed from the gateway-parsed SOMA filter. `genes_found`/`genes_not_found` are recomputed from `var.feature_name`. `vbt.analysis.survival` reads the expression (`cbioportal.molecular_data`, one gene per request) and the clinical tables through the client. It uses the REST API only when the client cannot answer, and names what it read that way in `attrs["vbt_prov_fallback"]`. Limits: a Census pull is admitted on an estimate fitted to spleen pulls of 1,842-149,759 cells with 2 genes and one 7,750-cell pull with every gene; other filters and gene counts are not measured. All of this was run against the live APIs on 2026-10-08 ([DATA_LAYER_REAL_DATA.md](DATA_LAYER_REAL_DATA.md) §2.6, §4.2); the CD276 Cox results in LUAD equal the archived ones. |
| F21 Third-party servers, envelope kind | implemented | `plugins/envelopes/`, `cli.py` (`overlay init`) | |
| F22 Replay, drift, row-level citations | implemented | `replay.py`, `src/vbt/verify.py` | Under `--data`, a pinned or cited table that is gone, a failed fingerprint read and a lost provenance record are problems (INCOMPLETE). Replay `auto` uses the guarded bridge; `inprocess` is opt-in. |
| F23 Calibration, soak, graduation | partial | `memory/calibrate.py`, `memory/estimate.py`, `configs/default.yaml` (`data.memory`), `tests/datalayer/test_dl_soak.py`, `cli.py` (`graduate`) | Tooling and fixture-scale soak are in place. **Calibrated on the real 25.09 tables:** the shipped memory factors are fitted to whole-table pandas loads of 26 tables (6 MB to 3.0 GB). On the 18 tables over 50 MB, estimate/measured is 0.70-1.46 (median 1.00), and 16 of the 18 are within ±30%; study (1.46) and literature_vector (1.32) are overestimated. The seed factors had been 1.44-9.39 times too high. The sample tier (`vbt ds calibrate`) counts the pandas frame and the Arrow table the loader holds together: 0.54-1.40, median 1.06 (pandas alone: 0.34-0.85). **Witness latency:** p95 82.3 ms on warm target-server calls (p50 58.1 ms). **Gaps.** The five tables whose load exceeds 5.4 GB (interaction_evidence, expression, target_essentiality, interaction, l2g_prediction) could not be measured. The witness p95 was not measured across servers after first calls stopped waiting for `_stats`, and a single interaction witness takes 1.5-9.2 s. `profile: fidelity` on the paper scenarios and retro-audit evidence for graduation need model runs of those scenarios. Numbers: [DATA_LAYER_REAL_DATA.md §5](DATA_LAYER_REAL_DATA.md#5-memory-and-latency). |
| F24 Coverage and drift guards | implemented | `tests/datalayer/test_dl_coverage_guards.py`, `diff_release.py`, `docs/DATA_LAYER_RUNBOOK.md` | |

## Review findings closed in this pass

| Finding | Status | Regression test |
|---|---|---|
| F20 remote witness not dispatched | fixed | `test_dl_live_sources.py::test_shipped_count_tool_gets_the_remote_witness` (the monkeypatch is gone) |
| F20 Census count-first unwired | fixed | `test_dl_live_sources.py::test_census_pulls_are_admitted_count_first`, `::test_genes_found_are_recomputed_from_feature_name` |
| F20 live find unreachable | fixed | `test_dl_live_sources.py::test_public_find_and_lookup_serve_live_tables` |
| F17F18 native verbs not upgraded | fixed | `test_dl_native_tools.py::test_listed_schemas_offer_the_phase3_verbs`, `::test_expand_is_a_public_verb` |
| F19 host budget never enabled | fixed | `test_dl_memory_v2.py::test_gateway_enables_the_host_budget` |
| F18 include_descendants on known_drug | fixed | `test_dl_ontology.py::test_include_descendants_on_search_known_drugs` |
| F19 access-path build ignored | fixed (`readiness`); `on_demand` unchanged, as specified | `test_dl_service_check.py::test_readiness_builds_declared_access_indexes` |
| F19 relay unconfigurable | fixed | `test_dl_launcher.py::test_build_launch_spec` |
| S10 survival not on the client | fixed (the expression download too, through `cbioportal.molecular_data`) | `test_dl_client.py::test_survival_reads_go_through_the_client`, `test_dl_f20_gaps.py::test_survival_expression_download_goes_through_the_client` |
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
| R8 broken catalog | fixed (per-file quarantine) | `test_dl_hardening.py::test_a_broken_descriptor_quarantines_only_the_tools_that_read_it`, `::test_the_runtime_starts_the_rest_and_refuses_only_the_quarantined_server`, `test_dl_runtime_plumbing.py::test_a_broken_overlay_refuses_enforced_servers_under_strict` |
| Refused servers started lazily | fixed (a refused server is never handed to the bridge) | `test_dl_hardening.py::test_a_refused_server_is_never_started_lazily`, `::test_a_server_the_bridge_refuses_is_never_started` |
| R9 hygiene | fixed | `test_dl_cli.py::test_status_and_explain_text_forms` |
| INV-1 observe renames files | fixed | `test_dl_gateway_robustness.py::test_observe_mode_never_renames_an_existing_output` |
| INV-1 memory exits without a cause | fixed (the reaper tees the child's stderr; `cause`: `watchdog`, `cgroup_oom_kill`, `memory_error`, `kernel_oom_kill`, `peak_rss`) | `test_dl_hardening.py::test_exit_causes`, `::test_an_rlimit_memory_error_is_a_memory_exit_named_from_the_teed_stderr`, `::test_the_crash_decision_names_the_cause` |
| Observe-mode side effects (SOMA vocabulary from the watched server, existence and anchor counts, remote resolution) | fixed | `test_dl_hardening.py::test_observe_mode_has_no_side_effects_end_to_end`, `::test_observe_mode_asks_the_watched_server_for_no_soma_vocabulary`, `::test_observe_mode_counts_nothing_and_asks_no_remote_question` |
| Explicit null on a defaulted filter dropped before upstream | fixed (sent as null when the upstream signature takes None) | `test_dl_hardening.py::test_an_explicit_null_reaches_an_upstream_that_takes_none` |
| INV-2 verify `--data` passes on missing data | fixed | `test_dl_verify_data.py::test_data_mode_reports_missing_tables_unreadable_fingerprints_and_lost_records` |
| INV-3 custom provenance dir | fixed | `test_dl_replay.py::test_a_custom_provenance_dir_is_found_by_replay_and_derived_from` |
| INV-4 in-process decoding | fixed | `test_dl_verify_data.py::test_fresh_fingerprints_never_decode_data_in_the_harness`, `test_dl_launcher.py::test_cli_data_child_commands_run_under_the_reaper` |
| INV-6 workspace `limit_kind` | fixed | `test_dl_client.py::test_workspace_limit_follows_limit_kind` |

## Real-data findings closed

### Descriptors and checks against the releases and APIs

Found by checking the descriptors against the real releases and APIs (snapshots in `tests/datalayer/real/`).

| Finding | Fix | Regression test |
|---|---|---|
| Grouping keys that repeat by design (pharmacogenomics, drug mechanism of action) were `key_violation` | `key.row_identity: content_hash`: R5b tests that no two rows are equal | `test_dl_real_data_checks.py::test_content_identity_tests_equal_rows_not_the_grouping_key` |
| R4b ignored `universe.where` and sampled one id prefix | the universe sample applies `where` (rows and list items) and keeps every prefix | `::test_universe_where_filters_rows_and_list_items`, `::test_universe_sample_keeps_every_prefix` |
| An item key could not declare a nullable part (GO `ecoId`, chemical probe `drugId`, safety `eventId`/`url`, DepMap `tissueId`) | `items_of.key.nullable`; item keys of a list in a list are unique per parent item | `::test_nullable_item_key_parts`, `::test_item_keys_of_a_list_in_a_list_are_unique_per_parent_item` |
| `target.go` item key repeated in 12,806 genes | the six-part key with nullable `ecoId` | `test_dl_real_ot_schema.py::test_item_tables_with_null_item_key_parts_are_checked` |
| `so.id` is `SO:NNNNNNN` in 25.09 | canonical `SO:`; the `SO_` spelling of variant consequences resolves to it | `test_dl_real_ot_schema.py::test_so_ids_are_canonical_with_a_colon` |
| Literature `pmid` holds Europe PMC ids (`PPR`, `PMC`, `IND`, ...) in 2,349,085 of 151,961,320 rows | id type `europepmc_id` | `test_dl_identifiers.py` (the plugin's conformance cases) |
| SIGNOR rows are stored in both orientations; the direction is in the biological roles | `edges.direction_from_roles`: edges carry `source_node`/`target_node` | `test_dl_real_data_checks.py::test_signor_direction_comes_from_the_roles` |
| `vbt ds check` ran the data child without the reaper (10.3 GiB on the interaction_evidence table, 27.3 M rows) | `preflight.run_contained` | `test_dl_real_data_checks.py::test_the_data_check_runs_under_the_reaper` |
| A composite reference held every target tuple (14.5 M) | the reference is checked on the sampled tuples | `test_dl_real_data_checks.py::test_composite_references_are_checked_on_the_sampled_tuples` |
| Remote parquet scans and sidecar footers read whole files | ranged reads through the `http_range` layout | `test_dl_real_ot_schema.py::test_remote_scans_and_sidecar_footers_read_what_local_ones_read` |
| An optional container the data lacks made every field `schema_drift` and blocked unrelated tools | a warning naming its fields; only its readers are blocked | `test_dl_real_data_checks.py::test_an_optional_container_the_data_lacks_is_a_warning_with_its_fields`, `::test_an_absent_optional_field_blocks_only_its_readers` |
| R10 reported 18,827 sampled interaction edges "without their reverse": the sample holds whole row groups, the reverse sits in another | the reverses of the sampled edges are looked up in the table (Arrow `is_in` over the two endpoint columns; 3.7 s on interaction, 5.4 s on interaction_evidence, none missing) | `test_dl_real_data_checks.py::test_edge_reverses_are_looked_up_beyond_the_sample` |
| Matrix tables skipped sentinels and the strict axis check | R8 sentinels on axis members; undeclared axis columns are `R4:undeclared` | `::test_a_strict_matrix_reports_undeclared_axis_columns` |
| Census pulls failed under `RLIMIT_DATA` (TileDB reserves read buffers) | per-server `limit_kind: rss` (cgroup, else the watchdog; no data limit) for `single_cell` | `test_dl_memory_v2.py::test_rss_containment_sets_no_data_limit`, `test_dl_launcher.py::test_build_launch_spec` |
| Native `data` tools lost the child's header and turned its errors into empty results | the child's `_vbt` is kept; `tool_error` envelopes are raised as their kind | `test_dl_native_tools.py::test_data_verbs_through_the_gateway_carry_the_calling_agent` |
| h5ad file checks required `feature_id` on obs | `var_key_columns` checks the var axis | `test_dl_gateway_leakage_files.py::test_file_checks_find_key_columns_on_either_axis` |
| Calls with free text had no remote witness | `engine_param` (see F20) | `test_dl_live_sources.py::test_engine_text_goes_to_its_request_parameter`, `::test_pubmed_query_is_counted_by_esearch` |
| interaction_evidence holds 24,280 exact copies of rows, so R5b's content check failed `key_violation` in every session | `key.row_identity: none` (copies occur and are counted as stored; R5b tests no uniqueness, R5 still reads key nulls), also on interval | `test_dl_real_data_checks.py::test_rows_without_identity_keep_their_copies_and_still_check_key_nulls` |
| target_go `gene: [id]` counted GO terms (63 "genes" for PCSK9): inside an item a bare name is the item's field | `/id`; lint warns when a bare name in an item table's grains, rank or key resolves to an item field while the table has that column | `test_dl_descriptor.py::test_rule_reference_item_field_shadowing_a_column` |
| The sample-tier memory estimate counted pandas bytes only: 0.51-0.85 of the measured peak of the 25.09 loads over 50 MB (target_prioritisation 0.34) | `peak_bytes_per_row`: pandas plus the co-resident Arrow table, 0.81-1.40 (median 1.06; target_prioritisation 0.54); older calibrations are summed from their columns | `test_dl_memory_v2.py::test_an_older_calibration_counts_its_arrow_columns_too`, `::test_fitted_factors_reproduce_the_sample` |
| A killed bridge left the reaper (re-parented to init) and a `vbt ds check` data child running | the reaper watches `getppid()`: SIGTERM to the child's group, SIGKILL 5 s later, `"orphaned": true` in the exit marker | `test_dl_launcher.py::test_reaper_and_child_die_with_the_launcher` |
| A first call waited for `_stats` only to type witness keys (16.5 s for target, 36.0 s for target_go on 25.09) | the session check reports the footers' storage types and the witness uses them | `test_dl_gateway_flow.py::test_witness_types_come_from_the_session_check_not_a_stats_request`, `test_dl_real_data_checks.py::test_the_check_reports_the_storage_types_stats_would` |

### Running the unmodified servers on Open Targets 25.09

Every 25.09 shard is a single row group (study: one group of 1,964,234 rows), and most of these bugs come
from that shape. The tests are in `test_dl_real_ot_servers.py`. Its offline cases use small fixtures with
the real shape; the opt-in cases need `VBT_DL_REAL_DATA`. Before and after numbers:
[DATA_LAYER_REAL_DATA.md §5](DATA_LAYER_REAL_DATA.md#5-memory-and-latency).

| Finding | Fix | Regression test |
|---|---|---|
| Readiness converted whole row groups to Python, tested R9 references row by row, rendered every key, fell back to Python for keys wider than 63 bits, and ran declared sampled item-key checks in full (target standard check 184 s, expression 330 s) | samples convert only the sampled rows; R9 uses Arrow `is_in`; keys and container counts are counted on Arrow arrays, wide keys too; item keys are sampled by parent row (target 51 s, expression 27 s) | `::test_r9_matches_flat_references_with_arrow_not_row_by_row`, `::test_a_sample_converts_only_the_sampled_rows_of_a_large_row_group`, `::test_flat_key_uniqueness_is_counted_on_arrow_arrays`, `::test_a_wide_key_is_counted_on_arrow_arrays_too`, `::test_a_sampled_item_key_check_samples_parent_rows`, `::test_container_counts_are_read_from_arrow_arrays`; opt-in `::test_real_standard_checks_of_the_target_server_tables_are_ready_in_minutes` |
| A scan converted a whole row group at once: the data child died with MemoryError on `get_interactions` | scans convert slices; item tables push conjuncts on their parent row to Arrow (expression witness 25.1 s / 1,458 MB, now 1.43 s / 338 MB) | `::test_a_scan_converts_a_large_row_group_in_slices`, `::test_an_item_table_pushes_conjuncts_on_its_parent_row` |
| A predicate on a column that only says what null means was not pushed to Arrow | `missing:` is not cleaning | `::test_a_predicate_on_a_column_that_only_says_what_null_means_is_pushed_to_arrow` |
| The derived name search matched labels of other entities listed in a row (TP63 first for "TP53") | only a row's own labels count | `::test_search_matches_a_rows_own_labels_not_those_of_the_entities_it_lists` |
| Every session rebuilt the resolver sidecar already on disk (28-30 s) | reused (3.6-3.8 s) | `::test_a_built_resolver_index_is_reused_for_the_same_data` |
| Returned grains were counted on renamed or projected output rows | counted on the matches, as the totals are | `::test_returned_grains_are_counted_on_the_matches_not_the_output_rows` |
| The seed memory factors were up to 9.4 times too high (target refused at 12.3 GB for a 3.0 GB load) | factors fitted to the measured loads, with a `flat_value` overhead | `::test_the_shipped_memory_factors_estimate_a_real_load`, `::test_flat_values_cost_their_slot_even_when_dictionary_encoded` |
| `vbt ds calibrate` read whole row groups and 30,000 nested rows (MemoryError) | capped by rows and by the seed model's 256 MB budget | `::test_calibration_measures_at_most_the_row_cap_of_a_one_group_shard`, `::test_calibration_of_nested_rows_stays_within_the_seed_budget` |
| Killed `vbt ds check` processes left 4.2 GB of key-check spills | the one-shot check sweeps them too | `::test_a_one_shot_check_sweeps_the_spills_of_killed_checks` |
| `_stats` failed on an item table keyed by position | fixed | `::test_stats_count_an_item_table_keyed_by_position` |

### Review of the real-data work

A review ran the layer again on 25.09 and the live APIs and confirmed the findings below. Each one is fixed,
and the call was repeated on real data
([DATA_LAYER_REAL_DATA.md §4](DATA_LAYER_REAL_DATA.md#4-wrong-answers-of-the-layer-itself-found-on-real-data)).
Every test file names its finding ids.

| Finding | Fix | Tests |
|---|---|---|
| RV-OT-01 A derived serve over its scan budget answered `empty` (`get_interaction_evidence`, every gene) | `too_large` (`scan_budget`) naming `mcp__data__find` | `test_dl_review_realdata.py` |
| RV-OT-02 The W6 short-page re-call rewrote the agent's `output_path` with every match | a tool that writes its rows to a file is never re-called or inflated; a declared `preview` is not a short page; a re-call records the limit it sent | `test_dl_review_realdata.py` |
| RV-OT-03 T11 overwrote count fields that count something else (`num_high_quality` 9 became 228) | only list-form counts of the rows listed are recomputed; `num_high_quality` counts `isHighQuality` | `test_dl_review_realdata.py` |
| RV-OT-04 Rows stored under a salt id were never counted for the parent id (`empty`) | `_vbt.family_rows`, status `partial`, a note naming the salt ids | `test_dl_review_realdata.py` |
| RV-OT-05 Record trims ran twice, the cap was reported as the total, the declared order was ignored | trimmed once by the overlay's cap, the stored length as total, `order: key` applied | `test_dl_review_realdata.py` |
| RV-OT-06 After one over-memory scan the data child stayed at its limit, and every later read failed as an unreadable Parquet file | an Arrow allocation failure is out of memory, never a bad file, and ends the child with a memory exit; scan chunks are sized by the values a row holds | `test_dl_review_realdata.py` |
| RV-OT-07 Returned grains were recounted on output rows; list-path grains counted lists, null included | grains count list elements, never null; a derived result takes the data child's count | `test_dl_review_realdata.py` |
| RV-OT-08 `prioritize_targets(min_genetic_constraint=...)` cut by `targetId` | most constrained first, as upstream ranks; the phase scale is listed as a number | `test_dl_review_realdata.py` |
| RV-OT-09 Substring searches that matched several values were refused | `search_drugs`, `search_pathways` and `get_drug_mechanisms(mechanism)` pool their matches | `test_dl_review_realdata.py` |
| RV-OT-10 Ties within a match class were broken by Ensembl id | the overlay's declared order (`approvedSymbol`) | `test_dl_review_realdata.py` |
| RV-OT-11 `ENSG..._PAR_Y` resolved to the X-chromosome gene | `ambiguous` between the X and Y genes | `test_dl_review_realdata.py` |
| LIVE-1 A trial record withheld by the evidence ceiling stayed in the text | nothing of a withheld record reaches the text | `test_dl_review_realdata.py` |
| LIVE-2 Native find/lookup on CT.gov ignored `data.leakage.ceiling` | the ceiling bounds the request (first posted and last updated), the count, the rows, the header and the provenance | `test_dl_live_review_fixes.py` |
| LIVE-3 `search_pubmed` under a literature ceiling was always `tool_defect` | the witness counts under `VBT_LITERATURE_MAXDATE` | `test_dl_live_review_fixes.py` |
| LIVE-4 The provenance of native live calls had no source, release, table, key or record versions | recorded | `test_dl_review_realdata.py` |
| LIVE-5 Strict `gt`/`lt` were sent as inclusive RANGEs | sent as the next whole number or day | `test_dl_live_review_fixes.py` |
| LIVE-6 Equality on a CT.gov text column became a word search with a fuzzy total | refused; `{search: text}` is the engine match, and says so | `test_dl_live_review_fixes.py` |
| LIVE-7 Census find/lookup ran out of memory under the data-child limit | row reads name their columns and use 16 MiB buffers (counts 32 MiB) | `test_dl_live_review_fixes.py` |
| LIVE-8 Native Census finds counted non-primary cells | `is_primary_data == True` applied and disclosed | `test_dl_live_review_fixes.py` |
| LIVE-9 `columns` on a nested live table returned `{}` rows | dotted paths and the key are kept; an unknown column is `invalid_argument` | `test_dl_live_review_fixes.py` |
| LIVE-10 Unknown ids inside a set were answered as complete successes | `not_found_items`, status `partial` | `test_dl_live_review_fixes.py`, `test_dl_review_realdata.py` |
| LIVE-11 cBioPortal live finds named the fetch time as the release | the study's `importDate`, also recorded as its version | `test_dl_live_sources.py` |
| LIVE-12 `single_cell` results recorded no Census release; `count_cells` had no witness | the resolved release; `count_cells` witnessed from the parsed SOMA filter | `test_dl_live_review_fixes.py` |
| RR-1 The reaper's stderr tee reordered the Bash tool's merged output | merged output keeps its order | `test_dl_review_regressions.py` |
| RR-2 A quarantined overlay dropped its `same_as` bindings, so the generic guard served the tool | the tool is quarantined | `test_dl_review_regressions.py` |
| RR-3 A `limit_kind` typo launched the server without containment | the default containment applies; `vbt doctor` and `vbt ds lint` name the typo | `test_dl_review_regressions.py` |
| RR-4 A CT.gov find failed when only the `/version` request failed | the rows are kept | `test_dl_live_review_fixes.py` |
| RR-5 `vbt ds check --tool` reported a quarantined tool as ready | reported as quarantined (also by `ds explain` and `ds graduate`) | `test_dl_review_regressions.py` |
| RR-6 The offline suite's data child made a live request to cbioportal.org | offline runs point the live sources at a dead loopback port; `VBT_DL_NETWORK=1` keeps the real ones | `test_dl_review_regressions.py` |
| RR-7 Any OOM kill on the host turned an external SIGKILL into `kernel_oom_kill` | only a kernel log record naming the child counts; otherwise the marker says `possible_kernel_oom` | `test_dl_review_regressions.py` |
| RR-8 `VBT_DL_REAL_DATA` meant different directories in different tests | one value, the shared root or the OT 25.09 directory, for every module | `test_dl_review_regressions.py` |
| ACC-1 61 human target genes are stored only with a version suffix as interaction endpoints | stored forms `as_stored` on columns with `integrity: partial` | `test_dl_review_realdata.py` |
| ACC-2 4 real UKB_PPP study ids were rejected | the patterns accept all 1,964,234 study ids | `test_dl_review_realdata.py` |
| ACC-3 The uncontained 10.3 GiB deep check was attributed to the literature table | it was interaction_evidence (corrected above) | |
| ACC-4 The `drug_metadata.targets` delimiter `, ` did not split one real value | `,` | `test_dl_real_tahoe_depmap.py` |
| ACC-5 Two numbers in the literature comments were mislabelled | corrected in `open_targets.yaml` | |

### Live sources served derived, item keys in Arrow (round 3)

The third round ran on the same releases and the live APIs on 2026-10-08
([DATA_LAYER_REAL_DATA.md §8](DATA_LAYER_REAL_DATA.md#8-round-3-derived-live-routes-item-keys-in-arrow-census-admission)).

| Finding | Fix | Tests |
|---|---|---|
| `get_anndata_donor_balanced` balanced donor labels across datasets (six spleen labels belong to two datasets each: 58 labels, 64 donors) and was refused unless one `dataset_id` was fixed | served as the derived `(dataset_id, donor_id)` sample (`count_first.sample`); with one dataset it picks upstream's own 293 of 293 cells | `test_dl_round3_live.py` (offline on a fake Census; `VBT_DL_NETWORK=1` on the real one) |
| A `soma_joinid in [...]` filter asked the server for a 217 M-value vocabulary; "No cells found" was `not_found` on `ensembl_ids` | key columns list no vocabulary; a count-first count of 0 is `empty` | `test_dl_round3_live.py` |
| Count-first admission counted only `gene_symbols` and estimated 40,000,000 bytes for a 200,000-cell pull that was killed at 4,500 MB | every gene argument counts; `count_first.estimate` (a base plus a per-cell slope, every gene of the cell when none is named) fitted to real pulls | `test_dl_round3_live.py::test_count_first_estimates_cover_the_real_pulls`, `::test_live_spleen_pulls_are_sized_by_the_calibrated_estimate` |
| `get_clinical_data` was served upstream only | `serve: derived` from the live cBioPortal tables (equal to upstream for 5 and 92 ACC samples and the 61 samples of `lgg_ucsf_2014`); a phantom sample is `not_found` before any attribute is read; sized like the upstream pull | `test_dl_round3_live.py`, `test_dl_correctness_six.py` (CT-2 against a cBioPortal REST stub of the same study), `test_dl_review_fixes.py::test_clinical_data_is_sized_before_the_download` |
| A derived read's typed error (an unknown study's 404) crossed MCPBridge as `service_unavailable` | `ServeResponse.error` goes on the wire as `refusal` (`ipc.SERVE_ERROR`), for the hierarchy verbs too | `test_dl_round3_live.py::test_live_serve_answers_an_unknown_study_as_not_found` |
| Release lookups, the derived sample and a file's cells travelled as untyped extras of `_census_count` | a `_release` verb (`ReleaseRequest`/`ReleaseResponse`); typed `sample`, `file_cells` and `cells_file` | `test_dl_round3_live.py::test_census_count_request_model_carries_the_sample`, `test_dl_contracts.py` |
| Upstream-served live calls named no release or version (cBioPortal, PubMed, unwitnessed CT.gov counts) | data release, `source.versions` and per-record releases from each source | `test_dl_round3_live.py` |
| `get_census_info` answered `census_version: "stable"` | `result.release_alias`: the body names the dated release (2025-11-08) | `test_dl_live_review_fixes.py::test_count_cells_is_witnessed_and_records_the_release` |
| `eligibility_text` counts had no witness | bound to the criteria column: each phrase is counted as `AREA[EligibilityCriteria]"..."` | `test_dl_live_sources.py::test_engine_text_goes_to_its_request_parameter` |
| Under the evidence ceiling an upstream CT.gov count (first posted) and a native find (also last updated) disagreed without saying why (2,323 against 87) | both totals in `_vbt.ceiling_totals`, with a note | `test_dl_round3_live.py` |
| Deep item-key checks rendered every item in Python (l2g_prediction 791.6 s, target_essentiality 542.3 s) | item keys are counted in Arrow: 23.7-26.5 s and 12.1-12.6 s, every keyed item table of the 31 tables | `test_dl_round3_items.py` |
| R6 counted struct containers as lists (`tep`, `hallmarks` null in every gene) | a struct container counts as null or present (`tep` 41 present) | `test_dl_round3_items.py` |
| No checked way to download an Open Targets release; `R2:manifest_absent` on every table | `vbt data ot list\|fetch\|manifest`: sha1 against `release_data_integrity`, the upstream manifest format | `test_dl_round3_items.py` |

Not done in round 3 (R6, R7, R9 and R10 on matrix tables are done since; see the next section): the CT.gov
leakage policy stays `rows: withhold` (see the comment in `configs/data/sources/clinicaltrials.yaml`).
An item table whose parent has more than 16 M rows (evidence `mutatedSamples[]`) still uses the spilled row
scan. Biosample `ancestors`/`descendants` (refuted as closures) get no readiness finding: hierarchy columns
have no `on_refute` facet and R10 checks only ancestor closures.

### Host-scaled memory, `vbt validate` and projects (wave C)

Packages D3 (host-scaled defaults, `vbt validate`) and D5 (projects), integrated on 2026-10-09 and run on the same
releases.

| Finding | Fix | Tests |
|---|---|---|
| Memory settings were fixed numbers (12,000 MB per server, a 3,000 MB data child) whatever the host | every budget ships `auto` and scales with the host (`memory/sizing.py`, DATA_LAYER.md §14.2): 8,192 MB per server on 16 GB, 293,601 MB on 512 GB; `DataSettings` resolves the data child's and the witness/readiness budgets; `vbt setup` writes the same rule's numbers | `test_dl_round3_memory.py`, `test_setup.py::test_sizing_is_the_rule_auto_applies_at_run_time` |
| `RLIMIT_DATA` was the default containment; Census reads fail under it with `std::bad_alloc` | `limit_kind: rss` by default (a memory cgroup, else the RSS watchdog) | `test_dl_round3_memory.py` (live Census opt-in) |
| `target.get_chemical_probes` and `get_genetic_constraint` were declared `projection` reads, which admission does not size; the unmodified loader read the whole `target` table and the server was OOM-killed at 3,000 MB; functional_genomics' `target_essentiality` was declared a bounded scan and its upstream calls were OOM-killed at 4,400 MB | the 21 `projection` reads of target, pathway, disease and drug and the five `target_essentiality` reads are `full_table` (the upstream loader's `get_dataset` reads every column) | `test_dl_reads_complete.py::test_whole_table_loads_are_not_declared_projections`, `::test_functional_genomics_loads_target_essentiality_whole` |
| The readiness key pass (R5b) held every key part's values: 2,105 MB on 25.09 `interaction`, the floor of every session check | rows are hashed one part and row group at a time; only rows whose hash repeats are counted exactly, in passes bounded by `data.service.max_resident_mb` (which nothing read before). `interaction`'s standard check: 692 MB, 23.6 s; the whole standard session check: largest data child 1,078 MB (`study`), 294.9 s, statuses unchanged | `test_dl_real_ot_servers.py::test_the_arrow_key_pass_is_bounded_by_the_resident_budget` |
| R6, R7, R9 and R10 did not run on matrix tables | they run on each axis's declared columns (`checks.matrix_axis_checks`); DepMap's `@row.ModelID -> model.ModelID` resolves | `test_dl_round3_memory.py::test_matrix_axis_columns_get_r9_references` |
| A column whose name holds a dot (HGNC `pseudogene.org`) could not be declared: R4 split the name at the dot (`schema_drift`), a CSV scan asked for `pseudogene` | a descriptor column key is a literal name everywhere; drafts declare such columns | `test_dl_literal_names_and_links.py`, `test_utilities.py::test_drafts_of_real_shapes_register_as_drafted` |
| The layouts' directory walk followed a directory link loop forever | each real directory is walked once | `test_dl_literal_names_and_links.py` |
| No command checked a host end to end | `vbt validate` (DEPLOYMENT.md §7.5) | `test_validate.py` |

Projects (docs/PROJECTS.md) are wired into the harness: `vbt project ...` and `--project NAME` on `chat`, `run` and
`setup`; `data.project_dir` is a typed setting carried to the data child; plugin discovery imports a project's
plugin module only when its provenance record says registered and the file is unchanged (a module written into
`plugins/` by hand or by an agent's Bash is never imported); the runtime lists the project's utilities from the
configuration; `Runtime.reload_data_layer` and `DataGateway.reload_catalog` serve a registration in the running
session; the run pins its project with digests; `vbt data acquire` puts a project's sources under `<project>/data`;
at the close of a session the run's new agent notes are offered to the project (`projects.notes_at_close`).

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

`when_service_down: strict` (default) refuses guarded calls when the data child is down. A descriptor or
overlay file that cannot be loaded is quarantined on its own: only the servers whose own overlay is
quarantined and the tools that depend on the file are refused (`quarantined`, naming the file); everything
else starts guarded. `lenient` lets those calls run unguarded.

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
   * `engine_param` on a `free_text` argument with `interpreted_as: engine` names the source's request
     parameter the text fills (`query.cond`), so the remote witness counts the same search.
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
| `vbt ds index build [--id-type T] [--table S.T] [--access-paths [--huge]]` | resolver and access-path sidecars; `--table` alone builds only the id types those tables hold |
| `vbt ds estimate [--json]`, `calibrate`, `status` | memory estimates, calibration, per-server memory and the host budget |
| `vbt ds conformance [--kind K] [--plugin NAME] [--list]` | the plugin conformance suites of both registries (the data child's kinds and the harness's `acquisition`); a passing `--plugin` run writes its stamp under `data.cache_dir/conformance/` |
| `vbt ds replay <run> <tool_use_id...> \| --all [--backend bridge\|inprocess]` | re-execute recorded calls (the default `auto` is the guarded bridge) |
| `vbt ds diff-release`, `graduate`, `retro-audit` | release drift, the observe-to-enforce checklist, offline re-classification |
| `vbt verify <run> --data` | fresh fingerprints and replays; missing tables, unreadable fingerprints and lost records leave the run INCOMPLETE |
| `vbt validate [--depth D] [--only/--skip STEPS]` | certify this host on its real data: lint, the session check, the six correctness tests with the gateway enforcing and off, latency, memory, live sources, replication, the model server (DEPLOYMENT.md §7.5) |
| `vbt project init\|list\|show\|check\|profile\|approve\|reject\|memory` | projects (docs/PROJECTS.md); `--project NAME` on `chat`, `run` and `setup` |

Open Targets release files are fetched with `vbt data ot list|fetch|manifest` (see the README): each file's
sha1 is checked against the release's `release_data_integrity`, and `.download-manifest.json` (the upstream
downloader's format, plus sha1) is what readiness R2 reads. `vbt data ot manifest` writes it for tables
already on disk without downloading anything.

### Acquiring data

[DATA_SETUP.md](DATA_SETUP.md) is the guide. Each descriptor's `acquisition` section (`descriptor/models.py`
`AcquisitionSpec`) names a transport plugin, the files of each table, optional preparation steps, the
release, the licence and the variables that point the data layer at the files. Transports are plugins of the
harness-side kind `acquisition` (`plugins/acquisition/`: `http`, `huggingface`, `s3`, `gcs`, `json_index`,
`zip_member`; registry `discover_harness`, conformance suite A-1 to A-7, run with `vbt ds conformance --kind
acquisition`). The engine is `vbt.data.acquire`:

| Command | What it does |
|---|---|
| `vbt data acquire <source>[.<table>] ... [--plan] [--json]` | list the release, plan (bytes, files, time, licence), download in parallel with resume, verify size and checksum, run the preparation step, write `.download-manifest.json`; `--plan --json` prints the plan (an empty plan too) |
| `vbt data acquire --for-tools T ... \| --for-agents A ... \| --all` | the tables those tools or agents read |
| `vbt data acquire --missing \| --pending` | what readiness reports missing; what `auto: ask` queued |
| `vbt data status [--check]` | per source: home, release, files present and verified, readiness |

The download manifests are declared in the descriptors (`manifests:`) of Open Targets, DepMap, GO, MSigDB
and the Zenodo archive (whose manifest sits one level above its root; entries are taken relative to the
root), so R2 compares every file's size with it and requires `complete: true`; DepMap, MSigDB and Zenodo
also compare its `release` with `release.expect`. The Cell Ontology declares none: its table path is a file
variable (`VBT_CL_OBO`) with no root to resolve the manifest against.

A `not_ready` reason whose status acquiring fixes (`missing`, `partial`, `stale`) carries `acquire` (command,
bytes, files, preparation steps, licence, login, and the `data.acquisition.auto` decision) in the refusal's
payload. `data.acquisition` (typed in `settings.py`, defaults in `configs/default.yaml`) sets the root
(`${VBT_DATA_DIR:-data}/sources`), `auto` (`"off"` | `ask` | `under_budget`), `budget_bytes`, `workers`,
`rate_mbps`, `reserve_bytes`, `retries` and `timeout_s`. Under `ask` or `under_budget` the session hands
each turn's `not_ready` refusals to `vbt.data.ondemand.between_turns` after the turn; the next turn waits for
it, records a `data_acquisition` event and re-checks what it acquired. Every acquisition is logged in
`<data.acquisition.root>/acquisitions.jsonl` and, during a run, in the run's `data_acquisitions.jsonl`. The
data child reads the directories its descriptors' variables named when it started: a home they do not point
at is served once they do (`--env-file`, the `host.env` that `vbt setup` writes), and the event lists those
variables as `env_needed`.

### Native tools for agents

`mcp__data__{resolve, describe, lookup, find, search, vocab, members, aggregate, similar, neighbors,
expand, enrich}` are granted to every data agent (`configs/agents.yaml`). `find` and `lookup` on a
live table (CT.gov, cBioPortal) are answered within the source's request budget. A result cut by the
source's page budget is `partial`.

### Memory settings

`data.memory.*` keys:

* `host_mb`, `default_server_mb`, `host_budget_mb` (and the data child's `data.service.mem_limit_mb`,
  `max_resident_mb`, the witness and readiness budgets): `auto` by default, scaled with the host by one rule
  (DATA_LAYER.md §14.2; `vbt validate`'s `host` step prints every value on the host); a number stays as given.
* `limit_kind`: `rss` (the default), `rlimit_data`, `cgroup`, `watchdog` or `none`. It applies to MCP servers and
  to the Bash workspace.
  * A server in `configs/mcp_servers.yaml` may set its own `limit_kind`. `single_cell` uses `rss`: resident
    memory only (a memory cgroup, else the RSS watchdog), with no `RLIMIT_DATA`. The upstream Census
    pulls failed under every `RLIMIT_DATA` tested (4,500 to 40,000 MB), because TileDB reserves read
    buffers.
  * A misspelled value launches the server under the default containment, and `vbt doctor` and
    `vbt ds lint` name it.
* `host_budget_mb`: `auto`, a number, or `off`. The host budget with LRU idle recycle.
* `relay_max_message_mb`: 0 means off. A server message larger than this is replaced by an error with
  the same id.
* `workspace_mb`: the memory limit of agent Bash commands.
