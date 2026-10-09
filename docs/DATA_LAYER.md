# The vbt data layer: descriptors, plugins and one gateway in front of every MCP server

Status: design specification, revision 2 (after the shape and correctness stress test), ready for
implementation. Date: 2026-10-06.
Scope: harness code under `src/vbt/`, configuration under `configs/`, tests under `tests/`.
Hard rule: nothing under `third_party/TheVirtualBiotech` is ever edited. Upstream code is read,
imported read-only (with `python -B`) by detector tests, and launched unchanged as MCP children.

This document synthesises three candidate designs (correctness-first, extensibility-first,
operations-first) and the judges' reviews of them. Appendix B records which idea came from where and
which flaws were deliberately avoided. The test feedback that motivated the work only summarised the
plan (24 fixes in five phases, about 14 engineer-weeks, six correctness tests, 12 data shapes); the
fixes, tests and shapes below were reconstructed from the code-reader maps of all 103 bridged tools.

Revision 2 folds in a stress test: one tester per data shape wrote a real descriptor for a real
instance of that shape (Zenodo and Open Targets 25.09 extracts, Tahoe-100M footers and metadata,
DepMap and GTEx matrices, AnnData cohorts, cBioPortal, ClinicalTrials.gov, `literature_vector`) and
checked it against the models and lint rules of revision 1, and one tester per correctness test
reproduced the claimed failure on unmodified upstream code and checked whether the spec could make it
pass. They found 233 gaps for the shapes (25 blockers, 114 major) and 61 for the tests (6 blockers,
28 major). Every blocker and major gap is fixed below; most minor ones are folded in. §25 lists each
shape and test with its verdict after the revision and names what is still deferred.

Conventions in this document:

- **VERIFIED** marks behaviour a code reader or stress tester reproduced by running unmodified upstream
  code on Open Targets (OT) 25.09-shaped fixtures or on real 25.09 shards and extracts.
- **SCHEMA-DEPENDENT** marks a fact that needs a one-column check on the full real 25.09 release, which
  is not on this machine. The layer never assumes these facts (invariant I9).
- **EST** marks an estimate.
- Upstream paths such as `drug_mcp/tools.py:590` are relative to
  `third_party/TheVirtualBiotech/src/mcp_servers/`. Harness paths start with `src/vbt/`.
- **(rev 2)** marks a mechanism added or changed by the stress test, so reviewers of revision 1 can
  find the deltas.

---

## 0. Summary

1. **Descriptors.** Each data source has one YAML file, `configs/data/sources/<source>.yaml`. It
   declares the source's tables, the **complete key of one record** of each table (composite where
   needed, with declared nullable parts), and exactly **one role for every column**, including fields
   inside nested lists and structs and the axes of wide matrices. Nested containers can be exposed as
   **item tables** with their own grain and key (rev 2). Schema knowledge lives only here.
2. **Overlays.** Each MCP server (upstream, the harness PubMed server, any third-party stdio or HTTP
   server) has one YAML file, `configs/data/overlays/<server>.yaml`. It binds each tool's arguments
   and result fields to descriptor columns, and chooses how the gateway serves the tool: `pass` (call
   upstream behind the guards), `derived` (answer from the descriptor roles), or `block` (typed error
   that names an alternative). Overlays describe code we may not edit.
3. **Four plugin kinds**: `format`, `layout`, `statistic`, `identifier`. Each kind has a Python
   `Protocol`, an entry-point group and a conformance suite that runs automatically against every
   plugin of that kind. Adding a plugin of an existing kind is a new module and nothing else. The
   core changes only to add a new **kind** (phase 4 adds `envelope` as the worked example; a later round
   added `derived`, the grouped computations a `serve: derived` binding names in `derived.split`, so the
   DepMap essentiality and tissue-specificity answers are shipped plugins configured by the overlays and a
   project can add its own; the harness-side `acquisition` kind holds the transports). Optional
   protocol capabilities that later phases need (live requests, matrix slicing, set tests, paired
   aggregation) are declared in phase 1, so later phases add plugins, not protocol methods (rev 2).
4. **One gateway inside `MCPBridge.call`.** Every MCP call passes through it, including
   `preflight.smoke_mcp` probes and third-party servers. It resolves names (by a recorded rule, across
   synonyms, retired IDs, cross-references and crosswalks), turns "not found" into a typed error
   instead of a citable empty answer, checks results against an independent bounded read of the data
   (the **witness**), repairs or refuses truncate-before-rank results, applies role semantics (unknown
   and in-band "unknown" codes never satisfy a filter, negated items never count as support, scope
   dimensions are never pooled silently, patient-level values are never counted per sample), enforces
   memory limits, gates each tool on the readiness of the columns and partitions it reads, withholds
   rows that leak past a configured date ceiling, and stamps every result with a provenance record.
5. **Two processes, two jobs.** The gateway runs in the harness process and never imports pyarrow or
   pandas. All data reading (witness scans, resolver index builds, readiness probes, derived serving)
   runs in a harness-owned MCP child called `data`, launched and memory-limited like every other
   server, with the same interpreter as the upstream servers.
6. **Phase 1 (4.5 engineer-weeks, 10 fixes) stops silent wrong answers without rebuilding data.** It
   reads the existing Parquet files in place (footers plus projected leaf columns) and writes only
   small, deletable, fingerprint-keyed sidecars (resolver tables, row-group value indexes, readiness
   caches). Phases 2 to 5 (9.5 engineer-weeks, 14 fixes) complete the descriptors and plugins, add
   matrix formats, native role-derived tools and an in-process read client, statistics, enrichment,
   composite-key and ontology semantics, memory at scale, live and third-party sources, and
   verification hardening. Total: **24 fixes, 14.0 engineer-weeks**. Revision 2 moved 0.5 ew of work
   into phase 1 (it grew from 4.0 to 4.5) and took it from phases 4 and 5 (§18).
7. **Six correctness tests** run the unmodified upstream servers through `MCPBridge` against small
   fixtures, per case: with the gateway off (a strict `xfail` per case plus positive controls that
   must pass, proving the fixture reproduces today's wrong answer) and on (must pass). All six pass at
   the end of phase 1. The stress test reproduced every claimed current failure on unmodified upstream
   code (§25).

---

## 1. Goals and non-goals

### Goals

- **G1. No silent wrong answers.** Every tool response is exactly one of: rows (`ok` or `partial`),
  an empty result that states its coverage (`empty`), an empty result that cannot be cited
  (`empty_unverified`), or a typed error. "Not found" is always an error.
- **G2. Schema knowledge in data, not code.** A new dataset is a YAML descriptor (plus an overlay if
  it arrives through an MCP server). No per-dataset Python in the core. Identifier syntax that varies
  per dataset (prefix sets, local key patterns) is a plugin **option** in YAML, not new Python (rev 2).
- **G3. Extensible by plugins.** New formats, storage layouts, statistics and identifier rules are
  plugins with conformance suites. The core changes only to add a plugin kind.
- **G4. One gateway for every server**, including servers we do not own.
- **G5. Readiness scoped to the data each tool needs**, down to the columns, nested containers and
  partitions a call reads, enforced at call time.
- **G6. Memory safety**: no server can take down the harness, no out-of-memory (OOM) query is retried
  blindly, and calls that cannot fit are refused before they allocate.
- **G7. Checkable provenance**: every result carries source release, table fingerprints, resolved
  arguments, coverage and canonical row keys; claims and `vbt verify` can check them.
- **G8. Upstream untouched**, and the paper's tool names unchanged so `configs/agents.yaml` keeps
  working.
- **G9. Phase 1 needs no data rebuild.**
- **G10. Determinism.** The same call on the same data gives the same rows in the same order, in the
  gateway and in the upstream children (rev 2: the hash seed pin actually takes effect, §14.2).

### Non-goals

- **N1.** Replacing the upstream servers wholesale. Tools keep their names; most stay upstream-served.
- **N2.** A query engine. pyarrow compute plus the bounded-scan pattern of upstream
  `src/data/query.py::scan_top_rows` is enough; duckdb and polars are not installed and not needed.
- **N3.** Writing to reference data. `OPEN_TARGETS_DATA_PATH` and `TAHOE_DATA_PATH` can be mounted
  read-only.
- **N4.** Fixing scientific judgement. The layer guarantees that answers are what the data says; it
  does not decide whether an agent's interpretation is right.
- **N5.** Monkeypatching upstream code at runtime (no in-process loader guards).
- **N6.** Pinning a floating upstream release we cannot control (Census `"stable"`, hard-coded at
  `single_cell_mcp/tools.py:29,51`). We record the resolved release and detect drift instead.
- **N7.** A mandatory stdout relay between bridge and servers. An optional relay with a byte cap
  arrives in phase 4 with its own fault tests.
- **N8.** Guarding **in-process** reads in phase 1 (rev 2). Harness modules that read reference files
  with pandas (`src/vbt/case_studies/trial_outcomes/pipeline.py:81-86`, `replicate.py:105-108`,
  `vbt.analysis.biomarker.load_processed_cohort`) and agent notebooks bypass MCP. Phase 1 says so in
  the readiness report; phase 2 (F14) adds `vbt.datalayer.client`, an in-process read API that goes
  through the data child with the same resolution, limits and provenance, and moves those harness
  readers onto it.

---

## 2. Failure classes this layer removes

Each row is grounded in the code-reader maps. The last column names the mechanism (section) and fix.

| # | Class | Representative evidence | Mechanism | Fix |
|---|---|---|---|---|
| C1 | Wrong identifier form reads as absence: symbol, versioned ENSG, lower case, `EFO:` colon CURIE, `chr19`, PMCID as PMID | `target_mcp/tools.py:123`, `disease_mcp/tools.py:101`, `genetics_mcp/tools.py:141`, `pubmed_server.py:232` (`PMC1234` returns PMID 1234, VERIFIED) | Identifier plugins normalise, the resolver resolves by recorded rule, unresolved values raise `not_found` (§11.5) | F4, F6 |
| C2 | Not-found and empty are citable successes | `_EMPTY_LOOKUP` at `src/vbt/tools/mcp_bridge.py:132`; `success:true,count:0` in 15 of 20 drug/association tools; `audit/claims.py` tool_call branch (~L290) checks only `is_error` | Three-way classification; claims accept `empty` only as an explicit absence with covered coverage (§12, §15) | F6, F10 |
| C3 | Wrong key column | `get_mouse_phenotype` matches human IDs against the mouse-ortholog column (`target_mcp/tools.py:808`); `get_interaction_evidence` searches `targetA` only (`interaction_mcp/tools.py:247`) | Witness reads the column the role names; contradiction gives derived rows or `tool_defect` (§11.6) | F6 |
| C4 | Nested-type traps: tools always return empty | `isinstance(x,(list,tuple))` on numpy arrays or dicts: pgx drug path (`drug_mcp/tools.py:649-657`), `search_drugs` target path (`:111-118`), `get_evidence_by_publication` (`association_mcp/tools.py:906-911`), `get_gene_ontology`/`search_go_terms` (`pathway_mcp/tools.py:236,287`), disease synonyms (`disease_mcp/tools.py:161-168`) | Witness contradiction (0 returned, M > 0 present); format conformance forbids numpy leaks in harness reads | F6, F3 |
| C5 | Truncation before ranking, no totals, statistics over the truncated set | `drug_mcp/tools.py:590→601`, `:407→422`, `target_mcp/tools.py:1112→1123`, `genetics_mcp/tools.py:302→313`, regulatory `head(limit*10)` before filtering; `summary_stats` over top N | Limit inflation to the witness total, gateway-side sort and cut, verified top-k, refusal when unverifiable; totals always reported (§11.6) | F6 |
| C6 | Partial composite keys and pooling | Tahoe ignores `concentration` and `plate`; `donor_id` without `dataset_id` (`single_cell_mcp/tools.py:1096`); `known_drug` rows counted as drugs | Complete keys in descriptors (nullable parts declared), `scope` facet with a pooling policy, one scope-completeness rule in `prepare` raising `incomplete_key` in every serve mode (§6.3, §7, §11.3) | F7, F17 |
| C7 | Ignored, overriding or unknown arguments | pgx and mechanisms if/elif; `study_id` overrides filters (`genetics_mcp/tools.py:933`); unknown `method` falls into eCAVIAR (`:1424-1429`); unknown `sort_by` ignored (`target_mcp/tools.py:709-716`) | Argument contracts: vocabularies, `require_any`, `exclusive`, honour-every-bound-argument post-filter (§11.4, §11.7) | F7 |
| C8 | Missing values pass filters or become 0 | `no_safety_events` keeps NaN (`target_mcp/tools.py:670`); `min_year` keeps null years (`association_mcp/tools.py:831`); `min_sample_size` keeps null (`genetics_mcp/tools.py:971`) | Predicate IR uses three-valued logic; unknown never satisfies (I6) | F7 |
| C9 | Wrong scale or encoding | `hasSafetyEvent` −1 vs 1, `maxClinicalTrialPhase` 0–1 vs "0–4", `min_identity` 0–100 | Measure facets `scale`, `encoding`; unverified facts disclosure-only until readiness confirms them from row-group min/max (I9) | F7, F9 |
| C10 | Regex and substring matching over vocabularies | `str.contains(regex=True)` at `target_mcp:183-184`, `disease_mcp:156`, `drug_mcp:107,490`; `'PC-3'` also matches `BxPC-3`; `'cancer'` matches `Non-Cancerous` | Literal escaping; exact category resolution with a substring-collision check | F7 |
| C11 | First row of a non-unique match | rsID → variant `iloc[0]` (`genetics_mcp/tools.py:715`), chromosome+position (`:831`), cell-line metadata `iloc[0]` | Identifier `cardinality: many`; witness one-to-many check → `ambiguous` | F4, F6 |
| C12 | Wrong entity substituted | `fetch_abstracts(['PMC1234'])`; `nctId` overwritten | Wrong-kind rejection before the call; echo reconciliation after it | F4, F6 |
| C13 | Data outage reads as a biological negative | Profile sections become `[]` with no `section_errors` (`target_mcp/tools.py:1382,1471-1481`); degraded servers stay callable (`src/vbt/runtime.py:588`) | Tool-scoped readiness enforced at call time; sections marked unavailable | F9 |
| C14 | OOM, blind retry, server poisoned | Whole-table pandas loads cached forever (upstream `src/data/loader.py:91-147`); bridge retries an OOM once (`mcp_bridge.py:606`); dead session stays `ready` (VERIFIED) | Footer-based admission, cold-call serialisation, reaper launcher with `RLIMIT_DATA`, no retry on OOM (§14) | F8 |
| C15 | Readiness blind to data problems | One check per release (`src/vbt/preflight.py:318`); smoke passes on empty (`:509`) | Per-table schema/role/sentinel checks; smoke fails on empty | F9 |
| C16 | Provenance cannot be checked | No release, no row keys; Census version floats; output files overwritten | `vbt.dataprov/1` record per call; pinned data block | F10, F22 |
| C17 | Agent-facing text wrong | Docstrings promise `chromosome`, "phases 0-4", `'P'` aspect, `R-HSA-109582`; `genomics_burden_addendum.md:20` calls zero rows "a valid outcome" | Text derived from roles; `drop_promises`; prompt fixed | F1, F13 |
| C18 | Defaults narrow scope silently | `country='United States'` (89 vs 274 trials, VERIFIED), `min_score=0.5`, `soma_joinid < max_cells` | Effective defaults materialised and disclosed in `_vbt.scope` | F7 |
| C19 | Statistical defects | BH `m` counts only overlapping sets; background ignores aspect; NaN drops the most selective genes; cartesian merge on `gene_name` | Phase 1 blocks with alternatives; phase 3 statistic plugins and native replacements | F7, F15, F16 |
| C20 | Float key and scope literals miss (rev 2) | Tahoe `concentration` is float32: `ds.field('concentration') == 0.05` matches 0 rows, `pa.scalar(0.05, float32)` matches (VERIFIED, pyarrow 25.0.1); witness and serve agree, so the empty looks honest | Literals compiled in the column's storage type; numeric arguments snapped to the vocabulary snapshot; shortest round-trip rendering (§9.2, §11.3) | F3, F7 |
| C21 | Uncorrelated conditions over nested items (rev 2) | `tissue = liver AND rna.value >= 10` evaluated as two row-level existentials returns a gene whose liver value is low (VERIFIED, probe3.py) | `Any(path, predicate)` quantifier evaluated per item; bindings on one container compile into one `Any` (§9.2) | F3, F7 |
| C22 | Retired, cross-referenced or stored-differently identifiers read as absent (rev 2) | `EFO_0001360` retired into `MONDO_0005148`; `DOID:9352` only in `dbXRefs`; Tahoe stores `'Erdafitinib '`, metadata `'Erdafitinib'`; 29.5% of disease IDs use prefixes outside the plugin's list | Resolver rules `retired`, `xref`, `crosswalk`, per-table stored forms, prefixes taken from the universe (§11.5) | F4 |
| C23 | Patient values counted per sample (rev 2) | `get_clinical_data` copies patient attributes onto every sample (`clinicaltrials_mcp/tools.py:1516-1521`); docstring takes the median OS over samples (VERIFIED with a stubbed pybioportal) | `level` facet and the grain-split transform; statistics deduplicate to the declared level (§7, §11.7) | F7 |
| C24 | Phantom records (rev 2) | unknown `sample_ids` come back as `{sampleId, patientId: null}` and count in `sample_count` (`clinicaltrials_mcp/tools.py:1470-1474`, VERIFIED) | `ResultSpec.exists_when`; null-reference rows withheld (W6) (§11.6) | F6 |
| C25 | Answers leak past the evidence date (rev 2) | ClinicalTrials.gov returns the current record: `overallStatus`, `whyStopped`, results posted after the ceiling (`scenarios/__init__.py:97-99`) | `LeakageSpec` with withhold/redact/inject-filter/block policies, filter injection and transform T1 (§6.1, §11.3, §11.7) | F7 |
| C26 | Nondeterministic upstream output (rev 2) | `-E` makes CPython ignore `PYTHONHASHSEED`; `get_interaction_network` gave edge counts 2, 2, 4, 4, 3, 4 over six runs (VERIFIED) | Reaper strips `-E`/`-I` and builds an explicit allow-listed environment; gateway output sorted by full key (§14.2) | F8 |
| C27 | In-band "unknown" codes and placeholders pass filters (rev 2) | `maximumClinicalTrialPhase = -1`, expression `rna.level = -1`, `drugType = 'Unknown'`, string `'nan'` categories | `missing_values`, `unknown_when` and category `placeholders` map to unknown before any predicate, rank or aggregate (§7) | F7 |
| C28 | Stored flags contradict structure (rev 2) | `ontology.leaf` is false for all 39,530 disease terms while 31,635 have no children; `get_disease_hierarchy` returns it as `is_leaf` (`disease_mcp/tools.py:265`) | Relation constraints checked at readiness; refuted fields dropped or recomputed (§6.3, §13) | F9 |
| C29 | Corrupt or missing shards silently shrink the data (rev 2) | `exclude_invalid_files` skips a truncated shard (22,521 → 4,800 rows, no error); a missing `sourceId=` directory gives a smaller valid dataset (VERIFIED) | Layouts list fragments by name and every listed fragment must be readable; partition sets checked against a declared list (§9.2, §13) | F3, F9 |
| C30 | Ranking or thresholds across incomparable groups (rev 2) | STRING and IntAct scores mixed in one top-k; one `min_confidence` across sources; padj across BH families | `RankSpec.within`, per-group top-k, threshold guard on `comparable_within` measures (§11.6, §11.3) | F6, F7 |

---

## 3. Invariants

The gateway enforces these. Each has a named test (§21).

- **I1. No guessed identity.** An argument that names an entity resolves to exactly one canonical key
  of the **bound column's** id_type by a recorded rule, or the call fails with `not_found`,
  `ambiguous` or `invalid_argument`. No substring matching, no first-of-many, no silent case or
  version folding (folding is a rule and is reported in `_vbt.notes`).
- **I2. Not-found is never a success.** An identifier absent from its identity universe is an error.
  When existence cannot be decided (remote universe unreachable, scan over budget) the outcome is
  `unknown` and a zero-row result becomes `empty_unverified`, never `not_found` and never `empty`.
- **I3. Empty is not absence.** A resolved entity with zero rows gives `status: empty` plus a coverage
  statement, and is citable only as an explicit absence finding with `covered` coverage. Nested
  sections never inherit a parent table's `absent` coverage.
- **I4. Contradiction withholds.** If the witness shows rows the tool did not return, rows outside the
  bound predicate, a different top-k, or rows for records that do not exist, the tool's result is
  never delivered unchanged: it is repaired from the data child or replaced by a `tool_defect` error.
- **I5. Honest counts.** Every list result carries `returned`, `total` (or `total: null` with
  `total_method: unknown`), `truncated` and `order`. Upstream summaries computed over a truncated set
  are recomputed or removed. Counts are stated at their grain (rows, items, entities).
- **I6. Unknown never satisfies a filter.** Null, NaN, declared in-band unknown codes
  (`missing_values`, `unknown_when`) and placeholder values are unknown; unknown fails every predicate
  (three-valued logic, including list quantifiers) and is counted in `excluded_unknown`. Values that do
  not apply (`applies_when`) are counted separately in `excluded_not_applicable`.
- **I7. Complete keys or declared pooling.** Each returned row is identifiable by its table's (or item
  table's) complete key, or the result says which key parts were pooled. Merging, ranking or limiting
  across a scope dimension that the arguments do not fix and whose pooling policy is `forbid` is an
  `incomplete_key` error, in every serve mode.
- **I8. Every argument is honoured or rejected.** No argument is silently ignored, overridden or
  replaced by a fallback. An unbound filter argument with a non-default value is `unsupported_filter`.
- **I9. Unverified schema facts are disclosure-only.** A descriptor fact marked `verified: false`
  (measure facets, encodings, flags, vocabularies, aliases, keys, column existence, hierarchy and
  relation facts) drives filtering or repair only after readiness confirmed it from the data.
- **I10. Readiness is scoped and enforced.** A call runs only if the columns, nested containers and
  partitions it reads pass their checks; every other call keeps working.
- **I11. Reproducible and attributable.** Every result has a provenance record with release, table
  fingerprints, resolved arguments, coverage and canonical row keys, pinned per run.
- **I12. The harness never decodes data.** Harness-process modules never import pyarrow or pandas;
  an architecture test enforces it.
- **I13. Deterministic order (rev 2).** Gateway output is ordered by the declared rank and then by the
  full canonical key; children run with an effective `PYTHONHASHSEED=0`, recorded in provenance.
- **I14. No silent shrinkage (rev 2).** A fragment that is listed but unreadable, or a declared
  partition that is missing, makes the affected partition or table not ready; it never yields a
  smaller valid table.
- **I15. Counted at the declared level (rev 2).** A value whose `level` is coarser than the row
  (patient attributes on sample rows, drug facts on trial records, salt forms of one parent) is
  counted, aggregated and cited once per level key, never once per row.

---

## 4. Architecture

```
  agent ──tool_use──► Runtime._execute (src/vbt/runtime.py:1458)
                          │  reads DataResult.status / .provenance, GatewayError.kind
                          ▼
                     Tool.handler  (MCPBridge._make_handler, mcp_bridge.py:601; now passes ctx)
                          ▼
  ┌──────────────── MCPBridge.call(server, tool, args, ctx)  (mcp_bridge.py:606) ─────────────────┐
  │  plan = await gateway.prepare(server, tool, args, ctx)                                          │
  │     contract lookup ─► readiness gate (columns, partitions) ─► argument contracts ─►            │
  │     identifier resolution ─► witness pre-scan ─► scope completeness ─► leakage filter ─►         │
  │     memory admission (upstream route only) ─► limit inflation ─► route (upstream|derived|blocked)│
  │  if route == upstream:                                                                          │
  │     async with plan.hold():   # cold-call lock per server                                       │
  │        raw = await _call_attempts(...)  # existing retry loop; crashes -> gateway.on_crash      │
  │  return await gateway.finish(plan, raw)                                                         │
  │     classify ─► extract rows via field map ─► role transforms ─► witness checks ─►              │
  │     derived repair ─► sort/cut ─► recount ─► _vbt header ─► record                              │
  └───────────────┬─────────────────────────────────────────────────────┬──────────────────────────┘
                  │ JSON-RPC (stdio)                                     │ call_raw("data", "_witness"|"_serve"|...)
                  ▼                                                      ▼
  reaper.py (stdlib launcher: RLIMIT_DATA, explicit     reaper.py ──► data child (harness-owned FastMCP)
   env with effective PYTHONHASHSEED=0, status file,     src/vbt/datalayer/service/server.py
   VBT_CHILD_EXIT marker)
        ▼                                                  catalog + plugins + bounded reader (pyarrow)
  upstream FastMCP server (unedited)                       verbs: _stats _check _witness _serve
  ${vars.upstream}/src/mcp_servers/*/server.py                    _build_index _vocab _resolve_remote
                                                                  _aggregate _similar (P2), _expand _enrich (P3)
        ▼                                                         (phase 2: public mcp__data__* tools)
  OPEN_TARGETS_DATA_PATH / TAHOE_DATA_PATH (read-only)  ◄──────────┘

  Harness side (no pyarrow):   DataGateway ── Catalog (descriptors + overlays + plugin registry)
                               ├─ Resolver + ResolverIndex (sidecar TSV built by the data child;
                               │   rules per id_type, stored forms, crosswalk chains)
                               ├─ ToolReadiness cache (stat-only signatures per turn)
                               ├─ AdmissionController + ResidencyLedger (per server generation)
                               └─ provenance record builder ─► Runtime ─► trace / logs/data_provenance/
```

**Why two processes.** The gateway must sit in front of every call, so it lives in the harness. Data
reading needs pyarrow, which is not a base dependency of the harness (`pyproject.toml` lists it only
in the `mcp` and `analysis` extras) and whose scans can be large. The `data` child runs under the
same launcher and memory limit as every other server, so a heavy witness scan can never take down
the orchestrator. It uses `${vars.mcp_python}`, the interpreter that already runs the upstream
servers, so it adds no dependency: if the OT servers can run, the data child can run.

**Launching the data child.** `${vars.mcp_python} -E <project_root>/src/vbt/datalayer/service/server.py`.
Because `-E` ignores `PYTHONPATH` and `vbt` may not be installed in that interpreter, `server.py`
starts with a bootstrap that inserts `Path(__file__).resolve().parents[3]` (the `src/` directory) into
`sys.path` before importing `vbt`. It needs `pydantic`, `pyyaml`, `fastmcp` and `pyarrow` in that
interpreter; preflight probes the import (reusing `preflight.probe_imports`, L410). The gateway injects
the server spec (`DataGateway.extra_servers()`), so `configs/mcp_servers.yaml` needs no entry;
an explicit `name: data` entry overrides the injected one.

**Request walk-through** (`mcp__target__get_target_info({"target_id": "PCSK9"})`):
1. `prepare` finds the overlay contract (`target.get_target_info` binds `target_id` to
   `open_targets.target.id`, accepts `ensembl_gene` and `hgnc_symbol`).
2. Readiness: table `open_targets.target` is ready (cached; stat signature unchanged).
3. Resolution: `ensembl_gene.normalize("PCSK9")` rejects; `hgnc_symbol` resolves through the
   sidecar index to `ENSG00000169174` by rule `label_exact:approvedSymbol`, and because the bound
   column's id_type is `ensembl_gene`, the resolver returns the Ensembl ID, not the symbol (I1).
4. Admission: `target` is resident in the target server (ledger), so this is a warm call. The data
   child's own reads are charged to its budget, not to the target server.
5. Upstream is called with `target_id="ENSG00000169174"`.
6. `finish`: the envelope is a record; echo check `$.id == ENSG00000169174` (the field is read from
   the record, not copied from the request); nested arrays over the cap are trimmed in declared order
   with counts; the `_vbt` header is inserted as the first key; the provenance record is attached to
   the `DataResult`; `Runtime._execute` writes it to the trace.

---

## 5. Module layout

All new code lives under `src/vbt/datalayer/`. Files marked **(no pyarrow)** are imported by the
harness process and must not import pyarrow or pandas at module or call level (architecture test).

```
src/vbt/datalayer/
  __init__.py            lazy re-exports: DataResult, GatewayError, build_gateway, load_catalog   (no pyarrow)
  errors.py              ErrorKind, MODEL_SIDE_KINDS, GatewayError(ToolFailure), error_envelope()   (no pyarrow)
  result.py              ResultStatus, DataResult, Header, inject_header(), HEADER_KEY = "_vbt"   (no pyarrow)
  record.py              DataProvenance (vbt.dataprov/1), prov_id(), summary()                     (no pyarrow)
  settings.py            DataSettings.from_config(config): every `data.*` key with in-code defaults (no pyarrow)
  api.py                 GatewayProtocol, CallPlan, RawResult, CrashDecision, ListingDecision, LaunchSpec (no pyarrow)
  ipc.py                 pydantic request/response models for the data child's hidden verbs        (no pyarrow)
  roles.py               Role enum, ROLE_FACETS, COMMON_FACETS, ARROW_COMPAT, parse_path() (EBNF §6.4) (no pyarrow)
  predicate.py           Predicate IR (Eq, In, Cmp, CmpAbs, Range, Contains, Any, All, NonEmpty, KindMatch,
                         CensoredCmp, TextMatch, IsNull, Not, And, Or, Param), RankKey, evaluate() oracle (no pyarrow)
  rowkey.py              canonical row-key JSON, storage-typed float rendering, content_hash (rev 2) (no pyarrow)
  descriptor/
    models.py            SourceDescriptor, IdTypeSpec, TableSpec, ItemsOf, KeySpec, CoverageSpec, MatrixSpec,
                         LeakageSpec, EnrichmentSpec and every other sub-model of §6.1          (no pyarrow)
    columns.py           ColumnSpec: union discriminated by role, one facet model per role (§7) (no pyarrow)
    overlay.py           Overlay, ToolBinding, ArgBinding, ResultSpec, FieldMap, DerivedSpec, BlockSpec, DefectSpec (no pyarrow)
    scoping.py           reference scoping (sibling → enclosing item → table; ^. and / forms) (no pyarrow)
    load.py              YAML load with ${VAR}/${vars.x} expansion (vbt.config helpers), digests (no pyarrow)
    lint.py              lint_descriptor(), lint_overlay() -> list[Finding]                          (no pyarrow)
  catalog.py             Catalog, ToolContract, TableRef; build_catalog(settings, registry)         (no pyarrow)
  plugins/
    __init__.py          KINDS
    base.py              FormatPlugin, LayoutPlugin, StatisticPlugin, IdentifierPlugin + dataclasses (no pyarrow)
    registry.py          PluginRegistry, discover(): builtins + entry points "vbt.datalayer.<kind>"   (no pyarrow)
    conformance/         __init__ (runner), identifier.py, format.py, layout.py, statistic.py, golden.py
    identifiers/         ensembl.py hgnc.py ot_disease.py chembl.py drug_name.py go.py reactome.py so.py hpo.py
                         pmid.py pmcid.py doi.py europepmc.py nct.py variant.py rsid.py study_locus.py chromosome.py
                         depmap.py tahoe.py cbio.py uniprot.py ncbi.py inchikey.py local_key.py (P1) census.py (P4)
    formats/             parquet.py (P1) csv.py jsonl.py h5ad.py zarr.py obo.py gmt.py (P2) soma.py rest_json.py (P4)
    layouts/             single_file.py sharded_dir.py hive.py upstream_only.py (P1) zip_member.py http_range.py (P2) live_api.py soma.py (P4)
    statistics/          numeric.py score.py ordinal.py factor.py pvalue.py llr.py count.py (P1) fdr.py effect.py
                         gene_effect.py coloc.py expression.py similarity.py enrichment.py (P3)
    envelopes/           (P4: the fifth kind, worked example of adding a kind)
  resolve/
    index.py             ResolverIndex over sidecar TSV (gzip + csv, stdlib only)                  (no pyarrow)
    rules.py             closed rule grammar and per-id_type rule lists (rev 2)                   (no pyarrow)
    resolver.py          Resolver: rules, families, retired/xref/crosswalk chains, ambiguity, wrong kind (no pyarrow)
  derive/
    schema.py text.py    argument schemas and agent-facing text from roles (P1 subset, P2 all tools) (no pyarrow)
    tools.py             native tool specs from roles (P2)                                         (no pyarrow)
  gateway/
    gateway.py           DataGateway (implements api.GatewayProtocol), build_gateway()              (no pyarrow)
    classify.py          envelope + structural classification (owns the _EMPTY_LOOKUP semantics)  (no pyarrow)
    contracts.py         argument contracts (selector, order_by, abs ops, list args, escaping)      (no pyarrow)
    scope.py             scope-completeness rule, per-group limits (rev 2)                          (no pyarrow)
    fields.py            result field map: extraction, parent keys, computed fields (rev 2)         (no pyarrow)
    transforms.py        role-driven result transforms T1–T14, sort/cut, recount, trim, header     (no pyarrow)
    soma_filter.py       SOMA value_filter parser → Predicate IR (rev 2)                            (no pyarrow)
    files.py             FileCheckSpec reconciliation, write-once outputs (rev 2)                  (no pyarrow)
    readiness.py         TableReadiness/ToolReadiness cache, stat signatures                       (no pyarrow)
    service_client.py    typed calls to the data child via MCPBridge.call_raw                      (no pyarrow)
  memory/
    estimate.py ledger.py admission.py crash.py                                                    (no pyarrow)
  launch/
    reaper.py            stdlib-only launcher, executed by path (never imports vbt)
  service/               runs only inside the data child; may import pyarrow lazily
    server.py            FastMCP entry, sys.path bootstrap, verb discovery from service/verbs/*.py
    reader.py            bounded scan: projection, pushdown, nested membership, global top-k, exact totals
    verbs/               stats.py check.py witness.py serve.py index_build.py vocab.py resolve_remote.py (P1)
                         + later verbs as new files (aggregate, similar, describe/public (P2); expand, enrich (P3))
  client.py              (P2) in-process read API through the data child (`vbt.datalayer.client`)  (no pyarrow)
  retro_audit.py         offline re-classification of recorded traces                              (no pyarrow)
  cli.py                 add_datasource_parsers(sub) -> `vbt datasource ...` (alias `vbt ds`)      (no pyarrow)

configs/data/sources/    open_targets.yaml tahoe.yaml census.yaml clinicaltrials.yaml cbioportal.yaml pubmed.yaml zenodo.yaml
                         (P2: depmap.yaml gene_ontology.yaml msigdb.yaml cell_ontology.yaml)
configs/data/overlays/   target.yaml disease.yaml drug.yaml association.yaml genetics.yaml expression.yaml
                         interaction.yaml functional_genomics.yaml pathway.yaml single_cell.yaml
                         clinicaltrials.yaml pubmed.yaml _generic.yaml
tests/datalayer/         test_dl_*.py (unique basenames), conftest.py, dl_fixtures.py, dl_upstream.py, servers/
```

Rules:

- `gateway/`, `derive/`, `catalog.py`, `descriptor/` and `service/reader.py` never name a plugin or a
  dataset. Plugin names appear only in YAML and in `plugins/`. An architecture test greps for this.
- Plugin modules import pyarrow only inside the methods that read data, so the harness can import a
  layout plugin to compute stat-only signatures.
- Existing files are changed by as few work packages as possible (§24).
- Every YAML example in this document is extracted and linted in CI (`test_dl_doc_examples.py`), so
  the spec cannot drift from the models again (rev 2; the stress test found 15 lint failures in the
  revision-1 examples).

---

## 6. Descriptors

### 6.1 File and top-level schema

One file per source, `configs/data/sources/<source>.yaml`, schema `vbt.datasource/1`. Validated by
pydantic v2 models with `extra="forbid"`; `${VAR}`, `${VAR:-x}` and `${vars.x}` are expanded with the
same helpers as `vbt.config` (`load_config` expansion), plus `${run.mcp_output_dir}` for tables that a
tool call materialises during a run (rev 2). Revision 1 named several sub-models without defining
them; the stress testers could not write deterministic validators. All of them are defined here
(rev 2), and every YAML example in this document is linted in CI.

```python
# src/vbt/datalayer/descriptor/models.py (no pyarrow). All models: extra="forbid".
class SourceDescriptor(BaseModel):
    schema_: Literal["vbt.datasource/1"] = Field(alias="schema")
    source: str                                   # "open_targets"; qualifies id_types: "open_targets:ensembl_gene"
    title: str
    kind: Literal["local", "remote"] = "local"
    root: str | None = None                       # directory, or a file when the only layout is single_file; None for remote
    release: ReleaseSpec
    manifests: list[ManifestSpec] = []
    defaults: TableDefaults = TableDefaults()     # format, layout, missing
    strict: bool = False                          # phase 1 false; phase 2+ true (every column roled)
    budget: RemoteBudget | None = None            # remote: {requests_per_min: 50, max_pages: 10}
    id_types: dict[str, IdTypeSpec] = {}          # names are qualified by source (§11.5)
    tables: dict[str, TableSpec]
    views: dict[str, ViewSpec] = {}               # phase 2: composite outputs with per-section status
    leakage: LeakageSpec | None = None            # date ceiling policy (rev 2, phase 1)
    concepts: dict[str, list[str]] = {}           # one concept across tables: {plate: [de_permissive.plate, sample_metadata.plate]}

class ReleaseSpec(BaseModel):
    expect: str | None = None                     # "25.09": R2 compares
    from_: str | list[str] = Field(alias="from")  # "literal" | "as_of" | "manifest.<jsonpath>" | "format.<key>"
                                                  # | "endpoint:<path>#<jsonpath>" | "git_commit:<path>"
                                                  # a list combines parts: "<rev>@<sha256(filters)[:12]>"
    resolve: dict[str, str] | None = None         # {result: "$.versionDate"} or {table: version, column: dataTimestamp}
    per: dict[str, str] | None = None             # per-record release: {table: study, column: importDate}

class ManifestSpec(BaseModel):
    path: str | None = None                       # relative to root
    inline: dict[str, dict[str, Any]] | None = None   # {"CRISPRGeneEffect.csv": {bytes: 412345678, md5: "..."}}
    entries: str = "$.files"                      # JSONPath of the {relpath: {bytes, sha256|md5}} map
    required: bool = False                        # absent manifest: warning + release_verified false (rev 2)
    require: dict[str, Any] = {}                  # {complete: true}
    checks: dict[str, str] = {}                   # {rows: "$..."}; "<table>.rows" scopes a check to one table

class IdTypeSpec(BaseModel):
    plugin: str                                   # identifier plugin = identity space ("ensembl_gene")
    options: dict[str, Any] = {}                  # plugin parameters (rev 2): {prefixes: from_universe}, {canonical: "^HALLMARK_..."}
    universe: str | list[str] | UniverseSpec | None = None   # identity universe: decides not_found (I2); list = union
    universe_via: RemoteUniverse | None = None    # {tool, args, path, ttl_s}: an upstream listing tool (remote)
    label_of: str | None = None                   # label kind (drug_name): existence and candidates come from that type
    union: list[str] = []                         # multi-kind id_type (embedding vocabulary words)
    resolve_via: list[str] = []                   # label/synonym columns; containers expand to their synonym leaves
    rules: list[str] | None = None                # ordered resolver rules (§11.5); default per plugin
    retired: RetiredSpec | None = None            # {listed_in: [disease.obsoleteTerms], flag, label_prefix, replaced_by, consider}
    xref_via: list[XrefSpec] = []                 # {column, namespaces: {DOID: doid}} | {column, id_type_from: {field, map}}
    crosswalks: list[CrosswalkSpec] = []          # {name, table, from, to, cardinality: one|many}
    maps_to: list[MapsTo] = []                    # {id_type, via: <crosswalk name | source.table>, cardinality}
    stored_forms: dict[str, str] = {}             # {"de_permissive.drug": "as_stored"}: index keeps per-table stored values
    canonicalize: CanonicalizeSpec | None = None  # {parent: drug_molecule.parentId}: salt/parent families (§11.5)
    disambiguate_with: list[str] = []             # columns shown on every ambiguous candidate
    prefer: list[str] = []                        # disclosed tie-break order among candidate prefixes
    hierarchy: HierarchyRef | None = None         # {table: "gene_ontology.term", columns: [is_a, part_of], reflexive: false}
    extends: str | None = None                    # "open_targets:go_term": add hierarchy/xrefs to another source's type
    index: Literal["local", "remote"] = "local"   # remote: resolve through the data child (huge universes)
    resolvable: bool = True                       # false: opaque key; syntax and existence only
    authority: Literal["current", "release_snapshot"] = "current"   # symbols frozen at a release are not current labels

class UniverseSpec(BaseModel):
    table: str                                    # "target" or "source.table"
    keys: list[str]                               # columns or paths; ["@row.ModelID"] for matrix axes
    mode: Literal["tuple", "union"] = "tuple"     # tuple: (studyId, sampleId); union: targetA ∪ targetB
    where: dict[str, Any] | None = None           # Predicate JSON; may use {"param": "<argument>"}
    per_scope: list[str] = []                     # evaluated per value of these scope columns fixed by the call
    per_fragment: bool = False                    # one universe per fragment (cohort files)

class TableSpec(BaseModel):
    kind: Literal["entity", "fact", "crosswalk", "ontology", "edges", "sets", "matrix",
                  "vectors", "records", "entity_detail"]
    path: str | None = None                       # relative to root; resource name for remote
    items_of: ItemsOf | None = None               # item table (rev 2): {table: target, path: "go[]"}; no path/layout/format
    implements: str | None = None                 # "open_targets:disease@25.09": same logical table as another source's
    lineage: Lineage | None = None                # {source: open_targets, release: "25.09", table: disease}
    layout: str | LayoutRef | None = None
    format: str | FormatRef | None = None         # "none" for remote tables served only upstream
    grain: str                                    # one sentence: what ONE record is
    key: KeySpec                                  # complete key (§6.3)
    alternate_keys: list[KeySpec] = []            # other unique keys (inchiKey); used by resolution and R5b
    grains: dict[str, list[str] | GrainSpec] = {} # counting units; GrainSpec allows canonicalized and unordered grains
    rank: list[RankSpec] = []                     # default order
    constraints: list[ConstraintSpec] = []        # facts about stored values and relations (§6.3)
    coverage: CoverageSpec | None = None
    sentinels: SentinelSpec | None = None
    partitions: dict[str, PartitionSpec] = {}     # hive partition columns: logical columns of the table (§6.3)
    access_paths: list[AccessPath] = []           # declared pruning paths (rev 2): row-group stats or sidecar index
    conditions: ConditionsSpec | None = None      # tuple universe of profiled combinations (Tahoe)
    aggregated_over: list[AggregatedOver] = []    # dimensions pooled by construction (DepMap screens)
    edge: EdgeSpec | None = None                  # kind edges: endpoints, orientation, per-side columns
    pivot: PivotSpec | None = None                # long API rows → wide logical columns
    column_patterns: dict[str, ColumnSpec] = {}   # templated roles: {"{stem}_STATUS": {role: flag, event_of: "{stem}_MONTHS"}}
    roles_from: RolesFrom | None = None           # roles of undeclared columns from a metadata table
    fragment_key: FragmentKey | None = None       # {name: cohort, from: filename_regex, pattern: "(GSE\\d+)"}
    fragment_overrides: dict[str, dict[str, dict[str, Any]]] = {}   # per fragment: column -> facet overrides / absent
    size_class: Literal["auto", "small", "large", "huge"] = "auto"
    size_from: SizeFrom | None = None             # remote: {table: study, column: allSampleCount, row_bytes: 2000,
                                                  #   via: server.tool, arg, path}: the count read before the call
    max_scan_bytes: int | None = None             # per-table override of data.witness.max_scan_bytes
    evidence_nature: EvidenceNature | None = None # {kind: literature_cooccurrence, caveat: "..."}
    materialized_by: MaterializedBy | None = None # tables written by a tool call during a run (Census h5ad)
    remote_probe: RemoteProbe | None = None       # {tool, arg, from_sentinel: true}
    expose: ExposeSpec = ExposeSpec()             # {native: true, withhold_from: [trial-annotator], reason}
    columns: dict[str, ColumnSpec] = {}           # required unless kind == matrix or items_of is set
    matrix: MatrixSpec | None = None              # kind == matrix (§6.7)
    strict: bool | None = None                    # overrides source.strict

class ItemsOf(BaseModel):                         # item table over a nested container (rev 2)
    table: str                                    # parent table
    path: str                                     # "go[]", "geneEssentiality[].depMapEssentiality[].screens[]"

class KeySpec(BaseModel):
    columns: list[str]                            # ordered; nested paths allowed; partition columns allowed
    nullable: list[str] = []                      # (rev 2) parts that may be null; NULLS NOT DISTINCT
    check: Literal["full", "sampled", "none"] = "sampled"
    sample_prefix: list[str] = []                 # sampled: prefix blocks (default: all columns but the last)
    row_identity: Literal["key", "content_hash", "none"] = "key"   # content_hash: key.columns are the grouping
                                                  # key only; none: exact copies occur, counted as stored (no R5b)
    version: str | None = None                    # live records: column identifying the record version
    verified: bool = True                         # false: uniqueness disclosed only until R5b confirms it

class ItemKey(BaseModel):                         # on nested/member containers; a bare list is shorthand for columns
    columns: list[str] = []
    identity: Literal["key", "value", "position"] = "key"   # value: list<string> items; position: no natural key
    check: Literal["full", "sampled", "none"] = "sampled"
    max_items: int | None = None                  # 1 = singleton container (geneEssentiality[])

class GrainSpec(BaseModel):
    columns: list[str] = []
    canonicalize: str | None = None               # "chembl_molecule.parent": count parent families, not salts
    unordered: list[str] = []                     # [targetA, targetB]: count unordered pairs
    by: list[str] = []                            # [sourceDatabase]

class CoverageSpec(BaseModel):
    statement: str                                # agent-facing scope sentence (required)
    absence_means: Literal["absent", "unknown", "censored"] = "unknown"
    censor: CensorSpec | None = None              # censored: {column: padj, op: ">=", value: 0.10, plus: [padj_na]}
    universe: UniverseSpec | None = None          # coverage universe: was the entity measured here
    per_scope: dict[str, str] = {}                # statement per scope value ({signor: "causal edges only"})
    verified: bool = True
    applies_to: Literal["rows", "entity"] = "rows"   # entity: existence only; never inherited by nested containers

class ConstraintSpec(BaseModel):
    column: str
    op: Literal["<", "<=", ">", ">=", "==", "!=", "in", "is_finite", "not_null", "equals_expr", "len_eq", "subset_of"]
    value: Any = None
    expr: str | None = None                       # relation (rev 2): "len(children) == 0", "len(rows)", "l2(vector)"
    tolerance: float | None = None
    origin: str                                   # file:line or document where the fact comes from
    verified: bool = True
    on_refute: Literal["not_ready", "drop_field", "recompute"] = "not_ready"

class Sentinel(BaseModel):
    key: dict[str, Any]                           # {col: value}; {id: X} is shorthand for a one-column key
    orientation: Literal["as_stored", "any"] = "as_stored"   # edges: match either orientation
    expect: dict[str, Any] = {}                   # {col: value | {op: value}, nonempty: [...], contains: {"go[].id": "GO:..."},
                                                  #  items: {"tissues[]": ">=1"}, is_null: [...], is_empty: [...], min_rows: n}
    via: dict[str, Any] | None = None             # remote: {tool, args}
class SentinelSpec(BaseModel):
    present: list[Sentinel] = []; absent: list[Sentinel] = []

class RankSpec(BaseModel):
    column: str                                   # column path, "match_class" (search) or "similarity" (vectors)
    direction: Literal["asc", "desc", "asc_abs", "desc_abs"] = "desc"
    nulls: Literal["last", "first"] = "last"
    statistic: str | None = None
    within: list[str] = []                        # (rev 2) ranking only within these groups (comparable_within)
    verified_by: Literal["witness", "source_server_side", "upstream_full_sort", "none"] = "witness"

class PartitionSpec(BaseModel):                   # hive partition column: a logical column (§6.3)
    column: ColumnSpec                            # role and facets (scope with kind partition, usually)
    type: Literal["string", "int64", "date"] = "string"
    expect: Literal["declared", "manifest", "any"] = "declared"   # declared: directories must equal column.vocab
    mirrored_by: list[str] = []                   # physical columns equal to the partition (datasourceId)

class AccessPath(BaseModel):                      # (rev 2) how a predicate on these columns is pruned
    columns: list[str]
    via: Literal["partition", "row_group_stats", "sidecar_index"]
    build: Literal["readiness", "on_demand"] = "on_demand"   # sidecar: value -> (fragment, row group), in cache_dir

class ConditionsSpec(BaseModel):                  # tuple universe of profiled combinations (rev 2)
    columns: list[str]
    from_: str = Field("self", alias="from")      # "self" or another table
    value_map: dict[str, dict[str, Any]] = {}     # {plate: {strip_prefix: plate}}

class EdgeSpec(BaseModel):                        # kind edges (rev 2)
    a: str; b: str
    directed: bool = False
    directed_when: dict[str, list[Any]] = {}      # {sourceDatabase: [signor]}
    orientation: Literal["canonical", "as_reported", "both"] = "as_reported"
    verified: bool = False                        # confirmed by the sampled reverse-edge check (R10)
    sides: dict[Literal["a", "b"], list[str]] = {}  # columns that swap with the endpoints

class LeakageSpec(BaseModel):                     # (rev 2) the source can return facts dated after the evidence ceiling
    ceiling_from: str = "data.leakage.ceiling"    # config path, or web.literature_max_date (PubMed: the server
                                                  # applies VBT_LITERATURE_MAXDATE itself; the data child bounds its
                                                  # counts and live reads by it, the gateway's T1 does not)
    available_at: str                             # column: when the record became public (studyFirstPostDateStruct.date)
    changed_at: str | None = None                 # column: last content change (lastUpdatePostDateStruct.date)
    partial_dates: Literal["latest", "earliest"] = "latest"   # "2004-01" counts as 2004-01-31
    rows: Literal["withhold", "redact", "stamp"] = "withhold"
    redact: list[str] = []                        # columns nulled when changed_at > ceiling (status, results)
    counts: Literal["inject_filter", "block", "stamp"] = "block"

class MaterializedBy(BaseModel):                  # (rev 2) e.g. Census AnnData written by get_anndata
    tool: str                                     # "single_cell.get_anndata"
    path_from: str                                # "$.output_path"
    value_from_arg: dict[str, dict[str, str]] = {}   # {layer: {raw: count, normalized: normalized_count}}
    release_from: Literal["producer", "file"] = "producer"
```

The remaining small models are: `TableDefaults(format, layout, missing)`, `LayoutRef/FormatRef(plugin,
options)` (a plain string is accepted), `RemoteBudget`, `RemoteUniverse(tool, args, path, ttl_s)`,
`RetiredSpec`, `XrefSpec`, `CrosswalkSpec`, `MapsTo`, `CanonicalizeSpec(parent)`, `HierarchyRef(table,
columns, predicates, reflexive)`, `Lineage(source, release, table)`, `CensorSpec(column, op, value,
plus)`, `AggregatedOver(dimension, how, origin)`, `PivotSpec(index, name_column, value_column,
level_from)`, `RolesFrom(table, name, level_from, type_from, defaults)`, `FragmentKey(name, from,
pattern)`, `SizeFrom(table, column, row_bytes)`, `EvidenceNature(kind, caveat)`, `RemoteProbe(tool, arg,
from_sentinel)`, `ExposeSpec(native, withhold_from, reason)`, `ViewSpec(sections: {name: {verb, table,
single, key_from_args}})`, and `MatrixSpec`/`AxisSpec` (§6.7). `ColumnSpec` is defined in
`descriptor/columns.py` as a union discriminated by `role`, with one facet model per role and one
shared set of common facets (§7); `ColumnSpec.model_validate` rejects any facet not listed for the
role. Each model has one example in this section or in §6.5–§6.8, and each example is linted in CI.

### 6.2 Table kinds

The kind decides which derivations and native verbs apply.

| kind | one record is | key | native verbs (phase 2) |
|---|---|---|---|
| `entity` | one entity; the key universe of an id_type | `[id]` | lookup, search, resolve |
| `entity_detail` | one detail row of an entity (cell line × driver mutation, sample of a patient) | `[id, detail…]` | lookup (list), find |
| `fact` | an observation about one or more entities | composite | find (filter, rank, top-k, distinct, aggregate) |
| `crosswalk` | one identifier mapping, possibly one-to-many | `[from, to]` | resolve |
| `ontology` | one term with parent/ancestor columns | `[id]` | lookup, expand |
| `edges` | one edge from one source | `[source, a, b, …]` | neighbors |
| `sets` | one set membership | `[set_id, member]` | members, enrich (phase 3) |
| `matrix` | one cell of a row × column matrix | `[@row key, @col key]` | find and aggregate on the logical long view (§6.7) |
| `vectors` | one embedding | `[id]` | similar |
| `records` | one record of a live API | `[id]` | none until phase 4 (upstream-served, stamped with as-of) |

**Item tables (rev 2).** Any nested or member container can be declared as a table with `items_of:
{table, path}`. An item table shares the parent's fragments, fingerprint, residency-ledger entry and
roles (its columns are the container's `fields`, declared once on the parent); it adds its own
`grain`, `coverage`, `rank`, `sentinels` and `key.check`. Its complete key is composed automatically:
parent key + the item keys of every intermediate container + the container's own item key, so
`target_essentiality_screens` (path `geneEssentiality[].depMapEssentiality[].screens[]`) is keyed
`(id, tissueId, depmapId)` (the singleton `geneEssentiality[]` has `identity: position, max_items:
1`). `_witness` and `_serve` accept item tables as `table`, so counts, top-k, row keys and coverage
are per item. Native verbs list item tables in their `table` enum. A tool whose rows are items names
the item table in `ResultSpec.rows_of`.

### 6.3 Keys, scope, coverage, constraints, sentinels, verified facts

- **Complete key.** `key.columns` must identify one record: no two records share it. Hive partition
  columns and nested paths may be key parts.
- **Nullable key parts (rev 2).** `key.nullable` lists parts that may be null (OT `known_drug.status`:
  6,021 nulls on 126,689 rows; `interaction.targetB` for non-gene interactors; PharmGKB `genotypeId`
  vs `haplotypeId`). Uniqueness, row-key extraction, row-key citations and witness key sets use
  NULLS-NOT-DISTINCT semantics; readiness R5 skips nullable parts and reports their null counts;
  tie-breaks put nulls last. Lint fails when every key part is nullable or when a `self` identifier is
  nullable.
- **Key checks (rev 2).** `check: full` is a streamed, hash-partitioned uniqueness pass with bounded
  memory (temporary files in the cache). It runs at **standard** readiness depth (R5b) for tables at or
  under `data.readiness.key_check_full_max_rows`, and at deep depth otherwise. `check: sampled` checks
  **prefix blocks**: it samples N values of `sample_prefix` (default: every key column but the last),
  pushes each prefix down and verifies that the remaining parts are unique within the block; for
  sharded layouts it also hash-bucket-samples across fragments, so duplicates that span shards or sit
  in one huge single file are found with probability that does not shrink with file size. Provenance
  records which check last confirmed the key and when. A failed check marks the table (or item
  table) not ready only for tools that read it, because an assumed key corrupts every count.
- **Canonical row keys (rev 2).** `rowkey.canonical(values, storage_types)` renders a key as JSON:
  floats as the shortest repr that round-trips in the **storage** type (float32 `0.05` → `0.05`, not
  `0.05000000074505806`), explicit `null`, list and set parts as sorted canonical JSON, dicts with
  sorted keys, NaN as `null`. Witness key sets, `row_keys`, `row_keys_sha256`, `content_hash` and replay
  use this one function on both sides of every comparison.
- **Alternate keys.** `alternate_keys` (inchiKey on `drug_molecule`) are checked by R5b and offered to
  resolution through an identifier facet `alternate_key: true`.
- **Scope (rev 2).** A column is a scope dimension when its role is `scope` or when it carries the
  common `scope` facet (an identifier such as `Cell_ID_DepMap`, a category such as `datatypeId`, or a
  partition such as `sourceId`). The facet says `kind: condition | replicate | batch | stratum |
  partition` and `pooling: forbid | group | list` (defaults: condition → forbid, replicate and batch →
  list, stratum → group, partition → group) and optionally `pool_ok_for: [exists, count, distinct]`
  and `determined_by: [cols]` (functional dependency, sampled at readiness). The single
  scope-completeness rule (§11.3 step 7) applies it in every serve mode. Lint: every scope column of a
  bound table is in the key or the enclosing item key, or declares `determined_by`, or the table
  declares `aggregated_over` for it; and every scope key column of a bound table is either fixed by
  an argument of the binding (a `gateway_only` argument counts and is auto-derived, §10.1) or has
  `pooling` other than `forbid`.
- **Coverage.** `absence_means: absent` says a resolved entity with no rows truly has none in this
  source; `unknown` (the default) says absence may mean "not covered"; `censored` (rev 2) says a
  missing row means "did not pass the stored filter or was not tested" and is citable only with the
  censor statement ("not significant at padj 0.10, or not tested"). The coverage value of a result is
  computed as follows:

  | absence_means | universe declared | entity in coverage universe | excluded_unknown or excluded_negated > 0 | `_vbt.coverage` |
  |---|---|---|---|---|
  | `unknown` | any | any | any | `unknown` |
  | `absent` | no | — | no | `covered` |
  | `absent` | yes | yes | no | `covered` |
  | `absent` | yes | no | — | `not_covered` |
  | `absent` | any | yes or no universe | yes | `partial_unknown` |
  | `censored` | any | — | — | `censored` |

  Only `covered` makes an empty result citable as an absence (§15.3). The **identity universe**
  (`IdTypeSpec.universe`, decides `not_found`) and the **coverage universe** (`CoverageSpec.universe`,
  decides covered vs not covered) are separate (rev 2): a real gene that DepMap never screened is
  `empty` with `coverage: not_covered`, never `not_found`. Tables named in a coverage universe join
  the tool's readiness set; when such a table is unavailable the coverage is `unknown` with a note.
- **Nested coverage (rev 2).** Nested and member containers never inherit `absent` from their table;
  their default is `unknown`. They declare their own `coverage` and the facets `null_means` and
  `empty_means` (each `absent | unknown | not_assessed`), because a null list (not assessed), an empty
  list (assessed, none) and a list of null items differ. Readiness counts null versus empty lists
  from definition levels by scanning the projected leaf, never from footer `null_count` (which counts
  empty lists too, VERIFIED). The table-level coverage of an `entity` table is `applies_to: entity`.
- **Constraints** record facts about stored data: `{column: padj, op: "<", value: 0.10, origin:
  "tools/prepare_tahoe.py:72"}`. They bound argument schemas (`max_padj ≤ 0.10`) and appear in text.
  Relation constraints (rev 2) tie columns together: `{column: ontology.leaf, op: equals_expr, expr:
  "len(children) == 0", verified: false, on_refute: drop_field}`, `{column: linkedTargets.count, op:
  len_eq, expr: "len(linkedTargets.rows)"}`, `{column: norm, op: equals_expr, expr: "l2(vector)",
  tolerance: 1e-6}`. Readiness checks every constraint (row-group min/max for literal bounds, a sampled
  scan for relations); a refuted constraint applies `on_refute` and the overlay result fields bound to
  that column are dropped or recomputed and listed in `_vbt.removed_fields`.
- **Sentinels** list keys that must exist (`present`) and must not (`absent`), with the expectation
  grammar of `Sentinel.expect` (field equality, `{op: value}`, `nonempty`, `contains` on nested paths,
  `items` counts, `is_null`, `is_empty`, `min_rows`) and composite keys. Readiness and
  `vbt doctor --smoke` use them as positive and negative controls; remote tables name the probing call
  with `via` or `remote_probe`.
- **Verified facts.** Any fact can carry `verified: false`: measure facets (scale, encoding,
  direction), flags (`true_means`), category vocabularies and aliases, keys, column existence,
  coverage, hierarchy facts (closure, inverse, orientation) and constraints (rev 2 widens the list).
  Readiness confirms or refutes each from row-group statistics, a vocabulary snapshot, a sampled scan
  or a sentinel (R6, §13). Until confirmed, the gateway uses the fact only for disclosure (I9). A fact
  that only documentation can confirm (a multiple-testing family) declares `verified_by:
  upstream_doc` (or `manifest`) with the citation, which suppresses the permanent lint warning and is
  disclosed as such.
- **Optional and fragment-specific columns (rev 2).** `optional: true` marks a release-dependent column
  (`known_drug.ctIds`, `disease_hpo.parents`): if missing, readiness warns and only bindings that read
  it become not ready. `present_in: [fragments]` and `fragment_overrides` describe multi-file tables
  whose files differ (four GEO cohorts with different response columns and types); readiness and
  universes are then evaluated per fragment, and existence reports `not_measured_in: [GSE73661]`
  instead of a single yes or no.
- **Partial referential integrity (rev 2).** `integrity: partial` on a `ref`, `hierarchy` or member
  column turns a dangling-reference finding into a warning with counts (2,704 of 4,301
  `childChemblIds` dangle in the real extract) instead of a false outage.
- **Partition columns (rev 2)** are logical columns of the table for every purpose: keys, bindings,
  predicates, facets, sentinels, row keys and `_vbt.scope`. `partitions.<col>.expect: declared` makes
  R1 require the set of partition directories to equal the declared vocabulary: a missing value makes
  that partition not ready and an extra value is `schema_drift`. `mirrored_by` names physical
  columns equal to the partition (`datasourceId`); the reader rewrites predicates on the mirror to the
  partition for pruning, R7 checks equality on a sample per fragment, and strict lint treats the pair
  as one logical column with one role.
- **Cross-source tables (rev 2).** `implements: "open_targets:disease@25.09"` declares that a table of
  another source (the Zenodo extract) is the same logical table, so overlays bound to
  `open_targets.disease` can be served from it when the operator selects that source
  (`data.sources.alias`), and `lineage` records the upstream release in provenance. `ref` may name
  `source.table.column`; lint resolves it across loaded sources and warns (not errors) when the
  source is not loaded.

### 6.4 Path grammar, reference scoping and Parquet leaf mapping (rev 2: normative)

```
path      ::= [ "@" axis "." ] segment { "." segment }
segment   ::= name { "[]" | "[" cond "]" }
cond      ::= name "=" literal                    (item predicate: dbXrefs[source=NCBI_Gene].id)
axis      ::= "row" | "col" | "obs" | "var"       (obs = row, var = col; matrix tables only)
name      ::= identifier | "`" any-char-but-backtick "`"
literal   ::= quoted string | number | "true" | "false" | "null"
```

- Depth is unlimited: `geneEssentiality[].depMapEssentiality[].screens[].geneEffect`,
  `tissues[].protein.cell_type[].name` and `path[][]` (Reactome `list<list<string>>`) are valid.
- `col[]` on a `list<primitive>` addresses the elements; an item key of `["[]"]` means "the value is
  the key" (`identity: value`).
- Keys, bindings, predicates, grains, sentinels, universes and result field maps use the same grammar.
- **Reference scoping** for facets that name another column (`of`, `unit_from`, `comparable_within`,
  `item_key`, `family`, `event_of`, `norm_column`, `length_of`, `determined_by`): a bare name resolves
  to a sibling field in the same struct or item first, then to the fields of each enclosing item
  outward, then to a table-level column. `^.name` forces the parent level, `/name` forces table level.
  Lint resolves every reference under these rules and reports ambiguity.
- **Non-container roles on lists.** A role other than `nested` and `member` on a `list<T>` column (or a
  `list_delimiter` string) applies per element; an equality or membership binding on it compiles to
  `Any(path, Eq(...))`.
- **Parquet leaf mapping.** The format plugin maps `a[].b` to the footer's `path_in_schema`
  (`a.list.element.b`, and the legacy encodings `a.array`, `a.bag.array_element`, `a.element`);
  `large_list` and `large_string` map like their small forms. The mapping is tested in the format suite
  (F-11), because `ColumnStats` and leaf projection are keyed by it.

### 6.5 Example: Open Targets `target` and its item tables

```yaml
# configs/data/sources/open_targets.yaml (excerpt)
schema: vbt.datasource/1
source: open_targets
title: Open Targets Platform
kind: local
release: {expect: "25.09", from: manifest.release}
root: ${OPEN_TARGETS_DATA_PATH}
manifests:
  - {path: .download-manifest.json, required: false, require: {complete: true}}   # bytes + sha256 per file
defaults: {format: parquet, layout: sharded_dir, missing: unknown}
strict: true

id_types:
  ensembl_gene:     {plugin: ensembl_gene, universe: target.id,
                     resolve_via: [target.approvedSymbol, target.symbolSynonyms, target.obsoleteSymbols],
                     rules: [exact, normalized, "label_exact:approvedSymbol", "label_casefold:approvedSymbol",
                             "synonym:previous", "synonym:alias"],
                     disambiguate_with: [biotype, genomicLocation.chromosome]}
  hgnc_symbol:      {plugin: hgnc_symbol, label_of: ensembl_gene}       # 1,613 duplicate symbols in 25.09: label, not key
  ensembl_gene_any: {plugin: ensembl_gene_any, resolvable: false}       # homologue IDs: syntax only
  uniprot_accession: {plugin: uniprot_accession,
                      universe: {table: target, keys: ["proteinIds[].id"], where: {in: ["proteinIds[].source", [uniprot_swissprot, uniprot_trembl]]}}}
  ncbi_taxon:       {plugin: ncbi_taxon, resolvable: false}             # digits; species names are categories
  chemical_probe:   {plugin: local_key, options: {canonical: "^[A-Za-z0-9][A-Za-z0-9_.\\- ]*$"}, resolvable: false}
  ot_disease:       {plugin: ot_disease, options: {prefixes: from_universe}, universe: disease.id,
                     resolve_via: [disease.name, disease.synonyms],
                     retired: {listed_in: [disease.obsoleteTerms]},
                     xref_via: [{column: disease.dbXRefs, namespaces: {DOID: doid, MESH: mesh, OMIM: omim, UMLS: umls}}],
                     disambiguate_with: [ontology.isTherapeuticArea, therapeuticAreas]}
  disease_name:     {plugin: disease_name, label_of: ot_disease}
  chembl_molecule:  {plugin: chembl_molecule, universe: drug_molecule.id,
                     resolve_via: [drug_molecule.name, drug_molecule.synonyms, drug_molecule.tradeNames],
                     canonicalize: {parent: drug_molecule.parentId}}
  drug_name:        {plugin: drug_name, label_of: chembl_molecule}
  reactome_pathway: {plugin: reactome, universe: reactome.id, resolve_via: [reactome.label]}
  go_term:          {plugin: go, universe: go.id, resolve_via: [go.name],
                     retired: {label_prefix: "obsolete "}}               # 8,259 obsolete names in OT go (VERIFIED)
  rsid:             {plugin: rsid, universe: "variant.rsIds[]", index: remote}

tables:
  target:
    kind: entity
    path: target
    grain: one Ensembl human gene
    key: {columns: [id], check: full}
    coverage: {statement: "Every Ensembl human gene in Open Targets 25.09 (existence only).",
               absence_means: absent, applies_to: entity}
    sentinels:
      present: [{key: {id: ENSG00000169174}, expect: {approvedSymbol: PCSK9, nonempty: [pathways, go]}}]
      absent:  [{key: {id: ENSG00000000000}}]
    columns:
      id:              {role: identifier, id_type: ensembl_gene, self: true}
      approvedSymbol:  {role: label, of: id, id_type: hgnc_symbol, unique: false}
      approvedName:    {role: label, of: id}
      biotype:         {role: category, vocab: data}
      symbolSynonyms:  {role: synonym, of: id, synonym_kind: alias,    path: "[].label"}
      obsoleteSymbols: {role: synonym, of: id, synonym_kind: previous, path: "[].label"}
      nameSynonyms:    {role: synonym, of: id, synonym_kind: related,  path: "[].label"}
      obsoleteNames:   {role: synonym, of: id, synonym_kind: obsolete, path: "[].label"}
      functionDescriptions: {role: text}
      genomicLocation:
        role: nested
        fields:
          chromosome: {role: position, part: chrom, build: GRCh38, chrom_style: bare}
          start:      {role: position, part: start}
          end:        {role: position, part: end}
          strand:     {role: qualifier, effect: direction}
      tractability:
        role: nested
        item_key: [modality, id]
        null_means: not_assessed
        empty_means: not_assessed
        fields:
          modality: {role: category, vocab: [SM, AB, PR, OC], scope: {kind: stratum, pooling: group}}
          id:       {role: category, vocab: data}
          value:    {role: flag, missing: unknown, true_means: "bucket criterion met", partition_items: true}
      pathways:
        role: member
        item_key: [pathwayId, topLevelTerm]       # a pathway can sit under two top-level terms (SCHEMA-DEPENDENT)
        membership: {set: {path: "pathwayId", id_type: reactome_pathway}, member: {parent: id},
                     propagation: mixed, propagate_via: {id_type: reactome_pathway, relation: ancestor}}
        coverage: {statement: "Lowest-level Reactome annotations (some items also list an ancestor). A gene missing from a pathway is unannotated there, not shown to be outside it.",
                   absence_means: unknown}
        rank: [{column: pathwayId, direction: asc}]
        fields:
          pathwayId:    {role: identifier, id_type: reactome_pathway, ref: reactome.id}
          pathway:      {role: label, of: pathwayId}
          topLevelTerm: {role: category, vocab: data, projection_of: reactome_pathway, lossy: true}
      go:
        role: member
        item_key: {columns: [id, aspect, evidence, source, geneProduct], check: sampled}
        membership: {set: {path: id, id_type: go_term}, member: {parent: id}, propagation: direct,
                     count_grain: go_membership}
        coverage: {statement: "Direct GO annotations (all evidence codes incl. IEA), not propagated to ancestor terms. A missing annotation is not evidence the gene lacks the function.",
                   absence_means: unknown}
        rank: [{column: id, direction: asc}]
        fields:
          id:          {role: identifier, id_type: go_term, ref: go.id}
          aspect:      {role: category, vocab: [C, F, P], verified: false,
                        aliases: {cellular_component: C, molecular_function: F, biological_process: P},
                        scope: {kind: stratum, pooling: group, determined_by: [id]}}
          evidence:    {role: qualifier, effect: evidence_code}
          source:      {role: reference, ref_kinds: [pmid, go_ref, reactome]}
          geneProduct: {role: identifier, id_type: uniprot_accession}
          ecoId:       {role: payload}
      homologues:
        role: nested
        item_key: [speciesId, targetGeneId, homologyType]
        fields:
          speciesId:   {role: identifier, id_type: ncbi_taxon, verified: false}   # str vs int: SCHEMA-DEPENDENT
          speciesName: {role: category, vocab: data, match: casefold}
          targetGeneId: {role: identifier, id_type: ensembl_gene_any}
          targetGeneSymbol: {role: label, of: targetGeneId}
          homologyType: {role: category, vocab: data}
          queryPercentageIdentity:  {role: measure, statistic: percent, scale: [0, 100], missing: unknown}
          targetPercentageIdentity: {role: measure, statistic: percent, scale: [0, 100], missing: unknown}
          isHighConfidence: {role: flag, missing: unknown}
          priority: {role: measure, statistic: ordinal, missing: unknown}
      chemicalProbes:
        role: nested
        item_key: [id, origin]
        null_means: unknown
        empty_means: absent
        coverage: {statement: "Probes listed by Probes & Drugs as integrated in OT 25.09; an empty list means none listed, not proof that no probe exists.",
                   absence_means: absent, verified: false}   # confirmed at readiness from null vs empty list counts
        fields:
          id:            {role: identifier, id_type: chemical_probe}
          origin:        {role: category, vocab: data}
          isHighQuality: {role: flag, missing: unknown}
          mechanismOfAction: {role: text}
          probesDrugsScore:  {role: measure, statistic: score_0_100, missing: unknown}
          probeMinerScore:   {role: measure, statistic: score_0_100, missing: unknown}
          scoreInCells:      {role: measure, statistic: score_0_100, missing: unknown}
          scoreInOrganisms:  {role: measure, statistic: score_0_100, missing: unknown}
          control: {role: payload}
          drugId:  {role: identifier, id_type: chembl_molecule, ref: drug_molecule.id, integrity: partial}
          targetFromSourceId: {role: payload}
          urls: {role: payload}
      safetyLiabilities: {role: nested, item_key: [event, eventId, datasource], fields: {…}}
      hallmarks:   {role: nested, fields: {attributes: {…}, cancerHallmarks: {…}}}
      constraint:  {role: nested, item_key: [constraintType], fields: {…}}
      tep:         {role: nested, fields: {…}}
      subcellularLocations: {role: nested, item_key: [location, source], fields: {…}}
      targetClass: {role: nested, item_key: [id, level], fields: {…}}
      proteinIds:  {role: nested, item_key: [id, source], fields: {id: {role: identifier, id_type: uniprot_accession}, source: {role: category, vocab: data}}}
      dbXrefs:     {role: nested, item_key: [id, source], fields: {id: {role: payload}, source: {role: category, vocab: data}}}
      transcriptIds: {role: payload}
      canonicalTranscript: {role: payload}
      canonicalExons: {role: payload}
      alternativeGenes: {role: payload}
      # strict: true -> lint fails if any physical column (or nested field) has no role.
      # "{…}" marks fields elided in this document; the shipped file roles every field.

  target_go:                                    # item table (rev 2): rows of get_gene_ontology
    kind: sets
    items_of: {table: target, path: "go[]"}
    grain: one direct GO annotation record of one gene
    # complete key composed automatically: (id, go[].id, go[].aspect, go[].evidence, go[].source, go[].geneProduct)
    key: {columns: [], check: sampled}
    grains: {go_membership: [id, "go[].id"], gene: [id], term: ["go[].id"]}
    coverage: {statement: "Direct GO annotations, not propagated.", absence_means: unknown}
    sentinels: {present: [{key: {id: ENSG00000169174}, expect: {min_rows: 1}}]}

  target_essentiality_screens:                  # item table over three levels
    kind: fact
    items_of: {table: target_essentiality, path: "geneEssentiality[].depMapEssentiality[].screens[]"}
    grain: one DepMap CRISPR screen result of one gene in one cell line (grouped by tissue)
    key: {columns: [], check: sampled}          # composed: (id, depMapEssentiality[].tissueId, screens[].depmapId)
    rank: [{column: geneEffect, direction: asc, nulls: last}]
    coverage: {statement: "DepMap CRISPR screens of cancer cell lines only; a missing gene or line means not screened.",
               absence_means: unknown}
```

`key: {columns: []}` on an item table means "composed from the parent", and lint refuses a non-empty
list there. Phase 1 ships the item tables that tools return as rows: `target_go`, `target_pathways`,
`target_tractability`, `target_homologues`, `target_chemical_probes`, `expression_tissues`,
`target_essentiality_screens`, `drug_indications` and `disease_phenotype_evidence`. For `target_essentiality`, the container `geneEssentiality[]` declares `item_key:
{identity: position, max_items: 1}`, `depMapEssentiality[]` declares `item_key: [tissueId]` and
`screens[]` declares `item_key: [depmapId]`.

### 6.6 Example: composite keys (`known_drug` and Tahoe DE)

```yaml
# configs/data/sources/open_targets.yaml (continued)
  known_drug:
    kind: fact
    path: known_drug
    grain: one (drug, target, disease) clinical-precedence record per phase and trial status
    key: {columns: [drugId, targetId, diseaseId, phase, status], nullable: [phase, status], check: full}
    # unique with NULLS NOT DISTINCT on the real 126,689-row 25.09 file (VERIFIED); up to 21 rows per triple
    grains:
      drug: [drugId]
      parent_drug: {columns: [drugId], canonicalize: chembl_molecule.parent}
      indication: [drugId, diseaseId]
      triple: [drugId, targetId, diseaseId]
    rank: [{column: phase, direction: desc, nulls: last}, {column: drugId, direction: asc}]
    coverage: {statement: "ChEMBL-curated drug-target-indication precedence only; absence means 'no curated drug', not 'undruggable'.",
               absence_means: unknown}
    sentinels: {present: [{key: {drugId: CHEMBL3990033, targetId: ENSG00000169174, diseaseId: EFO_0000319, phase: 3.0, status: null}}]}
    columns:
      drugId:    {role: identifier, id_type: chembl_molecule, ref: drug_molecule.id, form: as_stored}
      targetId:  {role: identifier, id_type: ensembl_gene, ref: target.id}
      diseaseId: {role: identifier, id_type: ot_disease, ref: disease.id}
      ancestors: {role: hierarchy, relation: ancestor, of: diseaseId, closure: transitive, reflexive: false, verified: false}
      phase:     {role: measure, statistic: clinical_phase, scale: [0, 4], missing: unknown, verified: false}
      status:    {role: category, vocab: data}
      urls:      {role: nested, item_key: [url], fields: {niceName: {role: category, vocab: data},
                                                         url: {role: reference, ref_kinds: [url]}}}
      prefName:  {role: label, of: drugId}
      tradeNames: {role: synonym, of: drugId, synonym_kind: alias, path: "[]"}
      synonyms:  {role: synonym, of: drugId, synonym_kind: alias, path: "[]"}
      drugType:  {role: category, vocab: data, placeholders: [Unknown]}
      mechanismOfAction: {role: text}
      approvedSymbol: {role: label, of: targetId}
      approvedName: {role: label, of: targetId}
      targetName: {role: label, of: targetId}
      label:     {role: label, of: diseaseId}
      targetClass: {role: payload}
      ctIds:     {role: reference, ref_kinds: [nct], optional: true}   # declared in 25.06 docs, absent from the 25.09 file
```

```yaml
# configs/data/sources/tahoe.yaml (excerpt)
schema: vbt.datasource/1
source: tahoe_100m
title: Tahoe-100M pseudobulk differential expression (prepared by tools/prepare_tahoe.py)
release: {from: [manifest.source_revision, manifest.filters]}   # "<rev>@<sha256(filters)[:12]>"
root: ${TAHOE_DATA_PATH}
manifests:
  - {path: preparation_manifest.json, required: true, require: {complete: true},
     checks: {de_permissive.rows: "$.rows.permissive", de_permissive.constraints: "$.filters.permissive"}}
id_types:
  tahoe_drug:       {plugin: tahoe_drug, universe: drug_metadata.drug,
                     stored_forms: {de_permissive.drug: as_stored, sample_metadata.drug: as_stored}}
                     # 'Erdafitinib ' (DE) vs 'Erdafitinib' (metadata): the index keeps both forms (VERIFIED)
  depmap_cell_line: {plugin: depmap_cell_line, universe: {table: de_permissive, keys: [Cell_ID_DepMap]},
                     resolve_via: [cell_line_metadata.cell_name]}
                     # qualified name tahoe_100m:depmap_cell_line: never confused with the DepMap source's universe
  tahoe_gene:       {plugin: tahoe_gene_name, universe: gene_metadata.gene_symbol,
                     crosswalks: [{name: tahoe_ensembl, table: gene_metadata, from: ensembl_id, to: gene_symbol, cardinality: one}],
                     maps_to: [{id_type: "open_targets:ensembl_gene", via: tahoe_ensembl, cardinality: one}]}
tables:
  de_permissive:
    kind: fact
    path: tahoe_permissive_padj010.parquet
    layout: single_file
    size_class: huge                                        # EST 1e8-3e8 rows, ~62,600 row groups
    grain: one gene's DESeq2 contrast for one drug at one concentration in one cell line on one plate (replicate wells pooled)
    key: {columns: [drug, concentration, concentration_unit, Cell_ID_DepMap, plate, gene_name],
          check: sampled, sample_prefix: [drug, concentration, concentration_unit, Cell_ID_DepMap, plate]}
    grains: {gene: [gene_name], drug: [drug], drug_dose: [drug, concentration, concentration_unit],
             contrast: [drug, concentration, concentration_unit, Cell_ID_DepMap, plate]}
    access_paths:
      - {columns: [drug], via: row_group_stats}             # 3,924 of 3,987 row groups hold one drug (VERIFIED)
      - {columns: [gene_name], via: sidecar_index, build: readiness}   # no row group holds one gene
    conditions: {columns: [drug, concentration, concentration_unit, Cell_ID_DepMap, plate], from: self}
    constraints:
      - {column: padj, op: "<", value: 0.10, origin: "third_party/TheVirtualBiotech/tools/prepare_tahoe.py:72"}
      - {column: padj, op: ">=", value: 0, origin: "tools/prepare_tahoe.py:71"}
      - {column: log2FoldChange, op: is_finite, origin: "tools/prepare_tahoe.py:70"}
    coverage:
      statement: "Only contrasts with finite padj < 0.10 are stored; 76% of source rows have padj NA and were dropped."
      absence_means: censored
      censor: {column: padj, op: ">=", value: 0.10, plus: [padj_na, not_profiled]}
    rank: [{column: padj, direction: asc, nulls: last, within: [drug, concentration, concentration_unit, Cell_ID_DepMap, plate]},
           {column: log2FoldChange, direction: desc_abs, nulls: last}]
    columns:
      drug:               {role: identifier, id_type: tahoe_drug, ref: drug_metadata.drug}
      concentration:      {role: scope, scope: {kind: condition, pooling: forbid}, unit_from: concentration_unit,
                           vocab: data, statistic: numeric}     # float32 storage: literals cast, rendered 0.05
      concentration_unit: {role: scope, scope: {kind: condition, pooling: forbid}, vocab: data}
      Cell_ID_DepMap:     {role: identifier, id_type: depmap_cell_line, scope: {kind: condition, pooling: forbid}}
      Cell_ID_Cellosaur:  {role: identifier, id_type: cellosaurus, resolvable: false}
      Cell_Name_Vevo:     {role: label, of: Cell_ID_DepMap}
      plate:              {role: scope, scope: {kind: replicate, pooling: list}, vocab: data,
                           aliases_from: {table: sample_metadata, column: plate, strip_prefix: plate}}   # '1' here, 'plate1' there
      gene_name:          {role: identifier, id_type: tahoe_gene, ref: gene_metadata.gene_symbol}
      baseMean:           {role: measure, statistic: numeric, missing: unknown}
      log2FoldChange:     {role: measure, statistic: numeric, direction: signed, missing: unknown}
      lfcSE:              {role: measure, statistic: numeric, of: log2FoldChange, missing: unknown}
      stat:               {role: measure, statistic: numeric, missing: unknown}
      pvalue:             {role: measure, statistic: numeric, scale: [0, 1], missing: unknown}
      padj:               {role: measure, statistic: numeric, scale: [0, 1], missing: unknown,
                           family: [drug, concentration, concentration_unit, Cell_ID_DepMap, plate],
                           comparable_within: [drug, concentration, concentration_unit, Cell_ID_DepMap, plate],
                           verified: false, verified_by: upstream_doc}
      n_cells_trt:        {role: count, counts: "treated cells of this line in the pooled wells"}
      n_cells_ctrl:       {role: count, counts: "DMSO control cells of this line on the plate"}
  cell_line_metadata:
    kind: entity_detail
    path: metadata/cell_line_metadata.parquet
    layout: single_file
    grain: one driver alteration of one cell line (1-57 rows per line)   # NOT one cell line
    key: {columns: [cell_name, Driver_Gene_Symbol, Driver_VarType, Driver_ProtEffect_or_CdnaEffect],
          nullable: [Driver_ProtEffect_or_CdnaEffect], check: full}   # 25 null effects; 5 null Cell_ID_DepMap (VERIFIED)
    columns: {…}
```

Phase-1 descriptors use only phase-1 statistic names (`numeric`, `score_0_1`, `score_0_100`, `percent`,
`ordinal`, `clinical_phase`, `signed_factor`, `pvalue_mantissa_exponent`, `llr_critval`, `count`). A
phase-3 statistic may be named early as `statistic: gene_effect` with `fallback: numeric`; lint accepts
it and the fallback is used, disclosed, until the plugin is registered (rev 2).

### 6.7 Matrices: `MatrixSpec`, the logical long view, DepMap and AnnData (rev 2)

Revision 1 named `MatrixSpec` without defining it and used two incompatible shapes (`layers/obs/var`
and `matrix.axes`). Both are now one model.

```python
class ParseSpec(BaseModel):                       # header- or cell-encoded identifiers ("A1BG (1)")
    pattern: str                                  # regex with named groups
    fields: dict[str, ColumnSpec]                 # each group becomes a virtual field with a role
    on_mismatch: Literal["error", "warn"] = "error"

class AxisSpec(BaseModel):
    name: str                                     # long-view name of the axis ("model", "gene", "cell")
    from_: Literal["column", "header", "index", "table", "positional", "file"] = Field(alias="from")
    column: str | dict[str, int] | None = None    # row-ID column, or {index: 0} when its header varies
    aliases: list[str] = []                       # header spellings across releases (ModelID, DepMap_ID, "")
    exclude: list[str] = []                       # header columns that are not axis members
    parse: ParseSpec | None = None
    ids_from: str | None = None                   # sidecar vocabulary file (embedding matrices)
    index_name: str | None = None                 # AnnData: expose the index under this name
    key: KeySpec
    columns: dict[str, ColumnSpec] = {}           # attributes stored with the axis (obs/var frames)
    attributes_from: dict[str, Any] | None = None # {table: model, key: ModelID, columns: [OncotreeLineage]}

class MatrixSpec(BaseModel):
    axes: dict[Literal["row", "col"], AxisSpec]   # "obs"/"var" accepted as aliases of row/col
    values: dict[str, ColumnSpec]                 # "X", "layers.counts", "raw" (raw may declare its own col axis)
    storage: Literal["dense", "csr", "csc", "text"] = "dense"
    implicit: Literal["none", "zero", "zero_if_measured"] = "none"
    measured_by: dict[str, Any] | None = None     # zero_if_measured: {table: feature_presence, key: [dataset_id, feature_id]}
    sections: dict[str, ColumnSpec] = {}          # obsm (vector role), obsp (edges), uns (payload)
```

- **Logical long view.** A matrix table is served as rows `(<row-axis key fields>, <col-axis key
  fields>, <value names>)` plus any axis attributes the call requests. Paths address axes with the
  reserved prefixes `@row.` and `@col.` (`@obs.`, `@var.`), so `@row.ModelID` is never read as field
  `ModelID` of a struct column. `TableSpec.columns` is optional for matrices; the long-view schema is
  derived from the axes.
- **Cells and missing values.** A cell whose value role has `missing: unknown` exists in the long view
  with a null value: returned as null on lookups, counted in `excluded_unknown` on thresholds. With
  `implicit: zero` (sparse counts) absent cells are zeros; with `zero_if_measured` a cell is zero only
  if `measured_by` says the gene was measured in that dataset, otherwise unknown (I6). A predicate that
  zero satisfies is refused unless both axes are restricted within the scan budget.
- **Counting.** Limits, totals and row keys count logical cells. `est_row_bytes` for a matrix is key
  bytes plus value bytes; batch sizes and budgets are in bytes; `key_check_full_max_rows` applies to
  each axis length.
- **Readiness.** Every header must parse (R4), the parsed axis key must be unique, and each value
  column must be numeric (an all-empty column inferred as `null` is reported). Readers address value
  columns by position after the header check, never by a possibly duplicated name (pyarrow keeps
  duplicate CSV headers and `include_columns` silently returns the first, VERIFIED). Axis membership
  changes between releases are reported by `vbt ds diff-release` as membership diffs, not as
  `schema_drift`; a header axis counts as roled for `strict` lint.
- **Formats.** Matrix-capable format plugins implement `axis_values(frag, axis)` and
  `slice(frag, value, row_predicate, col_keys, budget)` returning long-view batches (§9.2). Phase 2
  (F12) ships `csv` (with GCT preamble and dimension-line shape check), `h5ad` and `zarr`.

```yaml
# configs/data/sources/depmap.yaml (phase 2; S4) — DepMap Public 24Q4 CRISPR (Chronos)
schema: vbt.datasource/1
source: depmap
title: DepMap Public CRISPR (Chronos) gene effect and model annotations
root: ${DEPMAP_DATA_PATH}
release: {expect: "24Q4", from: literal}
manifests:
  - inline: {CRISPRGeneEffect.csv: {bytes: 412345678, md5: "<figshare md5>"},    # EST; filled by `vbt ds fingerprint --write`
             Model.csv: {bytes: 1234567, md5: "<figshare md5>"}}
    required: true
defaults: {format: csv, layout: single_file, missing: unknown}
strict: true
id_types:
  depmap_model: {plugin: depmap_cell_line, universe: model.ModelID,
                 resolve_via: [model.CellLineName, model.StrippedCellLineName, model.CCLEName]}
  ncbi_gene:    {plugin: ncbi_gene, options: {input_requires_prefix: true},          # "NCBIGene:3845"; bare digits look like PMIDs
                 universe: {table: gene_effect, keys: ["@col.entrez_id"]},
                 maps_to: [{id_type: "open_targets:ensembl_gene", via: hgnc.complete_set, cardinality: many}]}
tables:
  gene_effect:
    kind: matrix
    path: CRISPRGeneEffect.csv
    grain: one (model, gene) Chronos gene-effect estimate, integrated over all CRISPR screens of the model
    key: {columns: ["@row.ModelID", "@col.entrez_id"], check: full}
    aggregated_over: [{dimension: screen_library, how: chronos_integration, origin: "DepMap 24Q4 release notes"}]
    rank: [{column: gene_effect, direction: asc, nulls: last}]
    coverage:
      statement: "Models that passed CRISPR QC in 24Q4 and genes whose guides passed filtering; an empty cell means not measured, never 0. Not comparable across releases."
      absence_means: unknown
      universe: {table: gene_effect, keys: ["@row.ModelID", "@col.entrez_id"], mode: tuple}
    sentinels:
      present: [{key: {ModelID: ACH-000001, entrez_id: "6122"}, expect: {gene_effect: {lt: -0.5}}}]   # RPL3
      absent:  [{key: {ModelID: ACH-999999}}]
    matrix:
      storage: text
      axes:
        row: {name: model, from: column, column: {index: 0}, aliases: [ModelID, DepMap_ID, ""],
              key: {columns: [ModelID]},
              columns: {ModelID: {role: identifier, id_type: depmap_model, self: true, ref: model.ModelID}},
              attributes_from: {table: model, key: ModelID, columns: [OncotreeLineage, OncotreePrimaryDisease]}}
        col: {name: gene, from: header, exclude: [ModelID, DepMap_ID, ""],
              parse: {pattern: '^(?P<symbol>\S+) \((?P<entrez_id>\d+)\)$',
                      fields: {entrez_id: {role: identifier, id_type: ncbi_gene, self: true},
                               symbol:    {role: label, of: entrez_id, unique: false, authority: release_snapshot}}},
              key: {columns: [entrez_id]}}
      values:
        gene_effect: {role: measure, statistic: gene_effect, fallback: numeric, direction: lower_is_stronger,
                      cutoff: {value: -0.5, op: le, meaning: dependency, origin: "DepMap convention"},
                      comparable_within: [release], missing: unknown}
  model:
    kind: entity
    path: Model.csv
    grain: one DepMap model
    key: {columns: [ModelID], check: full}
    columns: {…}                                             # aliases: per column, release-dependent names
```

```yaml
# configs/data/sources/zenodo.yaml (excerpt; S5) — four GEO cohorts, dense bulk log2 microarray (VERIFIED)
  ibd_cohorts:
    kind: matrix
    path: osmr/code/data/GSE*.h5ad                 # one fragment per file (single_file accepts a glob)
    layout: single_file
    format: {plugin: h5ad, options: {backed: r}}
    fragment_key: {name: cohort, from: filename_regex, pattern: "(GSE\\d+)\\.h5ad$"}
    fragment_overrides:
      GSE73661: {response_clinical: {absent: true}, response_mucosal_healing: {role: category, vocab: [R, NR]}}
      GSE23597: {response_clinical_w30: {role: category, vocab: [R, NR]}}
    grain: one (sample, gene) log2 microarray intensity in one GEO cohort file
    key: {columns: [cohort, "@row.sample_id", "@col.gene_symbol"], check: full}
    coverage: {statement: "A gene absent from a cohort was not on that platform; symbols are pre-2020 HGNC.",
               absence_means: unknown, universe: {table: ibd_cohorts, keys: ["@col.gene_symbol"], per_fragment: true}}
    matrix:
      axes:
        obs: {name: sample, from: index, index_name: sample_id, key: {columns: [sample_id]},
              columns: {sample_id: {role: identifier, id_type: geo_gsm, self: true},
                        patient_id: {role: category, vocab: data, unique_within: [cohort], scope: {kind: replicate, pooling: list}},
                        timepoint: {role: category, vocab: data, placeholders: ["nan"]},
                        age: {role: measure, statistic: numeric, unit: years, missing: unknown, optional: true},
                        sex: {role: category, vocab: data, optional: true}}}
        var: {name: gene, from: index, index_name: gene_symbol, key: {columns: [gene_symbol]},
              columns: {gene_symbol: {role: identifier, id_type: cohort_symbol, self: true, authority: release_snapshot}}}
      values:
        X: {role: measure, statistic: log2_intensity, fallback: numeric, missing: unknown,
            comparable_within: [cohort], verified: false}   # confirmed by the sample_values readiness check
      sections: {uns.cohort_info: {role: payload}, obsm.X_pca: {role: vector, dim: 50}}
```

```yaml
# configs/data/sources/census.yaml (excerpt): remote SOMA, phase 4 layout; phase 1 serves upstream only
schema: vbt.datasource/1
source: cellxgene_census
kind: remote
release: {from: as_of, resolve: {result: "$.census_version"}}   # 'stable' is resolved and recorded, never assumed
tables:
  obs:
    kind: records
    layout: upstream_only                               # phase 4: {plugin: soma, uri: census_data/homo_sapiens/obs}
    format: none
    grain: one cell in one dataset
    key: {columns: [soma_joinid]}
    grains: {donor: [dataset_id, donor_id], dataset: [dataset_id]}
    columns:
      soma_joinid:     {role: identifier, id_type: census_joinid, self: true}
      dataset_id:      {role: category, vocab: data, scope: {kind: batch, pooling: list}}
      donor_id:        {role: category, vocab: data, unique_within: [dataset_id], scope: {kind: replicate, pooling: list}}
      is_primary_data: {role: qualifier, effect: duplicate, default_filter: true}   # enforced and disclosed (§7)
      cell_type:       {role: category, vocab: data, hierarchy_via: cell_type_ontology_term_id}
      cell_type_ontology_term_id: {role: identifier, id_type: cell_ontology}
      tissue:          {role: category, vocab: data}
  anndata_outputs:                                      # files written by get_anndata during a run (rev 2)
    kind: matrix
    path: "${run.mcp_output_dir}"
    materialized_by: {tool: single_cell.get_anndata, path_from: "$.output_path",
                      value_from_arg: {layer: {raw: count, normalized: normalized_count}}}
    grain: one (cell, gene) Census X value for one cell returned by one tool call
    key: {columns: ["@row.soma_joinid", "@col.feature_id"]}
    matrix:
      implicit: zero_if_measured
      measured_by: {table: feature_dataset_presence, key: [dataset_id, feature_id]}
      axes:
        obs: {name: cell, from: index, key: {columns: [soma_joinid]},
              columns: {soma_joinid: {role: identifier, id_type: census_joinid, self: true}}}
        var: {name: gene, from: index, key: {columns: [feature_id]},
              columns: {feature_id: {role: identifier, id_type: census_feature, self: true},   # never positional var_names
                        feature_name: {role: label, of: feature_id, id_type: hgnc_symbol}}}
      values: {X: {role: measure, statistic: count, missing: zero, verified: false}}
```

`materialized_by` makes the gateway register the written file as a fragment of the table in
`finish()` (with its sha256 and the producing call's provenance id), reconcile it with
`FileCheckSpec` (exists, `n_obs == $.n_cells`, key columns present, var index not positional digits
and not `-N` uniquified), and report the table `awaiting_producer` (not `missing`) until a call writes
it (§13).

### 6.8 Stress-test facets by example (rev 2)

```yaml
# S1: disease retired IDs and xrefs; drug_molecule salt families (open_targets.yaml / zenodo.yaml)
  disease:
    kind: ontology
    path: disease
    grain: one ontology term used by Open Targets as a disease, phenotype, measurement or process
    key: {columns: [id], check: full}
    constraints:
      - {column: ontology.leaf, op: equals_expr, expr: "len(children) == 0", verified: false, on_refute: drop_field,
         origin: "false for all 39,530 terms while 31,635 have no children (Zenodo 22259123)"}
    columns:
      id:            {role: identifier, id_type: ot_disease, self: true}
      name:          {role: label, of: id, unique: false}
      obsoleteTerms: {role: identifier, id_type: ot_disease, retired_into: id, path: "[]"}   # 4,880 retired IDs
      dbXRefs:       {role: identifier, xref: true, id_type_from: prefix, cardinality: many, path: "[]"}
      synonyms:      {role: nested, fields: {hasExactSynonym: {role: synonym, of: /id, synonym_kind: exact, path: "[]"},
                                             hasRelatedSynonym: {role: synonym, of: /id, synonym_kind: related, path: "[]"},
                                             hasNarrowSynonym: {role: synonym, of: /id, synonym_kind: narrow, path: "[]"},
                                             hasBroadSynonym: {role: synonym, of: /id, synonym_kind: broad, path: "[]"}}}
      parents:       {role: hierarchy, relation: parent, of: id, closure: direct, inverse_of: children}
      ancestors:     {role: hierarchy, relation: ancestor, of: id, closure: transitive, closure_of: parents, reflexive: false}
      ontology:      {role: nested, fields: {isTherapeuticArea: {role: flag, missing: unknown},
                                             leaf: {role: flag, missing: unknown, verified: false}}}
  drug_molecule:
    kind: entity
    path: drug_molecule
    grain: one ChEMBL molecule (a parent compound or one of its salt forms)
    key: {columns: [id], check: full}
    alternate_keys: [{columns: [inchiKey], nullable: [inchiKey], check: full}]
    grains: {molecule: [id], parent_molecule: {columns: [id], canonicalize: chembl_molecule.parent}}
    columns:
      id:       {role: identifier, id_type: chembl_molecule, self: true}
      parentId: {role: identifier, id_type: chembl_molecule, ref: drug_molecule.id, integrity: partial}   # 31 parents absent
      inchiKey: {role: identifier, id_type: inchikey, alternate_key: true}
      name:     {role: label, of: id, unique: false, placeholder_when: equals_key}   # 5,625 names equal their own ID
      drugType: {role: category, vocab: data, placeholders: [Unknown]}
      maximumClinicalTrialPhase: {role: measure, statistic: clinical_phase, scale: [0, 4], missing_values: [-1],
                                  encoding: {0.5: early_phase_1, 4: approved}, missing: unknown, verified: false}
      childChemblIds: {role: identifier, id_type: chembl_molecule, ref: drug_molecule.id, integrity: partial, path: "[]"}
      crossReferences: {role: nested, item_key: [source], fields: {
                          source: {role: category, vocab: data},
                          ids: {role: identifier, xref: true, path: "[]",
                                id_type_from: {field: ^.source, map: {drugbank: drugbank_id}}}}}
      linkedTargets: {role: nested, fields: {
                          rows: {role: member, membership: {set: {parent: id}, member: {path: "[]", id_type: "open_targets:ensembl_gene"}},
                                 ref: open_targets.target.id},
                          count: {role: count, counts: "linked targets", length_of: rows}}}

# S6: hive evidence with partition checks, mirrored partition column and partition-specific facets
  evidence:
    kind: fact
    path: evidence
    layout: hive
    size_class: huge
    grain: one evidence string, i.e. one target-disease assertion from one data source
    key: {columns: [sourceId, id], check: sampled}
    partitions:
      sourceId: {column: {role: category, vocab: [cancer_biomarkers, cancer_gene_census, chembl, clingen, crispr, crispr_screen,
                          europepmc, eva, eva_somatic, expression_atlas, gene2phenotype, gene_burden, genomics_england,
                          gwas_credible_sets, impc, intogen, orphanet, progeny, reactome, slapenrich, sysbio,
                          uniprot_literature, uniprot_variants],
                          scope: {kind: partition, pooling: group, pool_ok_for: [exists, count, distinct]}},
                 type: string, expect: declared, mirrored_by: [datasourceId]}
    access_paths:
      - {columns: [sourceId], via: partition}
      - {columns: [targetId], via: sidecar_index, build: readiness}       # europepmc shards span every targetId
      - {columns: [diseaseId], via: sidecar_index, build: readiness}
      - {columns: ["literature[]"], via: sidecar_index, build: on_demand}
    columns:
      targetId:  {role: identifier, id_type: ensembl_gene, ref: target.id}
      targetFromSourceId: {role: identifier, resolvable: false,
                           by_partition: {chembl: {id_type: uniprot_accession}, gwas_credible_sets: {id_type: ensembl_gene}}}
      score:     {role: measure, statistic: score_0_1, scale: [0, 1], comparable_within: [sourceId], missing: unknown}
      resourceScore: {role: measure, statistic: numeric, comparable_within: [sourceId], missing: unknown,
                      by_partition: {gwas_credible_sets: {scale: [0, 1]}}}
      clinicalPhase: {role: measure, statistic: clinical_phase, applies_when: {sourceId: [chembl]}, missing: unknown}
      publicationYear: {role: time, precision: year, missing: unknown,
                        fallback: [{column: publicationDate, precision: day}]}   # gwas_credible_sets: year null, date set
      publicationDate: {role: time, precision: day, stored_as: iso8601_string, missing: unknown}
      literature: {role: reference, ref_kinds: [pmid, europepmc_ppr], path: "[]"}
      pValueMantissa: {role: measure, statistic: pvalue_mantissa_exponent, columns: {mantissa: pValueMantissa, exponent: pValueExponent}}
      mutatedSamples: {role: nested, item_key: [functionalConsequenceId], fields: {
                         numberMutatedSamples: {role: count, counts: samples, stored_as: float64}}}

# S8: interaction edges with nullable endpoint and per-side columns
  interaction:
    kind: edges
    path: interaction
    grain: one interactor pair mapped to one gene pair, aggregated over the pair's evidence in one source database
    key: {columns: [sourceDatabase, intA, intB, targetA, targetB], nullable: [targetB], check: full}
    edge: {a: targetA, b: targetB, directed: false, directed_when: {sourceDatabase: [signor]},
           orientation: as_reported, verified: false,
           sides: {a: [intA, intABiologicalRole, speciesA], b: [intB, intBBiologicalRole, speciesB]}}
    grains: {edge: {unordered: [targetA, targetB], by: [sourceDatabase]}, partner: {unordered: [targetA, targetB]}}
    rank: [{column: scoring, direction: desc, nulls: last, within: [sourceDatabase]}]
    coverage:
      statement: "Interactions from IntAct, Reactome, SIGNOR and STRING mapped to genes by OT 25.09; absence means no record in these sources."
      absence_means: unknown
      universe: {table: interaction, keys: [targetA, targetB], mode: union, per_scope: [sourceDatabase]}
    columns:
      sourceDatabase: {role: category, vocab: [intact, reactome, signor, string], scope: {kind: stratum, pooling: group}}
      targetA: {role: endpoint, side: a, id_type: ensembl_gene, ref: target.id}
      targetB: {role: endpoint, side: b, id_type: ensembl_gene, ref: target.id, missing: non_entity}
      intA: {role: identifier, side: a, id_type_from: {field: intASource, map: {uniprotkb: uniprot_accession, chebi: chebi}}}
      intB: {role: identifier, side: b, id_type_from: {field: intBSource, map: {uniprotkb: uniprot_accession, chebi: chebi}}}
      scoring: {role: measure, statistic: score_0_1, scale: [0, 1], comparable_within: [sourceDatabase],
                missing: unknown, by_partition: {reactome: {missing: not_applicable}, signor: {missing: not_applicable}}}
      count: {role: count, counts: "evidence records for this pair in this source", comparable_within: [sourceDatabase]}

# S10: cBioPortal clinical data at two levels, with censoring
  patient_clinical:
    kind: entity_detail
    path: "studies/{studyId}/clinical-data?clinicalDataType=PATIENT"
    pivot: {index: [studyId, patientId], name_column: clinicalAttributeId, value_column: value}
    roles_from: {table: clinical_attribute, name: clinicalAttributeId, level_from: patientAttribute, type_from: datatype}
    grain: one patient's patient-level clinical attributes in one study
    key: {columns: [studyId, patientId], check: none}
    column_patterns:
      "{stem}_MONTHS": {role: time, unit: months, parse: number, missing_values: ["[Not Available]", "[Not Applicable]", "[Unknown]", "[Discrepancy]", NA]}
      "{stem}_STATUS": {role: flag, event_of: "{stem}_MONTHS", encoding: {"1:DECEASED": true, "0:LIVING": false, "DECEASED": true, "LIVING": false,
                        "1:Recurred/Progressed": true, "0:DiseaseFree": false}, missing_values: ["[Not Available]"], verified: false}
    columns:
      studyId:   {role: identifier, id_type: cbio_study, ref: study.studyId}
      patientId: {role: identifier, id_type: cbio_patient, unique_within: [studyId]}
  sample:
    kind: entity_detail
    grain: one specimen of one patient in one study
    key: {columns: [studyId, sampleId], check: none}
    grains: {patient: [studyId, patientId]}
    columns:
      studyId:   {role: identifier, id_type: cbio_study, ref: study.studyId}
      sampleId:  {role: identifier, id_type: cbio_sample, unique_within: [studyId]}
      patientId: {role: identifier, id_type: cbio_patient, ref: {table: patient, on: {studyId: studyId, patientId: patientId}}}
      SAMPLE_TYPE: {role: category, vocab: data}
# id_types: cbio_study  {plugin: cbio_study, universe_via: {tool: clinicaltrials.search_studies, path: "$.studies[*].studyId", ttl_s: 3600}}
#           cbio_sample {plugin: cbio_sample, universe: {table: sample, keys: [studyId, sampleId], mode: tuple}}
#   No upstream tool lists a study's samples, so sample existence is decided per returned row by the
#   overlay's ResultSpec.exists_when (§8.1) and W6 (§11.6), not before the call.

# S12: literature_vector with a multi-kind identifier and a vector relation
  literature_vector:
    kind: vectors
    path: literature_vector
    grain: one word2vec embedding of one Open Targets entity ID learned from Europe PMC co-mentions
    key: {columns: [word], check: full}
    evidence_nature: {kind: literature_cooccurrence, caveat: "Similarity means co-mention in the literature, not function, mechanism or causality."}
    constraints:
      - {column: norm, op: ">", value: 0, origin: "cosine undefined at norm 0", on_refute: not_ready}
      - {column: norm, op: equals_expr, expr: "l2(vector)", tolerance: 1.0e-6, origin: "platform-etl-backend Vectors.scala", verified: false}
    columns:
      word:     {role: identifier, id_type: literature_word, self: true,
                 kind_from: {column: category, map: {target: ensembl_gene, drug: chembl_molecule}, otherwise: ot_disease}}
      category: {role: category, vocab: [target, drug, disease],
                 values: {disease: "any ID that is not ENSG or CHEMBL, including phenotypes and measurements"}}
      norm:     {role: measure, statistic: numeric, missing: unknown}
      vector:   {role: vector, dim: 100, metric: cosine, normalized: false, norm_column: norm, verified: false}
# id_types: literature_word {plugin: ot_entity_any, union: [ensembl_gene, ot_disease, chembl_molecule],
#                            universe: literature_vector.word}

# S9: a GMT collection with a YAML-configured local key and an enrichment contract
  hallmark:
    kind: sets
    path: h.all.v2024.1.Hs.symbols.gmt
    format: gmt
    grain: one (Hallmark gene set, member gene symbol) membership
    key: {columns: [set_id, "members[]"], check: full}
    columns:
      set_id:      {role: identifier, id_type: msigdb_set, self: true}
      description: {role: reference, ref_kinds: [url]}
      members:
        role: member
        item_key: {identity: value}
        membership:
          set: {parent: set_id, id_type: msigdb_set}
          member: {path: "[]", id_type: hgnc_symbol, maps_to: "open_targets:ensembl_gene"}
          propagation: direct
          min_resolved_fraction: 0.95
          enrichment:
            universe: {table: open_targets.target, keys: [id], where: {eq: [biotype, protein_coding]}}
            family: {include_zero_overlap: true, size_bounds: {min_arg: min_size, max_arg: max_size, counted_within: universe}}
            test: hypergeom_enrichment
            correction: fdr_bh
# id_types: msigdb_set {plugin: local_key, options: {canonical: "^HALLMARK_[A-Z0-9_]+$", normalize: [strip, upper]}}
```

### 6.9 Lint rules (`descriptor/lint.py`)

| rule | severity |
|---|---|
| Every physical column and nested field has exactly one role (`strict: true`); header axes and `column_patterns`/`roles_from` count as roled | error (phase 2+); warning in phase 1 |
| `key.columns` exist and have keyable roles (not `text`, `payload`, `vector`, `ignore`); not every part nullable; no nullable `self` identifier; item tables have empty `key.columns` | error |
| Every scope column is in the key or enclosing item key, or declares `determined_by`, or the table declares `aggregated_over` | error |
| Every scope key column of a bound table is fixed by an argument of each binding (auto-derived gateway-only argument counts) or has `pooling` other than `forbid` | error |
| References (`ref`, `of`, `unit_from`, `comparable_within`, `event_of`, `length_of`, `norm_column`, `item_key`) resolve under the §6.4 scoping rules; `source.table.column` refs to unloaded sources are warnings | error |
| `id_type` names resolve to exactly one qualified id_type; two sources declaring the same bare name with different universes require qualified `accepts` in every binding | error |
| Plugin names are registered, or carry `fallback:` (statistics) | error; per-table `plugin_unavailable` at runtime |
| `verified: false` facts have a confirmation path (stats, vocabulary, sampled scan, sentinel) or `verified_by: manifest | upstream_doc` | warning |
| `coverage.absence_means: absent` or `censored` without `statement`; `censored` without `censor` | error |
| Hive partition columns declared in `partitions` with roles; `expect: declared` has a vocabulary | error |
| A digits-only identifier kind is not accepted for free agent input unless `input_requires_prefix` | error |
| `canonicalize`, `retired`, `xref_via`, `crosswalks` and `stored_forms` name existing columns | error |
| Every YAML example in `docs/DATA_LAYER.md` passes this linter (`test_dl_doc_examples.py`) | CI error |

---

## 7. Role vocabulary

Every column, every field of a nested column, and every matrix axis field has exactly **one** primary
role. Facets refine it. The vocabulary is part of the descriptor schema version (`vbt.datasource/1`);
a new role is a schema version bump, not a plugin. The 12-shape stress test (§20, §25) kept the
count at 20: every semantic variation it found became a facet (`scope`, `level`, `missing_values`,
`applies_when`, `event_of`, `kind_from`, ...) or a plugin name, not a new role. Revision 1's facet table
disagreed with its own examples; revision 2 defines `ColumnSpec` as a union discriminated by `role`
(`descriptor/columns.py`) whose facet models are exactly the facets listed here.

| # | Role | Meaning | Role-specific facets |
|---|---|---|---|
| 1 | `identifier` | An entity ID: the table's own (`self: true`) or a foreign key (`ref`) | `id_type`, `self`, `ref` (`table.col`, `source.table.col`, or `{table, on: {col: col}}` for composite refs), `maps_to`, `cardinality` (one, many), `form` (parent, as_stored, any), `alternate_key`, `retired_into`, `xref`, `id_type_from` (prefix or `{field, map}`), `kind_from` (`{column, map, otherwise}`), `resolvable` |
| 2 | `label` | Display name of an identifier | `of`, `id_type` (when the label is itself an ID kind), `unique`, `authority` (current, release_snapshot) |
| 3 | `synonym` | Alternative names of an identifier | `of`, `synonym_kind` (exact, alias, previous, obsolete, related, broad, narrow), `synonym_kind_from` (per-item scope field, OBO) |
| 4 | `category` | Controlled vocabulary | `vocab` (`data` or a list), `match` (exact, casefold), `aliases`, `aliases_from` (`{table, column, strip_prefix}`), `values` (meaning of each value, rendered in text), `hierarchy_via`, `negative_values`, `projection_of` + `lossy` |
| 5 | `scope` | A category that is a condition of the measurement; shorthand for `category` + `scope: {kind: condition}` | the category facets plus `unit_from`, `statistic` |
| 6 | `qualifier` | Changes a row's meaning | `effect` (negate, direction, evidence_code, estimated_actual, duplicate), `default_filter` |
| 7 | `measure` | A number interpreted by a statistic plugin | `statistic` (+ `fallback`), `scale`, `unit`/`unit_from`, `direction` (higher_is_stronger, lower_is_stronger, signed, none), `encoding`, `cutoff` (`{value, op, meaning, origin}`), `comparable_within`, `significant_above`, `of`, `part_of`, `columns` (composite measures: `{mantissa, exponent}`), `family`, `undefined_when`, `levels` (ordered string ordinals) |
| 8 | `count` | Non-negative count of declared things | `counts` (free text: what is counted; renamed from `of`), `distinct_by`, `length_of` (item-relative list it must equal), `comparable_within` |
| 9 | `flag` | Boolean | `true_means`, `encoding` (stored value → true/false), `event_of` (the time column this status censors), `partition_items` |
| 10 | `time` | Date, year or duration | `precision` (year, month, day, second, variable), `unit`, `as_of`, `fallback` (`[{column, precision}]`), `partial_dates` (latest, earliest) |
| 11 | `position` | Genomic coordinate part | `part` (chrom, start, end, pos), `build`, `chrom_style`, `coordinate_base` |
| 12 | `hierarchy` | Ontology link column | `relation` (parent, child, ancestor, descendant), `of`, `closure` (direct, transitive), `reflexive`, `closure_of`, `inverse_of`, `predicate` (is_a, part_of, ...), `target_id_type` |
| 13 | `member` | Set membership (nested list or edge form) | `membership: {set, member, propagation (direct, propagated, mixed), propagate_via, count_grain, min_resolved_fraction, enrichment}`, `propagated_over` |
| 14 | `endpoint` | Graph edge endpoint | `side` (a, b), `directed`, `id_type`, `ref` |
| 15 | `vector` | Embedding | `dim`, `metric`, `normalized`, `norm_column`, `element` (float32, float64) |
| 16 | `text` | Free text | `searchable` |
| 17 | `reference` | Citation pointer | `ref_kinds` (a list of identifier plugin names or `url`; replaces `ref_kind`) |
| 18 | `nested` | Container with typed fields | `fields`, `item_key` (list or `{columns, identity, check, max_items}`), `null_means`, `empty_means`, `null_items` (unknown, skip), `coverage`, `rank` (item order for trims), `grain` |
| 19 | `payload` | Returned verbatim, never filtered or ranked | — |
| 20 | `ignore` | Present, unused, not returned by native verbs | `reason` |

**Common facets (rev 2)**, valid on every role unless the column is `payload` or `ignore`:

| facet | meaning |
|---|---|
| `path` | physical sub-path when the role applies to a nested leaf (`"[].label"`) |
| `missing` | what null means: `unknown` (default), `zero`, `absent`, `not_applicable`, `non_entity` (an edge endpoint that is not a gene), `false` |
| `missing_values` | in-band codes that mean unknown (`-1`, `"[Not Available]"`, `"nan"`); mapped to null before any predicate, rank or aggregate |
| `unknown_when` | item-relative conditions that make the value unknown (`[{eq: -1}]`, `[{column: unit, eq: ""}]`) |
| `placeholders`, `placeholder_when` | display placeholders (`"Unknown"`) or `equals_key` (a name equal to its own ID) → null with a note; on labels and categories |
| `optional`, `present_in` | release- or fragment-dependent existence (§6.3) |
| `applies_when`, `by_partition` | where the column is meaningful, and per partition-value overrides of `id_type`, `scale`, `statistic`, `missing`, `coverage`, `directed` |
| `verified`, `verified_by` | I9 (§6.3) |
| `equals` | this column mirrors another (strict lint treats the pair as one) |
| `level` | the grain at which the value is defined (`patient` on `OS_MONTHS` in a sample-level result) |
| `scope` | `{kind, pooling, pool_ok_for, determined_by}`: the column is also a scope dimension (§6.3) |
| `unique_within` | the value is unique only within these columns (`donor_id` within `dataset_id`) |
| `side` | the endpoint side a per-side column belongs to (edge tables) |
| `list_delimiter` | a string holding a delimited list (`"a|b"`): element semantics as `col[]`; elements are stripped of surrounding whitespace (`","` splits `"A, B"` and `"A,B"` alike) |
| `parse` | `number`, `boolean`, `date`, or a `ParseSpec` producing virtual fields |
| `stored_as` | physical encoding when it differs from the role's natural type (`float64` integral counts, `iso8601_string` dates); R4 accepts it and R6 confirms integrality |
| `integrity` | `full` (default) or `partial` for references that may dangle |
| `remote_name` | the field name in a remote query language (`AREA[LocationCountry]`) |
| `description` | one sentence shown in derived text |

```python
# src/vbt/datalayer/descriptor/columns.py (no pyarrow)
class _Common(BaseModel, extra="forbid"):
    path: str | None = None; missing: MissingKind | None = None; missing_values: list[Any] = []
    unknown_when: list[dict[str, Any]] = []; placeholders: list[Any] = []; placeholder_when: Literal["equals_key"] | None = None
    optional: bool = False; present_in: list[str] | None = None
    applies_when: dict[str, list[Any]] | None = None; by_partition: dict[str, dict[str, Any]] = {}
    verified: bool = True; verified_by: Literal["data", "manifest", "upstream_doc", "none"] = "data"
    equals: str | None = None; level: str | None = None; scope: ScopeFacet | None = None
    unique_within: list[str] = []; side: Literal["a", "b"] | None = None; list_delimiter: str | None = None
    parse: Literal["number", "boolean", "date"] | ParseSpec | None = None; stored_as: str | None = None
    integrity: Literal["full", "partial"] = "full"; remote_name: str | None = None; description: str | None = None

class IdentifierCol(_Common):
    role: Literal["identifier"]; id_type: str | None = None; self_: bool = Field(False, alias="self")
    ref: str | CompositeRef | None = None; maps_to: str | None = None; cardinality: Literal["one", "many"] = "one"
    form: Literal["parent", "as_stored", "any"] | None = None; alternate_key: bool = False
    retired_into: str | None = None; xref: bool = False
    id_type_from: Literal["prefix"] | IdTypeFrom | None = None; kind_from: KindFrom | None = None
    resolvable: bool | None = None
# ... LabelCol, SynonymCol, CategoryCol, ScopeCol, QualifierCol, MeasureCol, CountCol, FlagCol, TimeCol,
#     PositionCol, HierarchyCol, MemberCol, EndpointCol, VectorCol, TextCol, ReferenceCol, NestedCol (fields:
#     dict[str, ColumnSpec], recursive), PayloadCol, IgnoreCol — each with exactly the facets of the table above.
ColumnSpec = Annotated[Union[IdentifierCol, LabelCol, ...], Field(discriminator="role")]
```

`identifier` with `self: true` may omit `id_type` when no tool resolves it (an evidence hash);
`ArgBinding`s cannot then bind it with `accepts`.

### 7.1 What each role derives

| Role | Tool arguments | Agent-facing text | Readiness | Memory | Provenance |
|---|---|---|---|---|---|
| identifier | `type: string` (`array` for list arguments), `x-vbt-id-type`, `x-vbt-accepts`, examples; resolved before the call (no strict `pattern` for resolvable kinds) | "accepts Ensembl ID or HGNC symbol/alias; resolved; unknown values are errors" | column present, string-like type (or `stored_as`); key nulls = 0 for non-nullable parts; R4b canonical form of universe keys; FK sample (R9) | key projection bytes | resolved value, matched and canonical id_type, rule, family |
| label | none directly; feeds the resolver index | "exact name first, never substring" | decodes to strings | sidecar size | `via: label_exact:<col>` |
| synonym | none; feeds the resolver per the id_type's rule list | lists synonym kinds searched; broad/narrow only in `search` | nested path decodes to strings | sidecar size | `via: synonym:<kind>` |
| category | `enum` when vocabulary ≤ `data.derive.enum_max`, else validated at call time with `x-vbt-vocabulary`; numeric vocabularies rendered with storage-typed shortest repr | lists values (with `values` meanings) or names `mcp__data__vocab` | vocabulary snapshot per fingerprint; placeholders excluded | — | value echoed |
| scope (role or facet) | gateway argument (`x-gateway: true`) auto-derived for every scope key column the upstream signature lacks, with enum | "results are per <scope>; pass <arg> or get incomplete_key" (forbid) or "listed per <scope>" (list) | vocabulary snapshot; `determined_by` sampled | — | always echoed in `_vbt.scope` |
| qualifier | `include_negated` or `include_duplicates` (default false), both disclosed; the default filter is enforced or the text says "NOT applied" | "negated annotations excluded (N)", "duplicate cells excluded (N)" | type check | — | excluded counts |
| measure | `minimum`/`maximum` from `scale` and `constraints`; `order_by` enum; abs thresholds (`gt_abs`) | scale, encoding legend, direction, cutoff and its inclusivity, null policy, comparability, family | numeric type; R6 confirms `verified: false` facets (min/max for scales, distinct snapshot for encodings) | 8 B/value | rank and thresholds applied |
| count | integer ≥ 0 | "counts <counts>" | integer type or `stored_as` with integrality check | — | — |
| flag | boolean (tri-state filters true/false/unknown) | "null = unknown, never false"; censoring for `event_of` | boolean type or `encoding` snapshot; censoring consistency (R10) | — | — |
| time | `min_*`/`max_*`; null excluded | "rows with unknown year excluded (N)"; fallback disclosed | type or `stored_as: iso8601_string` | — | leakage ceiling, as-of |
| position | region string normalised (`chr` stripped, build checked) | "GRCh38, chromosome without 'chr'; overlap semantics" | type, build | — | normalised region |
| hierarchy | `include_descendants` (default false; §11.5 rule) | "direct terms only unless include_descendants" | relation invariants (R10), targets ⊆ universe unless `integrity: partial` | list sizes | expansion hash and count |
| member | membership args (`drug_id` matches `drugs[].drugId`); gene lists with `min_resolved_fraction` | "direct annotations only" / "propagated" / "mixed (N%)" | nested type; item-key check; set ids ⊆ set universe (R9); member resolution fraction (R10) | nested factor + num_values | member keys; enrichment statistics block |
| endpoint | one entity argument matching either side when undirected; pair binding | "undirected; one row per unordered pair per source" | both sides present; orientation fact (R10) | — | canonical unordered pair |
| vector | anchor `entity_id`, `top_k ≥ 1` | "cosine in [-1, 1]; uncalibrated; literature co-occurrence" (from `evidence_nature`) | list or fixed-size list of float; `num_values == rows × dim`; norms finite and > 0 | dim × element width | metric, anchor key, values hash |
| text | literal search only | "literal text, not a pattern" | — | excluded from default projection | — |
| reference | membership argument (`pmid`) | "cites PMIDs or preprints" | — | — | carried as references |
| nested | per field | item grain sentence (`grain` facet) | nested type matches `fields`; null vs empty counts from definition levels | num_values × object overhead | item keys |
| payload | none | not described | exists | full-row bytes | — |
| ignore | none | none | exists (warning if missing) | excluded from native projection | — |

---

## 8. Overlays: binding tools we cannot edit

### 8.1 Purpose and schema

An overlay (`configs/data/overlays/<server>.yaml`, schema `vbt.overlay/1`) says, per tool, what the
tool reads, what each argument means in descriptor terms, which result fields are which descriptor
columns, and how the gateway serves it. Overlays never contain code. Revision 2 adds the binding
forms the stress test needed (marked `# rev 2`).

```python
class ToolBinding(BaseModel):
    status: Literal["reviewed", "unreviewed"] = "reviewed"   # unreviewed -> generic guard (§11.9)
    same_as: list[str] = []            # duplicated tools share one binding (target.get_pharmacogenomics)
    reads: dict[str, ReadSpec]         # "open_targets.known_drug": {access: full_table, columns: [...]}
    args: dict[str, ArgBinding]
    require_any: list[list[str]] = []
    exclusive: list[list[str]] = []    # at most one of each group
    result: ResultSpec
    serve: Literal["pass", "derived", "block"] = "pass"
    derived: DerivedSpec | None = None
    block: BlockSpec | None = None     # reason, alternatives, until_phase, hidden
    on_contradiction: Literal["derived", "tool_defect"] = "derived"   # derived only if `derived` is set
    witness: bool = True               # false: no inflation, no ranking refusal, total_method unknown (rev 2)
    leakage_filter: LeakageFilter | None = None   # rev 2: {arg: advanced_filter, template: "AREA[StudyFirstPostDate]RANGE[MIN,{ceiling}]"}
                                       # injected as "(<existing>) AND (<fragment>)": an OR in the value cannot escape it
    requires_fixed: list[RequiresFixed] = []   # {arg: value_filter, columns: [dataset_id], reason, alternatives}:
                                       # the SOMA filter must fix each column, else unsupported_combination
    defects: list[DefectSpec] = []     # documentation + detector tests; never drives code
    text: TextSpec = TextSpec()        # summary, drop_promises, notes
    hidden: bool = False

class ReadSpec(BaseModel):
    access: Literal["full_table", "bounded_scan", "projection", "remote", "upstream"] = "full_table"
    columns: list[str] = []            # paths (nested containers allowed); default: containers bound or returned
    when: dict[str, Any] | None = None # rev 2: argument-dependent reads ({include_indirect: true})
    est_row_bytes: int | None = None   # remote reads
    count_via: str | None = None       # remote: a count tool run first for count-first admission

class ArgBinding(BaseModel):
    binds: str | dict[str, str] | None = None   # column path; or {selector value: column} with a selector arg
    binds_any: list[str] = []          # rev 2: match on any of these columns (speciesA OR speciesB) -> Or
    role: Literal["filter", "limit", "free_text", "output_path", "projection", "flag", "selector",
                  "order_by", "order_direction", "anchor", "family_param", "universe_override",
                  "unbound"] = "filter"
    accepts: list[str] = []            # id_types (bare or "source:id_type") the resolver may use, in order
    op: Literal["eq", "in", "contains", "overlaps", "ge", "gt", "le", "lt", "ne", "range",
                "ge_abs", "gt_abs", "le_abs", "lt_abs", "nonempty", "has_kind"] = "eq"   # rev 2: abs, nonempty, has_kind, overlaps
    values: dict[str, Any] = {}        # selector: {coloc: open_targets.colocalisation_coloc, ecaviar: ...}; order_direction: {true: asc}
    match: Literal["exact", "casefold"] = "exact"   # selector and closed-value matching
    when_true: dict[str, Any] | None = None    # flag args: positive predicate ({in: [none_recorded]}); `ne` on codes is a lint error
    when_false: dict[str, Any] | None = None
    interpreted_as: Literal["exact", "regex", "substring", "casefold_substring", "engine"] = "exact"
    engine_doc: str | None = None      # interpreted_as engine: registry search semantics (synonyms, word match)
    escape: str | None = None          # rev 2: format plugin whose quote() escapes this value (essie, eutils, soma)
    forbid: list[str] = []             # characters that make the value invalid_argument
    pattern: str | None = None
    send_as: Literal["canonical", "label", "raw", "stored", "native_label", "alias"] = "canonical"
    send_map: dict[str, Any] = {}      # rev 2: {P: biological_process, F: molecular_function, C: cellular_component}
    snap: bool = True                  # numeric scope/category values snapped to the storage-typed vocabulary
    min: float | None = None; max: float | None = None
    each: bool = False                 # list argument: every element resolved (inferred from an array schema)
    max_items: int | None = None; min_items: int | None = None
    on_missing: Literal["error", "partial", "drop_disclosed"] = "error"   # list elements not found
    min_resolved_fraction: float | None = None   # gene lists: below it -> insufficient_resolution
    dedupe: bool = True
    default_disclosed: bool = True     # schema default materialised into args_sent and _vbt.scope
    gateway_only: bool = False         # x-gateway arg: consumed by the gateway, stripped before upstream
    wrap: str | None = None            # e.g. "({value})" for Essie fragments
    existence: Literal["universe", "bound", "upstream", "off"] = "universe"   # rev 2 (§11.5)
    universe: str | None = None        # override the id_type's identity universe for this binding
    universe_where: dict[str, Any] | None = None   # subset universe (isTherapeuticArea = true)
    qualified_by: list[str] = []       # arguments that qualify the key (study_id for sample_ids)
    item_filter: bool = False          # rev 2: filter nested items, not rows
    drop_empty_parents: bool = False   # rev 2: a parent left with no qualifying items is removed and counted
    family: Literal["exact", "include"] = "exact"   # rev 2: salt/parent family expansion (§11.5)
    limit_grain: str | None = None     # rev 2: the limit counts this grain (drugs), not rows
    limit_mode: Literal["per_group", "refuse"] = "per_group"   # rev 2: ranking within comparable groups

class ResultSpec(BaseModel):
    kind: Literal["rows", "record", "count", "file"] = "rows"   # rev 2
    rows: str | list[str] | None = "$" # JSONPath(s) to row lists; "$" + kind=record for one record
    rows_of: str | None = None         # rev 2: item table the rows are items of ("open_targets.target_go")
    record_when: str | None = None     # a payload test: a by-key reply is the one record at "$" (study_id)
    grain: str | None = None           # rev 2: the result's grain (a grains name); coarser than the key -> scope rule
    fields: dict[str, FieldMap] = {}   # rev 2: {"entity_id": {column: word}, "similarity": {computed: {...}}}
    parent_key: dict[str, str] = {}    # rev 2: {"id": "$.target_id"}: item rows carry the parent key; "/id" when
                                       # the items have an `id` of their own (an argument bound to the parent's
                                       # id is then checked on "/id", never on the item's id)
    row_key: list[str] | Literal["from_descriptor"] = "from_descriptor"   # JSONPaths allowed; $parent, $root
    key_from_args: dict[str, str] = {} # rev 2: {studyId: study_id}: key parts fixed by equality-bound args
    exists_when: str | None = None     # rev 2: per-row JSONPath predicate ($.patientId != null)
    on_unknown_items: Literal["not_found", "partial"] = "not_found"
    echo: dict[str, str | EchoSpec] = {}   # {"target_id": "$.id"} or {path, accept: [canonical, synonym:<col>, redirect], source: record}
    echo_set: EchoSet | None = None    # rev 2: {arg, path, mode: subset|equal, withheld_path}
    order: list[RankSpec] = []         # the intended ranking of rows
    order_from_arg: str | None = None  # rev 2: an order_by argument decides the order
    order_source: Literal["witness", "upstream_full_sort", "source_server_side"] = "witness"   # rev 2
    total: str | TotalSpec | None = None   # JSONPath, or {path, method: upstream, valid_unless: [...], partial_when: [...]}
    as_of: str | None = None           # rev 2: JSONPath of the record version date
    count_fields: list[str] | dict[str, str] = []   # recomputed; dict form maps nested counts: {"$.diseases[*].evidence_count": "len($.diseases[*].evidence)"}
    summary_fields: dict[str, Literal["drop"] | RecomputeSpec] = {}   # recompute: {agg, of, group_by}
    drop_fields: dict[str, str] = {}   # known-wrong fields -> reason (listed in _vbt.removed_fields)
    arg_echo: dict[str, str] = {}      # rev 2: {limit: $.limit}: echoed args restored to the requested values
    trim: dict[str, TrimSpec | int] = {}   # {path: {max, order: depth|key|declared}}
    sections: dict[str, SectionSpec] = {}  # multi-table results: {path, table, verb, single}
    levels: dict[str, list[str]] = {}  # rev 2: {patient: [OS_MONTHS, OS_STATUS, AGE]}: columns defined at a coarser level
    nested_errors: list[str] = []      # JSONPaths whose presence means a hidden failure
    not_found_when: list[str] = []     # JSONPath predicates meaning not found (evaluated first, §11.4)
    statistics: StatisticsSpec | None = None   # rev 2: enrichment results: test, correction, family, universe
    files: list[FileCheckSpec] = []    # returned artifacts to reconcile (phase 1 for Census outputs, rev 2)

class FieldMap(BaseModel):             # rev 2: result field -> descriptor column
    column: str | None = None          # "word", "tissues[].rna.value", "/id"
    computed: dict[str, Any] | None = None   # {statistic: cosine, from: vector, anchor: entity_id}
    placeholders: list[Any] = []; on_placeholder: Literal["null", "dangling_ref"] = "null"

class DerivedSpec(BaseModel):          # rev 2: fully specified
    verb: Literal["lookup", "find", "search", "members", "count", "aggregate", "similar", "expand", "enrich", "compare"]
    table: str                         # table or item table
    columns: list[str] = []            # projection; default: reads.columns; nested containers excluded unless named
    explode: list[str] = []            # multi-level, ancestor keys injected
    carry: list[str] = []              # parent fields copied onto item rows
    rename: dict[str, str] = {}        # descriptor path -> output field (keeps upstream's shape)
    split: dict[str, Any] | None = None    # {by_sign: log2FoldChange, into: {"+": $.up, "-": $.down}}; limit per list
    nest: dict[str, Any] | None = None     # {group_by: [disease], items: evidence, count_as: evidence_count, having: {min_arg: min_evidence}}
    group_by: list[str] = []; aggregate: dict[str, Any] = {}   # {gene_count: {count_distinct: gene}}
    sections: dict[str, SectionSpec] = {}  # multi-table derived results, each with its own status
    envelope: dict[str, Any] = {}      # static and echoed top-level fields ({success: true, pmid: "{pmid}"})
    compose: list["DerivedSpec"] = []  # two-step derivations (search sets, then count members)
```

`SectionSpec` is `{path, table, verb, single, key_from_args}`; a section on an `entity_detail` table
returns a list (never the first of many). `TotalSpec`, `EchoSpec`, `EchoSet`, `TrimSpec`,
`RecomputeSpec`, `StatisticsSpec`, `LeakageFilter` and `FileCheckSpec(path_from, must_exist,
echo_checks: {n_obs: $.n_cells}, key_columns, forbid_positional_index, write_once)` are small pydantic
models with the fields shown in the comments and examples.

### 8.2 Serve modes

- `pass`: upstream answers; every gateway mechanism applies (resolution, existence, witness checks,
  limit inflation, role transforms, header, provenance). A contradiction is repaired by `derived` if
  declared, otherwise it becomes `tool_defect`.
- `derived`: the data child answers with a generic verb driven by roles; upstream is not called. Rows
  are placed at `result.rows` with upstream field names (`rename`) so the shape stays familiar, and
  the data child's own count is the total (no second witness scan).
- `block`: `quarantined` error with reason and alternatives; `hidden: true` also removes it from the
  listing (only for no-op tools such as `clinicaltrials.clear_trial_cache`).

### 8.3 Why there is no quirk vocabulary in core

Upstream bugs are not encoded as executable "quirk kinds". The gateway applies the same generic
mechanisms to **every** bound tool: resolution, existence, witness counts and top-k, limit inflation
whenever a `limit` argument and an `order` exist, honour-every-bound-argument post-filters, unknown
and negation semantics, the scope-completeness rule, echo checks and key completeness. These follow
from bindings and roles, not from knowing the bug. A newly discovered class of upstream bug therefore
needs no core change: either the generic checks already catch it (witness contradiction), or the
overlay switches the tool to `serve: derived` (role-derived replacement) or `block`. The `defects`
list is documentation plus a detector test per entry (§21), which imports the unmodified upstream
function and fails when upstream fixes the bug, so the registry never goes stale.

### 8.4 Example: upstream `drug` overlay (excerpt)

```yaml
schema: vbt.overlay/1
server: drug
sources: [open_targets]
upstream_commit: ${vars.upstream_commit}
tools:
  search_known_drugs:
    reads: {open_targets.known_drug: {access: full_table}}
    args:
      target_id:  {binds: open_targets.known_drug.targetId, accepts: [ensembl_gene, hgnc_symbol]}
      disease_id: {binds: open_targets.known_drug.diseaseId, accepts: [ot_disease, disease_name]}
      min_phase:  {binds: open_targets.known_drug.phase, op: ge, min: 0, max: 4}
      limit:      {role: limit, min: 1, max: 200}
    require_any: [[target_id, disease_id]]
    result:
      rows: $.drugs
      order: [{column: phase, direction: desc, nulls: last}, {column: drugId, direction: asc}]
      count_fields: [$.count]
      arg_echo: {limit: $.limit}
    serve: pass
    defects:
      - {id: OT-DRUG-003, what: "head(limit) before sort_values('phase')", where: "drug_mcp/tools.py:590,601",
         test: test_dl_defect_detectors.py::test_ot_drug_003, effect: silent_wrong}
      - {id: OT-DRUG-004, what: "null-phase rows dropped even at min_phase=0", where: "drug_mcp/tools.py:583-588",
         test: test_dl_defect_detectors.py::test_ot_drug_004, effect: silent_wrong}

  get_pharmacogenomics:
    same_as: [target.get_pharmacogenomics]
    reads: {open_targets.pharmacogenomics: {access: bounded_scan}}
    args:
      target_id: {binds: open_targets.pharmacogenomics.targetFromSourceId, accepts: [ensembl_gene, hgnc_symbol]}
      drug_id:   {binds: "open_targets.pharmacogenomics.drugs[].drugId", op: contains,
                  accepts: [chembl_molecule, drug_name], existence: bound}
      limit:     {role: limit, min: 1, max: 500}
    require_any: [[target_id, drug_id]]
    result: {rows: $.pgx_relationships, order: [{column: evidenceLevel, statistic: ordinal}]}
    serve: derived
    derived: {verb: find, table: open_targets.pharmacogenomics}     # both args ANDed
    defects:
      - {id: OT-DRUG-009, what: "isinstance(list) on ndarray + key 'id' vs 'drugId': drug path always empty",
         where: "drug_mcp/tools.py:649-657", test: test_dl_defect_detectors.py::test_ot_drug_009}
      - {id: OT-DRUG-010, what: "drug_id ignored when target_id given (if/elif): rows violating drug_id returned",
         where: "drug_mcp/tools.py:645-647"}

  get_drug_info:
    reads: {open_targets.drug_molecule: {access: full_table}}
    args: {drug_id: {binds: open_targets.drug_molecule.id, accepts: [chembl_molecule, drug_name], family: exact}}
    result: {kind: record, rows: $, echo: {drug_id: {path: $.id, source: record}}}
    text: {drop_promises: [mechanismOfAction]}       # the docstring promises a column absent in 25.09
```

`pharmacogenomics.evidenceLevel` is declared `{role: measure, statistic: ordinal, levels: ["1A", "1B",
"2A", "2B", "3", "4"]}`, which the ordinal plugin uses for both sorting and thresholds.

### 8.5 Example: `functional_genomics` Tahoe tools (composite key)

```yaml
  query_drug_perturbation:
    reads: {tahoe_100m.de_permissive: {access: bounded_scan}, tahoe_100m.drug_metadata: {access: full_table},
            tahoe_100m.cell_line_metadata: {access: full_table}}
    args:
      drug_name:      {binds: tahoe_100m.de_permissive.drug, accepts: [tahoe_drug], send_as: stored}
      cell_line_id:   {binds: tahoe_100m.de_permissive.Cell_ID_DepMap, accepts: [depmap_cell_line]}
      concentration:  {binds: tahoe_100m.de_permissive.concentration, gateway_only: true}   # auto-derived x-gateway
      plate:          {binds: tahoe_100m.de_permissive.plate, gateway_only: true}           # replicate: listed if absent
      max_padj:       {binds: tahoe_100m.de_permissive.padj, op: le, max: 0.10}             # from the constraint
      min_abs_log2fc: {binds: tahoe_100m.de_permissive.log2FoldChange, op: gt_abs, min: 0}  # upstream uses strict >
      top_n:          {role: limit, min: 1, max: 500, limit_grain: gene}
    result:
      rows: [$.top_upregulated, $.top_downregulated]
      grain: gene                                   # one row per gene within one contrast
      summary_fields: {$.num_total_significant: {recompute: {agg: count_distinct, of: gene_name}},
                       $.num_cell_lines_tested: {recompute: {agg: count_distinct, of: Cell_ID_DepMap}}}
    serve: derived
    derived:
      verb: find
      table: tahoe_100m.de_permissive
      split: {by_sign: log2FoldChange, into: {"+": $.top_upregulated, "-": $.top_downregulated}}   # top_n per list
      rename: {gene_name: gene, drug: drug_name, Cell_ID_DepMap: cell_line, n_cells_trt: n_cells_treatment}
      sections:
        drug_info:      {path: $.drug_info, table: tahoe_100m.drug_metadata, verb: lookup, single: true, key_from_args: {drug: drug_name}}
        cell_line_info: {path: $.cell_line_info, table: tahoe_100m.cell_line_metadata, verb: lookup, key_from_args: {Cell_ID_DepMap: cell_line_id}}

  find_drugs_affecting_gene:
    reads: {tahoe_100m.de_permissive: {access: bounded_scan}, tahoe_100m.drug_metadata: {access: full_table}}
    args:
      gene_name:        {binds: tahoe_100m.de_permissive.gene_name, accepts: [tahoe_gene, "open_targets:ensembl_gene", hgnc_symbol], send_as: stored}
      cell_line_filter: {binds: tahoe_100m.de_permissive.Cell_ID_DepMap, accepts: [depmap_cell_line]}
      min_abs_log2fc:   {binds: tahoe_100m.de_permissive.log2FoldChange, op: gt_abs, min: 0}
      max_padj:         {binds: tahoe_100m.de_permissive.padj, op: le, max: 0.10}
      top_n:            {role: limit, min: 1, max: 500, limit_grain: drug}   # each drug's best contrast shown
    result: {rows: [$.top_upregulators, $.top_downregulators], grain: row,
             order: [{column: log2FoldChange, direction: desc_abs, nulls: last}]}
    serve: derived
    derived: {verb: find, table: tahoe_100m.de_permissive, split: {by_sign: log2FoldChange, into: {"+": $.top_upregulators, "-": $.top_downregulators}}}
```

For `query_drug_perturbation` the result grain is one gene within one contrast, so when
`concentration` is not passed and the rows for the drug and cell line span three doses, the
scope-completeness rule raises `incomplete_key` with `values: [0.05, 0.5, 5.0]` and `unit: uM` (I7)
before any scan of rows. `plate` has `pooling: list`, so contrasts on several plates (157 of 1,138
drug-dose pairs, VERIFIED) are listed with their plate rather than refused. For
`find_drugs_affecting_gene` each row is a complete contrast ranked by |log2FC|, which is comparable
across contrasts, so no scope dimension is merged and the call is answered; the gene predicate is
pruned through the `gene_name` sidecar index (§11.6), not a full decode.

### 8.6 Example: third-party HTTP server, and the generic overlay

```yaml
schema: vbt.overlay/1
server: uniprot                     # name in configs/mcp_servers.yaml (url: ...)
match: {url_prefix: "https://mcp.example.org/uniprot"}
sources: [uniprot_rest]             # optional remote descriptor (release via its /release endpoint)
tools:
  get_entry:
    args: {accession: {binds: uniprot_rest.entry.primaryAccession, accepts: [uniprot_accession, hgnc_symbol]}}
    result:
      kind: record
      rows: $
      echo: {accession: {path: $.primaryAccession, accept: [canonical, "synonym:secondaryAccessions"]}}
      not_found_when: ["$.status == 404", "$.message =~ '(?i)not found'"]
  search:
    args: {query: {role: free_text, interpreted_as: engine, escape: lucene}, size: {role: limit, min: 1, max: 500}}
    result: {rows: $.results, total: {path: $.totalCount, method: upstream}, row_key: [primaryAccession],
             order_source: source_server_side}
```

`configs/data/overlays/_generic.yaml` applies to any server or tool without a reviewed binding (§11.9).

### 8.7 Examples for remote, clinical and vector tools (rev 2)

```yaml
# clinicaltrials overlay (excerpt): cBioPortal clinical data and ClinicalTrials.gov with a leakage ceiling
  get_clinical_data:
    reads: {cbioportal.sample: {access: upstream}, cbioportal.patient_clinical: {access: upstream},
            cbioportal.sample_clinical: {access: upstream}}
    args:
      study_id:   {binds: cbioportal.sample.studyId, accepts: [cbio_study], existence: upstream}
      sample_ids: {binds: cbioportal.sample.sampleId, op: in, accepts: [cbio_sample], each: true, min_items: 1,
                   qualified_by: [study_id], existence: off}
    result:
      rows: $.data
      row_key: [sampleId]
      key_from_args: {studyId: study_id}
      exists_when: "$.patientId != null"          # phantom samples come back with patientId null (VERIFIED)
      levels: {patient: [OS_MONTHS, OS_STATUS, DFS_MONTHS, DFS_STATUS, PFS_MONTHS, PFS_STATUS, AGE, SEX]}
      count_fields: [$.sample_count]
      not_found_when: ["$.error =~ '(?i)status code: 404'"]
    serve: pass
  search_clinical_trials:
    reads: {clinicaltrials_gov.studies: {access: upstream}}
    args:
      condition: {role: free_text, interpreted_as: engine, escape: essie,
                  engine_doc: "registry search: synonym expansion and word match ('Korea' matches both Koreas)"}
      eligibility_text: {role: free_text, interpreted_as: engine, escape: essie, forbid: ['"']}
      country:   {binds: "clinicaltrials_gov.studies.protocolSection.contactsLocationsModule.locations[].country",
                  op: contains, interpreted_as: engine, default_disclosed: true}
      phase:     {binds: "clinicaltrials_gov.studies.protocolSection.designModule.phases[]", op: overlaps}
      advanced_filter: {role: free_text, wrap: "({value})", escape: essie}
      sort:      {role: order_by, values: {"LastUpdatePostDate:desc": {column: protocolSection.statusModule.lastUpdatePostDateStruct.date, direction: desc}}}
    leakage_filter: {arg: advanced_filter, template: "AREA[StudyFirstPostDate]RANGE[MIN,{ceiling}]"}
    result:
      rows: $.trials
      row_key: [nctId]
      fields: {nctId: {column: protocolSection.identificationModule.nctId},
               startDate: {column: protocolSection.statusModule.startDateStruct.date}}
      total: {path: $.total_count, method: upstream, partial_when: ["$.warning"]}
      order_from_arg: sort
      order_source: source_server_side

# association overlay (excerpt): embedding similarity with an anchor
  find_similar_entities:
    reads: {open_targets.literature_vector: {access: full_table, columns: [word, category, norm, vector]}}
    args:
      entity_id: {role: anchor, binds: open_targets.literature_vector.word,
                  accepts: [literature_word, ensembl_gene, hgnc_symbol, ot_disease, disease_name, chembl_molecule, drug_name]}
      category:  {binds: open_targets.literature_vector.category, default_disclosed: true}
      top_k:     {role: limit, min: 1, max: 500}
    result:
      rows: $.similar_entities
      fields: {entity_id: {column: word}, category: {column: category}, norm: {column: norm},
               similarity: {computed: {statistic: cosine, from: vector, anchor: entity_id}}}
      echo: {entity_id: $.query_entity_id}
      order: [{column: similarity, direction: desc, nulls: last}]
      order_source: upstream_full_sort            # detector test proves upstream sorts all candidates first
    serve: pass
    on_contradiction: tool_defect                 # phase 2 (F14): derived: {verb: similar, table: open_targets.literature_vector}
```

The anchor is resolved and checked for existence in the bound column: present → the call proceeds and
the anchor row is excluded from candidates and from `_vbt.total`; absent from the column but present in
an accepted id_type's universe → `empty` with `_vbt.undefined: {argument: entity_id, reason:
not_in_table}` and the table's coverage; absent everywhere → `not_found`. An anchor is never a row
predicate, so the witness counts candidates under the other bound filters (`category`), and
`compute_entity_similarity` binds two anchors without producing the impossible predicate
`word == a AND word == b`.

---

## 9. Plugins

### 9.1 Kinds and registry

```python
# src/vbt/datalayer/plugins/__init__.py
KINDS: dict[str, type] = {"format": FormatPlugin, "layout": LayoutPlugin,
                          "statistic": StatisticPlugin, "identifier": IdentifierPlugin}
# phase 4 adds "envelope": EnvelopePlugin (the worked example of adding a kind)
HARNESS_KINDS: dict[str, type] = {"acquisition": AcquisitionPlugin}   # harness side only
```

`acquisition` is a harness-side kind: the transports of `vbt data acquire` (`http`, `huggingface`, `s3`,
`gcs`, `json_index`, `zip_member`; [DATA_SETUP.md](DATA_SETUP.md)), which list a release's files with their
sizes and checksums and read them with byte ranges. `discover_harness(settings)` finds them the same way
(builtins in `vbt.datalayer.plugins.acquisition`, entry points in `vbt.datalayer.acquisition`,
`data.plugins.paths`, `disabled`, `override`); they are not part of the data child's registry. A
descriptor's `acquisition` section names the transport and the files of each table. Its conformance suite
(A-1 to A-7) runs with the others: `vbt ds conformance [--kind K] [--plugin NAME]` covers both registries.

`PluginRegistry.discover(settings)` loads, in order: in-tree builtins (`vbt.datalayer.plugins.<kind>s.*`
modules decorated with `@register`), Python entry points in group `vbt.datalayer.<kind>`, and modules
listed in `data.plugins.paths`. Each plugin declares `kind`, `name`, `version`, `api == API_VERSION`,
`capabilities`, `requires` (importable modules, probed by preflight; a missing module makes tables that
use the plugin `plugin_unavailable` with an install hint) and `conformance_cases()`. Name collisions
are an error unless `data.plugins.override` names the winner. Builtins are conformance-tested in CI,
not gated at runtime; third-party plugins can be gated with `data.plugins.require_conformance: true`,
which checks a stamp written by `vbt datasource conformance --plugin <name>` (module digest + suite
version, under `data.cache_dir/conformance/`; `vbt ds conformance --list` shows each plugin's stamp).

**Capabilities (rev 2).** Protocol methods that only some plugins implement are **optional
capabilities declared in phase 1**: a plugin lists the capability, the conformance suite runs only
the cases for declared capabilities, and the core calls the method only when the capability is
present. Later phases therefore add plugins, never protocol methods (G3). Capabilities: format
`tabular`, `pushdown`, `stats`, `stats_scan` (no footers: a bounded pass computes stats once per
fingerprint), `nested`, `leaf_projection`, `matrix`, `vectors`, `string_compile`; layout `scan`,
`count`, `live` (remote requests), `upstream_only`; statistic `test` (set tests), `paired` (time and
event pairs), `veto_labels`.

A third-party package registers like this:

```toml
[project.entry-points."vbt.datalayer.format"]
bigwig = "vbt_bigwig:BigWigFormat"
```

### 9.2 Protocols (`src/vbt/datalayer/plugins/base.py`)

```python
API_VERSION = 1

@dataclass(frozen=True)
class Fragment:
    uri: str; size: int | None; mtime_ns: int | None
    partition: Mapping[str, Any] = field(default_factory=dict)
    fragment_key: str | None = None                 # rev 2: e.g. "GSE12251" from TableSpec.fragment_key
    sha256: str | None = None                       # from a manifest when known

@dataclass(frozen=True)
class ColumnStats:
    uncompressed_bytes: int; null_count: int | None
    num_values: int | None = None                   # rev 2: leaf values incl. nested items (object-overhead estimate)
    max_rep_level: int = 0; max_def_level: int = 0  # rev 2
    min: Any = None; max: Any = None
    storage_type: str | None = None                 # rev 2: "float", "large_string", "int32", ...
    kind: Literal["flat", "string", "nested", "dense_matrix", "sparse_matrix"] = "flat"

@dataclass(frozen=True)
class FragmentStats:
    rows: int | None                                # rev 2: None when unknown without a scan (OBO, text)
    row_groups: int; columns: Mapping[str, ColumnStats]
    method: Literal["footer", "scan", "size_only"] = "footer"
    shape: tuple[int, int] | None = None            # matrices: (n_rows, n_cols); nnz in columns["X"].num_values

class FormatPlugin(Protocol):
    kind: ClassVar[str] = "format"; name: ClassVar[str]; version: ClassVar[str]
    capabilities: ClassVar[frozenset[str]]
    requires: ClassVar[tuple[str, ...]] = ()        # rev 2: ("h5py", "anndata", "scipy")
    def logical_schema(self, frag: Fragment) -> "pa.Schema": ...      # columns as descriptors see them (rev 2)
    def leaf_path(self, path: str, schema: "pa.Schema") -> str: ...   # §6.4 path -> path_in_schema (rev 2)
    def stats(self, frag: Fragment) -> FragmentStats: ...             # footer/header only, or a cached scan (stats_scan)
    def metadata(self, frag: Fragment) -> Mapping[str, str]: ...      # rev 2: header fields (OBO data-version)
    def scan(self, frags: Sequence[Fragment], *, columns: list[str], predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None) -> Iterator["pa.RecordBatch"]: ...
    def compile(self, predicate: Predicate, schema: "pa.Schema") -> tuple[Any, Predicate | None]: ...
        # rev 2: returns (pushdown, residual). Literals are cast to the column's storage type
        # (pa.scalar(v, field.type)); NaN compiles as null for measure/count/time/flag columns;
        # Contains/Any/All over lists are always residual. String-compiling formats (SOMA, Essie) quote here.
    def to_native(self, table: "pa.Table") -> list[dict[str, Any]]: ...   # list->list, struct->dict, NaN->None
    # capability leaf_projection (rev 2): read only these leaves, per fragment and row group
    def read_leaves(self, frag: Fragment, leaves: list[str], row_groups: Sequence[int] | None) -> "pa.Table": ...
    # capability matrix (rev 2)
    def axis_values(self, frag: Fragment, axis: str) -> "pa.Table": ...
    def slice(self, frag: Fragment, value: str, *, row_predicate: Predicate | None, col_keys: Sequence[Any] | None,
              budget_bytes: int) -> Iterator["pa.RecordBatch"]: ...   # long-view batches (row keys, col keys, value)
    @classmethod
    def conformance_cases(cls) -> "FormatCases": ...

class LayoutPlugin(Protocol):
    kind: ClassVar[str] = "layout"; name: ClassVar[str]; version: ClassVar[str]
    capabilities: ClassVar[frozenset[str]]
    def fragments(self, root: str, spec: LayoutSpec) -> list[Fragment]: ...
        # lists by NAME (*.parquet etc.); excludes *.part, *.part.json, dotfiles, _SUCCESS; never skips a
        # listed file because it is unreadable (no pyarrow exclude_invalid_files) — rev 2, I14
    def partition_columns(self, spec: LayoutSpec) -> dict[str, str]: ...    # name -> declared type
    def signature(self, root: str, spec: LayoutSpec) -> str: ...      # stat-only, stdlib only (harness side)
    def fingerprint(self, frags: list[Fragment], manifest: "Manifest | None") -> str: ...
    def partition_fingerprints(self, frags: list[Fragment], manifest: "Manifest | None") -> dict[str, str]: ...  # rev 2
    def probe(self, root: str, spec: LayoutSpec, manifest: "Manifest | None") -> list[CheckItem]: ...
    def as_of(self, root: str, spec: LayoutSpec) -> str | None: ...
    # capability live (rev 2; plugins arrive in phase 4)
    def request(self, spec: LayoutSpec, *, predicate: Predicate | None, projection: list[str],
                page_token: str | None, budget: "RemoteBudget") -> "Page": ...   # Page(rows, total, next, as_of)

class StatisticPlugin(Protocol):
    kind: ClassVar[str] = "statistic"; name: ClassVar[str]; version: ClassVar[str]
    capabilities: ClassVar[frozenset[str]] = frozenset()
    def sort_key(self, column: str, spec: MeasureSpec, direction: str | None) -> RankKey: ...  # nulls last
    def predicate(self, column: str, op: str, value: Any, spec: MeasureSpec,
                  confirmed: ConfirmedFacts | None, fixed_scope: Mapping[str, Any]) -> Predicate: ...
        # raises UnsupportedFilter (I9, scale; and a threshold on a comparable_within measure whose
        # group is not fixed by fixed_scope — rev 2)
    def bounds(self, spec: MeasureSpec, constraints: Sequence[ConstraintSpec]) -> dict[str, Any]: ...
    def validate(self, stats: ColumnStats, snapshot: "ValueSnapshot | None", spec: MeasureSpec) -> ConfirmResult: ...
        # rev 2: encodings confirmed from a distinct-value snapshot (every code observed, values ⊆ codes ∪ {null},
        # NaN counted), scales from min/max
    def aggregate(self, values: Sequence[float | None], how: str, spec: MeasureSpec,
                  keys: Sequence[Any] | None = None) -> AggResult: ...   # empty -> None; deduplicates on level keys
    def comparable(self, a: Mapping[str, Any], b: Mapping[str, Any], spec: MeasureSpec) -> bool: ...
    def describe(self, spec: MeasureSpec) -> str: ...   # names direction, scale, cutoff and its inclusivity
    def family(self, spec: MeasureSpec) -> FamilySpec | None: ...
    # capability test (rev 2; hypergeometric and Fisher plugins arrive in phase 3)
    def test(self, overlap: int, set_n: int, query_n: int, universe_n: int, spec: MeasureSpec) -> float: ...
    # capability paired (rev 2): time + event (survival)
    def aggregate_pair(self, times: Sequence[float | None], events: Sequence[bool | None], how: str,
                       spec: MeasureSpec) -> AggResult: ...
    # capability veto_labels (rev 2): sibling string fields that bin this statistic (cosine "interpretation")
    def vetoed_companions(self, spec: MeasureSpec) -> tuple[str, ...]: ...

@dataclass(frozen=True)
class Normalized: value: str; steps: tuple[str, ...]      # steps from the I-5 whitelist
@dataclass(frozen=True)
class Rejected: reason: str; looks_like: tuple[str, ...] = ()

class IdentifierPlugin(Protocol):
    kind: ClassVar[str] = "identifier"; name: ClassVar[str]; version: ClassVar[str]
    id_type: ClassVar[str]                           # "ensembl_gene"
    canonical: ClassVar[str]                         # regex of the canonical form (may depend on options)
    examples: ClassVar[tuple[str, ...]]
    cardinality: ClassVar[Literal["one", "many"]] = "one"
    overlaps: ClassVar[frozenset[str]] = frozenset() # kinds that legitimately share syntax
    def configure(self, options: Mapping[str, Any], universe_sample: Sequence[str] | None) -> "IdentifierPlugin": ...
        # rev 2: YAML options (prefixes: from_universe, canonical for local_key, input_requires_prefix)
    def normalize(self, raw: str) -> Normalized | Rejected: ...        # agent input; syntactic only
    def normalize_stored(self, stored: str) -> Normalized | Rejected: ...   # rev 2: stored values (index build)
    def looks_like(self, raw: str) -> float: ...                       # 0..1, for wrong-kind diagnostics
    def label_key(self, raw: str) -> str: ...                          # NFKC + casefold key for label lookup
    def describe(self) -> str: ...                                     # <= 200 chars, includes an example
```

`Predicate` (`src/vbt/datalayer/predicate.py`) is a small JSON-serialisable IR: `Eq`, `In`, `Cmp`,
`CmpAbs(column, op, value)` (rev 2), `Range`, `Contains(path, value)`, `Any(path, predicate)` and
`All(path, predicate)` (rev 2; the inner predicate is evaluated per item with item-relative paths and
nests for multi-level paths), `NonEmpty(path)` and `KindMatch(path, id_type)` (rev 2), `CensoredCmp(time,
event, op, value)` (rev 2), `TextMatch(column, text, mode)` (exact, casefold, literal substring, word),
`IsNull`, `Not`, `And`, `Or`, `Param(name)` (a call-time parameter inside universe predicates), plus
`RankKey(column, direction, nulls, statistic, within)`. It uses **three-valued (Kleene) logic**:

- a comparison with null, NaN or a declared unknown code is unknown; `Not(unknown)` is unknown; a row
  whose predicate is unknown is excluded and counted in `excluded_unknown` (I6);
- `Any([]) = false`, `Any(null list) = unknown`, `Any` with no true item and at least one unknown item
  is unknown (null items count as unknown unless the container declares `null_items: skip`);
  `All([]) = true`, `All(null) = unknown`;
- several bindings on the same container compile into **one** `Any` so their conditions apply to the
  same item (rev 2, C21);
- `CensoredCmp(time, event, "<=", t)` is true when `time <= t` and the event occurred, false when
  `time > t`, and unknown when `time <= t` and the case is censored;
- `CmpAbs` compares `|column|` (upstream's strict `>` is kept by binding `gt_abs`).

Each format plugin compiles it with correct quoting and storage-typed literals, which fixes the SOMA
quote-joining (`single_cell_mcp/tools.py:458-462`), Essie parenthesisation
(`clinicaltrials_mcp/tools.py:768-772`) and float32 equality (C20) bug classes once for all sources.
Pushdown never changes the result: anything a format cannot push down is returned as the residual and
evaluated by the reader with `predicate.evaluate` semantics.

### 9.3 Conformance suites (`src/vbt/datalayer/plugins/conformance/`)

Each suite is a parametrized pytest module over `registry.all(kind)`. A new plugin is tested without
editing any test. External packages import the same suites. Golden cases are **tagged by capability**,
and a plugin runs only the cases for capabilities it declares (rev 2: an h5ad plugin is not failed by
list-of-struct goldens that AnnData obs cannot hold).

**Identifier suite**
- I-1 every `(raw, expected)` case from `conformance_cases()` holds.
- I-2 `normalize` and `normalize_stored` are idempotent.
- I-3 canonical examples pass unchanged and every normalised output matches `canonical`.
- I-4 **confusion matrix** over all registered identifier plugins: for each ordered pair (A, B) with
  B ∉ A.overlaps, `A.normalize(e)` is `Rejected` for every example e of B. Catches `PMC1234` accepted
  as PMID 1234, symbols accepted as Ensembl IDs, `HP:` vs `HP_` collisions. Declared overlaps (rev 2):
  `ot_disease` ↔ `go`, `hpo`, `uberon` (OT disease IDs include `GO_`, `HP_` and `UBERON_` terms);
  `ncbi_gene` ↔ `pmid` only for stored values (agent input requires a prefix).
- I-5 normalisation steps come from the whitelist (`strip`, `upper`, `lower`, `strip_version`,
  `curie_colon_to_underscore`, `curie_underscore_to_colon`, `strip_chr`, `separator_to_underscore`,
  `strip_prefix`, `canonical_prefix_case` (rev 2: `orphanet_558` → `Orphanet_558`), `strip_suffix`
  with a declared qualifier (rev 2: `_PAR_Y`)); a step such as `extract_digits` fails.
- I-6 `describe()` is non-empty, at most 200 characters and contains an example.
- I-7 (rev 2) `configure(options, universe_sample)` with `prefixes: from_universe` accepts every sampled
  universe key and still rejects the other kinds' examples.

**Format suite** (golden tables in `golden.py`: unicode string IDs, int32 and int64, float32 and
float64 with NaN and null, bool with null, `list<string>`, `large_list`/`large_string` variants,
`list<struct<drugId:string,x:double>>`, a three-level `list<struct<list<struct<list<struct>>>>>`
(target_essentiality shape), a null struct parent versus a null child field, a null list versus `[]`
versus a list of null items, `struct<rows:list<string>, count:int64>`, struct-of-lists like
`disease.synonyms`, `list<list<string>>`, timestamp; 0-row and 1M-row variants; matrix goldens with
header-encoded IDs, NaN cells, a duplicate header, an all-empty column, a row-boundary truncation and
a 2k × 20k wide case)
- F-1 round trip: `to_native(scan(write(golden)))` equals golden; lists come back as `list`, structs as
  `dict`, NaN and null as `None`; null and empty values round-trip at every definition level; any
  `numpy.ndarray` in output fails.
- F-2 projection decodes only requested columns; with `leaf_projection`, allocation is proportional to
  the projected leaves, including leaves inside lists (verified separately per level).
- F-3 pushdown equivalence: `scan(where=p) == filter(scan(), p)` for 50 seeded random predicates,
  including nulls, NaN, `CmpAbs`, and correlated `Any` over the three-level golden; null never satisfies.
- F-4 nested membership (`Contains`, `Any`, `All` on one, two and three levels; empty, null and
  null-item lists) equals the `predicate.evaluate` oracle.
- F-5 `stats().rows` equals scanned rows when not None; `stats_scan` plugins cache the scan per
  fingerprint.
- F-6 bounded top-k over the 1M-row variant keeps peak Arrow allocation under
  `k × row_bytes × 4 + 2 × batch_bytes` (the `scan_top_rows` property).
- F-7 top-k equals a full sort-then-head oracle, ties broken by the canonical key (nullable and
  nested key parts included), nulls last.
- F-8 corrupt, truncated or zero-length files raise `FormatError`, never an empty table.
- F-9 `compile()` of string literals containing `'`, `"`, `(`, `\` and unicode matches exactly those
  literals (injection safety for string-compiling formats).
- F-10 (rev 2) equality and `In` on float32 key columns: `Eq(concentration, 0.05)` pushdown equals the
  oracle and renders as `0.05`.
- F-11 (rev 2) `leaf_path` maps every golden path, including legacy list encodings, to `path_in_schema`.
- F-12 (rev 2, capability `matrix`) long-view round trip against a dense oracle; implicit-zero policy;
  axis exposure under the declared name; categorical code −1 → null while the string `'nan'` stays a
  string; positional-index detection; gzip-chunked CSR inputs; reading by position after a duplicate
  header.

**Layout suite**
- L-1 `fragments()` lists exactly the data files by name and excludes `*.part`, `*.part.json`,
  `_SUCCESS` and dotfiles (VERIFIED upstream failure: one stray `*.part` breaks `ds.dataset`).
- L-2 hive partition columns are restored with the declared type, and the reader's pruning of
  partition-only conjuncts (three-valued, `__HIVE_DEFAULT_PARTITION__` as null) reads only matching
  partitions.
- L-3 an empty or missing location yields a not-ready finding, never an empty table (layouts with
  capability `upstream_only` are exempt and report "served by upstream tools only").
- L-4 `fingerprint` is stable and changes on byte change, file add, remove and rename; partition
  fingerprints change only for the partition touched.
- L-5 `signature` uses only `os.stat` (asserted by patching `open`) and changes when a file is added,
  removed or resized.
- L-6 live layouts return a non-null `as_of`.
- L-7 (rev 2) a truncated `.parquet` shard in a listed fragment gives `FormatError` and a not-ready
  partition, never fewer rows (VERIFIED: `exclude_invalid_files` silently returned 4,800 of 22,521).

**Statistic suite**
- S-1 `sort_key` gives a total, deterministic order: shuffled inputs give the same order; nulls last.
- S-2 `predicate` never passes unknown, for every op, including in-band codes (`missing_values`),
  `unknown_when` conditions and NaN; boundary values are checked against the declared cutoff
  inclusivity.
- S-3 thresholds outside `scale` raise `UnsupportedFilter`; unconfirmed `verified: false` encodings
  raise `UnsupportedFilter` rather than guessing.
- S-4 `aggregate([])` and `aggregate([None])` return `None`, never 0; with level keys, duplicates are
  counted once.
- S-5 `comparable()` is false across `comparable_within` groups, and aggregation across groups raises.
- S-6 `describe()` names direction and scale.
- S-7 (phase 3) `family`-based BH matches a reference implementation using the full declared family m
  (including zero-overlap members).
- S-8 (rev 2) a threshold on a `comparable_within` measure with the group not fixed raises
  `UnsupportedFilter`.
- S-9 (rev 2, capability `test`/`paired`) set tests match scipy on hand-computed cases; paired
  survival aggregation matches a reference Kaplan-Meier implementation and refuses means of censored
  times.

### 9.4 Builtin plugins by phase

| Kind | Phase 1 | Later |
|---|---|---|
| identifier | `ensembl_gene` (case, whitespace, `.N` version), `ensembl_gene_any`, `ensembl_gene_mouse`, `hgnc_symbol` (label kind), `ot_disease` (prefixes from the universe via options, so OBA, GO, MP, NCIT, OBI, GSSO, PATO, OGMS and UBERON IDs are accepted; `:`→`_`; IRIs; `canonical_prefix_case`), `disease_name`, `chembl_molecule` (case; parent/salt families), `drug_name`, `inchikey`, `go` (`GO_`→`GO:`), `reactome` (strip version), `so`, `hpo` (canonical `HP_\d{7}` for OT tables, accepts `HP:`), `pmid` (digits only; rejects PMCID, DOI, `n.v`), `pmcid` and `doi` (validation only, so `PMC1234` is recognised and rejected as a PMID), `europepmc_ppr`, `nct_id` (strict `NCT\d{8}`), `ot_variant` (`chr` strip, `:`/`-`→`_`), `chromosome`, `rsid` (cardinality many), `study_locus_id`, `gwas_study`, `depmap_cell_line` (ACH-; punctuation-insensitive exact name), `tahoe_drug` (whitespace, case, salt suffix → candidates; `normalize_stored` strips trailing whitespace), `tahoe_gene_name` (symbol or bare ENSG), `cbio_cancer_type` (lower case), `cbio_study`, `cbio_sample`, `cbio_patient` (syntax only), `uniprot_accession`, `ncbi_taxon` (int and string forms), `cellosaurus`, `local_key` (YAML-configured canonical regex and steps for opaque keys: chemical probes, MSigDB set names, PharmGKB IDs; I-4 treats it as accepting only its declared pattern), `ot_entity_any` (union kind) | `ncbi_gene` (prefix required for agent input), `geo_gsm`, `census_joinid`, `cell_barcode`, `cell_ontology`/`uberon` (wrapping `src/vbt/analysis/ontology.py`) (P2); `census_feature` (P4) |
| format | `parquet` (pushdown, nested, footer stats, `leaf_projection`, `large_*` and int32 types) | `csv`/`tsv` (`stats_scan`, header parse, `matrix`, GCT), `jsonl`, `h5ad` (backed; `matrix`; index exposed under its `_index` name; gzip CSR), `zarr`, `obo` (published projection schema; `metadata()` reads `data-version`), `gmt` (`logical_schema` set_id, description, members), `npy`/`safetensors` embeddings with `ids_from` (P2); `soma`, `rest_json` (P4) |
| layout | `single_file` (a glob gives one fragment per file), `sharded_dir`, `hive` (partitions parsed from directory names; no `exclude_invalid_files`), `upstream_only` (remote sources served only by their upstream tools in phase 1: no fragments, as-of = call time unless the result carries one) | `zip_member` (Zenodo archive in place), `http_range` (P2); `live_api`, `soma` (P4) |
| statistic | `numeric`, `score_0_1`, `score_0_100`, `percent`, `ordinal` (with `levels`), `signed_factor` (OT prioritisation), `clinical_phase` (OT 0–4 vs normalised 0–1 decided by confirmed range; ChEMBL −1 = unknown via `missing_values`), `pvalue_mantissa_exponent` (composite `columns`), `llr_critval`, `count` | `fdr_bh` (declared family), `log2fc`, `std_error`, `wald`, `effect_beta`, `gene_effect` (declared cutoff and inclusivity), `clpp`, `coloc_h4`, `tpm` (unit guard), `log2_intensity`, `cosine` (`veto_labels`), `hypergeom_enrichment` and `fisher` (`test`), `survival_time` (`paired`) (P3) |

### 9.5 Adding a plugin versus adding a kind

- **New dataset**: YAML descriptor (+ overlay if served by an MCP server), `vbt datasource lint`,
  `vbt datasource check`. No Python. Opaque or dataset-specific identifier syntax is a `local_key`
  id_type with YAML options, not a new plugin (rev 2).
- **New format or layout** (e.g. BigWig): implement the protocol (and any optional capabilities),
  register the entry point, pass the suite (`vbt datasource conformance --plugin bigwig`). No core
  file changes.
- **New kind** (the only core change): add one entry to `KINDS`, a protocol in `plugins/base.py`, a
  conformance suite module, and the descriptor field that names it. Phase 4 does this for `envelope`
  (result-shape decoders for third-party servers that JSONPath cannot describe).

---

## 10. Derivations

All derivations are pure functions in `src/vbt/datalayer/derive/` (harness side) or
`gateway/readiness.py` and `memory/estimate.py`, snapshot-tested with golden files.

### 10.1 Argument JSON schema (`derive/schema.py`)

The upstream schema from `list_tools` is kept and annotated; derived tools get a full schema.

| bound to | schema additions |
|---|---|
| identifier (or label/synonym through `accepts`) | `x-vbt-id-type` (qualified), `x-vbt-accepts`, examples, description from plugins. No strict `pattern` for resolvable kinds (symbols must reach the resolver). `runtime._argument_problem` (L1549) checks only required and type, so resolution errors come from the gateway as typed errors, not `schema_mismatch`. Non-resolvable kinds (NCT) get `pattern`. List arguments (`each`) get `minItems`/`maxItems`. |
| category / scope, vocabulary ≤ `enum_max` | `enum` from the fingerprinted vocabulary snapshot; numeric values rendered with the storage-typed shortest repr (`0.05`, never `0.05000000074505806`) |
| category / scope, larger vocabulary | `x-vbt-vocabulary: table.column` and text pointing to `mcp__data__vocab` (P2) |
| scope key column the upstream signature lacks (rev 2) | an auto-derived gateway-only property (`x-gateway: true`) with the vocabulary as enum (Tahoe `concentration`, `plate`) |
| `role: selector` (rev 2) | `enum` of the selector values (`method: [coloc, ecaviar]`) |
| `role: order_by` / `order_direction` (rev 2) | `enum` of measure, count and time columns of the bound table (or the declared values); boolean for direction |
| measure threshold | `minimum`/`maximum` from `scale` tightened by `constraints` (`max_padj ≤ 0.10`); abs ops say "compared on the absolute value"; scale/encoding/direction/cutoff inclusivity in the description; "unknown values never pass" |
| `role: limit` | `minimum: 1`, `maximum = min(overlay.max, memory-derived max)`; `limit_grain` adds "counts <grain>, not rows" |
| `role: free_text` with `interpreted_as: regex` | "literal text, not a pattern" (the gateway escapes it) |
| `interpreted_as: engine` (rev 2) | the `engine_doc` sentence ("matched by the registry's search engine with synonym expansion") |
| `gateway_only: true` | added property with `x-gateway: true`; stripped before the upstream call because FastMCP schemas carry `additionalProperties: false` |
| `require_any` / `exclusive` | `x-vbt-require-any` / `x-vbt-exclusive` plus one sentence |
| schema `default` on a filter argument | description: "defaults to United States; pass null for all countries" |
| `role: anchor` (rev 2) | "the reference entity; excluded from the results" |

### 10.2 Agent-facing text (`derive/text.py`)

Description = upstream first sentence (unless an overlay `text.drop_promises` entry contradicts it)
plus a generated block, capped at `data.derive.description_max_chars` (1,200). Upstream `Returns:` and
`Example:` sections are dropped because several promise fields that do not exist (`chromosome`,
`cancer_hallmarks`, `mechanismOfAction`, a `type` column).

```
Data: Open Targets 25.09 · table known_drug — one record = one (drug, target, disease) record per phase and trial status.
Key: drugId, targetId, diseaseId, phase, status (phase and status may be null).
Arguments: target_id accepts an Ensembl gene ID or HGNC symbol/alias (resolved; unknown -> error).
Results: ranked by phase desc (verified by the harness); `_vbt.total` counts all matches; `_vbt.grains.drug` counts drugs.
Unknown phase rows are excluded and counted in `_vbt.excluded_unknown`.
Empty vs not found: unknown identifiers are errors. An empty result means no ChEMBL-curated drug; cite it only as an absence.
```

Revision 2 adds generated sentences for: item grains ("one row = one GO annotation of the gene;
`_vbt.total` counts annotations"), levels ("OS_MONTHS is a patient value repeated on each sample;
count patients with `_vbt.grains.patient`"), cutoffs ("dependent = gene_effect ≤ −0.5, inclusive"),
propagation ("direct annotations only" / "mixed: 9% of items also list an ancestor"), lossy
projections ("topLevelTerm keeps one root; events under two roots are missed — use expand"),
multiple-testing families ("padj is BH-adjusted within one contrast"), censoring ("times are
right-censored when status is false; medians need Kaplan-Meier"), evidence nature ("literature
co-occurrence, not function"), censored coverage, and default filters that are **not** applied
("is_primary_data: NOT applied by this tool; duplicate cells included").

Blocked tools keep their listing with the first line `UNAVAILABLE: <reason>; use <alternative>`
(or are hidden when `hidden: true`). `src/vbt/prompts/data_layer_addendum.md` (about 15 lines,
injected by `agents.py` into every data agent's volatile prompt when the gateway is enforcing)
explains statuses, `_vbt`, the absence rule and the evidence-nature caveat.
`src/vbt/prompts/genomics_burden_addendum.md:20` changes from "A query that succeeds with zero rows
is a valid outcome" to "A resolved query with `status: empty` means no rows in this source's coverage;
record it only as an absence finding (`supports: absence`), never as support. A `not_found` error
means the identifier is wrong."

### 10.3 Readiness checks (`gateway/readiness.py`, executed by `service/verbs/check.py`)

Per table, per item table, per column and per partition: R1–R10 (§13). Per call: the conjunction
over the columns and partitions the call reads (resolved from `reads`, the bound columns, the
selector's table and the partition predicate), the resolver indexes of the id_types it uses (degrading
per accepted kind, §13), the tables of coverage universes, plus server state.

### 10.4 Memory estimates (`memory/estimate.py`)

Revision 1 counted bytes only and was 3.8× low on `drug_molecule` and about 9× low on a three-level
nested fixture (VERIFIED: footer 11.6 MB, Arrow 130 MB, pandas peak RSS +1,001 MB against a 107 MB
estimate). Revision 2 uses two estimators:

- **Upstream pandas load** (admission of `full_table` reads by upstream servers):
  `peak(t) = (Σ_leaves uncompressed_bytes × expansion(kind) + Σ_leaves num_values ×
  object_overhead(kind, depth)) × fragmentation`, with seed overheads (string 50 B, nested item 120 B
  per depth level, dict per struct item 240 B) calibrated in the format conformance suite on the
  three-level golden and replaced by measured factors in phase 4 (F19).
- **Data-child Arrow scan**: projected leaf bytes × decode factor, after partition, row-group-stats and
  sidecar-index pruning; this is what `max_scan_bytes` is compared with.

For a call: `need = Σ_{t ∈ reads, access = full_table, t ∉ resident(server)} peak(t) + transient`, where
`transient` for `bounded_scan` reads is computed from footer statistics under the predicate. Admission
applies only when the route is `upstream`; derived and witness work is charged to the data child's
own budget (rev 2). `limit.maximum = floor(data.memory.max_result_bytes / row_bytes_p99)`, where
`row_bytes_p99` is measured on a sample taken during readiness **for the binding's output grain**
(stored row, item, matrix cell or computed result row), not the stored-row mean (one disease row is
307 KB while the mean is 1.1 KB).

### 10.5 Provenance template

Key columns (composed for item tables, axis keys for matrices) give the row-key extractor;
`ResultSpec.fields`, `parent_key` and `key_from_args` give the row-key values; release, lineage and
fingerprints give the header; resolutions give `request.resolutions`; scope columns give
`request.scope`. Format in §15.1.

### 10.6 Native tools (`derive/tools.py`, phase 2)

Generated per table kind and served by the data child under the `data` server:
`mcp__data__resolve(id_type, values[])`, `mcp__data__describe(source[, table])`,
`mcp__data__lookup(table, key)`, `mcp__data__find(table, where, rank_by, limit, distinct, group_by)`,
`mcp__data__aggregate(table, group_by, measure, how, min_n)` (rev 2: grouped aggregation over facts and
matrix long views, with axis attributes joined through `attributes_from`, statistic-plugin
aggregation, minimum n, nulls excluded and counted, per-group n reported),
`mcp__data__search(entity, text)` (each row carries `match`: exact, casefold, previous, alias, prefix,
word, substring; ordered by match class, then rank, then key),
`mcp__data__vocab(table, column)`, `mcp__data__members(set_table, set_id, propagate)`,
`mcp__data__similar(table, anchor, filter, top_k)` (rev 2), `mcp__data__expand(ontology, term,
relation, predicates)` and `mcp__data__enrich(...)` (phase 3), `mcp__data__neighbors(edges, node)`.
The `table` argument is an enum of ready tables and item tables whose roles support the verb. **Every
`where` and `key` entry is bound to a long-view column and resolved exactly like an overlay argument**
(the column's id_type, labels, synonyms, crosswalks), so `find` raises `not_found`, `ambiguous` and
`invalid_argument` like any bound tool (rev 2); the JSON schema of `where` is derived from the roles
(`x-vbt-id-type`, enums, threshold bounds). Agents receive the tools through `configs/agents.yaml`
globs (`mcp__data__*`), except for tables with `expose.native: false` or `expose.withhold_from` naming
the agent (the Case 1 answer key `trial_labels` is withheld from `trial-annotator` and
`clinical-trialist`).

**In-process client (rev 2, phase 2).** `vbt.datalayer.client` gives harness modules and notebooks the
same verbs as Python functions (`client.find("open_targets.disease", where=...)`,
`client.open_matrix("zenodo_vbt.ibd_cohorts")`). Each call goes through the data child, applies the same
resolution, limits and readiness, and writes a `vbt.dataprov/1` record, so artifacts built from it can
cite their inputs (`register_artifact(derived_from=[prov ids])`). The Case 1 readers in
`src/vbt/case_studies/trial_outcomes/` and `vbt.analysis.biomarker.load_processed_cohort` move onto it.

---

## 11. The gateway

### 11.1 Hook points

| File : symbol | Change |
|---|---|
| `src/vbt/tools/mcp_bridge.py : MCPServerConfig` (L84) | New fields `mem_limit_mb: int \| None`, `overlay: str \| None`, `sources: list[str]`, `launcher: bool \| None`. `runtime.py:569` and `preflight.py:520` filter by `__dataclass_fields__`, so the fields must exist there to survive. |
| `MCPBridge.__init__` (L276) | `gateway: GatewayProtocol \| None = None`, `on_tools_changed: Callable[[list[Tool]], None] \| None = None`. With `gateway=None` behaviour is byte-identical to today (`tests/test_mcp_bridge.py` stays valid). Calls `gateway.bind_bridge(self)`. |
| `MCPBridge._open` (L388) | For stdio servers, `spec = gateway.launch_spec(cfg)` rewrites command/args/env to run under `launch/reaper.py`; HTTP servers unchanged. |
| `MCPBridge._register` (L514) | Calls `gateway.rewrite_listing(server, tool_name, description, schema)`; hidden tools (all `_`-prefixed tools of `data`) are not registered. Replaces the `if full in known: continue` skip with **update in place** (description, schema, handler) and calls `on_tools_changed(new_or_updated)`. |
| `MCPBridge._ready_session` (L561) / `_restart` (L576) | New `st.state = "broken"` after a failed retry: next call restarts first instead of reusing the dead session (VERIFIED bug). OOM kills count against a separate `max_oom_kills` budget. New `recycle(server, wait_s)` restarts an idle server without counting a crash. |
| `MCPBridge._make_handler` (L601) | Passes `ctx` to `call(server, tool, args, ctx=ctx)`. |
| `MCPBridge.call` (L606) | Split: `call_raw()` is today's logic (used for internal data-child calls); `call()` runs `gateway.prepare` once before the attempt loop (a retry reuses resolved arguments), holds `plan.hold()` (cold-call lock), invokes with `plan.args_sent`, consults `gateway.on_crash` on transport errors (no retry on OOM), then returns `await gateway.finish(plan, raw)`. |
| `MCPBridge._convert` (L675) | New keyword `classify_only`. With it, legacy error envelopes, `isError` and `_EMPTY_LOOKUP` matches return `RawResult(text, structured, parts, envelope)` instead of raising, so the gateway owns classification. `tool_result_error` (L152) and `_EMPTY_LOOKUP` (L132) are unchanged. |
| `src/vbt/runtime.py : Runtime.start_mcp` (L567) | Builds the gateway (`vbt.datalayer.build_gateway(config, run)` when `data.enabled`), appends `gateway.extra_servers()` to the specs, passes `gateway=` and `on_tools_changed=` (→ `registry.extend` + `_system_cache.clear()`; fixes late tools never reaching the registry). Exposes `${run.mcp_output_dir}` to descriptor expansion (rev 2). |
| `Runtime._execute` (L1458) | A `DataResult` return: content = `result.text`, shaping uses `result.obj`; `end["result_status"]`, `end["data_provenance"] = summary` (including coverage), full record to `logs/data_provenance/<tool_use_id>.json`. The gateway hands over the **unshrunk** payload (`DataResult.full_text`) so the runtime spill to `logs/tool_outputs` keeps the complete record before the 2 MB model-facing cap applies (rev 2; a live record cannot be fetched again in the same version). A `GatewayError`: `end["error_kind"]`, `rec["error_kind"]`. |
| `Runtime.set_degraded` (L595) / `unavailable_servers` (L588) / `_list_tools_tool` (L1770) | Per-tool readiness map (`set_tool_readiness`); ListTools and the prompt list unready tools grouped by table. |
| `src/vbt/tools/base.py : shrink_json` (L381) | Never shrinks the value of the top-level `_vbt` key. |
| `src/vbt/failures.py : _data_failure` (L66) | Also exempts `error_kind ∈ MODEL_SIDE_KINDS` (lookup misses are the model's input errors, not outages); they stay `is_error` and uncitable. |
| `src/vbt/session.py : Run._index_event` (L429), `mark_degraded` (L371) | Copy `result_status`, `error_kind`, `prov` and `coverage` into the live index; degraded becomes `{servers, tools, tables, reason, at}`. |
| `src/vbt/audit/provenance.py : Provenance._build` (L345, tool_end branch ~L483) / `_finish_call` (L516) | Keep `result_status`, `error_kind`, `coverage` and the provenance summary; old traces parse as `unknown`. |
| `src/vbt/audit/claims.py : check_evidence` (L233, tool_call branch ~L290) | Status, coverage, leakage and evidence-nature rules (§15.3). |
| `src/vbt/preflight.py` | Per-table and per-column `CheckResult`s (new optional `scope` field), `degraded_tools`, `degraded_servers` derived from it, smoke through the gateway with sentinels; the legacy aggregate data result is `ok` unless no granted data tool is ready (rev 2). |
| `src/vbt/orchestrator.py` (~L858-894) | `MANIFEST.degraded` built from `degraded_tools` as `{servers, tools, tables, reason}`; never the `"(reference data)"` pseudo-server (rev 2). |
| `src/vbt/pinning.py : build_pinned_config` (L277) | `pinned["data"]` from `gateway.pinned()`, fail-soft. |
| `src/vbt/mcp_servers/pubmed_server.py` (L31) | Phase 1: `EUTILS` may be overridden by `VBT_EUTILS_BASE` so tests can stub E-utilities and count requests (rev 2). Phase 4 fixes the remaining PubMed defects in place. |
| `src/vbt/case_studies/trial_outcomes/schema.py` (L116) | `pubmed_ids` joined with `|`, matching the released labels file (rev 2). |
| `src/vbt/context.py` | No change: the header is the first key or first line, never appended after a truncation note, so `_SPILLED_RE` (L86) keeps working. |

### 11.2 Contracts between bridge and gateway (`src/vbt/datalayer/api.py`)

```python
@dataclass
class ListingDecision:
    visible: bool; description: str; input_schema: dict[str, Any]; reason: str | None = None

@dataclass
class RawResult:
    text: str; structured: Any; parts: list[Any] | None
    envelope: Literal["ok", "empty_lookup", "legacy_error", "is_error"]
    error_text: str | None = None

@dataclass
class CallPlan:
    server: str; tool: str; args_raw: dict[str, Any]; args_sent: dict[str, Any]
    route: Literal["upstream", "derived", "none"]
    contract: Any; resolutions: list[dict[str, Any]]; witness: dict[str, Any] | None
    bound_table: str | None = None                 # after selector resolution (rev 2)
    scope: dict[str, Any] = field(default_factory=dict)        # fixed and listed scope values (rev 2)
    existence: dict[str, Literal["exists", "absent", "unknown"]] = field(default_factory=dict)   # per argument
    gateway_args: dict[str, Any] = field(default_factory=dict)
    cold_tables: tuple[str, ...] = (); cold_lock: asyncio.Lock | None = None
    record: "DataProvenance | None" = None
    def hold(self) -> AsyncContextManager[None]: ...      # acquires cold_lock when set

@dataclass
class CrashDecision:
    retry: bool; error: "GatewayError | None"; oom: bool = False

@dataclass
class LaunchSpec:
    command: str; args: list[str]; env: dict[str, str]; status_path: str

class GatewayProtocol(Protocol):
    mode: Literal["off", "observe", "enforce"]
    def bind_bridge(self, bridge: Any) -> None: ...
    def extra_servers(self) -> list[dict[str, Any]]: ...
    def launch_spec(self, cfg: Any) -> LaunchSpec | None: ...
    def rewrite_listing(self, server: str, tool: str, description: str, input_schema: dict) -> ListingDecision: ...
    async def prepare(self, server: str, tool: str, args: dict, ctx: Any) -> CallPlan: ...   # raises GatewayError
    async def finish(self, plan: CallPlan, raw: RawResult | None) -> Any: ...                # DataResult; raises GatewayError
    async def on_crash(self, server: str, plan: CallPlan, reason: str, log_tail: str) -> CrashDecision: ...
    def pinned(self) -> dict[str, Any]: ...
    def readiness_snapshot(self) -> dict[str, Any]: ...
```

In `observe` mode `prepare` never raises and `finish` returns the upstream result unchanged, but both
emit `data_observe` trace events with the decision they would have taken (`would_not_found`,
`would_repair`, `would_refuse_memory`, ...). In `off` mode the bridge is constructed with `gateway=None`.

### 11.3 `prepare(server, tool, args, ctx) → CallPlan`

1. **Contract lookup**: `Catalog.contract(server, tool)` (reviewed binding, `same_as` alias, or
   generic). A `role: selector` argument chooses the bound table now (`method: coloc` →
   `colocalisation_coloc`; `include_indirect: true` → the indirect table); an unknown selector value
   is `invalid_argument` with the valid values (rev 2). Every later step, including readiness and the
   witness, uses the selected table.
2. **Block**: `serve: block` → `quarantined` (reason, alternatives).
3. **Readiness**: cached readiness of exactly the columns, nested containers and partitions this call
   reads (§13), with a stat-signature revalidation of those tables only. An accepted id_type whose
   index is not ready is dropped from `accepts` for this call with a `_vbt.notes` entry; the call is
   `not_ready` only when no accepted kind is ready or the bound data is not (rev 2). Failure →
   `not_ready` naming the table, column or partition and the failed check.
4. **Argument contracts** (`gateway/contracts.py`):
   - unknown argument names (when the upstream schema forbids them), `require_any`, `exclusive`;
     limit bounds (≤ 0 → `invalid_argument`); list arguments: `min_items`, `max_items` (reject instead
     of truncating), deduplication;
   - enum and vocabulary resolution: exact, then casefold if the facet allows, then `aliases` and
     `aliases_from`; numeric values are cast to the column's storage type and snapped to the
     vocabulary snapshot (relative tolerance 1e-6), and the stored value is echoed in `_vbt.scope`
     (rev 2, C20); else `invalid_argument` with the payload of §12.1;
   - `order_by` and `order_direction` validated against their enums; `send_map` and `send_as: alias`
     translate to the form upstream expects (`P` → `biological_process`);
   - substring-collision check for `interpreted_as: substring`; regex escaping for `interpreted_as:
     regex`; `escape` through the named format's `quote()`; `forbid` and `pattern` checks
     (`invalid_argument` when a value cannot be quoted safely); `wrap`;
   - flags: `when_true`/`when_false` must name codes positively (`{in: [none_recorded]}`), so NaN, a
     value between codes or a new code never counts as the favourable state;
   - an **unbound** filter argument (`role: unbound`) with a non-default value → `unsupported_filter`
     (rev 2, I8); a threshold on a `comparable_within` measure whose group no argument fixes →
     `unsupported_combination` naming the group argument (rev 2, C30);
   - `output_path` confinement (absolute or `..` paths rejected) and write-once: an existing target is
     renamed to `<stem>.<prov>.<ext>` and disclosed (rev 2, phase 1);
   - materialisation of disclosed schema defaults; `gateway_only` arguments stripped;
   - SOMA `value_filter` strings are parsed by `gateway/soma_filter.py` into the Predicate IR (`==`,
     `!=`, `in`, `<`, `>`, `and`, `or`, `not`); each column and value is resolved against Census
     vocabularies (fetched with the upstream `list_metadata_values` tool and cached with a TTL);
     `is_primary_data == True` is ANDed in unless `include_duplicates=true`; the filter is recompiled
     with SOMA quoting. Unparsable filters are `invalid_argument` in enforce mode (rev 2);
   - `role: projection` arguments (`obs_columns`) get the descriptor's key columns, `unique_within`
     qualifiers and default-filter columns added (disclosed); a call whose output could not carry
     the complete key is refused (rev 2).
5. **Identifier resolution** for every identifier-bound and anchor argument (§11.5). Values are
   replaced by canonical keys (or the forms named by `send_as`). Each resolution is recorded with
   matched and canonical id_types, rule, family and hops.
6. **Witness pre-scan** (§11.6) when the binding allows it and the bound table is scannable: totals,
   `excluded_unknown`, `unknown_total`, distinct values of unfixed scope key columns, group totals.
7. **Scope completeness (rev 2: one rule for every serve mode, I7).** For each key column of the bound
   table (or item table) that is a scope dimension, is not fixed by an equality-bound argument
   (gateway-only arguments included), and has more than one distinct value under the predicate:
   - if the result's `grain` is coarser than the table key and omits the dimension (rows would be
     merged), or a `summary_fields` recompute or a `limit_grain` would aggregate across it, or
     upstream rows do not carry it (pass mode, via the field map): `pooling: forbid` →
     `incomplete_key` with `{dimension, argument, values, unit}`; `group` → per-value groups;
     `list` → rows listed with the value and disclosed in `_vbt.scope`;
   - if the order ranks by a measure whose `comparable_within` includes the dimension: rank within
     groups when `limit_mode: per_group` (the cut keeps k rows per group, stated in `_vbt.order`), or
     `incomplete_key` with `subkind: incomparable_order` when `limit_mode: refuse` or when the cut
     cannot be applied per group (pass mode without successful inflation);
   - otherwise (rows carry the dimension and the ranking measure is comparable across it) the rows
     are listed and the values disclosed in `_vbt.scope`.
   `pool_ok_for` lets `exists`, `count` and `distinct` verbs pool across a partition dimension.
8. **Leakage** (rev 2, phase 1): when `data.leakage.ceiling` is set and a read table's source declares
   `leakage`, count and search tools get the overlay's `leakage_filter` injected (`AREA[StudyFirstPostDate]
   RANGE[MIN,<ceiling>]`); a counting tool without a `leakage_filter` is `quarantined` (reason:
   leakage) under `counts: block`; row tools are filtered in `finish` (T1).
9. **Memory admission** (§14.3), only when the route is `upstream`: estimate, residency ledger,
   recycle or `too_large`; cold-call lock assigned when the call will load a non-resident table.
10. **Limit inflation** (§11.6) and **route**: `upstream` for `pass`, `derived` for `serve: derived`.

### 11.4 `finish(plan, raw) → DataResult`

1. **Classification** (`gateway/classify.py`), in this order (rev 2 fixes the precedence):
   1. transport failure handled by `on_crash`; memory signatures (`MemoryError`, `Unable to allocate`,
      `ArrowMemoryError`, `bad_alloc`) in any error text → `oom`;
   2. `not_found_when` predicates and the explicit not-found patterns (`_EMPTY_LOOKUP`), evaluated
      **before** legacy envelopes are mapped to `source_error`, for reviewed and generic bindings alike
      (cBioPortal's "Failed to retrieve ... status code: 404" is `not_found`, not a data-side outage);
      then, for a binding whose bound table is the id_type's universe table and whose existence mode is
      `upstream`, → `not_found`; if resolution already proved the entity exists and the bound table is
      the universe table → contradiction (repair or `tool_defect`); if the bound table is a different
      table → `empty` (confirmed by witness count 0); unbound → `not_found` (generic guard);
   3. `nested_errors` present (e.g. Tahoe `drug_info.error`) → `source_error`, unless `total.partial_when`
      matches, which gives `partial` with `total: null`;
   4. remaining `legacy_error` / `is_error` → `source_error`.
2. **Rows**: extract at `result.rows` (one or several paths), parse JSON text; non-JSON text keeps a
   header line. Map fields to descriptor columns with `result.fields` (renamed fields, computed fields
   such as `similarity`), inject `parent_key` and `key_from_args`, and attach rows to the item table in
   `rows_of`. When a bound filter column or key column cannot be located in the rows, the check that
   needed it is recorded with `ok: false` in provenance instead of being skipped silently.
3. **Role transforms** T1–T6 (§11.7: leakage, in-band unknowns, existence per row, honour arguments,
   unknown never passes, negation).
4. **Witness checks** (§11.6) on the rows after T1–T6: W1–W6. A returned row whose re-applied bound
   predicate is false or unknown has already been excluded and counted by T4/T5; it is not a W2
   contradiction (rev 2).
5. **Remaining transforms** T7–T14: duplicates, key-completeness disclosure, level split, order and cut
   (per group, `limit_grain`), counts and summaries, trims, flag partition, measure validity.
6. **Sections and files**: per-section readiness and witness; an unavailable section becomes
   `{"_vbt_unavailable": "<table> not ready: <check>"}` and summaries that depend on it are removed;
   `FileCheckSpec` reconciliation and `materialized_by` registration (§6.7).
7. **Status**: `ok`, `partial` (truncated, a section unavailable, total unknown, pooled, family rows
   elsewhere, or unknown items withheld), `empty` (resolved, zero rows, coverage attached), `empty_unverified`
   (zero rows without proof, or existence `unknown`).
8. **Header and record**: `_vbt` inserted as the first JSON key (≤ 1,200 chars); `DataProvenance`
   built; the unshrunk payload is handed to the runtime spill before the result cap applies.

### 11.5 Resolution (`resolve/rules.py`, `resolve/resolver.py`, `resolve/index.py`)

- **Indexes are derived, not rebuilt data.** For each id_type the data child projects the universe
  key, the label and synonym columns (`resolve_via`, with containers expanded to their synonym-role
  leaves), retired-ID columns, xref columns and crosswalk tables, and writes a gzip TSV sidecar
  `${data.cache_dir}/<source>/<fingerprint>/index/<id_type>.tsv.gz` with columns
  `label_key, canonical, rule, label, stored_table, stored_value, family` (rev 2 adds the last three).
  The harness loads it lazily with stdlib `gzip`+`csv`. Huge universes (`rsid`, `ot_variant`,
  `study_locus_id`) are `index: remote` and resolved by the data child.
- **Qualified id_types (rev 2).** id_type names are qualified by source (`tahoe_100m:depmap_cell_line`
  vs `depmap:depmap_model`). A bare name in a descriptor means that source's own type; a bare name in an
  overlay's `accepts` means the bound table's source, then a unique match across loaded sources;
  lint fails on ambiguity. The identifier **plugin** is the identity space and the universe is per
  source, so two sources that share a plugin share syntax and can be compared, while each decides its
  own existence. `ArgBinding.universe` overrides the universe for one binding.
- **The resolver always yields the bound column's id_type (I1, rev 2).** An accepted kind that is not
  the bound column's kind reaches it through a declared edge: `label_of`, `maps_to`, a crosswalk, or a
  union member. Without such an edge lint fails. `request.resolutions` records `matched_id_type` and
  `canonical_id_type`.
- **Closed rule grammar (rev 2)**, recorded in `Resolution.rule`:
  `exact` | `raw_member` | `normalized:<step>[+<step>…]` | `label_exact:<column>` |
  `label_casefold:<column>` | `synonym:<kind>` | `retired:<column>` | `xref:<namespace>` |
  `crosswalk:<name>[><name>]` | `parent_family` | `stored_form`. CT-1 asserts
  `label_exact:approvedSymbol`, `normalized:strip_version`, `normalized:upper` and `synonym:alias`.
- **Rule order** is a per-id_type list (`IdTypeSpec.rules`), defaulting to: `raw_member` (the raw value
  is itself a universe key: checked **before** normalisation so stored forms such as `GO_0000002` in
  `disease.id` are found) → `exact` → `normalized` → `label_exact` → `label_casefold` (unique) →
  `synonym:previous` (unique) → `synonym:alias` (unique) → `synonym:exact` (unique) →
  `synonym:related` (unique, disclosed with a warning note) → `retired` → `xref` → `crosswalk`.
  `synonym:narrow` and `synonym:broad` never resolve: they appear only in `search` results, labelled
  `narrow_synonym` / `broad_synonym` (a unique broad synonym would substitute a broader concept, I1).
  `synonym_kind_from` reads the kind per item (OBO `synonym: "x" RELATED []`).
- **Retired IDs (rev 2).** After a canonical miss and before `not_found`, the `retired` rule looks the
  normalised value up in the columns named by `IdTypeSpec.retired` (`disease.obsoleteTerms`: 4,880
  retired IDs, each folded into one current term; GO terms whose name starts with `obsolete ` and
  their `replaced_by`). One replacement → resolved, disclosed in `_vbt.resolved` and `_vbt.notes`;
  several (`consider`) → `ambiguous`; none → `not_found` with `subkind: obsolete` naming the
  replacement candidates.
- **Cross-references (rev 2).** `xref_via` lists columns of CURIEs (`disease.dbXRefs`: 105,986 xrefs
  over 12+ namespaces) with a namespace → id_type map, or with `id_type_from: {field: source, map}`
  when a sibling field names the namespace (`drug_molecule.crossReferences`). `DOID:9352` resolves to
  `MONDO_0005148` by rule `xref:DOID`; an xref shared by several terms (4,372 do) → `ambiguous`.
- **Crosswalk chains and stored forms (rev 2).** `crosswalks` declare mapping tables; `maps_to` links
  canonical keys of different id_types through one (`tahoe_gene` → `open_targets:ensembl_gene` via
  `gene_metadata.ensembl_id`). The resolver follows at most `data.resolution.max_hops` (2) edges
  (OT symbol → ENSG → Tahoe name), recording each hop in the rule (`crosswalk:tahoe_ensembl`); a
  one-to-many hop → `ambiguous`. The index keeps, per table that stores the type, the stored value of
  each canonical key (`normalize_stored`: Tahoe stores `'Erdafitinib '`), so `send_as: stored` sends
  the bound table's own spelling and `send_as: native_label` sends the bound table's own label
  (`SEPT9` in a pre-2020 cohort for the current symbol `SEPTIN9`). A canonical key that two stored
  values collapse onto (`ENSG00000002586.18` and `..._PAR_Y`) is a `key_violation` readiness finding
  unless the plugin declares the `strip_suffix` qualifier as a key part.
- **Parent and salt families (rev 2).** `canonicalize: {parent: drug_molecule.parentId}` groups a parent
  with its salt forms. Identifier columns declare `form: parent | as_stored | any`. The resolver
  collapses candidates that share one parent to the parent (rule `parent_family`; members listed in
  `_vbt.notes` and in `request.resolutions[].family`), so `Tarceva` resolves to erlotinib instead of
  `ambiguous` (290 of 389 ambiguous trade names fall inside one family). For a fact column with
  `form: any` (`chembl_clinical_nct_data.drugId`: 481 salt IDs among 3,936), a parent query with
  `family: exact` is answered for the exact ID and the witness counts rows stored under other family
  members: status `partial` with `_vbt.family_rows: {CHEMBL1079742: 12}`; with `family: include` the
  predicate becomes `In(family members)` and every row carries its stored ID. A parent outside the
  universe keeps the stored ID with a note (31 parents are missing from the real extract). Counting by
  family uses a canonicalized grain (`parent_molecule`).
- **Ambiguity.** More than one candidate → `ambiguous` with up to `data.resolution.max_candidates`
  `{id, label, via}` plus the values of `disambiguate_with` columns (`rheumatoid arthritis` is both
  `HP_0001370` and `EFO_0000685`; `isTherapeuticArea` tells them apart). `prefer` declares a disclosed
  order that resolves such ties only when the descriptor states it.
- **Wrong kind and syntax (rev 2).** If every accepted kind's `normalize()` returns `Rejected` and no
  label, synonym, retired or xref rule applies, the error is `invalid_argument` carrying each
  `Rejected.reason` and the union of `looks_like` hints and plugin `looks_like` scores ≥ 0.8 (`PMC1234`
  → "looks like a PMCID"). `not_found` is reserved for syntactically valid values absent from the
  universe.
- **Existence (I2, rev 2).** `ArgBinding.existence`:
  - `universe` (default): absent from the identity universe → `not_found` with same-id_type
    suggestions (edit distance 1 or shared label key) and a `tried` list;
  - `bound`: `not_found` only if the value is absent from both the universe and the bound column (one
    witness count); a value found in the bound column but not in the universe (a pgx `drugId` missing
    from `drug_molecule`) is resolved with a note;
  - `upstream`: the universe is remote and is decided by upstream's explicit not-found or
    `not_found_when` on a tool whose bound table is the universe table; `IdTypeSpec.universe_via` may
    name an upstream listing tool cached with a TTL (`search_studies` for `cbio_study`);
  - `off`: syntax only.
  When existence cannot be decided (remote universe unreachable, `_resolve_remote` over budget), the
  outcome is `unknown`: the call proceeds, a note says so, and a zero-row result becomes
  `empty_unverified`, never `not_found` or `empty`. `ArgBinding.universe_where` restricts the universe
  (`isTherapeuticArea = true`), so `cancer` (an exact MONDO term that is not a therapeutic area) is
  `invalid_argument` listing the 26 valid values.
- **List arguments (rev 2).** With `each: true`, elements are resolved one by one and deduplicated (a
  symbol and its ENSG count once). An invalid or not-found element fails the whole call with
  `items: [{index, value, reason}]` under `on_missing: error`, or is dropped and listed in
  `_vbt.resolution_summary` under `drop_disclosed`/`partial`; no partial upstream call is made under
  `error`. `min_resolved_fraction` turns a mostly unresolved gene list into
  `insufficient_resolution`. `_vbt.resolution_summary` reports `{requested, resolved, unresolved[],
  ambiguous[], outside_universe[], duplicates[]}`.
- **Union kinds (rev 2).** For `IdTypeSpec.union` (embedding words), every member kind is tried; more
  than one distinct canonical → `ambiguous`; existence is checked against the bound column (anchor),
  and the identifier facet `kind_from` tells readiness which member kind each stored value must satisfy.
- **Hierarchy expansion.** `include_descendants(X)` means `Eq(of, X) OR Contains(<ancestor column>, X)`
  when the bound row carries an ancestor list (OT `ancestors` excludes the term itself: 0 of 126,689
  rows contain their own `diseaseId`, VERIFIED), and `In(of, expand(X) ∪ {X})` when it does not
  (phase 3, through the id_type's `hierarchy`, which may live in another source via `extends`). A
  column with `propagated_over` (indirect associations) refuses `include_descendants`
  (`unsupported_combination`), because its rows already include descendants.

### 11.6 Witness, limit inflation and contradictions

The **witness** is an independent bounded read in the data child of the tool's bound table or item
table, filtered by the predicate built from the bound arguments (resolved values; `Any` over nested
items; Kleene logic; storage-typed literals), projected onto key, filter and order leaves, with
partition, row-group-statistics and sidecar-index pruning.

**Request** (rev 2 adds the fields in italics): `table` (an item table counts items), *`grain`*,
`predicate`, `key`, `order`, `k`, *`group_by`* (comparable groups), `distinct` columns,
*`grains: {name: [cols] | GrainSpec}`*, *`unknown_columns`*, `key_set_max`, `budget_bytes`.

**Response**: `total`, `total_method` (`scan` | `footer` | `index` | `unknown`), `topk` (canonical keys
under the declared order, per group when `group_by` is set), `key_set` when `total ≤
data.witness.max_key_set`, `distinct` values, *`excluded_unknown: {column: n}`* (rows where every other
conjunct is true and the column's comparison is unknown, attributed to every such column, with
`_rows` for the total), *`excluded_not_applicable`*, *`unknown_total`*, *`distinct_counts: {grain:
n}`* (over all matching rows), *`group_totals`*, `one_to_many`, `scanned_bytes`, `reason`. When the
predicate involves an argument the witness cannot express (an unbound filter, free text matched by a
remote engine, a remote table), or the scan would exceed `max_scan_bytes` (per table or global), it
returns `total_method: unknown` and the gateway makes no ranking or count claim.

**Reading cost (rev 2).** The reader scans in two passes: the key and filter leaves first (projected
per fragment and row group with `read_leaves`, which works for leaves inside lists where the dataset
API does not), then the order and output leaves only for matching row groups or rows (`take`).
Partition columns are constant per fragment and never materialised per row. An `access_paths` entry
with `via: sidecar_index` uses a fingerprint-keyed sidecar (`value → (fragment, row group)`) built in
`data.cache_dir` by `_build_index` at readiness or on first use, which needs no data rebuild; this is
what makes `find_drugs_affecting_gene` (a gene predicate never prunes Tahoe row groups) and
`query_evidence(target_id=…)` (each europepmc shard spans every `targetId`) answerable within the
2 GB scan budget in phase 1. Derived serving returns its own total and top-k, so a derived call costs
one scan; the existence check for a value bound to the scanned column reuses that scan.

**Limit inflation** (generic: any binding with a `limit` argument and a `result.order` or rows list):
- if `total + unknown_total ≤ data.witness.max_inflate_rows` and that many rows ×
  `row_bytes_p99` ≤ `max_inflate_bytes`, send `limit = max(total + unknown_total, requested)`; the
  upstream head then sees every match even when it lets unknown rows through; the gateway applies
  T1–T6, sorts with the statistic plugin's `sort_key` (per group when the order is `within` a group)
  and cuts to the requested limit (or per group, or per `limit_grain`). No upstream tool clamps
  `limit` (checked: no `min(limit` or bound in `mcp_servers/*/tools.py`), and upstream already holds
  the table in memory, so the extra cost is serialisation. Echoed limits are restored (`arg_echo`);
- **inflation is disabled** for bindings with an `output_path` argument (the written file would hold
  the inflated rows) and when `witness: false`; those tools use top-k comparison or the refusal below
  (rev 2);
- else send the requested limit and **compare** the returned keys with the witness top-k: equal (up to
  ties) → `order: verified`; different → `derived` repair if declared, else `tool_defect`;
- if the witness could not compute a top-k and the tool declares an order: **refuse the ranking**
  with `too_large` (`subkind: unranked_truncation`, hint: narrow the query or use `mcp__data__find`),
  unless `order_source` says the ranking is produced by the source (`source_server_side`: disclosed
  "ranked by source, not verified") or by a full upstream sort proven by a detector test
  (`upstream_full_sort`). The gateway never re-sorts a prefix it knows may be truncated and calls it
  ranked. `data.witness.topk: false` disables only the top-k part (counts stay), which CT-4 uses.
- `witness: false` on a binding means: no inflation, no ranking refusal, `total_method: unknown`, and
  existence decided by resolution alone.

**Contradictions** (I4), evaluated on rows after T1–T6:
- W1 empty: upstream 0 rows and witness `total > 0`.
- W2 outside: returned rows that satisfy the re-applied predicate but whose keys are not in the
  witness key set (an argument was ignored or the wrong column was matched).
- W3 top-k: as above.
- W4 echo: a single-record tool returned a key other than the requested canonical. The echo path
  must read a record field (`source: record`); an echo copied from the request is lint-rejected. A
  mismatch the source explains (an HTTP redirect to the canonical ID, or a declared synonym column
  such as `nctIdAliases`) is a resolution with rule `alias_redirect`, disclosed and substituted in the
  row keys, never a defect (rev 2; 3,162 CT.gov studies have aliases).
- W5 one-to-many: a lookup returned one row while the witness found several for the key.
- W6 phantom rows (rev 2): a returned row that fails `ResultSpec.exists_when`, or whose `ref`
  identifier is null when the descriptor declares it not nullable (cBioPortal sample rows with
  `patientId: null`), is withheld; the call raises `not_found` listing the items, or returns `partial`
  with `_vbt.not_found_items` under `on_unknown_items: partial`. W6 also covers short pages: if
  `returned < min(total, limit)` after T1–T6, the gateway re-calls once with the corrected limit, then
  repairs through `derived`, else returns `partial` with the reason.

Repair (`on_contradiction: derived`, only when `derived` is declared and the table is within
`data.witness.repair_max_bytes`) answers from the data child with full keys and
`_vbt.served_by: repaired`; otherwise the result is withheld as `tool_defect` naming the witness counts.
The witness reproduces the **intended** semantics declared by bindings, not upstream's code; where the
two cannot be compared (free text, remote engines) it reports `unknown` rather than raising a false
`tool_defect`. Remote tables get a witness in phase 4 (an independent count request compiled from the
bound predicates, within the source's request budget); until then every remote tool carries a
`defects` entry with a detector test for each known upstream miscount (`count_clinical_trials`
returning `totalCount` default 0).

### 11.7 Role-driven result transforms (`gateway/transforms.py`)

Applied in this order to the extracted rows and nested items (rev 2 renumbers and extends):

- T1 **Leakage**: rows with `available_at` after the ceiling are withheld; rows with `changed_at`
  after it are withheld or have `redact` columns nulled per `LeakageSpec.rows`; counts in
  `_vbt.withheld.leakage`; `provenance.leakage.risk` set when anything could not be checked.
- T2 **In-band unknowns and placeholders**: `missing_values`, `unknown_when`, NaN and category/label
  `placeholders` (`"Unknown"`, `'nan'`, a name equal to its own key) become null with a note; result
  fields mapped through `FieldMap.placeholders` with `on_placeholder: dangling_ref` are reported as
  dangling references with a count (upstream writes `"Unknown"` for a missing parent term).
- T3 **Existence per row**: W6 (`exists_when`, null non-nullable references).
- T4 **Honour bound arguments**: each filter argument bound to a column present in the rows (via the
  field map) is re-applied, with `binds_any` as `Or` and `item_filter` per nested item; failing rows
  or items are dropped and counted in `excluded.<arg>`; with `drop_empty_parents` a parent left with
  no qualifying items is removed and counted (CT-6: a disease whose only PCS item is negated).
- T5 **Unknown never passes** (I6): rows whose bound measure, count, flag or time is unknown are
  dropped from threshold results and counted in `excluded_unknown.<column>`; rows outside
  `applies_when` are counted in `excluded_not_applicable.<column>` instead. A null value takes
  precedence over T4: it is counted as unknown, not as excluded.
- T6 **Negation**: rows or nested items whose `qualifier` with `effect: negate` is true are removed
  unless `include_negated=true` and counted in `excluded_negated`; nested counts declared in
  `count_fields` (dict form) are recomputed; with `DerivedSpec.nest`, grouping and `having`
  (`min_evidence`) are applied after negation.
- T7 **Duplicates**: with a `qualifier` `effect: duplicate` and `default_filter: true`, duplicate rows
  are dropped unless `include_duplicates=true` and counted; when the filter cannot be applied (a file
  result), the text says "NOT applied" and the count comes from a count query (phase 4).
- T8 **Key completeness disclosure**: row-key columns absent from rows are listed in
  `_vbt.pooled_over` (the refusal itself happens in `prepare`, §11.3 step 7).
- T9 **Levels (I15)**: columns listed in `ResultSpec.levels` (or carrying the `level` facet) are moved
  into a `<level>s` section with one row per level key (`patients` beside `samples`); counts, events
  and statistics over them are computed per level key; `_vbt.grains.<level>` is reported.
- T10 **Order and cut**: statistic `sort_key`, nulls last, ties by canonical key; groups for `within`
  orders; `limit_grain` keeps each grain's best row; the cut is never applied to a declared grain
  boundary (a patient's samples are kept together when the result is truncated).
- T11 **Counts and summaries**: `count_fields` set to returned counts; `_vbt.total` from the witness,
  the data child or `result.total` (`total_method: upstream`, or `upstream_upper_bound` when the
  gateway dropped rows from an upstream-reported total); `grains` as `{returned, total}` per declared
  grain; `summary_fields` recomputed (with `group_by`) from witness rows or dropped; `drop_fields` and
  statistic `vetoed_companions` (cosine `interpretation`) removed with reasons; `arg_echo` restored.
- T12 **Trim** nested arrays over `result.trim` or `data.results.relation_list_max` in the declared
  item order (`rank` on the container, default item key), keeping counts in
  `_vbt.trimmed.<path>: {returned, total}` with a hint to `mcp__data__expand`.
- T13 **Flag partition**: nested items with a `flag` facet `partition_items: true` are split into
  `<path>` (true), `<path>_not_met` (false) and `<path>_unknown` (null), so a `value:false` tractability
  bucket can no longer be read as tractable.
- T14 **Measure validity**: `undefined_when` facets null out values whose denominator is zero
  (single-cell `sparsity` when `n_measured_obs == 0`); non-finite computed values (cosine of a
  zero-norm vector) become null and are excluded from ranking.

### 11.8 Data child interface (`src/vbt/datalayer/ipc.py`)

Hidden FastMCP tools on the `data` server, called by `gateway/service_client.py` via
`MCPBridge.call_raw("data", name, payload)`. All payloads are pydantic models serialised as JSON.
Verbs are discovered from `service/verbs/*.py` (each module exports `VERBS: dict[str, callable]`), so
later phases add verbs as new files without editing `server.py`.

| verb | request | response |
|---|---|---|
| `_stats` | `{tables: [TableRef]}` | per table: fingerprint, partition fingerprints, rows, fragments, bytes_on_disk, per-leaf `{uncompressed_bytes, num_values, max_rep_level, max_def_level, kind, storage_type, null_count, min, max}`, `row_bytes_p99` per grain |
| `_check` | `{tables: [TableRef], depth: shallow\|standard\|deep}` | per table: status, per-column/container/partition statuses, checks `[{name, ok, detail, hint, column?, partition?}]`, fingerprint, signature, confirmed facts, vocab snapshot ids, key-check record |
| `_witness` | see §11.6 | see §11.6 |
| `_serve` | `{table, verb: lookup\|find\|search\|members\|count\|aggregate\|similar\|expand, predicate, columns, order, limit, limit_grain, group_by, distinct, explode, carry, rename, split, nest, sections, anchor}` | `{rows, total, truncated, key_columns, grains, excluded_unknown, excluded_not_applicable, served_by, sections}` |
| `_build_index` | `{source, id_type}` or `{table, access_path}` | `{path, rows, fingerprint}` (resolver sidecar or row-group value index) |
| `_resolve_remote` | `{source, id_type, values}` | `{resolutions: [...], existence: exists\|absent\|unknown}` |
| `_vocab` | `{table, column}` | `{values (storage-typed, rendered), fingerprint}` |

The child learns the catalog location from `VBT_DATA_SETTINGS` (JSON: descriptors and overlays
directories, cache directory, budgets) placed in its environment by `DataGateway.extra_servers()`.

### 11.9 Generic guard (servers or tools without a reviewed binding)

- An explicit not-found message (`_EMPTY_LOOKUP` or overlay `not_found_when`) → `not_found` error,
  evaluated before legacy envelopes.
- Legacy envelopes and `isError` otherwise keep today's semantics (`source_error`).
- A structural empty (`count == 0`, `num_results == 0`, empty top-level list) → `empty_unverified`:
  a success that **cannot be cited**, either as support or as absence.
- Parameters named `*_id`, `gene*`, `pmid`, `nct*` and similar are linted in observe mode with every
  identifier plugin; a conflict ("looks like hgnc_symbol; parameter suggests ensembl_gene") becomes a
  `_vbt.notes` warning.
- Result-size soft cap (after spilling the full payload), crash/OOM classification (stdio only),
  timeouts, minimal provenance (server, tool, args sha256, result sha256, `retrieved_at`).
- HTTP servers (`url:`) get everything except the process memory limit.

### 11.10 Modes and profiles

`data.gateway.mode`: `off` (bridge built without a gateway), `observe` (decisions traced, upstream
returned), `enforce`. `data.gateway.enforce_servers` stages enforcement per server.
`data.gateway.profile: fidelity` (paper-replication runs) keeps resolution, not-found, empty, readiness,
leakage and memory rules and returns upstream outputs or nothing: tools whose phase-1 mode is
`derived` are served `pass` behind the witness (upstream answers when the witness agrees) and refused
with an error naming the defect only on a contradiction; repairs become errors (rev 2: revision 1 turned
every derived tool into an error, which left no usable Tahoe tool). The mode and profile used for each
call are recorded in provenance.

---

## 12. Error and result contract returned to agents

### 12.1 Errors

A gateway error is `GatewayError(ToolFailure)` with `kind` and `payload`; the agent sees `Error:
<json>` and `is_error=True` exactly as today. The envelope extends the existing
`{"status": "tool_error", ...}` envelope (`mcp_bridge.py:78`):

```json
{"status": "tool_error", "contract": "vbt.data/1", "kind": "not_found",
 "tool": "mcp__target__get_target_info", "argument": "target_id", "value": "PCSK99",
 "id_type": "ensembl_gene", "accepts": ["ensembl_gene", "hgnc_symbol"],
 "tried": ["ensembl_gene: not an Ensembl ID", "hgnc_symbol exact: none", "casefold: none",
           "previous: none", "alias: none"],
 "suggestions": [{"id": "ENSG00000169174", "label": "PCSK9", "why": "edit distance 1"}],
 "source": "open_targets@25.09", "table": "target", "citable": false,
 "retryable": "after_fixing_input",
 "next": "Fix the identifier (see suggestions) or call mcp__data__resolve.",
 "instruction": "No record with this identifier exists in this source. This is not evidence about biology and must not be cited."}
```

| kind | raised when | side | `failures` data-source failure | retryable |
|---|---|---|---|---|
| `not_found` | identifier absent from its universe; explicit not-found from an unbound tool; phantom items (W6). Subkinds: `obsolete` (replacement named), `combination_not_profiled` (each value exists, the tuple was never profiled; profiled values listed), `not_measured_in` (fragments listed) | model | no | after fixing input |
| `ambiguous` | several candidates (listed with labels and `disambiguate_with` values) | model | no | yes, with a candidate |
| `invalid_argument` | unknown enum/vocabulary/selector value, bad limit or list size, wrong identifier kind or bad syntax, escaping path, unquotable value, unparsable SOMA filter | model | no | yes |
| `unsupported_combination` | exclusive or overriding arguments together; threshold on a `comparable_within` measure with the group unfixed; `include_descendants` on a propagated column | model | no | yes, as separate calls |
| `incomplete_key` | rows would be merged, aggregated or ranked across a scope dimension with several values (values listed). Subkind `incomparable_order`: a global top-k across incomparable groups | model | no | yes, with the dimension |
| `unsupported_filter` | a threshold outside the scale; an encoding not confirmed in this release; an unbound filter argument with a non-default value | model | no | after changing the filter |
| `insufficient_resolution` (rev 2) | a list argument resolved below `min_resolved_fraction` (unresolved and ambiguous elements listed) | model | no | after fixing the list |
| `not_ready` | a table the tool reads failed readiness (table and check named) | data | yes | no |
| `too_large` | memory estimate over budget; ranking unverifiable (`unranked_truncation`); learned refusal | data | yes | with narrower arguments |
| `quarantined` | `serve: block` (reason and alternative named) | data | yes | no |
| `tool_defect` | witness contradiction without repair; wrong entity | data | yes | no |
| `oom` | memory error inside a server or memory kill | data | yes | no (narrow) |
| `server_crashed` | transport failure after the allowed retry | data | yes | once |
| `source_error` | upstream error envelope, nested error, HTTP 5xx | data | yes | maybe |
| `service_unavailable` | the data child is down; tools needing it cannot be guarded | data | yes | later |

`MODEL_SIDE_KINDS = {not_found, ambiguous, invalid_argument, unsupported_combination, incomplete_key,
unsupported_filter, insufficient_resolution}`. All kinds are `is_error`, so none can be cited.

Payloads are fixed per kind (rev 2; the correctness tests assert these fields):

| kind | payload fields |
|---|---|
| `not_found` | `argument, value, id_type, accepts, tried[], suggestions[{id, label, why}], source, table, subkind?, replacement?, profiled_values?` |
| `ambiguous` | `argument, value, candidates[{id, label, via, <disambiguate_with columns>}]` |
| `invalid_argument` | `argument, value, valid_values` (the full vocabulary when ≤ `data.derive.enum_max`, else the 10 nearest by casefold and edit distance) `, vocabulary_ref?` (for `mcp__data__vocab`) `, near[], looks_like[], items?[{index, value, reason}]` |
| `incomplete_key` | `dimension, argument, values` (sorted, JSON-typed, rendered in the storage type) `, unit?, subkind?, retry_with` |
| `unsupported_filter` | `argument, column, reason` (`scale`, `unconfirmed_encoding`, `unbound_argument`) `, confirmed_range?` |
| `insufficient_resolution` | `argument, requested, resolved, unresolved[], ambiguous[], outside_universe[], min_resolved_fraction` |
| `not_ready` | `tables[{name, column?, partition?, check, detail, hint, acquire?}]`; `acquire` (when acquiring the files fixes it): `{command, source, table, release, bytes, files, prepare, mode, licence?, login?, policy?, decision?}` |
| `tool_defect` | `check (W1..W6), witness{total, topk?}, returned, defect_ids[]` |

### 12.2 Success: the `_vbt` header

```json
{"_vbt": {"v": 1, "status": "partial", "source": "open_targets@25.09", "tables": ["known_drug"],
          "key": ["drugId", "targetId", "diseaseId", "phase", "status", "urls"],
          "returned": 20, "total": 61, "total_method": "witness_scan", "truncated": true,
          "order": "phase desc (verified)", "grains": {"drug": {"returned": 4, "total": 13}},
          "resolved": {"target_id": "PCSK9 -> ENSG00000169174 (label_exact:approvedSymbol)"},
          "excluded_unknown": {"phase": 3}, "served_by": "upstream",
          "notes": ["rows are trial-source records; 20 rows cover 4 drugs"],
          "cite": "partial: cite as 'top 20 of 61'", "prov": "dp_3f9c1a2b7e01"},
 "success": true, "count": 20, "drugs": ["..."]}
```

For `empty`:

```json
{"_vbt": {"v": 1, "status": "empty", "source": "open_targets@25.09",
          "tables": ["openfda_significant_adverse_target_reactions"], "returned": 0, "total": 0,
          "coverage": "unknown",
          "coverage_statement": "FAERS target signals exist only for targets of marketed drugs; absence is not evidence of safety.",
          "cite": "not citable as support; absence claim requires covered coverage", "prov": "dp_..."},
 "success": true, "target_id": "ENSG00000141510", "count": 0, "adverse_events": []}
```

Header fields added in revision 2: `excluded_not_applicable`, `excluded_negated`, `withheld`
(`{leakage, phantom, duplicates}`), `family_rows`, `resolution_summary`, `not_found_items`,
`trimmed`, `undefined` (anchor arguments without a value in the table), `evidence` (the table's
`evidence_nature` caveat, present for every status), `hash_seed`, and `grains` as `{grain: {returned,
total}}`. `coverage` takes the values `covered`, `unknown`, `not_covered`, `partial_unknown` and
`censored` (§6.3); `order` names the group when ranking is `within` a group ("score desc within
sourceDatabase (verified)").

Text results (non-JSON) get the header as the first line:
`[vbt-data dp_7c1e0942 · ok · open_targets@25.09/known_drug · 20 of 61 · ranked phase desc]`.

---

## 13. Readiness scoped to each tool

- **Units of readiness (rev 2).** Readiness is computed per table **and** per column, nested container,
  item table and partition. A table's status is the worst status over its parts; a call's readiness is
  the worst status over the parts it reads (its `reads.columns`, the columns its arguments and result
  fields bind, the containers it returns, the partitions its partition predicate can include, the
  tables of coverage universes, and the resolver indexes of its id_types). One drifted field
  (`homologues.speciesId` str vs int) no longer takes down the 16 target tools that never read it,
  and one europepmc `.part` file no longer blocks `query_evidence(datasource_id="chembl")`.
- **Statuses**: `ready | missing | partial | schema_drift | encoding_drift | key_violation | stale |
  unreachable | plugin_unavailable | awaiting_producer | unbound`. `partial` is not servable for the
  affected partitions: a call that can include them is `not_ready`, or `partial` with
  `_vbt.coverage.unavailable_partitions` when the binding allows it; a call whose partition predicate
  excludes them is ready. `awaiting_producer` (a table that a tool writes during the run) and `unbound`
  (a declared table no tool reads) never appear as failures in `vbt doctor`.
- **TableReadiness** is computed by the data child's `_check` and cached harness-side in
  `${data.cache_dir}/readiness/<source>.json` keyed by `(descriptor sha256, table signature)`. Depths:
  - **shallow** (every turn, harness-side, no subprocess): stat-only `signature` per table directory and
    manifest compared with the cache. Today `require_ready(per_turn=True)` walks all files with
    `rglob` on every turn.
  - **standard** (session start, and whenever a signature changes):
    - **R1** layout present; no partial files; every listed fragment exists; partition directories
      equal the declared vocabulary when `expect: declared` (missing value → that partition not ready;
      extra value → `schema_drift`).
    - **R2** manifests, attributed to tables by path prefix under `TableSpec.path`: missing entries make
      that table `missing`, size or hash mismatches `partial`; release mismatch and `complete: false`
      are source-level failures; an absent manifest is a warning (`release_verified: false`) unless
      `required: true`; `inline` manifests are checked by whole-file hash when the signature changes;
      `checks` compare manifest row counts and filters with the data.
    - **R3** every listed fragment's footer (or header, or stats scan for formats without footers) is
      readable; an unreadable fragment fails its partition (I14).
    - **R4** declared columns present with role-compatible types (`roles.ARROW_COMPAT`: `large_*`,
      int32 counts and string storage with `parse` or `stored_as` accepted); optional columns missing →
      warning; undeclared physical columns → drift finding under `strict`; every header of a header
      axis parses.
    - **R4b** (rev 2) a sample of universe keys (all keys for small tables) satisfies the identifier
      plugin's canonical form after `configure(options)`, and `normalize` is idempotent on it; failing
      prefixes are named (the `ot_disease` prefix list missed 29.5% of real `disease.id` values).
    - **R5** key columns that are not `nullable` have no nulls: footer null counts for flat columns,
      a projected-leaf scan (definition levels) for nested paths, because footer `null_count` counts
      empty and null lists too (VERIFIED: 3 for one null leaf).
    - **R5b** (rev 2) key uniqueness: `check: full` streamed at this depth for tables at or under
      `key_check_full_max_rows`; `sampled` prefix blocks otherwise; item keys per container;
      `alternate_keys`.
    - **R6** `verified: false` facts confirmed or refuted: scales from row-group min/max; encodings,
      flags, vocabularies and aliases from distinct-value snapshots (every declared code observed,
      values ⊆ codes ∪ {null}, NaN counted separately, never from min/max alone); column existence;
      nested null versus empty list counts (`null_means`/`empty_means`, chemicalProbes coverage);
      literal constraints from min/max; matrix value statistics from a bounded, seeded
      `sample_values` scan (integrality, range, NaN share) for formats without statistics.
    - **R7** vocabulary snapshots for category and scope columns: from row-group `min == max` statistics
      when every row group is single-valued (Tahoe drug, plate, dose), else a bounded scan within
      `data.readiness.vocab_budget_bytes`; placeholders and `missing_values` excluded; strings
      `'nan'`, `'None'` and `''` in categorical columns produce a warning; mirrored columns checked for
      equality per fragment.
    - **R8** sentinels with the extended expectation grammar (present found with expected fields,
      absent missing); remote sentinels through `via`/`remote_probe` within the request budget.
    - **R9** (rev 2) referential samples: every `ref` column against its universe (and its stored
      form), membership set ids against the set universe, hierarchy targets against the universe,
      composite refs; findings are errors, or warnings with counts under `integrity: partial`. A
      mismatch between stored forms and the universe marks that id_type's index not ready for bindings
      on that column and names the unmatched values (Tahoe `'Erdafitinib '`).
    - **R10** (rev 2) relation and domain invariants on a sample: relation constraints
      (`leaf == (len(children) == 0)`, `count == len(rows)`, `norm == l2(vector)`, norms finite and
      > 0); hierarchy closure (`ancestors == closure(parents)`), inverses (`children` vs `parents`),
      irreflexivity and acyclicity; edge orientation facts (reverse-edge sample for `orientation:
      both|canonical`); censoring consistency for `event_of` pairs (times without status, status
      without time, non-positive times counted: 9, 35 and 4 rows in the authors' LUAD file);
      member resolution fraction against `min_resolved_fraction`; `determined_by` functional
      dependencies.
  - **deep** (`vbt datasource check --deep`, offline): full key uniqueness for tables above the
    standard-depth limit (bounded memory), full foreign-key passes, relation invariants on all rows
    (direct ⊆ indirect associations).
  - **remote** (live sources, TTL `data.readiness.remote_ttl_s`): reachability, release or as-of, a
    sentinel record through the declared tool call, within the request budget.
- **Partial downloads**: a stray `*.part` marks only that table's affected partition `partial`, never
  the whole release.
- **ToolReadiness** = conjunction over the parts the call reads (above), server state and
  admissibility (§14). A tool whose only problem is a missing optional section is ready with that
  section marked unavailable. An accepted id_type whose index is not ready degrades the call (that
  kind is not tried, with a note) instead of blocking it, unless the value can only resolve through
  that kind (rev 2).
- **Completeness of `reads` (rev 2).** A shipped-config test AST-scans each registered upstream tool,
  transitively through same-module helper calls (the profile calls `search_known_drugs`,
  `get_target_adverse_events` and `get_mouse_phenotype`), for `get_dataset`/`get_arrow_dataset`
  string literals and fails when a dataset is missing from the binding's `reads`; argument-dependent
  reads use `ReadSpec.when`. A per-server fault test deletes each table in turn on the fixture and
  requires every tool to return `not_ready` or an error, never `ok` or `empty`.
- **Call-time enforcement**: the gateway raises `not_ready` for unready calls; listings keep the tools
  with an `UNAVAILABLE:` prefix only when every call of the tool would be unready, so `agents.yaml`
  allowlists stay stable.
- **Preflight integration** (`src/vbt/preflight.py`): `check_reference_data` (L357), when
  `data.enabled`, runs the data child's check entry point once as a subprocess
  (`<mcp_python> -E .../service/server.py --check --json`) and returns one `CheckResult(kind="data",
  scope={"source", "table", "column"?, "partition"?})` per finding, while keeping the legacy aggregate
  labels (`OPEN_TARGETS_DATA_PATH`, `Tahoe`) that existing tests match; with the data layer enabled the
  aggregate result is `ok` unless no granted data tool is ready, so `degraded_servers` no longer names
  every OT server when one table is missing. `degraded_tools(config, results) -> {tool: reason}`
  replaces the label-substring logic; `degraded_servers` (L608) returns only servers whose **every**
  tool is unready. `require_ready` (L577) blocks on credentials, and on data only when no data tool
  granted to any agent is ready and `allow_missing_data` is false. Preflight also probes the data
  child's imports and every plugin's `requires`.
- **Tahoe**: readiness covers exactly what tools read (`tahoe_permissive_padj010.parquet`,
  `metadata/drug_metadata.parquet`, `metadata/cell_line_metadata.parquet`, `metadata/gene_metadata.parquet`
  for gene crosswalks) and requires `preparation_manifest.json.complete == true`; the unread
  `pseudobulk_de_significant/` and `pseudobulk_de_high_quality/` no longer gate anything; a missing
  Tahoe file leaves the five DepMap tools ready. The fingerprint of the multi-GB single file is the
  manifest sha256 + file size + footer sha256 (hashing the footer is cheap); the parsed footer
  (~90 MB EST at ~62,600 row groups) is cached per fingerprint in the data child.
- **Smoke** (`vbt doctor --smoke`): positive controls from `sentinels.present`, run through the gateway;
  a smoke passes only if the sentinel key comes back with its key columns populated, so an empty
  success fails, and a negative control must return `not_found` (for cBioPortal, through
  `exists_when`, so a phantom sample fails the smoke). The old `SMOKE_CALLS` (L73) that force multi-GB
  loads remain available as `--smoke=upstream`, admission-controlled.
- **Prompt**: `agents._unavailable_text` (L299) lists unready tools grouped by table and column, e.g.
  "functional_genomics: Tahoe tools unavailable (tahoe_100m.de_permissive missing); DepMap tools ready."
- **In-process readers**: tables read only by harness modules (Case 1 pipelines) are reported as
  "read in-process; not guarded until phase 2 (F14)" so the gap is visible (N8).

---

## 14. Memory limits

### 14.1 Where memory blows up today

| Where | Mechanism | Control |
|---|---|---|
| Upstream server, first call | whole table to pandas: Arrow table plus pandas copy at peak; cached forever, no lock | admission, cold-call lock, process limit |
| Upstream server, over a session | resident tables accumulate | residency ledger, recycle-to-evict |
| Upstream fallbacks | genetics `except` → full pandas load of `variant`/`interval` on any error | process limit (cannot be predicted) |
| Concurrent first calls | no lock in `get_dataset` | cold-call serialisation per server |
| Harness | a huge JSON-RPC message held in memory (339 KB VERIFIED; GBs possible) | inflation caps bound result size; gateway soft cap; optional relay (P4) |
| Census pulls | fetch first, size check after (`single_cell_mcp/tools.py:483-490`) | count-first admission (P4) |
| Data child | witness and derived scans | two-pass leaf-projected scans with byte budgets, sidecar indexes, own process limit |
| Upstream nested loads (rev 2) | pandas creates one Python object per nested item: footer bytes understate the peak about 9× on a three-level fixture (VERIFIED) | `num_values` × object overhead in the estimate (§10.4) |
| Dense reads of HDF5 (rev 2) | `X.toarray()` and `read_h5ad` without `backed` densify n_obs × n_vars (Zenodo cohorts, Census outputs) | densification term in the estimate; in-process client from phase 2; workspace process limit (F14) |
| Remote pulls (rev 2) | `get_clinical_data` downloads a whole study at both levels (page size 10,000,000) | `size_from` count-first admission in phase 1; grain-aware truncation |

Measured RSS/Parquet ratios from the maps (compressed Parquet basis): target about 49–60×,
known_drug about 85×, evidence about 51×, literature about 43×, indirect associations about 24×;
target_essentiality about 1.3 GB per 10% part (10–13 GB EST in full). evidence (hundreds of GB EST)
and literature (about 90 GB EST) can never be materialised; their bindings are `serve: derived`.

### 14.2 Process limits (`launch/reaper.py`)

The launcher is one stdlib-only file executed by path with the harness interpreter
(`sys.executable -E <project_root>/src/vbt/datalayer/launch/reaper.py --limit-mb N --status <path>
--server <name> -- <command> <args...>`). It never imports `vbt`.

1. It forks. The **child** sets `RLIMIT_DATA` (Linux ≥ 4.7 counts brk and private writable mappings,
   so pandas and Arrow heaps are bounded without the false failures `RLIMIT_AS` causes with allocator
   and thread-stack reservations) and `exec`s the original command, inheriting fds 0, 1 and 2.
2. The **parent** (the reaper) closes its copies of fds 0 and 1 (so pipe EOF reaches the bridge when
   the child dies), forwards SIGTERM and SIGINT to the child's process group, polls
   `/proc/<pid>/status` every 250 ms, writes `<log_dir>/<server>.status.json` atomically each second
   (`pid, rss_mb, peak_rss_mb, limit_mb, containment`), and after `waitpid` writes one line
   `VBT_CHILD_EXIT {"pid":..,"code":..,"signal":..,"maxrss_kb":..,"reason":..}` to stderr (the server
   log), then exits with the child's status.
3. Child environment additions: `ARROW_DEFAULT_MEMORY_POOL=system` (allocation failures surface as
   `MemoryError`/`ArrowMemoryError` inside the tool, so FastMCP returns `isError` and the server
   usually survives), `MALLOC_ARENA_MAX=2`, `PYTHONHASHSEED=0`, and `PRELOAD_MCP_DATA=0` (upstream
   already defaults to `"0"`; forcing it is defensive only).
4. **The hash seed must take effect (rev 2).** Every server is launched with `python -E`
   (`configs/mcp_servers.yaml`), and `-E` makes CPython ignore `PYTHONHASHSEED`: with the seed set,
   `get_interaction_network` still returned edge counts 2, 2, 4, 4, 3, 4 over six runs (VERIFIED). For a
   Python child (interpreter basename `python*`), the reaper therefore removes `-E` and `-I` from argv
   and instead builds the child environment explicitly: it drops every `PYTHON*` variable and adds back
   only an allow-list (`PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1` when `-I`
   was requested, `PYTHONUSERBASE` when `src/vbt/envpolicy.py` sets it). This keeps what `-E` was for
   (the environment cannot redirect imports through `PYTHONPATH` or `PYTHONHOME`) while making the seed
   effective. The reaper writes `hash_seed` and the stripped flags into the status file; the data child
   reports `sys.flags.hash_randomization` in `_check`; provenance and `pinned["data"]` record them. A
   launcher test runs a fixture server twice and requires identical set order. Correctness never
   depends on the seed: gateway output is sorted by the full canonical key (I13), and the network tools
   stay blocked until the phase-3 traversal orders its frontier by key.
5. Phase 4 adds cgroup v2 (`memory.max`, `memory.oom.group`) or cgroup v1 where delegated, and an RSS
   watchdog fallback that kills the child at `limit − max(512 MB, 5%)` on hosts without delegation.
   (VERIFIED, R3) This host mounts the cgroup v1 `memory` controller at `/sys/fs/cgroup/memory`, and the
   session's own cgroup is writable as root, so `--containment cgroup` selects `cgroup_v1` here: a child
   over its 200 MB cgroup was SIGKILLed and labelled `cgroup_oom_kill`. `RLIMIT_DATA` stays set under
   `cgroup` and `watchdog`; a server whose libraries reserve far more address space than they touch (the
   Census server's TileDB reads failed with `std::bad_alloc` under `RLIMIT_DATA` of 4,500, 12,000 and 40,000 MB
   while under 3 GB was resident) declares `limit_kind: rss` in `configs/mcp_servers.yaml`: no data limit, contained
   by its resident memory (cgroup, else the watchdog).

Per-server `mem_limit_mb` comes from `configs/mcp_servers.yaml` or defaults to
`data.memory.default_server_mb`; the data child defaults to `data.service.mem_limit_mb`. Every `auto` scales with
`plan`, the memory the harness may plan with (`memory/sizing.py`): the smaller of MemTotal and the memory cgroup
limit of this process and its ancestors, else `data.memory.host_mb` when it is a number, `$VBT_HOST_MEMORY_MB`
overriding the probe. The rules (each with a floor; a number stays as configured):

| Setting | `auto` | Floor |
|---|---|---|
| `data.memory.host_budget_mb` | 0.75 × plan − max(`harness_reserve_mb` 2,048, 5% of plan) | 1,024 |
| `data.memory.default_server_mb`, a server's `mem_limit_mb: auto` | 0.8 × host budget | 2,048 |
| `data.service.mem_limit_mb` (the data child) | 5% of plan, at most 32,768 | 3,000 |
| `data.service.max_resident_mb` | 2/3 of the data child's limit | |
| `data.witness.max_scan_bytes`, `max_inflate_bytes`, `max_key_set`, `repair_max_bytes`, `data.readiness.vocab_budget_bytes` | the shipped value × data child / 3,000 | the shipped value |

`DataSettings` resolves the data child's and the budgets' `auto` when the settings are read (the typed settings
always hold numbers; `raw` keeps `auto`); the launcher resolves the server limits at launch. On a 16 GB host the
values are 10,240 / 8,192 / 3,000 MB, on 512 GB 367,002 / 293,601 / 26,214 MB. A `too_large` over the limit names
the host (`host_mb`), `limit_source` (`auto` or `configured`) and, for `auto`, `host_mb_needed`. Containment
defaults to `rss` (`data.memory.limit_kind`; `DEFAULT_LIMIT_KIND`): a memory cgroup, else the RSS watchdog, with no
`RLIMIT_DATA`, which works for every server (5. above); `rlimit_data` is opt-in per server or host.

### 14.3 Admission (`memory/admission.py`, `memory/ledger.py`)

```python
async def admit(self, server, contract, args, route) -> Admission:
    if route != "upstream":                                            # rev 2: derived/witness work is the data child's
        return Admission(cold_tables=(), lock=None)
    cold = [t for t in contract.full_table_reads if t not in self.ledger.resident(server)]
    need = sum(self.est.peak(t) for t in cold) + self.est.transient(contract, args)
    if self.est.peak_all(contract.full_table_reads) * SAFETY > self.limit(server) - self.baseline(server):
        raise GatewayError("too_large", permanent=True, need_mb=..., limit_mb=..., alternative=contract.alternative)
    if (server, frozenset(cold)) in self.learned_refusals:            # killed before at this size
        raise GatewayError("too_large", learned=True, ...)
    if self.resident_mb(server) + need * SAFETY > self.limit(server):
        if self.can_recycle(server):
            await self.bridge.recycle(server, wait_s=self.recycle_wait_s)   # empties the forever-cache
        else:
            raise GatewayError("too_large", retryable=True, ...)
    return Admission(cold_tables=tuple(cold), lock=self.cold_locks[server] if cold else None)
```

- **Cold-call serialisation**: at most one call per server that will load tables is in flight;
  warm calls run concurrently. This prevents double loads and makes OOM attribution unambiguous
  (the culprit is the single cold call).
- **Ledger**: per server *generation* (reset on restart or recycle); resident RSS comes from the
  reaper status file when present, else from estimates.
- **Recycle** restarts an idle server (waits for in-flight calls up to `recycle_wait_s`), emits
  `mcp_recycle`, and does not count toward `max_restarts`. A thrash guard refuses after
  `max_recycles_per_10min`.
- **Host budget** (phase 4): total RSS across servers capped by `data.memory.host_budget_mb`, evicting
  the least-recently-used idle server.

### 14.4 Crash and OOM policy (`memory/crash.py`)

| Signal | Classified as | Retry | Effect |
|---|---|---|---|
| `isError` text matching memory signatures | `oom` | no | mark `(server, cold tables)` as a learned refusal; proactive recycle (upstream may hold a partial cache) |
| transport error and `VBT_CHILD_EXIT` with signal 9 or rc 137, or `reason: memory_limit` | `oom` (`oom_killed`) | **no** (today the bridge re-runs the same query) | lazy restart against `max_oom_kills` (default 3) instead of `max_restarts`; learned refusal |
| transport error otherwise | `server_crashed` | once (today's behaviour) | unchanged |
| timeout | `source_error` (timeout) | no | today's message plus a narrowing hint derived from roles |

The bridge waits up to 500 ms for the exit marker after a transport error (the pipe can close before
the reaper writes it). After any failed retry the session is torn down and the state set to
`broken`, so the next call restarts instead of hitting the dead pipe (VERIFIED bug).

### 14.5 Result size and data child budgets

`data.memory.max_result_bytes` (2 MB soft) is the gateway cap on what the **model** sees: oversized
results are shrunk structure-aware with `truncated: true`, never splitting a declared grain (a
patient's samples stay together), after the full payload has been handed to the runtime spill
(rev 2). Single-record results never lose fields; long fields are truncated with markers. Limit
inflation is capped by `max_inflate_rows` and `max_inflate_bytes`. Remote reads with `size_from` or
`count_via` are admitted count-first: `total × est_row_bytes` over the cap is `too_large` before the
call. The data child enforces `data.witness.max_scan_bytes` (decoded bytes after
pruning) and `data.service.max_resident_mb`; over budget it answers `total_method: unknown`
(witness) or `too_large` (derived), never a partial answer presented as complete. `max_resident_mb` bounds the
readiness key pass (R5b on Arrow arrays): rows are hashed one key part and one row group at a time, only the rows
whose hash repeats are counted exactly, and over the budget that exact count runs in hash-partitioned passes that
each fit. Holding every part's values whole had taken the data child to 2,105 MB on the 14.5 M rows of 25.09
`interaction` (seven parts), the floor of every session check; the bounded pass peaked at 582 MB (a process of its
own, imports included) and that table's whole standard check at 692 MB in the data child (VERIFIED, Wave C).

### 14.6 Budgets on this host (EST; `vbt datasource estimate` prints them)

This host has about 15–16 GB RAM, 4 CPUs and cgroup v1 on tmpfs (measured). Upstream full loads of
`evidence`, `literature`, `variant` and `interaction_evidence` are not admissible on any host and are
served derived. `target_essentiality` (10–13 GB EST) is not admissible on a 16 GB host, so the four
DepMap aggregate tools return `too_large` naming the native alternative until phase 3 (F16); the data
child reads it through `leaf_projection` (a `geneEffect` threshold decodes the screens leaves of
matching row groups, not the whole `geneEssentiality` column). The `target` table is cached separately
by the target, drug and pathway servers (about 3.5 GB each EST); phase 1 serves
`pathway.get_gene_pathways` derived to keep one upstream copy fewer. Sidecar row-group indexes (Tahoe
`gene_name`, evidence `targetId`/`diseaseId`) are a few MB to a few hundred MB EST in `data.cache_dir`
and are built once per fingerprint.

---

## 15. Provenance, claims, verification and pinning

### 15.1 Record `vbt.dataprov/1`

Written by the runtime to `<run>/logs/data_provenance/<tool_use_id>.json`; a summary (`prov`, status,
**coverage** and coverage statement, source@release, tables with fingerprints, returned/total,
`row_keys_sha256`, `evidence_nature`, `leakage.risk`) goes into the trace `tool_end.data_provenance`
(rev 2: revision 1 never carried coverage to the claims checker).

```json
{"schema": "vbt.dataprov/1", "id": "dp_3f9c1a2b7e01", "tool_use_id": "toolu_01...",
 "tool": "mcp__drug__search_known_drugs", "server": "drug", "mode": "enforce", "gateway_version": "1.0",
 "served_by": "upstream",
 "source": {"name": "open_targets", "release": "25.09", "release_verified": true, "descriptor_sha256": "...",
            "overlay_sha256": "...", "manifest_sha256": "...", "scope_versions": {}},
 "plugins": {"identifier/hgnc_symbol": "1.0", "statistic/clinical_phase": "1.0"},
 "tables": [{"name": "known_drug", "layout": "sharded_dir", "fragments": 2,
             "fingerprint": "fp1:manifest:sha256:...", "access": "upstream_full_table",
             "partitions_read": null, "partition_fingerprints": null,
             "lineage": null, "key_check": {"method": "full", "at": "2026-10-06T09:12:00Z"}}],
 "request": {"args_raw": {"target_id": "PCSK9", "limit": 20},
             "args_sent": {"target_id": "ENSG00000169174", "limit": 61},
             "resolutions": [{"arg": "target_id", "raw": "PCSK9", "matched_id_type": "open_targets:hgnc_symbol",
                              "canonical_id_type": "open_targets:ensembl_gene",
                              "canonical": "ENSG00000169174", "rule": "label_exact:approvedSymbol",
                              "family": null, "hops": [], "index_fingerprint": "..."}],
             "expansions": [], "scope": {}, "selector": null},
 "result": {"status": "partial", "returned": 20, "total": 61, "total_method": "witness_scan",
            "coverage": "unknown", "coverage_statement": "ChEMBL-curated ... not 'undruggable'.",
            "truncated": true, "order": {"by": "phase", "direction": "desc", "within": [], "verified": true},
            "key_columns": ["drugId", "targetId", "diseaseId", "phase", "status", "urls"],
            "row_keys": [["CHEMBL...", "ENSG00000169174", "EFO_...", 4.0, "...", "..."]],
            "row_keys_complete": true, "row_keys_sha256": "...", "output_sha256": "...",
            "output_rows_sha256": "...", "computed": null, "statistics": null,
            "transforms": ["limit_inflation 20->61", "sort phase desc", "cut 20"]},
 "checks": [{"name": "existence", "ok": true}, {"name": "witness_count", "ok": true},
            {"name": "topk_order", "ok": true}],
 "memory": {"estimate_peak_mb": 900, "admission": "warm"},
 "as_of": null, "retrieved_at": "2026-10-06T09:14:03Z", "leakage": null, "evidence_nature": null,
 "upstream": {"commit": "<submodule sha>", "server": "drug_mcp", "hash_seed": 0, "flags_stripped": ["-E"]},
 "t_ms": {"prepare": 18, "call": 412, "finish": 9}}
```

`row_keys` are stored in full up to `data.provenance.row_keys_max` (10,000), beyond which only the
hash and a sample are kept (`row_keys_complete: false`). Row keys use the canonical encoding of §6.3,
per declared grain when the result has levels (patient keys beside sample keys), and include
relation pairs for relation results (`(term, descendant)`). Revision 2 also records: `expansions`
(`{arg, term, relation, predicates, n_terms, terms_sha256, hierarchy_fingerprint}`, never the
expanded list itself; `data.resolution.max_expand` caps fan-out with `too_large`, subkind
`expansion`), `computed` (`{metric, anchor_keys, input_row_keys_sha256, values_sha256}` for similarity
scores), `statistics` (`{test, correction, family_m, universe: {def, n, fingerprint}, query:
{requested, resolved, in_universe}, propagation: {mode, dag_fingerprint}}` for enrichment), the
source's per-scope versions (`interactionResources.databaseVersion` per `sourceDatabase`), the
partitions read with per-partition fingerprints (so a replay of a `datasource_id=chembl` call ignores
europepmc refreshes), and live-record versions (`KeySpec.version`). Live sources set `as_of` from the
payload when `ResultSpec.as_of` or `release.resolve` names it, else from the call time, and always
`retrieved_at`; with a ceiling configured, `leakage: {"ceiling": "...", "withheld": n, "risk":
true|false}`. Without a manifest, files up to 64 MB get `fp1:sha256:` content fingerprints and larger
files `fp1:stat:` plus a footer hash.

### 15.2 Flow

`Runtime._execute` adds `result_status`, `error_kind` and the provenance summary to `tool_end`;
`session.Run._index_event` copies `result_status`, `error_kind` and `prov` into the live call index;
`audit/provenance.Provenance._build` keeps them in `to_dict()["tool_calls"]`. Old traces without the
fields parse with `result_status: "unknown"`.

### 15.3 Claims (`src/vbt/audit/claims.py : check_evidence`, tool_call branch)

The record_claims evidence schema (`src/vbt/tools/provenance.py`, L120) gains optional
`supports: "presence" | "absence"` (default presence) and `row_key`. The `kind` enum (L68) is
unchanged, so `test_tool_schemas_document_citation_fields_and_kinds` still holds.

| cited call | supports = presence | supports = absence |
|---|---|---|
| error (any kind) | problem (as today) | problem |
| `ok` | verified; entry copies `prov`, source, release, table fingerprints | verified with a warning ("absence claim cites rows; the note must say which rows show it") |
| `partial` | verified with warning "N of M rows"; problem if the claim says "top" and order is unverified | verified with warning |
| `empty` | **problem**: "an empty result cannot support a positive finding" | verified as `evidence_status: "absence"` **only if** coverage is `covered`; `censored` is accepted only when the claim text carries the censor statement; `unknown`, `not_covered` and `partial_unknown` → problem naming the coverage |
| `empty_unverified` | problem | problem |
| any status with `leakage.risk: true` | problem: "cites data that may postdate the evidence ceiling" | problem |
| any status from a table with `evidence_nature` | warning naming the caveat; problem when `confidence` is `strong` and this is the claim's only evidence | as for its status |
| no status (old trace or non-gateway tool) | as today plus warning "no data-layer status" | warning |

`check_evidence` copies `prov`, `coverage`, `coverage_statement`, source, release and table fingerprints
from the call record (rev 2: `Run._index_event` and `audit/provenance.Provenance._build` carry
`coverage`; verify rebuilds the same index from audit provenance, so record_claims and `vbt verify`
decide identically). An enrichment result that returns every tested set with a declared family and
universe has `coverage: covered`, so "not enriched at FDR 0.05 (m = 50, N = 19,000)" is citable as an
absence for a tested set (phase 3). `claim_stats` adds an `absence` count. From phase 5, `row_key` is
checked against the stored row keys.

### 15.4 Verify (`src/vbt/verify.py`)

New problem kinds: `empty_result_cited` (INCOMPLETE; phase 1: an empty or `empty_unverified` result
cited with `supports` other than `absence`, or with coverage other than `covered`/`censored`-with-statement),
`degraded_run` (reads
`MANIFEST.degraded`, which verify ignores today; phase 1), `data_version_drift` (pinned table
fingerprint differs from the current one; warning, INCOMPLETE under `--data`; phase 5),
`replay_mismatch` (phase 5). None joins `INTEGRITY_KINDS`, because reference data lives outside the
run directory.

### 15.5 Pinning (`src/vbt/pinning.py`)

`pinned["data"]` (fail-soft, redacted, built from `gateway.pinned()`):

```yaml
data:
  mode: enforce
  profile: safe
  gateway_version: "1.0"
  catalog_sha256: "..."
  descriptors: {open_targets: "sha256:...", tahoe_100m: "sha256:..."}
  overlays: {drug: "sha256:...", target: "sha256:..."}
  plugins: {"format/parquet": "1.0", "identifier/ensembl_gene": "1.0"}
  sources:
    open_targets: {release: "25.09", manifest_sha256: "...", tables: {target: "fp1:...", known_drug: "fp1:..."}}
    tahoe_100m: {release: "2dc57900...", manifest_complete: true}
    cellxgene_census: {requested: stable, resolved: "2025-xx-xx"}
  readiness: {tools_not_ready: {"mcp__functional_genomics__query_drug_perturbation": "tahoe_100m.de_permissive missing"},
              columns_not_ready: {"open_targets.target.homologues.speciesId": schema_drift}}
  memory: {server_limits_mb: {target: 7168, data: 3000}, containment: rlimit_data}
  determinism: {hash_seed: 0, flags_stripped: ["-E"]}
  leakage: {ceiling: null}
```

`MANIFEST.degraded` becomes `{servers, tools, tables, reason, at}`, with `servers` kept for
compatibility and defaults added in `session._upgrade_manifest`.

---

## 16. CLI

`src/vbt/datalayer/cli.py : add_datasource_parsers(sub)`, imported lazily in `cli.build_parser`
(L1071). The command is `vbt datasource` with alias `vbt ds`, because `vbt data` is the Zenodo
fetcher. Handlers follow the repo convention `handler(args, config) -> int`.

| Command | Phase | Does |
|---|---|---|
| `vbt ds list` | 1 | sources, tables, release, readiness summary |
| `vbt ds describe <source>[.<table>]` | 1 | grain, key, roles, coverage, vocabularies |
| `vbt ds lint [--strict]` | 1 | descriptors and overlays: roles, keys, bindings, plugin names, defect citations |
| `vbt ds check [--depth shallow\|standard\|deep] [--table S.T] [--column S.T.C] [--tool server.tool] [--json]` | 1 | table, column, partition and tool readiness (`vbt doctor --data` calls it) |
| `vbt ds resolve <id_type> <value...>` | 1 | resolutions with rules and candidates |
| `vbt ds explain <server>.<tool> \| --all` | 1 | binding, serve mode, reads, derived schema and text diff, memory estimate, defects |
| `vbt ds fingerprint [--write]` | 1 | table fingerprints (what gets pinned) |
| `vbt ds index build [--id-type T] [--access-paths]` | 1 | prebuild resolver sidecars and row-group value indexes |
| `vbt ds estimate --table S.T \| --tool server.tool` | 1 | memory estimate and admissibility on this host |
| `vbt ds retro-audit <run>` | 1 | re-classify a recorded run offline; counts cited calls that are now not_found/empty/defect/blocked |
| `vbt ds plugins [--kind K]` and `vbt ds conformance [--kind K] [--plugin P]` | 2 | registry listing; run suites and write stamps |
| `vbt ds overlay init <server>` | 4 | scaffold an overlay from `list_tools` (every arg `binds: unknown`, tool `status: unreviewed`) |
| `vbt ds status [--run R]` | 4 | live per-server RSS, residents, limits, recycles, OOM kills |
| `vbt ds calibrate <source>.<table>` | 4 | sample-and-scale memory calibration |
| `vbt ds replay <run> <tool_use_id>` | 5 | re-execute a call on current data and compare row-key hashes |
| `vbt ds diff-release --from 25.09 --to 25.12` | 5 | role-column drift, encodings, vocabularies |

`vbt doctor` prints a per-tool readiness summary; `vbt doctor --smoke` runs sentinel positive
controls through the gateway.

---

## 17. Config keys

Added to `vbt.config.CODE_DEFAULTS["data"]` (`src/vbt/config.py`, L45) and mirrored with comments in
`configs/default.yaml`. Consumers use `vbt.datalayer.settings.DataSettings.from_config(config)`, which
carries the same in-code defaults, so hand-built configs work before the config package lands.

```yaml
data:
  enabled: true                       # false = exact pre-datalayer behaviour (bridge built with gateway=None)
  descriptors_dir: configs/data/sources
  overlays_dir: configs/data/overlays
  cache_dir: ${VBT_DATA_DIR:-data}/.vbt-datalayer   # index/, readiness/, stats/; always safe to delete
  project_dir: null                   # the active project (docs/PROJECTS.md); set by --project or its profile
  gateway:
    mode: enforce                     # "off" | observe | enforce (quote "off")
    enforce_servers: all              # or a list, for staged rollout
    profile: safe                     # safe | fidelity
    when_service_down: strict         # strict: guarded tools not_ready; lenient: generic guard
    unbound_empty: empty_unverified   # empty_unverified | error
  service:
    mem_limit_mb: auto                # the data child: 5% of memory.host_mb within 3,000-32,768 MB (§14.6)
    max_concurrency: 4
    timeout_s: 600
    max_resident_mb: auto             # 2/3 of mem_limit_mb: what one request holds resident (R5 key pass, §14.5)
  resolution:
    max_candidates: 10
    allow: [raw_member, normalized, label, previous, alias, exact_synonym, related_synonym, retired, xref, crosswalk, parent_family]
    max_hops: 2                       # crosswalk chain length
    max_expand: 5000                  # descendant expansion fan-out; above it: too_large (expansion)
    min_resolved_fraction: 0.95       # default for gene-list arguments
    remote_ttl_s: 3600                # universe_via listing caches
  witness:
    enabled: true
    max_scan_bytes: auto              # auto budgets: the floor x service.mem_limit_mb / 3,000 (§14.6)
    max_inflate_rows: 5000
    max_inflate_bytes: auto           # at least 20,000,000
    max_key_set: auto                 # at least 20,000
    repair_max_bytes: auto            # at least 500,000,000
    topk: true                        # false: counts only (no top-k comparison)
  derive: {enum_max: 64, description_max_chars: 1200}
  memory:
    host_mb: auto                     # planned with: the smaller of MemTotal and the memory cgroup limit (§14.6)
    default_server_mb: auto           # one upstream server: 0.8 x the host budget, at least 2,048 MB (§14.2)
    limit_kind: rss                   # rss (cgroup, else watchdog; no RLIMIT_DATA) | rlimit_data | cgroup | watchdog | none
    estimate_safety: 1.3
    expansion: {flat: 1.5, string: 3.5, nested: 8.0, fragmentation: 1.15}
    object_overhead_bytes: {string: 50, nested_item: 120, struct_item: 240}   # per num_values (§10.4)
    recycle_idle_servers: true
    recycle_wait_s: 30
    max_recycles_per_10min: 4
    max_oom_kills: 3
    max_result_bytes: 2000000
    host_budget_mb: auto              # the sum over servers: 0.75 x host_mb - reserve; a number; off
    harness_reserve_mb: 2048          # the reserve is max(this, 5% of host_mb)
    relay_max_message_mb: 0           # > 0 relays server stdout with this message cap
  readiness:
    per_turn_depth: shallow
    session_depth: standard
    key_check_full_max_rows: 50000000
    vocab_budget_bytes: auto          # bounded vocabulary scans (R7): at least 500,000,000
    remote_ttl_s: 3600
    sentinels: true
    block_when: all_unready           # session blocks on data only when no granted data tool is ready
  results: {relation_list_max: 50}
  leakage:
    ceiling: null                     # e.g. ${web.literature_max_date}; set by no-web scenarios
  sources:
    alias: {}                         # {open_targets: zenodo_vbt}: serve open_targets.* from tables that implement them
  provenance: {row_keys_max: 10000, dir: logs/data_provenance}
  plugins: {paths: [], entry_points: true, disabled: [], override: {}, require_conformance: false}
  acquisition:                        # `vbt data acquire` (docs/DATA_SETUP.md)
    root: ${VBT_DATA_DIR:-data}/sources   # homes <root>/<source>/<release>; acquisitions.jsonl logs every run
    auto: "off"                       # "off" | ask | under_budget (what a not_ready refusal leads to between turns)
    budget_bytes: 0                   # under_budget: acquisitions up to this size happen between turns
    workers: auto                     # parallel transfers (auto: 4 per CPU, at most 32)
    rate_mbps: 50
    reserve_bytes: 1073741824         # free disk kept after a download
    retries: 4
    timeout_s: 120
```

Per server in `configs/mcp_servers.yaml` (all optional): `mem_limit_mb`, `overlay` (default
`configs/data/overlays/<name>.yaml`), `sources`, `launcher` (false disables the reaper for that server).

---

## 18. The 24 fixes in five phases (14.0 engineer-weeks)

### Phase 1: stop silent wrong answers without rebuilding data (4.5 ew)

Exit criteria: the six correctness tests xfail per case with the gateway off (with their positive
controls passing) and pass with it on; the fixture readiness precondition passes; the existing suite is
green; every YAML example in this document lints; retro-audit on recorded runs reviewed; `vbt ds lint`
clean; observe mode run for at least one recorded scenario per server before switching to enforce.

| # | Fix | Delivers | ew |
|---|---|---|---|
| F1 | **Typed outcome channel and agent contract** | `GatewayError(ToolFailure)` kinds (incl. `insufficient_resolution`) with fixed per-kind payloads; `DataResult` with `_vbt` header first and the unshrunk payload for the runtime spill; `Runtime._execute` records `result_status`/`error_kind`/`data_provenance` (with coverage) and writes `logs/data_provenance/<id>.json`; `failures._data_failure` exempts model-side kinds; session and audit indexes carry status, kind and coverage; `shrink_json` keeps `_vbt`; `data_layer_addendum.md`; `genomics_burden_addendum.md:20` corrected; `${run.mcp_output_dir}` expansion | 0.25 |
| F2 | **Gateway seam in front of every MCP server** | `MCPBridge(gateway=, on_tools_changed=)`; ctx through `_make_handler`; `prepare` once before the retry loop; `call_raw`; `_convert(classify_only)`; update-in-place registration and late tools reaching `Runtime.registry`; hidden internal tools; generic guard (`empty_unverified`) with not-found evaluated before legacy envelopes; off/observe/enforce; `VBT_EUTILS_BASE` override in the harness PubMed server for tests | 0.35 |
| F3 | **Data child and bounded reader** | `data` FastMCP child with sys.path bootstrap; two-pass reader with leaf projection (leaves inside lists), storage-typed literals, NaN as null, `Any`/`All` quantifiers, item tables (multi-level explode with ancestor keys), canonical row keys, global and per-group top-k, exact totals, byte budgets; row-group value sidecar indexes for declared `access_paths` (no data rebuild); parquet format (incl. `large_*`, int32) and single_file/sharded_dir/hive/upstream_only layouts that list fragments by name and never skip unreadable ones; conformance suites with capability tags (F-10, F-11, L-7); `_stats`, `_serve` (lookup/find/search/members/count) returning its own totals | 0.55 |
| F4 | **Identifier plugins and resolver** | phase-1 identifier plugins (§9.4) incl. `pmcid`, `doi`, `local_key`, `cbio_sample`, `cbio_patient`, `ncbi_taxon`, `inchikey`; YAML options (`prefixes: from_universe`, `canonical_prefix_case`), `normalize_stored`; identifier suite with confusion matrix and declared overlaps; sidecar indexes with stored forms and families; qualified id_types; closed rule grammar and per-id_type rule lists; synonym-kind policy; retired IDs, xrefs, crosswalk chains, parent families, union kinds, `label_of`; ambiguity with `disambiguate_with`; all-rejected → `invalid_argument`; existence modes incl. `unknown`; list arguments with `min_resolved_fraction` | 0.55 |
| F5 | **Phase-1 descriptors and overlays for all 103 bridged tools** | the complete models of §6.1 and §8.1 (`ColumnSpec` union, item tables, `MatrixSpec`, `LeakageSpec`, enrichment and every sub-model), the normative path grammar and reference scoping, load, lint (incl. doc examples in CI), catalog with qualified id_types; non-strict descriptors (keys with nullable parts, bound columns, coverage, sentinels, `verified: false` facts) for OT, Tahoe, Zenodo, cBioPortal, CT.gov, PubMed and Census; overlays for 101 upstream + 2 PubMed tools with serve modes (Appendix A); defect registry with `file:line` and detector tests importing unmodified upstream | 0.6 |
| F6 | **Not-found contract and witness** | classification precedence owning `_EMPTY_LOOKUP` semantics; `_witness` over tables and item tables with `group_by`, `excluded_unknown`, `unknown_total`, `distinct_counts`; W1–W6 evaluated after T1–T6 (phantom rows, short pages, echo with redirects and synonyms, set echoes); limit inflation (disabled with `output_path`), per-group cuts, ranking refusal or declared `order_source`; repair from the data child or `tool_defect`; derived serving for the phase-1 derived set; quarantine of blocked tools | 0.5 |
| F7 | **Argument contracts and role-driven transforms** | vocabularies and enums with storage-typed snapping, selectors, `order_by`, abs ops, `binds_any`, nonempty/has_kind, unbound-with-value → `unsupported_filter`, threshold group guard, regex escaping, `escape`/`forbid`/`pattern`, substring collision, require_any/exclusive, list bounds, wrap, output_path confinement and write-once, disclosed defaults, gateway-only args (auto-derived for scope key columns); the scope-completeness rule (`incomplete_key` in every serve mode); SOMA filter parser with `is_primary_data` enforcement; projection injection; leakage filter injection and T1; transforms T1–T14 (in-band unknowns, phantom rows, honour arguments per item, unknown never passes incl. counts and not-applicable, negation with nested recount, duplicates, levels, order and cut per group and grain, recount, trims in order, flag partition, measure validity); Census `FileCheckSpec` and `materialized_by`; minimal statistic plugins with their suite | 0.5 |
| F8 | **Memory and process safety** | `reaper.py` (RLIMIT_DATA, status file, exit marker; strips `-E`/`-I` and builds an explicit allow-listed environment so `PYTHONHASHSEED=0` takes effect, recorded in provenance); child env pins; two-estimator memory model with `num_values` object overhead and per-grain `row_bytes_p99`; admission only for the upstream route; residency ledger; cold-call lock; recycle-to-evict; `too_large`; count-first admission for remote reads with `size_from`; OOM classification with no retry and a separate budget; dead-session fix | 0.35 |
| F9 | **Tool-scoped readiness** | `_check` R1–R10 per table, column, container, item table and partition (manifest attribution by path, partition `expect`, R4b canonical forms, R5 from leaf scans for nested keys, R5b key uniqueness, R6 confirmation from distinct snapshots, R9 referential samples, R10 relation and domain invariants); statuses incl. `plugin_unavailable`, `awaiting_producer`, `unbound`; harness cache with stat-only per-turn signatures; call-scoped `ToolReadiness` with per-kind degradation; `degraded_tools`; `degraded_servers` only when all tools are unready and no `(reference data)` pseudo-server; AST check that `reads` is complete; Tahoe narrowed to the files tools read; smoke through the gateway with positive and negative controls | 0.4 |
| F10 | **Claims, provenance v1, correctness tests and retro-audit** | `vbt.dataprov/1` records (coverage, canonical row keys per grain, lineage, partitions read, hash seed, leakage, evidence nature); claims rules (`supports`, coverage values, censored statements, leakage risk, evidence-nature warnings); verify `empty_result_cited` and `degraded_run`; `pinned["data"]`; `vbt datasource` phase-1 commands incl. `retro-audit`; OT-, Tahoe-, PubMed-stub and cBioPortal-stub fixtures with readiness preconditions; the six tests per case in off/enforce with positive controls and a fidelity variant of CT-3; upstream-untouched, determinism and architecture tests; `schema.py` `pubmed_ids` delimiter fix | 0.45 |

**What phase 1 explicitly does not need:**
- no data rebuild, re-download, re-preparation or format conversion; nothing is written under
  `OPEN_TARGETS_DATA_PATH` or `TAHOE_DATA_PATH` (they can be mounted read-only);
- no new derived data beyond regenerable sidecars in `data.cache_dir` (resolver TSVs of a few MB,
  row-group value indexes, readiness JSON, footer stats); deleting the cache only costs a rebuild;
- no strict descriptors (every column roled) — only the columns bound by overlays, keys, sentinels
  and readiness checks;
- no native `mcp__data__*` tools exposed to agents, and no change to `configs/agents.yaml`;
- no matrix formats (csv, h5ad, zarr), no statistic plugins beyond the minimal set, no cgroups, no
  stdout relay, no live-source layouts (the remote protocol is declared, its plugins come in phase 4);
- no change to upstream code, upstream prompts or upstream tests.

### Phase 2: complete descriptors, plugin kinds, derivation and native tools (3.5 ew)

| # | Fix | Delivers | ew |
|---|---|---|---|
| F11 | **Strict descriptors** | every column of all 38 OT tables (including item tables for the nested containers tools return), Tahoe (DE + four metadata tables), the Zenodo subset (incl. matrix cohorts with `fragment_key` and overrides, and `implements` aliases of the OT 25.09 extracts with `lineage`), DepMap (S4), GO OBO (`extends: open_targets:go_term` hierarchy), MSigDB GMT (S9) roled; keys verified (`check --deep`); `lint --strict` in CI; drift findings for undeclared columns; optional and fragment-specific columns | 0.8 |
| F12 | **Format and layout plugins v2** | csv/tsv (`stats_scan`, header parse, `matrix`, GCT preamble), jsonl, h5ad (backed; `matrix`; index exposure; gzip CSR; `requires`), zarr, obo (published projection schema; `data-version`), gmt (`logical_schema`, file-stem `fragment_key`), npy/safetensors embeddings with `ids_from`; zip_member, http_range; matrix conformance suite (F-12); entry-point discovery and conformance stamps for third-party plugins; identifier plugins `ncbi_gene`, `geo_gsm`, `census_joinid`, `cell_barcode`, `cell_ontology`/`uberon` | 0.9 |
| F13 | **Derivation engine for every tool** | argument schemas and agent text for all 103 tools from roles, including item grains, levels, cutoffs, families, censoring, propagation and evidence-nature sentences; golden snapshots; `vbt ds explain --all`; overlays reduced to bindings plus serve modes | 0.8 |
| F14 | **Native role-derived tools and the in-process client** | `mcp__data__{resolve,describe,lookup,find,search,vocab,members,aggregate,similar,neighbors}` with `where`/`key` resolution and derived schemas; matrix long views with `attributes_from`; views (target_profile with per-section status) replacing the blocked comprehensive profile; derived set comparison replacing `compare_direct_indirect`; `vbt.datalayer.client` with provenance and artifact lineage (`register_artifact(derived_from=…)`), Case 1 readers moved onto it; workspace subprocess memory limit; `configs/agents.yaml` grants with `expose.withhold_from` honoured | 1.0 |

### Phase 3: statistics, enrichment, composite keys and ontology (2.5 ew)

| # | Fix | Delivers | ew |
|---|---|---|---|
| F15 | **Statistic plugins v2** | fdr_bh with declared family, log2fc/SE, wald, effect_beta, gene_effect (declared cutoff and inclusivity), clpp, coloc_h4, tpm unit guard, log2_intensity, cosine (`veto_labels`), hypergeom/fisher (`test`), survival_time (`paired`: Kaplan-Meier, `CensoredCmp`); suites S-7 and S-9 | 0.6 |
| F16 | **Native enrichment and aggregation** | `enrich` verb from the member role's `enrichment` contract: declared universe per scope (aspect-specific background, caller override), full family incl. zero-overlap sets, size bounds counted within the universe, propagation through the id_type's hierarchy (GO from the OBO source), BH over the full family, `resolution_summary`, statistics provenance; DepMap aggregation over `target_essentiality` item tables and the S4 matrix with minimum n and exact vocabularies (streamed, replacing the 10–13 GB upstream load) | 0.6 |
| F17 | **Composite-key completion** | Tahoe per-dose/per-plate comparisons and selectivity (NaN-safe; no cartesian merge; `conditions` tuple universe with `combination_not_profiled`); interaction edges with orientation, unordered grains, per-source top-k and a specified `neighbors`/`_expand` traversal (hops limit, frontier in key order, per-source edges, truncation as `partial` or `too_large`, node-set hash in provenance) unblocking the network tools; ChEMBL parent/salt fan-out with merge by key | 0.7 |
| F18 | **Ontology semantics** | `include_descendants` (Eq OR Contains(ancestors), or expansion through `IdTypeSpec.hierarchy` with predicates and a closure sidecar), `propagated_over` refusals, Reactome hierarchy membership (`find_genes_in_pathway` for the 564 IDs without direct genes), `propagation: mixed` disclosure, lossy `topLevelTerm` with an exact alternative, Cell Ontology/UBERON subtrees with declared predicates, expansion records with `max_expand` | 0.6 |

### Phase 4: memory at scale, live and third-party sources (1.75 ew)

| # | Fix | Delivers | ew |
|---|---|---|---|
| F19 | **Memory at scale** | calibration tiers (sample-and-scale, measured from status files) replacing seed factors; host budget with LRU idle recycle; cgroup v2/v1 and RSS watchdog; optional stdout relay with byte cap and its own fault tests; huge-table indexes beyond phase 1 (literature `keywordId`) unblocking `search_literature` | 0.6 |
| F20 | **Live and remote sources** | `live_api` and `soma` layout plugins and `rest_json`/`soma` formats implementing the phase-1 `live` capability; remote witness (count requests within the request budget); remote vocabularies with TTL and drift checks; CT.gov, E-utilities, cBioPortal and Census descriptors completed (record versions, `release.resolve`, per-study `importDate`); Census resolved release with drift detection, count-first admission and the `(dataset_id, donor_id)` donor key with donor-balanced sampling per dataset; derived lookup/find on live tables; PubMed fixes in place in harness-owned `pubmed_server.py` (errorlist/querytranslation surfaced, PMID reconciliation, no silent 50 cap); single-cell `genes_found` recomputed from returned `var.feature_name` | 0.65 |
| F21 | **Third-party servers and the envelope kind** | `vbt ds overlay init`; HTTP failure classification; the `envelope` plugin kind (protocol, suite, registry entry: the worked example of adding a kind); overlay conformance on recorded outputs; documentation for adding a server | 0.5 |

### Phase 5: verification and hardening (1.75 ew)

| # | Fix | Delivers | ew |
|---|---|---|---|
| F22 | **Replay, drift and row-level citations** | `vbt ds replay` comparing row-key hashes, output-row hashes (enrichment `(set, k, K, p, q)`), computed values within tolerance and per-partition fingerprints; `vbt verify --data` (`data_version_drift`, `replay_mismatch`, `source_updated` for changed live record versions); `row_key` citations validated against stored row keys; `refresh_claims` marks evidence unresolved on fingerprint mismatch | 0.6 |
| F23 | **Calibration, soak and enforce graduation** | measured memory factors on real 25.09 subsets (estimates within ±30%, nested tables included); witness p95 overhead under 300 ms on composite-key tables; observe→enforce graduation per server with retro-audit evidence; `profile: fidelity` validated on the paper scenarios | 0.55 |
| F24 | **Coverage and drift guards** | CI: every registered tool bound, every column roled, upstream tool-schema snapshot drift, defect detectors, architecture tests, determinism across hash seeds (gateway transforms and `_serve` byte-identical), submodule clean; `vbt ds diff-release` (role columns, types, encodings, vocabularies, axis membership); operator runbook (symptom → command) | 0.6 |

Totals: phase 1 = 4.5, phase 2 = 3.5, phase 3 = 2.5, phase 4 = 1.75, phase 5 = 1.75;
**24 fixes, 14.0 engineer-weeks**.

Effort honesty: revision 2 moved work forward. The stress test showed that several silent wrong
answers in phase-1 tools could only be stopped by mechanisms revision 1 had scheduled late or not at
all (storage-typed literals, nullable keys, the scope rule, item-level witness and coverage, phantom
rows, leakage, determinism, partition completeness), so phase 1 grows from 4.0 to 4.5 ew. Phases 4 and
5 shrink by 0.25 ew each because the remote protocol, write-once outputs, the leakage transform,
spill-before-cap and canonical row keys now land in phase 1, and their later fixes only add plugins
and checks. Phase 1 still holds at 4.5 because (a) all 27 phase-1 derived tools are instances of a few
generic verbs configured in YAML, not per-tool adapters; (b) overlay authoring is weighted by measured
grants in `configs/agents.yaml` (`get_target_info` and `search_targets_by_name`: 8 agents each;
`search_known_drugs`: 4; `get_comprehensive_target_profile`: 0, so it can be blocked at no cost), and
any tool not reviewed in time falls back to the generic guard, which is safe (`empty_unverified`, never
citable) rather than wrong; (c) phase-1 descriptors are non-strict; (d) the matrix and enrichment
models are defined in phase 1 but implemented in phases 2 and 3.

---

## 19. The six correctness tests

All six live in `tests/datalayer/test_dl_correctness_six.py`. Each starts the **unmodified** upstream
servers through `MCPBridge` (stdio, `python -B`, `PYTHONDONTWRITEBYTECODE=1`) with
`OPEN_TARGETS_DATA_PATH`/`TAHOE_DATA_PATH` pointing at OT-25.09-shaped and Tahoe-shaped fixtures
built in `tmp_path` by `tests/datalayer/dl_fixtures.py`. Expected answers come from a pyarrow
**oracle** computed directly from the fixture files, never from the code under test. No network.
All six pass at the end of phase 1. The stress test reproduced every "today" column below on
unmodified upstream code (§25).

Revision 2 changes how the tests are built (the stress test showed revision 1's form could pass for
the wrong reason):

- **Parametrized per case × gateway.** Each call is its own case. In `off` mode each case is a strict
  `xfail` on the correct behaviour **and** pins today's exact wrong behaviour (`is_error=False` with
  the text `Target X not found`; `TP53BP1` as `results[0]`; PMID 1234 returned), so an upstream fix
  surfaces as a detector change instead of a silent pass.
- **Positive controls** that must pass in both modes: `get_target_info("ENSG00000169174")`,
  `get_disease_info("MONDO_0005148")`, `get_drug_info("CHEMBL25")` return their rows; before any
  deletion, `disease` and `drug` are not degraded and `search_known_drugs` is ready.
- **Fixture preconditions.** `dl_fixtures.py` writes every declared column of each read table with the
  25.09 nested types, the descriptor sentinels (PCSK9 with non-empty `pathways` and `go`), and a
  `.download-manifest.json` in the upstream downloader's format (`release`, `base`, `complete: true`,
  `files: {relpath: {bytes, sha256}}`, as checked by `tools/doctor.py:41-57`); placeholder shards are
  written for all 38 `OPEN_TARGETS_DATASETS` directories so the legacy baseline can fail on the table
  under test. A guard test asserts that `vbt ds check` reports every fixture table and the resolver
  indexes for `ensembl_gene`, `ot_disease` and `chembl_molecule` ready; a failure there is reported as
  a fixture error, not as a CT failure. `test_dl_shipped_configs.py` asserts the fixture schemas
  conform to the shipped descriptors.
- **Stubs.** PubMed runs against a local E-utilities stub through `VBT_EUTILS_BASE`, which logs every
  request; cBioPortal runs against a network-free `pybioportal` stub module placed first on the
  server's `sys.path` by the test (the upstream server is not edited).
- **Claims through the runtime.** Claim checks run the calls through `Runtime` with the mock provider
  scripted to issue the tool calls and then `record_claims`, so `is_error`, `result_status` and
  `coverage` come from the real recording path.
- Calls run with `vbt.preflight` in degraded mode (`allow_missing_data: true`) where a table is deleted.

### CT-1: identifiers of the wrong form are resolved or rejected, never answered as "not found"

- **Fixture.** `target`: ENSG00000169174 (approvedSymbol PCSK9, `symbolSynonyms [{label: NARC1}]`),
  ENSG00000141510 (TP53, synonyms `[{label: LFS1}]`), and TP53BP1 (ENSG00000067369) and TP53I3
  (ENSG00000115129) stored **before** TP53 in file order and sorting before it by key; `disease`:
  EFO_0000685, MONDO_0005148; `drug_molecule`: CHEMBL25.
- **Calls and expectations.**

| call | today (VERIFIED) | required |
|---|---|---|
| `target.get_target_info("PCSK9")` | `{"found": false, "error": "Target PCSK9 not found"}`, `is_error=False` | PCSK9 row; rule `label_exact:approvedSymbol`; `canonical_id_type` ensembl_gene |
| `get_target_info("ENSG00000169174.12")` / `("ensg00000169174")` | same | resolved by `normalized:strip_version` / `normalized:upper`, note in `_vbt.notes` |
| `get_target_info("NARC1")` | same | resolved, rule `synonym:alias`, note in `_vbt.notes` |
| `get_target_info("ENSG00000999999")` | same | `is_error`, `kind: not_found`, `citable: false`, `tried` non-empty |
| `disease.get_disease_info("EFO:0000685")` | non-error not found | resolved to `EFO_0000685` by `normalized:curie_colon_to_underscore` |
| `drug.get_drug_info("chembl25")` | non-error not found | resolved to `CHEMBL25` |
| `target.search_targets_by_name("TP53", limit=1)` | TP53BP1 first (regex substring, file order) | `results[0].id == "ENSG00000141510"`, `match: exact`; rows ordered by match class (§10.6), so key order cannot rescue a wrong implementation |
| `pubmed.fetch_abstracts(["PMC1234"])` (E-utilities stubbed) | PMID 1234 returned (a 1975 paper; VERIFIED live 2026-10-06) | `invalid_argument` with `looks_like: [pmcid]`; the stub's request log is empty |

- **Claims.** A claim citing the `ENSG00000999999` call is rejected ("cites failed tool call").

### CT-2: unknown is an error, known-but-empty carries coverage, an outage is not a negative

- **Fixture.** `openfda_significant_adverse_target_reactions` has rows for PCSK9 only;
  `target_prioritisation` present; `target` row for ENSG00000141510 with `chemicalProbes: []` and
  `hallmarks: []`; a target with `chemicalProbes: null`; `known_drug` present; manifest listing the
  `known_drug` files. A `pybioportal` stub with one study, two patients, three samples.
- **Calls and expectations.**
  - `get_target_safety_profile("ENSG00000999999")`, `get_target_prioritisation_scores(...)`,
    `get_chemical_probes(...)`: today `success: true` with zero counts, or (`get_chemical_probes`)
    `{success: false, error: "Target … not found"}` with `is_error=False` through `_EMPTY_LOOKUP`;
    required: `not_found`. `get_comprehensive_target_profile(...)` with that ID: `quarantined`.
  - `get_target_safety_profile("ENSG00000141510")`: `_vbt.status == "empty"`, `coverage ==
    "unknown"`, `coverage_statement` mentions "not evidence of safety", the upstream `message` ("No
    adverse event data for this target") removed; a claim citing it with `supports: presence` is
    rejected; with `supports: absence` it is also rejected ("coverage unknown").
  - `get_chemical_probes("ENSG00000141510")` (rows are items of the `target_chemical_probes` item
    table): `status: empty`, `coverage: covered`; a claim with `supports: absence` is accepted as
    `evidence_status: "absence"` with coverage copied; `supports: presence` is rejected. The target
    with `chemicalProbes: null` gives `coverage: unknown`. `get_target_hallmarks(TP53)` gives
    `status: empty`, `coverage: unknown` (no nested coverage declared).
  - Delete `known_drug/`, rerun readiness (once with the fixture manifest, once without a manifest):
    `drug.search_known_drugs(target_id=...)` returns `not_ready` naming `open_targets.known_drug`;
    `target.get_target_info(...)` still succeeds; `get_comprehensive_target_profile(PCSK9)` is
    `quarantined` (phase 2: the view's `known_drugs` section is `{_vbt_unavailable: ...}` and no
    `num_drugs` field exists); `degraded_servers` is empty and `MANIFEST.degraded.tools` names only
    tools reading `known_drug`; no call returns `num_drugs: 0` as success.
  - cBioPortal (stub): `get_clinical_data("study_x", sample_ids=[real, "NOPE-01"])`: today
    `success: true`, `sample_count: 2`, a `{sampleId: "NOPE-01", patientId: null}` record (VERIFIED);
    required: `not_found` listing `NOPE-01` (W6). `sample_ids=[]`: today "No samples found" as an
    `_EMPTY_LOOKUP` success; required: `invalid_argument` (`min_items`). An unknown study: today a
    generic "Failed to retrieve clinical data"; required: `not_found`. A real call returns a `patients`
    section with one row per patient and `_vbt.grains.patient == 2` (no OS value counted twice).

### CT-3: a tool that would return a false empty never does

- **Fixture** (rows pinned so each assertion discriminates). `pharmacogenomics` rows for targets T and
  T2: (T, [CHEMBL3]), (T, [CHEMBL3, CHEMBL25]), (T2, [CHEMBL3]), (T, [CHEMBL25]), (T2, [CHEMBL99]);
  `drug_molecule`: CHEMBL3, CHEMBL25, CHEMBL1000 (`linkedTargets = {rows: ["ENSG00000169174"],
  count: 1}`) — CHEMBL99 is deliberately absent from `drug_molecule`; `mouse_phenotype`: 2 rows with
  `targetFromSourceId = ENSG00000169174`, `targetInModelEnsemblId = ENSMUSG00000044254`;
  `target.go`: 2 annotations for ENSG00000169174 and `go: []` for TP53; `disease`: MONDO_0005148
  (type 2 diabetes mellitus) with `synonyms.hasExactSynonym = ["NIDDM"]` and `obsoleteTerms =
  ["EFO_0001360"]`; `go` table with `GO:1000001`; `biosample` with a synonym-only term;
  `evidence/sourceId=europepmc` with `literature` containing `30595370`.
- **Calls.** `drug.get_pharmacogenomics(drug_id="CHEMBL3")` and the `target` copy;
  `get_pharmacogenomics(target_id=T, drug_id="CHEMBL3")`; `get_pharmacogenomics(drug_id="CHEMBL99")`;
  `drug.search_drugs(target_id="ENSG00000169174")`; `target.get_mouse_phenotype("ENSG00000169174")`;
  `pathway.get_gene_ontology("ENSG00000169174")`; `disease.search_diseases_by_name("NIDDM")`;
  `disease.get_disease_info("EFO_0001360")`; `association.get_evidence_by_publication("30595370")`;
  `pathway.search_go_terms("GO:1")`; `expression.search_biosample_ontology(query=<synonym-only term>)`.
  Negative controls: `get_gene_ontology(TP53)`, `get_mouse_phenotype` and `search_drugs` for an
  existing target with no rows, `get_pharmacogenomics(drug_id="CHEMBL25")` restricted to T2.
- **Today** (VERIFIED). All single-filter calls return 0 rows as success; the two-argument pgx call
  returns 3 rows, one of which (`[CHEMBL25]`) violates `drug_id` (if/elif); `search_go_terms("GO:1")`
  and the biosample synonym search return 0 rows.
- **Required.** No call returns `ok`/`empty` with 0 rows where the oracle has rows: drug pgx 3 rows;
  the two-argument call exactly the 2 oracle row keys; `CHEMBL99` 1 row with a note (existence
  `bound`: absent from `drug_molecule`, present in the bound column); `search_drugs` 1; mouse 2; GO 2
  item rows whose row keys are the item table's complete keys; NIDDM 1 (`MONDO_0005148`, rule
  `synonym:exact`); `EFO_0001360` resolves to `MONDO_0005148` by rule `retired:obsoleteTerms`;
  evidence ≥ 1 equal to the oracle count; GO search and biosample search equal the oracle. Every
  result's provenance holds full row keys, served `derived` or `repaired`. Negative controls give
  `status: empty` with coverage, never rows and never `tool_defect`.
- **CT-3b (fidelity and pass).** The same calls with the overlays forced to `serve: pass,
  on_contradiction: tool_defect`, and again with `profile: fidelity`: bound-ID calls return
  `tool_defect` whose payload has `witness.total` equal to the oracle count; the two-argument pgx call
  is refused (W2 after T4 counts the violating row as excluded and the witness top-k differs); the
  free-text disease search returns `empty_unverified`, and `record_claims` rejects citations of it
  both ways.

### CT-4: ranked results are the global top-k with honest totals

- **Fixture.** `known_drug` for target T (a real ENSG): 30 rows of phases 1–3 first in file order,
  then 5 phase-4 rows, plus 2 unknown-phase rows (one null, one NaN), with `status` null on 7 rows and
  `urls` lists of varying order; the key `[drugId, targetId, diseaseId, phase, status]` with
  `nullable: [phase, status]`; `openfda_significant_adverse_drug_reactions` for CHEMBL559288 with the
  highest llr stored last (descriptor key `[chembl_id, meddraCode]`, rank `llr desc`); `l2g_prediction`:
  150 loci for gene G, max score 0.8725 stored last; `interaction` for T with `scoring` ascending in
  file order, T on side B in half the rows, two `sourceDatabase` values in a separate assertion; the
  universe rows for T, G and CHEMBL559288; preconditions asserted from the files: the first k
  predicate-matching rows in file order differ from the oracle top-k for each table, and at least 5
  rows with `score >= 0.05` precede the L2G maximum.
- **Calls.** `drug.search_known_drugs(target_id=T, limit=5)`,
  `drug.get_drug_adverse_events("CHEMBL559288", limit=1)`,
  `genetics.query_l2g_predictions(gene_id=G, min_score=0.05, limit=5)`,
  `interaction.get_interactions(T, limit=5)`; then the first call again with
  `data.witness.max_inflate_rows = 10` and `data.witness.topk: false`, and once with the top-k enabled.
- **Today** (VERIFIED). Max phase 3 and `count: 5` against 35; top llr missing; max L2G 0.51 (with the
  default `min_score=0.5` upstream happens to return the maximum, so the test uses 0.05); the 5
  lowest-scoring partners.
- **Required.** Row-key assertions (run in both modes; `off` must xfail on exactly these): returned key
  lists equal the oracle top-k under the declared order, nulls last, ties by the canonical key.
  Header assertions (enforce only): `_vbt.total == 35` for `known_drug` with
  `_vbt.excluded_unknown.phase == 2` (the null and the NaN row, counted by the witness although
  upstream drops them before returning); 150 for L2G; `truncated: true`; `_vbt.grains.drug ==
  {returned: <distinct drugs in the top 5>, total: <distinct drugs among the 35>}`; the response's
  echoed `limit` is 5. Interactions: T is matched on either side; with two sources the top-k is per
  `sourceDatabase` and `_vbt.order` says so. With `max_inflate_rows = 10` and the top-k disabled the
  call returns `too_large` with `subkind: unranked_truncation` and the upstream is never called; with
  the top-k enabled it returns `tool_defect` (W3) naming witness total 35.

### CT-5: every argument is honoured or rejected, and keys are complete

- **Fixture.** `association_by_datasource_direct` and `_indirect` with datasources
  `gwas_credible_sets` and `eva`; `association_overall_direct` ⊆ `association_by_overall_indirect` with
  7 ancestor pairs scoring above the direct ones in the indirect top 10; coloc and eCAVIAR tables;
  `target_prioritisation`; `disease` named "type 2 diabetes mellitus (T2D)"; `pharmacogenomics` with
  target T and drugs CHEMBL3 (D1) and CHEMBL25 (D2); `known_drug` rows for T; Tahoe `de_permissive`
  (float32 `concentration`) with drug Bortezomib in ACH-000681 and one other line × concentrations
  0.05, 0.5, 5.0 µM on plate `1`, plus one dose on two plates; gene SELECTIVE1 significant at all
  doses in ACH-000681; Tahoe `drug_metadata`, `cell_line_metadata` (several rows per line) and
  `gene_metadata`.

| call | today (VERIFIED) | required |
|---|---|---|
| `filter_by_datasource(datasource="gwas_catalog", target_id=T, output_path="x")` | empty success | `invalid_argument`; `valid_values == ["eva", "gwas_credible_sets"]` (full vocabulary ≤ enum_max) |
| `filter_by_datasource(datasource="eva", target_id=T, include_indirect=True, output_path="x2")` | reads the indirect table | rows from the indirect table; witness and readiness on the selected table |
| `get_colocalisation_by_chromosome(chromosome="19", method="colc")` | silently queries eCAVIAR, echoes `colc` | `invalid_argument`, `valid_values: [coloc, ecaviar]` (selector) |
| `prioritize_targets(sort_by="nonexistent")` | file order, `filters_applied {}` | `invalid_argument` (order_by enum) |
| `search_diseases_by_name("mellitus (T2D)")` | 0 results (regex group) | the disease found (literal match) |
| `get_pharmacogenomics(target_id=T, drug_id="CHEMBL3")` on both servers | rows violating `drug_id` included | only the CHEMBL3 rows |
| `search_known_drugs(target_id=T, limit=0)` / `limit=-1` | empty success / all but the last row | `invalid_argument` |
| `query_drug_perturbation("Bortezomib", cell_line_id="ACH-000681")` | SELECTIVE1 three times, no concentration; row counts reported as genes | `incomplete_key`: `dimension: concentration`, `values: [0.05, 0.5, 5.0]`, `unit: uM` (in derived mode, from the scope rule) |
| same with `concentration=0.5` | n/a (`unexpected_keyword_argument` if sent upstream) | every row carries the complete key; no duplicate keys; `_vbt.grains.gene == {returned, total}` equal to the oracle; `num_total_significant` recomputed as distinct genes |
| same with `concentration=0.05` | n/a | rows found (float32 literal cast), `_vbt.scope.concentration == 0.05` |
| same with the two-plate dose | n/a | rows listed per plate with `plate` in each key (pooling `list`) |
| `compare_direct_indirect(target_id=T, limit=10, output_path="y")` | `unique_to_direct_count` 7 although direct ⊆ indirect | `quarantined`, alternatives naming `query_associations(include_indirect=false/true)` (phase 2: derived with `unique_to_direct_count == 0`) |

### CT-6: unknown or negated evidence never passes as the favourable answer

- **Fixture.** `target_prioritisation`: A `hasSafetyEvent = -1`, B null, C 0, D NaN; descriptor
  `scale: [-1, 0], encoding: {-1: known_unfavourable, 0: none_recorded}, verified: false`; the binding
  `no_safety_events: {role: flag, when_true: {in: [none_recorded]}}`. Two variants for the
  unconfirmable case: hasSafetyEvent ∈ {-1, null} only (0 never observed), and ∈ {1, 0, null}
  (refuted). `evidence/sourceId=europepmc` for T (a real ENSG): `publicationYear` 2024, 2019 and null.
  `disease_phenotype` for HP_0001250 (and the `disease_hpo` universe): X = [PCS with `qualifierNot:
  true`], Y = [IEA], Z = [PCS, PCS negated, IEA], W = [IEA negated]. `study` with 25 null-`nSamples`
  studies stored before one 200,000-sample study `Sbig`.
- **Calls and expectations** (each its own case).
  - `prioritize_targets(no_safety_events=True)`: today A, B and C are returned (VERIFIED). Required:
    readiness confirms the encoding from the distinct snapshot (codes −1 and 0 both observed, values ⊆
    codes ∪ {null}, NaN counted); only C is returned; `_vbt.excluded_unknown.hasSafetyEvent == 2` (B
    and D). In both unconfirmable variants the call is `unsupported_filter`, and
    `get_target_prioritisation_scores` stays ready (the refuted fact is scoped to its column).
  - `query_evidence(target_id=T, min_year=2024)`: only the 2024 row;
    `excluded_unknown.publicationYear == 1`. `max_year=2020`: only the 2019 row with the same count.
    `min_year=2030`: `status: empty` with `coverage: partial_unknown`, and an absence claim citing it
    is rejected.
  - `find_diseases_by_phenotype("HP_0001250", evidence_type="PCS")`: today every disease is listed,
    Y and W with `evidence_count` 0, X counted 1 and Z counted 2 (VERIFIED). Required: Z only, with
    only its non-negated PCS item and `evidence_count == 1`; X dropped as an empty parent and counted
    in `excluded_negated`. Without `evidence_type`: X and W excluded, Y and Z listed with recomputed
    counts.
  - `find_diseases_by_phenotype("HP:0001250")`: normalised to `HP_0001250`; `HP_9999999` → `not_found`;
    never a citable empty.
  - `get_study_metadata(min_sample_size=100000, limit=20)`: today 20 null-size rows, never `Sbig`
    (VERIFIED). Required: only `Sbig`, `excluded_unknown.nSamples == 25` (from the witness; inflation
    uses `total + unknown_total`).

---

## 20. The twelve data shapes

Each shape was walked through the descriptor model, plugins, derivations and gateway, and in revision
2 stress-tested with a real instance (§25). The last column records what the shape forced into the
design; "(rev 2)" marks what the stress test added.

| # | Shape | Exemplar here | Layout / format | Complete key | Roles exercised | Silent-wrong risk it exposes | What it forced |
|---|---|---|---|---|---|---|---|
| S1 | **Long table, single key** | `disease.parquet` (Zenodo single file, 39,530 rows), `drug_molecule` (18,119 rows) | single_file, sharded_dir / parquet | `[id]` (+ alternate `inchiKey`) | identifier (self), label, synonym, category, text, hierarchy | symbol/CURIE/case → citable not-found; salt forms as separate keys; retired IDs; xref-only IDs; in-band −1 phases; stored flags that contradict structure | `id_types.universe` + `resolve_via`; existence checks; (rev 2) parent families and `parent_molecule` grain, `retired` and `xref` rules, prefixes from the universe, relation constraints, `missing_values`, category placeholders, `implements`/`lineage` |
| S2 | **Composite-key long table** | Tahoe DE (6-column key), `association_by_datatype_direct` (4.17M rows; 3-column key), `known_drug` (5-column key with nullable parts) | single_file, sharded_dir / parquet | composite | identifier ×2, scope (role and facet), measure with `comparable_within`, count | dose pooling; rows counted as drugs; `datatype=None` mixing incomparable scores; float32 dose equality; stored spellings differ across tables | `scope` role and lint rule; `incomplete_key`; `grains`; prefilter `constraints`; (rev 2) storage-typed literals, nullable keys, scope kinds and pooling with one rule in `prepare`, abs ops, selectors, `limit_grain`, `access_paths` with sidecar indexes, stored forms and crosswalk chains, censored coverage, per-group top-k |
| S3 | **Nested list/struct Parquet** | `target` (tractability, go, pathways, homologues), `target_essentiality` (3 levels), `expression.tissues`, `pharmacogenomics.drugs`, `drug_molecule.linkedTargets` struct{rows,count}, `disease.synonyms` struct of lists | sharded_dir / parquet | entity key; item tables composed from item keys | nested, member, flag (`partition_items`), measure on items | ndarray/dict traps → always empty (seven tools); `value:false` buckets read as tractable; uncorrelated item conditions; nested empties cited as absence | path grammar; `Contains`; format suite F-1/F-4; flag partition; (rev 2) item tables, `Any`/`All` with list Kleene rules, nested coverage never inherited, `null_means`/`empty_means`, field maps with parent keys, per-column readiness, `leaf_projection`, `num_values` memory term |
| S4 | **Wide matrix** | DepMap Chronos gene-effect CSV (models × genes, header `SYMBOL (EntrezID)`), GTEx GCT | single_file / csv, zarr | logical `(@row key, @col key)` | identifier on both axes (header parse), measure (`gene_effect`), attributes from `Model.csv` | header identifiers drift; inclusive vs strict cutoffs; NaN read as 0; duplicate headers read silently; truncated files parse | (rev 2) `MatrixSpec`/`AxisSpec` with `parse`, axis path prefixes, long-view semantics, header axes roled as a whole, `stats_scan`, inline manifests, identity vs coverage universes, `attributes_from` and `aggregate`, `cutoff` facet, prefix-required `ncbi_gene` |
| S5 | **AnnData (dense or sparse)** | Zenodo `GSE*.h5ad` cohorts (dense bulk microarray, VERIFIED), the 2.5 GB merged lung object, Census `get_anndata` outputs | single_file / h5ad (backed) | X `(@row key, @col key)` per cohort; donors `(dataset_id, donor_id)` | identifier (var key), label, qualifier (`is_primary_data`), scope | positional `var_names`; donors merged across datasets; duplicate cells; unmeasured zeros; files differ per cohort; outputs outside the gateway | (rev 2) `fragment_key` and per-fragment overrides, `index_name`, `unique_within`, SOMA filter parsing with `is_primary_data` enforced, `materialized_by` + `FileCheckSpec` in phase 1, `zero_if_measured`, in-process client (phase 2) |
| S6 | **Hive-partitioned shards** | `evidence/sourceId=*` (8.5 GB, 23 partitions, 90 columns) | hive / parquet | `[sourceId, id]` | partition as scope facet, time (`missing: unknown`, fallback), reference (pmid and PPR lists), measures with `applies_when`/`by_partition` | corrupt shard skipped; missing partition unnoticed; partition column lost; full load OOM; null years pass `min_year`; scores mixed across sources | layout restores partitions; `size_class: huge` → derived; (rev 2) fragments listed by name with no skipping, `expect: declared`, partition columns as logical columns with `mirrored_by`, `applies_when`, time `fallback`, `NonEmpty`, sidecar indexes on `targetId`/`diseaseId`, per-partition readiness and fingerprints |
| S7 | **Ontology DAG** | `disease`, `reactome`, `go`, HPO; `tests/fixtures/mini_cl.obo`; GO DAG from go-basic.obo | parquet, obo | `[id]` | hierarchy (`closure`), label, synonym | 564 Reactome IDs "not found" by direct-only membership; "Unknown" placeholders; unbounded descendant lists; obsolete terms live; broad synonyms substituted; hierarchy in another source | `include_descendants`; propagation disclosure; trims with counts; (rev 2) `IdTypeSpec.hierarchy` with `extends`, include rule `Eq OR Contains(ancestors)`, closure/inverse facets and R10, retired terms, synonym-kind policy, `universe_where`, `propagated_over`, ordered trims, expansion records |
| S8 | **Edge list / graph** | `interaction` (STRING, IntAct, Reactome, SIGNOR), `interaction_evidence` | sharded_dir / parquet | `(sourceDatabase, intA, intB, targetA, targetB)`, `targetB` nullable | endpoint (`directed: false`), scope facet (`sourceDatabase`), measure (`comparable_within`) | one-sided matching; orientation duplicates; cross-source score mixing; hash-seed nondeterminism | undirected matching; (rev 2) effective hash seed, `EdgeSpec` with sides and orientation, unordered grains, per-group top-k and threshold guard, per-scope coverage universe, `binds_any`, pair bindings, a specified traversal |
| S9 | **Gene sets / membership** | `target.pathways`, `target.go`, GMT files, GO DAG | parquet nested, gmt, obo | `(set_id, member_id)` | member (`membership`, `enrichment`) | BH m over overlapping sets only; background ignores aspect; unmapped symbols silently dropped; duplicate items inflate counts | `sets` kind; (rev 2) `membership.{set, member, count_grain, propagate_via, enrichment}`, item tables as set views, `local_key`, list arguments with `min_resolved_fraction`, statistic `test` capability, R9/R10 membership checks, statistics provenance |
| S10 | **Time-to-event / clinical** | cBioPortal `OS_MONTHS`/`OS_STATUS` (patient vs sample), `clinical_trial_labels_reconciled.csv` | upstream_only (P1), live_api (P4) / rest_json, csv | `(studyId, patientId)`, `(studyId, sampleId)` | time, flag with `event_of`, category, `level` | patient values copied to samples (double counting); strings for numbers; phantom samples; censoring ignored by thresholds | (rev 2) `exists_when` and W6, `level` facet and grain split, `event_of` on the status flag with `CensoredCmp`, `encoding`/`parse`/`missing_values`, pivots, `column_patterns`/`roles_from`, composite universes and refs, `list_delimiter`, `expose.withhold_from` |
| S11 | **Remote API-backed source** | ClinicalTrials.gov v2, PubMed E-utilities, cBioPortal REST, Census SOMA | upstream_only (P1), live_api, soma (P4) / rest_json, soma | `nctId`, `pmid`, `studyId`, `soma_joinid` | identifier (strict NCT/PMID), category, time (`as_of`) | US-only default (89 vs 274 trials); OR escaping; PMCID → wrong article; floating `stable`; 1000 cap unflagged; post-ceiling outcomes leak; aliases look like wrong entities | `as_of`; disclosed scope defaults; IR quoting; echo checks; (rev 2) `LeakageSpec` with withhold/redact/inject/block, existence `upstream`/`unknown`, echo redirects and set echoes, field maps, `escape`/`engine`, order roles and `order_source`, `TotalSpec`, spill before cap, the `live` capability declared in phase 1 |
| S12 | **Embeddings / vectors** | `literature_vector` (about 55k × 100 float64) | sharded_dir / parquet | `[word]` | vector (`metric: cosine`, `norm_column`), category, identifier (multi-kind word) | category typo → empty success; uncalibrated "Low similarity" labels; NaN cosine scrambles order; co-occurrence cited as function | `vector` role; (rev 2) `anchor` argument role, union id_types with `kind_from`, computed result fields, `veto_labels`, norm constraints at readiness, `evidence_nature` on every result and in claims, `similar` verb (phase 2) |

Shapes folded into the twelve: identifier crosswalks with one-to-many maps (`variant.rsIds`,
`chembl_clinical_nct_data` with many rows per NCT) are `crosswalk` tables under S2 consumed by
identifier plugins (`cardinality: many`); small lookups with several rows per entity (Tahoe
`cell_line_metadata`, one row per driver mutation) are `entity_detail` under S3; genomic intervals
(`interval`, `credible_set`) use the `position` role under S2; remote HTTP-range Parquet is a layout
variant of S2/S6.

---

## 21. Test strategy

| Layer | Files | What it proves |
|---|---|---|
| Contracts | `test_dl_contracts.py`, `test_dl_descriptor.py`, `test_dl_catalog.py`, `test_dl_doc_examples.py` | models validate and lint **every** YAML example in this document (extracted from the Markdown); lint rules; reference scoping; path grammar; qualified id_types; catalog contract lookup and `same_as`; canonical row keys (float32, null, list parts) |
| Plugin conformance | `test_dl_conformance_{identifier,format,layout,statistic}.py` | §9.3 suites, parametrized over the registry; a new plugin is tested without edits |
| Resolver | `test_dl_resolver.py` | rule grammar and per-id_type order, synonym-kind policy, retired IDs, xrefs, crosswalk chains, stored forms, parent families, union kinds, ambiguity with disambiguation, all-rejected → `invalid_argument`, existence modes incl. `unknown`, list arguments, sidecar loading without pyarrow |
| Derivation goldens | `test_dl_derive.py` | schemas and text for about 20 representative tools; snapshot updates require review |
| Gateway unit | `test_dl_gateway_*.py` | prepare/finish with a fake bridge and fake data child: every error kind and payload, every transform T1–T14 in order, the scope rule in pass and derived modes, selectors, abs ops, leakage, SOMA filters, field maps, inflation (and its exceptions), W1–W6 after T1–T6, per-group cuts, classification precedence, modes and the fidelity profile |
| Bridge seam | `test_dl_bridge_seam.py` (`needs_fastmcp`) | gateway hooks through a fixture FastMCP server; `gateway=None` byte-identical; update-in-place after restart; late tools reach the registry; hidden tools; generic guard |
| Launcher and memory | `test_dl_launcher.py`, `test_dl_memory.py` | a fixture tool allocating 2 GB under a 512 MB `RLIMIT_DATA` → `oom`, no retry, server alive; SIGKILL → exit marker → `oom_killed`, no retry, separate budget; no dead-session reuse; admission refuses before allocation; recycle-to-evict |
| Data child | `test_dl_service_*.py` | reader bounded top-k and totals vs oracle; item tables and correlated `Any`; leaf projection inside lists; storage-typed literals; hive restore, partition `expect` and corrupt-shard failure; `.part` exclusion; sidecar row-group indexes; `_check` statuses per column and partition (R1–R10); sidecar contents |
| Shipped configs | `test_dl_shipped_configs.py`, `test_dl_reads_complete.py` | every shipped descriptor and overlay lints clean; all 101 upstream registered tools (AST-parsed from `server.py` `register_tool` calls, no import) plus the 2 PubMed tools are bound; defect entries cite existing `file:line`; fixture schemas conform to descriptors; every dataset an upstream tool loads (AST, transitively through same-module helpers) is in its `reads`; deleting each table in turn yields `not_ready` or an error, never `ok`/`empty` |
| Defect detectors | `test_dl_defect_detectors.py` | each `defects` entry is reproduced by importing the **unmodified** upstream function (temporary `sys.path`, eviction afterwards, the `preflight.upstream_doctor` pattern at L247) against fixtures; a detector that stops reproducing fails, forcing the entry's removal |
| Correctness | `test_dl_correctness_six.py` (marker `correctness`) | §19, per case × {off (strict xfail + pinned wrong answer), enforce}, positive controls, fixture readiness precondition, CT-3b pass/fidelity variant |
| Plumbing | `test_dl_runtime_plumbing.py`, `test_dl_claims.py`, `test_dl_verify_pinning.py`, `test_dl_preflight.py`, `test_dl_cli.py` | trace fields, failures exemption, claims table §15.3, verify kinds, pinned block, scoped readiness, CLI parser wiring (`test_doctor_parser` pattern) |
| Retro-audit | `test_dl_retro_audit.py` + manual run on `runs/` | re-classification of recorded traces; the false-positive rate of the classifier is reviewed before enforce |
| Architecture | `test_dl_architecture.py`, `test_dl_upstream_untouched.py` | harness-side modules import no pyarrow/pandas (import with both blocked); core modules name no plugin; `git -C third_party/TheVirtualBiotech status --porcelain` empty and HEAD equals the pinned commit; no `__pycache__` created there |
| Determinism (rev 2) | `test_dl_launcher.py::test_hash_seed_effective`, `test_dl_determinism.py` | two launches of a fixture server under the reaper give identical set order despite `-E`; gateway transforms and `_serve` on the S8 fixture give byte-identical headers, rows and `row_keys_sha256` under two `PYTHONHASHSEED` values |
| Matrices and client (phase 2) | `test_dl_formats_v2.py`, `test_dl_native_tools.py`, `test_dl_client.py` | F-12 matrix suite; S4 DepMap fixture through `find`/`aggregate` with symbol, Ensembl, unknown and unscreened genes; h5ad per-fragment overrides; client calls write provenance |
| Soak and calibration (phase 5) | `@pytest.mark.slow` | estimates within ±30% of measured RSS on real 25.09 subsets; witness p95 < 300 ms on S2 tables |

CI tiers: **fast** (no fastmcp, no large pyarrow work, under 60 s), **mcp** (`needs_fastmcp` plus
pyarrow; real upstream servers on fixtures), **slow** (1M-row conformance, memory, soak; Linux only).
Existing tests that change on purpose: `tests/test_mcp_bridge.py` keeps its bridge-level assertions
(lines 46–54, 197 hold with `gateway=None`); `tests/test_preflight.py` gains per-tool cases and its
"missing data blocks the session" cases run with `data.enabled: false`; `tests/test_claims.py` gains
`supports`/status cases with the `kind` enum unchanged.

---

## 22. Rollout

1. **Observe first.** Ship with `data.gateway.mode: observe` for one recorded scenario per server;
   review `data_observe` events and `vbt ds retro-audit` on existing `runs/` (counts of calls that
   would now be `not_found`, `empty`, `tool_defect`, `quarantined`, and claims that cited them).
2. **Enforce in stages** with `enforce_servers`: target and disease first (most-granted tools), then
   drug and association, then the rest. Switch the default to `enforce` at the end of phase 1.
3. **Kill switches**: `data.enabled: false` restores today's behaviour exactly; `mode: observe`;
   per-server `enforce_servers`; per-tool `serve: pass` in an overlay; `profile: fidelity` for
   paper-replication runs. The mode is pinned and recorded per call, so runs stay comparable.
4. **Compatibility**: tool names unchanged (agents.yaml globs keep working; phase 2 adds
   `mcp__data__*`); `CheckResult` legacy labels kept; record_claims `kind` enum unchanged;
   `MANIFEST.degraded.servers` kept.
5. **Release upgrades** (e.g. 25.12): `vbt ds diff-release` reports role-column, type, encoding and
   vocabulary changes; drifted tables become `schema_drift`, so only their tools are `not_ready` until
   the descriptor is updated.
6. **Upstream drift**: each overlay records the upstream commit; the CI drift test hashes each upstream
   tool's input schema and fails on unbound new tools or parameters; unbound tools get the generic guard.
7. **Leakage ceilings (rev 2)**: no-web scenarios set `data.leakage.ceiling` from
   `web.literature_max_date`; until then `scenarios/__init__.py:97-99` and `profiles/no-web.yaml:17`
   keep documenting that ClinicalTrials.gov leaks, and readiness reports the ceiling as unset.
8. **Source aliases (rev 2)**: `data.sources.alias: {open_targets: zenodo_vbt}` serves `open_targets.*`
   bindings from the Zenodo tables that declare `implements`, for hosts that only have the Zenodo
   extract; provenance records both the serving source and the lineage.

---

## 23. Risks and open questions

- **Schema facts to confirm on the full real 25.09 release** (RESOLVED on the real release; footers and value
  facts in `tests/datalayer/real/ot_25_09/`, checked by `tests/datalayer/test_dl_real_ot_schema.py`, opt-in
  with `VBT_DL_REAL_DATA` / `VBT_DL_NETWORK`):
  - `hasSafetyEvent`: only −1 (943 rows) and null (77,783); "no event" is null, so `no_safety_events` is
    `unsupported_filter` and the encoding is `{-1}`.
  - `maxClinicalTrialPhase`: scale [0, 1], observed [0.25, 1.0] (0.25 per phase); `min_clinical_phase` has
    maximum 1.
  - mouse_phenotype: the key is unique over 210,579 rows, with ENSMUSG, `MGI:` and `MP:` id forms.
  - Interaction orientation: every gene pair is stored in both orientations in every source
    (`orientation: both`). In SIGNOR the two rows of a pair swap the biological roles (36,636 rows: 18,221
    with the regulator as A, 18,415 reversed), so the direction comes from the roles
    (`edges.direction_from_roles`).
  - Tahoe `cell_line_metadata`: 1,000 rows, 102 cell names, 99 DepMap ids (5 rows without one); the DE
    shards' `Cell_ID_DepMap` holds the string `NA` for one line, declared missing (R4).
  - pharmacogenomics: the grouping key repeats 9,703 times with no two rows equal; the table is
    content-identified (`row_identity: content_hash`), and so is drug_mechanism_of_action.
  - `homologues.speciesId` is a string.
  - `target.go`: `(id, aspect, evidence, source, geneProduct)` repeats in 12,806 genes; with `ecoId` (nullable)
    the item key is unique over all 821,377 items. `target.pathways` `(pathwayId, topLevelTerm)` is unique.
  - Expression in-band codes: `rna.unit` "" in 2,732,448 items, `rna.level` −1 in 2,525,761, `protein.level`
    −1 in 4,350,738, `rna.zscore` −1 in 3,763,002. `protein.cell_type` repeats a name in 12,998 of 4,940,421
    lists; `(name, level)` is unique.
  - The 25.09 `disease_hpo` file has exactly id, name, description, dbXRefs, parents and obsoleteTerms; `id`
    holds 19,034 `HP_` terms and 12,081 imported terms, so the hpo universe is restricted to `HP_` (`where`).
  - Also found: `so.id` is `SO:NNNNNNN`; `literature.pmid` holds Europe PMC ids (2,349,085 of 151,961,320
    rows are PPR, IND, PMC, CAIN, c or FNI ids), typed `europepmc_id`.
  Still `verified: false`, with the counts in the descriptor: the disease `ontology.leaf` flag, biosample
  closures and chemical-probe coverage. interval and interaction_evidence hold exact duplicate rows
  (interaction_evidence 24,280 of 27,286,700): their rows have no identity (`row_identity: none`), so R5b
  tests no uniqueness and counts are over the rows as stored. Several revision-1
  assumptions were refuted by the stress test on real extracts and are corrected here: `known_drug`
  has no `ctIds` in 25.09 and its key needs nullable `status`; `approvedSymbol` is not unique (1,613
  duplicates); `ontology.leaf` is wrong for 31,635 terms; 29.5% of `disease.id` values use prefixes
  outside the plugin's revision-1 list.
- **Witness and intended semantics diverge**: the witness implements the binding's intended semantics,
  which can disagree with upstream on purpose (that is the point). Where comparison is impossible
  (free text, remote engines) it reports `unknown`; the detector tests and golden snapshots catch
  binding mistakes. A remote engine's free text with an `engine_param` is counted by the same engine
  (CT.gov `query.cond`, PubMed `term`), so those calls do get a total to compare.
- **Overlay and descriptor authoring cost**: 103 tools in phase 1, now with field maps, scope facets and
  item tables. Mitigated by grant-weighted review order, generic verbs, doc-example linting, and a
  safe fallback (generic guard). The phase-1 derived set (27 tools) remains the least certain estimate,
  and phase 1 is 0.5 ew larger than in revision 1.
- **Sidecar indexes in phase 1** add build time at readiness for huge tables (Tahoe `gene_name`, evidence
  `targetId`/`diseaseId`; minutes EST on first build, then cached per fingerprint). If a build exceeds
  its budget the affected tools return `too_large` naming `vbt ds index build --access-paths`.
- **Data child availability**: if it cannot start, guarded tools become `service_unavailable`
  (`when_service_down: strict`). It uses the upstream servers' interpreter, so it fails only when they
  would; preflight probes its imports and every plugin's `requires`.
- **Stripping `-E`** in the reaper replaces an interpreter flag with an environment the reaper builds
  itself. The allow-list is tested; a regression would reintroduce `PYTHONPATH` redirection, so the
  launcher test asserts that a hostile `PYTHONPATH` in the parent does not reach the child.
- **RLIMIT_DATA scope**: Linux ≥ 4.7 only; file-backed and shared mappings are not counted. Phase 4 adds
  cgroups and a watchdog; on macOS the layer falls back to estimates and result caps.
- **Latency**: resolution is an in-process dict lookup; the witness adds one IPC hop and a bounded scan
  (target p95 under 300 ms on composite-key tables, checked in F23); the scope rule adds no scan when the
  witness already ran.
- **More errors may make agents loop**: errors carry candidates, valid values and next steps; existing
  budget controls apply; the addendum explains the contract.
- **Census pinning** cannot be enforced while upstream hard-codes `"stable"`: we record the resolved
  release and detect drift; true pinning needs native Census tools (beyond this plan).
- **HTTP third-party servers** cannot be memory-limited by the harness: only result caps, timeouts and
  classification apply.
- **`require_ready` semantics change** (data no longer blocks the whole session when some tools can
  run): intentional; existing tests that assert blocking run with `data.enabled: false`.
- **In-process readers stay unguarded in phase 1** (N8): Case 1 pandas reads and agent notebooks bypass
  the gateway until the phase-2 client; readiness reports it.
- **Remote tools have no witness until phase 4**: their known miscounts are covered by defect detectors;
  a new upstream miscount on a remote tool would pass until then. (Phase 4: the remote witness counts every
  call to a table whose layout declares `count`, free text with an `engine_param` included.)
- **Upstream prompts** (e.g. "try both Ensembl ID and gene symbol") cannot be edited; resolution makes
  that advice harmless.

---

## 24. Implementation packages

Fifteen work packages in eight waves. Packages in one wave own disjoint files and import only what
earlier waves (or the wave-1 contracts) define. Existing files are each owned by exactly one package
per wave. Revision 2 splits phase 2 into two waves (descriptors and plugins, then derivation and
native tools) and lets every later phase declare its YAML facts in the phase-2 strict descriptors, so
the phase-3 and phase-4 packages never edit the same source descriptor.

| Wave | Package | Phase | Fixes | Owns (summary) |
|---|---|---|---|---|
| 1 | P01 Core contracts, models and catalog | 1 | F1, F2, F5 | `datalayer/{__init__,errors,result,record,settings,api,ipc,roles,predicate,rowkey,catalog}.py`, `descriptor/{models,columns,overlay,scoping,load,lint}.py`, `plugins/{__init__,base,registry}.py`, `plugins/conformance/__init__.py`, doc-example lint test |
| 2 | P02 Identifier plugins and resolver | 1 | F4 | `plugins/identifiers/*` (phase-1 set), `plugins/conformance/identifier.py`, `resolve/{index,rules,resolver}.py` |
| 2 | P03 Format, layout and statistic plugins (phase 1) | 1 | F3, F7 | `plugins/{formats,layouts,statistics}/*` (phase-1 set), conformance format/layout/statistic/golden |
| 2 | P04 Bridge seam, launcher and memory | 1 | F2, F8 | `src/vbt/tools/mcp_bridge.py`, `launch/*`, `memory/*`, fixture servers |
| 2 | P05 Harness and audit plumbing | 1 | F1, F10 | `tools/base.py`, `runtime.py`, `failures.py`, `session.py`, `agents.py`, `audit/claims.py`, `audit/provenance.py`, `tools/provenance.py`, `verify.py`, `pinning.py`, two prompts, `mcp_servers/pubmed_server.py` (stub override), `case_studies/trial_outcomes/schema.py` |
| 2 | P06 Fixtures, stubs and the six correctness tests | 1 | F10 | `tests/datalayer/{conftest,dl_fixtures,dl_upstream}.py`, E-utilities and pybioportal stubs, `test_dl_correctness_six.py`, `test_dl_upstream_untouched.py` |
| 3 | P07 Data child | 1 | F3, F4, F6, F9 | `service/*` (reader, leaf projection, item tables, sidecar indexes, verbs) |
| 3 | P08 Gateway | 1 | F2, F6, F7, F9 | `gateway/*`, `derive/{__init__,schema,text}.py` |
| 3 | P09 Shipped descriptors and overlays | 1 | F5 | `configs/data/**` (phase 1), shipped-config, defect-detector and reads-completeness tests |
| 4 | P10 Integration | 1 | F9, F10 | `preflight.py`, `orchestrator.py`, `config.py`, `configs/default.yaml`, `configs/mcp_servers.yaml`, `cli.py`, `datalayer/cli.py`, `datalayer/retro_audit.py`, architecture, determinism and fixture-readiness tests |
| 5 | P11 Strict descriptors and plugins v2 | 2 | F11, F12 | local source descriptors (OT, Tahoe, Zenodo, DepMap, GO, MSigDB, CL), formats csv/jsonl/h5ad/zarr/obo/gmt/npy, layouts zip_member/http_range, phase-2 identifiers, matrix conformance |
| 6 | P12 Derivation, native tools and the client | 2 | F13, F14 | `derive/*`, public and view verbs, `client.py`, local-server overlays, `agents.yaml`, Case 1 readers, workspace limit |
| 7 | P13 Phase 3 | 3 | F15–F18 | statistic plugins v2, enrich/pairs/network/hierarchy verbs, phase-3 overlays |
| 7 | P14 Phase 4 | 4 | F19–F21 | memory at scale, live plugins and remote witness, remote descriptors and overlays, envelope kind |
| 8 | P15 Phase 5 | 5 | F22–F24 | replay, drift, row-level citations, calibration soak, CI guards, runbook |

---

## 25. Stress test results

**Method.** Revision 1 was stress-tested by one tester per data shape and one per correctness test.
Shape testers wrote a descriptor (and overlay where a tool reads the data) for a real instance, checked
it mechanically against a literal transcription of the revision-1 models and lint rules, probed the
real files with pyarrow 25.0.1, and judged seven behaviours: complete key, resolution, not-found
error, memory limit, readiness, provenance, and derived tools and text. Test testers ran the unmodified
upstream functions (`python -B`; the submodule stayed clean) against fixtures and judged whether the
spec, as written, would make each test pass. Their working files are in the session scratchpad
(`s1/` … `s12/`, `ct1/` … `ct6v/`).

**Findings.** 233 gaps for the shapes (25 blockers, 114 major, 94 minor) and 61 for the tests
(6 blockers, 28 major, 27 minor). Every blocker and major gap is addressed in this revision; minor gaps
are folded in where cheap, and the few that are only partly addressed are listed under "deferred".
Every "today" failure in §19 was reproduced on unmodified upstream code.

**Verdicts after revision** are design-level: each tester's descriptor and calls were re-walked
through the revised models, rules and mechanisms. They become executable when the phase-1 packages
land, and the six correctness tests are the acceptance gate. Legend: **P1** works at the end of phase 1;
**P2/P3/P4** works from that phase; "upstream" means the call stays upstream-served behind the guards.

### 25.1 Shapes

| # | Instance tested | Gaps (B/M/m) | Before (rev 1) | Key changes (sections) | After (rev 2): key · resolution · not-found · memory · readiness · provenance · tools/text | Deferred or residual |
|---|---|---|---|---|---|---|
| S1 | Zenodo OT 25.09 `disease` (39,530 rows) and `drug_molecule` (18,119) | 1/8/11 | key and not-found for absent IDs worked; salt families, retired IDs, xrefs, in-band −1, wrong `leaf` flag inexpressible; 29.5% of disease IDs rejected; no gateway-routed consumer | parent families and `form`, retired/xref rules, `options.prefixes: from_universe`, relation constraints, `missing_values`, placeholders on categories, `implements`/`lineage`, alternate keys, `ColumnSpec` union, `num_values` memory (§6, §7, §11.5, §10.4) | P1 · P1 · P1 · P1 · P1 · P1 · P1 (pass tools); in-process Case 1 readers P2 | in-process pandas reads guarded only from P2 (N8); family fan-out and merge P3 (F17) |
| S2 | Tahoe DE footers + metadata; OT `association_by_datatype_direct` (4.17M rows); `known_drug` (126,689) | 4/13/9 | float32 dose gave silent empties; `incomplete_key` contradictory; null key parts failed readiness; abs thresholds unexpressible; spec examples failed lint | storage-typed literals and snapping, scope kinds/pooling with one rule, nullable keys, abs ops, selectors, `limit_grain`, stored forms and crosswalks, `access_paths` with sidecar indexes, prefix-block key checks, sections, field maps, per-group top-k, censored coverage, examples linted (§6.3, §8.1, §11.3, §11.5, §11.6) | P1 · P1 · P1 (incl. `combination_not_profiled` from P3) · P1 · P1 · P1 · P1 | per-dose comparison and selectivity tools P3 (F17); tuple-universe errors P3 |
| S3 | OT `target` nested containers, `target_essentiality` (3 levels), `expression`, `pharmacogenomics`, `drug_molecule.linkedTargets`, `disease.synonyms` | 3/9/10 | entity keys only; uncorrelated item filters (VERIFIED wrong answer); nested empties citable; footer null counts gave false key violations; estimate ~9× low; list leaves not projectable | item tables, `Any`/`All` with list Kleene rules, nested coverage default `unknown` with `null_means`/`empty_means`, field maps with parent keys, `item_key` checks from leaf scans, per-column readiness, `leaf_projection`, two estimators, path EBNF, reference scoping (§6.2–6.4, §9.2, §10.4, §13) | P1 · P1 · P1 · P1 (DepMap aggregates `too_large` until P3) · P1 · P1 · P1 | streamed DepMap aggregation P3 (F16); calibrated nested overheads P4 (F19) |
| S4 | DepMap 24Q4 CRISPRGeneEffect (models × ~17,900 genes), `Model.csv`, GTEx GCT | 2/11/7 | key not declarable (no `MatrixSpec`); header IDs unparseable; no footer stats; gene crosswalk missing; native args unresolved | `MatrixSpec`/`AxisSpec` with `parse`, `@row`/`@col` paths, long-view semantics, header axes roled as a whole, `stats_scan`, inline manifests, identity vs coverage universes, `maps_to` via crosswalks, `attributes_from` + `aggregate`, native-arg resolution, `cutoff` facet, `ncbi_gene` with required prefix, column aliases (§6.7, §9.2, §10.6, §11.5) | P2 · P2 · P2 · P2 · P2 · P2 · P2 (no upstream tool reads S4) | GTEx-size text matrices need conversion to zarr or a seek sidecar (P2 decision recorded; `size_class: huge` forces it); release-dated symbols resolved through HGNC crosswalk only when an HGNC source is loaded |
| S5 | Zenodo GEO cohorts (dense bulk, VERIFIED), merged lung scRNA object, Census `get_anndata` outputs | 5/14/6 | no `MatrixSpec`, no matrix protocol; nothing reached the gateway; Census outputs had no lifecycle; `is_primary_data` disclosure was false | `MatrixSpec`, `fragment_key` and overrides, `index_name`, `unique_within`, SOMA filter parser with `is_primary_data` enforced, `materialized_by` + `FileCheckSpec` + write-once in P1, `zero_if_measured`, `sample_values` readiness, `requires` probing, in-process client (§6.7, §7, §11.3, §13, §10.6) | key P2 · resolution P2 (native-label for old symbols) · not-found P2 · memory P1 for Census tools, P2 for files · readiness P1 (outputs `awaiting_producer`) · provenance P1 (Census outputs) · tools/text P1 (single_cell), P2 (cohorts) | donor key and donor-balanced sampling per dataset P4 (F20); agent notebooks reading h5ad directly stay outside the gateway (workspace memory limit from P2) |
| S6 | OT 25.09 `evidence/sourceId=*` (23 partitions, 90 columns; chembl and gwas_credible_sets shards, europepmc footer) | 1/11/6 | corrupt shard silently skipped (VERIFIED); missing partition unnoticed; both tools `too_large` at real size; null years with known dates excluded | fragments listed by name, L-7, partition `expect: declared`, partition columns as logical columns with `mirrored_by`, `applies_when`/`by_partition`, time `fallback`, `NonEmpty`, sidecar indexes, per-partition readiness and fingerprints, existence `unknown` (§6.3, §6.8, §9.2, §11.6, §13) | P1 · P1 · P1 · P1 · P1 · P1 · P1 | `search_literature` P4 (F19); cross-fragment key duplicates found probabilistically at standard depth, exhaustively at deep depth |
| S7 | OT 25.09 `disease`, `reactome`, `go`, `disease_hpo`; `mini_cl.obo` | 1/8/8 | include-descendants undefined (literal reading dropped the term itself); GO DAG absent from OT; obsolete terms live; broad synonyms could resolve; `is_leaf` wrong | `IdTypeSpec.hierarchy` and `extends`, the `Eq OR Contains(ancestors)` rule, closure/inverse facets with R10, retired terms, synonym policy, `universe_where`, field maps with dangling-reference handling, `propagated_over`, predicates on hierarchy edges, ordered trims, expansion records (§6.8, §7, §11.5, §11.7, §13) | P1 · P1 · P1 · P1 · P1 · P1 · P1 (expansion through another source's DAG P3) | GO propagation needs the GO OBO source (F11 descriptor, F18 traversal); release skew between OT's GOA snapshot and the OBO is reported, not reconciled |
| S8 | OT 25.09 `interaction` and `interaction_evidence`; upstream `interaction_mcp` | 2/7/7 | `PYTHONHASHSEED` ignored under `-E` (VERIFIED nondeterminism); null `targetB` failed readiness; orientation unexpressible; mixed-source top-k | reaper strips `-E` with an explicit env, nullable keys with `missing: non_entity`, `EdgeSpec` with sides and orientation, unordered grains, per-group top-k, threshold guard, `binds_any`, pair bindings, per-scope coverage universe, specified traversal (§6.8, §11.3, §11.6, §14.2) | P1 · P1 · P1 · P1 · P1 · P1 · P1 (network tools blocked until P3) | `neighbors` traversal and network tools P3 (F17); per-source database versions recorded from P1, compared by replay in P5 |
| S9 | OT `target.pathways`/`target.go`, OT `reactome`/`go`, go-basic.obo, MSigDB Hallmark GMT | 1/8/9 | enrichment contract had no slot (VERIFIED wrong background and family); no enrich verb; gene lists all-or-nothing | `membership.{set, member, count_grain, propagate_via, enrichment}`, item tables as set views, `local_key`, list arguments with `min_resolved_fraction`, statistic `test` capability, R9/R10 membership checks, statistics provenance, coverage for tested negatives (§6.8, §7, §9.2, §11.5, §15) | P1 for membership lookups · P1 · P1 · P1 · P1 · P1 · enrichment tools blocked until P3 | enrichment and GMT/OBO formats P2–P3 (F12, F16); Entrez-keyed GMTs need `ncbi_gene` (P2) and an HGNC crosswalk source |
| S10 | cBioPortal clinical data (stubbed `pybioportal`), `clinical_trial_labels_reconciled.csv`, authors' LUAD `raw_data.csv` | 2/9/7 | phantom samples returned as success and patient values counted per sample (both VERIFIED); censoring, string-typed numbers, per-study IDs, remote not-found inexpressible | `exists_when` + W6, `level` + grain split, `event_of` on the status flag with `CensoredCmp`, `encoding`/`parse`/`missing_values`, pivots, `column_patterns`/`roles_from`, composite universes and refs, `universe_via`, not-found precedence, `key_from_args`, `list_delimiter`, `expose.withhold_from`, `size_from` (§6.8, §8.7, §11.4, §11.6, §11.7) | P1 · P1 · P1 · P1 · P1 · P1 · P1 (pass with transforms); live derived lookups P4 | Kaplan-Meier statistics P3 (F15); per-study release from `importDate` and live record versions P4; survival scripts outside MCP move to the client in P2 |
| S11 | ClinicalTrials.gov v2 through the unedited `clinicaltrials` server (live facts checked 2026-10-06); PubMed and cBioPortal where shape-wide | 1/10/8 | leakage stamps undeclarable while CT.gov leaks outcomes in no-web runs; not-found precedence undefined; aliases looked like wrong entities; totals and order uncheckable | `LeakageSpec`, `leakage_filter` injection and T1, existence `upstream`/`unknown`, echo redirects/synonyms and set echoes, field maps, `escape`/`engine`, order roles with `order_source`, `TotalSpec`, spill before cap, `live` capability declared in P1, `ResultSpec.kind: count`, remote readiness probes (§6.1, §8.1, §11.3, §11.4, §11.6) | P1 · P1 · P1 · P1 · P1 · P1 · P1 | independent remote witness, remote vocabularies and record versions P4 (F20); until then remote miscounts are covered by defect detectors only |
| S12 | OT `literature_vector` (55k × 100 float64; local copy) | 2/6/6 | no similarity verb; no anchor binding, so the witness raised false `tool_defect`s; zero-norm vectors scramble upstream order (VERIFIED) | `anchor` role, union id_types with `kind_from`, computed fields, `veto_labels`, norm constraints at readiness, `order_source: upstream_full_sort`, `evidence_nature` in headers and claims, element-width memory, per-kind readiness degradation (§6.8, §8.7, §9.2, §15.3) | P1 · P1 · P1 · P1 · P1 · P1 · P1 for the pass tools; `similar` verb and repair P2 | the tools are granted to no agent today; native `mcp__data__similar` arrives with the P2 grants |

### 25.2 Correctness tests

| Test | Reproduced on unmodified upstream | Gaps (B/M/m) | Why revision 1 would not pass | Changes | Verdict after revision |
|---|---|---|---|---|---|
| CT-1 identifiers | all 9 cases (incl. live NCBI `PMC1234` → PMID 1234) and the verified claim | 0/4/7 | search order undefined (TP53BP1 sorts before TP53 by key); `PMC1234` fell through to `not_found`; fixture failed shipped readiness; one bundled xfail | match-class ordering in `search`; `pmcid`/`doi` plugins and all-rejected → `invalid_argument`; generated fixtures with manifest and sentinels plus a readiness precondition; per-case xfails with pinned wrong answers and positive controls; closed rule grammar; resolver yields the bound id_type; E-utilities stub; claims through `Runtime` | passes at end of P1 |
| CT-2 not-found / empty / outage | all claims, plus `get_chemical_probes` unknown-ID non-error and swallowed profile sections | 1/4/5 | item rows had no witness or coverage; coverage never reached claims; manifest check source-wide; vacuous sub-assertions; `reads` completeness unchecked | item tables and `rows_of`; coverage in records, index and claims; per-table manifest attribution; off-mode pinned assertions and positive controls; AST `reads` check and per-table fault test; cBioPortal phantom-sample cases folded in | passes at end of P1 (profile view P2) |
| CT-3 false empties | all seven tools; the two-argument pgx call returns rows violating `drug_id` (not zero rows, corrected) | 1/3/6 | `CHEMBL3` absent from `drug_molecule` gave `not_found` first; pmid binding undefined; witness could not count items; fidelity path untested | existence `bound`; pmid binding with existence `off`; witness over item tables; CT-3b pass/fidelity variant; pinned pgx fixture; retired ID and synonym cases on MONDO_0005148; `search_go_terms` and biosample synonym cases | passes at end of P1 |
| CT-4 global top-k | all four tools (L2G fails only with `min_score=0.05`, noted) | 1/4/5 | null-phase key parts made `known_drug` not ready; `excluded_unknown` unreachable for pass tools; list-valued key part unsortable; grains ambiguous | nullable key parts with scalar key; witness `excluded_unknown` and `distinct_counts`; canonical keys; grains `{returned, total}`; `witness.topk` setting; `arg_echo`; either-side endpoint matching; NaN as unknown; file-order preconditions | passes at end of P1 |
| CT-5 arguments and keys | all nine rows | 2/4/3 | `incomplete_key` never fired in derived mode; `method` could not be bound; `sort_by` had no role; error payloads undefined | scope rule in `prepare`; selector role; `order_by`; fixed payloads; declared grains; complete base fixture; float32 `concentration=0.05` case; two-plate case; blocked `compare_direct_indirect` with named alternatives | passes at end of P1 (compare tool derived in P2) |
| CT-6 unknown and negated | all five claims (fixture adjusted so negation and recount discriminate) | 1/9/1 | negated items left parents listed with count 0; W2 fired before the unknown-value transform on null rows; inflation filled with null rows; `count` role outside I6; min/max cannot confirm encodings; NaN passed; `ne` flag form; HPO undeclared | `item_filter`/`drop_empty_parents`/`nest`; witness checks after T1–T6; `unknown_total` inflation and W6 short pages; I6 for counts; distinct-snapshot confirmation; NaN as null; positive `when_true`; HPO id_type; `partial_unknown` coverage | passes at end of P1 |

### 25.3 What remains open

- Facts marked SCHEMA-DEPENDENT (§23) were checked on the real 25.09 release after implementation (§23;
  [DATA_LAYER_REAL_DATA.md](DATA_LAYER_REAL_DATA.md)). The three still `verified: false` stay
  disclosure-only (I9).
- In-process readers (N8) and remote tools without a witness (until P4) are the two places where a
  silent wrong answer can still pass in phase 1; both are reported by readiness and covered by defect
  detectors where the upstream defect is known.
- Wide text matrices at GTEx scale need a seek sidecar or a zarr conversion, decided in P2.
- Census pinning depends on upstream's `"stable"` (N6).

---

## Appendix A. Phase-1 treatment of the 103 bridged tools

Counts: 101 upstream tools (target 16, disease 6, drug 9, association 11, genetics 10, expression 6,
interaction 5, functional_genomics 9, pathway 10, single_cell 11, clinicaltrials 8; the 4
`provenance_mcp` tools are replaced by native harness tools and not bridged) plus 2 PubMed tools.
Phase-1 modes: **pass** = upstream behind every guard (resolution, existence, witness, inflation,
transforms); **derived** = generic verb in the data child; **block** = `quarantined` with alternative.
Totals: 64 pass, 27 derived, 12 block.

| Server | pass | derived | block (phase that unblocks) |
|---|---|---|---|
| target | get_target_info (ordered trims of go/homologues/dbXrefs with counts; `chromosome` promise dropped), get_target_tractability (flag partition; item table), get_target_prioritisation_scores (existence; encoding legend), get_target_safety_profile (inflation, llr order, coverage unknown, upstream `message` dropped), get_target_hallmarks (`num_hallmarks` dropped; nested coverage unknown), get_target_tep, get_chemical_probes (item table `target_chemical_probes` with covered coverage; `num_probes` dropped), get_genetic_constraint, get_subcellular_locations (`num_locations`, `primary_locations` dropped), get_target_class (`primary_class` dropped), get_homologues (`min_identity` 0–100; species and identity on the same item via `Any`; `model_organisms` dropped) | search_targets_by_name, prioritize_targets, get_mouse_phenotype, get_pharmacogenomics | get_comprehensive_target_profile (P2 view) |
| disease | get_disease_info (CURIE, retired IDs, xrefs; trims), get_disease_hierarchy (field map; `Unknown` → dangling-reference count; `is_leaf` dropped while the `ontology.leaf` relation constraint is refuted) | search_diseases_by_name (literal over name and exact/related synonyms; broad/narrow labelled), get_disease_phenotypes (negation), find_diseases_by_therapeutic_area (`universe_where` isTherapeuticArea), find_diseases_by_phenotype (item filter, empty parents dropped, nested recount) | — |
| drug | get_drug_info (parent family disclosure; `mechanismOfAction` promise dropped), get_target_tractability (same_as target), get_drug_warnings, get_drug_mechanisms (exclusive args; literal text), search_known_drugs (inflation, phase order, nullable key, grains incl. parent families) | search_drugs (`drugType` placeholder `Unknown`), get_drug_indications (item table), get_drug_adverse_events, get_pharmacogenomics (same_as target; existence `bound`; `levels` ordinal) | — |
| association | query_associations, get_associations_for_disease, get_associations_for_target (totals; `summary_stats` dropped; output_path confined and write-once; `include_indirect` selector), filter_by_datatype, filter_by_datasource (selector for direct/indirect; enums; inflation disabled because of `output_path`; datatype or datasource unfixed → `incomplete_key` subkind `incomparable_order`), find_similar_entities (anchor; category enum; `order_source: upstream_full_sort`; norm constraint at readiness), compute_entity_similarity (two anchors; `interpretation` vetoed) | query_evidence (top-k per `sourceId`; sidecar indexes; `require_pubmed` → `NonEmpty`; time fallback), get_evidence_by_publication (sidecar index on `literature[]`; existence `off`) | compare_direct_indirect (P2 derived), search_literature (P4 index) |
| genetics | query_gwas_associations (`chr`), query_l2g_predictions (inflation), get_credible_sets (study_type enum), get_variant_annotation (one-to-many → ambiguous), get_study_metadata (`study_id` exclusive; unknown sizes excluded and counted by the witness; inflation by `total + unknown_total`), query_regulatory_regions (witness safety net), query_colocalisation (method selector; method required when both exist), get_colocalisation_by_chromosome (method selector, inflation) | convert_rsid_to_variant_id (all alleles or ambiguous) | get_qtl_colocalization (P3; alternative query_colocalisation) |
| expression | list_available_tissues (`num_expressed_genes` dropped), query_expression_by_gene (tissue exact; field map; in-band `-1` levels unknown), compare_expression_across_tissues (tissue exact; unit disclosed) | query_expression_by_tissue (item table over `tissues[]`, tissue and `min_expression` on the same item, `unknown_when` for no-data codes), search_biosample_ontology (synonyms searched) | find_tissue_specific_genes (P3) |
| interaction | get_interactions (inflation; T matched on either side via `binds_any`; top-k per `sourceDatabase`; species on either side) | search_interactions (undirected pair binding), get_interaction_evidence (either side) | get_interaction_network (P3), find_common_interactors (P3) |
| functional_genomics | query_gene_essentiality, find_essential_genes, compare_essentiality_across_diseases, find_selective_dependencies (exact disease/tissue vocabularies with collision check; overlapping groups → `invalid_argument`; field map over the three-level tree; per-tissue summaries recomputed with `group_by` or dropped; admission may refuse the 10–13 GB load) | query_cell_line_dependency (exact ACH resolution; item table with correlated `Any`), query_drug_perturbation (scope rule on concentration; plate listed; sections for drug and cell-line metadata), find_drugs_affecting_gene (gene sidecar index; stored forms; `limit_grain: drug`) | compare_drug_effects (P3), find_cell_line_selective_effects (P3) |
| pathway | get_go_term_info (obsolete terms → retired rule), get_pathway_info, get_sequence_ontology_term (normalisation, label resolution) | get_gene_pathways (item table; avoids a second `target` copy), search_pathways (ID or name; `gene_count` by aggregate), get_gene_ontology (item table `target_go`; aspect `send_map`), search_go_terms, find_genes_in_pathway (direct membership disclosed; hierarchy P3) | get_pathway_enrichment (P3), get_go_enrichment (P3) |
| single_cell | get_census_info (release recorded), list_metadata_values (zero-count rows dropped; true distinct count), search_genes (normalisation), query_cell_metadata (key columns injected via `projection`; `column_stats` dropped; output_path confined), get_anndata (SOMA filter parsed; `is_primary_data` enforced unless `include_duplicates`; `FileCheckSpec` and `materialized_by`; write-once), count_cells (`recommendation` dropped when 0), get_gene_statistics (`sparsity` undefined when unmeasured), summarize_datasets, get_cell_type_tissue_matrix (capped counts dropped), get_expression_for_genes (`genes_found`/`genes_not_found`/`summary_stats` dropped until P4; `max_cells` truncation disclosed), get_anndata_donor_balanced (accepted only when the parsed filter fixes one `dataset_id`, else `unsupported_combination` naming get_anndata; `n_donors`, `donor_summary` dropped; "No cells found" → `empty_unverified`) | — | — |
| clinicaltrials | get_clinical_trial_details (NCT syntax; existence `upstream`; echo accepts alias redirects; leakage T1), count_clinical_trials and search_clinical_trials (country default disclosed; Essie-escaped and wrapped; leakage filter injected when a ceiling is set; 1000 cap → `truncated`; `partial_when: $.warning`; `sort` as `order_by` with `order_source: source_server_side`), get_all_cancer_types, search_studies (lower-case), get_study_details (not_found before legacy errors; echo from the record), get_clinical_data (`exists_when` → phantom samples `not_found`; `min_items`; patient level split; study 404 → `not_found`) | — | clear_trial_cache (hidden no-op) |
| pubmed | search_pubmed (count 0 → `empty_unverified`), fetch_abstracts (`each` list with `max_items: 50` instead of silent truncation; PMID plugin rejects PMCID/DOI as `invalid_argument`; `echo_set` subset) | — | — |

---

## Appendix B. Synthesis record

**Base**: the correctness-first design (witness, limit inflation, quirk-detector tests, invariant I9,
off/enforce tests, retro-audit, translate-or-derive for wrong key columns, echo reconciliation).

**Grafted from the extensibility-first design**: pyarrow-free gateway plus a harness-owned `data`
child for all reads; conformance suites with the identifier confusion matrix, format F-1 (no ndarray)
and F-3/F-9; the Predicate IR compiled per format with correct quoting; `returns_key`-style
`incomplete_key`; gateway-only arguments stripped before upstream (FastMCP schemas forbid extra
properties); `RLIMIT_DATA` rather than `RLIMIT_AS`.

**Grafted from the operations-first design**: absence citations require covered coverage; grant-weighted
triage; the `scope` role and its key lint rule; cold-call serialisation, residency ledger,
recycle-to-evict and learned refusals; an exit marker written by a parent before EOF; stat-only
per-turn readiness; architecture tests; staged rollout with kill switches and `profile: fidelity`.

**Flaws deliberately avoided**:
- re-sorting a possibly truncated prefix and calling it ranked (we refuse with `unranked_truncation`);
- citing unverified empties with only a warning (`empty_unverified` is uncitable both ways);
- citable absence without coverage proof in pass mode (absence requires `covered`);
- pyarrow scans inside the orchestrator (all reads in the data child);
- an exec-only launcher that cannot observe the child's exit (we fork a reaper);
- `RLIMIT_AS` false failures;
- phase-1 tests depending on later-phase features (readiness, encoding confirmation and incomplete_key
  are all phase 1);
- disclosure-only notes for known-wrong semantics (negation, value:false buckets and null sizes are
  transforms, not notes);
- a closed executable quirk vocabulary or per-dataset transforms in core (generic mechanisms plus
  `serve` modes; defects are documentation with detector tests);
- in-process monkeypatching of upstream loaders;
- a mandatory relay as a single point of failure (optional, phase 4, with fault tests);
- builtin plugins gated on stamps in a data directory (CI runs the suites instead);
- more than the four named base kinds (the fifth, `envelope`, is added in phase 4 as the explicit example);
- launching the data child with `-E` without a `sys.path` bootstrap;
- miscounting tools (101 upstream + 2 PubMed = 103; the 4 provenance tools are not bridged);
- treating forced `PRELOAD_MCP_DATA=0` as a fix (upstream defaults to `"0"`).

**Revision 2 (stress test)**: one tester per shape wrote a real descriptor against the revision-1
models and lint rules and checked the six required behaviours (complete key, resolution, not-found
error, memory limit, readiness, provenance) plus derived tools and text; one tester per correctness
test reproduced the claimed failure on unmodified upstream code and judged whether the spec could make
it pass. Their findings are folded in as marked (rev 2) and summarised in §25. Ideas adopted from the
testers' proposals: item tables and item quantifiers; nullable key parts; storage-typed literals and
canonical row keys; scope kinds with pooling policies and one scope rule in `prepare`; qualified
id_types with identity versus coverage universes; retired, xref and crosswalk resolution with stored
forms and parent families; `MatrixSpec` with header parsing and a logical long view; field maps and
computed fields; `exists_when`, levels and censoring for clinical records; `LeakageSpec`; anchors and
union id_types for embeddings; the effective hash seed; partition completeness and the ban on
`exclude_invalid_files`; per-column and per-partition readiness; the two-estimator memory model;
capability-tagged conformance; doc examples linted in CI. Proposals deliberately not adopted: making
`scope` a pure facet with no role (the role stays as shorthand, so the vocabulary keeps 20 roles); a
seventh correctness test for phantom samples (folded into CT-2 so the plan keeps six); a separate
plugin kind for set tests or remote requests (optional capabilities of existing kinds instead, so the
core still changes only to add `envelope`).
