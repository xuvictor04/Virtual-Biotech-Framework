# Data layer on real data

Through phase 5, the data layer ([DATA_LAYER.md](DATA_LAYER.md), status in
[DATA_LAYER_STATUS.md](DATA_LAYER_STATUS.md)) was tested only on generated fixtures, stubs and recorded
responses. On 2026-10-08 and 2026-10-09 the descriptors, the data child and the gateway were run against the
real releases and live APIs listed below, in three rounds. This page records what was checked, which
descriptor facts held, which were wrong and how they were fixed, the six correctness tests on the real Open
Targets release, the memory and latency that were measured, and what is still unchecked because the data or
a model run could not be obtained here.

The third round (sections 8 and 9, and §4.3) also covered the rest of the path an owner follows on their own
host: acquisition from the publishers, the host bring-up (`vbt setup`), memory limits that scale with the
host, host certification (`vbt validate`), projects whose descriptors and utilities the system created on real
files, and one end-to-end agent run with a small model on CPU.

Every number on this page was observed in those runs; nothing is extrapolated unless it says so.

* **Machine.** 4 CPUs, 16,094 MB RAM (a 13,680 MB memory cgroup limit on the agents' commands, cgroup v1),
  no swap, no GPU, Linux 6.18. Outbound HTTPS went through a proxy. Other jobs sometimes ran at the same
  time, so wall-clock times are approximate. This machine is where the harness was built and smoke-tested;
  the owners run it on their own hosts, whose limits scale with their memory (§5.7).
* **Upstream code.** The authors' MCP servers were launched unchanged from `third_party/TheVirtualBiotech`
  through `MCPBridge`. "Off" below means `data.gateway.mode: "off"` (the upstream answer as an agent gets it
  today). "Enforce" means the gateway guards the server.
* **Memory.** Memory figures are peak RSS. They come either from the reaper's status file of one process,
  or from the summed VmRSS of a whole process tree sampled every 0.25-1 s. Runs kept their process tree
  under a 5,500-6,000 MB cap. The one early check that ran without the reaper is named in section 5.2.
  "MB" in memory figures is MiB (2^20 bytes), as in the limits, admission and the reaper; download sizes
  are decimal (GB = 10^9 bytes).
* **Oracle.** Expected answers come from pyarrow reads of the same files, written independently of the
  code under test.

## Contents

1. [Sources and releases](#1-sources-and-releases)
2. [Descriptor facts: confirmed and corrected](#2-descriptor-facts-confirmed-and-corrected)
3. [The six correctness tests on real data](#3-the-six-correctness-tests-on-real-data)
4. [Wrong answers of the layer itself, found on real data](#4-wrong-answers-of-the-layer-itself-found-on-real-data)
5. [Memory and latency](#5-memory-and-latency)
6. [What still needs data or model runs](#6-what-still-needs-data-or-model-runs)
7. [Re-running the checks](#7-re-running-the-checks)
8. [Round 3: derived live routes, item keys in Arrow, Census admission](#8-round-3-derived-live-routes-item-keys-in-arrow-census-admission)
9. [Round 3: acquisition, host bring-up, host-scaled memory, projects, an end-to-end run](#9-round-3-acquisition-host-bring-up-host-scaled-memory-projects-an-end-to-end-run)

## 1. Sources and releases

| Source | Release observed | URL | What was read |
|---|---|---|---|
| Open Targets Platform | 25.09 (`manifest.json` started 2025-08-18, modified 2025-09-17) | `https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.09/output/` | The footers of all 3,508 Parquet shards of the 38 tables, read with HTTP range requests (7,016 requests, 197,646,123 bytes). 31 tables downloaded whole: 730 files, 1,925,429,912 bytes, every file's sha1 equal to the release's `release_data_integrity`. Samples of two more tables: `literature` (4 of 334 shards, plus range reads of the `pmid` column of all 334) and `interval` (2 of 83 shards). |
| Tahoe-100M | Hugging Face `tahoebio/Tahoe-100M`, revision `2dc57900b7981cfcf5e211527169a0b006546a95` (the revision upstream pins; still `main`) | `https://huggingface.co/datasets/tahoebio/Tahoe-100M` | The gene, drug, cell-line and sample metadata (1.4 MB, sha256 equal to the LFS object ids). The footers of all 1,026 DE shards (5.96 GB, 529 s). 13 complete contrasts read with range requests. 6 of those were prepared with the unmodified `tools/prepare_tahoe.py`. |
| DepMap | Public 24Q4 (figshare article 27993248, published 2024-12-10) | `https://ndownloader.figshare.com/files/<id>` (md5 checked) | `Model.csv`, `CRISPRGeneEffect.csv`, `CRISPRGeneDependency.csv`, `CRISPRInferredCommonEssentials.csv` (812 MB) |
| Gene Ontology | `go-basic.obo`, data-version `releases/2026-07-26` | `http://purl.obolibrary.org/obo/go/go-basic.obo` | the whole file (32,227,785 bytes) |
| Cell Ontology | `cl-basic.obo`, `cl/releases/2026-06-08` | `http://purl.obolibrary.org/obo/cl/cl-basic.obo` | the whole file (3,347,518 bytes) |
| MSigDB Hallmark | 2024.1.Hs (2026.1.Hs compared) | `https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt` | the whole file (48,690 bytes) |
| Zenodo 22259123 | the authors' case-study archive, Case 1 subset | local extract `data/zenodo/virtualbiotech_submission` | full reads of the six declared tables |
| ClinicalTrials.gov | API v2, `apiVersion` 2.0.5; `dataTimestamp` 2026-10-07T09:00:06, then 2026-10-08T09:00:05 | `https://clinicaltrials.gov/api/v2` | live requests |
| cBioPortal | public API (550 studies) | `https://www.cbioportal.org/api` | live requests |
| NCBI E-utilities | esearch, esummary, efetch; the PMC ID converter | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils`, `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/` | live requests |
| CELLxGENE Census | `stable` = 2025-11-08 (`latest` 2025-11-17), `census_schema_version` 2.4.0; read with cellxgene-census 1.18.0 and tiledbsoma 2.3.0 | `https://census.cellxgene.cziscience.com/cellxgene-census/v1/release.json`, `s3://cellxgene-census-public-us-west-2/cell-census/2025-11-08/soma/` | counts, obs reads, an `X["raw"]` slice, and `get_anndata` through the upstream server |
| Open Targets Platform 25.06 (round 3) | `release_data_integrity` of 2,952,307 bytes | `https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.06/` | `drug_warning` (1 shard) and `go` (4 shards), 1,172,632 bytes, fetched by a descriptor the system wrote in a project (§9.4) |
| HGNC complete set (round 3) | file of 2026-10-06 (`Last-Modified` 13:38:36 GMT, MD5 `af43fd56...`) | `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt` | the whole file (16,973,125 bytes; 45,233 rows x 53 columns), registered in a project (§9.4) |
| ClinVar `gene_condition_source_id` (round 3) | file of 2026-10-07 (`Last-Modified` 14:17:23 GMT), retrieved 2026-10-09 | `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id` | the whole file (1,323,448 bytes; 14,211 rows x 9 columns), a dataset with no shipped descriptor in the end-to-end run (§9.5). The live file had changed by the end of the day (1,323,651 bytes, `Last-Modified` 2026-10-09 14:17:40) |

The DepMap portal's download API (`https://depmap.org/portal/api/download/files`) answered HTTP 200 with a
4,842-byte Cloudflare "verify you are a person" page, so the newest DepMap release on figshare (24Q4) was
used. Newer releases exist only on the portal.

**Downloaded files.** They are in the shared, git-ignored directory `data/real/<source>/<release>/` and are
not committed: about 3.0 GB of data (Open Targets 1.9 GB, DepMap 0.81 GB, the rest under 0.1 GB each), laid
out as the acquisition homes `vbt data acquire` writes (`<source>/<release>`), plus 3.8 GB of model weights
for the end-to-end run. The test suite carries only small snapshots of what was measured, each naming its
source URL and retrieval date:

| Directory under `tests/datalayer/real/` | Contents |
|---|---|
| `ot_25_09/` | per table: source URL, shards, rows, row groups, bytes, writer, the Arrow schema and per-leaf statistics. `_release.json` (manifest, integrity file). `values.json`: the vocabularies, codes, scales, keys and orientations measured on the data. |
| `ot_25_09_servers/` | the footer statistics of the 25.09 `target` table, used to test the shipped memory factors |
| `tahoe/` | DE footer facts, metadata excerpts, 30 prepared DE rows |
| `depmap/`, `ontology/` | CSV excerpts of the DepMap files; whole OBO stanzas and three GMT lines |
| `live/` | recorded CT.gov, cBioPortal and E-utilities responses and replayed request sequences; Census read summaries; `live/round3/` the round-3 exchanges (9 files, 68 KB) |
| `acquisition/` | the listings the acquisition transports parse: the Hugging Face tree of the Tahoe metadata, the figshare file list of DepMap 24Q4, the Zenodo record, an S3 listing page of the Census (12.6 KB) |
| `ot_25_09/_release_data_integrity.excerpt.json` | an excerpt of the release's checksum list (3.9 KB) |

`tests/datalayer/real/record_live.py` re-records the replayed live exchanges through the shipped descriptors
(`python tests/datalayer/real/record_live.py real_live <scenario>`, or `round3 --all --dry-run` to see what it
would write).

## 2. Descriptor facts: confirmed and corrected

### 2.1 Open Targets 25.09

**Confirmed.**

* **Tables and schemas.** All 38 declared tables are in the release. Each has exactly one schema across its
  shards. The `evidence/` table has 23 `sourceId=` partitions, and in each one `datasourceId` equals the
  partition value.
* **Keys and scales from the footers.** No flat key part of a checked key has a null in the footers. All 38
  declared measure scales that have footer min/max hold on those bounds.
* **The `verified: false` facts.** 41 of the 50 facts marked `verified: false` at the start were confirmed.
  They are now `verified: true`, with the 25.09 counts in the descriptor. Among them:
  * `target.go` `aspect` takes C 323,220 / F 255,814 / P 242,343.
  * disease `ancestors` and `descendants` equal the transitive closures of parents and children for all
    39,530 terms.
  * The literature key `(pmid, keywordId)` is unique over all 151,961,320 rows. This was a remote column
    read: 1,669 requests, 1.42 GB, 243.8 s.
  * `literature_vector.norm` equals the L2 norm of a 100-dimensional vector.
  * `interactionScore` lies in [0.15, 1.0] and is null exactly in the reactome and signor rows.
  * Every interaction pair is stored in both orientations in every source: string 12,990,032, intact
    1,094,900, reactome 49,310 and signor 35,070 ordered pairs.
  * All 27,286,700 `interaction_evidence` records match one of the 14,524,000 `interaction` rows (64-bit
    hashed keys of every row of both tables).
  * `maxClinicalTrialPhase` lies in [0, 1]; the values are 0.25, 0.5, 0.75 and 1.0.
* **Keys measured on whole tables.** `l2g_prediction` `(studyLocusId, geneId)` is unique over 1,651,053
  rows. The keys of mouse_phenotype (210,579 rows), both openfda tables, drug_warning and literature are
  unique. known_drug is unique with NULLS NOT DISTINCT over 253,442 rows (`status` is null in 11,951).

**Wrong, and corrected.** The fixes are in `configs/data/sources/open_targets.yaml`, the overlays, or the
readiness checks:

| Before | 25.09 holds | Now |
|---|---|---|
| disease `directLocationIds`, `indirectLocationIds`, `ontology.name`; disease_hpo `code`, `namespace`; interval `datatypeId`; interaction `intASource`/`intBSource`, `speciesA/B.scientificName`/`taxonId` | absent | `optional: true` with a note |
| interaction `speciesA/B.scientificName`, `taxonId` | renamed `scientific_name`, `taxon_id` | both spellings declared; the interaction overlay binds the 25.09 one |
| (not declared) | credible_set `isTransQtl`, `ldSet`, ...; variant `variantEffect`, `transcriptConsequences`, `alleleFrequencies`, `dbXrefs`; study `sumstatQCValues`; interval `resourceScore`; interaction_evidence species, sources, PSI-MI method fields | declared with roles (strict readiness would report `schema_drift`) |
| `so.id` is `SO_0001583` | `SO:0001583` in `so`, `SO_0001583` in `variant.mostSevereConsequenceId` | `SO:` canonical, `SO_` a declared stored form |
| `hasSafetyEvent` codes {-1, 0} | only -1 (943 rows) and null (77,783): "no event" is null | encoding {-1}; `prioritize_targets(no_safety_events=...)` is `unsupported_filter` |
| `maxClinicalTrialPhase` on the 0-4 phase scale | [0, 1], 0.25 per phase | `min_clinical_phase` has maximum 1 |
| `isCancerDriverGene`, `hasTEP` codes | only -1/null and 1/null | re-encoded: -1 = driver gene (signed factor), 1 = a TEP exists |
| `homologues.isHighConfidence` codes | the strings "1" (171,042), "0" (97,128) and "NULL" (3,552,426) | "NULL" read as null |
| `target.go` item key `(id, aspect, evidence, source, geneProduct)` | repeats in 12,806 genes (98,095 items) that differ only in `ecoId` | six-part key with nullable `ecoId`, unique over 821,377 items |
| interaction key `(sourceDatabase, intA, intB, targetA, targetB)` | repeats 6,981 times over 14,524,000 rows | key extended with the two biological roles (unique) |
| R5b tested the grouping key of the content-identified pharmacogenomics and drug_mechanism_of_action tables: a false `key_violation` | the grouping keys repeat 9,703 and 4,651 times; no two rows are equal | R5b tests that no two rows are equal (`row_identity: content_hash`) |
| interaction_evidence, interval rows identified by content | 24,280 exact copies in interaction_evidence (whole table); 67 in 2 interval shards | `row_identity: none`: copies are counted as stored |
| `literature.pmid` holds PMIDs | 2,349,085 rows hold Europe PMC ids: PPR 1,659,430, IND 388,537, PMC 301,096, CAIN 12, c 8, FNI 2 | id type `europepmc_id` |
| literature `pmid` lookups pruned by row-group statistics | each of the 334 shards is one row group spanning the whole pmid range (every range contains 2, 9, 28304224 and 39000000) | `sidecar_index` on `pmid` |
| the hpo universe is `disease_hpo.id` | 19,034 `HP_` terms and 12,081 imported terms (UBERON 5,624, GO 2,518, ...) | the universe is restricted with `where` (R4b now applies it) |
| the id patterns of biosample, gwas_study and interval `studyId` | biosample ids include URLs and other forms; study ids include eQTL Catalogue ids and 4 hyphenated UKB_PPP ids; interval `studyId` holds ENCODE file accessions | biosample canonical `^\S+$`; study patterns extended (0 of 1,964,234 rejected); interval `studyId` `resolvable: false` |
| expression `protein.cell_type` (not declared) | 1,109,986 items; a `name` repeats within 12,998 of 4,940,421 lists; `(name, level)` unique | declared optional with item key `[name, level]` |
| SIGNOR edges directed by `directed_when: {sourceDatabase: [signor]}` | both orientations stored; 36,636 SIGNOR rows, 18,221 with the regulator as A, 18,415 reversed | direction from the biological roles (`edges.direction_from_roles`) |
| target_go grains `gene: [id]` | inside the item table `id` is the GO term's own id: PCSK9 counted 63 "genes" | `gene: [/id]`: 1 gene, 63 terms, 142 rows; lint warns about the pattern |
| interaction endpoints are unversioned ENSG ids | 543 distinct endpoints are not a `target.id`; 65 are versioned human ENSG ids, and 61 of them name target genes (684 interaction rows, 694 interaction_evidence rows, never in the unversioned form) | stored forms `as_stored`, columns `integrity: partial`; the resolver maps ENSG00000198972 to the stored ENSG00000198972.3 |
| `ENSG..._PAR_Y` normalizes to the X-chromosome gene | 25.09 stores the Y copy under its own id (64 Y-chromosome PAR genes; 40 share a symbol with an X copy) | `ambiguous` between the X and Y genes |

**Still `verified: false`**, with the counts in the descriptor:

* disease `ontology.leaf`: false for all 39,530 terms, while 31,635 have no children. The field is dropped.
* biosample `ancestors`/`descendants`: equal to the closures for 27,680 and 33,029 of 35,007 terms; 4,978
  parent links have no inverse.
* chemical-probe coverage: 917 genes list probes, 77,809 are null and none has an empty list, so "none
  listed" cannot be told from "not covered".

The footer read also found a bug in the harness: remote `footer_stats` reported Parquet physical types and no
min/max, while the same file read locally reported Arrow types and min/max. Remote and local reads now share
one code path. Value scans and sidecar footers, which read local files only, now also work on `http_range`
tables through ranged reads.

### 2.2 Tahoe-100M

**Confirmed.**

* **Schema.** All 1,026 DE shards have one schema, and its 16 columns are exactly the declared ones.
* **Size.** There are 4,089,820,780 source rows: 65,218 contrasts times the 62,710 genes of
  `gene_metadata`. No gene is duplicated within a contrast (13 contrasts checked row by row).
* **Doses.** `concentration` is float32, with values 0.05, 0.5 and 5.0 uM. Each of the 379 drugs has
  exactly three (1,137 drug-dose pairs).
* **Plates.** The DE shards name plates `1`..`14` and `sample_metadata` names them `plate1`..`plate14`.
  The declared `strip_prefix` alias maps one to the other exactly.
* **padj.** In each of the 13 contrasts, `padj` equals Benjamini-Hochberg over that contrast's rows with a
  non-null padj (maximum absolute difference 7.3e-8).
* **gene_metadata.** `gene_symbol` and `ensembl_id` are each unique.

**Corrected.**

* Two DE drug spellings carry a trailing space, `'Erdafitinib '` and `'Selinexor '`. They are now stored
  forms; before, R9 reported them as dangling references.
* `sample_metadata.drug` holds the vehicle `DMSO_TF`. It is now a reference with `integrity: partial`.
* One DE cell line has `Cell_ID_DepMap` `'NA'` (hTERT-HPNE). It is declared as a missing value.
* `drug_metadata.targets` was declared `|`-delimited. It is comma-delimited: 123 values contain `,` and 122
  contain `, `. The one exception, Clobetasol propionate, is `CYP3A5,CYP3A4`.
* The coverage text said 76% of `padj` is null. The footers give 79.6% (3,257,147,608 of 4,089,820,780).

After the fixes, all seven `tahoe_100m` tables are `ready` on the prepared sample (deep check 17 s).

### 2.3 DepMap 24Q4

**Confirmed.**

* **Matrices.** Both are 1,178 models by 17,916 genes. All 17,916 headers parse as `SYMBOL (ENTREZ)`.
  The 180,307 unmeasured cells are empty strings, and every row's model is in `Model.csv`.
* **Sentinel.** ACH-000001 / RPL3 (6122) has gene effect -2.1308 and dependency 1.0.
* **Common essentials.** The file lists 1,523 genes, all present in the gene-effect header.
* **Encoding.** The `SerumFreeMedia` encoding holds.

**Corrected.** `Model.csv` has 11 columns the strict descriptor did not declare, so the check reported
`schema_drift`. They are now declared optional and release-dependent, with the placeholder `Unknown`. After
the fix, all four tables are `ready` (deep check 378 s, peak 419 MB).

### 2.4 Ontologies and gene sets

* **GO.** 48,340 terms, 10,248 of them obsolete. Every one of the 48,165 GO terms in Open Targets 25.09's
  `go` table is in the file. `gene_ontology.term` is `ready`.
* **Cell Ontology.** 3,540 terms. Nine obsolete `CP:` ids were moved into CL; they are now accepted.
  `relationship` lines use CURIE predicates, and all 399 targets are CL terms: the descriptor's comment that
  part_of edges point into UBERON was wrong. CL:4023064 asserts the same synonym twice, so synonym items are
  keyed by position. Now `ready`. Since round 3 the descriptor reads the pinned 2026-06-08 release only and
  checks the file's `data-version`; the 14-term test fixture it used to fall back to is `stale` (§9.6).
* **MSigDB Hallmark.** 50 sets, 7,322 memberships, 4,384 symbols. 4,377 of the symbols (99.84%) are an
  Open Targets 25.09 `approvedSymbol`, and the lowest set resolves 99.1%, so `min_resolved_fraction: 0.95`
  holds. Now `ready`.

### 2.5 Zenodo case-study archive

All six tables are `ready` (93-100 s, 730-749 MB).

**Confirmed and set `verified: true`.**

* disease `ancestors`/`descendants` are the transitive closures.
* `drug_molecule.maximumClinicalTrialPhase` is 4 exactly when `isApproved`.
* `ibd_cohorts.X` holds log2 intensities: float32, no NaN or negative values.

**Found and declared.** The cohorts' obs columns `response_remission`, `mayo_score`, `cohort` and
`platform`, which were not declared. `obsm` is absent from every file, so it is now optional.

**Confirmed from the comments.** `chembl_clinical_nct` has 488,361 rows, and its key `(nct_id, drugId,
targetId, diseaseId)` is unique.

### 2.6 Live sources

| Descriptor or code assumed | Observed | Now |
|---|---|---|
| CT.gov `pageSize=0` makes a count request cheap | the registry ignores it: 142,198 bytes and 10 full studies per count | count requests add `fields=NCTId` (790 bytes) |
| a CT.gov find with a limit | read ten pages of 100 (11,757,502 bytes, 11.9 s) to return 2 rows | pages sized by the limit: 2.2-3.3 s |
| the release of a live result is the HTTP `Date` | CT.gov's `/version` gives the registry's `dataTimestamp`; each cBioPortal study has an `importDate` (a space, no zone: not ISO 8601) | CT.gov `dataTimestamp`; a cBioPortal study's `importDate` |
| cBioPortal `pageNumber` is an offset | it is a page index: 500 of `msk_impact_2017`'s 10,336 patients were returned as complete | 20 pages, `truncated: true`, total 10,336 |
| any user agent works | `Python-urllib/3.11` gets HTTP 403 | the client sends `vbt-datalayer` (pinned by a test) |
| an unknown study is an empty answer | HTTP 404 `Study not found`, which escaped as a crash | `not_found` naming `studyId` |
| a `{stem}_STATUS` encoding for survival endpoints | PFS and DSS use other codes, and the pattern also matched `PERSON_NEOPLASM_CANCER_STATUS` | OS, DFS, PFS and DSS declared one by one with verified encodings |
| 8 patient-level clinical attributes | 30 attributes are patient-level in every one of the 550 studies that defines them | the patient-level list has 30 |
| a PMCID can be passed as a PMID | `efetch id=PMC6309485` returns PMID 6309485, a different article; the converter maps PMC6309485 to PMID 30009548 | refused before the call (`looks_like: pmcid`) |
| Census `get_anndata` obs/var keys are the index | `soma_joinid` and `feature_id` are columns; the index is positional | `from: column` |
| Census `X["raw"]` holds counts with missing as zero | a 200 x 3 slice held 65 stored values, all whole numbers, no stored zeros | `verified: true` |
| donor_id names a donor | it is unique only within `dataset_id`: spleen has 58 labels but 64 (dataset, donor) pairs | the donor key is `(dataset_id, donor_id)` |
| Census reads fit the process limit | `open_soma` reserves 1 GiB per column read; under `RLIMIT_DATA` reads fail with `std::bad_alloc` at 164-210 MB resident (the upstream `get_anndata` still failed at 12,000 and 40,000 MB) | the data child uses 32 MiB buffers for counts and 16 MiB for row reads; the `single_cell` server runs with `limit_kind: rss` (a memory cgroup, else the RSS watchdog) |

The Cox analysis of `vbt.analysis.survival` on CD276 in LUAD
(`luad_tcga_pan_can_atlas_2018`) read through the data client reproduces the archived values in
[ZENODO_REPLICATION.md](ZENODO_REPLICATION.md):

| Endpoint | HR | 95% CI | p | n | Events |
|---|---|---|---|---|---|
| OS | 1.619447 | 1.053621-2.489138 | 0.027940 | 238 | 95 |
| DFS | 2.055312 | 1.083938-3.897186 | 0.027323 | 142 | 43 |

The run took 3 requests, 3.7 s and 222 MB.

## 3. The six correctness tests on real data

The six tests of [DATA_LAYER.md §19](DATA_LAYER.md#19-the-six-correctness-tests) were run on the 31
downloaded Open Targets 25.09 tables. They used the unmodified `target` and `drug` servers, with the
gateway off and with it enforcing, and compared both against the pyarrow oracle.

| Test | Call | Off: the real wrong answer | Enforce | Oracle |
|---|---|---|---|---|
| CT-1 identifier form | `get_target_info` with "PCSK9", "ENSG00000169174.12", "ensg00000169174", "NARC1" | "Target ... not found" for all four | the PCSK9 row, resolved by `label_exact:approvedSymbol`, `normalized:strip_version`, `normalized:upper`, `synonym:alias` | ENSG00000169174 |
| CT-1 | `get_drug_info` "chembl25", "CHEMBL:25"; `get_disease_info` "EFO:0000685" | not found | resolved to CHEMBL25, EFO_0000685 | present |
| CT-2 unknown id | `get_target_info`, safety, prioritisation, chemical probes and profile with "ENSG00000999999" | an error text, or `success` with count 0 ("No adverse event data") | `not_found`, not citable, for all five | absent |
| CT-3 name search | `search_targets_by_name("TP53", limit=1)` | TP53AIP1 (first in file order) | TP53, `match: exact`, total 71 | TP53 |
| CT-4 top-k | `search_known_drugs(PCSK9, limit=5)` | phases 3, 3, 3, 3, 2, although 23 phase-4 rows exist; with "PCSK9" as the id, count 0 | phases 4, 4, 4, 4, 4: the oracle's five rows, total 110 (also for "PCSK9") | total 110 |
| CT-5 unknown filter | `prioritize_targets(no_safety_events=True)` | 20 targets, 3 of them with `hasSafetyEvent` -1 (an event recorded) | `unsupported_filter` | 943 targets have -1 |
| CT-5 | `filter_by_datasource("gwas_catalog")` | `success`, count 0 | `invalid_argument` listing the 23 real datasource ids | not a datasource |
| CT-6 wrong values | `search_known_drugs` with limit 0 / -1 | count 0 "No known drugs" / 109 rows (all but the last) | `invalid_argument` for both | 110 rows |
| CT-6 | `prioritize_targets(sort_by="nonexistent")`; colocalisation method "colc" | 100 unsorted rows; "dataset not found" | `invalid_argument` for both (the valid methods are coloc and ecaviar) | |

The same runs caught more wrong answers. In each case the gateway's answer equals the oracle:

| Call | Off | Enforce | Oracle |
|---|---|---|---|
| `get_pharmacogenomics(drug CHEMBL3)` | count 0 | total 508 | 508 |
| the same with target ENSG00000112038 | 20 rows, 14 of them for another drug | total 73, none for another drug | 73 |
| `get_mouse_phenotype(PCSK9)` | 0 | 18 | 18 |
| `get_gene_ontology(PCSK9)` | 0 | 142 | 142 |
| `search_go_terms("GO:1")` | 0 | total 2,025 | 2,025 |
| `search_diseases("NIDDM")` | 0 | 2 | |
| `search_diseases("blindness (disorder)")` | 0 | MONDO_0001941 | MONDO_0001941 |
| `find_diseases_by_phenotype("HP:0001250")` | 0 | total 1,389 | |
| `search_biosample_ontology("acide hyaluronique")` | 0 | CHEBI_16336 | CHEBI_16336 |
| `compare_direct_indirect(PCSK9)` | "9 unique to direct" (from two top-10 lists) | no direct-only target; total 1,936 | direct 992, indirect 1,936, none direct-only |
| `get_study_metadata(min_sample_size=100000, limit=20)` | 20 rows in file order, no total | `tool_defect` (W3): the 20 are not the top 20; witness total 19,930 | 19,930 |
| `get_interactions(PCSK9)` without a source | | `incomplete_key`: scores compare only within a source, so retry with one of the four sources | |
| `get_interactions(PCSK9)` with a source | | `too_large`: upstream loads all 14.5 M interaction rows | intact 105 rows, string 2,651 |
| `get_interaction_network(PCSK9, 1 hop, 50)` | | served derived, 101 rows | |

**Pinned tests.** Two opt-in tests in `tests/datalayer/test_dl_real_ot_servers.py` pin these results. Both
need `VBT_DL_REAL_DATA` and the upstream checkout.

* `test_real_six_tests_through_the_unmodified_servers_with_the_gateway` runs the enforce column against the
  oracle.
* `test_real_wrong_answers_without_the_gateway` runs the off column. If upstream fixes one of these
  answers, the test fails and shows it.

Both passed when first run on the real data (206.9 s and 16.0 s). They passed again in the opt-in run after
the review fixes of section 4, with one exception. That run put the whole pytest tree under
`prlimit --as=6000000000`. The gateway-off servers inherited the limit, and the target server answered
`{"error": ""}` instead of "Target PCSK9 not found". Without the limit, the off test passed (14.2 s, tree
4,763 MB).

**Not reproducible on 25.09.**

* CT-5's negated-evidence case: `disease_phenotype` has no negated evidence items.
* `query_evidence`, `get_evidence_by_publication` and the colocalisation tools: their tables were not
  downloaded (section 6), so these tools were seen only refusing with `not_ready`.

**Re-run for this page.** On 2026-10-08, at commit `d4aadfe`, both tests were run once more with
`VBT_DL_REAL_DATA=data/real` and no address-space limit. Both passed in 182.5 s: the enforce test in
170.9 s, which includes the session check, and the off test in 11.3 s. The process tree peaked at 5,438 MB.
After the round-3 fixes (commit `3f2139c`, 2026-10-09) the whole module, with `VBT_DL_REAL_DATA_STRICT=1`,
gave 26 passed in 213.9 s (tree peak 5,356 MB); the six tests through the unmodified servers took 160.0 s
of it.

The same six tests now also run as `vbt validate`'s `correctness` step on any host, with cases planned from
the bindings and an oracle that reads the files with pyarrow alone; its runs here are in §9.3.

## 4. Wrong answers of the layer itself, found on real data

The first enforce runs on real data also showed wrong answers coming from the gateway, the data child and
the descriptors. Each one was fixed with an offline regression test, and the call was then repeated on real
data.

### 4.1 Open Targets 25.09

Each OT server ran on its own, next to the data child. The regression tests are in
`tests/datalayer/test_dl_review_realdata.py`, which names each finding id, unless the row names another
file.

| Call | Before | Now | Finding |
|---|---|---|---|
| `search_targets_by_name("TP53")` | TP63 ranked first: the search matched the labels of the homologues listed in TP53's row; ties within a match class went by Ensembl id | TP53 first, total 71 (was 437); ties broken by `approvedSymbol`, as the overlay declares | `test_dl_real_ot_servers.py`; RV-OT-10 |
| `interaction.get_interaction_evidence` TP53 / PCSK9 | `empty`, 0 rows (TP53 has 31,685 rows) | `too_large` / `scan_budget` ("needs ~759037123 decoded bytes after pruning, over the 500000000-byte scan budget"), naming `mcp__data__find` as the alternative | RV-OT-01 |
| `get_associations_for_target(PCSK9, limit=100, output_path=...)` | the file held 992 rows (a re-call with a raised limit overwrote it) | 100 rows, no re-call, header total 992 | RV-OT-02 |
| `get_chemical_probes(MTOR)` | `num_high_quality` 228 | 9 | RV-OT-03 |
| `get_drug_adverse_events("amifampridine")` | `empty`: the 61 rows are stored under the salt CHEMBL3301611 | `partial`, `family_rows {CHEMBL3301611: 61}`, the salt id named in a note | RV-OT-04 |
| `get_disease_info(EFO_0001444)` | `descendants` trimmed twice, reported as `{returned 50, total 100}`, in storage order | `{returned 100, total 18649}`, ids in key order | RV-OT-05 |
| `query_expression_by_tissue("liver")`, then other expression and essentiality calls | `service_unavailable` after 95.5 s; the data child stayed at its 3,000 MB limit, and every later call failed as "not a readable Parquet file (ArrowMemoryError)" or `not_ready` | `partial`, total 43,804 in 245.8 s, data child 1,040 MB. An allocation failure is now out of memory, never a bad file, and ends the child with a memory exit. | RV-OT-06 |
| `find_essential_genes("Hepatocellular Carcinoma", limit=5)` | `service_unavailable` after 52-54 s (a bare MemoryError) | `partial`, total 3,393 in 305.0 s (data child 2,042 MB); `query_gene_essentiality(TP53)` afterwards is `ok` | RV-OT-06 |
| `get_pharmacogenomics(ENSG00000001626)` | grains `drug {returned 1, total 10}` (lists counted, null included) | `{returned 8, total 8}` | RV-OT-07 |
| `prioritize_targets(min_genetic_constraint=-0.5, limit=10)` | the lowest Ensembl ids | most constrained first: ENSG00000169180 (-1.0), ENSG00000123066, ...; total 4,496 | RV-OT-08 |
| `search_drugs("statin")`, `search_pathways("cholesterol")` | `invalid_argument` (the text matches several values) | `partial`, total 58 (SIMVASTATIN and PRAVASTATIN first) and 9 | RV-OT-09 |
| `get_target_info("ENSG00000182484_PAR_Y")` | the X-chromosome WASH6P | `ambiguous`: ENSG00000182484 (X) or ENSG00000292372 (Y) | RV-OT-11 |
| interaction tables, after declaring the versioned endpoints as stored forms | `not_ready`: the other species' versioned ids on the same columns became key violations | `ready`; unmatched stored values are a warning | ACC-1 |

### 4.2 Live sources

These runs used MCPBridge with the gateway enforcing: the unmodified `clinicaltrials` and `single_cell`
servers, the harness `pubmed` server and the data child. The regression tests are in
`tests/datalayer/test_dl_live_review_fixes.py`, `test_dl_review_realdata.py`, `test_dl_live_sources.py`
and `test_dl_hardening.py`.

| Call | Before | Now |
|---|---|---|
| `count_clinical_trials(NSCLC, RECRUITING, phase 2/3, interventional, country=null)` | 300 with the note "no restriction": the null was dropped, so upstream applied its default `United States` | 705, as the registry counts worldwide |
| `search_clinical_trials(..., country="Iceland")` | `empty`: the overlay did not map upstream's flattened `locations` | 3 of 3 |
| counts and searches with a condition or free text | no witness | the remote witness counts the same search: glioblastoma recruiting phase 3 15/15, NSCLC 65/65, PCSK9 + evolocumab 115/115, glioblastoma phase 2+3 1,070/1,070, PubMed "PCSK9 AND evolocumab" 1,212/1,212 |
| `get_clinical_trial_details(NCT04368728)` under a 2017-12-31 ceiling | header `empty`, `withheld {leakage: 1}`, but the text (3,077,746 characters) held the whole post-ceiling record | 612 characters, no field of the record |
| `mcp__data__find` of RECRUITING trials under the same ceiling | trials first posted in 2025; total 64,639 | every row first posted and last updated by the ceiling; total 87 |
| `search_pubmed("PCSK9 AND evolocumab")` with `VBT_LITERATURE_MAXDATE` 2017/12/31 | `tool_defect` (upstream 304, witness 1,212 without the date bound) | `partial`, total 304, witnessed |
| `mcp__data__find` with enrollment `{gt: 100, lt: 101}` | 18,953 (sent as inclusive RANGEs) | 0 |
| `mcp__data__find` with `briefTitle == "Cancer"` | 65,307 word-search matches presented as equality | `invalid_argument`; `{search: "Cancer"}` gives the 65,307 engine matches and says so |
| `mcp__data__find` with `columns` naming a nested field | two `{}` rows | rows with the nctId |
| `nctId in [NCT04368728, NCT99999999]` | `ok`, nothing about NCT99999999 | `partial`, `not_found_items ["NCT99999999"]` |
| a native live lookup's provenance | no source, release, table, key or record versions | `clinicaltrials_gov@2026-10-08T09:00:05`, table `studies`, key, row key, record version 2026-03-25 |
| cBioPortal finds | source release = the fetch time | the study's `importDate` (`2026-06-05 15:19:54`), also recorded as its version |
| `mcp__data__find`/`lookup` on `cellxgene_census.obs` | `oom` (`std::bad_alloc`) under the 3,000 MB data-child limit | answered in 9.4-13.5 s |
| the same find (B cells, adrenal gland, one dataset) | 3 of 158 rows, all `is_primary_data: false` | 0, with "is_primary_data == true added"; 158 when both values are asked for |
| `get_census_info`, `count_cells` | release null; no witness | `cellxgene_census@2025-11-08`; `count_cells(tissue_general == 'adrenal gland')` 555,767, witnessed by a remote count of 555,767 |

### 4.3 Live sources, third review (round 3)

A third review drove the live APIs through MCPBridge with the gateway enforcing (2026-10-09, 14:42-14:51 UTC,
commit `cab2979`; the unmodified `clinicaltrials` and `single_cell` servers, the harness `pubmed` server and
the data child). Each finding was fixed with an offline regression test
(`tests/datalayer/test_dl_review_round3.py`, `test_dl_real_live.py`, `test_dl_round3_live.py`), and the same
calls were repeated with the same driver on 2026-10-09 between 21:25 and 21:31 UTC at `3f2139c` ("Now"; the
evidence ceiling, where named, is `data.leakage.ceiling` 2017-12-31 and the literature ceiling 2017/12/31).

| Call | Before | Now | Finding |
|---|---|---|---|
| `count_clinical_trials(glioblastoma, TERMINATED, PHASE3, country=null)` under the ceiling | `ok`, total 8, leakage risk false; 3 of the 8 were terminated after the ceiling | `partial`, total 4 (`total_method: ceiling_unchanged`), `ceiling_totals {available: 8, available_and_unchanged: 4}`, and a note that the source holds `overallStatus` only as it is today | LIVE3-01 |
| `count_clinical_trials(glioblastoma, advanced_filter AREA[ResultsFirstPostDate]RANGE[2018-01-01,MAX])` under the ceiling | `ok`, total 194 (results posted after the ceiling), risk false | `partial`, total 0, `available` 194 | LIVE3-01 |
| `get_clinical_trial_details("NCT00062153")`, an alias of NCT00060528 | `empty`, body `{}`: the redirect was accepted, then the record removed | `ok`, 1 record; header `resolved: NCT00062153 -> NCT00060528 (alias_redirect)` and a note | LIVE3-02 |
| native `lookup` of `NCT00062153` and of `nct00761280`; `find` of `[NCT00062153, NCT04368728]` | `not_found`, `not_found`, `partial` naming NCT00062153 | `ok`, `ok`, `ok` total 2 (also under the ceiling) | LIVE3-03 |
| `search_genes(ensembl_ids=[ENSG00000103855, ENSG00000999999])`; `get_gene_statistics([ENSG00000999999])` | `ok` with both "resolved"; `empty` | `partial`, `not_found_items [ENSG00000999999]`; `not_found` | LIVE3-04 |
| under the ceiling: `get_study_details(luad_tcga_pan_can_atlas_2018)`, `search_studies(difg)`, Census `count_cells` | served with provenance `leakage: null` | the same answers, with leakage `{risk: true, reason: "source not dated"}` and a header note | LIVE3-05 |
| `search_studies("gbmx")`, `search_studies("GBM")` | `empty_unverified`: existence "could not be decided remotely" | `not_found`, decided against cBioPortal's cancer-type listing (`GBM` normalizes to `gbm`; `/api/cancer-types/gbm` answers 404 "Cancer type not found: gbm") | LIVE3-06 |
| native `find` with the misspelled path `protocolSection.statusModule.overalStatus` | `empty_unverified` after reading 10 pages, 11.15 s | `invalid_argument` before any request, 0.02 s | LIVE3-07 |
| `fetch_abstracts(["20301295", "28304224"])` (20301295 is a GeneReviews `PubmedBookArticle`) | `partial`, 20301295 "returned no record" | `ok`, 2 of 2 | LIVE3-08 |
| `fetch_abstracts(["30403574"])` under the literature ceiling | `empty_unverified`, no `withheld`, leakage `null` | `empty`, `withheld {leakage: 1}`, leakage `{withheld: 1}` | LIVE3-09 |
| native `lookup` of PMID 99999999 and 28304224; `find` with `title {search: "PCSK9 AND evolocumab"}` under the ceiling | `too_large` ("narrow where") for both lookups; `invalid_argument` | `not_found` (0.59 s), `ok` (0.71 s); `partial`, total 32 | LIVE3-10 |
| `search_studies(difg)` (20 studies); a `molecular_data` find of an unknown gene | no record versions; the release was the fetch time | each study's `importDate` in the provenance's `record_versions`; `cbioportal@2026-06-05 15:19:54`, the profile's study | LIVE3-11 |
| native `find` (conditions search glioblastoma, TERMINATED, PHASE3) under the ceiling | total 2, with a note giving the wrong reason why the upstream count (4) was larger | total 4: the same four records, two of which match only through the engine's keyword search | LIVE3-12 |

The title search of the last rows exposed a fact about E-utilities, checked by `esearch` with `maxdate` 2017/12/31:
a field tag after a parenthesised group is ignored. `(PCSK9 AND evolocumab)[ti]` counted 304 records, the same as
the untagged search, while `PCSK9[ti] AND evolocumab[ti]` counted 32 (`[tiab]`: 224). The PubMed descriptor now
tags every term, and the replay was re-recorded with `record_live.py`.

Timings and memory of the repeat: the five runs without the Census took 12.6-24.6 s each (process trees 236-709 MB);
the Census runs 50.2 s (686 MB) and 268.2 s (2,962 MB; `single_cell` 2,564 MB, data child 1,336 MB). In the
second, `count_cells` on the adrenal B cells (679, witnessed) took 236.0 s; repeated alone it took 24.0 s, and the
review had measured 21.8 s.

## 5. Memory and latency

### 5.1 Whole-table loads and the memory estimate

The upstream servers load whole tables with pandas. Each 25.09 table was loaded that way in a child under
the reaper. "Measured" is the peak RSS minus the interpreter baseline (about 108 MB).

Five tables did not fit: the child was killed at 5,900 MB, with peaks of 5,399-5,524 MB. They are
interaction_evidence, expression, target_essentiality, interaction and l2g_prediction.

The table compares three estimates with the measured load (ratio = estimate / measured):

* **Seed factors**: the factors that shipped before these runs.
* **Fitted factors**: the factors now shipped in `configs/default.yaml` (`expansion {flat: 1.0, string:
  4.0, nested: 2.1, fragmentation: 1.15}`, `object_overhead_bytes {flat_value: 20, string: 16,
  nested_item: 8, struct_item: 170}`, `estimate_safety: 1.3`).
* **Sample tier**: `vbt ds calibrate`'s measurement of a sample, now counting the pandas frame and the Arrow
  table the loader holds together.

| Table | Measured MB | Seed factors | Fitted factors | Sample tier |
|---|---:|---:|---:|---:|
| target | 3,031 | 4.06 | 1.00 | 1.02 |
| study | 2,556 | 3.81 | 1.46 | 1.14 |
| association_by_datasource_indirect | 1,780 | 1.79 | 0.99 | 1.29 |
| association_by_datatype_indirect | 1,470 | 1.64 | 1.02 | 1.32 |
| association_by_overall_indirect | 983 | 1.48 | 1.11 | 1.40 |
| known_drug | 606 | 3.50 | 0.78 | 1.12 |
| association_by_datasource_direct | 567 | 1.80 | 1.01 | 1.29 |
| association_by_datatype_direct | 482 | 1.63 | 1.03 | 1.29 |
| association_overall_direct | 369 | 1.44 | 1.10 | 1.33 |
| mouse_phenotype | 323 | 3.08 | 0.77 | 1.11 |
| disease_phenotype | 285 | 3.40 | 0.71 | 0.99 |
| disease | 199 | 2.41 | 0.78 | 0.95 |
| literature_vector | 119 | 9.39 | 1.32 | 0.93 |
| biosample | 97 | 3.07 | 0.76 | 0.93 |
| pharmacogenomics | 97 | 3.13 | 1.27 | 0.95 |
| drug_indication | 75 | 3.14 | 0.70 | 0.99 |
| drug_molecule | 53 | 1.86 | 0.74 | 0.81 |
| target_prioritisation | 51 | 0.22 | 0.72 | 0.54 |

* **Seed factors.** They overestimated 17 of the 18 loads by 1.44-9.39 times; target_prioritisation was
  underestimated (0.22). Target was estimated at 12.3 GB for a 3.0 GB load, so every target tool was
  refused `too_large` under a 5 GB server limit.
* **Fitted factors.** On the 18 tables above 50 MB they give 0.70-1.46 (median 1.00). 16 of the 18 are
  within 30% of the measured load (drug_indication, at 0.70, is on the edge). The other two, study (1.46)
  and literature_vector (1.32), are overestimates.
* **Sample tier.** Counting pandas bytes alone gave 0.34-0.85 (median 0.70), which underestimates every
  table. With the Arrow table counted too, it gives 0.54-1.40 (median 1.06).
* **Small tables.** Below 50 MB the server baseline (300 MB in admission) dominates.

**Admission.**

* In a session, study was refused (`over_limit`) on the factors alone and admitted once a calibration
  existed (sample-tier estimate 2,139.5 MB before the safety factor). The real load was 2,590.9 MB, and the
  feedback factor 1.211 was recorded.
* l2g_prediction, interaction (through `get_interactions`) and expression (through
  `query_expression_by_gene`) are refused `too_large` before any load. This is correct: all three exceed
  5.4 GB when loaded.

**Reaper on real loads.** These were whole-table pandas loads of 25.09 `target` and `known_drug` under each
containment:

| Table | Containment | Limit MB | Exit | Peak RSS MB | Last stderr line | Reaper cause |
|---|---|---:|---|---:|---|---|
| target | rlimit_data | 400 | SIGABRT | 148.6 | `what(): Resource temporarily unavailable` (Arrow could not start a thread) | `memory_error` (was `signal`) |
| target | watchdog | 1,200 | SIGKILL | 729.2 | | `watchdog` |
| target | cgroup v1 | 600 | 1 | 357.6 | `pyarrow.lib.ArrowMemoryError: malloc of size 7653632 failed` | `memory_error` (was `exit_code`) |
| known_drug | rlimit_data | 150 | 1 | 39.4 | `OpenBLAS error: Memory allocation still failed after 10 retries` | `memory_error` (was `exit_code`) |
| target | rlimit_data | 6,000 | 0 | 3,478.0 | 78,726 rows in 10.4 s | |
| known_drug | rlimit_data | 3,000 | 0 | 909.5 | 253,442 rows in 1.2 s | |

"Was" is what the reaper at the base commit `59900de` reported. It labelled three of these memory failures
as ordinary exits and gave none of them a cause.

cgroup v1 is usable here as root (`--containment cgroup` selects it), and `/dev/kmsg` is readable. The host's
`/proc/vmstat oom_kill` counter also moves for kills outside the child (it went from 0 to 7 while the probes
caused 3). So a SIGKILL is attributed to the kernel OOM killer only from a kernel log record naming the
child's pid.

**Round 3: every whole-table read is sized.** In the first full `vbt validate` run (§9.3),
`target.get_chemical_probes` and `target.get_genetic_constraint` were admitted at a 3,000 MB server limit, and
the unmodified server was killed loading `target` (maxrss 3,137,564 kB). Their overlays had declared the reads
`projection`, which admission did not size, while upstream's loader reads every column. The 21 `projection`
reads of the target, pathway, disease and drug overlays, and functional_genomics' five `target_essentiality`
reads (declared bounded scans; its upstream calls were killed at 4,400 MB), are now `full_table`, so they are
sized before the call. `vbt ds estimate --tool` on the real 25.09 files now uses admission's rule: MiB, the
estimate x1.3 plus the server's 300 MB idle baseline. Before the fix it printed decimal MB without either and
called loads admissible that the gateway refused. Re-run on 2026-10-09 at `3f2139c` with the default profile
(server limit `auto`: 6,569 MB on this host; 4.6-17.6 s and 376-956 MB per command):

| Tool | Table loaded whole | Upstream peak (estimate) | Needed | Here |
|---|---|---:|---:|---|
| `target.get_chemical_probes` | target (measured load: 3,031 MB) | 3,023 MB | 4,230 MB | admissible |
| `genetics.get_study_metadata` | study | 3,740 MB | 5,162 MB | admissible |
| `expression.query_expression_by_gene` | expression | 5,408 MB | 7,331 MB | too_large |
| `functional_genomics.query_gene_essentiality` | target_essentiality | 10,765 MB | 14,294 MB | too_large |
| `genetics.query_l2g_predictions` | l2g_prediction | 11,564 MB | 15,334 MB | too_large |

A refusal names the plan the limit came from (`host_mb`), whether the limit was `auto` or configured
(`limit_source`) and, for `auto`, the smallest host whose limit admits the load with its baseline
(`host_mb_needed`). Under a profile that configures no MCP server (`mock`) the command has no server limit to
compare with and still printed "admissible" (14,294 MB needed on this 13,680 MB host); run it under the profile
that serves the tool.

**Sample-tier calibration, measured again.** `vbt validate`'s memory step loaded the four largest tables that
fit a 3,000 MB server limit with the upstream loader, in a contained process (ratio = estimate / measured):

| Table | Rows | Measured MB | Shipped estimate | Sample tier |
|---|---:|---:|---|---|
| association_by_datasource_indirect | 13,242,757 | 1,898 | 1,757 (0.93) | 2,302 (1.21) |
| association_by_datatype_indirect | 12,850,289 | 1,560 | 1,505 (0.96) | 1,937 (1.24) |
| association_by_overall_indirect | 10,989,518 | 1,103 | 1,088 (0.99) | 1,376 (1.25) |
| association_by_datasource_direct | 4,200,235 | 669 | 572 (0.86) | 732 (1.09) |

The sample tier over-estimates by 9-25%, so admission errs on the safe side; the shipped estimate
under-estimates by 1-14%, which the x1.3 safety covers.

**Containment.** `data.memory.limit_kind` is `rss` by default since round 3: a memory cgroup when one can be
created, else the RSS watchdog, and `RLIMIT_DATA` only when a server or the host asks for `rlimit_data`. Live
run through MCPBridge with the gateway enforcing, the unmodified `single_cell` server, Census 2025-11-08,
dataset `f7c1c579-2dc0-47e2-ba19-8165c5a0e353`, server limit 6,569 MB:

| `limit_kind` | Containment | `count_cells` | `get_anndata_donor_balanced` | Peak MB |
|---|---|---|---|---:|
| default (`rss`) | cgroup v1, no `RLIMIT_DATA` | 4,062,980 cells | 293 cells, 72 cell types, `served_by: derived` | 3,192 |
| `watchdog` (forced) | watchdog, kill at 6,057 MB | 4,062,980 | the same | 3,193 |
| `rlimit_data` (forced) | `RLIMIT_DATA` 6,569 MB | answered | typed `oom` (`std::bad_alloc`) | |

Which call fails under `rlimit_data` varies from run to run: in another run the count itself failed. Agents'
Bash commands, notebooks and project utility tests still ran under `RLIMIT_DATA` after that change: a review
ran `cellxgene_census.get_anndata` of 104 cells x 6 genes the way the Bash tool runs it, and it crashed
(`std::bad_alloc`, then SIGSEGV at 256 MB maxrss, three times out of three), while the same script under `rss`
with the same 6,840 MB limit answered in 15.1 s at 1,946 MB maxrss. Those commands now use the shipped `rss`
containment too.

The Census pulls and their calibrated admission are in §8.1.

### 5.2 Readiness checks

Each table was checked in its own `vbt ds check --table open_targets.<t> --json` process.

Standard depth (what a session's first call waits for), before and after the real-data fixes:

| Table | Before | After |
|---|---|---|
| target | 184.2 s, 955 MB | 50.9 s, 695 MB |
| expression | 330.1 s, 2,414 MB | 27.2 s, 1,547 MB |
| association_overall_direct | 55.7 s | 6.8 s |
| interaction_evidence | more than 15 min | 96.5 s, 796 MB; 45.5 s, 759 MB with `row_identity: none` |

Deep checks, all `ready`. The column "Code" says whether a row was measured before or after most of the
speed-ups of the real-data fixes:

| Tables | Seconds | Peak MB | Code |
|---|---:|---:|---|
| so, reactome, drug_warning, drug_mechanism_of_action, both openfda tables, go, disease_hpo, drug_molecule, biosample, literature_vector | 4.0-9.3 each | 206-393 | before |
| target_prioritisation | 50.9; 2.1-2.3 once R9 matched references with Arrow `is_in` | 267 | before; after |
| drug_indication, disease, pharmacogenomics, disease_phenotype | 9.8-16.6 | 339-551 | before |
| mouse_phenotype, known_drug | 28.0, 31.3 | 702, 1,000 | before |
| the three direct association tables | 86.5-106.9 | 638-704 | before |
| study | 49.7 | 1,436 | before |
| target | 209.5 | 1,577 | before |
| the three indirect association tables | 11.6-15.2 | 1,678-1,918 | after |
| expression | 135.0 | 1,684 | after |
| interaction | 35.3 (477.1 at 3,939 MB before) | 2,299 | after |

* **Item keys.** The deep check of l2g_prediction was stopped at 900 s here: the full key check of its
  features item table rendered every item in Python. Item keys are now counted in Arrow (section 8.3): the
  deep checks of l2g_prediction (44,578,431 feature items) and target_essentiality finish in 23.7-26.5 s and
  12.1-12.6 s.
* **10.3 GiB.** An early deep check of interaction_evidence (971.3 s) peaked at 10.3 GiB, because
  `vbt ds check` started the data child without the reaper. It now runs under the reaper
  (`preflight.run_contained`). A composite reference also keeps only the sampled tuples now; it kept all
  14.5 M.
* R10's edge check now looks up the reverse of each sampled edge in the whole table: 3.7 s on interaction
  and 5.4 s on interaction_evidence, with no reverse missing. Before, it reported 18,827 false
  "without their reverse" warnings.

**The session check, round 3.** This is the check a session's preflight runs on every table at standard depth
(95 tables: Open Targets, DepMap, the ontologies, the Zenodo archive, Tahoe's prepared sample and the live
tables), each in a data child under the reaper:

| Code | Data-child processes | Wall | Largest data child | Statuses |
|---|---:|---:|---:|---|
| `d6c444b` (before) | 1 | 661.0 s | 2,705 MB | 85 ready, 10 missing |
| large tables (32 MB or more on disk) in data children of their own | 14 | 289.6 s | 2,293 MB (interaction's key pass) | the same |
| plus a bounded R5b key pass | 14 | 294.9 s | 1,078 MB (study) | the same |
| `3f2139c`, re-run for this page with `vbt validate --only host,lint,check --depth standard` | 14 | 253.7 s | 1,078 MB (study); process tree 1,177 MB | 79 ready, 16 absent (the 10, and the six Zenodo tables, because `VBT_ZENODO_DIR` named the wrong directory; with the acquisition home they checked `ready` in 10.6 s, 385 MB) |

The 10 tables that are absent are the 7 Open Targets tables not downloaded (with `credible_set_locus` and
`evidence_mutated_samples`, their item tables) and `cellxgene_census.anndata_outputs` (files a run writes).

* **Key pass.** The R5b key pass held every key part's values: on 25.09 `interaction` (14,524,000 rows, a
  7-part key) it took 24.7 s and 2,101 MB. It now hashes each row to 64 bits and counts exactly only the rows
  whose hash repeats, in passes bounded by `data.service.max_resident_mb`: 12.5 s, 582 MB, 0 duplicates (a
  review measured 9.1-9.8 s and 591-601 MB). The table's standard check in a data child: 24.6 s, 675 MB.
* **Wide CSV.** DepMap's `gene_effect` (1,178 rows x 17,917 columns) spent 182.3 s of a session check in its
  R3 statistics, read in 8 MiB blocks. In a benchmark on the same file, blocks of 8, 16, 32 and 64 MiB took
  23.2, 13.5, 7.7 and 5.9 s (505-735 MB); 32 MiB is shipped.
* **Matrix tables.** R6, R7, R9 and R10 now run on each axis's declared columns. On DepMap 24Q4 `gene_effect`
  and `gene_dependency`, R9 resolves all 1,178 sampled `ModelID`s in `model`; on the Zenodo `ibd_cohorts`, R7
  counts the values of 11 obs columns (patient_id 222, timepoint 17, cohort 4, drug 4, disease 3,
  response_remission 0, sex 0). The three tables checked `ready` in 49.0 s, largest child 815 MB.
* **Deep.** The last deep session check (before the bounded key pass) took 319.2 s, 14 processes, largest
  2,205 MB; it was not repeated after it.

### 5.3 Resolver indexes

* **Build.** `vbt ds index build` for every id type with a universe took 263.1 s and peaked at 2,104 MB. It
  built 17 indexes; 11 failed only because their sources are not on this machine. Index sizes include
  ensembl_gene 465,164 rows, gwas_study 1,964,234 and ot_disease 335,308.
* **Reuse.** A session used to rebuild the `ensembl_gene` index although the same sidecar was on disk
  (28.2-30.1 s). It now reuses it (3.6-3.8 s).
* **Only what the tools read.** `vbt setup`'s index step built every id type of the 31 tables: 343 s, 20
  indexes. `vbt ds index build --table open_targets.go --table open_targets.target` builds only the three id
  types those tables hold (`ensembl_gene` 465,286 rows, `go_term` 96,330, `uniprot_accession` 264,497): 45.2 s,
  681 MB.
* **Cell Ontology.** On the real `cl-basic.obo` (2026-06-08) the index has 11,016 rows; building it and
  resolving `B cell` (CL:0000236, by `label_exact:name`), `CL:0000236` and `T cell` (CL:0000084) took 5.0 s.

### 5.4 Sessions through MCPBridge

The upstream servers had a 5,000 MB limit and the data child 3,000 MB. "Readiness" is the session check
that a server's first call waits for.

| Servers | Readiness + first call (fresh cache) | Peak MB (servers / data child) | Tree MB |
|---|---|---|---:|
| target | 121.6 s + 32.8 s (before the fixes: 656.8 s in all, 614.9 s of it readiness) | 3,233 / 2,569 | 4,609 |
| drug | 29.0 s in all | 826 / 773 | 1,561 |
| disease, pathway | 78.6 s in all | 374, 169 / 653 | 1,787 |
| genetics, interaction | 178.3 s + 8.5 s (before: 1,014.2 s in all, 1,004.4 s of it readiness) | 2,760, 169 / 2,334 | 3,816 |
| association, expression | 68.4 s in all | 2,192, 169 / 1,882 | 4,424 |

The same calls with the gateway off peaked at 3,582 MB (target), 1,029 MB (drug), 4,214 MB (disease and
pathway), 2,963 MB (genetics) and 2,769 MB (association and expression).

In these round-2 sessions the data child reached 2.3-2.6 GB of its 3 GB limit during the session check;
since round 3 the largest data child of the check peaks at 1,078 MB (§5.2). While serving calls, the data
child peaked at 2,787 MB (interaction) and 2,723 MB (target) in `vbt validate`'s runs (§9.3); its limit is now
`auto` (5% of the plan, 3,000-32,768 MB).

**Live sources.**

* **ClinicalTrials.gov and PubMed.** Calls take 0.9-2.4 s. Peaks: clinicaltrials 173-263 MB, pubmed
  88-96 MB, data child 113-186 MB; the whole tree 553-784 MB.
* **Census.** `get_census_info` 5.0-6.6 s. `count_cells` 9.0-20.4 s. Native obs reads 9.4-14.7 s, with
  the data child at 1,328-1,400 MB. `get_anndata` (4,771 cells x 2 genes) took 38-42 s with the
  `single_cell` server at 2,782 MB; the server peaked at 2,575-2,624 MB in the other Census runs.

**Sessions with a model (round 3).** The end-to-end run (§9.5) gave the harness a 6,000 MB share of this host
(`VBT_HOST_MEMORY_MB=6000`): data child 3,000 MB, each upstream server 2,048 MB, host budget 2,452 MB.

* **Start.** A second readiness check ran at every session start, with the data child at 81% CPU and
  1,454 MB; the preflight's check now reaches the gateway before the bridge lists the tools, and the data
  child stayed at 114 MB through session start.
* **Served model, session 1** (Qwen3.5-2B on llama.cpp, 7 Open Targets servers): 9,339 s, process tree peak
  4,056 MB, 80 tool calls (44 to the upstream servers, 10 native). 22 calls were `too_large`: 11 over a
  server's limit, and 11 `host_busy` because the data child's memory had been charged to the upstream host
  budget (now it is never counted there). The data child's recorded peak was 2,089 MB under its 3,000 MB
  limit.
* **Scripted model over the same stack:** session 1 started in 7.8 s and its turn took 26.7 s (tree peak
  2,380 MB); session 2 (a dataset with no descriptor; §9.5) started in 8.6 s, turn 12.2 s, 1,778 MB.

### 5.5 Witness and gateway latency

Measured on the target server, three repeats of the same calls:

* **Warm calls** (22 witnessed calls): witness p50 58.1 ms, p95 82.3 ms, max 98.2 ms. The whole call took
  p50 93 ms and p95 985 ms.
* **First calls.** Over all three repeats the witness p95 was 4,194 ms and the max 14,544 ms. A first call
  waited for the data child's `_stats` of the target tables (14.5 s) only to learn the storage types. The
  session check now reports those types. The `_stats` requests it avoids took 16.5 s for target and 36.0 s
  for target_go. These three repeats were not run again after this change; the per-server figures of round 3
  are below.
* **Resolution.** p50 0.6 ms, p95 77-80 ms. Unknown ids walk the near-match rules.
* **Single-call witnesses on other tables.** study 2.0-2.2 s; interaction 1.5 s (intact) to 9.2 s (all
  sources). The expression witness for one gene took 128 s; once item tables pushed conjuncts on their parent
  row to Arrow, the same witness took 1.43 s and 338 MB in process (it took 25.07 s and 1,458 MB before).

**Across servers (round 3).** `vbt validate`'s latency step timed 212 calls on 9 servers (2026-10-09, deep run,
server limits 3,000 MB; times in ms). "Off" is the same call without the gateway; "warm overhead" is enforce
minus off on warm pairs:

| Server | Enforce p50 / p95 | Off p50 / p95 | Warm overhead p50 / p95 | Witness p50 / p95 |
|---|---|---|---|---|
| association | 2,487 / 12,985 | 767 / 1,735 | 25.5 / 4,440 | 1,052 / 1,891 |
| disease | 52.9 / 2,716 | 12.4 / 585 | 24.3 / 402 | 15.9 / 18 |
| drug | 170 / 2,482 | 10.4 / 165 | 65.4 / 1,978 | 427 / 1,447 |
| expression | 218 / 6,868 | 9.4 / 191 | 24.2 / 44.7 | 460 / 564 |
| functional_genomics | 405 / 3,109 | 15.3 / 34.1 | -3.9 / 562 | 153 / 322 |
| genetics | 320 / 20,688 | not called | | 518 / 10,752 |
| interaction | 50 / 24,785 | not called | | 7,942 / 8,680 |
| pathway | 68.3 / 2,982 | 5.7 / 7.2 | 23.6 / 29.7 | 10.2 / 28.5 |
| target | 50.8 / 10,626 | (server down) | | 64.1 / 371 |

With the `target` server alone at 4,400 MB, the warm overhead was p50 36 ms and p95 266 ms, and the witness
p50 50 ms and p95 303 ms. So the soak target (a witness p95 under 300 ms on composite-key tables) holds for
the disease and pathway servers only: the association, drug, expression, genetics and interaction witnesses
scan tables of millions of rows, and their p95 is 0.56-10.8 s.

### 5.6 Test suite

The full suite at `d4aadfe` (offline, default settings): 3,305 passed, 148 skipped, 87 xfailed, 0 failed,
in 787.5 s, with a tree peak of 5,093 MB. A run with this page added gave the same counts in 789.1 s
(peak 5,065 MB). The baseline at `59900de` was 3,065 passed, 115 skipped and 87 xfailed. The added skips
are the opt-in real-data and network tests.

Opt-in runs:

* `VBT_DL_REAL_DATA=data/real` (OT schema, OT servers, Tahoe and DepMap, data checks, review real-data):
  119 passed and 4 skipped in 593.6 s. One test failed only under the address-space limit described in
  section 3, and passed without it.
* `VBT_DL_NETWORK=1` (`test_dl_real_live.py`, `test_dl_live_review_fixes.py`): 61 passed.

**Round 3.** The suite grew with the round's tests; at `3f2139c` (2026-10-09) the full offline run gave
3,869 passed, 184 skipped, 87 xfailed, 0 failed in 1,378.6 s (largest child RSS 1,206 MB); the run made with
this page is in §9.7. The offline suite is now hermetic: it reads no `.env`, pins the host memory to 16,384 MB,
drops the data-root and model-server variables an operator may export, and refuses in-process requests to
non-loopback hosts unless `VBT_DL_NETWORK` enables them. The whole `tests/datalayer` directory in one pytest
process went from a 5,018 MB to a 3,866 MB process-tree peak once `off` and `enforce` share one set of upstream
servers and memory is released after each module.

Opt-in runs at `3f2139c`:

* `VBT_DL_REAL_DATA=data/real VBT_DL_REAL_DATA_STRICT=1`: the OT schema, Tahoe/DepMap and review real-data
  modules, 77 passed and 4 skipped (the 4 need the network) in 75.5 s, tree 1,106 MB; repeated for this page,
  77 passed and 4 skipped in 76.3 s, tree 1,049 MB. The OT servers module: 26 passed in 213.9 s, tree
  5,356 MB. `VBT_DL_REAL_DATA_STRICT=1` fails a run in which no real-data test passed, so a wrong directory is
  not a green run of skips; the data root is found through the acquisition-home layout
  (`<root>/<acquisition.dir>`, e.g. `open_targets/25.09`).
* `VBT_DL_NETWORK=1`: the live scenarios of `test_dl_real_live.py` (11) and `test_dl_round3_live.py` (9)
  passed in 19.5 s and 8.2 s; repeated for this page, 20 passed in 24.9 s (tree 98 MB). The Census sample
  tests: 5 passed in 72.2 s, tree 1,503 MB.

### 5.7 Limits that scale with the host

Every memory budget ships as `auto`, computed from the memory the harness may plan with (the smaller of
MemTotal and the memory cgroup limit, or `data.memory.host_mb`, or `$VBT_HOST_MEMORY_MB`; rules in
[DATA_LAYER_RUNBOOK.md](DATA_LAYER_RUNBOOK.md#host-scaled-limits)). `vbt validate --only host` printed these
effective values for simulated plans (2026-10-09, `3f2139c`; MB):

| Plan | Host budget | One upstream server | Data child | Agent command | All at full load (8 agents) |
|---:|---:|---:|---:|---:|---:|
| 13,680 (this host) | 8,212 | 6,569 | 3,000 | 6,840 | 65,932 |
| 16,384 | 10,240 | 8,192 | 3,000 | 8,000 | 77,240 |
| 65,536 | 45,875 | 36,700 | 3,276 | 8,000 | 113,151 |
| 131,072 | 91,750 | 73,400 | 6,553 | 8,000 | 162,303 |
| 524,288 | 367,002 | 293,601 | 26,214 | 13,107 | 498,072 |
| 1,048,576 | 734,003 | 587,202 | 32,768 | 28,672 | 996,147 |

The witness and readiness budgets scale with the data child (at 512 GB: `max_scan_bytes` 17,476,000,000,
`max_key_set` 174,760). Below 512 GB the 8,000 MB floor of an agent command makes the full-load sum exceed the
plan, and the `host` step warns. On a 512 GB host every Open Targets whole-table load estimated in §5.1
(at most 15,334 MB with safety and baseline) fits one server's 293,601 MB; on this host three of them are
refused before the call. No host larger than this one was available: the larger rows are the rule's output,
asserted in tests, not runs.

## 6. What still needs data or model runs

### 6.1 Closed in round 3

| Open after round 2 | What round 3 did | Where |
|---|---|---|
| No checked way to fetch a release; `R2:manifest_absent` on every table | `vbt data acquire` fetches any declared source or table from its publisher, verifies every file and writes the manifest R2 reads; the 7 Open Targets tables not downloaded here and the whole Tahoe release are one command each (29.32 GB and 88.86 GB left to fetch into `data/real`, as planned on 2026-10-09) | §9.1 |
| Matrix tables got no R6, R7, R9 or R10 | they run on each axis's declared columns (DepMap 24Q4 and the Zenodo cohorts) | §5.2 |
| The witness p95 was measured on one server only | measured on 9 servers: under 300 ms on disease and pathway only; 0.56-10.8 s on the servers whose witnesses scan tables of millions of rows | §5.5 |
| The session check held 2.3-2.6 GB of the data child's 3 GB | 14 data children, the largest 1,078 MB | §5.2 |
| Fixed memory limits (12,000 MB per server, a 3,000 MB data child) | every budget `auto`, scaled from the host's plan; `rss` containment by default | §5.7 |
| Reads declared `projection` or bounded that upstream loads whole were admitted and killed | declared `full_table` and sized before the call | §5.1 |
| No agent session had run on real data with a model | one end-to-end run with a 2B model on CPU, on the real Open Targets release, with a project | §9.5 |
| No check of a whole host | `vbt validate`, run here on the real data | §9.3 |
| Dataset-specific helpers would have needed core code | descriptors, acquisition specs and utilities created by the system in a project, on real files and a live release | §9.4 |

### 6.2 Still needs data

* **Open Targets tables not downloaded.** They did not fit this machine's disk (6.3 GB free on 2026-10-09),
  so only their footers were read:
  * evidence (8.99 GB)
  * variant (3.18 GB)
  * credible_set (2.59 GB)
  * colocalisation_coloc (3.92 GB)
  * colocalisation_ecaviar (5.16 GB)

  Only footers and samples were read for literature (4 of 334 shards) and interval (2 of 83). Their value
  facts beyond the footer bounds are unchecked, and no deep check ran on them. `query_evidence`,
  `get_evidence_by_publication`, `query_gwas_associations` and the colocalisation tools were seen only
  refusing with `not_ready`, and they leave `vbt verify --data` of a run on this host `degraded_run`. The
  interval content-identity finding rests on 2 of 83 shards. On a host with the disk,
  `vbt data acquire open_targets` fetches the 2,811 missing files (29.32 GB) and verifies them.
* **Whole-table loads above about 5.9 GB.** expression, interaction, interaction_evidence, l2g_prediction and
  target_essentiality did not fit a 5.9 GB test child, and the colocalisation tables were not downloaded, so
  the largest whole-table load measured is `target` (3,031 MB). The larger ones are admitted or refused on
  estimates (§5.1: up to 15,334 MB needed for l2g_prediction; DEPLOYMENT.md §2.2 for the whole release). On this host they are refused
  `too_large`; on a 512 GB host the `auto` limit admits them, but no such host was available.
* **`vbt validate` end to end after round 3.** Its last full run here (correctness, latency, memory) was
  before the overlays were changed to `full_table` and used 3,000 MB server limits to stay under this
  machine's 6 GB cap: FAIL, with 0 wrong answers but 4 calls killed and 13 unanswered on `target` (§9.3).
  The `target` server alone at 4,400 MB gave 39 cases, 37 correct, 2 refused, 0 wrong. The opt-in
  `test_real_data_validation` needs one server holding `target`, the data child and pytest at once, which
  exceeded the 6 GB cap here.
* **Item tables under a parent of more than 16 M rows.** Item keys are counted in Arrow (section 8.3), except
  under a parent table of more than 16 M rows (evidence `mutatedSamples[]`, 30.4 M evidence rows), which
  keeps the spilled row scan. Evidence is not downloaded here, so that case was not measured.
* **Tahoe-100M.** 13 of 65,218 contrasts were read, and the unmodified preparation ran on one DE shard
  (3,986,181 rows; §9.1). The prepared DE files of the whole release (82.76 GiB of source shards) were not
  built, and `obs_metadata.parquet` (2.29 GB) was not downloaded. The figure of about 1.4e8 permissive rows
  is an estimate from the 13 contrasts.
* **CT-5's negation case.** It needs negated `disease_phenotype` evidence, which 25.09 does not hold.
* **DepMap.** Releases newer than 24Q4 are only on the portal, behind its browser check.
* **Census.** The pull estimate (section 8.1) is fitted to spleen pulls of 1,842-149,759 cells with 2 genes
  and one 7,750-cell pull with every gene. Other filters and gene counts were not measured. A 200,000-cell
  pull does not fit the `single_cell` server's 4,500 MB limit and is refused before the call.
* **Descriptor facts still `verified: false`.** Disease `ontology.leaf` (refuted on 16,017 of 20,000 sampled
  rows; the field is dropped), biosample `ancestors`/`descendants` as exact closures (hierarchy columns have
  no `on_refute`, and R10 checks ancestor closures only), and chemical-probe coverage (77,809 null lists, 0
  empty: "none listed" cannot be told from "not covered").
* **The Case 1 archive through the generic engine.** The acquisition home of the Zenodo record
  (`<root>/zenodo/22259123`) holds the descriptor's six tables, not the Case 1 inputs: `vbt validate`'s
  replication step is skipped there ("the archive lacks a Case 1 input"), and passes on the extract that
  `vbt data zenodo fetch --preset case1` writes.

### 6.3 Still needs model runs

* **The production model.** No GPU was available: Qwen3.8-27B on vLLM has not driven a session. The
  end-to-end run used Qwen3.5-2B on CPU (§9.5). It delegated, used the data tools and was held to review, but
  it filed no claims (its CSO was at 13 of 14 turns when the model server was stopped by this sandbox's
  two-hour limit on background commands), and it could not drive the data engineer's steps, which the scripted
  model drove over the same real stack. `vbt validate`'s `model` step was skipped (no model server).
* **F23 (calibration, soak, graduation).** Validating `profile: fidelity` on the paper scenarios needs model
  runs of those scenarios; none were made. `vbt ds graduate` ran on the one served-model run (the target,
  pathway, drug, association, genetics and interaction servers and the data child all graduated, 6.6 s), which
  is evidence from one research question, not from the scenarios.
* **Throughput.** Bulk annotation (the paper's 37,075 trials) and parallel specialists at full context were
  not run on any model here.

## 7. Re-running the checks

On an owner's host the whole check is one command: `vbt validate` runs lint, the session check, the six
correctness tests (enforce, off, oracle), latency, memory, the live sources, the replication extension and the
model server on that host's data, and writes `validate.md` and `validate.json` (DEPLOYMENT.md §7.5,
DATA_LAYER_RUNBOOK.md). The default test suite is offline; the real-data tests are opt-in:

| Variable | Tests | Needs |
|---|---|---|
| `VBT_DL_REAL_DATA=<dir>` | `test_dl_real_ot_schema.py` (downloaded files against the snapshots), `test_dl_real_ot_servers.py` (standard checks, the six tests, the wrong answers off), `test_dl_real_tahoe_depmap.py`, `test_dl_review_realdata.py`, `test_dl_hardening.py` (whole-table loads under the reaper), `test_dl_round3_items.py` (item keys of the real tables in Arrow, the download manifest), `test_dl_acquisition.py` (the manifests `vbt data acquire` wrote still match the files), `tests/test_utilities.py` (system-drafted descriptors registered on the real files), `tests/test_validate.py` (`vbt validate`'s correctness step) | `<dir>` is the acquisition root that `vbt setup` and `vbt data acquire` fill (`$VBT_HOME/data` or its `sources/`; each source at its `<acquisition.dir>`, e.g. `open_targets/25.09`), or the `open_targets/25.09` directory itself. `VBT_DL_REAL_DATA_STRICT=1` fails the run when no real-data test passed. The server tests also need `third_party/TheVirtualBiotech` checked out (or `VBT_UPSTREAM`). |
| `VBT_DL_NETWORK=1` (also `true`, `yes`, `on`; `full` reads every shard) | `test_dl_real_live.py`, `test_dl_live_review_fixes.py`, `test_dl_round3_live.py` (live CT.gov, cBioPortal, E-utilities, Census; the derived routes and the calibrated Census admission), `test_dl_round3_memory.py` (the Census under each containment), `test_dl_acquisition.py` (every shipped source listed live against the declared sizes, by HEAD where a listing gives none; two small tables acquired), `tests/test_utilities.py` (the live HGNC set and Open Targets 25.06 from system-authored specs), the release integrity list of `test_dl_round3_items.py`, the live footer reads of `test_dl_real_ot_schema.py` and `test_dl_real_tahoe_depmap.py` | network access; `cellxgene-census` for the Census tests. Any other value, `0` included, keeps them off |
| `VBT_DL_HOST_LIMITS=1` | the cgroup kill tests of `test_dl_hardening.py` | a writable cgroup v1 hierarchy |
| `VBT_E2E_MODEL_URL=<url>` | the served-model session of `tests/test_e2e_stack.py` | an OpenAI-compatible model server (docs/E2E_RUN.md) |

For example:

```bash
VBT_DL_REAL_DATA=$VBT_HOME/data VBT_DL_REAL_DATA_STRICT=1 python -m pytest -q tests/datalayer/test_dl_real_ot_servers.py
VBT_DL_NETWORK=1 python -m pytest -q tests/datalayer/test_dl_real_live.py
```

The server tests start the unmodified upstream servers and the data child; their process tree peaked at
5.4-5.6 GB here. Do not run them under an address-space limit (`prlimit --as`): the gateway-off servers
inherit it and fail. Data is fetched with `vbt data acquire <source>[.<table>]` (for Open Targets also its alias
`vbt data ot fetch <table...>`): each file's checksum is checked against the publisher's list and the
download manifest is written; `vbt data ot manifest` writes the manifest for Open Targets tables already on
disk (section 8.4). Then `vbt ds check --table <source>.<table>` checks what you fetched.

**Updating the snapshots.** `VBT_UPDATE_REAL_SNAPSHOT=1` rewrites the Open Targets snapshots from the live
release. `tests/datalayer/real/record_live.py` re-records the live exchanges the offline suite replays
(`tests/datalayer/real/live/README.md`).

## 8. Round 3: derived live routes, item keys in Arrow, Census admission

Run on 2026-10-08 on the same machine, releases and APIs. The upstream servers were launched unchanged
(MCPBridge, enforce mode) or, for the Census calibration, their functions were called unchanged in a fresh
process.

### 8.1 Census (2025-11-08)

**Derived donor-balanced sample.** `get_anndata_donor_balanced` is served as the derived `(dataset_id,
donor_id)` sample: the data child replays upstream's cell-type-stratified draw (`RandomState(42)`, the same
order) with donors keyed by dataset, and the unmodified server is asked for exactly the drawn cells.

| Call | Result |
|---|---|
| one dataset (`f7c1c579-...`, 4,062,980 matching cells), `max_cells` 300 | the same 293 `soma_joinid`s upstream's own draw picked; 28 donors; the file holds them (`sample_cells`) |
| two datasets (1,611,261 cells), `max_cells` 2,000 | 1,982 cells over 36 `(dataset_id, donor_id)` donors; the file holds them |
| spleen (`tissue_general == 'spleen'`, 577,677 primary cells, 7 datasets) | six `donor_id` labels occur in two datasets each: 58 labels, 64 donors. Upstream's own balancing merges them |
| spleen, `max_cells` 200,000 | 199,744 drawn over all 64 donors and 110 cell types; the `value_filter` is 2,104,665 characters; the soma layout counted it as 199,744 in 4.72 s at 1,345 MB, and `count_cells` through the unmodified server answered 199,744 (22.2 s) |
| a filter selecting no cell | upstream's "No cells found" was `not_found` on `ensembl_ids`; with a count-first count of 0 it is `empty` |

**Calibrated admission.** Before, the estimate was `cells x (200 + genes x 4)` bytes with only
`gene_symbols` counted: the 200,000-cell spleen pull (Ensembl IDs only) was estimated at 40,000,000 bytes, admitted,
and killed at the server's 4,500 MB cgroup limit (maxrss 4,701,844 kB, 79.5 s). Peak RSS (VmHWM) of pulls
made by calling the unmodified upstream function in a fresh process:

| Pull | Cells | Genes | Seconds | Peak MB |
|---|---:|---:|---:|---:|
| `get_anndata`, one dataset (`59632ec0-...`) | 7,750 | 2 | 13.9 | 1,805 |
| `get_anndata`, the same dataset | 7,750 | all (61,497) | 17.9 | 2,878 |
| `get_anndata_donor_balanced`, drawn spleen cells (`soma_joinid in [...]`) | 1,842 | 2 | 41.5 | 3,821 |
| the same | 19,782 | 2 | 90.2 | 3,199 |
| the same | 49,783 | 2 | 144.2 | 3,190 |
| the same | 99,771 | 2 | 84.1 | 3,136 |
| the same | 149,759 | 2 | 67.9 | 4,489 |

The footprint is mostly fixed, so `count_first.estimate` is a base plus a per-cell slope fitted to the
largest pulls: `get_anndata_donor_balanced` 3,900 MB plus 4,500 bytes per cell (plus 100 bytes per matching
cell when the server draws its own sample), `get_anndata` and `get_expression_for_genes` 2,000 MB plus 4,500
bytes per cell, and 145,000 bytes per cell more when no gene is named. Through the gateway with the data
child on the real Census and the server at 4,500 MB, spleen pulls of 20,000 and 100,000 cells were admitted
(estimates 3,987 and 4,339 MB; 19,782 and 99,771 cells drawn) and pulls of 150,000 and 200,000 cells were
refused `too_large` before the call (4,559 and 4,778 MB), each decision in 10.4-11.4 s at a 1,067 MB
process tree.

**Body release.** `get_census_info` answered `census_version: "stable"`; the body now names 2025-11-08, as
the header does.

### 8.2 Live sources

**`get_clinical_data` served derived** from the live cBioPortal tables (the samples, their sample
attributes, their patients' attributes, the study's attribute ids), compared with upstream through the
unmodified server:

| Call | Upstream | Derived |
|---|---|---|
| `acc_tcga_pan_can_atlas_2018`, 5 samples | 5 rows, 4.98 s | equal, 5.34 s |
| the same study, all 92 samples | 92 rows, 3.63 s | equal, 3.98 s |
| `lgg_ucsf_2014` (61 samples of 23 patients) | 61 rows, 4.18 s | equal, 4.94 s |
| a phantom `TCGA-OR-A5ZZ-01` with one real sample | `not_found`, 3.4 s | `not_found` before any attribute is read, 1.09 s |
| unknown study `no_such_study_xyz` | `not_found`, 1.5 s | `not_found` naming `study_id` (HTTP 404), 2.0 s |

"Equal" covers the rows, the `patients` section and the attribute list. Through MCPBridge the data child's
answer to an unknown study had come back `service_unavailable`: a reply with a top-level `error` key is a
failed call to the bridge. The refusal now goes on the wire as `refusal`.

**Release provenance of upstream-served calls.**

| Call | Release | Versions |
|---|---|---|
| CT.gov calls, including the unwitnessed `eligibility_text` count (before: no release) | `dataTimestamp` 2026-10-08T09:00:05 | `apiVersion` 2.0.5 |
| `get_study_details acc_tcga_pan_can_atlas_2018` (before: no release) | the study's `importDate` 2026-06-05 15:19:54 | `portalVersion` v7.1.2, `dbVersion` 3.0.0 |
| `search_pubmed` | none (PubMed names no data release) | `dbbuild` Build-2026.10.08.12.28 at 20:41 UTC, Build-2026.10.08.12.38 at 20:49 UTC |

**Evidence ceiling 2017-12-31.** RECRUITING trials: 2,323 first posted by the ceiling, 87 also last updated
by it. Glioblastoma, completed or terminated, phase 3, Germany: 21 and 13 (8 withheld). Both totals are in
`_vbt.ceiling_totals`. Native finds on cBioPortal (33 rows) and the Census (158 rows) cut to `limit` 3 now
say every matching row was read.

### 8.3 Item keys in Arrow (Open Targets 25.09)

Every keyed item table and keyed container of the 31 downloaded tables is counted in Arrow; none falls
back to the row scan. Every composed key is unique over all items: l2g_features 44,578,431,
target_essentiality_screens 20,955,265 (null `tissueId` grouped), expression_tissues 4,940,421,
target_homologues 3,820,596, target_go 821,377 (nullable `ecoId`), drug_indications 61,629,
target_chemical_probes 5,090 (nullable `drugId`), target_safety_liabilities 4,484 (nullable `eventId`,
`url`).

| `vbt ds check` of | Depth | Before | After |
|---|---|---|---|
| target_essentiality | deep | 542.3 s, 1,951 MB | 12.1-12.6 s, 694-729 MB |
| l2g_prediction | deep | 791.6 s, 1,250 MB | 23.7-26.5 s, 932-994 MB |
| target | deep | 147.3 s, 1,405 MB | 15.6-16.2 s, 1,376-1,394 MB |
| interaction_evidence | deep | 36.3 s, 648 MB | 38.3-44.8 s, 650-656 MB (no item tables) |
| target_essentiality | standard | 35.0 s, 2,077 MB | 7.5 s, 504 MB |
| target | standard | 38.0 s, 628 MB | 11.8 s, 452 MB |
| l2g_prediction | standard | 49.8 s, 920 MB | 17.4 s, 670 MB |
| expression | standard | 21.9 s, 1,051 MB | 7.0 s, 993 MB |

All 31 tables are `ready` at both depths (deep: 299.6 s summed, largest tree 2,217 MB). Measured on the
real data along the way:

* 25.09 key parts are nested values: chemical probe `origin` is `list<string>`, `urls` is
  `list<struct<niceName,url>>`, drug indication `references` is `list<struct<source, ids[]>>`, and
  `maxPhaseForIndication` holds the in-band code -1. Lists compare as multisets, structs field by field,
  codes as null.
* pyarrow's `list_parent_indices` does not skip a null list slot that still spans values (parents
  `[0,0,1,2,2]` against 4 flattened values); parents come from the list value lengths.
* R6 counted struct containers as lists, so 25.09 `tep` and `hallmarks` read null in all 78,726 genes. They
  now count 41 and 368 present.
* Chemical probes: 77,809 genes null, 0 empty, 917 with probes. "None listed" is never an empty list, so the
  coverage fact stays `verified: false`.

### 8.4 Open Targets release download

`release_data_integrity` of 25.09 lists 22,557 files (sha1 `872decc7ef350306d0843ba6279019662b004a42`, equal to
its `.sha1`), of which 3,508 are Parquet files in 38 tables, as the footer scan counted. `vbt data ot fetch so
go reactome drug_warning` downloaded 11 files (1.4 MB) in 4.3 s, every sha1 matching; a rerun downloaded
nothing (0.1 s). `vbt data ot manifest` on `data/real/open_targets/25.09` downloaded nothing and recorded
31 tables, 697 files and 1,807,308,195 bytes in 3.5 s; all 697 entries equal the earlier download record. The
unmodified upstream downloader's `load_manifest` and `metadata_matches` accept all 697 entries, and its doctor
accepts them when narrowed to the 31 downloaded tables (with the full list it reports the 7 tables not
downloaded). `R2:manifest_absent` is gone from every table's check.

## 9. Round 3: acquisition, host bring-up, host-scaled memory, projects, an end-to-end run

Run on 2026-10-08 and 2026-10-09 on the same machine. These are the steps an owner's host goes through
(README, "Run it on your own infrastructure"), each run here on the real sources.

### 9.1 Acquisition from the publishers

Each descriptor declares how its files are acquired (transport plugin, release, files per table, checks,
preparation, licence); one engine, `vbt data acquire`, does the rest ([DATA_SETUP.md](DATA_SETUP.md)).

**What the sources hold, as declared and listed live:**

| Source | Transport | Release | Declared |
|---|---|---|---|
| Open Targets | `http`, inventory and sha1 from `release_data_integrity` (22,557 lines, checked against its `.sha1`) | 25.09 | 38 tables, 3,508 files, 31,131,380,890 bytes |
| Tahoe-100M | `huggingface`, tree API | revision `2dc57900...5a95` | 1,026 DE shards, 88,859,715,303 bytes; 4 metadata files, 1,451,950 bytes |
| DepMap | `json_index` (figshare article 27993248) | 24Q4 | 4 files, 850,460,784 bytes |
| Gene Ontology | `http`, sha256 pinned | archive 2026-08-05 | 32,227,785 bytes |
| Cell Ontology | `http`, sha256 pinned | 2026-06-08 | 3,347,518 bytes (UBERON 2026-10-01, optional, 12,155,980) |
| MSigDB Hallmark | `http`, sha256 pinned | 2024.1.Hs | 48,690 bytes |
| Zenodo archive | `zip_member` through the record's JSON | record 22259123 | 30 members, 92,001,168 bytes |
| Census | `s3`, read live | 2025-11-08 | nothing to download |

A review checked the declared Open Targets bytes against the local files of the 31 downloaded tables and a
HEAD request for each of the 2,811 shards of the other 7: all 38 tables equal.

**Plans.** `vbt data acquire --all --plan --dest data/real` lists every source live and writes nothing. Re-run
for this page (2026-10-09, `3f2139c`): 11.1 s, process tree 90 MB. Left to fetch: 118,183,787,997 bytes,
of which Open Targets 29,324,072,694 (the 2,811 files of the 7 tables not downloaded) and Tahoe 88,859,715,303
(the DE shards); every other source complete; 2,364 s at the assumed 50 MB/s. `vbt data ot list`: 38 tables,
3,508 files, 5.1 s, 84 MB.

**Fetches.**

* Five small Open Targets tables (`so`, `go`, `reactome`, `drug_warning`, `disease_hpo`) into a scratch root:
  12 files, 3,161,205 bytes, 7.2 s, tree 111 MB; a rerun downloaded nothing (2.3 s). The unmodified upstream
  doctor (narrowed to those tables) and downloader (`load_manifest`, `metadata_matches`) accepted the result.
* One command into `data/real`: GO (32.23 MB, 1.4 s), the Cell Ontology and UBERON (15.50 MB, 1.1 s), the 30
  Zenodo members read from the 2.9 GB zip by range requests and checked by CRC-32 (92.00 MB, 11.6 s), and
  DepMap, MSigDB and the Tahoe metadata already present and verified: 139.73 MB in 18.5 s, tree 273 MB.
* Tahoe with one DE shard (a scratch copy of the descriptors): 5 files (shard 0 is 91,442,763 bytes) in 4.3 s;
  the unmodified `prepare_tahoe.py` read 3,986,181 rows and wrote 155,254 permissive, 117,979 significant and
  82,616 high-quality rows in 2.0 s (365 MB); all 7 Tahoe tables were then `ready` (18.0 s).
* Manifests: `vbt ds check` of 12 DepMap, GO, MSigDB and Zenodo tables against the manifests the engine wrote:
  all `ready`, `R2:manifest` ok, each manifest's release equal to the descriptor's (6 min 15 s, 1,083 MB). The
  Zenodo manifest sits one level above the descriptor's root; its 30 entries are rebased onto it.
* On demand: with `data.acquisition.auto: under_budget` and a 50 MB budget, a call to
  `drug.get_drug_warnings` with the table absent was refused `not_ready`; the refusal's `acquire` entry named
  `vbt data acquire open_targets.drug_warning`, 203,238 bytes, 1 file, the licence and the decision. The
  between-turns step fetched it (5.1 s, 1.1 s of transfer), and the next check was `ready` (11.6 s for the
  whole script, 320 MB).

**Facts found on the way.** The GO archive `releases/2026-08-05/go-basic.obo` holds `data-version:
releases/2026-07-26`, and `release.geneontology.org/2026-07-26` answers 404, so the archive path is the pinned
release. Zenodo record 22259123 redirects to 22259124; its zip is 2,909,966,318 bytes with 2,866 members
(4,386,500,704 bytes uncompressed). The figshare article lists 73 files (30,825,074,613 bytes), of which the
descriptor takes 4. The Census client's `release.json` (stable 2025-11-08) and the bucket's
`cell-census/release.json` (stable 2025-01-30, latest 2025-11-10) are different files. A HEAD request for the
MSigDB file reports the gzip length (20,551 bytes) unless it sends `Accept-Encoding: identity` (48,690).

### 9.2 Host bring-up: `vbt setup`

`vbt setup` ran end to end on the 31 Open Targets tables, on this host and in the harness image
(`docker run --memory 6g`); DEPLOYMENT.md §9 has its numbers (host run 585 s, process tree 2,663 MB; a second
run skipped every data step in 18 s). Its acquire step plans with `vbt data acquire --for-tools <every tool the
enabled agents may call>`.

Re-run for this page: `vbt --profile production setup --plan --home <empty directory>` (2026-10-09, `3f2139c`)
took 12.6 s at 187 MB and wrote nothing under the home. It planned the acquire step on a fresh host (before
round 3's fix it could not, because it named a `host.yaml` that did not exist yet): 85 tools; sources
`open_targets` 25.09 and `tahoe_100m` local, `gene_ontology` local (the optional hierarchy of
`get_go_enrichment`), and cBioPortal, the Census, ClinicalTrials.gov and PubMed remote; 117.71 GB to fetch
(39 min at an assumed 50 MB/s). Its time estimates for the data steps (size 21 min, index 6.4 h, check 2.7 h,
calibrate 67 min; 11.2 h in all) extrapolate linearly from rates measured on 1.76 GB of Open Targets, which
is unverified at that size. It noted that the full-load sum is over this host's plan, that there is no GPU,
and that 117.71 GB exceed the 6.34 GB free.

### 9.3 `vbt validate` on this host

The first full run (2026-10-09, before the overlays declared the whole-table reads `full_table`): deep check,
replication quick, a profile with 3,000 MB server limits and a 3,200 MB host budget to keep one agent's
process tree under this machine's 6 GB cap. 829.1 s, process tree 4,677 MB, exit 1.

| Step | Status | Summary |
|---|---|---|
| host | PASS | plan 13,680 MB (MemTotal 16,095, cgroup 13,680); containment cgroup |
| lint | PASS | 11 sources, 113 bound tools, 0 errors |
| check | PASS | deep: 85 ready, 10 absent; 14 processes, largest 2,205 MB; 319.2 s |
| correctness | FAIL | 145 cases on 9 servers: 111 correct, 17 typed refusals, 0 wrong, 4 admitted but killed the server, 13 unanswered (server down); 392.0 s |
| latency | PASS | 212 timed calls on 9 servers (§5.5) |
| memory | PASS | 4 tables measured, sample tier 1.09-1.25 of the measured load (§5.1) |
| live | PASS | 3 of 3 endpoints; 6 of 6 server checks on single_cell, clinicaltrials, pubmed |
| replication | PASS | 1,441 of 1,441 compared rows match the authors' tables |
| model | SKIPPED | no model server answered |

All 17 failed cases were on `target`: `get_chemical_probes` and `get_genetic_constraint` were admitted and the
server was killed at 3,000 MB loading `target`, and after four kills the bridge stopped restarting it (§5.1 has
the fix). With `target` alone at 4,400 MB: 39 cases, 37 correct, 2 refused, 0 wrong.

| Test | Correct | Refused | Killed | Unavailable | Skipped |
|---|---:|---:|---:|---:|---:|
| CT-1 identifier forms | 23 | 7 | 2 | 6 | 2 |
| CT-2 absent identifiers | 38 | | | | 2 |
| CT-3 exact counts | 23 | 7 | 2 | 6 | 2 |
| CT-4 top k | 4 | 2 | | 1 | 4 |
| CT-5 invalid arguments | 23 | | | | |
| CT-6 thresholds and nulls | | 1 | | | 5 |

The `off` calls (67 of 145 were made; the others were skipped by the guard that sizes what upstream would
load, or followed a typed refusal) showed the upstream answers the gateway corrects: 16 absent identifiers
answered as success (7 with 0 rows), 11 invalid arguments accepted (10 with 0 rows), and 2 of 3 top-k answers
different from the oracle (`drug.get_drug_adverse_events` returned 146.2, 87.7, 81.8 against the oracle's
1,974.9, 659.8, 494.8 of 26 rows; `drug.search_known_drugs` 3, 3, 2 against 3, 3, 3 of 61). Enforce mode was
correct on all three. An earlier
run reported 33 wrong answers; all were bugs in `validate`'s own case planning and judging, fixed before the
runs above.

Since then: the replication step is an extension (`validate.extensions`); `--only host,lint,replication
--replicate quick` passed with 1,441 of 1,441 rows in 78.3 s (81.0 s in all, largest child 1,517 MB). The
verdict is INCOMPLETE (exit 1) when a correctness, live or model step that applies was skipped or the enabled
agents read tables the host lacks. The session check re-run for this page is in §5.2. The full run was not
repeated after the overlay change.

### 9.4 Projects: descriptors and utilities the system created

Project data is described, acquired and served by the same general mechanisms; the data engineer's tools
(`InspectDataset`, `RegisterDataSpec`, `RegisterUtility`) drafted, validated and registered everything below,
and no project Python entered the core ([PROJECTS.md](PROJECTS.md)).

**Real files registered as drafted** (each registration: staging, lint, the data child's standard check under
the reaper, install, provenance):

| File | Rows x columns | Key drafted | Registration |
|---|---|---|---|
| DepMap 24Q4 `Model.csv` | 2,105 x 47 | `ModelID` | `ready`, 3.48 s |
| Tahoe-100M `drug_metadata.parquet` | 379 x 9 | `drug` (pattern `^\S(?:.*\S)?$`) | `ready`, 3.13 s |
| Tahoe-100M `cell_line_metadata.parquet` | 1,000 x 10 | `cell_name, Driver_Gene_Symbol, Driver_ProtEffect_or_CdnaEffect` (last part nullable, 25 nulls) | `ready`, 3.36 s |
| HGNC complete set (live) | 45,233 x 53 | `hgnc_id` | `ready`, 4.75 s |
| Open Targets 25.09 `go`, `target_prioritisation`, `expression`, `target` (shard directories, registered in place) | 48,165 / 78,726 / 43,804 / 78,726 rows | `id` / `targetId` / `id` / `id` | `ready`, 3.86-4.30 s each |
| Open Targets 25.09 `drug_mechanism_of_action` | 6,332 | `mechanismOfAction, targetName, targetType`, `row_identity: content_hash` | `ready`, 3.82 s |
| ClinVar `gene_condition_source_id` (§9.5) | 14,211 x 9 | `#GeneID, DiseaseName, SourceID` | `ready`, 4.6 s |

What the real files showed that fixtures had not, each now handled by the drafts: key patterns (124 of 379
Tahoe drug names, such as "Almonertinib (mesylate)", fail `^[A-Za-z0-9_.:-]+$`); a table unique only with a
column that has nulls (a nullable composite key); a TSV whose column 39 is empty in the first block and text
later (`column_types`); a tab-separated `.txt`; a column named `pseudogene.org` (now a literal column name);
a header `#GeneID` that is not a path; rows that differ only in a nested list (`content_hash`).
`InspectDataset` profiles untrusted files in the sandbox: on the `expression` and `target` directories the
harness process stayed at 74.5-75.3 MB while the sandboxed child peaked at 1,015-1,361 MB.

**System-authored acquisition specs.** A project descriptor `hgnc_live` with an `http` acquisition section
registered `missing`, with the note naming the fetch command; that command, run unchanged, downloaded
16,973,125 bytes in 1.82 s (sha256 verified) and the table checked `ready` (4.99 s). A project descriptor for a
release the core does not ship, Open Targets 25.06 (`drug_warning` and `go`, the generic `http` transport with
25.06's `release_data_integrity` as checksum list), fetched 1,172,632 bytes (1 + 4 shards) in 6.29 s, and both
tables checked `ready` (3.69 s). Fetched again without `--dest`, the files went to `<project>/data/ot2506` (6.72 s,
1.41 s of transfer). 25.06 against 25.09: `drug_warning` 1,676 rows in both; `go` 48,030 rows in 4 files
against 48,165 in 8. A descriptor rooted at `/etc` was refused ("outside permitted roots"), one at the checkout's
`.git` too ("blocked by policy").

**Queries through the gateway**, each compared with pandas on the same files: `find depmap_models.models where
OncotreeLineage=Lung` (total 260), `find hgnc_genes.genes where symbol=EGFR` (HGNC:3236), `find
tahoe_drugs.drugs where targets=EGFR` (6), `aggregate tahoe_cell_lines.lines group_by Organ` (15 groups; Bowel
426), `aggregate ot2506.drug_warning group_by warningType` (Black Box Warning 1,136, Withdrawn 540), `find
ot2506.go where id=GO:0005515` (protein binding), `aggregate ...target_prioritisation group_by isInMembrane`
(null 59,727; 0: 14,885; 1: 4,114). All equal. A utility the engineer registered (`util__lineage_counts`) passed
its test in the sandbox (`bwrap+netns`) and answered in the next turn.

### 9.5 An end-to-end run with a small model

[E2E_RUN.md](E2E_RUN.md) records it in full. The model was `unsloth/Qwen3.5-2B-GGUF` `Qwen3.5-2B-Q4_K_M.gguf`
(1,280,835,840 bytes), from the default model's family, served by llama.cpp on the 4 CPUs (one request alone:
prefill 108.4 tokens/s, decode 11.7 tokens/s; 2 slots of 64K tokens), with the harness's `e2e-cpu` profile, the
seven unmodified Open Targets servers on the 31 downloaded 25.09 tables, the gateway enforcing, the data
child, a project, and memory limits `auto` from a 6,000 MB share.

* **Session 1, a research question.** The third served run (9,339 s) ran orientation, a plan and three
  delegations: 80 tool calls (44 upstream, 10 native), PCSK9 resolved to ENSG00000169174 by
  `search_targets_by_name` and `mcp__data__search`, 4 Reactome pathways. Review was enforced when the CSO tried
  to end without it. The model filed no claims: the CSO was at 13 of 14 turns when this sandbox's two-hour
  limit on background commands stopped the model server.
* **Session 2, a dataset with no descriptor** (ClinVar `gene_condition_source_id`). The 2B model, as data
  engineer, never called `InspectDataset` and invented its YAML; the turn was interrupted after 35 minutes. The
  same steps driven by the scripted model over the real stack: `InspectDataset` profiled the 14,211 rows, the
  draft registered unchanged and checked `ready`, a utility and its tests registered (2 passed in the
  sandbox), and the genomics analyst then found the 2 PCSK9 rows (MONDO:0005439, MONDO:0011369) with
  `mcp__data__find` and summarised them with `util__condition_summary`; one claim was filed, and the answer
  cites it.
* **After the run.** `vbt verify --data` of the scripted session 1: INCOMPLETE, 2 problems (the refused
  `get_target_info`, and `degraded_run` because 7 tables are absent), 2 claims with valid evidence, 30 tables
  unchanged, replays 2 of 2 matching. `vbt ds retro-audit` (5.2 s) and `vbt ds graduate` (6.6 s) ran on the
  served run.
* **Nine harness bugs** surfaced and were fixed with regression tests (`tests/test_e2e_stack.py`): a second
  readiness check at every session start; the native data tools listing every table's column map (the genomics
  analyst's tools went from 345,188 to 84,230 characters, its first request from 117,265 to 29,699 tokens); an
  overflow message that did not name the fixed part; the data child charged to the upstream host budget (11 of
  22 `too_large`); the llama.cpp provider's hint naming vLLM; `ds retro-audit` without `--project`; drafts that
  wrote `#GeneID` as a path; and two retro-audit misreadings of refused and native calls.

### 9.6 The third review's other fixes, checked on real data

* **Cell Ontology.** The shipped descriptor no longer falls back to the 14-term test fixture and checks the
  file's `data-version`. `vbt ds check --table cell_ontology.term --depth deep`: the real 2026-06-08 file
  `ready` (4.5 s); the fixture `stale` ("data-version mini-cl/test is not the pinned release 2026-06-08",
  naming the acquire command; 3.4 s); no file `missing` (3.2 s).
* **Derived computations as plugins.** The DepMap essentiality and tissue-specificity computations left the
  data child's core and became the shipped `derived` plugins. The same 12 calls through the gateway before and
  after gave identical answers on a real-row subset (the 25.09 rows of 402 genes; 880-882 MB, 94.5-101.6 s per
  run of 12). A plain pyarrow read of TP53's `target_essentiality` row (29 tissues, 1,183 screens, not
  essential) equals the derived record. Running the 12 calls on the whole `target_essentiality` table in one
  in-process gateway without a data-child limit reached 4.8 GB in 10 minutes without an answer and was stopped
  (the same path in both versions; not investigated further).
* **The Census sample's keys from the descriptor.** With the id column, seed and filter syntax read from the
  descriptor and overlay instead of core code, spleen pulls of 20,000 and 100,000 cells were drawn as 19,782 and
  99,771 cells (the same draws as §8.1) and 200,000 was refused before the call (577,677 cells match), in
  11.9-19.0 s at 1,078 MB.

### 9.7 The suite with this page

`ruff check src tests`: clean. `python -m pytest -q -p no:cacheprovider tests` (offline, default settings) on
the tree with this page: 3,869 passed, 184 skipped, 87 xfailed, 0 failed in 1,359.4 s, process-tree peak
3,942 MB. The counts equal those at `3f2139c`; the documentation tests (the runbook's commands, the deployment
files) pass. The baseline at the start of round 3 (`da6ffd7`) was 3,305 passed, 148 skipped and 87 xfailed; the
added skips are opt-in real-data, network and served-model tests.
