# Data layer on real data

Through phase 5, the data layer ([DATA_LAYER.md](DATA_LAYER.md), status in
[DATA_LAYER_STATUS.md](DATA_LAYER_STATUS.md)) was tested only on generated fixtures, stubs and recorded
responses. On 2026-10-08 the descriptors, the data child and the gateway were run against the real
releases and live APIs listed below. This page records what was checked, which descriptor facts held, which
were wrong and how they were fixed, the six correctness tests on the real Open Targets release, the memory
and latency that were measured, and what is still unchecked because the data could not be obtained here.

Every number on this page was observed in those runs; nothing is extrapolated unless it says so.

* **Machine.** 4 CPUs, 16,094 MB RAM, no swap, Linux 6.18. Outbound HTTPS went through a proxy. Other
  jobs sometimes ran at the same time, so wall-clock times are approximate.
* **Upstream code.** The authors' MCP servers were launched unchanged from `third_party/TheVirtualBiotech`
  through `MCPBridge`. "Off" below means `data.gateway.mode: "off"` (the upstream answer as an agent gets it
  today). "Enforce" means the gateway guards the server.
* **Memory.** Memory figures are peak RSS. They come either from the reaper's status file of one process,
  or from the summed VmRSS of a whole process tree sampled every 0.25-1 s. Runs kept their process tree
  under a 5,800-6,000 MB cap. The one early check that ran without the reaper is named in section 5.2.
* **Oracle.** Expected answers come from pyarrow reads of the same files, written independently of the
  code under test.

## Contents

1. [Sources and releases](#1-sources-and-releases)
2. [Descriptor facts: confirmed and corrected](#2-descriptor-facts-confirmed-and-corrected)
3. [The six correctness tests on real data](#3-the-six-correctness-tests-on-real-data)
4. [Wrong answers of the layer itself, found on real data](#4-wrong-answers-of-the-layer-itself-found-on-real-data)
5. [Memory and latency](#5-memory-and-latency)
6. [What still needs data we could not get](#6-what-still-needs-data-we-could-not-get)
7. [Re-running the checks](#7-re-running-the-checks)

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

The DepMap portal's download API (`https://depmap.org/portal/api/download/files`) answered HTTP 200 with a
4,842-byte Cloudflare "verify you are a person" page, so the newest DepMap release on figshare (24Q4) was
used. Newer releases exist only on the portal.

**Downloaded files.** They are in the shared, git-ignored directory `data/real/<source>/<release>/`
(about 2.8 GB) and are not committed. The test suite carries only small snapshots of what was measured,
each naming its source URL and retrieval date:

| Directory under `tests/datalayer/real/` | Contents |
|---|---|
| `ot_25_09/` | per table: source URL, shards, rows, row groups, bytes, writer, the Arrow schema and per-leaf statistics. `_release.json` (manifest, integrity file). `values.json`: the vocabularies, codes, scales, keys and orientations measured on the data. |
| `ot_25_09_servers/` | the footer statistics of the 25.09 `target` table, used to test the shipped memory factors |
| `tahoe/` | DE footer facts, metadata excerpts, 30 prepared DE rows |
| `depmap/`, `ontology/` | CSV excerpts of the DepMap files; whole OBO stanzas and three GMT lines |
| `live/` | recorded CT.gov, cBioPortal and E-utilities responses and replayed request sequences; Census read summaries |

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
  keyed by position. Now `ready`.
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

* **Not finished.** The deep check of l2g_prediction was stopped at 900 s: the full key check of its
  features item table renders every item in Python (35.5 M items when stopped). The deep checks of
  target_essentiality and interaction_evidence were not run to completion after the fixes. An earlier run
  took 729.8 s at 3,008 MB for target_essentiality.
* **10.3 GiB.** An early deep check of interaction_evidence (971.3 s) peaked at 10.3 GiB, because
  `vbt ds check` started the data child without the reaper. It now runs under the reaper
  (`preflight.run_contained`). A composite reference also keeps only the sampled tuples now; it kept all
  14.5 M.
* R10's edge check now looks up the reverse of each sampled edge in the whole table: 3.7 s on interaction
  and 5.4 s on interaction_evidence, with no reverse missing. Before, it reported 18,827 false
  "without their reverse" warnings.

### 5.3 Resolver indexes

* **Build.** `vbt ds index build` for every id type with a universe took 263.1 s and peaked at 2,104 MB. It
  built 17 indexes; 11 failed only because their sources are not on this machine. Index sizes include
  ensembl_gene 465,164 rows, gwas_study 1,964,234 and ot_disease 335,308.
* **Reuse.** A session used to rebuild the `ensembl_gene` index although the same sidecar was on disk
  (28.2-30.1 s). It now reuses it (3.6-3.8 s).

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

During a session check the data child reaches 2.3-2.6 GB of its 3 GB limit.

**Live sources.**

* **ClinicalTrials.gov and PubMed.** Calls take 0.9-2.4 s. Peaks: clinicaltrials 173-263 MB, pubmed
  88-96 MB, data child 113-186 MB; the whole tree 553-784 MB.
* **Census.** `get_census_info` 5.0-6.6 s. `count_cells` 9.0-20.4 s. Native obs reads 9.4-14.7 s, with
  the data child at 1,328-1,400 MB. `get_anndata` (4,771 cells x 2 genes) took 38-42 s with the
  `single_cell` server at 2,782 MB; the server peaked at 2,575-2,624 MB in the other Census runs.

### 5.5 Witness and gateway latency

Measured on the target server, three repeats of the same calls:

* **Warm calls** (22 witnessed calls): witness p50 58.1 ms, p95 82.3 ms, max 98.2 ms. The whole call took
  p50 93 ms and p95 985 ms.
* **First calls.** Over all three repeats the witness p95 was 4,194 ms and the max 14,544 ms. A first call
  waited for the data child's `_stats` of the target tables (14.5 s) only to learn the storage types. The
  session check now reports those types. The `_stats` requests it avoids took 16.5 s for target and 36.0 s
  for target_go. The p95 was not measured again after this change.
* **Resolution.** p50 0.6 ms, p95 77-80 ms. Unknown ids walk the near-match rules.
* **Single-call witnesses on other tables.** study 2.0-2.2 s; interaction 1.5 s (intact) to 9.2 s (all
  sources). The expression witness for one gene took 128 s; once item tables pushed conjuncts on their parent
  row to Arrow, the same witness took 1.43 s and 338 MB in process (it took 25.07 s and 1,458 MB before).

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

## 6. What still needs data we could not get

* **Open Targets tables not downloaded.** They did not fit this machine's download budget, so only their
  footers were read:
  * evidence (8.99 GB)
  * variant (3.18 GB)
  * credible_set (2.59 GB)
  * colocalisation_coloc (3.92 GB)
  * colocalisation_ecaviar (5.16 GB)

  Only footers and samples were read for literature (4 of 334 shards) and interval (2 of 83). Their value
  facts beyond the footer bounds are unchecked, and no deep check ran on them. `query_evidence`,
  `get_evidence_by_publication` and the colocalisation tools were seen only refusing with `not_ready`. The
  interval content-identity finding rests on 2 of 83 shards.
* **CT-5's negation case.** It needs negated `disease_phenotype` evidence, which 25.09 does not hold.
* **Tables too large for the upstream loader.** expression, interaction, interaction_evidence,
  l2g_prediction and target_essentiality exceed 5.4 GB when loaded whole. Under a 5 GB server limit the
  unmodified servers cannot serve them, so the gateway refuses them `too_large` before loading. They are
  served only where an overlay declares a derived answer (the interaction network) or where a `find` stays
  within the scan budget.
* **Deep key checks of large item tables.** These render every item in Python:
  * l2g_prediction features: stopped at 900 s after 35.5 M items.
  * target_essentiality and interaction_evidence: not run to completion after the fixes.

  An Arrow path for item keys (flattened lists with parent indices) would do what the flat-key fixes did.
* **Tahoe-100M.** Only 13 of 65,218 contrasts were read. A prepared DE file for the whole release
  (83 GiB of source) was not built, and `obs_metadata.parquet` (2.29 GB) was not downloaded. The figure
  of about 1.4e8 permissive rows is an estimate from the 13 contrasts.
* **DepMap.** Releases newer than 24Q4 are only on the portal, behind its browser check.
* **Census.**
  * `get_anndata_donor_balanced` still runs upstream behind `requires_fixed: dataset_id`. Upstream
    stratifies by cell type, the derived `(dataset_id, donor_id)` sample does not, and a `soma_joinid in
    [...]` filter of up to 200,000 ids through the unmodified server is untested.
  * The Census row read keeps the note "N page(s) read within the source's budget" when it cuts a full match
    to `limit`.
* **CT.gov under the evidence ceiling.** A native find counts the trials both first posted and last updated
  by the ceiling: those are the rows it can return under `rows: withhold`. The upstream count tool bounds
  only the first posting, so the two totals differ: 87 against 2,323 for RECRUITING trials under 2017-12-31.
* **Matrix tables.** R6, R7, R9 and R10 (value facts, vocabularies, references, relations) do not run on
  them.
* **F23 (calibration, soak, graduation).**
  * The witness p95 target (under 300 ms on composite-key tables) holds for warm calls on the target
    server (82.3 ms). It was not measured across servers after the storage-type change, and a single
    interaction witness takes 1.5-9.2 s.
  * Validating `profile: fidelity` on the paper scenarios, and retro-audit evidence for graduation, need
    model runs of those scenarios. None were made.

## 7. Re-running the checks

The default suite is offline. The real-data tests are opt-in:

| Variable | Tests | Needs |
|---|---|---|
| `VBT_DL_REAL_DATA=<dir>` | `test_dl_real_ot_schema.py` (downloaded files against the snapshots), `test_dl_real_ot_servers.py` (standard checks, the six tests, the wrong answers off), `test_dl_real_tahoe_depmap.py`, `test_dl_review_realdata.py`, `test_dl_hardening.py` (whole-table loads under the reaper) | `<dir>` is the shared root (`data/real`, holding `open_targets/25.09/`, `tahoe/<revision>/`, `depmap/24Q4/`, ...) or the `open_targets/25.09` directory itself. The server tests also need `third_party/TheVirtualBiotech` checked out (or `VBT_UPSTREAM`). |
| `VBT_DL_NETWORK=1` | `test_dl_real_live.py`, `test_dl_live_review_fixes.py` (live CT.gov, cBioPortal, E-utilities, Census), the live footer reads of `test_dl_real_ot_schema.py` (`full`: every shard) and `test_dl_real_tahoe_depmap.py` | network access; `cellxgene-census` for the Census tests |
| `VBT_DL_HOST_LIMITS=1` | the cgroup kill tests of `test_dl_hardening.py` | a writable cgroup v1 hierarchy |

For example:

```bash
VBT_DL_REAL_DATA=data/real python -m pytest -q tests/datalayer/test_dl_real_ot_servers.py
VBT_DL_NETWORK=1 python -m pytest -q tests/datalayer/test_dl_real_live.py
```

The server tests start the unmodified upstream servers and the data child; their process tree peaked at
5.4-5.6 GB here. Do not run them under an address-space limit (`prlimit --as`): the gateway-off servers
inherit it and fail. To fetch Open Targets 25.09, use the upstream downloader
(`third_party/TheVirtualBiotech/tools/download_open_targets.py`). Then run `vbt ds check --table
open_targets.<table>` on what you downloaded.

**Updating the snapshots.** `VBT_UPDATE_REAL_SNAPSHOT=1` rewrites the Open Targets snapshots from the live
release. `tests/datalayer/real/live/README.md` explains how to re-record the live responses.
