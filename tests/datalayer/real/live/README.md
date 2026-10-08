# Recorded live-source responses

Small real responses of the live sources, retrieved on 2026-10-08 (UTC) and replayed offline by
`tests/datalayer/test_dl_real_live.py`. Every file names the URL, the parameters, the HTTP status, the
retrieval time and, under `trimmed`, what was cut to keep it small. `MANIFEST.json` lists the single
responses (and the two user-agent probes of cBioPortal, which keep only their status).

| Directory | Source | Base URL |
|---|---|---|
| `ctgov/` | ClinicalTrials.gov API v2 (`apiVersion` 2.0.5, `dataTimestamp` 2026-10-07T09:00:06) | `https://clinicaltrials.gov/api/v2` |
| `cbioportal/` | cBioPortal public API (550 public studies) | `https://www.cbioportal.org/api` |
| `eutils/` | NCBI E-utilities and the PMC ID converter | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils`, `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/` |
| `census/` | CELLxGENE Census (`stable` = 2025-11-08), read with cellxgene-census 1.18.0 / tiledbsoma 2.3.0 | `https://census.cellxgene.cziscience.com/cellxgene-census/v1/release.json`, `s3://cellxgene-census-public-us-west-2/cell-census/2025-11-08/soma/` |
| `replay/` | every request one data-child scenario of the test module made, in order | as above |

`ctgov/upstream_search_iceland.json` and `cbioportal/upstream_clinical_data_phantom.json` are the answers of
the unmodified upstream `clinicaltrials` server's own functions (`third_party/TheVirtualBiotech`,
`src/mcp_servers/clinicaltrials_mcp/tools.py`) for the arguments they record, trimmed as noted.

The `census/` files are summaries of reads made on 2026-10-08 (counts, the donors of two spleen datasets, a
200 x 3 slice of `X["raw"]`): the Census is not an HTTP API, so there is no response to keep.

To re-record (polite: one request at a time, small pages), run the scenarios of `test_dl_real_live.SCENARIOS`
with a transport that keeps each exchange; with `VBT_DL_NETWORK=1` the test module runs them live.
