# Round 3 recordings (live and remote sources, S1)

Exchanges with the real APIs, recorded on 2026-10-08 (UTC; `retrieved` in each file) by running the scenarios
of `tests/datalayer/test_dl_round3_live.py` (`SCENARIOS`) through the shipped descriptors with a transport that
kept every request. Offline the test module replays them: a request that was not recorded fails the test.

| File | Scenario | Requests |
|---|---|---|
| `clinical_derived.json` | `get_clinical_data` served derived for 2 ACC samples | `https://www.cbioportal.org/api/info`, `/studies/acc_tcga_pan_can_atlas_2018`, `/samples`, `/clinical-data` (SAMPLE, PATIENT), `/clinical-attributes` |
| `clinical_phantom.json` | the same with a sample the study does not hold | `/info`, the study, `/samples` |
| `clinical_unknown_study.json` | an unknown study (HTTP 404) | `/info`, `/studies/no_such_study_xyz` (404), its `/samples` (404) |
| `study_details_release.json` | `get_study_details` upstream-served: release and versions | `/info`, the study |
| `release_ctgov.json`, `release_pubmed.json`, `release_cbio.json` | what each source reports about its data | `https://clinicaltrials.gov/api/v2/version`; `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pubmed`; `/info` and the study |
| `ceiling_count.json` | `count_clinical_trials` RECRUITING under a 2017-12-31 ceiling | the remote count (first posting bounded), `/version`, the count with the last update bounded too |
| `find_ceiling.json` | `mcp__data__find` RECRUITING, limit 2, same ceiling | one page of 2, `/version`, the count |

Trimmed (each exchange says how under `trimmed`): the study's sample list and clinical data are cut to the
requested samples and their patients, `uniqueSampleKey`/`uniquePatientKey` dropped, clinical attributes keep their
id, study, level and type; CT.gov study records keep `identificationModule`, `statusModule` and `hasResults`;
the einfo answer keeps `dbname`, `dbbuild`, `count` and `lastupdate`; cBioPortal's `/info` drops its git fields.
`result` keeps the scenario's header (or error) as it was answered live.
