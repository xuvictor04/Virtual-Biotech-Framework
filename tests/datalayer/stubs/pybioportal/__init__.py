"""A network-free ``pybioportal`` for the correctness tests (§19 CT-2).

One study (``study_x``), two patients and three samples (P-01 has two). The functions the
upstream clinicaltrials tools import return DataFrames shaped like the real package's
(``process_response`` flattens the REST JSON into one row per record), and an unknown study
raises the way ``process_response`` does on a 404. The real package never invents records:
the phantom ``{sampleId: "NOPE-01", patientId: None}`` row that upstream ``get_clinical_data``
returns is built by the upstream tool itself from the requested ids.
"""

from __future__ import annotations

import types

import pandas as pd

STUDY = "study_x"
PATIENTS = ("P-01", "P-02")
SAMPLES = {"S-01": "P-01", "S-02": "P-01", "S-03": "P-02"}
PATIENT_DATA = {"P-01": {"OS_MONTHS": "24.5", "OS_STATUS": "1:DECEASED", "AGE": "61"},
                "P-02": {"OS_MONTHS": "40.0", "OS_STATUS": "0:LIVING", "AGE": "55"}}
SAMPLE_DATA = {"S-01": {"SAMPLE_TYPE": "Primary"}, "S-02": {"SAMPLE_TYPE": "Metastasis"},
               "S-03": {"SAMPLE_TYPE": "Primary"}}
ATTRIBUTES = [("OS_MONTHS", "Overall Survival (Months)", "NUMBER", True),
              ("OS_STATUS", "Overall Survival Status", "STRING", True),
              ("AGE", "Diagnosis Age", "NUMBER", True),
              ("SAMPLE_TYPE", "Sample Type", "STRING", False)]

calls: list[tuple[str, dict]] = []   # (function, kwargs) for tests that inspect what was asked


def _record(name: str, **kwargs) -> None:
    calls.append((name, kwargs))


def _not_found(fail_msg: str, study_id: str) -> Exception:
    # __aux_funcs.process_response: "<fail_msg> Status code: 404\n Error messagge: <message>"
    return Exception(f"{fail_msg} Status code: 404\n Error messagge: Study not found: {study_id}")


def _check(study_id: str, fail_msg: str) -> None:
    if study_id != STUDY:
        raise _not_found(fail_msg, study_id)


def get_all_cancer_types(*args, **kwargs) -> pd.DataFrame:
    _record("get_all_cancer_types")
    return pd.DataFrame([{"cancerTypeId": "luad", "name": "Lung Adenocarcinoma", "parent": "nsclc",
                          "dedicatedColor": "Gainsboro", "shortName": "LUAD"}])


def get_all_studies(*args, **kwargs) -> pd.DataFrame:
    _record("get_all_studies")
    return pd.DataFrame([{"studyId": STUDY, "name": "Fixture study", "description": "two patients",
                          "cancerTypeId": "luad", "pmid": "1", "citation": "Fixture 2026", "allSampleCount": 3}])


def get_study(study_id: str) -> pd.DataFrame:
    _record("get_study", study_id=study_id)
    _check(study_id, "Failed to get the study by ID.")
    return pd.DataFrame([{"studyId": STUDY, "name": "Fixture study", "description": "two patients",
                          "cancerTypeId": "luad", "sequencedSampleCount": 3, "cnaSampleCount": 3,
                          "mrnaRnaSeqV2SampleCount": 0, "rppaSampleCount": 0, "completeSampleCount": 3,
                          "citation": "Fixture 2026", "pmid": "1"}])


def get_all_molecular_profiles_in_study(study_id: str, **kwargs) -> pd.DataFrame:
    _record("get_all_molecular_profiles_in_study", study_id=study_id)
    _check(study_id, "Failed to get all molecular profiles in a study.")
    return pd.DataFrame([{"molecularProfileId": f"{STUDY}_mutations", "molecularAlterationType": "MUTATION_EXTENDED",
                          "datatype": "MAF", "name": "Mutations", "description": "fixture"}])


def get_all_clinical_attributes_in_study(study_id: str, **kwargs) -> pd.DataFrame:
    _record("get_all_clinical_attributes_in_study", study_id=study_id)
    _check(study_id, "Failed to get all clinical attributes in a study.")
    return pd.DataFrame([{"clinicalAttributeId": a, "displayName": d, "datatype": t, "patientAttribute": p,
                          "priority": "1", "description": d, "studyId": STUDY} for a, d, t, p in ATTRIBUTES])


def get_all_samples_in_study(study_id: str, **kwargs) -> pd.DataFrame:
    _record("get_all_samples_in_study", study_id=study_id)
    _check(study_id, "Failed to get all samples in a study.")
    return pd.DataFrame([{"uniqueSampleKey": f"k{s}", "uniquePatientKey": f"k{p}", "sampleType": "Primary Solid Tumor",
                          "sampleId": s, "patientId": p, "studyId": STUDY} for s, p in SAMPLES.items()])


def get_all_clinical_data_in_study(study_id: str, attribute_id=None, clinical_data_type: str = "SAMPLE",
                                   **kwargs) -> pd.DataFrame:
    _record("get_all_clinical_data_in_study", study_id=study_id, clinical_data_type=clinical_data_type)
    _check(study_id, "Failed to get all clinical data in a study.")
    rows = []
    if clinical_data_type == "PATIENT":
        for pid, attrs in PATIENT_DATA.items():
            rows += [{"uniquePatientKey": f"k{pid}", "patientId": pid, "studyId": STUDY, "clinicalAttributeId": a,
                      "value": v} for a, v in attrs.items()]
    else:
        for sid, attrs in SAMPLE_DATA.items():
            rows += [{"uniqueSampleKey": f"k{sid}", "sampleId": sid, "patientId": SAMPLES[sid], "studyId": STUDY,
                      "clinicalAttributeId": a, "value": v} for a, v in attrs.items()]
    return pd.DataFrame(rows)


def _module(name: str, **functions) -> types.ModuleType:
    mod = types.ModuleType(f"{__name__}.{name}")
    mod.__dict__.update(functions)
    return mod


cancer_types = _module("cancer_types", get_all_cancer_types=get_all_cancer_types)
studies = _module("studies", get_all_studies=get_all_studies, get_study=get_study)
molecular_profiles = _module("molecular_profiles",
                             get_all_molecular_profiles_in_study=get_all_molecular_profiles_in_study)
clinical_attributes = _module("clinical_attributes",
                              get_all_clinical_attributes_in_study=get_all_clinical_attributes_in_study)
clinical_data = _module("clinical_data", get_all_clinical_data_in_study=get_all_clinical_data_in_study)
samples = _module("samples", get_all_samples_in_study=get_all_samples_in_study)
mutations = _module("mutations")
molecular_data = _module("molecular_data")
discrete_copy_number_alterations = _module("discrete_copy_number_alterations")
genes = _module("genes")
