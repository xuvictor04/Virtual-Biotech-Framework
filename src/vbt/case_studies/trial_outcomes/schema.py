"""Pydantic output schema for per-trial outcome annotation (Case study 1).

Field vocabulary follows the released Virtual Biotech clinical-trial dataset
(``datasets/clinical_trials/clinical_trial_labels_reconciled.csv``) so agent
output can be compared row-for-row with the published labels. Stop-reason
categories follow the Razuvayevskaya et al. (Nat Genet 2024) ontology used by
Open Targets.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EndpointResult(str, Enum):
    POSITIVE = "POSITIVE"          # sufficient evidence the endpoint was met
    NEGATIVE = "NEGATIVE"          # endpoint not met / no significant benefit
    UNKNOWN = "UNKNOWN"            # applicable, but no results located
    NOT_APPLICABLE = "NOT_APPLICABLE"  # e.g. trial withdrawn before enrolment / stopped before readout


class StopCategory(str, Enum):
    ANOTHER_STUDY = "Another study"
    BUSINESS = "Business or administrative"
    COVID = "COVID-19"
    INSUFFICIENT_DATA = "Insufficient data"
    INSUFFICIENT_ENROLLMENT = "Insufficient enrollment"
    INTERIM_ANALYSIS = "Interim analysis"
    INVALID_REASON = "Invalid reason"
    LOGISTICS = "Logistics or resources"
    NEGATIVE = "Negative"
    NO_CONTEXT = "No context"
    REGULATORY = "Regulatory"
    SAFETY = "Safety or side effects"
    SLOW_RECRUITMENT = "Slow recruitment"
    STUDY_DESIGN = "Study design"
    STAFF_MOVED = "Study staff moved"
    SUCCESS = "Success"
    UNCATEGORISED = "Uncategorised"


class EvidenceSource(str, Enum):
    CTGOV = "ClinicalTrials.gov"
    PUBMED = "PubMed"
    PRESS_RELEASE = "Press release"
    REGULATORY = "Regulatory announcement"
    OTHER_WEB = "Other web"
    NONE = "None"


AE_ORGAN_SYSTEMS = ("infections", "gastrointestinal", "cardiac", "blood_lymphatic", "nervous",
                    "respiratory", "general", "vascular", "renal", "injury")


class SeriousAERates(BaseModel):
    """Percent of treated participants with >=1 serious adverse event, by MedDRA system organ class
    (experimental arms pooled). Leave a field null when not reported."""

    infections: Optional[float] = Field(None, ge=0, le=100, description="Infections and infestations")
    gastrointestinal: Optional[float] = Field(None, ge=0, le=100, description="Gastrointestinal disorders")
    cardiac: Optional[float] = Field(None, ge=0, le=100, description="Cardiac disorders")
    blood_lymphatic: Optional[float] = Field(None, ge=0, le=100, description="Blood and lymphatic system disorders")
    nervous: Optional[float] = Field(None, ge=0, le=100, description="Nervous system disorders")
    respiratory: Optional[float] = Field(None, ge=0, le=100, description="Respiratory, thoracic and mediastinal")
    general: Optional[float] = Field(None, ge=0, le=100, description="General disorders / administration site")
    vascular: Optional[float] = Field(None, ge=0, le=100, description="Vascular disorders")
    renal: Optional[float] = Field(None, ge=0, le=100, description="Renal and urinary disorders")
    injury: Optional[float] = Field(None, ge=0, le=100, description="Injury, poisoning and procedural complications")


class TrialAnnotation(BaseModel):
    """Structured outcome record for one ClinicalTrials.gov study."""

    nct_id: str = Field(pattern=r"^NCT\d{8}$")
    overall_status: str = Field(description="Registry status, e.g. Completed, Terminated, Withdrawn")
    primary_endpoint_result: EndpointResult
    primary_endpoint_rationale: str = Field(
        description="1-3 sentences: the endpoint, effect size / p-value, and why it is POSITIVE/NEGATIVE/UNKNOWN")
    secondary_endpoint_result: EndpointResult
    secondary_endpoint_rationale: str
    stop_reason_categories: list[StopCategory] = Field(
        default_factory=list, description="Only for stopped trials (terminated/withdrawn/suspended); else empty")
    stop_reason_text: Optional[str] = None
    serious_ae_pct: SeriousAERates = Field(default_factory=SeriousAERates)
    total_serious_ae_pct: Optional[float] = Field(None, ge=0, le=100)
    results_source: EvidenceSource = Field(description="Tier that supplied the endpoint results")
    ae_source: EvidenceSource = Field(description="Tier that supplied the adverse-event numbers")
    pubmed_ids: list[str] = Field(default_factory=list, description="PMIDs confirmed to report THIS trial")
    source_urls: list[str] = Field(default_factory=list, description="Press releases / regulatory pages used")
    tiers_consulted: list[EvidenceSource] = Field(description="Every tier searched, in order")
    confidence: str = Field(pattern="^(high|medium|low)$")
    notes: Optional[str] = Field(None, description="Ambiguities: multi-arm, borderline p, single-arm designs")

    @field_validator("pubmed_ids")
    @classmethod
    def _digits(cls, v: list[str]) -> list[str]:
        bad = [p for p in v if not str(p).isdigit()]
        if bad:
            raise ValueError(f"PMIDs must be numeric: {bad}")
        return [str(p) for p in v]

    def to_label_row(self) -> dict:
        """Flatten to the released-label column layout."""
        def res(r: EndpointResult):
            return None if r is EndpointResult.NOT_APPLICABLE else r.value
        row = {
            "nct_id": self.nct_id,
            "primary_endpoint_result": res(self.primary_endpoint_result),
            "secondary_endpoint_result": res(self.secondary_endpoint_result),
            "virtualbiotech_stop_reason_categories": "|".join(c.value for c in self.stop_reason_categories) or None,
            "results_source": self.results_source.value,
            "ae_source": self.ae_source.value,
            "pubmed_ids": ",".join(self.pubmed_ids) or None,
            "confidence": self.confidence,
        }
        for organ in AE_ORGAN_SYSTEMS:
            row[f"ae_serious_{organ}_pct"] = getattr(self.serious_ae_pct, organ)
        return row
