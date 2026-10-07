"""OT-25.09-shaped and Tahoe-shaped fixtures, pyarrow oracles and file-order preconditions (§19).

The six correctness tests run the unmodified upstream servers on these files and compare what
comes back with an **oracle** computed here, directly from the files with pyarrow, never from
the code under test. Everything is generated (no network, no real data):

* :func:`build_ot_fixture` writes an Open Targets 25.09 directory: real column names and nested
  types (``large_string`` where the real files use it), the rows each correctness test pins, a few
  well-formed rows for the DepMap, genetics and interaction-evidence tables the readiness checks
  bind, empty-but-valid placeholder shards for every other directory of
  ``OPEN_TARGETS_DATASETS`` and a ``.download-manifest.json`` in the upstream downloader's format.
* :func:`build_tahoe_fixture` writes a prepared Tahoe directory (``tools/prepare_tahoe.py``
  layout): float32 ``concentration``, two plates for one dose, a trailing-space drug name.
* ``oracle_*`` helpers return the expected rows, keys, totals and grains.
* :func:`assert_file_order_traps` checks that the fixture can discriminate: the first k matching
  rows in file order differ from the oracle top-k, so a file-order answer cannot pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

STR, LSTR, F64, F32 = pa.string(), pa.large_string(), pa.float64(), pa.float32()
I64, I32, BOOL = pa.int64(), pa.int32(), pa.bool_()

RELEASE = "25.09"
BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.09/output/"
MANIFEST = ".download-manifest.json"

# src/config/datasets.py OPEN_TARGETS_DATASETS (upstream, 25.09): every directory doctor requires.
OPEN_TARGETS_DATASETS = (
    "target", "credible_set", "l2g_prediction", "expression", "target_essentiality", "known_drug",
    "drug_mechanism_of_action", "disease", "disease_phenotype", "disease_hpo",
    "association_overall_direct", "association_by_overall_indirect", "association_by_datatype_direct",
    "association_by_datatype_indirect", "association_by_datasource_direct",
    "association_by_datasource_indirect", "variant", "study", "interval", "biosample", "drug_molecule",
    "drug_indication", "drug_warning", "openfda_significant_adverse_target_reactions",
    "openfda_significant_adverse_drug_reactions", "target_prioritisation", "mouse_phenotype",
    "pharmacogenomics", "go", "reactome", "so", "interaction", "interaction_evidence",
    "colocalisation_coloc", "colocalisation_ecaviar", "evidence", "literature", "literature_vector",
)

# The 23 evidence partitions of 25.09 (the descriptor declares them with ``expect: declared``).
EVIDENCE_SOURCES = (
    "cancer_biomarkers", "cancer_gene_census", "chembl", "clingen", "crispr", "crispr_screen", "europepmc", "eva",
    "eva_somatic", "expression_atlas", "gene2phenotype", "gene_burden", "genomics_england", "gwas_credible_sets",
    "impc", "intogen", "orphanet", "progeny", "reactome", "slapenrich", "sysbio", "uniprot_literature",
    "uniprot_variants",
)

# ---------------------------------------------------------------------------- identifiers

PCSK9 = "ENSG00000169174"
TP53 = "ENSG00000141510"
TP53BP1 = "ENSG00000067369"
TP53I3 = "ENSG00000115129"
G_L2G = "ENSG00000072864"          # NDE1: the L2G gene (150 loci)
UNKNOWN_GENE = "ENSG00000999999"   # well-formed, absent from target
PCSK9_MOUSE = "ENSMUSG00000044254"

T = PCSK9                          # "T" of CT-2..CT-6 (pharmacogenomics, known_drug, interaction, evidence)
T2 = TP53                          # the second pharmacogenomics target
PRIO = {"A": PCSK9, "B": TP53, "C": TP53BP1, "D": TP53I3}   # hasSafetyEvent -1, null, 0, NaN

T2D = "MONDO_0005148"
T2D_NAME = "type 2 diabetes mellitus (T2D)"
T2D_RETIRED = "EFO_0001360"
RA = "EFO_0000685"
SEIZURE = "HP_0001250"
UNKNOWN_HP = "HP_9999999"
# disease_phenotype for HP_0001250 (CT-6): X = [PCS negated], Y = [IEA], Z = [PCS, PCS negated, IEA],
# W = [IEA negated].
PHENO = {"X": "MONDO_0100001", "Y": "MONDO_0100002", "Z": "MONDO_0100003", "W": "MONDO_0100004"}

CHEMBL3, CHEMBL25, CHEMBL1000, CHEMBL99 = "CHEMBL3", "CHEMBL25", "CHEMBL1000", "CHEMBL99"
CHEMBL559288 = "CHEMBL559288"
PMID_EPMC = "30595370"

KNOWN_DRUG_SENTINEL = {"drugId": "CHEMBL3990033", "targetId": PCSK9, "diseaseId": "EFO_0000319",
                       "phase": 3.0, "status": None}
KNOWN_DRUG_KEY = ("drugId", "targetId", "diseaseId", "phase", "status")

STUDY_BIG = "GCST90000001"         # "Sbig": 200,000 samples, stored after 25 null-size studies
BIOSAMPLE_SYNONYM = "pulmo"        # matches a biosample by synonym only

# Tahoe (CT-5)
BORTEZOMIB = "Bortezomib"
A549 = "ACH-000681"
OTHER_LINE = "ACH-000001"
CONCENTRATIONS = (0.05, 0.5, 5.0)
TWO_PLATE_DOSE = 5.0               # Bortezomib x ACH-000681 at 5.0 uM was profiled on plates '1' and '2'
SELECTIVE = "SELECTIVE1"
TAHOE_DE = "tahoe_permissive_padj010.parquet"

# ---------------------------------------------------------------------------- schemas (25.09)


def _nn(name: str, typ: pa.DataType) -> pa.Field:
    return pa.field(name, typ, nullable=False)


def _el(typ: pa.DataType) -> pa.Field:
    return pa.field("element", typ, nullable=False)


_LABEL = pa.list_(pa.struct([("label", STR), ("source", STR)]))
_URLS = pa.list_(pa.struct([("niceName", STR), ("url", STR)]))

SCHEMAS: dict[str, list[Any]] = {
    "target": [
        ("id", STR), ("approvedSymbol", STR), ("biotype", STR), ("transcriptIds", pa.list_(STR)),
        ("canonicalTranscript", pa.struct([("id", STR), ("chromosome", STR), ("start", I64), ("end", I64),
                                           ("strand", STR)])),
        ("canonicalExons", pa.list_(STR)),
        _nn("genomicLocation", pa.struct([("chromosome", STR), ("start", I64), ("end", I64), ("strand", I32)])),
        ("alternativeGenes", pa.list_(STR)), _nn("approvedName", STR),
        ("go", pa.list_(pa.struct([("id", STR), ("source", STR), ("evidence", STR), ("aspect", STR),
                                   ("geneProduct", STR), ("ecoId", STR)]))),
        ("hallmarks", pa.struct([
            ("attributes", pa.list_(_el(pa.struct([("pmid", I64), ("description", STR), ("attribute_name", STR)])))),
            ("cancerHallmarks", pa.list_(_el(pa.struct([("pmid", I64), ("description", STR), ("impact", STR),
                                                        ("label", STR)]))))])),
        _nn("synonyms", _LABEL), _nn("symbolSynonyms", _LABEL), _nn("nameSynonyms", _LABEL),
        ("functionDescriptions", pa.list_(STR)),
        ("subcellularLocations", pa.list_(pa.struct([("location", STR), ("source", STR), ("termSL", STR),
                                                     ("labelSL", STR)]))),
        ("targetClass", pa.list_(_el(pa.struct([("id", I64), ("label", STR), _nn("level", STR)])))),
        _nn("obsoleteSymbols", _LABEL), _nn("obsoleteNames", _LABEL),
        ("constraint", pa.list_(_el(pa.struct([
            _nn("constraintType", STR), ("score", F32), ("exp", F32), ("obs", I32), ("oe", F32), ("oeLower", F32),
            ("oeUpper", F32), ("upperRank", I32), ("upperBin", I32), ("upperBin6", I32)])))),
        ("tep", pa.struct([("targetFromSourceId", STR), ("description", STR), ("therapeuticArea", STR),
                           ("url", STR)])),
        ("proteinIds", pa.list_(_el(pa.struct([("id", STR), ("source", STR)])))),
        _nn("dbXrefs", pa.list_(pa.struct([("id", STR), ("source", STR)]))),
        ("chemicalProbes", pa.list_(_el(pa.struct([
            ("control", STR), ("drugId", STR), ("id", STR), ("isHighQuality", BOOL),
            ("mechanismOfAction", pa.list_(STR)), ("origin", pa.list_(STR)), ("probeMinerScore", I64),
            ("probesDrugsScore", I64), ("scoreInCells", I64), ("scoreInOrganisms", I64),
            ("targetFromSourceId", STR), ("urls", _URLS)])))),
        ("homologues", pa.list_(_el(pa.struct([
            ("speciesId", STR), ("speciesName", STR), ("homologyType", STR), ("targetGeneId", STR),
            ("isHighConfidence", STR), ("targetGeneSymbol", STR), ("queryPercentageIdentity", F64),
            ("targetPercentageIdentity", F64), ("priority", I32)])))),
        ("tractability", pa.list_(_el(pa.struct([_nn("modality", STR), _nn("id", STR), _nn("value", BOOL)])))),
        ("safetyLiabilities", pa.list_(_el(pa.struct([
            ("event", STR), ("eventId", STR), ("effects", pa.list_(pa.struct([("direction", STR), ("dosing", STR)]))),
            ("biosamples", pa.list_(pa.struct([("cellFormat", STR), ("cellLabel", STR), ("tissueId", STR),
                                               ("tissueLabel", STR)]))),
            ("datasource", STR), ("literature", STR), ("url", STR),
            ("studies", pa.list_(pa.struct([("description", STR), ("name", STR), ("type", STR)])))])))),
        ("pathways", pa.list_(_el(pa.struct([("pathwayId", STR), ("pathway", STR), ("topLevelTerm", STR)])))),
        ("tss", I64),
    ],
    "drug_molecule": [
        ("id", STR), ("canonicalSmiles", STR), ("inchiKey", STR), ("drugType", STR), ("blackBoxWarning", BOOL),
        ("name", STR), ("yearOfFirstApproval", I64), ("maximumClinicalTrialPhase", F64), ("parentId", STR),
        ("hasBeenWithdrawn", BOOL), ("isApproved", BOOL), _nn("tradeNames", pa.list_(STR)),
        _nn("synonyms", pa.list_(STR)),
        ("crossReferences", pa.list_(_el(pa.struct([_nn("source", STR), ("ids", pa.list_(STR))])))),
        ("childChemblIds", pa.list_(_el(STR))),
        ("linkedDiseases", pa.struct([_nn("rows", pa.list_(STR)), _nn("count", I32)])),
        ("linkedTargets", pa.struct([_nn("rows", pa.list_(_el(STR))), _nn("count", I32)])),
        ("description", STR),
    ],
    "known_drug": [
        ("drugId", STR), ("targetId", STR), ("diseaseId", STR), ("phase", F64), ("status", STR),
        _nn("urls", _URLS), ("ancestors", pa.list_(STR)), ("label", STR), ("approvedSymbol", STR),
        ("approvedName", STR), ("targetClass", pa.list_(STR)), ("prefName", STR), ("tradeNames", pa.list_(STR)),
        ("synonyms", pa.list_(STR)), ("drugType", STR), ("mechanismOfAction", STR), ("targetName", STR),
    ],
    "pharmacogenomics": [
        ("datasourceId", STR), ("datasourceVersion", STR), ("datatypeId", STR), ("directionality", STR),
        ("evidenceLevel", STR), ("genotype", STR), ("genotypeAnnotationText", STR), ("genotypeId", STR),
        ("haplotypeFromSourceId", STR), ("haplotypeId", STR), ("literature", pa.list_(STR)), ("pgxCategory", STR),
        ("phenotypeFromSourceId", STR), ("phenotypeText", STR),
        ("variantAnnotation", pa.list_(pa.struct([
            ("baseAlleleOrGenotype", STR), ("comparisonAlleleOrGenotype", STR), ("directionality", STR),
            ("effect", STR), ("effectDescription", STR), ("effectType", STR), ("entity", STR), ("id", STR),
            ("literature", STR)]))),
        ("studyId", STR), ("targetFromSourceId", STR), ("variantFunctionalConsequenceId", STR),
        ("variantRsId", STR), ("variantId", STR), _nn("isDirectTarget", BOOL),
        _nn("drugs", pa.list_(_el(pa.struct([("drugFromSource", STR), ("drugId", STR)])))),
    ],
    "openfda_significant_adverse_drug_reactions": [
        ("chembl_id", STR), ("event", STR), _nn("count", I64), ("llr", F64), ("critval", F64), ("meddraCode", STR),
    ],
    "openfda_significant_adverse_target_reactions": [
        ("targetId", STR), ("event", STR), _nn("count", I64), ("llr", F64), ("critval", F64), ("meddraCode", STR),
    ],
    "association_by_datasource_direct": [
        ("datatypeId", STR), ("datasourceId", STR), ("diseaseId", STR), ("targetId", STR), ("score", F64),
        _nn("evidenceCount", I64),
    ],
    "association_by_datatype_direct": [
        ("diseaseId", STR), ("targetId", STR), ("datatypeId", STR), ("score", F64), ("evidenceCount", I64),
    ],
    "association_overall_direct": [("diseaseId", STR), ("targetId", STR), ("score", F64), ("evidenceCount", I64)],
    "go": [("id", STR), ("name", STR)],
    "reactome": [
        ("id", STR), ("label", STR), ("ancestors", pa.list_(STR)), ("descendants", pa.list_(STR)),
        ("children", pa.list_(STR)), ("parents", pa.list_(STR)), ("path", pa.list_(pa.list_(STR))),
    ],
    "so": [("id", LSTR), ("label", LSTR)],
    "literature": [
        ("pmid", STR), ("pmcid", STR), ("date", pa.date32()), ("year", I32), ("month", I32), ("day", I32),
        ("keywordId", STR), ("relevance", F64), ("keywordType", STR),
    ],
    "literature_vector": [
        _nn("category", STR), ("word", STR), _nn("norm", F64), _nn("vector", pa.list_(_el(F64))),
    ],
    "drug_mechanism_of_action": [
        ("actionType", STR), ("mechanismOfAction", STR), ("chemblIds", pa.list_(STR)), ("targetName", STR),
        ("targetType", STR), ("targets", pa.list_(_el(STR))),
        _nn("references", pa.list_(_el(pa.struct([("source", STR), _nn("ids", pa.list_(_el(STR))),
                                                  _nn("urls", pa.list_(_el(STR)))])))),
    ],
    "drug_indication": [
        ("id", STR),
        _nn("indications", pa.list_(_el(pa.struct([
            ("disease", STR), ("efoName", STR),
            _nn("references", pa.list_(_el(pa.struct([("source", STR), _nn("ids", pa.list_(_el(STR)))])))),
            ("maxPhaseForIndication", F64)])))),
        _nn("approvedIndications", pa.list_(_el(STR))), _nn("indicationCount", I32),
    ],
    "drug_warning": [
        ("chemblIds", pa.list_(STR)), ("toxicityClass", STR), ("country", STR), ("description", STR), ("id", I64),
        ("references", pa.list_(pa.struct([("ref_id", STR), ("ref_type", STR), ("ref_url", STR)]))),
        ("warningType", STR), ("year", I64), ("efo_term", STR), ("efo_id", STR), ("efo_id_for_warning_class", STR),
    ],
    "disease": [
        ("id", STR), ("code", STR), ("dbXRefs", pa.list_(STR)), ("description", STR), ("name", STR),
        ("directLocationIds", pa.list_(STR)), ("obsoleteTerms", pa.list_(STR)), ("obsoleteXRefs", pa.list_(STR)),
        ("indirectLocationIds", pa.list_(STR)),
        ("synonyms", pa.struct([("hasBroadSynonym", pa.list_(STR)), ("hasExactSynonym", pa.list_(STR)),
                                ("hasNarrowSynonym", pa.list_(STR)), ("hasRelatedSynonym", pa.list_(STR))])),
        ("parents", pa.list_(STR)), ("children", pa.list_(STR)), ("ancestors", pa.list_(STR)),
        ("descendants", pa.list_(STR)), ("therapeuticAreas", pa.list_(STR)),
        ("ontology", pa.struct([("isTherapeuticArea", BOOL), ("leaf", BOOL), ("name", STR),
                                ("sources", pa.struct([("name", STR), ("url", STR)]))])),
    ],
    "disease_hpo": [
        ("id", STR), ("code", STR), ("dbXRefs", pa.list_(STR)), ("description", STR), ("name", STR),
        ("namespace", pa.list_(STR)), ("obsoleteTerms", pa.list_(STR)), ("parents", pa.list_(STR)),
    ],
    "disease_phenotype": [
        ("disease", STR), ("phenotype", STR),
        ("evidence", pa.list_(pa.struct([
            ("aspect", STR), ("bioCuration", STR), ("diseaseFromSource", STR), ("diseaseFromSourceId", STR),
            ("diseaseName", STR), ("evidenceType", STR), ("frequency", STR), ("modifiers", pa.list_(STR)),
            ("onset", pa.list_(STR)), ("qualifier", STR), ("qualifierNot", BOOL), ("references", pa.list_(STR)),
            ("sex", STR), ("resource", STR)]))),
    ],
    "mouse_phenotype": [
        ("biologicalModels", pa.list_(pa.struct([("allelicComposition", STR), ("geneticBackground", STR),
                                                 ("id", STR), ("literature", pa.list_(STR))]))),
        ("modelPhenotypeClasses", pa.list_(pa.struct([("id", STR), ("label", STR)]))),
        ("modelPhenotypeId", STR), ("modelPhenotypeLabel", STR), ("targetFromSourceId", STR),
        ("targetInModel", STR), ("targetInModelEnsemblId", STR), ("targetInModelMgiId", STR),
    ],
    "target_prioritisation": [
        ("targetId", STR), ("isInMembrane", F64), ("isSecreted", F64), ("hasSafetyEvent", F64), ("hasPocket", F64),
        ("hasLigand", F64), ("hasSmallMoleculeBinder", F64), ("geneticConstraint", F64),
        ("paralogMaxIdentityPercentage", F64), ("mouseOrthologMaxIdentityPercentage", F64),
        ("isCancerDriverGene", F64), ("hasTEP", F64), ("mouseKOScore", F64), ("hasHighQualityChemicalProbes", F64),
        ("maxClinicalTrialPhase", F64), ("tissueSpecificity", F64), ("tissueDistribution", F64),
    ],
    "l2g_prediction": [
        ("studyLocusId", STR), ("geneId", STR), ("score", F64),
        ("features", pa.list_(pa.struct([("name", STR), ("value", F64), ("shapValue", F64)]))),
        ("shapBaseValue", F64),
    ],
    "interaction": [
        ("sourceDatabase", STR), ("targetA", STR), ("intA", STR), ("intABiologicalRole", STR), ("targetB", STR),
        ("intB", STR), ("intBBiologicalRole", STR),
        ("speciesA", pa.struct([("mnemonic", STR), ("scientificName", STR), ("taxonId", I64)])),
        ("speciesB", pa.struct([("mnemonic", STR), ("scientificName", STR), ("taxonId", I64)])),
        ("count", I64), ("scoring", F64), ("intASource", STR), ("intBSource", STR),
    ],
    "study": [
        ("studyId", STR), ("geneId", STR), ("projectId", STR), ("studyType", STR), ("traitFromSource", STR),
        ("traitFromSourceMappedIds", pa.list_(STR)), ("biosampleFromSourceId", STR), ("pubmedId", STR),
        ("publicationTitle", STR), ("publicationFirstAuthor", STR), ("publicationDate", STR),
        ("publicationJournal", STR), ("backgroundTraitFromSourceMappedIds", pa.list_(STR)),
        ("initialSampleSize", STR), ("nCases", I32), ("nControls", I32), ("nSamples", I32),
        ("cohorts", pa.list_(STR)),
        ("ldPopulationStructure", pa.list_(pa.struct([("ldPopulation", STR), ("relativeSampleSize", F64)]))),
        ("discoverySamples", pa.list_(pa.struct([("sampleSize", I32), ("ancestry", STR)]))),
        ("replicationSamples", pa.list_(pa.struct([("sampleSize", I32), ("ancestry", STR)]))),
        ("qualityControls", pa.list_(STR)), ("analysisFlags", pa.list_(STR)), ("summarystatsLocation", STR),
        ("hasSumstats", BOOL), ("condition", STR), ("diseaseIds", pa.list_(STR)),
        ("backgroundDiseaseIds", pa.list_(STR)), ("biosampleId", STR),
    ],
    "biosample": [
        ("biosampleId", STR), ("biosampleName", STR), ("description", STR), ("xrefs", pa.list_(STR)),
        ("synonyms", pa.list_(STR)), ("parents", pa.list_(STR)), ("ancestors", pa.list_(STR)),
        ("children", pa.list_(STR)), ("descendants", pa.list_(STR)),
    ],
    "colocalisation_coloc": [
        ("leftStudyLocusId", STR), ("rightStudyLocusId", STR), ("chromosome", STR), ("rightStudyType", STR),
        ("numberColocalisingVariants", I64), ("h0", F64), ("h1", F64), ("h2", F64), ("h3", F64), ("h4", F64),
        ("colocalisationMethod", STR), ("betaRatioSignAverage", F64),
    ],
    "colocalisation_ecaviar": [
        ("leftStudyLocusId", STR), ("rightStudyLocusId", STR), ("chromosome", STR), ("rightStudyType", STR),
        ("numberColocalisingVariants", I64), ("clpp", F64), ("colocalisationMethod", STR),
        ("betaRatioSignAverage", F64),
    ],
    "expression": [
        ("id", STR),
        ("tissues", pa.list_(pa.struct([
            ("efo_code", STR), ("label", STR), ("organs", pa.list_(STR)), ("anatomical_systems", pa.list_(STR)),
            ("rna", pa.struct([("value", F64), ("zscore", I64), ("level", I64), ("unit", STR)])),
            ("protein", pa.struct([("reliability", BOOL), ("level", I64)]))]))),
    ],
    "evidence": [   # all 90 columns of 25.09 (hive-partitioned by sourceId, which is not stored in the files)
        ("id", STR), ("datasourceId", STR), ("targetId", STR), ("alleleOrigins", pa.list_(STR)),
        ("allelicRequirements", pa.list_(STR)), ("ancestry", STR), ("ancestryId", STR), ("beta", F64),
        ("betaConfidenceIntervalLower", F64), ("betaConfidenceIntervalUpper", F64),
        ("biologicalModelAllelicComposition", STR), ("biologicalModelGeneticBackground", STR),
        ("biologicalModelId", STR), ("biomarkerName", STR),
        ("biomarkers", pa.struct([("geneExpression", pa.list_(pa.struct([("id", STR), ("name", STR)]))), ("geneticVariation", pa.list_(pa.struct([("functionalConsequenceId", STR), ("id", STR), ("name", STR)])))])),
        ("biosamplesFromSource", pa.list_(STR)), ("cellType", STR), ("clinicalPhase", F64),
        ("clinicalSignificances", pa.list_(STR)), ("clinicalStatus", STR), ("cohortDescription", STR),
        ("cohortId", STR), ("cohortPhenotypes", pa.list_(STR)), ("cohortShortName", STR), ("confidence", STR),
        ("contrast", STR), ("crisprScreenLibrary", STR), ("datatypeId", STR),
        ("diseaseCellLines", pa.list_(pa.struct([("id", STR), ("name", STR), ("tissue", STR), ("tissueId", STR)]))),
        ("diseaseFromSource", STR), ("diseaseFromSourceId", STR), ("diseaseFromSourceMappedId", STR),
        ("diseaseModelAssociatedHumanPhenotypes", pa.list_(pa.struct([("id", STR), ("label", STR)]))),
        ("diseaseModelAssociatedModelPhenotypes", pa.list_(pa.struct([("id", STR), ("label", STR)]))),
        ("drugFromSource", STR), ("drugId", STR), ("drugResponse", STR), ("geneticBackground", STR),
        ("literature", pa.list_(STR)), ("log2FoldChangePercentileRank", I64), ("log2FoldChangeValue", F64),
        ("mutatedSamples", pa.list_(pa.struct([("functionalConsequenceId", STR), ("numberMutatedSamples", F64), ("numberSamplesTested", F64), ("numberSamplesWithMutationType", I64)]))),
        ("oddsRatio", F64), ("oddsRatioConfidenceIntervalLower", F64), ("oddsRatioConfidenceIntervalUpper", F64),
        ("pValueExponent", I64), ("pValueMantissa", F64),
        ("pathways", pa.list_(pa.struct([("id", STR), ("name", STR)]))), ("projectId", STR), ("reactionId", STR),
        ("reactionName", STR), ("releaseDate", STR), ("releaseVersion", STR), ("resourceScore", F64),
        ("sex", pa.list_(STR)), ("significantDriverMethods", pa.list_(STR)), ("statisticalMethod", STR),
        ("statisticalMethodOverview", STR), ("statisticalTestTail", STR), ("studyCases", I64),
        ("studyCasesWithQualifyingVariants", I64), ("studyId", STR), ("studyOverview", STR),
        ("studySampleSize", I64), ("studyStartDate", STR), ("studyStopReason", STR),
        ("studyStopReasonCategories", pa.list_(STR)), ("targetFromSource", STR), ("targetFromSourceId", STR),
        ("targetInModel", STR), ("targetInModelEnsemblId", STR), ("targetInModelMgiId", STR),
        ("targetModulation", STR), ("urls", pa.list_(pa.struct([("niceName", STR), ("url", STR)]))),
        ("variantAminoacidDescriptions", pa.list_(STR)), ("variantFromSourceId", STR),
        ("variantFunctionalConsequenceId", STR), ("variantHgvsId", STR), ("variantId", STR), ("variantRsId", STR),
        ("pmcIds", pa.list_(STR)), ("publicationYear", I64), ("studyLocusId", STR),
        ("textMiningSentences", pa.list_(pa.struct([("dEnd", I64), ("dStart", I64), ("section", STR), ("tEnd", I64), ("tStart", I64), ("text", STR)]))),
        ("diseaseId", STR), ("score", F64), ("publicationDate", STR), ("evidenceDate", STR), ("variantEffect", STR),
        ("directionOnTrait", STR),
    ],
}
_SCREEN = pa.struct([("depmapId", STR), ("cellLineName", STR), ("diseaseFromSource", STR),
                     ("diseaseCellLineId", STR), ("expression", F64), ("geneEffect", F64), ("mutation", STR)])
_RESOURCES = pa.struct([("sourceDatabase", STR), ("databaseVersion", STR)])
SCHEMAS.update({
    "target_essentiality": [
        ("id", STR),
        ("geneEssentiality", pa.list_(pa.struct([
            ("isEssential", BOOL),
            ("depMapEssentiality", pa.list_(pa.struct([("tissueId", STR), ("tissueName", STR),
                                                       ("screens", pa.list_(_SCREEN))])))]))),
    ],
    "credible_set": [
        ("studyLocusId", STR), ("studyId", STR), ("variantId", STR), ("chromosome", STR), ("position", I32),
        ("region", STR), ("beta", F64), ("zScore", F64), ("pValueMantissa", F32), ("pValueExponent", I32),
        ("standardError", F64), ("finemappingMethod", STR), ("credibleSetIndex", I32), ("credibleSetlog10BF", F64),
        ("purityMeanR2", F64), ("purityMinR2", F64), ("locusStart", I32), ("locusEnd", I32), ("sampleSize", I32),
        ("locus", pa.list_(pa.struct([("is95CredibleSet", BOOL), ("is99CredibleSet", BOOL), ("logBF", F64),
                                      ("posteriorProbability", F64), ("variantId", STR), ("pValueMantissa", F32),
                                      ("pValueExponent", I32), ("beta", F64), ("standardError", F64),
                                      ("r2Overall", F64)]))),
        ("confidence", STR), ("studyType", STR), ("qualityControls", pa.list_(STR)),
    ],
    "variant": [
        ("variantId", STR), ("chromosome", STR), ("position", I32), ("referenceAllele", STR),
        ("alternateAllele", STR), ("mostSevereConsequenceId", STR), ("rsIds", pa.list_(STR)), ("hgvsId", STR),
        ("variantDescription", STR),
    ],
    "interval": [
        ("chromosome", STR), ("start", I64), ("end", I64), ("geneId", STR), ("biosampleName", STR),
        ("intervalType", STR), ("distanceToTss", I64), ("score", F64), ("datasourceId", STR), ("datatypeId", STR),
        ("pmid", STR), ("biosampleId", STR), ("studyId", STR), ("intervalId", STR),
    ],
    "interaction_evidence": [
        ("interactionIdentifier", STR), ("interactionResources", _RESOURCES), ("interactionScore", F64),
        ("intA", STR), ("intB", STR), ("targetA", STR), ("targetB", STR), ("pubmedId", STR),
        ("intABiologicalRole", STR), ("intBBiologicalRole", STR), ("interactionDetectionMethodShortName", STR),
    ],
})
SCHEMAS["association_by_datasource_indirect"] = SCHEMAS["association_by_datasource_direct"]
SCHEMAS["association_by_datatype_indirect"] = SCHEMAS["association_by_datatype_direct"]
SCHEMAS["association_by_overall_indirect"] = SCHEMAS["association_overall_direct"]


def schema(name: str) -> pa.Schema:
    fields = [f if isinstance(f, pa.Field) else pa.field(*f) for f in SCHEMAS[name]]
    return pa.schema(fields)


def _default(typ: pa.DataType) -> Any:
    """A value for a non-nullable field a row leaves out."""
    if pa.types.is_list(typ) or pa.types.is_large_list(typ):
        return []
    if pa.types.is_struct(typ):
        return {f.name: (_default(f.type) if not f.nullable else None) for f in typ}
    if pa.types.is_boolean(typ):
        return False
    if pa.types.is_integer(typ) or pa.types.is_floating(typ):
        return 0
    return ""


def table(name: str, rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    """``rows`` as a table with the 25.09 schema of ``name`` (absent nullable fields are null)."""
    sch = schema(name)
    filled = []
    for row in rows:
        unknown = set(row) - set(sch.names)
        if unknown:
            raise KeyError(f"{name}: columns not in the 25.09 schema: {sorted(unknown)}")
        filled.append({f.name: row.get(f.name, None if f.nullable else _default(f.type)) for f in sch})
    return pa.Table.from_pylist(filled, schema=sch)


def _part_name(i: int) -> str:
    return f"part-{i:05d}-fixture-c000.snappy.parquet"


def write_table(root: Path, name: str, tbl: pa.Table, *, shards: int = 1, subdir: str | None = None) -> list[Path]:
    """Write ``tbl`` as ``shards`` Spark-style files under ``root/name[/subdir]`` in row order."""
    d = root / name / subdir if subdir else root / name
    d.mkdir(parents=True, exist_ok=True)
    out = []
    step = max(1, math.ceil(tbl.num_rows / max(shards, 1)))
    for i in range(max(shards, 1)):
        path = d / _part_name(i)
        pq.write_table(tbl.slice(i * step, step), path, compression="snappy")
        out.append(path)
    return out


# ---------------------------------------------------------------------------- Open Targets rows


def _target_rows() -> list[dict[str, Any]]:
    loc = lambda chrom, start: {"chromosome": chrom, "start": start, "end": start + 1000, "strand": 1}  # noqa: E731
    go_pcsk9 = [
        {"id": "GO:0006629", "source": "UniProt", "evidence": "IDA", "aspect": "P", "geneProduct": "Q8NBP7",
         "ecoId": "ECO_0000314"},
        {"id": "GO:1000001", "source": "GO_Central", "evidence": "IEA", "aspect": "F", "geneProduct": "Q8NBP7",
         "ecoId": "ECO_0000501"},
    ]
    probe = {"control": None, "drugId": "CHEMBL4297185", "id": "PROBE-1", "isHighQuality": True,
             "mechanismOfAction": ["inhibitor"], "origin": ["experimental"], "probeMinerScore": 60,
             "probesDrugsScore": 70, "scoreInCells": 50, "scoreInOrganisms": 10, "targetFromSourceId": "Q8NBP7",
             "urls": [{"niceName": "Probes&Drugs", "url": "https://www.probes-drugs.org/"}]}
    pathway = {"pathwayId": "R-HSA-8964043", "pathway": "Plasma lipoprotein clearance",
               "topLevelTerm": "Transport of small molecules"}
    hallmark = {"pmid": 1, "description": "promotes proliferation", "impact": "promotes",
                "label": "proliferative signalling"}
    tract = [{"modality": "SM", "id": "Approved Drug", "value": False},
             {"modality": "AB", "id": "Approved Drug", "value": True}]
    return [
        # TP53BP1 and TP53I3 come before TP53 in file order (and by key): a substring scan in file
        # order answers "TP53" with TP53BP1.
        {"id": TP53BP1, "approvedSymbol": "TP53BP1", "approvedName": "tumor protein p53 binding protein 1",
         "biotype": "protein_coding", "genomicLocation": loc("15", 43403061),
         "symbolSynonyms": [{"label": "53BP1", "source": "HGNC"}], "go": [], "pathways": [],
         "chemicalProbes": None, "tractability": [], "functionDescriptions": ["DNA damage response"]},
        {"id": TP53I3, "approvedSymbol": "TP53I3", "approvedName": "tumor protein p53 inducible protein 3",
         "biotype": "protein_coding", "genomicLocation": loc("2", 24076568), "symbolSynonyms": [],
         "go": [], "pathways": [], "chemicalProbes": [], "tractability": []},
        {"id": TP53, "approvedSymbol": "TP53", "approvedName": "tumor protein p53", "biotype": "protein_coding",
         "genomicLocation": loc("17", 7661779), "symbolSynonyms": [{"label": "LFS1", "source": "HGNC"}],
         "go": [], "pathways": [], "chemicalProbes": [],
         "hallmarks": {"attributes": [], "cancerHallmarks": []}, "tractability": tract,
         "functionDescriptions": ["Acts as a tumor suppressor"]},
        {"id": PCSK9, "approvedSymbol": "PCSK9", "approvedName": "proprotein convertase subtilisin/kexin type 9",
         "biotype": "protein_coding", "genomicLocation": loc("1", 55039548),
         "symbolSynonyms": [{"label": "NARC1", "source": "HGNC"}],
         "obsoleteSymbols": [{"label": "HCHOLA3", "source": "HGNC"}],
         "go": go_pcsk9, "pathways": [pathway], "chemicalProbes": [probe],
         "hallmarks": {"attributes": [], "cancerHallmarks": [hallmark]}, "tractability": tract,
         "functionDescriptions": ["Regulates LDL receptor degradation"], "safetyLiabilities": [],
         "homologues": [{"speciesId": "10090", "speciesName": "Mouse", "homologyType": "ortholog_one2one",
                         "targetGeneId": PCSK9_MOUSE, "isHighConfidence": "1", "targetGeneSymbol": "Pcsk9",
                         "queryPercentageIdentity": 77.0, "targetPercentageIdentity": 78.0, "priority": 1}]},
        # NDE1 carries the declared codes no other row shows (readiness R6 confirms every declared
        # code from the data): GO aspect C, a low-confidence homologue and the three constraint types.
        {"id": G_L2G, "approvedSymbol": "NDE1", "approvedName": "nudE neurodevelopment protein 1",
         "biotype": "protein_coding", "genomicLocation": loc("16", 15643267),
         "go": [{"id": "GO:0005813", "source": "UniProt", "evidence": "IDA", "aspect": "C", "geneProduct": "Q9NXR1",
                 "ecoId": "ECO_0000314"}],
         "pathways": [], "chemicalProbes": [], "tractability": [],
         "homologues": [{"speciesId": "7955", "speciesName": "Zebrafish", "homologyType": "ortholog_one2many",
                         "targetGeneId": "ENSDARG00000001234", "isHighConfidence": "0", "targetGeneSymbol": "nde1",
                         "queryPercentageIdentity": 55.0, "targetPercentageIdentity": 54.0, "priority": 2}],
         "constraint": [{"constraintType": t, "score": sc, "exp": 10.0, "obs": 8, "oe": 0.8, "oeLower": 0.5,
                         "oeUpper": 1.2, "upperRank": 100, "upperBin": 3, "upperBin6": 2}
                        for t, sc in (("syn", 0.1), ("mis", 0.4), ("lof", 0.9))]},
    ]


def _disease_rows() -> list[dict[str, Any]]:
    def syn(exact=(), related=()):
        return {"hasBroadSynonym": [], "hasExactSynonym": list(exact), "hasNarrowSynonym": [],
                "hasRelatedSynonym": list(related)}

    def onto(leaf=True, ta=False):
        return {"isTherapeuticArea": ta, "leaf": leaf, "name": None,
                "sources": {"name": "MONDO", "url": "http://purl.obolibrary.org/obo/MONDO"}}

    rows = [
        {"id": "EFO_0000651", "code": "http://www.ebi.ac.uk/efo/EFO_0000651", "name": "endocrine system disease",
         "description": "therapeutic area", "synonyms": syn(), "parents": [], "children": [T2D], "ancestors": [],
         "descendants": [T2D], "therapeuticAreas": ["EFO_0000651"], "ontology": onto(leaf=False, ta=True)},
        {"id": T2D, "code": "http://purl.obolibrary.org/obo/MONDO_0005148", "name": T2D_NAME,
         "description": "A type of diabetes mellitus", "synonyms": syn(exact=["NIDDM"], related=["T2DM"]),
         "obsoleteTerms": [T2D_RETIRED], "parents": ["EFO_0000651"], "children": [], "ancestors": ["EFO_0000651"],
         "descendants": [], "therapeuticAreas": ["EFO_0000651"], "ontology": onto()},
        {"id": RA, "code": "http://www.ebi.ac.uk/efo/EFO_0000685", "name": "rheumatoid arthritis",
         "description": "An autoimmune disease", "synonyms": syn(exact=["RA"]), "parents": [], "children": [],
         "ancestors": [], "descendants": [], "therapeuticAreas": ["EFO_0000540"], "ontology": onto()},
    ]
    for label, did in PHENO.items():
        rows.append({"id": did, "code": f"http://purl.obolibrary.org/obo/{did}", "name": f"seizure disorder {label}",
                     "description": "fixture", "synonyms": syn(), "parents": [], "children": [], "ancestors": [],
                     "descendants": [], "therapeuticAreas": ["EFO_0000618"], "ontology": onto()})
    return rows


def _pheno_ev(evidence_type: str, negated: bool, disease_label: str) -> dict[str, Any]:
    return {"aspect": "P", "bioCuration": "HPO:skoehler[2009-02-17]", "diseaseFromSource": disease_label,
            "diseaseFromSourceId": f"OMIM:{600000 + ord(disease_label[0])}",
            "diseaseName": f"seizure disorder {disease_label}", "evidenceType": evidence_type, "frequency": None,
            "modifiers": [], "onset": [], "qualifier": "NOT" if negated else None, "qualifierNot": negated,
            "references": ["PMID:1"], "sex": None, "resource": "HPO"}


def _disease_phenotype_rows() -> list[dict[str, Any]]:
    spec = {"X": [("PCS", True)], "Y": [("IEA", False)], "Z": [("PCS", False), ("PCS", True), ("IEA", False)],
            "W": [("IEA", True)]}
    rows = [{"disease": PHENO[k], "phenotype": SEIZURE, "evidence": [_pheno_ev(t, n, k) for t, n in items]}
            for k, items in spec.items()]
    rows.append({"disease": T2D, "phenotype": "HP_0000822", "evidence": [_pheno_ev("IEA", False, "T2D")]})
    return rows


def _drug_rows(known_drug_ids: Iterable[str]) -> list[dict[str, Any]]:
    lt = lambda *ts: {"rows": list(ts), "count": len(ts)}  # noqa: E731
    named = [
        {"id": CHEMBL3, "name": "NICOTINE", "drugType": "Small molecule", "maximumClinicalTrialPhase": 4.0,
         "linkedTargets": lt(), "isApproved": True},
        {"id": CHEMBL25, "name": "ASPIRIN", "drugType": "Small molecule", "maximumClinicalTrialPhase": 4.0,
         "linkedTargets": lt(TP53I3), "isApproved": True, "synonyms": ["acetylsalicylic acid"],
         "tradeNames": ["Aspirin"]},
        {"id": CHEMBL1000, "name": "CETIRIZINE", "drugType": "Small molecule", "maximumClinicalTrialPhase": 4.0,
         "linkedTargets": lt(PCSK9), "isApproved": True},
        {"id": CHEMBL559288, "name": "FIXTURE DRUG 559288", "drugType": "Small molecule",
         "maximumClinicalTrialPhase": 2.0, "linkedTargets": lt(), "isApproved": False},
    ]
    seen = {r["id"] for r in named}
    for did in sorted(set(known_drug_ids) - seen):
        named.append({"id": did, "name": f"KNOWN {did}", "drugType": "Antibody", "maximumClinicalTrialPhase": 3.0,
                      "linkedTargets": lt(), "isApproved": False})
    for r in named:
        r.setdefault("linkedDiseases", {"rows": [], "count": 0})
    return named


def _known_drug_rows() -> list[dict[str, Any]]:
    """CT-4: for T, 30 rows of phases 1-3 first in file order, then 5 phase-4 rows, then a null-phase
    and a NaN-phase row; ``status`` null on 7 rows; ``urls`` lists in varying order."""
    a = {"niceName": "ClinicalTrials", "url": "https://clinicaltrials.gov/search?id=%22NCT0000001%22"}
    b = {"niceName": "FDA", "url": "https://www.accessdata.fda.gov/"}
    rows: list[dict[str, Any]] = []
    for i in range(30):
        rows.append({"drugId": f"CHEMBL40{i % 8:02d}", "targetId": T, "diseaseId": f"EFO_00{10000 + i:05d}",
                     "phase": float(1 + i % 3), "status": None if i in (4, 11) else "Completed",
                     "urls": [a, b] if i % 2 else [b, a]})
    rows[2] = dict(KNOWN_DRUG_SENTINEL, urls=[a])    # a phase-3 row with null status (descriptor sentinel)
    for i in range(5):
        rows.append({"drugId": f"CHEMBL50{i % 3:02d}", "targetId": T, "diseaseId": f"EFO_00{20000 + i:05d}",
                     "phase": 4.0, "status": None if i < 2 else "Recruiting", "urls": [b] if i % 2 else [a, b]})
    rows.append({"drugId": "CHEMBL6000", "targetId": T, "diseaseId": "EFO_0030000", "phase": None, "status": None,
                 "urls": []})
    rows.append({"drugId": "CHEMBL6001", "targetId": T, "diseaseId": "EFO_0030001", "phase": float("nan"),
                 "status": None, "urls": []})
    rows.append({"drugId": "CHEMBL7000", "targetId": TP53, "diseaseId": "EFO_0000311", "phase": 4.0,
                 "status": "Completed", "urls": [a]})
    for r in rows:
        r.setdefault("prefName", f"drug {r['drugId']}")
        r.setdefault("approvedSymbol", "PCSK9" if r["targetId"] == T else "TP53")
        r.setdefault("label", f"disease {r['diseaseId']}")
        r.setdefault("drugType", "Antibody")
    return rows


def _pgx_rows() -> list[dict[str, Any]]:
    """CT-3/CT-5: (T, [CHEMBL3]), (T, [CHEMBL3, CHEMBL25]), (T2, [CHEMBL3]), (T, [CHEMBL25]), (T2, [CHEMBL99])."""
    spec = [(T, [CHEMBL3]), (T, [CHEMBL3, CHEMBL25]), (T2, [CHEMBL3]), (T, [CHEMBL25]), (T2, [CHEMBL99])]
    names = {CHEMBL3: "nicotine", CHEMBL25: "aspirin", CHEMBL99: "fixture-99"}
    return [{"datasourceId": "pharmgkb", "datasourceVersion": "2025-08", "datatypeId": "clinical_annotation",
             "evidenceLevel": "3", "genotype": "AA", "genotypeId": f"1_100{i}_A_A,A", "literature": [str(1000 + i)],
             "pgxCategory": "efficacy", "phenotypeText": f"response {i}", "studyId": f"PA{1000 + i}",
             "targetFromSourceId": tgt, "variantRsId": f"rs{i + 1}", "variantId": f"1_100{i}_A_G",
             "isDirectTarget": True, "drugs": [{"drugFromSource": names[d], "drugId": d} for d in drugs]}
            for i, (tgt, drugs) in enumerate(spec)]


def _l2g_rows() -> list[dict[str, Any]]:
    """CT-4: 150 loci for G, the maximum 0.8725 stored last; the first five peak at 0.51."""
    scores = [0.47, 0.49, 0.51, 0.30, 0.10] + [round(0.06 + (i % 40) * 0.01, 4) for i in range(144)] + [0.8725]
    feats = [{"name": "distanceTssMean", "value": 0.9, "shapValue": 0.1}]
    rows = [{"studyLocusId": f"{i:032x}", "geneId": G_L2G, "score": s, "features": feats, "shapBaseValue": 0.12}
            for i, s in enumerate(scores)]
    rows.append({"studyLocusId": f"{999:032x}", "geneId": PCSK9, "score": 0.99, "features": feats,
                 "shapBaseValue": 0.12})
    return rows


def _interaction_rows(sources: Sequence[str]) -> list[dict[str, Any]]:
    """CT-4: T's partners with ``scoring`` ascending in file order, T on side B in half the rows."""
    sp = {"mnemonic": "human", "scientificName": "Homo sapiens", "taxonId": 9606}
    rows = []
    n = 12
    for s_i, src in enumerate(sources):
        for i in range(n):
            partner = f"ENSG0000010{s_i}{i:03d}"
            a_side = i % 2 == 0
            rows.append({"sourceDatabase": src, "targetA": T if a_side else partner, "intA": f"P{s_i}A{i:03d}",
                         "intABiologicalRole": "unspecified role", "targetB": partner if a_side else T,
                         "intB": f"P{s_i}B{i:03d}", "intBBiologicalRole": "unspecified role", "speciesA": sp,
                         "speciesB": sp, "count": 1 + i % 4, "scoring": round(0.15 + 0.07 * i + 0.003 * s_i, 4),
                         "intASource": "uniprotkb", "intBSource": "uniprotkb"})
    rows.append({"sourceDatabase": sources[0], "targetA": TP53, "intA": "PX", "intABiologicalRole": "unspecified role",
                 "targetB": TP53BP1, "intB": "PY", "intBBiologicalRole": "unspecified role", "speciesA": sp,
                 "speciesB": sp, "count": 9, "scoring": 0.999, "intASource": "uniprotkb", "intBSource": "uniprotkb"})
    return rows


def _evidence_rows() -> list[dict[str, Any]]:
    """europepmc literature evidence: two rows citing 30595370 (other targets) and T's rows
    with publicationYear 2024, 2019 and null (CT-6)."""
    base = {"datasourceId": "europepmc", "datatypeId": "literature"}
    rows = [
        dict(base, id="epmc-0001", targetId=TP53, diseaseId=RA, score=0.61, literature=[PMID_EPMC, "29000001"],
             publicationYear=2019),
        dict(base, id="epmc-0002", targetId=TP53BP1, diseaseId=T2D, score=0.42, literature=[PMID_EPMC],
             publicationYear=2019),
        dict(base, id="epmc-0003", targetId=TP53, diseaseId=T2D, score=0.30, literature=["31000001"],
             publicationYear=2020),
        dict(base, id="epmc-2024", targetId=T, diseaseId=T2D, score=0.9, literature=["38000001"],
             publicationYear=2024),
        dict(base, id="epmc-2019", targetId=T, diseaseId=T2D, score=0.8, literature=["31000002"],
             publicationYear=2019),
        dict(base, id="epmc-null", targetId=T, diseaseId=RA, score=0.7, literature=["35000003"],
             publicationYear=None),
    ]
    return rows


def _study_rows() -> list[dict[str, Any]]:
    """CT-6: 25 studies with null nSamples stored before the 200,000-sample ``Sbig``."""
    rows = [{"studyId": f"GCST9{i:07d}0", "studyType": "gwas", "traitFromSource": f"trait {i}", "projectId": "GCST",
             "nSamples": None} for i in range(25)]
    rows.append({"studyId": STUDY_BIG, "studyType": "gwas", "traitFromSource": "LDL cholesterol",
                 "projectId": "GCST", "nSamples": 200000, "nCases": None, "nControls": None})
    rows.append({"studyId": "GCST90000002", "studyType": "gwas", "traitFromSource": "height", "projectId": "GCST",
                 "nSamples": 5000})
    return rows


def _prioritisation_rows(variant: str) -> list[dict[str, Any]]:
    """CT-6 ``hasSafetyEvent``: default A -1, B null, C 0, D NaN; ``no_zero`` uses only {-1, null}
    (the 0 code is never observed); ``refuted`` uses {1, 0, null} (1 is not a declared code)."""
    values = {"default": {"A": -1.0, "B": None, "C": 0.0, "D": float("nan")},
              "no_zero": {"A": -1.0, "B": None, "C": None, "D": -1.0},
              "refuted": {"A": 1.0, "B": None, "C": 0.0, "D": None}}[variant]
    rows = []
    for i, (k, tgt) in enumerate(PRIO.items()):
        # every binary factor shows both codes, as in the release (readiness confirms them)
        rows.append({"targetId": tgt, "hasSafetyEvent": values[k], "isInMembrane": float(i % 2),
                     "isSecreted": 1.0 if k == "A" else 0.0, "hasPocket": 1.0 if i < 2 else 0.0,
                     "hasLigand": 1.0 if i < 3 else 0.0, "hasSmallMoleculeBinder": 1.0 if i == 0 else 0.0,
                     "isCancerDriverGene": float(i % 2), "hasTEP": 1.0 if i == 1 else 0.0,
                     "hasHighQualityChemicalProbes": 1.0 if i == 2 else 0.0,
                     "geneticConstraint": -0.5 + 0.3 * i, "maxClinicalTrialPhase":
                     [1.0, 0.75, 0.25, 0.0][i], "tissueSpecificity": 0.1, "tissueDistribution": -0.2})
    return rows


def _colocalisation_rows(method: str) -> list[dict[str, Any]]:
    common = {"chromosome": "19", "rightStudyType": "eqtl", "numberColocalisingVariants": 3,
              "betaRatioSignAverage": 1.0}
    if method == "coloc":
        return [dict(common, leftStudyLocusId=f"coloc-{i}", rightStudyLocusId=f"r{i}", h0=0.0, h1=0.0, h2=0.02,
                     h3=0.03, h4=h, colocalisationMethod="COLOC") for i, h in enumerate((0.95, 0.9, 0.5))]
    return [dict(common, leftStudyLocusId=f"ecaviar-{i}", rightStudyLocusId=f"r{i}", clpp=c,
                 colocalisationMethod="eCAVIAR") for i, c in enumerate((0.99, 0.85))]


def _association_rows() -> dict[str, list[dict[str, Any]]]:
    """CT-5: direct ⊆ indirect, the indirect top 10 holding 7 ancestor pairs above the direct ones."""
    direct = [{"targetId": T, "diseaseId": f"EFO_00{40000 + i:05d}", "score": round(0.5 - 0.02 * i, 4),
               "evidenceCount": 3} for i in range(10)]
    indirect = [{"targetId": T, "diseaseId": f"EFO_00{50000 + i:05d}", "score": round(0.95 - 0.01 * i, 4),
                 "evidenceCount": 12} for i in range(7)]
    indirect += [dict(r, score=round(r["score"] + 0.05, 4)) for r in direct]
    direct.append({"targetId": TP53, "diseaseId": RA, "score": 0.2, "evidenceCount": 1})
    indirect.append({"targetId": TP53, "diseaseId": RA, "score": 0.2, "evidenceCount": 1})
    by_ds = [
        {"datatypeId": "genetic_association", "datasourceId": "gwas_credible_sets", "diseaseId": T2D, "targetId": T,
         "score": 0.8, "evidenceCount": 4},
        {"datatypeId": "genetic_association", "datasourceId": "eva", "diseaseId": T2D, "targetId": T, "score": 0.6,
         "evidenceCount": 2},
        {"datatypeId": "genetic_association", "datasourceId": "gwas_credible_sets", "diseaseId": RA,
         "targetId": TP53, "score": 0.3, "evidenceCount": 1},
    ]
    by_ds_ind = by_ds + [{"datatypeId": "genetic_association", "datasourceId": "eva", "diseaseId": "EFO_0000651",
                          "targetId": T, "score": 0.65, "evidenceCount": 2}]
    return {"association_overall_direct": direct, "association_by_overall_indirect": indirect,
            "association_by_datasource_direct": by_ds, "association_by_datasource_indirect": by_ds_ind}


def _genetics_rows() -> dict[str, list[dict[str, Any]]]:
    """A few rows each for the DepMap, genetics and interaction-evidence tables no correctness test pins
    (so readiness finds them well-formed rather than drifted placeholders)."""
    screen = lambda effect: {"depmapId": A549, "cellLineName": "A549", "diseaseFromSource": "Lung Cancer",  # noqa: E731
                             "diseaseCellLineId": "CVCL_0023", "expression": 1.5, "geneEffect": effect,
                             "mutation": None}
    essentiality = [
        {"id": gene, "geneEssentiality": [{"isEssential": essential, "depMapEssentiality": [
            {"tissueId": "UBERON_0002048", "tissueName": "lung", "screens": [screen(effect)]}]}]}
        for gene, essential, effect in ((PCSK9, False, -0.1), (TP53, True, -1.2))]
    variant = "1_55039974_G_T"
    return {
        "target_essentiality": essentiality,
        "variant": [{"variantId": variant, "chromosome": "1", "position": 55039974, "referenceAllele": "G",
                     "alternateAllele": "T", "mostSevereConsequenceId": "SO_0001583", "rsIds": ["rs11591147"],
                     "variantDescription": "fixture missense variant in PCSK9"}],
        "credible_set": [{"studyLocusId": "a1b2c3d4e5f60718293a4b5c6d7e8f90", "studyId": STUDY_BIG,
                          "variantId": variant, "chromosome": "1", "position": 55039974, "beta": -0.5,
                          "pValueMantissa": 1.5, "pValueExponent": -120, "finemappingMethod": "SuSie",
                          "credibleSetIndex": 1, "credibleSetlog10BF": 110.0, "studyType": "gwas",
                          "locus": [{"variantId": variant, "posteriorProbability": 0.99, "is95CredibleSet": True,
                                     "is99CredibleSet": True}]}],
        "interval": [{"chromosome": "1", "start": 55030000, "end": 55031000, "geneId": PCSK9, "biosampleName": "liver",
                      "intervalType": "enhancer", "distanceToTss": 9366, "score": 0.8, "datasourceId": "e2g",
                      "datatypeId": "interval", "biosampleId": "UBERON_0002107"}],
        "interaction_evidence": [{"interactionIdentifier": "EBI-0000001", "intA": "P04637", "intB": "Q8NBP7",
                                  "targetA": TP53, "targetB": PCSK9, "interactionScore": 0.4, "pubmedId": "15805190",
                                  "interactionResources": {"sourceDatabase": "intact",
                                                           "databaseVersion": "2025-06"}}],
    }


def ot_rows(*, safety_variant: str = "default", interaction_sources: Sequence[str] = ("intact",)
            ) -> dict[str, list[dict[str, Any]]]:
    """Every non-placeholder table of the OT fixture as rows (file order)."""
    known = _known_drug_rows()
    rows: dict[str, list[dict[str, Any]]] = {
        "target": _target_rows(),
        "disease": _disease_rows(),
        "disease_hpo": [{"id": SEIZURE, "code": "http://purl.obolibrary.org/obo/HP_0001250", "name": "Seizure",
                         "namespace": ["human_phenotype"], "parents": ["HP_0012638"]},
                        {"id": "HP_0000822", "code": "http://purl.obolibrary.org/obo/HP_0000822",
                         "name": "Hypertension", "namespace": ["human_phenotype"], "parents": []}],
        "disease_phenotype": _disease_phenotype_rows(),
        "drug_molecule": _drug_rows(r["drugId"] for r in known),
        "known_drug": known,
        "pharmacogenomics": _pgx_rows(),
        "target_prioritisation": _prioritisation_rows(safety_variant),
        "openfda_significant_adverse_target_reactions": [
            {"targetId": T, "event": "myalgia", "count": 120, "llr": 40.5, "critval": 5.1, "meddraCode": "10028411"},
            {"targetId": T, "event": "nasopharyngitis", "count": 80, "llr": 21.0, "critval": 5.1,
             "meddraCode": "10028810"}],
        "openfda_significant_adverse_drug_reactions": [
            {"chembl_id": CHEMBL559288, "event": f"event {i}", "count": 10 + i, "llr": llr, "critval": 5.0,
             "meddraCode": f"1000{i:04d}"}
            for i, llr in enumerate([10.0 + i for i in range(9)] + [99.5])
        ] + [{"chembl_id": CHEMBL25, "event": "gastric ulcer", "count": 400, "llr": 300.0, "critval": 5.0,
              "meddraCode": "10017822"}],
        "mouse_phenotype": [
            {"targetFromSourceId": PCSK9, "targetInModel": "Pcsk9", "targetInModelEnsemblId": PCSK9_MOUSE,
             "targetInModelMgiId": "MGI:2140260", "modelPhenotypeId": "MP:0000187",
             "modelPhenotypeLabel": "abnormal triglyceride level",
             "modelPhenotypeClasses": [{"id": "MP:0005376", "label": "homeostasis/metabolism phenotype"}],
             "biologicalModels": [{"allelicComposition": "Pcsk9<tm1Jdh>/Pcsk9<tm1Jdh>",
                                   "geneticBackground": "B6", "id": "MGI:3700001", "literature": ["15805190"]}]},
            {"targetFromSourceId": PCSK9, "targetInModel": "Pcsk9", "targetInModelEnsemblId": PCSK9_MOUSE,
             "targetInModelMgiId": "MGI:2140260", "modelPhenotypeId": "MP:0000180",
             "modelPhenotypeLabel": "abnormal circulating cholesterol level", "modelPhenotypeClasses": [],
             "biologicalModels": []},
            {"targetFromSourceId": TP53BP1, "targetInModel": "Trp53bp1", "targetInModelEnsemblId": "ENSMUSG00000043909",
             "targetInModelMgiId": "MGI:1351320", "modelPhenotypeId": "MP:0002169",
             "modelPhenotypeLabel": "no abnormal phenotype detected"},
        ],
        "l2g_prediction": _l2g_rows(),
        "interaction": _interaction_rows(interaction_sources),
        "evidence": _evidence_rows(),
        "study": _study_rows(),
        "go": [{"id": "GO:0006629", "name": "lipid metabolic process"},
               {"id": "GO:1000001", "name": "fixture GO term"},
               {"id": "GO:0008150", "name": "biological_process"}],
        "reactome": [{"id": "R-HSA-8964043", "label": "Plasma lipoprotein clearance", "ancestors": ["R-HSA-382551"],
                      "descendants": [], "children": [], "parents": ["R-HSA-382551"],
                      "path": [["R-HSA-382551", "R-HSA-8964043"]]},
                     {"id": "R-HSA-382551", "label": "Transport of small molecules", "ancestors": [],
                      "descendants": ["R-HSA-8964043"], "children": ["R-HSA-8964043"], "parents": [], "path": []}],
        "so": [{"id": "SO_0001583", "label": "missense_variant"}, {"id": "SO_0001587", "label": "stop_gained"}],
        "biosample": [
            {"biosampleId": "UBERON_0002048", "biosampleName": "lung", "description": "respiration organ",
             "synonyms": [BIOSAMPLE_SYNONYM, "respiratory organ"], "parents": [], "ancestors": [], "children": [],
             "descendants": [], "xrefs": []},
            {"biosampleId": "UBERON_0000948", "biosampleName": "heart", "description": "pump",
             "synonyms": ["cor"], "parents": [], "ancestors": [], "children": [], "descendants": [], "xrefs": []}],
        "colocalisation_coloc": _colocalisation_rows("coloc"),
        "colocalisation_ecaviar": _colocalisation_rows("ecaviar"),
        "expression": [{"id": PCSK9, "tissues": [
            {"efo_code": "UBERON_0002107", "label": "liver", "organs": ["liver"], "anatomical_systems": [],
             "rna": {"value": 120.0, "zscore": 4, "level": 3, "unit": "TPM"}, "protein": None}]}],
    }
    rows.update(_association_rows())
    rows.update(_genetics_rows())
    _close_references(rows)
    return rows


def _close_references(rows: dict[str, list[dict[str, Any]]]) -> None:
    """Add minimal ``disease`` and ``target`` rows for every id the other tables reference, as the
    release has them (readiness R9 samples references with full integrity); retired ids stay absent."""
    retired = {t for r in rows["disease"] for t in r.get("obsoleteTerms") or []}
    have = {r["id"] for r in rows["disease"]}
    refs = {r["diseaseId"] for name in ("association_overall_direct", "association_by_overall_indirect",
                                        "association_by_datasource_direct", "association_by_datasource_indirect",
                                        "known_drug") for r in rows[name]}
    refs |= {ta for r in rows["disease"] for ta in r.get("therapeuticAreas") or []}
    for did in sorted(refs - have - retired):
        rows["disease"].append({"id": did, "code": f"http://www.ebi.ac.uk/efo/{did}", "name": f"fixture disease {did}",
                                "description": "fixture", "parents": [], "children": [], "ancestors": [],
                                "descendants": [], "therapeuticAreas": [],
                                "synonyms": {"hasBroadSynonym": [], "hasExactSynonym": [], "hasNarrowSynonym": [],
                                             "hasRelatedSynonym": []},
                                "ontology": {"isTherapeuticArea": False, "leaf": True, "name": None,
                                             "sources": {"name": "EFO", "url": "http://www.ebi.ac.uk/efo"}}})
    genes = {r["id"] for r in rows["target"]}
    partners = {r[side] for r in rows["interaction"] for side in ("targetA", "targetB") if r.get(side)}
    for gid in sorted(partners - genes):
        rows["target"].append({"id": gid, "approvedSymbol": f"FX{gid[-5:]}", "approvedName": f"fixture gene {gid}",
                               "biotype": "protein_coding", "genomicLocation": {"chromosome": "3", "start": int(gid[-6:]),
                                                                                "end": int(gid[-6:]) + 1000, "strand": 1},
                               "go": [], "pathways": [], "chemicalProbes": [], "tractability": []})


TABLE_SHARDS = {"known_drug": 2, "l2g_prediction": 2, "target": 2, "interaction": 2}


def build_ot_fixture(root: str | Path, *, drop: Iterable[str] = (), with_part_file: str | None = None,
                     manifest: bool = True, safety_variant: str = "default",
                     interaction_sources: Sequence[str] = ("intact",)) -> Path:
    """Write the OT 25.09 fixture under ``root`` and return it.

    ``drop`` leaves those dataset directories out entirely; ``with_part_file`` adds a stray
    ``<relpath>`` partial download (doctor and readiness must notice it); ``manifest=False``
    omits ``.download-manifest.json``. ``safety_variant`` selects the ``hasSafetyEvent`` values
    (``default``, ``no_zero``, ``refuted``); ``interaction_sources`` the ``sourceDatabase`` values.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dropped = set(drop)
    data = ot_rows(safety_variant=safety_variant, interaction_sources=interaction_sources)
    for name in OPEN_TARGETS_DATASETS:
        if name in dropped:
            continue
        if name == "evidence":
            # every declared partition exists; europepmc and chembl hold rows, the rest are empty shards
            parts = {"europepmc": data[name],
                     "chembl": [{"id": "chembl-0001", "datasourceId": "chembl", "datatypeId": "known_drug",
                                 "targetId": TP53, "diseaseId": "EFO_0000311", "score": 0.99, "literature": [],
                                 "clinicalPhase": 4.0, "publicationYear": None}]}
            for source in EVIDENCE_SOURCES:
                write_table(root, name, table(name, parts.get(source, [])), subdir=f"sourceId={source}")
        elif name in data:
            write_table(root, name, table(name, data[name]), shards=TABLE_SHARDS.get(name, 1))
        elif name in SCHEMAS:
            write_table(root, name, table(name, []))
        else:   # placeholder: an empty but valid shard so the legacy inventory check passes
            write_table(root, name, pa.table({"id": pa.array([], STR)}))
    if manifest:
        write_manifest(root)
    if with_part_file:
        part = root / with_part_file
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"PAR1 partial")
    return root


def write_manifest(root: Path, *, complete: bool = True) -> dict[str, Any]:
    """``.download-manifest.json`` as ``tools/download_open_targets.py`` writes it (doctor checks
    release, base, complete, expected_files and byte sizes)."""
    entries = {}
    for path in sorted(root.rglob("*.parquet")):
        data = path.read_bytes()
        entries[path.relative_to(root).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"release": RELEASE, "base": BASE, "expected_files": len(entries), "complete": complete,
                "files": entries}
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def copy_fixture(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def delete_table(root: Path, name: str) -> None:
    """Remove a dataset directory, leaving the manifest (which still lists its files) in place."""
    shutil.rmtree(root / name)


# ---------------------------------------------------------------------------- Tahoe

TAHOE_DE_SCHEMA = pa.schema([
    ("gene_name", STR), ("baseMean", F64), ("log2FoldChange", F64), ("lfcSE", F64), ("stat", F64), ("pvalue", F64),
    ("padj", F64), ("plate", STR), ("n_cells_trt", I64), ("n_cells_ctrl", I64), ("Cell_ID_Cellosaur", STR),
    ("Cell_ID_DepMap", STR), ("drug", STR), ("concentration", F32), ("concentration_unit", STR),
    ("Cell_Name_Vevo", STR),
])

TAHOE_GENES = {SELECTIVE: "ENSG00000999001", "GENE2": "ENSG00000999002", "GENE3": "ENSG00000999003",
               "GENE4": "ENSG00000999004"}
_CELLS = {A549: ("A549", "CVCL_0023", "Lung"), OTHER_LINE: ("NIH:OVCAR-3", "CVCL_0465", "Ovary")}


def tahoe_rows() -> list[dict[str, Any]]:
    """Bortezomib in ACH-000681 and one other line at 0.05, 0.5 and 5.0 uM on plate '1', plus the
    5.0 uM dose in ACH-000681 again on plate '2'; SELECTIVE1 is significant at every dose in
    ACH-000681; an 'Erdafitinib ' (trailing space) block that the metadata spells 'Erdafitinib'."""
    rows = []

    def add(drug, line, conc, plate, gene, lfc, padj):
        name, cvcl, _ = _CELLS[line]
        rows.append({"gene_name": gene, "baseMean": 100.0 + len(rows), "log2FoldChange": lfc, "lfcSE": 0.2,
                     "stat": lfc / 0.2, "pvalue": padj / 10, "padj": padj, "plate": plate, "n_cells_trt": 50,
                     "n_cells_ctrl": 400, "Cell_ID_Cellosaur": cvcl, "Cell_ID_DepMap": line, "drug": drug,
                     "concentration": conc, "concentration_unit": "uM", "Cell_Name_Vevo": name})

    for line in (A549, OTHER_LINE):
        for c_i, conc in enumerate(CONCENTRATIONS):
            if line == A549:
                add(BORTEZOMIB, line, conc, "1", SELECTIVE, 2.0 + c_i, 0.001 * (c_i + 1))
            add(BORTEZOMIB, line, conc, "1", "GENE2", -1.0 - c_i, 0.02)
            if c_i == 1:
                add(BORTEZOMIB, line, conc, "1", "GENE3", 0.7, 0.05)
    add(BORTEZOMIB, A549, TWO_PLATE_DOSE, "2", SELECTIVE, 3.5, 0.002)
    add(BORTEZOMIB, A549, TWO_PLATE_DOSE, "2", "GENE4", -0.9, 0.04)
    add("Erdafitinib ", A549, 0.5, "1", "GENE2", 1.1, 0.03)
    return rows


def build_tahoe_fixture(root: str | Path) -> Path:
    """A prepared Tahoe directory (``tools/prepare_tahoe.py`` output layout) under ``root``."""
    root = Path(root)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    de = pa.Table.from_pylist(tahoe_rows(), schema=TAHOE_DE_SCHEMA)
    pq.write_table(de, root / TAHOE_DE, compression="zstd")
    sig = de.filter(pc.less(de["padj"], 0.05))
    high = sig.filter(pc.greater(pc.abs(sig["log2FoldChange"]), 0.5))
    for directory, tbl in (("pseudobulk_de_significant", sig), ("pseudobulk_de_high_quality", high)):
        (root / directory).mkdir(exist_ok=True)
        pq.write_table(tbl, root / directory / "part-00000.parquet", compression="zstd")
    meta = root / "metadata"
    pq.write_table(pa.Table.from_pylist([
        {"drug": BORTEZOMIB, "targets": "PSMB5", "moa-broad": "inhibitor/antagonist",
         "moa-fine": "Proteasome inhibitor", "human-approved": "yes", "clinical-trials": "yes",
         "gpt-notes-approval": "Approved for multiple myeloma.", "canonical_smiles": "B(O)O", "pubchem_cid": 387447.0},
        {"drug": "Erdafitinib", "targets": "FGFR1|FGFR2|FGFR3|FGFR4", "moa-broad": "inhibitor/antagonist",
         "moa-fine": "FGFR inhibitor", "human-approved": "yes", "clinical-trials": "yes",
         "gpt-notes-approval": "Approved for urothelial carcinoma.", "canonical_smiles": "C", "pubchem_cid": 67462786.0},
    ]), meta / "drug_metadata.parquet")
    cell_rows = []
    for line, (name, cvcl, organ) in _CELLS.items():
        drivers = [("KRAS", "Hom", "Missense", "p.G12S"), ("CDKN2A", "Hom", "Deletion", None),
                   ("STK11", "Hom", "Nonsense", "p.Q37*")] if line == A549 else [("TP53", "Hom", "Missense", "p.R248Q")]
        for gene, zyg, vtype, effect in drivers:
            cell_rows.append({"cell_name": name, "Cell_ID_DepMap": line, "Cell_ID_Cellosaur": cvcl, "Organ": organ,
                              "Driver_Gene_Symbol": gene, "Driver_VarZyg": zyg, "Driver_VarType": vtype,
                              "Driver_ProtEffect_or_CdnaEffect": effect, "Driver_Mech_InferDM": "LoF",
                              "Driver_GeneType_DM": "Suppressor"})
    pq.write_table(pa.Table.from_pylist(cell_rows), meta / "cell_line_metadata.parquet")
    pq.write_table(pa.Table.from_pylist([{"gene_symbol": g, "ensembl_id": e, "token_id": i}
                                         for i, (g, e) in enumerate(TAHOE_GENES.items())]),
                   meta / "gene_metadata.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"sample": f"smp_{i}", "plate": f"plate{p}", "mean_gene_count": 1300.0, "mean_tscp_count": 2000.0,
         "mean_mread_count": 2400.0, "mean_pcnt_mito": 0.05, "drug": d, "drugname_drugconc": f"[('{d}', {c}, 'uM')]"}
        for i, (p, d, c) in enumerate([(1, BORTEZOMIB, 0.05), (1, BORTEZOMIB, 0.5), (1, BORTEZOMIB, 5.0),
                                       (2, BORTEZOMIB, 5.0), (1, "Erdafitinib ", 0.5)])
    ], schema=pa.schema([("sample", STR), ("plate", STR), ("mean_gene_count", F64), ("mean_tscp_count", F64),
                         ("mean_mread_count", F64), ("mean_pcnt_mito", F32), ("drug", STR),
                         ("drugname_drugconc", STR)])), meta / "sample_metadata.parquet")
    manifest = {
        "source_dataset": "tahoebio/Tahoe-100M", "source_revision": "fixture0000000000000000000000000000000000",
        "source_files": ["metadata/pseudobulk_differential_expression/part-00000.parquet"],
        "filters": {"permissive": "padj < 0.10", "significant": "padj < 0.05",
                    "high_quality": "padj < 0.05 and abs(log2FoldChange) > 0.5",
                    "validity": "finite padj and log2FoldChange; padj >= 0"},
        "rows": {"source": de.num_rows, "permissive": de.num_rows, "significant": sig.num_rows,
                 "high_quality": high.num_rows},
        "complete": True,
    }
    (root / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


# ---------------------------------------------------------------------------- oracles


def read_rows(root: Path, name: str) -> list[dict[str, Any]]:
    """All rows of a fixture table in file order (hive partitions restored for ``evidence``)."""
    path = Path(root) / name
    part = "hive" if name == "evidence" else None
    files = sorted(str(p) for p in path.rglob("*.parquet"))
    return ds.dataset(files, format="parquet", partitioning=part, partition_base_dir=str(path)
                      ).to_table().to_pylist()


def is_unknown(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _desc_key(value: Any) -> tuple:
    """Sort key for 'descending, nulls last'."""
    return (1, 0.0) if is_unknown(value) else (0, -value)


def _null_last(v: Any) -> tuple:
    return (1, "") if is_unknown(v) else (0, v)


def oracle_rows(root: Path, name: str, pred: Callable[[dict[str, Any]], bool] = lambda r: True
                ) -> list[dict[str, Any]]:
    return [r for r in read_rows(root, name) if pred(r)]


def oracle_total(root: Path, name: str, pred: Callable[[dict[str, Any]], bool] = lambda r: True) -> int:
    return len(oracle_rows(root, name, pred))


def oracle_topk(rows: Sequence[Mapping[str, Any]], order: str, key: Sequence[str], k: int) -> list[tuple]:
    """Top ``k`` key tuples by ``order`` descending, nulls last, ties by the canonical key."""
    ranked = sorted(rows, key=lambda r: (_desc_key(r[order]), tuple(_null_last(r[c]) for c in key)))
    return [tuple(r[c] for c in key) for r in ranked[:k]]


def oracle_grains(rows: Sequence[Mapping[str, Any]], column: str) -> int:
    return len({r[column] for r in rows if not is_unknown(r[column])})


def first_k_in_file_order(rows: Sequence[Mapping[str, Any]], key: Sequence[str], k: int) -> list[tuple]:
    return [tuple(r[c] for c in key) for r in rows[:k]]


def oracle_known_drug(root: Path, target_id: str = T) -> dict[str, Any]:
    """CT-4: rows of ``target_id`` with a known phase, the unknown-phase count and drug grains."""
    rows = oracle_rows(root, "known_drug", lambda r: r["targetId"] == target_id)
    known = [r for r in rows if not is_unknown(r["phase"])]
    return {"rows": known, "total": len(known), "unknown_phase": len(rows) - len(known),
            "drugs_total": oracle_grains(known, "drugId")}


def oracle_known_drug_topk(root: Path, target_id: str = T, k: int = 5) -> list[tuple]:
    return oracle_topk(oracle_known_drug(root, target_id)["rows"], "phase", KNOWN_DRUG_KEY, k)


def oracle_adverse_topk(root: Path, chembl_id: str = CHEMBL559288, k: int = 1) -> list[tuple]:
    rows = oracle_rows(root, "openfda_significant_adverse_drug_reactions", lambda r: r["chembl_id"] == chembl_id)
    return oracle_topk(rows, "llr", ("chembl_id", "meddraCode"), k)


def oracle_l2g(root: Path, gene_id: str = G_L2G, min_score: float = 0.05) -> list[dict[str, Any]]:
    return oracle_rows(root, "l2g_prediction",
                       lambda r: r["geneId"] == gene_id and not is_unknown(r["score"]) and r["score"] >= min_score)


def oracle_l2g_topk(root: Path, gene_id: str = G_L2G, min_score: float = 0.05, k: int = 5) -> list[tuple]:
    return oracle_topk(oracle_l2g(root, gene_id, min_score), "score", ("studyLocusId", "geneId"), k)


INTERACTION_KEY = ("sourceDatabase", "intA", "intB", "targetA", "targetB")


def oracle_interactions(root: Path, target_id: str = T) -> list[dict[str, Any]]:
    """Edges with ``target_id`` on either side (undirected)."""
    return oracle_rows(root, "interaction", lambda r: target_id in (r["targetA"], r["targetB"]))


def oracle_interactions_topk(root: Path, target_id: str = T, k: int = 5) -> dict[str, list[tuple]]:
    """Top ``k`` by ``scoring`` within each ``sourceDatabase`` (scores are comparable only within one)."""
    rows = oracle_interactions(root, target_id)
    return {src: oracle_topk([r for r in rows if r["sourceDatabase"] == src], "scoring", INTERACTION_KEY, k)
            for src in sorted({r["sourceDatabase"] for r in rows})}


def pgx_drug_ids(row: Mapping[str, Any]) -> list[str]:
    return [d["drugId"] for d in row.get("drugs") or []]


PGX_KEY = ("targetFromSourceId", "variantRsId", "genotypeId")


def oracle_pgx(root: Path, *, target_id: str | None = None, drug_id: str | None = None) -> list[tuple]:
    """Row keys of pharmacogenomics rows matching **every** given argument."""
    def ok(r):
        return ((target_id is None or r["targetFromSourceId"] == target_id)
                and (drug_id is None or drug_id in pgx_drug_ids(r)))
    return [tuple(r[c] for c in PGX_KEY) for r in oracle_rows(root, "pharmacogenomics", ok)]


def oracle_drugs_for_target(root: Path, target_id: str) -> list[str]:
    return [r["id"] for r in oracle_rows(root, "drug_molecule",
                                         lambda r: target_id in ((r.get("linkedTargets") or {}).get("rows") or []))]


def oracle_mouse(root: Path, target_id: str) -> list[dict[str, Any]]:
    return oracle_rows(root, "mouse_phenotype", lambda r: r["targetFromSourceId"] == target_id)


GO_ITEM_KEY = ("id", "aspect", "evidence", "source", "geneProduct")


def oracle_go_items(root: Path, target_id: str) -> list[tuple]:
    """GO item rows of a target with their complete item keys (parent id + item key)."""
    out = []
    for r in oracle_rows(root, "target", lambda r: r["id"] == target_id):
        for item in r.get("go") or []:
            out.append((r["id"], *(item[c] for c in GO_ITEM_KEY)))
    return out


def oracle_go_search(root: Path, query: str) -> dict[str, int]:
    """GO ids containing ``query`` (casefold) annotated on targets, with their gene counts."""
    counts: dict[str, int] = {}
    for r in read_rows(root, "target"):
        for item in r.get("go") or []:
            if query.casefold() in item["id"].casefold():
                counts[item["id"]] = counts.get(item["id"], 0) + 1
    return counts


def oracle_disease_search(root: Path, text: str) -> list[str]:
    """Diseases whose name or any synonym contains ``text`` literally (casefold)."""
    q = text.casefold()
    out = []
    for r in read_rows(root, "disease"):
        names = [r["name"] or ""] + [s for v in (r.get("synonyms") or {}).values() for s in (v or [])]
        if any(q in n.casefold() for n in names):
            out.append(r["id"])
    return out


def oracle_retired(root: Path, retired_id: str) -> list[str]:
    return [r["id"] for r in read_rows(root, "disease") if retired_id in (r.get("obsoleteTerms") or [])]


def oracle_evidence_by_publication(root: Path, pmid: str) -> list[str]:
    return [r["id"] for r in read_rows(root, "evidence") if pmid in (r.get("literature") or [])]


def oracle_biosample_search(root: Path, query: str) -> list[str]:
    q = query.casefold()
    return [r["biosampleId"] for r in read_rows(root, "biosample")
            if q in (r["biosampleName"] or "").casefold() or any(q in s.casefold() for s in r.get("synonyms") or [])]


def oracle_evidence_years(root: Path, target_id: str = T, *, min_year: int | None = None,
                          max_year: int | None = None) -> dict[str, Any]:
    """Evidence ids of ``target_id`` inside the year window, and rows with an unknown year."""
    rows = oracle_rows(root, "evidence", lambda r: r["targetId"] == target_id)
    known = [r for r in rows if not is_unknown(r["publicationYear"])]
    keep = [r["id"] for r in known if (min_year is None or r["publicationYear"] >= min_year)
            and (max_year is None or r["publicationYear"] <= max_year)]
    return {"ids": keep, "unknown": len(rows) - len(known)}


def oracle_phenotype(root: Path, phenotype: str = SEIZURE, evidence_type: str | None = None) -> dict[str, Any]:
    """Diseases with at least one non-negated (matching) evidence item, their recomputed counts,
    and the diseases excluded because every remaining item is negated."""
    listed: dict[str, int] = {}
    negated: list[str] = []
    for r in oracle_rows(root, "disease_phenotype", lambda r: r["phenotype"] == phenotype):
        items = [e for e in r["evidence"] or [] if evidence_type is None or e["evidenceType"] == evidence_type]
        positive = [e for e in items if not e["qualifierNot"]]
        if positive:
            listed[r["disease"]] = len(positive)
        elif items:
            negated.append(r["disease"])
    return {"diseases": listed, "excluded_negated": sorted(negated)}


def oracle_studies(root: Path, min_sample_size: int) -> dict[str, Any]:
    rows = read_rows(root, "study")
    known = [r for r in rows if not is_unknown(r["nSamples"])]
    return {"ids": [r["studyId"] for r in known if r["nSamples"] >= min_sample_size],
            "unknown": len(rows) - len(known)}


SAFETY_CODES = {-1.0: "known_unfavourable", 0.0: "none_recorded"}


def oracle_no_safety_events(root: Path) -> dict[str, Any]:
    """Targets whose ``hasSafetyEvent`` is the declared ``none_recorded`` code (0), and the rows
    whose value is unknown (null or NaN)."""
    rows = read_rows(root, "target_prioritisation")
    return {"ids": [r["targetId"] for r in rows if r["hasSafetyEvent"] == 0.0],
            "unknown": sum(is_unknown(r["hasSafetyEvent"]) for r in rows),
            "observed": sorted({r["hasSafetyEvent"] for r in rows if not is_unknown(r["hasSafetyEvent"])})}


def oracle_datasources(root: Path, *, indirect: bool = False) -> list[str]:
    name = "association_by_datasource_indirect" if indirect else "association_by_datasource_direct"
    return sorted({r["datasourceId"] for r in read_rows(root, name)})


def oracle_datasource_rows(root: Path, datasource: str, target_id: str, *, indirect: bool) -> list[tuple]:
    name = "association_by_datasource_indirect" if indirect else "association_by_datasource_direct"
    return [(r["targetId"], r["diseaseId"], r["datasourceId"]) for r in read_rows(root, name)
            if r["datasourceId"] == datasource and r["targetId"] == target_id]


def oracle_direct_indirect(root: Path, target_id: str = T) -> dict[str, Any]:
    direct = {(r["targetId"], r["diseaseId"]) for r in read_rows(root, "association_overall_direct")
              if r["targetId"] == target_id}
    indirect = {(r["targetId"], r["diseaseId"]) for r in read_rows(root, "association_by_overall_indirect")
                if r["targetId"] == target_id}
    return {"direct": direct, "indirect": indirect, "unique_to_direct": len(direct - indirect)}


def oracle_coloc(root: Path, method: str, chromosome: str, min_score: float = 0.8) -> list[str]:
    name, col = ("colocalisation_coloc", "h4") if method == "coloc" else ("colocalisation_ecaviar", "clpp")
    return [r["leftStudyLocusId"] for r in read_rows(root, name)
            if r["chromosome"] == chromosome and r[col] >= min_score]


# Tahoe

TAHOE_KEY = ("drug", "concentration", "concentration_unit", "Cell_ID_DepMap", "plate", "gene_name")


def tahoe_table(root: Path) -> pa.Table:
    return pq.read_table(Path(root) / TAHOE_DE)


def oracle_tahoe(root: Path, drug: str, cell_line: str | None = None, concentration: float | None = None,
                 max_padj: float = 0.10) -> list[dict[str, Any]]:
    """Rows with the float32 storage type honoured: the literal is cast to float32 before comparing."""
    import struct

    def f32(x: float) -> float:
        return struct.unpack("f", struct.pack("f", x))[0]

    rows = tahoe_table(root).to_pylist()
    return [r for r in rows if r["drug"] == drug and r["padj"] <= max_padj
            and (cell_line is None or r["Cell_ID_DepMap"] == cell_line)
            and (concentration is None or r["concentration"] == f32(concentration))]


def oracle_tahoe_concentrations(root: Path, drug: str, cell_line: str | None = None) -> list[float]:
    """Distinct doses, sorted, rendered in the float32 storage type (0.05, not 0.05000000074505806)."""
    from vbt.datalayer.rowkey import render_float

    values = {r["concentration"] for r in oracle_tahoe(root, drug, cell_line)}
    return [float(render_float(v, "float")) for v in sorted(values)]


def tahoe_keys(rows: Iterable[Mapping[str, Any]]) -> list[tuple]:
    from vbt.datalayer.rowkey import render_float

    return [tuple(render_float(r[c], "float") if c == "concentration" else r[c] for c in TAHOE_KEY) for r in rows]


# ---------------------------------------------------------------------------- preconditions


def assert_file_order_traps(root: Path) -> None:
    """Fail (as a fixture error) unless every ranked table can expose a file-order answer."""
    problems = []
    kd = oracle_known_drug(root)
    if first_k_in_file_order(kd["rows"], KNOWN_DRUG_KEY, 5) == oracle_known_drug_topk(root):
        problems.append("known_drug: first 5 rows in file order equal the oracle top 5")
    ae = oracle_rows(root, "openfda_significant_adverse_drug_reactions", lambda r: r["chembl_id"] == CHEMBL559288)
    if first_k_in_file_order(ae, ("chembl_id", "meddraCode"), 1) == oracle_adverse_topk(root):
        problems.append("openfda drug reactions: the first row is the top llr")
    l2g = oracle_l2g(root)
    if first_k_in_file_order(l2g, ("studyLocusId", "geneId"), 5) == oracle_l2g_topk(root):
        problems.append("l2g_prediction: first 5 rows equal the oracle top 5")
    best = max(range(len(l2g)), key=lambda i: l2g[i]["score"])
    if best < 5:
        problems.append("l2g_prediction: fewer than 5 rows with score >= 0.05 precede the maximum")
    inter = oracle_interactions(root)
    for src, top in oracle_interactions_topk(root).items():
        rows = [r for r in inter if r["sourceDatabase"] == src]
        if first_k_in_file_order(rows, INTERACTION_KEY, 5) == top:
            problems.append(f"interaction/{src}: first 5 rows equal the oracle top 5")
    if not any(r["targetB"] == T for r in inter) or not any(r["targetA"] == T for r in inter):
        problems.append("interaction: T is not on both sides")
    targets = [r["id"] for r in read_rows(root, "target")]
    if targets.index(TP53BP1) > targets.index(TP53) or targets.index(TP53I3) > targets.index(TP53):
        problems.append("target: TP53BP1/TP53I3 must precede TP53 in file order")
    studies = read_rows(root, "study")
    if [r["studyId"] for r in studies].index(STUDY_BIG) < 20:
        problems.append("study: Sbig must come after 20 null-size studies")
    if problems:
        raise AssertionError("FIXTURE: " + "; ".join(problems))


def check_descriptor_sentinels(root: Path) -> list[str]:
    """Problems with the descriptor sentinels (PCSK9 with non-empty pathways and go; the
    known_drug sentinel row)."""
    out = []
    pcsk9 = oracle_rows(root, "target", lambda r: r["id"] == PCSK9)
    if not pcsk9 or not pcsk9[0]["pathways"] or not pcsk9[0]["go"] or pcsk9[0]["approvedSymbol"] != "PCSK9":
        out.append("target sentinel PCSK9 (approvedSymbol, non-empty pathways and go)")
    if (root / "known_drug").exists():
        hit = [r for r in read_rows(root, "known_drug")
               if all(r[k] == v for k, v in KNOWN_DRUG_SENTINEL.items())]
        if len(hit) != 1:
            out.append("known_drug sentinel row")
    return out


__all__ = [n for n in dir() if not n.startswith("_")]
