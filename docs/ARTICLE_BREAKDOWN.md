# Article breakdown — The Virtual Biotech (Science, 2026)

## Core idea
A multi-agent AI organization modeled on a drug-development company. A virtual Chief
Scientific Officer (CSO) decomposes a user's question, routes sub-tasks to specialist
scientist agents (each with only its expertise-aligned data tools), has a reviewer agent
check their work, and synthesizes a recommendation. Claimed benefits: isolated per-agent
contexts, massive parallelism, and a unified reasoning layer that weighs conflicting evidence.

## Organization (11 agents)
| Division | Agent | Scope | Data |
|---|---|---|---|
| CSO office | CSO | interview, decompose, route, synthesize; never touches data | — |
| | Chief of Staff (Haiku 4.5) | field briefing, data landscape, recent news | web, tool inventory |
| | Scientific Reviewer (Haiku 4.5) | gaps, unsupported claims, alignment; triggers re-delegation | — |
| Target ID & Prioritization | Statistical genetics | GWAS, L2G, credible sets, burden, QTL coloc | Open Targets |
| | Functional genomics & perturbation | CRISPR essentiality, drug perturbation | DepMap, Tahoe-100M |
| | Single-cell atlas | cell-type expression, DE, cell-cell communication | CELLxGENE Census, Tabula Sapiens |
| Target Safety | Bio-pathways & PPI | pathway/interaction safety reasoning | Reactome, GO, PPI |
| | FDA safety officer | adverse events, warnings, mouse KO | OpenFDA, labels |
| Modality Selection | Target biologist | protein class, localization, tractability | HPA, tractability |
| | Pharmacologist | precedent drugs/probes, practicality | ChEMBL |
| Clinical Officers | Clinical trialist | prior trials, failure analysis, survival | ClinicalTrials.gov, cBioPortal, PubMed |

## Workflow (Fig. 1C)
User query → (CSO clarification interview ∥ Chief-of-Staff briefing) → CSO delegation
(parallel, isolated contexts) → scientists call MCP tools, write and run code → Scientific
Reviewer → gaps? re-delegate : CSO synthesis → user follow-ups (multi-turn).

## Implementation facts
Claude Agent SDK; Sonnet 4.5 for CSO + scientists, Haiku 4.5 for CoS + reviewer; 10 FastMCP
servers, >100 tools returning summaries + previews; file ops + code execution; Skills with
progressive disclosure; Pydantic-validated JSON for bulk extraction; UI with downloadable
data, code and reports.

## Case studies
1. **Target prioritization.** 37,075 parallel clinical-trialist agents (one per NCT ID;
   ClinicalTrials.gov → PubMed → press releases; $0.23 median/trial; 6 h vs ~77 days serial).
   Agreement: 88.4 / 88.4 / 92.4% (primary / secondary / AE) vs manual review; 85.6% vs TDC.
   Features on Tabula Sapiens: τ specificity and bimodality coefficient (ρ=0.54). Cell-type-specific
   targets: +48% likelihood of reaching Phase IV, −32% adverse events; OR 1.12 primary endpoint,
   1.27 Phase I→II; robust to permutation, mixed-effects and genetic-evidence adjustment.
2. **B7-H3 in lung cancer** (no web; $50). Weak genetics (non-disqualifying) → fibroblast-concentrated
   up-regulation (SCLC log2FC 2.13, LUAD 1.79) → LIANA: 180/226 interactions → reviewer asks for spatial →
   Visium/Cell2Location: immune exclusion around B7-H3-high spots → TCGA LUAD Cox: OS HR 1.62, DFS HR 2.06
   → ADC nominated (later matched by FDA Breakthrough designation for ifinatamab deruxtecan).
3. **OSMR / MOONGLOW failure** ($59). Refractory population → TAURUS atlas → STAT1 activity up in 9/10
   stromal types → Shapley/LMG: OSMR not the dominant gp130-family driver (redundancy) → 10-gene gp130-axis
   score beats OSMR alone across 4 GEO cohorts, ≈ Arijs signature → cross-disease OSMR program (112 diseases).

Verification: 4 experts re-implemented 8 analyses; all conclusions and code judged correct.
Compared favourably with Biomni, Kosmos, PantheonOS.

## Limitations stated
Decision support only; weaker for rare/understudied diseases; observational trial analyses;
possible LLM prior bias (mitigated by post-cutoff case studies); no wet-lab; absence of binders
is an evidence gap, not infeasibility.
