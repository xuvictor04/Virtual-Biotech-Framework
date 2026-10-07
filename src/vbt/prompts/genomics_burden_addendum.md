# Harness note: rare-variant burden and other association evidence

You also have Open Targets association tools (`mcp__association__*`). Use them
for genetic evidence that your GWAS/L2G/colocalisation tools do not cover:

- Rare-variant **gene burden** evidence: query datasource `gene_burden`, e.g.
  `mcp__association__query_evidence(target_id=..., disease_id=..., datasource_id="gene_burden")`
  for the evidence strings, or
  `mcp__association__filter_by_datasource(output_path=..., target_id=..., datasource="gene_burden")`
  for the association scores (saved as parquet in your workspace). Report the
  burden test, cohort/resource and effect direction when present.
- Other genetic datasources (e.g. `gwas_credible_sets`, `eva` (ClinVar),
  `genomics_england`, `orphanet`, `clingen`, `gene2phenotype`, `uniprot_variants`)
  and the `genetic_association` datatype (`mcp__association__filter_by_datatype`).
- `mcp__association__compare_direct_indirect` to check whether support comes
  from the disease itself or only from descendant terms;
  `mcp__association__get_associations_for_target` / `query_associations` for
  the overall picture.

A resolved query with `status: empty` means no rows in this source's coverage;
record it only as an absence finding (`supports: absence`), never as support. A
`not_found` error means the identifier is wrong. A failed query is not evidence
of absence.
