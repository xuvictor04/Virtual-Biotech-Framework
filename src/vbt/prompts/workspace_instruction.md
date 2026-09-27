
## Your workspace layout

All file operations (Write, Edit, Bash output files) MUST go under your own
directory `{workspace}`. Do not write to another agent's directory or to the
run root. Use absolute paths:

  {workspace}/code/scripts/     analysis scripts you write
  {workspace}/data/raw/         data as pulled from a tool or database
  {workspace}/data/processed/   data after your QC / transformation
  {workspace}/results/figures/  plots (.png, .pdf)
  {workspace}/results/tables/   result tables (.csv, .tsv, .parquet)
  {workspace}/results/reports/  your written findings (.md)

Name files for what they contain (e.g. `il33_expression_by_celltype.csv`).
Every file you leave behind is an audit artifact someone else has to interpret.

Before starting, CHECK FOR EXISTING WORK from earlier in this run:
  ls {run_dir}/work/*/results/ {run_dir}/work/*/data/processed/ 2>/dev/null
If a prior analysis already produced what you need, load it rather than
recomputing. MCP data tools save query outputs under
`{run_dir}/work/_mcp/data/processed/`.

Before returning findings, call `mcp__provenance__register_artifact` for each
supporting file with a short description, and quote its exact path in your
report so the CSO can cite it.

If a data tool fails, report the failed source and the resulting limitation.
Identify web or other replacement sources explicitly; never describe them as
results from the unavailable database. A successful query with zero matches is
a separate outcome and must retain its scope and filters.

When writing code: errors are debugging problems, not stopping points. Read the
traceback, fix, and re-run. Do not silently downgrade the requested method; if
a fallback is unavoidable, say exactly what failed and flag the downgrade.
