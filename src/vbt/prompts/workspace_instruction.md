## Your workspace layout

All file operations (Write, Edit, NotebookEdit, Bash output files) MUST go under
your own workspace, `work/{agent}/` in the run directory (its absolute path is
in the Session section; relative paths resolve there). Do not write to another
agent's directory or to the run root. Use this layout:

  code/scripts/       analysis scripts you write
  data/raw/           data as pulled from a tool or database
  data/processed/     data after your QC / transformation
  results/figures/    plots (.png, .pdf)
  results/tables/     result tables (.csv, .tsv, .parquet)
  results/reports/    your written findings (.md)

Name files for what they contain (e.g. `il33_expression_by_celltype.csv`), not
for the order you made them. Every file you leave behind is an audit artifact
someone else has to interpret.

Before starting, CHECK FOR EXISTING WORK from earlier in this run, e.g.
`Glob(pattern="work/*/results/**/*")` and `work/*/data/processed/*`. If a prior
analysis already produced what you need, load it rather than recomputing. MCP
data tools save query outputs under `work/_mcp/data/processed/`.

Before returning findings, call `mcp__provenance__register_artifact` for each
supporting file with a short description, and quote its exact run-relative path
(`work/{agent}/results/...`) in your report so the CSO can cite it.

If a data tool fails, report the failed source and the resulting limitation.
Identify web or other replacement sources explicitly; never describe them as
results from the unavailable database. A successful query with zero matches is
a separate outcome and must retain its scope and filters.

When writing code: errors are debugging problems, not stopping points. Read the
traceback, fix, and re-run. Do not silently downgrade the requested method; if
a fallback is unavoidable, say exactly what failed and flag the downgrade.
