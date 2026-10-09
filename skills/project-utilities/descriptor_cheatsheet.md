# Descriptor cheat sheet (project data sources)

The full specification is docs/DATA_LAYER.md (§6); the shipped descriptors in `configs/data/sources/` are worked
examples (`depmap.yaml` for CSV entity and matrix tables, `open_targets.yaml` for Parquet with nested columns).

## A minimal entity table (CSV)

```yaml
schema: vbt.datasource/1
source: lab_assays                       # lowercase identifier, not a shipped source name
title: In-house drug sensitivity assays
root: ${VBT_PROJECT_DIR}/data/lab_assays # the files passed as RegisterDataSpec(files=[...]) land here
release: {expect: "2026-10", from: literal}
defaults: {format: csv, layout: single_file, missing: unknown}   # csv | tsv | parquet | jsonl | ...
id_types:
  assay: {plugin: local_key, options: {canonical: '^A\d+$'}, universe: assays.assay_id}
tables:
  assays:
    kind: entity
    path: assays.csv
    grain: one assay measurement (one compound on one cell line)
    key: {columns: [assay_id], check: full}
    coverage: {statement: "The assays of the 2026-10 export; an absent assay was not run or not exported.",
               absence_means: unknown}
    columns:
      assay_id:    {role: identifier, id_type: assay, self: true}
      gene_symbol: {role: category, vocab: data}
      cell_line:   {role: category, vocab: data}
      ic50_nm:     {role: measure, statistic: numeric, unit: nM, missing: unknown}
      tissue:      {role: category, vocab: data}
```

Write regular expressions in single quotes in YAML (`'^A\d+$'`); in double quotes a backslash is an escape.

## Column roles (most used)

| role | for | notes |
|------|-----|-------|
| `identifier` | IDs (`id_type` required; `self: true` on the table's own key) | a shipped id type when the values are those IDs, e.g. `open_targets:ensembl_gene`; else `local_key` with `options.canonical` |
| `label` | a human name of an identifier (`of: <id column>`) | `unique: false` when names repeat |
| `category` | small sets of values (`vocab: data` reads them from the data) | `placeholders: [Unknown]` for values that mean missing |
| `measure` | numbers with a `statistic` (`numeric`, `score_0_1`, `pvalue`, `percent`, `count`...) | `unit`, `direction`, `scale`, `missing: unknown|zero` |
| `flag` | booleans | `encoding: {"True": true, "False": false}` for text booleans |
| `time` | dates | |
| `payload` | free text or anything not filtered on | the data tools return it but do not filter on it |

## Coverage

`absence_means: unknown` (default stance), `absent` (the source documents that a missing row means "does not
exist"), or `censored`. `statement` says in one sentence what the table covers.

## Errors you will meet

| message (lint `[rule]` or check `R..`) | fix |
|---|---|
| `source 'x' is shipped` | choose another source name |
| `R4b ... cannot be configured (local_key needs options.canonical)` | add `options: {canonical: '<regex>'}` |
| `R4b ... keys are not in the canonical form` | the regex does not match the values: check the examples (and the quoting) |
| `R5b key not unique` | the key is not the row identity: add the missing key column(s) |
| `R5 nulls in key` | a key part is empty in some rows: report it, or declare it in `key.nullable` if that is the data's meaning |
| `R1 ... missing` | `root`/`path` do not point at the file: import it with `files` or fix `path` |
| `schema_drift` / `R3` | a declared column is not in the file, or its type differs: fix the column names |
| lint `[reference]` | a role names a column that does not exist |

`vbt ds lint` and `vbt ds check --table <source>.<table>` run the same validation by hand.
