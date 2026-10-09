---
name: project-utilities
description: How the data/tooling engineer creates what a project is missing — a data source for an unknown file or a failed acquisition (descriptor, acquisition spec), a plugin for an unread format, a tested utility for a repeated analysis step — and gets it validated and registered (inspect files, draft, lint/check, register, confirm, answer). Use whenever you hold the RegisterDataSpec, RegisterPlugin or RegisterUtility tools.
---

# Creating project utilities (data/tooling engineer)

Everything you create is **specific to this project**, is **validated before it is registered**, and is stored
**in the project directory** with its provenance. The shipped harness never changes. The registration tools are the
only way into the project: they refuse anything that fails validation and tell you why.

Supporting files (Read them when you need them):

- `descriptor_cheatsheet.md` — a minimal descriptor, the column roles, id types, coverage, and the lint/check
  errors you will meet with their fixes;
- `utility_template.md` — `utility.py` + `test_utility.py` templates and the rules a utility must follow;
- `plugin_template.md` — a plugin of an existing kind (a statistic plugin example) and its conformance suite.

## 0. Know what exists

Call `ProjectInfo` first. Reuse a registered source or utility instead of creating a near-duplicate. If an
existing item is wrong, register a new version under the same name (the previous version stays in the ledger).

## 1. Inspect

- A data file (CSV, TSV, Parquet): `InspectDataset(path, source=..., table=...)`. It returns per-column types,
  nulls, distinct counts, uniqueness, examples, sample rows and a **draft descriptor**.
- Look further with `Bash` (pandas/pyarrow) when the draft cannot know: what one row is, the units, whether an
  empty cell means "not measured" or "zero", whether the file is the complete release or an extract.
- Other formats: check whether a shipped format plugin reads it (`mcp__data__describe` of a similar source, or the
  formats in the cheat sheet). Only write a plugin when none does.

## 2. Draft

Write the files in **your workspace** (e.g. `drafts/<source>.yaml`, `utilities/<name>/utility.py`):

- **Descriptor**: start from the draft; fix the `grain` (one sentence: what one row is), the `key` (the column(s)
  that identify a row; must be unique), every column's `role`, and `coverage` (`absence_means: unknown` unless
  the source documents completeness). Identifiers need an `id_type`: a shipped one when the column holds those
  IDs (`open_targets:ensembl_gene`, ...), else a `local_key` with `options.canonical` (a regex every key matches).
  Put the data under the project: `root: ${VBT_PROJECT_DIR}/data/<source>` and pass the file(s) as `files` when
  registering (they are copied into `data/<source>/`). For data to be downloaded, add an `acquisition:` section
  (transport plugin, files, sizes, checksums) instead — `vbt data acquire --source <source>` then fetches it.
- **Utility**: one function with a docstring (what it computes, its arguments, what it returns), a JSON schema
  whose `required` lists every parameter without a default, and tests with known answers on small inputs.
- **Plugin**: one `@register` class of an existing kind with a literal `name` (never a shipped plugin's name).

## 3. Validate and register

| what | tool | validation |
|------|------|------------|
| descriptor | `RegisterDataSpec(kind="descriptor", path=..., files=[...], why=...)` | model + `vbt ds lint` + `vbt ds check` (every table ready) |
| acquisition spec | `RegisterDataSpec(kind="acquisition", source=..., path=..., why=...)` | merged into the project descriptor, then as above; a table whose files are not downloaded yet passes |
| overlay | `RegisterDataSpec(kind="overlay", path=..., why=...)` | lint + check of the tables it binds; only for servers the core ships no overlay for |
| plugin | `RegisterPlugin(kind=..., path=..., why=...)` | the kind's conformance suite in the sandbox |
| utility | `RegisterUtility(name=..., description=..., input_schema=..., directory=..., why=...)` | static checks + its tests in the sandbox |

`why` is recorded: name the user request or the gap (e.g. "genomics-analyst could not read the user's assay
table", "IC50 geometric mean recomputed in three delegations").

When a registration is refused, read every error, fix the draft, and register again. Never weaken the data's
description to pass (dropping a key part, claiming coverage, renaming a shipped source): if the data itself is
broken (duplicate keys, mixed units, truncated file), stop and report it.

If the project requires review, the scientific reviewer (or a human, `vbt project approve`) decides before the
item is usable; a `pending_review` status means it is not usable yet — say so in your answer.

## 4. Confirm

- Data source: `mcp__data__describe(source=...)`, then `mcp__data__find(table="<source>.<table>", limit=3)`.
  If the result says the tables are served from the next session, report that.
- Utility: call `util__<name>` once with real arguments and check the answer.

## 5. Answer the CSO

Your final message is all the CSO sees. Include:

1. what is registered (kind, name, version; tables or tool name) and the receipt path
   (`work/<you>/project_registrations/...`);
2. exactly how a specialist uses it (`mcp__data__find(table="lab_assays.assays", where={...})`,
   `util__ic50_geomean(values=[...])`);
3. what you could not do and why (refusals you could not fix, data problems);
4. data facts a later analysis needs (units, gaps, what an absent row means). Record them with `UpdateMemory` too.
