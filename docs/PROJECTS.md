# Projects: the system creates project-specific utilities as needed

The harness ships **general mechanisms** — descriptors, plugin kinds, the gateway, generic acquisition,
validation — and nothing specific to one dataset or project. What a project needs beyond that (its own datasets,
an unusual file format, a helper its analyses keep re-implementing) is **created by the system at run time**: the
data/tooling engineer agent drafts it, the harness validates it, and it is stored with the project, with its
provenance, for later turns and sessions. Owners never need a developer to write per-dataset or per-project code,
and nothing a project creates touches the shipped harness or other projects.

| piece | where |
|-------|-------|
| project directory, activation, `vbt project` | `src/vbt/projects/` (`model.py`, `cli.py`) |
| validation, registration, review, provenance | `src/vbt/projects/authoring.py`, `ledger.py` |
| sandbox for project code | `src/vbt/projects/sandbox.py`, `runner.py` |
| agent tools | `src/vbt/tools/utilities.py` (`ProjectInfo`, `InspectDataset`, `RegisterDataSpec`, `RegisterPlugin`, `RegisterUtility`) and `util__<name>` (`src/vbt/projects/utilities.py`) |
| the role | `data-engineer` in `configs/agents.yaml`, `src/vbt/prompts/data_engineer.md`, `skills/project-utilities/` |
| CSO delegation rules | `src/vbt/prompts/cso_project_addendum.md` |
| catalog search paths | `src/vbt/datalayer/catalog.py`, `src/vbt/datalayer/descriptor/load.py` (`VBT_PROJECT_DIR`) |
| tests | `tests/test_projects.py`, `tests/test_utilities.py` |

## Quick start

```bash
vbt project init oncology --description "In-house oncology screens"     # or: python -m vbt.projects init ...
vbt --profile <projects>/oncology/profile.yaml chat                     # run a session in the project
vbt project show oncology                                               # what the system registered
vbt project check oncology --tests                                      # files vs provenance, lint, tests again
```

`<projects>` is `projects.root`, else the deployment layout's projects directory: `$VBT_PROJECTS_DIR`,
`<VBT_HOME>/projects` (`/srv/vbt/projects` in the image), or `data/projects` in a checkout. `vbt project init`
prints the exact `--profile` argument. Once the CLI wires `--project` (see *Status*), `vbt chat --project
oncology`, `vbt run --project oncology ...` and `vbt setup --project oncology` do the same without a profile path.

In the session, ask for what you need ("we have an assay export in `incoming/assays.csv`; how do lung lines
compare?"). The CSO delegates to the `data-engineer` when a dataset, format or helper is missing; the engineer
registers it; the CSO then re-delegates the analysis to a specialist, who uses the new table through the data tools
(`mcp__data__find(table="lab_assays.assays", ...)`) or the new tool (`util__ic50_geomean(values=[...])`).

## The project directory

```
<projects>/<name>/
  project.yaml        schema vbt.project/1: name, description, created, settings (e.g. review: human)
  profile.yaml        generated: activates the project for any vbt command (vbt --profile <it> ...)
  descriptors/        <source>.yaml   data sources the project added (acquisition sections inside)
  overlays/           <server>.yaml   bindings of MCP servers the core ships no overlay for
  plugins/<kind>/     <name>.py       plugins of existing kinds (format, layout, statistic, identifier, ...)
  utilities/<name>/   utility.py, test_utility.py, utility.json, fixtures/   -> the tool util__<name>
  skills/<name>/      SKILL.md        project skills
  memory/<agent>/     MEMORY.md       project notes injected into that role's prompt in every session
  data/<source>/                      data files imported with a descriptor (${VBT_PROJECT_DIR}/data/<source>)
  runs/                               the project's run records (projects.runs_in_project)
  provenance/ledger.jsonl             every registration attempt: registered, refused, pending, approved, rejected
  provenance/<kind>/<name>.json       the current record of each registered item
  provenance/pending/<kind>/<name>/   items waiting for a human decision (projects.review: human)
```

Only the authoring tools and `vbt project` write here; agents cannot (the project is outside every run directory,
so the path policy never lets `Write`, `Edit` or `Bash` redirect into it; under bwrap it is mounted read-only for
every command). `vbt project check` reports any file whose hash no longer matches its record, and a changed
utility is not loaded as a tool.

## Activation

Activating a project (`vbt.projects.activate`, or the generated `profile.yaml`) changes the configuration only by
**adding after** what is shipped:

| setting | with a project |
|---------|----------------|
| `project` | `{name, dir}` |
| `paths.skills` | the shipped roots, then `<project>/skills` (a project skill cannot shadow a shipped one) |
| `paths.read_roots` | the shipped roots, then the project directory (read-only for agents) |
| `paths.runs_dir` | `<project>/runs` (unless `projects.runs_in_project: false`) |
| `data.plugins.paths` | the configured paths, then the project's plugin modules |
| `tool_env.VBT_PROJECT_DIR` | the project directory |

`VBT_PROJECT_DIR` reaches every process that reads data: the harness's catalog (as `env.VBT_PROJECT_DIR`), the
data child, `vbt ds` commands, Bash and the data client. The catalog loads `<project>/descriptors` and
`<project>/overlays` **after** `data.descriptors_dir`/`overlays_dir`, and a project can only add: a project file
naming a shipped source or server, a generic overlay (`_*.yaml`), an overlay whose `same_as` names another
server's tool, or a file that does not load and whose name cannot be told apart from a shipped one is **refused**
on its own (`Catalog.project_refused`, an error in `vbt ds lint`) and never marks a shipped tool quarantined.
Other broken project files are quarantined like shipped ones (only the project's own tools depend on them).

The roster: `data-engineer` has `requires: [project]`, so the paper's roster (and the parity test) is unchanged
outside a project. In a project every agent with `Bash` also gets `util__*`, the CSO's stable prompt gets the
project addendum (when to delegate to the engineer), and every agent's volatile prompt ends with the project block:
the registered sources and tables, utilities (tool and description), plugins, skills, items pending review, and the
project notes for its role.

## What the system can create, and how each is validated

| item | tool | validated by | refused when |
|------|------|--------------|--------------|
| descriptor (+ data files) | `RegisterDataSpec(kind="descriptor", path\|content, files?, why)` | model, `vbt ds lint` rules (`lint_descriptor` against the shipped + project catalog), `vbt ds check` (the data child's `--check` at `projects.check_depth`, under its memory limit) on a staged copy of the project | lint errors; a table not `ready` (a `missing` table passes only when the descriptor's `acquisition:` lists it); a check error (key not unique, encoding drift, schema drift ...); a shipped source name |
| acquisition spec | `RegisterDataSpec(kind="acquisition", source, path\|content, why)` | merged into the project descriptor (a new version), then as above | no project descriptor `source`; as above |
| overlay | `RegisterDataSpec(kind="overlay", ...)` | lint (`lint_overlay`) and check of the tables it binds | a shipped server, a foreign `same_as` |
| plugin | `RegisterPlugin(kind, path\|content, why)` | exactly one `@register` class with a literal `name`, not taken by a shipped or project plugin; the kind's conformance suite (`vbt/datalayer/plugins/conformance/<kind>.py`) restricted to it, in the sandbox | an unknown kind (a new kind is a core change), a name collision, any failing case, no case collected |
| utility | `RegisterUtility(name, description, input_schema, directory\|code+tests, entry?, mode?, why)` | static checks (entry function with a docstring, schema is a JSON schema whose `required` covers every parameter without a default, `test_*` functions present), then its tests in the sandbox | any static problem, a failing or missing test, tests that modify the code they test |

`InspectDataset(path)` profiles a CSV/TSV/Parquet file (types, nulls, distinct values, uniqueness, examples) and
drafts a descriptor that the engineer reviews and edits. The draft follows what the format plugins will do with
the file, so a well-formed file's draft registers as is (checked on DepMap 24Q4 `Model.csv`, the Tahoe-100M drug
and cell-line metadata and the live HGNC complete set):

- the delimiter comes from the header line (a tab-separated `.txt` is `tsv`); column types are those the CSV
  plugin infers from the first MiB, and a column a later block does not convert (a number column that turns into
  text) is declared `format.options.column_types: {col: string}` — without it the plugin refuses the file;
  only an unquoted empty cell is null, as in the plugin (`n/a` and `NA` are values);
- the key is a unique null-free column (`*_id`, `key`, `accession` and `code` names first), else the first unique
  combination of two or three columns, else one that needs a column with empty values (declared `key.nullable`);
  a single key gets a `local_key` id type whose `canonical` is the narrowest generic pattern every value matches;
- numeric columns are measures, low-cardinality text columns categories, other text payload;
- a column whose name holds a `.` (HGNC's `pseudogene.org`) reads as a nested path in a descriptor, so the draft
  leaves it out and says so in `notes`.

`ProjectInfo` shows what exists.

Every refusal returns the errors (lint findings, failed checks, failing tests with tracebacks) so the engineer can
fix the draft and register again; a refused attempt writes nothing to the project except its ledger line.

## The sandbox

Project code (a utility's tests and calls, a plugin's conformance suite) runs as `python -I runner.py ...` with:

- **filesystem** (`projects.sandbox`): `auto` (default) uses bubblewrap when it works on the host — everything
  read-only except the calling agent's own work directory and the run's `.tmp`/`.home`, `paths.blocked_read`
  hidden — and otherwise falls back to the next two points only; `bwrap` requires it (fails closed); `none`
  never uses it;
- **memory**: the reaper's `RLIMIT_DATA` limit (`data.memory.workspace_mb`, as for Bash);
- **network**: none for tests and conformance suites (`projects.test_network: false`; bwrap `--unshare-net`, else
  `unshare -rn`); utility calls follow `bash.network_isolation` like Bash;
- **environment**: the allow-listed child environment (no provider keys) plus `tool_env`;
- **time**: `projects.test_timeout_s` (600) for tests, `projects.call_timeout_s` (1800, or the utility's own
  `timeout_s`) per call; the process group is killed on timeout.

The provenance of every test run names the sandbox that applied (`bwrap+netns+8000MB`, `rlimit+netns+8000MB`,
...). Production hosts should install bubblewrap; the tests are a quality gate on what the system wrote, not a
security boundary against a malicious author (an author can write trivial tests), which is what review is for.

## Review

`projects.review` (host configuration) and the project's own `settings.review` (which may only ask for more):

- `none` (default): validation alone decides;
- `reviewer`: the scientific reviewer is shown the item, its files and its validation results and must answer
  `VERDICT: APPROVE`; anything else refuses it (the review is a delegation in the run's trace). Where no reviewer
  agent can run (outside a session), the item waits for a human as below;
- `human`: the item waits under `provenance/pending/`; `vbt project approve NAME KIND ITEM` installs it (after
  checking its files are the ones validated), `vbt project reject ...` discards it.

## Provenance

Each registered item's record (`provenance/<kind>/<name>.json`) holds who (agent, run id, invocation id, tool use
id — or the user, from the CLI), when, why (the `why` argument), the files and their sha256, a combined source
hash, the validation (lint findings, check results with fingerprints, tests passed with their names and sandbox,
conformance exit code and stamp), the review, and the previous version. The ledger keeps every attempt.

In the run, each registration is also written to `work/<agent>/project_registrations/<kind>-<name>-v<N>.json` and
registered as an artifact (kind `project_<kind>`), so it is listed in the run's `MANIFEST.json` and `audit.html`;
the trace has `project_registration` and `project_utility_call` events.

## In-session and later sessions

- A **utility** becomes a tool in the running session at once (added to the registry) and is listed at the start of
  every later session of the project (from the project's skill root, `vbt.tools.builtin.builtin_tools`).
- A **data spec or plugin**: the session's catalog and plugin registry are rebuilt and handed to the running
  gateway, and the data child is restarted with the new settings, so the `mcp__data__*` tools serve the new tables
  in the next turn (`vbt.projects.reload`). If any step fails, the tool result says the item is served from the
  next session of the project, which builds everything from the project directory.
- **Project notes**: `vbt project memory NAME --from-run RUN` appends a run's agent notes (`memory/<agent>/`) to the
  project's, which every later session injects into that role's prompt.

## Configuration (`projects.*`)

| key | default | |
|-----|---------|--|
| `root` | null | the projects directory (null: `$VBT_PROJECTS_DIR`, `<VBT_HOME>/projects`, `data/projects`) |
| `review` | `none` | `none` \| `reviewer` \| `human` |
| `sandbox` | `auto` | `auto` \| `bwrap` \| `none` |
| `test_timeout_s` | 600 | a utility's tests or a plugin's conformance suite |
| `call_timeout_s` | 1800 | one utility call |
| `test_network` | false | network for tests and suites |
| `max_source_bytes` | 2000000 | largest descriptor, overlay, plugin or utility file |
| `check_depth` | `standard` | the `vbt ds check` depth a data spec must pass (`deep` on big hosts for full scans) |
| `runs_in_project` | true | run records under `<project>/runs` |

The defaults live in `vbt.projects.model.PROJECT_DEFAULTS`; a `projects:` block in a profile or the host
configuration overrides them.

## Status and limits

- `vbt project ...` is implemented (`vbt.projects.cli.add_project_parser`) and runs today as
  `python -m vbt.projects ...`; registering it under `vbt` and adding `--project` to `vbt chat|run|setup`
  (`add_project_argument`, `apply_project`) is a three-line change in `src/vbt/cli.py` that this package did not
  own. Until then a project is activated with its `profile.yaml`.
- The profile copies the shipped `paths.skills`/`read_roots`/`data.plugins.paths` lists as written in the
  configuration files (`${...}` kept); `vbt project check` reports a stale profile and `vbt project profile NAME`
  rewrites it.
- The run's pinned configuration does not yet record the project; the trace and the registration receipts do.
- Overlays only matter for MCP servers the owners add to their configuration; project data is normally served by
  the native data tools, which need no overlay.
