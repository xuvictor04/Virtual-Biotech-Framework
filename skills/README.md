# Local skills

Skills are packaged workflows loaded on demand with the `Skill` tool (progressive
disclosure: the agent reads `SKILL.md`, then only the procedure files it needs).
This directory is searched before the upstream skills in
`third_party/TheVirtualBiotech/.claude/skills` (single-cell QC, single-cell
analysis, evidence citation, run organization), so a local skill with the same
frontmatter `name` overrides the upstream one. Add a skill as
`skills/<name>/SKILL.md` plus any supporting files.

How skills reach the agents:

- The `Skill` tool's description lists the catalog (name and one-line description of
  every skill found in `paths.skills`), so agents can discover skills without a search.
  `Skill` accepts `name`, `/name` and `.claude/skills/name`.
- At session start every skill is linked (or copied) into `<run>/.claude/skills/<name>`,
  and its content hash is pinned in the run's `inputs/config.json` (`skill_hashes`), so
  a run records exactly which skill versions its agents could load.
- The skill roots are read-only roots for every agent (`paths.read_roots`); agents
  cannot edit a skill.

Harness skills:

- `project-utilities` — the data/tooling engineer's procedure (docs/PROJECTS.md): inspect files, draft a
  descriptor, acquisition spec, plugin or utility, get it validated and registered in the active project, confirm
  it works, answer the CSO; with a descriptor cheat sheet and utility and plugin templates. Projects add their own
  skills under `<project>/skills/`, searched after these roots (a project skill never shadows a shipped one).

Local overrides:

- `run-organization` (overrides the upstream skill of the same name) — the upstream
  skill describes the upstream app's run
  directory. This version describes the layout this harness actually writes
  (MANIFEST.json, README.md and audit.html rewritten each turn,
  `inputs/config.json`, `evidence/{artifacts,claims,provenance}.json`,
  `work/<agent>/`, `memory/<agent>/`, `logs/`), and notes that the restricted
  CSO records the plan and claims through the provenance tools rather than
  `Write`/`Bash`.
