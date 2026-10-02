# Local skills

Skills are packaged workflows loaded on demand with the `Skill` tool (progressive
disclosure: the agent reads `SKILL.md`, then only the procedure files it needs).
This directory is searched before the upstream skills in
`third_party/TheVirtualBiotech/.claude/skills` (single-cell QC, single-cell
analysis, evidence citation, run organization), so a local skill with the same
frontmatter `name` overrides the upstream one. Add a skill as
`skills/<name>/SKILL.md` plus any supporting files.

Local overrides:

- `run-organization` — the upstream skill describes the upstream app's run
  directory. This version describes the layout this harness actually writes
  (MANIFEST.json, README.md and audit.html rewritten each turn,
  `inputs/config.json`, `evidence/{artifacts,claims,provenance}.json`,
  `work/<agent>/`, `memory/<agent>/`, `logs/`), and notes that the restricted
  CSO records the plan and claims through the provenance tools rather than
  `Write`/`Bash`.
