# Local skills

Skills are packaged workflows loaded on demand with the `Skill` tool (progressive
disclosure: the agent reads `SKILL.md`, then only the procedure files it needs).
This directory is searched before the upstream skills in
`third_party/TheVirtualBiotech/.claude/skills` (single-cell QC, single-cell
analysis, evidence citation, run organization). Add a skill as
`skills/<name>/SKILL.md` plus any supporting files.
