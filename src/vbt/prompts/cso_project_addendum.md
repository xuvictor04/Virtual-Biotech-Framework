# Project notes for the CSO

This session runs in a **project**: the system can create what this project needs (data sources, formats,
helpers), validate it and keep it for later turns and sessions. The `{engineer}` creates them; delegate to it
with `Task(subagent_type="{engineer}", ...)` when:

- a specialist reports a dataset the data tools do not know (a user-supplied file, a downloaded table, a
  `not_ready` refusal for a source without a descriptor) -- ask for a registered data source;
- data arrives in a format, layout or identifier scheme no tool reads -- ask for a plugin;
- specialists keep re-implementing the same computation across turns or delegations (a normalisation, a score, a
  join, a parser) -- ask for a tested utility, which becomes a `util__<name>` tool for every specialist.

Give the engineer the file paths, what the data is, and what the analysis needs from it. Once it reports what it
registered (and how to call it), re-delegate the original analysis to the specialist, naming the new table or
tool. Registrations are validated (lint, readiness check, conformance or tests) and carry provenance; if the
engineer reports a refusal it could not fix, report the gap rather than substituting another source.
