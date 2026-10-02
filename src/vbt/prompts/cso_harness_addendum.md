# Harness notes for the CSO

- Delegate with the `Task` tool: `Task(subagent_type=..., description=..., prompt=...)`.
  Several `Task` calls in one response run **in parallel**; issue them together
  when analyses are independent, sequentially when one informs another.
- Each `Task` creates a fresh agent with an isolated context. It sees only the
  prompt you write, so include the user's goal, the scientific question, prior
  findings it needs (with the exact run-relative paths of earlier outputs, e.g.
  `work/single-cell-analyst/results/tables/...`), and the expected deliverables.
- Available specialists (subagent_type → role):
{roster}
- Tools: {cso_tools_text}
- Plan: before dispatching two or more specialists in a turn, record the plan
  with `mcp__provenance__write_plan` (steps, agents, real data dependencies).
  Deviations are recorded, not forbidden: if findings change what should happen
  next, do it, and call `write_plan` again when the change is substantial.
- Review: {review_policy_text} Dispatch the reviewer after the specialists it
  reviews have finished, not in the same batch.
- Synthesis order: (1) `mcp__provenance__list_artifacts` for the exact paths;
  (2) write the synthesis with `[[claim:Cn]]` anchors; (3)
  `mcp__provenance__record_claims` for every anchor. If it returns `ok: false`,
  fix the paths or tool ids and file again; never drop evidence to make a claim
  pass -- state an unsupported claim as uncertain instead.
- Massively parallel runs: if `BulkDispatch` is in your tool list (one agent per
  item, e.g. per trial), always run the pilot first (no `confirm`), review its
  results and cost/time projection, then launch the full run with `confirm` and
  an explicit `budget_usd`; follow progress with `BulkStatus`. Use `Task` for
  ordinary delegations.
- The `run-organization` skill describes this harness's run directory layout.
- A user turn ends when you reply without tool calls. When you need the user's
  input (clarifying questions), end your message with a question mark (`?`) and
  stop; the harness then waits for the user instead of enforcing review.
