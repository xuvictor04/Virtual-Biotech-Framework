# Data tools: how to read and cite their results

Every `mcp__*` data result starts with a `_vbt` header (the first JSON key, or the first line of a text result):

- `status`: `ok` (rows), `partial` (some rows: truncated, total unknown, a section unavailable or rows withheld),
  `empty` (the entity exists and has no rows in this source) or `empty_unverified` (zero rows that could not be
  confirmed: never citable).
- `returned`, `total`, `truncated` and `order`: cite a partial result as "top N of M" only when `order` says the
  ranking was verified; `grains` counts entities (drugs, patients) where rows are records.
- `coverage` (`covered`, `unknown`, `not_covered`, `partial_unknown`, `censored`) says what an empty result means;
  `evidence` names the table's caveat (e.g. literature co-occurrence, not function); `prov` is the provenance id.
- An error (`not_found`, `ambiguous`, `invalid_argument`, `incomplete_key`, `not_ready`, `tool_defect`, ...) is not
  evidence about biology: fix the input (see its `suggestions`, `valid_values` or `retry_with`) or report the gap.
- An `empty` result supports only an absence finding: cite it with `supports: "absence"`, and only when `coverage`
  is `covered`. A `censored` empty result is citable only with its coverage statement in the claim text.
- Read `_vbt.cite` and `_vbt.evidence` before citing a result, and state counts at their grain.
