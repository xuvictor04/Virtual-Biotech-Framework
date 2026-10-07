# Derived tool snapshots (docs/DATA_LAYER.md §10.1, §10.2, F13)

One JSON file per bridged tool, `<server>.<tool>.json` (103 files):

```json
{"tool": "drug.search_known_drugs", "serve": "pass", "schema": {...}, "description": "..."}
```

* `schema` is the upstream argument schema (parsed from the unmodified function signature: types,
  defaults, required arguments) annotated by `vbt.datalayer.derive.annotate_schema` from the shipped
  descriptors and overlays: `x-vbt-id-type`, `x-vbt-accepts`, examples, enums, threshold bounds, limit
  bounds, gateway-only scope arguments, `x-vbt-require-any`/`x-vbt-exclusive`.
* `description` is what agents read: `vbt.datalayer.derive.describe_tool` on the upstream docstring
  (first sentence plus the generated block: source, grain, key, arguments, order, totals, levels,
  cutoffs, families, censoring, propagation, lossy projections, evidence nature, unapplied default
  filters, coverage), capped at `data.derive.description_max_chars`. It is exactly the `derived text`
  of `vbt ds explain <server>.<tool>` for the same docstring and schema.

`tests/datalayer/test_dl_derive_all_tools.py` rebuilds every snapshot and compares. The upstream
checkout must be present (`git submodule update --init`, or `VBT_UPSTREAM=<checkout>`); without it the
snapshot tests skip.

## Updating snapshots

A failing snapshot means the derivation, a descriptor, an overlay or an upstream signature changed.
Regenerate and review the diff like code:

```bash
VBT_UPDATE_GOLDEN=1 python -m pytest -q tests/datalayer/test_dl_derive_all_tools.py
git diff tests/datalayer/golden/
```

Review checklist for each changed file:

1. The change is intended (name the descriptor, overlay or derivation change that caused it).
2. No identifier argument lost `x-vbt-id-type`/`x-vbt-accepts`, and no resolvable kind gained a strict
   `pattern` (symbols must reach the resolver).
3. Enums, bounds and defaults still match the data (a vocabulary enum lists only observed values).
4. The description still states what one record is, what `_vbt.total` counts, how empty differs from
   not found, and every caveat the roles call for; it promises no field that does not exist.
5. A blocked tool still starts with `UNAVAILABLE:` and names an alternative that is not blocked.
6. The description fits the cap (the test enforces it); if the cap now cuts an important sentence,
   shorten the overlay's `text.summary`/`notes` rather than raising the cap.

Commit the regenerated snapshots with the change that caused them, never separately.
