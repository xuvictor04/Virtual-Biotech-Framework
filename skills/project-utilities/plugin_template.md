# Plugin template (existing kinds only)

Plugin kinds: `format` (read a file format), `layout` (where a table's files are), `statistic` (what a measure's
values mean: thresholds, ordering, aggregation), `identifier` (an ID scheme: normalisation, validation),
`envelope` (decoding an MCP server's result), `acquisition` (a download transport). A new **kind** is a core change:
ask the CSO to report it instead.

The base classes and protocols are in `vbt/datalayer/plugins/base.py`; the shipped plugins under
`vbt/datalayer/plugins/<kind>s/` are worked examples, and each kind's conformance suite is
`vbt/datalayer/plugins/conformance/<kind>.py` (the cases your plugin must pass; a plugin may supply its own
`conformance_cases()`).

## A statistic plugin

```python
"""``score_0_10``: a bounded 0-10 score, higher is stronger (an in-house assay's quality score)."""

from typing import ClassVar

from vbt.datalayer.plugins.registry import register
from vbt.datalayer.plugins.statistics.score import Score01Statistic


@register
class Score010Statistic(Score01Statistic):
    name: ClassVar[str] = "score_0_10"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, 10.0)
```

Register it with `RegisterPlugin(kind="statistic", path="plugins/score_0_10.py", why=...)`. The suite runs in the
sandbox, restricted to your plugin; every case must pass. Once registered, descriptors registered afterwards may
use it (`statistic: score_0_10`), and the project's profile lists it in `data.plugins.paths`.

Rules: one `@register` class per module; `name` and `version` are string literals; the name must not be a shipped
plugin's name; no network, file writes or global state at import time.
