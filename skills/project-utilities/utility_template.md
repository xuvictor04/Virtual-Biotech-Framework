# Utility template

A utility is one directory in your workspace, registered with
`RegisterUtility(name=..., description=..., input_schema=..., directory="utilities/<name>", why=...)`:

```
utilities/<name>/
  utility.py          the code: one entry function (default `run`) with a docstring
  test_utility.py     test_* functions (plain asserts); `import utility`
  fixtures/           optional small test data (read it relative to the test file)
```

## utility.py

```python
"""Summaries of IC50 measurements."""

import math


def run(values: list, min_value: float = 0.0) -> dict:
    """Geometric mean of the IC50 values above ``min_value`` (nM).

    Args:
        values: IC50 values in nM; None and non-positive values are ignored.
        min_value: values at or below it are ignored.
    Returns:
        {"n": number of values used, "geomean": the geometric mean in nM, or None when n == 0}.
    """
    vals = [float(v) for v in values if v is not None and float(v) > max(min_value, 0.0)]
    if not vals:
        return {"n": 0, "geomean": None}
    return {"n": len(vals), "geomean": math.exp(sum(math.log(v) for v in vals) / len(vals))}
```

## test_utility.py

```python
from pathlib import Path

import utility

FIXTURES = Path(__file__).parent / "fixtures"


def test_known_answer():
    assert abs(utility.run([1, 100])["geomean"] - 10.0) < 1e-9


def test_empty_input_is_not_zero():
    assert utility.run([None, -1]) == {"n": 0, "geomean": None}


def test_writes_only_where_allowed(tmp_path):
    (tmp_path / "out.txt").write_text("ok")      # tmp_path is a fresh writable directory
```

## input_schema

```json
{"type": "object",
 "properties": {"values": {"type": "array", "items": {"type": ["number", "null"]}},
                "min_value": {"type": "number"}},
 "required": ["values"]}
```

## Rules (checked before the tests run)

- `name`: lowercase identifier; the tool is `util__<name>`.
- `description`: at least 20 characters; it is the tool's description for every specialist.
- The entry function has a docstring; every parameter without a default is in `required`; every schema property
  is a parameter (or the function takes `**kwargs`).
- At least one `test_*` function, and all of them pass in the sandbox: no network, memory-limited, read-only
  outside the working directory, no package installs. Tests must not modify `utility.py`.
- The function returns JSON-serialisable data (dicts, lists, numbers, strings). Large outputs: write a file into
  the caller's working directory (the current directory when the tool runs) and return its path.
- Script mode (`mode="script"`): `utility.py` reads the JSON arguments from stdin and prints its result; it needs a
  module docstring.
- To read a registered project table inside a utility, use the data client (`from vbt.datalayer.client import
  ...`); never hard-code paths of the project's data files.
