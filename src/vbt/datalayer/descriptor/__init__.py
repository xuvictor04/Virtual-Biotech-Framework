"""Descriptor and overlay schemas, reference scoping, loading and lint (§6, §8). No pyarrow.

* ``models.py``  - source descriptors (``vbt.datasource/1``) and every sub-model;
* ``columns.py`` - ``ColumnSpec``, the union of one model per role (§7);
* ``overlay.py`` - overlays (``vbt.overlay/1``): tool bindings, arguments, result field maps;
* ``scoping.py`` - reference scoping (sibling -> enclosing item -> table; ``^.`` and ``/``);
* ``load.py``    - YAML loading with ``vbt.config`` expansion and ``${run.*}``, digests;
* ``lint.py``    - ``lint_descriptor`` and ``lint_overlay`` (§6.9).
"""

from __future__ import annotations

from .columns import ColumnSpec, validate_column
from .load import DescriptorError, digest, load_descriptor, load_descriptors, load_overlay, load_overlays
from .models import SourceDescriptor, TableSpec
from .overlay import ArgBinding, Overlay, ResultSpec, ToolBinding

__all__ = [
    "ColumnSpec", "validate_column", "SourceDescriptor", "TableSpec", "Overlay", "ToolBinding", "ArgBinding",
    "ResultSpec", "DescriptorError", "digest", "load_descriptor", "load_descriptors", "load_overlay",
    "load_overlays",
]
