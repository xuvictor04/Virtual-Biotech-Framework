"""``python -m vbt.projects init|list|show|check|profile|approve|reject|memory ...`` (see :mod:`vbt.projects.cli`)."""

import sys

from .cli import main

sys.exit(main())
