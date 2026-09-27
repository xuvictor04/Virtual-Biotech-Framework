import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vbt.config import load_config  # noqa: E402


@pytest.fixture
def config(tmp_path):
    return load_config(["mock"], overrides={"paths": {"runs_dir": str(tmp_path / "runs")}})
