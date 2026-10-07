"""The upstream authors' code is launched, never edited (§21 Architecture).

* ``git status --porcelain`` of ``third_party/TheVirtualBiotech`` is empty;
* its HEAD is the commit the superproject records for the submodule;
* no ``__pycache__`` appeared under it during this test session (servers run with ``-B`` and
  ``PYTHONDONTWRITEBYTECODE=1``; harness code importing upstream modules must not write either).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from dl_upstream import REPO, UPSTREAM, UPSTREAM_PYCACHE_AT_START, upstream_pycache

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed"),
    pytest.mark.skipif(not (UPSTREAM / ".git").exists(), reason="the upstream submodule is not checked out"),
]


def git(*args: str, cwd=REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def test_upstream_worktree_is_clean() -> None:
    assert git("status", "--porcelain", cwd=UPSTREAM) == ""


def test_upstream_head_is_the_recorded_commit() -> None:
    recorded = git("ls-tree", "HEAD", "third_party/TheVirtualBiotech").split()
    if not recorded:
        pytest.skip("the superproject records no submodule commit")
    assert recorded[1] == "commit", recorded
    assert git("rev-parse", "HEAD", cwd=UPSTREAM).strip() == recorded[2]


def test_no_new_pycache_under_upstream() -> None:
    new = sorted(upstream_pycache() - UPSTREAM_PYCACHE_AT_START)
    assert new == [], f"__pycache__ written into the upstream checkout during this session: {new}"
