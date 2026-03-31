"""Validate shebang_check.py against fixture files.

Tests both positive cases (should pass) and negative cases (should flag)
using fixture files in edge_cases/shebang_*.py.

Uses --no-config for fixture tests to bypass per-file-ignores (same
pattern as test_suppression_rationale_linter.py and siblings).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
CC_DIR = TEST_DIR.parent.parent
LINTER = CC_DIR / 'linters' / 'shebang_check.py'
EDGE_CASES_DIR = TEST_DIR / 'edge_cases'


def run_linter(*paths: Path, no_config: bool = False) -> subprocess.CompletedProcess[str]:
    """Run shebang_check.py on one or more paths."""
    cmd = [sys.executable, str(LINTER)]
    if no_config:
        cmd.append('--no-config')
    cmd.extend(str(p) for p in paths)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


# -- Should pass (exit 0) ----------------------------------------------------


@pytest.mark.parametrize(
    'fixture',
    [
        'shebang_good_uv_run.py',
        'shebang_good_uv_run_script.py',
        'shebang_good_no_shebang.py',
        'shebang_good_comment_python3.py',
    ],
    ids=lambda f: f.removesuffix('.py'),
)
def test_good_shebangs(fixture: str) -> None:
    """Files with valid shebangs (or no shebang) should pass."""
    result = run_linter(EDGE_CASES_DIR / fixture, no_config=True)
    assert result.returncode == 0, f'Expected pass, got:\n{result.stdout}'


# -- Should fail (exit 1) ----------------------------------------------------


@pytest.mark.parametrize(
    'fixture',
    [
        'shebang_bad_python3.py',
        'shebang_bad_python_bare.py',
        'shebang_bad_absolute_path.py',
    ],
    ids=lambda f: f.removesuffix('.py'),
)
def test_bad_shebangs(fixture: str) -> None:
    """Files with bare python shebangs should flag."""
    result = run_linter(EDGE_CASES_DIR / fixture, no_config=True)
    assert result.returncode == 1, f'Expected failure, got:\n{result.stdout}'
    assert 'should use uv run' in result.stdout


# -- Whole-repo scan ----------------------------------------------------------


def test_workspace_clean() -> None:
    """The entire workspace should have zero shebang violations.

    Bad fixtures in edge_cases/ are excluded via per-file-ignores
    in pyproject.toml ([tool.shebang-check.per-file-ignores]).
    """
    result = run_linter(CC_DIR)
    assert result.returncode == 0, f'Workspace has shebang violations:\n{result.stdout}'
