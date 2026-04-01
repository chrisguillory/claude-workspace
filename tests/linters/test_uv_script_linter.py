"""Validate uv_script_linter.py against fixture files.

Tests both positive cases (should pass) and negative cases (should flag)
using fixture files in edge_cases/uv_script/*.py.

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
LINTER = CC_DIR / 'linters' / 'uv_script_linter.py'
EDGE_CASES_DIR = TEST_DIR / 'edge_cases' / 'uv_script'


def run_linter(*paths: Path, no_config: bool = False) -> subprocess.CompletedProcess[str]:
    """Run uv_script_linter.py on one or more paths."""
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


# -- UVS001: Should pass (exit 0) ---------------------------------------------


@pytest.mark.parametrize(
    'fixture',
    [
        'good_uv_run.py',
        'good_uv_run_script.py',
        'good_no_shebang.py',
        'good_comment_python3.py',
    ],
    ids=lambda f: f.removesuffix('.py'),
)
def test_uvs001_good(fixture: str) -> None:
    """Files with valid shebangs (or no shebang) should pass."""
    result = run_linter(EDGE_CASES_DIR / fixture, no_config=True)
    assert result.returncode == 0, f'Expected pass, got:\n{result.stdout}'


# -- UVS001: Should fail (exit 1) ---------------------------------------------


@pytest.mark.parametrize(
    'fixture',
    [
        'bad_python3.py',
        'bad_python_bare.py',
        'bad_absolute_path.py',
    ],
    ids=lambda f: f.removesuffix('.py'),
)
def test_uvs001_bad(fixture: str) -> None:
    """Files with bare python shebangs should flag UVS001."""
    result = run_linter(EDGE_CASES_DIR / fixture, no_config=True)
    assert result.returncode == 1, f'Expected failure, got:\n{result.stdout}'
    assert 'UVS001' in result.stdout
    assert 'should use uv run' in result.stdout


# -- UVS002: Should pass (exit 0) ---------------------------------------------


@pytest.mark.parametrize(
    'fixture',
    [
        'good_script_flag.py',
        'good_no_project_script.py',
        'good_uv_no_metadata.py',
        'good_non_script_block.py',
    ],
    ids=lambda f: f.removesuffix('.py'),
)
def test_uvs002_good(fixture: str) -> None:
    """Files with --script or no PEP 723 block should pass."""
    result = run_linter(EDGE_CASES_DIR / fixture, no_config=True)
    assert result.returncode == 0, f'Expected pass, got:\n{result.stdout}'


# -- UVS002: Should fail (exit 1) ---------------------------------------------


@pytest.mark.parametrize(
    'fixture',
    [
        'bad_no_script_with_metadata.py',
        'bad_bare_uv_with_metadata.py',
        'bad_quiet_no_script.py',
    ],
    ids=lambda f: f.removesuffix('.py'),
)
def test_uvs002_bad(fixture: str) -> None:
    """Files with uv run + PEP 723 metadata but no --script should flag UVS002."""
    result = run_linter(EDGE_CASES_DIR / fixture, no_config=True)
    assert result.returncode == 1, f'Expected failure, got:\n{result.stdout}'
    assert 'UVS002' in result.stdout
    assert '--script' in result.stdout


# -- UVS003: Should pass (exit 0) ---------------------------------------------


@pytest.mark.parametrize(
    'fixture',
    [
        'good_empty_deps.py',
        'good_multiline_deps.py',
        'good_single_dep_multiline.py',
        'good_no_deps_key.py',
    ],
    ids=lambda f: f.removesuffix('.py'),
)
def test_uvs003_good(fixture: str) -> None:
    """Files with properly formatted deps should pass."""
    result = run_linter(EDGE_CASES_DIR / fixture, no_config=True)
    assert result.returncode == 0, f'Expected pass, got:\n{result.stdout}'


# -- UVS003: Should fail (exit 1) ---------------------------------------------


@pytest.mark.parametrize(
    'fixture',
    [
        'bad_single_line_deps.py',
        'bad_multi_deps_one_line.py',
        'bad_unsorted_deps.py',
        'bad_no_trailing_comma.py',
    ],
    ids=lambda f: f.removesuffix('.py'),
)
def test_uvs003_bad(fixture: str) -> None:
    """Files with improperly formatted deps should flag UVS003."""
    result = run_linter(EDGE_CASES_DIR / fixture, no_config=True)
    assert result.returncode == 1, f'Expected failure, got:\n{result.stdout}'
    assert 'UVS003' in result.stdout


# -- Whole-repo scan ----------------------------------------------------------


def test_workspace_clean() -> None:
    """The entire workspace should have zero violations.

    Bad fixtures in edge_cases/ are excluded via per-file-ignores
    in pyproject.toml ([tool.uv-script-linter.per-file-ignores]).
    """
    result = run_linter(CC_DIR)
    assert result.returncode == 0, f'Workspace has violations:\n{result.stdout}'
