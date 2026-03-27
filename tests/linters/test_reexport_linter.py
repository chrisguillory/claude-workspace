"""Validate reexport_linter.py against its test cases.

Validates three fixture categories:

1. reexport_linter_test_cases.py (Instructive)
   - Re-exports that should flag REX001
   - Local definitions that should NOT flag

2. reexport_linter_edge_cases.py (Regression)
   - TYPE_CHECKING imports, aliases, module imports, type aliases

3. reexport/ structural fixtures (Structural)
   - Empty __all__, star imports, dynamic __all__, annotated/tuple forms
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping, Set
from pathlib import Path

import pytest

# -- Configuration ------------------------------------------------------------

TEST_DIR = Path(__file__).parent
CC_DIR = TEST_DIR.parent.parent
LINTER = CC_DIR / 'linters' / 'reexport_linter.py'

EDGE_CASES_DIR = TEST_DIR / 'edge_cases'
TEST_FILE = EDGE_CASES_DIR / 'reexport_linter_test_cases.py'
EDGE_CASE_FILE = EDGE_CASES_DIR / 'reexport_linter_edge_cases.py'
STRUCTURAL_DIR = EDGE_CASES_DIR / 'reexport'

# -- Expected Violations: Instructive Test File -------------------------------

# Maps __all__ name to whether it should trigger REX001
# True = should flag, False = should NOT flag
EXPECTED_TEST_CASES: Mapping[str, bool] = {
    'join': True,  # from os.path import join
    'OrderedDict': True,  # from collections import OrderedDict
    'json': True,  # import json
    'local_function': False,
    'LocalClass': False,
    'LOCAL_CONSTANT': False,
    'Any': False,  # shadowed by local assignment
}

# -- Expected Violations: Edge Case File --------------------------------------

EXPECTED_EDGE_CASES: Mapping[str, bool] = {
    'Path': True,  # TYPE_CHECKING import — still a re-export
    'path_exists': True,  # aliased import — should flag
    'os': True,  # module import — should flag
    'StringAlias': False,  # type alias — local def
    'SOME_VALUE': False,  # annotated assignment — local def
    'some_submodule': True,  # relative import (from . import X)
}

# -- Structural Fixtures ------------------------------------------------------

STRUCTURAL_CASES: list[tuple[str, int]] = [
    ('empty_all.py', 0),
    ('all_local_only.py', 0),
    ('star_import.py', 0),
    ('dynamic_all.py', 0),
    ('annotated_all.py', 2),
    ('tuple_all.py', 2),
]

# -- Linter Output Parsing ---------------------------------------------------


def run_linter(test_file: Path, linter: Path) -> str:
    """Run the linter and return combined stdout+stderr."""
    result = subprocess.run(
        [sys.executable, str(linter), '--no-skip-file', '--no-config', str(test_file)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.stdout + result.stderr


def parse_flagged_names(output: str) -> Set[str]:
    """Parse linter output to extract names flagged with REX001.

    Extracts the symbol name from the error line itself:
        path/file.py:42:0: error: REX001 'join' re-exported via __all__
    """
    flagged: set[str] = set()

    for line in output.splitlines():
        if 'REX001' in line:
            match = re.search(r"REX001 '([^']+)'", line)
            if match:
                flagged.add(match.group(1))

    return flagged


# -- Module-Scoped Fixtures ---------------------------------------------------


@pytest.fixture(scope='module')
def instructive_output() -> str:
    """Run linter on instructive test file once for all parametrized cases."""
    return run_linter(TEST_FILE, LINTER)


@pytest.fixture(scope='module')
def instructive_flagged(instructive_output: str) -> Set[str]:
    """Parse flagged names from instructive test file output."""
    return parse_flagged_names(instructive_output)


@pytest.fixture(scope='module')
def edge_case_output() -> str:
    """Run linter on edge case file once for all parametrized cases."""
    return run_linter(EDGE_CASE_FILE, LINTER)


@pytest.fixture(scope='module')
def edge_case_flagged(edge_case_output: str) -> Set[str]:
    """Parse flagged names from edge case file output."""
    return parse_flagged_names(edge_case_output)


# -- Parametrized Tests -------------------------------------------------------


@pytest.mark.parametrize(
    ('name', 'should_flag'),
    EXPECTED_TEST_CASES.items(),
    ids=EXPECTED_TEST_CASES.keys(),
)
def test_instructive_case(name: str, should_flag: bool, instructive_flagged: Set[str]) -> None:
    """Each symbol in the instructive fixture is correctly flagged or not."""
    if should_flag:
        assert name in instructive_flagged, f'{name}: expected REX001 violation but not flagged'
    else:
        assert name not in instructive_flagged, f'{name}: should NOT be flagged but got REX001'


def test_instructive_no_unexpected(instructive_flagged: Set[str]) -> None:
    """No unexpected flags in instructive test file."""
    unexpected = instructive_flagged - set(EXPECTED_TEST_CASES.keys())
    assert not unexpected, f'Unexpected REX001 violations: {sorted(unexpected)}'


@pytest.mark.parametrize(
    ('name', 'should_flag'),
    EXPECTED_EDGE_CASES.items(),
    ids=EXPECTED_EDGE_CASES.keys(),
)
def test_edge_case(name: str, should_flag: bool, edge_case_flagged: Set[str]) -> None:
    """Each symbol in the edge case fixture is correctly flagged or not."""
    if should_flag:
        assert name in edge_case_flagged, f'{name}: expected REX001 violation but not flagged'
    else:
        assert name not in edge_case_flagged, f'{name}: should NOT be flagged but got REX001'


def test_edge_case_no_unexpected(edge_case_flagged: Set[str]) -> None:
    """No unexpected flags in edge case file."""
    unexpected = edge_case_flagged - set(EXPECTED_EDGE_CASES.keys())
    assert not unexpected, f'Unexpected REX001 violations: {sorted(unexpected)}'


@pytest.mark.parametrize(
    ('filename', 'expected_count'),
    STRUCTURAL_CASES,
    ids=[case[0] for case in STRUCTURAL_CASES],
)
def test_structural_case(filename: str, expected_count: int) -> None:
    """Structural edge cases produce the expected violation count."""
    fixture = STRUCTURAL_DIR / filename
    output = run_linter(fixture, LINTER)
    actual = output.count('REX001')
    assert actual == expected_count, f'{filename}: expected {expected_count} violations, got {actual}'
