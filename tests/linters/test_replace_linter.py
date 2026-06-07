"""Validate replace_linter.py against its fixtures.

Two fixture categories:

1. replace_linter_test_cases.py (Instructive)
   - ``flag_*`` functions: a model_copy(update=...) that should trigger RPL001
   - ``clean_*`` functions: a model_copy form (or lookalike) that should NOT

2. replace/ structural fixtures (Structural)
   - update-less clones, multi-violation files, nested calls — count-keyed
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Set
from pathlib import Path

import pytest

from tests.linters.helpers import get_def_ranges, run_linter

# -- Configuration ------------------------------------------------------------

TEST_DIR = Path(__file__).parent
CC_DIR = TEST_DIR.parent.parent
LINTER = CC_DIR / 'linters' / 'replace_linter.py'

EDGE_CASES_DIR = TEST_DIR / 'edge_cases'
TEST_FILE = EDGE_CASES_DIR / 'replace_linter_test_cases.py'
STRUCTURAL_DIR = EDGE_CASES_DIR / 'replace'

# Every ``flag_*``/``clean_*`` function in the instructive fixture, read from source so a
# newly added case is auto-covered. Polarity is the name prefix; the test asserts both.
INSTRUCTIVE_CASES: list[str] = sorted(
    name for name in get_def_ranges(TEST_FILE) if name.startswith(('flag_', 'clean_'))
)

# -- Structural Fixtures ------------------------------------------------------

STRUCTURAL_CASES: list[tuple[str, int]] = [
    ('no_update.py', 0),
    ('nested_call.py', 1),
    ('two_updates.py', 2),
]

# -- Module-Scoped Fixtures ---------------------------------------------------


@pytest.fixture(scope='module')
def instructive_flagged() -> Set[str]:
    """Names of fixture functions containing a flagged RPL001 line (linter run once)."""
    output = run_linter(TEST_FILE, LINTER)
    return _flagged_functions(TEST_FILE, output)


# -- Parametrized Tests -------------------------------------------------------


@pytest.mark.parametrize('name', INSTRUCTIVE_CASES)
def test_instructive_case(name: str, instructive_flagged: Set[str]) -> None:
    """``flag_*`` functions trigger RPL001; ``clean_*`` functions do not.

    Polarity is read from the function-name prefix, so a fixture authored with the
    wrong prefix fails loudly rather than silently passing.
    """
    if name.startswith('flag_'):
        assert name in instructive_flagged, f'{name}: expected RPL001 violation but not flagged'
    else:
        assert name not in instructive_flagged, f'{name}: should NOT be flagged but got RPL001'


def test_instructive_no_unexpected(instructive_flagged: Set[str]) -> None:
    """Only ``flag_*`` functions are flagged — no strays."""
    unexpected = {n for n in instructive_flagged if not n.startswith('flag_')}
    assert not unexpected, f'Unexpected RPL001 violations in: {sorted(unexpected)}'


@pytest.mark.parametrize(
    ('filename', 'expected_count'),
    STRUCTURAL_CASES,
    ids=[case[0] for case in STRUCTURAL_CASES],
)
def test_structural_case(filename: str, expected_count: int) -> None:
    """Structural fixtures produce the expected RPL001 violation count."""
    fixture = STRUCTURAL_DIR / filename
    output = run_linter(fixture, LINTER)
    actual = output.count('RPL001')
    assert actual == expected_count, f'{filename}: expected {expected_count} violations, got {actual}'


# -- Linter Output Parsing ----------------------------------------------------


def _flagged_functions(fixture: Path, output: str) -> Set[str]:
    """Map each RPL001 violation line back to the function that encloses it.

    Violations carry no symbol name, so flagged lines are matched against the
    fixture's def ranges instead — robust to line shifts as the fixture grows.
    """
    ranges: Mapping[str, tuple[int, int]] = get_def_ranges(fixture)
    flagged: set[str] = set()
    for line in output.splitlines():
        match = re.search(rf'{re.escape(fixture.name)}:(\d+):\d+: error: RPL001', line)
        if not match:
            continue
        lineno = int(match.group(1))
        for name, (start, end) in ranges.items():
            if start <= lineno <= end:
                flagged.add(name)
    return flagged
