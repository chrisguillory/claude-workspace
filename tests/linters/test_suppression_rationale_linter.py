"""Validate suppression_rationale_linter.py against its test cases.

Validates two test files:

1. suppression_rationale_linter_test_cases.py (Instructive)
   - Each scenario tests exactly one rule
   - Used for documentation and teaching

2. suppression_rationale_linter_edge_cases.py (Regression)
   - False positive prevention, string contexts, boundary conditions

Uses line-level tags:
    # EXPECT: SUP001     -- this line should trigger SUP001
    # EXPECT: SUP005     -- this line should trigger SUP005
    # OK                 -- this line should NOT trigger any violation
    (no tag)             -- line is not tested (infrastructure, blank, etc.)
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
LINTER = CC_DIR / 'linters' / 'suppression_rationale_linter.py'

EDGE_CASES_DIR = TEST_DIR / 'edge_cases'
TEST_FILE = EDGE_CASES_DIR / 'suppression_rationale_linter_test_cases.py'
EDGE_CASE_FILE = EDGE_CASES_DIR / 'suppression_rationale_linter_edge_cases.py'

# -- Linter Output Parsing ----------------------------------------------------


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


def parse_linter_output(output: str) -> Mapping[int, str]:
    """Parse linter output to extract {line_number: rule_code} mapping.

    If multiple violations on the same line, keeps the first.
    """
    violations: dict[int, str] = {}
    pattern = re.compile(r'.+:(\d+):\d+: error: (SUP\d+)')

    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            line_num = int(match.group(1))
            rule_code = match.group(2)
            if line_num not in violations:
                violations[line_num] = rule_code

    return violations


# -- Test File Parsing ---------------------------------------------------------

EXPECT_RE = re.compile(r'#\s*EXPECT:\s*(SUP\d+)')
OK_RE = re.compile(r'#\s*OK\b')


def parse_expectations(filepath: Path) -> tuple[Mapping[int, str], Set[int]]:
    """Parse test file for EXPECT and OK tags.

    Tags apply to the NEXT non-blank line after the tag. This avoids tags
    interfering with the linter's separator detection (a ``# EXPECT:`` on the
    same line would be parsed as rationale).

    Returns:
        (expected_violations, ok_lines) where:
        - expected_violations: {line_number: expected_rule_code}
        - ok_lines: set of line numbers that should produce no violation
    """
    expected: dict[int, str] = {}
    ok_lines: set[int] = set()
    lines = filepath.read_text().splitlines()

    pending_expect: str | None = None
    pending_ok = False

    for i, line in enumerate(lines):
        lineno = i + 1
        stripped = line.strip()

        expect_match = EXPECT_RE.search(stripped)
        if expect_match:
            pending_expect = expect_match.group(1)
            pending_ok = False
            continue

        if OK_RE.search(stripped) and stripped.startswith('# OK'):
            pending_ok = True
            pending_expect = None
            continue

        # Associate pending tag with this line (skip blank lines)
        if not stripped:
            continue

        if pending_expect:
            expected[lineno] = pending_expect
            pending_expect = None
        elif pending_ok:
            ok_lines.add(lineno)
            pending_ok = False

    return expected, ok_lines


def _get_source_line(filepath: Path, lineno: int) -> str:
    """Read a single source line (1-indexed) from a file."""
    return filepath.read_text().splitlines()[lineno - 1].strip()


# -- Module-Scoped Fixtures ---------------------------------------------------


# Pre-parse expectations at module level (used by parametrize decorators AND fixtures)
_INSTRUCTIVE_EXPECTED, _INSTRUCTIVE_OK = parse_expectations(TEST_FILE)
_EDGE_CASE_EXPECTED, _EDGE_CASE_OK = parse_expectations(EDGE_CASE_FILE)


@pytest.fixture(scope='module')
def instructive_actual() -> Mapping[int, str]:
    """Run linter on instructive file, return actual violations."""
    output = run_linter(TEST_FILE, LINTER)
    return parse_linter_output(output)


@pytest.fixture(scope='module')
def edge_case_actual() -> Mapping[int, str]:
    """Run linter on edge case file, return actual violations."""
    output = run_linter(EDGE_CASE_FILE, LINTER)
    return parse_linter_output(output)


def _expect_id(lineno: int, code: str) -> str:
    """Generate a readable test ID for an EXPECT tag."""
    return f'line{lineno}_{code}'


def _ok_id(lineno: int) -> str:
    """Generate a readable test ID for an OK tag."""
    return f'line{lineno}_OK'


# -- Parametrized Tests: Instructive ------------------------------------------


@pytest.mark.parametrize(
    ('lineno', 'expected_code'),
    _INSTRUCTIVE_EXPECTED.items(),
    ids=[_expect_id(ln, code) for ln, code in _INSTRUCTIVE_EXPECTED.items()],
)
def test_instructive_expected(
    lineno: int, expected_code: str, instructive_actual: Mapping[int, str]
) -> None:
    """Each EXPECT-tagged line triggers the expected rule."""
    actual = instructive_actual
    actual_code = actual.get(lineno)
    assert actual_code is not None, f'line {lineno}: expected {expected_code}, got nothing'
    assert actual_code == expected_code, f'line {lineno}: expected {expected_code}, got {actual_code}'


@pytest.mark.parametrize(
    'lineno',
    sorted(_INSTRUCTIVE_OK),
    ids=[_ok_id(ln) for ln in sorted(_INSTRUCTIVE_OK)],
)
def test_instructive_ok(lineno: int, instructive_actual: Mapping[int, str]) -> None:
    """Each OK-tagged line produces no violation."""
    actual = instructive_actual
    actual_code = actual.get(lineno)
    if actual_code is not None:
        source_line = _get_source_line(TEST_FILE, lineno)
        raise AssertionError(f'line {lineno}: expected OK, got {actual_code} ({source_line[:60]})')


def test_instructive_no_unexpected(
    instructive_actual: Mapping[int, str],
) -> None:
    """No violations on untagged lines in the instructive file."""
    actual = instructive_actual
    tagged_lines = set(_INSTRUCTIVE_EXPECTED.keys()) | _INSTRUCTIVE_OK
    unexpected = []
    for lineno, actual_code in sorted(actual.items()):
        if lineno not in tagged_lines:
            source_line = _get_source_line(TEST_FILE, lineno)
            unexpected.append(f'line {lineno}: unexpected {actual_code} (untagged: {source_line[:60]})')
    assert not unexpected, '\n'.join(unexpected)


# -- Parametrized Tests: Edge Cases -------------------------------------------


@pytest.mark.parametrize(
    ('lineno', 'expected_code'),
    _EDGE_CASE_EXPECTED.items(),
    ids=[_expect_id(ln, code) for ln, code in _EDGE_CASE_EXPECTED.items()],
)
def test_edge_case_expected(
    lineno: int, expected_code: str, edge_case_actual: Mapping[int, str]
) -> None:
    """Each EXPECT-tagged line triggers the expected rule."""
    actual = edge_case_actual
    actual_code = actual.get(lineno)
    assert actual_code is not None, f'line {lineno}: expected {expected_code}, got nothing'
    assert actual_code == expected_code, f'line {lineno}: expected {expected_code}, got {actual_code}'


@pytest.mark.parametrize(
    'lineno',
    sorted(_EDGE_CASE_OK),
    ids=[_ok_id(ln) for ln in sorted(_EDGE_CASE_OK)],
)
def test_edge_case_ok(lineno: int, edge_case_actual: Mapping[int, str]) -> None:
    """Each OK-tagged line produces no violation."""
    actual = edge_case_actual
    actual_code = actual.get(lineno)
    if actual_code is not None:
        source_line = _get_source_line(EDGE_CASE_FILE, lineno)
        raise AssertionError(f'line {lineno}: expected OK, got {actual_code} ({source_line[:60]})')


def test_edge_case_no_unexpected(
    edge_case_actual: Mapping[int, str],
) -> None:
    """No violations on untagged lines in the edge case file."""
    actual = edge_case_actual
    tagged_lines = set(_EDGE_CASE_EXPECTED.keys()) | _EDGE_CASE_OK
    unexpected = []
    for lineno, actual_code in sorted(actual.items()):
        if lineno not in tagged_lines:
            source_line = _get_source_line(EDGE_CASE_FILE, lineno)
            unexpected.append(f'line {lineno}: unexpected {actual_code} (untagged: {source_line[:60]})')
    assert not unexpected, '\n'.join(unexpected)
