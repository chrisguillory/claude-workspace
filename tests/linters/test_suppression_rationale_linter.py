#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Validate suppression_rationale_linter.py against its test cases.

This script validates two test files:

1. suppression_rationale_linter_test_cases.py (Instructive)
   - Each scenario tests exactly one rule
   - Used for documentation and teaching

2. suppression_rationale_linter_edge_cases.py (Regression)
   - False positive prevention, string contexts, boundary conditions

Unlike the AST-based linter test runners (which map violations to functions),
this runner uses line-level tags:
    # EXPECT: SUP001     — this line should trigger SUP001
    # EXPECT: SUP005     — this line should trigger SUP005
    # OK                 — this line should NOT trigger any violation
    (no tag)             — line is not tested (infrastructure, blank, etc.)

Run: ./test_suppression_rationale_linter.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping, Set
from dataclasses import dataclass
from pathlib import Path

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


# -- Validation ---------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating a single test file."""

    file_name: str
    errors: list[str]
    expected_count: int
    actual_count: int


def validate_file(test_file: Path) -> ValidationResult:
    """Validate a test file against its EXPECT/OK tags."""
    expected, ok_lines = parse_expectations(test_file)

    output = run_linter(test_file, LINTER)
    actual = parse_linter_output(output)

    errors: list[str] = []

    # Check expected violations fired correctly
    for lineno, expected_code in sorted(expected.items()):
        actual_code = actual.get(lineno)
        if actual_code is None:
            errors.append(f'  line {lineno}: expected {expected_code}, got nothing')
        elif actual_code != expected_code:
            errors.append(f'  line {lineno}: expected {expected_code}, got {actual_code}')

    # Check OK lines produced no violations
    for lineno in sorted(ok_lines):
        actual_code = actual.get(lineno)
        if actual_code is not None:
            source_line = test_file.read_text().splitlines()[lineno - 1].strip()
            errors.append(f'  line {lineno}: expected OK, got {actual_code} ({source_line[:60]})')

    # Check for unexpected violations on untagged lines
    tagged_lines = set(expected.keys()) | ok_lines
    for lineno, actual_code in sorted(actual.items()):
        if lineno not in tagged_lines:
            source_line = test_file.read_text().splitlines()[lineno - 1].strip()
            errors.append(f'  line {lineno}: unexpected {actual_code} (untagged line: {source_line[:60]})')

    return ValidationResult(
        file_name=test_file.name,
        errors=errors,
        expected_count=len(expected),
        actual_count=len(actual),
    )


# -- Main ---------------------------------------------------------------------


def main() -> int:
    """Run validation and return exit code."""
    if not LINTER.exists():
        print(f'ERROR: Linter not found: {LINTER}')
        return 1

    all_passed = True
    results: list[ValidationResult] = []

    for test_file in [TEST_FILE, EDGE_CASE_FILE]:
        if test_file.exists():
            result = validate_file(test_file)
            results.append(result)
            if result.errors:
                all_passed = False
        else:
            print(f'WARNING: Test file not found: {test_file}')
            all_passed = False

    if not all_passed:
        print('VALIDATION FAILED')
        print()
        for result in results:
            if result.errors:
                print(f'  {result.file_name}:')
                for error in result.errors:
                    print(f'    {error}')
        print()
        return 1

    print('VALIDATION PASSED')
    print()
    for result in results:
        print(f'  {result.file_name}:')
        print(f'    Expected violations: {result.expected_count}')
        print(f'    Actual violations: {result.actual_count}')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
