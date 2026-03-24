#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Validate reexport_linter.py against its test cases.

This script validates two test files:

1. reexport_linter_test_cases.py (Instructive)
   - Re-exports that should flag REX001
   - Local definitions that should NOT flag

2. reexport_linter_edge_cases.py (Regression)
   - TYPE_CHECKING imports, aliases, module imports, type aliases
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path

# -- Configuration ------------------------------------------------------------

TEST_DIR = Path(__file__).parent
CC_DIR = TEST_DIR.parent.parent
LINTER = CC_DIR / 'linters' / 'reexport_linter.py'

EDGE_CASES_DIR = TEST_DIR / 'edge_cases'
TEST_FILE = EDGE_CASES_DIR / 'reexport_linter_test_cases.py'
EDGE_CASE_FILE = EDGE_CASES_DIR / 'reexport_linter_edge_cases.py'

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
}


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

    Expects lines like:
        path/file.py:42:0: error: REX001 Symbol imported from ...
            'join',
    The source_line contains the __all__ entry with the symbol name.
    """
    flagged: set[str] = set()

    # Pattern: Match the source line that follows the error line
    # The source line contains the symbol name in quotes
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if 'REX001' in line:
            # Next line is the indented source line containing the symbol
            if i + 1 < len(lines):
                source_line = lines[i + 1].strip()
                # Extract name from patterns like: 'join', or 'join'
                match = re.search(r"'([^']+)'", source_line)
                if match:
                    flagged.add(match.group(1))

    return flagged


# -- Validation ---------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating a single test file."""

    file_name: str
    errors: Sequence[str]
    flagged_count: int
    expected_flag_count: int


def validate_file(
    test_file: Path,
    expected: Mapping[str, bool],
) -> ValidationResult:
    """Validate a single test file against expected violations."""
    output = run_linter(test_file, LINTER)
    flagged_names = parse_flagged_names(output)

    errors: list[str] = []
    expected_flag_count = 0

    for name, should_flag in expected.items():
        if should_flag:
            expected_flag_count += 1
            if name not in flagged_names:
                errors.append(f'{name}: expected REX001 violation, but not flagged')
        elif name in flagged_names:
            errors.append(f'{name}: should NOT be flagged, but got REX001')

    # Check for unexpected flags (names not in expected map)
    errors.extend(
        f'{name}: unexpected REX001 violation (not in expected map)' for name in flagged_names if name not in expected
    )

    return ValidationResult(
        file_name=test_file.name,
        errors=errors,
        flagged_count=len(flagged_names),
        expected_flag_count=expected_flag_count,
    )


# -- Main ---------------------------------------------------------------------


def main() -> int:
    """Run validation and return exit code."""
    if not LINTER.exists():
        print(f'ERROR: Linter not found: {LINTER}')
        return 1

    all_passed = True
    results: list[ValidationResult] = []

    # Validate instructive test file
    if TEST_FILE.exists():
        result = validate_file(TEST_FILE, EXPECTED_TEST_CASES)
        results.append(result)
        if result.errors:
            all_passed = False
    else:
        print(f'WARNING: Test file not found: {TEST_FILE}')
        all_passed = False

    # Validate edge case file
    if EDGE_CASE_FILE.exists():
        result = validate_file(EDGE_CASE_FILE, EXPECTED_EDGE_CASES)
        results.append(result)
        if result.errors:
            all_passed = False
    else:
        print(f'WARNING: Edge case file not found: {EDGE_CASE_FILE}')
        all_passed = False

    # Report results
    if not all_passed:
        print('VALIDATION FAILED')
        print()
        for result in results:
            if result.errors:
                print(f'  {result.file_name}:')
                for error in sorted(result.errors):
                    print(f'    - {error}')
        print()
        return 1

    # Success summary
    print('VALIDATION PASSED')
    print()

    for result in results:
        print(f'  {result.file_name}:')
        print(f'    Expected violations: {result.expected_flag_count}')
        print(f'    Actual violations: {result.flagged_count}')
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
