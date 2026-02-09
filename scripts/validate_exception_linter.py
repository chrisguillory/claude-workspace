#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Validate exception_safety_linter.py against its test cases.

This script validates two test files:

1. exception_safety_test_cases.py (Instructive)
   - Each violation function triggers exactly one rule (no pollution)
   - Used for documentation and teaching

2. exception_safety_edge_cases.py (Regression)
   - Edge cases and false positive prevention
   - Comprehensive coverage of all linter code paths

Run: ./validate_exception_linter.py
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


class LineRange(NamedTuple):
    """Line range for a function definition (inclusive)."""

    start: int
    end: int


# Maps function name to set of violation codes found/expected in that function
type ViolationMap = dict[str, set[str]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
LINTER = SCRIPT_DIR / 'exception_safety_linter.py'

# Test file definitions
TEST_FILE = SCRIPT_DIR / 'exception_safety_test_cases.py'
EDGE_CASE_FILE = SCRIPT_DIR / 'exception_safety_edge_cases.py'

# ---------------------------------------------------------------------------
# Expected Violations: Instructive Test File
# ---------------------------------------------------------------------------

# Each violation function triggers exactly one rule (no pollution)
EXPECTED_VIOLATIONS: ViolationMap = {
    # EXC001: Bare except
    'exc001_violation_basic': {'EXC001'},
    # EXC002: Swallowed exception (broad catch without re-raise)
    'exc002_violation_basic': {'EXC002'},
    'exc002_violation_base_exception': {'EXC002'},
    # EXC003: Control flow in finally
    'exc003_violation_return': {'EXC003'},
    'exc003_violation_break': {'EXC003'},
    'exc003_violation_continue': {'EXC003'},
    # EXC004: Raise without from
    'exc004_violation_basic': {'EXC004'},
    # EXC005: Unused exception variable
    'exc005_violation_basic': {'EXC005'},
    # EXC006: Logger without exc_info
    'exc006_violation_basic': {'EXC006'},
    'exc006_violation_critical': {'EXC006'},
    # EXC007: CancelledError not raised
    'exc007_violation_basic': {'EXC007'},
    'exc007_violation_return_in_worker': {'EXC007'},
    # EXC008: GeneratorExit not raised
    'exc008_violation_basic': {'EXC008'},
}

# Functions with suppression directives (should NOT appear in linter output)
SUPPRESSED_FUNCTIONS: set[str] = {
    'exc002_suppressed_intentional',
    'exc003_suppressed_top_level',
    'exc006_correct_when_suppressing',  # Has EXC002 suppression
}

# ---------------------------------------------------------------------------
# Expected Violations: Edge Case File
# ---------------------------------------------------------------------------

# Edge cases for comprehensive testing. May have multiple rules or test
# false positive prevention (set() means expect NO violations).
EXPECTED_EDGE_CASES: ViolationMap = {
    # EXC002: Tuple exception handling
    'edge_tuple_with_broad_exception': {'EXC002'},
    # EXC002: Non-canonical patterns (without pass)
    'edge_swallowed_with_return': {'EXC002'},
    'edge_swallowed_with_action': {'EXC002'},
    # EXC003: Finally edge cases
    'edge_finally_with_raise': set(),  # raise in finally is NOT EXC003
    # EXC004: False positive prevention (should NOT trigger)
    'edge_raise_caught_variable': set(),  # Re-raising caught var is OK
    'edge_raise_caught_with_explicit_chain': set(),  # With 'from' is OK
    # Outer scope raise: EXC004 (unclear chain) + EXC002 (outer handler has no direct raise)
    'edge_raise_outer_scope_variable': {'EXC002', 'EXC004'},
    # EXC006: All logger methods
    'edge_logger_fatal': {'EXC006'},
    'edge_logger_warning': {'EXC006'},
    'edge_logger_warn': {'EXC006'},
    'edge_logger_exc_info_false': {'EXC006'},  # False is not True
    'edge_logger_info_no_violation': set(),  # info() is not flagged
    # EXC007: BaseException in async (also triggers EXC002 - related rules)
    'edge_async_base_exception_no_raise': {'EXC002', 'EXC007'},
    'edge_async_base_exception_with_raise': set(),  # With raise is OK
    # EXC007: CancelledError variants
    'edge_cancelled_error_short_name': {'EXC007'},  # Short name import
    'edge_cancelled_error_aliased': {'EXC007'},  # Aliased import
    'edge_cancelled_error_in_tuple': {'EXC007'},  # CancelledError in tuple
    'edge_cancelled_error_with_return': {'EXC007'},  # return instead of raise
    'edge_sync_nested_in_async': set(),  # Sync nested function not subject to EXC007
    # EXC008: GeneratorExit variants
    'edge_generator_base_exception_no_raise': {'EXC002', 'EXC008'},
    'edge_generator_base_exception_with_raise': set(),
    'edge_generator_exit_in_tuple': {'EXC008'},
    'edge_generator_exit_with_return': {'EXC008'},
    'edge_async_generator_generator_exit': {'EXC008'},
    'edge_async_generator_both_violations': {'EXC002', 'EXC007', 'EXC008'},
    'edge_sync_nested_in_generator': set(),
    'edge_regular_function_generator_exit': set(),
    'edge_generator_expression_not_generator': set(),
    'edge_yield_from_is_generator': {'EXC008'},
    # TryStar (except*)
    'edge_trystar_no_raise': {'EXC002'},
    'edge_trystar_with_raise': set(),  # With raise is OK
    'edge_trystar_specific': set(),  # Specific exception is OK
}

# Functions with suppression directives
EDGE_CASE_SUPPRESSED: set[str] = {
    'edge_multi_code_suppression',  # Tests comma-separated directive codes
    'edge_logger_suppressed',  # Tests EXC006 suppression directive
}


# ---------------------------------------------------------------------------
# AST Parsing
# ---------------------------------------------------------------------------


def get_function_line_ranges(filepath: Path) -> Mapping[str, LineRange]:
    """Parse AST to get line ranges for each function.

    Returns dict mapping function name to (start_line, end_line).
    """
    source = filepath.read_text(encoding='utf-8')
    tree = ast.parse(source)

    ranges: dict[str, LineRange] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # end_lineno is always set for nodes parsed from source (not constructed)
            assert node.end_lineno is not None
            ranges[node.name] = LineRange(node.lineno, node.end_lineno)

    return ranges


# ---------------------------------------------------------------------------
# Linter Output Parsing
# ---------------------------------------------------------------------------


def run_linter(test_file: Path, linter: Path) -> str:
    """Run the linter and return combined stdout+stderr."""
    result = subprocess.run(
        [sys.executable, str(linter), str(test_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout + result.stderr


def parse_linter_output(output: str) -> list[tuple[int, str]]:
    """Parse linter output to extract (line_number, rule_code) tuples."""
    violations: list[tuple[int, str]] = []

    # Pattern: "filename:line:col: error: EXCNNN message"
    pattern = re.compile(r'.+:(\d+):\d+: error: (EXC\d+)')

    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            line_num = int(match.group(1))
            rule_code = match.group(2)
            violations.append((line_num, rule_code))

    return violations


def map_violations_to_functions(
    violations: list[tuple[int, str]],
    ranges: Mapping[str, LineRange],
) -> ViolationMap:
    """Map violations to the functions they occur in.

    Returns dict mapping function name to set of rule codes.
    """
    result: ViolationMap = {}

    for line_num, rule_code in violations:
        for func_name, (start, end) in ranges.items():
            if start <= line_num <= end:
                result.setdefault(func_name, set()).add(rule_code)
                break

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating a single test file."""

    file_name: str
    errors: list[str]
    violation_count: int
    expected_count: int
    rule_counts: dict[str, int]


def validate(
    actual: ViolationMap,
    expected: ViolationMap,
    suppressed: set[str],
    all_functions: set[str],
) -> list[str]:
    """Validate actual violations against expected.

    Returns list of error messages (empty if all passed).
    """
    errors: list[str] = []

    # Check expected violation functions
    for func, expected_rules in expected.items():
        actual_rules = actual.get(func, set())
        if actual_rules != expected_rules:
            errors.append(
                f'{func}: expected {sorted(expected_rules)}, got {sorted(actual_rules) if actual_rules else "nothing"}'
            )

    # Check for unexpected violations in correct/suppressed functions
    for func, actual_rules in actual.items():
        if func in expected:
            continue  # Already checked above

        if func in suppressed:
            # Suppressed functions should have their violations suppressed
            errors.append(f'{func}: suppression not working, got {sorted(actual_rules)}')
        else:
            # Any function not in expected with violations is unexpected
            errors.append(f'{func}: unexpected violations {sorted(actual_rules)}')

    # Check that all expected functions exist in the file
    errors.extend(f'{func}: function not found in test file' for func in expected if func not in all_functions)

    return errors


def validate_file(
    test_file: Path,
    expected: ViolationMap,
    suppressed: set[str],
) -> ValidationResult:
    """Validate a single test file against expected violations."""
    # Parse test file AST
    ranges = get_function_line_ranges(test_file)
    all_functions = set(ranges.keys())

    # Run linter
    output = run_linter(test_file, LINTER)

    # Parse violations
    violations = parse_linter_output(output)
    actual = map_violations_to_functions(violations, ranges)

    # Validate
    errors = validate(actual, expected, suppressed, all_functions)

    # Count violations by rule
    rule_counts: dict[str, int] = {}
    for _, rule in violations:
        rule_counts[rule] = rule_counts.get(rule, 0) + 1

    return ValidationResult(
        file_name=test_file.name,
        errors=errors,
        violation_count=len(violations),
        expected_count=sum(1 for rules in expected.values() if rules),
        rule_counts=rule_counts,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run validation and return exit code."""
    # Check linter exists
    if not LINTER.exists():
        print(f'ERROR: Linter not found: {LINTER}')
        return 1

    # Track overall results
    all_passed = True
    results: list[ValidationResult] = []

    # Validate instructive test file
    if TEST_FILE.exists():
        result = validate_file(TEST_FILE, EXPECTED_VIOLATIONS, SUPPRESSED_FUNCTIONS)
        results.append(result)
        if result.errors:
            all_passed = False
    else:
        print(f'WARNING: Test file not found: {TEST_FILE}')
        all_passed = False

    # Validate edge case file
    if EDGE_CASE_FILE.exists():
        result = validate_file(EDGE_CASE_FILE, EXPECTED_EDGE_CASES, EDGE_CASE_SUPPRESSED)
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
        print(f'    Violation functions: {result.expected_count}')
        print(f'    Total violations: {result.violation_count}')
        if result.rule_counts:
            print('    By rule:')
            for rule in sorted(result.rule_counts.keys()):
                print(f'      {rule}: {result.rule_counts[rule]}')
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
