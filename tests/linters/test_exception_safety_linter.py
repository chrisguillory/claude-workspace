"""Validate exception_safety_linter.py against its test cases.

Validates two test files:

1. exception_safety_test_cases.py (Instructive)
   - Each violation function triggers exactly one rule (no pollution)
   - Used for documentation and teaching

2. exception_safety_edge_cases.py (Regression)
   - Edge cases and false positive prevention
   - Comprehensive coverage of all linter code paths
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import NamedTuple

import pytest

# -- Data Types ---------------------------------------------------------------


class LineRange(NamedTuple):
    """Line range for a function definition (inclusive)."""

    start: int
    end: int


# Maps function name to set of violation codes found/expected in that function
type ViolationMap = dict[str, set[str]]


# -- Configuration ------------------------------------------------------------

TEST_DIR = Path(__file__).parent
CC_DIR = TEST_DIR.parent.parent
LINTER = CC_DIR / 'linters' / 'exception_safety_linter.py'

# Test file definitions
EDGE_CASES_DIR = TEST_DIR / 'edge_cases'
TEST_FILE = EDGE_CASES_DIR / 'exception_safety_test_cases.py'
EDGE_CASE_FILE = EDGE_CASES_DIR / 'exception_safety_edge_cases.py'

# -- Expected Violations: Instructive Test File -------------------------------

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
SUPPRESSED_FUNCTIONS: Set[str] = {
    'exc002_suppressed_intentional',
    'exc003_suppressed_top_level',
    'exc006_correct_when_suppressing',  # Has EXC002 suppression
}

# -- Expected Violations: Edge Case File --------------------------------------

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
    # Entry-point error boundary
    'edge_sys_exit_in_main': {'EXC002'},  # sys.exit(1) is not re-raise
}

# Functions with suppression directives
EDGE_CASE_SUPPRESSED: Set[str] = {
    'edge_multi_code_suppression',  # Tests comma-separated directive codes
    'edge_logger_suppressed',  # Tests EXC006 suppression directive
}


# -- AST Parsing --------------------------------------------------------------


def get_function_line_ranges(filepath: Path) -> Mapping[str, LineRange]:
    """Parse AST to get line ranges for each function.

    Returns dict mapping function name to (start_line, end_line).
    Fixture files must use unique function names — duplicates overwrite silently.
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


def parse_linter_output(output: str) -> Sequence[tuple[int, str]]:
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
    violations: Sequence[tuple[int, str]],
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


def _build_actual_map(test_file: Path) -> tuple[ViolationMap, Set[str]]:
    """Run linter and build actual violation map for a test file.

    Returns (violation_map, all_function_names).
    """
    ranges = get_function_line_ranges(test_file)
    output = run_linter(test_file, LINTER)
    violations = parse_linter_output(output)
    actual = map_violations_to_functions(violations, ranges)
    return actual, set(ranges.keys())


# -- Module-Scoped Fixtures ---------------------------------------------------


@pytest.fixture(scope='module')
def instructive_results() -> tuple[ViolationMap, Set[str]]:
    """Run linter on instructive test file and return (actual_map, all_functions)."""
    return _build_actual_map(TEST_FILE)


@pytest.fixture(scope='module')
def edge_case_results() -> tuple[ViolationMap, Set[str]]:
    """Run linter on edge case file and return (actual_map, all_functions)."""
    return _build_actual_map(EDGE_CASE_FILE)


# -- Parametrized Tests: Instructive ------------------------------------------


@pytest.mark.parametrize(
    ('func', 'expected_rules'),
    EXPECTED_VIOLATIONS.items(),
    ids=EXPECTED_VIOLATIONS.keys(),
)
def test_instructive_violation(
    func: str, expected_rules: set[str], instructive_results: tuple[ViolationMap, Set[str]]
) -> None:
    """Each instructive violation function triggers the expected rules."""
    actual_map, all_functions = instructive_results
    assert func in all_functions, f'{func}: function not found in test file'
    actual_rules = actual_map.get(func, set())
    assert actual_rules == expected_rules, (
        f'{func}: expected {sorted(expected_rules)}, got {sorted(actual_rules) if actual_rules else "nothing"}'
    )


@pytest.mark.parametrize('func', sorted(SUPPRESSED_FUNCTIONS), ids=sorted(SUPPRESSED_FUNCTIONS))
def test_instructive_suppression(func: str, instructive_results: tuple[ViolationMap, Set[str]]) -> None:
    """Suppressed functions produce no violations."""
    actual_map, _ = instructive_results
    actual_rules = actual_map.get(func, set())
    assert not actual_rules, f'{func}: suppression not working, got {sorted(actual_rules)}'


def test_instructive_no_unexpected(instructive_results: tuple[ViolationMap, Set[str]]) -> None:
    """No unexpected violations in functions not in expected or suppressed maps."""
    actual_map, _ = instructive_results
    known = set(EXPECTED_VIOLATIONS.keys()) | SUPPRESSED_FUNCTIONS
    unexpected = {
        f'{func}: unexpected violations {sorted(rules)}' for func, rules in actual_map.items() if func not in known
    }
    assert not unexpected, '\n'.join(sorted(unexpected))


# -- Parametrized Tests: Edge Cases -------------------------------------------


@pytest.mark.parametrize(
    ('func', 'expected_rules'),
    EXPECTED_EDGE_CASES.items(),
    ids=EXPECTED_EDGE_CASES.keys(),
)
def test_edge_case_violation(
    func: str, expected_rules: set[str], edge_case_results: tuple[ViolationMap, Set[str]]
) -> None:
    """Each edge case function triggers the expected rules (or none)."""
    actual_map, all_functions = edge_case_results
    assert func in all_functions, f'{func}: function not found in test file'
    actual_rules = actual_map.get(func, set())
    assert actual_rules == expected_rules, (
        f'{func}: expected {sorted(expected_rules)}, got {sorted(actual_rules) if actual_rules else "nothing"}'
    )


@pytest.mark.parametrize('func', sorted(EDGE_CASE_SUPPRESSED), ids=sorted(EDGE_CASE_SUPPRESSED))
def test_edge_case_suppression(func: str, edge_case_results: tuple[ViolationMap, Set[str]]) -> None:
    """Edge case suppressed functions produce no violations."""
    actual_map, _ = edge_case_results
    actual_rules = actual_map.get(func, set())
    assert not actual_rules, f'{func}: suppression not working, got {sorted(actual_rules)}'


def test_edge_case_no_unexpected(edge_case_results: tuple[ViolationMap, Set[str]]) -> None:
    """No unexpected violations in edge case functions not in expected or suppressed maps."""
    actual_map, _ = edge_case_results
    known = set(EXPECTED_EDGE_CASES.keys()) | EDGE_CASE_SUPPRESSED
    unexpected = {
        f'{func}: unexpected violations {sorted(rules)}' for func, rules in actual_map.items() if func not in known
    }
    assert not unexpected, '\n'.join(sorted(unexpected))
