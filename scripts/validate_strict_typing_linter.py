#!/usr/bin/env python3
"""Validate strict_typing_linter.py against its test cases.

This script validates two types of test files:

1. strict_typing_linter_test_cases.py (Instructive)
   - Each violation class triggers exactly one rule (no pollution)
   - Used for documentation and teaching

2. strict_typing_linter_edge_cases.py (Regression)
   - Edge cases and false positive prevention
   - Comprehensive coverage of all linter code paths

Run: ./validate_strict_typing_linter.py
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
LINTER = SCRIPT_DIR / 'strict_typing_linter.py'

# Test file definitions
TEST_FILE = SCRIPT_DIR / 'strict_typing_linter_test_cases.py'
EDGE_CASE_FILE = SCRIPT_DIR / 'strict_typing_linter_edge_cases.py'

# ---------------------------------------------------------------------------
# Expected Violations: Instructive Test File
# ---------------------------------------------------------------------------

# Maps class/function name to expected violation codes
# Each test class should trigger exactly one violation type
EXPECTED_VIOLATIONS: dict[str, set[str]] = {
    # tuple-field violations
    'TupleFieldViolationBasic': {'tuple-field'},
    'TupleFieldViolationNested': {'tuple-field'},
    'TupleFieldViolationUnion': {'tuple-field'},
    'TupleFieldViolationClassVar': {'tuple-field'},
    # hashable-field violations
    'HashableFieldViolationSequence': {'hashable-field'},
    'HashableFieldViolationMapping': {'hashable-field'},
    'HashableFieldViolationList': {'mutable'},  # list in hashable suggests tuple
    'HashableFieldViolationNestedSequence': {'hashable-field'},
    'HashableFieldViolationUnion': {'hashable-field'},
    # Nested class independence
    'InnerNotHashable': {'tuple-field'},  # Nested class doesn't inherit flag
    # Regular class with hashable
    'RegularClassHashableViolation': {'hashable-field'},
}

# Classes/functions with suppression directives (should NOT appear in output)
SUPPRESSED_NAMES: set[str] = {
    'TupleFieldSuppressed',
    'HashableFieldSuppressed',
}

# ---------------------------------------------------------------------------
# Expected Violations: Edge Case File
# ---------------------------------------------------------------------------

EXPECTED_EDGE_CASES: dict[str, set[str]] = {
    # Nested tuple edge cases
    'EdgeNestedTupleInMapping': {'tuple-field'},
    'EdgeNestedTupleInSequence': {'tuple-field'},
    'EdgeDeeplyNestedTuple': {'tuple-field'},
    'EdgeTupleInAnnotatedWithMetadata': {'tuple-field'},
    # Union edge cases
    'EdgeTupleUnionWithNone': {'tuple-field'},
    'EdgeTupleUnionOldSyntax': {'tuple-field'},
    'EdgeTupleInComplexUnion': {'tuple-field'},
    # False positive prevention (should NOT trigger)
    'EdgeFixedTupleMultiType': set(),
    'EdgeEmptyTuple': set(),
    'EdgeTupleNoSubscript': set(),
    'EdgeTupleInFunctionOnly': set(),
    # Hashable nested edge cases
    'EdgeHashableNestedSequenceInTuple': {'hashable-field'},
    'EdgeHashableNestedMappingInTuple': {'hashable-field'},
    'EdgeHashableSequenceInAnnotated': {'hashable-field'},
    # Hashable union edge cases
    'EdgeHashableSequenceUnion': {'hashable-field'},
    'EdgeHashableMappingUnion': {'hashable-field'},
    # Hashable list/dict suggestions
    'EdgeHashableListSuggestion': {'mutable'},
    'EdgeHashableDictSuggestion': {'mutable'},
    # ClassVar not flagged for hashability
    'EdgeHashableClassVarSequence': set(),
    'EdgeHashableClassVarMapping': set(),
    'EdgeHashableClassVarTuple': set(),
    # Nested class independence
    'EdgeInnerNoInherit': {'tuple-field'},
    'EdgeInnerDataclass': {'tuple-field'},
    # Multiple violations
    'EdgeMultipleViolationsTupleAndMutable': {'tuple-field', 'mutable'},
    'EdgeHashableMultipleViolations': {'hashable-field', 'mutable'},
    # String annotations
    'EdgeStringAnnotationTuple': {'tuple-field'},
    'EdgeStringAnnotationNested': {'tuple-field'},
    # Non-frozen dataclass skipped
    'EdgeNonFrozenSkipped': set(),
    'EdgeExplicitlyNotFrozen': set(),
    # Frozen without hashable flag
    'EdgeFrozenNoHashableFlag': {'tuple-field'},
    # Regular class with hashable flag
    'EdgeRegularClassHashable': set(),  # tuple OK in hashable
    'EdgeRegularClassHashableSequence': {'hashable-field'},
    # Pydantic examples
    'EdgePydanticStrictModel': {'tuple-field'},
    'EdgePydanticHashableModel': set(),  # tuple OK in hashable
    'EdgePydanticHashableSequence': {'hashable-field'},
    # attrs frozen detection
    'EdgeAttrsFrozenTuple': {'tuple-field'},
    'EdgeAttrsNonFrozenSkipped': set(),
    'EdgeAttrsDefineFrozenTuple': {'tuple-field'},
    'EdgeAttrsFrozenHashable': set(),  # tuple OK in hashable
    'EdgeAttrsFrozenHashableSequence': {'hashable-field'},
    # Nested class inside non-frozen dataclass (regression: _skip_class_fields leak)
    'EdgeNonFrozenOuterWithInner': set(),  # non-frozen, fields skipped
    'EdgeInnerInsideNonFrozen': {'tuple-field'},
    # 4+ levels of nesting
    'Level4': {'tuple-field'},
}

EDGE_CASE_SUPPRESSED: set[str] = {
    'EdgeSuppressedTupleField',
    'EdgeSuppressedHashableField',
    'EdgeMultipleSuppressionCodes',
}

# ---------------------------------------------------------------------------
# AST Parsing
# ---------------------------------------------------------------------------


def get_class_line_ranges(filepath: Path) -> dict[str, tuple[int, int]]:
    """Parse AST to get line ranges for each class.

    Returns dict mapping class name to (start_line, end_line).
    Uses ast.walk to find classes inside any compound statement (try, if, with, etc.).

    Note: Duplicate class names (same name at different scopes) will overwrite each other.
    Test files should use unique class names to avoid this limitation.
    """
    source = filepath.read_text(encoding='utf-8')
    tree = ast.parse(source)

    ranges: dict[str, tuple[int, int]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.end_lineno is not None:
            ranges[node.name] = (node.lineno, node.end_lineno)

    return ranges


# ---------------------------------------------------------------------------
# Linter Output Parsing
# ---------------------------------------------------------------------------


def run_linter(test_file: Path, linter: Path) -> tuple[str, int]:
    """Run the linter and return (output, return_code)."""
    result = subprocess.run(
        [sys.executable, str(linter), str(test_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout + result.stderr, result.returncode


def parse_linter_output(output: str) -> list[tuple[int, str]]:
    """Parse linter output to extract (line_number, violation_kind) tuples.

    Pattern: "filepath:line:col: error: {Type} '{bad_type}' in {context} annotation"
    """
    violations: list[tuple[int, str]] = []

    # Match violation types from error output
    patterns = [
        (r'.+:(\d+):\d+: error: Variable-length tuple', 'tuple-field'),
        (r'.+:(\d+):\d+: error: Unhashable type', 'hashable-field'),
        (r'.+:(\d+):\d+: error: Mutable type', 'mutable'),
        (r'.+:(\d+):\d+: error: Loose type', 'loose'),
    ]

    for line in output.splitlines():
        for pattern, kind in patterns:
            match = re.match(pattern, line)
            if match:
                line_num = int(match.group(1))
                violations.append((line_num, kind))
                break

    return violations


def map_violations_to_classes(
    violations: list[tuple[int, str]],
    ranges: dict[str, tuple[int, int]],
) -> dict[str, set[str]]:
    """Map violations to the classes they occur in.

    For nested classes with overlapping ranges, attributes violations to the
    innermost (most specific) class.

    Returns dict mapping class name to set of violation kinds.
    """
    result: dict[str, set[str]] = {}

    for line_num, kind in violations:
        # Find all classes that contain this line
        matching_classes = [
            (class_name, start, end) for class_name, (start, end) in ranges.items() if start <= line_num <= end
        ]

        if matching_classes:
            # Pick the innermost (smallest range)
            best = min(matching_classes, key=lambda x: x[2] - x[1])
            result.setdefault(best[0], set()).add(kind)

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
    kind_counts: dict[str, int]


def validate(
    actual: dict[str, set[str]],
    expected: dict[str, set[str]],
    suppressed: set[str],
    all_classes: set[str],
) -> list[str]:
    """Validate actual violations against expected.

    Returns list of error messages (empty if all passed).
    """
    errors: list[str] = []

    # Check expected violation classes
    for class_name, expected_kinds in expected.items():
        actual_kinds = actual.get(class_name, set())
        if actual_kinds != expected_kinds:
            errors.append(
                f'{class_name}: expected {sorted(expected_kinds)}, '
                f'got {sorted(actual_kinds) if actual_kinds else "nothing"}'
            )

    # Check for unexpected violations in correct/suppressed classes
    for class_name, actual_kinds in actual.items():
        if class_name in expected:
            continue  # Already checked above

        if class_name in suppressed:
            errors.append(f'{class_name}: suppression not working, got {sorted(actual_kinds)}')
        elif class_name not in expected:
            # Unexpected violation in a class not in our expected map
            errors.append(f'{class_name}: unexpected violations {sorted(actual_kinds)}')

    # Check that all expected classes exist
    errors.extend(
        f'{class_name}: class not found in test file' for class_name in expected if class_name not in all_classes
    )

    return errors


def validate_file(
    test_file: Path,
    expected: dict[str, set[str]],
    suppressed: set[str],
) -> ValidationResult:
    """Validate a single test file against expected violations."""
    # Parse test file AST
    ranges = get_class_line_ranges(test_file)
    all_classes = set(ranges.keys())

    # Run linter
    output, _ = run_linter(test_file, LINTER)

    # Parse violations
    violations = parse_linter_output(output)
    actual = map_violations_to_classes(violations, ranges)

    # Validate
    errors = validate(actual, expected, suppressed, all_classes)

    # Count violations by kind
    kind_counts: dict[str, int] = {}
    for _, kind in violations:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    return ValidationResult(
        file_name=test_file.name,
        errors=errors,
        violation_count=len(violations),
        expected_count=sum(len(kinds) for kinds in expected.values() if kinds),
        kind_counts=kind_counts,
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
        result = validate_file(TEST_FILE, EXPECTED_VIOLATIONS, SUPPRESSED_NAMES)
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
        print(f'    Violation classes: {result.expected_count}')
        print(f'    Total violations: {result.violation_count}')
        if result.kind_counts:
            print('    By kind:')
            for kind in sorted(result.kind_counts.keys()):
                print(f'      {kind}: {result.kind_counts[kind]}')
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
