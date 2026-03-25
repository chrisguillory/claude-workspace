"""Validate strict_typing_linter.py against its test cases.

Validates two test files:

1. strict_typing_linter_test_cases.py (Instructive)
   - Each violation class triggers exactly one rule (no pollution)
   - Used for documentation and teaching

2. strict_typing_linter_edge_cases.py (Regression)
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
    """Line range for a class definition (inclusive)."""

    start: int
    end: int


# Maps class name to set of violation kinds found/expected in that class
type ViolationMap = dict[str, set[str]]


# -- Configuration ------------------------------------------------------------

TEST_DIR = Path(__file__).parent
CC_DIR = TEST_DIR.parent.parent
LINTER = CC_DIR / 'linters' / 'strict_typing_linter.py'

# Test file definitions
EDGE_CASES_DIR = TEST_DIR / 'edge_cases'
TEST_FILE = EDGE_CASES_DIR / 'strict_typing_linter_test_cases.py'
EDGE_CASE_FILE = EDGE_CASES_DIR / 'strict_typing_linter_edge_cases.py'

# -- Expected Violations: Instructive Test File -------------------------------

# Maps class/function name to expected violation codes
# Each test class should trigger exactly one violation type
EXPECTED_VIOLATIONS: ViolationMap = {
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
SUPPRESSED_NAMES: Set[str] = {
    'TupleFieldSuppressed',
    'HashableFieldSuppressed',
}

# -- Expected Violations: Edge Case File --------------------------------------

EXPECTED_EDGE_CASES: ViolationMap = {
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
    'EdgeOuterHashable': set(),  # outer wrapper, no violations
    'EdgeOuterHashableWithInnerDataclass': set(),  # outer wrapper
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
    'EdgeDeeplyNested4Levels': set(),  # non-frozen outer, fields skipped
    'Level2': set(),  # regular class, no annotated fields
    'Level3': set(),  # frozen dataclass wrapper, no violating fields
    'Level4': {'tuple-field'},
    # Cross-type hashability: runtime inspector (attrs -> dataclass -> Pydantic)
    # attrs
    'EdgeCrossTypeAttrsUnhashable': {'hashable-field'},
    'EdgeCrossTypeAttrsNonFrozen': {'hashable-field'},
    'EdgeCrossTypeAttrsHashable': set(),
    # dataclass
    'EdgeCrossTypeDataclassUnhashable': {'hashable-field'},
    'EdgeCrossTypeDataclassNonFrozen': {'hashable-field'},
    'EdgeCrossTypeDataclassInUnion': {'hashable-field'},
    'EdgeCrossTypeDataclassHashable': set(),
    'EdgeCrossTypeDataclassHashExcluded': set(),
    'EdgeCrossTypeDataclassHashableUnion': set(),
    # Pydantic
    'EdgeCrossTypePydanticUnhashable': {'hashable-field'},
    'EdgeCrossTypePydanticNonFrozen': {'hashable-field'},
    'EdgeCrossTypeInUnion': {'hashable-field'},
    'EdgeCrossTypePydanticHashable': set(),
    'EdgeCrossTypeHashableUnion': set(),
    # Mixed framework cross-references
    'EdgeCrossTypeAttrsRefsPydanticUnhashable': {'hashable-field'},
    'EdgeCrossTypeAttrsRefsPydanticHashable': set(),
    'EdgeCrossTypeAttrsRefsDataclassUnhashable': {'hashable-field'},
    'EdgeCrossTypeAttrsRefsDataclassHashable': set(),
    'EdgeCrossTypeDataclassRefsPydanticUnhashable': {'hashable-field'},
    'EdgeCrossTypeDataclassRefsPydanticHashable': set(),
    'EdgeCrossTypeDataclassRefsAttrsUnhashable': {'hashable-field'},
    'EdgeCrossTypeDataclassRefsAttrsHashable': set(),
    'EdgeCrossTypePydanticRefsAttrsUnhashable': {'hashable-field'},
    'EdgeCrossTypePydanticRefsAttrsHashable': set(),
    'EdgeCrossTypePydanticRefsDataclassUnhashable': {'hashable-field'},
    'EdgeCrossTypePydanticRefsDataclassHashable': set(),
}

EDGE_CASE_SUPPRESSED: Set[str] = {
    'EdgeSuppressedTupleField',
    'EdgeSuppressedHashableField',
    'EdgeMultipleSuppressionCodes',
}

# -- AST Parsing --------------------------------------------------------------


def get_class_line_ranges(filepath: Path) -> Mapping[str, LineRange]:
    """Parse AST to get line ranges for each class.

    Returns dict mapping class name to (start_line, end_line).
    Uses ast.walk to find classes inside any compound statement (try, if, with, etc.).

    Note: Duplicate class names (same name at different scopes) will overwrite each other.
    Test files should use unique class names to avoid this limitation.
    """
    source = filepath.read_text(encoding='utf-8')
    tree = ast.parse(source)

    ranges: dict[str, LineRange] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.end_lineno is not None:
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
    violations: Sequence[tuple[int, str]],
    ranges: Mapping[str, LineRange],
) -> ViolationMap:
    """Map violations to the classes they occur in.

    For nested classes with overlapping ranges, attributes violations to the
    innermost (most specific) class.

    Returns dict mapping class name to set of violation kinds.
    """
    result: ViolationMap = {}

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


def _build_actual_map(test_file: Path) -> tuple[ViolationMap, Set[str]]:
    """Run linter and build actual violation map for a test file.

    Returns (violation_map, all_class_names).
    """
    ranges = get_class_line_ranges(test_file)
    output = run_linter(test_file, LINTER)
    violations = parse_linter_output(output)
    actual = map_violations_to_classes(violations, ranges)
    return actual, set(ranges.keys())


# -- Module-Scoped Fixtures ---------------------------------------------------


@pytest.fixture(scope='module')
def instructive_results() -> tuple[ViolationMap, Set[str]]:
    """Run linter on instructive test file and return (actual_map, all_classes)."""
    return _build_actual_map(TEST_FILE)


@pytest.fixture(scope='module')
def edge_case_results() -> tuple[ViolationMap, Set[str]]:
    """Run linter on edge case file and return (actual_map, all_classes)."""
    return _build_actual_map(EDGE_CASE_FILE)


# -- Parametrized Tests: Instructive ------------------------------------------


@pytest.mark.parametrize(
    ('class_name', 'expected_kinds'),
    EXPECTED_VIOLATIONS.items(),
    ids=EXPECTED_VIOLATIONS.keys(),
)
def test_instructive_violation(
    class_name: str, expected_kinds: set[str], instructive_results: tuple[ViolationMap, Set[str]]
) -> None:
    """Each instructive violation class triggers the expected kinds."""
    actual_map, all_classes = instructive_results
    assert class_name in all_classes, f'{class_name}: class not found in test file'
    actual_kinds = actual_map.get(class_name, set())
    assert actual_kinds == expected_kinds, (
        f'{class_name}: expected {sorted(expected_kinds)}, got {sorted(actual_kinds) if actual_kinds else "nothing"}'
    )


@pytest.mark.parametrize('class_name', sorted(SUPPRESSED_NAMES), ids=sorted(SUPPRESSED_NAMES))
def test_instructive_suppression(class_name: str, instructive_results: tuple[ViolationMap, Set[str]]) -> None:
    """Suppressed classes produce no violations."""
    actual_map, _ = instructive_results
    actual_kinds = actual_map.get(class_name, set())
    assert not actual_kinds, f'{class_name}: suppression not working, got {sorted(actual_kinds)}'


def test_instructive_no_unexpected(instructive_results: tuple[ViolationMap, Set[str]]) -> None:
    """No unexpected violations in classes not in expected or suppressed maps."""
    actual_map, _ = instructive_results
    known = set(EXPECTED_VIOLATIONS.keys()) | SUPPRESSED_NAMES
    unexpected = {
        f'{name}: unexpected violations {sorted(kinds)}' for name, kinds in actual_map.items() if name not in known
    }
    assert not unexpected, '\n'.join(sorted(unexpected))


# -- Parametrized Tests: Edge Cases -------------------------------------------


@pytest.mark.parametrize(
    ('class_name', 'expected_kinds'),
    EXPECTED_EDGE_CASES.items(),
    ids=EXPECTED_EDGE_CASES.keys(),
)
def test_edge_case_violation(
    class_name: str, expected_kinds: set[str], edge_case_results: tuple[ViolationMap, Set[str]]
) -> None:
    """Each edge case class triggers the expected kinds (or none)."""
    actual_map, all_classes = edge_case_results
    assert class_name in all_classes, f'{class_name}: class not found in test file'
    actual_kinds = actual_map.get(class_name, set())
    assert actual_kinds == expected_kinds, (
        f'{class_name}: expected {sorted(expected_kinds)}, got {sorted(actual_kinds) if actual_kinds else "nothing"}'
    )


@pytest.mark.parametrize('class_name', sorted(EDGE_CASE_SUPPRESSED), ids=sorted(EDGE_CASE_SUPPRESSED))
def test_edge_case_suppression(class_name: str, edge_case_results: tuple[ViolationMap, Set[str]]) -> None:
    """Edge case suppressed classes produce no violations."""
    actual_map, _ = edge_case_results
    actual_kinds = actual_map.get(class_name, set())
    assert not actual_kinds, f'{class_name}: suppression not working, got {sorted(actual_kinds)}'


def test_edge_case_no_unexpected(edge_case_results: tuple[ViolationMap, Set[str]]) -> None:
    """No unexpected violations in edge case classes not in expected or suppressed maps."""
    actual_map, _ = edge_case_results
    known = set(EXPECTED_EDGE_CASES.keys()) | EDGE_CASE_SUPPRESSED
    unexpected = {
        f'{name}: unexpected violations {sorted(kinds)}' for name, kinds in actual_map.items() if name not in known
    }
    assert not unexpected, '\n'.join(sorted(unexpected))
