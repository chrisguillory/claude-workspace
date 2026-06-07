"""Empirical ``--report-unused-directives`` coverage across the custom linters.

Each linter's named entities run through the real linter: a ``used_*`` directive
suppresses a real violation (must NOT be flagged), an ``unused_*`` directive matches
nothing (must be flagged). Polarity is declared by the entity name; the ``EXPECTED_*``
maps bind each entity to the suppressible code it exercises, which a completeness guard
checks against each linter's own code table — so a new code without both-polarity
coverage fails the suite.

strict_typing's module-*structural* rules (missing-all, trailing-comma, ordering,
class-ordering) and the proximity regression are mutually constraining, so they live in
single-purpose fixtures under ``edge_cases/strict_typing/`` and are line-keyed instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from pathlib import Path

import pytest

from tests.linters.helpers import (
    extract_directive_codes,
    get_class_ranges,
    get_def_ranges,
    parse_unused_directives,
    run_linter,
    sole_directive_line,
)

# -- Configuration ------------------------------------------------------------

TEST_DIR = Path(__file__).parent
CC_DIR = TEST_DIR.parent.parent
LINTERS_DIR = CC_DIR / 'linters'
EDGE_CASES_DIR = TEST_DIR / 'edge_cases'

EXCLUDED_CODES = {'unused-directive', 'skip-file'}

# Maps each named entity to the suppressible code it exercises. Polarity is read from
# the name (``used_*``/``Used*`` vs ``unused_*``/``Unused*``); the value drives the guard.
type CodeMap = Mapping[str, str]
# (fixture filename, code, is the directive expected unused?) for line-keyed structural cases.
type StructuralCase = tuple[str, str, bool]
# A linter run's result: each entity/entry -> its directive line, and the set of lines flagged unused.
type DirectiveScan = tuple[Mapping[str, int], Set[int]]


# -- exception_safety: paths + expectations -----------------------------------

EXCEPTION_SAFETY = LINTERS_DIR / 'exception_safety_linter.py'
EXCEPTION_SAFETY_FIXTURE = EDGE_CASES_DIR / 'exception_safety_unused_directives.py'

EXPECTED_EXCEPTION_SAFETY: CodeMap = {
    'used_bare_except': 'bare-except',
    'unused_bare_except': 'bare-except',
    'used_swallowed_exception': 'swallowed-exception',
    'unused_swallowed_exception': 'swallowed-exception',
    'used_finally_control_flow': 'finally-control-flow',
    'unused_finally_control_flow': 'finally-control-flow',
    'used_raise_without_from': 'raise-without-from',
    'unused_raise_without_from': 'raise-without-from',
    'used_unused_exception_var': 'unused-exception-var',
    'unused_unused_exception_var': 'unused-exception-var',
    'used_logger_no_exc_info': 'logger-no-exc-info',
    'unused_logger_no_exc_info': 'logger-no-exc-info',
    'used_cancelled_not_raised': 'cancelled-not-raised',
    'unused_cancelled_not_raised': 'cancelled-not-raised',
    'used_generator_exit_not_raised': 'generator-exit-not-raised',
    'unused_generator_exit_not_raised': 'generator-exit-not-raised',
    'UsedInitNotPickleable': 'init-not-pickleable',
    'UnusedInitNotPickleable': 'init-not-pickleable',
}


# -- reexport: paths + expectations -------------------------------------------

REEXPORT = LINTERS_DIR / 'reexport_linter.py'
REEXPORT_FIXTURE = EDGE_CASES_DIR / 'reexport_unused_directives.py'

EXPECTED_REEXPORT: CodeMap = {
    'used_reexport': 'reexported-symbol',
    'unused_reexport': 'reexported-symbol',
}


# -- replace: paths + expectations --------------------------------------------

REPLACE = LINTERS_DIR / 'replace_linter.py'
REPLACE_FIXTURE = EDGE_CASES_DIR / 'replace_unused_directives.py'

EXPECTED_REPLACE: CodeMap = {
    'used_model_copy_update': 'model-copy-update',
    'unused_model_copy_update': 'model-copy-update',
}


# -- strict_typing: paths + expectations --------------------------------------

STRICT_TYPING = LINTERS_DIR / 'strict_typing_linter.py'
STRICT_TYPING_TYPE_FIXTURE = EDGE_CASES_DIR / 'strict_typing_unused_directives.py'
STRICT_TYPING_DIR = EDGE_CASES_DIR / 'strict_typing'

# Type rules are named frozen-dataclass / plain classes (mapped via get_class_ranges).
EXPECTED_STRICT_TYPING_TYPE: CodeMap = {
    'UsedMutableType': 'mutable-type',
    'UnusedMutableType': 'mutable-type',
    'UsedLooseTyping': 'loose-typing',
    'UnusedLooseTyping': 'loose-typing',
    'UsedTupleField': 'tuple-field',
    'UnusedTupleField': 'tuple-field',
    'UsedHashableField': 'hashable-field',
    'UnusedHashableField': 'hashable-field',
    'UsedFrozensetConstant': 'frozenset-constant',
    'UnusedFrozensetConstant': 'frozenset-constant',
}

# Structural rules are single-purpose fixtures (line-keyed). proximity_mutable_type locks in
# find_unused_directives' ``| {code}`` union (its return directive is matched via Strategy 2).
STRUCTURAL_CASES: Sequence[StructuralCase] = [
    ('missing_all_used.py', 'missing-all', False),
    ('missing_all_unused.py', 'missing-all', True),
    ('trailing_comma_used.py', 'trailing-comma', False),
    ('trailing_comma_unused.py', 'trailing-comma', True),
    ('ordering_used.py', 'ordering', False),
    ('ordering_unused.py', 'ordering', True),
    ('class_ordering_used.py', 'class-ordering', False),
    ('class_ordering_unused.py', 'class-ordering', True),
    ('proximity_mutable_type.py', 'mutable-type', False),
]


# -- Completeness guard config ------------------------------------------------

# (linter, code-table name, named EXPECTED map, structural cases). The guard computes covered
# codes from these at call time; a declared code without both-polarity coverage fails it.
COVERAGE_SPECS: Sequence[tuple[Path, str, CodeMap, Sequence[StructuralCase]]] = [
    (EXCEPTION_SAFETY, 'ERROR_CODES', EXPECTED_EXCEPTION_SAFETY, ()),
    (REEXPORT, 'ERROR_CODES', EXPECTED_REEXPORT, ()),
    (REPLACE, 'ERROR_CODES', EXPECTED_REPLACE, ()),
    (STRICT_TYPING, 'CODE_TO_KINDS', EXPECTED_STRICT_TYPING_TYPE, STRUCTURAL_CASES),
]


# -- Fixtures (grouped before tests: pytest fixtures sort with classes) --------


@pytest.fixture(scope='module')
def exception_safety_unused() -> DirectiveScan:
    """Run exception_safety with --report-unused-directives once; return (lines, flagged)."""
    output = run_linter(EXCEPTION_SAFETY_FIXTURE, EXCEPTION_SAFETY, report_unused=True)
    lines = _directive_lines(EXCEPTION_SAFETY_FIXTURE, EXCEPTION_SAFETY, EXPECTED_EXCEPTION_SAFETY)
    return lines, parse_unused_directives(output)


@pytest.fixture(scope='module')
def reexport_unused() -> DirectiveScan:
    """Run reexport with --report-unused-directives once; return (lines, flagged)."""
    output = run_linter(REEXPORT_FIXTURE, REEXPORT, report_unused=True)
    lines = {entry: _reexport_directive_line(REEXPORT_FIXTURE, entry) for entry in EXPECTED_REEXPORT}
    return lines, parse_unused_directives(output)


@pytest.fixture(scope='module')
def replace_unused() -> DirectiveScan:
    """Run replace with --report-unused-directives once; return (lines, flagged)."""
    output = run_linter(REPLACE_FIXTURE, REPLACE, report_unused=True)
    lines = _directive_lines(REPLACE_FIXTURE, REPLACE, EXPECTED_REPLACE)
    return lines, parse_unused_directives(output)


@pytest.fixture(scope='module')
def strict_typing_type_unused() -> DirectiveScan:
    """Run strict_typing on the type-rule fixture once; return (lines, flagged)."""
    output = run_linter(STRICT_TYPING_TYPE_FIXTURE, STRICT_TYPING, report_unused=True)
    lines = _directive_lines(STRICT_TYPING_TYPE_FIXTURE, STRICT_TYPING, EXPECTED_STRICT_TYPING_TYPE)
    return lines, parse_unused_directives(output)


# -- Tests --------------------------------------------------------------------


@pytest.mark.parametrize('entity', EXPECTED_EXCEPTION_SAFETY, ids=list(EXPECTED_EXCEPTION_SAFETY))
def test_exception_safety_directive(entity: str, exception_safety_unused: DirectiveScan) -> None:
    """Each used_ directive suppresses a real violation; each unused_ directive is flagged."""
    lines, unused = exception_safety_unused
    _assert_polarity(entity, lines, unused)


def test_exception_safety_no_unexpected(exception_safety_unused: DirectiveScan) -> None:
    """No directive lines beyond the unused_ entities are flagged."""
    lines, unused = exception_safety_unused
    _assert_no_strays(EXPECTED_EXCEPTION_SAFETY, lines, unused)


@pytest.mark.parametrize('entry', EXPECTED_REEXPORT, ids=list(EXPECTED_REEXPORT))
def test_reexport_directive(entry: str, reexport_unused: DirectiveScan) -> None:
    """The imported re-export's directive is used; the local entry's directive is flagged unused."""
    lines, unused = reexport_unused
    _assert_polarity(entry, lines, unused)


def test_reexport_no_unexpected(reexport_unused: DirectiveScan) -> None:
    """Only the unused_ entry's directive line is flagged."""
    lines, unused = reexport_unused
    _assert_no_strays(EXPECTED_REEXPORT, lines, unused)


@pytest.mark.parametrize('entity', EXPECTED_REPLACE, ids=list(EXPECTED_REPLACE))
def test_replace_directive(entity: str, replace_unused: DirectiveScan) -> None:
    """The update= call's directive is used; the bare-clone directive is flagged unused."""
    lines, unused = replace_unused
    _assert_polarity(entity, lines, unused)


def test_replace_no_unexpected(replace_unused: DirectiveScan) -> None:
    """Only the unused_ entity's directive line is flagged."""
    lines, unused = replace_unused
    _assert_no_strays(EXPECTED_REPLACE, lines, unused)


@pytest.mark.parametrize('entity', EXPECTED_STRICT_TYPING_TYPE, ids=list(EXPECTED_STRICT_TYPING_TYPE))
def test_strict_typing_type_directive(entity: str, strict_typing_type_unused: DirectiveScan) -> None:
    """Each Used type-rule directive suppresses a real violation; each Unused one is flagged."""
    lines, unused = strict_typing_type_unused
    _assert_polarity(entity, lines, unused)


def test_strict_typing_type_no_unexpected(strict_typing_type_unused: DirectiveScan) -> None:
    """No type-rule directive lines beyond the Unused classes are flagged."""
    lines, unused = strict_typing_type_unused
    _assert_no_strays(EXPECTED_STRICT_TYPING_TYPE, lines, unused)


@pytest.mark.parametrize(('filename', 'code', 'is_unused'), STRUCTURAL_CASES, ids=[c[0] for c in STRUCTURAL_CASES])
def test_strict_typing_structural(filename: str, code: str, is_unused: bool) -> None:
    """Each structural fixture's directive(s) carry the expected polarity; used fixtures stay clean."""
    fixture = STRICT_TYPING_DIR / filename
    output = run_linter(fixture, STRICT_TYPING, report_unused=True)
    unused = parse_unused_directives(output)
    for line in _structural_directive_lines(fixture):
        flagged = line in unused
        assert flagged == is_unused, f'{filename}: directive @{line} flagged={flagged}, expected {is_unused}'
    if not is_unused:
        violations = [ln for ln in output.splitlines() if ': error:' in ln and 'does not match any violation' not in ln]
        assert not violations, f'{filename}: directive must suppress cleanly, got {violations}'


@pytest.mark.parametrize(
    ('linter', 'table', 'expected', 'structural'),
    COVERAGE_SPECS,
    ids=[spec[0].stem for spec in COVERAGE_SPECS],
)
def test_code_coverage_complete(
    linter: Path, table: str, expected: CodeMap, structural: Sequence[StructuralCase]
) -> None:
    """Every suppressible code in each linter's table has both used and unused coverage."""
    covered = _covered_named(expected) | _covered_structural(structural)
    declared = extract_directive_codes(linter, table) - EXCLUDED_CODES
    assert covered <= declared, f'{linter.name}: EXPECTED references unknown codes {sorted(covered - declared)}'
    assert declared <= covered, f'{linter.name}: codes lacking both-polarity coverage {sorted(declared - covered)}'


# -- Shared helpers (private) -------------------------------------------------


def _is_unused(entity: str) -> bool:
    """Polarity from the entity name: ``unused_*`` / ``Unused*`` expect a flagged directive."""
    return entity.lower().startswith('unused')


def _directive_lines(fixture: Path, linter: Path, entities: CodeMap) -> Mapping[str, int]:
    """Map each named entity to its sole directive line (functions and classes)."""
    ranges = {**get_def_ranges(fixture), **get_class_ranges(fixture)}
    missing = sorted(set(entities) - set(ranges))
    assert not missing, f'entities not found in {fixture.name}: {missing}'
    return {name: sole_directive_line(fixture, ranges[name], linter) for name in entities}


def _assert_polarity(entity: str, lines: Mapping[str, int], unused: Set[int]) -> None:
    """A used_ directive must be absent from the flagged set; an unused_ directive present."""
    line = lines[entity]
    if _is_unused(entity):
        assert line in unused, f'{entity}: directive @{line} should be flagged unused'
    else:
        assert line not in unused, f'{entity}: directive @{line} should be used (not flagged)'


def _assert_no_strays(expected: CodeMap, lines: Mapping[str, int], unused: Set[int]) -> None:
    """Only unused_ entities' directive lines may be flagged."""
    expected_lines = {lines[name] for name in expected if _is_unused(name)}
    assert unused == expected_lines, (
        f'stray={sorted(unused - expected_lines)}, missing={sorted(expected_lines - unused)}'
    )


def _reexport_directive_line(fixture: Path, entry: str) -> int:
    """Line of the ``__all__`` entry ``entry`` carrying a reexported-symbol directive.

    Reexport directives sit on ``__all__`` entries (inside a list literal), not in any
    def/class range, so they're located by entry name rather than via ``sole_directive_line``.
    """
    needle = f"'{entry}'"
    matches = [
        lineno
        for lineno, line in enumerate(fixture.read_text(encoding='utf-8').splitlines(), 1)
        if needle in line and 'reexport_linter.py: reexported-symbol' in line
    ]
    assert len(matches) == 1, f'expected one {entry!r} directive entry in {fixture.name}, found {matches}'
    return matches[0]


def _structural_directive_lines(fixture: Path) -> Sequence[int]:
    """All strict_typing directive lines in a single-purpose structural fixture (1, or 2 for double-emit)."""
    prefix = '# strict_typing_linter.py:'
    lines = [n for n, line in enumerate(fixture.read_text(encoding='utf-8').splitlines(), 1) if prefix in line]
    assert lines, f'no strict_typing directive in {fixture.name}'
    return lines


def _covered_named(expected: CodeMap) -> Set[str]:
    """Codes with both a used_ and an unused_ entity in a named-entity EXPECTED map."""
    used = {code for name, code in expected.items() if not _is_unused(name)}
    unused = {code for name, code in expected.items() if _is_unused(name)}
    return used & unused


def _covered_structural(cases: Sequence[StructuralCase]) -> Set[str]:
    """Codes with both a used and an unused single-purpose structural fixture."""
    used = {code for _, code, is_unused in cases if not is_unused}
    unused = {code for _, code, is_unused in cases if is_unused}
    return used & unused
