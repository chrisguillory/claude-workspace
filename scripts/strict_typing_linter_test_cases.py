# strict_typing_linter.py: skip-file
# ruff: noqa: F841
# mypy: disable-error-code="type-arg"
"""Strict typing linter test cases - demonstrates tuple-field and hashable-field rules.

Run the linter on this file: ./strict_typing_linter.py strict_typing_linter_test_cases.py

Each rule has separate functions/classes:
- VIOLATION: Code that triggers the linter (intentionally bad)
- CORRECT: Proper patterns to follow
- SUPPRESSED: Cases where suppression is justified (if any)

Design principle: Each test class should trigger exactly one violation to make
validation straightforward and errors easy to diagnose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

# =============================================================================
# tuple-field: Variable-length tuple in class fields
# =============================================================================


class TupleFieldViolationBasic:
    """VIOLATION: tuple[X, ...] in class field should use Sequence[T] instead."""

    values: tuple[int, ...]  # tuple-field: variable-length tuple in field


class TupleFieldViolationNested:
    """VIOLATION: Nested tuple[X, ...] also triggers the rule."""

    data: Mapping[str, tuple[int, ...]]  # tuple-field: nested variable-length tuple


class TupleFieldViolationUnion:
    """VIOLATION: tuple[X, ...] in union type is still flagged."""

    values: tuple[int, ...] | None  # tuple-field: union with variable-length tuple


class TupleFieldViolationClassVar:
    """VIOLATION: ClassVar also checked - prefer abstract interfaces everywhere."""

    DEFAULTS: ClassVar[tuple[int, ...]] = (1, 2, 3)  # tuple-field: use Sequence


class TupleFieldCorrectInParam:
    """CORRECT: tuple[X, ...] in function parameter is allowed."""

    def process(self, items: tuple[int, ...]) -> None:
        pass


class TupleFieldCorrectInReturn:
    """CORRECT: tuple[X, ...] in return type is allowed."""

    def get_items(self) -> tuple[int, ...]:
        return (1, 2, 3)


class TupleFieldCorrectFixedLength:
    """CORRECT: Fixed-length tuples are allowed in fields."""

    point: tuple[int, int]  # OK: fixed-length
    record: tuple[str, int, float]  # OK: fixed-length


class TupleFieldCorrectBareTuple:
    """CORRECT: Bare tuple annotation defers to type checker.

    The type checker (mypy/pyright) will require a more specific annotation.
    Our linter doesn't duplicate what type checkers already do well.
    """

    data: tuple  # OK: type checker handles bare tuple


class TupleFieldCorrectSequence:
    """CORRECT: Use Sequence[T] instead of tuple[X, ...] in fields."""

    values: Sequence[int]  # OK: proper immutable interface


class TupleFieldCorrectClassVarSequence:
    """CORRECT: ClassVar with Sequence is the preferred pattern."""

    DEFAULTS: ClassVar[Sequence[int]] = (1, 2, 3)  # OK: abstract interface


class TupleFieldSuppressed:
    """SUPPRESSED: Rare case where variable-length tuple is needed."""

    values: tuple[int, ...]  # strict_typing_linter.py: tuple-field


# =============================================================================
# hashable-field: Unhashable types in hashable class fields
# =============================================================================


@dataclass(frozen=True)
class HashableFieldViolationSequence:
    """VIOLATION: Sequence[X] is not hashable at runtime."""

    __strict_typing_linter__hashable_fields__ = True

    items: Sequence[int]  # hashable-field: Sequence not hashable


@dataclass(frozen=True)
class HashableFieldViolationMapping:
    """VIOLATION: Mapping[K, V] is not hashable at runtime.

    Suggestion: frozendict[K, V] (third-party) or restructure.
    Note: There's no stdlib hashable equivalent to Mapping.
    """

    __strict_typing_linter__hashable_fields__ = True

    data: Mapping[str, int]  # hashable-field: Mapping not hashable


@dataclass(frozen=True)
class HashableFieldViolationList:
    """VIOLATION: list[X] in hashable class suggests tuple[T, ...] not Sequence[T].

    The suggestion is different because Sequence would also be flagged.
    """

    __strict_typing_linter__hashable_fields__ = True

    items: list[int]  # mutable-type: list in hashable class (suggests tuple)


@dataclass(frozen=True)
class HashableFieldViolationNestedSequence:
    """VIOLATION: Nested Sequence inside tuple is still unhashable."""

    __strict_typing_linter__hashable_fields__ = True

    data: tuple[Sequence[int], ...]  # hashable-field: nested Sequence


@dataclass(frozen=True)
class HashableFieldViolationUnion:
    """VIOLATION: Sequence in union is still unhashable."""

    __strict_typing_linter__hashable_fields__ = True

    items: Sequence[int] | None  # hashable-field: Sequence in union


@dataclass(frozen=True)
class HashableFieldCorrectTuple:
    """CORRECT: tuple[X, ...] is allowed in hashable classes (needed for hashability)."""

    __strict_typing_linter__hashable_fields__ = True

    values: tuple[int, ...]  # OK: tuple is hashable


@dataclass(frozen=True)
class HashableFieldCorrectFrozenset:
    """CORRECT: frozenset is hashable."""

    __strict_typing_linter__hashable_fields__ = True

    items: frozenset[int]  # OK: frozenset is hashable


@dataclass(frozen=True)
class HashableFieldCorrectClassVar:
    """CORRECT: ClassVar fields don't affect __hash__ - not checked for hashability."""

    __strict_typing_linter__hashable_fields__ = True

    DEFAULTS: ClassVar[Sequence[int]] = []  # OK: ClassVar doesn't affect hash


@dataclass(frozen=True)
class HashableFieldSuppressed:
    """SUPPRESSED: Rare case where unhashable field is acceptable."""

    __strict_typing_linter__hashable_fields__ = True

    items: Sequence[int]  # strict_typing_linter.py: hashable-field


# =============================================================================
# Interaction: Without hashable flag, Sequence/Mapping are fine
# =============================================================================


class NonHashableClassSequenceOk:
    """CORRECT: Without hashable flag, Sequence[X] is encouraged."""

    items: Sequence[int]  # OK: not in hashable class


class NonHashableClassMappingOk:
    """CORRECT: Without hashable flag, Mapping[K, V] is encouraged."""

    data: Mapping[str, int]  # OK: not in hashable class


# =============================================================================
# Context: Hashable flag does NOT inherit to nested classes
# =============================================================================


@dataclass(frozen=True)
class OuterHashable:
    """Parent class with hashable flag."""

    __strict_typing_linter__hashable_fields__ = True

    outer_values: tuple[int, ...]  # OK: hashable class allows tuple

    class InnerNotHashable:
        """Nested class does NOT inherit hashable flag."""

        inner_values: tuple[int, ...]  # tuple-field: flag doesn't inherit


# =============================================================================
# Context: Non-frozen dataclasses are skipped entirely
# =============================================================================


@dataclass
class NonFrozenDataclass:
    """Non-frozen dataclass fields are not checked (mutable by design)."""

    items: list[int]  # OK: non-frozen dataclass is mutable by design
    values: tuple[int, ...]  # OK: non-frozen dataclass not checked


# NOTE: Non-frozen dataclass with __strict_typing_linter__hashable_fields__ = True
# raises ValueError (contradiction). See edge_cases for error test.


# =============================================================================
# Context: Frozen dataclass with hashable flag
# =============================================================================


@dataclass(frozen=True)
class FrozenDataclassHashable:
    """Frozen dataclass with hashable flag - full strictness."""

    __strict_typing_linter__hashable_fields__ = True

    id: str
    values: tuple[int, ...]  # OK: tuple allowed in hashable


# =============================================================================
# Context: Regular class (not dataclass) with hashable flag
# =============================================================================


class RegularClassHashable:
    """Regular classes can also use hashable flag (e.g., Pydantic models)."""

    __strict_typing_linter__hashable_fields__ = True

    values: tuple[int, ...]  # OK: hashable context allows tuple


class RegularClassHashableViolation:
    """Regular class with hashable flag - Sequence is flagged."""

    __strict_typing_linter__hashable_fields__ = True

    items: Sequence[int]  # hashable-field: Sequence not hashable


# =============================================================================
# Stub Functions (for class method signatures)
# =============================================================================


def standalone_function(items: tuple[int, ...]) -> tuple[str, ...]:
    """CORRECT: Functions can use tuple[X, ...] freely."""
    return tuple(str(x) for x in items)
