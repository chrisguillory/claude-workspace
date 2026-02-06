# ruff: noqa: F841
# mypy: disable-error-code="type-arg"
"""Strict typing linter edge cases - regression testing and comprehensive coverage.

This file complements strict_typing_linter_test_cases.py with:
- Edge cases for unusual but valid Python patterns
- False positive prevention (valid code that should NOT trigger rules)
- Comprehensive coverage of all linter code paths
- String annotations and complex nested types

Run: ./validate_strict_typing_linter.py (validates both test files)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, ClassVar

import attrs
import pydantic

# =============================================================================
# tuple-field: Nested Type Edge Cases
# =============================================================================


class EdgeNestedTupleInMapping:
    """VIOLATION: Inner tuple[X, ...] in Mapping value should flag."""

    data: Mapping[str, tuple[int, ...]]  # tuple-field: nested inside Mapping


class EdgeNestedTupleInSequence:
    """VIOLATION: Inner tuple[X, ...] in Sequence should flag."""

    items: Sequence[tuple[str, ...]]  # tuple-field: nested inside Sequence


class EdgeDeeplyNestedTuple:
    """VIOLATION: Deeply nested tuple[X, ...] should flag."""

    data: Mapping[str, Mapping[str, tuple[int, ...]]]  # tuple-field: deeply nested


class EdgeTupleInAnnotatedWithMetadata:
    """VIOLATION: Annotated with extra metadata still flags tuple."""

    value: Annotated[tuple[int, ...], pydantic.Field(ge=0), 'extra']  # tuple-field


# =============================================================================
# tuple-field: Union Type Edge Cases
# =============================================================================


class EdgeTupleUnionWithNone:
    """VIOLATION: tuple[X, ...] | None should flag."""

    values: tuple[int, ...] | None  # tuple-field


class EdgeTupleUnionOldSyntax:
    """VIOLATION: Union[tuple[X, ...], None] should flag (old syntax)."""

    values: tuple[int, ...] | None  # tuple-field


class EdgeTupleInComplexUnion:
    """VIOLATION: tuple[X, ...] in complex union should flag."""

    values: str | tuple[int, ...] | None  # tuple-field


# =============================================================================
# tuple-field: False Positive Prevention
# =============================================================================


class EdgeFixedTupleMultiType:
    """CORRECT: Fixed-length tuple with different types is OK."""

    record: tuple[str, int, float, bool]  # OK: fixed-length


class EdgeEmptyTuple:
    """CORRECT: Empty tuple annotation is OK."""

    empty: tuple[()]  # OK: explicitly empty tuple


class EdgeTupleNoSubscript:
    """CORRECT: Bare tuple without subscript is OK."""

    any_tuple: tuple  # OK: not variable-length pattern


class EdgeTupleInFunctionOnly:
    """CORRECT: tuple[X, ...] in method signatures is allowed."""

    def method(self, items: tuple[int, ...]) -> tuple[str, ...]:
        """Both param and return can use tuple[X, ...]."""
        return tuple(str(x) for x in items)


# =============================================================================
# hashable-field: Nested Unhashable Edge Cases
# =============================================================================


@dataclass(frozen=True)
class EdgeHashableNestedSequenceInTuple:
    """VIOLATION: Sequence nested in tuple is still unhashable."""

    __strict_typing_linter__hashable_fields__ = True

    # tuple itself is hashable, but Sequence inside isn't
    data: tuple[Sequence[int], ...]  # hashable-field: nested Sequence


@dataclass(frozen=True)
class EdgeHashableNestedMappingInTuple:
    """VIOLATION: Mapping nested in tuple is still unhashable."""

    __strict_typing_linter__hashable_fields__ = True

    data: tuple[Mapping[str, int], ...]  # hashable-field: nested Mapping


@dataclass(frozen=True)
class EdgeHashableSequenceInAnnotated:
    """VIOLATION: Annotated doesn't protect Sequence from hashable check."""

    __strict_typing_linter__hashable_fields__ = True

    items: Annotated[Sequence[int], pydantic.Field()]  # hashable-field


# =============================================================================
# hashable-field: Union Edge Cases
# =============================================================================


@dataclass(frozen=True)
class EdgeHashableSequenceUnion:
    """VIOLATION: Sequence in union is unhashable."""

    __strict_typing_linter__hashable_fields__ = True

    items: Sequence[int] | None  # hashable-field


@dataclass(frozen=True)
class EdgeHashableMappingUnion:
    """VIOLATION: Mapping in union is unhashable."""

    __strict_typing_linter__hashable_fields__ = True

    data: Mapping[str, int] | None  # hashable-field


# =============================================================================
# hashable-field: list → tuple suggestion
# =============================================================================


@dataclass(frozen=True)
class EdgeHashableListSuggestion:
    """VIOLATION: list in hashable class suggests tuple[T, ...] not Sequence[T].

    The suggestion is different because Sequence would also be flagged.
    """

    __strict_typing_linter__hashable_fields__ = True

    items: list[int]  # mutable-type: suggests tuple[T, ...] in this context


@dataclass(frozen=True)
class EdgeHashableDictSuggestion:
    """VIOLATION: dict in hashable class suggests frozendict (not Mapping, which would also be flagged)."""

    __strict_typing_linter__hashable_fields__ = True

    data: dict[str, int]  # mutable-type: suggests frozendict in hashable context


# =============================================================================
# hashable-field: ClassVar is NOT an instance field
# =============================================================================


@dataclass(frozen=True)
class EdgeHashableClassVarSequence:
    """CORRECT: ClassVar[Sequence] doesn't affect __hash__."""

    __strict_typing_linter__hashable_fields__ = True

    CLASS_DATA: ClassVar[Sequence[int]] = []  # OK: ClassVar


@dataclass(frozen=True)
class EdgeHashableClassVarMapping:
    """CORRECT: ClassVar[Mapping] doesn't affect __hash__."""

    __strict_typing_linter__hashable_fields__ = True

    CLASS_MAP: ClassVar[Mapping[str, int]] = {}  # OK: ClassVar


@dataclass(frozen=True)
class EdgeHashableClassVarTuple:
    """CORRECT: ClassVar with tuple also OK."""

    __strict_typing_linter__hashable_fields__ = True

    CLASS_VALUES: ClassVar[tuple[int, ...]] = ()  # OK: ClassVar


# =============================================================================
# Context: Nested class independence
# =============================================================================


@dataclass(frozen=True)
class EdgeOuterHashable:
    """Outer class with hashable flag."""

    __strict_typing_linter__hashable_fields__ = True

    outer_data: tuple[int, ...]  # OK in outer

    class EdgeInnerNoInherit:
        """Nested class does NOT inherit hashable flag."""

        # In non-hashable context, tuple[X, ...] is flagged as tuple-field
        inner_values: tuple[int, ...]  # tuple-field: no inheritance


@dataclass(frozen=True)
class EdgeOuterHashableWithInnerDataclass:
    """Outer hashable with nested dataclass."""

    __strict_typing_linter__hashable_fields__ = True

    @dataclass(frozen=True)
    class EdgeInnerDataclass:
        """Nested dataclass must declare its own hashable flag."""

        # Without its own flag, this would trigger tuple-field
        items: tuple[int, ...]  # tuple-field: nested class independent


# =============================================================================
# Multiple Violations in One Field
# =============================================================================


class EdgeMultipleViolationsTupleAndMutable:
    """VIOLATION: Both tuple-field and potential mutable violations."""

    # tuple[X, ...] triggers tuple-field
    # The list inside is also checked and flagged
    nested: tuple[list[int], ...]  # tuple-field (and mutable inside)


@dataclass(frozen=True)
class EdgeHashableMultipleViolations:
    """VIOLATION: Multiple hashable violations."""

    __strict_typing_linter__hashable_fields__ = True

    # Mapping is unhashable, and the inner list is mutable
    data: Mapping[str, list[int]]  # hashable-field: Mapping unhashable


# =============================================================================
# String Annotations (PEP 563)
# =============================================================================


class EdgeStringAnnotationTuple:
    """VIOLATION: String annotation with tuple[X, ...] should still flag."""

    values: tuple[int, ...]  # tuple-field: string annotation parsed


class EdgeStringAnnotationNested:
    """VIOLATION: String annotation with nested tuple should flag."""

    data: Mapping[str, tuple[int, ...]]  # tuple-field: nested in string


# =============================================================================
# Non-frozen Dataclass - Fields Not Checked
# =============================================================================


@dataclass  # NOT frozen
class EdgeNonFrozenSkipped:
    """Non-frozen dataclass fields are skipped entirely."""

    items: list[int]  # OK: non-frozen dataclass
    values: tuple[int, ...]  # OK: non-frozen dataclass not checked
    data: Mapping[str, int]  # OK: non-frozen dataclass not checked


@dataclass(frozen=False)  # Explicitly not frozen
class EdgeExplicitlyNotFrozen:
    """Explicitly non-frozen dataclass also skipped."""

    items: list[int]  # OK: explicitly non-frozen


# NOTE: Non-frozen dataclass with __strict_typing_linter__hashable_fields__ = True
# would raise ValueError during linting (tested separately in error validation).


# =============================================================================
# Frozen Dataclass Without Hashable Flag
# =============================================================================


@dataclass(frozen=True)
class EdgeFrozenNoHashableFlag:
    """Frozen dataclass without hashable flag still checks tuple-field."""

    # Without __strict_typing_linter__hashable_fields__, tuple[X, ...] is flagged
    values: tuple[int, ...]  # tuple-field: not in hashable context


# =============================================================================
# Regular Class (Not Dataclass) with Hashable Flag
# =============================================================================


class EdgeRegularClassHashable:
    """Regular class (e.g., Pydantic model) can also use hashable flag."""

    __strict_typing_linter__hashable_fields__ = True

    # In hashable context, tuple[X, ...] is allowed
    values: tuple[int, ...]  # OK: hashable context


class EdgeRegularClassHashableSequence:
    """Regular class with hashable flag flags Sequence."""

    __strict_typing_linter__hashable_fields__ = True

    items: Sequence[int]  # hashable-field: Sequence in hashable class


# =============================================================================
# Suppression Directive Edge Cases
# =============================================================================


class EdgeSuppressedTupleField:
    """Suppression directive for tuple-field."""

    values: tuple[int, ...]  # strict_typing_linter.py: tuple-field


@dataclass(frozen=True)
class EdgeSuppressedHashableField:
    """Suppression directive for hashable-field."""

    __strict_typing_linter__hashable_fields__ = True

    items: Sequence[int]  # strict_typing_linter.py: hashable-field


class EdgeMultipleSuppressionCodes:
    """Multiple suppression codes on one line."""

    # If a field had both violations (tuple-field context):
    values: tuple[int, ...]  # strict_typing_linter.py: tuple-field, mutable-type


# =============================================================================
# Pydantic Model Examples
# =============================================================================


class EdgePydanticStrictModel(pydantic.BaseModel):
    """VIOLATION: Pydantic strict model with tuple field."""

    model_config = pydantic.ConfigDict(strict=True, frozen=True)

    values: tuple[float, ...]  # tuple-field


class EdgePydanticHashableModel(pydantic.BaseModel):
    """CORRECT: Pydantic model with hashable flag allows tuple."""

    model_config = pydantic.ConfigDict(strict=True, frozen=True)
    __strict_typing_linter__hashable_fields__ = True

    values: tuple[float, ...]  # OK: hashable context


class EdgePydanticHashableSequence(pydantic.BaseModel):
    """VIOLATION: Pydantic hashable model flags Sequence."""

    model_config = pydantic.ConfigDict(strict=True, frozen=True)
    __strict_typing_linter__hashable_fields__ = True

    items: Sequence[int]  # hashable-field


# =============================================================================
# attrs: Frozen Detection
# =============================================================================


@attrs.frozen
class EdgeAttrsFrozenTuple:
    """VIOLATION: Frozen attrs class — tuple[X, ...] flagged as tuple-field."""

    values: tuple[int, ...]  # tuple-field


@attrs.define
class EdgeAttrsNonFrozenSkipped:
    """CORRECT: Non-frozen attrs class fields are skipped entirely."""

    items: list[int]  # OK: non-frozen attrs not checked
    values: tuple[int, ...]  # OK: non-frozen attrs not checked


@attrs.define(frozen=True)
class EdgeAttrsDefineFrozenTuple:
    """VIOLATION: @define(frozen=True) detected as frozen."""

    values: tuple[int, ...]  # tuple-field


@attrs.frozen
class EdgeAttrsFrozenHashable:
    """CORRECT: Frozen attrs with hashable flag allows tuple."""

    __strict_typing_linter__hashable_fields__ = True

    values: tuple[int, ...]  # OK: hashable context


@attrs.frozen
class EdgeAttrsFrozenHashableSequence:
    """VIOLATION: Frozen attrs with hashable flag flags Sequence."""

    __strict_typing_linter__hashable_fields__ = True

    items: Sequence[int]  # hashable-field


# =============================================================================
# Nested Class Inside Non-Frozen Dataclass (regression: _skip_class_fields leak)
# =============================================================================


@dataclass
class EdgeNonFrozenOuterWithInner:
    """Non-frozen outer — fields skipped. Inner class must NOT inherit skip."""

    items: list[int]  # OK: non-frozen dataclass

    class EdgeInnerInsideNonFrozen:
        """Inner class must be checked independently."""

        values: tuple[int, ...]  # tuple-field: inner is not a non-frozen dataclass


@dataclass
class EdgeDeeplyNested4Levels:
    """Four levels of nesting tests save/restore state at arbitrary depth."""

    class Level2:
        @dataclass(frozen=True)
        class Level3:
            class Level4:
                values: tuple[int, ...]  # tuple-field: deepest level checked


# =============================================================================
# Stub Helpers
# =============================================================================


def edge_function_params_ok(
    items: tuple[int, ...],
    data: Sequence[str],
) -> tuple[str, ...]:
    """Functions can use tuple[X, ...] and Sequence freely."""
    return tuple(str(x) for x in items)
