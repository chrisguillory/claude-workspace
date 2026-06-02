# strict_typing_linter.py is the linter under test; the others skip this file.
# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""``Used``/``Unused`` fixtures for strict_typing_linter --report-unused-directives (type rules).

Each dataclass isolates one type rule (mutable-type, loose-typing, tuple-field, hashable-field).
A ``Used*`` class's field triggers the rule so its directive suppresses a real violation; an
``Unused*`` class's field does not, so its directive matches nothing. tuple-field is a non-frozen
concern (it wants ``Sequence``); hashable-field is a frozen concern (``Sequence``/``Mapping`` are
unhashable); concrete ``dict``/``list`` are mutable-type even when frozen. A valid ordered ``__all__``
keeps the module free of structural (missing-all / trailing-comma) noise; those rules live in their
own single-purpose fixtures under ``strict_typing/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    'UnusedHashableField',
    'UnusedLooseTyping',
    'UnusedMutableType',
    'UnusedTupleField',
    'UsedHashableField',
    'UsedLooseTyping',
    'UsedMutableType',
    'UsedTupleField',
]


@dataclass(frozen=True)
class UsedMutableType:
    """A concrete dict field fires mutable-type (suggests Mapping)."""

    value: dict[str, int]  # strict_typing_linter.py: mutable-type


@dataclass(frozen=True)
class UnusedMutableType:
    """A scalar field is fine, so the mutable-type directive matches nothing."""

    value: int  # strict_typing_linter.py: mutable-type


@dataclass(frozen=True)
class UsedLooseTyping:
    """An ``Any`` field fires loose-typing."""

    value: Any  # strict_typing_linter.py: loose-typing


@dataclass(frozen=True)
class UnusedLooseTyping:
    """A concrete field is fine, so the loose-typing directive matches nothing."""

    value: int  # strict_typing_linter.py: loose-typing


class UsedTupleField:
    """A variable-length tuple field fires tuple-field (suggests Sequence)."""

    value: tuple[int, ...]  # strict_typing_linter.py: tuple-field


class UnusedTupleField:
    """A fixed-length tuple field is fine, so the tuple-field directive matches nothing."""

    value: tuple[int, int]  # strict_typing_linter.py: tuple-field


@dataclass(frozen=True)
class UsedHashableField:
    """In a hashable-marked frozen dataclass, a Sequence field fires hashable-field."""

    __strict_typing_linter__hashable_fields__ = True

    value: Sequence[int]  # strict_typing_linter.py: hashable-field


@dataclass(frozen=True)
class UnusedHashableField:
    """A hashable scalar field is fine, so the hashable-field directive matches nothing."""

    __strict_typing_linter__hashable_fields__ = True

    value: int  # strict_typing_linter.py: hashable-field
