#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Runtime hashability inspector for user-defined types.

Companion module to strict_typing_linter.py. Imports classes at linter runtime
and inspects their field metadata via framework-specific APIs (attrs, dataclasses,
Pydantic) to recursively verify that all fields in hashable-marked classes are
actually hashable.

When the linter encounters a class with __strict_typing_linter__hashable_fields__ = True
and a field type classified as 'unknown' (user-defined type), it delegates to this module
for runtime verification.

Framework support: attrs classes, dataclasses, Pydantic BaseModel.
Generic classes without framework metadata return None (unverifiable).

Standalone usage (requires --project for project-specific types):
    uv run --project mcp/document-search python scripts/hashability_inspector.py \
        document_search.schemas.vectors.VectorPoint

When imported by strict_typing_linter.py, the linter passes source_roots (scan directories)
which the inspector adds to sys.path so project types are importable.
Import failures from missing packages raise ImportError (not swallowed).
"""

from __future__ import annotations

__all__ = [
    'HashabilityInspector',
    'InspectionResult',
    'QualifiedName',
    'UnhashableField',
]

import collections.abc
import dataclasses
import enum
import importlib
import logging
import sys
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fully qualified dotted type name (e.g., 'document_search.schemas.vectors.VectorPoint')
type QualifiedName = str

# Runtime type annotation object (from typing.get_type_hints, get_origin, get_args).
# No standard type exists yet — see Python discuss.python.org/t/typeform-spelling/51435.
type TypeAnnotation = Any


# =============================================================================
# Known type classifications (avoids importing external packages)
# =============================================================================

HASHABLE_PRIMITIVES: frozenset[type] = frozenset(
    {
        int,
        str,
        float,
        bool,
        bytes,
        complex,
        type(None),
    }
)

UNHASHABLE_TYPES: frozenset[type] = frozenset(
    {
        list,
        dict,
        set,
        bytearray,
    }
)

UNHASHABLE_ORIGINS: frozenset[type] = frozenset(
    {
        list,
        dict,
        set,
        collections.abc.Sequence,
        collections.abc.MutableSequence,
        collections.abc.Mapping,
        collections.abc.MutableMapping,
        collections.abc.Set,
        collections.abc.MutableSet,
    }
)

# External types we can classify without importing
KNOWN_HASHABLE: frozenset[QualifiedName] = frozenset(
    {
        'uuid.UUID',
        'datetime.datetime',
        'datetime.date',
        'datetime.time',
        'datetime.timedelta',
        'datetime.timezone',
        'decimal.Decimal',
        'fractions.Fraction',
        'pathlib.Path',
        'pathlib.PurePath',
        'pathlib.PosixPath',
        'pathlib.WindowsPath',
        'ipaddress.IPv4Address',
        'ipaddress.IPv6Address',
        're.Pattern',
    }
)

KNOWN_UNHASHABLE: frozenset[QualifiedName] = frozenset(
    {
        'numpy.ndarray',
        'numpy.matrix',
        'pandas.DataFrame',
        'pandas.Series',
    }
)


# =============================================================================
# Result types
# =============================================================================


@dataclass(frozen=True)
class UnhashableField:
    """A single unhashable field found during inspection."""

    name: str
    annotation_repr: str


@dataclass(frozen=True)
class InspectionResult:
    """Result of hashability inspection for a user-defined type."""

    is_hashable: bool
    type_name: str
    qualified_name: QualifiedName
    unhashable_fields: Sequence[UnhashableField]


# =============================================================================
# Inspector
# =============================================================================


class HashabilityInspector:
    """Runtime hashability inspector for user-defined types.

    Created once per linter run. Caches results across files to avoid
    redundant imports and analysis. Not thread-safe.

    Args:
        source_roots: Directories to add to sys.path for importing project types.
            Typically the top-level scan directories passed to the linter
            (e.g., 'mcp/document-search/' which contains 'document_search/').
    """

    def __init__(self, source_roots: tuple[Path, ...] = ()) -> None:
        self._cache: dict[QualifiedName, InspectionResult | None] = {}
        self._in_progress: set[QualifiedName] = set()

        # Append source roots to sys.path so project types are importable.
        # Uses append (not insert) to preserve stdlib priority and avoid shadowing.
        # Normalize existing entries for reliable dedup (sys.path may contain '', '.', etc.)
        existing = {str(Path(p).resolve()) for p in sys.path if p}
        for root in source_roots:
            resolved = str(root.resolve())
            if resolved not in existing:
                sys.path.append(resolved)
                existing.add(resolved)

    def check(self, qualified_name: QualifiedName) -> InspectionResult | None:
        """Check if a user-defined type is hashable via runtime inspection.

        Args:
            qualified_name: Fully qualified type name from the linter's import map
                            (e.g., 'document_search.schemas.vectors.VectorPoint')

        Returns:
            InspectionResult if the type could be inspected,
            None if the type couldn't be imported or isn't a recognized framework class.
        """
        # Whitelist/blacklist check (no import needed)
        if qualified_name in KNOWN_HASHABLE:
            return InspectionResult(
                is_hashable=True,
                type_name=qualified_name.rsplit('.', 1)[-1],
                qualified_name=qualified_name,
                unhashable_fields=(),
            )
        if qualified_name in KNOWN_UNHASHABLE:
            return InspectionResult(
                is_hashable=False,
                type_name=qualified_name.rsplit('.', 1)[-1],
                qualified_name=qualified_name,
                unhashable_fields=(),
            )

        # Cache check
        if qualified_name in self._cache:
            logger.debug('[HASH] Cache hit: %s', qualified_name)
            return self._cache[qualified_name]

        # Cycle detection
        if qualified_name in self._in_progress:
            logger.debug('[HASH] Cycle detected: %s (returning None)', qualified_name)
            return None

        self._in_progress.add(qualified_name)
        try:
            result = self._verify(qualified_name)
            self._cache[qualified_name] = result
            return result
        finally:
            self._in_progress.discard(qualified_name)

    # -----------------------------------------------------------------
    # Import resolution
    # -----------------------------------------------------------------

    def _import_class(self, qualified_name: QualifiedName) -> type | None:
        """Import a class by qualified name, handling nested classes.

        Tries progressively shorter module paths to support nested classes
        like 'pkg.module.OuterClass.InnerClass'.
        """
        parts = qualified_name.split('.')
        if len(parts) < 2:
            return None

        for i in range(len(parts) - 1, 0, -1):
            module_name = '.'.join(parts[:i])
            class_path = parts[i:]

            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            obj: Any = module
            for attr_name in class_path:
                obj = getattr(obj, attr_name, None)
                if obj is None:
                    break

            if isinstance(obj, type):
                return obj

        return None

    # -----------------------------------------------------------------
    # Core verification dispatch
    # -----------------------------------------------------------------

    def _verify(self, qualified_name: QualifiedName) -> InspectionResult | None:
        """Import and verify a type's hashability."""
        cls = self._import_class(qualified_name)
        if cls is None:
            logger.debug('[HASH] Could not import: %s', qualified_name)
            return None

        type_name = cls.__name__
        logger.debug('[HASH] Inspecting %s (%s)', qualified_name, cls)

        # Dispatch order: attrs → Pydantic → dataclass (Pydantic before dataclass
        # because pydantic.dataclasses pass is_dataclass but need BaseModel-style checking)
        if _has_attrs(cls):
            logger.debug('[HASH]   -> attrs class')
            return self._check_attrs(cls, type_name, qualified_name)

        if _is_pydantic_model(cls):
            logger.debug('[HASH]   -> Pydantic model')
            return self._check_pydantic(cls, type_name, qualified_name)

        if _is_dataclass(cls):
            logger.debug('[HASH]   -> dataclass')
            return self._check_dataclass(cls, type_name, qualified_name)

        # No recognized framework — can't verify
        logger.debug('[HASH]   -> no framework metadata, skipping')
        return None

    # -----------------------------------------------------------------
    # attrs checker
    # -----------------------------------------------------------------

    def _check_attrs(self, cls: type, type_name: str, qualified_name: QualifiedName) -> InspectionResult:
        """Verify attrs class hashability via field inspection."""
        import attrs

        # Non-frozen attrs classes have __hash__ set to None
        if getattr(cls, '__hash__', None) is None:
            logger.debug('[HASH]     no __hash__ -> unhashable')
            return InspectionResult(
                is_hashable=False,
                type_name=type_name,
                qualified_name=qualified_name,
                unhashable_fields=(),
            )

        # Resolve string annotations (from __future__ import annotations)
        type_hints = typing.get_type_hints(cls)

        unhashable: list[UnhashableField] = []

        for field in attrs.fields(cls):
            # Skip fields excluded from hash
            if field.hash is False:
                continue

            # Prefer resolved type hints, fall back to field.type
            annotation = type_hints.get(field.name, field.type)

            # String annotation we couldn't resolve — skip
            if isinstance(annotation, str) or annotation is None:
                continue

            if not self._is_annotation_hashable(annotation):
                unhashable.append(
                    UnhashableField(
                        name=field.name,
                        annotation_repr=_annotation_repr(annotation),
                    )
                )
                logger.debug('[HASH]     %s: %s -> unhashable', field.name, annotation)

        return InspectionResult(
            is_hashable=len(unhashable) == 0,
            type_name=type_name,
            qualified_name=qualified_name,
            unhashable_fields=tuple(unhashable),
        )

    # -----------------------------------------------------------------
    # Dataclass checker
    # -----------------------------------------------------------------

    def _check_dataclass(self, cls: type, type_name: str, qualified_name: QualifiedName) -> InspectionResult:
        """Verify dataclass hashability via field inspection."""
        # Non-frozen dataclasses have __hash__ set to None
        if getattr(cls, '__hash__', None) is None:
            logger.debug('[HASH]     no __hash__ -> unhashable')
            return InspectionResult(
                is_hashable=False,
                type_name=type_name,
                qualified_name=qualified_name,
                unhashable_fields=(),
            )

        # Resolve string annotations (from __future__ import annotations)
        type_hints = typing.get_type_hints(cls)

        unhashable: list[UnhashableField] = []

        for field in dataclasses.fields(cls):
            # Skip fields excluded from hash
            # field.hash: None = use compare (default True), True = include, False = exclude
            if field.hash is False:
                continue

            # Prefer resolved type hints, fall back to field.type
            annotation = type_hints.get(field.name, field.type)

            # String annotation we couldn't resolve — skip
            if isinstance(annotation, str) or annotation is None:
                continue

            if not self._is_annotation_hashable(annotation):
                unhashable.append(
                    UnhashableField(
                        name=field.name,
                        annotation_repr=_annotation_repr(annotation),
                    )
                )
                logger.debug('[HASH]     %s: %s -> unhashable', field.name, annotation)

        return InspectionResult(
            is_hashable=len(unhashable) == 0,
            type_name=type_name,
            qualified_name=qualified_name,
            unhashable_fields=tuple(unhashable),
        )

    # -----------------------------------------------------------------
    # Pydantic checker
    # -----------------------------------------------------------------

    def _check_pydantic(self, cls: type, type_name: str, qualified_name: QualifiedName) -> InspectionResult:
        """Verify Pydantic model hashability via model_fields inspection."""
        model_config = getattr(cls, 'model_config', {})
        is_frozen = model_config.get('frozen', False)

        if not is_frozen:
            logger.debug('[HASH]     not frozen -> unhashable')
            return InspectionResult(
                is_hashable=False,
                type_name=type_name,
                qualified_name=qualified_name,
                unhashable_fields=(),
            )

        # model_fields gives resolved annotations (Annotated stripped, forward refs resolved)
        model_fields = getattr(cls, 'model_fields', {})
        unhashable: list[UnhashableField] = []

        for field_name, field_info in model_fields.items():
            annotation = field_info.annotation
            if not self._is_annotation_hashable(annotation):
                unhashable.append(
                    UnhashableField(
                        name=field_name,
                        annotation_repr=_annotation_repr(annotation),
                    )
                )
                logger.debug('[HASH]     %s: %s -> unhashable', field_name, annotation)

        return InspectionResult(
            is_hashable=len(unhashable) == 0,
            type_name=type_name,
            qualified_name=qualified_name,
            unhashable_fields=tuple(unhashable),
        )

    # -----------------------------------------------------------------
    # Recursive annotation checker
    # -----------------------------------------------------------------

    def _is_annotation_hashable(self, annotation: TypeAnnotation) -> bool:
        """Recursively check if a type annotation represents a hashable type."""
        # None / NoneType
        if annotation is None or annotation is type(None):
            return True

        # Unresolved forward reference
        if isinstance(annotation, str):
            return True

        # Known primitives
        if annotation in HASHABLE_PRIMITIVES:
            return True
        if annotation in UNHASHABLE_TYPES:
            return False

        # Generic types (list[int], Sequence[str], tuple[int, ...], Union, etc.)
        origin = typing.get_origin(annotation)
        if origin is not None:
            return self._check_generic_origin(origin, typing.get_args(annotation))

        # Concrete type — recurse via framework inspection
        if isinstance(annotation, type):
            return self._check_concrete_type(annotation)

        # TypeVar, ParamSpec, etc. — can't verify
        return True

    def _check_concrete_type(self, cls: type) -> bool:
        """Check a concrete type's hashability."""
        # Enums are always hashable
        if issubclass(cls, enum.Enum):
            return True

        # Recurse into the inspector for framework-supported types
        qualified = f'{cls.__module__}.{cls.__qualname__}'
        result = self.check(qualified)
        if result is not None:
            return result.is_hashable

        # No framework metadata — fall back to __hash__ existence
        return getattr(cls, '__hash__', None) is not None

    def _check_generic_origin(self, origin: TypeAnnotation, args: tuple[TypeAnnotation, ...]) -> bool:
        """Check a generic type's hashability based on its origin and args."""
        # Mutable containers and abstract types that could be mutable
        if origin in UNHASHABLE_ORIGINS:
            return False

        # tuple — immutable, check contents
        if origin is tuple:
            return all(self._is_annotation_hashable(arg) for arg in args if arg is not Ellipsis)

        # frozenset — enforces hashability of contents
        if origin is frozenset:
            return True

        # Union types (typing.Union and types.UnionType for X | Y syntax)
        if origin is typing.Union or origin is types.UnionType:
            return all(self._is_annotation_hashable(arg) for arg in args)

        # Annotated — check first arg only (rest is metadata)
        if origin is typing.Annotated:
            return self._is_annotation_hashable(args[0]) if args else True

        # Literal — literal values are always hashable
        if origin is typing.Literal:
            return True

        # Unknown generic — assume hashable
        return True


# =============================================================================
# Framework detection (duck typing, no imports)
# =============================================================================


def _has_attrs(cls: type) -> bool:
    """Check if cls is an attrs class."""
    return hasattr(cls, '__attrs_attrs__')


def _is_dataclass(cls: type) -> bool:
    """Check if cls is a dataclass (stdlib or pydantic.dataclasses)."""
    return dataclasses.is_dataclass(cls) and isinstance(cls, type)


def _is_pydantic_model(cls: type) -> bool:
    """Check if cls is a Pydantic BaseModel subclass."""
    return hasattr(cls, 'model_fields') and hasattr(cls, 'model_config')


# =============================================================================
# Utilities
# =============================================================================


def _annotation_repr(annotation: TypeAnnotation) -> str:
    """Human-readable representation of a type annotation."""
    # typing types have good __repr__ already
    if hasattr(annotation, '__origin__'):
        return str(annotation)

    if isinstance(annotation, type):
        return annotation.__qualname__

    return repr(annotation)


# =============================================================================
# Standalone CLI
# =============================================================================


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    if len(sys.argv) < 2:
        print('Usage: uv run scripts/hashability_inspector.py <qualified_name> [...]')
        print('Example: uv run scripts/hashability_inspector.py document_search.schemas.vectors.VectorPoint')
        sys.exit(2)

    inspector = HashabilityInspector()
    exit_code = 0

    for name in sys.argv[1:]:
        result = inspector.check(name)
        if result is None:
            print(f'{name}: could not verify (not importable or no framework metadata)')
        elif result.is_hashable:
            print(f'{name}: hashable')
        else:
            print(f'{name}: UNHASHABLE')
            exit_code = 1
            for f in result.unhashable_fields:
                print(f'  - {f.name}: {f.annotation_repr}')

    sys.exit(exit_code)
