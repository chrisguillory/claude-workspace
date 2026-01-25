#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Strict typing linter for Python annotations.

This script enforces immutable interface types and module organization:
1. IMMUTABLE TYPES: Use Sequence/Mapping/Set instead of list/dict/set
2. STRICT TYPING: No Any, Mapping[str, Any], Sequence[Any] etc.
3. MODULE ORDERING (opt-in): Public before private, classes before functions

Checks ALL function/method parameters, return types, and class field annotations.
Skips non-frozen dataclasses (mutable by design).
Respects .gitignore when scanning directories.

Module ordering is opt-in via __strict_module_ordering__ = True in package __init__.py.
When enabled, all submodules in that package must:
- Define __all__
- Order definitions: __all__ items first, then public, then private
- Within each group: classes before functions

Design Philosophy:
    - Error-only, no auto-fix: Forces conscious decision at each occurrence
    - Pure analysis: Checks exactly the files it's given (no internal filtering)
    - Escape hatches:
      - # strict_typing_linter.py: mutable-type - suppress mutable type violations
      - # strict_typing_linter.py: loose-typing - suppress loose typing violations
      - # strict_typing_linter.py: ordering - suppress ordering violations

Usage:
    ./scripts/strict_typing_linter.py <files...>
    ./scripts/strict_typing_linter.py .          # All .py files in cwd

File selection is handled by the caller (pre-commit, shell globs, etc.).
This script is a pure analysis function - it checks what it's given.

Exit codes:
    0 - No violations found
    1 - Violations found
"""

from __future__ import annotations

import argparse
import ast
import io
import subprocess
import sys
import tokenize
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# =============================================================================
# Configuration - What to check (not WHERE to check - that's the caller's job)
# =============================================================================

# Builtin mutable types (lowercase) - always flagged
MUTABLE_TYPES: Set[str] = {
    'list',
    'dict',
    'set',
}

# Types that trigger loose-typing errors
LOOSE_TYPES: Set[str] = {
    'Any',
}

# Combined forbidden types (short names only)
FORBIDDEN_TYPES: Set[str] = MUTABLE_TYPES | LOOSE_TYPES

# Fully qualified mutable types (typing module aliases for builtins + explicit mutable interfaces)
QUALIFIED_MUTABLE: Set[str] = {
    'typing.List',
    'typing.Dict',
    'typing.Set',
    'typing.MutableMapping',
    'typing.MutableSequence',
    'typing.MutableSet',
    'collections.abc.MutableMapping',
    'collections.abc.MutableSequence',
    'collections.abc.MutableSet',
}

# Fully qualified allowed types (abstract interfaces)
QUALIFIED_ALLOWED: Set[str] = {
    'collections.abc.Set',
    'collections.abc.Mapping',
    'collections.abc.Sequence',
    'typing.AbstractSet',
    'typing.Mapping',
    'typing.Sequence',
}

# Fully qualified transparent types (check nested contents)
QUALIFIED_TRANSPARENT: Set[str] = {
    'collections.abc.Mapping',
    'collections.abc.Sequence',
    'typing.Mapping',
    'typing.Sequence',
}

# Immutable containers that don't need content checking
# Note: tuple is NOT here - we check its contents because tuple[list, int] is still problematic
# frozenset elements must be hashable, so mutable types are impossible anyway
ALLOWED_CONTAINERS: Set[str] = {
    'frozenset',
    'FrozenSet',
}

# Position-aware allowlist for Any in generic type parameters (library boundaries)
# Maps fully qualified type name to positions where Any is acceptable (0-indexed)
# None = all positions allowed, tuple of ints = only those positions allowed
# Names are resolved via import map, so only canonical names needed here
ANY_ALLOWED_POSITIONS: Mapping[str, Sequence[int] | None] = {
    # FastMCP Context[ServerSessionT, LifespanContextT, RequestT] - all positions
    'mcp.server.fastmcp.Context': None,
    # Generator[YieldType, SendType, ReturnType] - SendType and ReturnType often unused
    'typing.Generator': (1, 2),
    'collections.abc.Generator': (1, 2),
    # AsyncGenerator[YieldType, SendType] - SendType often unused
    'typing.AsyncGenerator': (1,),
    'collections.abc.AsyncGenerator': (1,),
}


def _build_import_map(tree: ast.Module) -> Mapping[str, str]:
    """Build a mapping from local names to fully qualified names from imports.

    Examples:
        import typing -> {'typing': 'typing'}
        import typing as t -> {'t': 'typing'}
        from typing import Generator -> {'Generator': 'typing.Generator'}
        from typing import Generator as Gen -> {'Gen': 'typing.Generator'}
    """
    import_map: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name
                import_map[local_name] = alias.name

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    import_map[local_name] = f'{node.module}.{alias.name}'

    return import_map


def _resolve_qualified_name(node: ast.expr, import_map: Mapping[str, str]) -> str:
    """Resolve a type node to its fully qualified canonical name.

    Uses the import map to resolve local names to their fully qualified origins.
    For attribute access (e.g., typing.Generator), resolves the base module.
    """
    if isinstance(node, ast.Name):
        # Simple name - look up in import map
        return import_map.get(node.id, node.id)

    elif isinstance(node, ast.Attribute):
        # Qualified name like typing.Generator or t.Generator
        # Get the base (could be a module alias)
        base_node = node.value
        if isinstance(base_node, ast.Name):
            # Resolve the base module name
            resolved_base = import_map.get(base_node.id, base_node.id)
            return f'{resolved_base}.{node.attr}'
        else:
            # Nested attributes - recursively resolve
            resolved_base = _resolve_qualified_name(base_node, import_map)
            return f'{resolved_base}.{node.attr}'

    return ''


# Directive prefix uses the script filename for discoverability - readers immediately
# know which tool owns the directive, and searching for the filename finds both the
# script and all its annotations in the codebase.
DIRECTIVE_PREFIX = '# strict_typing_linter.py:'


# =============================================================================
# Data Types
# =============================================================================


# Type aliases for constrained string values
type FieldContext = Literal['field', 'parameter', 'return']
type DirectiveCode = Literal['mutable-type', 'loose-typing', 'ordering']

# Violation kind literals - used in discriminated unions
type TypeViolationKind = Literal['mutable', 'loose']
type OrderingViolationKind = Literal['ordering', 'missing-all', 'trailing-comma']
type ViolationKind = TypeViolationKind | OrderingViolationKind


@dataclass(frozen=True)
class TypeViolation:
    """A type annotation violation (mutable or loose types)."""

    filepath: Path
    line: int
    column: int
    context: FieldContext
    bad_type: str
    kind: Literal['mutable', 'loose']
    source_line: str
    suggestion: str


@dataclass(frozen=True)
class OrderingViolation:
    """A module ordering violation."""

    filepath: Path
    line: int
    kind: Literal['ordering', 'missing-all', 'trailing-comma']
    message: str
    source_line: str
    suggestion: str


# Discriminated union of all violation types
type Violation = TypeViolation | OrderingViolation


# =============================================================================
# AST Visitor
# =============================================================================


def _is_frozen_dataclass(node: ast.ClassDef) -> bool | None:
    """Check if class has @dataclass decorator and whether it's frozen.

    Returns:
        True if @dataclass(frozen=True)
        False if @dataclass (not frozen)
        None if not a dataclass
    """
    for decorator in node.decorator_list:
        # @dataclass (no args)
        if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
            return False
        # @dataclasses.dataclass (qualified name, no args)
        if isinstance(decorator, ast.Attribute) and decorator.attr == 'dataclass':
            return False
        # @dataclass(...) or @dataclasses.dataclass(...) with args
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == 'dataclass':
                # Check for frozen=True
                for keyword in decorator.keywords:
                    if keyword.arg == 'frozen':
                        if isinstance(keyword.value, ast.Constant):
                            return keyword.value.value is True
                return False
            # dataclasses.dataclass(...)
            if isinstance(func, ast.Attribute) and func.attr == 'dataclass':
                for keyword in decorator.keywords:
                    if keyword.arg == 'frozen':
                        if isinstance(keyword.value, ast.Constant):
                            return keyword.value.value is True
                return False
    return None


class AnnotationChecker(ast.NodeVisitor):
    """AST visitor that checks for mutable and loose types in annotations."""

    SUGGESTIONS: Mapping[str, str] = {
        # Mutable type suggestions
        'list': 'Sequence[T] or tuple[T, ...]',
        'List': 'Sequence[T] or tuple[T, ...]',
        'dict': 'Mapping[K, V]',
        'Dict': 'Mapping[K, V]',
        'set': 'Set[T] (from collections.abc)',
        'Set': 'Set[T] (from collections.abc)',
        # Loose type suggestions
        'Any': 'a specific type (TypedDict, StrictModel, or concrete type)',
    }

    def __init__(self, filepath: Path, source_lines: Sequence[str], import_map: Mapping[str, str]) -> None:
        self.filepath = filepath
        self.source_lines = source_lines
        self.import_map = import_map
        self.violations: list[Violation] = []  # Internal - OK to be mutable
        self._function_depth = 0  # Track nesting depth in functions
        self._skip_class_fields = False  # Skip field checking for non-frozen dataclasses

    def _get_type_name(self, node: ast.expr) -> str:
        """Extract the name from a type expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            return self._get_type_name(node.value)
        return ''

    def _classify_by_qualified_name(self, node: ast.expr) -> Literal['mutable', 'transparent', 'allowed', 'unknown']:
        """Classify type using fully qualified name resolution.

        Used to disambiguate types like 'Set' which could be:
        - typing.Set (mutable, deprecated)
        - collections.abc.Set (abstract interface, allowed)

        Returns 'unknown' when qualified name doesn't provide disambiguation,
        signaling the caller to fall back to short name matching.
        """
        qualified = _resolve_qualified_name(node, self.import_map)

        if qualified in QUALIFIED_MUTABLE:
            return 'mutable'
        if qualified in QUALIFIED_TRANSPARENT:
            return 'transparent'
        if qualified in QUALIFIED_ALLOWED:
            return 'allowed'
        return 'unknown'

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definitions, skipping non-frozen dataclasses."""
        frozen = _is_frozen_dataclass(node)

        # Save previous state
        prev_skip = self._skip_class_fields

        # Skip field checking for non-frozen dataclasses (mutable by design)
        if frozen is False:
            self._skip_class_fields = True

        # Visit children
        self.generic_visit(node)

        # Restore state
        self._skip_class_fields = prev_skip

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check annotated assignments (x: Type = value)."""
        # Check all annotated assignments at module/class level (not in functions)
        # Skip if we're in a non-frozen dataclass
        if self._function_depth == 0 and not self._skip_class_fields:
            self._check_annotation(node.annotation, node.lineno, 'field')
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function parameter annotations."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check async function parameter annotations."""
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Shared logic for visiting function definitions."""
        # Check ALL function parameters
        self._check_function_parameters(node)

        # Check return type annotation
        if node.returns:
            self._check_annotation(node.returns, node.lineno, 'return')

        # Track that we're in a function body
        self._function_depth += 1

        # Visit function body
        self.generic_visit(node)

        self._function_depth -= 1

    def _check_function_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check parameter annotations on a function/method."""
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs

        for arg in all_args:
            if arg.annotation:
                self._check_annotation(arg.annotation, arg.lineno, 'parameter')

        # Check *args annotation
        if node.args.vararg and node.args.vararg.annotation:
            self._check_annotation(node.args.vararg.annotation, node.args.vararg.lineno, 'parameter')

        # Check **kwargs annotation
        if node.args.kwarg and node.args.kwarg.annotation:
            self._check_annotation(node.args.kwarg.annotation, node.args.kwarg.lineno, 'parameter')

    def _check_annotation(self, node: ast.expr, lineno: int, context: FieldContext) -> None:
        """Check if an annotation contains forbidden types."""
        bad_type = self._find_forbidden_type(node)
        if bad_type:
            # Determine violation kind
            kind: TypeViolationKind = 'loose' if bad_type in LOOSE_TYPES else 'mutable'

            # Check for directive comment with appropriate code
            if self._has_directive(lineno, kind):
                return

            # Get the source line for context
            source_line = ''
            if 0 < lineno <= len(self.source_lines):
                source_line = self.source_lines[lineno - 1].strip()

            self.violations.append(
                TypeViolation(
                    filepath=self.filepath,
                    line=lineno,
                    column=getattr(node, 'col_offset', 0),
                    context=context,
                    bad_type=bad_type,
                    kind=kind,
                    source_line=source_line,
                    suggestion=self.SUGGESTIONS.get(bad_type, 'a more specific type'),
                )
            )

    def _find_forbidden_type(self, node: ast.expr) -> str | None:
        """Recursively find if annotation contains a forbidden type.

        Returns the forbidden type name if found, None otherwise.
        Uses qualified name resolution for all types.
        """
        if isinstance(node, ast.Name):
            # Resolve and classify
            classification = self._classify_by_qualified_name(node)
            if classification == 'mutable':
                return node.id
            if classification in ('transparent', 'allowed'):
                return None

            # Fall back to short name for builtins
            if node.id in FORBIDDEN_TYPES:
                return node.id
            return None

        elif isinstance(node, ast.Subscript):
            container_name = self._get_type_name(node.value)

            # Resolve and classify the container
            classification = self._classify_by_qualified_name(node.value)
            if classification == 'mutable':
                return container_name
            if classification == 'transparent':
                return self._find_forbidden_in_slice(node.slice)
            if classification == 'allowed':
                return None

            # Fall back to short name checks for builtins
            if container_name in FORBIDDEN_TYPES:
                return container_name

            # Immutable containers that don't need content checking
            if container_name in ALLOWED_CONTAINERS:
                return None

            # Resolve qualified name for remaining checks
            resolved_name = _resolve_qualified_name(node.value, self.import_map)

            # Union, Optional - check all components
            if resolved_name in (
                'typing.Union',
                'typing.Optional',
                'typing_extensions.Union',
                'typing_extensions.Optional',
            ):
                return self._find_forbidden_in_slice(node.slice)

            # Literal types contain values, not type annotations - skip entirely
            if resolved_name in ('typing.Literal', 'typing_extensions.Literal'):
                return None

            # Annotated types: only check the first argument (the actual type), skip metadata
            if resolved_name in ('typing.Annotated', 'typing_extensions.Annotated'):
                if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                    return self._find_forbidden_type(node.slice.elts[0])
                return self._find_forbidden_type(node.slice)

            # Position-aware Any allowances for library types
            if resolved_name in ANY_ALLOWED_POSITIONS:
                allowed_positions = ANY_ALLOWED_POSITIONS[resolved_name]
                return self._find_forbidden_in_slice_with_positions(node.slice, allowed_positions)

            # Unknown container - recurse into slice to be safe
            return self._find_forbidden_in_slice(node.slice)

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type: X | Y
            left_violation = self._find_forbidden_type(node.left)
            if left_violation:
                return left_violation
            return self._find_forbidden_type(node.right)

        elif isinstance(node, ast.Attribute):
            # Qualified name like 'typing.List' - resolve and classify
            classification = self._classify_by_qualified_name(node)
            if classification == 'mutable':
                return node.attr
            if classification in ('transparent', 'allowed'):
                return None

            # Fall back to short name for builtins
            if node.attr in FORBIDDEN_TYPES:
                return node.attr
            return None

        elif isinstance(node, ast.Constant):
            # String annotation (forward reference or PEP 563)
            if isinstance(node.value, str):
                return self._check_string_annotation(node.value)
            return None

        elif isinstance(node, ast.Tuple):
            # Multiple types (e.g., in Union slice)
            for element in node.elts:
                violation = self._find_forbidden_type(element)
                if violation:
                    return violation
            return None

        elif isinstance(node, ast.List):
            # Callable parameter lists: Callable[[int, str], ReturnType]
            for element in node.elts:
                violation = self._find_forbidden_type(element)
                if violation:
                    return violation
            return None

        return None

    def _find_forbidden_in_slice(self, node: ast.expr) -> str | None:
        """Check for forbidden types in a subscript slice."""
        if isinstance(node, ast.Tuple):
            # Multiple type args like Mapping[str, int]
            for element in node.elts:
                violation = self._find_forbidden_type(element)
                if violation:
                    return violation
            return None
        else:
            # Single type arg like Sequence[str]
            return self._find_forbidden_type(node)

    def _find_forbidden_in_slice_with_positions(
        self, node: ast.expr, allowed_positions: Sequence[int] | None
    ) -> str | None:
        """Check for forbidden types in a subscript slice with position-aware Any allowances.

        Args:
            node: The slice node (single type or tuple of types)
            allowed_positions: Positions where Any is allowed (0-indexed), or None for all

        Returns:
            The forbidden type name if found, None otherwise.
        """
        if allowed_positions is None:
            # All positions allow Any - only check for mutable types
            if isinstance(node, ast.Tuple):
                for element in node.elts:
                    violation = self._find_forbidden_type_no_any(element)
                    if violation:
                        return violation
                return None
            else:
                return self._find_forbidden_type_no_any(node)

        # Check each position, skipping Any at allowed positions
        if isinstance(node, ast.Tuple):
            for i, element in enumerate(node.elts):
                if i in allowed_positions:
                    # This position allows Any - only check for mutable types
                    violation = self._find_forbidden_type_no_any(element)
                else:
                    # Check for all forbidden types including Any
                    violation = self._find_forbidden_type(element)
                if violation:
                    return violation
            return None
        else:
            # Single type arg at position 0
            if 0 in allowed_positions:
                return self._find_forbidden_type_no_any(node)
            else:
                return self._find_forbidden_type(node)

    def _find_forbidden_type_no_any(self, node: ast.expr) -> str | None:
        """Like _find_forbidden_type but only checks mutable types, not Any.

        Uses qualified name resolution consistently with _find_forbidden_type.
        """
        if isinstance(node, ast.Name):
            # Use qualified resolution first
            classification = self._classify_by_qualified_name(node)
            if classification == 'mutable':
                return node.id
            if classification in ('transparent', 'allowed'):
                return None
            # Fall back to short name for builtins
            if node.id in MUTABLE_TYPES:
                return node.id
            return None

        elif isinstance(node, ast.Subscript):
            container_name = self._get_type_name(node.value)

            # Use qualified resolution first
            classification = self._classify_by_qualified_name(node.value)
            if classification == 'mutable':
                return container_name
            if classification == 'transparent':
                return self._find_forbidden_in_slice_no_any(node.slice)
            if classification == 'allowed':
                return None

            # Fall back to short name for builtins
            if container_name in MUTABLE_TYPES:
                return container_name

            # Immutable containers
            if container_name in ALLOWED_CONTAINERS:
                return None

            # Resolve qualified name for remaining checks
            resolved_name = _resolve_qualified_name(node.value, self.import_map)

            # Union, Optional - check all components
            if resolved_name in (
                'typing.Union',
                'typing.Optional',
                'typing_extensions.Union',
                'typing_extensions.Optional',
            ):
                return self._find_forbidden_in_slice_no_any(node.slice)

            # Literal types contain values, not type annotations - skip entirely
            if resolved_name in ('typing.Literal', 'typing_extensions.Literal'):
                return None

            # Annotated types: only check the first argument (the actual type), skip metadata
            if resolved_name in ('typing.Annotated', 'typing_extensions.Annotated'):
                if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                    return self._find_forbidden_type_no_any(node.slice.elts[0])
                return self._find_forbidden_type_no_any(node.slice)

            # Recurse into slice for unknown containers
            return self._find_forbidden_in_slice_no_any(node.slice)

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left_violation = self._find_forbidden_type_no_any(node.left)
            if left_violation:
                return left_violation
            return self._find_forbidden_type_no_any(node.right)

        elif isinstance(node, ast.Attribute):
            # Use qualified resolution
            classification = self._classify_by_qualified_name(node)
            if classification == 'mutable':
                return node.attr
            if classification in ('transparent', 'allowed'):
                return None
            # Fall back to short name for builtins
            if node.attr in MUTABLE_TYPES:
                return node.attr
            return None

        elif isinstance(node, ast.Tuple):
            for element in node.elts:
                violation = self._find_forbidden_type_no_any(element)
                if violation:
                    return violation
            return None

        elif isinstance(node, ast.List):
            # Callable parameter lists: Callable[[int, str], ReturnType]
            for element in node.elts:
                violation = self._find_forbidden_type_no_any(element)
                if violation:
                    return violation
            return None

        elif isinstance(node, ast.Constant):
            # String annotation (forward reference or PEP 563)
            if isinstance(node.value, str):
                return self._check_string_annotation_no_any(node.value)
            return None

        return None

    def _find_forbidden_in_slice_no_any(self, node: ast.expr) -> str | None:
        """Like _find_forbidden_in_slice but only checks mutable types."""
        if isinstance(node, ast.Tuple):
            for element in node.elts:
                violation = self._find_forbidden_type_no_any(element)
                if violation:
                    return violation
            return None
        else:
            return self._find_forbidden_type_no_any(node)

    def _check_string_annotation(self, annotation_str: str) -> str | None:
        """Parse and check a string annotation for forbidden types."""
        expr_node = ast.parse(annotation_str, mode='eval').body
        return self._find_forbidden_type(expr_node)

    def _check_string_annotation_no_any(self, annotation_str: str) -> str | None:
        """Parse and check a string annotation for mutable types only (not Any)."""
        expr_node = ast.parse(annotation_str, mode='eval').body
        return self._find_forbidden_type_no_any(expr_node)

    # Mapping from violation kind to directive code
    _DIRECTIVE_CODES: Mapping[str, str] = {
        'mutable': 'mutable-type',
        'loose': 'loose-typing',
    }

    def _has_directive(self, lineno: int, violation_kind: Literal['mutable', 'loose']) -> bool:
        """Check if line has a directive comment for this check.

        Args:
            lineno: Line number to check
            violation_kind: 'mutable' or 'loose' - determines which code to check

        Returns:
            True if a matching directive comment is found
        """
        if 0 < lineno <= len(self.source_lines):
            line = self.source_lines[lineno - 1]

            # Look for our directive prefix
            prefix_lower = DIRECTIVE_PREFIX.lower()
            if prefix_lower not in line.lower():
                return False

            # Find the directive part
            directive_idx = line.lower().find(prefix_lower)
            codes_part = line[directive_idx + len(DIRECTIVE_PREFIX) :].strip()

            # Strip trailing comment (# explanation)
            if ' #' in codes_part:
                codes_part = codes_part.split(' #')[0].strip()

            # Split by comma and extract just the code (strip any trailing text)
            codes = [c.strip().lower().split()[0] for c in codes_part.split(',') if c.strip()]

            # Look up the code for this violation kind
            expected_code = self._DIRECTIVE_CODES.get(violation_kind)
            if expected_code and expected_code in codes:
                return True

        return False


# =============================================================================
# Module Ordering Checks (opt-in via __strict_module_ordering__)
# =============================================================================


# Recognized __strict_* variables (fail-fast on typos)
RECOGNIZED_STRICT_VARS: Set[str] = {
    '__strict_module_ordering__',
}


def _get_strict_module_ordering(init_path: Path) -> bool:
    """Check if __init__.py opts into strict module ordering.

    Raises:
        ValueError: If unrecognized __strict_* variable found (fail-fast on typos)
        OSError: If file cannot be read
        SyntaxError: If file contains invalid Python syntax
    """
    if not init_path.exists():
        return False

    source = init_path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(init_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Fail-fast on unrecognized __strict_* variables
                    if name.startswith('__strict_') and name.endswith('__'):
                        if name not in RECOGNIZED_STRICT_VARS:
                            raise ValueError(
                                f'{init_path}:{node.lineno}: Unrecognized strict variable '
                                f"'{name}'. Did you mean one of: {', '.join(sorted(RECOGNIZED_STRICT_VARS))}?"
                            )
                        if name == '__strict_module_ordering__':
                            if isinstance(node.value, ast.Constant):
                                return node.value.value is True
    return False


def _extract_all_names(tree: ast.Module) -> Set[str] | None:
    """Extract names from __all__ if defined."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, ast.List):
                        names = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                names.add(elt.value)
                        return names
    return None


@dataclass(frozen=True)
class _Definition:
    """A top-level definition in a module."""

    name: str
    line: int
    is_class: bool
    is_private: bool
    in_all: bool

    @property
    def sort_key(self) -> tuple[int, int, int]:
        """Sort key: (in_all, is_private, is_function)."""
        all_group = 0 if self.in_all else 1
        private_group = 1 if self.is_private else 0
        type_group = 0 if self.is_class else 1
        return (all_group, private_group, type_group)


def _extract_definitions(tree: ast.Module, all_names: Set[str] | None) -> Sequence[_Definition]:
    """Extract top-level class and function definitions."""
    definitions = []
    all_names = all_names or set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            definitions.append(
                _Definition(
                    name=node.name,
                    line=node.lineno,
                    is_class=True,
                    is_private=node.name.startswith('_'),
                    in_all=node.name in all_names,
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append(
                _Definition(
                    name=node.name,
                    line=node.lineno,
                    is_class=False,
                    is_private=node.name.startswith('_'),
                    in_all=node.name in all_names,
                )
            )

    return definitions


def _has_ordering_directive(source_lines: Sequence[str], lineno: int) -> bool:
    """Check if line has ordering suppression directive."""
    if 0 < lineno <= len(source_lines):
        line = source_lines[lineno - 1].lower()
        if DIRECTIVE_PREFIX.lower() in line:
            idx = line.find(DIRECTIVE_PREFIX.lower())
            codes_part = line[idx + len(DIRECTIVE_PREFIX) :]
            codes = [c.strip().split()[0] for c in codes_part.split(',') if c.strip()]
            return 'ordering' in codes
    return False


def check_module_ordering(filepath: Path, tree: ast.Module, source_lines: Sequence[str]) -> Sequence[OrderingViolation]:
    """Check module ordering: __all__ items first, public before private, classes before functions."""
    violations: list[OrderingViolation] = []

    all_names = _extract_all_names(tree)
    definitions = _extract_definitions(tree, all_names)

    if not definitions:
        return violations

    expected_order = sorted(definitions, key=lambda d: d.sort_key)

    for actual, expected in zip(definitions, expected_order):
        if actual.name != expected.name:
            if _has_ordering_directive(source_lines, actual.line):
                continue

            # Describe what's wrong
            if expected.in_all and not actual.in_all:
                reason = f"'{actual.name}' should come after __all__ items"
            elif not expected.is_private and actual.is_private:
                reason = f"private '{actual.name}' should come after public definitions"
            elif expected.is_class and not actual.is_class:
                reason = f"function '{actual.name}' should come after classes"
            else:
                reason = f"'{actual.name}' is out of order (expected '{expected.name}' here)"

            violations.append(
                OrderingViolation(
                    filepath=filepath,
                    line=actual.line,
                    kind='ordering',
                    message=reason,
                    source_line=source_lines[actual.line - 1].strip() if actual.line <= len(source_lines) else '',
                    suggestion='Reorder: __all__ items (classes→functions) → public (classes→functions) → private (classes→functions)',
                )
            )
            break  # Report first violation only

    return violations


def check_all_defined(filepath: Path, tree: ast.Module, source_lines: Sequence[str]) -> Sequence[OrderingViolation]:
    """Check that __all__ is defined."""
    all_names = _extract_all_names(tree)
    if all_names is None:
        line = 1
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
                line = node.lineno
                break

        return [
            OrderingViolation(
                filepath=filepath,
                line=line,
                kind='missing-all',
                message='__all__ is not defined',
                source_line=source_lines[line - 1].strip() if line <= len(source_lines) else '',
                suggestion='Add __all__ = [...] to explicitly declare public API',
            )
        ]
    return []


def _has_trailing_comma(source_lines: Sequence[str], list_node: ast.List) -> bool:
    """Check if list has trailing comma. Uses tokenize to handle comments correctly."""
    if not list_node.elts:
        return True  # Empty list - no check needed

    start_line, end_line = list_node.lineno, list_node.end_lineno
    if start_line is None or end_line is None:
        return True

    source_chunk = '\n'.join(source_lines[start_line - 1 : end_line])

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source_chunk).readline))
    except tokenize.TokenError:
        return True  # Can't tokenize - assume OK

    bracket_depth = 0
    last_at_depth_1: tokenize.TokenInfo | None = None

    for tok in tokens:
        if tok.type == tokenize.OP:
            if tok.string == '[':
                bracket_depth += 1
            elif tok.string == ']':
                bracket_depth -= 1
                if bracket_depth == 0:
                    return last_at_depth_1 is not None and last_at_depth_1.string == ','
            elif tok.string == ',' and bracket_depth == 1:
                last_at_depth_1 = tok
        elif tok.type in (tokenize.STRING, tokenize.NAME) and bracket_depth == 1:
            last_at_depth_1 = tok

    return True  # Unexpected structure - assume OK


def check_all_trailing_comma(
    filepath: Path,
    tree: ast.Module,
    source_lines: Sequence[str],
) -> Sequence[OrderingViolation]:
    """Check that __all__ has trailing comma when it has 1+ items."""
    violations: list[OrderingViolation] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, ast.List) and node.value.elts:
                        if _has_ordering_directive(source_lines, node.lineno):
                            continue
                        if not _has_trailing_comma(source_lines, node.value):
                            violations.append(
                                OrderingViolation(
                                    filepath=filepath,
                                    line=node.lineno,
                                    kind='trailing-comma',
                                    message='__all__ list should have trailing comma',
                                    source_line=source_lines[node.lineno - 1].strip()
                                    if node.lineno <= len(source_lines)
                                    else '',
                                    suggestion='Add trailing comma after last item',
                                )
                            )
    return violations


# =============================================================================
# File Processing
# =============================================================================


def check_file(filepath: Path, strict_ordering_packages: Set[Path]) -> Sequence[Violation]:
    """Check a single file for all violations."""
    source = filepath.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(filepath))
    source_lines = source.splitlines()
    violations: list[Violation] = []

    # Build import map for resolving type names
    import_map = _build_import_map(tree)

    # Type checking
    type_checker = AnnotationChecker(filepath, source_lines, import_map)
    type_checker.visit(tree)
    violations.extend(type_checker.violations)

    # Module ordering (if package opted in)
    package_dir = filepath.parent
    if package_dir in strict_ordering_packages and filepath.name != '__init__.py':
        violations.extend(check_all_defined(filepath, tree, source_lines))
        violations.extend(check_all_trailing_comma(filepath, tree, source_lines))
        violations.extend(check_module_ordering(filepath, tree, source_lines))

    return violations


def _get_git_ignored_files(file_paths: Sequence[Path], directory: Path) -> Set[Path]:
    """Use git check-ignore to identify ignored files.

    Returns set of ignored paths, or empty set if not a git repo or git unavailable.
    """
    if not file_paths:
        return set()

    try:
        result = subprocess.run(
            ['git', 'check-ignore', '--stdin'],
            input='\n'.join(str(p) for p in file_paths),
            capture_output=True,
            text=True,
            cwd=directory,
            timeout=30,
        )
        # Exit codes: 0 = some ignored, 1 = none ignored, 128 = not a git repo
        if result.returncode == 128:
            return set()
        return {Path(line) for line in result.stdout.splitlines() if line}
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return set()


def find_python_files(root: Path, exclude_dirs: Set[str], respect_gitignore: bool = True) -> Sequence[Path]:
    """Find all .py files recursively under root, skipping excluded directories."""
    all_files = sorted(path for path in root.rglob('*.py') if not any(part in exclude_dirs for part in path.parts))

    if respect_gitignore:
        ignored = _get_git_ignored_files(all_files, root)
        all_files = [f for f in all_files if f not in ignored]

    return all_files


def _discover_strict_ordering_packages(files: Sequence[Path]) -> Set[Path]:
    """Find packages that have opted into strict module ordering.

    Scans all __init__.py files in the file list for __strict_module_ordering__ = True.
    Raises ValueError on unrecognized __strict_* variables (fail-fast on typos).
    """
    packages: set[Path] = set()
    init_files = {f for f in files if f.name == '__init__.py'}

    for init_path in init_files:
        # _get_strict_module_ordering raises ValueError on unrecognized __strict_* vars
        if _get_strict_module_ordering(init_path):
            packages.add(init_path.parent)

    return packages


# =============================================================================
# Output Formatting
# =============================================================================


def format_violation(v: Violation) -> str:
    """Format a violation for display."""
    if isinstance(v, TypeViolation):
        type_desc = 'Loose type' if v.kind == 'loose' else 'Mutable type'
        directive_code = 'loose-typing' if v.kind == 'loose' else 'mutable-type'
        return (
            f'{v.filepath}:{v.line}:{v.column}: error: '
            f"{type_desc} '{v.bad_type}' in {v.context} annotation\n"
            f'    {v.source_line}\n'
            f'    Suggestion: {v.suggestion}\n'
            f'    Silence with: {DIRECTIVE_PREFIX} {directive_code}'
        )
    else:  # OrderingViolation
        directive_code = 'ordering'
        return (
            f'{v.filepath}:{v.line}: error: {v.message}\n'
            f'    {v.source_line}\n'
            f'    Suggestion: {v.suggestion}\n'
            f'    Silence with: {DIRECTIVE_PREFIX} {directive_code}'
        )


# =============================================================================
# Main Entry Point
# =============================================================================


# Map from directive codes to violation kinds for --ignore flag
# 'ordering' maps to both 'ordering' and 'missing-all' since they're both ordering-related
_CODE_TO_KINDS: Mapping[DirectiveCode, Set[ViolationKind]] = {
    'mutable-type': {'mutable'},
    'loose-typing': {'loose'},
    'ordering': {'ordering', 'missing-all', 'trailing-comma'},
}


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Check for mutable and loose types in annotations.',
        epilog='Example: %(prog)s . --exclude .venv .git',
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Files or directories to check (use . for current directory)',
    )
    parser.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        metavar='DIR',
        help='Directories to exclude when searching recursively',
    )
    parser.add_argument(
        '--ignore',
        nargs='*',
        default=[],
        metavar='CODE',
        choices=['mutable-type', 'loose-typing', 'ordering'],
        help='Violation codes to ignore (mutable-type, loose-typing, ordering)',
    )
    parser.add_argument(
        '--no-gitignore',
        action='store_true',
        help='Do not respect .gitignore when scanning directories',
    )

    args = parser.parse_args()
    exclude_dirs = set(args.exclude)
    ignored_kinds: set[ViolationKind] = set()
    for code in args.ignore:
        ignored_kinds.update(_CODE_TO_KINDS[code])
    respect_gitignore = not args.no_gitignore

    # Collect files to check
    files: list[Path] = []
    for arg in args.paths:
        path = Path(arg)
        if arg == '.' or path.is_dir():
            # Directory: find all .py files recursively
            files.extend(find_python_files(path, exclude_dirs, respect_gitignore))
        elif path.suffix == '.py' and path.is_file():
            files.append(path)
        # Silently skip non-.py files (allows pre-commit to pass mixed file lists)

    if not files:
        return 0

    # Discover packages with strict module ordering enabled
    # Raises ValueError on unrecognized __strict_* variables (fail-fast on typos)
    strict_ordering_packages = _discover_strict_ordering_packages(files)

    # Check all files
    all_violations: list[Violation] = []
    for filepath in files:
        violations = check_file(filepath, strict_ordering_packages)
        all_violations.extend(violations)

    # Filter out ignored violation kinds
    if ignored_kinds:
        all_violations = [v for v in all_violations if v.kind not in ignored_kinds]

    # Report violations
    if all_violations:
        for v in sorted(all_violations, key=lambda x: (x.filepath, x.line)):
            print(format_violation(v))
            print()

        file_count = len({v.filepath for v in all_violations})
        mutable_count = sum(1 for v in all_violations if v.kind == 'mutable')
        loose_count = sum(1 for v in all_violations if v.kind == 'loose')
        ordering_count = sum(1 for v in all_violations if v.kind in ('ordering', 'missing-all', 'trailing-comma'))
        summary_parts = []
        if mutable_count:
            summary_parts.append(f'{mutable_count} mutable')
        if loose_count:
            summary_parts.append(f'{loose_count} loose')
        if ordering_count:
            summary_parts.append(f'{ordering_count} ordering')
        print(f'Found {len(all_violations)} violation(s) ({", ".join(summary_parts)}) in {file_count} file(s).')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
