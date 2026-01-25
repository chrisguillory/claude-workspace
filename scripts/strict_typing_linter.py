#!/usr/bin/env -S uv run
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
import subprocess
import sys
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# =============================================================================
# Configuration - What to check (not WHERE to check - that's the caller's job)
# =============================================================================

# Types that trigger mutable-type errors
MUTABLE_TYPES: Set[str] = {
    # Lowercase built-ins (Python 3.9+) - concrete mutable types
    'list',
    'dict',
    'set',
    # Uppercase from typing module (legacy, deprecated in 3.9+)
    'List',
    'Dict',
    # Note: typing.Set is NOT here because it's an alias for collections.abc.Set
    # (the abstract interface), not the concrete set type
}

# Types that trigger loose-typing errors
LOOSE_TYPES: Set[str] = {
    'Any',  # Escape hatch that hides structure
}

# Combined forbidden types
FORBIDDEN_TYPES: Set[str] = MUTABLE_TYPES | LOOSE_TYPES

# Types that are always allowed (don't recurse into these looking for violations)
ALLOWED_CONTAINERS: Set[str] = {
    'Mapping',
    'Sequence',
    'Set',  # collections.abc.Set (abstract interface for set-like types)
    'tuple',
    'frozenset',
    'FrozenSet',  # typing module variant
}

# Types that indicate "check nested contents" even though container is allowed
# e.g., Mapping[str, list[int]] - Mapping is allowed, but check the value type
TRANSPARENT_CONTAINERS: Set[str] = {
    'Mapping',
    'Sequence',  # Check what's inside these
}

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
type OrderingViolationKind = Literal['ordering', 'missing-all']
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
    kind: Literal['ordering', 'missing-all']
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
        # Loose type suggestions
        'Any': 'a specific type (TypedDict, StrictModel, or concrete type)',
    }

    def __init__(self, filepath: Path, source_lines: Sequence[str]) -> None:
        self.filepath = filepath
        self.source_lines = source_lines
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
        """
        if isinstance(node, ast.Name):
            # Simple name like 'list' or 'Sequence'
            if node.id in FORBIDDEN_TYPES:
                return node.id
            return None

        elif isinstance(node, ast.Subscript):
            # Generic like 'list[str]' or 'Mapping[str, Any]'
            container_name = self._get_type_name(node.value)

            # If the container itself is forbidden, report immediately
            if container_name in FORBIDDEN_TYPES:
                return container_name

            # If it's a transparent container, check nested contents
            if container_name in TRANSPARENT_CONTAINERS:
                return self._find_forbidden_in_slice(node.slice)

            # If it's an allowed container (tuple, frozenset), stop checking
            if container_name in ALLOWED_CONTAINERS:
                return None

            # For Union, Optional, etc. - check all components
            if container_name in ('Union', 'Optional'):
                return self._find_forbidden_in_slice(node.slice)

            # Unknown container - recurse into slice to be safe
            return self._find_forbidden_in_slice(node.slice)

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type: X | Y
            left_violation = self._find_forbidden_type(node.left)
            if left_violation:
                return left_violation
            return self._find_forbidden_type(node.right)

        elif isinstance(node, ast.Attribute):
            # Qualified name like 'typing.List'
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

    def _check_string_annotation(self, annotation_str: str) -> str | None:
        """Parse and check a string annotation for forbidden types."""
        try:
            # Parse the string as a Python expression
            expr_node = ast.parse(annotation_str, mode='eval').body
            return self._find_forbidden_type(expr_node)
        except SyntaxError:
            # Invalid annotation - let other tools catch this
            return None

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
    """
    if not init_path.exists():
        return False

    try:
        source = init_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(init_path))
    except (OSError, SyntaxError):
        return False

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


# =============================================================================
# File Processing
# =============================================================================


def check_file(filepath: Path, strict_ordering_packages: Set[Path]) -> Sequence[Violation]:
    """Check a single file for all violations."""
    try:
        source = filepath.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as e:
        print(f'Warning: Could not read {filepath}: {e}', file=sys.stderr)
        return []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        print(f'Warning: Syntax error in {filepath}: {e}', file=sys.stderr)
        return []

    source_lines = source.splitlines()
    violations: list[Violation] = []

    # Type checking
    type_checker = AnnotationChecker(filepath, source_lines)
    type_checker.visit(tree)
    violations.extend(type_checker.violations)

    # Module ordering (if package opted in)
    package_dir = filepath.parent
    if package_dir in strict_ordering_packages and filepath.name != '__init__.py':
        violations.extend(check_all_defined(filepath, tree, source_lines))
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
    'ordering': {'ordering', 'missing-all'},
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
        ordering_count = sum(1 for v in all_violations if v.kind in ('ordering', 'missing-all'))
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
