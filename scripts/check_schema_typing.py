#!/usr/bin/env -S uv run
"""Check for mutable and loose type patterns in annotations.

This script enforces immutable interface types:
1. IMMUTABLE TYPES: Use Sequence/Mapping/Set instead of list/dict/set
2. STRICT TYPING: No Any, Mapping[str, Any], Sequence[Any] etc.

Checks ALL function/method parameters and class field annotations.
Skips non-frozen dataclasses (mutable by design).
Respects .gitignore when scanning directories.

Design Philosophy:
    - Error-only, no auto-fix: Forces conscious decision at each occurrence
    - Pure analysis: Checks exactly the files it's given (no internal filtering)
    - Escape hatches:
      - # check_schema_typing.py: mutable-type - suppress mutable type violations
      - # check_schema_typing.py: loose-typing - suppress loose typing violations

Usage:
    ./scripts/check_schema_typing.py <files...>
    ./scripts/check_schema_typing.py .          # All .py files in cwd

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


# =============================================================================
# Data Types
# =============================================================================


# Type aliases for constrained string values
type ViolationKind = Literal['mutable', 'loose']
type FieldContext = Literal['field', 'parameter']
type DirectiveCode = Literal['mutable-type', 'loose-typing']


@dataclass(frozen=True)
class Violation:
    """A single type violation (mutable or loose)."""

    filepath: Path
    line: int
    column: int
    context: FieldContext
    bad_type: str
    violation_kind: ViolationKind
    source_line: str
    suggestion: str


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
        # @dataclass(...) with args
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


class SchemaTypeChecker(ast.NodeVisitor):
    """AST visitor that checks for mutable and loose types in annotations."""

    SUGGESTIONS: dict[str, str] = {
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

        # Track that we're in a function body
        self._function_depth += 1

        # Visit function body
        self.generic_visit(node)

        self._function_depth -= 1

    def _check_function_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check parameter annotations on a function/method."""
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs

        for arg in all_args:
            # Skip 'self' and 'cls' parameters
            if arg.arg in ('self', 'cls'):
                continue
            if arg.annotation:
                self._check_annotation(arg.annotation, arg.lineno, 'parameter')

    def _check_annotation(self, node: ast.expr, lineno: int, context: FieldContext) -> None:
        """Check if an annotation contains forbidden types."""
        bad_type = self._find_forbidden_type(node)
        if bad_type:
            # Determine violation kind
            kind: ViolationKind = 'loose' if bad_type in LOOSE_TYPES else 'mutable'

            # Check for directive comment with appropriate code
            if self._has_directive(lineno, kind):
                return

            # Get the source line for context
            source_line = ''
            if 0 < lineno <= len(self.source_lines):
                source_line = self.source_lines[lineno - 1].strip()

            self.violations.append(
                Violation(
                    filepath=self.filepath,
                    line=lineno,
                    column=getattr(node, 'col_offset', 0),
                    context=context,
                    bad_type=bad_type,
                    violation_kind=kind,
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
    _DIRECTIVE_CODES: Mapping[ViolationKind, DirectiveCode] = {
        'mutable': 'mutable-type',
        'loose': 'loose-typing',
    }

    # The directive prefix - uses the script filename for maximum clarity
    _DIRECTIVE_PREFIX = '# check_schema_typing.py:'

    def _has_directive(self, lineno: int, violation_kind: ViolationKind) -> bool:
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
            prefix_lower = self._DIRECTIVE_PREFIX.lower()
            if prefix_lower not in line.lower():
                return False

            # Find the directive part
            directive_idx = line.lower().find(prefix_lower)
            codes_part = line[directive_idx + len(self._DIRECTIVE_PREFIX) :].strip()

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
# File Processing
# =============================================================================


def check_file(filepath: Path) -> list[Violation]:
    """Check a single file for mutable type violations."""
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
    checker = SchemaTypeChecker(filepath, source_lines)
    checker.visit(tree)

    return checker.violations


def _get_git_ignored_files(file_paths: Sequence[Path], directory: Path) -> set[Path]:
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


def find_python_files(root: Path, exclude_dirs: Set[str], respect_gitignore: bool = True) -> list[Path]:
    """Find all .py files recursively under root, skipping excluded directories."""
    all_files = sorted(
        path for path in root.rglob('*.py') if not any(part in exclude_dirs for part in path.parts)
    )

    if respect_gitignore:
        ignored = _get_git_ignored_files(all_files, root)
        all_files = [f for f in all_files if f not in ignored]

    return all_files


# =============================================================================
# Output Formatting
# =============================================================================


def format_violation(v: Violation) -> str:
    """Format a violation for display."""
    if v.violation_kind == 'loose':
        type_desc = 'Loose type'
        directive_code = 'loose-typing'
    else:
        type_desc = 'Mutable type'
        directive_code = 'mutable-type'

    return (
        f'{v.filepath}:{v.line}:{v.column}: error: '
        f"{type_desc} '{v.bad_type}' in {v.context} annotation\n"
        f'    {v.source_line}\n'
        f'    Suggestion: Use {v.suggestion}\n'
        f'    Silence with: # check_schema_typing.py: {directive_code}'
    )


# =============================================================================
# Main Entry Point
# =============================================================================


# Map from directive codes to violation kinds for --ignore flag
_CODE_TO_KIND: dict[str, ViolationKind] = {
    'mutable-type': 'mutable',
    'loose-typing': 'loose',
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
        choices=['mutable-type', 'loose-typing'],
        help='Violation codes to ignore (mutable-type, loose-typing)',
    )
    parser.add_argument(
        '--no-gitignore',
        action='store_true',
        help='Do not respect .gitignore when scanning directories',
    )

    args = parser.parse_args()
    exclude_dirs = set(args.exclude)
    ignored_kinds: set[ViolationKind] = {_CODE_TO_KIND[code] for code in args.ignore}
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

    # Check all files
    all_violations: list[Violation] = []
    for filepath in files:
        violations = check_file(filepath)
        all_violations.extend(violations)

    # Filter out ignored violation kinds
    if ignored_kinds:
        all_violations = [v for v in all_violations if v.violation_kind not in ignored_kinds]

    # Report violations
    if all_violations:
        for v in sorted(all_violations, key=lambda x: (x.filepath, x.line)):
            print(format_violation(v))
            print()

        file_count = len({v.filepath for v in all_violations})
        mutable_count = sum(1 for v in all_violations if v.violation_kind == 'mutable')
        loose_count = sum(1 for v in all_violations if v.violation_kind == 'loose')
        summary_parts = []
        if mutable_count:
            summary_parts.append(f'{mutable_count} mutable')
        if loose_count:
            summary_parts.append(f'{loose_count} loose')
        print(f'Found {len(all_violations)} violation(s) ({", ".join(summary_parts)}) in {file_count} file(s).')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())