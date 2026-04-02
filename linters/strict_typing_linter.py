#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Strict typing linter for Python annotations.

This script enforces immutable interface types and module organization:
1. IMMUTABLE TYPES: Use Sequence/Mapping/Set instead of list/dict/set
2. STRICT TYPING: No Any, Mapping[str, Any], Sequence[Any] etc.
3. TUPLE FIELDS: Flag tuple[X, ...] in class fields (use Sequence[T] instead)
4. HASHABLE FIELDS (opt-in): Enforce hashable types via __strict_typing_linter__hashable_fields__
5. MODULE ORDERING: Public before private, classes before functions

Checks ALL function/method parameters, return types, and class field annotations.
Skips non-frozen dataclasses and attrs classes (mutable by design).
Note: pydantic_settings.BaseSettings (intentionally non-frozen) is not supported at this time.
Respects .gitignore when scanning directories.

Module ordering is always enabled. Suppress per-area via per-file-ignores in pyproject.toml:
    [tool.strict-typing-linter.per-file-ignores]
    "scripts/**" = ["missing-all", "class-ordering"]
Files with ordering enabled must:
- Define __all__
- Order definitions: __all__ items first, then public, then private
- Within each group: classes before functions

Design Philosophy:
    - Error-only, no auto-fix: Forces conscious decision at each occurrence
    - Pure analysis: Checks exactly the files it's given (no internal filtering)
    - Escape hatches:
      - # strict_typing_linter.py: skip-file - skip entire file
      - # strict_typing_linter.py: mutable-type - suppress mutable type violations
      - # strict_typing_linter.py: loose-typing - suppress loose typing violations
      - # strict_typing_linter.py: tuple-field - suppress tuple-in-field violations
      - # strict_typing_linter.py: hashable-field - suppress unhashable-field violations
      - # strict_typing_linter.py: ordering - suppress all ordering violations
      - # strict_typing_linter.py: class-ordering - suppress class-before-function violations
      - # strict_typing_linter.py: missing-all - suppress missing __all__ violations
      - # strict_typing_linter.py: trailing-comma - suppress trailing comma violations

Usage:
    ./linters/strict_typing_linter.py <files...>
    ./linters/strict_typing_linter.py .          # All .py files in cwd

File selection is handled by the caller (pre-commit, shell globs, etc.).
This script is a pure analysis function - it checks what it's given.

Exit codes:
    0 - No violations found
    1 - Violations found
"""

from __future__ import annotations

import argparse
import ast
import builtins
import enum
import io
import sys
import tokenize
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from _lib.config import (
    find_config,
    find_python_files,
    get_per_file_ignored_codes,
    load_per_file_ignores,
)
from _lib.hashability_inspector import HashabilityInspector, QualifiedName

# -- Configuration ------------------------------------------------------------

# Builtin names — skip these in the inspector (int, str, bool, etc. are not user-defined types)
BUILTIN_NAMES: Set[str] = set(dir(builtins))

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
QUALIFIED_MUTABLE: Set[QualifiedName] = {
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
QUALIFIED_ALLOWED: Set[QualifiedName] = {
    'collections.abc.Set',
    'collections.abc.Mapping',
    'collections.abc.Sequence',
    'typing.AbstractSet',
    'typing.Mapping',
    'typing.Sequence',
}

# Fully qualified transparent types (check nested contents)
QUALIFIED_TRANSPARENT: Set[QualifiedName] = {
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
ANY_ALLOWED_POSITIONS: Mapping[QualifiedName, Sequence[int] | None] = {
    # FastMCP Context[ServerSessionT, LifespanContextT, RequestT] - all positions
    'mcp.server.fastmcp.Context': None,
    # Generator[YieldType, SendType, ReturnType] - SendType and ReturnType often unused
    'typing.Generator': (1, 2),
    'collections.abc.Generator': (1, 2),
    # AsyncGenerator[YieldType, SendType] - SendType often unused
    'typing.AsyncGenerator': (1,),
    'collections.abc.AsyncGenerator': (1,),
    # Coroutine[T_co, T_contra, V_co] - all positions commonly Any in fire-and-forget patterns
    'typing.Coroutine': None,
    'collections.abc.Coroutine': None,
}

# Directive prefix uses the script filename for discoverability - readers immediately
# know which tool owns the directive, and searching for the filename finds both the
# script and all its annotations in the codebase.
DIRECTIVE_PREFIX = '# strict_typing_linter.py:'

# -- Data Types ---------------------------------------------------------------

# Type aliases for constrained string values
# Name resolution
type LocalName = str  # Local identifier as it appears in source (e.g., 'Generator', 't')
type TypeClassification = Literal['mutable', 'transparent', 'allowed', 'unknown']

type FieldContext = Literal['field', 'parameter', 'return']
type DirectiveCode = Literal[
    'mutable-type',
    'loose-typing',
    'tuple-field',
    'hashable-field',
    'ordering',
    'class-ordering',
    'missing-all',
    'trailing-comma',
    'unused-directive',
]

# Violation kind literals - used in discriminated unions
type TypeViolationKind = Literal['mutable', 'loose', 'tuple-field', 'hashable-field']
type OrderingViolationKind = Literal['ordering', 'class-ordering', 'missing-all', 'trailing-comma', 'unused-directive']
type ViolationKind = TypeViolationKind | OrderingViolationKind


class SubscriptKind(enum.Enum):
    """Classification of subscript type forms in annotations.

    Literal: contains values, not types — skip entirely
    Annotated: first arg is type, rest is metadata — recurse first arg only
    Generic: regular parameterized type (including Union) — recurse all args
    """

    LITERAL = 'literal'
    ANNOTATED = 'annotated'
    GENERIC = 'generic'


@dataclass(frozen=True)
class TypeViolation:
    """A type annotation violation (mutable or loose types)."""

    filepath: Path
    line: int
    column: int
    context: FieldContext
    bad_type: str
    kind: TypeViolationKind
    source_line: str
    suggestion: str


@dataclass(frozen=True)
class OrderingViolation:
    """A module ordering violation."""

    filepath: Path
    line: int
    kind: OrderingViolationKind
    message: str
    source_line: str
    suggestion: str


# Discriminated union of all violation types
type Violation = TypeViolation | OrderingViolation


@dataclass(frozen=True)
class DirectiveInstance:
    """A suppression directive found in source code."""

    line: int
    codes: Sequence[str]
    raw_text: str


# -- Main Entry Point ---------------------------------------------------------


# Map from directive codes to violation kinds for --ignore flag
# 'ordering' is a superset that suppresses all ordering-related kinds
CODE_TO_KINDS: Mapping[DirectiveCode, Set[ViolationKind]] = {
    'mutable-type': {'mutable'},
    'loose-typing': {'loose'},
    'tuple-field': {'tuple-field'},
    'hashable-field': {'hashable-field'},
    'ordering': {'ordering', 'class-ordering', 'missing-all', 'trailing-comma'},
    'class-ordering': {'class-ordering'},
    'missing-all': {'missing-all'},
    'trailing-comma': {'trailing-comma'},
}


def main() -> int:
    """CLI entry point: parse args, collect files, check, report."""
    args = parse_args()
    exclude_dirs = set(args.exclude)
    ignored_kinds: set[ViolationKind] = set()
    for code in args.ignore:
        ignored_kinds.update(CODE_TO_KINDS[code])
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

    # Compute source roots from expanded files so the inspector can import project types.
    # Uses files (not args.paths) to handle nested projects when scanning from repo root.
    source_roots = tuple(sorted({find_source_root(f) for f in files}))

    # Shared inspector for runtime hashability checking across all files
    inspector = HashabilityInspector(source_roots=source_roots)

    # Resolve config path once if --config was given
    explicit_config = Path(args.config) if args.config else None

    # Check all files
    all_violations: list[Violation] = []
    for filepath in files:
        # Per-file-ignores from pyproject.toml
        per_file_ignored_kinds: set[ViolationKind] = set()
        skip_file_via_config = False
        if not args.no_config:
            if explicit_config is not None:
                config_path = explicit_config
                project_root = explicit_config.parent
            else:
                result = find_config(filepath, 'strict-typing-linter')
                if result is not None:
                    config_path, project_root = result
                else:
                    config_path = None
                    project_root = None

            if config_path is not None and project_root is not None:
                per_file_ignores = load_per_file_ignores('strict-typing-linter', config_path)
                codes = get_per_file_ignored_codes(filepath, per_file_ignores, project_root)
                if 'skip-file' in codes:
                    skip_file_via_config = True
                for code in codes:
                    for directive_code, kind_set in CODE_TO_KINDS.items():
                        if code == directive_code:
                            per_file_ignored_kinds.update(kind_set)

        if skip_file_via_config:
            continue

        violations = check_file(
            filepath,
            inspector=inspector,
            respect_skip_file=not args.no_skip_file,
            report_unused_directives=args.report_unused_directives,
        )

        # Filter by per-file ignored kinds
        if per_file_ignored_kinds:
            violations = [v for v in violations if v.kind not in per_file_ignored_kinds]

        all_violations.extend(violations)

    # Filter out globally ignored violation kinds (from --ignore)
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
        tuple_count = sum(1 for v in all_violations if v.kind == 'tuple-field')
        hashable_count = sum(1 for v in all_violations if v.kind == 'hashable-field')
        ordering_count = sum(
            1 for v in all_violations if v.kind in ('ordering', 'class-ordering', 'missing-all', 'trailing-comma')
        )
        summary_parts = []
        if mutable_count:
            summary_parts.append(f'{mutable_count} mutable')
        if loose_count:
            summary_parts.append(f'{loose_count} loose')
        if tuple_count:
            summary_parts.append(f'{tuple_count} tuple-field')
        if hashable_count:
            summary_parts.append(f'{hashable_count} hashable-field')
        if ordering_count:
            summary_parts.append(f'{ordering_count} ordering')
        print(f'Found {len(all_violations)} violation(s) ({", ".join(summary_parts)}) in {file_count} file(s).')
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
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
        choices=[
            'mutable-type',
            'loose-typing',
            'tuple-field',
            'hashable-field',
            'ordering',
            'class-ordering',
            'missing-all',
            'trailing-comma',
        ],
        help='Violation codes to ignore (mutable-type, loose-typing, tuple-field, hashable-field, ordering, missing-all, trailing-comma)',
    )
    parser.add_argument(
        '--no-gitignore',
        action='store_true',
        help='Do not respect .gitignore when scanning directories',
    )
    parser.add_argument(
        '--no-skip-file',
        action='store_true',
        help='Ignore skip-file directives (used by validation harnesses)',
    )
    parser.add_argument(
        '--report-unused-directives',
        action='store_true',
        help='Report suppression directives that do not match any violation',
    )
    parser.add_argument(
        '--config',
        default=None,
        metavar='PATH',
        help='Path to pyproject.toml (default: auto-discover by walking up from each file)',
    )
    parser.add_argument(
        '--no-config',
        action='store_true',
        help='Disable reading per-file-ignores from pyproject.toml',
    )
    return parser.parse_args()


# -- File Processing ----------------------------------------------------------


def check_file(
    filepath: Path,
    *,
    inspector: HashabilityInspector | None = None,
    respect_skip_file: bool = True,
    report_unused_directives: bool = False,
) -> Sequence[Violation]:
    """Check a single file for all violations."""
    source = filepath.read_text(encoding='utf-8')
    source_lines = source.splitlines()

    # File-level skip directive (check first 10 lines)
    has_skip_file = False
    prefix_lower = DIRECTIVE_PREFIX.lower()
    for line in source_lines[:10]:
        if prefix_lower in line.lower() and 'skip-file' in line.lower():
            has_skip_file = True
            break

    if has_skip_file and respect_skip_file and not report_unused_directives:
        return []

    tree = ast.parse(source, filename=str(filepath))
    violations: list[Violation] = []

    # Build import map for resolving type names
    import_map = build_import_map(tree)

    # Type checking
    type_checker = AnnotationChecker(filepath, source_lines, import_map, inspector=inspector)
    type_checker.visit(tree)

    # Collect all raw violations (type + ordering) for unused directive detection
    all_raw: list[tuple[int, str]] = list(type_checker.raw_violations)

    # If skip-file is present and valid, suppress normal violations
    if has_skip_file and respect_skip_file:
        if report_unused_directives and not all_raw:
            # skip-file is stale — no violations to suppress
            skip_line = next(
                i + 1
                for i, line in enumerate(source_lines[:10])
                if prefix_lower in line.lower() and 'skip-file' in line.lower()
            )
            return [
                OrderingViolation(
                    filepath=filepath,
                    line=skip_line,
                    kind='unused-directive',
                    message='skip-file directive does not suppress any violations',
                    source_line=source_lines[skip_line - 1].strip(),
                    suggestion='Remove the stale skip-file directive',
                ),
            ]
        return []

    violations.extend(type_checker.violations)

    # Module ordering (always enabled, suppress per-area via per-file-ignores)
    # Skip entry points (__main__.py, main.py, shebang scripts)
    raw_ordering: list[tuple[int, str]] = []
    if not _is_entry_point(filepath, source_lines):
        # Skip __init__.py only if it has no definitions (just boilerplate)
        if filepath.name != '__init__.py' or extract_definitions(tree, extract_all_names(tree)):
            violations.extend(check_all_defined(filepath, tree, source_lines, raw_ordering))
            violations.extend(check_all_trailing_comma(filepath, tree, source_lines, raw_ordering))
            violations.extend(check_module_ordering(filepath, tree, source_lines, raw_ordering))
            violations.extend(check_class_method_ordering(filepath, tree, source_lines, raw_ordering))

    # Merge ordering raw violations into all_raw for unused-directive detection
    all_raw.extend(raw_ordering)

    if report_unused_directives:
        directives = collect_directives(source_lines)
        unused = find_unused_directives(
            directives,
            all_raw,
            filepath,
            source_lines,
            type_checker.matched_directive_lines,
        )
        violations.extend(unused)

    return violations


# -- AST Visitor --------------------------------------------------------------


class AnnotationChecker(ast.NodeVisitor):
    """AST visitor that checks for mutable and loose types in annotations."""

    SUGGESTIONS: Mapping[str, str] = {
        # Mutable type suggestions
        'list': 'Sequence[T]',
        'List': 'Sequence[T]',
        'dict': 'Mapping[K, V]',
        'Dict': 'Mapping[K, V]',
        'set': 'Set[T] (from collections.abc)',
        'Set': 'Set[T] (from collections.abc)',
        # Loose type suggestions
        'Any': 'a specific type (TypedDict, StrictModel, or concrete type)',
    }

    def __init__(
        self,
        filepath: Path,
        source_lines: Sequence[str],
        import_map: Mapping[LocalName, QualifiedName],
        inspector: HashabilityInspector | None = None,
    ) -> None:
        self.filepath = filepath
        self.source_lines = source_lines
        self.import_map = import_map
        self.violations: list[Violation] = []  # Internal - OK to be mutable
        self.raw_violations: list[tuple[int, DirectiveCode]] = []
        self._function_depth = 0  # Track nesting depth in functions
        self._skip_class_fields = False  # Skip field checking for non-frozen dataclasses
        self._hashable_fields = False  # In a __strict_typing_linter__hashable_fields__ class
        self._inspector = inspector
        self._class_name_stack: list[str] = []  # Track enclosing class names for private-class detection
        self._skip_any = False  # When True, _check_annotation uses _find_forbidden_type_no_any
        self._classvar_depth = 0  # When > 0, mutable-type violations suppressed (ClassVar is class state, not a field)
        self.matched_directive_lines: set[tuple[int, str]] = (
            set()
        )  # (line, code) pairs of directives that suppressed violations

    # -- Visitor Methods ------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definitions, skipping non-frozen dataclasses."""
        frozen = is_frozen_dataclass(node)
        if frozen is None:
            frozen = is_frozen_attrs(node, self.import_map)
        hashable = get_strict_hashable_fields(node)

        # Save previous state
        prev_skip = self._skip_class_fields
        prev_hashable = self._hashable_fields

        # Skip field checking for non-frozen dataclasses/attrs (mutable by design)
        # Must unconditionally assign (not just set True) so nested classes reset properly
        self._skip_class_fields = frozen is False

        if frozen is False and hashable:
            raise ValueError(
                f"Class '{node.name}' at line {node.lineno}: "
                f'__strict_typing_linter__hashable_fields__ = True is incompatible with non-frozen class. '
                f'Non-frozen classes are mutable and cannot be hashable.',
            )

        # Set hashable fields mode - each class starts fresh (no inheritance from parent)
        self._hashable_fields = hashable

        # Track class name for private-class detection
        self._class_name_stack.append(node.name)

        # Visit children
        self.generic_visit(node)

        # Restore state
        self._class_name_stack.pop()
        self._skip_class_fields = prev_skip
        self._hashable_fields = prev_hashable

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check annotated assignments (x: Type = value)."""
        # Check all annotated assignments at module/class level (not in functions)
        # Skip if we're in a non-frozen dataclass
        if self._function_depth == 0 and not self._skip_class_fields:
            is_cv = self._is_classvar(node.annotation)
            if is_cv:
                self._classvar_depth += 1
            self._check_annotation(node.annotation, node.lineno, 'field')
            if is_cv:
                self._classvar_depth -= 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function parameter annotations."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check async function parameter annotations."""
        self._visit_function(node)

    # -- Private Helper Methods -----------------------------------------------

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Shared logic for visiting function definitions."""
        # Skip loose-typing (Any) checks on internal methods where Any is idiomatic:
        # - Dunder methods: implement Python protocols (__getattr__, __next__, __enter__, etc.)
        # - Private methods: internal implementation with small blast radius
        # - Methods on private classes: public protocol methods on _-prefixed proxy classes
        # Mutable type checks (list, dict, set) still apply in all cases.
        is_dunder = node.name.startswith('__') and node.name.endswith('__')
        is_private = node.name.startswith('_')
        in_private_class = bool(self._class_name_stack and self._class_name_stack[-1].startswith('_'))
        prev_skip_any = self._skip_any
        self._skip_any = is_dunder or is_private or in_private_class

        # Check ALL function parameters
        self._check_function_parameters(node)

        # Check return type annotation — but skip -> Any on generic wrapper
        # functions identified by *args: Any, **kwargs: Any (the full decorator
        # wrapper pattern: accept anything, return anything).
        if node.returns:
            if self._is_generic_wrapper(node) and self._get_type_name(node.returns) == 'Any':
                pass  # Skip -> Any on decorator wrappers
            else:
                # Scope covers entire signature: from `def` line to last line before body.
                # Uses body start - 1 (not annotation end) because the ): closing line
                # may be after the annotation and is a natural place for directives.
                scope_end = node.body[0].lineno - 1 if node.body else node.returns.lineno
                self._check_annotation(
                    node.returns,
                    node.returns.lineno,
                    'return',
                    scope_start=node.lineno,
                    scope_end=scope_end,
                )

        self._skip_any = prev_skip_any

        # Track that we're in a function body
        self._function_depth += 1

        # Visit function body
        self.generic_visit(node)

        self._function_depth -= 1

    def _get_type_name(self, node: ast.expr) -> str:
        """Extract the name from a type expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            return self._get_type_name(node.value)
        return ''

    def _classify_by_qualified_name(self, node: ast.expr) -> TypeClassification:
        """Classify type using fully qualified name resolution.

        Used to disambiguate types like 'Set' which could be:
        - typing.Set (mutable, deprecated)
        - collections.abc.Set (abstract interface, allowed)

        Returns 'unknown' when qualified name doesn't provide disambiguation,
        signaling the caller to fall back to short name matching.
        """
        qualified = resolve_qualified_name(node, self.import_map)

        if qualified in QUALIFIED_MUTABLE:
            return 'mutable'
        if qualified in QUALIFIED_TRANSPARENT:
            return 'transparent'
        if qualified in QUALIFIED_ALLOWED:
            return 'allowed'
        return 'unknown'

    def _is_generic_wrapper(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function is a generic wrapper (*args: Any, **kwargs: Any)."""
        vararg = node.args.vararg
        kwarg = node.args.kwarg
        vararg_is_any = vararg and vararg.annotation and self._get_type_name(vararg.annotation) == 'Any'
        kwarg_is_any = kwarg and kwarg.annotation and self._get_type_name(kwarg.annotation) == 'Any'
        return bool(vararg_is_any and kwarg_is_any)

    def _check_function_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check parameter annotations on a function/method."""
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs

        for arg in all_args:
            if arg.annotation:
                self._check_annotation(arg.annotation, arg.lineno, 'parameter')

        # Skip *args: Any, **kwargs: Any on generic wrappers (decorator pattern).
        vararg = node.args.vararg
        kwarg = node.args.kwarg
        both_any = self._is_generic_wrapper(node)

        if vararg and vararg.annotation and not both_any:
            self._check_annotation(vararg.annotation, vararg.lineno, 'parameter')
        if kwarg and kwarg.annotation and not both_any:
            self._check_annotation(kwarg.annotation, kwarg.lineno, 'parameter')

    def _check_annotation(
        self,
        node: ast.expr,
        lineno: int,
        context: FieldContext,
        *,
        scope_start: int | None = None,
        scope_end: int | None = None,
    ) -> None:
        """Check if an annotation contains forbidden types.

        Three-pass checking for field annotations:
        1. tuple-field: Flag tuple[X, ...] in fields (outside hashable context)
        2. hashable-field: Flag Sequence/Mapping in fields (inside hashable context)
        3. mutable/loose: Existing checks (all contexts)
        """
        source_line = ''
        if 0 < lineno <= len(self.source_lines):
            source_line = self.source_lines[lineno - 1].strip()

        # Pass 1: tuple-field check (fields only, outside hashable context)
        # Check ClassVar too — prefer abstract interfaces (Sequence) everywhere for consistency
        if context == 'field' and not self._hashable_fields:
            if self._find_tuple_field(node):
                self.raw_violations.append((lineno, 'tuple-field'))
                if not self._has_directive(lineno, 'tuple-field', scope_start=scope_start, scope_end=scope_end):
                    self.violations.append(
                        TypeViolation(
                            filepath=self.filepath,
                            line=lineno,
                            column=getattr(node, 'col_offset', 0),
                            context=context,
                            bad_type='tuple',
                            kind='tuple-field',
                            source_line=source_line,
                            suggestion='Sequence[T]',
                        ),
                    )

        # Pass 2: hashable-field check (fields only, inside hashable context)
        # Skip ClassVar fields — not instance fields, don't affect __hash__
        if context == 'field' and self._hashable_fields and not self._is_classvar(node):
            unhashable = self._find_unhashable_field(node)
            if unhashable:
                self.raw_violations.append((lineno, 'hashable-field'))
                if not self._has_directive(lineno, 'hashable-field', scope_start=scope_start, scope_end=scope_end):
                    self.violations.append(
                        TypeViolation(
                            filepath=self.filepath,
                            line=lineno,
                            column=getattr(node, 'col_offset', 0),
                            context=context,
                            bad_type=unhashable,
                            kind='hashable-field',
                            source_line=source_line,
                            suggestion=self._hashable_suggestion(unhashable),
                        ),
                    )

        # Pass 3: existing mutable/loose check (all contexts)
        # When _skip_any is set (dunders, private methods, methods on private classes),
        # only check for mutable types — Any is allowed in internal/protocol code.
        bad_type = self._find_forbidden_type_no_any(node) if self._skip_any else self._find_forbidden_type(node)
        if bad_type:
            kind: TypeViolationKind = 'loose' if bad_type in LOOSE_TYPES else 'mutable'
            # ClassVar is class-level state, not a Pydantic field — mutable types are appropriate
            if kind == 'mutable' and self._classvar_depth > 0:
                return
            self.raw_violations.append((lineno, self._DIRECTIVE_CODES[kind]))

            if self._has_directive(lineno, kind, scope_start=scope_start, scope_end=scope_end):
                return

            suggestion = self.SUGGESTIONS.get(bad_type, 'a more specific type')

            # Context-dependent overrides in hashable classes:
            # list→tuple (Sequence would also be flagged), dict→frozendict (Mapping would also be flagged)
            if self._hashable_fields and context == 'field':
                if bad_type in ('list', 'List'):
                    suggestion = 'tuple[T, ...]'
                elif bad_type in ('dict', 'Dict'):
                    suggestion = 'frozendict[K, V] (third-party) or restructure'

            self.violations.append(
                TypeViolation(
                    filepath=self.filepath,
                    line=lineno,
                    column=getattr(node, 'col_offset', 0),
                    context=context,
                    bad_type=bad_type,
                    kind=kind,
                    source_line=source_line,
                    suggestion=suggestion,
                ),
            )

    def _is_classvar(self, node: ast.expr) -> bool:
        """Check if annotation is a ClassVar (not an instance field)."""
        if isinstance(node, ast.Subscript):
            resolved = resolve_qualified_name(node.value, self.import_map)
            return resolved in ('typing.ClassVar', 'typing_extensions.ClassVar')
        if isinstance(node, ast.Name):
            resolved = resolve_qualified_name(node, self.import_map)
            return resolved in ('typing.ClassVar', 'typing_extensions.ClassVar')
        return False

    def _hashable_suggestion(self, unhashable: str) -> str:
        """Get suggestion for replacing unhashable type in hashable class."""
        if unhashable == 'Sequence':
            return 'tuple[T, ...]'
        elif unhashable == 'Mapping':
            return 'frozendict[K, V] (third-party) or restructure'
        else:
            return f'ensure {unhashable} fields use hashable types (tuple instead of Sequence/list)'

    def _classify_subscript(self, node: ast.Subscript) -> tuple[SubscriptKind, str]:
        """Classify a subscript node by its type form.

        Returns (kind, resolved_name). The resolved name is used by
        _find_forbidden_type for the ANY_ALLOWED_POSITIONS check.
        """
        resolved = resolve_qualified_name(node.value, self.import_map)
        if resolved in ('typing.Literal', 'typing_extensions.Literal'):
            return SubscriptKind.LITERAL, resolved
        if resolved in ('typing.Annotated', 'typing_extensions.Annotated'):
            return SubscriptKind.ANNOTATED, resolved
        return SubscriptKind.GENERIC, resolved

    @staticmethod
    def _get_annotated_type_arg(node: ast.Subscript) -> ast.expr:
        """Extract the type argument from Annotated[Type, ...metadata]."""
        if isinstance(node.slice, ast.Tuple) and node.slice.elts:
            return node.slice.elts[0]
        return node.slice

    def _find_tuple_field(self, node: ast.expr) -> bool:
        """Recursively check if annotation contains tuple[X, ...] (variable-length).

        Walks the entire type tree to catch nested tuples like Mapping[str, tuple[X, ...]].
        """
        if is_variable_length_tuple(node, self.import_map):
            return True

        if isinstance(node, ast.Subscript):
            kind, _ = self._classify_subscript(node)
            if kind is SubscriptKind.LITERAL:
                return False
            if kind is SubscriptKind.ANNOTATED:
                return self._find_tuple_field(self._get_annotated_type_arg(node))
            return self._find_tuple_field_in_slice(node.slice)

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._find_tuple_field(node.left) or self._find_tuple_field(node.right)

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # String annotation (forward reference or PEP 563)
            expr_node = ast.parse(node.value, mode='eval').body
            return self._find_tuple_field(expr_node)

        return False

    def _find_tuple_field_in_slice(self, node: ast.expr) -> bool:
        """Check slice for tuple[X, ...] patterns."""
        if isinstance(node, ast.Tuple):
            return any(self._find_tuple_field(elt) for elt in node.elts)
        return self._find_tuple_field(node)

    def _find_unhashable_field(self, node: ast.expr) -> str | None:
        """Recursively check if annotation contains unhashable types.

        Returns the type name if unhashable, None otherwise.
        Walks the entire type tree to catch nested unhashables like tuple[Sequence[int], ...].

        Checks two layers:
        1. AST-level: Sequence/Mapping detected by name via QUALIFIED_TRANSPARENT
        2. Runtime-level: User-defined types checked via HashabilityInspector import
        """
        # Check if this node is Sequence or Mapping
        classification = self._classify_by_qualified_name(node)
        if classification == 'transparent':
            # 'transparent' means Sequence or Mapping — these are unhashable
            return self._get_type_name(node)

        # Runtime check for user-defined types (VectorPoint, Config, etc.)
        if classification == 'unknown':
            result = self._check_type_via_inspector(node)
            if result is not None:
                return result

        if isinstance(node, ast.Subscript):
            kind, _ = self._classify_subscript(node)
            if kind is SubscriptKind.LITERAL:
                return None
            if kind is SubscriptKind.ANNOTATED:
                return self._find_unhashable_field(self._get_annotated_type_arg(node))

            # Container-level check (Sequence/Mapping are unhashable)
            container_class = self._classify_by_qualified_name(node.value)
            if container_class == 'transparent':
                return self._get_type_name(node.value)
            return self._find_unhashable_in_slice(node.slice)

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._find_unhashable_field(node.left)
            if left:
                return left
            return self._find_unhashable_field(node.right)

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # String annotation (forward reference or PEP 563)
            expr_node = ast.parse(node.value, mode='eval').body
            return self._find_unhashable_field(expr_node)

        return None

    def _check_type_via_inspector(self, node: ast.expr) -> str | None:
        """Check user-defined type hashability via runtime inspector.

        Returns type name if unhashable, None if hashable or unverifiable.
        """
        if self._inspector is None:
            return None

        qualified = resolve_qualified_name(node, self.import_map)
        if not qualified:
            return None

        # Same-file types have no module prefix — compute importable module path
        if '.' not in qualified:
            if qualified in BUILTIN_NAMES:
                return None
            module_path = compute_module_path(self.filepath)
            if not module_path:
                return None  # Standalone __init__.py with no package — can't resolve
            qualified = f'{module_path}.{qualified}'

        result = self._inspector.check(qualified)
        if result is not None and not result.is_hashable:
            return self._get_type_name(node)

        return None

    def _find_unhashable_in_slice(self, node: ast.expr) -> str | None:
        """Check slice for unhashable type patterns."""
        if isinstance(node, ast.Tuple):
            for elt in node.elts:
                result = self._find_unhashable_field(elt)
                if result:
                    return result
            return None
        return self._find_unhashable_field(node)

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
            kind, resolved_name = self._classify_subscript(node)
            if kind is SubscriptKind.LITERAL:
                return None
            if kind is SubscriptKind.ANNOTATED:
                return self._find_forbidden_type(self._get_annotated_type_arg(node))

            # Container-level checks (mutable, transparent, allowed)
            container_name = self._get_type_name(node.value)
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
            if container_name in ALLOWED_CONTAINERS:
                return None

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
        self,
        node: ast.expr,
        allowed_positions: Sequence[int] | None,
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
        # Single type arg at position 0
        elif 0 in allowed_positions:
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
            kind, _ = self._classify_subscript(node)
            if kind is SubscriptKind.LITERAL:
                return None
            if kind is SubscriptKind.ANNOTATED:
                return self._find_forbidden_type_no_any(self._get_annotated_type_arg(node))

            # Container-level checks (mutable, transparent, allowed)
            container_name = self._get_type_name(node.value)
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
            if container_name in ALLOWED_CONTAINERS:
                return None

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
    _DIRECTIVE_CODES: Mapping[TypeViolationKind, DirectiveCode] = {
        'mutable': 'mutable-type',
        'loose': 'loose-typing',
        'tuple-field': 'tuple-field',
        'hashable-field': 'hashable-field',
    }

    def _has_directive(
        self,
        lineno: int,
        violation_kind: TypeViolationKind,
        *,
        scope_start: int | None = None,
        scope_end: int | None = None,
    ) -> bool:
        """Check if line (or nearby lines within scope) has a directive comment.

        When scope boundaries are provided, scans the full scope range. This
        handles ruff-format wrapping function signatures across multiple lines:
        the directive may be on the ``def`` line (above the violation) or on
        the closing ``):`` line (below the violation).

        Without scope boundaries, falls back to scanning 4 lines forward from
        the violation (the original heuristic for non-scoped annotations).
        """
        prefix_lower = DIRECTIVE_PREFIX.lower()
        expected_code = self._DIRECTIVE_CODES.get(violation_kind)
        if not expected_code:
            return False

        start = scope_start if scope_start is not None else lineno
        end = min(scope_end if scope_end is not None else lineno + 4, len(self.source_lines))

        for check_lineno in range(start, end + 1):
            if check_lineno < 1:
                continue
            line = self.source_lines[check_lineno - 1]

            if prefix_lower not in line.lower():
                continue

            directive_idx = line.lower().find(prefix_lower)
            codes_part = line[directive_idx + len(DIRECTIVE_PREFIX) :].strip()

            # Strip trailing comment (# explanation) and rationale after separator
            if ' #' in codes_part:
                codes_part = codes_part.split(' #')[0].strip()

            codes = [c.strip().lower().split()[0] for c in codes_part.split(',') if c.strip()]
            if expected_code in codes:
                self.matched_directive_lines.add((check_lineno, expected_code))
                return True

        return False


# -- Decorator & Class Inspection Helpers -------------------------------------


def is_frozen_dataclass(node: ast.ClassDef) -> bool | None:
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


def is_frozen_attrs(node: ast.ClassDef, import_map: Mapping[LocalName, QualifiedName]) -> bool | None:
    """Check if class has an attrs decorator and whether it's frozen.

    Uses import tracking to resolve decorator names, avoiding false positives
    from custom decorators with similar names.

    Returns:
        True if frozen attrs class (@frozen, @define(frozen=True), etc.)
        False if non-frozen attrs class (@define, @mutable, etc.)
        None if not an attrs class
    """
    # Fully qualified attrs decorator names
    ALWAYS_FROZEN = {'attrs.frozen', 'attr.frozen'}
    ALWAYS_MUTABLE = {'attrs.mutable', 'attr.mutable'}
    CONFIGURABLE = {'attrs.define', 'attr.define', 'attrs.s', 'attr.s', 'attrs.attrs', 'attr.attrs'}

    for decorator in node.decorator_list:
        # Resolve decorator to qualified name
        decorator_node = decorator.func if isinstance(decorator, ast.Call) else decorator
        resolved = resolve_qualified_name(decorator_node, import_map)

        # Check against known attrs patterns
        if resolved in ALWAYS_FROZEN:
            return True
        if resolved in ALWAYS_MUTABLE:
            return False
        if resolved in CONFIGURABLE:
            # For Call decorators with frozen= keyword
            if isinstance(decorator, ast.Call):
                for kw in decorator.keywords:
                    if kw.arg == 'frozen' and isinstance(kw.value, ast.Constant):
                        return kw.value.value is True
            # Default for configurable decorators without frozen=True
            return False

    return None


def is_variable_length_tuple(node: ast.expr, import_map: Mapping[LocalName, QualifiedName]) -> bool:
    """Check if an AST node is a variable-length tuple annotation: tuple[X, ...].

    Only matches the homogeneous form with Ellipsis. Does NOT match:
    - tuple (bare, no subscript)
    - tuple[int, str] (fixed-length)
    - tuple[()] (empty tuple)
    """
    if not isinstance(node, ast.Subscript):
        return False

    # Check the container resolves to tuple
    resolved = resolve_qualified_name(node.value, import_map)
    if isinstance(node.value, ast.Name):
        if node.value.id != 'tuple' and resolved not in ('builtins.tuple', 'typing.Tuple'):
            return False
    elif isinstance(node.value, ast.Attribute):
        if resolved not in ('builtins.tuple', 'typing.Tuple'):
            return False
    else:
        return False

    # Check slice is (Type, Ellipsis)
    if not isinstance(node.slice, ast.Tuple):
        return False
    if len(node.slice.elts) != 2:
        return False
    return isinstance(node.slice.elts[1], ast.Constant) and node.slice.elts[1].value is Ellipsis


def is_strict_var(name: str) -> bool:
    """Check if a name matches any recognized strict variable prefix."""
    return any(name.startswith(prefix) for prefix in STRICT_VAR_PREFIXES) and name.endswith('__')


def validate_strict_var_in_class(name: str, class_name: str, lineno: int) -> None:
    """Validate a __strict_*__ variable found in a class body.

    Raises:
        ValueError: If unrecognized
    """
    if name not in RECOGNIZED_STRICT_VARS:
        raise ValueError(
            f'{class_name}:{lineno}: Unrecognized strict variable '
            f"'{name}'. Did you mean one of: {', '.join(sorted(RECOGNIZED_STRICT_VARS))}?",
        )


def get_strict_hashable_fields(node: ast.ClassDef) -> bool:
    """Check if class body declares __strict_typing_linter__hashable_fields__ = True.

    Also validates unrecognized __strict_*__ variables at the class level (fail-fast on typos).

    Raises:
        ValueError: If unrecognized or misplaced __strict_*__ variable found
    """
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and is_strict_var(target.id):
                    validate_strict_var_in_class(target.id, node.name, item.lineno)
                    if target.id == '__strict_typing_linter__hashable_fields__':
                        if isinstance(item.value, ast.Constant):
                            return item.value.value is True
        # Also handle annotated assignment: ClassVar[bool] = True
        elif isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name) and is_strict_var(item.target.id):
                validate_strict_var_in_class(item.target.id, node.name, item.lineno)
                if item.target.id == '__strict_typing_linter__hashable_fields__' and item.value is not None:
                    if isinstance(item.value, ast.Constant):
                        return item.value.value is True
    return False


# -- Module Ordering Checks ---------------------------------------------------


# Recognized __strict_* variables (fail-fast on typos)
RECOGNIZED_STRICT_VARS: Set[str] = {
    '__strict_typing_linter__hashable_fields__',
}

CLASS_LEVEL_STRICT_VARS: Set[str] = {
    '__strict_typing_linter__hashable_fields__',
}

# Prefixes that trigger fail-fast validation on unrecognized names
STRICT_VAR_PREFIXES: Sequence[str] = (
    '__strict_typing_linter__',
    '__strict_',
)


def extract_all_names(tree: ast.Module) -> Set[str] | None:
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
class Definition:
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


def _collect_import_time_refs(tree: ast.Module) -> Set[str]:
    """Collect private function names referenced in module-level non-definition statements.

    These are functions used in type aliases, Annotated[..., BeforeValidator(fn)],
    Discriminator(fn), decorator arguments, etc. They must be defined before the
    classes/aliases that reference them — exempt from "private after public" ordering.
    """
    # Collect all private function names first
    private_fns = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('_')
    }
    if not private_fns:
        return set()

    # Scan module-level statements for references to private functions.
    # Includes: type alias assignments, class body annotations, and function/class decorators.
    # Excludes: function/method bodies (those are runtime-only, no import-time dependency).
    refs: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip body but scan decorators (evaluated at definition time)
            for decorator in node.decorator_list:
                for child in ast.walk(decorator):
                    if isinstance(child, ast.Name) and child.id in private_fns:
                        refs.add(child.id)
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in private_fns:
                refs.add(child.id)

    return refs


def extract_definitions(tree: ast.Module, all_names: Set[str] | None) -> Sequence[Definition]:
    """Extract top-level class and function definitions."""
    definitions = []
    all_names = all_names or set()
    import_time_refs = _collect_import_time_refs(tree)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            definitions.append(
                Definition(
                    name=node.name,
                    line=node.lineno,
                    is_class=True,
                    is_private=node.name.startswith('_'),
                    in_all=node.name in all_names,
                ),
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private functions referenced at import time (BeforeValidator, Discriminator, etc.)
            if node.name in import_time_refs:
                continue
            definitions.append(
                Definition(
                    name=node.name,
                    line=node.lineno,
                    is_class=False,
                    is_private=node.name.startswith('_'),
                    in_all=node.name in all_names,
                ),
            )

    return definitions


def has_ordering_directive(source_lines: Sequence[str], lineno: int) -> bool:
    """Check if line has ordering suppression directive."""
    if 0 < lineno <= len(source_lines):
        line = source_lines[lineno - 1].lower()
        if DIRECTIVE_PREFIX.lower() in line:
            idx = line.find(DIRECTIVE_PREFIX.lower())
            codes_part = line[idx + len(DIRECTIVE_PREFIX) :]
            codes = [c.strip().split()[0] for c in codes_part.split(',') if c.strip()]
            return 'ordering' in codes or 'class-ordering' in codes
    return False


def check_module_ordering(
    filepath: Path,
    tree: ast.Module,
    source_lines: Sequence[str],
    raw_ordering: list[tuple[int, str]]
    | None = None,  # strict_typing_linter.py: mutable-type — caller appends to this list
) -> Sequence[OrderingViolation]:
    """Check module ordering: __all__ items first, public before private, classes before functions."""
    violations: list[OrderingViolation] = []

    all_names = extract_all_names(tree)
    definitions = extract_definitions(tree, all_names)

    if not definitions:
        return violations

    expected_order = sorted(definitions, key=lambda d: d.sort_key)

    for actual, expected in zip(definitions, expected_order, strict=True):
        if actual.name != expected.name:
            # Determine violation kind: class-ordering when the only difference is type_group
            # (i.e., a function appears where a class should be, or vice versa)
            is_class_ordering = (
                actual.sort_key[0] == expected.sort_key[0]  # same all_group
                and actual.sort_key[1] == expected.sort_key[1]  # same private_group
                and actual.sort_key[2] != expected.sort_key[2]  # different type_group
            )
            violation_kind: OrderingViolationKind = 'class-ordering' if is_class_ordering else 'ordering'

            # Record raw violation for unused-directive detection
            if raw_ordering is not None:
                raw_ordering.append((actual.line, violation_kind))

            if has_ordering_directive(source_lines, actual.line):
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
                    kind=violation_kind,
                    message=reason,
                    source_line=source_lines[actual.line - 1].strip() if actual.line <= len(source_lines) else '',
                    suggestion='Reorder: __all__ items (classes→functions) → public (classes→functions) → private (classes→functions)',
                ),
            )
            break  # Report first violation only

    return violations


def _is_entry_point(filepath: Path, source_lines: Sequence[str]) -> bool:
    """Detect entry point files that don't need __all__.

    Returns True if ANY of these signals match:
    - __main__.py: package entry point (python -m)
    - main.py: CLI entry point by convention (pyproject.toml entry points)
    - Shebang on line 1: directly executable script
    """
    if filepath.name in ('__main__.py', 'main.py'):
        return True
    return bool(source_lines) and source_lines[0].startswith('#!')


def check_all_defined(
    filepath: Path,
    tree: ast.Module,
    source_lines: Sequence[str],
    raw_ordering: list[tuple[int, str]]
    | None = None,  # strict_typing_linter.py: mutable-type — caller appends to this list
) -> Sequence[OrderingViolation]:
    """Check that __all__ is defined."""
    all_names = extract_all_names(tree)
    if all_names is None:
        line = 1
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
                line = node.lineno
                break

        # Record raw violation for unused-directive detection
        if raw_ordering is not None:
            raw_ordering.append((line, 'missing-all'))

        return [
            OrderingViolation(
                filepath=filepath,
                line=line,
                kind='missing-all',
                message='__all__ is not defined',
                source_line=source_lines[line - 1].strip() if line <= len(source_lines) else '',
                suggestion='Add __all__ = [...] to explicitly declare public API',
            ),
        ]
    return []


def has_trailing_comma(source_lines: Sequence[str], list_node: ast.List) -> bool:
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
    raw_ordering: list[tuple[int, str]]
    | None = None,  # strict_typing_linter.py: mutable-type — caller appends to this list
) -> Sequence[OrderingViolation]:
    """Check that __all__ has trailing comma when it has 1+ items."""
    violations: list[OrderingViolation] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, ast.List) and node.value.elts:
                        if not has_trailing_comma(source_lines, node.value):
                            # Record raw violation for unused-directive detection
                            if raw_ordering is not None:
                                raw_ordering.append((node.lineno, 'trailing-comma'))
                            if has_ordering_directive(source_lines, node.lineno):
                                continue
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
                                ),
                            )
    return violations


def check_class_method_ordering(
    filepath: Path,
    tree: ast.Module,
    source_lines: Sequence[str],
    raw_ordering: list[tuple[int, str]]
    | None = None,  # strict_typing_linter.py: mutable-type — caller appends to this list
) -> Sequence[OrderingViolation]:
    """Check that public methods come before private methods within classes."""
    violations: list[OrderingViolation] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Extract methods from class body
        methods: list[tuple[str, int, bool]] = []  # (name, line, is_private)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_private = item.name.startswith('_') and not (item.name.startswith('__') and item.name.endswith('__'))
                methods.append((item.name, item.lineno, is_private))

        if not methods:
            continue

        # Check ordering: public methods should come before private methods
        seen_private = False
        for name, line, is_private in methods:
            if is_private:
                seen_private = True
            elif seen_private:
                # Public method after private method - violation
                # Record raw violation for unused-directive detection
                if raw_ordering is not None:
                    raw_ordering.append((line, 'ordering'))

                if has_ordering_directive(source_lines, line):
                    continue

                violations.append(
                    OrderingViolation(
                        filepath=filepath,
                        line=line,
                        kind='ordering',
                        message=f"public method '{name}' should come before private methods in class '{node.name}'",
                        source_line=source_lines[line - 1].strip() if line <= len(source_lines) else '',
                        suggestion='Reorder: dunder methods → public methods → private methods',
                    ),
                )
                break  # Report first violation per class only

    return violations


# -- Unused Directive Detection -----------------------------------------------


def collect_directives(source_lines: Sequence[str]) -> Sequence[DirectiveInstance]:
    """Scan source for suppression directives, returning structured instances.

    Skips skip-file directives (handled separately in check_file).
    Only matches directives in actual comments (not in strings or code).
    """
    prefix_lower = DIRECTIVE_PREFIX.lower()
    directives: list[DirectiveInstance] = []
    in_multiline_string = False

    for i, line in enumerate(source_lines):
        lineno = i + 1
        stripped = line.strip()

        # Track triple-quoted string boundaries (skip directives inside docstrings)
        for quote in ('"""', "'''"):
            count = stripped.count(quote)
            if count % 2 != 0:
                in_multiline_string = not in_multiline_string

        if in_multiline_string:
            continue

        line_lower = line.lower()
        if prefix_lower not in line_lower:
            continue
        if 'skip-file' in line_lower:
            continue

        idx = line_lower.find(prefix_lower)

        # Skip if the # is inside a string literal (heuristic: count unmatched quotes before idx)
        before = line[:idx]
        if before.count("'") % 2 != 0 or before.count('"') % 2 != 0:
            continue

        codes_part = line_lower[idx + len(DIRECTIVE_PREFIX) :]

        # Strip rationale after separator (—, --, //, #)
        for sep in (' — ', ' -- ', ' // ', ' #'):
            if sep in codes_part:
                codes_part = codes_part.split(sep)[0]
                break

        codes = [c.strip().split()[0] for c in codes_part.split(',') if c.strip()]
        if codes:
            directives.append(DirectiveInstance(line=lineno, codes=codes, raw_text=line.strip()))

    return directives


def find_unused_directives(
    directives: Sequence[DirectiveInstance],
    raw_violations: Sequence[tuple[int, str]],
    filepath: Path,
    source_lines: Sequence[str],
    matched_directive_lines: Set[tuple[int, str]] | None = None,
) -> Sequence[Violation]:
    """Compare directive inventory against raw violations to find stale directives.

    Uses two matching strategies:
    1. Exact match: directive was recorded as matched during scope-aware checking
       (handles multi-line signatures where directive and violation are far apart).
    2. Proximity match: directive on line D suppresses violations on [D-4, D]
       (handles non-scoped annotations where _has_directive uses the 4-line window).
    """
    unused: list[Violation] = []
    matched = matched_directive_lines or set()

    for directive in directives:
        for code in directive.codes:
            # Strategy 1: scope-aware match recorded during checking
            found = (directive.line, code) in matched
            # Strategy 2: proximity match (inverse of the 4-line forward window)
            if not found:
                found = any(
                    kind == code and (directive.line - 4) <= lineno <= directive.line for lineno, kind in raw_violations
                )
            if not found:
                unused.append(
                    OrderingViolation(
                        filepath=filepath,
                        line=directive.line,
                        kind='unused-directive',
                        message=f"Suppression directive '{code}' does not match any violation",
                        source_line=source_lines[directive.line - 1].strip()
                        if directive.line <= len(source_lines)
                        else '',
                        suggestion='Remove the stale suppression directive',
                    ),
                )
                break  # One unused code per directive is enough

    return unused


# -- Utility Functions --------------------------------------------------------


def build_import_map(tree: ast.Module) -> Mapping[LocalName, QualifiedName]:
    """Build a mapping from local names to fully qualified names from imports.

    Examples:
        import typing -> {'typing': 'typing'}
        import typing as t -> {'t': 'typing'}
        from typing import Generator -> {'Generator': 'typing.Generator'}
        from typing import Generator as Gen -> {'Gen': 'typing.Generator'}
    """
    import_map: dict[LocalName, QualifiedName] = {}

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


def compute_module_path(filepath: Path) -> str:
    """Compute importable module path by walking the package hierarchy.

    Resolves to absolute path first to prevent infinite loops when
    Path('.').parent returns Path('.') on relative paths.

    Walks up from filepath through __init__.py chain to find the top-level package.
    Example: mcp/document-search/document_search/schemas/vectors.py
             -> 'document_search.schemas.vectors'
    """
    filepath = filepath.resolve()
    # __init__.py represents the package itself, not a submodule named '__init__'
    if filepath.name == '__init__.py':
        parts: list[str] = []
        parent = filepath.parent
    else:
        parts = [filepath.stem]
        parent = filepath.parent
    while (parent / '__init__.py').exists():
        parts.append(parent.name)
        parent = parent.parent
    return '.'.join(reversed(parts))


def find_source_root(path: Path) -> Path:
    """Find the source root for a path (directory above the top-level package).

    For directories, returns the directory itself (resolved to absolute).
    For files, walks up past __init__.py chain to find the package root.
    Resolves to absolute first to prevent infinite loops on relative paths.
    """
    if path.is_dir():
        return path.resolve()
    path = path.resolve()
    parent = path.parent
    while (parent / '__init__.py').exists():
        parent = parent.parent
    return parent


def resolve_qualified_name(node: ast.expr, import_map: Mapping[LocalName, QualifiedName]) -> QualifiedName:
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
            resolved_base = resolve_qualified_name(base_node, import_map)
            return f'{resolved_base}.{node.attr}'

    return ''


# -- Output Formatting --------------------------------------------------------


def format_violation(v: Violation) -> str:
    """Format a violation for display."""
    if isinstance(v, TypeViolation):
        # Map violation kind to human-readable description and directive code
        KIND_INFO: Mapping[TypeViolationKind, tuple[str, DirectiveCode]] = {
            'mutable': ('Mutable type', 'mutable-type'),
            'loose': ('Loose type', 'loose-typing'),
            'tuple-field': ('Variable-length tuple', 'tuple-field'),
            'hashable-field': ('Unhashable type', 'hashable-field'),
        }
        type_desc, directive_code = KIND_INFO.get(v.kind, ('Type error', v.kind))
        return (
            f'{v.filepath}:{v.line}:{v.column}: error: '
            f"{type_desc} '{v.bad_type}' in {v.context} annotation\n"
            f'    {v.source_line}\n'
            f'    Suggestion: {v.suggestion}\n'
            f'    Silence with: {DIRECTIVE_PREFIX} {directive_code}'
        )
    else:  # OrderingViolation
        directive_code = v.kind if v.kind != 'unused-directive' else 'ordering'
        return (
            f'{v.filepath}:{v.line}: error: {v.message}\n'
            f'    {v.source_line}\n'
            f'    Suggestion: {v.suggestion}\n'
            f'    Silence with: {DIRECTIVE_PREFIX} {directive_code}'
        )


if __name__ == '__main__':
    sys.exit(main())
