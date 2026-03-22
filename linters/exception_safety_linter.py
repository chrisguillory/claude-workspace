#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Exception safety linter for Python code.

Detects common exception handling anti-patterns that cause silent failures,
lost debugging context, and broken async cancellation. Designed to catch
issues commonly introduced by AI-generated code.

Rules:
    EXC001 bare-except              Bare except catches KeyboardInterrupt, SystemExit, CancelledError, GeneratorExit
    EXC002 swallowed-exception      Broad catch (Exception/BaseException) without re-raise
    EXC003 finally-control-flow     Return/break/continue in finally suppresses exceptions
    EXC004 raise-without-from       Raising new exception without 'from' uses implicit chaining
    EXC005 unused-exception-var     Catching 'as e' but never using e
    EXC006 logger-no-exc-info       Logger error/warning calls in except without exc_info
    EXC007 cancelled-not-raised     CancelledError caught but not re-raised in async
    EXC008 generator-exit-not-raised GeneratorExit caught but not re-raised in generator

Escape hatches (inline suppression):
    # exception_safety_linter.py: skip-file
    # exception_safety_linter.py: bare-except
    # exception_safety_linter.py: swallowed-exception, raise-without-from

Design Philosophy:
    - Error-only, no auto-fix: Forces conscious decision at each occurrence
    - Standalone: No external dependencies, works with any Python 3.13+ install
    - Instructive: Points to test cases file for correct patterns

Usage:
    ./linters/exception_safety_linter.py           # Check current directory
    ./linters/exception_safety_linter.py <files>   # Check specific files
    ./linters/exception_safety_linter.py src/      # Check specific directory

Exit codes:
    0 - No violations found
    1 - Violations found

Known Limitations (static analysis trade-offs):
    - Conditional re-raise: `if cond: raise` is seen as having a raise, even if
      the exception is swallowed when the condition is False. Use suppression
      directive if the conditional pattern is intentional.
    - Logger detection: Any method named error/warning/critical/fatal/warn triggers
      EXC006, even on non-logger objects. Use suppression if false positive.
    - Dynamic exc_info: `logger.error(msg, **config)` where config contains exc_info
      is not detected. The linter only sees explicit `exc_info=True` keyword args.
    - Wildcard imports: `from asyncio import *` prevents CancelledError resolution.
    - Exception subclasses: Custom subclasses of CancelledError are not detected.
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

from _config import find_config, get_per_file_ignored_codes, load_per_file_ignores

# Name resolution (import map usage)
type QualifiedName = str  # Fully qualified dotted name (e.g., 'builtins.Exception', 'asyncio.CancelledError')
type LocalName = str  # Local identifier as it appears in source (e.g., 'Exception', 'CancelledError')

# Error code identifier
type ErrorCode = str  # Violation code like 'EXC001', 'EXC002', etc.

# -- Configuration ------------------------------------------------------------

# Directive prefix - uses script filename for discoverability
DIRECTIVE_PREFIX = '# exception_safety_linter.py:'

# Test cases file for reference
TEST_CASES_FILE = 'tests/linters/edge_cases/exception_safety_test_cases.py'

# Broad exception types (fully qualified only - like strict_typing_linter.py pattern)
# These are the canonical names after import resolution
QUALIFIED_BROAD_EXCEPTIONS: Set[QualifiedName] = {
    'builtins.Exception',
    'builtins.BaseException',
}

# Short names for builtins when not explicitly imported
BUILTIN_BROAD_EXCEPTIONS: Set[LocalName] = {
    'Exception',
    'BaseException',
}

# CancelledError types (fully qualified)
QUALIFIED_CANCELLED_ERROR: Set[QualifiedName] = {
    'asyncio.CancelledError',
}

# Short name for CancelledError
BUILTIN_CANCELLED_ERROR: Set[LocalName] = {
    'CancelledError',
}

# Exception types that catch CancelledError (Python 3.8+)
# Note: CancelledError inherits from BaseException, NOT Exception.
# So `except Exception:` does NOT catch CancelledError.
QUALIFIED_CATCHES_CANCELLED: Set[QualifiedName] = {
    'builtins.BaseException',
}

BUILTIN_CATCHES_CANCELLED: Set[LocalName] = {
    'BaseException',
}

# GeneratorExit types (fully qualified)
QUALIFIED_GENERATOR_EXIT: Set[QualifiedName] = {
    'builtins.GeneratorExit',
}

# Short name for GeneratorExit
BUILTIN_GENERATOR_EXIT: Set[LocalName] = {
    'GeneratorExit',
}

# Exception types that catch GeneratorExit
# Like CancelledError, GeneratorExit inherits from BaseException, NOT Exception.
# So `except Exception:` does NOT catch GeneratorExit.
QUALIFIED_CATCHES_GENERATOR_EXIT: Set[QualifiedName] = {
    'builtins.BaseException',
}

BUILTIN_CATCHES_GENERATOR_EXIT: Set[LocalName] = {
    'BaseException',
}

# Logger method names that should have exc_info in except blocks
# Note: These are method names from logging module's Logger class
LOGGER_ERROR_METHODS: Set[str] = {
    'error',
    'critical',
    'fatal',
    'warning',
    'warn',
}

# -- Data Types ---------------------------------------------------------------

# Violation kind - used as directive codes and internal identifiers
type ViolationKind = Literal[
    'bare-except',
    'swallowed-exception',
    'finally-control-flow',
    'raise-without-from',
    'unused-exception-var',
    'logger-no-exc-info',
    'cancelled-not-raised',
    'generator-exit-not-raised',
    'unused-directive',
]

# Maps kind to error code for display
ERROR_CODES: Mapping[ViolationKind, ErrorCode] = {
    'bare-except': 'EXC001',
    'swallowed-exception': 'EXC002',
    'finally-control-flow': 'EXC003',
    'raise-without-from': 'EXC004',
    'unused-exception-var': 'EXC005',
    'logger-no-exc-info': 'EXC006',
    'cancelled-not-raised': 'EXC007',
    'generator-exit-not-raised': 'EXC008',
    'unused-directive': 'EXC009',
}

# Short descriptions for each violation
VIOLATION_MESSAGES: Mapping[ViolationKind, str] = {
    'bare-except': 'Bare except catches KeyboardInterrupt, SystemExit, CancelledError, GeneratorExit',
    'swallowed-exception': 'Broad catch without re-raise hides errors',
    'finally-control-flow': 'Control flow in finally suppresses exceptions',
    'raise-without-from': 'Raising new exception without "from" uses implicit chaining (unclear intent)',
    'unused-exception-var': 'Exception variable captured but never used',
    'logger-no-exc-info': 'Logger call without exc_info loses traceback',
    'cancelled-not-raised': 'CancelledError swallowed (task.cancelled() returns False, breaking orchestrator logic)',
    'generator-exit-not-raised': 'GeneratorExit swallowed (generator.close() cannot complete, resources leak)',
    'unused-directive': 'Suppression directive does not match any violation',
}

# Fix suggestions for each violation
FIX_SUGGESTIONS: Mapping[ViolationKind, str] = {
    'bare-except': "Use 'except Exception:' or catch specific types",
    'swallowed-exception': "Add 'raise' after cleanup, or catch specific expected exceptions",
    'finally-control-flow': 'Move return/break/continue outside finally block',
    'raise-without-from': "Use 'raise NewError() from e' for explicit chaining, or 'from None' to suppress",
    'unused-exception-var': "Remove 'as e' or use e (log it, inspect it, etc.)",
    'logger-no-exc-info': 'Use logger.exception() or add exc_info=True',
    'cancelled-not-raised': "Add 'raise' after cleanup, or remove the try/except entirely if no cleanup needed",
    'generator-exit-not-raised': "Add 'raise' after cleanup, or use 'finally' block instead (preferred)",
    'unused-directive': 'Remove the stale suppression directive',
}


@dataclass(frozen=True)
class Violation:
    """A detected exception safety violation."""

    filepath: Path
    line: int
    column: int
    kind: ViolationKind
    source_line: str


@dataclass(frozen=True)
class TestReference:
    """Reference to correct pattern in test file."""

    kind: ViolationKind
    function_name: str
    line: int


@dataclass(frozen=True)
class DirectiveInstance:
    """A suppression directive found in source code."""

    line: int
    codes: Sequence[str]
    raw_text: str


# -- Main Entry Point ---------------------------------------------------------


def main() -> int:
    """CLI entry point: parse args, collect files, check, report."""
    args = parse_args()
    exclude_dirs = set(args.exclude) | {'.venv', 'venv', '__pycache__', '.git'}
    ignored_kinds: set[ViolationKind] = set(args.ignore)
    respect_gitignore = not args.no_gitignore

    # Collect files to check
    files: list[Path] = []
    for arg in args.paths:
        path = Path(arg)
        if arg == '.' or path.is_dir():
            files.extend(find_python_files(path, exclude_dirs, respect_gitignore))
        elif path.suffix == '.py' and path.is_file():
            files.append(path)

    if not files:
        return 0

    # Resolve config path once if --config was given
    explicit_config = Path(args.config) if args.config else None

    # Check all files
    all_violations: list[Violation] = []
    for filepath in files:
        # Per-file-ignores from pyproject.toml
        per_file_codes: Set[str] = set()
        skip_file_via_config = False
        if not args.no_config:
            if explicit_config is not None:
                config_path = explicit_config
                project_root = explicit_config.parent
            else:
                result = find_config(filepath, 'exception-safety-linter')
                if result is not None:
                    config_path, project_root = result
                else:
                    config_path = None
                    project_root = None

            if config_path is not None and project_root is not None:
                per_file_ignores = load_per_file_ignores('exception-safety-linter', config_path)
                per_file_codes = get_per_file_ignored_codes(filepath, per_file_ignores, project_root)
                if 'skip-file' in per_file_codes:
                    skip_file_via_config = True

        if skip_file_via_config:
            continue

        try:
            violations = check_file(
                filepath,
                respect_skip_file=not args.no_skip_file,
                report_unused_directives=args.report_unused_directives,
            )

            # Filter by per-file ignored codes (codes are violation kinds directly)
            if per_file_codes:
                violations = [v for v in violations if v.kind not in per_file_codes]

            all_violations.extend(violations)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f'{filepath}: {e}', file=sys.stderr)

    # Filter ignored kinds
    if ignored_kinds:
        all_violations = [v for v in all_violations if v.kind not in ignored_kinds]

    # Report violations
    if all_violations:
        for v in sorted(all_violations, key=lambda x: (x.filepath, x.line)):
            print(format_violation(v))
            print()

        # Summary
        file_count = len({v.filepath for v in all_violations})
        print(f'Found {len(all_violations)} violation(s) in {file_count} file(s).')

        # Find test file references for violated rules
        test_file = Path(__file__).parent.parent / TEST_CASES_FILE
        test_refs = find_test_references(test_file)

        # Show references for the specific violations found
        found_kinds = {v.kind for v in all_violations}
        relevant_refs = {k: v for k, v in test_refs.items() if k in found_kinds}

        if relevant_refs:
            print('For correct patterns, see:')
            for kind in sorted(relevant_refs.keys(), key=lambda k: ERROR_CODES[k]):
                ref = relevant_refs[kind]
                code = ERROR_CODES[kind]
                print(f'  {code} {kind}: {TEST_CASES_FILE}:{ref.line} ({ref.function_name})')
        else:
            print(f'For correct patterns, see: {TEST_CASES_FILE}')

        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Check for exception safety violations.',
        epilog=f'For correct patterns and examples, see: {TEST_CASES_FILE}',
    )
    parser.add_argument(
        'paths',
        nargs='*',
        default=['.'],
        help='Files or directories to check (default: current directory)',
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
        choices=list(ERROR_CODES.keys()),
        help='Violation codes to ignore globally',
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
    respect_skip_file: bool = True,
    report_unused_directives: bool = False,
) -> Sequence[Violation]:
    """Check a single file for exception safety violations."""
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
    import_map = build_import_map(tree)

    checker = ExceptionSafetyChecker(filepath, source_lines, import_map)
    checker.visit(tree)

    # If skip-file is present and valid, suppress normal violations
    if has_skip_file and respect_skip_file:
        if report_unused_directives and not checker.raw_violations:
            # skip-file is stale — no violations to suppress
            skip_line = next(
                i + 1
                for i, line in enumerate(source_lines[:10])
                if prefix_lower in line.lower() and 'skip-file' in line.lower()
            )
            return [
                Violation(
                    filepath=filepath,
                    line=skip_line,
                    column=0,
                    kind='unused-directive',
                    source_line=source_lines[skip_line - 1].strip(),
                ),
            ]
        return []

    violations = list(checker.violations)

    if report_unused_directives:
        directives = collect_directives(source_lines)
        unused = find_unused_directives(directives, checker.raw_violations, filepath, source_lines)
        violations.extend(unused)

    return violations


def find_python_files(
    root: Path,
    exclude_dirs: Set[str],
    respect_gitignore: bool = True,
) -> Sequence[Path]:
    """Find all .py files recursively under root."""
    all_files = sorted(path for path in root.rglob('*.py') if not any(part in exclude_dirs for part in path.parts))

    if respect_gitignore:
        ignored = get_git_ignored_files(all_files, root)
        all_files = [f for f in all_files if f not in ignored]

    return all_files


# -- AST Visitor --------------------------------------------------------------


class ExceptionSafetyChecker(ast.NodeVisitor):
    """AST visitor that checks for exception safety violations."""

    def __init__(
        self,
        filepath: Path,
        source_lines: Sequence[str],
        import_map: Mapping[LocalName, QualifiedName],
    ) -> None:
        self.filepath = filepath
        self.source_lines = source_lines
        self.import_map = import_map
        self.violations: list[Violation] = []
        self.raw_violations: list[tuple[int, ViolationKind]] = []

        # Context tracking
        self._in_async_function = False
        self._in_generator_function = False
        self._in_except_handler = False
        self._in_finally_block = False
        self._handler_stack: list[ast.ExceptHandler] = []

    # -- Visitor Methods ------------------------------------------------------

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track async function context and generator status."""
        prev_async = self._in_async_function
        prev_gen = self._in_generator_function

        self._in_async_function = True
        self._in_generator_function = self._is_generator_function(node)

        self.generic_visit(node)

        self._in_async_function = prev_async
        self._in_generator_function = prev_gen

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit regular function (reset async context, track generator status)."""
        prev_async = self._in_async_function
        prev_gen = self._in_generator_function

        self._in_async_function = False
        self._in_generator_function = self._is_generator_function(node)

        self.generic_visit(node)

        self._in_async_function = prev_async
        self._in_generator_function = prev_gen

    def visit_Try(self, node: ast.Try) -> None:
        """Visit try block - check handlers and finally."""
        self._visit_try_structure(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        """Visit try* block (except* handlers) - same structure as try."""
        self._visit_try_structure(node)

    def _visit_try_structure(self, node: ast.Try | ast.TryStar) -> None:
        """Common visitor for try and try* blocks."""
        # Visit try body
        for child in node.body:
            self.visit(child)

        # Visit except/except* handlers
        for handler in node.handlers:
            self.visit(handler)

        # Visit else
        for child in node.orelse:
            self.visit(child)

        # Visit finally with context
        if node.finalbody:
            prev = self._in_finally_block
            self._in_finally_block = True
            for child in node.finalbody:
                self.visit(child)
            self._in_finally_block = prev

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Check except handler for violations."""
        # EXC001: Bare except
        if node.type is None:
            self._add_violation(node, 'bare-except')

        # Track context for nested checks
        prev_in_except = self._in_except_handler
        self._in_except_handler = True
        self._handler_stack.append(node)

        # Check for broad exception types, CancelledError, and GeneratorExit
        is_broad_catch = False
        is_cancelled_error_catch = False
        is_generator_exit_catch = False

        if node.type is not None:
            is_broad_catch = self._is_broad_exception_type(node.type)
            is_cancelled_error_catch = self._is_cancelled_error_type(node.type)
            is_generator_exit_catch = self._is_generator_exit_type(node.type)

        # Visit handler body (may recurse into nested try/except)
        self.generic_visit(node)

        # EXC002: Swallowed exception (broad catch without raise)
        # Skip if nested inside a handler that re-raises - the outer handler
        # preserves the original exception, so the inner is just handling
        # cleanup/enrichment errors
        if is_broad_catch and not self._handler_has_raise(node):
            if not self._outer_handler_reraises():
                self._add_violation(node, 'swallowed-exception')

        # EXC005: Unused exception variable
        if node.name and not self._name_used_in_body(node.name, node.body):
            self._add_violation(node, 'unused-exception-var')

        # EXC007: CancelledError not raised in async function
        # Same reasoning - skip if outer handler re-raises
        if self._in_async_function and is_cancelled_error_catch and not self._handler_has_raise(node):
            if not self._outer_handler_reraises():
                self._add_violation(node, 'cancelled-not-raised')

        # EXC008: GeneratorExit not raised in generator function
        # Same reasoning - skip if outer handler re-raises
        if self._in_generator_function and is_generator_exit_catch and not self._handler_has_raise(node):
            if not self._outer_handler_reraises():
                self._add_violation(node, 'generator-exit-not-raised')

        # Restore context
        self._handler_stack.pop()
        self._in_except_handler = prev_in_except

    def visit_Raise(self, node: ast.Raise) -> None:
        """Check raise statements."""
        # EXC004: Raise without from (in except handler, raising new exception)
        # Skip if just re-raising the caught exception variable (raise e where e is caught)
        if (
            self._in_except_handler
            and node.exc is not None  # Raising something (not bare raise)
            and node.cause is None  # No 'from' clause
            and not self._is_reraise_of_caught_var(node.exc)
        ):
            self._add_violation(node, 'raise-without-from')

        self.generic_visit(node)

    def _is_reraise_of_caught_var(self, exc: ast.expr) -> bool:
        """Check if the raised expression is just the caught exception variable.

        `raise e` where `e` is the caught variable is NOT implicit chaining -
        it's re-raising the same exception object. EXC004 should not fire.
        (Though bare `raise` is preferred for preserving the original traceback.)
        """
        if not isinstance(exc, ast.Name):
            return False
        if not self._handler_stack:
            return False
        # Check if the name matches the current handler's exception variable
        current_handler = self._handler_stack[-1]
        return current_handler.name == exc.id

    def visit_Return(self, node: ast.Return) -> None:
        """Check return in finally."""
        if self._in_finally_block:
            self._add_violation(node, 'finally-control-flow')
        self.generic_visit(node)

    def visit_Break(self, node: ast.Break) -> None:
        """Check break in finally."""
        if self._in_finally_block:
            self._add_violation(node, 'finally-control-flow')
        self.generic_visit(node)

    def visit_Continue(self, node: ast.Continue) -> None:
        """Check continue in finally."""
        if self._in_finally_block:
            self._add_violation(node, 'finally-control-flow')
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check logger calls in except handlers."""
        if self._in_except_handler:
            self._check_logger_call(node)
        self.generic_visit(node)

    # -- Private Helper Methods -----------------------------------------------

    def _get_source_line(self, lineno: int) -> str:
        """Get source line, stripped."""
        if 0 < lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ''

    def _has_directive(self, lineno: int, kind: ViolationKind) -> bool:
        """Check if line (or nearby continuation lines) has suppression directive.

        Scans up to 4 lines forward from the violation to handle cases where
        ruff-format wraps statements across lines (e.g., ``except (\\n Exception\\n):``),
        placing the directive on a subsequent line.
        """
        # unused-directive violations cannot be suppressed (like RUF100)
        if kind == 'unused-directive':
            return False

        prefix_lower = DIRECTIVE_PREFIX.lower()
        end = min(lineno + 4, len(self.source_lines))

        for check_lineno in range(lineno, end + 1):
            if check_lineno < 1:
                continue
            line = self.source_lines[check_lineno - 1].lower()

            if prefix_lower not in line:
                continue

            idx = line.find(prefix_lower)
            codes_part = line[idx + len(DIRECTIVE_PREFIX) :]

            # Strip trailing comment
            if ' #' in codes_part:
                codes_part = codes_part.split(' #')[0]

            codes = [c.strip().split()[0] for c in codes_part.split(',') if c.strip()]
            if kind in codes:
                return True

        return False

    def _add_violation(self, node: ast.AST, kind: ViolationKind) -> None:
        """Add a violation if not suppressed. Always records raw violations."""
        lineno = getattr(node, 'lineno', 0)
        self.raw_violations.append((lineno, kind))

        if self._has_directive(lineno, kind):
            return

        self.violations.append(
            Violation(
                filepath=self.filepath,
                line=lineno,
                column=getattr(node, 'col_offset', 0),
                kind=kind,
                source_line=self._get_source_line(lineno),
            ),
        )

    def _is_broad_exception_type(self, node: ast.expr) -> bool:
        """Check if exception type is a broad catch (Exception/BaseException).

        Uses import resolution like strict_typing_linter.py:
        1. Resolve to qualified name via import map
        2. Check against QUALIFIED_BROAD_EXCEPTIONS
        3. Fall back to short name check for builtins
        """
        if isinstance(node, ast.Tuple):
            # Multiple exceptions - check each
            return any(self._is_broad_exception_type(elt) for elt in node.elts)

        # Resolve using import map
        resolved = resolve_name(node, self.import_map)
        if resolved in QUALIFIED_BROAD_EXCEPTIONS:
            return True

        # Fall back to short name for builtins (when not explicitly imported)
        if isinstance(node, ast.Name) and node.id in BUILTIN_BROAD_EXCEPTIONS:
            return True

        return False

    def _is_cancelled_error_type(self, node: ast.expr) -> bool:
        """Check if exception type catches CancelledError."""
        if isinstance(node, ast.Tuple):
            return any(self._is_cancelled_error_type(elt) for elt in node.elts)

        # Resolve using import map
        resolved = resolve_name(node, self.import_map)

        # Only BaseException catches CancelledError (not Exception!)
        if resolved in QUALIFIED_CATCHES_CANCELLED:
            return True
        if isinstance(node, ast.Name) and node.id in BUILTIN_CATCHES_CANCELLED:
            return True

        # Check for explicit CancelledError
        if resolved in QUALIFIED_CANCELLED_ERROR:
            return True
        if isinstance(node, ast.Name) and node.id in BUILTIN_CANCELLED_ERROR:
            return True

        return False

    def _is_generator_exit_type(self, node: ast.expr) -> bool:
        """Check if exception type catches GeneratorExit."""
        if isinstance(node, ast.Tuple):
            return any(self._is_generator_exit_type(elt) for elt in node.elts)

        # Resolve using import map
        resolved = resolve_name(node, self.import_map)

        # Only BaseException catches GeneratorExit (not Exception!)
        if resolved in QUALIFIED_CATCHES_GENERATOR_EXIT:
            return True
        if isinstance(node, ast.Name) and node.id in BUILTIN_CATCHES_GENERATOR_EXIT:
            return True

        # Check for explicit GeneratorExit
        if resolved in QUALIFIED_GENERATOR_EXIT:
            return True
        if isinstance(node, ast.Name) and node.id in BUILTIN_GENERATOR_EXIT:
            return True

        return False

    def _is_generator_function(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function is a generator (contains yield expressions).

        Does NOT recurse into nested functions, classes, lambdas, or generator
        expressions. A function is only a generator if it directly contains yield.
        """
        finder = _YieldFinder()
        for stmt in func.body:
            finder.visit(stmt)
        return finder.found

    def _handler_has_raise(self, handler: ast.ExceptHandler) -> bool:
        """Check if exception handler directly contains a raise statement.

        Uses _RaiseFinder to traverse only the handler's direct control flow,
        stopping at nested try/except blocks and scope boundaries.
        """
        finder = _RaiseFinder()
        for stmt in handler.body:
            finder.visit(stmt)
        return finder.found

    def _outer_handler_reraises(self) -> bool:
        """Check if any enclosing handler (not current) re-raises.

        When an outer handler re-raises, nested handlers within it cannot
        swallow the exception that triggered the outer handler. They can
        only handle errors during cleanup/enrichment.

        Stack structure: [..., outer, current]
        We check outer and above (exclude current).
        """
        # Check all handlers except the current one (last in stack)
        return any(self._handler_has_raise(handler) for handler in self._handler_stack[:-1])

    def _name_used_in_body(self, name: str, body: Sequence[ast.stmt]) -> bool:
        """Check if a name is used in a body of statements."""
        for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
            if isinstance(node, ast.Name) and node.id == name:
                return True
        return False

    def _check_logger_call(self, node: ast.Call) -> None:
        """Check if a logger error/warning call is missing exc_info."""
        if not isinstance(node.func, ast.Attribute):
            return

        method_name = node.func.attr
        if method_name not in LOGGER_ERROR_METHODS:
            return

        # Check if exc_info=True is present
        # Note: logger.exception() is not checked - it's the correct pattern
        # and is not in LOGGER_ERROR_METHODS
        has_exc_info = False
        for keyword in node.keywords:
            if keyword.arg == 'exc_info':
                if (isinstance(keyword.value, ast.Constant) and keyword.value.value) or not isinstance(
                    keyword.value,
                    ast.Constant,
                ):
                    has_exc_info = True

        if not has_exc_info:
            self._add_violation(node, 'logger-no-exc-info')


class _RaiseFinder(ast.NodeVisitor):
    """Find raise statements in handler's direct control flow.

    Traverses control flow (if, for, while, with, match) but stops at:
    - Nested try/except/try* blocks (different exception context)
    - Nested function/class definitions (different scope)

    This prevents false negatives where a raise in a nested handler
    incorrectly counts as the outer handler re-raising.
    """

    def __init__(self) -> None:
        self.found = False

    def visit_Raise(self, node: ast.Raise) -> None:
        self.found = True

    # Stop recursion at exception context boundaries
    def visit_Try(self, node: ast.Try) -> None:
        pass  # Don't recurse into nested try blocks

    def visit_TryStar(self, node: ast.TryStar) -> None:
        pass  # Don't recurse into nested try* blocks

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        pass  # Don't recurse into nested handlers

    # Stop recursion at scope boundaries
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass  # Don't recurse into nested functions

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass  # Don't recurse into nested async functions

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass  # Don't recurse into nested classes

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass  # Don't recurse into lambdas


class _YieldFinder(ast.NodeVisitor):
    """Find yield expressions in function body, excluding nested scopes.

    Follows the same pattern as _RaiseFinder - stops at scope boundaries
    to avoid false positives from nested generators.

    A function is a generator only if it directly contains yield expressions,
    not if a nested function or generator expression contains yield.
    """

    def __init__(self) -> None:
        self.found = False

    def visit_Yield(self, node: ast.Yield) -> None:
        self.found = True

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.found = True

    # Stop at scope boundaries
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass  # Don't recurse into nested functions

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass  # Don't recurse into nested async functions

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass  # Don't recurse into nested classes

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass  # Don't recurse into lambdas

    # Generator expressions create their own scope
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        pass  # Don't recurse - genexp doesn't make containing function a generator


# -- Utility Functions --------------------------------------------------------


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
    raw_violations: Sequence[tuple[int, ViolationKind]],
    filepath: Path,
    source_lines: Sequence[str],
) -> Sequence[Violation]:
    """Compare directive inventory against raw violations to find stale directives.

    A directive on line D suppresses violations on lines [D-4, D] (matching the
    forward scan window in _has_directive: violation V finds directives on V..V+4).
    """
    unused: list[Violation] = []

    for directive in directives:
        for code in directive.codes:
            matched = any(
                kind == code and (directive.line - 4) <= lineno <= directive.line for lineno, kind in raw_violations
            )
            if not matched:
                unused.append(
                    Violation(
                        filepath=filepath,
                        line=directive.line,
                        column=0,
                        kind='unused-directive',
                        source_line=source_lines[directive.line - 1].strip()
                        if directive.line <= len(source_lines)
                        else '',
                    ),
                )
                break  # One unused code per directive is enough

    return unused


def build_import_map(tree: ast.Module) -> Mapping[LocalName, QualifiedName]:
    """Build mapping from local names to fully qualified names.

    Examples:
        import asyncio -> {'asyncio': 'asyncio'}
        from asyncio import CancelledError -> {'CancelledError': 'asyncio.CancelledError'}

    Note: Builtins like Exception, BaseException are NOT in the import map
    (available without explicit import). This is why we need BUILTIN_* sets.
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


def resolve_name(node: ast.expr, import_map: Mapping[LocalName, QualifiedName]) -> QualifiedName:
    """Resolve exception type to fully qualified name."""
    if isinstance(node, ast.Name):
        name = node.id
        # Check import map first
        if name in import_map:
            return import_map[name]
        # Assume builtins for known exception names not explicitly imported
        if name in BUILTIN_BROAD_EXCEPTIONS | BUILTIN_CANCELLED_ERROR | BUILTIN_GENERATOR_EXIT:
            return f'builtins.{name}'
        return name  # Local definition or unknown

    elif isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            base = import_map.get(node.value.id, node.value.id)
            return f'{base}.{node.attr}'
        # Nested attributes
        base = resolve_name(node.value, import_map)
        return f'{base}.{node.attr}'

    return ''


def find_test_references(test_file_path: Path) -> Mapping[ViolationKind, TestReference]:
    """Parse test file to find correct pattern examples.

    Looks for functions named like:
    - exc001_correct() -> EXC001 bare-except
    - exc002_correct() -> EXC002 swallowed-exception

    Returns mapping from violation kind to test reference (function name + line).
    """
    if not test_file_path.exists():
        return {}

    try:
        source = test_file_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(test_file_path))
    except (OSError, SyntaxError):
        return {}

    references: dict[ViolationKind, TestReference] = {}
    code_to_kind: dict[ErrorCode, ViolationKind] = {code: kind for kind, code in ERROR_CODES.items()}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Match pattern: excNNN_correct
            if node.name.startswith('exc') and '_correct' in node.name:
                # Extract code: exc001_correct -> 001
                parts = node.name.split('_')
                if parts and parts[0].startswith('exc') and len(parts[0]) == 6:
                    code_num = parts[0][3:]  # '001'
                    code = f'EXC{code_num}'

                    if code in code_to_kind:
                        kind = code_to_kind[code]
                        references[kind] = TestReference(
                            kind=kind,
                            function_name=node.name,
                            line=node.lineno,
                        )

    return references


def get_git_ignored_files(file_paths: Sequence[Path], directory: Path) -> Set[Path]:
    """Use git check-ignore to identify ignored files."""
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
            check=False,
        )
        if result.returncode == 128:
            return set()
        return {Path(line) for line in result.stdout.splitlines() if line}
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return set()


# -- Output Formatting --------------------------------------------------------


def format_violation(v: Violation) -> str:
    """Format a violation for display."""
    code = ERROR_CODES[v.kind]
    message = VIOLATION_MESSAGES[v.kind]
    fix = FIX_SUGGESTIONS[v.kind]

    return (
        f'{v.filepath}:{v.line}:{v.column}: error: {code} {message}\n'
        f'    {v.source_line}\n'
        f'    Fix: {fix}\n'
        f'    Silence: {DIRECTIVE_PREFIX} {v.kind}'
    )


if __name__ == '__main__':
    sys.exit(main())
