#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Re-export linter for Python modules.

Detects symbols in ``__all__`` that are imported from other modules rather than
defined locally.  These re-exports create an indirection layer that hides
the true source of a symbol, complicates IDE navigation, and inflates import
graphs.  Consumers should import directly from the defining module.

Rules:
    REX001 reexported-symbol  Symbol imported from another module and re-exported via __all__

Escape hatches (inline suppression):
    # reexport_linter.py: skip-file
    # reexport_linter.py: reexported-symbol

Design Philosophy:
    - Error-only, no auto-fix: Forces conscious decision at each occurrence
    - Standalone: No external dependencies, works with any Python 3.13+ install
    - Instructive: Fix message names the source module for direct import

Usage:
    ./linters/reexport_linter.py                       # Check current directory
    ./linters/reexport_linter.py <files>               # Check specific files
    ./linters/reexport_linter.py src/                   # Check specific directory

Exit codes:
    0 - No violations found
    1 - Violations found

Known Limitations (static analysis trade-offs):
    - Dynamic __all__: ``__all__ = get_exports()`` or ``__all__ += [...]`` is skipped
      (only static list literals are analysed).
    - Star imports: Files containing ``from mod import *`` are skipped entirely
      because the imported names are unknowable at analysis time.
    - Aliased re-exports: ``from mod import X as Y`` flags Y, not X.
    - Module imports: ``import os`` with ``'os'`` in __all__ is flagged.
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

# -- Configuration ------------------------------------------------------------

# Directive prefix - uses script filename for discoverability
DIRECTIVE_PREFIX = '# reexport_linter.py:'

# Test cases file for reference
TEST_CASES_FILE = 'tests/linters/edge_cases/reexport_linter_test_cases.py'

# -- Data Types ---------------------------------------------------------------

# Violation kind - used as directive codes and internal identifiers
type ViolationKind = Literal[
    'reexported-symbol',
    'unused-directive',
]

# Maps kind to error code for display
type ErrorCode = str

ERROR_CODES: Mapping[ViolationKind, ErrorCode] = {
    'reexported-symbol': 'REX001',
    'unused-directive': 'REX002',
}

# Short descriptions for each violation
VIOLATION_MESSAGES: Mapping[ViolationKind, str] = {
    'reexported-symbol': 'Symbol imported from another module and re-exported via __all__',
    'unused-directive': 'Suppression directive does not match any violation',
}


@dataclass(frozen=True)
class Violation:
    """A detected re-export violation."""

    filepath: Path
    line: int
    column: int
    kind: ViolationKind
    source_line: str
    # Dynamic fix message (includes source module name)
    fix_suggestion: str


@dataclass(frozen=True)
class DirectiveInstance:
    """A suppression directive found in source code."""

    line: int
    codes: Sequence[str]
    raw_text: str


@dataclass(frozen=True)
class ImportedName:
    """An imported symbol with its source module and import line."""

    source_module: str
    line: int


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
        elif path.is_file() and path.suffix == '.py':
            # Explicitly given files are checked even if not __init__.py
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
                result = find_config(filepath, 'reexport-linter')
                if result is not None:
                    config_path, project_root = result
                else:
                    config_path = None
                    project_root = None

            if config_path is not None and project_root is not None:
                per_file_ignores = load_per_file_ignores('reexport-linter', config_path)
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
        print(f'For correct patterns, see: {TEST_CASES_FILE}')

        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Check for re-exported symbols in Python modules.',
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
    """Check a single file for re-exported symbols in __all__."""
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

    # Check for star imports — skip file entirely (unknowable names)
    if _has_star_import(tree):
        return []

    # Extract __all__ names (static only)
    all_names = _extract_all_names(tree)
    if not all_names:
        return []

    # Build import map: local_name -> ImportedName
    import_map = _build_import_map(tree)

    # Build local definitions set
    local_defs = _build_local_defs(tree)

    # Find violations: name in __all__ AND in import map AND NOT in local defs
    raw_violations: list[tuple[int, ViolationKind, str, str]] = []
    violations: list[Violation] = []

    for name, all_line in all_names.items():
        if name in import_map and name not in local_defs:
            imported = import_map[name]
            raw_violations.append((all_line, 'reexported-symbol', name, imported.source_module))

            # Check for inline suppression
            if not _has_directive(source_lines, all_line, 'reexported-symbol'):
                fix = f"Import directly from '{imported.source_module}'"
                violations.append(
                    Violation(
                        filepath=filepath,
                        line=all_line,
                        column=0,
                        kind='reexported-symbol',
                        source_line=_get_source_line(source_lines, all_line),
                        fix_suggestion=fix,
                    ),
                )

    # Handle skip-file with report_unused_directives
    if has_skip_file and respect_skip_file:
        if report_unused_directives and not raw_violations:
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
                    fix_suggestion='Remove the stale suppression directive',
                ),
            ]
        return []

    if report_unused_directives:
        directives = collect_directives(source_lines)
        raw_tuples = [(line, kind) for line, kind, _, _ in raw_violations]
        unused = find_unused_directives(directives, raw_tuples, filepath, source_lines)
        violations.extend(unused)

    return violations


def find_python_files(
    root: Path,
    exclude_dirs: Set[str],
    respect_gitignore: bool = True,
) -> Sequence[Path]:
    """Find all Python files recursively under root."""
    all_files = sorted(path for path in root.rglob('*.py') if not any(part in exclude_dirs for part in path.parts))

    if respect_gitignore:
        ignored = get_git_ignored_files(all_files, root)
        all_files = [f for f in all_files if f not in ignored]

    return all_files


# -- AST Analysis -------------------------------------------------------------


def _has_star_import(tree: ast.Module) -> bool:
    """Check if file contains any star imports (from X import *)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == '*':
                    return True
    return False


def _extract_all_names(tree: ast.Module) -> Mapping[str, int]:
    """Extract names from a static __all__ assignment.

    Returns mapping from name to the line number of that name in __all__.
    Skips dynamic __all__ (augmented assignments, non-list values, etc.).
    """
    for node in ast.iter_child_nodes(tree):
        # __all__ = [...]
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    return _extract_names_from_list(node.value)

        # __all__: list[str] = [...]
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == '__all__' and node.value is not None:
                return _extract_names_from_list(node.value)

    return {}


def _extract_names_from_list(node: ast.expr) -> Mapping[str, int]:
    """Extract string constant names from a list/tuple literal."""
    if not isinstance(node, ast.List | ast.Tuple):
        return {}

    names: dict[str, int] = {}
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            names[elt.value] = elt.lineno
    return names


def _build_import_map(tree: ast.Module) -> Mapping[str, ImportedName]:
    """Build mapping from local name to import source.

    Handles:
        import mod           -> {'mod': ImportedName('mod', line)}
        from mod import X    -> {'X': ImportedName('mod', line)}
        from mod import X as Y -> {'Y': ImportedName('mod', line)}
    """
    import_map: dict[str, ImportedName] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue

        lineno = node.lineno

        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name
                import_map[local_name] = ImportedName(source_module=alias.name, line=lineno)

        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local_name = alias.asname or alias.name
                import_map[local_name] = ImportedName(source_module=node.module, line=lineno)

    return import_map


def _build_local_defs(tree: ast.Module) -> Set[str]:
    """Build set of names defined locally in the module (not imported).

    Includes: ClassDef, FunctionDef, AsyncFunctionDef, Assign, AnnAssign,
    TypeAlias (type X = ...).
    """
    local_defs: set[str] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            local_defs.add(node.name)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Skip __all__ itself
                    if target.id != '__all__':
                        local_defs.add(target.id)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id != '__all__':
                local_defs.add(node.target.id)

        elif isinstance(node, ast.TypeAlias):
            if isinstance(node.name, ast.Name):
                local_defs.add(node.name.id)

    return local_defs


# -- Directive Handling -------------------------------------------------------


def _get_source_line(source_lines: Sequence[str], lineno: int) -> str:
    """Get source line, stripped."""
    if 0 < lineno <= len(source_lines):
        return source_lines[lineno - 1].strip()
    return ''


def _has_directive(source_lines: Sequence[str], lineno: int, kind: ViolationKind) -> bool:
    """Check if line (or nearby continuation lines) has suppression directive.

    Scans up to 4 lines forward from the violation to handle cases where
    __all__ entries span multiple lines.
    """
    # unused-directive violations cannot be suppressed (like RUF100)
    if kind == 'unused-directive':
        return False

    prefix_lower = DIRECTIVE_PREFIX.lower()
    end = min(lineno + 4, len(source_lines))

    for check_lineno in range(lineno, end + 1):
        if check_lineno < 1:
            continue
        line = source_lines[check_lineno - 1].lower()

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

        # Strip rationale after separator
        for sep in (' \u2014 ', ' -- ', ' // ', ' #'):
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
                        fix_suggestion='Remove the stale suppression directive',
                    ),
                )
                break  # One unused code per directive is enough

    return unused


# -- Utility Functions --------------------------------------------------------


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

    return (
        f'{v.filepath}:{v.line}:{v.column}: error: {code} {message}\n'
        f'    {v.source_line}\n'
        f'    Fix: {v.fix_suggestion}\n'
        f'    Silence: {DIRECTIVE_PREFIX} {v.kind}'
    )


if __name__ == '__main__':
    sys.exit(main())
