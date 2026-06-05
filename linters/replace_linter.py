#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Steer Pydantic field updates from ``model_copy(update=...)`` to ``__replace__()``.

``X.model_copy(update={'field': value})`` types ``update`` as ``dict[str, Any]``,
so typos and wrong value types slip through. The repo's ``__replace__`` mypy
plugin (``plugins/pydantic_replace.py``) synthesizes a per-field signature, so
``X.__replace__(field=value)`` catches both at type-check time. This linter flags
the ``update=`` keyword form and points at the typed replacement.

Only the ``update=`` form is flagged: a bare ``model_copy()`` / ``model_copy(deep=True)``
is a plain clone with no field changes, which ``__replace__`` does not replace.

Rules:
    RPL001 model-copy-update  ``model_copy(update=...)`` — use ``__replace__()`` instead

Escape hatches (inline suppression):
    # replace_linter.py: skip-file
    # replace_linter.py: model-copy-update

Design Philosophy:
    - Error-only, no auto-fix: ``update`` dict keys carry no type info, so a
      mechanical rewrite to ``__replace__(**literal)`` can't be verified safe at
      lint time — the plugin does that once a human applies it.
    - Standalone: No external dependencies, works with any Python 3.13+ install.
    - Syntactic: Flags by method name + keyword, not type inference. ``model_copy``
      is Pydantic-specific, so a non-Pydantic ``.model_copy(update=...)`` is the only
      false positive — covered by the per-file-ignore and inline-suppression hatches.

Usage:
    ./linters/replace_linter.py                        # Check current directory
    ./linters/replace_linter.py <files>                # Check specific files

Exit codes:
    0 - No violations found
    1 - Violations found
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from _lib.config import find_config, find_python_files, get_per_file_ignored_codes, load_per_file_ignores

# -- Configuration ------------------------------------------------------------

# Directive prefix - uses script filename for discoverability
DIRECTIVE_PREFIX = '# replace_linter.py:'

# -- Data Types ---------------------------------------------------------------

# Violation kind - used as directive codes and internal identifiers
type ViolationKind = Literal[
    'model-copy-update',
    'unused-directive',
]

# Maps kind to error code for display
type ErrorCode = str

ERROR_CODES: Mapping[ViolationKind, ErrorCode] = {
    'model-copy-update': 'RPL001',
    'unused-directive': 'RPL002',
}

# Short descriptions for each violation
VIOLATION_MESSAGES: Mapping[ViolationKind, str] = {
    'model-copy-update': 'model_copy(update=...) — use the typed __replace__() instead',
    'unused-directive': 'Suppression directive does not match any violation',
}

FIX_SUGGESTIONS: Mapping[ViolationKind, str] = {
    'model-copy-update': 'Replace with model.__replace__(field=value) (type-checked by the pydantic_replace plugin)',
    'unused-directive': 'Remove the stale suppression directive',
}


@dataclass(frozen=True)
class Violation:
    """A detected model_copy(update=...) violation."""

    filepath: Path
    line: int
    column: int
    kind: ViolationKind
    source_line: str


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
        elif path.is_file() and path.suffix == '.py':
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
                result = find_config(filepath, 'replace-linter')
                if result is not None:
                    config_path, project_root = result
                else:
                    config_path = None
                    project_root = None

            if config_path is not None and project_root is not None:
                per_file_ignores = load_per_file_ignores('replace-linter', config_path)
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
        for v in sorted(all_violations, key=lambda x: (x.filepath, x.line, x.column)):
            print(format_violation(v))
            print()

        # Summary
        file_count = len({v.filepath for v in all_violations})
        print(f'Found {len(all_violations)} violation(s) in {file_count} file(s).')

        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Flag Pydantic model_copy(update=...) in favor of the typed __replace__().',
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
    """Check a single file for model_copy(update=...) calls."""
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

    # Find violations: model_copy calls carrying an update= keyword
    raw_violations: list[tuple[int, ViolationKind]] = []
    violations: list[Violation] = []

    for node in _iter_model_copy_update_calls(tree):
        lineno = node.lineno
        raw_violations.append((lineno, 'model-copy-update'))

        # Check for inline suppression
        if not _has_directive(source_lines, lineno, 'model-copy-update'):
            violations.append(
                Violation(
                    filepath=filepath,
                    line=lineno,
                    column=node.col_offset,
                    kind='model-copy-update',
                    source_line=_get_source_line(source_lines, lineno),
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
                ),
            ]
        return []

    if report_unused_directives:
        directives = collect_directives(source_lines)
        unused = find_unused_directives(directives, raw_violations, filepath, source_lines)
        violations.extend(unused)

    return violations


# -- AST Analysis -------------------------------------------------------------


def _iter_model_copy_update_calls(tree: ast.Module) -> Sequence[ast.Call]:
    """Yield every ``X.model_copy(...)`` call that passes an ``update=`` keyword.

    Matched purely on the attribute name and keyword: the receiver type is not
    inferred (``model_copy`` is Pydantic-specific, so the name carries the signal).
    A bare ``model_copy()`` or ``model_copy(deep=True)`` is not flagged — it is a
    plain clone with no field changes, which ``__replace__`` does not replace.
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'model_copy'
        and any(kw.arg == 'update' for kw in node.keywords)
    ]


# -- Directive Handling -------------------------------------------------------


def _get_source_line(source_lines: Sequence[str], lineno: int) -> str:
    """Get source line, stripped."""
    if 0 < lineno <= len(source_lines):
        return source_lines[lineno - 1].strip()
    return ''


def _has_directive(source_lines: Sequence[str], lineno: int, kind: ViolationKind) -> bool:
    """Check if line (or nearby continuation lines) has suppression directive.

    Scans up to 4 lines forward from the violation to handle calls whose
    ``update=`` keyword spans multiple lines below the ``model_copy`` token.
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
