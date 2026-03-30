#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Enforce uv run shebangs -- reject bare python/python3 interpreters.

Checks line 1 of each Python file for shebangs that invoke python directly
instead of through uv run. Only flags files that HAVE shebangs -- files
without shebangs are not entry points and are not checked.

Rules:
    SHB001 bare-python-shebang  Shebang uses python directly instead of uv run

Escape hatches:
    Per-file-ignores in pyproject.toml:
        [tool.shebang-check.per-file-ignores]
        "tests/linters/edge_cases/**" = ["skip-file"]

Usage:
    ./linters/shebang_check.py                    # Check current directory
    ./linters/shebang_check.py <files>            # Check specific files
    ./linters/shebang_check.py src/               # Check specific directory

Exit codes:
    0 - No violations found
    1 - Violations found
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Set
from pathlib import Path

from _config import find_config, find_python_files, get_per_file_ignored_codes, load_per_file_ignores

TOOL_NAME = 'shebang-check'


def check_file(path: Path) -> bool:
    """Check a single file. Returns True if violation found."""
    try:
        first_line = path.read_text(encoding='utf-8').split('\n', 1)[0]
    except (OSError, UnicodeDecodeError):
        return False
    if not first_line.startswith('#!'):
        return False
    if 'python' in first_line and 'uv run' not in first_line:
        print(f'{path}:1: shebang uses python directly, should use uv run')
        print(f'    {first_line}')
        return True
    return False


def main() -> int:
    """CLI entry point: parse args, collect files, check, report."""
    args = parse_args()
    exclude_dirs = set(args.exclude) | {'.venv', 'venv', '__pycache__', '.git'}
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
    violations = 0
    for filepath in files:
        # Per-file-ignores from pyproject.toml
        per_file_codes: Set[str] = set()
        skip_file_via_config = False
        if not args.no_config:
            if explicit_config is not None:
                config_path = explicit_config
                project_root = explicit_config.parent
            else:
                result = find_config(filepath, TOOL_NAME)
                if result is not None:
                    config_path, project_root = result
                else:
                    config_path = None
                    project_root = None

            if config_path is not None and project_root is not None:
                per_file_ignores = load_per_file_ignores(TOOL_NAME, config_path)
                per_file_codes = get_per_file_ignored_codes(filepath, per_file_ignores, project_root)
                if 'skip-file' in per_file_codes:
                    skip_file_via_config = True

        if skip_file_via_config:
            continue

        violations += check_file(filepath)

    if violations:
        print(f'\nFound {violations} shebang violation(s).')
    return 1 if violations else 0


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Check that shebangs use uv run instead of bare python.',
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
        '--no-gitignore',
        action='store_true',
        help='Do not respect .gitignore when scanning directories',
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


if __name__ == '__main__':
    sys.exit(main())
