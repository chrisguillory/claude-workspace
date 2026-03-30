#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Enforce uv run shebangs — reject bare python/python3 interpreters.

Checks line 1 of each Python file for shebangs that invoke python directly
instead of through uv run. Only flags files that HAVE shebangs — files
without shebangs are not entry points and are not checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _config import find_python_files

EXCLUDE_DIRS = {'.venv', '__pycache__', '.git', '.mypy_cache', '.ruff_cache', 'node_modules'}


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    violations = 0
    for path in find_python_files(root, EXCLUDE_DIRS):
        try:
            first_line = path.read_text(encoding='utf-8').split('\n', 1)[0]
        except (OSError, UnicodeDecodeError):
            continue
        if not first_line.startswith('#!'):
            continue
        if 'python' in first_line and 'uv run' not in first_line:
            print(f'{path}:1: shebang uses python directly, should use uv run')
            print(f'    {first_line}')
            violations += 1
    if violations:
        print(f'\nFound {violations} shebang violation(s).')
    return 1 if violations else 0


if __name__ == '__main__':
    sys.exit(main())
