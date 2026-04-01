#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Enforce uv run shebangs and PEP 723 dependency formatting.

Checks shebangs and inline script metadata for common misconfigurations.
Only flags files that HAVE shebangs (SHB001/SHB002) or PEP 723 blocks (SHB003).

Rules:
    SHB001 bare-python-shebang    Shebang uses python directly instead of uv run
    SHB002 missing-script-flag    Shebang has uv run with PEP 723 metadata but no --script
    SHB003 deps-format            PEP 723 dependencies not in canonical format
                                  (one-per-line, trailing comma, alphabetically sorted)

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
import dataclasses
import re
import sys
from collections.abc import Sequence, Set
from pathlib import Path

from _config import find_config, find_python_files, get_per_file_ignored_codes, load_per_file_ignores

TOOL_NAME = 'shebang-check'

# Package name extraction from a dependency string (strips version specifiers, extras, markers).
PACKAGE_NAME_RE = re.compile(r'([A-Za-z0-9][-A-Za-z0-9_.]*)')

# Single-line non-empty dependencies: # dependencies = ["something", ...]
SINGLE_LINE_DEPS_RE = re.compile(r'^#\s*dependencies\s*=\s*\[.+\]\s*$')

# Dependency line inside a multi-line block: #   "package>=1.0",
DEP_LINE_RE = re.compile(r'^#\s+"([^"]+)"\s*,?\s*$')


@dataclasses.dataclass(frozen=True)
class Violation:
    """A detected shebang or PEP 723 violation."""

    code: str
    path: Path
    line: int
    source_line: str
    message: str
    fix: str


# -- Main Entry Point ---------------------------------------------------------


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
    total_violations = 0
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

        try:
            content = filepath.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue

        violations = check_file(filepath, content)

        # Filter by per-file-ignores
        for v in violations:
            if v.code not in per_file_codes:
                print_violation(v)
                total_violations += 1

    if total_violations:
        print(f'\nFound {total_violations} shebang violation(s).')
    return 1 if total_violations else 0


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


# -- File Checking ------------------------------------------------------------


def check_file(path: Path, content: str) -> Sequence[Violation]:
    """Check a single file for shebang and PEP 723 violations.

    SHB001/SHB002 are gated on shebang presence.
    SHB003 runs on any file with a PEP 723 block.
    """
    first_line = content.split('\n', 1)[0]
    has_shebang = first_line.startswith('#!')
    violations: list[Violation] = []

    if has_shebang:
        # SHB001: bare python shebang
        if 'python' in first_line and 'uv run' not in first_line:
            violations.append(
                Violation(
                    code='SHB001',
                    path=path,
                    line=1,
                    source_line=first_line,
                    message='shebang uses python directly, should use uv run',
                    fix="Use '#!/usr/bin/env -S uv run --script' or similar",
                )
            )

        # SHB002: uv run with PEP 723 metadata but missing --script
        if 'uv run' in first_line and '--script' not in first_line:
            if extract_pep723_block(content) is not None:
                violations.append(
                    Violation(
                        code='SHB002',
                        path=path,
                        line=1,
                        source_line=first_line,
                        message='shebang has `uv run` with PEP 723 metadata but missing `--script` flag',
                        fix='Add `--script` to the shebang',
                    )
                )

    # SHB003: dependency formatting (not gated on shebang)
    block = extract_pep723_block(content)
    if block is not None:
        dep_violation = check_deps_format(path, block)
        if dep_violation is not None:
            violations.append(dep_violation)

    return violations


def print_violation(v: Violation) -> None:
    """Print a violation in the standard linter output format."""
    print(f'{v.path}:{v.line}: {v.code} {v.message}')
    print(f'    {v.source_line}')
    print(f'    Fix: {v.fix}')


# -- PEP 723 Block Extraction ------------------------------------------------


def extract_pep723_block(content: str) -> Sequence[tuple[int, str]] | None:
    """Extract PEP 723 script metadata block lines with line numbers.

    Returns (1-indexed line number, raw line text) pairs for all lines
    between ``# /// script`` (exclusive) and ``# ///`` (exclusive).
    Returns None if no valid block exists.
    """
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip() == '# /// script':
            block_lines: list[tuple[int, str]] = []
            for j in range(i + 1, len(lines)):
                stripped = lines[j].strip()
                if stripped == '# ///':
                    return block_lines
                if stripped == '' or stripped.startswith('#'):
                    block_lines.append((j + 1, lines[j]))  # 1-indexed
                    continue
                break  # Non-comment, non-empty line: block is unclosed/invalid
    return None


# -- SHB003: Dependency Format Checking ---------------------------------------


def dep_sort_key(dep: str) -> str:
    """Extract lowercase package name from a dependency string for sorting."""
    m = PACKAGE_NAME_RE.match(dep.strip())
    return m.group(1).lower() if m else dep.lower()


def check_deps_format(path: Path, block_lines: Sequence[tuple[int, str]]) -> Violation | None:
    """Check SHB003: dependency formatting within a PEP 723 block.

    Checks three sub-concerns (reports the most fundamental violation):
    1. One dep per line (not single-line array)
    2. Trailing comma on last dep
    3. Alphabetical sort order (case-insensitive, by package name)
    """
    deps_start_idx: int | None = None
    for idx, (line_num, raw_line) in enumerate(block_lines):
        stripped = raw_line.strip()

        # Single-line deps: # dependencies = ["pydantic", "cc_lib"]
        if SINGLE_LINE_DEPS_RE.match(stripped):
            if re.match(r'^#\s*dependencies\s*=\s*\[\s*\]\s*$', stripped):
                return None  # Empty deps, fine
            return Violation(
                code='SHB003',
                path=path,
                line=line_num,
                source_line=raw_line.rstrip(),
                message='PEP 723 dependencies should use one-per-line format',
                fix='Expand to multi-line with one dep per line, trailing commas, sorted alphabetically',
            )

        # Multi-line deps start: # dependencies = [
        if re.match(r'^#\s*dependencies\s*=\s*\[\s*$', stripped):
            deps_start_idx = idx
            break

    if deps_start_idx is None:
        return None  # No dependencies key found

    # Collect deps from multi-line block
    dep_strings: list[str] = []
    dep_lines: list[tuple[int, str]] = []
    deps_end_idx: int | None = None

    for idx in range(deps_start_idx + 1, len(block_lines)):
        line_num, raw_line = block_lines[idx]
        stripped = raw_line.strip()

        if re.match(r'^#\s*\]\s*$', stripped):
            deps_end_idx = idx
            break

        m = DEP_LINE_RE.match(stripped)
        if m:
            dep_strings.append(m.group(1))
            dep_lines.append((line_num, raw_line))

    if deps_end_idx is None or not dep_strings:
        return None  # Malformed or empty multi-line block

    # Check trailing comma on last dep line
    last_dep_stripped = dep_lines[-1][1].strip()
    if not last_dep_stripped.rstrip().endswith(','):
        return Violation(
            code='SHB003',
            path=path,
            line=dep_lines[-1][0],
            source_line=dep_lines[-1][1].rstrip(),
            message='PEP 723 dependency missing trailing comma',
            fix='Add trailing comma after the last dependency',
        )

    # Check sort order
    sorted_deps = sorted(dep_strings, key=dep_sort_key)
    if dep_strings != sorted_deps:
        for i, (actual, expected) in enumerate(zip(dep_strings, sorted_deps)):
            if actual != expected:
                return Violation(
                    code='SHB003',
                    path=path,
                    line=dep_lines[i][0],
                    source_line=dep_lines[i][1].rstrip(),
                    message=f'PEP 723 dependencies not sorted alphabetically (expected {expected!r} here)',
                    fix='Sort dependencies alphabetically by package name',
                )

    return None


if __name__ == '__main__':
    sys.exit(main())
