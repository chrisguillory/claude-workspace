#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "packaging>=24.0",
# ]
# ///
"""Lint uv script shebangs and PEP 723 inline metadata.

Checks shebangs and inline script metadata for common misconfigurations.
Only flags files that HAVE shebangs (UVS001/UVS002) or PEP 723 blocks (UVS003/UVS004).

Rules:
    UVS001 bare-python-shebang    Shebang uses python directly instead of uv run
    UVS002 missing-script-flag    Shebang has uv run with PEP 723 metadata but no --script
    UVS003 deps-format            PEP 723 dependencies not in canonical format
                                  (one-per-line, trailing comma, alphabetically sorted)
    UVS004 missing-requires-py    PEP 723 block with deps but missing requires-python >= 3.13

Escape hatches:
    Per-file-ignores in pyproject.toml:
        [tool.uv-script-linter.per-file-ignores]
        "tests/linters/edge_cases/**" = ["skip-file"]

Usage:
    ./linters/uv_script_linter.py                    # Check current directory
    ./linters/uv_script_linter.py <files>            # Check specific files
    ./linters/uv_script_linter.py src/               # Check specific directory

Exit codes:
    0 - No violations found
    1 - Violations found
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence, Set
from pathlib import Path

from _lib.config import find_config, find_python_files, get_per_file_ignored_codes, load_per_file_ignores
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

TOOL_NAME = 'uv-script-linter'


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
        print(f'\nFound {total_violations} violation(s).')
    return 1 if total_violations else 0


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Lint uv script shebangs and PEP 723 inline metadata.',
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


def check_file(path: Path, content: str) -> Sequence[Violation]:
    """Check a single file for shebang and PEP 723 violations.

    UVS001/UVS002 are gated on shebang presence.
    UVS003/UVS004 run on any file with a PEP 723 block.
    """
    first_line = content.split('\n', 1)[0]
    has_shebang = first_line.startswith('#!')
    block = Pep723Block.from_script(content)
    violations: list[Violation] = []

    if has_shebang:
        # UVS001: bare python shebang
        if 'python' in first_line and 'uv run' not in first_line:
            violations.append(
                Violation(
                    code='UVS001',
                    path=path,
                    line=1,
                    source_line=first_line,
                    message='shebang uses python directly, should use uv run',
                    fix="Use '#!/usr/bin/env -S uv run --script' or similar",
                )
            )

        # UVS002: uv run with PEP 723 metadata but missing --script
        if 'uv run' in first_line and '--script' not in first_line.split() and block is not None:
            violations.append(
                Violation(
                    code='UVS002',
                    path=path,
                    line=1,
                    source_line=first_line,
                    message='shebang has `uv run` with PEP 723 metadata but missing `--script` flag',
                    fix='Add `--script` to the shebang',
                )
            )

    # UVS003/UVS004: PEP 723 checks (not gated on shebang)
    if block is not None:
        dep_violation = block.check_deps_format(path)
        if dep_violation is not None:
            violations.append(dep_violation)

        py_violation = block.check_requires_python(path)
        if py_violation is not None:
            violations.append(py_violation)

    return violations


def print_violation(v: Violation) -> None:
    """Print a violation in the standard linter output format."""
    print(f'{v.path}:{v.line}: {v.code} {v.message}')
    print(f'    {v.source_line}')
    print(f'    Fix: {v.fix}')


@dataclasses.dataclass(frozen=True)
class Violation:
    """A detected shebang or PEP 723 violation."""

    code: str
    path: Path
    line: int
    source_line: str
    message: str
    fix: str


class Pep723Block:
    """Parsed PEP 723 script metadata block.

    Encapsulates block extraction (PEP 723 reference regex + tomllib),
    structured access to metadata fields, and all PEP 723 lint checks
    (UVS003 dep formatting, UVS004 requires-python).
    """

    # PEP 723 reference regex (from the PEP itself).
    BLOCK_RE = re.compile(r'(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$')

    # Package name extraction for sort-key comparison.
    PACKAGE_NAME_RE = re.compile(r'([A-Za-z0-9][-A-Za-z0-9_.]*)')

    # Probe versions below 3.13 for UVS004.
    # If any satisfy a specifier, it allows Python below the project minimum.
    PROBES_BELOW_313: Sequence[Version] = tuple(
        Version(f'3.{minor}.{micro}') for minor in range(13) for micro in (0, 5, 99)
    )

    def __init__(
        self,
        metadata: Mapping[str, object],
        raw_content: str,
        start_line: int,
        source_lines: Sequence[str],
    ) -> None:
        self.metadata = metadata
        self.raw_content = raw_content
        self.start_line = start_line
        self.source_lines = source_lines

    @classmethod
    def from_script(cls, content: str) -> Pep723Block | None:
        """Extract and parse PEP 723 block using the canonical regex + tomllib."""
        matches = [m for m in cls.BLOCK_RE.finditer(content) if m.group('type') == 'script']
        if len(matches) != 1:
            return None
        match = matches[0]
        raw_content = match.group('content')
        start_line = content[: match.start()].count('\n') + 1
        # Strip comment prefixes (PEP 723 reference algorithm)
        toml_str = ''.join(
            line[2:] if line.startswith('# ') else line[1:] for line in raw_content.splitlines(keepends=True)
        )
        try:
            metadata = tomllib.loads(toml_str)
        except tomllib.TOMLDecodeError:
            return None
        return cls(
            metadata=metadata,
            raw_content=raw_content,
            start_line=start_line,
            source_lines=content.split('\n'),
        )

    @property
    def dependencies(self) -> Sequence[str]:
        """Parsed dependency list from TOML metadata."""
        deps = self.metadata.get('dependencies', [])
        if not isinstance(deps, list):
            return []
        return deps

    @property
    def requires_python(self) -> str | None:
        """Raw requires-python specifier string."""
        value = self.metadata.get('requires-python')
        if not isinstance(value, str):
            return None
        return value

    # -- UVS003: Dependency Format Checking -----------------------------------

    def check_deps_format(self, path: Path) -> Violation | None:
        """Check UVS003: one-per-line, trailing comma, alphabetically sorted.

        Inspects raw source text because TOML parsing discards formatting.
        """
        if not self.dependencies:
            return None

        lines = self.source_lines
        for i in range(self.start_line, len(lines)):
            stripped = lines[i].strip()
            if stripped == '# ///':
                break

            # Single-line non-empty deps
            if re.match(r'^#\s*dependencies\s*=\s*\[.+\]', stripped):
                if re.match(r'^#\s*dependencies\s*=\s*\[\s*\]', stripped):
                    return None
                return Violation(
                    code='UVS003',
                    path=path,
                    line=i + 1,
                    source_line=lines[i].rstrip(),
                    message='PEP 723 dependencies should use one-per-line format',
                    fix='Expand to multi-line with one dep per line, trailing commas, sorted alphabetically',
                )

            # Multi-line start
            if re.match(r'^#\s*dependencies\s*=\s*\[$', stripped):
                return self._check_multiline_deps(path, lines, i)

        return None

    def _check_multiline_deps(self, path: Path, lines: Sequence[str], deps_start: int) -> Violation | None:
        """Check trailing comma and sort order on multi-line dependency block."""
        dep_entries: list[tuple[int, str, str]] = []  # (line_num, raw_line, dep_string)

        for i in range(deps_start + 1, len(lines)):
            stripped = lines[i].strip()
            if re.match(r'^#\s*\]', stripped):
                break
            m = re.match(r'^#\s+"([^"]+)"\s*,?\s*$', stripped)
            if m:
                dep_entries.append((i + 1, lines[i], m.group(1)))

        if not dep_entries:
            return None

        # Trailing comma
        if not dep_entries[-1][1].strip().endswith(','):
            return Violation(
                code='UVS003',
                path=path,
                line=dep_entries[-1][0],
                source_line=dep_entries[-1][1].rstrip(),
                message='PEP 723 dependency missing trailing comma',
                fix='Add trailing comma after the last dependency',
            )

        # Sort order
        dep_strings = [d[2] for d in dep_entries]
        sorted_deps = sorted(dep_strings, key=self._dep_sort_key)
        if dep_strings != sorted_deps:
            for (line_num, raw_line, actual), expected in zip(dep_entries, sorted_deps):
                if actual != expected:
                    return Violation(
                        code='UVS003',
                        path=path,
                        line=line_num,
                        source_line=raw_line.rstrip(),
                        message=f'PEP 723 dependencies not sorted alphabetically (expected {expected!r} here)',
                        fix='Sort dependencies alphabetically by package name',
                    )

        return None

    def _dep_sort_key(self, dep: str) -> str:
        """Extract lowercase package name from a dependency string for sorting."""
        m = self.PACKAGE_NAME_RE.match(dep.strip())
        return m.group(1).lower() if m else dep.lower()

    # -- UVS004: Requires-Python Check ----------------------------------------

    def check_requires_python(self, path: Path) -> Violation | None:
        """Check UVS004: requires-python >= 3.13 when dependencies exist."""
        if not self.dependencies:
            return None

        spec_str = self.requires_python

        if spec_str is None:
            line_num, source_line = self._find_line('dependencies')
            return Violation(
                code='UVS004',
                path=path,
                line=line_num,
                source_line=source_line,
                message='PEP 723 block with dependencies missing requires-python',
                fix='Add: # requires-python = ">=3.13"',
            )

        line_num, source_line = self._find_line('requires-python')

        try:
            if self._allows_python_below_313(spec_str):
                return Violation(
                    code='UVS004',
                    path=path,
                    line=line_num,
                    source_line=source_line,
                    message=f'requires-python = "{spec_str}" allows Python below 3.13',
                    fix='Update to: # requires-python = ">=3.13"',
                )
        except InvalidSpecifier:
            return Violation(
                code='UVS004',
                path=path,
                line=line_num,
                source_line=source_line,
                message=f'requires-python = "{spec_str}" is not a valid PEP 440 specifier',
                fix='Use a valid specifier, e.g.: # requires-python = ">=3.13"',
            )

        return None

    def _allows_python_below_313(self, spec_str: str) -> bool:
        """Check if a PEP 440 specifier allows any Python version below 3.13."""
        ss = SpecifierSet(spec_str)
        return any(probe in ss for probe in self.PROBES_BELOW_313)

    def _find_line(self, keyword: str) -> tuple[int, str]:
        """Find a source line containing keyword within the block for error reporting."""
        for i in range(self.start_line, len(self.source_lines)):
            line = self.source_lines[i]
            if line.strip() == '# ///':
                break
            if keyword in line:
                return (i + 1, line.rstrip())
        return (self.start_line, '(PEP 723 block)')


if __name__ == '__main__':
    sys.exit(main())
