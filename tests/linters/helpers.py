"""Shared subprocess driver and AST utilities for the linter test suites."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

__all__ = [
    'LineRange',
    'get_class_ranges',
    'get_def_ranges',
    'run_linter',
]


class LineRange(NamedTuple):
    """Inclusive (start, end) line range for an AST node."""

    start: int
    end: int


def get_class_ranges(filepath: Path) -> Mapping[str, LineRange]:
    """Return {class_name: LineRange} for every ClassDef in the file.

    Uses ast.walk, so classes nested inside compound statements (try/if/with)
    are included. Raises ValueError on duplicate names.
    """
    ranges: dict[str, LineRange] = {}
    for node in ast.walk(ast.parse(filepath.read_text(encoding='utf-8'))):
        if isinstance(node, ast.ClassDef):
            # end_lineno is always set for nodes parsed from source.
            assert node.end_lineno is not None
            if node.name in ranges:
                prior = ranges[node.name]
                raise ValueError(
                    f'duplicate class name {node.name!r} in {filepath}: '
                    f'lines {prior.start}-{prior.end} and {node.lineno}-{node.end_lineno}',
                )
            ranges[node.name] = LineRange(node.lineno, node.end_lineno)
    return ranges


def get_def_ranges(filepath: Path) -> Mapping[str, LineRange]:
    """Return {function_name: LineRange} for every (Async)FunctionDef in the file.

    Uses ast.walk, so functions nested inside compound statements (try/if/with)
    or other functions are included. Raises ValueError on duplicate names.
    """
    ranges: dict[str, LineRange] = {}
    for node in ast.walk(ast.parse(filepath.read_text(encoding='utf-8'))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # end_lineno is always set for nodes parsed from source.
            assert node.end_lineno is not None
            if node.name in ranges:
                prior = ranges[node.name]
                raise ValueError(
                    f'duplicate function name {node.name!r} in {filepath}: '
                    f'lines {prior.start}-{prior.end} and {node.lineno}-{node.end_lineno}',
                )
            ranges[node.name] = LineRange(node.lineno, node.end_lineno)
    return ranges


def run_linter(test_file: Path, linter: Path) -> str:
    """Run the linter and return combined stdout+stderr."""
    result = subprocess.run(
        [sys.executable, str(linter), '--no-skip-file', '--no-config', str(test_file)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.stdout + result.stderr
