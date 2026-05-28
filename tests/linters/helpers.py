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
    """Return {function_name: LineRange} for module-level + nested-function defs.

    Recurses into compound statements (if/try/with) and nested functions, but
    stops at ``ClassDef`` bodies — methods on test-fixture classes (``__init__``,
    ``__reduce__``, etc.) are not test targets; tests map violations to
    top-level ``excNNN_*`` functions. Raises ValueError on duplicate names
    among the included functions.
    """
    ranges: dict[str, LineRange] = {}
    _collect_function_ranges(ast.parse(filepath.read_text(encoding='utf-8')), ranges, filepath)
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


def _collect_function_ranges(node: ast.AST, ranges: dict[str, LineRange], filepath: Path) -> None:
    """Walk children, recording (Async)FunctionDef ranges but stopping at ClassDef bodies."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            continue
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            # end_lineno is always set for nodes parsed from source.
            assert child.end_lineno is not None
            if child.name in ranges:
                prior = ranges[child.name]
                raise ValueError(
                    f'duplicate function name {child.name!r} in {filepath}: '
                    f'lines {prior.start}-{prior.end} and {child.lineno}-{child.end_lineno}',
                )
            ranges[child.name] = LineRange(child.lineno, child.end_lineno)
        _collect_function_ranges(child, ranges, filepath)
