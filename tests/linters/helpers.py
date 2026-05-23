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
    """Return {class_name: LineRange} for every ClassDef in the file."""
    ranges: dict[str, LineRange] = {}
    for node in ast.walk(ast.parse(filepath.read_text(encoding='utf-8'))):
        if isinstance(node, ast.ClassDef):
            assert node.end_lineno is not None
            ranges[node.name] = LineRange(node.lineno, node.end_lineno)
    return ranges


def get_def_ranges(filepath: Path) -> Mapping[str, LineRange]:
    """Return {function_name: LineRange} for every (Async)FunctionDef in the file."""
    ranges: dict[str, LineRange] = {}
    for node in ast.walk(ast.parse(filepath.read_text(encoding='utf-8'))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            assert node.end_lineno is not None
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
