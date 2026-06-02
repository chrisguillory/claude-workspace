"""Shared subprocess driver and AST utilities for the linter test suites."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tokenize
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


def run_linter(test_file: Path, linter: Path, *, report_unused: bool = False) -> str:
    """Run the linter and return combined stdout+stderr.

    ``report_unused`` adds ``--report-unused-directives`` so the linter flags
    suppression directives that match no violation — every linter leaves that check
    off by default, so the suites must opt in to exercise it.
    """
    args = [sys.executable, str(linter), '--no-skip-file', '--no-config']
    if report_unused:
        args.append('--report-unused-directives')
    args.append(str(test_file))
    result = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
    return result.stdout + result.stderr


def parse_unused_directives(output: str) -> set[int]:
    """Return the source-line numbers flagged with an unused-directive violation.

    Captures the line from each linter's ``:<line>:`` prefix and matches the
    unused-directive message across all three formats: exception_safety
    (``EXC009`` + ``:line:col:``), strict_typing (quoted ``'<code>'``, no column,
    no numbered code), and reexport (``REX002``). The numbered-code prefix, the
    optional column, and the quoted code are absorbed by the wildcards; ordinary
    violation lines do not match.
    """
    pattern = re.compile(
        r':(\d+):(?:\d+:)? error:.*Suppression directive.*does not match any violation',
    )
    return {int(match.group(1)) for line in output.splitlines() if (match := pattern.search(line))}


def extract_directive_codes(linter_path: Path, dict_name: str) -> set[str]:
    """Return the string keys of a linter's ``dict_name`` mapping, read via AST.

    The linters can't be imported (uv-script shebang + ``from _lib …`` + pydantic),
    so the suppressible-code inventory is parsed straight from source. The table is
    declared ``dict_name: Mapping[...] = {...}`` — an ``AnnAssign`` — so an
    ``Assign``-only walk would miss it. Raises if the name is absent or not a dict
    literal, so a renamed table trips the completeness guard loudly.
    """
    tree = ast.parse(linter_path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        if isinstance(target, ast.Name) and target.id == dict_name and isinstance(value, ast.Dict):
            return {key.value for key in value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    raise ValueError(f'{dict_name} not found as a dict literal in {linter_path}')


def sole_directive_line(filepath: Path, line_range: LineRange, linter: Path) -> int:
    """Return the single ``linter`` directive-comment line within ``line_range``.

    Tokenizes so a ``#``-prefixed directive sitting *inside* a docstring (a STRING
    token, not a COMMENT) is never miscounted — mirroring the linters' own
    docstring-skip. Raises if the entity carries zero or several directives, so a
    fixture-authoring slip fails loudly instead of silently mapping the wrong line.
    """
    prefix = re.compile(rf'#\s*{re.escape(linter.name)}:')
    with filepath.open('rb') as handle:
        hits = [
            token.start[0]
            for token in tokenize.tokenize(handle.readline)
            if token.type == tokenize.COMMENT
            and line_range.start <= token.start[0] <= line_range.end
            and prefix.search(token.string)
        ]
    if len(hits) != 1:
        raise ValueError(
            f'{filepath}: expected exactly one {linter.name} directive in '
            f'lines {line_range.start}-{line_range.end}, found {hits}',
        )
    return hits[0]


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
