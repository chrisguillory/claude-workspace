#!/usr/bin/env -S uv run --quiet --no-project --script
"""Analyze Claude Code session history to measure approve-compound-bash hook impact.

Extracts all Bash tool calls from JSONL session transcripts, classifies them
as simple or compound, and checks compound commands against configured Bash
prefixes to determine how many would be auto-approved by the hook.

Usage:
    uv run --no-project --script scripts/analyze-bash-history.py [--sessions DIR] [--cwd DIR] [--top N]

The --quiet flag suppresses uv's dependency resolution output to keep stdout
clean for the report. Progress messages go to stderr.
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "lazy-object-proxy",
#   "local_lib",
# ]
#
# [tool.uv.sources]
# local_lib = { path = "../local-lib/", editable = true }
# ///

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import lazy_object_proxy
import pydantic
from local_lib.error_boundary import ErrorBoundary
from local_lib.schemas.hooks import BashToolInput
from local_lib.types import JsonDatetime


class CommandRecord(pydantic.BaseModel):
    """A single Bash tool call extracted from a session transcript."""

    model_config = pydantic.ConfigDict(frozen=True)

    command: str  # "git log --oneline -5 && echo '---'"
    description: str | None  # "Show recent commits and separator"
    timestamp: JsonDatetime | None  # 2026-02-13T16:56:04.583Z
    session_file: Path  # ~/.claude/projects/-Users-chris-myproject/019c7783-8f01-7ef8-9aee-f8feb8d1940c.jsonl


@dataclass(frozen=True)
class AnalysisResult:
    """Aggregated analysis of all Bash commands."""

    total: int
    simple: int
    compound_approved: int
    compound_no_match: int
    compound_dangerous: int
    compound_parse_error: int
    approved_patterns: Sequence[tuple[str, int]]
    dangerous_examples: Sequence[str]
    parse_error_examples: Sequence[str]


boundary = ErrorBoundary(exit_code=1)


@boundary
def main() -> None:
    """Parse args, extract commands, analyze, and print report."""
    args = parse_args()

    prefixes: Set[str] = hook.load_bash_prefixes(args.cwd)
    if not prefixes:
        print('warning: no Bash prefixes found in settings hierarchy', file=sys.stderr)

    print(f'Scanning {args.sessions} ...', file=sys.stderr)
    session_files: list[Path] = list(SessionExtractor.find_files(args.sessions))
    print(f'Found {len(session_files)} session files', file=sys.stderr)

    commands: list[CommandRecord] = []
    for sf in session_files:
        commands.extend(SessionExtractor.extract_commands(sf))
    print(f'Extracted {len(commands)} Bash commands', file=sys.stderr)

    analyzer = CommandAnalyzer(hook_module=hook, prefixes=prefixes)
    result = analyzer.analyze(commands)
    ReportPrinter().display(result, prefixes, top_n=args.top)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Analyze Bash command history for hook impact.')
    parser.add_argument(
        '--sessions',
        type=Path,
        default=Path.home() / '.claude' / 'projects',
        help='Directory containing session JSONL files (default: ~/.claude/projects)',
    )
    parser.add_argument(
        '--cwd',
        type=str,
        default=str(Path.cwd()),
        help='Working directory for loading Bash prefixes (default: current directory)',
    )
    parser.add_argument(
        '--top',
        type=int,
        default=15,
        help='Number of top patterns to show (default: 15)',
    )
    return parser.parse_args()


class SessionModel(pydantic.BaseModel):
    """Base model for Claude Code session JSONL records — ignores unknown fields."""

    model_config = pydantic.ConfigDict(extra='ignore')


class ContentBlock(SessionModel):
    """A single content block from an assistant message."""

    type: str  # "tool_use", "text", "thinking"
    name: str | None = None  # "Bash", "Read", "Write" (only on tool_use)
    # Tool input payload (only on tool_use)
    input: Mapping[str, Any] | None = None  # strict_typing_linter.py: loose-typing


class AssistantMessage(SessionModel):
    """The message field of an assistant JSONL record."""

    content: Sequence[ContentBlock] | str = ()


class SessionRecord(SessionModel):
    """A single line from a Claude Code session JSONL file.

    Only validates the fields needed for Bash command extraction.
    See claude-session-mcp/src/schemas/session/models.py for the full schema.
    """

    type: Literal['assistant', 'user', 'summary', 'progress', 'result'] | str
    message: AssistantMessage | None = None
    timestamp: JsonDatetime | None = None  # 2026-02-13T16:56:04.583Z


class SessionExtractor:
    """Extract Bash commands from Claude Code session JSONL files."""

    @staticmethod
    def find_files(sessions_dir: Path) -> Iterator[Path]:
        """Walk the sessions directory for JSONL files."""
        if not sessions_dir.is_dir():
            return
        yield from sessions_dir.rglob('*.jsonl')

    @staticmethod
    def extract_commands(jsonl_path: Path) -> Iterator[CommandRecord]:
        """Extract Bash tool_use commands from a session JSONL file."""
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = SessionRecord.model_validate_json(line)

                if record.type != 'assistant' or record.message is None:
                    continue

                content = record.message.content
                if isinstance(content, str):
                    continue

                for block in content:
                    if block.type != 'tool_use' or block.name != 'Bash' or block.input is None:
                        continue
                    bash_input = BashToolInput.model_validate(block.input)
                    if bash_input.command:
                        yield CommandRecord(
                            command=bash_input.command,
                            description=bash_input.description,
                            timestamp=record.timestamp,
                            session_file=jsonl_path,
                        )


class CommandAnalyzer:
    """Analyze Bash commands against configured hook prefixes."""

    def __init__(self, hook_module: HookModule.Interface, prefixes: Set[str]) -> None:
        self._hook = hook_module
        self._prefixes = prefixes

    def analyze(self, commands: Sequence[CommandRecord], max_examples: int = 10) -> AnalysisResult:
        """Classify all commands and produce aggregated results."""
        total = 0
        simple = 0
        compound_approved = 0
        compound_no_match = 0
        compound_dangerous = 0
        compound_parse_error = 0
        approved_patterns: Counter[str] = Counter()
        dangerous_examples: list[str] = []
        parse_error_examples: list[str] = []

        for rec in commands:
            total += 1
            category = self._classify(rec.command)

            if category == 'simple':
                simple += 1
            elif category == 'approved':
                compound_approved += 1
                approved_patterns[rec.command[:80]] += 1
            elif category == 'no_match':
                compound_no_match += 1
            elif category == 'dangerous':
                compound_dangerous += 1
                if len(dangerous_examples) < max_examples:
                    dangerous_examples.append(rec.command[:120])
            elif category == 'parse_error':
                compound_parse_error += 1
                if len(parse_error_examples) < max_examples:
                    parse_error_examples.append(rec.command[:120])

        return AnalysisResult(
            total=total,
            simple=simple,
            compound_approved=compound_approved,
            compound_no_match=compound_no_match,
            compound_dangerous=compound_dangerous,
            compound_parse_error=compound_parse_error,
            approved_patterns=approved_patterns.most_common(),
            dangerous_examples=dangerous_examples,
            parse_error_examples=parse_error_examples,
        )

    def _classify(self, command: str) -> str:
        """Classify a single Bash command by category."""
        try:
            subcommands = self._hook.analyze_command(command)
        except self._hook.ApproveCompoundBashException:
            return 'dangerous'
        except self._hook.BashlexBoundaryException:
            return 'parse_error'

        if len(subcommands) <= 1:
            return 'simple'

        if all(self._hook.matches_prefix(cmd, self._prefixes) for cmd in subcommands):
            return 'approved'

        return 'no_match'


class ReportPrinter:
    """Format and print the analysis report to stdout."""

    def display(self, result: AnalysisResult, prefixes: Set[str], top_n: int = 15) -> None:
        """Print a formatted analysis report."""
        compound_total = (
            result.compound_approved
            + result.compound_no_match
            + result.compound_dangerous
            + result.compound_parse_error
        )

        print()
        print('Bash Command Impact Analysis')
        print('=' * 60)
        print(f'  Total Bash tool calls:         {result.total:>6,}')
        print(f'  Simple (single command):        {result.simple:>6,}  ({self._pct(result.simple, result.total)})')
        print(f'  Compound commands:              {compound_total:>6,}  ({self._pct(compound_total, result.total)})')
        print()

        if compound_total > 0:
            print('Compound Command Breakdown')
            print('=' * 60)
            print(
                f'  Would auto-approve (all match): {result.compound_approved:>6,}'
                f'  ({self._pct(result.compound_approved, compound_total)})'
            )
            print(
                f'  Would defer (prefix mismatch):  {result.compound_no_match:>6,}'
                f'  ({self._pct(result.compound_no_match, compound_total)})'
            )
            print(
                f'  Would defer (dangerous):        {result.compound_dangerous:>6,}'
                f'  ({self._pct(result.compound_dangerous, compound_total)})'
            )
            print(
                f'  Would defer (parse error):      {result.compound_parse_error:>6,}'
                f'  ({self._pct(result.compound_parse_error, compound_total)})'
            )
            print()

        if result.approved_patterns:
            shown = min(top_n, len(result.approved_patterns))
            print(f'Top {shown} Auto-Approved Patterns')
            print('=' * 60)
            for pattern, count in result.approved_patterns[:top_n]:
                print(f'  {count:>4,}x  {pattern}')
            print()

        if result.dangerous_examples:
            print(f'Dangerous Construct Examples (first {len(result.dangerous_examples)})')
            print('=' * 60)
            for ex in result.dangerous_examples:
                print(f'  {ex}')
            print()

        if result.parse_error_examples:
            print(f'Parse Error Examples (first {len(result.parse_error_examples)})')
            print('=' * 60)
            for ex in result.parse_error_examples:
                print(f'  {ex}')
            print()

        print(f'Configured Bash prefixes: {len(prefixes)}')

    def _pct(self, part: int, whole: int) -> str:
        """Format a percentage string."""
        if whole == 0:
            return '  0.0%'
        return f'{100 * part / whole:5.1f}%'


class HookModule:
    """Typed interface and lazy loader for the approve-compound-bash hook.

    Uses importlib because the hook filename has hyphens (approve-compound-bash.py),
    which are invalid in Python import statements.
    """

    class LoadError(Exception):
        """Hook module could not be loaded."""

    class Interface(Protocol):
        """Structural type for the hook module's public API."""

        ApproveCompoundBashException: type[Exception]
        BashlexBoundaryException: type[Exception]

        @staticmethod
        def analyze_command(command: str) -> Sequence[str]: ...

        @staticmethod
        def load_bash_prefixes(cwd: str) -> Set[str]: ...

        @staticmethod
        def matches_prefix(base_command: str, prefixes: Set[str]) -> bool: ...

    @staticmethod
    def load() -> HookModule.Interface:
        """Import approve-compound-bash.py via importlib."""
        hook_path = Path(__file__).parent.parent / 'hooks' / 'approve-compound-bash.py'
        if not hook_path.is_file():
            raise HookModule.LoadError(f'hook not found at {hook_path}')
        spec = importlib.util.spec_from_file_location('approve_compound_bash', hook_path)
        if spec is None or spec.loader is None:
            raise HookModule.LoadError(f'failed to create module spec for {hook_path}')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod  # type: ignore[return-value]


hook: HookModule.Interface = lazy_object_proxy.Proxy(HookModule.load)  # type: ignore[assignment]


# Boundary handlers — registered after main, dispatched when main raises


@boundary.handler(HookModule.LoadError)
def _handle_hook_load_error(exc: HookModule.LoadError) -> None:
    print(f'hook load error: {exc}', file=sys.stderr)


@boundary.handler(pydantic.ValidationError)
def _handle_validation_error(exc: pydantic.ValidationError) -> None:
    print(f'validation error: {exc.error_count()} error(s) in {exc.title}', file=sys.stderr)


@boundary.handler(json.JSONDecodeError)
def _handle_json_error(exc: json.JSONDecodeError) -> None:
    print(f'malformed JSONL: {exc.msg} at line {exc.lineno}, col {exc.colno}', file=sys.stderr)


@boundary.handler(FileNotFoundError)
def _handle_file_error(exc: FileNotFoundError) -> None:
    print(f'file not found: {exc.filename}', file=sys.stderr)


@boundary.handler(Exception)
def _handle_error(exc: Exception) -> None:
    print(f'error: {exc!r}', file=sys.stderr)


if __name__ == '__main__':
    main()
