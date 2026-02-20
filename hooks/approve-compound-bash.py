#!/usr/bin/env -S uv run --quiet --no-project --script
"""PreToolUse hook: auto-approve compound Bash commands.

Works around a Claude Code bug where quoted strings in compound commands
(e.g. ``echo "---" && git log``) trigger permission prompts even when each
subcommand individually matches an allowed ``Bash(prefix:*)`` pattern.

Simple (non-compound) commands are ignored — the built-in permission system
handles those correctly. This hook only intervenes when the command contains
``&&``, ``||``, ``;``, ``|``, or newline operators.

Uses bashlex (a port of bash's own parser) to produce a typed AST, enabling
structural detection of dangerous constructs (command substitution, process
substitution, file redirections) rather than fragile regex matching.

Reads Bash allow prefixes from the settings hierarchy:
  1. ``~/.claude/settings.json`` (user scope)
  2. ``<cwd>/.claude/settings.json`` (project shared)
  3. ``<cwd>/.claude/settings.local.json`` (project local)

Related Claude Code bugs:
  - Wildcard rejects quoted arguments:
    https://github.com/anthropics/claude-code/issues/23670
  - Pattern matching fails with quoted hyphen args:
    https://github.com/anthropics/claude-code/issues/16449
  - Permission bypass with ``&&`` chaining:
    https://github.com/anthropics/claude-code/issues/16180

Hook docs: https://code.claude.com/docs/en/hooks#pretooluse
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "bashlex",
#   "pydantic>=2.0.0",
#   "local_lib",
# ]
#
# [tool.uv.sources]
# local_lib = { path = "../local-lib/", editable = true }
# ///
from __future__ import annotations

__all__ = [
    'SubcommandInfo',
    'analyze_command',
    'load_bash_prefixes',
    'matches_prefix',
]

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import bashlex
import bashlex.ast
from local_lib.error_boundary import ErrorBoundary
from local_lib.schemas.hooks import (
    PreToolUseDecision,
    PreToolUseHookInput,
    PreToolUseHookOutput,
)

# --- Error boundary (process-level) ---
# On unexpected error: log to stderr, exit 0 (passthrough to built-in permissions)

boundary = ErrorBoundary(exit_code=0)


@boundary.handler(Exception)
def _handle_error(exc: Exception) -> None:
    print(f'approve-compound-bash hook error: {exc}', file=sys.stderr)


# --- Decision helper ---

type PermissionDecision = Literal['allow', 'deny', 'ask']


def decide(decision: PermissionDecision, reason: str) -> None:
    """Emit a permission decision to stdout."""
    output = PreToolUseHookOutput(
        hook_specific_output=PreToolUseDecision(
            permission_decision=decision,
            permission_decision_reason=reason,
        )
    )
    print(output.model_dump_json(by_alias=True, exclude_none=True))


# --- Settings loading ---

_BASH_PREFIX_RE = re.compile(r'^Bash\((.+):\*\)$')


def load_bash_prefixes(cwd: str) -> set[str]:
    """Load Bash allow prefixes from the settings hierarchy."""
    prefixes: set[str] = set()

    settings_paths = [
        Path.home() / '.claude' / 'settings.json',
        Path(cwd) / '.claude' / 'settings.json',
        Path(cwd) / '.claude' / 'settings.local.json',
    ]

    for path in settings_paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f'approve-compound-bash: skipping {path}: {exc}', file=sys.stderr)
            continue

        for entry in data.get('permissions', {}).get('allow', []):
            if m := _BASH_PREFIX_RE.match(entry):
                prefixes.add(m.group(1))

    return prefixes


# --- bashlex AST analysis ---

# Commands that wrap other commands — skip these + their args to find the real command
_WRAPPER_COMMANDS = frozenset({'timeout', 'time', 'nice', 'nohup'})
_SUBSTITUTION_PATTERNS = ('$(', '`', '<(', '>(')  # Patterns indicating code execution


@dataclass(frozen=True, slots=True)
class SubcommandInfo:
    """A simple command extracted from a compound bash expression."""

    text: str
    base_command: str
    is_dangerous: bool = False


def analyze_command(command: str) -> list[SubcommandInfo]:
    """Parse command into subcommands using bashlex AST.

    Returns one SubcommandInfo per simple command found. Compound operators
    (&&, ||, ;), pipes, and newlines all produce separate entries.
    Raises on malformed input — ErrorBoundary handles this.
    """
    parts = bashlex.parse(command)
    results: list[SubcommandInfo] = []
    for part in parts:
        _collect_commands(command, part, results)
    return results


def _collect_commands(source: str, node: bashlex.ast.node, results: list[SubcommandInfo]) -> None:
    """Recursively extract simple commands from the AST.

    Only decomposes node kinds we can fully analyze. Unknown kinds (if, for,
    while, until, function, case) are marked dangerous so the hook falls
    through to built-in permissions rather than silently approving hidden commands.
    """
    kind = node.kind

    if kind == 'command':
        results.append(_analyze_command_node(source, node))
    elif kind == 'list':
        for child in node.parts:
            if getattr(child, 'kind', '') != 'operator':
                _collect_commands(source, child, results)
    elif kind == 'pipeline':
        for child in node.parts:
            if getattr(child, 'kind', '') != 'pipe':
                _collect_commands(source, child, results)
    elif kind == 'compound':
        # Check for redirects on the compound itself: (cmds) > file
        if hasattr(node, 'redirects') and node.redirects:
            start, end = node.pos
            results.append(
                SubcommandInfo(
                    text=source[start:end].strip(),
                    base_command='',
                    is_dangerous=True,
                )
            )
        for child in node.list:
            _collect_commands(source, child, results)
    elif kind == 'reservedword':
        pass  # Syntax delimiters: (, ), {, }, do, done, then, fi, etc.
    else:
        # Unrecognized construct — mark as unanalyzable (fail closed)
        start, end = node.pos
        results.append(
            SubcommandInfo(
                text=source[start:end].strip(),
                base_command='',
                is_dangerous=True,
            )
        )


def _analyze_command_node(source: str, node: bashlex.ast.node) -> SubcommandInfo:
    """Inspect a single command node for its base command and danger signals."""
    start, end = node.pos
    text = source[start:end].strip()

    words: list[str] = []
    is_dangerous = False

    for part in node.parts:
        pk = part.kind

        if pk == 'word':
            words.append(part.word)
            if hasattr(part, 'parts') and part.parts:
                for child in part.parts:
                    if child.kind in ('commandsubstitution', 'processsubstitution'):
                        is_dangerous = True
                    elif child.kind == 'parameter':
                        # bashlex stores expansion text as opaque string,
                        # e.g. ${x:-$(cmd)} → parameter.value = 'x:-$(cmd)'
                        val = getattr(child, 'value', '')
                        if any(pat in val for pat in _SUBSTITUTION_PATTERNS):
                            is_dangerous = True

        elif pk == 'redirect':
            if part.type == '<<<':
                # Here-string: data input, but value may contain substitutions
                if hasattr(part.output, 'parts') and part.output.parts:
                    for child in part.output.parts:
                        if child.kind in ('commandsubstitution', 'processsubstitution'):
                            is_dangerous = True
            elif part.type == '<<':
                pass  # Heredoc: content not in AST, delimiter word is safe
            elif hasattr(part.output, 'kind'):
                # File redirect: output is a node (has .kind), not an int (fd-to-fd)
                is_dangerous = True

        elif pk == 'assignment':
            # Assignment values may contain substitutions: VAR=$(cmd) echo test
            if hasattr(part, 'parts') and part.parts:
                for child in part.parts:
                    if child.kind in ('commandsubstitution', 'processsubstitution'):
                        is_dangerous = True
                    elif child.kind == 'parameter':
                        val = getattr(child, 'value', '')
                        if any(pat in val for pat in _SUBSTITUTION_PATTERNS):
                            is_dangerous = True

    base_command = _resolve_base_command(words)

    return SubcommandInfo(
        text=text,
        base_command=base_command,
        is_dangerous=is_dangerous,
    )


def _resolve_base_command(words: list[str]) -> str:
    """Find the real command, skipping wrapper prefixes like timeout/nohup.

    bashlex parses ``timeout 5s git log`` as a single command with words
    ['timeout', '5s', 'git', 'log']. This skips known wrappers and their
    arguments to find the actual command being wrapped.
    """
    if not words:
        return ''

    i = 0
    while i < len(words) and words[i] in _WRAPPER_COMMANDS:
        cmd = words[i]
        i += 1  # skip the wrapper command itself

        if cmd == 'timeout' and i < len(words):
            # timeout takes a duration arg (e.g., '5s', '10', '30m')
            i += 1
        elif cmd == 'nice' and i < len(words) and words[i] == '-n':
            # nice -n <priority>
            i += 2

        # time, nohup take no required args before the command

    # Reconstruct the real command from remaining words
    if i < len(words):
        return ' '.join(words[i:])
    return ''  # No real command after stripping wrappers


# --- Prefix matching ---


def matches_prefix(info: SubcommandInfo, prefixes: set[str]) -> bool:
    """Check if a subcommand matches any allowed Bash prefix.

    Rejects subcommands containing dangerous constructs (command substitution,
    process substitution, file redirections) regardless of prefix match.
    """
    if info.is_dangerous:
        return False
    if not info.base_command:
        return False  # Assignment-only subcommands never match
    return any(info.base_command == prefix or info.base_command.startswith(prefix + ' ') for prefix in prefixes)


# --- Main ---


@boundary
def main() -> None:
    hook_data = PreToolUseHookInput.model_validate_json(sys.stdin.read())

    if hook_data.tool_name != 'Bash':
        return

    command = hook_data.tool_input.get('command', '')
    if not command:
        return

    subcommands = analyze_command(command)
    if len(subcommands) <= 1:
        return

    prefixes = load_bash_prefixes(hook_data.cwd)
    if not prefixes:
        return

    if all(matches_prefix(info, prefixes) for info in subcommands):
        decide('allow', f'all {len(subcommands)} subcommands match allowed Bash prefixes')


if __name__ == '__main__':
    main()
