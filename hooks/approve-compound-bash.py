#!/usr/bin/env -S uv run --quiet --no-project --script
"""PreToolUse hook: auto-approve compound Bash commands.

Works around Claude Code bugs where quoted strings and fd redirects in compound
commands trigger prompts even when each subcommand matches a Bash(prefix:*)
pattern. For example::

    git diff --cached --stat 2>&1 && echo "---" && git log --oneline -5 2>&1

All three subcommands (git diff, echo, git log) match allowed prefixes, but
Claude Code prompts due to the quoted "---" and 2>&1 redirects.

Parses commands into a bashlex AST, walks it to extract base command strings
(rejecting unanalyzable constructs), then checks each against allowed Bash
prefixes. Only emits allow decisions for compounds (&&, ||, ;, |, newline).
Simple commands defer to built-in permissions.

Related bugs:
  - https://github.com/anthropics/claude-code/issues/23670
  - https://github.com/anthropics/claude-code/issues/16449
  - https://github.com/anthropics/claude-code/issues/16180

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

import json
import re
import sys
from collections.abc import Iterator, Sequence, Set
from pathlib import Path

import bashlex
from local_lib.error_boundary import ErrorBoundary
from local_lib.schemas.hooks import (
    PreToolUseDecision,
    PreToolUseHookInput,
    PreToolUseHookOutput,
)

boundary = ErrorBoundary(exit_code=0)


@boundary.handler(Exception)
def _handle_error(exc: Exception) -> None:
    print(f'approve-compound-bash hook error: {exc!r}', file=sys.stderr)


class ApproveCompoundBashException(Exception):
    """Base exception for this hook. Defers to Claude Code's native permissions."""


# Extract the prefix from Claude Code's Bash permission pattern.
# Matches: "Bash(git log:*)" → "git log", "Bash(echo:*)" → "echo"
# Ignores: "WebSearch", "mcp__foo__bar", "Bash(git log:something)"
BASH_PREFIX_RE = re.compile(r'^Bash\((.+):\*\)$')

# Simple shell variable name; rejects complex expansions (${!FOO}, ${x:-$(cmd)}, ${x@P}).
SIMPLE_VAR_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


@boundary
def main() -> None:
    """Read a PreToolUse hook event from stdin and auto-approve if safe.

    Emits an 'allow' decision only when every subcommand in a compound command
    matches an allowed Bash prefix. All other cases produce no output, deferring
    to Claude Code's native permission system.
    """
    hook_data = PreToolUseHookInput.model_validate_json(sys.stdin.read())

    if hook_data.tool_name != 'Bash':
        return  # Only handle Bash tool invocations

    command = hook_data.tool_input.get('command', '')
    if not isinstance(command, str) or not command:
        return  # No command to analyze

    subcommands = analyze_command(command)
    if len(subcommands) <= 1:
        return  # Simple commands are handled by built-in permissions

    prefixes = load_bash_prefixes(hook_data.cwd)
    if not prefixes:
        return  # No allowed prefixes configured

    if all(matches_prefix(cmd, prefixes) for cmd in subcommands):
        output = PreToolUseHookOutput(
            hook_specific_output=PreToolUseDecision(
                permission_decision='allow',
                permission_decision_reason=f'all {len(subcommands)} subcommands match allowed Bash prefixes',
            )
        )
        print(output.model_dump_json(by_alias=True, exclude_none=True))


def analyze_command(command: str) -> Sequence[str]:
    """Parse command into base command strings using bashlex AST.

    Returns one entry per simple command. Compound operators (&&, ||, ;),
    pipes, and newlines produce separate entries.

    Raises ApproveCompoundBashException on unanalyzable constructs (file redirects,
    assignments, substitutions, control flow).
    """
    parts = bashlex.parse(command)
    return [cmd for part in parts for cmd in _iter_subcommands(command, part)]


def load_bash_prefixes(cwd: str) -> Set[str]:
    """Load Bash allow prefixes from the settings hierarchy.

    Settings hierarchy: https://code.claude.com/docs/en/settings#what-uses-scopes
    """
    prefixes = set[str]()

    settings_paths = [
        Path.home() / '.claude' / 'settings.json',
        Path(cwd) / '.claude' / 'settings.json',
        Path(cwd) / '.claude' / 'settings.local.json',
    ]

    for path in settings_paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())

        for entry in data.get('permissions', {}).get('allow', []):
            if m := BASH_PREFIX_RE.match(entry):
                prefixes.add(m.group(1))

    return prefixes


def matches_prefix(base_command: str, prefixes: Set[str]) -> bool:
    """True if base_command equals a prefix or starts with 'prefix '."""
    if not base_command:
        return False
    return any(base_command == prefix or base_command.startswith(prefix + ' ') for prefix in prefixes)


def _iter_subcommands(source: str, node: bashlex.ast.node) -> Iterator[str]:
    """Walk the AST, yielding one base command string per simple command.

    Decomposes commands, lists, pipelines, and compounds. Unknown node kinds
    (if, for, while, until, function) raise ApproveCompoundBashException.
    """
    kind = node.kind

    if kind == 'command':
        yield _extract_base_command(source, node)
    elif kind == 'list':
        for child in node.parts:
            if child.kind != 'operator':
                yield from _iter_subcommands(source, child)
    elif kind == 'pipeline':
        for child in node.parts:
            if child.kind != 'pipe':
                yield from _iter_subcommands(source, child)
    elif kind == 'compound':
        for redir in node.redirects:
            if not isinstance(redir.output, int):
                start, end = node.pos
                raise ApproveCompoundBashException(f'file redirect on compound: {source[start:end].strip()}')
        for child in node.list:
            yield from _iter_subcommands(source, child)
    elif kind == 'reservedword':
        pass  # Syntax delimiters: (, ), {, }, do, done, then, fi, !, etc.
    else:
        start, end = node.pos
        raise ApproveCompoundBashException(f'unanalyzable construct ({kind}): {source[start:end].strip()}')


def _extract_base_command(source: str, node: bashlex.ast.node) -> str:
    """Extract the base command string from a command node.

    Allows simple words, $VAR expansions, and fd-to-fd redirects. Raises
    ApproveCompoundBashException on non-fd redirects, assignments, substitutions,
    or unknown part kinds.
    """
    start, end = node.pos
    words: list[str] = []

    for part in node.parts:
        if part.kind == 'word':
            words.append(part.word)
            if part.parts:
                _check_word_expansions(part.parts, part.word)

        elif part.kind == 'redirect':
            if not isinstance(part.output, int):
                raise ApproveCompoundBashException(f'file redirect in: {source[start:end].strip()}')

        elif part.kind == 'assignment':
            raise ApproveCompoundBashException(f'inline assignment in: {source[start:end].strip()}')

        else:
            raise ApproveCompoundBashException(f'unknown part kind ({part.kind}) in: {source[start:end].strip()}')

    return ' '.join(words)


def _check_word_expansions(children: Sequence[bashlex.ast.node], word: str) -> None:
    """Raise if any child is not a tilde or simple parameter expansion ($VAR)."""
    for child in children:
        if child.kind == 'tilde':
            continue  # Tilde expansion (~, ~/path) is safe path resolution
        if child.kind == 'parameter' and (not child.value or SIMPLE_VAR_RE.fullmatch(child.value)):
            continue  # Simple $VAR or empty value ($"..." locale quoting)
        raise ApproveCompoundBashException(f'unsafe expansion in: {word}')


if __name__ == '__main__':
    main()
