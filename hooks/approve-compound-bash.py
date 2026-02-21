#!/usr/bin/env -S uv run --quiet --no-project --script
"""PreToolUse hook: auto-approve compound Bash commands.

Works around Claude Code bugs where quoted strings and fd redirects in compound
commands trigger prompts even when each subcommand matches a Bash(prefix:*)
pattern. For example::

    git diff --cached --stat 2>&1 && echo "---" && git log --oneline -5 2>&1

This should auto-approve since all three subcommands (git diff, echo, git log)
match allowed prefixes, but Claude Code prompts due to the quoted "---" and 2>&1.

Uses bashlex to parse into subcommands. Only intervenes for compounds (&&, ||, ;,
|, newline). Simple commands are handled by built-in permissions.

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

import json
import re
import sys
from collections.abc import Sequence, Set
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

boundary = ErrorBoundary(exit_code=0)


@boundary.handler(Exception)
def _handle_error(exc: Exception) -> None:
    print(f'approve-compound-bash hook error: {exc!r}', file=sys.stderr)


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


# Extract the prefix from Claude Code's Bash permission pattern.
# Matches: "Bash(git log:*)" → "git log", "Bash(echo:*)" → "echo"
# Ignores: "WebSearch", "mcp__foo__bar", "Bash(git log:something)"
BASH_PREFIX_RE = re.compile(r'^Bash\((.+):\*\)$')


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


# Substitution patterns in parameter expansion values: ${x:-$(cmd)}, ${x:-`cmd`}
SUBSTITUTION_PATTERNS = ('$(', '`', '<(', '>(')

# Parameter transform operators: ${x@P} executes prompt expansion, ${x@E} interprets
# escapes, etc. Fail closed on all @ operators — they transform values in ways that
# may execute code or leak information.
TRANSFORM_OPERATOR_RE = re.compile(r'@[A-Za-z]')


def analyze_command(command: str) -> Sequence[str]:
    """Parse command into base command strings using bashlex AST.

    Returns one base_command per simple command. Compound operators (&&, ||, ;),
    pipes, and newlines produce separate entries.

    Raises ValueError on dangerous/unanalyzable constructs (file redirects,
    assignments, substitutions, control flow). ErrorBoundary handles exceptions.
    """
    parts = bashlex.parse(command)
    results: list[str] = []
    for part in parts:
        _collect_commands(command, part, results)
    return results


def _collect_commands(source: str, node: bashlex.ast.node, results: list[str]) -> None:
    """Recursively extract simple commands from the AST.

    Only decomposes node kinds we can fully analyze. Unknown kinds (if, for,
    while, until, function, case) raise ValueError so the hook falls through
    to built-in permissions rather than silently approving hidden commands.
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
        if hasattr(node, 'redirects') and node.redirects:
            for redir in node.redirects:
                if not isinstance(redir.output, int):
                    start, end = node.pos
                    raise ValueError(f'file redirect on compound: {source[start:end].strip()}')
        for child in node.list:
            _collect_commands(source, child, results)
    elif kind == 'reservedword':
        pass  # Syntax delimiters: (, ), {, }, do, done, then, fi, !, etc.
    else:
        start, end = node.pos
        raise ValueError(f'unanalyzable construct ({kind}): {source[start:end].strip()}')


def _analyze_command_node(source: str, node: bashlex.ast.node) -> str:
    """Inspect a single command node using an allowlist of safe AST patterns.

    Safe patterns: simple words, simple $VAR expansions, fd-to-fd redirects.
    Everything else (file redirects, assignments, heredocs, substitutions,
    unknown part kinds) raises ValueError.
    """
    start, end = node.pos
    words: list[str] = []

    for part in node.parts:
        pk = part.kind

        if pk == 'word':
            words.append(part.word)
            if hasattr(part, 'parts') and part.parts:
                if not _all_safe_word_children(part.parts):
                    raise ValueError(f'unsafe parameter expansion in: {part.word}')

        elif pk == 'redirect':
            if not isinstance(part.output, int):
                raise ValueError(f'file redirect in: {source[start:end].strip()}')

        elif pk == 'assignment':
            raise ValueError(f'inline assignment in: {source[start:end].strip()}')

        else:
            raise ValueError(f'unknown part kind ({pk}) in: {source[start:end].strip()}')

    return ' '.join(words) if words else ''


def _all_safe_word_children(children: Sequence[bashlex.ast.node]) -> bool:
    """True only if all children are tilde or simple parameter expansions ($VAR)."""
    for child in children:
        if child.kind == 'tilde':
            continue  # Tilde expansion (~, ~/path) is safe path resolution
        elif child.kind == 'parameter':
            val = child.value
            if any(pat in val for pat in SUBSTITUTION_PATTERNS):
                return False
            if TRANSFORM_OPERATOR_RE.search(val):
                return False
        else:
            # commandsubstitution, processsubstitution, or anything else
            return False
    return True


def matches_prefix(base_command: str, prefixes: Set[str]) -> bool:
    """Check if a base command matches any allowed Bash prefix."""
    if not base_command:
        return False
    return any(base_command == prefix or base_command.startswith(prefix + ' ') for prefix in prefixes)


@boundary
def main() -> None:
    hook_data = PreToolUseHookInput.model_validate_json(sys.stdin.read())

    if hook_data.tool_name != 'Bash':
        return

    command = hook_data.tool_input.get('command', '')
    if not isinstance(command, str) or not command:
        return

    subcommands = analyze_command(command)
    if not subcommands or len(subcommands) <= 1:
        return

    prefixes = load_bash_prefixes(hook_data.cwd)
    if not prefixes:
        return

    if all(matches_prefix(cmd, prefixes) for cmd in subcommands):
        decide('allow', f'all {len(subcommands)} subcommands match allowed Bash prefixes')


if __name__ == '__main__':
    main()
