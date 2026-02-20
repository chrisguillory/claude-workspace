#!/usr/bin/env -S uv run --quiet --no-project --script
"""PreToolUse hook: auto-approve compound Bash commands.

Works around a Claude Code bug where quoted strings in compound commands
(e.g. ``echo "---" && git log``) trigger permission prompts even when each
subcommand individually matches an allowed ``Bash(prefix:*)`` pattern.

Simple (non-compound) commands are ignored — the built-in permission system
handles those correctly. This hook only intervenes when the command contains
``&&``, ``||``, ``;``, ``|``, or newline operators.

Uses bashlex (a port of bash's own parser) to produce a typed AST. Safety
is enforced via an **allowlist of safe AST patterns** — only simple words,
simple ``$VAR`` expansions, tilde expansion, and fd-to-fd redirects are
approved. Everything else (file redirects, assignments, heredocs,
substitutions, parameter transform operators, unknown constructs) triggers
passthrough to built-in permissions.

**Limitation — pipe composition:** Each subcommand in a pipe is validated
independently. Two individually-safe commands can be dangerous when composed
via pipe (e.g. ``cat .env | python3 -c "exec(sys.stdin.read())"``). Users
should audit prefix combinations — pipe compositions inherit the combined
power of all allowed prefixes.

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
from collections.abc import Sequence, Set
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
    print(f'approve-compound-bash hook error: {exc!r}', file=sys.stderr)


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


def load_bash_prefixes(cwd: str) -> Set[str]:
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
            if not isinstance(entry, str):
                continue
            if m := _BASH_PREFIX_RE.match(entry):
                prefixes.add(m.group(1))

    return prefixes


# --- bashlex AST analysis ---

# Substitution patterns in parameter expansion values: ${x:-$(cmd)}, ${x:-`cmd`}
_SUBSTITUTION_PATTERNS = ('$(', '`', '<(', '>(')

# Parameter transform operators: ${x@P} executes prompt expansion, ${x@E} interprets
# escapes, etc. Fail closed on all @ operators — they transform values in ways that
# may execute code or leak information.
_TRANSFORM_OPERATOR_RE = re.compile(r'@[A-Za-z]')


@dataclass(frozen=True, slots=True)
class SubcommandInfo:
    """A simple command extracted from a compound bash expression."""

    text: str
    base_command: str
    is_dangerous: bool = False


def analyze_command(command: str) -> Sequence[SubcommandInfo]:
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
        # Any non-fd-to-fd redirect on the compound itself is dangerous
        if hasattr(node, 'redirects') and node.redirects:
            for redir in node.redirects:
                if not isinstance(redir.output, int):
                    start, end = node.pos
                    results.append(
                        SubcommandInfo(
                            text=source[start:end].strip(),
                            base_command='',
                            is_dangerous=True,
                        )
                    )
                    break
        for child in node.list:
            _collect_commands(source, child, results)
    elif kind == 'reservedword':
        pass  # Syntax delimiters: (, ), {, }, do, done, then, fi, !, etc.
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
    """Inspect a single command node using an allowlist of safe AST patterns.

    Safe patterns: simple words, simple $VAR expansions, fd-to-fd redirects.
    Everything else (file redirects, assignments, heredocs, here-strings,
    substitutions, unknown part kinds) is marked dangerous.
    """
    start, end = node.pos
    text = source[start:end].strip()

    words: list[str] = []
    is_dangerous = False

    for part in node.parts:
        pk = part.kind

        if pk == 'word':
            words.append(part.word)
            # Words with children must have ONLY safe parameter expansions
            if hasattr(part, 'parts') and part.parts:
                if not _all_safe_word_children(part.parts):
                    is_dangerous = True

        elif pk == 'redirect':
            # Only fd-to-fd redirects (output is int, e.g. 2>&1) are safe
            if not isinstance(part.output, int):
                is_dangerous = True

        elif pk == 'assignment':
            # All inline env var assignments are dangerous — avoids incomplete
            # denylists of LD_PRELOAD, DYLD_INSERT_LIBRARIES, BASH_ENV, etc.
            is_dangerous = True

        else:
            # Unknown part kind — fail closed
            is_dangerous = True

    base_command = ' '.join(words) if words else ''

    return SubcommandInfo(
        text=text,
        base_command=base_command,
        is_dangerous=is_dangerous,
    )


def _all_safe_word_children(children: Sequence[bashlex.ast.node]) -> bool:
    """True only if all children are tilde or simple parameter expansions ($VAR)."""
    for child in children:
        if child.kind == 'tilde':
            continue  # Tilde expansion (~, ~/path) is safe path resolution
        elif child.kind == 'parameter':
            val = getattr(child, 'value', '')
            if any(pat in val for pat in _SUBSTITUTION_PATTERNS):
                return False
            if _TRANSFORM_OPERATOR_RE.search(val):
                return False
        else:
            # commandsubstitution, processsubstitution, or anything else
            return False
    return True


# --- Prefix matching ---


def matches_prefix(info: SubcommandInfo, prefixes: Set[str]) -> bool:
    """Check if a subcommand matches any allowed Bash prefix.

    Rejects subcommands containing dangerous constructs regardless of match.
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
    if not isinstance(command, str) or not command:
        return

    # Null bytes cause parsing divergence between bashlex and actual shell
    if '\x00' in command:
        return

    subcommands = analyze_command(command)
    if not subcommands or len(subcommands) <= 1:
        return

    prefixes = load_bash_prefixes(hook_data.cwd)
    if not prefixes:
        return

    if all(matches_prefix(info, prefixes) for info in subcommands):
        decide('allow', f'all {len(subcommands)} subcommands match allowed Bash prefixes')


if __name__ == '__main__':
    main()
