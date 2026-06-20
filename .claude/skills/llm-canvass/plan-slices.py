#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc-lib",
#     "typer>=0.12",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../../../cc-lib/", editable = true }
# ///
"""Plan slices for /llm-canvass.

Enumerates in-scope files, bin-packs them into balanced token slices, writes
the brief and slice files to the session scratchpad, and prints a summary
that Phase 2 of the skill consumes.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer
from cc_lib.error_boundary import ErrorBoundary

error_boundary = ErrorBoundary(exit_code=1)


class LLMCanvassError(Exception): ...


class NoSessionId(LLMCanvassError):
    """CLAUDE_CODE_SESSION_ID not set; not running inside a Claude Code session."""


class NoFilesMatched(LLMCanvassError):
    """`rg --files` returned no results under the given filters."""


class NoScope(LLMCanvassError):
    """Caller did not specify any scope — --type, --glob, and --all are all unset."""


class ConflictingScope(LLMCanvassError):
    """Caller passed --all alongside --type or --glob — pick one."""


@error_boundary
def main(
    task: Annotated[
        # strict_typing_linter.py: mutable-type — typer rejects Sequence[str] for variadic params; list required
        list[str],
        typer.Argument(help='Task description (multi-word; quote or pass as separate args).'),
    ],
    file_type: Annotated[
        str | None,
        typer.Option('--type', help='Restrict to one rg --type (e.g. "py", "ts"). Run `rg --type-list` for options.'),
    ] = None,
    glob: Annotated[
        str | None,
        typer.Option('--glob', help='Glob pattern passed to rg --glob (e.g. "**/*.py").'),
    ] = None,
    all_: Annotated[
        bool,
        typer.Option(
            '--all',
            help='Scan every file rg discovers (still respects .gitignore). Mutually exclusive with --type/--glob.',
        ),
    ] = False,
    agent: Annotated[
        str,
        typer.Option('--agent', help='Subagent type for fan-out.'),
    ] = 'general-purpose',
    background: Annotated[
        bool,
        typer.Option('--background', help='Launch subagents non-blocking instead of synchronously.'),
    ] = False,
    max_agents: Annotated[
        int,
        typer.Option('--max-agents', help='Cap on parallel agents.'),
    ] = 8,
    per_agent_tokens: Annotated[
        int,
        typer.Option('--per-agent-tokens', help='Source-token budget per agent.'),
    ] = 400_000,
) -> None:
    """Enumerate files, bin-pack into balanced slices, write brief + slices, print the plan."""
    if not (file_type or glob or all_):
        raise NoScope()
    if all_ and (file_type or glob):
        raise ConflictingScope()

    task_text = ' '.join(task)

    cmd = ['rg', '--files']
    if file_type:
        cmd.extend(['--type', file_type])
    if glob:
        cmd.extend(['--glob', glob])
    files = subprocess.check_output(cmd, text=True).splitlines()
    if not files:
        raise NoFilesMatched()

    sizes = {f: max(1, os.path.getsize(f) // 4) for f in files}
    total = sum(sizes.values())
    n_agents = max(1, min(max_agents, -(-total // per_agent_tokens)))

    # Greedy bin-pack: largest-first into smallest current bin.
    bins: list[list[str]] = [[] for _ in range(n_agents)]
    totals = [0] * n_agents
    for f in sorted(files, key=lambda x: -sizes[x]):
        i = totals.index(min(totals))
        bins[i].append(f)
        totals[i] += sizes[f]

    session_id = os.environ.get('CLAUDE_CODE_SESSION_ID')
    if not session_id:
        raise NoSessionId()
    invocation_id = f'{time.strftime("%Y%m%dT%H%M%S")}-{os.getpid()}'
    scratchpad = Path('/tmp') / 'llm-canvass' / session_id / invocation_id
    scratchpad.mkdir(parents=True, exist_ok=True)

    (scratchpad / 'brief.md').write_text(f'# Task\n\n{task_text}\n')
    for i, bucket in enumerate(bins, 1):
        (scratchpad / f'slice-{i}.txt').write_text('\n'.join(bucket))

    typer.echo(f'Plan for: {task_text[:80]}{"..." if len(task_text) > 80 else ""}')
    typer.echo(f'Files: {len(files)}   Tokens: {total:,}   Agents: {n_agents}')
    typer.echo(f'Agent type: {agent}')
    typer.echo(f'Background: {background}')
    typer.echo(f'Scratchpad: {scratchpad}')
    typer.echo('')
    typer.echo('Slices:')
    for i, (b, t) in enumerate(zip(bins, totals, strict=True), 1):
        typer.echo(f'  slice-{i}: {len(b)} files, ~{t:,} tokens')


@error_boundary.handler(NoSessionId)
def _handle_no_session_id(exc: NoSessionId) -> None:
    typer.secho('CLAUDE_CODE_SESSION_ID not set; run inside a Claude Code session.', err=True, fg=typer.colors.RED)


@error_boundary.handler(NoFilesMatched)
def _handle_no_files_matched(exc: NoFilesMatched) -> None:
    typer.secho('No files matched.', err=True, fg=typer.colors.RED)


@error_boundary.handler(NoScope)
def _handle_no_scope(exc: NoScope) -> None:
    typer.secho('Scope required: pass --type <name>, --glob <pattern>, or --all.', err=True, fg=typer.colors.RED)


@error_boundary.handler(ConflictingScope)
def _handle_conflicting_scope(exc: ConflictingScope) -> None:
    typer.secho('--all is mutually exclusive with --type / --glob.', err=True, fg=typer.colors.RED)


if __name__ == '__main__':
    typer.run(main)
