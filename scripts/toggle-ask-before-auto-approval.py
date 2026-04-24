#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc_lib",
#     "typer>=0.9.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///

"""Toggle the ask-before-auto-approval hook's per-session marker.

The companion hook (`hooks/ask-before-auto-approval.py`) emits
``permissionDecision: "ask"`` for write-class tools under auto mode,
polyfilling the `permissions.ask` bug (#42797). The hook checks for
a per-session marker file and short-circuits (exit 0) when present —
this CLI toggles that marker.

Subcommands:
    status  (default)  Report gate state. With --session, target one
                       session; without, list all active markers.
    off                Touch marker → hook short-circuits for the session
    on                 Remove marker → hook re-engages

All mutation subcommands accept ``--session <uuid>`` (or ``-s``) to
target a specific session from an external terminal. Without it, the
command uses ``$CLAUDE_CODE_SESSION_ID`` from the current environment.

User-only enforcement:
    Sub-agent invocations of ``off``/``on`` are refused — detected via
    the ``CLAUDE_CODE_AGENT_ID`` env var that ``hooks/inject-agent-id.py``
    injects into every sub-agent Bash command. Main-thread invocations
    have no ``CLAUDE_CODE_AGENT_ID`` and proceed.

    ``status`` is read-only and is allowed from any caller.

    For main-agent tool-call invocations, add a user-settings rule:
        "permissions": {"deny": ["Bash(ask-before-auto-approval:*)"]}
    User ``!``-prefix invocations from the REPL bypass the tool pipeline
    (shell pass-through), preserving the fast path.

Install (enables tab completion):
    scripts/toggle-ask-before-auto-approval.py install
    # Restart shell, then:
    toggle-ask-before-auto-approval <TAB>

See the PR #80 Claude Code Plan gist for the ``Ma_``/``FJH``
permission-pipeline background.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from cc_lib.cli import add_install_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

GATE_DIR = Path.home() / '.claude-workspace' / 'ask-before-auto-approval'

app = create_app(help="Toggle the ask-before-auto-approval hook's per-session marker.")
add_install_command(app, script_path=__file__)
error_boundary = ErrorBoundary(exit_code=1)


SessionOption = Annotated[
    str | None,
    typer.Option('--session', '-s', help='Session UUID (defaults to $CLAUDE_CODE_SESSION_ID).'),
]


@app.command()
@error_boundary
def status(session: SessionOption = None) -> None:
    """Report gate state. With --session, target one session; without, list all active markers."""
    sid = session or os.environ.get('CLAUDE_CODE_SESSION_ID')
    if sid:
        marker = GATE_DIR / f'disabled-{sid}'
        exists = marker.exists()
        typer.echo(f'Gate state: {"disabled" if exists else "enabled"}')
        typer.echo(f'Session: {sid}')
        typer.echo(f'Marker: {marker}  ({"exists" if exists else "absent"})')
        return
    # No session specified and not in a Claude Code session → list all markers
    markers = sorted(GATE_DIR.glob('disabled-*')) if GATE_DIR.exists() else []
    if not markers:
        typer.echo('Gate enabled for all tracked sessions (no markers present).')
        return
    typer.echo(f'Active disabled sessions ({len(markers)}):')
    for marker in markers:
        typer.echo(f'  {marker.name.removeprefix("disabled-")}')


@app.command()
@error_boundary
def off(session: SessionOption = None) -> None:
    """Disable the hook for a session (touch marker)."""
    _refuse_if_subagent('off')
    sid = _resolve_session(session)
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    (GATE_DIR / f'disabled-{sid}').touch()
    typer.echo(f'Gate disabled for session {sid}')


@app.command()
@error_boundary
def on(session: SessionOption = None) -> None:
    """Re-enable the hook for a session (remove marker)."""
    _refuse_if_subagent('on')
    sid = _resolve_session(session)
    (GATE_DIR / f'disabled-{sid}').unlink(missing_ok=True)
    typer.echo(f'Gate enabled for session {sid}')


# -- Private helpers ----------------------------------------------------------


def _refuse_if_subagent(action: str) -> None:
    """Raise ``SubagentRefused`` if called from a sub-agent.

    Sub-agents are identified by the ``CLAUDE_CODE_AGENT_ID`` env var,
    which ``hooks/inject-agent-id.py`` exports into every sub-agent
    Bash command. Main-thread commands don't have it.
    """
    agent_id = os.environ.get('CLAUDE_CODE_AGENT_ID')
    if agent_id:
        raise SubagentRefused(action=action, agent_id=agent_id)


def _resolve_session(session: str | None) -> str:
    """Resolve the target session UUID from --session or env, or raise."""
    sid = session or os.environ.get('CLAUDE_CODE_SESSION_ID')
    if not sid:
        raise NoSessionSpecified
    return sid


# -- Exceptions + handlers ----------------------------------------------------


class NoSessionSpecified(Exception):
    """Neither ``--session`` nor ``$CLAUDE_CODE_SESSION_ID`` is available."""


class SubagentRefused(Exception):
    """A sub-agent attempted a mutation (``off`` or ``on``)."""

    def __init__(self, *, action: str, agent_id: str) -> None:
        super().__init__(f'Refusing {action!r} from sub-agent (CLAUDE_CODE_AGENT_ID={agent_id}).')
        self.action = action
        self.agent_id = agent_id


@error_boundary.handler(NoSessionSpecified)
def _handle_no_session(exc: NoSessionSpecified) -> None:
    typer.echo(
        'No session specified. Pass --session <uuid> or run from a Claude Code session.',
        err=True,
    )
    sys.exit(2)


@error_boundary.handler(SubagentRefused)
def _handle_subagent_refused(exc: SubagentRefused) -> None:
    typer.echo(f'{exc} Only the user (or main thread) may toggle the gate.', err=True)
    sys.exit(3)


if __name__ == '__main__':
    run_app(app)
