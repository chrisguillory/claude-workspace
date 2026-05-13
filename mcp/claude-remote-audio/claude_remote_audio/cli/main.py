from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.types import OutputFormat
from claude_remote_bash.exceptions import RemoteBashError

from claude_remote_audio.orchestrator import ApplyError, ApplyResult, apply

__all__ = [
    'main',
]

logger = logging.getLogger(__name__)

app = create_app(help='Multi-Mac audio topology orchestration over roc-vad + roc-toolkit.')
add_completion_command(app)
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)


def main() -> None:
    """Entry point for claude-remote-audio CLI."""
    run_app(app)


@app.command(name='apply')
@error_boundary
def apply_cmd(
    target: Annotated[
        str,
        typer.Option('--target', '-t', help='Host selector — alias, comma-list, group name, or ip:port'),
    ],
    hub: Annotated[
        str,
        typer.Option('--hub', help='Which host (alias or ip:port) acts as the audio hub'),
    ],
    input_device: Annotated[
        str | None,
        typer.Option('--input', '-i', help='Hub Core Audio input device (mic) — opt in to manage roc-send'),
    ] = None,
    output_device: Annotated[
        str | None,
        typer.Option('--output', '-o', help='Hub default output device — opt in to manage roc-recv'),
    ] = None,
    format: Annotated[OutputFormat, typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Resolve the target topology and converge each host toward the declared audio state.

    With ``--format json``, emits a single ``ApplyResult`` JSON object on stdout —
    per-host outcomes (actions taken, success, error) captured inside. Exit code
    is ``0`` when every host succeeded, ``1`` otherwise.
    """
    result = asyncio.run(
        apply(
            target=target,
            hub=hub,
            input_device=input_device,
            output_device=output_device,
        )
    )

    if format == 'json':
        typer.echo(result.model_dump_json())
    else:
        _print_text(result)

    raise SystemExit(0 if result.overall_success else 1)


def _print_text(result: ApplyResult) -> None:
    """Render per-host outcomes as human-readable lines + a one-line summary."""
    for host in result.hosts:
        status = 'ok' if host.success else 'failed'
        typer.echo(f'[{status}] {host.host} ({host.role})')
        for action in host.actions:
            typer.echo(f'  - {action}')
        if host.error:
            typer.echo(f'  ERROR: {host.error}', err=True)

    summary = 'ok' if result.overall_success else 'failed'
    typer.echo(f'\napply: {summary}')


@error_boundary.handler(ApplyError)
def _handle_apply_error(exc: ApplyError) -> None:
    """User-facing configuration / constraint violation — clean message, no traceback."""
    typer.echo(f'apply: {exc}', err=True)


@error_boundary.handler(RemoteBashError)
def _handle_dispatch_error(exc: RemoteBashError) -> None:
    """Fallback: dispatch-layer failure that escaped the orchestrator's wrapping — print cleanly."""
    typer.echo(f'dispatch: {exc}', err=True)
