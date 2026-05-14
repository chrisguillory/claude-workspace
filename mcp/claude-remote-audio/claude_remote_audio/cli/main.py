from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.types import OutputFormat
from claude_remote_bash.exceptions import RemoteBashError

from claude_remote_audio.cli import completion
from claude_remote_audio.orchestrator import ApplyError, ApplyResult, apply

__all__ = [
    'main',
]

logger = logging.getLogger(__name__)

app = create_app(help='Multi-Mac audio topology orchestration over roc-vad + roc-toolkit.')
add_completion_command(app)
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)

# Must come AFTER create_app — that's what calls typer's completion_init() and
# registers the default ZshComplete. Our override has to land last to win.
completion.register_zsh_template()


def main() -> None:
    """Entry point for claude-remote-audio CLI."""
    run_app(app)


@app.command(name='apply')
@error_boundary
def apply_cmd(
    target: Annotated[
        str,
        typer.Option(
            '--target',
            '-t',
            help='Host selector — alias, comma-list, group name, or ip:port',
            autocompletion=completion.complete_target,
        ),
    ],
    hub: Annotated[
        str | None,
        typer.Option(
            '--hub',
            help='Hub host alias (default: the local machine)',
            autocompletion=completion.complete_hub,
        ),
    ] = None,
    input_device: Annotated[
        str | None,
        typer.Option(
            '--input',
            '-i',
            help='Hub Core Audio input device (mic) — opt in to manage roc-send',
            autocompletion=completion.complete_input_devices,
        ),
    ] = None,
    output_device: Annotated[
        str | None,
        typer.Option(
            '--output',
            '-o',
            help='Hub default output device — opt in to manage roc-recv',
            autocompletion=completion.complete_output_devices,
        ),
    ] = None,
    format: Annotated[OutputFormat, typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Resolve the target topology and converge each host toward the declared audio state.

    Locality: ``--hub`` defaults to the local machine (the discovered daemon whose
    advertised IPs overlap with a local interface). Run the command on the Mac you
    want to act as hub; pass ``--hub`` explicitly to override.

    With ``--format json``, emits a single ``ApplyResult`` JSON object on stdout —
    per-host outcomes (actions taken, success, error) captured inside. Exit code
    is ``0`` when every host succeeded, ``1`` otherwise.

    Tab completion for ``--hub`` / ``--target`` / ``--input`` / ``--output`` is
    automatic — first TAB per cold host pays a short dispatch, subsequent TABs
    within the cache window are instant.
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


@error_boundary.handler(ApplyError)
def _handle_apply_error(exc: ApplyError) -> None:
    """User-facing configuration / constraint violation — clean message, no traceback."""
    typer.echo(f'apply: {exc}', err=True)


@error_boundary.handler(RemoteBashError)
def _handle_dispatch_error(exc: RemoteBashError) -> None:
    """Fallback: dispatch-layer failure that escaped the orchestrator's wrapping — print cleanly."""
    typer.echo(f'dispatch: {exc}', err=True)


def _print_text(result: ApplyResult) -> None:
    """Render per-host apply outcomes as human-readable lines + a one-line summary."""
    for host in result.hosts:
        status = 'ok' if host.success else 'failed'
        typer.echo(f'[{status}] {host.host} ({host.role})')
        for action in host.actions:
            typer.echo(f'  - {action}')
        if host.error:
            typer.echo(f'  ERROR: {host.error}', err=True)

    summary = 'ok' if result.overall_success else 'failed'
    typer.echo(f'\napply: {summary}')
