from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, TextIO

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary, render_recovery
from cc_lib.exceptions import ResolvableError
from cc_lib.types import OutputFormat
from claude_remote_bash.exceptions import RemoteBashError
from claude_remote_bash.selector import SelectorError

from claude_remote_audio import paths
from claude_remote_audio.cli import completion
from claude_remote_audio.exceptions import ApplyError
from claude_remote_audio.orchestrator import ApplyResult, apply

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
    install_prereqs: Annotated[
        bool,
        typer.Option(
            '--install-prereqs',
            help='Dispatch scripts/bootstrap.sh to hosts missing audio prereqs before applying.',
        ),
    ] = False,
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
    log_path = _configure_apply_logging()
    typer.echo(f'apply log: {log_path}', err=True)

    result = asyncio.run(
        apply(
            target=target,
            hub=hub,
            input_device=input_device,
            output_device=output_device,
            install_prereqs=install_prereqs,
        )
    )

    if format == 'json':
        typer.echo(result.model_dump_json())
    else:
        _print_text(result)

    raise SystemExit(0 if result.overall_success else 1)


class _TeeStream:
    """Tee writes to multiple file-like streams (terminal + log file).

    Replace ``sys.stderr`` and/or ``sys.stdout`` with one of these and any
    code writing to those streams — ``typer.echo``, ``render_recovery``,
    ``print``, third-party libraries — naturally lands in every backing
    stream. No per-handler bookkeeping in user-facing output paths.

    ``isatty`` / ``fileno`` delegate to the first stream so context-aware
    renderers (e.g. ``render_recovery``'s TTY-vs-piped branch) keep detecting
    the terminal correctly.

    Composition-tee idiom (vs. ``os.dup2`` FD redirect, ``StreamToLogger``
    coercion, or ``contextlib.redirect_stderr``) — the right shape for a
    pure-Python CLI that wants raw-text capture with the terminal untouched.
    """

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self._streams) and self._streams[0].isatty()

    def fileno(self) -> int:
        return self._streams[0].fileno()


def _configure_apply_logging() -> Path:
    """Wire stdout/stderr tees + root logger to per-run log file.

    Three write paths, all landing in the same log file:

    - **Stderr tee**: ``sys.stderr`` becomes ``_TeeStream(real_stderr, log_fp)``.
      Raw user-facing output via ``typer.echo(err=True)``, ``render_recovery``,
      and anything else writing to ``sys.stderr`` flows to terminal AND log.
    - **Stdout tee**: ``sys.stdout`` becomes ``_TeeStream(real_stdout, log_fp)``.
      Captures the success path (``_print_text`` action lines, ``apply: ok``
      summary, ``--format json`` payload) into the log too.
    - **File logger**: a ``StreamHandler`` bound to the same ``log_fp``
      captures ``logger.*`` calls with timestamp + level prefix.

    The console handlers are bound to the real ``sys.__stderr__`` /
    ``sys.__stdout__`` (not the tees), so ``logger.*`` events land in the log
    file once (via the file handler with timestamp) rather than twice (via the
    tee on top of the file handler).
    """
    paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime('%Y%m%d-%H%M%S')
    log_path = paths.LOGS_DIR / f'apply-{timestamp}.log'
    log_fp = log_path.open('a', buffering=1, encoding='utf-8')

    real_stderr = sys.__stderr__
    real_stdout = sys.__stdout__
    if real_stderr is None or real_stdout is None:
        raise RuntimeError('sys.__stderr__ / sys.__stdout__ is None — cannot configure apply logging')

    # Wrap stdout/stderr BEFORE constructing the console StreamHandler — the
    # handler binds its stream eagerly at __init__, so we bind it to the real
    # underlying stream explicitly to bypass the tee (avoiding double-write
    # to the log file when logger events fire).
    sys.stderr = _TeeStream(real_stderr, log_fp)
    sys.stdout = _TeeStream(real_stdout, log_fp)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for existing in list(root.handlers):
        root.removeHandler(existing)

    console = logging.StreamHandler(real_stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    root.addHandler(console)

    file_log = logging.StreamHandler(log_fp)
    file_log.setLevel(logging.DEBUG)
    file_log.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root.addHandler(file_log)

    return log_path


@error_boundary.handler(ApplyError)
def _handle_apply_error(exc: ApplyError) -> None:
    """User-facing configuration / constraint violation — clean message, no traceback.

    For ``ResolvableApplyError`` (or any ApplyError that also subclasses
    ``ResolvableError``), append a context-aware recovery footer — agent-engagement
    hint inside Claude Code, red-text CTA in a bare terminal, silent when piped.
    """
    typer.echo(f'apply: {exc}', err=True)
    if isinstance(exc, ResolvableError):
        render_recovery(exc)


@error_boundary.handler(RemoteBashError)
def _handle_dispatch_error(exc: RemoteBashError) -> None:
    """Fallback: dispatch-layer failure that escaped the orchestrator's wrapping — print cleanly."""
    typer.echo(f'dispatch: {exc}', err=True)


@error_boundary.handler(SelectorError)
def _handle_selector_error(exc: SelectorError) -> None:
    """``--target`` selector grammar / discovery-mismatch errors — clean message, no traceback."""
    typer.echo(f'target: {exc}', err=True)


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
