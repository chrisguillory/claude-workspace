from __future__ import annotations

import asyncio
import logging
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, TextIO

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary, render_recovery
from cc_lib.exceptions import ResolvableError
from cc_lib.schemas import ClosedModel
from cc_lib.types import OutputFormat
from claude_remote_bash.exceptions import RemoteBashError
from claude_remote_bash.selector import SelectorError
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel

from claude_remote_audio import paths
from claude_remote_audio.cli import completion
from claude_remote_audio.exceptions import ApplyError
from claude_remote_audio.orchestrator import ApplyResult, HostApplyOutcome, HostError, apply

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

    try:
        result = asyncio.run(
            apply(
                target=target,
                hub=hub,
                input_device=input_device,
                output_device=output_device,
                install_prereqs=install_prereqs,
            )
        )
    except Exception as exc:
        # `--format json` users want JSON on every code path — including errors.
        # Without this, the documented "machine-readable for scripting" contract
        # is silently broken on failure (ErrorBoundary prints prose-to-stderr).
        # Catches the full Exception hierarchy so unexpected types (KeyError,
        # OSError, etc.) still produce a JSON envelope rather than escaping to
        # the boundary's default prose-on-stderr path. For text-format callers,
        # re-raise so the boundary's typed handlers run unchanged (preserves
        # the recovery footer + structured-error rendering per exception type).
        if format == 'json':
            _emit_json_error(exc)
            raise SystemExit(1) from exc
        raise

    if format == 'json':
        typer.echo(result.model_dump_json())
    else:
        _print_text(result)

    raise SystemExit(0 if result.overall_success else 1)


class _ApplyErrorEnvelope(ClosedModel):
    """Error-path JSON envelope matching ``ApplyResult``'s shape.

    Same ``alias_generator=to_camel`` as ``ApplyResult`` so the wire shape is
    structurally symmetric across success and failure paths. ``hosts`` is empty
    on top-level (non-per-host) failures; the carried ``error`` is a
    ``HostError`` mirroring the per-host outcome shape.
    """

    model_config = ConfigDict(alias_generator=to_camel)

    hosts: Sequence[HostApplyOutcome]
    overall_success: bool
    error: HostError


def _emit_json_error(exc: Exception) -> None:
    """Emit a JSON error envelope on stdout, mirroring ApplyResult's shape.

    Accepts any ``Exception`` so the JSON contract holds across both the
    documented per-host failure types (``ApplyError`` / ``RemoteBashError`` /
    ``SelectorError``) and unexpected types (``KeyError``, ``OSError``, etc.)
    that would otherwise escape to the boundary's prose-on-stderr path. Goes
    through a Pydantic envelope model (rather than hand-built dict) so the
    output stays structurally symmetric with the success path: same camelCase
    aliases (``overallSuccess``, ``docsUrl``), same ``Mapping[str, str]`` shape
    for ``context`` (empty dict, never null), same Pydantic-validated types
    across both code paths. The ``ResolvableError`` fields are carried as a
    nested ``ErrorEnvelope`` matching ``HostError``'s shape for cross-path
    consistency; non-resolvable exceptions surface as a message-only
    ``HostError``.
    """
    error = (
        HostError(
            message=str(exc),
            code=exc.code,
            title=exc.title,
            suggestions=tuple(exc.suggestions),
            docs_url=exc.docs_url,
            context=dict(exc.context),
        )
        if isinstance(exc, ResolvableError)
        else HostError(message=str(exc))
    )
    envelope = _ApplyErrorEnvelope(overall_success=False, hosts=(), error=error)
    typer.echo(envelope.model_dump_json(by_alias=True))


# Strip CSI escape sequences from log-file writes. Click/typer auto-color
# when the target ``isatty()`` returns True — the tee reports True (delegated
# to the real terminal) so colorized output flows to both streams. Without
# stripping, the log file fills with ``^[[31mERROR^[[0m``-style bytes that
# defeat grep. Covers the common ESC [ ... letter form; rarer OSC/DCS
# escapes (hyperlinks, etc.) we don't emit aren't worth a more elaborate
# parser.
_ANSI_CSI_PATTERN = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')


class _TeeStream:
    """Tee writes to the real terminal stream + the apply-log file.

    Replace ``sys.stderr`` and/or ``sys.stdout`` with one of these and any
    code writing to those streams — ``typer.echo``, ``render_recovery``,
    ``print``, third-party libraries — naturally lands in both. No per-handler
    bookkeeping in user-facing output paths.

    Terminal writes pass through verbatim (color/style preserved); log writes
    have CSI escapes stripped so the file stays grep-friendly. ``isatty`` /
    ``fileno`` delegate to the real terminal so context-aware renderers (e.g.
    ``render_recovery``'s TTY-vs-piped branch) keep detecting the terminal.

    Composition-tee idiom (vs. ``os.dup2`` FD redirect, ``StreamToLogger``
    coercion, or ``contextlib.redirect_stderr``) — the right shape for a
    pure-Python CLI that wants raw-text capture with the terminal untouched.
    """

    def __init__(self, real: TextIO, log: TextIO) -> None:
        self._real = real
        self._log = log

    def write(self, data: str) -> int:
        self._real.write(data)
        self._real.flush()
        self._log.write(_ANSI_CSI_PATTERN.sub('', data))
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._real.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._real.isatty()

    def fileno(self) -> int:
        return self._real.fileno()


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
    # Microsecond precision so back-to-back hot-cache applies (sub-second on a
    # single host) don't share a filename and interleave output via append mode.
    timestamp = datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')
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
    """Render per-host apply outcomes as human-readable lines + a one-line summary.

    When a host's ``error`` carries a structured ``code`` (from a
    ``ResolvableError`` subclass), call ``render_recovery`` per failing host so
    the in-loop agent-engagement footer + bare-terminal CTA fire for each
    failure, not just the top-level escape path.
    """
    for host in result.hosts:
        status = 'ok' if host.success else 'failed'
        typer.echo(f'[{status}] {host.host} ({host.role})')
        for action in host.actions:
            typer.echo(f'  - {action}')
        if host.error is not None:
            typer.echo(f'  ERROR: {host.error.message}', err=True)
            if host.error.code is not None:
                render_recovery(_render_recovery_shim(host.error))

    summary = 'ok' if result.overall_success else 'failed'
    typer.echo(f'\napply: {summary}')


def _render_recovery_shim(host_error: HostError) -> ResolvableError:
    """Adapter: build a transient ``ResolvableError`` from ``HostError`` for ``render_recovery``.

    ``render_recovery`` accepts any ``ResolvableError``-typed object; we synthesize
    one whose fields mirror the ``HostError`` so the renderer treats it
    identically to a freshly-raised exception.
    """
    return ResolvableError(
        host_error.message,
        code=host_error.code or 'host-error',
        title=host_error.title,
        suggestions=host_error.suggestions,
        docs_url=host_error.docs_url,
        context=host_error.context,
    )
