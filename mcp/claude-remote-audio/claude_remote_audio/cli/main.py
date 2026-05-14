from __future__ import annotations

import asyncio
import functools
import logging
import traceback
from collections.abc import Callable, Sequence
from typing import Annotated, Literal

import click.shell_completion
import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.types import OutputFormat
from claude_remote_bash import DispatchService
from claude_remote_bash.cache import HostsCache
from claude_remote_bash.client_config import ClientConfig
from claude_remote_bash.discovery import browse_hosts
from claude_remote_bash.exceptions import RemoteBashError
from typer._completion_classes import ZshComplete  # noqa: PLC2701 — official extension point not re-exported

from claude_remote_audio.cache import read_devices
from claude_remote_audio.orchestrator import ApplyError, ApplyResult, apply, enumerate_devices

__all__ = [
    'main',
]

logger = logging.getLogger(__name__)


class _DiagnosticZshComplete(ZshComplete):
    """Render single-item diagnostic completions via ``compadd -x`` (per line).

    Convention: a callback signals "couldn't fetch real completions, here's why"
    by returning exactly one item with ``value=' '`` and a non-empty ``help``.
    Typer's default template would render it as ``_arguments '*: :((" ":"help"))'``,
    which zsh auto-completes (the space renders as ``\\``) and hides the description.
    We intercept that case and emit ``compadd -x '<line>'`` once per help line. ``-x``
    is zsh's "non-selectable message" flag — multiple calls stack as separate display
    rows in the completion area. Pattern lifted from Cobra ActiveHelp (kubectl / gh /
    docker / helm); ``_message -r`` only renders the first call.
    """

    def complete(self) -> str:
        args, incomplete = self.get_completion_args()
        completions = self.get_completions(args, incomplete)
        if len(completions) == 1 and completions[0].value == ' ' and completions[0].help:
            # ``;`` separator, not newline — typer's zsh template wraps our output in
            # ``eval $(...)``, and ``$(...)`` collapses newlines into spaces. Statements
            # need explicit separators or they merge into one mis-parsed command.
            parts = []
            for line in completions[0].help.splitlines():
                escaped = line.replace("'", "'\\''")
                parts.append(f"compadd -x '{escaped}'")
            return '; '.join(parts)
        res = [self.format_completion(item) for item in completions]
        if res:
            args_str = '\n'.join(res)
            return f"_arguments '*: :(({args_str}))'"
        return '_files'


app = create_app(help='Multi-Mac audio topology orchestration over roc-vad + roc-toolkit.')
add_completion_command(app)
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)

# Must come AFTER create_app — that's what calls typer's completion_init(), which
# registers the default ZshComplete. Our override has to land last to win.
click.shell_completion.add_completion_class(_DiagnosticZshComplete, 'zsh')


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
            autocompletion=lambda incomplete: _complete_target(incomplete),
        ),
    ],
    hub: Annotated[
        str | None,
        typer.Option(
            '--hub',
            help='Hub host alias (default: the local machine)',
            autocompletion=lambda incomplete: _complete_hub(incomplete),
        ),
    ] = None,
    input_device: Annotated[
        str | None,
        typer.Option(
            '--input',
            '-i',
            help='Hub Core Audio input device (mic) — opt in to manage roc-send',
            autocompletion=lambda ctx, incomplete: _complete_input_devices(ctx, incomplete),
        ),
    ] = None,
    output_device: Annotated[
        str | None,
        typer.Option(
            '--output',
            '-o',
            help='Hub default output device — opt in to manage roc-recv',
            autocompletion=lambda ctx, incomplete: _complete_output_devices(ctx, incomplete),
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


# -- Completion callbacks ------------------------------------------------------


def _completion_safe[**P](
    fn: Callable[P, Sequence[str | tuple[str, str]]],
) -> Callable[P, Sequence[str | tuple[str, str]]]:
    """Wrap a completion callback so any exception renders as a traceback in the menu.

    Python's default exception hook prints the traceback to stderr — which in a
    zsh tab-completion subprocess mangles the user's prompt. We catch broadly,
    format via ``traceback.format_exception``, and return it as a diagnostic
    ``(value, help)`` tuple. ``_DiagnosticZshComplete`` then emits one
    ``compadd -x`` per line, so the full traceback (frames, file paths, line
    numbers) renders in the completion area — the dev-friendly signal.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Sequence[str | tuple[str, str]]:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # exception_safety_linter.py: swallowed-exception — completion path renders traceback in menu, never to stderr
            return [(' ', ''.join(traceback.format_exception(exc)))]

    return wrapper


def _resolve_hub_for_completion(ctx: typer.Context) -> str | None:
    """Hub to use for tab completion — explicit ``--hub`` wins, else local self from bash cache."""
    hub: str | None = ctx.params.get('hub')
    if hub:
        return hub
    cache = HostsCache.load(max_age_s=None)
    if cache is None:
        return None
    self_entry = next((h for h in cache.hosts if h.is_self), None)
    return self_entry.alias if self_entry else None


def _complete_from_cache(
    ctx: typer.Context,
    incomplete: str,
    *,
    kind: Literal['output', 'input'],
) -> Sequence[str]:
    """Prefix-match the per-hub device cache; dispatch + populate on cache miss.

    Exceptions propagate — the ``@_completion_safe`` decorator on the public callbacks
    catches them and renders the traceback in the menu.
    """
    hub = _resolve_hub_for_completion(ctx)
    if not hub:
        return []
    cached = read_devices(hub)
    if cached is None:
        cached = asyncio.run(enumerate_devices(DispatchService(), hub))
    names = cached.outputs if kind == 'output' else cached.inputs
    return [n for n in names if n.startswith(incomplete)]


@_completion_safe
def _complete_output_devices(ctx: typer.Context, incomplete: str) -> Sequence[str | tuple[str, str]]:
    """Tab-complete ``--output`` from the resolved hub's output devices."""
    return _complete_from_cache(ctx, incomplete, kind='output')


@_completion_safe
def _complete_input_devices(ctx: typer.Context, incomplete: str) -> Sequence[str | tuple[str, str]]:
    """Tab-complete ``--input`` from the resolved hub's input devices."""
    return _complete_from_cache(ctx, incomplete, kind='input')


@_completion_safe
def _complete_hub(incomplete: str) -> Sequence[str | tuple[str, str]]:
    """Tab-complete ``--hub`` from discovered daemon aliases.

    Case matching is left to the shell (zsh ``setopt nocaseglob`` etc.) — click's
    completion wrapper post-filters case-sensitively.
    """
    return [a for a in _alias_atoms() if a.startswith(incomplete)]


@_completion_safe
def _complete_target(incomplete: str) -> Sequence[str | tuple[str, str]]:
    """Tab-complete ``--target`` from aliases + group names; preserves comma-list prefix."""
    atoms = [*_alias_atoms(), *ClientConfig.load().groups.keys()]
    if ',' in incomplete:
        prefix, _, suffix = incomplete.rpartition(',')
        return [f'{prefix},{a}' for a in atoms if a.startswith(suffix)]
    return [a for a in atoms if a.startswith(incomplete)]


def _alias_atoms() -> Sequence[str]:
    """Return all daemon aliases — bash hosts cache when warm, fresh mDNS browse on miss."""
    cache = HostsCache.load(max_age_s=None)
    if cache is not None:
        return [h.alias for h in cache.hosts]
    hosts = asyncio.run(browse_hosts(timeout=3.0))
    HostsCache.from_browse(hosts).write()
    return [h.alias for h in hosts]


# -- Rendering helpers ---------------------------------------------------------


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
