from __future__ import annotations

import asyncio
import functools
import traceback
from collections.abc import Callable, Sequence
from typing import Literal

import click.shell_completion
import typer
from claude_remote_bash import DispatchService
from claude_remote_bash.cache import HostsCache
from claude_remote_bash.client_config import ClientConfig
from claude_remote_bash.discovery import browse_hosts
from typer._completion_classes import ZshComplete  # noqa: PLC2701 — official extension point not re-exported

from claude_remote_audio.cache import read_devices
from claude_remote_audio.orchestrator import enumerate_devices

__all__ = [
    'complete_hub',
    'complete_input_devices',
    'complete_output_devices',
    'complete_target',
    'register_zsh_template',
]


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

    Defined here (above the public callbacks it decorates) because Python evaluates
    decorators at definition time; the linter exempts it from "public-first" ordering
    via its import-time-refs rule.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Sequence[str | tuple[str, str]]:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # exception_safety_linter.py: swallowed-exception — completion path renders traceback in menu, never to stderr
            return [(' ', ''.join(traceback.format_exception(exc)))]

    return wrapper


def register_zsh_template() -> None:
    """Override typer's default ZshComplete with the diagnostic-aware variant.

    Must be called AFTER ``create_app()`` — that's what calls typer's
    ``completion_init()``, which registers the default ZshComplete. We have to
    land last to win.
    """
    click.shell_completion.add_completion_class(_DiagnosticZshComplete, 'zsh')


@_completion_safe
def complete_output_devices(ctx: typer.Context, incomplete: str) -> Sequence[str | tuple[str, str]]:
    """Tab-complete ``--output`` from the resolved hub's output devices."""
    return _complete_from_cache(ctx, incomplete, kind='output')


@_completion_safe
def complete_input_devices(ctx: typer.Context, incomplete: str) -> Sequence[str | tuple[str, str]]:
    """Tab-complete ``--input`` from the resolved hub's input devices."""
    return _complete_from_cache(ctx, incomplete, kind='input')


@_completion_safe
def complete_hub(incomplete: str) -> Sequence[str | tuple[str, str]]:
    """Tab-complete ``--hub`` from discovered daemon aliases.

    Case matching is left to the shell (zsh ``setopt nocaseglob`` etc.) — click's
    completion wrapper post-filters case-sensitively.
    """
    return [a for a in _alias_atoms() if a.startswith(incomplete)]


@_completion_safe
def complete_target(incomplete: str) -> Sequence[str | tuple[str, str]]:
    """Tab-complete ``--target`` from aliases + group names; preserves comma-list prefix."""
    atoms = [*_alias_atoms(), *ClientConfig.load().groups.keys()]
    if ',' in incomplete:
        prefix, _, suffix = incomplete.rpartition(',')
        return [f'{prefix},{a}' for a in atoms if a.startswith(suffix)]
    return [a for a in atoms if a.startswith(incomplete)]


# -- Private implementation ----------------------------------------------------


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


def _resolve_hub_for_completion(ctx: typer.Context) -> str | None:
    """Hub to use for tab completion — explicit ``--hub`` wins, else local self from bash cache."""
    hub: str | None = ctx.params.get('hub')
    if hub:
        return hub
    cache = HostsCache.load(max_age_s=None)
    if cache is None or cache.local_daemon is None:
        return None
    return cache.local_daemon.alias


def _complete_from_cache(
    ctx: typer.Context,
    incomplete: str,
    *,
    kind: Literal['output', 'input'],
) -> Sequence[str]:
    """Prefix-match the per-hub device cache; dispatch + populate on cache miss.

    Exceptions propagate to the ``@_completion_safe`` decorator on the public callbacks.
    """
    hub = _resolve_hub_for_completion(ctx)
    if not hub:
        return []
    cached = read_devices(hub)
    if cached is None:
        cached = asyncio.run(enumerate_devices(DispatchService(), hub))
    names = cached.outputs if kind == 'output' else cached.inputs
    return [n for n in names if n.startswith(incomplete)]


def _alias_atoms() -> Sequence[str]:
    """Return all daemon aliases — bash hosts cache when warm, fresh mDNS browse on miss."""
    cache = HostsCache.load(max_age_s=None)
    if cache is not None:
        return [h.alias for h in cache.all_hosts()]
    browse = asyncio.run(browse_hosts(timeout=3.0))
    HostsCache.from_browse(browse).write()
    return [h.alias for h in [*browse.remote_daemons, *([browse.local_daemon] if browse.local_daemon else [])]]
