"""Command-line interface for grok-kit."""

from __future__ import annotations

__all__ = [
    'app',
    'main',
]

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

from grok_kit.auth import (
    DEFAULT_COOKIE_PATH,
    LOAD_BEARING_COOKIES,
    expired_load_bearing,
    import_state,
    load_state,
    missing_load_bearing,
)
from grok_kit.service import GrokService

logger = logging.getLogger(__name__)

app = create_app(help='grok-kit — mirror grok.com conversations.')
error_boundary = ErrorBoundary(exit_code=1)


@app.callback(invoke_without_command=True)
def _configure_logging(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option('--verbose', '-v', help='Show detailed output')] = False,
) -> None:
    """Configure logging and show help when no command given."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(message)s', stream=sys.stderr, force=True)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command('list', rich_help_panel='Conversations')
@error_boundary
def list_cmd(
    title_contains: Annotated[
        str | None,
        typer.Option('--title-contains', '-t', help='Filter by title substring.'),
    ] = None,
    case_sensitive: Annotated[
        bool,
        typer.Option('--case-sensitive', help='Match title case-sensitively.'),
    ] = False,
    limit: Annotated[int, typer.Option('--limit', '-n', help='Max results.')] = 60,
) -> None:
    """List grok.com conversations as JSON to stdout."""
    convs = GrokService.from_cookies().list_conversations(
        limit=None if title_contains else limit,
    )
    if title_contains is not None:
        needle = title_contains if case_sensitive else title_contains.lower()
        convs = [c for c in convs if needle in (c.title if case_sensitive else c.title.lower())][:limit]
    typer.echo('[' + ',\n'.join(c.model_dump_json(by_alias=True) for c in convs) + ']')


@app.command('get', rich_help_panel='Conversations')
@error_boundary
def get_cmd(
    conversation_id: Annotated[str, typer.Argument(help='UUID of the conversation.')],
) -> None:
    """Fetch one conversation in full and emit it as JSON to stdout."""
    typer.echo(
        GrokService.from_cookies().get_full_conversation(conversation_id).model_dump_json(by_alias=True, indent=2),
    )


@app.command('auth-status', rich_help_panel='Auth')
@error_boundary
def auth_status(
    cookie_path: Annotated[
        Path,
        typer.Option('--cookie-path', help='Cookie file location.'),
    ] = DEFAULT_COOKIE_PATH,
) -> None:
    """Report cookie file presence, load-bearing-cookie coverage, and expirations."""
    if not cookie_path.exists():
        typer.echo(f'No cookie file at {cookie_path}')
        typer.echo('Run the `grok-kit-auth` skill to bootstrap.')
        raise typer.Exit(2)

    state = load_state(cookie_path)
    missing = missing_load_bearing(state)
    expired = expired_load_bearing(state)

    typer.echo(f'Cookie file: {cookie_path}')
    typer.echo(f'Total cookies: {len(state.cookies)}')
    typer.echo(f'Load-bearing required: {", ".join(LOAD_BEARING_COOKIES)}')
    typer.echo(f'Missing: {", ".join(missing) if missing else "none ✓"}')
    typer.echo(f'Expired: {", ".join(expired) if expired else "none ✓"}')


@app.command('auth-logout', rich_help_panel='Auth')
@error_boundary
def auth_logout(
    cookie_path: Annotated[
        Path,
        typer.Option('--cookie-path', help='Cookie file to delete.'),
    ] = DEFAULT_COOKIE_PATH,
) -> None:
    """Delete the cookie file."""
    if cookie_path.exists():
        cookie_path.unlink()
        typer.echo(f'Deleted {cookie_path}')
    else:
        typer.echo(f'No cookie file at {cookie_path} (nothing to do)')


@app.command('auth-import', rich_help_panel='Auth')
@error_boundary
def auth_import(
    state_file: Annotated[Path, typer.Argument(help='Profile-state JSON to import.')],
    cookie_path: Annotated[
        Path,
        typer.Option('--cookie-path', help='Where to write the imported cookies.'),
    ] = DEFAULT_COOKIE_PATH,
) -> None:
    """Import a profile-state JSON into grok-kit's cookie store."""
    result = import_state(state_file, target_path=cookie_path)
    typer.echo(f'Imported {result.cookie_count} cookies → {result.cookie_path}')
    typer.echo(
        f'Missing load-bearing: {", ".join(result.missing_load_bearing) if result.missing_load_bearing else "none ✓"}'
    )
    typer.echo(
        f'Expired load-bearing: {", ".join(result.expired_load_bearing) if result.expired_load_bearing else "none ✓"}'
    )
    if result.missing_load_bearing:
        raise typer.Exit(2)


# Register documentation and shell-completion commands last so their panels
# appear after Conversations and Auth in --help output.
add_help_command(app)
add_completion_command(app)


def main() -> None:
    """CLI entry point."""
    run_app(app)
