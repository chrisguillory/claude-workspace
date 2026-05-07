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
    load_cookies,
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
        typer.echo('Run `grok-kit auth-login` to bootstrap.')
        raise typer.Exit(2)

    cookies = load_cookies(cookie_path)
    missing = missing_load_bearing(cookies)
    expired = expired_load_bearing(cookies)

    typer.echo(f'Cookie file: {cookie_path}')
    typer.echo(f'Total cookies: {len(cookies.cookies)}')
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


@app.command('auth-login', rich_help_panel='Auth')
@error_boundary
def auth_login(
    username: Annotated[str, typer.Option('--username', help='X username or email.')],
    password_stdin: Annotated[
        bool,
        typer.Option('--password-stdin', help='Read password from stdin (avoids argv leak).'),
    ] = False,
    totp_stdin: Annotated[
        bool,
        typer.Option('--totp-stdin', help='Read TOTP from stdin (alongside password).'),
    ] = False,
) -> None:
    """Drive a fresh-Chromium X SSO login and save the resulting grok.com cookies.

    NOT YET IMPLEMENTED — the Chromium-driving + X login form-fill flow is
    its own work item. For now, manually save state via the selenium-browser
    MCP against an authenticated grok.com session.
    """
    _ = (username, password_stdin, totp_stdin)
    raise NotImplementedError(
        'auth-login (fresh-Chromium X SSO drive) not yet implemented. '
        f'For now, manually save state via the selenium-browser MCP to {DEFAULT_COOKIE_PATH}.'
    )


# Register documentation and shell-completion commands last so their panels
# appear after Conversations and Auth in --help output.
add_help_command(app)
add_completion_command(app)


def main() -> None:
    """CLI entry point."""
    run_app(app)
