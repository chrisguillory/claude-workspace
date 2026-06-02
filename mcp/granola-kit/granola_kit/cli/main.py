from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Annotated

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

from granola_kit.clients.granola_api import granola_api_client
from granola_kit.exceptions import GranolaError
from granola_kit.schemas.results import Meeting
from granola_kit.services.meetings import MeetingService

app = create_app(help='granola-kit — Granola.ai meeting notes, transcripts, and chat-based Q&A.')
add_completion_command(app)
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)


@app.command('list-meetings', rich_help_panel='Meetings')
def list_meetings(
    limit: Annotated[int, typer.Option('--limit', '-n', help='Max meetings, most-recently-updated first')] = 20,
) -> None:
    """List recent Granola meetings."""
    meetings = asyncio.run(_list_meetings(limit))
    typer.echo(json.dumps([m.model_dump() for m in meetings], indent=2, default=str))


@error_boundary
def main() -> None:
    """Run the granola-kit CLI."""
    run_app(app)


async def _list_meetings(limit: int) -> Sequence[Meeting]:
    """Resolve meetings through a per-invocation client + service."""
    async with granola_api_client() as client:
        return await MeetingService(client).list_meetings(limit=limit)


@error_boundary.handler(GranolaError)
def _handle_granola_error(exc: GranolaError) -> None:
    """Render granola-kit errors as a clean stderr message rather than a traceback."""
    typer.secho(f'Error: {exc}', fg=typer.colors.RED, err=True)
