"""Command-line interface for imessage-kit."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from cc_lib.cli import add_completion_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

from imessage_kit.attachments import AttachmentHandler
from imessage_kit.contacts import ContactResolver
from imessage_kit.db import ChatDB, FullDiskAccessError
from imessage_kit.sender import MessageSender
from imessage_kit.service import IMessageService, ServerState
from imessage_kit.sources import SourceRegistry
from imessage_kit.system import detect_root_app

app = create_app(help='imessage-kit — Read, search, and send iMessages via macOS chat.db.')
add_completion_command(app)
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


# -- Commands --


@app.command('diagnose')
@error_boundary
def diagnose() -> None:
    """Check FDA, DB access, contacts, and macOS version.

    \b
    Always works, even without Full Disk Access — reports exactly what's wrong.
    """
    service = _make_service()
    _print_json(service.diagnose())


@app.command('list-chats')
@error_boundary
def list_chats(
    limit: Annotated[int, typer.Option('--limit', '-n', help='Max chats to return')] = 20,
    is_group: Annotated[bool | None, typer.Option('--group/--no-group', help='Filter group or 1:1 chats')] = None,
) -> None:
    """List recent chats sorted by last message date."""
    service = _make_service()
    _print_json(service.list_chats(limit=limit, is_group=is_group))


@app.command('get-messages')
@error_boundary
def get_messages(
    handle: Annotated[str, typer.Argument(help='Phone number or email')],
    limit: Annotated[int, typer.Option('--limit', '-n', help='Max messages to return')] = 20,
    from_me: Annotated[bool | None, typer.Option('--from-me/--received', help='Filter sent or received')] = None,
    has_attachment: Annotated[
        bool | None, typer.Option('--has-attachment/--no-attachment', help='Filter by attachment')
    ] = None,
) -> None:
    """Get messages from a chat by handle.

    \b
    Examples:
        imessage-kit get-messages +15555550100
        imessage-kit get-messages +15555550100 --limit 50 --from-me
    """
    service = _make_service()
    _print_json(service.get_messages(handle=handle, limit=limit, from_me=from_me, has_attachment=has_attachment))


@app.command('search')
@error_boundary
def search(
    query: Annotated[str, typer.Argument(help='Search text')],
    limit: Annotated[int, typer.Option('--limit', '-n', help='Max results')] = 10,
    handle: Annotated[str | None, typer.Option('--handle', help='Scope to chats with this handle')] = None,
) -> None:
    """Search messages across all chats.

    \b
    Two-pass search: SQL LIKE on text column, then parses attributedBody
    blobs for messages with NULL text.
    """
    service = _make_service()
    _print_json(service.search_messages(query, limit=limit, handle=handle))


@app.command('lookup-contact')
@error_boundary
def lookup_contact(
    query: Annotated[str, typer.Argument(help='Name, phone, or email')],
    limit: Annotated[int, typer.Option('--limit', '-n', help='Max matches')] = 10,
    source: Annotated[
        str | None,
        typer.Option('--source', help="Filter by source display name (e.g., 'Google', 'iCloud')"),
    ] = None,
) -> None:
    """Search macOS AddressBook for a contact.

    \b
    Results are deduplicated across all AddressBook sync sources
    (Google, iCloud, iCloud CloudKit). Each result lists the sources
    it was merged from. Use --source to restrict to a single source.
    """
    service = _make_service()
    _print_json(service.lookup_contact(query, limit=limit, source=source))


@app.command('get-unread')
@error_boundary
def get_unread() -> None:
    """Get unread messages across all chats."""
    service = _make_service()
    _print_json(service.get_unread())


@app.command('get-chat-info')
@error_boundary
def get_chat_info(
    handle: Annotated[str, typer.Argument(help='Phone number or email')],
) -> None:
    """Get detailed chat metadata: participants, counts, date range."""
    service = _make_service()
    _print_json(service.get_chat_info(handle=handle))


@app.command('list-attachments')
@error_boundary
def list_attachments(
    handle: Annotated[str, typer.Argument(help='Phone number or email')],
    limit: Annotated[int, typer.Option('--limit', '-n', help='Max attachments')] = 20,
    mime_type: Annotated[str | None, typer.Option('--mime', help="Prefix filter (e.g., 'image/')")] = None,
) -> None:
    """List attachment metadata for a chat."""
    service = _make_service()
    _print_json(service.list_attachments(handle=handle, limit=limit, mime_type=mime_type))


def main() -> None:
    """Entry point for imessage-kit CLI."""
    run_app(app)


# -- Private helpers --


def _make_service() -> IMessageService:
    """Create a service instance for CLI commands."""
    db = ChatDB()
    db.connect()

    temp_dir_obj = tempfile.TemporaryDirectory(prefix='imessage-kit-cli-')
    temp_path = Path(temp_dir_obj.name)

    registry = SourceRegistry.discover()

    state = ServerState(
        db=db,
        contacts=ContactResolver(sources=registry.sources),
        sources=registry.sources,
        attachments=AttachmentHandler(temp_path),
        sender=MessageSender(),
        temp_dir=temp_path,
        fda_available=True,
    )
    return IMessageService(state)


def _print_json(result: object) -> None:
    """Serialize a Pydantic model or sequence to JSON."""
    if hasattr(result, 'model_dump'):
        data = result.model_dump()
    elif isinstance(result, (list, tuple)):
        data = [r.model_dump() for r in result]
    else:
        data = result
    typer.echo(json.dumps(data, indent=2, default=str))


# -- Error handlers --


@error_boundary.handler(FullDiskAccessError)
def _handle_fda_error(exc: FullDiskAccessError) -> None:
    root_app = detect_root_app()
    typer.secho(
        f'Full Disk Access required.\n'
        f'Grant it to: {root_app}\n'
        f'System Settings → Privacy & Security → Full Disk Access\n'
        f'\n'
        f'To open the settings pane:\n'
        f'  open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"',
        fg=typer.colors.RED,
        err=True,
    )


@error_boundary.handler(ValueError)
def _handle_value_error(exc: ValueError) -> None:
    typer.secho(f'Error: {exc}', fg=typer.colors.RED, err=True)
