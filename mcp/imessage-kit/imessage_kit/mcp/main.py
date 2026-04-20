"""FastMCP server for imessage-kit."""

from __future__ import annotations

__all__ = ['main', 'server']

import asyncio
import contextlib
import logging
import sys
import tempfile
import typing
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import mcp.server.fastmcp
import mcp.types

from imessage_kit.attachments import AttachmentHandler
from imessage_kit.contacts import ContactResolver
from imessage_kit.db import ChatDB, FullDiskAccessError
from imessage_kit.sender import MessageSender
from imessage_kit.service import IMessageService, ServerState
from imessage_kit.sources import SourceRegistry
from imessage_kit.system import detect_root_app
from imessage_kit.types import (
    AttachmentMeta,
    AttachmentMode,
    AttachmentSave,
    AttachmentView,
    Chat,
    Contact,
    ContactSource,
    DiagnosticResult,
    Message,
    SendResult,
)

logger = logging.getLogger(__name__)


def register_tools(service: IMessageService) -> None:
    """Register service methods as MCP tools via closures."""
    sources_summary = _format_sources_summary(service.sources)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='List Chats',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def list_chats(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        limit: int = 20,
        offset: int = 0,
        is_group: bool | None = None,
        active_since: datetime | None = None,
        handle: str | None = None,
    ) -> Sequence[Chat]:
        """List chats ordered by most recent message date.

        Args:
            limit: Max chats to return (default 20).
            offset: Skip first N chats.
            is_group: Filter to group chats (true) or 1:1 chats (false).
            active_since: Only chats with messages after this date.
            handle: Filter to chats involving this handle (matches via chat_handle_join).
        """
        return service.list_chats(
            limit=limit,
            offset=offset,
            is_group=is_group,
            active_since=active_since,
            handle=handle,
        )

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Get Messages',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_messages(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        handle: str | None = None,
        chat_id: int | None = None,
        limit: int = 50,
        before_rowid: int | None = None,
        after_rowid: int | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        has_attachment: bool | None = None,
        from_me: bool | None = None,
    ) -> Sequence[Message]:
        """Get messages from a chat by handle (phone/email) or chat_id.

        Parses attributedBody for messages with NULL text column.
        Includes attachment metadata and reactions inline.

        Cursor pagination: use before_rowid/after_rowid with ROWIDs from
        previous results. Mutually exclusive.

        Args:
            handle: Phone number or email (e.g., '+15555550100').
            chat_id: Direct chat ROWID (alternative to handle).
            limit: Max messages (default 50, max 500).
            before_rowid: Get messages older than this ROWID.
            after_rowid: Get messages newer than this ROWID.
            date_from: Messages after this date.
            date_to: Messages before this date.
            has_attachment: Filter to messages with/without attachments.
            from_me: Filter to sent (true) or received (false).
        """
        return service.get_messages(
            handle=handle,
            chat_id=chat_id,
            limit=min(limit, 500),
            before_rowid=before_rowid,
            after_rowid=after_rowid,
            date_from=date_from,
            date_to=date_to,
            has_attachment=has_attachment,
            from_me=from_me,
        )

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Search Messages',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def search_messages(
        query: str,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        limit: int = 25,
        offset: int = 0,
        handle: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        has_attachment: bool | None = None,
        from_me: bool | None = None,
    ) -> Sequence[Message]:
        """Full-text search across all chats.

        Two-pass: SQL LIKE on text column, then batch-parses attributedBody
        blobs for messages with NULL text. Capped at 50K row scan for pass 2.

        Args:
            query: Search text (case-insensitive substring).
            limit: Max results (default 25, max 100).
            offset: Skip first N results.
            handle: Scope search to chats with this handle.
            date_from: Messages after this date.
            date_to: Messages before this date.
            has_attachment: Filter to messages with/without attachments.
            from_me: Filter to sent (true) or received (false).
        """
        return service.search_messages(
            query,
            limit=min(limit, 100),
            offset=offset,
            handle=handle,
            date_from=date_from,
            date_to=date_to,
            has_attachment=has_attachment,
            from_me=from_me,
        )

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='List Attachments',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def list_attachments(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        handle: str | None = None,
        chat_id: int | None = None,
        limit: int = 20,
        offset: int = 0,
        mime_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        from_me: bool | None = None,
    ) -> Sequence[AttachmentMeta]:
        """List attachment metadata for a chat.

        Returns metadata only — use get_attachment with attachment_id
        to retrieve content.

        Args:
            handle: Phone/email to identify the chat.
            chat_id: Direct chat ROWID (alternative to handle).
            limit: Max attachments (default 20).
            offset: Skip first N attachments.
            mime_type: Prefix filter (e.g., 'image/', 'application/pdf').
            date_from: Attachments after this date.
            date_to: Attachments before this date.
            from_me: Filter to sent (true) or received (false).
        """
        return service.list_attachments(
            handle=handle,
            chat_id=chat_id,
            limit=limit,
            offset=offset,
            mime_type=mime_type,
            date_from=date_from,
            date_to=date_to,
            from_me=from_me,
        )

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Get Attachment',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_attachment(
        attachment_id: int,
        mode: AttachmentMode,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
    ) -> AttachmentView | AttachmentSave:
        """Retrieve an attachment by ID.

        mode='view': Base64 for Claude vision (HEIC auto-converted to JPEG).
        mode='save': Native file copied to temp dir, path returned.

        Args:
            attachment_id: Attachment ROWID from list_attachments or get_messages output.
            mode: 'view' (base64 inline) or 'save' (file path, native format).
        """
        return service.get_attachment(attachment_id, mode)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Send Message',
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def send_message(
        confirm: bool,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        text: str = '',
        handle: str | None = None,
        chat_guid: str | None = None,
        service_type: str = 'auto',
        attachments: Sequence[str] | None = None,
    ) -> SendResult:
        """Send text and/or attachments to a handle (1:1) or chat GUID (group).

        Two-step UX: call with confirm=false to preview, confirm=true to send.
        Security boundary is ToolAnnotations(destructiveHint=True).

        Each part (text + each attachment) is dispatched separately and arrives as
        its own bubble. Messages.app may visually group consecutive images on the
        receiving end. Fail-fast: if any part fails, subsequent parts are skipped
        and SendResult.parts_sent reflects what was already dispatched.

        Args:
            confirm: Must be true to send. False returns a preview.
            text: Message body. Defaults to empty for attachments-only sends.
            handle: Phone/email for 1:1 chats.
            chat_guid: Chat GUID for group chats (from list_chats output).
            service_type: 'auto' (default), 'iMessage', or 'SMS'. Only applies to 1:1
                sends; IGNORED when chat_guid is supplied (group chats route by
                the chat row's existing service).
            attachments: Ordered list of file paths (absolute or relative to cwd)
                to attach. Each must exist and be readable.
        """
        # Offload to a worker thread so the event loop stays responsive during
        # the up-to-45s chat.db polling (15s row discovery + 30s attachment delivery).
        # service.send_message is deliberately sync for CLI parity; wrapping with
        # asyncio.to_thread at the MCP boundary is the minimal way to un-block the
        # event loop without making the core sync-or-async bimodal.
        return await asyncio.to_thread(
            service.send_message,
            text,
            handle=handle,
            chat_guid=chat_guid,
            service=service_type,  # type: ignore[arg-type]  # MCP signature is plain str; service.send_message validates against SendService Literal at boundary
            confirm=confirm,
            attachments=attachments,
        )

    @server.tool(
        description=_inject_sources(LOOKUP_CONTACT_BASE, sources_summary),
        annotations=mcp.types.ToolAnnotations(
            title='Lookup Contact',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def lookup_contact(
        query: str,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        limit: int = 10,
        source: str | None = None,
    ) -> Sequence[Contact]:
        return service.lookup_contact(query, limit=limit, source=source)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Get Unread Messages',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )
    async def get_unread(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
    ) -> Sequence[Message]:
        """Get unread incoming messages across all chats.

        Returns all unread messages (is_read=0, is_from_me=0), excluding
        tapback reactions. Cross-device read sync may cause false positives.
        """
        return service.get_unread()

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Diagnose iMessage Access',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def diagnose(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
    ) -> DiagnosticResult:
        """Check Full Disk Access, DB connectivity, contacts, macOS version.

        Run this first to verify imessage-kit is properly configured.
        Always works, even without FDA — reports exactly what's wrong.
        """
        return service.diagnose()

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Get Chat Thread',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_chat_thread(
        thread_originator_guid: str,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        limit: int = 50,
    ) -> Sequence[Message]:
        """Get inline reply thread within a chat.

        Returns all replies to the specified message, plus the original.
        thread_originator_guid is available in get_messages output.

        Args:
            thread_originator_guid: GUID of the original message.
            limit: Max replies (default 50).
        """
        return service.get_chat_thread(thread_originator_guid, limit=limit)

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Get Chat Info',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_chat_info(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any],
        handle: str | None = None,
        chat_id: int | None = None,
    ) -> Chat:
        """Detailed chat metadata: participants, group name, counts.

        Args:
            handle: Phone/email to identify the chat.
            chat_id: Direct chat ROWID (alternative to handle).
        """
        return service.get_chat_info(handle=handle, chat_id=chat_id)


LOOKUP_CONTACT_BASE = """Search AddressBook contacts by name, phone, or email.

Matches names fuzzily (token_sort_ratio) and handles (phone/email) by exact
substring. Returns contacts deduplicated across all AddressBook sync sources,
with each result tagged by the sources it was merged from.

Args:
    query: Name (fuzzy), phone number, or email.
    limit: Max matches (default 10).
    source: Restrict to a single source display name (case-insensitive, e.g.,
        'Google', 'iCloud'). Raises if the source does not exist."""


def _format_sources_summary(sources: Sequence[ContactSource]) -> str:
    """Format AddressBook sources for injection into tool descriptions."""
    if not sources:
        return ''
    items = [f'{s.display_name} ({s.contact_count})' for s in sources]
    return f'**Available sources:** {", ".join(items)}'


def _inject_sources(base_docstring: str, summary: str) -> str:
    """Inject sources summary into tool description before Args block."""
    if not summary:
        return base_docstring

    if 'Args:' in base_docstring:
        args_pos = base_docstring.find('Args:')
        before_args = base_docstring[:args_pos].rstrip()
        after_args = base_docstring[args_pos:]
        return f'{before_args}\n\n{summary}\n\n{after_args}'
    return f'{base_docstring.rstrip()}\n\n{summary}'


@contextlib.asynccontextmanager
async def lifespan(
    server_instance: mcp.server.fastmcp.FastMCP,
) -> typing.AsyncIterator[None]:
    """Initialize DB, contacts, attachments, sender at startup."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )

    db = ChatDB()
    fda_available = True
    temp_dir_obj = tempfile.TemporaryDirectory(prefix='imessage-kit-')
    temp_path = Path(temp_dir_obj.name)

    try:
        db.connect()
    except FullDiskAccessError:
        fda_available = False
        root_app = detect_root_app()
        logger.exception(
            'Full Disk Access required. Grant it to: %s\n'
            '  System Settings → Privacy & Security → Full Disk Access\n'
            '  Then restart Claude Code.',
            root_app,
        )

    registry = SourceRegistry.discover()
    contacts = ContactResolver(sources=registry.sources)
    attachments = AttachmentHandler(temp_path)
    sender = MessageSender()

    state = ServerState(
        db=db,
        contacts=contacts,
        sources=registry.sources,
        attachments=attachments,
        sender=sender,
        temp_dir=temp_path,
        fda_available=fda_available,
    )
    service = IMessageService(state)
    register_tools(service)

    logger.info(
        'imessage-kit MCP server initialized (fda=%s, sources=%d)',
        fda_available,
        len(registry.sources),
    )

    yield

    db.close()
    temp_dir_obj.cleanup()
    logger.info('imessage-kit server shutdown complete')


server = mcp.server.fastmcp.FastMCP('imessage-kit', lifespan=lifespan)


def main() -> None:
    """Entry point for imessage-kit-mcp."""
    server.run()
