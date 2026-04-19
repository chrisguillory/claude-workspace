"""IMessageService — orchestration layer for all tool logic."""

from __future__ import annotations

__all__ = [
    'IMessageService',
    'ServerState',
]

import logging
import pathlib
import platform
import plistlib
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from imessage_kit.attachments import AttachmentHandler
from imessage_kit.contacts import ContactResolver
from imessage_kit.db import ChatDB, apple_ts_to_datetime
from imessage_kit.parser import extract_text
from imessage_kit.sender import MessageSender
from imessage_kit.system import detect_root_app
from imessage_kit.types import (
    AttachmentMeta,
    AttachmentSave,
    AttachmentView,
    Chat,
    Contact,
    ContactSource,
    DiagnosticResult,
    EditEvent,
    Message,
    Reaction,
    ReactionName,
    SendResult,
    SendService,
)

logger = logging.getLogger(__name__)

REACTION_LABELS: Mapping[int, ReactionName] = {
    2000: 'love',
    2001: 'like',
    2002: 'dislike',
    2003: 'laugh',
    2004: 'emphasize',
    2005: 'question',
    3000: 'love',
    3001: 'like',
    3002: 'dislike',
    3003: 'laugh',
    3004: 'emphasize',
    3005: 'question',
}


@dataclass(frozen=True)
class ServerState:
    """Shared state created in the MCP lifespan."""

    db: ChatDB
    contacts: ContactResolver
    sources: Sequence[ContactSource]
    """AddressBook sources discovered at startup, labeled by provider."""

    attachments: AttachmentHandler
    sender: MessageSender
    temp_dir: pathlib.Path
    fda_available: bool


class IMessageService:
    """All tool business logic. Thin MCP tools delegate here."""

    def __init__(self, state: ServerState) -> None:
        self._state = state
        self._handle_map: Mapping[int, str] | None = None
        self._resolved_handle_map: Mapping[int, str] | None = None

    @property
    def sources(self) -> Sequence[ContactSource]:
        """AddressBook sources discovered at startup."""
        return self._state.sources

    # -- Public tool methods --

    def list_chats(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        is_group: bool | None = None,
        active_since: datetime | None = None,
        handle: str | None = None,
    ) -> Sequence[Chat]:
        """List chats ordered by most recent message date."""
        rows = self._state.db.list_chats(
            limit=limit,
            offset=offset,
            is_group=is_group,
            active_since=active_since,
            handle=handle,
        )
        chats: list[Chat] = []
        for r in rows:
            is_grp = r['style'] == 43
            participants = self._state.db.get_chat_participants(r['chat_id'])
            participant_names = [self._state.contacts.resolve(p) for p in participants]

            # For 1:1 chats, display_name is the resolved contact name
            display_name = r['display_name']
            if not is_grp and not display_name and participants:
                display_name = self._state.contacts.resolve(participants[0])

            chats.append(
                Chat(
                    chat_id=r['chat_id'],
                    guid=r['guid'],
                    display_name=display_name,
                    handle=r['chat_identifier'] if not is_grp else None,
                    service=r['service_name'] or 'iMessage',
                    is_group=is_grp,
                    participants=participant_names,
                    last_message_text=_clean_text(r['last_message_text']),
                    last_message_date=apple_ts_to_datetime(r['last_message_date']),
                    unread_count=r['unread_count'],
                    message_count=r['message_count'],
                    attachment_count=r['attachment_count'],
                )
            )
        return chats

    def get_messages(
        self,
        *,
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
        """Get messages from a chat with cursor pagination and filtering."""
        resolved_chat_id = self._resolve_chat_id(handle, chat_id)
        rows = self._state.db.get_messages(
            resolved_chat_id,
            limit=limit,
            before_rowid=before_rowid,
            after_rowid=after_rowid,
            date_from=date_from,
            date_to=date_to,
            has_attachment=has_attachment,
            from_me=from_me,
        )
        return [self._row_to_message(r) for r in rows]

    def search_messages(
        self,
        query: str,
        *,
        limit: int = 25,
        offset: int = 0,
        handle: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        has_attachment: bool | None = None,
        from_me: bool | None = None,
    ) -> Sequence[Message]:
        """Two-pass search: SQL LIKE on text column, then parse attributedBody."""
        # Pass 1: text column search
        text_rows = self._state.db.search_messages(
            query,
            limit=limit,
            offset=offset,
            handle=handle,
            date_from=date_from,
            date_to=date_to,
            has_attachment=has_attachment,
            from_me=from_me,
        )
        results = [self._row_to_message(r) for r in text_rows]

        # Pass 2: attributedBody search (for messages with NULL text)
        if len(results) < limit:
            body_rows = self._state.db.get_attributed_body_candidates(
                limit=50000,
                handle=handle,
                date_from=date_from,
                date_to=date_to,
                from_me=from_me,
            )
            query_lower = query.lower()
            seen_rowids = {m.rowid for m in results}
            for r in body_rows:
                if r['rowid'] in seen_rowids:
                    continue
                text = extract_text(r['attributedBody'])
                if text and query_lower in text.lower():
                    results.append(self._row_to_message(r))
                    if len(results) >= limit:
                        break

        return results

    def get_attachment(self, attachment_id: int, mode: str) -> AttachmentView | AttachmentSave:
        """Retrieve an attachment by ID in the specified mode."""
        row = self._state.db.get_attachment_by_id(attachment_id)
        if row is None:
            msg = f'Attachment not found: {attachment_id}'
            raise ValueError(msg)

        path = self._state.attachments.resolve_path(row['filename'])
        if path is None:
            msg = f'Attachment has no file path: {attachment_id}'
            raise ValueError(msg)

        if mode == 'view':
            return self._state.attachments.view(path, row['mime_type'], attachment_id)
        return self._state.attachments.save(path, row['mime_type'], attachment_id)

    def list_attachments(
        self,
        *,
        handle: str | None = None,
        chat_id: int | None = None,
        limit: int = 20,
        offset: int = 0,
        mime_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        from_me: bool | None = None,
    ) -> Sequence[AttachmentMeta]:
        """List attachment metadata for a chat."""
        resolved_chat_id = self._resolve_chat_id(handle, chat_id)
        rows = self._state.db.list_attachments(
            resolved_chat_id,
            limit=limit,
            offset=offset,
            mime_type=mime_type,
            date_from=date_from,
            date_to=date_to,
            from_me=from_me,
        )
        result: list[AttachmentMeta] = []
        for r in rows:
            path = self._state.attachments.resolve_path(r['filename'])
            mime = r['mime_type'] or ''
            result.append(
                AttachmentMeta(
                    attachment_id=r['attachment_id'],
                    filename=r['filename'],
                    mime_type=r['mime_type'],
                    transfer_name=r['transfer_name'],
                    total_bytes=r['total_bytes'],
                    is_image=mime.startswith('image/'),
                    is_available=path.exists() if path else False,
                )
            )
        return result

    def send_message(
        self,
        text: str,
        *,
        handle: str | None,
        chat_guid: str | None,
        service: SendService,
        confirm: bool,
    ) -> SendResult:
        """Send a message. Requires confirm=True to actually send."""
        if not confirm:
            recipient = chat_guid or handle or ''
            return SendResult(
                success=False,
                recipient=recipient,
                service=None,
                error=f'Message preview — set confirm=true to send. To: {recipient}, Text: {text!r}',
            )
        return self._state.sender.send(text, handle=handle, chat_guid=chat_guid, service=service)

    def lookup_contact(
        self,
        query: str,
        *,
        limit: int = 10,
        source: str | None = None,
    ) -> Sequence[Contact]:
        """Search AddressBook by name, phone, or email.

        Args:
            query: Name (fuzzy), phone, or email.
            limit: Max matches to return.
            source: Filter to a source display name (e.g., 'Google', 'iCloud').
                Raises ValueError if the source does not exist.
        """
        results = self._state.contacts.lookup(query, source=source)
        return results[:limit]

    def get_unread(self) -> Sequence[Message]:
        """Get all unread incoming messages."""
        rows = self._state.db.get_unread_messages()
        return [self._row_to_message(r) for r in rows]

    def get_chat_thread(self, thread_originator_guid: str, *, limit: int = 50) -> Sequence[Message]:
        """Get all replies in an inline reply thread."""
        rows = self._state.db.get_thread_messages(thread_originator_guid, limit=limit)
        return [self._row_to_message(r) for r in rows]

    def get_chat_info(self, *, handle: str | None = None, chat_id: int | None = None) -> Chat:
        """Get detailed chat metadata."""
        resolved_chat_id = self._resolve_chat_id(handle, chat_id)
        row = self._state.db.get_chat_by_id(resolved_chat_id)
        if row is None:
            msg = f'Chat not found: {resolved_chat_id}'
            raise ValueError(msg)

        is_grp = row['style'] == 43
        participants = self._state.db.get_chat_participants(resolved_chat_id)
        participant_names = [self._state.contacts.resolve(p) for p in participants]

        display_name = row['display_name']
        if not is_grp and not display_name and participants:
            display_name = self._state.contacts.resolve(participants[0])

        # Aggregate counts
        msg_count = self._state.db.conn.execute(
            'SELECT COUNT(*) FROM chat_message_join WHERE chat_id = ?',
            (resolved_chat_id,),
        ).fetchone()[0]
        att_count = self._state.db.conn.execute(
            """
            SELECT COUNT(*) FROM message_attachment_join maj
            JOIN chat_message_join cmj ON cmj.message_id = maj.message_id
            WHERE cmj.chat_id = ?
        """,
            (resolved_chat_id,),
        ).fetchone()[0]
        unread = self._state.db.conn.execute(
            """
            SELECT COUNT(*) FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE cmj.chat_id = ? AND m.is_read = 0 AND m.is_from_me = 0
              AND m.associated_message_type = 0
        """,
            (resolved_chat_id,),
        ).fetchone()[0]

        # Last message
        last_msg = self._state.db.conn.execute(
            """
            SELECT m.text, m.date FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE cmj.chat_id = ?
            ORDER BY m.date DESC LIMIT 1
        """,
            (resolved_chat_id,),
        ).fetchone()

        return Chat(
            chat_id=resolved_chat_id,
            guid=row['guid'],
            display_name=display_name,
            handle=row['chat_identifier'] if not is_grp else None,
            service=row['service_name'] or 'iMessage',
            is_group=is_grp,
            participants=participant_names,
            last_message_text=_clean_text(last_msg['text']) if last_msg else None,
            last_message_date=apple_ts_to_datetime(last_msg['date']) if last_msg else None,
            unread_count=unread,
            message_count=msg_count,
            attachment_count=att_count,
        )

    def diagnose(self) -> DiagnosticResult:
        """Health check: FDA, DB, contacts, macOS version."""
        errors: list[str] = []

        # FDA / DB access
        fda = self._state.fda_available
        db_readable = fda
        db_path = str(self._state.db.path)
        message_count: int | None = None

        if fda:
            try:
                message_count = self._state.db.get_message_count()
            except sqlite3.Error as e:
                errors.append(f'Failed to count messages: {e}')
                db_readable = False

        # Contacts
        contacts_accessible = self._state.contacts.is_accessible
        contacts_count: int | None = None
        if contacts_accessible:
            contacts_count = self._state.contacts.contact_count

        # macOS version
        macos_version = platform.mac_ver()[0] or 'unknown'

        if not fda:
            root_app = detect_root_app()
            errors.append(
                f'Full Disk Access required. '
                f'Grant it to: {root_app} '
                f'(System Settings → Privacy & Security → Full Disk Access). '
                f'Run: open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"'
            )

        return DiagnosticResult(
            full_disk_access=fda,
            db_path=db_path,
            db_readable=db_readable,
            message_count=message_count,
            contacts_sources=self._state.sources,
            contacts_accessible=contacts_accessible,
            contacts_count=contacts_count,
            macos_version=macos_version,
            errors=errors,
        )

    # -- Private helpers: handle resolution --

    def _get_handle_map(self) -> Mapping[int, str]:
        """Lazily load and cache the handle ROWID → identifier map."""
        if self._handle_map is None:
            self._handle_map = self._state.db.get_handle_map()
        return self._handle_map

    def _get_resolved_handle_map(self) -> Mapping[int, str]:
        """Lazily load and cache the handle ROWID → display name map."""
        if self._resolved_handle_map is None:
            raw = self._get_handle_map()
            self._resolved_handle_map = self._state.contacts.resolve_handle_map(raw)
        return self._resolved_handle_map

    def _resolve_sender(self, row: sqlite3.Row) -> str:
        """Resolve a message row's sender to a display name."""
        if row['is_from_me']:
            return 'You'
        handle_id = row['handle_id']
        resolved = self._get_resolved_handle_map()
        return resolved.get(handle_id, self._get_handle_map().get(handle_id, 'Unknown'))

    def _resolve_chat_id(self, handle: str | None, chat_id: int | None) -> int:
        """Resolve a handle or chat_id to a definitive chat ROWID."""
        if chat_id is not None:
            return chat_id
        if handle is None:
            msg = 'Either handle or chat_id is required.'
            raise ValueError(msg)
        chat = self._state.db.get_chat_for_handle(handle)
        if chat is None:
            msg = f'No chat found for handle: {handle}'
            raise ValueError(msg)
        return cast(int, chat['chat_id'])

    # -- Private helpers: row conversion --

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Convert a database row to a Message model."""
        # Text extraction: text column first, attributedBody fallback
        text = row['text']
        if text is None and row['attributedBody'] is not None:
            text = extract_text(row['attributedBody'])
        text = _clean_text(text)

        # Edited message handling
        is_edited = bool(row['date_edited'])
        edit_history: Sequence[EditEvent] | None = None
        is_retracted = False

        if is_edited and row['message_summary_info']:
            edit_history, is_retracted = self._parse_edit_history(row['message_summary_info'])

        # Attachments
        attachments: list[AttachmentMeta] = []
        if row['cache_has_attachments']:
            att_rows = self._state.db.get_attachments_for_message(row['rowid'])
            for att in att_rows:
                path = self._state.attachments.resolve_path(att['filename'])
                mime = att['mime_type'] or ''
                attachments.append(
                    AttachmentMeta(
                        attachment_id=att['attachment_id'],
                        filename=att['filename'],
                        mime_type=att['mime_type'],
                        transfer_name=att['transfer_name'],
                        total_bytes=att['total_bytes'],
                        is_image=mime.startswith('image/'),
                        is_available=path.exists() if path else False,
                    )
                )

        # Reactions
        reactions = self._get_reactions(row['guid'])

        timestamp = apple_ts_to_datetime(row['date'])
        if timestamp is None:
            msg = f'Message rowid={row["rowid"]} has NULL/zero date; messages must have a timestamp.'
            raise ValueError(msg)

        return Message(
            rowid=row['rowid'],
            guid=row['guid'],
            text=text,
            sender=self._resolve_sender(row),
            is_from_me=bool(row['is_from_me']),
            timestamp=timestamp,
            date_read=apple_ts_to_datetime(row['date_read']),
            service=row['service'] or 'iMessage',
            is_edited=is_edited,
            edit_history=edit_history,
            is_retracted=is_retracted,
            attachments=attachments,
            reactions=reactions,
            thread_originator_guid=row['thread_originator_guid'],
        )

    def _get_reactions(self, message_guid: str) -> Sequence[Reaction]:
        """Fetch and resolve reactions for a message."""
        rows = self._state.db.get_reactions_for_message(message_guid)
        reactions: list[Reaction] = []
        for r in rows:
            rtype = r['associated_message_type']
            label = REACTION_LABELS.get(rtype)
            if label is None:
                continue
            if r['is_from_me']:
                sender = 'You'
            else:
                handle_id = r['handle_id']
                resolved = self._get_resolved_handle_map()
                sender = resolved.get(handle_id, self._get_handle_map().get(handle_id, 'Unknown'))
            reactions.append(
                Reaction(
                    reaction_type=rtype,
                    reaction_label=label,
                    sender=sender,
                    is_from_me=bool(r['is_from_me']),
                )
            )
        return reactions

    def _parse_edit_history(self, blob: bytes) -> tuple[Sequence[EditEvent] | None, bool]:
        """Parse message_summary_info plist for edit history and retractions."""
        try:
            info = plistlib.loads(blob)
        except ValueError:
            logger.warning('Malformed message_summary_info plist', exc_info=True)
            return None, False

        events: list[EditEvent] = []
        is_retracted = bool(info.get('rp'))

        ec = info.get('ec', {})
        for edits in ec.values():
            if not isinstance(edits, list):
                continue
            for edit in edits:
                ts = apple_ts_to_datetime(edit.get('d'))
                text_blob = edit.get('t')
                text = extract_text(text_blob) if isinstance(text_blob, bytes) else None
                if ts and text:
                    events.append(EditEvent(timestamp=ts, text=text))

        return events if events else None, is_retracted


def _clean_text(text: str | None) -> str | None:
    """Strip U+FFFC (attachment placeholder) and whitespace. Return None if empty."""
    if text is None:
        return None
    cleaned = text.replace('\ufffc', '').strip()
    return cleaned or None
