"""Read-only SQLite access to macOS iMessage chat.db.

Handles connection lifecycle, Apple timestamp conversion, and all
queries against the message, chat, handle, and attachment tables.
"""

from __future__ import annotations

__all__ = [
    'ChatDB',
    'FullDiskAccessError',
    'apple_ts_to_datetime',
]

import logging
import pathlib
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

logger = logging.getLogger(__name__)

APPLE_EPOCH_OFFSET = 978307200  # seconds from Unix epoch (1970) to Apple epoch (2001)
DB_PATH = pathlib.Path.home() / 'Library' / 'Messages' / 'chat.db'


class FullDiskAccessError(Exception):
    """chat.db is not readable — Full Disk Access not granted."""

    def __init__(self, db_path: pathlib.Path, cause: Exception, root_app: str | None = None) -> None:
        self.db_path = db_path
        self.cause = cause
        self.root_app = root_app
        hint = f'Add: {root_app}' if root_app else 'Check System Settings manually'
        super().__init__(
            f'Cannot read {db_path}: {cause}\n\n'
            f'Full Disk Access required.\n'
            f'Fix: System Settings → Privacy & Security → Full Disk Access\n'
            f'{hint}\n'
            f'Then restart Claude Code.'
        )


class ChatDB:
    """Read-only connection to iMessage chat.db."""

    def __init__(self, db_path: pathlib.Path | None = None) -> None:
        self._path = db_path or DB_PATH
        self._conn: sqlite3.Connection | None = None

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def connect(self) -> sqlite3.Connection:
        """Open a read-only connection. Raises FullDiskAccessError if FDA is not granted."""
        if self._conn is not None:
            return self._conn

        if not self._path.exists():
            raise FileNotFoundError(f'{self._path} not found. Messages.app may not have been used on this Mac.')

        try:
            uri = f'file:{self._path}?mode=ro'
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA busy_timeout = 5000')
            conn.execute('SELECT 1 FROM message LIMIT 1')
        except sqlite3.OperationalError as e:
            raise FullDiskAccessError(self._path, e) from e

        self._conn = conn
        logger.info('Connected to %s', self._path)
        return conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info('Closed connection to %s', self._path)

    @property
    def conn(self) -> sqlite3.Connection:
        """Get the active connection, connecting if needed."""
        if self._conn is None:
            self.connect()
        assert self._conn is not None  # noqa: S101 — post-connect invariant
        return self._conn

    # -- Queries: Chats --

    def list_chats(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        is_group: bool | None = None,
        active_since: datetime | None = None,
        handle: str | None = None,
    ) -> Sequence[sqlite3.Row]:
        """List chats ordered by most recent message date."""
        conditions = []
        params: list[object] = []

        if is_group is True:
            conditions.append('c.style = 43')  # group chat style
        elif is_group is False:
            conditions.append('c.style != 43')

        if active_since is not None:
            apple_ns = int((active_since.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)
            conditions.append('last_msg.date > ?')
            params.append(apple_ns)

        if handle is not None:
            conditions.append("""
                c.ROWID IN (
                    SELECT chj.chat_id FROM chat_handle_join chj
                    JOIN handle h ON chj.handle_id = h.ROWID
                    WHERE h.id = ?
                )
            """)
            params.append(handle)

        where = f'WHERE {" AND ".join(conditions)}' if conditions else ''

        sql = f"""
            SELECT
                c.ROWID as chat_id,
                c.guid,
                c.display_name,
                c.chat_identifier,
                c.service_name,
                c.style,
                (SELECT m.date FROM message m
                 JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                 WHERE cmj.chat_id = c.ROWID
                 ORDER BY m.date DESC LIMIT 1) as last_message_date,
                (SELECT m.text FROM message m
                 JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                 WHERE cmj.chat_id = c.ROWID
                 ORDER BY m.date DESC LIMIT 1) as last_message_text,
                (SELECT COUNT(*) FROM message m
                 JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                 WHERE cmj.chat_id = c.ROWID) as message_count,
                (SELECT COUNT(*) FROM message m
                 JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                 JOIN message_attachment_join maj ON maj.message_id = m.ROWID
                 WHERE cmj.chat_id = c.ROWID) as attachment_count,
                (SELECT COUNT(*) FROM message m
                 JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                 WHERE cmj.chat_id = c.ROWID
                   AND m.is_read = 0 AND m.is_from_me = 0
                   AND m.associated_message_type = 0) as unread_count
            FROM chat c
            LEFT JOIN (
                SELECT cmj.chat_id, MAX(m.date) as date
                FROM message m
                JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                GROUP BY cmj.chat_id
            ) last_msg ON last_msg.chat_id = c.ROWID
            {where}
            ORDER BY last_message_date DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        return self.conn.execute(sql, params).fetchall()

    def get_chat_for_handle(self, handle: str) -> sqlite3.Row | None:
        """Find the 1:1 chat for a specific handle (phone/email).

        Prefers 1:1 chats (style != 43) over group chats. A handle may
        appear in many group chats but typically has one 1:1 chat.
        """
        return cast(
            'sqlite3.Row | None',
            self.conn.execute(
                """
                SELECT c.ROWID as chat_id, c.guid, c.display_name,
                       c.chat_identifier, c.service_name, c.style
                FROM chat c
                JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
                JOIN handle h ON chj.handle_id = h.ROWID
                WHERE h.id = ?
                ORDER BY (c.style != 43) DESC, c.ROWID DESC
                LIMIT 1
            """,
                (handle,),
            ).fetchone(),
        )

    def get_chat_by_id(self, chat_id: int) -> sqlite3.Row | None:
        """Get a chat by its ROWID."""
        return cast(
            'sqlite3.Row | None',
            self.conn.execute(
                """
                SELECT ROWID as chat_id, guid, display_name,
                       chat_identifier, service_name, style
                FROM chat WHERE ROWID = ?
            """,
                (chat_id,),
            ).fetchone(),
        )

    def get_chat_participants(self, chat_id: int) -> Sequence[str]:
        """Get handle identifiers for all participants in a chat."""
        rows = self.conn.execute(
            """
            SELECT h.id
            FROM chat_handle_join chj
            JOIN handle h ON chj.handle_id = h.ROWID
            WHERE chj.chat_id = ?
        """,
            (chat_id,),
        ).fetchall()
        return [r['id'] for r in rows]

    # -- Queries: Messages --

    def get_messages(
        self,
        chat_id: int,
        *,
        limit: int = 50,
        before_rowid: int | None = None,
        after_rowid: int | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        has_attachment: bool | None = None,
        from_me: bool | None = None,
    ) -> Sequence[sqlite3.Row]:
        """Get messages in a chat with cursor pagination and filtering."""
        conditions = ['cmj.chat_id = ?', 'm.associated_message_type = 0']
        params: list[object] = [chat_id]

        if before_rowid is not None:
            conditions.append('m.ROWID < ?')
            params.append(before_rowid)
        elif after_rowid is not None:
            conditions.append('m.ROWID > ?')
            params.append(after_rowid)

        if date_from is not None:
            apple_ns = int((date_from.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)
            conditions.append('m.date >= ?')
            params.append(apple_ns)

        if date_to is not None:
            apple_ns = int((date_to.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)
            conditions.append('m.date <= ?')
            params.append(apple_ns)

        if has_attachment is True:
            conditions.append('m.cache_has_attachments = 1')
        elif has_attachment is False:
            conditions.append('m.cache_has_attachments = 0')

        if from_me is True:
            conditions.append('m.is_from_me = 1')
        elif from_me is False:
            conditions.append('m.is_from_me = 0')

        where = f'WHERE {" AND ".join(conditions)}'

        return self.conn.execute(
            f"""
            SELECT
                m.ROWID as rowid, m.guid, m.text, m.attributedBody,
                m.is_from_me, m.date, m.date_read, m.date_delivered,
                m.service, m.handle_id, m.date_edited,
                m.message_summary_info, m.cache_has_attachments,
                m.thread_originator_guid, m.thread_originator_part
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            {where}
            ORDER BY m.date DESC
            LIMIT ?
        """,
            [*params, limit],
        ).fetchall()

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
    ) -> Sequence[sqlite3.Row]:
        """Search messages by text content (text column only, pass 1 of two-pass search)."""
        conditions = ['m.text LIKE ?', 'm.associated_message_type = 0']
        params: list[object] = [f'%{query}%']

        if handle is not None:
            conditions.append("""
                m.ROWID IN (
                    SELECT cmj.message_id FROM chat_message_join cmj
                    JOIN chat_handle_join chj ON chj.chat_id = cmj.chat_id
                    JOIN handle h ON chj.handle_id = h.ROWID
                    WHERE h.id = ?
                )
            """)
            params.append(handle)

        if date_from is not None:
            apple_ns = int((date_from.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)
            conditions.append('m.date >= ?')
            params.append(apple_ns)

        if date_to is not None:
            apple_ns = int((date_to.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)
            conditions.append('m.date <= ?')
            params.append(apple_ns)

        if has_attachment is True:
            conditions.append('m.cache_has_attachments = 1')
        elif has_attachment is False:
            conditions.append('m.cache_has_attachments = 0')

        if from_me is True:
            conditions.append('m.is_from_me = 1')
        elif from_me is False:
            conditions.append('m.is_from_me = 0')

        where = f'WHERE {" AND ".join(conditions)}'

        return self.conn.execute(
            f"""
            SELECT
                m.ROWID as rowid, m.guid, m.text, m.attributedBody,
                m.is_from_me, m.date, m.date_read, m.date_delivered,
                m.service, m.handle_id, m.date_edited,
                m.message_summary_info, m.cache_has_attachments,
                m.thread_originator_guid, m.thread_originator_part
            FROM message m
            {where}
            ORDER BY m.date DESC
            LIMIT ? OFFSET ?
        """,
            [*params, limit, offset],
        ).fetchall()

    def get_attributed_body_candidates(
        self,
        *,
        limit: int = 50000,
        handle: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        from_me: bool | None = None,
    ) -> Sequence[sqlite3.Row]:
        """Get messages with attributedBody but NULL text (pass 2 of two-pass search)."""
        conditions = ['m.text IS NULL', 'm.attributedBody IS NOT NULL', 'm.associated_message_type = 0']
        params: list[object] = []

        if handle is not None:
            conditions.append("""
                m.ROWID IN (
                    SELECT cmj.message_id FROM chat_message_join cmj
                    JOIN chat_handle_join chj ON chj.chat_id = cmj.chat_id
                    JOIN handle h ON chj.handle_id = h.ROWID
                    WHERE h.id = ?
                )
            """)
            params.append(handle)

        if date_from is not None:
            apple_ns = int((date_from.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)
            conditions.append('m.date >= ?')
            params.append(apple_ns)

        if date_to is not None:
            apple_ns = int((date_to.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)
            conditions.append('m.date <= ?')
            params.append(apple_ns)

        if from_me is True:
            conditions.append('m.is_from_me = 1')
        elif from_me is False:
            conditions.append('m.is_from_me = 0')

        where = f'WHERE {" AND ".join(conditions)}'

        return self.conn.execute(
            f"""
            SELECT
                m.ROWID as rowid, m.guid, m.text, m.attributedBody,
                m.is_from_me, m.date, m.date_read, m.date_delivered,
                m.service, m.handle_id, m.date_edited,
                m.message_summary_info, m.cache_has_attachments,
                m.thread_originator_guid, m.thread_originator_part
            FROM message m
            {where}
            ORDER BY m.date DESC
            LIMIT ?
        """,
            [*params, limit],
        ).fetchall()

    def get_unread_messages(self) -> Sequence[sqlite3.Row]:
        """Get all unread incoming messages, excluding tapbacks."""
        return self.conn.execute("""
            SELECT
                m.ROWID as rowid, m.guid, m.text, m.attributedBody,
                m.is_from_me, m.date, m.date_read, m.date_delivered,
                m.service, m.handle_id, m.date_edited,
                m.message_summary_info, m.cache_has_attachments,
                m.thread_originator_guid, m.thread_originator_part,
                cmj.chat_id
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE m.is_read = 0 AND m.is_from_me = 0
              AND m.associated_message_type = 0
            ORDER BY m.date DESC
        """).fetchall()

    # -- Queries: Attachments --

    def get_attachments_for_message(self, message_rowid: int) -> Sequence[sqlite3.Row]:
        """Get attachments linked to a specific message."""
        return self.conn.execute(
            """
            SELECT a.ROWID as attachment_id, a.filename, a.mime_type,
                   a.transfer_name, a.total_bytes
            FROM attachment a
            JOIN message_attachment_join maj ON maj.attachment_id = a.ROWID
            WHERE maj.message_id = ?
        """,
            (message_rowid,),
        ).fetchall()

    def list_attachments(
        self,
        chat_id: int,
        *,
        limit: int = 20,
        offset: int = 0,
        mime_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        from_me: bool | None = None,
    ) -> Sequence[sqlite3.Row]:
        """List attachments in a chat with filtering."""
        conditions = ['cmj.chat_id = ?']
        params: list[object] = [chat_id]

        if mime_type is not None:
            conditions.append('a.mime_type LIKE ?')
            params.append(f'{mime_type}%')

        if date_from is not None:
            apple_ns = int((date_from.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)
            conditions.append('m.date >= ?')
            params.append(apple_ns)

        if date_to is not None:
            apple_ns = int((date_to.timestamp() - APPLE_EPOCH_OFFSET) * 1_000_000_000)
            conditions.append('m.date <= ?')
            params.append(apple_ns)

        if from_me is True:
            conditions.append('m.is_from_me = 1')
        elif from_me is False:
            conditions.append('m.is_from_me = 0')

        where = f'WHERE {" AND ".join(conditions)}'

        return self.conn.execute(
            f"""
            SELECT a.ROWID as attachment_id, a.filename, a.mime_type,
                   a.transfer_name, a.total_bytes,
                   m.ROWID as message_rowid, m.date, m.is_from_me, m.handle_id
            FROM attachment a
            JOIN message_attachment_join maj ON maj.attachment_id = a.ROWID
            JOIN message m ON maj.message_id = m.ROWID
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            {where}
            ORDER BY m.date DESC
            LIMIT ? OFFSET ?
        """,
            [*params, limit, offset],
        ).fetchall()

    def get_attachment_by_id(self, attachment_id: int) -> sqlite3.Row | None:
        """Get a single attachment by its ROWID."""
        return cast(
            'sqlite3.Row | None',
            self.conn.execute(
                """
                SELECT ROWID as attachment_id, filename, mime_type,
                       transfer_name, total_bytes
                FROM attachment WHERE ROWID = ?
            """,
                (attachment_id,),
            ).fetchone(),
        )

    # -- Queries: Reactions --

    def get_reactions_for_message(self, message_guid: str) -> Sequence[sqlite3.Row]:
        """Get tapback reactions targeting a specific message."""
        return self.conn.execute(
            """
            SELECT m.associated_message_type, m.associated_message_guid,
                   m.is_from_me, m.handle_id
            FROM message m
            WHERE m.associated_message_guid LIKE ?
              AND m.associated_message_type BETWEEN 2000 AND 3006
        """,
            (f'%{message_guid}',),
        ).fetchall()

    # -- Queries: Threads --

    def get_thread_messages(
        self,
        thread_originator_guid: str,
        *,
        limit: int = 50,
    ) -> Sequence[sqlite3.Row]:
        """Get all replies in an inline thread."""
        return self.conn.execute(
            """
            SELECT
                m.ROWID as rowid, m.guid, m.text, m.attributedBody,
                m.is_from_me, m.date, m.date_read, m.date_delivered,
                m.service, m.handle_id, m.date_edited,
                m.message_summary_info, m.cache_has_attachments,
                m.thread_originator_guid, m.thread_originator_part
            FROM message m
            WHERE m.thread_originator_guid = ?
            ORDER BY m.date ASC
            LIMIT ?
        """,
            (thread_originator_guid, limit),
        ).fetchall()

    # -- Queries: Handles --

    def get_handle_map(self) -> Mapping[int, str]:
        """Map handle ROWIDs to identifier strings."""
        rows = self.conn.execute('SELECT ROWID, id FROM handle').fetchall()
        return {r['ROWID']: r['id'] for r in rows}

    # -- Queries: Diagnostics --

    def get_message_count(self) -> int:
        """Total message count."""
        return cast(int, self.conn.execute('SELECT COUNT(*) FROM message').fetchone()[0])


def apple_ts_to_datetime(ns: int | None) -> datetime | None:
    """Convert Apple nanosecond timestamp to UTC-aware datetime.

    Apple uses nanoseconds since 2001-01-01 00:00:00 UTC.
    Values < 1e12 are treated as seconds (older columns).
    """
    if ns is None or ns == 0:
        return None
    unix_ts: float = ns + APPLE_EPOCH_OFFSET if ns < 1e12 else (ns / 1_000_000_000) + APPLE_EPOCH_OFFSET
    return datetime.fromtimestamp(unix_ts, tz=UTC)
