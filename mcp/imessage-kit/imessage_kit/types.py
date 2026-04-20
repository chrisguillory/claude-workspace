"""Pydantic models for iMessage entities."""

from __future__ import annotations

__all__ = [
    'AttachmentMeta',
    'AttachmentMode',
    'AttachmentSave',
    'AttachmentView',
    'Chat',
    'Contact',
    'ContactSource',
    'ContactSourceKind',
    'DiagnosticResult',
    'EditEvent',
    'Message',
    'MessageService',
    'Reaction',
    'ReactionName',
    'SendResult',
]

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from cc_lib.schemas.base import ClosedModel
from cc_lib.types import JsonDatetime

type MessageService = Literal['iMessage', 'SMS', 'RCS', 'iMessageLite']
type ReactionName = Literal['love', 'like', 'dislike', 'laugh', 'emphasize', 'question']
type AttachmentMode = Literal['view', 'save']
type ContactSourceKind = Literal['Google', 'iCloud']

# Terminal states of a send as observed in chat.db, after polling post-dispatch.
# AppleScript `send` returning success is NOT a reliable delivery signal on its own —
# Messages.app writes a row into chat.db whose `is_sent`, `is_delivered`, `error`,
# `date_delivered`, `date_read` columns (and the attachment `transfer_state`) are
# the authoritative source of truth.
type DeliveryStatus = Literal[
    # Terminal success states (polling stops as soon as any of these is seen):
    'read',  # delivered AND recipient has opened it (message.date_read > 0)
    'delivered',  # recipient device acknowledged (is_delivered=1); not yet read
    'sent',  # dispatched to APNs (is_sent=1); recipient device may be offline.
    #        # APNs holds the message and delivers when the recipient comes back online.
    #        # From our perspective this is terminal — we've done our job.
    # Terminal failure state:
    'failed',  # message.error != 0 or any attachment.transfer_state == 6
    # Non-terminal "we don't know" states — polling ran out of budget:
    'pending',  # row exists but no terminal flag observed within the poll window
    'timeout',  # we saw no resolution at all within the poll window
    'not_found',  # no matching message row ever appeared in chat.db (AppleScript lie)
]

# RCS is intentionally NOT exposed as a send target. Apple's Messages.app dictionary
# enumerates RCS as a service type, but selecting it in AppleScript does not give us
# deterministic control over the wire protocol — RCS routing is decided by the paired
# iPhone via Text Message Forwarding when conditions are met. Choosing 'SMS' here may
# transparently send as RCS to RCS-capable Android recipients with no programmatic
# signal back. Re-evaluate if Apple ever documents an RCS send path with deterministic
# behavior or exposes one in chat.db's send-side schema.
type SendService = Literal['auto', 'iMessage', 'SMS']


class ContactSource(ClosedModel):
    """An AddressBook sync source with account identification."""

    source_uuid: str
    """Filesystem UUID. e.g., '14A0861D-E59E-4051-9ACA-D05A487C4384'."""

    kind: ContactSourceKind
    """Provider family. Controls filter-matching and dedup semantics."""

    display_name: str
    """Human-readable label. e.g., 'Google', 'iCloud', 'iCloud (CloudKit)'."""

    email: str | None
    """Account email. e.g., 'user@example.com'. None for CloudKit (no CardDAV mapping)."""

    contact_count: int
    """Total ZABCDRECORD rows in this source's database."""

    db_path: Path
    """Absolute path to the AddressBook-v22.abcddb file."""


class Contact(ClosedModel):
    """A person from macOS AddressBook. May have multiple handles."""

    # Identity
    display_name: str | None
    """e.g., 'Jane Doe'. None if not in AddressBook."""

    first_name: str | None
    last_name: str | None

    # Handles
    phone_numbers: Sequence[str]
    """e.g., ['+15555550100']."""

    emails: Sequence[str]
    """e.g., ['jane@example.com']."""

    # Provenance
    sources: Sequence[str]
    """Display names of sources this contact was merged from. e.g., ['Google', 'iCloud']."""


class AttachmentMeta(ClosedModel):
    """Metadata for a message attachment (no file content)."""

    # Identity
    attachment_id: int
    """ROWID — pass to get_attachment."""

    # Details
    filename: str | None
    """e.g., '~/Library/Messages/Attachments/ab/14/att_42/IMG_0001.jpeg'."""

    mime_type: str | None
    """e.g., 'image/heic', 'application/pdf'."""

    transfer_name: str | None
    """Original filename. e.g., 'IMG_0001.jpeg'."""

    total_bytes: int | None
    is_image: bool
    """True if mime_type starts with 'image/'."""

    is_available: bool
    """True if the file exists on disk. False for iCloud-offloaded."""


class Reaction(ClosedModel):
    """A tapback reaction on a message."""

    reaction_type: int
    """2000-3006 code. e.g., 2000 = love, 2001 = like."""

    reaction_label: ReactionName
    """e.g., 'love', 'like', 'laugh'."""

    sender: str
    """Resolved display name or raw handle."""

    is_from_me: bool


class EditEvent(ClosedModel):
    """A single edit in a message's edit history."""

    timestamp: JsonDatetime
    text: str
    """The message text at this edit point."""


class Message(ClosedModel):
    """A single iMessage/SMS message."""

    # Identity
    rowid: int
    """For cursor pagination (before_rowid / after_rowid)."""

    guid: str
    """Unique message identifier."""

    # Content
    text: str | None
    """Extracted from text column or attributedBody. None for attachment-only messages."""

    # Sender
    sender: str
    """Resolved display name or raw handle. 'You' when is_from_me=True."""

    is_from_me: bool

    # Timestamps
    timestamp: JsonDatetime
    date_read: JsonDatetime | None

    # Classification
    service: MessageService

    # Edit state
    is_edited: bool
    edit_history: Sequence[EditEvent] | None
    """Populated when is_edited=True. Chronological: first entry is original."""

    is_retracted: bool
    """True if message was unsent by the sender."""

    # Related entities
    attachments: Sequence[AttachmentMeta]
    reactions: Sequence[Reaction]
    thread_originator_guid: str | None
    """Non-None if this is an inline reply."""


class Chat(ClosedModel):
    """A chat thread (1:1 or group) from chat.db."""

    # Identity
    chat_id: int
    """ROWID."""

    guid: str
    """e.g., 'iMessage;-;+15555550100' or 'iMessage;+;chat123456789'."""

    # Display
    display_name: str | None
    """Group name, or resolved contact name for 1:1."""

    handle: str | None
    """Phone/email for 1:1 chats, None for groups."""

    # Classification
    service: MessageService
    is_group: bool

    # Participants
    participants: Sequence[str]
    """Resolved display names."""

    # Activity
    last_message_text: str | None
    last_message_date: JsonDatetime | None
    unread_count: int
    message_count: int | None
    attachment_count: int | None


class AttachmentView(ClosedModel):
    """Image content for Claude vision (mode='view'). HEIC auto-converted to JPEG."""

    attachment_id: int
    filename: str | None
    original_mime_type: str | None
    """e.g., 'image/heic'."""

    delivered_mime_type: str
    """After conversion. e.g., 'image/jpeg'."""

    total_bytes: int | None
    base64_data: str
    """Base64-encoded image data."""


class AttachmentSave(ClosedModel):
    """Native file saved to temp dir (mode='save'). No conversion applied."""

    attachment_id: int
    filename: str | None
    mime_type: str | None
    total_bytes: int | None
    saved_path: str
    """Absolute path to temp file in native format."""


class DiagnosticResult(ClosedModel):
    """Health check results from diagnose tool."""

    full_disk_access: bool
    db_path: str
    db_readable: bool
    message_count: int | None
    contacts_accessible: bool
    contacts_count: int | None
    contacts_sources: Sequence[ContactSource]
    """AddressBook sources discovered and labeled."""

    macos_version: str
    errors: Sequence[str]


class SendResult(ClosedModel):
    """Result of sending a message and/or attachments.

    `success` is True only when the post-dispatch poll of chat.db observes a
    terminal delivery state of 'delivered' or 'read'. AppleScript accepting
    the send is surfaced separately in `applescript_succeeded` — it is not a
    reliable delivery signal on its own (observed in the wild: AppleScript
    returns success while the attachment transfer fails silently).
    """

    success: bool
    recipient: str
    """Handle or chat_guid the message was directed to."""

    service: MessageService | None
    """May be None for auto-routed sends where Messages.app chose the service."""

    parts_sent: int
    """How many parts were dispatched via AppleScript. total = 1 (text, if any) + N (attachments).
    On fail-fast: reflects parts attempted before the failure. Independent of actual delivery."""

    # Post-dispatch chat.db provenance
    delivery_status: DeliveryStatus
    message_rowid: int | None
    """chat.db message.ROWID of the row we identified as ours. None if not_found."""

    message_guid: str | None
    """chat.db message.guid. None if not_found."""

    error_code: int | None
    """chat.db message.error column value (non-zero = failure). None if not_found."""

    attachment_transfer_states: Sequence[int]
    """One entry per sent attachment, in order. 5 = delivered, 6 = failed.
    Empty for text-only sends. Missing entries if some attachments never registered in chat.db."""

    # AppleScript hop provenance (separate from delivery truth)
    applescript_succeeded: bool
    """True if every osascript invocation exited 0. Does NOT imply delivery."""

    applescript_error: str | None
    """osascript stderr from the first failed invocation, if any."""

    error: str | None
    """Human-readable summary of the terminal outcome. None on full success."""
