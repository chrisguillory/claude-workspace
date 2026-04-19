"""Send iMessages and SMS via AppleScript."""

from __future__ import annotations

__all__ = [
    'MessageSender',
]

import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence, Set
from pathlib import Path
from typing import get_args

from imessage_kit.types import SendResult, SendService

logger = logging.getLogger(__name__)


class MessageSender:
    """Send messages via osascript."""

    # Handle: phone (E.164 or local with separators), email, or numeric short code.
    # Chars excluded: quote, backslash, newline — anything that could break out of
    # the AppleScript string context "{handle}". Restrictive whitelist over blacklist.
    HANDLE_RE = re.compile(r'^(\+?[\d\-() .]+|[\w.+\-]+@[\w.\-]+\.[a-zA-Z]{2,})$')

    # chat_guid: shape from chat.db, e.g. 'any;-;voyanquist@gmail.com' or
    # 'any;+;3c9967c3251c4a88b8783fa7e7ab9557'. Alphanumerics + chat-id separators.
    CHAT_GUID_RE = re.compile(r'^[\w;.+\-@]+$')

    # Attachment path: any POSIX path except chars that would break out of the
    # AppleScript "POSIX file "{path}"" string context. Relative paths are
    # resolved to absolute via Path.resolve() before this check.
    ATTACHMENT_PATH_RE = re.compile(r'^[^"\\\n]+$')

    # SendService is a PEP 695 `type` alias (TypeAliasType), so we unwrap via
    # __value__ before get_args — direct get_args(SendService) returns empty.
    VALID_SERVICES: Set[str] = set(get_args(SendService.__value__))

    def send(
        self,
        text: str,
        *,
        handle: str | None,
        chat_guid: str | None,
        service: SendService,
        attachments: Sequence[str] | None = None,
    ) -> SendResult:
        """Send text and/or attachments to a handle (1:1) or chat GUID (group).

        Each part (text + each attachment) is dispatched as a separate AppleScript
        send, so they arrive as separate Messages bubbles. Messages.app on the
        receiving end may visually group consecutive images.

        Fail-fast semantics: if any part fails, subsequent parts are not attempted
        and the returned SendResult.parts_sent reflects how much of the payload
        was already dispatched.

        Uses file-based message passing (tempfile + UTF-8 read) for the text body
        to avoid AppleScript quote escaping. Attachment paths are validated and
        interpolated directly.

        Args:
            text: Message body. Pass empty string for attachments-only send.
            handle: Phone number or email for 1:1 chats.
            chat_guid: Chat GUID for group chats (from list_chats output).
            service: 'auto' (Messages.app decides), 'iMessage', or 'SMS'.
                Only applies to 1:1 sends.
            attachments: Ordered list of file paths (absolute or relative).
                Relative paths are resolved against the current working directory.
                Each must exist and be readable.
        """
        if not handle and not chat_guid:
            return SendResult(
                success=False,
                recipient='',
                service=None,
                parts_sent=0,
                error='Either handle or chat_guid is required.',
            )

        if not text and not attachments:
            return SendResult(
                success=False,
                recipient=chat_guid or handle or '',
                service=None,
                parts_sent=0,
                error='Must provide text or at least one attachment.',
            )

        # Validate before interpolating into AppleScript. Handle, chat_guid, service,
        # and attachment paths are interpolated directly (only the text body is
        # tempfile-safe), so we whitelist their shape to prevent AppleScript injection.
        if handle is not None and not self.HANDLE_RE.match(handle):
            msg = f'Invalid handle shape: {handle!r}. Expected phone, email, or short code.'
            raise ValueError(msg)
        if chat_guid is not None and not self.CHAT_GUID_RE.match(chat_guid):
            msg = f'Invalid chat_guid shape: {chat_guid!r}. Expected chat.db GUID format.'
            raise ValueError(msg)
        if service not in self.VALID_SERVICES:
            msg = f'Invalid service: {service!r}. Expected one of {sorted(self.VALID_SERVICES)}.'
            raise ValueError(msg)

        resolved_attachments = self._resolve_attachments(attachments or [])

        recipient = chat_guid or handle or ''
        parts_sent = 0

        if text:
            ok, err = self._dispatch_text(text, handle=handle, chat_guid=chat_guid, service=service)
            if not ok:
                logger.warning('Text send failed to %s (service=%s): %s', recipient, service, err)
                return SendResult(
                    success=False,
                    recipient=recipient,
                    service=None,
                    parts_sent=0,
                    error=f'Text send failed: {err}',
                )
            parts_sent += 1

        for i, path in enumerate(resolved_attachments, start=1):
            ok, err = self._dispatch_attachment(path, handle=handle, chat_guid=chat_guid, service=service)
            if not ok:
                logger.warning(
                    'Attachment %d/%d failed (%s, service=%s): %s',
                    i,
                    len(resolved_attachments),
                    path,
                    service,
                    err,
                )
                return SendResult(
                    success=False,
                    recipient=recipient,
                    service=None,
                    parts_sent=parts_sent,
                    error=f'Attachment {i} of {len(resolved_attachments)} failed ({path}): {err}',
                )
            parts_sent += 1

        logger.info(
            'Sent %d part(s) to %s (service=%s, attachments=%d)',
            parts_sent,
            recipient,
            service,
            len(resolved_attachments),
        )
        return SendResult(
            success=True,
            recipient=recipient,
            service=service if service != 'auto' else None,
            parts_sent=parts_sent,
            error=None,
        )

    def _resolve_attachments(self, raw_paths: Sequence[str]) -> Sequence[Path]:
        """Resolve relative paths and validate each attachment.

        Raises ValueError on any malformed shape, missing file, or unreadable file —
        before any send happens, so we never partially dispatch on bad input.
        """
        resolved: list[Path] = []
        for raw in raw_paths:
            path = Path(raw).resolve(strict=False)
            if not self.ATTACHMENT_PATH_RE.match(str(path)):
                msg = f'Invalid attachment path shape: {path!r}. No quotes/backslashes/newlines allowed.'
                raise ValueError(msg)
            if not path.is_file():
                msg = f'Attachment not found or not a regular file: {path}'
                raise ValueError(msg)
            if not os.access(path, os.R_OK):
                msg = f'Attachment not readable: {path}'
                raise ValueError(msg)
            resolved.append(path)
        return resolved

    def _dispatch_text(
        self,
        text: str,
        *,
        handle: str | None,
        chat_guid: str | None,
        service: SendService,
    ) -> tuple[bool, str | None]:
        """Send a text message body via tempfile-backed AppleScript read."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            msg_file = Path(f.name)

        try:
            if chat_guid:
                script = f'''
                    set msgText to read (POSIX file "{msg_file}") as «class utf8»
                    tell application "Messages"
                        send msgText to chat id "{chat_guid}"
                    end tell
                '''
            elif service == 'auto':
                script = f'''
                    set msgText to read (POSIX file "{msg_file}") as «class utf8»
                    tell application "Messages"
                        send msgText to buddy "{handle}"
                    end tell
                '''
            else:
                script = f'''
                    set msgText to read (POSIX file "{msg_file}") as «class utf8»
                    tell application "Messages"
                        set targetService to 1st service whose service type = {service}
                        set targetBuddy to participant "{handle}" of targetService
                        send msgText to targetBuddy
                    end tell
                '''
            return self._run_osascript(script)
        finally:
            msg_file.unlink(missing_ok=True)

    def _dispatch_attachment(
        self,
        path: Path,
        *,
        handle: str | None,
        chat_guid: str | None,
        service: SendService,
    ) -> tuple[bool, str | None]:
        """Send a single file attachment via AppleScript POSIX file reference."""
        if chat_guid:
            script = f'''
                tell application "Messages"
                    send (POSIX file "{path}") to chat id "{chat_guid}"
                end tell
            '''
        elif service == 'auto':
            script = f'''
                tell application "Messages"
                    send (POSIX file "{path}") to buddy "{handle}"
                end tell
            '''
        else:
            script = f'''
                tell application "Messages"
                    set targetService to 1st service whose service type = {service}
                    set targetBuddy to participant "{handle}" of targetService
                    send (POSIX file "{path}") to targetBuddy
                end tell
            '''
        return self._run_osascript(script)

    def _run_osascript(self, script: str) -> tuple[bool, str | None]:
        """Run osascript. Returns (success, error_message_if_any)."""
        result = subprocess.run(
            ['osascript', '-e', script],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, None
