"""Send iMessages and SMS via AppleScript."""

from __future__ import annotations

__all__ = [
    'MessageSender',
]

import logging
import re
import subprocess
import tempfile
from collections.abc import Set
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
    ) -> SendResult:
        """Send a message to a handle (1:1) or chat GUID (group).

        Uses file-based message passing to avoid AppleScript quote escaping.

        Args:
            text: Message text.
            handle: Phone number or email for 1:1 chats.
            chat_guid: Chat GUID for group chats (from list_chats output).
            service: 'auto' (Messages.app decides), 'iMessage', or 'SMS'. Only applies to 1:1.
        """
        if not handle and not chat_guid:
            return SendResult(
                success=False,
                recipient='',
                service=None,
                error='Either handle or chat_guid is required.',
            )

        # Validate before interpolating into AppleScript. Handle, chat_guid, and
        # service are interpolated directly (only the message body is tempfile-safe),
        # so we whitelist their shape to prevent AppleScript injection.
        if handle is not None and not self.HANDLE_RE.match(handle):
            msg = f'Invalid handle shape: {handle!r}. Expected phone, email, or short code.'
            raise ValueError(msg)
        if chat_guid is not None and not self.CHAT_GUID_RE.match(chat_guid):
            msg = f'Invalid chat_guid shape: {chat_guid!r}. Expected chat.db GUID format.'
            raise ValueError(msg)
        if service not in self.VALID_SERVICES:
            msg = f'Invalid service: {service!r}. Expected one of {sorted(self.VALID_SERVICES)}.'
            raise ValueError(msg)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            msg_file = Path(f.name)

        recipient = chat_guid or handle or ''

        try:
            if chat_guid:
                # Group chat: send to chat by GUID
                script = f'''
                    set msgText to read (POSIX file "{msg_file}") as «class utf8»
                    tell application "Messages"
                        send msgText to chat id "{chat_guid}"
                    end tell
                '''
            elif service == 'auto':
                # 1:1 auto-routing: Messages.app picks iMessage or SMS
                script = f'''
                    set msgText to read (POSIX file "{msg_file}") as «class utf8»
                    tell application "Messages"
                        send msgText to buddy "{handle}"
                    end tell
                '''
            else:
                # 1:1 explicit service
                script = f'''
                    set msgText to read (POSIX file "{msg_file}") as «class utf8»
                    tell application "Messages"
                        set targetService to 1st service whose service type = {service}
                        set targetBuddy to participant "{handle}" of targetService
                        send msgText to targetBuddy
                    end tell
                '''

            result = subprocess.run(
                ['osascript', '-e', script],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        finally:
            msg_file.unlink(missing_ok=True)

        if result.returncode != 0:
            error = result.stderr.strip()
            logger.warning('Send failed to %s (service=%s): %s', recipient, service, error)
            return SendResult(success=False, recipient=recipient, service=None, error=error)

        logger.info('Sent message to %s (service=%s)', recipient, service)
        return SendResult(
            success=True,
            recipient=recipient,
            service=service if service != 'auto' else None,
            error=None,
        )
