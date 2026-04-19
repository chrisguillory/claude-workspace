"""Send iMessages and SMS via AppleScript."""

from __future__ import annotations

__all__ = [
    'MessageSender',
]

import logging
import subprocess
import tempfile
from pathlib import Path

from imessage_kit.types import SendResult, SendService

logger = logging.getLogger(__name__)


class MessageSender:
    """Send messages via osascript."""

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
