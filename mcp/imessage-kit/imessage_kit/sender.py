"""Send iMessages and SMS via AppleScript."""

from __future__ import annotations

__all__ = [
    'DispatchOutcome',
    'MessageSender',
]

import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from imessage_kit.types import SendService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchOutcome:
    """Internal result of an AppleScript dispatch attempt.

    This is NOT the user-facing send result — it only describes the AppleScript
    hop. The service layer augments it with chat.db delivery polling to produce
    the final SendResult that callers see. AppleScript success does NOT imply
    delivery; Messages.app may accept the call and later fail the transfer.
    """

    applescript_succeeded: bool
    """True if every osascript invocation exited 0 for this send."""

    applescript_error: str | None
    """osascript stderr from the first failed invocation, if any."""

    parts_sent: int
    """How many parts (text + attachments) AppleScript accepted before any failure."""


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
    ) -> DispatchOutcome:
        """Dispatch text and/or attachments via AppleScript.

        Returns a DispatchOutcome describing the AppleScript hop only. Whether the
        message actually reached the recipient is determined post-hoc by the
        service layer polling chat.db — this method has no access to that truth.

        Fail-fast: if any part's osascript call fails, subsequent parts are skipped
        and parts_sent reflects what was already dispatched.

        Args:
            text: Message body. Pass empty string for attachments-only send.
            handle: Phone number or email for 1:1 chats.
            chat_guid: Chat GUID for group chats (from list_chats output).
            service: 'auto' (Messages.app decides), 'iMessage', or 'SMS'.
            attachments: Ordered list of file paths (absolute or relative to cwd).
                Each must exist and be readable. Raises ValueError on bad input.
        """
        if not handle and not chat_guid:
            msg = 'Either handle or chat_guid is required.'
            raise ValueError(msg)

        if not text and not attachments:
            msg = 'Must provide text or at least one attachment.'
            raise ValueError(msg)

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
                return DispatchOutcome(
                    applescript_succeeded=False,
                    applescript_error=f'Text send failed: {err}',
                    parts_sent=0,
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
                return DispatchOutcome(
                    applescript_succeeded=False,
                    applescript_error=f'Attachment {i} of {len(resolved_attachments)} failed ({path}): {err}',
                    parts_sent=parts_sent,
                )
            parts_sent += 1

        logger.info(
            'AppleScript dispatched %d part(s) to %s (service=%s, attachments=%d)',
            parts_sent,
            recipient,
            service,
            len(resolved_attachments),
        )
        return DispatchOutcome(
            applescript_succeeded=True,
            applescript_error=None,
            parts_sent=parts_sent,
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
