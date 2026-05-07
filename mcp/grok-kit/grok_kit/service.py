"""GrokService — orchestration layer for all grok-kit tool logic."""

from __future__ import annotations

__all__ = [
    'Conversation',
    'GrokService',
]

from collections.abc import Sequence
from pathlib import Path

from cc_lib.schemas.base import ClosedModel
from grok_kit_sdk import GrokKit, errors, models

from grok_kit.auth import DEFAULT_COOKIE_PATH, format_cookie_header, load_state
from grok_kit.exceptions import AuthExpiredError


class Conversation(ClosedModel):
    """A grok.com conversation with messages ordered chronologically.

    Composed from three SDK calls (metadata + tree + bodies). The SDK's
    Pydantic types pass through directly for ``summary`` and ``messages``;
    only the wrapper here uses ``ClosedModel``.
    """

    summary: models.ConversationSummary
    messages: Sequence[models.MessageBody]


class GrokService:
    """Orchestrates the Speakeasy-generated SDK with auth-aware error mapping.

    The SDK is held as instance state so MCP and CLI layers can each
    construct one ``GrokService`` and reuse it for all calls in a session.
    """

    def __init__(self, sdk: GrokKit) -> None:
        self._sdk = sdk

    @classmethod
    def from_cookies(cls, cookie_path: Path = DEFAULT_COOKIE_PATH) -> GrokService:
        """Load cookies and construct a ready service."""
        return cls(GrokKit(cookie_header=format_cookie_header(load_state(cookie_path))))

    def list_conversations(
        self,
        *,
        page_size: int = 60,
        limit: int | None = None,
    ) -> Sequence[models.ConversationSummary]:
        """List conversations, auto-paginating through ``nextPageToken``.

        Server caps page size at 60. ``limit=None`` fetches all pages.
        Results come back most-recently-modified first.
        """
        out: list[models.ConversationSummary] = []
        page_token: str | None = None
        while True:
            try:
                page = self._sdk.conversations.list_conversations(
                    page_size=page_size,
                    page_token=page_token,
                )
            except errors.GrokKitDefaultError as exc:
                _check_auth(exc)
                raise
            out.extend(page.conversations)
            if limit is not None and len(out) >= limit:
                return out[:limit]
            if not page.next_page_token:
                return out
            page_token = page.next_page_token

    def get_full_conversation(self, conversation_id: str) -> Conversation:
        """Fetch metadata, tree, and message bodies; return chronologically ordered."""
        try:
            detail = self._sdk.conversations.get_conversation(conversation_id=conversation_id)
            tree = self._sdk.messages.get_response_tree(conversation_id=conversation_id)
            if not tree.response_nodes:
                return Conversation(summary=detail.conversation, messages=())
            ids = [node.response_id for node in tree.response_nodes]
            bodies = self._sdk.messages.load_messages(
                conversation_id=conversation_id,
                response_ids=ids,
            )
        except errors.GrokKitDefaultError as exc:
            _check_auth(exc)
            raise
        ordered = sorted(bodies.responses, key=lambda m: m.create_time)
        return Conversation(summary=detail.conversation, messages=ordered)


def _check_auth(exc: errors.GrokKitDefaultError) -> None:
    """Promote 401/403 to ``AuthExpiredError``; return otherwise."""
    raw = getattr(exc, 'raw_response', None)
    status = getattr(raw, 'status_code', None) if raw is not None else None
    if status in (401, 403):
        raise AuthExpiredError(status) from exc
