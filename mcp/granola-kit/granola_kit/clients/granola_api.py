from __future__ import annotations

__all__ = [
    'BatchDocumentsResponse',
    'DocumentSetEntry',
    'DocumentSetResponse',
    'GranolaAPIClient',
    'GranolaDocument',
    'granola_api_client',
]

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TypeVar

import httpx
from cc_lib.schemas.base import OpenModel
from cc_lib.types import JsonObject

from granola_kit.clients.auth import get_access_token
from granola_kit.clients.identity import GranolaIdentity
from granola_kit.exceptions import GranolaUnsupportedClientError

_T = TypeVar('_T', bound=OpenModel)


# ── Granola API wire entities (mirror the upstream response shapes) ──


class GranolaDocument(OpenModel):
    """A Granola meeting document, as the API returns it.

    Full top-level fidelity. The rich nested structures (people, calendar event,
    ProseMirror notes, attachments) are typed loosely for now and tighten to full
    models as their slices land.
    """

    id: str
    created_at: str
    updated_at: str
    user_id: str
    transcribe: bool
    public: bool
    meeting_end_count: int
    has_shareable_link: bool
    creation_source: str
    sharing_link_visibility: str
    subscription_plan_id: str | None
    privacy_mode_enabled: bool | None
    workspace_id: str | None
    title: str | None
    type: str | None
    notes_markdown: str | None
    valid_meeting: bool | None
    show_private_notes: bool | None
    people: JsonObject | None
    google_calendar_event: JsonObject | None
    attachments: Sequence[JsonObject] | None
    notes: JsonObject | None = None
    notes_plain: str | None = None
    last_indexed_at: str | None = None
    is_shared_direct: bool | None = None
    cloned_from: str | None = None
    deleted_at: str | None = None
    overview: str | None = None
    status: str | None = None
    external_transcription_id: str | None = None
    audio_file_handle: str | None = None
    last_viewed_panel: JsonObject | None = None
    was_trashed: bool | None = None
    is_primary_event_note: bool | None = None
    ydoc_state: str | None = None
    ydoc_version: int | None = None
    zoom_rtms_permission: str | None = None
    document_user_role: str | None = None
    is_scratchpad: bool | None = None
    # Always null in the current API — typed None so a future non-null value fails fast.
    chapters: None = None
    selected_template: None = None
    summary: None = None
    affinity_note_id: None = None
    hubspot_note_url: None = None
    visibility: None = None
    notification_config: None = None
    transcript_deleted_at: None = None
    metadata: None = None
    attio_shared_at: None = None


class DocumentSetEntry(OpenModel):
    """A document's lightweight entry in the get-document-set index."""

    updated_at: str
    owner: bool | None = None
    shared: bool | None = None
    has_ydoc: bool | None = None
    has_notes_ydoc: bool | None = None


class DocumentSetResponse(OpenModel):
    """Response from POST /v1/get-document-set — the full document index, no pagination."""

    documents: Mapping[str, DocumentSetEntry]


class BatchDocumentsResponse(OpenModel):
    """Response from POST /v1/get-documents-batch."""

    docs: Sequence[GranolaDocument]


# ── Client ──


class GranolaAPIClient:
    """Async transport for Granola's private API.

    Injects the bearer token (auth) + desktop identity headers, raises on the
    HTTP-200 "Unsupported client" envelope, and validates each response into the
    wire entity the caller names.
    """

    def __init__(self, http: httpx.AsyncClient, identity: GranolaIdentity) -> None:
        self._http = http
        self._identity = identity

    async def post(self, endpoint: str, body: Mapping[str, object], *, into: type[_T]) -> _T:
        """POST a JSON body to ``api.granola.ai/v1/{endpoint}`` and validate the response into ``into``."""
        headers = {
            'Authorization': f'Bearer {await get_access_token()}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            **self._identity.headers(),
        }
        response = await self._http.post(
            f'https://api.granola.ai/v1/{endpoint}',
            json=dict(body),
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
        _guard_unsupported_client(data)
        return into.model_validate(data)


@asynccontextmanager
async def granola_api_client() -> AsyncIterator[GranolaAPIClient]:
    """Open a GranolaAPIClient with the standard timeout and detected identity.

    The single construction seam shared by the CLI (per-invocation) and the MCP
    server (lifespan-scoped); each owns only the ``async with`` scope.
    """
    async with httpx.AsyncClient(timeout=30) as http:
        yield GranolaAPIClient(http, GranolaIdentity.detect())


def _guard_unsupported_client(data: object) -> None:
    """Raise on Granola's HTTP-200 'Unsupported client' envelope — value-match, not key-presence."""
    if isinstance(data, dict) and data.get('message') == 'Unsupported client':
        raise GranolaUnsupportedClientError('Granola returned a 200 "Unsupported client" envelope')
