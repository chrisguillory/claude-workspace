from __future__ import annotations

__all__ = [
    'MeetingService',
]

from collections.abc import Sequence

from granola_kit.clients.granola_api import BatchDocumentsResponse, DocumentSetResponse, GranolaAPIClient
from granola_kit.schemas.results import Meeting


class MeetingService:
    """Meeting views derived from the Granola document store."""

    def __init__(self, client: GranolaAPIClient) -> None:
        self._client = client

    async def list_meetings(self, *, limit: int = 20) -> Sequence[Meeting]:
        """Recent meetings, most-recently-updated first, deleted excluded, projected to Meeting."""
        doc_set = await self._client.post('get-document-set', {}, into=DocumentSetResponse)
        entries = sorted(doc_set.documents.items(), key=lambda item: item[1].updated_at, reverse=True)
        if limit:
            entries = entries[:limit]
        is_shared = {doc_id: bool(entry.shared) for doc_id, entry in entries}
        batch = await self._client.post(
            'get-documents-batch', {'document_ids': list(is_shared)}, into=BatchDocumentsResponse
        )
        return [
            Meeting(
                id=doc.id,
                title=doc.title,
                created_at=doc.created_at,
                is_shared=is_shared.get(doc.id, False),
                type=doc.type,
            )
            for doc in batch.docs
            if doc.deleted_at is None
        ]
