"""Document vector repository for persisting chunks with embeddings.

Typed interface over the Qdrant client. All public methods accept and return
strict Pydantic models from schemas.vectors.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence
from uuid import UUID

from document_search.clients.qdrant import QdrantClient
from document_search.schemas.chunking import FileType
from document_search.schemas.vectors import (
    CollectionInfo,
    SearchHit,
    SearchQuery,
    SearchResult,
    VectorPoint,
)


class DocumentVectorRepository:
    """Repository for document vector storage and retrieval.

    Provides typed interface for storing document chunks with their
    embeddings and searching by vector similarity.
    """

    def __init__(self, client: QdrantClient) -> None:
        """Initialize repository.

        Args:
            client: Qdrant client instance.
        """
        self._client = client

    def ensure_collection(self, vector_dimension: int) -> None:
        """Ensure collection exists with correct configuration.

        Args:
            vector_dimension: Size of embedding vectors (768 for Gemini).
        """
        self._client.ensure_collection(vector_dimension)

    def upsert(self, points: Sequence[VectorPoint]) -> int:
        """Store or update document vectors.

        Args:
            points: Typed vector points to store.

        Returns:
            Number of points upserted.
        """
        raw_points = [
            (
                point.id,
                list(point.vector),
                {
                    'source_path': point.source_path,
                    'chunk_index': point.chunk_index,
                    'file_type': point.file_type,
                    'text': point.text,
                    'start_char': point.start_char,
                    'end_char': point.end_char,
                    'heading_context': point.heading_context,
                    'page_number': point.page_number,
                    'json_path': point.json_path,
                },
            )
            for point in points
        ]
        return self._client.upsert(raw_points)

    def search(self, query: SearchQuery) -> SearchResult:
        """Search for similar documents.

        Args:
            query: Typed search query with vector and filters.

        Returns:
            Typed search results.

        Note:
            source_path_prefix filtering is done via post-filtering since
            Qdrant doesn't support native prefix matching.
        """
        file_types = list(query.file_types) if query.file_types else None

        # Request extra results if we'll be post-filtering by path prefix
        fetch_limit = query.limit * 3 if query.source_path_prefix else query.limit

        raw_results = self._client.search(
            vector=list(query.vector),
            limit=fetch_limit,
            score_threshold=query.score_threshold,
            file_types=file_types,
        )

        # Post-filter by source_path_prefix if specified
        if query.source_path_prefix:
            raw_results = [r for r in raw_results if r['source_path'].startswith(query.source_path_prefix)][
                : query.limit
            ]

        hits = [
            SearchHit(
                id=UUID(result['id']),
                score=result['score'],
                source_path=result['source_path'],
                chunk_index=result['chunk_index'],
                file_type=typing.cast(FileType, result['file_type']),
                text=result['text'],
                start_char=result['start_char'],
                end_char=result['end_char'],
                heading_context=result.get('heading_context'),
                page_number=result.get('page_number'),
                json_path=result.get('json_path'),
            )
            for result in raw_results
        ]

        return SearchResult(hits=hits, total=len(hits))

    def get(self, point_ids: Sequence[UUID]) -> Sequence[SearchHit]:
        """Retrieve documents by ID.

        Args:
            point_ids: IDs to retrieve.

        Returns:
            Typed document data (without scores).
        """
        raw_results = self._client.get(point_ids)

        return [
            SearchHit(
                id=UUID(result['id']),
                score=0.0,  # No score for direct retrieval
                source_path=result['source_path'],
                chunk_index=result['chunk_index'],
                file_type=typing.cast(FileType, result['file_type']),
                text=result['text'],
                start_char=result['start_char'],
                end_char=result['end_char'],
                heading_context=result.get('heading_context'),
                page_number=result.get('page_number'),
                json_path=result.get('json_path'),
            )
            for result in raw_results
        ]

    def delete(self, point_ids: Sequence[UUID]) -> int:
        """Delete documents by ID.

        Args:
            point_ids: IDs to delete.

        Returns:
            Number of points deleted.
        """
        return self._client.delete(point_ids)

    def count(self) -> int:
        """Get total document count."""
        return self._client.count()

    def get_collection_info(self) -> CollectionInfo | None:
        """Get collection metadata.

        Returns:
            Collection info if exists, None otherwise.
        """
        info = self._client.get_collection_info()
        if info is None:
            return None

        return CollectionInfo(
            name=info['name'],
            vector_dimension=info['vector_dimension'],
            points_count=info['points_count'],
            status=info['status'],
        )
