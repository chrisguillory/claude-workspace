"""Document vector repository for persisting chunks with embeddings.

Typed interface over the Qdrant client. All public methods accept and return
strict Pydantic models from schemas.vectors.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from document_search.clients.qdrant import QdrantClient
from document_search.schemas.chunking import EXTENSION_MAP, FileType
from document_search.schemas.vectors import (
    CollectionInfo,
    FileIndexStatus,
    IndexBreakdown,
    IndexedFile,
    SearchHit,
    SearchQuery,
    SearchResult,
    VectorPoint,
)

__all__ = [
    'DocumentVectorRepository',
]


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
        """Store or update document vectors with hybrid embeddings.

        Args:
            points: Typed vector points to store (dense + sparse vectors).

        Returns:
            Number of points upserted.
        """
        raw_points = [
            (
                point.id,
                list(point.dense_vector),
                list(point.sparse_indices),
                list(point.sparse_values),
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
        """Search documents using configurable strategy.

        Args:
            query: Typed search query with vectors, filters, and search_type.

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
            search_type=query.search_type,
            dense_vector=list(query.dense_vector) if query.dense_vector else None,
            sparse_indices=list(query.sparse_indices) if query.sparse_indices else None,
            sparse_values=list(query.sparse_values) if query.sparse_values else None,
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

    def delete_by_source_path_prefix(self, prefix: str) -> int:
        """Delete all vectors for files under a directory prefix.

        Used for full_reindex to remove all existing chunks before re-indexing.
        This is authoritative cleanup that finds orphaned chunks not tracked in state.

        Args:
            prefix: Directory path prefix.

        Returns:
            Number of points deleted.
        """
        return self._client.delete_by_source_path_prefix(prefix)

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

    # Visibility methods for index introspection

    def get_index_breakdown(self) -> IndexBreakdown:
        """Get overview of what's in the document index.

        Returns:
            IndexBreakdown with total chunks, breakdown by file type,
            unique file count, and list of supported types.
        """
        total_chunks = self._client.count()
        by_file_type = self._client.facet_by_file_type()

        # Get unique file count via scroll (more efficient than full path list)
        unique_paths = self._client.get_unique_source_paths()
        unique_files = len(unique_paths)

        # Get supported types from EXTENSION_MAP (static)
        supported_types = sorted(set(EXTENSION_MAP.values()))

        return IndexBreakdown(
            total_chunks=total_chunks,
            by_file_type=by_file_type,
            unique_files=unique_files,
            supported_types=supported_types,
        )

    def is_file_indexed(self, path: str) -> FileIndexStatus:
        """Check if a specific file is indexed.

        Args:
            path: Absolute path to the file.

        Returns:
            FileIndexStatus with indexed flag, reason, chunk count, and file type.
        """
        # Check if file type is supported
        path_obj = Path(path)
        extension = path_obj.suffix.lower()
        file_type = EXTENSION_MAP.get(extension)

        if file_type is None:
            return FileIndexStatus(
                indexed=False,
                reason='unsupported_type',
                chunk_count=0,
                file_type=None,
            )

        # Check if file has chunks in the index
        chunk_count = self._client.count_by_source_path(path)

        if chunk_count == 0:
            return FileIndexStatus(
                indexed=False,
                reason='not_found',
                chunk_count=0,
                file_type=file_type,
            )

        return FileIndexStatus(
            indexed=True,
            reason='indexed',
            chunk_count=chunk_count,
            file_type=file_type,
        )

    def list_indexed_files(
        self,
        path_prefix: str | None = None,
        file_type: str | None = None,
        limit: int = 50,
    ) -> Sequence[IndexedFile]:
        """List files in the index with optional filtering.

        Args:
            path_prefix: Filter to files under this path prefix.
            file_type: Filter to this file type.
            limit: Maximum number of files to return.

        Returns:
            List of IndexedFile sorted by chunk count descending.
        """
        raw_paths = self._client.get_unique_source_paths(
            path_prefix=path_prefix,
            file_type=file_type,
            limit=limit,
        )

        return [
            IndexedFile(path=path, chunk_count=count, file_type=typing.cast(FileType, ftype))
            for path, ftype, count in raw_paths
        ]
