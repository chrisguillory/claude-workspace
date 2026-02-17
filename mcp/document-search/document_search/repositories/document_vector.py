"""Document vector repository for persisting chunks with embeddings.

Typed interface over the Qdrant client. All public methods accept and return
strict Pydantic models from schemas.vectors.

Uses BatchLoader pattern for upserts to coalesce concurrent requests.
"""

from __future__ import annotations

import asyncio
import logging
import typing
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import attrs
from local_lib.batch_loader import GenericBatchLoader

from document_search.clients.qdrant import QdrantClient
from document_search.schemas.chunking import EXTENSION_MAP, FileType
from document_search.schemas.vectors import (
    ContentStats,
    FileIndexStatus,
    IndexedFile,
    SearchHit,
    SearchQuery,
    SearchResult,
    StorageStats,
    VectorPoint,
)

__all__ = [
    'DocumentVectorRepository',
]

logger = logging.getLogger(__name__)


class DocumentVectorRepository:
    """Repository for document vector storage and retrieval.

    Provides typed interface for storing document chunks with their
    embeddings and searching by vector similarity.

    Each repository instance is bound to a specific collection.
    Uses UpsertLoader for automatic batching of concurrent upsert requests.
    """

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        """Initialize repository for a specific collection.

        Args:
            client: Qdrant client instance.
            collection_name: Name of the collection to operate on.
        """
        self._client = client
        self._collection_name = collection_name
        self._upsert_loader = UpsertLoader(client, collection_name)

    async def ensure_collection(self, vector_dimension: int) -> None:
        """Ensure collection exists with correct configuration.

        Args:
            vector_dimension: Size of embedding vectors (e.g., 768).
        """
        await self._client.ensure_collection(self._collection_name, vector_dimension)

    async def get_indexing_threshold(self) -> int:
        """Get current HNSW indexing threshold in KB."""
        return await self._client.get_indexing_threshold(self._collection_name)

    async def set_indexing_threshold(self, threshold_kb: int) -> None:
        """Set HNSW indexing threshold for bulk operations.

        Use 0 to disable index building during bulk upserts,
        then restore original value after to trigger rebuild.
        """
        await self._client.set_indexing_threshold(self._collection_name, threshold_kb)

    async def upsert(self, points: Sequence[VectorPoint]) -> int:
        """Store or update document vectors with hybrid embeddings (automatically batched).

        Points are batched with concurrent requests from other callers for efficiency.

        Args:
            points: Typed vector points to store (dense + sparse vectors).

        Returns:
            Number of points upserted.
        """
        tasks = [self._upsert_loader.load(UpsertLoader.Request(point=point)) for point in points]
        results = await asyncio.gather(*tasks)
        return sum(results)

    async def search(self, query: SearchQuery) -> SearchResult:
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

        raw_results = await self._client.search(
            self._collection_name,
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

    async def get(self, point_ids: Sequence[UUID]) -> Sequence[SearchHit]:
        """Retrieve documents by ID.

        Args:
            point_ids: IDs to retrieve.

        Returns:
            Typed document data (without scores).
        """
        raw_results = await self._client.get(self._collection_name, point_ids)

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

    async def delete(self, point_ids: Sequence[UUID]) -> int:
        """Delete documents by ID.

        Args:
            point_ids: IDs to delete.

        Returns:
            Number of points deleted.
        """
        return await self._client.delete(self._collection_name, point_ids)

    async def delete_by_source_path_prefix(self, prefix: str) -> int:
        """Delete all vectors for files under a directory prefix.

        Used for full_reindex to remove all existing chunks before re-indexing.
        This is authoritative cleanup that finds orphaned chunks not tracked in state.

        Args:
            prefix: Directory path prefix.

        Returns:
            Number of points deleted.
        """
        return await self._client.delete_by_source_path_prefix(self._collection_name, prefix)

    async def count(self) -> int:
        """Get total document count."""
        return await self._client.count(self._collection_name)

    async def get_storage_stats(self) -> StorageStats | None:
        """Get Qdrant storage statistics.

        Returns:
            Storage stats if collection exists, None otherwise.
        """
        info = await self._client.get_collection_info(self._collection_name)
        if info is None:
            return None

        return StorageStats(
            vector_dimension=info['vector_dimension'],
            points_count=info['points_count'],
            status=info['status'],
        )

    # Visibility methods for index introspection

    async def get_content_stats(self, path: str | None = None) -> ContentStats:
        """Get content breakdown statistics.

        Args:
            path: Scope stats to this path. Use "**" or None for global stats.

        Returns:
            ContentStats with total chunks, breakdown by file type,
            unique file count, and list of supported types.
        """
        if path is None or path == '**':
            # Global stats - use efficient facet API
            total_chunks = await self._client.count(self._collection_name)
            by_file_type = dict(await self._client.facet_by_file_type(self._collection_name))
            unique_paths = await self._client.get_unique_source_paths(self._collection_name)
            unique_files = len(unique_paths)
        else:
            # Scoped stats - aggregate from filtered paths
            raw_paths = await self._client.get_unique_source_paths(self._collection_name, path_prefix=path)
            unique_files = len(raw_paths)
            total_chunks = sum(count for _, _, count in raw_paths)
            by_file_type_agg: dict[str, int] = {}
            for _, ftype, count in raw_paths:
                by_file_type_agg[ftype] = by_file_type_agg.get(ftype, 0) + count
            by_file_type = by_file_type_agg

        supported_types = sorted(set(EXTENSION_MAP.values()))

        return ContentStats(
            total_chunks=total_chunks,
            by_file_type=by_file_type,
            unique_files=unique_files,
            supported_types=supported_types,
        )

    async def is_file_indexed(self, path: str) -> FileIndexStatus:
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
        chunk_count = await self._client.count_by_source_path(self._collection_name, path)

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

    async def list_indexed_files(
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
        raw_paths = await self._client.get_unique_source_paths(
            self._collection_name,
            path_prefix=path_prefix,
            file_type=file_type,
            limit=limit,
        )

        return [
            IndexedFile(path=path, chunk_count=count, file_type=typing.cast(FileType, ftype))
            for path, ftype, count in raw_paths
        ]

    async def clear_documents(self, path: str | None = None) -> tuple[int, int]:
        """Clear documents from the index.

        Args:
            path: Path to clear. Use "**" for entire index.
                  If None, this is a programming error (caller should resolve CWD).

        Returns:
            Tuple of (files_removed, chunks_removed).
        """
        if path == '**':
            # Drop entire collection (instant vs scrolling through all points)
            chunks_count = await self._client.count(self._collection_name)
            await self._client.delete_collection(self._collection_name)
            return 0, chunks_count  # Can't count files after drop

        if path is None:
            raise ValueError('path must be provided (use "**" for entire index)')

        # Check if path is a file or directory by looking at indexed content
        chunk_count = await self._client.count_by_source_path(self._collection_name, path)
        if chunk_count > 0:
            # Exact file match
            await self._client.delete_by_source_path(self._collection_name, path)
            return 1, chunk_count

        # Directory prefix
        raw_paths = await self._client.get_unique_source_paths(self._collection_name, path_prefix=path)
        files_count = len(raw_paths)
        chunks_count = await self._client.delete_by_source_path_prefix(self._collection_name, path)
        return files_count, chunks_count


class UpsertLoader(GenericBatchLoader['UpsertLoader.Request', int]):
    """BatchLoader for Qdrant upserts with automatic batching.

    Concurrency control is handled by QdrantClient's semaphore.
    Each loader is bound to a specific collection.
    """

    @attrs.define(frozen=True, kw_only=True)
    class Request:
        """Hashable request for batching."""

        __strict_typing_linter__hashable_fields__ = True

        point: VectorPoint

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name
        super().__init__(
            bulk_load=self._bulk_load,
            batch_size=100,  # Reduced from 300 to lower burst pressure
        )

    async def _bulk_load(self, requests: Sequence[Request]) -> Sequence[int]:
        """Transform requests to raw points and call bulk API."""
        raw_points = [
            (
                req.point.id,
                list(req.point.dense_vector),
                list(req.point.sparse_indices),
                list(req.point.sparse_values),
                {
                    'source_path': req.point.source_path,
                    'chunk_index': req.point.chunk_index,
                    'file_type': req.point.file_type,
                    'text': req.point.text,
                    'start_char': req.point.start_char,
                    'end_char': req.point.end_char,
                    'heading_context': req.point.heading_context,
                    'page_number': req.point.page_number,
                    'json_path': req.point.json_path,
                },
            )
            for req in requests
        ]

        total_chars = sum(len(req.point.text) for req in requests)
        logger.debug(f'[BATCH] Upserting {len(requests)} points ({total_chars:,} chars)')

        await self._client.upsert(self._collection_name, raw_points)
        return [1] * len(requests)
