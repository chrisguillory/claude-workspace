"""Low-level Qdrant vector database client.

Thin wrapper around qdrant-client. Handles API calls only - no business logic.
Type translation happens in the repository layer.

Uses AsyncQdrantClient for non-blocking I/O in async contexts.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict
from uuid import UUID

import httpx
import tenacity
from local_lib import ConcurrencyTracker
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from document_search.clients import _retry

logger = logging.getLogger(__name__)


__all__ = [
    'CollectionInfoDict',
    'QdrantClient',
    'SearchResultDict',
]


class CollectionInfoDict(TypedDict):
    """Raw collection metadata from Qdrant."""

    name: str
    vector_dimension: int
    points_count: int
    status: str


class SearchResultDict(TypedDict):
    """Raw search result from Qdrant."""

    id: str
    score: float
    source_path: str
    chunk_index: int
    file_type: str
    text: str
    start_char: int
    end_char: int
    heading_context: str | None
    page_number: int | None
    json_path: str | None


class QdrantClient:
    """Low-level async Qdrant client for vector operations."""

    DEFAULT_URL = 'http://localhost:6333'
    DEFAULT_COLLECTION = 'document_chunks'

    # Max concurrent upsert operations
    DEFAULT_MAX_CONCURRENT_UPSERTS = 8

    # Connection pool and timeout tuning
    # - Pool size scales with CPU (2x) to handle concurrent workers
    # - Timeout doubled from default 5s to handle large batch upserts under load
    # - Must pass explicit limits to override qdrant-client's localhost defaults
    #   which disable keep-alive (max_keepalive_connections=0)
    DEFAULT_TIMEOUT = 10
    DEFAULT_POOL_SIZE = (os.cpu_count() or 8) * 2

    def __init__(
        self,
        url: str = DEFAULT_URL,
        collection_name: str = DEFAULT_COLLECTION,
        max_concurrent_upserts: int = DEFAULT_MAX_CONCURRENT_UPSERTS,
        timeout: int = DEFAULT_TIMEOUT,
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        """Initialize client.

        Args:
            url: Qdrant server URL.
            collection_name: Default collection for operations.
            max_concurrent_upserts: Max concurrent upsert API calls.
            timeout: HTTP timeout in seconds (default 10).
            pool_size: HTTP connection pool size (default 2x CPU count).
        """
        self._url = url
        self._collection_name = collection_name

        # Override localhost defaults to enable connection pooling with keep-alive
        limits = httpx.Limits(max_connections=pool_size, max_keepalive_connections=pool_size)
        self._client = AsyncQdrantClient(url=url, timeout=timeout, limits=limits)

        self._upsert_semaphore = asyncio.Semaphore(max_concurrent_upserts)
        self._tracker = ConcurrencyTracker('QDRANT_UPSERT')

    async def ensure_collection(self, vector_dimension: int) -> None:
        """Create collection with hybrid search support if it doesn't exist.

        Creates a collection with:
        - Dense vectors: Semantic embeddings (Gemini, 768 dimensions)
        - Sparse vectors: BM25 keyword embeddings (fastembed)
        - Keyword index on file_type for faceting

        Args:
            vector_dimension: Size of dense embedding vectors (e.g., 768 for Gemini).
        """
        collections = await self._client.get_collections()
        exists = any(c.name == self._collection_name for c in collections.collections)

        if not exists:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    'dense': VectorParams(
                        size=vector_dimension,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    'sparse': SparseVectorParams(
                        modifier=Modifier.IDF,
                    ),
                },
            )

        # Ensure keyword index on file_type for faceting (idempotent)
        await self._ensure_file_type_index()

    async def delete_collection(self) -> bool:
        """Delete the entire collection. Returns True if deleted, False if didn't exist."""
        collections = await self._client.get_collections()
        exists = any(c.name == self._collection_name for c in collections.collections)

        if exists:
            await self._client.delete_collection(self._collection_name)
            return True
        return False

    @_retry.qdrant_breaker
    @tenacity.retry(
        retry=tenacity.retry_if_exception(_retry.is_retryable_qdrant_error),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=0.5, max=5),
        before_sleep=_retry.log_qdrant_retry,
    )
    async def upsert(
        self,
        points: Sequence[  # strict_typing_linter.py: loose-typing # Qdrant payload
            tuple[UUID, Sequence[float], Sequence[int], Sequence[float], Mapping[str, Any]]
        ],
    ) -> int:
        """Insert or update points with hybrid vectors.

        Retries on transient network errors (ReadError, WriteError).

        Args:
            points: Sequence of (id, dense_vector, sparse_indices, sparse_values, payload) tuples.

        Returns:
            Number of points upserted.
        """
        point_structs = [
            PointStruct(
                id=str(point_id),
                vector={
                    'dense': list(dense_vector),
                    'sparse': SparseVector(
                        indices=list(sparse_indices),
                        values=list(sparse_values),
                    ),
                },
                payload=dict(payload),
            )
            for point_id, dense_vector, sparse_indices, sparse_values, payload in points
        ]

        async with self._upsert_semaphore, self._tracker.track():
            await self._client.upsert(
                collection_name=self._collection_name,
                points=point_structs,
            )
        return len(point_structs)

    async def search(
        self,
        search_type: str = 'hybrid',
        dense_vector: Sequence[float] | None = None,
        sparse_indices: Sequence[int] | None = None,
        sparse_values: Sequence[float] | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        file_types: Sequence[str] | None = None,
    ) -> Sequence[SearchResultDict]:
        """Search with configurable strategy.

        Args:
            search_type: Search strategy:
                - 'hybrid': Dense + sparse with RRF fusion. Recommended for most queries.
                - 'lexical': BM25 sparse only. Best for exact term/symbol matching.
                - 'embedding': Dense only. Primarily for debugging/comparison.
            dense_vector: Gemini embedding (required for hybrid/embedding).
            sparse_indices: BM25 sparse indices (required for hybrid/lexical).
            sparse_values: BM25 sparse values (required for hybrid/lexical).
            limit: Maximum results.
            score_threshold: Minimum similarity score.
            file_types: Filter by file types.

        Returns:
            List of results with id, score, and payload.

        Note:
            Path prefix filtering is handled at the repository layer via
            post-filtering, as Qdrant doesn't support native prefix matching.
        """
        # Build filter conditions
        conditions = []
        if file_types:
            conditions.append(FieldCondition(key='file_type', match=MatchAny(any=list(file_types))))

        query_filter = Filter(must=conditions) if conditions else None

        # Execute search based on strategy
        if search_type == 'lexical':
            # BM25 sparse vectors only - keyword/full-text matching
            if sparse_indices is None or sparse_values is None:
                raise ValueError('lexical search requires sparse_indices and sparse_values')
            results = await self._client.query_points(
                collection_name=self._collection_name,
                query=SparseVector(
                    indices=list(sparse_indices),
                    values=list(sparse_values),
                ),
                using='sparse',
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
        elif search_type == 'embedding':
            # Dense vectors only - neural similarity
            if dense_vector is None:
                raise ValueError('embedding search requires dense_vector')
            results = await self._client.query_points(
                collection_name=self._collection_name,
                query=list(dense_vector),
                using='dense',
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
        else:
            # Hybrid: RRF fusion of dense + sparse (default)
            if dense_vector is None or sparse_indices is None or sparse_values is None:
                raise ValueError('hybrid search requires dense_vector, sparse_indices, and sparse_values')
            results = await self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    Prefetch(
                        query=list(dense_vector),
                        using='dense',
                        limit=50,
                        filter=query_filter,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=list(sparse_indices),
                            values=list(sparse_values),
                        ),
                        using='sparse',
                        limit=50,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                score_threshold=score_threshold,
            )

        return [
            SearchResultDict(
                id=str(hit.id),
                score=hit.score,
                source_path=hit.payload['source_path'],
                chunk_index=hit.payload['chunk_index'],
                file_type=hit.payload['file_type'],
                text=hit.payload['text'],
                start_char=hit.payload['start_char'],
                end_char=hit.payload['end_char'],
                heading_context=hit.payload.get('heading_context'),
                page_number=hit.payload.get('page_number'),
                json_path=hit.payload.get('json_path'),
            )
            for hit in results.points
            if hit.payload is not None
        ]

    async def get(  # strict_typing_linter.py: loose-typing # Qdrant payload
        self, point_ids: Sequence[UUID]
    ) -> Sequence[Mapping[str, Any]]:
        """Retrieve points by ID.

        Args:
            point_ids: IDs to retrieve.

        Returns:
            List of points with id and payload.
        """
        results = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[str(pid) for pid in point_ids],
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                'id': point.id,
                **point.payload,
            }
            for point in results
            if point.payload is not None
        ]

    async def delete(self, point_ids: Sequence[UUID]) -> int:
        """Delete points by ID.

        Args:
            point_ids: IDs to delete.

        Returns:
            Number of points requested for deletion (not confirmed count).
            Qdrant's delete is idempotent - non-existent IDs are silently ignored.
        """
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=[str(pid) for pid in point_ids],
        )
        return len(point_ids)

    async def delete_by_source_path(self, source_path: str) -> None:
        """Delete all points for a specific source file.

        Used when re-indexing a file to remove old chunks before inserting new ones.

        Args:
            source_path: Exact source_path to match.
        """
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key='source_path',
                            match=MatchValue(value=source_path),
                        )
                    ]
                )
            ),
        )

    async def delete_by_source_path_prefix(self, prefix: str) -> int:
        """Delete all points where source_path starts with prefix.

        Used for full_reindex to clean slate a directory before re-indexing.
        Scrolls entire collection since Qdrant doesn't support prefix filtering.

        Args:
            prefix: Directory path prefix. Matches files within this directory.
                    /a/b matches /a/b/c.md but NOT /a/b-other/x.md

        Returns:
            Number of points deleted.
        """
        prefix = prefix.rstrip('/')
        deleted = 0
        offset = None

        while True:
            results, offset = await self._client.scroll(
                collection_name=self._collection_name,
                limit=1000,
                offset=offset,
                with_payload=['source_path'],
                with_vectors=False,
            )

            if not results:
                break

            ids_to_delete = []
            for point in results:
                if point.payload is None:
                    continue
                path = point.payload.get('source_path', '')
                # Match exact prefix OR path within prefix directory
                if path == prefix or path.startswith(prefix + '/'):
                    ids_to_delete.append(point.id)

            if ids_to_delete:
                await self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=ids_to_delete,
                )
                deleted += len(ids_to_delete)

            if offset is None:
                break

        return deleted

    async def count(self) -> int:
        """Get total point count in collection."""
        info = await self._client.get_collection(self._collection_name)
        return info.points_count or 0

    async def collection_exists(self) -> bool:
        """Check if collection exists."""
        collections = await self._client.get_collections()
        return any(c.name == self._collection_name for c in collections.collections)

    async def get_collection_info(self) -> CollectionInfoDict | None:
        """Get collection metadata.

        Returns:
            Typed dict with name, vector_dimension, points_count, status.
            None if collection doesn't exist.
        """
        if not await self.collection_exists():
            return None

        info = await self._client.get_collection(self._collection_name)

        # Handle both named vectors (dict) and single vector config
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            # Named vectors - get dense vector dimension
            dense_config = vectors_config.get('dense')
            vector_dimension = dense_config.size if dense_config else 0
        elif vectors_config is not None:
            # Single vector config (legacy)
            vector_dimension = vectors_config.size
        else:
            vector_dimension = 0

        return CollectionInfoDict(
            name=self._collection_name,
            vector_dimension=vector_dimension,
            points_count=info.points_count or 0,
            status=str(info.status),
        )

    # Visibility methods for index introspection

    async def facet_by_file_type(self) -> Mapping[str, int]:
        """Get chunk counts by file type using Facet API.

        Returns:
            Mapping of file_type to chunk count.
        """
        result = await self._client.facet(
            collection_name=self._collection_name,
            key='file_type',
            limit=100,  # More than enough for our ~7 file types
        )
        return {str(hit.value): hit.count for hit in result.hits}

    async def count_by_source_path(self, source_path: str) -> int:
        """Count chunks for a specific source file.

        Args:
            source_path: Exact path to check.

        Returns:
            Number of chunks indexed for this file (0 if not indexed).
        """
        result = await self._client.count(
            collection_name=self._collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key='source_path',
                        match=MatchValue(value=source_path),
                    )
                ]
            ),
        )
        return result.count

    async def get_unique_source_paths(
        self,
        path_prefix: str | None = None,
        file_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[tuple[str, str, int]]:
        """Get unique source_paths with file types and chunk counts.

        Uses scroll API to iterate all points and aggregate by source_path.
        Results are sorted by chunk count descending.

        Args:
            path_prefix: Filter to paths starting with this prefix.
            file_type: Filter to this file type.
            limit: Maximum number of files to return.

        Returns:
            List of (source_path, file_type, chunk_count) tuples.
        """
        # Build filter if needed
        conditions = []
        if file_type:
            conditions.append(FieldCondition(key='file_type', match=MatchValue(value=file_type)))
        scroll_filter = Filter(must=conditions) if conditions else None

        # Aggregate by source_path
        path_data: dict[str, tuple[str, int]] = {}  # path -> (file_type, count)
        offset = None

        while True:
            results, offset = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                limit=5000,
                offset=offset,
                with_payload=['source_path', 'file_type'],
                with_vectors=False,
            )

            if not results:
                break

            for point in results:
                if point.payload is None:
                    continue
                path = point.payload.get('source_path', '')
                ftype = point.payload.get('file_type', '')

                # Apply path_prefix filter client-side (Qdrant doesn't support prefix)
                if path_prefix and not (path == path_prefix or path.startswith(path_prefix + '/')):
                    continue

                if path in path_data:
                    _, count = path_data[path]
                    path_data[path] = (ftype, count + 1)
                else:
                    path_data[path] = (ftype, 1)

            if offset is None:
                break

        # Sort by count descending and apply limit
        sorted_paths = sorted(
            [(path, ftype, count) for path, (ftype, count) in path_data.items()],
            key=lambda x: x[2],
            reverse=True,
        )

        if limit:
            sorted_paths = sorted_paths[:limit]

        return sorted_paths

    async def _ensure_file_type_index(self) -> None:
        """Create keyword index on file_type for faceting if not exists.

        This is idempotent - checks if index exists before creating.
        Index creation happens asynchronously in the background.
        """
        if not await self.collection_exists():
            return

        # Check if file_type index already exists in payload schema
        info = await self._client.get_collection(self._collection_name)
        payload_schema = getattr(info, 'payload_schema', {}) or {}
        if 'file_type' in payload_schema:
            logger.debug('file_type index already exists')
            return

        await self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name='file_type',
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.debug('Created file_type keyword index')
