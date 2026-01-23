"""Low-level Qdrant vector database client.

Thin wrapper around qdrant-client. Handles API calls only - no business logic.
Type translation happens in the repository layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict
from uuid import UUID

from qdrant_client import QdrantClient as _QdrantClient
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
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


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
    """Low-level Qdrant client for vector operations."""

    DEFAULT_URL = 'http://localhost:6333'
    DEFAULT_COLLECTION = 'document_chunks'

    def __init__(
        self,
        url: str = DEFAULT_URL,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        """Initialize client.

        Args:
            url: Qdrant server URL.
            collection_name: Default collection for operations.
        """
        self._url = url
        self._collection_name = collection_name
        self._client = _QdrantClient(url=url)

    def ensure_collection(self, vector_dimension: int) -> None:
        """Create collection with hybrid search support if it doesn't exist.

        Creates a collection with:
        - Dense vectors: Semantic embeddings (Gemini, 768 dimensions)
        - Sparse vectors: BM25 keyword embeddings (fastembed)

        Args:
            vector_dimension: Size of dense embedding vectors (e.g., 768 for Gemini).
        """
        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)

        if not exists:
            self._client.create_collection(
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

    def upsert(
        self,
        points: Sequence[tuple[UUID, Sequence[float], Sequence[int], Sequence[float], Mapping[str, Any]]],
    ) -> int:
        """Insert or update points with hybrid vectors.

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

        self._client.upsert(
            collection_name=self._collection_name,
            points=point_structs,
        )
        return len(point_structs)

    def search(
        self,
        dense_vector: Sequence[float],
        sparse_indices: Sequence[int],
        sparse_values: Sequence[float],
        limit: int = 10,
        score_threshold: float | None = None,
        file_types: Sequence[str] | None = None,
    ) -> Sequence[SearchResultDict]:
        """Hybrid search combining dense and sparse vectors with RRF fusion.

        Args:
            dense_vector: Semantic query vector (Gemini embedding).
            sparse_indices: BM25 sparse vector indices.
            sparse_values: BM25 sparse vector values.
            limit: Maximum results.
            score_threshold: Minimum similarity score.
            file_types: Filter by file types.

        Returns:
            List of results with id, score, and payload.

        Note:
            Uses Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
            Path prefix filtering is handled at the repository layer via
            post-filtering, as Qdrant doesn't support native prefix matching.
        """
        # Build filter conditions
        conditions = []
        if file_types:
            conditions.append(FieldCondition(key='file_type', match=MatchAny(any=list(file_types))))

        query_filter = Filter(must=conditions) if conditions else None

        # Hybrid search with RRF fusion
        # Prefetch 50 candidates from each search, fuse with RRF, return top `limit`
        results = self._client.query_points(
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
        ]

    def get(self, point_ids: Sequence[UUID]) -> list[dict[str, Any]]:
        """Retrieve points by ID.

        Args:
            point_ids: IDs to retrieve.

        Returns:
            List of points with id and payload.
        """
        results = self._client.retrieve(
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
        ]

    def delete(self, point_ids: Sequence[UUID]) -> int:
        """Delete points by ID.

        Args:
            point_ids: IDs to delete.

        Returns:
            Number of points requested for deletion (not confirmed count).
            Qdrant's delete is idempotent - non-existent IDs are silently ignored.
        """
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=[str(pid) for pid in point_ids],
        )
        return len(point_ids)

    def delete_by_source_path(self, source_path: str) -> None:
        """Delete all points for a specific source file.

        Used when re-indexing a file to remove old chunks before inserting new ones.

        Args:
            source_path: Exact source_path to match.
        """
        self._client.delete(
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

    def count(self) -> int:
        """Get total point count in collection."""
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0

    def collection_exists(self) -> bool:
        """Check if collection exists."""
        collections = self._client.get_collections().collections
        return any(c.name == self._collection_name for c in collections)

    def get_collection_info(self) -> CollectionInfoDict | None:
        """Get collection metadata.

        Returns:
            Typed dict with name, vector_dimension, points_count, status.
            None if collection doesn't exist.
        """
        if not self.collection_exists():
            return None

        info = self._client.get_collection(self._collection_name)

        # Handle both named vectors (dict) and single vector config
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            # Named vectors - get dense vector dimension
            dense_config = vectors_config.get('dense')
            vector_dimension = dense_config.size if dense_config else 0
        else:
            # Single vector config (legacy)
            vector_dimension = vectors_config.size

        return CollectionInfoDict(
            name=self._collection_name,
            vector_dimension=vector_dimension,
            points_count=info.points_count or 0,
            status=str(info.status),
        )
