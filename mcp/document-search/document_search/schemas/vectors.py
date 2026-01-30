"""Vector storage schemas for Qdrant operations.

Typed models for vector database operations. These schemas define
the interface between the service layer and the Qdrant client.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal
from uuid import UUID

import pydantic
from local_lib.types import JsonDatetime

from document_search.schemas.base import StrictModel
from document_search.schemas.chunking import Chunk, FileType
from document_search.schemas.config import EmbeddingProvider

__all__ = [
    'ClearResult',
    'CollectionMetadata',
    'ContentStats',
    'EmbeddingInfo',
    'FileIndexReason',
    'FileIndexStatus',
    'IndexedFile',
    'IndexInfo',
    'SearchHit',
    'SearchQuery',
    'SearchResult',
    'SearchType',
    'StorageStats',
    'VectorPoint',
]

# Search strategy type - explicit naming to avoid terminology confusion.
#
# - 'hybrid': Dense + sparse vectors with RRF fusion. Industry standard approach that
#   combines the precision of keyword matching with conceptual understanding from
#   embeddings. Recommended for most queries. This is the default.
#
# - 'lexical': BM25 sparse vectors only. Traditional keyword/full-text search that
#   matches on word tokens (lexemes). Best for exact term matching, symbol lookup,
#   and identifier search where you want precise matches without conceptual noise.
#
# - 'embedding': Dense vectors only. Pure neural similarity search.
#   Primarily useful for debugging or comparing search strategies. In practice,
#   hybrid mode provides better results for most use cases.
type SearchType = Literal['hybrid', 'lexical', 'embedding']


class VectorPoint(StrictModel):
    """A point to store in Qdrant with hybrid vectors.

    Combines a chunk with dense (semantic) and sparse (BM25) embeddings.
    The chunk data is stored as payload for retrieval.
    """

    id: UUID
    # Dense vector for semantic similarity
    dense_vector: Sequence[float]
    # Sparse vector for keyword matching (BM25)
    sparse_indices: Sequence[int]
    sparse_values: Sequence[float]
    # Payload contains chunk data for retrieval
    source_path: str
    chunk_index: int
    file_type: FileType
    text: str
    # Flattened metadata for Qdrant filtering
    start_char: int
    end_char: int
    heading_context: str | None = None
    page_number: int | None = None
    json_path: str | None = None

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        dense_vector: Sequence[float],
        sparse_indices: Sequence[int],
        sparse_values: Sequence[float],
        point_id: UUID,
    ) -> VectorPoint:
        """Create VectorPoint from Chunk and embeddings."""
        return cls(
            id=point_id,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            source_path=chunk.source_path,
            chunk_index=chunk.chunk_index,
            file_type=chunk.file_type,
            text=chunk.text,
            start_char=chunk.metadata.start_char,
            end_char=chunk.metadata.end_char,
            heading_context=chunk.metadata.heading_context,
            page_number=chunk.metadata.page_number,
            json_path=chunk.metadata.json_path,
        )


class SearchQuery(StrictModel):
    """Vector similarity search query with configurable strategy."""

    search_type: SearchType = 'hybrid'
    # Dense vector for embedding similarity (required for hybrid/embedding modes)
    dense_vector: Sequence[float] | None = None
    # Sparse vector for keyword matching (required for hybrid/lexical modes)
    sparse_indices: Sequence[int] | None = None
    sparse_values: Sequence[float] | None = None
    limit: Annotated[int, pydantic.Field(ge=1, le=100)] = 10
    score_threshold: float | None = None
    # Optional filters
    file_types: Sequence[FileType] | None = None
    source_path_prefix: str | None = None


class SearchHit(StrictModel):
    """A single search result with score and chunk data."""

    id: UUID
    score: float
    source_path: str
    chunk_index: int
    file_type: FileType
    text: str
    start_char: int
    end_char: int
    heading_context: str | None = None
    page_number: int | None = None
    json_path: str | None = None


class SearchResult(StrictModel):
    """Search results container."""

    hits: Sequence[SearchHit]
    total: int


class CollectionMetadata(StrictModel):
    """Collection metadata from registry."""

    name: str
    description: str | None
    created_at: JsonDatetime
    provider: EmbeddingProvider


class EmbeddingInfo(StrictModel):
    """Embedding configuration for the collection's provider.

    Flattens provider-specific config into a single schema for display.
    Provider-specific fields are None when not applicable.
    """

    provider: EmbeddingProvider
    model: str
    dimensions: int
    batch_size: int
    requests_per_minute: int | None = None  # Gemini only


class StorageStats(StrictModel):
    """Qdrant storage statistics."""

    vector_dimension: int
    points_count: int
    status: str


# Visibility schemas for index introspection


class ContentStats(StrictModel):
    """Content breakdown statistics."""

    total_chunks: int
    by_file_type: Mapping[str, int]
    unique_files: int
    supported_types: Sequence[str]


type FileIndexReason = Literal['indexed', 'not_found', 'unsupported_type']


class FileIndexStatus(StrictModel):
    """Status of whether a specific file is indexed."""

    indexed: bool
    reason: FileIndexReason
    chunk_count: int
    file_type: FileType | None


class IndexedFile(StrictModel):
    """A file in the index with its chunk count."""

    path: str
    chunk_count: int
    file_type: FileType


class IndexInfo(StrictModel):
    """Comprehensive collection information.

    Combines collection metadata, embedding config, storage stats, and content breakdown.
    Content stats can be scoped by path.
    """

    collection: CollectionMetadata
    embedding: EmbeddingInfo
    storage: StorageStats
    content: ContentStats
    path: str | None = None  # Scope used for content stats (None = CWD)


class ClearResult(StrictModel):
    """Result of clearing documents from index."""

    files_removed: int
    chunks_removed: int
    path: str | None  # None means entire index was cleared
