"""Vector storage schemas for Qdrant operations.

Typed models for vector database operations. These schemas define
the interface between the service layer and the Qdrant client.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal
from uuid import UUID

import pydantic

from document_search.schemas.base import StrictModel
from document_search.schemas.chunking import Chunk, FileType

__all__ = [
    'CollectionInfo',
    'FileIndexReason',
    'FileIndexStatus',
    'IndexBreakdown',
    'IndexedFile',
    'SearchHit',
    'SearchQuery',
    'SearchResult',
    'SearchType',
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
# - 'embedding': Dense vectors only (Gemini embeddings). Pure neural similarity search.
#   Primarily useful for debugging or comparing search strategies. In practice,
#   hybrid mode provides better results for most use cases.
type SearchType = Literal['hybrid', 'lexical', 'embedding']


class VectorPoint(StrictModel):
    """A point to store in Qdrant with hybrid vectors.

    Combines a chunk with dense (semantic) and sparse (BM25) embeddings.
    The chunk data is stored as payload for retrieval.
    """

    id: UUID
    # Dense vector for semantic similarity (Gemini embeddings)
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


class CollectionInfo(StrictModel):
    """Qdrant collection metadata."""

    name: str
    vector_dimension: int
    points_count: int
    status: str


# Visibility schemas for index introspection


class IndexBreakdown(StrictModel):
    """Overview of what's in the document index."""

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
