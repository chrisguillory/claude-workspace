"""Vector storage schemas for Qdrant operations.

Typed models for vector database operations. These schemas define
the interface between the service layer and the Qdrant client.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

import pydantic

from document_search.schemas.base import StrictModel
from document_search.schemas.chunking import Chunk, FileType


class VectorPoint(StrictModel):
    """A point to store in Qdrant.

    Combines a chunk with its embedding vector for storage.
    The chunk data is stored as payload for retrieval.
    """

    id: UUID
    vector: Sequence[float]
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
    def from_chunk(cls, chunk: Chunk, vector: Sequence[float], point_id: UUID) -> VectorPoint:
        """Create VectorPoint from Chunk and embedding."""
        return cls(
            id=point_id,
            vector=vector,
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
    """Vector similarity search query."""

    vector: Sequence[float]
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
