"""Embedding operation schemas.

Typed models for Gemini embedding API. Enforces API constraints at the type level.

Verified constraints (2026-01):
- Batch size: Max 100 items per request (hard error)
- Text length: Silently truncates to ~2048 tokens (no error, data loss)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

import pydantic

from document_search.schemas.base import StrictModel

__all__ = [
    'EmbedBatchRequest',
    'EmbedBatchResponse',
    'EmbedRequest',
    'EmbedResponse',
    'MAX_TEXT_CHARS',
    'TaskType',
]

# Task type for embedding optimization.
# RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for search queries.
type TaskType = Literal[
    'RETRIEVAL_DOCUMENT',
    'RETRIEVAL_QUERY',
    'SEMANTIC_SIMILARITY',
    'CLASSIFICATION',
    'CLUSTERING',
    'QUESTION_ANSWERING',
    'FACT_VERIFICATION',
]

# Max characters before truncation risk (~2048 tokens * ~4 chars/token, conservative)
MAX_TEXT_CHARS = 6000


class EmbedRequest(StrictModel):
    """Single text embedding request."""

    text: Annotated[str, pydantic.Field(max_length=MAX_TEXT_CHARS)]
    task_type: TaskType = 'RETRIEVAL_DOCUMENT'


class EmbedBatchRequest(StrictModel):
    """Batch embedding request.

    Enforces Gemini API limit of 100 items per batch.
    Individual texts are validated against MAX_TEXT_CHARS to prevent silent truncation.
    """

    texts: Annotated[Sequence[str], pydantic.Field(max_length=100)]
    task_type: TaskType = 'RETRIEVAL_DOCUMENT'

    @pydantic.field_validator('texts')
    @classmethod
    def validate_text_lengths(cls, v: Sequence[str]) -> Sequence[str]:
        """Validate individual text lengths to prevent silent API truncation."""
        for i, text in enumerate(v):
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(f'Text at index {i} exceeds max length of {MAX_TEXT_CHARS} characters')
        return v


class EmbedResponse(StrictModel):
    """Single embedding result."""

    values: tuple[float, ...]
    dimensions: int


class EmbedBatchResponse(StrictModel):
    """Batch embedding results."""

    embeddings: Sequence[EmbedResponse]
