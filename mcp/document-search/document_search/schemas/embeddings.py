"""Embedding operation schemas.

Provider-agnostic types for embedding operations.
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
    'TaskIntent',
]

# Embedding intent - document (for indexing) or query (for search)
# Each provider translates this to their specific format
type TaskIntent = Literal['document', 'query']

# Max characters before truncation risk with most embedding models
# Conservative limit that works across providers (~2048 tokens * ~3 chars/token)
MAX_TEXT_CHARS = 6000


class EmbedRequest(StrictModel):
    """Single text embedding request."""

    text: Annotated[str, pydantic.Field(max_length=MAX_TEXT_CHARS)]
    intent: TaskIntent = 'document'


class EmbedBatchRequest(StrictModel):
    """Batch embedding request."""

    texts: Sequence[str]
    intent: TaskIntent

    @pydantic.field_validator('texts')
    @classmethod
    def validate_text_lengths(cls, v: Sequence[str]) -> Sequence[str]:
        """Validate individual text lengths to prevent silent truncation."""
        for i, text in enumerate(v):
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(f'Text at index {i} exceeds max length of {MAX_TEXT_CHARS} characters')
        return v


class EmbedResponse(StrictModel):
    """Single embedding result."""

    values: Sequence[float]
    dimensions: int


class EmbedBatchResponse(StrictModel):
    """Batch embedding results."""

    embeddings: Sequence[EmbedResponse]
