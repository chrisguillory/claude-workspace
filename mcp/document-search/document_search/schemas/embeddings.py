"""Embedding operation schemas.

Provider-agnostic types for embedding operations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

import numpy as np
import pydantic
from cc_lib.schemas import StrictModel
from numpy.typing import NDArray

__all__ = [
    'MAX_TEXT_CHARS',
    'EmbedBatchRequest',
    'EmbedBatchResponse',
    'EmbedRequest',
    'EmbedResponse',
    'EmbeddingVector',
    'SparseIndices',
    'SparseValues',
    'TaskIntent',
]

# Embedding intent - document (for indexing) or query (for search)
# Each provider translates this to their specific format
type TaskIntent = Literal['document', 'query']

# A single embedding vector: Python list (from API) or numpy array (from cache)
type EmbeddingVector = Sequence[float] | NDArray[np.float32]

# Sparse vector components: Python list (from cold path) or numpy array (from pipeline)
type SparseIndices = Sequence[int] | NDArray[np.int32]
type SparseValues = Sequence[float] | NDArray[np.float32]

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
    """Single embedding result.

    Values may be a Python list (from API) or numpy array (from cache).
    The hot path (indexing) keeps numpy for 8x memory savings; the cold
    path (search) calls list(response.values) when native types are needed.
    """

    values: EmbeddingVector
    dimensions: int

    @classmethod
    def from_numpy(cls, values: NDArray[np.float32]) -> EmbedResponse:
        """Construct from numpy array (cache hot path).

        Bypasses Pydantic validation — numpy arrays satisfy Sequence[float]
        at runtime but Pydantic can't generate a schema for NDArray.
        """
        return cls.model_construct(values=values, dimensions=len(values))


class EmbedBatchResponse(StrictModel):
    """Batch embedding results."""

    embeddings: Sequence[EmbedResponse]
