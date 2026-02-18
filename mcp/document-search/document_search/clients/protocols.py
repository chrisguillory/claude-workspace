"""Protocol definitions for embedding clients.

Defines the interface that embedding clients must satisfy.
All embedding clients implement this protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from document_search.schemas.embeddings import TaskIntent

__all__ = [
    'EmbeddingClient',
]


class EmbeddingClient(Protocol):
    """Protocol for embedding clients.

    Any client with a compatible `embed` method satisfies this protocol.
    Used by EmbeddingService for type-safe client injection.
    """

    async def embed(self, texts: Sequence[str], *, intent: TaskIntent) -> Sequence[Sequence[float]]:
        """Embed texts into vectors.

        Args:
            texts: Texts to embed.
            intent: 'document' for indexing, 'query' for search.
                Each provider translates to their specific format.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    async def close(self) -> None:
        """Release resources. No-op for clients without external connections."""
        ...

    @property
    def errors_429(self) -> int:
        """Cumulative count of 429 rate limit errors encountered.

        Used for dashboard monitoring. Resets only on client recreation.
        """
        ...
