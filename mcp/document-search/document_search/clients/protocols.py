"""Protocol definitions for embedding clients.

Defines the interface that embedding clients must satisfy.
All embedding clients implement this protocol.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal, Protocol

from document_search.schemas.embeddings import EmbeddingVector, TaskIntent

__all__ = [
    'EmbeddingClient',
    'TransientErrorCategory',
]

type TransientErrorCategory = Literal[
    'bad_gateway',
    'empty_response',
    'provider_unavailable',
    'rate_limit',
    'server_error',
    'timeout',
    'truncated_response',
]


class EmbeddingClient(Protocol):
    """Protocol for embedding clients.

    Any client with a compatible `embed` method satisfies this protocol.
    Used by EmbeddingService for type-safe client injection.
    """

    transient_errors: Counter[TransientErrorCategory]
    """Categorized transient error counts.

    Used for dashboard monitoring. Resets only on client recreation.
    """

    async def embed(self, texts: Sequence[str], *, intent: TaskIntent) -> Sequence[EmbeddingVector]:
        """Embed texts into vectors."""
        ...

    def on_transient_error(self, category: TransientErrorCategory) -> None:
        """Called from before_sleep on any categorized transient error.

        Fires inside the retry loop — before the retry sleep, not after
        the call returns. Override to wire adaptive behavior.
        """
        ...

    async def close(self) -> None:
        """Release resources. No-op for clients without external connections."""
        ...
