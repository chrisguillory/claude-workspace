"""Protocol definitions for embedding clients.

Defines the interface that embedding clients must satisfy.
All embedding clients implement this protocol.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
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

    @property
    def http_p50_ms(self) -> float:
        """Median HTTP round-trip time across all API calls."""
        ...

    @property
    def http_p99_ms(self) -> float:
        """99th percentile HTTP round-trip time across all API calls."""
        ...

    async def embed(self, texts: Sequence[str], *, intent: TaskIntent) -> Sequence[EmbeddingVector]:
        """Embed texts into vectors."""
        ...

    on_transient_error: Callable[[TransientErrorCategory], None]
    """Hook fired from the retry loop's ``before_sleep`` on any categorized
    transient error — before the retry sleep, not after the call returns.

    Reassign per-instance to wire adaptive behavior (e.g. AIMD batch sizing).
    """

    async def close(self) -> None:
        """Release resources. No-op for clients without external connections."""
        ...
