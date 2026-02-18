"""API clients for external services."""

from __future__ import annotations

from document_search.clients.gemini import GeminiClient
from document_search.clients.openrouter import OpenRouterClient
from document_search.clients.protocols import EmbeddingClient
from document_search.clients.qdrant import QdrantClient
from document_search.clients.redis import RedisClient
from document_search.schemas.config import EmbeddingConfig, GeminiConfig, OpenRouterConfig

__all__ = [
    'EmbeddingClient',
    'GeminiClient',
    'OpenRouterClient',
    'QdrantClient',
    'RedisClient',
    'create_embedding_client',
]


def create_embedding_client(config: EmbeddingConfig) -> EmbeddingClient:
    """Create embedding client based on configuration.

    Args:
        config: Embedding configuration (GeminiConfig or OpenRouterConfig).

    Returns:
        Configured embedding client.
    """
    match config:
        case GeminiConfig():
            return GeminiClient(
                model=config.embedding_model,
                output_dimensionality=config.embedding_dimensions,
                requests_per_minute=config.requests_per_minute,
            )
        case OpenRouterConfig():
            return OpenRouterClient(
                model=config.embedding_model,
                dimensions=config.embedding_dimensions,
                requests_per_minute=config.requests_per_minute,
            )

    raise TypeError(f'Unknown config type: {type(config).__name__}')
