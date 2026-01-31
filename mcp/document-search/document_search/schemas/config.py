"""Embedding configuration schema.

Manages persistent embedding model configuration for the document search index.
Supports migration between embedding models via clear + reconfigure workflow.
"""

from __future__ import annotations

from typing import Literal

from document_search.schemas.base import StrictModel

__all__ = [
    'EmbeddingConfig',
    'EmbeddingProvider',
    'GeminiConfig',
    'OpenRouterConfig',
    'create_config',
    'default_config',
]

# Provider type
type EmbeddingProvider = Literal['gemini', 'openrouter']


class GeminiConfig(StrictModel):
    """Gemini embedding configuration.

    Requires requests_per_minute for API quota enforcement.
    """

    provider: Literal['gemini'] = 'gemini'
    embedding_model: str
    embedding_dimensions: int
    batch_size: int
    requests_per_minute: int

    @classmethod
    def default(cls) -> GeminiConfig:
        """Create default Gemini config."""
        return cls(
            embedding_model='gemini-embedding-001',
            embedding_dimensions=768,
            batch_size=100,  # Max per Gemini API call
            requests_per_minute=3000,
        )


class OpenRouterConfig(StrictModel):
    """OpenRouter embedding configuration.

    Uses semaphore-based concurrency only (no rate limiting).
    """

    provider: Literal['openrouter'] = 'openrouter'
    embedding_model: str
    embedding_dimensions: int
    batch_size: int
    requests_per_minute: int | None = None

    @classmethod
    def default(cls) -> OpenRouterConfig:
        """Create default OpenRouter config."""
        return cls(
            embedding_model='qwen/qwen3-embedding-8b',
            embedding_dimensions=768,
            batch_size=1000,
        )


# Discriminated union - type alias for annotations
type EmbeddingConfig = GeminiConfig | OpenRouterConfig


def default_config(provider: EmbeddingProvider = 'gemini') -> EmbeddingConfig:
    """Create config with defaults for the specified provider."""
    if provider == 'openrouter':
        return OpenRouterConfig.default()
    return GeminiConfig.default()


def create_config(
    provider: EmbeddingProvider,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
) -> EmbeddingConfig:
    """Create config with optional overrides from provider defaults."""
    if provider == 'openrouter':
        openrouter = OpenRouterConfig.default()
        return OpenRouterConfig(
            embedding_model=embedding_model or openrouter.embedding_model,
            embedding_dimensions=embedding_dimensions or openrouter.embedding_dimensions,
            batch_size=openrouter.batch_size,
        )
    gemini = GeminiConfig.default()
    return GeminiConfig(
        embedding_model=embedding_model or gemini.embedding_model,
        embedding_dimensions=embedding_dimensions or gemini.embedding_dimensions,
        batch_size=gemini.batch_size,
        requests_per_minute=gemini.requests_per_minute,
    )
