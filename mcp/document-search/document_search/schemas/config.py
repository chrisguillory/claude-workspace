"""Embedding configuration schema.

Manages persistent embedding model configuration for the document search index.
Supports migration between embedding models via clear + reconfigure workflow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic
from pydantic import Field, TypeAdapter

from document_search.schemas.base import StrictModel

if TYPE_CHECKING:
    from document_search.schemas.vectors import EmbeddingInfo

__all__ = [
    'CONFIG_PATH',
    'EmbeddingConfig',
    'EmbeddingProvider',
    'GeminiConfig',
    'OpenRouterConfig',
    'create_config',
    'default_config',
    'load_config',
    'save_config',
]

logger = logging.getLogger(__name__)

# Config file location
CONFIG_PATH = Path.home() / '.claude-workspace' / 'config' / 'document_search.json'

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

    def to_info(self) -> EmbeddingInfo:
        """Convert to display schema."""
        from document_search.schemas.vectors import EmbeddingInfo

        return EmbeddingInfo(
            provider=self.provider,
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
            batch_size=self.batch_size,
            requests_per_minute=self.requests_per_minute,
        )


class OpenRouterConfig(StrictModel):
    """OpenRouter embedding configuration.

    Uses semaphore-based concurrency only (no rate limiting).
    """

    provider: Literal['openrouter'] = 'openrouter'
    embedding_model: str
    embedding_dimensions: int
    batch_size: int

    @classmethod
    def default(cls) -> OpenRouterConfig:
        """Create default OpenRouter config."""
        return cls(
            embedding_model='qwen/qwen3-embedding-8b',
            embedding_dimensions=768,
            batch_size=1000,
        )

    def to_info(self) -> EmbeddingInfo:
        """Convert to display schema."""
        from document_search.schemas.vectors import EmbeddingInfo

        return EmbeddingInfo(
            provider=self.provider,
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
            batch_size=self.batch_size,
            requests_per_minute=None,  # Not applicable
        )


# Discriminated union - type alias for annotations
type EmbeddingConfig = GeminiConfig | OpenRouterConfig

# TypeAdapter for deserializing with discriminator
_config_adapter: TypeAdapter[GeminiConfig | OpenRouterConfig] = TypeAdapter(
    Annotated[GeminiConfig | OpenRouterConfig, Field(discriminator='provider')]
)


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


def load_config() -> EmbeddingConfig | None:
    """Load config from file if it exists.

    Automatically migrates legacy config formats to current schema.

    Returns:
        EmbeddingConfig if file exists and is valid, None otherwise.

    Raises:
        ValueError: If config file exists but is invalid.
    """
    if not CONFIG_PATH.exists():
        return None

    try:
        data = json.loads(CONFIG_PATH.read_text())

        # Auto-migrate legacy format
        original_data = data.copy()
        data = _migrate_legacy_config(data)

        config = _config_adapter.validate_python(data)

        # Re-save if migrated
        if data != original_data:
            save_config(config)
            logger.info(f'Migrated config saved to {CONFIG_PATH}')

        return config
    except (json.JSONDecodeError, pydantic.ValidationError) as e:
        raise ValueError(f'Invalid config file at {CONFIG_PATH}: {e}') from e


def save_config(config: EmbeddingConfig) -> None:
    """Save config to file, creating parent directories if needed."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            config.model_dump(mode='json'),
            indent=2,
        )
        + '\n'
    )
    logger.info(f'Saved embedding config: model={config.embedding_model}, dimensions={config.embedding_dimensions}')


def _migrate_legacy_config(data: dict[str, Any]) -> dict[str, Any]:  # strict_typing_linter.py: mutable-type
    """Migrate legacy config format to current schema.

    Legacy format (pre-v0.4.1):
        {"embedding_model": "...", "embedding_dimensions": 768, "requests_per_minute": 3000}

    Current format:
        {"provider": "gemini", "embedding_model": "...", "embedding_dimensions": 768,
         "batch_size": 100, "requests_per_minute": 3000}
    """
    if 'provider' in data:
        return data  # Already migrated

    logger.info('Migrating legacy config format')

    # Infer provider from existing fields
    if 'requests_per_minute' in data:
        data['provider'] = 'gemini'
        data.setdefault('batch_size', 100)
    else:
        data['provider'] = 'openrouter'
        data.setdefault('batch_size', 1000)

    return data
