"""Embedding configuration schema.

Manages persistent embedding model configuration for the document search index.
Supports migration between embedding models via clear + reconfigure workflow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pydantic

from document_search.schemas.base import StrictModel

__all__ = [
    'CONFIG_PATH',
    'EmbeddingConfig',
    'MigrationRequiredError',
    'load_config',
    'save_config',
]

logger = logging.getLogger(__name__)

# Config file location
CONFIG_PATH = Path.home() / '.claude-workspace' / 'config' / 'document_search.json'

# New defaults for fresh installations
DEFAULT_MODEL = 'gemini-embedding-001'
DEFAULT_DIMENSIONS = 768
DEFAULT_REQUESTS_PER_MINUTE = 3000  # Tier 1 limit (free tier is 100 RPD)


class EmbeddingConfig(StrictModel):
    """Embedding model configuration.

    Persisted to ~/.claude-workspace/config/document_search.json.
    Changes require clearing the index (embeddings are model-specific).
    """

    embedding_model: str
    embedding_dimensions: int
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE

    @classmethod
    def default(cls) -> EmbeddingConfig:
        """Create config with current defaults."""
        return cls(
            embedding_model=DEFAULT_MODEL,
            embedding_dimensions=DEFAULT_DIMENSIONS,
            requests_per_minute=DEFAULT_REQUESTS_PER_MINUTE,
        )


class MigrationRequiredError(Exception):
    """Raised when index needs migration to a new embedding model.

    Provides guidance for the user to clear and re-index.
    """

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error


def load_config() -> EmbeddingConfig | None:
    """Load config from file if it exists.

    Returns:
        EmbeddingConfig if file exists and is valid, None otherwise.

    Raises:
        ValueError: If config file exists but is invalid.
    """
    if not CONFIG_PATH.exists():
        return None

    try:
        data = json.loads(CONFIG_PATH.read_text())
        return EmbeddingConfig.model_validate(data)
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
