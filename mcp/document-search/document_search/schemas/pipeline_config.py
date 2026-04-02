"""Pipeline configuration from environment variables.

All variables prefixed with DOCUMENT_SEARCH_. Example:
    DOCUMENT_SEARCH_EMBED_QUEUE_SIZE=200
    DOCUMENT_SEARCH_LOG_LEVEL=DEBUG
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    'PipelineConfig',
]


class PipelineConfig(BaseSettings):
    """Pipeline tuning parameters, validated at server startup."""

    model_config = SettingsConfigDict(
        env_prefix='DOCUMENT_SEARCH_',
        frozen=True,
    )

    # Queue sizes (backpressure between stages)
    embed_queue_size: int = Field(default=200, ge=1)
    upsert_queue_size: int = Field(default=200, ge=1)

    # Logging
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'

    @field_validator('log_level', mode='before')
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Accept lowercase log levels (e.g., 'debug' → 'DEBUG')."""
        return v.upper() if isinstance(v, str) else v
