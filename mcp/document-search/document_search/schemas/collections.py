"""Collection management schemas.

Defines collection metadata and registry for multi-collection support.
Each collection has its own embedding provider configuration.
"""

from __future__ import annotations

from collections.abc import Mapping

from local_lib.types import JsonDatetime

from document_search.schemas.base import StrictModel
from document_search.schemas.config import EmbeddingProvider

__all__ = [
    'Collection',
    'CollectionRegistry',
]


class Collection(StrictModel):
    """A document collection with its embedding configuration.

    Each collection is stored as a separate Qdrant collection and uses
    a specific embedding provider for all its vectors.
    """

    name: str
    description: str | None = None
    created_at: JsonDatetime
    provider: EmbeddingProvider


class CollectionRegistry(StrictModel, frozen=False):
    """Registry of all document collections.

    Persisted to disk and shared between MCP server and dashboard.
    frozen=False allows mutation for add/remove operations.
    """

    collections: Mapping[str, Collection] = {}
