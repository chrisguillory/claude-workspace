"""Pydantic schemas for document search operations."""

from __future__ import annotations

from document_search.schemas.base import StrictModel
from document_search.schemas.chunking import (
    EXTENSION_MAP,
    Chunk,
    ChunkMetadata,
    ChunkResult,
    FileType,
    get_file_type,
)
from document_search.schemas.config import (
    CONFIG_PATH,
    EmbeddingConfig,
    GeminiConfig,
    OpenRouterConfig,
    create_config,
    default_config,
    load_config,
    save_config,
)
from document_search.schemas.embeddings import (
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
    TaskIntent,
)
from document_search.schemas.indexing import (
    CHUNK_STRATEGY_VERSION,
    DirectoryIndexState,
    ErrorCategory,
    FileIndexState,
    FileProcessingError,
    IndexingProgress,
    IndexingResult,
    ProgressCallback,
)
from document_search.schemas.vectors import (
    SearchHit,
    SearchQuery,
    SearchResult,
    VectorPoint,
)

__all__ = [
    # Base
    'StrictModel',
    # Config
    'CONFIG_PATH',
    'EmbeddingConfig',
    'GeminiConfig',
    'OpenRouterConfig',
    'create_config',
    'default_config',
    'load_config',
    'save_config',
    # Chunking
    'FileType',
    'EXTENSION_MAP',
    'get_file_type',
    'ChunkMetadata',
    'Chunk',
    'ChunkResult',
    # Embeddings
    'TaskIntent',
    'EmbedRequest',
    'EmbedBatchRequest',
    'EmbedResponse',
    'EmbedBatchResponse',
    # Vectors
    'SearchHit',
    'SearchQuery',
    'SearchResult',
    'VectorPoint',
    # Indexing
    'CHUNK_STRATEGY_VERSION',
    'FileIndexState',
    'DirectoryIndexState',
    'FileProcessingError',
    'ErrorCategory',
    'IndexingResult',
    'IndexingProgress',
    'ProgressCallback',
]
