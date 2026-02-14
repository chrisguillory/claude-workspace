"""Indexing operation schemas.

Models for tracking indexing state, progress, and results.
Supports incremental indexing via content hash comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Literal

import pydantic
from local_lib.types import JsonDatetime, JsonUuid

from document_search.schemas.base import StrictModel
from document_search.schemas.chunking import FileType
from document_search.schemas.tracing import PipelineTimingReport

__all__ = [
    'CHUNK_STRATEGY_VERSION',
    'DirectoryIndexState',
    'ErrorCategory',
    'FileIndexState',
    'FileProcessingError',
    'FileTypeStats',
    'IndexingProgress',
    'IndexingResult',
    'ProgressCallback',
    'StopAfterStage',
]

# Current chunking strategy version - bump when ChunkingService changes
# v2: Added PDF and CSV chunking support with parallel processing
# v3: Hybrid search (BM25 sparse vectors), email MIME parsing, min chunk filter
CHUNK_STRATEGY_VERSION = 3


class FileIndexState(StrictModel):
    """Persistent state for a single file's indexing.

    Tracks content hash and chunk IDs to enable incremental updates.
    When file hash changes, old chunks are deleted and new ones created.
    """

    file_path: str  # Absolute path
    file_hash: str  # SHA256 of file contents
    file_size: int  # Bytes, for quick change detection
    chunk_count: int
    chunk_ids: Sequence[JsonUuid]  # IDs of chunks in Qdrant
    indexed_at: JsonDatetime
    chunk_strategy_version: int = CHUNK_STRATEGY_VERSION


class DirectoryIndexState(StrictModel):
    """Persistent state for entire directory indexing.

    Stored as JSON at ~/.claude-workspace/cache/document_search_index_state.json
    """

    directory_path: str
    files: Mapping[str, FileIndexState]  # Keyed by path relative to directory
    last_full_scan: JsonDatetime
    total_chunks: int = 0
    total_files: int = 0
    metadata_version: int = 1


class FileProcessingError(StrictModel):
    """Single file processing error with actionable context."""

    file_path: str
    file_type: FileType | None = None  # For grouping errors by type
    error_type: str  # e.g., "UnicodeDecodeError", "PermissionError"
    message: str
    context: str | None = None  # Additional debug info or fix suggestion
    recoverable: bool = True  # Can user fix and retry?


class ErrorCategory(StrictModel):
    """Grouped errors by type with actionable guidance."""

    error_type: str  # "encoding", "permission", "api_error", etc.
    count: int
    action: str  # Human-readable fix suggestion
    files: Sequence[str]  # Affected file paths


class FileTypeStats(StrictModel):
    """Per-file-type indexing statistics."""

    scanned: int = 0  # Found with this extension
    indexed: int = 0  # Created > 0 chunks
    no_content: int = 0  # Processed, created 0 chunks
    cached: int = 0  # Hash unchanged
    errored: int = 0  # Failed
    chunks: int = 0  # Total chunks created

    def to_summary(self) -> str:
        """Format as compact string, omitting zero values."""
        parts = []
        if self.scanned:
            parts.append(f'scanned={self.scanned}')
        if self.indexed:
            parts.append(f'indexed={self.indexed}')
        if self.cached:
            parts.append(f'cached={self.cached}')
        if self.no_content:
            parts.append(f'no_content={self.no_content}')
        if self.errored:
            parts.append(f'errored={self.errored}')
        if self.chunks:
            parts.append(f'chunks={self.chunks}')
        return ' '.join(parts) if parts else 'empty'


# Pipeline stage boundaries for stop_after parameter
type StopAfterStage = Literal['scan', 'chunk', 'embed']


class IndexingResult(StrictModel):
    """Result of directory indexing operation."""

    # Scan phase
    files_scanned: int  # Total files found with supported extensions
    files_ignored: int = 0  # Files filtered by gitignore

    # Processing outcomes
    files_indexed: int  # Created > 0 chunks
    files_cached: int  # Hash unchanged, not reprocessed
    files_no_content: int = 0  # Processed but created 0 chunks

    # Chunk operations
    chunks_created: int
    chunks_deleted: int  # Old chunks soft-deleted
    embeddings_created: int
    embed_cache_hits: int  # Embeddings served from Redis
    embed_cache_misses: int  # Embeddings computed via API

    # Per-file-type breakdown (values are summary strings)
    by_file_type: Mapping[FileType, str] = {}

    # Index state after operation
    index_files: int = 0  # Total files in index
    index_chunks: int = 0  # Total chunks in index

    # Timing and errors
    elapsed_seconds: float
    errors: Sequence[FileProcessingError]

    # Pipeline control
    stopped_after: StopAfterStage | None = None

    # Pipeline tracing (populated when full pipeline runs)
    timing: PipelineTimingReport | None = None

    @property
    def success_rate(self) -> float:
        """Percentage of files processed without error."""
        total = self.files_indexed + self.files_no_content + len(self.errors)
        return (self.files_indexed + self.files_no_content) / total if total > 0 else 1.0

    @property
    def error_summary(self) -> str:
        """Human-readable error summary."""
        if not self.errors:
            return 'All files indexed successfully'

        grouped: dict[str, int] = {}
        for err in self.errors:
            grouped[err.error_type] = grouped.get(err.error_type, 0) + 1

        lines = [f'{len(self.errors)} errors:']
        for err_type, count in sorted(grouped.items()):
            lines.append(f'  {err_type}: {count}')
        return '\n'.join(lines)

    def errors_by_category(self) -> Sequence[ErrorCategory]:
        """Group errors with actionable guidance."""
        # Map error types to categories and actions
        category_map: dict[str, tuple[str, str]] = {
            'UnicodeDecodeError': ('encoding', 'Files are not UTF-8. Try: file -i <filename>'),
            'PermissionError': ('permission', 'Run: chmod u+r <file>'),
            'FileNotFoundError': ('missing', 'File was deleted or moved'),
            'JSONDecodeError': ('invalid_json', 'Fix JSON syntax errors'),
            'IsADirectoryError': ('directory', 'Path is a directory, not a file'),
        }
        default_action = 'Check file manually'

        grouped: dict[str, list[str]] = {}
        for err in self.errors:
            category, _ = category_map.get(err.error_type, (err.error_type, default_action))
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(err.file_path)

        return [
            ErrorCategory(
                error_type=category,
                count=len(files),
                action=category_map.get(category, (category, default_action))[1],
                files=files,
            )
            for category, files in grouped.items()
        ]


class IndexingProgress(StrictModel):
    """Progress update during indexing operation.

    Emitted via callback for real-time progress reporting.
    """

    files_scanned: int
    files_total: int
    files_indexed: int  # Created > 0 chunks
    files_cached: int  # Hash unchanged
    files_no_content: int = 0  # Processed, 0 chunks
    chunks_created: int
    embeddings_pending: int  # Chunks waiting to be embedded
    current_file: str | None = None
    current_phase: Annotated[str, pydantic.Field(pattern=r'^(scanning|chunking|embedding|storing)$')] = 'scanning'
    errors_so_far: int = 0
    elapsed_seconds: float = 0.0

    @property
    def percent_complete(self) -> float:
        """Completion percentage (0-100)."""
        if self.files_total == 0:
            return 0.0
        return (self.files_indexed + self.files_cached + self.files_no_content) / self.files_total * 100

    @property
    def estimated_remaining_seconds(self) -> float | None:
        """Estimated time remaining based on current rate."""
        processed = self.files_indexed + self.files_cached + self.files_no_content
        if processed == 0 or self.elapsed_seconds == 0:
            return None
        rate = processed / self.elapsed_seconds
        remaining = self.files_total - processed
        return remaining / rate if rate > 0 else None


# Type alias for progress callback
type ProgressCallback = Callable[[IndexingProgress], None]
