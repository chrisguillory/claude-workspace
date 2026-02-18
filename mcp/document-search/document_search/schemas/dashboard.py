"""Dashboard schemas.

Shared data models for:
- MCP server registration and dashboard coordination
- Operation progress tracking and monitoring
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from local_lib.types import JsonDatetime

if TYPE_CHECKING:
    from document_search.services.indexing import PipelineSnapshot

from document_search.schemas.base import StrictModel
from document_search.schemas.chunking import FileType
from document_search.schemas.indexing import IndexingResult
from document_search.schemas.tracing import QueueDepthSample

__all__ = [
    'DashboardState',
    'McpServer',
    'OperationProgress',
    'OperationState',
]


class McpServer(StrictModel):
    """A registered MCP server instance.

    Identified by PID with started_at for robustness against PID reuse.
    """

    pid: int
    started_at: JsonDatetime


class DashboardState(StrictModel):
    """Persisted dashboard coordination state.

    Tracks the dashboard server and registered MCP servers.
    File location: paths.DASHBOARD_STATE_PATH
    """

    port: int
    server_pid: int
    mcp_servers: Sequence[McpServer]


# Operation status type
type OperationStatus = Literal['running', 'complete', 'failed']


class OperationProgress(StrictModel):
    """Progress snapshot for an indexing operation.

    Updated every 1-2 seconds during indexing.
    Supports concurrent scanning and pipeline processing.
    """

    # Overall status
    status: OperationStatus
    elapsed_seconds: float

    # Scanning (discovery phase)
    scan_complete: bool
    files_found: int
    files_to_process: int
    files_cached: int
    files_errored: int

    # Pipeline queues (file counts waiting at each stage)
    files_awaiting_chunk: int
    files_awaiting_embed: int
    files_awaiting_store: int

    # In-flight (files being processed inside workers)
    files_in_chunk: int
    files_in_embed: int
    files_in_store: int

    # Pipeline stages (cumulative chunk counts through each stage)
    chunks_ingested: int  # Output of chunk stage
    chunks_embedded: int  # Output of embed stage
    embed_cache_hits: int  # Embeddings served from Redis
    embed_cache_misses: int  # Embeddings computed via API
    chunks_stored: int  # Output of store stage
    chunks_skipped: int  # Unchanged chunks (chunk-level cache)

    # Per-stage file completion (files that finished each stage)
    files_chunked: int
    files_embedded: int
    files_stored: int

    # Completion
    files_done: int
    errors_429: int

    # File type breakdown (live during scan, mirrors IndexingResult.by_file_type)
    by_file_type: Mapping[FileType, str]

    # Queue depth time series (1Hz samples from tracer)
    queue_depth_series: Sequence[QueueDepthSample]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PipelineSnapshot,
        *,
        status: OperationStatus,
        errors_429: int,
    ) -> OperationProgress:
        """Build from PipelineSnapshot with cross-layer metrics.

        Args:
            snapshot: PipelineSnapshot from IndexingService.
            status: Lifecycle status ('running', 'complete', 'failed').
            errors_429: Count of rate limit errors (tracked by embedding client).

        Returns:
            Complete OperationProgress ready for dashboard.
        """
        return cls(
            status=status,
            elapsed_seconds=snapshot.elapsed_seconds,
            scan_complete=snapshot.scan_complete,
            files_found=snapshot.files_found,
            files_to_process=snapshot.files_to_process,
            files_cached=snapshot.files_cached,
            files_errored=snapshot.files_errored,
            files_awaiting_chunk=snapshot.files_awaiting_chunk,
            files_awaiting_embed=snapshot.files_awaiting_embed,
            files_awaiting_store=snapshot.files_awaiting_store,
            files_in_chunk=snapshot.files_in_chunk,
            files_in_embed=snapshot.files_in_embed,
            files_in_store=snapshot.files_in_store,
            chunks_ingested=snapshot.chunks_ingested,
            chunks_embedded=snapshot.chunks_embedded,
            embed_cache_hits=snapshot.embed_cache_hits,
            embed_cache_misses=snapshot.embed_cache_misses,
            chunks_stored=snapshot.chunks_stored,
            chunks_skipped=snapshot.chunks_skipped,
            files_chunked=snapshot.files_chunked,
            files_embedded=snapshot.files_embedded,
            files_stored=snapshot.files_stored,
            files_done=snapshot.files_done,
            errors_429=errors_429,
            by_file_type=snapshot.by_file_type,
            queue_depth_series=snapshot.queue_depth_series,
        )

    @classmethod
    def from_result(
        cls,
        result: IndexingResult,
        *,
        errors_429: int,
    ) -> OperationProgress:
        """Build final progress from IndexingResult with all queues drained.

        Single source of truth for result → progress conversion.
        Eliminates manual field-by-field construction in progress writer.

        Args:
            result: Completed IndexingResult from IndexingService.
            errors_429: Count of rate limit errors from prior progress snapshots.

        Returns:
            Complete OperationProgress with status='complete'.
        """
        return cls(
            status='complete',
            elapsed_seconds=result.elapsed_seconds,
            scan_complete=True,
            files_found=result.files_scanned,
            files_to_process=result.files_indexed + result.files_no_content,
            files_cached=result.files_cached,
            files_errored=len(result.errors),
            files_awaiting_chunk=0,
            files_awaiting_embed=0,
            files_awaiting_store=0,
            files_in_chunk=0,
            files_in_embed=0,
            files_in_store=0,
            chunks_ingested=result.chunks_created,
            chunks_embedded=result.chunks_created,
            embed_cache_hits=result.embed_cache_hits,
            embed_cache_misses=result.embed_cache_misses,
            chunks_stored=result.chunks_created,
            chunks_skipped=result.chunks_skipped,
            files_chunked=result.files_indexed + result.files_no_content,
            files_embedded=result.files_indexed + result.files_no_content,
            files_stored=result.files_indexed + result.files_no_content,
            files_done=result.files_indexed + result.files_no_content,
            errors_429=errors_429,
            by_file_type=result.by_file_type,
            queue_depth_series=(),
        )


class OperationState(StrictModel):
    """Full state for an indexing operation.

    Path: ~/.claude-workspace/document_search/operations/{operation_id}.json
    """

    operation_id: str
    mcp_server_pid: int
    collection_name: str
    directory: str

    created_at: JsonDatetime
    updated_at: JsonDatetime
    ended_at: JsonDatetime | None

    progress: OperationProgress | None
    result: IndexingResult | None
    error: str | None
