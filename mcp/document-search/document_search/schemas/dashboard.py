"""Dashboard schemas.

Shared data models for:
- MCP server registration and dashboard coordination
- Operation progress tracking and monitoring
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from local_lib.types import JsonDatetime

if TYPE_CHECKING:
    from document_search.services.indexing import PipelineSnapshot

from document_search.schemas.base import StrictModel
from document_search.schemas.indexing import IndexingResult

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

    # Pipeline stages (cumulative chunk counts through each stage)
    chunks_ingested: int  # Output of chunk stage
    chunks_embedded: int  # Output of embed stage
    chunks_stored: int  # Output of store stage

    # Completion
    files_done: int
    errors_429: int

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
            chunks_ingested=snapshot.chunks_ingested,
            chunks_embedded=snapshot.chunks_embedded,
            chunks_stored=snapshot.chunks_stored,
            files_done=snapshot.files_done,
            errors_429=errors_429,
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
