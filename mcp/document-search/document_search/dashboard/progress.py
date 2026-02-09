"""Progress writer for dashboard operation monitoring.

Writes OperationState snapshots to files for dashboard consumption.
Called by server.py during indexing operations.

Each operation gets:
- {operation_id}.json: State snapshots (progress, result, error)
- {operation_id}.log: Full debug logs from the operation's lifetime
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from uuid6 import uuid7

from document_search.paths import OPERATIONS_DIR
from document_search.schemas.dashboard import OperationProgress, OperationState
from document_search.schemas.indexing import IndexingResult

__all__ = [
    'ProgressWriter',
]


class ProgressWriter:
    """Writes operation progress and logs to files for dashboard consumption.

    Files: ~/.claude-workspace/document_search/operations/{operation_id}.{json,log}
    JSON updates written on-demand (called by server.py monitoring loop).
    Log file captures all logger output during the operation's lifetime.
    """

    def __init__(self, mcp_server_pid: int) -> None:
        self._pid = mcp_server_pid
        self._operation_id: str | None = None
        self._state: OperationState | None = None
        self._log_handler: logging.FileHandler | None = None

    def start_operation(
        self,
        collection_name: str,
        directory: str,
    ) -> str:
        """Start tracking a new operation.

        Creates the operation JSON file and attaches a per-operation log handler
        to the root logger. All log output during the operation is captured.

        Args:
            collection_name: Name of the collection being indexed.
            directory: Directory being indexed.

        Returns:
            operation_id for this operation.
        """
        op_id = str(uuid7())
        now = _now()

        self._operation_id = op_id
        self._state = OperationState(
            operation_id=op_id,
            mcp_server_pid=self._pid,
            collection_name=collection_name,
            directory=directory,
            created_at=now,
            updated_at=now,
            ended_at=None,
            progress=None,
            result=None,
            error=None,
        )

        OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Attach per-operation log file to root logger
        log_path = OPERATIONS_DIR / f'{op_id}.log'
        self._log_handler = logging.FileHandler(log_path)
        self._log_handler.setLevel(logging.DEBUG)
        self._log_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logging.getLogger().addHandler(self._log_handler)

        self._write()
        return op_id

    def update_progress(self, progress: OperationProgress) -> None:
        """Update operation progress.

        Called from server.py monitoring loop.

        Args:
            progress: Current progress snapshot.
        """
        if self._state is None:
            return

        self._state = OperationState(
            operation_id=self._state.operation_id,
            mcp_server_pid=self._state.mcp_server_pid,
            collection_name=self._state.collection_name,
            directory=self._state.directory,
            created_at=self._state.created_at,
            updated_at=_now(),
            ended_at=None,
            progress=progress,
            result=None,
            error=None,
        )

        self._write()

    def complete_with_success(self, result: IndexingResult) -> None:
        """Mark operation as successfully completed.

        Builds final progress from result with all queues drained.

        Args:
            result: Final indexing result.
        """
        if self._state is None:
            return

        # Build final progress from result (always, even if monitor never ran)
        prior_429 = self._state.progress.errors_429 if self._state.progress else 0
        final_progress = OperationProgress(
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
            chunks_ingested=result.chunks_created,
            chunks_embedded=result.chunks_created,
            embed_cache_hits=result.embed_cache_hits,
            embed_cache_misses=result.embed_cache_misses,
            chunks_stored=result.chunks_created,
            files_done=result.files_indexed + result.files_no_content,
            errors_429=prior_429,
        )

        self._finalize(progress=final_progress, result=result, error=None)

    def complete_with_error(self, error: str) -> None:
        """Mark operation as failed, preserving last known progress.

        The last progress snapshot is kept with status updated to 'failed',
        so the dashboard shows how far the operation got before crashing.

        Args:
            error: Error message with traceback.
        """
        if self._state is None:
            return

        # Preserve last known progress, update status to failed
        final_progress: OperationProgress | None = None
        if self._state.progress is not None:
            final_progress = self._state.progress.model_copy(update={'status': 'failed'})

        self._finalize(progress=final_progress, result=None, error=error)

    def _finalize(
        self,
        *,
        progress: OperationProgress | None,
        result: IndexingResult | None,
        error: str | None,
    ) -> None:
        """Write final state and close log handler."""
        if self._state is None:
            return

        now = _now()
        self._state = OperationState(
            operation_id=self._state.operation_id,
            mcp_server_pid=self._state.mcp_server_pid,
            collection_name=self._state.collection_name,
            directory=self._state.directory,
            created_at=self._state.created_at,
            updated_at=now,
            ended_at=now,
            progress=progress,
            result=result,
            error=error,
        )

        self._write()
        self._close_log_handler()

    def _write(self) -> None:
        """Write state to file atomically (temp + rename)."""
        if self._state is None or self._operation_id is None:
            return

        OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = OPERATIONS_DIR / f'{self._operation_id}.json'
        temp_path = file_path.with_suffix('.tmp')

        temp_path.write_text(json.dumps(self._state.model_dump(mode='json'), indent=2) + '\n')
        temp_path.rename(file_path)

    def _close_log_handler(self) -> None:
        """Flush and remove per-operation log handler."""
        if self._log_handler is not None:
            self._log_handler.flush()
            self._log_handler.close()
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None


_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
_LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'


def _now() -> datetime:
    """Current time in local timezone."""
    return datetime.now().astimezone()
