"""Progress writer for dashboard operation monitoring.

Writes OperationState snapshots to files for dashboard consumption.
Called by server.py during indexing operations.

Each operation gets:
- {operation_id}.json: State snapshots (progress, result, error)
- {operation_id}.log: Full debug logs from the operation's lifetime

Staleness detection: tracks a composite progress fingerprint across
consecutive updates. If the fingerprint is unchanged for >5 minutes,
the operation status is set to 'stalled'. The fingerprint covers all
pipeline counters so that progress in any stage clears the stall.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from uuid6 import uuid7

from document_search.clients.redis import ConnectionStats
from document_search.paths import OPERATIONS_DIR
from document_search.schemas.dashboard import OperationProgress, OperationState
from document_search.schemas.indexing import IndexingResult
from document_search.schemas.tracing import PipelineTimingReport

__all__ = [
    'ProgressWriter',
]

logger = logging.getLogger(__name__)

# How long progress must be unchanged before marking stalled (seconds).
STALL_THRESHOLD_SECONDS = 300


class ProgressWriter:
    """Writes operation progress and logs to files for dashboard consumption.

    Files: ~/.claude-workspace/document_search/operations/{operation_id}.{json,log}
    JSON updates written on-demand (called by server.py monitoring loop).
    Log file captures all logger output during the operation's lifetime.

    Staleness detection: on each update, a fingerprint of progress counters
    is compared to the last-changed fingerprint. If unchanged for longer
    than STALL_THRESHOLD_SECONDS, the status is set to 'stalled' and
    stalled_since is recorded. Progress resuming clears the stall.
    """

    def __init__(self, mcp_server_pid: int) -> None:
        self._pid = mcp_server_pid
        self._operation_id: str | None = None
        self._state: OperationState | None = None
        self._log_handler: logging.FileHandler | None = None

        # Staleness tracking: composite fingerprint of progress counters
        self._last_fingerprint: tuple[int, ...] | None = None
        self._last_progress_time: datetime | None = None
        self._stalled_since: datetime | None = None

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
        self._last_fingerprint = None
        self._last_progress_time = None
        self._stalled_since = None
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

    def update_timing(self, timing: PipelineTimingReport) -> None:
        """Write timing detail to separate file (called every ~5s).

        The timing file contains the full PipelineTimingReport including
        completion series data for the interactive chart. Kept separate
        from the main progress JSON to avoid writing large data at 500ms.
        """
        if self._operation_id is None:
            return
        OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = OPERATIONS_DIR / f'{self._operation_id}-timing.json'
        temp_path = file_path.with_suffix('.tmp')
        temp_path.write_text(json.dumps(timing.model_dump(mode='json'), indent=2) + '\n')
        temp_path.rename(file_path)

    def update_progress(self, progress: OperationProgress) -> None:
        """Update operation progress with staleness detection.

        Compares a composite fingerprint of pipeline counters against the
        last-changed snapshot. If unchanged for >STALL_THRESHOLD_SECONDS,
        the progress status is changed to 'stalled' and stalled_since is set.

        The fingerprint covers all counters that could indicate forward
        movement in any pipeline stage, so legitimate pauses in one stage
        (e.g. chunks_stored flat while embed is working) are not flagged
        as long as some counter is advancing.

        Called from server.py monitoring loop (~500ms).

        Args:
            progress: Current progress snapshot.
        """
        if self._state is None:
            return

        now = _now()
        fingerprint = _progress_fingerprint(progress)

        if self._last_fingerprint is None or fingerprint != self._last_fingerprint:
            # Progress is advancing — reset staleness tracking
            self._last_fingerprint = fingerprint
            self._last_progress_time = now
            self._stalled_since = None
        elif self._last_progress_time is not None:
            elapsed = (now - self._last_progress_time).total_seconds()
            if elapsed >= STALL_THRESHOLD_SECONDS and self._stalled_since is None:
                self._stalled_since = now
                logger.warning(
                    'Operation %s stalled: no progress for %.0fs',
                    self._operation_id,
                    elapsed,
                )

        # Apply stalled status if detected
        if self._stalled_since is not None:
            progress = progress.model_copy(
                update={'status': 'stalled', 'stalled_since': self._stalled_since},
            )

        self._state = OperationState(
            operation_id=self._state.operation_id,
            mcp_server_pid=self._state.mcp_server_pid,
            collection_name=self._state.collection_name,
            directory=self._state.directory,
            created_at=self._state.created_at,
            updated_at=now,
            ended_at=None,
            progress=progress,
            result=None,
            error=None,
        )

        self._write()

    def complete_with_success(
        self,
        result: IndexingResult,
        redis_conn_stats: ConnectionStats | None = None,
    ) -> None:
        """Mark operation as successfully completed.

        Carries forward per-stage file counters from the last live snapshot
        to avoid a visible counter jump (chunk-cached and no-content files
        skip stages but from_result would count them in all stages).

        Args:
            result: Final indexing result.
            redis_conn_stats: Final Redis connection diagnostics (post-drain HWM).
        """
        if self._state is None:
            return

        prior = self._state.progress
        final_progress = OperationProgress.from_result(
            result,
            transient_errors=dict(prior.transient_errors) if prior else {},
            prior_progress=prior,
            redis_conn_stats=redis_conn_stats,
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
        self._clean_timing_file()
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

    def _clean_timing_file(self) -> None:
        """Remove timing file on completion (data is in the result JSON)."""
        if self._operation_id is not None:
            timing_path = OPERATIONS_DIR / f'{self._operation_id}-timing.json'
            timing_path.unlink(missing_ok=True)

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
    """Current time in local timezone for dashboard display.

    Uses local tz (not UTC) so dashboard timestamps match the user's clock.
    """
    return datetime.now().astimezone()


def _progress_fingerprint(progress: OperationProgress) -> tuple[int, ...]:
    """Composite fingerprint of all counters that indicate pipeline movement.

    Covers every stage so that progress in any single stage resets the
    stall timer. Includes both chunk-level and file-level counters.
    """
    return (
        progress.files_found,
        progress.files_done,
        progress.files_cached,
        progress.files_errored,
        progress.files_chunked,
        progress.files_embedded,
        progress.files_stored,
        progress.chunks_ingested,
        progress.chunks_embedded,
        progress.chunks_stored,
        progress.embed_cache_hits,
        progress.embed_cache_misses,
    )
