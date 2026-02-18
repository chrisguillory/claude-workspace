"""Pipeline tracer for per-item timing and queue depth monitoring.

Collects per-item timing events and queue depth samples during pipeline
execution, then produces a frozen PipelineTimingReport on completion.

Replaces the log-only _QueueMonitor with stored, typed time series data.

Thread safety: All record() calls happen in the main asyncio thread.
ProcessPoolExecutor workers don't call the tracer — the main thread
records before/after run_in_executor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence, Set
from pathlib import Path
from typing import TYPE_CHECKING

from document_search.schemas.tracing import (
    PercentileStats,
    PipelineTimingReport,
    QueueDepthSample,
    StageCompletionData,
    StageTimingReport,
    TimingEvent,
    TraceableStage,
)

if TYPE_CHECKING:
    from document_search.services.indexing import _ChunkedFile, _EmbeddedFile

__all__ = [
    'PipelineTracer',
]

logger = logging.getLogger(__name__)

# Ordered stages for report generation
_ALL_STAGES: Sequence[TraceableStage] = ('chunk', 'embed', 'embed_dense', 'embed_sparse', 'store')

# Stages with queue wait (have an input queue)
_QUEUED_STAGES: Set[TraceableStage] = {'chunk', 'embed', 'store'}

# Items processed before this count are flagged as warm-up
_WARMUP_ITEMS = 100


class PipelineTracer:
    """Collects per-item timing events and queue depth samples.

    Mutable during pipeline execution. Produces a frozen
    PipelineTimingReport via build_report() on completion.

    Absorbs _QueueMonitor responsibility — replaces log-only
    queue monitoring with stored typed time series data.
    """

    def __init__(
        self,
        file_queue: asyncio.Queue[Path],
        embed_queue: asyncio.Queue[_ChunkedFile],
        upsert_queue: asyncio.Queue[_EmbeddedFile],
    ) -> None:
        self._start = time.perf_counter()
        self._scan_seconds: float = 0.0

        # Queue references for depth sampling
        self._file_queue = file_queue
        self._embed_queue = embed_queue
        self._upsert_queue = upsert_queue

        # Per-item timing: item_id → {stage:event → timestamp}
        self._events: dict[str, dict[str, float]] = {}

        # Per-item CPU time: item_id → {stage → cpu_seconds}
        self._cpu_times: dict[str, dict[str, float]] = {}

        # Per-item Rust-internal wall time: item_id → {stage → wall_seconds}
        self._wall_times: dict[str, dict[str, float]] = {}

        # Rayon thread count (set via set_sparse_threads)
        self._sparse_threads: int | None = None

        # Per-item batch size: item_id → batch_size
        self._batch_sizes: dict[str, int] = {}

        # Queue depth time series
        self._queue_depths: list[QueueDepthSample] = []

        # Monitor task
        self._monitor_task: asyncio.Task[None] | None = None

        # Track item order for warm-up flagging
        self._item_order: list[str] = []

    @property
    def start_time(self) -> float:
        """Pipeline start time (perf_counter)."""
        return self._start

    # ── Per-item timing ─────────────────────────────────────────

    def record(self, item_id: str, stage: TraceableStage, event: TimingEvent) -> None:
        """Record a wall-clock timing event for an item at a stage."""
        key = f'{stage}:{event}'
        item_events = self._events.setdefault(item_id, {})
        item_events[key] = time.perf_counter()

        # Track item order (first event for each item)
        if len(item_events) == 1:
            self._item_order.append(item_id)

    def record_cpu(self, item_id: str, stage: TraceableStage, cpu_seconds: float) -> None:
        """Record CPU time for a stage (sum of per-task durations from Rust)."""
        cpus = self._cpu_times.setdefault(item_id, {})
        cpus[stage] = cpu_seconds

    def record_wall(self, item_id: str, stage: TraceableStage, wall_seconds: float) -> None:
        """Record Rust-internal wall time for a stage (parallel section latency)."""
        walls = self._wall_times.setdefault(item_id, {})
        walls[stage] = wall_seconds

    def set_sparse_threads(self, thread_count: int) -> None:
        """Store rayon thread count for efficiency ratio computation."""
        self._sparse_threads = thread_count

    def record_batch_size(self, item_id: str, batch_size: int) -> None:
        """Record the batch size this item was processed in."""
        self._batch_sizes[item_id] = batch_size

    def record_scan_seconds(self, seconds: float) -> None:
        """Record total scan phase duration (pipeline-level, not per-item)."""
        self._scan_seconds = seconds

    # ── Queue depth monitoring ──────────────────────────────────

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start periodic queue depth sampling."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop(interval))

    def stop_monitoring(self) -> None:
        """Stop periodic queue depth sampling."""
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None

    def get_queue_depths(self) -> Sequence[QueueDepthSample]:
        """Return collected queue depth samples (for live dashboard)."""
        return self._queue_depths

    def get_in_flight_counts(self) -> tuple[int, int, int]:
        """Count items currently being processed in each stage.

        Returns (chunk, embed, store) counts of items that have
        entered a stage but not yet completed or errored.
        """
        chunk_in = 0
        embed_in = 0
        store_in = 0
        for events in self._events.values():
            if 'chunk:started' in events and 'chunk:completed' not in events and 'chunk:errored' not in events:
                chunk_in += 1
            if 'embed:dequeued' in events and 'embed:completed' not in events and 'embed:errored' not in events:
                embed_in += 1
            if 'store:started' in events and 'store:completed' not in events and 'store:errored' not in events:
                store_in += 1
        return chunk_in, embed_in, store_in

    def get_completion_counts(self) -> tuple[int, int, int]:
        """Count files that have completed each stage.

        Returns (chunk, embed, store) counts of files with completion events.
        """
        chunk_done = sum(1 for events in self._events.values() if 'chunk:completed' in events)
        embed_done = sum(1 for events in self._events.values() if 'embed:completed' in events)
        store_done = sum(1 for events in self._events.values() if 'store:completed' in events)
        return chunk_done, embed_done, store_done

    # ── Report generation ───────────────────────────────────────

    def build_report(self) -> PipelineTimingReport:
        """Compute final timing report from collected data."""
        total_elapsed = time.perf_counter() - self._start

        # Identify warm-up items (first N)
        warmup_ids = set(self._item_order[:_WARMUP_ITEMS])

        stage_reports: list[StageTimingReport] = []

        for stage in _ALL_STAGES:
            processing_times = self._collect_processing_times(stage, warmup_ids)
            if not processing_times:
                continue

            queue_wait = self._collect_queue_wait(stage, warmup_ids)
            batch_wait = self._collect_batch_wait(stage, warmup_ids)
            cpu_times = self._collect_cpu_times(stage, warmup_ids)
            wall_times = self._collect_wall_times(stage, warmup_ids)

            throughput = len(processing_times) / total_elapsed if total_elapsed > 0 else 0.0

            # Average batch size for embed stage
            avg_batch: float | None = None
            if stage == 'embed':
                batch_sizes = [
                    self._batch_sizes[item_id]
                    for item_id in self._events
                    if item_id not in warmup_ids and item_id in self._batch_sizes
                ]
                if batch_sizes:
                    avg_batch = sum(batch_sizes) / len(batch_sizes)

            stage_reports.append(
                StageTimingReport(
                    stage=stage,
                    processing=_percentiles(processing_times),
                    queue_wait=_percentiles(queue_wait) if queue_wait else None,
                    batch_wait=_percentiles(batch_wait) if batch_wait else None,
                    cpu=_percentiles(cpu_times) if cpu_times else None,
                    wall=_percentiles(wall_times) if wall_times else None,
                    throughput_per_sec=round(throughput, 2),
                    avg_batch_size=round(avg_batch, 1) if avg_batch is not None else None,
                )
            )

        completion_series = self._build_completion_series()

        return PipelineTimingReport(
            scan_seconds=round(self._scan_seconds, 3),
            stages=tuple(stage_reports),
            queue_depth_series=tuple(self._queue_depths),
            total_items=len(self._events),
            total_elapsed_seconds=round(total_elapsed, 3),
            sparse_threads=self._sparse_threads,
            completion_series=completion_series,
        )

    # ── Private: queue monitoring ────────────────────────────────

    async def _monitor_loop(self, interval: float) -> None:
        """Sample queue depths, in-flight counts, and completion counts periodically."""
        while True:
            await asyncio.sleep(interval)
            chunk_in, embed_in, store_in = self.get_in_flight_counts()
            chunk_done, embed_done, store_done = self.get_completion_counts()
            self._queue_depths.append(
                QueueDepthSample(
                    elapsed_seconds=round(time.perf_counter() - self._start, 3),
                    file_queue=self._file_queue.qsize(),
                    embed_queue=self._embed_queue.qsize(),
                    upsert_queue=self._upsert_queue.qsize(),
                    chunk_in_flight=chunk_in,
                    embed_in_flight=embed_in,
                    store_in_flight=store_in,
                    files_chunk_done=chunk_done,
                    files_embed_done=embed_done,
                    files_store_done=store_done,
                )
            )

    # ── Private: completion series ─────────────────────────────

    def _build_completion_series(self) -> Sequence[StageCompletionData]:
        """Build per-item completion data for all stages.

        Includes ALL items (no warm-up exclusion) so the client can see
        the full timeline including ramp-up behavior. Sorted by completion
        time for efficient client-side binary search windowing.
        """
        series: list[StageCompletionData] = []
        for stage in _ALL_STAGES:
            start_key, end_key = _stage_event_keys(stage)
            pairs: list[tuple[float, float]] = []
            for events in self._events.values():
                start = events.get(start_key)
                end = events.get(end_key)
                if start is not None and end is not None:
                    pairs.append((end - self._start, (end - start) * 1000))
            if not pairs:
                continue
            pairs.sort()
            series.append(
                StageCompletionData(
                    stage=stage,
                    completions=tuple(round(p[0], 3) for p in pairs),
                    durations=tuple(round(p[1], 3) for p in pairs),
                )
            )
        return tuple(series)

    # ── Private: metric collection ──────────────────────────────

    def _collect_processing_times(
        self,
        stage: TraceableStage,
        warmup_ids: Set[str],
    ) -> Sequence[float]:
        """Collect processing times (completed - started) for a stage in ms."""
        start_key, end_key = _stage_event_keys(stage)
        times: list[float] = []
        for item_id, events in self._events.items():
            if item_id in warmup_ids:
                continue
            start = events.get(start_key)
            end = events.get(end_key)
            if start is not None and end is not None:
                times.append((end - start) * 1000)
        return times

    def _collect_queue_wait(
        self,
        stage: TraceableStage,
        warmup_ids: Set[str],
    ) -> Sequence[float]:
        """Collect queue wait times (started/dequeued - queued) for a stage in ms."""
        if stage not in _QUEUED_STAGES:
            return []

        waits: list[float] = []
        queued_event = f'{stage}:queued'
        # For embed, queue wait ends at dequeued (entering accumulator)
        # For chunk/store, queue wait ends at started
        pickup_event = f'{stage}:dequeued' if stage == 'embed' else f'{stage}:started'

        for item_id, events in self._events.items():
            if item_id in warmup_ids:
                continue
            queued = events.get(queued_event)
            picked_up = events.get(pickup_event)
            if queued is not None and picked_up is not None:
                waits.append((picked_up - queued) * 1000)
        return waits

    def _collect_batch_wait(
        self,
        stage: TraceableStage,
        warmup_ids: Set[str],
    ) -> Sequence[float]:
        """Collect batch accumulation wait (batch_started - dequeued) in ms."""
        if stage != 'embed':
            return []

        waits: list[float] = []
        for item_id, events in self._events.items():
            if item_id in warmup_ids:
                continue
            dequeued = events.get('embed:dequeued')
            batch_started = events.get('embed:batch_started')
            if dequeued is not None and batch_started is not None:
                waits.append((batch_started - dequeued) * 1000)
        return waits

    def _collect_cpu_times(
        self,
        stage: TraceableStage,
        warmup_ids: Set[str],
    ) -> Sequence[float]:
        """Collect CPU times for a stage in ms."""
        times: list[float] = []
        for item_id, cpus in self._cpu_times.items():
            if item_id in warmup_ids:
                continue
            cpu = cpus.get(stage)
            if cpu is not None:
                times.append(cpu * 1000)
        return times

    def _collect_wall_times(
        self,
        stage: TraceableStage,
        warmup_ids: Set[str],
    ) -> Sequence[float]:
        """Collect Rust-internal wall times for a stage in ms."""
        times: list[float] = []
        for item_id, walls in self._wall_times.items():
            if item_id in warmup_ids:
                continue
            wall = walls.get(stage)
            if wall is not None:
                times.append(wall * 1000)
        return times


# ── Module-level helpers ────────────────────────────────────────


def _stage_event_keys(stage: TraceableStage) -> tuple[str, str]:
    """Return (start_event_key, end_event_key) for stage processing time.

    Embed uses batch_started → completed to measure processing time from
    when the batch starts processing (after accumulation wait), excluding
    queue wait (dequeued → batch_started). All other stages measure from
    their started event.
    """
    if stage == 'embed':
        return f'{stage}:batch_started', f'{stage}:completed'
    return f'{stage}:started', f'{stage}:completed'


def _percentiles(values: Sequence[float]) -> PercentileStats:
    """Compute percentile stats from a list of values in ms."""
    if not values:
        return PercentileStats(p50_ms=0, p95_ms=0, p99_ms=0, max_ms=0, count=0)
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return PercentileStats(
        p50_ms=round(sorted_vals[int(n * 0.50)], 3),
        p95_ms=round(sorted_vals[min(int(n * 0.95), n - 1)], 3),
        p99_ms=round(sorted_vals[min(int(n * 0.99), n - 1)], 3),
        max_ms=round(sorted_vals[-1], 3),
        count=n,
    )
