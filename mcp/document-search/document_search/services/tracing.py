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
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from document_search.schemas.tracing import (
    PercentileStats,
    PipelineTimingReport,
    QueueDepthSample,
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
_QUEUED_STAGES: frozenset[TraceableStage] = frozenset({'chunk', 'embed', 'store'})

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

        # Per-item batch size: item_id → batch_size
        self._batch_sizes: dict[str, int] = {}

        # Queue depth time series
        self._queue_depths: list[QueueDepthSample] = []

        # Monitor task
        self._monitor_task: asyncio.Task[None] | None = None

        # Track item order for warm-up flagging
        self._item_order: list[str] = []

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
        """Record CPU time for a stage (measured in subprocess)."""
        cpus = self._cpu_times.setdefault(item_id, {})
        cpus[stage] = cpu_seconds

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
            if 'embed:dequeued' in events and 'embed:completed' not in events:
                embed_in += 1
            if 'store:started' in events and 'store:completed' not in events:
                store_in += 1
        return chunk_in, embed_in, store_in

    # ── Report generation ───────────────────────────────────────

    def build_report(self) -> PipelineTimingReport:
        """Compute final timing report from collected data."""
        total_elapsed = time.perf_counter() - self._start

        # Identify warm-up items (first N)
        warmup_ids = frozenset(self._item_order[:_WARMUP_ITEMS])

        stage_reports: list[StageTimingReport] = []

        for stage in _ALL_STAGES:
            processing_times = self._collect_processing_times(stage, warmup_ids)
            if not processing_times:
                continue

            queue_wait = self._collect_queue_wait(stage, warmup_ids)
            batch_wait = self._collect_batch_wait(stage, warmup_ids)
            cpu_times = self._collect_cpu_times(stage, warmup_ids)

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
                    throughput_per_sec=round(throughput, 2),
                    avg_batch_size=round(avg_batch, 1) if avg_batch is not None else None,
                )
            )

        return PipelineTimingReport(
            scan_seconds=round(self._scan_seconds, 3),
            stages=tuple(stage_reports),
            queue_depth_series=tuple(self._queue_depths),
            total_items=len(self._events),
            total_elapsed_seconds=round(total_elapsed, 3),
        )

    # ── Private: queue monitoring ────────────────────────────────

    async def _monitor_loop(self, interval: float) -> None:
        """Sample queue depths periodically."""
        while True:
            await asyncio.sleep(interval)
            self._queue_depths.append(
                QueueDepthSample(
                    elapsed_seconds=round(time.perf_counter() - self._start, 3),
                    file_queue=self._file_queue.qsize(),
                    embed_queue=self._embed_queue.qsize(),
                    upsert_queue=self._upsert_queue.qsize(),
                )
            )

    # ── Private: metric collection ──────────────────────────────

    def _collect_processing_times(
        self,
        stage: TraceableStage,
        warmup_ids: frozenset[str],
    ) -> Sequence[float]:
        """Collect processing times (completed - started) for a stage in ms."""
        times: list[float] = []
        # Sub-stages and top-level embed use batch_started as the start event
        if stage in ('embed', 'embed_dense', 'embed_sparse'):
            start_event = f'{stage}:batch_started' if stage == 'embed' else f'{stage}:started'
        else:
            start_event = f'{stage}:started'
        end_event = f'{stage}:completed'

        for item_id, events in self._events.items():
            if item_id in warmup_ids:
                continue
            start = events.get(start_event)
            end = events.get(end_event)
            if start is not None and end is not None:
                times.append((end - start) * 1000)  # Convert to ms
        return times

    def _collect_queue_wait(
        self,
        stage: TraceableStage,
        warmup_ids: frozenset[str],
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
        warmup_ids: frozenset[str],
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
        warmup_ids: frozenset[str],
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


# ── Module-level helpers ────────────────────────────────────────


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
