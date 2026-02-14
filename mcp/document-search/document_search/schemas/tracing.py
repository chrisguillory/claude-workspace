"""Pipeline tracing schemas for per-item observability.

Frozen Pydantic models for timing reports, queue depth time series,
and bottleneck identification. All models are immutable after creation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from document_search.schemas.base import StrictModel

__all__ = [
    'EmbedSubStage',
    'PercentileStats',
    'PipelineStage',
    'PipelineTimingReport',
    'QueueDepthSample',
    'StageTimingReport',
    'TimingEvent',
    'TraceableStage',
]

# Stage hierarchy: top-level stages have queues, sub-stages are within embed
type PipelineStage = Literal['chunk', 'embed', 'store']
type EmbedSubStage = Literal['embed_dense', 'embed_sparse']
type TraceableStage = PipelineStage | EmbedSubStage

# Events recorded at stage boundaries
type TimingEvent = Literal[
    'queued',  # placed on input queue
    'dequeued',  # worker accepted from queue (embed: into accumulator)
    'batch_started',  # batch processing began (embed flush_batch)
    'started',  # processing began (chunk, store)
    'completed',  # processing finished
    'errored',  # processing failed
]


class PercentileStats(StrictModel):
    """Distribution statistics for a timing metric."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    count: int


class StageTimingReport(StrictModel):
    """Timing report for a single pipeline stage.

    Top-level stages (chunk, embed, store) have queue_wait.
    The embed stage has batch_wait (accumulation delay).
    Sub-stages (embed_dense, embed_sparse) have processing only.
    """

    stage: TraceableStage
    processing: PercentileStats
    queue_wait: PercentileStats | None = None
    batch_wait: PercentileStats | None = None
    cpu: PercentileStats | None = None
    throughput_per_sec: float
    avg_batch_size: float | None = None


class QueueDepthSample(StrictModel):
    """Point-in-time queue depth measurement (1Hz sampling)."""

    elapsed_seconds: float
    file_queue: int
    embed_queue: int
    upsert_queue: int


class PipelineTimingReport(StrictModel):
    """Complete timing report for a pipeline run.

    Produced by PipelineTracer.build_report() after pipeline completion.
    Includes per-stage timing percentiles and queue depth time series.
    """

    scan_seconds: float
    stages: Sequence[StageTimingReport]
    queue_depth_series: Sequence[QueueDepthSample]
    total_items: int
    total_elapsed_seconds: float
