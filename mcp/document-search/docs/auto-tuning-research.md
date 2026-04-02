# Pipeline Auto-Tuning Research Report

Research into adaptive parameter tuning for the document-search indexing pipeline.
Goal: design a self-tuning system that adjusts pipeline parameters at runtime
based on observed metrics, similar to TCP congestion control.

**Date:** 2026-03-29

---

## Table of Contents

1. [Production Auto-Tuning Patterns](#1-production-auto-tuning-patterns)
2. [RAG-Specific Auto-Tuning](#2-rag-specific-auto-tuning)
3. [Parameter Classification](#3-parameter-classification)
4. [Controller Design](#4-controller-design)
5. [Empirical Test Plan](#5-empirical-test-plan)
6. [Anti-Patterns and Risks](#6-anti-patterns-and-risks)

---

## 1. Production Auto-Tuning Patterns

### 1.1 Netflix Concurrency Limits (Gradient2)

**The closest analogue to our problem.** Netflix's `concurrency-limits` library
dynamically adjusts the number of concurrent requests a service accepts, using
latency as a congestion signal. The library separates limit calculation
(algorithm) from enforcement (limiter), a pattern directly applicable to our
pipeline controller.

**Gradient2 algorithm (recommended variant):**

```
gradient = max(0.5, min(1.0, tolerance * longRtt / shortRtt))
newLimit = estimatedLimit * gradient + queueSize
newLimit = estimatedLimit * (1 - smoothing) + newLimit * smoothing
newLimit = clamp(newLimit, minLimit, maxLimit)
```

Key design decisions:
- **Uses smoothed average RTT, not minimum.** Minimum RTT is too sensitive to
  outliers in bursty workloads (embedding API latency is highly variable).
  The exponential moving average over 600 samples provides stability.
- **Tolerance factor (default 1.5)** allows the system to tolerate some latency
  increase before reducing limits. Without this, the controller would be overly
  aggressive during normal variance.
- **Smoothing (default 0.2)** dampens oscillation. Each new limit is 80% old +
  20% new, preventing large swings.
- **Drift prevention:** When `longRtt / shortRtt > 2`, applies rapid decay
  (`longRtt * 0.95`) to recover from load spikes without waiting for the
  exponential average to catch up.
- **Queue size allowance:** Adds headroom proportional to sqrt(limit) for small
  limits (fast initial growth) that diminishes for large limits (stability).

**AIMD variant (simpler):**
- Increase by 1 when inflight >= limit/2
- Decrease by `backoffRatio` (default 0.9) on drop or timeout
- Bounds: min=20, max=200
- Better for systems where the primary signal is errors rather than latency

**Applicability to our pipeline:** The gradient approach maps well to queue
depth. Our `embed_queue.qsize() / embed_queue.maxsize` ratio is analogous to
the RTT gradient. When the queue fill ratio approaches 1.0, the upstream
(chunk workers) should slow down. When it drops toward 0.0, we have spare
capacity to increase.

### 1.2 Apache Flink Buffer Debloating

**Most relevant streaming analogue.** Flink 1.14 introduced automatic buffer
sizing that adjusts in-flight data based on observed throughput.

**Core formula:**

```
target_buffers = throughput * target_consumption_time / buffer_size
```

For our pipeline, this translates to:

```
target_queue_size = embed_throughput_per_sec * target_latency_seconds
```

If embed workers process 50 files/sec and we want 2 seconds of buffered work:
`target_queue_size = 50 * 2 = 100`

Key design decisions:
- **Recalculation period:** 200ms default. Too fast causes churn; too slow
  misses workload transitions.
- **Change threshold:** 25% default. Only adjusts if the new value differs
  from current by more than 25%. This is a deadband that prevents
  micro-adjustments (the primary anti-oscillation mechanism).
- **Per-gate calculation:** Flink learned that subtask-level averaging fails
  when inputs have different throughputs. Moved to per-input-gate calculation.
  Lesson: our embed queue and upsert queue should be tuned independently.

### 1.3 Kubernetes HPA

**Best model for asymmetric scaling.**

**Core formula:**

```
desiredReplicas = ceil(currentReplicas * (currentMetric / desiredMetric))
```

Key design decisions that apply to our pipeline:
- **10% tolerance band:** No action if the ratio is within [0.9, 1.1]. This
  prevents constant micro-adjustments (deadband).
- **Asymmetric stabilization windows:** Scale-up: fast (0s default).
  Scale-down: slow (300s default). The reasoning: over-provisioning wastes
  resources, under-provisioning causes failures. For our pipeline: increasing
  queue sizes is cheap and safe; decreasing risks stalls.
- **selectPolicy:** When multiple recommendations exist within the stabilization
  window, HPA picks the maximum (most conservative for scale-down). For our
  pipeline: when deciding to reduce queue size, use the maximum recommended
  queue size from the last N samples.

### 1.4 Java ForkJoinPool (Work Stealing)

**Relevant for worker count adaptation.** ForkJoinPool dynamically adds,
suspends, or resumes worker threads based on task availability. Workers
maintain local deques; idle workers steal from busy workers' deques.

Applicability: If we made embed_workers dynamic, we could:
- Start with a conservative count (16)
- Spawn more workers when embed_queue stays above a threshold
- Suspend workers when queue empties
- Key constraint: each worker holds an API connection slot, so
  max_workers <= max_concurrent (semaphore limit)

### 1.5 Kafka Consumer Scaling (KEDA)

**Relevant for lag-based scaling.**

Formula: `desiredReplicas = ceil(currentReplicas * (currentLag / targetLag))`

Key lesson: simple lag-based thresholds cause oscillation. When consumers
scale up aggressively, lag drops to zero, causing immediate scale-down,
which causes lag to spike again. KEDA mitigates this with:
- Cooldown periods (separate for scale-up and scale-down)
- Maximum step size (at most 3 new pods in 15 seconds)
- Partition-count ceiling (consumers cannot exceed partition count)

For our pipeline: embed_workers should not exceed the embed semaphore limit,
and scaling steps should be bounded.

### 1.6 HikariCP Connection Pool

**Relevant for the fixed-vs-dynamic pool question.**

HikariCP's recommendation: **use fixed-size pools for performance-critical
systems.** The `connections = (core_count * 2) + spindle_count` formula is
a good starting heuristic, and the overhead of dynamic sizing (monitoring,
connection creation latency) rarely justifies the savings.

The spike demand handler is clever: start creating a connection, but cancel
the creation if an existing connection becomes available before the new one
is ready. This avoids over-provisioning during transient spikes.

For our pipeline: this argues for fixed worker counts with dynamic queue
sizes. Workers are cheap (they are asyncio tasks, not threads), but queue
sizes directly control memory pressure.

### 1.7 Spark Adaptive Query Execution

**Relevant for per-operation tuning.** Spark AQE makes decisions at runtime
between stages: after completing a shuffle, it observes actual partition
sizes and coalesces small partitions before the next stage begins. This is
"inspect-then-decide" rather than "continuously adjust."

For our pipeline: after the scan phase completes, we know the exact workload
(file count, file sizes, cache hit rate). This is the ideal time to set
initial parameters before the pipeline starts, rather than starting with
defaults and adjusting.

### 1.8 Envoy Adaptive Concurrency

Envoy's production implementation of the Netflix gradient algorithm confirms
the pattern works at scale (used at Lyft). Key addition: the `minRTT` is
periodically re-measured by temporarily reducing concurrency to near-zero and
observing latency. This "probing" phase ensures the baseline stays accurate
as system conditions change.

For our pipeline: we could periodically measure "unloaded" embed latency
by timing a single embed call with no contention, updating the baseline
for queue sizing calculations.

---

## 2. RAG-Specific Auto-Tuning

### 2.1 LlamaIndex

LlamaIndex's `IngestionPipeline` supports parallel execution via
`multiprocessing.Pool` with configurable `num_workers`. It includes
node-level caching (hash-based deduplication). **No adaptive batching
or backpressure mechanisms.** The parallelism is fixed at pipeline
creation time. Batch sizes are static.

### 2.2 Haystack

Haystack pipelines are directed multigraphs with `AsyncPipeline` for
concurrent execution. Supports branching and conditional routing.
**No backpressure handling or adaptive processing.** Pipeline execution
is fire-and-forget per component.

### 2.3 Qdrant

Qdrant's Python client `upload_points` internally batches large point
lists with fixed chunk sizes. Research shows batch size has a clear
optimization curve: performance peaks around batch size 64-128 and
degrades at larger sizes. Concurrent insertion shows diminishing returns
past 50-60 concurrent processes.

For our pipeline: the Qdrant upsert batch size (currently 100 in
`DocumentVectorRepository`) is in the right ballpark and should remain
static.

### 2.4 Embedding Infrastructure

Adaptive batching for embeddings adjusts batch size based on text length
to prevent OOM errors. Quantized models peak at batch size 128 for
256-token texts. vLLM supports continuous batching for embedding
serving. Production targets: query embedding 10-50ms, vector search
10-100ms.

### 2.5 Gap Analysis

**No RAG framework implements runtime adaptive pipeline tuning.** All use
static configurations set at initialization. This is an opportunity, but
also a signal: the complexity of adaptive tuning may not yet justify the
engineering cost for most RAG workloads. Our pipeline is unusual in that
it processes large corpora (6+ GB RSS observed) where static parameters
provably fail at scale.

---

## 3. Parameter Classification

### 3.1 Complete Parameter Table

| # | Parameter | Current Value | Category | Signal | Risk | Impact |
|---|-----------|--------------|----------|--------|------|--------|
| **Pipeline Queues** |
| 1 | `EMBED_QUEUE_SIZE` | 500 | **Auto-tune (runtime)** | Queue fill ratio, RSS | Low: queue.maxsize is mutable | **High** — 32% speedup when reduced to 200 on warm cache |
| 2 | `UPSERT_QUEUE_SIZE` | 500 | **Auto-tune (runtime)** | Queue fill ratio, RSS | Low: same mechanism | Medium — less critical than embed queue |
| **Worker Counts** |
| 3 | `NUM_CHUNK_WORKERS` | 16 | **Auto-tune (per-op)** | File count, file sizes | Medium: task creation/cancellation complexity | Medium — capped by disk I/O |
| 4 | `NUM_EMBED_WORKERS` | 64 | **Auto-tune (per-op)** | Cache hit rate, file count | Medium: 16 best warm, 64 best cold | **High** — wrong value causes either starvation or contention |
| 5 | `NUM_UPSERT_WORKERS` | 16 | **Static (good default)** | N/A | Low: 16 matches Qdrant's 8 concurrent upserts | Low — rarely bottleneck |
| **Batch Sizes** |
| 6 | `DENSE_SUB_BATCH_SIZE` | 100 | **Static (good default)** | N/A | Low: well-tuned for API | Low — only affects large files |
| 7 | `EmbedLoader.batch_size` | 100-1000 (config) | **Static (good default)** | N/A | Medium: provider API limits | Low — API-constrained |
| 8 | `EmbedLoader.coalesce_delay` | 10ms | **Auto-tune (runtime)** | Embed queue depth | Low: just a timer | Medium — too high wastes latency, too low wastes API calls |
| 9 | `CacheLoader.batch_size` | 100 | **Static (good default)** | N/A | Low: Redis handles well | Low — MGET is efficient at 100 |
| 10 | `CacheLoader.coalesce_delay` | 1ms | **Static (good default)** | N/A | Low: Redis is fast | Low — 1ms is near-optimal |
| 11 | Cache write `batch_size` | 200 | **Static (good default)** | N/A | Low: fire-and-forget | Low — writes are async |
| 12 | Cache write `coalesce_delay` | 50ms | **Static (good default)** | N/A | Low: writes are non-blocking | Low — latency-insensitive |
| 13 | Index write `batch_size` | 100 | **Static (good default)** | N/A | Low: fire-and-forget | Low |
| 14 | Index write `coalesce_delay` | 50ms | **Static (good default)** | N/A | Low | Low |
| 15 | Qdrant upsert `batch_size` | 100 | **Static (good default)** | N/A | Low: in optimal range | Low — matches research |
| **API Concurrency** |
| 16 | OpenRouter `max_concurrent` | 200 | **Static (dangerous)** | N/A | **High**: too many = 429s, too few = underutilization | Medium — rate limit is provider-side |
| 17 | Gemini `max_concurrent` | 200 | **Static (dangerous)** | N/A | **High**: same risk | Medium |
| 18 | Redis `max_concurrent` | 500 | **Static (good default)** | N/A | Low: Redis handles high concurrency | Low |
| 19 | Redis `max_concurrent_pipelines` | 500 | **Static (good default)** | N/A | Low | Low |
| 20 | Qdrant `max_concurrent_upserts` | 8 | **Static (good default)** | N/A | Medium: too high overwhelms gRPC | Low |
| **Timeouts** |
| 21 | `FILE_CHUNK_TIMEOUT_SECONDS` | 60 | **Static (good default)** | N/A | Low: safety net | Low |
| 22 | OpenRouter `timeout_ms` | 30000 | **Static (good default)** | N/A | Medium: affects error detection | Low |
| 23 | Qdrant `timeout` | 10 | **Static (good default)** | N/A | Medium: affects error detection | Low |

### 3.2 Priority Ranking

**High-impact auto-tunable (start here):**
1. `EMBED_QUEUE_SIZE` — proven 32% impact, safe to mutate at runtime
2. `NUM_EMBED_WORKERS` — 4x difference between warm/cold optimal
3. `UPSERT_QUEUE_SIZE` — secondary impact, same mechanism as #1

**Medium-impact, per-operation:**
4. `NUM_CHUNK_WORKERS` — set based on file count at scan time
5. `EmbedLoader.coalesce_delay` — adjust based on queue pressure

**Leave static:**
Everything else. The 18 remaining parameters are either well-tuned,
API-constrained, or dangerous to change at runtime.

---

## 4. Controller Design

### 4.1 Recommended Algorithm: Throughput-Targeting with Deadband

After evaluating all production patterns, the recommended approach combines:
- **Flink's throughput-targeting** for queue sizing (most directly applicable)
- **Netflix's smoothing** for stability (prevents oscillation)
- **K8s HPA's asymmetry** for safety (fast scale-up, slow scale-down)
- **Deadband from control theory** to prevent micro-adjustments

The algorithm is simpler than a PID controller (no integral or derivative
terms) because our system has discrete states and the tuning parameters
have natural bounds.

### 4.2 Core Algorithm

```python
@dataclass
class ControllerState:
    """Mutable state for the pipeline controller."""
    # Smoothed throughput estimates (exponential moving average)
    chunk_throughput: float = 0.0   # files/sec leaving chunk stage
    embed_throughput: float = 0.0   # files/sec leaving embed stage
    store_throughput: float = 0.0   # files/sec leaving store stage

    # Target buffer duration (seconds of work in each queue)
    target_buffer_seconds: float = 2.0

    # Smoothing factor for throughput EMA (higher = more responsive)
    alpha: float = 0.3

    # Deadband: ignore adjustments smaller than this fraction
    deadband: float = 0.25  # 25%, matching Flink's default

    # Asymmetric cooldowns (K8s HPA pattern)
    scale_up_cooldown: float = 10.0    # seconds
    scale_down_cooldown: float = 60.0  # seconds

    # Bounds
    min_queue_size: int = 50
    max_queue_size: int = 1000

    # Last adjustment timestamps
    last_embed_queue_adjust: float = 0.0
    last_upsert_queue_adjust: float = 0.0


class PipelineController:
    """Adjusts pipeline parameters based on observed metrics.

    Runs alongside the monitor loop (1Hz). Observes queue depths
    and completion rates, adjusts queue sizes to maintain target
    buffer duration.

    Design principles:
    - Observe throughput, not just queue depth (Flink pattern)
    - Smooth estimates to avoid reacting to transient spikes (Netflix)
    - Asymmetric response: fast scale-up, slow scale-down (K8s HPA)
    - Deadband to prevent oscillation (control theory)
    - Log every adjustment for observability
    """

    def __init__(
        self,
        embed_queue: asyncio.Queue,
        upsert_queue: asyncio.Queue,
        state: ControllerState | None = None,
    ):
        self._embed_queue = embed_queue
        self._upsert_queue = upsert_queue
        self._state = state or ControllerState()
        self._prev_embed_done: int = 0
        self._prev_store_done: int = 0
        self._prev_time: float = time.monotonic()
        self._adjustments: list[dict] = []  # audit log

    def on_sample(self, sample: QueueDepthSample) -> None:
        """Process a queue depth sample. Called at 1Hz by monitor loop.

        Algorithm:
        1. Compute instantaneous throughput from completion deltas
        2. Update smoothed throughput (EMA)
        3. Compute target queue size = throughput * target_buffer_seconds
        4. Apply deadband: skip if change < 25%
        5. Apply cooldown: skip if too soon since last adjustment
        6. Apply asymmetry: immediate scale-up, delayed scale-down
        7. Clamp to bounds and apply
        """
        now = time.monotonic()
        dt = now - self._prev_time
        if dt < 0.5:  # Need at least 0.5s of data
            return

        # 1. Instantaneous throughput
        embed_delta = sample.files_embed_done - self._prev_embed_done
        store_delta = sample.files_store_done - self._prev_store_done
        instant_embed_tput = embed_delta / dt
        instant_store_tput = store_delta / dt

        self._prev_embed_done = sample.files_embed_done
        self._prev_store_done = sample.files_store_done
        self._prev_time = now

        # 2. Smoothed throughput (EMA)
        s = self._state
        s.embed_throughput = (
            s.alpha * instant_embed_tput
            + (1 - s.alpha) * s.embed_throughput
        )
        s.store_throughput = (
            s.alpha * instant_store_tput
            + (1 - s.alpha) * s.store_throughput
        )

        # 3. Target queue sizes
        target_embed = int(s.embed_throughput * s.target_buffer_seconds)
        target_upsert = int(s.store_throughput * s.target_buffer_seconds)

        # Adjust embed queue
        self._maybe_adjust(
            queue=self._embed_queue,
            name="embed_queue",
            target=target_embed,
            now=now,
            last_adjust_attr="last_embed_queue_adjust",
            health_signal=sample.rss_memory_mb,
        )

        # Adjust upsert queue
        self._maybe_adjust(
            queue=self._upsert_queue,
            name="upsert_queue",
            target=target_upsert,
            now=now,
            last_adjust_attr="last_upsert_queue_adjust",
            health_signal=sample.rss_memory_mb,
        )

    def _maybe_adjust(
        self,
        queue: asyncio.Queue,
        name: str,
        target: int,
        now: float,
        last_adjust_attr: str,
        health_signal: float,
    ) -> None:
        s = self._state
        current = queue.maxsize
        target = max(s.min_queue_size, min(s.max_queue_size, target))

        # 4. Deadband: skip if change is less than 25%
        if current > 0:
            change_ratio = abs(target - current) / current
            if change_ratio < s.deadband:
                return

        # 5. Cooldown check
        last_adjust = getattr(s, last_adjust_attr)
        is_scale_up = target > current
        cooldown = (
            s.scale_up_cooldown if is_scale_up
            else s.scale_down_cooldown
        )
        if now - last_adjust < cooldown:
            return

        # 6. Emergency override: if RSS > threshold, force reduce
        if health_signal > 4000 and target > current:
            # Over 4GB RSS: do not increase queue sizes
            return
        if health_signal > 5000:
            # Over 5GB: force minimum queue sizes
            target = s.min_queue_size

        # 7. Apply
        queue.maxsize = target
        setattr(s, last_adjust_attr, now)

        adjustment = {
            "time": now,
            "queue": name,
            "old": current,
            "new": target,
            "reason": "scale_up" if is_scale_up else "scale_down",
            "embed_throughput": round(s.embed_throughput, 1),
            "store_throughput": round(s.store_throughput, 1),
            "rss_mb": health_signal,
        }
        self._adjustments.append(adjustment)
        logger.info(
            f'[CONTROLLER] {name}: {current} -> {target} '
            f'({adjustment["reason"]}, '
            f'throughput={s.embed_throughput:.1f} files/s, '
            f'RSS={health_signal:.0f}MB)'
        )
```

### 4.3 Per-Operation Initialization (Spark AQE Pattern)

Set worker counts once after scan phase, based on workload characteristics:

```python
def compute_initial_params(
    file_count: int,
    cache_hit_rate: float,
    total_bytes: int,
) -> PipelineParams:
    """Compute pipeline parameters from scan-phase data.

    Called once after scan completes, before pipeline starts.
    Uses Spark AQE pattern: observe actual data, then decide.
    """
    files_to_embed = int(file_count * (1 - cache_hit_rate))

    # Embed workers: scale with work volume
    # Warm cache (>80% hits): 16 workers suffice (less API contention)
    # Cold run (<20% hits): 64 workers to saturate API
    # Mixed: interpolate
    if cache_hit_rate > 0.8:
        embed_workers = 16
    elif cache_hit_rate < 0.2:
        embed_workers = 64
    else:
        # Linear interpolation: 64 at 0% hits, 16 at 100% hits
        embed_workers = int(64 - (cache_hit_rate * 48))

    # Chunk workers: scale with file count, cap at CPU count
    chunk_workers = min(16, max(4, file_count // 100))

    # Initial queue sizes: conservative, controller will adjust
    # Start smaller than max to leave room for growth
    embed_queue_size = min(200, files_to_embed)
    upsert_queue_size = min(200, files_to_embed)

    return PipelineParams(
        chunk_workers=chunk_workers,
        embed_workers=embed_workers,
        upsert_workers=16,  # Always 16 (static)
        embed_queue_size=max(50, embed_queue_size),
        upsert_queue_size=max(50, upsert_queue_size),
    )
```

### 4.4 Signal Selection

| Signal | Source | Use | Sampling Rate |
|--------|--------|-----|---------------|
| Queue fill ratio | `queue.qsize() / queue.maxsize` | Primary: queue size adjustment | 1Hz (existing) |
| Stage throughput | Completion count deltas | Primary: target calculation | 1Hz (derived) |
| RSS memory | `psutil.Process().memory_info().rss` | Safety: emergency reduction | 1Hz (existing) |
| Event loop lag | Sleep overshoot measurement | Safety: detect overload | 1Hz (existing) |
| Cache hit rate | `embedding_service.hits / (hits + misses)` | Init: worker count selection | Once (scan phase) |
| Embed API latency | `openrouter._http_latencies` | Future: API health signal | Available but unused |
| 429 error count | `openrouter.errors_429` | Future: concurrency reduction | Available but unused |

### 4.5 Adjustment Interval

**1Hz sampling, 10-60s adjustment cycle.** The monitor loop already samples
at 1Hz. The controller evaluates every sample but the cooldown mechanism
ensures actual adjustments happen at most every 10s (scale-up) or 60s
(scale-down).

Why not faster? Throughput EMA needs 5-10 samples to stabilize after a
workload change. Adjusting faster than this reacts to noise.

Why not slower? The pipeline can process 50+ files/sec. A 60s scale-up
delay means 3000 files processed with suboptimal parameters. 10s is a
reasonable compromise (500 files).

### 4.6 Preventing Oscillation

Five mechanisms work together:

1. **Deadband (25%):** Small deviations are ignored entirely. Prevents
   the controller from chasing noise.

2. **Asymmetric cooldowns:** Scale-up: 10s. Scale-down: 60s. This creates
   a natural ratchet that prefers larger queues, which is safe (costs only
   memory) while under-provisioned queues cause stalls.

3. **EMA smoothing (alpha=0.3):** Each throughput sample is blended with
   history. A sudden spike affects only 30% of the estimate, preventing
   overreaction to transient bursts.

4. **Clamped bounds (50-1000):** Queue sizes cannot go below 50 (starvation
   risk) or above 1000 (memory risk). This bounds the system's behavior
   even if the algorithm malfunctions.

5. **Emergency override (RSS-based):** If memory exceeds 4GB, no scale-up.
   If memory exceeds 5GB, force minimum queue sizes. This circuit-breaker
   prevents the tuning system from contributing to memory exhaustion.

### 4.7 Independent vs. Joint Adjustment

**Recommend independent adjustment.** Each queue connects two specific
stages with different characteristics (CPU-bound chunking vs I/O-bound
embedding vs I/O-bound storage). Joint optimization would require modeling
cross-stage interactions, which adds complexity without clear benefit.

The queues are naturally decoupled by their positions in the pipeline.
Flink's experience confirms this: they moved from subtask-level (joint)
to gate-level (independent) buffer management because heterogeneous
throughputs made joint optimization counterproductive.

### 4.8 Cold Start

**Use the per-operation initialization (Section 4.3) for cold start.**
The controller starts with parameters computed from scan-phase data rather
than arbitrary defaults. The EMA throughput estimates initialize to zero
and converge within 10-15 samples (10-15 seconds of pipeline operation).

During the convergence window, the controller's deadband prevents premature
adjustments: since smoothed throughput starts at zero, the target queue size
starts at zero, but the deadband prevents reducing below the initial value
until enough data accumulates.

---

## 5. Empirical Test Plan

### 5.1 Test Matrix

| Dimension | Values | Rationale |
|-----------|--------|-----------|
| **Workload** | warm (>90% cache), cold (<10%), mixed (50%), large-files (PDFs), many-small (thousands of markdown) | Cover all workload archetypes |
| **Corpus size** | 100 files, 1K files, 10K files | Scale sensitivity |
| **Controller mode** | off (static defaults), init-only (per-op, no runtime), full (init + runtime) | Isolate value of each mechanism |
| **Queue size preset** | 500 (current), 200 (proven good for warm), controller-managed | Baseline comparison |

**Total configurations:** 5 workloads x 3 sizes x 3 modes = 45 configs.
At 3 runs each for variance: 135 total runs.

### 5.2 Metrics to Collect

| Metric | Unit | Source | Why |
|--------|------|--------|-----|
| Wall-clock time | seconds | `Timer` | Primary: overall throughput |
| Peak RSS | MB | `psutil` (existing) | Safety: memory footprint |
| Event loop lag HWM | ms | Tracer (existing) | Health: responsiveness |
| Embed API calls | count | `openrouter._http_latencies` | Efficiency: batching quality |
| 429 errors | count | `openrouter.errors_429` | Safety: API politeness |
| Embed queue fill ratio | 0.0-1.0 time series | QueueDepthSample (existing) | Utilization: backpressure health |
| Controller adjustments | count + log | Controller audit log | Stability: is it oscillating? |
| GC gen2 collections | count | Tracer (existing) | Efficiency: memory pressure |
| Files/sec throughput | files/sec time series | Completion deltas | Performance: stage balance |

### 5.3 Statistical Rigor

- **3 runs per configuration** for variance estimation
- Report **median** wall-clock time (robust to outliers)
- Report **95th percentile** for latency-sensitive metrics
- Use **coefficient of variation** (CV = stddev/mean) to flag unstable configs
- Consider a config "better" only if median improvement > 5% AND p-value < 0.05
  (paired t-test vs baseline)
- Exclude first run from each config to account for JIT/cache warming effects

### 5.4 Specific Experiments

**Experiment 1: Queue size sweep (validate controller targets)**
- Fix workers at defaults, sweep queue sizes: 50, 100, 200, 500, 1000
- For each workload type
- Establishes ground truth for optimal queue size per workload

**Experiment 2: Controller convergence**
- Start with queue_size=500 (current default)
- Run controller on warm, cold, and mixed workloads
- Verify: controller converges to values near Experiment 1 optima
- Measure: convergence time (samples until within 10% of final value)

**Experiment 3: Workload transition**
- Start with warm cache workload (controller tunes for it)
- Switch to cold workload mid-run (invalidate cache, add new files)
- Verify: controller adapts within 30 seconds
- Compare: static config vs adaptive

**Experiment 4: Memory pressure**
- 10K file cold run (known to cause 6+ GB RSS)
- Verify: emergency override triggers before OOM
- Compare: with vs without controller

**Experiment 5: Per-operation worker selection**
- Cold run: compare 16 vs 32 vs 64 embed workers
- Warm run: compare 16 vs 32 vs 64 embed workers
- Verify: init function selects near-optimal worker count

---

## 6. Anti-Patterns and Risks

### 6.1 Oscillation

**Problem:** The controller adjusts queue size down because throughput is low,
but low queue size starves workers, reducing throughput further, triggering
more reductions. Or: increase queue -> more memory -> GC pressure ->
slower processing -> queue fills -> increase queue.

**Mitigation:**
- Deadband (25%) prevents reacting to small changes
- Asymmetric cooldowns (10s up, 60s down) prevent rapid cycling
- EMA smoothing (alpha=0.3) dampens transient signals
- Minimum bound (50) prevents starvation

**Detection:** If the controller log shows >5 adjustments in the same
direction within 5 minutes, something is wrong. Add a circuit breaker
that freezes the controller after 5 consecutive same-direction changes.

### 6.2 Cascading Failures

**Problem:** Reducing embed queue causes chunk workers to block on
`queue.put()`, which holds file handles and memory. If chunk workers
stall for too long, file_chunk_timeout triggers, causing errors that
reduce throughput, which causes the controller to reduce queue size
further.

**Mitigation:**
- `min_queue_size=50` ensures chunk workers can always make progress
- File chunk timeout (60s) is a safety net, not a normal path
- Controller monitors error rate (future: pause adjustments if error
  rate spikes)

### 6.3 Feedback Loops

**Problem:** Controller reduces queue size -> embed workers see empty
queue -> throughput drops (workers are idle) -> controller sees low
throughput -> computes even smaller target -> positive feedback loop
toward minimum.

**Mitigation:** The throughput-targeting formula naturally handles this.
When the queue is small and throughput drops, `target = throughput * 2s`
produces a small target that matches the small queue. The deadband
prevents further reduction unless the change exceeds 25%. The minimum
bound (50) stops the descent.

**Alternative consideration:** Use queue fill ratio instead of throughput
for sizing. `target = current * (1 + alpha * (fill_ratio - 0.5))` grows
the queue when fill > 50% and shrinks when fill < 50%. This avoids the
throughput-drop feedback loop but does not account for actual processing
speed. Hybrid approach: use throughput-targeting as primary, with a
fill-ratio safety check.

### 6.4 Over-Fitting to Transient Conditions

**Problem:** A burst of small files produces high throughput temporarily.
Controller scales up. Then large files arrive and throughput drops. The
over-provisioned queue now holds too many large files, consuming memory.

**Mitigation:**
- EMA smoothing prevents reacting to short bursts (need 10+ samples)
- RSS-based emergency override catches memory issues regardless of cause
- Per-operation initialization sets reasonable starting points based on
  workload characteristics (file sizes are known at scan time)

### 6.5 Configuration Complexity (Tuning the Tuner)

**Problem:** The controller itself has parameters: `alpha`, `deadband`,
`target_buffer_seconds`, cooldowns, bounds. If these need tuning per
workload, we have not eliminated complexity, only moved it.

**Mitigation:**
- Use well-researched defaults from production systems:
  - alpha=0.3 (Netflix Gradient2 uses 0.2, we use slightly more responsive)
  - deadband=25% (Flink default)
  - cooldowns: 10s/60s (K8s HPA pattern)
- Make the controller conservative: it is better to leave a parameter
  unchanged than to make a wrong adjustment. The deadband and cooldowns
  enforce this by default.
- Document all parameters with their provenance ("deadband=25% from Flink
  buffer debloating default") so future developers understand why.

### 6.6 Integral Windup (PID-Specific Risk)

**Not applicable** to our design because we use a proportional controller
(target = throughput * buffer_time), not a PID controller. There is no
integral term that could accumulate error. The EMA naturally decays old
information without needing anti-windup.

### 6.7 Race Conditions

**Problem:** `queue.maxsize` is mutable at runtime, but CPython's GIL
protects single-attribute writes. However, reducing maxsize below current
qsize() does not evict items. The queue will be temporarily over-capacity
until items are consumed.

**Mitigation:** This is a feature, not a bug. Over-capacity items drain
naturally as workers consume them. The reduced maxsize prevents new items
from being added until the queue drains below the new limit. No items are
lost, and the system converges to the new target within one queue drain
cycle.

---

## Summary: Recommended Implementation Path

### Phase 1: Per-Operation Initialization (Low Risk, Proven Value)

Implement `compute_initial_params()` to set worker counts and initial
queue sizes based on scan-phase data (cache hit rate, file count, sizes).
This is the Spark AQE pattern and is completely safe: no runtime mutation,
just smarter defaults.

**Effort:** Small. One function, called after scan phase.
**Expected impact:** Correct worker count alone was measured at 4x difference.

### Phase 2: Runtime Queue Adjustment (Medium Risk, High Value)

Implement `PipelineController` with throughput-targeting and deadband.
Start with only queue size adjustment (not worker counts). The existing
QueueDepthSample data stream provides all signals needed.

**Effort:** Medium. New class, integration with monitor loop.
**Expected impact:** 20-30% improvement on mixed workloads where static
queue sizes are wrong for part of the run.

### Phase 3: Worker Scaling (Higher Risk, Incremental Value)

Add dynamic embed worker spawning/cancellation based on queue fill ratio.
This is more complex because asyncio task lifecycle management is tricky.

**Effort:** High. Task cancellation, graceful shutdown, semaphore coordination.
**Expected impact:** Incremental over Phase 2. Workers are cheap; queue
sizes are the primary lever.

### Phase 4: Embed Coalesce Delay Tuning (Low Risk, Small Value)

Adjust `EmbedLoader.coalesce_delay` based on embed queue depth. When queue
is deep (many items waiting), reduce delay to flush batches faster. When
queue is shallow, increase delay to coalesce more items per API call.

**Effort:** Small. Timer adjustment in BatchLoader.
**Expected impact:** 5-10% improvement in API efficiency.

---

## Sources

### Production Systems
- [Netflix Performance Under Load (Gradient algorithm)](https://netflixtechblog.medium.com/performance-under-load-3e6fa9a60581)
- [Netflix concurrency-limits GitHub](https://github.com/Netflix/concurrency-limits)
- [Netflix Gradient2Limit.java](https://github.com/Netflix/concurrency-limits/blob/main/concurrency-limits-core/src/main/java/com/netflix/concurrency/limits/limit/Gradient2Limit.java)
- [Envoy Adaptive Concurrency Filter](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/adaptive_concurrency_filter)
- [Envoy Adaptive Concurrency Analysis (Alibaba)](https://www.alibabacloud.com/blog/brief-analysis-of-envoy-adaptive-concurrency-filter_600658)
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)
- [K8s HPA Configurable Scaling Velocity KEP](https://github.com/kubernetes/enhancements/blob/master/keps/sig-autoscaling/853-configurable-hpa-scale-velocity/README.md)
- [Flink Network Buffer Tuning](https://nightlies.apache.org/flink/flink-docs-release-1.16/docs/deployment/memory/network_mem_tuning/)
- [Flink Buffer Debloating JIRA](https://issues.apache.org/jira/browse/FLINK-24189)
- [Spark AQE (Databricks)](https://www.databricks.com/blog/2020/05/29/adaptive-query-execution-speeding-up-spark-sql-at-runtime.html)
- [HikariCP Pool Sizing](https://github.com/brettwooldridge/HikariCP/wiki/About-Pool-Sizing)
- [KEDA Kafka Consumer Autoscaling](https://dev.to/azure/how-to-auto-scale-kafka-applications-on-kubernetes-with-keda-1k9n)
- [Java ForkJoinPool Documentation](https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/ForkJoinPool.html)

### RAG Frameworks
- [LlamaIndex Ingestion Pipeline](https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/)
- [LlamaIndex Parallel Ingestion](https://developers.llamaindex.ai/python/examples/ingestion/parallel_execution_ingestion_pipeline/)
- [Haystack Pipelines](https://docs.haystack.deepset.ai/docs/pipelines)
- [Qdrant Vector Search in Production](https://qdrant.tech/articles/vector-search-production/)
- [Embedding Infrastructure at Scale (Introl)](https://introl.com/blog/embedding-infrastructure-scale-vector-generation-production-guide-2025)

### Control Theory
- [AIMD Algorithm (GeeksforGeeks)](https://www.geeksforgeeks.org/aimd-algorithm/)
- [TCP Congestion Control (Wikipedia)](https://en.wikipedia.org/wiki/TCP_congestion_control)
- [Deadband (Wikipedia)](https://en.wikipedia.org/wiki/Deadband)
- [PID Anti-Windup (Scilab)](https://www.scilab.org/pid-anti-windup-schemes)
- [Adaptive AIMD (ACM)](https://dl.acm.org/doi/10.1145/872035.872089)

### Netflix concurrency-limits Architecture
- [Netflix concurrency-limits DeepWiki](https://deepwiki.com/Netflix/concurrency-limits)
- [AIMD and Other Algorithms DeepWiki](https://deepwiki.com/Netflix/concurrency-limits/3.3-aimd-and-other-algorithms)
- [Adaptive Concurrency Limits (Michal Drozd)](https://www.michal-drozd.com/en/blog/adaptive-concurrency-limits/)
