from __future__ import annotations

__all__ = [
    'GenericBatchLoader',
]

import asyncio
import collections
import logging
from collections.abc import Callable, Coroutine, Hashable, Sequence, Set
from typing import Any, cast

import more_itertools

logger = logging.getLogger(__name__)


class GenericBatchLoader[TRequest: Hashable, TResponse]:
    """Aggregates individual requests into bulk requests.

    Concurrent callers of load() have their requests coalesced into batches,
    reducing API calls while maintaining per-request semantics.

    Two usage modes:
    - Request-response: load() / load_many() — caller awaits, self-draining.
    - Fire-and-forget: submit() / submit_many() — caller returns immediately.
      Requires drain() at operation end. drain() closes the submit path.

    All modes share the same coalescing, deduplication, and bulk_load pipeline.

    Type Parameters:
        TRequest: Request type (must be hashable for deduplication).
        TResponse: Response type.
    """

    def __init__(
        self,
        *,
        bulk_load: Callable[[Sequence[TRequest]], Coroutine[Any, Any, Sequence[TResponse]]],
        batch_size: int = 100,
        coalesce_delay: float = 0.005,
        sort_requests: bool = False,
    ) -> None:
        """Initialize batch loader.

        Args:
            bulk_load: Async function that processes a batch of requests.
                       Must return responses in same order as requests.
            batch_size: Maximum requests per batch.
            coalesce_delay: Seconds to wait for more requests before processing.
            sort_requests: If True, sort requests for deterministic ordering.
        """
        self.bulk_load = bulk_load
        self.batch_size = batch_size
        self.coalesce_delay = coalesce_delay
        self.sort_requests = sort_requests

        self._lock = asyncio.Lock()
        # Three pools: load() → Futures, load_many() → BatchRecords, submit() → set
        self._individual: dict[TRequest, list[asyncio.Future[TResponse]]] = collections.defaultdict(list)
        self._batches: list[_BatchRecord[TRequest, TResponse]] = []
        self._submitted: set[TRequest] = set()
        self._load_task: asyncio.Task[None] | None = None

        # Fire-and-forget lifecycle
        self._submit_closed = False
        self._first_error: BaseException | None = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bulk_load={self.bulk_load.__name__}, batch_size={self.batch_size})'

    async def load(self, request: TRequest) -> TResponse:
        """Load a single request, batched with concurrent requests.

        Args:
            request: The request to process.

        Returns:
            Response for this request.

        Raises:
            Exception: If bulk_load fails, the exception is propagated.
        """
        future: asyncio.Future[TResponse] = asyncio.Future()

        async with self._lock:
            self._individual[request].append(future)
            if not self._load_task:
                self._load_task = asyncio.create_task(self._process_pending())

        # Wait for our future to be resolved
        while not future.done():
            async with self._lock:
                load_task = self._load_task
            if load_task is None:
                raise RuntimeError('BatchLoader: no batch running but future not resolved')
            await load_task

        return future.result()

    async def load_many(self, requests: Sequence[TRequest]) -> Sequence[TResponse]:
        """Load multiple requests as a single batch-aware call.

        Creates one _BatchRecord per call instead of N Futures. All delivery
        happens via position_map writes into a shared results array. Single
        Future fires when all unique requests resolve — 1 event loop callback
        instead of N.

        Args:
            requests: Sequence of requests to process.

        Returns:
            Responses in same order as requests.
        """
        if not requests:
            return []

        batch = _BatchRecord[TRequest, TResponse].from_requests(requests)

        async with self._lock:
            self._batches.append(batch)
            if not self._load_task:
                self._load_task = asyncio.create_task(self._process_pending())

        await batch.future
        # Batch.results is list[TResponse | None] but all None slots are filled
        # after batch.future resolves — cast narrows to caller's expected type.
        return cast(list[TResponse], batch.results)

    # ── Fire-and-forget: submit / drain ──────────────────────────

    async def submit(self, request: TRequest) -> None:
        """Submit a request without waiting for its result.

        The request is coalesced into the next bulk_load batch.
        Errors are captured and raised on drain() or check_health().

        Raises:
            RuntimeError: If submit path is closed (drain() was called).
        """
        async with self._lock:
            if self._submit_closed:
                raise RuntimeError(f'{self!r}: submit() after drain()')
            self._submitted.add(request)
            if not self._load_task:
                self._load_task = asyncio.create_task(self._process_pending())

    async def submit_many(self, requests: Sequence[TRequest]) -> None:
        """Submit multiple requests without waiting for results.

        All requests are enqueued in one lock acquisition.

        Raises:
            RuntimeError: If submit path is closed (drain() was called).
        """
        async with self._lock:
            if self._submit_closed:
                raise RuntimeError(f'{self!r}: submit_many() after drain()')
            if not requests:
                return
            self._submitted.update(requests)
            if not self._load_task:
                self._load_task = asyncio.create_task(self._process_pending())

    async def drain(self) -> None:
        """Flush all submitted items and close the submit path.

        Awaits completion of all in-flight processing cycles.
        After drain returns, submit() and submit_many() raise RuntimeError.
        load() and load_many() are unaffected (they are self-draining).
        """
        self._submit_closed = True
        while True:
            async with self._lock:
                load_task = self._load_task
                has_pending = bool(self._submitted)
            if load_task is not None:
                await load_task
            elif has_pending:
                await asyncio.sleep(0)
            else:
                break
        if self._first_error is not None:
            raise self._first_error

    def check_health(self) -> None:
        """Raise first captured error from fire-and-forget operations.

        O(1) field read. Call in hot loops for early failure detection.
        """
        if self._first_error is not None:
            raise self._first_error

    def cancel_all(self) -> None:
        """Cancel in-flight processing, discard pending submits, close submit path.

        Note: Only safe when no load() callers are awaiting Futures. CancelledError
        bypasses the except Exception handler in _process_pending, leaving Futures
        unresolved. This is fine for submit-only batchers (the current usage).
        """
        self._submit_closed = True
        if self._load_task is not None:
            self._load_task.cancel()
        self._submitted.clear()

    # ── Internal processing ──────────────────────────────────────

    async def _process_pending(self) -> None:
        """Process all pending requests in batches.

        Collects unique requests across all three pools, chunks them,
        calls bulk_load, and delivers results to Futures (load),
        BatchRecords (load_many). Submitted items are processed but
        results are discarded.
        """
        # Wait for more requests to coalesce
        await asyncio.sleep(self.coalesce_delay)

        # Swap out all three pools atomically
        async with self._lock:
            individual = self._individual
            self._individual = collections.defaultdict(list)
            batches = self._batches
            self._batches = []
            submitted = self._submitted
            self._submitted = set()

        # Collect unique requests across all pools
        all_unique: dict[TRequest, None] = dict.fromkeys(individual.keys())
        for batch in batches:
            all_unique.update(dict.fromkeys(batch.pending_requests))
        all_unique.update(dict.fromkeys(submitted))

        if self.sort_requests:
            all_unique = dict(sorted(all_unique.items(), key=lambda x: str(x[0])))

        if not all_unique:
            async with self._lock:
                self._load_task = None
            return

        # Split into batch_size chunks and process
        chunks = list(more_itertools.chunked(all_unique.keys(), n=self.batch_size))
        tasks: list[asyncio.Task[Sequence[TResponse]]] = [
            asyncio.create_task(self.bulk_load(list(chunk))) for chunk in chunks
        ]

        try:
            for chunk, task in zip(chunks, tasks, strict=True):
                chunk_responses = await task

                # Deliver to both pools
                for request, response in zip(chunk, chunk_responses, strict=True):
                    # Individual callers (Futures)
                    for future in individual.get(request, ()):
                        if not future.done():
                            future.set_result(response)

                    # Batch callers (BatchRecords)
                    # NOTE: O(B) per request where B = concurrent batches
                    # For large B, consider reverse index optimization (see docs)
                    for batch in batches:
                        batch.deliver(request, response)

        except Exception as e:  # exception_safety_linter.py: swallowed-exception — error delivered to callers via futures, not re-raised in background task
            # Propagate exception to all waiting callers
            for futures in individual.values():
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

            for batch in batches:
                batch.deliver_error(e)

            # Capture for fire-and-forget callers
            if submitted and self._first_error is None:
                self._first_error = e
                logger.error(  # exception_safety_linter.py: logger-no-exc-info — traceback delivered via future
                    '%r: bulk_load failed: %s',
                    self,
                    e,
                )

            # Schedule next batch if more requests arrived
            async with self._lock:
                if self._individual or self._batches or self._submitted:
                    self._load_task = asyncio.create_task(self._process_pending())
                else:
                    self._load_task = None
            return

        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

        # Schedule next batch if more requests arrived during processing
        async with self._lock:
            if self._individual or self._batches or self._submitted:
                self._load_task = asyncio.create_task(self._process_pending())
            else:
                self._load_task = None


class _BatchRecord[TRequest: Hashable, TResponse]:
    """Batch-level tracking for load_many() calls.

    Owns delivery logic: position_map tracks where each request's result goes,
    remaining counter triggers the batch Future when all unique requests resolve.
    One object per load_many() call instead of N per-item objects.
    """

    __slots__ = ('_position_map', '_remaining', 'future', 'results')

    def __init__(
        self,
        results: list[TResponse | None],  # strict_typing_linter.py: mutable-type — deliver() mutates via __setitem__
        future: asyncio.Future[None],
        position_map: dict[
            TRequest,
            tuple[int, ...],
        ],  # strict_typing_linter.py: mutable-type — deliver() mutates via .pop()
        remaining: int,
    ) -> None:
        self.results = results
        self.future = future
        self._position_map = position_map
        self._remaining = remaining

    @classmethod
    def from_requests(cls, requests: Sequence[TRequest]) -> _BatchRecord[TRequest, TResponse]:
        """Build batch record from input requests with deduplication tracking."""
        position_map: dict[TRequest, list[int]] = collections.defaultdict(list)
        for i, req in enumerate(requests):
            position_map[req].append(i)

        return cls(
            results=[None] * len(requests),
            future=asyncio.Future(),
            position_map={k: tuple(v) for k, v in position_map.items()},
            remaining=len(position_map),  # count of UNIQUE requests
        )

    def deliver(self, request: TRequest, response: TResponse) -> None:
        """Write result to all positions for this request, trigger Future when done."""
        positions = self._position_map.pop(request, None)
        if positions is None:
            return  # Request not in this batch
        for pos in positions:
            self.results[pos] = response
        self._remaining -= 1
        if self._remaining == 0 and not self.future.done():
            self.future.set_result(None)

    def deliver_error(self, exc: BaseException) -> None:
        """Fail the entire batch with one exception."""
        if not self.future.done():
            self.future.set_exception(exc)

    @property
    def pending_requests(self) -> Set[TRequest]:
        """Requests still waiting for delivery."""
        return set(self._position_map.keys())
