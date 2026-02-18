"""Generic batch loader for coalescing individual requests into bulk operations.

Aggregates concurrent individual requests into efficient batch API calls.
Services create typed wrappers with domain-specific defaults.

Pattern from mainstay-io/monorepo.
"""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Callable, Coroutine, Hashable, Sequence
from typing import Any

import more_itertools


class GenericBatchLoader[TRequest: Hashable, TResponse]:
    """Aggregates individual requests into bulk requests.

    Concurrent callers of load() have their requests coalesced into batches,
    reducing API calls while maintaining per-request semantics.

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
        # Split pools: load() → Futures, load_many() → BatchRecords
        self._individual: dict[TRequest, list[asyncio.Future[TResponse]]] = collections.defaultdict(list)
        self._batches: list[_BatchRecord[TRequest, TResponse]] = []
        self._load_task: asyncio.Task[None] | None = None

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
        return batch.results  # type: ignore[return-value]

    async def _process_pending(self) -> None:
        """Process all pending requests in batches.

        Unified processing for both load() and load_many() pools.
        Collects unique requests across both, chunks, calls bulk_load,
        and delivers to Futures (load) and BatchRecords (load_many).
        """
        # Wait for more requests to coalesce
        await asyncio.sleep(self.coalesce_delay)

        # Swap out both pools atomically
        async with self._lock:
            individual = self._individual
            self._individual = collections.defaultdict(list)
            batches = self._batches
            self._batches = []

        # Collect unique requests across both pools
        all_unique: dict[TRequest, None] = dict.fromkeys(individual.keys())
        for batch in batches:
            all_unique.update(dict.fromkeys(batch.pending_requests))

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

        except Exception as e:
            # Propagate exception to all waiting callers
            for futures in individual.values():
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

            for batch in batches:
                batch.deliver_error(e)

            # Schedule next batch if more requests arrived
            async with self._lock:
                if self._individual or self._batches:
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
            if self._individual or self._batches:
                self._load_task = asyncio.create_task(self._process_pending())
            else:
                self._load_task = None


class _BatchRecord[TRequest: Hashable, TResponse]:
    """Batch-level tracking for load_many() calls.

    Owns delivery logic: position_map tracks where each request's result goes,
    remaining counter triggers the batch Future when all unique requests resolve.
    One object per load_many() call instead of N per-item objects.
    """

    __slots__ = ('results', 'future', '_position_map', '_remaining')

    def __init__(
        self,
        results: list[TResponse | None],
        future: asyncio.Future[None],
        position_map: dict[TRequest, tuple[int, ...]],
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
    def pending_requests(self) -> set[TRequest]:
        """Requests still waiting for delivery."""
        return set(self._position_map.keys())
