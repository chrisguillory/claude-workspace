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
        self._pending: dict[TRequest, list[asyncio.Future[TResponse]]] = collections.defaultdict(list)
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
            self._pending[request].append(future)
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

    async def _process_pending(self) -> None:
        """Process all pending requests in batches."""
        # Wait for more requests to coalesce
        await asyncio.sleep(self.coalesce_delay)

        # Grab all pending requests
        async with self._lock:
            requests_to_load = self._pending
            self._pending = collections.defaultdict(list)

        if self.sort_requests:
            requests_to_load = dict(sorted(requests_to_load.items(), key=lambda x: str(x[0])))

        if not requests_to_load:
            return

        # Split into batches and process
        chunks = list(more_itertools.chunked(requests_to_load.keys(), n=self.batch_size))
        tasks: list[asyncio.Task[Sequence[TResponse]]] = [
            asyncio.create_task(self.bulk_load(list(chunk))) for chunk in chunks
        ]

        try:
            for chunk, task in zip(chunks, tasks, strict=True):
                chunk_responses = await task
                for request, response in zip(chunk, chunk_responses, strict=True):
                    for future in requests_to_load[request]:
                        if not future.done():
                            future.set_result(response)

        except Exception as e:
            # Propagate exception to all waiting callers
            for futures in requests_to_load.values():
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

            # Schedule next batch if more requests arrived
            async with self._lock:
                if self._pending:
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
            if self._pending:
                self._load_task = asyncio.create_task(self._process_pending())
            else:
                self._load_task = None
