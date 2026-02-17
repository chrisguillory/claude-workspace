"""Indexing service - orchestrates document indexing pipeline.

Coordinates: scan → chunk → embed → store with batched processing.
Supports incremental indexing via file content hashing.

Architecture:
- Files processed in batches for efficiency (fewer API calls)
- Deletion happens AFTER successful upsert (atomicity)
- State updated per-batch for crash recovery
- Embedding count validated against input
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import filelock
import git
from local_lib.background_tasks import BackgroundTaskGroup
from local_lib.utils import Timer

from document_search.clients import QdrantClient, create_embedding_client
from document_search.clients.redis import RedisClient
from document_search.repositories.document_vector import DocumentVectorRepository
from document_search.schemas.chunking import EXTENSION_MAP, Chunk, FileType, get_file_type
from document_search.schemas.config import EmbeddingConfig
from document_search.schemas.embeddings import EmbedResponse
from document_search.schemas.indexing import (
    CHUNK_STRATEGY_VERSION,
    DirectoryIndexState,
    FileIndexState,
    FileProcessingError,
    FileTypeStats,
    IndexingResult,
    ProgressCallback,
    StopAfterStage,
)
from document_search.schemas.tracing import QueueDepthSample
from document_search.schemas.vectors import ClearResult, VectorPoint
from document_search.services.chunking import ChunkingService
from document_search.services.embedding import EmbeddingService
from document_search.services.sparse_embedding import SparseEmbeddingService
from document_search.services.tracing import PipelineTracer

__all__ = [
    'IndexingService',
    'PipelineSnapshot',
    'create_indexing_service',
]

logger = logging.getLogger(__name__)

# Dense embedding dimension
EMBEDDING_DIMENSION = 768

# Default state file location
DEFAULT_STATE_PATH = Path.home() / '.claude-workspace' / 'cache' / 'document_search_index_state.json'

# Pipeline worker configuration - each stage is independently tunable
NUM_CHUNK_WORKERS = 16  # CPU-bound, limited by disk I/O
NUM_EMBED_WORKERS = 64  # I/O-bound, feeds embedding API
NUM_UPSERT_WORKERS = 16  # I/O-bound, feeds Qdrant API

# Queue size limits for backpressure between stages
EMBED_QUEUE_SIZE = 500
UPSERT_QUEUE_SIZE = 500

# Timeout for file chunking operations (detects deadlocks)
FILE_CHUNK_TIMEOUT_SECONDS = 60


class IndexingService:
    """Orchestrates document indexing pipeline.

    Coordinates chunking, embedding, and storage with:
    - Incremental indexing via content hash comparison
    - Rate limiting with exponential backoff
    - Progress reporting via callbacks
    - Error collection and categorization
    """

    @classmethod
    async def create(
        cls,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
        repository: DocumentVectorRepository,
        *,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> IndexingService:
        """Async factory - preferred way to create IndexingService.

        Handles async-context initialization (lock) and loads existing state.
        """
        state_lock = asyncio.Lock()
        file_lock = filelock.FileLock(str(state_path) + '.lock')

        # Load existing state or create empty (under file lock for safety)
        with file_lock:
            if state_path.exists():
                data = json.loads(state_path.read_text())
                state = DirectoryIndexState.model_validate(data)
            else:
                state = DirectoryIndexState(
                    directory_path='',
                    files={},
                    last_full_scan=datetime.now(UTC),
                )

        return cls(
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            sparse_embedding_service=sparse_embedding_service,
            repository=repository,
            state_lock=state_lock,
            file_lock=file_lock,
            state=state,
            state_path=state_path,
        )

    def __init__(
        self,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
        repository: DocumentVectorRepository,
        state_lock: asyncio.Lock,
        file_lock: filelock.FileLock,
        state: DirectoryIndexState,
        *,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        """Initialize indexing service. Use create() instead for proper async initialization."""
        self._chunking = chunking_service
        self._embedding = embedding_service
        self._sparse_embedding = sparse_embedding_service
        self._repo = repository
        self._state_path = state_path
        self._state = state
        self._state_lock = state_lock
        self._file_lock = file_lock  # Cross-process lock for state file
        self._operation: _OperationState | None = None

    async def get_index_stats(self) -> Mapping[str, int | str]:
        """Get current index statistics."""
        stats = await self._repo.get_storage_stats()
        if stats is None:
            return {'status': 'not_initialized'}

        return {
            'status': stats.status,
            'vector_dimension': stats.vector_dimension,
            'points_count': stats.points_count,
        }

    def get_pipeline_snapshot(self) -> PipelineSnapshot | None:
        """Get point-in-time snapshot of running pipeline.

        Returns None if no operation is running.
        Caller adds lifecycle status and cross-layer metrics.
        """
        if self._operation is None:
            return None

        op = self._operation
        files_to_process = op.files_found - op.files_cached
        in_chunk, in_embed, in_store = op.tracer.get_in_flight_counts()

        # Build live file type summary from scan data
        by_file_type: dict[FileType, str] = {}
        for ft in op.scanned_by_type:
            stats = FileTypeStats(
                scanned=op.scanned_by_type.get(ft, 0),
                cached=op.cached_by_type.get(ft, 0),
            )
            by_file_type[ft] = stats.to_summary()

        return PipelineSnapshot(
            scan_complete=op.scan_complete,
            files_found=op.files_found,
            files_to_process=files_to_process,
            files_cached=op.files_cached,
            chunks_ingested=op.counters.chunks_ingested,
            chunks_embedded=op.counters.chunks_embedded,
            embed_cache_hits=self._embedding.cache_hits,
            embed_cache_misses=self._embedding.cache_misses,
            chunks_stored=op.counters.chunks_stored,
            files_chunked=op.counters.files_chunked,
            files_embedded=op.counters.files_embedded,
            files_stored=op.counters.files_stored,
            files_awaiting_chunk=op.file_queue.qsize(),
            files_awaiting_embed=op.embed_queue.qsize(),
            files_awaiting_store=op.upsert_queue.qsize(),
            files_in_chunk=in_chunk,
            files_in_embed=in_embed,
            files_in_store=in_store,
            by_file_type=by_file_type,
            files_done=len(op.results),
            files_errored=sum(1 for v in op.results.values() if isinstance(v, FileProcessingError)),
            elapsed_seconds=time.monotonic() - op.start_time,
            queue_depth_series=list(op.tracer.get_queue_depths()),
        )

    async def index_file(self, file_path: Path) -> IndexingResult:
        """Index a single file directly (bypasses pipeline for simplicity).

        Useful for testing, quick re-indexing of specific files, or targeted updates.
        Follows same patterns as pipeline workers: fail-fast on infrastructure errors,
        upsert-then-delete for atomicity.

        Args:
            file_path: Path to file to index.

        Returns:
            IndexingResult with counts.
        """
        if not file_path.is_file():
            raise ValueError(f'Not a file: {file_path}')

        await self._repo.ensure_collection(EMBEDDING_DIMENSION)

        file_key = str(file_path)
        timer = Timer()

        # Chunk
        ft = get_file_type(file_path)
        chunks = list(await self._chunking.chunk_file(file_path))
        if not chunks:
            by_file_type: dict[FileType, str] = {}
            if ft is not None:
                by_file_type[ft] = FileTypeStats(scanned=1, no_content=1).to_summary()
            return IndexingResult(
                files_scanned=1,
                files_indexed=0,
                files_cached=0,
                files_no_content=1,
                chunks_created=0,
                chunks_deleted=0,
                embeddings_created=0,
                embed_cache_hits=self._embedding.cache_hits,
                embed_cache_misses=self._embedding.cache_misses,
                by_file_type=by_file_type,
                elapsed_seconds=round(timer.elapsed(), 3),
                errors=(),
            )

        chunk_ids = [_deterministic_chunk_id(file_key, c.chunk_index, c.text) for c in chunks]

        # Embed (dense + sparse)
        texts = [c.text for c in chunks]
        responses = await self._embedding.embed_texts(texts)
        dense = [r.values for r in responses]

        sparse_results, _wall, _cpu = await self._sparse_embedding.embed_batch(texts)
        # Build points using same factory as pipeline
        points = [
            VectorPoint.from_chunk(chunk, dense_emb, sparse_indices, sparse_values, chunk_id)
            for chunk, dense_emb, (sparse_indices, sparse_values), chunk_id in zip(
                chunks, dense, sparse_results, chunk_ids
            )
        ]

        # Upsert new chunks
        await self._repo.upsert(points)

        # Delete only obsolete chunks (upsert-then-delete for atomicity)
        async with self._state_lock:
            old_state = self._state.files.get(file_key) if self._state else None
        chunks_deleted = await self._delete_obsolete_chunks(old_state, chunk_ids)

        # Update state
        file_hash = _file_hash(file_path)
        async with self._state_lock:
            if self._state is not None:
                new_files = dict(self._state.files)
                new_files[file_key] = FileIndexState(
                    file_path=file_key,
                    file_hash=file_hash,
                    file_size=file_path.stat().st_size,
                    chunk_count=len(chunks),
                    chunk_ids=chunk_ids,
                    indexed_at=datetime.now(UTC),
                    chunk_strategy_version=CHUNK_STRATEGY_VERSION,
                )
                self._state = DirectoryIndexState(
                    directory_path=self._state.directory_path,
                    files=new_files,
                    last_full_scan=self._state.last_full_scan,
                    total_files=self._state.total_files,
                    total_chunks=self._state.total_chunks + len(chunks) - chunks_deleted,
                )

        self._save_state()

        # Build by_file_type for single file
        by_file_type_final: dict[FileType, str] = {}
        if ft is not None:
            by_file_type_final[ft] = FileTypeStats(scanned=1, indexed=1, chunks=len(chunks)).to_summary()

        # Get index totals
        index_files = len(self._state.files) if self._state else 0
        index_chunks = self._state.total_chunks if self._state else 0

        return IndexingResult(
            files_scanned=1,
            files_indexed=1,
            files_cached=0,
            files_no_content=0,
            chunks_created=len(chunks),
            chunks_deleted=chunks_deleted,
            embeddings_created=len(chunks),
            embed_cache_hits=self._embedding.cache_hits,
            embed_cache_misses=self._embedding.cache_misses,
            by_file_type=by_file_type_final,
            index_files=index_files,
            index_chunks=index_chunks,
            elapsed_seconds=round(timer.elapsed(), 3),
            errors=(),
        )

    async def index_directory(
        self,
        directory: Path,
        *,
        full_reindex: bool = False,
        respect_gitignore: bool | None = None,
        on_progress: ProgressCallback | None = None,
        stop_after: StopAfterStage | None = None,
    ) -> IndexingResult:
        """Index files using pipeline architecture with separate worker pools.

        Pipeline stages:
        1. Chunk workers: Read files and split into chunks (CPU-bound)
        2. Embed workers: Generate dense + sparse embeddings (Gemini I/O-bound)
        3. Upsert workers: Store in Qdrant and update state (Qdrant I/O-bound)

        Each stage runs independently, connected by queues. This allows:
        - Embed workers to always be making Gemini calls (not blocked on Qdrant)
        - Upsert workers to always be writing to Qdrant (not blocked on Gemini)
        - Independent tuning of worker counts per stage

        Args:
            directory: Directory to index.
            full_reindex: If True, reindex all files regardless of hash.
            respect_gitignore: Git ignore filtering behavior.
            on_progress: Optional callback for progress updates (not yet implemented).

        Returns:
            IndexingResult with counts and any errors.
        """
        if not directory.is_dir():
            raise ValueError(f'Not a directory: {directory}')

        await self._repo.ensure_collection(EMBEDDING_DIMENSION)
        self._state = self._load_state(directory)
        self._state_lock = asyncio.Lock()

        timer = Timer()
        chunks_deleted = 0

        # PHASE 1: Scan files
        extensions = set(EXTENSION_MAP.keys())
        git_root = _find_git_root(str(directory))

        # Create operation state before discovery so the monitor can report progress
        file_queue: asyncio.Queue[Path] = asyncio.Queue()
        embed_queue: asyncio.Queue[_ChunkedFile] = asyncio.Queue(maxsize=EMBED_QUEUE_SIZE)
        upsert_queue: asyncio.Queue[_EmbeddedFile] = asyncio.Queue(maxsize=UPSERT_QUEUE_SIZE)
        results: dict[str, int | FileProcessingError] = {}
        results_lock = asyncio.Lock()

        tracer = PipelineTracer(file_queue, embed_queue, upsert_queue)

        self._operation = _OperationState(
            file_queue=file_queue,
            embed_queue=embed_queue,
            upsert_queue=upsert_queue,
            counters=PipelineCounters(),
            results=results,
            results_lock=results_lock,
            scan_complete=False,
            files_found=0,
            files_cached=0,
            start_time=time.monotonic(),
            scanned_by_type={},
            cached_by_type={},
            tracer=tracer,
        )

        # Run file discovery in a thread so the event loop (and monitor) stays responsive.
        # Both paths update operation.files_found incrementally for dashboard visibility.
        logger.info('[SCAN] Starting file discovery...')
        if respect_gitignore is not False and git_root:
            all_files, files_ignored = await asyncio.to_thread(
                _get_git_files,
                directory,
                extensions,
                self._operation,
            )
        else:
            all_files = await asyncio.to_thread(
                _walk_files_counted,
                directory,
                extensions,
                self._operation,
            )
            files_ignored = 0

        files_total = len(all_files)
        self._operation.files_found = files_total
        ignored_text = f' ({files_ignored:,} ignored)' if files_ignored else ''
        logger.info(f'[SCAN] Discovered {files_total:,} files{ignored_text}')

        # FULL REINDEX: Clean slate
        if full_reindex:
            chunks_deleted = await self._repo.delete_by_source_path_prefix(str(directory))
            if chunks_deleted > 0:
                logger.info(f'[PIPELINE] Deleted {chunks_deleted} existing chunks for clean slate')

            if self._state:
                dir_prefix = str(directory)
                new_files = {
                    k: v
                    for k, v in self._state.files.items()
                    if not (k == dir_prefix or k.startswith(dir_prefix + '/'))
                }
                self._state = DirectoryIndexState(
                    directory_path=self._state.directory_path,
                    files=new_files,
                    last_full_scan=self._state.last_full_scan,
                    total_chunks=self._state.total_chunks - chunks_deleted,
                    total_files=len(new_files),
                    metadata_version=self._state.metadata_version,
                )

        # Classify files: determine which need indexing vs cached (single pass).
        # Yields to event loop periodically so the monitor can capture scan progress.
        logger.info(f'[SCAN] Classifying {files_total:,} files...')
        scanned_by_type = self._operation.scanned_by_type
        cached_by_type = self._operation.cached_by_type
        files_to_index: list[Path] = []

        for i, file_path in enumerate(all_files):
            ft = get_file_type(file_path)
            if ft is not None:
                scanned_by_type[ft] = scanned_by_type.get(ft, 0) + 1

            if full_reindex or self._needs_indexing(file_path):
                files_to_index.append(file_path)
            elif ft is not None:
                cached_by_type[ft] = cached_by_type.get(ft, 0) + 1
                self._operation.files_cached += 1

            # Yield to event loop every 100 files for monitoring visibility
            if i % 100 == 99:
                await asyncio.sleep(0)

        self._operation.scan_complete = True
        files_cached_count = sum(cached_by_type.values())
        logger.info(
            f'[PIPELINE] Scan complete in {timer.elapsed():.1f}s: '
            f'{len(files_to_index)} to index, {files_cached_count} cached'
        )

        if stop_after == 'scan':
            scan_by_type: dict[FileType, str] = {}
            for ft in scanned_by_type:
                stats = FileTypeStats(
                    scanned=scanned_by_type.get(ft, 0),
                    cached=cached_by_type.get(ft, 0),
                )
                scan_by_type[ft] = stats.to_summary()

            self._operation = None
            return IndexingResult(
                files_scanned=files_total,
                files_ignored=files_ignored,
                files_indexed=0,
                files_cached=sum(cached_by_type.values()),
                files_no_content=0,
                chunks_created=0,
                chunks_deleted=chunks_deleted,
                embeddings_created=0,
                embed_cache_hits=0,
                embed_cache_misses=0,
                by_file_type=scan_by_type,
                elapsed_seconds=round(timer.elapsed(), 3),
                errors=(),
                stopped_after='scan',
            )

        if not files_to_index:
            # Build by_file_type summary for cached-only result
            by_file_type: dict[FileType, str] = {}
            for ft in scanned_by_type:
                stats = FileTypeStats(
                    scanned=scanned_by_type.get(ft, 0),
                    cached=cached_by_type.get(ft, 0),
                )
                by_file_type[ft] = stats.to_summary()

            self._operation = None
            return IndexingResult(
                files_scanned=files_total,
                files_ignored=files_ignored,
                files_indexed=0,
                files_cached=sum(cached_by_type.values()),
                files_no_content=0,
                chunks_created=0,
                chunks_deleted=chunks_deleted,
                embeddings_created=0,
                embed_cache_hits=self._embedding.cache_hits,
                embed_cache_misses=self._embedding.cache_misses,
                by_file_type=by_file_type,
                elapsed_seconds=round(timer.elapsed(), 3),
                errors=(),
                stopped_after=stop_after,
            )

        # Record scan phase duration
        tracer.record_scan_seconds(timer.elapsed())

        # Populate file queue with chunk:queued events
        for path in files_to_index:
            tracer.record(str(path), 'chunk', 'queued')
            await file_queue.put(path)

        logger.info(
            f'[PIPELINE] Starting workers: {NUM_CHUNK_WORKERS} chunk, '
            f'{NUM_EMBED_WORKERS} embed, {NUM_UPSERT_WORKERS} upsert '
            f'for {len(files_to_index)} files'
        )

        # Disable HNSW indexing during bulk upsert to free CPU for embedding workers.
        # Save original value so we restore exactly what was configured.
        original_indexing_threshold = await self._repo.get_indexing_threshold()
        await self._repo.set_indexing_threshold(0)

        # Start queue depth monitoring (1Hz sampling for time series)
        tracer.start_monitoring()

        # Start workers for each stage (drain workers replace downstream stages when stop_after is set)
        counters = self._operation.counters
        chunk_tasks = [
            asyncio.create_task(
                self._pipeline_chunk_worker(file_queue, embed_queue, results, results_lock, counters, tracer)
            )
            for _ in range(min(NUM_CHUNK_WORKERS, len(files_to_index)))
        ]

        if stop_after == 'chunk':
            embed_tasks = [
                asyncio.create_task(self._drain_worker(embed_queue, counters, 'chunks_embedded', results, results_lock))
                for _ in range(NUM_EMBED_WORKERS)
            ]
            upsert_tasks = []
        else:
            embed_tasks = [
                asyncio.create_task(self._pipeline_embed_worker(embed_queue, upsert_queue, counters, tracer))
                for _ in range(NUM_EMBED_WORKERS)
            ]
            if stop_after == 'embed':
                upsert_tasks = [
                    asyncio.create_task(
                        self._drain_worker(upsert_queue, counters, 'chunks_stored', results, results_lock)
                    )
                    for _ in range(NUM_UPSERT_WORKERS)
                ]
            else:
                upsert_tasks = [
                    asyncio.create_task(
                        self._pipeline_upsert_worker(upsert_queue, results, results_lock, counters, tracer)
                    )
                    for _ in range(NUM_UPSERT_WORKERS)
                ]

        all_worker_tasks = chunk_tasks + embed_tasks + upsert_tasks

        # Create waiter task for queue draining
        async def wait_queues() -> None:
            await file_queue.join()
            await embed_queue.join()
            await upsert_queue.join()

        waiter = asyncio.create_task(wait_queues())

        # FAIL-FAST: Wait for either queues to drain OR any worker to crash
        # If a worker raises an exception, it completes without calling task_done(),
        # so the waiter would hang forever. asyncio.wait detects the worker crash first.
        try:
            pending: set[asyncio.Task[None]] = {waiter, *all_worker_tasks}

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    if task is waiter:
                        # Queues drained - success! Re-raise if join() somehow failed
                        task.result()
                        pending.clear()
                        break
                    elif not task.cancelled():
                        # A worker completed - check for exception
                        exc = task.exception()
                        if exc is not None:
                            raise exc
                        # Worker exited without exception (shouldn't happen normally)
        finally:
            # Stop tracer monitoring and cancel all workers
            tracer.stop_monitoring()
            waiter.cancel()
            for task in all_worker_tasks:
                task.cancel()
            await asyncio.gather(waiter, *all_worker_tasks, return_exceptions=True)
            # Restore HNSW indexing — Qdrant rebuilds the index in a single pass.
            # Must be in finally to prevent threshold=0 persisting after worker errors.
            await self._repo.set_indexing_threshold(original_indexing_threshold)

        # Collect results and aggregate by file type
        errors: list[FileProcessingError] = []
        files_indexed = 0
        files_no_content = 0
        chunks_created = 0

        # Track processing outcomes by file type
        indexed_by_type: dict[FileType, int] = {}
        no_content_by_type: dict[FileType, int] = {}
        errored_by_type: dict[FileType, int] = {}
        chunks_by_type: dict[FileType, int] = {}

        for file_key, result in results.items():
            ft = get_file_type(Path(file_key))
            if isinstance(result, FileProcessingError):
                errors.append(result)
                if ft is not None:
                    errored_by_type[ft] = errored_by_type.get(ft, 0) + 1
            elif result == 0:
                files_no_content += 1
                if ft is not None:
                    no_content_by_type[ft] = no_content_by_type.get(ft, 0) + 1
            else:
                files_indexed += 1
                chunks_created += result
                if ft is not None:
                    indexed_by_type[ft] = indexed_by_type.get(ft, 0) + 1
                    chunks_by_type[ft] = chunks_by_type.get(ft, 0) + result

        # Build by_file_type summary mapping
        # scanned_by_type contains all file types (cached_by_type is a subset)
        by_file_type_result: dict[FileType, str] = {}
        for ft in scanned_by_type:
            stats = FileTypeStats(
                scanned=scanned_by_type.get(ft, 0),
                indexed=indexed_by_type.get(ft, 0),
                no_content=no_content_by_type.get(ft, 0),
                cached=cached_by_type.get(ft, 0),
                errored=errored_by_type.get(ft, 0),
                chunks=chunks_by_type.get(ft, 0),
            )
            by_file_type_result[ft] = stats.to_summary()

        files_cached = sum(cached_by_type.values())

        # Save state
        self._save_state()

        previous_chunks = self._state.total_chunks if self._state else 0
        self._state = DirectoryIndexState(
            directory_path=str(directory),
            files=self._state.files if self._state else {},
            last_full_scan=datetime.now(UTC),
            total_files=files_indexed + files_no_content + files_cached,
            total_chunks=previous_chunks + chunks_created - chunks_deleted,
        )
        self._save_state()

        # Build timing report from tracer
        timing_report = tracer.build_report()

        # Get index totals
        index_files = len(self._state.files) if self._state else 0
        index_chunks = self._state.total_chunks if self._state else 0

        # Clear operation state
        self._operation = None

        return IndexingResult(
            files_scanned=files_total,
            files_ignored=files_ignored,
            files_indexed=files_indexed,
            files_cached=files_cached,
            files_no_content=files_no_content,
            chunks_created=chunks_created,
            chunks_deleted=chunks_deleted,
            embeddings_created=chunks_created,
            embed_cache_hits=self._embedding.cache_hits,
            embed_cache_misses=self._embedding.cache_misses,
            by_file_type=by_file_type_result,
            index_files=index_files,
            index_chunks=index_chunks,
            elapsed_seconds=round(timer.elapsed(), 3),
            errors=errors,
            stopped_after=stop_after,
            timing=timing_report,
        )

    def shutdown(self) -> None:
        """Shutdown services and release resources.

        Shuts down ProcessPoolExecutors used for PDF chunking and sparse embeddings.
        Should be called when the service is no longer needed.
        """
        self._chunking.shutdown()
        self._sparse_embedding.shutdown()

    async def clear_documents(self, path: str) -> ClearResult:
        """Clear documents from the index and update state file.

        Args:
            path: Resolved path to clear. Use "**" for entire index.

        Returns:
            ClearResult with counts of files and chunks removed.
        """
        # Delete from Qdrant
        files_removed, chunks_removed = await self._repo.clear_documents(path)

        # Update state file
        async with self._state_lock:
            if self._state is not None:
                if path == '**':
                    # Clear entire state (files_removed is 0 after collection drop)
                    self._state = DirectoryIndexState(
                        directory_path='',
                        files={},
                        last_full_scan=datetime.now(UTC),
                        total_files=0,
                        total_chunks=0,
                    )
                elif files_removed > 0:
                    # Remove matching entries from state (single pass)
                    new_files: dict[str, FileIndexState] = {}
                    removed_chunks = 0
                    for k, v in self._state.files.items():
                        if k == path or k.startswith(path + '/'):
                            removed_chunks += v.chunk_count
                        else:
                            new_files[k] = v
                    self._state = DirectoryIndexState(
                        directory_path=self._state.directory_path,
                        files=new_files,
                        last_full_scan=self._state.last_full_scan,
                        total_files=len(new_files),
                        total_chunks=max(0, self._state.total_chunks - removed_chunks),
                    )

        self._save_state()

        return ClearResult(
            files_removed=files_removed,
            chunks_removed=chunks_removed,
            path=None if path == '**' else path,
        )

    def _load_state(self, directory: Path) -> DirectoryIndexState:
        """Load or create indexing state for directory.

        State is shared across all directories - files are keyed by absolute path.
        """
        if self._state_path.exists():
            data = json.loads(self._state_path.read_text())
            return DirectoryIndexState.model_validate(data)

        return DirectoryIndexState(
            directory_path=str(directory),
            files={},
            last_full_scan=datetime.now(UTC),
        )

    def _save_state(self) -> None:
        """Persist indexing state to disk (cross-process safe)."""
        if self._state is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        # Use file lock for cross-process safety
        with self._file_lock:
            self._state_path.write_text(json.dumps(self._state.model_dump(mode='json'), indent=2))

    def _needs_indexing(self, path: Path) -> bool:
        """Check if file needs (re)indexing based on content hash."""
        if self._state is None:
            return True

        file_state = self._state.files.get(str(path))
        if file_state is None:
            return True  # Never indexed

        # Check if chunking strategy changed
        if file_state.chunk_strategy_version != CHUNK_STRATEGY_VERSION:
            return True

        # Check file size first (quick check)
        try:
            current_size = path.stat().st_size
            if current_size != file_state.file_size:
                return True
        except OSError:
            return True

        # Check content hash
        current_hash = _file_hash(path)
        return current_hash != file_state.file_hash

    async def _pipeline_chunk_worker(
        self,
        file_queue: asyncio.Queue[Path],
        embed_queue: asyncio.Queue[_ChunkedFile],
        results: dict[str, int | FileProcessingError],  # strict_typing_linter.py: mutable-type
        results_lock: asyncio.Lock,
        counters: PipelineCounters,
        tracer: PipelineTracer,
    ) -> None:
        """Stage 1: Chunk files and push to embed queue.

        Known file errors (encoding, read failures, timeout) are recorded and skipped.
        Infrastructure errors propagate immediately (fail-fast).
        """
        while True:
            file_path = await file_queue.get()

            file_key = str(file_path)
            tracer.record(file_key, 'chunk', 'started')
            try:
                chunks = list(
                    await asyncio.wait_for(
                        self._chunking.chunk_file(file_path),
                        timeout=FILE_CHUNK_TIMEOUT_SECONDS,
                    )
                )
                if chunks:
                    chunk_ids = [_deterministic_chunk_id(file_key, c.chunk_index, c.text) for c in chunks]
                    tracer.record(file_key, 'chunk', 'completed')
                    tracer.record(file_key, 'embed', 'queued')
                    await embed_queue.put(
                        _ChunkedFile(
                            file_path=file_path,
                            file_hash=_file_hash(file_path),
                            file_size=file_path.stat().st_size,
                            chunks=chunks,
                            chunk_ids=chunk_ids,
                        )
                    )
                    counters.chunks_ingested += len(chunks)
                    counters.files_chunked += 1
                else:
                    # File produced 0 chunks (content too short) - record in state to avoid re-processing
                    tracer.record(file_key, 'chunk', 'completed')
                    async with self._state_lock:
                        if self._state is not None:
                            new_files = dict(self._state.files)
                            new_files[file_key] = FileIndexState(
                                file_path=file_key,
                                file_hash=_file_hash(file_path),
                                file_size=file_path.stat().st_size,
                                chunk_count=0,
                                chunk_ids=(),
                                indexed_at=datetime.now(UTC),
                                chunk_strategy_version=CHUNK_STRATEGY_VERSION,
                            )
                            self._state = DirectoryIndexState(
                                directory_path=self._state.directory_path,
                                files=new_files,
                                last_full_scan=self._state.last_full_scan,
                                total_files=self._state.total_files,
                                total_chunks=self._state.total_chunks,
                            )
                    async with results_lock:
                        results[file_key] = 0  # 0 chunks created
                    logger.info(f'[CHUNK] {file_path.name}: 0 chunks (content too short)')

            except (TimeoutError, OSError, UnicodeDecodeError) as e:
                # Known file-level errors: record and continue
                tracer.record(file_key, 'chunk', 'errored')
                async with results_lock:
                    results[file_key] = FileProcessingError(
                        file_path=file_key,
                        file_type=get_file_type(file_path),
                        error_type=type(e).__name__,
                        message=str(e),
                        recoverable=True,
                    )
                logger.warning(f'[CHUNK] Skipping {file_path.name}: {type(e).__name__}: {e}', exc_info=True)

            # Mark done whether success or known error (unknown errors propagate)
            file_queue.task_done()

    async def _pipeline_embed_worker(
        self,
        embed_queue: asyncio.Queue[_ChunkedFile],
        upsert_queue: asyncio.Queue[_EmbeddedFile],
        counters: PipelineCounters,
        tracer: PipelineTracer,
    ) -> None:
        """Stage 2: Generate dense + sparse embeddings and push to upsert queue.

        Accumulates chunks from multiple small files before sending to
        ProcessPoolExecutor for sparse embeddings, reducing IPC overhead.
        Dense embeddings use BatchLoader which handles coalescing internally.

        Fail-fast: exceptions propagate immediately, task_done() only on success.
        """
        # Accumulation settings for sparse embedding batch efficiency.
        # Rayon needs enough work items to saturate cores — benchmarks show
        # ~100 texts per thread for 95%+ efficiency on medium/long texts.
        # The timeout ensures we don't stall when work arrives slowly.
        sparse_threads = self._sparse_embedding.thread_count
        BATCH_THRESHOLD = max(500, sparse_threads * 300)
        BATCH_TIMEOUT = 0.05  # 50ms - don't wait forever for small files
        logger.info(f'[EMBED] batch_threshold={BATCH_THRESHOLD} (sparse_threads={sparse_threads})')

        # Accumulators
        accumulated_files: list[_ChunkedFile] = []
        accumulated_texts: list[str] = []
        text_boundaries: list[int] = []  # Cumulative end index for each file

        async def flush_batch() -> None:
            """Generate embeddings and push files to upsert queue."""
            nonlocal accumulated_files, accumulated_texts, text_boundaries

            if not accumulated_texts:
                return

            batch_size = len(accumulated_files)
            flush_t0 = time.perf_counter()
            logger.info(
                f'[EMBED-FLUSH] {batch_size} files, {len(accumulated_texts)} texts, t={flush_t0 - tracer.start_time:.3f}s'
            )

            # Record batch start for all files in batch
            for chunked in accumulated_files:
                fk = str(chunked.file_path)
                tracer.record(fk, 'embed', 'batch_started')
                tracer.record(fk, 'embed_sparse', 'started')
                tracer.record(fk, 'embed_dense', 'started')
                tracer.record_batch_size(fk, batch_size)

            # Run sparse and dense embeddings in parallel with independent timing
            async def sparse_with_tracing() -> tuple[Sequence[tuple[Sequence[int], Sequence[float]]], float]:
                results, wall_secs, cpu_secs = await self._sparse_embedding.embed_batch(accumulated_texts)

                for chunked in accumulated_files:
                    fk = str(chunked.file_path)
                    tracer.record(fk, 'embed_sparse', 'completed')
                    tracer.record_cpu(fk, 'embed_sparse', cpu_secs)

                parallel = cpu_secs / wall_secs if wall_secs > 0 else 0.0
                logger.info(
                    f'[EMBED-SPARSE] {len(accumulated_texts)} texts in {wall_secs:.3f}s wall, '
                    f'{cpu_secs:.3f}s cpu ({parallel:.1f}x), t={time.perf_counter() - tracer.start_time:.3f}s'
                )
                return results, cpu_secs

            async def dense_with_tracing() -> Sequence[EmbedResponse]:
                t0 = time.perf_counter()
                results = await self._embedding.embed_texts(accumulated_texts)
                elapsed = time.perf_counter() - t0

                for chunked in accumulated_files:
                    tracer.record(str(chunked.file_path), 'embed_dense', 'completed')

                logger.info(
                    f'[EMBED-DENSE] {len(accumulated_texts)} texts done in {elapsed:.3f}s, '
                    f't={time.perf_counter() - tracer.start_time:.3f}s'
                )
                return results

            (sparse_results, _sparse_cpu), dense_results = await asyncio.gather(
                sparse_with_tracing(),
                dense_with_tracing(),
            )

            # Distribute results back to files and push to upsert queue
            start_idx = 0
            for i, chunked in enumerate(accumulated_files):
                end_idx = text_boundaries[i]
                file_sparse = list(sparse_results[start_idx:end_idx])
                file_dense = list(dense_results[start_idx:end_idx])

                # Create _EmbeddedFile with both embeddings
                embedded = _EmbeddedFile(
                    file_path=chunked.file_path,
                    file_hash=chunked.file_hash,
                    file_size=chunked.file_size,
                    chunks=chunked.chunks,
                    chunk_ids=chunked.chunk_ids,
                    dense_embeddings=[r.values for r in file_dense],
                    sparse_embeddings=file_sparse,
                )
                fk = str(chunked.file_path)
                tracer.record(fk, 'embed', 'completed')
                tracer.record(fk, 'store', 'queued')
                await upsert_queue.put(embedded)
                embed_queue.task_done()
                counters.chunks_embedded += len(chunked.chunks)
                counters.files_embedded += 1

                start_idx = end_idx

            # Reset accumulators
            accumulated_files = []
            accumulated_texts = []
            text_boundaries = []

        async def process_single_file(chunked: _ChunkedFile) -> None:
            """Process a single file (used for large files to avoid accumulation)."""
            fk = str(chunked.file_path)
            texts = [c.text for c in chunked.chunks]

            tracer.record(fk, 'embed', 'batch_started')
            tracer.record_batch_size(fk, 1)
            tracer.record(fk, 'embed_sparse', 'started')
            tracer.record(fk, 'embed_dense', 'started')

            # Run sparse and dense embeddings in parallel with independent completion tracking
            async def sparse_single() -> tuple[Sequence[tuple[Sequence[int], Sequence[float]]], float]:
                results, _wall, cpu_secs = await self._sparse_embedding.embed_batch(texts)
                tracer.record(fk, 'embed_sparse', 'completed')
                tracer.record_cpu(fk, 'embed_sparse', cpu_secs)
                return results, cpu_secs

            async def dense_single() -> Sequence[EmbedResponse]:
                results = await self._embedding.embed_texts(texts)
                tracer.record(fk, 'embed_dense', 'completed')
                return results

            (sparse_results, _sparse_cpu), dense_results = await asyncio.gather(
                sparse_single(),
                dense_single(),
            )

            # Create _EmbeddedFile
            embedded = _EmbeddedFile(
                file_path=chunked.file_path,
                file_hash=chunked.file_hash,
                file_size=chunked.file_size,
                chunks=chunked.chunks,
                chunk_ids=chunked.chunk_ids,
                dense_embeddings=[r.values for r in dense_results],
                sparse_embeddings=sparse_results,
            )
            tracer.record(fk, 'embed', 'completed')
            tracer.record(fk, 'store', 'queued')
            await upsert_queue.put(embedded)
            embed_queue.task_done()
            counters.chunks_embedded += len(chunked.chunks)
            counters.files_embedded += 1

        _worker_id = id(asyncio.current_task()) % 10000  # short ID for logs

        while True:
            try:
                # Try to get item with timeout to allow periodic flushing
                try:
                    get_t0 = time.perf_counter()
                    chunked = await asyncio.wait_for(
                        embed_queue.get(),
                        timeout=BATCH_TIMEOUT,
                    )
                    get_elapsed = time.perf_counter() - get_t0
                    if get_elapsed > 0.1:  # log slow gets (>100ms)
                        logger.info(
                            f'[EMBED-W{_worker_id}] get() took {get_elapsed:.3f}s, accum={len(accumulated_texts)} texts'
                        )
                except TimeoutError:
                    # No items available, flush what we have
                    if accumulated_texts:
                        logger.info(
                            f'[EMBED-W{_worker_id}] timeout flush: {len(accumulated_texts)} texts, '
                            f'{len(accumulated_files)} files, t={time.perf_counter() - tracer.start_time:.3f}s'
                        )
                    await flush_batch()
                    continue

                # Record dequeue event (entering accumulator)
                tracer.record(str(chunked.file_path), 'embed', 'dequeued')

                texts = [c.text for c in chunked.chunks]

                # Large files: process immediately without accumulation
                if len(texts) >= BATCH_THRESHOLD:
                    # Flush any accumulated small files first
                    await flush_batch()
                    # Process large file directly
                    await process_single_file(chunked)
                    continue

                # Small files: accumulate
                accumulated_files.append(chunked)
                accumulated_texts.extend(texts)
                text_boundaries.append(len(accumulated_texts))

                # Flush if we've accumulated enough
                if len(accumulated_texts) >= BATCH_THRESHOLD:
                    await flush_batch()

            except asyncio.CancelledError:
                # Final flush before exit, but preserve cancellation signal
                try:
                    await flush_batch()
                except Exception as e:
                    logger.warning(f'Failed to flush batch during shutdown: {e}', exc_info=True)
                raise  # Re-raise CancelledError to signal proper cancellation

    async def _pipeline_upsert_worker(
        self,
        upsert_queue: asyncio.Queue[_EmbeddedFile],
        results: dict[str, int | FileProcessingError],  # strict_typing_linter.py: mutable-type
        results_lock: asyncio.Lock,
        counters: PipelineCounters,
        tracer: PipelineTracer,
    ) -> None:
        """Stage 3: Upsert to Qdrant, delete old chunks, update state.

        Fail-fast: exceptions propagate immediately, task_done() only on success.
        """
        while True:
            embedded = await upsert_queue.get()

            file_key = str(embedded.file_path)
            tracer.record(file_key, 'store', 'started')

            # No try/finally - exceptions propagate, triggering fail-fast
            # Build points
            points = [
                VectorPoint.from_chunk(chunk, dense, sparse_indices, sparse_values, chunk_id)
                for chunk, dense, (sparse_indices, sparse_values), chunk_id in zip(
                    embedded.chunks,
                    embedded.dense_embeddings,
                    embedded.sparse_embeddings,
                    embedded.chunk_ids,
                )
            ]

            # Upsert new chunks
            await self._repo.upsert(points)
            counters.chunks_stored += len(embedded.chunks)
            counters.files_stored += 1

            # Delete only obsolete chunks
            if self._state_lock is None:
                raise RuntimeError('State lock not initialized')

            async with self._state_lock:
                old_state = self._state.files.get(file_key) if self._state else None
            await self._delete_obsolete_chunks(old_state, embedded.chunk_ids)

            # Update state
            async with self._state_lock:
                if self._state is not None:
                    new_files = dict(self._state.files)
                    new_files[file_key] = FileIndexState(
                        file_path=file_key,
                        file_hash=embedded.file_hash,
                        file_size=embedded.file_size,
                        chunk_count=len(embedded.chunks),
                        chunk_ids=embedded.chunk_ids,
                        indexed_at=datetime.now(UTC),
                        chunk_strategy_version=CHUNK_STRATEGY_VERSION,
                    )
                    self._state = DirectoryIndexState(
                        directory_path=self._state.directory_path,
                        files=new_files,
                        last_full_scan=self._state.last_full_scan,
                        total_files=self._state.total_files,
                        total_chunks=self._state.total_chunks,
                    )

            # Record success
            tracer.record(file_key, 'store', 'completed')
            async with results_lock:
                results[file_key] = len(embedded.chunks)

            logger.info(f'[UPSERT] {embedded.file_path.name}: {len(embedded.chunks)} chunks')

            # Only mark done on success
            upsert_queue.task_done()

    async def _drain_worker(
        self,
        queue: asyncio.Queue[_ChunkedFile] | asyncio.Queue[_EmbeddedFile],
        counters: PipelineCounters,
        counter_attr: str,
        results: dict[str, int | FileProcessingError],  # strict_typing_linter.py: mutable-type
        results_lock: asyncio.Lock,
    ) -> None:
        """Drain a queue, counting chunks without processing.

        Replaces real workers when stop_after truncates the pipeline.
        Records chunk counts in results so final aggregation works.
        """
        while True:
            item = await queue.get()
            chunk_count = len(item.chunks)
            setattr(counters, counter_attr, getattr(counters, counter_attr) + chunk_count)
            async with results_lock:
                results[str(item.file_path)] = chunk_count
            queue.task_done()

    async def _delete_obsolete_chunks(
        self,
        old_state: FileIndexState | None,
        new_chunk_ids: Sequence[UUID],
    ) -> int:
        """Delete chunks that no longer exist in the new set.

        Only deletes IDs that were in the old state but not in the new set.
        This prevents deleting chunks that were just upserted when IDs are
        deterministic (same file content = same IDs).

        Returns:
            Count of chunks deleted.
        """
        if not old_state:
            return 0
        obsolete_ids = set(old_state.chunk_ids) - set(new_chunk_ids)
        if not obsolete_ids:
            return 0
        await self._repo.delete(list(obsolete_ids))
        return len(obsolete_ids)


@dataclass(frozen=True)
class PipelineSnapshot:
    """Point-in-time snapshot of pipeline state.

    Contains everything IndexingService knows about current operation.
    Excludes lifecycle status and cross-layer metrics (handled by caller).
    """

    # Scanning phase
    scan_complete: bool
    files_found: int
    files_to_process: int
    files_cached: int

    # Pipeline stages (cumulative chunk counts)
    chunks_ingested: int
    chunks_embedded: int
    embed_cache_hits: int
    embed_cache_misses: int
    chunks_stored: int

    # Per-stage file completion (files that finished each stage)
    files_chunked: int
    files_embedded: int
    files_stored: int

    # Pipeline queues (file counts waiting)
    files_awaiting_chunk: int
    files_awaiting_embed: int
    files_awaiting_store: int

    # In-flight (files currently being processed inside workers)
    files_in_chunk: int
    files_in_embed: int
    files_in_store: int

    # Results
    files_done: int
    files_errored: int

    # Timing
    elapsed_seconds: float

    # File type breakdown (live during scan)
    by_file_type: Mapping[FileType, str]

    # Queue depth time series (from tracer, 1Hz samples)
    queue_depth_series: Sequence[QueueDepthSample]


async def create_indexing_service(
    config: EmbeddingConfig,
    collection_name: str,
    *,
    redis: RedisClient,
    cache_tasks: BackgroundTaskGroup,
    qdrant_url: str = 'http://localhost:6333',
    state_path: Path = DEFAULT_STATE_PATH,
) -> IndexingService:
    """Factory function to create IndexingService with default dependencies.

    Must be called from async context - ensures semaphores are bound correctly.

    Args:
        config: Embedding configuration with provider selection.
        collection_name: Name of the collection to operate on.
        redis: Redis client for embedding cache.
        cache_tasks: Background task group for cache write tracking.
        qdrant_url: URL of Qdrant server.
        state_path: Path to persist indexing state.

    Returns:
        Configured IndexingService.
    """
    embedding_client = create_embedding_client(config)
    qdrant_client = QdrantClient(url=qdrant_url)

    chunking_service = await ChunkingService.create()
    embedding_service = EmbeddingService(
        embedding_client,
        batch_size=config.batch_size,
        redis=redis,
        cache_tasks=cache_tasks,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )
    sparse_embedding_service = await SparseEmbeddingService.create()
    repository = DocumentVectorRepository(qdrant_client, collection_name)

    return await IndexingService.create(
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        sparse_embedding_service=sparse_embedding_service,
        repository=repository,
        state_path=state_path,
    )


@dataclass
class PipelineCounters:
    """Cumulative counts at pipeline stage boundaries.

    Tracks chunks and files exiting each stage for progress monitoring.
    Dashboard uses these to show per-stage completion and queue depths.

    Thread-safe: Python's += on integers is atomic under GIL.
    """

    chunks_ingested: int = 0  # Chunk worker → embed queue
    chunks_embedded: int = 0  # Embed worker → upsert queue
    chunks_stored: int = 0  # Upsert worker → complete

    # File-level completion tracking (one file → many chunks)
    files_chunked: int = 0  # Files that completed chunk stage
    files_embedded: int = 0  # Files that completed embed stage
    files_stored: int = 0  # Files that completed store stage


@dataclass
class _OperationState:
    """Per-operation mutable state.

    Created in index_directory(), cleared on completion.
    Exposes pipeline internals for progress monitoring.
    """

    # Scanning
    scan_complete: bool

    # Pipeline queues
    file_queue: asyncio.Queue[Path]
    embed_queue: asyncio.Queue[_ChunkedFile]
    upsert_queue: asyncio.Queue[_EmbeddedFile]

    # Tracking
    counters: PipelineCounters
    results: dict[str, int | FileProcessingError]
    results_lock: asyncio.Lock

    # Metadata
    files_found: int
    files_cached: int
    start_time: float  # From time.monotonic()

    # File type breakdown (populated during scan, exposed via snapshot)
    scanned_by_type: dict[FileType, int]
    cached_by_type: dict[FileType, int]

    # Tracing
    tracer: PipelineTracer


@dataclass
class _ChunkedFile:
    """Output of chunk stage, input to embed stage."""

    file_path: Path
    file_hash: str
    file_size: int
    chunks: Sequence[Chunk]
    chunk_ids: Sequence[UUID]


@dataclass
class _EmbeddedFile:
    """Output of embed stage, input to upsert stage."""

    file_path: Path
    file_hash: str
    file_size: int
    chunks: Sequence[Chunk]
    chunk_ids: Sequence[UUID]
    dense_embeddings: Sequence[Sequence[float]]
    sparse_embeddings: Sequence[tuple[Sequence[int], Sequence[float]]]


def _walk_files(directory: Path, extensions: Set[str]) -> Iterator[Path]:
    """Walk directory yielding files with matching extensions."""
    for root, _, filenames in os.walk(directory):
        root_path = Path(root).resolve()
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                yield root_path / filename


def _walk_files_counted(
    directory: Path,
    extensions: Set[str],
    operation: _OperationState,
) -> Sequence[Path]:
    """Walk directory, updating operation.files_found as files are discovered.

    Called from a thread via asyncio.to_thread. The monitor on the main thread
    reads operation.files_found every 500ms — CPython's GIL makes int attribute
    writes safe for cross-thread reads.
    """
    files: list[Path] = []
    for file_path in _walk_files(directory, extensions):
        files.append(file_path)
        if len(files) % 100 == 0:
            operation.files_found = len(files)
    operation.files_found = len(files)
    return files


def _get_git_files(
    directory: Path,
    extensions: Set[str],
    operation: _OperationState | None = None,
) -> tuple[Sequence[Path], int]:
    """Get non-ignored files and count of ignored files using git ls-files.

    Args:
        directory: Directory to scan.
        extensions: File extensions to include.
        operation: If provided, files_found is updated incrementally.

    Returns:
        Tuple of (files to index, count of ignored files with matching extensions).
    """
    # Run both commands
    included = subprocess.run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
        capture_output=True,
        text=True,
        cwd=directory,
        timeout=30,
    )
    if included.returncode != 0:
        raise RuntimeError(f'git ls-files failed: {included.stderr}')

    ignored = subprocess.run(
        ['git', 'ls-files', '--others', '--ignored', '--exclude-standard'],
        capture_output=True,
        text=True,
        cwd=directory,
        timeout=60,
    )
    if ignored.returncode != 0:
        raise RuntimeError(f'git ls-files --ignored failed: {ignored.stderr}')

    files = []
    directory_resolved = directory.resolve()
    for line in included.stdout.splitlines():
        if any(line.endswith(ext) for ext in extensions):
            file_path = (directory / line).resolve()
            # Only include files actually under the target directory
            # (git ls-files can return ../paths for files outside cwd)
            if file_path.is_relative_to(directory_resolved):
                files.append(file_path)
                if operation is not None and len(files) % 100 == 0:
                    operation.files_found = len(files)

    ignored_count = sum(1 for line in ignored.stdout.splitlines() if any(line.endswith(ext) for ext in extensions))

    return files, ignored_count


@functools.lru_cache(maxsize=128)
def _find_git_root(directory: str) -> str | None:
    """Find the git root directory containing this path. Cached."""
    try:
        repo = git.Repo(directory, search_parent_directories=True)
        return str(repo.working_dir)
    except git.InvalidGitRepositoryError:
        return None


def _file_hash(path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _deterministic_chunk_id(source_path: str, chunk_index: int, chunk_text: str) -> UUID:
    """Generate deterministic UUID from chunk provenance.

    Same file + chunk index + content = same UUID.
    Enables idempotent re-indexing.
    """
    chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
    key = f'{source_path}|{chunk_index}|{chunk_hash}'
    hash_bytes = hashlib.sha256(key.encode()).digest()[:16]
    return UUID(bytes=hash_bytes)
