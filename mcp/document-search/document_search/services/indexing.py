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
from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import filelock
import git
from local_lib.utils import Timer

from document_search.clients import QdrantClient, create_embedding_client
from document_search.repositories.document_vector import DocumentVectorRepository
from document_search.schemas.chunking import EXTENSION_MAP, Chunk, FileType, get_file_type
from document_search.schemas.config import EmbeddingConfig
from document_search.schemas.indexing import (
    CHUNK_STRATEGY_VERSION,
    DirectoryIndexState,
    FileIndexState,
    FileProcessingError,
    FileTypeStats,
    IndexingResult,
    ProgressCallback,
)
from document_search.schemas.vectors import ClearResult, VectorPoint
from document_search.services.chunking import ChunkingService
from document_search.services.embedding import EmbeddingService
from document_search.services.sparse_embedding import SparseEmbeddingService

__all__ = [
    'IndexingService',
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
                by_file_type=by_file_type,
                elapsed_seconds=round(timer.elapsed(), 3),
                errors=(),
            )

        chunk_ids = [_deterministic_chunk_id(file_key, c.chunk_index, c.text) for c in chunks]

        # Embed (dense + sparse)
        embed_tasks = [self._embedding.embed_text(c.text) for c in chunks]
        responses = await asyncio.gather(*embed_tasks)
        dense = [tuple(r.values) for r in responses]

        texts = [c.text for c in chunks]
        sparse_results = await self._sparse_embedding.embed_batch(texts)
        sparse = [(tuple(i), tuple(v)) for i, v in sparse_results]

        # Build points using same factory as pipeline
        points = [
            VectorPoint.from_chunk(chunk, dense_emb, sparse_indices, sparse_values, chunk_id)
            for chunk, dense_emb, (sparse_indices, sparse_values), chunk_id in zip(chunks, dense, sparse, chunk_ids)
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
                    chunk_ids=tuple(chunk_ids),
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

        if respect_gitignore is not False and git_root:
            all_files, files_ignored = _get_git_files(directory, extensions)
        else:
            all_files, files_ignored = list(_walk_files(directory, extensions)), 0

        files_total = len(all_files)
        logger.info(f'[PIPELINE] Found {files_total} files, {files_ignored} ignored')

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

        # Track files by type and determine which need indexing (single pass)
        scanned_by_type: dict[FileType, int] = {}
        cached_by_type: dict[FileType, int] = {}
        files_to_index: list[Path] = []

        for file_path in all_files:
            ft = get_file_type(file_path)
            if ft is not None:
                scanned_by_type[ft] = scanned_by_type.get(ft, 0) + 1

            if full_reindex or self._needs_indexing(file_path):
                files_to_index.append(file_path)
            elif ft is not None:
                cached_by_type[ft] = cached_by_type.get(ft, 0) + 1

        if not files_to_index:
            # Build by_file_type summary for cached-only result
            by_file_type: dict[FileType, str] = {}
            for ft in scanned_by_type:
                stats = FileTypeStats(
                    scanned=scanned_by_type.get(ft, 0),
                    cached=cached_by_type.get(ft, 0),
                )
                by_file_type[ft] = stats.to_summary()

            return IndexingResult(
                files_scanned=files_total,
                files_ignored=files_ignored,
                files_indexed=0,
                files_cached=sum(cached_by_type.values()),
                files_no_content=0,
                chunks_created=0,
                chunks_deleted=chunks_deleted,
                embeddings_created=0,
                by_file_type=by_file_type,
                elapsed_seconds=round(timer.elapsed(), 3),
                errors=(),
            )

        # PHASE 2: Pipeline processing
        # Create queues connecting stages (exposed as instance attrs for monitoring)
        file_queue: asyncio.Queue[Path] = asyncio.Queue()
        embed_queue: asyncio.Queue[_ChunkedFile] = asyncio.Queue(maxsize=EMBED_QUEUE_SIZE)
        upsert_queue: asyncio.Queue[_EmbeddedFile] = asyncio.Queue(maxsize=UPSERT_QUEUE_SIZE)
        self._file_queue = file_queue
        self._embed_queue = embed_queue
        self._upsert_queue = upsert_queue

        # Shared results collection (exposed for progress monitoring)
        results: dict[str, int | FileProcessingError] = {}  # path -> chunk_count or error
        self._results = results
        results_lock = asyncio.Lock()

        # Queue depth monitor for bottleneck analysis
        monitor = _QueueMonitor(file_queue, embed_queue, upsert_queue, results)

        # Populate file queue
        for path in files_to_index:
            await file_queue.put(path)

        logger.info(
            f'[PIPELINE] Starting workers: {NUM_CHUNK_WORKERS} chunk, '
            f'{NUM_EMBED_WORKERS} embed, {NUM_UPSERT_WORKERS} upsert '
            f'for {len(files_to_index)} files'
        )

        # Start queue monitor
        monitor.start()

        # Start workers for each stage
        chunk_tasks = [
            asyncio.create_task(self._pipeline_chunk_worker(file_queue, embed_queue, results, results_lock))
            for _ in range(min(NUM_CHUNK_WORKERS, len(files_to_index)))
        ]
        embed_tasks = [
            asyncio.create_task(self._pipeline_embed_worker(embed_queue, upsert_queue))
            for _ in range(NUM_EMBED_WORKERS)
        ]
        upsert_tasks = [
            asyncio.create_task(self._pipeline_upsert_worker(upsert_queue, results, results_lock))
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
            # Stop monitor and cancel all workers
            monitor.stop()
            waiter.cancel()
            for task in all_worker_tasks:
                task.cancel()
            await asyncio.gather(waiter, *all_worker_tasks, return_exceptions=True)

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

        # Get index totals
        index_files = len(self._state.files) if self._state else 0
        index_chunks = self._state.total_chunks if self._state else 0

        return IndexingResult(
            files_scanned=files_total,
            files_ignored=files_ignored,
            files_indexed=files_indexed,
            files_cached=files_cached,
            files_no_content=files_no_content,
            chunks_created=chunks_created,
            chunks_deleted=chunks_deleted,
            embeddings_created=chunks_created,
            by_file_type=by_file_type_result,
            index_files=index_files,
            index_chunks=index_chunks,
            elapsed_seconds=round(timer.elapsed(), 3),
            errors=tuple(errors),
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
    ) -> None:
        """Stage 1: Chunk files and push to embed queue.

        Known file errors (encoding, read failures, timeout) are recorded and skipped.
        Infrastructure errors propagate immediately (fail-fast).
        """
        while True:
            file_path = await file_queue.get()

            file_key = str(file_path)
            try:
                chunks = list(
                    await asyncio.wait_for(
                        self._chunking.chunk_file(file_path),
                        timeout=FILE_CHUNK_TIMEOUT_SECONDS,
                    )
                )
                if chunks:
                    chunk_ids = [_deterministic_chunk_id(file_key, c.chunk_index, c.text) for c in chunks]
                    await embed_queue.put(
                        _ChunkedFile(
                            file_path=file_path,
                            file_hash=_file_hash(file_path),
                            file_size=file_path.stat().st_size,
                            chunks=chunks,
                            chunk_ids=chunk_ids,
                        )
                    )
                else:
                    # File produced 0 chunks (content too short) - record in state to avoid re-processing
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
                    logger.debug(f'[CHUNK] {file_path.name}: 0 chunks (content too short)')

            except (TimeoutError, OSError, UnicodeDecodeError) as e:
                # Known file-level errors: record and continue
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
    ) -> None:
        """Stage 2: Generate dense + sparse embeddings and push to upsert queue.

        Accumulates chunks from multiple small files before sending to
        ProcessPoolExecutor for sparse embeddings, reducing IPC overhead.
        Dense embeddings use BatchLoader which handles coalescing internally.

        Fail-fast: exceptions propagate immediately, task_done() only on success.
        """
        # Accumulation settings (for sparse embedding ProcessPool efficiency)
        BATCH_THRESHOLD = 500  # Min texts before sending to ProcessPool
        BATCH_TIMEOUT = 0.05  # 50ms - don't wait forever for small files

        # Accumulators
        accumulated_files: list[_ChunkedFile] = []
        accumulated_texts: list[str] = []
        text_boundaries: list[int] = []  # Cumulative end index for each file

        async def flush_batch() -> None:
            """Generate embeddings and push files to upsert queue."""
            nonlocal accumulated_files, accumulated_texts, text_boundaries

            if not accumulated_texts:
                return

            # Get sparse embeddings (one batch to ProcessPool)
            sparse_results = await self._sparse_embedding.embed_batch(accumulated_texts)

            # Get dense embeddings (BatchLoader coalesces into batches of 100)
            dense_tasks = [self._embedding.embed_text(text) for text in accumulated_texts]
            dense_results = await asyncio.gather(*dense_tasks)

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
                    dense_embeddings=[tuple(r.values) for r in file_dense],
                    sparse_embeddings=[(tuple(indices), tuple(values)) for indices, values in file_sparse],
                )
                await upsert_queue.put(embedded)
                embed_queue.task_done()

                start_idx = end_idx

            # Reset accumulators
            accumulated_files = []
            accumulated_texts = []
            text_boundaries = []

        async def process_single_file(chunked: _ChunkedFile) -> None:
            """Process a single file (used for large files to avoid accumulation)."""
            texts = [c.text for c in chunked.chunks]

            # Get both embeddings
            sparse_results = await self._sparse_embedding.embed_batch(texts)
            dense_tasks = [self._embedding.embed_text(text) for text in texts]
            dense_results = await asyncio.gather(*dense_tasks)

            # Create _EmbeddedFile
            embedded = _EmbeddedFile(
                file_path=chunked.file_path,
                file_hash=chunked.file_hash,
                file_size=chunked.file_size,
                chunks=chunked.chunks,
                chunk_ids=chunked.chunk_ids,
                dense_embeddings=[tuple(r.values) for r in dense_results],
                sparse_embeddings=[(tuple(i), tuple(v)) for i, v in sparse_results],
            )
            await upsert_queue.put(embedded)
            embed_queue.task_done()

        while True:
            try:
                # Try to get item with timeout to allow periodic flushing
                try:
                    chunked = await asyncio.wait_for(
                        embed_queue.get(),
                        timeout=BATCH_TIMEOUT,
                    )
                except TimeoutError:
                    # No items available, flush what we have
                    await flush_batch()
                    continue

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
    ) -> None:
        """Stage 3: Upsert to Qdrant, delete old chunks, update state.

        Fail-fast: exceptions propagate immediately, task_done() only on success.
        """
        while True:
            embedded = await upsert_queue.get()

            file_key = str(embedded.file_path)

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
                        chunk_ids=tuple(embedded.chunk_ids),
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
            async with results_lock:
                results[file_key] = len(embedded.chunks)

            logger.debug(f'[UPSERT] {embedded.file_path.name}: {len(embedded.chunks)} chunks')

            # Only mark done on success
            upsert_queue.task_done()

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


async def create_indexing_service(
    config: EmbeddingConfig,
    collection_name: str,
    *,
    qdrant_url: str = 'http://localhost:6333',
    state_path: Path = DEFAULT_STATE_PATH,
) -> IndexingService:
    """Factory function to create IndexingService with default dependencies.

    Must be called from async context - ensures semaphores are bound correctly.

    Args:
        config: Embedding configuration with provider selection.
        collection_name: Name of the collection to operate on.
        qdrant_url: URL of Qdrant server.
        state_path: Path to persist indexing state.

    Returns:
        Configured IndexingService.
    """
    embedding_client = create_embedding_client(config)
    qdrant_client = QdrantClient(url=qdrant_url)

    chunking_service = await ChunkingService.create()
    embedding_service = EmbeddingService(embedding_client, batch_size=config.batch_size)
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
class _ChunkedFile:
    """Output of chunk stage, input to embed stage."""

    file_path: Path
    file_hash: str
    file_size: int
    chunks: list[Chunk]
    chunk_ids: list[UUID]


@dataclass
class _EmbeddedFile:
    """Output of embed stage, input to upsert stage."""

    file_path: Path
    file_hash: str
    file_size: int
    chunks: list[Chunk]
    chunk_ids: list[UUID]
    dense_embeddings: list[tuple[float, ...]]
    sparse_embeddings: list[tuple[tuple[int, ...], tuple[float, ...]]]


class _QueueMonitor:
    """Monitor queue depths for pipeline bottleneck analysis.

    Logs periodic snapshots showing where work is accumulating:
    - files: remaining files to chunk
    - embed: chunks waiting for embedding (high = chunk keeping up)
    - upsert: embedded chunks waiting for storage (high = embed keeping up)
    """

    def __init__(
        self,
        file_queue: asyncio.Queue[Path],
        embed_queue: asyncio.Queue[_ChunkedFile],
        upsert_queue: asyncio.Queue[_EmbeddedFile],
        results: dict[str, int | FileProcessingError],  # strict_typing_linter.py: mutable-type
        log_interval: float = 5.0,
    ) -> None:
        self._file_queue = file_queue
        self._embed_queue = embed_queue
        self._upsert_queue = upsert_queue
        self._results = results
        self._log_interval = log_interval
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start monitoring."""
        if self._task is None:
            self._task = asyncio.create_task(self._monitor_loop())

    def stop(self) -> None:
        """Stop monitoring."""
        if self._task:
            self._task.cancel()
            self._task = None

    async def _monitor_loop(self) -> None:
        """Log queue depths periodically."""
        while True:
            await asyncio.sleep(self._log_interval)
            done = len(self._results)
            logger.info(
                f'[QUEUES] files={self._file_queue.qsize()} '
                f'embed={self._embed_queue.qsize()}/{EMBED_QUEUE_SIZE} '
                f'upsert={self._upsert_queue.qsize()}/{UPSERT_QUEUE_SIZE} done={done}'
            )


def _walk_files(directory: Path, extensions: Set[str]) -> Iterator[Path]:
    """Walk directory yielding files with matching extensions."""
    for root, _, filenames in os.walk(directory):
        root_path = Path(root).resolve()
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                yield root_path / filename


def _get_git_files(directory: Path, extensions: Set[str]) -> tuple[Sequence[Path], int]:
    """Get non-ignored files and count of ignored files using git ls-files.

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
