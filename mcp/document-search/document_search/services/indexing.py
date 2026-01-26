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
import hashlib
import json
import logging
import subprocess
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from local_lib.utils import Timer

from document_search.clients.gemini import GeminiClient
from document_search.clients.qdrant import QdrantClient
from document_search.repositories.document_vector import DocumentVectorRepository
from document_search.schemas.chunking import EXTENSION_MAP, Chunk
from document_search.schemas.indexing import (
    CHUNK_STRATEGY_VERSION,
    DirectoryIndexState,
    FileIndexState,
    FileProcessingError,
    IndexingResult,
    ProgressCallback,
)
from document_search.schemas.vectors import VectorPoint
from document_search.services.chunking import ChunkingService
from document_search.services.embedding import EmbeddingService
from document_search.services.embedding_batch_loader import EmbeddingBatchLoader
from document_search.services.sparse_embedding import SparseEmbeddingService

__all__ = [
    'IndexingService',
    'create_indexing_service',
]

logger = logging.getLogger(__name__)

# Gemini embedding dimension
EMBEDDING_DIMENSION = 768

# Default state file location
DEFAULT_STATE_PATH = Path.home() / '.claude-workspace' / 'cache' / 'document_search_index_state.json'

# Pipeline worker configuration - each stage is independently tunable
NUM_CHUNK_WORKERS = 8  # CPU-bound, limited by disk I/O
NUM_EMBED_WORKERS = 64  # I/O-bound, feeds Gemini API
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
        coalesce_delay: float = 0.01,
    ) -> IndexingService:
        """Create indexing service in async context.

        Args:
            chunking_service: Service for splitting files into chunks.
            embedding_service: Service for creating dense embeddings.
            sparse_embedding_service: Service for creating BM25 sparse embeddings.
            repository: Repository for vector storage.
            state_path: Path to persist indexing state.
            coalesce_delay: Seconds to wait for embedding request coalescing (default 10ms).
        """
        # Create batch loader that coalesces embedding requests across workers
        batch_loader = EmbeddingBatchLoader(embedding_service, coalesce_delay=coalesce_delay)

        return cls(
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            sparse_embedding_service=sparse_embedding_service,
            batch_loader=batch_loader,
            repository=repository,
            state_path=state_path,
        )

    def __init__(
        self,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
        batch_loader: EmbeddingBatchLoader,
        repository: DocumentVectorRepository,
        *,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        """Initialize indexing service.

        Args:
            chunking_service: Service for splitting files into chunks.
            embedding_service: Service for creating dense embeddings.
            sparse_embedding_service: Service for creating BM25 sparse embeddings.
            batch_loader: Batch loader for coalescing embedding requests.
            repository: Repository for vector storage.
            state_path: Path to persist indexing state.
        """
        self._chunking = chunking_service
        self._embedding = embedding_service
        self._sparse_embedding = sparse_embedding_service
        self._batch_loader = batch_loader
        self._repo = repository
        self._state_path = state_path
        self._state: DirectoryIndexState | None = None
        self._state_lock: asyncio.Lock | None = None  # Initialized in async context

    async def get_index_stats(self) -> Mapping[str, int | str]:
        """Get current index statistics."""
        info = await self._repo.get_collection_info()
        if info is None:
            return {'status': 'not_initialized'}

        return {
            'status': info.status,
            'collection': info.name,
            'vector_dimension': info.vector_dimension,
            'points_count': info.points_count,
        }

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

        # PHASE 1: Scan and identify files
        all_files: list[Path] = []
        for ext in EXTENSION_MAP:
            all_files.extend(f for f in directory.glob(f'**/*{ext}') if f.is_file())

        files_found = len(all_files)

        # Filter git-ignored files
        files_ignored = 0
        if respect_gitignore is not False:
            ignored_files = _get_git_ignored_files(all_files, directory, strict=(respect_gitignore is True))
            all_files = [f for f in all_files if f not in ignored_files]
            files_ignored = len(ignored_files)

        files_total = len(all_files)
        logger.info(f'[PIPELINE] Scanned {files_found} files, {files_ignored} git-ignored, {files_total} to consider')

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

        # Determine which files need indexing
        files_to_index: list[Path] = []
        files_skipped = 0
        for file_path in all_files:
            if full_reindex or self._needs_indexing(file_path):
                files_to_index.append(file_path)
            else:
                files_skipped += 1

        if not files_to_index:
            return IndexingResult(
                files_scanned=files_total,
                files_ignored=files_ignored,
                files_processed=0,
                files_skipped=files_skipped,
                chunks_created=0,
                chunks_deleted=chunks_deleted,
                embeddings_created=0,
                elapsed_seconds=round(timer.elapsed(), 3),
                errors=(),
            )

        # PHASE 2: Pipeline processing
        # Create queues connecting stages
        file_queue: asyncio.Queue[Path] = asyncio.Queue()
        embed_queue: asyncio.Queue[_ChunkedFile] = asyncio.Queue(maxsize=EMBED_QUEUE_SIZE)
        upsert_queue: asyncio.Queue[_EmbeddedFile] = asyncio.Queue(maxsize=UPSERT_QUEUE_SIZE)

        # Shared results collection
        results: dict[str, int | FileProcessingError] = {}  # path -> chunk_count or error
        results_lock = asyncio.Lock()

        # Populate file queue
        for path in files_to_index:
            await file_queue.put(path)

        logger.info(
            f'[PIPELINE] Starting workers: {NUM_CHUNK_WORKERS} chunk, '
            f'{NUM_EMBED_WORKERS} embed, {NUM_UPSERT_WORKERS} upsert '
            f'for {len(files_to_index)} files'
        )

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
            # Cancel all workers and waiter
            waiter.cancel()
            for task in all_worker_tasks:
                task.cancel()
            await asyncio.gather(waiter, *all_worker_tasks, return_exceptions=True)

        # Collect results
        errors: list[FileProcessingError] = []
        files_processed = 0
        chunks_created = 0

        for result in results.values():
            if isinstance(result, FileProcessingError):
                errors.append(result)
            else:
                files_processed += 1
                chunks_created += result

        # Save state
        self._save_state()

        previous_chunks = self._state.total_chunks if self._state else 0
        self._state = DirectoryIndexState(
            directory_path=str(directory),
            files=self._state.files if self._state else {},
            last_full_scan=datetime.now(UTC),
            total_files=files_processed + files_skipped,
            total_chunks=previous_chunks + chunks_created - chunks_deleted,
        )
        self._save_state()

        return IndexingResult(
            files_scanned=files_total,
            files_ignored=files_ignored,
            files_processed=files_processed,
            files_skipped=files_skipped,
            chunks_created=chunks_created,
            chunks_deleted=chunks_deleted,
            embeddings_created=chunks_created,
            elapsed_seconds=round(timer.elapsed(), 3),
            errors=tuple(errors),
        )

    def shutdown(self) -> None:
        """Shutdown services and release resources.

        Shuts down the ProcessPoolExecutor used for PDF chunking.
        Should be called when the service is no longer needed.
        """
        self._chunking.shutdown()

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
        """Persist indexing state to disk."""
        if self._state is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        # Use model_dump with mode='json' for datetime serialization
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
            try:
                file_path = await file_queue.get()
            except asyncio.CancelledError:
                return

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

            except (TimeoutError, OSError, UnicodeDecodeError) as e:
                # Known file-level errors: record and continue
                async with results_lock:
                    results[file_key] = FileProcessingError(
                        file_path=file_key,
                        error_type=type(e).__name__,
                        message=str(e),
                        recoverable=True,
                    )
                logger.warning(f'[CHUNK] Skipping {file_path.name}: {type(e).__name__}: {e}')

            # Mark done whether success or known error (unknown errors propagate)
            file_queue.task_done()

    async def _pipeline_embed_worker(
        self,
        embed_queue: asyncio.Queue[_ChunkedFile],
        upsert_queue: asyncio.Queue[_EmbeddedFile],
    ) -> None:
        """Stage 2: Embed chunks (Gemini + sparse) and push to upsert queue.

        Fail-fast: exceptions propagate immediately, task_done() only on success.
        """
        while True:
            try:
                chunked = await embed_queue.get()
            except asyncio.CancelledError:
                return

            # No try/finally - exceptions propagate, triggering fail-fast
            # Dense embeddings via batch loader (Gemini API)
            embed_tasks = [self._batch_loader.embed(c.text) for c in chunked.chunks]
            responses = await asyncio.gather(*embed_tasks)
            dense = [tuple(r.values) for r in responses]

            # Sparse embeddings (local BM25, fast)
            texts = [c.text for c in chunked.chunks]
            sparse_results = self._sparse_embedding.embed_batch(texts)
            sparse = [(tuple(i), tuple(v)) for i, v in sparse_results]

            await upsert_queue.put(
                _EmbeddedFile(
                    file_path=chunked.file_path,
                    file_hash=chunked.file_hash,
                    file_size=chunked.file_size,
                    chunks=chunked.chunks,
                    chunk_ids=chunked.chunk_ids,
                    dense_embeddings=dense,
                    sparse_embeddings=sparse,
                )
            )

            # Only mark done on success
            embed_queue.task_done()

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
            try:
                embedded = await upsert_queue.get()
            except asyncio.CancelledError:
                return

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

            # Get and delete old chunks
            if self._state_lock is None:
                raise RuntimeError('State lock not initialized')

            async with self._state_lock:
                old_state = self._state.files.get(file_key) if self._state else None

            if old_state:
                await self._repo.delete(list(old_state.chunk_ids))

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


async def create_indexing_service(
    *,
    qdrant_url: str = 'http://localhost:6333',
    state_path: Path = DEFAULT_STATE_PATH,
) -> IndexingService:
    """Factory function to create IndexingService with default dependencies.

    Must be called from async context - ensures semaphores are bound correctly.

    Args:
        qdrant_url: URL of Qdrant server.
        state_path: Path to persist indexing state.

    Returns:
        Configured IndexingService.
    """
    gemini_client = GeminiClient()
    qdrant_client = QdrantClient(url=qdrant_url)

    chunking_service = await ChunkingService.create()
    embedding_service = EmbeddingService(gemini_client)
    sparse_embedding_service = SparseEmbeddingService()
    repository = DocumentVectorRepository(qdrant_client)

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


def _get_git_ignored_files(
    file_paths: Sequence[Path],
    directory: Path,
    *,
    strict: bool = False,
) -> Set[Path]:
    """Use git check-ignore to identify ignored files.

    Respects all gitignore rules: root, nested (e.g., .mypy_cache/.gitignore),
    .git/info/exclude, and global gitignore.

    Args:
        file_paths: List of file paths to check.
        directory: Working directory for git command.
        strict: If True, raise on non-git repos. If False, return empty set.

    Returns:
        Set of paths that are git-ignored.

    Raises:
        FileNotFoundError: If git is not installed (always raised).
        subprocess.TimeoutExpired: If git check-ignore takes too long (always raised).
        RuntimeError: If directory is not a git repository (only if strict=True).
    """
    if not file_paths:
        return set()

    result = subprocess.run(
        ['git', 'check-ignore', '--stdin'],
        input='\n'.join(str(p) for p in file_paths),
        capture_output=True,
        text=True,
        cwd=directory,
        timeout=30,
    )

    # Exit codes: 0 = some ignored, 1 = none ignored, 128 = not a git repo
    if result.returncode == 128:
        if strict:
            raise RuntimeError(f'Not a git repository: {directory}')
        return set()

    return {Path(line) for line in result.stdout.splitlines() if line}


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
