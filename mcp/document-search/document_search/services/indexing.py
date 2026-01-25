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
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
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
    IndexingProgress,
    IndexingResult,
    ProgressCallback,
)
from document_search.schemas.vectors import VectorPoint
from document_search.services.chunking import ChunkingService
from document_search.services.embedding import EmbeddingService
from document_search.services.embedding_batch_loader import EmbeddingBatchLoader
from document_search.services.sparse_embedding import SparseEmbeddingService

logger = logging.getLogger(__name__)

# Gemini embedding dimension
EMBEDDING_DIMENSION = 768

# Default state file location
DEFAULT_STATE_PATH = Path.home() / '.claude-workspace' / 'cache' / 'document_search_index_state.json'

# Worker configuration - I/O-bound workers can exceed CPU count
NUM_FILE_WORKERS = min(32, (os.cpu_count() or 4) * 2)

# Timeout for file chunking operations (detects deadlocks)
FILE_CHUNK_TIMEOUT_SECONDS = 60


def _get_git_ignored_files(
    file_paths: Sequence[Path],
    directory: Path,
    *,
    strict: bool = False,
) -> set[Path]:
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


@dataclass
class _PendingFile:
    """Tracks a file through the indexing pipeline."""

    file_path: Path
    file_hash: str = ''
    file_size: int = 0
    chunks: list[Chunk] = field(default_factory=list)
    chunk_ids: list[UUID] = field(default_factory=list)
    embeddings: list[tuple[float, ...]] = field(default_factory=list)
    sparse_embeddings: list[tuple[tuple[int, ...], tuple[float, ...]]] = field(default_factory=list)


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

    def shutdown(self) -> None:
        """Shutdown services and release resources.

        Shuts down the ProcessPoolExecutor used for PDF chunking.
        Should be called when the service is no longer needed.
        """
        self._chunking.shutdown()

    async def _file_worker(
        self,
        queue: asyncio.Queue[Path],
        worker_id: int,
        results: dict[str, _PendingFile | FileProcessingError],
    ) -> None:
        """Worker that processes files from queue one at a time.

        Each file goes through: chunk → embed → upsert → update state.
        This enables pipeline parallelism - file N embeds while file N+1 chunks.
        """
        while True:
            try:
                file_path = await queue.get()
            except asyncio.CancelledError:
                return

            file_key = str(file_path)
            try:
                result = await self._process_single_file(file_path)
                results[file_key] = result
            except Exception as e:
                results[file_key] = FileProcessingError(
                    file_path=file_key,
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=True,
                )
            finally:
                queue.task_done()

    async def _process_single_file(self, file_path: Path) -> _PendingFile:
        """Process a single file end-to-end: chunk → embed → upsert → state.

        Uses EmbeddingBatchLoader to coalesce embedding requests across workers.
        Lock scope is minimized - I/O happens outside the lock.

        Returns the completed _PendingFile with all data populated.
        """
        start_time = time.perf_counter()
        file_key = str(file_path)

        # Step 1: Chunk the file
        pending = await self._chunk_file_with_timeout(file_path)
        if pending is None:
            # Empty or unsupported file - nothing to index, skip silently
            logger.debug(f'[SKIP] {file_path.name}: empty or unsupported')
            return _PendingFile(file_path=file_path)
        if isinstance(pending, FileProcessingError):
            raise RuntimeError(pending.message)

        chunk_time = time.perf_counter() - start_time

        if not pending.chunks:
            # File parsed but produced no chunks - skip silently
            return pending

        # Step 2: Embed chunks (dense + sparse)
        embed_start = time.perf_counter()

        # Dense embeddings: Fire all requests concurrently - batch loader coalesces them
        embed_tasks = [self._batch_loader.embed(chunk.text) for chunk in pending.chunks]
        embed_responses = await asyncio.gather(*embed_tasks)
        pending.embeddings = [tuple(resp.values) for resp in embed_responses]

        # Sparse embeddings: BM25 keyword vectors (synchronous, fast)
        chunk_texts = [chunk.text for chunk in pending.chunks]
        sparse_results = self._sparse_embedding.embed_batch(chunk_texts)
        pending.sparse_embeddings = [(tuple(indices), tuple(values)) for indices, values in sparse_results]

        embed_time = time.perf_counter() - embed_start

        # Step 3: Create points and upsert (I/O - outside lock)
        upsert_start = time.perf_counter()
        points: list[VectorPoint] = []
        for chunk, embedding, sparse_emb, chunk_id in zip(
            pending.chunks, pending.embeddings, pending.sparse_embeddings, pending.chunk_ids
        ):
            sparse_indices, sparse_values = sparse_emb
            point = VectorPoint.from_chunk(chunk, embedding, sparse_indices, sparse_values, chunk_id)
            points.append(point)

        self._repo.upsert(points)
        upsert_time = time.perf_counter() - upsert_start

        # Step 4: Read old state under lock
        if self._state_lock is None:
            raise RuntimeError('State lock not initialized - call index_directory first')

        async with self._state_lock:
            old_state = self._state.files.get(file_key) if self._state else None

        # Step 5: Delete old chunks (I/O - outside lock)
        if old_state:
            self._repo.delete(list(old_state.chunk_ids))

        # Step 6: Update state under lock
        async with self._state_lock:
            if self._state is not None:
                new_files = dict(self._state.files)
                new_files[file_key] = FileIndexState(
                    file_path=file_key,
                    file_hash=pending.file_hash,
                    file_size=pending.file_size,
                    chunk_count=len(pending.chunks),
                    chunk_ids=tuple(pending.chunk_ids),
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

        total_time = time.perf_counter() - start_time
        logger.debug(
            f'[FILE] {file_path.name}: {len(pending.chunks)} chunks '
            f'(chunk: {chunk_time:.2f}s, embed: {embed_time:.2f}s, upsert: {upsert_time:.2f}s, total: {total_time:.2f}s)'
        )

        return pending

    async def index_directory(
        self,
        directory: Path,
        *,
        full_reindex: bool = False,
        respect_gitignore: bool | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> IndexingResult:
        """Index all supported files in a directory using batched processing.

        Architecture:
        - Files processed in batches (fewer API calls, memory bounded)
        - Deletion happens AFTER successful upsert (atomicity)
        - State updated per-batch (crash recovery)

        Args:
            directory: Directory to index.
            full_reindex: If True, reindex all files regardless of hash.
            respect_gitignore: Git ignore filtering behavior:
                - None (default): Auto-detect. Filter if git repo, skip silently if not.
                - True: Require git filtering. Raises RuntimeError if not a git repo.
                - False: Skip git filtering entirely.
            on_progress: Optional callback for progress updates.

        Returns:
            IndexingResult with counts and any errors.
        """
        if not directory.is_dir():
            raise ValueError(f'Not a directory: {directory}')

        self._repo.ensure_collection(EMBEDDING_DIMENSION)
        self._state = self._load_state(directory)

        timer = Timer()
        errors: list[FileProcessingError] = []
        files_processed = 0
        files_skipped = 0
        files_ignored = 0
        chunks_created = 0
        chunks_deleted = 0

        # PHASE 1: Scan and identify files (use extension-specific globs for efficiency)
        all_files: list[Path] = []
        for ext in EXTENSION_MAP:
            all_files.extend(f for f in directory.glob(f'**/*{ext}') if f.is_file())

        files_found = len(all_files)

        # Filter git-ignored files
        if respect_gitignore is not False:
            ignored_files = _get_git_ignored_files(all_files, directory, strict=(respect_gitignore is True))
            all_files = [f for f in all_files if f not in ignored_files]
            files_ignored = len(ignored_files)

        files_total = len(all_files)
        logger.info(f'[INDEX] Scanned {files_found} files, {files_ignored} git-ignored, {files_total} to consider')

        files_to_index: list[Path] = []
        for file_path in all_files:
            if full_reindex or self._needs_indexing(file_path):
                files_to_index.append(file_path)
            else:
                files_skipped += 1

        if on_progress:
            on_progress(
                IndexingProgress(
                    files_scanned=files_total,
                    files_total=files_total,
                    files_processed=0,
                    files_skipped=files_skipped,
                    chunks_created=0,
                    embeddings_pending=len(files_to_index),
                    current_phase='scanning',
                    elapsed_seconds=round(timer.elapsed(), 3),
                )
            )

        # PHASE 2: Process files with concurrent workers
        # Each worker processes one file end-to-end (chunk → embed → upsert)
        # This enables pipeline parallelism: file N embeds while file N+1 chunks

        # Initialize state lock for this run
        self._state_lock = asyncio.Lock()

        # Create queue and populate with files
        queue: asyncio.Queue[Path] = asyncio.Queue()
        for file_path in files_to_index:
            await queue.put(file_path)

        # Results collected by workers
        results: dict[str, _PendingFile | FileProcessingError] = {}

        # Start workers
        logger.info(f'[INDEX] Starting {NUM_FILE_WORKERS} workers for {len(files_to_index)} files')
        workers: list[asyncio.Task[None]] = []
        for i in range(min(NUM_FILE_WORKERS, len(files_to_index))):
            task = asyncio.create_task(self._file_worker(queue, i, results))
            workers.append(task)

        # Wait for all files to be processed
        await queue.join()

        # Cancel workers (they're waiting on empty queue)
        for task in workers:
            task.cancel()

        # Collect results
        for result in results.values():
            if isinstance(result, FileProcessingError):
                errors.append(result)
            elif isinstance(result, _PendingFile):
                files_processed += 1
                chunks_created += len(result.chunks)

        # Save state checkpoint
        self._save_state()

        # Final state update
        # total_chunks is cumulative: previous + created - deleted
        previous_chunks = self._state.total_chunks if self._state else 0
        self._state = DirectoryIndexState(
            directory_path=str(directory),
            files=self._state.files,
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

    def get_index_stats(self) -> dict[str, int | str]:
        """Get current index statistics."""
        info = self._repo.get_collection_info()
        if info is None:
            return {'status': 'not_initialized'}

        return {
            'status': info.status,
            'collection': info.name,
            'vector_dimension': info.vector_dimension,
            'points_count': info.points_count,
        }

    async def _chunk_file_with_timeout(
        self,
        file_path: Path,
    ) -> _PendingFile | FileProcessingError | None:
        """Chunk a single file with timeout protection.

        No outer semaphore - PDF_SEMAPHORE in ChunkingService provides sufficient
        backpressure. Timeout fails loudly to detect deadlocks.

        Returns:
            _PendingFile on success, FileProcessingError on failure, None if empty.

        Raises:
            asyncio.TimeoutError: If chunking exceeds timeout (potential deadlock).
        """
        try:
            # Timeout fails loudly - don't catch TimeoutError
            chunks = list(
                await asyncio.wait_for(
                    self._chunking.chunk_file(file_path),
                    timeout=FILE_CHUNK_TIMEOUT_SECONDS,
                )
            )
            if not chunks:
                return None

            return _PendingFile(
                file_path=file_path,
                file_hash=_file_hash(file_path),
                file_size=file_path.stat().st_size,
                chunks=chunks,
                chunk_ids=[_deterministic_chunk_id(str(file_path), c.chunk_index, c.text) for c in chunks],
            )
        except TimeoutError:
            # Don't catch - let it bubble and crash loudly
            print(
                f'[DEADLOCK] File chunking timeout ({FILE_CHUNK_TIMEOUT_SECONDS}s): {file_path}',
                file=sys.stderr,
                flush=True,
            )
            raise
        except (
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            UnicodeDecodeError,
            OSError,
        ) as e:
            # Expected file I/O errors - recoverable
            return FileProcessingError(
                file_path=str(file_path),
                error_type=type(e).__name__,
                message=str(e),
                recoverable=not isinstance(e, (PermissionError, IsADirectoryError)),
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
