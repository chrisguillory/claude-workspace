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
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import more_itertools
from tenacity import retry, stop_after_attempt, wait_exponential

from document_search.clients.gemini import GeminiClient
from document_search.clients.qdrant import QdrantClient
from document_search.repositories.document_vector import DocumentVectorRepository
from document_search.schemas.chunking import EXTENSION_MAP, Chunk, get_file_type
from document_search.schemas.embeddings import EmbedBatchRequest
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

# Gemini embedding dimension
EMBEDDING_DIMENSION = 768

# Default state file location
DEFAULT_STATE_PATH = Path.home() / '.claude-workspace' / 'cache' / 'document_search_index_state.json'

# Batch sizes for efficient processing
EMBEDDING_BATCH_SIZE = 100  # Gemini API limit
FILES_PER_BATCH = 20  # Process files in batches for memory efficiency
QDRANT_UPSERT_BATCH = 1000  # Qdrant can handle large batches

# Concurrency control - follows GeminiClient pattern (semaphore at service level)
FILE_CHUNKING_CONCURRENCY = 8  # Parallel file chunking within a batch


class _BatchResult(TypedDict):
    """Result of processing a batch of files."""

    files_processed: int
    chunks_created: int
    chunks_deleted: int
    tokens_used: int
    errors: list[FileProcessingError]


@dataclass
class _PendingFile:
    """Tracks a file through the indexing pipeline."""

    file_path: Path
    rel_path: str
    file_hash: str = ''
    file_size: int = 0
    chunks: list[Chunk] = field(default_factory=list)
    chunk_ids: list[UUID] = field(default_factory=list)
    embeddings: list[tuple[float, ...]] = field(default_factory=list)


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
        repository: DocumentVectorRepository,
        *,
        state_path: Path = DEFAULT_STATE_PATH,
        batch_size: int = 50,
    ) -> IndexingService:
        """Create indexing service in async context.

        Preferred factory method - ensures semaphore is bound to correct event loop.
        """
        return cls(
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            repository=repository,
            state_path=state_path,
            batch_size=batch_size,
            _file_chunking_semaphore=asyncio.Semaphore(FILE_CHUNKING_CONCURRENCY),
        )

    def __init__(
        self,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        repository: DocumentVectorRepository,
        *,
        state_path: Path = DEFAULT_STATE_PATH,
        batch_size: int = 50,
        _file_chunking_semaphore: asyncio.Semaphore,
    ) -> None:
        """Initialize indexing service. Use create() for async context safety.

        Args:
            chunking_service: Service for splitting files into chunks.
            embedding_service: Service for creating embeddings.
            repository: Repository for vector storage.
            state_path: Path to persist indexing state.
            batch_size: Number of texts per embedding API call.
            _file_chunking_semaphore: Internal - use create() instead.
        """
        self._chunking = chunking_service
        self._embedding = embedding_service
        self._repo = repository
        self._state_path = state_path
        self._batch_size = batch_size
        self._state: DirectoryIndexState | None = None
        self._file_chunking_semaphore = _file_chunking_semaphore

        # Embedding cache: text_hash -> embedding (deduplication within indexing run)
        self._embedding_cache: dict[str, tuple[float, ...]] = {}

    async def index_directory(
        self,
        directory: Path,
        *,
        full_reindex: bool = False,
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
            on_progress: Optional callback for progress updates.

        Returns:
            IndexingResult with counts and any errors.
        """
        if not directory.is_dir():
            raise ValueError(f'Not a directory: {directory}')

        self._repo.ensure_collection(EMBEDDING_DIMENSION)
        self._state = self._load_state(directory)

        start_time = time.time()
        errors: list[FileProcessingError] = []
        files_processed = 0
        files_skipped = 0
        chunks_created = 0
        chunks_deleted = 0
        tokens_used = 0

        # PHASE 1: Scan and identify files (use extension-specific globs for efficiency)
        all_files: list[Path] = []
        for ext in EXTENSION_MAP:
            all_files.extend(f for f in directory.glob(f'**/*{ext}') if f.is_file())
        files_total = len(all_files)

        files_to_index: list[Path] = []
        for file_path in all_files:
            rel_path = str(file_path.relative_to(directory))
            if full_reindex or self._needs_indexing(file_path, rel_path):
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
                    elapsed_seconds=time.time() - start_time,
                )
            )

        # PHASE 2: Process files in batches
        for batch_start in range(0, len(files_to_index), FILES_PER_BATCH):
            batch_files = files_to_index[batch_start : batch_start + FILES_PER_BATCH]

            batch_result = await self._process_file_batch(
                batch_files,
                directory,
                on_progress,
                start_time,
                files_total,
                files_processed,
                files_skipped,
                chunks_created,
            )

            files_processed += batch_result['files_processed']
            chunks_created += batch_result['chunks_created']
            chunks_deleted += batch_result['chunks_deleted']
            tokens_used += batch_result['tokens_used']
            errors.extend(batch_result['errors'])

            # Checkpoint state after each batch
            self._save_state()

            if on_progress:
                on_progress(
                    IndexingProgress(
                        files_scanned=files_total,
                        files_total=files_total,
                        files_processed=files_processed,
                        files_skipped=files_skipped,
                        chunks_created=chunks_created,
                        embeddings_pending=len(files_to_index) - batch_start - len(batch_files),
                        current_phase='storing',
                        errors_so_far=len(errors),
                        elapsed_seconds=time.time() - start_time,
                    )
                )

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
            files_processed=files_processed,
            files_skipped=files_skipped,
            chunks_created=chunks_created,
            chunks_deleted=chunks_deleted,
            embeddings_created=chunks_created,
            tokens_used=tokens_used,
            errors=tuple(errors),
        )

    async def index_file(self, file_path: Path) -> IndexingResult:
        """Index a single file.

        Args:
            file_path: Path to file.

        Returns:
            IndexingResult for the single file.
        """
        if not file_path.is_file():
            raise ValueError(f'Not a file: {file_path}')

        file_type = get_file_type(file_path)
        if file_type is None:
            raise ValueError(f'Unsupported file type: {file_path.suffix}')

        self._repo.ensure_collection(EMBEDDING_DIMENSION)

        # Use parent directory for state tracking
        directory = file_path.parent
        self._state = self._load_state(directory)

        # Process as a batch of 1
        batch_result = await self._process_file_batch(
            files=[file_path],
            directory=directory,
            on_progress=None,
            start_time=time.time(),
            files_total=1,
            files_processed_so_far=0,
            files_skipped=0,
            chunks_created_so_far=0,
        )

        self._save_state()

        return IndexingResult(
            files_scanned=1,
            files_processed=batch_result['files_processed'],
            files_skipped=0,
            chunks_created=batch_result['chunks_created'],
            chunks_deleted=batch_result['chunks_deleted'],
            embeddings_created=batch_result['chunks_created'],
            tokens_used=batch_result['tokens_used'],
            errors=tuple(batch_result['errors']),
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

    async def _chunk_file_with_semaphore(
        self,
        file_path: Path,
        directory: Path,
    ) -> _PendingFile | FileProcessingError | None:
        """Chunk a single file with concurrency control.

        Uses semaphore to limit concurrent file chunking operations,
        following the same pattern as GeminiClient for embeddings.

        Returns:
            _PendingFile on success, FileProcessingError on failure, None if empty.
        """
        rel_path = str(file_path.relative_to(directory))

        async with self._file_chunking_semaphore:
            try:
                chunks = list(await self._chunking.chunk_file(file_path))
                if not chunks:
                    return None

                return _PendingFile(
                    file_path=file_path,
                    rel_path=rel_path,
                    file_hash=_file_hash(file_path),
                    file_size=file_path.stat().st_size,
                    chunks=chunks,
                    chunk_ids=[_deterministic_chunk_id(str(file_path), c.chunk_index, c.text) for c in chunks],
                )
            except Exception as e:
                return FileProcessingError(
                    file_path=str(file_path),
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=not isinstance(e, (PermissionError, IsADirectoryError)),
                )

    async def _process_file_batch(
        self,
        files: Sequence[Path],
        directory: Path,
        on_progress: ProgressCallback | None,
        start_time: float,
        files_total: int,
        files_processed_so_far: int,
        files_skipped: int,
        chunks_created_so_far: int,
    ) -> _BatchResult:
        """Process a batch of files: chunk → embed → upsert → delete old → update state.

        Key design choices:
        - Chunk files in parallel (semaphore controls concurrency)
        - Batch embed across files (fewer API calls)
        - Upsert new chunks BEFORE deleting old (atomicity)
        - Update state after successful upsert (crash recovery)

        Returns:
            Dict with files_processed, chunks_created, chunks_deleted, tokens_used, errors.
        """
        errors: list[FileProcessingError] = []
        pending_files: list[_PendingFile] = []

        # STEP 1: Chunk all files in batch (PARALLEL with semaphore)
        if on_progress:
            on_progress(
                IndexingProgress(
                    files_scanned=files_total,
                    files_total=files_total,
                    files_processed=files_processed_so_far,
                    files_skipped=files_skipped,
                    chunks_created=chunks_created_so_far,
                    embeddings_pending=len(files),
                    current_phase='chunking',
                    elapsed_seconds=time.time() - start_time,
                )
            )

        # Fire all file chunking tasks concurrently - semaphore controls actual parallelism
        results = await asyncio.gather(
            *[self._chunk_file_with_semaphore(file_path, directory) for file_path in files],
            return_exceptions=True,
        )

        # Collect results: separate pending files from errors
        for result in results:
            if result is None:
                continue  # Empty file, skip
            if isinstance(result, FileProcessingError):
                errors.append(result)
            elif isinstance(result, Exception):
                # Unexpected exception from gather (shouldn't happen with return_exceptions=True)
                errors.append(
                    FileProcessingError(
                        file_path='unknown',
                        error_type=type(result).__name__,
                        message=str(result),
                        recoverable=True,
                    )
                )
            elif isinstance(result, _PendingFile):
                pending_files.append(result)

        if not pending_files:
            return {
                'files_processed': 0,
                'chunks_created': 0,
                'chunks_deleted': 0,
                'tokens_used': 0,
                'errors': errors,
            }

        # STEP 2: Flatten chunks for batch embedding
        all_chunk_refs: list[tuple[_PendingFile, int]] = []  # (file, chunk_index)
        all_texts: list[str] = []

        for pf in pending_files:
            for idx, chunk in enumerate(pf.chunks):
                all_chunk_refs.append((pf, idx))
                all_texts.append(chunk.text)

        # STEP 3: Batch embed (fewer API calls)
        if on_progress:
            on_progress(
                IndexingProgress(
                    files_scanned=files_total,
                    files_total=files_total,
                    files_processed=files_processed_so_far,
                    files_skipped=files_skipped,
                    chunks_created=chunks_created_so_far,
                    embeddings_pending=len(all_texts),
                    current_phase='embedding',
                    elapsed_seconds=time.time() - start_time,
                )
            )

        # Fire all batches concurrently - client handles rate limiting
        all_embeddings, tokens_used, batch_times = await self._embed_all_concurrent(all_texts)

        # Timing available in batch_times for analysis if needed
        # avg_time = sum(batch_times) / len(batch_times) if batch_times else 0
        # wall_time ≈ max(batch_times) due to concurrent execution

        # Distribute embeddings back to pending files
        for (pf, idx), embedding in zip(all_chunk_refs, all_embeddings):
            pf.embeddings.append(embedding)

        # STEP 4: Create points and batch upsert
        all_points: list[VectorPoint] = []
        for pf in pending_files:
            for chunk, embedding, chunk_id in zip(pf.chunks, pf.embeddings, pf.chunk_ids):
                point = VectorPoint.from_chunk(chunk, embedding, chunk_id)
                all_points.append(point)

        # Upsert in batches to avoid timeout
        for batch_start in range(0, len(all_points), QDRANT_UPSERT_BATCH):
            batch_points = all_points[batch_start : batch_start + QDRANT_UPSERT_BATCH]
            self._repo.upsert(batch_points)

        # STEP 5: Delete OLD chunks by ID AFTER successful upsert (atomicity!)
        # Key: Delete by specific IDs, not source_path, to avoid deleting new chunks.
        # Chunk IDs are deterministic (source_path + index + content_hash), so
        # changed content = different IDs. We delete only the OLD IDs from state.
        #
        # Note: chunks_deleted is the EXPECTED count from state, not actual deletion
        # count from Qdrant. Some chunks may already be deleted or never existed.
        # State is authoritative for tracking purposes.
        chunks_deleted = 0
        old_chunk_ids: list[UUID] = []
        for pf in pending_files:
            old_state = self._state.files.get(pf.rel_path) if self._state else None
            if old_state:
                chunks_deleted += old_state.chunk_count
                old_chunk_ids.extend(old_state.chunk_ids)

        if old_chunk_ids:
            self._repo.delete(old_chunk_ids)

        # STEP 6: Update state for all processed files
        if self._state is not None:
            new_files = dict(self._state.files)
            for pf in pending_files:
                new_files[pf.rel_path] = FileIndexState(
                    file_path=str(pf.file_path),
                    file_hash=pf.file_hash,
                    file_size=pf.file_size,
                    chunk_count=len(pf.chunks),
                    chunk_ids=tuple(pf.chunk_ids),
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

        return {
            'files_processed': len(pending_files),
            'chunks_created': len(all_points),
            'chunks_deleted': chunks_deleted,
            'tokens_used': tokens_used,
            'errors': errors,
        }

    def _load_state(self, directory: Path) -> DirectoryIndexState:
        """Load or create indexing state for directory."""
        if self._state_path.exists():
            data = json.loads(self._state_path.read_text())
            state = DirectoryIndexState.model_validate(data)
            if state.directory_path == str(directory):
                return state

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

    def _needs_indexing(self, path: Path, rel_path: str) -> bool:
        """Check if file needs (re)indexing based on content hash."""
        if self._state is None:
            return True

        file_state = self._state.files.get(rel_path)
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

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _embed_batch_with_retry(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed texts with exponential backoff on rate limit."""
        request = EmbedBatchRequest(texts=list(texts))
        response = await self._embedding.embed_batch(request)
        return [emb.values for emb in response.embeddings]

    async def _embed_all_concurrent(
        self,
        all_texts: Sequence[str],
    ) -> tuple[list[tuple[float, ...]], int, list[float]]:
        """Embed all texts concurrently with deduplication.

        Skips embedding for texts already in cache (same content = same embedding).
        Fires all batches at once - client handles rate limiting internally.

        Args:
            all_texts: All texts to embed.

        Returns:
            Tuple of (embeddings, tokens_used, batch_times_seconds).
        """
        # Deduplicate: hash each text, identify which need embedding
        text_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in all_texts]
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, (text, h) in enumerate(zip(all_texts, text_hashes)):
            if h not in self._embedding_cache:
                uncached_indices.append(i)
                uncached_texts.append(text)

        cache_hits = len(all_texts) - len(uncached_texts)
        if cache_hits > 0:
            # Log deduplication stats (visible in debug)
            pass  # Could add logging here if desired

        # Embed only uncached texts
        batch_times: list[float] = []
        new_embeddings: list[tuple[float, ...]] = []

        if uncached_texts:
            batches = list(more_itertools.chunked(uncached_texts, EMBEDDING_BATCH_SIZE))

            async def embed_batch_timed(batch: Sequence[str]) -> list[tuple[float, ...]]:
                """Embed single batch with timing."""
                t0 = time.perf_counter()
                result = await self._embed_batch_with_retry(list(batch))
                batch_times.append(time.perf_counter() - t0)
                return result

            # Fire all batches concurrently - client handles rate limiting
            results = await asyncio.gather(*[embed_batch_timed(batch) for batch in batches])

            # Flatten and cache new embeddings
            for batch_result in results:
                new_embeddings.extend(batch_result)

            # Verify embedding count matches (guards against API returning wrong count)
            if len(new_embeddings) != len(uncached_indices):
                raise RuntimeError(
                    f'Embedding count mismatch: expected {len(uncached_indices)}, '
                    f'got {len(new_embeddings)}. Check Gemini API response.'
                )

            # Update cache with new embeddings
            for idx, emb in zip(uncached_indices, new_embeddings, strict=True):
                self._embedding_cache[text_hashes[idx]] = emb

        # Build final result in original order (all now in cache)
        all_embeddings: list[tuple[float, ...]] = [self._embedding_cache[h] for h in text_hashes]

        # Calculate tokens only for texts we actually embedded
        tokens_used = sum(len(t) for t in uncached_texts) // 3

        return all_embeddings, tokens_used, batch_times


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
    repository = DocumentVectorRepository(qdrant_client)

    return await IndexingService.create(
        chunking_service=chunking_service,
        embedding_service=embedding_service,
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
