#!/usr/bin/env python3
"""Test pipeline indexing architecture."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

LOG_FILE = Path('/tmp/pipeline_test.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stderr),
    ],
)

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

sys.path.insert(0, '/Users/chris/claude-workspace/mcp/document-search')

from document_search.clients.gemini import GeminiClient  # noqa: E402
from document_search.clients.qdrant import QdrantClient  # noqa: E402
from document_search.repositories.document_vector import DocumentVectorRepository  # noqa: E402
from document_search.services.chunking import ChunkingService  # noqa: E402
from document_search.services.embedding import EmbeddingService  # noqa: E402
from document_search.services.embedding_batch_loader import EmbeddingBatchLoader  # noqa: E402
from document_search.services.indexing import IndexingService  # noqa: E402
from document_search.services.sparse_embedding import SparseEmbeddingService  # noqa: E402


async def main() -> None:
    directory = Path.cwd()

    print('=' * 60)
    print('PIPELINE ARCHITECTURE TEST')
    print('=' * 60)
    print(f'Directory: {directory}')
    print(f'Log file: {LOG_FILE}')
    print()
    print('Pipeline configuration:')
    print('  - Chunk workers: 16')
    print('  - Embed workers: 64')
    print('  - Upsert workers: 16')
    print('  - Qdrant max concurrent: 6')
    print()
    print('Results so far:')
    print('  - 4 qdrant:  ~11 files/sec (latency 464ms, queue saturated)')
    print('  - 8 qdrant:  16.3 files/sec (latency 508ms, baseline)')
    print('  - 12 qdrant: 14.4 files/sec (latency 788ms)')
    print('  - 16 qdrant: 13.3 files/sec (latency 940ms)')
    print('=' * 60)
    print()

    logger.info(f'Starting pipeline indexing of {directory}')

    # Create clients
    gemini_client = GeminiClient()
    qdrant_client = QdrantClient()

    # Create services
    chunking_service = await ChunkingService.create()
    embedding_service = EmbeddingService(gemini_client)
    embedding_loader = EmbeddingBatchLoader(embedding_service)
    sparse_service = SparseEmbeddingService()
    repository = DocumentVectorRepository(qdrant_client)

    indexing_service = IndexingService(
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        sparse_embedding_service=sparse_service,
        batch_loader=embedding_loader,
        repository=repository,
    )

    start_time = time.perf_counter()

    try:
        result = await indexing_service.index_directory(
            directory,
            full_reindex=True,
            respect_gitignore=False,
        )

        elapsed = time.perf_counter() - start_time

        print()
        print('=' * 60)
        print('RESULTS')
        print('=' * 60)
        print(f'Files scanned: {result.files_scanned}')
        print(f'Files processed: {result.files_processed}')
        print(f'Files skipped: {result.files_skipped}')
        print(f'Chunks created: {result.chunks_created:,}')
        print(f'Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)')
        print()

        if result.files_processed > 0:
            files_per_sec = result.files_processed / elapsed
            chunks_per_sec = result.chunks_created / elapsed
            print(f'Throughput: {files_per_sec:.1f} files/sec')
            print(f'Chunk throughput: {chunks_per_sec:.0f} chunks/sec')
            print()
            print('vs Baseline:')
            print(f'  Files/sec: {files_per_sec:.1f} vs 16.3 ({files_per_sec / 16.3 * 100:.0f}%)')

        print('=' * 60)

    finally:
        if hasattr(gemini_client, '_tracker'):
            gemini_client._tracker.stop()
            print(f'GEMINI FINAL: {gemini_client._tracker.stats}')
        if hasattr(qdrant_client, '_tracker'):
            qdrant_client._tracker.stop()
            print(f'QDRANT FINAL: {qdrant_client._tracker.stats}')


if __name__ == '__main__':
    asyncio.run(main())
