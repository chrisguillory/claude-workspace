"""Document Search MCP Server.

Semantic search over local documents using Gemini embeddings and Qdrant.

Tools:
- index_directory: Index documents in a directory for semantic search
- search_documents: Search indexed documents by natural language query
- get_index_stats: Get statistics about the current index
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import typing
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import mcp.server.fastmcp
import mcp.types
from local_lib.utils import DualLogger

from document_search.clients.gemini import GeminiClient
from document_search.clients.qdrant import QdrantClient
from document_search.repositories.document_vector import DocumentVectorRepository
from document_search.schemas.chunking import FileType
from document_search.schemas.embeddings import EmbedRequest
from document_search.schemas.indexing import IndexingProgress, IndexingResult
from document_search.schemas.vectors import SearchQuery, SearchResult
from document_search.services.chunking import ChunkingService
from document_search.services.embedding import EmbeddingService
from document_search.services.indexing import IndexingService
from document_search.services.reranker import RerankerService
from document_search.services.sparse_embedding import SparseEmbeddingService

# Embedding dimension for Gemini
EMBEDDING_DIMENSION = 768


@dataclass
class ServerState:
    """Container for all server state - initialized once at startup."""

    indexing_service: IndexingService
    embedding_service: EmbeddingService
    sparse_embedding_service: SparseEmbeddingService
    reranker_service: RerankerService
    repository: DocumentVectorRepository
    qdrant_url: str

    @classmethod
    async def create(cls, qdrant_url: str = 'http://localhost:6333') -> typing.Self:
        """Async factory method to create server state with all services wired.

        Must be called from async context to ensure semaphores are bound correctly.
        """
        gemini_client = GeminiClient()
        qdrant_client = QdrantClient(url=qdrant_url)

        chunking_service = await ChunkingService.create()
        embedding_service = EmbeddingService(gemini_client)
        sparse_embedding_service = SparseEmbeddingService()
        reranker_service = RerankerService()
        repository = DocumentVectorRepository(qdrant_client)

        indexing_service = await IndexingService.create(
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            sparse_embedding_service=sparse_embedding_service,
            repository=repository,
        )

        return cls(
            indexing_service=indexing_service,
            embedding_service=embedding_service,
            sparse_embedding_service=sparse_embedding_service,
            reranker_service=reranker_service,
            repository=repository,
            qdrant_url=qdrant_url,
        )


def register_tools(state: ServerState) -> None:
    """Register MCP tools with closure over server state."""

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Index Directory',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=False,
            openWorldHint=True,
        ),
    )
    async def index_directory(
        path: str | None = None,
        full_reindex: bool = False,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> IndexingResult:
        """Index all supported documents in a directory for semantic search.

        Processes files incrementally - only re-indexes files that have changed
        since the last indexing run. Supports markdown, text, JSON, and PDF files.

        Args:
            path: Path to directory to index. Defaults to current working directory
                if not specified. Supports absolute, relative, or ~ expansion.
            full_reindex: If True, reindex all files regardless of whether they've changed.
            ctx: MCP context for logging.

        Returns:
            IndexingResult with counts of files processed, chunks created, and any errors.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Default to current working directory
        if path is None:
            directory = Path.cwd()
        else:
            directory = Path(path).expanduser().resolve()

        if not directory.is_dir():
            raise ValueError(f'Not a directory: {directory}')

        await logger.info(f'Indexing directory: {directory}')
        if full_reindex:
            await logger.info('Full reindex requested - ignoring cache')

        # Progress callback that logs via MCP context
        async def on_progress(progress: IndexingProgress) -> None:
            if progress.percent_complete % 10 < 1 or progress.percent_complete >= 99:
                await logger.info(
                    f'[{progress.percent_complete:.0f}%] {progress.current_phase} | '
                    f'{progress.files_processed}/{progress.files_total} files | '
                    f'{progress.chunks_created} chunks'
                )

        # The on_progress callback is sync in IndexingService, so wrap it
        def sync_progress(progress: IndexingProgress) -> None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(on_progress(progress))
            except RuntimeError:
                pass  # No running loop, skip logging

        result = await state.indexing_service.index_directory(
            directory,
            full_reindex=full_reindex,
            on_progress=sync_progress,
        )

        await logger.info(
            f'Indexing complete: {result.files_processed} files, '
            f'{result.chunks_created} chunks, {len(result.errors)} errors'
        )

        return result

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Search Documents',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
    )
    async def search_documents(
        query: str,
        limit: int = 10,
        file_types: list[str] | None = None,
        path_prefix: str | None = None,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> SearchResult:
        """Search indexed documents using hybrid semantic + keyword search.

        Uses Reciprocal Rank Fusion (RRF) to combine:
        - Dense vectors: Semantic similarity via Gemini embeddings
        - Sparse vectors: BM25 keyword matching

        Args:
            query: Natural language search query.
            limit: Maximum number of results to return (1-100).
            file_types: Filter by file types (e.g., ['markdown', 'pdf']).
            path_prefix: Filter to files under this path prefix.
            ctx: MCP context for logging.

        Returns:
            SearchResult with ranked hits including text snippets and metadata.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)
        await logger.info(f'Searching: "{query[:50]}..."' if len(query) > 50 else f'Searching: "{query}"')

        # Dense embedding (semantic similarity)
        embed_request = EmbedRequest(text=query, task_type='RETRIEVAL_QUERY')
        embed_response = await state.embedding_service.embed(embed_request)

        # Sparse embedding (BM25 keyword matching)
        sparse_indices, sparse_values = state.sparse_embedding_service.embed(query)

        # Fetch more candidates for reranking (3x limit, max 50)
        effective_limit = min(max(limit, 1), 100)
        rerank_candidates = min(effective_limit * 3, 50)

        # Resolve path_prefix to handle ~ and relative paths
        resolved_path_prefix = str(Path(path_prefix).expanduser().resolve()) if path_prefix else None

        # Build hybrid search query
        search_query = SearchQuery(
            dense_vector=embed_response.values,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            limit=rerank_candidates,
            file_types=tuple(typing.cast(FileType, ft) for ft in file_types) if file_types else None,
            source_path_prefix=resolved_path_prefix,
        )

        # Layer 1: Hybrid search with RRF fusion
        result = state.repository.search(search_query)

        # Layer 2: Cross-encoder reranking
        result = await state.reranker_service.rerank(
            query=query,
            result=result,
            top_k=effective_limit,
        )

        await logger.info(f'Found {result.total} results (reranked top {effective_limit})')

        return result

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Get Index Stats',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
    )
    async def get_index_stats(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> dict[str, int | str]:
        """Get statistics about the current document index.

        Returns information about the Qdrant collection including
        vector count, dimension, and status.

        Args:
            ctx: MCP context for logging.

        Returns:
            Dictionary with index statistics.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)
        await logger.info('Getting index stats')

        stats = state.indexing_service.get_index_stats()

        if stats.get('status') == 'not_initialized':
            await logger.warning('Index not initialized - run index_directory first')
        else:
            await logger.info(f'Index has {stats.get("points_count", 0)} vectors')

        return stats


@contextlib.asynccontextmanager
async def lifespan(mcp_server: mcp.server.fastmcp.FastMCP) -> AsyncIterator[None]:
    """Manage server lifecycle - initialization before requests, cleanup after shutdown."""

    # Configure logging with timestamps to stderr for performance observability
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr,
    )
    # Silence noisy third-party loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)

    # Initialize state with all services (async for semaphore binding)
    state = await ServerState.create()

    # Ensure Qdrant collection exists
    state.repository.ensure_collection(EMBEDDING_DIMENSION)

    # Register tools with closure over state
    register_tools(state)

    print('✓ Document Search MCP server initialized', file=sys.stderr)
    print(f'  Qdrant: {state.qdrant_url}', file=sys.stderr)

    # Server is ready - yield control back to FastMCP
    yield

    # Cleanup ProcessPoolExecutor and other resources
    state.indexing_service.shutdown()
    print('✓ Document Search MCP server shutdown', file=sys.stderr)


# Create FastMCP server with lifespan
server = mcp.server.fastmcp.FastMCP('document-search', lifespan=lifespan)


def main() -> None:
    """Entry point for the MCP server."""
    server.run()


if __name__ == '__main__':
    main()
