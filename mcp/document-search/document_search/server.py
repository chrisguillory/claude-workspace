"""Document Search MCP Server.

Semantic search over local documents using Gemini embeddings and Qdrant.

Tools:
- index_documents: Index file or directory for semantic search
- clear_documents: Remove documents from the index
- search_documents: Search indexed documents by natural language query
- list_documents: List indexed documents with filtering
- get_info: Get index health and statistics
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import typing
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import mcp.server.fastmcp
import mcp.types
from local_lib.utils import DualLogger

from document_search.clients.gemini import GeminiClient
from document_search.clients.qdrant import QdrantClient
from document_search.repositories.document_vector import DocumentVectorRepository
from document_search.schemas.chunking import FileType
from document_search.schemas.config import (
    EmbeddingConfig,
    MigrationRequiredError,
    load_config,
    save_config,
)
from document_search.schemas.embeddings import EmbedRequest
from document_search.schemas.indexing import IndexingProgress, IndexingResult
from document_search.schemas.vectors import (
    ClearResult,
    IndexedFile,
    IndexInfo,
    SearchQuery,
    SearchResult,
    SearchType,
)
from document_search.services.chunking import ChunkingService
from document_search.services.embedding import EmbeddingService
from document_search.services.indexing import IndexingService
from document_search.services.reranker import RerankerService
from document_search.services.sparse_embedding import SparseEmbeddingService

__all__ = [
    'ServerState',
]


@dataclass
class ServerState:
    """Container for all server state - initialized once at startup."""

    config: EmbeddingConfig
    indexing_service: IndexingService
    embedding_service: EmbeddingService
    sparse_embedding_service: SparseEmbeddingService
    reranker_service: RerankerService
    repository: DocumentVectorRepository
    qdrant_url: str
    legacy_chunks: int = 0  # Non-zero if legacy data exists without config

    @classmethod
    async def create(
        cls,
        config: EmbeddingConfig,
        qdrant_url: str = 'http://localhost:6333',
        legacy_chunks: int = 0,
    ) -> typing.Self:
        """Async factory method to create server state with all services wired.

        Must be called from async context to ensure semaphores are bound correctly.
        """
        gemini_client = GeminiClient(
            model=config.embedding_model,
            output_dimensionality=config.embedding_dimensions,
            requests_per_minute=config.requests_per_minute,
        )
        qdrant_client = QdrantClient(url=qdrant_url)

        chunking_service = await ChunkingService.create()
        embedding_service = EmbeddingService(gemini_client)
        sparse_embedding_service = await SparseEmbeddingService.create()
        reranker_service = RerankerService()
        repository = DocumentVectorRepository(qdrant_client)

        indexing_service = await IndexingService.create(
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            sparse_embedding_service=sparse_embedding_service,
            repository=repository,
        )

        return cls(
            config=config,
            indexing_service=indexing_service,
            embedding_service=embedding_service,
            sparse_embedding_service=sparse_embedding_service,
            reranker_service=reranker_service,
            repository=repository,
            qdrant_url=qdrant_url,
            legacy_chunks=legacy_chunks,
        )

    def update_embedding_config(self, new_config: EmbeddingConfig) -> None:
        """Update embedding config and recreate dependent services.

        Called after clear_documents with new config params.
        """
        # Create new client and service with new config
        gemini_client = GeminiClient(
            model=new_config.embedding_model,
            output_dimensionality=new_config.embedding_dimensions,
            requests_per_minute=new_config.requests_per_minute,
        )
        new_embedding_service = EmbeddingService(gemini_client)

        # Update references
        self.config = new_config
        self.embedding_service = new_embedding_service
        # Update internal reference in indexing service
        self.indexing_service._embedding = new_embedding_service  # noqa: SLF001

    def require_migration(self) -> None:
        """Raise if in legacy mode (data exists but no config)."""
        if self.legacy_chunks:
            raise MigrationRequiredError(
                f'Index contains {self.legacy_chunks} chunks from deprecated text-embedding-004 model. '
                f"Run clear_documents(path='**') to clear, then re-index. "
                f'See: https://ai.google.dev/gemini-api/docs/deprecations'
            )


def register_tools(state: ServerState) -> None:
    """Register MCP tools with closure over server state."""

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Index Documents',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=False,
            openWorldHint=True,
        ),
    )
    async def index_documents(
        path: str | None = None,
        full_reindex: bool = False,
        respect_gitignore: bool | None = None,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> IndexingResult:
        """Index documents for semantic search (file or directory auto-detected).

        Processes files incrementally - only re-indexes files that have changed
        since the last indexing run. Supports markdown, text, JSON, and PDF files.

        Args:
            path: Path to file or directory to index. Defaults to current working
                directory if not specified. Supports absolute, relative, or ~ expansion.
                Note: "**" is not supported (indexing requires a specific path).
            full_reindex: If True, reindex all files regardless of whether they've changed.
                Only applies to directories (single files are always fully indexed).
            respect_gitignore: Control .gitignore filtering behavior:
                - None (default): Auto-detect git repos, respect gitignore if found.
                - True: Strictly respect gitignore, fail if not a git repo.
                - False: Ignore gitignore, index all supported files.
            ctx: MCP context for logging.

        Returns:
            IndexingResult with counts of files processed, chunks created, and any errors.
        """
        if not ctx:
            raise ValueError('MCP context required')

        # "**" not supported - indexing requires a specific path
        if path == '**':
            raise ValueError("index_documents does not support '**'. Specify a file or directory path.")

        state.require_migration()
        logger = DualLogger(ctx)

        # Default to current working directory
        if path is None:
            resolved_path = Path.cwd()
        else:
            resolved_path = Path(path).expanduser().resolve()

        # Auto-detect file vs directory
        if resolved_path.is_file():
            await logger.info(f'Indexing file: {resolved_path}')
            result = await state.indexing_service.index_file(resolved_path)
            await logger.info(f'Indexed: {result.chunks_created} chunks')
            return result

        if not resolved_path.is_dir():
            raise ValueError(f'Path not found: {resolved_path}')

        await logger.info(f'Indexing directory: {resolved_path}')
        if full_reindex:
            await logger.info('Full reindex requested - ignoring cache')

        # Progress callback that logs via MCP context
        async def on_progress(progress: IndexingProgress) -> None:
            if progress.percent_complete % 10 < 1 or progress.percent_complete >= 99:
                await logger.info(
                    f'[{progress.percent_complete:.0f}%] {progress.current_phase} | '
                    f'{progress.files_indexed}/{progress.files_total} files | '
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
            resolved_path,
            full_reindex=full_reindex,
            respect_gitignore=respect_gitignore,
            on_progress=sync_progress,
        )

        await logger.info(
            f'Indexing complete: {result.files_indexed} indexed, '
            f'{result.files_cached} cached, {result.chunks_created} chunks, '
            f'{len(result.errors)} errors'
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
        path: str | None = None,
        limit: int = 10,
        search_type: SearchType = 'hybrid',
        file_types: Sequence[str] | None = None,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> SearchResult:
        """Search indexed documents with configurable search strategy.

        Args:
            query: Natural language search query.
            path: Filter to files under this path. Defaults to CWD.
                Use "**" to search entire index.
            limit: Maximum number of results to return (1-100).
            search_type: Search strategy:
                - 'hybrid' (default): Dense + sparse vectors with RRF fusion.
                  Combines semantic similarity with keyword matching.
                - 'lexical': BM25 keyword matching only. Best for exact term
                  matching, symbol lookup, and identifier search.
                - 'embedding': Dense vector similarity only. Useful for
                  conceptual/semantic queries or debugging.
            file_types: Filter by file types (e.g., ['markdown', 'pdf']).
            ctx: MCP context for logging.

        Returns:
            SearchResult with ranked hits including text snippets and metadata.
        """
        if not ctx:
            raise ValueError('MCP context required')

        # Semantic search requires migration if legacy data exists
        if search_type in ('hybrid', 'embedding'):
            state.require_migration()

        logger = DualLogger(ctx)
        await logger.info(
            f'Searching ({search_type}): "{query[:50]}..."'
            if len(query) > 50
            else f'Searching ({search_type}): "{query}"'
        )

        # Compute embeddings based on search type
        if search_type == 'hybrid':
            # Both dense and sparse embeddings
            embed_request = EmbedRequest(text=query, task_type='RETRIEVAL_QUERY')
            embed_response = await state.embedding_service.embed(embed_request)
            dense_vector: Sequence[float] | None = embed_response.values
            sparse_indices, sparse_values = await state.sparse_embedding_service.embed(query)
        elif search_type == 'lexical':
            # Sparse only (BM25 keyword matching)
            dense_vector = None
            sparse_indices, sparse_values = await state.sparse_embedding_service.embed(query)
        elif search_type == 'embedding':
            # Dense only (semantic similarity)
            embed_request = EmbedRequest(text=query, task_type='RETRIEVAL_QUERY')
            embed_response = await state.embedding_service.embed(embed_request)
            dense_vector = embed_response.values
            sparse_indices = None
            sparse_values = None
        else:
            raise NotImplementedError

        # Fetch more candidates for reranking (3x limit, max 50)
        effective_limit = min(max(limit, 1), 100)
        rerank_candidates = min(effective_limit * 3, 50)

        # Resolve path (defaults to CWD, "**" means no filter)
        if path == '**':
            resolved_path: str | None = None
        elif path is None:
            resolved_path = str(Path.cwd())
        else:
            resolved_path = str(Path(path).expanduser().resolve())

        # Build search query
        search_query = SearchQuery(
            search_type=search_type,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            limit=rerank_candidates,
            file_types=tuple(typing.cast(FileType, ft) for ft in file_types) if file_types else None,
            source_path_prefix=resolved_path,
        )

        # Layer 1: Search with specified strategy
        result = await state.repository.search(search_query)

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
            title='Clear Documents',
            destructiveHint=True,
            idempotentHint=True,
            readOnlyHint=False,
            openWorldHint=False,
        ),
    )
    async def clear_documents(
        path: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> ClearResult:
        """Remove documents from the index.

        Does not delete files from disk - only removes them from the search index.

        When clearing the entire index (path='**'), you can optionally specify new
        embedding configuration for migrating to a new embedding model.

        Args:
            path: Path to clear. Defaults to CWD if not specified.
                Use "**" to clear entire index.
            embedding_model: New embedding model (only with path='**').
                Example: 'gemini-embedding-001'
            embedding_dimensions: New embedding dimensions (only with path='**').
                Example: 768
            ctx: MCP context for logging.

        Returns:
            ClearResult with counts of files and chunks removed.
        """
        if not ctx:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Config params only allowed with full clear
        if (embedding_model or embedding_dimensions) and path != '**':
            raise ValueError("embedding_model and embedding_dimensions require path='**'")

        # Resolve path (defaults to CWD)
        if path == '**':
            resolved_path = '**'
            await logger.info('Clearing entire index')
        elif path:
            resolved_path = str(Path(path).expanduser().resolve())
            await logger.info(f'Clearing documents: {resolved_path}')
        else:
            resolved_path = str(Path.cwd())
            await logger.info(f'Clearing documents under: {resolved_path}')

        result = await state.indexing_service.clear_documents(resolved_path)

        # Full clear: reset legacy mode, save/update config
        if path == '**':
            state.legacy_chunks = 0
            new_config = EmbeddingConfig(
                embedding_model=embedding_model or state.config.embedding_model,
                embedding_dimensions=embedding_dimensions or state.config.embedding_dimensions,
            )
            save_config(new_config)
            if embedding_model or embedding_dimensions:
                state.update_embedding_config(new_config)
                await logger.info(f'Updated config: {new_config.embedding_model} ({new_config.embedding_dimensions}d)')

        await logger.info(f'Cleared: {result.files_removed} files, {result.chunks_removed} chunks')

        return result

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='List Documents',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
    )
    async def list_documents(
        path: str | None = None,
        file_type: str | None = None,
        limit: int = 50,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> Sequence[IndexedFile]:
        """List indexed documents with optional filtering.

        Returns files sorted by chunk count (descending). Useful for auditing
        indexed content or checking if specific files are indexed.

        Args:
            path: Filter to files under this path. Defaults to CWD.
                Use "**" to search entire index.
            file_type: Filter to this file type (e.g., 'markdown', 'pdf').
            limit: Maximum number of files to return (default 50).
            ctx: MCP context for logging.

        Returns:
            List of IndexedFile with path, chunk_count, and file_type.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Resolve path (defaults to CWD, "**" for global)
        if path == '**':
            resolved_path = None
        elif path is None:
            resolved_path = str(Path.cwd())
        else:
            resolved_path = str(Path(path).expanduser().resolve())

        files = await state.repository.list_indexed_files(
            path_prefix=resolved_path,
            file_type=file_type,
            limit=limit,
        )

        await logger.info(f'Listed {len(files)} indexed documents')

        return files

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Get Info',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
    )
    async def get_info(
        path: str | None = None,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> IndexInfo:
        """Get index health and statistics.

        Returns infrastructure stats (always global) and content breakdown
        (optionally scoped by path).

        Args:
            path: Scope content stats to files under this path. Defaults to CWD.
                Use "**" for global stats across entire index.
            ctx: MCP context for logging.

        Returns:
            IndexInfo with infrastructure and content sections.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Resolve path (defaults to CWD, "**" for global)
        if path == '**':
            resolved_path = '**'
            await logger.info('Getting global index info')
        elif path is None:
            resolved_path = str(Path.cwd())
            await logger.info(f'Getting index info for: {resolved_path}')
        else:
            resolved_path = str(Path(path).expanduser().resolve())
            await logger.info(f'Getting index info for: {resolved_path}')

        try:
            info = await state.repository.get_index_info(resolved_path)
        except ValueError as e:
            await logger.warning(str(e))
            raise

        await logger.info(
            f'Index: {info.content.total_chunks} chunks, '
            f'{info.content.unique_files} files, '
            f'status={info.infrastructure.status}'
        )

        return info


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

    # Load or detect embedding config
    config = load_config()
    legacy_chunks = 0

    if config is None:
        # No config file - check if collection has data (legacy state)
        # TODO: Remove legacy detection after March 2026
        qdrant_client = QdrantClient(url='http://localhost:6333')
        collection_info = await qdrant_client.get_collection_info()

        if collection_info and collection_info['points_count'] > 0:
            # Legacy data exists - server starts but semantic ops blocked
            legacy_chunks = collection_info['points_count']
            config = EmbeddingConfig.default()
            print(
                f'⚠ Legacy index detected: {legacy_chunks} chunks from deprecated model. '
                f"Run clear_documents(path='**') to migrate.",
                file=sys.stderr,
            )
        else:
            # Empty or no collection - use defaults
            config = EmbeddingConfig.default()
            save_config(config)
            print(f'✓ Created embedding config: {config.embedding_model}', file=sys.stderr)

    # Initialize state with all services (async for semaphore binding)
    state = await ServerState.create(config, legacy_chunks=legacy_chunks)

    # Ensure Qdrant collection exists
    await state.repository.ensure_collection(config.embedding_dimensions)

    # Register tools with closure over state
    register_tools(state)

    print('✓ Document Search MCP server initialized', file=sys.stderr)
    print(f'  Model: {config.embedding_model} ({config.embedding_dimensions}d)', file=sys.stderr)
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
