"""Document Search MCP Server.

Semantic search over local documents using dense embeddings and Qdrant.

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
import os
import sys
import typing
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import mcp.server.fastmcp
import mcp.types
from local_lib.utils import DualLogger

from document_search.clients import QdrantClient, create_embedding_client
from document_search.clients.protocols import EmbeddingClient
from document_search.dashboard.launcher import ensure_dashboard
from document_search.dashboard.state import DashboardStateManager
from document_search.repositories.collection_registry import CollectionRegistryManager
from document_search.repositories.document_vector import DocumentVectorRepository
from document_search.schemas.chunking import FileType
from document_search.schemas.collections import Collection
from document_search.schemas.config import (
    EmbeddingProvider,
    default_config,
)
from document_search.schemas.embeddings import EmbedRequest
from document_search.schemas.indexing import IndexingProgress, IndexingResult
from document_search.schemas.vectors import (
    ClearResult,
    CollectionMetadata,
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
    """Container for all server state - initialized once at startup.

    Shared services are created once. Collection-specific services (embedding clients,
    repositories, indexing services) are created on-demand and cached.
    """

    # Shared infrastructure
    qdrant_client: QdrantClient
    collection_registry: CollectionRegistryManager

    # Shared services (collection-agnostic)
    chunking_service: ChunkingService
    sparse_embedding_service: SparseEmbeddingService
    reranker_service: RerankerService

    # Cached per-provider embedding clients (shared to respect rate limits)
    _embedding_clients: dict[EmbeddingProvider, EmbeddingClient]

    # Cached per-collection repositories
    _repositories: dict[str, DocumentVectorRepository]

    @classmethod
    async def create(
        cls,
        qdrant_url: str = 'http://localhost:6333',
    ) -> typing.Self:
        """Async factory method to create server state with shared services.

        Must be called from async context to ensure semaphores are bound correctly.
        """
        qdrant_client = QdrantClient(url=qdrant_url)
        collection_registry = CollectionRegistryManager()

        # Create shared services (expensive model loading happens here)
        chunking_service = await ChunkingService.create()
        sparse_embedding_service = await SparseEmbeddingService.create()
        reranker_service = RerankerService()

        return cls(
            qdrant_client=qdrant_client,
            collection_registry=collection_registry,
            chunking_service=chunking_service,
            sparse_embedding_service=sparse_embedding_service,
            reranker_service=reranker_service,
            _embedding_clients={},
            _repositories={},
        )

    def get_embedding_client(self, provider: EmbeddingProvider) -> EmbeddingClient:
        """Get or create embedding client for a provider.

        Clients are cached per provider to share rate limiting and semaphores.
        """
        if provider not in self._embedding_clients:
            config = default_config(provider)
            self._embedding_clients[provider] = create_embedding_client(config)
        return self._embedding_clients[provider]

    def get_repository(self, collection_name: str) -> DocumentVectorRepository:
        """Get or create repository for a collection.

        Repositories are cached per collection to share batch loaders.
        """
        if collection_name not in self._repositories:
            self._repositories[collection_name] = DocumentVectorRepository(self.qdrant_client, collection_name)
        return self._repositories[collection_name]

    def get_collection(self, collection_name: str) -> Collection:
        """Get collection from registry. Raises if not found."""
        collection = self.collection_registry.get(collection_name)
        if collection is None:
            raise ValueError(f"Collection '{collection_name}' not found. Use create_collection first.")
        return collection

    async def get_indexing_service(self, collection_name: str) -> IndexingService:
        """Create indexing service for a collection.

        Uses cached embedding client and repository, but creates fresh
        IndexingService instance (lightweight).
        """
        collection = self.get_collection(collection_name)
        embedding_client = self.get_embedding_client(collection.provider)
        config = default_config(collection.provider)
        embedding_service = EmbeddingService(embedding_client, batch_size=config.batch_size)
        repository = self.get_repository(collection_name)

        return await IndexingService.create(
            chunking_service=self.chunking_service,
            embedding_service=embedding_service,
            sparse_embedding_service=self.sparse_embedding_service,
            repository=repository,
        )

    async def close(self) -> None:
        """Close all cached embedding clients."""
        for client in self._embedding_clients.values():
            await client.close()


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
        collection_name: str,
        path: str | None = None,
        full_reindex: bool = False,
        respect_gitignore: bool | None = None,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> IndexingResult:
        """Index documents for semantic search (file or directory auto-detected).

        Processes files incrementally - only re-indexes files that have changed
        since the last indexing run. Supports markdown, text, JSON, and PDF files.

        Args:
            collection_name: Name of the collection to index into.
            path: Path to file or directory to index. Defaults to current working
                directory if not specified. Supports absolute, relative, or ~ expansion.
                Note: "**" is not supported (indexing requires a specific path).
            full_reindex: If True, reindex all files regardless of whether they've changed.
                Only applies to directories (single files are always fully indexed).
            respect_gitignore: Control .gitignore filtering behavior:
                - None (default): Auto-detect git repos, respect gitignore if found.
                - True: Strictly respect gitignore, fail if not a git repo.
                - False: Ignore gitignore, index all supported files.

        Returns:
            IndexingResult with counts of files processed, chunks created, and any errors.
        """
        if not ctx:
            raise ValueError('MCP context required')

        # "**" not supported - indexing requires a specific path
        if path == '**':
            raise ValueError("index_documents does not support '**'. Specify a file or directory path.")

        logger = DualLogger(ctx)

        # Get collection and indexing service
        collection = state.get_collection(collection_name)
        indexing_service = await state.get_indexing_service(collection_name)

        await logger.info(f'Using collection: {collection_name} ({collection.provider})')

        # Default to current working directory
        if path is None:
            resolved_path = Path.cwd()
        else:
            resolved_path = Path(path).expanduser().resolve()

        # Ensure Qdrant collection exists with correct dimensions
        config = default_config(collection.provider)
        repository = state.get_repository(collection_name)
        await repository.ensure_collection(config.embedding_dimensions)

        # Auto-detect file vs directory
        if resolved_path.is_file():
            await logger.info(f'Indexing file: {resolved_path}')
            result = await indexing_service.index_file(resolved_path)
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

        result = await indexing_service.index_directory(
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
        collection_name: str,
        path: str | None = None,
        limit: int = 10,
        search_type: SearchType = 'hybrid',
        file_types: Sequence[str] | None = None,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> SearchResult:
        """Search indexed documents with configurable search strategy.

        Args:
            query: Natural language search query.
            collection_name: Name of the collection to search.
            path: Filter to files under this path. Defaults to CWD.
                Use "**" to search entire collection.
            limit: Maximum number of results to return (1-100).
            search_type: Search strategy:
                - 'hybrid' (default): Dense + sparse vectors with RRF fusion.
                  Combines semantic similarity with keyword matching.
                - 'lexical': BM25 keyword matching only. Best for exact term
                  matching, symbol lookup, and identifier search.
                - 'embedding': Dense vector similarity only. Useful for
                  conceptual/semantic queries or debugging.
            file_types: Filter by file types (e.g., ['markdown', 'pdf']).

        Returns:
            SearchResult with ranked hits including text snippets and metadata.
        """
        if not ctx:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Get collection and services
        collection = state.get_collection(collection_name)
        embedding_client = state.get_embedding_client(collection.provider)
        config = default_config(collection.provider)
        embedding_service = EmbeddingService(embedding_client, batch_size=config.batch_size)
        repository = state.get_repository(collection_name)

        await logger.info(
            f'Searching {collection_name} ({search_type}): "{query[:50]}..."'
            if len(query) > 50
            else f'Searching {collection_name} ({search_type}): "{query}"'
        )

        # Compute embeddings based on search type
        if search_type == 'hybrid':
            # Both dense and sparse embeddings
            embed_request = EmbedRequest(text=query, intent='query')
            embed_response = await embedding_service.embed(embed_request)
            dense_vector: Sequence[float] | None = embed_response.values
            sparse_indices, sparse_values = await state.sparse_embedding_service.embed(query)
        elif search_type == 'lexical':
            # Sparse only (BM25 keyword matching)
            dense_vector = None
            sparse_indices, sparse_values = await state.sparse_embedding_service.embed(query)
        elif search_type == 'embedding':
            # Dense only (semantic similarity)
            embed_request = EmbedRequest(text=query, intent='query')
            embed_response = await embedding_service.embed(embed_request)
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
        result = await repository.search(search_query)

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
        collection_name: str,
        path: str | None = None,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> ClearResult:
        """Remove documents from a collection's index.

        Does not delete files from disk - only removes them from the search index.

        Args:
            collection_name: Name of the collection to clear documents from.
            path: Path to clear. Defaults to CWD if not specified.
                Use "**" to clear entire collection.

        Returns:
            ClearResult with counts of files and chunks removed.
        """
        if not ctx:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Verify collection exists
        collection = state.get_collection(collection_name)
        indexing_service = await state.get_indexing_service(collection_name)

        await logger.info(f'Clearing from collection: {collection_name} ({collection.provider})')

        # Resolve path (defaults to CWD)
        if path == '**':
            resolved_path = '**'
            await logger.info('Clearing entire collection')
        elif path:
            resolved_path = str(Path(path).expanduser().resolve())
            await logger.info(f'Clearing documents: {resolved_path}')
        else:
            resolved_path = str(Path.cwd())
            await logger.info(f'Clearing documents under: {resolved_path}')

        result = await indexing_service.clear_documents(resolved_path)

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
        collection_name: str,
        path: str | None = None,
        file_type: str | None = None,
        limit: int = 50,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> Sequence[IndexedFile]:
        """List indexed documents with optional filtering.

        Returns files sorted by chunk count (descending). Useful for auditing
        indexed content or checking if specific files are indexed.

        Args:
            collection_name: Name of the collection to list documents from.
            path: Filter to files under this path. Defaults to CWD.
                Use "**" to list entire collection.
            file_type: Filter to this file type (e.g., 'markdown', 'pdf').
            limit: Maximum number of files to return (default 50).

        Returns:
            List of IndexedFile with path, chunk_count, and file_type.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Verify collection exists and get repository
        state.get_collection(collection_name)
        repository = state.get_repository(collection_name)

        await logger.info(f'Listing documents in collection: {collection_name}')

        # Resolve path (defaults to CWD, "**" for global)
        if path == '**':
            resolved_path = None
        elif path is None:
            resolved_path = str(Path.cwd())
        else:
            resolved_path = str(Path(path).expanduser().resolve())

        files = await repository.list_indexed_files(
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
        collection_name: str,
        path: str | None = None,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> IndexInfo:
        """Get comprehensive collection information.

        Returns collection metadata, storage stats, and content breakdown.

        Args:
            collection_name: Name of the collection to get info for.
            path: Scope content stats to files under this path. Defaults to CWD.
                Use "**" for global stats across entire collection.

        Returns:
            IndexInfo with collection metadata, storage stats, and content breakdown.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Get collection metadata from registry
        collection = state.get_collection(collection_name)
        repository = state.get_repository(collection_name)

        await logger.info(f'Getting info for collection: {collection_name}')

        # Resolve path (defaults to CWD, "**" for global)
        if path == '**':
            resolved_path: str | None = '**'
            await logger.info('Getting global collection info')
        elif path is None:
            resolved_path = str(Path.cwd())
            await logger.info(f'Getting info for: {resolved_path}')
        else:
            resolved_path = str(Path(path).expanduser().resolve())
            await logger.info(f'Getting info for: {resolved_path}')

        # Get provider config and embedding info
        config = default_config(collection.provider)
        embedding_info = config.to_info()

        # Get storage stats from Qdrant
        storage = await repository.get_storage_stats()
        if storage is None:
            raise ValueError('Collection not initialized in Qdrant - run index_documents first')

        # Get content stats (scoped by path)
        content = await repository.get_content_stats(resolved_path)

        # Build comprehensive response
        info = IndexInfo(
            collection=CollectionMetadata(
                name=collection.name,
                description=collection.description,
                created_at=collection.created_at,
                provider=collection.provider,
            ),
            embedding=embedding_info,
            storage=storage,
            content=content,
            path=resolved_path,
        )

        await logger.info(
            f'Collection: {content.total_chunks} chunks, {content.unique_files} files, status={storage.status}'
        )

        return info

    # Collection management tools

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Create Collection',
            destructiveHint=False,
            idempotentHint=False,
            readOnlyHint=False,
            openWorldHint=False,
        ),
    )
    async def create_collection(
        name: str,
        description: str | None,
        provider: EmbeddingProvider,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> Collection:
        """Create a new document collection.

        Each collection uses a specific embedding provider and stores documents
        in a separate Qdrant collection.

        Args:
            name: Unique name for the collection (e.g., 'frontend-docs', 'api-specs').
            description: Optional description of the collection's purpose.
            provider: Embedding provider to use ('gemini' or 'openrouter').

        Returns:
            The created Collection with name, provider, and creation timestamp.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Validate name
        if name == '**':
            raise ValueError("Collection name '**' is reserved")

        collection = state.collection_registry.create_collection(
            name=name,
            provider=provider,
            description=description,
        )

        await logger.info(f'Created collection: {name} (provider: {provider})')

        return collection

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='List Collections',
            destructiveHint=False,
            idempotentHint=True,
            readOnlyHint=True,
            openWorldHint=False,
        ),
    )
    async def list_collections(
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> Sequence[Collection]:
        """List all document collections.

        Returns:
            List of all collections with their names, providers, and descriptions.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        collections = state.collection_registry.list_collections()

        return collections

    @server.tool(
        annotations=mcp.types.ToolAnnotations(
            title='Delete Collection',
            destructiveHint=True,
            idempotentHint=True,
            readOnlyHint=False,
            openWorldHint=False,
        ),
    )
    async def delete_collection(
        name: str,
        ctx: mcp.server.fastmcp.Context[typing.Any, typing.Any, typing.Any] | None = None,
    ) -> bool:
        """Delete a document collection.

        Removes both the collection metadata and all indexed documents in Qdrant.

        Args:
            name: Name of the collection to delete.

        Returns:
            True if deleted successfully.
        """
        if ctx is None:
            raise ValueError('MCP context required')

        logger = DualLogger(ctx)

        # Verify collection exists and delete from Qdrant
        state.get_collection(name)
        await state.qdrant_client.delete_collection(name)

        # Remove from registry
        state.collection_registry.delete_collection(name)

        # Clear from cache
        if name in state._repositories:
            del state._repositories[name]

        await logger.info(f'Deleted collection: {name}')

        return True


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

    # Initialize state with shared services (async for semaphore binding)
    state = await ServerState.create()

    # Migrate legacy collection if needed
    await _migrate_legacy_collection(state)

    # Register tools with closure over state
    register_tools(state)

    # Start dashboard and register this MCP server
    dashboard_state_manager = DashboardStateManager()
    dashboard_port = ensure_dashboard()
    dashboard_state_manager.register_mcp_server(os.getpid())

    # List existing collections
    collections = state.collection_registry.list_collections()
    collection_names = [c.name for c in collections]

    print('✓ Document Search MCP server initialized', file=sys.stderr)
    print(f'  Collections: {collection_names if collection_names else "(none)"}', file=sys.stderr)
    print(f'  Dashboard: http://127.0.0.1:{dashboard_port}', file=sys.stderr)

    # Server is ready - yield control back to FastMCP
    yield

    # Cleanup resources
    dashboard_state_manager.unregister_mcp_server(os.getpid())
    await state.close()
    print('✓ Document Search MCP server shutdown', file=sys.stderr)


server = mcp.server.fastmcp.FastMCP('document-search', lifespan=lifespan)


# Create FastMCP server with lifespan
def main() -> None:
    """Entry point for the MCP server."""
    server.run()


async def _migrate_legacy_collection(state: ServerState) -> None:
    """Migrate legacy 'document_chunks' collection to the registry if needed.

    For backwards compatibility: if the registry is empty but Qdrant has the
    old default collection, create a registry entry for it.
    """
    collections = state.collection_registry.list_collections()
    if collections:
        return  # Registry not empty, no migration needed

    if not await state.qdrant_client.collection_exists('document_chunks'):
        return  # No legacy collection to migrate

    state.collection_registry.create_collection(
        name='document_chunks',
        provider='gemini',
        description='Migrated from legacy single-collection setup',
    )
    print('  Migrated legacy collection: document_chunks (gemini)', file=sys.stderr)


if __name__ == '__main__':
    main()
