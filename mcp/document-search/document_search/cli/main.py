"""Command-line interface for document-search.

Semantic search over local documents from the terminal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import webbrowser
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import UUID

import typer
from cc_lib.cli import add_completion_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

from document_search.clients import QdrantClient, create_embedding_client
from document_search.clients.redis import RedisClient, discover_redis_port
from document_search.dashboard.state import DashboardStateManager
from document_search.paths import PROJECT_ROOT
from document_search.repositories.collection_registry import CollectionRegistryManager
from document_search.repositories.document_vector import DocumentVectorRepository
from document_search.repositories.index_state import IndexStateStore
from document_search.schemas.config import create_config, default_config
from document_search.schemas.embeddings import EmbedRequest
from document_search.schemas.indexing import StopAfterStage
from document_search.schemas.vectors import (
    ClearResult,
    CollectionMetadata,
    EmbeddingInfo,
    IndexInfo,
    SearchQuery,
    SearchResult,
)

logger = logging.getLogger(__name__)

app = create_app(help='Document search CLI — semantic search over local documents.')
add_completion_command(app)
error_boundary = ErrorBoundary(exit_code=1)


# -- Typer infrastructure (must precede commands that reference them) --


def _complete_collection_name(incomplete: str) -> Sequence[tuple[str, str]]:
    registry = CollectionRegistryManager()
    return [(c.name, f'{c.provider}/{c.model}') for c in registry.list_collections() if c.name.startswith(incomplete)]


def _resolve_collection(explicit: str | None) -> str:
    """Resolve collection name: explicit > env var > single-collection auto-detect > error."""
    if explicit:
        return explicit
    env_default = os.environ.get('DOCUMENT_SEARCH_DEFAULT_COLLECTION')
    if env_default:
        return env_default
    registry = CollectionRegistryManager()
    colls = registry.list_collections()
    if len(colls) == 1:
        return colls[0].name
    if not colls:
        typer.secho('No collections found. Create one via the MCP server.', fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    names = ', '.join(c.name for c in colls)
    typer.secho(f'Multiple collections exist: {names}', fg=typer.colors.RED, err=True)
    typer.echo('Specify a collection or set DOCUMENT_SEARCH_DEFAULT_COLLECTION.', err=True)
    raise typer.Exit(1)


# -- App callback --


@app.callback(invoke_without_command=True)
def _configure_logging(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, '--verbose', '-v', help='Show detailed output'),
) -> None:
    """Configure logging and show help when no command given."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(message)s', stream=sys.stderr, force=True)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# -- Dashboard command group --


dashboard_app = typer.Typer(help='Dashboard management.')
app.add_typer(dashboard_app, name='dashboard')


@dashboard_app.command('open')
@error_boundary
def dashboard_open(
    browser: Annotated[
        Literal['default', 'safari', 'chromium'], typer.Option('--browser', '-b', help='Browser to open with.')
    ] = 'default',
) -> None:
    """Open dashboard in browser.

    \b
    Examples:
        document-search dashboard open
        document-search dashboard open --browser safari
        document-search dashboard open -b chromium
    """
    port = DashboardStateManager().get_dashboard_port()
    if port is None:
        typer.secho('Dashboard is not running.', fg=typer.colors.RED, err=True)
        typer.echo('Start it by running the MCP server, or:', err=True)
        typer.secho('  document-search-dashboard', fg=typer.colors.CYAN, err=True)
        raise typer.Exit(1)
    url = f'http://127.0.0.1:{port}'
    if browser == 'default':
        typer.echo(f'Opening {url}')
        webbrowser.open(url)
    else:
        browser_name = 'safari' if browser == 'safari' else 'chromium-browser'
        webbrowser.get(browser_name).open(url)
        typer.echo(f'Opening {url} in {browser}')


@dashboard_app.command('status')
@error_boundary
def dashboard_status(
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Show dashboard status."""
    manager = DashboardStateManager()
    port = manager.get_dashboard_port()
    if port is None:
        if format == 'json':
            typer.echo('{"running": false}')
        else:
            typer.secho('Dashboard is not running.', fg=typer.colors.YELLOW)
        return
    live_processes = manager.get_live_processes()
    if format == 'json':
        typer.echo(
            json.dumps(
                {
                    'running': True,
                    'url': f'http://127.0.0.1:{port}',
                    'registered_processes': len(live_processes),
                },
                indent=2,
            )
        )
    else:
        typer.secho('Dashboard:', bold=True)
        typer.echo(f'  URL: http://127.0.0.1:{port}')
        typer.echo(f'  Processes: {len(live_processes)}')


@dashboard_app.command('url')
@error_boundary
def dashboard_url() -> None:
    """Print dashboard URL (for scripting).

    \b
    Examples:
        open $(document-search dashboard url)
        open -a Safari $(document-search dashboard url)
    """
    port = DashboardStateManager().get_dashboard_port()
    if port is None:
        raise typer.Exit(1)
    typer.echo(f'http://127.0.0.1:{port}')


# -- Commands: file-only tier --


@app.command()
@error_boundary
def collections(
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """List all collections."""
    registry = CollectionRegistryManager()
    colls = registry.list_collections()
    if format == 'json':
        typer.echo(json.dumps([c.model_dump(mode='json') for c in colls], indent=2, default=str))
        return
    if not colls:
        typer.echo('No collections. Create one via the MCP server.')
        return
    for c in colls:
        typer.secho(c.name, bold=True)
        typer.echo(f'  Provider: {c.provider}/{c.model}')
        typer.echo(f'  Dimensions: {c.dimensions}')
        if c.description:
            typer.echo(f'  Description: {c.description}')


# -- Commands: Qdrant tier --


@app.command()
@error_boundary
def info(
    collection: Annotated[
        str | None,
        typer.Argument(
            help='Collection name (optional if only one exists).',
            autocompletion=_complete_collection_name,
        ),
    ] = None,
    path: Annotated[str | None, typer.Option('--path', '-p', help='Scope to path ("**" for global).')] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Show collection info.

    \b
    Examples:
        document-search info
        document-search info my-collection --path "**"
        document-search info -f json
    """
    asyncio.run(_info_async(_resolve_collection(collection), path, format))


@app.command('list')
@error_boundary
def list_docs(
    collection: Annotated[
        str | None,
        typer.Argument(
            help='Collection name (optional if only one exists).',
            autocompletion=_complete_collection_name,
        ),
    ] = None,
    path: Annotated[str | None, typer.Option('--path', '-p', help='Scope to path ("**" for global).')] = None,
    file_type: Annotated[str | None, typer.Option('--type', '-t', help='Filter by file type.')] = None,
    limit: Annotated[int, typer.Option('--limit', '-n', help='Max files to return.')] = 50,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """List indexed documents.

    \b
    Examples:
        document-search list
        document-search list --type markdown --limit 10
        document-search list my-collection --path "**"
    """
    asyncio.run(_list_async(_resolve_collection(collection), path, file_type, limit, format))


@app.command()
@error_boundary
def clear(
    collection: Annotated[
        str | None,
        typer.Argument(
            help='Collection name (optional if only one exists).',
            autocompletion=_complete_collection_name,
        ),
    ] = None,
    path: Annotated[
        str | None, typer.Option('--path', '-p', help='Path to clear ("**" for entire collection).')
    ] = None,
    clear_cache: Annotated[bool, typer.Option('--clear-cache', help='Also delete cached embeddings.')] = False,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Clear documents from index.

    \b
    Examples:
        document-search clear
        document-search clear --path "**" --clear-cache
        document-search clear my-collection --path /old/docs
    """
    resolved = _resolve_collection(collection)
    scope = path or str(Path.cwd())
    typer.confirm(f'Clear documents from {resolved} (path: {scope})?', abort=True)
    asyncio.run(_clear_async(resolved, path, clear_cache, format))


# -- Commands: full-stack tier --


@app.command()
@error_boundary
def search(
    query: Annotated[str, typer.Argument(help='Search query.')],
    collection: Annotated[
        str | None,
        typer.Option(
            '--collection',
            '-c',
            help='Collection name (optional if only one exists).',
            autocompletion=_complete_collection_name,
        ),
    ] = None,
    path: Annotated[str | None, typer.Option('--path', '-p', help='Scope to path ("**" for global).')] = None,
    limit: Annotated[int, typer.Option('--limit', '-n', help='Max results.')] = 10,
    search_type: Annotated[
        Literal['hybrid', 'lexical', 'embedding'], typer.Option('--type', '-t', help='Search strategy.')
    ] = 'hybrid',
    exclude_path: Annotated[
        list[str] | None, typer.Option('--exclude', '-x', help='Exclude files under these paths.')
    ] = None,  # strict_typing_linter.py: mutable-type — typer requires list
    min_score: Annotated[
        float | None, typer.Option('--min-score', help='Minimum relevance score (0.0 = relevant).')
    ] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Search documents.

    \b
    Examples:
        document-search search "authentication flow"
        document-search search "API endpoints" -c api-specs --type lexical
        document-search search "error handling" --limit 5 -f json
        document-search search "config" --exclude /path/to/noise/
        document-search search "retry logic" --min-score 0.0
    """
    asyncio.run(
        _search_async(query, _resolve_collection(collection), path, limit, search_type, exclude_path, min_score, format)
    )


@app.command()
@error_boundary
def index(
    paths: Annotated[
        list[Path], typer.Argument(help='Files or directories to index.')
    ],  # strict_typing_linter.py: mutable-type — typer requires list
    collection: Annotated[
        str | None,
        typer.Option(
            '--collection',
            '-c',
            help='Collection name (optional if only one exists).',
            autocompletion=_complete_collection_name,
        ),
    ] = None,
    gitignore: Annotated[
        bool | None,
        typer.Option(
            '--gitignore/--no-gitignore',
            help='Respect .gitignore (default: auto-detect).',
        ),
    ] = None,
    stop_after: Annotated[
        str | None,
        typer.Option(
            '--stop-after',
            help='Stop pipeline at stage: scan, chunk, embed.',
        ),
    ] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Index documents for search.

    \b
    Examples:
        document-search index .
        document-search index /docs /api /guides
        document-search index ./docs/**/*.md -c other-collection
    """
    asyncio.run(_index_async(paths, _resolve_collection(collection), gitignore, stop_after, format))


# -- Entry point --


def main() -> None:
    """Entry point for CLI."""
    run_app(app)


# -- Private helpers (used by async implementations below) --


def _resolve_path(path: str | None) -> str | None:
    """Resolve path argument: '**' for global, None for CWD, else expand."""
    if path == '**':
        return '**'
    if path is None:
        return str(Path.cwd())
    return str(Path(path).expanduser().resolve())


@dataclass
class InfraContext:
    qdrant: QdrantClient
    redis: RedisClient
    registry: CollectionRegistryManager


@asynccontextmanager
async def infrastructure() -> AsyncIterator[InfraContext]:
    redis_port = discover_redis_port(PROJECT_ROOT)
    redis = RedisClient(host='127.0.0.1', port=redis_port)
    try:
        await redis.ping()
        qdrant = QdrantClient(url='http://localhost:6333')
        registry = CollectionRegistryManager()
        yield InfraContext(qdrant=qdrant, redis=redis, registry=registry)
    finally:
        await redis.close()


# -- Private async implementations --


async def _info_async(collection_name: str, path: str | None, format: Literal['text', 'json']) -> None:
    async with infrastructure() as ctx:
        collection = ctx.registry.get(collection_name)
        if collection is None:
            typer.secho(f"Collection '{collection_name}' not found.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        state_store = IndexStateStore(ctx.redis, collection_name)
        repository = DocumentVectorRepository(ctx.qdrant, collection_name, state_store)

        embedding_info = EmbeddingInfo.from_collection(collection)
        storage = await repository.get_storage_stats()
        if storage is None:
            typer.secho('Collection not initialized in Qdrant — run index first.', fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(1)

        resolved_path = _resolve_path(path)
        content = await repository.get_content_stats(resolved_path)

        dashboard_port = DashboardStateManager().get_dashboard_port()
        dashboard_url = f'http://127.0.0.1:{dashboard_port}' if dashboard_port else None

        info_result = IndexInfo(
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
            dashboard_url=dashboard_url,
        )

        if format == 'json':
            typer.echo(info_result.model_dump_json(indent=2))
            return

        typer.secho('Collection:', bold=True)
        typer.echo(f'  Name: {collection.name}')
        if collection.description:
            typer.echo(f'  Description: {collection.description}')
        typer.echo()

        typer.secho('Embedding:', bold=True)
        typer.echo(f'  Provider: {embedding_info.provider}/{embedding_info.model}')
        typer.echo(f'  Dimensions: {embedding_info.dimensions}')
        typer.echo(f'  Batch size: {embedding_info.batch_size}')
        typer.echo()

        typer.secho('Storage:', bold=True)
        typer.echo(f'  Points: {storage.points_count}')
        typer.echo(f'  Status: {storage.status}')
        typer.echo()

        typer.secho('Content:', bold=True)
        typer.echo(f'  Chunks: {content.total_chunks}')
        typer.echo(f'  Files: {content.unique_files}')
        if content.by_file_type:
            typer.echo(f'  Types: {dict(content.by_file_type)}')
        typer.echo()

        if dashboard_url:
            typer.secho(f'Dashboard: {dashboard_url}', fg=typer.colors.CYAN)


async def _list_async(
    collection_name: str,
    path: str | None,
    file_type: str | None,
    limit: int,
    format: Literal['text', 'json'],
) -> None:
    async with infrastructure() as ctx:
        collection = ctx.registry.get(collection_name)
        if collection is None:
            typer.secho(f"Collection '{collection_name}' not found.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        state_store = IndexStateStore(ctx.redis, collection_name)
        repository = DocumentVectorRepository(ctx.qdrant, collection_name, state_store)

        resolved_path = _resolve_path(path)
        # list_indexed_files uses None for "no filter", not "**"
        filter_path = None if resolved_path == '**' else resolved_path

        files = await repository.list_indexed_files(
            path_prefix=filter_path,
            file_type=file_type,
            limit=limit,
        )

        if format == 'json':
            typer.echo(json.dumps([f.model_dump(mode='json') for f in files], indent=2))
            return

        if not files:
            typer.echo('No indexed files found.')
            return

        for f in files:
            typer.echo(f'{f.chunk_count:4d} chunks  {f.file_type:<10s}  {f.path}')


async def _clear_async(
    collection_name: str,
    path: str | None,
    clear_cache: bool,
    format: Literal['text', 'json'],
) -> None:
    async with infrastructure() as ctx:
        collection = ctx.registry.get(collection_name)
        if collection is None:
            typer.secho(f"Collection '{collection_name}' not found.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        state_store = IndexStateStore(ctx.redis, collection_name)
        repository = DocumentVectorRepository(ctx.qdrant, collection_name, state_store)
        resolved_path = _resolve_path(path)

        if resolved_path == '**':
            if clear_cache:
                total_invalidated = 0
                async for index_key in ctx.redis.scan_iter(match='embed-idx:file:*'):
                    cache_keys = await ctx.redis.smembers(index_key)
                    if cache_keys:
                        await ctx.redis.unlink(*cache_keys, index_key)
                        total_invalidated += len(cache_keys)
                logger.info('Invalidated %d cached embeddings', total_invalidated)

            chunks_count = await repository.count()
            await repository.delete_collection()
            await state_store.clear_collection()
            result = ClearResult(files_removed=0, chunks_removed=chunks_count, path=None)
        else:
            if clear_cache:
                prefix = resolved_path
                if prefix and not prefix.endswith('/'):
                    prefix += '/'
                pattern = f'embed-idx:file:{prefix}*'
                async for index_key in ctx.redis.scan_iter(match=pattern):
                    cache_keys = await ctx.redis.smembers(index_key)
                    if cache_keys:
                        await ctx.redis.unlink(*cache_keys, index_key)

            file_state = await state_store.get_file_state(resolved_path)
            if file_state is not None:
                chunk_ids = list(file_state.chunk_ids)
                if chunk_ids:
                    await repository.delete([UUID(cid) for cid in chunk_ids])
                await state_store.delete_file_state(resolved_path)
                result = ClearResult(files_removed=1, chunks_removed=len(chunk_ids), path=resolved_path)
            else:
                all_chunk_ids = await state_store.get_chunk_ids_under_path(resolved_path)
                if all_chunk_ids:
                    await repository.delete([UUID(cid) for cid in all_chunk_ids])
                files_under = await state_store.get_files_under_path(resolved_path)
                await state_store.delete_files_under_path(resolved_path)
                result = ClearResult(
                    files_removed=len(files_under),
                    chunks_removed=len(all_chunk_ids),
                    path=resolved_path,
                )

        if format == 'json':
            typer.echo(result.model_dump_json(indent=2))
            return

        typer.secho('Cleared:', fg=typer.colors.GREEN)
        typer.echo(f'  Files: {result.files_removed}')
        typer.echo(f'  Chunks: {result.chunks_removed}')


async def _search_async(
    query: str,
    collection_name: str,
    path: str | None,
    limit: int,
    search_type: Literal['hybrid', 'lexical', 'embedding'],
    exclude_paths: Sequence[str] | None,
    min_score: float | None,
    format: Literal['text', 'json'],
) -> None:
    async with infrastructure() as ctx:
        collection = ctx.registry.get(collection_name)
        if collection is None:
            typer.secho(f"Collection '{collection_name}' not found.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        state_store = IndexStateStore(ctx.redis, collection_name)
        repository = DocumentVectorRepository(ctx.qdrant, collection_name, state_store)

        config = default_config(collection.provider)
        embedding_client = create_embedding_client(
            create_config(
                provider=collection.provider,
                embedding_model=collection.model,
                embedding_dimensions=collection.dimensions,
            )
        )

        # fmt: off — lazy imports: ML frameworks add ~1.8s startup
        from document_search.services.embedding import EmbeddingService  # noqa: PLC0415 — lazy
        from document_search.services.reranker import RerankerService  # noqa: PLC0415 — lazy
        from document_search.services.sparse_embedding import SparseEmbeddingService  # noqa: PLC0415 — lazy
        # fmt: on

        embedding_service = EmbeddingService(
            embedding_client,
            batch_size=config.batch_size,
            redis=ctx.redis,
            model=collection.model,
            dimensions=collection.dimensions,
        )
        sparse_service = await SparseEmbeddingService.create()
        await sparse_service.embed_batch(['warmup'])
        reranker = RerankerService()

        try:
            dense_vector: list[float] | None = None
            sparse_indices: list[int] | None = None
            sparse_values: list[float] | None = None

            if search_type in ('hybrid', 'embedding'):
                embed_response = await embedding_service.embed(EmbedRequest(text=query, intent='query'))
                dense_vector = list(embed_response.values)
            if search_type in ('hybrid', 'lexical'):
                np_indices, np_values = await sparse_service.embed(query)
                sparse_indices = np_indices.tolist()
                sparse_values = np_values.tolist()

            resolved_path = _resolve_path(path)
            filter_path = None if resolved_path == '**' else resolved_path
            resolved_excludes = [str(Path(p).expanduser().resolve()) for p in exclude_paths] if exclude_paths else None

            effective_limit = min(max(limit, 1), 100)
            rerank_candidates = min(effective_limit * 3, 200)

            search_query = SearchQuery(
                search_type=search_type,
                dense_vector=dense_vector,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                limit=rerank_candidates,
                source_path_prefixes=[filter_path] if filter_path else None,
                exclude_path_prefixes=resolved_excludes,
            )

            result = await repository.search(search_query)
            result = await reranker.rerank(query=query, result=result, top_k=effective_limit)

            if min_score is not None:
                filtered = [h for h in result.hits if h.score >= min_score]
                result = SearchResult(hits=filtered, total=len(filtered))

            if format == 'json':
                typer.echo(result.model_dump_json(indent=2))
                return

            if not result.hits:
                typer.echo('No results found.')
                return

            typer.secho(f'{result.total} results:', bold=True)
            typer.echo()
            for hit in result.hits:
                typer.secho(f'  {hit.score:.4f}  {hit.source_path}', bold=True)
                if hit.heading_context:
                    typer.echo(f'         {hit.heading_context}')
                snippet = hit.text[:200].replace('\n', ' ')
                typer.echo(f'         {snippet}')
                typer.echo()
        finally:
            sparse_service.shutdown()
            await embedding_client.close()


async def _index_async(
    paths: Sequence[Path],
    collection_name: str,
    gitignore: bool | None,
    stop_after: str | None,
    format: Literal['text', 'json'],
) -> None:
    async with infrastructure() as ctx:
        collection = ctx.registry.get(collection_name)
        if collection is None:
            typer.secho(f"Collection '{collection_name}' not found.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)

        config = default_config(collection.provider)
        embedding_client = create_embedding_client(
            create_config(
                provider=collection.provider,
                embedding_model=collection.model,
                embedding_dimensions=collection.dimensions,
            )
        )

        # fmt: off — lazy imports: ML frameworks add ~1.8s startup
        from document_search.services.chunking import ChunkingService  # noqa: PLC0415 — lazy
        from document_search.services.embedding import EmbeddingService  # noqa: PLC0415 — lazy
        from document_search.services.indexing import IndexingService  # noqa: PLC0415 — lazy
        from document_search.services.sparse_embedding import SparseEmbeddingService  # noqa: PLC0415 — lazy
        # fmt: on

        chunking_service = await ChunkingService.create()
        sparse_service = await SparseEmbeddingService.create()
        await sparse_service.embed_batch(['warmup'])

        embedding_service = EmbeddingService(
            embedding_client,
            batch_size=config.batch_size,
            redis=ctx.redis,
            model=collection.model,
            dimensions=collection.dimensions,
        )

        state_store = IndexStateStore(ctx.redis, collection_name)
        repository = DocumentVectorRepository(ctx.qdrant, collection_name, state_store)
        await repository.ensure_collection(collection.dimensions)

        indexing_service = IndexingService(
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            sparse_embedding_service=sparse_service,
            repository=repository,
            state_store=state_store,
        )

        stop = cast(StopAfterStage, stop_after) if stop_after else None

        # Dashboard lifecycle — ensure dashboard is running and register this CLI process
        from document_search.dashboard.launcher import ensure_dashboard  # noqa: PLC0415 — lazy

        dashboard_state = DashboardStateManager()
        ensure_dashboard()
        dashboard_state.register_process(
            pid=os.getpid(),
            process_type='cli',
            session_id=os.environ.get('CLAUDE_CODE_SESSION_ID'),
        )

        try:
            # Validate all paths upfront
            resolved_paths: list[Path] = []
            for p in paths:
                resolved = p.expanduser().resolve()
                if not resolved.is_file() and not resolved.is_dir():
                    typer.secho(f'Path not found: {resolved}', fg=typer.colors.RED, err=True)
                    continue
                resolved_paths.append(resolved)

            if not resolved_paths:
                typer.secho('No valid paths to index.', fg=typer.colors.RED, err=True)
                raise typer.Exit(1)

            result = await indexing_service.index(
                resolved_paths,
                collection_name=collection_name,
                owner_pid=os.getpid(),
                respect_gitignore=gitignore,
                stop_after=stop,
                embedding_client=embedding_client,
                redis_client=ctx.redis,
            )

            if format == 'json':
                typer.echo(json.dumps(result.model_dump(mode='json'), indent=2, default=str))
                return

            typer.secho(f'Indexing complete ({result.elapsed_seconds:.1f}s):', fg=typer.colors.GREEN)
            typer.echo(f'  Scanned: {result.files_scanned}')
            typer.echo(f'  Indexed: {result.files_indexed}')
            typer.echo(f'  Cached:  {result.files_cached}')
            if result.files_ignored:
                typer.echo(f'  Ignored: {result.files_ignored}')
            if result.files_no_content:
                typer.echo(f'  No content: {result.files_no_content}')
            typer.echo(f'  Chunks:  {result.chunks_created}')
            if result.chunks_skipped:
                typer.echo(f'  Chunks skipped: {result.chunks_skipped}')
            if result.chunks_deleted:
                typer.echo(f'  Chunks deleted: {result.chunks_deleted}')
            typer.echo(f'  Embeddings: {result.embed_cache_hits} cached, {result.embed_cache_misses} computed')
            if result.by_file_type:
                typer.echo('  By type:')
                for ft, summary in result.by_file_type.items():
                    typer.echo(f'    {ft}: {summary}')
            if result.stopped_after:
                typer.echo(f'  Stopped after: {result.stopped_after}')
            if result.errors:
                typer.secho(f'  Errors: {len(result.errors)}', fg=typer.colors.YELLOW)
                for err in result.errors:
                    typer.secho(f'    {err.file_path}: {err.message}', fg=typer.colors.YELLOW)

            dashboard_port = dashboard_state.get_dashboard_port()
            if dashboard_port:
                typer.echo()
                typer.secho(f'Dashboard: http://127.0.0.1:{dashboard_port}', fg=typer.colors.CYAN)
        finally:
            dashboard_state.unregister_process(os.getpid())
            chunking_service.shutdown()
            sparse_service.shutdown()
            await embedding_client.close()


@error_boundary.handler(RuntimeError)
def _handle_infra_error(exc: RuntimeError) -> None:
    typer.secho(f'Error: {exc}', fg=typer.colors.RED, err=True)
