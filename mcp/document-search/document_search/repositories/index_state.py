"""Redis-backed per-file index state.

Each file's indexing state (content hash, chunk IDs, etc.) is stored
as a Redis hash for O(1) lookups and atomic writes.

Key pattern: idx:{collection}:{absolute_path}
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from uuid import UUID

from document_search.clients.redis import RedisClient
from document_search.schemas.indexing import FileIndexState

__all__ = [
    'IndexStateStore',
]

logger = logging.getLogger(__name__)


class IndexStateStore:
    """Per-file index state backed by Redis hashes.

    Key pattern: idx:{collection}:{absolute_path}
    Hash fields: file_hash, file_size, chunk_count, chunk_ids,
                 indexed_at, chunk_strategy_version

    Directory enumeration via SCAN glob handles prefix matching
    that Qdrant keyword indexes cannot do natively.
    """

    def __init__(self, redis: RedisClient, collection_name: str) -> None:
        self._redis = redis
        self._prefix = f'idx:{collection_name}:'

    # --- Per-file operations (O(1)) ---

    async def get_file_state(self, file_path: str) -> FileIndexState | None:
        """Get full state for a single file."""
        raw = await self._redis.hgetall(self._key(file_path))
        if not raw:
            return None
        return _decode_file_state(file_path, raw)

    async def put_file_state(self, file_path: str, state: FileIndexState) -> None:
        """Store state for a single file. Atomic write."""
        await self._redis.hset(self._key(file_path), _encode_file_state(state))

    async def delete_file_state(self, file_path: str) -> None:
        """Remove state for a single file."""
        await self._redis.delete(self._key(file_path))

    # --- Batch operations (pipeline, one round-trip) ---

    async def get_all_states(self, file_paths: Sequence[str]) -> Mapping[str, FileIndexState]:
        """Pre-load states for all file paths in one Redis round-trip.

        Used during classification to build a local dict for synchronous
        change detection. Upsert workers read from this dict, avoiding
        per-file Redis lookups during the pipeline.
        """
        if not file_paths:
            return {}

        pipe = self._redis.pipeline()
        for path in file_paths:
            pipe.hgetall(self._key(path))
        results = await pipe.execute()

        states: dict[str, FileIndexState] = {}
        for path, raw in zip(file_paths, results):
            if raw:
                states[path] = _decode_file_state(path, raw)
        return states

    # --- Directory operations (SCAN-based) ---

    async def get_chunk_ids_under_path(self, path_prefix: str) -> Sequence[str]:
        """Get all chunk IDs for files under a directory prefix.

        Two-phase: SCAN for matching keys, then pipeline HGET chunk_ids.
        """
        clean = path_prefix.rstrip('/')
        pattern = f'{self._prefix}{clean}/*'

        keys = [key async for key in self._redis.scan_iter(match=pattern, count=1000)]
        if not keys:
            return []

        pipe = self._redis.pipeline()
        for key in keys:
            pipe.hget(key, b'chunk_ids')
        results = await pipe.execute()

        all_ids: list[str] = []
        for raw_ids in results:
            if raw_ids:
                all_ids.extend(json.loads(raw_ids))
        return all_ids

    async def get_files_under_path(self, path_prefix: str) -> Sequence[tuple[str, FileIndexState]]:
        """Get all file states under a directory prefix.

        Two-phase: SCAN for matching keys, then pipeline HGETALL.
        """
        clean = path_prefix.rstrip('/')
        pattern = f'{self._prefix}{clean}/*'

        keys = [key async for key in self._redis.scan_iter(match=pattern, count=1000)]
        if not keys:
            return []

        pipe = self._redis.pipeline()
        for key in keys:
            pipe.hgetall(key)
        results = await pipe.execute()

        prefix_len = len(self._prefix)
        entries: list[tuple[str, FileIndexState]] = []
        for key, raw in zip(keys, results):
            if raw:
                file_path = key[prefix_len:]
                entries.append((file_path, _decode_file_state(file_path, raw)))
        return entries

    async def delete_files_under_path(self, path_prefix: str) -> int:
        """Delete all state entries under a directory prefix."""
        clean = path_prefix.rstrip('/')
        pattern = f'{self._prefix}{clean}/*'

        keys = [key async for key in self._redis.scan_iter(match=pattern, count=1000)]
        if not keys:
            return 0

        pipe = self._redis.pipeline()
        for key in keys:
            pipe.delete(key)
        results = await pipe.execute()
        return sum(1 for r in results if r)

    async def clear_collection(self) -> int:
        """Delete all state entries for this collection."""
        pattern = f'{self._prefix}*'

        keys = [key async for key in self._redis.scan_iter(match=pattern, count=1000)]
        if not keys:
            return 0

        pipe = self._redis.pipeline()
        for key in keys:
            pipe.delete(key)
        results = await pipe.execute()
        return sum(1 for r in results if r)

    # --- Private ---

    def _key(self, file_path: str) -> str:
        return f'{self._prefix}{file_path}'


# --- Encode/decode helpers ---


def _encode_file_state(state: FileIndexState) -> Mapping[str, str]:
    """Encode FileIndexState to Redis hash fields (all string values)."""
    return {
        'file_hash': state.file_hash,
        'file_size': str(state.file_size),
        'chunk_count': str(state.chunk_count),
        'chunk_ids': json.dumps([str(uid) for uid in state.chunk_ids]),
        'indexed_at': state.indexed_at.isoformat(),
        'chunk_strategy_version': str(state.chunk_strategy_version),
    }


def _decode_file_state(file_path: str, raw: Mapping[bytes, bytes]) -> FileIndexState:
    """Decode Redis hash bytes to FileIndexState."""
    return FileIndexState(
        file_path=file_path,
        file_hash=raw[b'file_hash'].decode(),
        file_size=int(raw[b'file_size']),
        chunk_count=int(raw[b'chunk_count']),
        chunk_ids=[UUID(uid) for uid in json.loads(raw[b'chunk_ids'])],
        indexed_at=datetime.fromisoformat(raw[b'indexed_at'].decode()),
        chunk_strategy_version=int(raw[b'chunk_strategy_version']),
    )
