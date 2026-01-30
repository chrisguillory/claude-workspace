"""Collection registry management with file locking.

Manages persistent collection metadata. Shared between MCP server and dashboard.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import filelock

from document_search.paths import COLLECTIONS_LOCK_PATH, COLLECTIONS_STATE_PATH
from document_search.schemas.collections import Collection, CollectionRegistry
from document_search.schemas.config import EmbeddingProvider

__all__ = [
    'CollectionRegistryManager',
]


class CollectionRegistryManager:
    """Manages collection registry with file locking.

    Thread-safe persistence of collection metadata.
    """

    def __init__(
        self,
        state_path: Path = COLLECTIONS_STATE_PATH,
        lock_path: Path = COLLECTIONS_LOCK_PATH,
    ) -> None:
        self._state_path = state_path
        self._lock_path = lock_path
        self._lock = filelock.FileLock(lock_path)

    def load(self) -> CollectionRegistry:
        """Load registry from file. Returns empty registry if not exists."""
        if not self._state_path.exists():
            return CollectionRegistry()
        data = json.loads(self._state_path.read_text())
        return CollectionRegistry.model_validate(data)

    def save(self, registry: CollectionRegistry) -> None:
        """Save registry atomically with lock."""
        with self._lock:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._state_path.with_suffix('.tmp')
            temp_path.write_text(json.dumps(registry.model_dump(mode='json'), indent=2, default=str) + '\n')
            temp_path.rename(self._state_path)

    def create_collection(
        self,
        name: str,
        provider: EmbeddingProvider,
        description: str | None = None,
    ) -> Collection:
        """Create a new collection. Raises ValueError if name exists."""
        with self._lock:
            registry = self.load()

            if name in registry.collections:
                raise ValueError(f"Collection '{name}' already exists")

            collection = Collection(
                name=name,
                provider=provider,
                created_at=datetime.now(UTC),
                description=description,
            )

            new_collections = dict(registry.collections)
            new_collections[name] = collection
            new_registry = CollectionRegistry(collections=new_collections)
            self._save_unlocked(new_registry)

            return collection

    def get(self, name: str) -> Collection | None:
        """Get collection by name. Returns None if not found."""
        registry = self.load()
        return registry.collections.get(name)

    def list_collections(self) -> Sequence[Collection]:
        """List all collections."""
        registry = self.load()
        return list(registry.collections.values())

    def delete_collection(self, name: str) -> bool:
        """Delete collection from registry. Returns True if deleted, False if not found."""
        with self._lock:
            registry = self.load()

            if name not in registry.collections:
                return False

            new_collections = {k: v for k, v in registry.collections.items() if k != name}
            new_registry = CollectionRegistry(collections=new_collections)
            self._save_unlocked(new_registry)

            return True

    def _save_unlocked(self, registry: CollectionRegistry) -> None:
        """Save without acquiring lock. Caller must hold lock."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._state_path.with_suffix('.tmp')
        temp_path.write_text(json.dumps(registry.model_dump(mode='json'), indent=2, default=str) + '\n')
        temp_path.rename(self._state_path)
