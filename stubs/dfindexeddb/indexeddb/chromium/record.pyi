"""Type stubs for dfindexeddb.indexeddb.chromium.record.

Partial stubs covering record types and FolderReader used for IndexedDB parsing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dfindexeddb.indexeddb.chromium.definitions import (
    DatabaseMetaDataKeyType,
    IndexMetaDataKeyType,
    ObjectStoreMetaDataKeyType,
)

class KeyPrefix:
    """Key prefix containing database and object store IDs."""

    database_id: int
    object_store_id: int

    def __getattr__(self, name: str) -> Any: ...

class IDBKey:
    """IndexedDB key with encoded value."""

    value: Any

    def __getattr__(self, name: str) -> Any: ...

class IDBKeyPath:
    """IndexedDB key path (can be string, array, or null)."""

    value: str | list[str] | None

    def __getattr__(self, name: str) -> Any: ...

class ObjectStoreDataValue:
    """Value from object store data record."""

    value: Any

    def __getattr__(self, name: str) -> Any: ...

class DatabaseNameKey:
    """Key for database name record."""

    database_name: str

    def __getattr__(self, name: str) -> Any: ...

class DatabaseMetaDataKey:
    """Key for database metadata record."""

    key_prefix: KeyPrefix
    metadata_type: DatabaseMetaDataKeyType

    def __getattr__(self, name: str) -> Any: ...

class ObjectStoreMetaDataKey:
    """Key for object store metadata record."""

    key_prefix: KeyPrefix
    object_store_id: int
    metadata_type: ObjectStoreMetaDataKeyType

    def __getattr__(self, name: str) -> Any: ...

class IndexMetaDataKey:
    """Key for index metadata record."""

    key_prefix: KeyPrefix
    object_store_id: int
    index_id: int
    metadata_type: IndexMetaDataKeyType

    def __getattr__(self, name: str) -> Any: ...

class ObjectStoreDataKey:
    """Key for object store data record."""

    key_prefix: KeyPrefix
    encoded_user_key: IDBKey

    def __getattr__(self, name: str) -> Any: ...

class IndexedDBRecord:
    """Record from IndexedDB LevelDB."""

    key: DatabaseNameKey | DatabaseMetaDataKey | ObjectStoreMetaDataKey | IndexMetaDataKey | ObjectStoreDataKey
    value: Any

    def __getattr__(self, name: str) -> Any: ...

class FolderReader:
    """Reader for IndexedDB LevelDB folders."""

    def __init__(self, path: Path) -> None: ...
    def GetRecords(self) -> Iterator[IndexedDBRecord]: ...
    def __getattr__(self, name: str) -> Any: ...
