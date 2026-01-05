"""Type stubs for dfindexeddb.indexeddb.chromium.definitions.

Partial stubs covering metadata key type enums used for IndexedDB parsing.
"""

from __future__ import annotations

from enum import Enum

class DatabaseMetaDataKeyType(Enum):
    """Enum for database metadata types."""

    IDB_INTEGER_VERSION = ...

class ObjectStoreMetaDataKeyType(Enum):
    """Enum for object store metadata types."""

    OBJECT_STORE_NAME = ...
    KEY_PATH = ...
    AUTO_INCREMENT_FLAG = ...

class IndexMetaDataKeyType(Enum):
    """Enum for index metadata types."""

    INDEX_NAME = ...
    KEY_PATH = ...
    UNIQUE_FLAG = ...
    MULTI_ENTRY_FLAG = ...
